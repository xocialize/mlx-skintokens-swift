import Foundation

/// S2b — the grammar that constrains SkinTokens' autoregressive decode. **Pure integer logic, no
/// MLX, never evals** → lives in `SkinTokensCore` and is exercised in `swift test` (metallib-safe).
///
/// This is a faithful port of the oracle's two pieces (VAST-AI/SkinTokens, CPU oracle at
/// `DEV/SkinTokensDev/skintokens-oracle`):
///   • `TokenizerPart.next_posible_token` / `.bones_in_sequence`  (src/tokenizer/tokenizer_part.py)
///   • `VocabSwitchingLogitsProcessor.__call__`                    (src/model/tokenrig.py:40)
///
/// The decode has two phases over one AR sequence (mesh soft-prefix → `inputs_embeds`, so `init` =
/// the start tokens re-supplied for grammar tracking, exactly as the oracle prepends `self.init`):
///
///   Skeleton phase (before the switch token = tokenizer EOS appears): a 7-state DFA masks logits to
///   `next_posible_token` — bos, then cls/part specials, then joint-coord TRIPLES (3 discrete <256
///   per joint XYZ) with optional `branch`/`part` tokens between joints, terminated by EOS.
///
///   Skin phase (after the switch token appears): force the skin vocab (ids ≥ switch token); when
///   exactly J×tokensPerSkin skin tokens have been emitted (J = `bonesInSequence`), force the LM EOS.
///
/// ⚠ beam-1 is NOT a fallback — P0 proved it emits degenerate skeletons (giraffe seed0 depth-62
/// chain). Batched beam(10) feeds this per-beam/per-step mask ahead of the top-k. This file is the
/// int substrate for that; the MLX mask application + beam land at S2b proper.
public enum SkinTokenGrammar {

    // MARK: - Token layout (mirrors TokenizerPart.parse offset arithmetic)

    /// The discrete token id layout of a SkinTokens tokenizer. Ids below `numDiscrete` are joint
    /// coordinate bins; everything at/above is a special. The LM stacks the FSQ skin codebook and a
    /// single LM-EOS on top of the tokenizer vocab: `lmVocabSize = tokenizerVocabSize + fsqCodebook + 1`.
    public struct Layout: Equatable, Sendable {
        public let numDiscrete: Int       // 256 coordinate bins  [0, numDiscrete)
        public let branch: Int            // token_id_branch
        public let bos: Int               // token_id_bos
        public let eos: Int               // token_id_eos (== the phase switch token)
        public let pad: Int               // token_id_pad
        public let spring: Int            // token_id_spring (the "None"/spring part)
        public let clsNone: Int           // token_id_cls_none
        public let partsTokens: [Int]     // part ids, ascending (e.g. body, hand)
        public let clsTokens: [Int]       // cls ids, ascending (e.g. rignet, vroid, articulation)
        public let tokenizerVocabSize: Int
        public let fsqCodebook: Int       // FSQ codebook size (skin vocab), e.g. 8^5 = 32768

        /// Total LM vocab (tokenizer + FSQ skin + 1 LM-EOS).
        public var lmVocabSize: Int { tokenizerVocabSize + fsqCodebook + 1 }
        /// The single LM end-of-sequence id (== lmVocabSize - 1), distinct from the tokenizer EOS.
        public var lmEos: Int { lmVocabSize - 1 }
        /// The phase-switch marker = the tokenizer EOS (skeleton terminator), per `get_logits_processor`.
        public var switchTokenId: Int { eos }

        /// Build the layout via the exact offset arithmetic of `TokenizerPart.parse`. Part/cls ids are
        /// assigned by their config integer value (0,1,2,…) added to a running offset, so passing the
        /// counts reproduces the ids: parts = base+[0..<partsCount], cls = base+[0..<clsCount].
        public init(numDiscrete: Int = 256, partsCount: Int = 2, clsCount: Int = 3, fsqCodebook: Int = 32768) {
            self.numDiscrete = numDiscrete
            var offset = numDiscrete
            self.branch = offset + 0
            self.bos    = offset + 1
            self.eos    = offset + 2
            self.pad    = offset + 3
            offset += 4
            self.spring = offset + 0
            offset += 1
            let partsBase = offset
            self.partsTokens = (0..<partsCount).map { partsBase + $0 }
            offset += partsCount
            self.clsNone = offset + 0
            offset += 1
            let clsBase = offset
            self.clsTokens = (0..<clsCount).map { clsBase + $0 }
            offset += clsCount
            self.tokenizerVocabSize = offset
            self.fsqCodebook = fsqCodebook
        }
    }

    /// The shipped SkinTokens layout: num_discrete 256, parts {body,hand}, cls {rignet,vroid,articulation},
    /// FSQ [8,8,8,8,8]=32768. Yields tokenizer vocab 267, LM vocab 33036, LM-EOS 33035, switch token 258.
    public static let skinTokens = Layout()

    // MARK: - DFA state (7 states; `expectCls` is present in the oracle build map but unreachable)

    public enum State: Equatable, Hashable, Sendable {
        case expectBos
        case expectClsOrPartOrJoint
        case expectPartOrJoint
        case expectJoint2
        case expectJoint3
        case expectBranchOrPartOrJoint
        case expectJoint
    }

    public enum GrammarError: Swift.Error, CustomStringConvertible {
        case notBos(Int)
        public var description: String {
            switch self {
            case .notBos(let id): return "skeleton stream does not start with bos (got \(id))"
            }
        }
    }

    /// One DFA transition (mirrors the `if state == …` chain shared by both oracle functions).
    @inline(__always)
    static func advance(_ state: State, _ id: Int, _ L: Layout) throws -> State {
        switch state {
        case .expectBos:
            guard id == L.bos else { throw GrammarError.notBos(id) }
            return .expectClsOrPartOrJoint
        case .expectClsOrPartOrJoint:
            if id < L.numDiscrete { return .expectJoint2 }
            else if id == L.clsNone || L.clsTokens.contains(id) { return .expectPartOrJoint }
            else { return .expectJoint }                 // a part
        case .expectPartOrJoint:
            if id < L.numDiscrete { return .expectJoint2 } else { return .expectPartOrJoint }
        case .expectJoint2:
            return .expectJoint3
        case .expectJoint3:
            return .expectBranchOrPartOrJoint
        case .expectBranchOrPartOrJoint:
            if id == L.branch { return .expectJoint }
            else if id < L.numDiscrete { return .expectJoint2 }
            else { return .expectJoint }                 // a part
        case .expectJoint:
            return .expectJoint2
        }
    }

    // MARK: - next_posible_token (skeleton-phase allowed set)

    /// The set of allowed next tokens for a skeleton-phase prefix — port of
    /// `TokenizerPart.next_posible_token`. Returns ascending, de-duplicated ids (mask semantics).
    public static func nextPossibleTokens(prefix ids: [Int], layout L: Layout = skinTokens) throws -> [Int] {
        if ids.isEmpty { return [L.bos] }
        return allowedTokens(state: try stateAfter(ids, layout: L), layout: L)
    }

    /// The DFA state reached after consuming `ids` (skeleton phase). Throws if not bos-prefixed.
    public static func stateAfter(_ ids: [Int], layout L: Layout = skinTokens) throws -> State {
        var state: State = .expectBos
        for id in ids { state = try advance(state, id, L) }
        return state
    }

    /// Allowed next tokens for a given DFA state (ascending, deduped) — the state-keyed core of
    /// `next_posible_token`, factored out so a per-state logit mask can be built + cached once.
    public static func allowedTokens(state: State, layout L: Layout = skinTokens) -> [Int] {
        var s: [Int] = []
        func addCls()   { s.append(L.clsNone); s.append(contentsOf: L.clsTokens) }
        func addPart()  { s.append(L.spring);  s.append(contentsOf: L.partsTokens) }
        func addJoint() { s.append(contentsOf: 0..<L.numDiscrete) }
        func addBranch(){ s.append(L.branch) }
        func addEos()   { s.append(L.eos) }
        func addBos()   { s.append(L.bos) }
        switch state {
        case .expectBos:                    addBos()
        case .expectClsOrPartOrJoint:       addCls(); addPart(); addJoint()
        case .expectPartOrJoint:            addPart(); addJoint(); addEos()
        case .expectJoint2:                 addJoint()
        case .expectJoint3:                 addJoint()
        case .expectBranchOrPartOrJoint:    addJoint(); addPart(); addBranch(); addEos()
        case .expectJoint:                  addJoint()
        }
        return s.sorted()
    }

    /// The mask class for a running AR sequence — the phase decision reduced to a **cacheable key**
    /// so a beam decoder builds each distinct logit mask once (skeleton masks key on DFA state; the
    /// skin-window and force-EOS masks are singletons). Same decision surface as `decision(sequence:)`.
    public enum MaskClass: Equatable, Hashable, Sendable {
        case skeleton(State)
        case skinContinue
        case forceEos
    }

    public static func maskClass(sequence ids: [Int], tokensPerSkin: Int, layout L: Layout = skinTokens) throws -> MaskClass {
        if let whereSwitch = ids.firstIndex(of: L.switchTokenId) {
            let j = try bonesInSequence(ids, layout: L)
            if (ids.count - whereSwitch) == j * tokensPerSkin { return .forceEos }
            return .skinContinue
        }
        return .skeleton(try stateAfter(ids, layout: L))
    }

    // MARK: - bones_in_sequence (joint count over a full sequence)

    /// Number of joints (bones) implied by a sequence — port of `TokenizerPart.bones_in_sequence`.
    /// Counts a joint at the close of each coordinate triple unless it was the branch (parent) triple,
    /// and stops at the first tokenizer EOS.
    public static func bonesInSequence(_ ids: [Int], layout L: Layout = skinTokens) throws -> Int {
        var s = 0
        var isBranch = false
        var state: State = .expectBos
        for id in ids {
            switch state {
            case .expectBos:
                guard id == L.bos else { throw GrammarError.notBos(id) }
                state = .expectClsOrPartOrJoint
            case .expectClsOrPartOrJoint:
                if id < L.numDiscrete { state = .expectJoint2 }
                else if id == L.clsNone || L.clsTokens.contains(id) { state = .expectPartOrJoint }
                else { state = .expectJoint }
            case .expectPartOrJoint:
                if id < L.numDiscrete { state = .expectJoint2 } else { state = .expectPartOrJoint }
            case .expectJoint2:
                state = .expectJoint3
            case .expectJoint3:
                if !isBranch { s += 1 }
                isBranch = false
                state = .expectBranchOrPartOrJoint
            case .expectBranchOrPartOrJoint:
                if id == L.branch { state = .expectJoint; isBranch = true }
                else if id < L.numDiscrete { state = .expectJoint2 }
                else { state = .expectJoint }
            case .expectJoint:
                state = .expectJoint2
            }
            if id == L.eos { break }
        }
        return s
    }

    // MARK: - VocabSwitchingLogitsProcessor (per-sequence decision)

    /// The masking decision for one AR sequence — port of `VocabSwitchingLogitsProcessor.__call__`.
    public enum Decision: Equatable, Sendable {
        /// Skeleton phase: only these ids are allowed (ascending); everything else → -inf.
        case skeleton([Int])
        /// Skin phase, incomplete: allow the skin vocab window [switchTokenId, lmVocabSize) EXCEPT the
        /// LM-EOS (oracle: `mask[switch:] = 0; mask[eos] = -inf`).
        case skinContinue
        /// Skin phase, exactly J×tokensPerSkin skin tokens emitted: force the LM-EOS only.
        case forceEos
    }

    /// `sequence` = the full running AR sequence the processor sees (init ⊕ generated-so-far).
    public static func decision(sequence ids: [Int], tokensPerSkin: Int, layout L: Layout = skinTokens) throws -> Decision {
        if let whereSwitch = ids.firstIndex(of: L.switchTokenId) {
            let j = try bonesInSequence(ids, layout: L)
            if (ids.count - whereSwitch) == j * tokensPerSkin { return .forceEos }
            return .skinContinue
        }
        return .skeleton(try nextPossibleTokens(prefix: ids, layout: L))
    }

    /// Whether `id` would be an allowed next token under a decision — the membership test a
    /// per-beam/per-step logit mask enforces. Avoids materializing the 32k-wide skin window.
    public static func allows(_ id: Int, _ d: Decision, layout L: Layout = skinTokens) -> Bool {
        switch d {
        case .skeleton(let set): return set.contains(id)
        case .skinContinue:      return id >= L.switchTokenId && id < L.lmVocabSize && id != L.lmEos
        case .forceEos:          return id == L.lmEos
        }
    }
}
