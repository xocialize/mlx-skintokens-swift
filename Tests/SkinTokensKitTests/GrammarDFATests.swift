// GrammarDFATests.swift — S2b skeleton-DFA parity, validated against oracle goldens.
//
// Pure INT logic (no MLXArray, never evals) → safe in the SPM test target (swift-port-parity.md
// gate-lane discipline). Goldens are dumped straight from the oracle TokenizerPart + captured
// generation streams by scratchpad/dump_grammar_goldens.py:
//   • nextPossible  — set equality vs TokenizerPart.next_posible_token over a state-covering battery
//   • bones         — integer equality vs TokenizerPart.bones_in_sequence (incl. branch + eos)
//   • replays       — every actual next token of a REAL captured stream (giraffe/humanoid
//                     output_ids) must be grammar-allowed by the ported VocabSwitchingLogitsProcessor
//
// This is FRONT 2's parity gate: it proves the pole-risk DFA before any MLX/beam is wired.

import Foundation
import XCTest
@testable import SkinTokensCore

final class GrammarDFATests: XCTestCase {

    // MARK: - Goldens loading (source-tree relative; no SPM resource bundle needed)

    struct Goldens: Decodable {
        struct LayoutG: Decodable {
            let numDiscrete, bos, eos, branch, pad, spring, clsNone: Int
            let parts, cls: [Int]
            let tokenizerVocabSize, fsqCodebook, lmVocabSize, lmEos: Int
        }
        struct NP: Decodable { let prefix: [Int]; let allowed: [Int] }
        struct BN: Decodable { let seq: [Int]; let J: Int }
        struct RP: Decodable { let name: String; let output_ids: [Int]; let len: Int; let hasEOS: Bool }
        let layout: LayoutG
        let tokensPerSkin: Int
        let nextPossible: [NP]
        let bones: [BN]
        let replays: [RP]
    }

    static let goldens: Goldens = {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/grammar_goldens.json")
        let data = try! Data(contentsOf: url)
        return try! JSONDecoder().decode(Goldens.self, from: data)
    }()

    let L = SkinTokenGrammar.skinTokens

    // MARK: - Layout: the ported offset arithmetic reproduces the oracle token ids

    func testLayoutMatchesOracle() {
        let g = Self.goldens.layout
        XCTAssertEqual(L.numDiscrete, g.numDiscrete)
        XCTAssertEqual(L.branch, g.branch)
        XCTAssertEqual(L.bos, g.bos)
        XCTAssertEqual(L.eos, g.eos)
        XCTAssertEqual(L.pad, g.pad)
        XCTAssertEqual(L.spring, g.spring)
        XCTAssertEqual(L.clsNone, g.clsNone)
        XCTAssertEqual(L.partsTokens, g.parts)
        XCTAssertEqual(L.clsTokens, g.cls)
        XCTAssertEqual(L.tokenizerVocabSize, g.tokenizerVocabSize)   // 267
        XCTAssertEqual(L.fsqCodebook, g.fsqCodebook)                 // 32768
        XCTAssertEqual(L.lmVocabSize, g.lmVocabSize)                 // 33036
        XCTAssertEqual(L.lmEos, g.lmEos)                             // 33035
        XCTAssertEqual(L.switchTokenId, g.eos)                       // 258
        XCTAssertEqual(Self.goldens.tokensPerSkin, 4)
    }

    // MARK: - next_posible_token: exact set equality across every DFA state

    func testNextPossibleTokensMatchOracle() throws {
        for (i, np) in Self.goldens.nextPossible.enumerated() {
            let got = try SkinTokenGrammar.nextPossibleTokens(prefix: np.prefix, layout: L)
            XCTAssertEqual(Set(got), Set(np.allowed),
                           "next_posible_token mismatch for battery #\(i) prefix=\(np.prefix): " +
                           "missing=\(Set(np.allowed).subtracting(got).sorted().prefix(8)) " +
                           "extra=\(Set(got).subtracting(np.allowed).sorted().prefix(8))")
            // ascending + de-duplicated (mask semantics)
            XCTAssertEqual(got, got.sorted())
            XCTAssertEqual(got.count, Set(got).count, "duplicate ids in allowed set #\(i)")
        }
    }

    func testEmptyPrefixIsBos() throws {
        XCTAssertEqual(try SkinTokenGrammar.nextPossibleTokens(prefix: [], layout: L), [L.bos])
    }

    func testMalformedPrefixThrows() {
        XCTAssertThrowsError(try SkinTokenGrammar.nextPossibleTokens(prefix: [999], layout: L))
        XCTAssertThrowsError(try SkinTokenGrammar.bonesInSequence([999], layout: L))
    }

    // MARK: - bones_in_sequence: integer equality (incl. branch-parent + eos-terminated)

    func testBonesInSequenceMatchOracle() throws {
        for (i, bn) in Self.goldens.bones.enumerated() {
            let got = try SkinTokenGrammar.bonesInSequence(bn.seq, layout: L)
            XCTAssertEqual(got, bn.J, "bones_in_sequence mismatch for battery #\(i) seq=\(bn.seq)")
        }
    }

    // MARK: - Replay: every actual next token of a REAL stream is grammar-allowed

    func testReplayCapturedStreamsAllAllowed() throws {
        for rp in Self.goldens.replays {
            let ids = rp.output_ids
            XCTAssertGreaterThan(ids.count, 1, "\(rp.name): empty stream")
            for k in 1..<ids.count {
                let prefix = Array(ids[0..<k])
                let d = try SkinTokenGrammar.decision(sequence: prefix,
                                                      tokensPerSkin: Self.goldens.tokensPerSkin, layout: L)
                XCTAssertTrue(SkinTokenGrammar.allows(ids[k], d, layout: L),
                              "\(rp.name): token #\(k)=\(ids[k]) NOT allowed by decision \(d)")
            }
        }
    }

    /// The two decode invariants: the stream ends by forcing the LM-EOS, and the skeleton/skin phase
    /// boundary is exactly the tokenizer EOS (switch token).
    func testReplayPhaseBoundaryAndForcedEos() throws {
        for rp in Self.goldens.replays {
            let ids = rp.output_ids
            XCTAssertEqual(ids.last, L.lmEos, "\(rp.name): stream must end with LM-EOS")
            // decision at the penultimate prefix must FORCE eos (J×tokensPerSkin skin tokens emitted)
            let penult = Array(ids[0..<(ids.count - 1)])
            let d = try SkinTokenGrammar.decision(sequence: penult,
                                                  tokensPerSkin: Self.goldens.tokensPerSkin, layout: L)
            XCTAssertEqual(d, .forceEos, "\(rp.name): final step must be forceEos")

            // exactly one tokenizer-EOS (skeleton terminator), and it precedes the skin tokens
            let switchCount = ids.filter { $0 == L.switchTokenId }.count
            XCTAssertEqual(switchCount, 1, "\(rp.name): expected exactly one switch/tokenizer-EOS")
            let whereSwitch = ids.firstIndex(of: L.switchTokenId)!
            // skeleton phase: all tokens up to (not incl.) switch are coord/special (< tokenizerVocab)
            for t in ids[0..<whereSwitch] {
                XCTAssertLessThan(t, L.tokenizerVocabSize, "\(rp.name): skeleton token \(t) out of tokenizer vocab")
            }
            // Skin phase: between the switch and the final LM-EOS sit J×tokensPerSkin − 1 real FSQ
            // tokens. ⚠ Oracle quirk (tokenrig.decode): the slice `ids[w+1 : w+1+J×tps]` is J×tps
            // wide but its LAST element is the FORCED LM-EOS, because force-eos fires when
            // `len(prefix) − where == J×tps` (i.e. after J×tps−1 skin tokens). So only J×tps−1 real
            // FSQ tokens are emitted; the last skin "group"'s 4th token is the LM-EOS (→ FSQ index
            // fsqCodebook, off the nominal codebook). S1d/S2b Swift decode must replicate this.
            let J = try SkinTokenGrammar.bonesInSequence(ids, layout: L)
            let skin = Array(ids[(whereSwitch + 1)..<(ids.count - 1)])
            XCTAssertEqual(skin.count, J * Self.goldens.tokensPerSkin - 1,
                           "\(rp.name): real FSQ token count != J×tps−1")
            for t in skin {
                XCTAssertGreaterThanOrEqual(t, L.tokenizerVocabSize, "\(rp.name): skin token \(t) below FSQ base")
                XCTAssertLessThan(t, L.lmEos, "\(rp.name): skin token \(t) at/above LM-EOS")
            }
        }
    }
}
