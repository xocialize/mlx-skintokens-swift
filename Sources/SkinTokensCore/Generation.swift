import Foundation
import MLX
import MLXRandom

/// S2b(3) — the grammar-constrained batched-KV beam decoder (the P1 pole). Runs the TokenRig Qwen3
/// backbone on `inputs_embeds` (512-token mesh soft prefix ⊕ start tokens), applying
/// `SkinTokenGrammar`'s VocabSwitchingLogitsProcessor as a **per-beam, per-step logit mask BEFORE the
/// beam selection**: skeleton phase = DFA allowed-set mask; skin phase = force the skin window
/// (ids ≥ switch, minus LM-EOS); force LM-EOS at exactly J×tokensPerSkin skin tokens.
///
/// ⚠ Validation is NOT parity (P0 proved CPU generation is non-reproducible under a fixed seed).
/// Correctness is guaranteed by CONSTRUCTION: the DFA mask makes every emitted stream grammar-valid,
/// and the emitted skin tokens decode through S1d to a rowsum≈1 skin. **Deterministic beam search
/// (do_sample=false) is the default** — it maximizes cumulative masked log-prob under the grammar,
/// which reproducibly yields the compact healthy skeleton P0 attributed to beam(10) (beam1 went
/// degenerate: giraffe seed0 depth-62 chain). HF beam-sample (do_sample=true) is available via the
/// Gumbel-top-k trick for callers that want the ckpt's stochastic `generate_kwargs`.
public enum SkinGen {

    public struct Params: Sendable {
        public var numBeams = 10
        public var doSample = false          // deterministic beam by default (reproducible gate)
        public var topK = 0                  // 0 = disabled (grammar already bounds support)
        public var temperature: Float = 1.5
        public var repetitionPenalty: Float = 2.0
        public var maxNewTokens = 2040
        public var lengthPenalty: Float = 1.0
        public var seed: UInt64 = 0
        public init() {}
    }

    public struct Result: Sendable {
        public var outputIds: [Int]          // start_tokens ⊕ generated (incl. the final LM-EOS)
        public var generated: [Int]          // generated only (incl. the final LM-EOS)
        public var startTokens: [Int]
        public var score: Float              // length-penalized log-prob of the winning hypothesis
        public var violations: Int           // grammar violations on the emitted stream (must be 0)
    }

    // MARK: - Mask cache (one MLXArray per distinct MaskClass; built once, reused every step)

    private final class MaskCache {
        let L: SkinTokenGrammar.Layout
        let V: Int
        var cache: [SkinTokenGrammar.MaskClass: MLXArray] = [:]
        init(_ L: SkinTokenGrammar.Layout) { self.L = L; self.V = L.lmVocabSize }

        func mask(_ cls: SkinTokenGrammar.MaskClass) -> MLXArray {
            if let m = cache[cls] { return m }
            var v = [Float](repeating: -1e30, count: V)
            switch cls {
            case .skeleton(let state):
                for id in SkinTokenGrammar.allowedTokens(state: state, layout: L) { v[id] = 0 }
            case .skinContinue:
                for id in L.switchTokenId..<V { v[id] = 0 }
                v[L.lmEos] = -1e30
            case .forceEos:
                v[L.lmEos] = 0
            }
            let m = MLXArray(v)
            cache[cls] = m
            return m
        }
    }

    // MARK: - Completed-hypothesis pool (mirrors HF BeamHypotheses, length_penalty=1)

    private struct HypPool {
        let numBeams: Int
        let lengthPenalty: Float
        var hyps: [(score: Float, seq: [Int])] = []   // finalScore-sorted desc
        var worst: Float = .infinity

        mutating func add(seq: [Int], sumLogprob: Float) {
            let score = sumLogprob / pow(Float(seq.count), lengthPenalty)
            if hyps.count < numBeams || score > worst {
                hyps.append((score, seq))
                hyps.sort { $0.score > $1.score }
                if hyps.count > numBeams { hyps.removeLast() }
                worst = hyps.last!.score
            }
        }

        /// HF early_stopping=False: done once full AND no active beam can beat the worst completed.
        func isDone(bestSumLogprob: Float, curLen: Int) -> Bool {
            guard hyps.count >= numBeams else { return false }
            let best = bestSumLogprob / pow(Float(curLen), lengthPenalty)
            return worst >= best
        }
    }

    // MARK: - Beam decode

    /// Generate a full token stream from an injected mesh soft-prefix. `meshCond` is `[1, P, hidden]`
    /// (the 512-token learned_mesh_cond); `startTokens` is the grammar `init` (e.g. `[bos, cls]`).
    /// `progress` is called once per generation step (every emitted token). It may THROW — a caller
    /// passes `try Task.checkCancellation()` to get cooperative per-token cancellation (the CAN
    /// cadence); the thrown error unwinds `generate` unchanged (it `rethrows`).
    public static func generate(decoder: Qwen3Decoder, meshCond: MLXArray, startTokens: [Int],
                                layout L: SkinTokenGrammar.Layout = SkinTokenGrammar.skinTokens, tokensPerSkin: Int = 4,
                                params: Params = Params(),
                                progress: ((_ step: Int, _ activeBeams: Int, _ completed: Int) throws -> Void)? = nil) rethrows -> Result {
        let V = L.lmVocabSize
        let H = meshCond.shape[2]
        let masks = MaskCache(L)
        let embTable = Qwen3.embedTable(decoder.w, prefix: decoder.prefix)  // [vocab, hidden]

        // --- Prefill ONCE at B=1 (all beams start identical), then tile the KV cache to B beams. The
        // prefill attention is O(T²); doing it at B=1 instead of B=beams cuts that transient ~beams×
        // (the dominant activation-memory driver for a long skeleton prefix). ------------------------
        var B = params.numBeams
        let startArr = MLXArray(startTokens.map { Int32($0) })
        let startEmb = embTable[startArr].reshaped([1, startTokens.count, H])     // [1,S,H]
        let prompt1 = concatenated([meshCond.asType(.float32), startEmb], axis: 1) // [1,P+S,H]
        let last1 = decoder.prefill(prompt1)[0..., (prompt1.shape[1] - 1)..., 0...]  // [1,1,V]
        decoder.tileCaches(to: B)                                                   // KV [1,…] → [B,…]
        var lastLogits = tiled(last1, repetitions: [B, 1, 1])                        // [B,1,V]
        eval(lastLogits)

        // Beam state (host): per-beam generated ids + cumulative masked log-prob.
        var genSeqs: [[Int]] = Array(repeating: [], count: B)
        var beamScores = [Float](repeating: -1e30, count: B); beamScores[0] = 0   // expand only beam0 first
        var pool = HypPool(numBeams: params.numBeams, lengthPenalty: params.lengthPenalty)

        var rngKey = MLXRandom.key(params.seed)

        for stepCount in 0..<params.maxNewTokens {
            // log_softmax over the raw logits → [B,V]
            let lg = lastLogits[0..., 0, 0...]                                    // [B,V]
            var lp = lg - lg.logSumExp(axis: -1, keepDims: true)

            // repetition penalty over each beam's generated tokens (HF: once per unique token).
            if params.repetitionPenalty != 1 {
                var presentFlat: [Int32] = []
                for b in 0..<B {
                    if genSeqs[b].isEmpty { continue }
                    for t in Set(genSeqs[b]) { presentFlat.append(Int32(b * V + t)) }
                }
                if !presentFlat.isEmpty {
                    var flat = lp.reshaped([B * V])
                    let idxA = MLXArray(presentFlat)
                    flat[idxA] = flat[idxA] * params.repetitionPenalty   // lp<0 → more negative
                    lp = flat.reshaped([B, V])
                }
            }

            // per-beam grammar mask (skeleton DFA / skin window / force-EOS), stacked → [B,V].
            var maskRows: [MLXArray] = []
            maskRows.reserveCapacity(B)
            var violated = false
            for b in 0..<B {
                let seq = startTokens + genSeqs[b]
                let cls = (try? SkinTokenGrammar.maskClass(sequence: seq, tokensPerSkin: tokensPerSkin, layout: L))
                    ?? .skeleton(.expectBos)
                if cls == .skeleton(.expectBos) && !seq.isEmpty { violated = true }
                maskRows.append(masks.mask(cls))
            }
            _ = violated
            let maskB = stacked(maskRows, axis: 0)                                // [B,V]
            let beamCol = MLXArray(beamScores).reshaped([B, 1])
            let cand = lp + maskB + beamCol                                       // [B,V] cumulative

            // selection scores: deterministic = cand; sampling = tempered + Gumbel (top-k w/o replacement)
            var sel = cand
            if params.doSample {
                let (k1, k2) = MLXRandom.split(key: rngKey); rngKey = k2
                let g = MLXRandom.gumbel([B, V], key: k1)
                sel = cand / params.temperature + g
            }
            let K = min(2 * params.numBeams, B * V)
            let order = argSort(-sel.reshaped([B * V]))                           // largest sel first
            let topIdx = order[0..<K]
            let candFlat = cand.reshaped([B * V])
            let topScores = candFlat[topIdx]
            eval(topIdx, topScores)
            let hostIdx = topIdx.asArray(Int32.self)
            let hostScore = topScores.asArray(Float.self)

            // beam scorer: EOS candidates in the top-numBeams complete; non-EOS fill the next beams.
            var nextTokens: [Int] = [], nextSrc: [Int] = [], nextScores: [Float] = []
            for rank in 0..<K {
                let flatIdx = Int(hostIdx[rank]); let score = hostScore[rank]
                if score <= -1e29 { continue }
                let beamIdx = flatIdx / V, token = flatIdx % V
                if token == L.lmEos {
                    if rank < params.numBeams {
                        pool.add(seq: genSeqs[beamIdx] + [token], sumLogprob: score)
                    }
                } else {
                    nextTokens.append(token); nextSrc.append(beamIdx); nextScores.append(score)
                }
                if nextTokens.count == params.numBeams { break }
            }

            let curLen = stepCount + 1
            if pool.isDone(bestSumLogprob: hostScore.first ?? -1e30, curLen: curLen) { break }
            if nextTokens.isEmpty { break }

            // graft: reorder KV caches + host beam state onto the selected source beams.
            B = nextTokens.count
            decoder.reorderBeams(MLXArray(nextSrc.map { Int32($0) }))
            genSeqs = zip(nextSrc, nextTokens).map { genSeqs[$0] + [$1] }
            beamScores = nextScores

            let tokArr = MLXArray(nextTokens.map { Int32($0) })
            let stepEmb = embTable[tokArr].reshaped([B, 1, H])                    // [B,1,H]
            lastLogits = decoder.step(stepEmb)
            eval(lastLogits)
            MLX.Memory.clearCache()
            try progress?(stepCount, B, pool.hyps.count)   // per-token CAN checkpoint (rethrows)
        }

        // finalize: prefer a COMPLETED hypothesis (one that emitted the grammar-forced LM-EOS). Only
        // fold active beams as a fallback when nothing completed (e.g. maxNewTokens cut it off) — an
        // active beam is one step short of its forced EOS, and because length-normalization divides by
        // a length one smaller it would spuriously outscore every completed beam (missing skin tokens).
        if pool.hyps.isEmpty {
            for b in 0..<B { pool.add(seq: genSeqs[b], sumLogprob: beamScores[b]) }
        }
        let best = pool.hyps.first ?? (score: 0, seq: [])
        let outputIds = startTokens + best.seq
        let violations = countViolations(outputIds, tokensPerSkin: tokensPerSkin, layout: L)
        return Result(outputIds: outputIds, generated: best.seq, startTokens: startTokens,
                      score: best.score, violations: violations)
    }

    /// Per-step membership replay of a full stream (start ⊕ generated) — counts grammar violations.
    /// Zero by construction from a masked decode; a nonzero count means the mask leaked.
    public static func countViolations(_ ids: [Int], tokensPerSkin: Int,
                                       layout L: SkinTokenGrammar.Layout = SkinTokenGrammar.skinTokens) -> Int {
        // `ids` = startTokens ⊕ generated. The grammar `init` (startTokens) is fixed; each GENERATED
        // token must be allowed given the running sequence up to that point. We can't know startTokens'
        // length here, so validate the whole stream as if init were empty: the first token must be bos.
        var violations = 0
        for i in 0..<ids.count {
            let prefix = Array(ids[0..<i])
            guard let d = try? SkinTokenGrammar.decision(sequence: prefix, tokensPerSkin: tokensPerSkin, layout: L) else {
                violations += 1; continue
            }
            if !SkinTokenGrammar.allows(ids[i], d, layout: L) { violations += 1 }
        }
        return violations
    }
}
