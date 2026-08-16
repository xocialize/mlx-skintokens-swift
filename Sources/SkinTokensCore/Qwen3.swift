import Foundation
import MLX
import MLXNN

/// S2 — TokenRig transformer forward (`transformer.*`). A stock Qwen3-0.6B backbone with an UN-TIED
/// rig-vocab head (33036), run on `inputs_embeds` (512-token mesh soft prefix + token embeds).
///
/// Decision — LIFT vs TRANSLATE: for the SHIPPING package (S7), lift the upstream `mlx-swift-lm`
/// `Qwen3.swift` (strip `transformer.` prefix, untied head, add an inputs-embeds seam). For the S2
/// PARITY GATE this compact fp32 dict-functional translation is cleaner: it needs no upstream edit
/// and avoids `MLXFast` SDPA's CPU/Metal phantom-drift (explicit attention on the CPU stream). Config
/// from tokenrig.hparams + headers: hidden 896, 28 layers, n_heads 16, kv_heads 8, head_dim 128,
/// intermediate 3072, rms_eps 1e-6, rope_theta 1e6, rotate_half RoPE, per-head QK RMSNorm (before RoPE).
public enum Qwen3 {

    public struct Config {
        public var hidden = 896, layers = 28, heads = 16, kvHeads = 8, headDim = 128, intermediate = 3072
        public var vocab = 33036
        public var ropeTheta: Double = 1_000_000, rmsEps: Float = 1e-6
        /// Dtype the incremental decoder STORES the KV cache in (compute stays fp32 — upcast on read).
        /// `.float32` = parity-exact (the `--s2b-kv-check` gate). `.bfloat16` halves the persistent beam
        /// cache (~2.75 GB → ~1.4 GB) for generation — the biggest activation-memory lever; the tiny
        /// bf16-rounding is within the already-non-reproducible generation envelope.
        public var kvDtype: DType = .float32
        public init() {}
    }

    /// rotate_half RoPE (NeoX). Returns (cos, sin) as [T, headDim], fp32, built in Double.
    static func ropeTables(_ T: Int, headDim: Int, theta: Double, offset: Int = 0) -> (MLXArray, MLXArray) {
        let half = headDim / 2
        var cosT = [Float](repeating: 0, count: T * headDim)
        var sinT = [Float](repeating: 0, count: T * headDim)
        for p in 0..<T {
            let pos = Double(p + offset)
            for i in 0..<half {
                let inv = pow(theta, -Double(2 * i) / Double(headDim))
                let a = pos * inv
                let c = Float(cos(a)), s = Float(sin(a))
                cosT[p * headDim + i] = c;        cosT[p * headDim + half + i] = c
                sinT[p * headDim + i] = s;        sinT[p * headDim + half + i] = s
            }
        }
        return (MLXArray(cosT, [T, headDim]), MLXArray(sinT, [T, headDim]))
    }

    /// rotate_half(x) = cat[-x[half:], x[:half]]. x: [..., headDim].
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.shape.last!
        let x1 = x[.ellipsis, 0..<(d / 2)]
        let x2 = x[.ellipsis, (d / 2)...]
        return concatenated([-x2, x1], axis: -1)
    }

    static func applyRope(_ x: MLXArray, _ cosT: MLXArray, _ sinT: MLXArray) -> MLXArray {
        // x: [B,H,T,Dh]; cos/sin: [T,Dh] → broadcast over B,H.
        let c = cosT.reshaped([1, 1, cosT.shape[0], cosT.shape[1]])
        let s = sinT.reshaped([1, 1, sinT.shape[0], sinT.shape[1]])
        return x.asType(.float32) * c + rotateHalf(x.asType(.float32)) * s
    }

    /// Full forward on a precomputed embedding sequence `h` [1, T, hidden]. Causal, no cache.
    /// Returns logits [1, T, vocab]. Computed fp32 (28-layer bf16 accumulation drifts a tight gate).
    public static func forward(inputsEmbeds h0: MLXArray, weights w: [String: MLXArray],
                               config c: Config = Config(), prefix p: String = "transformer",
                               returnHidden: Bool = false) -> MLXArray {
        var h = h0.asType(.float32)
        let T = h.shape[1]
        let (cosT, sinT) = ropeTables(T, headDim: c.headDim, theta: c.ropeTheta)
        // causal mask [T,T]: 0 on/below diag, -inf above.
        let idx = MLXArray(Array(0..<Int32(T)))
        let mask = (idx.reshaped([T, 1]) .< idx.reshaped([1, T])) * Float(-1e30)   // upper-tri → -inf
        let mask4 = mask.reshaped([1, 1, T, T])
        let scale = 1.0 / Float(c.headDim).squareRoot()

        for i in 0..<c.layers {
            let lp = "\(p).model.layers.\(i)"
            let hn = SkinOps.rmsNorm(h, w["\(lp).input_layernorm.weight"]!, eps: c.rmsEps)
            var q = SkinOps.linear(hn, fp(w["\(lp).self_attn.q_proj.weight"]!))   // [1,T,2048]
            var k = SkinOps.linear(hn, fp(w["\(lp).self_attn.k_proj.weight"]!))   // [1,T,1024]
            var v = SkinOps.linear(hn, fp(w["\(lp).self_attn.v_proj.weight"]!))
            q = q.reshaped([1, T, c.heads, c.headDim])
            k = k.reshaped([1, T, c.kvHeads, c.headDim])
            v = v.reshaped([1, T, c.kvHeads, c.headDim])
            // per-head QK RMSNorm over head_dim, BEFORE RoPE
            q = SkinOps.rmsNorm(q, w["\(lp).self_attn.q_norm.weight"]!, eps: c.rmsEps)
            k = SkinOps.rmsNorm(k, w["\(lp).self_attn.k_norm.weight"]!, eps: c.rmsEps)
            q = q.swappedAxes(1, 2)   // [1,heads,T,Dh]
            k = k.swappedAxes(1, 2)   // [1,kv,T,Dh]
            v = v.swappedAxes(1, 2)
            q = applyRope(q, cosT, sinT)
            k = applyRope(k, cosT, sinT)
            // GQA: repeat_interleave kv heads by heads/kvHeads
            let rep = c.heads / c.kvHeads
            k = repeatKV(k, rep: rep)   // [1,heads,T,Dh]
            v = repeatKV(v, rep: rep)
            var att = matmul(q, k.swappedAxes(-1, -2)) * scale + mask4   // [1,heads,T,T]
            att = softmax(att, axis: -1)
            let o = matmul(att, v)                                       // [1,heads,T,Dh]
            let merged = SkinOps.mergeHeads(o)                           // [1,T,heads*Dh=2048]
            let attnOut = SkinOps.linear(merged, fp(w["\(lp).self_attn.o_proj.weight"]!))  // [1,T,896]
            h = h + attnOut
            let hn2 = SkinOps.rmsNorm(h, w["\(lp).post_attention_layernorm.weight"]!, eps: c.rmsEps)
            let gate = SkinOps.linear(hn2, fp(w["\(lp).mlp.gate_proj.weight"]!))
            let up = SkinOps.linear(hn2, fp(w["\(lp).mlp.up_proj.weight"]!))
            let mlp = SkinOps.linear(silu(gate) * up, fp(w["\(lp).mlp.down_proj.weight"]!))
            h = h + mlp
        }
        if returnHidden { return h }
        h = SkinOps.rmsNorm(h, w["\(p).model.norm.weight"]!, eps: c.rmsEps)
        return SkinOps.linear(h, fp(w["\(p).lm_head.weight"]!))          // [1,T,vocab]
    }

    static func fp(_ x: MLXArray) -> MLXArray { x.asType(.float32) }

    /// repeat_interleave kv heads by `rep` along the head axis: [1,kv,T,Dh] → [1,kv*rep,T,Dh].
    /// `repeated(count:axis:)` == np.repeat (interleaved: h0,h0,h1,h1,…) = HF `repeat_kv`.
    static func repeatKV(_ x: MLXArray, rep: Int) -> MLXArray {
        rep == 1 ? x : repeated(x, count: rep, axis: 1)
    }

    /// Embed token ids via `transformer.model.embed_tokens.weight` [vocab, hidden] → [1, L, hidden].
    public static func embedTokens(_ ids: [Int32], weights w: [String: MLXArray], prefix p: String = "transformer") -> MLXArray {
        let emb = w["\(p).model.embed_tokens.weight"]!.asType(.float32)   // [vocab,hidden]
        let idxA = MLXArray(ids)
        return emb[idxA].expandedDimensions(axis: 0)                      // [1,L,hidden]
    }

    /// The raw embedding table [vocab, hidden] (fp32) — for a decoder that embeds one row per step.
    public static func embedTable(_ w: [String: MLXArray], prefix p: String = "transformer") -> MLXArray {
        w["\(p).model.embed_tokens.weight"]!.asType(.float32)
    }
}

// MARK: - Incremental KV-cached decode (S2b beam/greedy generation)

/// A per-layer key/value cache. Keys/values are `[B, kvHeads, T, Dh]`; the batch axis is the beam
/// axis, so beam reordering is a gather along axis 0. `update` appends the new step's k/v and returns
/// the full cached tensors to attend over.
public final class Qwen3KVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public private(set) var length: Int = 0
    public init() {}

    /// Append `[B,kvHeads,Tq,Dh]` and return the full `[B,kvHeads,length,Dh]`.
    func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if let ek = keys, let ev = values {
            keys = concatenated([ek, k], axis: 2)
            values = concatenated([ev, v], axis: 2)
        } else {
            keys = k; values = v
        }
        length = keys!.shape[2]
        return (keys!, values!)
    }

    /// Reorder the beam (batch) axis by `idx` (an int MLXArray of source-beam indices).
    func reorder(_ idx: MLXArray) {
        if keys != nil { keys = keys![idx]; values = values![idx] }
    }

    /// Tile a B=1 cache to `B` identical beams along the batch axis (after a single-beam prefill).
    func tile(to B: Int) {
        if let k = keys, let v = values {
            keys = tiled(k, repetitions: [B, 1, 1, 1]); values = tiled(v, repetitions: [B, 1, 1, 1])
        }
    }
}

/// Incremental, batched (beam-axis) Qwen3 decoder over `inputs_embeds`. Same math as `Qwen3.forward`
/// (fp32, explicit attention — no `MLXFast` SDPA, so it stays on whatever stream is current), but with
/// a per-layer KV cache so each generation step is O(T) not O(T²). The batch axis IS the beam axis:
/// `reorderBeams` gathers every layer's cache so beam search can graft candidate hypotheses onto their
/// source beams. Position ids are absolute over the whole sequence (the 512 mesh soft-prefix tokens
/// occupy positions 0..511, exactly as HF `inputs_embeds` generation assigns them).
public final class Qwen3Decoder {
    let w: [String: MLXArray]
    let c: Qwen3.Config
    let prefix: String
    var caches: [Qwen3KVCache]
    /// Absolute position of the NEXT token to be processed (= number of tokens already cached).
    public private(set) var offset: Int = 0

    public init(weights: [String: MLXArray], config: Qwen3.Config = .init(), prefix: String = "transformer") {
        self.w = weights; self.c = config; self.prefix = prefix
        self.caches = (0..<config.layers).map { _ in Qwen3KVCache() }
    }

    /// Reorder every layer's KV cache along the beam axis (beam-search graft).
    public func reorderBeams(_ idx: MLXArray) { for cache in caches { cache.reorder(idx) } }

    /// Tile every layer's B=1 cache to `B` identical beams — call after a single-beam prefill so the
    /// expensive T² prefill attention runs ONCE (B=1), not per beam, then the cache fans out.
    public func tileCaches(to B: Int) { for cache in caches { cache.tile(to: B) } }

    /// Run one chunk of embeddings `[B, Tq, hidden]` through all layers at the current offset,
    /// updating the caches. Returns the post-block hidden `[B, Tq, hidden]` (pre final-norm). Advances
    /// `offset` by `Tq`. When `Tq > 1` a causal mask is applied over the (offset+Tq)-wide cache.
    func runLayers(_ h0: MLXArray) -> MLXArray {
        let B = h0.shape[0], Tq = h0.shape[1]
        var h = h0.asType(.float32)
        let (cosT, sinT) = Qwen3.ropeTables(Tq, headDim: c.headDim, theta: c.ropeTheta, offset: offset)
        let scale = 1.0 / Float(c.headDim).squareRoot()
        let Tcache = offset + Tq
        var mask4: MLXArray? = nil
        if Tq > 1 {
            // query i has absolute position offset+i; it may attend key positions 0..offset+i.
            let qpos = MLXArray(Array(0..<Int32(Tq))).reshaped([Tq, 1]) + Int32(offset)
            let kpos = MLXArray(Array(0..<Int32(Tcache))).reshaped([1, Tcache])
            mask4 = ((kpos .> qpos) * Float(-1e30)).reshaped([1, 1, Tq, Tcache])
        }
        let rep = c.heads / c.kvHeads
        for i in 0..<c.layers {
            let lp = "\(prefix).model.layers.\(i)"
            let hn = SkinOps.rmsNorm(h, w["\(lp).input_layernorm.weight"]!, eps: c.rmsEps)
            var q = SkinOps.linear(hn, Qwen3.fp(w["\(lp).self_attn.q_proj.weight"]!))
            var k = SkinOps.linear(hn, Qwen3.fp(w["\(lp).self_attn.k_proj.weight"]!))
            var v = SkinOps.linear(hn, Qwen3.fp(w["\(lp).self_attn.v_proj.weight"]!))
            q = q.reshaped([B, Tq, c.heads, c.headDim])
            k = k.reshaped([B, Tq, c.kvHeads, c.headDim])
            v = v.reshaped([B, Tq, c.kvHeads, c.headDim])
            q = SkinOps.rmsNorm(q, w["\(lp).self_attn.q_norm.weight"]!, eps: c.rmsEps)
            k = SkinOps.rmsNorm(k, w["\(lp).self_attn.k_norm.weight"]!, eps: c.rmsEps)
            q = q.swappedAxes(1, 2)   // [B,heads,Tq,Dh]
            k = k.swappedAxes(1, 2)   // [B,kv,Tq,Dh]
            v = v.swappedAxes(1, 2)
            q = Qwen3.applyRope(q, cosT, sinT)
            k = Qwen3.applyRope(k, cosT, sinT)
            let (fullK, fullV) = caches[i].update(k.asType(c.kvDtype), v.asType(c.kvDtype))  // stored kvDtype
            let kk = Qwen3.repeatKV(fullK, rep: rep).asType(.float32)  // upcast for fp32 compute
            let vv = Qwen3.repeatKV(fullV, rep: rep).asType(.float32)
            var att = matmul(q, kk.swappedAxes(-1, -2)) * scale  // [B,heads,Tq,Tcache]
            if let m4 = mask4 { att = att + m4 }
            att = softmax(att, axis: -1)
            let o = matmul(att, vv)                       // [B,heads,Tq,Dh]
            let merged = SkinOps.mergeHeads(o)            // [B,Tq,heads*Dh]
            let attnOut = SkinOps.linear(merged, Qwen3.fp(w["\(lp).self_attn.o_proj.weight"]!))
            h = h + attnOut
            let hn2 = SkinOps.rmsNorm(h, w["\(lp).post_attention_layernorm.weight"]!, eps: c.rmsEps)
            let gate = SkinOps.linear(hn2, Qwen3.fp(w["\(lp).mlp.gate_proj.weight"]!))
            let up = SkinOps.linear(hn2, Qwen3.fp(w["\(lp).mlp.up_proj.weight"]!))
            let mlp = SkinOps.linear(silu(gate) * up, Qwen3.fp(w["\(lp).mlp.down_proj.weight"]!))
            h = h + mlp
        }
        offset += Tq
        return h
    }

    /// Final-norm + un-tied head → logits `[B, Tq, vocab]`.
    func logits(_ h: MLXArray) -> MLXArray {
        let hn = SkinOps.rmsNorm(h, w["\(prefix).model.norm.weight"]!, eps: c.rmsEps)
        return SkinOps.linear(hn, Qwen3.fp(w["\(prefix).lm_head.weight"]!))
    }

    /// Prefill the prompt embeddings `[B, Tprompt, hidden]`; returns logits at every prompt position.
    public func prefill(_ embeds: MLXArray) -> MLXArray { logits(runLayers(embeds)) }

    /// One incremental step from `[B, 1, hidden]`; returns logits `[B, 1, vocab]`.
    public func step(_ embeds: MLXArray) -> MLXArray { logits(runLayers(embeds)) }
}
