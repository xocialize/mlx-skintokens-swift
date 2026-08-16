import Foundation
import MLX

/// S1b — Michelangelo / 3DShape2VecSet point encoder (`mesh_encoder.encoder.*`) → `learned_mesh_cond`
/// (1, 512, 896). Ported dict-functional (weights via `[String: MLXArray]`, trellis2 component-gate
/// methodology) so the attention math is gated in isolation from checkpoint-key plumbing; the S0 key
/// contract already proved the keys match.
///
/// Config (converted/tokenrig.hparams.json `mesh_encoder`, `__target__ = michelangelo_encoder` =
/// `ShapeAsLatentPerceiverEncoder`, `no_query=True`, `query_method=False` → the FPS path):
///   num_latents 256 · width 512 · heads 8 · 8 encoder layers · embed_dim 64 (UNUSED for this
///   encoder-only variant — no pre_kl) · point_feats 3 · num_freqs 8 · include_pi **false** ·
///   qkv_bias **false** · use_ln_post **true** · token_num 512 · use_full_input true · flash true.
///
/// Forward (`CrossAttentionEncoder._forward`, non-query path):
///   1. KV  = input_proj( cat[ fourier(pc_full 54000), feats ] )        (full input, use_full_input)
///   2. query points = 512 FPS-selected points (rng.choice(N,2048)+fps ratio¼). ⚠ RNG is numpy
///      PCG64 — the parity gate INJECTS the captured query indices; PCG64+FPS reproduction is a
///      separable task needed only for real e2e (see PORTING-SPEC S1b note).
///   3. Q  = input_proj( cat[ fourier(pc_query 512), feats_query ] )
///   4. latents = self_attn( cross_attn(Q, KV) )   → 512 tokens
///   5. ln_post
///   6. output_proj: Linear(512→896) + RMSNorm(896)                     → (1, 512, 896)
///
/// Numerics pinned from the oracle:
///   • FourierEmbedder: logspace, include_pi=false → freqs = 2^[0..7]; embed = cat[x, sin(x⊗f), cos(x⊗f)]
///     (dim 3·(2·8+1)=51); channel order [x0f0..x0f7, x1.., x2..] per sin/cos block.
///   • attn scale = 1/√(head_dim=64) (miche applies 1/√√64 to q AND k → same product scale); softmax fp32.
///   • MLP: c_fc (×4) → GELU **erf** (`nn.GELU()` default) → c_proj. LayerNorm eps 1e-5.
///   • qkv_bias=false: c_qkv / c_q / c_kv carry NO bias; c_proj / mlp / input_proj DO (default True).
public enum Michelangelo {

    /// FourierEmbedder(num_freqs=8, include_pi=false, logspace=true, include_input=true, input_dim=3).
    /// x: [..., 3] → [..., 51].
    /// Computed in fp32 (buffers are fp32 upstream; bf16 sin/cos drifts at high freq). Returns fp32.
    public static func fourier(_ x: MLXArray, numFreqs: Int = 8, includePi: Bool = false) -> MLXArray {
        let xf = x.asType(.float32)
        var f = [Float](repeating: 0, count: numFreqs)
        for i in 0..<numFreqs { f[i] = Float(pow(2.0, Double(i))) }     // logspace: 2^i
        if includePi { for i in 0..<numFreqs { f[i] *= Float.pi } }
        let freqs = MLXArray(f)                                        // fp32
        let s = xf.shape                                              // [..., 3]
        let xp = xf.expandedDimensions(axis: -1) * freqs             // [..., 3, numFreqs]
        let flat = xp.reshaped(Array(s.dropLast()) + [s.last! * numFreqs])  // [..., 3*numFreqs]
        return concatenated([xf, SkinOps.sinR(flat), SkinOps.cosR(flat)], axis: -1)  // [..., 3+2*3*numFreqs] fp32
    }

    /// Build input tokens: input_proj( cat[ fourier(pc), feats ] ). pc/feats: [B, N, 3] → [B, N, 512].
    static func inputTokens(_ pc: MLXArray, _ feats: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let emb = fourier(pc)                                          // [B, N, 51] fp32
        let data = concatenated([emb, feats.asType(.float32)], axis: -1).asType(.bfloat16)  // [B, N, 54] bf16
        return SkinOps.linear(data, w["\(p).input_proj.weight"]!, w["\(p).input_proj.bias"]!)
    }

    /// ResidualCrossAttentionBlock: x = x + attn(ln_1(x), ln_2(data)); x = x + mlp(ln_3(x)).
    static func crossAttn(_ x: MLXArray, _ data: MLXArray, heads: Int, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let h = SkinOps.layerNorm(x, w["\(p).ln_1.weight"]!, w["\(p).ln_1.bias"]!)
        let d = SkinOps.layerNorm(data, w["\(p).ln_2.weight"]!, w["\(p).ln_2.bias"]!)
        let q = SkinOps.linear(h, w["\(p).attn.c_q.weight"]!)          // no bias
        let kv = SkinOps.linear(d, w["\(p).attn.c_kv.weight"]!)        // no bias, → 2*width
        let width = q.shape.last!
        let dh = width / heads
        let qh = SkinOps.splitHeads(q, heads: heads)                  // [B,H,Tq,dh]
        let kvh = SkinOps.splitHeads(kv, heads: heads)               // [B,H,Tk,2*dh]
        let parts = kvh.split(parts: 2, axis: -1)
        let out = SkinOps.sdpa(qh, parts[0], parts[1], scale: 1.0 / Float(dh).squareRoot())
        let merged = SkinOps.mergeHeads(out)                          // [B,Tq,width]
        let proj = SkinOps.linear(merged, w["\(p).attn.c_proj.weight"]!, w["\(p).attn.c_proj.bias"]!)
        var y = x + proj
        let m = SkinOps.layerNorm(y, w["\(p).ln_3.weight"]!, w["\(p).ln_3.bias"]!)
        let mlp = mlpForward(m, w, "\(p).mlp")
        y = y + mlp
        return y
    }

    /// ResidualAttentionBlock (self): x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)). c_qkv → 3*width.
    static func selfAttn(_ x: MLXArray, heads: Int, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let h = SkinOps.layerNorm(x, w["\(p).ln_1.weight"]!, w["\(p).ln_1.bias"]!)
        let qkv = SkinOps.linear(h, w["\(p).attn.c_qkv.weight"]!)     // no bias, → 3*width
        let width = x.shape.last!
        let dh = width / heads
        let qkvh = SkinOps.splitHeads(qkv, heads: heads)             // [B,H,T,3*dh]
        let parts = qkvh.split(parts: 3, axis: -1)
        let out = SkinOps.sdpa(parts[0], parts[1], parts[2], scale: 1.0 / Float(dh).squareRoot())
        let merged = SkinOps.mergeHeads(out)
        let proj = SkinOps.linear(merged, w["\(p).attn.c_proj.weight"]!, w["\(p).attn.c_proj.bias"]!)
        var y = x + proj
        let m = SkinOps.layerNorm(y, w["\(p).ln_2.weight"]!, w["\(p).ln_2.bias"]!)
        y = y + mlpForward(m, w, "\(p).mlp")
        return y
    }

    /// MLP: c_proj( gelu_erf( c_fc(x) ) ). Both linears carry bias.
    static func mlpForward(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let h = SkinOps.linear(x, w["\(p).c_fc.weight"]!, w["\(p).c_fc.bias"]!)
        let a = SkinOps.geluErf(h)
        return SkinOps.linear(a, w["\(p).c_proj.weight"]!, w["\(p).c_proj.bias"]!)
    }

    /// The full encoder forward → latents (1, 512, 512), given the KV point cloud (full input) and
    /// the pre-selected 512 query points/feats. `prefix` = "mesh_encoder.encoder".
    public static func encode(kvPc: MLXArray, kvFeats: MLXArray,
                              queryPc: MLXArray, queryFeats: MLXArray,
                              heads: Int = 8, numResBlocks: Int = 8,
                              weights w: [String: MLXArray],
                              prefix p: String = "mesh_encoder.encoder") -> MLXArray {
        let kv = inputTokens(kvPc, kvFeats, w, p)                     // [1, 54000, 512]
        let q = inputTokens(queryPc, queryFeats, w, p)               // [1, 512, 512]
        var latents = crossAttn(q, kv, heads: heads, w, "\(p).cross_attn")
        for i in 0..<numResBlocks {
            latents = selfAttn(latents, heads: heads, w, "\(p).self_attn.resblocks.\(i)")
        }
        latents = SkinOps.layerNorm(latents, w["\(p).ln_post.weight"]!, w["\(p).ln_post.bias"]!)
        return latents
    }

    /// output_proj: Linear(512→896) + RMSNorm(896). Applied by tokenrig after the encoder to make the
    /// mesh soft prefix. `learned_mesh_cond = output_proj(encode(...))` → (1, 512, 896).
    public static func outputProj(_ latents: MLXArray, weights w: [String: MLXArray],
                                  prefix p: String = "output_proj", rmsEps: Float = 1e-6) -> MLXArray {
        let lin = SkinOps.linear(latents, w["\(p).0.weight"]!, w["\(p).0.bias"]!)
        return SkinOps.rmsNorm(lin, w["\(p).1.weight"]!, eps: rmsEps)
    }
}
