import Foundation
import MLX

/// S1c/S1d — SkinVAE (`vae.model.*` embedded in tokenrig). Diffusers-style `Tripo2` encoder/decoder
/// (NOT the miche path — `is_miche=false`). Ported dict-functional; the S0 key contract already
/// pinned the keys. Config (skinvae.hparams): width 768, heads 12 (head_dim 64), cond_channels 3,
/// latent 512, embed_frequency 8, **embed_include_pi = true**, use_pmpe = true. FSQ [8,8,8,8,8]=32768.
///
/// ⚠ Corrections vs first-glance assumptions (subagent audit):
///   • FF is PLAIN erf-GELU (`activation_fn="gelu"`): `net.0.proj` 768→3072 (NOT GEGLU 768→6144), `net.2` 3072→768.
///   • attn scale = 1/√64 = 0.125 (diffusers/SDPA default), no qk-norm, no RoPE.
///   • QKV "cat→view(heads)→split" interleave (Tripo2AttnProcessor2_0): self splits [Q|K|V] per head;
///     cross splits only [K|V] per head, query standard. Reproduced by splitHeads(concat)+split.
///   • decoder output is SIGMOID-activated (skin weights ∈[0,1]) despite being called "logits".
///   • FSQ index 32768 wraps → index 0 (mod 8^5) — no clamp/error.
public enum SkinVAE {

    public static let heads = 12
    public static let scale: Float = 1.0 / 8.0                 // 1/√(head_dim 64)

    // MARK: - pmpe Fourier embedder (FrequencyPositionalEmbedding, include_pi=true, use_pmpe=true)

    /// x: [..., 3] → [..., 51].  out = cat[ x, sin(embed)+sin(phase), cos(embed)+cos(phase) ].
    ///   embed[c*8+i]  = x[c] * (2^i · π)
    ///   phase[c*8+i]  = x[c] · π · 0.5 + phaseTab[i],   phaseTab[i] = (8^(1-(i+1)/8) + (i+1)/8) · 2π
    /// ⚠ Computed in **fp32** — the oracle's `frequencies`/`phase` are fp32 buffers and `include_pi`
    /// pushes arguments to 128π≈402 where bf16 sin/cos is garbage (bf16 embed → cos 0.996 fail). The
    /// bf16 cast happens at the consuming Linear (autocast), not before sin/cos.
    /// `bf16Arg=true` mirrors the DECODER path: `_decode` casts positions to z.dtype (bf16) BEFORE
    /// the embedder, so the argument `pos·freq` is bf16-rounded and the whole embed is bf16 — the
    /// trained proj_query expects exactly that. cond_encoder (S1c) instead embeds fp32 (bf16Arg=false).
    public static func embed(_ x: MLXArray, numFreqs: Int = 8, bf16Arg: Bool = false) -> MLXArray {
        let xf = bf16Arg ? x.asType(.bfloat16).asType(.float32) : x.asType(.float32)
        var fr = [Float](repeating: 0, count: numFreqs)
        var ph = [Float](repeating: 0, count: numFreqs)
        for i in 0..<numFreqs {
            fr[i] = Float(pow(2.0, Double(i)) * Double.pi)
            let n = Double(numFreqs)
            ph[i] = Float((pow(n, 1.0 - Double(i + 1) / n) + Double(i + 1) / n) * 2.0 * Double.pi)
        }
        // ⚠ the oracle's frequencies/phase buffers are bf16 (model `.to(bfloat16)`), so the table
        // VALUES are bf16-rounded (128π: 402.12→402) while the sin/cos runs fp32 (fp32×bf16 promotes).
        // Round the tables to bf16 precision, then compute fp32. (S1b's freqs are exact powers of 2.)
        let freqs = MLXArray(fr).asType(.bfloat16).asType(.float32)
        let phase = MLXArray(ph).asType(.bfloat16).asType(.float32)
        let s = xf.shape
        let base = xf.expandedDimensions(axis: -1)                    // [...,3,1]
        let rnd: (MLXArray) -> MLXArray = bf16Arg ? { $0.asType(.bfloat16).asType(.float32) } : { $0 }
        let embed = rnd((base * freqs)).reshaped(Array(s.dropLast()) + [s.last! * numFreqs])   // [...,24]
        let phaseT = rnd((base * Float.pi * 0.5 + phase)).reshaped(Array(s.dropLast()) + [s.last! * numFreqs])
        let res = concatenated([SkinOps.sinR(embed) + SkinOps.sinR(phaseT),
                                SkinOps.cosR(embed) + SkinOps.cosR(phaseT)], axis: -1)  // [...,48]
        let out = concatenated([xf, res], axis: -1)                  // [...,51]
        return bf16Arg ? out.asType(.bfloat16) : out
    }

    /// cat[ embed(positions), features ] → [B, N, 54], cast bf16 for the proj_in Linear (autocast).
    public static func qkvInput(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let pos = xf[0..., 0..., 0..<3]
        let feats = xf[0..., 0..., 3...]
        return concatenated([embed(pos), feats], axis: -1).asType(.bfloat16)
    }

    // MARK: - diffusers Tripo2 attention (Tripo2AttnProcessor2_0)

    /// Self-attention: qkv = concat[to_q,to_k,to_v] → splitHeads(12) → split(3). to_out.0 (+bias).
    static func attnSelf(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let q = SkinOps.linear(x, w["\(p).to_q.weight"]!)
        let k = SkinOps.linear(x, w["\(p).to_k.weight"]!)
        let v = SkinOps.linear(x, w["\(p).to_v.weight"]!)
        let qkv = concatenated([q, k, v], axis: -1)                  // [B,N,3*768]
        let heads3 = SkinOps.splitHeads(qkv, heads: heads)          // [B,12,N,192]
        let parts = heads3.split(parts: 3, axis: -1)
        let out = SkinOps.sdpa(parts[0], parts[1], parts[2], scale: scale)
        let merged = SkinOps.mergeHeads(out)
        return SkinOps.linear(merged, w["\(p).to_out.0.weight"]!, w["\(p).to_out.0.bias"]!)
    }

    /// Cross-attention: query standard head-split; KV = concat[to_k,to_v] → splitHeads(12) → split(2).
    /// norm_cross(enc) is applied to the KV source BEFORE to_k/to_v (diffusers cross_attention_norm).
    static func attnCross(_ x: MLXArray, _ enc: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let encN = SkinOps.layerNorm(enc, w["\(p).norm_cross.weight"]!, w["\(p).norm_cross.bias"]!)
        let q = SkinOps.linear(x, w["\(p).to_q.weight"]!)            // [B,Nq,768]
        let k = SkinOps.linear(encN, w["\(p).to_k.weight"]!)
        let v = SkinOps.linear(encN, w["\(p).to_v.weight"]!)
        let qh = SkinOps.splitHeads(q, heads: heads)                // standard [B,12,Nq,64]
        let kv = concatenated([k, v], axis: -1)                     // [B,Nkv,1536]
        let kvh = SkinOps.splitHeads(kv, heads: heads)              // [B,12,Nkv,128]
        let parts = kvh.split(parts: 2, axis: -1)
        let out = SkinOps.sdpa(qh, parts[0], parts[1], scale: scale)
        let merged = SkinOps.mergeHeads(out)
        return SkinOps.linear(merged, w["\(p).to_out.0.weight"]!, w["\(p).to_out.0.bias"]!)
    }

    /// FeedForward: plain erf-GELU. net.0.proj (768→3072, bias) → gelu → net.2 (3072→768, bias).
    static func ff(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        let h = SkinOps.linear(x, w["\(p).net.0.proj.weight"]!, w["\(p).net.0.proj.bias"]!)
        let a = SkinOps.geluErf(h)
        return SkinOps.linear(a, w["\(p).net.2.weight"]!, w["\(p).net.2.bias"]!)
    }

    /// Self DiTBlock: h += attn1(norm1(h)); h += ff(norm3(h)).
    public static func selfBlock(_ x: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        var h = x + attnSelf(SkinOps.layerNorm(x, w["\(p).norm1.weight"]!, w["\(p).norm1.bias"]!), w, "\(p).attn1")
        h = h + ff(SkinOps.layerNorm(h, w["\(p).norm3.weight"]!, w["\(p).norm3.bias"]!), w, "\(p).ff")
        return h
    }

    /// Cross DiTBlock: h += attn2(norm2(h), enc); h += ff(norm3(h)).
    public static func crossBlock(_ x: MLXArray, _ enc: MLXArray, _ w: [String: MLXArray], _ p: String) -> MLXArray {
        var h = x + attnCross(SkinOps.layerNorm(x, w["\(p).norm2.weight"]!, w["\(p).norm2.bias"]!), enc, w, "\(p).attn2")
        h = h + ff(SkinOps.layerNorm(h, w["\(p).norm3.weight"]!, w["\(p).norm3.bias"]!), w, "\(p).ff")
        return h
    }

    // MARK: - S1c: cond_encoder → cond_latents (1, 384, 512)

    /// `_encode` cond path. condQ = qkvInput(384 FPS-sampled cond pts); condKv = qkvInput(full 54000).
    /// cond_encoder: proj_in (shared) → [cross@0, self, self] → norm_out → cond_quant(768→512).
    public static func condEncode(condQ: MLXArray, condKv: MLXArray,
                                  weights w: [String: MLXArray], prefix p: String = "vae.model") -> MLXArray {
        let ce = "\(p).cond_encoder"
        var hs = SkinOps.linear(condQ, w["\(ce).proj_in.weight"]!, w["\(ce).proj_in.bias"]!)   // [1,384,768]
        let enc = SkinOps.linear(condKv, w["\(ce).proj_in.weight"]!, w["\(ce).proj_in.bias"]!) // [1,54000,768]
        hs = crossBlock(hs, enc, w, "\(ce).blocks.0")
        hs = selfBlock(hs, w, "\(ce).blocks.1")
        hs = selfBlock(hs, w, "\(ce).blocks.2")
        hs = SkinOps.layerNorm(hs, w["\(ce).norm_out.weight"]!, w["\(ce).norm_out.bias"]!)
        return SkinOps.linear(hs, w["\(p).cond_quant.weight"]!, w["\(p).cond_quant.bias"]!)      // [1,384,512]
    }

    // MARK: - S1d: FSQ lookup + chunked decoder

    /// FSQ.indices_to_codes: digit extraction (mod 8, basis [1,8,64,512,4096]) → (digit-4)/4 → project_out(5→512).
    /// Wraps mod 8^5 (index 32768 → 0). `indices` int32 [M] → codes [M, 512].
    public static func fsqIndicesToCodes(_ indices: MLXArray, weights w: [String: MLXArray],
                                         prefix p: String = "vae.model") -> MLXArray {
        let idx = indices.asType(.int32).asArray(Int32.self)
        let basis: [Int32] = [1, 8, 64, 512, 4096]
        var flat = [Float](repeating: 0, count: idx.count * 5)
        for (m, n0) in idx.enumerated() {
            let n = ((n0 % 32768) + 32768) % 32768                   // wrap mod 8^5
            for l in 0..<5 {
                let digit = (n / basis[l]) % 8
                flat[m * 5 + l] = (Float(digit) - 4.0) / 4.0
            }
        }
        let codes5 = MLXArray(flat, [idx.count, 5]).asType(.bfloat16)
        return SkinOps.linear(codes5, w["\(p).FSQ.project_out.weight"]!, w["\(p).FSQ.project_out.bias"]!)  // [M,512]
    }

    /// `_decode` for one joint chunk. z: [b, 4, 512]; cond: [b, 384, 512]; sampledCond: [b, 54000, 6].
    /// → sigmoid logits [b, 54000, 1]. post_quant(cat[z,cond]) → 10 self blocks → kv; proj_query(queries)
    /// → cross block10 → norm_out → proj_out → sigmoid.
    public static func decode(z: MLXArray, cond: MLXArray, sampledCond: MLXArray,
                              weights w: [String: MLXArray], prefix p: String = "vae.model") -> MLXArray {
        let dec = "\(p).decoder"
        let kv = decoderKV(z: z, cond: cond, weights: w, prefix: p)   // [b,388,768]
        return decoderQuery(kv: kv, sampledCond: sampledCond, weights: w, dec: dec)   // [b,54000,1]
    }

    /// The decoder KV stack (z⊕cond → post_quant → 10 self-blocks) — computed ONCE per joint.
    static func decoderKV(z: MLXArray, cond: MLXArray, weights w: [String: MLXArray], prefix p: String) -> MLXArray {
        let dec = "\(p).decoder"
        let zc = concatenated([z, cond], axis: 1)                    // [b,388,512]
        var hs = SkinOps.linear(zc, w["\(p).post_quant.weight"]!, w["\(p).post_quant.bias"]!)  // [b,388,768]
        for i in 0..<10 { hs = selfBlock(hs, w, "\(dec).blocks.\(i)") }
        return hs
    }

    /// The decoder query pass (sampled points cross-attend the KV) for a slice of query points.
    static func decoderQuery(kv: MLXArray, sampledCond: MLXArray, weights w: [String: MLXArray], dec: String) -> MLXArray {
        // ⚠ decoder queries embed in BF16 (`_decode` casts positions to z.dtype before the embedder)
        let sc = sampledCond.asType(.float32)
        let queries = concatenated([embed(sc[0..., 0..., 0..<3], bf16Arg: true),
                                    sc[0..., 0..., 3...].asType(.bfloat16)], axis: -1)  // [b,Nq,54] bf16
        let q = SkinOps.linear(queries, w["\(dec).proj_query.weight"]!, w["\(dec).proj_query.bias"]!)  // [b,Nq,768]
        let l = crossBlock(q, kv, w, "\(dec).blocks.10")
        let ln = SkinOps.layerNorm(l, w["\(dec).norm_out.weight"]!, w["\(dec).norm_out.bias"]!)
        let logits = SkinOps.linear(ln, w["\(dec).proj_out.weight"]!, w["\(dec).proj_out.bias"]!)      // [b,Nq,1]
        return sigmoid(logits)
    }

    /// Point-BATCHED decode: identical output to `decode` (query points cross-attend only the shared
    /// KV — never each other, so batching is exact), but the `[b, heads, Nq, 388]` cross-attention
    /// transient is capped at `pointBatch` (default 8192) instead of the full 54000 → ~7× less peak
    /// activation in the skin-decode phase. KV is computed once and reused across batches.
    public static func decodeBatched(z: MLXArray, cond: MLXArray, sampledCond: MLXArray, pointBatch: Int = 8192,
                                     weights w: [String: MLXArray], prefix p: String = "vae.model") -> MLXArray {
        let dec = "\(p).decoder"
        let kv = decoderKV(z: z, cond: cond, weights: w, prefix: p)
        let N = sampledCond.shape[1]
        var cols: [MLXArray] = []
        var s = 0
        while s < N {
            let e = min(s + pointBatch, N)
            let lg = decoderQuery(kv: kv, sampledCond: sampledCond[0..., s..<e, 0...], weights: w, dec: dec)
            eval(lg); cols.append(lg)
            s = e
        }
        return concatenated(cols, axis: 1)   // [b,N,1]
    }
}
