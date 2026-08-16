import Foundation

/// S0 — the key contract (PORTING-SPEC.md S0; swift-port-parity.md §S0). Generate each component's
/// expected key set programmatically from architecture parameters and assert it equals the actual
/// `.safetensors` headers, **0 missing / 0 unused**, for every variant (refuse-partial-load). Pinning
/// the weight layout as ground truth BEFORE any model code turns every later "wrong numerics" bug
/// into a localized one.
///
/// ✅ PINNED (2026-07-10) from the real converted headers. The oracle PyTorch-Lightning `.ckpt`s were
/// converted CPU-only in the oracle venv:
///   `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` → `converted/tokenrig.safetensors`
///   `experiments/skin_vae_2_10_32768/last.ckpt`                          → `converted/skinvae.safetensors`
/// tokenrig.safetensors = 672 tensors:
///   • `transformer.*`      311  — Qwen3-0.6B backbone (28 layers, hidden 896) + rig-vocab head, NOT
///                                 tied (both `transformer.model.embed_tokens.weight` and
///                                 `transformer.lm_head.weight` present, each [33036, 896]).
///   • `mesh_encoder.encoder.*` 106 — Michelangelo point encoder (1 cross-attn block + 8 self-attn resblocks).
///   • `output_proj.*`        3  — Linear(512→896) + RMSNorm mesh soft-prefix projection.
///   • `vae.model.*`        252  — embedded SkinVAE (encoder 3 blocks / decoder 11 / cond_encoder 3 + FSQ).
/// skinvae.safetensors = 252 tensors under the `model.model.*` prefix (the standalone SkinVAE — the
/// SAME 252-key SkinVAE, different container prefix).
///
/// Everything here is pure Foundation (no MLX, never evals) → runs under `swift test` and the
/// `--s0-gate` CLI mode alike.
public enum KeyContract {

    /// The result of diffing a generated set against on-disk headers.
    public struct Diff: CustomStringConvertible {
        public let missing: [String]   // expected but not on disk
        public let unused: [String]    // on disk but not expected
        public var isClean: Bool { missing.isEmpty && unused.isEmpty }
        public var description: String {
            if isClean { return "CLEAN (0 missing / 0 unused)" }
            var s = "missing=\(missing.count) unused=\(unused.count)"
            if !missing.isEmpty { s += "\n  first missing: \(missing.prefix(8).joined(separator: ", "))" }
            if !unused.isEmpty { s += "\n  first unused:  \(unused.prefix(8).joined(separator: ", "))" }
            return s
        }
    }

    /// Refuse-partial-load contract: expected == on-disk, both directions.
    public static func diff(expected: Set<String>, onDisk: Set<String>) -> Diff {
        Diff(missing: expected.subtracting(onDisk).sorted(),
             unused: onDisk.subtracting(expected).sorted())
    }

    // MARK: - Qwen3 backbone + rig head (TokenRig `transformer.*`)

    /// Standard HF Qwen3 decoder key set for `numLayers` blocks, `prefix`-scoped. Qwen3 carries
    /// per-head QK RMSNorm (`self_attn.{q,k}_norm.weight`) — that is what distinguishes it from Qwen2.
    /// SkinTokens' TokenRig REPLACES the text head with a rig-vocab (33036) head and does NOT tie it
    /// to the embedding: both `…model.embed_tokens.weight` and `…lm_head.weight` are on disk, each
    /// [33036, 896]. So pass `tieWordEmbeddings: false` (keep `lm_head.weight`).
    public static func qwen3BackboneKeys(numLayers: Int = 28,
                                         prefix: String = "",
                                         tieWordEmbeddings: Bool = true) -> Set<String> {
        let m = "\(prefix)model"
        var keys: Set<String> = ["\(m).embed_tokens.weight", "\(m).norm.weight"]
        if !tieWordEmbeddings { keys.insert("\(prefix)lm_head.weight") }
        for i in 0..<numLayers {
            let p = "\(m).layers.\(i)"
            keys.formUnion([
                "\(p).self_attn.q_proj.weight",
                "\(p).self_attn.k_proj.weight",
                "\(p).self_attn.v_proj.weight",
                "\(p).self_attn.o_proj.weight",
                "\(p).self_attn.q_norm.weight",
                "\(p).self_attn.k_norm.weight",
                "\(p).mlp.gate_proj.weight",
                "\(p).mlp.up_proj.weight",
                "\(p).mlp.down_proj.weight",
                "\(p).input_layernorm.weight",
                "\(p).post_attention_layernorm.weight",
            ])
        }
        return keys
    }

    /// TokenRig's full `transformer.*` expected set (Qwen3 backbone + un-tied rig head).
    public static var tokenRigTransformerKeys: Set<String> {
        qwen3BackboneKeys(numLayers: 28, prefix: "transformer.", tieWordEmbeddings: false)
    }

    // MARK: - Mesh soft-prefix projection (`output_proj.*`)

    /// Linear(512→896) + RMSNorm (RMSNorm carries weight only, no bias) — the Michelangelo output_proj.
    public static var outputProjKeys: Set<String> {
        ["output_proj.0.weight", "output_proj.0.bias", "output_proj.1.weight"]
    }

    // MARK: - Michelangelo point encoder (`mesh_encoder.encoder.*`)

    /// Michelangelo Perceiver encoder: input_proj → 1 ResidualCrossAttentionBlock → `numResBlocks`
    /// self-attention resblocks → ln_post. num_latents 256, width 512, 8L/8H, embed 64, num_freqs 8,
    /// include_pi=false, point_feats 3 → learned_mesh_cond (1,512,896). 106 keys at defaults.
    public static func michelangeloKeys(prefix: String = "mesh_encoder.encoder",
                                        numResBlocks: Int = 8) -> Set<String> {
        var keys: Set<String> = [
            "\(prefix).input_proj.weight", "\(prefix).input_proj.bias",
            "\(prefix).ln_post.weight",    "\(prefix).ln_post.bias",
        ]
        // cross-attention block: attn (c_q, c_kv, c_proj+bias), ln_1/2/3 (weight+bias), mlp (c_fc, c_proj +bias)
        let ca = "\(prefix).cross_attn"
        keys.formUnion([
            "\(ca).attn.c_q.weight", "\(ca).attn.c_kv.weight",
            "\(ca).attn.c_proj.weight", "\(ca).attn.c_proj.bias",
            "\(ca).ln_1.weight", "\(ca).ln_1.bias",
            "\(ca).ln_2.weight", "\(ca).ln_2.bias",
            "\(ca).ln_3.weight", "\(ca).ln_3.bias",
            "\(ca).mlp.c_fc.weight", "\(ca).mlp.c_fc.bias",
            "\(ca).mlp.c_proj.weight", "\(ca).mlp.c_proj.bias",
        ])
        // self-attention resblocks: attn (c_qkv, c_proj+bias), ln_1/2 (weight+bias), mlp (c_fc, c_proj +bias)
        for i in 0..<numResBlocks {
            let p = "\(prefix).self_attn.resblocks.\(i)"
            keys.formUnion([
                "\(p).attn.c_qkv.weight",
                "\(p).attn.c_proj.weight", "\(p).attn.c_proj.bias",
                "\(p).ln_1.weight", "\(p).ln_1.bias",
                "\(p).ln_2.weight", "\(p).ln_2.bias",
                "\(p).mlp.c_fc.weight", "\(p).mlp.c_fc.bias",
                "\(p).mlp.c_proj.weight", "\(p).mlp.c_proj.bias",
            ])
        }
        return keys
    }

    // MARK: - SkinVAE (`vae.model.*` embedded, or `model.model.*` standalone)

    /// diffusers-style BasicTransformerBlock. Two shapes on disk:
    ///   self  (attn1 + norm1 + norm3 + ff)              — 13 keys
    ///   cross (attn2 + norm_cross + norm2 + norm3 + ff)  — 15 keys (no attn1/norm1; a query-driven block)
    private static func vaeBlockKeys(_ p: String, cross: Bool) -> Set<String> {
        // GEGLU feed-forward: net.0.proj (weight+bias), net.2 (weight+bias)
        var keys: Set<String> = [
            "\(p).norm3.weight", "\(p).norm3.bias",
            "\(p).ff.net.0.proj.weight", "\(p).ff.net.0.proj.bias",
            "\(p).ff.net.2.weight", "\(p).ff.net.2.bias",
        ]
        if cross {
            keys.formUnion([
                "\(p).attn2.norm_cross.weight", "\(p).attn2.norm_cross.bias",
                "\(p).attn2.to_q.weight", "\(p).attn2.to_k.weight", "\(p).attn2.to_v.weight",
                "\(p).attn2.to_out.0.weight", "\(p).attn2.to_out.0.bias",
                "\(p).norm2.weight", "\(p).norm2.bias",
            ])
        } else {
            keys.formUnion([
                "\(p).attn1.to_q.weight", "\(p).attn1.to_k.weight", "\(p).attn1.to_v.weight",
                "\(p).attn1.to_out.0.weight", "\(p).attn1.to_out.0.bias",
                "\(p).norm1.weight", "\(p).norm1.bias",
            ])
        }
        return keys
    }

    /// A transformer stack (`<prefix>.blocks.i`) whose block kinds are given in order.
    private static func vaeStack(_ prefix: String, crossAt: Set<Int>, count: Int) -> Set<String> {
        var keys: Set<String> = []
        for i in 0..<count {
            keys.formUnion(vaeBlockKeys("\(prefix).blocks.\(i)", cross: crossAt.contains(i)))
        }
        return keys
    }

    /// The SkinVAE key set (252 keys). `prefix` = `vae.model` (embedded in TokenRig) or `model.model`
    /// (standalone `skinvae.safetensors`). Structure pinned from headers:
    ///   • encoder      3 blocks (cross@0), + proj_in, learned_queries, norm_out
    ///   • decoder     11 blocks (cross@10), + proj_out, proj_query, norm_out
    ///   • cond_encoder 3 blocks (cross@0), + proj_in, norm_out
    ///   • FSQ project_in/out, and quant / post_quant / cond_quant projections
    public static func skinVAEKeys(prefix: String = "vae.model") -> Set<String> {
        var keys: Set<String> = []
        // FSQ + quant projections (weight+bias each)
        for k in ["FSQ.project_in", "FSQ.project_out", "quant", "post_quant", "cond_quant"] {
            keys.insert("\(prefix).\(k).weight"); keys.insert("\(prefix).\(k).bias")
        }
        // encoder
        let enc = "\(prefix).encoder"
        keys.formUnion(["\(enc).proj_in.weight", "\(enc).proj_in.bias",
                        "\(enc).learned_queries",
                        "\(enc).norm_out.weight", "\(enc).norm_out.bias"])
        keys.formUnion(vaeStack(enc, crossAt: [0], count: 3))
        // decoder
        let dec = "\(prefix).decoder"
        keys.formUnion(["\(dec).proj_out.weight", "\(dec).proj_out.bias",
                        "\(dec).proj_query.weight", "\(dec).proj_query.bias",
                        "\(dec).norm_out.weight", "\(dec).norm_out.bias"])
        keys.formUnion(vaeStack(dec, crossAt: [10], count: 11))
        // cond_encoder
        let ce = "\(prefix).cond_encoder"
        keys.formUnion(["\(ce).proj_in.weight", "\(ce).proj_in.bias",
                        "\(ce).norm_out.weight", "\(ce).norm_out.bias"])
        keys.formUnion(vaeStack(ce, crossAt: [0], count: 3))
        return keys
    }

    // MARK: - Whole-file expected sets (the S0 refuse-partial-load contract)

    /// Full expected key set for `converted/tokenrig.safetensors` (672 keys).
    public static var tokenRigKeys: Set<String> {
        tokenRigTransformerKeys
            .union(michelangeloKeys())
            .union(outputProjKeys)
            .union(skinVAEKeys(prefix: "vae.model"))
    }

    /// Full expected key set for the standalone `converted/skinvae.safetensors` (252 keys).
    public static var skinVAEStandaloneKeys: Set<String> {
        skinVAEKeys(prefix: "model.model")
    }
}
