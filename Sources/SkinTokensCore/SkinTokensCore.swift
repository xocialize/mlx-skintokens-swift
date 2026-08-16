import Foundation

/// SkinTokensCore — the ported MLX building blocks of VAST-AI/SkinTokens mesh auto-rigging.
/// Pure MLX / Foundation; no MLXToolKit or engine import (the only thing actually *ported*).
///
/// P1 phase map (PORTING-SPEC.md; parity by INJECTING P0 captures, NOT seed-matching):
///
///   S0   KeyContract + SafetensorsHeader  — ✅ PASSED (2026-07-10): tokenrig 672/672, skinvae 252/252
///   S1a  Mesh I/O + predict-transform + SamplerMix(54000)          — TODO
///   S1b  Michelangelo point encoder → learned_mesh_cond (1,512,896) — TODO
///   S1c  SkinVAE _encode → cond_latents (1,384,512)                 — TODO
///   S1d  FSQ[8,8,8,8,8] + chunked decoder → skin_pred (54000,J)     — TODO
///   S2   TokenRig forward (Qwen3-0.6B, inputs_embeds) vs step_logits — TODO (reuse MLXQwenLLM layers)
///   S2b  ⭐ 7-state DFA + VocabSwitchingLogitsProcessor + beam(10)   — 🔶 DFA PASSED (SkeletonGrammar,
///        oracle-golden set/count/replay parity); MLX per-beam mask + batched beam(10) still TODO
///   S3   cKDTree propagate 54k sampled → ORIGINAL glTF verts        — TODO
///   S4   GLB inject JOINTS_0/WEIGHTS_0                              — TODO (SkinTokensKit/GLBIO)
///   S5   Route-B skin-only (cls='vroid')                           — TODO
///
/// The shared fixture reader (`NPYReader`), S0 substrate (`SafetensorsHeader`, `KeyContract`), and the
/// S2b `SkinTokenGrammar` DFA are live now; every `--sNN-gate` CLI mode reads the captures the same way.
public enum SkinTokensCore {
    /// Bumped as phases land; stamped into gate output for provenance.
    public static let version = "0.0.2-p1-s0-passed+dfa"
}
