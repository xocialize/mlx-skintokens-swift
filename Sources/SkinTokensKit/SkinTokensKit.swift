import Foundation

/// SkinTokensKit — the MLXToolKit layer for the SkinTokens `meshRig` package.
///
/// **P1 (this scaffold):** only `SkinTokensConfiguration` is live — the born-clean contract shell
/// (`PackageConfiguration`/`ModelStorable`/`QuantConfigured`/`WeightSourcing`/`WeightPrewarming`)
/// that compiles + passes the offline MAT-1..5 gate against mlx-engine-swift 0.27.0.
///
/// **S7 (deferred — needs the engine `meshRig` contract bump):**
///   • `SkinTokensPackage: ModelPackage` — manifest (split footprint, license twice, specialties),
///     `load()` (auto-materialize → SkinTokensCore assembly), `run()` (mesh GLB → rig; CAN entry
///     checkpoint first, per-token checkpoints in the beam loop), C0–C13 + CAN-1..3 gates.
///   • `GLBIO` — original-glTF parse (verts/faces/normals/UVs) + in-place JOINTS_0/WEIGHTS_0 +
///     skeleton-node injection, preserving buffers/materials/UVs (S4; Trellis2 GLBEncoder heritage).
///   • `WeightMaterializer` — HubClient snapshot download of `xocialize/skintokens-mlx` (S7).
///
/// See ../../SkinTokensDev/PORTING-SPEC.md (phase table) and HANDOFF.md.
public enum SkinTokensKit {
    public static let version = "0.0.1-p1-scaffold"
}
