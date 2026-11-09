import Foundation
import MLXToolKit

/// Init-time configuration (C9) for the SkinTokens mesh auto-rigging package.
///
/// `PackageConfiguration` + `ModelStorable` + `QuantConfigured` + `WeightSourcing` +
/// `WeightPrewarming`, consumed by the `meshRig` `SkinTokensPackage` conformer. Passes the offline
/// MAT-1..5 gate (MaterializationTests).
public struct SkinTokensConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var weightsRepo: String
    public var revision: String?
    public var quant: Quant
    public var modelsRootDirectory: URL?   // ModelStorable — the engine stamps the store root

    /// Route B (companion characters): feed a caller-provided skeleton as the generation prefix and
    /// predict skin only (P0-verified: J preserved, clean weights). nil ⇒ `auto` (learn skeleton+skin).
    public var skinOnly: Bool
    /// Skeleton-conditioning class token. `"vroid"` is the trained VRM-humanoid cls (Route B); nil for `auto`.
    public var cls: String?

    /// Explicit weights root (out-of-sandbox snapshot / dev escape hatch). When set, `load()` resolves
    /// the checkpoints under this directory FIRST and never touches the network. nil ⇒ the
    /// auto-materialization path (ModelStore layout, then the HF cache fallback for the CLIs).
    public var weightsRootOverride: URL?

    public init(weightsRepo: String = "VAST-AI/SkinTokens",
                revision: String? = "main",
                quant: Quant = .bf16,
                modelsRootDirectory: URL? = nil,
                skinOnly: Bool = false,
                cls: String? = nil,
                weightsRootOverride: URL? = nil) {
        self.weightsRepo = weightsRepo
        self.revision = revision
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.skinOnly = skinOnly
        self.cls = cls
        self.weightsRootOverride = weightsRootOverride
    }
}

// MARK: - Weight sources (auto-materialization, engine ≥0.19.0 MAT gate)

extension SkinTokensConfiguration: WeightSourcing {
    /// The consolidated, remap-free Swift-layout repo (one download). Upstream ships PyTorch `.ckpt`
    /// (grpo_1400 TokenRig + skin_vae last); the S0 conversion (`.ckpt → .safetensors`, CPU,
    /// `_orig_mod` compile-wrapper stripped) produced this repo. **PUBLISHED 2026-07-10 to
    /// mlx-community** (community naming `<Name>-<precision>`; bf16 = the dominant/loaded tokenrig
    /// dtype) — public, MIT (+ the embedded Qwen3-0.6B backbone, Apache-2.0).
    public static let consolidatedRepo = "mlx-community/SkinTokens-bf16"

    /// Every file the port needs present. The probe is strict (a half-finished download must re-report
    /// missing rather than silently degrade). ✅ safetensors names CONFIRMED against the converted
    /// headers (2026-07-10): tokenrig 672 keys / skinvae 252 keys, both S0-CLEAN — kept in lockstep
    /// with `KeyContract.{tokenRigKeys, skinVAEStandaloneKeys}`. `config.json` (resolved from the
    /// emitted `*.hparams.json`) + the Route-B template accompany the safetensors in the published repo.
    public static let probeFiles = [
        "config.json",              // resolved TokenRig + SkinVAE + tokenizer config (from *.hparams.json)
        "tokenrig.safetensors",     // 672: Qwen3-0.6B backbone + un-tied rig head + Michelangelo + embedded VAE
        "skinvae.safetensors",      // 252: standalone SkinVAE — FSQ[8,8,8,8,8] + enc3/dec11 chunked decoder
        "skeleton_vroid.json",      // Route-B canonical VRM template (configs/skeleton/vroid.yaml → json)
    ]

    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: Self.consolidatedRepo, revision: revision,
                      matching: ["*.safetensors", "*.json"])]
    }

    /// Explicit `weightsRootOverride` first (flat `<dir>/<file>` or `<dir>/<org>/<name>/<file>`), then
    /// the ModelStore repo folder (`<root>/models--mlx-community--SkinTokens-bf16/`, files flat — the
    /// materializer copies the snapshot there). The real HF cache stays a load()-time
    /// fallback only, so the fresh-machine posture (MAT-4) stays honest.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = weightsRootOverride,
           Self.snapshotPresent(at: dir)
            || Self.snapshotPresent(at: dir.appending(path: Self.consolidatedRepo)) {
            return []
        }
        if let dir = ModelStore(root: storeRoot).directory(for: Self.consolidatedRepo),
           Self.snapshotPresent(at: dir) {
            return []
        }
        return weightSources
    }

    static func snapshotPresent(at dir: URL) -> Bool {
        let fm = FileManager.default
        return probeFiles.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
    }
}

// MARK: - Cold-start prewarm

extension SkinTokensConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        let dir = weightsRootOverride
            ?? ModelStore(root: modelsRootDirectory).directory(for: Self.consolidatedRepo)
        guard let dir else { return [] }
        return [dir.appending(path: "config.json"), dir]
    }
}
