import Foundation
import MLX
import MLXToolKit
import SkinTokensCore
import SkinTokensRig

/// Runtime errors for the SkinTokens `meshRig` package.
public enum SkinTokensPackageError: Error, Sendable {
    case weightsNotFound(String)
    /// Only `.bf16` (default) and `.int8` are supported. int4 was measured and rejected: it reassigns
    /// ~27% of vertices to a different bone (AB-R-0812).
    case unsupportedQuant(Quant)
}

/// MLXToolKit `ModelPackage` conformer for SkinTokens mesh auto-rigging (`meshRig`, contract 1.19.0).
///
/// One model, two surfaces of the SAME capability, dispatched inside `run(_:)` on `request.mode`:
///   • `MeshRigContract.auto` — generate a skeleton + skin (any mesh, e.g. a TRELLIS.2 output).
///   • `MeshRigContract.skinOnly` — skin the input GLB's own skeleton (the companion VRM path).
/// The heavy pipeline lives in the engine-free `SkinTokensRig.SkinTokensPipeline` (the same code the
/// `--e2e` / `--route-b` CLI gates validated); this type is the contract wrapper (C13 inversion of
/// control — the engine constructs, loads, drives, evicts it; it never self-caches).
///
/// License: SkinTokens is MIT (weights, C7) and our Swift port is MIT (C8) — the default
/// `.permissiveOnly` gate admits it. `Qwen3-0.6B` backbone weights ship INSIDE the SkinTokens
/// checkpoint (Apache-2.0, also permissive).
@InferenceActor
public final class SkinTokensPackage: ModelPackage {
    public typealias Configuration = SkinTokensConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "VAST-AI/SkinTokens", revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // Split footprint — MEASURED (`RunSkinTokens --measure`, M-series GPU) after the P2 +
                // bf16-KV efficiency levers: resident weights floor = 1.19 GB (tokenrig incl. embedded
                // VAE). Peak activation, max over modes: auto (giraffe) ~2.5 GB, skinOnly (VRM J=141)
                // **4.08 GB** — worst case (bf16 beam(10) KV over a ~1k-token skeleton prefix). Declared
                // 4.2 GB → total peak ~5.3 GB, so admission now fits an **8 GB machine** (was ≥16 GB).
                // Levers applied: (1) prefill ONCE at B=1 then tile the KV to 10 beams (cuts the O(T²)
                // prefill attention 10×); (2) point-BATCHED skin decode (caps the 54000-pt cross-attn
                // transient); (3) **bf16 KV cache** (halves the ~2.75 GB beam cache — the biggest lever;
                // `Qwen3.Config.kvDtype`, compute stays fp32, generation is non-parity so safe). Net
                // 8.21 → 4.08 GB (50%). Further (unimplemented): fewer beams for skinOnly (fixed skeleton).
                //
                // v0.2.0 re-measure (2026-11-09, M5 Max, Release, fused SDPA default — AB-R-0812):
                // bf16 resident 1.19 GB; int8 resident 0.83 GB (backbone quantized at load, dense
                // originals dropped). Activations are identical across tiers on the same workload
                // (VRM 2.42 GB either way); the longest sequence measured — 210k-vert TRELLIS cap, auto,
                // 1,731 steps — peaked at 3.97 GB activation. 4.2 GB stays the declared max for both.
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 1_200_000_000,
                                            peakActivationBytes: 4_200_000_000),
                             QuantFootprint(quant: .int8, residentBytes: 850_000_000,
                                            peakActivationBytes: 4_200_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .base),
            specialties: [SpecialtyWeight(Specialty("mesh-rigging"), strength: 1.0),
                          SpecialtyWeight(Specialty("character-rigging"), strength: 1.0)],
            surfaces: [MeshRigContract.descriptor(
                name: "skintokens-mesh-rig",
                summary: "Auto-rig a 3D mesh (GLB): generate a skeleton + skin weights (auto), or skin a "
                       + "provided/embedded skeleton (skinOnly — the VRM companion-character path).",
                modes: [MeshRigContract.auto, MeshRigContract.skinOnly])])
    }

    /// Quant tiers this package can load (each has a declared `QuantFootprint`).
    public nonisolated static let supportedQuants: Set<Quant> = [.bf16, .int8]

    private let configuration: Configuration
    private var weights: [String: MLXArray]?     // host-owned lifecycle; no global singleton (C13)

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    /// Harness convenience mirroring the other `-swift` packages: the engine-side registration value.
    public nonisolated static var registration: PackageRegistration { .of(SkinTokensPackage.self) }

    public func load() async throws {
        let root = configuration.modelsRootDirectory
        let override = configuration.weightsRootOverride
        // First-run auto-materialization (engine ≥0.19.0 MAT gate): a dir-less config with a stamped
        // store root downloads the consolidated snapshot into the ModelStore layout.
        if override == nil, let root {
            let missing = configuration.missingWeightSources(storeRoot: root)
            if !missing.isEmpty { try await WeightMaterializer.materialize(missing, into: root) }
        }
        guard Self.supportedQuants.contains(configuration.quant) else {
            throw SkinTokensPackageError.unsupportedQuant(configuration.quant)
        }
        let path = try Self.resolve("tokenrig.safetensors", root: root, override: override)
        var w = try Weights.load(path)   // CPU-stream load (Metal-watchdog rule)
        if configuration.quant == .int8 {
            // Quantize the Qwen3 backbone projections from the bf16 snapshot and drop the dense
            // originals — one weight source on the Hub, no separate -8bit repo (AB-R-0812).
            Qwen3.prequantize(&w, bits: 8)
            MLX.Memory.clearCache()
        }
        weights = w
    }

    public func unload() async { weights = nil; MLX.Memory.clearCache() }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN gate: the entry checkpoint is the FIRST act — before capability validation and the
        // notLoaded guard — so a pre-cancelled run refuses before touching weights (CAN-1).
        try Task.checkCancellation()
        guard request.capability == .meshRig, let req = request as? MeshRigRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        guard let w = weights else { throw PackageError.notLoaded }

        // Mode: request `mode` wins, else the configuration default (skinOnly bool).
        let skinOnly = (req.mode == MeshRigContract.skinOnly)
            || (req.mode == nil && configuration.skinOnly)
        // cls conditioning: request metaData override, else the configured cls (default "vroid" for skinOnly).
        let cls = (req.metaData["cls"] as? String) ?? configuration.cls ?? (skinOnly ? "vroid" : nil)
        // For skinOnly with a separately-supplied skeleton GLB, that skeleton IS the mesh to rig only
        // when the caller passes it as the source; V1 uses the mesh's own embedded skeleton (`req.skeleton`
        // is lane-ready — a future explicit-skeleton path merges it before rigging).
        // CAN cadence: `checkpoint` fires every generation step (per token) + at stage seams. The
        // rethrown CancellationError unwinds unchanged so the engine classifies user-cancel vs preempt.
        var params = SkinTokensPipeline.Params()
        if configuration.quant == .int8 { params.gen.weights = .int8 }
        let riggedGLB = try SkinTokensPipeline.rig(glbData: req.mesh.data, skinOnly: skinOnly, cls: cls,
                                                   weights: w, params: params,
                                                   checkpoint: { try Task.checkCancellation() })

        // CAN: a mid-run cancellation surfaces here (the generate loop already returned); rethrow before
        // materializing the response so the engine classifies it as a user cancel, not a result.
        try Task.checkCancellation()
        return MeshRigResponse(mesh: Mesh(format: .glb, data: riggedGLB))
    }

    // MARK: - weight resolution (override → ModelStore layout → HF cache)

    private static func resolve(_ file: String, root: URL?, override: URL?) throws -> String {
        let fm = FileManager.default
        let repo = Configuration.consolidatedRepo
        if let override {
            let flat = override.appendingPathComponent(file).path
            if fm.fileExists(atPath: flat) { return flat }
            let nested = override.appending(path: repo).appendingPathComponent(file).path
            if fm.fileExists(atPath: nested) { return nested }
        }
        if let root, let dir = ModelStore(root: root).directory(for: repo) {
            let p = dir.appendingPathComponent(file).path
            if fm.fileExists(atPath: p) { return p }
        }
        // real HF hub cache fallback (non-sandboxed CLIs)
        let home = fm.homeDirectoryForCurrentUser.path
        let base = "\(home)/.cache/huggingface/hub/models--\(repo.replacingOccurrences(of: "/", with: "--"))/snapshots"
        if let snaps = try? fm.contentsOfDirectory(atPath: base) {
            for s in snaps { let p = "\(base)/\(s)/\(file)"; if fm.fileExists(atPath: p) { return p } }
        }
        throw SkinTokensPackageError.weightsNotFound("\(repo)/\(file)")
    }
}

// Registration (call site, when integrating):
//   try await engine.register(PackageRegistration.of(SkinTokensPackage.self),
//                             configuration: SkinTokensConfiguration())
