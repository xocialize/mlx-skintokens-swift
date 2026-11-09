// swift-tools-version:5.9
import PackageDescription

// mlx-skintokens-swift — Swift-MLX port of VAST-AI/SkinTokens mesh auto-rigging (UniRig successor).
//
// Pipeline: mesh GLB → surface point sampling (SamplerMix 54000) → Michelangelo point encoder →
//   SkinVAE _encode → TokenRig AR transformer (stock Qwen3-0.6B on inputs_embeds, grammar-constrained
//   batched beam(10) + 7-state DFA) → FSQ[8,8,8,8,8] chunked decoder → per-sampled-point skin →
//   cKDTree propagate to ORIGINAL glTF verts → in-place JOINTS_0/WEIGHTS_0 GLB injection.
//
// SkinTokensCore: the parity-validated ported building blocks (pure MLX; NO MLXToolKit / engine
//   import — the only thing actually ported). Each stage is gated by INJECTING the P0 PyTorch
//   captures (captures/{giraffe,humanoid}_beam1_s0/*.npy) and matching the deterministic stages,
//   NOT by seed-matching (P0 proved CPU generation is non-reproducible under a fixed seed).
//
// SkinTokensKit: the MLXToolKit layer — the `meshRig` `ModelPackage` conformer (SkinTokensPackage),
//   its WeightSourcing/MAT configuration, and first-run weight materialization. Offline C0–C13 +
//   CAN-1..3 + MAT-1..5 run under `swift test`.
let package = Package(
    name: "mlx-skintokens-swift",
    platforms: [.macOS("26.0")],   // MLXToolKit (engine contract) floor
    products: [
        .library(name: "SkinTokensCore", targets: ["SkinTokensCore"]),
        .library(name: "SkinTokensGLTF", targets: ["SkinTokensGLTF"]),
        .library(name: "SkinTokensRig", targets: ["SkinTokensRig"]),
        .library(name: "SkinTokensKit", targets: ["SkinTokensKit"]),
    ],
    dependencies: [
        // mlx-swift comes through the fleet's pinned fork (AB-D-0208: upstream 0.32.3 + the
        // _updateInternal leak fix). Must be the SAME URL the engine uses, or SPM sees two
        // `mlx-swift` identities and refuses to resolve.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.32.4"),
        // MLXEngine contract + coordinator. `meshRig` landed in contract 1.19.0 (engine v0.30.0);
        // re-floored on the current engine for the PROD promotion.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.76.0"),
        // First-run auto-materialization (MAT gate): HubClient snapshot download of VAST-AI/SkinTokens
        // (once converted to safetensors) into the engine ModelStore layout.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        // The ported MLX blocks + the shared fixture readers (.npy) and S0 key contract
        // (safetensors headers). Pure MLX / Foundation — no MLXToolKit, no engine.
        .target(name: "SkinTokensCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
        ]),
        // Vendored GLTFKit (from xocialize-convert-to-usdz/converter) — dependency-free VRM/glTF
        // parser: world-space node-transform composition + existing-skeleton + humanoid bone map, for
        // the Route-B (VRM skin-only) path. See Sources/SkinTokensGLTF/VENDORED.md.
        .target(name: "SkinTokensGLTF"),
        // Engine-FREE orchestration: VRM/GLB load + the end-to-end rig pipeline (auto + Route-B),
        // shared by the CLI gates and the ModelPackage conformer. Deps Core + GLTF, NO engine — so
        // the gate harness stays engine-free and the conformer just wraps this.
        .target(name: "SkinTokensRig", dependencies: [
            "SkinTokensCore",
            "SkinTokensGLTF",
            .product(name: "MLX", package: "mlx-swift"),
        ], resources: [
            // Canonical VRoid T-pose skeleton (141 joints, 54 VRM humanoid bones) harvested from a
            // reference VRoid — the Route-B template fitted onto bare generated meshes.
            .copy("Resources/vroid-template.json"),
        ]),
        // The MLXToolKit layer: the `meshRig` ModelPackage conformer + Configuration
        // (WeightSourcing/ModelStorable/QuantConfigured) + first-run weight materialization.
        .target(name: "SkinTokensKit", dependencies: [
            "SkinTokensCore",
            "SkinTokensRig",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
        ]),
        // GPU end-to-end + parity-gate driver CLI. Gates run as `--sNN-gate` subcommands (NOT
        // `swift test` — SPM test-target metallib is unreliable). Run heavy gates detached, -c release.
        .executableTarget(name: "RunSkinTokens", dependencies: [
            "SkinTokensCore",
            "SkinTokensGLTF",
            "SkinTokensRig",
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        // Engine-driven GPU e2e CLI: drives the package THROUGH MLXServeEngine
        // (register → prepare → run); MODELS_ROOT exercises first-run materialization.
        .executableTarget(name: "RunSkinTokensEngine", dependencies: [
            "SkinTokensKit",
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        // Offline conformance suite (C0–C13 + CAN-1..3 + MAT-1..5 + Core structural tests — no MLX
        // kernels, no weights; safe under `swift test`).
        .testTarget(name: "SkinTokensKitTests", dependencies: [
            "SkinTokensKit",
            "SkinTokensCore",
            .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
        ], resources: [
            // S2b DFA goldens (dumped from the oracle TokenizerPart + captured output_ids). Loaded
            // via #filePath in-tree; declared here so SPM stops flagging it as unhandled.
            .copy("Fixtures/grammar_goldens.json"),
        ]),
    ]
)
