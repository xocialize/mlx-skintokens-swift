# mlx-skintokens-swift

Swift/MLX port of [VAST-AI/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), mesh
auto-rigging (the UniRig successor), for Apple Silicon. A GLB/VRM mesh goes in; the same geometry
comes out with a skeleton and per-vertex skin weights (`JOINTS_0`/`WEIGHTS_0`). It is the
MLXEngine **`meshRig`** capability.

- **Modes:** `auto` learns a skeleton and skin for any mesh (for example a TRELLIS.2 output).
  `skinOnly` skins the input's own skeleton (the VRM companion-character path: the joints that go
  in are the joints that come out).
- **Weights:** [`mlx-community/SkinTokens-bf16`](https://huggingface.co/mlx-community/SkinTokens-bf16),
  public, MIT. They download on first `prepare` into the engine's model store.
- **Footprint:** about 1.2 GB resident weights and a 4.2 GB peak (worst case is `skinOnly` on a
  141-joint VRM), so it fits an 8 GB Mac.
- **License:** MIT port of MIT code and weights (Qwen3-0.6B backbone weights are Apache-2.0), so
  the engine's default `.permissiveOnly` gate admits it. See `THIRD_PARTY_LICENSES.md`.

## Use it

```swift
.package(url: "https://github.com/xocialize/mlx-skintokens-swift", from: "0.1.0"),
// target dependency:
.product(name: "SkinTokensKit", package: "mlx-skintokens-swift"),
```

```swift
import MLXServeCore
import MLXToolKit
import SkinTokensKit

let engine = MLXServeEngine()
await engine.useModelStore(ModelStore(root: modelsFolder))    // weights land here on first run
let id = try await engine.register(SkinTokensPackage.registration,
                                   configuration: SkinTokensConfiguration())
try await engine.prepare(.meshRig, package: id)
let response = try await engine.run(
    MeshRigRequest(mesh: Mesh(format: .glb, data: glbBytes), mode: MeshRigContract.auto),
    package: id) as! MeshRigResponse
try response.mesh.data.write(to: outURL)                       // rigged GLB
```

For `skinOnly`, pass `mode: MeshRigContract.skinOnly` (cls defaults to `"vroid"`).

Requires macOS 26 and `mlx-engine-swift` ≥ 0.76.0. mlx-swift comes in through the engine, from
the fleet's pinned fork `xocialize/mlx-swift`.

## Products

| Product | What it is |
|---|---|
| `SkinTokensKit` | The `meshRig` `ModelPackage` conformer (`SkinTokensPackage`) and its configuration. This is the one apps use. |
| `SkinTokensRig` | Engine-free pipeline: VRM/GLB load, rig, inject. Shared by the CLIs and the conformer. |
| `SkinTokensCore` | The ported MLX blocks: Michelangelo point encoder, SkinVAE (FSQ[8^5]), Qwen3-0.6B backbone with KV cache, grammar-constrained batched beam search (beam 10), KD-tree skin propagation. |
| `SkinTokensGLTF` | Vendored dependency-free glTF/VRM parser (see `Sources/SkinTokensGLTF/VENDORED.md`). |

## Test

```
swift test                     # offline: C0–C13 conformance, CAN-1..3, MAT-1..5, DFA goldens (44 tests)
```

Live GPU run through the engine (downloads the weights into `MODELS_ROOT` on first run):

```
swift build -c release --product RunSkinTokensEngine
IN_GLB=mesh.glb OUT_GLB=out.glb MODELS_ROOT=/path/to/models .build/release/RunSkinTokensEngine
```

`MODE=skinOnly` runs the VRM path; `WEIGHTS_DIR=` points at a local snapshot instead.
`RunSkinTokens` holds the stage parity gates (`--sNN-gate`); run them detached with `-c release`.

## History

Parity was proved by injecting P0 PyTorch captures and matching the deterministic stages
(Michelangelo cos .9994, SkinVAE .9998, FSQ .99999, Qwen3 teacher-forced .9998). Seed-matching
was not used, because CPU generation upstream is not reproducible under a fixed seed. The
original plan and handoff docs were lost (AB-T-0046). Their verdicts survive on the bridge as
**AB-R-0057** (P0 measurements) and **AB-L-0032** (port spec). The engine registry row in
`mlx-engine-swift/docs/model-registry.md` has the full stage record.
