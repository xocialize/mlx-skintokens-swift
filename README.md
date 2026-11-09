# mlx-skintokens-swift

Swift/MLX port of [VAST-AI/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens), mesh
auto-rigging (the UniRig successor), for Apple Silicon. A GLB/VRM mesh goes in; the same geometry
comes out with a skeleton and per-vertex skin weights (`JOINTS_0`/`WEIGHTS_0`). It is the
MLXEngine **`meshRig`** capability.

- **Modes:** `auto` learns a skeleton and skin for any mesh (for example a TRELLIS.2 output).
  `skinOnly` skins the input's own skeleton (the VRM companion-character path: the joints that go
  in are the joints that come out). A bare mesh in `skinOnly` gets the bundled VRoid T-pose
  skeleton fitted onto it, with VRM humanoid bone names.
- **Weights:** [`mlx-community/SkinTokens-bf16`](https://huggingface.co/mlx-community/SkinTokens-bf16),
  public, MIT. They download on first `prepare` into the engine's model store. The int8 tier is
  quantized from these at load, so there is only one weight repo.
- **License:** MIT port of MIT code and weights (Qwen3-0.6B backbone weights are Apache-2.0), so
  the engine's default `.permissiveOnly` gate admits it. See `THIRD_PARTY_LICENSES.md`.

## Use it

```swift
.package(url: "https://github.com/xocialize/mlx-skintokens-swift", from: "0.2.0"),
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
                                   configuration: SkinTokensConfiguration())   // quant: .int8 for the smaller tier
try await engine.prepare(.meshRig, package: id)
let response = try await engine.run(
    MeshRigRequest(mesh: Mesh(format: .glb, data: glbBytes), mode: MeshRigContract.auto),
    package: id) as! MeshRigResponse
try response.mesh.data.write(to: outURL)                       // rigged GLB
```

For `skinOnly`, pass `mode: MeshRigContract.skinOnly` (cls defaults to `"vroid"`).

Requires macOS 26 and `mlx-engine-swift` ≥ 0.76.0. mlx-swift comes in through the engine, from
the fleet's pinned fork `xocialize/mlx-swift`. If your app also depends on a package that pins
upstream `ml-explore/mlx-swift`, SwiftPM warns about conflicting `mlx-swift` identities and may
pick upstream, which loses the fork's leak fix. Keep every package on the fork.

## Quant tiers

| `quant` | Resident weights | Peak activation (declared) | Fits | Notes |
|---|---|---|---|---|
| `.bf16` (default) | 1.2 GB | 4.2 GB | 8 GB Mac | Reference output. |
| `.int8` | 0.85 GB | 4.2 GB | 8 GB Mac | Qwen3 backbone quantized at load (affine, group 64). 10–13% faster end to end. Skin is close to bf16: 98.9–99.4% of vertices keep the same main bone. |

int4 is not supported, and `load()` throws `unsupportedQuant` for it. When measured it was no
faster than int8 and reassigned about 27% of vertices to a different bone.

Activation memory is the same on both tiers; it grows with the generated sequence length (beam
KV cache), not with mesh size. Measured on an M5 Max, Release build: 2.42 GB activation for a
humanoid VRM, and 3.97 GB for the longest sequence seen (a 210k-vertex TRELLIS cap, 1,731
generation steps).

## Performance

Generation runs a grammar-constrained beam search (beam 10) over the Qwen3-0.6B backbone, so time
scales with the number of generated tokens, which tracks joint count, not vertex count.

M5 Max, Release, quiet GPU, total rig time through `MLXServeEngine` (v0.2.0):

| Input | Mode | bf16 | int8 |
|---|---|---|---|
| girl-base.vrm (22.5k verts, 4 skins) | `skinOnly` | 12.7 s | 11.4 s |
| Spec.vrm (34.5k verts) → 52 joints | `auto` | 19.4 s | 16.8 s |

v0.2.0 made fused attention (`MLXFast` SDPA with native GQA) the generation default: 27–33% less
time per step than v0.1.0's explicit attention, with output the same on in-distribution meshes.
The first run in a fresh process is slower while Metal compiles its shaders.

**Out-of-distribution meshes in `auto` mode** (props, rigid objects, very large TRELLIS outputs)
can generate long, degenerate skeletons: a TRELLIS cap produced a 173-joint, 40-deep chain and
took about 2–3 minutes. On those meshes the skeleton is also sensitive to tiny numeric changes, so
bf16, int8 and different library versions can each return a different skeleton. For characters,
prefer `skinOnly` with a real or template skeleton.

## Products

| Product | What it is |
|---|---|
| `SkinTokensKit` | The `meshRig` `ModelPackage` conformer (`SkinTokensPackage`) and its configuration. This is the one apps use. |
| `SkinTokensRig` | Engine-free pipeline: VRM/GLB load, rig, inject. Shared by the CLIs and the conformer. |
| `SkinTokensCore` | The ported MLX blocks: Michelangelo point encoder, SkinVAE (FSQ[8^5]), Qwen3-0.6B backbone with KV cache, grammar-constrained batched beam search (beam 10), KD-tree skin propagation. |
| `SkinTokensGLTF` | Vendored dependency-free glTF/VRM parser (see `Sources/SkinTokensGLTF/VENDORED.md`). |

## Test and measure

```
swift test                     # offline: C0–C13, CAN-1..3, MAT-1..5, quant tiers, DFA goldens (49 tests)
```

The GPU tools run from a Release build:

```
swift build -c release
```

Live run through the engine. It downloads the weights into `MODELS_ROOT` on first run:

```
IN_GLB=mesh.glb OUT_GLB=out.glb MODELS_ROOT=/path/to/models .build/release/RunSkinTokensEngine
```

`MODE=skinOnly` runs the VRM path, `QUANT=int8` the int8 tier, and `WEIGHTS_DIR=` points at a
local snapshot instead of the store.

Footprint (resident weights and peak activation), the numbers behind the manifest:

```
.build/release/RunSkinTokens --measure <weights-dir> <mesh.glb> [--route-b] [--quant int8]
```

Speed and quality comparison of decoder variants on one mesh. It writes one GLB per variant to
diff against the first:

```
.build/release/RunSkinTokens --quant-spike <weights-dir> <mesh.glb> --out <dir> [--route-b] \
    --variants castPerStep,castPerStep+sdpa,int8+sdpa
```

`RunSkinTokens` also holds the stage parity gates (`--sNN-gate`); run them detached.

## History

- **v0.2.0** (2026-11-09): fused SDPA attention is the generation default. New `.int8` tier with
  its own footprint. int4 measured and rejected. Footprints re-measured. Bridge receipt AB-R-0812.
- **v0.1.0** (2026-11-09): first tag; promoted to PROD, repo made public, re-floored on engine
  0.76.0. Bridge receipt AB-R-0808.

Parity was proved by injecting P0 PyTorch captures and matching the deterministic stages
(Michelangelo cos .9994, SkinVAE .9998, FSQ .99999, Qwen3 teacher-forced .9998). Seed-matching
was not used, because CPU generation upstream is not reproducible under a fixed seed. The
original plan and handoff docs were lost (AB-T-0046). Their verdicts survive on the bridge as
**AB-R-0057** (P0 measurements) and **AB-L-0032** (port spec). The engine registry row in
`mlx-engine-swift/docs/model-registry.md` has the full stage record.
