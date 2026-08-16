# mlx-skintokens-swift

Swift-MLX port of [VAST-AI/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) mesh
auto-rigging (UniRig successor) — mesh GLB in → skeleton + per-vertex skin weights out, as one
autoregressive token sequence. Targets a net-new MLXEngine `meshRig` capability.

- **SkinTokensCore** — the ported MLX blocks (pure MLX; no MLXToolKit/engine import). Parity by
  **injecting the P0 PyTorch captures** and matching deterministic stages (NOT seed-matching — P0
  proved CPU generation is non-reproducible under a fixed seed).
- **SkinTokensKit** — the MLXToolKit layer. P1 ships the born-MAT-clean `SkinTokensConfiguration`
  shell; the `meshRig` `ModelPackage` conformer + C0–C13/CAN land at **S7** (engine contract bump).
- **RunSkinTokens** — parity-gate + GPU-e2e CLI (`--sNN-gate` subcommands; not `swift test`).
- **RunSkinTokensEngine** — engine-driven e2e CLI (S7 placeholder).

Plan, phase tracker, oracle captures, and handoff live in
⚠️ `../../DEV/SkinTokensDev/` (`PLAN.md`, `PORTING-SPEC.md`, `HANDOFF.md`) **NO LONGER EXISTS** —
unrecoverable (AB-T-0046). Rescued P0 verdicts: **AB-R-0057**; Swift port spec: **AB-L-0032**.

## P1 status

Scaffold + offline contract build green. Live now: the shared `.npy` fixture reader (`NPYReader`),
the S0 substrate (`SafetensorsHeader`, `KeyContract`), and the offline **MAT-1..5** gate on the
Configuration. S0's key diff is BLOCKED on a `.ckpt → .safetensors` conversion of the oracle weights.

```
swift build && swift test              # offline contract + MAT gate + Core structural tests
swift run RunSkinTokens --version
swift run RunSkinTokens --npy ../../DEV/SkinTokensDev/captures/giraffe_beam1_s0/skin_pred.npy
```
