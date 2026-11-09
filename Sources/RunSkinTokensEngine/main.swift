import Foundation
import MLX
import MLXToolKit
import MLXServeCore
import SkinTokensKit

// RunSkinTokensEngine — the engine-driven GPU e2e CLI (mlx-swift-integration Stage-2 step 7).
//
// Drives the SkinTokens `meshRig` package THROUGH `MLXServeEngine` — the real coordinator path:
//   register (license gate C7/C8 + device eligibility C10) → prepare (memory admission, timed)
//   → run (timed) → decode the rigged `Mesh`/GLB artifact → write it. Proves the S7 integration
//   without the Xcode app. Real weights on the GPU stream; run DETACHED (heavy GPU, ~30–60s).
//
// Env:
//   IN_GLB      input mesh/VRM GLB (required)
//   OUT_GLB     output rigged GLB (default ../skintokens_engine_out.glb)
//   MODE        "auto" (default) or "skinOnly" (the VRM companion path)
//   CLS         cls conditioning ("vroid" default for skinOnly)
//   WEIGHTS_DIR weights root override (e.g. the local converted/ dir for pre-upload validation)
//   MODELS_ROOT engine model-store root (`engine.useModelStore`) — the app's path: prepare()
//               materializes mlx-community/SkinTokens-bf16 there on first run
//   QUANT       "bf16" (default) or "int8" (backbone quantized at load)

setvbuf(stdout, nil, _IONBF, 0)

func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }

let env = ProcessInfo.processInfo.environment
guard let inGLB = env["IN_GLB"], let glbData = FileManager.default.contents(atPath: inGLB) else {
    fatalError("missing IN_GLB=<mesh.glb>")
}
let outGLB = env["OUT_GLB"] ?? "\(FileManager.default.currentDirectoryPath)/../skintokens_engine_out.glb"
let modeStr = env["MODE"] ?? "auto"
let mode: Mode = modeStr == "skinOnly" ? MeshRigContract.skinOnly : MeshRigContract.auto
let cls = env["CLS"]
let weightsOverride = env["WEIGHTS_DIR"].map { URL(fileURLWithPath: $0) }
let modelsRoot = env["MODELS_ROOT"].map { URL(fileURLWithPath: $0, isDirectory: true) }
let quant: Quant = env["QUANT"] == "int8" ? .int8 : .bf16

print("================ SkinTokens meshRig · ENGINE-DRIVEN e2e ================")
print("[engine] input: \(inGLB) (\(glbData.count) bytes) | mode=\(mode.rawValue) cls=\(cls ?? "-") quant=\(quant.rawValue)")

let engine = MLXServeEngine()
if let modelsRoot { await engine.useModelStore(ModelStore(root: modelsRoot)) }
print("[engine] device budget: \(gb(await engine.memory.budgetBytes)) | available \(gb(await engine.memory.availableBytes))")

do {
    // 1. Register — the two-layer license gate (C7 MIT weights + C8 MIT port) + device eligibility
    //    (C10) run here, before any instance exists. `.permissiveOnly` admits (both MIT).
    let cfg = SkinTokensConfiguration(quant: quant, skinOnly: mode == MeshRigContract.skinOnly, cls: cls,
                                      weightsRootOverride: weightsOverride)
    let pkgID = try await engine.register(SkinTokensPackage.registration, configuration: cfg)
    print("[engine] registered packageID=\(pkgID.description) | backers(meshRig)=\(await engine.packages(for: .meshRig).map { $0.description })")

    // 2. Prepare — engine constructs (C13) + load()s the weights (memory admission applies). Timed.
    let tLoad = Date()
    try await engine.prepare(.meshRig, package: pkgID)
    let loadS = -tLoad.timeIntervalSinceNow
    let afterLoad = await engine.memory
    print(String(format: "[engine] prepare OK in %.1fs | charged %@ | resident %@ / budget %@",
                 loadS, gb(afterLoad.residents[.meshRig] ?? 0), gb(afterLoad.residentBytes), gb(afterLoad.budgetBytes)))

    // 3. Run — the coordinator drives run() on the InferenceActor. Timed.
    let req = MeshRigRequest(mesh: Mesh(format: .glb, data: glbData), mode: mode)
    let tRun = Date()
    let response = try await engine.run(req, package: pkgID)
    let runS = -tRun.timeIntervalSinceNow
    guard let resp = response as? MeshRigResponse else { fatalError("unexpected response type") }
    try resp.mesh.data.write(to: URL(fileURLWithPath: outGLB))
    print(String(format: "[engine] run OK in %.1fs → wrote %@ (%d bytes)", runS, outGLB, resp.mesh.data.count))
    print("[engine] ✅ meshRig ran through MLXServeEngine (register → prepare → run).")
} catch {
    print("[engine] FAIL: \(error)")
    exit(1)
}
