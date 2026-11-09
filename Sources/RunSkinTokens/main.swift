import Foundation
import MLX
import SkinTokensCore
import SkinTokensRig

// RunSkinTokens — the parity-gate + GPU-e2e harness for the SkinTokens Swift port.
//
// Gates run as `--sNN-gate` subcommands here (NOT `swift test` — the SPM test-target metallib is
// unreliable for GPU; swift-port-parity.md "Where gates can actually run"). Heavy gates: build
// `-c release`, launch detached (`nohup … & disown`), progress markers to stderr.
//
//   swift run RunSkinTokens --version
//   swift run RunSkinTokens --npy <path.npy>                 probe the shared fixture reader
//   swift run RunSkinTokens --s0-headers <file.safetensors>  dump safetensors header (key: dtype shape)
//   swift run RunSkinTokens --s0-gate <converted-dir>        S0 refuse-partial-load key diff (both variants)

setvbuf(stdout, nil, _IONBF, 0)   // unbuffered: stream progress to the detached log in real time

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

let args = Array(CommandLine.arguments.dropFirst())
let mode = args.first ?? "--help"

func requireArg(_ i: Int, _ what: String) -> String {
    guard args.count > i else { err("[skintokens] missing \(what)"); exit(2) }
    return args[i]
}

switch mode {
case "--version":
    print("RunSkinTokens — SkinTokensCore \(SkinTokensCore.version)")

case "--dump-template":
    // Harvest a canonical skeleton (joints + parents + VRM humanoid names) from a rigged GLB/VRM,
    // to bundle as the Route-B template fitted onto bare generated meshes. Pure GLB parse — no weights/MLX.
    //   swift run RunSkinTokens --dump-template <in.glb> <out.json>
    let inGLB = requireArg(1, "<in.glb>")
    let outJSON = requireArg(2, "<out.json>")
    do {
        let (_, skel, _) = try VRMLoad.load(inGLB)
        let obj: [String: Any] = [
            "J": skel.J,
            "joints": skel.joints.map { $0.map { Double($0) } },  // J×3 world/bind positions
            "parents": skel.parents,                               // parent joint index (-1 = root)
            "humanoid": skel.humanoid,                             // VRM bone name → joint index
            "bindWorld": skel.bindWorld,                           // J×16 world bind matrices (ROTATION-AWARE)
        ]
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outJSON))
        print("[dump-template] J=\(skel.J) humanoidBones=\(skel.humanoid.count) → \(outJSON)")
    } catch { err("[dump-template] FAIL: \(error)"); exit(1) }

case "--npy":
    let path = requireArg(1, "<path.npy>")
    do {
        let a = try NPYReader.load(path)
        eval(a)
        print("[npy] \(path)")
        print("[npy] shape=\(a.shape) dtype=\(a.dtype) count=\(a.size)")
        // A cheap distribution fingerprint (mean/min/max) — enough to eyeball a capture load.
        if a.dtype == .int32 {
            let v = a.asArray(Int32.self)
            print("[npy] int32 min=\(v.min() ?? 0) max=\(v.max() ?? 0)")
        } else {
            let f = a.asType(.float32).asArray(Float.self)
            let m = f.reduce(0, +) / Float(max(f.count, 1))
            print("[npy] mean=\(m) min=\(f.min() ?? .nan) max=\(f.max() ?? .nan)")
        }
    } catch { err("[npy] FAIL: \(error)"); exit(1) }

case "--s0-headers":
    let path = requireArg(1, "<file.safetensors>")
    do {
        let info = try SafetensorsHeader.read(path)
        print("[s0] \(path) — \(info.count) tensors")
        for key in info.keys.sorted() {
            let t = info[key]!
            print("  \(key): \(t.dtype) \(t.shape)")
        }
    } catch { err("[s0] FAIL: \(error)"); exit(1) }

case "--s0-gate":
    // S0 refuse-partial-load: generated per-component key sets vs the actual converted headers, for
    // BOTH variants (0 missing / 0 unused each). Pass either a directory containing
    // tokenrig.safetensors + skinvae.safetensors, or a single tokenrig.safetensors file.
    let arg = requireArg(1, "<converted-dir | tokenrig.safetensors>")
    var isDir: ObjCBool = false
    let fm = FileManager.default
    let exists = fm.fileExists(atPath: arg, isDirectory: &isDir)
    guard exists else { err("[s0] not found: \(arg)"); exit(2) }
    let tokenrigPath = isDir.boolValue ? (arg as NSString).appendingPathComponent("tokenrig.safetensors") : arg
    let skinvaePath  = isDir.boolValue ? (arg as NSString).appendingPathComponent("skinvae.safetensors")  : nil
    func gateFile(_ label: String, _ path: String, _ expected: Set<String>) -> Bool {
        do {
            let onDisk = try SafetensorsHeader.keys(path)
            let d = KeyContract.diff(expected: expected, onDisk: onDisk)
            print("[s0] \(label): expected=\(expected.count) onDisk=\(onDisk.count) — \(d)")
            return d.isClean
        } catch { err("[s0] \(label) FAIL: \(error)"); return false }
    }
    var ok = gateFile("tokenrig.safetensors", tokenrigPath, KeyContract.tokenRigKeys)
    if let sv = skinvaePath, fm.fileExists(atPath: sv) {
        ok = gateFile("skinvae.safetensors", sv, KeyContract.skinVAEStandaloneKeys) && ok
    } else if isDir.boolValue {
        err("[s0] NOTE: skinvae.safetensors not present in \(arg) — gated tokenrig only.")
    }
    print("[s0] S0 \(ok ? "PASSED ✅" : "FAILED ❌")")
    exit(ok ? 0 : 1)

case "--s1b-gate":
    // S1b — Michelangelo point encoder parity. INJECT captured FPS query points/feats + full
    // sampled cloud → learned_mesh_cond (1,512,896), cos ≥ 0.999 vs s1b_learned_mesh_cond.npy.
    // Runs entirely on the CPU stream (explicit attention, no MLXFast SDPA) to match the CPU fixture.
    //   swift run -c release RunSkinTokens --s1b-gate <converted-dir> <capture-dir>
    let conv = requireArg(1, "<converted-dir>")
    let cap  = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    var ok = false
    do {
        try Device.withDefaultDevice(.cpu) {
            err("[s1b] loading weights \(tokenrig)")
            let w = try Weights.load(tokenrig)
            func npy(_ n: String) throws -> MLXArray {
                try NPYReader.load((cap as NSString).appendingPathComponent(n))
            }
            let queryPc    = try npy("s1b_query_pc.npy").expandedDimensions(axis: 0)     // (1,512,3)
            let queryFeats = try npy("s1b_query_feats.npy").expandedDimensions(axis: 0)  // (1,512,3)
            let kvPc       = try npy("sampled_vertices.npy").expandedDimensions(axis: 0) // (1,54000,3)
            let kvFeats    = try npy("sampled_normals.npy").expandedDimensions(axis: 0)
            let target     = try npy("s1b_learned_mesh_cond.npy")                        // (1,512,896)
            err("[s1b] shapes q=\(queryPc.shape) kv=\(kvPc.shape) target=\(target.shape) — encoding (bf16)")
            let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }
            let latents = Michelangelo.encode(kvPc: bf(kvPc), kvFeats: bf(kvFeats),
                                              queryPc: bf(queryPc), queryFeats: bf(queryFeats),
                                              weights: w)
            let lmc = Michelangelo.outputProj(latents, weights: w)
            eval(lmc)
            let cos = SkinOps.cosine(lmc, target)
            let mad = SkinOps.maxAbs(lmc, target)
            print("[s1b] learned_mesh_cond \(lmc.shape) — cos=\(String(format: "%.6f", cos)) maxAbs=\(String(format: "%.4e", mad))")
            ok = cos >= 0.999
            print("[s1b] S1b \(ok ? "PASSED ✅ (cos ≥ 0.999)" : "FAILED ❌ (cos < 0.999)")")
        }
    } catch { err("[s1b] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--s1c-gate":
    // S1c — SkinVAE cond_encoder parity. INJECT captured 384 cond-query pts + full cloud →
    // cond_latents (1,384,512), cos ≥ 0.999 vs s1c_cond_latents.npy. CPU stream.
    let conv = requireArg(1, "<converted-dir>")
    let cap  = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    var ok = false
    do {
        try Device.withDefaultDevice(.cpu) {
            let w = try Weights.load(tokenrig)
            func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
            let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }
            let condQ  = bf(try npy("s1c_cond_q.npy").expandedDimensions(axis: 0))        // (1,384,6)
            let verts  = try npy("sampled_vertices.npy")
            let norms  = try npy("sampled_normals.npy")
            let condKv = bf(concatenated([verts, norms], axis: -1).expandedDimensions(axis: 0)) // (1,54000,6)
            let target = try npy("s1c_cond_latents.npy")
            err("[s1c] condQ=\(condQ.shape) condKv=\(condKv.shape) target=\(target.shape) — encoding (bf16)")
            let cl = SkinVAE.condEncode(condQ: SkinVAE.qkvInput(condQ), condKv: SkinVAE.qkvInput(condKv), weights: w)
            eval(cl)
            let cos = SkinOps.cosine(cl, target), mad = SkinOps.maxAbs(cl, target)
            print("[s1c] cond_latents \(cl.shape) — cos=\(String(format: "%.6f", cos)) maxAbs=\(String(format: "%.4e", mad))")
            ok = cos >= 0.999
            print("[s1c] S1c \(ok ? "PASSED ✅" : "FAILED ❌")")
        }
    } catch { err("[s1c] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--s1c-bisect":
    let conv = requireArg(1, "<converted-dir>"); let cap = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    do { try Device.withDefaultDevice(.cpu) {
        let w = try Weights.load(tokenrig)
        func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
        let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }
        let condQpts = bf(try npy("s1c_cond_q.npy").expandedDimensions(axis: 0))
        let verts = try npy("sampled_vertices.npy"); let norms = try npy("sampled_normals.npy")
        let condKvPts = bf(concatenated([verts, norms], axis: -1).expandedDimensions(axis: 0))
        let condQ = SkinVAE.qkvInput(condQpts)           // (1,384,54)
        let condKv = SkinVAE.qkvInput(condKvPts)
        func chk(_ label: String, _ got: MLXArray, _ file: String) throws {
            let t = try npy(file); eval(got)
            print("[bisect] \(label): cos=\(String(format: "%.6f", SkinOps.cosine(got, t))) maxAbs=\(String(format: "%.4e", SkinOps.maxAbs(got, t)))")
        }
        try chk("condq(embed)", condQ, "s1c_dbg_condq.npy")
        let ce = "vae.model.cond_encoder"
        var hs = SkinOps.linear(condQ, w["\(ce).proj_in.weight"]!, w["\(ce).proj_in.bias"]!)
        let enc = SkinOps.linear(condKv, w["\(ce).proj_in.weight"]!, w["\(ce).proj_in.bias"]!)
        try chk("proj_in", hs, "s1c_dbg_projin.npy")
        hs = SkinVAE.crossBlock(hs, enc, w, "\(ce).blocks.0"); try chk("block0(cross)", hs, "s1c_dbg_b0.npy")
        hs = SkinVAE.selfBlock(hs, w, "\(ce).blocks.1"); try chk("block1(self)", hs, "s1c_dbg_b1.npy")
        hs = SkinVAE.selfBlock(hs, w, "\(ce).blocks.2"); try chk("block2(self)", hs, "s1c_dbg_b2.npy")
        hs = SkinOps.layerNorm(hs, w["\(ce).norm_out.weight"]!, w["\(ce).norm_out.bias"]!); try chk("norm_out", hs, "s1c_dbg_norm.npy")
    } } catch { err("[bisect] FAIL: \(error)"); exit(1) }

case "--s1d-gate":
    // S1d — FSQ lookup + chunked decoder parity. FEED captured fsq_indices/decoder_z + injected
    // cond_latents + sampled cloud → decoder logits per chunk (cos ≥ 0.999 vs decoder_logits_chunkN),
    // and assembled skin_pred (54000,J) cos ≥ 0.999. Deterministic (no RNG). CPU stream.
    let conv = requireArg(1, "<converted-dir>")
    let cap  = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    var ok = false
    do {
        try Device.withDefaultDevice(.cpu) {
            let w = try Weights.load(tokenrig)
            func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
            let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }
            let fsqIdx = try npy("fsq_indices.npy")                        // (260,) int
            let fsqCodes = try npy("fsq_codes.npy")                        // (260,512) target
            let condLat = bf(try npy("cond_latents.npy"))                  // (1,384,512) injected (P0 capture)
            let verts = try npy("sampled_vertices.npy")
            let norms = try npy("sampled_normals.npy")
            let sampledCond = bf(concatenated([verts, norms], axis: -1).expandedDimensions(axis: 0)) // (1,54000,6)
            let skinTarget = try npy("skin_pred.npy")                     // (54000, J)
            let J = skinTarget.shape[1]

            // (a) FSQ lookup: indices → codes vs fsq_codes
            let codes = SkinVAE.fsqIndicesToCodes(fsqIdx, weights: w)     // (260,512)
            eval(codes)
            let fsqCos = SkinOps.cosine(codes, fsqCodes)
            print("[s1d] FSQ indices→codes \(codes.shape) — cos=\(String(format: "%.6f", fsqCos)) vs fsq_codes")

            // (b) per-chunk decoder + assemble skin_pred. z_j = codes[j*4:(j+1)*4] reshaped (1,4,512).
            let codesBf = codes.asType(.bfloat16)
            var cols: [MLXArray] = []
            var chunkCosSum: Float = 0, chunkCosMin: Float = 1
            for j in 0..<J {
                let z = codesBf[(j*4)..<((j+1)*4), 0...].reshaped([1, 4, 512])
                let logits = SkinVAE.decode(z: z, cond: condLat, sampledCond: sampledCond, weights: w) // (1,54000,1)
                eval(logits)
                if j < 3 {
                    let tgt = try npy("decoder_logits_chunk\(j).npy")
                    let c = SkinOps.cosine(logits, tgt)
                    chunkCosSum += c; chunkCosMin = min(chunkCosMin, c)
                    err("[s1d]   chunk \(j) decoder logits cos=\(String(format: "%.6f", c))")
                }
                cols.append(logits.reshaped([54000, 1]))
            }
            let skinPred = concatenated(cols, axis: 1)                    // (54000, J)
            eval(skinPred)
            let skinCos = SkinOps.cosine(skinPred, skinTarget)
            let skinMad = SkinOps.maxAbs(skinPred, skinTarget)
            print("[s1d] skin_pred \(skinPred.shape) — cos=\(String(format: "%.6f", skinCos)) maxAbs=\(String(format: "%.4e", skinMad))")
            ok = fsqCos >= 0.999 && skinCos >= 0.999
            print("[s1d] S1d \(ok ? "PASSED ✅ (FSQ + skin_pred cos ≥ 0.999)" : "FAILED ❌")")
        }
    } catch { err("[s1d] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--s2-bisect":
    let conv = requireArg(1, "<converted-dir>"); let cap = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    do { try Device.withDefaultDevice(.cpu) {
        let w = try Weights.load(tokenrig)
        func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
        let meshCond = try npy("learned_mesh_cond.npy")
        let ids = try npy("output_ids.npy").asType(.int32).asArray(Int32.self)
        let L = ids.count
        let tokEmb = Qwen3.embedTokens(Array(ids[0..<(L - 1)]), weights: w)
        let inputs = concatenated([meshCond.asType(.float32), tokEmb], axis: 1)
        // layer-0 only
        let h0 = Qwen3.forward(inputsEmbeds: inputs, weights: w, config: { var c = Qwen3.Config(); c.layers = 1; return c }(), returnHidden: true)
        eval(h0)
        let h0t = try npy("s2_dbg_h0.npy").expandedDimensions(axis: 0)
        print("[s2b] layer0 hidden \(h0.shape) cos=\(String(format: "%.6f", SkinOps.cosine(h0, h0t))) maxAbs=\(String(format: "%.3e", SkinOps.maxAbs(h0, h0t)))")
        // full logits
        let logits = Qwen3.forward(inputsEmbeds: inputs, weights: w)
        eval(logits)
        let lt = try npy("s2_dbg_logits.npy").expandedDimensions(axis: 0)
        print("[s2b] full logits \(logits.shape) cos=\(String(format: "%.6f", SkinOps.cosine(logits, lt))) maxAbs=\(String(format: "%.3e", SkinOps.maxAbs(logits, lt)))")
        // per-position cos at a few positions
        for pos in [513, 600, 800, 1008] {
            let a = logits[0, pos, 0...]; let b = lt[0, pos, 0...]
            err("[s2b]  pos \(pos) cos=\(String(format: "%.6f", SkinOps.cosine(a, b)))")
        }
    } } catch { err("[s2b] FAIL: \(error)"); exit(1) }

case "--s2b-kv-check":
    // S2b prep — validate the incremental KV cache against the (S2-PASSED) teacher-forced forward.
    // Feed captured learned_mesh_cond + output_ids: prefill the 512 mesh soft-prefix, then STEP the
    // token embeds one at a time; each step's logits must match the full-forward logits at that
    // position (cos ~1.0). Proves the cache + RoPE offset + beam-axis batching are correct. CPU stream.
    let conv = requireArg(1, "<converted-dir>")
    let cap  = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    var ok = false
    do {
        try Device.withDefaultDevice(.cpu) {
            let w = try Weights.load(tokenrig)
            func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
            let meshCond = try npy("learned_mesh_cond.npy").asType(.float32)   // (1,512,896)
            let ids = try npy("output_ids.npy").asType(.int32).asArray(Int32.self)
            let L = ids.count
            let tokEmb = Qwen3.embedTokens(Array(ids[0..<(L - 1)]), weights: w)  // (1,L-1,896)
            let inputs = concatenated([meshCond, tokEmb], axis: 1)              // (1,512+L-1,896)
            err("[s2b-kv] reference full forward (T=\(inputs.shape[1]))")
            let ref = Qwen3.forward(inputsEmbeds: inputs, weights: w)           // (1,T,V)
            eval(ref)
            // Incremental: prefill mesh, then step each token embed.
            let dec = Qwen3Decoder(weights: w)
            let pfx = meshCond.shape[1]
            _ = dec.prefill(meshCond)                                           // positions 0..511
            var stepLogits: [MLXArray] = []
            let nSteps = tokEmb.shape[1]
            for i in 0..<nSteps {
                let e = tokEmb[0..., i..<(i + 1), 0...]                         // (1,1,896)
                let lg = dec.step(e)                                            // (1,1,V)
                if i % 100 == 0 { err("[s2b-kv]   step \(i)/\(nSteps) offset=\(dec.offset)") }
                stepLogits.append(lg)
            }
            let stepped = concatenated(stepLogits, axis: 1)                     // (1,nSteps,V)
            eval(stepped)
            // stepped[i] is the logit AFTER consuming token-embed i, i.e. reference position pfx+i.
            let refSlice = ref[0..., pfx..<(pfx + nSteps), 0...]
            let cos = SkinOps.cosine(stepped, refSlice)
            let mad = SkinOps.maxAbs(stepped, refSlice)
            let am1 = argMax(stepped[0, 0..., 0...], axis: -1)
            let am2 = argMax(refSlice[0, 0..., 0...], axis: -1)
            eval(am1, am2)
            let agree = (am1 .== am2).asType(.int32).sum().item(Int32.self)
            print("[s2b-kv] stepped vs teacher-forced — cos=\(String(format: "%.6f", cos)) maxAbs=\(String(format: "%.3e", mad)) argmaxAgree=\(agree)/\(nSteps)")
            ok = cos >= 0.9999
            print("[s2b-kv] KV-cache check \(ok ? "PASSED ✅ (cos ≥ 0.9999)" : "FAILED ❌")")
        }
    } catch { err("[s2b-kv] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--s2-gate":
    // S2 — TokenRig transformer (Qwen3-0.6B, un-tied head) teacher-forced. INJECT learned_mesh_cond
    // (512 soft prefix) + output_ids → per-step logits (N,33036), cos ≥ 0.999 vs step_logits.npy.
    let conv = requireArg(1, "<converted-dir>")
    let cap  = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    var ok = false
    do {
        try Device.withDefaultDevice(.cpu) {
            let w = try Weights.load(tokenrig)
            func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
            let meshCond = try npy("learned_mesh_cond.npy")            // (1,512,896) — P0 capture (soft prefix)
            let ids = try npy("output_ids.npy").asType(.int32).asArray(Int32.self)  // (L,)
            let teacher = try npy("s2_teacher_logits.npy")            // (T,33036) PyTorch teacher-forced (deterministic target)
            let L = ids.count
            err("[s2] L=\(L) meshCond=\(meshCond.shape) — building inputs_embeds")
            // inputs_embeds = [mesh_cond(512), embed_tokens(ids[0:L-1])] — a single deterministic pass.
            let tokEmb = Qwen3.embedTokens(Array(ids[0..<(L - 1)]), weights: w)  // (1,L-1,896)
            let inputs = concatenated([meshCond.asType(.float32), tokEmb], axis: 1)  // (1, 512+L-1, 896)
            err("[s2] inputs_embeds \(inputs.shape) — running 28-layer forward (fp32, CPU)")
            let logits = Qwen3.forward(inputsEmbeds: inputs, weights: w)  // (1, 512+L-1, 33036)
            eval(logits)
            // GATE: full teacher-forced logits vs the PyTorch teacher-forced pass (both single-pass,
            // same inputs_embeds — the deterministic stage). argmax agreement over the generated slice.
            let l2 = logits[0, 0..., 0...]
            let cos = SkinOps.cosine(l2, teacher)
            let mad = SkinOps.maxAbs(l2, teacher)
            let am1 = argMax(l2, axis: -1); let am2 = argMax(teacher.asType(.float32), axis: -1)
            eval(am1, am2)
            let agree = (am1 .== am2).asType(.int32).sum().item(Int32.self)
            print("[s2] teacher logits \(l2.shape) — cos=\(String(format: "%.6f", cos)) maxAbs=\(String(format: "%.4e", mad)) argmaxAgree=\(agree)/\(l2.shape[0])")
            // INFORMATIONAL: vs the autoregressive step_logits.npy — expected ~0.991 (bf16 KV-cache
            // artifact; the oracle's OWN teacher-forced pass matches it only that well). NOT the gate.
            if let sl = try? npy("step_logits.npy") {
                let S = sl.shape[0], base = meshCond.shape[1] + (L - S) - 1
                let slc = logits[0, base..<(base + S), 0...]
                print("[s2] (info) vs autoregressive step_logits: cos=\(String(format: "%.6f", SkinOps.cosine(slc, sl))) — cache-limited, not the gate")
            }
            ok = cos >= 0.999
            print("[s2] S2 \(ok ? "PASSED ✅ (teacher-forced cos ≥ 0.999)" : "FAILED ❌")")
        }
    } catch { err("[s2] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--detok-check":
    // S4-prep — validate SkinDetokenize against the oracle's captured joints/parents by feeding the
    // captured output_ids: parents must match EXACTLY, joint coords to the 256-bin undiscretize floor.
    let cap = requireArg(1, "<capture-dir>")
    var ok = false
    do {
        func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
        let ids = try npy("output_ids.npy").asType(.int32).asArray(Int32.self).map { Int($0) }
        let jTarget = try npy("joints.npy").asType(.float32)                    // (J,3)
        let pTarget = try npy("parents.npy").asType(.int32).asArray(Int32.self) // (J,)
        let skel = try SkinDetokenize.detokenize(ids)
        let jFlat = jTarget.asArray(Float.self)
        let Jt = pTarget.count
        var maxCoord: Float = 0
        for i in 0..<min(skel.J, Jt) {
            for k in 0..<3 { maxCoord = max(maxCoord, abs(skel.joints[i][k] - jFlat[i * 3 + k])) }
        }
        let parentsMatch = skel.parents == pTarget.map { Int($0) }
        print("[detok] J: got=\(skel.J) target=\(Jt) — parentsExact=\(parentsMatch) maxCoordΔ=\(String(format: "%.5f", maxCoord)) (256-bin floor≈0.004)")
        ok = skel.J == Jt && parentsMatch && maxCoord < 0.01
        print("[detok] detokenize \(ok ? "PASSED ✅" : "FAILED ❌")")
    } catch { err("[detok] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--tokenize-check":
    // S5-prep — validate SkinDetokenize.tokenize against detokenize (roundtrip on captured skeletons):
    // tokenize(joints,parents) → detokenize → parents EXACT, coords to 256-bin floor, 0 grammar viol.
    let cap = requireArg(1, "<capture-dir>")
    var ok = false
    do {
        func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
        let jFlat = try npy("joints.npy").asType(.float32).asArray(Float.self)
        let pTarget = try npy("parents.npy").asType(.int32).asArray(Int32.self).map { Int($0) }
        let J = pTarget.count
        var joints = [[Float]](); for i in 0..<J { joints.append([jFlat[i*3], jFlat[i*3+1], jFlat[i*3+2]]) }
        let L = SkinTokenGrammar.skinTokens
        let toks = SkinDetokenize.tokenize(joints: joints, parents: pTarget, cls: L.clsTokens[2], layout: L)  // articulation
        let viol = SkinGen.countViolations(toks, tokensPerSkin: 4, layout: L)
        let skel = try SkinDetokenize.detokenize(toks, layout: L)
        var maxΔ: Float = 0
        for i in 0..<min(skel.J, J) { for k in 0..<3 { maxΔ = max(maxΔ, abs(skel.joints[i][k] - joints[i][k])) } }
        let parentsMatch = skel.parents == pTarget
        print("[tok] J: in=\(J) tokenize→detok=\(skel.J) — parentsExact=\(parentsMatch) maxCoordΔ=\(String(format: "%.5f", maxΔ)) grammarViol=\(viol) streamLen=\(toks.count)")
        ok = skel.J == J && parentsMatch && maxΔ < 0.01 && viol == 0
        print("[tok] tokenize \(ok ? "PASSED ✅" : "FAILED ❌")")
    } catch { err("[tok] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--s2b-gate":
    // S2b(3) — the grammar-constrained batched-KV beam decoder. INJECT captured learned_mesh_cond
    // (512 mesh soft prefix) + start tokens [bos, cls], run SkinGen.generate, then:
    //   (a) assert 0 grammar violations on the emitted stream (guaranteed by the DFA mask),
    //   (b) decode the emitted skin tokens through S1d (FSQ→decoder) → skin_pred rowsum≈1, 0 unweighted,
    //   (c) print aggregate stats (J, skeleton/skin lengths) for a plausibility sanity vs P0 beam10.
    // Runs on the GPU stream by default (sustained generation); pass --cpu for the watchdog-safe path.
    //   swift run -c release RunSkinTokens --s2b-gate <conv> <cap> [--beams N] [--sample] [--max N] [--cpu] [--no-skin]
    let conv = requireArg(1, "<converted-dir>")
    let cap  = requireArg(2, "<capture-dir>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    func flag(_ n: String) -> Bool { args.contains(n) }
    func opt(_ n: String) -> String? { args.firstIndex(of: n).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    var p = SkinGen.Params()
    p.numBeams = opt("--beams").flatMap { Int($0) } ?? 10
    p.doSample = flag("--sample")
    p.maxNewTokens = opt("--max").flatMap { Int($0) } ?? 2040
    p.seed = opt("--seed").flatMap { UInt64($0) } ?? 0
    let useCPU = flag("--cpu")
    let decodeSkin = !flag("--no-skin")
    var ok = false
    let L = SkinTokenGrammar.skinTokens
    func run() throws {
        let w = try Weights.load(tokenrig)
        func npy(_ n: String) throws -> MLXArray { try NPYReader.load((cap as NSString).appendingPathComponent(n)) }
        let meshCond = try npy("learned_mesh_cond.npy").asType(.float32)          // (1,512,896)
        let outIds = try npy("output_ids.npy").asType(.int32).asArray(Int32.self)
        let startTokens = [Int(outIds[0]), Int(outIds[1])]                        // [bos, cls] (init)
        err("[s2b] start=\(startTokens) beams=\(p.numBeams) sample=\(p.doSample) max=\(p.maxNewTokens) stream=\(useCPU ? "cpu" : "gpu")")
        let dec = Qwen3Decoder(weights: w)
        let t0 = Date()
        let res = SkinGen.generate(decoder: dec, meshCond: meshCond, startTokens: startTokens,
                                   layout: L, tokensPerSkin: 4, params: p) { step, beams, done in
            if step % 50 == 0 { err("[s2b]   step \(step) beams=\(beams) completed=\(done)") }
        }
        let dt = Date().timeIntervalSince(t0)
        let ids = res.outputIds
        // (a) grammar validity
        let switchIdx = ids.firstIndex(of: L.switchTokenId)
        let J = (try? SkinTokenGrammar.bonesInSequence(ids, layout: L)) ?? -1
        let skelLen = (switchIdx ?? -1) + 1
        let skinLen = switchIdx.map { ids.count - $0 - 1 } ?? 0
        // detokenize → joints/parents for the P0-beam10 plausibility sanity (J, tree depth, symmetry).
        if let skel = try? SkinDetokenize.detokenize(ids, layout: L) {
            let (xlo, xhi) = skel.xBounds
            print("[s2b] skeleton — J=\(skel.J) depth=\(skel.treeDepth) x∈[\(String(format: "%.3f", xlo)),\(String(format: "%.3f", xhi))] (P0 beam10 giraffe ≈ J72 depth14 symmetric)")
        }
        print("[s2b] generated \(ids.count) tokens in \(String(format: "%.1f", dt))s — J=\(J) skeletonLen=\(skelLen) skinLen=\(skinLen) score=\(String(format: "%.3f", res.score))")
        print("[s2b] grammar violations = \(res.violations)  (must be 0)")
        // (b) decode skin through S1d
        var rowsumOK = true, unweighted = -1
        if decodeSkin, let sw = switchIdx, J > 0 {
            let skinToks = Array(ids[(sw + 1)..<min(sw + 1 + J * 4, ids.count)])
            if skinToks.count == J * 4 {
                let idxInts = skinToks.map { Int32($0 - L.tokenizerVocabSize) }   // → FSQ indices (last = 32768)
                let fsqIdx = MLXArray(idxInts)
                let codes = SkinVAE.fsqIndicesToCodes(fsqIdx, weights: w)          // (J*4,512)
                let condLat = (try npy("cond_latents.npy")).asType(.bfloat16)      // injected (1,384,512)
                let verts = try npy("sampled_vertices.npy"); let norms = try npy("sampled_normals.npy")
                let sampledCond = concatenated([verts, norms], axis: -1).expandedDimensions(axis: 0).asType(.bfloat16)
                let codesBf = codes.asType(.bfloat16)
                var cols: [MLXArray] = []
                for j in 0..<J {
                    let z = codesBf[(j * 4)..<((j + 1) * 4), 0...].reshaped([1, 4, 512])
                    let lg = SkinVAE.decode(z: z, cond: condLat, sampledCond: sampledCond, weights: w) // (1,54000,1)
                    eval(lg); cols.append(lg.reshaped([54000, 1]))
                    if !useCPU { MLX.Memory.clearCache() }
                }
                // `SkinVAE.decode` output IS the per-joint sigmoid skin (∈[0,1], per-joint independent —
                // NOT yet rowsum-normalized; normalize_skin runs at assembly/S4). The meaningful gate on
                // the raw decode: every point has ≥1 nonzero influence (0 unweighted rows), so the later
                // per-row normalization is safe and gives rowsum≈1 by construction.
                let skin = concatenated(cols, axis: 1).asType(.float32)             // (54000,J) sigmoid
                let rawRowsum = skin.sum(axis: 1); eval(rawRowsum)
                let rs = rawRowsum.asArray(Float.self)
                let minRS = rs.min() ?? 0, maxRS = rs.max() ?? 0
                let sMin = skin.min().item(Float.self), sMax = skin.max().item(Float.self)
                let maxInf = skin.max(axis: 1); eval(maxInf)
                unweighted = Int((maxInf .< 1e-6).asType(.int32).sum().item(Int32.self))
                rowsumOK = sMin >= -1e-4 && sMax <= 1.0 + 1e-4 && minRS > 1e-4      // ∈[0,1], no all-zero row
                print("[s2b] skin_pred (54000,\(J)) — sigmoid∈[\(String(format: "%.4f", sMin)),\(String(format: "%.4f", sMax))] rawRowsum[min=\(String(format: "%.4f", minRS)) max=\(String(format: "%.4f", maxRS))] unweighted=\(unweighted)")
            } else {
                print("[s2b] ⚠ skin token count \(skinToks.count) != J*4=\(J*4) — skipping skin decode")
            }
        }
        ok = res.violations == 0 && J > 0 && (!decodeSkin || (rowsumOK && unweighted == 0))
        print("[s2b] S2b(3) \(ok ? "PASSED ✅ (0 violations, plausible rig)" : "FAILED ❌")")
    }
    do {
        if useCPU { try Device.withDefaultDevice(.cpu) { try run() } } else { try run() }
    } catch { err("[s2b] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--route-b":
    // ⭐ S5 — Route-B skin-only: feed the VRM's OWN skeleton as the tokenized prefix (cls=vroid) →
    // beam generates SKIN ONLY (grammar starts in skin phase) → decode → propagate → write
    // WEIGHTS_0/JOINTS_0 onto the EXISTING joints (rotation-aware IBMs, no new skeleton). This is the
    // companion-character path. Gate: J-in==J-out, 0 violations, 0 unweighted, roundtrip analyze_rig.py.
    //   swift run -c release RunSkinTokens --route-b <conv> <avatar.vrm> [--out rigged.glb] [--beams N] [--cls ID]
    let conv = requireArg(1, "<converted-dir>")
    let vrm  = requireArg(2, "<avatar.vrm|.glb>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    func opt3(_ n: String) -> String? { args.firstIndex(of: n).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    let L = SkinTokenGrammar.skinTokens
    let beams = opt3("--beams").flatMap { Int($0) } ?? 10
    let clsTok = opt3("--cls").flatMap { Int($0) } ?? L.clsTokens[1]   // vroid
    var ok = false
    do {
        err("[rb] loading weights + VRM")
        let w = try Weights.load(tokenrig)
        let (mesh, skel, _) = try VRMLoad.load(vrm)
        // combined affine over verts AND joints (AugmentAffine bounds both).
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3), hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for i in 0..<mesh.vertexCount { for k in 0..<3 { let x = mesh.vertices[i*3+k]; lo[k] = min(lo[k], x); hi[k] = max(hi[k], x) } }
        for j in skel.joints { for k in 0..<3 { lo[k] = min(lo[k], j[k]); hi[k] = max(hi[k], j[k]) } }
        let center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]
        let scaleV = max(hi[0]-lo[0], max(hi[1]-lo[1], hi[2]-lo[2])) / 2
        let aff = MeshIO.Affine(center: center, scale: scaleV == 0 ? 1 : scaleV)
        let normMesh = Mesh(vertices: MeshIO.applyAffine(mesh.vertices, aff), faces: mesh.faces, normals: mesh.normals, primRanges: mesh.primRanges)
        let normJoints = skel.joints.map { j in (0..<3).map { (j[$0] - aff.center[$0]) / aff.scale } }
        let cloud = MeshIO.samplerMix(normMesh, seed: 0)
        err("[rb] VRM verts=\(mesh.vertexCount) J=\(skel.J); encoders")
        let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }
        let kvPc = MLXArray(cloud.points, [1, cloud.count, 3]); let kvFeats = MLXArray(cloud.normals, [1, cloud.count, 3])
        let qIdx = MeshIO.fpsQuery(points: cloud.points, count: cloud.count, subset: 2048, k: 512, seed: 0)
        var qp = [Float](repeating: 0, count: 512*3), qf = [Float](repeating: 0, count: 512*3)
        for (i, idx) in qIdx.enumerated() { for k in 0..<3 { qp[i*3+k] = cloud.points[idx*3+k]; qf[i*3+k] = cloud.normals[idx*3+k] } }
        let latents = Michelangelo.encode(kvPc: bf(kvPc), kvFeats: bf(kvFeats), queryPc: bf(MLXArray(qp, [1,512,3])), queryFeats: bf(MLXArray(qf, [1,512,3])), weights: w)
        let meshCond = Michelangelo.outputProj(latents, weights: w).asType(.float32); eval(meshCond)
        let cIdx = MeshIO.fpsQuery(points: cloud.points, count: cloud.count, subset: 1536, k: 384, seed: 1)
        var cq = [Float](repeating: 0, count: 384*6)
        for (i, idx) in cIdx.enumerated() { for k in 0..<3 { cq[i*6+k] = cloud.points[idx*3+k]; cq[i*6+3+k] = cloud.normals[idx*3+k] } }
        let condKvPts = concatenated([kvPc, kvFeats], axis: -1)
        let condLat = SkinVAE.condEncode(condQ: SkinVAE.qkvInput(bf(MLXArray(cq, [1,384,6]))), condKv: SkinVAE.qkvInput(bf(condKvPts)), weights: w); eval(condLat)
        // OPTIONAL skeleton clean (--clean): the model is trained on bpy-parsed DEFORMING skeletons; a raw
        // VRM skin.joints list (armature Root at 0, *_end leaves, interleaved order) is out-of-distribution →
        // skin binds to wrong bones. Drop Root/*_end, DFS-reorder (parent-before-child → few branches like the
        // reference's 42), remap parents/nodes/IBM. Skin cols then align to the cleaned order; inject uses it.
        func cleanSkelRouteB() -> (joints: [[Float]], parents: [Int], jointNodes: [Int], inverseBind: [Float], rootNode: Int) {
            let ib = skel.inverseBind(); let names = skel.jointNames; let par = skel.parents
            let bipOnly = args.contains("--bip")
            func keep(_ i: Int) -> Bool {
                let n = names[i]
                if bipOnly { return n.hasPrefix("J_Bip_") }   // deforming humanoid bones only (body + hands)
                if n.caseInsensitiveCompare("Root") == .orderedSame { return false }
                if n.contains("_end") { return false }
                return true
            }
            let kept = (0..<skel.J).filter(keep); let keptSet = Set(kept)
            func keptParent(_ i: Int) -> Int { var p = par[i]; while p >= 0 && !keptSet.contains(p) { p = par[p] }; return p }
            var kChildren = [Int: [Int]](); var roots: [Int] = []
            for i in kept { let p = keptParent(i); if p == -1 { roots.append(i) } else { kChildren[p, default: []].append(i) } }
            var order: [Int] = []
            func dfs(_ i: Int) { order.append(i); for c in (kChildren[i] ?? []).sorted() { dfs(c) } }
            for r in roots.sorted() { dfs(r) }
            let origToNew = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
            let oJoints = order.map { normJoints[$0] }
            let oParents = order.map { keptParent($0) == -1 ? -1 : origToNew[keptParent($0)]! }
            let oNodes = order.map { skel.jointNodes[$0] }
            var oIBM = [Float](repeating: 0, count: order.count * 16)
            for (ni, oi) in order.enumerated() { for k in 0..<16 { oIBM[ni*16+k] = ib[oi*16+k] } }
            err("[rb] --clean: \(skel.J) → \(order.count) bones (dropped Root/_end), DFS-reordered")
            return (oJoints, oParents, oNodes, oIBM, oNodes.first ?? skel.rootNode)
        }
        let cleaned = args.contains("--clean") ? cleanSkelRouteB() : nil
        let tokJoints = cleaned?.joints ?? normJoints
        let tokParents = cleaned?.parents ?? skel.parents
        // tokenize the (cleaned) skeleton as the prefix (ends in the switch/eos → skin phase).
        let prefix = SkinDetokenize.tokenize(joints: tokJoints, parents: tokParents, cls: clsTok, layout: L)
        let Jin = (try? SkinTokenGrammar.bonesInSequence(prefix + [L.lmEos], layout: L)) ?? tokJoints.count
        err("[rb] skeleton prefix=\(prefix.count) tok (J=\(Jin)); beam(\(beams)) skin-only decode")
        var pp = SkinGen.Params(); pp.numBeams = beams; pp.maxNewTokens = Jin * 4 + 8
        let dec = Qwen3Decoder(weights: w)
        let res = SkinGen.generate(decoder: dec, meshCond: meshCond, startTokens: prefix, layout: L, tokensPerSkin: 4, params: pp) { s, b, d in
            if s % 50 == 0 { err("[rb]   skin step \(s) beams=\(b)") }
        }
        let ids = res.outputIds
        let Jout = (try? SkinTokenGrammar.bonesInSequence(ids, layout: L)) ?? -1
        print("[rb] generated \(ids.count) tok — J-in=\(Jin) J-out=\(Jout) violations=\(res.violations)")
        // decode skin + propagate + inject.
        var unweighted = -1
        if let sw = ids.firstIndex(of: L.switchTokenId), Jout > 0 {
            let skinToks = Array(ids[(sw+1)..<min(sw+1+Jout*4, ids.count)])
            if skinToks.count == Jout*4 {
                let codes = SkinVAE.fsqIndicesToCodes(MLXArray(skinToks.map { Int32($0 - L.tokenizerVocabSize) }), weights: w).asType(.bfloat16)
                let sampledCond = bf(condKvPts); let clBf = condLat.asType(.bfloat16)
                var cols: [MLXArray] = []
                for j in 0..<Jout {
                    let z = codes[(j*4)..<((j+1)*4), 0...].reshaped([1,4,512])
                    let lg = SkinVAE.decode(z: z, cond: clBf, sampledCond: sampledCond, weights: w)
                    eval(lg); cols.append(lg.reshaped([54000,1])); MLX.Memory.clearCache()
                }
                let skin = concatenated(cols, axis: 1).asType(.float32); eval(skin)
                let mI = skin.max(axis: 1); eval(mI); unweighted = Int((mI .< 1e-6).asType(.int32).sum().item(Int32.self))
                let skHost = skin.asArray(Float.self)
                print("[rb] skin_pred (54000,\(Jout)) — 0-unweighted? \(unweighted == 0)")
                if let outPath = opt3("--out") {
                    err("[rb] propagate → \(mesh.vertexCount) verts; inject (reuse skeleton, rotation-aware IBM)")
                    let vertsNorm = MeshIO.applyAffine(mesh.vertices, aff)
                    let perVert = SkinRig.propagate(sampledPoints: cloud.points, sampledSkin: skHost, J: Jout, queryPts: vertsNorm)
                    let sw4 = SkinRig.topFour(skin: perVert, J: Jout)
                    try SkinRig.injectRigRouteB(glbPath: vrm, outPath: outPath,
                                                jointNodes: cleaned?.jointNodes ?? skel.jointNodes,
                                                inverseBind: cleaned?.inverseBind ?? skel.inverseBind(),
                                                skeletonRoot: cleaned?.rootNode ?? skel.rootNode,
                                                skin: sw4, primRanges: mesh.primRanges)
                    print("[rb] wrote rigged VRM/GLB: \(outPath) (J=\(Jout), \(mesh.vertexCount) verts, \(mesh.primRanges.count) prims)")
                }
            }
        }
        ok = res.violations == 0 && Jout == Jin && Jout > 0 && unweighted == 0
        print("[rb] ROUTE-B \(ok ? "PASSED ✅ (skeleton preserved, skin-only rig)" : "FAILED ❌")")
    } catch { err("[rb] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--template-rig":
    // ⭐ 2b — Route-B on a BARE mesh (no embedded skeleton, e.g. a TRELLIS output): fit the bundled
    // canonical VRoid template → tokenize → skin-only decode → author VRM-named bones. Iterate the
    // up-axis here (TRELLIS orientation varies) before wiring into the app.
    //   swift run -c release RunSkinTokens --template-rig <conv> <mesh.glb> [--out rigged.glb] [--up x|y|z] [--beams N]
    let convT = requireArg(1, "<converted-dir>")
    let meshGLB = requireArg(2, "<mesh.glb>")
    let tokenrigT = (convT as NSString).appendingPathComponent("tokenrig.safetensors")
    func optT(_ n: String) -> String? { args.firstIndex(of: n).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    let upT: VRoidTemplate.UpAxis? = optT("--up").flatMap { s in
        switch s.lowercased() { case "x": return .x; case "y": return .y; case "z": return .z; default: return nil }
    }
    let beamsT = optT("--beams").flatMap { Int($0) } ?? 10
    do {
        err("[tpl] loading weights + mesh (up=\(upT.map { "\($0)" } ?? "auto/longest"))")
        let wT = try Weights.load(tokenrigT)
        let glb = try Data(contentsOf: URL(fileURLWithPath: meshGLB))
        var pT = SkinTokensPipeline.Params(); pT.numBeams = beamsT
        let rigged = try SkinTokensPipeline.rig(glbData: glb, skinOnly: true, cls: "vroid",
                                                weights: wT, params: pT, up: upT) { s in
            if s % 50 == 0 { err("[tpl]   step \(s)") }
        }
        let outPath = optT("--out") ?? ((meshGLB as NSString).deletingPathExtension + "_tpl_rigged.glb")
        try rigged.write(to: URL(fileURLWithPath: outPath))
        print("[tpl] wrote \(outPath) (\(rigged.count) bytes)")
    } catch { err("[tpl] FAIL: \(error)"); exit(1) }

case "--quant-spike":
    // Weight-quant / decoder speed spike. Same mesh, same deterministic beam search, one run per
    // variant (weights loaded once); times the whole rig + the generation phase per token and writes
    // each variant's rigged GLB for an offline quality diff against the first (baseline) variant.
    //   swift run -c release RunSkinTokens --quant-spike <conv> <mesh.glb> --out <dir>
    //       [--route-b] [--variants castPerStep,fp32,int8,int4,fp32+sdpa,...]
    // List a variant twice to re-measure it warm (the first run of a process pays shader compile).
    let conv = requireArg(1, "<converted-dir>")
    let glb  = requireArg(2, "<mesh.glb>")
    func optQ(_ n: String) -> String? { args.firstIndex(of: n).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    let outDir = optQ("--out") ?? FileManager.default.currentDirectoryPath
    let routeB = args.contains("--route-b")
    let variantSpecs = (optQ("--variants") ?? "castPerStep,fp32,int8,int4").split(separator: ",").map(String.init)
    func parse(_ spec: String) -> Qwen3.GenOptions? {
        let parts = spec.split(separator: "+").map(String.init)
        guard let wm = Qwen3.GenOptions.WeightMode(rawValue: parts[0]) else { return nil }
        return Qwen3.GenOptions(weights: wm, fusedAttention: parts.dropFirst().contains("sdpa"))
    }
    do {
        let w = try Weights.load((conv as NSString).appendingPathComponent("tokenrig.safetensors"))
        for (_, v) in w.prefix(4) { eval(v.sum()) }
        let glbData = try Data(contentsOf: URL(fileURLWithPath: glb))
        try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        var seen: [String: Int] = [:]
        print("variant\trun\ttotal_s\tgen_s\tsteps\tms_per_step\tms_step_q1\tms_step_q4\tpeak_GB\tout")
        for spec in variantSpecs {
            guard let opts = parse(spec) else { err("[spike] bad variant \(spec)"); exit(2) }
            let run = (seen[spec] ?? 0) + 1; seen[spec] = run
            MLX.Memory.clearCache(); MLX.GPU.resetPeakMemory()
            var stamps: [Date] = []
            let t0 = Date()
            let params = SkinTokensPipeline.Params(gen: opts)
            let out = try SkinTokensPipeline.rig(glbData: glbData, skinOnly: routeB, cls: routeB ? "vroid" : nil,
                                                 weights: w, params: params, progress: { _ in stamps.append(Date()) })
            let total = Date().timeIntervalSince(t0)
            let path = (outDir as NSString).appendingPathComponent("\(spec)_r\(run).glb")
            try out.write(to: URL(fileURLWithPath: path))
            let steps = stamps.count
            let gen = steps > 1 ? stamps.last!.timeIntervalSince(stamps.first!) : 0
            func ms(_ a: Int, _ b: Int) -> Double {
                guard b > a, b < steps else { return 0 }
                return stamps[b].timeIntervalSince(stamps[a]) * 1000 / Double(b - a)
            }
            let q = max(1, steps / 4)
            print(String(format: "%@\t%d\t%.1f\t%.1f\t%d\t%.1f\t%.1f\t%.1f\t%.2f\t%@", spec, run, total, gen, steps,
                         ms(0, steps - 1), ms(0, q), ms(steps - 1 - q, steps - 1),
                         Double(MLX.Memory.peakMemory) / 1e9, path))
        }
    } catch { err("[spike] FAIL: \(error)"); exit(1) }

case "--measure":
    // Footprint harness (memory-harness.md): resident weights floor (post-load) + peak activation
    // (transient scratch during a full run, on the GPU stream = production). Feeds the manifest's
    // split QuantFootprint. Run detached; auto (default) or --route-b (heavier VRM 141-joint envelope).
    //   swift run -c release RunSkinTokens --measure <conv> <mesh.glb> [--route-b] [--cls vroid] [--quant int8]
    let conv = requireArg(1, "<converted-dir>")
    let glb  = requireArg(2, "<mesh.glb>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    let routeB = args.contains("--route-b")
    let clsM = args.firstIndex(of: "--cls").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
    func mb(_ b: Int) -> String { String(format: "%.0f MB", Double(b) / 1e6) }
    do {
        err("[mem] loading weights (GPU stream)")
        var w = try Weights.load(tokenrig)
        let int8M = (args.firstIndex(of: "--quant").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }) == "int8"
        if int8M { Qwen3.prequantize(&w, bits: 8); MLX.Memory.clearCache() }
        // Touch the weights on the GPU stream so they're resident where inference runs.
        for (_, v) in w.prefix(4) { eval(v.sum()) }
        let residentAfterLoad = MLX.Memory.activeMemory
        MLX.GPU.resetPeakMemory()
        let glbData = try Data(contentsOf: URL(fileURLWithPath: glb))
        err("[mem] resident post-load=\(mb(residentAfterLoad)) — running \(routeB ? "route-b" : "auto")")
        let t0 = Date()
        var mParams = SkinTokensPipeline.Params()
        if int8M { mParams.gen.weights = .int8 }
        _ = try SkinTokensPipeline.rig(glbData: glbData, skinOnly: routeB, cls: clsM, weights: w, params: mParams, progress: { s in
            if s % 100 == 0 { err("[mem]   step \(s) active=\(mb(MLX.Memory.activeMemory)) peak=\(mb(MLX.Memory.peakMemory))") }
        })
        let dt = Date().timeIntervalSince(t0)
        let peak = MLX.Memory.peakMemory
        let activation = max(0, peak - residentAfterLoad)
        print("[mem] ===== FOOTPRINT (\(routeB ? "skinOnly/VRM" : "auto"), \(int8M ? "int8" : "bf16")) =====")
        print("[mem] resident (weights floor, post-load) = \(mb(residentAfterLoad)) [\(residentAfterLoad) B]")
        print("[mem] peak (during run)                    = \(mb(peak)) [\(peak) B]")
        print("[mem] peakActivation (peak − resident)     = \(mb(activation)) [\(activation) B]")
        print("[mem] run \(String(format: "%.1f", dt))s")
    } catch { err("[mem] FAIL: \(error)"); exit(1) }

case "--vrm-load":
    // S5-prep — validate the GLTFKit VRM loader: merged mesh (world/bind space) + existing skeleton
    // (joints/parents/humanoid-bone-map). Sanity: single-root skeleton, plausible bounds, bone map.
    //   swift run RunSkinTokens --vrm-load <avatar.vrm|.glb>
    let vrm = requireArg(1, "<avatar.vrm|.glb>")
    do {
        let (mesh, skel, asset) = try VRMLoad.load(vrm)
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3), hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for i in 0..<mesh.vertexCount { for k in 0..<3 { let x = mesh.vertices[i*3+k]; lo[k] = min(lo[k], x); hi[k] = max(hi[k], x) } }
        let roots = skel.parents.filter { $0 < 0 }.count
        print("[vrm] \(vrm) isVRM=\(asset.isVRM) \(asset.vrmVersion.map { "\($0)" } ?? "-")")
        print("[vrm] merged mesh: verts=\(mesh.vertexCount) faces=\(mesh.faceCount) prims=\(mesh.primRanges.count)")
        print("[vrm] bounds lo=\(lo.map { String(format: "%.2f", $0) }) hi=\(hi.map { String(format: "%.2f", $0) })")
        print("[vrm] skeleton: J=\(skel.J) roots=\(roots) humanoidBones=\(skel.humanoid.count)")
        let key = ["hips", "spine", "head", "leftHand", "rightHand", "leftUpperLeg"].compactMap { skel.humanoid[$0] != nil ? $0 : nil }
        print("[vrm] humanoid keys present: \(key)")
        let ok = mesh.vertexCount > 0 && skel.J > 0 && roots == 1
        print("[vrm] VRM load \(ok ? "PASSED ✅ (single-root skeleton + merged mesh)" : "FAILED ❌ (roots=\(roots))")")
        exit(ok ? 0 : 1)
    } catch { err("[vrm] FAIL: \(error)"); exit(1) }

case "--s1a-gate":
    // S1a — mesh GLB I/O + predict-transform + SamplerMix. Gate = count + distribution (NOT element
    // match — query/RNG isn't numpy-PCG64-exact; the injected-points path is validated by S1b–S2).
    //   swift run RunSkinTokens --s1a-gate <mesh.glb>
    let glb = requireArg(1, "<mesh.glb>")
    var ok = false
    do {
        let mesh = try MeshIO.loadGLB(glb)
        err("[s1a] loaded \(glb): verts=\(mesh.vertexCount) faces=\(mesh.faceCount) normals=\(mesh.normals.count/3)")
        let aff = MeshIO.predictAffine(mesh)
        let normVerts = MeshIO.applyAffine(mesh.vertices, aff)
        let normMesh = Mesh(vertices: normVerts, faces: mesh.faces, normals: mesh.normals)
        let cloud = MeshIO.samplerMix(normMesh, numSamples: 54000, numVertexSamples: 16384, seed: 0)
        // distribution: all sampled points must live in [-1,1]³ (affine target); normals unit.
        var pmin: Float = .greatestFiniteMagnitude, pmax = -Float.greatestFiniteMagnitude
        for x in cloud.points { pmin = min(pmin, x); pmax = max(pmax, x) }
        var nBad = 0
        for i in 0..<cloud.count {
            let x = cloud.normals[i*3], y = cloud.normals[i*3+1], z = cloud.normals[i*3+2]
            if abs((x*x+y*y+z*z).squareRoot() - 1) > 1e-3 { nBad += 1 }
        }
        print("[s1a] affine center=\(aff.center.map { String(format: "%.3f", $0) }) scale=\(String(format: "%.4f", aff.scale))")
        print("[s1a] sampled \(cloud.count) pts — coord∈[\(String(format: "%.3f", pmin)),\(String(format: "%.3f", pmax))] non-unit-normals=\(nBad)")
        ok = cloud.count == 54000 && pmin >= -1.001 && pmax <= 1.001 && nBad == 0
        print("[s1a] S1a \(ok ? "PASSED ✅ (54000 pts in [-1,1]³, unit normals)" : "FAILED ❌")")
    } catch { err("[s1a] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--e2e":
    // END-TO-END (real mesh → rig): GLB → affine → SamplerMix(54000) → S1b Michelangelo →
    // S1c SkinVAE cond → grammar beam(10) → S1d skin decode → detokenize. The whole neural core on a
    // REAL mesh (no injected captures). Gate = 0 grammar violations + skin 0-unweighted + plausible J.
    //   swift run -c release RunSkinTokens --e2e <converted-dir> <mesh.glb> [--beams N] [--cls ID] [--max N]
    let conv = requireArg(1, "<converted-dir>")
    let glb  = requireArg(2, "<mesh.glb>")
    let tokenrig = (conv as NSString).appendingPathComponent("tokenrig.safetensors")
    func opt2(_ n: String) -> String? { args.firstIndex(of: n).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
    let beams = opt2("--beams").flatMap { Int($0) } ?? 10
    let clsTok = opt2("--cls").flatMap { Int($0) } ?? SkinTokenGrammar.skinTokens.clsTokens[2]  // articulation
    let maxNew = opt2("--max").flatMap { Int($0) } ?? 2040
    let L = SkinTokenGrammar.skinTokens
    var ok = false
    do {
        err("[e2e] loading weights + mesh")
        let w = try Weights.load(tokenrig)
        let mesh = try MeshIO.loadGLB(glb)
        let aff = MeshIO.predictAffine(mesh)
        let normMesh = Mesh(vertices: MeshIO.applyAffine(mesh.vertices, aff), faces: mesh.faces, normals: mesh.normals)
        let cloud = MeshIO.samplerMix(normMesh, seed: 0)
        err("[e2e] mesh verts=\(mesh.vertexCount) → sampled \(cloud.count); building encoder inputs")
        let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }
        let kvPc = MLXArray(cloud.points, [1, cloud.count, 3])
        let kvFeats = MLXArray(cloud.normals, [1, cloud.count, 3])
        // S1b: michelangelo query = FPS(512) of a 2048 subset; kv = full cloud.
        let qIdx = MeshIO.fpsQuery(points: cloud.points, count: cloud.count, subset: 2048, k: 512, seed: 0)
        var qp = [Float](repeating: 0, count: 512 * 3), qf = [Float](repeating: 0, count: 512 * 3)
        for (i, idx) in qIdx.enumerated() { for k in 0..<3 { qp[i*3+k] = cloud.points[idx*3+k]; qf[i*3+k] = cloud.normals[idx*3+k] } }
        let queryPc = MLXArray(qp, [1, 512, 3]), queryFeats = MLXArray(qf, [1, 512, 3])
        err("[e2e] S1b michelangelo encode (512-query / 54000-KV)")
        let latents = Michelangelo.encode(kvPc: bf(kvPc), kvFeats: bf(kvFeats), queryPc: bf(queryPc), queryFeats: bf(queryFeats), weights: w)
        let meshCond = Michelangelo.outputProj(latents, weights: w).asType(.float32)   // (1,512,896)
        eval(meshCond)
        // S1c: skinvae cond query = FPS(384); cond_kv = full cloud (pts⊕normals).
        let cIdx = MeshIO.fpsQuery(points: cloud.points, count: cloud.count, subset: 1536, k: 384, seed: 1)
        var cq = [Float](repeating: 0, count: 384 * 6)
        for (i, idx) in cIdx.enumerated() {
            for k in 0..<3 { cq[i*6+k] = cloud.points[idx*3+k]; cq[i*6+3+k] = cloud.normals[idx*3+k] }
        }
        let condQ = bf(MLXArray(cq, [1, 384, 6]))
        let condKvPts = concatenated([kvPc, kvFeats], axis: -1)                          // (1,54000,6)
        let condLat = SkinVAE.condEncode(condQ: SkinVAE.qkvInput(condQ), condKv: SkinVAE.qkvInput(bf(condKvPts)), weights: w)
        eval(condLat)
        err("[e2e] S1c cond_latents \(condLat.shape); beam(\(beams)) decode")
        // Beam decode.
        var p = SkinGen.Params(); p.numBeams = beams; p.maxNewTokens = maxNew
        let dec = Qwen3Decoder(weights: w)
        let res = SkinGen.generate(decoder: dec, meshCond: meshCond, startTokens: [L.bos, clsTok],
                                   layout: L, tokensPerSkin: 4, params: p) { step, b, d in
            if step % 50 == 0 { err("[e2e]   gen step \(step) beams=\(b) completed=\(d)") }
        }
        let ids = res.outputIds
        let skel = try SkinDetokenize.detokenize(ids, layout: L)
        let (xlo, xhi) = skel.xBounds
        print("[e2e] generated \(ids.count) tok — J=\(skel.J) depth=\(skel.treeDepth) x∈[\(String(format: "%.3f", xlo)),\(String(format: "%.3f", xhi))] violations=\(res.violations)")
        // S1d skin decode.
        var unweighted = -1
        var skinHost: [Float]? = nil
        let J = skel.J
        if let sw = ids.firstIndex(of: L.switchTokenId), J > 0 {
            let skinToks = Array(ids[(sw + 1)..<min(sw + 1 + J * 4, ids.count)])
            if skinToks.count == J * 4 {
                let fsqIdx = MLXArray(skinToks.map { Int32($0 - L.tokenizerVocabSize) })
                let codes = SkinVAE.fsqIndicesToCodes(fsqIdx, weights: w).asType(.bfloat16)
                let sampledCond = bf(condKvPts)
                let clBf = condLat.asType(.bfloat16)
                var cols: [MLXArray] = []
                for j in 0..<J {
                    let z = codes[(j*4)..<((j+1)*4), 0...].reshaped([1, 4, 512])
                    let lg = SkinVAE.decode(z: z, cond: clBf, sampledCond: sampledCond, weights: w)
                    eval(lg); cols.append(lg.reshaped([54000, 1])); MLX.Memory.clearCache()
                }
                let skin = concatenated(cols, axis: 1).asType(.float32)   // (54000,J) row-major
                eval(skin)
                let maxInf = skin.max(axis: 1); eval(maxInf)
                unweighted = Int((maxInf .< 1e-6).asType(.int32).sum().item(Int32.self))
                skinHost = skin.asArray(Float.self)
                print("[e2e] skin_pred (54000,\(J)) — 0-unweighted? \(unweighted == 0) (unweighted=\(unweighted))")
            }
        }

        // --- S3 propagate to ORIGINAL glTF verts + S4 GLB inject (if --out given) ---
        if let outPath = opt2("--out"), let sk = skinHost, J > 0 {
            err("[e2e] S3 propagating skin to \(mesh.vertexCount) original verts (KD-tree nearest)")
            let vertsNorm = MeshIO.applyAffine(mesh.vertices, aff)
            let perVert = SkinRig.propagate(sampledPoints: cloud.points, sampledSkin: sk, J: J, queryPts: vertsNorm)
            let sw4 = SkinRig.topFour(skin: perVert, J: J)
            // invert affine on joints → original space.
            var jointsOrig = [Float](repeating: 0, count: J * 3)
            for j in 0..<J { for k in 0..<3 { jointsOrig[j*3+k] = skel.joints[j][k] * aff.scale + aff.center[k] } }
            err("[e2e] S4 injecting rig → \(outPath) (\(mesh.primRanges.count) primitives)")
            try SkinRig.injectRig(glbPath: glb, outPath: outPath, jointsOrig: jointsOrig, parents: skel.parents, skin: sw4, primRanges: mesh.primRanges)
            print("[e2e] wrote rigged GLB: \(outPath) (J=\(J), \(mesh.vertexCount) skinned verts, \(mesh.primRanges.count) prims)")
        }

        ok = res.violations == 0 && J > 0 && unweighted == 0
        print("[e2e] END-TO-END \(ok ? "PASSED ✅ (real mesh → plausible rig)" : "FAILED ❌")")
    } catch { err("[e2e] FAIL: \(error)"); exit(1) }
    exit(ok ? 0 : 1)

case "--s3-gate", "--s4-gate":
    err("[skintokens] \(mode): S3/S4 are exercised end-to-end via `--e2e` (KD-tree propagate + GLB inject). See PORTING-SPEC.md.")
    exit(3)

case "--s5-gate", "--s6-gate":
    err("[skintokens] \(mode): not yet implemented (P1 remaining). See ../SkinTokensDev/PORTING-SPEC.md for the phase this gate covers.")
    exit(3)

default:
    print("""
    RunSkinTokens — SkinTokens Swift-MLX parity/e2e harness (\(SkinTokensCore.version))
      --version
      --npy <path.npy>                          probe the shared fixture reader (NPYReader)
      --s0-headers <file.safetensors>           dump a safetensors header
      --s0-gate <converted-dir>                 S0 key-contract diff (both variants, 0 missing/0 unused)
      --s1b/1c/1d/2-gate <conv> <cap>           deterministic numeric parity ladder (cos ≥0.999)
      --s2b-kv-check <conv> <cap>               incremental KV cache vs teacher-forced (cos ~1.0)
      --s2b-gate <conv> <cap> [--beams N …]     grammar-constrained batched beam(10) decode
      --detok-check <cap>                       skeleton detokenize vs captured joints/parents
      --detok-check / --tokenize-check <cap>     skeleton ↔ tokens roundtrip vs captured
      --s1a-gate <mesh.glb>                      mesh I/O + affine + SamplerMix(54000)
      --vrm-load <avatar.vrm>                    GLTFKit VRM parse: world-space merge + skeleton
      --e2e <conv> <mesh.glb> [--out rigged.glb] ⭐ AUTO: arbitrary mesh → rigged GLB (new skeleton)
      --route-b <conv> <avatar.vrm> [--out g]    ⭐ ROUTE-B: VRM's own skeleton → skin-only (companion)
      (S7 engine-drive: `swift run RunSkinTokensEngine`  IN_GLB=… MODE=auto|skinOnly WEIGHTS_DIR=…)
    """)
}
