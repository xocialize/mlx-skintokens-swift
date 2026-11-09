import Foundation
import MLX
import SkinTokensCore

/// The end-to-end SkinTokens rig pipeline (engine-free) — the exact orchestration validated by the
/// `--e2e` (auto) and `--route-b` (skin-only) CLI gates, factored so BOTH the CLI and the
/// `ModelPackage` conformer call one implementation. Weights are injected (`[String: MLXArray]`,
/// loaded by the caller). Runs on the current MLX stream (GPU in production).
public enum SkinTokensPipeline {

    public struct Params: Sendable {
        public var numBeams = 10
        public var maxNewTokens = 2040
        /// Generation decoder options. Default (v0.2.0): bf16 weights + fused SDPA attention — 27–33%
        /// faster per step than the explicit attention, output-equivalent on in-distribution meshes
        /// (AB-R-0812). Pass `.int8` weights for the quantized tier (weights pre-quantized at load).
        public var gen: Qwen3.GenOptions = Params.defaultGen
        public static let defaultGen = Qwen3.GenOptions(weights: .castPerStep, fusedAttention: true)
        public init(numBeams: Int = 10, maxNewTokens: Int = 2040, gen: Qwen3.GenOptions = Params.defaultGen) {
            self.numBeams = numBeams; self.maxNewTokens = maxNewTokens; self.gen = gen
        }
    }

    /// The one canonical entry: GLB bytes → rigged GLB bytes. `skinOnly` selects Route-B (the input
    /// GLB's own skeleton is used, `cls` conditions the model, e.g. "vroid"); otherwise `auto`
    /// generates a fresh skeleton. Weights = the loaded `tokenrig.safetensors` dict.
    /// `checkpoint` is called every generation step (per token) and at stage seams — a caller passes
    /// `try Task.checkCancellation()` for cooperative cancellation (the CAN cadence); it rethrows.
    public static func rig(glbData: Data, skinOnly: Bool, cls: String?, weights: [String: MLXArray],
                           params: Params = Params(), up: VRoidTemplate.UpAxis? = nil,
                           checkpoint: @escaping () throws -> Void = {},
                           progress: ((_ step: Int) -> Void)? = nil) throws -> Data {
        let tmp = FileManager.default.temporaryDirectory
        let inURL = tmp.appendingPathComponent("skintokens_in_\(UUID().uuidString).glb")
        let outURL = tmp.appendingPathComponent("skintokens_out_\(UUID().uuidString).glb")
        try glbData.write(to: inURL)
        defer { try? FileManager.default.removeItem(at: inURL); try? FileManager.default.removeItem(at: outURL) }
        if skinOnly {
            do {
                try rigRouteB(inPath: inURL.path, outPath: outURL.path, weights: weights, cls: cls, params: params, checkpoint: checkpoint, progress: progress)
            } catch VRMLoad.Err.noSkeleton {
                // Bare mesh (e.g. a TRELLIS output) — no embedded skeleton. Fit the bundled canonical
                // VRoid template onto it and rig against that (the AnimeStudio Route-B path).
                let template = try VRoidTemplate.loadBundled()
                try rigWithTemplate(inPath: inURL.path, outPath: outURL.path, weights: weights,
                                    template: template, up: up, cls: cls ?? "vroid", params: params,
                                    checkpoint: checkpoint, progress: progress)
            }
        } else {
            try rigAuto(inPath: inURL.path, outPath: outURL.path, weights: weights, params: params, checkpoint: checkpoint, progress: progress)
        }
        return try Data(contentsOf: outURL)
    }

    private static let L = SkinTokenGrammar.skinTokens
    private static let bf: (MLXArray) -> MLXArray = { $0.asType(.bfloat16) }

    /// Generation decoder config: bf16 KV cache (halves the persistent beam cache — the biggest
    /// activation lever; compute stays fp32). Generation is non-parity, so the bf16-rounding is safe.
    private static var genConfig: Qwen3.Config { var c = Qwen3.Config(); c.kvDtype = .bfloat16; return c }

    /// Shared encoder pass: sampled cloud → learned_mesh_cond (1,512,896) + cond_latents (1,384,512).
    private static func encode(_ cloud: MeshIO.SampleCloud, weights w: [String: MLXArray])
        -> (meshCond: MLXArray, condLat: MLXArray, condKvPts: MLXArray) {
        let kvPc = MLXArray(cloud.points, [1, cloud.count, 3]); let kvFeats = MLXArray(cloud.normals, [1, cloud.count, 3])
        let qIdx = MeshIO.fpsQuery(points: cloud.points, count: cloud.count, subset: 2048, k: 512, seed: 0)
        var qp = [Float](repeating: 0, count: 512 * 3), qf = [Float](repeating: 0, count: 512 * 3)
        for (i, idx) in qIdx.enumerated() { for k in 0..<3 { qp[i*3+k] = cloud.points[idx*3+k]; qf[i*3+k] = cloud.normals[idx*3+k] } }
        let latents = Michelangelo.encode(kvPc: bf(kvPc), kvFeats: bf(kvFeats),
                                          queryPc: bf(MLXArray(qp, [1,512,3])), queryFeats: bf(MLXArray(qf, [1,512,3])), weights: w)
        let meshCond = Michelangelo.outputProj(latents, weights: w).asType(.float32); eval(meshCond)
        let cIdx = MeshIO.fpsQuery(points: cloud.points, count: cloud.count, subset: 1536, k: 384, seed: 1)
        var cq = [Float](repeating: 0, count: 384 * 6)
        for (i, idx) in cIdx.enumerated() { for k in 0..<3 { cq[i*6+k] = cloud.points[idx*3+k]; cq[i*6+3+k] = cloud.normals[idx*3+k] } }
        let condKvPts = concatenated([kvPc, kvFeats], axis: -1)
        let condLat = SkinVAE.condEncode(condQ: SkinVAE.qkvInput(bf(MLXArray(cq, [1,384,6]))), condKv: SkinVAE.qkvInput(bf(condKvPts)), weights: w); eval(condLat)
        return (meshCond, condLat, condKvPts)
    }

    /// Shared skin decode: emitted skin tokens → per-sampled-point skin `[54000, J]` (host, row-major).
    private static func decodeSkin(ids: [Int], J: Int, condLat: MLXArray, condKvPts: MLXArray,
                                   weights w: [String: MLXArray]) -> [Float]? {
        guard let sw = ids.firstIndex(of: L.switchTokenId), J > 0 else { return nil }
        let skinToks = Array(ids[(sw+1)..<min(sw+1+J*4, ids.count)])
        guard skinToks.count == J * 4 else { return nil }
        let codes = SkinVAE.fsqIndicesToCodes(MLXArray(skinToks.map { Int32($0 - L.tokenizerVocabSize) }), weights: w).asType(.bfloat16)
        let sampledCond = bf(condKvPts); let clBf = condLat.asType(.bfloat16)
        var cols: [MLXArray] = []
        for j in 0..<J {
            let z = codes[(j*4)..<((j+1)*4), 0...].reshaped([1, 4, 512])
            // point-BATCHED decode: caps the [b,heads,Nq,388] cross-attn transient (P2 efficiency lever).
            let lg = SkinVAE.decodeBatched(z: z, cond: clBf, sampledCond: sampledCond, weights: w)
            eval(lg); cols.append(lg.reshaped([54000, 1])); MLX.Memory.clearCache()
        }
        let skin = concatenated(cols, axis: 1).asType(.float32); eval(skin)
        return skin.asArray(Float.self)
    }

    // MARK: - auto (generate skeleton + skin)

    static func rigAuto(inPath: String, outPath: String, weights w: [String: MLXArray],
                        params: Params, checkpoint: @escaping () throws -> Void = {}, progress: ((Int) -> Void)?) throws {
        let mesh = try MeshIO.loadGLB(inPath)
        let aff = MeshIO.predictAffine(mesh)
        let normMesh = Mesh(vertices: MeshIO.applyAffine(mesh.vertices, aff), faces: mesh.faces, normals: mesh.normals, primRanges: mesh.primRanges)
        let cloud = MeshIO.samplerMix(normMesh, seed: 0)
        try checkpoint()
        let (meshCond, condLat, condKvPts) = encode(cloud, weights: w)
        try checkpoint()
        var p = SkinGen.Params(); p.numBeams = params.numBeams; p.maxNewTokens = params.maxNewTokens
        let dec = Qwen3Decoder(weights: w, config: genConfig, options: params.gen)
        let res = try SkinGen.generate(decoder: dec, meshCond: meshCond, startTokens: [L.bos, L.clsTokens[2]],  // articulation
                                       layout: L, tokensPerSkin: 4, params: p) { s, _, _ in progress?(s); try checkpoint() }
        try checkpoint()
        let ids = res.outputIds
        let skel = try SkinDetokenize.detokenize(ids, layout: L)
        guard skel.J > 0, let skinHost = decodeSkin(ids: ids, J: skel.J, condLat: condLat, condKvPts: condKvPts, weights: w) else {
            throw RigError.decodeFailed
        }
        let vertsNorm = MeshIO.applyAffine(mesh.vertices, aff)
        let perVert = SkinRig.propagate(sampledPoints: cloud.points, sampledSkin: skinHost, J: skel.J, queryPts: vertsNorm)
        let sw4 = SkinRig.topFour(skin: perVert, J: skel.J)
        var jointsOrig = [Float](repeating: 0, count: skel.J * 3)
        for j in 0..<skel.J { for k in 0..<3 { jointsOrig[j*3+k] = skel.joints[j][k] * aff.scale + aff.center[k] } }
        try SkinRig.injectRig(glbPath: inPath, outPath: outPath, jointsOrig: jointsOrig, parents: skel.parents, skin: sw4, primRanges: mesh.primRanges)
    }

    // MARK: - skinOnly / Route-B (VRM's own skeleton → skin)

    static func rigRouteB(inPath: String, outPath: String, weights w: [String: MLXArray],
                          cls: String?, params: Params, checkpoint: @escaping () throws -> Void = {}, progress: ((Int) -> Void)?) throws {
        let (mesh, skel, _) = try VRMLoad.load(inPath)
        // combined affine over verts AND joints.
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3), hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for i in 0..<mesh.vertexCount { for k in 0..<3 { let x = mesh.vertices[i*3+k]; lo[k] = min(lo[k], x); hi[k] = max(hi[k], x) } }
        for j in skel.joints { for k in 0..<3 { lo[k] = min(lo[k], j[k]); hi[k] = max(hi[k], j[k]) } }
        let center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]
        let scaleV = max(hi[0]-lo[0], max(hi[1]-lo[1], hi[2]-lo[2])) / 2
        let aff = MeshIO.Affine(center: center, scale: scaleV == 0 ? 1 : scaleV)
        let normMesh = Mesh(vertices: MeshIO.applyAffine(mesh.vertices, aff), faces: mesh.faces, normals: mesh.normals, primRanges: mesh.primRanges)
        let normJoints = skel.joints.map { j in (0..<3).map { (j[$0] - aff.center[$0]) / aff.scale } }
        let cloud = MeshIO.samplerMix(normMesh, seed: 0)
        try checkpoint()
        let (meshCond, condLat, condKvPts) = encode(cloud, weights: w)
        try checkpoint()
        // CLEAN the skeleton to the deforming humanoid set (drop Root/_end/secondary, DFS-reorder) — the
        // model is trained on this; a raw 141-bone VRoid list shatters under animation (see SkeletonClean).
        // cls = articulation (the reference's class for a parsed skeleton; CLI-proven 4.0% NN-mismatch).
        let (order, cParents) = SkeletonClean.humanoidReorder(names: skel.jointNames, parents: skel.parents,
                                                              keepIndices: Set(skel.humanoid.values))
        let cJoints = order.map { normJoints[$0] }
        let cNodes  = order.map { skel.jointNodes[$0] }
        let ib = skel.inverseBind()
        let cIBM = order.flatMap { oi in (0..<16).map { ib[oi*16 + $0] } }
        let clsTok = L.clsTokens[2]   // articulation
        let prefix = SkinDetokenize.tokenize(joints: cJoints, parents: cParents, cls: clsTok, layout: L)
        let Jin = (try? SkinTokenGrammar.bonesInSequence(prefix + [L.lmEos], layout: L)) ?? cJoints.count
        var p = SkinGen.Params(); p.numBeams = params.numBeams; p.maxNewTokens = Jin * 4 + 8
        let dec = Qwen3Decoder(weights: w, config: genConfig, options: params.gen)
        let res = try SkinGen.generate(decoder: dec, meshCond: meshCond, startTokens: prefix, layout: L, tokensPerSkin: 4, params: p) { s, _, _ in progress?(s); try checkpoint() }
        try checkpoint()
        let ids = res.outputIds
        let Jout = (try? SkinTokenGrammar.bonesInSequence(ids, layout: L)) ?? -1
        guard Jout > 0, let skinHost = decodeSkin(ids: ids, J: Jout, condLat: condLat, condKvPts: condKvPts, weights: w) else {
            throw RigError.decodeFailed
        }
        let vertsNorm = MeshIO.applyAffine(mesh.vertices, aff)
        let perVert = SkinRig.propagate(sampledPoints: cloud.points, sampledSkin: skinHost, J: Jout, queryPts: vertsNorm)
        let sw4 = SkinRig.topFour(skin: perVert, J: Jout)
        try SkinRig.injectRigRouteB(glbPath: inPath, outPath: outPath, jointNodes: cNodes,
                                    inverseBind: cIBM, skeletonRoot: cNodes.first ?? skel.rootNode,
                                    skin: sw4, primRanges: mesh.primRanges)
    }

    // MARK: - template (bare mesh → fitted canonical VRoid skeleton → skin)

    /// Route-B for a mesh with NO embedded skeleton (e.g. a TRELLIS output): fit the bundled canonical
    /// VRoid T-pose skeleton to the mesh, tokenize it as the prefix, predict skin only, and author the
    /// VRM-named bones via `injectRig`. This is the AnimeStudio-bound path — the output GLB carries
    /// VRM humanoid bone names by construction.
    static func rigWithTemplate(inPath: String, outPath: String, weights w: [String: MLXArray],
                                template: VRoidTemplate, up: VRoidTemplate.UpAxis?, cls: String?,
                                params: Params, checkpoint: @escaping () throws -> Void = {},
                                progress: ((Int) -> Void)?) throws {
        let mesh = try MeshIO.loadGLB(inPath)
        let fitted = template.fit(toMeshVertices: mesh.vertices, up: up)   // [J][3] in mesh space

        // combined affine over verts AND fitted joints (mirrors rigRouteB normalization).
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3), hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for i in 0..<mesh.vertexCount { for k in 0..<3 { let x = mesh.vertices[i*3+k]; lo[k] = min(lo[k], x); hi[k] = max(hi[k], x) } }
        for j in fitted { for k in 0..<3 { lo[k] = min(lo[k], j[k]); hi[k] = max(hi[k], j[k]) } }
        let center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]
        let scaleV = max(hi[0]-lo[0], max(hi[1]-lo[1], hi[2]-lo[2])) / 2
        let aff = MeshIO.Affine(center: center, scale: scaleV == 0 ? 1 : scaleV)
        let normMesh = Mesh(vertices: MeshIO.applyAffine(mesh.vertices, aff), faces: mesh.faces, normals: mesh.normals, primRanges: mesh.primRanges)
        let normJoints = fitted.map { j in (0..<3).map { (j[$0] - aff.center[$0]) / aff.scale } }
        let cloud = MeshIO.samplerMix(normMesh, seed: 0)
        try checkpoint()
        let (meshCond, condLat, condKvPts) = encode(cloud, weights: w)
        try checkpoint()
        // CLEAN the template to the deforming humanoid set (drop Root/_end/secondary, DFS-reorder) so the
        // model sees the in-distribution skeleton it was trained on — otherwise skin binds to junk bones
        // and the mesh shatters under animation (see SkeletonClean; CLI-proven 4.0% vs 26% NN-mismatch).
        // Body-only keep-set: drop the 30 FINGER bones. On a fitted template the finger joints are the
        // least reliable placements (bbox fit + chunky/low-detail generated hands) → verts bind to
        // misplaced finger bones → spikes when a clip curls fingers. Without them the hand moves rigidly
        // with the hand bone (clean), and fingers are optional VRM humanoid bones so retarget just skips
        // them. (rigRouteB keeps fingers — a real VRM's own finger joints are exact and animate fine.)
        let fingers = ["thumb", "index", "middle", "ring", "little"]
        let bodyOnly = template.humanoid.filter { k, _ in !fingers.contains { k.lowercased().contains($0) } }
        let (order, cParents) = SkeletonClean.humanoidReorder(names: template.names, parents: template.parents,
                                                              keepIndices: Set(bodyOnly.values))
        let o2n = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
        let cNormJoints = order.map { normJoints[$0] }
        let cFitted     = order.map { fitted[$0] }
        let cNames      = order.map { template.names[$0] }
        let cHumanoid   = template.humanoid.compactMapValues { o2n[$0] }
        let clsTok = L.clsTokens[2]   // articulation (reference class for a parsed skeleton)
        let prefix = SkinDetokenize.tokenize(joints: cNormJoints, parents: cParents, cls: clsTok, layout: L)
        let Jin = (try? SkinTokenGrammar.bonesInSequence(prefix + [L.lmEos], layout: L)) ?? cNormJoints.count
        var p = SkinGen.Params(); p.numBeams = params.numBeams; p.maxNewTokens = Jin * 4 + 8
        let dec = Qwen3Decoder(weights: w, config: genConfig, options: params.gen)
        let res = try SkinGen.generate(decoder: dec, meshCond: meshCond, startTokens: prefix, layout: L, tokensPerSkin: 4, params: p) { s, _, _ in progress?(s); try checkpoint() }
        try checkpoint()
        let ids = res.outputIds
        let Jout = (try? SkinTokenGrammar.bonesInSequence(ids, layout: L)) ?? -1
        guard Jout == cNormJoints.count, let skinHost = decodeSkin(ids: ids, J: Jout, condLat: condLat, condKvPts: condKvPts, weights: w) else {
            throw RigError.decodeFailed
        }
        let vertsNorm = MeshIO.applyAffine(mesh.vertices, aff)
        let perVert = SkinRig.propagate(sampledPoints: cloud.points, sampledSkin: skinHost, J: Jout, queryPts: vertsNorm)
        let sw4 = SkinRig.topFour(skin: perVert, J: Jout)
        // author fresh VRM-named bones from the CLEANED fitted joints (mesh space) — injectRig, not RouteB.
        // Also emit the VRMC_vrm humanoid extension so the AnimeStudio converter produces humanoid.json.
        let jointsOrig = cFitted.flatMap { $0 }
        try SkinRig.injectRig(glbPath: inPath, outPath: outPath, jointsOrig: jointsOrig,
                              parents: cParents, skin: sw4, primRanges: mesh.primRanges,
                              names: cNames, humanoidBones: cHumanoid)
    }

    public enum RigError: Swift.Error, CustomStringConvertible {
        case decodeFailed
        public var description: String { "SkinTokens rig decode failed (empty skeleton or skin-token mismatch)" }
    }
}
