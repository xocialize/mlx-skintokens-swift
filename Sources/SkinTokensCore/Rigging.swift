import Foundation

/// S3 + S4 — skin propagation (sampled points → ORIGINAL glTF render-verts via nearest-neighbour) and
/// in-place GLB rig injection (JOINTS_0 / WEIGHTS_0 + a skeleton node hierarchy + a glTF skin). Owning
/// mesh I/O and propagating onto the ORIGINAL verts (NOT a bpy-welded set) makes the injection
/// order-correct BY CONSTRUCTION for any mesh (HANDOFF vertex-identity note).
public enum SkinRig {

    // MARK: - S3: KD-tree nearest-neighbour propagation

    /// A compact 3D KD-tree over a flat `[M*3]` point array (indices into it).
    public final class KDTree {
        let pts: [Float]
        var nodes: [Int]          // point index at each tree position (median layout)
        struct Build { var lo: Int; var hi: Int; var axis: Int }
        // We store a flat array `order` = point indices arranged so a recursive median split is implicit.
        var order: [Int]

        public init(points: [Float]) {
            self.pts = points
            let n = points.count / 3
            self.order = Array(0..<n)
            self.nodes = []
            buildRange(0, n, 0)
        }

        private func coord(_ i: Int, _ a: Int) -> Float { pts[order[i]*3 + a] }

        // In-place median (nth_element-ish) build via recursive partition on `order`.
        private func buildRange(_ lo: Int, _ hi: Int, _ depth: Int) {
            if hi - lo <= 1 { return }
            let axis = depth % 3
            let mid = (lo + hi) / 2
            nthElement(lo, hi, mid, axis)
            buildRange(lo, mid, depth + 1)
            buildRange(mid + 1, hi, depth + 1)
        }

        private func nthElement(_ lo: Int, _ hi: Int, _ k: Int, _ axis: Int) {
            var l = lo, r = hi - 1
            while l < r {
                let pivot = coord((l + r) / 2, axis)
                var i = l, j = r
                while i <= j {
                    while coord(i, axis) < pivot { i += 1 }
                    while coord(j, axis) > pivot { j -= 1 }
                    if i <= j { order.swapAt(i, j); i += 1; j -= 1 }
                }
                if k <= j { r = j } else if k >= i { l = i } else { break }
            }
        }

        /// Index (into the original point array) of the nearest stored point to `(x,y,z)`.
        public func nearest(_ x: Float, _ y: Float, _ z: Float) -> Int {
            var bestI = -1; var bestD = Float.greatestFiniteMagnitude
            func recurse(_ lo: Int, _ hi: Int, _ depth: Int) {
                if hi - lo <= 0 { return }
                let axis = depth % 3
                let mid = (lo + hi) / 2
                let pi = order[mid]
                let dx = pts[pi*3]-x, dy = pts[pi*3+1]-y, dz = pts[pi*3+2]-z
                let d = dx*dx + dy*dy + dz*dz
                if d < bestD { bestD = d; bestI = pi }
                let diff = (axis == 0 ? x - pts[pi*3] : axis == 1 ? y - pts[pi*3+1] : z - pts[pi*3+2])
                let near = diff < 0 ? (lo, mid) : (mid + 1, hi)
                let far  = diff < 0 ? (mid + 1, hi) : (lo, mid)
                recurse(near.0, near.1, depth + 1)
                if diff * diff < bestD { recurse(far.0, far.1, depth + 1) }
            }
            recurse(0, order.count, 0)
            return bestI
        }
    }

    /// Propagate a sampled-point skin `[M,J]` (row-major) onto `queryPts` `[Q*3]` by nearest sampled
    /// point. Returns the per-query skin `[Q,J]` (row-major, raw — still per-joint sigmoid).
    public static func propagate(sampledPoints: [Float], sampledSkin: [Float], J: Int,
                                 queryPts: [Float]) -> [Float] {
        let tree = KDTree(points: sampledPoints)
        let Q = queryPts.count / 3
        var out = [Float](repeating: 0, count: Q * J)
        for q in 0..<Q {
            let ni = tree.nearest(queryPts[q*3], queryPts[q*3+1], queryPts[q*3+2])
            for j in 0..<J { out[q*J + j] = sampledSkin[ni*J + j] }
        }
        return out
    }

    // MARK: - top-4 skinning weights (glTF JOINTS_0/WEIGHTS_0 are 4-wide)

    public struct SkinWeights: Sendable { public var joints: [UInt16]; public var weights: [Float]; public var Q: Int }

    /// Reduce a per-vertex skin `[Q,J]` to the 4 strongest influences, normalized to sum 1.
    public static func topFour(skin: [Float], J: Int) -> SkinWeights {
        let Q = skin.count / J
        var jointsOut = [UInt16](repeating: 0, count: Q * 4)
        var weightsOut = [Float](repeating: 0, count: Q * 4)
        for q in 0..<Q {
            // find top-4 by value
            var top = [(w: Float, j: Int)](repeating: (0, 0), count: 4)
            for j in 0..<J {
                let w = skin[q*J + j]
                if w > top[3].w {
                    top[3] = (w, j)
                    var k = 3
                    while k > 0 && top[k].w > top[k-1].w { top.swapAt(k, k-1); k -= 1 }
                }
            }
            let s = max(top[0].w + top[1].w + top[2].w + top[3].w, 1e-8)
            for k in 0..<4 { jointsOut[q*4+k] = UInt16(top[k].j); weightsOut[q*4+k] = top[k].w / s }
        }
        return SkinWeights(joints: jointsOut, weights: weightsOut, Q: Q)
    }

    // MARK: - S4: GLB rig injection

    /// Inject a rig into a `.glb`: append a per-primitive JOINTS_0/WEIGHTS_0 slice to EVERY primitive
    /// (sliced from the merged `skin` by `primRanges`), add one node per joint (translation in ORIGINAL
    /// space), a glTF skin (joints + inverseBindMatrices), bind every rigged mesh's node to the skin,
    /// and write a new `.glb`. `jointsOrig` = joint heads in ORIGINAL mesh space `[J*3]`.
    public static func injectRig(glbPath: String, outPath: String,
                                 jointsOrig: [Float], parents: [Int],
                                 skin: SkinWeights, primRanges: [PrimRange],
                                 names: [String]? = nil,
                                 humanoidBones: [String: Int]? = nil) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: glbPath))
        func u32(_ off: Int) -> UInt32 { data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: UInt32.self) } }
        let jsonLen = Int(u32(12))
        let jsonStart = 20
        var json = try JSONSerialization.jsonObject(with: data.subdata(in: jsonStart..<jsonStart+jsonLen)) as! [String: Any]
        var p = jsonStart + jsonLen; p = (p + 3) & ~3
        let binLen = Int(u32(p)); let binStart = p + 8
        var bin = data.subdata(in: binStart..<binStart + binLen)

        var buffers = json["buffers"] as! [[String: Any]]
        var bufferViews = (json["bufferViews"] as? [[String: Any]]) ?? []
        var accessors = (json["accessors"] as? [[String: Any]]) ?? []
        var nodes = (json["nodes"] as? [[String: Any]]) ?? []
        var meshes = json["meshes"] as! [[String: Any]]
        let J = jointsOrig.count / 3

        // helper: append bytes to BIN (4-byte aligned) → return (bufferView index).
        func addBufferView(_ bytes: Data) -> Int {
            while bin.count % 4 != 0 { bin.append(0) }
            let off = bin.count
            bin.append(bytes)
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": bytes.count])
            return bufferViews.count - 1
        }
        func f32(_ arr: [Float]) -> Data { arr.withUnsafeBytes { Data($0) } }
        func u16(_ arr: [UInt16]) -> Data { arr.withUnsafeBytes { Data($0) } }

        // --- inverseBindMatrices (J × mat4, column-major; IBM = translate(-joint) ) ---
        var ibm = [Float](repeating: 0, count: J * 16)
        for j in 0..<J {
            ibm[j*16+0] = 1; ibm[j*16+5] = 1; ibm[j*16+10] = 1; ibm[j*16+15] = 1   // identity
            ibm[j*16+12] = -jointsOrig[j*3]; ibm[j*16+13] = -jointsOrig[j*3+1]; ibm[j*16+14] = -jointsOrig[j*3+2]
        }
        let ibmAcc: Int = { let bv = addBufferView(f32(ibm))
            accessors.append(["bufferView": bv, "componentType": 5126, "count": J, "type": "MAT4"]); return accessors.count - 1 }()

        // --- joint nodes (translation relative to parent) ---
        let jointNodeBase = nodes.count
        var children = [[Int]](repeating: [], count: J)
        var root = 0
        for j in 0..<J { if parents[j] >= 0 { children[parents[j]].append(jointNodeBase + j) } else { root = j } }
        for j in 0..<J {
            let parentPos = parents[j] >= 0 ? [jointsOrig[parents[j]*3], jointsOrig[parents[j]*3+1], jointsOrig[parents[j]*3+2]] : [Float(0),0,0]
            let t = [jointsOrig[j*3]-parentPos[0], jointsOrig[j*3+1]-parentPos[1], jointsOrig[j*3+2]-parentPos[2]]
            var node: [String: Any] = ["translation": t, "name": names?[j] ?? "bone_\(j)"]
            if !children[j].isEmpty { node["children"] = children[j] }
            nodes.append(node)
        }
        let rootNode = jointNodeBase + root

        // --- glTF skin ---
        var skins = (json["skins"] as? [[String: Any]]) ?? []
        skins.append(["joints": (0..<J).map { jointNodeBase + $0 }, "inverseBindMatrices": ibmAcc, "skeleton": rootNode])
        let skinIdx = skins.count - 1

        // --- per-primitive JOINTS_0/WEIGHTS_0 (slice the merged weights by primRange) ---
        var riggedMeshes = Set<Int>()
        for pr in primRanges {
            let vo = pr.vertexOffset, vc = pr.vertexCount
            let jslice = Array(skin.joints[(vo*4)..<((vo+vc)*4)])
            let wslice = Array(skin.weights[(vo*4)..<((vo+vc)*4)])
            let jbv = addBufferView(u16(jslice))
            accessors.append(["bufferView": jbv, "componentType": 5123, "count": vc, "type": "VEC4"])
            let jointsAcc = accessors.count - 1
            let wbv = addBufferView(f32(wslice))
            accessors.append(["bufferView": wbv, "componentType": 5126, "count": vc, "type": "VEC4"])
            let weightsAcc = accessors.count - 1
            var prims = meshes[pr.meshIndex]["primitives"] as! [[String: Any]]
            var prim = prims[pr.primIndex]
            var attrs = prim["attributes"] as! [String: Any]
            attrs["JOINTS_0"] = jointsAcc; attrs["WEIGHTS_0"] = weightsAcc
            prim["attributes"] = attrs; prims[pr.primIndex] = prim
            meshes[pr.meshIndex]["primitives"] = prims
            riggedMeshes.insert(pr.meshIndex)
        }

        // bind every node referencing a rigged mesh to the skin; ensure each is in the scene.
        var boundNodes: [Int] = []
        for (i, n) in nodes.enumerated() {
            if let m = n["mesh"] as? Int, riggedMeshes.contains(m) { nodes[i]["skin"] = skinIdx; boundNodes.append(i) }
        }
        if boundNodes.isEmpty, let anyMesh = riggedMeshes.first {   // no node references it → make one
            nodes.append(["mesh": anyMesh, "skin": skinIdx]); boundNodes.append(nodes.count - 1)
        }

        var scenes = (json["scenes"] as? [[String: Any]]) ?? [["nodes": boundNodes]]
        var sceneNodes = (scenes[0]["nodes"] as? [Int]) ?? []
        if !sceneNodes.contains(rootNode) { sceneNodes.append(rootNode) }
        for bn in boundNodes where !sceneNodes.contains(bn) { sceneNodes.append(bn) }
        scenes[0]["nodes"] = sceneNodes

        // --- VRMC_vrm humanoid extension (bone name → authored joint-node index) ---
        // Downstream converters (AnimeStudio vrm2usdz) read this to emit humanoid.json for retargeting;
        // VRM-named nodes alone are NOT read. Only humanBones[bone].node is required.
        if let hb = humanoidBones, !hb.isEmpty {
            var humanBones: [String: Any] = [:]
            for (bone, jointIdx) in hb where jointIdx >= 0 && jointIdx < J {
                humanBones[bone] = ["node": jointNodeBase + jointIdx]
            }
            var ext = (json["extensions"] as? [String: Any]) ?? [:]
            ext["VRMC_vrm"] = ["specVersion": "1.0", "humanoid": ["humanBones": humanBones]]
            json["extensions"] = ext
            var used = (json["extensionsUsed"] as? [String]) ?? []
            if !used.contains("VRMC_vrm") { used.append("VRMC_vrm") }
            json["extensionsUsed"] = used
        }

        // reassemble JSON
        buffers[0]["byteLength"] = bin.count
        json["buffers"] = buffers
        json["bufferViews"] = bufferViews
        json["accessors"] = accessors
        json["nodes"] = nodes
        json["meshes"] = meshes
        json["skins"] = skins
        json["scenes"] = scenes

        // --- write GLB ---
        try Self.writeGLB(json: json, bin: bin, to: outPath)
    }

    /// Route-B (S5) rig injection: REUSE the VRM's existing joint nodes (no new skeleton) and write a
    /// fresh skin whose weights come from the model. `jointNodes` = existing glTF node index per skin
    /// column (in the tokenized joint order); `inverseBind` = J×16 column-major IBM (proper inverse of
    /// each joint's world bind, rotation-aware — NOT the auto-path translation-only bind). Per-primitive
    /// JOINTS_0 columns index directly into `jointNodes`. Mesh vertices/materials/UVs are untouched.
    public static func injectRigRouteB(glbPath: String, outPath: String,
                                       jointNodes: [Int], inverseBind: [Float], skeletonRoot: Int,
                                       skin: SkinWeights, primRanges: [PrimRange]) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: glbPath))
        func u32(_ off: Int) -> UInt32 { data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: UInt32.self) } }
        let jsonLen = Int(u32(12)); let jsonStart = 20
        var json = try JSONSerialization.jsonObject(with: data.subdata(in: jsonStart..<jsonStart+jsonLen)) as! [String: Any]
        var p = jsonStart + jsonLen; p = (p + 3) & ~3
        let binLen = Int(u32(p)); let binStart = p + 8
        var bin = data.subdata(in: binStart..<binStart + binLen)

        var buffers = json["buffers"] as! [[String: Any]]
        var bufferViews = (json["bufferViews"] as? [[String: Any]]) ?? []
        var accessors = (json["accessors"] as? [[String: Any]]) ?? []
        var nodes = json["nodes"] as! [[String: Any]]
        var meshes = json["meshes"] as! [[String: Any]]
        let J = jointNodes.count

        func addBufferView(_ bytes: Data) -> Int {
            while bin.count % 4 != 0 { bin.append(0) }
            let off = bin.count; bin.append(bytes)
            bufferViews.append(["buffer": 0, "byteOffset": off, "byteLength": bytes.count])
            return bufferViews.count - 1
        }
        func f32(_ a: [Float]) -> Data { a.withUnsafeBytes { Data($0) } }
        func u16(_ a: [UInt16]) -> Data { a.withUnsafeBytes { Data($0) } }

        // IBM accessor (J × MAT4, column-major, provided).
        let ibmAcc: Int = { let bv = addBufferView(f32(inverseBind))
            accessors.append(["bufferView": bv, "componentType": 5126, "count": J, "type": "MAT4"]); return accessors.count - 1 }()

        // new skin reusing existing joint nodes.
        var skins = (json["skins"] as? [[String: Any]]) ?? []
        skins.append(["joints": jointNodes, "inverseBindMatrices": ibmAcc, "skeleton": skeletonRoot])
        let skinIdx = skins.count - 1

        // per-primitive JOINTS_0 (column idx) / WEIGHTS_0.
        var riggedMeshes = Set<Int>()
        for pr in primRanges {
            let vo = pr.vertexOffset, vc = pr.vertexCount
            let jbv = addBufferView(u16(Array(skin.joints[(vo*4)..<((vo+vc)*4)])))
            accessors.append(["bufferView": jbv, "componentType": 5123, "count": vc, "type": "VEC4"])
            let jointsAcc = accessors.count - 1
            let wbv = addBufferView(f32(Array(skin.weights[(vo*4)..<((vo+vc)*4)])))
            accessors.append(["bufferView": wbv, "componentType": 5126, "count": vc, "type": "VEC4"])
            let weightsAcc = accessors.count - 1
            var prims = meshes[pr.meshIndex]["primitives"] as! [[String: Any]]
            var prim = prims[pr.primIndex]; var attrs = prim["attributes"] as! [String: Any]
            attrs["JOINTS_0"] = jointsAcc; attrs["WEIGHTS_0"] = weightsAcc
            prim["attributes"] = attrs; prims[pr.primIndex] = prim
            meshes[pr.meshIndex]["primitives"] = prims
            riggedMeshes.insert(pr.meshIndex)
        }
        // re-point every rigged mesh's node to the new skin.
        for (i, n) in nodes.enumerated() {
            if let m = n["mesh"] as? Int, riggedMeshes.contains(m) { nodes[i]["skin"] = skinIdx }
        }

        buffers[0]["byteLength"] = bin.count
        json["buffers"] = buffers; json["bufferViews"] = bufferViews; json["accessors"] = accessors
        json["nodes"] = nodes; json["meshes"] = meshes; json["skins"] = skins
        try Self.writeGLB(json: json, bin: bin, to: outPath)
    }

    /// Assemble + write a binary glTF (JSON + BIN chunks, 4-byte aligned).
    static func writeGLB(json: [String: Any], bin binIn: Data, to outPath: String) throws {
        var bin = binIn
        var jsonData = try JSONSerialization.data(withJSONObject: json)
        while jsonData.count % 4 != 0 { jsonData.append(0x20) }   // pad JSON with spaces
        while bin.count % 4 != 0 { bin.append(0) }
        var out = Data()
        func appendU32(_ v: UInt32) { var x = v; out.append(Data(bytes: &x, count: 4)) }
        appendU32(0x46546C67); appendU32(2); appendU32(UInt32(12 + 8 + jsonData.count + 8 + bin.count))
        appendU32(UInt32(jsonData.count)); appendU32(0x4E4F534A)   // "JSON"
        out.append(jsonData)
        appendU32(UInt32(bin.count)); appendU32(0x004E4942)        // "BIN\0"
        out.append(bin)
        try out.write(to: URL(fileURLWithPath: outPath))
    }
}
