import Foundation

/// S1a — mesh GLB I/O + predict-transform + `SamplerMix`. Owns mesh I/O directly (does NOT go through
/// bpy — the Swift port propagates skin onto the ORIGINAL glTF render-verts, so it must read them; see
/// HANDOFF vertex-identity note). Pure Foundation + a minimal binary-glTF (`.glb`) accessor reader for
/// POSITION / NORMAL / indices; normals are computed area-weighted when the file omits them.
///
/// The parity gate here is count + distribution (NOT element-match): the query-point selection the
/// encoders use is `numpy.default_rng(seed=0).choice` + torch-geometric FPS, which we reproduce
/// STRUCTURALLY (a deterministic subset + exact farthest-point sampling) but not bit-for-bit — the
/// injected-points path is already validated by S1b–S2 PASSING on the captured cloud, and generation
/// is non-reproducible regardless (P0). A different-but-valid query set yields a different-but-valid rig.
public struct Mesh: Sendable {
    public var vertices: [Float]     // flat [N*3]
    public var faces: [Int]          // flat [F*3] (triangle vertex indices, into the merged vertex set)
    public var normals: [Float]      // flat [N*3], unit
    /// Where each source glTF primitive's vertices live in the merged set — so S4 can write each
    /// primitive its OWN JOINTS_0/WEIGHTS_0 slice (a mesh may have many primitives / a VRM many meshes).
    public var primRanges: [PrimRange]
    public var vertexCount: Int { vertices.count / 3 }
    public var faceCount: Int { faces.count / 3 }
    public init(vertices: [Float], faces: [Int], normals: [Float], primRanges: [PrimRange] = []) {
        self.vertices = vertices; self.faces = faces; self.normals = normals; self.primRanges = primRanges
    }
}

public struct PrimRange: Sendable {
    public var meshIndex: Int
    public var primIndex: Int
    public var vertexOffset: Int    // start of this primitive's verts in the merged set
    public var vertexCount: Int
    public init(meshIndex: Int, primIndex: Int, vertexOffset: Int, vertexCount: Int) {
        self.meshIndex = meshIndex; self.primIndex = primIndex; self.vertexOffset = vertexOffset; self.vertexCount = vertexCount
    }
}

public enum MeshIO {

    // MARK: - GLB parse

    enum GLBError: Swift.Error, CustomStringConvertible {
        case badMagic, noBIN, missingAttr(String), unsupported(String)
        public var description: String {
            switch self {
            case .badMagic: return "not a binary glTF (bad magic)"
            case .noBIN: return "glb has no BIN chunk"
            case .missingAttr(let a): return "primitive missing \(a)"
            case .unsupported(let s): return "unsupported glTF: \(s)"
            }
        }
    }

    /// Read + MERGE every primitive of every mesh in a `.glb`: POSITION, indices, NORMAL (computed if
    /// absent). Vertices/faces/normals are concatenated with per-primitive index offsets, and
    /// `primRanges` records where each source primitive landed (so S4 rigs the WHOLE model, not just
    /// prim 0). Matches the oracle's bpy-merged Asset for sampling coverage.
    public static func loadGLB(_ path: String) throws -> Mesh {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        func u32(_ off: Int) -> UInt32 { data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: UInt32.self) } }
        guard data.count >= 12, u32(0) == 0x46546C67 else { throw GLBError.badMagic }   // "glTF"
        let jsonLen = Int(u32(12))
        let jsonStart = 20
        let json = try JSONSerialization.jsonObject(with: data.subdata(in: jsonStart..<jsonStart+jsonLen)) as! [String: Any]
        var p = jsonStart + jsonLen; p = (p + 3) & ~3
        guard p + 8 <= data.count else { throw GLBError.noBIN }
        let binLen = Int(u32(p)); let binType = u32(p + 4)
        guard binType == 0x004E4942 else { throw GLBError.noBIN }                        // "BIN\0"
        let binStart = p + 8
        let bin = data.subdata(in: binStart..<binStart + binLen)

        let accessors = json["accessors"] as! [[String: Any]]
        let bufferViews = json["bufferViews"] as! [[String: Any]]
        let meshes = json["meshes"] as! [[String: Any]]

        func readAccessor(_ ai: Int) -> (comp: Int, type: String, count: Int, byteOffset: Int, stride: Int) {
            let a = accessors[ai]
            let bv = bufferViews[a["bufferView"] as! Int]
            let bvOff = (bv["byteOffset"] as? Int) ?? 0
            let accOff = (a["byteOffset"] as? Int) ?? 0
            let comp = a["componentType"] as! Int
            let type = a["type"] as! String
            let compCount = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][type]!
            let compSize = [5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4][comp]!
            let stride = (bv["byteStride"] as? Int) ?? (compCount * compSize)
            return (comp, type, a["count"] as! Int, bvOff + accOff, stride)
        }

        var vertices: [Float] = [], normals: [Float] = [], faces: [Int] = []
        var primRanges: [PrimRange] = []
        for (mi, mesh) in meshes.enumerated() {
            let prims = mesh["primitives"] as! [[String: Any]]
            for (pi, prim) in prims.enumerated() {
                let attrs = prim["attributes"] as! [String: Any]
                guard let posAI = attrs["POSITION"] as? Int else { continue }
                let vOffset = vertices.count / 3
                let pos = readAccessor(posAI)
                var vlocal = [Float](repeating: 0, count: pos.count * 3)
                bin.withUnsafeBytes { (rp: UnsafeRawBufferPointer) in
                    for i in 0..<pos.count { let base = pos.byteOffset + i * pos.stride
                        for k in 0..<3 { vlocal[i*3+k] = rp.loadUnaligned(fromByteOffset: base + k*4, as: Float.self) } }
                }
                // normals
                var nlocal: [Float]
                if let nAI = attrs["NORMAL"] as? Int {
                    let n = readAccessor(nAI)
                    nlocal = [Float](repeating: 0, count: n.count * 3)
                    bin.withUnsafeBytes { (rp: UnsafeRawBufferPointer) in
                        for i in 0..<n.count { let base = n.byteOffset + i * n.stride
                            for k in 0..<3 { nlocal[i*3+k] = rp.loadUnaligned(fromByteOffset: base + k*4, as: Float.self) } }
                    }
                    normalizeInPlace(&nlocal)
                } else { nlocal = [Float](repeating: 0, count: pos.count * 3) }   // fill after faces
                // indices (offset into merged set)
                var flocal: [Int] = []
                if let idxAI = prim["indices"] as? Int {
                    let idx = readAccessor(idxAI); flocal.reserveCapacity(idx.count)
                    bin.withUnsafeBytes { (rp: UnsafeRawBufferPointer) in
                        for i in 0..<idx.count { let off = idx.byteOffset + i * idx.stride
                            let v = idx.comp == 5125 ? Int(rp.loadUnaligned(fromByteOffset: off, as: UInt32.self))
                                                     : Int(rp.loadUnaligned(fromByteOffset: off, as: UInt16.self))
                            flocal.append(v + vOffset) }
                    }
                } else { flocal = (0..<pos.count).map { $0 + vOffset } }
                if attrs["NORMAL"] == nil {
                    // compute local normals from local faces (indices are already merged-offset).
                    let ln = computeVertexNormals(vertices: vlocal, faces: flocal.map { $0 - vOffset })
                    nlocal = ln
                }
                vertices.append(contentsOf: vlocal)
                normals.append(contentsOf: nlocal)
                faces.append(contentsOf: flocal)
                primRanges.append(PrimRange(meshIndex: mi, primIndex: pi, vertexOffset: vOffset, vertexCount: pos.count))
            }
        }
        return Mesh(vertices: vertices, faces: faces, normals: normals, primRanges: primRanges)
    }

    /// Area-weighted vertex normals (unnormalized face normal = 2×area × unit → natural area weight).
    public static func computeVertexNormals(vertices: [Float], faces: [Int]) -> [Float] {
        let N = vertices.count / 3
        var normals = [Float](repeating: 0, count: N * 3)
        func v(_ i: Int, _ k: Int) -> Float { vertices[i * 3 + k] }
        var f = 0
        while f < faces.count {
            let a = faces[f], b = faces[f + 1], c = faces[f + 2]; f += 3
            let e0 = (v(b,0)-v(a,0), v(b,1)-v(a,1), v(b,2)-v(a,2))
            let e1 = (v(c,0)-v(a,0), v(c,1)-v(a,1), v(c,2)-v(a,2))
            let nx = e0.1*e1.2 - e0.2*e1.1, ny = e0.2*e1.0 - e0.0*e1.2, nz = e0.0*e1.1 - e0.1*e1.0
            for j in [a, b, c] { normals[j*3] += nx; normals[j*3+1] += ny; normals[j*3+2] += nz }
        }
        normalizeInPlace(&normals)
        return normals
    }

    static func normalizeInPlace(_ n: inout [Float]) {
        let N = n.count / 3
        for i in 0..<N {
            let x = n[i*3], y = n[i*3+1], z = n[i*3+2]
            let len = (x*x + y*y + z*z).squareRoot()
            if len > 1e-12 { n[i*3] = x/len; n[i*3+1] = y/len; n[i*3+2] = z/len }
            else { n[i*3] = 0; n[i*3+1] = 0; n[i*3+2] = 1 }
        }
    }

    // MARK: - predict-transform (affine → [-1,1]³)

    /// The recorded similarity transform (uniform): `v' = (v - center) / scale`. Invert on joints (S4)
    /// to place the armature back in original mesh space. Mirrors `AugmentAffine` (predict has no joints,
    /// bias 0 for [-1,1]): center = (min+max)/2, scale = max_extent / 2.
    public struct Affine: Sendable {
        public var center: [Float]; public var scale: Float
        public init(center: [Float], scale: Float) { self.center = center; self.scale = scale }
    }

    public static func predictAffine(_ mesh: Mesh) -> Affine {
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3)
        var hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        let N = mesh.vertexCount
        for i in 0..<N { for k in 0..<3 { let x = mesh.vertices[i*3+k]; lo[k] = min(lo[k], x); hi[k] = max(hi[k], x) } }
        let center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]
        let scale = max(hi[0]-lo[0], max(hi[1]-lo[1], hi[2]-lo[2])) / 2   // normalize_into span = 2
        return Affine(center: center, scale: scale == 0 ? 1 : scale)
    }

    /// Apply the affine to a flat [M*3] point array in place (normals are scale/translate-invariant).
    public static func applyAffine(_ pts: [Float], _ a: Affine) -> [Float] {
        var out = pts
        let M = pts.count / 3
        for i in 0..<M { for k in 0..<3 { out[i*3+k] = (pts[i*3+k] - a.center[k]) / a.scale } }
        return out
    }

    // MARK: - deterministic RNG (SplitMix64 — not numpy-PCG64, sampler gate is count+distribution)

    struct RNG {
        var state: UInt64
        init(_ seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func uniform() -> Float { Float(next() >> 40) * (1.0 / 16777216.0) }   // [0,1)
        mutating func int(_ n: Int) -> Int { Int(next() % UInt64(n)) }
    }

    // MARK: - SamplerMix (54000 = 16384 vertex-samples + area-weighted rest)

    public struct SampleCloud: Sendable { public var points: [Float]; public var normals: [Float]; public var count: Int }

    public static func samplerMix(_ mesh: Mesh, numSamples: Int = 54000, numVertexSamples: Int = 16384,
                                  seed: UInt64 = 0) -> SampleCloud {
        var rng = RNG(seed)
        let N = mesh.vertexCount
        // vertex samples: a partial Fisher–Yates permutation → up to numVertexSamples unique verts.
        var perm = Array(0..<N)
        let nv = min(numVertexSamples, N)
        for i in 0..<nv { let j = i + rng.int(N - i); perm.swapAt(i, j) }
        let vIdx = Array(perm[0..<nv])
        // area samples: cumulative face-area CDF + barycentric.
        let F = mesh.faceCount
        var cum = [Float](repeating: 0, count: F)
        func v(_ i: Int, _ k: Int) -> Float { mesh.vertices[i*3+k] }
        var run: Float = 0
        for fi in 0..<F {
            let a = mesh.faces[fi*3], b = mesh.faces[fi*3+1], c = mesh.faces[fi*3+2]
            let e0 = (v(b,0)-v(a,0), v(b,1)-v(a,1), v(b,2)-v(a,2))
            let e1 = (v(c,0)-v(a,0), v(c,1)-v(a,1), v(c,2)-v(a,2))
            let cx = e0.1*e1.2 - e0.2*e1.1, cy = e0.2*e1.0 - e0.0*e1.2, cz = e0.0*e1.1 - e0.1*e1.0
            run += (cx*cx + cy*cy + cz*cz).squareRoot()   // 2×triangle area (weight)
            cum[fi] = run
        }
        let total = max(run, 1e-12)
        func pickFace(_ r: Float) -> Int {   // searchsorted on cum
            var lo = 0, hi = F - 1
            let target = r * total
            while lo < hi { let mid = (lo + hi) / 2; if cum[mid] < target { lo = mid + 1 } else { hi = mid } }
            return lo
        }
        let nArea = numSamples - nv
        var points = [Float](repeating: 0, count: numSamples * 3)
        var norms = [Float](repeating: 0, count: numSamples * 3)
        // block 1: vertex samples
        for (i, vi) in vIdx.enumerated() {
            for k in 0..<3 { points[i*3+k] = mesh.vertices[vi*3+k]; norms[i*3+k] = mesh.normals[vi*3+k] }
        }
        // block 2: area samples
        for s in 0..<nArea {
            let fi = pickFace(rng.uniform())
            let a = mesh.faces[fi*3], b = mesh.faces[fi*3+1], c = mesh.faces[fi*3+2]
            var u = rng.uniform(), w = rng.uniform()
            if u + w > 1 { u = 1 - u; w = 1 - w }   // reflect into triangle (trimesh convention)
            let dst = (nv + s) * 3
            for k in 0..<3 {
                let p = v(a,k) + u * (v(b,k) - v(a,k)) + w * (v(c,k) - v(a,k))
                points[dst+k] = p
            }
            // face normal (unit)
            let e0 = (v(b,0)-v(a,0), v(b,1)-v(a,1), v(b,2)-v(a,2))
            let e1 = (v(c,0)-v(a,0), v(c,1)-v(a,1), v(c,2)-v(a,2))
            var fn = [e0.1*e1.2 - e0.2*e1.1, e0.2*e1.0 - e0.0*e1.2, e0.0*e1.1 - e0.1*e1.0]
            let len = (fn[0]*fn[0]+fn[1]*fn[1]+fn[2]*fn[2]).squareRoot()
            if len > 1e-12 { fn = [fn[0]/len, fn[1]/len, fn[2]/len] } else { fn = [0,0,1] }
            for k in 0..<3 { norms[dst+k] = fn[k] }
        }
        return SampleCloud(points: points, normals: norms, count: numSamples)
    }

    // MARK: - FPS query selection (subset → farthest-point sampling)

    /// Reproduce the encoder query pick STRUCTURALLY: a deterministic `subset` of the cloud, then
    /// exact farthest-point sampling down to `k` (random_start=false → start at index 0). Returns the
    /// `k` selected indices INTO the original cloud.
    public static func fpsQuery(points: [Float], count: Int, subset: Int, k: Int, seed: UInt64 = 0) -> [Int] {
        var rng = RNG(seed)
        // subset: partial permutation of the cloud.
        var perm = Array(0..<count)
        let m = min(subset, count)
        for i in 0..<m { let j = i + rng.int(count - i); perm.swapAt(i, j) }
        let sub = Array(perm[0..<m])
        // FPS over the subset, start at sub[0].
        var selected = [Int]()
        selected.reserveCapacity(k)
        var dist = [Float](repeating: .greatestFiniteMagnitude, count: m)
        var cur = 0
        for _ in 0..<min(k, m) {
            selected.append(sub[cur])
            let cx = points[sub[cur]*3], cy = points[sub[cur]*3+1], cz = points[sub[cur]*3+2]
            var far = 0; var farD: Float = -1
            for i in 0..<m {
                let px = points[sub[i]*3], py = points[sub[i]*3+1], pz = points[sub[i]*3+2]
                let d = (px-cx)*(px-cx) + (py-cy)*(py-cy) + (pz-cz)*(pz-cz)
                if d < dist[i] { dist[i] = d }
                if dist[i] > farD { farD = dist[i]; far = i }
            }
            cur = far
        }
        return selected
    }
}
