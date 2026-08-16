import Foundation
import simd
import SkinTokensCore
import SkinTokensGLTF

/// Route-B (S5) VRM I/O bridge: parse a VRM via the vendored `GLTFKit`, extract (a) the merged mesh in
/// global bind space (all skinned primitives' POSITION are already the rendered bind-pose positions,
/// which share the joint world-position space — so no per-vertex node transform is applied) and (b) the
/// EXISTING skeleton (joint world positions + parents + node indices). Route-B feeds this skeleton to
/// the model as the prefix and predicts skin only, then writes weights onto these existing joints.
public enum VRMLoad {

    public struct Skeleton: Sendable {
        public var joints: [[Float]]        // (J,3) world/bind positions, model space
        public var parents: [Int]           // (J) parent joint index, -1 for root
        public var jointNodes: [Int]        // (J) glTF node index of each joint (for JOINTS_0 mapping)
        public var jointNames: [String]     // (J) glTF node name of each joint ("" if unnamed)
        public var bindWorld: [[Double]]    // (J) world bind matrix, 16 col-major (simd flatten) per joint
        public var humanoid: [String: Int]  // VRM bone name → joint index (subset)
        public var J: Int { joints.count }
        public var rootNode: Int { jointNodes[parents.firstIndex(of: -1) ?? 0] }

        /// Column-major inverse-bind matrices (J×16, Float) = inverse(worldBind) — rotation-aware.
        public func inverseBind() -> [Float] {
            var out = [Float](repeating: 0, count: J * 16)
            for j in 0..<J {
                let b = bindWorld[j]
                let m = simd_double4x4(SIMD4(b[0], b[1], b[2], b[3]), SIMD4(b[4], b[5], b[6], b[7]),
                                       SIMD4(b[8], b[9], b[10], b[11]), SIMD4(b[12], b[13], b[14], b[15]))
                let inv = m.inverse
                let cols = [inv.columns.0, inv.columns.1, inv.columns.2, inv.columns.3]
                for c in 0..<4 { for r in 0..<4 { out[j*16 + c*4 + r] = Float(cols[c][r]) } }
            }
            return out
        }
    }

    /// Load a VRM/glTF from a file path.
    public static func load(_ path: String) throws -> (mesh: Mesh, skeleton: Skeleton, asset: GLTFAsset) {
        try load(asset: try GLTFAsset.load(url: URL(fileURLWithPath: path)))
    }

    /// Load a VRM/glTF from GLB bytes (the engine passes `Mesh.data`).
    public static func load(data: Data) throws -> (mesh: Mesh, skeleton: Skeleton, asset: GLTFAsset) {
        try load(asset: try GLTFAsset.load(data: data))
    }

    /// Core: merged `Mesh` (all skinned primitives) + the existing `Skeleton`.
    static func load(asset: GLTFAsset) throws -> (mesh: Mesh, skeleton: Skeleton, asset: GLTFAsset) {
        let doc = asset.document
        guard let meshes = doc.meshes else { throw Err.noMesh }

        // --- merged mesh across all primitives of all meshes (raw POSITION = global bind space) ---
        var vertices: [Float] = [], normals: [Float] = [], faces: [Int] = []
        var primRanges: [PrimRange] = []
        for (mi, mesh) in meshes.enumerated() {
            for (pi, prim) in mesh.primitives.enumerated() {
                guard prim.attributes["POSITION"] != nil else { continue }
                let vOffset = vertices.count / 3
                let pos = try asset.decoder.readVec3(accessorIndex: prim.attributes["POSITION"]!)
                for v in pos { vertices.append(v.x); vertices.append(v.y); vertices.append(v.z) }
                // normals
                if let nAcc = prim.attributes["NORMAL"] {
                    let ns = try asset.decoder.readVec3(accessorIndex: nAcc)
                    for n in ns { normals.append(n.x); normals.append(n.y); normals.append(n.z) }
                } else {
                    normals.append(contentsOf: [Float](repeating: 0, count: pos.count * 3))
                }
                // indices
                if let iAcc = prim.indices {
                    let idx = try asset.decoder.readIndices(accessorIndex: iAcc)
                    for i in idx { faces.append(Int(i) + vOffset) }
                } else {
                    for i in 0..<pos.count { faces.append(i + vOffset) }
                }
                primRanges.append(PrimRange(meshIndex: mi, primIndex: pi, vertexOffset: vOffset, vertexCount: pos.count))
            }
        }
        // fill any computed normals for primitives that lacked NORMAL (rare on VRM).
        if normals.contains(where: { $0.isNaN }) || normals.allSatisfy({ $0 == 0 }) {
            normals = MeshIO.computeVertexNormals(vertices: vertices, faces: faces)
        }
        let mesh = Mesh(vertices: vertices, faces: faces, normals: normals, primRanges: primRanges)

        // --- existing skeleton via GLTFKit (world/bind transforms, node hierarchy) ---
        guard let skel = asset.buildSkeleton() else { throw Err.noSkeleton }
        let J = skel.joints.count
        // joint index → node index (invert nodeToJoint).
        var jointNodes = [Int](repeating: -1, count: J)
        for (node, j) in skel.nodeToJoint { jointNodes[j] = node }
        // world positions + full world bind matrix (bindRowMajor = simd column flatten; t at 12,13,14).
        var joints = [[Float]](repeating: [0, 0, 0], count: J)
        var bindWorld = [[Double]](repeating: [], count: J)
        for j in 0..<J {
            let b = skel.joints[j].bindRowMajor
            joints[j] = [Float(b[12]), Float(b[13]), Float(b[14])]
            bindWorld[j] = b
        }
        // parents (joint-index space) from the node hierarchy: parent node of each joint node,
        // walking up until we hit another joint node.
        var parentNode = [Int: Int]()
        if let nodes = doc.nodes {
            for (i, n) in nodes.enumerated() { for c in (n.children ?? []) { parentNode[c] = i } }
        }
        var parents = [Int](repeating: -1, count: J)
        for j in 0..<J {
            var p = parentNode[jointNodes[j]]
            while let pp = p, skel.nodeToJoint[pp] == nil { p = parentNode[pp] }
            parents[j] = p.flatMap { skel.nodeToJoint[$0] } ?? -1
        }
        // per-joint node names (for skeleton cleaning: identify Root / *_end / non-deforming bones).
        var jointNames = [String](repeating: "", count: J)
        if let nodes = doc.nodes {
            for j in 0..<J { let n = jointNodes[j]; if n >= 0 && n < nodes.count { jointNames[j] = nodes[n].name ?? "" } }
        }
        // humanoid bone name → joint index
        var humanoid = [String: Int]()
        for (name, node) in asset.humanoidBones() { if let j = skel.nodeToJoint[node] { humanoid[name] = j } }

        return (mesh, Skeleton(joints: joints, parents: parents, jointNodes: jointNodes,
                               jointNames: jointNames, bindWorld: bindWorld, humanoid: humanoid), asset)
    }

    enum Err: Swift.Error, CustomStringConvertible {
        case noMesh, noSkeleton
        var description: String { self == .noMesh ? "glTF has no meshes" : "glTF has no skeleton/skins" }
    }
}
