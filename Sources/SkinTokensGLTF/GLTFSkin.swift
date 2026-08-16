import Foundation
import simd

/// Skeletal rig extracted from a glTF: a joint hierarchy with rest (local) and
/// bind (world) transforms, plus helpers to resolve a skin's joints to USD joint
/// paths. Bind transforms are computed by forward kinematics from the rest pose
/// (rest == bind for VRoid avatars), which avoids inverse-bind-matrix mismatches.
public struct GLTFSkeleton: Sendable {
    public struct Joint: Sendable {
        /// USD joint token — a path relative to the skeleton root, e.g.
        /// "Hips/Spine/Chest".
        public let path: String
        /// Local (rest) transform, row-major 4×4 (USD GfMatrix4d convention).
        public let restRowMajor: [Double]
        /// World (bind) transform, row-major 4×4.
        public let bindRowMajor: [Double]
    }
    public let joints: [Joint]
    /// glTF node index → index within `joints`.
    public let nodeToJoint: [Int: Int]

    /// USD joint paths for a skin's joints, in the skin's own order (for a mesh's
    /// `skel:joints`). nil if any joint node isn't in the skeleton.
    public func jointPaths(forSkinJoints nodes: [Int]) -> [String]? {
        var out: [String] = []
        for n in nodes {
            guard let j = nodeToJoint[n] else { return nil }
            out.append(joints[j].path)
        }
        return out
    }
}

extension GLTFAsset {

    /// Local TRS matrix of a node (column-major simd).
    func localMatrix(_ node: GLTFDocument.Node) -> simd_double4x4 {
        if let m = node.matrix, m.count == 16 {
            return simd_double4x4(
                SIMD4(Double(m[0]), Double(m[1]), Double(m[2]), Double(m[3])),
                SIMD4(Double(m[4]), Double(m[5]), Double(m[6]), Double(m[7])),
                SIMD4(Double(m[8]), Double(m[9]), Double(m[10]), Double(m[11])),
                SIMD4(Double(m[12]), Double(m[13]), Double(m[14]), Double(m[15])))
        }
        let t = node.translation ?? [0, 0, 0]
        let r = node.rotation ?? [0, 0, 0, 1]   // xyzw
        let s = node.scale ?? [1, 1, 1]
        let T = SIMD3(Double(t[0]), Double(t[1]), Double(t[2]))
        let q = simd_quatd(ix: Double(r[0]), iy: Double(r[1]), iz: Double(r[2]), r: Double(r[3]))
        let S = SIMD3(Double(s[0]), Double(s[1]), Double(s[2]))
        var m = simd_double4x4(q)                 // rotation
        m.columns.0 *= S.x; m.columns.1 *= S.y; m.columns.2 *= S.z   // * scale
        m.columns.3 = SIMD4(T, 1)                 // translation
        return m
    }

    /// VRM humanoid bone name → glTF node index (both 1.0 and 0.x).
    public func humanoidBones() -> [String: Int] {
        guard let ext = document.extensions else { return [:] }
        var out: [String: Int] = [:]
        if let hb = ext["VRMC_vrm"]?["humanoid"]?["humanBones"]?.objectValue {        // VRM 1.0
            for (k, v) in hb { if let n = v["node"]?.intValue { out[k] = n } }
        } else if let bones = ext["VRM"]?["humanoid"]?["humanBones"]?.arrayValue {     // VRM 0.x
            for b in bones { if let name = b["bone"]?.stringValue, let n = b["node"]?.intValue { out[name] = n } }
        }
        return out
    }

    /// Builds the skeleton from the union of all skins' joints. Bind transforms
    /// come from forward kinematics over the rest pose. When `relaxArmsRadians` is
    /// set, the upper-arm joints' REST transforms are rotated down about world-Z
    /// (bind stays at T-pose) so the skinned avatar rests with arms lowered.
    public func buildSkeleton(relaxArmsRadians: Double? = nil,
                              forwardArmsRadians: Double = 0) -> GLTFSkeleton? {
        guard let nodes = document.nodes, let skins = document.skins, !skins.isEmpty else { return nil }

        // Parent map.
        var parent = [Int: Int]()
        for (i, n) in nodes.enumerated() {
            for c in n.children ?? [] { parent[c] = i }
        }

        // Joint set = union of all skins' joints + every ancestor up to a root,
        // so the hierarchy is complete (UsdSkel needs each joint's parent present).
        var jointSet = Set<Int>()
        for skin in skins { for j in skin.joints { jointSet.insert(j) } }
        for j in Array(jointSet) {
            var p = parent[j]
            while let pp = p, !jointSet.contains(pp) { jointSet.insert(pp); p = parent[pp] }
        }

        // World transforms via memoized FK.
        var worldCache = [Int: simd_double4x4]()
        func world(_ node: Int) -> simd_double4x4 {
            if let w = worldCache[node] { return w }
            let local = localMatrix(nodes[node])
            let w: simd_double4x4 = parent[node].map { world($0) * local } ?? local
            worldCache[node] = w
            return w
        }

        // Sanitized, unique node names for path building.
        var nameFor = [Int: String]()
        var usedNames = Set<String>()
        func name(_ node: Int) -> String {
            if let n = nameFor[node] { return n }
            var base = USDName.sanitize(nodes[node].name ?? "joint_\(node)")
            var unique = base
            var k = 1
            while usedNames.contains(unique) { unique = "\(base)_\(k)"; k += 1 }
            usedNames.insert(unique); nameFor[node] = unique
            return unique
        }

        // Joint path = chain of names from the topmost joint ancestor down.
        func path(_ node: Int) -> String {
            var chain = [name(node)]
            var p = parent[node]
            while let pp = p, jointSet.contains(pp) { chain.append(name(pp)); p = parent[pp] }
            return chain.reversed().joined(separator: "/")
        }

        // Topological order (parents before children) by hierarchy depth.
        func depth(_ node: Int) -> Int {
            var d = 0; var p = parent[node]
            while let pp = p, jointSet.contains(pp) { d += 1; p = parent[pp] }
            return d
        }
        let ordered = jointSet.sorted { depth($0) != depth($1) ? depth($0) < depth($1) : $0 < $1 }

        // Optional rest-pose tweak: lower the arms (world-Z, left −θ/right +θ) and
        // splay them forward (world-X) so the hands rest in front of the skirt
        // rather than buried in its sides. Pivot at the shoulder; bind stays T-pose;
        // the result is baked into the joint's LOCAL rest transform.
        var restOverride = [Int: simd_double4x4]()
        if let theta = relaxArmsRadians {
            let bones = humanoidBones()
            let fwd = Self.rotationX(-forwardArmsRadians)   // negative X → arms toward +Z (forward)
            for (bone, sign) in [("leftUpperArm", -1.0), ("rightUpperArm", 1.0)] {
                guard let node = bones[bone], jointSet.contains(node) else { continue }
                let wb = world(node)
                let pivot = wb.columns.3
                var posed = fwd * Self.rotationZ(sign * theta) * wb
                posed.columns.3 = pivot
                let parentWorld = parent[node].map { world($0) } ?? matrix_identity_double4x4
                restOverride[node] = parentWorld.inverse * posed
            }
        }

        var joints: [GLTFSkeleton.Joint] = []
        var nodeToJoint = [Int: Int]()
        for node in ordered {
            nodeToJoint[node] = joints.count
            let restLocal = restOverride[node] ?? localMatrix(nodes[node])
            joints.append(GLTFSkeleton.Joint(
                path: path(node),
                restRowMajor: Self.rowMajor(restLocal),
                bindRowMajor: Self.rowMajor(world(node))))
        }
        return GLTFSkeleton(joints: joints, nodeToJoint: nodeToJoint)
    }

    /// Rotation about +Z by `theta` (column-major simd).
    static func rotationZ(_ theta: Double) -> simd_double4x4 {
        let c = cos(theta), s = sin(theta)
        return simd_double4x4(SIMD4(c, s, 0, 0), SIMD4(-s, c, 0, 0),
                              SIMD4(0, 0, 1, 0), SIMD4(0, 0, 0, 1))
    }

    /// Rotation about +X by `theta` (column-major simd).
    static func rotationX(_ theta: Double) -> simd_double4x4 {
        let c = cos(theta), s = sin(theta)
        return simd_double4x4(SIMD4(1, 0, 0, 0), SIMD4(0, c, s, 0),
                              SIMD4(0, -s, c, 0), SIMD4(0, 0, 0, 1))
    }

    /// simd (column-major, column-vector) → USD GfMatrix4d row-major flat array.
    /// USD uses the row-vector convention, so M_usd = M_simd^T; storing that
    /// transpose row-major is identical to flattening simd's columns directly
    /// (translation lands at indices 12,13,14, as USD expects).
    static func rowMajor(_ m: simd_double4x4) -> [Double] {
        [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }
}

/// USD prim/joint name sanitization (shared with the authoring layer's rules).
enum USDName {
    static func sanitize(_ name: String) -> String {
        var out = ""
        for (i, ch) in name.enumerated() {
            if ch.isLetter || ch == "_" || (ch.isNumber && i > 0) { out.append(ch) } else { out.append("_") }
        }
        if let first = out.first, first.isNumber { out = "_" + out }
        return out.isEmpty ? "_" : out
    }
}
