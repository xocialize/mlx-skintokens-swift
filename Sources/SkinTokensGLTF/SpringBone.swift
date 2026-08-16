import Foundation

/// VRM secondary-physics (`VRMC_springBone` 1.0 / `VRM.secondaryAnimation` 0.x)
/// extracted as joint *chains* with per-joint parameters, resolved to USD joint
/// paths so a runtime can simulate hair/cloth sway. Colliders are parsed but the
/// v1 runtime sim ignores collision (sway only).
public struct SpringBoneRig: Codable, Sendable {
    public var springs: [Spring]

    public struct Spring: Codable, Sendable {
        public var name: String
        /// Joint chain (USD joint paths), root → tip.
        public var joints: [String]
        /// Per-joint parameters, parallel to `joints`.
        public var stiffness: [Float]
        public var dragForce: [Float]
        public var gravityPower: [Float]
        /// Shared gravity direction (x,y,z).
        public var gravityDir: [Float]
        public var hitRadius: [Float]
        /// Sphere/capsule colliders this spring collides with.
        public var colliders: [Collider]? = nil
    }

    /// A collider attached to a joint (offset/radius in joint-local space). When
    /// `tail` is set it is a capsule (offset→tail segment, swept radius); else a
    /// sphere. Per the VRM spring-bone spec.
    public struct Collider: Codable, Sendable {
        public var joint: String
        public var offset: [Float]   // x,y,z (sphere center / capsule start)
        public var radius: Float
        public var tail: [Float]? = nil   // capsule end (x,y,z)
    }
}

public extension GLTFAsset {

    /// Builds the spring-bone rig (joint node indices → USD joint paths via the
    /// skeleton). When `limbSway` is set, appends synthetic stiff spring chains
    /// for the arms (via the VRM humanoid map) so the limbs lag/settle with
    /// motion instead of staying rigid. Returns nil if there are no springs.
    func springBoneRig(skeleton: GLTFSkeleton, limbSway: Bool = true) -> SpringBoneRig? {
        guard let ext = document.extensions else { return nil }
        var rig: SpringBoneRig?
        if let sb = ext["VRMC_vrm"]?["springBone"] ?? ext["VRMC_springBone"] {   // VRM 1.0
            rig = parseVRM1SpringBone(sb, skeleton: skeleton)
        } else if let sa = ext["VRM"]?["secondaryAnimation"] {                   // VRM 0.x
            rig = parseVRM0SpringBone(sa, skeleton: skeleton)
        }
        if limbSway, let arms = armSprings(skeleton: skeleton), !arms.isEmpty {
            if rig == nil { rig = SpringBoneRig(springs: []) }
            rig!.springs.append(contentsOf: arms)
        }
        return rig
    }

    /// Synthetic spring chains for the arms (upperArm→lowerArm→hand per side),
    /// stiff and well-damped so they hold the rest pose but trail with motion.
    private func armSprings(skeleton: GLTFSkeleton) -> [SpringBoneRig.Spring]? {
        let bones = humanoidBones()
        guard !bones.isEmpty else { return nil }
        let chains = [
            ("arm_L", ["leftUpperArm", "leftLowerArm", "leftHand"]),
            ("arm_R", ["rightUpperArm", "rightLowerArm", "rightHand"]),
        ]
        // Stiff (snaps back to the arms-down pose), moderate drag, no gravity.
        let stiff: Float = 2.0, drag: Float = 0.55
        var springs: [SpringBoneRig.Spring] = []
        for (name, boneNames) in chains {
            let paths: [String] = boneNames.compactMap { b in
                guard let node = bones[b] else { return nil }
                return skeleton.jointPaths(forSkinJoints: [node])?.first
            }
            guard paths.count >= 2 else { continue }
            let n = paths.count
            springs.append(.init(name: name, joints: paths,
                                 stiffness: Array(repeating: stiff, count: n),
                                 dragForce: Array(repeating: drag, count: n),
                                 gravityPower: Array(repeating: 0, count: n),
                                 gravityDir: [0, -1, 0],
                                 hitRadius: Array(repeating: 0, count: n)))
        }
        return springs
    }

    // MARK: VRM 1.0 — extensions.VRMC_springBone.springs[].joints[]

    private func parseVRM1SpringBone(_ sb: JSONValue, skeleton: GLTFSkeleton) -> SpringBoneRig? {
        guard let springsJSON = sb["springs"]?.arrayValue else { return nil }

        // Resolve colliders (sphere or capsule) → joint path + offset + radius (+ tail).
        let collidersJSON = sb["colliders"]?.arrayValue ?? []
        let resolvedColliders: [SpringBoneRig.Collider?] = collidersJSON.map { c in
            guard let node = c["node"]?.intValue,
                  let path = skeleton.jointPaths(forSkinJoints: [node])?.first,
                  let shape = c["shape"] else { return nil }
            func vec(_ v: JSONValue?) -> [Float] { v?.arrayValue?.map { Float($0.doubleValue ?? 0) } ?? [0, 0, 0] }
            if let sphere = shape["sphere"] {
                return SpringBoneRig.Collider(joint: path, offset: vec(sphere["offset"]),
                                              radius: Float(sphere["radius"]?.doubleValue ?? 0))
            }
            if let cap = shape["capsule"] {
                return SpringBoneRig.Collider(joint: path, offset: vec(cap["offset"]),
                                              radius: Float(cap["radius"]?.doubleValue ?? 0),
                                              tail: vec(cap["tail"]))
            }
            return nil
        }
        // Collider groups → list of collider indices.
        let groups: [[Int]] = (sb["colliderGroups"]?.arrayValue ?? []).map { g in
            (g["colliders"]?.arrayValue ?? []).compactMap { $0.intValue }
        }
        // NOTE on hand/forearm-vs-skirt: we tried synthesizing forearm capsules and
        // adding them to the skirt springs (the research's recommended fix). But on
        // models with large bell sleeves (e.g. Anna) the sleeve cloth — not the bare
        // arm — overlaps the skirt, so a forearm-sized collider doesn't separate the
        // meshes, while a sleeve-sized one floats the skirt unnaturally at rest (the
        // arm rests against the skirt naturally). Per the three-vrm maintainer, this
        // is a known model-side limitation with no clean app-side fix. So we keep the
        // capsule-collision *capability* (used for any authored capsule colliders)
        // but don't synthesize arm colliders onto the skirt.

        var springs: [SpringBoneRig.Spring] = []
        for (i, spring) in springsJSON.enumerated() {
            guard let jointsJSON = spring["joints"]?.arrayValue else { continue }
            var paths: [String] = []
            var stiffness: [Float] = [], drag: [Float] = [], gravP: [Float] = [], hitR: [Float] = []
            var gravDir: [Float] = [0, -1, 0]
            for j in jointsJSON {
                guard let node = j["node"]?.intValue,
                      let path = skeleton.jointPaths(forSkinJoints: [node])?.first else { continue }
                paths.append(path)
                stiffness.append(Float(j["stiffness"]?.doubleValue ?? 1.0))
                drag.append(Float(j["dragForce"]?.doubleValue ?? 0.5))
                gravP.append(Float(j["gravityPower"]?.doubleValue ?? 0.0))
                hitR.append(Float(j["hitRadius"]?.doubleValue ?? 0.0))
                if let gd = j["gravityDir"]?.arrayValue, gd.count == 3 {
                    gravDir = gd.map { Float($0.doubleValue ?? 0) }
                }
            }
            guard paths.count >= 2 else { continue }   // need a chain to sway

            // Resolve this spring's collider groups → flat sphere collider list.
            var colliders: [SpringBoneRig.Collider] = []
            for gi in (spring["colliderGroups"]?.arrayValue ?? []).compactMap({ $0.intValue }) {
                guard groups.indices.contains(gi) else { continue }
                for ci in groups[gi] where resolvedColliders.indices.contains(ci) {
                    if let c = resolvedColliders[ci] { colliders.append(c) }
                }
            }
            let springName = spring["name"]?.stringValue ?? "spring_\(i)"
            springs.append(.init(name: springName,
                                 joints: paths, stiffness: stiffness, dragForce: drag,
                                 gravityPower: gravP, gravityDir: gravDir, hitRadius: hitR,
                                 colliders: colliders.isEmpty ? nil : colliders))
        }
        return springs.isEmpty ? nil : SpringBoneRig(springs: springs)
    }

    // MARK: VRM 0.x — extensions.VRM.secondaryAnimation.boneGroups[].bones[]

    private func parseVRM0SpringBone(_ sa: JSONValue, skeleton: GLTFSkeleton) -> SpringBoneRig? {
        guard let groups = sa["boneGroups"]?.arrayValue else { return nil }
        var springs: [SpringBoneRig.Spring] = []
        for (gi, g) in groups.enumerated() {
            let stiff = Float(g["stiffiness"]?.doubleValue ?? g["stiffness"]?.doubleValue ?? 1.0)
            let drag = Float(g["dragForce"]?.doubleValue ?? 0.5)
            let gravP = Float(g["gravityPower"]?.doubleValue ?? 0.0)
            var gravDir: [Float] = [0, -1, 0]
            if let gd = g["gravityDir"]?.objectValue {
                gravDir = [Float(gd["x"]?.doubleValue ?? 0), Float(gd["y"]?.doubleValue ?? -1), Float(gd["z"]?.doubleValue ?? 0)]
            }
            let hitR = Float(g["hitRadius"]?.doubleValue ?? 0.0)
            // 0.x lists root bones; each becomes a chain down the node hierarchy.
            for rootNode in (g["bones"]?.arrayValue ?? []).compactMap({ $0.intValue }) {
                let chain = descendantChain(from: rootNode)
                let paths = chain.compactMap { skeleton.jointPaths(forSkinJoints: [$0])?.first }
                guard paths.count >= 2 else { continue }
                springs.append(.init(name: "boneGroup_\(gi)", joints: paths,
                                     stiffness: Array(repeating: stiff, count: paths.count),
                                     dragForce: Array(repeating: drag, count: paths.count),
                                     gravityPower: Array(repeating: gravP, count: paths.count),
                                     gravityDir: gravDir,
                                     hitRadius: Array(repeating: hitR, count: paths.count)))
            }
        }
        return springs.isEmpty ? nil : SpringBoneRig(springs: springs)
    }

    /// Follows first-child links to form a chain (VRM 0.x spring roots).
    private func descendantChain(from root: Int) -> [Int] {
        guard let nodes = document.nodes else { return [root] }
        var chain = [root]
        var cur = root
        while let first = nodes[cur].children?.first {
            chain.append(first); cur = first
            if chain.count > 24 { break }
        }
        return chain
    }
}
