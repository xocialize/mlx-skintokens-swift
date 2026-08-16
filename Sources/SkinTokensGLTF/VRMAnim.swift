import Foundation
import simd

/// A VRM Animation (.vrma) reduced to humanoid-bone rotation tracks (+ hips
/// translation), keyed by the standard VRM humanoid bone names — the portable
/// key for retargeting onto any VRM. Emitted as `<name>.anim.json` for the
/// runtime player to consume (the runtime has no GLB parser).
public struct VRMAClip: Codable, Sendable {
    public var name: String
    public var duration: Float
    /// boneName → rotation keyframes.
    public var bones: [String: Track]
    /// Optional hips translation track (root motion).
    public var hips: HipsTrack?
    /// The .vrma's rest local rotation per bone (xyzw) — lets the runtime retarget
    /// as a delta from rest if direct application doesn't match a target's bind.
    public var restRotations: [String: [Float]]
    /// Rest world position of the clip skeleton's hips (xyz, meters) — the
    /// baseline the runtime subtracts from the hips track to get root-motion
    /// deltas it can scale to the target avatar's hips height.
    public var hipsRest: [Float]?

    public struct Track: Codable, Sendable {
        public var times: [Float]
        public var rotations: [[Float]]   // each [x,y,z,w]
    }
    public struct HipsTrack: Codable, Sendable {
        public var times: [Float]
        public var translations: [[Float]]  // each [x,y,z]
    }
}

public extension GLTFAsset {

    /// True if this asset is a VRM Animation (.vrma) rather than a model.
    var isVRMAnimation: Bool {
        document.extensions?["VRMC_vrm_animation"] != nil
    }

    /// Parses a loaded .vrma into humanoid bone tracks. Returns nil if not a VRMA.
    func parseVRMAnimation(name: String) -> VRMAClip? {
        guard let va = document.extensions?["VRMC_vrm_animation"],
              let humanBones = va["humanoid"]?["humanBones"]?.objectValue,
              let anim = document.animations?.first else { return nil }

        // Channel maps: node → sampler index, per path.
        var rotSampler = [Int: Int](), transSampler = [Int: Int]()
        for ch in anim.channels {
            guard let node = ch.target.node else { continue }
            switch ch.target.path {
            case "rotation": rotSampler[node] = ch.sampler
            case "translation": transSampler[node] = ch.sampler
            default: break
            }
        }

        func readSampler(_ index: Int) -> (times: [Float], values: [Float], stride3IfSpline: Bool)? {
            guard anim.samplers.indices.contains(index) else { return nil }
            let s = anim.samplers[index]
            guard let times = try? decoder.readFloats(accessorIndex: s.input),
                  let values = try? decoder.readFloats(accessorIndex: s.output) else { return nil }
            return (times, values, (s.interpolation ?? "LINEAR") == "CUBICSPLINE")
        }

        // CUBICSPLINE stores [inTangent, value, outTangent] per keyframe → take value.
        func keyframeValues(_ values: [Float], comps: Int, spline: Bool) -> [[Float]] {
            var out: [[Float]] = []
            if spline {
                let group = comps * 3
                var i = 0
                while i + group <= values.count {
                    out.append(Array(values[(i + comps)..<(i + 2 * comps)]))
                    i += group
                }
            } else {
                var i = 0
                while i + comps <= values.count { out.append(Array(values[i..<(i + comps)])); i += comps }
            }
            return out
        }

        // Rebase every track into VRM-normalized space (identity rest, world-aligned
        // local frames) — the retarget three-vrm applies to Mixamo-sourced clips:
        //   q' = parentRestWorld(node) * q * inverse(restWorld(node))
        // VRoid/normalized VRMAs have identity rest world rotations, so this is a
        // no-op for them; Mixamo skeletons carry joint orientations that otherwise
        // fold the target avatar in half. Scale is illegal in VRMA, and translation
        // doesn't affect orientation, so the rotation-only chain is sufficient.
        let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        func nodeQuat(_ a: [Float]?) -> simd_quatf {
            guard let a, a.count == 4 else { return identity }
            return simd_quatf(ix: a[0], iy: a[1], iz: a[2], r: a[3])
        }
        var parentOf = [Int: Int]()
        for (i, n) in (document.nodes ?? []).enumerated() {
            for c in n.children ?? [] { parentOf[c] = i }
        }
        var restWorldCache = [Int: simd_quatf]()
        func restWorld(_ i: Int) -> simd_quatf {
            if let q = restWorldCache[i] { return q }
            let local = nodeQuat(document.nodes?[safe: i]?.rotation)
            let q = ((parentOf[i].map { restWorld($0) } ?? identity) * local).normalized
            restWorldCache[i] = q
            return q
        }
        func nodeVec(_ a: [Float]?) -> SIMD3<Float> {
            guard let a, a.count == 3 else { return .zero }
            return SIMD3(a[0], a[1], a[2])
        }
        var restWorldPosCache = [Int: SIMD3<Float>]()
        func restWorldPos(_ i: Int) -> SIMD3<Float> {
            if let p = restWorldPosCache[i] { return p }
            let local = nodeVec(document.nodes?[safe: i]?.translation)
            let p: SIMD3<Float>
            if let par = parentOf[i] {
                p = restWorldPos(par) + restWorld(par).act(local)
            } else {
                p = local
            }
            restWorldPosCache[i] = p
            return p
        }

        var bones: [String: VRMAClip.Track] = [:]
        var restRotations: [String: [Float]] = [:]
        var duration: Float = 0

        for (boneName, v) in humanBones {
            guard let node = v["node"]?.intValue else { continue }
            // Tracks below are rebased to normalized space, so the rest is identity.
            restRotations[boneName] = [0, 0, 0, 1]
            guard let si = rotSampler[node], let s = readSampler(si) else { continue }
            let parentWorld = parentOf[node].map { restWorld($0) } ?? identity
            let restWorldInv = restWorld(node).inverse
            let quats = keyframeValues(s.values, comps: 4, spline: s.stride3IfSpline)
                .map { arr -> [Float] in
                    let q = (parentWorld * nodeQuat(arr) * restWorldInv).normalized
                    return [q.imag.x, q.imag.y, q.imag.z, q.real]
                }
            bones[boneName] = .init(times: s.times, rotations: quats)
            duration = max(duration, s.times.last ?? 0)
        }

        var hips: VRMAClip.HipsTrack?
        var hipsRest: [Float]?
        if let hipsNode = humanBones["hips"]?["node"]?.intValue,
           let si = transSampler[hipsNode], let s = readSampler(si) {
            // Emit hips positions in rest-world space (parent offset + rotated
            // local), so runtime root-motion deltas are (position − hipsRest).
            let parentWorld = parentOf[hipsNode].map { restWorld($0) } ?? identity
            let parentPos = parentOf[hipsNode].map { restWorldPos($0) } ?? .zero
            let trans = keyframeValues(s.values, comps: 3, spline: s.stride3IfSpline)
                .map { arr -> [Float] in
                    guard arr.count == 3 else { return arr }
                    let p = parentPos + parentWorld.act(SIMD3<Float>(arr[0], arr[1], arr[2]))
                    return [p.x, p.y, p.z]
                }
            hips = .init(times: s.times, translations: trans)
            let rest = restWorldPos(hipsNode)
            hipsRest = [rest.x, rest.y, rest.z]
            duration = max(duration, s.times.last ?? 0)
        }

        guard !bones.isEmpty else { return nil }
        return VRMAClip(name: name, duration: duration, bones: bones, hips: hips,
                        restRotations: restRotations, hipsRest: hipsRest)
    }
}
