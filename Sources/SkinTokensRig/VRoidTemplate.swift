import Foundation

/// Canonical VRoid T-pose humanoid skeleton, bundled as a resource, used to rig BARE generated meshes
/// (e.g. a TRELLIS.2 output) that carry no embedded skeleton. The template is source-agnostic to the
/// SkinTokens generator: fit its joints to the target mesh, tokenize them as the Route-B prefix, and
/// the model predicts skin only — then `injectRig` authors these VRM-named bones + weights.
///
/// The template is Y-up, feet at y≈0, arms horizontal (T-pose), in meters. `fit(...)` maps it into an
/// arbitrary mesh's coordinate frame (TRELLIS output is NOT reliably Y-up — the fit re-bases axes).
public struct VRoidTemplate: Sendable {
    public let joints: [[Float]]   // J×3, template space
    public let parents: [Int]      // J, parent joint index (-1 root)
    public let names: [String]     // J, VRM humanoid name where applicable else "bone_j"
    public let humanoid: [String: Int]  // VRM humanoid bone name → joint index (for the VRMC_vrm ext)
    public var J: Int { joints.count }

    private struct Raw: Decodable { let joints: [[Float]]; let parents: [Int]; let humanoid: [String: Int] }

    public static func loadBundled() throws -> VRoidTemplate {
        guard let url = Bundle.module.url(forResource: "vroid-template", withExtension: "json") else {
            throw Err.missingResource
        }
        return try load(url: url)
    }

    public static func load(url: URL) throws -> VRoidTemplate {
        let raw = try JSONDecoder().decode(Raw.self, from: try Data(contentsOf: url))
        var names = (0..<raw.joints.count).map { "bone_\($0)" }
        for (name, idx) in raw.humanoid where idx >= 0 && idx < names.count { names[idx] = name }
        return VRoidTemplate(joints: raw.joints, parents: raw.parents, names: names, humanoid: raw.humanoid)
    }

    public enum Err: Swift.Error, CustomStringConvertible {
        case missingResource
        public var description: String { "vroid-template.json not bundled in SkinTokensRig" }
    }

    /// Which mesh axis is "up" (head→foot). TRELLIS output orientation varies, so the fit picks the
    /// longest bbox extent as up by default; callers who know the orientation can override.
    public enum UpAxis: Int, Sendable { case x = 0, y = 1, z = 2 }

    /// Fit the template into a mesh's coordinate frame. `meshVertices` = flat [x,y,z,…].
    /// Re-bases axes: up = longest extent (or `up` override), depth = shortest, side = the remaining;
    /// scales uniformly so template height matches the mesh's up-extent; centers side/depth on the mesh,
    /// aligns feet to the mesh's up-min. Returns fitted joints [J][3] in mesh space.
    public func fit(toMeshVertices verts: [Float], up upOverride: UpAxis? = nil) -> [[Float]] {
        // mesh bbox
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3)
        var hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        let n = verts.count / 3
        for i in 0..<n { for k in 0..<3 { let x = verts[i*3+k]; lo[k] = min(lo[k], x); hi[k] = max(hi[k], x) } }
        let ext = [hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]]
        let center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]

        let up = upOverride?.rawValue ?? (0..<3).max(by: { ext[$0] < ext[$1] })!
        let depth = (0..<3).min(by: { ext[$0] < ext[$1] })!
        let side = (0..<3).first { $0 != up && $0 != depth } ?? (3 - up - depth)

        // template axes are fixed: x = side, y = up, z = depth
        var tLoY = Float.greatestFiniteMagnitude, tHiY = -Float.greatestFiniteMagnitude
        for j in joints { tLoY = min(tLoY, j[1]); tHiY = max(tHiY, j[1]) }
        let tHeight = max(tHiY - tLoY, 1e-6)
        let scale = ext[up] / tHeight

        var out = [[Float]](repeating: [0, 0, 0], count: J)
        for j in 0..<J {
            let t = joints[j]                 // (side=x, up=y, depth=z) in template space
            var p = [Float](repeating: 0, count: 3)
            p[up]    = lo[up] + (t[1] - tLoY) * scale         // feet→up-min, head→up-max
            p[side]  = center[side] + t[0] * scale            // arm span, centered
            p[depth] = center[depth] + t[2] * scale           // front-back, centered
            out[j] = p
        }
        return out
    }
}
