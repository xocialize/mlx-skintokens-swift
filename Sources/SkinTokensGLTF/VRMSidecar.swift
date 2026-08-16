import Foundation

/// The sidecar JSON the runtime consumes: each VRM expression resolved to a list
/// of `{blendShape, weight}` contributions referencing the morph-target names we
/// author into USD. Lets a runtime call `setExpression("happy", 0.8)` and fan it
/// out to RealityKit blend-shape weights, reconstructing semantics USD can't hold.
public struct VRMSidecar: Codable, Sendable {
    public var vrmVersion: String
    public var expressions: [Expression]

    public struct Expression: Codable, Sendable {
        public var name: String
        public var isPreset: Bool
        public var isBinary: Bool
        public var overrideBlink: String?
        public var overrideLookAt: String?
        public var overrideMouth: String?
        public var binds: [Bind]
    }

    public struct Bind: Codable, Sendable {
        public var blendShape: String
        public var weight: Float
    }
}

public extension GLTFAsset {

    /// Builds the runtime sidecar from the VRM expression layer, resolving every
    /// bind to a blend-shape name. Binds that can't be resolved (missing mesh/
    /// morph) are dropped and reported in `unresolved`.
    func buildVRMSidecar() -> (sidecar: VRMSidecar, unresolved: Int)? {
        guard let set = vrmExpressions() else { return nil }
        var unresolved = 0
        let expressions: [VRMSidecar.Expression] = set.expressions.map { expr in
            var binds: [VRMSidecar.Bind] = []
            for b in expr.binds {
                if let name = resolveBlendShapeName(for: b) {
                    binds.append(VRMSidecar.Bind(blendShape: name, weight: b.weight))
                } else {
                    unresolved += 1
                }
            }
            return VRMSidecar.Expression(
                name: expr.name, isPreset: expr.isPreset, isBinary: expr.isBinary,
                overrideBlink: expr.overrideBlink, overrideLookAt: expr.overrideLookAt,
                overrideMouth: expr.overrideMouth, binds: binds)
        }
        return (VRMSidecar(vrmVersion: set.version.rawValue, expressions: expressions), unresolved)
    }

    /// Encodes the sidecar to pretty-printed JSON with stable key ordering.
    func encodeVRMSidecar() -> Data? {
        guard let (sidecar, _) = buildVRMSidecar() else { return nil }
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? enc.encode(sidecar)
    }
}
