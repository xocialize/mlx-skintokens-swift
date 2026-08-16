import Foundation

/// Detection of ARKit face-tracking blend shapes ("Perfect Sync"). Avatars built
/// for ARKit name their morph targets exactly after the 52
/// `ARFaceAnchor.BlendShapeLocation` values; when present, an ARKit-driven
/// RealityKit face pipeline can drive the converted USDZ directly. For VRM-native
/// models these are absent and the VRM expression sidecar is used instead.
public enum ARKit52 {

    /// The 52 canonical ARKit blend-shape location names (camelCase).
    public static let names: [String] = [
        "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
        "cheekPuff","cheekSquintLeft","cheekSquintRight",
        "eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft","eyeLookDownRight",
        "eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight",
        "eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight",
        "eyeWideLeft","eyeWideRight",
        "jawForward","jawLeft","jawOpen","jawRight",
        "mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight",
        "mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight",
        "mouthPressLeft","mouthPressRight","mouthPucker","mouthRight",
        "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
        "mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight",
        "mouthUpperUpLeft","mouthUpperUpRight",
        "noseSneerLeft","noseSneerRight","tongueOut"
    ]

    /// Lower-cased lookup for case-insensitive matching.
    static let lowerSet: Set<String> = Set(names.map { $0.lowercased() })

    /// Returns the ARKit names present among `morphNames` (case-insensitive),
    /// in canonical order.
    public static func detect(in morphNames: [String]) -> [String] {
        let present = Set(morphNames.map { $0.lowercased() })
        return names.filter { present.contains($0.lowercased()) }
    }
}

public extension GLTFAsset {
    /// All morph-target names across every mesh in the document.
    func allMorphTargetNames() -> [String] {
        guard let meshes = document.meshes else { return [] }
        return (0..<meshes.count).flatMap { morphTargetNames(meshIndex: $0) }
    }

    /// The ARKit-52 blend shapes detected among this asset's morph targets, and
    /// whether it qualifies as a "Perfect Sync" model (a strong majority present).
    func detectARKitBlendShapes() -> (matched: [String], isPerfectSync: Bool) {
        let matched = ARKit52.detect(in: allMorphTargetNames())
        return (matched, matched.count >= 40)
    }
}
