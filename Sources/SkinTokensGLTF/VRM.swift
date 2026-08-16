import Foundation

/// The VRM semantic layer extracted from the glTF `extensions`. VRM adds, on top
/// of vanilla glTF morph targets, a binding layer that names *expressions*
/// (happy / aa / blink …) as weighted combinations of morph targets.
///
/// This normalizes both dialects into one model:
///  - VRM 1.0: `extensions.VRMC_vrm.expressions.preset/custom`, weights 0–1,
///    binds reference a glTF *node* + morph index.
///  - VRM 0.x: `extensions.VRM.blendShapeMaster.blendShapeGroups`, weights 0–100,
///    binds reference a *mesh* + morph index.
public struct VRMExpressionSet: Sendable {
    public let version: GLTFAsset.VRMVersion
    public let expressions: [VRMExpression]
}

public struct VRMExpression: Sendable {
    public let name: String
    /// True if this name is one of the spec's presets (vs. a custom expression).
    public let isPreset: Bool
    /// Binary expressions snap to 0/1 rather than blending.
    public let isBinary: Bool
    /// VRM 1.0 override semantics (none|block|blend); nil for 0.x.
    public let overrideBlink: String?
    public let overrideLookAt: String?
    public let overrideMouth: String?
    public let binds: [VRMMorphBind]
}

/// One morph contribution within an expression. Exactly one of mesh/node is set,
/// depending on dialect; `weight` is always normalized to 0–1.
public struct VRMMorphBind: Sendable {
    public let meshIndex: Int?
    public let nodeIndex: Int?
    public let morphIndex: Int
    public let weight: Float
}

public extension GLTFAsset {

    /// The canonical VRM 1.0 preset expression keys (everything else is custom).
    static let vrm1Presets: Set<String> = [
        "aa","ih","ou","ee","oh","blink","blinkLeft","blinkRight",
        "happy","angry","sad","relaxed","surprised",
        "lookUp","lookDown","lookLeft","lookRight","neutral"
    ]

    /// Parses the VRM expression layer, or nil if this isn't a VRM document.
    func vrmExpressions() -> VRMExpressionSet? {
        guard let ext = document.extensions, let version = vrmVersion else { return nil }
        switch version {
        case .v1: return parseVRM1(ext)
        case .v0: return parseVRM0(ext)
        }
    }

    // MARK: VRM 1.0

    private func parseVRM1(_ ext: JSONValue) -> VRMExpressionSet? {
        guard let expressions = ext["VRMC_vrm"]?["expressions"] else { return nil }
        var out: [VRMExpression] = []
        for (groupKey, isPreset) in [("preset", true), ("custom", false)] {
            guard let group = expressions[groupKey]?.objectValue else { continue }
            for (name, value) in group {
                out.append(parseVRM1Expression(name: name, isPreset: isPreset, value: value))
            }
        }
        return VRMExpressionSet(version: .v1, expressions: out.sorted { $0.name < $1.name })
    }

    private func parseVRM1Expression(name: String, isPreset: Bool, value: JSONValue) -> VRMExpression {
        var binds: [VRMMorphBind] = []
        for b in value["morphTargetBinds"]?.arrayValue ?? [] {
            guard let node = b["node"]?.intValue, let index = b["index"]?.intValue else { continue }
            let w = Float(b["weight"]?.doubleValue ?? 1.0)  // VRM 1.0 weights already 0–1
            binds.append(VRMMorphBind(meshIndex: nil, nodeIndex: node, morphIndex: index, weight: w))
        }
        return VRMExpression(
            name: name,
            isPreset: isPreset && GLTFAsset.vrm1Presets.contains(name),
            isBinary: value["isBinary"]?.boolValue ?? false,
            overrideBlink: value["overrideBlink"]?.stringValue,
            overrideLookAt: value["overrideLookAt"]?.stringValue,
            overrideMouth: value["overrideMouth"]?.stringValue,
            binds: binds)
    }

    // MARK: VRM 0.x

    private func parseVRM0(_ ext: JSONValue) -> VRMExpressionSet? {
        guard let groups = ext["VRM"]?["blendShapeMaster"]?["blendShapeGroups"]?.arrayValue else { return nil }
        var out: [VRMExpression] = []
        for g in groups {
            // Unique id is presetName (if known) else name, upper-cased (per spec).
            let preset = g["presetName"]?.stringValue
            let rawName = g["name"]?.stringValue
            let hasPreset = (preset != nil && !(preset!.isEmpty) && preset!.lowercased() != "unknown")
            let name = (hasPreset ? preset! : (rawName ?? "unnamed"))
            var binds: [VRMMorphBind] = []
            for b in g["binds"]?.arrayValue ?? [] {
                guard let mesh = b["mesh"]?.intValue, let index = b["index"]?.intValue else { continue }
                // VRM 0.x weights are 0–100; normalize to 0–1.
                let w = Float(b["weight"]?.doubleValue ?? 100.0) / 100.0
                binds.append(VRMMorphBind(meshIndex: mesh, nodeIndex: nil, morphIndex: index, weight: w))
            }
            out.append(VRMExpression(
                name: name,
                isPreset: hasPreset,
                isBinary: g["isBinary"]?.boolValue ?? false,
                overrideBlink: nil, overrideLookAt: nil, overrideMouth: nil,
                binds: binds))
        }
        return VRMExpressionSet(version: .v0, expressions: out.sorted { $0.name < $1.name })
    }

    // MARK: Bind → blend-shape-name resolution

    /// Resolves a bind's (node|mesh, morphIndex) to the morph-target *name* we
    /// author into USD, so the sidecar can drive blend shapes by name at runtime.
    func resolveBlendShapeName(for bind: VRMMorphBind) -> String? {
        let meshIndex: Int?
        if let m = bind.meshIndex {
            meshIndex = m
        } else if let n = bind.nodeIndex {
            meshIndex = document.nodes?[safe: n]?.mesh
        } else {
            meshIndex = nil
        }
        guard let mi = meshIndex, document.meshes?.indices.contains(mi) == true else { return nil }
        let names = morphTargetNames(meshIndex: mi)
        return names[safe: bind.morphIndex]
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
