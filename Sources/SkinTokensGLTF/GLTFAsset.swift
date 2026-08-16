import Foundation
import simd

/// A loaded glTF/GLB/VRM asset: the decoded JSON document plus resolved buffers,
/// with helpers to pull out the geometry and morph targets the USD writer needs.
public struct GLTFAsset {
    public let document: GLTFDocument
    public let buffers: [Data]
    public let decoder: AccessorDecoder
    /// True if `extensionsUsed` declares a VRM extension (VRMC_vrm or VRM).
    public let isVRM: Bool
    public let vrmVersion: VRMVersion?

    public enum VRMVersion: String, Sendable { case v0 = "0.x"; case v1 = "1.0" }

    public enum Error: Swift.Error, CustomStringConvertible {
        case dataURIUnsupportedScheme(String)
        case missingBinChunkForBuffer(Int)

        public var description: String {
            switch self {
            case .dataURIUnsupportedScheme(let s): return "glTF: unsupported buffer uri scheme: \(s)"
            case .missingBinChunkForBuffer(let i): return "glTF: buffer \(i) has no uri and no GLB BIN chunk"
            }
        }
    }

    /// Loads from raw bytes — either a `.glb`/`.vrm` (GLB container) or a plain
    /// `.gltf` JSON document (with embedded data: URIs or, later, sidecar buffers).
    public static func load(data: Data) throws -> GLTFAsset {
        let jsonData: Data
        let binChunk: Data?
        if GLB.isGLB(data) {
            let chunks = try GLB.parse(data)
            jsonData = chunks.json
            binChunk = chunks.bin
        } else {
            jsonData = data
            binChunk = nil
        }

        let document = try JSONDecoder().decode(GLTFDocument.self, from: jsonData)

        // Resolve every buffer to bytes.
        var buffers: [Data] = []
        for (i, buffer) in (document.buffers ?? []).enumerated() {
            if let uri = buffer.uri {
                buffers.append(try resolveURI(uri))
            } else if let binChunk {
                // The first buffer with no uri is the GLB BIN chunk.
                buffers.append(binChunk)
            } else {
                throw Error.missingBinChunkForBuffer(i)
            }
        }

        let used = document.extensionsUsed ?? []
        let isVRM = used.contains("VRMC_vrm") || used.contains("VRM")
        let version: VRMVersion? = used.contains("VRMC_vrm") ? .v1
            : (used.contains("VRM") ? .v0 : nil)

        return GLTFAsset(
            document: document,
            buffers: buffers,
            decoder: AccessorDecoder(document: document, buffers: buffers),
            isVRM: isVRM,
            vrmVersion: version
        )
    }

    public static func load(url: URL) throws -> GLTFAsset {
        try load(data: try Data(contentsOf: url))
    }

    private static func resolveURI(_ uri: String) throws -> Data {
        if uri.hasPrefix("data:") {
            // data:[<mediatype>][;base64],<data>
            guard let comma = uri.firstIndex(of: ",") else {
                throw Error.dataURIUnsupportedScheme(uri)
            }
            let meta = uri[uri.index(uri.startIndex, offsetBy: 5)..<comma]
            let payload = String(uri[uri.index(after: comma)...])
            if meta.contains("base64") {
                return Data(base64Encoded: payload) ?? Data()
            } else {
                return Data((payload.removingPercentEncoding ?? payload).utf8)
            }
        }
        throw Error.dataURIUnsupportedScheme(uri)
    }
}

// MARK: - Morph-target extraction

public extension GLTFAsset {
    /// One morph target's decoded position deltas, with its resolved name.
    struct MorphTarget {
        public let name: String
        /// Per-vertex POSITION deltas (length == base vertex count).
        public let positionDeltas: [SIMD3<Float>]
        /// Per-vertex NORMAL deltas, if the target authored them.
        public let normalDeltas: [SIMD3<Float>]?
    }

    /// A primitive's base geometry plus its morph targets, ready for USD authoring.
    struct PrimitiveGeometry {
        public let positions: [SIMD3<Float>]
        public let normals: [SIMD3<Float>]?
        public let indices: [UInt32]?
        public let morphTargets: [MorphTarget]
    }

    /// Extracts geometry + morph targets for `meshes[meshIndex].primitives[primIndex]`,
    /// resolving per-target names from `mesh.extras.targetNames` (falling back to
    /// `morph_<i>` when unnamed).
    func primitiveGeometry(meshIndex: Int, primitiveIndex: Int) throws -> PrimitiveGeometry {
        let mesh = document.meshes![meshIndex]
        let prim = mesh.primitives[primitiveIndex]

        let positions = try prim.attributes["POSITION"].map { try decoder.readVec3(accessorIndex: $0) } ?? []
        let normals = try prim.attributes["NORMAL"].map { try decoder.readVec3(accessorIndex: $0) }
        let indices = try prim.indices.map { try decoder.readIndices(accessorIndex: $0) }

        let names = mesh.extras?.targetNames
        var targets: [MorphTarget] = []
        for (i, target) in (prim.targets ?? []).enumerated() {
            let pDeltas = try target["POSITION"].map { try decoder.readVec3(accessorIndex: $0) } ?? []
            let nDeltas = try target["NORMAL"].map { try decoder.readVec3(accessorIndex: $0) }
            let name = (names?.indices.contains(i) == true) ? names![i] : "morph_\(i)"
            targets.append(MorphTarget(name: name, positionDeltas: pDeltas, normalDeltas: nDeltas))
        }

        return PrimitiveGeometry(positions: positions, normals: normals,
                                 indices: indices, morphTargets: targets)
    }

    /// All morph-target names declared on a mesh (in index order).
    func morphTargetNames(meshIndex: Int) -> [String] {
        let mesh = document.meshes![meshIndex]
        if let names = mesh.extras?.targetNames { return names }
        let count = mesh.primitives.first?.targets?.count ?? 0
        return (0..<count).map { "morph_\($0)" }
    }
}
