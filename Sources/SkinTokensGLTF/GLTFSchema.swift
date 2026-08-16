import Foundation

/// A decoded glTF 2.0 JSON document — the subset relevant to avatar geometry,
/// morph targets, and skinning. Freeform `extras`/`extensions` are captured as
/// raw JSON where we need to reach into them (VRM lives in `extensions`).
public struct GLTFDocument: Decodable {
    public var asset: Asset
    public var scene: Int?
    public var scenes: [Scene]?
    public var nodes: [Node]?
    public var meshes: [Mesh]?
    public var skins: [Skin]?
    public var accessors: [Accessor]?
    public var bufferViews: [BufferView]?
    public var buffers: [Buffer]?
    public var materials: [Material]?
    public var textures: [Texture]?
    public var images: [Image]?
    public var samplers: [Sampler]?
    public var animations: [Animation]?
    public var extensionsUsed: [String]?
    public var extensionsRequired: [String]?
    /// Raw top-level `extensions` object, kept as JSON for VRM (VRMC_vrm / VRM).
    public var extensions: JSONValue?

    public struct Asset: Decodable {
        public var version: String
        public var generator: String?
    }

    public struct Scene: Decodable {
        public var name: String?
        public var nodes: [Int]?
    }

    public struct Node: Decodable {
        public var name: String?
        public var mesh: Int?
        public var skin: Int?
        public var children: [Int]?
        public var matrix: [Float]?
        public var translation: [Float]?
        public var rotation: [Float]?   // quaternion xyzw
        public var scale: [Float]?
    }

    public struct Mesh: Decodable {
        public var name: String?
        public var primitives: [Primitive]
        /// Default morph-target weights, one per target.
        public var weights: [Float]?
        public var extras: Extras?

        public struct Extras: Decodable {
            /// Per-morph-target names. Non-standard but the de-facto convention
            /// (Blender/UniVRM/three.js all write it); VRM binds morphs by index.
            public var targetNames: [String]?
        }
    }

    public struct Primitive: Decodable {
        /// Attribute name (POSITION, NORMAL, TANGENT, …) → accessor index.
        public var attributes: [String: Int]
        public var indices: Int?
        public var material: Int?
        public var mode: Int?
        /// Morph targets: each is attribute name → accessor index of the *delta*.
        public var targets: [[String: Int]]?
    }

    public struct Skin: Decodable {
        public var name: String?
        public var inverseBindMatrices: Int?
        public var skeleton: Int?
        public var joints: [Int]
    }

    public struct Accessor: Decodable {
        public var bufferView: Int?
        public var byteOffset: Int?
        public var componentType: Int
        public var normalized: Bool?
        public var count: Int
        public var type: String      // SCALAR, VEC2, VEC3, VEC4, MAT4, …
        public var max: [Float]?
        public var min: [Float]?
    }

    public struct BufferView: Decodable {
        public var buffer: Int
        public var byteOffset: Int?
        public var byteLength: Int
        public var byteStride: Int?
        public var target: Int?
    }

    public struct Buffer: Decodable {
        /// Absent for the GLB binary chunk; a data: URI or relative path otherwise.
        public var uri: String?
        public var byteLength: Int
    }

    // MARK: Materials & textures

    public struct Material: Decodable {
        public var name: String?
        public var pbrMetallicRoughness: PBR?
        public var alphaMode: String?      // OPAQUE (default), MASK, BLEND
        public var alphaCutoff: Float?     // default 0.5 when MASK
        public var doubleSided: Bool?
        public var emissiveFactor: [Float]?
        /// Detect KHR_materials_unlit (VRoid uses it; we approximate in USD).
        public var extensions: JSONValue?

        public struct PBR: Decodable {
            public var baseColorFactor: [Float]?
            public var baseColorTexture: TextureRef?
            public var metallicFactor: Float?
            public var roughnessFactor: Float?
        }
        public struct TextureRef: Decodable {
            public var index: Int
            public var texCoord: Int?
        }
        public var isUnlit: Bool { extensions?["KHR_materials_unlit"] != nil }
    }

    public struct Texture: Decodable {
        public var source: Int?    // image index
        public var sampler: Int?
    }

    public struct Image: Decodable {
        public var name: String?
        public var mimeType: String?   // image/png, image/jpeg
        public var bufferView: Int?    // GLB-embedded
        public var uri: String?        // or data:/relative
    }

    public struct Sampler: Decodable {
        public var magFilter: Int?
        public var minFilter: Int?
        public var wrapS: Int?
        public var wrapT: Int?
    }

    // MARK: Animations

    public struct Animation: Decodable {
        public var name: String?
        public var channels: [Channel]
        public var samplers: [AnimationSampler]

        public struct Channel: Decodable {
            public var sampler: Int
            public var target: Target
            public struct Target: Decodable {
                public var node: Int?
                public var path: String   // translation | rotation | scale | weights
            }
        }
        public struct AnimationSampler: Decodable {
            public var input: Int          // accessor: keyframe times (SCALAR)
            public var output: Int         // accessor: values (VEC3/VEC4)
            public var interpolation: String?  // LINEAR (default) | STEP | CUBICSPLINE
        }
    }
}

// MARK: - glTF component / accessor type enums

public enum ComponentType: Int {
    case byte = 5120
    case unsignedByte = 5121
    case short = 5122
    case unsignedShort = 5123
    case unsignedInt = 5125
    case float = 5126

    public var byteSize: Int {
        switch self {
        case .byte, .unsignedByte: return 1
        case .short, .unsignedShort: return 2
        case .unsignedInt, .float: return 4
        }
    }
}

public enum AccessorType: String {
    case scalar = "SCALAR"
    case vec2 = "VEC2"
    case vec3 = "VEC3"
    case vec4 = "VEC4"
    case mat2 = "MAT2"
    case mat3 = "MAT3"
    case mat4 = "MAT4"

    public var componentCount: Int {
        switch self {
        case .scalar: return 1
        case .vec2: return 2
        case .vec3: return 3
        case .vec4: return 4
        case .mat2: return 4
        case .mat3: return 9
        case .mat4: return 16
        }
    }
}
