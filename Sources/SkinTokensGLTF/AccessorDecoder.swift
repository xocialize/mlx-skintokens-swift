import Foundation
import simd

/// Resolves glTF accessors into typed Swift arrays by reading through
/// bufferView → buffer bytes, honoring byteOffset, byteStride, componentType,
/// and normalization. This is the bridge from raw GLB bytes to geometry.
public struct AccessorDecoder {
    public let document: GLTFDocument
    /// Decoded buffer bytes by buffer index (GLB BIN chunk, or data: URIs).
    public let buffers: [Data]

    public enum Error: Swift.Error, CustomStringConvertible {
        case missingAccessors
        case accessorOutOfRange(Int)
        case accessorHasNoBufferView(Int)
        case bufferViewOutOfRange(Int)
        case bufferOutOfRange(Int)
        case unknownComponentType(Int)
        case unknownAccessorType(String)
        case readOutOfBounds

        public var description: String {
            switch self {
            case .missingAccessors: return "glTF: document has no accessors"
            case .accessorOutOfRange(let i): return "glTF: accessor index \(i) out of range"
            case .accessorHasNoBufferView(let i): return "glTF: accessor \(i) has no bufferView (sparse accessors not yet supported)"
            case .bufferViewOutOfRange(let i): return "glTF: bufferView index \(i) out of range"
            case .bufferOutOfRange(let i): return "glTF: buffer index \(i) out of range"
            case .unknownComponentType(let c): return "glTF: unknown componentType \(c)"
            case .unknownAccessorType(let t): return "glTF: unknown accessor type \(t)"
            case .readOutOfBounds: return "glTF: accessor read runs past buffer end"
            }
        }
    }

    public init(document: GLTFDocument, buffers: [Data]) {
        self.document = document
        self.buffers = buffers
    }

    /// Reads accessor `index` as a flat array of Floats (component-major).
    /// Integer component types are converted to Float; if `normalized`, integers
    /// are mapped to [0,1] (unsigned) or [-1,1] (signed) per the glTF spec.
    public func readFloats(accessorIndex index: Int) throws -> [Float] {
        guard let accessors = document.accessors else { throw Error.missingAccessors }
        guard accessors.indices.contains(index) else { throw Error.accessorOutOfRange(index) }
        let accessor = accessors[index]

        guard let comp = ComponentType(rawValue: accessor.componentType) else {
            throw Error.unknownComponentType(accessor.componentType)
        }
        guard let type = AccessorType(rawValue: accessor.type) else {
            throw Error.unknownAccessorType(accessor.type)
        }
        guard let bvIndex = accessor.bufferView else {
            throw Error.accessorHasNoBufferView(index)
        }
        guard let bufferViews = document.bufferViews, bufferViews.indices.contains(bvIndex) else {
            throw Error.bufferViewOutOfRange(bvIndex)
        }
        let bv = bufferViews[bvIndex]
        guard buffers.indices.contains(bv.buffer) else { throw Error.bufferOutOfRange(bv.buffer) }
        let bytes = buffers[bv.buffer]

        let compCount = type.componentCount
        let compSize = comp.byteSize
        let elementSize = compCount * compSize
        // Stride defaults to tightly packed; only valid for vertex attributes per spec.
        let stride = bv.byteStride ?? elementSize
        let baseOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
        let normalized = accessor.normalized ?? false

        var out = [Float](); out.reserveCapacity(accessor.count * compCount)

        // Read element by element so byteStride is honored.
        try bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for e in 0..<accessor.count {
                let elemOffset = baseOffset + e * stride
                for c in 0..<compCount {
                    let off = elemOffset + c * compSize
                    guard off + compSize <= raw.count else { throw Error.readOutOfBounds }
                    let v = readComponent(raw, at: off, comp: comp, normalized: normalized)
                    out.append(v)
                }
            }
        }
        return out
    }

    /// Reads a VEC3 accessor as `[SIMD3<Float>]` (positions, normals, deltas).
    public func readVec3(accessorIndex index: Int) throws -> [SIMD3<Float>] {
        let flat = try readFloats(accessorIndex: index)
        var out = [SIMD3<Float>](); out.reserveCapacity(flat.count / 3)
        var i = 0
        while i + 3 <= flat.count {
            out.append(SIMD3<Float>(flat[i], flat[i + 1], flat[i + 2]))
            i += 3
        }
        return out
    }

    /// Reads a VEC2 accessor as `[SIMD2<Float>]` (texture coordinates).
    public func readVec2(accessorIndex index: Int) throws -> [SIMD2<Float>] {
        let flat = try readFloats(accessorIndex: index)
        var out = [SIMD2<Float>](); out.reserveCapacity(flat.count / 2)
        var i = 0
        while i + 2 <= flat.count {
            out.append(SIMD2<Float>(flat[i], flat[i + 1]))
            i += 2
        }
        return out
    }

    /// Reads a VEC4 accessor as `[SIMD4<Float>]` (joint indices / weights).
    public func readVec4(accessorIndex index: Int) throws -> [SIMD4<Float>] {
        let flat = try readFloats(accessorIndex: index)
        var out = [SIMD4<Float>](); out.reserveCapacity(flat.count / 4)
        var i = 0
        while i + 4 <= flat.count {
            out.append(SIMD4<Float>(flat[i], flat[i + 1], flat[i + 2], flat[i + 3]))
            i += 4
        }
        return out
    }

    /// Reads a SCALAR integer accessor (e.g. mesh indices) as `[UInt32]`.
    public func readIndices(accessorIndex index: Int) throws -> [UInt32] {
        let flat = try readFloats(accessorIndex: index)
        return flat.map { UInt32($0.rounded()) }
    }

    private func readComponent(_ raw: UnsafeRawBufferPointer, at off: Int,
                               comp: ComponentType, normalized: Bool) -> Float {
        switch comp {
        case .float:
            return raw.loadUnaligned(fromByteOffset: off, as: Float.self)
        case .unsignedByte:
            let v = raw.load(fromByteOffset: off, as: UInt8.self)
            return normalized ? Float(v) / 255.0 : Float(v)
        case .byte:
            let v = raw.load(fromByteOffset: off, as: Int8.self)
            return normalized ? max(Float(v) / 127.0, -1.0) : Float(v)
        case .unsignedShort:
            let v = raw.loadUnaligned(fromByteOffset: off, as: UInt16.self)
            return normalized ? Float(v) / 65535.0 : Float(v)
        case .short:
            let v = raw.loadUnaligned(fromByteOffset: off, as: Int16.self)
            return normalized ? max(Float(v) / 32767.0, -1.0) : Float(v)
        case .unsignedInt:
            let v = raw.loadUnaligned(fromByteOffset: off, as: UInt32.self)
            return Float(v)
        }
    }
}
