import Foundation

/// Parses the binary glTF (GLB) container: a 12-byte header followed by chunks.
/// VRM files are GLB with a `.vrm` extension, so this is the common entry point.
///
/// Layout (little-endian, glTF 2.0 §4.4):
///   header:  uint32 magic = 0x46546C67 ("glTF"), uint32 version = 2, uint32 length
///   chunk:   uint32 chunkLength, uint32 chunkType, byte[chunkLength] data
///   chunk 0 is JSON (type 0x4E4F534A), optional chunk 1 is BIN (type 0x004E4942).
public enum GLB {
    public enum Error: Swift.Error, CustomStringConvertible {
        case tooSmall
        case badMagic(UInt32)
        case unsupportedVersion(UInt32)
        case truncatedChunk
        case missingJSONChunk

        public var description: String {
            switch self {
            case .tooSmall: return "GLB: file too small to contain a header"
            case .badMagic(let m): return "GLB: bad magic 0x\(String(m, radix: 16)) (expected 'glTF')"
            case .unsupportedVersion(let v): return "GLB: unsupported version \(v) (expected 2)"
            case .truncatedChunk: return "GLB: chunk extends past end of file"
            case .missingJSONChunk: return "GLB: no JSON chunk found"
            }
        }
    }

    static let magic: UInt32 = 0x4654_6C67       // "glTF"
    static let jsonChunkType: UInt32 = 0x4E4F_534A // "JSON"
    static let binChunkType: UInt32 = 0x004E_4942  // "BIN\0"

    /// The two payloads a GLB carries: the JSON document and the optional binary buffer.
    public struct Chunks {
        public let json: Data
        public let bin: Data?
    }

    /// Detects whether `data` is a GLB (vs a plain-text `.gltf`) by its magic.
    public static func isGLB(_ data: Data) -> Bool {
        data.count >= 4 && readUInt32(data, 0) == magic
    }

    /// Splits a GLB into its JSON and BIN chunks. Throws if malformed.
    public static func parse(_ data: Data) throws -> Chunks {
        guard data.count >= 12 else { throw Error.tooSmall }
        guard readUInt32(data, 0) == magic else { throw Error.badMagic(readUInt32(data, 0)) }
        let version = readUInt32(data, 4)
        guard version == 2 else { throw Error.unsupportedVersion(version) }
        // total length at offset 8 is advisory; we walk chunks by their own lengths.

        var json: Data?
        var bin: Data?
        var offset = 12
        while offset + 8 <= data.count {
            let chunkLength = Int(readUInt32(data, offset))
            let chunkType = readUInt32(data, offset + 4)
            let dataStart = offset + 8
            let dataEnd = dataStart + chunkLength
            guard dataEnd <= data.count else { throw Error.truncatedChunk }
            let chunk = data.subdata(in: dataStart..<dataEnd)
            switch chunkType {
            case jsonChunkType where json == nil: json = chunk
            case binChunkType where bin == nil: bin = chunk
            default: break // ignore unknown / extra chunks
            }
            offset = dataEnd
        }
        guard let json else { throw Error.missingJSONChunk }
        return Chunks(json: json, bin: bin)
    }

    /// Little-endian uint32 read at a byte offset.
    static func readUInt32(_ data: Data, _ offset: Int) -> UInt32 {
        let b = data
        return UInt32(b[b.startIndex + offset])
            | (UInt32(b[b.startIndex + offset + 1]) << 8)
            | (UInt32(b[b.startIndex + offset + 2]) << 16)
            | (UInt32(b[b.startIndex + offset + 3]) << 24)
    }
}
