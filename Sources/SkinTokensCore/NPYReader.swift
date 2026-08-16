import Foundation
import MLX

/// Minimal NumPy `.npy` v1.0/v2.0 reader → `MLXArray`, the shared fixture loader for every CLI
/// parity gate (PORTING-SPEC.md S0–S6). Reads the P0 PyTorch captures in
/// `captures/{giraffe,humanoid}_beam1_s0/*.npy` (bf16 saved as fp32, ids as int32).
///
/// Deliberately strict: **C-order only**. `np.save` preserves the layout of a transposed/permuted
/// tensor, so an F-order golden silently transposes if you accept it — the fix is to re-dump with
/// `np.ascontiguousarray(...)`, not to reinterpret here (swift-port-parity.md: "Fortran-order .npy
/// fixtures keep recurring"). We THROW on F-order rather than guess.
public enum NPYReader {

    public enum NPYError: Error, CustomStringConvertible {
        case notNPY(String)
        case fortranOrder(String)
        case unsupportedDType(String)
        case truncated(String)

        public var description: String {
            switch self {
            case .notNPY(let s): return "not a .npy file: \(s)"
            case .fortranOrder(let s): return "F-order .npy (re-dump with np.ascontiguousarray): \(s)"
            case .unsupportedDType(let s): return "unsupported .npy dtype: \(s)"
            case .truncated(let s): return "truncated .npy: \(s)"
            }
        }
    }

    /// The parsed header (descr + shape), before the payload is materialized.
    public struct Header {
        public let descr: String       // e.g. "<f4", "<i4", "<f2"
        public let shape: [Int]
        public let dataOffset: Int      // byte offset where the raw payload begins
        public var count: Int { shape.reduce(1, *) }
    }

    /// Load `path` into an `MLXArray`. fp16/fp32 → `.float32`; int32 → `.int32`; int64 → `.int64`.
    /// (Everything is materialized to a Swift array first, so this runs on any stream / no metallib.)
    public static func load(_ path: String) throws -> MLXArray {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw NPYError.truncated("cannot read \(path)")
        }
        return try load(data: data, label: path)
    }

    public static func load(data: Data, label: String = "<data>") throws -> MLXArray {
        let h = try parseHeader(data, label: label)
        return try materialize(data, header: h, label: label)
    }

    /// Parse the header only (magic + version + dict), no payload copy — used by count/shape probes.
    public static func parseHeader(_ data: Data, label: String = "<data>") throws -> Header {
        let bytes = [UInt8](data)
        guard bytes.count >= 10, bytes[0] == 0x93,
              bytes[1] == 0x4E, bytes[2] == 0x55, bytes[3] == 0x4D, bytes[4] == 0x50, bytes[5] == 0x59
        else { throw NPYError.notNPY(label) }   // \x93 N U M P Y

        let major = bytes[6]
        let headerLen: Int
        let dictStart: Int
        if major >= 2 {
            guard bytes.count >= 12 else { throw NPYError.truncated(label) }
            headerLen = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
            dictStart = 12
        } else {
            headerLen = Int(bytes[8]) | Int(bytes[9]) << 8
            dictStart = 10
        }
        let dictEnd = dictStart + headerLen
        guard dictEnd <= bytes.count else { throw NPYError.truncated(label) }
        let dict = String(decoding: bytes[dictStart..<dictEnd], as: UTF8.self)

        guard let descr = match(dict, key: "descr") else {
            throw NPYError.unsupportedDType("no descr in header of \(label)")
        }
        if dict.replacingOccurrences(of: " ", with: "").contains("'fortran_order':True") {
            throw NPYError.fortranOrder(label)
        }
        let shape = parseShape(dict)
        return Header(descr: descr, shape: shape, dataOffset: dictEnd)
    }

    // MARK: - payload

    private static func materialize(_ data: Data, header h: Header, label: String) throws -> MLXArray {
        let n = h.count
        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> MLXArray in
            let base = raw.baseAddress!.advanced(by: h.dataOffset)
            switch h.descr {
            case "<f4", "|f4", "=f4":
                let need = h.dataOffset + n * 4
                guard need <= data.count else { throw NPYError.truncated(label) }
                let p = base.bindMemory(to: Float.self, capacity: n)
                return MLXArray(Array(UnsafeBufferPointer(start: p, count: n)), h.shape)
            case "<f2", "|f2", "=f2":
                let need = h.dataOffset + n * 2
                guard need <= data.count else { throw NPYError.truncated(label) }
                let p = base.bindMemory(to: UInt16.self, capacity: n)
                var out = [Float](repeating: 0, count: n)
                for i in 0..<n { out[i] = Float(Float16(bitPattern: p[i])) }
                return MLXArray(out, h.shape)
            case "<i4", "|i4", "=i4":
                let need = h.dataOffset + n * 4
                guard need <= data.count else { throw NPYError.truncated(label) }
                let p = base.bindMemory(to: Int32.self, capacity: n)
                return MLXArray(Array(UnsafeBufferPointer(start: p, count: n)), h.shape)
            case "<i8", "|i8", "=i8":
                let need = h.dataOffset + n * 8
                guard need <= data.count else { throw NPYError.truncated(label) }
                let p = base.bindMemory(to: Int64.self, capacity: n)
                let arr = Array(UnsafeBufferPointer(start: p, count: n)).map { Int32(truncatingIfNeeded: $0) }
                return MLXArray(arr, h.shape)
            default:
                throw NPYError.unsupportedDType("\(h.descr) in \(label)")
            }
        }
    }

    // MARK: - header dict parsing (tiny, format-specific — not a general Python-literal parser)

    private static func match(_ dict: String, key: String) -> String? {
        // '<key>': '<value>'
        guard let kr = dict.range(of: "'\(key)'") else { return nil }
        let after = dict[kr.upperBound...]
        guard let q1 = after.range(of: "'") else { return nil }
        let rest = after[q1.upperBound...]
        guard let q2 = rest.range(of: "'") else { return nil }
        return String(rest[..<q2.lowerBound])
    }

    private static func parseShape(_ dict: String) -> [Int] {
        guard let sr = dict.range(of: "'shape'") else { return [] }
        let after = dict[sr.upperBound...]
        guard let open = after.range(of: "("), let close = after.range(of: ")") else { return [] }
        let inner = after[open.upperBound..<close.lowerBound]
        return inner.split(whereSeparator: { $0 == "," })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .compactMap { $0.isEmpty ? nil : Int($0) }
    }
}
