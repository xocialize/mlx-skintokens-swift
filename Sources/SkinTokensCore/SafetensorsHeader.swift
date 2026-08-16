import Foundation

/// Reads a `.safetensors` **header only** (8-byte LE length + JSON) — pure Foundation, no MLX,
/// never evals. This is the S0 key-contract substrate (swift-port-parity.md §S0): pin per-component
/// key sets + dtypes + shapes from the actual file headers, before any model code, for every variant.
///
/// A `.safetensors` file is: `[u64 LE headerLen][headerLen bytes of JSON][raw tensor bytes]`. The
/// JSON maps `key -> {dtype, shape, data_offsets:[start,end]}`, plus an optional `__metadata__` key.
public enum SafetensorsHeader {

    public enum Error: Swift.Error, CustomStringConvertible {
        case unreadable(String)
        case badHeader(String)
        public var description: String {
            switch self {
            case .unreadable(let s): return "cannot read safetensors: \(s)"
            case .badHeader(let s): return "malformed safetensors header: \(s)"
            }
        }
    }

    public struct TensorInfo: Equatable {
        public let dtype: String     // "BF16", "F32", "F16", "I32", …
        public let shape: [Int]
    }

    /// key -> TensorInfo for every tensor in the file (excludes `__metadata__`).
    public static func read(_ path: String) throws -> [String: TensorInfo] {
        guard let fh = FileHandle(forReadingAtPath: path) else { throw Error.unreadable(path) }
        defer { try? fh.close() }
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
            throw Error.badHeader("\(path): no 8-byte length prefix")
        }
        var headerLen: UInt64 = 0
        for i in 0..<8 { headerLen |= UInt64(lenData[lenData.startIndex + i]) << (8 * UInt64(i)) }
        guard headerLen > 0, headerLen < 512_000_000 else {
            throw Error.badHeader("\(path): implausible header length \(headerLen)")
        }
        guard let json = try fh.read(upToCount: Int(headerLen)), json.count == Int(headerLen) else {
            throw Error.badHeader("\(path): header truncated")
        }
        guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Error.badHeader("\(path): header is not a JSON object")
        }
        var out: [String: TensorInfo] = [:]
        for (key, value) in obj {
            if key == "__metadata__" { continue }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Any] else {
                throw Error.badHeader("\(path): bad entry for '\(key)'")
            }
            out[key] = TensorInfo(dtype: dtype, shape: shape.compactMap { ($0 as? NSNumber)?.intValue })
        }
        return out
    }

    /// Just the key set — the S0 refuse-partial-load contract compares this against the generated set.
    public static func keys(_ path: String) throws -> Set<String> { Set(try read(path).keys) }
}
