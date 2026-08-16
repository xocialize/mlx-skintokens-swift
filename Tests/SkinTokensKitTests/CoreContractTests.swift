// CoreContractTests.swift — pure structural tests for the S0 substrate + shared fixture reader.
// NEVER constructs an MLXArray or evals, so it is robust in the SPM test target regardless of the
// metallib state (swift-port-parity.md). The MLXArray materialization path is exercised by the GPU
// CLI lane (`RunSkinTokens --npy`), not here.

import Foundation
import XCTest
@testable import SkinTokensCore

final class CoreContractTests: XCTestCase {

    // MARK: - NPYReader.parseHeader (header-only; no payload materialization)

    /// Build a valid `.npy` v1.0 header (64-byte aligned, dict ends with '\n').
    private func npyBytes(descr: String, shape: [Int], fortran: Bool) -> Data {
        let shapeStr = shape.count == 1
            ? "(\(shape[0]),)"
            : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        let base = "{'descr': '\(descr)', 'fortran_order': \(fortran ? "True" : "False"), 'shape': \(shapeStr), }"
        let pad = (64 - ((10 + base.count + 1) % 64)) % 64
        let dict = base + String(repeating: " ", count: pad) + "\n"
        let dictBytes = Array(dict.utf8)
        var out: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00]
        out.append(UInt8(dictBytes.count & 0xff))
        out.append(UInt8((dictBytes.count >> 8) & 0xff))
        out += dictBytes
        return Data(out)   // payload omitted — parseHeader never reads it
    }

    func testNPYHeaderCOrder() throws {
        let h = try NPYReader.parseHeader(npyBytes(descr: "<f4", shape: [54000, 3], fortran: false))
        XCTAssertEqual(h.descr, "<f4")
        XCTAssertEqual(h.shape, [54000, 3])
        XCTAssertEqual(h.count, 162000)
    }

    func testNPYHeader1D() throws {
        let h = try NPYReader.parseHeader(npyBytes(descr: "<i4", shape: [498], fortran: false))
        XCTAssertEqual(h.descr, "<i4")
        XCTAssertEqual(h.shape, [498])
    }

    func testNPYRejectsFortranOrder() {
        XCTAssertThrowsError(try NPYReader.parseHeader(npyBytes(descr: "<f4", shape: [4, 5], fortran: true))) {
            guard case NPYReader.NPYError.fortranOrder = $0 else {
                return XCTFail("expected .fortranOrder, got \($0)")
            }
        }
    }

    func testNPYRejectsNonNPY() {
        XCTAssertThrowsError(try NPYReader.parseHeader(Data("not a numpy file".utf8))) {
            guard case NPYReader.NPYError.notNPY = $0 else {
                return XCTFail("expected .notNPY, got \($0)")
            }
        }
    }

    // MARK: - SafetensorsHeader

    private func writeSafetensors(_ entries: [(key: String, dtype: String, shape: [Int])]) throws -> String {
        var obj: [String: Any] = [:]
        for e in entries {
            obj[e.key] = ["dtype": e.dtype, "shape": e.shape, "data_offsets": [0, 0]]
        }
        let json = try JSONSerialization.data(withJSONObject: obj)
        var data = Data()
        var len = UInt64(json.count).littleEndian
        withUnsafeBytes(of: &len) { data.append(contentsOf: $0) }
        data.append(json)
        let path = FileManager.default.temporaryDirectory
            .appending(path: "st-\(UUID().uuidString).safetensors")
        try data.write(to: path)
        return path.path
    }

    func testSafetensorsHeaderReadsKeysDtypesShapes() throws {
        let path = try writeSafetensors([
            ("model.embed_tokens.weight", "BF16", [33036, 896]),
            ("model.layers.0.self_attn.q_norm.weight", "BF16", [128]),
        ])
        defer { try? FileManager.default.removeItem(atPath: path) }
        let info = try SafetensorsHeader.read(path)
        XCTAssertEqual(info.count, 2)
        XCTAssertEqual(info["model.embed_tokens.weight"],
                       SafetensorsHeader.TensorInfo(dtype: "BF16", shape: [33036, 896]))
        XCTAssertEqual(try SafetensorsHeader.keys(path),
                       ["model.embed_tokens.weight", "model.layers.0.self_attn.q_norm.weight"])
    }

    func testSafetensorsExcludesMetadata() throws {
        // __metadata__ is not a tensor; the header reader must skip it.
        let json = try JSONSerialization.data(withJSONObject: [
            "__metadata__": ["format": "pt"],
            "w": ["dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]],
        ])
        var data = Data()
        var len = UInt64(json.count).littleEndian
        withUnsafeBytes(of: &len) { data.append(contentsOf: $0) }
        data.append(json)
        let path = FileManager.default.temporaryDirectory
            .appending(path: "st-\(UUID().uuidString).safetensors")
        try data.write(to: path)
        defer { try? FileManager.default.removeItem(at: path) }
        XCTAssertEqual(try SafetensorsHeader.keys(path.path), ["w"])
    }

    // MARK: - KeyContract (S0)

    func testQwen3BackboneKeyCount() {
        // 2 globals (embed_tokens, norm) + 11 per layer; tied embeddings ⇒ no lm_head.
        let keys = KeyContract.qwen3BackboneKeys(numLayers: 2)
        XCTAssertEqual(keys.count, 2 + 2 * 11)
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.q_norm.weight"),
                      "Qwen3 carries per-head QK RMSNorm — the marker vs Qwen2")
        XCTAssertTrue(keys.contains("model.embed_tokens.weight"))
        XCTAssertFalse(keys.contains("lm_head.weight"))
    }

    func testQwen3UntiedAddsLMHead() {
        let keys = KeyContract.qwen3BackboneKeys(numLayers: 1, tieWordEmbeddings: false)
        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testKeyContractDiff() {
        let expected: Set<String> = ["a", "b", "c"]
        XCTAssertTrue(KeyContract.diff(expected: expected, onDisk: expected).isClean)

        let dirty = KeyContract.diff(expected: expected, onDisk: ["a", "b", "z"])
        XCTAssertFalse(dirty.isClean)
        XCTAssertEqual(dirty.missing, ["c"])
        XCTAssertEqual(dirty.unused, ["z"])
    }
}
