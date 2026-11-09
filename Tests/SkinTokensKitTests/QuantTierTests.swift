// QuantTierTests.swift — offline checks for the v0.2.0 quant tiers. No MLX kernels, no weights.
// The engine picks the `QuantFootprint` matching `configuration.quant` for admission, so every tier
// the package will load must carry its own measured footprint, and rejected tiers must stay out.

import Foundation
import MLXToolKit
import XCTest
import SkinTokensCore
@testable import SkinTokensKit

final class QuantTierTests: XCTestCase {

    func testSupportedQuantsAreBf16AndInt8() {
        XCTAssertEqual(SkinTokensPackage.supportedQuants, [.bf16, .int8])
        // int4 was measured and rejected (≈27% of vertices reassigned to another bone, AB-R-0812).
        XCTAssertFalse(SkinTokensPackage.supportedQuants.contains(.int4))
    }

    func testEverySupportedQuantHasAFootprint() {
        let declared = Set(SkinTokensPackage.manifest.requirements.footprints.map(\.quant))
        XCTAssertEqual(declared, SkinTokensPackage.supportedQuants)
    }

    func testInt8IsLighterResidentThanBf16() throws {
        let fps = SkinTokensPackage.manifest.requirements.footprints
        let bf16 = try XCTUnwrap(fps.first { $0.quant == .bf16 })
        let int8 = try XCTUnwrap(fps.first { $0.quant == .int8 })
        XCTAssertLessThan(int8.residentBytes, bf16.residentBytes)
    }

    func testDefaultConfigurationIsBf16() {
        XCTAssertEqual(SkinTokensConfiguration().quant, .bf16)
    }

    /// The projection list the int8 tier quantizes: 28 layers × 7 projections + the untied rig head.
    func testProjectionNamesCoverTheBackbone() {
        let names = Qwen3.projectionNames()
        XCTAssertEqual(names.count, 28 * 7 + 1)
        XCTAssertTrue(names.contains("transformer.lm_head.weight"))
        XCTAssertFalse(names.contains { $0.contains("embed_tokens") })
    }
}
