// KeyContractTests.swift — S0 key-contract structure, hermetic (no on-disk files, no MLX, never evals).
//
// These pin the generated key-set CARDINALITY and disjointness that the on-disk `--s0-gate` diff
// depends on. The actual refuse-partial-load diff against the converted headers runs in the CLI lane
// (`swift run RunSkinTokens --s0-gate <converted-dir>`), since it needs the multi-GB converted repo.
// Counts here are the ground-truth totals measured from converted/tokenrig.safetensors (672) and
// converted/skinvae.safetensors (252) on 2026-07-10.

import Foundation
import XCTest
@testable import SkinTokensCore

final class KeyContractTests: XCTestCase {

    func testComponentCardinalities() {
        XCTAssertEqual(KeyContract.tokenRigTransformerKeys.count, 311)  // Qwen3-0.6B (28L) + un-tied rig head
        XCTAssertEqual(KeyContract.michelangeloKeys().count, 106)       // 1 cross block + 8 self resblocks
        XCTAssertEqual(KeyContract.outputProjKeys.count, 3)             // Linear(w,b) + RMSNorm(w)
        XCTAssertEqual(KeyContract.skinVAEKeys(prefix: "vae.model").count, 252)
    }

    func testWholeFileCardinalities() {
        XCTAssertEqual(KeyContract.tokenRigKeys.count, 672)             // 311 + 106 + 3 + 252
        XCTAssertEqual(KeyContract.skinVAEStandaloneKeys.count, 252)
    }

    func testTokenRigComponentsAreDisjoint() {
        // union == sum ⟺ the four component sets share no key (so the whole-file diff is well-posed)
        let sets = [
            KeyContract.tokenRigTransformerKeys,
            KeyContract.michelangeloKeys(),
            KeyContract.outputProjKeys,
            KeyContract.skinVAEKeys(prefix: "vae.model"),
        ]
        let sum = sets.reduce(0) { $0 + $1.count }
        let union = sets.reduce(into: Set<String>()) { $0.formUnion($1) }
        XCTAssertEqual(sum, union.count, "TokenRig component key sets overlap")
        XCTAssertEqual(union.count, 672)
    }

    func testQwen3RigHeadNotTied() {
        // TokenRig ships BOTH the rig-vocab embedding and an un-tied rig-vocab head.
        XCTAssertTrue(KeyContract.tokenRigTransformerKeys.contains("transformer.lm_head.weight"))
        XCTAssertTrue(KeyContract.tokenRigTransformerKeys.contains("transformer.model.embed_tokens.weight"))
        // A tied backbone would drop lm_head.
        XCTAssertFalse(KeyContract.qwen3BackboneKeys(numLayers: 28, tieWordEmbeddings: true).contains("lm_head.weight"))
    }

    func testSkinVAEPrefixSwap() {
        // The embedded (vae.model) and standalone (model.model) sets differ ONLY by prefix.
        let embedded = KeyContract.skinVAEKeys(prefix: "vae.model")
        let standalone = KeyContract.skinVAEStandaloneKeys
        XCTAssertEqual(embedded.count, standalone.count)
        let remapped = Set(embedded.map { "model.model" + $0.dropFirst("vae.model".count) })
        XCTAssertEqual(remapped, standalone)
    }

    func testDiffDetectsMissingAndUnused() {
        let d = KeyContract.diff(expected: ["a", "b", "c"], onDisk: ["b", "c", "d"])
        XCTAssertEqual(d.missing, ["a"])
        XCTAssertEqual(d.unused, ["d"])
        XCTAssertFalse(d.isClean)
        XCTAssertTrue(KeyContract.diff(expected: ["a"], onDisk: ["a"]).isClean)
    }
}
