// MaterializationTests.swift — the offline MAT gate (MAT-1..5): no network, no MLX kernels, no
// weights. Proves the born-clean `SkinTokensConfiguration` WeightSourcing declaration:
//   MAT-1 ModelStorable · MAT-2 non-empty sources · MAT-3 role/repo hygiene ·
//   MAT-4 fresh-machine posture (nil store + no override ⇒ the snapshot reports missing) ·
//   MAT-5 explicit paths satisfy (tiny probe files in a temp override).
// The `check` API takes `any PackageConfiguration`, so this runs in P1 WITHOUT the (deferred) meshRig
// ModelPackage conformer — SkinTokens is born MAT-clean at kickoff.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import SkinTokensKit

final class MaterializationTests: XCTestCase {

    /// A directory holding a tiny stand-in for every probe file of the consolidated snapshot.
    private func stagedSnapshotDir(subpath: String? = nil) throws -> (dir: URL, cleanup: URL) {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "skintokens-mat-\(UUID().uuidString)")
        var dir = root
        if let subpath { dir = root.appending(path: subpath) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in SkinTokensConfiguration.probeFiles {
            FileManager.default.createFile(atPath: dir.appending(path: file).path,
                                           contents: Data("probe".utf8))
        }
        return (dir, root)
    }

    // MARK: - MAT-1..5

    func testMATGate() throws {
        let fresh = SkinTokensConfiguration()   // fresh machine: no override, no store root
        let (dir, cleanup) = try stagedSnapshotDir()
        defer { try? FileManager.default.removeItem(at: cleanup) }
        let satisfied = SkinTokensConfiguration(weightsRootOverride: dir)

        let report = MaterializationConformance.check(
            freshConfiguration: fresh, satisfiedConfiguration: satisfied)
        XCTAssertTrue(report.passed, report.summary)
    }

    /// The materializer's own destination (`<root>/xocialize/skintokens-mlx`) reads back as present.
    func testStoreLayoutSatisfies() throws {
        let (_, root) = try stagedSnapshotDir(subpath: SkinTokensConfiguration.consolidatedRepo)
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = SkinTokensConfiguration()
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty,
                      "staged store layout should satisfy the snapshot source")
    }

    /// An override in the `<override>/<org>/<name>` layout also satisfies.
    func testOverrideRepoLayoutSatisfies() throws {
        let (_, root) = try stagedSnapshotDir(subpath: SkinTokensConfiguration.consolidatedRepo)
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = SkinTokensConfiguration(weightsRootOverride: root)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty)
    }

    /// A half-materialized snapshot must re-report missing (strict probe — no silent degrade).
    func testPartialSnapshotReportsMissing() throws {
        let (dir, root) = try stagedSnapshotDir(subpath: SkinTokensConfiguration.consolidatedRepo)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.removeItem(at: dir.appending(path: "skinvae.safetensors"))
        let cfg = SkinTokensConfiguration()
        XCTAssertFalse(cfg.missingWeightSources(storeRoot: root).isEmpty,
                       "a snapshot missing skinvae must still report the source missing")
    }

    /// Prewarm resolves the store layout for dir-less configs; an explicit override wins; a bare
    /// fresh config has nothing to prewarm.
    func testPrewarmPaths() {
        let root = URL(fileURLWithPath: "/tmp/models")
        let storeResolved = SkinTokensConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(storeResolved.prewarmPaths.count, 2)
        XCTAssertTrue(storeResolved.prewarmPaths[0].path.hasPrefix(
            root.appending(path: SkinTokensConfiguration.consolidatedRepo).path))

        let override = URL(fileURLWithPath: "/tmp/dev-weights")
        let overridden = SkinTokensConfiguration(modelsRootDirectory: root, weightsRootOverride: override)
        XCTAssertTrue(overridden.prewarmPaths.contains(override))

        XCTAssertTrue(SkinTokensConfiguration().prewarmPaths.isEmpty)
    }
}
