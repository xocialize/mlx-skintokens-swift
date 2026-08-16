// ConformanceTests.swift — the offline C0–C13 + CAN gate for `SkinTokensPackage` (the `meshRig`
// conformer). No MLX kernels, no weights: the C-checks are static assertions on the manifest (the
// reviewable C0–C13 checklist encoded as tests), and the CAN checks drive the package's real `run()`
// under a pre-set cancellation + verify the declared checkpoint cadence. MAT-1..5 live in
// MaterializationTests.swift. Together these are the born-clean conformance suite.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import SkinTokensKit

final class ConformanceTests: XCTestCase {

    var manifest: PackageManifest { SkinTokensPackage.manifest }

    // MARK: - C0–C13 (static manifest checklist)

    /// C0 — the package declares the contract version it targets.
    func testC0_ContractVersionDeclared() {
        XCTAssertEqual(manifest.contractVersion, ContractVersion.current)
        // meshRig landed at 1.19.0 — the build must be at least that.
        XCTAssertGreaterThanOrEqual(manifest.contractVersion, SemanticVersion(major: 1, minor: 19, patch: 0))
    }

    /// C1 — capability registration: at least one canonical surface, and it's `meshRig`.
    func testC1_CapabilityRegistered() {
        XCTAssertEqual(manifest.capabilities, [.meshRig])
        XCTAssertEqual(manifest.surfaces.count, 1)
        XCTAssertEqual(manifest.surfaces.first?.capability, .meshRig)
    }

    /// C2 — canonical schema conformance: `meshRig`'s canonical output is `.mesh`, and the response
    /// type carries a `Mesh`.
    func testC2_CanonicalOutput() {
        XCTAssertEqual(Capability.meshRig.canonicalOutput, .mesh)
        let resp = MeshRigResponse(mesh: Mesh(format: .glb, data: Data()))
        XCTAssertEqual(resp.mesh.format, .glb)
    }

    /// C4 — modes are advertised as tags on the surface (auto/skinOnly), not separate capabilities.
    func testC4_ModesAdvertised() {
        let modes = Set(manifest.surfaces[0].supportedModes.map(\.rawValue))
        XCTAssertTrue(modes.contains(MeshRigContract.auto.rawValue))
        XCTAssertTrue(modes.contains(MeshRigContract.skinOnly.rawValue))
    }

    /// C2/descriptor — the parameter schema is the canonical surface: `mesh` required, `skeleton`
    /// optional, `mode` optional.
    func testDescriptorParameters() {
        let params = Dictionary(uniqueKeysWithValues: manifest.surfaces[0].parameters.map { ($0.name, $0) })
        XCTAssertEqual(params["mesh"]?.kind, .mesh)
        XCTAssertEqual(params["mesh"]?.required, true)
        XCTAssertEqual(params["skeleton"]?.required, false)
        XCTAssertEqual(params["mode"]?.required, false)
    }

    /// C7 + C8 — the two-layer license gate: SkinTokens weights (MIT) and our port (MIT) are both
    /// permissive, so the default `.permissiveOnly` policy admits the package.
    func testC7C8_TwoLayerLicensePermissive() {
        XCTAssertEqual(manifest.license.weightLicense, .mit)
        XCTAssertEqual(manifest.license.portCodeLicense, .mit)
    }

    /// C10 — device eligibility is declared separately from the tool surface: a footprint per quant,
    /// the required backend, an OS floor, a chip floor.
    func testC10_DeviceEligibility() {
        XCTAssertFalse(manifest.requirements.footprints.isEmpty)
        let fp = manifest.requirements.footprints[0]
        XCTAssertGreaterThan(fp.residentBytes, 0)
        XCTAssertGreaterThan(fp.peakActivationBytes, 0)   // split footprint (1.14 contract)
        XCTAssertTrue(manifest.requirements.requiredBackends.contains(.metalGPU))
    }

    /// Specialties advertise what the model is FOR (Model-Manager ranking), never a surface.
    func testSpecialtiesAdvertised() {
        let specs = Set(manifest.specialties.map { $0.specialty.rawValue })
        XCTAssertTrue(specs.contains("mesh-rigging"))
    }

    // MARK: - CAN-1..3 (executable cancellation gate)

    /// CAN-1 / CAN-2 — a pre-cancelled `run()` refuses at the entry checkpoint and surfaces the
    /// `CancellationError` unchanged (never `PackageError.notLoaded`, never a laundered package error).
    func testCAN12_EntryCheckpointPropagation() async {
        let pkg = SkinTokensPackage(configuration: SkinTokensConfiguration())   // not loaded — entry checkpoint must fire first
        let request = MeshRigRequest(mesh: Mesh(format: .glb, data: Data()))
        let report = await CancellationConformance.checkRun(package: pkg, request: request)
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — the declared checkpoint cadence matches the manifest's long-run envelope (`meshRig` is a
    /// long-run capability). We checkpoint per TOKEN in the `generate` phase (the beam loop calls the
    /// checkpoint every emitted token) + at stage seams.
    func testCAN3_CadenceDeclaration() {
        let report = CancellationConformance.checkCadence(
            manifest: manifest,
            posture: .cadence([.init(phase: .generate, unit: .token, reportsRunProgress: false)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
