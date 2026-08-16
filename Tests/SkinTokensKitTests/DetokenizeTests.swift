import XCTest
@testable import SkinTokensCore

/// Offline (no-MLX) tests for the S2b/S4 pure-integer pieces: skeleton detokenization, the beam
/// decoder's mask-class reduction, and the grammar violation counter. These never eval a kernel, so
/// they live in `swift test`; the numeric beam decode itself is gated by the `--s2b-gate` CLI mode.
final class DetokenizeTests: XCTestCase {

    let L = SkinTokenGrammar.skinTokens   // numDiscrete 256, bos 257, eos 258, clsNone 263, cls {264,265,266}

    /// A minimal two-joint chain: bos, cls_none, root triple, child triple, eos.
    func testDetokenizeSimpleChain() throws {
        let ids = [L.bos, L.clsNone, 10, 20, 30, 40, 50, 60, L.eos]
        let skel = try SkinDetokenize.detokenize(ids, layout: L)
        XCTAssertEqual(skel.J, 2)
        XCTAssertEqual(skel.parents, [-1, 0])
        XCTAssertNil(skel.cls)
        // undiscretize(10) = (10.5/256)*2 - 1
        XCTAssertEqual(skel.joints[0][0], (10.5 / 256) * 2 - 1, accuracy: 1e-6)
        XCTAssertEqual(skel.treeDepth, 2)
    }

    /// A branch: after two chained joints, `branch` + parent-triple + joint-triple re-parents to an
    /// EARLIER joint (nearest head position), not the immediately preceding one.
    func testDetokenizeBranchReparents() throws {
        // root@(10,10,10) → child@(200,200,200); then branch back onto root → leaf@(20,20,20)
        let ids = [L.bos, L.clsNone,
                   10, 10, 10,
                   200, 200, 200,
                   L.branch, 10, 10, 10, 20, 20, 20,
                   L.eos]
        let skel = try SkinDetokenize.detokenize(ids, layout: L)
        XCTAssertEqual(skel.J, 3)
        // joint 2 (the branch joint) parents to joint 0 (root), whose head matches the branch triple.
        XCTAssertEqual(skel.parents, [-1, 0, 0])
    }

    func testDetokenizeReadsClsToken() throws {
        let ids = [L.bos, L.clsTokens[2], 10, 20, 30, L.eos]   // articulation cls
        let skel = try SkinDetokenize.detokenize(ids, layout: L)
        XCTAssertEqual(skel.cls, L.clsTokens[2])
        XCTAssertEqual(skel.J, 1)
        XCTAssertEqual(skel.parents, [-1])
    }

    /// The beam decoder's MaskClass must agree with `decision(sequence:)` across the phases.
    func testMaskClassMatchesDecision() throws {
        // skeleton phase
        let skelSeq = [L.bos, L.clsNone]
        let mc = try SkinTokenGrammar.maskClass(sequence: skelSeq, tokensPerSkin: 4, layout: L)
        if case .skeleton(let st) = mc {
            XCTAssertEqual(SkinTokenGrammar.allowedTokens(state: st, layout: L),
                           try SkinTokenGrammar.nextPossibleTokens(prefix: skelSeq, layout: L))
        } else { XCTFail("expected skeleton mask class") }
        // skin phase (switch present, not yet complete)
        let skinSeq = [L.bos, L.clsNone, 10, 20, 30, L.eos, 300, 301]
        XCTAssertEqual(try SkinTokenGrammar.maskClass(sequence: skinSeq, tokensPerSkin: 4, layout: L), .skinContinue)
    }

    /// force-EOS fires at exactly J×tokensPerSkin tokens past the switch.
    func testForceEosBoundary() throws {
        // one joint (J=1) → J*4 = 4 skin tokens then EOS. switch at index 5.
        var seq = [L.bos, L.clsNone, 10, 20, 30, L.eos]   // J=1, switch at idx 5
        XCTAssertEqual(try SkinTokenGrammar.bonesInSequence(seq, layout: L), 1)
        seq += [300, 301, 302]                            // 3 skin tokens → (count-switch)=4 == J*4 → forceEos
        XCTAssertEqual(try SkinTokenGrammar.maskClass(sequence: seq, tokensPerSkin: 4, layout: L), .forceEos)
    }

    /// tokenize → detokenize roundtrips a skeleton: parents EXACT, coords to the 256-bin floor,
    /// branch structure preserved (a re-parent emits `branch` + parent triple).
    func testTokenizeDetokenizeRoundtrip() throws {
        // root → child (chain) → leaf that re-parents to root (branch).
        let joints: [[Float]] = [[0, 0, 0], [0.2, 0.3, -0.1], [-0.4, 0.1, 0.5]]
        let parents = [-1, 0, 0]   // joint 2 parents to root (not the previous joint 1) → branch
        let toks = SkinDetokenize.tokenize(joints: joints, parents: parents, cls: L.clsTokens[1], layout: L)
        XCTAssertEqual(SkinGen.countViolations(toks, tokensPerSkin: 4, layout: L), 0)
        XCTAssertTrue(toks.contains(L.branch))   // joint 2 is a branch
        let skel = try SkinDetokenize.detokenize(toks, layout: L)
        XCTAssertEqual(skel.J, 3)
        XCTAssertEqual(skel.parents, parents)
        XCTAssertEqual(skel.cls, L.clsTokens[1])
        for i in 0..<3 { for k in 0..<3 { XCTAssertEqual(skel.joints[i][k], joints[i][k], accuracy: 0.01) } }
    }

    /// countViolations is 0 on a grammar-valid stream and positive when a token breaks the DFA.
    func testViolationCounter() {
        let valid = [L.bos, L.clsNone, 10, 20, 30, L.eos, 300, 301, 302, L.lmEos]
        XCTAssertEqual(SkinGen.countViolations(valid, tokensPerSkin: 4, layout: L), 0)
        let bad = [L.bos, L.clsNone, 10, 20, 30, L.eos, 5 /* coord token in skin phase */, 301, 302, L.lmEos]
        XCTAssertGreaterThan(SkinGen.countViolations(bad, tokensPerSkin: 4, layout: L), 0)
    }
}
