import Foundation

/// S2b/S4 — skeleton detokenization: a SkinTokens skeleton token stream → joint head positions +
/// integer parent indices. Pure Foundation (no MLX eval). Faithful port of
/// `TokenizerPart.detokenize` + `make_skeleton`'s parent-build loop (the tail-extrusion branches only
/// affect bone TAILS for display — a glTF armature is a node hierarchy of joint heads, so JOINTS_0/
/// WEIGHTS_0 rigging needs only heads + parents). `J` here equals the skin-column count (num_bones).
public enum SkinDetokenize {

    public struct Skeleton: Sendable {
        public var joints: [[Float]]     // (J,3) head positions in normalized [-1,1]³ space
        public var parents: [Int]        // (J) parent index, -1 for root
        public var cls: Int?             // cls token id, or nil (cls_none)

        public var J: Int { joints.count }

        /// Tree depth = longest root→leaf chain (1-based: a lone root is depth 1).
        public var treeDepth: Int {
            var best = 0
            for i in 0..<parents.count {
                var d = 1, p = parents[i], guardN = 0
                while p >= 0 && guardN < parents.count { d += 1; p = parents[p]; guardN += 1 }
                best = max(best, d)
            }
            return best
        }

        /// X-axis extent of joints (symmetry sanity — a symmetric rig straddles 0).
        public var xBounds: (Float, Float) {
            let xs = joints.map { $0[0] }
            return (xs.min() ?? 0, xs.max() ?? 0)
        }
    }

    /// `undiscretize` (continuous_range [-1,1], num_discrete bins): bin → center coordinate.
    @inline(__always)
    static func undiscretize(_ t: Int, numDiscrete: Int, lo: Float = -1, hi: Float = 1) -> Float {
        (Float(t) + 0.5) / Float(numDiscrete) * (hi - lo) + lo
    }

    /// `discretize` (inverse): coordinate → bin index, `clip(round((t-lo)/(hi-lo)·nd), 0, nd-1)`.
    /// numpy `round` is round-half-to-even → match with `.toNearestOrEven`.
    @inline(__always)
    static func discretize(_ t: Float, numDiscrete: Int, lo: Float = -1, hi: Float = 1) -> Int {
        let b = ((t - lo) / (hi - lo) * Float(numDiscrete)).rounded(.toNearestOrEven)
        return max(0, min(numDiscrete - 1, Int(b)))
    }

    /// S5 — TOKENIZE a skeleton (joints/parents/cls) → a SkinTokens skeleton stream (the inverse of
    /// `detokenize`). Port of `TokenizerPart.tokenize` for the Route-B prefix: `bos`, cls, then per
    /// joint either 3 coord tokens (parent == previous joint) or `branch` + parent-triple + own-triple
    /// (a re-parent), terminated by the tokenizer `eos` (= the phase-switch token). Parts/order are
    /// omitted (Route-B doesn't supply joint names). `joints` MUST be in post-affine normalized space
    /// ([-1,1]³ — coords outside silently clamp). `cls` = a cls token id (e.g. vroid) or nil (cls_none).
    public static func tokenize(joints: [[Float]], parents: [Int], cls: Int?,
                                layout L: SkinTokenGrammar.Layout = SkinTokenGrammar.skinTokens) -> [Int] {
        let nd = L.numDiscrete
        func disc(_ v: [Float]) -> [Int] { [discretize(v[0], numDiscrete: nd), discretize(v[1], numDiscrete: nd), discretize(v[2], numDiscrete: nd)] }
        var tokens = [L.bos]
        tokens.append(cls ?? L.clsNone)
        for i in 0..<joints.count {
            // branch iff this joint's parent is NOT the immediately preceding joint (i-1); root never.
            let isBranch = i != 0 && parents[i] != i - 1
            if isBranch {
                tokens.append(L.branch)
                tokens.append(contentsOf: disc(joints[parents[i]]))   // parent position triple
            }
            tokens.append(contentsOf: disc(joints[i]))                // own position triple
        }
        tokens.append(L.eos)
        return tokens
    }

    /// Detokenize a FULL stream (start ⊕ generated). Slices out the skeleton (up to the first
    /// tokenizer-EOS / switch token) and walks it. Throws if the stream isn't bos-prefixed.
    public static func detokenize(_ ids: [Int], layout L: SkinTokenGrammar.Layout = SkinTokenGrammar.skinTokens) throws -> Skeleton {
        guard let first = ids.first, first == L.bos else { throw SkinTokenGrammar.GrammarError.notBos(ids.first ?? -1) }
        // skeleton = bos .. first switch(eos) inclusive; drop bos + trailing eos.
        let switchIdx = ids.firstIndex(of: L.switchTokenId) ?? ids.count
        let body = Array(ids[1..<switchIdx])   // between bos and switch

        var joints: [[Float]] = []
        var pJoints: [[Float]] = []
        var cls: Int? = nil
        var i = 0
        var isBranch = false
        var lastJoint: [Float]? = nil
        let nd = L.numDiscrete
        func coord(_ k: Int) -> [Float] { [undiscretize(body[k], numDiscrete: nd),
                                           undiscretize(body[k + 1], numDiscrete: nd),
                                           undiscretize(body[k + 2], numDiscrete: nd)] }
        while i < body.count {
            let id = body[i]
            if id < nd {
                let current: [Float]
                if isBranch {
                    let pj = coord(i)
                    current = coord(i + 3)
                    joints.append(current); pJoints.append(pj)
                    i += 6
                } else {
                    current = coord(i)
                    joints.append(current)
                    if pJoints.isEmpty { pJoints.append(current) }        // root → self
                    else { pJoints.append(lastJoint!) }                   // chain → previous joint
                    i += 3
                }
                lastJoint = current
                isBranch = false
            } else if id == L.branch {
                isBranch = true; lastJoint = nil; i += 1
            } else if id == L.spring || L.partsTokens.contains(id) {
                i += 1                                                    // part token (skip — display only)
            } else if L.clsTokens.contains(id) {
                cls = id; i += 1
            } else if id == L.clsNone {
                cls = nil; i += 1
            } else {
                i += 1                                                    // unexpected — skip defensively
            }
        }
        guard !joints.isEmpty else { return Skeleton(joints: [], parents: [], cls: cls) }

        // make_skeleton parent-build: parents[i] = argmin_{j<i} ||joints[j] - pJoints[i]||² (root=-1).
        var parents = [Int](repeating: -1, count: joints.count)
        for i in 1..<joints.count {
            let pj = pJoints[i]
            var best = -1; var bestD = Float.greatestFiniteMagnitude
            for j in stride(from: i - 1, through: 0, by: -1) {
                let h = joints[j]
                let d = (h[0]-pj[0])*(h[0]-pj[0]) + (h[1]-pj[1])*(h[1]-pj[1]) + (h[2]-pj[2])*(h[2]-pj[2])
                if d < bestD { bestD = d; best = j }
            }
            parents[i] = best
        }
        return Skeleton(joints: joints, parents: parents, cls: cls)
    }
}
