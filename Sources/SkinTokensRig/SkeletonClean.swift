import Foundation

/// Clean a VRM/template skeleton down to the DEFORMING HUMANOID bones the SkinTokens model was trained on.
///
/// ROOT-CAUSE FIX (2026-07-11, CLI-proven): feeding the model a raw VRoid skin.joints list — armature
/// `Root` at index 0, `*_end` leaves, and ~60 interleaved secondary bones (`J_Sec_*` hair/skirt/bust,
/// `J_Adj_*` eyes) — is out-of-distribution (the reference trains/infers on bpy-parsed deforming skeletons)
/// → the skin decode binds vertices to those junk bones → the mesh shatters under animation. Keeping ONLY
/// the `J_Bip_*` humanoid bones (body + hands, ~52), DFS-reordered (parent-before-child → the compact
/// branch structure the model expects), drops NN dominant-joint noise from ~26% to ~4% (BEATS the Python
/// reference's 6.6% and hand-authored 5.1%). Dropped-bone vertices (hair/skirt) simply bind to the nearest
/// surviving body bone (e.g. hair → Head) — correct for body-only animation.
public enum SkeletonClean {

    /// Fallback keep-predicate when no explicit deforming set is supplied: VRoid names humanoid bones
    /// `J_Bip_*`. NOTE the bundled template + VRM humanoid maps use STANDARD names (head/hips/…), so
    /// prefer the `keepIndices` overload (the humanoid map's values) — this is only the last resort.
    public static func isDeformingHumanoid(_ name: String) -> Bool { name.hasPrefix("J_Bip_") }

    /// Clean + DFS-reorder a skeleton. Returns `order` (surviving ORIGINAL joint indices in the new
    /// canonical sequence) and `parents` (in NEW-index space, -1 for root). Reindex any parallel arrays
    /// (joints, node ids, names, inverse-bind, humanoid map) through `order`; the model's skin columns
    /// then align 1:1 with the new order. Falls back to keep-all if the filter would empty the skeleton
    /// so we never produce a boneless rig.
    ///
    /// `keepIndices` = the authoritative deforming-bone set (pass the humanoid map's `values`); when nil,
    /// falls back to the `keepFn` name predicate. Prefer `keepIndices` — naming conventions vary
    /// (raw VRoid = `J_Bip_*`, bundled template + humanoid maps = standard `head`/`hips`/`leftEye`/…).
    public static func humanoidReorder(names: [String], parents: [Int],
                                       keepIndices: Set<Int>? = nil,
                                       keep keepFn: (String) -> Bool = isDeformingHumanoid)
        -> (order: [Int], parents: [Int]) {
        let J = names.count
        var kept = keepIndices.map { ks in (0..<J).filter { ks.contains($0) } }
            ?? (0..<J).filter { keepFn(names[$0]) }
        if kept.isEmpty { kept = Array(0..<J) }          // safety: unknown naming → keep everything
        let keptSet = Set(kept)
        func keptParent(_ i: Int) -> Int { var p = parents[i]; while p >= 0 && !keptSet.contains(p) { p = parents[p] }; return p }
        var children = [Int: [Int]](); var roots: [Int] = []
        for i in kept { let p = keptParent(i); if p == -1 { roots.append(i) } else { children[p, default: []].append(i) } }
        var order: [Int] = []
        func dfs(_ i: Int) { order.append(i); for c in (children[i] ?? []).sorted() { dfs(c) } }
        for r in roots.sorted() { dfs(r) }
        let origToNew = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
        let newParents = order.map { keptParent($0) == -1 ? -1 : origToNew[keptParent($0)]! }
        return (order, newParents)
    }
}
