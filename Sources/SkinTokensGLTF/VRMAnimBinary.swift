import Foundation
import simd

// Compact binary encoder for VRMAClip — MUST match byte-for-byte the decoder in
// VRMRuntime/HumanoidAnimation.swift (`decodeBinary`). See that file for the full
// layout. Summary: magic "ANM1", version, flags (bit0 hasHips, bit1 hasHipsRest),
// duration, frameCount N, name, shared times[N], restRotations, bones (int16×4
// quats × N), hipsRest, hips translations. Quaternions are int16-quantized.

public extension VRMAClip {
    enum BinaryError: Error { case nonUniformTracks }

    /// Serializes to `.anim.bin`. Throws `nonUniformTracks` if the bone tracks
    /// don't all share one times array (the format's invariant) — the caller
    /// should emit JSON for that clip instead.
    func encodeBinary() throws -> Data {
        var w = ByteWriter()
        w.bytes(Array("ANM1".utf8))
        w.u16(1)

        guard let canonicalTimes = bones.values.first?.times else {
            w.u16(hipsRest != nil ? 2 : 0)
            w.f32(duration); w.u32(0); w.str(name)
            w.u16(0); w.u16(0)
            if let r = hipsRest, r.count == 3 { w.f32(r[0]); w.f32(r[1]); w.f32(r[2]) }
            return Data(w.out)
        }
        let n = canonicalTimes.count
        // Times must match by VALUE, not just count — the decoder hands the one
        // shared times array to every track, so two same-length-but-differently-
        // timed samplers would silently retime.
        for t in bones.values where t.rotations.count != n || t.times != canonicalTimes {
            throw BinaryError.nonUniformTracks
        }
        // A hips track that doesn't share the canonical timeline (or carries a
        // malformed entry) must NOT be silently dropped or under-write the layout
        // the decoder expects — kick the clip to JSON instead.
        if let h = hips, h.translations.count != n || h.times != canonicalTimes
            || !h.translations.allSatisfy({ $0.count == 3 }) {
            throw BinaryError.nonUniformTracks
        }

        var flags: UInt16 = 0
        if hips != nil { flags |= 1 }
        if hipsRest != nil { flags |= 2 }

        w.u16(flags)
        w.f32(duration)
        w.u32(UInt32(n))
        w.str(name)
        for t in canonicalTimes { w.f32(t) }

        w.u16(UInt16(restRotations.count))
        for (k, q) in restRotations { w.str(k); w.quat(q) }

        w.u16(UInt16(bones.count))
        for (k, t) in bones {
            w.str(k)
            for q in t.rotations { w.quat(q) }
        }

        if let r = hipsRest, r.count == 3 { w.f32(r[0]); w.f32(r[1]); w.f32(r[2]) }
        if flags & 1 != 0, let h = hips {
            // Arity was validated above — write exactly N vec3s, as the decoder expects.
            for v in h.translations { w.f32(v[0]); w.f32(v[1]); w.f32(v[2]) }
        }
        return Data(w.out)
    }
}

/// Little-endian byte accumulator. Mirrors VRMRuntime's private writer.
private struct ByteWriter {
    var out = [UInt8]()
    mutating func bytes(_ b: [UInt8]) { out += b }
    mutating func u16(_ v: UInt16) { out.append(UInt8(v & 0xff)); out.append(UInt8(v >> 8)) }
    mutating func u32(_ v: UInt32) { for k in 0..<4 { out.append(UInt8((v >> (8 * UInt32(k))) & 0xff)) } }
    mutating func f32(_ v: Float) { u32(v.bitPattern) }
    mutating func i16(_ v: Int16) { u16(UInt16(bitPattern: v)) }
    mutating func str(_ s: String) { let b = [UInt8](s.utf8); u16(UInt16(b.count)); out += b }
    /// Quantize a [x,y,z,w] quaternion to int16 (components clamped to [-1,1]).
    mutating func quat(_ q: [Float]) {
        let a = q.count == 4 ? q : [0, 0, 0, 1]
        for c in a { i16(Int16((max(-1, min(1, c)) * 32767).rounded())) }
    }
}
