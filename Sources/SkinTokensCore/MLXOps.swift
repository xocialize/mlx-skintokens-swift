import Foundation
import MLX
import MLXNN

/// Shared low-level ops for the SkinTokens parity ports (S1b–S2). Deliberately EXPLICIT (no
/// `MLXFast.*`) so every op stays on whatever stream is current — the parity gates pin `.cpu`
/// (weight loads + fixtures were dumped CPU-side), and `MLXFast` SDPA/layerNorm route to Metal even
/// under a CPU default → the ~1e-3 phantom-drift trap (swift-port-parity.md). Matmul + softmax +
/// mean/rsqrt on the CPU stream match PyTorch `mx.*` bit-for-bit (trellis2 lesson).
public enum SkinOps {

    /// PyTorch `nn.Linear`: weight is `[out, in]`, so `y = x @ Wᵀ (+ b)`.
    public static func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
        var y = matmul(x, w.swappedAxes(-1, -2))
        if let b { y = y + b }
        return y
    }

    /// `nn.LayerNorm(dim, eps=1e-5)` over the last axis (weight + bias). Two-pass, fp32 accumulation.
    public static func layerNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let xf = x.asType(.float32)
        let mu = mean(xf, axis: -1, keepDims: true)
        let xc = xf - mu
        let v = mean(xc * xc, axis: -1, keepDims: true)
        let y = xc * rsqrt(v + eps)
        return (y * w.asType(.float32) + b.asType(.float32)).asType(x.dtype)
    }

    /// `nn.RMSNorm(dim)` over the last axis (weight only, no bias). fp32 accumulation.
    public static func rmsNorm(_ x: MLXArray, _ w: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let xf = x.asType(.float32)
        let ms = mean(xf * xf, axis: -1, keepDims: true)
        let y = xf * rsqrt(ms + eps)
        return (y * w.asType(.float32)).asType(x.dtype)
    }

    /// Exact (erf) GELU — `nn.GELU()` default `approximate='none'`. NOT the tanh approximation.
    public static func geluErf(_ x: MLXArray) -> MLXArray { gelu(x) }

    /// Scaled-dot-product attention, explicit, softmax upcast to fp32.
    /// q: [B, H, Tq, Dh], k/v: [B, H, Tk, Dh]. scale = 1/sqrt(Dh) unless overridden.
    /// (Michelangelo applies `1/sqrt(sqrt(Dh))` to both q and k → the same `1/sqrt(Dh)` on the product.)
    public static func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float) -> MLXArray {
        let qf = q.asType(.float32), kf = k.asType(.float32), vf = v.asType(.float32)
        var w = matmul(qf, kf.swappedAxes(-1, -2)) * scale
        w = softmax(w, axis: -1)
        return matmul(w, vf).asType(q.dtype)
    }

    /// Split a packed head axis: x [B, T, C] → [B, H, T, C/H].
    public static func splitHeads(_ x: MLXArray, heads: Int) -> MLXArray {
        let s = x.shape
        return x.reshaped([s[0], s[1], heads, s[2] / heads]).swappedAxes(1, 2)
    }

    /// Merge heads: x [B, H, T, Dh] → [B, T, H*Dh].
    public static func mergeHeads(_ x: MLXArray) -> MLXArray {
        let s = x.shape          // [B, H, T, Dh]
        return x.swappedAxes(1, 2).reshaped([s[0], s[2], s[1] * s[3]])
    }

    /// Range-reduced sin/cos (fp32): reduce the argument mod 2π before the transcendental. High-freq
    /// Fourier embedders push arguments to ~128π; MLX's fp32 sin/cos range-reduction diverges from
    /// PyTorch's there (~0.02–0.8 abs), so reduce first — mathematically identical, numerically stable.
    private static let twoPi: Float = 2.0 * .pi
    public static func sinR(_ x: MLXArray) -> MLXArray {
        let a = x.asType(.float32)
        return sin(a - round(a / twoPi) * twoPi)
    }
    public static func cosR(_ x: MLXArray) -> MLXArray {
        let a = x.asType(.float32)
        return cos(a - round(a / twoPi) * twoPi)
    }

    /// Cosine similarity between two flattened arrays (fp32). The universal parity metric.
    public static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.asType(.float32).reshaped([-1])
        let bf = b.asType(.float32).reshaped([-1])
        let dot = sum(af * bf)
        let na = sqrt(sum(af * af))
        let nb = sqrt(sum(bf * bf))
        let c = dot / (na * nb + 1e-20)
        return c.item(Float.self)
    }

    /// Max-abs difference (fp32) — the tighter companion metric.
    public static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
