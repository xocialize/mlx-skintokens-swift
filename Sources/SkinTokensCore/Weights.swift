import Foundation
import MLX

/// Load a `.safetensors` file into a `[String: MLXArray]` on the CPU stream (Metal-watchdog rule:
/// weight loads ride the CPU stream). Values keep their on-disk dtype (bf16 stays bf16); the parity
/// gates upcast per-op inside the fp32 math. Used by every `--sNN-gate` CLI mode.
public enum Weights {
    public static func load(_ path: String) throws -> [String: MLXArray] {
        let url = URL(fileURLWithPath: path)
        var out: [String: MLXArray] = [:]
        try Device.withDefaultDevice(.cpu) {
            let arrays = try MLX.loadArrays(url: url)
            for (k, v) in arrays { out[k] = v }
            eval(Array(out.values))
        }
        return out
    }
}
