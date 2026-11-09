# Third-party licenses

`mlx-skintokens-swift` is a Swift/MLX port of upstream code and loads upstream weights. No
third-party source is vendored byte-for-byte except where noted; the ported modules are
translations of the upstream PyTorch implementation.

| What | Origin | License | Copyright |
|---|---|---|---|
| Ported model + pipeline (`Sources/SkinTokensCore/`, `Sources/SkinTokensRig/`) | [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) | MIT | © 2025 VAST-AI-Research |
| Weights (downloaded at runtime, not in this repo) | [mlx-community/SkinTokens-bf16](https://huggingface.co/mlx-community/SkinTokens-bf16), converted from `VAST-AI/SkinTokens` | MIT | © 2025 VAST-AI-Research |
| Qwen3-0.6B backbone inside the TokenRig checkpoint (weights only) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0 | © Alibaba Cloud |
| `Sources/SkinTokensGLTF/` (vendored GLTFKit) | Xocialize's own VRM/glTF→USDZ converter (see `VENDORED.md`) | MIT (this repo's `LICENSE`) | © 2026 Xocialize |

The engine's two-layer licence gate sees MIT weights (C7) and an MIT port (C8), so the default
`.permissiveOnly` policy admits the package.

---

## SkinTokens

```
MIT License

Copyright (c) 2025 VAST-AI-Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen3-0.6B

Licensed under the Apache License, Version 2.0 — <https://www.apache.org/licenses/LICENSE-2.0>.
Only the weights are used (embedded in the converted TokenRig checkpoint); no Qwen source code is
included in this repository.
