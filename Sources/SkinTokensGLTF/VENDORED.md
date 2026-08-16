# Vendored: GLTFKit

These files are a **verbatim vendored copy** of `GLTFKit` from
`~/Development/AnimeStudio/xocialize-convert-to-usdz/converter/Sources/GLTFKit`
(the native VRM/glTF→USDZ converter). Dependency-free (Foundation + simd only).

Reused by the SkinTokens Swift auto-rigger for the Route-B (VRM skin-only) path:
world-space VRM parse (node-transform composition via `GLTFSkeleton.world`), existing-
skeleton extraction (`buildSkeleton`, `humanoidBones`), and accessor decoding
(`AccessorDecoder`, `GLTFAsset.primitiveGeometry`).

⚠ VENDORED — not a live dependency. If the upstream GLTFKit changes materially, re-sync
by re-copying. Kept as a copy (not a path/shared package) to avoid dragging the converter's
SwiftUsd dependency into the MLX graph. Decision: user-approved 2026-07-10.
