import Foundation

/// Resolved, USD-authoring-friendly material info: the base-color texture (as
/// extractable image bytes) plus factors and alpha mode. VRoid materials are
/// `KHR_materials_unlit` with the color carried entirely by the base-color
/// texture — that texture is what makes the avatar look right.
public struct ResolvedMaterial: Sendable {
    public let index: Int
    public let name: String
    public let baseColorFactor: SIMD4<Float>
    /// Image index for the base-color texture, if any.
    public let baseColorImage: Int?
    public let alphaMode: String     // OPAQUE | MASK | BLEND
    public let alphaCutoff: Float
    public let doubleSided: Bool
    public let unlit: Bool
}

/// Raw image bytes extracted from the glTF, ready to write to disk for USDZ.
public struct ExtractedImage: Sendable {
    public let index: Int
    public let data: Data
    public let fileExtension: String   // "png" | "jpg"
}

public extension GLTFAsset {

    /// Resolves a material index into a USD-friendly description.
    func resolvedMaterial(_ index: Int) -> ResolvedMaterial? {
        guard let mats = document.materials, mats.indices.contains(index) else { return nil }
        let m = mats[index]
        let pbr = m.pbrMetallicRoughness
        let bcf = pbr?.baseColorFactor ?? [1, 1, 1, 1]
        let factor = SIMD4<Float>(bcf.count > 0 ? bcf[0] : 1, bcf.count > 1 ? bcf[1] : 1,
                                  bcf.count > 2 ? bcf[2] : 1, bcf.count > 3 ? bcf[3] : 1)
        // baseColorTexture → texture → image
        var imageIndex: Int?
        if let texIdx = pbr?.baseColorTexture?.index,
           let textures = document.textures, textures.indices.contains(texIdx),
           let src = textures[texIdx].source {
            imageIndex = src
        }
        return ResolvedMaterial(
            index: index,
            name: m.name ?? "material_\(index)",
            baseColorFactor: factor,
            baseColorImage: imageIndex,
            alphaMode: m.alphaMode ?? "OPAQUE",
            alphaCutoff: m.alphaCutoff ?? 0.5,
            doubleSided: m.doubleSided ?? false,
            unlit: m.isUnlit)
    }

    /// Extracts an image's raw bytes (from a GLB buffer view or a data: URI).
    func extractImage(_ index: Int) throws -> ExtractedImage? {
        guard let images = document.images, images.indices.contains(index) else { return nil }
        let img = images[index]
        let ext = (img.mimeType == "image/jpeg") ? "jpg" : "png"

        if let bvIndex = img.bufferView,
           let bvs = document.bufferViews, bvs.indices.contains(bvIndex) {
            let bv = bvs[bvIndex]
            guard buffers.indices.contains(bv.buffer) else { return nil }
            let buf = buffers[bv.buffer]
            let start = buf.startIndex + (bv.byteOffset ?? 0)
            let end = start + bv.byteLength
            guard end <= buf.endIndex else { return nil }
            return ExtractedImage(index: index, data: buf.subdata(in: start..<end), fileExtension: ext)
        }
        if let uri = img.uri, uri.hasPrefix("data:"),
           let comma = uri.firstIndex(of: ","),
           let data = Data(base64Encoded: String(uri[uri.index(after: comma)...])) {
            return ExtractedImage(index: index, data: data, fileExtension: ext)
        }
        return nil
    }
}
