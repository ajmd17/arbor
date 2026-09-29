//! KTX2 images, decoded to eight-bit RGBA on the CPU.
//!
//! glTF carries them through `KHR_texture_basisu`. Two kinds are understood: Basis
//! Universal (ETC1S and UASTC, transcoded straight to RGBA by [`basisu`]), and the
//! uncompressed 8- and 16-bit formats. GPU block formats (BC7, ASTC and the like) are not:
//! they would need a decoder each, and the baker wants plain pixels anyway.
//!
//! Only the top mip level of a plain 2D image is read.

use std::io::Read;

use ktx2::{Format, SupercompressionScheme};

use crate::import::Image;

const MAGIC: [u8; 12] = [0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, b'\r', b'\n', 0x1A, b'\n'];

pub fn is_ktx2(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

pub fn decode(bytes: &[u8]) -> Result<Image, String> {
    let reader = ktx2::Reader::new(bytes).map_err(|e| format!("not a valid KTX2 file ({e:?})"))?;
    let h = reader.header();
    if h.pixel_depth > 1 || h.face_count > 1 || h.layer_count > 1 {
        return Err("only plain 2D KTX2 images can be used (no cube maps, arrays or volumes)".into());
    }
    let (w, ht) = (h.pixel_width as usize, h.pixel_height as usize);
    if w == 0 || ht == 0 || w.saturating_mul(ht) > 1 << 28 {
        return Err(format!("a {w}x{ht} image is not usable"));
    }
    let rgba = match h.format {
        // No format: the data is Basis Universal, in whichever codec.
        None => basis(bytes, w, ht)?,
        Some(format) => plain(&reader, format, w, ht)?,
    };
    Ok(Image { width: w as u32, height: ht as u32, rgba })
}

fn basis(bytes: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    use basisu::SourceFormat::{AstcHdr6x6, UastcHdr4x4, UastcHdr6x6};
    let tex = basisu::Transcoder::new(bytes).map_err(|e| format!("Basis Universal: {e:?}"))?;
    if matches!(tex.source_format(), UastcHdr4x4 | AstcHdr6x6 | UastcHdr6x6) {
        return Err("HDR Basis textures cannot be shown as eight-bit images".into());
    }
    let px = tex
        .transcode(0, basisu::TargetFormat::Rgba32, basisu::DecodeFlags::NONE)
        .map_err(|e| format!("Basis Universal: {e:?}"))?;
    if px.len() != w * h * 4 {
        return Err(format!("Basis Universal gave {} bytes for a {w}x{h} image", px.len()));
    }
    Ok(px)
}

/// How a plain format lays out a pixel: the channels it has and the bytes each takes, and
/// where red, green, blue and alpha are among them.
struct Layout {
    bytes: usize,
    /// The channel for each of red, green, blue and alpha, or `None` to fill.
    map: [Option<usize>; 4],
}

fn layout(format: Format) -> Option<Layout> {
    let rgba = |bytes, map| Some(Layout { bytes, map });
    match format {
        Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB => rgba(1, [Some(0), Some(1), Some(2), Some(3)]),
        Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => rgba(1, [Some(2), Some(1), Some(0), Some(3)]),
        Format::R8G8B8_UNORM | Format::R8G8B8_SRGB => rgba(1, [Some(0), Some(1), Some(2), None]),
        Format::B8G8R8_UNORM | Format::B8G8R8_SRGB => rgba(1, [Some(2), Some(1), Some(0), None]),
        Format::R8G8_UNORM | Format::R8G8_SRGB => rgba(1, [Some(0), Some(1), None, None]),
        Format::R8_UNORM | Format::R8_SRGB => rgba(1, [Some(0), Some(0), Some(0), None]),
        Format::R16G16B16A16_UNORM => rgba(2, [Some(0), Some(1), Some(2), Some(3)]),
        Format::R16G16B16_UNORM => rgba(2, [Some(0), Some(1), Some(2), None]),
        Format::R16G16_UNORM => rgba(2, [Some(0), Some(1), None, None]),
        Format::R16_UNORM => rgba(2, [Some(0), Some(0), Some(0), None]),
        _ => None,
    }
}

fn channels(format: Format) -> usize {
    match format {
        Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB | Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => 4,
        Format::R16G16B16A16_UNORM => 4,
        Format::R8G8B8_UNORM | Format::R8G8B8_SRGB | Format::B8G8R8_UNORM | Format::B8G8R8_SRGB => 3,
        Format::R16G16B16_UNORM => 3,
        Format::R8G8_UNORM | Format::R8G8_SRGB | Format::R16G16_UNORM => 2,
        _ => 1,
    }
}

fn plain(reader: &ktx2::Reader<&[u8]>, format: Format, w: usize, h: usize) -> Result<Vec<u8>, String> {
    let lay = layout(format).ok_or_else(|| {
        format!("{format:?} is not supported; use Basis Universal or an uncompressed 8- or 16-bit format")
    })?;
    let level = reader.levels().next().ok_or("the file has no image data")?;
    let data: Vec<u8> = match reader.header().supercompression_scheme {
        None => level.data.to_vec(),
        Some(SupercompressionScheme::Zstandard) => {
            let mut out = Vec::with_capacity(level.uncompressed_byte_length as usize);
            ruzstd::StreamingDecoder::new(level.data)
                .map_err(|e| format!("Zstandard: {e}"))?
                .read_to_end(&mut out)
                .map_err(|e| format!("Zstandard: {e}"))?;
            out
        }
        Some(other) => return Err(format!("{other:?} supercompression is not supported")),
    };
    let per_pixel = channels(format) * lay.bytes;
    if data.len() < w * h * per_pixel {
        return Err(format!("the image data is {} bytes, short of the {} a {w}x{h} image needs", data.len(), w * h * per_pixel));
    }
    let mut rgba = Vec::with_capacity(w * h * 4);
    for px in data.chunks_exact(per_pixel).take(w * h) {
        // Sixteen-bit channels keep their high byte, which is the top of a little-endian pair
        // at the second byte.
        let channel = |c: usize| px[c * lay.bytes + lay.bytes - 1];
        for (i, slot) in lay.map.iter().enumerate() {
            rgba.push(slot.map_or(if i == 3 { 255 } else { 0 }, channel));
        }
    }
    Ok(rgba)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A one-level, unsupercompressed 2D KTX2 file holding `data`.
    pub fn ktx2_file(format: u32, width: u32, height: u32, data: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(&MAGIC);
        for v in [format, 1, width, height, 0, 0, 1, 1, 0] {
            f.extend_from_slice(&v.to_le_bytes());
        }
        // A minimal data format descriptor: its total size, then nothing else.
        let dfd_offset = 80 + 24;
        let dfd = 4u32.to_le_bytes();
        let data_offset = dfd_offset + dfd.len() as u64;
        for v in [dfd_offset as u32, dfd.len() as u32, 0, 0] {
            f.extend_from_slice(&v.to_le_bytes());
        }
        f.extend_from_slice(&0u64.to_le_bytes());
        f.extend_from_slice(&0u64.to_le_bytes());
        for v in [data_offset, data.len() as u64, data.len() as u64] {
            f.extend_from_slice(&v.to_le_bytes());
        }
        f.extend_from_slice(&dfd);
        f.extend_from_slice(data);
        f
    }

    const R8G8B8A8_UNORM: u32 = 37;
    const B8G8R8A8_SRGB: u32 = 50;
    const R8G8B8_UNORM: u32 = 23;
    const R8_UNORM: u32 = 9;
    const R16G16B16A16_UNORM: u32 = 91;
    const BC7_UNORM_BLOCK: u32 = 145;

    #[test]
    fn a_file_is_recognised_by_its_magic() {
        assert!(is_ktx2(&ktx2_file(R8G8B8A8_UNORM, 1, 1, &[0; 4])));
        assert!(!is_ktx2(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn plain_eight_bit_formats_decode() {
        let px = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        let img = decode(&ktx2_file(R8G8B8A8_UNORM, 2, 2, &px)).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(img.rgba, px);

        // Blue first, and sRGB makes no difference to the bytes.
        let img = decode(&ktx2_file(B8G8R8A8_SRGB, 1, 1, &[30, 20, 10, 40])).unwrap();
        assert_eq!(img.rgba, [10, 20, 30, 40]);

        // No alpha means opaque; one channel means grey.
        let img = decode(&ktx2_file(R8G8B8_UNORM, 1, 1, &[1, 2, 3])).unwrap();
        assert_eq!(img.rgba, [1, 2, 3, 255]);
        let img = decode(&ktx2_file(R8_UNORM, 2, 1, &[7, 9])).unwrap();
        assert_eq!(img.rgba, [7, 7, 7, 255, 9, 9, 9, 255]);
    }

    #[test]
    fn sixteen_bit_channels_keep_their_high_byte() {
        // Little endian: 0x1200, 0x3400, 0x5600, 0xFF00.
        let img = decode(&ktx2_file(R16G16B16A16_UNORM, 1, 1, &[0, 0x12, 0, 0x34, 0, 0x56, 0, 0xFF])).unwrap();
        assert_eq!(img.rgba, [0x12, 0x34, 0x56, 0xFF]);
    }

    #[test]
    fn unsupported_and_damaged_files_are_errors_not_panics() {
        let bc7 = decode(&ktx2_file(BC7_UNORM_BLOCK, 4, 4, &[0; 16]));
        assert!(bc7.err().unwrap().contains("not supported"));
        // Too little data for the size the header claims.
        assert!(decode(&ktx2_file(R8G8B8A8_UNORM, 4, 4, &[0; 8])).err().unwrap().contains("short of"));
        // Cut off in the header.
        assert!(decode(&MAGIC).is_err());
        assert!(decode(b"not a ktx2 file at all, but long enough to have a header...............").is_err());
    }

    fn reference() -> image::RgbaImage {
        image::load_from_memory(include_bytes!("../tests/fixtures/reference.png")).unwrap().to_rgba8()
    }

    /// Mean absolute error per colour channel against the original, and whether the
    /// transparent hole in the middle and the opaque corner both came through.
    fn compare(img: &Image) -> (f32, bool) {
        let want = reference();
        assert_eq!((img.width, img.height), want.dimensions());
        let mut err = 0.0;
        for (a, b) in img.rgba.chunks(4).zip(want.pixels()) {
            for c in 0..3 {
                err += (a[c] as f32 - b.0[c] as f32).abs();
            }
        }
        let alpha_at = |x: u32, y: u32| img.rgba[((y * img.width + x) * 4 + 3) as usize];
        (err / (img.rgba.len() / 4 * 3) as f32, alpha_at(32, 32) < 32 && alpha_at(2, 2) > 224)
    }

    #[test]
    fn uastc_from_the_reference_encoder_decodes_close_to_the_original() {
        let img = decode(include_bytes!("../tests/fixtures/uastc.ktx2")).unwrap();
        let (error, alpha) = compare(&img);
        assert!(error < 4.0, "mean error {error}");
        assert!(alpha, "the hole and the opaque corner should survive");
    }

    #[test]
    fn etc1s_from_the_reference_encoder_decodes_with_alpha() {
        let img = decode(include_bytes!("../tests/fixtures/etc1s.ktx2")).unwrap();
        let (error, alpha) = compare(&img);
        // ETC1S is a small, lossy codec: this only says it is the picture and not noise.
        assert!(error < 25.0, "mean error {error}");
        assert!(alpha, "the hole and the opaque corner should survive");
    }
}
