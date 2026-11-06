//! Reading a species' texture maps, for anything that draws or exports it.
//!
//! The viewer and the exporters have to agree exactly on what a species' leaves look
//! like, and for a clustering species that is not a file at all: the art on disk is one
//! leaf and the atlas is grown from it. Loading lives here so there is one place that
//! does it.
//!
//! Maps are read through a `MapSource`: a folder on disk, or, in the viewer on the web,
//! the files it has fetched.

use std::path::Path;

use image::ImageEncoder as _;

use crate::cluster::{bake_cluster, BakedMaps, Bitmap, LeafMaps};
use crate::species::LeafParams;

/// Where the texture maps are kept, relative to the repository root.
pub const TEXTURE_DIR: &str = "assets/textures";

/// Every map a set of textures can have. Only the albedo is needed.
pub const MAPS: [&str; 3] = ["albedo", "normal", "roughness"];

/// The map only foliage has: the colour of the light a leaf lets through.
pub const TRANSLUCENCY: &str = "translucency";

/// Somewhere texture maps are read from, by file name.
pub trait MapSource {
    /// The bytes of `file`, or `None` when there is no such file.
    fn read(&self, file: &str) -> Option<Vec<u8>>;

    /// `file` decoded. A source that holds pixels rather than files answers this
    /// directly, and `read` for it encodes them.
    fn read_bitmap(&self, file: &str) -> Option<Bitmap> {
        decode(&self.read(file)?)
    }
}

/// A folder of maps on disk.
impl MapSource for Path {
    fn read(&self, file: &str) -> Option<Vec<u8>> {
        std::fs::read(self.join(file)).ok()
    }
}

/// `<name>_<map>.png`, the one naming scheme every map follows.
pub fn map_file(name: &str, map: &str) -> String {
    format!("{name}_{map}.png")
}

/// Every file a set of textures called `name` may be read from.
pub fn map_files(name: &str) -> Vec<String> {
    MAPS.iter().map(|map| map_file(name, map)).collect()
}

/// Every file a set of leaf textures called `name` may be read from.
pub fn leaf_map_files(name: &str) -> Vec<String> {
    let mut files = map_files(name);
    files.push(map_file(name, TRANSLUCENCY));
    files
}

/// An image file, decoded to RGBA8. `None` when it is unreadable, which every caller
/// treats as the map being absent rather than as an error.
pub fn decode(bytes: &[u8]) -> Option<Bitmap> {
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Bitmap::from_rgba(w, h, rgba.into_raw())
}

/// One map of the set called `name`, decoded.
pub fn load_bitmap<S: MapSource + ?Sized>(source: &S, name: &str, map: &str) -> Option<Bitmap> {
    source.read_bitmap(&map_file(name, map))
}

/// The albedo, normal and roughness maps a set of bark is drawn with. `None` when there
/// is no albedo; a missing normal or roughness map is just left out.
pub fn load_bark_maps<S: MapSource + ?Sized>(source: &S, name: &str) -> Option<BakedMaps> {
    Some(BakedMaps {
        albedo: load_bitmap(source, name, "albedo")?,
        normal: load_bitmap(source, name, "normal"),
        roughness: load_bitmap(source, name, "roughness"),
        translucency: None,
    })
}

/// The maps a species' leaf cards are drawn with: the art as it is on disk, or, when
/// the species clusters, the atlas grown from it.
pub fn load_leaf_maps<S: MapSource + ?Sized>(source: &S, lp: &LeafParams) -> Option<BakedMaps> {
    let mut albedo = load_bitmap(source, &lp.texture, "albedo")?;
    let normal = load_bitmap(source, &lp.texture, "normal");
    let roughness = load_bitmap(source, &lp.texture, "roughness");
    let mut translucency = load_bitmap(source, &lp.texture, TRANSLUCENCY);
    saturate(&mut albedo, lp.saturation);
    if let Some(t) = translucency.as_mut() {
        saturate(t, lp.saturation);
        scale_colour(t, lp.transmission);
    }
    Some(match &lp.cluster {
        Some(cluster) => bake_cluster(
            cluster,
            lp.atlas_cols,
            lp.atlas_rows,
            LeafMaps {
                albedo: &albedo,
                normal: normal.as_ref(),
                roughness: roughness.as_ref(),
                translucency: translucency.as_ref(),
            },
        ),
        None => BakedMaps {
            albedo,
            normal,
            roughness,
            translucency,
        },
    })
}

/// Scales each colour channel of an sRGB map by its own factor, leaving its alpha. The
/// factor is on the light, as a shader multiplying the sampled colour applies it, so
/// it is taken out of sRGB, scaled and put back.
pub fn scale_channels(map: &mut Bitmap, k: [f32; 3]) {
    if k.iter().all(|&v| (v - 1.0).abs() < 1e-4) {
        return;
    }
    // One table per channel: the maps are millions of pixels and only 256 values.
    let tables = k.map(|f| {
        let mut t = [0u8; 256];
        for (i, out) in t.iter_mut().enumerate() {
            *out = linear_to_srgb8(srgb8_to_linear(i as u8) * f);
        }
        t
    });
    for px in map.pixels.chunks_exact_mut(4) {
        for (c, t) in px[..3].iter_mut().zip(&tables) {
            *c = t[*c as usize];
        }
    }
}

/// Scales a map's colour, leaving its alpha, by a factor on the light; see
/// `scale_channels`.
pub fn scale_colour(map: &mut Bitmap, k: f32) {
    scale_channels(map, [k; 3]);
}

/// `x.clamp(0.0, 255.0).round() as u8`, without the library call `f32::round` is on
/// targets where rounding half away from zero is not one instruction, which includes
/// the web and this toolchain's desktop. Bakes round millions of channels each.
/// The same byte for every input, NaN included: the fraction is exact below 2^23.
#[inline]
pub(crate) fn round_byte(x: f32) -> u8 {
    let x = x.clamp(0.0, 255.0);
    let whole = x as u32;
    (whole + u32::from(x - whole as f32 >= 0.5)) as u8
}

pub(crate) fn srgb8_to_linear(v: u8) -> f32 {
    let c = f32::from(v) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub(crate) fn linear_to_srgb8(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
    round_byte(c * 255.0)
}

#[cfg(test)]
mod round_byte_tests {
    use super::round_byte;

    #[test]
    fn rounds_half_away_from_zero() {
        // Not `f32::round` itself: this toolchain's C runtime rounds 0.49999997 up, as
        // floor(x + 0.5) does, where Rust defines it as 0 and wasm agrees.
        let reference = |x: f32| {
            let x = x.clamp(0.0, 255.0);
            let down = x.floor();
            (if x - down >= 0.5 { down + 1.0 } else { down }) as u8
        };
        // Every float from just under 0 to just over 255 a million steps apart, every
        // half-integer and its neighbours, and the ends.
        let mut probes: Vec<f32> = (0..=1_000_000).map(|k| -1.0 + k as f32 * 257.0 / 1e6).collect();
        for k in 0..=512 {
            let h = k as f32 * 0.5;
            probes.extend([h, f32::from_bits(h.to_bits().wrapping_sub(1)), f32::from_bits(h.to_bits() + 1)]);
        }
        probes.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, 1e30, -1e30]);
        for x in probes {
            assert_eq!(round_byte(x), reference(x), "at {x:?}");
        }
    }
}

/// Pulls a map's colour toward its own grey by `amount` kept, 1 leaving it as it is.
/// Done on the stored values, as the art was authored.
pub fn saturate(map: &mut Bitmap, amount: f32) {
    if (amount - 1.0).abs() < 1e-4 {
        return;
    }
    for px in map.pixels.chunks_exact_mut(4) {
        let [r, g, b] = [px[0], px[1], px[2]].map(f32::from);
        let grey = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        for (c, v) in px[..3].iter_mut().zip([r, g, b]) {
            *c = round_byte(grey + (v - grey) * amount);
        }
    }
}

/// How far, in rg chromaticity, a pixel can stray from the atlas's leaf colour and still
/// let light through: this many times as far as three in four of its pixels do, and
/// never less than the floor, so art with no twigs in it is not masked at all.
const MASK_LEAF_SPREAD: f32 = 1.5;
const MASK_LEAF_NEAR_MIN: f32 = 0.08;
/// How much further than that before a pixel lets no light through.
const MASK_LEAF_FALLOFF: f32 = 0.08;
/// Pixels this dark have no colour worth reading, and are taken as leaf.
const MASK_DARK_LO: f32 = 0.02;
const MASK_DARK_HI: f32 = 0.06;
/// The painted translucency that counts as letting all the light through: near the top
/// of the leaf's own range, so a few bright veins don't dim the rest of it.
const MASK_FULL_PERCENTILE: f32 = 0.95;

/// How much of the light behind each pixel of a leaf atlas comes through it, 0 to 255.
///
/// Painted translucency is used when there is one the size of the albedo. Without it,
/// the mask comes from the albedo: art photographed from a sprig has its twigs and
/// stalks in it, and those block the light. The leaf colour is the atlas's own, the
/// median chromaticity of its opaque pixels, so green needles and autumn leaves both
/// read as leaf, and how far counts as straying from it scales with how much the art's
/// own leaves vary, so only what stands apart from all of them is masked out.
pub fn transmission_mask(albedo: &Bitmap, translucency: Option<&Bitmap>) -> Vec<u8> {
    let opaque = |px: &[u8]| px[3] >= 128;
    if let Some(t) = translucency.filter(|t| t.width == albedo.width && t.height == albedo.height) {
        let luma = |px: &[u8]| 0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32;
        let mut lit: Vec<f32> = t
            .pixels
            .chunks_exact(4)
            .zip(albedo.pixels.chunks_exact(4))
            .filter(|(_, a)| opaque(a))
            .map(|(px, _)| luma(px))
            .collect();
        let full = percentile(&mut lit, MASK_FULL_PERCENTILE).max(1.0);
        return t.pixels.chunks_exact(4).map(|px| unorm(luma(px) / full)).collect();
    }

    let chroma = |px: &[u8]| {
        let sum = px[0] as f32 + px[1] as f32 + px[2] as f32;
        let inv = 1.0 / sum.max(1.0);
        ([px[0] as f32 * inv, px[1] as f32 * inv], sum / 765.0)
    };
    let voters: Vec<[f32; 2]> = albedo
        .pixels
        .chunks_exact(4)
        .filter(|px| opaque(px))
        .map(chroma)
        .filter(|&(_, brightness)| brightness >= MASK_DARK_HI)
        .map(|(c, _)| c)
        .collect();
    if voters.is_empty() {
        return vec![255; albedo.pixels.len() / 4];
    }
    let mut reds: Vec<f32> = voters.iter().map(|c| c[0]).collect();
    let mut greens: Vec<f32> = voters.iter().map(|c| c[1]).collect();
    let leaf = [percentile(&mut reds, 0.5), percentile(&mut greens, 0.5)];
    let off = |c: [f32; 2]| ((c[0] - leaf[0]).powi(2) + (c[1] - leaf[1]).powi(2)).sqrt();
    let mut spread: Vec<f32> = voters.iter().map(|&c| off(c)).collect();
    let near = (percentile(&mut spread, 0.75) * MASK_LEAF_SPREAD).max(MASK_LEAF_NEAR_MIN);
    albedo
        .pixels
        .chunks_exact(4)
        .map(|px| {
            let (c, brightness) = chroma(px);
            let off = off(c) * smoothstep(MASK_DARK_LO, MASK_DARK_HI, brightness);
            unorm(1.0 - smoothstep(near, near + MASK_LEAF_FALLOFF, off))
        })
        .collect()
}

/// The value `fraction` of the way through `values`, which it sorts.
fn percentile(values: &mut [f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * fraction).round() as usize]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn unorm(x: f32) -> u8 {
    round_byte(x.clamp(0.0, 1.0) * 255.0)
}

/// Encodes RGBA8 pixels as a PNG.
pub fn encode_png(bitmap: &Bitmap) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(
            &bitmap.pixels,
            bitmap.width,
            bitmap.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| format!("could not encode a {}x{} PNG: {e}", bitmap.width, bitmap.height))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-row atlas of `leaf` pixels with each of `others` after them.
    fn strip(leaf: [u8; 4], leaves: usize, others: &[[u8; 4]]) -> Bitmap {
        let mut pixels: Vec<u8> = (0..leaves).flat_map(|_| leaf).collect();
        pixels.extend(others.iter().flatten());
        Bitmap::from_rgba((leaves + others.len()) as u32, 1, pixels).unwrap()
    }

    const NEEDLE: [u8; 4] = [60, 90, 40, 255];
    const TWIG: [u8; 4] = [180, 140, 100, 255];

    #[test]
    fn the_twigs_in_a_sprig_block_the_light_and_the_needles_pass_it() {
        let mask = transmission_mask(&strip(NEEDLE, 20, &[TWIG]), None);
        assert_eq!(mask[0], 255);
        assert_eq!(mask[20], 0);
    }

    #[test]
    fn the_leaf_colour_is_the_atlases_own() {
        // Autumn art: the yellow is the leaf, and the green stray is what gets masked.
        let yellow = [200, 180, 40, 255];
        let mask = transmission_mask(&strip(yellow, 20, &[NEEDLE]), None);
        assert_eq!(mask[0], 255);
        assert!(mask[20] < 64, "{}", mask[20]);
    }

    #[test]
    fn shadowed_leaf_and_cut_out_pixels_dont_decide_anything() {
        let shade = [10, 5, 2, 255];
        let hole = [250, 10, 10, 0];
        let mask = transmission_mask(&strip(NEEDLE, 20, &[shade, hole]), None);
        assert_eq!(mask[20], 255, "a dark pixel was masked for its colour");
        let all_holes = transmission_mask(&strip(hole, 4, &[]), None);
        assert!(all_holes.iter().all(|&m| m == 255));
    }

    #[test]
    fn painted_translucency_wins_over_the_albedo() {
        let albedo = strip(NEEDLE, 3, &[TWIG]);
        let painted = Bitmap::from_rgba(4, 1, [[200; 4], [200; 4], [100; 4], [200; 4]].concat()).unwrap();
        let mask = transmission_mask(&albedo, Some(&painted));
        assert_eq!([mask[0], mask[1], mask[3]], [255; 3]);
        assert!(mask[2].abs_diff(128) <= 1, "{}", mask[2]);

        // A map that doesn't line up with the albedo can't say anything about it.
        let small = Bitmap::from_rgba(1, 1, vec![0; 4]).unwrap();
        assert_eq!(transmission_mask(&albedo, Some(&small))[0], 255);
    }
}
