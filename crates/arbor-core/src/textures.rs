//! Reading a species' texture maps off disk, for anything that draws or exports it.
//!
//! The viewer and the exporters have to agree exactly on what a species' leaves look
//! like, and for a clustering species that is not a file at all: the art on disk is one
//! leaf and the atlas is grown from it. Loading lives here so there is one place that
//! does it.

use std::path::Path;

use image::ImageEncoder as _;

use crate::cluster::{bake_cluster, BakedMaps, Bitmap, LeafMaps};
use crate::species::LeafParams;

/// Where the texture maps are kept, relative to the repository root.
pub const TEXTURE_DIR: &str = "assets/textures";

/// `<dir>/<name>_<map>.png`, the one naming scheme every map follows.
pub fn map_path(dir: &Path, name: &str, map: &str) -> std::path::PathBuf {
    dir.join(format!("{name}_{map}.png"))
}

/// One image, decoded to RGBA8. `None` when it is missing or unreadable, which every
/// caller treats as the map being absent rather than as an error.
pub fn load_bitmap(path: &Path) -> Option<Bitmap> {
    let rgba = image::open(path).ok()?.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Bitmap::from_rgba(w, h, rgba.into_raw())
}

/// The albedo, normal and roughness maps a set of bark is drawn with. `None` when there
/// is no albedo; a missing normal or roughness map is just left out.
pub fn load_bark_maps(dir: &Path, name: &str) -> Option<BakedMaps> {
    Some(BakedMaps {
        albedo: load_bitmap(&map_path(dir, name, "albedo"))?,
        normal: load_bitmap(&map_path(dir, name, "normal")),
        roughness: load_bitmap(&map_path(dir, name, "roughness")),
    })
}

/// The maps a species' leaf cards are drawn with: the art as it is on disk, or, when
/// the species clusters, the atlas grown from it.
pub fn load_leaf_maps(dir: &Path, lp: &LeafParams) -> Option<BakedMaps> {
    let albedo = load_bitmap(&map_path(dir, &lp.texture, "albedo"))?;
    let normal = load_bitmap(&map_path(dir, &lp.texture, "normal"));
    let roughness = load_bitmap(&map_path(dir, &lp.texture, "roughness"));
    Some(match &lp.cluster {
        Some(cluster) => bake_cluster(
            cluster,
            lp.atlas_cols,
            lp.atlas_rows,
            LeafMaps {
                albedo: &albedo,
                normal: normal.as_ref(),
                roughness: roughness.as_ref(),
            },
        ),
        None => BakedMaps {
            albedo,
            normal,
            roughness,
        },
    })
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
