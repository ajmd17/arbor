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

/// Somewhere texture maps are read from, by file name.
pub trait MapSource {
    /// The bytes of `file`, or `None` when there is no such file.
    fn read(&self, file: &str) -> Option<Vec<u8>>;
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

/// An image file, decoded to RGBA8. `None` when it is unreadable, which every caller
/// treats as the map being absent rather than as an error.
pub fn decode(bytes: &[u8]) -> Option<Bitmap> {
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Bitmap::from_rgba(w, h, rgba.into_raw())
}

/// One map of the set called `name`, decoded.
pub fn load_bitmap<S: MapSource + ?Sized>(source: &S, name: &str, map: &str) -> Option<Bitmap> {
    decode(&source.read(&map_file(name, map))?)
}

/// The albedo, normal and roughness maps a set of bark is drawn with. `None` when there
/// is no albedo; a missing normal or roughness map is just left out.
pub fn load_bark_maps<S: MapSource + ?Sized>(source: &S, name: &str) -> Option<BakedMaps> {
    Some(BakedMaps {
        albedo: load_bitmap(source, name, "albedo")?,
        normal: load_bitmap(source, name, "normal"),
        roughness: load_bitmap(source, name, "roughness"),
    })
}

/// The maps a species' leaf cards are drawn with: the art as it is on disk, or, when
/// the species clusters, the atlas grown from it.
pub fn load_leaf_maps<S: MapSource + ?Sized>(source: &S, lp: &LeafParams) -> Option<BakedMaps> {
    let albedo = load_bitmap(source, &lp.texture, "albedo")?;
    let normal = load_bitmap(source, &lp.texture, "normal");
    let roughness = load_bitmap(source, &lp.texture, "roughness");
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
