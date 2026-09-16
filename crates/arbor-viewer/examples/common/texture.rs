//! Shared plumbing for the asset tools: PNG in, PNG out, and the bridge to the
//! `Bitmap` that arbor-core's texture work speaks in.
//!
//! Not an example itself: cargo only picks up `examples/*.rs` and `examples/*/main.rs`,
//! so this is pulled in with `#[path]` by the tools that need it.

// Each tool uses the subset of these it needs, so not every helper is live in
// every binary that pulls the module in.
#![allow(dead_code)]

use arbor_core::cluster::Bitmap;
use image::RgbaImage;

pub fn open(path: &str) -> RgbaImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
        .to_rgba8()
}

pub fn to_bitmap(img: &RgbaImage) -> Bitmap {
    Bitmap::from_rgba(img.width(), img.height(), img.as_raw().clone())
        .expect("an RgbaImage is always four bytes a pixel")
}

pub fn from_bitmap(b: &Bitmap) -> RgbaImage {
    RgbaImage::from_raw(b.width, b.height, b.pixels.clone())
        .expect("a Bitmap is always four bytes a pixel")
}

pub fn save(img: &RgbaImage, dir: &std::path::Path, name: &str, suffix: &str) {
    let path = dir.join(format!("{name}_{suffix}.png"));
    img.save(&path)
        .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!("{} ({} KB)", path.display(), bytes / 1024);
}
