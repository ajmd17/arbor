//! Conditions source PBR maps into the trimmed set the viewer loads from
//! `assets/textures`. Run it again to re-import from a different source pack.
//!
//! cargo run --release -p arbor-viewer --example import_textures -- \
//!     <out-name> --albedo a.png [--alpha m.png] [--normal n.png]
//!         [--rough r.png | --spec s.png] [--size 1024]
//!
//! The albedo gets its alpha from `--alpha` (red channel) and has its colour bled
//! outward into the transparent region.
//!
//! `--spec` takes a specular-intensity map instead of a roughness one and inverts it,
//! since bright specular means a smooth surface. Art authored for a specular workflow
//! ships that map rather than the roughness the shader here wants.

use image::{imageops, RgbaImage};

#[path = "common/texture.rs"]
mod texture;
use texture::{from_bitmap, open, save, to_bitmap};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(name) = args.first().filter(|a| !a.starts_with("--")).cloned() else {
        eprintln!("usage: import_textures <out-name> --albedo a.png [--alpha m.png] [--normal n.png] [--rough r.png] [--size N]");
        std::process::exit(2);
    };
    let opt = |key: &str| -> Option<String> {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let size: u32 = opt("--size").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let out_dir = std::path::Path::new("assets/textures");
    std::fs::create_dir_all(out_dir).expect("create assets/textures");

    if let Some(albedo) = opt("--albedo") {
        let mut img = open(&albedo);
        match opt("--alpha") {
            Some(alpha) => apply_alpha(&mut img, &open(&alpha)),
            None => {
                if img.pixels().all(|p| p.0[3] == 255) {
                    println!("  note: {albedo} has no alpha and no --alpha map given");
                }
            }
        }
        // Core owns the bleed, so the conditioning here and the cluster generation
        // the renderer does at load cannot drift apart.
        let mut bitmap = to_bitmap(&img);
        arbor_core::cluster::bleed_color_outward(&mut bitmap);
        let img = from_bitmap(&bitmap);
        let out = imageops::resize(&img, size, size, imageops::FilterType::Triangle);
        save(&out, out_dir, &name, "albedo");
    }
    for (key, suffix) in [("--normal", "normal"), ("--rough", "roughness")] {
        if let Some(path) = opt(key) {
            let img = open(&path);
            let out = imageops::resize(&img, size, size, imageops::FilterType::Triangle);
            save(&out, out_dir, &name, suffix);
        }
    }
    if let Some(path) = opt("--spec") {
        let mut img = open(&path);
        for p in img.pixels_mut() {
            let rough = 255 - p.0[0];
            p.0 = [rough, rough, rough, 255];
        }
        let out = imageops::resize(&img, size, size, imageops::FilterType::Triangle);
        save(&out, out_dir, &name, "roughness");
    }
}

fn apply_alpha(img: &mut RgbaImage, mask: &RgbaImage) {
    let (w, h) = img.dimensions();
    let (mw, mh) = mask.dimensions();
    for y in 0..h {
        for x in 0..w {
            // The mask is often authored at a different resolution than the albedo.
            let mx = (x as u64 * mw as u64 / w as u64).min(mw as u64 - 1) as u32;
            let my = (y as u64 * mh as u64 / h as u64).min(mh as u64 - 1) as u32;
            img.get_pixel_mut(x, y).0[3] = mask.get_pixel(mx, my).0[0];
        }
    }
}
