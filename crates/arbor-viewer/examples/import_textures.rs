//! Conditions source PBR maps into the trimmed set the viewer loads from
//! `assets/textures`. Run it again to re-import from a different source pack.
//!
//! cargo run --release -p arbor-viewer --example import_textures -- \
//!     <out-name> --albedo a.png [--alpha m.png] [--normal n.png]
//!         [--rough r.png | --spec s.png] [--translucency t.png] [--size 1024]
//!
//! cargo run --release -p arbor-viewer --example import_textures -- \
//!     <out-name> --rawcatalog <folder>[,<folder>...] [--size 1024]
//!
//! The albedo gets its alpha from `--alpha` (red channel) and has its colour bled
//! outward into the transparent region.
//!
//! `--spec` takes a specular-intensity map instead of a roughness one and inverts it,
//! since bright specular means a smooth surface. Art authored for a specular workflow
//! ships that map rather than the roughness the shader here wants.
//!
//! `--translucency` is the colour of the light a leaf lets through. It is cut out by the
//! same mask as the albedo, so the two filter alike, and levelled to the albedo's
//! brightness: a leaf passes on about as much light as it throws back, where a scanned
//! map is authored several times darker for an engine that multiplies it back up. What
//! is kept of it is its hue and where the veins fall. The lift stops at six times: a
//! map darker than that is of something that really is nearly opaque, a spruce needle
//! rather than a birch leaf, and lifting it further only brings up the scan's noise.
//!
//! `--rawcatalog` takes a Raw Catalog set as it unpacks, picking each map out of the
//! folder by its suffix. Several folders are stacked down one sheet in the order given,
//! which is the layout a cluster's `sources` reads: scans of five different leaves
//! become five alternatives for the bake to draw from.

use std::path::{Path, PathBuf};

use image::{imageops, RgbaImage};

#[path = "common/texture.rs"]
mod texture;
use texture::{from_bitmap, open, save, to_bitmap};

/// The source files of one set. Only the albedo is needed.
#[derive(Default)]
struct Sources {
    albedo: Option<String>,
    alpha: Option<String>,
    normal: Option<String>,
    rough: Option<String>,
    spec: Option<String>,
    translucency: Option<String>,
}

/// One set, conditioned and resized: the maps as they will be written.
#[derive(Default)]
struct Conditioned {
    albedo: Option<RgbaImage>,
    normal: Option<RgbaImage>,
    roughness: Option<RgbaImage>,
    translucency: Option<RgbaImage>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(name) = args.first().filter(|a| !a.starts_with("--")).cloned() else {
        eprintln!("usage: import_textures <out-name> --albedo a.png [--alpha m.png] [--normal n.png] [--rough r.png] [--translucency t.png] [--size N]");
        eprintln!("       import_textures <out-name> --rawcatalog <folder>[,<folder>...] [--size N]");
        std::process::exit(2);
    };
    let opt = |key: &str| -> Option<String> {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let size: u32 = opt("--size").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let out_dir = Path::new("assets/textures");
    std::fs::create_dir_all(out_dir).expect("create assets/textures");

    let sets: Vec<Sources> = match opt("--rawcatalog") {
        Some(folders) => folders.split(',').map(|f| raw_catalog(Path::new(f.trim()))).collect(),
        None => vec![Sources {
            albedo: opt("--albedo"),
            alpha: opt("--alpha"),
            normal: opt("--normal"),
            rough: opt("--rough"),
            spec: opt("--spec"),
            translucency: opt("--translucency"),
        }],
    };
    let sets: Vec<Conditioned> = sets.iter().map(|s| condition(s, size)).collect();

    let maps: [(&str, fn(&Conditioned) -> Option<&RgbaImage>); 4] = [
        ("albedo", |c| c.albedo.as_ref()),
        ("normal", |c| c.normal.as_ref()),
        ("roughness", |c| c.roughness.as_ref()),
        ("translucency", |c| c.translucency.as_ref()),
    ];
    for (suffix, pick) in maps {
        let found: Vec<&RgbaImage> = sets.iter().filter_map(pick).collect();
        if found.is_empty() {
            continue;
        }
        // A sheet with a row missing would put every later leaf in the wrong cell.
        assert!(found.len() == sets.len(), "only some of the sets have a {suffix} map");
        save(&stack(&found, size), out_dir, &name, suffix);
    }
}

/// The maps of a Raw Catalog set, found by what their names end in.
fn raw_catalog(folder: &Path) -> Sources {
    let files: Vec<PathBuf> = std::fs::read_dir(folder)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", folder.display()))
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    let find = |suffix: &str| -> Option<String> {
        files
            .iter()
            .find(|p| p.file_stem().is_some_and(|s| s.to_string_lossy().ends_with(suffix)))
            .map(|p| p.to_string_lossy().into_owned())
    };
    let albedo = find("_BaseColor");
    assert!(albedo.is_some(), "no *_BaseColor map in {}", folder.display());
    Sources {
        albedo,
        alpha: find("_Opacity"),
        // The older sets ship one normal map and the newer ones both conventions.
        normal: find("_NormalOpenGL").or_else(|| find("_Normal")),
        rough: find("_Roughness"),
        spec: None,
        translucency: find("_Translucency"),
    }
}

fn condition(src: &Sources, size: u32) -> Conditioned {
    let resize = |img: &RgbaImage| imageops::resize(img, size, size, imageops::FilterType::Triangle);
    let mask = src.alpha.as_deref().map(open);
    // Core owns the bleed, so the conditioning here and the cluster generation the
    // renderer does at load cannot drift apart.
    //
    // `solid_only` takes the colour to bleed from the leaf's interior alone. A scan's
    // albedo arrives with its colour already run out past the cutout; its translucency
    // map does not, and its edge pixels are part leaf and part white backing sheet, so
    // bleeding from them rings every backlit leaf in white.
    let cut_out = |path: &str, solid_only: bool| -> RgbaImage {
        let mut img = open(path);
        match &mask {
            Some(mask) => apply_alpha(&mut img, mask),
            None => {
                if img.pixels().all(|p| p.0[3] == 255) {
                    println!("  note: {path} has no alpha and no --alpha map given");
                }
            }
        }
        let mut bitmap = to_bitmap(&img);
        if solid_only {
            let alpha: Vec<u8> = bitmap.pixels.iter().skip(3).step_by(4).copied().collect();
            let (w, h) = (bitmap.width as usize, bitmap.height as usize);
            // The mask is cut a little outside what the map shows, so the interior is
            // what is still solid a few pixels in from every edge.
            let reach = (w / 1024).max(1);
            let inner = erode(&erode(&alpha, w, h, reach, true), w, h, reach, false);
            for (px, a) in bitmap.pixels.chunks_exact_mut(4).zip(inner) {
                if a < SOLID {
                    px[3] = 0;
                }
            }
            arbor_core::cluster::bleed_color_outward(&mut bitmap);
            for (px, a) in bitmap.pixels.chunks_exact_mut(4).zip(alpha) {
                px[3] = a;
            }
        } else {
            arbor_core::cluster::bleed_color_outward(&mut bitmap);
        }
        resize(&from_bitmap(&bitmap))
    };

    let mut out = Conditioned {
        albedo: src.albedo.as_deref().map(|path| cut_out(path, false)),
        translucency: src.translucency.as_deref().map(|path| cut_out(path, true)),
        ..Conditioned::default()
    };
    if let (Some(albedo), Some(through)) = (&out.albedo, &mut out.translucency) {
        level(through, albedo);
    }
    out.normal = src.normal.as_deref().map(|path| resize(&open(path)));
    out.roughness = src.rough.as_deref().map(|path| resize(&open(path)));
    if let Some(path) = &src.spec {
        let mut img = open(path);
        for p in img.pixels_mut() {
            let rough = 255 - p.0[0];
            p.0 = [rough, rough, rough, 255];
        }
        out.roughness = Some(resize(&img));
    }
    out
}

/// The least alpha within `reach` pixels of each pixel, along rows or down columns.
fn erode(alpha: &[u8], w: usize, h: usize, reach: usize, rows: bool) -> Vec<u8> {
    let mut out = vec![0u8; alpha.len()];
    for y in 0..h {
        for x in 0..w {
            let (at, len) = if rows { (x, w) } else { (y, h) };
            let (lo, hi) = (at.saturating_sub(reach), (at + reach).min(len - 1));
            out[y * w + x] = (lo..=hi)
                .map(|k| if rows { alpha[y * w + k] } else { alpha[k * w + x] })
                .min()
                .unwrap_or(0);
        }
    }
    out
}

/// Alpha from which a pixel is wholly leaf, with none of the backing sheet in it.
const SOLID: u8 = 250;

/// Most a translucency map is brightened toward its albedo.
const MAX_TRANSLUCENCY_LIFT: f32 = 6.0;

/// Scales `img` in linear light until it is as bright as `like` over the leaf.
fn level(img: &mut RgbaImage, like: &RgbaImage) {
    let linear = |c: u8| (c as f32 / 255.0).powf(2.2);
    let mean = |m: &RgbaImage| -> f32 {
        let (mut sum, mut n) = (0.0f64, 0u32);
        for p in m.pixels().filter(|p| p.0[3] > 200) {
            sum += (0.2126 * linear(p.0[0]) + 0.7152 * linear(p.0[1]) + 0.0722 * linear(p.0[2])) as f64;
            n += 1;
        }
        (sum / n.max(1) as f64) as f32
    };
    let (have, want) = (mean(img), mean(like));
    if have <= 1e-5 {
        return;
    }
    let gain = (want / have).min(MAX_TRANSLUCENCY_LIFT);
    println!("  translucency levelled by {gain:.2}");
    for p in img.pixels_mut() {
        for c in &mut p.0[..3] {
            *c = ((linear(*c) * gain).min(1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        }
    }
}

/// Sets one under the next down a single sheet.
fn stack(sets: &[&RgbaImage], size: u32) -> RgbaImage {
    let mut sheet = RgbaImage::new(size, size * sets.len() as u32);
    for (i, img) in sets.iter().enumerate() {
        imageops::replace(&mut sheet, *img, 0, (i as u32 * size) as i64);
    }
    sheet
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
