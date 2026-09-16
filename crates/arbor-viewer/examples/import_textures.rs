//! Conditions source PBR maps into the trimmed set the viewer loads from
//! `assets/textures`. Run it again to re-import from a different source pack.
//!
//! cargo run --release -p arbor-viewer --example import_textures -- \
//!     <out-name> --albedo a.png [--alpha m.png] [--normal n.png] [--rough r.png] [--size 1024]
//!
//! The albedo gets its alpha from `--alpha` (red channel) and has its colour bled
//! outward into the transparent region. Without that bleed, both the downscale here
//! and the mip chain on the GPU average leaf colour against the black background and
//! every leaf picks up a dark halo.

use image::{imageops, Rgba, RgbaImage};

const BLEED_PASSES: u32 = 24;

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
        bleed_color_outward(&mut img);
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
}

fn open(path: &str) -> RgbaImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
        .to_rgba8()
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

/// Flood the colour of opaque pixels outward across the transparent background so
/// filtering never mixes leaf colour with whatever was behind it.
fn bleed_color_outward(img: &mut RgbaImage) {
    let (w, h) = img.dimensions();
    let mut known: Vec<bool> = img.pixels().map(|p| p.0[3] > 0).collect();
    if !known.iter().any(|&k| k) {
        return;
    }

    for _ in 0..BLEED_PASSES {
        let source = known.clone();
        let mut filled_any = false;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if source[i] {
                    continue;
                }
                let (mut sum, mut n) = ([0u32; 3], 0u32);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let ni = (ny as u32 * w + nx as u32) as usize;
                    if !source[ni] {
                        continue;
                    }
                    let p = img.get_pixel(nx as u32, ny as u32).0;
                    for k in 0..3 {
                        sum[k] += p[k] as u32;
                    }
                    n += 1;
                }
                if n == 0 {
                    continue;
                }
                let a = img.get_pixel(x, y).0[3];
                img.put_pixel(
                    x,
                    y,
                    Rgba([
                        (sum[0] / n) as u8,
                        (sum[1] / n) as u8,
                        (sum[2] / n) as u8,
                        a,
                    ]),
                );
                known[i] = true;
                filled_any = true;
            }
        }
        if !filled_any {
            break;
        }
    }

    // Anything the bleed never reached keeps filtering neutral by taking the mean.
    let (mut sum, mut n) = ([0u64; 3], 0u64);
    for (i, p) in img.pixels().enumerate() {
        if known[i] {
            for (k, total) in sum.iter_mut().enumerate() {
                *total += p.0[k] as u64;
            }
            n += 1;
        }
    }
    if n == 0 {
        return;
    }
    let mean = [
        (sum[0] / n) as u8,
        (sum[1] / n) as u8,
        (sum[2] / n) as u8,
    ];
    for y in 0..h {
        for x in 0..w {
            if !known[(y * w + x) as usize] {
                let a = img.get_pixel(x, y).0[3];
                img.put_pixel(x, y, Rgba([mean[0], mean[1], mean[2], a]));
            }
        }
    }
}

fn save(img: &RgbaImage, dir: &std::path::Path, name: &str, suffix: &str) {
    let path = dir.join(format!("{name}_{suffix}.png"));
    img.save(&path).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!("{} ({} KB)", path.display(), bytes / 1024);
}
