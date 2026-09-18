//! Moves the colour of one band of hues in an albedo, leaving everything else alone.
//!
//! cargo run --release -p arbor-viewer --example recolor -- \
//!     <name> --band 45,150 --hue 90 [--sat 0.4] [--val 1.3] [--feather 12]
//!
//! A scanned foliage set is the colour of the plant that was scanned, and that is not
//! always the species it is being used for: the pine atlas is an olive, yellow-green
//! Scots pine where a glaucous blue-green one is wanted. A multiplier cannot make that
//! move — the blue channel of an olive needle is nearly empty, so scaling it up far
//! enough to matter turns every brown twig in the same texture purple. This rotates
//! the hue of pixels inside `--band` (degrees) by `--hue`, scales their saturation and
//! value, and feathers the edge of the band so nothing on the boundary is cut in two.
//! Twigs, bud scales and dry needles sit below the band and come through untouched.

use image::Rgba;

#[path = "common/texture.rs"]
mod texture;

const DIR: &str = "assets/textures";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(name) = args.first().filter(|a| !a.starts_with("--")).cloned() else {
        eprintln!("usage: recolor <name> --band lo,hi --hue deg [--sat k] [--val k] [--feather deg]");
        std::process::exit(2);
    };
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let num = |k: &str, d: f32| opt(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (lo, hi) = {
        let b = opt("--band").expect("--band lo,hi is required");
        let (a, c) = b.split_once(',').expect("--band wants lo,hi");
        (a.trim().parse::<f32>().unwrap(), c.trim().parse::<f32>().unwrap())
    };
    let shift = num("--hue", 0.0);
    let sat_k = num("--sat", 1.0);
    let val_k = num("--val", 1.0);
    let feather = num("--feather", 12.0).max(0.1);

    let path = format!("{DIR}/{name}_albedo.png");
    let mut img = texture::open(&path);
    let mut moved = 0usize;
    for p in img.pixels_mut() {
        let [r, g, b, a] = p.0;
        let (h, s, v) = rgb_to_hsv(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
        // How far inside the band this hue sits, ramping over `feather` at each edge. A
        // grey has no hue worth moving, so saturation gates it as well.
        let inside = ((h - lo) / feather).min((hi - h) / feather).clamp(0.0, 1.0);
        let w = inside * (s / 0.12).clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        moved += 1;
        let h2 = (h + shift * w).rem_euclid(360.0);
        let s2 = (s * (1.0 + (sat_k - 1.0) * w)).clamp(0.0, 1.0);
        let v2 = (v * (1.0 + (val_k - 1.0) * w)).clamp(0.0, 1.0);
        let (r2, g2, b2) = hsv_to_rgb(h2, s2, v2);
        *p = Rgba([
            (r2 * 255.0).round() as u8,
            (g2 * 255.0).round() as u8,
            (b2 * 255.0).round() as u8,
            a,
        ]);
    }
    texture::save(&img, std::path::Path::new(DIR), &name, "albedo");
    println!(
        "recoloured {moved} of {} pixels in {path}",
        img.width() * img.height()
    );
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 1e-6 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    let s = if max <= 1e-6 { 0.0 } else { d / max };
    (h, s, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}
