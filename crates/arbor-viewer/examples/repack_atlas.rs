//! Repacks an irregular photo atlas into the regular grid the renderer samples.
//!
//! cargo run --release -p arbor-viewer --example repack_atlas -- \
//!     <in-name> <out-name> [--rot 0,0,0] [--cell 640x1088] [--keep 3]
//!
//! Scanned foliage atlases come with their sprays dropped wherever they fitted on the
//! sheet, at whatever angle they were photographed. The renderer indexes cells as a
//! `cols x rows` grid, so the sprays have to be cut out and stacked one per row first.
//! Each is found by its alpha, cropped to what it actually covers, turned upright, and
//! fitted to a cell without stretching, so the shape of a spray survives the move.

use image::{imageops, Rgba, RgbaImage};

#[path = "../src/mipmap.rs"]
mod mipmap;
#[path = "common/texture.rs"]
mod texture;
use texture::{open, save};

const DIR: &str = "assets/textures";
/// Alpha above this counts as foliage. The opacity map arrives as a JPEG, so its edges
/// carry ringing that a cutoff at 1 would pick up as stray specks.
const SOLID: u8 = 40;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 2 {
        eprintln!("usage: repack_atlas <in-name> <out-name> [--rot a,b,c] [--cell WxH] [--keep N]");
        std::process::exit(2);
    }
    let (src, out) = (pos[0].clone(), pos[1].clone());
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let keep: usize = opt("--keep").and_then(|s| s.parse().ok()).unwrap_or(3);
    let (cw, ch) = {
        let c = opt("--cell").unwrap_or_else(|| "640x1088".into());
        let (a, b) = c.split_once('x').expect("--cell wants WxH");
        (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())
    };
    let rots: Vec<u32> = opt("--rot")
        .map(|s| s.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![0; keep]);

    let albedo = open(&format!("{DIR}/{src}_albedo.png"));
    let normal = std::path::Path::new(&format!("{DIR}/{src}_normal.png"))
        .exists()
        .then(|| open(&format!("{DIR}/{src}_normal.png")));
    let rough = std::path::Path::new(&format!("{DIR}/{src}_roughness.png"))
        .exists()
        .then(|| open(&format!("{DIR}/{src}_roughness.png")));

    let boxes = components(&albedo, keep);
    println!("found {} sprays in {src}:", boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        println!(
            "  {i}: {}x{} at ({},{})  aspect {:.2}  rot {}",
            b.2 - b.0,
            b.3 - b.1,
            b.0,
            b.1,
            (b.2 - b.0) as f32 / (b.3 - b.1) as f32,
            rots.get(i).copied().unwrap_or(0)
        );
    }

    let pack = |img: &RgbaImage, name: &str, alpha_from: Option<&RgbaImage>| {
        let mut sheet = RgbaImage::new(cw, ch * boxes.len() as u32);
        for (i, b) in boxes.iter().enumerate() {
            let mut cell = imageops::crop_imm(img, b.0, b.1, b.2 - b.0, b.3 - b.1).to_image();
            // Maps other than the albedo have no alpha of their own, so the cut has to
            // come from the albedo's or they arrive as opaque rectangles.
            if let Some(a) = alpha_from {
                let av = imageops::crop_imm(a, b.0, b.1, b.2 - b.0, b.3 - b.1).to_image();
                for (p, q) in cell.pixels_mut().zip(av.pixels()) {
                    p.0[3] = q.0[3];
                }
            }
            for _ in 0..(rots.get(i).copied().unwrap_or(0) / 90) % 4 {
                cell = imageops::rotate90(&cell);
            }
            // Fit without stretching: a spray that is squeezed to a cell's aspect stops
            // looking like the thing that was photographed.
            let (w, h) = (cell.width() as f32, cell.height() as f32);
            let k = (cw as f32 / w).min(ch as f32 / h);
            let (nw, nh) = ((w * k) as u32, (h * k) as u32);
            let fitted = imageops::resize(&cell, nw.max(1), nh.max(1), imageops::Lanczos3);
            let mut slot = RgbaImage::from_pixel(cw, ch, Rgba([0, 0, 0, 0]));
            imageops::overlay(
                &mut slot,
                &fitted,
                ((cw - nw.min(cw)) / 2) as i64,
                ((ch - nh.min(ch)) / 2) as i64,
            );
            imageops::overlay(&mut sheet, &slot, 0, (ch * i as u32) as i64);
        }
        // Colour has to run under the cutout edge or the mips pull background through it.
        let mut bm = texture::to_bitmap(&sheet);
        arbor_core::cluster::bleed_color_outward(&mut bm);
        save(&texture::from_bitmap(&bm), std::path::Path::new(DIR), &out, name);
    };

    pack(&albedo, "albedo", None);
    if let Some(n) = &normal {
        pack(n, "normal", Some(&albedo));
    }
    if let Some(r) = &rough {
        pack(r, "roughness", Some(&albedo));
    }
    println!("wrote {out} as 1 x {} cells of {cw}x{ch}", boxes.len());
}

/// Bounding boxes of the largest connected runs of opaque pixels, biggest first.
fn components(img: &RgbaImage, keep: usize) -> Vec<(u32, u32, u32, u32)> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let solid: Vec<bool> = img.pixels().map(|p| p.0[3] > SOLID).collect();
    let mut seen = vec![false; w * h];
    let mut found: Vec<(usize, (u32, u32, u32, u32))> = Vec::new();
    for start in 0..w * h {
        if !solid[start] || seen[start] {
            continue;
        }
        let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (w, h, 0usize, 0usize);
        let mut area = 0usize;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            area += 1;
            lo_x = lo_x.min(x);
            lo_y = lo_y.min(y);
            hi_x = hi_x.max(x);
            hi_y = hi_y.max(y);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if solid[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        found.push((area, (lo_x as u32, lo_y as u32, hi_x as u32 + 1, hi_y as u32 + 1)));
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.truncate(keep);
    // Back into the order they sat on the sheet, so `--rot` is easy to line up with what
    // the eye sees rather than with an area ranking.
    found.sort_by_key(|(_, b)| (b.1, b.0));
    found.into_iter().map(|(_, b)| b).collect()
}
