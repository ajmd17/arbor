//! Repacks an irregular photo atlas into the regular grid the renderer samples.
//!
//! cargo run --release -p arbor-viewer --example repack_atlas -- \
//!     <in-name> <out-name> [--rot 0,0,0] [--cell 640x1088] [--keep 3] [--pick 2,3]
//!
//! Scanned foliage atlases come with their sprays dropped wherever they fitted on the
//! sheet, at whatever angle they were photographed. The renderer indexes cells as a
//! `cols x rows` grid, so the sprays have to be cut out and stacked one per row first.
//! Each is found by its alpha, cropped to what it actually covers, turned upright, and
//! fitted to a cell without stretching, so the shape of a spray survives the move.
//!
//! `--pick` keeps only the sprays at those positions in the list printed (sheet order),
//! for a species that wants some of a sheet's sprays and not others. `--rot` then lines
//! up with the picked ones.

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
/// Alpha above this still belongs to whichever spray it touches: the soft needle edges
/// and the tips too faint to count as foliage on their own.
const FAINT: u8 = 6;
/// A run of foliage smaller than this share of the biggest is a speck, not a spray.
const MIN_SHARE: f32 = 0.02;

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

    let (all_boxes, labels) = components(&albedo, keep);
    // Each kept spray remembers the label it was found under, so picking a subset does
    // not change which pixels belong to which.
    let picked: Vec<usize> = match opt("--pick") {
        Some(list) => list
            .split(',')
            .map(|v| v.trim().parse().expect("--pick wants indices"))
            .collect(),
        None => (0..all_boxes.len()).collect(),
    };
    println!("found {} sprays in {src}:", all_boxes.len());
    for (i, b) in all_boxes.iter().enumerate() {
        let slot = picked.iter().position(|&p| p == i);
        println!(
            "  {i}: {}x{} at ({},{})  aspect {:.2}  {}",
            b.2 - b.0,
            b.3 - b.1,
            b.0,
            b.1,
            (b.2 - b.0) as f32 / (b.3 - b.1) as f32,
            match slot {
                Some(k) => format!("rot {}", rots.get(k).copied().unwrap_or(0)),
                None => "not picked".to_string(),
            }
        );
    }
    let boxes: Vec<(u32, (u32, u32, u32, u32))> =
        picked.iter().map(|&i| (i as u32 + 1, all_boxes[i])).collect();

    let pack = |img: &RgbaImage, name: &str, alpha_from: Option<&RgbaImage>| {
        let mut sheet = RgbaImage::new(cw, ch * boxes.len() as u32);
        for (i, &(label, b)) in boxes.iter().enumerate() {
            let mut cell = imageops::crop_imm(img, b.0, b.1, b.2 - b.0, b.3 - b.1).to_image();
            // Maps other than the albedo have no alpha of their own, so the cut has to
            // come from the albedo's or they arrive as opaque rectangles.
            if let Some(a) = alpha_from {
                let av = imageops::crop_imm(a, b.0, b.1, b.2 - b.0, b.3 - b.1).to_image();
                for (p, q) in cell.pixels_mut().zip(av.pixels()) {
                    p.0[3] = q.0[3];
                }
            }
            // A box is a rectangle and sprays on a crowded sheet reach into each other's,
            // so anything in it that belongs to another spray is cut away.
            let sheet_w = img.width();
            for (x, y, p) in cell.enumerate_pixels_mut() {
                let at = ((b.1 + y) * sheet_w + b.0 + x) as usize;
                if labels[at] != label {
                    p.0[3] = 0;
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

/// The largest connected runs of opaque pixels, in the order they sit on the sheet, with
/// a label per pixel saying which of them it belongs to (0 for none, else index + 1).
///
/// A spray is found by its solid core and then grown out through everything faint that
/// touches it, so its soft edges and loose needle tips come with it; whatever is left
/// over belongs to no spray and is dropped. Runs under `MIN_SHARE` of the biggest are
/// specks and are not kept however few sprays the sheet holds.
fn components(img: &RgbaImage, keep: usize) -> (Vec<(u32, u32, u32, u32)>, Vec<u32>) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let alpha: Vec<u8> = img.pixels().map(|p| p.0[3]).collect();
    let mut comp = vec![0u32; w * h];
    let mut found: Vec<(usize, u32, (usize, usize))> = Vec::new();
    let neighbours = |i: usize| {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        (-1i64..=1).flat_map(move |dy| (-1i64..=1).map(move |dx| (x + dx, y + dy))).filter_map(
            move |(nx, ny)| {
                (nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64)
                    .then(|| ny as usize * w + nx as usize)
            },
        )
    };
    let mut next = 0u32;
    for start in 0..w * h {
        if alpha[start] <= SOLID || comp[start] != 0 {
            continue;
        }
        next += 1;
        let mut area = 0usize;
        let mut stack = vec![start];
        comp[start] = next;
        while let Some(i) = stack.pop() {
            area += 1;
            for j in neighbours(i) {
                if alpha[j] > SOLID && comp[j] == 0 {
                    comp[j] = next;
                    stack.push(j);
                }
            }
        }
        found.push((area, next, (start % w, start / w)));
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    let biggest = found.first().map_or(0, |f| f.0);
    found.retain(|f| f.0 as f32 >= biggest as f32 * MIN_SHARE);
    found.truncate(keep);
    // Back into the order they sat on the sheet, so `--rot` is easy to line up with what
    // the eye sees rather than with an area ranking.
    found.sort_by_key(|&(_, _, (x, y))| (y, x));

    // Relabel the kept cores 1..=n and grow them out through the faint pixels together,
    // a ring at a time, so a pixel between two sprays goes to the nearer one.
    let mut label = vec![0u32; w * h];
    let mut frontier = Vec::new();
    for (n, &(_, id, _)) in found.iter().enumerate() {
        for i in 0..w * h {
            if comp[i] == id {
                label[i] = n as u32 + 1;
                frontier.push(i);
            }
        }
    }
    while !frontier.is_empty() {
        let mut ring = Vec::new();
        for &i in &frontier {
            for j in neighbours(i) {
                if label[j] == 0 && alpha[j] > FAINT {
                    label[j] = label[i];
                    ring.push(j);
                }
            }
        }
        frontier = ring;
    }

    let mut boxes = vec![(u32::MAX, u32::MAX, 0u32, 0u32); found.len()];
    for (i, &l) in label.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let b = &mut boxes[l as usize - 1];
        let (x, y) = ((i % w) as u32, (i / w) as u32);
        *b = (b.0.min(x), b.1.min(y), b.2.max(x + 1), b.3.max(y + 1));
    }
    (boxes, label)
}
