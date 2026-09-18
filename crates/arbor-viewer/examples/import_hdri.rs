//! Brings a photographed environment into the repository, shrunk to a size the viewer
//! and the web page can afford, and reports the sun the viewer will find in it.
//!
//! cargo run --release -p arbor-viewer --example import_hdri -- \
//!     <source.hdr> [--name <key>] [--width 2048] [--preview out.png]
//!
//! Writes `assets/hdri/<key>.hdr`, the key defaulting to the source's file name less
//! any `_4k`-style suffix. `--preview` writes a small tonemapped picture of the result
//! with the sun it found marked, which is the quickest way to check that the sun came
//! from the sun and not from a bright window.

// Shared with the viewer, which uses parts of them this does not.
#[path = "../src/hdri.rs"]
#[allow(dead_code)]
mod hdri;
#[path = "../src/lighting.rs"]
#[allow(dead_code)]
mod lighting;

use glam::Vec3;
use hdri::{Equirect, HDRI_DIR};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |key: &str| args.iter().position(|a| a == key).and_then(|i| args.get(i + 1));
    let Some(source) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: import_hdri <source.hdr> [--name <key>] [--width 2048] [--preview out.png]");
        std::process::exit(2);
    };
    let width: usize = flag("--width").and_then(|s| s.parse().ok()).unwrap_or(2048);
    let name = flag("--name").cloned().unwrap_or_else(|| {
        let stem = std::path::Path::new(source).file_stem().unwrap().to_string_lossy().to_string();
        match stem.rsplit_once('_') {
            Some((head, tail)) if tail.ends_with('k') && tail[..tail.len() - 1].parse::<u32>().is_ok() => {
                head.to_string()
            }
            _ => stem,
        }
    });

    let bytes = std::fs::read(source).unwrap_or_else(|e| panic!("{source}: {e}"));
    let full = Equirect::decode(&bytes).unwrap_or_else(|e| panic!("{source}: {e}"));
    let img = full.shrink_to(width);
    println!("{source}: {}x{} -> {}x{}", full.width, full.height, img.width, img.height);

    std::fs::create_dir_all(HDRI_DIR).expect("create the hdri folder");
    let out = format!("{HDRI_DIR}/{name}.hdr");
    std::fs::write(&out, img.encode().expect("encode")).expect("write");
    println!("wrote {out} ({:.1} MB)", std::fs::metadata(&out).unwrap().len() as f64 / 1e6);

    let analysed = img.analyse();
    let mean = lighting::luminance(analysed.image.mean());
    match analysed.sun {
        Some(sun) => {
            let elevation = sun.dir.y.asin().to_degrees();
            println!(
                "sun: elevation {elevation:.1} deg, irradiance {:.2} (lum {:.2}), radius {:.2} deg; sky mean {mean:.3}, sun/sky {:.1}",
                sun.irradiance,
                lighting::luminance(sun.irradiance),
                sun.angular_radius.to_degrees(),
                lighting::luminance(sun.irradiance) / (mean * std::f32::consts::PI).max(1e-6),
            );
        }
        None => println!("no sun stands out; sky mean {mean:.3}"),
    }

    if let Some(path) = flag("--preview") {
        let small = analysed.image.shrink_to(1024);
        let exposure = 0.18 / mean.max(1e-6);
        let mut png = image::RgbImage::new(small.width as u32, small.height as u32);
        for y in 0..small.height {
            for x in 0..small.width {
                let c = small.pixels[y * small.width + x] * exposure;
                let c = c / (Vec3::ONE + c);
                let g = |v: f32| (v.max(0.0).powf(1.0 / 2.2) * 255.0) as u8;
                png.put_pixel(x as u32, y as u32, image::Rgb([g(c.x), g(c.y), g(c.z)]));
            }
        }
        if let Some(sun) = analysed.sun {
            let (u, v) = hdri::uv_from_dir(sun.dir);
            let (cx, cy) = ((u * small.width as f32) as i32, (v * small.height as f32) as i32);
            for d in -12..=12 {
                for (x, y) in [(cx + d, cy), (cx, cy + d)] {
                    if x >= 0 && y >= 0 && (x as usize) < small.width && (y as usize) < small.height {
                        png.put_pixel(x as u32, y as u32, image::Rgb([255, 0, 255]));
                    }
                }
            }
        }
        png.save(path).expect("write preview");
        println!("wrote {path}");
    }
}
