//! Writes out a leaf-cluster atlas as PNGs, for looking at one without launching the
//! viewer or tuning a species by trial and error.
//!
//! cargo run --release -p arbor-viewer --example bake_cluster -- leaf_pine [options]
//!
//! The viewer does not need this. A species that sets a `cluster` block grows its own
//! atlas from the single-leaf art every time it loads, and this tool runs exactly the
//! same code in arbor-core so what it writes is what the renderer will sample. The
//! numbers it prints are the ones a species needs:
//!
//!   - `coverage` is the share of a card that shows foliage, which sets how many cards
//!     a canopy needs to look full.
//!   - `leaf of cell` is what `card_length` has to be divided by to keep leaves at the
//!     real-world size they had before clustering.
//!
//! `--aspect` is the width against the length of the card the source texture was drawn
//! on, and matches `source_aspect` in the species. Leaves are rotated in pixels, so a
//! cell whose pixels are not square in world terms has to be stretched first or every
//! rotated leaf comes out sheared.

use arbor_core::cluster::{bake_cluster, LeafMaps};
use arbor_core::species::LeafClusterParams;

#[path = "common/texture.rs"]
mod texture;
use texture::{from_bitmap, open, save, to_bitmap};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(src) = args.first().filter(|a| !a.starts_with("--")).cloned() else {
        eprintln!(
            "usage: bake_cluster <src-name> [--out name] [--count N] [--cols N] [--rows N] [--aspect W/L] ..."
        );
        std::process::exit(2);
    };
    let opt = |k: &str| -> Option<String> {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let num = |k: &str, d: f32| opt(k).and_then(|s| s.parse().ok()).unwrap_or(d);

    let dir = std::path::Path::new("assets/textures");
    let out = opt("--out").unwrap_or_else(|| format!("{src}_cluster"));
    let cols = num("--cols", 1.0).max(1.0) as u32;
    let rows = num("--rows", 1.0).max(1.0) as u32;
    let d = LeafClusterParams::default();
    let params = LeafClusterParams {
        count: num("--count", d.count as f32) as u32,
        leaf_length: num("--leaf-length", d.leaf_length),
        base_angle_deg: num("--base-deg", d.base_angle_deg),
        tip_angle_deg: num("--tip-deg", d.tip_angle_deg),
        tip_scale: num("--tip-scale", d.tip_scale),
        leaf_narrow: num("--narrow", d.leaf_narrow),
        angle_variance_deg: num("--jitter-deg", d.angle_variance_deg),
        roll_deg: num("--roll-deg", d.roll_deg),
        roll_variance_deg: num("--roll-jitter-deg", d.roll_variance_deg),
        depth: num("--depth", d.depth),
        blade_twist_deg: num("--twist-deg", d.blade_twist_deg),
        depth_shade: num("--depth-shade", d.depth_shade),
        size_variance: num("--size-variance", d.size_variance),
        spacing_variance: num("--spacing-variance", d.spacing_variance),
        shoot_curve: num("--curve", d.shoot_curve),
        shoots: num("--shoots", d.shoots as f32) as u32,
        variants: num("--variants", d.variants as f32) as u32,
        shoot_base: num("--rachis-base", d.shoot_base),
        shoot_tip: num("--rachis-tip", d.shoot_tip),
        source_aspect: num("--aspect", d.source_aspect),
        sources: num("--sources", d.sources as f32) as u32,
        cell_size: num("--size", d.cell_size as f32) as u32,
        seed: num("--seed", d.seed as f32) as u64,
    };

    let albedo = to_bitmap(&open(&dir.join(format!("{src}_albedo.png")).to_string_lossy()));
    let normal = try_open(dir, &src, "normal");
    let rough = try_open(dir, &src, "roughness");

    let baked = bake_cluster(
        &params,
        cols,
        rows,
        LeafMaps {
            albedo: &albedo,
            normal: normal.as_ref(),
            roughness: rough.as_ref(),
        },
    );

    std::fs::create_dir_all(dir).expect("create assets/textures");
    save(&from_bitmap(&baked.albedo), dir, &out, "albedo");
    if let Some(img) = baked.normal.as_ref() {
        save(&from_bitmap(img), dir, &out, "normal");
    }
    if let Some(img) = baked.roughness.as_ref() {
        save(&from_bitmap(img), dir, &out, "roughness");
    }

    let leaf_of_cell = albedo
        .alpha_bounds(8)
        .map(|(_, y0, _, y1)| (y1 - y0 + 1) as f32 / (albedo.height / rows.max(1)) as f32)
        .unwrap_or(0.0);
    println!(
        "\ncluster of {}: coverage {:.3}   source leaf was {:.3} of its cell",
        params.count,
        baked.albedo.mean_alpha(),
        leaf_of_cell
    );
}

fn try_open(
    dir: &std::path::Path,
    src: &str,
    suffix: &str,
) -> Option<arbor_core::cluster::Bitmap> {
    let path = dir.join(format!("{src}_{suffix}.png"));
    path.exists()
        .then(|| to_bitmap(&open(&path.to_string_lossy())))
}
