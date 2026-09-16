//! What each possible saving is actually worth, measured rather than guessed.
//!
//! cargo run --release -p arbor-core --example levers -- [pine|oak]

use arbor_core::species::{builtin_presets, parse_species, SpeciesParams};
use arbor_core::{build_leaves, build_mesh, grow};

fn totals(p: &SpeciesParams) -> (usize, usize, usize) {
    let sk = grow(p);
    let m = build_mesh(&sk, p);
    let l = build_leaves(&sk, p);
    (m.triangle_count(), l.triangle_count(), m.vertex_count() + l.vertex_count())
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "pine".into());
    let src = builtin_presets()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s.to_string())
        .unwrap();
    let base = parse_species(&src).unwrap();
    let (b0, l0, v0) = totals(&base);
    let t0 = b0 + l0;
    println!("=== {} baseline ===", base.name);
    println!("bark {b0}  leaf {l0}  total {t0}  verts {v0}\n");

    let row = |label: &str, p: &SpeciesParams| {
        let (b, l, v) = totals(p);
        let t = b + l;
        println!(
            "{label:<38} bark {b:>7} leaf {l:>7} total {t:>7} ({:>5.1}% of base)  verts {v:>7}",
            t as f32 / t0 as f32 * 100.0
        );
    };

    for r in [5u32, 4, 3] {
        let mut p = base.clone();
        p.mesh.min_radial = r;
        row(&format!("min_radial {r} (from {})", base.mesh.min_radial), &p);
    }
    for k in [0.75f32, 0.5] {
        let mut p = base.clone();
        p.mesh.radial_per_meter = base.mesh.radial_per_meter * k;
        row(&format!("radial_per_meter x{k}"), &p);
    }

    // Fewer, bigger cards: card area held constant so the canopy keeps its coverage.
    for d in [2.0f32, 3.0, 4.0] {
        let mut p = base.clone();
        p.leaves.density = base.leaves.density / d;
        p.leaves.card_length = base.leaves.card_length * d.sqrt();
        p.leaves.card_width = base.leaves.card_width * d.sqrt();
        row(&format!("leaf density /{d}, card x{:.2} (equal area)", d.sqrt()), &p);
    }
    for c in [3u32, 2, 1] {
        let mut p = base.clone();
        p.leaves.cluster_size = c;
        let k = (base.leaves.cluster_size as f32 / c as f32).sqrt();
        p.leaves.card_length = base.leaves.card_length * k;
        p.leaves.card_width = base.leaves.card_width * k;
        row(&format!("cluster_size {c} (from {}), equal area", base.leaves.cluster_size), &p);
    }
    for lv in [3u8, 4] {
        let mut p = base.clone();
        p.leaves.min_level = lv;
        row(&format!("leaf min_level {lv} (from {})", base.leaves.min_level), &p);
    }

    // Everything cheap at once.
    let mut p = base.clone();
    p.mesh.min_radial = 4;
    p.leaves.density = base.leaves.density / 3.0;
    p.leaves.card_length = base.leaves.card_length * 3.0f32.sqrt();
    p.leaves.card_width = base.leaves.card_width * 3.0f32.sqrt();
    row("combined: min_radial 4 + density /3", &p);

    // Card area against the crown, to see how far over real foliage we are.
    let sk = grow(&base);
    let leaves = build_leaves(&sk, &base);
    let area = leaves.leaf_count() as f32 * base.leaves.card_length * base.leaves.card_width;
    let (mn, mx) = build_mesh(&sk, &base).aabb();
    let footprint = std::f32::consts::PI * (((mx[0] - mn[0]) + (mx[2] - mn[2])) * 0.25).powi(2);
    println!(
        "\ncards {}  card area {:.0} m2  crown footprint {:.0} m2  -> leaf area index {:.1}",
        leaves.leaf_count(),
        area,
        footprint,
        area / footprint
    );
}
