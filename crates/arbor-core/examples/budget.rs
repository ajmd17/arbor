//! Where the triangles in a tree actually go.
//!
//! cargo run --release -p arbor-core --example budget -- [pine|oak]

use std::collections::HashMap;

use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_leaves, build_mesh, grow, stem_costs};

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "pine".into());
    let src = builtin_presets()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s.to_string())
        .unwrap();
    let params = parse_species(&src).unwrap();
    let sk = grow(&params);
    let mesh = build_mesh(&sk, &params);
    let leaves = build_leaves(&sk, &params);
    let costs = stem_costs(&sk, &params);

    let (lo, hi) = mesh.aabb();
    println!("=== {} ===", params.name);
    println!(
        "height {:.1} m   spread {:.1} x {:.1} m   trunk radius {:.2} m",
        hi[1] - lo[1],
        hi[0] - lo[0],
        hi[2] - lo[2],
        params.trunk.radius
    );
    println!(
        "nodes {}   stems {}   bark tris {}   bark verts {}   leaf tris {}",
        sk.nodes.len(),
        costs.len(),
        mesh.triangle_count(),
        mesh.vertex_count(),
        leaves.triangle_count()
    );
    let total: usize = costs.iter().map(|c| c.triangles).sum();
    println!("  stems account for {total} of {} bark tris", mesh.triangle_count());

    // By level: where in the tree the cost sits.
    let mut by_level: HashMap<u8, (usize, usize, usize, usize)> = HashMap::new();
    for c in &costs {
        let e = by_level.entry(c.level).or_default();
        e.0 += 1;
        e.1 += c.triangles;
        e.2 += c.vertices;
        e.3 += c.rings;
    }
    let mut levels: Vec<_> = by_level.into_iter().collect();
    levels.sort_by_key(|(l, _)| *l);
    println!("\nlevel   stems     tris    (%)     verts    rings  tris/stem");
    for (level, (stems, tris, verts, rings)) in &levels {
        println!(
            "  {level}  {stems:>6} {tris:>8}  {:>5.1}% {verts:>8} {rings:>8} {:>8.1}",
            *tris as f32 / total as f32 * 100.0,
            *tris as f32 / *stems as f32
        );
    }

    // What each level actually came out at, which is what decides whether a tree
    // reads as massive or spindly.
    let mut rad: HashMap<u8, (f32, f32, usize)> = HashMap::new();
    for run in sk.stem_runs() {
        let level = sk.nodes[run[0] as usize].level;
        let r = run
            .iter()
            .map(|&i| sk.nodes[i as usize].radius)
            .fold(0.0f32, f32::max);
        let e = rad.entry(level).or_insert((0.0, 0.0, 0));
        e.0 += r;
        e.1 = e.1.max(r);
        e.2 += 1;
    }
    let mut rl: Vec<_> = rad.into_iter().collect();
    rl.sort_by_key(|(l, _)| *l);
    println!("
level   mean radius   thickest");
    for (level, (sum, max, n)) in &rl {
        println!("  {level}      {:.3} m       {max:.3} m", sum / *n as f32);
    }

    // Spikes are already the cheap path; everything else is rings and end caps.
    let spikes: Vec<_> = costs.iter().filter(|c| c.spike).collect();
    let tubes: Vec<_> = costs.iter().filter(|c| !c.spike).collect();
    let sum = |v: &[&arbor_core::StemCost]| -> usize { v.iter().map(|c| c.triangles).sum() };
    println!(
        "\nsingle-segment spikes: {:>5} stems, {:>7} tris ({:.1}%)",
        spikes.len(),
        sum(&spikes),
        sum(&spikes) as f32 / total as f32 * 100.0
    );
    println!(
        "swept tubes:           {:>5} stems, {:>7} tris ({:.1}%)",
        tubes.len(),
        sum(&tubes),
        sum(&tubes) as f32 / total as f32 * 100.0
    );

    // A swept tube pays 2 triangles per side per segment, plus one end cone, plus a
    // ground cap on the trunk. Splitting that out says which part is worth attacking.
    // A spike is nothing but its cone, so counting those here would be counting the
    // same triangles twice.
    let cones: usize = tubes.iter().map(|c| c.radial as usize).sum();
    let rings_total: usize = tubes.iter().map(|c| c.rings).sum();
    println!(
        "  of which end cones:  {cones:>7} tris ({:.1}% of all bark)",
        cones as f32 / total as f32 * 100.0
    );
    println!("  rings swept:         {rings_total:>7}");

    let adaptive: usize = costs.iter().map(|c| c.adaptive_triangles).sum();
    println!(
        "
if every ring took its own side count: {adaptive} tris ({:+.1}%)",
        (adaptive as f32 / total as f32 - 1.0) * 100.0
    );

    // How many sides stems are actually swept at.
    let mut by_radial: HashMap<u32, (usize, usize)> = HashMap::new();
    for c in &costs {
        let e = by_radial.entry(c.radial).or_default();
        e.0 += 1;
        e.1 += c.triangles;
    }
    let mut radials: Vec<_> = by_radial.into_iter().collect();
    radials.sort_by_key(|(r, _)| *r);
    println!("\nsides   stems     tris");
    for (r, (stems, tris)) in &radials {
        println!("  {r:>3} {stems:>7} {tris:>8}");
    }

    println!(
        "\nshare of the whole tree: bark {:.0}%, leaves {:.0}%",
        mesh.triangle_count() as f32 / (mesh.triangle_count() + leaves.triangle_count()) as f32
            * 100.0,
        leaves.triangle_count() as f32 / (mesh.triangle_count() + leaves.triangle_count()) as f32
            * 100.0
    );
}
