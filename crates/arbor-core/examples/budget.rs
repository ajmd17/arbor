//! Where the triangles in a tree actually go.
//!
//! cargo run --release -p arbor-core --example budget -- [pine|oak]

use std::collections::HashMap;

use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_leaves, build_mesh, grow};
use glam::Vec3;

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

    let runs = sk.stem_runs();
    println!("=== {} ===", params.name);
    println!(
        "nodes {}   stems {}   bark tris {}   leaf tris {}",
        sk.nodes.len(),
        runs.len(),
        mesh.triangle_count(),
        leaves.triangle_count()
    );

    // Stems by level, and how many nodes each carries.
    let mut per_level: HashMap<u8, (usize, usize)> = HashMap::new();
    for run in &runs {
        let level = sk.nodes[run[0] as usize].level;
        let e = per_level.entry(level).or_default();
        e.0 += 1;
        e.1 += run.len();
    }
    let mut levels: Vec<_> = per_level.into_iter().collect();
    levels.sort_by_key(|(l, _)| *l);
    println!("\nstems and nodes by level");
    for (level, (stems, nodes)) in &levels {
        // Each node adds one ring; each stem also pays for its anchor ring and tip.
        println!("  level {level}: {stems:>6} stems, {nodes:>6} nodes");
    }

    // A stem with one node is a stub or a twig cut short: it still pays for a whole
    // tube, an anchor ring and a cone.
    let short = runs.iter().filter(|r| r.len() == 1).count();
    let two = runs.iter().filter(|r| r.len() == 2).count();
    println!("\nstems of 1 node: {short}   of 2 nodes: {two}   (of {} total)", runs.len());

    // Triangle sizes. Anything far below a pixel at normal viewing distance is
    // costing a draw and contributing nothing.
    let mut areas: Vec<f32> = Vec::with_capacity(mesh.triangle_count());
    let mut degenerate = 0usize;
    for tri in mesh.indices.chunks_exact(3) {
        let p: Vec<Vec3> = tri
            .iter()
            .map(|&i| Vec3::from(mesh.positions[i as usize]))
            .collect();
        let a = (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5;
        if a < 1e-9 {
            degenerate += 1;
        }
        areas.push(a);
    }
    areas.sort_by(f32::total_cmp);
    let total: f32 = areas.iter().sum();
    let pct = |q: f32| areas[((areas.len() as f32 * q) as usize).min(areas.len() - 1)];
    println!("\nbark triangle area (m2)");
    println!(
        "  median {:.6}   p90 {:.6}   max {:.4}   total {:.1}",
        pct(0.5),
        pct(0.9),
        areas.last().unwrap(),
        total
    );
    println!("  degenerate (zero area): {degenerate}");

    // How much of the surface the smallest triangles account for: if half the
    // triangles carry a few percent of the area, they are detail nobody can see.
    for q in [0.25f32, 0.5, 0.75] {
        let cut = (areas.len() as f32 * q) as usize;
        let share: f32 = areas[..cut].iter().sum::<f32>() / total * 100.0;
        println!(
            "  smallest {:>3.0}% of triangles carry {share:>5.2}% of the surface",
            q * 100.0
        );
    }

    // Attribute triangles to the stem that emitted them, by replaying what the
    // mesher does: one ring per node plus an anchor ring, swept at `radial` sides,
    // and a cone to close the end.
    let mp = &params.mesh;
    let mut tris_by_level: HashMap<u8, usize> = HashMap::new();
    let mut tris_single = 0usize;
    let mut tris_cones = 0usize;
    let mut fully_inside = 0usize;
    for run in &runs {
        let first = &sk.nodes[run[0] as usize];
        let r_max = run
            .iter()
            .map(|&i| sk.nodes[i as usize].radius)
            .fold(0.0f32, f32::max);
        let socket = if first.parent.is_some() {
            1.0 + mp.socket_flare
        } else {
            1.0
        };
        let radial = ((mp.radial_per_meter * r_max * socket).round() as i32)
            .clamp(mp.min_radial.max(3) as i32, mp.max_radial.max(3) as i32)
            as usize;
        // A single-segment branch is emitted as one cone rather than a tube plus a cone.
        let rings = run.len() + usize::from(first.parent.is_some());
        let single = first.parent.is_some() && rings == 2;
        let tris = if single {
            radial
        } else {
            (rings - 1) * radial * 2 + radial
        };
        *tris_by_level.entry(first.level).or_default() += tris;
        tris_cones += radial;
        if run.len() == 1 {
            tris_single += tris;
        }
        if let Some(p) = first.parent {
            let parent = &sk.nodes[p as usize];
            let reach = run
                .iter()
                .map(|&i| (sk.nodes[i as usize].position - parent.position).length())
                .fold(0.0f32, f32::max);
            if reach < parent.radius {
                fully_inside += 1;
            }
        }
    }
    let mut by_level: Vec<_> = tris_by_level.into_iter().collect();
    by_level.sort_by_key(|(l, _)| *l);
    let est: usize = by_level.iter().map(|(_, t)| *t).sum();
    println!("
estimated bark triangles by level");
    for (level, tris) in &by_level {
        println!(
            "  level {level}: {tris:>7}  ({:>4.1}%)",
            *tris as f32 / est as f32 * 100.0
        );
    }
    println!("  estimated {est} against {} actual", mesh.triangle_count());
    println!(
        "
  from stems of a single node: {tris_single} ({:.1}%)",
        tris_single as f32 / est as f32 * 100.0
    );
    println!(
        "  end cones, one per stem:     {tris_cones} ({:.1}%)",
        tris_cones as f32 / est as f32 * 100.0
    );
    println!("  stems that never clear their parent: {fully_inside}");
    println!(
        "
share of the whole tree: bark {:.0}%, leaves {:.0}%",
        mesh.triangle_count() as f32 / (mesh.triangle_count() + leaves.triangle_count()) as f32 * 100.0,
        leaves.triangle_count() as f32 / (mesh.triangle_count() + leaves.triangle_count()) as f32 * 100.0
    );

}
