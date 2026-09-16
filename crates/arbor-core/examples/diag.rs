use arbor_core::species::{parse_species, OAK_RON, PINE_RON};
use arbor_core::{build_mesh, grow};
use glam::Vec3;

fn main() {
    for (name, src) in [("oak", OAK_RON), ("pine", PINE_RON)] {
        let p = parse_species(src).unwrap();
        let sk = grow(&p);
        println!("=== {name} === nodes={} trunk.radius={}", sk.nodes.len(), p.trunk.radius);

        // trunk chain radius profile
        let mut cur = 0usize;
        let mut prof = Vec::new();
        loop {
            let n = &sk.nodes[cur];
            prof.push((n.position.y, n.radius, n.children.len()));
            match n.children.iter().find(|&&c| sk.nodes[c as usize].level == 0) {
                Some(&c) => cur = c as usize,
                None => break,
            }
        }
        println!("trunk chain len={}", prof.len());
        for (i, (y, r, c)) in prof.iter().enumerate() {
            if i % 2 == 0 || i == prof.len() - 1 {
                println!("  i={i:2} y={y:6.2} r={r:.4} children={c}");
            }
        }

        // gap analysis: distance between a child-branch first node and its attach parent
        let mut gaps: Vec<f32> = Vec::new();
        for n in &sk.nodes {
            if let Some(par) = n.parent {
                let pn = &sk.nodes[par as usize];
                if pn.level != n.level {
                    gaps.push((n.position - pn.position).length());
                }
            }
        }
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("branch-start offsets: n={} min={:.3} med={:.3} max={:.3}",
            gaps.len(), gaps.first().unwrap_or(&0.0),
            gaps.get(gaps.len()/2).unwrap_or(&0.0), gaps.last().unwrap_or(&0.0));

        // stem-break analysis (mesh stems): count nodes with >1 child at same level etc.
        let mut breaks = 0;
        for n in &sk.nodes {
            if let Some(par) = n.parent {
                let pn = &sk.nodes[par as usize];
                if pn.level == n.level && pn.children.len() != 1 { breaks += 1; }
            }
        }
        println!("same-level continuations that start a NEW mesh stem (=> gap): {breaks}");

        let m = build_mesh(&sk, &p);
        println!("verts={} tris={}", m.vertex_count(), m.triangle_count());
        let _ = Vec3::ZERO;
        println!();
    }
}
