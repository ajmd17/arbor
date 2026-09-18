//! TEMP probe: children at stem tips, and laterals long against the parent ahead of them.
use arbor_core::{build_leaves, build_mesh, grow};
use arbor_core::species::{builtin_presets, parse_species};

fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "{:<18} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} | {:>7} {:>7} {:>7} | {:>6} {:>6}",
        "species", "lats", "atTip", "tip10%", "forks", "fTip", "f10%", ">ahead", ">2xahd", ">.5L@70",
        "stems", "barePct"
    );
    println!("  then: nodes, living tips, crown reach p50/p95 (m from trunk axis), height, lateral >ahead by parent level");
    for (name, src) in builtin_presets() {
        if !only.is_empty() && !only.iter().any(|o| o == name) {
            continue;
        }
        let p = parse_species(src).unwrap();
        let sk = grow(&p);
        let n = sk.nodes.len();
        // Arc along its own stem of every node, measured from where the stem leaves its
        // parent node.
        let mut arc = vec![0.0f32; n];
        let mut len = vec![0.0f32; n];
        for i in 0..n {
            let nd = &sk.nodes[i];
            if let Some(pp) = nd.parent {
                let run = (nd.position - sk.nodes[pp as usize].position).length();
                let same = sk.nodes[pp as usize].stem == nd.stem;
                arc[i] = if same { arc[pp as usize] + run } else { run };
            }
            if !nd.broken {
                let s = nd.stem as usize;
                len[s] = len[s].max(arc[i]);
            }
        }
        let (mut lats, mut at_tip, mut tip10) = (0, 0, 0);
        let (mut forks, mut f_tip, mut f10) = (0, 0, 0);
        let (mut over, mut over2, mut late_long) = (0, 0, 0);
        let mut stems = 0;
        let mut bare_sum = 0.0f32;
        let mut bare_n = 0;
        // Per parent stem: the arc of its last child.
        let mut last_child = vec![-1.0f32; n];
        for i in 0..n {
            let nd = &sk.nodes[i];
            if nd.broken || nd.dead {
                continue;
            }
            let Some(pp) = nd.parent else { continue };
            let par = &sk.nodes[pp as usize];
            if par.stem == nd.stem {
                continue;
            }
            // `nd` heads a stem.
            let plen = len[par.stem as usize];
            if plen <= 0.0 {
                continue;
            }
            let at = arc[pp as usize];
            let ahead = plen - at;
            last_child[par.stem as usize] = last_child[par.stem as usize].max(at);
            let fork = par.level == nd.level;
            if fork {
                forks += 1;
                if ahead < 1e-4 {
                    f_tip += 1;
                }
                if ahead < 0.1 * plen {
                    f10 += 1;
                }
            } else {
                lats += 1;
                if ahead < 1e-4 {
                    at_tip += 1;
                }
                if ahead < 0.1 * plen {
                    tip10 += 1;
                }
                let mine = len[nd.stem as usize];
                if mine > ahead.max(1e-3) {
                    over += 1;
                }
                if mine > 2.0 * ahead.max(1e-3) {
                    over2 += 1;
                }
                if at > 0.7 * plen && mine > 0.5 * plen {
                    late_long += 1;
                }
            }
        }
        for s in 0..n {
            if last_child[s] >= 0.0 && len[s] > 0.0 {
                stems += 1;
                bare_sum += (len[s] - last_child[s]) / len[s];
                bare_n += 1;
            }
        }
        println!(
            "{:<18} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} | {:>7} {:>7} {:>7} | {:>6} {:>5.1}%",
            name,
            lats,
            at_tip,
            tip10,
            forks,
            f_tip,
            f10,
            over,
            over2,
            late_long,
            stems,
            100.0 * bare_sum / bare_n.max(1) as f32
        );
        let mut reach: Vec<f32> = sk
            .nodes
            .iter()
            .filter(|nd| !nd.broken && !nd.dead && nd.children.is_empty() && nd.level > 0)
            .map(|nd| (nd.position.x * nd.position.x + nd.position.z * nd.position.z).sqrt())
            .collect();
        reach.sort_by(f32::total_cmp);
        let q = |f: f32| reach.get(((reach.len() as f32 * f) as usize).min(reach.len().saturating_sub(1))).copied().unwrap_or(0.0);
        let mut by_level = [(0u32, 0u32); 6];
        for i in 0..n {
            let nd = &sk.nodes[i];
            if nd.broken || nd.dead { continue; }
            let Some(pp) = nd.parent else { continue };
            let par = &sk.nodes[pp as usize];
            if par.stem == nd.stem || par.level == nd.level { continue; }
            let plen = len[par.stem as usize];
            let ahead = plen - arc[pp as usize];
            let e = &mut by_level[(par.level as usize).min(5)];
            e.0 += 1;
            if len[nd.stem as usize] > ahead.max(1e-3) { e.1 += 1; }
        }
        let lv: Vec<String> = by_level.iter().enumerate().filter(|(_, e)| e.0 > 0).map(|(l, e)| format!("L{l}:{}/{}", e.1, e.0)).collect();
        println!(
            "    nodes {:>6}  tips {:>6}  reach {:>5.2}/{:>5.2}  h {:>5.1}  {}",
            n, reach.len(), q(0.5), q(0.95), sk.stats().height, lv.join(" ")
        );
        println!("    leaves {:>6}  bark tris {:>7}", build_leaves(&sk, &p).leaf_count(), build_mesh(&sk, &p).triangle_count());
    }
}
