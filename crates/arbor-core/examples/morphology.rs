//! What the branch structure actually measures, level by level.
//!
//! cargo run --release -p arbor-core --example morphology -- <preset>
//!
//! Screenshots say a crown looks wrong; they do not say why. These are the numbers a
//! forester would take off a real tree, so a preset can be argued with rather than
//! eyeballed: how long the stems at each level run, how much they vary, how far they
//! turn between base and tip, and how many children each one carries.
//!
//! The spread of stem length within a level is the one to watch. A real crown is a few
//! dominant limbs and a great many suppressed ones, so the spread is wide and the
//! distribution is skewed. A level whose stems are all near the mean is a bottle brush.

use std::collections::HashMap;

use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{grow, Skeleton};

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "oak".into());
    let Some((_, src)) = builtin_presets().into_iter().find(|(n, _)| *n == name) else {
        let names: Vec<&str> = builtin_presets().into_iter().map(|(n, _)| n).collect();
        eprintln!("usage: morphology [{}]", names.join("|"));
        std::process::exit(2);
    };
    let params = parse_species(src).expect("preset parses");
    let sk = grow(&params);

    println!("=== {} ===", params.name);
    println!(
        "{:>5}  {:>6}  {:>26}  {:>14}  {:>8}  {:>8}",
        "level", "stems", "length m  (min/mean/max)", "spread", "turn deg", "children"
    );

    let mut by_level: HashMap<u8, Vec<Stem>> = HashMap::new();
    for run in sk.stem_runs() {
        if run.is_empty() {
            continue;
        }
        by_level
            .entry(sk.nodes[run[0] as usize].level)
            .or_default()
            .push(measure(&sk, &run));
    }

    let mut levels: Vec<u8> = by_level.keys().copied().collect();
    levels.sort();
    for level in levels {
        let stems = &by_level[&level];
        let lengths: Vec<f32> = stems.iter().map(|s| s.length).collect();
        let (lo, mean, hi) = span(&lengths);
        let sd = deviation(&lengths, mean);
        let turn = mean_of(&stems.iter().map(|s| s.turn_deg).collect::<Vec<_>>());
        let kids = mean_of(&stems.iter().map(|s| s.children as f32).collect::<Vec<_>>());
        println!(
            "{level:>5}  {:>6}  {lo:>8.2} {mean:>8.2} {hi:>8.2}  {:>13.0}%  {turn:>8.1}  {kids:>8.1}",
            stems.len(),
            if mean > 1e-4 { sd / mean * 100.0 } else { 0.0 },
        );
    }

    // Where a stem's children sit along it decides whether the crown is carried out to
    // the periphery or spread evenly down every limb. Real broadleaves put their
    // strongest growth near the ends of what grew last year.
    let mut distal: Vec<f32> = Vec::new();
    for node in &sk.nodes {
        if node.level > 0 && node.parent.is_some() {
            distal.push(node.stem_fraction);
        }
    }
    let (_, mean_frac, _) = span(&distal);
    println!(
        "\nchildren attach on average {:.0}% along their parent (0.5 is an even spread)",
        mean_frac * 100.0
    );

    // Raw node counts, which include the wood the tree has lost. The runs above hide
    // that, so a level can read as empty there and be full here.
    let mut raw = [0usize; 8];
    let mut gone = [0usize; 8];
    let mut born = [0usize; 8];
    for node in &sk.nodes {
        let l = (node.level as usize).min(7);
        raw[l] += 1;
        if node.dead || node.broken {
            gone[l] += 1;
        }
        if node.parent.is_none_or(|p| sk.nodes[p as usize].stem != node.stem) {
            born[l] += 1;
        }
    }
    println!(
        "\n{:>5}  {:>9}  {:>11}  {:>13}",
        "level", "nodes", "stems born", "lost to death"
    );
    for (l, &count) in raw.iter().enumerate().take(7) {
        if count == 0 {
            continue;
        }
        println!(
            "{l:>5}  {count:>9}  {:>11}  {:>12}%",
            born[l],
            gone[l] * 100 / count.max(1)
        );
    }
}

struct Stem {
    length: f32,
    turn_deg: f32,
    children: usize,
}

fn measure(sk: &Skeleton, run: &[u32]) -> Stem {
    let mut length = 0.0;
    let mut children = 0;
    // A stem's first segment runs from where it attaches to its parent, and that node
    // belongs to the parent's run. Leaving it out reports every one-segment stem as
    // having no length at all.
    if let Some(p) = sk.nodes[run[0] as usize].parent {
        length += (sk.nodes[run[0] as usize].position - sk.nodes[p as usize].position).length();
    }
    for w in run.windows(2) {
        length += (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length();
    }
    for &i in run {
        // A child on another stem is a branch; one on this stem is the next segment.
        children += sk.nodes[i as usize]
            .children
            .iter()
            .filter(|&&c| sk.nodes[c as usize].stem != sk.nodes[i as usize].stem)
            .count();
    }
    // How far the stem turned from where it set out, which is what decides whether it
    // reads as a limb that found its way or as a spoke.
    let turn_deg = if run.len() >= 2 {
        let first = sk.nodes[run[1] as usize].position - sk.nodes[run[0] as usize].position;
        let last = sk.nodes[run[run.len() - 1] as usize].position
            - sk.nodes[run[run.len() - 2] as usize].position;
        match (first.try_normalize(), last.try_normalize()) {
            (Some(a), Some(b)) => a.dot(b).clamp(-1.0, 1.0).acos().to_degrees(),
            _ => 0.0,
        }
    } else {
        0.0
    };
    Stem {
        length,
        turn_deg,
        children,
    }
}

fn span(v: &[f32]) -> (f32, f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let lo = v.iter().copied().fold(f32::MAX, f32::min);
    let hi = v.iter().copied().fold(f32::MIN, f32::max);
    (lo, v.iter().sum::<f32>() / v.len() as f32, hi)
}

fn deviation(v: &[f32], mean: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

fn mean_of(v: &[f32]) -> f32 {
    span(v).1
}
