//! Measures how far each stem turns along its length, to find stems that curl.
use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{grow, Skeleton};
use glam::Vec3;

fn main() {
    let src = builtin_presets()
        .into_iter()
        .find(|(n, _)| *n == "pine")
        .unwrap()
        .1;
    let params = parse_species(src).unwrap();
    let sk = grow(&params);
    report(&sk);
}

fn report(sk: &Skeleton) {
    // Per level: worst total turn, worst turn per metre, and how many stems loop.
    let mut worst = [(0.0f32, 0usize); 8];
    let mut over_180 = [0usize; 8];
    let mut over_360 = [0usize; 8];
    let mut count = [0usize; 8];
    let mut worst_per_m = [0.0f32; 8];

    for run in sk.stem_runs() {
        if run.len() < 3 {
            continue;
        }
        let level = (sk.nodes[run[0] as usize].level as usize).min(7);
        let mut dirs: Vec<Vec3> = Vec::new();
        let mut len = 0.0f32;
        for w in run.windows(2) {
            let a = sk.nodes[w[0] as usize].position;
            let b = sk.nodes[w[1] as usize].position;
            let d = b - a;
            if d.length() > 1e-6 {
                len += d.length();
                dirs.push(d.normalize());
            }
        }
        let mut turn = 0.0f32;
        for w in dirs.windows(2) {
            turn += w[0].dot(w[1]).clamp(-1.0, 1.0).acos();
        }
        let deg = turn.to_degrees();
        count[level] += 1;
        if deg > worst[level].0 {
            worst[level] = (deg, run.len());
        }
        if len > 1e-3 {
            worst_per_m[level] = worst_per_m[level].max(deg / len);
        }
        if deg > 180.0 {
            over_180[level] += 1;
        }
        if deg > 360.0 {
            over_360[level] += 1;
        }
    }

    println!("level  stems   worst turn  (segs)   worst deg/m   >180deg   >360deg");
    for l in 0..8 {
        if count[l] == 0 {
            continue;
        }
        println!(
            "{l:>5} {:>7} {:>11.0}° {:>7} {:>13.0} {:>9} {:>9}",
            count[l], worst[l].0, worst[l].1, worst_per_m[l], over_180[l], over_360[l]
        );
    }
}
