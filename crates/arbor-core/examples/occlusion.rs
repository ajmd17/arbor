//! How much of a canopy is buried behind the rest of it.
//!
//! cargo run --release -p arbor-core --example occlusion -- [pine|oak]
//!
//! For every card this marches outward in a spread of directions and counts the cards
//! lying beyond it, scoring the card by its emptiest direction — the one it would be
//! seen from. That is a cheap stand-in for how much foliage a viewer has to look
//! through before reaching it.
//!
//! Measured 2026-09-17 to decide whether culling hidden cards was worth writing, and
//! the answer was no: with occlusion required from every direction, only 4% of the
//! oak's canopy and 7% of the pine's has twenty cards in front of it, and nothing at
//! all has forty. `leaves.leafy_length` already keeps foliage to the ends of the
//! shoots, so the crown is a shell rather than a solid volume and there is next to
//! nothing buried inside it to remove. Scoring by the outward ray alone says 21% and
//! 37%, which is the answer to a different question: a card hidden along one ray is
//! visible from somewhere else, and culling on that number punches holes in the crown
//! as the camera moves.

use std::collections::HashMap;

use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_leaves, grow};
use glam::Vec3;

const CELL: f32 = 0.9;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "oak".into());
    let Some((_, src)) = builtin_presets().into_iter().find(|(n, _)| *n == name) else {
        eprintln!("usage: occlusion [pine|oak]");
        std::process::exit(2);
    };
    let params = parse_species(src).expect("preset parses");
    let sk = grow(&params);
    let leaves = build_leaves(&sk, &params);

    let centres: Vec<Vec3> = leaves
        .positions
        .chunks_exact(4)
        .map(|q| {
            q.iter().fold(Vec3::ZERO, |a, p| a + Vec3::from(*p)) / 4.0
        })
        .collect();
    if centres.is_empty() {
        println!("no cards");
        return;
    }

    let key = |p: Vec3| {
        (
            (p.x / CELL).floor() as i32,
            (p.y / CELL).floor() as i32,
            (p.z / CELL).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), u32> = HashMap::new();
    for &c in &centres {
        *grid.entry(key(c)).or_insert(0) += 1;
    }

    let centre = centres.iter().fold(Vec3::ZERO, |a, c| a + *c) / centres.len() as f32;
    let reach = centres
        .iter()
        .map(|c| (*c - centre).length())
        .fold(0.0f32, f32::max);

    // A viewer is not confined to the outward ray, so a card is only safely hidden if
    // it is hidden from every direction. Marched over a spread of them and scored by
    // the emptiest one: that is the direction it would be seen from.
    let mut dirs: Vec<Vec3> = Vec::new();
    for i in 0..14 {
        // Fibonacci sphere, which spreads directions evenly without clumping at a pole.
        let y = 1.0 - (i as f32 + 0.5) / 14.0 * 2.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let a = i as f32 * 2.399_963_2;
        dirs.push(Vec3::new(a.cos() * r, y, a.sin() * r));
    }

    let mut beyond: Vec<u32> = Vec::with_capacity(centres.len());
    for &c in &centres {
        let mut least = u32::MAX;
        for &dir in &dirs {
            let mut total = 0;
            let mut seen = std::collections::HashSet::new();
            let mut t = CELL;
            while t < reach {
                let cell = key(c + dir * t);
                if seen.insert(cell) {
                    total += grid.get(&cell).copied().unwrap_or(0);
                }
                t += CELL * 0.5;
            }
            least = least.min(total);
        }
        beyond.push(least);
    }

    let mut sorted = beyond.clone();
    sorted.sort_unstable();
    let at = |q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize];
    println!("=== {} ===", params.name);
    println!("{} cards, crown reach {reach:.1} m", centres.len());
    println!(
        "cards beyond, by quantile:  10% {}   50% {}   90% {}   max {}",
        at(0.10),
        at(0.50),
        at(0.90),
        sorted[sorted.len() - 1]
    );
    for threshold in [5u32, 10, 20, 40, 80] {
        let buried = beyond.iter().filter(|&&b| b >= threshold).count();
        println!(
            "  {:>3} or more cards outside it: {:>6} cards ({:>4.1}% of the canopy)",
            threshold,
            buried,
            buried as f32 / centres.len() as f32 * 100.0
        );
    }
}
