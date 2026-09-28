//! Wildflowers for the blade atlas: a stem, and a head drawn as the eye meets it from
//! the side, among the grass of a cell.
//!
//! Three shapes cover most meadow flowers at card distance:
//!
//! - `Radial`: a disc of petals round a centre, seen tilted, so the far petals are
//!   drawn behind the centre and the near ones in front of it. Daisies, buttercups,
//!   dandelions, knapweed, depending on petal count, width and colour.
//! - `Spike`: florets stacked up the top of the stem, a cone narrowing to buds at the
//!   top. Lupins, foxgloves, salvias.
//! - `Umbel`: rays fanning from the top of the stem to a flat head of tiny florets.
//!   Yarrow, cow parsley, hogweed.
//! - `Brush`: a scaly knob with a tuft of thin florets spraying from its top. Knapweed,
//!   thistles, cornflowers. `petals` is the florets in the tuft, `centre` and
//!   `centre_size` the knob.
//!
//! Everything is built from the same strokes grass blades are, so it composites, lights
//! and bleeds exactly as they do.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::seed::PortableRng;

/// The shape of a flower's head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowerKind {
    #[default]
    Radial,
    Spike,
    Umbel,
    Brush,
}

/// The flowers standing in one atlas cell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowerParams {
    pub kind: FlowerKind,
    /// Flowers in the cell.
    pub count: u32,
    /// Stem length, as a fraction of the cell's height: shortest and longest.
    pub height: (f32, f32),
    /// Across the head, as a fraction of the cell's height: the disc of a radial
    /// flower, the base of a spike, the spread of an umbel.
    pub size: f32,
    /// Length of a spike's flowering part, as a fraction of the cell's height.
    pub length: f32,
    /// Petals round a radial head, and how broad each is against its length: a
    /// daisy's many thin ones, a buttercup's five round ones.
    pub petals: u32,
    pub petal_width: f32,
    /// A radial flower's centre, against the head's size, and its colour.
    pub centre_size: f32,
    pub centre: [f32; 3],
    /// The flower's colour, sRGB, and its spread from flower to flower.
    pub color: [f32; 3],
    pub color_variance: f32,
    pub stem: [f32; 3],
    /// Degrees a stem may lean either way.
    pub lean_deg: f32,
}

impl Default for FlowerParams {
    fn default() -> Self {
        Self {
            kind: FlowerKind::Radial,
            count: 6,
            height: (0.45, 0.8),
            size: 0.06,
            length: 0.3,
            petals: 18,
            petal_width: 0.2,
            centre_size: 0.3,
            centre: [0.85, 0.65, 0.12],
            color: [0.92, 0.92, 0.88],
            color_variance: 0.08,
            stem: [0.20, 0.30, 0.12],
            lean_deg: 8.0,
        }
    }
}

/// A shape to paint: a spine, its half width along it, and what it looks like. The
/// cell painter turns these into its own strokes.
pub struct Shape {
    pub spine: Vec<[f32; 2]>,
    pub half: Vec<f32>,
    pub root: [f32; 3],
    pub tip: [f32; 3],
    /// Darkening at the root end, 1 for none.
    pub root_shade: f32,
    /// How far it turns from the card's plane, in radians, for its normal.
    pub tilt: f32,
    pub roughness: f32,
    /// How much of a grass blade's detail it is drawn with: the lighter midrib, the
    /// veins either side and the fold. Right for a blade, a little for a petal, and
    /// none for a floret, where it draws rings and blotches.
    pub detail: f32,
}

/// One flower, as shapes in the order they are painted, standing at `root_x` on the
/// bottom of a cell `w` by `h` pixels. `stem_spine` builds a stem the way the cell
/// builds blades, so flowers bend as the grass round them does.
pub fn flower(
    p: &FlowerParams,
    rng: &mut PortableRng,
    root_x: f32,
    w: f32,
    h: f32,
    stem_spine: &dyn Fn(f32, f32, f32, f32) -> Vec<[f32; 2]>,
) -> Vec<Shape> {
    let (lo, hi) = (p.height.0.min(p.height.1), p.height.0.max(p.height.1));
    let length = h * rng.random_range(lo..=hi.max(lo + 1e-4)).clamp(0.05, 0.97);
    let lean = rng.random_range(-1.0f32..=1.0) * p.lean_deg.to_radians();
    let size = (p.size * h).max(3.0);
    // Leave room at the top for whatever sits on the stem.
    let head_room = match p.kind {
        FlowerKind::Radial => size * 0.5,
        FlowerKind::Spike => 0.0,
        FlowerKind::Umbel => size * 0.45,
        FlowerKind::Brush => size * 0.9,
    };
    let length = length.min(h - 8.0 - head_room);
    let spine = stem_spine(root_x, lean, length, w);
    let tint = vary(rng, p.color_variance);
    let color = mul(p.color, tint);
    let mut out = Vec::new();

    let stem_w = (size * 0.07).clamp(0.8, 3.0);
    let n = spine.len();
    let spike_from = if p.kind == FlowerKind::Spike {
        (1.0 - (p.length * h) / length.max(1.0)).clamp(0.2, 0.95)
    } else {
        1.0
    };
    out.push(Shape {
        half: (0..n).map(|k| stem_w * (1.0 - 0.35 * k as f32 / (n - 1) as f32)).collect(),
        spine: spine.clone(),
        root: mul(p.stem, [0.8; 3]),
        tip: p.stem,
        root_shade: 0.8,
        tilt: rng.random_range(-0.3..=0.3),
        roughness: 0.6,
        detail: 0.5,
    });

    let top = spine[n - 1];
    let dir = {
        let a = spine[n - 2];
        let d = [top[0] - a[0], top[1] - a[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-4);
        [d[0] / l, d[1] / l]
    };
    match p.kind {
        FlowerKind::Radial => radial(p, rng, top, size, color, &mut out),
        FlowerKind::Spike => spike(p, rng, &spine, spike_from, size, color, &mut out),
        FlowerKind::Umbel => umbel(p, rng, top, dir, size, color, &mut out),
        FlowerKind::Brush => brush(p, rng, top, dir, size, color, &mut out),
    }
    out
}

/// A disc of petals seen tilted toward the viewer by `face`: 0 edge-on, 1 face-on.
///
/// The petals are a ring round the centre, not spokes through it: each starts at the
/// edge of the disc, and the disc is drawn last, over the petals' roots, as a real
/// flower's head sits on its ring of rays. The far petals go down first so the near
/// ones overlap them.
fn radial(p: &FlowerParams, rng: &mut PortableRng, at: [f32; 2], size: f32, color: [f32; 3], out: &mut Vec<Shape>) {
    // Mostly turned toward the light and the viewer: an edge-on disc is a bar.
    let face = rng.random_range(0.4f32..0.9);
    let turn = rng.random_range(-0.25f32..0.25);
    let petals = p.petals.clamp(3, 64);
    let r = size * 0.5;
    let rc = r * p.centre_size.clamp(0.0, 0.9);
    let spin = rng.random_range(0.0..std::f32::consts::TAU);
    let mut ring: Vec<(f32, Shape)> = Vec::new();
    for i in 0..petals {
        let theta = spin + std::f32::consts::TAU * (i as f32 + rng.random_range(-0.15..=0.15)) / petals as f32;
        let (s, c) = theta.sin_cos();
        // In the disc: x across the screen, y away from the viewer. Seen tilted, the
        // away axis shortens by `face` and runs up the screen.
        let d = rotate([c, -s * face], turn);
        let across = ((s * s) + (c * face) * (c * face)).sqrt().max(0.2);
        let len = r * rng.random_range(0.9..=1.05);
        let width = len * p.petal_width.clamp(0.05, 1.0) * across;
        // From just inside the disc's edge, so the disc covers the root and no gap
        // shows between them.
        let start = rc * 0.8;
        let spine: Vec<[f32; 2]> = (0..6)
            .map(|k| {
                let f = start + (len - start) * k as f32 / 5.0;
                [at[0] + d[0] * f, at[1] + d[1] * f]
            })
            .collect();
        // Narrow at the root, broadest two thirds out, rounded at the tip.
        let half = (0..6)
            .map(|k| {
                let u = k as f32 / 5.0;
                (width * 0.5 * (0.45 + 0.55 * (std::f32::consts::PI * u * 0.75).sin()) * (1.0 - u.powi(4) * 0.6)).max(0.5)
            })
            .collect();
        let shade = 1.0 + rng.random_range(-0.04f32..=0.04);
        ring.push((
            s,
            Shape {
                spine,
                half,
                root: [color[0] * 0.9 * shade, color[1] * 0.9 * shade, color[2] * 0.9 * shade],
                tip: [color[0] * shade, color[1] * shade, color[2] * shade],
                root_shade: 1.0,
                // Cupped a little, toward the viewer on the near side.
                tilt: -s * 0.35,
                roughness: 0.85,
                detail: 0.3,
            },
        ));
    }
    // Far (up the screen, positive) first.
    ring.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.extend(ring.into_iter().map(|(_, s)| s));

    if rc > 0.5 {
        // The disc, domed: seen tilted it is an ellipse, and its top shows above the
        // near petals. Built as a short fat stroke across the head.
        let a = rotate([-rc, 0.0], turn);
        let b = rotate([rc, 0.0], turn);
        let lift = rotate([0.0, -rc * 0.15], turn);
        let spine: Vec<[f32; 2]> = (0..7)
            .map(|k| {
                let f = k as f32 / 6.0;
                [at[0] + lift[0] + a[0] + (b[0] - a[0]) * f, at[1] + lift[1] + a[1] + (b[1] - a[1]) * f]
            })
            .collect();
        let half = (0..7)
            .map(|k| {
                let u = (k as f32 / 6.0) * 2.0 - 1.0;
                (rc * face.max(0.45) * (1.0 - u * u).max(0.0).sqrt()).max(0.6)
            })
            .collect();
        out.push(Shape {
            spine,
            half,
            root: p.centre,
            tip: p.centre,
            root_shade: 1.0,
            tilt: 0.0,
            roughness: 0.85,
            detail: 0.0,
        });
    }
}

/// Florets up the top of a stem, widest at the bottom, buds at the top.
fn spike(
    p: &FlowerParams,
    rng: &mut PortableRng,
    spine: &[[f32; 2]],
    from: f32,
    size: f32,
    color: [f32; 3],
    out: &mut Vec<Shape>,
) {
    let floret = size * 0.38;
    let levels = (((1.0 - from) * spine_length(spine)) / (floret * 0.5)).ceil().clamp(3.0, 80.0) as usize;
    let mut front = Vec::new();
    for j in 0..levels {
        let u = j as f32 / (levels - 1).max(1) as f32;
        let (at, dir) = along(spine, from + (1.0 - from) * u);
        let across = [-dir[1], dir[0]];
        let half_w = size * 0.5 * (1.0 - 0.75 * u);
        // Buds at the top are closed, smaller and greener.
        let bud = ((u - 0.75) / 0.25).clamp(0.0, 1.0);
        // The lower flowers open first and fade: paler and pinker toward the bottom.
        let faded = mix(color, [0.75, 0.62, 0.80], 0.15 * (1.0 - u) * rng.random::<f32>());
        let col = mix(faded, mix(color, p.stem, 0.6), bud);
        // Whorls: the florets of a spike come in rings with a gap between.
        if j % 3 == 2 && u < 0.8 {
            continue;
        }
        for side in [-1.0f32, 1.0, 0.0] {
            let off = side * half_w * rng.random_range(0.6..=1.0);
            // Florets stand out and slightly down from the stem, pea-flower fashion.
            let out_dir = normalize([across[0] * side * 0.9 - dir[0] * 0.3, across[1] * side * 0.9 - dir[1] * 0.3]);
            let base = [at[0] + across[0] * off * 0.5, at[1] + across[1] * off * 0.5];
            let len = floret * (1.0 - 0.4 * u) * rng.random_range(0.8..=1.1);
            let d = if side == 0.0 { dir } else { out_dir };
            let spine: Vec<[f32; 2]> = (0..4)
                .map(|k| {
                    let f = len * k as f32 / 3.0;
                    [base[0] + d[0] * f, base[1] + d[1] * f]
                })
                .collect();
            let half = (0..4)
                .map(|k| (len * 0.45 * (std::f32::consts::PI * (k as f32 + 0.6) / 4.2).sin()).max(0.6))
                .collect();
            let shade = vary(rng, p.color_variance.max(0.05));
            // The keel of a pea flower is dark; the banner above it pales toward white.
            let banner = mix(col, [0.92, 0.90, 0.95], 0.2 * rng.random::<f32>() * (1.0 - bud));
            let shape = Shape {
                spine,
                half,
                root: mul(col, [0.6 * shade[0], 0.6 * shade[1], 0.6 * shade[2]]),
                tip: mul(banner, shade),
                root_shade: 1.0,
                tilt: side * 0.6,
                roughness: 0.85,
                detail: 0.25,
            };
            if side == 0.0 {
                front.push(shape);
            } else {
                out.push(shape);
            }
        }
    }
    out.extend(front);
}

/// Rays fanning up from the top of the stem to a domed head of tiny florets.
///
/// Each ray carries a small cluster, and the clusters crowd into one head. The florets
/// are specks a pixel or two across, packed with small gaps and nearly one colour: the
/// head reads by its outline and its texture, not by any one floret.
fn umbel(
    p: &FlowerParams,
    rng: &mut PortableRng,
    at: [f32; 2],
    dir: [f32; 2],
    size: f32,
    color: [f32; 3],
    out: &mut Vec<Shape>,
) {
    let rays = rng.random_range(8..=14);
    let r = size * 0.5;
    let face = rng.random_range(0.35f32..0.7);
    let mut heads = Vec::new();
    for i in 0..rays {
        let x = (i as f32 + 0.5) / rays as f32 * 2.0 - 1.0 + rng.random_range(-0.06..=0.06);
        // Domed rather than dead flat, so from the side it is a mound, not a bar.
        let depth = rng.random_range(-1.0f32..=1.0);
        let end = [at[0] + x * r, at[1] - r * (0.4 + 0.3 * (1.0 - x * x)) + depth * r * face * 0.3];
        let spine: Vec<[f32; 2]> = (0..4)
            .map(|k| {
                let f = k as f32 / 3.0;
                // Out from the stem along its own direction first, then over.
                let bow = [at[0] + dir[0] * r * 0.3 * f, at[1] + dir[1] * r * 0.3 * f];
                [bow[0] + (end[0] - bow[0]) * f * f, bow[1] + (end[1] - bow[1]) * f * f]
            })
            .collect();
        out.push(Shape {
            half: vec![0.55; 4],
            spine,
            root: p.stem,
            tip: p.stem,
            root_shade: 1.0,
            tilt: 0.0,
            roughness: 0.6,
            detail: 0.0,
        });
        // A little dome of florets on each ray, wider than it is tall.
        let cluster = r * 0.26;
        let dot = (size * 0.03).max(0.8);
        let dots = ((cluster * cluster) / (dot * dot) * 0.9).clamp(6.0, 60.0) as usize;
        for _ in 0..dots {
            let a = rng.random_range(0.0..std::f32::consts::PI);
            let d = rng.random_range(0.0f32..1.0).sqrt() * cluster;
            let c = [end[0] + a.cos() * d, end[1] - a.sin() * d * 0.55];
            // Nearly white throughout; the ones low in the head a touch darker.
            let v = 0.92 + 0.08 * a.sin() + rng.random_range(-0.03f32..=0.03);
            let col = [color[0] * v, color[1] * v, color[2] * v];
            heads.push(Shape {
                spine: vec![c, [c[0] + dot * 0.3, c[1]]],
                half: vec![dot, dot],
                root: col,
                tip: col,
                root_shade: 1.0,
                tilt: 0.0,
                roughness: 0.6,
                detail: 0.0,
            });
        }
    }
    out.extend(heads);
}

/// A shaving-brush head: a hard, scaly knob with a tuft of thin florets spraying up
/// and out of its top. Knapweed, thistles, cornflowers.
///
/// The florets go down first and the knob over their bases, so they spring from it
/// rather than being stuck on. A few longer ones splay wide, as greater knapweed's
/// ragged outer ring does. The knob carries rows of scales, dark with pale fringed
/// tips, which is what tells it from a plain green bud at any distance.
fn brush(
    p: &FlowerParams,
    rng: &mut PortableRng,
    at: [f32; 2],
    dir: [f32; 2],
    size: f32,
    color: [f32; 3],
    out: &mut Vec<Shape>,
) {
    let across = [-dir[1], dir[0]];
    let ri = (size * 0.5 * p.centre_size.clamp(0.2, 1.0)).max(1.5);
    let knob_len = ri * 2.3;
    let crown = [at[0] + dir[0] * knob_len * 0.85, at[1] + dir[1] * knob_len * 0.85];
    let pale = mix(color, [0.95, 0.85, 0.95], 0.3);

    // The tuft, in a random order so no side is always on top.
    let florets = p.petals.clamp(8, 96);
    let outer = florets / 6;
    let mut tuft: Vec<Shape> = Vec::new();
    for i in 0..florets + outer {
        let ring = i >= florets;
        let u = rng.random_range(-1.0f32..=1.0);
        // Upright in the middle, fanning out toward the edges of the tuft.
        let spread = if ring { rng.random_range(1.0f32..1.45) * u.signum() } else { u * 0.95 };
        let (s, c) = spread.sin_cos();
        let d = [dir[0] * c + across[0] * s, dir[1] * c + across[1] * s];
        let len = size * 0.62 * rng.random_range(0.75..=1.1) * if ring { 1.2 } else { 1.0 - 0.25 * u.abs() };
        let base = [crown[0] + across[0] * u * ri * 0.7, crown[1] + across[1] * u * ri * 0.7];
        // Florets droop a little as they reach out, the outer ones most.
        let droop = spread * 0.25;
        let spine: Vec<[f32; 2]> = (0..5)
            .map(|k| {
                let f = k as f32 / 4.0;
                let (sd, cd) = (spread + droop * f).sin_cos();
                let dd = [dir[0] * cd + across[0] * sd, dir[1] * cd + across[1] * sd];
                let bend = [(dd[0] - d[0]) * f * 0.5, (dd[1] - d[1]) * f * 0.5];
                [base[0] + (d[0] + bend[0]) * len * f, base[1] + (d[1] + bend[1]) * len * f]
            })
            .collect();
        let w = (size * if ring { 0.026 } else { 0.016 }).max(0.5);
        let half = (0..5)
            .map(|k| {
                let f = k as f32 / 4.0;
                // A thin tube that flares to a split tip.
                w * (0.6 + 0.6 * f.powi(2)).min(1.0)
            })
            .collect();
        let shade = 1.0 + rng.random_range(-0.08f32..=0.08);
        tuft.push(Shape {
            spine,
            half,
            root: mul(color, [0.7 * shade; 3]),
            tip: mul(pale, [shade; 3]),
            root_shade: 1.0,
            tilt: s * 0.4,
            roughness: 0.85,
            detail: 0.0,
        });
    }
    out.extend(tuft);

    // The knob: an egg, widest a little below its middle.
    let spine: Vec<[f32; 2]> = (0..7)
        .map(|k| {
            let f = k as f32 / 6.0 * knob_len;
            [at[0] + dir[0] * f, at[1] + dir[1] * f]
        })
        .collect();
    let half = (0..7)
        .map(|k| {
            let f = k as f32 / 6.0;
            (ri * (std::f32::consts::PI * (0.1 + 0.85 * f)).sin().powf(0.7) * (1.0 - 0.2 * f)).max(0.6)
        })
        .collect();
    out.push(Shape {
        spine,
        half,
        root: mul(p.centre, [0.8; 3]),
        tip: p.centre,
        root_shade: 1.0,
        tilt: 0.0,
        roughness: 0.85,
        detail: 0.0,
    });

    // Scales in rows up the knob, alternating, each dark at its base with a pale
    // fringed tip pointing up.
    // Knapweed's bracts are near black, with only a thin paler fringe.
    let fringe = mix(p.centre, [0.70, 0.62, 0.46], 0.22);
    let rows = 5;
    for row in 0..rows {
        let along = 0.15 + 0.7 * row as f32 / (rows - 1) as f32;
        let width_here = ri * (std::f32::consts::PI * (0.1 + 0.85 * along)).sin().powf(0.7);
        let per = 4 + (row % 2);
        for k in 0..per {
            let x = ((k as f32 + 0.5 + rng.random_range(-0.2f32..=0.2)) / per as f32 * 2.0 - 1.0) * width_here * 0.8;
            let base = [
                at[0] + dir[0] * knob_len * along + across[0] * x,
                at[1] + dir[1] * knob_len * along + across[1] * x,
            ];
            let len = ri * 0.55;
            let spine: Vec<[f32; 2]> = (0..4)
                .map(|j| {
                    let f = j as f32 / 3.0 * len;
                    [base[0] + dir[0] * f, base[1] + dir[1] * f]
                })
                .collect();
            let half = (0..4)
                .map(|j| (ri * 0.28 * (std::f32::consts::PI * (j as f32 + 0.5) / 4.0).sin()).max(0.5))
                .collect();
            out.push(Shape {
                spine,
                half,
                root: mul(p.centre, [0.6; 3]),
                tip: mul(fringe, [1.0 + rng.random_range(-0.1f32..=0.1); 3]),
                root_shade: 1.0,
                tilt: x / width_here.max(1e-3) * 0.6,
                roughness: 0.6,
                detail: 0.0,
            });
        }
    }
}

fn spine_length(s: &[[f32; 2]]) -> f32 {
    s.windows(2)
        .map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt())
        .sum()
}

/// The point `t` of the way along a spine by index, and the direction there.
fn along(spine: &[[f32; 2]], t: f32) -> ([f32; 2], [f32; 2]) {
    let last = spine.len() - 1;
    let f = t.clamp(0.0, 1.0) * last as f32;
    let i = (f.floor() as usize).min(last - 1);
    let u = f - i as f32;
    let (a, b) = (spine[i], spine[i + 1]);
    let d = normalize([b[0] - a[0], b[1] - a[1]]);
    ([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u], d)
}

fn normalize(v: [f32; 2]) -> [f32; 2] {
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l]
}

fn rotate(v: [f32; 2], a: f32) -> [f32; 2] {
    let (s, c) = a.sin_cos();
    [v[0] * c - v[1] * s, v[0] * s + v[1] * c]
}

fn vary(rng: &mut PortableRng, v: f32) -> [f32; 3] {
    let v = v.clamp(0.0, 1.0);
    let b = 1.0 + if v > 0.0 { rng.random_range(-v..=v) } else { 0.0 };
    let h = if v > 0.0 { rng.random_range(-v..=v) * 0.5 } else { 0.0 };
    [b * (1.0 + h), b, b * (1.0 - h)]
}

fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}
