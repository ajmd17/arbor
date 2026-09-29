//! Fern fronds for the blade atlas: one pinnate leaf laid flat in a cell, root at the
//! bottom edge and tip at the top, for a card to arch out of a clump.
//!
//! A frond is a stalk (the stipe) running on up the middle as the rachis, with pinnae
//! paired up both sides. Each pinna is a midrib carrying pinnules, small lobes angled
//! toward its tip. How far the pinnae are cut into lobes is `division`: at 0 a pinna is
//! one leaflet with a scalloped edge, as a sword fern's is; at 1 it is a row of separate
//! pinnules, as a lady fern's or bracken's is, and light shows through between them.
//!
//! The outline is the lance shape most ferns share: short pinnae at the bottom, the
//! longest a third of the way up, tapering to a point. The frond is seen face on, and its
//! two sides are tilted apart in the normal map, since a real frond is a shallow V along
//! its rachis and its halves catch the light differently.
//!
//! Everything is built from the strokes blades are, so it composites, lights and bleeds
//! exactly as they do.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::flowers::Shape;
use crate::seed::PortableRng;

/// Pixels kept clear of a cell's sides, beyond the atlas's own margin.
const EDGE_PAD: f32 = 6.0;
/// Points up the rachis.
const RACHIS_POINTS: usize = 32;
/// How far a frond's two halves are tilted apart in the normal map, in radians.
const HALF_TILT: f32 = 0.35;

/// The fronds drawn in one atlas cell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FrondParams {
    /// Fronds in the cell, side by side, each in its own share of the width. One fills
    /// the cell, which is what a card arching out of a fern wants.
    pub count: u32,
    /// Length of a frond as a fraction of the cell's height: shortest and longest.
    pub length: (f32, f32),
    /// Share of the frond that is bare stalk below the lowest pinnae.
    pub stipe: f32,
    /// Pinnae up each side.
    pub pinnae: u32,
    /// Reach of the longest pinna, as a share of half the frond's width.
    pub reach: f32,
    /// How far up the leafy part the longest pinnae are, 0 at the bottom, 1 at the tip.
    pub widest: f32,
    /// Length of the lowest pinnae against the longest. Some ferns taper to nothing at
    /// the base, bracken hardly at all.
    pub base_width: f32,
    /// Degrees a pinna stands off the rachis at the bottom of the frond and at its tip:
    /// near square at the bottom, swept toward the tip higher up.
    pub angle_deg: f32,
    pub tip_angle_deg: f32,
    /// How far a pinna bends toward the frond's tip along its length, in degrees.
    pub curve_deg: f32,
    /// Pinnules along each side of the longest pinna. Shorter pinnae carry fewer.
    pub lobes: u32,
    /// Degrees a pinnule stands off its pinna, swept toward the pinna's tip.
    pub lobe_angle_deg: f32,
    /// Width of a pinnule against the spacing between pinnules: 1 touches its
    /// neighbours, below leaves gaps.
    pub lobe_width: f32,
    /// How deeply pinnae are cut into pinnules, from a whole leaflet with a lobed edge at
    /// 0 to separate pinnules at 1.
    pub division: f32,
    /// Share of the space between one pinna and the next up the same side that is left
    /// open. Lady fern pinnae nearly touch; a sword fern's leaflets stand apart.
    pub gap: f32,
    /// Colour of a pinna, of the young growth at the tip, and of the stalk, sRGB.
    pub color: [f32; 3],
    pub tip_color: [f32; 3],
    pub rachis: [f32; 3],
    /// Spread of brightness and yellowing from pinna to pinna.
    pub color_variance: f32,
    /// Share of pinnae browned through, in `dead_color`, lowest first: a frond dies back
    /// from its base.
    pub dead: f32,
    pub dead_color: [f32; 3],
    pub roughness: f32,
}

impl Default for FrondParams {
    fn default() -> Self {
        Self {
            count: 1,
            length: (0.93, 0.97),
            stipe: 0.07,
            pinnae: 22,
            reach: 0.92,
            widest: 0.32,
            base_width: 0.45,
            angle_deg: 72.0,
            tip_angle_deg: 42.0,
            curve_deg: 12.0,
            lobes: 13,
            lobe_angle_deg: 42.0,
            lobe_width: 0.95,
            division: 0.85,
            gap: 0.1,
            color: [0.27, 0.45, 0.12],
            tip_color: [0.42, 0.55, 0.16],
            rachis: [0.40, 0.46, 0.20],
            color_variance: 0.1,
            dead: 0.0,
            dead_color: [0.47, 0.34, 0.17],
            roughness: 0.6,
        }
    }
}

/// Every frond in a cell `w` by `h` pixels, as shapes in the order they are painted,
/// standing on the bottom edge with `margin` pixels clear all round.
pub fn fronds(p: &FrondParams, rng: &mut PortableRng, w: f32, h: f32, margin: f32) -> Vec<Shape> {
    let count = p.count.clamp(1, 4);
    let slot = w / count as f32;
    let mut out = Vec::new();
    for i in 0..count {
        let centre = slot * (i as f32 + 0.5);
        out.extend(frond(p, rng, centre, slot, h, margin));
    }
    out
}

/// One frond standing at `x`, `slot` pixels wide.
fn frond(p: &FrondParams, rng: &mut PortableRng, x: f32, slot: f32, h: f32, margin: f32) -> Vec<Shape> {
    let (lo, hi) = (p.length.0.min(p.length.1), p.length.0.max(p.length.1));
    let length = (h - 2.0 * margin) * rng.random_range(lo..=hi.max(lo + 1e-4)).clamp(0.1, 0.97);
    let base = h - margin - 1.0;
    // A gentle S up the rachis, so the frond is not ruled straight.
    let (sway, phase) = (rng.random_range(0.004..0.012) * h, rng.random_range(0.0..std::f32::consts::TAU));
    let rachis: Vec<[f32; 2]> = (0..RACHIS_POINTS)
        .map(|k| {
            let t = k as f32 / (RACHIS_POINTS - 1) as f32;
            [x + sway * (t * 2.6 + phase).sin() - sway * phase.sin(), base - length * t]
        })
        .collect();

    let half_room = (slot * 0.5 - margin - EDGE_PAD).max(4.0);
    let pinnae = p.pinnae.clamp(2, 64);
    let stipe = p.stipe.clamp(0.0, 0.6);
    let lobes_max = p.lobes.clamp(2, 48) as f32;
    let division = p.division.clamp(0.0, 1.0);
    let widest = p.widest.clamp(0.05, 0.95);
    let base_width = p.base_width.clamp(0.0, 1.0);
    let dead_share = p.dead.clamp(0.0, 1.0);
    let tint = |rng: &mut PortableRng| vary(rng, p.color_variance);

    // Longest pinna first, so its lobes are known before the reach is fitted to the cell:
    // a swept pinna's lobes stick out past its tip.
    let reach_px = half_room * p.reach.clamp(0.1, 1.0);
    let lobe_angle = p.lobe_angle_deg.clamp(0.0, 85.0).to_radians();
    // Up the rachis from one pinna to the next on the same side.
    let pinna_gap = length * (1.0 - stipe) / pinnae as f32;
    let open = p.gap.clamp(0.0, 0.95);

    let mut blades: Vec<Shape> = Vec::new();
    let mut ribs: Vec<Shape> = Vec::new();
    for side in [-1.0f32, 1.0] {
        for i in 0..pinnae {
            // The two sides alternate a little rather than pairing exactly.
            let offset = if side > 0.0 { 0.35 } else { 0.0 };
            let u = ((i as f32 + offset + rng.random_range(-0.1..=0.1)) / pinnae as f32).clamp(0.0, 0.999);
            let t = stipe + (1.0 - stipe) * u;
            let (at, dir) = along(&rachis, t);
            // Lance-shaped: from `base_width` at the bottom up to 1 at `widest`, then down
            // to a point.
            let shape = if u < widest {
                base_width + (1.0 - base_width) * smooth(u / widest).sqrt()
            } else {
                let v = (u - widest) / (1.0 - widest);
                (1.0 - v).powf(0.8)
            };
            let angle = (p.angle_deg + (p.tip_angle_deg - p.angle_deg) * u).clamp(5.0, 90.0).to_radians();
            // Along the pinna: off the rachis to its side, swept up toward the tip.
            let across = [-dir[1] * side, dir[0] * side];
            let pdir = normalize([dir[0] * angle.cos() + across[0] * angle.sin(), dir[1] * angle.cos() + across[1] * angle.sin()]);
            // Its length, cut so the pinna and the lobes at its tip stay in the cell.
            let reach_side = reach_px * shape * rng.random_range(0.92..=1.04);
            let len = (reach_side / angle.sin().max(0.3)).max(2.0);
            if len < 3.0 {
                continue;
            }
            let dead = u < dead_share * rng.random_range(0.8..=1.2);
            let c = if dead {
                mul(p.dead_color, tint(rng))
            } else {
                let young = smooth((u - 0.6) / 0.4);
                mul(mix(p.color, p.tip_color, young), tint(rng))
            };
            let curve = p.curve_deg.to_radians() * side;
            let steps = 10usize;
            let axis: Vec<[f32; 2]> = (0..=steps)
                .map(|k| {
                    let s = k as f32 / steps as f32;
                    // Bent toward the frond's tip: the pinna's direction turns toward the
                    // rachis's as it goes.
                    let a = -curve * s * s;
                    let d = rotate(pdir, a);
                    [at[0] + d[0] * len * s, at[1] + d[1] * len * s]
                })
                .collect();
            let axis = clamp_x(axis, x - slot * 0.5 + margin + EDGE_PAD, x + slot * 0.5 - margin - EDGE_PAD);

            // Pinnules up both edges of the pinna, shrinking toward its tip.
            let n_lobes = ((lobes_max * (len / (reach_px / 0.9).max(1.0))).round() as usize).clamp(2, 48);
            let spacing = len / n_lobes as f32;
            // A pinnule reaches no further out from its pinna than halfway to the next
            // pinna, less the gap, so neighbouring pinnae do not run into one another.
            let room = 0.5 * pinna_gap * angle.sin() * (1.0 - open);
            let lobe_max = (spacing * 1.7).min(room / lobe_angle.sin().max(0.3));
            let tilt = HALF_TILT * side;
            for k in 0..n_lobes {
                let s = (k as f32 + 0.5) / n_lobes as f32;
                let (pa, pd) = along(&axis, s);
                let lobe_len = (lobe_max * (1.0 - 0.8 * s.powf(1.6)) * (0.6 + 0.4 * shape)).max(1.5);
                // Longer than it is wide, however wide the pinnules are asked to be.
                let lobe_w = (spacing * 0.5 * p.lobe_width.clamp(0.2, 1.6)).min(lobe_len * 0.45);
                for lobe_side in [-1.0f32, 1.0] {
                    let pacr = [-pd[1] * lobe_side, pd[0] * lobe_side];
                    let ld = normalize([
                        pd[0] * lobe_angle.cos() + pacr[0] * lobe_angle.sin(),
                        pd[1] * lobe_angle.cos() + pacr[1] * lobe_angle.sin(),
                    ]);
                    let pts: Vec<[f32; 2]> = (0..5)
                        .map(|j| {
                            let f = j as f32 / 4.0 * lobe_len;
                            [pa[0] + ld[0] * f, pa[1] + ld[1] * f]
                        })
                        .collect();
                    // Blunt where it joins the pinna, broadest a third out, to a point.
                    let half = (0..5)
                        .map(|j| {
                            let f = j as f32 / 4.0;
                            (lobe_w * (0.8 + 0.7 * f - 1.45 * f * f).max(0.05)).max(0.35)
                        })
                        .collect();
                    let shade = 1.0 + rng.random_range(-0.06f32..=0.06);
                    blades.push(Shape {
                        spine: pts,
                        half,
                        root: mul(c, [0.86 * shade; 3]),
                        tip: mul(c, [1.04 * shade; 3]),
                        root_shade: 1.0,
                        tilt: tilt + 0.25 * lobe_side,
                        roughness: p.roughness,
                        detail: 0.35,
                    });
                }
            }
            // The body of the pinna, as wide as its lobes are uncut, so an undivided
            // pinna is one leaflet with a scalloped edge.
            if division < 0.999 {
                let body = axis.len();
                let body_half: Vec<f32> = (0..body)
                    .map(|k| {
                        let s = k as f32 / (body - 1) as f32;
                        let lobe_len = lobe_max * (1.0 - 0.8 * s.powf(1.6)) * (0.6 + 0.4 * shape);
                        (lobe_len * lobe_angle.sin() * (1.0 - division) * 0.95).max(0.6)
                    })
                    .collect();
                blades.push(Shape {
                    spine: axis.clone(),
                    half: body_half,
                    root: mul(c, [0.9; 3]),
                    tip: c,
                    root_shade: 1.0,
                    tilt,
                    roughness: p.roughness,
                    detail: 0.6,
                });
            }
            // Its midrib, thin and a little paler.
            let rib_w = (h * 0.0016).max(0.55);
            ribs.push(Shape {
                half: (0..axis.len()).map(|k| rib_w * (1.0 - 0.6 * k as f32 / (axis.len() - 1) as f32)).collect(),
                spine: axis,
                root: mix(c, p.rachis, 0.5),
                tip: mix(c, p.rachis, 0.3),
                root_shade: 1.0,
                tilt,
                roughness: p.roughness + 0.05,
                detail: 0.0,
            });
        }
    }

    let rachis_w = (h * 0.0055).max(1.0);
    let stalk = Shape {
        half: (0..RACHIS_POINTS)
            .map(|k| rachis_w * (1.0 - 0.75 * k as f32 / (RACHIS_POINTS - 1) as f32))
            .collect(),
        spine: rachis,
        root: mul(p.rachis, [0.7; 3]),
        tip: p.rachis,
        root_shade: 0.7,
        tilt: 0.0,
        roughness: p.roughness + 0.1,
        detail: 0.2,
    };
    let mut out = blades;
    out.extend(ribs);
    out.push(stalk);
    out
}

/// Keeps every point between `lo` and `hi`, squashing the spine toward its first point
/// when it would cross either.
fn clamp_x(mut pts: Vec<[f32; 2]>, lo: f32, hi: f32) -> Vec<[f32; 2]> {
    let root = pts[0];
    let mut k: f32 = 1.0;
    for p in &pts {
        let dx = p[0] - root[0];
        if dx > 1e-3 && root[0] + dx > hi {
            k = k.min(((hi - root[0]) / dx).max(0.0));
        } else if dx < -1e-3 && root[0] + dx < lo {
            k = k.min(((lo - root[0]) / dx).max(0.0));
        }
    }
    if k < 1.0 {
        for p in &mut pts {
            p[0] = root[0] + (p[0] - root[0]) * k;
            p[1] = root[1] + (p[1] - root[1]) * k;
        }
    }
    pts
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
    let y = if v > 0.0 { rng.random_range(-v..=v) } else { 0.0 };
    [b * (1.0 + 0.4 * y), b * (1.0 + 0.1 * y), b * (1.0 - 0.6 * y)]
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

#[cfg(test)]
mod tests {
    use crate::blades::{bake_blades, BladeAtlasParams, BladeCell};

    use super::*;

    fn cell(frond: FrondParams) -> BladeAtlasParams {
        BladeAtlasParams {
            cell_height: 256,
            cell_aspect: 0.5,
            cells: vec![BladeCell { count: 0, frond: Some(frond), ..BladeCell::default() }],
        }
    }

    #[test]
    fn a_frond_stays_inside_its_cell() {
        for division in [0.0, 0.5, 1.0] {
            for reach in [0.5, 1.0] {
                let img = bake_blades(&cell(FrondParams { division, reach, curve_deg: 40.0, ..FrondParams::default() })).maps.albedo;
                for y in 0..img.height {
                    assert_eq!(img.at(0, y)[3], 0, "left edge, division {division} reach {reach}");
                    assert_eq!(img.at(img.width - 1, y)[3], 0, "right edge, division {division} reach {reach}");
                }
                for x in 0..img.width {
                    assert_eq!(img.at(x, 0)[3], 0, "top edge, division {division} reach {reach}");
                }
            }
        }
    }

    #[test]
    fn a_frond_is_widest_below_its_middle_and_stands_on_the_bottom() {
        let img = bake_blades(&cell(FrondParams::default())).maps.albedo;
        let covered = |y: u32| (0..img.width).filter(|&x| img.at(x, y)[3] > 128).count();
        let h = img.height;
        let lower = covered(h * 6 / 10);
        let upper = covered(h * 2 / 10);
        assert!(lower > upper, "widest low down: {lower} against {upper} near the tip");
        assert!(covered(h - 6) > 0, "the stalk reaches the ground");
    }

    #[test]
    fn cutting_the_pinnae_lets_light_through() {
        // A lady fern's divided pinnae cover less of the cell than a sword fern's whole ones.
        let whole = bake_blades(&cell(FrondParams { division: 0.0, ..FrondParams::default() })).coverage[0];
        let cut = bake_blades(&cell(FrondParams { division: 1.0, lobe_width: 0.7, ..FrondParams::default() })).coverage[0];
        assert!(cut < whole * 0.9, "cut {cut} against whole {whole}");
        assert!(cut > 0.05);
    }

    #[test]
    fn browned_pinnae_are_browner() {
        let mean = |f: FrondParams| {
            let img = bake_blades(&cell(f)).maps.albedo;
            let (mut r, mut g, mut n) = (0.0f64, 0.0f64, 0.0f64);
            for y in 0..img.height {
                for x in 0..img.width {
                    let p = img.at(x, y);
                    if p[3] > 200 {
                        r += p[0] as f64;
                        g += p[1] as f64;
                        n += 1.0;
                    }
                }
            }
            (r / n, g / n)
        };
        let (gr, gg) = mean(FrondParams::default());
        let (br, bg) = mean(FrondParams { dead: 0.6, ..FrondParams::default() });
        assert!(br - bg > gr - gg + 15.0, "green ({gr:.0}, {gg:.0}) browned ({br:.0}, {bg:.0})");
    }
}
