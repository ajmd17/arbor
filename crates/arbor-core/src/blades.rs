//! Drawing a grass-blade atlas from nothing.
//!
//! Ground cover is drawn as alpha-tested cards, each showing a spray of blades, for the
//! same reason a canopy is: a blade is thinner than a pixel a few metres off, so real
//! blade triangles alias and sparkle, where a card's texture is filtered and its
//! coverage kept by the mip chain. This bakes the texture those cards read.
//!
//! Every blade is drawn procedurally, so there is no source art to license and every
//! property of it is a number. A blade is a stroke up a curved spine, tapering to a
//! point, folded along its midrib; seed heads are a thin stalk with grains paired up
//! its top. Blades are painted back to front with the ones behind darkened, as leaves
//! are in a cluster, and the normal and roughness maps are composited through the same
//! coverage so all three agree pixel for pixel.
//!
//! The atlas is one row of cells, each its own kind of blade — lush, dry, seeded — and
//! a card picks the cell it shows. Cells are taller than they are wide, standing on
//! their bottom edge: that edge is the ground.

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::cluster::{bleed_color_outward, BakedMaps, Bitmap};
use crate::ferns::{fronds, FrondParams};
use crate::flowers::{flower, FlowerParams};
use crate::seed::PortableRng;

/// Most cells one atlas may hold, so a preset cannot ask for a sheet no GPU takes.
const MAX_CELLS: usize = 16;
/// Pixels left clear round every cell, so filtering never pulls one cell into the next.
const MARGIN: f32 = 3.0;
/// Points along a blade's spine. Enough that the curve reads as a curve at any cell
/// size this is used at.
const SPINE_POINTS: usize = 24;
/// Decorrelates one cell's random stream from the next.
const CELL_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;
/// The fold along a blade's midrib, in radians either side of it. A grass blade is a
/// shallow V, and the two halves catching light differently is most of what makes it
/// read as a blade rather than a green line.
const MIDRIB_FOLD: f32 = 0.55;
/// Furthest a blade may turn off vertical anywhere along it, in radians.
const MAX_TURN: f32 = 1.75;

/// One cell of the atlas: a spray of blades of one kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BladeCell {
    /// Drawn by dry tufts rather than lush ones. See `CoverColour`.
    pub dry: bool,
    /// How often a card picks this cell against the other cells of its kind.
    pub weight: f32,
    /// Blades in the cell.
    pub count: u32,
    /// Stalks carrying a seed head, drawn over the blades.
    pub heads: u32,
    /// Length of a blade, as a fraction of the cell's height: shortest and longest.
    pub length: (f32, f32),
    /// Width of a blade at its root, as a fraction of the cell's height.
    pub width: f32,
    /// How far a blade leans off vertical at its root, either way, in degrees.
    pub lean_deg: f32,
    /// How much further it bends by its tip, in degrees, toward the side it already
    /// leans: the droop that makes long grass arch over.
    pub curve_deg: f32,
    /// How that bend is spread along the blade. A grass blade stands stiff near its
    /// sheath and gives way toward the tip, so the bend is bunched there: 1 is an even
    /// arc, 2 or 3 keeps the lower blade straight and lets the top flop.
    pub droop_power: f32,
    /// Random bending along the blade, in degrees: no blade is a clean arc.
    pub wiggle_deg: f32,
    /// Share of blades folded over partway, as long leaves that have given way do.
    pub kink: f32,
    /// How far a blade's lean follows where its root is: 1 fans every blade away from
    /// the middle, 0 leans each its own way. A little fan keeps long blades from
    /// crossing; a lot makes every card a fountain.
    pub fan: f32,
    /// Share of the cell's width the roots are spread over, centred.
    pub root_spread: f32,
    /// Colour at the root and at the tip, sRGB.
    pub root: [f32; 3],
    pub tip: [f32; 3],
    /// Spread of brightness and of yellowing from blade to blade.
    pub color_variance: f32,
    /// Share of blades that are dead through, in `dead_color`: every sward carries old
    /// leaves among the new, and without them grass reads as plastic.
    pub dead: f32,
    pub dead_color: [f32; 3],
    /// Unevenness of colour along a blade, as a share of its brightness.
    pub mottle: f32,
    /// Share of blades whose tip has dried, and how much of the blade it reaches.
    pub tip_burn: f32,
    pub burn_length: f32,
    pub burn: [f32; 3],
    /// Darkening of the blades furthest back, the one depth cue a flat card keeps.
    pub depth_shade: f32,
    /// Darkening at the root: the base of a tuft is in its own shade.
    pub root_shade: f32,
    /// Length of a seed head as a fraction of the cell's height, its width against
    /// its length, and its colour.
    pub head_length: f32,
    pub head_width: f32,
    /// From a tight spike, grains pressed to the stalk, at 0, to a loose feathery
    /// panicle of small grains on splayed stalklets at 1.
    pub head_spread: f32,
    pub head: [f32; 3],
    pub roughness: f32,
    /// Wildflowers standing among the blades, if this is a flowering cell. A cell with
    /// flowers is drawn by flowering tufts rather than lush or dry ones.
    pub flowers: Option<FlowerParams>,
    /// A fern frond laid over the cell, in front of any blades. A fern's cells are
    /// usually fronds alone, with `count` at 0.
    pub frond: Option<FrondParams>,
    pub seed: u64,
}

impl Default for BladeCell {
    fn default() -> Self {
        Self {
            dry: false,
            weight: 1.0,
            count: 36,
            heads: 0,
            length: (0.5, 0.95),
            width: 0.016,
            lean_deg: 14.0,
            curve_deg: 22.0,
            droop_power: 2.2,
            wiggle_deg: 6.0,
            kink: 0.06,
            fan: 0.35,
            root_spread: 0.8,
            root: [0.20, 0.26, 0.10],
            tip: [0.32, 0.40, 0.15],
            color_variance: 0.2,
            dead: 0.12,
            dead_color: [0.45, 0.40, 0.27],
            mottle: 0.12,
            tip_burn: 0.15,
            burn_length: 0.2,
            burn: [0.58, 0.52, 0.29],
            depth_shade: 0.45,
            root_shade: 0.75,
            head_length: 0.16,
            head_width: 0.22,
            head_spread: 0.5,
            head: [0.64, 0.57, 0.36],
            roughness: 0.62,
            flowers: None,
            frond: None,
            seed: 11,
        }
    }
}

/// The whole atlas.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BladeAtlasParams {
    /// Height of one cell in pixels.
    pub cell_height: u32,
    /// Width of a cell against its height. A card shows one cell with square pixels,
    /// so this is also the shape of the card.
    pub cell_aspect: f32,
    pub cells: Vec<BladeCell>,
}

impl Default for BladeAtlasParams {
    fn default() -> Self {
        Self {
            cell_height: 512,
            cell_aspect: 0.5,
            cells: vec![BladeCell::default()],
        }
    }
}

impl BladeAtlasParams {
    /// The cells that are actually baked.
    pub fn cells(&self) -> &[BladeCell] {
        &self.cells[..self.cells.len().min(MAX_CELLS)]
    }

    pub fn cell_size(&self) -> (u32, u32) {
        let h = self.cell_height.clamp(32, 4096);
        let w = ((h as f32 * self.cell_aspect.clamp(0.1, 4.0)).round() as u32).max(8);
        (w, h)
    }
}

/// A baked atlas, and the share of each cell the blades cover.
pub struct BakedBlades {
    pub maps: BakedMaps,
    /// Mean alpha of each cell: what decides how many cards a clump needs to look full.
    pub coverage: Vec<f32>,
}

/// Bakes every cell, left to right, into one sheet.
pub fn bake_blades(p: &BladeAtlasParams) -> BakedBlades {
    let (cw, ch) = p.cell_size();
    let cells = p.cells();
    let n = cells.len().max(1) as u32;
    let mut albedo = Bitmap::new(cw * n, ch);
    let mut normal = Bitmap::new(cw * n, ch);
    let mut rough = Bitmap::new(cw * n, ch);
    let mut coverage = Vec::with_capacity(cells.len());
    for (i, cell) in cells.iter().enumerate() {
        let seed = cell.seed ^ (i as u64).wrapping_mul(CELL_STRIDE);
        let baked = bake_cell(cell, cw, ch, seed);
        coverage.push(baked.albedo.mean_alpha());
        let ox = i as u32 * cw;
        for y in 0..ch {
            for x in 0..cw {
                albedo.put(ox + x, y, baked.albedo.at(x, y));
                normal.put(ox + x, y, baked.normal.at(x, y));
                rough.put(ox + x, y, baked.roughness.at(x, y));
            }
        }
    }
    BakedBlades {
        maps: BakedMaps {
            albedo,
            normal: Some(normal),
            roughness: Some(rough),
        },
        coverage,
    }
}

struct Cell {
    albedo: Bitmap,
    normal: Bitmap,
    roughness: Bitmap,
}

/// One stroke up a spine: a blade, a stalk, or a single grain of a seed head.
struct Stroke {
    spine: Vec<[f32; 2]>,
    /// Half its width at each point of the spine, in pixels.
    half: Vec<f32>,
    root: [f32; 3],
    tip: [f32; 3],
    /// From how far along the tip colour gives way to `burn`; above 1 for none.
    burn_from: f32,
    burn: [f32; 3],
    /// Brightness multiplier: its depth shade and its own variation.
    value: f32,
    /// How dark its root is, 1 for not at all.
    root_shade: f32,
    /// How far it is turned about its own length, in radians, which tilts its normal.
    tilt: f32,
    roughness: f32,
    /// Brightness wobble along it: amount, and two frequencies and phases.
    mottle: (f32, [f32; 4]),
    /// How much of a blade's midrib, veins and fold it is drawn with, 0 to 1.
    detail: f32,
}

/// Premultiplied accumulation of every map in one cell.
struct Canvas {
    w: u32,
    h: u32,
    rgb: Vec<[f32; 3]>,
    a: Vec<f32>,
    n: Vec<[f32; 3]>,
    r: Vec<f32>,
}

fn bake_cell(c: &BladeCell, w: u32, h: u32, seed: u64) -> Cell {
    let mut rng = PortableRng::seed_from_u64(seed);
    let hf = h as f32;
    let wf = w as f32;
    let mut strokes: Vec<(f32, Stroke)> = Vec::new();

    let spread = c.root_spread.clamp(0.0, 1.0);
    let (lo, hi) = (c.length.0.min(c.length.1), c.length.0.max(c.length.1));
    let jitter = |rng: &mut PortableRng, v: f32| if v > 0.0 { rng.random_range(-v..=v) } else { 0.0 };
    let tint = |rng: &mut PortableRng, v: f32| {
        // Brightness and hue, drawn apart: a blade can be pale without being dry, and
        // grass runs from blue-green to yellow-green blade by blade.
        let bright = 1.0 + jitter(rng, v);
        let yellow = jitter(rng, v);
        [bright * (1.0 + 0.45 * yellow), bright * (1.0 + 0.1 * yellow), bright * (1.0 - 0.7 * yellow)]
    };
    let mottle = |rng: &mut PortableRng| {
        (
            c.mottle.max(0.0),
            [
                rng.random_range(4.0..9.0),
                rng.random_range(0.0..std::f32::consts::TAU),
                rng.random_range(13.0..29.0),
                rng.random_range(0.0..std::f32::consts::TAU),
            ],
        )
    };

    let total = c.count.min(2048) + c.heads.min(256);
    for i in 0..total {
        let is_head = i >= c.count.min(2048);
        let depth: f32 = rng.random();
        let root_x = wf * (0.5 + (rng.random::<f32>() - 0.5) * spread);
        // Lengths weighted toward the short end: a sward is mostly low leaves with a
        // few long ones standing out of it, not an even crop.
        let u: f32 = rng.random();
        let length = hf * (lo + (hi.max(lo + 1e-4) - lo) * u.powf(1.4)).clamp(0.02, 1.0);
        // Each blade leans its own way, pulled away from the middle by `fan` as far as
        // its root is off it, so long blades mostly splay rather than cross. It droops
        // the way it leans, and further the longer it is.
        let off = ((root_x / wf - 0.5) / (0.5 * spread).max(1e-3)).clamp(-1.0, 1.0);
        let fan = c.fan.clamp(0.0, 1.0);
        let lean_u = (fan * off + (1.0 - fan) * rng.random_range(-1.0f32..=1.0)).clamp(-1.0, 1.0);
        let lean = (c.lean_deg * lean_u * rng.random_range(0.7..=1.2)).to_radians();
        let side = if lean >= 0.0 { 1.0 } else { -1.0 };
        let curve = (c.curve_deg * (0.3 + 0.7 * rng.random::<f32>()) * length / hf).to_radians() * side;
        let bend = Bend {
            lean,
            curve,
            power: c.droop_power.clamp(0.5, 5.0),
            wiggle: (
                c.wiggle_deg.to_radians() * rng.random::<f32>(),
                rng.random_range(0.0..std::f32::consts::TAU),
            ),
            kink: (!is_head && rng.random::<f32>() < c.kink.clamp(0.0, 1.0))
                .then(|| (rng.random_range(0.6..0.85), rng.random_range(0.5..1.0) * side)),
        };
        let spine = fit_inside(bend.spine(root_x, hf - MARGIN, length), wf);
        let t = tint(&mut rng, c.color_variance.clamp(0.0, 1.0));
        let value = 1.0 - c.depth_shade.clamp(0.0, 1.0) * 0.5 * (1.0 - depth);
        let tilt = jitter(&mut rng, 0.45);

        if is_head {
            // A thin stalk in the head's colour, up to the head, and grains paired up
            // the top of it.
            let stalk_w = (c.width * hf * 0.22).max(0.7);
            let head_len = (c.head_length * hf).min(length * 0.6);
            let head_from = 1.0 - head_len / length;
            let half = spine
                .iter()
                .enumerate()
                .map(|(k, _)| stalk_w * (1.0 - 0.3 * k as f32 / (SPINE_POINTS - 1) as f32))
                .collect();
            let stalk_col = mix3(c.tip, c.head, 0.5);
            strokes.push((
                depth,
                Stroke {
                    spine: spine.clone(),
                    half,
                    root: mul3(c.root, t),
                    tip: mul3(stalk_col, t),
                    burn_from: 2.0,
                    burn: c.burn,
                    value,
                    root_shade: c.root_shade,
                    tilt,
                    roughness: c.roughness + 0.08,
                    mottle: mottle(&mut rng),
                    detail: 1.0,
                },
            ));
            let loose = c.head_spread.clamp(0.0, 1.0);
            let grain_len = (head_len * (0.16 - 0.08 * loose)).max(2.5);
            let grain_w = (grain_len * c.head_width.clamp(0.05, 1.0)).max(0.7);
            let grains = ((head_len / (grain_len * (0.45 - 0.2 * loose))).ceil() as usize).clamp(2, 96);
            for g in 0..grains {
                let step = (g as f32 + 0.5 + jitter(&mut rng, 0.4 * loose)) / grains as f32;
                let along = head_from + (1.0 - head_from) * step.clamp(0.0, 1.0);
                let (at, dir) = along_spine(&spine, along);
                let across = [-dir[1], dir[0]];
                // Alternate sides, splayed off the stalk, the top ones tighter in. A loose
                // panicle holds its grains out on stalklets, widest at the bottom.
                let s = if g % 2 == 0 { 1.0 } else { -1.0 };
                let low = 1.0 - (along - head_from) / (1.0 - head_from).max(1e-3);
                let splay = (0.15 + 0.3 * low + 0.5 * loose * low) * s;
                let (sn, cs) = (splay + jitter(&mut rng, 0.15 + 0.2 * loose)).sin_cos();
                let gdir = [dir[0] * cs + across[0] * sn, dir[1] * cs + across[1] * sn];
                let reach = grain_w * 0.3 + loose * grain_len * 1.5 * low * rng.random::<f32>();
                let start = [at[0] + across[0] * s * reach, at[1] + across[1] * s * reach];
                let pts: Vec<[f32; 2]> = (0..5)
                    .map(|k| {
                        let f = k as f32 / 4.0 * grain_len;
                        [start[0] + gdir[0] * f, start[1] + gdir[1] * f]
                    })
                    .collect();
                let half = (0..5)
                    .map(|k| grain_w * 0.5 * (std::f32::consts::PI * (k as f32 + 0.5) / 5.0).sin().powf(0.6))
                    .collect();
                let g_tint = tint(&mut rng, c.color_variance.clamp(0.0, 1.0) * 0.5);
                strokes.push((
                    depth + 1e-4,
                    Stroke {
                        spine: fit_inside(pts, wf),
                        half,
                        root: mul3(mul3(c.head, t), g_tint),
                        tip: mul3(mul3(c.head, t), g_tint),
                        burn_from: 2.0,
                        burn: c.burn,
                        value: value * rng.random_range(0.8..=1.05),
                        root_shade: 1.0,
                        tilt: jitter(&mut rng, 0.8),
                        roughness: c.roughness + 0.1,
                        mottle: (0.0, [0.0; 4]),
                        detail: 0.3,
                    },
                ));
            }
            continue;
        }

        // Short blades are narrower too: young leaves, and the sheaths of old ones.
        let young = 0.6 + 0.4 * length / (hi * hf).max(1.0);
        let w0 = (c.width * hf * rng.random_range(0.6..=1.2) * young).max(0.9) * 0.5;
        let half = (0..SPINE_POINTS)
            .map(|k| {
                let u = k as f32 / (SPINE_POINTS - 1) as f32;
                // Tapering the whole way from the sheath, faster toward the point.
                w0 * (1.0 - u).powf(0.6) * (1.0 - 0.15 * u)
            })
            .collect();
        let dead = rng.random::<f32>() < c.dead.clamp(0.0, 1.0);
        let (root_col, tip_col) = if dead {
            let d = mul3(c.dead_color, tint(&mut rng, 0.25));
            (mul3(d, [0.8, 0.8, 0.8]), d)
        } else {
            (mul3(c.root, t), mul3(c.tip, t))
        };
        let burnt = !dead && rng.random::<f32>() < c.tip_burn.clamp(0.0, 1.0);
        let burn_from = if burnt {
            1.0 - c.burn_length.clamp(0.0, 1.0) * rng.random_range(0.5..=1.2)
        } else {
            2.0
        };
        strokes.push((
            // Dead leaves lie low in the sward, under the living ones.
            if dead { depth * 0.5 } else { depth },
            Stroke {
                spine,
                half,
                root: root_col,
                tip: tip_col,
                burn_from,
                burn: mul3(c.burn, t),
                value: if dead { value * 0.85 } else { value },
                root_shade: c.root_shade,
                tilt,
                roughness: c.roughness + if dead { 0.12 } else { jitter(&mut rng, 0.06) },
                mottle: mottle(&mut rng),
                detail: 1.0,
            },
        ));
    }

    if let Some(f) = &c.flowers {
        let wiggle = c.wiggle_deg.to_radians();
        let count = f.count.min(64);
        for i in 0..count {
            // Toward the front of the grass: a flower stands out of the sward.
            let depth = rng.random_range(0.35f32..1.0);
            let pad = (f.size * hf * 0.6).min(wf * 0.3);
            // Spread across the cell, one to each share of it, so two flowers never
            // stand on one spot and read as one.
            let slot = (i as f32 + rng.random_range(0.15f32..0.85)) / count.max(1) as f32;
            let root_x = pad + slot * (wf - 2.0 * pad).max(1.0);
            let phase = rng.random_range(0.0..std::f32::consts::TAU);
            let stem = |x: f32, lean: f32, length: f32, w: f32| {
                let bend = Bend { lean, curve: lean * 0.5, power: 2.0, wiggle: (wiggle * 0.5, phase), kink: None };
                fit_inside(bend.spine(x, hf - MARGIN, length), w)
            };
            let value = 1.0 - c.depth_shade.clamp(0.0, 1.0) * 0.35 * (1.0 - depth);
            let mut shapes = flower(f, &mut rng, root_x, wf, hf, &stem);
            // The stem is kept inside the cell as it is built; the head is not, so the
            // whole flower is slid back in if its head overhangs an edge.
            let pad = MARGIN + 2.0;
            let (lo, hi) = shapes.iter().flat_map(|sh| sh.spine.iter().zip(&sh.half)).fold(
                (f32::MAX, f32::MIN),
                |(lo, hi), (pt, r)| (lo.min(pt[0] - r - 1.0), hi.max(pt[0] + r + 1.0)),
            );
            let shift = (pad - lo).max(0.0) - (hi - (wf - pad)).max(0.0);
            let top = shapes
                .iter()
                .flat_map(|sh| sh.spine.iter().zip(&sh.half))
                .map(|(pt, r)| pt[1] - r - 1.0)
                .fold(f32::MAX, f32::min);
            let drop = (pad - top).max(0.0);
            for sh in &mut shapes {
                for pt in &mut sh.spine {
                    pt[0] += shift;
                    pt[1] += drop;
                }
            }
            for (k, shape) in shapes.into_iter().enumerate() {
                strokes.push((
                    depth + k as f32 * 1e-6,
                    Stroke {
                        spine: shape.spine,
                        half: shape.half,
                        root: shape.root,
                        tip: shape.tip,
                        burn_from: 2.0,
                        burn: shape.tip,
                        value,
                        root_shade: shape.root_shade,
                        tilt: shape.tilt,
                        roughness: shape.roughness,
                        mottle: (0.0, [0.0; 4]),
                        detail: shape.detail,
                    },
                ));
            }
        }
    }

    if let Some(f) = &c.frond {
        // In front of everything else in the cell, painted in the order they come.
        for (k, shape) in fronds(f, &mut rng, wf, hf, MARGIN).into_iter().enumerate() {
            strokes.push((
                2.0 + k as f32 * 1e-6,
                Stroke {
                    spine: shape.spine,
                    half: shape.half,
                    root: shape.root,
                    tip: shape.tip,
                    burn_from: 2.0,
                    burn: shape.tip,
                    value: 1.0,
                    root_shade: shape.root_shade,
                    tilt: shape.tilt,
                    roughness: shape.roughness,
                    mottle: (0.0, [0.0; 4]),
                    detail: shape.detail,
                },
            ));
        }
    }

    // Back to front.
    strokes.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut canvas = Canvas::new(w, h);
    for (_, s) in &strokes {
        canvas.draw(s);
    }
    canvas.finish()
}

/// How a blade's spine turns along its length.
struct Bend {
    /// Off vertical at the root, in radians.
    lean: f32,
    /// Further turn by the tip, and how it is bunched toward the tip.
    curve: f32,
    power: f32,
    /// Amplitude and phase of a slow random bending.
    wiggle: (f32, f32),
    /// Where along the blade it folds over, and by how much, in radians.
    kink: Option<(f32, f32)>,
}

impl Bend {
    fn angle(&self, u: f32) -> f32 {
        let mut a = self.lean + self.curve * u.powf(self.power);
        a += self.wiggle.0 * (u * 7.0 + self.wiggle.1).sin() * u;
        if let Some((at, by)) = self.kink {
            // A fold is sharp but not a corner: over a few percent of the blade.
            a += by * smooth((u - at) / 0.06);
        }
        // A blade may hang over, but not loop back up: past a little beyond level it
        // would run across the whole card as a straight line.
        a.clamp(-MAX_TURN, MAX_TURN)
    }

    /// The spine, `length` pixels up from `(x, y)`.
    fn spine(&self, x: f32, y: f32, length: f32) -> Vec<[f32; 2]> {
        let steps = SPINE_POINTS - 1;
        let ds = length / steps as f32;
        let mut out = Vec::with_capacity(SPINE_POINTS);
        let mut p = [x, y];
        out.push(p);
        for k in 0..steps {
            // Midpoint angle of the step, so the curve does not drift with the step count.
            let a = self.angle((k as f32 + 0.5) / steps as f32);
            p = [p[0] + a.sin() * ds, p[1] - a.cos() * ds];
            out.push(p);
        }
        out
    }
}

/// Slides a spine sideways, and if it has to squashes it, so it stays clear of the
/// cell's edges. A blade cut off by the edge draws a straight line across the card.
fn fit_inside(mut pts: Vec<[f32; 2]>, w: f32) -> Vec<[f32; 2]> {
    let pad = MARGIN + 3.0;
    let (lo, hi) = pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    let room = w - 2.0 * pad;
    if hi - lo > room {
        // Too wide to fit however it is placed: pull it in toward its own root.
        let root = pts[0][0];
        let k = room / (hi - lo);
        for p in &mut pts {
            p[0] = root + (p[0] - root) * k;
        }
    }
    let (lo, hi) = pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    let shift = (pad - lo).max(0.0) - (hi - (w - pad)).max(0.0);
    let top = pts.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let squash = if top < pad {
        let base = pts[0][1];
        (base - pad) / (base - top).max(1e-3)
    } else {
        1.0
    };
    let base = pts[0][1];
    for p in &mut pts {
        p[0] += shift;
        p[1] = base + (p[1] - base) * squash;
    }
    pts
}

/// The point `t` of the way along a spine by index, and the spine's direction there.
fn along_spine(spine: &[[f32; 2]], t: f32) -> ([f32; 2], [f32; 2]) {
    let last = spine.len() - 1;
    let f = t.clamp(0.0, 1.0) * last as f32;
    let i = (f.floor() as usize).min(last - 1);
    let u = f - i as f32;
    let (a, b) = (spine[i], spine[i + 1]);
    let d = [b[0] - a[0], b[1] - a[1]];
    let len = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-6);
    ([a[0] + d[0] * u, a[1] + d[1] * u], [d[0] / len, d[1] / len])
}

impl Canvas {
    fn new(w: u32, h: u32) -> Self {
        let n = (w * h) as usize;
        Self {
            w,
            h,
            rgb: vec![[0.0; 3]; n],
            a: vec![0.0; n],
            n: vec![[0.0; 3]; n],
            r: vec![0.0; n],
        }
    }

    /// Paints one stroke over what is there.
    ///
    /// Each pixel near the stroke takes the segment of the spine it is nearest the edge
    /// of, which gives how far along the stroke it is and how far across. Worked
    /// segment by segment into a scratch box, so a pixel is only visited by the
    /// segments that could reach it, and composited once.
    fn draw(&mut self, s: &Stroke) {
        let reach = s.half.iter().copied().fold(0.0, f32::max) + 1.5;
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &s.spine {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        let bx0 = ((x0 - reach).floor().max(0.0)) as i32;
        let by0 = ((y0 - reach).floor().max(0.0)) as i32;
        let bx1 = ((x1 + reach).ceil() as i32).min(self.w as i32 - 1);
        let by1 = ((y1 + reach).ceil() as i32).min(self.h as i32 - 1);
        if bx1 < bx0 || by1 < by0 {
            return;
        }
        let bw = (bx1 - bx0 + 1) as usize;
        let bh = (by1 - by0 + 1) as usize;
        // Per pixel: how far inside the edge (positive is inside), how far along, how
        // far across as a fraction of the half width, and which way across is.
        let mut best = vec![(f32::MIN, 0.0f32, 0.0f32, [0.0f32; 2]); bw * bh];
        let last = (s.spine.len() - 1).max(1) as f32;
        for k in 0..s.spine.len().saturating_sub(1) {
            let (a, b) = (s.spine[k], s.spine[k + 1]);
            let (ha, hb) = (s.half[k], s.half[k + 1]);
            let d = [b[0] - a[0], b[1] - a[1]];
            let len2 = (d[0] * d[0] + d[1] * d[1]).max(1e-8);
            let len = len2.sqrt();
            let dir = [d[0] / len, d[1] / len];
            let across = [-dir[1], dir[0]];
            let r = ha.max(hb) + 1.5;
            let sx0 = ((a[0].min(b[0]) - r).floor() as i32).max(bx0);
            let sy0 = ((a[1].min(b[1]) - r).floor() as i32).max(by0);
            let sx1 = ((a[0].max(b[0]) + r).ceil() as i32).min(bx1);
            let sy1 = ((a[1].max(b[1]) + r).ceil() as i32).min(by1);
            for y in sy0..=sy1 {
                for x in sx0..=sx1 {
                    let p = [x as f32 + 0.5 - a[0], y as f32 + 0.5 - a[1]];
                    let u = ((p[0] * d[0] + p[1] * d[1]) / len2).clamp(0.0, 1.0);
                    let off = [p[0] - d[0] * u, p[1] - d[1] * u];
                    let dist = (off[0] * off[0] + off[1] * off[1]).sqrt();
                    let half = ha + (hb - ha) * u;
                    let inside = half - dist;
                    let i = (y - by0) as usize * bw + (x - bx0) as usize;
                    if inside > best[i].0 {
                        let side = off[0] * across[0] + off[1] * across[1];
                        let s_across = if half > 1e-3 { (side / half).clamp(-1.0, 1.0) } else { 0.0 };
                        best[i] = (inside, (k as f32 + u) / last, s_across, across);
                    }
                }
            }
        }

        for (i, &(inside, t, across_s, across)) in best.iter().enumerate() {
            let a = (inside + 0.5).clamp(0.0, 1.0);
            if a <= 0.0 {
                continue;
            }
            let x = bx0 as usize + i % bw;
            let y = by0 as usize + i / bw;
            let j = y * self.w as usize + x;
            let keep = 1.0 - a;

            let mut col = mix3(s.root, s.tip, t.powf(1.2));
            if t > s.burn_from {
                let k = ((t - s.burn_from) / (1.0 - s.burn_from).max(1e-3)).clamp(0.0, 1.0);
                col = mix3(col, s.burn, smooth(k * 1.6));
            }
            let shade = s.root_shade + (1.0 - s.root_shade) * smooth((t / 0.3).min(1.0));
            // A lighter midrib and faint veins either side of it.
            let rib = 1.0 + 0.1 * s.detail * (-(across_s / 0.18).powi(2)).exp();
            let veins = 1.0 + 0.04 * s.detail * (across_s * 9.0).sin();
            let (m, f) = s.mottle;
            let blotch = 1.0 + m * (0.6 * (t * f[0] + f[1]).sin() + 0.4 * (t * f[2] + f[3]).sin());
            let v = s.value * shade * rib * veins * blotch;
            for k in 0..3 {
                self.rgb[j][k] = (col[k] * v).clamp(0.0, 1.0) * a + self.rgb[j][k] * keep;
            }

            // Folded along the midrib, and the whole blade turned a little about its
            // length. Image y runs down and the normal's green runs up.
            let phi = s.tilt + MIDRIB_FOLD * s.detail * across_s;
            let (sp, cp) = phi.sin_cos();
            let nrm = [across[0] * sp, -across[1] * sp, cp];
            for k in 0..3 {
                self.n[j][k] = nrm[k] * a + self.n[j][k] * keep;
            }
            let rough = s.roughness - 0.06 * s.detail * (-(across_s / 0.18).powi(2)).exp();
            self.r[j] = rough.clamp(0.0, 1.0) * a + self.r[j] * keep;
            self.a[j] = a + self.a[j] * keep;
        }
    }

    fn finish(self) -> Cell {
        let mut albedo = Bitmap::new(self.w, self.h);
        let mut normal = Bitmap::new(self.w, self.h);
        let mut roughness = Bitmap::new(self.w, self.h);
        for i in 0..self.a.len() {
            let (x, y) = (i as u32 % self.w, i as u32 / self.w);
            let a = self.a[i].clamp(0.0, 1.0);
            let inv = if a > 1e-4 { 1.0 / a } else { 0.0 };
            let c = self.rgb[i].map(|v| to_u8(v * inv));
            albedo.put(x, y, [c[0], c[1], c[2], to_u8(a)]);
            let v = self.n[i].map(|v| v * inv);
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let v = if len > 1e-4 { v.map(|c| c / len) } else { [0.0, 0.0, 1.0] };
            normal.put(x, y, [to_u8(v[0] * 0.5 + 0.5), to_u8(v[1] * 0.5 + 0.5), to_u8(v[2] * 0.5 + 0.5), 255]);
            let r = to_u8(if a > 1e-4 { self.r[i] * inv } else { 0.7 });
            roughness.put(x, y, [r, r, r, 255]);
        }
        // Flood the blade colour out over the transparent background, so neither the
        // mip chain nor bilinear filtering darkens the edge of every blade.
        bleed_color_outward(&mut albedo);
        Cell { albedo, normal, roughness }
    }
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(cells: Vec<BladeCell>) -> BladeAtlasParams {
        BladeAtlasParams {
            cell_height: 128,
            cell_aspect: 0.5,
            cells,
        }
    }

    #[test]
    fn cells_sit_side_by_side_and_each_is_covered() {
        let p = small(vec![
            BladeCell::default(),
            BladeCell { dry: true, heads: 5, ..BladeCell::default() },
        ]);
        let baked = bake_blades(&p);
        assert_eq!(baked.maps.albedo.width, 64 * 2);
        assert_eq!(baked.maps.albedo.height, 128);
        assert_eq!(baked.coverage.len(), 2);
        for c in &baked.coverage {
            assert!(*c > 0.05 && *c < 0.9, "coverage {c}");
        }
    }

    #[test]
    fn nothing_reaches_the_edge_of_a_cell() {
        // Neighbouring cells butt up against each other, so anything drawn on the
        // edge bleeds into the next one as soon as the texture is filtered.
        let p = small(vec![BladeCell {
            count: 80,
            lean_deg: 60.0,
            curve_deg: 80.0,
            length: (0.9, 1.0),
            root_spread: 1.0,
            ..BladeCell::default()
        }]);
        let img = bake_blades(&p).maps.albedo;
        for x in 0..img.width {
            assert_eq!(img.at(x, 0)[3], 0, "top edge at {x}");
        }
        for y in 0..img.height {
            assert_eq!(img.at(0, y)[3], 0, "left edge at {y}");
            assert_eq!(img.at(img.width - 1, y)[3], 0, "right edge at {y}");
        }
    }

    #[test]
    fn blades_stand_on_the_bottom_of_the_cell() {
        // The bottom of a cell is the ground: the root end of a card.
        let img = bake_blades(&small(vec![BladeCell::default()])).maps.albedo;
        let covered = |y: u32| (0..img.width).filter(|&x| img.at(x, y)[3] > 128).count();
        assert!(covered(img.height - 6) > covered(8), "more blade at the root than at the top");
    }

    #[test]
    fn dry_cells_are_yellower_than_lush_ones() {
        let lush = BladeCell::default();
        let dry = BladeCell {
            dry: true,
            root: [0.40, 0.36, 0.2],
            tip: [0.78, 0.7, 0.45],
            ..BladeCell::default()
        };
        let img = bake_blades(&small(vec![lush, dry])).maps.albedo;
        let mean = |x0: u32| {
            let mut s = [0.0f64; 3];
            let mut n = 0.0;
            for y in 0..img.height {
                for x in x0..x0 + 64 {
                    let p = img.at(x, y);
                    if p[3] > 200 {
                        for k in 0..3 {
                            s[k] += p[k] as f64;
                        }
                        n += 1.0;
                    }
                }
            }
            s.map(|v| v / n)
        };
        let (l, d) = (mean(0), mean(64));
        assert!(d[0] - d[2] > l[0] - l[2] + 20.0, "lush {l:?} dry {d:?}");
    }

    #[test]
    fn normals_are_unit_and_face_out_of_the_card() {
        let baked = bake_blades(&small(vec![BladeCell { heads: 4, ..BladeCell::default() }]));
        let (a, n) = (&baked.maps.albedo, baked.maps.normal.as_ref().unwrap());
        let mut checked = 0;
        for y in 0..a.height {
            for x in 0..a.width {
                if a.at(x, y)[3] < 250 {
                    continue;
                }
                let v = n.at(x, y).map(|c| c as f32 / 255.0 * 2.0 - 1.0);
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                assert!((len - 1.0).abs() < 0.03, "length {len}");
                assert!(v[2] > 0.3, "normal turned away from the card: {v:?}");
                checked += 1;
            }
        }
        assert!(checked > 200);
    }

    #[test]
    fn every_kind_of_flower_shows_its_colour_and_stays_in_its_cell() {
        use crate::flowers::{FlowerKind, FlowerParams};
        for kind in [FlowerKind::Radial, FlowerKind::Spike, FlowerKind::Umbel, FlowerKind::Brush] {
            let cell = BladeCell {
                count: 20,
                flowers: Some(FlowerParams {
                    kind,
                    count: 4,
                    size: 0.12,
                    color: [0.9, 0.2, 0.8],
                    ..FlowerParams::default()
                }),
                ..BladeCell::default()
            };
            let img = bake_blades(&BladeAtlasParams { cell_height: 256, cell_aspect: 0.5, cells: vec![cell] }).maps.albedo;
            let flowery = (0..img.height)
                .flat_map(|y| (0..img.width).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let p = img.at(x, y);
                    p[3] > 200 && p[0] > p[1] + 40 && p[2] > p[1] + 40
                })
                .count();
            assert!(flowery > 30, "{kind:?}: only {flowery} flower-coloured texels");
            for y in 0..img.height {
                assert_eq!(img.at(0, y)[3], 0, "{kind:?} reaches the left edge");
                assert_eq!(img.at(img.width - 1, y)[3], 0, "{kind:?} reaches the right edge");
            }
        }
    }

    #[test]
    fn baking_is_deterministic() {
        let p = small(vec![BladeCell { heads: 3, ..BladeCell::default() }]);
        assert_eq!(bake_blades(&p).maps.albedo, bake_blades(&p).maps.albedo);
    }
}
