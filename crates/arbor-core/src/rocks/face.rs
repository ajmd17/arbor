//! A cliff face: the front of a rock carved as a sea cliff's is, where a boulder's would be
//! an ellipsoid cut by planes.
//!
//! The rock is a box, its front a depth carved into it across the face: blocks the stone
//! has broken into along its joints, each a plane turned to one of a few joint sets so the
//! same faces come round again and again across the cliff, standing proud of or set back
//! from the blocks beside it with clefts between; a smaller breaking of each of those;
//! chimneys running up it; flutes worn down it, as in chalk; the bedding stepping in and
//! out as ledges and roofs; and the whole leaning back, rounded over at the brow, flared at
//! the foot and tucked back at the ends, so one face set overlapping the next hides the
//! join. The depth is worked out once on a grid a few centimetres fine and read from it,
//! since everything it carries is the mesh's: the stone's own grain is the maps', as on
//! any rock.
//!
//! Rays from the rock's middle meet the face as a depth across it, so the face can step
//! out above as well as below: a bed standing proud of the one under it is a roof.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use super::{mix, par_map, rehash, smoothstep, unit, Noise};
use crate::ranged::{key, Scalar};
use crate::seed::PortableRng;
use rand::SeedableRng;

/// How a cliff face is carved. Metres unless said otherwise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockFace::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockFace<V = f32> {
    /// How far apart the joints the stone has broken along run across the face, and how
    /// much taller than wide the blocks between them stand.
    pub joint_spacing: V,
    pub joint_aspect: V,
    /// How many ways the joints run, 1 to 4: the same few faces come round again across
    /// the cliff, as they do where stone breaks along joint sets.
    pub joint_sets: V,
    /// How far, in degrees, the blocks' faces turn from the cliff's own.
    pub turn: V,
    /// How far the blocks stand proud of or back from one another.
    pub relief: V,
    /// A smaller breaking of each block, as a share of the first.
    pub detail: V,
    /// Chimneys running up the face, how many in each hundred metres across, and how deep
    /// and wide each is.
    pub chimneys: V,
    pub chimney_depth: V,
    pub chimney_width: V,
    /// Flutes worn down the face, as down chalk: how deep, and how far apart.
    pub flutes: V,
    pub flute_spacing: V,
    /// The bedding stepping in and out up the face, as ledges and roofs: how far a bed
    /// stands out or back, and how far apart the beds are.
    pub ledges: V,
    pub ledge_spacing: V,
    /// Degrees the face leans back from upright.
    pub lean: V,
    /// How far the top rounds over at the brow, the foot flares out, and the ends tuck
    /// back; and how far the face curves back across its width.
    pub brow: V,
    pub foot: V,
    pub tuck: V,
    pub bow: V,
}

impl Default for RockFace {
    fn default() -> Self {
        Self {
            joint_spacing: 9.0,
            joint_aspect: 2.5,
            joint_sets: 2.0,
            turn: 28.0,
            relief: 3.5,
            detail: 0.45,
            chimneys: 3.0,
            chimney_depth: 6.0,
            chimney_width: 2.5,
            flutes: 0.0,
            flute_spacing: 3.0,
            ledges: 1.2,
            ledge_spacing: 7.0,
            lean: 6.0,
            brow: 3.0,
            foot: 2.0,
            tuck: 6.0,
            bow: 1.5,
        }
    }
}

impl<V: Scalar> RockFace<V> {
    pub fn defaults() -> Self {
        RockFace::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> RockFace<W> {
        RockFace {
            joint_spacing: f(&key(at, "joint_spacing"), self.joint_spacing),
            joint_aspect: f(&key(at, "joint_aspect"), self.joint_aspect),
            joint_sets: f(&key(at, "joint_sets"), self.joint_sets),
            turn: f(&key(at, "turn"), self.turn),
            relief: f(&key(at, "relief"), self.relief),
            detail: f(&key(at, "detail"), self.detail),
            chimneys: f(&key(at, "chimneys"), self.chimneys),
            chimney_depth: f(&key(at, "chimney_depth"), self.chimney_depth),
            chimney_width: f(&key(at, "chimney_width"), self.chimney_width),
            flutes: f(&key(at, "flutes"), self.flutes),
            flute_spacing: f(&key(at, "flute_spacing"), self.flute_spacing),
            ledges: f(&key(at, "ledges"), self.ledges),
            ledge_spacing: f(&key(at, "ledge_spacing"), self.ledge_spacing),
            lean: f(&key(at, "lean"), self.lean),
            brow: f(&key(at, "brow"), self.brow),
            foot: f(&key(at, "foot"), self.foot),
            tuck: f(&key(at, "tuck"), self.tuck),
            bow: f(&key(at, "bow"), self.bow),
        }
    }
}

/// Share of the rock's depth behind the point its rays leave from: the rest is in front,
/// so they meet the face near square on.
pub(super) const BACK_SHARE: f32 = 0.15;

/// Steeper than the face is anywhere, for tracing rays to it.
const MAX_SLOPE: f32 = 4.0;

/// How steeply a block's face falls away past its own ground, where it meets its
/// neighbour's.
const BLOCK_FALL: f32 = 1.1;

/// Share of the joints between blocks that have opened into clefts.
const OPEN_JOINTS: f32 = 0.35;

/// A chimney up the face.
struct Chimney {
    x: f32,
    width: f32,
    depth: f32,
    from: f32,
    to: f32,
    phase: f32,
}

/// The ways the joints run: each set's turn across the face and its tilt up it, radians.
struct JointSets {
    turn: Vec<f32>,
    tilt: Vec<f32>,
}

/// The face's depth, carved into the rock, sampled finely across it.
pub(super) struct FaceField {
    /// The grid's corner across and up the face, rock-local metres, and its spacing.
    min: Vec2,
    cell: f32,
    nx: usize,
    ny: usize,
    depth: Vec<f32>,
    /// How far each point lies carved deeper than the face round it, 0 to 1: the clefts
    /// and chimneys, dark in a photograph.
    deep: Vec<f32>,
    /// How far forward of the rock's middle the face stands before it is carved, and its
    /// least depth: as far as it ever stands out.
    front: f32,
    least: f32,
}

impl FaceField {
    /// The face of a rock `half` across and up, standing `front` forward of its middle.
    pub(super) fn new(face: &RockFace, half: Vec3, front: f32, seed: u64) -> Self {
        let mut rng = PortableRng::seed_from_u64(seed ^ 0xC11F_FACE_0000_0001);
        let noise = Noise::new(&mut rng);
        let salt = mix((seed as u32) ^ ((seed >> 32) as u32) ^ 0xFACE_C11F);

        let width = half.x * 2.0;
        let height = half.y * 2.0;
        let cell = (width.max(height) / 1024.0).clamp(0.03, 0.25);
        let margin = 2.0 * cell;
        let nx = ((width + 2.0 * margin) / cell).ceil() as usize + 1;
        let ny = ((height + 2.0 * margin) / cell).ceil() as usize + 1;
        let min = Vec2::new(-half.x - margin, -half.y - margin);

        let count = (face.joint_sets.round() as i32).clamp(1, 4) as usize;
        let turn = face.turn.max(0.0).to_radians();
        let sets = JointSets {
            turn: (0..count)
                .map(|k| {
                    let spread = if count == 1 { 0.0 } else { -turn + 2.0 * turn * k as f32 / (count - 1) as f32 };
                    spread + (unit(rehash(salt, 100 + k as u32)) - 0.5) * 0.2
                })
                .collect(),
            tilt: (0..count).map(|k| (unit(rehash(salt, 200 + k as u32)) - 0.5) * turn * 0.6).collect(),
        };

        let chimneys: Vec<Chimney> = {
            let n = ((face.chimneys.max(0.0) * width / 100.0).round() as usize).min(32);
            (0..n as u32)
                .map(|k| {
                    let h = rehash(salt, 300 + k);
                    Chimney {
                        x: (unit(rehash(h, 1)) - 0.5) * 1.6 * half.x,
                        width: 0.0,
                        depth: face.chimney_depth.max(0.0) * (0.5 + 0.7 * unit(rehash(h, 3))),
                        from: height * 0.4 * unit(rehash(h, 4)),
                        to: height * (0.7 + 0.3 * unit(rehash(h, 5))),
                        phase: 50.0 * unit(rehash(h, 6)),
                    }
                })
                .map(|mut c: Chimney| {
                    // Never steeper than the face can be traced: as wide as most of its depth.
                    let h = rehash(salt, 300 + c.phase.to_bits());
                    c.width = (face.chimney_width.max(0.2) * (0.6 + 0.8 * unit(rehash(h, 2)))).max(c.depth * 0.6);
                    c
                })
                .collect()
        };

        let carve = Carve { face, half, sets, chimneys, noise: &noise, salt, front };
        let rows = par_map(ny, |j| (0..nx).map(|i| carve.depth(min.x + i as f32 * cell, min.y + j as f32 * cell)).collect::<Vec<f32>>());
        let depth: Vec<f32> = rows.into_iter().flatten().collect();

        // Deep against the face round it: the clefts and chimneys, measured over a few
        // metres, the size of the blocks they lie between.
        let reach = ((face.joint_spacing.max(1.0) * 0.35).max(2.5) / cell).round() as usize;
        let blurred = box_blur(&depth, nx, ny, reach.max(1));
        let scale = face.relief.max(0.5) * 0.8 + 0.5;
        let deep = depth.iter().zip(&blurred).map(|(&d, &b)| smoothstep(0.0, scale, d - b)).collect();
        let least = depth.iter().copied().fold(f32::MAX, f32::min);

        Self { min, cell, nx, ny, depth, deep, front, least }
    }

    fn lookup(&self, grid: &[f32], x: f32, y: f32) -> f32 {
        let gx = ((x - self.min.x) / self.cell).clamp(0.0, (self.nx - 1) as f32 - 1e-3);
        let gy = ((y - self.min.y) / self.cell).clamp(0.0, (self.ny - 1) as f32 - 1e-3);
        let (i, j) = (gx as usize, gy as usize);
        let (fx, fy) = (gx - i as f32, gy - j as f32);
        let at = |i: usize, j: usize| grid[j * self.nx + i];
        let top = at(i, j) + (at(i + 1, j) - at(i, j)) * fx;
        let bottom = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * fx;
        top + (bottom - top) * fy
    }

    /// How far the face is carved back at `(x, y)` across and up it.
    pub(super) fn depth_at(&self, x: f32, y: f32) -> f32 {
        self.lookup(&self.depth, x, y)
    }

    /// How deep a cleft `(x, y)` lies in, 0 to 1.
    pub(super) fn deep_at(&self, x: f32, y: f32) -> f32 {
        self.lookup(&self.deep, x, y)
    }

    /// Where the face stands forward of the rock's middle at `(x, y)`.
    fn z(&self, x: f32, y: f32) -> f32 {
        self.front - self.depth_at(x, y)
    }

    /// The way the face looks at `(x, y)`.
    fn normal(&self, x: f32, y: f32) -> Vec3 {
        let e = self.cell;
        let dx = (self.depth_at(x + e, y) - self.depth_at(x - e, y)) / (2.0 * e);
        let dy = (self.depth_at(x, y + e) - self.depth_at(x, y - e)) / (2.0 * e);
        Vec3::new(dx, dy, 1.0).normalize()
    }

    /// How far out along `u`, a unit direction from the rock's middle, it leaves the rock
    /// through the face, and which way the face looks there; `None` for a direction that
    /// never reaches it. The first crossing out from the middle, so a ray past a roof
    /// stops at the roof.
    pub(super) fn exit(&self, u: Vec3) -> Option<(f32, Vec3)> {
        if u.z <= 1e-4 {
            return None;
        }
        let outside = |t: f32| u.z * t - self.z(u.x * t, u.y * t);
        let far = (self.front - self.least).max(0.01) / u.z * 1.001;
        // Traced: no nearer than the face could be at its steepest, so a thin wall of a
        // cleft is never stepped over, and long strides where the face is far.
        let rate = u.z + MAX_SLOPE * Vec2::new(u.x, u.y).length();
        let least_step = self.cell;
        let (mut lo, mut hi) = (0.0f32, far);
        let mut t = 0.0f32;
        let mut found = false;
        for _ in 0..600 {
            let g = outside(t);
            if g >= 0.0 {
                hi = t;
                found = true;
                break;
            }
            lo = t;
            t = (t + (-g / rate).max(least_step)).min(far);
            if lo >= far {
                break;
            }
        }
        if !found {
            // Traced too long: searched evenly instead, finely enough for a cleft.
            let steps = ((far / (self.cell * 4.0)).ceil() as u32).clamp(64, 4096);
            lo = 0.0;
            for k in 1..=steps {
                let t = far * k as f32 / steps as f32;
                if outside(t) >= 0.0 {
                    hi = t;
                    found = true;
                    break;
                }
                lo = t;
            }
            if !found {
                return Some((far, Vec3::Z));
            }
        }
        for _ in 0..12 {
            let mid = 0.5 * (lo + hi);
            if outside(mid) >= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let t = 0.5 * (lo + hi);
        Some((t, self.normal(u.x * t, u.y * t)))
    }
}

/// What carves the face, and what it needs to.
struct Carve<'a> {
    face: &'a RockFace,
    half: Vec3,
    sets: JointSets,
    chimneys: Vec<Chimney>,
    noise: &'a Noise,
    salt: u32,
    front: f32,
}

impl Carve<'_> {
    /// How far the face is carved back at `x` across and `y` up the rock, from its middle.
    fn depth(&self, x: f32, y: f32) -> f32 {
        let f = self.face;
        let height = self.half.y * 2.0;
        let v = y + self.half.y;
        let relief = f.relief.max(0.0);
        let spacing = f.joint_spacing.max(1.0);
        let aspect = f.joint_aspect.max(0.2);

        // The blocks, then each broken smaller.
        let mut out = self.blocks(x, v, spacing, aspect, relief, self.salt);
        if f.detail > 0.0 {
            // Broken again at a third the size, and again at a tenth: fracture at every
            // scale down to where the maps take over.
            out += self.blocks(x, v, spacing * 0.35, aspect * 0.8, relief * 0.35 * f.detail, rehash(self.salt, 7));
            out += self.blocks(x, v, spacing * 0.12, aspect * 0.6, relief * 0.12 * f.detail, rehash(self.salt, 8));
        }
        let mut depth = relief - out;

        // Broad rolls across the whole face, the size of the cliff's buttresses and bays.
        depth += relief * 0.35 * self.noise.fbm(Vec3::new(x / 28.0, v / 28.0, 5.5), 3);

        for c in &self.chimneys {
            let wander = c.x + 0.7 * c.width * self.noise.at(Vec3::new(v / 14.0, c.phase, 1.3));
            let across = (x - wander).abs() / c.width;
            if across < 1.0 {
                let along = smoothstep(c.from, c.from + 5.0, v) * (1.0 - smoothstep(c.to - 5.0, c.to, v));
                depth += c.depth * (1.0 - across).powf(1.6) * along;
            }
        }

        if f.flutes > 0.0 {
            // Grooves along the contours of noise drawn out up the face: narrow, between
            // broad rounded ribs.
            let s = f.flute_spacing.max(0.3);
            let n = self.noise.fbm(Vec3::new(x / s, v / (s * 7.0), 3.7), 3);
            depth += f.flutes * (1.0 - n.abs().min(1.0)).powi(5);
        }

        if f.ledges > 0.0 {
            // Each bed its own way in or out, stepping to the next over a short riser.
            let s = f.ledge_spacing.max(0.5);
            let b = (v + 0.06 * x) / s + 0.35 * self.noise.at(Vec3::new(x / 25.0, v / 25.0, 9.1));
            let bed = b.floor();
            let within = b - bed;
            let offset = |k: f32| f.ledges * (unit(rehash(self.salt ^ 0x1ED6_E5E7, k as i32 as u32)) - 0.5) * 2.0;
            depth += offset(bed) + (offset(bed + 1.0) - offset(bed)) * smoothstep(0.6, 1.0, within);
        }

        // Leaning back about its middle, so the lean does not carry the top away.
        depth += f.lean.clamp(-30.0, 60.0).to_radians().tan() * (v - height * 0.5);
        depth += f.bow * (x / self.half.x.max(0.1)).powi(2);

        let brow = f.brow.max(0.0);
        if brow > 0.0 {
            let s = ((v - (height - brow)) / brow).clamp(0.0, 1.0);
            depth += brow * (1.0 - (1.0 - s * s).max(0.0).sqrt());
        }
        let foot = f.foot.max(0.0);
        if foot > 0.0 {
            let s = (1.0 - v / (foot * 3.0)).clamp(0.0, 1.0);
            depth -= foot * s * s;
        }
        let tuck = f.tuck.max(0.0);
        if tuck > 0.0 {
            let reach = (tuck * 1.5).min(self.half.x * 0.4);
            let s = ((x.abs() - (self.half.x - reach)) / reach).clamp(0.0, 1.0);
            depth += tuck * s * s;
        }

        // Never through to the back, so the rock's middle stays inside it.
        depth.clamp(-(foot + relief + 1.0), self.front * 0.8)
    }

    /// How far the face stands out at `(x, v)`, from blocks about `spacing` across and
    /// `aspect` times as tall that tile it, as joints tile a face of stone: each block's
    /// face a plane turned to its joint set, eased into its neighbours' over a metre or so
    /// so a step between them is steep but never sheer; and some of the joints open into
    /// clefts.
    fn blocks(&self, x: f32, v: f32, spacing: f32, aspect: f32, relief: f32, salt: u32) -> f32 {
        let tall = spacing * aspect;
        let (gx, gy) = ((x / spacing).floor() as i32, (v / tall).floor() as i32);
        // Each block nearby: how far, the face it would give here, and its hash.
        let mut near = [(f32::MAX, 0.0f32, 0u32, Vec2::ZERO); 25];
        let mut n = 0;
        for dj in -2..=2 {
            for di in -2..=2 {
                let (i, j) = (gx + di, gy + dj);
                let h = mix(salt ^ (i as u32).wrapping_mul(0x9E37_79B9) ^ (j as u32).wrapping_mul(0x85EB_CA6B));
                let cx = (i as f32 + 0.15 + 0.7 * unit(rehash(h, 1))) * spacing;
                let cy = (j as f32 + 0.15 + 0.7 * unit(rehash(h, 2))) * tall;
                let set = (rehash(h, 3) as usize) % self.sets.turn.len();
                let across = (self.sets.turn[set] + (unit(rehash(h, 4)) - 0.5) * 0.16).tan();
                let up = (self.sets.tilt[set] + (unit(rehash(h, 5)) - 0.5) * 0.16).tan();
                let proud = relief * (unit(rehash(h, 6)) - 0.5) * 2.0;
                // Its tilt bounded, so a block's face carried far past its own ground,
                // up a tall block's length, never towers over the blocks there.
                let tilt = (-across * (x - cx) - up * (v - cy)).clamp(-1.5 * relief, 1.5 * relief);
                let plane = proud + tilt;
                // Drawn out up the face, so the blocks stand tall.
                let centre = Vec2::new(cx, cy / aspect);
                let far = Vec2::new(x, v / aspect).distance(centre);
                near[n] = (far, plane, h, centre);
                n += 1;
            }
        }
        let near = &mut near[..n];
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (d1, d2) = (near[0].0, near[1].0);

        // The highest of the planes round here, each falling away from its own block
        // at a slope: so neighbours meet in a crisp crease, as faces of broken stone do,
        // and never in a sheer step.
        let mut out = near[0].1;
        let here = Vec2::new(x, v / aspect);
        for &(_, plane, _, centre) in near[1..].iter() {
            // How far past the straight joint with the nearest block this one's face has
            // to reach, so joints run straight as they do in stone.
            let across = centre - near[0].3;
            let past = (here - (near[0].3 + centre) * 0.5).dot(across) / across.length().max(1e-3);
            out = out.max(plane - BLOCK_FALL * (-past).max(0.0));
        }

        // How far into the nearest block from the joint with the next: half the
        // difference of their squared distances over how far apart they are.
        let apart = near[0].3.distance(near[1].3).max(1e-3);
        let edge = (d2 * d2 - d1 * d1) / (2.0 * apart);
        // The joint between these two opens into a cleft, or not; the same seen from
        // either side.
        let pair = mix(near[0].2 ^ near[1].2 ^ 0x0C1E_F75A);
        if unit(pair) < OPEN_JOINTS {
            let depth = relief * (0.6 + 1.0 * unit(rehash(pair, 2)));
            // A V cut in, never steeper than its walls can be seen.
            let width = (spacing * (0.04 + 0.08 * unit(rehash(pair, 1)))).max(depth * 0.6);
            out -= depth * (1.0 - smoothstep(0.0, width, edge));
        }
        out
    }
}

/// `grid` blurred by a box `reach` cells either way, twice over, so it is smooth.
fn box_blur(grid: &[f32], nx: usize, ny: usize, reach: usize) -> Vec<f32> {
    let pass = |src: &[f32], along_x: bool| -> Vec<f32> {
        let (outer, inner) = if along_x { (ny, nx) } else { (nx, ny) };
        let at = |o: usize, k: usize| if along_x { o * nx + k } else { k * nx + o };
        let lines = par_map(outer, |o| {
            let mut prefix = Vec::with_capacity(inner + 1);
            prefix.push(0.0f64);
            for k in 0..inner {
                prefix.push(prefix[k] + src[at(o, k)] as f64);
            }
            (0..inner)
                .map(|k| {
                    let (a, b) = (k.saturating_sub(reach), (k + reach + 1).min(inner));
                    ((prefix[b] - prefix[a]) / (b - a) as f64) as f32
                })
                .collect::<Vec<f32>>()
        });
        let mut out = vec![0.0; src.len()];
        for (o, line) in lines.into_iter().enumerate() {
            for (k, v) in line.into_iter().enumerate() {
                out[at(o, k)] = v;
            }
        }
        out
    };
    let mut out = grid.to_vec();
    for _ in 0..2 {
        out = pass(&out, true);
        out = pass(&out, false);
    }
    out
}
