//! Climbers: plants that grow over something else instead of holding themselves up.
//!
//! A vine is grown into the same skeleton a tree is, so the bark mesher, the leaf cards,
//! the wind and the exporters all take it as they take a tree. What differs is how the
//! stems find their way. A tree's stems are steered by light, weight and the crown
//! envelope; a vine's are steered by the surface under them. Each growing tip is held
//! to its support (`adhesion`), pushed up or down along it by the level's own
//! `phototropism` and `gravity` read in the plane of the surface, and never allowed
//! inside it. A tip that loses the surface — over the top of a wall, off its end, or a
//! shoot the species throws out into the air — carries on as a free stem and sags.
//!
//! The support is described as a distance field, so a wall, a column and the ground
//! are all the same thing to the grower: how far away it is, and which way is out.
//! It is not part of the plant and is not exported; the vine is grown in the support's
//! own frame, with the foot of the support at the origin, so it drops onto a wall of
//! the same size placed at the same point.
//!
//! Three things make the difference between ivy and a tangle of wire:
//!
//! - Side shoots leave in the plane of the surface, not round the stem, and a level
//!   that climbs sends its shoots up while one that hangs sends them down.
//! - Growth stops where the surface is already covered. Real ivy is shaded out under
//!   its own leaves; here a tip that runs for `crowding` metres through ground another
//!   stem already holds stops, which is what spreads a vine into an even cover.
//! - Leaves turn their faces out along the surface normal (`SkeletonNode::surface`),
//!   and shingle over one another rather than standing out like a tree's.

use std::collections::HashMap;

use glam::Vec3;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::growth::{
    break_dead_wood, child_dir, lateral_cap, levels_total, mark_dieback, nominal_vigor,
    norm_or_up, rand_perpendicular, resolve_radii, stem_params, tip_reach, turn_toward,
    AHEAD_SALT, MAX_DRIVE, MAX_NODES, MIN_VIGOR, ROOT_PATH, SUPPRESSION_FLOOR,
    TIP_REACH_SPREAD, TRUNK_STEM,
};
use crate::math::{ortho_of, ortho_unit, transport};
use crate::mesh::Mesh;
use crate::ranged::{key, Scalar};
use crate::seed::{child_path, range_f32, TreeRng};
use crate::skeleton::Skeleton;
use crate::species::{ChildPattern, SpeciesParams, StemParams};
use crate::wind::SwayAt;

/// How far a vine may turn, in radians per metre. Several times what a tree's limb is
/// allowed: a climber follows the corners of what it grows on, and a tendril wraps a
/// wire in a few centimetres.
const MAX_TURN_PER_M: f32 = 4.0;
/// How far a stem free of its support may turn, in radians per metre. With nothing to
/// follow, the turning a clinging stem needs only curls a free one into loops.
const FREE_TURN_PER_M: f32 = 2.0;
/// Share of the level's wander a free stem keeps, for the same reason.
const FREE_WANDER: f32 = 0.35;
/// How much of the previous wander a segment keeps, so a stem meanders in long arcs
/// rather than jittering from node to node.
const BEND_KEEP: f32 = 0.8;
/// Step for the finite differences the surface normal is read with, in metres.
const NORMAL_EPS: f32 = 1e-3;
/// Salt for the per-shoot draws that decide which way a side shoot leaves.
const SIDE_SALT: u64 = 0x5DE5_0F7A_11E5_0002;
/// How hard a stem off its support sags under its own weight, per segment. A climber
/// puts its wood into length rather than into holding itself up, so a shoot that has
/// lost the wall hangs rather than standing into the air the way a tree's would.
const FREE_SAG: f32 = 0.08;
/// Share of the level's pull upward a free stem keeps.
const FREE_LIFT: f32 = 0.25;
/// Share of side shoots that go the way their level leans — up for a climber, down
/// for a trailer — rather than the other way.
const SIDE_BIAS: f32 = 0.85;

/// What a vine grows over, in metres, with its foot at the origin.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Support {
    /// A slab standing on the ground, its face toward +Z at z = 0, centred on x = 0.
    Wall { width: f32, height: f32, thickness: f32 },
    /// A round column standing on the ground at the origin: a post, a pillar, or a
    /// trunk for the vine to climb.
    Pillar { radius: f32, height: f32 },
    /// Nothing but the ground itself, for a creeper that runs across it.
    Ground,
}

/// Where the plant is rooted on its support.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VineStart {
    /// At the foot of the support, facing out: a climber.
    #[default]
    Foot,
    /// At the top edge of the support, just over its face: a plant in a planter on a
    /// wall, or growing along a ledge, whose shoots trail down the face.
    Top,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "VineParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct VineParams<V = f32> {
    pub support: Support,
    pub start: VineStart,
    /// Stems the plant puts out from its root. They are its trunk: level 0.
    pub runners: u32,
    /// Which way the runners set off across the support, in degrees from straight up
    /// it: 0 climbs, 90 runs sideways, 180 heads down. Runners alternate sides of it.
    /// On the ground they fan out all the way round instead.
    pub heading_deg: V,
    /// Random spread either side of that heading, in degrees.
    pub fan_deg: V,
    /// How hard a stem holds to its support, from 0 to 1. At 0 nothing clings and
    /// every stem hangs free: a curtain of lianas off a ledge. At 1 a stem lies on the
    /// surface wherever it can reach it: ivy.
    pub adhesion: V,
    /// How far off the surface a stem can still find it, in metres. A stem further
    /// out than this is free of it, and sags.
    pub reach: V,
    /// Sideways pull along the surface, per segment, for a vine that winds round its
    /// support rather than going straight up it. The sign is the hand of the spiral.
    pub twine: V,
    /// Share of side shoots that grow out away from the support instead of along it.
    /// Ivy that has reached the light puts out bushy shoots that stand off the wall;
    /// they are what gives an old ivy its depth.
    pub stand_off: V,
    /// How far those shoots lean out from the surface, in degrees.
    pub stand_off_deg: V,
    /// Metres of ground another stem already holds that a tip may run through before
    /// it stops: shaded out under its neighbour's leaves. Zero never stops a stem.
    pub crowding: V,
    /// Size of the cells that ground is measured in, in metres. About the spread of
    /// one stem's leaves.
    pub crowding_cell: V,
    /// How far a clinging stem that has lost its support grows on in search of it, in
    /// metres, before it gives up. A climber does not build wood that holds itself up,
    /// so what outgrows the wall flops over and stops rather than reaching on into the
    /// air. Zero lets it grow on as far as its length allows, which is what a trailer
    /// hanging free wants. Shoots thrown out on purpose (`stand_off`) are not held to
    /// it.
    pub free_length: V,
    /// How far a leaf on a clinging stem stands off the surface, in degrees. Near 0
    /// they lie flat like shingles; higher lifts them off the wall toward the light.
    pub leaf_lift_deg: V,
}

impl Default for VineParams {
    fn default() -> Self {
        Self {
            support: Support::Wall {
                width: 4.0,
                height: 3.0,
                thickness: 0.3,
            },
            start: VineStart::Foot,
            runners: 3,
            heading_deg: 0.0,
            fan_deg: 30.0,
            adhesion: 0.9,
            reach: 0.12,
            twine: 0.0,
            stand_off: 0.1,
            stand_off_deg: 50.0,
            crowding: 0.25,
            crowding_cell: 0.1,
            free_length: 0.4,
            leaf_lift_deg: 25.0,
        }
    }
}

impl<V: Scalar> VineParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        VineParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> VineParams<W> {
        VineParams {
            support: self.support,
            start: self.start,
            runners: self.runners,
            heading_deg: f(&key(at, "heading_deg"), self.heading_deg),
            fan_deg: f(&key(at, "fan_deg"), self.fan_deg),
            adhesion: f(&key(at, "adhesion"), self.adhesion),
            reach: f(&key(at, "reach"), self.reach),
            twine: f(&key(at, "twine"), self.twine),
            stand_off: f(&key(at, "stand_off"), self.stand_off),
            stand_off_deg: f(&key(at, "stand_off_deg"), self.stand_off_deg),
            crowding: f(&key(at, "crowding"), self.crowding),
            crowding_cell: f(&key(at, "crowding_cell"), self.crowding_cell),
            free_length: f(&key(at, "free_length"), self.free_length),
            leaf_lift_deg: f(&key(at, "leaf_lift_deg"), self.leaf_lift_deg),
        }
    }
}

impl Support {
    /// Signed distance from `p` to the support: negative inside it.
    ///
    /// The wall and the column carry on a metre below the ground, so a stem at their
    /// foot sees their face and not a bottom edge it would try to wrap round.
    pub fn distance(&self, p: Vec3) -> f32 {
        match *self {
            Support::Wall {
                width,
                height,
                thickness,
            } => {
                let bottom = -1.0;
                let center = Vec3::new(0.0, (height + bottom) * 0.5, -thickness * 0.5);
                let half = Vec3::new(width * 0.5, (height - bottom) * 0.5, thickness * 0.5);
                let q = (p - center).abs() - half;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
            Support::Pillar { radius, height } => {
                let bottom = -1.0;
                let radial = Vec3::new(p.x, 0.0, p.z).length() - radius;
                let mid = (height + bottom) * 0.5;
                let vertical = (p.y - mid).abs() - (height - bottom) * 0.5;
                let outside = Vec3::new(radial.max(0.0), vertical.max(0.0), 0.0).length();
                outside + radial.max(vertical).min(0.0)
            }
            Support::Ground => p.y,
        }
    }

    /// The way out of the support nearest `p`.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        let e = NORMAL_EPS;
        let d = |o: Vec3| self.distance(p + o) - self.distance(p - o);
        Vec3::new(d(Vec3::X * e), d(Vec3::Y * e), d(Vec3::Z * e)).normalize_or(Vec3::Y)
    }

    /// Where the plant is rooted, and which way is out of the support there.
    pub fn root(&self, start: VineStart) -> (Vec3, Vec3) {
        match (*self, start) {
            (Support::Wall { .. }, VineStart::Foot) => (Vec3::ZERO, Vec3::Z),
            (Support::Wall { height, .. }, VineStart::Top) => {
                (Vec3::new(0.0, (height - 0.03).max(0.0), 0.0), Vec3::Z)
            }
            (Support::Pillar { radius, .. }, VineStart::Foot) => {
                (Vec3::new(0.0, 0.0, radius), Vec3::Z)
            }
            (Support::Pillar { radius, height }, VineStart::Top) => {
                (Vec3::new(0.0, (height - 0.03).max(0.0), radius), Vec3::Z)
            }
            (Support::Ground, _) => (Vec3::ZERO, Vec3::Y),
        }
    }

    /// The support as a plain mesh, for a viewer to stand the vine against. It is no
    /// part of the plant and nothing exports it.
    pub fn mesh(&self) -> Mesh {
        let mut mesh = Mesh::default();
        match *self {
            Support::Wall {
                width,
                height,
                thickness,
            } => {
                let (x0, x1) = (-width * 0.5, width * 0.5);
                let (y0, y1) = (-0.05, height);
                let (z0, z1) = (-thickness, 0.0);
                // Each face as its corner, the two edges along it, and its normal.
                let faces = [
                    (Vec3::new(x0, y0, z1), Vec3::X * width, Vec3::Y * (y1 - y0)),
                    (Vec3::new(x1, y0, z0), -Vec3::X * width, Vec3::Y * (y1 - y0)),
                    (Vec3::new(x1, y0, z1), -Vec3::Z * thickness, Vec3::Y * (y1 - y0)),
                    (Vec3::new(x0, y0, z0), Vec3::Z * thickness, Vec3::Y * (y1 - y0)),
                    (Vec3::new(x0, y1, z1), Vec3::X * width, -Vec3::Z * thickness),
                ];
                for (corner, u, v) in faces {
                    push_quad(&mut mesh, corner, u, v);
                }
            }
            Support::Pillar { radius, height } => {
                const SIDES: u32 = 40;
                let base = mesh.positions.len() as u32;
                let circumference = std::f32::consts::TAU * radius;
                for i in 0..=SIDES {
                    let a = i as f32 / SIDES as f32 * std::f32::consts::TAU;
                    let n = Vec3::new(a.sin(), 0.0, a.cos());
                    let tangent = Vec3::new(a.cos(), 0.0, -a.sin());
                    for (y, v) in [(-0.05, 0.0), (height, height)] {
                        mesh.positions.push((n * radius + Vec3::Y * y).to_array());
                        mesh.normals.push(n.to_array());
                        mesh.tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
                        mesh.uvs.push([i as f32 / SIDES as f32 * circumference, v]);
                    }
                }
                for i in 0..SIDES {
                    let a = base + i * 2;
                    mesh.indices.extend_from_slice(&[a, a + 2, a + 3, a, a + 3, a + 1]);
                }
                let center = mesh.positions.len() as u32;
                push_vertex(&mut mesh, Vec3::Y * height, Vec3::Y, Vec3::X, [0.0, 0.0]);
                for i in 0..=SIDES {
                    let a = i as f32 / SIDES as f32 * std::f32::consts::TAU;
                    let p = Vec3::new(a.sin() * radius, height, a.cos() * radius);
                    push_vertex(&mut mesh, p, Vec3::Y, Vec3::X, [p.x, p.z]);
                }
                for i in 0..SIDES {
                    mesh.indices
                        .extend_from_slice(&[center, center + 1 + i, center + 2 + i]);
                }
            }
            Support::Ground => {}
        }
        let n = mesh.positions.len();
        mesh.weathering = vec![0.0; n];
        mesh.sway = vec![[[0.0; 4]; crate::wind::SWAY_ORDERS]; n];
        mesh.sway_at = vec![SwayAt::default(); n];
        mesh
    }
}

fn push_vertex(mesh: &mut Mesh, p: Vec3, n: Vec3, t: Vec3, uv: [f32; 2]) {
    mesh.positions.push(p.to_array());
    mesh.normals.push(n.to_array());
    mesh.tangents.push([t.x, t.y, t.z, 1.0]);
    mesh.uvs.push(uv);
}

/// One face, from `corner` along `u` and `v`, facing `u x v`. Textured a metre to a
/// repeat.
fn push_quad(mesh: &mut Mesh, corner: Vec3, u: Vec3, v: Vec3) {
    let n = u.cross(v).normalize_or(Vec3::Y);
    let t = u.normalize_or(Vec3::X);
    let base = mesh.positions.len() as u32;
    let (lu, lv) = (u.length(), v.length());
    for (p, uv) in [
        (corner, [0.0, 0.0]),
        (corner + u, [lu, 0.0]),
        (corner + u + v, [lu, lv]),
        (corner + v, [0.0, lv]),
    ] {
        push_vertex(mesh, p, n, t, uv);
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// The part of `v` lying along a surface with normal `n`.
fn along_surface(v: Vec3, n: Vec3) -> Vec3 {
    v - n * v.dot(n)
}

/// Up, as it runs across a surface with normal `n`: the way a climber on it goes. Zero
/// on a surface that faces straight up, where there is no uphill.
fn uphill(n: Vec3) -> Vec3 {
    let u = along_surface(Vec3::Y, n);
    if u.length_squared() < 1e-4 {
        Vec3::ZERO
    } else {
        u.normalize()
    }
}

/// A child settled on while its parent is still growing; see `growth::Pending`.
struct Pending {
    attach: u32,
    at_len: f32,
    dir: Vec3,
    vigor: f32,
    level: u8,
    path: u64,
    /// Set for a shoot thrown out into the air, which holds to nothing and nor does
    /// anything it carries.
    free: bool,
}

/// Ground held, by cell, and by which stem.
#[derive(Default)]
struct Cover {
    cell: f32,
    held: HashMap<(i32, i32, i32), u32>,
}

impl Cover {
    fn key(&self, p: Vec3) -> (i32, i32, i32) {
        let c = |v: f32| (v / self.cell).floor() as i32;
        (c(p.x), c(p.y), c(p.z))
    }

    /// Whether the cell at `p` is already held by a stem other than `stem` and the one
    /// it grows from, which a new shoot has to cross to get anywhere.
    fn taken(&self, p: Vec3, stem: u32, parent_stem: u32) -> bool {
        self.held
            .get(&self.key(p))
            .is_some_and(|&s| s != stem && s != parent_stem)
    }

    fn hold(&mut self, p: Vec3, stem: u32) {
        let k = self.key(p);
        self.held.entry(k).or_insert(stem);
    }
}

struct VineCtx<'a> {
    params: &'a SpeciesParams,
    vine: &'a VineParams,
    levels_total: u8,
    nominal: Vec<f32>,
    rng: TreeRng,
}

impl VineCtx<'_> {
    fn nominal(&self, level: u8) -> f32 {
        self.nominal
            .get(level as usize)
            .copied()
            .unwrap_or(1.0)
            .max(1e-4)
    }
}

/// Grows a climber over its support. `params.vine` must be set.
pub fn grow_vine(params: &SpeciesParams, vine: &VineParams) -> Skeleton {
    let mut skeleton = Skeleton::default();
    let (root_pos, root_n) = vine.support.root(vine.start);
    let root = skeleton.push_node(None, root_pos, 0, ROOT_PATH, 1.0, 0.0, TRUNK_STEM);
    skeleton.nodes[root as usize].surface = root_n;
    let ctx = VineCtx {
        params,
        vine,
        levels_total: levels_total(params),
        nominal: nominal_vigor(params),
        rng: TreeRng::new(params.seed ^ 0x0717_E5EE_D0F0_71E5),
    };
    let mut cover = Cover {
        cell: vine.crowding_cell.max(0.01),
        ..Default::default()
    };
    let mut rng = ctx.rng.stream(ROOT_PATH);

    // The frame the runners set off in: up the surface, and across it.
    let up = {
        let u = uphill(root_n);
        if u == Vec3::ZERO { ortho_of(root_n) } else { u }
    };
    let across = root_n.cross(up).normalize_or(Vec3::X);
    let runners = vine.runners.max(1);
    for k in 0..runners {
        let heading = if vine.support == Support::Ground {
            (k as f32 + 0.5) / runners as f32 * 360.0
        } else if k % 2 == 0 {
            vine.heading_deg
        } else {
            -vine.heading_deg
        };
        let a = (heading + range_f32(&mut rng, -vine.fan_deg, vine.fan_deg)).to_radians();
        let dir = up * a.cos() + across * a.sin();
        let vigor = 1.0 + range_f32(&mut rng, -0.25, 0.25);
        // The first runner carries on the root's own stem, so the plant has one: the
        // rest leave it at the root.
        let stem = if k == 0 {
            TRUNK_STEM
        } else {
            skeleton.nodes.len() as u32
        };
        grow_stem(
            &ctx,
            &mut skeleton,
            &mut cover,
            root,
            dir,
            vigor,
            0,
            child_path(ROOT_PATH, k),
            stem,
            f32::INFINITY,
            false,
        );
    }

    resolve_radii(params, &mut skeleton);
    // Everything clinging was held off the surface by its level's radius while it
    // grew. Now the real radii are known, the thick old runners are lifted clear of it
    // so their bark lies on the wall rather than in it.
    for node in &mut skeleton.nodes {
        if node.surface != Vec3::ZERO {
            let d = vine.support.distance(node.position);
            if d < node.radius {
                node.position += vine.support.normal(node.position) * (node.radius - d);
            }
        }
    }
    let keep = mark_dieback(params, &mut skeleton);
    break_dead_wood(params, &mut skeleton, keep.as_deref());
    skeleton
}

#[allow(clippy::too_many_arguments)]
fn grow_stem(
    ctx: &VineCtx,
    skeleton: &mut Skeleton,
    cover: &mut Cover,
    base_node: u32,
    dir: Vec3,
    vigor: f32,
    level: u8,
    path: u64,
    stem: u32,
    max_len: f32,
    free: bool,
) {
    if skeleton.nodes.len() >= MAX_NODES {
        return;
    }
    let Some(sp) = stem_params(ctx.params, level) else {
        return;
    };
    let vine = ctx.vine;
    let support = &vine.support;
    let mut rng = ctx.rng.stream(path);
    let nominal = ctx.nominal(level);
    let drive = (vigor / nominal).clamp(0.0, MAX_DRIVE);
    let length_factor = (SUPPRESSION_FLOOR + (1.0 - SUPPRESSION_FLOOR) * drive).min(MAX_DRIVE);
    let stem_len = (sp.length
        * length_factor
        * range_f32(&mut rng, 1.0 - sp.length_variance, 1.0 + sp.length_variance).max(0.2))
    .min(max_len);
    let seg_count = ((stem_len / sp.segment_length.max(1e-3)).ceil() as usize).clamp(2, 4096);
    let seg_len = stem_len / seg_count as f32;
    // Held this far off the surface, which is where the centreline of a stem lying on
    // it is. The real radius is not known until the whole plant has grown.
    let hold = sp.radius.max(0.002);
    let adhesion = if free { 0.0 } else { vine.adhesion.clamp(0.0, 1.0) };
    let parent_stem = skeleton.nodes[base_node as usize].stem;

    let mut pos = skeleton.nodes[base_node as usize].position;
    let mut cur_dir = norm_or_up(dir);
    let mut frame = ortho_of(cur_dir);
    let mut bend = rand_perpendicular(&mut rng, cur_dir);
    let mut v = vigor;
    let mut cur = base_node;
    let mut slot: u32 = 0;
    let mut side_turn = 0u32;
    let mut pending: Vec<Pending> = Vec::new();
    let mut grown_len = 0.0f32;
    let mut crowded = 0.0f32;
    // Metres grown since the stem last held to its support.
    let mut adrift = 0.0f32;
    let can_spawn = level + 1 < ctx.levels_total;
    // How far along the stem a new shoot is due, in metres.
    let child_sp = &sp.children;
    let mut until_child = match child_sp.pattern {
        ChildPattern::Continuous { density } if density > 0.0 => {
            range_f32(&mut rng, 0.0, 1.0 / density)
        }
        _ => f32::INFINITY,
    };

    for seg in 0..seg_count {
        let along = seg as f32 / seg_count as f32;
        let d = support.distance(pos);
        let clinging = adhesion > 0.0 && d < vine.reach;
        let n = support.normal(pos);
        let sag = sp.gravity * (1.0 + sp.droop * along * along);

        let mut jitter = rand_perpendicular(&mut rng, cur_dir);
        if clinging {
            jitter = ortho_unit(along_surface(jitter, n), cur_dir);
        }
        bend = ortho_unit(bend * BEND_KEEP + jitter * (1.0 - BEND_KEEP), cur_dir);
        let wander = if clinging { sp.curvature } else { sp.curvature * FREE_WANDER };
        let mut goal = cur_dir + bend * wander;
        if clinging {
            // Up and down are read along the surface: a stem lying on a wall climbs it,
            // it does not lift off it toward the sky.
            let up = uphill(n);
            goal += up * (sp.phototropism - sag);
            if vine.twine != 0.0 {
                goal += n.cross(up.normalize_or(ortho_of(n))) * vine.twine;
            }
            goal = along_surface(goal, n).normalize_or(cur_dir);
            // Drawn back to where a stem lying on the surface sits, harder the further
            // it has strayed.
            let stray = ((d - hold) / vine.reach.max(1e-3)).clamp(-1.0, 1.0);
            goal -= n * (stray * adhesion);
        } else {
            goal += Vec3::Y * (sp.phototropism * FREE_LIFT - sag - FREE_SAG);
        }
        let turn = if clinging { MAX_TURN_PER_M } else { FREE_TURN_PER_M };
        let next = turn_toward(cur_dir, norm_or_up(goal), turn * seg_len);

        let mut step = pos + next * seg_len;
        // Never into the support, and never under the ground.
        let inside = support.distance(step);
        if inside < hold {
            step += support.normal(step) * (hold - inside);
        }
        step.y = step.y.max(hold);
        let moved = step - pos;
        if moved.length_squared() < 1e-10 {
            break;
        }
        let next = moved.normalize();
        frame = transport(cur_dir, next, frame);
        bend = ortho_unit(transport(cur_dir, next, bend), next);
        cur_dir = next;
        pos = step;

        // Ground another stem already holds shades this one out, a little at a time.
        if vine.crowding > 0.0 && level > 0 && cover.taken(pos, stem, parent_stem) {
            crowded += seg_len;
            if crowded > vine.crowding {
                break;
            }
        } else {
            crowded = (crowded - seg_len * 0.5).max(0.0);
        }

        // A clinging stem that has lost its support gives up after a while.
        if clinging || free {
            adrift = 0.0;
        } else {
            adrift += seg_len;
            if adhesion > 0.0 && vine.free_length > 0.0 && adrift > vine.free_length {
                break;
            }
        }

        let frac = (seg + 1) as f32 / seg_count as f32;
        grown_len += moved.length();
        cur = skeleton.push_node(Some(cur), pos, level, child_path(path, seg as u32), v, frac, stem);
        let here_n = support.normal(pos);
        let holding = adhesion > 0.0 && support.distance(pos) < vine.reach;
        if holding {
            skeleton.nodes[cur as usize].surface = here_n;
        }
        cover.hold(pos, stem);

        if can_spawn && frac >= child_sp.start_fraction && frac <= child_sp.end_fraction {
            let mut births = 0u32;
            match child_sp.pattern {
                ChildPattern::None => {}
                ChildPattern::Whorl { every, count } => {
                    if ((seg + 1) as u32).is_multiple_of(every.max(1)) {
                        births = count;
                    }
                }
                ChildPattern::Continuous { density } => {
                    until_child -= moved.length();
                    let spacing = 1.0 / density.max(1e-3);
                    while until_child <= 0.0 {
                        births += 1;
                        until_child += spacing * range_f32(&mut rng, 0.5, 1.5);
                    }
                }
            }
            for _ in 0..births.min(4) {
                slot += 1;
                side_turn += 1;
                let child_level = level + 1;
                let child_path_id = child_path(path, slot);
                let crotch = (child_sp.crotch_angle_deg
                    + range_f32(
                        &mut rng,
                        -child_sp.crotch_variance_deg,
                        child_sp.crotch_variance_deg,
                    ))
                .to_radians()
                .max(0.05);
                let spread =
                    1.0 + range_f32(&mut rng, -child_sp.scale_variance, child_sp.scale_variance);
                let child_vigor = v * (child_sp.scale * spread).max(0.02);
                let throw_out = !free
                    && holding
                    && vine.stand_off > 0.0
                    && rng.random::<f32>() < vine.stand_off;
                let dir = if holding {
                    let across = here_n.cross(cur_dir).normalize_or(ortho_of(cur_dir));
                    let side = shoot_side(
                        ctx,
                        child_level,
                        child_path_id,
                        side_turn,
                        cur_dir,
                        across,
                    );
                    let mut d = cur_dir * crotch.cos() + across * (side * crotch.sin());
                    if throw_out {
                        let lean = vine.stand_off_deg.to_radians();
                        d = d * lean.cos() + here_n * lean.sin();
                    }
                    norm_or_up(d)
                } else {
                    let az = range_f32(&mut rng, 0.0, std::f32::consts::TAU);
                    child_dir(cur_dir, frame, az, crotch)
                };
                pending.push(Pending {
                    attach: cur,
                    at_len: grown_len,
                    dir,
                    vigor: child_vigor,
                    level: child_level,
                    path: child_path_id,
                    free: free || throw_out,
                });
            }
        }

        v *= 1.0 - sp.vigor_falloff / seg_count as f32;
        if v < MIN_VIGOR * nominal {
            break;
        }
    }

    // As a tree's are: measured against what the stem really reached, and never out of
    // its last season.
    let tip_room = (grown_len / seg_count.max(1) as f32).max(1e-3) * 0.999;
    let end_at = grown_len * (child_sp.end_fraction + 1e-4);
    let reach = tip_reach(sp, level);
    for child in pending {
        if skeleton.nodes.len() >= MAX_NODES {
            return;
        }
        let ahead = grown_len - child.at_len;
        if ahead < tip_room || child.at_len > end_at {
            continue;
        }
        let share = reach.map(|reach| {
            let u = crate::growth::hash_unit(ctx.params.seed ^ AHEAD_SALT, child.path);
            reach * (1.0 - TIP_REACH_SPREAD * u)
        });
        let cap = lateral_cap(grown_len, child.at_len, share);
        let child_stem = skeleton.nodes.len() as u32;
        grow_stem(
            ctx,
            skeleton,
            cover,
            child.attach,
            child.dir,
            child.vigor,
            child.level,
            child.path,
            child_stem,
            cap,
            child.free,
        );
    }
}

/// Which side of its parent a shoot on a surface leaves from: +1 or -1 across it.
///
/// A level that climbs sends most of its shoots up the surface and one that trails
/// sends them down, the way ivy on a wall fills upward and a plant spilling from a
/// planter hangs its shoots toward the ground. A level that does neither alternates.
fn shoot_side(
    ctx: &VineCtx,
    level: u8,
    path: u64,
    turn: u32,
    parent_dir: Vec3,
    across: Vec3,
) -> f32 {
    let lean = stem_params(ctx.params, level).map_or(0.0, |sp: &StemParams| {
        sp.phototropism - sp.gravity
    });
    let alternate = if turn.is_multiple_of(2) { 1.0 } else { -1.0 };
    if lean.abs() < 1e-3 {
        return alternate;
    }
    // The side whose shoot points the way the level leans.
    let toward = if (across.y * lean) >= 0.0 { 1.0 } else { -1.0 };
    // A parent running straight up or down has no side that is higher than the other.
    if across.y.abs() < 0.05 && parent_dir.y.abs() > 0.9 {
        return alternate;
    }
    let u = crate::growth::hash_unit(ctx.params.seed ^ SIDE_SALT, path);
    if u < SIDE_BIAS { toward } else { -toward }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, IVY_RON, HANGING_VINE_RON, TWINING_VINE_RON};

    fn vines() -> Vec<(&'static str, SpeciesParams)> {
        [("ivy", IVY_RON), ("hanging", HANGING_VINE_RON), ("twining", TWINING_VINE_RON)]
            .into_iter()
            .map(|(name, src)| (name, parse_species(src).unwrap()))
            .collect()
    }

    #[test]
    fn the_distance_fields_measure_what_they_describe() {
        let wall = Support::Wall { width: 4.0, height: 3.0, thickness: 0.3 };
        assert!((wall.distance(Vec3::new(0.0, 1.0, 0.5)) - 0.5).abs() < 1e-5);
        assert!(wall.distance(Vec3::new(0.0, 1.0, -0.1)) < 0.0);
        assert!((wall.distance(Vec3::new(0.0, 3.2, -0.1)) - 0.2).abs() < 1e-5);
        assert!(wall.normal(Vec3::new(0.3, 1.0, 0.2)).dot(Vec3::Z) > 0.99);
        assert!(wall.normal(Vec3::new(0.3, 3.2, -0.1)).dot(Vec3::Y) > 0.99);

        let pillar = Support::Pillar { radius: 0.3, height: 4.0 };
        assert!((pillar.distance(Vec3::new(0.5, 1.0, 0.0)) - 0.2).abs() < 1e-5);
        assert!(pillar.distance(Vec3::new(0.1, 1.0, 0.0)) < 0.0);
        assert!(pillar.normal(Vec3::new(0.0, 2.0, 0.4)).dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn a_vine_never_grows_into_its_support_or_the_ground() {
        for (name, params) in vines() {
            let vine = params.vine.as_ref().unwrap();
            let sk = crate::grow(&params);
            assert!(sk.nodes.len() > 500, "{name}: only {} nodes", sk.nodes.len());
            for node in &sk.nodes[1..] {
                let d = vine.support.distance(node.position);
                assert!(d > -1e-3, "{name}: a node sits {d} m inside the support");
                assert!(node.position.y > -1e-3, "{name}: a node is under the ground");
            }
        }
    }

    #[test]
    fn ivy_lies_on_its_wall_and_covers_it() {
        let params = parse_species(IVY_RON).unwrap();
        let vine = params.vine.as_ref().unwrap();
        let sk = crate::grow(&params);
        let clinging = sk.nodes.iter().filter(|n| n.surface != Vec3::ZERO).count();
        assert!(
            clinging * 10 > sk.nodes.len() * 7,
            "only {clinging} of {} nodes hold to the wall",
            sk.nodes.len()
        );
        // Spread over the face rather than a column up the middle of it.
        let Support::Wall { width, height, .. } = vine.support else {
            panic!("the ivy is meant to climb a wall");
        };
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for n in &sk.nodes {
            lo = lo.min(n.position);
            hi = hi.max(n.position);
        }
        assert!(hi.x - lo.x > width * 0.6, "the ivy spans {} m of a {width} m wall", hi.x - lo.x);
        assert!(hi.y > height * 0.7, "the ivy reaches {} m up a {height} m wall", hi.y);
    }

    #[test]
    fn a_trailing_vine_hangs_down_its_face() {
        let params = parse_species(HANGING_VINE_RON).unwrap();
        let sk = crate::grow(&params);
        let root_y = sk.nodes[0].position.y;
        let low = sk
            .nodes
            .iter()
            .map(|n| n.position.y)
            .fold(f32::MAX, f32::min);
        assert!(root_y - low > 1.0, "the strands only hang {} m", root_y - low);
        // Nothing climbs far above where it was planted.
        let high = sk.nodes.iter().map(|n| n.position.y).fold(f32::MIN, f32::max);
        assert!(high - root_y < 0.5, "a strand climbed {} m over the ledge", high - root_y);
    }

    #[test]
    fn a_twining_vine_winds_round_its_pillar() {
        let params = parse_species(TWINING_VINE_RON).unwrap();
        let sk = crate::grow(&params);
        // The angle round the pillar the leader covers on its way up.
        let mut turned = 0.0f32;
        let mut prev: Option<f32> = None;
        for node in sk.nodes.iter().filter(|n| n.stem == TRUNK_STEM) {
            let a = node.position.x.atan2(node.position.z);
            if let Some(p) = prev {
                let mut d = a - p;
                if d > std::f32::consts::PI {
                    d -= std::f32::consts::TAU;
                } else if d < -std::f32::consts::PI {
                    d += std::f32::consts::TAU;
                }
                turned += d;
            }
            prev = Some(a);
        }
        assert!(turned.abs() > std::f32::consts::TAU, "the leader wound {turned} rad round its pillar");
    }

    #[test]
    fn a_vine_is_deterministic() {
        for (name, params) in vines() {
            let a = crate::grow(&params);
            let b = crate::grow(&params);
            assert_eq!(a.nodes.len(), b.nodes.len(), "{name}");
            for (x, y) in a.nodes.iter().zip(&b.nodes) {
                assert_eq!(x.position, y.position, "{name}");
            }
        }
    }

    #[test]
    fn ivy_leaves_face_out_from_the_wall() {
        let params = parse_species(IVY_RON).unwrap();
        let sk = crate::grow(&params);
        let leaves = crate::build_leaves(&sk, &params);
        assert!(leaves.leaf_count() > 1000, "only {} leaves", leaves.leaf_count());
        let facing = leaves
            .normals
            .chunks_exact(4)
            .filter(|n| Vec3::from(n[0]).z > 0.5)
            .count();
        assert!(
            facing * 10 > leaves.leaf_count() * 7,
            "{facing} of {} leaves face out from the wall",
            leaves.leaf_count()
        );
    }

    #[test]
    fn a_support_mesh_is_closed_off_and_well_formed() {
        for support in [
            Support::Wall { width: 3.0, height: 2.0, thickness: 0.25 },
            Support::Pillar { radius: 0.2, height: 3.0 },
        ] {
            let mesh = support.mesh();
            assert!(mesh.triangle_count() > 0);
            assert_eq!(mesh.normals.len(), mesh.vertex_count());
            assert_eq!(mesh.sway.len(), mesh.vertex_count());
            assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.vertex_count()));
        }
        assert_eq!(Support::Ground.mesh().triangle_count(), 0);
    }
}
