use std::cell::RefCell;

use glam::Vec3;
use rand::rngs::SmallRng;
use rand::Rng;

use crate::envelope::EnvelopeParams;
use crate::math::{ortho_of, ortho_unit, transport};
use crate::seed::{child_path, range_f32, TreeRng};
use crate::skeleton::Skeleton;
use crate::species::{ChildPattern, SpeciesParams, StemParams};

const MAX_NODES: usize = 150_000;
const MAX_SEGMENTS: usize = 512;
const MIN_VIGOR: f32 = 0.05;
const ROOT_PATH: u64 = 1;
const MIN_RADIUS: f32 = 0.004;
/// The root node and the trunk share a stem id so the trunk meshes as one
/// unbroken tube starting at the ground.
const TRUNK_STEM: u32 = 0;
/// Share of the bend direction re-rolled each segment. Keeping most of the previous
/// bend makes the deflection correlated, which reads as a smooth arc, not noise.
const BEND_WANDER: f32 = 0.22;
/// Hardest a stem may bend, in radians per metre grown. A bend radius of about 0.7 m:
/// slack enough that nothing in the presets reaches it, tight enough that no run of
/// influences can fold a stem back on itself over a single segment.
const MAX_TURN_PER_M: f32 = 1.5;
/// Total turning any one stem may bank over its whole length, in radians. Enough for
/// a limb to be bent into line by the crown and to sag under its own weight on the
/// way out, far short of the turn it would take to come round on itself.
const MAX_STEM_TURN: f32 = 1.0;
/// A fork takes this share of the parent vigor; the original stem keeps the rest.
const SPLIT_VIGOR: f32 = 0.72;
const SPLIT_KEEP: f32 = 0.86;
/// Length a stem keeps at zero vigor. Each level already declares its own `length`,
/// so letting vigor scale length outright would shorten deep twigs twice over and
/// collapse them into specks. Vigor modulates the declared length, it does not
/// replace it.
const LENGTH_FLOOR: f32 = 0.35;

struct GrowCtx<'a> {
    params: &'a SpeciesParams,
    env: EnvelopeParams,
    levels_total: u8,
    /// The leader centreline, appended to as the trunk climbs. The crown envelope is
    /// described around a vertical axis, so a trunk that leans would grow out of its
    /// own crown and everything on the upper trunk would be pruned at birth. Hanging
    /// the envelope off this instead makes the whole crown lean with the tree.
    leader: RefCell<Vec<Vec3>>,
}

impl GrowCtx<'_> {
    fn record_leader(&self, pos: Vec3) {
        self.leader.borrow_mut().push(pos);
    }

    /// Where the trunk sits horizontally at height `y`. Above the part of the trunk
    /// grown so far it holds at the last known point, which is what a branch just
    /// spawned there should see.
    fn crown_offset(&self, y: f32) -> Vec3 {
        let leader = self.leader.borrow();
        let Some(first) = leader.first() else {
            return Vec3::ZERO;
        };
        if y <= first.y {
            return horizontal(*first);
        }
        for pair in leader.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if y <= b.y {
                let span = b.y - a.y;
                let t = if span > 1e-5 { (y - a.y) / span } else { 0.0 };
                return horizontal(a.lerp(b, t.clamp(0.0, 1.0)));
            }
        }
        horizontal(*leader.last().expect("leader is not empty"))
    }
}

pub fn grow(params: &SpeciesParams) -> Skeleton {
    let mut skeleton = Skeleton::default();
    let root = skeleton.push_node(None, Vec3::ZERO, 0, ROOT_PATH, 1.0, 0.0, TRUNK_STEM);
    let ctx = GrowCtx {
        env: params.envelope.scaled(params.envelope_scale),
        levels_total: params
            .max_levels
            .min(params.branch_levels.len() as u8 + 1)
            .max(1),
        params,
        leader: RefCell::new(Vec::new()),
    };
    let tree_rng = TreeRng::new(params.seed);
    grow_stem(
        &ctx,
        &mut skeleton,
        &tree_rng,
        root,
        Vec3::Y,
        1.0,
        0,
        ROOT_PATH,
        0,
        TRUNK_STEM,
        Vec3::ZERO,
        MAX_STEM_TURN,
    );
    resolve_radii(params, &mut skeleton);
    skeleton
}

fn stem_params(params: &SpeciesParams, level: u8) -> Option<&StemParams> {
    if level == 0 {
        Some(&params.trunk)
    } else {
        params.branch_levels.get(level as usize - 1)
    }
}

#[allow(clippy::too_many_arguments)]
fn grow_stem(
    ctx: &GrowCtx,
    skeleton: &mut Skeleton,
    tree_rng: &TreeRng,
    base_node: u32,
    dir: Vec3,
    vigor: f32,
    level: u8,
    path: u64,
    split_depth: u32,
    stem: u32,
    // Normal of the flat plane this stem and its children lie in, or zero for a
    // stem that spreads in every direction.
    spray: Vec3,
    // Turning this stem starts with. A fork carries on with what its parent had left,
    // because it leaves at a shallow angle and reads as the same limb continuing; a
    // fresh budget at every fork would let a limb come round through them one fork at
    // a time. A child at the next level down is a new limb and starts full.
    budget: f32,
) {
    if skeleton.nodes.len() >= MAX_NODES {
        return;
    }
    let Some(sp) = stem_params(ctx.params, level) else {
        return;
    };
    let mut rng = tree_rng.stream(path);
    let base_pos = skeleton.nodes[base_node as usize].position;
    let drive = vigor.clamp(0.0, 1.6);
    let length_factor = (LENGTH_FLOOR + (1.0 - LENGTH_FLOOR) * drive).min(1.6);
    let stem_len = sp.length
        * length_factor
        * range_f32(&mut rng, 1.0 - sp.length_variance, 1.0 + sp.length_variance).max(0.2);
    let seg_count =
        ((stem_len / sp.segment_length.max(1e-3)).ceil() as usize).clamp(2, MAX_SEGMENTS);
    let seg_len = stem_len / seg_count as f32;
    let mut pos = base_pos;
    let mut cur_dir = norm_or_up(dir);
    // Carried along the stem so child azimuths stay fixed relative to the stem
    // itself rather than to the world axes.
    let mut frame = ortho_of(cur_dir);
    let mut bend = rand_perpendicular(&mut rng, cur_dir);
    let mut v = vigor;
    let mut cur = base_node;
    let mut azimuth = range_f32(&mut rng, 0.0, std::f32::consts::TAU);
    let mut slot: u32 = 0;
    let can_spawn = level + 1 < ctx.levels_total;
    // The leader keeps its own course, but a co-dominant fork is part of the crown
    // and has to be shaped by the envelope like any branch, or the tree grows as a
    // bundle of parallel poles.
    let shaped_by_envelope = level > 0 || split_depth > 0;
    let is_leader = level == 0 && split_depth == 0;
    // How much turning this stem has left to spend, across every influence that bends
    // it. Without a ceiling the crown pull alone settles into an orbit: whatever its
    // strength, there is a radius at which it supplies exactly the turn a circle
    // needs, and the stem rides it round.
    let mut turn_budget = budget.max(0.0);

    for seg in 0..seg_count {
        let mut jitter = rand_perpendicular(&mut rng, cur_dir);
        if spray != Vec3::ZERO {
            // A stem that belongs to a flat spray has to wander within it, or the
            // plane it was placed in dissolves over a few segments.
            jitter = flatten_into(jitter, spray, 1.0);
        }
        bend = ortho_unit(bend * (1.0 - BEND_WANDER) + jitter * BEND_WANDER, cur_dir);
        let next_dir = steer(
            ctx,
            sp,
            cur_dir,
            pos,
            bend,
            seg_len,
            shaped_by_envelope,
            &mut turn_budget,
        );
        frame = transport(cur_dir, next_dir, frame);
        bend = transport(cur_dir, next_dir, bend);
        cur_dir = next_dir;
        pos += cur_dir * seg_len;

        // Anything the envelope steers, it also prunes. Steering a stem that can
        // never be cut just leaves it circling the crown boundary forever.
        if shaped_by_envelope
            && ctx.env.density(pos - ctx.crown_offset(pos.y)) < ctx.params.envelope.kill_threshold
        {
            // Pruned before it grew at all means this stem was born outside the
            // crown. On the bare lower trunk of a conifer those are the dead stubs,
            // so leave one behind rather than nothing.
            if seg == 0 && sp.dead_stub_length > 0.0 {
                let stub = base_pos + cur_dir * sp.dead_stub_length;
                skeleton.push_node(Some(cur), stub, level, child_path(path, 0), v, 1.0, stem);
            }
            break;
        }

        let frac = (seg + 1) as f32 / seg_count as f32;
        let seg_path = child_path(path, seg as u32);
        cur = skeleton.push_node(Some(cur), pos, level, seg_path, v, frac, stem);
        if is_leader {
            // Recorded before anything spawns here, so branches born at this height
            // already see the trunk they are hanging off.
            ctx.record_leader(pos);
        }

        if can_spawn {
            spawn_children(
                ctx,
                skeleton,
                tree_rng,
                &mut rng,
                cur,
                cur_dir,
                frame,
                v,
                sp,
                level + 1,
                seg,
                seg_count,
                &mut azimuth,
                &mut slot,
                path,
                spray,
            );
        }

        if can_spawn
            && split_depth < ctx.params.max_split_depth
            && sp.split_probability > 0.0
            && frac >= sp.split_start_fraction
            && skeleton.nodes.len() < MAX_NODES
            && rng.random::<f32>() < sp.split_probability
        {
            slot += 1;
            let az = azimuth + std::f32::consts::FRAC_PI_2 + range_f32(&mut rng, -0.6, 0.6);
            let spread = sp.split_angle_deg.to_radians().max(0.02);
            let crotch = spread * range_f32(&mut rng, 0.7, 1.3);
            let split_dir = child_dir(cur_dir, frame, az, crotch);
            let p = child_path(path, slot);
            let fork_stem = skeleton.nodes.len() as u32;
            grow_stem(
                ctx,
                skeleton,
                tree_rng,
                cur,
                split_dir,
                v * SPLIT_VIGOR,
                level,
                p,
                split_depth + 1,
                fork_stem,
                spray,
                turn_budget,
            );
            // The parent gives up part of its drive to the fork instead of both
            // halves carrying on at full strength.
            v *= SPLIT_KEEP;
        }

        v *= 1.0 - sp.vigor_falloff / seg_count as f32;
        if v < MIN_VIGOR {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_children(
    ctx: &GrowCtx,
    skeleton: &mut Skeleton,
    tree_rng: &TreeRng,
    rng: &mut SmallRng,
    attach: u32,
    dir: Vec3,
    frame: Vec3,
    vigor: f32,
    sp: &StemParams,
    child_level: u8,
    seg: usize,
    seg_count: usize,
    azimuth: &mut f32,
    slot: &mut u32,
    stem_path: u64,
    spray: Vec3,
) {
    let child = &sp.children;
    let frac = (seg + 1) as f32 / seg_count as f32;
    if frac < child.start_fraction || frac > child.end_fraction {
        return;
    }
    if skeleton.nodes.len() >= MAX_NODES {
        return;
    }

    let spawn_one = |az: f32, skeleton: &mut Skeleton, slot: &mut u32, rng: &mut SmallRng| {
        *slot += 1;
        // Children leave at a shallower or steeper angle depending how far along the
        // parent they are, so a stem does not carry every child at one fixed angle.
        // Weighted hard toward the tip: a conifer holds its branches out level
        // through the body of the crown and only the last whorls stand up, so a
        // straight blend would tilt the whole crown and flatten away its layers.
        let t = frac.clamp(0.0, 1.0);
        let along = child.crotch_angle_deg
            + (child.crotch_angle_tip_deg - child.crotch_angle_deg) * t * t * t;
        let crotch_deg =
            along + range_f32(rng, -child.crotch_variance_deg, child.crotch_variance_deg);
        let mut d = child_dir(dir, frame, az, crotch_deg.to_radians().max(0.02));
        // Inside a spray the child is pulled into the plane; the first branch off a
        // stem that has no plane is the one that sets it for everything below.
        let child_spray = if spray != Vec3::ZERO {
            d = flatten_into(d, spray, child.planarity);
            spray
        } else if child.planarity > 0.0 {
            plane_of(d)
        } else {
            Vec3::ZERO
        };
        let drive = child.scale * (1.0 + range_f32(rng, -child.scale_variance, child.scale_variance));
        let p = child_path(stem_path, *slot);
        let child_stem = skeleton.nodes.len() as u32;
        grow_stem(
            ctx,
            skeleton,
            tree_rng,
            attach,
            d,
            vigor * drive.max(0.02),
            child_level,
            p,
            0,
            child_stem,
            child_spray,
            MAX_STEM_TURN,
        );
    };

    match child.pattern {
        ChildPattern::None => {}
        ChildPattern::Whorl { every, count } => {
            let every = every.max(1);
            if !((seg + 1) as u32).is_multiple_of(every) {
                return;
            }
            // Real whorls are not all the same size, and a node that draws an empty
            // one leaves a gap: without that every node carries a whorl and the
            // trunk comes out looking like a ladder.
            let spread = child.count_variance as f32;
            let count = ((count as f32 + range_f32(rng, -spread, spread + 1.0).floor()).round()
                as i32)
                .max(0) as u32;
            if count == 0 {
                return;
            }
            // Roll the whole whorl on so successive whorls do not stack up in the
            // same vertical planes.
            *azimuth += child.phyllotaxis_deg.to_radians();
            for k in 0..count {
                let az = *azimuth
                    + (k as f32) * std::f32::consts::TAU / count as f32
                    + range_f32(rng, -child.roll_variance_deg, child.roll_variance_deg).to_radians();
                spawn_one(az, skeleton, slot, rng);
            }
        }
        ChildPattern::Continuous { density } => {
            *azimuth += child.phyllotaxis_deg.to_radians();
            if rng.random::<f32>() < density {
                let az = *azimuth
                    + range_f32(rng, -child.roll_variance_deg, child.roll_variance_deg).to_radians();
                spawn_one(az, skeleton, slot, rng);
            }
        }
    }
}

fn child_dir(forward: Vec3, frame: Vec3, azimuth: f32, crotch: f32) -> Vec3 {
    let f = norm_or_up(forward);
    let u = ortho_unit(frame, f);
    let v = f.cross(u);
    let lateral = u * azimuth.cos() + v * azimuth.sin();
    norm_or_up(f * crotch.cos() + lateral * crotch.sin())
}

#[allow(clippy::too_many_arguments)]
fn steer(
    ctx: &GrowCtx,
    sp: &StemParams,
    dir: Vec3,
    pos: Vec3,
    bend: Vec3,
    seg_len: f32,
    shaped_by_envelope: bool,
    // Turning this stem has left to spend, drawn down by whatever it uses.
    turn_budget: &mut f32,
) -> Vec3 {
    let mut d = dir
        + Vec3::Y * sp.phototropism * ctx.params.phototropism_multiplier
        - Vec3::Y * sp.gravity * ctx.params.gravity_multiplier;
    d += bend * sp.curvature;
    let mut d = norm_or_up(d);

    if shaped_by_envelope {
        // The envelope is described around the origin, so growth is measured in that
        // space and the answer brought back to where the crown actually sits.
        let crown_offset = ctx.crown_offset(pos.y);
        let local = pos - crown_offset;
        let dens = ctx.env.density(local);
        if dens < 1.0 {
            let target = ctx.env.steer_target(local) + crown_offset;
            let inward = horizontal(target - pos);
            let error = inward.length();
            if error > 1e-4 {
                let inward = inward / error;
                // Only a stem on its way out of the crown is worth turning back. A
                // pull that keeps acting once the stem has come round is a fixed
                // force toward a fixed point on something travelling at fixed speed,
                // which is an orbit: the stem circles the axis instead of settling.
                // Fading the correction out as it turns inward leaves it tracking the
                // crown surface, which is the shaping that was wanted.
                let heading = horizontal(d).normalize_or_zero();
                let leaving = (0.5 - 0.5 * inward.dot(heading)).clamp(0.0, 1.0);
                // Measured per metre grown rather than per segment, so a level with
                // short segments is not steered several times harder than one with
                // long segments for the same setting.
                let turn = ctx.env.pull_strength * (1.0 - dens) * leaving * seg_len;
                d = turn_toward(d, inward, turn);
            }
        }
    }
    // Every influence above is a nudge per segment, so the turn a stem banks grows
    // with the number of segments it is cut into: a long limb bends further than a
    // short one on the same settings, without limit, until it comes round on itself.
    // Spending from a budget is what stops that. A stem bends while it has turning
    // left and runs on straight once it has not, which is also how a real limb reads:
    // shaped near the trunk, committed to a direction further out.
    let limit = (MAX_TURN_PER_M * seg_len).min(*turn_budget);
    let out = turn_toward(dir, d, limit);
    *turn_budget -= dir.dot(out).clamp(-1.0, 1.0).acos();
    out
}

/// Rotates `dir` toward `goal` by `angle` radians, stopping at `goal` rather than
/// turning past it. Rotating, rather than adding a vector and renormalising, is what
/// makes `angle` mean the same thing however far apart the two directions are, and
/// that is what lets the callers bound it.
fn turn_toward(dir: Vec3, goal: Vec3, angle: f32) -> Vec3 {
    if angle <= 0.0 {
        return dir;
    }
    let cos = dir.dot(goal).clamp(-1.0, 1.0);
    let remaining = cos.acos();
    if remaining < 1e-5 {
        return dir;
    }
    if angle >= remaining {
        return goal;
    }
    // The part of `goal` lying across `dir`: the two span the plane the rotation
    // happens in, so the turn is a plain rotation within it.
    let across = goal - dir * cos;
    let Some(across) = across.try_normalize() else {
        return dir;
    };
    norm_or_up(dir * angle.cos() + across * angle.sin())
}

/// Normal of the flat plane that contains `d` and lies as level as possible. A limb
/// growing out from the trunk carries its spray in this plane.
fn plane_of(d: Vec3) -> Vec3 {
    let across = Vec3::Y.cross(d);
    if across.length_squared() < 1e-6 {
        // Straight up or down has no level plane to pick.
        return Vec3::ZERO;
    }
    d.cross(across.normalize()).normalize_or_zero()
}

/// Pulls `dir` toward the plane with normal `plane`, by `amount` from 0 to 1.
fn flatten_into(dir: Vec3, plane: Vec3, amount: f32) -> Vec3 {
    if plane == Vec3::ZERO || amount <= 0.0 {
        return dir;
    }
    let in_plane = dir - plane * dir.dot(plane);
    if in_plane.length_squared() < 1e-8 {
        return dir;
    }
    dir.lerp(in_plane.normalize(), amount.clamp(0.0, 1.0))
        .normalize_or(dir)
}

/// The horizontal part of a position: how far the trunk has wandered from the axis
/// the envelope is described around.
fn horizontal(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

fn norm_or_up(v: Vec3) -> Vec3 {
    let len = v.length();
    if len > 1e-8 { v / len } else { Vec3::Y }
}

fn rand_perpendicular(rng: &mut SmallRng, dir: Vec3) -> Vec3 {
    let a = ortho_of(dir);
    let b = dir.cross(a);
    let ang = range_f32(rng, 0.0, std::f32::consts::TAU);
    a * ang.cos() + b * ang.sin()
}

/// Thickness is resolved in two passes: a top-down pass giving every stem a base
/// radius that tapers along its own length, then a bottom-up pass that widens any
/// node carrying more cross-section than its taper alone would provide.
fn resolve_radii(params: &SpeciesParams, skeleton: &mut Skeleton) {
    let n = skeleton.nodes.len();
    if n == 0 {
        return;
    }
    let mut radii = vec![0.0f32; n];
    // The untapered radius at the foot of the stem each node belongs to.
    let mut stem_base = vec![0.0f32; n];
    for i in 0..n {
        let (parent, level, frac, stem, vigor) = {
            let node = &skeleton.nodes[i];
            (
                node.parent,
                node.level,
                node.stem_fraction,
                node.stem,
                node.vigor,
            )
        };
        let lp = stem_params(params, level).expect("level params exist for pushed node");
        let base = match parent {
            None => params.trunk.radius,
            Some(p) => {
                if skeleton.nodes[p as usize].stem == stem {
                    // Same stem: share the base the whole run was given. Taper alone
                    // thins it, so the ratio is not compounded once per segment.
                    stem_base[p as usize]
                } else {
                    // A new stem is sized by how much growth it actually carries -
                    // its own level radius scaled by the vigor it started with, the
                    // same quantity that sets its length. The parent ratio is only a
                    // ceiling, so a short twig on a thick limb stays a twig.
                    let own = lp.radius * vigor.clamp(0.0, 1.6);
                    let ceiling = lp.radius_ratio.clamp(0.05, 0.95) * radii[p as usize];
                    own.min(ceiling).max(MIN_RADIUS)
                }
            }
        };
        stem_base[i] = base;
        let t = frac.clamp(0.0, 1.0);
        radii[i] = (base * (1.0 - (1.0 - lp.taper.clamp(0.0, 1.0)) * t)).max(MIN_RADIUS);
    }

    let mut sums = vec![0.0f64; n];
    for i in (0..n).rev() {
        if sums[i] > 0.0 {
            let exp = da_vinci_exp(params, skeleton.nodes[i].level);
            radii[i] = radii[i].max((sums[i] as f32).powf(1.0 / exp));
        }
        if let Some(p) = skeleton.nodes[i].parent {
            let p_exp = da_vinci_exp(params, skeleton.nodes[p as usize].level);
            sums[p as usize] += (radii[i] as f64).powf(p_exp as f64);
        }
    }

    for (i, node) in skeleton.nodes.iter_mut().enumerate() {
        node.radius = radii[i];
    }
}

fn da_vinci_exp(params: &SpeciesParams, level: u8) -> f32 {
    stem_params(params, level)
        .map(|s| s.da_vinci_exponent.max(1.0))
        .unwrap_or(2.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, OAK_RON, PINE_RON};

    #[test]
    fn trunk_keeps_its_thickness_up_the_stem() {
        // The taper ratio applies once over a whole stem. Compounding it per segment
        // instead shrinks the trunk to the radius floor within a couple of metres.
        for src in [OAK_RON, PINE_RON] {
            let params = parse_species(src).unwrap();
            let sk = grow(&params);
            let trunk_stem = sk.nodes[0].stem;
            let mid = sk
                .nodes
                .iter()
                .filter(|n| n.stem == trunk_stem)
                .min_by(|a, b| {
                    (a.stem_fraction - 0.5)
                        .abs()
                        .total_cmp(&(b.stem_fraction - 0.5).abs())
                })
                .expect("trunk has nodes");
            assert!(
                mid.radius > params.trunk.radius * 0.4,
                "{}: trunk is {} at half its length, base is {}",
                params.name,
                mid.radius,
                params.trunk.radius
            );
        }
    }

    #[test]
    fn trunk_radius_decreases_monotonically() {
        let params = parse_species(OAK_RON).unwrap();
        let sk = grow(&params);
        let trunk_stem = sk.nodes[0].stem;
        let mut run: Vec<&crate::skeleton::SkeletonNode> =
            sk.nodes.iter().filter(|n| n.stem == trunk_stem).collect();
        run.sort_by(|a, b| a.stem_fraction.total_cmp(&b.stem_fraction));
        for pair in run.windows(2) {
            assert!(
                pair[1].radius <= pair[0].radius + 1e-4,
                "trunk widens going up: {} then {}",
                pair[0].radius,
                pair[1].radius
            );
        }
    }

    #[test]
    fn low_vigor_modulates_stem_length_instead_of_erasing_it() {
        // Each level already declares its own length, so vigor may only modulate it.
        // Multiplying length by vigor as well shortens a stem once per level, and by
        // the deepest level that compounds into centimetre-long specks.
        //
        // Built from a bare species rather than a preset so the only thing varying is
        // vigor: one branch level, a crown big enough that nothing is pruned, and
        // children handed a fraction of what drives the trunk.
        let mut params = SpeciesParams {
            max_levels: 2,
            max_split_depth: 0,
            ..Default::default()
        };
        params.envelope.volumes = vec![crate::envelope::EnvelopeVolume::Ellipsoid {
            center: [0.0, 0.0, 0.0],
            radii: [500.0, 500.0, 500.0],
        }];
        params.trunk.split_probability = 0.0;
        params.trunk.children.pattern = ChildPattern::Continuous { density: 1.0 };
        // Low, but above MIN_VIGOR: below it a stem is cut off after one segment,
        // which is a separate rule and would mask what is being measured here.
        params.trunk.children.scale = 0.1;
        params.trunk.vigor_falloff = 0.0;
        let branch = &mut params.branch_levels[0];
        branch.length = 4.0;
        branch.length_variance = 0.0;
        branch.vigor_falloff = 0.0;
        branch.split_probability = 0.0;

        let declared = params.branch_levels[0].length;
        let sk = grow(&params);
        let lengths: Vec<f32> = sk
            .stem_runs()
            .iter()
            .filter(|run| sk.nodes[run[0] as usize].level == 1)
            .map(|run| {
                // A stem starts at its attachment point, which belongs to the parent
                // run, so that first segment counts toward its length.
                let first = &sk.nodes[run[0] as usize];
                let from_parent = first
                    .parent
                    .map(|p| (first.position - sk.nodes[p as usize].position).length())
                    .unwrap_or(0.0);
                from_parent
                    + run
                        .windows(2)
                        .map(|w| {
                            (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position)
                                .length()
                        })
                        .sum::<f32>()
            })
            .collect();

        assert!(lengths.len() > 5, "expected branches, got {}", lengths.len());
        let shortest = lengths.iter().copied().fold(f32::MAX, f32::min);
        // Vigor here is near zero, so every stem should sit at the floor. Scaling
        // length by vigor directly would put them near zero instead.
        assert!(
            shortest >= declared * LENGTH_FLOOR - 1e-3,
            "a branch at minimum vigor is {shortest} of a declared {declared}"
        );
        assert!(
            shortest <= declared + 1e-3,
            "vigor should not lengthen a stem past what it declares: {shortest}"
        );
    }

    #[test]
    fn a_stem_is_one_unbroken_run_of_nodes() {
        let params = parse_species(OAK_RON).unwrap();
        let sk = grow(&params);
        for run in sk.stem_runs() {
            for w in run.windows(2) {
                let child = &sk.nodes[w[1] as usize];
                assert_eq!(
                    child.parent,
                    Some(w[0]),
                    "stem {} is not a chain: {} does not follow {}",
                    child.stem,
                    w[1],
                    w[0]
                );
            }
        }
    }

    #[test]
    fn same_seed_gives_identical_skeleton() {
        let params = parse_species(PINE_RON).unwrap();
        let a = grow(&params);
        let b = grow(&params);
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.position, nb.position);
            assert_eq!(na.radius, nb.radius);
        }
    }

    #[test]
    fn different_seed_changes_tree() {
        let mut a = parse_species(PINE_RON).unwrap();
        let mut b = parse_species(PINE_RON).unwrap();
        a.seed = 42;
        b.seed = 43;
        let ta = grow(&a);
        let tb = grow(&b);
        let pa: Vec<_> = ta.nodes.iter().map(|n| n.position).collect();
        let pb: Vec<_> = tb.nodes.iter().map(|n| n.position).collect();
        assert_ne!(pa, pb, "different seeds should produce different trees");
    }

    #[test]
    fn child_radius_never_exceeds_parent() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = grow(&params);
        for node in &sk.nodes {
            if let Some(p) = node.parent {
                let parent_radius = sk.nodes[p as usize].radius;
                assert!(
                    node.radius <= parent_radius + 1e-4,
                    "child {} r={} > parent {} r={}",
                    node.path,
                    node.radius,
                    parent_radius,
                    sk.nodes[p as usize].radius
                );
            }
        }
    }

    #[test]
    fn da_vinci_sum_holds_at_junctions() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = grow(&params);
        for node in &sk.nodes {
            if node.children.is_empty() {
                continue;
            }
            let exp = da_vinci_exp(&params, node.level);
            let lhs = node.radius.powf(exp);
            let sum: f32 = node
                .children
                .iter()
                .map(|c| sk.nodes[*c as usize].radius.powf(exp))
                .sum();
            assert!(
                lhs + 1e-3 >= sum,
                "node {}: r^e {} < sum child r^e {}",
                node.path,
                lhs,
                sum
            );
        }
    }

    #[test]
    fn branch_nodes_stay_inside_envelope() {
        // The crown hangs off the trunk rather than the world axis, so a node is
        // judged against the crown anchored somewhere on the trunk at or below it.
        // That set of anchors is exactly what was available while it grew: the trunk
        // is only known up to the height reached so far.
        for src in [PINE_RON, OAK_RON] {
            let params = parse_species(src).unwrap();
            let env = params.envelope.scaled(params.envelope_scale);
            let sk = grow(&params);
            let trunk_stem = sk.nodes[0].stem;
            let mut leader: Vec<Vec3> = sk
                .nodes
                .iter()
                .filter(|n| n.stem == trunk_stem)
                .map(|n| n.position)
                .collect();
            leader.sort_by(|a, b| a.y.total_cmp(&b.y));

            // A dead stub is deliberately outside the crown: it is what is left of a
            // branch the crown pruned the moment it appeared.
            let mut stubs: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for run in sk.stem_runs() {
                if run.len() != 1 {
                    continue;
                }
                let node = &sk.nodes[run[0] as usize];
                let Some(parent) = node.parent else { continue };
                let expected = stem_params(&params, node.level)
                    .map(|sp| sp.dead_stub_length)
                    .unwrap_or(0.0);
                let reach = (node.position - sk.nodes[parent as usize].position).length();
                if expected > 0.0 && (reach - expected).abs() < 1e-3 {
                    stubs.insert(run[0]);
                }
            }

            let mut checked = 0;
            for (i, node) in sk.nodes.iter().enumerate() {
                if node.level == 0 || stubs.contains(&(i as u32)) {
                    continue;
                }
                // Growth interpolates along the trunk, so the anchors it could have
                // used are the whole polyline below the node, not just its vertices.
                let mut best = env.density(node.position);
                for pair in leader.windows(2) {
                    // Fine enough that the sampled anchor matches the continuous one
                    // growth used to within far less than the crown falloff.
                    for k in 0..=64 {
                        let p = pair[0].lerp(pair[1], k as f32 / 64.0);
                        if p.y > node.position.y + 1e-4 {
                            continue;
                        }
                        best = best.max(env.density(node.position - horizontal(p)));
                    }
                }
                assert!(
                    best >= params.envelope.kill_threshold - 1e-4,
                    "{}: node at {:?} is outside every crown the trunk offers, best {best}",
                    params.name,
                    node.position
                );
                checked += 1;
            }
            assert!(checked > 100, "{}: only {checked} nodes", params.name);
        }
    }

    /// Total length of a stem, counting the segment from its attachment point.
    fn stem_length(sk: &Skeleton, run: &[u32]) -> f32 {
        let first = &sk.nodes[run[0] as usize];
        let from_parent = first
            .parent
            .map(|p| (first.position - sk.nodes[p as usize].position).length())
            .unwrap_or(0.0);
        from_parent
            + run
                .windows(2)
                .map(|w| {
                    (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length()
                })
                .sum::<f32>()
    }

    #[test]
    fn a_spray_keeps_its_branchlets_in_one_plane() {
        // Conifer branchlets grow in the flat plane of the limb carrying them, and
        // those plates are most of what gives a fir its layered look. Without it the
        // branchlets spiral around the limb and every spray reads as a bottle brush.
        let params = parse_species(PINE_RON).unwrap();
        assert!(
            params.branch_levels[0].children.planarity > 0.5,
            "this preset should use flat sprays"
        );
        let sk = grow(&params);

        // Each limb established its plane from the direction it left the trunk on.
        let mut limb_plane: std::collections::HashMap<u32, Vec3> = std::collections::HashMap::new();
        for run in sk.stem_runs() {
            let first = &sk.nodes[run[0] as usize];
            if first.level != 1 {
                continue;
            }
            let Some(parent) = first.parent else { continue };
            // Only a limb leaving the trunk sets a plane. A fork of a limb is also
            // level 1 but inherits the plane it was already growing in, so deriving
            // one from its own direction would be the wrong plane to judge against.
            if sk.nodes[parent as usize].level != 0 {
                continue;
            }
            let dir = (first.position - sk.nodes[parent as usize].position).normalize_or(Vec3::Y);
            limb_plane.insert(first.stem, plane_of(dir));
        }

        let mut checked = 0;
        let mut worst: f32 = 0.0;
        for run in sk.stem_runs() {
            let first = &sk.nodes[run[0] as usize];
            if first.level != 2 {
                continue;
            }
            let Some(parent) = first.parent else { continue };
            let Some(&plane) = limb_plane.get(&sk.nodes[parent as usize].stem) else {
                continue;
            };
            if plane == Vec3::ZERO {
                continue;
            }
            let dir = (first.position - sk.nodes[parent as usize].position).normalize_or(Vec3::Y);
            worst = worst.max(dir.dot(plane).abs());
            checked += 1;
        }
        assert!(checked > 100, "only {checked} branchlets");
        assert!(
            worst < 0.35,
            "a branchlet leaves its spray plane by {worst}, so the sprays are not flat"
        );
    }

    #[test]
    fn a_bare_lower_trunk_keeps_its_dead_stubs() {
        // Branches born below the crown are pruned the moment they appear. Leaving a
        // short stub behind is what puts the dead branch remnants on the bare lower
        // trunk of a conifer instead of a clean pole.
        let params = parse_species(PINE_RON).unwrap();
        let stub_len = params.branch_levels[0].dead_stub_length;
        assert!(stub_len > 0.0, "this preset should keep stubs");
        let crown_base = match params.envelope.volumes.first() {
            Some(crate::envelope::EnvelopeVolume::Cone { base_y, .. }) => *base_y,
            other => panic!("expected a cone crown, got {other:?}"),
        };

        let sk = grow(&params);
        let stubs = sk
            .stem_runs()
            .iter()
            .filter(|run| {
                let n = &sk.nodes[run[0] as usize];
                run.len() == 1 && n.level == 1 && n.position.y < crown_base
            })
            .count();
        assert!(stubs > 5, "only {stubs} dead stubs below the crown");
    }

    #[test]
    fn siblings_in_a_whorl_get_different_lengths() {
        // Without a spread on the drive handed to children, every branch in a whorl
        // gets the same vigor and so the same length, which reads as a wheel of
        // identical spokes rather than a tree.
        fn whorl_lengths(scale_variance: f32) -> Vec<f32> {
            let mut params = SpeciesParams {
                max_levels: 2,
                max_split_depth: 0,
                ..Default::default()
            };
            params.envelope.volumes = vec![crate::envelope::EnvelopeVolume::Ellipsoid {
                center: [0.0, 0.0, 0.0],
                radii: [500.0, 500.0, 500.0],
            }];
            params.trunk.split_probability = 0.0;
            params.trunk.children.pattern = ChildPattern::Whorl {
                every: 4,
                count: 5,
            };
            params.trunk.children.scale_variance = scale_variance;
            params.branch_levels[0].length_variance = 0.0;
            params.branch_levels[0].split_probability = 0.0;

            let sk = grow(&params);
            // One whorl: the branches sharing the lowest attachment point.
            let mut runs: Vec<(u32, f32)> = sk
                .stem_runs()
                .iter()
                .filter(|r| sk.nodes[r[0] as usize].level == 1)
                .filter_map(|r| {
                    sk.nodes[r[0] as usize]
                        .parent
                        .map(|p| (p, stem_length(&sk, r)))
                })
                .collect();
            runs.sort_by_key(|(p, _)| *p);
            let first_attach = runs.first().expect("branches exist").0;
            runs.iter()
                .filter(|(p, _)| *p == first_attach)
                .map(|(_, len)| *len)
                .collect()
        }

        let uniform = whorl_lengths(0.0);
        assert!(uniform.len() >= 3, "expected a whorl, got {uniform:?}");
        let spread = |v: &[f32]| {
            let hi = v.iter().copied().fold(f32::MIN, f32::max);
            let lo = v.iter().copied().fold(f32::MAX, f32::min);
            hi - lo
        };
        assert!(
            spread(&uniform) < 1e-3,
            "without a spread a whorl should be uniform, got {uniform:?}"
        );

        let varied = whorl_lengths(0.5);
        assert!(
            spread(&varied) > 0.2 * uniform[0],
            "a spread of 0.5 barely changed the whorl: {varied:?}"
        );
    }

    #[test]
    fn the_crown_follows_a_leaning_trunk() {
        // A trunk that drifts sideways used to leave its own crown behind: branches
        // near the top were measured against a cone still centred on the world axis,
        // so they were pruned the moment they were born and the leader came out bare.
        let params = parse_species(PINE_RON).unwrap();
        let env = params.envelope.scaled(params.envelope_scale);
        let sk = grow(&params);

        let trunk_stem = sk.nodes[0].stem;
        let top = sk
            .nodes
            .iter()
            .filter(|n| n.stem == trunk_stem)
            .max_by(|a, b| a.position.y.total_cmp(&b.position.y))
            .expect("trunk has nodes");
        let drift = horizontal(top.position).length();
        assert!(drift > 0.5, "this preset should lean; drift is only {drift}");
        assert!(
            env.density(top.position) < params.envelope.kill_threshold,
            "the leaning top should be outside a world-centred crown for this to mean anything"
        );

        // Branches have to reach the upper trunk, where the old behaviour left a gap.
        let upper = sk
            .nodes
            .iter()
            .filter(|n| n.level > 0 && n.position.y > top.position.y - 2.0)
            .count();
        assert!(upper > 20, "only {upper} branch nodes near the leaning top");
    }

    #[test]
    fn level_cap_is_respected() {
        let mut params = parse_species(PINE_RON).unwrap();
        params.max_levels = 2;
        let sk = grow(&params);
        for node in &sk.nodes {
            assert!(node.level <= 1, "found level {}", node.level);
        }
    }
}
