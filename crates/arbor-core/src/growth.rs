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
/// A fork takes this share of the parent vigor; the original stem keeps the rest.
const SPLIT_VIGOR: f32 = 0.72;
const SPLIT_KEEP: f32 = 0.86;

struct GrowCtx<'a> {
    params: &'a SpeciesParams,
    env: EnvelopeParams,
    levels_total: u8,
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
) {
    if skeleton.nodes.len() >= MAX_NODES {
        return;
    }
    let Some(sp) = stem_params(ctx.params, level) else {
        return;
    };
    let mut rng = tree_rng.stream(path);
    let base_pos = skeleton.nodes[base_node as usize].position;
    let vigor_clamped = vigor.clamp(0.1, 1.6);
    let stem_len = sp.length
        * vigor_clamped
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

    for seg in 0..seg_count {
        let jitter = rand_perpendicular(&mut rng, cur_dir);
        bend = ortho_unit(bend * (1.0 - BEND_WANDER) + jitter * BEND_WANDER, cur_dir);
        let next_dir = steer(ctx, sp, cur_dir, pos, level, bend);
        frame = transport(cur_dir, next_dir, frame);
        bend = transport(cur_dir, next_dir, bend);
        cur_dir = next_dir;
        pos += cur_dir * seg_len;

        if level > 0 && ctx.env.density(pos) < ctx.params.envelope.kill_threshold {
            break;
        }

        let frac = (seg + 1) as f32 / seg_count as f32;
        let seg_path = child_path(path, seg as u32);
        cur = skeleton.push_node(Some(cur), pos, level, seg_path, v, frac, stem);

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
            );
        }

        if can_spawn
            && split_depth < ctx.params.max_split_depth
            && sp.split_probability > 0.0
            && skeleton.nodes.len() < MAX_NODES
            && rng.random::<f32>() < sp.split_probability
        {
            slot += 1;
            let az = azimuth + 1.5708 + range_f32(&mut rng, -0.6, 0.6);
            let crotch = 0.21 + range_f32(&mut rng, 0.0, 0.35);
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
        let crotch_deg = child.crotch_angle_deg
            + range_f32(rng, -child.crotch_variance_deg, child.crotch_variance_deg);
        let d = child_dir(dir, frame, az, crotch_deg.to_radians().max(0.02));
        let p = child_path(stem_path, *slot);
        let child_stem = skeleton.nodes.len() as u32;
        grow_stem(
            ctx,
            skeleton,
            tree_rng,
            attach,
            d,
            vigor * child.scale,
            child_level,
            p,
            0,
            child_stem,
        );
    };

    match child.pattern {
        ChildPattern::None => {}
        ChildPattern::Whorl { every, count } => {
            let every = every.max(1);
            let count = count.max(1);
            if ((seg + 1) as u32) % every != 0 {
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

fn steer(ctx: &GrowCtx, sp: &StemParams, dir: Vec3, pos: Vec3, level: u8, bend: Vec3) -> Vec3 {
    let mut d = dir
        + Vec3::Y * sp.phototropism * ctx.params.phototropism_multiplier
        - Vec3::Y * sp.gravity * ctx.params.gravity_multiplier;
    d += bend * sp.curvature;

    if level > 0 {
        let dens = ctx.env.density(pos);
        if dens < 1.0 {
            let target = ctx.env.steer_target(pos);
            let pull = Vec3::new(target.x - pos.x, 0.0, target.z - pos.z);
            if pull.length_squared() > 1e-8 {
                d += pull.normalize() * ctx.env.pull_strength * (1.0 - dens);
            }
        }
    }
    norm_or_up(d)
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
    use crate::species::{parse_species, PINE_RON};

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
        let params = parse_species(PINE_RON).unwrap();
        let env = params.envelope.scaled(params.envelope_scale);
        let sk = grow(&params);
        for node in &sk.nodes {
            if node.level > 0 {
                let d = env.density(node.position);
                assert!(
                    d >= params.envelope.kill_threshold - 1e-4,
                    "node at {:?} has density {d}",
                    node.position
                );
            }
        }
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
