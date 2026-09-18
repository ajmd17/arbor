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
/// Turning a stem may bank, in radians per metre of the length it will grow.
///
/// The bank exists because every influence is a nudge per segment, so a stem cut into
/// more segments bends further on the same settings, without limit, until it comes
/// round on itself. Budgeting the total is what stops that. Scaling the budget with
/// length is what keeps the rule fair between a fourteen-metre limb and a twig: a flat
/// allowance is nothing to the twig and a straitjacket on the limb, and a limb held to
/// a few degrees over its whole run is exactly what reads as extruded rather than
/// grown.
const TURN_BANK_PER_M: f32 = 0.26;
/// Ceiling on that bank however long the stem, in radians. A limb may turn through a
/// right angle and more on its way out; it may not come round on itself.
const MAX_STEM_TURN: f32 = 1.7;
/// Shares of the parent drive a fork and the stem it leaves come away with, when the
/// fork is a side branch and the original carries on as the leader. `split_evenness`
/// moves both toward `SPLIT_EVEN`, where neither is the leader any more.
const SPLIT_VIGOR: f32 = 0.72;
const SPLIT_KEEP: f32 = 0.86;
const SPLIT_EVEN: f32 = 0.86;
/// Side of the cells the crowding field counts wood into, in metres. About the reach
/// of one season's shoot: fine enough to tell the inside of a crown from its surface,
/// coarse enough that a stem is not judged by its own thickness.
const CROWDING_CELL: f32 = 1.2;
/// Longest a lateral may grow, as a share of the stem it leaves.
///
/// Measured against what the parent actually grew rather than what it set out to, so a
/// stem the crown cut short does not hand its children a budget drawn from length it
/// never reached. A co-dominant fork is exempt: it is the same axis carrying on, not a
/// lateral off it.
const LATERAL_MAX_SHARE: f32 = 0.8;
/// Where along a stem a lateral starts being held down by how little is left in front
/// of it, as a share of the stem's length.
///
/// A gentle ramp, and all the trunk answers to by default: its limbs are shaped by the
/// crown envelope. Anything leaving a branch answers to `tip_reach` as well, which is
/// much stricter.
const LATERAL_TIP_SHARE: f32 = 0.35;
/// Shortest the ramp above may make a lateral, as a share of the stem it leaves.
const LATERAL_MIN_SHARE: f32 = 0.08;
/// What `tip_reach` means when a branch level leaves it at zero: a child is never longer
/// than the branch still to come past it. See `ChildParams::tip_reach`.
const TIP_REACH: f32 = 1.0;
/// How far below `tip_reach` a lateral's allowance may be drawn, as a share of it.
/// Siblings leaving one node all see the same length ahead of them, so a hard cap would
/// pin every one it binds to the same number and grow them as a wheel of identical
/// spokes.
const TIP_REACH_SPREAD: f32 = 0.3;
/// Salt for the draw above. Taken from a hash of the stem's path rather than from its
/// random stream, so it does not move a single other draw in the tree.
const AHEAD_SALT: u64 = 0xA4EA_D5A1_7E57_0001;
/// Bare wood a fork needs past it on the stem it leaves, in the stem's own segments.
/// A lateral needs one: a stem's last season carries no side shoots yet. A fork needs
/// two, because one segment past a fork is not a stem carrying on but a stub beside it.
const FORK_BARE_SEGMENTS: f32 = 2.0;
/// Most children one node of a parent may carry, so a runaway rate cannot spend the
/// whole node budget at one point on one stem.
const MAX_CHILDREN_PER_NODE: u32 = 8;
/// Shape of the draw that decides how unequally siblings share their parent's drive.
/// Cubed, so most children come away well under the mean and a few come away at
/// several times it; the times four puts the mean back at one.
const DOMINANCE_SHAPE: f32 = 3.0;
/// Length a stem keeps at zero drive, as a share of what its level declares.
///
/// It used to be high, because drive was measured absolutely and compounded down the
/// levels, so without a floor a deep twig was shortened once by its level's `length`
/// and again by the scales above it. Drive is relative now, which frees the floor to
/// do the job it should: a suppressed stem has to come out genuinely short, or the
/// dominant and the suppressed end up within a factor of two of each other and the
/// crown reads as a bottle brush however unequally the drive was shared out.
///
/// It applies to a stem's thickness as well as its length, and it has to apply to both
/// or the two come apart. Losing the contest should scale a stem down, not distort it:
/// at the floor a stem is fifteen per cent of its declared length *and* fifteen per cent
/// of its declared radius, which leaves its proportions where they were. Radius used to
/// stop at `MIN_RADIUS` instead — an absolute four millimetres, the same for a trunk as
/// for a twiglet — so a suppressed limb came out 15% as long but pinned to the thickness
/// of a wire, at nearly four times the length-to-radius of its healthy neighbours. Those
/// are the stringy bits.
const SUPPRESSION_FLOOR: f32 = 0.15;

/// The most a stem may grow past what its level declares, as a multiple of it.
///
/// Vigor is meant to modulate a stem's length, and mostly it shortens: a suppressed
/// stem sits near `LENGTH_FLOOR`. But a little headroom above 1 is what keeps the
/// siblings of one whorl from all pinning to the same number and growing as a wheel of
/// identical spokes, which `siblings_in_a_whorl_get_different_lengths` watches for.
///
/// It used to be 1.6, and that was too much to be paid for twice: `length_variance`
/// multiplies on top, so a level declaring 2.2 m with a variance of 0.4 could put out a
/// stem of 4.9 m. Those are the stray hairs — single twigs two or three times the length
/// of everything around them, running out of the crown with nothing on them. 1.15 is the
/// most that can be allowed while the whorl still reads as siblings rather than spokes.
const MAX_DRIVE: f32 = 1.15;

/// A child settled on while its parent was still growing, held back until it has
/// finished.
///
/// Everything about the child is decided at the moment the parent reaches it, so the
/// random stream is the same as if it were grown there and then; only the growing
/// waits. What it waits for is the one thing the parent cannot know while it is still
/// climbing — how long it actually turned out to be. That decides both how long the
/// child may grow and whether it is grown at all: a stem the crown or its own vigor
/// stopped short may have settled on children at what turned out to be its tip.
struct Pending {
    attach: u32,
    /// How far along the parent it leaves, in metres of the parent actually grown.
    at_len: f32,
    dir: Vec3,
    vigor: f32,
    level: u8,
    path: u64,
    spray: Vec3,
    /// Set when the child is a fork rather than a lateral.
    fork: Option<PendingFork>,
}

/// What a fork carries away from the stem it divides, beyond what a lateral does.
struct PendingFork {
    split_depth: u32,
    /// The parent's unspent turning where it left; see `grow_stem`'s `budget`.
    budget: f32,
    /// The parent's `split_evenness`, which decides how far the fork is held to what
    /// the parent had left.
    evenness: f32,
}

struct GrowCtx<'a> {
    params: &'a SpeciesParams,
    env: EnvelopeParams,
    levels_total: u8,
    /// The leader centreline, appended to as the trunk climbs. The crown envelope is
    /// described around a vertical axis, so a trunk that leans would grow out of its
    /// own crown and everything on the upper trunk would be pruned at birth. Hanging
    /// the envelope off this instead makes the whole crown lean with the tree.
    leader: RefCell<Vec<Vec3>>,
    /// The drive a stem at each level comes away with when nothing has gone wrong for
    /// it: the product of the `children.scale` of every level above. Vigor is measured
    /// against this rather than against 1, so `length` at a level means the length a
    /// healthy stem there actually reaches. Without it the scales compound and a level
    /// five deep runs at a tenth of what the species asked for, which makes the whole
    /// file impossible to reason about.
    nominal: Vec<f32>,
}

impl GrowCtx<'_> {
    fn record_leader(&self, pos: Vec3) {
        self.leader.borrow_mut().push(pos);
    }

    /// The drive expected of a healthy stem at this level.
    fn nominal(&self, level: u8) -> f32 {
        self.nominal
            .get(level as usize)
            .copied()
            .unwrap_or(1.0)
            .max(1e-4)
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
    let levels_total = levels_total(params);
    let nominal = nominal_vigor(params);
    let ctx = GrowCtx {
        env: params.grown_envelope(),
        levels_total,
        params,
        leader: RefCell::new(Vec::new()),
        nominal,
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
        f32::MAX,
        f32::INFINITY,
    );
    resolve_radii(params, &mut skeleton);
    let shade_keep = mark_dieback(params, &mut skeleton);
    // Reads the radii, so it has to follow them, and it is its own pass rather than
    // the tail of the one above: a species that loses no wood to dieback still has
    // wood the crown pruned, and that has to break back too.
    break_dead_wood(params, &mut skeleton, shade_keep.as_deref());
    sag_dead_limbs(params, &mut skeleton);
    skeleton
}

/// How hemmed in every node is by the rest of its own tree, from 0 to 1.
///
/// What shades a branch out is almost always its own neighbours, so the wood packed
/// into the space around a stem stands in for the light that never reaches it. Wood is
/// measured as length rather than as a count of nodes, because a level with short
/// segments lays down many more nodes per metre than one with long segments and
/// counting them would report the finest twigs — which live on the outside of the
/// crown — as the most crowded thing in the tree.
///
/// This is what hollows a crown into a shell of foliage over open branchwork. Without
/// it the only thing that removes a stem is growing outside the envelope, so the crown
/// fills solid to the trunk and every limb inside it is buried.
fn crowding_field(skeleton: &Skeleton) -> Vec<f32> {
    let cell = |v: f32| (v / CROWDING_CELL).floor() as i32;
    let key = |p: Vec3| (cell(p.x), cell(p.y), cell(p.z));

    let mut wood: std::collections::HashMap<(i32, i32, i32), f32> =
        std::collections::HashMap::new();
    for node in &skeleton.nodes {
        let Some(p) = node.parent else { continue };
        let run = (node.position - skeleton.nodes[p as usize].position).length();
        *wood.entry(key(node.position)).or_insert(0.0) += run;
    }

    let mut local: Vec<f32> = Vec::with_capacity(skeleton.nodes.len());
    for node in &skeleton.nodes {
        let (cx, cy, cz) = key(node.position);
        let mut total = 0.0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    total += wood.get(&(cx + dx, cy + dy, cz + dz)).copied().unwrap_or(0.0);
                }
            }
        }
        local.push(total);
    }

    // Measured against a high quantile rather than the maximum, so one freak cell at a
    // fork cannot push the whole crown toward zero.
    let mut sorted = local.clone();
    sorted.sort_by(f32::total_cmp);
    let hi = sorted
        .get(sorted.len().saturating_mul(9) / 10)
        .copied()
        .unwrap_or(1.0)
        .max(1e-3);
    local.iter().map(|&v| (v / hi).clamp(0.0, 1.0)).collect()
}

/// How far along its own stem every node sits, in metres, and how long each stem
/// actually grew, indexed by stem id.
///
/// Stem ids are the index of a node, so a vector the length of the skeleton holds them
/// all. Nodes are created parents first, so one forward pass is enough.
fn stem_arcs(skeleton: &Skeleton) -> (Vec<f32>, Vec<f32>) {
    let n = skeleton.nodes.len();
    let mut arc = vec![0.0f32; n];
    let mut grown = vec![0.0f32; n];
    for i in 0..n {
        let node = &skeleton.nodes[i];
        if let Some(p) = node.parent {
            let parent = &skeleton.nodes[p as usize];
            let run = (node.position - parent.position).length();
            // A fork is the same axis carrying on, not a new stem starting at zero:
            // a whorl a tenth of the way along a fork near the top of the tree is
            // near the top of the tree, and measuring it from the fork instead put a
            // whole whorl of the pine's crown under the shade line and killed it.
            let continues = parent.stem == node.stem || parent.level == node.level;
            arc[i] = if continues { arc[p as usize] + run } else { run };
        }
        let s = node.stem as usize;
        if s < n {
            grown[s] = grown[s].max(arc[i]);
        }
    }
    (arc, grown)
}

/// The share of its own length a child shaded out at `t` along its parent keeps: the
/// parent's `shade_keep` at its base, running up to all of it at the shade line.
fn shade_keep_share(children: &crate::species::ChildParams, t: f32) -> f32 {
    let keep = children.shade_keep.clamp(0.0, 1.0);
    let rise = (t / children.shade_line.max(1e-4)).clamp(0.0, 1.0);
    keep + (1.0 - keep) * rise
}

/// The share of children leaving a parent at `t` along it (0 at its base, 1 at its tip)
/// that its crown has shaded out, from the parent's `shade_line` and `shade_blend`.
fn shaded_share(children: &crate::species::ChildParams, t: f32) -> f32 {
    if children.shade_line <= 0.0 {
        return 0.0;
    }
    if children.shade_blend <= 1e-4 {
        return if t < children.shade_line { 1.0 } else { 0.0 };
    }
    ((children.shade_line - t) / children.shade_blend).clamp(0.0, 1.0)
}

/// Marks the stems the tree has lost.
///
/// A branch dies as a unit and takes everything it carries with it, so this runs as a
/// pass over the finished skeleton rather than during growth: a stem cannot know
/// whether it will be shaded out until the neighbours that shade it exist. Nodes are
/// created parents first, so one forward pass both decides and propagates.
///
/// Two things kill a stem. `shade_line` on the parent's children is where it sits:
/// low on its parent, under a crown that has since grown over it. `dieback` on its own
/// level is what it is: weak, and buried among its neighbours.
///
/// Returns, when anything was shaded out, the length in metres each stem keeps before
/// it breaks off, indexed by stem id: infinite for everything but the shaded, which are
/// cut back by their parent's `shade_keep`. The breaking itself is left to
/// `break_dead_wood`, which has to run after the radii are known anyway.
fn mark_dieback(params: &SpeciesParams, skeleton: &mut Skeleton) -> Option<Vec<f32>> {
    let levels = std::iter::once(&params.trunk).chain(params.branch_levels.iter());
    let (any_dieback, any_shade) = levels.fold((false, false), |(d, s), sp| {
        (d || sp.dieback > 0.0, s || sp.children.shade_line > 0.0)
    });
    if !any_dieback && !any_shade {
        return None;
    }
    // Crowding is the expensive part and only dieback reads it.
    let crowding = if any_dieback {
        crowding_field(skeleton)
    } else {
        vec![0.0; skeleton.nodes.len()]
    };
    let (arc, grown) = if any_shade {
        stem_arcs(skeleton)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut keep = if any_shade {
        Some(vec![f32::INFINITY; skeleton.nodes.len()])
    } else {
        None
    };
    for (i, &crowd) in crowding.iter().enumerate() {
        let node = &skeleton.nodes[i];
        let (parent, level, stem, path, vigor) =
            (node.parent, node.level, node.stem, node.path, node.vigor);
        let inherited = parent.is_some_and(|p| skeleton.nodes[p as usize].dead);
        if inherited {
            skeleton.nodes[i].dead = true;
            continue;
        }
        // Only the first node of a stem decides; the rest of the run follows it.
        let starts_stem = parent.is_none_or(|p| skeleton.nodes[p as usize].stem != stem);
        if !starts_stem {
            skeleton.nodes[i].dead = skeleton.nodes[parent.unwrap() as usize].dead;
            continue;
        }
        // Shaded out where it stands. Only a lateral answers to this: a fork is the
        // parent carrying on, not a child hanging off it.
        if any_shade && let Some(p) = parent {
            let p_node = &skeleton.nodes[p as usize];
            if level == p_node.level + 1
                && let Some(parent_sp) = stem_params(params, p_node.level)
            {
                let along = arc[p as usize] / grown[p_node.stem as usize].max(1e-4);
                let share = shaded_share(&parent_sp.children, along);
                if share > 0.0 && hash_unit(params.seed ^ 0x5AD3_11E5, path) < share {
                    skeleton.nodes[i].dead = true;
                    if let Some(keep) = keep.as_mut() {
                        let s = stem as usize;
                        if s < keep.len() {
                            keep[s] = grown[s] * shade_keep_share(&parent_sp.children, along);
                        }
                    }
                    continue;
                }
            }
        }
        let Some(sp) = stem_params(params, level) else {
            continue;
        };
        if sp.dieback <= 0.0 {
            continue;
        }
        // The weakest go first, and so do the ones with the most wood around them:
        // being suppressed is what makes a stem vulnerable, and being buried in the
        // middle of the crown is what finishes it. Crowding is the term that puts the
        // losses on the inside rather than scattering them evenly, which is the
        // difference between a hollow crown and a solid one.
        //
        // Both act on the share that survives rather than on the share that dies, so
        // they compose without ever contradicting the number the species wrote down:
        // a level told to lose everything loses everything, however exposed its stems,
        // and a level told to lose nothing loses nothing however buried they are.
        let weakness = ((0.5 - vigor.min(0.5)) / 0.5).clamp(0.0, 1.0);
        let exposure = ((0.4 + 1.6 * weakness) * (0.35 + 1.65 * crowd) / 1.5).clamp(0.0, 4.0);
        let survives = (1.0 - sp.dieback.clamp(0.0, 1.0)).powf(exposure);
        let chance = (1.0 - survives).clamp(0.0, 1.0);
        let roll = hash_unit(params.seed ^ 0xDEAD_5EED, path);
        if roll < chance {
            skeleton.nodes[i].dead = true;
        }
    }
    keep
}

/// Snaps the thin, exposed ends off the wood the tree has lost.
///
/// Dead wood does not stand intact: the further out along a dead branch, the thinner
/// and more exposed it is, and it goes first. What is left is a stub. A limb thicker
/// than its level's `snap_radius` keeps its length; below that it breaks back in
/// proportion, and everything it carried goes with it.
///
/// A stem shaded out low on its parent is cut back further, to the length `shade_keep`
/// gave it in `keep` (indexed by stem id), since it died long ago and stopped growing
/// then.
fn break_dead_wood(params: &SpeciesParams, skeleton: &mut Skeleton, keep: Option<&[f32]>) {
    // The radius the stem started at, carried along its run.
    let mut stem_base = vec![0.0f32; skeleton.nodes.len()];
    let arc = keep.map(|_| stem_arcs(skeleton).0);
    for i in 0..skeleton.nodes.len() {
        let node = &skeleton.nodes[i];
        let (parent, stem, radius, frac, level, dead) = (
            node.parent,
            node.stem,
            node.radius,
            node.stem_fraction,
            node.level,
            node.dead,
        );

        if parent.is_some_and(|p| skeleton.nodes[p as usize].broken) {
            skeleton.nodes[i].broken = true;
            continue;
        }
        let same_stem = parent.is_some_and(|p| skeleton.nodes[p as usize].stem == stem);
        stem_base[i] = if same_stem {
            stem_base[parent.unwrap() as usize]
        } else {
            radius
        };
        if !dead {
            continue;
        }
        // Cut back to what it had when it died. Like the snap below, never the node
        // the stem starts on: the stub stays.
        if let (Some(keep), Some(arc)) = (keep, arc.as_ref())
            && same_stem
            && arc[i] > keep.get(stem as usize).copied().unwrap_or(f32::INFINITY)
        {
            skeleton.nodes[i].broken = true;
            continue;
        }
        let Some(sp) = stem_params(params, level) else {
            continue;
        };
        if sp.snap_radius <= 0.0 {
            continue;
        }
        let keeps = (stem_base[i] / sp.snap_radius).clamp(0.0, 1.0);
        // Never take the node a stem starts on. Dead wood breaks back to a stub, which
        // is what this pass is for, and a stub that breaks too is just an absence. It
        // matters most for the one the envelope leaves on a bare bole: that is a stem
        // of a single node at `stem_fraction` 1.0, so any `keeps` below one snapped it
        // off the moment it was marked dead and the bole came out clean again.
        if frac > keeps && same_stem {
            skeleton.nodes[i].broken = true;
        }
    }
}

/// Settles the limbs the tree has lost down about their bases.
///
/// Shape is decided while a stem grows and death is decided afterwards, so a dead limb
/// comes out shaped exactly like a living one: held up as though it were still laying
/// down wood against its weight. This is what lets it settle. Each dead subtree whose
/// root hangs off living wood is tilted rigidly about that attachment, down, by a share
/// of the parent's `dead_sag_deg` drawn so most limbs settle a little and a few settle
/// hard — the rack of dead limbs under a crown is not tidy. A rigid tilt keeps whatever
/// curve the limb grew with; it is the hinge at the base that a dead limb reads by.
fn sag_dead_limbs(params: &SpeciesParams, skeleton: &mut Skeleton) {
    let any = std::iter::once(&params.trunk)
        .chain(params.branch_levels.iter())
        .any(|sp| sp.children.dead_sag_deg > 0.0);
    if !any {
        return;
    }
    for i in 0..skeleton.nodes.len() {
        let node = &skeleton.nodes[i];
        let Some(p) = node.parent else { continue };
        let parent = &skeleton.nodes[p as usize];
        // Only a lateral settles about its base. A fork is the axis carrying on, and
        // a dead leader stands as it grew.
        if !node.dead || parent.dead || node.level != parent.level + 1 {
            continue;
        }
        let Some(psp) = stem_params(params, parent.level) else { continue };
        let max = psp.children.dead_sag_deg;
        if max <= 0.0 {
            continue;
        }
        let base = parent.position;
        let heading = horizontal(node.position - base);
        if heading.length_squared() < 1e-6 {
            // Straight up or down has no side to settle toward.
            continue;
        }
        let u = hash_unit(params.seed ^ 0x5A6D_EAD5, node.path);
        let angle = (max * u * u).to_radians();
        // Tilting about the horizontal across the limb's heading swings its tip down.
        let axis = Vec3::Y.cross(heading.normalize()).normalize();
        let turn = glam::Quat::from_axis_angle(axis, angle);
        let mut stack = vec![i as u32];
        while let Some(k) = stack.pop() {
            let at = &mut skeleton.nodes[k as usize];
            at.position = base + turn * (at.position - base);
            stack.extend(at.children.iter().copied());
        }
    }
}

/// One value in 0..1 from a seed and a path, so dieback is the same every time a tree
/// is grown and different for every stem in it.
fn hash_unit(seed: u64, path: u64) -> f32 {
    let mut x = seed ^ path.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f32 / (1u64 << 53) as f32
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
    // The longest this stem may grow, worked out by whatever spawned it. Infinite for
    // the trunk, which leaves nothing and answers to nothing.
    max_len: f32,
) {
    if skeleton.nodes.len() >= MAX_NODES {
        return;
    }
    let Some(sp) = stem_params(ctx.params, level) else {
        return;
    };
    let mut rng = tree_rng.stream(path);
    let base_pos = skeleton.nodes[base_node as usize].position;
    // Drive is read against what a healthy stem at this level would have, not against
    // the trunk's. `children.scale` already says how much smaller each level is, and
    // each level's `length` says it again; measuring vigor absolutely charged the tree
    // for it twice over, so a level five deep ran at a tenth of its declared length
    // and no number in the species file meant what it said.
    let nominal = ctx.nominal(level);
    let drive = (vigor / nominal).clamp(0.0, MAX_DRIVE);
    let length_factor =
        (SUPPRESSION_FLOOR + (1.0 - SUPPRESSION_FLOOR) * drive).min(MAX_DRIVE);
    let mut stem_len = sp.length
        * length_factor
        * range_f32(&mut rng, 1.0 - sp.length_variance, 1.0 + sp.length_variance).max(0.2);
    stem_len = stem_len.min(max_len);
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
    let mut pending: Vec<Pending> = Vec::new();
    // What the stem really grew, as against `stem_len`, which is what it set out to.
    let mut grown_len = 0.0f32;
    let can_spawn = level + 1 < ctx.levels_total;
    // The leader keeps its own course, but a co-dominant fork is part of the crown
    // and has to be shaped by the envelope like any branch, or the tree grows as a
    // bundle of parallel poles.
    let shaped_by_envelope = level > 0 || split_depth > 0;
    let is_leader = level == 0 && split_depth == 0;
    // How much turning this stem has left to spend, across every influence that bends
    // it. Without a ceiling the crown pull alone settles into an orbit: whatever its
    // strength, there is a radius at which it supplies exactly the turn a circle
    // needs, and the stem rides it round. The bank is earned by the length the stem is
    // about to grow, so a long limb may wander and a twig may not; `budget` caps it
    // with whatever a fork's parent had left.
    let bank = if sp.turn_bank > 0.0 {
        sp.turn_bank
    } else {
        TURN_BANK_PER_M
    };
    let mut turn_budget = budget.min(bank * stem_len).clamp(0.0, MAX_STEM_TURN);

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
            seg as f32 / seg_count.max(1) as f32,
            shaped_by_envelope,
            &mut turn_budget,
        );
        frame = transport(cur_dir, next_dir, frame);
        bend = transport(cur_dir, next_dir, bend);
        cur_dir = next_dir;

        // The jog at each node. It alternates sides, so it costs nothing from the turn
        // bank and leaves the stem on the course everything above chose for it; what it
        // breaks is the smooth extrusion, which is what a stem grown a season at a time
        // never is.
        if sp.zigzag_deg > 1e-3 {
            let kink = (sp.zigzag_deg * range_f32(&mut rng, 0.55, 1.45)).to_radians();
            let side = if seg % 2 == 0 { 1.0 } else { -1.0 };
            let across = ortho_unit(frame, cur_dir) * side;
            cur_dir = turn_toward(cur_dir, norm_or_up(cur_dir + across), kink);
        }
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
                let node =
                    skeleton.push_node(Some(cur), stub, level, child_path(path, 0), v, 1.0, stem);
                // A stem the crown refused is dead the moment it is born, and has to
                // say so: `dead` is what gives the mesher a snapped-off flat end
                // rather than a twig's taper, and what keeps foliage off it. Left
                // alive, the stubs on a self-pruned bole read as live branchlets.
                skeleton.nodes[node as usize].dead = true;
            }
            break;
        }

        let frac = (seg + 1) as f32 / seg_count as f32;
        let seg_path = child_path(path, seg as u32);
        grown_len += seg_len;
        cur = skeleton.push_node(Some(cur), pos, level, seg_path, v, frac, stem);
        if is_leader {
            // Recorded before anything spawns here, so branches born at this height
            // already see the trunk they are hanging off.
            ctx.record_leader(pos);
        }

        if can_spawn {
            spawn_children(
                &mut rng,
                cur,
                cur_dir,
                frame,
                v,
                sp,
                level + 1,
                seg,
                seg_count,
                seg_len,
                grown_len,
                &mut azimuth,
                &mut slot,
                path,
                spray,
                &mut pending,
            );
        }

        // The most a fork taken here may grow, and the one thing a fork was never held
        // to. A lateral answers to `lateral_cap`; the fork was exempted from it on the
        // grounds that it is the same axis carrying on, and so grew against nothing but
        // its own level's declared length however far out it was taken.
        //
        // That exemption is the argument for capping it against the axis, not for
        // letting it run. A subordinate fork — `split_evenness` at zero, which is every
        // conifer here — is a side branch by the species file's own account, and it
        // divides what the stem has left rather than starting a fresh run of its own.
        // Letting it start one is where the stringy bits come from: it takes 72% of the
        // drive, which buys most of a limb's length wherever along the parent it is
        // taken, while its thickness is capped at `radius_ratio` of a parent that has
        // already tapered. A fork four fifths of the way out starts at a fifth of a
        // limb's radius and still grows half a limb's length, and it and everything it
        // carries sit near `MIN_RADIUS` — a crown's worth of wire hanging off one point.
        //
        // A co-dominant fork keeps the exemption, because for that one the claim is
        // true: an oak trading its trunk for two leaders is the axis starting again,
        // not dividing, and holding each half to what the trunk had left would cost the
        // tree its crown. `split_evenness` already says which of the two a fork is, so
        // it is what moves the cap between them.
        //
        // This is the cap as it stands while the stem is still growing, measured against
        // what it set out to grow; the fork is held back with the laterals and measured
        // again against what the stem actually reached before it is grown.
        let evenness = sp.split_evenness.clamp(0.0, 1.0);
        let fork_max = fork_cap(stem_len - grown_len, max_len, evenness);
        if can_spawn
            && split_depth < ctx.params.max_split_depth
            && sp.split_probability > 0.0
            && frac >= sp.split_start_fraction
            // A fork with less than a couple of segments to divide is not a fork. The
            // stem is within a twig's length of finishing and the level below is already
            // putting twigs there; splitting the axis this late only adds a second tip
            // beside the one that was coming anyway.
            && fork_max >= sp.segment_length * 2.0
            && skeleton.nodes.len() < MAX_NODES
            && rng.random::<f32>() < sp.split_probability
        {
            slot += 1;
            let az = azimuth + std::f32::consts::FRAC_PI_2 + range_f32(&mut rng, -0.6, 0.6);
            let spread = sp.split_angle_deg.to_radians().max(0.02);
            let crotch = spread * range_f32(&mut rng, 0.7, 1.3);
            let split_dir = child_dir(cur_dir, frame, az, crotch);
            let p = child_path(path, slot);
            // How the two halves divide what the stem had. A leader keeps the larger
            // share and the fork is a side branch; at full evenness they come away
            // equal and the stem stops being a leader at all, which is how a broadleaf
            // trades a single trunk for a crown of co-dominant limbs.
            let fork_share = SPLIT_VIGOR + (SPLIT_EVEN - SPLIT_VIGOR) * evenness;
            let keep_share = SPLIT_KEEP + (SPLIT_EVEN - SPLIT_KEEP) * evenness;
            // Held back like a lateral, because the stem may yet stop within a segment
            // or two of here — the crown prunes it, or it runs out of vigor — and a
            // fork taken at what turns out to be the tip is not a fork at all but a
            // branch sprouting out of the end of another.
            pending.push(Pending {
                attach: cur,
                at_len: grown_len,
                dir: split_dir,
                vigor: v * fork_share,
                level,
                path: p,
                spray,
                fork: Some(PendingFork {
                    split_depth: split_depth + 1,
                    budget: turn_budget,
                    evenness,
                }),
            });
            // The parent gives up part of its drive to the fork instead of both
            // halves carrying on at full strength.
            v *= keep_share;
        }

        v *= 1.0 - sp.vigor_falloff / seg_count as f32;
        if v < MIN_VIGOR * nominal {
            break;
        }
    }

    // Now the stem has stopped, so its children can be measured against what it
    // reached rather than what it intended. A limb the crown cut off at a third of its
    // length would otherwise hand its twigs a cap drawn from the other two thirds, and
    // they come out longer than the branch carrying them.
    //
    // It also decides which of them are grown at all. Every stem ends in a bare
    // stretch: its last season, which has not put out side shoots yet. `end_fraction`
    // says where that starts, but it was only ever measured against the length the
    // stem set out to grow, so a stem the crown pruned, or one that ran out of vigor,
    // kept every child it had settled on up to the node it stopped at, and put out
    // branches straight from its own tip. Measured again here against what it reached,
    // and never less than one of its own segments.
    let tip_room = seg_len * 0.999;
    let end_at = grown_len * (sp.children.end_fraction + 1e-4);
    let reach = tip_reach(sp, level);
    for child in pending {
        if skeleton.nodes.len() >= MAX_NODES {
            return;
        }
        let ahead = grown_len - child.at_len;
        let (split_depth, budget, cap) = match &child.fork {
            Some(fork) => {
                let mut cap = fork_cap(ahead, max_len, fork.evenness);
                // Even a co-dominant fork is held to this. Two equal halves are twins,
                // and the other half is the stem carrying on past the fork: an "even"
                // fork that outgrows it is not even, it is a limb sprouting out of the
                // end of a branch. The trunk is let off by default, because an oak
                // trading its leader for a crown of limbs is exactly that.
                if let Some(reach) = reach {
                    cap = cap.min(ahead * reach);
                }
                if ahead < FORK_BARE_SEGMENTS * tip_room
                    || cap < sp.segment_length * FORK_BARE_SEGMENTS
                {
                    continue;
                }
                (fork.split_depth, fork.budget, cap)
            }
            None => {
                if ahead < tip_room || child.at_len > end_at {
                    continue;
                }
                let share = reach.map(|reach| {
                    let u = hash_unit(ctx.params.seed ^ AHEAD_SALT, child.path);
                    reach * (1.0 - TIP_REACH_SPREAD * u)
                });
                (0, MAX_STEM_TURN, lateral_cap(grown_len, child.at_len, share))
            }
        };
        let child_stem = skeleton.nodes.len() as u32;
        grow_stem(
            ctx,
            skeleton,
            tree_rng,
            child.attach,
            child.dir,
            child.vigor,
            child.level,
            child.path,
            split_depth,
            child_stem,
            child.spray,
            budget,
            cap,
        );
    }
}

/// How far past the length still ahead of it a child of a stem at `level` may run; see
/// `ChildParams::tip_reach`. None is no limit, which is the trunk's default: the crown
/// envelope shapes its limbs.
fn tip_reach(sp: &StemParams, level: u8) -> Option<f32> {
    if sp.children.tip_reach > 0.0 {
        Some(sp.children.tip_reach)
    } else if level > 0 {
        Some(TIP_REACH)
    } else {
        None
    }
}

/// The longest a fork may grow, with `ahead` metres of the stem it divides still to
/// come past it and `max_len` the most that stem was allowed.
///
/// A subordinate fork divides what the stem has left; a co-dominant one is the axis
/// starting again and answers only to what the stem itself answered to. `evenness`
/// moves between the two. Written out rather than as one lerp, because the trunk's
/// `max_len` is infinite and infinity times a zero evenness is not zero but NaN — which
/// `min` then quietly discarded, so every subordinate fork off a conifer's leader grew
/// as though it were co-dominant.
fn fork_cap(ahead: f32, max_len: f32, evenness: f32) -> f32 {
    let ahead = ahead.max(0.0).min(max_len);
    if evenness <= 0.0 {
        ahead
    } else {
        (ahead + (max_len - ahead) * evenness).min(max_len)
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_children(
    rng: &mut SmallRng,
    attach: u32,
    dir: Vec3,
    frame: Vec3,
    vigor: f32,
    sp: &StemParams,
    child_level: u8,
    seg: usize,
    seg_count: usize,
    seg_len: f32,
    at_len: f32,
    azimuth: &mut f32,
    slot: &mut u32,
    stem_path: u64,
    spray: Vec3,
    pending: &mut Vec<Pending>,
) {
    let child = &sp.children;
    let frac = (seg + 1) as f32 / seg_count as f32;
    if frac < child.start_fraction || frac > child.end_fraction {
        return;
    }
    let spawn_one = |az: f32,
                     pending: &mut Vec<Pending>,
                     slot: &mut u32,
                     rng: &mut SmallRng| {
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
        // What the child comes away with. Three things decide it, and only the first
        // was here before: the level's `scale`, a symmetric jitter, a skewed lottery
        // that makes a few siblings dominant and most of them suppressed, and where on
        // the parent it sits. Siblings drawn from one narrow band around the mean are
        // what makes a crown read as a bottle brush however well everything else is
        // tuned.
        let spread = 1.0 + range_f32(rng, -child.scale_variance, child.scale_variance);
        let lottery = {
            let u: f32 = rng.random();
            let skewed = u.powf(DOMINANCE_SHAPE) * (DOMINANCE_SHAPE + 1.0);
            1.0 + child.dominance.clamp(0.0, 1.0) * (skewed - 1.0)
        };
        // Acrotony: the strongest shoots of a season form at the far end of what grew
        // last season, which is what carries a crown outward and leaves its inside
        // open. Spread evenly instead and the foliage comes out as a band down every
        // limb.
        let acro = 1.0 + child.acrotony.clamp(-1.0, 1.0) * (2.0 * t - 1.0);
        let drive = child.scale * spread * lottery.max(0.0) * acro.max(0.05);
        pending.push(Pending {
            attach,
            at_len,
            dir: d,
            vigor: vigor * drive.max(0.02),
            level: child_level,
            path: child_path(stem_path, *slot),
            spray: child_spray,
            fork: None,
        });
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
                spawn_one(az, pending, slot, rng);
            }
        }
        ChildPattern::Continuous { density } => {
            // Children per metre of the parent, counted over the length of this
            // segment. It used to be the chance that this one segment carried a child,
            // which meant the ramification of a level moved whenever its segment
            // length did, and — worse — that every rate of one or more was the same
            // rate, since a chance saturates. Three of the oak's four levels were
            // pinned there, which is why its branching collapsed two orders early
            // however high the number went. Whole children are spawned outright and
            // the fraction left over is taken as a chance, so a rate below one child
            // per segment still behaves as it did.
            let expected = (density * seg_len).max(0.0);
            let mut count = expected.floor() as u32;
            if rng.random::<f32>() < expected.fract() {
                count += 1;
            }
            let count = count.min(MAX_CHILDREN_PER_NODE);
            if count == 0 {
                // The bud was there and came to nothing; the spiral still moves on.
                *azimuth += child.phyllotaxis_deg.to_radians();
                return;
            }
            for _ in 0..count {
                *azimuth += child.phyllotaxis_deg.to_radians();
                let az = *azimuth
                    + range_f32(rng, -child.roll_variance_deg, child.roll_variance_deg).to_radians();
                spawn_one(az, pending, slot, rng);
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
    // How far along the stem this segment sits, 0 at the base and 1 at the tip.
    along: f32,
    shaped_by_envelope: bool,
    // Turning this stem has left to spend, drawn down by whatever it uses.
    turn_budget: &mut f32,
) -> Vec3 {
    // A limb is a cantilever: the moment it carries grows with distance from where it
    // is held and the wood thins as it goes, so the sag is far from even along it.
    let sag = sp.gravity * (1.0 + sp.droop * along * along);
    let mut d = dir
        + Vec3::Y * sp.phototropism * ctx.params.phototropism_multiplier
        - Vec3::Y * sag * ctx.params.gravity_multiplier;
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

/// The longest a lateral leaving `at` along a stem of `grown` may be.
///
/// Never more than `LATERAL_MAX_SHARE` of the stem, ramping down over the last
/// `LATERAL_TIP_SHARE` of it, and never more than `ahead_share` times the stem still to
/// come past it when that is given.
fn lateral_cap(grown: f32, at: f32, ahead_share: Option<f32>) -> f32 {
    if !grown.is_finite() || grown <= 0.0 {
        return f32::INFINITY;
    }
    let beyond = (grown - at).max(0.0);
    let held = (beyond / (grown * LATERAL_TIP_SHARE)).clamp(0.0, 1.0);
    let cap = grown * (LATERAL_MAX_SHARE * held).max(LATERAL_MIN_SHARE);
    match ahead_share {
        Some(share) => cap.min(beyond * share),
        None => cap,
    }
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

fn levels_total(params: &SpeciesParams) -> u8 {
    params
        .max_levels
        .min(params.branch_levels.len() as u8 + 1)
        .max(1)
}

/// The vigor a healthy stem at each level would carry, as the running product of the
/// `scale` every level hands its children.
///
/// Both the length pass and the thickness pass measure a stem's drive against this
/// rather than against the trunk's vigor outright, and they have to. `children.scale`
/// already says how much smaller each level is, and that level's own `length` and
/// `radius` say it again; charging a stem for its depth a second time by scaling with
/// raw vigor compounds it, so a level four or five deep comes out at a fraction of
/// everything its own line of the species file asks for.
///
/// The length pass was fixed for this long ago and the thickness pass was not, which is
/// what made the stray hairs: level-2 stems were carrying vigor around 0.17 against a
/// nominal of 0.28, so they came out at 19% of their declared radius — a twig a fifth
/// of its proper thickness, which at any length reads as a wire rather than a branch.
fn nominal_vigor(params: &SpeciesParams) -> Vec<f32> {
    let total = levels_total(params);
    let mut nominal = Vec::with_capacity(total as usize + 1);
    nominal.push(1.0f32);
    for level in 0..total {
        let scale = stem_params(params, level).map_or(1.0, |sp| sp.children.scale.max(1e-3));
        let last = *nominal.last().expect("seeded with the trunk");
        nominal.push(last * scale);
    }
    nominal
}

/// Thickness is resolved in two passes: a top-down pass giving every stem a base
/// radius that tapers along its own length, then a bottom-up pass that widens any
/// node carrying more cross-section than its taper alone would provide.
fn resolve_radii(params: &SpeciesParams, skeleton: &mut Skeleton) {
    let n = skeleton.nodes.len();
    if n == 0 {
        return;
    }
    let nominal = nominal_vigor(params);
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
                    // its own level radius scaled by the drive it started with, the
                    // same quantity that sets its length. The parent ratio is only a
                    // ceiling, so a short twig on a thick limb stays a twig.
                    //
                    // Drive, not raw vigor: measured against what a healthy stem at this
                    // level would carry, exactly as the length pass measures it. See
                    // `nominal_vigor` — using vigor here charged every stem for its own
                    // depth twice and left the fine levels at a fifth of their declared
                    // thickness.
                    let n = nominal
                        .get(level as usize)
                        .copied()
                        .unwrap_or(1.0)
                        .max(1e-4);
                    let drive = (vigor / n).clamp(0.0, MAX_DRIVE);
                    // Floored the same way the length is, so suppression scales a stem
                    // rather than stretching it. `MIN_RADIUS` stays below as a backstop
                    // against zero, but it should no longer be what decides a stem's
                    // thickness: as an absolute length it made every suppressed stem the
                    // same four millimetres whatever level it belonged to.
                    let factor =
                        (SUPPRESSION_FLOOR + (1.0 - SUPPRESSION_FLOOR) * drive).min(MAX_DRIVE);
                    let own = lp.radius * factor;
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
    use crate::species::{parse_species, BIRCH_RON, OAK_RON, PINE_RON, SPRUCE_RON};

    /// Every stem in the skeleton, as (level, length, children it carries, the
    /// fraction along its parent where it attaches).
    fn stems(sk: &Skeleton) -> Vec<(u8, f32, usize, f32)> {
        sk.stem_runs()
            .into_iter()
            .filter(|run| !run.is_empty())
            .map(|run| {
                let head = &sk.nodes[run[0] as usize];
                let mut length = 0.0;
                if let Some(p) = head.parent {
                    length += (head.position - sk.nodes[p as usize].position).length();
                }
                for w in run.windows(2) {
                    length +=
                        (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length();
                }
                let children = run
                    .iter()
                    .map(|&i| {
                        sk.nodes[i as usize]
                            .children
                            .iter()
                            .filter(|&&c| sk.nodes[c as usize].stem != sk.nodes[i as usize].stem)
                            .count()
                    })
                    .sum();
                let attach = head.parent.map_or(0.0, |p| sk.nodes[p as usize].stem_fraction);
                (head.level, length, children, attach)
            })
            .collect()
    }

    fn mean(v: &[f32]) -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    }

    #[test]
    fn a_continuous_rate_is_children_per_metre_not_per_segment() {
        // It was a chance per segment, which saturated at 1.0 and moved whenever a
        // level's segment length did. Both of those silently capped how far the
        // presets could ramify, and neither was visible in the file.
        let base = parse_species(OAK_RON).unwrap();
        let carried = |density: f32, segment: f32| {
            let mut params = base.clone();
            params.branch_levels[1].segment_length = segment;
            if let ChildPattern::Continuous { density: d } =
                &mut params.branch_levels[1].children.pattern
            {
                *d = density;
            } else {
                panic!("the oak's third level is meant to branch continuously");
            }
            let sk = grow(&params);
            let counts: Vec<f32> = stems(&sk)
                .iter()
                .filter(|(level, ..)| *level == 2)
                .map(|(_, _, kids, _)| *kids as f32)
                .collect();
            mean(&counts)
        };

        // A rate above one per segment is not the same as a rate of one per segment.
        let single = carried(2.0, 0.28);
        let double = carried(4.0, 0.28);
        assert!(
            double > single * 1.5,
            "doubling the rate moved the children per stem from {single:.1} to {double:.1}"
        );
        // And the rate means the same thing however the stem is cut into segments.
        let coarse = carried(3.0, 0.42);
        let fine = carried(3.0, 0.21);
        assert!(
            (coarse - fine).abs() < coarse * 0.35,
            "halving the segment length moved the children per stem from {coarse:.1} to {fine:.1}"
        );
    }

    #[test]
    fn a_levels_declared_length_is_what_its_stems_reach() {
        // Drive used to be measured absolutely and compounded down the levels, so each
        // level ran at a smaller fraction of what it declared than the one above and
        // no number in the file meant what it said. The crown still cuts stems short,
        // so the test is that the shortfall stops growing with depth.
        let params = parse_species(OAK_RON).unwrap();
        let sk = grow(&params);
        let all = stems(&sk);
        let mut ratios = Vec::new();
        for level in 1..params.branch_levels.len() as u8 {
            let declared = params.branch_levels[level as usize - 1].length;
            let lengths: Vec<f32> = all
                .iter()
                .filter(|(l, ..)| *l == level)
                .map(|(_, len, _, _)| *len)
                .collect();
            if lengths.len() < 20 {
                continue;
            }
            ratios.push((level, mean(&lengths) / declared));
        }
        assert!(ratios.len() >= 3, "not enough levels to compare");
        let shallow = ratios.first().expect("checked above").1;
        let deep = ratios.last().expect("checked above").1;
        assert!(
            deep > shallow * 0.45,
            "levels run at {ratios:?} of what they declare: the shortfall still compounds"
        );
    }

    #[test]
    fn dominance_makes_some_siblings_win_and_most_lose() {
        // Siblings drawn from one narrow band around the mean are what makes a crown
        // read as a bottle brush however well everything else is tuned.
        let spread_at = |dominance: f32| {
            let mut params = parse_species(OAK_RON).unwrap();
            for level in &mut params.branch_levels {
                level.children.dominance = dominance;
                level.dieback = 0.0;
            }
            params.trunk.children.dominance = dominance;
            let sk = grow(&params);
            let lengths: Vec<f32> = stems(&sk)
                .iter()
                .filter(|(l, ..)| *l == 2)
                .map(|(_, len, _, _)| *len)
                .collect();
            let m = mean(&lengths);
            let sd = (lengths.iter().map(|x| (x - m).powi(2)).sum::<f32>()
                / lengths.len().max(1) as f32)
                .sqrt();
            sd / m.max(1e-4)
        };
        let flat = spread_at(0.0);
        let ranked = spread_at(1.0);
        assert!(
            ranked > flat * 1.25,
            "dominance barely widened the spread of sibling length: {flat:.2} to {ranked:.2}"
        );
    }

    #[test]
    fn acrotony_carries_the_growth_to_the_ends_of_the_parent() {
        // Spread evenly instead and the foliage comes out as a band down every limb
        // rather than massing at the surface of the crown.
        let attach_at = |acrotony: f32| {
            let mut params = parse_species(OAK_RON).unwrap();
            for level in &mut params.branch_levels {
                level.children.acrotony = acrotony;
            }
            params.trunk.children.acrotony = acrotony;
            let sk = grow(&params);
            // Weighted by the drive each child came away with, which is the thing
            // acrotony moves. Weighting by length instead reads the cap that holds a
            // lateral down near the tip of its parent, and the two work against each
            // other: what acrotony hands the distal children in drive, the cap takes
            // straight back off them in length.
            let (mut sum, mut total) = (0.0f32, 0.0f32);
            for node in &sk.nodes {
                let Some(p) = node.parent else { continue };
                let parent = &sk.nodes[p as usize];
                if node.level < 2 || node.level == parent.level || node.stem == parent.stem {
                    continue;
                }
                sum += parent.stem_fraction * node.vigor;
                total += node.vigor;
            }
            sum / total.max(1e-4)
        };
        // Both ends of the range rather than one: children only spawn between
        // `start_fraction` and `end_fraction`, so the measure sits near 0.6 before
        // acrotony touches it and comparing against that offset leaves very little to
        // see. The negative half is what a conifer uses, so it is worth pinning too.
        let basal = attach_at(-0.8);
        let even = attach_at(0.0);
        let distal = attach_at(0.8);
        assert!(
            distal > even && even > basal,
            "acrotony did not order the attachment point: {basal:.2} / {even:.2} / {distal:.2}"
        );
        assert!(
            distal > basal + 0.08,
            "acrotony moved the weighted attachment point only from {basal:.2} to {distal:.2}"
        );
    }

    #[test]
    fn no_lateral_outgrows_the_stem_it_leaves() {
        // A branch is not longer than the branch it grows out of. Measured on what the
        // parent actually reached, not what it set out to grow: a stem the crown cuts
        // off at a third of its length still hands its children a cap, and taking that
        // cap from the length it never achieved is how twigs end up lying across the
        // branchwork longer than the wood carrying them.
        for src in [OAK_RON, PINE_RON, BIRCH_RON] {
            let params = parse_species(src).unwrap();
            let sk = grow(&params);

            // Length of every stem, from the raw nodes: `stem_runs` hides the wood that
            // died, and a stem measured without it reads as shorter than it grew.
            let mut length: std::collections::HashMap<u32, f32> =
                std::collections::HashMap::new();
            let mut head: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for (i, node) in sk.nodes.iter().enumerate() {
                let Some(p) = node.parent else { continue };
                let run = (node.position - sk.nodes[p as usize].position).length();
                *length.entry(node.stem).or_insert(0.0) += run;
                if sk.nodes[p as usize].stem != node.stem {
                    head.insert(node.stem, i as u32);
                }
            }

            let mut checked = 0;
            for (&stem, &first) in &head {
                let node = &sk.nodes[first as usize];
                let parent = &sk.nodes[node.parent.expect("a head has a parent") as usize];
                // A co-dominant fork is the same axis carrying on, not a lateral off
                // it, and may match the stem it left.
                if node.level == parent.level {
                    continue;
                }
                let (Some(&mine), Some(&theirs)) =
                    (length.get(&stem), length.get(&parent.stem))
                else {
                    continue;
                };
                // How far along the parent this one leaves, walking its parent's run
                // back to where that stem began.
                let mut at = 0.0f32;
                let mut walk = node.parent.expect("a head has a parent");
                while let Some(up) = sk.nodes[walk as usize].parent {
                    if sk.nodes[up as usize].stem != parent.stem {
                        break;
                    }
                    at += (sk.nodes[walk as usize].position - sk.nodes[up as usize].position)
                        .length();
                    walk = up;
                }
                // `at` misses the first segment, which belongs to the parent's run but
                // is measured from the grandparent; that makes the bound generous by one
                // segment rather than wrong.
                // The loosest allowance the draw can give.
                let reach = stem_params(&params, parent.level)
                    .and_then(|sp| tip_reach(sp, parent.level));
                let cap = lateral_cap(theirs, at, reach);
                assert!(
                    mine <= cap + 1e-3,
                    "{}: a level {} stem ran {mine:.2} m off a level {} stem of                      {theirs:.2} m, {at:.2} m along it, where the most it may be is                      {cap:.2} m",
                    params.name,
                    node.level,
                    parent.level
                );
                checked += 1;
            }
            assert!(checked > 200, "{}: only {checked} laterals to check", params.name);
        }
    }

    /// How long every stem grew and how far along its own stem every node sits, both in
    /// metres and counting the wood that later broke off, since what is being checked
    /// is what growth did.
    fn grown_lengths(sk: &Skeleton) -> (Vec<f32>, Vec<f32>) {
        let n = sk.nodes.len();
        let mut arc = vec![0.0f32; n];
        let mut length = vec![0.0f32; n];
        for (i, node) in sk.nodes.iter().enumerate() {
            if let Some(p) = node.parent {
                let run = (node.position - sk.nodes[p as usize].position).length();
                let same = sk.nodes[p as usize].stem == node.stem;
                arc[i] = if same { arc[p as usize] + run } else { run };
            }
            let s = node.stem as usize;
            length[s] = length[s].max(arc[i]);
        }
        (arc, length)
    }

    #[test]
    fn no_stem_puts_out_a_branch_from_its_own_tip() {
        // A stem's last season has no side shoots yet, so every stem ends in a bare
        // stretch. `end_fraction` was measured against the length a stem set out to
        // grow, so one the crown pruned or that ran out of vigor kept every child it
        // had settled on up to the node it stopped at, and sprouted branches — whole
        // co-dominant forks on the oak — straight out of its own end.
        for (name, src) in crate::species::builtin_presets() {
            let params = parse_species(src).unwrap();
            let sk = grow(&params);
            let (arc, length) = grown_lengths(&sk);
            // The node a stem carries on to past `i`, if it does.
            let next = |i: u32| {
                let stem = sk.nodes[i as usize].stem;
                sk.nodes[i as usize]
                    .children
                    .iter()
                    .copied()
                    .find(|&c| sk.nodes[c as usize].stem == stem)
            };
            let mut checked = 0;
            for node in &sk.nodes {
                let Some(p) = node.parent else { continue };
                let parent = &sk.nodes[p as usize];
                if parent.stem == node.stem {
                    continue;
                }
                // A stub the crown refused at birth is not a branch.
                if node.children.is_empty() && node.dead && node.stem_fraction >= 1.0 {
                    continue;
                }
                let fork = parent.level == node.level;
                let past = next(p);
                assert!(
                    past.is_some(),
                    "{name}: a level {} {} leaves the very tip of a level {} stem",
                    node.level,
                    if fork { "fork" } else { "lateral" },
                    parent.level
                );
                if fork {
                    assert!(
                        past.and_then(next).is_some(),
                        "{name}: a fork leaves within one segment of its parent's tip"
                    );
                }
                let Some(sp) = stem_params(&params, parent.level) else { continue };
                if !fork {
                    let along = arc[p as usize] / length[parent.stem as usize].max(1e-4);
                    assert!(
                        along <= sp.children.end_fraction + 1e-3,
                        "{name}: a lateral leaves {along:.3} along a stem whose children end \
                         at {:.3} of what it grew",
                        sp.children.end_fraction
                    );
                }
                checked += 1;
            }
            assert!(checked > 100, "{name}: only {checked} children to check");
        }
    }

    #[test]
    fn nothing_off_a_branch_outgrows_the_branch_ahead_of_it() {
        // A side shoot is no older than the length its parent went on to grow past it,
        // and out towards the tip there is less and less branch to carry its weight. A
        // long limb hanging off the last metre of a branch reads as one the branch could
        // never have held up.
        let offenders = |params: &SpeciesParams| {
            let sk = grow(params);
            let (arc, length) = grown_lengths(&sk);
            let (mut checked, mut over) = (0, Vec::new());
            for node in &sk.nodes {
                let Some(p) = node.parent else { continue };
                let parent = &sk.nodes[p as usize];
                if parent.stem == node.stem || parent.level == 0 {
                    continue;
                }
                let ahead = length[parent.stem as usize] - arc[p as usize];
                let reach = stem_params(params, parent.level)
                    .and_then(|sp| tip_reach(sp, parent.level))
                    .expect("a branch always has a reach");
                let mine = length[node.stem as usize];
                checked += 1;
                if mine > ahead * reach + 1e-3 {
                    over.push((node.level, mine, ahead));
                }
            }
            (checked, over)
        };
        for (name, src) in crate::species::builtin_presets() {
            let params = parse_species(src).unwrap();
            let (checked, over) = offenders(&params);
            assert!(checked > 100, "{name}: only {checked} children to check");
            assert!(
                over.is_empty(),
                "{name}: {} children outgrew the branch ahead of them, e.g. (level, length, \
                 ahead) {:?}",
                over.len(),
                &over[..over.len().min(5)]
            );
        }

        // And it is the knob doing it, not something else holding them in anyway: let
        // children run to four times what is ahead of them and plenty pass it once.
        let mut loose = parse_species(OAK_RON).unwrap();
        for level in &mut loose.branch_levels {
            level.children.tip_reach = 4.0;
        }
        let sk = grow(&loose);
        let (arc, length) = grown_lengths(&sk);
        let (mut children, mut past) = (0, 0);
        for node in &sk.nodes {
            let Some(p) = node.parent else { continue };
            let parent = &sk.nodes[p as usize];
            if parent.stem == node.stem || parent.level == 0 {
                continue;
            }
            children += 1;
            if length[node.stem as usize] > length[parent.stem as usize] - arc[p as usize] + 1e-3
            {
                past += 1;
            }
        }
        assert!(
            past * 5 > children,
            "at a tip reach of 4 only {past} of {children} children passed the branch ahead"
        );
    }

    #[test]
    fn a_subordinate_fork_off_the_trunk_is_held_to_what_the_leader_has_left() {
        // The trunk's allowance is infinite, and infinity times a zero evenness is NaN,
        // which `min` discarded: every subordinate fork off a conifer's leader was let
        // grow as though it were co-dominant.
        assert_eq!(fork_cap(3.0, f32::INFINITY, 0.0), 3.0);
        assert_eq!(fork_cap(3.0, f32::INFINITY, 0.5), f32::INFINITY);
        assert_eq!(fork_cap(3.0, 5.0, 0.5), 4.0);
        assert_eq!(fork_cap(8.0, 5.0, 0.0), 5.0);
    }

    #[test]
    fn a_tighter_turn_bank_stops_a_limb_coming_round_on_itself() {
        // A limb long against the crown it grows in has to be bent back to stay inside,
        // and with turning to spare the crown pull spends all of it: the limb closes a
        // quarter circle and comes out as a shepherd's crook. The bank is what says how
        // much it may spend.
        let worst_turn = |bank: f32| {
            let mut params = parse_species(BIRCH_RON).unwrap();
            for level in &mut params.branch_levels {
                level.turn_bank = bank;
            }
            let sk = grow(&params);
            let mut worst = 0.0f32;
            for run in sk.stem_runs() {
                if run.len() < 4 {
                    continue;
                }
                let step = |i: usize| {
                    sk.nodes[run[i] as usize].position - sk.nodes[run[i - 1] as usize].position
                };
                let (Some(a), Some(b)) =
                    (step(1).try_normalize(), step(run.len() - 1).try_normalize())
                else {
                    continue;
                };
                worst = worst.max(a.dot(b).clamp(-1.0, 1.0).acos().to_degrees());
            }
            worst
        };
        // Loose enough that some limb is long enough to spend it: a lateral is held to
        // the branch still ahead of it, so few birch limbs have the run they once had.
        let loose = worst_turn(0.60);
        let tight = worst_turn(0.08);
        assert!(
            loose > 75.0,
            "a loose bank should let some limb come well round, got {loose:.0} degrees"
        );
        assert!(
            tight < loose * 0.7,
            "tightening the bank barely straightened anything: {loose:.0} to {tight:.0} degrees"
        );
    }

    #[test]
    fn a_zigzag_breaks_the_line_without_bending_the_stem() {
        // The jog has to leave a stem going where everything else sent it, or it is
        // just another source of curvature and the model already has several.
        let measure = |zigzag: f32| {
            let mut params = parse_species(OAK_RON).unwrap();
            for level in &mut params.branch_levels {
                level.zigzag_deg = zigzag;
            }
            let sk = grow(&params);
            let (mut net, mut wander, mut n) = (0.0f32, 0.0f32, 0usize);
            for run in sk.stem_runs() {
                if run.len() < 5 {
                    continue;
                }
                let step = |i: usize| {
                    sk.nodes[run[i] as usize].position - sk.nodes[run[i - 1] as usize].position
                };
                let (Some(a), Some(b)) = (step(1).try_normalize(), step(run.len() - 1).try_normalize())
                else {
                    continue;
                };
                net += a.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
                let path: f32 = (1..run.len()).map(|i| step(i).length()).sum();
                let reach = (sk.nodes[run[run.len() - 1] as usize].position
                    - sk.nodes[run[0] as usize].position)
                    .length();
                wander += 1.0 - reach / path.max(1e-4);
                n += 1;
            }
            (net / n as f32, wander / n as f32)
        };
        let (straight_net, straight_wander) = measure(0.0);
        let (kinked_net, kinked_wander) = measure(12.0);
        // A jog of a few degrees is meant to be a small effect on the path and no
        // effect at all on the heading, so this is a ratio rather than a margin.
        assert!(
            kinked_wander > straight_wander * 1.8,
            "the jog added no path length: {straight_wander:.3} to {kinked_wander:.3}"
        );
        assert!(
            kinked_net < straight_net + 12.0,
            "the jog bent the stems as well: {straight_net:.0} to {kinked_net:.0} degrees net"
        );
    }

    #[test]
    fn an_even_split_gives_up_the_leader() {
        // A conifer keeps one trunk to the top; a mature broadleaf trades it for a
        // crown of co-dominant limbs. The difference is the share a fork comes away
        // with, and it should show up as how much of the tree the longest level-zero
        // stem accounts for.
        // What the fork comes away with against what the stem it left keeps. Measured
        // at the fork itself: further out the crown prunes whichever limb reaches its
        // boundary first, and that says more about where a fork happened to start than
        // about which of the two is the leader.
        let fork_against_leader = |evenness: f32| {
            let mut params = parse_species(OAK_RON).unwrap();
            params.trunk.split_evenness = evenness;
            let sk = grow(&params);
            let mut ratios = Vec::new();
            for node in &sk.nodes {
                let Some(p) = node.parent else { continue };
                let parent = &sk.nodes[p as usize];
                // A fork carries on at its parent's level on a stem of its own; a
                // child at the next level down is a branch, not a fork.
                if node.level != 0 || parent.level != 0 || node.stem == parent.stem {
                    continue;
                }
                // Against the parent's own continuation, not against the parent
                // node: the stem gives up its share to the fork after that node was
                // recorded, so the node still carries the undivided drive.
                let carries_on = parent
                    .children
                    .iter()
                    .map(|&c| &sk.nodes[c as usize])
                    .find(|c| c.stem == parent.stem);
                if let Some(carries_on) = carries_on.filter(|c| c.vigor > 1e-4) {
                    ratios.push(node.vigor / carries_on.vigor);
                }
            }
            (mean(&ratios), ratios.len())
        };
        let (subordinate, forks_low) = fork_against_leader(0.0);
        let (co_dominant, forks_high) = fork_against_leader(1.0);
        assert!(forks_low > 0 && forks_high > 0, "the trunk never forked at all");
        assert!(
            subordinate < 0.9,
            "a fork with no evenness should be the lesser of the two, not {subordinate:.2}"
        );
        assert!(
            co_dominant > 0.95,
            "an even split should leave neither limb the leader, got {co_dominant:.2}"
        );
    }

    #[test]
    fn crowding_takes_the_inside_of_the_crown_before_the_outside() {
        // What hollows a crown into a shell. Losses scattered evenly over the whole
        // volume leave it solid however many of them there are.
        let mut params = parse_species(OAK_RON).unwrap();
        for level in &mut params.branch_levels {
            level.dieback = 0.3;
        }
        let sk = grow(&params);
        let crowding = crowding_field(&sk);
        let mut inside = (0usize, 0usize);
        let mut outside = (0usize, 0usize);
        for (i, node) in sk.nodes.iter().enumerate() {
            if node.level < 2 {
                continue;
            }
            let bucket = if crowding[i] > 0.6 {
                &mut inside
            } else if crowding[i] < 0.2 {
                &mut outside
            } else {
                continue;
            };
            bucket.1 += 1;
            if node.dead {
                bucket.0 += 1;
            }
        }
        assert!(
            inside.1 > 200 && outside.1 > 200,
            "not enough wood either side of the crown to compare: {inside:?} {outside:?}"
        );
        // Compared as survival rather than as death. Death rates saturate: once both
        // are near certain the ratio between them collapses however much harder the
        // inside is being hit, and the test starts reporting on the ceiling instead of
        // on the crowding.
        let lived = |(dead, all): (usize, usize)| (all - dead) as f32 / all as f32;
        assert!(
            lived(outside) > lived(inside) * 1.5,
            "wood in the open survived at {:.2} against {:.2} for the crowded wood",
            lived(outside),
            lived(inside)
        );
    }

    #[test]
    fn dieback_takes_whole_branches_and_everything_they_carry() {
        // A branch dies as a unit: the tree does not keep a live twig on a dead limb.
        let mut params = parse_species(OAK_RON).unwrap();
        for level in &mut params.branch_levels {
            level.dieback = 0.0;
        }
        let alive = grow(&params);
        assert!(
            alive.nodes.iter().all(|n| !n.dead),
            "nothing should die when no level dies back"
        );

        for level in &mut params.branch_levels {
            level.dieback = 0.25;
        }
        let sk = grow(&params);
        let dead = sk.nodes.iter().filter(|n| n.dead).count();
        assert!(dead > 50, "only {dead} nodes died at a quarter dieback");
        assert!(dead < sk.nodes.len(), "the whole tree died");

        for node in &sk.nodes {
            if !node.dead {
                continue;
            }
            for &child in &node.children {
                assert!(
                    sk.nodes[child as usize].dead,
                    "a live stem hangs off dead wood"
                );
            }
        }

        // And the same tree twice over gives the same dead wood.
        assert_eq!(
            grow(&params).nodes.iter().filter(|n| n.dead).count(),
            dead,
            "dieback is not deterministic"
        );
    }

    #[test]
    fn trunk_keeps_its_thickness_up_the_stem() {
        // The taper ratio applies once over a whole stem. Compounding it per segment
        // instead shrinks the trunk to the radius floor within a couple of metres.
        for src in [OAK_RON, PINE_RON, BIRCH_RON] {
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
    fn a_vigorous_stem_may_not_run_past_its_ceiling() {
        // The stray-hair failure, and the one end of this that was never guarded.
        // `low_vigor_modulates_stem_length_instead_of_erasing_it` looks like it covers
        // it — its own message says "vigor should not lengthen a stem past what it
        // declares" — but it asserts that on the *shortest* stem of a species built to
        // have low vigor throughout, so it can never see a stem running long. That is
        // exactly the case that goes wrong: one twig several times its neighbours,
        // running out past the foliage with nothing on it.
        //
        // So this one hands the children full drive instead, which is what puts them
        // over the ceiling, and holds `length_variance` at zero so the ceiling is the
        // only thing deciding the longest stem.
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
        // Full drive to the children, and a contest they can win outright, which is
        // what carries a stem past nominal and into the ceiling. With the drive shared
        // evenly no stem ever exceeds nominal and the ceiling is never reached, so the
        // test would pass with the clamp taken out entirely.
        params.trunk.children.scale = 1.0;
        params.trunk.children.dominance = 0.95;
        params.trunk.vigor_falloff = 0.0;
        let branch = &mut params.branch_levels[0];
        branch.length = 4.0;
        branch.length_variance = 0.0;
        branch.vigor_falloff = 0.0;
        branch.split_probability = 0.0;

        let declared = params.branch_levels[0].length;
        let sk = grow(&params);
        let longest = sk
            .stem_runs()
            .iter()
            .filter(|run| sk.nodes[run[0] as usize].level == 1)
            .map(|run| stem_length(&sk, run))
            .fold(f32::MIN, f32::max);
        assert!(
            longest > declared * 0.5,
            "the branches never grew, so nothing is being tested: {longest}"
        );
        assert!(
            longest <= declared * MAX_DRIVE + 1e-3,
            "a stem grew to {longest:.2} of a declared {declared:.2}, past the \
             {:.2} that MAX_DRIVE allows; unbounded this is what makes the stray hairs",
            declared * MAX_DRIVE
        );
    }

    #[test]
    fn a_subordinate_fork_may_not_out_run_the_stem_it_divides() {
        // The stringy bits, and the other end of the stray-hair failure.
        // `a_vigorous_stem_may_not_run_past_its_ceiling` holds a stem to its own
        // level's declared length, which a fork never exceeded — the fault is that for
        // a fork the declared length is the wrong measure entirely. A fork taken four
        // fifths of the way along a limb came away with 72% of the drive and so grew
        // most of a limb again, out of a point where the parent had tapered to a wire
        // and `radius_ratio` held the fork to a couple of centimetres. Long, thin, and
        // carrying a whole subtree pinned near `MIN_RADIUS` behind it.
        //
        // A subordinate fork divides what the stem has left rather than starting a
        // fresh run, so the whole fork is measured against the length the parent still
        // had ahead of it at the point it left, not against the level's `length`.
        //
        // Split from the very foot of the stem so the offence has the longest possible
        // run to show up in, and with the crown big enough that nothing is pruned, so
        // the length the fork reaches is the length the rule allowed it.
        // Three levels, because a stem may only fork on a level that still has one
        // below it to spawn into, so a two-level tree never forks a branch at all.
        let mut params = SpeciesParams {
            max_levels: 3,
            max_split_depth: 1,
            branch_levels: vec![
                StemParams::branch_default(1),
                StemParams::branch_default(2),
            ],
            ..Default::default()
        };
        params.envelope.volumes = vec![crate::envelope::EnvelopeVolume::Ellipsoid {
            center: [0.0, 0.0, 0.0],
            radii: [500.0, 500.0, 500.0],
        }];
        params.trunk.split_probability = 0.0;
        params.trunk.children.pattern = ChildPattern::Continuous { density: 1.0 };
        params.branch_levels[1].split_probability = 0.0;
        let branch = &mut params.branch_levels[0];
        branch.length = 8.0;
        branch.length_variance = 0.0;
        branch.vigor_falloff = 0.0;
        // Every segment past the first forks, so the test sees forks taken at every
        // fraction along the limb rather than only wherever a low rate happened to
        // land them.
        branch.split_probability = 1.0;
        branch.split_evenness = 0.0;
        branch.split_start_fraction = 0.0;

        let sk = grow(&params);
        let mut checked = 0;
        for run in sk.stem_runs() {
            let head = &sk.nodes[run[0] as usize];
            let Some(parent) = head.parent else { continue };
            let parent = &sk.nodes[parent as usize];
            // A fork, rather than a lateral: the same level, carrying on off a stem of
            // its own kind.
            if parent.level != head.level || head.level == 0 {
                continue;
            }
            let ahead = sk
                .stem_runs()
                .iter()
                .find(|r| sk.nodes[r[0] as usize].stem == parent.stem)
                .map(|r| stem_length(&sk, r))
                .unwrap_or(0.0)
                * (1.0 - parent.stem_fraction);
            let grew = stem_length(&sk, &run);
            checked += 1;
            assert!(
                grew <= ahead.max(0.2) * 1.35 + 1e-3,
                "a fork leaving at {:.2} along its parent grew {grew:.2} m where the                  parent had {ahead:.2} m left to run; unbounded this is what makes the                  stringy bits",
                parent.stem_fraction
            );
        }
        assert!(checked > 5, "no forks were grown, so nothing is being tested");
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
            shortest >= declared * SUPPRESSION_FLOOR - 1e-3,
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
        for src in [PINE_RON, OAK_RON, BIRCH_RON] {
            let params = parse_species(src).unwrap();
            let env = params.grown_envelope();
            let sk = grow(&params);
            let trunk_stem = sk.nodes[0].stem;
            // In the order the trunk grew, not sorted by height. A trunk that leans
            // hard enough to dip has a polyline whose sorted form is a different shape
            // altogether, and reconstructing the anchor from that one answers a
            // question growth never asked.
            let leader: Vec<Vec3> = sk
                .nodes
                .iter()
                .filter(|n| n.stem == trunk_stem)
                .map(|n| n.position)
                .collect();

            // What `crown_offset` would have returned for a node at `y`, given that
            // only the first `grown` points of the leader had been recorded when it
            // was asked. Growth is depth first, so a branch sees the trunk only as far
            // as the trunk had climbed, and every prefix is a height some branch saw.
            let anchor_at = |y: f32, grown: usize| -> Vec3 {
                let seen = &leader[..grown];
                let Some(first) = seen.first() else {
                    return Vec3::ZERO;
                };
                if y <= first.y {
                    return horizontal(*first);
                }
                for pair in seen.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    if y <= b.y {
                        let span = b.y - a.y;
                        let t = if span > 1e-5 { (y - a.y) / span } else { 0.0 };
                        return horizontal(a.lerp(b, t.clamp(0.0, 1.0)));
                    }
                }
                horizontal(*seen.last().expect("checked non-empty"))
            };

            // A dead stub is deliberately outside the crown: it is what is left of a
            // branch the crown pruned the moment it appeared. Found from the nodes
            // themselves rather than from `stem_runs`, which hides the wood that has
            // since died and broken and would hand back no run for a stub at all.
            let mut stubs: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for (i, node) in sk.nodes.iter().enumerate() {
                let Some(parent) = node.parent else { continue };
                let alone = sk.nodes[parent as usize].stem != node.stem && node.children.is_empty();
                if !alone {
                    continue;
                }
                let expected = stem_params(&params, node.level)
                    .map(|sp| sp.dead_stub_length)
                    .unwrap_or(0.0);
                let reach = (node.position - sk.nodes[parent as usize].position).length();
                if expected > 0.0 && (reach - expected).abs() < 1e-3 {
                    stubs.insert(i as u32);
                }
            }

            let mut checked = 0;
            for (i, node) in sk.nodes.iter().enumerate() {
                // Dead wood is exempt: a shaded-out limb settles below where it grew,
                // and the crown only ever shaped living growth.
                if node.level == 0 || node.dead || stubs.contains(&(i as u32)) {
                    continue;
                }
                // Every prefix of the leader is a trunk some branch was grown against,
                // so the node has to sit inside the crown at least one of them offers.
                let mut best = env.density(node.position);
                for grown in 1..=leader.len() {
                    let offset = anchor_at(node.position.y, grown);
                    best = best.max(env.density(node.position - offset));
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
        //
        // The spruce, because its sprays are flat plates. The pine used to be the
        // fixture here, but a pine's shoots really are bottle brushes and its preset no
        // longer asks for plates.
        let params = parse_species(SPRUCE_RON).unwrap();
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
        // The foot of the crown is the lowest any of its volumes reaches, since the
        // envelope is their union and a preset may build it from several.
        let crown_base = params
            .envelope
            .volumes
            .iter()
            .map(|v| match v {
                crate::envelope::EnvelopeVolume::Cone { base_y, apex_y, .. } => base_y.min(*apex_y),
                crate::envelope::EnvelopeVolume::Cylinder { base_y, top_y, .. } => base_y.min(*top_y),
                crate::envelope::EnvelopeVolume::Ellipsoid { center, radii } => center[1] - radii[1],
            })
            .fold(f32::INFINITY, f32::min);

        let sk = grow(&params);
        let stubs: Vec<&crate::SkeletonNode> = sk
            .stem_runs()
            .iter()
            .filter(|run| {
                let n = &sk.nodes[run[0] as usize];
                run.len() == 1 && n.level == 1 && n.position.y < crown_base
            })
            .map(|run| &sk.nodes[run[0] as usize])
            .collect();
        assert!(stubs.len() > 5, "only {} dead stubs below the crown", stubs.len());
        // Dead is not decoration on a stub: the mesher reads it to end the tube in a
        // flat break instead of a twig's taper, and the foliage pass reads it to keep
        // leaves off. A stub that is born alive is a live branchlet on a clear bole.
        let alive = stubs.iter().filter(|n| !n.dead).count();
        assert_eq!(alive, 0, "{alive} of {} stubs were left alive", stubs.len());
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
        //
        // The spruce, because it leans and its spire pinches in hard enough that a
        // leaning top falls outside a crown centred on the world axis. The pine it used
        // to use now stands straight under a broad crown, and would test nothing.
        let params = parse_species(SPRUCE_RON).unwrap();
        let env = params.grown_envelope();
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
    #[test]
    fn a_shade_line_kills_the_limbs_under_it_and_keeps_their_wood() {
        // A tree overtops its own lowest limbs and they die where they stand: dead,
        // carrying their twigs, but none of their foliage. Everything under the shade
        // line is lost and everything over it lives, measured along the length the
        // trunk actually grew.
        let mut params = SpeciesParams {
            max_levels: 3,
            max_split_depth: 0,
            branch_levels: vec![StemParams::branch_default(1), StemParams::branch_default(2)],
            ..Default::default()
        };
        params.envelope.volumes = vec![crate::envelope::EnvelopeVolume::Ellipsoid {
            center: [0.0, 0.0, 0.0],
            radii: [500.0, 500.0, 500.0],
        }];
        params.trunk.split_probability = 0.0;
        params.trunk.children.pattern = ChildPattern::Whorl { every: 1, count: 3 };
        params.trunk.children.start_fraction = 0.05;
        params.trunk.children.shade_line = 0.5;
        params.trunk.children.shade_blend = 0.0;
        for level in &mut params.branch_levels {
            level.split_probability = 0.0;
            level.dieback = 0.0;
            level.snap_radius = 0.0;
        }
        params.branch_levels[0].children.pattern = ChildPattern::Continuous { density: 2.0 };

        let sk = grow(&params);
        let height = sk
            .nodes
            .iter()
            .filter(|n| n.level == 0)
            .map(|n| n.position.y)
            .fold(0.0f32, f32::max);
        let (mut below, mut above) = (0, 0);
        for run in sk.stem_runs() {
            let head = &sk.nodes[run[0] as usize];
            if head.level != 1 {
                continue;
            }
            let attach = sk.nodes[head.parent.unwrap() as usize].position.y;
            // Stay clear of the line itself, where a limb's attachment and the trunk's
            // arc length can disagree by a segment.
            if attach < height * 0.42 {
                below += 1;
                assert!(head.dead, "a limb at {attach:.1} m of {height:.1} lived under the shade line");
            } else if attach > height * 0.58 {
                above += 1;
                assert!(!head.dead, "a limb at {attach:.1} m of {height:.1} died over the shade line");
            }
        }
        assert!(below > 5 && above > 5, "only {below} limbs below and {above} above");
        // It keeps what it grew: dead wood with its twigs still on it.
        let dead_twigs = sk.nodes.iter().filter(|n| n.dead && n.level == 2 && !n.broken).count();
        assert!(dead_twigs > 20, "the shaded-out limbs carry only {dead_twigs} twig nodes");

        // Shading decides what is dead, not what grew: the same tree without it has
        // exactly the same wood, and none of it is dead. It is off by default.
        params.trunk.children.shade_line = 0.0;
        let unshaded = grow(&params);
        assert_eq!(unshaded.nodes.len(), sk.nodes.len(), "shading changed how the tree grew");
        assert!(unshaded.nodes.iter().all(|n| !n.dead));
    }
    #[test]
    fn shade_keep_cuts_the_oldest_dead_limbs_back_hardest() {
        // A limb shaded out long ago stopped growing then and has lost its end since,
        // so the lowest dead limbs are the shortest and the ones that only just died
        // under the crown keep nearly everything. `shade_keep` is the share kept at the
        // parent's base, rising to all of it at the shade line.
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
        params.trunk.children.pattern = ChildPattern::Whorl { every: 1, count: 3 };
        params.trunk.children.start_fraction = 0.05;
        params.trunk.children.scale_variance = 0.0;
        params.trunk.children.dominance = 0.0;
        params.trunk.children.shade_line = 0.6;
        params.trunk.children.shade_blend = 0.0;
        let limb = &mut params.branch_levels[0];
        limb.split_probability = 0.0;
        limb.snap_radius = 0.0;
        limb.length_variance = 0.0;

        // Total standing length of dead limbs attached in a band of the trunk's height.
        let dead_length = |params: &SpeciesParams, lo: f32, hi: f32| -> f32 {
            let sk = grow(params);
            let height = sk.stats().height;
            sk.stem_runs()
                .iter()
                .filter(|run| {
                    let head = &sk.nodes[run[0] as usize];
                    let at = sk.nodes[head.parent.unwrap_or(0) as usize].position.y / height;
                    head.level == 1 && head.dead && at >= lo && at < hi
                })
                .map(|run| stem_length(&sk, run))
                .sum()
        };

        params.trunk.children.shade_keep = 1.0;
        let (low_full, high_full) = (dead_length(&params, 0.0, 0.2), dead_length(&params, 0.45, 0.58));
        params.trunk.children.shade_keep = 0.3;
        let (low_cut, high_cut) = (dead_length(&params, 0.0, 0.2), dead_length(&params, 0.45, 0.58));

        assert!(low_full > 1.0 && high_full > 1.0, "no dead limbs to cut: {low_full} {high_full}");
        assert!(
            low_cut < low_full * 0.5,
            "the lowest dead limbs kept {low_cut:.1} m of {low_full:.1} m at a keep of 0.3"
        );
        assert!(
            high_cut > high_full * 0.75,
            "limbs that only just died kept only {high_cut:.1} m of {high_full:.1} m"
        );
    }
    #[test]
    fn the_crown_stretches_with_the_trunk() {
        // The envelope is drawn in metres for one trunk length. Declared longer, the
        // trunk used to climb out of the top of it and come out as a bare spike over
        // the old crown; with `for_trunk_length` set the crown stretches to follow.
        let bare_tip = |params: &SpeciesParams| -> f32 {
            let sk = grow(params);
            let height = sk.stats().height;
            let top = sk
                .nodes
                .iter()
                .filter(|n| n.level > 0 && !n.dead)
                .map(|n| n.position.y)
                .fold(0.0f32, f32::max);
            (height - top) / height
        };
        let mut params = parse_species(PINE_RON).unwrap();
        assert!(
            params.envelope.for_trunk_length > 0.0,
            "the fixture has to say what its envelope was drawn for"
        );
        let at_preset = bare_tip(&params);
        params.trunk.length *= 1.6;
        let taller = bare_tip(&params);
        assert!(
            taller < at_preset + 0.05,
            "a trunk half again as long left {:.0}% of the tree bare at the top against \
             {:.0}% at the preset's length",
            taller * 100.0,
            at_preset * 100.0
        );
        // And the reference is what does it: absolute volumes leave the spike.
        params.envelope.for_trunk_length = 0.0;
        let absolute = bare_tip(&params);
        assert!(
            absolute > taller + 0.15,
            "with absolute volumes the crown should have stayed put and the leader come \
             out bare, but the bare tip is {:.0}% against {:.0}% stretched",
            absolute * 100.0,
            taller * 100.0
        );
    }
    #[test]
    fn a_dead_limb_settles_and_a_living_one_does_not() {
        // Dead limbs are shaped as though alive, so without this they radiate from the
        // trunk as tidily as the crown does. With it every dead subtree tilts about its
        // base and nothing living moves at all.
        let mut params = parse_species(PINE_RON).unwrap();
        params.trunk.children.dead_sag_deg = 0.0;
        let held = grow(&params);
        params.trunk.children.dead_sag_deg = 40.0;
        let settled = grow(&params);
        assert_eq!(held.nodes.len(), settled.nodes.len());

        let (mut dead_limbs, mut dropped, mut living_moved) = (0, 0, 0);
        for (a, b) in held.nodes.iter().zip(settled.nodes.iter()) {
            if !a.dead {
                if (a.position - b.position).length() > 1e-5 {
                    living_moved += 1;
                }
                continue;
            }
            // A dead limb's own head, off the trunk.
            let Some(p) = a.parent else { continue };
            if held.nodes[p as usize].dead || a.level != 1 {
                continue;
            }
            dead_limbs += 1;
            // Its tip is the deepest node of its subtree; any drop at all counts, and a
            // limb that drew a small share of the sag may drop only a little.
            if b.position.y < a.position.y - 1e-4 {
                dropped += 1;
            }
        }
        assert_eq!(living_moved, 0, "settling dead limbs moved living wood");
        assert!(dead_limbs > 20, "only {dead_limbs} dead limbs to settle");
        assert!(
            dropped * 4 > dead_limbs * 3,
            "only {dropped} of {dead_limbs} dead limbs settled at all"
        );
    }
    #[test]
    fn the_shade_line_carries_on_through_a_fork() {
        // A trunk that forks is one axis, and where a limb sits on it is where it sits
        // on the tree. Measuring the shade line from the fork instead restarts it at
        // zero there, and a whorl just past a fork near the top of the tree is killed
        // as though it were at the foot of the trunk.
        let mut params = SpeciesParams {
            max_levels: 2,
            max_split_depth: 1,
            ..Default::default()
        };
        params.envelope.volumes = vec![crate::envelope::EnvelopeVolume::Ellipsoid {
            center: [0.0, 0.0, 0.0],
            radii: [500.0, 500.0, 500.0],
        }];
        params.trunk.split_probability = 1.0;
        params.trunk.split_start_fraction = 0.55;
        params.trunk.split_evenness = 0.9;
        params.trunk.children.pattern = ChildPattern::Whorl { every: 1, count: 3 };
        params.trunk.children.start_fraction = 0.05;
        params.trunk.children.shade_line = 0.5;
        params.trunk.children.shade_blend = 0.0;
        params.branch_levels[0].split_probability = 0.0;
        params.branch_levels[0].dieback = 0.0;

        let sk = grow(&params);
        let height = sk.stats().height;
        let trunk_stem = sk.nodes[0].stem;
        let (mut on_fork, mut dead_on_fork) = (0, 0);
        for run in sk.stem_runs() {
            let head = &sk.nodes[run[0] as usize];
            let Some(p) = head.parent else { continue };
            let parent = &sk.nodes[p as usize];
            // A limb on a fork of the trunk, well above the shade line.
            if head.level != 1 || parent.level != 0 || parent.stem == trunk_stem {
                continue;
            }
            if parent.position.y < height * 0.65 {
                continue;
            }
            on_fork += 1;
            dead_on_fork += usize::from(head.dead);
        }
        assert!(on_fork > 5, "only {on_fork} limbs found on a fork above the shade line");
        assert_eq!(
            dead_on_fork, 0,
            "{dead_on_fork} of {on_fork} limbs high on a fork were shaded out as though at the foot"
        );
    }
}
