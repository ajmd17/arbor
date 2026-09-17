use arbor_core::species::{
    parse_species, BIRCH_RON, DOUGLAS_FIR_RON, OAK_RON, PINE_RON, SPRUCE_RON,
};
use arbor_core::grow;

fn stems_at_level(
    skeleton: &arbor_core::Skeleton,
    level: u8,
) -> (usize, usize) {
    let mut stems = 0;
    let mut nodes = 0;
    for node in &skeleton.nodes {
        if node.level != level {
            continue;
        }
        if let Some(p) = node.parent {
            if skeleton.nodes[p as usize].level != level {
                stems += 1;
            }
        }
        nodes += 1;
    }
    (stems, nodes)
}

#[test]
fn pine_has_branches_at_every_level() {
    let params = parse_species(PINE_RON).unwrap();
    let sk = grow(&params);
    for level in 0..3u8 {
        let (stems, _) = stems_at_level(&sk, level + 1);
        assert!(stems > 10, "pine level {level} has only {stems} stems");
    }
    let stats = sk.stats();
    assert!(stats.height > 10.0, "pine height {}", stats.height);
}

#[test]
fn oak_has_split_trunk_and_dense_canopy() {
    let params = parse_species(OAK_RON).unwrap();
    let sk = grow(&params);
    let trunk_forks = sk
        .nodes
        .iter()
        .filter(|n| {
            n.level == 0
                && n.children.iter().filter(|c| sk.nodes[**c as usize].level == 0).count() >= 2
        })
        .count();
    assert!(trunk_forks >= 1, "oak should fork into co-dominant stems, got {trunk_forks} forks");
    let (level3_stems, _) = stems_at_level(&sk, 3);
    assert!(level3_stems > 20, "oak level-3 twigs: {level3_stems}");
}

/// What separates the spruce from the pine, which is the preset it is nearest to: it
/// keeps one leader for its whole height, and it wears its crown nearly to the ground
/// rather than on a bare pole.
#[test]
fn spruce_keeps_one_leader_and_a_crown_to_the_ground() {
    let params = parse_species(SPRUCE_RON).unwrap();
    let sk = grow(&params);
    let stats = sk.stats();
    assert!(stats.height > 22.0, "spruce height {}", stats.height);

    // split_evenness is 0, so a fork must stay a side branch: no two level-0 runs may
    // both be a substantial share of the tree, or the leader has been given up.
    let mut trunk_runs: Vec<f32> = sk
        .stem_runs()
        .into_iter()
        .filter(|r| r.len() > 1 && sk.nodes[r[0] as usize].level == 0)
        .map(|r| {
            r.windows(2)
                .map(|w| {
                    (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length()
                })
                .sum()
        })
        .collect();
    trunk_runs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if let Some(second) = trunk_runs.get(1) {
        assert!(
            *second < trunk_runs[0] * 0.5,
            "spruce grew a co-dominant trunk: runs of {:.1} m and {second:.1} m",
            trunk_runs[0]
        );
    }

    // Foliage in the bottom fifth of the tree, which is what a pine does not have.
    let leaves = arbor_core::build_leaves(&sk, &params);
    // Counted in card corners, which is all the mesh keeps; four to a card.
    let low = leaves
        .positions
        .iter()
        .filter(|p| p[1] < stats.height * 0.2)
        .count();
    assert!(
        low > 400,
        "spruce carries only {} cards below {:.1} m",
        low / 4,
        stats.height * 0.2
    );
}

/// The douglas fir is the opposite tree to the spruce and the test is the mirror of
/// it: a clear bole over the bottom third, and a crown that is the sparser of the two.
#[test]
fn douglas_fir_has_a_clear_bole_and_an_open_crown() {
    let params = parse_species(DOUGLAS_FIR_RON).unwrap();
    let sk = grow(&params);
    let stats = sk.stats();
    assert!(stats.height > 44.0, "douglas fir height {}", stats.height);

    // Where the foliage starts is the whole difference between these two presets, and
    // it is worth stating as one assertion over both: the fir self-prunes its bottom
    // third away, the spruce keeps its crown to the ground.
    let spruce = parse_species(SPRUCE_RON).unwrap();
    let spruce_sk = grow(&spruce);
    let fir_base = lowest_leaf_fraction(&sk, &params);
    let spruce_base = lowest_leaf_fraction(&spruce_sk, &spruce);
    assert!(
        fir_base > 0.3,
        "douglas fir carries foliage from {:.0}% of its height; the bottom third          should be clear bole",
        fir_base * 100.0
    );
    assert!(
        spruce_base < 0.15,
        "spruce carries foliage only from {:.0}% of its height; its crown should          reach the ground",
        spruce_base * 100.0
    );

    // Sparser, too — and the honest way to say that is against the spruce, which is
    // the same needles on the opposite habit. An absolute figure would just be whatever
    // this preset happened to measure on the day.
    let open = crown_card_area_per_m3(&sk, &params);
    let solid = crown_card_area_per_m3(&spruce_sk, &spruce);
    assert!(
        open < solid * 0.8,
        "douglas fir carries {open:.2} square metres of card per cubic metre against          the spruce's {solid:.2}; it is meant to be the open one"
    );
}

/// Height of the lowest leaf card, as a fraction of the tree's own height.
fn lowest_leaf_fraction(sk: &arbor_core::Skeleton, params: &arbor_core::SpeciesParams) -> f32 {
    let leaves = arbor_core::build_leaves(sk, params);
    let base = leaves
        .positions
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min);
    base / sk.stats().height
}

/// Square metres of leaf card per cubic metre of the crown's own bounding cylinder, the
/// crown being whatever part of the tree carries foliage.
///
/// Area, not a count. A card is a unit of geometry and not a unit of foliage, so
/// counting them prices the wrong thing: two crowns carrying exactly the same foliage
/// score differently the moment their `card_length` differs, and the species that draws
/// its canopy on smaller cards is punished for it. That is not a theoretical worry — it
/// is what the count did to the fir. Holding coverage while halving a card costs four
/// times the cards, so under a count the openness a fir is meant to have and the fine
/// spray grain it is meant to have were bidding against each other, and the preset
/// could not have both. Measuring the area they cover leaves that choice free: draw the
/// same canopy on many small cards or a few large ones and this number does not move.
///
/// What it deliberately does not fold in is how much of a card the alpha actually keeps.
/// That would need the baked atlas, and so the source art, in a test that otherwise
/// touches no files — and both conifers grow from `needle_conifer`, so for the one
/// comparison this makes it would very nearly cancel anyway.
fn crown_card_area_per_m3(sk: &arbor_core::Skeleton, params: &arbor_core::SpeciesParams) -> f32 {
    let leaves = arbor_core::build_leaves(sk, params);
    if leaves.positions.is_empty() {
        return 0.0;
    }
    let (mut base, mut top, mut r2) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
    for p in &leaves.positions {
        base = base.min(p[1]);
        top = top.max(p[1]);
        r2 = r2.max(p[0] * p[0] + p[2] * p[2]);
    }
    let volume = std::f32::consts::PI * r2 * (top - base).max(0.1);
    let card = params.leaves.card_length * params.leaves.card_width;
    (leaves.positions.len() / 4) as f32 * card / volume
}

#[test]
fn node_budget_is_respected() {
    for (_, src) in arbor_core::species::builtin_presets() {
        let params = parse_species(src).unwrap();
        let sk = grow(&params);
        assert!(sk.nodes.len() < 150_000, "node explosion: {}", sk.nodes.len());
    }
}

/// Leaves are the cheap half of a tree to author and the expensive half to draw, and
/// nothing was watching them.
///
/// `douglas_fir_has_a_clear_bole_and_an_open_crown` used to count cards per cubic metre,
/// which meant that while it was busy asking the wrong question about openness it was
/// quietly answering the right one about cost — a preset could not run away with the
/// card count without tripping it. Measuring area instead frees the count on purpose,
/// so the budget it was standing in for has to be said out loud, and said for every
/// species rather than for the one that happened to be compared against another.
#[test]
fn leaf_budget_is_respected() {
    for (name, src) in arbor_core::species::builtin_presets() {
        let params = parse_species(src).unwrap();
        let sk = grow(&params);
        let leaves = arbor_core::build_leaves(&sk, &params);
        assert!(
            leaves.triangle_count() < 150_000,
            "{name} draws {} leaf triangles",
            leaves.triangle_count()
        );
    }
}

/// The furthest any one stem turns over its length, in degrees, and the tightest
/// bend reached, in degrees per metre. A stem that turns far enough has come round on
/// itself, which is what an unbounded pull toward the crown axis produces: whatever
/// its strength there is a radius at which it supplies exactly the turn a circle
/// needs, and the stem rides it round instead of settling.
/// The worst net turn any stem makes between setting out and finishing, in degrees,
/// and the worst ratio of how far a stem got to how far it travelled.
///
/// Both are about curling, and neither counts the jog a stem makes at every node.
/// Summing the turn between consecutive segments would: a shoot that alternates a few
/// degrees each way banks a large total while going perfectly straight, and that jog
/// is wanted. What is not wanted is a stem that comes round on itself, which shows up
/// as a large net turn, or as a path far longer than the distance it covered.
fn worst_stem_turn(sk: &arbor_core::Skeleton) -> (f32, f32) {
    let (mut worst, mut worst_wander) = (0.0f32, 1.0f32);
    for run in sk.stem_runs() {
        let mut dirs = Vec::new();
        let mut length = 0.0f32;
        for pair in run.windows(2) {
            let step = sk.nodes[pair[1] as usize].position - sk.nodes[pair[0] as usize].position;
            if step.length() > 1e-6 {
                length += step.length();
                dirs.push(step.normalize());
            }
        }
        let (Some(first), Some(last)) = (dirs.first(), dirs.last()) else {
            continue;
        };
        worst = worst.max(first.dot(*last).clamp(-1.0, 1.0).acos().to_degrees());
        // Short stems are two or three segments long, so their straightness says more
        // about the segment count than about their shape.
        if length > 1.0 {
            let reach = sk.nodes[run[run.len() - 1] as usize].position
                - sk.nodes[run[0] as usize].position;
            worst_wander = worst_wander.min(reach.length() / length);
        }
    }
    (worst, worst_wander)
}

#[test]
fn stems_arc_without_curling_round_on_themselves() {
    for (name, ron) in [
        ("pine", PINE_RON),
        ("oak", OAK_RON),
        ("birch", BIRCH_RON),
        ("spruce", SPRUCE_RON),
        ("fir", DOUGLAS_FIR_RON),
    ] {
        let params = parse_species(ron).unwrap();
        for seed in 1..=8u64 {
            let mut params = params.clone();
            params.seed = seed;
            let sk = grow(&params);
            let (turn, straightness) = worst_stem_turn(&sk);
            // A limb may sweep from upright to below the horizontal on its way out —
            // that is the whole shape of an oak limb — but it may not come round.
            assert!(
                turn < 115.0,
                "{name} seed {seed} has a stem turning {turn:.0} degrees end to end"
            );
            assert!(
                straightness > 0.6,
                "{name} seed {seed} has a stem covering only {straightness:.2} of its own length"
            );
        }
    }
}
