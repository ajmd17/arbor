use arbor_core::species::{
    parse_species, BIRCH_RON, DOUGLAS_FIR_OPEN_RON, DOUGLAS_FIR_RON, OAK_RON, PINE_RON,
    SPRUCE_RON,
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
    // this preset happened to measure on the day. Needle rather than card, since the
    // two carry their needle at different densities per card.
    let open = crown_needle_area_per_m3(&sk, &params);
    let solid = crown_needle_area_per_m3(&spruce_sk, &spruce);
    assert!(
        open < solid * 0.8,
        "douglas fir carries {open:.2} square metres of needle per cubic metre against          the spruce's {solid:.2}; it is meant to be the open one"
    );
}

/// The open-grown fir is the same species on the opposite upbringing, and the risk it
/// runs is the opposite one.
///
/// `douglas_fir.ron` has to be kept from turning into a spruce by its outline, and
/// `douglas_fir_has_a_clear_bole_and_an_open_crown` is what does it. This tree gives
/// that defence up on purpose: grown in the open it keeps its lower limbs, so it *is* a
/// broad cone carried most of the way to the ground, which is the spruce's outline. What
/// still separates them is the limb, and so that is what this asserts. A fir limb sets
/// off below horizontal and its branchlets turn their last third back up to the light; a
/// spruce's leave level and hang. Measured as where a branchlet's tip finishes against
/// where it set out, the fir's rise and the spruce's fall, and no amount of shared
/// silhouette changes that.
#[test]
fn open_grown_douglas_fir_is_a_sparse_spire_that_still_hangs() {
    let params = parse_species(DOUGLAS_FIR_OPEN_RON).unwrap();
    let sk = grow(&params);
    let stats = sk.stats();

    // A yard tree, not a forest giant. The forest preset is the one that goes to fifty.
    let forest = parse_species(DOUGLAS_FIR_RON).unwrap();
    let forest_sk = grow(&forest);
    assert!(
        (24.0..38.0).contains(&stats.height),
        "open-grown fir is {:.0} m; it is meant to be a yard tree",
        stats.height
    );
    assert!(
        stats.height < forest_sk.stats().height * 0.75,
        "open-grown fir is {:.0} m against the forest one's {:.0}; it should be much the \
         shorter",
        stats.height,
        forest_sk.stats().height
    );

    // It keeps its lower limbs, which is the whole of what growing in the open means.
    let open_base = lowest_leaf_fraction(&sk, &params);
    let forest_base = lowest_leaf_fraction(&forest_sk, &forest);
    assert!(
        open_base < 0.25,
        "open-grown fir carries foliage only from {:.0}% of its height; it never \
         self-pruned and should be leafy far lower",
        open_base * 100.0
    );
    assert!(
        open_base < forest_base * 0.6,
        "open-grown fir starts its crown at {:.0}% against the forest one's {:.0}%; the \
         two habits are supposed to be unmistakable",
        open_base * 100.0,
        forest_base * 100.0
    );

    // A spire: narrow, and still a cone — widest low down, narrowing the whole way to
    // the leader. Bounded both ways, because the width is load-bearing in two
    // directions. Too broad and it reads as a garden conifer rather than a forest one;
    // too narrow and the envelope has cut the limbs so short that there is no crown left
    // to speak of, which is not a silhouette problem but a foliage one.
    let profile = crown_radius_profile(&sk, &params);
    let widest = profile.iter().cloned().fold(0.0f32, f32::max);
    let slenderness = widest * 2.0 / stats.height;
    assert!(
        (0.22..0.36).contains(&slenderness),
        "open-grown fir is {slenderness:.2} as wide as it is tall; a forest conifer is a \
         spire, and neither a garden cone nor a pole"
    );
    let lower = profile[2..5].iter().sum::<f32>() / 3.0;
    let upper = profile[7..10].iter().sum::<f32>() / 3.0;
    assert!(
        lower > upper * 1.3,
        "open-grown fir measures {lower:.1} m across its lower crown against {upper:.1} \
         up top; it is meant to taper the whole way, not stand up like a column"
    );

    // Sparser than the spruce, which had no assertion and went wrong exactly because of
    // that: derived from the forest preset it came out at 82% of the spruce's card area
    // per cubic metre and 119% of the needle you look through side on, so the fir was
    // the denser of the two. Openness cannot come from the outline here, because the
    // outline is the spruce's, so it has to come from the branchwork being thinner and
    // this is what holds it there.
    let open_area = crown_needle_area_per_m3(&sk, &params);
    let spruce_params = parse_species(SPRUCE_RON).unwrap();
    let spruce_grown = grow(&spruce_params);
    let solid = crown_needle_area_per_m3(&spruce_grown, &spruce_params);
    // Needle rather than card, because the two carry very different amounts of it per
    // card: this fir draws one airy photographed spray to a card, the spruce a baked
    // bundle of thirteen shoots.
    //
    // The ceiling has moved twice. Once with the habit: this is per cubic metre, and
    // the same foliage in a spire rather than a broad cone sits in well under half the
    // volume, so the number rises without a card being added. And once with the
    // spruce's triangle budget, which cut it to a third of its cards and left it about
    // a third less needle per cubic metre — mostly needle that sat behind other needle
    // and was never seen. This fir did not change, and measured 0.27 of the spruce
    // before that and 0.41 after; what still holds is that it is well under half as
    // dense, which is what seeing the trunk through one and not the other comes to.
    assert!(
        open_area < solid * 0.5,
        "open-grown fir carries {open_area:.2} square metres of needle per cubic metre \
         against the spruce's {solid:.2}; you are meant to see the trunk and the branch \
         tips through it"
    );
    assert!(
        open_area > solid * 0.06,
        "open-grown fir is down to {open_area:.2} against the spruce's {solid:.2}; that \
         is a skeleton, not an open crown"
    );

    // The branches slope down at the foot of the crown and point further up the higher
    // they are, and you can see that they do. This is the difference between a conifer
    // that reads as designed and one that reads as scribble, and it is a signal-to-noise
    // problem rather than a question of any one angle: the limbs of a single whorl land
    // about 17 degrees apart however they are tuned, because the envelope cuts them off
    // at different lengths and `droop` then sags them by different amounts. So the climb
    // from the bottom of the crown to the top has to be large against that, or the eye
    // cannot pick it out and the crown looks random. It was 29 degrees of climb against
    // 19 of scatter and looked like bed-head; it is now about 50 against 17.
    let (climb, scatter) = limb_climb(&sk);
    assert!(
        climb > 35.0,
        "open-grown fir limbs climb only {climb:.0} degrees from the foot of the crown to \
         the top; they are meant to slope down low and point up high"
    );
    assert!(
        climb / scatter.max(0.01) > 2.0,
        "open-grown fir limbs climb {climb:.0} degrees against {scatter:.0} of scatter \
         within a whorl; at that ratio the structure is lost in the noise"
    );

    // No stray hairs: no level puts out a stem several times the length of its
    // neighbours. They read as whiskers shooting out of the crown, and they have come
    // back twice from different causes — first vigor overshooting the declared length,
    // which `MAX_DRIVE` now bounds, and then a declared length so far above what the
    // envelope allows that it only ever bit on the few limbs travelling *along* the
    // envelope rather than across it. Both show up here, as the spread between the
    // longest stem at a level and the ordinary one, so this catches the next cause too
    // without having to know what it is.
    //
    // The ordinary one is the long end of the level, its 90th percentile, and not its
    // median. A child is held to the branch still ahead of it, so a level's lengths now
    // spread with where each stem sits on its parent — long near the base, short near
    // the tip — and the median is mostly those short tip shoots, which say nothing
    // about whether the longest is a whisker. A whisker is one or a few stems, far too
    // few to move the 90th percentile, and it still stands well clear of it.
    for level in 1..=4u8 {
        let mut lens = grown_lengths(&sk, level);
        if lens.len() < 8 {
            continue;
        }
        lens.sort_by(|a, b| a.partial_cmp(b).expect("lengths are finite"));
        let long = lens[lens.len() * 9 / 10].max(1e-3);
        let longest = lens[lens.len() - 1];
        assert!(
            longest / long < 2.2,
            "open-grown fir level {level}: the longest stem is {longest:.2} m against \
             {long:.2} for the long end of the level, which is {:.1} times it — that is a \
             whisker, not a branch",
            longest / long
        );
    }

    // And no bristles: a stem has to be thick enough for its length to read as wood.
    // Length over base radius is what the eye judges that by — a real branch runs about
    // forty to eighty, and much past that it is a wire however correct its length
    // is. This is the other half of the stray hairs, and the half that survived two
    // passes at their length: the thickness pass was scaling every stem by raw vigor
    // instead of by its drive, so the fine levels came out at a fifth of their declared
    // radius. See `nominal_vigor` in `growth.rs`.
    for level in 1..=4u8 {
        let mut ratios = length_over_radius(&sk, level);
        if ratios.len() < 8 {
            continue;
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).expect("ratios are finite"));
        let median = ratios[ratios.len() / 2];
        assert!(
            median < 80.0,
            "open-grown fir level {level}: the typical stem is {median:.0} times as long \
             as it is thick, which is a bristle rather than a branch"
        );
    }

    // And still a fir. This is the one that matters: the outline is now the spruce's,
    // so the limb has to carry the difference on its own.
    let fir_rise = mean_tip_rise(&sk, 2);
    let spruce_rise = mean_tip_rise(&spruce_grown, 2);
    assert!(
        fir_rise > 0.05,
        "open-grown fir branchlet tips finish {fir_rise:+.2} m against where they set \
         out; a fir spray turns its last third back up to the light"
    );
    assert!(
        spruce_rise < 0.0 && fir_rise > spruce_rise + 0.2,
        "open-grown fir branchlet tips rise {fir_rise:+.2} m and the spruce's {spruce_rise:+.2}; \
         with the outlines this close that gap is the only thing left telling them apart"
    );
}

/// Every living stem at `level`, by how many times its own base radius it is long.
///
/// This is the number that decides whether a stem reads as wood or as wire, and it is
/// worth measuring rather than trusting: a species declares a `radius` per level, but
/// what a stem actually gets is that scaled by its drive and then capped by its parent,
/// so the declared figure can be several times what comes out.
fn length_over_radius(sk: &arbor_core::Skeleton, level: u8) -> Vec<f32> {
    let mut out = Vec::new();
    for run in sk.stem_runs() {
        let first = run[0] as usize;
        let node = &sk.nodes[first];
        if node.level != level || node.broken {
            continue;
        }
        let Some(parent) = node.parent else {
            continue;
        };
        let mut len = (node.position - sk.nodes[parent as usize].position).length();
        for w in run.windows(2) {
            len += (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length();
        }
        out.push(len / node.radius.max(1e-4));
    }
    out
}

/// Every living stem at `level`, by the length it actually grew.
///
/// Grown rather than declared, because the two are only loosely related: the envelope
/// cuts most stems off well short of what their level asks for, so what a species
/// declares is a ceiling on the outliers rather than a description of the typical stem.
fn grown_lengths(sk: &arbor_core::Skeleton, level: u8) -> Vec<f32> {
    let mut out = Vec::new();
    for run in sk.stem_runs() {
        let first = run[0] as usize;
        let node = &sk.nodes[first];
        if node.level != level || node.broken {
            continue;
        }
        let Some(parent) = node.parent else {
            continue;
        };
        let mut len = (node.position - sk.nodes[parent as usize].position).length();
        for w in run.windows(2) {
            len += (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length();
        }
        out.push(len);
    }
    out
}

/// How far a tree's limbs swing from sloping down at the foot of the crown to pointing up
/// at the top, in degrees, and how far apart the limbs of one whorl land, also in degrees.
///
/// Each limb is measured as the elevation of the line from where it attaches to where it
/// ends, so it is the line the eye actually follows rather than the angle it set out at —
/// a limb that leaves level and then sags is a sloping limb. The scatter is pooled
/// *within* height bands on purpose: taken over every limb at once it would count the
/// climb itself as noise, which is the one thing it must not do.
fn limb_climb(sk: &arbor_core::Skeleton) -> (f32, f32) {
    let height = sk.stats().height.max(0.1);
    let mut limbs: Vec<(f32, f32)> = Vec::new();
    for run in sk.stem_runs() {
        let first = run[0] as usize;
        if sk.nodes[first].level != 1 || sk.nodes[first].dead || sk.nodes[first].broken {
            continue;
        }
        let Some(parent) = sk.nodes[first].parent else {
            continue;
        };
        if sk.nodes[parent as usize].level != 0 {
            continue;
        }
        let from = sk.nodes[parent as usize].position;
        let to = sk.nodes[*run.last().expect("a run is never empty") as usize].position;
        let d = to - from;
        let flat = (d.x * d.x + d.z * d.z).sqrt();
        // A limb that went nowhere horizontally has no meaningful elevation.
        if flat < 0.3 {
            continue;
        }
        limbs.push((from.y / height, d.y.atan2(flat).to_degrees()));
    }
    if limbs.len() < 8 {
        return (0.0, 1.0);
    }
    let mean_of = |lo: f32, hi: f32| {
        let v: Vec<f32> = limbs
            .iter()
            .filter(|(h, _)| *h >= lo && *h < hi)
            .map(|(_, e)| *e)
            .collect();
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f32>() / v.len() as f32)
        }
    };
    let low = mean_of(0.0, 0.45).unwrap_or(0.0);
    let high = mean_of(0.75, 1.01).unwrap_or(0.0);
    let (mut ss, mut n) = (0.0f32, 0usize);
    for band in 0..10 {
        let (lo, hi) = (band as f32 / 10.0, (band + 1) as f32 / 10.0);
        let v: Vec<f32> = limbs
            .iter()
            .filter(|(h, _)| *h >= lo && *h < hi)
            .map(|(_, e)| *e)
            .collect();
        if v.len() < 2 {
            continue;
        }
        let m = v.iter().sum::<f32>() / v.len() as f32;
        ss += v.iter().map(|e| (e - m).powi(2)).sum::<f32>();
        n += v.len() - 1;
    }
    let scatter = if n == 0 { 1.0 } else { (ss / n as f32).sqrt() };
    (high - low, scatter)
}

/// Where each branchlet at `level` finishes, against where it set out, in metres. A fir
/// spray hangs and then lifts, so it ends above its own attachment; a spruce's hangs and
/// stays there.
fn mean_tip_rise(sk: &arbor_core::Skeleton, level: u8) -> f32 {
    let (mut rise, mut n) = (0.0f32, 0usize);
    for run in sk.stem_runs() {
        let first = run[0] as usize;
        if sk.nodes[first].level != level || sk.nodes[first].dead || sk.nodes[first].broken {
            continue;
        }
        let Some(parent) = sk.nodes[first].parent else {
            continue;
        };
        let from = sk.nodes[parent as usize].position;
        let to = sk.nodes[*run.last().expect("a run is never empty") as usize].position;
        // Short stems say more about the segment count than about the shape.
        if (to - from).length() < 0.3 {
            continue;
        }
        rise += to.y - from.y;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    rise / n as f32
}

/// Crown radius per tenth of the tree's height, taken at the ninetieth percentile of the
/// leaf cards in each band. A maximum would be set by one straggling branchlet hanging
/// out of the crown and would say nothing about where the foliage actually is.
fn crown_radius_profile(
    sk: &arbor_core::Skeleton,
    params: &arbor_core::SpeciesParams,
) -> Vec<f32> {
    let leaves = arbor_core::build_leaves(sk, params);
    let height = sk.stats().height.max(0.1);
    let mut bands: Vec<Vec<f32>> = vec![Vec::new(); 10];
    for p in leaves.positions.iter() {
        let band = ((p[1] / height * 10.0) as usize).min(9);
        bands[band].push((p[0] * p[0] + p[2] * p[2]).sqrt());
    }
    bands
        .iter_mut()
        .map(|b| {
            if b.is_empty() {
                return 0.0;
            }
            b.sort_by(|x, y| x.partial_cmp(y).expect("leaf radii are finite"));
            b[(b.len() as f32 * 0.9) as usize % b.len()]
        })
        .collect()
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

/// Share of a card that is foliage rather than air, read off the atlas the species
/// actually samples: its leaf art, baked into clusters first where the species asks
/// for that, exactly as the renderer does.
fn card_coverage(params: &arbor_core::SpeciesParams) -> f32 {
    let lp = &params.leaves;
    let path = format!(
        "{}/../../assets/textures/{}_albedo.png",
        env!("CARGO_MANIFEST_DIR"),
        lp.texture
    );
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let source = arbor_core::Bitmap::from_rgba(w, h, img.into_raw()).expect("rgba");
    let sheet = match &lp.cluster {
        Some(cluster) => {
            arbor_core::bake_cluster(
                cluster,
                lp.atlas_cols,
                lp.atlas_rows,
                arbor_core::LeafMaps {
                    albedo: &source,
                    normal: None,
                    roughness: None,
                },
            )
            .albedo
        }
        None => source,
    };
    sheet.mean_alpha()
}

/// Needle, not card, per cubic metre of crown: card area weighted by how much of each
/// card is foliage.
///
/// Card area alone stops meaning anything once two species carry their needle at
/// different densities per card, and that is exactly what a triangle budget does — a
/// crown drawn with a third of the cards, each baked three times as full, has a third
/// of the card area and nothing like a third less needle.
fn crown_needle_area_per_m3(sk: &arbor_core::Skeleton, params: &arbor_core::SpeciesParams) -> f32 {
    crown_card_area_per_m3(sk, params) * card_coverage(params)
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
        ("fir_open", DOUGLAS_FIR_OPEN_RON),
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

/// What makes the pine read as a forest-grown Scots pine rather than a young one: the
/// crown has climbed past the lower half of the bole and left its old limbs standing
/// dead beneath it, whole limbs with their twigs on and no foliage, over a clear bole
/// with stubs. Held across seeds, since a band that only one seed grows is luck.
#[test]
fn pine_carries_its_dead_limbs_under_a_living_crown() {
    for seed in 1..=4u64 {
        let mut params = parse_species(PINE_RON).unwrap();
        params.seed = seed;
        let sk = grow(&params);
        let height = sk.stats().height;

        // Limbs off the trunk, long enough to be limbs rather than the stubs left on
        // the bole below the crown.
        let (mut low, mut low_dead, mut high, mut high_alive) = (0, 0, 0, 0);
        for run in sk.stem_runs() {
            let head = &sk.nodes[run[0] as usize];
            let Some(parent) = head.parent else { continue };
            let parent = &sk.nodes[parent as usize];
            if head.level != 1 || parent.level != 0 || run.len() < 3 {
                continue;
            }
            let at = parent.position.y / height;
            if at < 0.45 {
                low += 1;
                low_dead += usize::from(head.dead);
            } else if at > 0.65 {
                high += 1;
                high_alive += usize::from(!head.dead);
            }
        }
        assert!(low >= 15, "seed {seed}: only {low} limbs in the dead band, so it is not a band");
        assert!(
            low_dead * 10 >= low * 9,
            "seed {seed}: {low_dead} of {low} limbs under the crown are dead"
        );
        assert!(
            high_alive * 10 >= high * 9,
            "seed {seed}: {high_alive} of {high} limbs in the crown are alive"
        );

        // The dead band is the crown the tree had when it was younger and smaller, so
        // its limbs are shorter than the living ones and most of their fine twigs have
        // fallen. It used to be grown as wide as the living crown: dead limbs reaching
        // five metres from the trunk carrying two kilometres of dead twig between them,
        // a thicket where the reference shows a sparse band. Both ends are held — the
        // floor is what stops it being thinned into a comb of bare pegs instead.
        let (mut reaches, mut twig_wood) = (Vec::new(), 0.0f32);
        for run in sk.stem_runs() {
            let head = &sk.nodes[run[0] as usize];
            let Some(parent) = head.parent else { continue };
            if !head.dead || head.position.y > height * 0.55 {
                continue;
            }
            let base = sk.nodes[parent as usize].position;
            let mut length = (head.position - base).length();
            for w in run.windows(2) {
                length += (sk.nodes[w[1] as usize].position - sk.nodes[w[0] as usize].position).length();
            }
            if head.level >= 2 {
                twig_wood += length;
            } else if head.level == 1 && run.len() >= 2 {
                let reach = run
                    .iter()
                    .map(|&i| {
                        let d = sk.nodes[i as usize].position - base;
                        (d.x * d.x + d.z * d.z).sqrt()
                    })
                    .fold(0.0f32, f32::max);
                reaches.push(reach);
            }
        }
        reaches.sort_by(f32::total_cmp);
        let p90 = reaches[reaches.len() * 9 / 10];
        assert!(
            p90 < 3.1,
            "seed {seed}: the dead limbs reach {p90:.2} m from the trunk (90th percentile)"
        );
        assert!(
            (150.0..1000.0).contains(&twig_wood),
            "seed {seed}: {twig_wood:.0} m of dead twig in the band"
        );

        // And the foliage is up in the crown, not hung on the dead band.
        let leaves = arbor_core::build_leaves(&sk, &params);
        let below = leaves
            .positions
            .iter()
            .filter(|p| p[1] < height * 0.4)
            .count();
        assert!(
            below * 10 < leaves.positions.len(),
            "seed {seed}: {below} of {} leaf vertices sit in the dead band",
            leaves.positions.len()
        );
    }
}
