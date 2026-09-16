use arbor_core::species::{parse_species, OAK_RON, PINE_RON};
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

#[test]
fn node_budget_is_respected() {
    for (_, src) in arbor_core::species::builtin_presets() {
        let params = parse_species(src).unwrap();
        let sk = grow(&params);
        assert!(sk.nodes.len() < 150_000, "node explosion: {}", sk.nodes.len());
    }
}
