//! How the tree hangs together, for a renderer to sway it.
//!
//! Wind is animated on the GPU, per vertex, and the renderer knows nothing of the
//! skeleton. What it gets instead is, for every vertex, the wood that vertex hangs off:
//! for each branch order — the limb off the trunk, the branch off that limb, and the
//! twigs off that branch taken together — where the stem of that order is attached, and
//! how far a point out along it swings. The shader bends each order about its own
//! attachment, finest first, and then the whole tree about its foot. A twig is carried
//! by its branch and the branch by its limb, so a leaf moves with everything under it
//! and adds its own motion on top, and nothing tears apart at a junction because a
//! stem's first point swings exactly as far as the point of its parent it leaves from.
//!
//! The trunk needs no data of its own: its bend is a function of height alone, and the
//! renderer has the height.

use glam::Vec3;

use crate::skeleton::Skeleton;

/// Branch orders a vertex records: limbs, branches, and everything finer as one.
pub const SWAY_ORDERS: usize = 3;

/// Per vertex, one entry per order: the point the stem of that order is attached at
/// (xyz), and how far a point there swings for a unit of flexibility (w, in metres).
/// An order the vertex does not reach — the trunk has none, a limb has only the first
/// — carries zero weight, and its pivot means nothing.
pub type Sway = [[f32; 4]; SWAY_ORDERS];

/// Whether a stem at `level` bends as part of `order`. The last order takes every level
/// from its own down, because twigs, sprigs and the shoots on them are too short against
/// one another for their bending to be told apart.
fn in_order(order: usize, level: u8) -> bool {
    let level = usize::from(level);
    if order + 1 == SWAY_ORDERS {
        level > order
    } else {
        level == order + 1
    }
}

/// Deflection along a cantilever under an even load, 0 at the fixed end and 1 at the
/// free one. A limb in the wind is exactly that, and the shape is what keeps the wood
/// near an attachment nearly still while the tip does most of the moving.
pub fn cantilever(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (6.0 - 4.0 * x + x * x) / 3.0
}

/// Where one node hangs in one order.
#[derive(Clone, Copy, Debug, Default)]
struct Hang {
    /// Where the stem of this order is attached.
    pivot: Vec3,
    /// How far along that stem the node is carried from, in metres. For a node past
    /// the order — a twig, seen as part of its limb — this is the point on the limb
    /// the twig's ancestry leaves from, so the whole twig rides that one point.
    arc: f32,
    /// First node of the stem of this order, which is what its reach is filed under.
    /// `None` for a node that does not reach the order.
    root: Option<u32>,
}

/// Every node's place in each branch order, worked out once per skeleton.
pub struct SwayField {
    nodes: Vec<[Hang; SWAY_ORDERS]>,
    /// Furthest any node reaches along the stem of an order, filed under that stem's
    /// first node. It sets how far the stem swings: a long limb carries its tip further
    /// than a short one bent through the same angle.
    reach: Vec<f32>,
}

impl SwayField {
    pub fn new(sk: &Skeleton) -> Self {
        let mut nodes: Vec<[Hang; SWAY_ORDERS]> = Vec::with_capacity(sk.nodes.len());
        let mut reach = vec![0.0f32; sk.nodes.len()];
        for (i, node) in sk.nodes.iter().enumerate() {
            let mut here = [Hang::default(); SWAY_ORDERS];
            if let Some(p) = node.parent {
                let p = p as usize;
                // Growth only ever appends, so a parent is always already filled in.
                debug_assert!(p < i, "node {i} has a later parent {p}");
                let parent = &sk.nodes[p];
                let up = nodes[p];
                let seg = (node.position - parent.position).length();
                for (o, hang) in here.iter_mut().enumerate() {
                    *hang = if !in_order(o, node.level) {
                        // Short of this order there is nothing to carry. Past it, the
                        // node rides wherever its ancestry left the order, and the
                        // parent already knows where that was.
                        if usize::from(node.level) <= o { Hang::default() } else { up[o] }
                    } else if in_order(o, parent.level) {
                        // Further along the same stem, or along a fork off it, which
                        // bends with the stem it forked from.
                        Hang { pivot: up[o].pivot, arc: up[o].arc + seg, root: up[o].root }
                    } else {
                        Hang { pivot: parent.position, arc: seg, root: Some(i as u32) }
                    };
                    if in_order(o, node.level)
                        && !node.broken
                        && let Some(root) = hang.root
                    {
                        let r = &mut reach[root as usize];
                        *r = r.max(hang.arc);
                    }
                }
            }
            nodes.push(here);
        }
        Self { nodes, reach }
    }

    /// The sway of node `i` itself.
    pub fn node(&self, i: usize) -> Sway {
        let mut out = [[0.0; 4]; SWAY_ORDERS];
        for (o, hang) in self.nodes[i].iter().enumerate() {
            out[o] = self.entry(hang.pivot, hang.arc, hang.root);
        }
        out
    }

    /// The sway along the stem whose first node is `first`, measured from where it
    /// leaves its parent — the same place the bark tube and the leaf walk start from.
    pub fn stem(&self, sk: &Skeleton, first: usize) -> StemSway {
        let mut orders = [OrderSway::default(); SWAY_ORDERS];
        let node = &sk.nodes[first];
        if let Some(p) = node.parent {
            let seg = (node.position - sk.nodes[p as usize].position).length();
            for (o, out) in orders.iter_mut().enumerate() {
                let hang = self.nodes[first][o];
                let Some(root) = hang.root else { continue };
                let grows = in_order(o, node.level);
                *out = OrderSway {
                    pivot: hang.pivot,
                    start: if grows { hang.arc - seg } else { hang.arc },
                    grows,
                    reach: self.reach[root as usize],
                };
            }
        }
        StemSway { orders }
    }

    fn entry(&self, pivot: Vec3, arc: f32, root: Option<u32>) -> [f32; 4] {
        let Some(root) = root else { return [0.0; 4] };
        [pivot.x, pivot.y, pivot.z, weight(arc, self.reach[root as usize])]
    }
}

/// How far a point `arc` along a stem of the given reach swings, per unit flexibility.
fn weight(arc: f32, reach: f32) -> f32 {
    if reach <= 1e-6 {
        0.0
    } else {
        reach * cantilever(arc / reach)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OrderSway {
    pivot: Vec3,
    /// Arc of this order at the stem's attachment.
    start: f32,
    /// Whether the arc runs on along this stem, or holds at `start` because the stem is
    /// finer than the order and rides one point of it.
    grows: bool,
    reach: f32,
}

/// The sway of one stem, ready to be read off at any distance along it.
#[derive(Clone, Copy, Debug, Default)]
pub struct StemSway {
    orders: [OrderSway; SWAY_ORDERS],
}

impl StemSway {
    /// The sway `along` metres from where the stem leaves its parent.
    pub fn at(&self, along: f32) -> Sway {
        let mut out = [[0.0; 4]; SWAY_ORDERS];
        for (o, s) in self.orders.iter().enumerate() {
            if s.reach <= 0.0 {
                continue;
            }
            let arc = s.start + if s.grows { along.max(0.0) } else { 0.0 };
            out[o] = [s.pivot.x, s.pivot.y, s.pivot.z, weight(arc, s.reach)];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, OAK_RON, PINE_RON};

    #[test]
    fn a_cantilever_is_still_at_its_root_and_free_at_its_tip() {
        assert_eq!(cantilever(0.0), 0.0);
        assert!((cantilever(1.0) - 1.0).abs() < 1e-6);
        let mut last = 0.0;
        for i in 1..=20 {
            let y = cantilever(i as f32 / 20.0);
            assert!(y > last, "the bend has to grow all the way out");
            last = y;
        }
        // A loaded beam bends most at its root, so the first half of it moves less
        // than half as far as the tip does — the whole point of the shape.
        assert!(cantilever(0.5) < 0.4);
    }

    #[test]
    fn the_trunk_hangs_off_nothing() {
        let params = parse_species(OAK_RON).unwrap();
        let sk = crate::grow(&params);
        let field = SwayField::new(&sk);
        for (i, node) in sk.nodes.iter().enumerate() {
            if node.level == 0 {
                assert!(
                    field.node(i).iter().all(|o| o[3] == 0.0),
                    "trunk node {i} carries branch sway"
                );
            }
        }
    }

    #[test]
    fn every_stem_leaves_its_parent_swinging_as_the_parent_does() {
        // Anything else tears the tree apart at the junctions the moment it moves.
        for src in [OAK_RON, PINE_RON] {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let field = SwayField::new(&sk);
            let mut checked = 0;
            for run in sk.stem_runs() {
                let first = run[0] as usize;
                let Some(parent) = sk.nodes[first].parent else { continue };
                let at_base = field.stem(&sk, first).at(0.0);
                let parent_sway = field.node(parent as usize);
                for o in 0..SWAY_ORDERS {
                    let (a, b) = (at_base[o], parent_sway[o]);
                    assert!(
                        (a[3] - b[3]).abs() < 1e-4,
                        "{}: stem at node {first} leaves order {o} at weight {} where its parent swings {}",
                        params.name,
                        a[3],
                        b[3]
                    );
                    if a[3] > 0.0 {
                        let gap = (Vec3::from_slice(&a[..3]) - Vec3::from_slice(&b[..3])).length();
                        assert!(gap < 1e-4, "stem at node {first} bends order {o} about another pivot");
                    }
                }
                // And reading the stem at its first node gives that node.
                let seg = (sk.nodes[first].position - sk.nodes[parent as usize].position).length();
                let along = field.stem(&sk, first).at(seg);
                let own = field.node(first);
                for o in 0..SWAY_ORDERS {
                    assert!((along[o][3] - own[o][3]).abs() < 1e-4);
                }
                checked += 1;
            }
            assert!(checked > 100, "{}: only {checked} junctions", params.name);
        }
    }

    #[test]
    fn a_limb_swings_most_at_its_tip() {
        let params = parse_species(OAK_RON).unwrap();
        let sk = crate::grow(&params);
        let field = SwayField::new(&sk);
        let mut limbs = 0;
        for run in sk.stem_runs() {
            let first = run[0] as usize;
            if sk.nodes[first].level != 1 {
                continue;
            }
            let weights: Vec<f32> = run.iter().map(|&i| field.node(i as usize)[0][3]).collect();
            assert!(
                weights.windows(2).all(|w| w[1] >= w[0] - 1e-6),
                "a limb swings less further out: {weights:?}"
            );
            // Finer orders are not the limb's business.
            assert!(run.iter().all(|&i| field.node(i as usize)[1][3] == 0.0));
            limbs += 1;
        }
        assert!(limbs > 5);
    }

    #[test]
    fn a_twig_rides_the_point_of_its_limb_it_grows_from() {
        // Seen as part of the limb, every node of a twig is carried by the one point of
        // the limb the twig leaves from; its own bending is the finer order's job.
        let params = parse_species(OAK_RON).unwrap();
        let sk = crate::grow(&params);
        let field = SwayField::new(&sk);
        let mut twigs = 0;
        for run in sk.stem_runs() {
            let first = run[0] as usize;
            if sk.nodes[first].level != 2 {
                continue;
            }
            let base = field.node(first)[0];
            for &i in &run {
                let limb = field.node(i as usize)[0];
                assert_eq!(limb, base, "twig node {i} moves with the limb as a different point");
            }
            assert!(field.node(*run.last().unwrap() as usize)[1][3] > 0.0);
            twigs += 1;
        }
        assert!(twigs > 20);
    }
}
