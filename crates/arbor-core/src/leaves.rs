//! Foliage as textured cards.
//!
//! Each leaf is a single quad anchored on a twig, so the shape of a leaf comes from
//! the alpha channel of its texture rather than from geometry. What makes a canopy
//! built this way read as foliage instead of as a heap of flat planes is the
//! shading: card normals are blended toward the direction pointing out of the crown,
//! curved across the width of each card, and darkened toward the interior. Those
//! three together are what stop every leaf from flashing its flat plane at the light
//! all at once.

use glam::Vec3;

use crate::math::{ortho_of, ortho_unit, transport};
use crate::seed::{range_f32, TreeRng};
use crate::skeleton::Skeleton;
use crate::species::{LeafParams, SpeciesParams};

/// Ceiling on card count, so a dense preset cannot allocate without bound.
const MAX_LEAVES: usize = 400_000;
/// A leaf sits on the twig surface, not on its centreline.
const SURFACE_BIAS: f32 = 0.85;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LeafMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Card-local, 0..1 across the width and along the length. The renderer maps
    /// these into whatever atlas cell it draws with, so the same mesh works for a
    /// front face and a back face.
    pub uvs: Vec<[f32; 2]>,
    /// rgb is a per-leaf colour multiplier, a is a crown-depth shade term.
    pub tints: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl LeafMesh {
    pub fn leaf_count(&self) -> usize {
        self.positions.len() / 4
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// One card, resolved before any vertices exist. Placement has to finish first
/// because the shading needs the centre and radius of the finished crown.
struct Card {
    origin: Vec3,
    /// Along the leaf, from stem to tip.
    axis: Vec3,
    /// Across the leaf.
    side: Vec3,
    length: f32,
    width: f32,
    hue: f32,
}

pub fn build_leaves(sk: &Skeleton, params: &SpeciesParams) -> LeafMesh {
    let lp = &params.leaves;
    let mut mesh = LeafMesh::default();
    if !lp.enabled || lp.density <= 0.0 || lp.card_length <= 0.0 {
        return mesh;
    }

    let tree_rng = TreeRng::new(params.seed ^ 0x1EAF_1EAF_1EAF_1EAF);
    let mut cards: Vec<Card> = Vec::new();

    for run in sk.stem_runs() {
        if cards.len() >= MAX_LEAVES {
            break;
        }
        let first = run[0] as usize;
        if sk.nodes[first].level < lp.min_level {
            continue;
        }
        place_on_stem(sk, lp, &tree_rng, &run, &mut cards);
    }

    if cards.is_empty() {
        return mesh;
    }

    let (center, radius) = crown_bounds(&cards);
    mesh.positions.reserve(cards.len() * 4);
    mesh.normals.reserve(cards.len() * 4);
    mesh.uvs.reserve(cards.len() * 4);
    mesh.tints.reserve(cards.len() * 4);
    mesh.indices.reserve(cards.len() * 6);
    for card in &cards {
        emit_card(&mut mesh, card, lp, center, radius);
    }
    mesh
}

fn place_on_stem(
    sk: &Skeleton,
    lp: &LeafParams,
    tree_rng: &TreeRng,
    run: &[u32],
    out: &mut Vec<Card>,
) {
    // Walk from the attachment point, the same polyline the bark tube is swept
    // along, so leaves cannot start in the gap between a twig and its parent.
    let first = run[0] as usize;
    let mut points: Vec<(Vec3, f32)> = Vec::with_capacity(run.len() + 1);
    if let Some(p) = sk.nodes[first].parent {
        points.push((sk.nodes[p as usize].position, sk.nodes[first].radius));
    }
    for &i in run {
        let node = &sk.nodes[i as usize];
        points.push((node.position, node.radius));
    }
    if points.len() < 2 {
        return;
    }

    let mut rng = tree_rng.stream(sk.nodes[first].path);
    let spacing = (1.0 / lp.density).max(1e-3);
    let mut frame = ortho_of((points[1].0 - points[0].0).normalize_or(Vec3::Y));
    let mut azimuth = range_f32(&mut rng, 0.0, std::f32::consts::TAU);
    // Stagger the first leaf so twigs do not all start their sequence flush with
    // the junction.
    let mut until_next = range_f32(&mut rng, 0.0, spacing);
    let mut prev_dir = (points[1].0 - points[0].0).normalize_or(Vec3::Y);

    for w in 0..points.len() - 1 {
        let (a, r_a) = points[w];
        let (b, r_b) = points[w + 1];
        let span = b - a;
        let seg_len = span.length();
        if seg_len < 1e-5 {
            continue;
        }
        let dir = span / seg_len;
        frame = ortho_unit(transport(prev_dir, dir, frame), dir);
        prev_dir = dir;

        let mut travelled = 0.0;
        while until_next <= seg_len - travelled {
            travelled += until_next;
            until_next = spacing;
            if out.len() >= MAX_LEAVES {
                return;
            }
            let t = travelled / seg_len;
            let radius = r_a + (r_b - r_a) * t;
            if radius > lp.max_twig_radius {
                continue;
            }
            let at = a + dir * travelled;
            // Every card in a cluster shares one anchor and one base direction, and
            // fans out from it, so the canopy reads as tufts rather than a uniform
            // spray of evenly spaced leaves.
            azimuth += lp.phyllotaxis_deg.to_radians();
            let base_azimuth = azimuth;
            for k in 0..lp.cluster_size.max(1) {
                if out.len() >= MAX_LEAVES {
                    return;
                }
                let fan = if lp.cluster_size > 1 {
                    let step = std::f32::consts::TAU / lp.cluster_size as f32;
                    k as f32 * step
                        + range_f32(&mut rng, -lp.cluster_spread_deg, lp.cluster_spread_deg)
                            .to_radians()
                } else {
                    0.0
                };
                out.push(make_card(
                    lp,
                    &mut rng,
                    at,
                    dir,
                    frame,
                    radius,
                    base_azimuth + fan,
                ));
            }
        }
        until_next -= seg_len - travelled;
    }
}

#[allow(clippy::too_many_arguments)]
fn make_card(
    lp: &LeafParams,
    rng: &mut rand::rngs::SmallRng,
    on_twig: Vec3,
    twig_dir: Vec3,
    frame: Vec3,
    twig_radius: f32,
    azimuth: f32,
) -> Card {
    let u = ortho_unit(frame, twig_dir);
    let v = twig_dir.cross(u);
    let radial = (u * azimuth.cos() + v * azimuth.sin()).normalize_or(u);

    let crotch = (lp.crotch_angle_deg
        + range_f32(rng, -lp.crotch_variance_deg, lp.crotch_variance_deg))
    .to_radians();
    let mut axis = (twig_dir * crotch.cos() + radial * crotch.sin()).normalize_or(twig_dir);

    // Weight lets the blade hang; the droop is applied after the crotch so the angle
    // a species declares is measured off the twig, not off the world.
    let droop = lp.droop_deg.to_radians();
    if droop.abs() > 1e-4 {
        axis = (axis - Vec3::Y * droop.sin()).normalize_or(axis);
    }

    // A blade turns its face to the sky, so the width runs horizontally. Near-vertical
    // leaves have no horizontal reference, so they fall back to the twig frame.
    let flat = axis.cross(Vec3::Y);
    let mut side = if flat.length_squared() > 1e-6 {
        flat.normalize()
    } else {
        ortho_unit(radial, axis)
    };
    let twist = range_f32(rng, -lp.twist_deg, lp.twist_deg).to_radians();
    if twist.abs() > 1e-4 {
        let perp = axis.cross(side);
        side = (side * twist.cos() + perp * twist.sin()).normalize_or(side);
    }

    let scale = 1.0 + range_f32(rng, -lp.size_variance, lp.size_variance);
    Card {
        origin: on_twig + radial * (twig_radius * SURFACE_BIAS),
        axis,
        side,
        length: (lp.card_length * scale).max(1e-3),
        width: (lp.card_width * scale).max(1e-3),
        hue: range_f32(rng, -1.0, 1.0),
    }
}

fn crown_bounds(cards: &[Card]) -> (Vec3, f32) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for c in cards {
        min = min.min(c.origin);
        max = max.max(c.origin);
    }
    let center = (min + max) * 0.5;
    let radius = ((max - min) * 0.5).length().max(1e-3);
    (center, radius)
}

fn emit_card(mesh: &mut LeafMesh, card: &Card, lp: &LeafParams, center: Vec3, radius: f32) {
    let half = card.side * (card.width * 0.5);
    let tip = card.axis * card.length;
    let leaf_center = card.origin + tip * 0.5;

    // The blend that turns a flat plane into part of a soft canopy: lean the shading
    // normal toward the outward direction of the crown.
    let outward = (leaf_center - center).normalize_or(Vec3::Y);
    let mut flat = card.side.cross(card.axis).normalize_or(outward);
    if flat.dot(outward) < 0.0 {
        flat = -flat;
    }
    let blended = flat.lerp(outward, lp.normal_blend.clamp(0.0, 1.0));

    // Leaves buried in the crown see less sky than the ones on the outside.
    let depth = (leaf_center - center).length() / radius;
    let shade = 1.0 - lp.interior_shade.clamp(0.0, 1.0) * (1.0 - depth.clamp(0.0, 1.0));
    let h = card.hue * lp.hue_variance;
    let tint = [
        (lp.tint[0] * (1.0 + h * 0.60)).max(0.0),
        (lp.tint[1] * (1.0 + h * 0.15)).max(0.0),
        (lp.tint[2] * (1.0 - h * 0.50)).max(0.0),
        shade,
    ];

    let base = mesh.positions.len() as u32;
    // v runs 1 at the stem to 0 at the tip: the source art has the leaf tip at the
    // top of the image, which is v = 0 once the rows are uploaded in order.
    for (corner, uv) in [
        (card.origin - half, [0.0, 1.0]),
        (card.origin + half, [1.0, 1.0]),
        (card.origin + tip + half, [1.0, 0.0]),
        (card.origin + tip - half, [0.0, 0.0]),
    ] {
        // Curving the normal across the width keeps a card from lighting as one flat
        // facet and gives each leaf a rounded falloff.
        let bend = card.side * ((uv[0] - 0.5) * 2.0 * lp.curvature);
        mesh.positions.push(corner.to_array());
        mesh.normals
            .push((blended + bend).normalize_or(blended).to_array());
        mesh.uvs.push(uv);
        mesh.tints.push(tint);
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, OAK_RON, PINE_RON};

    fn oak() -> SpeciesParams {
        parse_species(OAK_RON).unwrap()
    }

    #[test]
    fn presets_grow_a_canopy() {
        for src in [OAK_RON, PINE_RON] {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let leaves = build_leaves(&sk, &params);
            assert!(
                leaves.leaf_count() > 500,
                "{}: only {} leaves",
                params.name,
                leaves.leaf_count()
            );
            assert_eq!(leaves.vertex_count(), leaves.leaf_count() * 4);
            assert_eq!(leaves.triangle_count(), leaves.leaf_count() * 2);
            for &i in &leaves.indices {
                assert!((i as usize) < leaves.vertex_count(), "index {i} out of range");
            }
        }
    }

    #[test]
    fn disabled_leaves_produce_nothing() {
        let mut params = oak();
        params.leaves.enabled = false;
        let sk = crate::grow(&params);
        assert!(build_leaves(&sk, &params).is_empty());
    }

    #[test]
    fn leaves_are_deterministic() {
        let params = oak();
        let sk = crate::grow(&params);
        assert_eq!(build_leaves(&sk, &params), build_leaves(&sk, &params));
    }

    #[test]
    fn different_seeds_give_different_canopies() {
        let mut a = oak();
        let mut b = oak();
        a.seed = 3;
        b.seed = 4;
        let ma = build_leaves(&crate::grow(&a), &a);
        let mb = build_leaves(&crate::grow(&b), &b);
        assert_ne!(ma.positions, mb.positions);
    }

    #[test]
    fn normals_are_unit_and_uvs_in_range() {
        let params = oak();
        let sk = crate::grow(&params);
        let leaves = build_leaves(&sk, &params);
        for n in &leaves.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal length {len}");
        }
        for uv in &leaves.uvs {
            assert!(
                (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
                "uv {uv:?} is not card-local"
            );
        }
    }

    #[test]
    fn cards_keep_their_declared_size() {
        // A card is a rectangle; if the axis and side ever stop being perpendicular
        // the quad shears and the leaf texture skews with it.
        let params = oak();
        let sk = crate::grow(&params);
        let leaves = build_leaves(&sk, &params);
        for quad in leaves.positions.chunks_exact(4) {
            let p: Vec<Vec3> = quad.iter().map(|q| Vec3::from(*q)).collect();
            let width = (p[1] - p[0]).length();
            let length = (p[3] - p[0]).length();
            assert!((( p[2] - p[1]).length() - length).abs() < 1e-3, "not a rectangle");
            let across = (p[1] - p[0]).normalize_or_zero();
            let along = (p[3] - p[0]).normalize_or_zero();
            assert!(across.dot(along).abs() < 1e-3, "card is sheared");
            let expected = params.leaves.card_length * (1.0 + params.leaves.size_variance);
            assert!(length <= expected + 1e-3 && width > 0.0);
        }
    }

    #[test]
    fn leaves_only_grow_on_thin_twigs() {
        let params = oak();
        let sk = crate::grow(&params);
        let leaves = build_leaves(&sk, &params);
        let trunk_top = params.trunk.radius;
        for p in &leaves.positions {
            // Nothing should be sitting inside the trunk column near the ground.
            let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
            assert!(
                !(p[1] < 1.0 && r < trunk_top),
                "leaf buried in the trunk at {p:?}"
            );
        }
    }
}
