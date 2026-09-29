//! A bounding-volume hierarchy over triangles, for casting rays at a model.

use glam::Vec3;

/// A triangle's corners.
pub type Tri = [Vec3; 3];

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

struct Node {
    min: Vec3,
    max: Vec3,
    /// A leaf: the first entry of `order`. An inner node: its right child; the left one
    /// follows the node itself.
    first_or_right: u32,
    /// Triangles in a leaf, 0 for an inner node.
    count: u32,
}

const LEAF_SIZE: usize = 4;

pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
}

impl Bvh {
    pub fn build(tris: &[Tri]) -> Self {
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let centroids: Vec<Vec3> = tris.iter().map(|t| (t[0] + t[1] + t[2]) / 3.0).collect();
        let mut nodes = Vec::with_capacity(tris.len() / 2 + 1);
        if !tris.is_empty() {
            build_node(tris, &centroids, &mut order, 0, tris.len(), &mut nodes);
        }
        Self { nodes, order }
    }

    /// The distance to the nearest hit nearer than `tmax` that `accept` lets through, skipping triangle
    /// `ignore` (the one a ray starts on). `accept` gets the triangle and the barycentric
    /// weights of the hit, so a cut-out can look at its texture.
    pub fn closest(
        &self,
        tris: &[Tri],
        ray: &Ray,
        tmax: f32,
        ignore: u32,
        accept: &mut impl FnMut(u32, f32, f32) -> bool,
    ) -> Option<f32> {
        self.traverse(tris, ray, tmax, ignore, accept, false)
    }

    /// Whether anything accepted is in the way within `tmax`.
    pub fn any(
        &self,
        tris: &[Tri],
        ray: &Ray,
        tmax: f32,
        ignore: u32,
        accept: &mut impl FnMut(u32, f32, f32) -> bool,
    ) -> bool {
        self.traverse(tris, ray, tmax, ignore, accept, true).is_some()
    }

    fn traverse(
        &self,
        tris: &[Tri],
        ray: &Ray,
        mut tmax: f32,
        ignore: u32,
        accept: &mut impl FnMut(u32, f32, f32) -> bool,
        stop_at_first: bool,
    ) -> Option<f32> {
        if self.nodes.is_empty() {
            return None;
        }
        // A direction with a zero component would make an infinite reciprocal, and 0 times
        // infinity is NaN when the origin sits exactly on a box's face.
        let safe = |d: f32| if d.abs() < 1e-30 { 1e-30f32.copysign(d) } else { d };
        let inv = Vec3::new(1.0 / safe(ray.dir.x), 1.0 / safe(ray.dir.y), 1.0 / safe(ray.dir.z));
        let mut best = None;
        let mut stack = [0u32; 64];
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let index = stack[top] as usize;
            let node = &self.nodes[index];
            if !slab(node, ray.origin, inv, tmax) {
                continue;
            }
            if node.count == 0 {
                // Both children; the left is popped first.
                stack[top] = node.first_or_right;
                stack[top + 1] = index as u32 + 1;
                top += 2;
                continue;
            }
            for k in 0..node.count {
                let tri = self.order[(node.first_or_right + k) as usize];
                if tri == ignore {
                    continue;
                }
                if let Some((t, u, v)) = intersect(&tris[tri as usize], ray, tmax)
                    && accept(tri, u, v)
                {
                    tmax = t;
                    best = Some(t);
                    if stop_at_first {
                        return best;
                    }
                }
            }
        }
        best
    }
}

fn build_node(tris: &[Tri], centroids: &[Vec3], order: &mut [u32], lo: usize, hi: usize, nodes: &mut Vec<Node>) {
    let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let (mut cmin, mut cmax) = (min, max);
    for &i in &order[lo..hi] {
        for p in tris[i as usize] {
            min = min.min(p);
            max = max.max(p);
        }
        cmin = cmin.min(centroids[i as usize]);
        cmax = cmax.max(centroids[i as usize]);
    }
    let index = nodes.len();
    nodes.push(Node { min, max, first_or_right: lo as u32, count: (hi - lo) as u32 });
    let extent = cmax - cmin;
    if hi - lo <= LEAF_SIZE || extent.max_element() <= 0.0 {
        return;
    }
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let mid = (lo + hi) / 2;
    order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
        centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
    });
    build_node(tris, centroids, order, lo, mid, nodes);
    let right = nodes.len() as u32;
    build_node(tris, centroids, order, mid, hi, nodes);
    nodes[index].first_or_right = right;
    nodes[index].count = 0;
}

fn slab(node: &Node, origin: Vec3, inv: Vec3, tmax: f32) -> bool {
    let t1 = (node.min - origin) * inv;
    let t2 = (node.max - origin) * inv;
    let (near, far) = (t1.min(t2), t1.max(t2));
    let enter = near.x.max(near.y).max(near.z).max(0.0);
    let exit = far.x.min(far.y).min(far.z).min(tmax);
    enter <= exit
}

/// Möller-Trumbore, both faces.
fn intersect(tri: &Tri, ray: &Ray, tmax: f32) -> Option<(f32, f32, f32)> {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let to = ray.origin - tri[0];
    let u = to.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = to.cross(e1);
    let v = ray.dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 1e-7 && t < tmax).then_some((t, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_at_height(y: f32) -> Vec<Tri> {
        let p = |x: f32, z: f32| Vec3::new(x, y, z);
        vec![[p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0)], [p(-1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0)]]
    }

    fn down(x: f32, y: f32) -> Ray {
        Ray { origin: Vec3::new(x, y, 0.1), dir: Vec3::NEG_Y }
    }

    fn yes(_: u32, _: f32, _: f32) -> bool {
        true
    }

    #[test]
    fn a_ray_finds_the_nearest_of_two_planes() {
        let mut tris = quad_at_height(0.0);
        tris.extend(quad_at_height(1.0));
        let bvh = Bvh::build(&tris);
        let t = bvh.closest(&tris, &down(0.2, 5.0), 100.0, u32::MAX, &mut yes).unwrap();
        assert!((t - 4.0).abs() < 1e-5, "{t}");
    }

    #[test]
    fn rays_respect_their_range_and_can_miss() {
        let tris = quad_at_height(0.0);
        let bvh = Bvh::build(&tris);
        assert!(bvh.closest(&tris, &down(0.0, 5.0), 3.0, u32::MAX, &mut yes).is_none());
        assert!(bvh.closest(&tris, &down(3.0, 5.0), 100.0, u32::MAX, &mut yes).is_none());
        assert!(bvh.any(&tris, &down(0.0, 5.0), 6.0, u32::MAX, &mut yes));
    }

    #[test]
    fn the_start_triangle_and_rejected_hits_are_skipped() {
        let mut tris = quad_at_height(0.0);
        tris.extend(quad_at_height(-1.0));
        let bvh = Bvh::build(&tris);
        // Triangle 0 is the one over (0.2, 0.1). Ignoring it, the ray reaches the plane below.
        let skip = bvh.closest(&tris, &down(0.2, 5.0), 100.0, 0, &mut yes);
        assert!((skip.unwrap() - 6.0).abs() < 1e-5);
        // Refusing everything at the upper plane does the same.
        let refuse = bvh.closest(&tris, &down(0.2, 5.0), 100.0, u32::MAX, &mut |tri, _, _| tri >= 2);
        assert!((refuse.unwrap() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn many_triangles_agree_with_brute_force() {
        // A ripple of little triangles, so the tree has depth.
        let mut tris = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                let (x, z) = (i as f32 * 0.1, j as f32 * 0.1);
                let h = ((x * 3.0).sin() + (z * 2.0).cos()) * 0.2;
                tris.push([Vec3::new(x, h, z), Vec3::new(x + 0.1, h, z), Vec3::new(x, h, z + 0.1)]);
            }
        }
        let bvh = Bvh::build(&tris);
        for k in 0..200 {
            let f = k as f32;
            let ray = Ray {
                origin: Vec3::new((f * 0.37) % 4.0, 2.0, (f * 0.91) % 4.0),
                dir: Vec3::new((f * 0.13).sin() * 0.2, -1.0, (f * 0.29).cos() * 0.2).normalize(),
            };
            let fast = bvh.closest(&tris, &ray, 100.0, u32::MAX, &mut yes);
            let slow = tris.iter().filter_map(|t| intersect(t, &ray, 100.0).map(|h| h.0)).fold(None, |a: Option<f32>, t| {
                Some(a.map_or(t, |a| a.min(t)))
            });
            assert_eq!(fast.is_some(), slow.is_some(), "ray {k}");
            if let (Some(a), Some(b)) = (fast, slow) {
                assert!((a - b).abs() < 1e-5, "ray {k}: {a} vs {b}");
            }
        }
    }
}
