//! Cutting a closed mesh laid out in charts down to a triangle budget, by collapsing
//! edges, least error first.
//!
//! The error is Garland and Heckbert's: each vertex carries the sum of the planes of the
//! triangles it has stood in, weighted by their area, and a collapse costs how far the
//! vertex it keeps sits from all of them. Each edge collapses onto one of its own ends,
//! so every vertex left is one of the originals and still lies on the surface it was
//! sampled from.
//!
//! A triangle keeps the chart it started in. A vertex inside one chart may go along any
//! of its edges; one on the seam between two charts only along the seam, onto another
//! vertex of it, so both sides of the seam change together and it never opens; and one
//! where three or more charts meet stays put. A collapse is refused where it would fold
//! the surface, turn a triangle over in its chart, pinch the mesh into something that is
//! not a surface, or leave a sliver.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::{DVec3, Vec2, Vec3};

/// The finely sampled mesh a simplifier starts from.
pub(super) struct Source {
    pub positions: Vec<Vec3>,
    /// Where each vertex sits on the cube the surface was sampled through, each axis from
    /// -1 to 1: its place in every chart that holds it.
    pub cube: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub charts: Vec<u8>,
    /// How much each triangle's shape matters, against its area alone.
    pub weights: Vec<f32>,
}

/// A symmetric 4 by 4 matrix, the upper triangle of it: xx xy xz xw yy yz yw zz zw ww.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    /// The squared distance to the plane `n.p + d = 0`, scaled by `w`.
    fn plane(n: DVec3, d: f64, w: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self([a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d].map(|v| v * w))
    }

    fn add(&mut self, o: &Self) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a += b;
        }
    }

    fn sum(&self, o: &Self) -> Self {
        let mut q = *self;
        q.add(o);
        q
    }

    fn at(&self, p: DVec3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (p.x, p.y, p.z);
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x
            + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y
            + q[7] * z * z + 2.0 * q[8] * z
            + q[9]
    }
}

/// A collapse waiting its turn: `from` onto `to`, and the versions both ends were at
/// when it was costed, so one costed before either end changed is known to be stale.
struct Candidate {
    cost: f64,
    from: u32,
    to: u32,
    from_version: u32,
    to_version: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Cheapest first out of a max-heap; ties broken by vertex, so the order never
    /// depends on how the heap happened to be laid out.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.from.cmp(&self.from))
            .then_with(|| other.to.cmp(&self.to))
    }
}

/// Worst a triangle may be left, as 1 for equilateral down to 0 for flat, unless it
/// was already no better.
const MIN_QUALITY: f32 = 0.12;
/// Least cosine between a triangle's facing before and after a collapse.
const MIN_TURN: f32 = 0.3;

pub(super) struct Simplifier {
    positions: Vec<Vec3>,
    cube: Vec<Vec3>,
    /// Each chart's u and v across the cube.
    axes: [(Vec3, Vec3); 6],
    triangles: Vec<[u32; 3]>,
    charts: Vec<u8>,
    alive: Vec<bool>,
    removed: Vec<bool>,
    /// The live triangles round each vertex.
    around: Vec<Vec<u32>>,
    quadrics: Vec<Quadric>,
    versions: Vec<u32>,
    live: usize,
    heap: BinaryHeap<Candidate>,
}

impl Simplifier {
    pub fn new(src: Source, axes: [(Vec3, Vec3); 6]) -> Self {
        let n = src.positions.len();
        let mut around = vec![Vec::new(); n];
        let mut quadrics = vec![Quadric::default(); n];
        for (t, tri) in src.triangles.iter().enumerate() {
            let p = tri.map(|v| src.positions[v as usize].as_dvec3());
            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
            let area = cross.length() * 0.5;
            if area > 0.0 {
                let normal = cross / (2.0 * area);
                let q = Quadric::plane(normal, -normal.dot(p[0]), area * f64::from(src.weights[t]));
                for &v in tri {
                    quadrics[v as usize].add(&q);
                }
            }
            for &v in tri {
                around[v as usize].push(t as u32);
            }
        }
        let live = src.triangles.len();
        let mut s = Self {
            positions: src.positions,
            cube: src.cube,
            axes,
            alive: vec![true; live],
            triangles: src.triangles,
            charts: src.charts,
            removed: vec![false; n],
            around,
            quadrics,
            versions: vec![0; n],
            live,
            heap: BinaryHeap::new(),
        };
        s.refill();
        s
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn position(&self, v: u32) -> Vec3 {
        self.positions[v as usize]
    }

    /// Where vertex `v` sits in chart `chart`, each axis from -1 to 1.
    pub fn chart_st(&self, v: u32, chart: u8) -> Vec2 {
        let (u, w) = self.axes[chart as usize];
        let c = self.cube[v as usize];
        Vec2::new(c.dot(u), c.dot(w))
    }

    /// The live triangles and the chart each is in, in a fixed order.
    pub fn triangles(&self) -> impl Iterator<Item = ([u32; 3], u8)> + '_ {
        self.triangles
            .iter()
            .zip(&self.charts)
            .zip(&self.alive)
            .filter(|(_, alive)| **alive)
            .map(|((t, c), _)| (*t, *c))
    }

    /// Collapses edges until no more than `target` triangles are left, or until nothing
    /// more can be collapsed without breaking the mesh.
    pub fn simplify_to(&mut self, target: usize) {
        let mut last_refill = usize::MAX;
        while self.live > target {
            match self.heap.pop() {
                Some(c) => {
                    let (a, b) = (c.from as usize, c.to as usize);
                    if self.removed[a]
                        || self.removed[b]
                        || self.versions[a] != c.from_version
                        || self.versions[b] != c.to_version
                    {
                        continue;
                    }
                    // Something round the edge may have changed since it was costed.
                    if self.cost(c.from, c.to).is_some() {
                        self.collapse(c.from, c.to);
                    }
                }
                None => {
                    // Edges refused earlier may be allowed now that their neighbours have
                    // moved; once a refill brings nothing new, there is nothing left.
                    if last_refill == self.live {
                        break;
                    }
                    last_refill = self.live;
                    self.refill();
                }
            }
        }
    }

    fn refill(&mut self) {
        self.heap.clear();
        for t in 0..self.triangles.len() {
            if !self.alive[t] {
                continue;
            }
            let tri = self.triangles[t];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                // Each edge is in two triangles, once each way round.
                if a < b {
                    self.push_edge(a, b);
                }
            }
        }
    }

    /// Costs an edge both ways and queues the cheaper way that is allowed.
    fn push_edge(&mut self, a: u32, b: u32) {
        let best = match (self.cost(a, b), self.cost(b, a)) {
            (Some(x), Some(y)) if y < x => Some((y, b, a)),
            (Some(x), _) => Some((x, a, b)),
            (None, Some(y)) => Some((y, b, a)),
            (None, None) => None,
        };
        if let Some((cost, from, to)) = best {
            self.heap.push(Candidate {
                cost,
                from,
                to,
                from_version: self.versions[from as usize],
                to_version: self.versions[to as usize],
            });
        }
    }

    fn chart_mask(&self, v: u32) -> u8 {
        self.around[v as usize].iter().fold(0u8, |m, &t| m | 1 << self.charts[t as usize])
    }

    fn neighbours(&self, v: u32, out: &mut Vec<u32>) {
        out.clear();
        for &t in &self.around[v as usize] {
            for w in self.triangles[t as usize] {
                if w != v && !out.contains(&w) {
                    out.push(w);
                }
            }
        }
    }

    /// The error of collapsing `a` onto `b`, or `None` where that is not allowed.
    fn cost(&self, a: u32, b: u32) -> Option<f64> {
        let (ai, bi) = (a as usize, b as usize);
        if self.removed[ai] || self.removed[bi] {
            return None;
        }
        let charts_a = self.chart_mask(a);
        let kinds = charts_a.count_ones();
        if kinds >= 3 {
            return None;
        }
        // The two triangles on the edge.
        let mut edge = [0u32; 2];
        let mut found = 0;
        for &t in &self.around[ai] {
            if self.triangles[t as usize].contains(&b) {
                if found == 2 {
                    return None;
                }
                edge[found] = t;
                found += 1;
            }
        }
        if found != 2 {
            return None;
        }
        if kinds == 2 {
            // Along the seam only: the edge has a chart either side, and `b` is in both.
            if self.charts[edge[0] as usize] == self.charts[edge[1] as usize] {
                return None;
            }
            if self.chart_mask(b) & charts_a != charts_a {
                return None;
            }
        }
        // The only vertices both ends share are the two across the edge, or the collapse
        // would pinch the surface.
        let apex = |t: u32| self.triangles[t as usize].into_iter().find(|&v| v != a && v != b).unwrap_or(a);
        let (x, y) = (apex(edge[0]), apex(edge[1]));
        if x == y {
            return None;
        }
        let mut na = Vec::with_capacity(12);
        let mut nb = Vec::with_capacity(12);
        self.neighbours(a, &mut na);
        self.neighbours(b, &mut nb);
        if na.iter().any(|v| *v != b && *v != x && *v != y && nb.contains(v)) {
            return None;
        }

        let pb = self.positions[bi];
        for &t in &self.around[ai] {
            if t == edge[0] || t == edge[1] {
                continue;
            }
            let tri = self.triangles[t as usize];
            let before = tri.map(|v| self.positions[v as usize]);
            let after = tri.map(|v| if v == a { pb } else { self.positions[v as usize] });
            let n0 = (before[1] - before[0]).cross(before[2] - before[0]);
            let n1 = (after[1] - after[0]).cross(after[2] - after[0]);
            let l1 = n1.length();
            if l1 < 1e-12 || n0.dot(n1) < MIN_TURN * n0.length() * l1 {
                return None;
            }
            let chart = self.charts[t as usize];
            let uv0 = tri.map(|v| self.chart_st(v, chart));
            let uv1 = tri.map(|v| self.chart_st(if v == a { b } else { v }, chart));
            let (s0, s1) = (signed_area(uv0), signed_area(uv1));
            if s0 * s1 <= 0.0 || s1.abs() < 1e-9 {
                return None;
            }
            let q1 = quality(after);
            if q1 < MIN_QUALITY && q1 < quality(before) {
                return None;
            }
        }
        Some(self.quadrics[ai].sum(&self.quadrics[bi]).at(pb.as_dvec3()))
    }

    fn collapse(&mut self, a: u32, b: u32) {
        let (ai, bi) = (a as usize, b as usize);
        for t in std::mem::take(&mut self.around[ai]) {
            let tri = self.triangles[t as usize];
            if tri.contains(&b) {
                self.alive[t as usize] = false;
                self.live -= 1;
                for v in tri {
                    if v != a {
                        self.around[v as usize].retain(|&x| x != t);
                    }
                }
            } else {
                for v in &mut self.triangles[t as usize] {
                    if *v == a {
                        *v = b;
                    }
                }
                self.around[bi].push(t);
            }
        }
        self.removed[ai] = true;
        let qa = self.quadrics[ai];
        self.quadrics[bi].add(&qa);
        self.versions[bi] += 1;
        let mut ring = Vec::new();
        self.neighbours(b, &mut ring);
        for n in ring {
            self.push_edge(n, b);
        }
    }
}

fn signed_area(uv: [Vec2; 3]) -> f32 {
    (uv[1] - uv[0]).perp_dot(uv[2] - uv[0])
}

/// 1 for an equilateral triangle, falling to 0 as it flattens.
fn quality(p: [Vec3; 3]) -> f32 {
    let area2 = (p[1] - p[0]).cross(p[2] - p[0]).length();
    let sides = (p[1] - p[0]).length_squared() + (p[2] - p[1]).length_squared() + (p[0] - p[2]).length_squared();
    if sides <= 0.0 { 0.0 } else { 2.0 * 3f32.sqrt() * area2 / sides }
}
