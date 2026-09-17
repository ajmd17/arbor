use glam::Vec3;
use std::f32::consts::{PI, TAU};

use crate::math::{norm_or_zero, ortho_of, ortho_unit, transport};
use crate::skeleton::Skeleton;
use crate::species::{BarkIrregularity, MeshParams, SpeciesParams};

/// Angular step used for the finite-difference normal around a ring.
const NORMAL_DA: f32 = 0.01;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &self.positions {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        (min, max)
    }
}

fn seed_phases(seed: u64) -> (f32, f32) {
    let mut z = seed ^ 0x9E3779B97F4A7C15;
    let step = |z: &mut u64| {
        *z = z.wrapping_add(0x9E3779B97F4A7C15);
        let mut x = *z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    };
    let a = step(&mut z);
    let b = step(&mut z);
    (
        (a as f32 / u64::MAX as f32) * TAU,
        (b as f32 / u64::MAX as f32) * TAU,
    )
}


/// One value in 0..1 from a seed and an index, so every stem gets its own bark and the
/// same tree comes back the same every time.
fn hash01(seed: u64, i: u64) -> f32 {
    let mut z = seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f32 / (1u64 << 53) as f32
}

fn wrap_pi(a: f32) -> f32 {
    let mut a = a % TAU;
    if a > PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Catmull-Rom through four samples. Used to put extra rings on a stem without
/// creasing the centreline at every original node.
fn catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    ((p1 * 2.0) + (p2 - p0) * t + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
        * 0.5
}

fn catmull_f32(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    catmull(Vec3::X * p0, Vec3::X * p1, Vec3::X * p2, Vec3::X * p3, t).x
}


/// Extra rings along a smooth curve through the ones a stem already has.
fn subdivide(points: &[Vec3], radii: &[f32], per_meter: f32) -> (Vec<Vec3>, Vec<f32>) {
    let n = points.len();
    if n < 2 {
        return (points.to_vec(), radii.to_vec());
    }
    let mut out_p = Vec::with_capacity(n * 4);
    let mut out_r = Vec::with_capacity(n * 4);
    for i in 0..n - 1 {
        let (a, b) = (i.saturating_sub(1), (i + 2).min(n - 1));
        let steps = (((points[i + 1] - points[i]).length() * per_meter).ceil() as usize).max(1);
        for k in 0..steps {
            let t = k as f32 / steps as f32;
            out_p.push(catmull(points[a], points[i], points[i + 1], points[b], t));
            out_r.push(catmull_f32(radii[a], radii[i], radii[i + 1], radii[b], t).max(1e-4));
        }
    }
    out_p.push(points[n - 1]);
    out_r.push(radii[n - 1]);
    (out_p, out_r)
}

/// A swelling where a branch leaves its parent.
struct Collar {
    /// Distance along the parent at which the child attaches.
    arc: f32,
    /// Which way the child heads, across the parent's axis.
    dir: Vec3,
    radius: f32,
}

/// A local lump on the bole.
struct Burl {
    arc: f32,
    angle: f32,
    sigma_s: f32,
    sigma_a: f32,
    amp: f32,
}

/// Everything that stops a stem being a cylinder, sampled as a smooth function of the
/// angle around it and the distance along it.
///
/// Smooth is the requirement, not a nicety: the mesher reads its normals off finite
/// differences of the radius, so a shape with a continuous derivative shades correctly
/// with no extra work, and one without it facets.
struct Irregular {
    params: BarkIrregularity,
    /// Integer frequencies, so a cross-section closes on itself, with a phase and a
    /// twist each. Mixed rather than single so the bole is not a tidy cog.
    flutes: [(f32, f32, f32); 3],
    /// Angular frequency, phase and weight of each swelling along the length.
    swells: [(f32, f32, f32); 3],
    burls: Vec<Burl>,
    collars: Vec<Collar>,
    active: bool,
}

impl Irregular {
    fn none() -> Self {
        Self {
            params: BarkIrregularity::default(),
            flutes: [(0.0, 0.0, 0.0); 3],
            swells: [(0.0, 0.0, 0.0); 3],
            burls: Vec::new(),
            collars: Vec::new(),
            active: false,
        }
    }

    fn new(p: &BarkIrregularity, seed: u64, stem: u64, length: f32, radius: f32) -> Self {
        if radius < p.min_radius || length < 1e-3 {
            return Self::none();
        }
        let r = |i: u64| hash01(seed, stem.wrapping_mul(977).wrapping_add(i));

        let base = p.flute_waves.max(1.0).round();
        let flutes = [
            (base, r(1) * TAU, (r(2) - 0.5) * p.flute_twist * TAU),
            ((base * 1.75).round().max(2.0), r(3) * TAU, (r(4) - 0.5) * p.flute_twist * TAU),
            ((base * 0.5).round().max(1.0), r(5) * TAU, (r(6) - 0.5) * p.flute_twist * TAU),
        ];
        let period = p.swell_period.max(0.05);
        let swells = [
            (TAU / period, r(7) * TAU, 0.6),
            (TAU / (period * 0.45), r(8) * TAU, 0.3),
            (TAU / (period * 0.2), r(9) * TAU, 0.1),
        ];

        let count = (length * p.burl_density).round().max(0.0) as usize;
        let mut burls = Vec::with_capacity(count);
        for k in 0..count {
            let i = 20 + k as u64 * 7;
            let size = p.burl_size * (0.55 + 0.9 * r(i + 3));
            burls.push(Burl {
                arc: r(i) * length,
                angle: r(i + 1) * TAU,
                sigma_s: size.max(0.02),
                // Wider round a thin stem than a thick one, for the same lump.
                sigma_a: (size / radius.max(1e-3)).clamp(0.25, 1.6),
                amp: p.burl_depth * (0.5 + r(i + 2)),
            });
        }

        Self {
            params: p.clone(),
            flutes,
            swells,
            burls,
            collars: Vec::new(),
            active: true,
        }
    }

    /// Multiplier on the radius at distance `s` along the stem, at angle `a` whose
    /// outward direction is `e_r`.
    fn shape(&self, s: f32, a: f32, e_r: Vec3, radius: f32) -> f32 {
        if !self.active {
            return 1.0;
        }
        let p = &self.params;
        // Faded in off the socket, so a branch still starts inside its parent and the
        // junction stays welded.
        let fade = smoothstep(0.0, radius * 4.0, s);

        let mut flute = 0.0;
        for (f, phase, twist) in self.flutes {
            flute += (a * f + phase + s * twist).sin() / f.max(1.0).sqrt();
        }
        let mut swell = 0.0;
        for (w, phase, weight) in self.swells {
            swell += (s * w + phase).sin() * weight;
        }
        let mut lumps = 0.0;
        for b in &self.burls {
            let ds = (s - b.arc) / b.sigma_s;
            let da = wrap_pi(a - b.angle) / b.sigma_a;
            lumps += b.amp * (-(ds * ds + da * da)).exp();
        }
        for c in &self.collars {
            let reach = (c.radius * 2.6).max(0.04);
            let ds = (s - c.arc) / reach;
            // Only on the side the branch leaves from.
            let facing = e_r.dot(c.dir).max(0.0);
            let share = (c.radius / radius.max(1e-4)).min(1.0);
            lumps += p.collar_depth * share * (-(ds * ds)).exp() * facing * facing * facing;
        }

        1.0 + fade * (p.flute_depth * flute + p.swell_depth * swell + lumps)
    }
}

struct MeshSink {
    mesh: Mesh,
}

impl MeshSink {
    fn push_vertex(&mut self, pos: Vec3, normal: Vec3, tangent: Vec3, uv: [f32; 2]) {
        self.mesh.positions.push(pos.to_array());
        self.mesh.normals.push(normal.to_array());
        let t = norm_or_zero(tangent - normal * tangent.dot(normal));
        self.mesh.tangents.push([t.x, t.y, t.z, 1.0]);
        self.mesh.uvs.push(uv);
    }

    fn vertex_offset(&self) -> u32 {
        self.mesh.positions.len() as u32
    }
}


fn dirs_for(points: &[Vec3]) -> Vec<Vec3> {
    let last = points.len() - 1;
    let mut dirs: Vec<Vec3> = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let prev = if i == 0 {
            points[0] * 2.0 - points[1]
        } else {
            points[i - 1]
        };
        let next = if i < last {
            points[i + 1]
        } else {
            points[i] * 2.0 - prev
        };
        let d = norm_or_zero(next - prev);
        dirs.push(if d == Vec3::ZERO {
            *dirs.last().unwrap_or(&Vec3::Y)
        } else {
            d
        });
    }
    dirs
}

fn frames_for(dirs: &[Vec3]) -> Vec<(Vec3, Vec3)> {
    let mut frames = Vec::with_capacity(dirs.len());
    let mut n = ortho_of(dirs[0]);
    for i in 0..dirs.len() {
        if i > 0 {
            n = transport(dirs[i - 1], dirs[i], n);
        }
        n = ortho_unit(n, dirs[i]);
        frames.push((n, dirs[i].cross(n)));
    }
    frames
}

fn arcs_for(points: &[Vec3]) -> Vec<f32> {
    let mut arc = Vec::with_capacity(points.len());
    let mut travelled = 0.0;
    for i in 0..points.len() {
        if i > 0 {
            travelled += (points[i] - points[i - 1]).length();
        }
        arc.push(travelled);
    }
    arc
}

/// How far the surface moves if ring `i` is dropped and its neighbours joined directly.
///
/// Measured on the surface the mesher would actually emit, not on the centreline: a
/// ring sitting on a burl matters even where the centreline through it is straight.
fn ring_error(path: &StemPath, mp: &MeshParams, phase: (f32, f32), prev: usize, i: usize, next: usize) -> f32 {
    const ANGLES: usize = 8;
    let (a, b, c) = (path.points[prev], path.points[i], path.points[next]);
    let span = c - a;
    let t = if span.length_squared() > 1e-12 {
        ((b - a).dot(span) / span.length_squared()).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let mut worst = (a + span * t - b).length();
    for k in 0..ANGLES {
        let ang = k as f32 / ANGLES as f32 * TAU;
        let here = path.radius_at(i, ang, mp, phase);
        let guess = path.radius_at(prev, ang, mp, phase) * (1.0 - t)
            + path.radius_at(next, ang, mp, phase) * t;
        worst = worst.max((here - guess).abs());
    }
    worst
}

/// Which rings to keep. Walks the stem dropping any ring whose absence moves the
/// surface less than the tolerance, never two in a row, so the result still follows
/// every feature the bark has while spending nothing on the stretches between them.
fn keep_rings(path: &StemPath, mp: &MeshParams, phase: (f32, f32)) -> Vec<usize> {
    let tol = mp.irregularity.ring_tolerance;
    if tol <= 0.0 || path.len() < 3 {
        return (0..path.len()).collect();
    }
    let mut keep = Vec::with_capacity(path.len());
    keep.push(0);
    let mut i = 1;
    while i < path.len() - 1 {
        let prev = *keep.last().unwrap();
        let limit = (path.radii[i] * tol).max(3e-4);
        if ring_error(path, mp, phase, prev, i, i + 1) < limit {
            // Dropped. The next ring is measured against the same neighbour, so a long
            // smooth run collapses rather than losing every other ring.
            i += 1;
        } else {
            keep.push(i);
            i += 1;
        }
    }
    keep.push(path.len() - 1);
    keep
}

/// One stem laid out as a chain of ring centres, ready to be swept.
struct StemPath {
    /// Ring centres. For every stem but the trunk, the first entry sits on the parent
    /// centreline so the tube starts inside the parent instead of floating beside it.
    points: Vec<Vec3>,
    /// Axis direction at each ring, from a centred difference.
    dirs: Vec<Vec3>,
    /// Skeleton radius at each ring, before socket and root flare.
    radii: Vec<f32>,
    /// Extra widening at the foot of a branch so the junction reads as a socket.
    socket: Vec<f32>,
    /// Distance travelled along the stem, used for the V coordinate.
    arc: Vec<f32>,
    /// True for the stem that starts at the root node, which gets the buttress flare
    /// and the ground cap.
    is_trunk: bool,
    /// Parallel-transported frame at each ring. Built once here rather than in the
    /// sweep, because the bark needs to know which way a point on the surface faces
    /// before it can put a branch collar on the right side of the trunk.
    frames: Vec<(Vec3, Vec3)>,
    /// What keeps this stem from being a cylinder.
    irregular: Irregular,
}

impl StemPath {
    /// A branch that only ever got one segment: an anchor ring and a single node.
    /// Most of a tree is these, so what they cost decides what the mesh costs.
    fn is_single_segment(&self) -> bool {
        !self.is_trunk && self.points.len() == 2
    }

    fn build(
        sk: &Skeleton,
        stem: &[u32],
        mp: &MeshParams,
        seed: u64,
        children: &[Vec<u32>],
    ) -> Option<StemPath> {
        let first = *stem.first()? as usize;
        let is_trunk = sk.nodes[first].parent.is_none();

        let mut points: Vec<Vec3> = Vec::with_capacity(stem.len() + 1);
        let mut radii: Vec<f32> = Vec::with_capacity(stem.len() + 1);
        // How wide the socket may get before it stops being swallowed by the parent.
        let mut socket_ceiling = f32::INFINITY;
        if let Some(p) = sk.nodes[first].parent {
            // Anchor the stem on its parent so the two tubes overlap at the junction.
            let anchor = sk.nodes[p as usize].position;
            if (anchor - sk.nodes[first].position).length() > 1e-5 {
                points.push(anchor);
                radii.push(sk.nodes[first].radius);
            }
            socket_ceiling = sk.nodes[p as usize].radius;
        }
        for &i in stem {
            let node = &sk.nodes[i as usize];
            // Skip duplicate positions: a zero-length segment gives no usable axis.
            if points
                .last()
                .is_some_and(|p: &Vec3| (*p - node.position).length() < 1e-5)
            {
                continue;
            }
            points.push(node.position);
            radii.push(node.radius);
        }
        if points.len() < 2 {
            return None;
        }

        // A skeleton is segmented for growing, which is far coarser than the bark needs:
        // a half-metre between rings cannot describe a burl. Stems thick enough to show
        // the detail get extra rings along a smooth curve through the ones they have.
        let r_max = radii.iter().copied().fold(0.0f32, f32::max);
        let ir = &mp.irregularity;
        if r_max >= ir.min_radius && ir.rings_per_meter > 0.0 {
            let (p, r) = subdivide(&points, &radii, ir.rings_per_meter);
            points = p;
            radii = r;
        }

        let dirs = dirs_for(&points);
        let arc = arcs_for(&points);

        // The socket fades out over a few base radii, so the flare is sized by the
        // branch rather than by however finely the stem happens to be segmented.
        let socket_len = (radii[0] * 3.0).max(1e-4);
        // Cap the flare at whatever the parent can hide. Where both stems are already
        // at the radius floor there is nothing to hide it in, and an unclamped flare
        // would leave a bud sticking out of the junction.
        let max_scale = (socket_ceiling / radii[0].max(1e-5)).max(1.0);
        let socket = arc
            .iter()
            .map(|&s| {
                if is_trunk {
                    1.0
                } else {
                    let t = (s / socket_len).clamp(0.0, 1.0);
                    (1.0 + mp.socket_flare * (1.0 - t) * (1.0 - t)).min(max_scale)
                }
            })
            .collect();

        // Transport runs here rather than in the sweep, so the rings, the ground cap
        // and the bark all read one frame instead of three.
        let frames = frames_for(&dirs);

        let mut irregular = Irregular::new(
            &mp.irregularity,
            seed,
            sk.nodes[first].stem as u64,
            *arc.last().unwrap_or(&0.0),
            radii.iter().copied().fold(0.0f32, f32::max),
        );
        // A trunk thickens into every limb it carries. Collect where the children leave
        // so the bole can swell to meet them instead of meeting them at a seam.
        if irregular.active && mp.irregularity.collar_depth > 0.0 {
            for &node in stem {
                let here = sk.nodes[node as usize].position;
                for &child in children.get(node as usize).into_iter().flatten() {
                    let c = &sk.nodes[child as usize];
                    let mut best = (f32::MAX, 0usize);
                    for (j, p) in points.iter().enumerate() {
                        let d = (*p - here).length_squared();
                        if d < best.0 {
                            best = (d, j);
                        }
                    }
                    let j = best.1;
                    let away = c.position - here;
                    let across = away - dirs[j] * away.dot(dirs[j]);
                    let Some(dir) = across.try_normalize() else {
                        continue;
                    };
                    irregular.collars.push(Collar {
                        arc: arc[j],
                        dir,
                        radius: c.radius,
                    });
                }
            }
        }

        let dense = StemPath {
            points,
            dirs,
            radii,
            socket,
            arc,
            is_trunk,
            frames,
            irregular,
        };
        Some(dense.thinned(mp, seed_phases(seed)))
    }

    /// The same stem with every ring that was not earning its place removed.
    ///
    /// Rings are laid down densely because a burl needs them, then thinned against
    /// what dropping one would actually do to the surface. A smooth stretch of bole
    /// ends up costing what a smooth stretch should.
    fn thinned(self, mp: &MeshParams, phase: (f32, f32)) -> StemPath {
        let keep = keep_rings(&self, mp, phase);
        if keep.len() == self.len() {
            return self;
        }
        let points: Vec<Vec3> = keep.iter().map(|&i| self.points[i]).collect();
        let radii: Vec<f32> = keep.iter().map(|&i| self.radii[i]).collect();
        let socket: Vec<f32> = keep.iter().map(|&i| self.socket[i]).collect();
        // Arc comes from the dense polyline, so thinning cannot slide the bark along
        // the stem and move the burls.
        let arc: Vec<f32> = keep.iter().map(|&i| self.arc[i]).collect();
        let dirs = dirs_for(&points);
        let frames = frames_for(&dirs);
        StemPath {
            points,
            dirs,
            radii,
            socket,
            arc,
            is_trunk: self.is_trunk,
            frames,
            irregular: self.irregular,
        }
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    /// Surface radius of ring `i` at angle `a`, including socket, root flare and the
    /// flutes, swellings, burls and branch collars that make it bark rather than pipe.
    fn radius_at(&self, i: usize, a: f32, mp: &MeshParams, phase: (f32, f32)) -> f32 {
        let base = self.radii[i] * self.socket[i];
        let (n, b) = self.frames[i];
        let e_r = n * a.cos() + b * a.sin();
        let shaped = base * self.irregular.shape(self.arc[i], a, e_r, base);

        let y = self.points[i].y;
        if self.is_trunk && y < mp.flare_height && mp.flare_height > 1e-4 {
            let t = (y / mp.flare_height).clamp(0.0, 1.0);
            let k = (1.0 - t) * (1.0 - t);
            let lobe = 0.7 + 0.3 * (a * 3.0 + phase.0).sin() + 0.2 * (a * 7.0 + phase.1).sin();
            shaped * (1.0 + mp.root_flare * k * lobe.max(0.2))
        } else {
            shaped
        }
    }
}

pub fn build_mesh(sk: &Skeleton, params: &SpeciesParams) -> Mesh {
    let mp: &MeshParams = &params.mesh;
    let mut sink = MeshSink {
        mesh: Mesh::default(),
    };
    let phase = seed_phases(params.seed);

    let children = child_index(sk);

    for stem in sk.stem_runs() {
        let Some(path) = StemPath::build(sk, &stem, mp, params.seed, &children) else {
            continue;
        };
        if path.radii.iter().copied().fold(0.0f32, f32::max) < mp.min_bark_radius {
            continue;
        }
        emit_stem(&mut sink, &path, mp, phase);
    }

    sink.mesh
}

/// One cone: the anchor ring of a stem drawn straight to its tip.
fn emit_spike(
    sink: &mut MeshSink,
    path: &StemPath,
    mp: &MeshParams,
    phase: (f32, f32),
    radial: u32,
) {
    let d = path.dirs[0];
    let (n, b) = path.frames[0];
    let arc = path.arc[1];

    let base = sink.vertex_offset();
    for j in 0..=radial {
        let a = j as f32 / radial as f32 * TAU;
        let e_r = n * a.cos() + b * a.sin();
        let e_a = b * a.cos() - n * a.sin();
        let r = path.radius_at(0, a, mp, phase);
        // The cone narrows to nothing over its length, so the normal leans back
        // along the axis by that slope instead of pointing straight out.
        let slope = if arc > 1e-6 { -r / arc } else { 0.0 };
        let normal = norm_or_zero(e_r - d * slope);
        let normal = if normal == Vec3::ZERO { e_r } else { normal };
        let uv = [
            j as f32 / radial as f32 * path.radii[0] * TAU / mp.uv_scale.max(1e-4),
            0.0,
        ];
        sink.push_vertex(path.points[0] + e_r * r, normal, e_a, uv);
    }

    let tip = path.points[1] + path.dirs[1] * mp.tip_length.max(path.radii[1] * 1.2);
    let apex = sink.vertex_offset();
    sink.push_vertex(tip, path.dirs[1], ortho_of(path.dirs[1]), [0.0, arc]);
    for j in 0..radial {
        sink.mesh
            .indices
            .extend_from_slice(&[base + j, base + j + 1, apex]);
    }
}

/// Sides swept around a stem, from how far the resulting polygon may sit inside the
/// circle it stands for.
///
/// Follows the thickest ring, so a tapering stem keeps its silhouette all the way down
/// instead of being sized by its average. Solving the error bound rather than scaling
/// a count by the radius is what lets a trunk and a twig both be right: the error of a
/// polygon is proportional to the radius, so a fixed count per metre spends far too
/// much on a twig and far too little on a limb.
fn radial_for(path: &StemPath, mp: &MeshParams) -> u32 {
    let r_max = path.radii.iter().copied().fold(0.0f32, f32::max) * path.socket[0];
    let tol = mp.silhouette_tolerance.max(1e-5);
    let sides = if r_max <= tol {
        3.0
    } else {
        // r * (1 - cos(pi / n)) = tol, solved for n.
        PI / (1.0 - tol / r_max).clamp(-1.0, 1.0).acos()
    };
    (sides.ceil() as i32).clamp(mp.min_radial.max(3) as i32, mp.max_radial.max(3) as i32) as u32
}

/// What one stem costs the mesh.
#[derive(Clone, Copy, Debug)]
pub struct StemCost {
    pub level: u8,
    pub nodes: usize,
    pub rings: usize,
    pub radial: u32,
    pub triangles: usize,
    pub vertices: usize,
    pub spike: bool,
}

/// What every stem costs, measured by emitting it rather than by modelling what the
/// mesher does. A model of the mesher is a second implementation to keep in step, and
/// it was already wrong the first time the ring count changed.
pub fn stem_costs(sk: &Skeleton, params: &SpeciesParams) -> Vec<StemCost> {
    let mp = &params.mesh;
    let phase = seed_phases(params.seed);
    let children = child_index(sk);
    let mut out = Vec::new();
    for stem in sk.stem_runs() {
        let Some(path) = StemPath::build(sk, &stem, mp, params.seed, &children) else {
            continue;
        };
        if path.radii.iter().copied().fold(0.0f32, f32::max) < mp.min_bark_radius {
            continue;
        }
        let mut sink = MeshSink {
            mesh: Mesh::default(),
        };
        emit_stem(&mut sink, &path, mp, phase);
        out.push(StemCost {
            level: sk.nodes[stem[0] as usize].level,
            nodes: stem.len(),
            rings: path.len(),
            radial: radial_for(&path, mp),
            triangles: sink.mesh.triangle_count(),
            vertices: sink.mesh.vertex_count(),
            spike: path.is_single_segment(),
        });
    }
    out
}

/// Which nodes start a stem of their own, indexed by the node they hang off.
fn child_index(sk: &Skeleton) -> Vec<Vec<u32>> {
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); sk.nodes.len()];
    for (i, node) in sk.nodes.iter().enumerate() {
        if let Some(parent) = node.parent
            && sk.nodes[parent as usize].stem != node.stem
        {
            children[parent as usize].push(i as u32);
        }
    }
    children
}

fn emit_stem(sink: &mut MeshSink, path: &StemPath, mp: &MeshParams, phase: (f32, f32)) {
    let last = path.len() - 1;
    let radial = radial_for(path, mp);

    // A single-segment branch is drawn as one cone from its anchor ring to a point,
    // rather than a ring pair swept into a tube and then capped with a cone as well.
    // The shape is the same at the scale these appear, for a third of the triangles,
    // and the tree is overwhelmingly made of them.
    if path.is_single_segment() {
        emit_spike(sink, path, mp, phase, radial);
        return;
    }

    // A twig ends twice over: the last ring repeats the one before it at almost the
    // same radius, and then a cone closes it anyway. Tapering the final stretch
    // straight to the tip is the same silhouette for a third of the triangles, and on
    // something a centimetre across it is arguably the better shape. The trunk keeps
    // its full sweep, since its top is not a twig.
    let swept = if path.is_trunk {
        path.len()
    } else {
        path.len() - 1
    };
    let mut ring_bases: Vec<u32> = Vec::with_capacity(swept);

    for i in 0..swept {
        let d = path.dirs[i];
        let (n, b) = path.frames[i];

        // Neighbours used for the axial slope of the radius, so the normal follows
        // taper and flare instead of pointing straight out of the axis.
        let i_prev = i.saturating_sub(1);
        let i_next = (i + 1).min(last);
        let ds = path.arc[i_next] - path.arc[i_prev];

        let base = sink.vertex_offset();
        for j in 0..=radial {
            let a = j as f32 / radial as f32 * TAU;
            let e_r = n * a.cos() + b * a.sin();
            let e_a = b * a.cos() - n * a.sin();
            let r = path.radius_at(i, a, mp, phase);

            let dr_da = (path.radius_at(i, a + NORMAL_DA, mp, phase)
                - path.radius_at(i, a - NORMAL_DA, mp, phase))
                / (2.0 * NORMAL_DA);
            let dr_ds = if ds > 1e-6 {
                (path.radius_at(i_next, a, mp, phase) - path.radius_at(i_prev, a, mp, phase)) / ds
            } else {
                0.0
            };
            let normal = norm_or_zero(e_r - e_a * (dr_da / r.max(1e-5)) - d * dr_ds);
            let normal = if normal == Vec3::ZERO { e_r } else { normal };

            let uv = [
                j as f32 / radial as f32 * path.radii[i] * TAU / mp.uv_scale.max(1e-4),
                path.arc[i] / mp.uv_scale.max(1e-4),
            ];
            sink.push_vertex(path.points[i] + e_r * r, normal, e_a, uv);
        }
        ring_bases.push(base);
    }

    for w in 0..ring_bases.len() - 1 {
        let bi = ring_bases[w];
        let bn = ring_bases[w + 1];
        for j in 0..radial {
            let a0 = bi + j;
            let a1 = bi + j + 1;
            let b0 = bn + j;
            let b1 = bn + j + 1;
            sink.mesh
                .indices
                .extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
        }
    }

    if path.is_trunk {
        // Only the trunk meets the ground, so it is the only stem that needs a disc.
        // The rim is duplicated with the cap normal to keep the edge crisp.
        let cap_n = -path.dirs[0];
        let (n0, b0) = path.frames[0];
        let rim = sink.vertex_offset();
        for j in 0..=radial {
            let a = j as f32 / radial as f32 * TAU;
            let e_r = n0 * a.cos() + b0 * a.sin();
            let pos = path.points[0] + e_r * path.radius_at(0, a, mp, phase);
            sink.push_vertex(pos, cap_n, ortho_of(cap_n), [0.0, 0.0]);
        }
        let center = sink.vertex_offset();
        sink.push_vertex(path.points[0], cap_n, ortho_of(cap_n), [0.0, 0.0]);
        for j in 0..radial {
            sink.mesh
                .indices
                .extend_from_slice(&[center, rim + j + 1, rim + j]);
        }
    }

    // Every stem closes with a cone: from its last ring for the trunk, and from the
    // ring before it for everything else, which is what makes the tip a taper.
    let end_dir = path.dirs[last];
    let r_end = path.radii[last] * path.socket[last];
    let tip = path.points[last] + end_dir * mp.tip_length.max(r_end * 1.2);
    let apex = sink.vertex_offset();
    sink.push_vertex(tip, end_dir, ortho_of(end_dir), [0.0, path.arc[last]]);
    let base = ring_bases[swept - 1];
    for j in 0..radial {
        sink.mesh
            .indices
            .extend_from_slice(&[base + j, base + j + 1, apex]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, OAK_RON, PINE_RON};


    #[test]
    fn triangles_wind_counter_clockwise_when_seen_from_outside() {
        // Consistent winding is what lets the renderer cull back faces and trust the
        // normal it was given, instead of flipping normals toward the viewer to hide
        // not knowing which way a triangle faces.
        for src in [OAK_RON, PINE_RON] {
            let params = parse_species(src).unwrap();
            let mesh = build_mesh(&crate::grow(&params), &params);
            let mut checked = 0;
            for tri in mesh.indices.chunks_exact(3) {
                let p: Vec<Vec3> = tri
                    .iter()
                    .map(|&i| Vec3::from(mesh.positions[i as usize]))
                    .collect();
                let geometric = (p[1] - p[0]).cross(p[2] - p[0]);
                if geometric.length() < 1e-9 {
                    continue;
                }
                // The shading normals are built outward by construction, so they say
                // which side is outside.
                let shading: Vec3 = tri
                    .iter()
                    .map(|&i| Vec3::from(mesh.normals[i as usize]))
                    .sum();
                assert!(
                    geometric.normalize().dot(shading.normalize_or_zero()) > 0.0,
                    "{}: triangle {tri:?} is wound inward",
                    params.name
                );
                checked += 1;
            }
            assert!(checked > 1000, "{}: only {checked} triangles", params.name);
        }
    }

    #[test]
    fn every_branch_tube_starts_on_its_parent() {
        // The whole tree is one welded surface only because each stem begins with a
        // ring on the parent centreline. Starting at the first grown node instead
        // leaves a segment-length hole at every junction.
        for src in [OAK_RON, PINE_RON] {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let mut checked = 0;
            for run in sk.stem_runs() {
                let first = run[0] as usize;
                let Some(p) = sk.nodes[first].parent else {
                    continue;
                };
                let path = StemPath::build(&sk, &run, &params.mesh, params.seed, &child_index(&sk)).expect("stem builds");
                let anchor = sk.nodes[p as usize].position;
                assert!(
                    (path.points[0] - anchor).length() < 1e-4,
                    "{}: stem starts {:?}, parent is at {:?}",
                    params.name,
                    path.points[0],
                    anchor
                );
                checked += 1;
            }
            assert!(checked > 50, "{}: only {checked} junctions", params.name);
        }
    }

    #[test]
    fn stem_surfaces_overlap_at_every_junction() {
        // A ring on the parent centreline is inside the parent only while the branch
        // stays thinner than what it hangs off.
        let params = parse_species(OAK_RON).unwrap();
        let sk = crate::grow(&params);
        for run in sk.stem_runs() {
            let first = run[0] as usize;
            let Some(p) = sk.nodes[first].parent else {
                continue;
            };
            let parent_radius = sk.nodes[p as usize].radius;
            let path = StemPath::build(&sk, &run, &params.mesh, params.seed, &child_index(&sk)).expect("stem builds");
            let base = path.radius_at(0, 0.0, &params.mesh, (0.0, 0.0));
            assert!(
                base <= parent_radius + 1e-4,
                "branch base {base} pokes out of parent {parent_radius}"
            );
        }
    }

    fn mesh_aabb_width_at_height(mesh: &Mesh, max_y: f32) -> f32 {
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for (p, n) in mesh.positions.iter().zip(mesh.normals.iter()) {
            if p[1] <= max_y && n[1].abs() < 0.9 {
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
                min_z = min_z.min(p[2]);
                max_z = max_z.max(p[2]);
            }
        }
        (max_x - min_x).max(max_z - min_z)
    }

    /// The trunk's path, which is what the bark actually shapes. Asking it directly
    /// beats hunting for mesh vertices at a given height: where the rings land is a
    /// cost decision, and a test of the shape should not break when that changes.
    fn trunk_path(params: &SpeciesParams, sk: &Skeleton) -> StemPath {
        let children = child_index(sk);
        let run = sk
            .stem_runs()
            .into_iter()
            .find(|r| sk.nodes[r[0] as usize].parent.is_none())
            .expect("a trunk");
        StemPath::build(sk, &run, &params.mesh, params.seed, &children).expect("builds")
    }

    #[test]
    fn a_trunk_cross_section_is_not_a_circle() {
        // Flutes. A bole that is round at every height reads as pipe however good the
        // bark texture on it is.
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let path = trunk_path(&params, &sk);
        let phase = seed_phases(params.seed);
        let mut worst = 0.0f32;
        for i in (path.len() / 6)..(path.len() * 5 / 6) {
            let r: Vec<f32> = (0..32)
                .map(|k| path.radius_at(i, k as f32 / 32.0 * TAU, &params.mesh, phase))
                .collect();
            let lo = r.iter().copied().fold(f32::MAX, f32::min);
            let hi = r.iter().copied().fold(0.0f32, f32::max);
            worst = worst.max(hi / lo);
        }
        assert!(
            worst > 1.12,
            "the widest section is round to within {worst}, so the bole is a pipe"
        );
    }

    #[test]
    fn a_trunk_swells_and_waists_along_its_length() {
        // Taper alone is monotonic: a real bole also thickens and thins as it goes.
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let path = trunk_path(&params, &sk);
        let phase = seed_phases(params.seed);
        let mean: Vec<f32> = (0..path.len())
            .map(|i| {
                (0..16)
                    .map(|k| path.radius_at(i, k as f32 / 16.0 * TAU, &params.mesh, phase))
                    .sum::<f32>()
                    / 16.0
            })
            .collect();
        let rises = mean.windows(2).filter(|w| w[1] > w[0] + 1e-5).count();
        assert!(
            rises >= 6,
            "the trunk only ever narrows, so it is a cone: {rises} rises of {}",
            mean.len()
        );
    }

    #[test]
    fn twigs_are_left_round() {
        // The irregularity is paid for in rings. Anything too thin to show it should
        // not be carrying any, and a twig is round to begin with.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.irregularity.min_radius = 0.2;
        let sk = crate::grow(&params);
        let children = child_index(&sk);
        let mut checked = 0;
        for run in sk.stem_runs() {
            let path =
                StemPath::build(&sk, &run, &params.mesh, params.seed, &children).expect("builds");
            let thickest = path.radii.iter().copied().fold(0.0f32, f32::max);
            if thickest >= 0.2 {
                continue;
            }
            let i = path.len() / 2;
            let a = path.radius_at(i, 0.0, &params.mesh, (0.0, 0.0));
            let b = path.radius_at(i, 1.7, &params.mesh, (0.0, 0.0));
            assert!(
                (a - b).abs() < a * 1e-3,
                "a thin stem came out fluted: {a} against {b}"
            );
            checked += 1;
        }
        assert!(checked > 100, "only {checked} thin stems");
    }

    #[test]
    fn bark_irregularity_is_deterministic_and_seed_dependent() {
        let mut a = parse_species(PINE_RON).unwrap();
        let mut b = parse_species(PINE_RON).unwrap();
        a.seed = 21;
        b.seed = 22;
        let ma = build_mesh(&crate::grow(&a), &a);
        assert_eq!(ma, build_mesh(&crate::grow(&a), &a));
        let mb = build_mesh(&crate::grow(&b), &b);
        assert_ne!(ma.positions, mb.positions);
    }

    #[test]
    fn a_trunk_thickens_where_a_branch_leaves() {
        // A limb grows out of its parent, so the parent swells to meet it. Without the
        // collar the two tubes meet at a seam, which is the giveaway that a tree was
        // assembled out of pipes.
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let children = child_index(&sk);
        let run = sk
            .stem_runs()
            .into_iter()
            .find(|r| sk.nodes[r[0] as usize].parent.is_none())
            .expect("a trunk");
        let path = StemPath::build(&sk, &run, &params.mesh, params.seed, &children).expect("builds");
        assert!(
            !path.irregular.collars.is_empty(),
            "the trunk carries branches but recorded no collars"
        );

        let mut checked = 0;
        for collar in &path.irregular.collars {
            // The ring nearest where the branch leaves, and the angle facing it.
            let i = path
                .arc
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    (a.1 - collar.arc)
                        .abs()
                        .total_cmp(&(b.1 - collar.arc).abs())
                })
                .map(|(i, _)| i)
                .unwrap();
            let (n, b) = path.frames[i];
            // Recover the angle of the collar in this ring's frame.
            let a_face = collar.dir.dot(b).atan2(collar.dir.dot(n));
            let facing = path.radius_at(i, a_face, &params.mesh, (0.0, 0.0));
            let away = path.radius_at(i, a_face + PI, &params.mesh, (0.0, 0.0));
            if facing > away {
                checked += 1;
            }
        }
        assert!(
            checked * 2 > path.irregular.collars.len(),
            "only {checked} of {} collars thickened the side the branch leaves from",
            path.irregular.collars.len()
        );
    }

    #[test]
    fn thin_twigs_can_be_left_unmeshed() {
        // The finest twigs are a third of the bark and two thirds of its vertices, and
        // under foliage nobody can see them. Dropping them has to drop only them.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.min_bark_radius = 0.0;
        let sk = crate::grow(&params);
        let all = build_mesh(&sk, &params);

        let cutoff = 0.0045;
        params.mesh.min_bark_radius = cutoff;
        let trimmed = build_mesh(&sk, &params);
        assert!(
            trimmed.triangle_count() < all.triangle_count() * 4 / 5,
            "the cutoff saved almost nothing: {} against {}",
            trimmed.triangle_count(),
            all.triangle_count()
        );

        // Everything above the cutoff is still there, the trunk included.
        let kept: usize = stem_costs(&sk, &params).len();
        let thick = sk
            .stem_runs()
            .iter()
            .filter(|run| {
                run.iter()
                    .map(|&i| sk.nodes[i as usize].radius)
                    .fold(0.0f32, f32::max)
                    >= cutoff
            })
            .count();
        assert!(
            kept >= thick,
            "{kept} stems kept but {thick} are thick enough to keep"
        );
        let (min, max) = trimmed.aabb();
        assert!(min[1] > -0.3 && max[1] > 5.0, "the trunk went with them");

        // Nothing may be left floating: a stem that survives the cutoff must hang off
        // one that also survived. This holds because a child is never thicker than its
        // parent, and the moment that stops being true the tree comes apart.
        let thickest = |run: &[u32]| {
            run.iter()
                .map(|&i| sk.nodes[i as usize].radius)
                .fold(0.0f32, f32::max)
        };
        let runs = sk.stem_runs();
        let mut checked = 0;
        for run in &runs {
            if thickest(run) < cutoff {
                continue;
            }
            let Some(parent) = sk.nodes[run[0] as usize].parent else {
                continue;
            };
            let owner = runs
                .iter()
                .find(|r| r.contains(&parent))
                .expect("every node belongs to a stem");
            assert!(
                thickest(owner) >= cutoff,
                "a kept stem hangs off one that was dropped, so it floats"
            );
            checked += 1;
        }
        assert!(checked > 100, "only {checked} junctions checked");
    }

    #[test]
    fn mesh_is_deterministic() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let a = build_mesh(&sk, &params);
        let b = build_mesh(&sk, &params);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_meshes() {
        let mut a = parse_species(PINE_RON).unwrap();
        let mut b = parse_species(PINE_RON).unwrap();
        a.seed = 10;
        b.seed = 11;
        let ma = build_mesh(&crate::grow(&a), &a);
        let mb = build_mesh(&crate::grow(&b), &b);
        assert_ne!(ma.positions, mb.positions);
    }

    #[test]
    fn indices_in_range_and_normals_unit() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        let count = mesh.vertex_count() as u32;
        assert!(count > 100, "mesh too small: {count}");
        assert!(mesh.triangle_count() > 1000, "expected a substantial mesh");
        for &i in &mesh.indices {
            assert!(i < count, "index {i} out of range {count}");
        }
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 0.02, "normal length {len}");
        }
        for (n, t) in mesh.normals.iter().zip(mesh.tangents.iter()) {
            let dot = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
            assert!(dot.abs() < 0.02, "tangent not orthogonal: dot={dot}");
        }
    }

    #[test]
    fn uvs_are_sane() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        for uv in &mesh.uvs {
            assert!(uv[0].is_finite() && uv[1].is_finite());
            assert!(uv[0] >= -1e-4 && uv[1] >= -1e-4);
        }
    }

    #[test]
    fn root_flare_widens_base() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        let base_width = mesh_aabb_width_at_height(&mesh, 0.25);
        assert!(
            base_width > params.trunk.radius * 2.0 + 0.15,
            "base width {base_width} suggests no flare"
        );
    }

    #[test]
    fn mesh_sits_on_ground() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        let (min, max) = mesh.aabb();
        assert!(min[1] > -0.3, "mesh dips below ground: {}", min[1]);
        assert!(max[1] > 5.0, "mesh too short: {}", max[1]);
    }
}
