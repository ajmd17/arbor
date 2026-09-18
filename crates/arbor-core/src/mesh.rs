use glam::Vec3;
use std::f32::consts::{PI, TAU};

use crate::math::{norm_or_zero, ortho_of, ortho_unit, transport};
use crate::skeleton::Skeleton;
use crate::species::{BarkIrregularity, MeshParams, SpeciesParams};
use crate::wind::{Sway, SwayAt, StemSway, SwayField};

/// Angular step used for the finite-difference normal around a ring.
const NORMAL_DA: f32 = 0.01;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    /// Per vertex, 1 on wood the tree has lost and 0 on living wood, for the renderer
    /// to weather by `dead_wood_weathering`.
    pub weathering: Vec<f32>,
    /// Per vertex, the wood it hangs off, for the renderer to sway in the wind.
    pub sway: Vec<Sway>,
    /// Per vertex, the same told by stem, for an engine that keeps a table of stems.
    pub sway_at: Vec<SwayAt>,
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
/// Also returns where each new point came from: the span of the input it lies on, and
/// how far along that span, 0 at its start. The last point is the end of the last span.
#[allow(clippy::type_complexity)]
fn subdivide(
    points: &[Vec3],
    radii: &[f32],
    per_meter: f32,
) -> (Vec<Vec3>, Vec<f32>, Vec<(usize, f32)>) {
    let n = points.len();
    if n < 2 {
        return (points.to_vec(), radii.to_vec(), (0..n).map(|i| (i, 0.0)).collect());
    }
    let mut out_p = Vec::with_capacity(n * 4);
    let mut out_r = Vec::with_capacity(n * 4);
    let mut from = Vec::with_capacity(n * 4);
    for i in 0..n - 1 {
        let (a, b) = (i.saturating_sub(1), (i + 2).min(n - 1));
        let steps = (((points[i + 1] - points[i]).length() * per_meter).ceil() as usize).max(1);
        for k in 0..steps {
            let t = k as f32 / steps as f32;
            out_p.push(catmull(points[a], points[i], points[i + 1], points[b], t));
            // Clamped to the pair it sits between: a Catmull-Rom curve overshoots at
            // an inflection, and on a radius that shows up as the bole thickening
            // slightly where it should only ever taper.
            let (lo, hi) = (radii[i].min(radii[i + 1]), radii[i].max(radii[i + 1]));
            out_r.push(
                catmull_f32(radii[a], radii[i], radii[i + 1], radii[b], t)
                    .clamp(lo, hi)
                    .max(1e-4),
            );
            from.push((i, t));
        }
    }
    out_p.push(points[n - 1]);
    out_r.push(radii[n - 1]);
    from.push((n - 1, 0.0));
    (out_p, out_r, from)
}

/// A swelling where a branch leaves its parent.
struct Collar {
    /// Distance along the parent at which the child attaches.
    arc: f32,
    /// Which way the child heads, across the parent's axis.
    dir: Vec3,
    /// That direction as an angle in the ring's own frame, which is the space knots
    /// are placed in.
    angle: f32,
    radius: f32,
}

/// A scar where a limb was lost: a dimple inside a raised collar.
struct Knot {
    arc: f32,
    angle: f32,
    sigma_s: f32,
    sigma_a: f32,
    amp: f32,
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
    /// Each swelling along the length: frequency, phase, weight, and how many times it
    /// travels round the stem over one turn. That last term is what keeps a swelling
    /// from being a ring.
    swells: [(f32, f32, f32, f32); 3],
    burls: Vec<Burl>,
    knots: Vec<Knot>,
    collars: Vec<Collar>,
    active: bool,
    /// A trunk has no parent to start inside, so it keeps its bark all the way down.
    is_trunk: bool,
    /// Kept so knots can be drawn after the collars are known.
    seed: u64,
    stem: u64,
    length: f32,
    radius: f32,
}

impl Irregular {
    fn none() -> Self {
        Self {
            params: BarkIrregularity::default(),
            flutes: [(0.0, 0.0, 0.0); 3],
            swells: [(0.0, 0.0, 0.0, 0.0); 3],
            burls: Vec::new(),
            knots: Vec::new(),
            collars: Vec::new(),
            active: false,
            is_trunk: false,
            seed: 0,
            stem: 0,
            length: 0.0,
            radius: 0.0,
        }
    }

    fn new(
        p: &BarkIrregularity,
        seed: u64,
        stem: u64,
        length: f32,
        radius: f32,
        is_trunk: bool,
    ) -> Self {
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
        // Swellings travel round the bole as they climb it, so a bulge is a lump on one
        // side rather than a ring at one height. A term that varies only along the stem
        // puts the same bulge right the way round, and a bole built from those reads as
        // a stack of discs. The periods are deliberately not multiples of each other,
        // so a long stem never repeats the same profile twice.
        let period = p.swell_period.max(0.05);
        let swells = [
            (TAU / period, r(7) * TAU, 0.55, 1.0),
            (TAU / (period * 0.61), r(8) * TAU, 0.28, 2.0),
            (TAU / (period * 0.27), r(9) * TAU, 0.17, 3.0),
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
            knots: Vec::new(),
            collars: Vec::new(),
            active: true,
            is_trunk,
            seed,
            stem,
            length,
            radius,
        }
    }

    /// Places the knots, once the collars are known.
    ///
    /// A knot is a branch that died and was grown over, so one cannot sit on a limb the
    /// tree still has, and two of them on the same spot read as damage rather than as
    /// history. Candidates are drawn and rejected until they clear both, which is why
    /// this runs after the collars rather than in the constructor with everything else.
    ///
    /// Where they may sit at all comes from the same reasoning. A scar can only be
    /// where a branch once was, so the zone runs from a little under the lowest branch
    /// the stem still carries up to its tip; a stem carrying no branches at all has
    /// lost none either. And a stem has to be thicker than the scar to have grown over
    /// it, which keeps knots off the twigs and stops one wrapping half way round a thin
    /// limb.
    fn place_knots(&mut self) {
        let p = &self.params;
        if !self.active {
            return;
        }
        let Some(lowest) = self
            .collars
            .iter()
            .map(|c| c.arc)
            .reduce(f32::min)
        else {
            return;
        };
        let floor = (lowest - p.knot_reach.max(0.0) * self.radius).max(0.0);
        let span = self.length - floor;
        let want = (span * p.knot_density).round().max(0.0) as usize;
        if span <= 0.0 || want == 0 {
            return;
        }
        let r = |i: u64| hash01(self.seed, self.stem.wrapping_mul(1409).wrapping_add(i));
        let mut attempt = 0u64;
        while self.knots.len() < want && attempt < want as u64 * 24 {
            let i = 400 + attempt * 7;
            attempt += 1;
            let size = p.knot_size * (0.6 + 0.8 * r(i + 3));
            if self.radius < size * p.knot_min_stem.max(1.0) {
                continue;
            }
            let candidate = Knot {
                arc: floor + r(i) * span,
                angle: r(i + 1) * TAU,
                sigma_s: size.max(0.02),
                sigma_a: (size / self.radius.max(1e-3)).clamp(0.2, 0.9),
                amp: p.knot_depth * (0.6 + 0.8 * r(i + 2)),
            };
            if self.clashes(&candidate) {
                continue;
            }
            self.knots.push(candidate);
        }
    }

    /// Whether a knot would land on a living branch or on another knot.
    ///
    /// Both are measured in the same stretched space the knot is drawn in, so a knot
    /// that is wide round a thin stem needs a correspondingly wide berth.
    fn clashes(&self, k: &Knot) -> bool {
        let overlaps = |arc: f32, angle: f32, sigma_s: f32, sigma_a: f32| {
            let ds = (k.arc - arc) / (k.sigma_s + sigma_s).max(1e-4);
            let da = wrap_pi(k.angle - angle) / (k.sigma_a + sigma_a).max(1e-4);
            ds * ds + da * da < 1.0
        };
        self.collars.iter().any(|c| {
            // The collar's own reach along the stem, and how far round it swells.
            overlaps(c.arc, c.angle, (c.radius * 2.6).max(0.04), 0.9)
        }) || self.knots.iter().any(|o| overlaps(o.arc, o.angle, o.sigma_s, o.sigma_a))
    }

    /// Multiplier on the radius at distance `s` along the stem, at angle `a` whose
    /// outward direction is `e_r`.
    fn shape(&self, s: f32, a: f32, e_r: Vec3, radius: f32) -> f32 {
        if !self.active {
            return 1.0;
        }
        let p = &self.params;
        // Faded in off the socket, so a branch still starts inside its parent and the
        // junction stays welded. A trunk has nothing to start inside, and fading it
        // there left the one part of the tree people look at closest - the foot -
        // as the only part with no bark relief on it at all.
        let fade = if self.is_trunk {
            1.0
        } else {
            smoothstep(0.0, radius * 4.0, s)
        };

        let mut flute = 0.0;
        for (f, phase, twist) in self.flutes {
            flute += (a * f + phase + s * twist).sin() / f.max(1.0).sqrt();
        }
        let mut swell = 0.0;
        for (w, phase, weight, around) in self.swells {
            swell += (s * w + phase + a * around).sin() * weight;
        }
        let mut lumps = 0.0;
        for b in &self.burls {
            let ds = (s - b.arc) / b.sigma_s;
            let da = wrap_pi(a - b.angle) / b.sigma_a;
            lumps += b.amp * (-(ds * ds + da * da)).exp();
        }
        for k in &self.knots {
            let ds = (s - k.arc) / k.sigma_s;
            let da = wrap_pi(a - k.angle) / k.sigma_a;
            let d2 = ds * ds + da * da;
            // A raised collar at the rim with a shallow dish inside it. Both are
            // gaussians, so the whole thing stays smooth and the analytic normals
            // follow it. The rim leads and the dish trails it: wood laid down around a
            // lost branch swells over the wound faster than the wound itself sinks, so
            // what a healed scar reads as is a ring standing proud with a dip in the
            // middle, not a bore hole. Weighting them the other way up sank the centre
            // by half the radius of the stem, which is a cave.
            let d = d2.sqrt();
            let rim = (-((d - 1.0) * (d - 1.0)) / 0.35).exp();
            let core = (-d2 / 0.5).exp();
            lumps += k.amp * (0.77 * rim - 0.54 * core);
        }
        for c in &self.collars {
            let reach = (c.radius * 2.6).max(0.04);
            let ds = (s - c.arc) / reach;
            // Only on the side the branch leaves from.
            let facing = e_r.dot(c.dir).max(0.0);
            let share = (c.radius / radius.max(1e-4)).min(1.0);
            lumps += p.collar_depth * share * (-(ds * ds)).exp() * facing * facing * facing;

            // The branch bark ridge, above the crotch only: a much narrower seam than
            // the collar, peaking a little way up the parent and dying out above it.
            if p.bark_ridge > 0.0 {
                let up = ds - 0.55;
                let along = (-(up * up) / 0.45).exp() * smoothstep(-0.2, 0.2, ds);
                // Broad rather than knife-edged, because that is the shape a real
                // ridge is. Narrowing it saves nothing: what the sweep charges for is
                // the height of the bump, not how sharp it is, since the sides are
                // sized by how far a chord falls from the surface.
                let seam = facing.powi(4);
                lumps += p.bark_ridge * share * along * seam;
            }
        }

        1.0 + fade * (p.flute_depth * flute + p.swell_depth * swell + lumps)
    }
}

#[derive(Default)]
struct MeshSink {
    mesh: Mesh,
    /// Written onto every vertex pushed until it is changed: the stem being swept is
    /// dead or it is not, and everything in it is the same.
    weathering: f32,
    /// How the stem being swept hangs off the wood carrying it, read off at each
    /// vertex's distance along it.
    sway: StemSway,
}

impl MeshSink {
    /// `arc` is how far along the stem the vertex sits, from where the stem leaves its
    /// parent, which is what its sway is read off.
    fn push_vertex(&mut self, pos: Vec3, normal: Vec3, tangent: Vec3, uv: [f32; 2], arc: f32) {
        self.mesh.positions.push(pos.to_array());
        self.mesh.normals.push(normal.to_array());
        let t = norm_or_zero(tangent - normal * tangent.dot(normal));
        self.mesh.tangents.push([t.x, t.y, t.z, 1.0]);
        self.mesh.uvs.push(uv);
        self.mesh.weathering.push(self.weathering);
        self.mesh.sway.push(self.sway.at(arc));
        self.mesh.sway_at.push(self.sway.place(arc));
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

/// How far above `flare_height` the trunk keeps its fine rings, as a multiple of it.
/// The flare dies away smoothly rather than stopping, so its tail wants them too.
const FLARE_RING_REACH: f32 = 1.3;
/// Ring budget inside the trunk's flare zone, in metres, however coarse the rest.
const FLARE_RING_TOLERANCE: f32 = 0.01;
/// Most a stem may turn between one kept ring and the next, in radians: twenty degrees.
///
/// The ring budget is a distance, and a distance is the wrong measure for a bend. A limb
/// that curves five degrees a node for twenty nodes turns a hundred degrees in a smooth
/// arc, and every one of those nodes sits within three centimetres of the straight line
/// to its neighbours, so a coarse budget kept three rings of it and swept the arc as two
/// straight spans meeting at a sixty-degree elbow — thirty-five of the pine's limbs had
/// one, on a skeleton whose sharpest corner is five degrees. Nothing about the distance
/// budget would ever catch that, because the eye is reading the corner, not the offset.
/// Twenty degrees is where the elbows stop showing at the distance a bare limb is looked
/// at, for about half as many extra rings again as fifteen.
const MAX_SPAN_TURN: f32 = 0.349;

/// Which rings to keep. Walks the stem dropping any ring whose absence moves the
/// surface less than the tolerance, never two in a row, so the result still follows
/// every feature the bark has while spending nothing on the stretches between them.
fn keep_rings(path: &StemPath, mp: &MeshParams, phase: (f32, f32)) -> Vec<usize> {
    // One error budget for the whole surface. Measuring along the stem in metres and
    // around it in metres means a ring and a side are bought at the same price; while
    // this was a fraction of the radius, a twig was held to a fifth of a millimetre
    // along its length and a centimetre around it.
    let tol = mp.irregularity.ring_tolerance.min(mp.silhouette_tolerance);
    if tol <= 0.0 || path.len() < 3 {
        return (0..path.len()).collect();
    }
    let mut keep = Vec::with_capacity(path.len());
    keep.push(0);
    // The buttress is the place on a tree where the shape changes fastest along the
    // stem and matters most up close, and a coarse budget smooths its flare away over
    // a couple of metres. The trunk's flare zone is held to the finest budget the
    // mesher uses by default instead, which costs a handful of rings on one stem.
    let flare_top = mp.flare_height * FLARE_RING_REACH;
    let flare_tol = tol.min(FLARE_RING_TOLERANCE);
    let mut i = 1;
    while i < path.len() - 1 {
        let prev = *keep.last().unwrap();
        let fine = path.is_trunk && path.arc[i] < flare_top;
        let limit = if fine { flare_tol } else { tol };
        // Three things keep a ring: a branch leaving from it, the surface moving more
        // than the budget without it, and the stem turning more than a span may
        // across it.
        let turns = path.dirs[prev].angle_between(path.dirs[i + 1]) > MAX_SPAN_TURN;
        if !path.pinned[i] && !turns && ring_error(path, mp, phase, prev, i, i + 1) < limit {
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
    /// How far along the socket each ring sits, 0 at the parent and 1 where the flare
    /// has died away. The flare itself is evaluated per angle, not per ring, because a
    /// real collar is not the same all the way round.
    socket_t: Vec<f32>,
    /// Ceiling on the flare: what the parent can still hide.
    socket_max: f32,
    /// The underside of this stem, across its own axis.
    under: Vec3,
    /// Distance travelled along the stem, used for the V coordinate.
    arc: Vec<f32>,
    /// Distance along the skeleton from where the stem leaves its parent, at each
    /// ring: what its sway is read off. Not `arc`, which follows the smoothed curve the
    /// bark is swept along and so runs a little long of the skeleton on a bent stem.
    /// Read by that, the ring a branch leaves from swings a little further than the
    /// branch's own foot does, and the two part in the wind.
    sway_arc: Vec<f32>,
    /// Rings thinning may not drop: the ones a branch leaves from. The wind bends
    /// wood between rings but the renderer only moves the rings themselves, joining
    /// them with straight spans; a branch whose parent has no ring where it leaves
    /// rides the curve while the parent's surface there cuts the chord, and in a gust
    /// the two come apart by more than the parent is thick.
    pinned: Vec<bool>,
    /// True for the stem that starts at the root node, which gets the buttress flare
    /// and the ground cap.
    is_trunk: bool,
    /// True for a stem the tree has lost: no leaves, and a broken end rather than a
    /// tapered one.
    is_dead: bool,
    /// True for dead wood that reaches the trunk through nothing but dead wood: the
    /// band of dead limbs under a crown, with no foliage anywhere near to hide it.
    /// Dead wood hanging off a living limb is inside the crown, and is not.
    is_bare: bool,
    /// Branching order of the stem, 0 for the trunk.
    level: u8,
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
        Some(Self::build_dense(sk, stem, mp, seed, children)?.thinned(mp, seed_phases(seed)))
    }

    /// The stem at full ring density, before thinning.
    fn build_dense(
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
        let mut pinned: Vec<bool> = Vec::with_capacity(stem.len() + 1);
        // Only a branch sways about where it is attached; the trunk bends by height
        // alone, which is smooth across any ring spacing, so it pins nothing.
        let sways = sk.nodes[first].level > 0;
        // How wide the socket may get before it stops being swallowed by the parent.
        let mut socket_ceiling = f32::INFINITY;
        if let Some(p) = sk.nodes[first].parent {
            // Anchor the stem on its parent so the two tubes overlap at the junction.
            let anchor = sk.nodes[p as usize].position;
            if (anchor - sk.nodes[first].position).length() > 1e-5 {
                points.push(anchor);
                radii.push(sk.nodes[first].radius);
                pinned.push(false);
            }
            socket_ceiling = sk.nodes[p as usize].radius;
        }
        for &i in stem {
            let node = &sk.nodes[i as usize];
            // A branch too thin to be swept has nothing to come apart from, and pinning
            // a ring for every twig would keep nearly all of them on a fine stem.
            let carries = sways
                && children
                    .get(i as usize)
                    .is_some_and(|c| c.iter().any(|&k| gets_bark(sk, k as usize, mp)));
            // Skip duplicate positions: a zero-length segment gives no usable axis.
            if points
                .last()
                .is_some_and(|p: &Vec3| (*p - node.position).length() < 1e-5)
            {
                if let Some(pin) = pinned.last_mut() {
                    *pin |= carries;
                }
                continue;
            }
            points.push(node.position);
            radii.push(node.radius);
            pinned.push(carries);
        }
        if points.len() < 2 {
            return None;
        }
        // Measured before the trunk grows its root below, and before the bark is
        // smoothed: this is the skeleton's own length, which is what the sway is.
        let mut sway_arc = arcs_for(&points);

        // The trunk starts below the ground, so the ground plane hides its cap and the
        // roots are seen entering the soil rather than cut off flush with it.
        if is_trunk && mp.root_depth > 1e-4 {
            let down = norm_or_zero(points[0] - points[1]);
            let below = if down == Vec3::ZERO { Vec3::NEG_Y } else { down };
            points.insert(0, points[0] + below * mp.root_depth);
            radii.insert(0, radii[0]);
            sway_arc.insert(0, 0.0);
            pinned.insert(0, false);
        }

        // A skeleton is segmented for growing, which is far coarser than the bark needs:
        // a half-metre between rings cannot describe a burl. Stems thick enough to show
        // the detail get extra rings along a smooth curve through the ones they have.
        let r_max = radii.iter().copied().fold(0.0f32, f32::max);
        let ir = &mp.irregularity;
        if r_max >= ir.min_radius && ir.rings_per_meter > 0.0 {
            let (p, r, from) = subdivide(&points, &radii, ir.rings_per_meter);
            // Every original point comes through at a whole step, so the pins land on
            // exactly the rings they were set on; the rings between run along the
            // skeleton's span as the bark runs along its curve.
            let last = sway_arc.len() - 1;
            let (arcs, pins) = from
                .iter()
                .map(|&(i, t)| {
                    let arc = if i < last {
                        sway_arc[i] + (sway_arc[i + 1] - sway_arc[i]) * t
                    } else {
                        sway_arc[last]
                    };
                    (arc, t == 0.0 && pinned[i])
                })
                .unzip();
            points = p;
            radii = r;
            sway_arc = arcs;
            pinned = pins;
        }

        let dirs = dirs_for(&points);
        let arc = arcs_for(&points);

        // The socket fades out over a few base radii, so the flare is sized by the
        // branch rather than by however finely the stem happens to be segmented.
        let socket_len = (radii[0] * 3.0).max(1e-4);
        // Cap the flare at whatever the parent can hide. Where both stems are already
        // at the radius floor there is nothing to hide it in, and an unclamped flare
        // would leave a bud sticking out of the junction.
        let socket_max = (socket_ceiling / radii[0].max(1e-5)).max(1.0);
        let socket_t: Vec<f32> = if is_trunk {
            vec![1.0; arc.len()]
        } else {
            arc.iter().map(|&s| (s / socket_len).clamp(0.0, 1.0)).collect()
        };
        // Which way is down, across this stem. A limb thickens underneath where it
        // carries its own weight, and that asymmetry is most of what tells a grown
        // junction from a glued one.
        let under = {
            let axis = dirs[0];
            let d = Vec3::NEG_Y - axis * Vec3::NEG_Y.dot(axis);
            d.try_normalize().unwrap_or_else(|| ortho_of(axis))
        };

        // Transport runs here rather than in the sweep, so the rings, the ground cap
        // and the bark all read one frame instead of three.
        let frames = frames_for(&dirs);

        let mut irregular = Irregular::new(
            &mp.irregularity,
            seed,
            sk.nodes[first].stem as u64,
            *arc.last().unwrap_or(&0.0),
            radii.iter().copied().fold(0.0f32, f32::max),
            is_trunk,
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
                    let (fu, fv) = frames[j];
                    irregular.collars.push(Collar {
                        arc: arc[j],
                        angle: dir.dot(fv).atan2(dir.dot(fu)),
                        dir,
                        radius: c.radius,
                    });
                }
            }
        }

        // Now the collars are known, the knots can be drawn clear of them.
        irregular.place_knots();

        let dense = StemPath {
            points,
            dirs,
            radii,
            socket_t,
            socket_max,
            under,
            arc,
            sway_arc,
            pinned,
            is_trunk,
            is_dead: sk.nodes[first].dead,
            is_bare: hangs_from_trunk_by_dead_wood(sk, first),
            level: sk.nodes[first].level,
            frames,
            irregular,
        };
        Some(dense)
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
        let socket_t: Vec<f32> = keep.iter().map(|&i| self.socket_t[i]).collect();
        // Arc comes from the dense polyline, so thinning cannot slide the bark along
        // the stem and move the burls.
        let arc: Vec<f32> = keep.iter().map(|&i| self.arc[i]).collect();
        let sway_arc: Vec<f32> = keep.iter().map(|&i| self.sway_arc[i]).collect();
        let pinned: Vec<bool> = keep.iter().map(|&i| self.pinned[i]).collect();
        // Each kept ring keeps the aim it had on the dense stem rather than being
        // re-aimed at its new neighbours. Those can be metres off once a coarse budget
        // has thinned a smooth run — the ring after a socket, say, whose next kept ring
        // is far out along the limb — and a ring aimed across that gap turns until its
        // wall folds through the ring before it, inside out. The dense aim is the
        // stem's own tangent there, which is what the sweep wants anyway.
        let dirs: Vec<Vec3> = keep.iter().map(|&i| self.dirs[i]).collect();
        let frames = frames_for(&dirs);
        StemPath {
            points,
            dirs,
            radii,
            socket_t,
            socket_max: self.socket_max,
            under: self.under,
            arc,
            sway_arc,
            pinned,
            is_trunk: self.is_trunk,
            is_dead: self.is_dead,
            is_bare: self.is_bare,
            level: self.level,
            frames,
            irregular: self.irregular,
        }
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    /// Widening at the foot of a branch, where it meets its parent.
    ///
    /// Two swept tubes meeting can only ever intersect, so there is no fillet to be
    /// had; what stands in for one is flaring the limb hard over the last stretch
    /// before the surfaces cross, so both are already heading toward each other by the
    /// time they meet. The flare is heavier underneath, where a limb lays down wood to
    /// carry its own weight.
    fn socket_at(&self, i: usize, e_r: Vec3, mp: &MeshParams) -> f32 {
        let t = self.socket_t[i];
        if t >= 1.0 {
            return 1.0;
        }
        let k = (1.0 - t).powf(mp.socket_power.max(0.5));
        let under = e_r.dot(self.under).max(0.0);
        let lean = 1.0 + mp.socket_bias * under * under;
        (1.0 + mp.socket_flare * k * lean).min(self.socket_max)
    }

    /// Surface radius of ring `i` at angle `a`, including socket, root flare and the
    /// flutes, swellings, burls and branch collars that make it bark rather than pipe.
    fn radius_at(&self, i: usize, a: f32, mp: &MeshParams, phase: (f32, f32)) -> f32 {
        let (n, b) = self.frames[i];
        let e_r = n * a.cos() + b * a.sin();
        let base = self.radii[i] * self.socket_at(i, e_r, mp);
        let shaped = base * self.irregular.shape(self.arc[i], a, e_r, base);

        shaped * self.buttress(a, self.points[i].y, mp, phase)
    }

    /// Widening at the foot of the trunk, as a multiplier on the radius.
    ///
    /// A ring of peaked ridges rather than a cone: an old broadleaf stands on a few
    /// buttress roots with hollows between them, and it is the hollows that make it
    /// read as a root system rather than as a skirt. Everything here is a smooth
    /// function of the angle, so the analytic normals follow it for free.
    fn buttress(&self, a: f32, y: f32, mp: &MeshParams, phase: (f32, f32)) -> f32 {
        if !self.is_trunk || y >= mp.flare_height || mp.flare_height <= 1e-4 {
            return 1.0;
        }
        // Not clamped below zero: under the ground the flare keeps growing, which is
        // what carries the roots outward as they go down.
        let t = (y / mp.flare_height).min(1.0);
        let k = (1.0 - t).powf(mp.root_taper.max(0.2));
        let n = mp.root_count.max(1) as f32;
        // The lower the ridges go the narrower they get, so the ring of lobes parts
        // into separate arms on the way down instead of staying a skirt. This has to
        // build from the top of the flare downward, not from the ground: everything
        // below the ground is hidden by it, so sharpening only down there would shape
        // the one part of the tree nobody can see.
        let sharpen = 1.0 + mp.root_split * (1.0 - t).max(0.0).powi(2);
        // Warping the angle before the ridges are laid out spaces them unevenly,
        // which is what a real root collar does and a cog does not.
        let warped = a + 0.35 * (a + phase.1).sin();
        let ridge =
            (0.5 + 0.5 * (n * warped + phase.0).cos()).powf(mp.root_sharpness.max(0.1) * sharpen);
        // Roots are not evenly sized, so a slow wave rides over the ring of them, and
        // each one carries finer ridges of its own: a real buttress is grooved, not a
        // set of smooth cones.
        let uneven = 0.72 + 0.28 * (a * 2.0 + phase.1).sin();
        let grooves = 1.0 + mp.root_grooves * (a * n * 3.0 + phase.0 * 2.0).sin() * k;
        1.0 + mp.root_flare * k * ridge * uneven * grooves
    }
}

/// Whether the stem starting at `first` is dead and reaches the trunk through nothing
/// but dead wood.
fn hangs_from_trunk_by_dead_wood(sk: &Skeleton, first: usize) -> bool {
    let mut at = first;
    loop {
        let node = &sk.nodes[at];
        if !node.dead {
            return node.level == 0;
        }
        match node.parent {
            Some(p) => at = p as usize,
            None => return false,
        }
    }
}

pub fn build_mesh(sk: &Skeleton, params: &SpeciesParams) -> Mesh {
    let mp: &MeshParams = &params.mesh;
    let mut sink = MeshSink::default();
    let phase = seed_phases(params.seed);

    let children = child_index(sk);
    let field = SwayField::new(sk);

    for stem in sk.stem_runs() {
        let Some(path) = StemPath::build(sk, &stem, mp, params.seed, &children) else {
            continue;
        };
        if too_thin_for_bark(&path, mp) {
            continue;
        }
        sink.weathering = if path.is_dead { 1.0 } else { 0.0 };
        sink.sway = field.stem(sk, stem[0] as usize);
        emit_stem(&mut sink, &path, mp, phase);
    }

    sink.mesh
}

/// Whether a stem is below the radius worth sweeping. Living twigs are culled against
/// `min_bark_radius` because foliage hides them. Bare dead wood — the dead band, with
/// no foliage near it — answers to `dead_bark_radius` instead, when that is set. Dead
/// twigs inside the living crown are hidden exactly as well as the living ones, so
/// they take the living cutoff.
fn too_thin_for_bark(path: &StemPath, mp: &MeshParams) -> bool {
    bark_cutoff(path.level, path.is_bare, mp)
        .is_some_and(|cutoff| path.radii.iter().copied().fold(0.0f32, f32::max) < cutoff)
}

/// The thinnest a stem may be and still be swept, or `None` for one that always is.
fn bark_cutoff(level: u8, is_bare: bool, mp: &MeshParams) -> Option<f32> {
    // Structure is always swept; only twigs are culled.
    if u32::from(level) < mp.cull_from_level {
        return None;
    }
    Some(if is_bare && mp.dead_bark_radius > 0.0 {
        mp.dead_bark_radius
    } else {
        mp.min_bark_radius
    })
}

/// Whether the stem starting at node `first` will be swept, told from the skeleton
/// alone, so a parent can know before its own path is built which of its branches it
/// has to keep a ring for. The same answer `too_thin_for_bark` gives on the built path:
/// a stem is never thicker anywhere than where it starts, since every node is widened
/// to carry the one after it, and that first radius is the anchor ring's.
fn gets_bark(sk: &Skeleton, first: usize, mp: &MeshParams) -> bool {
    let node = &sk.nodes[first];
    bark_cutoff(node.level, hangs_from_trunk_by_dead_wood(sk, first), mp)
        .is_none_or(|cutoff| node.radius >= cutoff)
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
        sink.push_vertex(path.points[0] + e_r * r, normal, e_a, uv, path.sway_arc[0]);
    }

    let reach = mp.tip_length.max(path.radii[1] * 1.2);
    let tip = path.points[1] + path.dirs[1] * reach;
    let apex = sink.vertex_offset();
    sink.push_vertex(
        tip,
        path.dirs[1],
        ortho_of(path.dirs[1]),
        [0.0, arc],
        path.sway_arc[1] + reach,
    );
    for j in 0..radial {
        sink.mesh
            .indices
            .extend_from_slice(&[base + j, base + j + 1, apex]);
    }
}

/// Sides swept around a stem, from how far the swept polygon departs from the surface
/// it stands for.
///
/// Solving `r * (1 - cos(pi / n)) = tol` would size a circle correctly, but a bole is
/// not a circle: it is fluted, it stands on buttress roots, and it swells where limbs
/// leave it. A side count taken from the radius alone cannot see any of that, and the
/// detail comes out faceted however fine the tolerance is set. So the profile is
/// sampled and the polygon measured against it directly, which sizes the sweep by the
/// shape rather than by its girth.
fn radial_for(path: &StemPath, mp: &MeshParams, phase: (f32, f32)) -> u32 {
    const SAMPLES: usize = 256;
    let tol = mp.silhouette_tolerance.max(1e-5);
    let lo = mp.min_radial.max(3);
    let hi = mp.max_radial.max(lo);

    // The rings worth measuring: the widest few, where the detail is deepest.
    let mut order: Vec<usize> = (0..path.len()).collect();
    order.sort_by(|&a, &b| path.radii[b].total_cmp(&path.radii[a]));
    order.truncate(4);

    let mut needed = lo;
    for &i in &order {
        let profile: Vec<f32> = (0..SAMPLES)
            .map(|k| path.radius_at(i, k as f32 / SAMPLES as f32 * TAU, mp, phase))
            .collect();
        needed = needed.max(sides_for_profile(&profile, tol, lo, hi));
        if needed >= hi {
            break;
        }
    }
    needed
}

/// Smallest side count whose polygon stays within `tol` of the sampled profile.
fn sides_for_profile(profile: &[f32], tol: f32, lo: u32, hi: u32) -> u32 {
    for n in lo..hi {
        if profile_error(profile, n) <= tol {
            return n;
        }
    }
    hi
}

/// How far an `n`-sided sweep of this profile sits from the profile itself.
///
/// The polygon edge between two surface points is a straight chord, whose distance
/// from the centre at angle `a` has a closed form, so the error is read straight off
/// the samples rather than by building the geometry.
fn profile_error(profile: &[f32], n: u32) -> f32 {
    let m = profile.len();
    let at = |a: f32| -> f32 {
        let t = a / TAU * m as f32;
        let i = (t.floor() as usize) % m;
        let f = t - t.floor();
        profile[i] * (1.0 - f) + profile[(i + 1) % m] * f
    };
    let step = TAU / n as f32;
    let mut worst = 0.0f32;
    for j in 0..n {
        let a0 = j as f32 * step;
        let a1 = a0 + step;
        let (r0, r1) = (at(a0), at(a1));
        // Walk the arc this edge spans and compare the surface with the chord.
        for k in 1..8 {
            let a = a0 + step * k as f32 / 8.0;
            let denom = r0 * (a - a0).sin() + r1 * (a1 - a).sin();
            if denom.abs() < 1e-6 {
                continue;
            }
            let chord = r0 * r1 * step.sin() / denom;
            worst = worst.max((at(a) - chord).abs());
        }
    }
    worst
}

/// What one stem costs the mesh.
#[derive(Clone, Copy, Debug)]
pub struct StemCost {
    pub level: u8,
    pub nodes: usize,
    pub rings: usize,
    /// Sides at the widest ring, and at the narrowest.
    pub radial: u32,
    pub radial_min: u32,
    pub triangles: usize,
    pub vertices: usize,
    pub spike: bool,
    /// What the stem would cost if every ring were swept at the side count its own
    /// radius needs, instead of every ring taking the thickest ring's count.
    ///
    /// A strip between a ring of n vertices and one of m costs n + m triangles
    /// whatever n and m are, so a tapering stem can shed sides as it thins without
    /// any of them going to waste on the seam.
    pub adaptive_triangles: usize,
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
        if too_thin_for_bark(&path, mp) {
            continue;
        }
        let counts = ring_sides(&path, mp, phase);
        let mut sink = MeshSink::default();
        emit_stem(&mut sink, &path, mp, phase);
        let sides = |r: f32| -> usize {
            let tol = mp.silhouette_tolerance.max(1e-5);
            let n = if r <= tol {
                3.0
            } else {
                PI / (1.0 - tol / r).clamp(-1.0, 1.0).acos()
            };
            (n.ceil() as i32).clamp(mp.min_radial.max(3) as i32, mp.max_radial.max(3) as i32)
                as usize
        };
        let per_ring: Vec<usize> = path
            .radii
            .iter()
            .map(|r| sides(r * (1.0 + mp.socket_flare)))
            .collect();
        let swept = if path.is_trunk {
            path.len()
        } else {
            path.len() - 1
        };
        let mut adaptive: usize = per_ring[..swept]
            .windows(2)
            .map(|w| w[0] + w[1])
            .sum();
        adaptive += per_ring[swept - 1];
        if path.is_trunk {
            adaptive += per_ring[0];
        }

        out.push(StemCost {
            adaptive_triangles: adaptive,
            level: sk.nodes[stem[0] as usize].level,
            nodes: stem.len(),
            rings: path.len(),
            radial: *counts.iter().max().unwrap_or(&3),
            radial_min: *counts.iter().min().unwrap_or(&3),
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
        if node.broken {
            continue;
        }
        if let Some(parent) = node.parent
            && sk.nodes[parent as usize].stem != node.stem
        {
            children[parent as usize].push(i as u32);
        }
    }
    children
}


/// Sides for every ring of a stem, each sized from that ring's own profile.
///
/// A sweep held at one count the whole way is drawing the thin end of a stem at the
/// resolution its thick end needed. A buttressed bole is the extreme case: seventy-odd
/// sides are right at the foot and absurd fifteen metres up, where the same stem is a
/// few centimetres across.
fn ring_sides(path: &StemPath, mp: &MeshParams, phase: (f32, f32)) -> Vec<u32> {
    const SAMPLES: usize = 256;
    let tol = mp.silhouette_tolerance.max(1e-5);
    let lo = mp.min_radial.max(3);
    let hi = mp.max_radial.max(lo);

    let mut sides: Vec<u32> = (0..path.len())
        .map(|i| {
            let profile: Vec<f32> = (0..SAMPLES)
                .map(|k| path.radius_at(i, k as f32 / SAMPLES as f32 * TAU, mp, phase))
                .collect();
            sides_for_profile(&profile, tol, lo, hi)
        })
        .collect();

    // One ring dipping below its neighbours would pinch the resolution of a stretch
    // that needs it, so a dip is lifted to whichever neighbour is lower.
    let raw = sides.clone();
    for i in 1..raw.len().saturating_sub(1) {
        sides[i] = raw[i].max(raw[i - 1].min(raw[i + 1]));
    }
    sides
}

/// Joins two rings that need not have the same number of vertices.
///
/// A strip between a ring of n and one of m costs exactly n + m triangles whatever n
/// and m are, so a stem can shed sides as it thins without paying anything at the
/// seam. Walks both rims together, always advancing whichever is further behind.
fn stitch(sink: &mut MeshSink, a_base: u32, a_n: u32, b_base: u32, b_n: u32) {
    let (mut ia, mut ib) = (0u32, 0u32);
    while ia < a_n || ib < b_n {
        let a_next = if ia < a_n {
            (ia + 1) as f32 / a_n as f32
        } else {
            f32::INFINITY
        };
        let b_next = if ib < b_n {
            (ib + 1) as f32 / b_n as f32
        } else {
            f32::INFINITY
        };
        if a_next <= b_next {
            sink.mesh
                .indices
                .extend_from_slice(&[a_base + ia, a_base + ia + 1, b_base + ib]);
            ia += 1;
        } else {
            sink.mesh
                .indices
                .extend_from_slice(&[a_base + ia, b_base + ib + 1, b_base + ib]);
            ib += 1;
        }
    }
}

fn emit_stem(sink: &mut MeshSink, path: &StemPath, mp: &MeshParams, phase: (f32, f32)) {
    let last = path.len() - 1;

    // A single-segment branch is drawn as one cone from its anchor ring to a point,
    // rather than a ring pair swept into a tube and then capped with a cone as well.
    // The shape is the same at the scale these appear, for a third of the triangles,
    // and the tree is overwhelmingly made of them.
    if path.is_single_segment() {
        emit_spike(sink, path, mp, phase, radial_for(path, mp, phase));
        return;
    }

    // A twig ends twice over: the last ring repeats the one before it at almost the
    // same radius, and then a cone closes it anyway. Tapering the final stretch
    // straight to the tip is the same silhouette for a third of the triangles, and on
    // something a centimetre across it is arguably the better shape. The trunk keeps
    // its full sweep, since its top is not a twig.
    let swept = if path.is_trunk || path.is_dead {
        path.len()
    } else {
        path.len() - 1
    };
    let sides = ring_sides(path, mp, phase);
    let mut ring_bases: Vec<u32> = Vec::with_capacity(swept);

    for i in 0..swept {
        let radial = sides[i];
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
            sink.push_vertex(path.points[i] + e_r * r, normal, e_a, uv, path.sway_arc[i]);
        }
        ring_bases.push(base);
    }

    for w in 0..ring_bases.len() - 1 {
        stitch(sink, ring_bases[w], sides[w], ring_bases[w + 1], sides[w + 1]);
    }

    if path.is_trunk {
        // Only the trunk meets the ground, so it is the only stem that needs a disc.
        // The rim is duplicated with the cap normal to keep the edge crisp.
        let cap_n = -path.dirs[0];
        let (n0, b0) = path.frames[0];
        let rim_sides = sides[0];
        let rim = sink.vertex_offset();
        for j in 0..=rim_sides {
            let a = j as f32 / rim_sides as f32 * TAU;
            let e_r = n0 * a.cos() + b0 * a.sin();
            let pos = path.points[0] + e_r * path.radius_at(0, a, mp, phase);
            sink.push_vertex(pos, cap_n, ortho_of(cap_n), [0.0, 0.0], path.sway_arc[0]);
        }
        let center = sink.vertex_offset();
        sink.push_vertex(path.points[0], cap_n, ortho_of(cap_n), [0.0, 0.0], path.sway_arc[0]);
        for j in 0..rim_sides {
            sink.mesh
                .indices
                .extend_from_slice(&[center, rim + j + 1, rim + j]);
        }
    }

    // Every stem closes with a cone: from its last ring for the trunk, and from the
    // ring before it for everything else, which is what makes the tip a taper. A stem
    // the tree has lost ends where it snapped instead, so it gets a flat break.
    let end_dir = path.dirs[last];
    let r_end = path.radii[last];
    let reach = if path.is_dead { 0.0 } else { mp.tip_length.max(r_end * 1.2) };
    let tip = path.points[last] + end_dir * reach;
    let apex = sink.vertex_offset();
    sink.push_vertex(
        tip,
        end_dir,
        ortho_of(end_dir),
        [0.0, path.arc[last]],
        path.sway_arc[last] + reach,
    );
    let base = ring_bases[swept - 1];
    for j in 0..sides[swept - 1] {
        sink.mesh
            .indices
            .extend_from_slice(&[base + j, base + j + 1, apex]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, BIRCH_RON, OAK_RON, PINE_RON};

    #[test]
    fn every_branch_leaves_from_a_ring_that_sways_as_its_foot_does() {
        // The wind moves rings, and the renderer joins them with straight spans. A
        // branch whose parent had no ring where it leaves rode the true bend while the
        // parent's surface there cut across the chord, and in a gust the birch's
        // branches came away from their limbs by more than the limbs are thick. So a
        // swept branch needs a ring of its parent at its very foot, reading the same
        // sway the foot does; then whatever the shader does to one it does to the other.
        for (name, src) in crate::species::builtin_presets() {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let children = child_index(&sk);
            let field = crate::wind::SwayField::new(&sk);
            let mut swept: std::collections::HashMap<u32, (StemPath, crate::wind::StemSway)> =
                std::collections::HashMap::new();
            for run in sk.stem_runs() {
                let Some(path) = StemPath::build(&sk, &run, &params.mesh, params.seed, &children)
                else {
                    continue;
                };
                if !too_thin_for_bark(&path, &params.mesh) {
                    let first = run[0] as usize;
                    swept.insert(sk.nodes[first].stem, (path, field.stem(&sk, first)));
                }
            }
            let mut checked = 0;
            for (&stem, (_, sway)) in &swept {
                let first = stem as usize;
                let Some(p) = sk.nodes[first].parent else { continue };
                let parent = &sk.nodes[p as usize];
                // The trunk bends by height alone and has no sway of its own to match.
                if parent.level == 0 {
                    continue;
                }
                let (ppath, psway) = swept
                    .get(&parent.stem)
                    .unwrap_or_else(|| panic!("{name}: a swept branch hangs off unswept wood"));
                let ring = ppath
                    .points
                    .iter()
                    .position(|q| (*q - parent.position).length() < 1e-4)
                    .unwrap_or_else(|| {
                        panic!("{name}: a level {} branch leaves its parent between rings", sk.nodes[first].level)
                    });
                let (foot, there) = (sway.at(0.0), psway.at(ppath.sway_arc[ring]));
                for o in 0..crate::wind::SWAY_ORDERS {
                    assert!(
                        (foot[o][3] - there[o][3]).abs() < 1e-4,
                        "{name}: order {o} swings the foot {} and the ring it leaves {}",
                        foot[o][3],
                        there[o][3]
                    );
                    if foot[o][3] > 0.0 {
                        let pivot = |w: [f32; 4]| Vec3::new(w[0], w[1], w[2]);
                        assert!((pivot(foot[o]) - pivot(there[o])).length() < 1e-4);
                    }
                }
                checked += 1;
            }
            assert!(checked > 50, "{name}: only {checked} junctions");
        }
    }

    #[test]
    fn whether_a_stem_gets_bark_is_known_before_its_path_is_built() {
        // A parent keeps a ring for each branch that will be swept, and has to know
        // which those are before their own paths exist.
        for (name, src) in crate::species::builtin_presets() {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let children = child_index(&sk);
            for run in sk.stem_runs() {
                let Some(path) = StemPath::build(&sk, &run, &params.mesh, params.seed, &children)
                else {
                    continue;
                };
                assert_eq!(
                    gets_bark(&sk, run[0] as usize, &params.mesh),
                    !too_thin_for_bark(&path, &params.mesh),
                    "{name}: the skeleton and the built path disagree about stem {}",
                    run[0]
                );
            }
        }
    }


    #[test]
    fn triangles_wind_counter_clockwise_when_seen_from_outside() {
        // Consistent winding is what lets the renderer cull back faces and trust the
        // normal it was given, instead of flipping normals toward the viewer to hide
        // not knowing which way a triangle faces.
        for src in [OAK_RON, PINE_RON, BIRCH_RON] {
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
            // Enough to mean something, and within what the budgeted presets keep: the
            // birch's whole bark is under a thousand triangles.
            assert!(checked > 500, "{}: only {checked} triangles", params.name);
        }
    }

    #[test]
    fn every_branch_tube_starts_on_its_parent() {
        // The whole tree is one welded surface only because each stem begins with a
        // ring on the parent centreline. Starting at the first grown node instead
        // leaves a segment-length hole at every junction.
        for src in [OAK_RON, PINE_RON, BIRCH_RON] {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let children = child_index(&sk);
            let mut checked = 0;
            for run in sk.stem_runs() {
                let first = run[0] as usize;
                let Some(p) = sk.nodes[first].parent else {
                    continue;
                };
                let path = StemPath::build(&sk, &run, &params.mesh, params.seed, &children).expect("stem builds");
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
        let children = child_index(&sk);
        for run in sk.stem_runs() {
            let first = run[0] as usize;
            let Some(p) = sk.nodes[first].parent else {
                continue;
            };
            let parent_radius = sk.nodes[p as usize].radius;
            let path = StemPath::build(&sk, &run, &params.mesh, params.seed, &children).expect("stem builds");
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
        //
        // The pine is only the fixture here, and its preset trades the swelling away
        // for triangles by thinning its rings coarsely. What is under test is whether
        // the mesher can draw it, so the rings are held at the default the swelling was
        // built against.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.irregularity.ring_tolerance = BarkIrregularity::default().ring_tolerance;
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
        //
        // Held at the default ring spacing for the same reason as the swelling test: the
        // pine preset thins its rings past where the collars land, and it is the mesher,
        // not the preset's budget, under test.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.irregularity.ring_tolerance = BarkIrregularity::default().ring_tolerance;
        let sk = crate::grow(&params);
        let children = child_index(&sk);
        let run = sk
            .stem_runs()
            .into_iter()
            .find(|r| sk.nodes[r[0] as usize].parent.is_none())
            .expect("a trunk");
        let mut path =
            StemPath::build(&sk, &run, &params.mesh, params.seed, &children).expect("builds");
        assert!(
            !path.irregular.collars.is_empty(),
            "the trunk carries branches but recorded no collars"
        );

        // Measured as the difference the collar itself makes, by sampling the same
        // point with and without it. Comparing the facing side against the far side
        // instead reads the flutes and the swells, which wander round the stem on their
        // own account and are deeper than a collar is: that turns a question about one
        // feature into a coin toss about all of them.
        let collars = std::mem::take(&mut path.irregular.collars);
        let mut checked = 0;
        let mut resolved = 0;
        for collar in &collars {
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
            // A collar reaches `radius * 2.6` along the stem, and a branch thin
            // enough that this is narrower than the gap between rings falls between
            // them: no ring ever samples it, so it swells nothing. That is the honest
            // behaviour of a twig's collar, not a failure, so those are not counted.
            let reach = (collar.radius * 2.6).max(0.04);
            if (path.arc[i] - collar.arc).abs() > reach {
                continue;
            }
            let (n, b) = path.frames[i];
            // Recover the angle of the collar in this ring's frame.
            let a_face = collar.dir.dot(b).atan2(collar.dir.dot(n));
            let bare = path.radius_at(i, a_face, &params.mesh, (0.0, 0.0));
            path.irregular.collars.push(Collar {
                arc: collar.arc,
                dir: collar.dir,
                angle: collar.angle,
                radius: collar.radius,
            });
            let swelled = path.radius_at(i, a_face, &params.mesh, (0.0, 0.0));
            path.irregular.collars.clear();
            resolved += 1;
            if swelled > bare {
                checked += 1;
            }
        }
        assert!(
            resolved * 4 > collars.len(),
            "only {resolved} of {} collars landed on a ring at all",
            collars.len()
        );
        assert_eq!(
            checked, resolved,
            "only {checked} of {resolved} collars that land on a ring thickened the              side the branch leaves from"
        );
    }

    #[test]
    fn thin_twigs_can_be_left_unmeshed() {
        // The finest twigs are a third of the bark and two thirds of its vertices, and
        // under foliage nobody can see them. Dropping them has to drop only them.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.min_bark_radius = 0.0;
        // Dead wood answers to a cutoff of its own; this is about the living one, so
        // hold the dead to it too. `dead_twigs_answer_to_their_own_cutoff` covers the
        // other.
        params.mesh.dead_bark_radius = 0.0;
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
    fn a_buttressed_trunk_stands_on_distinct_roots() {
        // The point of the buttress is the hollows between the roots, not the extra
        // width: a cone of the same girth reads as a skirt. So count the ridges.
        let mut params = parse_species(OAK_RON).unwrap();
        params.mesh.root_count = 5;
        params.mesh.root_sharpness = 0.9;
        // Counting roots means counting roots. The finer grooves that ride on them are
        // turned off here and checked separately below, and so is the bark relief,
        // which now reaches the foot too and puts ridges of its own round it.
        params.mesh.root_grooves = 0.0;
        // Splitting narrows the ridges into arms, which is a separate control and
        // would leave the grooves nowhere to sit.
        params.mesh.root_split = 0.0;
        params.mesh.irregularity.flute_depth = 0.0;
        params.mesh.irregularity.swell_depth = 0.0;
        params.mesh.irregularity.burl_density = 0.0;
        let sk = crate::grow(&params);
        let path = trunk_path(&params, &sk);
        let phase = seed_phases(params.seed);

        const N: usize = 360;
        let r: Vec<f32> = (0..N)
            .map(|k| path.radius_at(0, k as f32 / N as f32 * TAU, &params.mesh, phase))
            .collect();
        // A root has to stand proud to count. Between sharp ridges the profile is flat
        // to within a rounding error, and a bare `greater than its neighbours` test
        // reads that noise as an extra root.
        let lo = r.iter().copied().fold(f32::MAX, f32::min);
        let hi = r.iter().copied().fold(0.0f32, f32::max);
        let floor = lo + (hi - lo) * 0.25;
        let peaks = (0..N)
            .filter(|&k| {
                let prev = r[(k + N - 1) % N];
                let next = r[(k + 1) % N];
                r[k] > floor && r[k] > prev && r[k] >= next
            })
            .count();
        assert_eq!(
            peaks, 5,
            "wanted five buttress roots round the foot, found {peaks}"
        );

        // And they have to stand proud of the hollows by a real margin.
        let lo = r.iter().copied().fold(f32::MAX, f32::min);
        let hi = r.iter().copied().fold(0.0f32, f32::max);
        assert!(hi > lo * 1.5, "roots barely stand out: {lo} to {hi}");

        // The buttress is a foot, not a taper: it has to be gone higher up the bole.
        let high = path.len() - 1;
        let top = path.radius_at(high, 0.0, &params.mesh, phase);
        let top_wide = path.radius_at(high, PI, &params.mesh, phase);
        assert!(
            (top - top_wide).abs() < top * 0.5,
            "the bole is still lobed at the top: {top} against {top_wide}"
        );

        // Grooves put finer relief on those roots without adding roots of their own.
        params.mesh.root_grooves = 0.3;
        let grooved: Vec<f32> = (0..N)
            .map(|k| path.radius_at(0, k as f32 / N as f32 * TAU, &params.mesh, phase))
            .collect();
        let g_lo = grooved.iter().copied().fold(f32::MAX, f32::min);
        let g_hi = grooved.iter().copied().fold(0.0f32, f32::max);
        let g_floor = g_lo + (g_hi - g_lo) * 0.05;
        let ripples = (0..N)
            .filter(|&k| {
                let prev = grooved[(k + N - 1) % N];
                let next = grooved[(k + 1) % N];
                grooved[k] > g_floor && grooved[k] > prev && grooved[k] >= next
            })
            .count();
        assert!(
            ripples > peaks * 2,
            "grooves added no relief: {ripples} ripples against {peaks} roots"
        );
    }

    #[test]
    fn a_bole_swells_in_lumps_rather_than_rings() {
        // A swelling that varies only along the stem puts the same bulge the whole way
        // round it, and a bole built from those reads as a stack of tins. The part of
        // the shape that is the same at every angle - the mean radius of a ring - must
        // therefore follow the taper and nothing else, with the swelling living in the
        // variation around the bole instead.
        let mut params = parse_species(OAK_RON).unwrap();
        // Configured for the property rather than borrowed from whatever the species
        // is tuned to today: one long unforked bole with a pronounced swelling on it,
        // so there is plenty to measure however the oak itself is set up.
        params.trunk.length = 14.0;
        params.trunk.split_probability = 0.0;
        params.mesh.flare_height = 0.5;
        params.mesh.irregularity.swell_depth = 0.14;
        params.mesh.irregularity.swell_period = 2.0;
        // Burls, knots and collars are all meant to be local, so they legitimately do
        // move the ring mean where they sit. This is about the swelling, which is the
        // one feature that runs the length of the stem, so they are out of the way.
        params.mesh.irregularity.burl_density = 0.0;
        params.mesh.irregularity.knot_density = 0.0;
        params.mesh.irregularity.collar_depth = 0.0;
        // And the rings at the default spacing: the oak's own are thinned for its
        // triangle budget, far past where there are enough of them to measure.
        params.mesh.silhouette_tolerance = MeshParams::default().silhouette_tolerance;
        params.mesh.irregularity.ring_tolerance = BarkIrregularity::default().ring_tolerance;
        let sk = crate::grow(&params);
        let path = trunk_path(&params, &sk);
        let phase = seed_phases(params.seed);

        const ANGLES: usize = 64;
        let ring_mean = |i: usize| -> f32 {
            (0..ANGLES)
                .map(|k| path.radius_at(i, k as f32 / ANGLES as f32 * TAU, &params.mesh, phase))
                .sum::<f32>()
                / ANGLES as f32
        };
        // Above the buttress, where taper is the only thing left that may change girth.
        let above: Vec<usize> = (0..path.len())
            .filter(|&i| path.points[i].y > params.mesh.flare_height * 1.15)
            .collect();
        assert!(above.len() > 10, "only {} rings above the flare", above.len());

        let mean: Vec<f32> = above.iter().map(|&i| ring_mean(i)).collect();
        let rises = mean.windows(2).filter(|w| w[1] > w[0] * 1.002).count();
        assert!(
            rises <= 1,
            "the bole puts {rises} rings of swelling round itself: {mean:?}"
        );

        // The swelling still has to be there, just round the bole rather than along it.
        let spread = above
            .iter()
            .map(|&i| {
                let r: Vec<f32> = (0..ANGLES)
                    .map(|k| path.radius_at(i, k as f32 / ANGLES as f32 * TAU, &params.mesh, phase))
                    .collect();
                let lo = r.iter().copied().fold(f32::MAX, f32::min);
                let hi = r.iter().copied().fold(0.0f32, f32::max);
                hi / lo
            })
            .fold(0.0f32, f32::max);
        assert!(
            spread > 1.15,
            "the bole came out round at every height: widest section only {spread}"
        );
    }

    #[test]
    fn knots_only_sit_where_a_branch_could_have_been() {
        // A scar can only be where a limb was, so the zone starts a little under the
        // lowest branch the stem still carries. Below that is clean bole — which on a
        // trunk is the stretch at eye level, the part anyone standing by the tree
        // looks at hardest, and the part knots used to land on most jarringly because
        // they were scattered over the whole length.
        let params = parse_species(OAK_RON).unwrap();
        let ir = &params.mesh.irregularity;
        let sk = crate::grow(&params);
        let children = child_index(&sk);

        let mut checked = 0;
        let mut on_trunk = 0;
        for run in sk.stem_runs() {
            let Some(path) = StemPath::build(&sk, &run, &params.mesh, params.seed, &children)
            else {
                continue;
            };
            let knots = &path.irregular.knots;
            if knots.is_empty() {
                continue;
            }
            let lowest = path
                .irregular
                .collars
                .iter()
                .map(|c| c.arc)
                .reduce(f32::min)
                .expect("a stem with knots carries branches");
            let floor = (lowest - ir.knot_reach * path.irregular.radius).max(0.0);
            for k in knots {
                assert!(
                    k.arc >= floor - 1e-4,
                    "a knot at {:.2} m sits {:.2} m below the branched part of its stem",
                    k.arc,
                    floor - k.arc
                );
                // Thin stems have not laid down enough wood to have buried anything.
                assert!(
                    path.irregular.radius >= k.sigma_s * ir.knot_min_stem - 1e-4,
                    "a knot {:.2} m across sits on a stem of radius {:.2} m",
                    k.sigma_s,
                    path.irregular.radius
                );
                checked += 1;
            }
            if path.is_trunk {
                on_trunk += knots.len();
            }
        }
        assert!(checked > 5, "only {checked} knots placed to check");
        assert!(on_trunk > 0, "the trunk carries no knots at all");
    }

    #[test]
    fn a_knot_is_a_scar_rather_than_a_bore_hole() {
        // The dish has to stay shallow against the stem carrying it, and the rim has
        // to stand proud of the dish. Weighted the other way up, a knot sank the
        // centre of the trunk by half its radius, which reads as damage rather than as
        // something the tree healed over decades ago.
        let params = parse_species(OAK_RON).unwrap();
        let sk = crate::grow(&params);
        let children = child_index(&sk);

        let mut worst_dish: f32 = 0.0;
        let mut checked = 0;
        for run in sk.stem_runs() {
            let Some(mut path) =
                StemPath::build(&sk, &run, &params.mesh, params.seed, &children)
            else {
                continue;
            };
            // Measured as the difference the knot itself makes, with the flutes, the
            // swells and the collars held still by sampling the same point twice.
            // Differencing against a nearby point instead reads the flutes, which run
            // deeper than a knot does.
            let knots = std::mem::take(&mut path.irregular.knots);
            let radius = path.irregular.radius;
            for k in &knots {
                let at = |ir: &Irregular, a: f32| {
                    ir.shape(k.arc, a, Vec3::new(a.cos(), 0.0, a.sin()), radius)
                };
                let bare_centre = at(&path.irregular, k.angle);
                let bare_rim = at(&path.irregular, k.angle + k.sigma_a);
                path.irregular.knots.push(Knot {
                    arc: k.arc,
                    angle: k.angle,
                    sigma_s: k.sigma_s,
                    sigma_a: k.sigma_a,
                    amp: k.amp,
                });
                let dish = bare_centre - at(&path.irregular, k.angle);
                let rim = at(&path.irregular, k.angle + k.sigma_a) - bare_rim;
                path.irregular.knots.clear();

                assert!(
                    dish > 0.0 && rim > 0.0,
                    "a knot with a dish of {dish:.3} and a rim of {rim:.3} is neither"
                );
                assert!(
                    rim > dish * 0.9,
                    "the rim stands {rim:.3} against a dish of {dish:.3}: that is a hole"
                );
                worst_dish = worst_dish.max(dish);
                checked += 1;
            }
        }
        assert!(checked > 5, "only {checked} knots to measure");
        assert!(
            worst_dish < 0.12,
            "the deepest knot takes {worst_dish:.2} of its stem's radius out of it"
        );
    }

    #[test]
    fn knots_keep_clear_of_branches_and_of_each_other() {
        // A knot is a branch the tree lost and grew over, so one cannot sit on a limb
        // it still has; and two on the same spot read as damage rather than as history.
        let mut params = parse_species(OAK_RON).unwrap();
        // Enough of them that placement has to work rather than get lucky.
        params.mesh.irregularity.knot_density = 2.0;
        let sk = crate::grow(&params);
        let children = child_index(&sk);

        let mut checked = 0;
        for run in sk.stem_runs() {
            let Some(path) = StemPath::build(&sk, &run, &params.mesh, params.seed, &children)
            else {
                continue;
            };
            let ir = &path.irregular;
            if ir.knots.len() < 2 {
                continue;
            }
            let apart = |a_arc: f32, a_ang: f32, a_s: f32, a_a: f32,
                         b_arc: f32, b_ang: f32, b_s: f32, b_a: f32| {
                let ds = (a_arc - b_arc) / (a_s + b_s).max(1e-4);
                let da = wrap_pi(a_ang - b_ang) / (a_a + b_a).max(1e-4);
                ds * ds + da * da >= 1.0
            };
            for (i, k) in ir.knots.iter().enumerate() {
                for other in &ir.knots[i + 1..] {
                    assert!(
                        apart(k.arc, k.angle, k.sigma_s, k.sigma_a,
                              other.arc, other.angle, other.sigma_s, other.sigma_a),
                        "two knots overlap at {} and {} along the stem",
                        k.arc,
                        other.arc
                    );
                }
                for c in &ir.collars {
                    assert!(
                        apart(k.arc, k.angle, k.sigma_s, k.sigma_a,
                              c.arc, c.angle, (c.radius * 2.6).max(0.04), 0.9),
                        "a knot sits on the branch leaving at {} along the stem",
                        c.arc
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 20, "only {checked} knots placed to check");
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
    fn every_vertex_knows_what_it_hangs_off() {
        let params = parse_species(OAK_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        assert_eq!(mesh.sway.len(), mesh.vertex_count());
        // The foot of the trunk is carried by nothing; the renderer bends the trunk
        // by height alone.
        for (p, w) in mesh.positions.iter().zip(&mesh.sway) {
            if p[1] < 0.5 {
                assert!(w.iter().all(|o| o[3] == 0.0), "trunk foot at {p:?} sways as a branch");
            }
        }
        // Each order has wood in it somewhere.
        for order in 0..crate::wind::SWAY_ORDERS {
            assert!(
                mesh.sway.iter().any(|w| w[order][3] > 0.0),
                "nothing in the oak bends as order {order}"
            );
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
    #[test]
    fn dead_twigs_answer_to_their_own_cutoff() {
        // Living twigs are culled from the bark because foliage hides them. Dead wood
        // carries none, so its thin twigs are on show, and a species can keep them
        // while still dropping the living ones.
        let mut params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        assert!(sk.nodes.iter().any(|n| n.dead), "this preset should carry dead wood");

        params.mesh.min_bark_radius = 0.02;
        params.mesh.dead_bark_radius = 0.0;
        let follows = build_mesh(&sk, &params);
        params.mesh.dead_bark_radius = 0.001;
        let keeps = build_mesh(&sk, &params);

        let dead_verts = |m: &Mesh| m.weathering.iter().filter(|&&w| w > 0.5).count();
        assert!(
            dead_verts(&keeps) > dead_verts(&follows) * 2,
            "a lower dead cutoff kept {} dead vertices against {}",
            dead_verts(&keeps),
            dead_verts(&follows)
        );
        let live_verts = |m: &Mesh| m.weathering.iter().filter(|&&w| w < 0.5).count();
        assert_eq!(
            live_verts(&keeps),
            live_verts(&follows),
            "the dead cutoff changed how much living wood was meshed"
        );
    }

    #[test]
    fn dead_wood_is_marked_for_weathering_and_living_wood_is_not() {
        // The renderer bleaches dead wood by a per-vertex flag, so the flag has to be
        // on every vertex of a dead stem, off every living one, and one per vertex.
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        assert_eq!(mesh.weathering.len(), mesh.positions.len());
        let dead = mesh.weathering.iter().filter(|&&w| w == 1.0).count();
        let alive = mesh.weathering.iter().filter(|&&w| w == 0.0).count();
        assert_eq!(dead + alive, mesh.weathering.len(), "weathering is 0 or 1 per stem");
        assert!(dead > 0 && alive > 0, "{dead} dead and {alive} living vertices");
        // The trunk is alive, and it is the stem that starts at the root.
        assert_eq!(mesh.weathering[0], 0.0, "the first vertex is the trunk's");
    }
    #[test]
    fn dead_twigs_inside_the_crown_are_culled_like_living_ones() {
        // The fine dead-wood cutoff is for the dead band, where nothing hides the wood.
        // A dead twig on a living limb sits inside the foliage and is as invisible as a
        // living one, so it answers to the living cutoff.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.cull_from_level = 0;
        params.mesh.min_bark_radius = 0.05;
        params.mesh.dead_bark_radius = 0.001;
        let sk = crate::grow(&params);
        let hidden: Vec<usize> = sk
            .stem_runs()
            .iter()
            .map(|run| run[0] as usize)
            .filter(|&i| sk.nodes[i].dead && !hangs_from_trunk_by_dead_wood(&sk, i))
            .collect();
        assert!(hidden.len() > 20, "only {} dead stems inside the crown", hidden.len());

        let children = child_index(&sk);
        let meshed = |first: usize| {
            let run: Vec<u32> = sk
                .stem_runs()
                .into_iter()
                .find(|r| r[0] as usize == first)
                .unwrap();
            StemPath::build(&sk, &run, &params.mesh, params.seed, &children)
                .is_some_and(|path| !too_thin_for_bark(&path, &params.mesh))
        };
        let swept = hidden.iter().filter(|&&i| meshed(i)).count();
        assert!(
            swept * 10 < hidden.len(),
            "{swept} of {} dead twigs inside the crown were swept below the living cutoff",
            hidden.len()
        );
        // And the bare ones still are: the dead band keeps its fine twigs.
        let bare_swept = sk
            .stem_runs()
            .iter()
            .map(|run| run[0] as usize)
            .filter(|&i| sk.nodes[i].level >= 2 && hangs_from_trunk_by_dead_wood(&sk, i))
            .filter(|&i| meshed(i))
            .count();
        assert!(bare_swept > 20, "only {bare_swept} bare dead twigs were swept");
    }

    #[test]
    fn culling_can_be_kept_off_the_limbs() {
        // The bark cutoffs are for twigs. A limb thin enough to fall under one is still
        // structure, and dropping it leaves its foliage floating in the gaps of a crown.
        let mut params = parse_species(PINE_RON).unwrap();
        params.mesh.min_bark_radius = 0.2;
        params.mesh.dead_bark_radius = 0.0;
        let sk = crate::grow(&params);
        let limbs = sk
            .stem_runs()
            .iter()
            .filter(|run| sk.nodes[run[0] as usize].level == 1)
            .count();
        let swept_limbs = |params: &SpeciesParams| {
            stem_costs(&sk, params).iter().filter(|c| c.level == 1).count()
        };

        params.mesh.cull_from_level = 0;
        let culled = swept_limbs(&params);
        params.mesh.cull_from_level = 2;
        let kept = swept_limbs(&params);
        assert!(culled < limbs / 2, "a 20 cm cutoff should drop most limbs, kept {culled} of {limbs}");
        assert_eq!(kept, limbs, "with culling from level 2 every limb is swept");
        // Twigs are still culled.
        let twigs = stem_costs(&sk, &params).iter().filter(|c| c.level >= 2).count();
        assert_eq!(twigs, 0, "{twigs} twigs survived a 20 cm cutoff");
    }
    #[test]
    fn thinning_does_not_put_elbows_in_a_smooth_limb() {
        // The ring budget is a distance, and a limb curving gently for metres stays
        // within it however few rings are kept — so a coarse budget once swept a smooth
        // hundred-degree arc as two straight spans meeting at a sixty-degree elbow.
        // Measured against the dense path each stem was thinned from, so this holds
        // whatever the skeleton did: a corner in the mesh may not be sharper than the
        // span limit plus whatever the stem itself turned there.
        for src in [PINE_RON, BIRCH_RON, OAK_RON] {
            let params = parse_species(src).unwrap();
            let sk = crate::grow(&params);
            let children = child_index(&sk);
            let corner = |pts: &[Vec3]| {
                pts.windows(3)
                    .map(|w| (w[1] - w[0]).angle_between(w[2] - w[1]))
                    .fold(0.0f32, f32::max)
            };
            let (mut checked, mut worst) = (0, 0.0f32);
            for stem in sk.stem_runs() {
                if sk.nodes[stem[0] as usize].level != 1 {
                    continue;
                }
                let Some(dense) = StemPath::build_dense(&sk, &stem, &params.mesh, params.seed, &children) else { continue };
                if too_thin_for_bark(&dense, &params.mesh) || dense.len() < 6 {
                    continue;
                }
                let skeleton = corner(&dense.points);
                let thinned = dense.thinned(&params.mesh, seed_phases(params.seed));
                let mesh = corner(&thinned.points);
                checked += 1;
                worst = worst.max(mesh);
                // A fixed bound rather than `MAX_SPAN_TURN`, which would move with it:
                // twenty-five degrees over whatever the stem itself turns there.
                assert!(
                    mesh <= 0.436 + skeleton,
                    "{}: thinning put a {:.0} degree corner in a limb whose own sharpest is {:.0}",
                    params.name,
                    mesh.to_degrees(),
                    skeleton.to_degrees()
                );
            }
            assert!(checked > 30, "{}: only {checked} limbs measured", params.name);
            assert!(worst > 0.05, "{}: no limb bends at all, so nothing was tested", params.name);
        }
    }
}
