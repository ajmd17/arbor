//! Rocks: boulders, slabs and stones with their own baked maps, for an engine to scatter
//! and half bury.
//!
//! A rock starts as a sphere cut by a handful of planes, which is what gives boulders
//! their broad faces and blunt edges, with the edges softened by how sharply the planes
//! meet. Noise then swells and dents it, raises ridges, and steps it where it has
//! fractured, and the bottom is cut flat where it sits in the ground. The surface is a
//! function of direction from the middle, so the mesh and the maps are sampled from the
//! same shape: the mesh at each LOD's resolution, the maps at far finer, with cracks and
//! grain only the maps carry.
//!
//! The mesh is a cube pushed out to the surface, six square faces of quads, and the maps
//! are laid out as those six faces in a three by two sheet, so every texel is a known
//! direction and there is nothing to unwrap. Colour, moss, occlusion and roughness are
//! all painted from the surface itself: moss where it faces up and in the hollows, grain
//! and lichen on the stone, darkening in the cracks.
//!
//! A preset is read like a species: any number in `shape`, `surface` and `moss` may be a
//! `(lo, hi)` range, landed once per seed.

use glam::Vec3;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::cluster::Bitmap;
use crate::ranged::{key, Ranged, Scalar};
use crate::seed::PortableRng;

pub const BOULDER_RON: &str = include_str!("../../../assets/rocks/boulder.ron");
pub const MOSSY_BOULDER_RON: &str = include_str!("../../../assets/rocks/mossy_boulder.ron");
pub const SLAB_RON: &str = include_str!("../../../assets/rocks/slab.ron");
pub const STONE_RON: &str = include_str!("../../../assets/rocks/stone.ron");

/// Where presets saved from the viewer go, and are found by name from the CLI.
pub const CUSTOM_ROCK_DIR: &str = "assets/rocks/custom";

pub fn builtin_rock_presets() -> Vec<(&'static str, &'static str)> {
    vec![
        ("boulder", BOULDER_RON),
        ("mossy_boulder", MOSSY_BOULDER_RON),
        ("slab", SLAB_RON),
        ("stone", STONE_RON),
    ]
}

/// A rock preset as its file describes it, ranges and all.
pub type RockTemplate = RockParams<Ranged>;

pub fn parse_rock_template(ron_src: &str) -> Result<RockTemplate, String> {
    ron::from_str(ron_src).map_err(|e| e.to_string())
}

/// Most planes a rock may be cut by, finest mesh and texture a preset may ask for, and
/// most LODs.
const MAX_FACETS: u32 = 32;
const MAX_MESH_RESOLUTION: u32 = 128;
const MAX_TEXTURE_RESOLUTION: u32 = 2048;
const MAX_LODS: usize = 4;
/// Texels left round each face of the sheet, filled by carrying the face on past its
/// edge, so filtering never pulls in the next face.
const TEXTURE_PAD: f32 = 8.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockParams::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockParams<V = f32> {
    pub name: String,
    pub seed: u64,
    pub shape: RockShape<V>,
    pub surface: RockSurface<V>,
    pub moss: RockMoss<V>,
    pub colour: RockColour,
    /// Finest first.
    pub lod: Vec<RockLod>,
    pub texture: RockTexture,
}

impl Default for RockParams {
    fn default() -> Self {
        Self {
            name: "rock".to_string(),
            seed: 1,
            shape: RockShape::default(),
            surface: RockSurface::default(),
            moss: RockMoss::default(),
            colour: RockColour::default(),
            lod: vec![
                RockLod { resolution: 40, screen_size: 0.0 },
                RockLod { resolution: 20, screen_size: 0.15 },
                RockLod { resolution: 10, screen_size: 0.05 },
            ],
            texture: RockTexture::default(),
        }
    }
}

impl<V: Scalar> RockParams<V> {
    pub fn defaults() -> Self {
        RockParams::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> RockParams<W> {
        RockParams {
            name: self.name.clone(),
            seed: self.seed,
            shape: self.shape.map(&key(at, "shape"), f),
            surface: self.surface.map(&key(at, "surface"), f),
            moss: self.moss.map(&key(at, "moss"), f),
            colour: self.colour.clone(),
            lod: self.lod.clone(),
            texture: self.texture.clone(),
        }
    }
}

impl RockTemplate {
    /// The one rock this preset makes at its seed.
    pub fn instance(&self) -> RockParams {
        let seed = self.seed;
        self.map("", &mut |key, v| v.land(seed, key))
    }

    /// Every number given as a range, with where this preset's seed lands in it.
    pub fn landings(&self) -> Vec<(String, Ranged, f32)> {
        let seed = self.seed;
        let mut found = Vec::new();
        self.map("", &mut |key, v| {
            if v.is_range() {
                found.push((key.to_string(), v, v.land(seed, key)));
            }
        });
        found
    }
}

impl RockParams {
    pub fn template(&self) -> RockTemplate {
        self.map("", &mut |_, v| Ranged::Fixed(v))
    }
}

/// The rock's form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockShape::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockShape<V = f32> {
    /// Metres across, high and deep, before noise.
    pub width: V,
    pub height: V,
    pub depth: V,
    /// Planes the rock is cut by, and how deep they cut, as a share of its radius: the
    /// broad faces of a boulder.
    pub facets: V,
    pub facet_depth: V,
    /// How sharply the faces meet: low rounds every edge off like a river stone, high
    /// leaves them crisp as fresh rubble.
    pub sharpness: V,
    /// Swelling and denting of the whole, as a share of the radius, and how many swells
    /// fit round it.
    pub bulge: V,
    pub bulge_scale: V,
    /// Ridges and grooves over the faces, as a share of the radius.
    pub ridges: V,
    /// Stepped ledges where the stone has split along its bedding, as a share of the
    /// radius.
    pub fracture: V,
    /// Share of the height cut flat at the bottom, where it sits on the ground.
    pub flat_bottom: V,
    /// Share of the height below the origin, so a rock placed on the ground is already
    /// sunk into it rather than balanced on its flat.
    pub bury: V,
}

impl Default for RockShape {
    fn default() -> Self {
        Self {
            width: 2.0,
            height: 1.3,
            depth: 1.6,
            facets: 11.0,
            facet_depth: 0.3,
            sharpness: 24.0,
            bulge: 0.07,
            bulge_scale: 1.3,
            ridges: 0.035,
            fracture: 0.02,
            flat_bottom: 0.3,
            bury: 0.12,
        }
    }
}

impl<V: Scalar> RockShape<V> {
    pub fn defaults() -> Self {
        RockShape::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> RockShape<W> {
        RockShape {
            width: f(&key(at, "width"), self.width),
            height: f(&key(at, "height"), self.height),
            depth: f(&key(at, "depth"), self.depth),
            facets: f(&key(at, "facets"), self.facets),
            facet_depth: f(&key(at, "facet_depth"), self.facet_depth),
            sharpness: f(&key(at, "sharpness"), self.sharpness),
            bulge: f(&key(at, "bulge"), self.bulge),
            bulge_scale: f(&key(at, "bulge_scale"), self.bulge_scale),
            ridges: f(&key(at, "ridges"), self.ridges),
            fracture: f(&key(at, "fracture"), self.fracture),
            flat_bottom: f(&key(at, "flat_bottom"), self.flat_bottom),
            bury: f(&key(at, "bury"), self.bury),
        }
    }
}

/// Detail only the maps carry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockSurface::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockSurface<V = f32> {
    /// Fine unevenness of the stone, as a share of the radius.
    pub grain: V,
    /// Depth of the cracks, as a share of the radius, and the share of the rock they
    /// run over.
    pub cracks: V,
    pub crack_spread: V,
    /// Fine banding across the bedding, as a share of the radius.
    pub strata: V,
    /// How strongly hollows and cracks are darkened and occluded.
    pub cavity: V,
}

impl Default for RockSurface {
    fn default() -> Self {
        Self { grain: 0.01, cracks: 0.014, crack_spread: 0.5, strata: 0.004, cavity: 1.0 }
    }
}

impl<V: Scalar> RockSurface<V> {
    pub fn defaults() -> Self {
        RockSurface::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> RockSurface<W> {
        RockSurface {
            grain: f(&key(at, "grain"), self.grain),
            cracks: f(&key(at, "cracks"), self.cracks),
            crack_spread: f(&key(at, "crack_spread"), self.crack_spread),
            strata: f(&key(at, "strata"), self.strata),
            cavity: f(&key(at, "cavity"), self.cavity),
        }
    }
}

/// Moss over the stone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockMoss::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockMoss<V = f32> {
    /// How much of the rock is mossed over, from none at -1 through tops only at 0 to
    /// nearly all of it at 1.
    pub amount: V,
    /// How far moss keeps to what faces up, against what is low on the rock and near
    /// the ground, where it creeps up from.
    pub upward: V,
    /// Patches across the rock: how broken up the moss is, and how soft its edge.
    pub patchiness: V,
    pub softness: V,
    /// How far moss gathers in hollows and cracks.
    pub crevices: V,
    /// Share of the moss that has dried to a paler, browner olive.
    pub dry: V,
}

impl Default for RockMoss {
    fn default() -> Self {
        Self { amount: 0.1, upward: 1.1, patchiness: 0.8, softness: 0.22, crevices: 0.3, dry: 0.3 }
    }
}

impl<V: Scalar> RockMoss<V> {
    pub fn defaults() -> Self {
        RockMoss::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> RockMoss<W> {
        RockMoss {
            amount: f(&key(at, "amount"), self.amount),
            upward: f(&key(at, "upward"), self.upward),
            patchiness: f(&key(at, "patchiness"), self.patchiness),
            softness: f(&key(at, "softness"), self.softness),
            crevices: f(&key(at, "crevices"), self.crevices),
            dry: f(&key(at, "dry"), self.dry),
        }
    }
}

/// The colours of stone and moss, sRGB, and their variation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RockColour {
    /// Cool and warm stone, blended in broad patches.
    pub stone: [f32; 3],
    pub warm: [f32; 3],
    /// Spread of brightness across the rock, broad and fine.
    pub tone: f32,
    pub grain: f32,
    /// Dark and pale mineral specks, as a share of the surface.
    pub speckle: f32,
    /// Crusts of lichen, as a share of the stone, and their colour.
    pub lichen: f32,
    pub lichen_color: [f32; 3],
    /// Darker streaks run down the faces by water.
    pub streaks: f32,
    pub moss: [f32; 3],
    /// Moss catching the light at its tips, and moss dried out.
    pub moss_tip: [f32; 3],
    pub moss_dry: [f32; 3],
    pub roughness: f32,
    pub moss_roughness: f32,
}

impl Default for RockColour {
    fn default() -> Self {
        Self {
            stone: [0.47, 0.47, 0.46],
            warm: [0.50, 0.47, 0.42],
            tone: 0.45,
            grain: 0.22,
            speckle: 0.5,
            lichen: 0.35,
            lichen_color: [0.68, 0.69, 0.62],
            streaks: 0.25,
            moss: [0.30, 0.37, 0.17],
            moss_tip: [0.46, 0.50, 0.20],
            moss_dry: [0.49, 0.47, 0.30],
            roughness: 0.78,
            moss_roughness: 0.97,
        }
    }
}

/// One level of detail: quads along each edge of each of the cube's six faces, and the
/// screen size it takes over at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RockLod {
    pub resolution: u32,
    pub screen_size: f32,
}

/// Texels along each edge of each face in the sheet. The sheet is three faces by two.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RockTexture {
    pub resolution: u32,
}

impl Default for RockTexture {
    fn default() -> Self {
        Self { resolution: 512 }
    }
}

impl RockTexture {
    /// Texels per face edge, and the sheet's width and height.
    pub fn size(&self) -> (u32, u32, u32) {
        let t = self.resolution.clamp(32, MAX_TEXTURE_RESOLUTION);
        (t, 3 * t, 2 * t)
    }
}

// ---------------------------------------------------------------------------------
// The surface

/// Gradient noise in three dimensions, seeded.
struct Noise {
    perm: [u8; 512],
    grads: [Vec3; 256],
}

impl Noise {
    fn new(rng: &mut PortableRng) -> Self {
        let mut p: Vec<u8> = (0..=255).collect();
        for i in (1..256).rev() {
            let j = rng.random_range(0..=i);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        let grads = std::array::from_fn(|_| {
            let v = Vec3::new(
                rng.random_range(-1.0f32..1.0),
                rng.random_range(-1.0f32..1.0),
                rng.random_range(-1.0f32..1.0),
            );
            v.normalize_or(Vec3::X)
        });
        Self { perm, grads }
    }

    /// About -1 to 1.
    fn at(&self, p: Vec3) -> f32 {
        let f = p.floor();
        let (xi, yi, zi) = (f.x as i32 & 255, f.y as i32 & 255, f.z as i32 & 255);
        let d = p - f;
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (u, v, w) = (fade(d.x), fade(d.y), fade(d.z));
        let perm = &self.perm;
        let g = |ix: i32, iy: i32, iz: i32, o: Vec3| {
            let h = perm[perm[perm[ix as usize] as usize + iy as usize] as usize + iz as usize];
            self.grads[h as usize].dot(o)
        };
        let (x1, y1, z1) = (xi + 1, yi + 1, zi + 1);
        let n000 = g(xi, yi, zi, d);
        let n100 = g(x1, yi, zi, d - Vec3::X);
        let n010 = g(xi, y1, zi, d - Vec3::Y);
        let n110 = g(x1, y1, zi, d - Vec3::new(1.0, 1.0, 0.0));
        let n001 = g(xi, yi, z1, d - Vec3::Z);
        let n101 = g(x1, yi, z1, d - Vec3::new(1.0, 0.0, 1.0));
        let n011 = g(xi, y1, z1, d - Vec3::new(0.0, 1.0, 1.0));
        let n111 = g(x1, y1, z1, d - Vec3::ONE);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x00 = lerp(n000, n100, u);
        let x10 = lerp(n010, n110, u);
        let x01 = lerp(n001, n101, u);
        let x11 = lerp(n011, n111, u);
        lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 1.4
    }

    fn fbm(&self, p: Vec3, octaves: u32) -> f32 {
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        for o in 0..octaves {
            let shift = Vec3::new(17.3, 31.7, 5.1) * o as f32;
            sum += self.at(p * freq + shift) * amp;
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }

    /// Creased noise, 0 to 1, peaking along its zero lines: ridges.
    fn ridged(&self, p: Vec3, octaves: u32) -> f32 {
        let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
        for o in 0..octaves {
            let shift = Vec3::new(17.3, 31.7, 5.1) * o as f32;
            let n = 1.0 - self.at(p * freq + shift).abs();
            sum += n * n * amp;
            norm += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        sum / norm
    }
}

/// A rock's surface, as a radius in every direction.
struct Surface {
    planes: Vec<(Vec3, f32)>,
    sharpness: f32,
    half: Vec3,
    shape: RockShape,
    surface: RockSurface,
    noise: Noise,
    noise2: Noise,
    /// Which way the stone's bedding runs: its ledges are steps across it.
    bedding: Vec3,
    /// Step across a face, in its -1 to 1 coordinates, over which the mesh's own normals
    /// and frames are found: one quad of LOD0. The maps' normals are told against that
    /// frame, so they have to agree on how smooth it is.
    step: f32,
}

impl Surface {
    fn new(p: &RockParams) -> Self {
        let mut rng = PortableRng::seed_from_u64(p.seed ^ 0x0B0C_4000_5EED_0001);
        let s = &p.shape;
        let facets = (s.facets.round().max(0.0) as u32).min(MAX_FACETS);
        let depth = s.facet_depth.clamp(0.0, 0.8);
        let planes = (0..facets)
            .map(|_| {
                // Squashed toward level: a boulder's big faces are its sides and top.
                let n = Vec3::new(
                    rng.random_range(-1.0f32..1.0),
                    rng.random_range(-1.0f32..1.0) * 0.7,
                    rng.random_range(-1.0f32..1.0),
                )
                .normalize_or(Vec3::Y);
                (n, rng.random_range(1.0 - depth..=1.0 - 0.25 * depth))
            })
            .collect();
        let noise = Noise::new(&mut rng);
        let noise2 = Noise::new(&mut rng);
        // Near level, as sediment is laid down, tipped a little as it has since been.
        let bedding = Vec3::new(rng.random_range(-0.35f32..0.35), 1.0, rng.random_range(-0.35f32..0.35)).normalize();
        let finest = p.lod.first().map_or(40, |l| l.resolution).clamp(2, MAX_MESH_RESOLUTION);
        Self {
            planes,
            sharpness: s.sharpness.clamp(2.0, 80.0),
            half: Vec3::new(s.width, s.height, s.depth).max(Vec3::splat(0.02)) * 0.5,
            shape: s.clone(),
            surface: p.surface.clone(),
            noise,
            noise2,
            bedding,
            step: 2.0 / finest as f32,
        }
    }

    /// Distance out in unit direction `d`, before the rock is scaled to its size. `hi`
    /// adds the detail only the maps carry.
    fn radius(&self, d: Vec3, hi: bool) -> f32 {
        // The cut sphere, its edges softened: a smooth minimum of the sphere and the
        // distance to each plane along `d`.
        let k = self.sharpness;
        let mut dists = [1.0f32; MAX_FACETS as usize + 1];
        let mut n = 1;
        for &(normal, offset) in &self.planes {
            let c = d.dot(normal);
            if c > 1e-3 {
                dists[n] = offset / c;
                n += 1;
            }
        }
        let lo = dists[..n].iter().copied().fold(f32::MAX, f32::min);
        let sum: f32 = dists[..n].iter().map(|&r| (-(r - lo) * k).exp()).sum();
        let mut r = lo - sum.ln() / k;

        let s = &self.shape;
        r *= 1.0 + s.bulge * self.noise.fbm(d * s.bulge_scale.max(0.1), 4);
        r += s.ridges * (self.noise2.ridged(d * 2.2, 3) - 0.5);
        // Ledges: height across the bedding, wavered a little, terraced into flat treads
        // with steep risers between, kept continuous so the surface does not tear where
        // one tread meets the next.
        let x = d.dot(self.bedding) * 4.0 + 0.6 * self.noise2.fbm(d * 1.5 + Vec3::splat(4.0), 2);
        let stair = x.floor() + smoothstep(0.6, 1.0, x - x.floor());
        r += s.fracture * (stair - x);
        if hi {
            let f = &self.surface;
            r += f.grain * self.noise.fbm(d * 16.0 + Vec3::new(3.0, 0.0, 0.0), 4);
            let spread = f.crack_spread.clamp(0.0, 1.0);
            let mask = smoothstep(0.5 - spread * 0.35, 0.8 - spread * 0.35, self.noise.fbm(d * 1.7 + Vec3::new(21.0, 0.0, 0.0), 2) * 0.5 + 0.5);
            let crack = 1.0 - self.noise2.at(Vec3::new(d.x * 4.0 + 9.0, d.y * 6.0, d.z * 4.0)).abs();
            r -= f.cracks * crack.powi(60) * mask;
            let band = (d.y * 22.0 + 1.5 * self.noise.at(d * 3.0)).sin() * 0.5 + 0.5;
            r -= f.strata * band.powi(6);
        }
        r
    }

    /// The point on the surface in unit direction `d`, in metres, before it is lifted
    /// onto its origin.
    fn point(&self, d: Vec3, hi: bool) -> Vec3 {
        let mut p = d * self.radius(d, hi) * self.half;
        let b = self.shape.flat_bottom.clamp(0.0, 0.9) * self.half.y;
        if p.y < -b {
            p.y = -b + (p.y + b) * 0.12;
        }
        p
    }

    /// How far a point from `point` is moved to stand on the rock's origin: its flat
    /// bottom on the ground, sunk by `bury`.
    fn lift(&self) -> f32 {
        let flat = self.shape.flat_bottom.clamp(0.0, 0.9);
        let height = self.half.y * (1.0 + flat);
        flat * self.half.y - self.shape.bury.clamp(0.0, 0.9) * height
    }

    /// The point and outward normal in direction `d`, the normal found by stepping `e`
    /// round the direction either way, so it agrees across the cube's seams.
    fn point_normal(&self, d: Vec3, hi: bool, e: f32) -> (Vec3, Vec3) {
        let a = if d.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let t1 = d.cross(a).normalize();
        let t2 = d.cross(t1);
        let p0 = self.point(d, hi);
        let p1 = self.point((d + t1 * e).normalize(), hi);
        let p2 = self.point((d + t2 * e).normalize(), hi);
        let mut n = (p1 - p0).cross(p2 - p0).normalize_or(d);
        if n.dot(p0) < 0.0 {
            n = -n;
        }
        (p0, n)
    }
}

/// The cube's faces: the axis each is centred on, and the directions its u and v run.
/// Each is laid so that u, v and the axis make it face outward seen from outside.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
];

/// The direction at `(s, t)` on face `f`, both from -1 to 1, spread by tangent so the
/// quads are near even in size over the sphere.
fn face_dir(f: usize, s: f32, t: f32) -> Vec3 {
    let (m, u, v) = FACES[f];
    let q = std::f32::consts::FRAC_PI_4;
    (Vec3::from(m) + Vec3::from(u) * (s * q).tan() + Vec3::from(v) * (t * q).tan()).normalize()
}

/// Where `(s, t)` on face `f` sits in the sheet, in texture coordinates, for a sheet
/// with `t` texels per face edge.
fn face_uv(f: usize, s: f32, t: f32, texels: u32) -> [f32; 2] {
    let tf = texels as f32;
    let (col, row) = ((f % 3) as f32, (f / 3) as f32);
    let inner = tf - 2.0 * TEXTURE_PAD;
    [
        (col * tf + TEXTURE_PAD + (s + 1.0) * 0.5 * inner) / (3.0 * tf),
        (row * tf + TEXTURE_PAD + (t + 1.0) * 0.5 * inner) / (2.0 * tf),
    ]
}

/// The surface's frame at `(s, t)` on face `f`: the smooth normal the mesh carries, the
/// tangent along the texture's u, and the handedness that makes the bitangent run up the
/// texture, as the maps are painted.
fn frame(surface: &Surface, f: usize, s: f32, t: f32) -> (Vec3, Vec3, Vec3, f32) {
    // Centred differences a quad wide: the frame the mesh interpolates across its quads.
    let h = surface.step * 0.5;
    let (p, n) = surface.point_normal(face_dir(f, s, t), false, surface.step * std::f32::consts::FRAC_PI_4);
    let ds = surface.point(face_dir(f, s + h, t), false) - surface.point(face_dir(f, s - h, t), false);
    let dt = surface.point(face_dir(f, s, t + h), false) - surface.point(face_dir(f, s, t - h), false);
    let tangent = (ds - n * ds.dot(n)).normalize_or(Vec3::X);
    // Up the texture is toward smaller v, which is smaller t.
    let w = if n.cross(tangent).dot(-dt) >= 0.0 { 1.0 } else { -1.0 };
    (p, n, tangent, w)
}

// ---------------------------------------------------------------------------------
// The mesh

/// One LOD's mesh. Metres, Y up, the origin on the ground under the rock.
#[derive(Clone, Debug, Default)]
pub struct RockLodMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// xyz along the u of the texture, w the handedness.
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub screen_size: f32,
}

impl RockLodMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// A whole rock: the mesh of each LOD.
#[derive(Clone, Debug, Default)]
pub struct RockMesh {
    pub lods: Vec<RockLodMesh>,
    /// Bounds of LOD0.
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Builds the rock a preset's seed lands on. Deterministic.
pub fn build_rock(p: &RockParams) -> RockMesh {
    let surface = Surface::new(p);
    let lift = Vec3::Y * surface.lift();
    let texels = p.texture.size().0;
    let lods: Vec<RockLodMesh> = p
        .lod
        .iter()
        .take(MAX_LODS)
        .map(|l| {
            let n = l.resolution.clamp(2, MAX_MESH_RESOLUTION);
            let mut out = RockLodMesh { screen_size: l.screen_size.max(0.0), ..Default::default() };
            for f in 0..6 {
                let base = out.positions.len() as u32;
                for j in 0..=n {
                    for i in 0..=n {
                        let (s, t) = (i as f32 / n as f32 * 2.0 - 1.0, j as f32 / n as f32 * 2.0 - 1.0);
                        let d = face_dir(f, s, t);
                        // The normal by stepping round the direction, which agrees on
                        // both sides of a seam; the tangent from the face's own u.
                        let p = surface.point(d, false);
                        let (_, normal, tangent, w) = frame(&surface, f, s, t);
                        out.positions.push((p + lift).to_array());
                        out.normals.push(normal.to_array());
                        out.tangents.push([tangent.x, tangent.y, tangent.z, w]);
                        out.uvs.push(face_uv(f, s, t, texels));
                    }
                }
                let row = n + 1;
                let mut tris = Vec::with_capacity((n * n * 6) as usize);
                for j in 0..n {
                    for i in 0..n {
                        let a = base + j * row + i;
                        let (b, c, d) = (a + 1, a + row, a + row + 1);
                        tris.extend_from_slice(&[a, c, b, b, c, d]);
                    }
                }
                // Wound to face out, whichever way the face's u and v happen to turn.
                let pos = &out.positions;
                let first = |k: usize| Vec3::from(pos[tris[k] as usize]);
                let facing = (first(1) - first(0)).cross(first(2) - first(0)).dot(first(0) - lift);
                if facing < 0.0 {
                    for tri in tris.chunks_exact_mut(3) {
                        tri.swap(1, 2);
                    }
                }
                out.indices.extend(tris);
            }
            out
        })
        .collect();
    let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
    if let Some(l) = lods.first() {
        for p in &l.positions {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
    }
    RockMesh { lods, min, max }
}

// ---------------------------------------------------------------------------------
// The maps

/// A rock's maps, laid out as its mesh's texture coordinates expect.
pub struct RockMaps {
    pub albedo: Bitmap,
    /// Tangent space, green up the texture.
    pub normal: Bitmap,
    /// Occlusion in red, roughness in green, no metal in blue: glTF's packing.
    pub orm: Bitmap,
}

/// Paints a rock's maps at the texture resolution its preset asks for.
pub fn bake_rock(p: &RockParams) -> RockMaps {
    bake_rock_at(p, p.texture.size().0)
}

/// Paints a rock's maps at `texels` per face edge, for a preview that needs them sooner.
pub fn bake_rock_at(p: &RockParams, texels: u32) -> RockMaps {
    let surface = Surface::new(p);
    let t = texels.clamp(32, MAX_TEXTURE_RESOLUTION);
    let (w, h) = (3 * t, 2 * t);
    let mut maps = RockMaps { albedo: Bitmap::new(w, h), normal: Bitmap::new(w, h), orm: Bitmap::new(w, h) };
    let faces: Vec<Vec<[[u8; 4]; 3]>> = run_faces(|f| paint_face(&surface, p, f, t));
    for (f, texels_of_face) in faces.iter().enumerate() {
        let (ox, oy) = ((f % 3) as u32 * t, (f / 3) as u32 * t);
        for y in 0..t {
            for x in 0..t {
                let [a, n, o] = texels_of_face[(y * t + x) as usize];
                maps.albedo.put(ox + x, oy + y, a);
                maps.normal.put(ox + x, oy + y, n);
                maps.orm.put(ox + x, oy + y, o);
            }
        }
    }
    maps
}

/// Runs `paint` for each of the six faces, on threads where there are threads.
fn run_faces<T: Send>(paint: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::scope(|scope| {
            let paint = &paint;
            let handles: Vec<_> = (0..6).map(|f| scope.spawn(move || paint(f))).collect();
            handles.into_iter().map(|h| h.join().expect("a face failed to paint")).collect()
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..6).map(paint).collect()
    }
}

/// Every texel of one face of the sheet: albedo, normal and packed occlusion and
/// roughness. The pad round the face is painted by carrying the face on past its edge.
fn paint_face(surface: &Surface, p: &RockParams, f: usize, t: u32) -> Vec<[[u8; 4]; 3]> {
    let c = &p.colour;
    let m = &p.moss;
    let cavity_k = p.surface.cavity.max(0.0);
    let inner = t as f32 - 2.0 * TEXTURE_PAD;
    // A step round the direction about as wide as a texel, for the fine normal.
    let e = 2.0 / inner * 0.55;
    let lin = |c: [f32; 3]| Vec3::from(c.map(srgb_to_linear));
    let (stone, warm, lichen_c) = (lin(c.stone), lin(c.warm), lin(c.lichen_color));
    let (moss_c, moss_tip, moss_dry) = (lin(c.moss), lin(c.moss_tip), lin(c.moss_dry));
    let n1 = &surface.noise;
    let n2 = &surface.noise2;
    let flat = surface.shape.flat_bottom.clamp(0.0, 0.9);
    let mut out = Vec::with_capacity((t * t) as usize);
    for y in 0..t {
        for x in 0..t {
            let s = ((x as f32 + 0.5 - TEXTURE_PAD) / inner) * 2.0 - 1.0;
            let tt = ((y as f32 + 0.5 - TEXTURE_PAD) / inner) * 2.0 - 1.0;
            let d = face_dir(f, s, tt);
            let (p_hi, n_hi) = surface.point_normal(d, true, e);
            let (p_lo, n_lo, tangent, w) = frame(surface, f, s, tt);
            let bitangent = n_lo.cross(tangent) * w;

            let q = d * 3.0;
            let r_hi = (p_hi / surface.half).length();
            let r_lo = (p_lo / surface.half).length();
            // Hollows sit below the smooth surface, bumps above it.
            let cavity = ((r_hi - r_lo) * 18.0).clamp(-1.0, 1.0);
            let height01 = ((p_hi.y / surface.half.y + flat) / (1.0 + flat)).clamp(0.0, 1.0);

            // Stone.
            let grain = n1.at(q * 40.0);
            let speck = smoothstep(0.35, 0.6, n2.at(q * 90.0)) * c.speckle.clamp(0.0, 1.0) * 2.0;
            let light = smoothstep(0.45, 0.65, n1.at(q * 110.0 + Vec3::splat(3.0))) * c.speckle.clamp(0.0, 1.0) * 2.0;
            let tone = n1.fbm(q * 2.0, 4);
            let warmth = smoothstep(-0.3, 0.3, n2.fbm(q * 0.8 + Vec3::new(5.0, 0.0, 0.0), 3));
            let mut col = stone.lerp(warm, warmth) * (1.0 + c.tone * tone + c.grain * grain);
            col *= (1.0 - 0.55 * speck.min(1.0)) * (1.0 + 0.8 * light.min(1.0));
            let streak = smoothstep(0.2, 0.7, n2.fbm(Vec3::new(q.x * 1.5, q.y * 10.0, q.z * 1.5), 3)) * c.streaks;
            col *= 1.0 - streak;
            let lichen = smoothstep(0.35, 0.55, n1.fbm(q * 6.0 + Vec3::new(7.0, 0.0, 0.0), 4)) * c.lichen.clamp(0.0, 1.0);
            col = col.lerp(lichen_c, lichen * 0.8);

            // Moss: what faces up, low on the rock, in its hollows, broken into patches.
            let patches = n2.fbm(q * 1.6 * m.patchiness.max(0.05) + Vec3::new(3.0, 0.0, 0.0), 5);
            let fine = n1.fbm(q * 25.0, 3);
            let field = n_hi.y * m.upward + patches * 0.8 + m.amount - 0.55 - 0.3 * (1.0 - height01)
                + m.crevices * (-cavity).clamp(0.0, 1.0);
            let soft = m.softness.clamp(0.01, 1.0);
            let moss = smoothstep(0.1, 0.1 + soft, field + fine * 0.15);
            let variety = n1.fbm(q * 4.0, 3);
            let mut mc = moss_c * (1.0 + 0.5 * variety);
            mc = mc.lerp(moss_tip, smoothstep(0.1, 0.5, variety));
            mc *= 0.75 + 0.25 * (fine * 0.5 + 0.5);
            let dried = smoothstep(0.3, 0.8, n2.fbm(q * 3.0 + Vec3::new(11.0, 0.0, 0.0), 3)) * m.dry.clamp(0.0, 1.0);
            mc = mc.lerp(moss_dry, dried * 0.7);
            let albedo = col.lerp(mc, moss);

            // Occlusion: hollows, and the underside near the ground.
            let occl = ((1.0 - (-cavity).clamp(0.0, 1.0) * 0.8 * cavity_k).clamp(0.2, 1.0))
                * (0.55 + 0.45 * smoothstep(-0.2, 0.5, height01));
            let rough = c.roughness + 0.06 * grain + (c.moss_roughness - c.roughness) * moss;

            // Moss is fuzz over the stone: its normal is jostled texel to texel.
            let fuzz = Vec3::new(n1.at(q * 70.0), n2.at(q * 70.0 + Vec3::splat(5.0)), 0.0);
            let bumped = (n_hi + (tangent * fuzz.x + bitangent * fuzz.y) * 0.35 * moss).normalize_or(n_hi);
            let tn = Vec3::new(bumped.dot(tangent), bumped.dot(bitangent), bumped.dot(n_lo)).normalize_or(Vec3::Z);

            let a = albedo.to_array().map(|v| to_u8(linear_to_srgb(v)));
            let n = [to_u8(tn.x * 0.5 + 0.5), to_u8(tn.y * 0.5 + 0.5), to_u8(tn.z * 0.5 + 0.5), 255];
            let o = [to_u8(occl), to_u8(rough.clamp(0.0, 1.0)), 0, 255];
            out.push([[a[0], a[1], a[2], 255], n, o]);
        }
    }
    out
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(src: &str) -> RockParams {
        let mut p = parse_rock_template(src).unwrap().instance();
        p.texture.resolution = 48;
        p
    }

    #[test]
    fn presets_parse_and_round_trip() {
        for (name, src) in builtin_rock_presets() {
            let t = parse_rock_template(src).unwrap_or_else(|e| panic!("{name}: {e}"));
            let text = ron::ser::to_string_pretty(&t, Default::default()).unwrap();
            assert_eq!(parse_rock_template(&text).unwrap(), t, "{name} does not round-trip");
        }
    }

    #[test]
    fn a_rock_is_the_same_every_time_it_is_built() {
        let p = small(BOULDER_RON);
        assert_eq!(build_rock(&p).lods[0].positions, build_rock(&p).lods[0].positions);
        assert_eq!(bake_rock(&p).albedo, bake_rock(&p).albedo);
    }

    #[test]
    fn a_rock_is_closed_faces_out_and_sits_sunk_in_the_ground() {
        for (name, src) in builtin_rock_presets() {
            let p = small(src);
            let m = build_rock(&p);
            let lod = &m.lods[0];
            // Every triangle faces away from the middle of the rock.
            let centre = (Vec3::from(m.min) + Vec3::from(m.max)) * 0.5;
            let mut inward = 0;
            for tri in lod.indices.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(lod.positions[tri[k] as usize]));
                let n = (b - a).cross(c - a);
                if n.dot((a + b + c) / 3.0 - centre) < 0.0 {
                    inward += 1;
                }
            }
            assert!(inward * 100 < lod.triangle_count(), "{name}: {inward} triangles face in");
            // Part of it below the origin, most of it above.
            assert!(m.min[1] < 0.0 && m.max[1] > -m.min[1] * 2.0, "{name}: y from {} to {}", m.min[1], m.max[1]);
            let size = Vec3::from(m.max) - Vec3::from(m.min);
            assert!((size.x - p.shape.width).abs() < p.shape.width * 0.4, "{name}: {size} against {}", p.shape.width);
        }
    }

    #[test]
    fn every_lod_is_coarser_than_the_one_before() {
        let m = build_rock(&small(BOULDER_RON));
        assert!(m.lods.len() >= 2);
        for pair in m.lods.windows(2) {
            assert!(pair[1].triangle_count() < pair[0].triangle_count());
        }
    }

    #[test]
    fn frames_are_unit_and_square() {
        let lod = &build_rock(&small(SLAB_RON)).lods[0];
        for (n, t) in lod.normals.iter().zip(&lod.tangents) {
            let (n, tv) = (Vec3::from(*n), Vec3::new(t[0], t[1], t[2]));
            assert!((n.length() - 1.0).abs() < 1e-3 && (tv.length() - 1.0).abs() < 1e-3);
            assert!(n.dot(tv).abs() < 1e-3);
            assert!(t[3].abs() == 1.0);
        }
        assert!(lod.uvs.iter().all(|uv| (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1])));
    }

    #[test]
    fn moss_grows_on_top_and_more_moss_is_greener() {
        let green = |amount: f32| {
            let mut p = small(BOULDER_RON);
            p.moss.amount = amount;
            let img = bake_rock(&p).albedo;
            // The top face is the third in the sheet.
            let t = p.texture.size().0;
            let (mut g, mut n) = (0.0f64, 0.0f64);
            for y in 0..t {
                for x in 2 * t..3 * t {
                    let px = img.at(x, y);
                    g += px[1] as f64 - (px[0] as f64 + px[2] as f64) * 0.5;
                    n += 1.0;
                }
            }
            g / n
        };
        let (bare, mossy) = (green(-1.0), green(0.6));
        assert!(mossy > bare + 10.0, "top face green excess {bare:.1} bare, {mossy:.1} mossy");
    }

    #[test]
    fn normal_maps_face_out_of_the_surface() {
        let maps = bake_rock(&small(BOULDER_RON));
        // Mostly close to the smooth surface's own normal; only crack walls and the
        // like turn far off it, and nothing turns right round.
        let n = &maps.normal;
        let (mut sum, mut steep, mut count) = (0.0f32, 0usize, 0usize);
        let t = 48u32;
        let pad = TEXTURE_PAD as u32;
        // Inside the faces: the pad round each is only ever reached by filtering.
        let inside = |v: u32| (pad..t - pad).contains(&(v % t));
        for y in (0..n.height).filter(|&y| inside(y)) {
            for x in (0..n.width).filter(|&x| inside(x)) {
                let z = n.at(x, y)[2] as f32 / 255.0 * 2.0 - 1.0;
                assert!(z > -0.05, "normal turned away at {x}, {y}");
                sum += z;
                steep += (z < 0.3) as usize;
                count += 1;
            }
        }
        assert!(sum / count as f32 > 0.85, "mean z {}", sum / count as f32);
        assert!(steep * 100 < count, "{steep} of {count} texels steeper than 70 degrees");
    }
}
