//! Rocks: boulders, slabs and stones with their own baked maps, for an engine to scatter
//! and half bury.
//!
//! A rock is a surface given as a radius in every direction from its middle, so it is
//! closed and can never fold over itself. Its form is an ellipsoid cut by planes into
//! broad faces, spread round it so no two crowd together, their edges worn round by
//! amounts that vary from edge to edge, and worked in metres so a slab's edges are no
//! rounder across its top than down its sides; then swelled and rolled by noise, chipped
//! where its edges are crisp enough to chip, each chip a small plane shearing the corner
//! off, and in bedded stone stepped into ledges down its sides. Detail finer than a mesh should carry is added for the maps alone: grain,
//! pits, and joint cracks, planes through the stone that show on its faces as nearly
//! straight lines, each thinning out and stopping somewhere along its length. All of it
//! is held back to what the texels can show, so nothing in the maps is finer than a
//! texel and nothing breaks into stair steps.
//!
//! The mesh is that surface sampled finely through a cube pushed out onto it, then cut
//! down to each LOD's triangle budget by collapsing edges, least error first, so the
//! triangles gather at edges and chips and thin out over flat faces. Each face of the
//! cube is one chart of the maps. A collapse never takes a seam off the line between
//! two charts, so every LOD keeps the charts whole and shares the one set of maps. The
//! charts are packed into one square sheet, each sized to the stone it covers, the
//! buried underside given less.
//!
//! The maps are baked against LOD0 itself: each texel of a triangle shows the point of
//! the surface the triangle stands in front of, with its normal told in the frame the
//! renderer will interpolate there, so the shading is the fine surface's wherever the
//! mesh has cut a corner. Colour, moss, occlusion and roughness are painted from the
//! surface: grain and lichen rosettes on the stone, lighter worn edges and fresher stone
//! where it has chipped, dark cracks, water streaks down its sides, soil where it meets
//! the ground, and moss in cushions on what faces up.
//!
//! A preset is read like a species: any number in `shape`, `surface` and `moss` may be a
//! `(lo, hi)` range, landed once per seed.

use std::f32::consts::FRAC_PI_4;

use glam::{Vec2, Vec3};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::cluster::Bitmap;
use crate::ranged::{key, Ranged, Scalar};
use crate::seed::PortableRng;

mod simplify;

use simplify::{Simplifier, Source};

pub const BOULDER_RON: &str = include_str!("../../../assets/rocks/boulder.ron");
pub const MOSSY_BOULDER_RON: &str = include_str!("../../../assets/rocks/mossy_boulder.ron");
pub const SLAB_RON: &str = include_str!("../../../assets/rocks/slab.ron");
pub const STONE_RON: &str = include_str!("../../../assets/rocks/stone.ron");
pub const GNEISS_RON: &str = include_str!("../../../assets/rocks/gneiss.ron");

/// Where presets saved from the viewer go, and are found by name from the CLI.
pub const CUSTOM_ROCK_DIR: &str = "assets/rocks/custom";

pub fn builtin_rock_presets() -> Vec<(&'static str, &'static str)> {
    vec![
        ("boulder", BOULDER_RON),
        ("mossy_boulder", MOSSY_BOULDER_RON),
        ("slab", SLAB_RON),
        ("stone", STONE_RON),
        ("gneiss", GNEISS_RON),
    ]
}

/// A rock preset as its file describes it, ranges and all.
pub type RockTemplate = RockParams<Ranged>;

pub fn parse_rock_template(ron_src: &str) -> Result<RockTemplate, String> {
    ron::from_str(ron_src).map_err(|e| e.to_string())
}

/// Most planes a rock may be cut by, and most joint cracks through it.
const MAX_FACETS: u32 = 32;
const MAX_CRACKS: u32 = 8;
/// Most chips off a rock's edges, the mesh's and the maps' together.
const MAX_CHIPS: usize = 96;
/// Fewest and most triangles a LOD may ask for, and most LODs.
const MIN_TRIANGLES: u32 = 24;
const MAX_TRIANGLES: u32 = 40_000;
const MAX_LODS: usize = 4;
/// Quads along each edge of each cube face the LODs are cut down from: fewest and most.
const MIN_SOURCE: u32 = 24;
const MAX_SOURCE: u32 = 128;
/// Smallest and largest sheet of maps, texels a side.
const MIN_TEXTURE: u32 = 64;
const MAX_TEXTURE: u32 = 4096;
/// Sheet left round each chart, as a share of the sheet's side, filled by repeating the
/// chart's edge outward so filtering never pulls in the next one. Twelve texels at
/// 1024, which holds good down to the fourth mip.
const ATLAS_PAD: f32 = 12.0 / 1024.0;
/// Share of the sheet a metre of the underside gets against a metre of the rest: most
/// of it is under the ground.
const UNDERSIDE_SHARE: f32 = 0.35;

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
                RockLod { triangles: 1600, screen_size: 0.0 },
                RockLod { triangles: 600, screen_size: 0.3 },
                RockLod { triangles: 200, screen_size: 0.12 },
                RockLod { triangles: 70, screen_size: 0.05 },
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

/// The rock's form: what the mesh carries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockShape::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockShape<V = f32> {
    /// Metres across, high and deep, before noise and the flat bottom.
    pub width: V,
    pub height: V,
    pub depth: V,
    /// Planes the rock is cut by, and how deep they cut, as a share of how far the
    /// ellipsoid reaches that way: the broad faces of a boulder.
    pub facets: V,
    pub facet_depth: V,
    /// How sharply the faces meet: low rounds every edge off like a river stone, high
    /// leaves them crisp as fresh rubble.
    pub sharpness: V,
    /// How unevenly the edges are worn: at 0 every edge is as round as the next, at 1
    /// some stay four times as crisp as others.
    pub wear: V,
    /// Swelling and denting of the whole, as a share of the radius, and how many swells
    /// fit round it.
    pub bulge: V,
    pub bulge_scale: V,
    /// Broad rolls and hollows over the faces, as a share of the radius.
    pub undulation: V,
    /// How deep chips are struck off the edges, as a share of the radius, and how far
    /// each runs along its edge, as a share of it too. Only edges crisp enough for a
    /// chip to bite past their rounding are chipped.
    pub chips: V,
    pub chip_size: V,
    /// Plates flaking off the stone, stacked a step above one another with a riser
    /// between: the share of the rock they cover, the height of each step and the size of
    /// a plate, both as shares of the radius, and how far they lie along the bedding, from
    /// shells following the surface at 0 to layers of slate at 1. The big plates are the
    /// mesh's; a finer flaking over them is the maps'.
    pub flaking: V,
    pub flake_step: V,
    pub flake_size: V,
    pub bedded: V,
    /// Stepped ledges where the stone has split along its bedding, as a share of the
    /// radius.
    pub fracture: V,
    /// Where the top is cut flat along the bedding, as a share of the half height
    /// above the middle: 0 for no flat top. A slab's.
    pub flat_top: V,
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
            facets: 13.0,
            facet_depth: 0.35,
            sharpness: 22.0,
            wear: 0.6,
            bulge: 0.06,
            bulge_scale: 1.2,
            undulation: 0.02,
            chips: 0.035,
            chip_size: 0.3,
            flaking: 0.6,
            flake_step: 0.012,
            flake_size: 0.35,
            bedded: 0.0,
            fracture: 0.0,
            flat_top: 0.0,
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
            wear: f(&key(at, "wear"), self.wear),
            bulge: f(&key(at, "bulge"), self.bulge),
            bulge_scale: f(&key(at, "bulge_scale"), self.bulge_scale),
            undulation: f(&key(at, "undulation"), self.undulation),
            chips: f(&key(at, "chips"), self.chips),
            chip_size: f(&key(at, "chip_size"), self.chip_size),
            flaking: f(&key(at, "flaking"), self.flaking),
            flake_step: f(&key(at, "flake_step"), self.flake_step),
            flake_size: f(&key(at, "flake_size"), self.flake_size),
            bedded: f(&key(at, "bedded"), self.bedded),
            fracture: f(&key(at, "fracture"), self.fracture),
            flat_top: f(&key(at, "flat_top"), self.flat_top),
            flat_bottom: f(&key(at, "flat_bottom"), self.flat_bottom),
            bury: f(&key(at, "bury"), self.bury),
        }
    }
}

/// Detail only the maps carry. Sizes here are in metres, since the grain of a stone
/// does not grow with the stone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "RockSurface::defaults", bound(deserialize = "V: Scalar + Deserialize<'de>"))]
pub struct RockSurface<V = f32> {
    /// Height of the fine unevenness of the stone, metres.
    pub grain: V,
    /// Share of the stone pitted with small holes.
    pub pitting: V,
    /// Joint cracks through the stone, and how wide they open, metres.
    pub cracks: V,
    pub crack_width: V,
    /// Bands of bedding across the stone, in colour and in how far each layer has worn
    /// back: 0 for none.
    pub strata: V,
    /// How strongly hollows and cracks are darkened and occluded.
    pub cavity: V,
    /// Height of the knobbly, broken relief a few centimetres across that weathering
    /// leaves on stone, metres: its crevices dark and its knobs pale, as in a photograph.
    pub crunch: V,
    /// How far the grain is drawn out into parallel lines, as in gneiss or schist: 0 for
    /// none.
    pub foliation: V,
}

impl Default for RockSurface {
    fn default() -> Self {
        Self { grain: 0.0015, pitting: 0.2, cracks: 1.0, crack_width: 0.005, strata: 0.0, cavity: 1.0, crunch: 0.008, foliation: 0.0 }
    }
}

impl<V: Scalar> RockSurface<V> {
    pub fn defaults() -> Self {
        RockSurface::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> RockSurface<W> {
        RockSurface {
            grain: f(&key(at, "grain"), self.grain),
            pitting: f(&key(at, "pitting"), self.pitting),
            cracks: f(&key(at, "cracks"), self.cracks),
            crack_width: f(&key(at, "crack_width"), self.crack_width),
            strata: f(&key(at, "strata"), self.strata),
            cavity: f(&key(at, "cavity"), self.cavity),
            crunch: f(&key(at, "crunch"), self.crunch),
            foliation: f(&key(at, "foliation"), self.foliation),
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
    /// How far moss gathers in hollows, cracks and along ledges.
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
    /// The stone, and the warmer crust it weathers to: the crust keeps to the plates
    /// standing proud, the stone shows where they have flaked away.
    pub stone: [f32; 3],
    pub warm: [f32; 3],
    /// Spread of brightness across the rock: in broad patches, and in blotches a hand
    /// across.
    pub tone: f32,
    pub grain: f32,
    /// Mineral grains through the stone, about `mineral_size` metres across: dark ones,
    /// pale `feldspar` ones and a few green, as in granite.
    pub speckle: f32,
    pub mineral_size: f32,
    pub feldspar: [f32; 3],
    /// Crusts of lichen, as a share of the stone: rosettes about `lichen_size` metres
    /// across, most of them `lichen_color`, some dark and a few `lichen_rare`.
    pub lichen: f32,
    pub lichen_size: f32,
    pub lichen_color: [f32; 3],
    pub lichen_rare: [f32; 3],
    /// Darker streaks run down the steep faces by water.
    pub streaks: f32,
    /// Soil splashed up the foot of the rock and packed into its cracks.
    pub dirt: f32,
    pub dirt_color: [f32; 3],
    /// How much paler the edges are, worn by weather.
    pub edge_wear: f32,
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
            stone: [0.40, 0.41, 0.37],
            warm: [0.52, 0.45, 0.40],
            tone: 0.35,
            grain: 0.25,
            speckle: 0.5,
            mineral_size: 0.009,
            feldspar: [0.60, 0.55, 0.52],
            lichen: 0.35,
            lichen_size: 0.08,
            lichen_color: [0.60, 0.62, 0.50],
            lichen_rare: [0.78, 0.68, 0.32],
            streaks: 0.25,
            dirt: 0.5,
            dirt_color: [0.34, 0.29, 0.23],
            edge_wear: 0.4,
            moss: [0.22, 0.25, 0.13],
            moss_tip: [0.38, 0.41, 0.19],
            moss_dry: [0.40, 0.34, 0.24],
            roughness: 0.9,
            moss_roughness: 0.97,
        }
    }
}

/// One level of detail: the triangles it is cut down to, and the screen size it takes
/// over at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RockLod {
    pub triangles: u32,
    pub screen_size: f32,
}

/// Texels along each side of the square sheet the maps are baked into.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RockTexture {
    pub size: u32,
}

impl Default for RockTexture {
    fn default() -> Self {
        Self { size: 2048 }
    }
}

impl RockTexture {
    /// Texels along each side of the sheet, as it will be baked.
    pub fn size(&self) -> u32 {
        self.size.clamp(MIN_TEXTURE, MAX_TEXTURE)
    }
}

// ---------------------------------------------------------------------------------
// Noise

/// Gradient noise in three dimensions, seeded.
struct Noise {
    perm: [u8; 512],
    grads: [Vec3; 256],
}

/// Turns each octave of noise away from the one before, so no two line up along an axis
/// and their repeats never agree.
fn turn(p: Vec3) -> Vec3 {
    Vec3::new(0.80 * p.y + 0.60 * p.z, -0.80 * p.x + 0.36 * p.y - 0.48 * p.z, -0.60 * p.x - 0.48 * p.y + 0.64 * p.z)
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
        let (mut amp, mut q, mut sum, mut norm) = (1.0, p, 0.0, 0.0);
        for _ in 0..octaves {
            sum += self.at(q) * amp;
            norm += amp;
            amp *= 0.5;
            q = turn(q) * 2.0 + Vec3::new(17.3, 31.7, 5.1);
        }
        sum / norm
    }

    /// Noise from `freq` cycles a metre up through `octaves` doublings, each octave
    /// fading out as it nears what texels `texel` metres wide can show. What is dropped
    /// is dropped from the sum, not made up by the rest.
    fn fbm_limited(&self, p: Vec3, freq: f32, texel: f32, octaves: u32) -> f32 {
        self.fbm_rough(p, freq, texel, octaves, 0.5)
    }

    /// As `fbm_limited`, each octave `gain` of the one before: above a half, the finer
    /// octaves count for more, and the surface reads rough rather than lumpy.
    fn fbm_rough(&self, p: Vec3, freq: f32, texel: f32, octaves: u32, gain: f32) -> f32 {
        let (mut amp, mut f, mut q, mut sum, mut norm) = (1.0, freq, p * freq, 0.0, 0.0);
        for _ in 0..octaves {
            norm += amp;
            let keep = shown(f, texel);
            if keep > 0.0 {
                sum += self.at(q) * amp * keep;
            }
            amp *= gain;
            f *= 2.0;
            q = turn(q) * 2.0 + Vec3::new(17.3, 31.7, 5.1);
        }
        sum / norm
    }
}

/// How much of detail at `freq` cycles a metre texels `texel` metres wide can show:
/// all of it at four texels a cycle, none by two and a half.
fn shown(freq: f32, texel: f32) -> f32 {
    1.0 - smoothstep(0.25, 0.4, freq * texel)
}

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

fn rehash(h: u32, k: u32) -> u32 {
    mix(h ^ k.wrapping_mul(0x9E37_79B9))
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Visits the point scattered in each cell of the unit lattice round `q`, with its
/// hash: everything within a cell of `q` is among them.
fn cells(q: Vec3, salt: u32, mut visit: impl FnMut(Vec3, u32)) {
    let (cx, cy, cz) = (q.x.floor() as i32, q.y.floor() as i32, q.z.floor() as i32);
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y, z) = (cx + dx, cy + dy, cz + dz);
                let h = mix(salt
                    ^ (x as u32).wrapping_mul(0x8DA6_B343)
                    ^ (y as u32).wrapping_mul(0xD816_3841)
                    ^ (z as u32).wrapping_mul(0xCB1A_B31F));
                let at = Vec3::new(x as f32 + unit(h), y as f32 + unit(rehash(h, 1)), z as f32 + unit(rehash(h, 2)));
                visit(at, h);
            }
        }
    }
}

const SALT_PITS: u32 = 0x510E_527F;
const SALT_LICHEN: u32 = 0x9B05_688C;
const SALT_TUFTS: u32 = 0x1F83_D9AB;
const SALT_MOSS: u32 = 0x5BE0_CD19;
const SALT_LITTER: u32 = 0x6A09_E667;
const SALT_MINERALS: u32 = 0x3C6E_F372;

// ---------------------------------------------------------------------------------
// The surface

/// A chip: a small plane shearing off an edge, reaching only so far round the rock
/// from where it was struck, so it leaves a flat scar with a crisp rim rather than a
/// crater.
struct Chip {
    /// Where it was struck, and how far round from there it reaches, as the cosine of
    /// the angle.
    at: Vec3,
    reach: f32,
    /// The plane, in metres, and how far it lies below the edge where it was struck.
    normal: Vec3,
    offset: f32,
    depth: f32,
    /// Too small for the mesh: the maps' alone.
    fine: bool,
}

/// A joint: a plane through the stone that shows where it meets the surface.
struct Crack {
    normal: Vec3,
    /// A point on the plane, metres from the rock's middle.
    origin: Vec3,
    /// How far across the plane from `origin` it runs before it has closed up.
    reach: f32,
    width: f32,
    /// How far one side has settled below the other.
    step: f32,
    /// Where its wander is read from the noise.
    offset: Vec3,
}

/// What the surface is doing at one point, beside where it is.
#[derive(Clone, Copy, Default)]
struct Sample {
    /// Distance out, as a share of the rock's half extents in that direction.
    r: f32,
    /// Near a worn edge between faces, 0 to 1.
    edge: f32,
    /// On a chip's scar: fresh stone.
    chip: f32,
    /// On a ledge's flat, where water and soil sit.
    tread: f32,
    /// In a crack, in a pit, in a small hollow of the grain.
    crack: f32,
    pit: f32,
    hollow: f32,
    /// On a knob of the fine relief, 0 to 1: the weathered tops that catch the light.
    ridge: f32,
    /// Which layer of the bedding, -1 to 1, for its colour.
    band: f32,
    /// How high the plate here stands among the others, 0 to 1, and a tint of its own,
    /// 0 to 1.
    plate: f32,
    tint: f32,
    /// Under the weathered crust, 0 to 1, where the crust decides the plates; `None`
    /// where it is painted from which way the stone faces instead.
    crust: Option<f32>,
    /// On a plate's step, at the foot of one where it meets the plate below, and along
    /// the lip at its top.
    riser: f32,
    foot: f32,
    lip: f32,
}

/// A place among plates.
struct Plates {
    /// Height, in steps.
    h: f32,
    tint: f32,
    /// Share of the fragments round here on a level of 0 or above: the crust's, when
    /// the levels are crust and bare stone.
    top: f32,
    /// At the foot of a step up to the next plate, and at the lip of a step down.
    foot: f32,
    lip: f32,
}

/// How many steps a plate may stand above or below the middle.
const PLATE_LEVELS: f32 = 1.4;

/// A rock's surface, as a radius in every direction.
struct Surface {
    /// Each face's plane: the way it faces, and how far out it stands, metres.
    planes: Vec<(Vec3, f32)>,
    sharpness: f32,
    /// Half extents, metres, and their mean: the rock's radius.
    half: Vec3,
    radius: f32,
    shape: RockShape,
    surface: RockSurface,
    noise: Noise,
    noise2: Noise,
    /// Which way the stone's bedding runs: its ledges are steps across it.
    bedding: Vec3,
    /// Square to the planes the grain is drawn out along, where it is foliated.
    foliate: Vec3,
    chips: Vec<Chip>,
    cracks: Vec<Crack>,
    salt: u32,
}

impl Surface {
    fn new(p: &RockParams) -> Self {
        let mut rng = PortableRng::seed_from_u64(p.seed ^ 0x0B0C_4000_5EED_0001);
        let s = &p.shape;
        let facets = (s.facets.round().max(0.0) as u32).min(MAX_FACETS);
        let depth = s.facet_depth.clamp(0.0, 0.8);
        let asked = Vec3::new(s.width, s.height, s.depth).max(Vec3::splat(0.02)) * 0.5;
        // Each face the best of a few tries at keeping clear of those before it, so the
        // rock is cut all round rather than hacked at from one side. Leaned toward level,
        // since a boulder's big faces are its sides and its top. Each is kept as the way
        // it faces and how far in it cuts, as a share of how far the ellipsoid reaches
        // that way, and set in metres once the rock's size is known.
        let mut faces: Vec<(Vec3, f32)> = Vec::new();
        for _ in 0..facets {
            let mut best = (Vec3::Y, f32::MAX);
            for _ in 0..6 {
                let n = Vec3::new(
                    rng.random_range(-1.0f32..1.0),
                    rng.random_range(-1.0f32..1.0) * 0.75,
                    rng.random_range(-1.0f32..1.0),
                )
                .normalize_or(Vec3::Y);
                let crowding = faces.iter().map(|(m, _)| n.dot(*m)).fold(-1.0f32, f32::max);
                if crowding < best.1 {
                    best = (n, crowding);
                }
            }
            faces.push((best.0, rng.random_range(1.0 - depth..=1.0 - 0.3 * depth)));
        }
        let noise = Noise::new(&mut rng);
        let noise2 = Noise::new(&mut rng);
        // Near level, as sediment is laid down, tipped a little as it has since been.
        let bedding = Vec3::new(rng.random_range(-0.3f32..0.3), 1.0, rng.random_range(-0.3f32..0.3)).normalize();
        let salt: u32 = rng.random();
        let sharpness = s.sharpness.clamp(2.0, 80.0);
        if s.flat_top > 0.0 {
            faces.push((bedding, s.flat_top.clamp(0.1, 1.0)));
        }
        let set = |half: Vec3| -> Vec<(Vec3, f32)> { faces.iter().map(|&(n, share)| (n, share * (n * half).length())).collect() };

        // The cuts take the rock in from the size it was asked for, so it is scaled back
        // out until its faces stand where width, height and depth say.
        let mut half = asked;
        for _ in 0..2 {
            let planes = set(half);
            let radius = (half.x + half.y + half.z) / 3.0;
            let mut lo = Vec3::splat(-0.3) * half;
            let mut hi = Vec3::splat(0.3) * half;
            for i in 0..48 {
                for j in 0..24 {
                    let (az, el) = (i as f32 / 48.0 * std::f32::consts::TAU, (j as f32 + 0.5) / 24.0 * std::f32::consts::PI);
                    let d = Vec3::new(el.sin() * az.cos(), el.cos(), el.sin() * az.sin());
                    let p = d * half * polytope(&planes, &[], false, d, half, radius, sharpness / radius).r;
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            half *= asked / ((hi - lo) * 0.5).max(half * 0.3);
        }
        let mut planes = set(half);
        // Then the flat it sits on, one more plane, left out of the sizing: the height
        // asked for is to where the rock would have gone on below it.
        planes.push((Vec3::NEG_Y, s.flat_bottom.clamp(0.0, 0.9) * half.y));

        let mut surface = Self {
            planes,
            sharpness,
            half,
            radius: (half.x + half.y + half.z) / 3.0,
            shape: s.clone(),
            surface: p.surface.clone(),
            noise,
            noise2,
            bedding,
            foliate: {
                // From the salt, not the stream, so every other draw stays where it was.
                let h = mix(salt ^ 0xF011_A7E0);
                Vec3::new(unit(h) - 0.5, unit(rehash(h, 1)) - 0.5, unit(rehash(h, 2)) - 0.5).normalize_or(Vec3::Y)
            },
            chips: Vec::new(),
            cracks: Vec::new(),
            salt,
        };

        // Chips struck off the edges: some big enough for the mesh to carry, and three
        // times as many small ones only the maps do. Each is a plane across an edge
        // between two faces, square to neither, set in below the edge so it shears the
        // corner off and nothing more.
        if s.chips > 0.0 {
            let size = s.chip_size.clamp(0.05, 1.0);
            let deep = s.chips * surface.radius;
            let big = ((5.0 / size).round() as usize).clamp(4, MAX_CHIPS / 4);
            let small = (big * 3).min(MAX_CHIPS - big);
            for (count, size, depth, fine) in [(big, size, deep, false), (small, size * 0.3, deep * 0.3, true)] {
                let mut placed = 0;
                for _ in 0..count * 80 {
                    if placed == count {
                        break;
                    }
                    let d = Vec3::new(
                        rng.random_range(-1.0f32..1.0),
                        rng.random_range(-1.0f32..1.0),
                        rng.random_range(-1.0f32..1.0),
                    );
                    let tip = rng.random_range(-0.3f32..0.3);
                    let cut_in = depth * rng.random_range(0.35f32..1.0);
                    let reach = size * rng.random_range(0.6f32..1.2);
                    if d.length_squared() > 1.0 || d.length_squared() < 1e-4 {
                        continue;
                    }
                    let d = d.normalize();
                    if let Some(chip) = surface.chip_at(d, tip, cut_in, reach, fine) {
                        surface.chips.push(chip);
                        placed += 1;
                    }
                }
            }
        }

        let f = &p.surface;
        let count = (f.cracks.round().max(0.0) as u32).min(MAX_CRACKS);
        let width = f.crack_width.max(0.0);
        for _ in 0..count {
            // Joints stand across the bedding: upright, mostly.
            let mut n = Vec3::new(
                rng.random_range(-1.0f32..1.0),
                rng.random_range(-1.0f32..1.0),
                rng.random_range(-1.0f32..1.0),
            )
            .normalize_or(Vec3::X);
            n = (n - bedding * n.dot(bedding) * 0.85).normalize_or(Vec3::X);
            let origin = Vec3::new(
                rng.random_range(-0.5f32..0.5),
                rng.random_range(-0.2f32..0.6),
                rng.random_range(-0.5f32..0.5),
            ) * half;
            let reach = rng.random_range(0.6f32..1.6) * surface.radius;
            let w = rng.random_range(0.6f32..1.4) * width;
            let step = rng.random_range(-0.6f32..0.6) * w;
            let offset = Vec3::new(rng.random_range(0.0f32..50.0), rng.random_range(0.0f32..50.0), rng.random_range(0.0f32..50.0));
            surface.cracks.push(Crack { normal: n, origin, reach, width: w, step, offset });
        }
        surface
    }

    /// A chip across the edge nearest `d`, if `d` is right by an edge between two faces
    /// that meet at more than a shallow angle, crisp enough there that the chip bites
    /// well past its rounding, and neither face is the flat the rock stands on: the plane
    /// halfway between the two, tipped by `tip` along the edge so it bites
    /// deeper at one end, set `cut_in` below the edge, or less where the faces meet so
    /// shallowly that it would not come back out of them within its `reach`.
    fn chip_at(&self, d: Vec3, tip: f32, cut_in: f32, reach: f32, fine: bool) -> Option<Chip> {
        let e = d * self.half;
        let u = e.normalize();
        let mut near = [(f32::MAX, Vec3::ZERO, 0.0f32); 2];
        for &(normal, offset) in &self.planes {
            let c = u.dot(normal);
            if c <= 1e-3 {
                continue;
            }
            let t = offset / c;
            if t < near[0].0 {
                near[1] = near[0];
                near[0] = (t, normal, offset);
            } else if t < near[1].0 {
                near[1] = (t, normal, offset);
            }
        }
        let [(ta, na, _), (tb, nb, ob)] = near;
        // Both faces cut the ellipsoid, the edge is close, and it is an edge worth the
        // name.
        if tb >= e.length() || tb - ta > 0.05 * self.radius || na.dot(nb) > 0.9 || na.y < -0.9 || nb.y < -0.9 {
            return None;
        }
        // The point of the edge nearest where `d` meets the first face.
        let x0 = u * ta;
        let c = na.dot(nb);
        let r2 = ob - nb.dot(x0);
        let det = 1.0 - c * c;
        let (a, b) = (-c * r2 / det, r2 / det);
        let edge = x0 + na * a + nb * b;
        let along = na.cross(nb).normalize_or_zero();
        let normal = ((na + nb).normalize() + along * tip).normalize_or(na);
        if edge.normalize_or(u).dot(normal) < 0.3 {
            return None;
        }
        let at = (edge / self.half).normalize_or(d);
        let half_turn = 0.5 * c.clamp(-1.0, 1.0).acos();
        let cut_in = cut_in.min(0.6 * reach * ta * half_turn.tan());
        // Cut into an edge worn rounder than that, a chip's scar is only a groove.
        if cut_in < 1.5 * std::f32::consts::LN_2 / self.wear_sharpness(d) {
            return None;
        }
        Some(Chip { at, reach: reach.cos(), normal, offset: normal.dot(edge) - cut_in, depth: cut_in, fine })
    }

    /// How sharply the faces meet in direction `d`, a metre: each edge worn by its own
    /// amount.
    fn wear_sharpness(&self, d: Vec3) -> f32 {
        let q = d * self.half / self.radius;
        self.sharpness / self.radius * (self.shape.wear.clamp(0.0, 1.0) * self.noise.at(q * 1.3 + Vec3::new(40.0, 0.0, 0.0))).exp2()
    }

    /// The surface in unit direction `d`: its radius as a share of the half extents, and
    /// what it is doing there. `texel` asks for the detail only the maps carry, held to
    /// what texels that many metres wide can show; `None` is the mesh's surface.
    fn sample(&self, d: Vec3, texel: Option<f32>) -> Sample {
        let s = &self.shape;
        let r_m = self.radius;
        // On the ellipsoid the rock is cut from, metres, and in radii.
        let e = d * self.half;
        let len = e.length().max(1e-4);
        let q = e / r_m;

        let k = self.wear_sharpness(d);
        let cut = polytope(&self.planes, &self.chips, texel.is_some(), d, self.half, r_m, k);
        let edge = cut.edge;
        let r = cut.r * (1.0 + s.bulge * self.noise.fbm(q * s.bulge_scale.max(0.1), 3));

        // Everything from here is metres out along the surface's own radius. First lumps
        // and hollows, bent so they do not line up.
        let mut out = 0.0;
        let bend = Vec3::new(self.noise.at(q * 1.1 + Vec3::splat(3.0)), self.noise.at(q * 1.1 + Vec3::splat(8.0)), self.noise.at(q * 1.1 + Vec3::splat(13.0)));
        out += s.undulation * r_m * self.noise2.fbm(q * 2.4 + bend * 0.6 + Vec3::new(0.0, 9.0, 0.0), 4);

        let mut tread = 0.0;
        if s.fracture > 0.0 {
            // Height across the bedding, wavered, with the layers unevenly thick; then
            // terraced into near flat treads with steep risers between.
            let mut x = e.dot(self.bedding) / (self.half.y * 0.22) + 1.1 * self.noise2.fbm(q * 2.0 + Vec3::splat(4.0), 3);
            x += 0.3 * self.noise.at(Vec3::new(x * 0.5, 3.1, 7.7));
            let f = x - x.floor();
            let stair = x.floor() + smoothstep(0.65, 1.0, f);
            // Not every stretch of the rock has split, and a face lying along the bedding
            // is one layer's top, not a flight of them.
            let across = 1.0 - smoothstep(0.55, 0.85, cut.facing.dot(self.bedding).abs());
            let mask = smoothstep(-0.25, 0.3, self.noise.fbm(q * 0.9 + Vec3::new(0.0, 0.0, 13.0), 2)) * across;
            // Each layer worn back by its own amount.
            let worn = 0.4 + 1.2 * unit(mix(self.salt ^ (x.floor() as i32 as u32).wrapping_mul(0x2545_F491)));
            out += s.fracture * r_m * (stair - x) * mask * worn;
            tread = mask * (1.0 - smoothstep(0.55, 0.7, f));
        }

        let mut sample = Sample { edge, chip: cut.chip, tread, ..Default::default() };
        sample.plate = 0.5;
        if let Some(tx) = texel {
            let f = &self.surface;
            // Plates flaking off the stone where it has weathered into them, and a finer
            // flaking over those: the maps' alone, since a step a centimetre high moves the
            // outline of a rock by nothing, and in the mesh would only blur.
            if s.flaking > 0.0 && s.flake_step > 0.0 && s.bedded > 0.0 {
                let mask = smoothstep(
                    1.0 - 1.2 * s.flaking,
                    1.35 - 1.2 * s.flaking,
                    0.5 + 0.5 * self.noise2.fbm(q * 0.8 + Vec3::new(71.0, 0.0, 0.0), 2),
                );
                if mask > 0.0 {
                    let size = s.flake_size.clamp(0.05, 2.0) * r_m;
                    let bedded = s.bedded.clamp(0.0, 1.0);
                    let layer = size * 0.3;
                    let level = |c: Vec3| {
                        let pq = c / size;
                        let warp = Vec3::new(self.noise2.at(pq * 0.7 + Vec3::splat(31.0)), self.noise2.at(pq * 0.7 + Vec3::splat(47.0)), 0.0);
                        PLATE_LEVELS * self.noise.fbm(pq + warp * 0.8, 2) + bedded * c.dot(self.bedding) / layer
                    };
                    // Edges worn back over a good part of a fragment, and never over less than
                    // three texels, so a step reads as eroded stone and not as a drawn line.
                    let cell = size * 0.11;
                    let rise = (3.0 * tx).max(0.12 * cell);
                    let big = self.plates(e, cell, rise, self.salt, bedded, level);
                    // Thick in one stretch, thinning to nothing in the next, so a plate's
                    // edge stands out along some of its length and runs back into the stone
                    // along the rest.
                    let thick = smoothstep(-0.5, 0.6, self.noise2.fbm(e / (size * 0.6) + Vec3::new(0.0, 0.0, 91.0), 3)) * 1.6;
                    out += s.flake_step * r_m * mask * thick * (big.h - bedded);
                    let height = (big.h - bedded + PLATE_LEVELS) / (2.0 * PLATE_LEVELS);
                    sample.plate = height.clamp(0.0, 1.0) * mask + 0.5 * (1.0 - mask);
                    sample.tint = big.tint;
                    sample.foot = big.foot * mask * thick.min(1.0);
                    sample.lip = big.lip * mask * thick.min(1.0);
                    sample.riser = sample.foot.max(sample.lip);
                }
            } else if s.flaking > 0.0 && s.flake_step > 0.0 {
                // Unbedded stone weathers to one crust, and the crust is what flakes: the
                // fragments are crust or bare stone, so the only steps are at the crust's
                // broken edge, and the colour changes across them with the height.
                let size = s.flake_size.clamp(0.05, 2.0) * r_m;
                let cell = size * 0.16;
                let rise = (3.0 * tx).max(0.15 * cell);
                let level = |c: Vec3| self.crust_field(c, cell).clamp(-0.99, 0.99);
                let big = self.plates(e, cell, rise, self.salt, 0.0, level);
                let thick = smoothstep(-0.6, 0.5, self.noise2.fbm(e / (size * 0.6) + Vec3::new(0.0, 0.0, 91.0), 3)) * 1.4;
                out += s.flake_step * r_m * thick * (big.h + 0.5);
                sample.plate = big.top;
                sample.crust = Some(big.top);
                sample.tint = big.tint;
                sample.foot = big.foot * thick.min(1.0);
                sample.lip = big.lip * thick.min(1.0);
                sample.riser = sample.foot.max(sample.lip);
            }
            // Where the mesh's surface is: the cracks are planes through the stone, and
            // are drawn where the stone actually is.
            let p_lo = e * (r + out / len);

            // The stone's texture at two scales: hollows and swells a hand across,
            // weathered into it, and the grain of its crystals, rougher in some stretches
            // than others.
            let fe = self.foliated(e);
            let worn = self.noise2.fbm_limited(e + Vec3::splat(9.0), 7.0, tx, 3);
            let crystals = self.noise.fbm_rough(fe, 45.0, tx, 6, 0.62);
            let rougher = 0.45 + 0.9 * smoothstep(-0.4, 0.4, self.noise2.fbm(e * 2.5 + Vec3::splat(21.0), 2));
            // The weathered crust on the plates is rougher than the stone they flaked from.
            let crust = smoothstep(0.35, 0.75, sample.plate);
            let grain = 2.5 * worn + 0.5 * rougher * (0.6 + 1.2 * crust) * crystals;
            out += f.grain.max(0.0) * grain;

            // Knobbly relief a few centimetres across, bent so it does not read as noise:
            // broad rounded knobs with narrow crevices between, folded smoothly rather than
            // creased, since a crease draws a line along every contour.
            let warp = Vec3::new(
                self.noise.at(fe * 9.0 + Vec3::splat(3.0)),
                self.noise.at(fe * 9.0 + Vec3::splat(13.0)),
                self.noise.at(fe * 9.0 + Vec3::splat(23.0)),
            );
            let knob = 0.6 * self.noise2.fbm_rough(fe + warp * 0.014 + Vec3::splat(3.3), 13.0, tx, 5, 0.55)
                + 0.4 * self.noise.fbm_rough(e + warp * 0.02 + Vec3::splat(8.1), 9.0, tx, 4, 0.55);
            let knob = knob - 0.7 * knob * knob;
            let crunch = f.crunch.max(0.0) * 2.2 * knob * (0.6 + 0.4 * rougher) * (1.0 - 0.6 * sample.chip);
            out += crunch;

            // How far this point stands above or sinks below the fine relief round it, as
            // a share of how far it runs: for darkening its crevices and paling its knobs.
            let reach = f.grain.max(0.0) * 1.3 + f.crunch.max(0.0) + 1e-5;
            let rel = (f.grain.max(0.0) * grain + crunch) / reach;
            sample.hollow = smoothstep(0.0, 0.9, -rel);
            sample.ridge = smoothstep(0.05, 0.7, rel);

            let (pit, pit_depth) = self.pits(e, f.pitting * (0.3 + 1.4 * smoothstep(0.35, 0.8, sample.plate)), tx);
            out -= pit_depth;
            sample.pit = pit;

            for c in &self.cracks {
                let rel = p_lo - c.origin;
                let along = rel.dot(c.normal);
                let inplane = (rel - c.normal * along).length();
                let fade = 1.0 - smoothstep(0.5 * c.reach, c.reach, inplane + 0.2 * c.reach * self.noise.at(p_lo * (2.0 / r_m) + c.offset));
                if fade <= 0.0 {
                    continue;
                }
                // A near straight line that wanders a little, and jags at a finer scale.
                let side = along
                    + 0.04 * r_m * self.noise2.fbm(p_lo * (1.6 / r_m) + c.offset, 3)
                    + 0.003 * self.noise.fbm_limited(p_lo + c.offset, 35.0, tx, 3);
                // Narrowing toward its ends. Never drawn narrower than a texel: a crack
                // finer than that is drawn a texel wide, and fainter.
                let w = c.width * (0.55 + 0.45 * self.noise.at(p_lo * 7.0 + c.offset)) * fade;
                let drawn = w.max(tx * 1.3);
                let ink = (w / drawn).clamp(0.0, 1.0);
                let g = (1.0 - side.abs() / drawn).max(0.0);
                let profile = g * g * (3.0 - 2.0 * g);
                out -= 1.6 * w * profile + c.step * fade * smoothstep(-drawn, drawn, side);
                sample.crack = sample.crack.max(profile * ink.sqrt());
            }

            if f.strata > 0.0 {
                // Layers some centimetres thick, of uneven thickness, each worn back by
                // its own amount.
                let b = p_lo.dot(self.bedding) / 0.03 + 1.2 * self.noise.fbm(p_lo * (1.4 / r_m) + Vec3::splat(7.0), 2);
                let band = self.noise2.at(Vec3::new(b * 0.45, 11.3, 4.1)) * 0.7 + self.noise.at(Vec3::new(b * 1.3, 2.2, 9.9)) * 0.3;
                out -= f.strata * 0.005 * smoothstep(-0.1, 0.5, band);
                sample.band = band;
            }
        }
        sample.r = r + out / len;
        sample
    }

    /// Plates at `e`: the stone broken into fragments about `cell` metres across, each
    /// taking the whole step of `level` at its middle, so that fragments on one step run
    /// together into a plate whose edge is the broken edge of the fragments along it. Each step stands an uneven height above or below the last, rising over
    /// `rise` metres either side of an edge. `bedded` plates stand out by random amounts,
    /// as layers of slate do; others stack, each on the one below.
    ///
    /// Heights are blended by how much nearer the nearest fragment is than each other,
    /// in squared distance, which is the same seen from either side of an edge, so the
    /// surface never tears where the nearest fragment changes.
    fn plates(&self, e: Vec3, cell: f32, rise: f32, salt: u32, bedded: f32, level: impl Fn(Vec3) -> f32) -> Plates {
        // Bent before the fragments are laid over it, so their edges wander rather than
        // zigzag like a jigsaw's.
        let q = e / cell;
        let bend = Vec3::new(self.noise.at(q * 0.45 + Vec3::splat(17.0)), self.noise.at(q * 0.45 + Vec3::splat(29.0)), self.noise.at(q * 0.45 + Vec3::splat(41.0)));
        let q = q + bend * 0.9;
        let mut found = [(Vec3::ZERO, 0.0f32); 27];
        let mut n = 0;
        cells(q, salt, |at, _| {
            found[n] = (at, (q - at).length_squared());
            n += 1;
        });
        let nearest = (0..n).min_by(|&a, &b| found[a].1.total_cmp(&found[b].1)).unwrap_or(0);
        let (c1, d1) = found[nearest];
        let jitter = |k: f32| unit(mix(salt ^ (k as i32 as u32).wrapping_mul(0x9E37_79B9)));
        let height = |k: f32| (1.0 - bedded) * (k + 0.6 * (jitter(k) - 0.5)) + bedded * 2.0 * jitter(k);
        let l1 = level(c1 * cell).floor();
        let h1 = height(l1);
        // A step's rise, in the squared distance its fragments differ by.
        let reach = 2.0 * (rise / cell).max(1e-3);
        let (mut sum, mut total, mut tint) = (h1, 1.0f32, jitter(l1 + 7919.0));
        let mut top = if l1 >= 0.0 { 1.0f32 } else { 0.0 };
        let (mut foot, mut lip) = (0.0f32, 0.0f32);
        for (j, &(cj, dj)) in found[..n].iter().enumerate() {
            if j == nearest {
                continue;
            }
            let u = (dj - d1) / reach;
            if u >= 1.0 {
                continue;
            }
            let w = (1.0 - u) * (1.0 - u);
            let lj = level(cj * cell).floor();
            let hj = height(lj);
            sum += w * hj;
            total += w;
            tint += w * jitter(lj + 7919.0);
            top += if lj >= 0.0 { w } else { 0.0 };
            let step = smoothstep(0.1, 0.5, (hj - h1).abs()) * w.sqrt();
            if hj > h1 {
                foot = foot.max(step);
            } else {
                lip = lip.max(step);
            }
        }
        Plates { h: sum / total, tint: tint / total, top: top / total, foot, lip }
    }

    /// `e` squeezed across the foliation and drawn out along it, so noise read there
    /// runs in parallel lines.
    fn foliated(&self, e: Vec3) -> Vec3 {
        let f = self.surface.foliation.clamp(0.0, 1.0);
        if f <= 0.0 {
            return e;
        }
        let n = self.foliate;
        let across = e.dot(n);
        (e - n * across) / (1.0 + 0.8 * f) + n * across * (1.0 + 3.0 * f)
    }

    /// Whether the weathered crust has held at `c`, metres from the middle: above 0 it
    /// has, below it has flaked away to the stone. Mostly on the faces turned up to the
    /// weather, in broad tongues, and broken at its edge fragment by fragment, `cell`
    /// metres across, so bare islands open in it and crusted ones stand off it.
    fn crust_field(&self, c: Vec3, cell: f32) -> f32 {
        let up = (c / (self.half * self.half)).normalize_or(Vec3::Y).y;
        let broad = self.noise2.fbm(c / self.radius * 1.4 + Vec3::new(5.0, 0.0, 0.0), 3);
        let broken = self.noise.at(c / cell * 0.8 + Vec3::splat(61.0));
        0.5 * up + 1.0 * broad + 1.2 * (self.shape.flaking - 0.5) - 0.3 + 0.35 * broken
    }

    /// Small pits: how deep in one `p` is, 0 to 1, and how far it is sunk, metres. Many
    /// tiny, a few large, most drawn out one way, and crowded in patches and bands.
    fn pits(&self, e: Vec3, pitting: f32, tx: f32) -> (f32, f32) {
        if pitting <= 0.0 {
            return (0.0, 0.0);
        }
        const CELL: f32 = 0.02;
        let q = e / CELL;
        // Gathered in clusters, as weathering finds the weak spots together, and the
        // clusters themselves in broader bands.
        let cluster = smoothstep(-0.1, 0.5, self.noise.fbm(e * 6.0 + Vec3::splat(33.0), 2))
            * smoothstep(-0.35, 0.25, self.noise2.fbm(e * 1.5 + Vec3::splat(77.0), 2));
        let density = (pitting * (0.08 + 2.2 * cluster)).min(1.0);
        // One wander for all of them here, so the larger are not perfect ovals.
        let ragged = 1.0 + 0.6 * self.noise2.at(q * 1.7 + Vec3::splat(5.0));
        let (mut best, mut depth) = (0.0f32, 0.0f32);
        cells(q, self.salt ^ SALT_PITS, |at, h| {
            if unit(h) >= density {
                return;
            }
            let rho = 0.04 + 0.34 * unit(rehash(h, 5)).powi(3);
            // A pit too small for the texels fades rather than flickers.
            let fade = smoothstep(1.0, 2.0, rho * CELL / tx);
            let along = Vec3::new(unit(rehash(h, 6)) - 0.5, unit(rehash(h, 7)) - 0.5, unit(rehash(h, 8)) - 0.5).normalize_or(Vec3::X);
            let stretch = 1.0 + 1.6 * unit(rehash(h, 9));
            let v = q - at;
            let a = v.dot(along);
            let d2 = (v.length_squared() - a * a * (1.0 - 1.0 / (stretch * stretch))) / (rho * rho) * ragged;
            if d2 < 1.0 && fade > 0.0 {
                let v = (1.0 - d2).sqrt() * fade;
                if v > best {
                    best = v;
                    depth = v * rho * CELL * 0.5;
                }
            }
        });
        (best, depth)
    }

    /// The point on the surface in unit direction `d`, metres from the rock's middle,
    /// before it is lifted onto its origin.
    fn point(&self, d: Vec3, texel: Option<f32>) -> Vec3 {
        d * self.sample(d, texel).r * self.half
    }

    /// The direction from the middle a point near the surface lies in.
    fn direction_of(&self, p: Vec3) -> Vec3 {
        (p / self.half).normalize_or(Vec3::Y)
    }

    /// How far a point from `point` is moved to stand on the rock's origin: its flat
    /// bottom on the ground, sunk by `bury`.
    fn lift(&self) -> f32 {
        let flat = self.shape.flat_bottom.clamp(0.0, 0.9);
        let height = self.half.y * (1.0 + flat);
        flat * self.half.y - self.shape.bury.clamp(0.0, 0.9) * height
    }

    /// The point, outward normal and sample in direction `d`, the normal found by
    /// stepping `e` round the direction either way.
    fn point_normal(&self, d: Vec3, texel: Option<f32>, e: f32) -> (Vec3, Vec3, Sample) {
        let a = if d.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let t1 = d.cross(a).normalize();
        let t2 = d.cross(t1);
        let s0 = self.sample(d, texel);
        let p0 = d * s0.r * self.half;
        let p1 = self.point((d + t1 * e).normalize(), texel);
        let p2 = self.point((d + t2 * e).normalize(), texel);
        let mut n = (p1 - p0).cross(p2 - p0).normalize_or(d);
        if n.dot(p0) < 0.0 {
            n = -n;
        }
        (p0, n, s0)
    }
}

/// The ellipsoid cut by planes, softened where they meet: a smooth minimum, `k` sharp a
/// metre, of the distances out along `d` to the ellipsoid and to each plane. Worked in
/// metres, so an edge is rounded as much across a slab's top as down its side.
struct Cut {
    /// Distance out, as a share of the half extents in direction `d`.
    r: f32,
    /// How near an edge between faces `d` is, 0 on a face and 1 on the edge, over a
    /// band wider than the rounding.
    edge: f32,
    /// Which way the faces there look, blended as they are across an edge.
    facing: Vec3,
    /// How far `d` is on a chip's scar, 0 to 1.
    chip: f32,
}

/// The rock's faces cut from the ellipsoid `half` across, and its chips; `fine` takes the
/// chips only the maps carry as well.
fn polytope(planes: &[(Vec3, f32)], chips: &[Chip], fine: bool, d: Vec3, half: Vec3, radius: f32, k: f32) -> Cut {
    let e = d * half;
    let len = e.length().max(1e-6);
    let u = e / len;
    // The ellipsoid, the facets and the flat top and bottom, then the chips.
    let mut dists = [(len, (d / half).normalize_or(u), false); MAX_FACETS as usize + 3 + MAX_CHIPS];
    let mut n = 1;
    for &(normal, offset) in planes {
        let c = u.dot(normal);
        if c > 1e-3 {
            dists[n] = (offset / c, normal, false);
            n += 1;
        }
    }
    let lo = dists[..n].iter().map(|x| x.0).fold(f32::MAX, f32::min);
    let (mut sum, mut near, mut facing) = (0.0f32, 0.0f32, Vec3::ZERO);
    for &(t, normal, _) in &dists[..n] {
        let w = (-(t - lo) * k).exp();
        sum += w;
        facing += normal * w;
        near += (-(t - lo) * k * 0.6).exp();
    }
    let worn = lo - sum.ln() / k;
    let facing = facing.normalize_or(u);

    // The chips, each its scar's plane, curving up out of the stone toward the ends of
    // its reach, so it tapers away there as a flake's scar does rather than stopping at
    // a wall.
    let mut m = 0;
    for chip in chips {
        let c = u.dot(chip.normal);
        let out = (1.0 - d.dot(chip.at)) / (1.0 - chip.reach).max(1e-6);
        if (fine || !chip.fine) && c > 0.05 && out < 1.5 {
            // Past its reach it lifts right away, so it has no say by the time it is
            // dropped.
            let lift = chip.depth * 1.3 * out + (out - 1.0).max(0.0).powi(2) * 0.5 * radius;
            dists[m] = (chip.offset / c + lift, chip.normal, true);
            m += 1;
        }
    }
    // A chip meets the worn faces crisply, by a much sharper minimum of its own, so its
    // plane lying just above the stone does not pull it in.
    let crisp = 90.0 / radius;
    let lo2 = dists[..m].iter().map(|x| x.0).fold(worn, f32::min);
    let base = (-(worn - lo2) * crisp).exp();
    let (mut sum2, mut scar, mut turned) = (base, 0.0f32, facing * base);
    for &(t, normal, _) in &dists[..m] {
        let w = (-(t - lo2) * crisp).exp();
        sum2 += w;
        scar += w;
        turned += normal * w;
    }
    Cut {
        r: (lo2 - sum2.ln() / crisp) / len,
        edge: (near - 1.0).clamp(0.0, 1.0),
        facing: turned.normalize_or(facing),
        chip: (scar / sum2).clamp(0.0, 1.0),
    }
}

/// The cube's faces: the axis each is centred on, and the directions its u and v run.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
];
/// The underside's face.
const UNDERSIDE: usize = 3;

fn face_axes() -> [(Vec3, Vec3); 6] {
    FACES.map(|(_, u, v)| (Vec3::from(u), Vec3::from(v)))
}

/// The direction through a point on the cube, each axis from -1 to 1, spread by tangent
/// so the cube's squares land near even in size over the sphere.
fn cube_dir(c: Vec3) -> Vec3 {
    Vec3::new((c.x * FRAC_PI_4).tan(), (c.y * FRAC_PI_4).tan(), (c.z * FRAC_PI_4).tan()).normalize()
}

/// The direction at `(s, t)` on face `f`, both from -1 to 1 on the face and running on
/// past it.
fn face_dir(f: usize, s: f32, t: f32) -> Vec3 {
    let (m, u, v) = FACES[f];
    (Vec3::from(m) + Vec3::from(u) * (s * FRAC_PI_4).tan() + Vec3::from(v) * (t * FRAC_PI_4).tan()).normalize()
}

// ---------------------------------------------------------------------------------
// The sheet

/// Where each chart lies in the sheet of maps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RockAtlas {
    /// Each face's rectangle, x, y, width and height from 0 to 1, its pad included.
    pub rects: [[f32; 4]; 6],
    /// Faces laid in turned, their u down the sheet and their v across it.
    pub turned: [bool; 6],
    /// Sheet widths a metre of each face's surface gets: its texel density, whatever
    /// size the sheet is baked at.
    pub per_metre: [f32; 6],
}

impl RockAtlas {
    /// Packs faces `extents` metres across and down into one square sheet, as large as
    /// will fit, a metre of each getting as much of the sheet as any other. With six
    /// faces every order is tried, each face as it is, or all turned to lie long or all
    /// to stand tall.
    fn pack(extents: [(f32, f32); 6]) -> Self {
        let share = |f: usize| if f == UNDERSIDE { UNDERSIDE_SHARE } else { 1.0 };
        let orders = permutations();
        let turnings: [[bool; 6]; 3] = [
            [false; 6],
            std::array::from_fn(|f| extents[f].1 > extents[f].0),
            std::array::from_fn(|f| extents[f].0 > extents[f].1),
        ];
        let fit = |k: f32| {
            turnings.iter().find_map(|turned| {
                let sizes: [(f32, f32); 6] = std::array::from_fn(|f| {
                    let (w, h) = if turned[f] { (extents[f].1, extents[f].0) } else { extents[f] };
                    (k * share(f) * w + 2.0 * ATLAS_PAD, k * share(f) * h + 2.0 * ATLAS_PAD)
                });
                orders.iter().find_map(|order| shelf(&sizes, order)).map(|at| (at, sizes, *turned))
            })
        };
        let (mut lo, mut hi) = (0.0f32, 64.0f32);
        let mut best = fit(lo).expect("the pads alone fit");
        for _ in 0..32 {
            let k = 0.5 * (lo + hi);
            match fit(k) {
                Some(found) => {
                    lo = k;
                    best = found;
                }
                None => hi = k,
            }
        }
        let (at, sizes, turned) = best;
        Self {
            rects: std::array::from_fn(|f| [at[f].0, at[f].1, sizes[f].0, sizes[f].1]),
            turned,
            per_metre: std::array::from_fn(|f| lo * share(f)),
        }
    }

    /// Where `st` on face `f` lands in the sheet.
    fn uv(&self, f: usize, st: Vec2) -> [f32; 2] {
        let st = if self.turned[f] { Vec2::new(st.y, st.x) } else { st };
        let [x, y, w, h] = self.rects[f];
        [
            x + ATLAS_PAD + (st.x + 1.0) * 0.5 * (w - 2.0 * ATLAS_PAD),
            y + ATLAS_PAD + (st.y + 1.0) * 0.5 * (h - 2.0 * ATLAS_PAD),
        ]
    }

    /// Where a point of the sheet lands on face `f`, running on past its edges.
    fn st(&self, f: usize, uv: Vec2) -> Vec2 {
        let [x, y, w, h] = self.rects[f];
        let st = Vec2::new(
            (uv.x - x - ATLAS_PAD) / (w - 2.0 * ATLAS_PAD) * 2.0 - 1.0,
            (uv.y - y - ATLAS_PAD) / (h - 2.0 * ATLAS_PAD) * 2.0 - 1.0,
        );
        if self.turned[f] { Vec2::new(st.y, st.x) } else { st }
    }

    /// The face whose rectangle holds a point of the sheet.
    fn chart_at(&self, uv: Vec2) -> Option<usize> {
        self.rects.iter().position(|&[x, y, w, h]| uv.x >= x && uv.x < x + w && uv.y >= y && uv.y < y + h)
    }
}

/// Lays rectangles into the unit square in rows, in `order`, or `None` if they do not
/// fit.
fn shelf(sizes: &[(f32, f32); 6], order: &[usize; 6]) -> Option<[(f32, f32); 6]> {
    let mut at = [(0.0, 0.0); 6];
    let (mut x, mut y, mut row) = (0.0f32, 0.0f32, 0.0f32);
    for &f in order {
        let (w, h) = sizes[f];
        if w > 1.0 {
            return None;
        }
        if x + w > 1.0 {
            y += row;
            x = 0.0;
            row = 0.0;
        }
        if y + h > 1.0 {
            return None;
        }
        at[f] = (x, y);
        x += w;
        row = row.max(h);
    }
    Some(at)
}

/// Every order of six things, by Heap's algorithm.
fn permutations() -> Vec<[usize; 6]> {
    let mut out = Vec::with_capacity(720);
    let mut a = [0, 1, 2, 3, 4, 5];
    let mut c = [0usize; 6];
    out.push(a);
    let mut i = 0;
    while i < 6 {
        if c[i] < i {
            if i % 2 == 0 {
                a.swap(0, i);
            } else {
                a.swap(c[i], i);
            }
            out.push(a);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    out
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

/// A whole rock: the mesh of each LOD, and where its charts lie in the maps they share.
#[derive(Clone, Debug, Default)]
pub struct RockMesh {
    pub lods: Vec<RockLodMesh>,
    pub atlas: RockAtlas,
    /// Bounds of LOD0.
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// The surface sampled finely through the cube: every vertex once, however many faces
/// of the cube share it.
struct Grid {
    n: u32,
    /// Each vertex's point on the cube, each axis from -1 to 1, and on the rock.
    cube: Vec<Vec3>,
    positions: Vec<Vec3>,
    /// The vertex at each face's grid point, face by face, row by row.
    index: Vec<u32>,
}

impl Grid {
    fn new(surface: &Surface, n: u32) -> Self {
        let mut keys = std::collections::HashMap::new();
        let mut cube = Vec::new();
        let mut index = Vec::with_capacity(6 * ((n + 1) * (n + 1)) as usize);
        let half = n as f32 * 0.5;
        for (m, u, v) in FACES {
            let (m, u, v) = (Vec3::from(m), Vec3::from(u), Vec3::from(v));
            for j in 0..=n {
                for i in 0..=n {
                    let (s, t) = (i as f32 / half - 1.0, j as f32 / half - 1.0);
                    let c = m + u * s + v * t;
                    // Whole lattice steps, so faces meeting at an edge find the same key.
                    let key = ((c + 1.0) * half).round().as_ivec3();
                    let id = *keys.entry(key).or_insert_with(|| {
                        cube.push(key.as_vec3() / half - 1.0);
                        cube.len() as u32 - 1
                    });
                    index.push(id);
                }
            }
        }
        let positions = par_map(cube.len(), |k| surface.point(cube_dir(cube[k]), None));
        Self { n, cube, positions, index }
    }

    fn at(&self, f: usize, i: u32, j: u32) -> u32 {
        let row = self.n + 1;
        self.index[f * (row * row) as usize + (j * row + i) as usize]
    }

    /// Each face's surface, metres across and down: the mean length of its rows and of
    /// its columns.
    fn extents(&self) -> [(f32, f32); 6] {
        let n = self.n;
        std::array::from_fn(|f| {
            let p = |i, j| self.positions[self.at(f, i, j) as usize];
            let (mut across, mut down) = (0.0, 0.0);
            for a in 0..=n {
                for b in 0..n {
                    across += (p(b + 1, a) - p(b, a)).length();
                    down += (p(a, b + 1) - p(a, b)).length();
                }
            }
            (across / (n + 1) as f32, down / (n + 1) as f32)
        })
    }

    /// The triangles, each face's wound to face out, cut along the shorter diagonal of
    /// each square. Triangles under the ground count for less, so the budget goes where
    /// the rock is seen.
    fn source(self, lift: f32, height: f32) -> Source {
        let n = self.n;
        let mut triangles = Vec::with_capacity(12 * (n * n) as usize);
        let mut charts = Vec::with_capacity(triangles.capacity());
        for (f, (m, u, v)) in FACES.into_iter().enumerate() {
            let outward = Vec3::from(u).cross(Vec3::from(v)).dot(Vec3::from(m)) > 0.0;
            for j in 0..n {
                for i in 0..n {
                    let (a, b, c, d) = (self.at(f, i, j), self.at(f, i + 1, j), self.at(f, i, j + 1), self.at(f, i + 1, j + 1));
                    let pos = |k: u32| self.positions[k as usize];
                    let pair = if (pos(a) - pos(d)).length_squared() < (pos(b) - pos(c)).length_squared() {
                        [[a, b, d], [a, d, c]]
                    } else {
                        [[a, b, c], [b, d, c]]
                    };
                    for mut tri in pair {
                        if !outward {
                            tri.swap(1, 2);
                        }
                        triangles.push(tri);
                        charts.push(f as u8);
                    }
                }
            }
        }
        let weights = triangles
            .iter()
            .map(|tri| {
                let y = tri.iter().map(|&k| self.positions[k as usize].y).sum::<f32>() / 3.0 + lift;
                0.1 + 0.9 * smoothstep(-0.08 * height, 0.0, y)
            })
            .collect();
        Source { positions: self.positions, cube: self.cube, triangles, charts, weights }
    }
}

/// Quads along each cube face edge of the fine mesh the LODs are cut down from: about
/// twenty source triangles to each one LOD0 keeps.
fn source_resolution(finest: u32) -> u32 {
    ((finest as f32 * 1.6).sqrt().ceil() as u32).clamp(MIN_SOURCE, MAX_SOURCE)
}

/// Builds the rock a preset's seed lands on. Deterministic.
pub fn build_rock(p: &RockParams) -> RockMesh {
    let surface = Surface::new(p);
    let lift = surface.lift();
    let targets: Vec<(u32, f32)> = p
        .lod
        .iter()
        .take(MAX_LODS)
        .map(|l| (l.triangles.clamp(MIN_TRIANGLES, MAX_TRIANGLES), l.screen_size.max(0.0)))
        .collect();
    let n = source_resolution(targets.first().map_or(1600, |t| t.0));
    let grid = Grid::new(&surface, n);
    let atlas = RockAtlas::pack(grid.extents());
    let mut simplifier = Simplifier::new(grid.source(lift, surface.half.y * 2.0), face_axes());
    let lods: Vec<RockLodMesh> = targets
        .into_iter()
        .map(|(triangles, screen_size)| {
            simplifier.simplify_to(triangles as usize);
            emit(&simplifier, &atlas, lift, screen_size)
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
    RockMesh { lods, atlas, min, max }
}

/// One LOD as it stands: a vertex for each chart each kept vertex is in, normals shared
/// across the seams, and tangents along each chart's u.
fn emit(simplifier: &Simplifier, atlas: &RockAtlas, lift: f32, screen_size: f32) -> RockLodMesh {
    let lift = Vec3::Y * lift;
    let mut out = RockLodMesh { screen_size, ..Default::default() };
    let mut slot = vec![u32::MAX; simplifier.vertex_count() * 6];
    let mut from = Vec::new();
    for (tri, chart) in simplifier.triangles() {
        for v in tri {
            let s = &mut slot[v as usize * 6 + chart as usize];
            if *s == u32::MAX {
                *s = out.positions.len() as u32;
                from.push(v);
                out.positions.push((simplifier.position(v) + lift).to_array());
                out.uvs.push(atlas.uv(chart as usize, simplifier.chart_st(v, chart)));
            }
            out.indices.push(*s);
        }
    }

    // Normals: each face's, weighted by the angle it makes at the corner.
    let mut normal = vec![Vec3::ZERO; simplifier.vertex_count()];
    for (tri, _) in simplifier.triangles() {
        let p = tri.map(|v| simplifier.position(v));
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        for k in 0..3 {
            let (a, b) = (p[(k + 1) % 3] - p[k], p[(k + 2) % 3] - p[k]);
            normal[tri[k] as usize] += n * a.angle_between(b);
        }
    }
    out.normals = from.iter().map(|&v| normal[v as usize].normalize_or(Vec3::Y).to_array()).collect();

    // Tangents: how the surface runs along u and along v, from each triangle's.
    let mut du = vec![Vec3::ZERO; from.len()];
    let mut dv = vec![Vec3::ZERO; from.len()];
    for tri in out.indices.chunks_exact(3) {
        let p = [0, 1, 2].map(|k| Vec3::from(out.positions[tri[k] as usize]));
        let uv = [0, 1, 2].map(|k| Vec2::from(out.uvs[tri[k] as usize]));
        let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
        let (d1, d2) = (uv[1] - uv[0], uv[2] - uv[0]);
        let det = d1.perp_dot(d2);
        if det.abs() < 1e-14 {
            continue;
        }
        let t = (e1 * d2.y - e2 * d1.y) / det;
        let b = (e2 * d1.x - e1 * d2.x) / det;
        for &k in tri {
            du[k as usize] += t;
            dv[k as usize] += b;
        }
    }
    out.tangents = (0..from.len())
        .map(|k| {
            let n = Vec3::from(out.normals[k]);
            let t = (du[k] - n * du[k].dot(n)).try_normalize().unwrap_or_else(|| n.any_orthonormal_vector());
            // Up the texture is toward smaller v.
            let w = if n.cross(t).dot(-dv[k]) >= 0.0 { 1.0 } else { -1.0 };
            [t.x, t.y, t.z, w]
        })
        .collect();
    out
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

/// Paints a rock's maps at the size its preset asks for.
pub fn bake_rock(p: &RockParams, mesh: &RockMesh) -> RockMaps {
    bake_rock_at(p, mesh, p.texture.size())
}

/// What LOD0 puts under one texel.
#[derive(Clone, Copy)]
struct Texel {
    /// The face whose chart it is in, or `NO_CHART`.
    chart: u8,
    /// Whether a triangle covers it, or it only borrows the frame of one nearby.
    covered: bool,
    framed: bool,
    /// The point of LOD0 there, before the lift, and the frame the renderer
    /// interpolates.
    point: Vec3,
    normal: Vec3,
    tangent: Vec3,
    sign: f32,
}

const NO_CHART: u8 = u8::MAX;

/// Paints a rock's maps `size` texels a side, for a preview that needs them sooner.
pub fn bake_rock_at(p: &RockParams, mesh: &RockMesh, size: u32) -> RockMaps {
    let surface = Surface::new(p);
    let size = size.clamp(MIN_TEXTURE, MAX_TEXTURE);
    let texels = frames(mesh, surface.lift(), size);
    let painter = Painter::new(&surface, p);
    let rows = par_map(size as usize, |y| {
        (0..size)
            .map(|x| {
                let t = &texels[y * size as usize + x as usize];
                let uv = Vec2::new((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
                painter.texel(t, uv, &mesh.atlas, size)
            })
            .collect::<Vec<_>>()
    });
    let mut maps = RockMaps { albedo: Bitmap::new(size, size), normal: Bitmap::new(size, size), orm: Bitmap::new(size, size) };
    for (y, row) in rows.into_iter().enumerate() {
        for (x, [a, n, o]) in row.into_iter().enumerate() {
            maps.albedo.put(x as u32, y as u32, a);
            maps.normal.put(x as u32, y as u32, n);
            maps.orm.put(x as u32, y as u32, o);
        }
    }
    maps
}

/// LOD0 laid out in the sheet: under each texel of each triangle, the point and frame
/// there; then each frame carried out over the pad round its chart, so the pad's normals
/// are told against a frame that continues its chart's.
fn frames(mesh: &RockMesh, lift: f32, size: u32) -> Vec<Texel> {
    let n = size as usize;
    let atlas = &mesh.atlas;
    let blank = Texel {
        chart: NO_CHART,
        covered: false,
        framed: false,
        point: Vec3::ZERO,
        normal: Vec3::Y,
        tangent: Vec3::X,
        sign: 1.0,
    };
    let mut texels = vec![blank; n * n];
    for y in 0..n {
        for x in 0..n {
            let uv = Vec2::new((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
            if let Some(f) = atlas.chart_at(uv) {
                texels[y * n + x].chart = f as u8;
            }
        }
    }
    let Some(lod) = mesh.lods.first() else { return texels };
    let scale = size as f32;
    for tri in lod.indices.chunks_exact(3) {
        let uv = [0, 1, 2].map(|k| Vec2::from(lod.uvs[tri[k] as usize]) * scale);
        let area = (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]);
        if area.abs() < 1e-9 {
            continue;
        }
        let Some(chart) = atlas.chart_at((uv[0] + uv[1] + uv[2]) / (3.0 * scale)) else { continue };
        let lo = uv[0].min(uv[1]).min(uv[2]).floor().max(Vec2::ZERO);
        let hi = uv[0].max(uv[1]).max(uv[2]).ceil().min(Vec2::splat(scale - 1.0));
        let p = [0, 1, 2].map(|k| Vec3::from(lod.positions[tri[k] as usize]) - Vec3::Y * lift);
        let nr = [0, 1, 2].map(|k| Vec3::from(lod.normals[tri[k] as usize]));
        let tg = [0, 1, 2].map(|k| {
            let t = lod.tangents[tri[k] as usize];
            Vec3::new(t[0], t[1], t[2])
        });
        let sign = lod.tangents[tri[0] as usize][3];
        for y in lo.y as usize..=hi.y as usize {
            for x in lo.x as usize..=hi.x as usize {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let w0 = (uv[2] - uv[1]).perp_dot(c - uv[1]) / area;
                let w1 = (uv[0] - uv[2]).perp_dot(c - uv[2]) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                    continue;
                }
                let t = &mut texels[y * n + x];
                if t.chart != chart as u8 {
                    continue;
                }
                *t = Texel {
                    chart: chart as u8,
                    covered: true,
                    framed: true,
                    point: p[0] * w0 + p[1] * w1 + p[2] * w2,
                    normal: nr[0] * w0 + nr[1] * w1 + nr[2] * w2,
                    tangent: tg[0] * w0 + tg[1] * w1 + tg[2] * w2,
                    sign,
                };
            }
        }
    }
    // Out over the pad, breadth first, never across into another chart.
    let mut queue: std::collections::VecDeque<usize> = (0..n * n).filter(|&i| texels[i].framed).collect();
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % n, i / n);
        let from = texels[i];
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if nx < 0 || ny < 0 || nx >= n as i32 || ny >= n as i32 {
                continue;
            }
            let j = ny as usize * n + nx as usize;
            let t = &mut texels[j];
            if !t.framed && t.chart == from.chart {
                t.framed = true;
                t.normal = from.normal;
                t.tangent = from.tangent;
                t.sign = from.sign;
                queue.push_back(j);
            }
        }
    }
    texels
}

/// Which lichen a rosette is.
#[derive(Clone, Copy)]
enum Lichen {
    Pale,
    Dark,
    Rare,
}

/// The colours, in linear light, and what else painting needs that does not change from
/// texel to texel.
struct Painter<'a> {
    surface: &'a Surface,
    colour: &'a RockColour,
    moss: &'a RockMoss,
    stone: Vec3,
    warm: Vec3,
    lichen: Vec3,
    lichen_rare: Vec3,
    feldspar: Vec3,
    dirt: Vec3,
    moss_c: Vec3,
    moss_tip: Vec3,
    moss_dry: Vec3,
    lift: f32,
    height: f32,
}

impl<'a> Painter<'a> {
    fn new(surface: &'a Surface, p: &'a RockParams) -> Self {
        let c = &p.colour;
        let lin = |c: [f32; 3]| Vec3::from(c.map(srgb_to_linear));
        Self {
            surface,
            colour: c,
            moss: &p.moss,
            stone: lin(c.stone),
            warm: lin(c.warm),
            lichen: lin(c.lichen_color),
            lichen_rare: lin(c.lichen_rare),
            feldspar: lin(c.feldspar),
            dirt: lin(c.dirt_color),
            moss_c: lin(c.moss),
            moss_tip: lin(c.moss_tip),
            moss_dry: lin(c.moss_dry),
            lift: surface.lift(),
            height: surface.half.y * (1.0 + surface.shape.flat_bottom.clamp(0.0, 0.9)),
        }
    }

    /// Albedo, normal and packed occlusion and roughness for one texel.
    fn texel(&self, t: &Texel, uv: Vec2, atlas: &RockAtlas, size: u32) -> [[u8; 4]; 3] {
        let s = self.surface;
        if t.chart == NO_CHART || !t.framed {
            // Sheet no chart uses: plain stone, so the smallest mips are not pulled
            // toward black.
            let a = self.stone.to_array().map(|v| to_u8(linear_to_srgb(v)));
            return [[a[0], a[1], a[2], 255], [128, 128, 255, 255], [255, to_u8(self.colour.roughness), 0, 255]];
        }
        let f = t.chart as usize;
        // The pad repeats its chart's edge outward, as the edge's frame is carried out:
        // the next face on past the seam turns too far from that frame to be told in it.
        let d = if t.covered {
            s.direction_of(t.point)
        } else {
            let st = atlas.st(f, uv).clamp(Vec2::NEG_ONE, Vec2::ONE);
            face_dir(f, st.x, st.y)
        };
        let texel = 1.0 / (atlas.per_metre[f] * size as f32).max(1e-6);

        // The frame the renderer builds from what it interpolates.
        let nf = t.normal.normalize_or(d);
        let tf = (t.tangent - nf * t.tangent.dot(nf)).try_normalize().unwrap_or_else(|| nf.any_orthonormal_vector());
        let bf = nf.cross(tf) * t.sign;

        let step = texel / s.point(d, None).length().max(1e-3) * 0.7;
        let (p, n_hi, smp) = s.point_normal(d, Some(texel), step);
        let hollow = self.hollow(d, nf);
        let c = self.colour;
        let r = s.radius;
        let n1 = &s.noise;
        let n2 = &s.noise2;
        let world_y = p.y + self.lift;
        let height01 = ((world_y) / self.height.max(1e-3)).clamp(0.0, 1.0);
        let cavity = s.surface.cavity.max(0.0);

        // Stone, weathered to a warmer crust on the plates standing proud and back to the
        // stone where they have flaked away; each plate its own shade.
        // The crust's edge is broken up where it has half flaked, not drawn round the plate.
        // It keeps mostly to the top, where the weather has had longest at it.
        let crust = match smp.crust {
            // Where the crust decides the plates, it is its colour too, only thinned here
            // and there where it is wearing through.
            Some(held) => held * (1.0 - 0.35 * smoothstep(0.1, 0.6, n1.fbm_limited(p + Vec3::splat(12.0), 6.0, texel, 3))),
            None => smoothstep(
                0.45,
                0.85,
                0.35 * smp.plate + 0.45 * nf.y + 0.3 * n2.fbm(p * (0.9 / r) + Vec3::new(5.0, 0.0, 0.0), 3) + 0.15 * n1.fbm_limited(p + Vec3::splat(12.0), 15.0, texel, 3),
            ),
        };
        let mut col = self.stone.lerp(self.warm, crust) * (1.0 + c.tone * 0.6 * n1.fbm(p * (0.9 / r), 3));
        col *= 1.0 + c.tone * 0.35 * (smp.tint - 0.5);
        // Cooler and warmer by turns over patches half a metre across, as a stone's
        // minerals are not spread evenly through it.
        let hue = n2.fbm(p * 2.2 + Vec3::new(0.0, 0.0, 57.0), 2);
        col *= Vec3::ONE + c.tone * 0.22 * hue * Vec3::new(1.0, 0.1, -0.9);
        // Mottled at every scale from a hand across down to a few millimetres.
        let mottle = n1.fbm_limited(p + Vec3::new(0.0, 3.0, 0.0), 7.0, texel, 5);
        col *= 1.0 + c.grain * 1.4 * mottle;
        // Mineral grains, a few millimetres, packed edge to edge; dimmer under the crust.
        let speckle = c.speckle.clamp(0.0, 1.0) * (1.0 - 0.3 * crust);
        // Two sizes of them, so they do not read as one even pattern.
        let size = c.mineral_size.max(0.001);
        col = self.minerals(p, col, speckle, size, SALT_MINERALS, texel);
        col = self.minerals(p, col, speckle * 0.7, size * 0.45, SALT_MINERALS ^ 0x55, texel);
        // Weathered stains, darker, in blotches a hand or two across.
        let stain = smoothstep(0.1, 0.55, n2.fbm(p * 3.5 + Vec3::new(0.0, 17.0, 0.0), 4));
        col *= 1.0 - 0.2 * c.tone * stain;
        // A green film of algae low on the shaded sides, where the stone stays damp.
        let damp = (1.0 - nf.y.max(0.0)) * (1.0 - smoothstep(0.1, 0.6, height01))
            * smoothstep(-0.2, 0.4, n1.fbm(p * 2.0 + Vec3::new(8.0, 0.0, 3.0), 3));
        col = col.lerp(col * Vec3::new(0.8, 0.95, 0.72), damp * 0.6);
        if s.surface.strata > 0.0 {
            col *= 1.0 + 0.3 * s.surface.strata.min(1.5) * smp.band;
        }
        // Worn edges pale, and the lips of the plates; their steps in shadow and grime.
        col *= 1.0 + c.edge_wear * (0.35 * smp.edge + 0.15 * smp.lip);
        // Fresh stone where it has chipped, paler and warmer.
        col = col.lerp(col * Vec3::new(1.2, 1.16, 1.1), smp.chip * 0.8);
        // Water down the steep faces.
        let steep = 1.0 - nf.y.abs();
        let streak = smoothstep(0.1, 0.6, n2.fbm(Vec3::new(p.x * 9.0, p.y * 0.9, p.z * 9.0), 3)) * c.streaks * steep;
        col *= 1.0 - 0.55 * streak;

        // Lichen: rosettes where the stone is exposed and settled, not on fresh breaks,
        // in the cracks or near the soil.
        let colony = smoothstep(-0.35, 0.35, n1.fbm(p * (1.1 / r) + Vec3::new(0.0, 21.0, 0.0), 3));
        let exposure = (0.55 + 0.45 * nf.y.max(0.0)) * (1.0 - smp.chip) * (1.0 - smp.crack) * smoothstep(0.03, 0.15, world_y);
        let (lichen, kind, rel) = self.lichen(p, c.lichen * colony * exposure, texel);
        if lichen > 0.0 {
            // Pale crusts mostly, some dark, a few of the rare colour; each paler at its
            // growing rim, and cracked into a crust.
            let tone = match kind {
                Lichen::Pale => self.lichen,
                Lichen::Dark => col * 0.62,
                Lichen::Rare => self.lichen_rare,
            } * (0.8 + 0.3 * rel);
            // Thin at the heart, where it is oldest, the stone showing through; thicker
            // toward its rim.
            let crust = 1.0 + 0.3 * n2.fbm_limited(p, 60.0, texel, 3);
            col = col.lerp(tone * crust, lichen * (0.5 + 0.3 * rel));
        }

        // Soil up the foot, and in the cracks and pits; then the cracks' own shadow.
        let splash = 0.07 + 0.12 * (n1.fbm(p * 4.0 + Vec3::splat(9.0), 2) * 0.5 + 0.5);
        let ground = 1.0 - smoothstep(0.0, splash, world_y);
        let grime = (0.9 * ground + 0.6 * smp.crack + 0.35 * smp.pit + 0.15 * smp.foot + 0.3 * smp.tread * (1.0 - nf.y.abs())).min(1.0);
        col = col.lerp(self.dirt * (0.8 + 0.3 * mottle), c.dirt.clamp(0.0, 1.0) * grime);
        // What is in shadow in the stone is dark in its colour too, as a photograph of it
        // would be: the cracks, pits and hollows, and the feet of the plates' steps.
        let shade = (0.7 * smp.crack + 0.3 * smp.pit + 0.5 * smp.hollow + 0.2 * smp.foot).min(1.0);
        col *= 1.0 - cavity.min(1.5) * 0.6 * shade;
        // And the knobs of the fine relief paler, weathered and dusty where they stand.
        col *= 1.0 + cavity.min(1.5) * 0.28 * smp.ridge;
        // The rock's own hollows, a hand or two across, darker and grimier; its swells
        // paler.
        col *= (1.0 - cavity.min(1.5) * 0.35 * hollow.max(0.0)) * (1.0 + 0.2 * (-hollow).max(0.0));

        // Moss.
        let (cover, tuft) = self.moss_at(p, nf.y, &smp, height01, texel);
        let mut albedo = col;
        let mut n = n_hi;
        if cover > 0.0 {
            let m = self.moss;
            let variety = n1.fbm(p * 5.0 + Vec3::new(0.0, 0.0, 30.0), 3);
            let broad = smoothstep(-0.4, 0.4, n2.fbm(p * 1.6 + Vec3::new(0.0, 40.0, 0.0), 3));
            let base = self.moss_c.lerp(self.moss_tip * 0.8, broad * 0.6);
            let mut mc = base * (0.4 + 0.8 * tuft) * (1.0 + 0.3 * variety);
            mc = mc.lerp(self.moss_tip, smoothstep(0.6, 1.0, tuft) * (0.3 + 0.4 * variety.max(0.0)));
            let dried = smoothstep(0.3, 0.8, n2.fbm(p * 3.0 + Vec3::new(11.0, 0.0, 0.0), 3)) * m.dry.clamp(0.0, 1.0);
            mc = mc.lerp(self.moss_dry * (0.75 + 0.35 * tuft), dried * 0.75);
            // Thin at its edge, where it is drier and browner.
            mc = mc.lerp(self.moss_dry * 0.85, (cover * (1.0 - cover) * 4.0) * 0.35);
            // Litter caught in it where it faces up: bits of leaf and needle.
            let litter = self.litter(p, nf.y, texel);
            mc = mc.lerp(self.dirt * 0.9, litter);
            albedo = col.lerp(mc, cover);

            // A cushion over the stone: the stone's own detail smothered under it, and
            // the moss's height told in the normal, its edge a lip.
            let h = |q: Vec3| {
                let (c, t) = self.moss_at(q, nf.y, &smp, height01, texel);
                c * 0.015 * (0.4 + 0.6 * t)
            };
            let h0 = h(p);
            let slope_t = (h(p + tf * texel) - h0) / texel;
            let slope_b = (h(p + bf * texel) - h0) / texel;
            n = (n_hi.lerp(nf, cover * 0.85) - tf * slope_t - bf * slope_b).normalize_or(n_hi);
        }
        let mut tn = Vec3::new(n.dot(tf), n.dot(bf), n.dot(nf)).normalize_or(Vec3::Z);
        // Under a ledge or across a chip's scar the stone can turn right round from the
        // mesh's smooth normal, which a normal map cannot say, and turned far it only goes
        // black: past a slope it is eased back, never nearer edge on than EDGE_ON.
        const EDGE_ON: f32 = 0.12;
        const EASE: f32 = 0.4;
        if tn.z < EASE {
            let z = EDGE_ON + (EASE - EDGE_ON) * ((tn.z - EASE) / (EASE - EDGE_ON)).exp();
            let across = Vec2::new(tn.x, tn.y).normalize_or(Vec2::X) * (1.0 - z * z).sqrt();
            tn = Vec3::new(across.x, across.y, z);
        }

        // Occlusion: cracks, pits, the grain's hollows, between the moss's cushions, and
        // a little at the foot where the ground closes in.
        let mut occl = (1.0 - cavity * shade) * (1.0 - cavity.min(1.5) * 0.45 * hollow.max(0.0));
        occl *= 1.0 - 0.35 * cover * (1.0 - tuft);
        occl *= 0.75 + 0.25 * smoothstep(-0.02, 0.2, world_y);
        let mut rough = c.roughness + 0.05 * mottle - 0.05 * smp.chip - 0.06 * streak + 0.06 * (crust - 0.5);
        rough += (0.92 - rough) * lichen;
        rough += (c.moss_roughness - rough) * cover;

        let a = albedo.to_array().map(|v| to_u8(linear_to_srgb(v)));
        let nm = [to_u8(tn.x * 0.5 + 0.5), to_u8(tn.y * 0.5 + 0.5), to_u8(tn.z * 0.5 + 0.5), 255];
        let o = [to_u8(occl.clamp(0.1, 1.0)), to_u8(rough.clamp(0.0, 1.0)), 0, 255];
        [[a[0], a[1], a[2], 255], nm, o]
    }

    /// `col` with the stone's mineral grains over it: cells about `mineral_size` across,
    /// bent out of round, each dark, pale feldspar, green or the stone itself a shade
    /// lighter or darker, softened at its edge over a texel and a half and faded out
    /// where the grains are too few texels across to draw.
    fn minerals(&self, p: Vec3, col: Vec3, amount: f32, size: f32, salt: u32, texel: f32) -> Vec3 {
        let fade = smoothstep(1.5, 3.0, size / texel) * amount;
        if fade <= 0.0 {
            return col;
        }
        let s = self.surface;
        let q = s.foliated(p) / size;
        let q = q + 0.35 * Vec3::new(s.noise.at(q * 0.5 + Vec3::splat(7.0)), s.noise.at(q * 0.5 + Vec3::splat(19.0)), 0.0);
        let (mut d1, mut d2, mut h1, mut h2) = (f32::MAX, f32::MAX, 0u32, 0u32);
        cells(q, s.salt ^ salt, |at, h| {
            let d = (q - at).length_squared();
            if d < d1 {
                (d2, h2) = (d1, h1);
                (d1, h1) = (d, h);
            } else if d < d2 {
                (d2, h2) = (d, h);
            }
        });
        let green = self.stone * Vec3::new(0.9, 1.08, 0.88);
        let grain = |h: u32| {
            let u = unit(h);
            let jitter = 0.85 + 0.3 * unit(rehash(h, 3));
            jitter * if u < 0.22 {
                col * 0.55
            } else if u < 0.47 {
                self.feldspar * (col / self.stone.max(Vec3::splat(1e-3))).powf(0.5)
            } else if u < 0.53 {
                green * (col / self.stone.max(Vec3::splat(1e-3))).powf(0.5)
            } else {
                col
            }
        };
        // How far into its own cell from the edge it shares with the next.
        let soft = (1.5 * texel / size).max(0.05);
        let w = 0.5 + 0.5 * smoothstep(0.0, soft, d2.sqrt() - d1.sqrt());
        let g = grain(h2).lerp(grain(h1), w);
        col.lerp(g, fade)
    }

    /// How far the mesh's surface in direction `d` lies below the surface round it, at a
    /// hand's breadth and at twice that, the rock's own curving allowed for: 1 in a
    /// hollow, 0 where it runs on even, below 0 on a swell or a lip.
    fn hollow(&self, d: Vec3, normal: Vec3) -> f32 {
        let s = self.surface;
        let here = s.point(d, None);
        let reach = here.length().max(1e-3);
        let a = normal.any_orthonormal_vector();
        let b = normal.cross(a);
        let mut total = 0.0;
        for span in [0.06f32, 0.15] {
            let mut sum = 0.0;
            for k in 0..6 {
                let turn = k as f32 * std::f32::consts::TAU / 6.0;
                let off = (a * turn.cos() + b * turn.sin()) * span;
                let q = s.point(s.direction_of(here + off), None);
                sum += (q - here).dot(normal);
            }
            // Round a sphere as big as the rock, the surface falls away this far anyway.
            let fall = span * span / (2.0 * reach);
            total += ((sum / 6.0 + fall) / (0.15 * span)).clamp(-1.0, 1.0);
        }
        total * 0.5
    }

    /// Lichen at `p` where `amount` of the stone would be crusted: how much, which kind,
    /// and how far out from its rosette's heart, 0 to 1. Rosettes half `lichen_size`
    /// apart and up to about that across, so where they are thick they run together
    /// into one crust with a lobed edge.
    fn lichen(&self, p: Vec3, amount: f32, texel: f32) -> (f32, Lichen, f32) {
        if amount <= 0.0 {
            return (0.0, Lichen::Pale, 0.0);
        }
        let s = self.surface;
        let cell = self.colour.lichen_size.max(0.005) * 0.5;
        let q = p / cell;
        // Softened over at least a texel and a half, so a rosette's rim never jags.
        let soft = (texel / cell * 1.5).max(0.12);
        let ragged = 0.3 * s.noise.fbm_limited(p + Vec3::splat(2.0), 2.5 / cell, texel, 3);
        let (mut cover, mut kind, mut rel) = (0.0f32, Lichen::Pale, 0.0f32);
        cells(q, s.salt ^ SALT_LICHEN, |at, h| {
            // Each rosette shows where the amount is past its own threshold, growing from
            // nothing as it rises, so none is ever cut off.
            let threshold = unit(h);
            let grow = smoothstep(threshold, threshold + 0.3, amount);
            if grow <= 0.0 {
                return;
            }
            let rho = (0.35 + 0.6 * unit(rehash(h, 6))) * grow;
            let dist = (q - at).length() + ragged;
            let a = 1.0 - smoothstep(rho - soft, rho, dist);
            if a > cover {
                cover = a;
                let pick = unit(rehash(h, 7));
                kind = if pick < 0.1 {
                    Lichen::Rare
                } else if pick < 0.3 {
                    Lichen::Dark
                } else {
                    Lichen::Pale
                };
                rel = (dist / rho.max(1e-3)).clamp(0.0, 1.0);
            }
        });
        (cover, kind, rel)
    }

    /// Moss at `p`: how far it covers the stone, and how high its cushion stands there,
    /// 0 between the cushions to 1 atop one.
    fn moss_at(&self, p: Vec3, up: f32, smp: &Sample, height01: f32, texel: f32) -> (f32, f32) {
        let m = self.moss;
        let s = self.surface;
        let patches = s.noise2.fbm(p * (1.4 * m.patchiness.max(0.05) / s.radius) + Vec3::new(3.0, 0.0, 0.0), 5);
        let fine = s.noise.fbm_limited(p, 25.0, texel, 3);
        let nooks = (smp.crack + 0.5 * smp.pit + 0.4 * smp.hollow + 0.35 * smp.tread).min(1.0);
        let upward = m.upward.max(0.0);
        let field = m.amount - 0.5 + upward * (up - 0.25) + (1.0 - upward.min(1.0)) * 0.5 * (1.0 - height01)
            + patches * 0.8
            + m.crevices * nooks
            - 0.3 * smp.edge;
        // Clumps a few centimetres across, each taking hold where the field is past its
        // own threshold and spreading as it rises, so moss breaks up at its edge into a
        // mosaic of cushions with the stone showing between, and where it is thick they
        // run together. Each clump is domed, and fuzzed with smaller cushions.
        let soft = m.softness.clamp(0.01, 1.0);
        let field = field + fine * 0.15 - 0.1;
        if field < -0.3 {
            return (0.0, 0.0);
        }
        const CLUMP: f32 = 0.04;
        let q = p / CLUMP;
        let rim = (texel / CLUMP * 1.5).max(0.1);
        let ragged = 0.3 * s.noise.fbm_limited(p + Vec3::splat(4.0), 2.0 / CLUMP, texel, 2);
        let (mut cover, mut dome) = (0.0f32, 0.0f32);
        cells(q, s.salt ^ SALT_MOSS, |at, h| {
            let threshold = (unit(h) - 0.5) * 0.6;
            let grow = smoothstep(threshold, threshold + soft, field);
            if grow <= 0.0 {
                return;
            }
            let rho = (0.55 + 0.45 * unit(rehash(h, 9))) * (0.35 + 0.85 * grow);
            let dist = (q - at).length() + ragged;
            let a = 1.0 - smoothstep(rho - rim, rho, dist);
            if a > 0.0 {
                cover = cover.max(a);
                dome = dome.max(a * (1.0 - (dist / rho).clamp(0.0, 1.0).powi(2)).sqrt());
            }
        });
        // Clumps too few texels across to draw blur into an even cover instead, rather
        // than thinning out.
        let drawn = smoothstep(2.5, 6.0, CLUMP / texel);
        let even = smoothstep(-0.15, soft, field);
        cover = even + (cover - even) * drawn;
        dome += (0.5 * even - dome) * (1.0 - drawn);
        if cover <= 0.0 {
            return (0.0, 0.0);
        }
        let keep = smoothstep(1.5, 3.0, 0.009 / texel);
        let mut tuft = 0.55;
        if keep > 0.0 {
            let small = cushions(p / 0.009, s.salt ^ SALT_TUFTS);
            let fuzz = s.noise2.fbm_limited(p + Vec3::splat(1.7), 160.0, texel, 2) * 0.5 + 0.5;
            tuft += (0.35 * small + 0.4 * fuzz + 0.25 * dome - tuft) * keep;
        }
        (cover, tuft)
    }
}

impl Painter<'_> {
    /// Litter caught in moss that faces up: short bits of leaf and needle lying every
    /// which way, 0 to 1.
    fn litter(&self, p: Vec3, up: f32, texel: f32) -> f32 {
        const CELL: f32 = 0.02;
        if up < 0.2 || CELL * 0.25 < texel {
            return 0.0;
        }
        let q = p / CELL;
        let mut best = 0.0f32;
        cells(q, self.surface.salt ^ SALT_LITTER, |at, h| {
            if unit(h) > 0.12 * smoothstep(0.2, 0.7, up) {
                return;
            }
            let along = Vec3::new(unit(rehash(h, 10)) - 0.5, unit(rehash(h, 11)) - 0.5, unit(rehash(h, 12)) - 0.5).normalize_or(Vec3::X);
            let v = q - at;
            let a = v.dot(along);
            let across2 = (v.length_squared() - a * a).max(0.0);
            let d2 = a * a / 0.36 + across2 / 0.012;
            best = best.max(1.0 - smoothstep(0.6, 1.0, d2));
        });
        best
    }
}

/// Domes packed close, each its own height: 0 in the deepest gap to 1 atop the
/// highest.
fn cushions(q: Vec3, salt: u32) -> f32 {
    let mut top = 0.0f32;
    cells(q, salt, |at, h| {
        let d2 = (q - at).length_squared() / 0.8;
        if d2 < 1.0 {
            top = top.max((1.0 - d2).sqrt() * (0.55 + 0.45 * unit(rehash(h, 8))));
        }
    });
    top
}

/// `f` for each of `0..n`, on threads where there are threads, taking work as they free
/// up so none sits idle behind a slow stretch.
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).clamp(1, 32);
        let next = AtomicUsize::new(0);
        let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
            let (f, next) = (&f, &next);
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(move || {
                        let mut done = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n {
                                break done;
                            }
                            done.push((i, f(i)));
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a worker failed")).collect()
        });
        let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
        for (i, t) in parts.into_iter().flatten() {
            out[i] = Some(t);
        }
        out.into_iter().map(|t| t.expect("every item done")).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).collect()
    }
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
        p.texture.size = 128;
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
        let (a, b) = (build_rock(&p), build_rock(&p));
        assert_eq!(a.lods[0].positions, b.lods[0].positions);
        assert_eq!(a.lods[0].indices, b.lods[0].indices);
        assert_eq!(bake_rock(&p, &a).albedo, bake_rock(&p, &b).albedo);
    }

    /// Every edge of every LOD, by where its ends are, is shared by exactly two
    /// triangles, running opposite ways: no seam has opened and nothing is pinched.
    #[test]
    fn every_lod_is_closed() {
        for (name, src) in builtin_rock_presets() {
            let m = build_rock(&small(src));
            for (l, lod) in m.lods.iter().enumerate() {
                let key = |i: u32| lod.positions[i as usize].map(f32::to_bits);
                let mut edges = std::collections::HashMap::new();
                for tri in lod.indices.chunks_exact(3) {
                    for k in 0..3 {
                        let (a, b) = (key(tri[k]), key(tri[(k + 1) % 3]));
                        *edges.entry((a, b)).or_insert(0) += 1;
                    }
                }
                for (&(a, b), &count) in &edges {
                    assert_eq!(count, 1, "{name} LOD{l}: an edge runs the same way twice");
                    assert_eq!(edges.get(&(b, a)), Some(&1), "{name} LOD{l}: an edge with nothing across it");
                }
            }
        }
    }

    #[test]
    fn a_rock_faces_out_and_sits_sunk_in_the_ground() {
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
            // As wide and deep as it was asked to be.
            let size = Vec3::from(m.max) - Vec3::from(m.min);
            assert!((size.x - p.shape.width).abs() < p.shape.width * 0.2, "{name}: {size} against {}", p.shape.width);
            assert!((size.z - p.shape.depth).abs() < p.shape.depth * 0.2, "{name}: {size} against {}", p.shape.depth);
        }
    }

    #[test]
    fn each_lod_keeps_to_its_budget() {
        for (name, src) in builtin_rock_presets() {
            let p = small(src);
            let m = build_rock(&p);
            assert_eq!(m.lods.len(), p.lod.len());
            for (lod, asked) in m.lods.iter().zip(&p.lod) {
                let got = lod.triangle_count() as u32;
                assert!(got <= asked.triangles && got * 10 >= asked.triangles * 9, "{name}: {got} triangles for {}", asked.triangles);
            }
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

    /// No two triangles of any LOD lie over the same texels, and each stays inside the
    /// chart it belongs to, clear of its pad.
    #[test]
    fn charts_never_overlap() {
        for (name, src) in builtin_rock_presets() {
            let m = build_rock(&small(src));
            let size = 256usize;
            for (l, lod) in m.lods.iter().enumerate() {
                let mut hits = vec![0u8; size * size];
                for tri in lod.indices.chunks_exact(3) {
                    let uv = [0, 1, 2].map(|k| Vec2::from(lod.uvs[tri[k] as usize]) * size as f32);
                    let area = (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]);
                    assert!(area.abs() > 0.0, "{name} LOD{l}: a triangle with no area in the sheet");
                    let charts: Vec<_> = uv.iter().map(|u| m.atlas.chart_at(*u / size as f32)).collect();
                    assert!(charts.iter().all(|c| c.is_some() && *c == charts[0]), "{name} LOD{l}: a triangle across charts");
                    let lo = uv[0].min(uv[1]).min(uv[2]).floor().max(Vec2::ZERO);
                    let hi = uv[0].max(uv[1]).max(uv[2]).ceil().min(Vec2::splat(size as f32 - 1.0));
                    for y in lo.y as usize..=hi.y as usize {
                        for x in lo.x as usize..=hi.x as usize {
                            let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                            let w0 = (uv[2] - uv[1]).perp_dot(c - uv[1]) / area;
                            let w1 = (uv[0] - uv[2]).perp_dot(c - uv[2]) / area;
                            if w0 > 1e-3 && w1 > 1e-3 && 1.0 - w0 - w1 > 1e-3 {
                                hits[y * size + x] += 1;
                            }
                        }
                    }
                }
                assert!(hits.iter().all(|&h| h <= 1), "{name} LOD{l}: triangles overlap in the sheet");
            }
        }
    }

    #[test]
    fn moss_grows_on_top_and_more_moss_is_greener() {
        let green = |amount: f32| {
            let mut p = small(BOULDER_RON);
            p.moss.amount = amount;
            let mesh = build_rock(&p);
            let img = bake_rock(&p, &mesh).albedo;
            // The top face's chart.
            let [x, y, w, h] = mesh.atlas.rects[2];
            let size = img.width as f32;
            let (mut g, mut n) = (0.0f64, 0.0f64);
            for py in (y * size) as u32..((y + h) * size) as u32 {
                for px in (x * size) as u32..((x + w) * size) as u32 {
                    let c = img.at(px, py);
                    // Moss is olive: green well over blue, where grey stone is not.
                    g += c[1] as f64 - c[2] as f64;
                    n += 1.0;
                }
            }
            g / n
        };
        let (bare, mossy) = (green(-1.0), green(0.6));
        assert!(mossy > bare + 12.0, "top face green over blue {bare:.1} bare, {mossy:.1} mossy");
    }

    #[test]
    fn normal_maps_face_out_of_the_surface() {
        for (name, src) in builtin_rock_presets() {
            let p = small(src);
            let mesh = build_rock(&p);
            let maps = bake_rock(&p, &mesh);
            // Mostly close to the mesh's own normal; only crack walls and the like turn
            // far off it, and nothing turns right round.
            let n = &maps.normal;
            let (mut sum, mut steep, mut count) = (0.0f32, 0usize, 0usize);
            for y in 0..n.height {
                for x in 0..n.width {
                    let uv = Vec2::new((x as f32 + 0.5) / n.width as f32, (y as f32 + 0.5) / n.height as f32);
                    if mesh.atlas.chart_at(uv).is_none() {
                        continue;
                    }
                    let z = n.at(x, y)[2] as f32 / 255.0 * 2.0 - 1.0;
                    assert!(z > -0.05, "{name}: normal turned away at {x}, {y}");
                    sum += z;
                    steep += (z < 0.34) as usize;
                    count += 1;
                }
            }
            assert!(sum / count as f32 > 0.85, "{name}: mean z {}", sum / count as f32);
            assert!(steep * 100 < count, "{name}: {steep} of {count} texels steeper than 70 degrees");
        }
    }
}
