//! Ground cover: small clumps of grass cards for an engine to scatter over terrain.
//!
//! A tree is grown once and placed a handful of times; a clump of grass is built once
//! and planted tens of thousands of times, so what matters is different. A clump is a
//! square `footprint` of tufts spread evenly enough that instances laid side by side do
//! not show where one ends, each tuft a fan of bent cards standing on one root area,
//! and each card showing one cell of the blade atlas (`blades`). Everything is cards,
//! not blade triangles, for the reason trees use leaf cards: they filter.
//!
//! A cover preset is read like a species: any number may be a `(lo, hi)` range, landed
//! once per seed, and `instance` gives the one clump a seed builds. The atlas, like a
//! leaf cluster's, is not ranged: it is baked once and shared by every seed.
//!
//! LODs are authored rather than simplified, because simplifying cards only tears
//! holes: each coarser level keeps a stable subset of the cards, chosen by a key every
//! card draws once, drawn with fewer segments and widened to hold the coverage.

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::blades::BladeAtlasParams;
use crate::ranged::{key, Ranged, Scalar};
use crate::seed::PortableRng;
use crate::species::WindParams;

pub const MEADOW_GRASS_RON: &str = include_str!("../../../assets/cover/meadow_grass.ron");
pub const SHORT_GRASS_RON: &str = include_str!("../../../assets/cover/short_grass.ron");
pub const DRY_GRASS_RON: &str = include_str!("../../../assets/cover/dry_grass.ron");

/// Where presets saved from the viewer go, and are found by name from the CLI.
pub const CUSTOM_COVER_DIR: &str = "assets/cover/custom";

pub fn builtin_cover_presets() -> Vec<(&'static str, &'static str)> {
    vec![
        ("meadow_grass", MEADOW_GRASS_RON),
        ("short_grass", SHORT_GRASS_RON),
        ("dry_grass", DRY_GRASS_RON),
    ]
}

/// A cover preset as its file describes it, ranges and all.
pub type CoverTemplate = CoverParams<Ranged>;

pub fn parse_cover_template(ron_src: &str) -> Result<CoverTemplate, String> {
    ron::from_str(ron_src).map_err(|e| e.to_string())
}

/// Most tufts a clump may hold, and most cards a tuft may, so a typo cannot ask for a
/// million-card mesh.
const MAX_TUFTS: u32 = 4096;
const MAX_CARDS: u32 = 64;
const MAX_SEGMENTS: u32 = 16;
/// Most LODs, which is what an engine is likely to switch between.
const MAX_LODS: usize = 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "CoverParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct CoverParams<V = f32> {
    pub name: String,
    pub seed: u64,
    pub footprint: FootprintParams<V>,
    pub tuft: TuftParams<V>,
    pub card: CardParams<V>,
    pub colour: CoverColour<V>,
    pub normals: CoverNormals<V>,
    pub wind: WindParams<V>,
    /// Coarsest last. The first is LOD0, the full clump.
    pub lod: Vec<LodParams<V>>,
    pub atlas: BladeAtlasParams,
}

impl Default for CoverParams {
    fn default() -> Self {
        Self {
            name: "grass".to_string(),
            seed: 1,
            footprint: FootprintParams::default(),
            tuft: TuftParams::default(),
            card: CardParams::default(),
            colour: CoverColour::default(),
            normals: CoverNormals::default(),
            wind: WindParams {
                // No trunk: a clump does not sway about its middle. The one order a card
                // bends as is the limb's.
                flexibility: [0.0, 0.6, 0.0, 0.0],
                flutter: 0.12,
                frequency: 1.1,
            },
            lod: vec![
                LodParams { cards: 1.0, segments: 4.0, width: 1.0, screen_size: 0.0 },
                LodParams { cards: 0.5, segments: 2.0, width: 1.35, screen_size: 0.08 },
                LodParams { cards: 0.2, segments: 1.0, width: 1.9, screen_size: 0.035 },
            ],
            atlas: BladeAtlasParams::default(),
        }
    }
}

impl<V: Scalar> CoverParams<V> {
    pub fn defaults() -> Self {
        CoverParams::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> CoverParams<W> {
        CoverParams {
            name: self.name.clone(),
            seed: self.seed,
            footprint: self.footprint.map(&key(at, "footprint"), f),
            tuft: self.tuft.map(&key(at, "tuft"), f),
            card: self.card.map(&key(at, "card"), f),
            colour: self.colour.map(&key(at, "colour"), f),
            normals: self.normals.map(&key(at, "normals"), f),
            wind: self.wind.map(&key(at, "wind"), f),
            lod: self
                .lod
                .iter()
                .enumerate()
                .map(|(i, l)| l.map(&key(at, &format!("lod.{i}")), f))
                .collect(),
            atlas: self.atlas.clone(),
        }
    }
}

impl CoverTemplate {
    /// The one clump this preset builds at its seed.
    pub fn instance(&self) -> CoverParams {
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

impl CoverParams {
    pub fn template(&self) -> CoverTemplate {
        self.map("", &mut |_, v| Ranged::Fixed(v))
    }
}

/// The square a clump covers, and how its tufts are spread over it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "FootprintParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct FootprintParams<V = f32> {
    /// Metres across the square.
    pub size: V,
    /// Tufts in it.
    pub tufts: V,
    /// How far a tuft may wander from the middle of its own grid square, from 0 (a
    /// planted grid) to 1 (anywhere in the square). A jittered grid spreads the tufts
    /// evenly, where scattering them at random leaves bald patches and clumps, which
    /// show up as a pattern once the clump is repeated.
    pub jitter: V,
}

impl Default for FootprintParams {
    fn default() -> Self {
        Self { size: 1.5, tufts: 36.0, jitter: 0.8 }
    }
}

impl<V: Scalar> FootprintParams<V> {
    pub fn defaults() -> Self {
        FootprintParams::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> FootprintParams<W> {
        FootprintParams {
            size: f(&key(at, "size"), self.size),
            tufts: f(&key(at, "tufts"), self.tufts),
            jitter: f(&key(at, "jitter"), self.jitter),
        }
    }
}

/// One tuft: a fan of cards from one root area.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "TuftParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct TuftParams<V = f32> {
    /// Cards in a tuft.
    pub cards: V,
    /// Radius the cards' roots are spread over, in metres.
    pub radius: V,
    /// Height of a card, in metres, and its spread from tuft to tuft and card to card,
    /// as fractions.
    pub height: V,
    pub height_variance: V,
    pub card_height_variance: V,
    /// How far a card leans outward from the middle of its tuft at the root, in
    /// degrees, and the spread of that.
    pub lean_deg: V,
    pub lean_variance_deg: V,
    /// How much further it bends by its tip, in degrees: the arch of long grass.
    pub curl_deg: V,
    /// How far a card turns about its own length from root to tip, in degrees, so a
    /// tuft is not a set of flat planes.
    pub twist_deg: V,
}

impl Default for TuftParams {
    fn default() -> Self {
        Self {
            cards: 5.0,
            radius: 0.06,
            height: 0.45,
            height_variance: 0.25,
            card_height_variance: 0.15,
            lean_deg: 14.0,
            lean_variance_deg: 8.0,
            curl_deg: 18.0,
            twist_deg: 20.0,
        }
    }
}

impl<V: Scalar> TuftParams<V> {
    pub fn defaults() -> Self {
        TuftParams::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> TuftParams<W> {
        TuftParams {
            cards: f(&key(at, "cards"), self.cards),
            radius: f(&key(at, "radius"), self.radius),
            height: f(&key(at, "height"), self.height),
            height_variance: f(&key(at, "height_variance"), self.height_variance),
            card_height_variance: f(&key(at, "card_height_variance"), self.card_height_variance),
            lean_deg: f(&key(at, "lean_deg"), self.lean_deg),
            lean_variance_deg: f(&key(at, "lean_variance_deg"), self.lean_variance_deg),
            curl_deg: f(&key(at, "curl_deg"), self.curl_deg),
            twist_deg: f(&key(at, "twist_deg"), self.twist_deg),
        }
    }
}

/// One card.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "CardParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct CardParams<V = f32> {
    /// Width against the width that keeps the atlas cell's pixels square. Above 1 the
    /// blades are drawn broader than they were baked.
    pub width: V,
    /// Width at the top against width at the root. Grass fans out, so a little above 1.
    pub width_top: V,
}

impl Default for CardParams {
    fn default() -> Self {
        Self { width: 1.0, width_top: 1.15 }
    }
}

impl<V: Scalar> CardParams<V> {
    pub fn defaults() -> Self {
        CardParams::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> CardParams<W> {
        CardParams {
            width: f(&key(at, "width"), self.width),
            width_top: f(&key(at, "width_top"), self.width_top),
        }
    }
}

/// How lush and dry blades are mixed.
///
/// Colour variety is carried entirely by which atlas cell a card shows: an engine is
/// free to ignore vertex colours, and the one this is built for does. Each tuft is lush
/// or dry, and its cards draw from that kind of cell, with `mix` of them straying to
/// the other kind so a tuft is not all one colour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "CoverColour::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct CoverColour<V = f32> {
    pub dry_fraction: V,
    pub mix: V,
}

impl Default for CoverColour {
    fn default() -> Self {
        Self { dry_fraction: 0.2, mix: 0.2 }
    }
}

impl<V: Scalar> CoverColour<V> {
    pub fn defaults() -> Self {
        CoverColour::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> CoverColour<W> {
        CoverColour {
            dry_fraction: f(&key(at, "dry_fraction"), self.dry_fraction),
            mix: f(&key(at, "mix"), self.mix),
        }
    }
}

/// How the clump is lit, and how it meets the ground.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "CoverNormals::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct CoverNormals<V = f32> {
    /// How far shading normals lean from the card's own face to straight up. Each card
    /// lit by its own face turns a field into speckle; grass lit as the ground under
    /// it reads as one surface. Baked into the exported normals against flat ground,
    /// and handed to the engine to re-bend against the real terrain normal.
    pub ground_normal_blend: V,
    /// Metres above the root that should take the terrain's colour.
    pub root_blend: V,
}

impl Default for CoverNormals {
    fn default() -> Self {
        Self { ground_normal_blend: 0.85, root_blend: 0.12 }
    }
}

impl<V: Scalar> CoverNormals<V> {
    pub fn defaults() -> Self {
        CoverNormals::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> CoverNormals<W> {
        CoverNormals {
            ground_normal_blend: f(&key(at, "ground_normal_blend"), self.ground_normal_blend),
            root_blend: f(&key(at, "root_blend"), self.root_blend),
        }
    }
}

/// One level of detail.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "LodParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct LodParams<V = f32> {
    /// Share of the clump's cards kept.
    pub cards: V,
    /// Segments up each card.
    pub segments: V,
    /// Width of each card against LOD0's, to make up the coverage the dropped cards
    /// took with them.
    pub width: V,
    /// Screen size this level takes over at, as the engine measures it.
    pub screen_size: V,
}

impl Default for LodParams {
    fn default() -> Self {
        Self { cards: 1.0, segments: 4.0, width: 1.0, screen_size: 0.0 }
    }
}

impl<V: Scalar> LodParams<V> {
    pub fn defaults() -> Self {
        LodParams::default().map("", &mut |_, v| V::fixed(v))
    }

    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> LodParams<W> {
        LodParams {
            cards: f(&key(at, "cards"), self.cards),
            segments: f(&key(at, "segments"), self.segments),
            width: f(&key(at, "width"), self.width),
            screen_size: f(&key(at, "screen_size"), self.screen_size),
        }
    }
}

/// One card of the clump, in the terms every LOD builds it from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Card {
    /// Where it stands on the ground.
    pub root: [f32; 3],
    /// Metres from root to tip along the card.
    pub height: f32,
    /// Width at the root, in metres, at LOD0.
    pub width: f32,
    /// Which way the card faces, as a yaw in radians, and which way it leans, as a
    /// horizontal unit vector.
    pub yaw: f32,
    pub lean_dir: [f32; 2],
    /// Lean off vertical at the root, and the further bend by the tip, in radians.
    pub lean: f32,
    pub curl: f32,
    pub twist: f32,
    /// The atlas cell it shows.
    pub cell: u32,
    /// Its place in the order cards are dropped in: a card is in every LOD whose share
    /// of cards is above this.
    pub keep: f32,
}

/// One LOD's mesh.
#[derive(Clone, Debug, Default)]
pub struct CoverLod {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// xyz along the u of the texture, w the handedness.
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// Root of the card each vertex belongs to.
    pub origins: Vec<[f32; 3]>,
    /// The card each vertex belongs to, as an index into `CoverMesh::cards`, and how
    /// far up it the vertex is, 0 at the root and 1 at the tip.
    pub sway: Vec<(u32, f32)>,
    pub screen_size: f32,
    pub card_count: usize,
}

impl CoverLod {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// A whole clump: its cards, and the mesh of each LOD.
#[derive(Clone, Debug, Default)]
pub struct CoverMesh {
    pub cards: Vec<Card>,
    pub lods: Vec<CoverLod>,
    /// Highest point of LOD0, for bounds and fading.
    pub height: f32,
}

/// Builds the clump a preset's seed lands on. Deterministic: the same params build the
/// same clump, and every LOD is a subset of the one before it.
pub fn build_cover(p: &CoverParams) -> CoverMesh {
    let cards = lay_out(p);
    let cells = p.atlas.cells().len().max(1) as u32;
    let blend = p.normals.ground_normal_blend.clamp(0.0, 1.0);
    let lods: Vec<CoverLod> = p
        .lod
        .iter()
        .take(MAX_LODS)
        .map(|l| {
            let share = l.cards.clamp(0.0, 1.0);
            let segments = (l.segments.round() as u32).clamp(1, MAX_SEGMENTS);
            let mut out = CoverLod {
                screen_size: l.screen_size.max(0.0),
                ..Default::default()
            };
            for (i, card) in cards.iter().enumerate() {
                if card.keep >= share {
                    continue;
                }
                card_mesh(&mut out, i as u32, card, segments, l.width.max(0.05), p.card.width_top, cells, blend);
                out.card_count += 1;
            }
            out
        })
        .collect();
    let height = lods
        .first()
        .map(|l| l.positions.iter().map(|p| p[1]).fold(0.0, f32::max))
        .unwrap_or(0.0);
    CoverMesh { cards, lods, height }
}

/// Where every tuft stands, and every card in it.
fn lay_out(p: &CoverParams) -> Vec<Card> {
    let mut rng = PortableRng::seed_from_u64(p.seed ^ 0xC0FE_60A5_5000_0001);
    let size = p.footprint.size.max(0.01);
    let tufts = (p.footprint.tufts.round().max(1.0) as u32).min(MAX_TUFTS);
    let jitter = p.footprint.jitter.clamp(0.0, 1.0);

    // A jittered grid, filled in shuffled order so a count that is not a square leaves
    // its gaps scattered rather than along one edge.
    let n = (tufts as f32).sqrt().ceil() as u32;
    let mut squares: Vec<u32> = (0..n * n).collect();
    for i in (1..squares.len()).rev() {
        let j = rng.random_range(0..=i);
        squares.swap(i, j);
    }
    squares.truncate(tufts as usize);
    squares.sort_unstable();

    let t = &p.tuft;
    let (aspect, width_k) = (p.atlas.cell_size().0 as f32 / p.atlas.cell_size().1 as f32, p.card.width.max(0.05));
    let pools = cell_pools(p);
    let mut cards = Vec::new();
    let per_tuft = (t.cards.round().max(1.0) as u32).min(MAX_CARDS);
    for sq in squares {
        let (gx, gz) = ((sq % n) as f32, (sq / n) as f32);
        let cell = size / n as f32;
        let cx = -size * 0.5 + (gx + 0.5 + jitter * (rng.random::<f32>() - 0.5)) * cell;
        let cz = -size * 0.5 + (gz + 0.5 + jitter * (rng.random::<f32>() - 0.5)) * cell;
        let tuft_h = t.height.max(0.01) * (1.0 + t.height_variance * (rng.random::<f32>() * 2.0 - 1.0)).max(0.1);
        let dry = rng.random::<f32>() < p.colour.dry_fraction.clamp(0.0, 1.0);
        // Cards fan round the tuft half a turn, since each is seen from both sides.
        let phase = rng.random_range(0.0..std::f32::consts::PI);
        for k in 0..per_tuft {
            let r = t.radius.max(0.0) * rng.random::<f32>().sqrt();
            let a = rng.random_range(0.0..std::f32::consts::TAU);
            let (ox, oz) = (r * a.cos(), r * a.sin());
            // Outward from the middle of the tuft, loosened so a tuft is not a perfect
            // starburst.
            let (rx, rz) = {
                let b = rng.random_range(0.0..std::f32::consts::TAU);
                (b.cos() * 0.6, b.sin() * 0.6)
            };
            let (lx, lz) = if r > 1e-4 { (ox / r + rx, oz / r + rz) } else { (rx, rz) };
            let ll = (lx * lx + lz * lz).sqrt().max(1e-4);
            let height = tuft_h * (1.0 + t.card_height_variance * (rng.random::<f32>() * 2.0 - 1.0)).max(0.2);
            let lean = (t.lean_deg + t.lean_variance_deg * (rng.random::<f32>() * 2.0 - 1.0)).max(0.0);
            let curl = t.curl_deg * rng.random_range(0.6..=1.2);
            let twist = t.twist_deg * (rng.random::<f32>() * 2.0 - 1.0);
            let yaw = phase + std::f32::consts::PI * (k as f32 + rng.random_range(-0.3..=0.3)) / per_tuft as f32;
            let from_other = rng.random::<f32>() < p.colour.mix.clamp(0.0, 1.0);
            let pool = if dry != from_other { &pools.1 } else { &pools.0 };
            let pool = if pool.is_empty() { &pools.2 } else { pool };
            let cell_pick = pick(&mut rng, pool);
            cards.push(Card {
                root: [cx + ox, 0.0, cz + oz],
                height,
                width: height * aspect * width_k,
                yaw,
                lean_dir: [lx / ll, lz / ll],
                lean: lean.to_radians(),
                curl: curl.to_radians(),
                twist: twist.to_radians(),
                cell: cell_pick,
                keep: 0.0,
            });
        }
    }
    // The order cards drop out in: drawn after every card has its shape, so the keys
    // take no draws the shapes depend on. Dealt as a shuffled even spread rather than
    // drawn independently, so a LOD keeping a fifth of the cards keeps very nearly a
    // fifth, spread over every part of the clump.
    let mut order: Vec<usize> = (0..cards.len()).collect();
    for i in (1..order.len()).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }
    let count = cards.len().max(1) as f32;
    for (rank, &i) in order.iter().enumerate() {
        cards[i].keep = (rank as f32 + 0.5) / count;
    }
    cards
}

/// The atlas cells by kind, with their weights: lush, dry, and every cell for when a
/// kind has none.
type Pool = Vec<(u32, f32)>;

fn cell_pools(p: &CoverParams) -> (Pool, Pool, Pool) {
    let (mut lush, mut dry, mut all) = (Vec::new(), Vec::new(), Vec::new());
    for (i, c) in p.atlas.cells().iter().enumerate() {
        let entry = (i as u32, c.weight.max(0.0));
        all.push(entry);
        if c.dry { dry.push(entry) } else { lush.push(entry) }
    }
    if all.is_empty() {
        all.push((0, 1.0));
    }
    (lush, dry, all)
}

fn pick(rng: &mut PortableRng, pool: &[(u32, f32)]) -> u32 {
    let total: f32 = pool.iter().map(|e| e.1).sum();
    if total <= 0.0 {
        return pool[0].0;
    }
    let mut x = rng.random::<f32>() * total;
    for &(i, w) in pool {
        if x < w {
            return i;
        }
        x -= w;
    }
    pool[pool.len() - 1].0
}

/// One card, bent up its spine in `segments` steps.
#[allow(clippy::too_many_arguments)]
fn card_mesh(
    out: &mut CoverLod,
    index: u32,
    c: &Card,
    segments: u32,
    width_scale: f32,
    width_top: f32,
    cells: u32,
    blend: f32,
) {
    let up = glam::Vec3::Y;
    let lean_dir = glam::Vec3::new(c.lean_dir[0], 0.0, c.lean_dir[1]);
    let root = glam::Vec3::from(c.root);
    let base = out.positions.len() as u32;
    let (u0, du) = (c.cell as f32 / cells as f32, 1.0 / cells as f32);

    for k in 0..=segments {
        let t = k as f32 / segments as f32;
        // A spine of constant curvature in the plane of its lean, found exactly rather
        // than stepped, so a coarse LOD's few points lie on LOD0's curve.
        let a = c.lean + c.curl * t;
        let spine = if c.curl.abs() > 1e-4 {
            let k = c.height / c.curl;
            root + up * (k * (a.sin() - c.lean.sin())) + lean_dir * (k * (c.lean.cos() - a.cos()))
        } else {
            root + (up * c.lean.cos() + lean_dir * c.lean.sin()) * (c.height * t)
        };
        let along = (up * a.cos() + lean_dir * a.sin()).normalize();
        let yaw = c.yaw + c.twist * t;
        // Kept level rather than square to the spine, so both root corners stand on the
        // ground; a card leaning along its own width is sheared rather than tipped.
        let across = glam::Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let half = 0.5 * c.width * width_scale * (1.0 + (width_top - 1.0) * t);
        // The face, and the shading normal leant from it toward the sky. With the
        // tangent along the texture's u, the bitangent comes out running up the card,
        // which is the way up the texture the normal map was drawn.
        let face = across.cross(along).normalize_or(glam::Vec3::Z);
        let normal = face.lerp(up, blend).normalize_or(up);
        let tangent = (across - normal * across.dot(normal)).normalize_or(across);
        let w = if normal.cross(tangent).dot(along) >= 0.0 { 1.0 } else { -1.0 };
        let v = 1.0 - t;
        for (side, u) in [(-1.0f32, u0), (1.0, u0 + du)] {
            out.positions.push((spine + across * (half * side)).to_array());
            out.normals.push(normal.to_array());
            out.tangents.push([tangent.x, tangent.y, tangent.z, w]);
            out.uvs.push([u, v]);
            out.origins.push(c.root);
            out.sway.push((index, t));
        }
    }
    for k in 0..segments {
        let a = base + 2 * k;
        out.indices.extend_from_slice(&[a, a + 1, a + 3, a, a + 3, a + 2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_and_round_trip() {
        for (name, src) in builtin_cover_presets() {
            let t = parse_cover_template(src).unwrap_or_else(|e| panic!("{name}: {e}"));
            let text = ron::ser::to_string_pretty(&t, Default::default()).unwrap();
            assert_eq!(parse_cover_template(&text).unwrap(), t, "{name} does not round-trip");
        }
    }

    #[test]
    fn a_clump_is_the_same_every_time_it_is_built() {
        let p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        let (a, b) = (build_cover(&p), build_cover(&p));
        assert_eq!(a.lods[0].positions, b.lods[0].positions);
        assert_eq!(a.cards, b.cards);
    }

    #[test]
    fn every_lod_keeps_a_subset_of_the_one_before() {
        let p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        let m = build_cover(&p);
        assert!(m.lods.len() >= 2);
        let cards_of = |l: &CoverLod| {
            let mut ids: Vec<u32> = l.sway.iter().map(|s| s.0).collect();
            ids.dedup();
            ids
        };
        for pair in m.lods.windows(2) {
            let (fine, coarse) = (cards_of(&pair[0]), cards_of(&pair[1]));
            assert!(coarse.len() < fine.len());
            assert!(coarse.iter().all(|c| fine.contains(c)), "a coarse LOD has a card the finer one lacks");
            assert!(pair[1].triangle_count() < pair[0].triangle_count());
        }
        // The share asked for is the share kept, near enough.
        let l1 = &p.lod[1];
        let got = m.lods[1].card_count as f32 / m.cards.len() as f32;
        assert!((got - l1.cards).abs() < 0.02, "kept {got} of the cards, asked for {}", l1.cards);
    }

    #[test]
    fn the_clump_stands_on_the_ground_inside_its_footprint() {
        let p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        let m = build_cover(&p);
        let half = p.footprint.size * 0.5;
        for c in &m.cards {
            assert_eq!(c.root[1], 0.0);
            assert!(c.root[0].abs() <= half + p.tuft.radius && c.root[2].abs() <= half + p.tuft.radius);
        }
        let low = m.lods[0].positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        assert!(low.abs() < 1e-4, "lowest point {low}");
        assert!(m.height > 0.1);
    }

    #[test]
    fn tufts_are_spread_evenly_rather_than_clumped() {
        // Every quadrant of the footprint gets close to its share: a clump with a bald
        // corner shows as a pattern once it is planted side by side.
        let p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        let m = build_cover(&p);
        let mut quads = [0usize; 4];
        for c in &m.cards {
            quads[(c.root[0] > 0.0) as usize + 2 * (c.root[2] > 0.0) as usize] += 1;
        }
        let mean = m.cards.len() as f32 / 4.0;
        for q in quads {
            assert!((q as f32 - mean).abs() < mean * 0.35, "quadrants {quads:?}");
        }
    }

    #[test]
    fn shading_normals_lean_to_the_sky_and_tangents_run_across() {
        let p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        let lod = &build_cover(&p).lods[0];
        for (n, t) in lod.normals.iter().zip(&lod.tangents) {
            let n = glam::Vec3::from(*n);
            assert!((n.length() - 1.0).abs() < 1e-3);
            assert!(n.y > 0.5, "normal {n} barely leans up");
            assert!(glam::Vec3::new(t[0], t[1], t[2]).dot(n).abs() < 1e-3);
        }
    }

    #[test]
    fn roots_sway_nothing_and_tips_the_most() {
        let p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        let lod = &build_cover(&p).lods[0];
        for (pos, (_, t)) in lod.positions.iter().zip(&lod.sway) {
            if *t == 0.0 {
                assert!(pos[1].abs() < 1e-4, "a root vertex off the ground");
            }
        }
        assert!(lod.sway.iter().any(|s| s.1 == 1.0));
    }
}
