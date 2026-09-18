//! Growing a leaf-cluster texture out of a single-leaf one.
//!
//! A canopy built from one-leaf cards pays two triangles and four vertices to draw a
//! sliver that covers under a fifth of its own quad. Compositing a whole shoot into
//! one cell lets a single card stand in for dozens of leaves at the same real-world
//! leaf size, which is where the geometry saving comes from: fewer cards, not smaller
//! foliage. It also fixes the alpha, because the quad being drawn ends up mostly
//! covered instead of mostly empty.
//!
//! This lives in core rather than in the renderer because what a cluster looks like is
//! part of what a species is, and because the arrangement is worth testing without a
//! GL context in the way. The renderer decodes the PNGs and hands the pixels over.

use std::collections::VecDeque;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::species::LeafClusterParams;

/// Alpha above which a pixel counts as part of the leaf when measuring its extent.
const BOUNDS_CUTOFF: u8 = 8;
/// Ceiling on the arrangements baked into one sheet, so a species cannot ask for an
/// atlas that will not fit on a GPU.
const MAX_VARIANTS: u32 = 16;
/// Most alternative leaves a source may stack. Each is a full grid of cells decoded
/// for the whole bake, so the ceiling is on memory rather than on sense.
const MAX_SOURCES: u32 = 16;
/// Decorrelates one variant's random stream from the next.
const VARIANT_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;
/// How far apart the shoots of a multi-shoot cluster stand, as a fraction of the cell.
/// Wide enough that they are separate sprays rather than one thick one, narrow enough
/// that the outermost still has room for its leaves before the edge.
const SHOOT_SPREAD: f32 = 0.46;
/// Narrowest a blade may be squeezed by turning away from the card, as a fraction of
/// its own width. A real leaf has thickness and a curl, so even edge-on it is a
/// sliver rather than nothing, and a cluster that let leaves vanish would lose the
/// coverage clustering exists to buy.
const MIN_BLADE_WIDTH: f32 = 0.22;

/// Decoded RGBA8 pixels, the currency between the image decoder and this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Bitmap {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width as usize * height as usize) * 4],
        }
    }

    pub fn from_rgba(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        (pixels.len() == width as usize * height as usize * 4).then_some(Self {
            width,
            height,
            pixels,
        })
    }

    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    fn put(&mut self, x: u32, y: u32, p: [u8; 4]) {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.pixels[i..i + 4].copy_from_slice(&p);
    }

    /// Mean alpha over the whole image: the share of a card that shows foliage, which
    /// is what decides how many cards a canopy needs.
    pub fn mean_alpha(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.pixels.iter().skip(3).step_by(4).map(|&a| a as f64).sum();
        (sum / (self.pixels.len() / 4) as f64 / 255.0) as f32
    }

    /// Tight box of everything the alpha test would keep, as (x0, y0, x1, y1).
    pub fn alpha_bounds(&self, cutoff: u8) -> Option<(u32, u32, u32, u32)> {
        let (mut x0, mut y0, mut x1, mut y1) = (self.width, self.height, 0u32, 0u32);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.at(x, y)[3] > cutoff {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        (x1 >= x0 && y1 >= y0).then_some((x0, y0, x1, y1))
    }

    fn sample(&self, x: f32, y: f32) -> Option<[f32; 4]> {
        let fx = x - 0.5;
        let fy = y - 0.5;
        if fx < -1.0 || fy < -1.0 || fx > self.width as f32 || fy > self.height as f32 {
            return None;
        }
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let at = |ix: f32, iy: f32| -> [f32; 4] {
            let cx = (ix as i32).clamp(0, self.width as i32 - 1) as u32;
            let cy = (iy as i32).clamp(0, self.height as i32 - 1) as u32;
            self.at(cx, cy).map(|c| c as f32 / 255.0)
        };
        let (a, b, c, d) = (
            at(x0, y0),
            at(x0 + 1.0, y0),
            at(x0, y0 + 1.0),
            at(x0 + 1.0, y0 + 1.0),
        );
        let mut out = [0.0f32; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bottom = c[k] + (d[k] - c[k]) * tx;
            out[k] = top + (bottom - top) * ty;
        }
        Some(out)
    }

    /// One cell of an atlas, resampled to `width` so that a pixel covers as much width
    /// as it does height on the card the texture gets drawn on.
    fn cell(&self, cols: u32, rows: u32, c: u32, r: u32, width: u32) -> Bitmap {
        let cw = (self.width / cols.max(1)).max(1);
        let ch = (self.height / rows.max(1)).max(1);
        let mut out = Bitmap::new(width.max(1), ch);
        for y in 0..ch {
            for x in 0..width.max(1) {
                let sx = (x as f32 + 0.5) * cw as f32 / width.max(1) as f32;
                let p = self
                    .sample_cell(c * cw, r * ch, cw, ch, sx, y as f32 + 0.5)
                    .unwrap_or([0.0; 4]);
                out.put(x, y, p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
            }
        }
        out
    }

    /// Bilinear sample confined to one cell, so filtering never reaches across the
    /// seam into the neighbouring cell.
    fn sample_cell(
        &self,
        ox: u32,
        oy: u32,
        cw: u32,
        ch: u32,
        x: f32,
        y: f32,
    ) -> Option<[f32; 4]> {
        let fx = (x - 0.5).clamp(0.0, cw.saturating_sub(1) as f32);
        let fy = (y - 0.5).clamp(0.0, ch.saturating_sub(1) as f32);
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let at = |ix: f32, iy: f32| -> [f32; 4] {
            let cx = ox + (ix as u32).min(cw - 1);
            let cy = oy + (iy as u32).min(ch - 1);
            self.at(cx.min(self.width - 1), cy.min(self.height - 1))
                .map(|c| c as f32 / 255.0)
        };
        let (a, b, c, d) = (
            at(x0, y0),
            at(x0 + 1.0, y0),
            at(x0, y0 + 1.0),
            at(x0 + 1.0, y0 + 1.0),
        );
        let mut out = [0.0f32; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bottom = c[k] + (d[k] - c[k]) * tx;
            out[k] = top + (bottom - top) * ty;
        }
        Some(out)
    }
}

/// The three maps of a leaf material.
pub struct LeafMaps<'a> {
    pub albedo: &'a Bitmap,
    pub normal: Option<&'a Bitmap>,
    pub roughness: Option<&'a Bitmap>,
}

/// The same three, grown into clusters.
pub struct BakedMaps {
    pub albedo: Bitmap,
    pub normal: Option<Bitmap>,
    pub roughness: Option<Bitmap>,
}

/// One leaf placed in a cell, held as the inverse map from destination pixel back to
/// source pixel so the resample is destination-driven.
struct Placement {
    attach: (f32, f32),
    cos: f32,
    sin: f32,
    /// Source pixels per destination pixel, across the leaf and along it. They differ
    /// when a species narrows its leaves into needles, and again for every leaf once
    /// the arrangement is flattened, since a blade turned away from the card covers
    /// less of it.
    inv_scale_x: f32,
    inv_scale_y: f32,
    /// -1 when the leaf is mirrored onto the other side of the shoot.
    mirror: f32,
    pivot: (f32, f32),
    bounds: (i32, i32, i32, i32),
    /// Multiplier on this leaf's colour, so the ones behind the shoot come out darker.
    value: f32,
    /// Which of the source's alternative leaves this one is drawn from.
    source: usize,
}

/// Composites `cols` x `rows` cluster cells out of the matching cells of `src`, once
/// per variant, stacking the variants down the sheet.
///
/// Within one variant every cell takes the same placements, so the front and back
/// faces of a card share one silhouette and cannot show through each other. All three
/// maps composite through the albedo's coverage, so they keep agreeing pixel for
/// pixel. Variants are a different matter: each is its own arrangement, and the
/// renderer picks between them per card so a canopy is not one motif repeated.
pub fn bake_cluster(p: &LeafClusterParams, cols: u32, rows: u32, src: LeafMaps) -> BakedMaps {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let size = p.cell_size.clamp(16, 4096);

    // Each alternative leaf is a whole `cols` x `rows` grid, stacked down the source.
    let sources = p.sources.clamp(1, MAX_SOURCES);
    let src_rows = rows * sources;

    // Leaves are rotated here in pixels, so a cell whose pixels are not square in world
    // terms has to be stretched first or every rotated leaf comes out sheared.
    let cell_h = (src.albedo.height / src_rows).max(1);
    let square_w = ((cell_h as f32 * p.source_aspect.max(0.05)).round() as u32).max(1);
    type Maps = (Bitmap, Option<Bitmap>, Option<Bitmap>);
    let grid = |k: u32| -> Vec<Maps> {
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .map(|(c, r)| {
                let r = k * rows + r;
                (
                    src.albedo.cell(cols, src_rows, c, r, square_w),
                    src.normal.map(|m| m.cell(cols, src_rows, c, r, square_w)),
                    src.roughness.map(|m| m.cell(cols, src_rows, c, r, square_w)),
                )
            })
            .collect()
    };

    // The leaf hinges on the middle of the base of whatever the alpha test keeps, so
    // the spray pivots where a real leaf meets the shoot rather than on a corner of an
    // arbitrarily large cell. Cell zero of each alternative decides it for that
    // alternative, and one that turns out empty is simply not drawn from.
    let mut alts: Vec<(Vec<Maps>, (u32, u32, u32, u32))> = Vec::new();
    for k in 0..sources {
        let cells = grid(k);
        if let Some(b) = cells[0].0.alpha_bounds(BOUNDS_CUTOFF) {
            alts.push((cells, b));
        }
    }
    if alts.is_empty() {
        // Nothing to arrange. Hand back the source untouched rather than a blank sheet.
        return BakedMaps {
            albedo: src.albedo.clone(),
            normal: src.normal.cloned(),
            roughness: src.roughness.cloned(),
        };
    }
    let bounds: Vec<(u32, u32, u32, u32)> = alts.iter().map(|a| a.1).collect();

    let variants = p.variants.clamp(1, MAX_VARIANTS);
    let sheet = || Bitmap::new(size * cols, size * rows * variants);
    let mut albedo = sheet();
    let mut normal = src.normal.map(|_| sheet());
    let mut roughness = src.roughness.map(|_| sheet());
    for v in 0..variants {
        // Each variant is a shoot in its own right, so it gets its own stream rather
        // than a continuation of the last one: a species stays reproducible even if
        // the count of variants changes under it.
        let places = lay_out(p, size, &bounds, p.seed ^ (v as u64).wrapping_mul(VARIANT_STRIDE));
        for i in 0..(cols * rows) as usize {
            let ox = (i as u32 % cols) * size;
            let oy = (v * rows + i as u32 / cols) * size;
            let maps: Vec<(&Bitmap, Option<&Bitmap>, Option<&Bitmap>)> = alts
                .iter()
                .map(|(cells, _)| (&cells[i].0, cells[i].1.as_ref(), cells[i].2.as_ref()))
                .collect();
            let baked = composite(&places, size, &bounds, &maps);
            blit(&mut albedo, &baked.albedo, ox, oy);
            if let (Some(dst), Some(s)) = (normal.as_mut(), baked.normal.as_ref()) {
                blit(dst, s, ox, oy);
            }
            if let (Some(dst), Some(s)) = (roughness.as_mut(), baked.roughness.as_ref()) {
                blit(dst, s, ox, oy);
            }
        }
    }

    BakedMaps {
        albedo,
        normal,
        roughness,
    }
}

fn blit(dst: &mut Bitmap, src: &Bitmap, ox: u32, oy: u32) {
    for y in 0..src.height {
        for x in 0..src.width {
            dst.put(ox + x, oy + y, src.at(x, y));
        }
    }
}

/// A shoot's worth of leaves, arranged in three dimensions and then flattened into
/// the cell.
///
/// The arrangement is the whole difference between a card that reads as foliage and
/// one that reads as a pattern. Leaves go on a spiral round the shoot, not in two flat
/// rows, at a crotch angle that closes from base to tip, and each twists about its own
/// stalk. All of it is worked out as though the shoot stood in space and only then
/// projected onto the card: a leaf pointing out of the cell foreshortens along its
/// length, a blade turned edge-on narrows to a sliver, and whatever lies behind the
/// shoot is darkened and drawn first. One source leaf therefore covers the cell in a
/// whole range of apparent shapes and sizes, the outline comes out ragged, and the eye
/// reads depth in something that is flat.
///
/// `seed` rather than `p.seed`, because every variant of a cluster is its own shoot.
fn lay_out(
    p: &LeafClusterParams,
    size: u32,
    sources: &[(u32, u32, u32, u32)],
    seed: u64,
) -> Vec<Placement> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let cell = size as f32;
    let count = p.count.clamp(1, 512);
    let narrow = p.leaf_narrow.clamp(0.02, 4.0);
    let depth = p.depth.clamp(0.0, 1.0);
    let angle_jitter = p.angle_variance_deg.abs().max(1e-4);
    let roll_jitter = p.roll_variance_deg.abs().max(1e-4);
    let size_jitter = p.size_variance.abs().max(1e-4);
    let twist_jitter = p.blade_twist_deg.abs().max(1e-4);
    let space_jitter = p.spacing_variance.abs().max(1e-4);

    // Where each shoot stands across the cell, how far up it starts, which way it
    // leans, and the phase of its own spiral. Drawn up front so the leaves can be
    // dealt out between them.
    let shoots = p.shoots.clamp(1, 16);
    let per_shoot = (count / shoots).max(1);
    let mut stand: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(shoots as usize);
    if shoots == 1 {
        // A single shoot stands where it always did and draws exactly what it always
        // drew. Taking the extra draws here anyway would reshuffle the leaves of every
        // species that has not asked for more than one shoot.
        stand.push((0.5, 0.0, 1.0, rng.random_range(0.0..std::f32::consts::TAU)));
    } else {
        for s in 0..shoots {
            let across = ((s as f32 + 0.5) / shoots as f32 - 0.5) * SHOOT_SPREAD;
            stand.push((
                0.5 + across + rng.random_range(-0.04f32..=0.04),
                rng.random_range(-0.05f32..=0.05),
                if rng.random::<bool>() { 1.0 } else { -1.0 },
                rng.random_range(0.0..std::f32::consts::TAU),
            ));
        }
    }

    let mut placed: Vec<(f32, Placement)> = Vec::with_capacity(count as usize);

    for i in 0..count {
        // Which alternative this leaf is. Only drawn when there is a choice, so a
        // single-source species takes exactly the draws it always did.
        let source = if sources.len() > 1 {
            rng.random_range(0..sources.len())
        } else {
            0
        };
        let src = sources[source];
        let (bx0, by0, bx1, by1) = src;
        let pivot = ((bx0 + bx1) as f32 * 0.5, by1 as f32);
        let leaf_px = (by1 - by0 + 1) as f32;

        // Leaves are dealt round the shoots rather than filling one and starting the
        // next, so every shoot is finished even when the count does not divide evenly.
        let shoot = (i % shoots) as usize;
        let rung = i / shoots;
        // A shoot does not lay its leaves down against a ruler, so the step from one
        // to the next carries its own jitter rather than every cluster in the canopy
        // sharing one rhythm.
        let step = rung as f32 + rng.random_range(-space_jitter..=space_jitter);
        let t = if per_shoot > 1 {
            (step / (per_shoot - 1) as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let roll = &mut stand[shoot].3;
        *roll += (p.roll_deg + rng.random_range(-roll_jitter..=roll_jitter)).to_radians();
        let roll = *roll;
        let crotch = (p.base_angle_deg
            + (p.tip_angle_deg - p.base_angle_deg) * t
            + rng.random_range(-angle_jitter..=angle_jitter))
        .to_radians();

        // The leaf in three dimensions: x across the cell, y up it toward the tip of
        // the shoot, z out of the card toward the viewer.
        let (sc, cc) = crotch.sin_cos();
        let (sr, cr) = roll.sin_cos();
        let (dx, dy, dz) = (sc * cr, cc, sc * sr);

        // Flattening it. Whatever points out of the card is lost from the leaf's
        // apparent length, and `depth` decides how much of that loss is taken: a blade
        // is not a line, so even one aimed straight at the viewer shows something of
        // itself.
        let flat = (dx * dx + dy * dy).sqrt();
        let along_len = 1.0 + depth * (flat - 1.0);
        let (sin, cos) = if flat > 1e-4 {
            (dx / flat, dy / flat)
        } else {
            (0.0, 1.0)
        };

        // The blade is a plane through the leaf's own length. Untwisted it stands
        // edge-up off the shoot; the twist rolls it about its stalk toward lying flat.
        // What survives the projection is the part of it running across the leaf, and
        // that is what decides how broad the blade comes out.
        let twist = rng.random_range(-twist_jitter..=twist_jitter).to_radians();
        let (st, ct) = twist.sin_cos();
        let across = (ct * sr - st * cc * cr, st * sc);
        let seen = (across.0 * cos - across.1 * sin).abs();
        let broad = MIN_BLADE_WIDTH + (1.0 - MIN_BLADE_WIDTH) * seen;
        let across_width = 1.0 + depth * (broad - 1.0);

        let scale_t = 1.0 + (p.tip_scale - 1.0) * t;
        let scale = (scale_t * (1.0 + rng.random_range(-size_jitter..=size_jitter))).max(0.05);
        let scale_px = (p.leaf_length * scale * cell) / leaf_px.max(1.0);
        let scale_x = (scale_px * narrow * across_width).max(1e-4);
        let scale_y = (scale_px * along_len).max(1e-4);

        // The shoot itself leans as it climbs, so a cluster is not built on a plumb
        // line, and the leaves ride that lean. Neighbouring shoots lean opposite ways
        // as often as not, which is what stops a cell of them reading as a comb.
        let (base_x, base_y, lean_dir, _) = stand[shoot];
        let along = p.shoot_base + base_y + (p.shoot_tip - p.shoot_base) * t;
        let lean = p.shoot_curve * lean_dir * t * t;
        let mut place = Placement {
            attach: (
                (base_x + lean + rng.random_range(-0.02f32..=0.02)) * cell,
                (along + rng.random_range(-0.02f32..=0.02)) * cell,
            ),
            cos,
            sin,
            inv_scale_x: 1.0 / scale_x,
            inv_scale_y: 1.0 / scale_y,
            // Which way a leaf points decides which of its own sides faces out, so the
            // veining and the notch at its base alternate down the shoot.
            mirror: if dx >= 0.0 { 1.0 } else { -1.0 },
            pivot,
            bounds: (0, 0, 0, 0),
            // Everything behind the shoot is in the shoot's own shade. It is the only
            // depth cue a card has left once the arrangement has been flattened into
            // it, and without it a cluster lights as one flat sheet of colour.
            value: 1.0 - p.depth_shade.clamp(0.0, 1.0) * 0.5 * (1.0 - dz),
            source,
        };

        let (lo, hi) = extent(&place, scale_x, scale_y, src);
        place.attach.0 += nudge_inside(lo.0, hi.0, cell);
        place.attach.1 += nudge_inside(lo.1, hi.1, cell);

        place.bounds = dest_bounds(&place, scale_x, scale_y, src, size);
        placed.push((dz, place));
    }

    // Painter's order for a spray that has depth: whatever sits behind goes down
    // first. Laying them base to tip instead gives the tidy overlapping stack of a
    // fern frond, which no broadleaf shoot has.
    placed.sort_by(|a, b| a.0.total_cmp(&b.0));
    placed.into_iter().map(|(_, place)| place).collect()
}

/// How far to slide an attachment point so the leaf hanging off it sits inside the
/// cell.
///
/// A leaf cut off by the edge leaves a straight line across the foliage, and cells
/// butt up against their neighbours in the atlas, so the cut shows up on a card's
/// front and its back at once. Sliding the attachment beats shrinking the leaf, which
/// would quietly break the size the species asked for; it also crowds the leaves
/// toward the tip, which is where a shoot really does carry them closest together.
/// A leaf too big for the cell at all cannot be saved, so it is centred and loses the
/// same amount at both ends rather than all of it at one.
fn nudge_inside(lo: f32, hi: f32, cell: f32) -> f32 {
    // A pixel of slack at each end: the resample reaches one past the leaf all round
    // to keep its edge smooth.
    let (low, high) = (1.0, cell - 2.0);
    if hi - lo > high - low {
        return (low + high - lo - hi) * 0.5;
    }
    (low - lo).max(0.0) - (hi - high).max(0.0)
}

/// The continuous box the source leaf's opaque region lands in.
fn extent(
    p: &Placement,
    scale_x: f32,
    scale_y: f32,
    src: (u32, u32, u32, u32),
) -> ((f32, f32), (f32, f32)) {
    let (x0, y0, x1, y1) = src;
    let mut lo = (f32::MAX, f32::MAX);
    let mut hi = (f32::MIN, f32::MIN);
    for (sx, sy) in [
        (x0 as f32, y0 as f32),
        (x1 as f32, y0 as f32),
        (x0 as f32, y1 as f32),
        (x1 as f32, y1 as f32),
    ] {
        let u = (sx - p.pivot.0) * p.mirror * scale_x;
        let v = (sy - p.pivot.1) * scale_y;
        let x = p.attach.0 + u * p.cos - v * p.sin;
        let y = p.attach.1 + u * p.sin + v * p.cos;
        lo = (lo.0.min(x), lo.1.min(y));
        hi = (hi.0.max(x), hi.1.max(y));
    }
    (lo, hi)
}

/// Where that box lands as whole pixels, so the resample only walks the pixels a leaf
/// can actually reach.
fn dest_bounds(
    p: &Placement,
    scale_x: f32,
    scale_y: f32,
    src: (u32, u32, u32, u32),
    size: u32,
) -> (i32, i32, i32, i32) {
    let (lo, hi) = extent(p, scale_x, scale_y, src);
    let last = size as i32 - 1;
    (
        (lo.0.floor() as i32 - 1).clamp(0, last),
        (lo.1.floor() as i32 - 1).clamp(0, last),
        (hi.0.ceil() as i32 + 1).clamp(0, last),
        (hi.1.ceil() as i32 + 1).clamp(0, last),
    )
}

/// Painter's algorithm in premultiplied alpha. Every map is driven by the albedo's
/// coverage, so a pixel's normal and roughness always come from whichever leaf owns
/// the colour there.
/// One cell's worth of source maps for each alternative leaf.
type SourceMaps<'a> = (&'a Bitmap, Option<&'a Bitmap>, Option<&'a Bitmap>);

fn composite(
    places: &[Placement],
    size: u32,
    boxes: &[(u32, u32, u32, u32)],
    sources: &[SourceMaps],
) -> BakedMaps {
    let n = (size as usize) * (size as usize);
    let mut acc_rgb = vec![[0.0f32; 3]; n];
    let mut acc_a = vec![0.0f32; n];
    let mut acc_n = vec![[0.0f32; 3]; n];
    let mut acc_r = vec![0.0f32; n];
    // Every alternative has the same set of maps, so the first says which exist.
    let (normal, rough) = (sources[0].1, sources[0].2);

    for p in places {
        let (albedo, normal, rough) = sources[p.source];
        let src_box = boxes[p.source];
        let (lo_x, lo_y) = (src_box.0 as f32 - 1.0, src_box.1 as f32 - 1.0);
        let (hi_x, hi_y) = (src_box.2 as f32 + 1.0, src_box.3 as f32 + 1.0);
        let (lx, ly, hx, hy) = p.bounds;
        for y in ly..=hy {
            for x in lx..=hx {
                let dx = x as f32 + 0.5 - p.attach.0;
                let dy = y as f32 + 0.5 - p.attach.1;
                let u = dx * p.cos + dy * p.sin;
                let v = -dx * p.sin + dy * p.cos;
                let sx = p.pivot.0 + u * p.inv_scale_x * p.mirror;
                let sy = p.pivot.1 + v * p.inv_scale_y;
                // A rotated leaf fills little more than half of its own bounding box,
                // so most of the pixels walked here land off the leaf entirely. Reject
                // those on a rectangle test rather than paying for three bilinear
                // samples to discover they were transparent.
                if sx < lo_x || sx > hi_x || sy < lo_y || sy > hi_y {
                    continue;
                }

                let Some(src) = albedo.sample(sx, sy) else {
                    continue;
                };
                let a = src[3];
                if a <= 0.0 {
                    continue;
                }
                let i = y as usize * size as usize + x as usize;
                let keep = 1.0 - a;

                for k in 0..3 {
                    acc_rgb[i][k] = src[k] * p.value * a + acc_rgb[i][k] * keep;
                }
                if let Some(map) = normal {
                    let t = map.sample(sx, sy).unwrap_or([0.5, 0.5, 1.0, 1.0]);
                    let vec = rotate_normal(
                        [t[0] * 2.0 - 1.0, t[1] * 2.0 - 1.0, t[2] * 2.0 - 1.0],
                        p.cos,
                        p.sin,
                        p.mirror,
                    );
                    for k in 0..3 {
                        acc_n[i][k] = vec[k] * a + acc_n[i][k] * keep;
                    }
                }
                if let Some(map) = rough {
                    let t = map.sample(sx, sy).unwrap_or([0.5; 4]);
                    acc_r[i] = t[0] * a + acc_r[i] * keep;
                }
                acc_a[i] = a + acc_a[i] * keep;
            }
        }
    }

    let mut out_a = Bitmap::new(size, size);
    let mut out_n = normal.map(|_| Bitmap::new(size, size));
    let mut out_r = rough.map(|_| Bitmap::new(size, size));
    for i in 0..n {
        let x = (i % size as usize) as u32;
        let y = (i / size as usize) as u32;
        let a = acc_a[i].clamp(0.0, 1.0);
        let inv = if a > 1e-4 { 1.0 / a } else { 0.0 };
        out_a.put(
            x,
            y,
            [
                to_u8(acc_rgb[i][0] * inv),
                to_u8(acc_rgb[i][1] * inv),
                to_u8(acc_rgb[i][2] * inv),
                to_u8(a),
            ],
        );
        if let Some(img) = out_n.as_mut() {
            // Overlapping leaves average their normals, so renormalise rather than
            // trusting the blend to have stayed unit length.
            let v = [acc_n[i][0] * inv, acc_n[i][1] * inv, acc_n[i][2] * inv];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let v = if len > 1e-4 {
                [v[0] / len, v[1] / len, v[2] / len]
            } else {
                [0.0, 0.0, 1.0]
            };
            img.put(
                x,
                y,
                [
                    to_u8(v[0] * 0.5 + 0.5),
                    to_u8(v[1] * 0.5 + 0.5),
                    to_u8(v[2] * 0.5 + 0.5),
                    255,
                ],
            );
        }
        if let Some(img) = out_r.as_mut() {
            let r = if a > 1e-4 { acc_r[i] * inv } else { 0.6 };
            let b = to_u8(r);
            img.put(x, y, [b, b, b, 255]);
        }
    }

    bleed_color_outward(&mut out_a);
    BakedMaps {
        albedo: out_a,
        normal: out_n,
        roughness: out_r,
    }
}

/// A tangent-space normal has to turn with the pixels it describes. Image y runs down
/// while the green channel runs up, so the rotation the normal sees is the opposite
/// sense to the one applied to the image, and a mirror flips x on its own.
fn rotate_normal(n: [f32; 3], cos: f32, sin: f32, mirror: f32) -> [f32; 3] {
    let nx = n[0] * mirror;
    [nx * cos + n[1] * sin, -nx * sin + n[1] * cos, n[2]]
}

/// Floods the colour of covered pixels outward across the transparent background, so
/// no filtering step ever mixes leaf colour with what was behind it. Without it both
/// the downscale and the mip chain average leaf colour against an empty background and
/// every leaf picks up a dark halo.
///
/// A breadth-first fill from every covered pixel at once, which visits each pixel once
/// rather than rescanning the whole sheet per pass. That matters because this now runs
/// at load rather than in an offline tool.
pub fn bleed_color_outward(img: &mut Bitmap) {
    let (w, h) = (img.width, img.height);
    let n = (w as usize) * (h as usize);
    if n == 0 {
        return;
    }
    let mut seen = vec![false; n];
    let mut queue: VecDeque<u32> = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            if img.at(x, y)[3] > 0 {
                seen[(y * w + x) as usize] = true;
                queue.push_back(y * w + x);
            }
        }
    }
    if queue.is_empty() {
        return;
    }

    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        let src = img.at(x, y);
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let (nx, ny) = (nx as u32, ny as u32);
            let j = (ny * w + nx) as usize;
            if seen[j] {
                continue;
            }
            seen[j] = true;
            // Colour only: whatever alpha the pixel had is what decides coverage.
            let a = img.at(nx, ny)[3];
            img.put(nx, ny, [src[0], src[1], src[2], a]);
            queue.push_back(ny * w + nx);
        }
    }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::parse_species;

    /// A tall thin blade on a transparent sheet, standing on the bottom edge: the
    /// shape every leaf texture in the project has.
    fn leaf(w: u32, h: u32, tint: u8) -> Bitmap {
        let mut img = Bitmap::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let along = y as f32 / h as f32;
                let half = (w as f32 * 0.09) * (0.25 + along * 0.75);
                let inside = ((x as f32 + 0.5) - w as f32 * 0.5).abs() < half && along > 0.08;
                img.put(x, y, if inside { [40, tint, 30, 255] } else { [0, 0, 0, 0] });
            }
        }
        img
    }

    fn params() -> LeafClusterParams {
        LeafClusterParams {
            count: 32,
            cell_size: 128,
            ..LeafClusterParams::default()
        }
    }

    /// One leaf standing straight up the cell at a known size, for the tests that
    /// measure what came out against what was asked for. Everything that would move
    /// it is off: an arrangement is not what those are about.
    fn upright(leaf_length: f32) -> LeafClusterParams {
        LeafClusterParams {
            count: 1,
            cell_size: 256,
            leaf_length,
            base_angle_deg: 0.0,
            tip_angle_deg: 0.0,
            angle_variance_deg: 0.0,
            tip_scale: 1.0,
            size_variance: 0.0,
            spacing_variance: 0.0,
            shoot_curve: 0.0,
            shoot_base: 1.0,
            ..LeafClusterParams::default()
        }
    }

    #[test]
    fn a_cluster_covers_far_more_of_its_cell_than_one_leaf() {
        // The whole point: the card being drawn stops being mostly empty, which is what
        // lets a canopy use a fraction of the cards for the same foliage.
        let src = leaf(128, 128, 120);
        let baked = bake_cluster(
            &params(),
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let before = src.mean_alpha();
        let after = baked.albedo.mean_alpha();
        assert!(
            after > before * 2.0,
            "clustering should multiply coverage: {before} to {after}"
        );
    }

    #[test]
    fn every_cell_of_an_atlas_shares_one_silhouette() {
        // A card samples one cell on its front and another on its back. If the two
        // disagree about where the leaves are, the alpha test cuts different holes in
        // each face and the card becomes see-through from one side.
        let mut src = Bitmap::new(256, 128);
        for y in 0..128 {
            for x in 0..128 {
                let front = leaf(128, 128, 200).at(x, y);
                let back = leaf(128, 128, 90).at(x, y);
                src.put(x, y, back);
                src.put(x + 128, y, front);
            }
        }
        let baked = bake_cluster(
            &params(),
            2,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let size = params().cell_size;
        assert_eq!(baked.albedo.width, size * 2);
        let mut compared = 0;
        for y in 0..size {
            for x in 0..size {
                assert_eq!(
                    baked.albedo.at(x, y)[3],
                    baked.albedo.at(x + size, y)[3],
                    "cells disagree on coverage at {x},{y}"
                );
                compared += 1;
            }
        }
        assert!(compared > 1000);
        // The two cells still carry their own colour; only the coverage is shared.
        let green = |b: &Bitmap, ox: u32| -> u32 {
            (0..size)
                .flat_map(|y| (0..size).map(move |x| (x, y)))
                .filter(|(x, y)| b.at(ox + x, *y)[3] > 200)
                .map(|(x, y)| b.at(ox + x, y)[1] as u32)
                .sum()
        };
        assert!(green(&baked.albedo, size) > green(&baked.albedo, 0));
    }

    #[test]
    fn normals_stay_unit_length_where_leaves_overlap() {
        let src = leaf(128, 128, 120);
        let mut normal = Bitmap::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                normal.put(x, y, [200, 40, 180, 255]);
            }
        }
        let baked = bake_cluster(
            &params(),
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: Some(&normal),
                roughness: None,
            },
        );
        let out = baked.normal.expect("a normal map goes in, one comes out");
        let mut checked = 0;
        for y in 0..out.height {
            for x in 0..out.width {
                if baked.albedo.at(x, y)[3] < 250 {
                    continue;
                }
                let v: Vec<f32> = out.at(x, y)[..3]
                    .iter()
                    .map(|&c| c as f32 / 255.0 * 2.0 - 1.0)
                    .collect();
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                assert!((len - 1.0).abs() < 0.02, "normal length {len} at {x},{y}");
                checked += 1;
            }
        }
        assert!(checked > 500, "only {checked} covered pixels");
    }

    #[test]
    fn leaves_keep_the_size_they_were_given() {
        // Clustering has to buy its triangles by fitting more leaves into a cell, not
        // by quietly shrinking them: a card scaled to the cluster must still show a
        // leaf of the length the species asked for.
        let src = leaf(256, 256, 120);
        let baked = bake_cluster(
            &upright(0.5),
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let (_, y0, _, y1) = baked.albedo.alpha_bounds(BOUNDS_CUTOFF).expect("a leaf");
        let got = (y1 - y0 + 1) as f32 / 256.0;
        // One leaf, upright, asked to be half the cell, with every source of jitter
        // turned off: it has to come out at the length it was promised.
        assert!(
            (got - 0.5).abs() < 0.02,
            "leaf came out {got} of the cell, wanted 0.5"
        );
    }

    #[test]
    fn narrowing_thins_a_leaf_without_shortening_it() {
        // A conifer with no needle art of its own gets its needles this way, so the
        // control has to touch width alone: a shorter leaf would be a smaller leaf.
        let src = leaf(256, 256, 120);
        let thinned = |narrow: f32| LeafClusterParams {
            leaf_narrow: narrow,
            ..upright(0.5)
        };
        let measure = |narrow: f32| {
            let baked = bake_cluster(
                &thinned(narrow),
                1,
                1,
                LeafMaps {
                    albedo: &src,
                    normal: None,
                    roughness: None,
                },
            );
            let (x0, y0, x1, y1) = baked.albedo.alpha_bounds(BOUNDS_CUTOFF).expect("a leaf");
            ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32)
        };
        let (wide_w, wide_h) = measure(1.0);
        let (thin_w, thin_h) = measure(0.25);
        assert!(
            (thin_w / wide_w - 0.25).abs() < 0.08,
            "width went {wide_w} to {thin_w}, wanted a quarter"
        );
        assert!(
            (thin_h - wide_h).abs() < wide_h * 0.02,
            "length changed with width: {wide_h} to {thin_h}"
        );
    }

    #[test]
    fn no_leaf_is_cut_off_by_the_edge_of_its_cell() {
        // Cells butt up against their neighbours in the atlas, so a leaf that ran over
        // the edge would be sliced by a straight line and would also bleed into the
        // cell next door. Checked on a cell packed hard enough to push against it.
        let src = leaf(256, 256, 120);
        // Packed hard, splayed wide and leaning, but with leaves that do fit the cell:
        // one longer than the cell itself cannot be saved by moving it.
        let p = LeafClusterParams {
            count: 40,
            cell_size: 192,
            leaf_length: 0.42,
            base_angle_deg: 95.0,
            tip_angle_deg: 40.0,
            shoot_curve: 0.3,
            ..LeafClusterParams::default()
        };
        let baked = bake_cluster(
            &p,
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let img = &baked.albedo;
        let last = img.width - 1;
        for i in 0..img.width {
            for (x, y) in [(i, 0), (i, last), (0, i), (last, i)] {
                assert_eq!(img.at(x, y)[3], 0, "foliage reaches the cell edge at {x},{y}");
            }
        }
    }

    #[test]
    fn the_oak_arrangement_stays_off_its_own_cell_edges() {
        // The preset has to be one of the arrangements that fits, in every variant it
        // bakes, or the canopy shows straight cuts across its foliage.
        let src = leaf(256, 256, 120);
        let oak = parse_species(crate::species::OAK_RON).expect("the oak preset parses");
        let p = LeafClusterParams {
            cell_size: 192,
            ..oak.leaves.cluster.expect("the oak clusters its leaves")
        };
        let baked = bake_cluster(
            &p,
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let img = &baked.albedo;
        assert_eq!(img.height, img.width * p.variants);
        let last = img.width - 1;
        for v in 0..p.variants {
            let top = v * img.width;
            for i in 0..img.width {
                for (x, y) in [(i, top), (i, top + last), (0, top + i), (last, top + i)] {
                    assert_eq!(img.at(x, y)[3], 0, "variant {v} reaches its edge at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn standing_shoots_side_by_side_is_what_fills_a_cell_of_small_leaves() {
        // One shoot can only reach a leaf's length either side of itself, so a species
        // whose leaves are small against its card gets a narrow column of foliage with
        // the corners left empty — and the leaves pile up in that column rather than
        // spreading, which turns the card into a solid scalloped slab. Standing two or
        // three shoots across the cell is what widens it and opens it at once.
        // A broad blade rather than the thin one the other tests use: piling up is
        // only a problem for a leaf wide enough to cover its neighbour.
        let mut src = Bitmap::new(192, 192);
        for y in 0..192 {
            for x in 0..192 {
                let u = (x as f32 + 0.5) / 192.0 - 0.5;
                let v = (y as f32 + 0.5) / 192.0 - 0.55;
                let inside = (u / 0.30).powi(2) + (v / 0.42).powi(2) < 1.0;
                src.put(x, y, if inside { [40, 120, 30, 255] } else { [0, 0, 0, 0] });
            }
        }
        let measure = |shoots: u32| {
            let p = LeafClusterParams {
                count: 45,
                shoots,
                cell_size: 192,
                // Small against the cell: the case the single shoot cannot fill.
                leaf_length: 0.17,
                ..LeafClusterParams::default()
            };
            let baked = bake_cluster(
                &p,
                1,
                1,
                LeafMaps {
                    albedo: &src,
                    normal: None,
                    roughness: None,
                },
            );
            let (x0, _, x1, _) = baked
                .albedo
                .alpha_bounds(BOUNDS_CUTOFF)
                .expect("a cluster of leaves");
            (
                (x1 - x0 + 1) as f32 / p.cell_size as f32,
                baked.albedo.mean_alpha(),
            )
        };
        let (one_wide, one_cover) = measure(1);
        let (three_wide, three_cover) = measure(3);
        assert!(
            three_wide > one_wide * 1.25,
            "three shoots spanned {three_wide:.2} of the cell against {one_wide:.2} for one"
        );
        // The same leaves spread out instead of stacking, so more of them show.
        assert!(
            three_cover > one_cover * 1.1,
            "spreading the leaves did not uncover any of them: {one_cover:.3} to {three_cover:.3}"
        );
    }

    #[test]
    fn a_single_shoot_lays_out_exactly_as_it_did_before_shoots_existed() {
        // The per-shoot values are drawn up front, so taking those draws when nothing
        // asked for more than one shoot would reshuffle every leaf of every species
        // already tuned. Pinned here because it is invisible until a preset shifts.
        let src = leaf(128, 128, 120);
        let bake = |p: &LeafClusterParams| {
            bake_cluster(
                p,
                1,
                1,
                LeafMaps {
                    albedo: &src,
                    normal: None,
                    roughness: None,
                },
            )
            .albedo
        };
        let single = LeafClusterParams {
            shoots: 1,
            ..params()
        };
        assert_eq!(bake(&single), bake(&params()), "the default is not one shoot");
        // And the cluster a single shoot lays is centred on the cell, not offset the
        // way a member of a row of shoots would be.
        let img = bake(&single);
        let (x0, _, x1, _) = img.alpha_bounds(BOUNDS_CUTOFF).expect("a cluster");
        let middle = (x0 + x1) as f32 * 0.5 / img.width as f32;
        assert!(
            (middle - 0.5).abs() < 0.08,
            "a lone shoot sits at {middle:.2} across the cell rather than in the middle"
        );
    }

    #[test]
    fn variants_are_different_arrangements_stacked_down_the_sheet() {
        // The whole point of them: a canopy drawing one cluster everywhere shows the
        // motif however the cards are turned.
        let src = leaf(128, 128, 120);
        let p = LeafClusterParams {
            variants: 3,
            ..params()
        };
        let baked = bake_cluster(
            &p,
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let size = p.cell_size;
        assert_eq!(baked.albedo.height, size * 3);
        assert_eq!(baked.albedo.width, size);
        let differing = |a: u32, b: u32| {
            (0..size)
                .flat_map(|y| (0..size).map(move |x| (x, y)))
                .filter(|(x, y)| {
                    baked.albedo.at(*x, a * size + y)[3] != baked.albedo.at(*x, b * size + y)[3]
                })
                .count()
        };
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            assert!(
                differing(a, b) > (size * size / 20) as usize,
                "variants {a} and {b} are near enough the same arrangement"
            );
        }
    }

    #[test]
    fn leaves_behind_the_shoot_come_out_darker_than_the_ones_in_front() {
        // The depth cue that stops a cluster lighting as one flat sheet of colour.
        // Measured as the spread of brightness over covered pixels, against the same
        // arrangement with the shading turned off.
        let src = leaf(192, 192, 120);
        let spread = |depth_shade: f32| {
            let p = LeafClusterParams {
                count: 24,
                cell_size: 160,
                depth_shade,
                ..LeafClusterParams::default()
            };
            let baked = bake_cluster(
                &p,
                1,
                1,
                LeafMaps {
                    albedo: &src,
                    normal: None,
                    roughness: None,
                },
            );
            let values: Vec<f32> = (0..baked.albedo.height)
                .flat_map(|y| (0..baked.albedo.width).map(move |x| (x, y)))
                .filter(|(x, y)| baked.albedo.at(*x, *y)[3] > 200)
                .map(|(x, y)| baked.albedo.at(x, y)[1] as f32)
                .collect();
            assert!(values.len() > 500, "only {} covered pixels", values.len());
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
        };
        let flat = spread(0.0);
        let shaded = spread(0.6);
        assert!(
            shaded > flat + 4.0,
            "shading the depth barely moved the spread: {flat} to {shaded}"
        );
    }

    #[test]
    fn baking_is_deterministic() {
        let src = leaf(128, 128, 120);
        let maps = || LeafMaps {
            albedo: &src,
            normal: None,
            roughness: None,
        };
        let a = bake_cluster(&params(), 1, 1, maps());
        let b = bake_cluster(&params(), 1, 1, maps());
        assert_eq!(a.albedo, b.albedo);
    }

    #[test]
    fn a_blank_source_is_handed_back_rather_than_blanked_further() {
        let src = Bitmap::new(32, 32);
        let baked = bake_cluster(
            &params(),
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        assert_eq!(baked.albedo, src);
    }
    #[test]
    fn a_cluster_draws_its_leaves_from_every_source() {
        // A photographed set is several different sprays stacked down the sheet, and a
        // tuft built from one of them repeated reads as that spray stamped round a
        // point. With `sources` set, every leaf picks one, so both show up; without it
        // the sheet is one tall cell and only the one leaf is ever drawn.
        let (a, b) = (leaf(128, 128, 60), leaf(128, 128, 220));
        let mut src = Bitmap::new(128, 256);
        for y in 0..128 {
            for x in 0..128 {
                src.put(x, y, a.at(x, y));
                src.put(x, y + 128, b.at(x, y));
            }
        }
        let greens = |p: &LeafClusterParams| {
            let baked = bake_cluster(
                p,
                1,
                1,
                LeafMaps {
                    albedo: &src,
                    normal: None,
                    roughness: None,
                },
            );
            let (mut dark, mut light) = (0, 0);
            for y in 0..baked.albedo.height {
                for x in 0..baked.albedo.width {
                    let px = baked.albedo.at(x, y);
                    // Solid interior only: edges blend the two and would count twice.
                    if px[3] < 250 {
                        continue;
                    }
                    if px[1] < 100 {
                        dark += 1;
                    } else if px[1] > 180 {
                        light += 1;
                    }
                }
            }
            (dark, light)
        };

        let mixed = LeafClusterParams { sources: 2, ..params() };
        let (dark, light) = greens(&mixed);
        assert!(dark > 50 && light > 50, "two sources baked {dark} dark and {light} light texels");

        // One source, and the default: the leaf is the whole sheet, both sprays drawn
        // as one, so the cluster is built of that single leaf shape every time.
        let single = bake_cluster(
            &params(),
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        let again = bake_cluster(
            &LeafClusterParams { sources: 1, ..params() },
            1,
            1,
            LeafMaps {
                albedo: &src,
                normal: None,
                roughness: None,
            },
        );
        assert_eq!(
            single.albedo.pixels, again.albedo.pixels,
            "sources: 1 has to bake exactly what the default always did"
        );
    }
}
