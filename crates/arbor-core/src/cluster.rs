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
    /// when a species narrows its leaves into needles.
    inv_scale_x: f32,
    inv_scale_y: f32,
    /// -1 when the leaf is mirrored onto the other side of the shoot.
    mirror: f32,
    pivot: (f32, f32),
    bounds: (i32, i32, i32, i32),
}

/// Composites `cols` x `rows` cluster cells out of the matching cells of `src`.
///
/// Every cell takes the same placements, so the front and back faces of a card share
/// one silhouette and cannot show through each other. All three maps composite through
/// the albedo's coverage, so they keep agreeing pixel for pixel.
pub fn bake_cluster(p: &LeafClusterParams, cols: u32, rows: u32, src: LeafMaps) -> BakedMaps {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let size = p.cell_size.clamp(16, 4096);

    // Leaves are rotated here in pixels, so a cell whose pixels are not square in world
    // terms has to be stretched first or every rotated leaf comes out sheared.
    let cell_h = (src.albedo.height / rows).max(1);
    let square_w = ((cell_h as f32 * p.source_aspect.max(0.05)).round() as u32).max(1);
    let cells: Vec<(Bitmap, Option<Bitmap>, Option<Bitmap>)> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (c, r)))
        .map(|(c, r)| {
            (
                src.albedo.cell(cols, rows, c, r, square_w),
                src.normal.map(|m| m.cell(cols, rows, c, r, square_w)),
                src.roughness.map(|m| m.cell(cols, rows, c, r, square_w)),
            )
        })
        .collect();

    // The leaf hinges on the middle of the base of whatever the alpha test keeps, so
    // the spray pivots where a real leaf meets the shoot rather than on a corner of an
    // arbitrarily large cell. Cell zero decides it for every cell.
    let Some(bounds) = cells[0].0.alpha_bounds(BOUNDS_CUTOFF) else {
        // Nothing to arrange. Hand back the source untouched rather than a blank sheet.
        return BakedMaps {
            albedo: src.albedo.clone(),
            normal: src.normal.cloned(),
            roughness: src.roughness.cloned(),
        };
    };
    let places = lay_out(p, size, bounds);

    let mut albedo = Bitmap::new(size * cols, size * rows);
    let mut normal = src.normal.map(|_| Bitmap::new(size * cols, size * rows));
    let mut roughness = src.roughness.map(|_| Bitmap::new(size * cols, size * rows));
    for (i, (a, n, r)) in cells.iter().enumerate() {
        let ox = (i as u32 % cols) * size;
        let oy = (i as u32 / cols) * size;
        let baked = composite(&places, size, bounds, a, n.as_ref(), r.as_ref());
        blit(&mut albedo, &baked.albedo, ox, oy);
        if let (Some(dst), Some(s)) = (normal.as_mut(), baked.normal.as_ref()) {
            blit(dst, s, ox, oy);
        }
        if let (Some(dst), Some(s)) = (roughness.as_mut(), baked.roughness.as_ref()) {
            blit(dst, s, ox, oy);
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

/// Leaves alternating down a shoot, splayed wide at the base and closing toward the
/// tip. That taper is what gives the cell a leaf-like silhouette of its own, so a card
/// reads as a leafy shoot rather than as a rectangle of foliage.
fn lay_out(p: &LeafClusterParams, size: u32, src: (u32, u32, u32, u32)) -> Vec<Placement> {
    let (bx0, by0, bx1, by1) = src;
    let pivot = ((bx0 + bx1) as f32 * 0.5, by1 as f32);
    let leaf_px = (by1 - by0 + 1) as f32;

    let mut rng = SmallRng::seed_from_u64(p.seed);
    let cell = size as f32;
    let count = p.count.clamp(1, 512);
    let mut places = Vec::with_capacity(count as usize);

    for i in 0..count {
        let t = if count > 1 {
            i as f32 / (count - 1) as f32
        } else {
            0.0
        };
        // Emitted base first, so leaves nearer the base sit behind the ones above.
        let side = if i % 2 == 0 { 1.0f32 } else { -1.0 };
        let jitter = p.angle_variance_deg.abs().max(1e-4);
        let splay = (p.base_angle_deg
            + (p.tip_angle_deg - p.base_angle_deg) * t
            + rng.random_range(-jitter..=jitter))
        .to_radians();
        let scale_t = 1.0 + (p.tip_scale - 1.0) * t;
        let scale = (scale_t * (1.0 + rng.random_range(-0.10f32..=0.10))).max(0.05);

        let along = p.shoot_base + (p.shoot_tip - p.shoot_base) * t;
        let attach = (
            (0.5 + rng.random_range(-0.012f32..=0.012)) * cell,
            (along + rng.random_range(-0.012f32..=0.012)) * cell,
        );

        // Rotation is measured off straight up the cell, then mirrored per side, so
        // both sides splay outward by the same angle.
        let (sin, cos) = (splay * side).sin_cos();
        let scale_px = (p.leaf_length * scale * cell) / leaf_px.max(1.0);
        let narrow = p.leaf_narrow.clamp(0.02, 4.0);

        let mut place = Placement {
            attach,
            cos,
            sin,
            inv_scale_x: 1.0 / (scale_px * narrow).max(1e-6),
            inv_scale_y: 1.0 / scale_px.max(1e-6),
            mirror: side,
            pivot,
            bounds: (0, 0, 0, 0),
        };
        place.bounds = dest_bounds(&place, scale_px * narrow, scale_px, src, size);
        places.push(place);
    }
    places
}

/// Where the source leaf's opaque box lands in the cell, so the resample only walks
/// the pixels a leaf can actually reach.
fn dest_bounds(
    p: &Placement,
    scale_x: f32,
    scale_y: f32,
    src: (u32, u32, u32, u32),
    size: u32,
) -> (i32, i32, i32, i32) {
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
fn composite(
    places: &[Placement],
    size: u32,
    src_box: (u32, u32, u32, u32),
    albedo: &Bitmap,
    normal: Option<&Bitmap>,
    rough: Option<&Bitmap>,
) -> BakedMaps {
    let n = (size as usize) * (size as usize);
    let mut acc_rgb = vec![[0.0f32; 3]; n];
    let mut acc_a = vec![0.0f32; n];
    let mut acc_n = vec![[0.0f32; 3]; n];
    let mut acc_r = vec![0.0f32; n];
    let (lo_x, lo_y) = (src_box.0 as f32 - 1.0, src_box.1 as f32 - 1.0);
    let (hi_x, hi_y) = (src_box.2 as f32 + 1.0, src_box.3 as f32 + 1.0);

    for p in places {
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
                    acc_rgb[i][k] = src[k] * a + acc_rgb[i][k] * keep;
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
            count: 16,
            cell_size: 128,
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
        let p = LeafClusterParams {
            count: 1,
            cell_size: 256,
            leaf_length: 0.5,
            base_angle_deg: 0.0,
            tip_angle_deg: 0.0,
            angle_variance_deg: 0.0,
            tip_scale: 1.0,
            shoot_base: 1.0,
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
        let (_, y0, _, y1) = baked.albedo.alpha_bounds(BOUNDS_CUTOFF).expect("a leaf");
        let got = (y1 - y0 + 1) as f32 / 256.0;
        // One leaf, upright, asked to be half the cell. The jitter on scale is +-10%.
        assert!(
            (got - 0.5).abs() < 0.08,
            "leaf came out {got} of the cell, wanted 0.5"
        );
    }

    #[test]
    fn narrowing_thins_a_leaf_without_shortening_it() {
        // A conifer with no needle art of its own gets its needles this way, so the
        // control has to touch width alone: a shorter leaf would be a smaller leaf.
        let src = leaf(256, 256, 120);
        let upright = |narrow: f32| LeafClusterParams {
            count: 1,
            cell_size: 256,
            leaf_length: 0.5,
            base_angle_deg: 0.0,
            tip_angle_deg: 0.0,
            angle_variance_deg: 0.0,
            tip_scale: 1.0,
            shoot_base: 1.0,
            leaf_narrow: narrow,
            ..LeafClusterParams::default()
        };
        let measure = |narrow: f32| {
            let baked = bake_cluster(
                &upright(narrow),
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
}
