//! Leaf art painted from a description, for species with no photographed leaf.
//!
//! The textures on disk are photographs, and there is not one of ivy, hazel or box among
//! them. A leaf is a simple enough thing to paint: an outline, a midrib and veins, a
//! colour for each face and how glossy each is. So these are painted here, into the
//! same layout the photographed sets use — two cells side by side, the underside on the
//! left and the lit face on the right, the stalk at the bottom middle of each cell and
//! the tip toward the top — and written to `assets/textures` by
//! `examples/paint_leaves.rs`, where the viewer and the exporters read them like any
//! other set.
//!
//! An outline is a handful of lobes radiating from where the stalk meets the blade,
//! each a pointed oval along its own axis, merged smoothly into one blade. One lobe is
//! an ovate leaf; one long lobe with two small ones turned back is a heart; five
//! spread across the top half is ivy.

use glam::Vec2;

use crate::cluster::{BakedMaps, Bitmap};

/// One lobe of a blade, radiating from where the stalk meets it.
#[derive(Clone, Copy, Debug)]
pub struct Lobe {
    /// Angle off the leaf's axis, in degrees: 0 points at the tip, 90 straight out.
    pub angle_deg: f32,
    /// Length, as a share of the cell.
    pub length: f32,
    /// Widest half-width, as a share of the length.
    pub width: f32,
    /// Where along the lobe it is widest, from 0 at its base to 1 at its point.
    pub widest: f32,
    /// How far behind the stalk's junction the lobe begins, as a share of the cell.
    /// A heart-shaped blade runs on past where its stalk meets it.
    pub start: f32,
}

impl Lobe {
    const fn new(angle_deg: f32, length: f32, width: f32, widest: f32) -> Self {
        Self { angle_deg, length, width, widest, start: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct LeafArt {
    /// Base name the maps are written under: `<name>_albedo.png` and so on.
    pub name: &'static str,
    /// Pixels on a side of one cell. The sheet is two cells wide.
    pub cell: u32,
    pub lobes: Vec<Lobe>,
    /// How far up the cell the stalk meets the blade, as a share of it.
    pub junction: f32,
    /// How softly the lobes merge into one another, as a share of the cell. Larger
    /// fills the sinuses between them.
    pub merge: f32,
    /// Half-angle of the notch cut into the base of the blade round the stalk, in
    /// degrees: the cleft of a heart. Zero cuts none.
    pub notch_deg: f32,
    /// Teeth along the margin, per unit of cell, and how deep they cut, as a share of
    /// the cell.
    pub teeth: f32,
    pub tooth_depth: f32,
    /// Secondary veins off each midrib, per unit of cell. Zero draws the midribs only.
    pub veins: f32,
    /// How far the veins are drawn apart in colour from the blade, from 0 to 1.
    pub vein_contrast: f32,
    /// sRGB colours of the lit face, the underside and the veins on the lit face.
    pub top: [f32; 3],
    pub under: [f32; 3],
    pub vein: [f32; 3],
    /// Roughness of each face.
    pub top_roughness: f32,
    pub under_roughness: f32,
    /// How much the colour wanders across the blade, from 0 to 1.
    pub mottle: f32,
    pub seed: u32,
}

impl LeafArt {
    /// Common ivy, *Hedera helix*, as it grows on a wall: five-lobed, dark and glossy,
    /// with pale veins running out from the stalk.
    pub fn ivy() -> Self {
        Self {
            name: "leaf_ivy",
            cell: 512,
            lobes: vec![
                Lobe::new(0.0, 0.66, 0.42, 0.25),
                Lobe::new(52.0, 0.46, 0.46, 0.3),
                Lobe::new(-52.0, 0.46, 0.46, 0.3),
                Lobe::new(108.0, 0.30, 0.5, 0.35),
                Lobe::new(-108.0, 0.30, 0.5, 0.35),
            ],
            junction: 0.30,
            merge: 0.05,
            notch_deg: 0.0,
            teeth: 0.0,
            tooth_depth: 0.0,
            veins: 9.0,
            vein_contrast: 0.55,
            top: [0.10, 0.20, 0.07],
            under: [0.30, 0.38, 0.20],
            vein: [0.55, 0.62, 0.44],
            top_roughness: 0.38,
            under_roughness: 0.75,
            mottle: 0.35,
            seed: 11,
        }
    }

    /// Hazel: a broad, soft, round leaf with a short point, a heart at its base and a
    /// double row of teeth.
    pub fn hazel() -> Self {
        Self {
            name: "leaf_hazel",
            cell: 512,
            lobes: vec![
                Lobe { start: 0.12, ..Lobe::new(0.0, 0.9, 0.5, 0.36) },
            ],
            junction: 0.16,
            merge: 0.0,
            notch_deg: 32.0,
            teeth: 38.0,
            tooth_depth: 0.012,
            veins: 16.0,
            vein_contrast: 0.3,
            top: [0.24, 0.38, 0.12],
            under: [0.40, 0.50, 0.27],
            vein: [0.34, 0.46, 0.20],
            top_roughness: 0.7,
            under_roughness: 0.85,
            mottle: 0.3,
            seed: 23,
        }
    }

    /// Box: a small, stiff, glossy oval, the leaf a clipped hedge is made of.
    pub fn boxwood() -> Self {
        Self {
            name: "leaf_box",
            cell: 256,
            lobes: vec![Lobe::new(0.0, 0.84, 0.36, 0.48)],
            junction: 0.08,
            merge: 0.02,
            notch_deg: 0.0,
            teeth: 0.0,
            tooth_depth: 0.0,
            veins: 0.0,
            vein_contrast: 0.3,
            top: [0.12, 0.25, 0.08],
            under: [0.34, 0.42, 0.18],
            vein: [0.28, 0.40, 0.16],
            top_roughness: 0.35,
            under_roughness: 0.7,
            mottle: 0.2,
            seed: 31,
        }
    }

    /// A smooth heart, pointed at the tip: bindweed, morning glory, the heart-leaved
    /// climbers and trailers.
    pub fn heart() -> Self {
        Self {
            name: "leaf_heart",
            cell: 512,
            lobes: vec![
                Lobe { start: 0.2, ..Lobe::new(0.0, 0.92, 0.46, 0.34) },
            ],
            junction: 0.24,
            merge: 0.0,
            notch_deg: 38.0,
            teeth: 0.0,
            tooth_depth: 0.0,
            veins: 10.0,
            vein_contrast: 0.3,
            top: [0.16, 0.32, 0.10],
            under: [0.34, 0.46, 0.22],
            vein: [0.40, 0.55, 0.28],
            top_roughness: 0.5,
            under_roughness: 0.8,
            mottle: 0.25,
            seed: 47,
        }
    }

    /// Every set of art the presets draw with.
    pub fn all() -> Vec<Self> {
        vec![Self::ivy(), Self::hazel(), Self::boxwood(), Self::heart()]
    }
}

/// What one pixel of one cell is: how much of it is leaf, and what of the leaf.
struct Sample {
    /// Signed distance inside the blade or stalk, in cells: positive inside.
    inside: f32,
    /// How much of a vein this is, from 0 to 1.
    vein: f32,
    /// Height of the blade surface, for the normal map: domed between the veins.
    height: f32,
}

fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 || !a.is_finite() || !b.is_finite() {
        return a.max(b);
    }
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    b + (a - b) * h + k * h * (1.0 - h)
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ seed.wrapping_mul(0xCB1A_B31F);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5BD1_E995);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 65535.0
}

fn value_noise(p: Vec2, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let (x, y) = (i.x as i32, i.y as i32);
    let a = hash(x, y, seed);
    let b = hash(x + 1, y, seed);
    let c = hash(x, y + 1, seed);
    let d = hash(x + 1, y + 1, seed);
    let top = a + (b - a) * u.x;
    let bottom = c + (d - c) * u.x;
    top + (bottom - top) * u.y
}

impl LeafArt {
    /// The leaf at `p`, in cell units: x across from the middle, y up from the bottom.
    fn sample(&self, p: Vec2) -> Sample {
        let base = Vec2::new(0.0, self.junction);
        let mut blade = f32::NEG_INFINITY;
        let mut vein = 0.0f32;
        let mut dome = 0.0f32;
        for lobe in &self.lobes {
            let a = lobe.angle_deg.to_radians();
            let axis = Vec2::new(a.sin(), a.cos());
            let across = Vec2::new(axis.y, -axis.x);
            let q = p - base;
            let u = q.dot(axis) + lobe.start;
            let v = q.dot(across);
            let t = u / lobe.length.max(1e-3);
            if !(-0.05..=1.05).contains(&t) {
                continue;
            }
            let t = t.clamp(0.0, 1.0);
            // A pointed oval: widest at `widest`, closing to a point at the tip and
            // rounding in to nothing at the base.
            let w = lobe.widest.clamp(0.05, 0.95);
            let shape = if t < w {
                // A blade that runs on behind its stalk rounds into its base; a lobe
                // of a palmate leaf is merged into its neighbours there and can be full.
                (t / w * std::f32::consts::FRAC_PI_2)
                    .sin()
                    .powf(if lobe.start > 0.0 { 1.4 } else { 0.7 })
            } else {
                let s = (t - w) / (1.0 - w);
                (1.0 - s * s).max(0.0).sqrt() * (1.0 - 0.25 * s)
            };
            let mut half = lobe.width * lobe.length * shape;
            if self.teeth > 0.0 {
                // Teeth along the margin, pointing toward the tip.
                let phase = (u * self.teeth).fract();
                half -= self.tooth_depth * (phase * phase);
            }
            let inside = half - v.abs();
            blade = smooth_max(blade, inside, self.merge);

            // The midrib, fading toward the lobe's point.
            let rib_w = 0.006 * (1.0 - 0.7 * t);
            vein = vein.max((-(v / rib_w).powi(2)).exp() * (t < 0.98) as u8 as f32);
            if self.veins > 0.0 && inside > 0.0 {
                // Secondary veins leave the midrib pointing toward the tip and curve
                // out to the margin.
                let reach = (v.abs() / half.max(1e-3)).clamp(0.0, 1.0);
                let line = (u - v.abs() * 0.75 * (1.0 - 0.3 * reach)) * self.veins;
                let d = (line - line.round()).abs() / self.veins;
                let width = 0.003 * (1.0 - 0.6 * reach);
                vein = vein.max((-(d / width).powi(2)).exp() * (1.0 - reach).powf(0.6));
            }
            // Each lobe domes between its midrib and its margin.
            if inside > 0.0 {
                let r = (v.abs() / half.max(1e-3)).clamp(0.0, 1.0);
                dome = dome.max((1.0 - r * r) * shape);
            }
        }
        if self.notch_deg > 0.0 && blade > 0.0 {
            // The cleft at the base: everything inside a V opening down from the
            // junction is cut away.
            let a = self.notch_deg.to_radians();
            let cleft = p.x.abs() * a.cos() - (self.junction - p.y) * a.sin();
            blade = blade.min(cleft);
        }
        // The stalk, from the bottom of the cell to the blade.
        let stalk_w = 0.012;
        let stalk = if p.y <= self.junction + 0.02 && p.y >= 0.0 {
            stalk_w - p.x.abs()
        } else {
            f32::NEG_INFINITY
        };
        let inside = blade.max(stalk);
        if stalk > blade {
            vein = 1.0;
        }
        Sample {
            inside,
            vein: vein.clamp(0.0, 1.0),
            height: dome,
        }
    }

    /// Paints the two cells: the underside on the left, the lit face on the right.
    pub fn paint(&self) -> BakedMaps {
        let n = self.cell;
        let (w, h) = (n * 2, n);
        let mut albedo = Bitmap::new(w, h);
        let mut normal = Bitmap::new(w, h);
        let mut rough = Bitmap::new(w, h);
        let px = 1.0 / n as f32;

        // Heights first, so the normal map can difference them.
        let mut heights = vec![0.0f32; (n * n) as usize];
        let mut samples = Vec::with_capacity((n * n) as usize);
        for y in 0..n {
            for x in 0..n {
                let p = Vec2::new((x as f32 + 0.5) * px - 0.5, 1.0 - (y as f32 + 0.5) * px);
                let s = self.sample(p);
                heights[(y * n + x) as usize] = s.height * 0.02 - s.vein * 0.004;
                samples.push(s);
            }
        }
        let height_at = |x: i32, y: i32| {
            let x = x.clamp(0, n as i32 - 1) as u32;
            let y = y.clamp(0, n as i32 - 1) as u32;
            heights[(y * n + x) as usize]
        };

        let srgb = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        for y in 0..n {
            for x in 0..n {
                let s = &samples[(y * n + x) as usize];
                let coverage = (s.inside / px + 0.5).clamp(0.0, 1.0);
                let p = Vec2::new(x as f32, y as f32) * px;
                let mottle = (value_noise(p * 14.0, self.seed) - 0.5) * 0.6
                    + (value_noise(p * 45.0, self.seed + 1) - 0.5) * 0.4;
                let shade = 1.0 + mottle * self.mottle;
                // Toward the margin the blade thins and goes a little lighter.
                let rim = 1.0 + 0.12 * (1.0 - (s.inside / 0.03).clamp(0.0, 1.0));

                // Tangent-space normal from the height: x across the cell, y up it.
                let dx = (height_at(x as i32 + 1, y as i32) - height_at(x as i32 - 1, y as i32))
                    / (2.0 * px);
                let dy = (height_at(x as i32, y as i32 - 1) - height_at(x as i32, y as i32 + 1))
                    / (2.0 * px);

                for face in 0..2u32 {
                    let top = face == 1;
                    let (base, rough_v) = if top {
                        (self.top, self.top_roughness)
                    } else {
                        (self.under, self.under_roughness)
                    };
                    let vein_c = if top {
                        self.vein
                    } else {
                        // On the underside the veins stand out paler still.
                        [
                            (self.under[0] + 0.12).min(1.0),
                            (self.under[1] + 0.12).min(1.0),
                            (self.under[2] + 0.1).min(1.0),
                        ]
                    };
                    let k = s.vein * self.vein_contrast;
                    let mut c = [0.0f32; 3];
                    for i in 0..3 {
                        c[i] = (base[i] * shade * rim) * (1.0 - k) + vein_c[i] * k;
                    }
                    // The underside mirrors the top, as turning a leaf over does.
                    let ox = if top { n + x } else { n - 1 - x };
                    let i = ((y * w + ox) * 4) as usize;
                    // Outside the leaf the colour is carried on, so filtering the edge
                    // does not pull in black.
                    albedo.pixels[i..i + 4].copy_from_slice(&[
                        srgb(c[0]),
                        srgb(c[1]),
                        srgb(c[2]),
                        srgb(coverage),
                    ]);
                    // Veins are sunk on the top face and stand proud underneath.
                    let (nx, ny) = if top { (-dx, -dy) } else { (-dx, dy) };
                    let nv = glam::Vec3::new(nx, ny, 1.0).normalize();
                    normal.pixels[i..i + 4].copy_from_slice(&[
                        srgb(nv.x * 0.5 + 0.5),
                        srgb(nv.y * 0.5 + 0.5),
                        srgb(nv.z * 0.5 + 0.5),
                        255,
                    ]);
                    let r = (rough_v + s.vein * 0.12).clamp(0.0, 1.0);
                    rough.pixels[i..i + 4].copy_from_slice(&[srgb(r), srgb(r), srgb(r), 255]);
                }
            }
        }
        BakedMaps {
            albedo,
            normal: Some(normal),
            roughness: Some(rough),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Share of a cell the leaf covers.
    fn coverage(maps: &BakedMaps, cell: u32) -> f32 {
        let mut sum = 0.0;
        for y in 0..maps.albedo.height {
            for x in cell * maps.albedo.height..(cell + 1) * maps.albedo.height {
                let i = ((y * maps.albedo.width + x) * 4 + 3) as usize;
                sum += maps.albedo.pixels[i] as f32 / 255.0;
            }
        }
        sum / (maps.albedo.height * maps.albedo.height) as f32
    }

    #[test]
    fn every_leaf_fills_a_fair_share_of_its_cell_on_both_faces() {
        for mut art in LeafArt::all() {
            art.cell = 128;
            let maps = art.paint();
            assert_eq!(maps.albedo.width, 256);
            let (under, top) = (coverage(&maps, 0), coverage(&maps, 1));
            assert!((under - top).abs() < 0.01, "{}: the faces differ", art.name);
            assert!((0.12..0.7).contains(&top), "{} covers {top:.2} of its cell", art.name);
        }
    }

    #[test]
    fn the_stalk_meets_the_bottom_of_the_cell_in_the_middle() {
        // A card hangs from the middle of its bottom edge, so that is where the leaf
        // has to be joined to its twig.
        for mut art in LeafArt::all() {
            art.cell = 128;
            let maps = art.paint();
            let bottom = maps.albedo.height - 1;
            let alpha = |x: u32| maps.albedo.pixels[((bottom * maps.albedo.width + x) * 4 + 3) as usize];
            assert!(alpha(128 + 64) > 128, "{}: no stalk at the foot of the lit face", art.name);
            assert!(alpha(128 + 8) < 16, "{}: the leaf runs off the bottom of its cell", art.name);
        }
    }
}
