//! What each kind of map puts in a texel, given the surface there.

use std::f32::consts::TAU;

use glam::Vec3;

use super::bvh::Ray;
use super::raster::{Surface, TexelMap};
use super::scene::{sample_nearest, Scene};
use super::{BakeParams, MapKind};
use crate::import::Slot;

/// Everything a map needs to shade a texel, shared across them.
pub struct Context<'a> {
    pub scene: &'a Scene,
    pub map: &'a TexelMap,
    pub params: &'a BakeParams,
    pub material: usize,
}

impl Context<'_> {
    pub fn shade(&self, kind: MapKind, s: &Surface, texel: usize, x: usize, y: usize) -> [f32; 4] {
        let rgb = |v: Vec3| [v.x, v.y, v.z, 1.0];
        match kind {
            MapKind::NormalWorld => rgb(s.n * 0.5 + Vec3::splat(0.5)),
            MapKind::NormalTangent => rgb(self.tangent_space_normal(s) * 0.5 + Vec3::splat(0.5)),
            MapKind::Position => {
                let size = (self.scene.max - self.scene.min).max(Vec3::splat(1e-6));
                rgb((s.p - self.scene.min) / size)
            }
            MapKind::ObjectId => rgb(id_color(self.scene.tris[s.tri as usize].instance)),
            MapKind::VertexColor => match &self.scene.tris[s.tri as usize].color {
                Some(_) => rgb(self.vertex_color(s)),
                None => [1.0; 4],
            },
            MapKind::AmbientOcclusion => {
                let v = self.occlusion(s, texel);
                [v, v, v, 1.0]
            }
            MapKind::Thickness => {
                let v = self.thickness(s, texel);
                [v, v, v, 1.0]
            }
            MapKind::Curvature => {
                let v = self.curvature(s, x, y);
                [v, v, v, 1.0]
            }
        }
    }

    fn vertex_color(&self, s: &Surface) -> Vec3 {
        let c = self.scene.tris[s.tri as usize].color.unwrap();
        (c[0] * s.w[0] + c[1] * s.w[1] + c[2] * s.w[2]).truncate()
    }

    /// The surface normal in tangent space: the material's normal map where it has one,
    /// straight up the surface where it does not.
    fn tangent_space_normal(&self, s: &Surface) -> Vec3 {
        let mat = &self.scene.source.materials[self.material];
        let Some(&(_, image)) = mat.textures.iter().find(|(slot, _)| *slot == Slot::Normal) else {
            return Vec3::Z;
        };
        let Some(Some(img)) = self.scene.source.images.get(image) else { return Vec3::Z };
        let px = sample_nearest(img, s.uv);
        let mut n = Vec3::new(px[0] as f32, px[1] as f32, px[2] as f32) / 255.0 * 2.0 - Vec3::ONE;
        n.x *= mat.normal_scale;
        n.y *= mat.normal_scale;
        n.normalize_or(Vec3::Z)
    }

    /// The share of the hemisphere above the surface that nothing blocks, within the
    /// occluder distance.
    fn occlusion(&self, s: &Surface, texel: usize) -> f32 {
        let scene = self.scene;
        let reach = self.params.ao_distance * scene.diagonal();
        let origin = s.p + s.n * (scene.diagonal() * 1e-4);
        let n = self.params.ao_rays.max(1);
        let mut blocked = 0;
        for k in 0..n {
            let dir = hemisphere_sample(s.n, k, n, texel as u32);
            if scene.blocked(&Ray { origin, dir }, reach, s.tri) {
                blocked += 1;
            }
        }
        1.0 - blocked as f32 / n as f32
    }

    /// How far, as a share of the reach, the surface is from open air behind it: rays
    /// into the surface, their hit distances averaged, with none capped at the reach.
    /// White is thick.
    fn thickness(&self, s: &Surface, texel: usize) -> f32 {
        let scene = self.scene;
        let reach = self.params.thickness_distance * scene.diagonal();
        let origin = s.p - s.n * (scene.diagonal() * 1e-4);
        let n = self.params.ao_rays.max(1);
        let mut total = 0.0;
        for k in 0..n {
            let dir = hemisphere_sample(-s.n, k, n, texel as u32 ^ 0x5bd1e995);
            total += scene.nearest(&Ray { origin, dir }, reach, s.tri).unwrap_or(reach);
        }
        (total / n as f32 / reach).clamp(0.0, 1.0)
    }

    /// Mean curvature, 0.5 flat, toward white where the surface is convex and toward
    /// black where it is concave. Read off how the normal turns between texels a set
    /// distance apart on the surface, so a hard edge shows as a line about that wide and
    /// detail finer than it averages out.
    fn curvature(&self, s: &Surface, x: usize, y: usize) -> f32 {
        let scene = self.scene;
        let tri = &scene.tris[s.tri as usize];
        let radius = self.params.curvature_radius * scene.diagonal();
        let (w, h) = (self.map.width as f32, self.map.height as f32);
        let mut sum = 0.0;
        // Along u and along v, each with the world length of one texel that way.
        for (dx, dy, step) in [(1, 0, tri.dpdu / w), (0, 1, tri.dpdv / h)] {
            let texel = step.length();
            if texel < 1e-12 {
                continue;
            }
            let r = ((radius / texel).round() as isize).clamp(1, 64);
            let reach = 2.0 * texel * r as f32;
            let at = |sign: isize| {
                let (nx, ny) = (x as isize + dx * r * sign, y as isize + dy * r * sign);
                if nx < 0 || ny < 0 || nx >= self.map.width as isize || ny >= self.map.height as isize {
                    return None;
                }
                // Far from where a step that long could reach is another island.
                self.map
                    .surface(scene, ny as usize * self.map.width + nx as usize)
                    .filter(|o| (o.p - s.p).length() <= reach)
            };
            let (a, b) = match (at(-1), at(1)) {
                (Some(a), Some(b)) => (a, b),
                (Some(a), None) => (a, *s),
                (None, Some(b)) => (*s, b),
                (None, None) => continue,
            };
            let dp = b.p - a.p;
            let len2 = dp.length_squared();
            if len2 > 1e-18 {
                sum += (b.n - a.n).dot(dp) / len2;
            }
        }
        let mean = sum * 0.5;
        let gain = self.params.curvature_strength * scene.diagonal() * 0.025;
        0.5 + 0.5 * (mean * gain).tanh()
    }
}

/// A distinct colour per object, by golden-angle hue.
fn id_color(instance: u32) -> Vec3 {
    let hue = (instance as f32 * 0.618_034).fract() * 6.0;
    let f = |n: f32| {
        let k = (n + hue) % 6.0;
        0.85 - 0.85 * 0.75 * k.min(4.0 - k).clamp(0.0, 1.0)
    };
    Vec3::new(f(5.0), f(3.0), f(1.0))
}

fn pcg(x: u32) -> u32 {
    let s = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

fn rand01(x: u32) -> f32 {
    (pcg(x) >> 8) as f32 / 16_777_216.0
}

/// The van der Corput sequence.
fn radical_inverse(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

/// The `k`th of `n` cosine-weighted directions about `normal`. A Hammersley set, shifted
/// differently at each texel (`seed`) so the error is noise rather than banding.
fn hemisphere_sample(normal: Vec3, k: u32, n: u32, seed: u32) -> Vec3 {
    let (s1, s2) = (rand01(seed.wrapping_mul(2).wrapping_add(1)), rand01(seed.wrapping_mul(2).wrapping_add(2)));
    let u1 = (k as f32 / n as f32 + s1).fract();
    let u2 = (radical_inverse(k) + s2).fract();
    let (r, phi) = (u1.sqrt(), TAU * u2);
    let (a, b) = normal.any_orthonormal_pair();
    let dir = a * (r * phi.cos()) + b * (r * phi.sin()) + normal * (1.0 - u1).max(0.0).sqrt();
    dir.normalize_or(normal)
}
