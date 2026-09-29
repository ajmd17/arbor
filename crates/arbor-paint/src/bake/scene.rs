//! The model as the baker sees it: every visible triangle in world space, with what the
//! bake needs at each corner, and a tree to cast rays at.

use std::sync::Arc;

use glam::{Mat3, Vec2, Vec3, Vec4};

use super::bvh::{Bvh, Ray, Tri};
use crate::import::{ImportedModel, Slot};

pub struct SceneTri {
    pub n: [Vec3; 3],
    pub uv: [Vec2; 3],
    pub color: Option<[Vec4; 3]>,
    pub instance: u32,
    pub material: u32,
    /// Whether the mesh has UVs to bake into.
    pub has_uv: bool,
    /// World-space change of position per unit of u and of v.
    pub dpdu: Vec3,
    pub dpdv: Vec3,
}

pub struct Scene {
    pub source: Arc<ImportedModel>,
    pub tris: Vec<SceneTri>,
    /// The triangles' corners, in the order `bvh` wants them.
    pub corners: Vec<Tri>,
    bvh: Bvh,
    pub min: Vec3,
    pub max: Vec3,
}

impl Scene {
    /// `visible[i]` says whether instance `i` of the model is in the scene.
    pub fn build(source: Arc<ImportedModel>, visible: &[bool]) -> Result<Self, String> {
        let mut tris = Vec::new();
        let mut corners: Vec<Tri> = Vec::new();
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for (i, inst) in source.instances.iter().enumerate() {
            if !visible.get(i).copied().unwrap_or(true) {
                continue;
            }
            let prim = &source.primitives[inst.primitive];
            let m = inst.transform;
            let normal_mat = Mat3::from_mat4(m).inverse().transpose();
            let world_p: Vec<Vec3> = prim.positions.iter().map(|&p| m.transform_point3(Vec3::from(p))).collect();
            let world_n: Vec<Vec3> =
                prim.normals.iter().map(|&n| (normal_mat * Vec3::from(n)).normalize_or(Vec3::Y)).collect();
            for tri in prim.indices.chunks(3) {
                let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
                let p = idx.map(|k| world_p[k]);
                for q in p {
                    min = min.min(q);
                    max = max.max(q);
                }
                let uv = idx.map(|k| prim.uvs.as_ref().map_or(Vec2::ZERO, |u| Vec2::from(u[k])));
                let color = prim.colors.as_ref().map(|c| idx.map(|k| Vec4::from(c[k])));
                let (dpdu, dpdv) = uv_derivatives(p, uv);
                tris.push(SceneTri {
                    n: idx.map(|k| world_n[k]),
                    uv,
                    color,
                    instance: i as u32,
                    material: prim.material as u32,
                    has_uv: prim.uvs.is_some(),
                    dpdu,
                    dpdv,
                });
                corners.push(p);
            }
        }
        if tris.is_empty() {
            return Err("there is nothing visible to bake".into());
        }
        let bvh = Bvh::build(&corners);
        Ok(Self { source, tris, corners, bvh, min, max })
    }

    /// The length of the box's diagonal: what distances are measured against.
    pub fn diagonal(&self) -> f32 {
        (self.max - self.min).length().max(1e-6)
    }

    /// Whether a ray hitting triangle `tri` at barycentric `(u, v)` is stopped by it.
    /// Cut-outs let light through where their texture is transparent, and materials that
    /// blend stop nothing.
    fn occludes(&self, tri: u32, u: f32, v: f32) -> bool {
        let t = &self.tris[tri as usize];
        let mat = &self.source.materials[t.material as usize];
        if mat.blend {
            return false;
        }
        if !mat.mask {
            return true;
        }
        let w = 1.0 - u - v;
        let uv = t.uv[0] * w + t.uv[1] * u + t.uv[2] * v;
        let mut alpha = mat.base_color[3];
        if let Some(c) = &t.color {
            alpha *= c[0].w * w + c[1].w * u + c[2].w * v;
        }
        if let Some(&(_, image)) = mat.textures.iter().find(|(s, _)| *s == Slot::BaseColor)
            && let Some(Some(img)) = self.source.images.get(image)
        {
            alpha *= sample_nearest(img, uv)[3] as f32 / 255.0;
        }
        alpha >= mat.alpha_cutoff
    }

    /// Whether anything is in the way of `ray` within `tmax`, other than triangle `ignore`.
    pub fn blocked(&self, ray: &Ray, tmax: f32, ignore: u32) -> bool {
        self.bvh.any(&self.corners, ray, tmax, ignore, &mut |t, u, v| self.occludes(t, u, v))
    }

    /// The nearest thing in the way of `ray` within `tmax`.
    pub fn nearest(&self, ray: &Ray, tmax: f32, ignore: u32) -> Option<f32> {
        self.bvh.closest(&self.corners, ray, tmax, ignore, &mut |t, u, v| self.occludes(t, u, v))
    }
}

/// How position changes with u and with v across a triangle; zero for one with no area
/// in UV space.
fn uv_derivatives(p: [Vec3; 3], uv: [Vec2; 3]) -> (Vec3, Vec3) {
    let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
    let (d1, d2) = (uv[1] - uv[0], uv[2] - uv[0]);
    let det = d1.x * d2.y - d2.x * d1.y;
    if det.abs() < 1e-20 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    ((e1 * d2.y - e2 * d1.y) / det, (e2 * d1.x - e1 * d2.x) / det)
}

/// The texel `uv` falls in, wrapping.
pub fn sample_nearest(img: &crate::import::Image, uv: Vec2) -> [u8; 4] {
    let x = ((uv.x.rem_euclid(1.0) * img.width as f32) as u32).min(img.width - 1);
    let y = ((uv.y.rem_euclid(1.0) * img.height as f32) as u32).min(img.height - 1);
    let i = ((y * img.width + x) * 4) as usize;
    [img.rgba[i], img.rgba[i + 1], img.rgba[i + 2], img.rgba[i + 3]]
}
