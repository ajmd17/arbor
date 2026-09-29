//! Which point of the surface each texel of a texture shows: the triangles laid out in
//! UV space and the texel centres looked up in them.

use glam::{Vec2, Vec3};

use super::scene::Scene;

pub const NONE: u32 = u32::MAX;

/// A texel's surface point: a triangle and the barycentric weights of its second and
/// third corner.
#[derive(Clone, Copy)]
pub struct TexelRef {
    pub tri: u32,
    pub b1: f32,
    pub b2: f32,
}

pub struct TexelMap {
    pub width: usize,
    pub height: usize,
    pub texels: Vec<TexelRef>,
    /// Texels claimed by more than one triangle, each counted once: a sign of overlapping
    /// or tiled UV islands.
    pub overlapped: usize,
    /// Triangles with UVs outside the 0..1 square, which are clipped to it.
    pub outside: usize,
}

/// The surface at a texel.
#[derive(Clone, Copy)]
pub struct Surface {
    pub tri: u32,
    pub p: Vec3,
    pub n: Vec3,
    pub uv: Vec2,
    /// Barycentric weights of the triangle's three corners.
    pub w: [f32; 3],
}

impl TexelMap {
    pub fn covered(&self, x: usize, y: usize) -> bool {
        self.texels[y * self.width + x].tri != NONE
    }

    pub fn coverage(&self) -> f32 {
        self.texels.iter().filter(|t| t.tri != NONE).count() as f32 / self.texels.len().max(1) as f32
    }

    /// The surface at texel `i`, if a triangle covers it.
    pub fn surface(&self, scene: &Scene, i: usize) -> Option<Surface> {
        let r = self.texels[i];
        if r.tri == NONE {
            return None;
        }
        let tri = &scene.tris[r.tri as usize];
        let w = [1.0 - r.b1 - r.b2, r.b1, r.b2];
        let c = &scene.corners[r.tri as usize];
        let mix3 = |v: [Vec3; 3]| v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
        Some(Surface {
            tri: r.tri,
            p: mix3(*c),
            n: mix3(tri.n).normalize_or(tri.n[0]),
            uv: tri.uv[0] * w[0] + tri.uv[1] * w[1] + tri.uv[2] * w[2],
            w,
        })
    }
}

/// Lays out the triangles of `material` over a `width` by `height` texture. Row 0 is the
/// top, where v = 0 is, as in glTF.
pub fn rasterize(scene: &Scene, material: u32, width: usize, height: usize) -> TexelMap {
    let mut map = TexelMap {
        width,
        height,
        texels: vec![TexelRef { tri: NONE, b1: 0.0, b2: 0.0 }; width * height],
        overlapped: 0,
        outside: 0,
    };
    let mut claimed_twice = vec![false; width * height];
    let (w, h) = (width as f32, height as f32);
    for (index, tri) in scene.tris.iter().enumerate() {
        if tri.material != material || !tri.has_uv {
            continue;
        }
        let q = tri.uv.map(|uv| Vec2::new(uv.x * w, uv.y * h));
        let area = (q[1] - q[0]).perp_dot(q[2] - q[0]);
        if area.abs() < 1e-9 {
            continue;
        }
        let lo = q[0].min(q[1]).min(q[2]);
        let hi = q[0].max(q[1]).max(q[2]);
        if lo.x < 0.0 || lo.y < 0.0 || hi.x > w || hi.y > h {
            map.outside += 1;
        }
        let (x0, x1) = ((lo.x.floor().max(0.0)) as usize, (hi.x.ceil().min(w)) as usize);
        let (y0, y1) = ((lo.y.floor().max(0.0)) as usize, (hi.y.ceil().min(h)) as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                // Weights of the second and third corner.
                let b1 = (c - q[0]).perp_dot(q[2] - q[0]) / area;
                let b2 = (q[1] - q[0]).perp_dot(c - q[0]) / area;
                if b1 < 0.0 || b2 < 0.0 || b1 + b2 > 1.0 {
                    continue;
                }
                let slot = &mut map.texels[y * width + x];
                if slot.tri != NONE && !claimed_twice[y * width + x] {
                    claimed_twice[y * width + x] = true;
                    map.overlapped += 1;
                }
                *slot = TexelRef { tri: index as u32, b1, b2 };
            }
        }
    }
    map
}
