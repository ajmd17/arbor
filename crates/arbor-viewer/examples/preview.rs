//! Renders a species to a PNG without opening a window, so a tree can be checked
//! from a terminal or a script.
//!
//! cargo run --release -p arbor-viewer --example preview -- \
//!     <pine|oak|path.ron> out.png [--seed N] [--yaw R] [--size N] [--zoom F]
//!     [--aim 0..1] [--no-leaves] [--plain-mips]
//!
//! This is a deliberately small software rasteriser that mirrors what the viewer
//! shaders do — alpha-tested leaf cards sampled from their atlas, wrapped diffuse and
//! backlit transmission — not a second renderer to keep in sync feature by feature.
//!
//! It samples a real mip chain, chosen per triangle, because the way a leaf texture
//! is minified is exactly what decides whether a canopy survives being zoomed away
//! from. `--plain-mips` swaps the coverage-preserving chain for a plain box-filtered
//! one, which is what the disappearing-canopy bug looked like.

#[path = "../src/mipmap.rs"]
mod mipmap;

use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_leaves, build_mesh, grow, LeafMesh, Mesh, SpeciesParams};
use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use image::{Rgb, RgbImage};

const TEXTURE_DIR: &str = "assets/textures";
const ALPHA_CUTOFF: f32 = 0.35;
const SKY: Vec3 = Vec3::new(0.52, 0.65, 0.84);

struct Target {
    color: Vec<Vec3>,
    depth: Vec<f32>,
    size: usize,
}

/// A vertex after projection, carrying whatever the shading needs.
#[derive(Clone, Copy)]
struct Vert {
    clip: Vec4,
    world: Vec3,
    normal: Vec3,
    uv: Vec2,
    tint: Vec4,
    /// Size in texels of the region this triangle samples, for picking a mip level.
    tex_size: Vec2,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().cloned().unwrap_or_else(|| "oak".into());
    let out = args.get(1).cloned().unwrap_or_else(|| "preview.png".into());
    let flag = |key: &str| args.iter().position(|a| a == key).and_then(|i| args.get(i + 1));
    let yaw: f32 = flag("--yaw").and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let size: usize = flag("--size").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let want_leaves = !args.iter().any(|a| a == "--no-leaves");
    let preserve_coverage = !args.iter().any(|a| a == "--plain-mips");

    let src = builtin_presets()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s.to_string())
        .unwrap_or_else(|| std::fs::read_to_string(&name).expect("read species"));
    let mut params: SpeciesParams = parse_species(&src).expect("parse species");
    if let Some(seed) = flag("--seed").and_then(|s| s.parse().ok()) {
        params.seed = seed;
    }
    params.leaves.enabled &= want_leaves;

    let sk = grow(&params);
    let mesh = build_mesh(&sk, &params);
    let leaves = build_leaves(&sk, &params);
    let bark = Tex::load(
        &format!("{TEXTURE_DIR}/{}_albedo.png", params.mesh.bark_texture),
        false,
    );
    let leaf_tex = Tex::load(
        &format!("{TEXTURE_DIR}/{}_albedo.png", params.leaves.texture),
        preserve_coverage,
    );

    let (min, max) = mesh.aabb();
    let mut lo = Vec3::from(min);
    let mut hi = Vec3::from(max);
    for p in &leaves.positions {
        lo = lo.min(Vec3::from(*p));
        hi = hi.max(Vec3::from(*p));
    }
    let zoom: f32 = flag("--zoom").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let aim: f32 = flag("--aim").and_then(|s| s.parse().ok()).unwrap_or(-1.0);
    let mut center = (lo + hi) * 0.5;
    if aim >= 0.0 {
        center.y = lo.y + (hi.y - lo.y) * aim;
    }
    let extent = (hi - lo).length().max(1.0);
    let dist = extent * 0.95 / zoom.max(0.05);
    let eye = center + Vec3::new(yaw.sin() * dist, extent * 0.08 / zoom.max(0.05), yaw.cos() * dist);
    let vp = Mat4::perspective_rh(50f32.to_radians(), 1.0, 0.05, dist * 6.0)
        * Mat4::look_at_rh(eye, center, Vec3::Y);
    let sun = Vec3::new(0.45, 0.72, 0.52).normalize();

    let mut target = Target {
        color: vec![SKY; size * size],
        depth: vec![f32::INFINITY; size * size],
        size,
    };
    draw_bark(&mut target, &mesh, vp, eye, sun, bark.as_ref());
    if params.leaves.enabled {
        draw_leaves(&mut target, &leaves, &params, vp, eye, sun, leaf_tex.as_ref());
    }

    let mut img = RgbImage::new(size as u32, size as u32);
    for (i, c) in target.color.iter().enumerate() {
        let srgb = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        img.put_pixel(
            (i % size) as u32,
            (i / size) as u32,
            Rgb([srgb(c.x), srgb(c.y), srgb(c.z)]),
        );
    }
    img.save(&out).expect("write png");
    println!(
        "{}: {} tris bark, {} leaves -> {out}",
        params.name,
        mesh.triangle_count(),
        leaves.leaf_count()
    );
}

/// A texture and its mip chain, sampled with a per-triangle level of detail.
struct Tex {
    levels: Vec<mipmap::MipLevel>,
}

impl Tex {
    fn load(path: &str, preserve_coverage: bool) -> Option<Tex> {
        let img = image::open(path).ok()?.to_rgba8();
        let (w, h) = (img.width(), img.height());
        let px = img.into_raw();
        let levels = if preserve_coverage {
            mipmap::coverage_preserving_chain(&px, w, h, ALPHA_CUTOFF)
        } else {
            let mut chain: Vec<mipmap::MipLevel> = vec![(px, w, h)];
            while chain.last().map(|(_, w, h)| *w > 1 || *h > 1) == Some(true) {
                let (s, sw, sh) = chain.last().expect("chain is never empty").clone();
                chain.push(mipmap::halve(&s, sw, sh));
            }
            chain
        };
        println!("  texture {path} {w}x{h}, {} mips", levels.len());
        Some(Tex { levels })
    }

    fn texel(&self, level: usize, x: i64, y: i64) -> Vec4 {
        let (px, w, h) = &self.levels[level.min(self.levels.len() - 1)];
        let x = x.clamp(0, *w as i64 - 1) as u32;
        let y = y.clamp(0, *h as i64 - 1) as u32;
        let i = ((y * w + x) * 4) as usize;
        // The textures are sRGB; shading happens in linear space.
        let lin = |v: u8| (v as f32 / 255.0).powf(2.2);
        Vec4::new(
            lin(px[i]),
            lin(px[i + 1]),
            lin(px[i + 2]),
            px[i + 3] as f32 / 255.0,
        )
    }

    fn bilinear(&self, level: usize, uv: Vec2) -> Vec4 {
        let level = level.min(self.levels.len() - 1);
        let (_, w, h) = self.levels[level];
        let fx = uv.x.rem_euclid(1.0) * w as f32 - 0.5;
        let fy = uv.y.rem_euclid(1.0) * h as f32 - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = self.texel(level, x0, y0).lerp(self.texel(level, x0 + 1, y0), tx);
        let bot = self
            .texel(level, x0, y0 + 1)
            .lerp(self.texel(level, x0 + 1, y0 + 1), tx);
        top.lerp(bot, ty)
    }

    /// Trilinear: blends the two levels around `lod`, as the GPU does.
    fn sample(&self, uv: Vec2, lod: f32) -> Vec4 {
        let top = self.levels.len() as f32 - 1.0;
        let lod = lod.clamp(0.0, top);
        let lo = lod.floor();
        let frac = lod - lo;
        let a = self.bilinear(lo as usize, uv);
        if frac < 1e-3 {
            return a;
        }
        a.lerp(self.bilinear(lo as usize + 1, uv), frac)
    }
}

fn sample(tex: Option<&Tex>, uv: Vec2, lod: f32) -> Vec4 {
    match tex {
        Some(t) => t.sample(uv, lod),
        None => Vec4::new(0.45, 0.38, 0.3, 1.0),
    }
}

/// Mip level for a triangle, from how many texels it covers per pixel. Per triangle
/// rather than per pixel, which for leaf cards is a couple of pixels wide is exact
/// enough to show the minification behaviour.
fn triangle_lod(uv: [Vec2; 3], screen_area: f32, tex_w: f32, tex_h: f32) -> f32 {
    let e1 = uv[1] - uv[0];
    let e2 = uv[2] - uv[0];
    let uv_area = (e1.x * e2.y - e1.y * e2.x).abs() * tex_w * tex_h;
    if screen_area.abs() < 1e-9 || uv_area <= 0.0 {
        return 0.0;
    }
    0.5 * (uv_area / screen_area.abs()).log2()
}

fn project(vp: Mat4, world: Vec3, normal: Vec3, uv: Vec2, tint: Vec4, tex_size: Vec2) -> Vert {
    Vert {
        clip: vp * world.extend(1.0),
        world,
        normal,
        uv,
        tint,
        tex_size,
    }
}

fn tex_dims(tex: Option<&Tex>) -> Vec2 {
    match tex {
        Some(t) => Vec2::new(t.levels[0].1 as f32, t.levels[0].2 as f32),
        None => Vec2::splat(1.0),
    }
}

fn draw_bark(t: &mut Target, mesh: &Mesh, vp: Mat4, eye: Vec3, sun: Vec3, tex: Option<&Tex>) {
    let dims = tex_dims(tex);
    for tri in mesh.indices.chunks_exact(3) {
        let v: Vec<Vert> = tri
            .iter()
            .map(|&i| {
                let i = i as usize;
                project(
                    vp,
                    Vec3::from(mesh.positions[i]),
                    Vec3::from(mesh.normals[i]),
                    Vec2::from(mesh.uvs[i]) * 0.5,
                    Vec4::ONE,
                    dims,
                )
            })
            .collect();
        raster(t, &v, |f, lod| {
            let albedo = sample(tex, f.uv, lod).xyz();
            let mut n = f.normal.normalize_or_zero();
            let view = (eye - f.world).normalize_or_zero();
            if n.dot(view) < 0.0 {
                n = -n;
            }
            let ndl = n.dot(sun).max(0.0);
            let hemi = 0.18 + 0.16 * (n.y * 0.5 + 0.5);
            Some(albedo * (hemi + ndl * 1.15))
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_leaves(
    t: &mut Target,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    vp: Mat4,
    eye: Vec3,
    sun: Vec3,
    tex: Option<&Tex>,
) {
    let dims = tex_dims(tex);
    let lp = &params.leaves;
    let cols = lp.atlas_cols.max(1) as f32;
    let rows = lp.atlas_rows.max(1) as f32;
    let scale = Vec2::new(1.0 / cols, 1.0 / rows);
    let cell = |index: u32| {
        let i = (index as f32).min(cols * rows - 1.0);
        Vec2::new((i % cols) / cols, (i / cols).floor() / rows)
    };
    let front = cell(lp.atlas_front);
    let back = cell(lp.atlas_back);

    for tri in leaves.indices.chunks_exact(3) {
        let v: Vec<Vert> = tri
            .iter()
            .map(|&i| {
                let i = i as usize;
                project(
                    vp,
                    Vec3::from(leaves.positions[i]),
                    Vec3::from(leaves.normals[i]),
                    // The card samples one atlas cell, so a card-local step covers
                    // only that fraction of the texture when picking a mip level.
                    Vec2::from(leaves.uvs[i]) * scale,
                    Vec4::from(leaves.tints[i]),
                    dims,
                )
            })
            .collect();
        raster(t, &v, |f, lod| {
            let mut n = f.normal.normalize_or_zero();
            let view = (eye - f.world).normalize_or_zero();
            // The shader picks the atlas cell from the facing of the triangle; here
            // the same decision comes from which side the camera is on.
            let facing_camera = n.dot(view) >= 0.0;
            if !facing_camera {
                n = -n;
            }
            let base = if facing_camera { front } else { back };
            let texel = sample(tex, base + f.uv, lod);
            if texel.w < ALPHA_CUTOFF {
                return None;
            }
            let albedo = texel.xyz() * f.tint.xyz();
            let wrapped = ((n.dot(sun) + 0.5) / 1.5).max(0.0);
            let through = (-n).dot(sun).max(0.0);
            let lobe = 0.35 + 0.65 * view.dot(-sun).max(0.0).powi(3);
            let transmitted = albedo * 0.9 * through * lobe;
            let hemi = 0.14 + 0.16 * (n.y * 0.5 + 0.5);
            Some((albedo * wrapped + transmitted) * 1.35 + albedo * hemi)
        });
    }
}

/// Scanline fill with a depth test and perspective-correct attributes. `shade`
/// returns None for a fragment the alpha test rejects.
fn raster(t: &mut Target, v: &[Vert], shade: impl Fn(&Vert, f32) -> Option<Vec3>) {
    let size = t.size as f32;
    let mut screen = [Vec3::ZERO; 3];
    for (k, vert) in v.iter().enumerate() {
        if vert.clip.w <= 1e-4 {
            return;
        }
        let ndc = vert.clip.xyz() / vert.clip.w;
        screen[k] = Vec3::new(
            (ndc.x * 0.5 + 0.5) * size,
            (1.0 - (ndc.y * 0.5 + 0.5)) * size,
            vert.clip.w,
        );
    }
    let area = edge(screen[0], screen[1], screen[2]);
    if area.abs() < 1e-7 {
        return;
    }
    let uv_tri = [v[0].uv, v[1].uv, v[2].uv];
    let lod = triangle_lod(uv_tri, area, v[0].tex_size.x, v[0].tex_size.y);
    let min_x = screen.iter().map(|p| p.x).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let max_x = screen.iter().map(|p| p.x).fold(f32::MIN, f32::max).ceil().min(size - 1.0);
    let min_y = screen.iter().map(|p| p.y).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let max_y = screen.iter().map(|p| p.y).fold(f32::MIN, f32::max).ceil().min(size - 1.0);
    if max_x < 0.0 || max_y < 0.0 || min_x as f32 > max_x || min_y as f32 > max_y {
        return;
    }

    for y in min_y..=max_y as usize {
        for x in min_x..=max_x as usize {
            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let mut w = [
                edge(screen[1], screen[2], p) / area,
                edge(screen[2], screen[0], p) / area,
                edge(screen[0], screen[1], p) / area,
            ];
            if w.iter().any(|&b| b < 0.0) {
                // Cards are two-sided, so accept the reversed winding as well.
                if w.iter().any(|&b| b > 0.0) {
                    continue;
                }
                w = w.map(|b| -b);
            }
            let inv_w: f32 = (0..3).map(|k| w[k] / screen[k].z).sum();
            if inv_w <= 1e-9 {
                continue;
            }
            let depth = 1.0 / inv_w;
            let di = y * t.size + x;
            if depth >= t.depth[di] {
                continue;
            }
            let pw = |k: usize| w[k] / screen[k].z * depth;
            let (a, b, c) = (pw(0), pw(1), pw(2));
            let frag = Vert {
                clip: Vec4::ZERO,
                world: v[0].world * a + v[1].world * b + v[2].world * c,
                normal: v[0].normal * a + v[1].normal * b + v[2].normal * c,
                uv: v[0].uv * a + v[1].uv * b + v[2].uv * c,
                tint: v[0].tint * a + v[1].tint * b + v[2].tint * c,
                tex_size: v[0].tex_size,
            };
            if let Some(color) = shade(&frag, lod) {
                t.depth[di] = depth;
                t.color[di] = aces(color * frag.tint.w);
            }
        }
    }
}

fn aces(x: Vec3) -> Vec3 {
    (x * (x * 2.51 + 0.03) / (x * (x * 2.43 + 0.59) + 0.14)).clamp(Vec3::ZERO, Vec3::ONE)
}

fn edge(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}
