//! Renders a species to a PNG without opening a window, so a tree can be checked
//! from a terminal or a script.
//!
//! cargo run --release -p arbor-viewer --example preview -- \
//!     <pine|oak|path.ron> out.png [--seed N] [--yaw R] [--size N] [--zoom F]
//!     [--aim 0..1] [--pitch D] [--sun-elevation D] [--sun-azimuth D]
//!     [--no-leaves] [--no-ground] [--plain-mips]
//!
//! This is a deliberately small software rasteriser that mirrors what the viewer
//! shaders do — alpha-tested leaf cards sampled from their atlas, wrapped diffuse and
//! backlit transmission — not a second renderer to keep in sync feature by feature.
//!
//! It is not a judge of how full a canopy looks, and the difference is not subtle. This
//! cuts every card at a hard [ALPHA_CUTOFF]; the viewer hands filtered alpha straight to
//! alpha-to-coverage while a card is bigger than a pixel, so cards that overlap blend
//! there and stack here. Foliage therefore reads crisper and denser in this picture than
//! in the build, and the two disagree about the direction of a change: shrinking cards
//! while raising density to hold coverage looks like finer grain here and like an even
//! featureless felt in the viewer. Use `arbor-viewer --screenshot` for anything about
//! card size, leaf density or canopy mass, and this for geometry, silhouette and mips.
//!
//! It samples a real mip chain, chosen per triangle, because the way a leaf texture
//! is minified is exactly what decides whether a canopy survives being zoomed away
//! from. `--plain-mips` swaps the coverage-preserving chain for a plain box-filtered
//! one, which is what the disappearing-canopy bug looked like.

// Shared with the viewer, which uses parts of it this does not.
#[path = "../src/lighting.rs"]
#[allow(dead_code)]
mod lighting;
#[path = "../src/mipmap.rs"]
mod mipmap;

use arbor_core::cluster::{bake_cluster, Bitmap, LeafMaps};
use arbor_core::species::{builtin_presets, parse_template, LeafParams};
use arbor_core::{build_leaves, build_mesh, grow, LeafMesh, Mesh, SpeciesParams};
use glam::{Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use image::{Rgb, RgbImage};

const TEXTURE_DIR: &str = "assets/textures";
const ALPHA_CUTOFF: f32 = 0.35;
const SHADOW_SIZE: i32 = 2048;
/// The shaders treat `sun_color` as irradiance, so a Lambert surface returns this
/// fraction of it. Mirrored here or the preview comes out several times too bright.
const INV_PI: f32 = std::f32::consts::FRAC_1_PI;

struct Target {
    color: Vec<Vec3>,
    depth: Vec<f32>,
    size: usize,
}

/// Scene depth from the light, sampled the same way the viewer shader samples it.
struct ShadowMap {
    depth: Vec<f32>,
    size: usize,
    view_proj: Mat4,
    bias: f32,
}

impl ShadowMap {
    fn render(mesh: &Mesh, leaves: &LeafMesh, tex: Option<&Tex>, params: &SpeciesParams,
              view_proj: Mat4, bias: f32) -> ShadowMap {
        let size = SHADOW_SIZE as usize;
        let mut map = ShadowMap { depth: vec![f32::INFINITY; size * size], size, view_proj, bias };
        let mut raster_into = |positions: &[[f32; 3]], indices: &[u32], alpha: Option<(&Tex, Vec2, Vec2)>,
                               uvs: Option<&[[f32; 2]]>| {
            for tri in indices.chunks_exact(3) {
                let p: Vec<Vec3> = tri.iter().map(|&i| Vec3::from(positions[i as usize])).collect();
                let mut ndc = [Vec3::ZERO; 3];
                for k in 0..3 {
                    let c = view_proj * p[k].extend(1.0);
                    if c.w.abs() < 1e-6 { return; }
                    let n = c.xyz() / c.w;
                    ndc[k] = Vec3::new((n.x * 0.5 + 0.5) * size as f32,
                                       (n.y * 0.5 + 0.5) * size as f32,
                                       n.z * 0.5 + 0.5);
                }
                let area = edge(ndc[0], ndc[1], ndc[2]);
                if area.abs() < 1e-9 { continue; }
                let min_x = ndc.iter().map(|q| q.x).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
                let max_x = ndc.iter().map(|q| q.x).fold(f32::MIN, f32::max).ceil().min(size as f32 - 1.0);
                let min_y = ndc.iter().map(|q| q.y).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
                let max_y = ndc.iter().map(|q| q.y).fold(f32::MIN, f32::max).ceil().min(size as f32 - 1.0);
                if max_x < 0.0 || max_y < 0.0 { continue; }
                for y in min_y..=max_y as usize {
                    for x in min_x..=max_x as usize {
                        let q = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                        let mut w = [edge(ndc[1], ndc[2], q) / area,
                                     edge(ndc[2], ndc[0], q) / area,
                                     edge(ndc[0], ndc[1], q) / area];
                        if w.iter().any(|&b| b < 0.0) {
                            if w.iter().any(|&b| b > 0.0) { continue; }
                            w = w.map(|b| -b);
                        }
                        // Leaf cards only block light where the cutout is opaque.
                        if let (Some((t, origin, scale)), Some(uvs)) = (alpha, uvs) {
                            let uv = Vec2::from(uvs[tri[0] as usize]) * w[0]
                                + Vec2::from(uvs[tri[1] as usize]) * w[1]
                                + Vec2::from(uvs[tri[2] as usize]) * w[2];
                            // A card casts the shadow of the arrangement it draws, not
                            // of whichever one happens to sit first in the sheet.
                            let at = origin
                                + Vec2::new(0.0, leaves.atlas_v[tri[0] as usize])
                                + uv * scale;
                            if t.sample(at, 0.0).w < ALPHA_CUTOFF { continue; }
                        }
                        let z = w[0] * ndc[0].z + w[1] * ndc[1].z + w[2] * ndc[2].z;
                        let i = y * size + x;
                        if z < map.depth[i] { map.depth[i] = z; }
                    }
                }
            }
        };
        raster_into(&mesh.positions, &mesh.indices, None, None);
        if !leaves.is_empty()
            && let Some(t) = tex
        {
            {
                let (scale, front, _) = Tex::atlas_frame(&params.leaves);
                raster_into(&leaves.positions, &leaves.indices, Some((t, front, scale)),
                            Some(&leaves.uvs));
            }
        }
        map
    }

    /// Percentage-closer filter, matching the kernel the viewer uses.
    fn visibility(&self, world: Vec3, normal: Vec3, ndl: f32) -> f32 {
        let c = self.view_proj * (world + normal * self.bias).extend(1.0);
        if c.w.abs() < 1e-6 { return 1.0; }
        let n = c.xyz() / c.w;
        let proj = Vec3::new(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5);
        if !(0.0..=1.0).contains(&proj.x) || !(0.0..=1.0).contains(&proj.y) || proj.z > 1.0 {
            return 1.0;
        }
        let bias = (0.0025 * (1.0 - ndl)).max(0.0008);
        let mut sum = 0.0;
        let mut n_taps = 0.0;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let x = (proj.x * self.size as f32) as i32 + dx;
                let y = (proj.y * self.size as f32) as i32 + dy;
                n_taps += 1.0;
                if x < 0 || y < 0 || x >= self.size as i32 || y >= self.size as i32 {
                    sum += 1.0;
                    continue;
                }
                let d = self.depth[y as usize * self.size + x as usize];
                sum += if proj.z - bias > d { 0.0 } else { 1.0 };
            }
        }
        sum / n_taps
    }
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
    let mut species = parse_template(&src).expect("parse species");
    if let Some(seed) = flag("--seed").and_then(|s| s.parse().ok()) {
        species.seed = seed;
    }
    let mut params: SpeciesParams = species.instance();
    params.leaves.enabled &= want_leaves;

    let sk = grow(&params);
    let mesh = build_mesh(&sk, &params);
    let leaves = build_leaves(&sk, &params);
    let bark = Tex::load(
        &format!("{TEXTURE_DIR}/{}_albedo.png", params.mesh.bark_texture),
        false,
    );
    let leaf_tex = Tex::load_leaf(
        &format!("{TEXTURE_DIR}/{}_albedo.png", params.leaves.texture),
        &params.leaves,
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
    let pitch: f32 = flag("--pitch")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(5.0)
        .to_radians();
    let eye = center
        + Vec3::new(
            yaw.sin() * pitch.cos() * dist,
            pitch.sin() * dist,
            yaw.cos() * pitch.cos() * dist,
        );
    let vp = Mat4::perspective_rh(50f32.to_radians(), 1.0, 0.05, dist * 6.0)
        * Mat4::look_at_rh(eye, center, Vec3::Y);
    let elev: f32 = flag("--sun-elevation").and_then(|s| s.parse().ok()).unwrap_or(13.0);
    let azim: f32 = flag("--sun-azimuth").and_then(|s| s.parse().ok()).unwrap_or(18.0);
    let (e, a) = (elev.to_radians(), azim.to_radians());
    let sun = Vec3::new(a.cos() * e.cos(), e.sin(), a.sin() * e.cos()).normalize();

    // The sky is built from the sun, so moving the sun moves the whole dome with it.
    let sky = lighting::SkyParams::for_sun(sun);
    let irr = lighting::SkyIrradiance::new(&sky);

    // The same fit the viewer uses, so the shadow seen here is the shadow it draws.
    let mut lo_all = lo;
    let hi_all = hi;
    lo_all.y = lo_all.y.min(0.0);
    let (light_vp, texel) = lighting::light_view_proj(
        (lo_all.to_array(), hi_all.to_array()),
        sun,
        SHADOW_SIZE,
    );
    let shadows = ShadowMap::render(
        &mesh,
        &leaves,
        leaf_tex.as_ref(),
        &params,
        light_vp,
        texel * 1.6,
    );

    let mut target = Target {
        color: vec![Vec3::ZERO; size * size],
        depth: vec![f32::INFINITY; size * size],
        size,
    };
    draw_sky(&mut target, vp, eye, &sky);
    if !args.iter().any(|a| a == "--no-ground") {
        draw_ground(&mut target, vp, eye, &sky, &irr, &shadows, extent * 14.0);
    }
    draw_bark(&mut target, &mesh, vp, &sky, &irr, &shadows, bark.as_ref());
    if params.leaves.enabled {
        draw_leaves(
            &mut target,
            &leaves,
            &params,
            vp,
            eye,
            &sky,
            &irr,
            &shadows,
            leaf_tex.as_ref(),
        );
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
        Tex::build(path, img.width(), img.height(), img.into_raw(), preserve_coverage)
    }

    /// The leaf sheet, grown into clusters first if the species asks for them.
    ///
    /// A species with a `cluster` block does not ship the sheet the renderer samples;
    /// it ships one leaf and a recipe, and the atlas is baked on load. Skipping that
    /// step here drew the source art on every card, so a whole cluster block — its
    /// leaf count, its arrangement, its variants — made no difference to anything this
    /// example put on screen, and a species could be tuned against a picture that the
    /// viewer would never draw.
    fn load_leaf(path: &str, lp: &LeafParams, preserve_coverage: bool) -> Option<Tex> {
        let img = image::open(path).ok()?.to_rgba8();
        let (w, h) = (img.width(), img.height());
        let src = Bitmap::from_rgba(w, h, img.into_raw())?;
        let Some(cluster) = &lp.cluster else {
            return Tex::build(path, w, h, src.pixels, preserve_coverage);
        };
        let baked = bake_cluster(
            cluster,
            lp.atlas_cols,
            lp.atlas_rows,
            LeafMaps { albedo: &src, normal: None, roughness: None },
        );
        println!(
            "  clustered {path} into {}x{} from {} leaves",
            baked.albedo.width, baked.albedo.height, cluster.count
        );
        Tex::build(
            path,
            baked.albedo.width,
            baked.albedo.height,
            baked.albedo.pixels,
            preserve_coverage,
        )
    }

    fn build(path: &str, w: u32, h: u32, px: Vec<u8>, preserve_coverage: bool) -> Option<Tex> {
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

    /// Where a card's own arrangement sits in the sheet, and how much of the sheet one
    /// card covers. Clustering stacks its variants down the atlas, so the sheet is
    /// `variants` times taller than the grid the species describes and a card reaches
    /// its own arrangement by the v offset `build_leaves` gave it.
    fn atlas_frame(lp: &LeafParams) -> (Vec2, Vec2, Vec2) {
        let cols = lp.atlas_cols.max(1);
        let rows = lp.atlas_rows.max(1);
        let variants = lp.cluster.as_ref().map_or(1, |c| c.variants.max(1));
        let tall = rows * variants;
        let cell = |index: u32| {
            let i = index.min(cols * rows - 1);
            Vec2::new(
                i.rem_euclid(cols) as f32 / cols as f32,
                (i / cols) as f32 / tall as f32,
            )
        };
        (
            Vec2::new(1.0 / cols as f32, 1.0 / tall as f32),
            cell(lp.atlas_front),
            cell(lp.atlas_back),
        )
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

/// Fills every pixel with the dome, before anything is drawn over it.
fn draw_sky(t: &mut Target, vp: Mat4, eye: Vec3, sky: &lighting::SkyParams) {
    let inv = vp.inverse();
    let n = t.size as f32;
    for y in 0..t.size {
        for x in 0..t.size {
            let ndc = Vec2::new(
                (x as f32 + 0.5) / n * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / n * 2.0,
            );
            let far = inv * Vec3::new(ndc.x, ndc.y, 1.0).extend(1.0);
            let dir = (far.xyz() / far.w - eye).normalize_or(Vec3::Y);
            t.color[y * t.size + x] = aces(sky.background(dir));
        }
    }
}

/// A ground plane catching the shadow, fading into the sky at range.
fn draw_ground(
    t: &mut Target,
    vp: Mat4,
    eye: Vec3,
    sky: &lighting::SkyParams,
    irr: &lighting::SkyIrradiance,
    shadows: &ShadowMap,
    extent: f32,
) {
    let albedo_base = Vec3::new(0.062, 0.058, 0.044);
    let (fade_start, fade_end) = (extent * 0.10, extent * 0.62);
    let corner = |sx: f32, sz: f32| Vec3::new(eye.x + sx * extent, 0.0, eye.z + sz * extent);
    let quad = [
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ];
    for tri in [[0usize, 1, 2], [0, 2, 3]] {
        let v: Vec<Vert> = tri
            .iter()
            .map(|&i| project(vp, quad[i], Vec3::Y, Vec2::ZERO, Vec4::ONE, Vec2::ONE))
            .collect();
        raster(t, &v, |f, _| {
            let n = Vec3::Y;
            let grain = hash_noise(f.world.xz() * 0.7) * 0.35 + hash_noise(f.world.xz() * 0.11) * 0.65;
            let albedo = albedo_base * (0.84 + 0.32 * grain);
            let ndl = n.dot(sky.sun_dir).max(0.0);
            let vis = shadows.visibility(f.world, n, ndl);
            let mut color = albedo * (ndl * vis * sky.sun_color * INV_PI + irr.eval(n));
            let view = (f.world - eye).normalize_or(Vec3::Y);
            let fade = smoothstep(fade_start, fade_end, (f.world.xz() - eye.xz()).length());
            color = color.lerp(sky.background(view), fade);
            Some(color)
        });
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash_noise(p: Vec2) -> f32 {
    let h = |v: Vec2| (v.dot(Vec2::new(127.1, 311.7)).sin() * 43758.545).fract().abs();
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let a = h(i);
    let b = h(i + Vec2::X);
    let c = h(i + Vec2::Y);
    let d = h(i + Vec2::ONE);
    (a * (1.0 - u.x) + b * u.x) * (1.0 - u.y) + (c * (1.0 - u.x) + d * u.x) * u.y
}

fn draw_bark(
    t: &mut Target,
    mesh: &Mesh,
    vp: Mat4,
    sky: &lighting::SkyParams,
    irr: &lighting::SkyIrradiance,
    shadows: &ShadowMap,
    tex: Option<&Tex>,
) {
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
            let n = f.normal.normalize_or_zero();
            let ndl = n.dot(sky.sun_dir).max(0.0);
            let vis = shadows.visibility(f.world, n, ndl);
            Some(albedo * (ndl * vis * sky.sun_color * INV_PI + irr.eval(n)))
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
    sky: &lighting::SkyParams,
    irr: &lighting::SkyIrradiance,
    shadows: &ShadowMap,
    tex: Option<&Tex>,
) {
    let dims = tex_dims(tex);
    let lp = &params.leaves;
    let (scale, front, back) = Tex::atlas_frame(lp);

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
                    Vec2::from(leaves.uvs[i]) * scale
                        + Vec2::new(0.0, leaves.atlas_v[i]),
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
            let ndl = n.dot(sky.sun_dir).max(0.0);
            let vis = shadows.visibility(f.world, n, ndl);
            let wrapped = ((n.dot(sky.sun_dir) + 0.5) / 1.5).max(0.0);
            let through = (-n).dot(sky.sun_dir).max(0.0);
            let lobe = 0.35 + 0.65 * view.dot(-sky.sun_dir).max(0.0).powi(3);
            let transmitted = albedo * 0.9 * through * lobe;
            Some(
                (albedo * wrapped * vis * INV_PI + transmitted * vis) * sky.sun_color
                    + albedo * irr.eval(n),
            )
        });
    }
}

fn lerp_vert(a: &Vert, b: &Vert, t: f32) -> Vert {
    Vert {
        clip: a.clip.lerp(b.clip, t),
        world: a.world.lerp(b.world, t),
        normal: a.normal.lerp(b.normal, t),
        uv: a.uv.lerp(b.uv, t),
        tint: a.tint.lerp(b.tint, t),
        tex_size: a.tex_size,
    }
}

/// Clips against the near plane, then fills. Geometry that reaches past the camera,
/// which the ground plane always does, has to be cut there: a vertex behind the eye
/// projects to nonsense, and dropping the whole triangle instead loses the ground.
fn raster(t: &mut Target, v: &[Vert], shade: impl Fn(&Vert, f32) -> Option<Vec3>) {
    const NEAR_W: f32 = 1e-3;
    if v.iter().all(|x| x.clip.w > NEAR_W) {
        raster_clipped(t, v, &shade);
        return;
    }
    let mut poly: Vec<Vert> = Vec::with_capacity(4);
    for i in 0..3 {
        let a = &v[i];
        let b = &v[(i + 1) % 3];
        let (a_in, b_in) = (a.clip.w > NEAR_W, b.clip.w > NEAR_W);
        if a_in {
            poly.push(*a);
        }
        if a_in != b_in {
            let denom = b.clip.w - a.clip.w;
            if denom.abs() > 1e-9 {
                poly.push(lerp_vert(a, b, (NEAR_W - a.clip.w) / denom));
            }
        }
    }
    if poly.len() < 3 {
        return;
    }
    for i in 1..poly.len() - 1 {
        raster_clipped(t, &[poly[0], poly[i], poly[i + 1]], &shade);
    }
}

/// Scanline fill with a depth test and perspective-correct attributes. `shade`
/// returns None for a fragment the alpha test rejects.
fn raster_clipped(t: &mut Target, v: &[Vert], shade: &impl Fn(&Vert, f32) -> Option<Vec3>) {
    let size = t.size as f32;
    let mut screen = [Vec3::ZERO; 3];
    for (k, vert) in v.iter().enumerate() {
        let ndc = vert.clip.xyz() / vert.clip.w.max(1e-6);
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
