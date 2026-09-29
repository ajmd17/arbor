//! Bakes maps of a model's surface into its own UV layout, the way Substance Painter's
//! mesh maps do with no separate high-poly model: for every texel of a texture set, find
//! the point of the surface that texel shows and measure something about it.
//!
//! Rays are cast on the CPU, against a bounding-volume hierarchy over the whole visible
//! model in world space, so a bake sees occluders from every object and not only those of
//! the texture set being baked, and alpha-cutout foliage lets light through its holes.
//! Nothing here touches GL, so a bake runs on worker threads and can be tested headless.

mod bvh;
mod maps;
mod raster;
pub mod scene;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use maps::Context;
use raster::rasterize;
pub use scene::Scene;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapKind {
    /// The surface normal in tangent space, with the material's normal map applied.
    NormalTangent,
    NormalWorld,
    /// World position, scaled to the model's bounding box.
    Position,
    ObjectId,
    AmbientOcclusion,
    Curvature,
    Thickness,
    VertexColor,
}

impl MapKind {
    pub const ALL: [MapKind; 8] = [
        Self::AmbientOcclusion,
        Self::Curvature,
        Self::Thickness,
        Self::NormalWorld,
        Self::NormalTangent,
        Self::Position,
        Self::ObjectId,
        Self::VertexColor,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::NormalTangent => "Normal (tangent space)",
            Self::NormalWorld => "Normal (world space)",
            Self::Position => "Position",
            Self::ObjectId => "ID (object)",
            Self::AmbientOcclusion => "Ambient occlusion",
            Self::Curvature => "Curvature",
            Self::Thickness => "Thickness",
            Self::VertexColor => "Vertex color",
        }
    }

    /// The end of an exported file's name.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::NormalTangent => "normal",
            Self::NormalWorld => "normal_world",
            Self::Position => "position",
            Self::ObjectId => "id",
            Self::AmbientOcclusion => "ao",
            Self::Curvature => "curvature",
            Self::Thickness => "thickness",
            Self::VertexColor => "color",
        }
    }

    /// Whether the map casts rays, and so takes a while.
    pub fn is_heavy(self) -> bool {
        matches!(self, Self::AmbientOcclusion | Self::Thickness)
    }

    /// What lies outside the UV islands, before padding.
    fn background(self) -> [f32; 4] {
        match self {
            Self::NormalTangent => [0.5, 0.5, 1.0, 1.0],
            Self::NormalWorld => [0.5, 0.5, 0.5, 1.0],
            Self::AmbientOcclusion | Self::VertexColor => [1.0; 4],
            Self::Curvature => [0.5, 0.5, 0.5, 1.0],
            Self::Thickness | Self::Position | Self::ObjectId => [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Clone, Debug)]
pub struct BakeParams {
    pub width: u32,
    pub height: u32,
    /// Texels the islands are grown by, so filtering and mip levels do not pull in the
    /// background.
    pub padding: u32,
    /// Baked at this many times the size along each side, then averaged down.
    pub supersample: u32,
    pub maps: Vec<MapKind>,
    /// Rays per texel, for occlusion and thickness.
    pub ao_rays: u32,
    /// How far an occluder counts, as a share of the model's diagonal.
    pub ao_distance: f32,
    /// How deep thickness looks, as a share of the model's diagonal.
    pub thickness_distance: f32,
    /// How far either side of a texel curvature compares normals across, as a share of
    /// the model's diagonal.
    pub curvature_radius: f32,
    pub curvature_strength: f32,
}

impl Default for BakeParams {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            padding: 8,
            supersample: 1,
            maps: vec![
                MapKind::AmbientOcclusion,
                MapKind::Curvature,
                MapKind::Thickness,
                MapKind::NormalWorld,
                MapKind::Position,
            ],
            ao_rays: 64,
            ao_distance: 0.25,
            thickness_distance: 0.25,
            curvature_radius: 0.01,
            curvature_strength: 1.0,
        }
    }
}

/// Shared with the thread doing a bake, to watch it and to stop it.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub cancel: AtomicBool,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.done.load(Ordering::Relaxed) as f32 / total as f32
        }
    }
}

pub struct BakedMap {
    pub kind: MapKind,
    pub width: u32,
    pub height: u32,
    /// Rows from the top, linear values.
    pub pixels: Vec<[f32; 4]>,
}

impl BakedMap {
    /// Eight-bit RGBA, for showing on screen.
    pub fn to_rgba8(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect()
    }

    /// Writes a PNG, sixteen bits a channel if asked.
    pub fn save_png(&self, path: &std::path::Path, sixteen_bit: bool) -> Result<(), String> {
        let err = |e: image::ImageError| format!("{}: {e}", path.display());
        if sixteen_bit {
            let data: Vec<u16> = self.pixels.iter().flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)).collect();
            let img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(self.width, self.height, data)
                .ok_or("bad image size")?;
            img.save(path).map_err(err)
        } else {
            image::save_buffer(path, &self.to_rgba8(), self.width, self.height, image::ColorType::Rgba8).map_err(err)
        }
    }
}

pub struct BakeResult {
    pub material: usize,
    pub maps: Vec<BakedMap>,
    /// Share of the texture the islands cover, before padding.
    pub coverage: f32,
    pub seconds: f32,
    pub warnings: Vec<String>,
}

/// Bakes `params.maps` for the triangles of `material`, an index into the model's
/// materials.
pub fn bake(scene: &Scene, material: usize, params: &BakeParams, progress: &Progress) -> Result<BakeResult, String> {
    let started = Instant::now();
    let ss = params.supersample.clamp(1, 4) as usize;
    let (w, h) = (params.width.max(1) as usize, params.height.max(1) as usize);
    let (bw, bh) = (w * ss, h * ss);
    if bw * bh > 1 << 26 {
        return Err("that size is too large to bake; lower the resolution or the antialiasing".into());
    }
    if params.maps.is_empty() {
        return Err("no maps are selected".into());
    }

    let map = rasterize(scene, material as u32, bw, bh);
    let coverage = map.coverage();
    if coverage == 0.0 {
        return Err("no triangle with UVs uses this material, so there is nothing to bake".into());
    }
    let mut warnings = Vec::new();
    if map.overlapped > 0 {
        warnings.push(format!(
            "{:.0}% of the texels are covered by more than one UV island, so overlapping or tiled UVs will bake wrongly; the later island wins",
            100.0 * map.overlapped as f32 / (bw * bh) as f32
        ));
    }
    if map.outside > 0 {
        warnings.push(format!("{} triangle(s) have UVs outside 0..1, and are clipped; a tiling texture cannot be baked", map.outside));
    }

    let ctx = Context { scene, map: &map, params, material };
    progress.done.store(0, Ordering::Relaxed);
    progress.total.store(params.maps.len() * bh, Ordering::Relaxed);

    // Which final texels have any surface in them, and the padded copy of it.
    let covered: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (0..ss * ss).any(|k| map.covered(x * ss + k % ss, y * ss + k / ss))
        })
        .collect();

    let mut baked = Vec::new();
    for &kind in &params.maps {
        let full = par_rows(bw, bh, progress, &|y, row| {
            for (x, out) in row.iter_mut().enumerate() {
                let i = y * bw + x;
                *out = match map.surface(scene, i) {
                    Some(s) => ctx.shade(kind, &s, i, x, y),
                    None => kind.background(),
                };
            }
        })
        .ok_or("the bake was cancelled")?;
        let mut pixels = downsample(&full, &map, bw, w, h, ss, kind.background());
        dilate(&mut pixels, &covered, w, h, params.padding);
        baked.push(BakedMap { kind, width: w as u32, height: h as u32, pixels });
    }

    Ok(BakeResult {
        material,
        maps: baked,
        coverage,
        seconds: started.elapsed().as_secs_f32(),
        warnings,
    })
}

/// Runs `f` over every row of a `width` by `height` image on as many threads as there
/// are cores, or `None` if the bake was cancelled meanwhile.
fn par_rows(
    width: usize,
    height: usize,
    progress: &Progress,
    f: &(dyn Fn(usize, &mut [[f32; 4]]) + Sync),
) -> Option<Vec<[f32; 4]>> {
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Option<Vec<[f32; 4]>>>> = Mutex::new(vec![None; height]);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let y = next.fetch_add(1, Ordering::Relaxed);
                if y >= height || progress.cancel.load(Ordering::Relaxed) {
                    return;
                }
                let mut row = vec![[0.0; 4]; width];
                f(y, &mut row);
                rows.lock().unwrap()[y] = Some(row);
                progress.done.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    let mut out = Vec::with_capacity(width * height);
    for row in rows {
        out.extend(row?);
    }
    Some(out)
}

/// Averages `ss` by `ss` blocks, over the sub-texels a triangle covers only, so an
/// island's edge is not dragged toward the background.
fn downsample(
    full: &[[f32; 4]],
    map: &raster::TexelMap,
    full_width: usize,
    w: usize,
    h: usize,
    ss: usize,
    background: [f32; 4],
) -> Vec<[f32; 4]> {
    let mut out = vec![background; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut n) = ([0.0f32; 4], 0.0);
            for j in 0..ss {
                for i in 0..ss {
                    let (sx, sy) = (x * ss + i, y * ss + j);
                    if map.covered(sx, sy) {
                        let p = full[sy * full_width + sx];
                        (0..4).for_each(|c| sum[c] += p[c]);
                        n += 1.0;
                    }
                }
            }
            if n > 0.0 {
                out[y * w + x] = sum.map(|c| c / n);
            }
        }
    }
    out
}

/// Grows the covered texels outward `padding` texels, each new one taking the mean of the
/// covered texels beside it.
fn dilate(pixels: &mut [[f32; 4]], covered: &[bool], w: usize, h: usize, padding: u32) {
    let mut have = covered.to_vec();
    for _ in 0..padding {
        let mut grown = have.clone();
        let mut next = pixels.to_vec();
        for y in 0..h {
            for x in 0..w {
                if have[y * w + x] {
                    continue;
                }
                let (mut sum, mut n) = ([0.0f32; 4], 0.0);
                for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let k = ny as usize * w + nx as usize;
                    if have[k] {
                        (0..4).for_each(|c| sum[c] += pixels[k][c]);
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    next[y * w + x] = sum.map(|c| c / n);
                    grown[y * w + x] = true;
                }
            }
        }
        pixels.copy_from_slice(&next);
        have = grown;
    }
}
