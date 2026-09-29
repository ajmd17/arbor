//! Ground cover mode: one clump, planted over a field the way an engine would plant it,
//! so tiling seams, density, root blending and LOD switches show up where they will be
//! seen.
//!
//! The field is built on the CPU into an ordinary leaf mesh and drawn by the leaf
//! pipeline, which already does what grass cards need: alpha-to-coverage with the
//! coverage-preserving mip chain, shadows, occlusion and wind. Each clump is drawn at
//! the LOD its screen size picks, as the engine would, and the field is rebuilt when
//! that choice changes.

use std::path::Path;

use arbor_core::blades::bake_blades;
use arbor_core::cover::{
    build_cover, builtin_cover_presets, parse_cover_template, CoverLod, CoverMesh, CUSTOM_COVER_DIR,
};
use arbor_core::{BladeAtlasParams, CoverParams, CoverTemplate, LeafMesh, Ranged};
use eframe::glow;
use glam::{Mat3, Vec3};

use crate::gpu::{self, LeafMaterialParams, MaterialTextures, LEAF_ALPHA_CUTOFF};
use crate::knobs::{Group, Knob};

/// Metres across the field preview, unless the panel says otherwise.
pub const DEFAULT_FIELD: f32 = 20.0;
/// Seeds of the preset planted among each other in the field.
pub const DEFAULT_VARIANTS: usize = 4;
/// How far a clump may stray from its grid square, as a share of the footprint, how
/// far its size varies either way, and how the field's layout is seeded. Placement is
/// the engine's business; these stand in for its rules.
const PLANT_JITTER: f32 = 0.15;
const PLANT_SCALE: f32 = 0.15;
const PLANT_SEED: u64 = 0x5EED_F1E1_D000_0001;
/// A colour per LOD, for the panel's "Colour LODs".
const LOD_TINTS: [[f32; 3]; 4] = [[1.0, 1.0, 1.0], [1.0, 0.45, 0.45], [0.45, 0.6, 1.0], [1.0, 1.0, 0.35]];

/// A cover preset the panel can load.
pub struct CoverPreset {
    pub name: String,
    /// The file it lives in, for a saved one; built-ins are compiled in.
    pub path: Option<std::path::PathBuf>,
    text: Option<&'static str>,
}

impl CoverPreset {
    pub fn load(&self) -> Result<CoverTemplate, String> {
        match (&self.text, &self.path) {
            (Some(text), _) => parse_cover_template(text),
            (None, Some(path)) => std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|t| parse_cover_template(&t)),
            (None, None) => Err("nothing to load".to_string()),
        }
    }
}

/// The built-in presets, then any saved beside them.
pub fn list_presets() -> Vec<CoverPreset> {
    let mut out: Vec<CoverPreset> = builtin_cover_presets()
        .into_iter()
        .map(|(name, text)| CoverPreset { name: name.to_string(), path: None, text: Some(text) })
        .collect();
    let mut saved: Vec<CoverPreset> = std::fs::read_dir(CUSTOM_COVER_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ron"))
        .filter_map(|p| {
            let name = p.file_stem()?.to_str()?.to_string();
            Some(CoverPreset { name, path: Some(p), text: None })
        })
        .collect();
    saved.sort_by(|a, b| a.name.cmp(&b.name));
    out.extend(saved);
    out
}

/// Writes a preset into the saved folder under `name`.
pub fn save_preset(name: &str, template: &CoverTemplate) -> Result<std::path::PathBuf, String> {
    let key = crate::presets::key_for(name)?;
    if builtin_cover_presets().iter().any(|(n, _)| *n == key) {
        return Err(format!("{key} is a built-in preset: save under another name"));
    }
    let dir = Path::new(CUSTOM_COVER_DIR);
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{key}.ron"));
    let mut out = template.clone();
    out.name = name.trim().to_string();
    let text = ron::ser::to_string_pretty(&out, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// One clump in the field.
#[derive(Clone, Copy, Debug)]
pub struct Plant {
    pub at: Vec3,
    pub yaw: f32,
    pub scale: f32,
    /// Which of the clump's variants is planted here.
    pub variant: usize,
}

/// Clumps over a square `size` metres across, one per footprint on a jittered grid,
/// each turned and sized at random, as an engine plants them.
pub fn plant(size: f32, footprint: f32, variants: usize) -> Vec<Plant> {
    use rand::{Rng, SeedableRng};
    let mut rng = arbor_core::seed::PortableRng::seed_from_u64(PLANT_SEED);
    let step = footprint.max(0.2);
    let n = ((size / step).round() as i32).clamp(1, 200);
    let half = n as f32 * step * 0.5;
    let mut out = Vec::with_capacity((n * n) as usize);
    for gz in 0..n {
        for gx in 0..n {
            let j = PLANT_JITTER * step;
            let x = -half + (gx as f32 + 0.5) * step + rng.random_range(-j..=j);
            let z = -half + (gz as f32 + 0.5) * step + rng.random_range(-j..=j);
            out.push(Plant {
                at: Vec3::new(x, 0.0, z),
                yaw: rng.random_range(0.0..std::f32::consts::TAU),
                scale: 1.0 + rng.random_range(-PLANT_SCALE..=PLANT_SCALE),
                variant: rng.random_range(0..variants.max(1)),
            });
        }
    }
    out
}

/// The share of the screen's height a clump covers from `eye`: the diameter of the
/// sphere round it over the height of the view at that distance.
pub fn screen_size(plant: &Plant, radius: f32, eye: Vec3, fov_y: f32) -> f32 {
    let centre = plant.at + Vec3::Y * radius * 0.3;
    let d = (centre - eye).length().max(1e-3);
    (2.0 * radius * plant.scale) / (2.0 * d * (fov_y * 0.5).tan())
}

/// The LOD a clump of that screen size is drawn at: the last whose switch size it has
/// shrunk below. LOD0's own size is only a placeholder and is not read.
pub fn lod_for(size: f32, lods: &[CoverLod]) -> usize {
    let mut level = 0;
    for (i, lod) in lods.iter().enumerate().skip(1) {
        if size < lod.screen_size {
            level = i;
        }
    }
    level
}

/// Everything ground cover mode holds.
pub struct CoverScene {
    pub presets: Vec<CoverPreset>,
    pub preset: Option<usize>,
    pub save_name: String,
    pub template: CoverTemplate,
    pub params: CoverParams,
    /// The clump at the preset's own seed, which the panel reports on.
    pub mesh: CoverMesh,
    /// The same preset at each seed after it, planted among it as an engine plants
    /// several variants, so the field is not one clump repeated. The first is `mesh`.
    pub variants: Vec<CoverMesh>,
    pub variant_count: usize,
    pub plants: Vec<Plant>,
    pub field: f32,
    /// Draw every clump at this LOD rather than the one its size picks.
    pub force_lod: Option<usize>,
    pub tint_lods: bool,
    /// The LOD each clump was last built at, so the field is only rebuilt when that
    /// changes.
    pub shown: Vec<usize>,
    /// The atlas the material on the GPU was baked from.
    pub baked_atlas: Option<BladeAtlasParams>,
    pub coverage: Vec<f32>,
    pub dirty: bool,
    pub field_dirty: bool,
    pub variations: u32,
    pub gen_ms: f32,
}

impl CoverScene {
    pub fn new(want: Option<&str>) -> (Self, Option<String>) {
        let presets = list_presets();
        let mut problem = None;
        let index = match want {
            Some(w) => presets.iter().position(|p| p.name == w).or_else(|| {
                problem = Some(format!("no cover preset called {w}"));
                None
            }),
            None => None,
        }
        .unwrap_or(0);
        let template = presets[index].load().expect("built-in cover presets parse");
        let params = template.instance();
        let mesh = build_cover(&params);
        let scene = Self {
            save_name: format!("{}_custom", presets[index].name),
            preset: Some(index),
            presets,
            template,
            params,
            mesh,
            variants: Vec::new(),
            variant_count: DEFAULT_VARIANTS,
            plants: Vec::new(),
            field: DEFAULT_FIELD,
            force_lod: None,
            tint_lods: false,
            shown: Vec::new(),
            baked_atlas: None,
            coverage: Vec::new(),
            dirty: true,
            field_dirty: true,
            variations: 1,
            gen_ms: 0.0,
        };
        (scene, problem)
    }

    pub fn pick(&mut self, i: usize) -> Result<(), String> {
        let mut t = self.presets[i].load()?;
        if self.presets[i].path.is_none() {
            // Browsing built-ins keeps the seed, so they compare like with like.
            t.seed = self.template.seed;
            self.save_name = format!("{}_custom", self.presets[i].name);
        } else {
            self.save_name = t.name.clone();
        }
        self.template = t;
        self.preset = Some(i);
        self.dirty = true;
        Ok(())
    }

    /// Lands the preset at its seed and builds the clump again, and the field with it.
    pub fn rebuild(&mut self) {
        let t = web_time::Instant::now();
        self.params = self.template.instance();
        self.mesh = build_cover(&self.params);
        self.variants = (1..self.variant_count.max(1))
            .map(|i| {
                let mut t = self.template.clone();
                t.seed = self.template.seed.wrapping_add(i as u64);
                build_cover(&t.instance())
            })
            .collect();
        self.plants = plant(self.field, self.params.footprint.size, self.variant_count);
        self.shown.clear();
        self.field_dirty = true;
        self.dirty = false;
        self.gen_ms = t.elapsed().as_secs_f32() * 1000.0;
    }

    /// The clump planted as variant `i`.
    pub fn variant(&self, i: usize) -> &CoverMesh {
        if i == 0 { &self.mesh } else { self.variants.get(i - 1).unwrap_or(&self.mesh) }
    }

    /// Radius of the sphere round one clump.
    pub fn radius(&self) -> f32 {
        let half = self.params.footprint.size * (0.5 + self.params.footprint.overhang.max(0.0)) + self.params.tuft.radius;
        (2.0 * half * half + self.mesh.height * self.mesh.height).sqrt()
    }

    /// Picks each clump's LOD from where the camera is, and says whether that differs
    /// from what the field was last built at.
    pub fn choose_lods(&mut self, eye: Vec3, fov_y: f32) -> bool {
        let radius = self.radius();
        let last = self.mesh.lods.len().saturating_sub(1);
        let wanted: Vec<usize> = self
            .plants
            .iter()
            .map(|p| match self.force_lod {
                Some(l) => l.min(last),
                None => lod_for(screen_size(p, radius, eye, fov_y), &self.mesh.lods),
            })
            .collect();
        if wanted != self.shown || self.field_dirty {
            self.shown = wanted;
            self.field_dirty = false;
            return true;
        }
        false
    }

    /// Clumps drawn at each LOD, and the triangles of the whole field.
    pub fn field_stats(&self) -> (Vec<usize>, usize) {
        let mut per = vec![0; self.mesh.lods.len()];
        let mut tris = 0;
        for (plant, &l) in self.plants.iter().zip(&self.shown) {
            per[l] += 1;
            tris += self.variant(plant.variant).lods.get(l).map_or(0, |lod| lod.triangle_count());
        }
        (per, tris)
    }

    /// The field as one leaf mesh, every clump at the LOD `choose_lods` picked.
    pub fn field_mesh(&self) -> LeafMesh {
        let mut out = LeafMesh::default();
        for (plant, &level) in self.plants.iter().zip(&self.shown) {
            let mesh = self.variant(plant.variant);
            let Some(lod) = mesh.lods.get(level) else { continue };
            let turn = Mat3::from_rotation_y(plant.yaw);
            let place = |p: [f32; 3]| (turn * Vec3::from(p) * plant.scale + plant.at).to_array();
            let tint = if self.tint_lods { LOD_TINTS[level.min(3)] } else { [1.0; 3] };
            let base = out.positions.len() as u32;
            for i in 0..lod.positions.len() {
                let (card, t) = lod.sway[i];
                let c = &mesh.cards[card as usize];
                let root = place(c.root);
                out.positions.push(place(lod.positions[i]));
                out.normals.push((turn * Vec3::from(lod.normals[i])).to_array());
                // The atlas as exported, read directly: the shader's cell offset and
                // scale are left at nothing and one.
                out.uvs.push(lod.uvs[i]);
                out.atlas_v.push(0.0);
                out.tints.push([tint[0], tint[1], tint[2], 1.0]);
                out.origins.push(root);
                // Bent about its root by its own weight, as the export tells the engine.
                let weight = c.height * plant.scale * arbor_core::wind::cantilever(t);
                out.sway.push([[root[0], root[1], root[2], weight], [0.0; 4], [0.0; 4]]);
                out.sway_at.push(Default::default());
            }
            out.indices.extend(lod.indices.iter().map(|i| i + base));
        }
        out
    }

    /// The field's bounds, for the shadow to be fitted to.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let half = self.plants.iter().map(|p| p.at.x.abs().max(p.at.z.abs())).fold(0.0, f32::max) + self.radius();
        ([-half, 0.0, -half], [half, self.mesh.height * (1.0 + PLANT_SCALE), half])
    }

    /// Whether the atlas on the GPU is out of date.
    pub fn atlas_stale(&self) -> bool {
        self.baked_atlas.as_ref() != Some(&self.params.atlas)
    }

    /// Bakes the atlas and puts it on the GPU, laid out as the export lays it out.
    pub unsafe fn bake_material(&mut self, gl: &glow::Context) -> MaterialTextures {
        let baked = bake_blades(&self.params.atlas);
        self.coverage = baked.coverage.clone();
        self.baked_atlas = Some(self.params.atlas.clone());
        let albedo = baked.maps.albedo;
        let rough = baked.maps.roughness;
        unsafe {
            MaterialTextures {
                albedo: gpu::create_cutout_texture(gl, &albedo.pixels, albedo.width, albedo.height),
                normal: gpu::create_texture(gl, &[128, 128, 255, 255], 1, 1, false),
                roughness: match rough {
                    Some(r) => gpu::create_texture(gl, &r.pixels, r.width, r.height, false),
                    None => gpu::create_texture(gl, &[170, 170, 170, 255], 1, 1, false),
                },
            }
        }
    }

    pub fn leaf_params(&self, translucency: f32, coverage_lod: f32) -> LeafMaterialParams {
        LeafMaterialParams {
            atlas_scale: [1.0, 1.0],
            atlas_front: [0.0, 0.0],
            atlas_back: [0.0, 0.0],
            alpha_cutoff: LEAF_ALPHA_CUTOFF,
            coverage_lod,
            translucency,
            edge_sharpness: 0.0,
            normal_blend: self.params.normals.ground_normal_blend.clamp(0.0, 1.0),
            // What the export tells the engine: the back of a card keeps its lean to
            // the sky.
            backface_volume: 1.0,
            self_shadow: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------------
// Controls

const fn knob<T>(
    label: &'static str,
    range: (f32, f32),
    help: &'static str,
    get: fn(&mut T) -> &mut Ranged,
) -> Knob<T> {
    Knob { label, range, log: false, help, get }
}

pub static FOOTPRINT: Group<CoverTemplate> = Group {
    title: "Footprint",
    knobs: &[
        knob("Size", (0.5, 4.0), "Metres across the square one clump covers. The engine plants clumps this far apart.", |p| &mut p.footprint.size),
        knob("Tufts", (1.0, 200.0), "Tufts in one clump, spread over its footprint.", |p| &mut p.footprint.tufts),
        knob("Overhang", (0.0, 0.5), "How far tufts run past the edge of the square, thinning out, so clumps planted at a random turn overlap rather than leaving bare corners.", |p| &mut p.footprint.overhang),
        knob("Jitter", (0.0, 1.0), "How far a tuft strays from its own grid square. 0 plants a grid; 1 anywhere in the square.", |p| &mut p.footprint.jitter),
    ],
    counts: &[],
};

pub static TUFT: Group<CoverTemplate> = Group {
    title: "Tuft",
    knobs: &[
        knob("Cards", (1.0, 16.0), "Cards fanned round each tuft.", |p| &mut p.tuft.cards),
        knob("Radius", (0.0, 0.3), "Metres the cards' roots spread over.", |p| &mut p.tuft.radius),
        knob("Height", (0.05, 1.5), "Height of a card, in metres.", |p| &mut p.tuft.height),
        knob("Height variance", (0.0, 0.8), "Spread of height from tuft to tuft, as a share.", |p| &mut p.tuft.height_variance),
        knob("Card height variance", (0.0, 0.8), "Spread of height from card to card within a tuft.", |p| &mut p.tuft.card_height_variance),
        knob("Lean", (0.0, 60.0), "Degrees a card leans out from its tuft at the root.", |p| &mut p.tuft.lean_deg),
        knob("Lean variance", (0.0, 40.0), "Spread of that lean, in degrees.", |p| &mut p.tuft.lean_variance_deg),
        knob("Curl", (-30.0, 90.0), "Degrees a card bends further by its tip: the arch of long grass.", |p| &mut p.tuft.curl_deg),
        knob("Twist", (0.0, 90.0), "Degrees a card turns about its length from root to tip.", |p| &mut p.tuft.twist_deg),
        knob("Understory", (0.0, 1.0), "Share of cards drawn short: the low layer of leaves under the tall ones, which hides the ground.", |p| &mut p.tuft.understory),
        knob("Understory height", (0.05, 1.0), "Height of those short cards against the rest.", |p| &mut p.tuft.understory_height),
        knob("Rosette", (0.0, 1.0), "0 fans a tuft's cards as grass; 1 spreads them evenly round the middle, each facing along its own arch, as a fern's fronds are.", |p| &mut p.tuft.rosette),
    ],
    counts: &[],
};

pub static CARD: Group<CoverTemplate> = Group {
    title: "Card & colour",
    knobs: &[
        knob("Width", (0.3, 2.0), "Card width against the width that keeps atlas pixels square.", |p| &mut p.card.width),
        knob("Width at top", (0.5, 2.0), "Width at the tip against width at the root.", |p| &mut p.card.width_top),
        knob("Fold", (0.0, 1.0), "How far a card's edges stand up from its middle, as a share of its half width: the shallow V along a frond's rachis. 0 is flat.", |p| &mut p.card.fold),
        knob("Dry tufts", (0.0, 1.0), "Share of tufts that draw from the dry cells.", |p| &mut p.colour.dry_fraction),
        knob("Mix", (0.0, 1.0), "Share of a tuft's cards that stray to the other kind of cell.", |p| &mut p.colour.mix),
        knob("Flowering tufts", (0.0, 1.0), "Share of tufts drawing from the cells with wildflowers in them.", |p| &mut p.colour.flower_fraction),
        knob("Flower drift", (0.0, 3.0), "Metres across the drifts flowering tufts gather in. 0 scatters them tuft by tuft.", |p| &mut p.colour.flower_patch),
        knob("Ground normal blend", (0.0, 1.0), "How far shading normals lean from the card to straight up. High reads as one lit surface; low as speckle.", |p| &mut p.normals.ground_normal_blend),
        knob("Root blend", (0.0, 0.5), "Metres above the root the engine blends into the terrain colour. Exported only; the viewer does not draw it.", |p| &mut p.normals.root_blend),
    ],
    counts: &[],
};

pub static WIND: Group<CoverTemplate> = Group {
    title: "Wind response",
    knobs: &[
        knob("Bend", (0.0, 2.0), "How far a card bends along its length in a full gale.", |p| &mut p.wind.flexibility[1]),
        knob("Flutter", (0.0, 1.0), "How far a card flutters about its root, in radians.", |p| &mut p.wind.flutter),
        knob("Frequency", (0.2, 4.0), "How fast the cards sway, in hertz.", |p| &mut p.wind.frequency),
    ],
    counts: &[],
};

macro_rules! lod_group {
    ($name:ident, $title:literal, $i:literal) => {
        pub static $name: Group<CoverTemplate> = Group {
            title: $title,
            knobs: &[
                knob("Cards kept", (0.0, 1.0), "Share of the clump's cards this LOD keeps. Always a subset of the finer LOD's.", |p| &mut p.lod[$i].cards),
                knob("Segments", (1.0, 8.0), "Segments up each card.", |p| &mut p.lod[$i].segments),
                knob("Width", (0.5, 3.0), "Card width against LOD0's, to make up the coverage the dropped cards took.", |p| &mut p.lod[$i].width),
                knob("Screen size", (0.0, 0.3), "Share of the screen's height below which this LOD takes over.", |p| &mut p.lod[$i].screen_size),
            ],
            counts: &[],
        };
    };
}
lod_group!(LOD0, "LOD0", 0);
lod_group!(LOD1, "LOD1", 1);
lod_group!(LOD2, "LOD2", 2);
lod_group!(LOD3, "LOD3", 3);
pub static LODS: [&Group<CoverTemplate>; 4] = [&LOD0, &LOD1, &LOD2, &LOD3];

pub static GROUPS: [&Group<CoverTemplate>; 4] = [&FOOTPRINT, &TUFT, &CARD, &WIND];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_values_fit_their_sliders() {
        for p in list_presets() {
            let mut t = p.load().unwrap();
            let lods = t.lod.len().min(LODS.len());
            let groups = GROUPS.iter().copied().chain(LODS[..lods].iter().copied());
            for g in groups {
                for k in g.knobs {
                    let v = (k.get)(&mut t);
                    for x in [v.lo(), v.hi()] {
                        assert!(
                            (k.range.0..=k.range.1).contains(&x),
                            "{}: {} = {x} outside {:?}",
                            p.name,
                            k.label,
                            k.range
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_field_fills_its_square_one_clump_per_footprint() {
        let plants = plant(20.0, 1.5, 3);
        assert!(plants.iter().all(|p| p.variant < 3));
        assert_eq!(plants.len(), 13 * 13);
        assert!(plants.iter().all(|p| p.at.x.abs() < 11.0 && p.at.z.abs() < 11.0));
    }

    #[test]
    fn far_clumps_take_coarser_lods() {
        let (mut scene, _) = CoverScene::new(None);
        scene.field = 240.0;
        scene.rebuild();
        scene.choose_lods(Vec3::new(0.0, 1.7, 0.0), 50f32.to_radians());
        let (per, _) = scene.field_stats();
        assert!(per.iter().all(|&n| n > 0), "every LOD in use across a big field: {per:?}");
        let mut near = scene.plants.iter().zip(&scene.shown).filter(|(p, _)| p.at.length() < 3.0);
        assert!(near.clone().count() > 0 && near.all(|(_, &l)| l == 0));
    }

    #[test]
    fn the_field_mesh_is_whole() {
        let (mut scene, _) = CoverScene::new(None);
        scene.rebuild();
        scene.choose_lods(Vec3::new(0.0, 1.7, 8.0), 50f32.to_radians());
        let m = scene.field_mesh();
        let n = m.positions.len();
        for len in [m.normals.len(), m.uvs.len(), m.atlas_v.len(), m.tints.len(), m.origins.len(), m.sway.len()] {
            assert_eq!(len, n);
        }
        assert!(m.indices.iter().all(|&i| (i as usize) < n));
        assert!(m.uvs.iter().all(|uv| (-1e-4..=1.0001).contains(&uv[0])), "atlas u runs 0 to 1");
    }
}
