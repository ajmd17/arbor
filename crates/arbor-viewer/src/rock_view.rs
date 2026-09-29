//! Rock mode: one rock, sunk into the ground as it was built to be, drawn through the
//! bark pipeline, which already does what stone needs: its maps, shadows and occlusion.
//!
//! The maps are baked small while the sliders move, then again at the preset's own size
//! (or less on the web) once they have been still a moment, so what is looked at closely
//! is what an export writes.

use std::path::Path;

use arbor_core::rocks::{
    bake_rock_at, build_rock, builtin_rock_presets, parse_rock_template, RockMesh, RockParams, RockTemplate,
    CUSTOM_ROCK_DIR,
};
use arbor_core::{Mesh, Ranged};
use eframe::glow;

use arbor_render::gpu::{self, BarkLook, MaterialTextures};
use crate::knobs::{Group, Knob};

/// Texels a side the maps are baked at while the sliders move, and at most once they
/// stop: less on the web, which has one thread to bake them on.
pub const QUICK_SIZE: u32 = if cfg!(target_arch = "wasm32") { 256 } else { 512 };
pub const PREVIEW_SIZE: u32 = if cfg!(target_arch = "wasm32") { 1024 } else { 2048 };
/// How long the sliders must be still before the full bake.
const SETTLE_SECONDS: f32 = 0.35;

/// A rock preset the panel can load.
pub struct RockPreset {
    pub name: String,
    /// The file it lives in, for a saved one; built-ins are compiled in.
    pub path: Option<std::path::PathBuf>,
    text: Option<&'static str>,
}

impl RockPreset {
    pub fn load(&self) -> Result<RockTemplate, String> {
        match (&self.text, &self.path) {
            (Some(text), _) => parse_rock_template(text),
            (None, Some(path)) => std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|t| parse_rock_template(&t)),
            (None, None) => Err("nothing to load".to_string()),
        }
    }
}

/// The built-in presets, then any saved beside them.
pub fn list_presets() -> Vec<RockPreset> {
    let mut out: Vec<RockPreset> = builtin_rock_presets()
        .into_iter()
        .map(|(name, text)| RockPreset { name: name.to_string(), path: None, text: Some(text) })
        .collect();
    let mut saved: Vec<RockPreset> = std::fs::read_dir(CUSTOM_ROCK_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ron"))
        .filter_map(|p| {
            let name = p.file_stem()?.to_str()?.to_string();
            Some(RockPreset { name, path: Some(p), text: None })
        })
        .collect();
    saved.sort_by(|a, b| a.name.cmp(&b.name));
    out.extend(saved);
    out
}

/// Writes a preset into the saved folder under `name`.
pub fn save_preset(name: &str, template: &RockTemplate) -> Result<std::path::PathBuf, String> {
    let key = crate::presets::key_for(name)?;
    if builtin_rock_presets().iter().any(|(n, _)| *n == key) {
        return Err(format!("{key} is a built-in preset: save under another name"));
    }
    let dir = Path::new(CUSTOM_ROCK_DIR);
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{key}.ron"));
    let mut out = template.clone();
    out.name = name.trim().to_string();
    let text = ron::ser::to_string_pretty(&out, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// Everything rock mode holds.
pub struct RockScene {
    pub presets: Vec<RockPreset>,
    pub preset: Option<usize>,
    pub save_name: String,
    pub template: RockTemplate,
    pub params: RockParams,
    pub mesh: RockMesh,
    /// The LOD drawn.
    pub lod: usize,
    /// The rock the maps on the GPU were baked for, the size they were baked at, and
    /// when the rock last changed.
    pub baked: Option<RockParams>,
    pub baked_size: u32,
    pub changed: web_time::Instant,
    pub dirty: bool,
    /// The mesh on the GPU is out of date.
    pub mesh_dirty: bool,
    pub variations: u32,
    pub gen_ms: f32,
    pub bake_ms: f32,
}

impl RockScene {
    pub fn new(want: Option<&str>) -> (Self, Option<String>) {
        let presets = list_presets();
        let mut problem = None;
        let index = match want {
            Some(w) => presets.iter().position(|p| p.name == w).or_else(|| {
                problem = Some(format!("no rock preset called {w}"));
                None
            }),
            None => None,
        }
        .unwrap_or(0);
        let template = presets[index].load().expect("built-in rock presets parse");
        let params = template.instance();
        let mesh = build_rock(&params);
        let scene = Self {
            save_name: format!("{}_custom", presets[index].name),
            preset: Some(index),
            presets,
            template,
            params,
            mesh,
            lod: 0,
            baked: None,
            baked_size: 0,
            changed: web_time::Instant::now(),
            dirty: true,
            mesh_dirty: true,
            variations: 1,
            gen_ms: 0.0,
            bake_ms: 0.0,
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

    /// Lands the preset at its seed and builds the rock again.
    pub fn rebuild(&mut self) {
        let t = web_time::Instant::now();
        self.params = self.template.instance();
        self.mesh = build_rock(&self.params);
        self.lod = self.lod.min(self.mesh.lods.len().saturating_sub(1));
        self.mesh_dirty = true;
        self.dirty = false;
        self.changed = web_time::Instant::now();
        self.gen_ms = t.elapsed().as_secs_f32() * 1000.0;
    }

    /// The LOD shown, as the bark pipeline takes a mesh: still in any wind.
    pub fn gpu_mesh(&self) -> Mesh {
        let Some(lod) = self.mesh.lods.get(self.lod) else { return Mesh::default() };
        let n = lod.positions.len();
        Mesh {
            positions: lod.positions.clone(),
            normals: lod.normals.clone(),
            tangents: lod.tangents.clone(),
            uvs: lod.uvs.clone(),
            weathering: vec![0.0; n],
            sway: vec![Default::default(); n],
            sway_at: vec![Default::default(); n],
            indices: lod.indices.clone(),
        }
    }

    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        (self.mesh.min, self.mesh.max)
    }

    /// The size the maps want baking at now, if they want baking: small straight after a
    /// change, full size once the rock has been still a moment, or at once when `hurry`.
    pub fn bake_due(&self, hurry: bool) -> Option<u32> {
        let full = PREVIEW_SIZE.min(self.params.texture.size());
        let quick = QUICK_SIZE.min(full);
        let settled = hurry || self.changed.elapsed().as_secs_f32() >= SETTLE_SECONDS;
        if self.baked.as_ref() != Some(&self.params) {
            Some(if settled { full } else { quick })
        } else if self.baked_size < full && settled {
            Some(full)
        } else {
            None
        }
    }

    /// Whether a full bake is still to come, so the viewer keeps drawing until it has.
    pub fn bake_pending(&self) -> bool {
        self.baked.as_ref() != Some(&self.params) || self.baked_size < PREVIEW_SIZE.min(self.params.texture.size())
    }

    /// Bakes the maps `size` texels a side and puts them on the GPU.
    pub unsafe fn bake_material(&mut self, gl: &glow::Context, size: u32) -> MaterialTextures {
        let t = web_time::Instant::now();
        let maps = bake_rock_at(&self.params, &self.mesh, size);
        self.baked = Some(self.params.clone());
        self.baked_size = size;
        self.bake_ms = t.elapsed().as_secs_f32() * 1000.0;
        // The bark shader reads roughness from red, where the export packs occlusion,
        // and takes the occlusion from green, where the export packs roughness.
        let mut rough = maps.orm.pixels.clone();
        for px in rough.chunks_exact_mut(4) {
            px.swap(0, 1);
        }
        unsafe {
            MaterialTextures {
                albedo: gpu::create_texture(gl, &maps.albedo.pixels, maps.albedo.width, maps.albedo.height, true),
                normal: gpu::create_texture(gl, &maps.normal.pixels, maps.normal.width, maps.normal.height, false),
                roughness: gpu::create_texture(gl, &rough, maps.orm.width, maps.orm.height, false),
            }
        }
    }

    /// Bark's weathering and moss off: the maps carry everything the stone looks like,
    /// its occlusion among it.
    pub fn look() -> BarkLook {
        BarkLook {
            moss_color: glam::Vec3::ZERO,
            moss_height: 0.0,
            moss_amount: 0.0,
            darken_low: 0.0,
            tint: glam::Vec3::ONE,
            dead_color: glam::Vec3::ZERO,
            dead_weathering: 0.0,
            baked_occlusion: 1.0,
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

pub static SHAPE: Group<RockTemplate> = Group {
    title: "Shape",
    knobs: &[
        knob("Width", (0.1, 6.0), "Metres across, before noise.", |p| &mut p.shape.width),
        knob("Height", (0.1, 4.0), "Metres high, before noise and the flat bottom.", |p| &mut p.shape.height),
        knob("Depth", (0.1, 6.0), "Metres deep, before noise.", |p| &mut p.shape.depth),
        knob("Facets", (0.0, 32.0), "Planes the rock is cut by: its broad faces.", |p| &mut p.shape.facets),
        knob("Facet depth", (0.0, 0.8), "How deep those planes cut, as a share of how far the rock reaches that way.", |p| &mut p.shape.facet_depth),
        knob("Sharpness", (2.0, 80.0), "How crisply the faces meet: low rounds the edges like a river stone, high leaves them like fresh rubble.", |p| &mut p.shape.sharpness),
        knob("Wear", (0.0, 1.0), "How unevenly the edges are worn: at 1 some stay four times as crisp as others.", |p| &mut p.shape.wear),
        knob("Bulge", (0.0, 0.3), "Swelling and denting of the whole, as a share of the radius.", |p| &mut p.shape.bulge),
        knob("Bulge scale", (0.2, 4.0), "How many swells fit round the rock.", |p| &mut p.shape.bulge_scale),
        knob("Undulation", (0.0, 0.1), "Broad rolls and hollows over the faces, as a share of the radius.", |p| &mut p.shape.undulation),
        knob("Chips", (0.0, 0.12), "How deep chips are struck off the crisper edges, as a share of the radius.", |p| &mut p.shape.chips),
        knob("Chip size", (0.05, 1.0), "How far each chip runs along its edge, as a share of the radius.", |p| &mut p.shape.chip_size),
        knob("Flaking", (0.0, 1.0), "Share of the rock broken into plates flaking off it, stacked a step above one another.", |p| &mut p.shape.flaking),
        knob("Flake step", (0.0, 0.05), "Height of each plate's step, as a share of the radius.", |p| &mut p.shape.flake_step),
        knob("Flake size", (0.05, 2.0), "Size of a plate, as a share of the radius.", |p| &mut p.shape.flake_size),
        knob("Bedded", (0.0, 1.0), "How far the plates lie along the bedding: shells following the surface at 0, layers of slate at 1.", |p| &mut p.shape.bedded),
        knob("Fracture", (0.0, 0.1), "Stepped ledges where the stone has split along its bedding.", |p| &mut p.shape.fracture),
        knob("Flat top", (0.0, 1.0), "Where the top is cut flat along the bedding, as a share of the half height: 0 for none.", |p| &mut p.shape.flat_top),
        knob("Flat bottom", (0.0, 0.9), "Share of the height cut flat where it sits on the ground.", |p| &mut p.shape.flat_bottom),
        knob("Bury", (0.0, 0.9), "Share of the height below the origin, so a rock set on the ground is sunk into it.", |p| &mut p.shape.bury),
    ],
    counts: &[],
};

pub static SURFACE: Group<RockTemplate> = Group {
    title: "Surface",
    knobs: &[
        knob("Grain", (0.0, 0.006), "Height of the fine unevenness of the stone, metres. Maps only.", |p| &mut p.surface.grain),
        knob("Pitting", (0.0, 1.0), "Share of the stone pitted with small holes. Maps only.", |p| &mut p.surface.pitting),
        knob("Cracks", (0.0, 8.0), "Joint cracks through the stone, showing as near straight lines. Maps only.", |p| &mut p.surface.cracks),
        knob("Crack width", (0.0, 0.02), "How wide the cracks open, metres.", |p| &mut p.surface.crack_width),
        knob("Strata", (0.0, 1.5), "Bands of bedding across the stone, in colour and in relief.", |p| &mut p.surface.strata),
        knob("Cavity", (0.0, 2.0), "How strongly hollows and cracks are darkened and occluded.", |p| &mut p.surface.cavity),
        knob("Crunch", (0.0, 0.012), "Height of the knobbly, broken relief a few centimetres across, metres. Maps only.", |p| &mut p.surface.crunch),
        knob("Foliation", (0.0, 1.0), "How far the grain is drawn out into parallel lines, as in gneiss. Maps only.", |p| &mut p.surface.foliation),
    ],
    counts: &[],
};

pub static MOSS: Group<RockTemplate> = Group {
    title: "Moss",
    knobs: &[
        knob("Amount", (-1.0, 1.0), "From none at -1, through tops only at 0, to nearly all of it at 1.", |p| &mut p.moss.amount),
        knob("Upward", (0.0, 2.0), "How far moss keeps to what faces up, against creeping up from the ground.", |p| &mut p.moss.upward),
        knob("Patchiness", (0.05, 3.0), "How broken up the moss is.", |p| &mut p.moss.patchiness),
        knob("Softness", (0.01, 1.0), "How soft the edge of a patch is.", |p| &mut p.moss.softness),
        knob("Crevices", (0.0, 1.0), "How far moss gathers in hollows, cracks and along ledges.", |p| &mut p.moss.crevices),
        knob("Dry", (0.0, 1.0), "Share of the moss dried to a paler olive.", |p| &mut p.moss.dry),
    ],
    counts: &[],
};

pub static GROUPS: [&Group<RockTemplate>; 3] = [&SHAPE, &SURFACE, &MOSS];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rock_values_fit_their_sliders() {
        for p in list_presets() {
            let mut t = p.load().unwrap();
            for g in GROUPS {
                for k in g.knobs {
                    let v = (k.get)(&mut t);
                    for x in [v.lo(), v.hi()] {
                        assert!((k.range.0..=k.range.1).contains(&x), "{}: {} = {x} outside {:?}", p.name, k.label, k.range);
                    }
                }
            }
        }
    }

    #[test]
    fn the_gpu_mesh_is_whole_and_still() {
        let (mut scene, _) = RockScene::new(None);
        scene.rebuild();
        let m = scene.gpu_mesh();
        let n = m.positions.len();
        for len in [m.normals.len(), m.tangents.len(), m.uvs.len(), m.weathering.len(), m.sway.len()] {
            assert_eq!(len, n);
        }
        assert!(m.indices.iter().all(|&i| (i as usize) < n));
        assert!(m.sway.iter().all(|s| s.iter().all(|o| o[3] == 0.0)), "a rock does not sway");
    }
}
