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
use arbor_core::Mesh;
use eframe::glow;

use arbor_render::gpu::{self, BarkLook, MaterialTextures};

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
            stem_radius: vec![0.0; n],
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
                translucency: None,
            }
        }
    }

    /// Bark's weathering and moss off: the maps carry everything the stone looks like,
    /// its occlusion among it.
    pub fn look() -> BarkLook {
        BarkLook {
            dead_color: glam::Vec3::ZERO,
            baked_occlusion: 1.0,
            ..BarkLook::plain()
        }
    }
}

pub use arbor_core::knobs::rock::*;

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
