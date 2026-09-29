//! Rock mode: one rock, sunk into the ground as it was built to be, drawn through the
//! bark pipeline, which already does what stone needs: its maps, shadows and occlusion.
//!
//! The maps are baked at a preview resolution while the sliders move, since a full bake
//! takes a second; an export bakes them at the preset's own.

use std::path::Path;

use arbor_core::rocks::{
    bake_rock_at, build_rock, builtin_rock_presets, parse_rock_template, RockMesh, RockParams, RockTemplate,
    CUSTOM_ROCK_DIR,
};
use arbor_core::{Mesh, Ranged};
use eframe::glow;

use arbor_render::gpu::{self, BarkLook, MaterialTextures};
use crate::knobs::{Group, Knob};

/// Texels per face edge the preview's maps are baked at.
pub const PREVIEW_TEXELS: u32 = 256;

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
    /// The rock the maps on the GPU were baked for.
    pub baked: Option<RockParams>,
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

    /// Whether the maps on the GPU are out of date.
    pub fn maps_stale(&self) -> bool {
        self.baked.as_ref() != Some(&self.params)
    }

    /// Bakes the maps at the preview resolution and puts them on the GPU.
    pub unsafe fn bake_material(&mut self, gl: &glow::Context) -> MaterialTextures {
        let t = web_time::Instant::now();
        let maps = bake_rock_at(&self.params, PREVIEW_TEXELS.min(self.params.texture.resolution));
        self.baked = Some(self.params.clone());
        self.bake_ms = t.elapsed().as_secs_f32() * 1000.0;
        // The bark shader reads roughness from red, where the export packs occlusion.
        let mut rough = maps.orm.pixels.clone();
        for px in rough.chunks_exact_mut(4) {
            px[0] = px[1];
        }
        unsafe {
            MaterialTextures {
                albedo: gpu::create_texture(gl, &maps.albedo.pixels, maps.albedo.width, maps.albedo.height, true),
                normal: gpu::create_texture(gl, &maps.normal.pixels, maps.normal.width, maps.normal.height, false),
                roughness: gpu::create_texture(gl, &rough, maps.orm.width, maps.orm.height, false),
            }
        }
    }

    /// Bark's weathering and moss off: the maps carry everything the stone looks like.
    pub fn look() -> BarkLook {
        BarkLook {
            moss_color: glam::Vec3::ZERO,
            moss_height: 0.0,
            moss_amount: 0.0,
            darken_low: 0.0,
            tint: glam::Vec3::ONE,
            dead_color: glam::Vec3::ZERO,
            dead_weathering: 0.0,
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
        knob("Height", (0.1, 4.0), "Metres high, before noise.", |p| &mut p.shape.height),
        knob("Depth", (0.1, 6.0), "Metres deep, before noise.", |p| &mut p.shape.depth),
        knob("Facets", (0.0, 32.0), "Planes the rock is cut by: its broad faces.", |p| &mut p.shape.facets),
        knob("Facet depth", (0.0, 0.8), "How deep those planes cut, as a share of the radius.", |p| &mut p.shape.facet_depth),
        knob("Sharpness", (2.0, 80.0), "How crisply the faces meet: low rounds the edges like a river stone, high leaves them like fresh rubble.", |p| &mut p.shape.sharpness),
        knob("Bulge", (0.0, 0.3), "Swelling and denting of the whole, as a share of the radius.", |p| &mut p.shape.bulge),
        knob("Bulge scale", (0.2, 4.0), "How many swells fit round the rock.", |p| &mut p.shape.bulge_scale),
        knob("Ridges", (0.0, 0.15), "Ridges and grooves over the faces.", |p| &mut p.shape.ridges),
        knob("Fracture", (0.0, 0.1), "Stepped ledges where the stone has split along its bedding.", |p| &mut p.shape.fracture),
        knob("Flat bottom", (0.0, 0.9), "Share of the height cut flat where it sits on the ground.", |p| &mut p.shape.flat_bottom),
        knob("Bury", (0.0, 0.9), "Share of the height below the origin, so a rock set on the ground is sunk into it.", |p| &mut p.shape.bury),
    ],
    counts: &[],
};

pub static SURFACE: Group<RockTemplate> = Group {
    title: "Surface",
    knobs: &[
        knob("Grain", (0.0, 0.05), "Fine unevenness of the stone. Maps only.", |p| &mut p.surface.grain),
        knob("Cracks", (0.0, 0.05), "Depth of the cracks. Maps only.", |p| &mut p.surface.cracks),
        knob("Crack spread", (0.0, 1.0), "Share of the rock the cracks run over.", |p| &mut p.surface.crack_spread),
        knob("Strata", (0.0, 0.03), "Fine banding across the bedding. Maps only.", |p| &mut p.surface.strata),
        knob("Cavity", (0.0, 2.0), "How strongly hollows and cracks are darkened and occluded.", |p| &mut p.surface.cavity),
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
        knob("Crevices", (0.0, 1.0), "How far moss gathers in hollows and cracks.", |p| &mut p.moss.crevices),
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
