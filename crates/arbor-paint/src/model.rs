//! A model on the GPU: what [`crate::import::ImportedModel`] becomes once there is a GL
//! context to upload it to.

use std::collections::HashMap;
use std::sync::Arc;

use arbor_render::gpu::create_texture;
use arbor_render::pbr::{AlphaMode, PbrItem, PbrMaterial, PbrMesh, PbrMeshData, PbrTextures};
use glam::{Mat4, Vec3};

use crate::bake::BakedMap;
use crate::import::{ImportedModel, Slot};

pub struct ModelInstance {
    pub name: String,
    mesh: usize,
    material: usize,
    pub transform: Mat4,
    pub visible: bool,
}

/// What the scene panel shows about a material.
pub struct MaterialInfo {
    pub name: String,
    /// The slots that have a texture and its size.
    pub textures: Vec<(Slot, u32, u32)>,
    /// Whether any object is drawn with it.
    pub used: bool,
}

#[derive(Default)]
pub struct ModelStats {
    pub triangles: usize,
    pub vertices: usize,
    /// Megabytes of texture, uncompressed and without mip levels.
    pub texture_mb: f32,
}

/// A baked map standing in for one material's textures, to look at.
struct Preview {
    material: usize,
    texture: glow::Texture,
    stand_in: PbrMaterial,
}

pub struct Model {
    pub name: String,
    /// The model as it was read, for baking from.
    pub source: Arc<ImportedModel>,
    preview: Option<Preview>,
    meshes: Vec<PbrMesh>,
    materials: Vec<PbrMaterial>,
    pub material_info: Vec<MaterialInfo>,
    pub instances: Vec<ModelInstance>,
    pub stats: ModelStats,
    pub warnings: Vec<String>,
}

impl Model {
    pub fn upload(gl: &glow::Context, imported: Arc<ImportedModel>) -> Result<Self, String> {
        let mut meshes: Vec<PbrMesh> = Vec::new();
        for p in &imported.primitives {
            let data = PbrMeshData {
                positions: &p.positions,
                normals: &p.normals,
                uvs: p.uvs.as_deref(),
                tangents: p.tangents.as_deref(),
                colors: p.colors.as_deref(),
                indices: &p.indices,
            };
            match PbrMesh::new(gl, &data) {
                Ok(m) => meshes.push(m),
                Err(e) => {
                    meshes.iter().for_each(|m| m.delete(gl));
                    return Err(format!("{}: {e}", p.name));
                }
            }
        }

        // An image used as both a colour and as data is uploaded once for each.
        let mut textures: HashMap<(usize, bool), glow::Texture> = HashMap::new();
        let mut materials = Vec::new();
        let mut material_info = Vec::new();
        let mut texture_bytes = 0usize;
        for m in &imported.materials {
            let mut tex = PbrTextures::default();
            let mut sizes = Vec::new();
            for &(slot, image) in &m.textures {
                let Some(img) = imported.images[image].as_ref() else { continue };
                let handle = *textures.entry((image, slot.srgb())).or_insert_with(|| {
                    texture_bytes += img.rgba.len();
                    unsafe { create_texture(gl, &img.rgba, img.width, img.height, slot.srgb()) }
                });
                match slot {
                    Slot::BaseColor => tex.base_color = Some(handle),
                    Slot::Normal => tex.normal = Some(handle),
                    Slot::MetallicRoughness => tex.metallic_roughness = Some(handle),
                    Slot::Occlusion => tex.occlusion = Some(handle),
                    Slot::Emissive => tex.emissive = Some(handle),
                }
                sizes.push((slot, img.width, img.height));
            }
            materials.push(PbrMaterial {
                textures: tex,
                base_color: m.base_color,
                metallic: m.metallic,
                roughness: m.roughness,
                emissive: m.emissive,
                normal_scale: m.normal_scale,
                occlusion_strength: m.occlusion_strength,
                alpha_mode: if m.blend {
                    AlphaMode::Blend
                } else if m.mask {
                    AlphaMode::Mask
                } else {
                    AlphaMode::Opaque
                },
                alpha_cutoff: m.alpha_cutoff,
                double_sided: m.double_sided,
            });
            material_info.push(MaterialInfo { name: m.name.clone(), textures: sizes, used: false });
        }

        let instances: Vec<ModelInstance> = imported
            .instances
            .iter()
            .map(|i| ModelInstance {
                name: i.name.clone(),
                mesh: i.primitive,
                material: imported.primitives[i.primitive].material,
                transform: i.transform,
                visible: true,
            })
            .collect();
        // The stand-in for primitives with no material is not worth listing unless one
        // of them ended up with it.
        for i in &instances {
            material_info[i.material].used = true;
        }
        // Shared meshes count once per instance, as they are drawn.
        let (mut triangles, mut vertices) = (0, 0);
        for i in &instances {
            triangles += meshes[i.mesh].triangle_count();
            vertices += imported.primitives[i.mesh].positions.len();
        }
        Ok(Self {
            name: imported.name.clone(),
            warnings: imported.warnings.clone(),
            source: imported,
            preview: None,
            meshes,
            materials,
            material_info,
            instances,
            stats: ModelStats { triangles, vertices, texture_mb: texture_bytes as f32 / 1_048_576.0 },
        })
    }

    /// Hands every GPU object back to the driver. Textures shared between materials are
    /// deleted once.
    pub fn delete(&self, gl: &glow::Context) {
        self.meshes.iter().for_each(|m| m.delete(gl));
        if let Some(p) = &self.preview {
            unsafe { glow::HasContext::delete_texture(gl, p.texture) };
        }
        let mut seen = std::collections::HashSet::new();
        for m in &self.materials {
            let t = &m.textures;
            for tex in [t.base_color, t.normal, t.metallic_roughness, t.occlusion, t.emissive].into_iter().flatten() {
                if seen.insert(tex) {
                    unsafe { glow::HasContext::delete_texture(gl, tex) };
                }
            }
        }
    }

    pub fn items(&self) -> Vec<PbrItem<'_>> {
        self.instances
            .iter()
            .filter(|i| i.visible)
            .map(|i| {
                let material = match &self.preview {
                    Some(p) if p.material == i.material => &p.stand_in,
                    _ => &self.materials[i.material],
                };
                PbrItem { mesh: &self.meshes[i.mesh], material, model: i.transform }
            })
            .collect()
    }

    /// Which objects are showing, in the order of the source model's instances.
    pub fn visibility(&self) -> Vec<bool> {
        self.instances.iter().map(|i| i.visible).collect()
    }

    /// Shows `map` in place of the textures of `material`, exactly as baked.
    pub fn set_preview(&mut self, gl: &glow::Context, material: usize, map: &BakedMap) {
        self.clear_preview(gl);
        // Uploaded as sRGB so the base-colour view, which encodes on the way out, shows
        // the stored values unchanged.
        let texture = unsafe { create_texture(gl, &map.to_rgba8(), map.width, map.height, true) };
        let stand_in = PbrMaterial {
            textures: PbrTextures { base_color: Some(texture), ..Default::default() },
            double_sided: self.materials[material].double_sided,
            ..Default::default()
        };
        self.preview = Some(Preview { material, texture, stand_in });
    }

    pub fn clear_preview(&mut self, gl: &glow::Context) {
        if let Some(p) = self.preview.take() {
            unsafe { glow::HasContext::delete_texture(gl, p.texture) };
        }
    }

    /// The box round everything visible.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        arbor_render::pbr::PbrRenderer::scene_bounds(&self.items())
    }
}
