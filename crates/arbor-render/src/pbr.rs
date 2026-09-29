//! A generic metallic-roughness surface, for meshes that are not trees: anything loaded
//! from a glTF, and anything a texture painter draws.
//!
//! It is lit by the same environment, sun shadow and occlusion as the tree passes and
//! goes through the same [`Renderer`] frame, so a model here looks the way the same
//! material would on a tree. [`PbrRenderer::draw`] runs the whole frame: shadow map,
//! camera depth and ambient occlusion, the scene, then bloom and tonemapping into the
//! caller's rectangle of the window.

use bytemuck::cast_slice;
use glam::{Mat3, Mat4, Vec3};
use glow::HasContext;

use crate::gpu::{compile_program, create_texture, GpuBackdrop, ShadowTarget};
use crate::lighting::shadow_frustum;
use crate::render::{assign_units, Lighting, PostSettings, Renderer};
use crate::shaders;

/// The vertex inputs, in the order the vertex array binds them.
pub const PBR_ATTRIBS: [&str; 5] = ["a_pos", "a_normal", "a_uv", "a_tangent", "a_color"];

/// Floats per vertex: position, normal, uv, tangent, colour.
const STRIDE_FLOATS: usize = 3 + 3 + 2 + 4 + 4;

const UNIT_BASE: u32 = 0;
const UNIT_NORMAL: u32 = 1;
const UNIT_METALLIC_ROUGHNESS: u32 = 2;
const UNIT_OCCLUSION: u32 = 10;
const UNIT_EMISSIVE: u32 = 11;

/// A mesh as plain slices, one entry per vertex. Only positions, normals and indices are
/// required.
pub struct PbrMeshData<'a> {
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    /// Zero where absent.
    pub uvs: Option<&'a [[f32; 2]]>,
    /// xyz and the sign of the bitangent. Zero where absent, which leaves the normal map
    /// unused.
    pub tangents: Option<&'a [[f32; 4]]>,
    /// White where absent.
    pub colors: Option<&'a [[f32; 4]]>,
    pub indices: &'a [u32],
}

/// A mesh on the GPU.
pub struct PbrMesh {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ibo: glow::Buffer,
    index_count: i32,
    /// Corners of the box round the vertices, in the mesh's own space.
    pub min: Vec3,
    pub max: Vec3,
}

impl PbrMesh {
    pub fn new(gl: &glow::Context, data: &PbrMeshData) -> Result<Self, String> {
        let n = data.positions.len();
        let check = |what: &str, len: Option<usize>| match len {
            Some(l) if l != n => Err(format!("{what}: {l} entries for {n} vertices")),
            _ => Ok(()),
        };
        check("normals", Some(data.normals.len()))?;
        check("uvs", data.uvs.map(|v| v.len()))?;
        check("tangents", data.tangents.map(|v| v.len()))?;
        check("colors", data.colors.map(|v| v.len()))?;
        if let Some(&bad) = data.indices.iter().find(|&&i| i as usize >= n) {
            return Err(format!("index {bad} is past the {n} vertices"));
        }
        if data.indices.len() % 3 != 0 {
            return Err(format!("{} indices is not a whole number of triangles", data.indices.len()));
        }

        let mut vertices = Vec::with_capacity(n * STRIDE_FLOATS);
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for i in 0..n {
            let p = Vec3::from(data.positions[i]);
            min = min.min(p);
            max = max.max(p);
            vertices.extend_from_slice(&data.positions[i]);
            vertices.extend_from_slice(&data.normals[i]);
            vertices.extend_from_slice(&data.uvs.map_or([0.0; 2], |v| v[i]));
            vertices.extend_from_slice(&data.tangents.map_or([0.0; 4], |v| v[i]));
            vertices.extend_from_slice(&data.colors.map_or([1.0; 4], |v| v[i]));
        }
        if n == 0 {
            (min, max) = (Vec3::ZERO, Vec3::ZERO);
        }

        unsafe {
            let vao = gl.create_vertex_array().expect("pbr vao");
            let vbo = gl.create_buffer().expect("pbr vbo");
            let ibo = gl.create_buffer().expect("pbr ibo");
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_slice(&vertices), glow::STATIC_DRAW);
            let stride = (STRIDE_FLOATS * 4) as i32;
            for (index, (width, offset)) in [(3, 0), (3, 3), (2, 6), (4, 8), (4, 12)].into_iter().enumerate() {
                gl.enable_vertex_attrib_array(index as u32);
                gl.vertex_attrib_pointer_f32(index as u32, width, glow::FLOAT, false, stride, offset * 4);
            }
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, cast_slice(data.indices), glow::STATIC_DRAW);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            Ok(Self { vao, vbo, ibo, index_count: data.indices.len() as i32, min, max })
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.index_count as usize / 3
    }

    pub fn delete(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo);
            gl.delete_buffer(self.ibo);
        }
    }

    unsafe fn draw(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_elements(glow::TRIANGLES, self.index_count, glow::UNSIGNED_INT, 0);
            gl.bind_vertex_array(None);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlphaMode {
    #[default]
    Opaque,
    /// Cut out where alpha is under the cutoff.
    Mask,
    Blend,
}

/// Which of a material's textures are present. Each is a plain 2D texture: base colour
/// and emissive uploaded as sRGB, the rest as linear. Missing ones read as white.
#[derive(Clone, Copy, Debug, Default)]
pub struct PbrTextures {
    pub base_color: Option<glow::Texture>,
    pub normal: Option<glow::Texture>,
    /// Roughness in green and metalness in blue, as glTF packs them.
    pub metallic_roughness: Option<glow::Texture>,
    /// Occlusion in red.
    pub occlusion: Option<glow::Texture>,
    pub emissive: Option<glow::Texture>,
}

/// The defaults are glTF's own.
#[derive(Clone, Copy, Debug)]
pub struct PbrMaterial {
    pub textures: PbrTextures,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            textures: PbrTextures::default(),
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

impl PbrMaterial {
    /// Hands the material's textures back to the driver.
    pub fn delete(&self, gl: &glow::Context) {
        let t = &self.textures;
        for tex in [t.base_color, t.normal, t.metallic_roughness, t.occlusion, t.emissive]
            .into_iter()
            .flatten()
        {
            unsafe { gl.delete_texture(tex) };
        }
    }
}

/// What the surface shows. The debug views are written as display colours and skip the
/// tonemapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewMode {
    #[default]
    Lit,
    BaseColor,
    Roughness,
    Metallic,
    Normals,
    /// The material's own occlusion map.
    MaterialOcclusion,
    Emissive,
    UvChecker,
    /// The screen-space occlusion the lit view is darkened by.
    ScreenOcclusion,
}

impl ViewMode {
    pub const ALL: [ViewMode; 9] = [
        Self::Lit,
        Self::BaseColor,
        Self::Roughness,
        Self::Metallic,
        Self::Normals,
        Self::MaterialOcclusion,
        Self::Emissive,
        Self::UvChecker,
        Self::ScreenOcclusion,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Lit => "Lit",
            Self::BaseColor => "Base colour",
            Self::Roughness => "Roughness",
            Self::Metallic => "Metallic",
            Self::Normals => "Normals",
            Self::MaterialOcclusion => "Material AO",
            Self::Emissive => "Emissive",
            Self::UvChecker => "UV checker",
            Self::ScreenOcclusion => "Screen AO",
        }
    }

    fn index(self) -> i32 {
        Self::ALL.iter().position(|&m| m == self).unwrap() as i32
    }
}

/// Which faces of a mesh a draw keeps, where the material has not settled it alone.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sides {
    /// What the material says: both if it is double-sided, else the front ones.
    Material,
    /// Only the faces turned away from the camera.
    Back,
    /// Only the faces turned toward it.
    Front,
}

/// One mesh with the material it is drawn in and where it stands.
pub struct PbrItem<'a> {
    pub mesh: &'a PbrMesh,
    pub material: &'a PbrMaterial,
    pub model: Mat4,
}

impl PbrItem<'_> {
    /// The world-space box round the mesh.
    fn bounds(&self) -> (Vec3, Vec3) {
        let (lo, hi) = (self.mesh.min, self.mesh.max);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let p = self.model.transform_point3(corner);
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }
}

/// Everything one frame needs.
pub struct PbrFrame<'a> {
    pub items: &'a [PbrItem<'a>],
    pub proj: Mat4,
    pub view_proj: Mat4,
    /// The environment and sun. Its shadow and occlusion fields are overwritten.
    pub lighting: Lighting,
    pub post: PostSettings,
    /// Where in the window the picture goes, `[x, y, width, height]` in pixels from the
    /// bottom left.
    pub clip: [i32; 4],
    pub mode: ViewMode,
    pub wireframe: bool,
    /// Draw the sky behind the model, where it is lit.
    pub background: bool,
}

/// The frame for [`PbrItem`]s.
pub struct PbrRenderer {
    pub renderer: Renderer,
    shadow: ShadowTarget,
    lit: glow::Program,
    depth: glow::Program,
    depth_mask: glow::Program,
    wire: glow::Program,
    backdrop: GpuBackdrop,
    white: glow::Texture,
}

impl PbrRenderer {
    pub fn new(gl: &glow::Context, shadow_size: i32) -> Self {
        unsafe {
            let lit = compile_program(gl, shaders::pbr_vs(), &shaders::pbr_fs(), &PBR_ATTRIBS);
            assign_units(gl, lit);
            gl.use_program(Some(lit));
            for (name, unit) in [
                ("u_base_tex", UNIT_BASE),
                ("u_normal_tex", UNIT_NORMAL),
                ("u_mr_tex", UNIT_METALLIC_ROUGHNESS),
                ("u_occlusion_tex", UNIT_OCCLUSION),
                ("u_emissive_tex", UNIT_EMISSIVE),
            ] {
                gl.uniform_1_i32(gl.get_uniform_location(lit, name).as_ref(), unit as i32);
            }
            let depth = compile_program(gl, shaders::PBR_DEPTH_VS, shaders::DEPTH_FS, &PBR_ATTRIBS);
            let depth_mask =
                compile_program(gl, shaders::PBR_DEPTH_MASK_VS, shaders::PBR_DEPTH_MASK_FS, &PBR_ATTRIBS);
            gl.use_program(Some(depth_mask));
            gl.uniform_1_i32(gl.get_uniform_location(depth_mask, "u_base_tex").as_ref(), UNIT_BASE as i32);
            let wire = compile_program(gl, shaders::PBR_DEPTH_VS, shaders::COLOR_FS, &PBR_ATTRIBS);
            gl.use_program(None);
            Self {
                renderer: Renderer::new(gl),
                shadow: ShadowTarget::new(gl, shadow_size),
                lit,
                depth,
                depth_mask,
                wire,
                backdrop: GpuBackdrop::new(gl),
                white: create_texture(gl, &[255; 4], 1, 1, true),
            }
        }
    }

    /// The box round every item, or `None` for an empty scene.
    pub fn scene_bounds(items: &[PbrItem]) -> Option<(Vec3, Vec3)> {
        items.iter().map(PbrItem::bounds).reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))
    }

    /// Draws one frame into `frame.clip`.
    ///
    /// # Safety
    /// Needs the GL context the meshes and textures were made in, current.
    pub unsafe fn draw(&mut self, gl: &glow::Context, frame: &PbrFrame) {
        let [_, _, w, h] = frame.clip;
        self.renderer.prepare(gl, w, h, frame.post.samples);
        let (lo, hi) = Self::scene_bounds(frame.items).unwrap_or((Vec3::splat(-1.0), Vec3::ONE));
        let frustum = shadow_frustum((lo.to_array(), hi.to_array()), frame.lighting.sun_dir, self.shadow.size);
        let lighting = Lighting {
            shadow: frustum,
            shadow_size: self.shadow.size,
            normal_bias: frustum.texel * 1.6,
            ao_texel: self.renderer.ao_texel(),
            ..frame.lighting
        };
        let sun_up = lighting.sun_color.max_element() > 0.0 && lighting.sun_dir.y > -0.1;

        unsafe {
            // The shadow map is cleared even when nothing is drawn into it: the lit
            // shaders read it either way, and stale depth reads as shadow.
            self.shadow.bind(gl);
            gl.viewport(0, 0, self.shadow.size, self.shadow.size);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.depth_mask(true);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);
            if sun_up {
                self.depth_items(gl, frame.items, frustum.view_proj);
            }

            if frame.post.ao {
                self.renderer.begin_prepass(gl);
                self.depth_items(gl, frame.items, frame.view_proj);
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                self.renderer.ambient_occlusion(gl, frame.proj, &frame.post);
            }

            self.renderer.begin_scene(gl);
            self.renderer.bind_lighting(gl, self.shadow.depth, frame.post.ao);
            if frame.mode == ViewMode::Lit && frame.background {
                self.backdrop.draw(gl, frame.view_proj, &lighting);
            } else {
                // Debug views are shown as they are, so they go on a plain grey.
                gl.clear_color(0.32, 0.33, 0.35, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
            }

            // Cut-outs with the opaque ones; what blends goes last, over them.
            let blended = |i: &&PbrItem| i.material.alpha_mode == AlphaMode::Blend;
            for item in frame.items.iter().filter(|i| !blended(i)) {
                self.draw_lit(gl, item, frame, &lighting, Sides::Material);
            }
            // Blending has to go far to near, and without writing depth, so that what is
            // behind still shows through. A double-sided mesh is drawn inside first and
            // then out, or its inner surfaces would land on top of its outer ones.
            let mut far_to_near: Vec<&PbrItem> = frame.items.iter().filter(blended).collect();
            let distance = |i: &PbrItem| {
                let (lo, hi) = i.bounds();
                ((lo + hi) * 0.5 - lighting.cam_pos).length_squared()
            };
            far_to_near.sort_by(|a, b| distance(b).total_cmp(&distance(a)));
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.depth_mask(false);
            for item in far_to_near {
                if item.material.double_sided {
                    self.draw_lit(gl, item, frame, &lighting, Sides::Back);
                    self.draw_lit(gl, item, frame, &lighting, Sides::Front);
                } else {
                    self.draw_lit(gl, item, frame, &lighting, Sides::Material);
                }
            }
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.front_face(glow::CCW);

            #[cfg(not(target_arch = "wasm32"))]
            if frame.wireframe {
                self.draw_wire(gl, frame);
            }

            self.renderer.unbind_lighting(gl);
            self.renderer.finish(gl, frame.clip, &frame.post, frame.mode != ViewMode::Lit);

            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.active_texture(glow::TEXTURE0);
        }
    }

    /// Depth of every item that hides something: opaque ones outright, cut-outs with
    /// their holes, and nothing that blends.
    unsafe fn depth_items(&self, gl: &glow::Context, items: &[PbrItem], view_proj: Mat4) {
        unsafe {
            for item in items {
                let m = item.material;
                let program = match m.alpha_mode {
                    AlphaMode::Blend => continue,
                    AlphaMode::Opaque => self.depth,
                    AlphaMode::Mask => self.depth_mask,
                };
                gl.use_program(Some(program));
                let u = |name: &str| gl.get_uniform_location(program, name);
                gl.uniform_matrix_4_f32_slice(u("u_view_proj").as_ref(), false, &view_proj.to_cols_array());
                gl.uniform_matrix_4_f32_slice(u("u_model").as_ref(), false, &item.model.to_cols_array());
                if m.alpha_mode == AlphaMode::Mask {
                    gl.uniform_1_f32(u("u_base_alpha").as_ref(), m.base_color[3]);
                    gl.uniform_1_f32(u("u_alpha_cutoff").as_ref(), m.alpha_cutoff);
                    gl.active_texture(glow::TEXTURE0 + UNIT_BASE);
                    gl.bind_texture(glow::TEXTURE_2D, Some(m.textures.base_color.unwrap_or(self.white)));
                }
                item.mesh.draw(gl);
            }
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(None);
        }
    }

    unsafe fn draw_lit(&self, gl: &glow::Context, item: &PbrItem, frame: &PbrFrame, lighting: &Lighting, sides: Sides) {
        let m = item.material;
        unsafe {
            gl.use_program(Some(self.lit));
            lighting.bind(gl, self.lit);
            let u = |name: &str| gl.get_uniform_location(self.lit, name);
            gl.uniform_matrix_4_f32_slice(u("u_view_proj").as_ref(), false, &frame.view_proj.to_cols_array());
            gl.uniform_matrix_4_f32_slice(u("u_model").as_ref(), false, &item.model.to_cols_array());
            let normal_mat = Mat3::from_mat4(item.model).inverse().transpose();
            gl.uniform_matrix_3_f32_slice(u("u_normal_mat").as_ref(), false, &normal_mat.to_cols_array());
            let b = m.base_color;
            gl.uniform_4_f32(u("u_base_factor").as_ref(), b[0], b[1], b[2], b[3]);
            gl.uniform_2_f32(u("u_mr_factor").as_ref(), m.metallic, m.roughness);
            gl.uniform_3_f32(u("u_emissive_factor").as_ref(), m.emissive[0], m.emissive[1], m.emissive[2]);
            gl.uniform_1_f32(u("u_normal_scale").as_ref(), m.normal_scale);
            gl.uniform_1_f32(u("u_occlusion_strength").as_ref(), m.occlusion_strength);
            gl.uniform_1_f32(u("u_alpha_cutoff").as_ref(), m.alpha_cutoff);
            gl.uniform_1_i32(u("u_alpha_mode").as_ref(), m.alpha_mode as i32);
            gl.uniform_1_i32(u("u_double_sided").as_ref(), i32::from(m.double_sided));
            gl.uniform_1_i32(u("u_has_normal").as_ref(), i32::from(m.textures.normal.is_some()));
            gl.uniform_1_i32(u("u_mode").as_ref(), frame.mode.index());

            let t = &m.textures;
            for (unit, tex) in [
                (UNIT_BASE, t.base_color),
                (UNIT_NORMAL, t.normal),
                (UNIT_METALLIC_ROUGHNESS, t.metallic_roughness),
                (UNIT_OCCLUSION, t.occlusion),
                (UNIT_EMISSIVE, t.emissive),
            ] {
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex.unwrap_or(self.white)));
            }
            gl.active_texture(glow::TEXTURE0);

            match (sides, m.double_sided) {
                (Sides::Material, true) => gl.disable(glow::CULL_FACE),
                (Sides::Material | Sides::Front, _) => {
                    gl.enable(glow::CULL_FACE);
                    gl.cull_face(glow::BACK);
                }
                (Sides::Back, _) => {
                    gl.enable(glow::CULL_FACE);
                    gl.cull_face(glow::FRONT);
                }
            }
            // A mirroring transform turns every triangle inside out.
            gl.front_face(if item.model.determinant() < 0.0 { glow::CW } else { glow::CCW });
            item.mesh.draw(gl);
            gl.use_program(None);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    unsafe fn draw_wire(&self, gl: &glow::Context, frame: &PbrFrame) {
        unsafe {
            gl.use_program(Some(self.wire));
            let u = |name: &str| gl.get_uniform_location(self.wire, name);
            gl.uniform_matrix_4_f32_slice(u("u_view_proj").as_ref(), false, &frame.view_proj.to_cols_array());
            gl.uniform_4_f32(u("u_color").as_ref(), 0.02, 0.02, 0.03, 1.0);
            gl.polygon_mode(glow::FRONT_AND_BACK, glow::LINE);
            gl.enable(glow::POLYGON_OFFSET_LINE);
            gl.polygon_offset(-1.0, -1.0);
            for item in frame.items {
                gl.uniform_matrix_4_f32_slice(u("u_model").as_ref(), false, &item.model.to_cols_array());
                item.mesh.draw(gl);
            }
            gl.disable(glow::POLYGON_OFFSET_LINE);
            gl.polygon_mode(glow::FRONT_AND_BACK, glow::FILL);
            gl.use_program(None);
        }
    }
}
