//! The environment on the GPU, in the form image-based lighting reads it (Filament's,
//! in outline): a cube map prefiltered for GGX, a mip per step of roughness, and the
//! DFG table the split sum is completed with. Irradiance does not come from here: it
//! is nine spherical-harmonic coefficients worked out on the CPU, in `lighting.rs` for
//! the procedural sky and `hdri.rs` for a photograph.

use eframe::glow::{self, HasContext};
use glam::Vec3;

use crate::gpu::compile_program;
use crate::hdri::Equirect;
use crate::lighting::SkyParams;
use crate::shaders;

/// Face size of the cube an environment is first captured into. Its full mip chain is
/// what the prefilter reads from.
const SOURCE_SIZE: i32 = 512;
/// Face size of the prefiltered cube's sharpest level, and its number of levels:
/// roughness 0 at 256 across down to roughness 1 at 16, as Filament lays it out.
const SPECULAR_SIZE: i32 = 256;
const SPECULAR_LEVELS: i32 = 5;
/// Samples per texel of the prefilter. Filtered importance sampling makes this plenty.
const PREFILTER_SAMPLES: i32 = 192;
const DFG_SIZE: i32 = 128;

/// The highest mip a lit shader asks the prefiltered cube for.
pub const MAX_LOD: f32 = (SPECULAR_LEVELS - 1) as f32;

/// Filament's mip for a perceptual roughness is `max_lod * r * (2 - r)`; this is the
/// roughness a level is prefiltered at so that lookups land on it.
fn level_roughness(level: i32) -> f32 {
    let t = level as f32 / MAX_LOD;
    1.0 - (1.0 - t).max(0.0).sqrt()
}

const GL_TEXTURE_CUBE_MAP_SEAMLESS: u32 = 0x884F;

pub struct EnvironmentMaps {
    /// The environment prefiltered for GGX.
    pub specular: glow::Texture,
    /// The photograph as it was taken, sun and all, for the background and the ground.
    /// A single black texel under the procedural sky, which draws its own.
    pub equirect: glow::Texture,
    source: glow::Texture,
    fbo: glow::Framebuffer,
    vao: glow::VertexArray,
    capture: glow::Program,
    prefilter: glow::Program,
}

unsafe fn cube_texture(gl: &glow::Context, size: i32, levels: i32) -> glow::Texture {
    unsafe {
        let tex = gl.create_texture().expect("cube texture");
        gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(tex));
        for level in 0..levels {
            let s = (size >> level).max(1);
            for face in 0..6 {
                gl.tex_image_2d(
                    glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    level,
                    glow::RGBA16F as i32,
                    s,
                    s,
                    0,
                    glow::RGBA,
                    glow::FLOAT,
                    glow::PixelUnpackData::Slice(None),
                );
            }
        }
        gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_BASE_LEVEL, 0);
        gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MAX_LEVEL, levels - 1);
        gl.tex_parameter_i32(
            glow::TEXTURE_CUBE_MAP,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        for wrap in [glow::TEXTURE_WRAP_S, glow::TEXTURE_WRAP_T, glow::TEXTURE_WRAP_R] {
            gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, wrap, glow::CLAMP_TO_EDGE as i32);
        }
        gl.bind_texture(glow::TEXTURE_CUBE_MAP, None);
        tex
    }
}

/// A float image as a texture the shaders can filter, with its mips.
unsafe fn upload_equirect(gl: &glow::Context, tex: glow::Texture, img: &Equirect) {
    unsafe {
        let rgba: Vec<f32> = img
            .pixels
            .iter()
            .flat_map(|p| [p.x.min(65000.0), p.y.min(65000.0), p.z.min(65000.0), 1.0])
            .collect();
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA16F as i32,
            img.width as i32,
            img.height as i32,
            0,
            glow::RGBA,
            glow::FLOAT,
            glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&rgba))),
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        // Around the world the image wraps; over the poles it cannot, and clamps.
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);
    }
}

/// The GL state every offscreen pass here wants: nothing but the triangle it draws.
unsafe fn plain_state(gl: &glow::Context) {
    unsafe {
        gl.disable(glow::SCISSOR_TEST);
        gl.disable(glow::DEPTH_TEST);
        gl.disable(glow::BLEND);
        gl.disable(glow::CULL_FACE);
        gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
        gl.color_mask(true, true, true, true);
    }
}

impl EnvironmentMaps {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            // Without it a rough reflection shows every cube face's edge as a seam. Always
            // on in WebGL, and not a capability it will even accept.
            if !cfg!(target_arch = "wasm32") {
                gl.enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
            }
            let equirect = gl.create_texture().expect("equirect texture");
            upload_equirect(gl, equirect, &Equirect { width: 1, height: 1, pixels: vec![Vec3::ZERO] });
            let source_levels = SOURCE_SIZE.ilog2() as i32 + 1;
            Self {
                specular: cube_texture(gl, SPECULAR_SIZE, SPECULAR_LEVELS),
                source: cube_texture(gl, SOURCE_SIZE, source_levels),
                equirect,
                fbo: gl.create_framebuffer().expect("environment fbo"),
                vao: gl.create_vertex_array().expect("environment vao"),
                capture: compile_program(gl, shaders::FULLSCREEN_VS, &shaders::capture_fs(), &[]),
                prefilter: compile_program(gl, shaders::FULLSCREEN_VS, &shaders::prefilter_fs(), &[]),
            }
        }
    }

    /// Captures the procedural sky, wash and all but without the disc, and prefilters it.
    pub fn load_sky(&mut self, gl: &glow::Context, sky: &SkyParams) {
        unsafe {
            gl.use_program(Some(self.capture));
            sky.bind_dome(gl, self.capture);
            let set3 = |name: &str, v: Vec3| {
                if let Some(l) = gl.get_uniform_location(self.capture, name) {
                    gl.uniform_3_f32(Some(&l), v.x, v.y, v.z);
                }
            };
            set3("u_sun_dir", sky.sun_dir);
            set3("u_sun_color", sky.sun_color);
            self.capture_faces(gl, 0);
            self.prefilter(gl);
        }
    }

    /// Takes on a photograph. `lit` is what lights the tree, with its sun taken out;
    /// `shown` is the photograph as it was, for what is seen behind and under the tree.
    pub fn load_photo(&mut self, gl: &glow::Context, lit: &Equirect, shown: &Equirect) {
        unsafe {
            upload_equirect(gl, self.equirect, lit);
            gl.use_program(Some(self.capture));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.equirect));
            if let Some(l) = gl.get_uniform_location(self.capture, "u_source_equirect") {
                gl.uniform_1_i32(Some(&l), 0);
            }
            self.capture_faces(gl, 1);
            gl.bind_texture(glow::TEXTURE_2D, None);
            upload_equirect(gl, self.equirect, shown);
            self.prefilter(gl);
        }
    }

    /// Renders the bound capture program into each face of the source cube, then
    /// builds its mips for the prefilter to read.
    unsafe fn capture_faces(&self, gl: &glow::Context, source: i32) {
        unsafe {
            plain_state(gl);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.bind_vertex_array(Some(self.vao));
            gl.viewport(0, 0, SOURCE_SIZE, SOURCE_SIZE);
            if let Some(l) = gl.get_uniform_location(self.capture, "u_source") {
                gl.uniform_1_i32(Some(&l), source);
            }
            let face_loc = gl.get_uniform_location(self.capture, "u_face");
            for face in 0..6u32 {
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    Some(self.source),
                    0,
                );
                gl.uniform_1_i32(face_loc.as_ref(), face as i32);
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_CUBE_MAP_POSITIVE_X,
                None,
                0,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(self.source));
            gl.generate_mipmap(glow::TEXTURE_CUBE_MAP);
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }

    /// Convolves the source with GGX into each level of the specular cube.
    unsafe fn prefilter(&self, gl: &glow::Context) {
        unsafe {
            plain_state(gl);
            let p = self.prefilter;
            gl.use_program(Some(p));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(self.source));
            let u = |name: &str| gl.get_uniform_location(p, name);
            gl.uniform_1_i32(u("u_source").as_ref(), 0);
            gl.uniform_1_f32(u("u_source_size").as_ref(), SOURCE_SIZE as f32);
            gl.uniform_1_i32(u("u_samples").as_ref(), PREFILTER_SAMPLES);
            let face_loc = u("u_face");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            gl.bind_vertex_array(Some(self.vao));
            for level in 0..SPECULAR_LEVELS {
                let size = SPECULAR_SIZE >> level;
                gl.viewport(0, 0, size, size);
                gl.uniform_1_f32(u("u_perceptual").as_ref(), level_roughness(level));
                gl.uniform_1_f32(u("u_target_size").as_ref(), size as f32);
                for face in 0..6u32 {
                    gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0,
                        glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                        Some(self.specular),
                        level,
                    );
                    gl.uniform_1_i32(face_loc.as_ref(), face as i32);
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                }
            }
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_CUBE_MAP_POSITIVE_X,
                None,
                0,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_vertex_array(None);
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, None);
            gl.use_program(None);
        }
    }
}

/// Bakes Filament's DFG table once, on the GPU.
pub fn bake_dfg(gl: &glow::Context) -> glow::Texture {
    unsafe {
        let tex = gl.create_texture().expect("dfg texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA16F as i32,
            DFG_SIZE,
            DFG_SIZE,
            0,
            glow::RGBA,
            glow::FLOAT,
            glow::PixelUnpackData::Slice(None),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);

        let program = compile_program(gl, shaders::FULLSCREEN_VS, &shaders::dfg_fs(), &[]);
        let fbo = gl.create_framebuffer().expect("dfg fbo");
        let vao = gl.create_vertex_array().expect("dfg vao");
        plain_state(gl);
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
        gl.viewport(0, 0, DFG_SIZE, DFG_SIZE);
        gl.use_program(Some(program));
        gl.bind_vertex_array(Some(vao));
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
        gl.bind_vertex_array(None);
        gl.use_program(None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.delete_framebuffer(fbo);
        gl.delete_vertex_array(vao);
        gl.delete_program(program);
        tex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_levels_land_where_the_shaders_look_for_them() {
        // A lookup at roughness r reads mip max_lod * r * (2 - r); each level has to be
        // prefiltered at the roughness that lands exactly on it.
        for level in 0..SPECULAR_LEVELS {
            let r = level_roughness(level);
            let lod = MAX_LOD * r * (2.0 - r);
            assert!((lod - level as f32).abs() < 1e-4, "level {level}: r {r} reads lod {lod}");
        }
        assert_eq!(level_roughness(0), 0.0);
        assert!((level_roughness(SPECULAR_LEVELS - 1) - 1.0).abs() < 1e-6);
    }
}
