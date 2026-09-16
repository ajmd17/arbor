use bytemuck::cast_slice;
use eframe::glow;
use eframe::glow::HasContext;
use glam::{Mat4, Vec3};

use crate::shaders;

pub unsafe fn compile_program(gl: &glow::Context, vs_src: &str, fs_src: &str) -> glow::Program {
    unsafe {
        let compile = |kind: u32, src: &str| {
            let shader = gl.create_shader(kind).expect("create_shader");
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!("shader compile: {}", gl.get_shader_info_log(shader));
            }
            shader
        };
        let program = gl.create_program().expect("create_program");
        let vs = compile(glow::VERTEX_SHADER, vs_src);
        let fs = compile(glow::FRAGMENT_SHADER, fs_src);
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("program link: {}", gl.get_program_info_log(program));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        program
    }
}

fn loc(gl: &glow::Context, program: &glow::Program, name: &str) -> glow::UniformLocation {
    unsafe { gl.get_uniform_location(*program, name) }
        .unwrap_or_else(|| panic!("uniform {name} not found"))
}

pub struct GpuLines {
    program: glow::Program,
    vbo: glow::Buffer,
    vao: glow::VertexArray,
    u_mvp: glow::UniformLocation,
    vertex_count: i32,
}

impl GpuLines {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, shaders::LINES_VS, shaders::LINES_FS);
            let u_mvp = loc(gl, &program, "u_mvp");
            let vao = gl.create_vertex_array().expect("create vao");
            let vbo = gl.create_buffer().expect("create vbo");
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let stride = 6 * 4;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, stride, 12);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            Self {
                program,
                vbo,
                vao,
                u_mvp,
                vertex_count: 0,
            }
        }
    }

    pub fn upload(&mut self, gl: &glow::Context, verts: &[f32]) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                cast_slice(verts),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
        self.vertex_count = (verts.len() / 6) as i32;
    }

    pub fn draw(
        &self,
        gl: &glow::Context,
        mvp: [f32; 16],
        clip: [i32; 4],
        screen: [i32; 2],
        depth_test: bool,
    ) {
        unsafe {
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(clip[0], clip[1], clip[2], clip[3]);
            gl.viewport(0, 0, screen[0].max(1), screen[1].max(1));
            if depth_test {
                gl.enable(glow::DEPTH_TEST);
                gl.depth_func(glow::LEQUAL);
                gl.clear_depth_f32(1.0);
                gl.clear(glow::DEPTH_BUFFER_BIT);
            } else {
                gl.disable(glow::DEPTH_TEST);
            }
            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(Some(&self.u_mvp), false, &mvp);
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::LINES, 0, self.vertex_count);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::SCISSOR_TEST);
        }
    }
}

pub struct MaterialTextures {
    pub albedo: glow::Texture,
    pub normal: glow::Texture,
    pub roughness: glow::Texture,
}

impl MaterialTextures {}

pub unsafe fn create_texture(
    gl: &glow::Context,
    rgba: &[u8],
    width: u32,
    height: u32,
    srgb: bool,
) -> glow::Texture {
    unsafe {
        let texture = gl.create_texture().expect("create texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        let internal = if srgb {
            glow::SRGB8_ALPHA8 as i32
        } else {
            glow::RGBA8 as i32
        };
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            internal,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(rgba)),
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);
        texture
    }
}

unsafe fn load_image(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Some((rgba.into_raw(), w, h))
}

fn hash2(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(374761393)
        ^ (y as u32).wrapping_mul(668265263)
        ^ seed.wrapping_mul(2246822519);
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    h = h.wrapping_mul(3184899411);
    h ^= h >> 13;
    (h & 0xFFFFFF) as f32 / 16777215.0
}

fn vnoise(u: f32, v: f32, fu: f32, fv: f32, seed: u32) -> f32 {
    let x = u * fu;
    let y = v * fv;
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (uu, vv) = (s(xf), s(yf));
    let a = hash2(xi, yi, seed);
    let b = hash2(xi + 1, yi, seed);
    let c = hash2(xi, yi + 1, seed);
    let d = hash2(xi + 1, yi + 1, seed);
    a * (1.0 - uu) * (1.0 - vv) + b * uu * (1.0 - vv) + c * (1.0 - uu) * vv + d * uu * vv
}

fn bark_height(u: f32, v: f32) -> f32 {
    let fibers = vnoise(u, v, 14.0, 2.5, 1);
    let ridged = 1.0 - (fibers * 2.0 - 1.0).abs();
    let fine = vnoise(u, v, 38.0, 7.0, 3);
    let blotch = vnoise(u, v, 4.0, 3.0, 7);
    ridged * 0.6 + fine * 0.25 + blotch * 0.15
}

fn wrap_diff(h: &dyn Fn(f32, f32) -> f32, u: f32, v: f32, e: f32) -> (f32, f32) {
    let du = h((u + e) % 1.0, v) - h((u + 1.0 - e) % 1.0, v);
    let dv = h(u, (v + e) % 1.0) - h(u, (v + 1.0 - e) % 1.0);
    (du, dv)
}

pub fn procedural_bark_albedo() -> (Vec<u8>, u32, u32) {
    const S: u32 = 256;
    let mut px = Vec::with_capacity((S * S * 4) as usize);
    for y in 0..S {
        for x in 0..S {
            let u = x as f32 / S as f32;
            let v = y as f32 / S as f32;
            let h = bark_height(u, v).clamp(0.0, 1.0);
            let tint = vnoise(u, v, 3.0, 3.0, 11);
            let lerp = |a: f32, b: f32| a + (b - a) * h;
            let r = lerp(48.0, 122.0) + tint * 14.0;
            let g = lerp(35.0, 96.0) + tint * 11.0;
            let b = lerp(26.0, 74.0) + tint * 8.0;
            px.extend_from_slice(&[r as u8, g as u8, b as u8, 255]);
        }
    }
    (px, S, S)
}

pub fn procedural_bark_normal() -> (Vec<u8>, u32, u32) {
    const S: u32 = 256;
    const E: f32 = 1.0 / S as f32;
    let mut px = Vec::with_capacity((S * S * 4) as usize);
    let h = |u: f32, v: f32| bark_height(u, v);
    for y in 0..S {
        for x in 0..S {
            let u = x as f32 / S as f32;
            let v = y as f32 / S as f32;
            let (du, dv) = wrap_diff(&h, u, v, E);
            let n = Vec3::new(-du * 6.0, -dv * 6.0, 1.0).normalize_or(Vec3::Z);
            px.extend_from_slice(&[
                ((n.x * 0.5 + 0.5) * 255.0) as u8,
                ((n.y * 0.5 + 0.5) * 255.0) as u8,
                ((n.z * 0.5 + 0.5) * 255.0) as u8,
                255,
            ]);
        }
    }
    (px, S, S)
}

pub fn procedural_bark_roughness() -> (Vec<u8>, u32, u32) {
    const S: u32 = 256;
    let mut px = Vec::with_capacity((S * S * 4) as usize);
    for y in 0..S {
        for x in 0..S {
            let u = x as f32 / S as f32;
            let v = y as f32 / S as f32;
            let r = 0.82 + 0.14 * vnoise(u, v, 20.0, 6.0, 5);
            px.extend_from_slice(&[(r * 255.0) as u8, (r * 255.0) as u8, (r * 255.0) as u8, 255]);
        }
    }
    (px, S, S)
}

pub unsafe fn load_material(gl: &glow::Context, dir: &str) -> MaterialTextures {
    unsafe {
        let load_or = |file: &str, fallback: fn() -> (Vec<u8>, u32, u32), srgb: bool| {
            match load_image(&format!("{dir}/{file}")) {
                Some((data, w, h)) => create_texture(gl, &data, w, h, srgb),
                None => {
                    let (data, w, h) = fallback();
                    create_texture(gl, &data, w, h, srgb)
                }
            }
        };
        MaterialTextures {
            albedo: load_or("bark_albedo.png", procedural_bark_albedo, true),
            normal: load_or("bark_normal.png", procedural_bark_normal, false),
            roughness: load_or("bark_roughness.png", procedural_bark_roughness, false),
        }
    }
}

pub struct ShadowTarget {
    pub fbo: glow::Framebuffer,
    pub depth: glow::Texture,
    pub size: i32,
}

impl ShadowTarget {
    pub fn new(gl: &glow::Context, size: i32) -> Self {
        unsafe {
            let depth = gl.create_texture().expect("shadow depth texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(depth));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::DEPTH_COMPONENT24 as i32,
                size,
                size,
                0,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            let fbo = gl.create_framebuffer().expect("shadow fbo");
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::TEXTURE_2D,
                Some(depth),
                0,
            );
            gl.draw_buffers(&[glow::NONE]);
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                panic!("shadow framebuffer incomplete");
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            Self { fbo, depth, size }
        }
    }

    pub fn bind(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
        }
    }
}

pub struct DepthPass {
    pub program: glow::Program,
    pub u_light_view_proj: glow::UniformLocation,
}

impl DepthPass {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, shaders::DEPTH_VS, shaders::DEPTH_FS);
            let u_light_view_proj = loc(gl, &program, "u_light_view_proj");
            Self {
                program,
                u_light_view_proj,
            }
        }
    }
}

pub struct ColorPass {
    pub program: glow::Program,
    pub u_view_proj: glow::UniformLocation,
    pub u_color: glow::UniformLocation,
}

impl ColorPass {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, shaders::COLOR_VS, shaders::COLOR_FS);
            Self {
                program,
                u_view_proj: loc(gl, &program, "u_view_proj"),
                u_color: loc(gl, &program, "u_color"),
            }
        }
    }

    pub unsafe fn draw_wire(
        &self,
        gl: &glow::Context,
        mesh: &GpuMesh,
        view_proj: Mat4,
        color: [f32; 4],
    ) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(
                Some(&self.u_view_proj),
                false,
                &view_proj.to_cols_array(),
            );
            gl.uniform_4_f32(Some(&self.u_color), color[0], color[1], color[2], color[3]);
            gl.polygon_mode(glow::FRONT_AND_BACK, glow::LINE);
            mesh.bind_and_draw(gl);
            gl.polygon_mode(glow::FRONT_AND_BACK, glow::FILL);
            gl.use_program(None);
        }
    }
}

pub struct GpuMesh {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo_pos: glow::Buffer,
    vbo_nrm: glow::Buffer,
    vbo_uv: glow::Buffer,
    vbo_tan: glow::Buffer,
    ibo: glow::Buffer,
    index_count: i32,
    u_view_proj: glow::UniformLocation,
    u_light_view_proj: glow::UniformLocation,
    u_cam_pos: glow::UniformLocation,
    u_sun_dir: glow::UniformLocation,
    u_sun_color: glow::UniformLocation,
    u_albedo_color: glow::UniformLocation,
    u_roughness: glow::UniformLocation,
    u_metallic: glow::UniformLocation,
    u_mode: glow::UniformLocation,
    u_use_normal_map: glow::UniformLocation,
}

pub struct MeshDrawParams<'a> {
    pub view_proj: Mat4,
    pub light_view_proj: Mat4,
    pub cam_pos: Vec3,
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub mode: i32,
    pub use_normal_map: bool,
    pub material: &'a MaterialTextures,
    pub shadow_depth: glow::Texture,
}

impl GpuMesh {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, shaders::MESH_VS, shaders::MESH_FS);
            let vao = gl.create_vertex_array().expect("mesh vao");
            let vbo_pos = gl.create_buffer().expect("vbo pos");
            let vbo_nrm = gl.create_buffer().expect("vbo nrm");
            let vbo_uv = gl.create_buffer().expect("vbo uv");
            let vbo_tan = gl.create_buffer().expect("vbo tan");
            let ibo = gl.create_buffer().expect("ibo");

            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_pos));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_nrm));
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, 12, 0);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_uv));
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 8, 0);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_tan));
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 4, glow::FLOAT, false, 16, 0);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);

            let u = |name: &str| {
                gl.get_uniform_location(program, name)
                    .unwrap_or_else(|| panic!("uniform {name}"))
            };
            gl.use_program(Some(program));
            gl.uniform_1_i32(Some(&u("u_albedo_tex")), 0);
            gl.uniform_1_i32(Some(&u("u_normal_tex")), 1);
            gl.uniform_1_i32(Some(&u("u_rough_tex")), 2);
            gl.uniform_1_i32(Some(&u("u_shadow_tex")), 3);
            gl.use_program(None);

            Self {
                program,
                vao,
                vbo_pos,
                vbo_nrm,
                vbo_uv,
                vbo_tan,
                ibo,
                index_count: 0,
                u_view_proj: u("u_view_proj"),
                u_light_view_proj: u("u_light_view_proj"),
                u_cam_pos: u("u_cam_pos"),
                u_sun_dir: u("u_sun_dir"),
                u_sun_color: u("u_sun_color"),
                u_albedo_color: u("u_albedo_color"),
                u_roughness: u("u_roughness"),
                u_metallic: u("u_metallic"),
                u_mode: u("u_mode"),
                u_use_normal_map: u("u_use_normal_map"),
            }
        }
    }

    pub fn upload(&mut self, gl: &glow::Context, mesh: &arbor_core::Mesh) {
        unsafe {
            gl.bind_vertex_array(Some(self.vao));
            for (buffer, data) in [
                (self.vbo_pos, cast_slice(&mesh.positions)),
                (self.vbo_nrm, cast_slice(&mesh.normals)),
                (self.vbo_uv, cast_slice(&mesh.uvs)),
                (self.vbo_tan, cast_slice(&mesh.tangents)),
            ] {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, data, glow::STATIC_DRAW);
            }
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ibo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                cast_slice(&mesh.indices),
                glow::STATIC_DRAW,
            );
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
        }
        self.index_count = mesh.indices.len() as i32;
    }

    pub unsafe fn bind_and_draw(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_elements(glow::TRIANGLES, self.index_count, glow::UNSIGNED_INT, 0);
            gl.bind_vertex_array(None);
        }
    }

    pub fn draw(&self, gl: &glow::Context, p: &MeshDrawParams) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(
                Some(&self.u_view_proj),
                false,
                &p.view_proj.to_cols_array(),
            );
            gl.uniform_matrix_4_f32_slice(
                Some(&self.u_light_view_proj),
                false,
                &p.light_view_proj.to_cols_array(),
            );
            gl.uniform_3_f32(Some(&self.u_cam_pos), p.cam_pos.x, p.cam_pos.y, p.cam_pos.z);
            gl.uniform_3_f32(Some(&self.u_sun_dir), p.sun_dir.x, p.sun_dir.y, p.sun_dir.z);
            gl.uniform_3_f32(
                Some(&self.u_sun_color),
                p.sun_color.x,
                p.sun_color.y,
                p.sun_color.z,
            );
            gl.uniform_3_f32(Some(&self.u_albedo_color), 1.0, 1.0, 1.0);
            gl.uniform_1_f32(Some(&self.u_roughness), 1.0);
            gl.uniform_1_f32(Some(&self.u_metallic), 0.0);
            gl.uniform_1_i32(Some(&self.u_mode), p.mode);
            gl.uniform_1_i32(
                Some(&self.u_use_normal_map),
                i32::from(p.use_normal_map),
            );
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.material.albedo));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.material.normal));
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.material.roughness));
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.shadow_depth));
            self.bind_and_draw(gl);
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(None);
        }
    }
}
