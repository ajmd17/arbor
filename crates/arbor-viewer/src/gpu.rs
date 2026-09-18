use bytemuck::cast_slice;
use eframe::glow;
use eframe::glow::HasContext;
use glam::{Mat4, Vec3};

use arbor_core::cluster::Bitmap;
use arbor_core::species::LeafParams;
use arbor_core::textures::{self, MapSource};

use crate::shaders;

/// `attribs` names the vertex inputs in the order the VAO binds them. GLSL 150 has
/// no `layout(location = ...)`, so without this the linker picks indices itself and a
/// shader that skips an input silently shifts every later one. Binding a name the
/// shader does not declare is ignored, so passing the full VAO layout is safe.
pub unsafe fn compile_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
    attribs: &[&str],
) -> glow::Program {
    unsafe {
        let compile = |kind: u32, src: &str| {
            let shader = gl.create_shader(kind).expect("create_shader");
            gl.shader_source(shader, &for_this_gl(src));
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
        for (index, name) in attribs.iter().enumerate() {
            gl.bind_attrib_location(program, index as u32, name);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("program link: {}", gl.get_program_info_log(program));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        program
    }
}

/// The shaders are written as GLSL 1.50. WebGL2 takes GLSL ES 3.00, which reads them
/// the same once the header says so and the precisions are spelled out: high all
/// through, or the shadow map's depths come back too coarse to compare.
fn for_this_gl(src: &str) -> std::borrow::Cow<'_, str> {
    #[cfg(target_arch = "wasm32")]
    if let Some(body) = src.strip_prefix("#version 150") {
        return format!(
            "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;{body}"
        )
        .into();
    }
    src.into()
}

/// Bark vertex attributes, in the order the mesh vertex array binds them. The colour
/// pass, the shadow pass and the wireframe all draw from that one array.
const MESH_ATTRIBS: [&str; 8] = [
    "a_pos",
    "a_normal",
    "a_uv",
    "a_tangent",
    "a_weathering",
    "a_wind1",
    "a_wind2",
    "a_wind3",
];

/// The wireframe draws both vertex arrays, so it names the bark's layout with the leaf
/// origin after it. The two arrays put the sway in the same slots, which is what lets
/// one program read either; the bark array leaves the origin's slot empty, and the
/// shader never reads it there.
const WIRE_ATTRIBS: [&str; 9] = [
    "a_pos",
    "a_normal",
    "a_uv",
    "a_tangent",
    "a_weathering",
    "a_wind1",
    "a_wind2",
    "a_wind3",
    "a_leaf_origin",
];

/// Where the sway sits in both vertex arrays: three vec4s, one per branch order, read
/// out of one buffer.
const SWAY_SLOT: u32 = 5;

/// Points the three sway attributes at the bound buffer of `arbor_core::Sway`s.
unsafe fn sway_attribs(gl: &glow::Context) {
    const STRIDE: i32 = (4 * arbor_core::wind::SWAY_ORDERS * 4) as i32;
    unsafe {
        for order in 0..arbor_core::wind::SWAY_ORDERS as u32 {
            gl.enable_vertex_attrib_array(SWAY_SLOT + order);
            gl.vertex_attrib_pointer_f32(
                SWAY_SLOT + order,
                4,
                glow::FLOAT,
                false,
                STRIDE,
                (order * 16) as i32,
            );
        }
    }
}

/// The wind as the tree's shaders take it: the weather the scene sets, and how the
/// species gives to it. The default is still air.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindUniforms {
    /// Seconds on the wind's clock.
    pub time: f32,
    /// The way the wind blows, on the ground: x and z of a unit vector.
    pub direction: [f32; 2],
    /// 0 for still air, 1 for a full gale.
    pub strength: f32,
    pub gustiness: f32,
    /// How far the trunk, limbs, branches and twigs bend in a full gale, in radians.
    pub flexibility: [f32; 4],
    /// How fast the trunk sways, in hertz.
    pub frequency: f32,
    /// How far a leaf flutters in a full gale, in radians.
    pub flutter: f32,
    /// The trunk bends along its height, so it has to know what that is.
    pub tree_height: f32,
}

impl WindUniforms {
    pub fn new(
        species: &arbor_core::WindParams,
        time: f32,
        direction_deg: f32,
        strength: f32,
        gustiness: f32,
        tree_height: f32,
    ) -> Self {
        let a = direction_deg.to_radians();
        Self {
            time,
            direction: [a.cos(), a.sin()],
            strength: strength.max(0.0),
            gustiness: gustiness.clamp(0.0, 1.0),
            flexibility: species.flexibility.map(|f| f.max(0.0)),
            frequency: species.frequency.max(0.0),
            flutter: species.flutter.max(0.0),
            tree_height,
        }
    }

    /// Furthest the tree can move from where it was grown, as a generous bound, so the
    /// shadow frustum can be fitted to where the tree actually is rather than to where it
    /// stands in still air. Lean and sway on the trunk at the top of a gust, plus a limb
    /// swinging on top of that.
    pub fn reach(&self) -> f32 {
        if self.strength <= 0.0 {
            return 0.0;
        }
        let peak = self.strength * (1.0 + self.gustiness) * 1.6;
        peak * (self.tree_height * (self.flexibility[0] + 0.4 * self.flexibility[1]) + 0.5)
    }

    /// Sets whichever of the wind uniforms the program kept. Ones the compiler dropped
    /// are skipped, like the sky's.
    pub unsafe fn bind(&self, gl: &glow::Context, program: glow::Program) {
        unsafe {
            if let Some(l) = gl.get_uniform_location(program, "u_time") {
                gl.uniform_1_f32(Some(&l), self.time);
            }
            if let Some(l) = gl.get_uniform_location(program, "u_wind") {
                gl.uniform_4_f32(
                    Some(&l),
                    self.direction[0],
                    self.direction[1],
                    self.strength,
                    self.gustiness,
                );
            }
            if let Some(l) = gl.get_uniform_location(program, "u_wind_flex") {
                let f = self.flexibility;
                gl.uniform_4_f32(Some(&l), f[0], f[1], f[2], f[3]);
            }
            if let Some(l) = gl.get_uniform_location(program, "u_wind_motion") {
                gl.uniform_2_f32(Some(&l), self.frequency, self.flutter);
            }
            if let Some(l) = gl.get_uniform_location(program, "u_tree_height") {
                gl.uniform_1_f32(Some(&l), self.tree_height);
            }
        }
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
            let program = compile_program(gl, shaders::LINES_VS, shaders::LINES_FS, &["a_pos", "a_col"]);
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

#[derive(Clone, Copy)]
pub struct MaterialTextures {
    pub albedo: glow::Texture,
    pub normal: glow::Texture,
    pub roughness: glow::Texture,
}

impl MaterialTextures {
    /// Materials are swapped whenever the species changes, so the old set has to go
    /// back to the driver rather than leak one texture trio per switch.
    pub fn delete(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.albedo);
            gl.delete_texture(self.normal);
            gl.delete_texture(self.roughness);
        }
    }
}

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

/// Uploads a foliage cutout with the mip chain that holds its alpha coverage.
///
/// Averaged alpha is the right coverage for alpha-to-coverage only while a card is
/// bigger than a pixel. Once it is not, two things break at once: a leaf that fills a
/// fraction of its cell averages down to that fraction, and alpha-to-coverage hands
/// equal coverages the same sample mask, so cards stacked over one pixel never add
/// up. The shader falls back to a cutoff there, and this is the chain that keeps a
/// cutoff honest.
pub unsafe fn create_cutout_texture(
    gl: &glow::Context,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> glow::Texture {
    unsafe {
        let chain =
            crate::mipmap::coverage_preserving_chain(rgba, width, height, LEAF_ALPHA_CUTOFF);
        let texture = gl.create_texture().expect("create texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        for (level, (px, w, h)) in chain.iter().enumerate() {
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                level as i32,
                glow::SRGB8_ALPHA8 as i32,
                *w as i32,
                *h as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(px)),
            );
        }
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, 0);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAX_LEVEL,
            chain.len() as i32 - 1,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        // Cells of an atlas must not bleed into each other at their edges.
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
        set_max_anisotropy(gl);
        gl.bind_texture(glow::TEXTURE_2D, None);
        texture
    }
}

/// Foliage is mostly viewed at glancing angles, where isotropic filtering blurs a
/// whole card into mush. Skipped silently where the driver does not offer it.
unsafe fn set_max_anisotropy(gl: &glow::Context) {
    const TEXTURE_MAX_ANISOTROPY: u32 = 0x84FE;
    const MAX_TEXTURE_MAX_ANISOTROPY: u32 = 0x84FF;
    unsafe {
        // WebGL names its extensions without the prefix.
        let extensions = gl.supported_extensions();
        if !extensions.contains("GL_EXT_texture_filter_anisotropic")
            && !extensions.contains("EXT_texture_filter_anisotropic")
        {
            return;
        }
        let max = gl.get_parameter_f32(MAX_TEXTURE_MAX_ANISOTROPY);
        if max > 1.0 {
            gl.tex_parameter_f32(glow::TEXTURE_2D, TEXTURE_MAX_ANISOTROPY, max.min(8.0));
        }
    }
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

/// Loads `<name>_albedo.png` and friends, falling back to the procedural bark so the
/// viewer still runs against a bare checkout with no texture assets.
pub unsafe fn load_material<S: MapSource + ?Sized>(
    gl: &glow::Context,
    maps: &S,
    name: &str,
) -> MaterialTextures {
    unsafe {
        let load_or = |map: &str, fallback: fn() -> (Vec<u8>, u32, u32), srgb: bool| {
            match textures::load_bitmap(maps, name, map) {
                Some(m) => create_texture(gl, &m.pixels, m.width, m.height, srgb),
                None => {
                    let (data, w, h) = fallback();
                    create_texture(gl, &data, w, h, srgb)
                }
            }
        };
        MaterialTextures {
            albedo: load_or("albedo", procedural_bark_albedo, true),
            normal: load_or("normal", procedural_bark_normal, false),
            roughness: load_or("roughness", procedural_bark_roughness, false),
        }
    }
}

/// Plain bark to draw with while the real maps are on their way, which on the web they
/// are for a moment.
pub unsafe fn placeholder_material(gl: &glow::Context) -> MaterialTextures {
    unsafe {
        MaterialTextures {
            albedo: create_texture(gl, &[88, 66, 50, 255], 1, 1, true),
            normal: create_texture(gl, &[128, 128, 255, 255], 1, 1, false),
            roughness: create_texture(gl, &[220, 220, 220, 255], 1, 1, false),
        }
    }
}

/// A leaf texture has no sensible procedural stand-in, so a missing one is reported
/// rather than silently replaced by bark.
///
/// When the species clusters, the art on disk is a single leaf and the atlas actually
/// sampled is grown from it here. That keeps one leaf in the repository instead of a
/// baked sheet per species, and lets the arrangement be retuned by editing numbers.
pub unsafe fn load_leaf_material<S: MapSource + ?Sized>(
    gl: &glow::Context,
    maps: &S,
    lp: &LeafParams,
) -> Option<MaterialTextures> {
    unsafe {
        // Loaded, and clustered where the species clusters, by the same code the
        // exporters use, so what is exported is what is on screen.
        let started = web_time::Instant::now();
        let maps = textures::load_leaf_maps(maps, lp)?;
        if let Some(cluster) = &lp.cluster {
            println!(
                "clustered {} into {}x{} from {} leaves, coverage {:.3} ({:.0} ms)",
                lp.texture,
                maps.albedo.width,
                maps.albedo.height,
                cluster.count,
                maps.albedo.mean_alpha(),
                started.elapsed().as_secs_f32() * 1000.0
            );
        }
        let (albedo, normal, roughness) = (maps.albedo, maps.normal, maps.roughness);

        let upload = |map: Option<Bitmap>, fill: u8| match map {
            Some(m) => create_texture(gl, &m.pixels, m.width, m.height, false),
            None => create_texture(gl, &[fill, fill, fill, 255], 1, 1, false),
        };
        Some(MaterialTextures {
            albedo: create_cutout_texture(gl, &albedo.pixels, albedo.width, albedo.height),
            normal: upload(normal, 128),
            roughness: upload(roughness, 200),
        })
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
            // Outside the map is fully lit, not a repeat of whatever depth happened to
            // sit on the border. Clamping to the edge instead smears that last row of
            // texels outward, and with a low sun one shadow texel covers metres of
            // ground, so the smear reads as long straight bands lying across it.
            // WebGL has no border to clamp to; there the shaders' own test for falling
            // off the map does the work, leaving only the filter taps at its very edge.
            #[cfg(not(target_arch = "wasm32"))]
            let (wrap, border) = (glow::CLAMP_TO_BORDER, true);
            #[cfg(target_arch = "wasm32")]
            let (wrap, border) = (glow::CLAMP_TO_EDGE, false);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, wrap as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap as i32);
            if border {
                gl.tex_parameter_f32_slice(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_BORDER_COLOR,
                    &[1.0, 1.0, 1.0, 1.0],
                );
            }
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
            let program = compile_program(gl, &shaders::depth_vs(), shaders::DEPTH_FS, &MESH_ATTRIBS);
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
    u_leaf: glow::UniformLocation,
}

impl ColorPass {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, &shaders::color_vs(), shaders::COLOR_FS, &WIRE_ATTRIBS);
            Self {
                program,
                u_view_proj: loc(gl, &program, "u_view_proj"),
                u_color: loc(gl, &program, "u_color"),
                u_leaf: loc(gl, &program, "u_leaf"),
            }
        }
    }

    /// Wireframes whatever is handed in. Leaf cards are most of a tree by triangle
    /// count, so a wireframe that only showed bark hid where the triangles were.
    pub unsafe fn draw_wire(
        &self,
        gl: &glow::Context,
        mesh: &GpuMesh,
        leaves: Option<&GpuLeaves>,
        view_proj: Mat4,
        color: [f32; 4],
        wind: &WindUniforms,
    ) {
        unsafe {
            gl.use_program(Some(self.program));
            wind.bind(gl, self.program);
            gl.uniform_matrix_4_f32_slice(
                Some(&self.u_view_proj),
                false,
                &view_proj.to_cols_array(),
            );
            gl.uniform_4_f32(Some(&self.u_color), color[0], color[1], color[2], color[3]);
            gl.polygon_mode(glow::FRONT_AND_BACK, glow::LINE);
            gl.uniform_1_i32(Some(&self.u_leaf), 0);
            mesh.bind_and_draw(gl);
            if let Some(leaves) = leaves {
                gl.uniform_1_i32(Some(&self.u_leaf), 1);
                leaves.bind_and_draw(gl);
            }
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
    vbo_weather: glow::Buffer,
    vbo_sway: glow::Buffer,
    ibo: glow::Buffer,
    index_count: i32,
    u_view_proj: glow::UniformLocation,
    u_light_view_proj: glow::UniformLocation,
    u_cam_pos: glow::UniformLocation,
    u_sun_dir: glow::UniformLocation,
    u_sun_color: glow::UniformLocation,
    u_albedo_color: glow::UniformLocation,
    u_dead_color: glow::UniformLocation,
    u_dead_weathering: glow::UniformLocation,
    u_moss_color: glow::UniformLocation,
    u_moss_height: glow::UniformLocation,
    u_moss_amount: glow::UniformLocation,
    u_bark_darken_low: glow::UniformLocation,
    u_roughness: glow::UniformLocation,
    u_metallic: glow::UniformLocation,
    u_mode: glow::UniformLocation,
    u_use_normal_map: glow::UniformLocation,
    u_normal_bias: glow::UniformLocation,
}

/// Colour of the moss on the bark, how far up it reaches, how much of the bark it
/// takes, and how far the bark weathers darker toward the foot of the tree.
#[derive(Clone, Copy, Debug)]
pub struct BarkLook {
    pub moss_color: Vec3,
    pub moss_height: f32,
    pub moss_amount: f32,
    pub darken_low: f32,
    pub tint: Vec3,
    pub dead_color: Vec3,
    pub dead_weathering: f32,
}

impl BarkLook {
    pub fn from_species(mp: &arbor_core::species::MeshParams) -> Self {
        Self {
            moss_color: Vec3::from(mp.moss_color),
            moss_height: mp.moss_height,
            moss_amount: mp.moss_amount,
            darken_low: mp.bark_darken_low,
            tint: Vec3::from(mp.bark_tint),
            dead_color: Vec3::from(mp.dead_wood_color),
            dead_weathering: mp.dead_wood_weathering,
        }
    }
}

pub struct MeshDrawParams<'a> {
    pub bark: BarkLook,
    pub sky: &'a SkyParams,
    pub normal_bias: f32,
    pub view_proj: Mat4,
    pub light_view_proj: Mat4,
    pub cam_pos: Vec3,
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub mode: i32,
    pub use_normal_map: bool,
    pub material: &'a MaterialTextures,
    pub shadow_depth: glow::Texture,
    pub wind: &'a WindUniforms,
}

impl GpuMesh {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, &shaders::mesh_vs(), &shaders::mesh_fs(), &MESH_ATTRIBS);
            let vao = gl.create_vertex_array().expect("mesh vao");
            let vbo_pos = gl.create_buffer().expect("vbo pos");
            let vbo_nrm = gl.create_buffer().expect("vbo nrm");
            let vbo_uv = gl.create_buffer().expect("vbo uv");
            let vbo_tan = gl.create_buffer().expect("vbo tan");
            let vbo_weather = gl.create_buffer().expect("vbo weathering");
            let vbo_sway = gl.create_buffer().expect("vbo sway");
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
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_weather));
            gl.enable_vertex_attrib_array(4);
            gl.vertex_attrib_pointer_f32(4, 1, glow::FLOAT, false, 4, 0);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_sway));
            sway_attribs(gl);
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
                vbo_weather,
                vbo_sway,
                ibo,
                index_count: 0,
                u_view_proj: u("u_view_proj"),
                u_light_view_proj: u("u_light_view_proj"),
                u_cam_pos: u("u_cam_pos"),
                u_sun_dir: u("u_sun_dir"),
                u_sun_color: u("u_sun_color"),
                u_albedo_color: u("u_albedo_color"),
                u_dead_color: u("u_dead_color"),
                u_dead_weathering: u("u_dead_weathering"),
                u_moss_color: u("u_moss_color"),
                u_moss_height: u("u_moss_height"),
                u_moss_amount: u("u_moss_amount"),
                u_bark_darken_low: u("u_bark_darken_low"),
                u_roughness: u("u_roughness"),
                u_metallic: u("u_metallic"),
                u_mode: u("u_mode"),
                u_use_normal_map: u("u_use_normal_map"),
                u_normal_bias: u("u_normal_bias"),
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
                (self.vbo_weather, cast_slice(&mesh.weathering)),
                (self.vbo_sway, cast_slice(&mesh.sway)),
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
            gl.uniform_3_f32(
                Some(&self.u_albedo_color),
                p.bark.tint.x,
                p.bark.tint.y,
                p.bark.tint.z,
            );
            gl.uniform_3_f32(
                Some(&self.u_dead_color),
                p.bark.dead_color.x,
                p.bark.dead_color.y,
                p.bark.dead_color.z,
            );
            gl.uniform_1_f32(Some(&self.u_dead_weathering), p.bark.dead_weathering);
            gl.uniform_1_f32(Some(&self.u_roughness), 1.0);
            gl.uniform_1_f32(Some(&self.u_metallic), 0.0);
            gl.uniform_3_f32(
                Some(&self.u_moss_color),
                p.bark.moss_color.x,
                p.bark.moss_color.y,
                p.bark.moss_color.z,
            );
            gl.uniform_1_f32(Some(&self.u_moss_height), p.bark.moss_height);
            gl.uniform_1_f32(Some(&self.u_moss_amount), p.bark.moss_amount);
            gl.uniform_1_f32(Some(&self.u_bark_darken_low), p.bark.darken_low);
            gl.uniform_1_i32(Some(&self.u_mode), p.mode);
            gl.uniform_1_i32(
                Some(&self.u_use_normal_map),
                i32::from(p.use_normal_map),
            );
            gl.uniform_1_f32(Some(&self.u_normal_bias), p.normal_bias);
            p.sky.bind(gl, self.program);
            p.wind.bind(gl, self.program);
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

/// Threshold the leaf shadow pass tests alpha against to carve its silhouette. The
/// colour pass resolves the same cutout with alpha-to-coverage instead, so it does not
/// share this value.
pub const LEAF_ALPHA_CUTOFF: f32 = 0.35;

/// Leaf vertex attributes, in the order the one leaf vertex array binds them. The
/// colour pass and the depth pass draw from that same array, so they have to agree on
/// it or the depth pass reads positions out of the tint buffer.
const LEAF_ATTRIBS: [&str; 9] = [
    "a_pos",
    "a_normal",
    "a_uv",
    "a_tint",
    "a_atlas_v",
    "a_wind1",
    "a_wind2",
    "a_wind3",
    "a_leaf_origin",
];

/// Which atlas cells a leaf card samples, and how hard the alpha test bites.
#[derive(Clone, Copy)]
pub struct LeafMaterialParams {
    pub atlas_scale: [f32; 2],
    pub atlas_front: [f32; 2],
    pub atlas_back: [f32; 2],
    pub alpha_cutoff: f32,
    /// Mip level where soft coverage gives way to the cutoff.
    pub coverage_lod: f32,
    pub translucency: f32,
    /// How far the cutout edge is sharpened to a pixel rather than left soft.
    pub edge_sharpness: f32,
    /// How far the shading normal was leaned toward the outside of the crown, and how
    /// much of that lean a card seen from behind keeps.
    pub normal_blend: f32,
    pub backface_volume: f32,
    /// How much of the sun a leaf in shadow still loses.
    pub self_shadow: f32,
}

impl LeafMaterialParams {
    pub fn from_species(lp: &arbor_core::LeafParams, translucency: f32) -> Self {
        let cols = lp.atlas_cols.max(1);
        let rows = lp.atlas_rows.max(1);
        // Clustering stacks its arrangements down the sheet, so the atlas the shader
        // samples is taller than the one the species describes. A card reaches its own
        // arrangement by the v offset it carries; the cells here are the first one's.
        let variants = lp.cluster.as_ref().map_or(1, |c| c.variants.max(1));
        let tall = rows * variants;
        let cell = |index: u32| {
            let i = index.min(cols * rows - 1);
            [
                (i % cols) as f32 / cols as f32,
                (i / cols) as f32 / tall as f32,
            ]
        };
        Self {
            atlas_scale: [1.0 / cols as f32, 1.0 / tall as f32],
            atlas_front: cell(lp.atlas_front),
            atlas_back: cell(lp.atlas_back),
            alpha_cutoff: LEAF_ALPHA_CUTOFF,
            // Late on purpose. The mip chain already rescales alpha to hold coverage,
            // so soft coverage stays dense well into the distance and the cutoff only
            // has to take over once a card is down to about a pixel. Crossing over
            // early costs nothing but the antialiasing on every leaf edge.
            coverage_lod: 1.3,
            translucency,
            edge_sharpness: lp.edge_sharpness.clamp(0.0, 1.0),
            normal_blend: lp.normal_blend.clamp(0.0, 1.0),
            backface_volume: lp.backface_volume.clamp(0.0, 1.0),
            self_shadow: lp.self_shadow.clamp(0.0, 1.0),
        }
    }
}

pub struct LeafDepthPass {
    pub program: glow::Program,
    u_light_view_proj: glow::UniformLocation,
    u_atlas_scale: glow::UniformLocation,
    u_atlas_front: glow::UniformLocation,
    u_alpha_cutoff: glow::UniformLocation,
}

impl LeafDepthPass {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, &shaders::leaf_depth_vs(), shaders::LEAF_DEPTH_FS, &LEAF_ATTRIBS);
            gl.use_program(Some(program));
            gl.uniform_1_i32(Some(&loc(gl, &program, "u_albedo_tex")), 0);
            gl.use_program(None);
            Self {
                u_light_view_proj: loc(gl, &program, "u_light_view_proj"),
                u_atlas_scale: loc(gl, &program, "u_atlas_scale"),
                u_atlas_front: loc(gl, &program, "u_atlas_front"),
                u_alpha_cutoff: loc(gl, &program, "u_alpha_cutoff"),
                program,
            }
        }
    }

    pub unsafe fn draw(
        &self,
        gl: &glow::Context,
        leaves: &GpuLeaves,
        light_view_proj: Mat4,
        material: &MaterialTextures,
        p: LeafMaterialParams,
        wind: &WindUniforms,
    ) {
        unsafe {
            gl.use_program(Some(self.program));
            wind.bind(gl, self.program);
            gl.uniform_matrix_4_f32_slice(
                Some(&self.u_light_view_proj),
                false,
                &light_view_proj.to_cols_array(),
            );
            gl.uniform_2_f32(Some(&self.u_atlas_scale), p.atlas_scale[0], p.atlas_scale[1]);
            gl.uniform_2_f32(Some(&self.u_atlas_front), p.atlas_front[0], p.atlas_front[1]);
            gl.uniform_1_f32(Some(&self.u_alpha_cutoff), p.alpha_cutoff);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(material.albedo));
            leaves.bind_and_draw(gl);
            gl.use_program(None);
        }
    }
}

pub struct LeafDrawParams<'a> {
    pub sky: &'a SkyParams,
    pub normal_bias: f32,
    pub view_proj: Mat4,
    pub light_view_proj: Mat4,
    pub cam_pos: Vec3,
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub mode: i32,
    pub material: &'a MaterialTextures,
    pub shadow_depth: glow::Texture,
    pub leaf: LeafMaterialParams,
    pub wind: &'a WindUniforms,
}

pub struct GpuLeaves {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo_pos: glow::Buffer,
    vbo_nrm: glow::Buffer,
    vbo_uv: glow::Buffer,
    vbo_tint: glow::Buffer,
    vbo_atlas_v: glow::Buffer,
    vbo_sway: glow::Buffer,
    vbo_origin: glow::Buffer,
    ibo: glow::Buffer,
    index_count: i32,
    u_view_proj: glow::UniformLocation,
    u_light_view_proj: glow::UniformLocation,
    u_cam_pos: glow::UniformLocation,
    u_sun_dir: glow::UniformLocation,
    u_sun_color: glow::UniformLocation,
    u_atlas_scale: glow::UniformLocation,
    u_atlas_front: glow::UniformLocation,
    u_atlas_back: glow::UniformLocation,
    u_alpha_cutoff: glow::UniformLocation,
    u_coverage_lod: glow::UniformLocation,
    u_translucency: glow::UniformLocation,
    u_edge_sharpness: glow::UniformLocation,
    u_normal_blend: glow::UniformLocation,
    u_backface_volume: glow::UniformLocation,
    u_self_shadow: glow::UniformLocation,
    u_mode: glow::UniformLocation,
    u_normal_bias: glow::UniformLocation,
}

impl GpuLeaves {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, &shaders::leaf_vs(), &shaders::leaf_fs(), &LEAF_ATTRIBS);
            let vao = gl.create_vertex_array().expect("leaf vao");
            let vbo_pos = gl.create_buffer().expect("leaf pos");
            let vbo_nrm = gl.create_buffer().expect("leaf nrm");
            let vbo_uv = gl.create_buffer().expect("leaf uv");
            let vbo_tint = gl.create_buffer().expect("leaf tint");
            let vbo_atlas_v = gl.create_buffer().expect("leaf atlas v");
            let vbo_sway = gl.create_buffer().expect("leaf sway");
            let vbo_origin = gl.create_buffer().expect("leaf origin");
            let ibo = gl.create_buffer().expect("leaf ibo");

            gl.bind_vertex_array(Some(vao));
            for (slot, buffer, size, stride) in [
                (0u32, vbo_pos, 3i32, 12i32),
                (1, vbo_nrm, 3, 12),
                (2, vbo_uv, 2, 8),
                (3, vbo_tint, 4, 16),
                (4, vbo_atlas_v, 1, 4),
                (8, vbo_origin, 3, 12),
            ] {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
                gl.enable_vertex_attrib_array(slot);
                gl.vertex_attrib_pointer_f32(slot, size, glow::FLOAT, false, stride, 0);
            }
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_sway));
            sway_attribs(gl);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);

            let u = |name: &str| loc(gl, &program, name);
            gl.use_program(Some(program));
            gl.uniform_1_i32(Some(&u("u_albedo_tex")), 0);
            gl.uniform_1_i32(Some(&u("u_rough_tex")), 2);
            gl.uniform_1_i32(Some(&u("u_shadow_tex")), 3);
            gl.use_program(None);

            Self {
                vao,
                vbo_pos,
                vbo_nrm,
                vbo_uv,
                vbo_tint,
                vbo_atlas_v,
                vbo_sway,
                vbo_origin,
                ibo,
                index_count: 0,
                u_view_proj: u("u_view_proj"),
                u_light_view_proj: u("u_light_view_proj"),
                u_cam_pos: u("u_cam_pos"),
                u_sun_dir: u("u_sun_dir"),
                u_sun_color: u("u_sun_color"),
                u_atlas_scale: u("u_atlas_scale"),
                u_atlas_front: u("u_atlas_front"),
                u_atlas_back: u("u_atlas_back"),
                u_alpha_cutoff: u("u_alpha_cutoff"),
                u_coverage_lod: u("u_coverage_lod"),
                u_edge_sharpness: u("u_edge_sharpness"),
                u_normal_blend: u("u_normal_blend"),
                u_backface_volume: u("u_backface_volume"),
                u_self_shadow: u("u_self_shadow"),
                u_translucency: u("u_translucency"),
                u_mode: u("u_mode"),
                u_normal_bias: u("u_normal_bias"),
                program,
            }
        }
    }

    pub fn upload(&mut self, gl: &glow::Context, leaves: &arbor_core::LeafMesh) {
        unsafe {
            gl.bind_vertex_array(Some(self.vao));
            for (buffer, data) in [
                (self.vbo_pos, cast_slice(&leaves.positions)),
                (self.vbo_nrm, cast_slice(&leaves.normals)),
                (self.vbo_uv, cast_slice(&leaves.uvs)),
                (self.vbo_tint, cast_slice(&leaves.tints)),
                (self.vbo_atlas_v, cast_slice(&leaves.atlas_v)),
                (self.vbo_sway, cast_slice(&leaves.sway)),
                (self.vbo_origin, cast_slice(&leaves.origins)),
            ] {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, data, glow::STATIC_DRAW);
            }
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ibo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                cast_slice(&leaves.indices),
                glow::STATIC_DRAW,
            );
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
        }
        self.index_count = leaves.indices.len() as i32;
    }

    pub unsafe fn bind_and_draw(&self, gl: &glow::Context) {
        unsafe {
            if self.index_count == 0 {
                return;
            }
            // Cards are single quads lit from either side, so culling has to be off
            // and the fragment shader decides which face it is looking at.
            gl.disable(glow::CULL_FACE);
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_elements(glow::TRIANGLES, self.index_count, glow::UNSIGNED_INT, 0);
            gl.bind_vertex_array(None);
            // Left disabled on purpose: that is the state the rest of the frame runs
            // in, and the bark tubes have never been checked for consistent winding.
        }
    }

    pub fn draw(&self, gl: &glow::Context, p: &LeafDrawParams) {
        if self.index_count == 0 {
            return;
        }
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
            let m = p.leaf;
            gl.uniform_2_f32(Some(&self.u_atlas_scale), m.atlas_scale[0], m.atlas_scale[1]);
            gl.uniform_2_f32(Some(&self.u_atlas_front), m.atlas_front[0], m.atlas_front[1]);
            gl.uniform_2_f32(Some(&self.u_atlas_back), m.atlas_back[0], m.atlas_back[1]);
            gl.uniform_1_f32(Some(&self.u_alpha_cutoff), m.alpha_cutoff);
            gl.uniform_1_f32(Some(&self.u_coverage_lod), m.coverage_lod);
            gl.uniform_1_f32(Some(&self.u_translucency), m.translucency);
            gl.uniform_1_f32(Some(&self.u_edge_sharpness), m.edge_sharpness);
            gl.uniform_1_f32(Some(&self.u_normal_blend), m.normal_blend);
            gl.uniform_1_f32(Some(&self.u_backface_volume), m.backface_volume);
            gl.uniform_1_f32(Some(&self.u_self_shadow), m.self_shadow);
            gl.uniform_1_i32(Some(&self.u_mode), p.mode);
            gl.uniform_1_f32(Some(&self.u_normal_bias), p.normal_bias);
            p.sky.bind(gl, self.program);
            p.wind.bind(gl, self.program);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.material.albedo));
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.material.roughness));
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.shadow_depth));
            // Bark tubes are closed and wound counter-clockwise when seen from
            // outside, so their interior is never worth rasterising. Culling it also
            // means a fragment always faces the camera, which is what lets the shader
            // use the normal it was handed instead of flipping it toward the viewer.
            gl.front_face(glow::CCW);
            gl.cull_face(glow::BACK);
            gl.enable(glow::CULL_FACE);
            // The filtered alpha becomes real per-sample coverage rather than a blend,
            // so a minified canopy resolves to its true density with no sorting and no
            // fringe. Blending on top of that would composite the leaf twice, once in
            // the coverage resolve and once in the blend, so it has to be off.
            gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            gl.disable(glow::BLEND);
            self.bind_and_draw(gl);
            gl.enable(glow::BLEND);
            gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            gl.disable(glow::CULL_FACE);
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(None);
        }
    }
}

use crate::lighting::{sh9_cached, SkyParams};

impl SkyParams {
    unsafe fn bind(&self, gl: &glow::Context, program: glow::Program) {
        unsafe {
            let set = |name: &str, v: Vec3| {
                if let Some(l) = gl.get_uniform_location(program, name) {
                    gl.uniform_3_f32(Some(&l), v.x, v.y, v.z);
                }
            };
            set("u_sky_zenith", self.zenith);
            set("u_sky_horizon", self.horizon);
            set("u_ground_bounce", self.ground_bounce);

            // The dome as harmonics, which is what every surface reads its ambient
            // from, and the exposure that belongs to this time of day.
            let sh = sh9_cached(self);
            let mut flat = [0.0f32; 27];
            for (i, c) in sh.iter().enumerate() {
                flat[i * 3..i * 3 + 3].copy_from_slice(&c.to_array());
            }
            if let Some(l) = gl.get_uniform_location(program, "u_sh[0]") {
                gl.uniform_3_f32_slice(Some(&l), &flat);
            }
            if let Some(l) = gl.get_uniform_location(program, "u_exposure") {
                gl.uniform_1_f32(Some(&l), self.exposure());
            }
        }
    }
}

/// Draws the sky as one full-screen triangle behind everything else.
pub struct GpuSky {
    program: glow::Program,
    vao: glow::VertexArray,
    u_inv_view_proj: glow::UniformLocation,
    u_cam_pos: glow::UniformLocation,
    u_sun_dir: glow::UniformLocation,
    u_sun_color: glow::UniformLocation,
}

impl GpuSky {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, shaders::SKY_VS, &shaders::sky_fs(), &["a_pos"]);
            let vao = gl.create_vertex_array().expect("sky vao");
            let vbo = gl.create_buffer().expect("sky vbo");
            // One oversized triangle covers the screen with no seam down the middle.
            let verts: [f32; 9] = [-1.0, -1.0, 0.0, 3.0, -1.0, 0.0, -1.0, 3.0, 0.0];
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_slice(&verts), glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            Self {
                u_inv_view_proj: loc(gl, &program, "u_inv_view_proj"),
                u_cam_pos: loc(gl, &program, "u_cam_pos"),
                u_sun_dir: loc(gl, &program, "u_sun_dir"),
                u_sun_color: loc(gl, &program, "u_sun_color"),
                program,
                vao,
            }
        }
    }

    pub fn draw(&self, gl: &glow::Context, view_proj: Mat4, cam_pos: Vec3, sky: &SkyParams) {
        unsafe {
            // Behind everything, and it writes no depth of its own.
            gl.depth_mask(false);
            gl.disable(glow::DEPTH_TEST);
            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(
                Some(&self.u_inv_view_proj),
                false,
                &view_proj.inverse().to_cols_array(),
            );
            gl.uniform_3_f32(Some(&self.u_cam_pos), cam_pos.x, cam_pos.y, cam_pos.z);
            gl.uniform_3_f32(
                Some(&self.u_sun_dir),
                sky.sun_dir.x,
                sky.sun_dir.y,
                sky.sun_dir.z,
            );
            gl.uniform_3_f32(
                Some(&self.u_sun_color),
                sky.sun_color.x,
                sky.sun_color.y,
                sky.sun_color.z,
            );
            sky.bind(gl, self.program);
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.depth_mask(true);
            gl.enable(glow::DEPTH_TEST);
        }
    }

}

pub struct GroundDrawParams<'a> {
    pub view_proj: Mat4,
    pub light_view_proj: Mat4,
    pub cam_pos: Vec3,
    pub sky: &'a SkyParams,
    pub shadow_depth: glow::Texture,
    pub albedo: Vec3,
    pub normal_bias: f32,
    pub extent: f32,
}

/// A ground quad that receives the tree shadow.
pub struct GpuGround {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_view_proj: glow::UniformLocation,
    u_light_view_proj: glow::UniformLocation,
    u_cam_pos: glow::UniformLocation,
    u_sun_dir: glow::UniformLocation,
    u_sun_color: glow::UniformLocation,
    u_albedo_color: glow::UniformLocation,
    u_normal_bias: glow::UniformLocation,
    u_fade_start: glow::UniformLocation,
    u_fade_end: glow::UniformLocation,
}

impl GpuGround {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = compile_program(gl, shaders::GROUND_VS, &shaders::ground_fs(), &["a_pos"]);
            let vao = gl.create_vertex_array().expect("ground vao");
            let vbo = gl.create_buffer().expect("ground vbo");
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(Some(program));
            gl.uniform_1_i32(Some(&loc(gl, &program, "u_shadow_tex")), 3);
            gl.use_program(None);
            Self {
                u_view_proj: loc(gl, &program, "u_view_proj"),
                u_light_view_proj: loc(gl, &program, "u_light_view_proj"),
                u_cam_pos: loc(gl, &program, "u_cam_pos"),
                u_sun_dir: loc(gl, &program, "u_sun_dir"),
                u_sun_color: loc(gl, &program, "u_sun_color"),
                u_albedo_color: loc(gl, &program, "u_albedo_color"),
                u_normal_bias: loc(gl, &program, "u_normal_bias"),
                u_fade_start: loc(gl, &program, "u_fade_start"),
                u_fade_end: loc(gl, &program, "u_fade_end"),
                program,
                vao,
                vbo,
            }
        }
    }

    /// The quad is rebuilt around the camera so it always reaches the horizon.
    fn upload(&self, gl: &glow::Context, center: Vec3, extent: f32) {
        unsafe {
            let (x, z, e) = (center.x, center.z, extent);
            let verts: [f32; 18] = [
                x - e,
                0.0,
                z - e,
                x + e,
                0.0,
                z - e,
                x + e,
                0.0,
                z + e,
                x - e,
                0.0,
                z - e,
                x + e,
                0.0,
                z + e,
                x - e,
                0.0,
                z + e,
            ];
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, cast_slice(&verts), glow::DYNAMIC_DRAW);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }

    pub fn draw(&self, gl: &glow::Context, p: &GroundDrawParams) {
        unsafe {
            self.upload(gl, p.cam_pos, p.extent);
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
            gl.uniform_3_f32(
                Some(&self.u_sun_dir),
                p.sky.sun_dir.x,
                p.sky.sun_dir.y,
                p.sky.sun_dir.z,
            );
            gl.uniform_3_f32(
                Some(&self.u_sun_color),
                p.sky.sun_color.x,
                p.sky.sun_color.y,
                p.sky.sun_color.z,
            );
            gl.uniform_3_f32(
                Some(&self.u_albedo_color),
                p.albedo.x,
                p.albedo.y,
                p.albedo.z,
            );
            gl.uniform_1_f32(Some(&self.u_normal_bias), p.normal_bias);
            gl.uniform_1_f32(Some(&self.u_fade_start), p.extent * 0.10);
            gl.uniform_1_f32(Some(&self.u_fade_end), p.extent * 0.62);
            p.sky.bind(gl, self.program);
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(p.shadow_depth));
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
            gl.bind_vertex_array(None);
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(None);
        }
    }

}
