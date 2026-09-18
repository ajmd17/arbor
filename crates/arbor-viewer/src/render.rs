//! The frame around the scene: the targets it is drawn into, the passes before it
//! (the camera's depth, ambient occlusion) and after it (resolve, bloom, tonemapping),
//! and the lighting every lit shader is handed.
//!
//! The scene is drawn into a multisampled half-float target rather than straight to the
//! window, so light can be as bright as it really is until the very last pass, which
//! is the only place it is fitted to a display. That is what lets the sun blaze, a leaf
//! lit from behind glow, and a shadow keep the colour of the sky it is lit by, instead
//! of everything above white being clipped and everything below a hand-tuned floor.

use eframe::glow::{self, HasContext};
use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::gpu::{compile_program, GpuLeaves, LeafMaterialParams, MaterialTextures, WindUniforms};
use crate::ibl::{self, EnvironmentMaps};
use crate::lighting::{ShadowFrustum, SkyParams};
use crate::shaders;

/// Where every lit program reads each of its textures. Set once per program, since a
/// sampler left unset reads unit 0, and two samplers of different kinds on one unit
/// fail every draw the program makes.
const SAMPLER_UNITS: [(&str, u32); 10] = [
    ("u_albedo_tex", 0),
    ("u_normal_tex", 1),
    ("u_rough_tex", 2),
    ("u_shadow_tex", UNIT_SHADOW),
    ("u_shadow_cmp", UNIT_SHADOW_CMP),
    ("u_env_specular", UNIT_ENV),
    ("u_dfg", UNIT_DFG),
    ("u_ao_tex", UNIT_AO),
    ("u_equirect", UNIT_EQUIRECT),
    ("u_canopy", UNIT_CANOPY),
];
const UNIT_SHADOW: u32 = 3;
const UNIT_SHADOW_CMP: u32 = 4;
const UNIT_ENV: u32 = 5;
const UNIT_DFG: u32 = 6;
const UNIT_AO: u32 = 7;
const UNIT_EQUIRECT: u32 = 8;
const UNIT_CANOPY: u32 = 9;

/// The crown's coverage from above is drawn at this size, and blurred at a quarter of it.
const CANOPY_SIZE: i32 = 256;
const CANOPY_BLUR_LOD: i32 = 2;
/// How much of the sky a fully closed crown takes from beneath it. Not all: light gets
/// in under the edge of the crown from the sky low down, which looking straight up
/// does not see.
const CANOPY_STRENGTH: f32 = 0.85;

/// Points each sampler `program` declares at its unit.
pub unsafe fn assign_units(gl: &glow::Context, program: glow::Program) {
    unsafe {
        gl.use_program(Some(program));
        for (name, unit) in SAMPLER_UNITS {
            if let Some(l) = gl.get_uniform_location(program, name) {
                gl.uniform_1_i32(Some(&l), unit as i32);
            }
        }
        gl.use_program(None);
    }
}

/// How scene radiance is fitted to the display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tonemap {
    /// Blender's default since 4.0: soft highlights that run toward white.
    Agx,
    /// AgX with Blender's Punchy look: more contrast and colour.
    AgxPunchy,
    /// Khronos PBR Neutral: base colours come out as they went in, for judging textures.
    Neutral,
    /// The fitted ACES curve the viewer used before.
    Aces,
}

impl Tonemap {
    pub const ALL: [Self; 4] = [Self::Agx, Self::AgxPunchy, Self::Neutral, Self::Aces];

    pub fn label(self) -> &'static str {
        match self {
            Self::Agx => "AgX",
            Self::AgxPunchy => "AgX punchy",
            Self::Neutral => "PBR neutral",
            Self::Aces => "ACES",
        }
    }

    // The command line's, and the web has none.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "agx" => Some(Self::Agx),
            "punchy" | "agx-punchy" => Some(Self::AgxPunchy),
            "neutral" | "pbr-neutral" => Some(Self::Neutral),
            "aces" => Some(Self::Aces),
            _ => None,
        }
    }

    fn index(self) -> i32 {
        match self {
            Self::Agx => 0,
            Self::AgxPunchy => 1,
            Self::Neutral => 2,
            Self::Aces => 3,
        }
    }
}

/// What the passes around the scene do.
#[derive(Clone, Copy, Debug)]
pub struct PostSettings {
    pub tonemap: Tonemap,
    /// Share of the image the bloom takes, 0 for none.
    pub bloom: f32,
    pub ao: bool,
    /// Metres the ambient occlusion looks out to.
    pub ao_radius: f32,
    /// Exponent on the occlusion: above 1 deepens it.
    pub ao_power: f32,
    /// Samples per pixel, 0 or 1 for none.
    pub samples: i32,
}

/// Everything a lit shader is told about the light, the camera and the shadow.
#[derive(Clone, Copy)]
pub struct Lighting {
    /// The procedural sky, which is the environment when there is no photograph and
    /// is drawn from its dome colours.
    pub sky: SkyParams,
    /// Whether the environment is a photograph.
    pub photo: bool,
    /// The environment's irradiance, sun excluded, in its own frame.
    pub sh: [Vec3; 9],
    /// Radians the photograph is turned about the vertical.
    pub env_rotation: f32,
    pub env_intensity: f32,
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    /// Angular radius of the sun disc, in radians.
    pub sun_radius: f32,
    /// How far the shadows soften against what a sun of that size gives.
    pub shadow_softness: f32,
    pub exposure: f32,
    pub cam_pos: Vec3,
    /// Height of the photograph's viewpoint above the ground, and the radius of the
    /// dome it is laid onto, when its ground is laid under the tree.
    pub ground_projection: Option<(f32, f32)>,
    pub background_blur: f32,
    pub shadow: ShadowFrustum,
    pub shadow_size: i32,
    pub normal_bias: f32,
    pub ao_texel: [f32; 2],
    /// Where the canopy map lies and how much it counts.
    pub canopy: CanopyPlacement,
}

/// Where the canopy map was drawn: its centre on the ground, one over its half-width
/// and how much it counts, then the bottom and top of the crown.
#[derive(Clone, Copy, Debug, Default)]
pub struct CanopyPlacement {
    pub map: [f32; 4],
    pub heights: [f32; 2],
}

impl Lighting {
    /// Sets whichever of these uniforms `program` kept; the compiler drops the rest.
    pub unsafe fn bind(&self, gl: &glow::Context, program: glow::Program) {
        unsafe {
            let loc = |name: &str| gl.get_uniform_location(program, name);
            let f1 = |name: &str, v: f32| {
                if let Some(l) = loc(name) {
                    gl.uniform_1_f32(Some(&l), v);
                }
            };
            let v3 = |name: &str, v: Vec3| {
                if let Some(l) = loc(name) {
                    gl.uniform_3_f32(Some(&l), v.x, v.y, v.z);
                }
            };
            self.sky.bind_dome(gl, program);
            v3("u_sun_dir", self.sun_dir);
            v3("u_sun_color", self.sun_color);
            f1("u_sun_radius", self.sun_radius);
            v3("u_cam_pos", self.cam_pos);
            if let Some(l) = loc("u_env_mode") {
                gl.uniform_1_i32(Some(&l), i32::from(self.photo));
            }
            if let Some(l) = loc("u_env_rot") {
                let (s, c) = self.env_rotation.sin_cos();
                let m = Mat3::from_cols(Vec3::new(c, 0.0, s), Vec3::Y, Vec3::new(-s, 0.0, c));
                gl.uniform_matrix_3_f32_slice(Some(&l), false, &m.to_cols_array());
            }
            f1("u_env_intensity", self.env_intensity);
            if let Some(l) = loc("u_sh[0]") {
                let flat: Vec<f32> = self.sh.iter().flat_map(|c| c.to_array()).collect();
                gl.uniform_3_f32_slice(Some(&l), &flat);
            }
            f1("u_env_max_lod", ibl::MAX_LOD);
            f1("u_exposure", self.exposure);
            if let Some(l) = loc("u_ground_proj") {
                let (height, radius, on) = self.ground_projection.map_or((0.0, 1.0, 0.0), |(h, r)| {
                    // The camera has to stay inside the dome, or it looks at the
                    // photograph from the outside. A camera backed off far enough to
                    // frame a tall tree gets the whole dome scaled up round it, height
                    // and all, so a room keeps its shape rather than its floor being
                    // stretched out to the new wall.
                    let r = r.max(h * 1.5);
                    let from_centre = (self.cam_pos - Vec3::new(0.0, h, 0.0)).length();
                    let k = (from_centre * 1.25 / r).max(1.0);
                    (h * k, r * k, 1.0)
                });
                gl.uniform_3_f32(Some(&l), height, on, radius);
            }
            f1("u_background_blur", self.background_blur);
            if let Some(l) = loc("u_shadow_params") {
                let s = &self.shadow;
                gl.uniform_4_f32(
                    Some(&l),
                    1.0 / self.shadow_size.max(1) as f32,
                    s.depth_range,
                    s.texel,
                    (self.sun_radius * self.shadow_softness).tan(),
                );
            }
            if let Some(l) = loc("u_light_view_proj") {
                gl.uniform_matrix_4_f32_slice(Some(&l), false, &self.shadow.view_proj.to_cols_array());
            }
            f1("u_normal_bias", self.normal_bias);
            if let Some(l) = loc("u_ao_texel") {
                gl.uniform_2_f32(Some(&l), self.ao_texel[0], self.ao_texel[1]);
            }
            if let Some(l) = loc("u_canopy_map") {
                let m = self.canopy.map;
                gl.uniform_4_f32(Some(&l), m[0], m[1], m[2], m[3]);
            }
            if let Some(l) = loc("u_canopy_heights") {
                let h = self.canopy.heights;
                gl.uniform_2_f32(Some(&l), h[0], h[1]);
            }
        }
    }
}

struct Level {
    fbo: glow::Framebuffer,
    tex: glow::Texture,
    width: i32,
    height: i32,
}

/// Everything sized to the viewport.
struct Targets {
    width: i32,
    height: i32,
    samples: i32,
    /// The scene, multisampled.
    scene_fbo: glow::Framebuffer,
    scene_color: glow::Renderbuffer,
    scene_depth: glow::Renderbuffer,
    /// The scene resolved to one sample, for the passes after it to read.
    resolve_fbo: glow::Framebuffer,
    hdr: glow::Texture,
    /// The camera's depth, laid down before the scene for the occlusion to read.
    prepass_fbo: glow::Framebuffer,
    prepass_depth: glow::Texture,
    ao_fbo: [glow::Framebuffer; 2],
    ao: [glow::Texture; 2],
    bloom: Vec<Level>,
}

unsafe fn texture_2d(
    gl: &glow::Context,
    width: i32,
    height: i32,
    internal: u32,
    format: u32,
    ty: u32,
    filter: u32,
) -> glow::Texture {
    unsafe {
        let tex = gl.create_texture().expect("target texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            internal as i32,
            width,
            height,
            0,
            format,
            ty,
            glow::PixelUnpackData::Slice(None),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);
        tex
    }
}

unsafe fn color_fbo(gl: &glow::Context, tex: glow::Texture) -> glow::Framebuffer {
    unsafe {
        let fbo = gl.create_framebuffer().expect("target fbo");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(tex), 0);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        fbo
    }
}

impl Targets {
    /// `None` when this GL cannot make a scene target of that format and sample count.
    unsafe fn new(gl: &glow::Context, width: i32, height: i32, samples: i32, float: bool) -> Option<Self> {
        unsafe {
            let (color_format, color_type) = if float {
                (glow::RGBA16F, glow::FLOAT)
            } else {
                (glow::RGBA8, glow::UNSIGNED_BYTE)
            };
            let scene_fbo = gl.create_framebuffer().ok()?;
            let scene_color = gl.create_renderbuffer().ok()?;
            let scene_depth = gl.create_renderbuffer().ok()?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(scene_color));
            gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, samples, color_format, width, height);
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(scene_depth));
            gl.renderbuffer_storage_multisample(
                glow::RENDERBUFFER,
                samples,
                glow::DEPTH_COMPONENT24,
                width,
                height,
            );
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scene_fbo));
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(scene_color),
            );
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::RENDERBUFFER,
                Some(scene_depth),
            );
            let complete = gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            if !complete {
                gl.delete_framebuffer(scene_fbo);
                gl.delete_renderbuffer(scene_color);
                gl.delete_renderbuffer(scene_depth);
                return None;
            }

            let hdr = texture_2d(gl, width, height, color_format, glow::RGBA, color_type, glow::LINEAR);
            let resolve_fbo = color_fbo(gl, hdr);

            let prepass_depth = texture_2d(
                gl,
                width,
                height,
                glow::DEPTH_COMPONENT24,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                glow::NEAREST,
            );
            let prepass_fbo = gl.create_framebuffer().ok()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(prepass_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::TEXTURE_2D,
                Some(prepass_depth),
                0,
            );
            gl.draw_buffers(&[glow::NONE]);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            let ao = [0, 1].map(|_| texture_2d(gl, width, height, glow::R8, glow::RED, glow::UNSIGNED_BYTE, glow::NEAREST));
            let ao_fbo = ao.map(|t| color_fbo(gl, t));

            // Half size down to a few texels, the way the bloom spreads.
            let mut bloom = Vec::new();
            let (mut w, mut h) = (width / 2, height / 2);
            while bloom.len() < 6 && w >= 4 && h >= 4 {
                let tex = texture_2d(gl, w, h, color_format, glow::RGBA, color_type, glow::LINEAR);
                bloom.push(Level { fbo: color_fbo(gl, tex), tex, width: w, height: h });
                w /= 2;
                h /= 2;
            }

            Some(Self {
                width,
                height,
                samples,
                scene_fbo,
                scene_color,
                scene_depth,
                resolve_fbo,
                hdr,
                prepass_fbo,
                prepass_depth,
                ao_fbo,
                ao,
                bloom,
            })
        }
    }

    unsafe fn delete(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_framebuffer(self.scene_fbo);
            gl.delete_renderbuffer(self.scene_color);
            gl.delete_renderbuffer(self.scene_depth);
            gl.delete_framebuffer(self.resolve_fbo);
            gl.delete_texture(self.hdr);
            gl.delete_framebuffer(self.prepass_fbo);
            gl.delete_texture(self.prepass_depth);
            for i in 0..2 {
                gl.delete_framebuffer(self.ao_fbo[i]);
                gl.delete_texture(self.ao[i]);
            }
            for level in &self.bloom {
                gl.delete_framebuffer(level.fbo);
                gl.delete_texture(level.tex);
            }
        }
    }
}

/// The crown seen from above, for the sky it takes from what stands beneath it.
struct Canopy {
    coverage: glow::Texture,
    coverage_fbo: glow::Framebuffer,
    blur: [glow::Texture; 2],
    blur_fbo: [glow::Framebuffer; 2],
    draw: glow::Program,
    blur_program: glow::Program,
}

impl Canopy {
    unsafe fn new(gl: &glow::Context) -> Self {
        unsafe {
            let coverage = texture_2d(
                gl,
                CANOPY_SIZE,
                CANOPY_SIZE,
                glow::RGBA8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::LINEAR,
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(coverage));
            gl.generate_mipmap(glow::TEXTURE_2D);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR_MIPMAP_LINEAR as i32,
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            let small = CANOPY_SIZE >> CANOPY_BLUR_LOD;
            let blur = [0, 1].map(|_| {
                texture_2d(gl, small, small, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, glow::LINEAR)
            });
            let draw = compile_program(
                gl,
                &shaders::leaf_depth_vs(),
                shaders::CANOPY_FS,
                &crate::gpu::LEAF_ATTRIBS,
            );
            gl.use_program(Some(draw));
            gl.uniform_1_i32(gl.get_uniform_location(draw, "u_albedo_tex").as_ref(), 0);
            gl.use_program(None);
            Self {
                coverage_fbo: color_fbo(gl, coverage),
                coverage,
                blur_fbo: blur.map(|t| color_fbo(gl, t)),
                blur,
                draw,
                blur_program: compile_program(gl, shaders::FULLSCREEN_VS, shaders::CANOPY_BLUR_FS, &[]),
            }
        }
    }
}

pub struct Renderer {
    targets: Option<Targets>,
    canopy: Canopy,
    /// Whether half-float targets can be made here. Without them the frame still
    /// draws, clipped at white.
    float: bool,
    max_samples: i32,
    vao: glow::VertexArray,
    gtao: glow::Program,
    ao_blur: glow::Program,
    bloom_down: glow::Program,
    bloom_up: glow::Program,
    composite: glow::Program,
    pub env: EnvironmentMaps,
    dfg: glow::Texture,
    white: glow::Texture,
    shadow_raw: glow::Sampler,
    shadow_cmp: glow::Sampler,
}

impl Renderer {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let fs = |src: &str| compile_program(gl, shaders::FULLSCREEN_VS, src, &[]);
            let white = texture_2d(gl, 1, 1, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, glow::NEAREST);
            gl.bind_texture(glow::TEXTURE_2D, Some(white));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                1,
                1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&[255, 255, 255, 255])),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);

            // Two ways of reading one shadow map: raw depth for finding blockers, and
            // compared in hardware, bilinearly, for filtering. Outside the map is
            // fully lit (see `ShadowTarget`), where the platform can say so.
            let sampler = |compare: bool| {
                let s = gl.create_sampler().expect("shadow sampler");
                let filter = if compare { glow::LINEAR } else { glow::NEAREST } as i32;
                gl.sampler_parameter_i32(s, glow::TEXTURE_MIN_FILTER, filter);
                gl.sampler_parameter_i32(s, glow::TEXTURE_MAG_FILTER, filter);
                #[cfg(not(target_arch = "wasm32"))]
                let wrap = glow::CLAMP_TO_BORDER;
                #[cfg(target_arch = "wasm32")]
                let wrap = glow::CLAMP_TO_EDGE;
                gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_S, wrap as i32);
                gl.sampler_parameter_i32(s, glow::TEXTURE_WRAP_T, wrap as i32);
                #[cfg(not(target_arch = "wasm32"))]
                gl.sampler_parameter_f32_slice(s, glow::TEXTURE_BORDER_COLOR, &[1.0, 1.0, 1.0, 1.0]);
                if compare {
                    gl.sampler_parameter_i32(
                        s,
                        glow::TEXTURE_COMPARE_MODE,
                        glow::COMPARE_REF_TO_TEXTURE as i32,
                    );
                    gl.sampler_parameter_i32(s, glow::TEXTURE_COMPARE_FUNC, glow::LEQUAL as i32);
                }
                s
            };

            let float = cfg!(not(target_arch = "wasm32"))
                || gl.supported_extensions().contains("EXT_color_buffer_float");
            Self {
                targets: None,
                canopy: Canopy::new(gl),
                float,
                max_samples: gl.get_parameter_i32(glow::MAX_SAMPLES).max(0),
                vao: gl.create_vertex_array().expect("post vao"),
                gtao: fs(&shaders::gtao_fs()),
                ao_blur: fs(&shaders::ao_blur_fs()),
                bloom_down: fs(shaders::BLOOM_DOWN_FS),
                bloom_up: fs(shaders::BLOOM_UP_FS),
                composite: fs(&shaders::composite_fs()),
                env: EnvironmentMaps::new(gl),
                dfg: ibl::bake_dfg(gl),
                white,
                shadow_raw: sampler(false),
                shadow_cmp: sampler(true),
            }
        }
    }

    /// Makes sure the targets fit a viewport of this size, rebuilding them if not.
    /// Falls back to fewer samples, then to eight-bit colour, if the GL refuses.
    pub fn prepare(&mut self, gl: &glow::Context, width: i32, height: i32, samples: i32) {
        let samples = samples.clamp(0, self.max_samples);
        let (width, height) = (width.max(1), height.max(1));
        if let Some(t) = &self.targets
            && (t.width, t.height, t.samples) == (width, height, samples)
        {
            return;
        }
        unsafe {
            if let Some(t) = self.targets.take() {
                t.delete(gl);
            }
            let mut s = samples;
            loop {
                if let Some(t) = Targets::new(gl, width, height, s, self.float) {
                    // Remembered at the count asked for, so a refused count is not
                    // retried every frame.
                    self.targets = Some(Targets { samples, ..t });
                    return;
                }
                if s > 0 {
                    s /= 2;
                } else if self.float {
                    eprintln!("half-float targets refused; drawing without HDR");
                    self.float = false;
                    s = samples;
                } else {
                    panic!("no scene target this GL will accept");
                }
            }
        }
    }

    fn targets(&self) -> &Targets {
        self.targets.as_ref().expect("prepare before drawing")
    }

    /// One texel of the occlusion, in uv, for the lit shaders to find theirs by.
    pub fn ao_texel(&self) -> [f32; 2] {
        let t = self.targets();
        [1.0 / t.width as f32, 1.0 / t.height as f32]
    }

    /// Draws the crown from above into the canopy map and blurs it, and says where it
    /// put it. `crown` is the leaves' bounding box.
    pub unsafe fn canopy(
        &self,
        gl: &glow::Context,
        leaves: &GpuLeaves,
        material: &MaterialTextures,
        p: LeafMaterialParams,
        wind: &WindUniforms,
        crown: (Vec3, Vec3),
    ) -> CanopyPlacement {
        let (lo, hi) = crown;
        if !leaves.has_cards() || hi.cmplt(lo).any() {
            return CanopyPlacement::default();
        }
        unsafe {
            let c = &self.canopy;
            // A ground point sees sky all round the vertical, weighted toward it; a
            // crown this high overhead takes a patch of it about this wide.
            let sigma = (0.3 * (lo.y + hi.y) * 0.5).max(0.5);
            let half = ((hi.x - lo.x).max(hi.z - lo.z) * 0.5 + 2.5 * sigma).max(1.0);
            let (cx, cz) = ((lo.x + hi.x) * 0.5, (lo.z + hi.z) * 0.5);
            // Straight down: x across the map, z up it, height ignored.
            let top_down = Mat4::from_cols(
                Vec4::new(1.0 / half, 0.0, 0.0, 0.0),
                Vec4::ZERO,
                Vec4::new(0.0, 1.0 / half, 0.0, 0.0),
                Vec4::new(-cx / half, -cz / half, 0.0, 1.0),
            );

            plain(gl);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(c.coverage_fbo));
            gl.viewport(0, 0, CANOPY_SIZE, CANOPY_SIZE);
            gl.clear_color(1.0, 1.0, 1.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func(glow::ZERO, glow::ONE_MINUS_SRC_ALPHA);
            gl.use_program(Some(c.draw));
            wind.bind(gl, c.draw);
            let u = |name: &str| gl.get_uniform_location(c.draw, name);
            gl.uniform_matrix_4_f32_slice(u("u_light_view_proj").as_ref(), false, &top_down.to_cols_array());
            gl.uniform_2_f32(u("u_atlas_scale").as_ref(), p.atlas_scale[0], p.atlas_scale[1]);
            gl.uniform_2_f32(u("u_atlas_front").as_ref(), p.atlas_front[0], p.atlas_front[1]);
            gl.uniform_2_f32(u("u_atlas_back").as_ref(), p.atlas_back[0], p.atlas_back[1]);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(material.albedo));
            leaves.bind_and_draw(gl);
            gl.disable(glow::BLEND);
            gl.bind_texture(glow::TEXTURE_2D, Some(c.coverage));
            gl.generate_mipmap(glow::TEXTURE_2D);

            // Blurred at a quarter size, across then up.
            let small = CANOPY_SIZE >> CANOPY_BLUR_LOD;
            let texel_m = 2.0 * half / small as f32;
            let b = c.blur_program;
            gl.use_program(Some(b));
            gl.bind_vertex_array(Some(self.vao));
            gl.viewport(0, 0, small, small);
            let u = |name: &str| gl.get_uniform_location(b, name);
            gl.uniform_1_i32(u("u_src").as_ref(), 0);
            gl.uniform_1_f32(u("u_sigma").as_ref(), sigma / texel_m);
            for (pass, (src, lod, step)) in [
                (c.coverage, CANOPY_BLUR_LOD as f32, [1.0 / small as f32, 0.0]),
                (c.blur[0], 0.0, [0.0, 1.0 / small as f32]),
            ]
            .into_iter()
            .enumerate()
            {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(c.blur_fbo[pass]));
                gl.bind_texture(glow::TEXTURE_2D, Some(src));
                gl.uniform_1_f32(u("u_lod").as_ref(), lod);
                gl.uniform_2_f32(u("u_step").as_ref(), step[0], step[1]);
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.use_program(None);

            CanopyPlacement {
                map: [cx, cz, 1.0 / half, CANOPY_STRENGTH],
                heights: [lo.y, hi.y],
            }
        }
    }

    /// Binds the camera depth target, cleared, for the depth pass to draw into.
    pub unsafe fn begin_prepass(&self, gl: &glow::Context) {
        unsafe {
            let t = self.targets();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.prepass_fbo));
            gl.viewport(0, 0, t.width, t.height);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);
        }
    }

    /// Works out the ambient occlusion from the camera's depth, if it is wanted.
    pub unsafe fn ambient_occlusion(&self, gl: &glow::Context, proj: Mat4, post: &PostSettings) {
        if !post.ao {
            return;
        }
        unsafe {
            let t = self.targets();
            plain(gl);
            gl.bind_vertex_array(Some(self.vao));
            gl.viewport(0, 0, t.width, t.height);
            let inv_proj = proj.inverse().to_cols_array();
            let texel = [1.0 / t.width as f32, 1.0 / t.height as f32];

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.ao_fbo[0]));
            let p = self.gtao;
            gl.use_program(Some(p));
            let u = |name: &str| gl.get_uniform_location(p, name);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(t.prepass_depth));
            gl.uniform_1_i32(u("u_depth").as_ref(), 0);
            gl.uniform_matrix_4_f32_slice(u("u_inv_proj").as_ref(), false, &inv_proj);
            gl.uniform_2_f32(u("u_texel").as_ref(), texel[0], texel[1]);
            gl.uniform_1_f32(u("u_radius").as_ref(), post.ao_radius);
            gl.uniform_1_f32(u("u_proj_scale").as_ref(), proj.y_axis.y * t.height as f32 * 0.5);
            gl.uniform_1_f32(u("u_power").as_ref(), post.ao_power);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.ao_fbo[1]));
            let p = self.ao_blur;
            gl.use_program(Some(p));
            let u = |name: &str| gl.get_uniform_location(p, name);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(t.ao[0]));
            gl.uniform_1_i32(u("u_depth").as_ref(), 0);
            gl.uniform_1_i32(u("u_ao").as_ref(), 1);
            gl.uniform_matrix_4_f32_slice(u("u_inv_proj").as_ref(), false, &inv_proj);
            gl.uniform_2_f32(u("u_texel").as_ref(), texel[0], texel[1]);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }

    /// Binds the scene target, cleared, with the state the scene passes expect.
    pub unsafe fn begin_scene(&self, gl: &glow::Context) {
        unsafe {
            let t = self.targets();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.scene_fbo));
            gl.viewport(0, 0, t.width, t.height);
            gl.disable(glow::SCISSOR_TEST);
            gl.color_mask(true, true, true, true);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            gl.disable(glow::BLEND);
        }
    }

    /// Binds the shadow map, the environment, the DFG table and the occlusion where
    /// every lit program reads them.
    pub unsafe fn bind_lighting(&self, gl: &glow::Context, shadow_depth: glow::Texture, ao: bool) {
        unsafe {
            let t = self.targets();
            gl.active_texture(glow::TEXTURE0 + UNIT_SHADOW);
            gl.bind_texture(glow::TEXTURE_2D, Some(shadow_depth));
            gl.bind_sampler(UNIT_SHADOW, Some(self.shadow_raw));
            gl.active_texture(glow::TEXTURE0 + UNIT_SHADOW_CMP);
            gl.bind_texture(glow::TEXTURE_2D, Some(shadow_depth));
            gl.bind_sampler(UNIT_SHADOW_CMP, Some(self.shadow_cmp));
            gl.active_texture(glow::TEXTURE0 + UNIT_ENV);
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(self.env.specular));
            gl.active_texture(glow::TEXTURE0 + UNIT_DFG);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.dfg));
            gl.active_texture(glow::TEXTURE0 + UNIT_AO);
            gl.bind_texture(glow::TEXTURE_2D, Some(if ao { t.ao[1] } else { self.white }));
            gl.active_texture(glow::TEXTURE0 + UNIT_EQUIRECT);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.env.equirect));
            gl.active_texture(glow::TEXTURE0 + UNIT_CANOPY);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.canopy.blur[1]));
            gl.active_texture(glow::TEXTURE0);
        }
    }

    /// Lets go of what `bind_lighting` bound. The sampler objects especially: left on a
    /// unit they would override egui's own textures' filtering there.
    pub unsafe fn unbind_lighting(&self, gl: &glow::Context) {
        unsafe {
            for unit in [UNIT_SHADOW, UNIT_SHADOW_CMP, UNIT_DFG, UNIT_AO, UNIT_EQUIRECT, UNIT_CANOPY] {
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_2D, None);
            }
            gl.active_texture(glow::TEXTURE0 + UNIT_ENV);
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, None);
            gl.bind_sampler(UNIT_SHADOW, None);
            gl.bind_sampler(UNIT_SHADOW_CMP, None);
            gl.active_texture(glow::TEXTURE0);
        }
    }

    /// Resolves the scene, blooms it, and tonemaps it into `clip` of the window.
    /// `raw` passes the debug views through untouched.
    pub unsafe fn finish(&self, gl: &glow::Context, clip: [i32; 4], post: &PostSettings, raw: bool) {
        unsafe {
            let t = self.targets();
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(t.scene_fbo));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(t.resolve_fbo));
            gl.blit_framebuffer(
                0,
                0,
                t.width,
                t.height,
                0,
                0,
                t.width,
                t.height,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            plain(gl);
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            let bloom = post.bloom > 0.0 && !raw && !t.bloom.is_empty();
            if bloom {
                let p = self.bloom_down;
                gl.use_program(Some(p));
                gl.uniform_1_i32(gl.get_uniform_location(p, "u_src").as_ref(), 0);
                let texel_loc = gl.get_uniform_location(p, "u_src_texel");
                let first_loc = gl.get_uniform_location(p, "u_first");
                let (mut src, mut sw, mut sh) = (t.hdr, t.width, t.height);
                for (i, level) in t.bloom.iter().enumerate() {
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(level.fbo));
                    gl.viewport(0, 0, level.width, level.height);
                    gl.bind_texture(glow::TEXTURE_2D, Some(src));
                    gl.uniform_2_f32(texel_loc.as_ref(), 1.0 / sw as f32, 1.0 / sh as f32);
                    gl.uniform_1_i32(first_loc.as_ref(), i32::from(i == 0));
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                    (src, sw, sh) = (level.tex, level.width, level.height);
                }
                let p = self.bloom_up;
                gl.use_program(Some(p));
                gl.uniform_1_i32(gl.get_uniform_location(p, "u_src").as_ref(), 0);
                let texel_loc = gl.get_uniform_location(p, "u_src_texel");
                gl.enable(glow::BLEND);
                gl.blend_equation(glow::FUNC_ADD);
                gl.blend_func(glow::ONE, glow::ONE);
                for i in (0..t.bloom.len() - 1).rev() {
                    let (dst, src) = (&t.bloom[i], &t.bloom[i + 1]);
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(dst.fbo));
                    gl.viewport(0, 0, dst.width, dst.height);
                    gl.bind_texture(glow::TEXTURE_2D, Some(src.tex));
                    gl.uniform_2_f32(texel_loc.as_ref(), 1.0 / src.width as f32, 1.0 / src.height as f32);
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                }
                gl.disable(glow::BLEND);
            }

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(clip[0], clip[1], clip[2], clip[3]);
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(clip[0], clip[1], clip[2], clip[3]);
            let p = self.composite;
            gl.use_program(Some(p));
            let u = |name: &str| gl.get_uniform_location(p, name);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(t.hdr));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(if bloom { t.bloom[0].tex } else { t.hdr }));
            gl.uniform_1_i32(u("u_hdr").as_ref(), 0);
            gl.uniform_1_i32(u("u_bloom").as_ref(), 1);
            gl.uniform_1_f32(u("u_bloom_strength").as_ref(), if bloom { post.bloom } else { 0.0 });
            gl.uniform_1_i32(u("u_tonemap").as_ref(), post.tonemap.index());
            gl.uniform_1_i32(u("u_raw").as_ref(), i32::from(raw));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

/// Nothing but the triangle a full-screen pass draws.
unsafe fn plain(gl: &glow::Context) {
    unsafe {
        gl.disable(glow::SCISSOR_TEST);
        gl.disable(glow::DEPTH_TEST);
        gl.disable(glow::BLEND);
        gl.disable(glow::CULL_FACE);
        gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
        gl.color_mask(true, true, true, true);
    }
}
