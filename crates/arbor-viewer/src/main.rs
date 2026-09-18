#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod assets;
mod export;
mod gpu;
mod hdri;
mod ibl;
mod knobs;
mod lighting;
mod mipmap;
mod presets;
mod render;
mod shaders;

use std::path::Path;
use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::glow;
use eframe::glow::HasContext;
use glam::{Mat4, Vec3};

use arbor_core::species::{LeafClusterParams, CUSTOM_PRESET_DIR};
use arbor_core::textures;
use arbor_core::{
    build_leaves, build_mesh, grow, Skeleton, SkeletonStats, SpeciesParams, SpeciesTemplate,
};

use arbor_core::textures::MapSource as _;
use gpu::{
    ColorPass, DepthPass, GpuBackdrop, GpuGround, GpuLeaves, GpuLines, GpuMesh, GroundDrawParams,
    GroundMode, LeafDepthPass, LeafDrawParams, LeafMaterialParams, MaterialTextures,
    MeshDrawParams, ShadowTarget, WindUniforms,
};
use lighting::{sh9_cached, shadow_frustum, SkyParams};
use render::{Lighting, PostSettings, Renderer, Tonemap};

const TEXTURE_DIR: &str = arbor_core::textures::TEXTURE_DIR;

#[derive(Clone, Copy, PartialEq)]
enum RenderMode {
    Shaded,
    UvChecker,
    Normals,
    /// The ambient occlusion the scene is being drawn with, alone.
    Occlusion,
}

/// Asks the viewer to save what it drew and quit, so a capture comes from the real
/// renderer rather than from anything that merely imitates it.
#[derive(Clone)]
struct Capture {
    path: String,
    /// Frames to let pass first: the tree is rebuilt on the first update and the
    /// camera frames itself a frame later.
    settle: u32,
}

/// Overrides a capture run can set, so a frame can be taken with one thing changed
/// and the difference attributed to it.
#[derive(Default)]
struct Startup {
    capture: Option<Capture>,
    species: Option<String>,
    seed: Option<u64>,
    leaves: Option<bool>,
    shadows: Option<bool>,
    translucency: Option<f32>,
    time_of_day: Option<f32>,
    wireframe: bool,
    sun_elevation: Option<f32>,
    sun_azimuth: Option<f32>,
    sun_intensity: Option<f32>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    distance: Option<f32>,
    target_y: Option<f32>,
    coverage_lod: Option<f32>,
    wind: Option<f32>,
    gustiness: Option<f32>,
    wind_direction: Option<f32>,
    wind_time: Option<f32>,
    view: Option<RenderMode>,
    /// `Some(None)` asks for the procedural sky, `Some(Some(name))` for a photograph.
    environment: Option<Option<String>>,
    env_rotation: Option<f32>,
    env_intensity: Option<f32>,
    exposure: Option<f32>,
    tonemap: Option<Tonemap>,
    bloom: Option<f32>,
    ao: Option<bool>,
    ao_radius: Option<f32>,
    background_blur: Option<f32>,
    ground: Option<Ground>,
    msaa: Option<i32>,
    sun_size: Option<f32>,
    shadow_softness: Option<f32>,
}

/// What stands under the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ground {
    /// A plain ground of its own.
    Plain,
    /// The ground in the photograph, when there is one; the plain ground otherwise.
    Photo,
    None,
}

impl Ground {
    // The command line's, and the web has none.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn parse(s: &str) -> Option<Self> {
        match s {
            "plain" => Some(Self::Plain),
            "photo" | "projected" => Some(Self::Photo),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

/// What the environment on the GPU was last built from, so it is rebuilt only when
/// that changes.
#[derive(Clone, PartialEq)]
enum Built {
    Nothing,
    Sky(SkyParams),
    Photo(String),
}

/// What was learned from a photograph when it was loaded.
struct Photo {
    sun: Option<hdri::Sun>,
    sh: [Vec3; 9],
}

/// The hour the viewer opens at: the low, warm light of just after sunrise.
const DEFAULT_HOUR: f32 = 6.32;

/// The wind the viewer opens in: a steady breeze with some gust in it, blowing across
/// the default view so the sway reads side-on rather than toward the camera.
const DEFAULT_WIND: f32 = 0.3;
const DEFAULT_GUSTINESS: f32 = 0.5;
const DEFAULT_WIND_DIRECTION: f32 = 150.0;

/// Elevation and azimuth of the sun at a given hour, on a day roughly like a temperate
/// equinox: up a little after six, down a little before eight, and swinging through
/// south at noon. Enough of an arc to light a tree by; not an ephemeris.
fn sun_at_hour(hour: f32) -> (f32, f32) {
    const SUNRISE: f32 = 6.0;
    const SUNSET: f32 = 20.0;
    const NOON_ELEVATION: f32 = 62.0;
    let t = ((hour - SUNRISE) / (SUNSET - SUNRISE)).clamp(-0.2, 1.2);
    // A sine arc puts the sun low for a long while near each end and high in the
    // middle, which is what makes the interesting light last.
    let elevation = (t * std::f32::consts::PI).sin() * NOON_ELEVATION;
    // East at sunrise, through south, to west at sunset.
    let azimuth = 90.0 + t * 180.0;
    (elevation, azimuth)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |key: &str| args.iter().position(|a| a == key).and_then(|i| args.get(i + 1));
    let num = |key: &str| -> Option<f32> { flag(key).and_then(|s| s.parse().ok()) };
    let startup = Startup {
        capture: flag("--screenshot").map(|path| Capture {
            path: path.clone(),
            settle: flag("--settle").and_then(|s| s.parse().ok()).unwrap_or(8),
        }),
        species: flag("--species").cloned(),
        seed: flag("--seed").and_then(|s| s.parse().ok()),
        leaves: args.iter().any(|a| a == "--no-leaves").then_some(false),
        wireframe: args.iter().any(|a| a == "--wireframe"),
        shadows: args.iter().any(|a| a == "--no-shadows").then_some(false),
        translucency: num("--translucency"),
        time_of_day: num("--time"),
        sun_elevation: num("--sun-elevation"),
        sun_azimuth: num("--sun-azimuth"),
        sun_intensity: num("--sun-intensity"),
        yaw: num("--yaw"),
        pitch: num("--pitch"),
        distance: num("--distance"),
        target_y: num("--target-y"),
        coverage_lod: num("--coverage-lod"),
        wind: num("--wind"),
        gustiness: num("--gustiness"),
        wind_direction: num("--wind-dir"),
        wind_time: num("--wind-time"),
        view: flag("--view").and_then(|v| match v.as_str() {
            "uv" => Some(RenderMode::UvChecker),
            "normals" => Some(RenderMode::Normals),
            "ao" | "occlusion" => Some(RenderMode::Occlusion),
            _ => None,
        }),
        environment: flag("--hdri").map(|h| (h != "sky" && h != "none").then(|| h.clone())),
        env_rotation: num("--env-rotation"),
        env_intensity: num("--env-intensity"),
        exposure: num("--exposure"),
        tonemap: flag("--tonemap").and_then(|t| Tonemap::parse(t)),
        bloom: num("--bloom"),
        ao: args.iter().any(|a| a == "--no-ao").then_some(false),
        ao_radius: num("--ao-radius"),
        background_blur: num("--blur"),
        ground: flag("--ground").and_then(|g| Ground::parse(g)),
        msaa: flag("--msaa").and_then(|s| s.parse().ok()),
        sun_size: num("--sun-size"),
        shadow_softness: num("--softness"),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 950.0])
            .with_title("Arbor"),
        // The scene is drawn into targets of its own, multisampled there, so the window
        // itself needs neither depth nor samples.
        depth_buffer: 0,
        multisampling: 0,
        ..Default::default()
    };
    eframe::run_native(
        "arbor",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, startup)))),
    )
}

/// In a page, the viewer takes over the canvas called `arbor`.
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;
    eframe::WebLogger::init(log::LevelFilter::Warn).ok();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().and_then(|w| w.document()).expect("a page to run in");
        let canvas = document
            .get_element_by_id("arbor")
            .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("a <canvas id=\"arbor\"> on the page");
        let started = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(App::new(cc, Startup::default())))),
            )
            .await;
        let loading = document.get_element_by_id("loading");
        match started {
            Ok(()) => {
                if let Some(loading) = loading {
                    loading.remove();
                }
            }
            Err(e) => {
                log::error!("the viewer could not start: {e:?}");
                if let Some(loading) = loading {
                    loading.set_text_content(Some("The viewer could not start: this browser may not have WebGL2."));
                }
            }
        }
    });
}

/// A seed nobody picked: the clock, stirred so that two clicks a moment apart land far
/// apart.
fn random_seed() -> u64 {
    let now = web_time::SystemTime::now()
        .duration_since(web_time::SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut z = now.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What the save field offers for a preset once it is loaded: a saved one's own name,
/// so saving again replaces it, and a variant name for a built-in, which cannot be
/// saved over.
fn save_name_for(preset: &presets::Preset, params: &SpeciesTemplate) -> String {
    if !preset.is_saved() {
        return format!("{}_custom", preset.name);
    }
    // The name as it was typed, when it still leads back to this file.
    match presets::key_for(&params.name) {
        Ok(key) if key == preset.name => params.name.clone(),
        _ => preset.name.clone(),
    }
}

/// Writes an egui screenshot out as a PNG.
fn save_screenshot(image: &egui::ColorImage, path: &str) {
    let [w, h] = image.size;
    let mut out = image::RgbaImage::new(w as u32, h as u32);
    for (i, px) in image.pixels.iter().enumerate() {
        out.put_pixel(
            (i % w) as u32,
            (i / w) as u32,
            image::Rgba([px.r(), px.g(), px.b(), 255]),
        );
    }
    match out.save(path) {
        Ok(()) => println!("wrote {path} ({w}x{h})"),
        Err(e) => eprintln!("screenshot failed: {e}"),
    }
}

#[derive(Clone, Copy)]
struct OrbitCamera {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
    aspect: f32,
}

impl OrbitCamera {
    fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let dir = Vec3::new(
            self.yaw.sin() * cp,
            self.pitch.sin(),
            self.yaw.cos() * cp,
        );
        self.target + dir * self.distance
    }

    fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or(Vec3::Y)
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or(Vec3::X)
    }

    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or(Vec3::Y)
    }

    fn proj(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect.max(0.05), 0.05, 600.0)
    }

    fn view_proj(&self) -> Mat4 {
        self.proj() * Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }
}

/// Reflectance of the plain ground: leaf litter and bare soil.
const GROUND_ALBEDO: Vec3 = Vec3::new(0.14, 0.12, 0.085);

const LEVEL_COLORS: [[f32; 3]; 8] = [
    [0.95, 0.75, 0.35],
    [0.45, 0.95, 0.40],
    [0.35, 0.75, 0.98],
    [0.90, 0.45, 0.95],
    [0.95, 0.55, 0.30],
    [0.70, 0.80, 0.42],
    [0.72, 0.82, 0.45],
    [0.75, 0.85, 0.50],
];

struct App {
    gl: Arc<glow::Context>,
    lines: Arc<Mutex<GpuLines>>,
    mesh_gpu: Arc<Mutex<GpuMesh>>,
    leaves_gpu: Arc<Mutex<GpuLeaves>>,
    shadow: Arc<ShadowTarget>,
    depth_pass: Arc<DepthPass>,
    leaf_depth_pass: Arc<LeafDepthPass>,
    color_pass: Arc<ColorPass>,
    backdrop: Arc<GpuBackdrop>,
    ground_pass: Arc<GpuGround>,
    /// The targets and the passes around the scene, and the environment maps.
    renderer: Arc<Mutex<Renderer>>,
    /// Where photographed environments come from.
    hdri_maps: assets::Maps,
    /// The photographs on offer.
    hdris: Vec<String>,
    /// The photograph asked for, or `None` for the procedural sky.
    environment: Option<String>,
    built: Built,
    photo: Option<Photo>,
    /// Why the photograph asked for is not on screen, when it is not.
    env_status: Option<(String, bool)>,
    /// Degrees the photograph is turned about the vertical.
    env_rotation: f32,
    env_intensity: f32,
    /// How bright the sun lifted out of a photograph is put back, against how bright it was.
    photo_sun: f32,
    background_blur: f32,
    ground: Ground,
    /// Metres above the ground the photograph is taken to have been shot from, and the
    /// radius of the dome it is laid onto.
    ground_height: f32,
    ground_radius: f32,
    /// Angular diameter of the procedural sun, in degrees.
    sun_size: f32,
    shadow_softness: f32,
    /// Stops of exposure on top of the automatic.
    exposure_ev: f32,
    tonemap: Tonemap,
    bloom: f32,
    ao: bool,
    ao_radius: f32,
    ao_strength: f32,
    msaa: i32,
    /// Where the texture maps come from: off disk, or fetched on the web.
    maps: assets::Maps,
    bark_material: MaterialTextures,
    leaf_material: Option<MaterialTextures>,
    loaded_bark: String,
    /// Texture name and cluster arrangement the leaf material was built from.
    /// Both go in the key, because changing either has to rebuild the atlas.
    loaded_leaf: (String, Option<LeafClusterParams>),
    presets: Vec<presets::Preset>,
    /// The preset the panel was last loaded from or saved to. None once that preset
    /// has been deleted: the tree stays on screen with nothing behind it until saved.
    preset: Option<usize>,
    /// Name typed for the next save.
    save_name: String,
    /// Outcome of the last load, save, delete or export, and whether it was a failure.
    status: export::Status,
    /// Delete has been clicked once and is waiting to be confirmed.
    confirm_delete: bool,
    exports: export::Exports,
    /// The species the panel edits and saves, ranges and all.
    params: SpeciesTemplate,
    /// What the tree on screen was grown from: `params` with every range landed where
    /// the seed puts it. Everything drawn reads this rather than `params`.
    grown: SpeciesParams,
    skeleton: Skeleton,
    stats: SkeletonStats,
    mesh_stats: (usize, usize),
    leaf_stats: (usize, usize),
    aabb: ([f32; 3], [f32; 3]),
    /// The leaves' bounding box: where the crown is, for the shade it casts on the sky.
    crown: (Vec3, Vec3),
    camera: OrbitCamera,
    dirty: bool,
    gen_ms: f32,
    render_mode: RenderMode,
    wireframe: bool,
    shadows: bool,
    show_skeleton: bool,
    sun_intensity: f32,
    show_leaves: bool,
    leaf_translucency: f32,
    show_grid: bool,
    use_normal_map: bool,
    sun_azimuth: f32,
    sun_elevation: f32,
    /// Hours, as the single control that moves the sun along its arc.
    time_of_day: f32,
    auto_frame: bool,
    capture: Option<Capture>,
    frames: u32,
    /// Mip level where soft leaf coverage gives way to a hard cutoff.
    coverage_lod: f32,
    /// The weather. How the tree gives to it is the species' business and lives in
    /// `params.wind`.
    wind_on: bool,
    wind_strength: f32,
    wind_gustiness: f32,
    /// Bearing the wind blows toward, in degrees, measured like the sun's azimuth.
    wind_direction: f32,
    /// Seconds on the wind's own clock. It only advances while the wind is running, so
    /// pausing holds the pose and resuming carries on from it without a jump.
    wind_clock: f32,
    wind_paused: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, startup: Startup) -> Self {
        let gl = cc.gl.clone().expect("eframe must run with the glow renderer");
        let presets = presets::list(Path::new(CUSTOM_PRESET_DIR));
        let mut status = None;
        let asked = startup
            .species
            .as_deref()
            .and_then(|want| presets.iter().position(|p| p.name == want));
        if let (Some(want), None) = (startup.species.as_deref(), asked) {
            status = Some((format!("no preset called {want}"), true));
        }
        let (preset_index, mut params) = match asked.map(|i| (i, presets[i].load())) {
            Some((i, Ok(p))) => (i, p),
            Some((i, Err(e))) => {
                status = Some((format!("{}: {e}", presets[i].name), true));
                (0, presets[0].load().expect("embedded preset parses"))
            }
            None => (0, presets[0].load().expect("embedded preset parses")),
        };
        let save_name = save_name_for(&presets[preset_index], &params);
        if let Some(seed) = startup.seed {
            params.seed = seed;
        }
        let grown = params.instance();
        let skeleton = grow(&grown);
        let mesh = build_mesh(&skeleton, &grown);
        let aabb = mesh.aabb();
        let mesh_stats = (mesh.vertex_count(), mesh.triangle_count());

        let leaves = build_leaves(&skeleton, &grown);
        let leaf_stats = (leaves.leaf_count(), leaves.triangle_count());
        let crown = crown_bounds(&leaves);

        let mut mesh_gpu = GpuMesh::new(&gl);
        mesh_gpu.upload(&gl, &mesh);
        let mut leaves_gpu = GpuLeaves::new(&gl);
        leaves_gpu.upload(&gl, &leaves);
        let lines = GpuLines::new(&gl);
        // Big enough that the edge of a leaf's shadow is not a staircase. The web
        // makes do with less memory.
        let shadow = ShadowTarget::new(&gl, if cfg!(target_arch = "wasm32") { 2048 } else { 4096 });
        let depth_pass = DepthPass::new(&gl);
        let leaf_depth_pass = LeafDepthPass::new(&gl);
        let color_pass = ColorPass::new(&gl);
        let backdrop = GpuBackdrop::new(&gl);
        let ground_pass = GpuGround::new(&gl);
        let renderer = Renderer::new(&gl);
        // The real maps are put in by `sync_materials` below, at once on the desktop and
        // once they have been fetched on the web.
        let bark_material = unsafe { gpu::placeholder_material(&gl) };

        let mut app = Self {
            gl: Arc::clone(&gl),
            lines: Arc::new(Mutex::new(lines)),
            mesh_gpu: Arc::new(Mutex::new(mesh_gpu)),
            leaves_gpu: Arc::new(Mutex::new(leaves_gpu)),
            shadow: Arc::new(shadow),
            depth_pass: Arc::new(depth_pass),
            leaf_depth_pass: Arc::new(leaf_depth_pass),
            color_pass: Arc::new(color_pass),
            backdrop: Arc::new(backdrop),
            ground_pass: Arc::new(ground_pass),
            renderer: Arc::new(Mutex::new(renderer)),
            hdri_maps: assets::Maps::new(hdri::HDRI_DIR),
            hdris: hdri::available(),
            // A photograph by default: it is what shows a tree best. The procedural
            // sky is a click away, with the time of day it brings.
            environment: Some(hdri::BUNDLED[0].to_string()),
            built: Built::Nothing,
            photo: None,
            env_status: None,
            env_rotation: 0.0,
            env_intensity: 1.0,
            photo_sun: 1.0,
            background_blur: 0.0,
            ground: Ground::Photo,
            ground_height: hdri::staging(hdri::BUNDLED[0]).shot_from,
            ground_radius: hdri::staging(hdri::BUNDLED[0]).radius,
            sun_size: 0.53,
            shadow_softness: 1.0,
            exposure_ev: 0.0,
            tonemap: Tonemap::Aces,
            bloom: 0.04,
            ao: true,
            ao_radius: 1.2,
            ao_strength: 1.0,
            msaa: 4,
            maps: assets::Maps::default(),
            bark_material,
            leaf_material: None,
            loaded_bark: String::new(),
            loaded_leaf: (String::new(), None),
            presets,
            preset: Some(preset_index),
            save_name,
            status,
            confirm_delete: false,
            exports: export::Exports::default(),
            params,
            grown,
            skeleton,
            stats: SkeletonStats::default(),
            mesh_stats,
            leaf_stats,
            aabb,
            crown,
            camera: OrbitCamera {
                target: Vec3::new(0.0, 6.0, 0.0),
                distance: 26.0,
                yaw: 0.6,
                pitch: 0.30,
                fov_y: 50.0f32.to_radians(),
                aspect: 1.6,
            },
            dirty: true,
            gen_ms: 0.0,
            render_mode: RenderMode::Shaded,
            wireframe: false,
            shadows: true,
            show_skeleton: false,
            sun_intensity: 1.0,
            show_leaves: true,
            leaf_translucency: 0.9,
            show_grid: false,
            use_normal_map: true,
            // Kept in step with the hour below: with the sky built from the sun, a
            // dawn look is a dawn time rather than a dawn palette over a high sun.
            sun_azimuth: sun_at_hour(DEFAULT_HOUR).1,
            sun_elevation: sun_at_hour(DEFAULT_HOUR).0,
            time_of_day: DEFAULT_HOUR,
            auto_frame: true,
            capture: startup.capture,
            frames: 0,
            coverage_lod: 1.3,
            wind_on: true,
            wind_strength: DEFAULT_WIND,
            wind_gustiness: DEFAULT_GUSTINESS,
            wind_direction: DEFAULT_WIND_DIRECTION,
            wind_clock: 0.0,
            wind_paused: false,
        };
        // A capture is a measurement, so it is taken in still air unless it asks for
        // wind, and on a stopped clock when it does, so one command gives one frame.
        if app.capture.is_some() {
            app.wind_on = false;
            app.wind_paused = true;
        }
        if let Some(v) = startup.wind {
            app.wind_on = v > 0.0;
            app.wind_strength = v;
        }
        if let Some(v) = startup.gustiness {
            app.wind_gustiness = v;
        }
        if let Some(v) = startup.wind_direction {
            app.wind_direction = v;
        }
        if let Some(t) = startup.wind_time {
            app.wind_clock = t;
            app.wind_paused = true;
        }
        if let Some(v) = startup.leaves {
            app.show_leaves = v;
            app.params.leaves.enabled = v;
        }
        if let Some(v) = startup.shadows {
            app.shadows = v;
        }
        if let Some(v) = startup.translucency {
            app.leaf_translucency = v;
        }
        app.wireframe = startup.wireframe;
        if let Some(h) = startup.time_of_day {
            app.time_of_day = h;
            let (el, az) = sun_at_hour(h);
            app.sun_elevation = el;
            app.sun_azimuth = az;
        }
        if let Some(v) = startup.sun_elevation {
            app.sun_elevation = v;
        }
        if let Some(v) = startup.sun_azimuth {
            app.sun_azimuth = v;
        }
        if let Some(v) = startup.sun_intensity {
            app.sun_intensity = v;
        }
        if let Some(v) = startup.yaw {
            app.camera.yaw = v;
        }
        if let Some(v) = startup.pitch {
            app.camera.pitch = v;
        }
        if let Some(v) = startup.distance {
            app.camera.distance = v;
            app.auto_frame = false;
        }
        if let Some(v) = startup.target_y {
            app.camera.target.y = v;
            app.auto_frame = false;
        }
        if let Some(v) = startup.coverage_lod {
            app.coverage_lod = v;
        }
        // Setting the sun by hand means the procedural sky, unless a photograph was
        // asked for as well.
        if startup.environment.is_none()
            && (startup.time_of_day.is_some() || startup.sun_elevation.is_some() || startup.sun_azimuth.is_some())
        {
            app.environment = None;
        }
        if let Some(env) = startup.environment {
            if let Some(name) = &env {
                app.stage(name);
            }
            app.environment = env;
        }
        if let Some(v) = startup.env_rotation {
            app.env_rotation = v;
        }
        if let Some(v) = startup.env_intensity {
            app.env_intensity = v;
        }
        if let Some(v) = startup.exposure {
            app.exposure_ev = v;
        }
        if let Some(v) = startup.tonemap {
            app.tonemap = v;
        }
        if let Some(v) = startup.bloom {
            app.bloom = v;
        }
        if let Some(v) = startup.ao {
            app.ao = v;
        }
        if let Some(v) = startup.ao_radius {
            app.ao_radius = v;
        }
        if let Some(v) = startup.background_blur {
            app.background_blur = v;
        }
        if let Some(v) = startup.ground {
            app.ground = v;
        }
        if let Some(v) = startup.view {
            app.render_mode = v;
        }
        if let Some(v) = startup.msaa {
            app.msaa = v;
        }
        if let Some(v) = startup.sun_size {
            app.sun_size = v;
        }
        if let Some(v) = startup.shadow_softness {
            app.shadow_softness = v;
        }
        app.stats = app.skeleton.stats();
        app.sync_materials();
        app.sync_environment();
        app.rebuild_overlay();
        app
    }

    fn regenerate(&mut self) {
        let t = web_time::Instant::now();
        self.grown = self.params.instance();
        self.skeleton = grow(&self.grown);
        let mesh = build_mesh(&self.skeleton, &self.grown);
        self.aabb = mesh.aabb();
        self.mesh_stats = (mesh.vertex_count(), mesh.triangle_count());
        self.mesh_gpu.lock().unwrap().upload(&self.gl, &mesh);
        let leaves = build_leaves(&self.skeleton, &self.grown);
        self.leaf_stats = (leaves.leaf_count(), leaves.triangle_count());
        self.crown = crown_bounds(&leaves);
        self.leaves_gpu.lock().unwrap().upload(&self.gl, &leaves);
        self.sync_materials();
        self.gen_ms = t.elapsed().as_secs_f32() * 1000.0;
        self.stats = self.skeleton.stats();
        self.rebuild_overlay();
        if self.auto_frame {
            self.frame_camera();
            self.auto_frame = false;
        }
    }

    /// Textures follow the species, so a preset switch has to swap them and hand the
    /// old ones back rather than keep loading new ones on top. On the web a species'
    /// maps have to be fetched first, and until they are all in, what was there stays.
    fn sync_materials(&mut self) {
        let bark = &self.grown.mesh.bark_texture;
        if *bark != self.loaded_bark && self.maps.ready(&textures::map_files(bark)) {
            self.bark_material.delete(&self.gl);
            self.loaded_bark = bark.clone();
            self.bark_material =
                unsafe { gpu::load_material(&self.gl, &self.maps, &self.loaded_bark) };
        }
        let leaves = &self.grown.leaves;
        let loaded = &self.loaded_leaf;
        if (&leaves.texture, &leaves.cluster) != (&loaded.0, &loaded.1)
            && self.maps.ready(&textures::map_files(&leaves.texture))
        {
            if let Some(old) = self.leaf_material.take() {
                old.delete(&self.gl);
            }
            self.loaded_leaf = (leaves.texture.clone(), leaves.cluster.clone());
            self.leaf_material =
                unsafe { gpu::load_leaf_material(&self.gl, &self.maps, &self.grown.leaves) };
        }
    }

    /// Sets the ground up the way the photograph `name` wants it, when it is picked.
    /// A tree left standing on nothing stays that way.
    fn stage(&mut self, name: &str) {
        let staging = hdri::staging(name);
        (self.ground_height, self.ground_radius) = (staging.shot_from, staging.radius);
        if self.ground != Ground::None {
            self.ground = if staging.has_ground { Ground::Photo } else { Ground::Plain };
        }
    }

    /// The procedural sky for where the sun is now.
    fn procedural_sky(&self) -> SkyParams {
        let mut sky = SkyParams::for_sun(self.sun_dir());
        sky.sun_color *= self.sun_intensity;
        sky
    }

    /// Builds the environment maps from whatever is asked for, when that has changed:
    /// the procedural sky whenever the sun moves, a photograph once it is in.
    fn sync_environment(&mut self) {
        let Some(name) = self.environment.clone() else {
            let sky = self.procedural_sky();
            if self.built != Built::Sky(sky) {
                self.renderer.lock().unwrap().env.load_sky(&self.gl, &sky);
                self.built = Built::Sky(sky);
                self.photo = None;
                self.env_status = None;
            }
            return;
        };
        if self.built == Built::Photo(name.clone()) {
            return;
        }
        let file = format!("{name}.hdr");
        if !self.hdri_maps.ready(std::slice::from_ref(&file)) {
            self.env_status = Some((format!("fetching {name}…"), false));
            return;
        }
        let started = web_time::Instant::now();
        let decoded = self
            .hdri_maps
            .read(&file)
            .ok_or_else(|| format!("no {file} in {}", hdri::HDRI_DIR))
            .and_then(|bytes| hdri::Equirect::decode(&bytes));
        match decoded {
            Ok(shown) => {
                let analysed = shown.clone().analyse();
                self.renderer
                    .lock()
                    .unwrap()
                    .env
                    .load_photo(&self.gl, &analysed.image, &shown);
                println!(
                    "environment {name}: {}x{}, sun {} ({:.0} ms)",
                    shown.width,
                    shown.height,
                    analysed.sun.map_or("none".to_string(), |s| format!(
                        "{:.1} deg up",
                        s.dir.y.asin().to_degrees()
                    )),
                    started.elapsed().as_secs_f32() * 1000.0
                );
                self.photo = Some(Photo { sun: analysed.sun, sh: analysed.sh });
                self.built = Built::Photo(name);
                self.env_status = None;
            }
            Err(e) => {
                // Back to the procedural sky, rather than a tree lit by nothing.
                self.env_status = Some((format!("{name}: {e}"), true));
                self.environment = None;
            }
        }
    }

    /// Everything about the light for this frame except what depends on the viewport
    /// and the shadow map, which the paint callback fills in.
    fn lighting(&self) -> Lighting {
        let procedural = self.procedural_sky();
        let ev = 2f32.powf(self.exposure_ev);
        let base = Lighting {
            sky: procedural,
            photo: false,
            sh: sh9_cached(&procedural),
            env_rotation: 0.0,
            env_intensity: 1.0,
            sun_dir: procedural.sun_dir,
            sun_color: procedural.sun_color,
            sun_radius: (self.sun_size * 0.5).to_radians(),
            shadow_softness: self.shadow_softness,
            exposure: procedural.exposure() * ev,
            cam_pos: self.camera.eye(),
            ground_projection: None,
            background_blur: 0.0,
            shadow: shadow_frustum(self.aabb, procedural.sun_dir, 1),
            shadow_size: 1,
            normal_bias: 0.0,
            ao_texel: [0.0, 0.0],
            canopy: render::CanopyPlacement::default(),
        };
        let (Some(photo), Built::Photo(name)) = (&self.photo, &self.built) else {
            return base;
        };
        let rotation = self.env_rotation.to_radians();
        let (sun_dir, sun_color, sun_radius) = match photo.sun {
            Some(sun) => (
                hdri::from_env(sun.dir, rotation),
                sun.irradiance * (self.env_intensity * self.photo_sun),
                sun.angular_radius.clamp(0.0046, 0.03),
            ),
            None => (Vec3::Y, Vec3::ZERO, 0.0046),
        };
        Lighting {
            photo: true,
            sh: photo.sh,
            env_rotation: rotation,
            env_intensity: self.env_intensity,
            sun_dir,
            sun_color,
            sun_radius,
            // A photograph comes already exposed: published HDRIs are saved so that
            // they look right shown as they are, which is also what Blender shows by
            // default. Metering the light on the tree instead blows a bright sky out.
            exposure: ev * 2f32.powf(hdri::staging(name).exposure),
            ground_projection: (self.ground == Ground::Photo)
                .then_some((self.ground_height, self.ground_radius)),
            background_blur: self.background_blur,
            ..base
        }
    }

    fn frame_camera(&mut self) {
        let h = self.stats.height.max(1.0);
        self.camera.target = Vec3::new(0.0, h * 0.45, 0.0);
        self.camera.distance = h * 1.9;
    }

    fn rebuild_overlay(&mut self) {
        let mut verts = Vec::new();
        if self.show_grid {
            push_grid(&mut verts);
        }
        if self.show_skeleton {
            push_skeleton(&self.skeleton, &mut verts);
        }
        self.lines.lock().unwrap().upload(&self.gl, &verts);
    }

    fn sun_dir(&self) -> Vec3 {
        let el = self.sun_elevation.to_radians();
        let az = self.sun_azimuth.to_radians();
        Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos())
    }

    /// Loads preset `i` over whatever the panel holds.
    ///
    /// A built-in keeps the seed being browsed, so flicking between species compares
    /// like with like. A saved preset brings its own, because the seed is part of the
    /// tree that was saved.
    fn pick_preset(&mut self, i: usize) {
        self.confirm_delete = false;
        let preset = &self.presets[i];
        match preset.load() {
            Ok(mut p) => {
                if !preset.is_saved() {
                    p.seed = self.params.seed;
                }
                self.save_name = save_name_for(preset, &p);
                self.params = p;
                self.preset = Some(i);
                self.dirty = true;
                self.status = None;
            }
            Err(e) => self.status = Some((format!("{}: {e}", preset.name), true)),
        }
    }

    fn save_preset(&mut self) {
        self.confirm_delete = false;
        let dir = Path::new(CUSTOM_PRESET_DIR);
        match presets::save(dir, &self.save_name, &self.params) {
            Ok(path) => {
                self.params.name = self.save_name.trim().to_string();
                self.presets = presets::list(dir);
                let key = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
                self.preset = self.presets.iter().position(|p| p.is_saved() && p.name == key);
                self.status = Some((format!("saved {}", path.display()), false));
            }
            Err(e) => self.status = Some((e, true)),
        }
    }

    fn delete_preset(&mut self, i: usize) {
        self.confirm_delete = false;
        match presets::delete(&self.presets[i]) {
            Ok(()) => {
                self.status = Some((format!("deleted {}", self.presets[i].name), false));
                self.presets = presets::list(Path::new(CUSTOM_PRESET_DIR));
                // The tree on screen stays as it is; it just has no preset behind it now.
                self.preset = None;
            }
            Err(e) => self.status = Some((e, true)),
        }
    }

    /// The name to save the species under, and the button that saves it.
    fn save_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Save as");
            ui.add(egui::TextEdit::singleline(&mut self.save_name).desired_width(150.0));
            let target = presets::path_for(Path::new(CUSTOM_PRESET_DIR), &self.save_name);
            let taken = target.as_ref().is_ok_and(|p| p.exists());
            let button = ui.add_enabled(
                target.is_ok(),
                egui::Button::new(if taken { "Overwrite" } else { "Save" }),
            );
            let button = match &target {
                Ok(path) => button.on_hover_text(format!(
                    "Write the species as it stands, every value and the seed included, to {}.{}",
                    path.display(),
                    if taken { " A preset of that name is already there and will be replaced." } else { "" }
                )),
                Err(why) => button.on_disabled_hover_text(why),
            };
            if button.clicked() {
                self.save_preset();
            }
        });
    }

    /// Picking, saving and deleting presets.
    fn presets_ui(&mut self, ui: &mut egui::Ui) {
        let mut picked = None;
        let current = self
            .preset
            .map_or_else(|| "(unsaved)".to_string(), |i| self.presets[i].name.clone());
        egui::ComboBox::from_label("Preset")
            .selected_text(current)
            .show_ui(ui, |ui| {
                let mut saved_heading = false;
                for (i, preset) in self.presets.iter().enumerate() {
                    if preset.is_saved() && !saved_heading {
                        ui.separator();
                        ui.weak("Saved");
                        saved_heading = true;
                    }
                    if ui.selectable_label(self.preset == Some(i), &preset.name).clicked() {
                        picked = Some(i);
                    }
                }
            });
        if let Some(i) = picked {
            self.pick_preset(i);
        }

        // A saved preset is a file in the repository, and a page has nowhere to put one.
        if !cfg!(target_arch = "wasm32") {
            self.save_ui(ui);
        }

        self.exports.ui(ui, &self.params, &self.save_name, &mut self.status);

        if let Some(i) = self.preset.filter(|&i| self.presets[i].is_saved()) {
            let name = self.presets[i].name.clone();
            if self.confirm_delete {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::YELLOW, format!("Delete {name} for good?"));
                    if ui.button("Delete").clicked() {
                        self.delete_preset(i);
                    }
                    if ui.button("Keep").clicked() {
                        self.confirm_delete = false;
                    }
                });
            } else if ui
                .button(format!("Delete {name}"))
                .on_hover_text("Remove this saved preset's file. Asks first. The tree on screen stays.")
                .clicked()
            {
                self.confirm_delete = true;
            }
        }

        if let Some((message, failed)) = &self.status {
            if *failed {
                ui.colored_label(egui::Color32::LIGHT_RED, message);
            } else {
                ui.weak(message);
            }
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Arbor");
        ui.separator();

        self.presets_ui(ui);

        ui.horizontal(|ui| {
            ui.label("Seed");
            if ui
                .add(egui::DragValue::new(&mut self.params.seed).speed(1.0))
                .changed()
            {
                self.dirty = true;
            }
            if ui.button("Random").clicked() {
                self.params.seed = random_seed();
                self.dirty = true;
            }
        });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.preset.is_some(), egui::Button::new("Reset"))
                .on_hover_text("Reload the preset, dropping every change made in the panel. A saved preset is read from disk again, seed and all.")
                .clicked()
                && let Some(i) = self.preset
            {
                self.pick_preset(i);
            }
            if ui
                .button("Copy as RON")
                .on_hover_text("Copy the species as it stands, every value included, to paste into a preset file.")
                .clicked()
            {
                match ron::ser::to_string_pretty(&self.params, ron::ser::PrettyConfig::default()) {
                    Ok(text) => ctx.copy_text(text),
                    Err(e) => eprintln!("could not write species as RON: {e}"),
                }
            }
        });

        // Where this seed lands in every range, for the panel to say under each one.
        // Taken from the panel's own species rather than from the tree on screen, so it
        // is never a level short of what the panel is drawing, and a range being dragged
        // reports where the tree is about to land.
        let mut landed = self.params.instance().template();

        ui.separator();
        ui.label("Shape");
        self.dirty |= knobs::knobs_ui(ui, &mut self.params, &mut landed, knobs::SHAPE);

        let mut levels = u32::from(self.params.max_levels);
        if ui
            .add(
                egui::Slider::new(&mut levels, knobs::BRANCH_LEVELS.0..=knobs::BRANCH_LEVELS.1)
                    .clamping(egui::SliderClamping::Edits)
                    .text("Branch levels"),
            )
            .on_hover_text("Levels grown, counting the trunk. Capped by how many levels the species declares.")
            .changed()
        {
            self.params.max_levels = levels as u8;
            self.dirty = true;
        }
        self.dirty |= knobs::count_ui(ui, &mut self.params, &knobs::SPLIT_DEPTH);
        self.dirty |= knobs::group_ui(
            ui,
            "envelope",
            &mut self.params.envelope,
            &mut landed.envelope,
            &knobs::ENVELOPE,
        );

        ui.separator();
        ui.label("Branching");
        let grown = usize::from(self.params.max_levels).min(knobs::level_count(&self.params));
        for level in 0..knobs::level_count(&self.params) {
            let name = if level == 0 { "Trunk".to_string() } else { format!("Level {level}") };
            // A level past `Branch levels` is still editable, so it can be set up
            // before it is switched on, but it says that it is not being grown.
            let title = if level < grown { name } else { format!("{name} (not grown)") };
            egui::CollapsingHeader::new(title)
                .id_salt(("level", level))
                .show(ui, |ui| {
                    self.dirty |= knobs::level_ui(ui, &mut self.params, &mut landed, level);
                });
        }

        ui.separator();
        ui.label("Render");
        egui::ComboBox::from_label("View")
            .selected_text(match self.render_mode {
                RenderMode::Shaded => "Shaded",
                RenderMode::UvChecker => "UV checker",
                RenderMode::Normals => "Normals",
                RenderMode::Occlusion => "Occlusion",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.render_mode, RenderMode::Shaded, "Shaded");
                ui.selectable_value(&mut self.render_mode, RenderMode::UvChecker, "UV checker");
                ui.selectable_value(&mut self.render_mode, RenderMode::Normals, "Normals");
                ui.selectable_value(&mut self.render_mode, RenderMode::Occlusion, "Occlusion");
            });
        ui.checkbox(&mut self.show_leaves, "Leaves");
        if self.leaf_material.is_none() {
            if self.loaded_leaf.0 != self.grown.leaves.texture {
                ui.weak("fetching the leaf textures…");
            } else {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!("no {}_albedo.png in {TEXTURE_DIR}", self.loaded_leaf.0),
                );
            }
        }
        ui.add(
            egui::Slider::new(&mut self.leaf_translucency, 0.0..=2.0).text("Leaf translucency"),
        );
        ui.add(
            egui::Slider::new(&mut self.coverage_lod, 0.0..=10.0).text("Leaf coverage LOD"),
        );
        // WebGL draws no lines in place of triangles.
        if !cfg!(target_arch = "wasm32") {
            ui.checkbox(&mut self.wireframe, "Wireframe");
        }
        ui.checkbox(&mut self.shadows, "Shadows");
        ui.checkbox(&mut self.use_normal_map, "Normal map");
        if ui.checkbox(&mut self.show_grid, "Grid").changed() {
            self.rebuild_overlay();
        }
        if ui.checkbox(&mut self.show_skeleton, "Skeleton").changed() {
            self.rebuild_overlay();
        }
        self.lighting_ui(ui);

        ui.separator();
        ui.label("Wind");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.wind_on, "Wind")
                .on_hover_text("Sway the tree. Off is still air: the tree exactly as it was grown.");
            ui.checkbox(&mut self.wind_paused, "Pause")
                .on_hover_text("Hold the tree where the wind has it, to look at one pose.");
        });
        ui.add(egui::Slider::new(&mut self.wind_strength, 0.0..=1.5).text("Strength"))
            .on_hover_text("How hard it blows. About 0.2 is a breeze, 0.5 a fresh wind, 1 a full gale.");
        ui.add(egui::Slider::new(&mut self.wind_gustiness, 0.0..=1.0).text("Gustiness"))
            .on_hover_text("How far the wind swells and lulls about its strength. Gusts travel downwind, so they cross a crown rather than arriving everywhere at once.");
        ui.add(egui::Slider::new(&mut self.wind_direction, 0.0..=360.0).text("Direction"))
            .on_hover_text("Bearing the wind blows toward, in degrees, measured like the sun's azimuth.");
        // These are the species' own and are saved with it, but nothing about them
        // changes what grows, so they never mark the tree for regrowing — only the sway
        // the tree on screen is drawn with has to follow.
        if knobs::group_ui(ui, "wind", &mut self.params.wind, &mut landed.wind, &knobs::WIND) {
            self.grown.wind = self.params.instance().wind;
        }

        ui.separator();
        ui.label("Foliage");
        let leaves = &mut self.params.leaves;
        let mut leafy = leaves.enabled;
        if ui.checkbox(&mut leafy, "Generate leaves").changed() {
            leaves.enabled = leafy;
            self.dirty = true;
        }
        self.dirty |=
            knobs::knobs_ui(ui, &mut self.params.leaves, &mut landed.leaves, knobs::FOLIAGE);
        let mut min_level = u32::from(self.params.leaves.min_level);
        if ui
            .add(
                egui::Slider::new(&mut min_level, knobs::LEAF_MIN_LEVEL.0..=knobs::LEAF_MIN_LEVEL.1)
                    .clamping(egui::SliderClamping::Edits)
                    .text("Leaf from level"),
            )
            .on_hover_text("Leaves grow on stems at this level and deeper, so the canopy sits on twigs rather than limbs.")
            .changed()
        {
            self.params.leaves.min_level = min_level as u8;
            self.dirty = true;
        }
        for group in knobs::FOLIAGE_MORE {
            self.dirty |=
                knobs::group_ui(ui, "foliage", &mut self.params.leaves, &mut landed.leaves, group);
        }

        ui.separator();
        ui.label("Bark & roots");
        for group in knobs::BARK {
            self.dirty |=
                knobs::group_ui(ui, "bark", &mut self.params.mesh, &mut landed.mesh, group);
        }
        for group in knobs::IRREGULARITY {
            self.dirty |= knobs::group_ui(
                ui,
                "bark",
                &mut self.params.mesh.irregularity,
                &mut landed.mesh.irregularity,
                group,
            );
        }

        ui.separator();
        ui.label("Stats");
        let s = &self.stats;
        ui.monospace(format!("nodes:  {}", s.node_count));
        ui.monospace(format!("segs:   {}", s.segment_count));
        ui.monospace(format!("tips:   {}", s.tip_count));
        ui.monospace(format!("height: {:.1} m", s.height));
        ui.monospace(format!("verts:  {}", self.mesh_stats.0));
        ui.monospace(format!("tris:   {}", self.mesh_stats.1));
        ui.monospace(format!("leaves: {}", self.leaf_stats.0));
        ui.monospace(format!("l.tris: {}", self.leaf_stats.1));
        ui.monospace(format!("gen:    {:.2} ms", self.gen_ms));
        ui.monospace(format!("frame:  {:.1} ms", ctx.input(|i| i.stable_dt) * 1000.0))
            .on_hover_text("Time between frames, smoothed. Held to the display's refresh rate while the GPU keeps up.");

        if ui.button("Frame tree").clicked() {
            self.frame_camera();
        }

        ui.separator();
        ui.label("Camera: LMB orbit, RMB/MMB pan, wheel zoom");
    }

    /// Where the light comes from, and what stands under the tree.
    fn lighting_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Lighting");
        let mut picked = None;
        egui::ComboBox::from_label("Environment")
            .selected_text(self.environment.as_deref().unwrap_or("Procedural sky"))
            .show_ui(ui, |ui| {
                if ui.selectable_label(self.environment.is_none(), "Procedural sky").clicked() {
                    picked = Some(None);
                }
                for name in &self.hdris {
                    let on = self.environment.as_deref() == Some(name.as_str());
                    if ui.selectable_label(on, name).clicked() {
                        picked = Some(Some(name.clone()));
                    }
                }
            })
            .response
            .on_hover_text(format!(
                "What lights the tree and stands behind it: a sky built from the time of day, or a photographed environment from {}. Any .hdr put there is offered.",
                hdri::HDRI_DIR
            ));
        if let Some(env) = picked {
            if let Some(name) = &env {
                self.stage(name);
            }
            self.environment = env;
            self.env_status = None;
        }
        if let Some((message, failed)) = &self.env_status {
            if *failed {
                ui.colored_label(egui::Color32::LIGHT_RED, message);
            } else {
                ui.weak(message);
            }
        }

        if self.environment.is_none() {
            if ui
                .add(
                    egui::Slider::new(&mut self.time_of_day, 3.5..=20.5)
                        .text("Time of day")
                        .custom_formatter(|h, _| {
                            format!("{:02}:{:02}", h as i32, ((h % 1.0) * 60.0) as i32)
                        }),
                )
                .changed()
            {
                let (el, az) = sun_at_hour(self.time_of_day);
                self.sun_elevation = el;
                self.sun_azimuth = az;
            }
            ui.add(egui::Slider::new(&mut self.sun_azimuth, 0.0..=360.0).text("Sun azimuth"));
            ui.add(egui::Slider::new(&mut self.sun_elevation, -8.0..=85.0).text("Sun elevation"));
            ui.add(egui::Slider::new(&mut self.sun_intensity, 0.1..=3.0).text("Sun intensity"));
            ui.add(egui::Slider::new(&mut self.sun_size, 0.1..=6.0).text("Sun size"))
                .on_hover_text("Angular diameter of the sun, in degrees. The real one is about half a degree; a bigger sun softens every shadow.");
        } else {
            ui.add(egui::Slider::new(&mut self.env_rotation, -180.0..=180.0).text("Rotation"))
                .on_hover_text("Turn the photograph, and the sun in it, about the vertical.");
            ui.add(
                egui::Slider::new(&mut self.env_intensity, 0.1..=4.0)
                    .logarithmic(true)
                    .text("Intensity"),
            )
            .on_hover_text("Brightness of the whole environment, sun included. Exposure follows it, so this mostly moves the tree against its background.");
            if self.photo.as_ref().is_some_and(|p| p.sun.is_some()) {
                ui.add(egui::Slider::new(&mut self.photo_sun, 0.0..=3.0).text("Sun strength"))
                    .on_hover_text("The sun is lifted out of the photograph and put back as a light that casts shadows. 1 puts it back as bright as it was.");
            } else if matches!(self.built, Built::Photo(_)) {
                ui.weak("No sun stands out in this one: the sky lights it alone.");
            }
            ui.add(egui::Slider::new(&mut self.background_blur, 0.0..=1.0).text("Background blur"));
        }

        let photo = self.environment.is_some();
        egui::ComboBox::from_label("Ground")
            .selected_text(match self.ground {
                Ground::Photo if photo => "Photographed",
                Ground::None => "None",
                _ => "Plain",
            })
            .show_ui(ui, |ui| {
                if photo {
                    ui.selectable_value(&mut self.ground, Ground::Photo, "Photographed")
                        .on_hover_text("The ground in the photograph, laid flat under the tree and taking its shadow.");
                }
                ui.selectable_value(&mut self.ground, Ground::Plain, "Plain");
                ui.selectable_value(&mut self.ground, Ground::None, "None");
            });
        if photo && self.ground == Ground::Photo {
            ui.add(
                egui::Slider::new(&mut self.ground_height, 0.5..=30.0)
                    .logarithmic(true)
                    .text("Shot from"),
            )
            .on_hover_text("How many metres above the ground the photograph was taken. Higher spreads the photographed ground wider under the tree.");
            ui.add(
                egui::Slider::new(&mut self.ground_radius, 5.0..=1000.0)
                    .logarithmic(true)
                    .text("Surroundings at"),
            )
            .on_hover_text("How many metres off the photograph's surroundings stand. The ground runs out to here and what is further stands up around it, rather than lying flat.");
        }
        ui.add(egui::Slider::new(&mut self.shadow_softness, 0.0..=6.0).text("Shadow softness"))
            .on_hover_text("How far a shadow softens with distance from what casts it, against what the sun's own size gives. 0 is a pin-sharp shadow.");

        ui.separator();
        ui.label("Camera");
        ui.add(egui::Slider::new(&mut self.exposure_ev, -4.0..=4.0).text("Exposure (EV)"))
            .on_hover_text("Stops brighter or darker than the automatic exposure.");
        egui::ComboBox::from_label("Tonemapping")
            .selected_text(self.tonemap.label())
            .show_ui(ui, |ui| {
                for t in Tonemap::ALL {
                    ui.selectable_value(&mut self.tonemap, t, t.label());
                }
            });
        ui.add(egui::Slider::new(&mut self.bloom, 0.0..=0.2).text("Bloom"));
        ui.checkbox(&mut self.ao, "Ambient occlusion")
            .on_hover_text("Darken the light from the sky where the tree crowds it out: inside the crown, in the forks, and on the ground at its foot.");
        if self.ao {
            ui.add(egui::Slider::new(&mut self.ao_radius, 0.2..=5.0).text("AO radius"))
                .on_hover_text("How far, in metres, the occlusion looks for what crowds a point.");
            ui.add(egui::Slider::new(&mut self.ao_strength, 0.3..=3.0).text("AO strength"));
        }
        egui::ComboBox::from_label("Anti-aliasing")
            .selected_text(match self.msaa {
                0 | 1 => "Off".to_string(),
                n => format!("{n}x MSAA"),
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.msaa, 0, "Off");
                for n in [2, 4, 8] {
                    ui.selectable_value(&mut self.msaa, n, format!("{n}x MSAA"));
                }
            })
            .response
            .on_hover_text("Samples per pixel. Leaf edges are resolved from them too, so fewer samples draw a coarser canopy.");
    }

    fn camera_input(&mut self, resp: &egui::Response, ctx: &egui::Context) {
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.camera.yaw -= d.x * 0.008;
            self.camera.pitch = (self.camera.pitch + d.y * 0.008).clamp(-1.45, 1.45);
        } else if resp.dragged_by(egui::PointerButton::Secondary)
            || resp.dragged_by(egui::PointerButton::Middle)
        {
            let d = resp.drag_delta();
            let k = self.camera.distance * 0.0016;
            let delta = self.camera.right() * d.x * k - self.camera.up() * d.y * k;
            self.camera.target -= delta;
            self.camera.target.y = self.camera.target.y.max(0.0);
        }
        if resp.hovered() {
            let scroll = ctx.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.distance =
                    (self.camera.distance * (1.0 - scroll * 0.0012)).clamp(0.8, 300.0);
            }
        }
    }
}

/// The bounding box of a canopy's cards, empty (min above max) for none.
fn crown_bounds(leaves: &arbor_core::LeafMesh) -> (Vec3, Vec3) {
    leaves.positions.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(Vec3::from(*p)), hi.max(Vec3::from(*p))),
    )
}

fn push_grid(verts: &mut Vec<f32>) {
    const N: i32 = 20;
    const STEP: f32 = 1.0;
    let minor = [0.30, 0.30, 0.34];
    let axis_x = [0.70, 0.30, 0.30];
    let axis_z = [0.30, 0.40, 0.70];
    let half = N as f32 * STEP;
    for i in -N..=N {
        let p = i as f32 * STEP;
        let col = if i == 0 { axis_x } else { minor };
        verts.extend_from_slice(&[p, 0.0, -half, col[0], col[1], col[2]]);
        verts.extend_from_slice(&[p, 0.0, half, col[0], col[1], col[2]]);
        let col = if i == 0 { axis_z } else { minor };
        verts.extend_from_slice(&[-half, 0.0, p, col[0], col[1], col[2]]);
        verts.extend_from_slice(&[half, 0.0, p, col[0], col[1], col[2]]);
    }
}

fn push_skeleton(skeleton: &Skeleton, verts: &mut Vec<f32>) {
    for (parent, child) in skeleton.segments() {
        let col = LEVEL_COLORS[(child.level as usize).min(7)];
        verts.extend_from_slice(&[
            parent.position.x,
            parent.position.y,
            parent.position.z,
            col[0],
            col[1],
            col[2],
        ]);
        verts.extend_from_slice(&[
            child.position.x,
            child.position.y,
            child.position.z,
            col[0],
            col[1],
            col[2],
        ]);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // A capture run draws a few frames, asks egui for the finished image, writes
        // it and quits. What lands on disk is the real renderer, not a stand-in.
        if let Some(capture) = self.capture.clone() {
            self.frames += 1;
            if self.frames == capture.settle {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                    egui::UserData::default(),
                ));
            }
            let shot = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(image) = shot {
                save_screenshot(&image, &capture.path);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if self.dirty {
            self.regenerate();
            self.dirty = false;
        }
        // Maps being fetched are put in on the frame they arrive.
        self.sync_materials();
        self.sync_environment();
        self.exports.poll(frame, &mut self.status, &self.maps);

        if self.wind_on && !self.wind_paused {
            // Held to a tenth of a second so a stall — a regrow, a dragged window —
            // does not throw the tree forward a whole gust in one frame.
            self.wind_clock += ctx.input(|i| i.stable_dt).min(0.1);
        }

        egui::SidePanel::left("controls")
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, ctx));
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let rect = ui.available_rect_before_wrap();
                let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                // A capture opens its window wherever the cursor happens to be, and a
                // stray scroll or drag over it would move the camera the command line
                // placed before the frame is taken.
                if self.capture.is_none() {
                    self.camera_input(&resp, ctx);
                }

                // The scene fills the panel, so the camera takes the panel's shape.
                self.camera.aspect = rect.width() / rect.height().max(1.0);

                let mesh_gpu = Arc::clone(&self.mesh_gpu);
                let leaves_gpu = Arc::clone(&self.leaves_gpu);
                let lines = Arc::clone(&self.lines);
                let shadow = Arc::clone(&self.shadow);
                let depth_pass = Arc::clone(&self.depth_pass);
                let leaf_depth_pass = Arc::clone(&self.leaf_depth_pass);
                let color_pass = Arc::clone(&self.color_pass);
                let backdrop = Arc::clone(&self.backdrop);
                let ground_pass = Arc::clone(&self.ground_pass);
                let renderer = Arc::clone(&self.renderer);
                // Sun, ambient and exposure all come out of the environment, so moving
                // the sun or turning the photograph moves the whole of the light.
                let base_lighting = self.lighting();
                let ground = match self.ground {
                    Ground::None => None,
                    Ground::Photo if base_lighting.photo => Some(GroundMode::Photo),
                    _ => Some(GroundMode::Plain),
                };
                let tree_height = self.stats.height.max(1.0);
                let wind = if self.wind_on {
                    WindUniforms::new(
                        &self.grown.wind,
                        self.wind_clock,
                        self.wind_direction,
                        self.wind_strength,
                        self.wind_gustiness,
                        tree_height,
                    )
                } else {
                    WindUniforms::default()
                };
                let bark_material = self.bark_material;
                let leaf_material = self.leaf_material;
                let bark_look = gpu::BarkLook::from_species(&self.grown.mesh);
                let mut leaf_params =
                    LeafMaterialParams::from_species(&self.grown.leaves, self.leaf_translucency);
                leaf_params.coverage_lod = self.coverage_lod;
                let draw_leaves = self.show_leaves && leaf_material.is_some();
                let cam = self.camera;
                let crown = self.crown;
                // The shadow is fitted to the tree, so it has to be fitted to where the
                // wind can take it, or a swaying crown runs off the edge of its own map.
                let aabb = {
                    let (lo, hi) = self.aabb;
                    let r = wind.reach();
                    ([lo[0] - r, lo[1], lo[2] - r], [hi[0] + r, hi[1] + r, hi[2] + r])
                };
                // A photograph with no sun to speak of, or a sun gone down, casts nothing.
                let shadows = self.shadows
                    && base_lighting.sun_color.max_element() > 0.0
                    && base_lighting.sun_dir.y > -0.1;
                let post = PostSettings {
                    tonemap: self.tonemap,
                    bloom: self.bloom,
                    ao: self.ao,
                    ao_radius: self.ao_radius,
                    ao_power: self.ao_strength,
                    samples: self.msaa,
                };
                let wire = self.wireframe && !cfg!(target_arch = "wasm32");
                let overlay = self.show_skeleton || self.show_grid;
                let use_normal_map = self.use_normal_map;
                let mode = match self.render_mode {
                    RenderMode::Shaded => 0,
                    RenderMode::UvChecker => 1,
                    RenderMode::Normals => 2,
                    RenderMode::Occlusion => 3,
                };
                let clip_rect = rect;

                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                        let gl = painter.gl();
                        let mesh_gpu = mesh_gpu.lock().unwrap();
                        let leaves_gpu = leaves_gpu.lock().unwrap();
                        let lines = lines.lock().unwrap();
                        let mut renderer = renderer.lock().unwrap();
                        let view_proj = cam.view_proj();
                        let mvp = view_proj.to_cols_array();
                        let ppp = info.pixels_per_point;
                        let sh = info.screen_size_px[1] as i32;
                        let x0 = clip_rect.min.x * ppp;
                        let y0_top = clip_rect.min.y * ppp;
                        let w = (clip_rect.width() * ppp).ceil() as i32;
                        let h = (clip_rect.height() * ppp).ceil() as i32;
                        let y0_gl = (sh as f32 - y0_top - h as f32).floor() as i32;
                        let clip = [x0.floor() as i32, y0_gl, w.max(1), h.max(1)];

                        renderer.prepare(gl, clip[2], clip[3], post.samples);
                        let frustum = shadow_frustum(aabb, base_lighting.sun_dir, shadow.size);
                        // The crown's shade on the sky belongs with the occlusion,
                        // and goes when it does.
                        let canopy = match (draw_leaves && post.ao, leaf_material) {
                            (true, Some(leaf_mat)) => unsafe {
                                renderer.canopy(gl, &leaves_gpu, &leaf_mat, leaf_params, &wind, crown)
                            },
                            _ => render::CanopyPlacement::default(),
                        };
                        let lighting = Lighting {
                            canopy,
                            shadow: frustum,
                            shadow_size: shadow.size,
                            // Enough to clear one shadow texel at a grazing angle, which
                            // is where a low sun puts everything.
                            normal_bias: frustum.texel * 1.6,
                            ao_texel: renderer.ao_texel(),
                            ..base_lighting
                        };
                        // Far enough out that the plane always meets the horizon,
                        // whatever the camera does.
                        let extent = (cam.distance + tree_height) * 12.0;

                        unsafe {
                            {
                                // The map is cleared whether or not it is drawn into:
                                // the lit passes sample it either way, and an
                                // uncleared depth texture reads as everything being
                                // in shadow.
                                shadow.bind(gl);
                                gl.viewport(0, 0, shadow.size, shadow.size);
                                // glClear obeys both the scissor box and the depth
                                // mask. egui has a scissor rect set when this callback
                                // runs, so without turning it off only the part of the
                                // shadow map under that rect is cleared and the rest
                                // keeps whatever was in it last frame. That stale
                                // depth reads as shadow, which is the hard-edged slab
                                // and the long straight bands lying across the ground.
                                gl.disable(glow::SCISSOR_TEST);
                                gl.disable(glow::BLEND);
                                gl.depth_mask(true);
                                gl.enable(glow::DEPTH_TEST);
                                gl.depth_func(glow::LEQUAL);
                                gl.clear_depth_f32(1.0);
                                gl.clear(glow::DEPTH_BUFFER_BIT);
                            }
                            if shadows {
                                gl.use_program(Some(depth_pass.program));
                                gl.uniform_matrix_4_f32_slice(
                                    Some(&depth_pass.u_light_view_proj),
                                    false,
                                    &frustum.view_proj.to_cols_array(),
                                );
                                wind.bind(gl, depth_pass.program);
                                mesh_gpu.bind_and_draw(gl);
                                gl.use_program(None);
                                // Leaves need their own alpha-tested depth pass or
                                // the canopy casts the shadow of its solid quads.
                                if let (true, Some(leaf_mat)) = (draw_leaves, leaf_material) {
                                    leaf_depth_pass.draw(
                                        gl,
                                        &leaves_gpu,
                                        frustum.view_proj,
                                        &leaf_mat,
                                        leaf_params,
                                        &wind,
                                    );
                                }
                            }

                            // The camera's own depth, the same passes drawn from the
                            // eye, for the ambient occlusion to read.
                            if post.ao {
                                renderer.begin_prepass(gl);
                                gl.use_program(Some(depth_pass.program));
                                gl.uniform_matrix_4_f32_slice(
                                    Some(&depth_pass.u_light_view_proj),
                                    false,
                                    &view_proj.to_cols_array(),
                                );
                                wind.bind(gl, depth_pass.program);
                                mesh_gpu.bind_and_draw(gl);
                                gl.use_program(None);
                                if let (true, Some(leaf_mat)) = (draw_leaves, leaf_material) {
                                    leaf_depth_pass.draw(
                                        gl,
                                        &leaves_gpu,
                                        view_proj,
                                        &leaf_mat,
                                        leaf_params,
                                        &wind,
                                    );
                                }
                                if ground.is_some() {
                                    ground_pass.draw_depth(gl, view_proj, cam.eye(), extent);
                                }
                                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                                renderer.ambient_occlusion(gl, cam.proj(), &post);
                            }

                            renderer.begin_scene(gl);
                            renderer.bind_lighting(gl, shadow.depth, post.ao);
                            if mode == 0 || mode == 3 {
                                if mode == 0 {
                                    backdrop.draw(gl, view_proj, &lighting);
                                } else {
                                    gl.clear_color(1.0, 1.0, 1.0, 1.0);
                                    gl.clear(glow::COLOR_BUFFER_BIT);
                                }
                                if let Some(ground) = ground {
                                    ground_pass.draw(
                                        gl,
                                        &GroundDrawParams {
                                            view_proj,
                                            lighting: &lighting,
                                            albedo: GROUND_ALBEDO,
                                            extent,
                                            mode: ground,
                                            view: mode,
                                        },
                                    );
                                }
                            } else {
                                // The debug views are shown as they are, not exposed,
                                // so they go on a plain grey.
                                gl.clear_color(0.32, 0.33, 0.35, 1.0);
                                gl.clear(glow::COLOR_BUFFER_BIT);
                            }

                            mesh_gpu.draw(
                                gl,
                                &MeshDrawParams {
                                    bark: bark_look,
                                    lighting: &lighting,
                                    view_proj,
                                    mode,
                                    use_normal_map,
                                    material: &bark_material,
                                    wind: &wind,
                                },
                            );

                            if let (true, Some(leaf_mat)) = (draw_leaves, leaf_material) {
                                leaves_gpu.draw(
                                    gl,
                                    &LeafDrawParams {
                                        lighting: &lighting,
                                        view_proj,
                                        mode,
                                        material: &leaf_mat,
                                        leaf: leaf_params,
                                        wind: &wind,
                                    },
                                );
                            }

                            if wire {
                                color_pass.draw_wire(
                                    gl,
                                    &mesh_gpu,
                                    draw_leaves.then_some(&*leaves_gpu),
                                    view_proj,
                                    [0.02, 0.02, 0.03, 1.0],
                                    &wind,
                                );
                            }
                            renderer.unbind_lighting(gl);

                            renderer.finish(gl, clip, &post, mode != 0);

                            if overlay {
                                lines.draw(gl, mvp, clip, false);
                            }

                            gl.disable(glow::SCISSOR_TEST);
                            gl.disable(glow::DEPTH_TEST);
                            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                            gl.active_texture(glow::TEXTURE0);
                        }
                    })),
                });
            });

        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbor_core::gltf::{ExportOptions, Format};
    use arbor_core::species::{builtin_presets, parse_template, ChildPattern};
    use arbor_core::Ranged;

    #[test]
    fn an_export_writes_its_trees_and_says_where() {
        // What the panel's Export buttons run, less the thread it runs on.
        let (_, src) = builtin_presets()
            .into_iter()
            .find(|(name, _)| *name == "fir_open")
            .expect("the open-grown fir is a built-in");
        let mut params = parse_template(src).unwrap();
        params.seed = 7;
        let dir = std::env::temp_dir().join(format!("arbor-viewer-export-{}", std::process::id()));
        let options = ExportOptions {
            textures: Some("../../assets/textures".into()),
            wind: true,
        };
        let one = export::export_trees(&dir, "fir", Format::Glb, &params, 1, &options, |_| true).unwrap();
        // Any texture it could not find would be reported after a semicolon.
        assert!(one.starts_with("exported") && !one.contains(';'), "{one}");
        assert_eq!(&std::fs::read(dir.join("fir.glb")).unwrap()[0..4], b"glTF");

        let mut heard = Vec::new();
        let three = export::export_trees(&dir, "fir", Format::Glb, &params, 3, &options, |done| {
            heard.push(done);
            true
        })
        .unwrap();
        assert!(three.starts_with("exported 3 trees"), "{three}");
        assert_eq!(heard, [1, 2, 3]);
        for seed in 7..10 {
            assert!(dir.join(format!("fir_seed{seed}.glb")).is_file());
        }

        let stopped = export::export_trees(&dir, "stop", Format::Glb, &params, 4, &options, |_| false).unwrap();
        assert!(stopped.starts_with("stopped after 1 of 4"), "{stopped}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every preset has to fit the panel that edits it.
    ///
    /// Walks the same tables the panel is drawn from, so every control is covered,
    /// and checks every level each preset declares. See the note at the top of
    /// `knobs`: the sliders no longer write a clamped value back over the species, but a
    /// range that does not reach a preset's value leaves a slider that cannot return to
    /// it once touched. A number the preset gives as a range has both its ends checked,
    /// since each has a handle of its own.
    #[test]
    fn species_values_fit_their_sliders() {
        use knobs::{Count, Group, Knob};

        fn check_value(bad: &mut Vec<String>, at: &str, v: Ranged, (lo, hi): (f32, f32)) {
            for end in [v.lo(), v.hi()] {
                if !(end >= lo && end <= hi) {
                    bad.push(format!("{at}: {end} outside {lo}..={hi}"));
                }
            }
        }
        fn check_knobs<T>(bad: &mut Vec<String>, at: &str, target: &mut T, knobs: &[Knob<T>]) {
            for k in knobs {
                check_value(bad, &format!("{at} {}", k.label), *(k.get)(target), k.range);
            }
        }
        fn check_counts<T>(bad: &mut Vec<String>, at: &str, target: &mut T, counts: &[Count<T>]) {
            for c in counts {
                let v = *(c.get)(target);
                if !(v >= c.range.0 && v <= c.range.1) {
                    bad.push(format!("{at} {}: {v} outside {}..={}", c.label, c.range.0, c.range.1));
                }
            }
        }
        fn check_groups<T>(bad: &mut Vec<String>, at: &str, target: &mut T, groups: &[Group<T>]) {
            for g in groups {
                let at = format!("{at} / {}", g.title);
                check_knobs(bad, &at, target, g.knobs);
                check_counts(bad, &at, target, g.counts);
            }
        }
        fn within(bad: &mut Vec<String>, at: &str, v: u32, (lo, hi): (u32, u32)) {
            if !(v >= lo && v <= hi) {
                bad.push(format!("{at}: {v} outside {lo}..={hi}"));
            }
        }

        let mut bad = Vec::new();
        for (name, src) in builtin_presets() {
            let mut p = parse_template(src).unwrap_or_else(|e| panic!("{name}: {e}"));
            check_knobs(&mut bad, name, &mut p, knobs::SHAPE);
            check_counts(&mut bad, name, &mut p, std::slice::from_ref(&knobs::SPLIT_DEPTH));
            within(&mut bad, &format!("{name} max_levels"), u32::from(p.max_levels), knobs::BRANCH_LEVELS);
            check_groups(&mut bad, name, &mut p.envelope, std::slice::from_ref(&knobs::ENVELOPE));
            for level in 0..knobs::level_count(&p) {
                let at = format!("{name} level {level}");
                if let Some(spawn) = knobs::spawn_mut(&mut p, level) {
                    match &mut spawn.pattern {
                        ChildPattern::None => {}
                        ChildPattern::Whorl { every, count } => {
                            within(&mut bad, &format!("{at} whorl every"), *every, knobs::WHORL_EVERY);
                            within(&mut bad, &format!("{at} per whorl"), *count, knobs::WHORL_COUNT);
                        }
                        ChildPattern::Continuous { density } => {
                            let at = format!("{at} density");
                            check_value(&mut bad, &at, *density, knobs::DENSITY);
                        }
                    }
                    check_groups(&mut bad, &format!("{at} spawn"), spawn, knobs::CHILDREN);
                }
                let stem = knobs::stem_mut(&mut p, level).expect("every offered level has params");
                check_groups(&mut bad, &at, stem, knobs::STEM);
            }
            check_knobs(&mut bad, &format!("{name} foliage"), &mut p.leaves, knobs::FOLIAGE);
            check_groups(&mut bad, &format!("{name} foliage"), &mut p.leaves, knobs::FOLIAGE_MORE);
            within(&mut bad, &format!("{name} leaf min_level"), u32::from(p.leaves.min_level), knobs::LEAF_MIN_LEVEL);
            check_groups(&mut bad, &format!("{name} bark"), &mut p.mesh, knobs::BARK);
            check_groups(&mut bad, &format!("{name} bark"), &mut p.mesh.irregularity, knobs::IRREGULARITY);
            check_groups(&mut bad, &format!("{name} wind"), &mut p.wind, std::slice::from_ref(&knobs::WIND));
        }
        assert!(bad.is_empty(), "preset values the panel cannot reach:
  {}", bad.join("
  "));
    }
}
