//! A texture painter on `arbor-render`. So far: a lit viewport, an orbit camera, and the
//! lighting and view controls a model is inspected with.

mod bake;
mod bake_ui;
mod camera;
mod import;
mod ktx;
mod model;
mod shapes;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

use arbor_render::environment::{sun_direction, Environment, EnvironmentSettings};
use arbor_render::hdri;
use arbor_render::lighting::SkyParams;
use arbor_render::pbr::{AlphaMode, PbrFrame, PbrItem, PbrMaterial, PbrMesh, PbrRenderer, ViewMode};
use arbor_render::render::{PostSettings, Tonemap};
use eframe::egui;
use glam::{Mat4, Vec3};

use bake_ui::Baker;
use camera::OrbitCamera;
use import::ImportedModel;
use model::Model;

const SHADOW_SIZE: i32 = 2048;

/// What the paint callback needs, behind one lock.
struct GpuScene {
    renderer: PbrRenderer,
    environment: Environment,
    model: Model,
    ground: PbrMesh,
    ground_material: PbrMaterial,
    ground_model: Mat4,
}

impl GpuScene {
    fn new(gl: &glow::Context) -> Self {
        let floor = shapes::floor();
        Self {
            renderer: PbrRenderer::new(gl, SHADOW_SIZE),
            environment: Environment::new(),
            model: Model::upload(gl, Arc::new(placeholder())).expect("placeholder model"),
            ground: PbrMesh::new(gl, &floor.data()).expect("floor"),
            ground_material: PbrMaterial {
                base_color: [0.42, 0.42, 0.4, 1.0],
                metallic: 0.0,
                roughness: 0.9,
                alpha_mode: AlphaMode::Opaque,
                double_sided: true,
                ..Default::default()
            },
            ground_model: Mat4::from_scale(Vec3::splat(3.0)),
        }
    }

    /// Swaps in a new model, lets go of the old one, and stands the ground under it.
    fn set_model(&mut self, gl: &glow::Context, model: Model) {
        self.model.delete(gl);
        self.model = model;
        if let Some((lo, hi)) = self.model.bounds() {
            // Wide enough to catch the shadow, at the model's feet.
            let half = ((hi - lo).max_element() * 1.5).max(0.5);
            let centre = (lo + hi) * 0.5;
            self.ground_model =
                Mat4::from_translation(Vec3::new(centre.x, lo.y, centre.z)) * Mat4::from_scale(Vec3::splat(half));
        }
    }
}

/// What stands in until a file is opened.
fn placeholder() -> ImportedModel {
    let sphere = shapes::sphere(64);
    ImportedModel {
        name: "Sphere".into(),
        primitives: vec![import::Primitive {
            name: "Sphere".into(),
            positions: sphere.positions,
            normals: sphere.normals,
            uvs: Some(sphere.uvs),
            tangents: Some(sphere.tangents),
            colors: None,
            indices: sphere.indices,
            material: 0,
        }],
        materials: vec![import::Material {
            name: "Grey".into(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            mask: false,
            blend: false,
            alpha_cutoff: 0.5,
            double_sided: false,
            textures: vec![],
        }],
        images: vec![],
        instances: vec![import::Instance {
            name: "Sphere".into(),
            primitive: 0,
            transform: Mat4::from_translation(Vec3::Y),
        }],
        warnings: vec![],
    }
}

/// Every control, apart from the camera.
struct Settings {
    mode: ViewMode,
    wireframe: bool,
    background: bool,
    ground: bool,
    post: PostSettings,
    /// The photograph's name, or `None` for the procedural sky.
    environment: Option<String>,
    sun_elevation: f32,
    sun_azimuth: f32,
    sun_intensity: f32,
    env: EnvironmentSettings,
}

impl Settings {
    fn new() -> Self {
        Self {
            mode: ViewMode::Lit,
            wireframe: false,
            background: true,
            ground: true,
            post: PostSettings::default(),
            environment: Some(hdri::BUNDLED[0].to_string()),
            sun_elevation: 35.0,
            sun_azimuth: 40.0,
            sun_intensity: 1.0,
            env: EnvironmentSettings::new(SkyParams::dawn()),
        }
    }

    fn sky(&self) -> SkyParams {
        let mut sky = SkyParams::for_sun(sun_direction(self.sun_elevation, self.sun_azimuth));
        sky.sun_color *= self.sun_intensity;
        sky
    }
}

struct App {
    gl: Arc<glow::Context>,
    scene: Arc<Mutex<GpuScene>>,
    settings: Settings,
    baker: Baker,
    hdris: Vec<String>,
    camera: OrbitCamera,
    /// A problem to show.
    status: Option<String>,
    /// A file being read on another thread, and its name.
    loading: Option<(String, Receiver<Result<ImportedModel, String>>)>,
    frames: u32,
    shot: Option<String>,
    shot_requested: bool,
}

impl App {
    fn new(gl: Arc<glow::Context>, shot: Option<String>, mode: ViewMode, open: Option<PathBuf>, bake_view: Option<bake::MapKind>) -> Self {
        let scene = GpuScene::new(&gl);
        let mut camera = OrbitCamera::default();
        if let Some((lo, hi)) = scene.model.bounds() {
            camera.frame(lo, hi);
        }
        let mut settings = Settings::new();
        settings.mode = mode;
        let mut app = Self {
            gl,
            scene: Arc::new(Mutex::new(scene)),
            settings,
            baker: Baker::new(),
            hdris: hdri::available(),
            camera,
            status: None,
            loading: None,
            frames: 0,
            shot,
            shot_requested: false,
        };
        if let Some(kind) = bake_view {
            app.baker.bake_on_load(kind);
        }
        if let Some(path) = open {
            app.open(&path);
        }
        app
    }

    fn frame_model(&mut self) {
        if let Some((lo, hi)) = self.scene.lock().unwrap().model.bounds() {
            self.camera.frame(lo, hi);
        }
    }

    /// Starts reading a glTF or GLB on another thread, so a big file does not freeze the
    /// window. [`Self::poll_loading`] picks up the result.
    fn open(&mut self, path: &Path) {
        let name = path.file_name().map_or("model".into(), |n| n.to_string_lossy().to_string());
        let (tx, rx) = mpsc::channel();
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let _ = tx.send(import::load(&path));
        });
        self.status = None;
        self.loading = Some((name, rx));
    }

    fn choose_file(&mut self) {
        let picked = rfd::FileDialog::new().add_filter("glTF", &["gltf", "glb"]).pick_file();
        if let Some(path) = picked {
            self.open(&path);
        }
    }

    /// Takes a finished read, if there is one, and puts the model on the GPU.
    fn poll_loading(&mut self, ctx: &egui::Context) {
        let Some((name, rx)) = &self.loading else { return };
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
                return;
            }
            Err(TryRecvError::Disconnected) => Err("the loader stopped unexpectedly".into()),
        };
        let name = name.clone();
        self.loading = None;
        match result.and_then(|imported| Model::upload(&self.gl, Arc::new(imported))) {
            Ok(model) => {
                self.baker.reset();
                self.scene.lock().unwrap().set_model(&self.gl, model);
                self.frame_model();
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("Arbor Paint - {name}")));
            }
            Err(e) => self.status = Some(format!("{name}: {e}")),
        }
    }

    /// Builds the environment maps from whatever is chosen, when that has changed.
    fn sync_environment(&mut self) {
        let mut scene = self.scene.lock().unwrap();
        let scene = &mut *scene;
        let maps = &mut scene.renderer.renderer.env;
        if let Some(name) = self.settings.environment.clone() {
            // Already built from it: not worth reading the file again every frame.
            if scene.environment.photo_name() == Some(name.as_str()) {
                return;
            }
            let path = format!("{}/{name}.hdr", hdri::HDRI_DIR);
            let loaded = std::fs::read(&path)
                .map_err(|e| format!("{path}: {e}"))
                .and_then(|bytes| scene.environment.load_photo(&self.gl, maps, &name, &bytes));
            match loaded {
                Ok(()) => return,
                Err(e) => {
                    // Back to the procedural sky, rather than a model lit by nothing.
                    self.status = Some(format!("{name}: {e}"));
                    self.settings.environment = None;
                }
            }
        }
        let sky = self.settings.sky();
        scene.environment.load_sky(&self.gl, maps, &sky);
    }

    fn menu(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open model…").clicked() {
                        ui.close();
                        self.choose_file();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Frame model  (F)").clicked() {
                        self.frame_model();
                        ui.close();
                    }
                });
            });
        });
    }

    fn scene_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("scene").default_width(240.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut scene = self.scene.lock().unwrap();
                let model = &mut scene.model;
                ui.heading(&model.name);
                let s = &model.stats;
                ui.weak(format!(
                    "{} triangles, {} vertices, {:.1} MB of textures",
                    s.triangles, s.vertices, s.texture_mb
                ));
                if let Some((lo, hi)) = model.bounds() {
                    let d = hi - lo;
                    ui.weak(format!("{:.3} x {:.3} x {:.3}", d.x, d.y, d.z));
                }
                for w in &model.warnings {
                    ui.colored_label(egui::Color32::from_rgb(220, 170, 60), format!("⚠ {w}"));
                }
                ui.separator();

                ui.strong("Objects");
                for instance in &mut model.instances {
                    ui.checkbox(&mut instance.visible, &instance.name);
                }
                ui.checkbox(&mut self.settings.ground, "Ground");

                ui.add_space(8.0);
                ui.strong("Materials");
                for info in model.material_info.iter().filter(|i| i.used) {
                    let maps: Vec<String> = info.textures.iter().map(|(slot, w, h)| format!("{slot:?} {w}x{h}")).collect();
                    let header = egui::CollapsingHeader::new(&info.name).id_salt(&info.name);
                    header.show(ui, |ui| {
                        if maps.is_empty() {
                            ui.weak("no textures");
                        }
                        for m in maps {
                            ui.label(m);
                        }
                    });
                }

                ui.add_space(8.0);
                ui.strong("Layers");
                ui.weak("Texture layers arrive with painting.");
            });
        });
    }

    fn properties_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("properties").default_width(260.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let s = &mut self.settings;
                ui.heading("View");
                egui::ComboBox::from_label("Channel").selected_text(s.mode.label()).show_ui(ui, |ui| {
                    for m in ViewMode::ALL {
                        ui.selectable_value(&mut s.mode, m, m.label());
                    }
                });
                ui.checkbox(&mut s.wireframe, "Wireframe");
                ui.checkbox(&mut s.background, "Show environment");

                ui.add_space(8.0);
                ui.heading("Lighting");
                let current = s.environment.clone().unwrap_or_else(|| "Procedural sky".into());
                egui::ComboBox::from_label("Environment").selected_text(&current).show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.environment, None, "Procedural sky");
                    for name in &self.hdris {
                        ui.selectable_value(&mut s.environment, Some(name.clone()), name);
                    }
                });
                if s.environment.is_none() {
                    ui.add(egui::Slider::new(&mut s.sun_elevation, -5.0..=90.0).text("Sun elevation"));
                    ui.add(egui::Slider::new(&mut s.sun_azimuth, 0.0..=360.0).text("Sun azimuth"));
                    ui.add(egui::Slider::new(&mut s.sun_intensity, 0.0..=3.0).text("Sun strength"));
                } else {
                    ui.add(egui::Slider::new(&mut s.env.rotation_deg, 0.0..=360.0).text("Rotation"));
                    ui.add(egui::Slider::new(&mut s.env.intensity, 0.0..=4.0).text("Intensity"));
                    ui.add(egui::Slider::new(&mut s.env.photo_sun, 0.0..=3.0).text("Sun strength"));
                }
                ui.add(egui::Slider::new(&mut s.env.exposure_ev, -4.0..=4.0).text("Exposure (EV)"));
                ui.add(egui::Slider::new(&mut s.env.shadow_softness, 0.0..=8.0).text("Shadow softness"));

                ui.add_space(8.0);
                ui.heading("Post");
                egui::ComboBox::from_label("Tonemap").selected_text(s.post.tonemap.label()).show_ui(ui, |ui| {
                    for t in [Tonemap::Neutral, Tonemap::Agx, Tonemap::AgxPunchy, Tonemap::Aces] {
                        ui.selectable_value(&mut s.post.tonemap, t, t.label());
                    }
                });
                ui.add(egui::Slider::new(&mut s.post.bloom, 0.0..=0.3).text("Bloom"));
                ui.checkbox(&mut s.post.ao, "Ambient occlusion");
                if s.post.ao {
                    ui.add(egui::Slider::new(&mut s.post.ao_radius, 0.05..=3.0).text("AO radius"));
                    ui.add(egui::Slider::new(&mut s.post.ao_power, 0.5..=3.0).text("AO strength"));
                }
                egui::ComboBox::from_label("MSAA").selected_text(format!("{}x", s.post.samples.max(1))).show_ui(ui, |ui| {
                    for n in [0, 2, 4, 8] {
                        ui.selectable_value(&mut s.post.samples, n, format!("{}x", n.max(1)));
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                let mut scene = self.scene.lock().unwrap();
                self.baker.ui(ui, &mut scene.model, &self.gl);
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some((name, _)) = &self.loading {
                    ui.spinner();
                    ui.label(format!("Loading {name}…"));
                } else {
                    let scene = self.scene.lock().unwrap();
                    let tris: usize = scene.model.items().iter().map(|i| i.mesh.triangle_count()).sum();
                    ui.label(format!("{tris} triangles"));
                }
                if let Some(status) = &self.status {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_RED, status);
                }
            });
        });
    }

    fn viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let drag = response.drag_delta();
            let drag = glam::Vec2::new(drag.x, drag.y);
            let pan = response.dragged_by(egui::PointerButton::Middle)
                || response.dragged_by(egui::PointerButton::Secondary)
                || (response.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.shift));
            if pan {
                self.camera.pan(drag, rect.height());
            } else if response.dragged_by(egui::PointerButton::Primary) {
                self.camera.orbit(drag);
            }
            if response.hovered() {
                let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                self.camera.zoom(scroll);
                if pinch != 1.0 {
                    self.camera.distance = (self.camera.distance / pinch).clamp(0.01, 10_000.0);
                }
                if ui.input(|i| i.key_pressed(egui::Key::F)) {
                    self.frame_model();
                }
            }

            let cam = self.camera;
            let aspect = rect.aspect_ratio();
            let s = &self.settings;
            let (mode, wireframe, background, ground, post) =
                (bake_ui::effective_mode(&self.baker, s.mode), s.wireframe, s.background, s.ground, s.post);
            let mut env = s.env;
            env.sky = s.sky();
            let scene = self.scene.clone();

            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                    let gl = painter.gl();
                    let ppp = info.pixels_per_point;
                    let h = (rect.height() * ppp).ceil() as i32;
                    let clip = [
                        (rect.min.x * ppp).floor() as i32,
                        (info.screen_size_px[1] as f32 - rect.min.y * ppp - h as f32).floor() as i32,
                        ((rect.width() * ppp).ceil() as i32).max(1),
                        h.max(1),
                    ];
                    let mut scene = scene.lock().unwrap();
                    let scene = &mut *scene;
                    let mut items = scene.model.items();
                    if ground {
                        items.push(PbrItem {
                            mesh: &scene.ground,
                            material: &scene.ground_material,
                            model: scene.ground_model,
                        });
                    }
                    let lighting = scene.environment.lighting(&env, cam.eye());
                    unsafe {
                        scene.renderer.draw(
                            gl,
                            &PbrFrame {
                                items: &items,
                                proj: cam.proj(aspect),
                                view_proj: cam.view_proj(aspect),
                                lighting,
                                post,
                                clip,
                                mode,
                                wireframe,
                                background,
                            },
                        );
                    }
                })),
            });
        });
    }

    /// Saves the window to a file and quits, for checking the app from a script.
    fn screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = &self.shot else { return };
        // Not before a file being opened has arrived.
        if self.frames >= 10 && self.loading.is_none() && !self.baker.running() && !self.baker.test_pending() && !self.shot_requested {
            self.shot_requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        let image = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let [w, h] = img.size;
            let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            image::save_buffer(path, &bytes, w as u32, h as u32, image::ColorType::Rgba8).expect("save screenshot");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.frames += 1;
        let dropped = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.open(&path);
        }
        self.poll_loading(ctx);
        self.baker.poll(ctx);
        if self.loading.is_none() {
            let mut scene = self.scene.lock().unwrap();
            self.baker.test_step(&mut scene.model, &self.gl);
        }
        self.sync_environment();
        self.menu(ctx);
        self.status_bar(ctx);
        self.scene_panel(ctx);
        self.properties_panel(ctx);
        self.viewport(ctx);
        self.screenshot(ctx);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let mode = flag("--mode")
        .and_then(|m| m.parse::<usize>().ok())
        .and_then(|i| ViewMode::ALL.get(i).copied())
        .unwrap_or_default();
    let shot = flag("--shot");
    let bake_view = flag("--bake-view").and_then(|i| i.parse::<usize>().ok()).and_then(|i| bake::MapKind::ALL.get(i).copied());
    // A path on its own, not the value of a flag.
    let open = args
        .iter()
        .enumerate()
        .skip(1)
        .find(|(i, a)| !a.starts_with("--") && !args[i - 1].starts_with("--"))
        .map(|(_, a)| PathBuf::from(a));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 900.0]).with_title("Arbor Paint"),
        // The scene is drawn into targets of its own, so the window needs neither depth
        // nor samples.
        depth_buffer: 0,
        multisampling: 0,
        ..Default::default()
    };
    eframe::run_native(
        "arbor-paint",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc.gl.clone().expect("a glow context"), shot, mode, open, bake_view)))),
    )
    .expect("run");
}
