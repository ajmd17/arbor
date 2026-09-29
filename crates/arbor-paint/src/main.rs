//! A texture painter on `arbor-render`. So far: a lit viewport, an orbit camera, and the
//! lighting and view controls a model is inspected with.

mod camera;
mod shapes;

use std::sync::{Arc, Mutex};

use arbor_render::environment::{sun_direction, Environment, EnvironmentSettings};
use arbor_render::hdri;
use arbor_render::lighting::SkyParams;
use arbor_render::pbr::{
    AlphaMode, PbrFrame, PbrItem, PbrMaterial, PbrMesh, PbrRenderer, ViewMode,
};
use arbor_render::render::{PostSettings, Tonemap};
use eframe::egui;
use glam::{Mat4, Vec3};

use camera::OrbitCamera;

const SHADOW_SIZE: i32 = 2048;

/// One mesh in the scene, with its material and where it stands.
struct Object {
    name: String,
    mesh: PbrMesh,
    material: PbrMaterial,
    model: Mat4,
    visible: bool,
    /// The ground: left out when framing the camera, and switched separately.
    is_ground: bool,
}

/// What the paint callback needs, behind one lock.
struct GpuScene {
    renderer: PbrRenderer,
    environment: Environment,
    objects: Vec<Object>,
}

impl GpuScene {
    fn new(gl: &glow::Context) -> Self {
        let mut objects = Vec::new();
        let sphere = shapes::sphere(64);
        objects.push(Object {
            name: "Sphere".into(),
            mesh: PbrMesh::new(gl, &sphere.data()).expect("sphere"),
            material: PbrMaterial {
                base_color: [0.8, 0.8, 0.8, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            },
            model: Mat4::from_translation(Vec3::Y),
            visible: true,
            is_ground: false,
        });
        let floor = shapes::floor();
        objects.push(Object {
            name: "Ground".into(),
            mesh: PbrMesh::new(gl, &floor.data()).expect("floor"),
            material: PbrMaterial {
                base_color: [0.42, 0.42, 0.4, 1.0],
                metallic: 0.0,
                roughness: 0.9,
                alpha_mode: AlphaMode::Opaque,
                double_sided: true,
                ..Default::default()
            },
            model: Mat4::from_scale(Vec3::splat(3.0)),
            visible: true,
            is_ground: true,
        });
        Self {
            renderer: PbrRenderer::new(gl, SHADOW_SIZE),
            environment: Environment::new(),
            objects,
        }
    }

    /// The box round what is being painted, leaving the ground out.
    fn model_bounds(&self) -> Option<(Vec3, Vec3)> {
        let items: Vec<PbrItem> = self
            .objects
            .iter()
            .filter(|o| !o.is_ground)
            .map(|o| PbrItem { mesh: &o.mesh, material: &o.material, model: o.model })
            .collect();
        PbrRenderer::scene_bounds(&items)
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
    hdris: Vec<String>,
    camera: OrbitCamera,
    /// A problem to show, and whether it is one.
    status: Option<String>,
    frames: u32,
    shot: Option<String>,
}

impl App {
    fn new(gl: Arc<glow::Context>, shot: Option<String>, mode: ViewMode) -> Self {
        let scene = GpuScene::new(&gl);
        let mut camera = OrbitCamera::default();
        if let Some((lo, hi)) = scene.model_bounds() {
            camera.frame(lo, hi);
        }
        let mut settings = Settings::new();
        settings.mode = mode;
        Self {
            gl,
            scene: Arc::new(Mutex::new(scene)),
            settings,
            hdris: hdri::available(),
            camera,
            status: None,
            frames: 0,
            shot,
        }
    }

    fn frame_model(&mut self) {
        if let Some((lo, hi)) = self.scene.lock().unwrap().model_bounds() {
            self.camera.frame(lo, hi);
        }
    }

    /// Builds the environment maps from whatever is chosen, when that has changed.
    fn sync_environment(&mut self) {
        let mut scene = self.scene.lock().unwrap();
        let scene = &mut *scene;
        let maps = &mut scene.renderer.renderer.env;
        if let Some(name) = self.settings.environment.clone() {
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
                    ui.add_enabled(false, egui::Button::new("Open model…"))
                        .on_disabled_hover_text("Loading glTF models comes next");
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
        egui::SidePanel::left("scene").default_width(220.0).show(ctx, |ui| {
            ui.heading("Scene");
            ui.separator();
            let mut scene = self.scene.lock().unwrap();
            for object in scene.objects.iter_mut().filter(|o| !o.is_ground) {
                ui.checkbox(&mut object.visible, &object.name);
            }
            ui.checkbox(&mut self.settings.ground, "Ground");
            ui.add_space(12.0);
            ui.heading("Layers");
            ui.separator();
            ui.weak("Texture layers arrive with painting.");
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
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let scene = self.scene.lock().unwrap();
                let tris: usize = scene
                    .objects
                    .iter()
                    .filter(|o| o.visible && (!o.is_ground || self.settings.ground))
                    .map(|o| o.mesh.triangle_count())
                    .sum();
                ui.label(format!("{tris} triangles"));
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
            let (mode, wireframe, background, ground, post) = (s.mode, s.wireframe, s.background, s.ground, s.post);
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
                    let items: Vec<PbrItem> = scene
                        .objects
                        .iter()
                        .filter(|o| o.visible && (!o.is_ground || ground))
                        .map(|o| PbrItem { mesh: &o.mesh, material: &o.material, model: o.model })
                        .collect();
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
        if self.frames == 10 {
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
        Box::new(move |cc| Ok(Box::new(App::new(cc.gl.clone().expect("a glow context"), shot, mode)))),
    )
    .expect("run");
}
