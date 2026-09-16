#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod gpu;
mod shaders;

use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::glow;
use eframe::glow::HasContext;
use glam::{Mat4, Vec3};

use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_mesh, grow, Skeleton, SkeletonStats, SpeciesParams};

use gpu::{ColorPass, DepthPass, GpuLines, GpuMesh, MeshDrawParams, ShadowTarget};

#[derive(Clone, Copy, PartialEq)]
enum RenderMode {
    Shaded,
    UvChecker,
    Normals,
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 950.0])
            .with_title("Arbor"),
        depth_buffer: 24,
        ..Default::default()
    };
    eframe::run_native("arbor", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
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

    fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect.max(0.05), 0.05, 600.0);
        proj * view
    }
}

fn light_view_proj(aabb: ([f32; 3], [f32; 3]), sun_dir: Vec3) -> Mat4 {
    let min = Vec3::from(aabb.0);
    let max = Vec3::from(aabb.1);
    let center = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5 + 1.0;
    let eye = center + sun_dir.normalize_or(Vec3::Y) * (radius * 3.0);
    let up = if sun_dir.y.abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, center, up);
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 7.0);
    proj * view
}

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
    shadow: Arc<ShadowTarget>,
    depth_pass: Arc<DepthPass>,
    color_pass: Arc<ColorPass>,
    material: Arc<gpu::MaterialTextures>,
    presets: Vec<(&'static str, &'static str)>,
    preset_index: usize,
    params: SpeciesParams,
    skeleton: Skeleton,
    stats: SkeletonStats,
    mesh_stats: (usize, usize),
    aabb: ([f32; 3], [f32; 3]),
    camera: OrbitCamera,
    dirty: bool,
    gen_ms: f32,
    render_mode: RenderMode,
    wireframe: bool,
    shadows: bool,
    show_skeleton: bool,
    show_grid: bool,
    use_normal_map: bool,
    sun_azimuth: f32,
    sun_elevation: f32,
    auto_frame: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.clone().expect("eframe must run with the glow renderer");
        let presets = builtin_presets();
        let params = parse_species(presets[0].1).expect("embedded pine preset parses");
        let skeleton = grow(&params);
        let mesh = build_mesh(&skeleton, &params);
        let aabb = mesh.aabb();
        let mesh_stats = (mesh.vertex_count(), mesh.triangle_count());

        let mesh_gpu = GpuMesh::new(&gl);
        let mut mesh_gpu = mesh_gpu;
        mesh_gpu.upload(&gl, &mesh);
        let lines = GpuLines::new(&gl);
        let shadow = ShadowTarget::new(&gl, 2048);
        let depth_pass = DepthPass::new(&gl);
        let color_pass = ColorPass::new(&gl);
        let material = unsafe { gpu::load_material(&gl, "assets/textures") };

        let mut app = Self {
            gl: Arc::clone(&gl),
            lines: Arc::new(Mutex::new(lines)),
            mesh_gpu: Arc::new(Mutex::new(mesh_gpu)),
            shadow: Arc::new(shadow),
            depth_pass: Arc::new(depth_pass),
            color_pass: Arc::new(color_pass),
            material: Arc::new(material),
            presets,
            preset_index: 0,
            params,
            skeleton,
            stats: SkeletonStats::default(),
            mesh_stats,
            aabb,
            camera: OrbitCamera {
                target: Vec3::new(0.0, 6.0, 0.0),
                distance: 26.0,
                yaw: 0.6,
                pitch: 0.22,
                fov_y: 50.0f32.to_radians(),
                aspect: 1.6,
            },
            dirty: true,
            gen_ms: 0.0,
            render_mode: RenderMode::Shaded,
            wireframe: false,
            shadows: true,
            show_skeleton: false,
            show_grid: true,
            use_normal_map: true,
            sun_azimuth: 40.0,
            sun_elevation: 45.0,
            auto_frame: true,
        };
        app.stats = app.skeleton.stats();
        app.rebuild_overlay();
        app
    }

    fn regenerate(&mut self) {
        let t = std::time::Instant::now();
        self.skeleton = grow(&self.params);
        let mesh = build_mesh(&self.skeleton, &self.params);
        self.aabb = mesh.aabb();
        self.mesh_stats = (mesh.vertex_count(), mesh.triangle_count());
        self.mesh_gpu.lock().unwrap().upload(&self.gl, &mesh);
        self.gen_ms = t.elapsed().as_secs_f32() * 1000.0;
        self.stats = self.skeleton.stats();
        self.rebuild_overlay();
        if self.auto_frame {
            self.frame_camera();
            self.auto_frame = false;
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

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Arbor");
        ui.separator();

        let mut preset_changed = false;
        egui::ComboBox::from_label("Preset")
            .selected_text(self.presets[self.preset_index].0)
            .show_ui(ui, |ui| {
                for (i, (name, _)) in self.presets.iter().enumerate() {
                    if ui
                        .selectable_value(&mut self.preset_index, i, *name)
                        .changed()
                    {
                        preset_changed = true;
                    }
                }
            });
        if preset_changed {
            let src = self.presets[self.preset_index].1;
            match parse_species(src) {
                Ok(mut p) => {
                    p.seed = self.params.seed;
                    self.params = p;
                    self.dirty = true;
                }
                Err(e) => {
                    ui.colored_label(egui::Color32::RED, format!("preset error: {e}"));
                }
            }
        }

        ui.horizontal(|ui| {
            ui.label("Seed");
            if ui
                .add(egui::DragValue::new(&mut self.params.seed).speed(1.0))
                .changed()
            {
                self.dirty = true;
            }
            if ui.button("Random").clicked() {
                use rand::Rng;
                self.params.seed = rand::rng().random();
                self.dirty = true;
            }
        });

        ui.separator();
        ui.label("Shape");
        shape_slider(ui, &mut self.params.trunk.length, 2.0..=30.0, "Trunk length", &mut self.dirty);
        shape_slider(ui, &mut self.params.envelope_scale, 0.4..=2.5, "Envelope scale", &mut self.dirty);
        shape_slider(ui, &mut self.params.gravity_multiplier, 0.0..=4.0, "Gravity", &mut self.dirty);
        shape_slider(ui, &mut self.params.phototropism_multiplier, 0.0..=4.0, "Phototropism", &mut self.dirty);

        let mut levels = i32::from(self.params.max_levels);
        if ui
            .add(egui::Slider::new(&mut levels, 1..=4).text("Branch levels"))
            .changed()
        {
            self.params.max_levels = levels as u8;
            self.dirty = true;
        }

        ui.separator();
        ui.label("Render");
        egui::ComboBox::from_label("View")
            .selected_text(match self.render_mode {
                RenderMode::Shaded => "Shaded",
                RenderMode::UvChecker => "UV checker",
                RenderMode::Normals => "Normals",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.render_mode, RenderMode::Shaded, "Shaded");
                ui.selectable_value(&mut self.render_mode, RenderMode::UvChecker, "UV checker");
                ui.selectable_value(&mut self.render_mode, RenderMode::Normals, "Normals");
            });
        ui.checkbox(&mut self.wireframe, "Wireframe");
        ui.checkbox(&mut self.shadows, "Shadows");
        ui.checkbox(&mut self.use_normal_map, "Normal map");
        if ui.checkbox(&mut self.show_grid, "Grid").changed() {
            self.rebuild_overlay();
        }
        if ui.checkbox(&mut self.show_skeleton, "Skeleton").changed() {
            self.rebuild_overlay();
        }
        ui.add(egui::Slider::new(&mut self.sun_azimuth, 0.0..=360.0).text("Sun azimuth"));
        ui.add(egui::Slider::new(&mut self.sun_elevation, 5.0..=85.0).text("Sun elevation"));

        ui.separator();
        ui.label("Stats");
        let s = &self.stats;
        ui.monospace(format!("nodes:  {}", s.node_count));
        ui.monospace(format!("segs:   {}", s.segment_count));
        ui.monospace(format!("tips:   {}", s.tip_count));
        ui.monospace(format!("height: {:.1} m", s.height));
        ui.monospace(format!("verts:  {}", self.mesh_stats.0));
        ui.monospace(format!("tris:   {}", self.mesh_stats.1));
        ui.monospace(format!("gen:    {:.2} ms", self.gen_ms));

        if ui.button("Frame tree").clicked() {
            self.frame_camera();
        }

        ui.separator();
        ui.label("Camera: LMB orbit, RMB/MMB pan, wheel zoom");
        let _ = ctx;
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

fn shape_slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    text: &str,
    dirty: &mut bool,
) {
    if ui.add(egui::Slider::new(value, range).text(text)).changed() {
        *dirty = true;
    }
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
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let screen = ctx.screen_rect();
        self.camera.aspect = screen.width() / screen.height().max(1.0);

        if self.dirty {
            self.regenerate();
            self.dirty = false;
        }

        egui::SidePanel::left("controls")
            .default_width(300.0)
            .show(ctx, |ui| self.controls(ui, ctx));

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let rect = ui.available_rect_before_wrap();
                let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                self.camera_input(&resp, ctx);

                let mesh_gpu = Arc::clone(&self.mesh_gpu);
                let lines = Arc::clone(&self.lines);
                let shadow = Arc::clone(&self.shadow);
                let depth_pass = Arc::clone(&self.depth_pass);
                let color_pass = Arc::clone(&self.color_pass);
                let material = Arc::clone(&self.material);
                let cam = self.camera;
                let aabb = self.aabb;
                let sun_dir = self.sun_dir();
                let shadows = self.shadows;
                let wire = self.wireframe;
                let overlay = self.show_skeleton || self.show_grid;
                let use_normal_map = self.use_normal_map;
                let mode = match self.render_mode {
                    RenderMode::Shaded => 0,
                    RenderMode::UvChecker => 1,
                    RenderMode::Normals => 2,
                };
                let clip_rect = rect;

                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                        let gl = painter.gl();
                        let mesh_gpu = mesh_gpu.lock().unwrap();
                        let lines = lines.lock().unwrap();
                        let view_proj = cam.view_proj();
                        let mvp = view_proj.to_cols_array();
                        let ppp = info.pixels_per_point;
                        let size_px = info.screen_size_px;
                        let sw = size_px[0] as i32;
                        let sh = size_px[1] as i32;
                        let x0 = clip_rect.min.x * ppp;
                        let y0_top = clip_rect.min.y * ppp;
                        let w = (clip_rect.width() * ppp).ceil() as i32;
                        let h = (clip_rect.height() * ppp).ceil() as i32;
                        let y0_gl = (sh as f32 - y0_top - h as f32).floor() as i32;
                        let clip = [x0.floor() as i32, y0_gl, w.max(1), h.max(1)];

                        unsafe {
                            if shadows {
                                shadow.bind(gl);
                                gl.viewport(0, 0, shadow.size, shadow.size);
                                gl.enable(glow::DEPTH_TEST);
                                gl.depth_func(glow::LEQUAL);
                                gl.clear_depth_f32(1.0);
                                gl.clear(glow::DEPTH_BUFFER_BIT);
                                gl.use_program(Some(depth_pass.program));
                                let lvp = light_view_proj(aabb, sun_dir);
                                gl.uniform_matrix_4_f32_slice(
                                    Some(&depth_pass.u_light_view_proj),
                                    false,
                                    &lvp.to_cols_array(),
                                );
                                mesh_gpu.bind_and_draw(gl);
                                gl.use_program(None);
                                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                            }

                            gl.viewport(0, 0, sw.max(1), sh.max(1));
                            gl.enable(glow::SCISSOR_TEST);
                            gl.scissor(clip[0], clip[1], clip[2], clip[3]);
                            gl.clear_color(0.52, 0.65, 0.84, 1.0);
                            gl.enable(glow::DEPTH_TEST);
                            gl.depth_func(glow::LEQUAL);
                            gl.clear_depth_f32(1.0);
                            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

                            let draw_params = MeshDrawParams {
                                view_proj,
                                light_view_proj: light_view_proj(aabb, sun_dir),
                                cam_pos: cam.eye(),
                                sun_dir,
                                sun_color: Vec3::new(2.4, 2.25, 2.05),
                                mode,
                                use_normal_map,
                                material: &material,
                                shadow_depth: shadow.depth,
                            };
                            mesh_gpu.draw(gl, &draw_params);

                            if wire {
                                color_pass.draw_wire(
                                    gl,
                                    &mesh_gpu,
                                    view_proj,
                                    [0.02, 0.02, 0.03, 1.0],
                                );
                            }

                            if overlay {
                                lines.draw(gl, mvp, clip, [sw, sh], false);
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
