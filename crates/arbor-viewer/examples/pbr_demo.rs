//! Exercises `arbor_render::pbr` on a real GL context: a few spheres and a floor with
//! different materials, an orbit camera, and every view mode.
//!
//! cargo run --release -p arbor-viewer --example pbr_demo
//! cargo run --release -p arbor-viewer --example pbr_demo -- --mode 4 --shot out.png

use std::sync::{Arc, Mutex};

use arbor_render::lighting::SkyParams;
use arbor_render::pbr::*;
use arbor_render::render::{Lighting, PostSettings};
use arbor_render::gpu::create_texture;
use eframe::egui;
use glam::{Mat4, Vec3};

struct Scene {
    renderer: PbrRenderer,
    meshes: Vec<PbrMesh>,
    materials: Vec<PbrMaterial>,
    models: Vec<Mat4>,
}

struct Demo {
    scene: Arc<Mutex<Scene>>,
    yaw: f32,
    pitch: f32,
    distance: f32,
    mode: ViewMode,
    wire: bool,
    frames: u32,
    shot: Option<String>,
}

fn sphere(segments: u32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 4]>, Vec<u32>) {
    let (mut p, mut n, mut uv, mut t, mut idx) = (vec![], vec![], vec![], vec![], vec![]);
    for j in 0..=segments {
        let v = j as f32 / segments as f32;
        let phi = v * std::f32::consts::PI;
        for i in 0..=segments * 2 {
            let u = i as f32 / (segments * 2) as f32;
            let theta = u * std::f32::consts::TAU;
            let d = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            p.push(d.to_array());
            n.push(d.to_array());
            uv.push([u, v]);
            let tan = Vec3::new(-theta.sin(), 0.0, theta.cos());
            t.push([tan.x, tan.y, tan.z, -1.0]);
        }
    }
    let row = segments * 2 + 1;
    for j in 0..segments {
        for i in 0..segments * 2 {
            let a = j * row + i;
            idx.extend_from_slice(&[a, a + 1, a + row, a + 1, a + row + 1, a + row]);
        }
    }
    (p, n, uv, t, idx)
}

fn checker(size: u32) -> Vec<u8> {
    let mut px = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let c = ((x / 16 + y / 16) % 2) as u8;
            let (r, g, b) = if c == 0 { (200, 60, 50) } else { (235, 225, 200) };
            px.extend_from_slice(&[r, g, b, 255]);
        }
    }
    px
}

/// Round bumps, as a tangent-space normal map.
fn bumps(size: u32) -> Vec<u8> {
    let mut px = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = ((x % 32) as f32 / 32.0 - 0.5, (y % 32) as f32 / 32.0 - 0.5);
            let r2 = fx * fx + fy * fy;
            let slope = if r2 < 0.16 { 1.6 } else { 0.0 };
            let n = Vec3::new(-fx * slope, -fy * slope, 1.0).normalize();
            px.extend_from_slice(&[
                (n.x * 127.5 + 127.5) as u8,
                (n.y * 127.5 + 127.5) as u8,
                (n.z * 127.5 + 127.5) as u8,
                255,
            ]);
        }
    }
    px
}

impl Scene {
    fn new(gl: &glow::Context) -> Self {
        let (p, n, uv, t, idx) = sphere(48);
        let ball = PbrMesh::new(
            gl,
            &PbrMeshData { positions: &p, normals: &n, uvs: Some(&uv), tangents: Some(&t), colors: None, indices: &idx },
        )
        .unwrap();
        let floor_p = [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
        let floor_n = [[0.0, 1.0, 0.0]; 4];
        let floor = PbrMesh::new(
            gl,
            &PbrMeshData {
                positions: &floor_p,
                normals: &floor_n,
                uvs: None,
                tangents: None,
                colors: None,
                indices: &[0, 2, 1, 0, 3, 2],
            },
        )
        .unwrap();
        let tex = |px: &[u8], srgb| Some(unsafe { create_texture(gl, px, 256, 256, srgb) });
        let materials = vec![
            PbrMaterial {
                textures: PbrTextures { base_color: tex(&checker(256), true), ..Default::default() },
                metallic: 0.0,
                roughness: 0.45,
                ..Default::default()
            },
            PbrMaterial { base_color: [1.0, 0.77, 0.34, 1.0], metallic: 1.0, roughness: 0.2, ..Default::default() },
            PbrMaterial {
                textures: PbrTextures { normal: tex(&bumps(256), false), ..Default::default() },
                base_color: [0.3, 0.5, 0.8, 1.0],
                metallic: 0.0,
                roughness: 0.35,
                ..Default::default()
            },
            PbrMaterial {
                base_color: [0.3, 0.9, 0.5, 0.35],
                metallic: 0.0,
                roughness: 0.1,
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                ..Default::default()
            },
            PbrMaterial {
                base_color: [0.6, 0.6, 0.58, 1.0],
                metallic: 0.0,
                roughness: 0.9,
                double_sided: true,
                ..Default::default()
            },
        ];
        let at = |x: f32| Mat4::from_translation(Vec3::new(x, 0.5, 0.0)) * Mat4::from_scale(Vec3::splat(0.5));
        Self {
            renderer: PbrRenderer::new(gl, 2048),
            meshes: vec![ball, floor],
            materials,
            models: vec![at(-1.2), at(0.0), at(1.2), Mat4::from_translation(Vec3::new(-0.6, 0.6, 0.6)) * Mat4::from_scale(Vec3::splat(0.9)), Mat4::from_scale(Vec3::new(4.0, 1.0, 3.0))],
        }
    }
}

impl eframe::App for Demo {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.frames += 1;
        if let Some(path) = &self.shot {
            if self.frames == 8 {
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
                image::save_buffer(path, &bytes, w as u32, h as u32, image::ColorType::Rgba8).unwrap();
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        egui::SidePanel::left("controls").show(ctx, |ui| {
            egui::ComboBox::from_label("view").selected_text(self.mode.label()).show_ui(ui, |ui| {
                for m in ViewMode::ALL {
                    ui.selectable_value(&mut self.mode, m, m.label());
                }
            });
            ui.checkbox(&mut self.wire, "wireframe");
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
            self.yaw -= response.drag_delta().x * 0.01;
            self.pitch = (self.pitch + response.drag_delta().y * 0.01).clamp(-0.2, 1.4);
            self.distance = (self.distance - ui.input(|i| i.smooth_scroll_delta.y) * 0.01).clamp(1.5, 20.0);
            let eye = Vec3::new(self.yaw.sin() * self.pitch.cos(), self.pitch.sin(), self.yaw.cos() * self.pitch.cos())
                * self.distance
                + Vec3::new(0.0, 0.5, 0.0);
            let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.5, 0.0), Vec3::Y);
            let proj = Mat4::perspective_rh_gl(0.7, rect.aspect_ratio().max(0.01), 0.05, 100.0);
            let scene = self.scene.clone();
            let (mode, wire) = (self.mode, self.wire);
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                    let gl = painter.gl();
                    let ppp = info.pixels_per_point;
                    let h = (rect.height() * ppp).ceil() as i32;
                    let clip = [
                        (rect.min.x * ppp).floor() as i32,
                        (info.screen_size_px[1] as f32 - rect.min.y * ppp - h as f32).floor() as i32,
                        (rect.width() * ppp).ceil() as i32,
                        h,
                    ];
                    let mut s = scene.lock().unwrap();
                    let s = &mut *s;
                    let items: Vec<PbrItem> = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4)]
                        .iter()
                        .zip(&s.models)
                        .map(|(&(mesh, mat), &model)| PbrItem { mesh: &s.meshes[mesh], material: &s.materials[mat], model })
                        .collect();
                    let sky = SkyParams::for_sun(Vec3::new(0.5, 0.6, 0.4).normalize());
                    unsafe {
                        s.renderer.draw(
                            gl,
                            &PbrFrame {
                                items: &items,
                                proj,
                                view_proj: proj * view,
                                lighting: Lighting::procedural(sky, eye, 0.0),
                                post: PostSettings::default(),
                                clip,
                                mode,
                                wireframe: wire,
                                background: true,
                            },
                        );
                    }
                })),
            });
        });
        ctx.request_repaint();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let mode = flag("--mode").and_then(|m| m.parse::<usize>().ok()).map_or(ViewMode::Lit, |i| ViewMode::ALL[i]);
    let shot = flag("--shot");
    eframe::run_native(
        "pbr demo",
        eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 700.0]), ..Default::default() },
        Box::new(move |cc| {
            let gl = cc.gl.clone().expect("glow context");
            Ok(Box::new(Demo {
                scene: Arc::new(Mutex::new(Scene::new(&gl))),
                yaw: 0.4,
                pitch: 0.45,
                distance: 5.5,
                mode,
                wire: args.iter().any(|a| a == "--wire"),
                frames: 0,
                shot,
            }))
        }),
    )
    .unwrap();
}
