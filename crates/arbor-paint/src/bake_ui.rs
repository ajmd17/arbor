//! The bake panel: what to bake, how, and the maps that come out.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;

use arbor_render::pbr::ViewMode;
use eframe::egui;

use crate::bake::{self, BakeParams, BakeResult, MapKind, Progress};
use crate::model::Model;

const SIZES: [u32; 5] = [256, 512, 1024, 2048, 4096];

struct Job {
    material: String,
    progress: Arc<Progress>,
    rx: Receiver<Result<BakeResult, String>>,
}

pub struct Baker {
    pub params: BakeParams,
    /// The material being baked, an index into the model's materials.
    material: usize,
    job: Option<Job>,
    /// The latest bake of each material.
    results: Vec<BakeResult>,
    /// The map being shown on the model.
    pub viewing: Option<(usize, MapKind)>,
    sixteen_bit: bool,
    message: Option<(String, bool)>,
    /// For scripted checks: bake on load, then show this map.
    test_view: Option<MapKind>,
    test_started: bool,
}

impl Baker {
    pub fn new() -> Self {
        Self {
            params: BakeParams::default(),
            material: 0,
            job: None,
            results: Vec::new(),
            viewing: None,
            sixteen_bit: false,
            message: None,
            test_view: None,
            test_started: false,
        }
    }

    /// Bakes as soon as a model is up and then shows `kind`, with no clicking.
    pub fn bake_on_load(&mut self, kind: MapKind) {
        self.test_view = Some(kind);
    }

    /// Whether a scripted bake has yet to finish.
    pub fn test_pending(&self) -> bool {
        self.test_view.is_some()
    }

    pub fn test_step(&mut self, model: &mut Model, gl: &glow::Context) {
        let Some(kind) = self.test_view else { return };
        if self.running() {
            return;
        }
        if !self.test_started {
            self.test_started = true;
            self.params.maps = vec![kind];
            self.material = (0..model.material_info.len()).find(|&i| model.material_info[i].used).unwrap_or(0);
            self.start(model);
        } else if let Some(map) = self.results.first().and_then(|r| r.maps.iter().find(|m| m.kind == kind)) {
            self.viewing = Some((self.material, kind));
            model.set_preview(gl, self.material, map);
            self.test_view = None;
        } else {
            // The bake failed; nothing more to wait for.
            self.test_view = None;
        }
    }

    /// Forgets everything about the model that was open.
    pub fn reset(&mut self) {
        self.job.as_ref().inspect(|j| j.progress.cancel.store(true, Ordering::Relaxed));
        self.job = None;
        self.results.clear();
        self.viewing = None;
        self.material = 0;
        self.message = None;
    }

    pub fn running(&self) -> bool {
        self.job.is_some()
    }

    fn start(&mut self, model: &Model) {
        let source = model.source.clone();
        let visible = model.visibility();
        let (params, material) = (self.params.clone(), self.material);
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let shared = progress.clone();
        std::thread::spawn(move || {
            let result = bake::Scene::build(source, &visible).and_then(|scene| bake::bake(&scene, material, &params, &shared));
            let _ = tx.send(result);
        });
        self.message = None;
        self.job = Some(Job { material: model.material_info[material].name.clone(), progress, rx });
    }

    /// Takes a finished bake, if there is one.
    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        let result = match job.rx.try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
                return;
            }
            Err(TryRecvError::Disconnected) => Err("the bake stopped unexpectedly".into()),
        };
        self.job = None;
        match result {
            Ok(r) => {
                self.results.retain(|old| old.material != r.material);
                self.results.push(r);
            }
            Err(e) => self.message = Some((e, true)),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, model: &mut Model, gl: &glow::Context) {
        let used: Vec<usize> = (0..model.material_info.len()).filter(|&i| model.material_info[i].used).collect();
        if !used.contains(&self.material) {
            self.material = used.first().copied().unwrap_or(0);
        }
        ui.heading("Bake");
        ui.add_enabled_ui(!self.running(), |ui| {
            egui::ComboBox::from_label("Texture set")
                .selected_text(&model.material_info[self.material].name)
                .show_ui(ui, |ui| {
                    for &i in &used {
                        ui.selectable_value(&mut self.material, i, &model.material_info[i].name);
                    }
                });
            let mut size = self.params.width;
            egui::ComboBox::from_label("Size").selected_text(format!("{size}")).show_ui(ui, |ui| {
                for s in SIZES {
                    ui.selectable_value(&mut size, s, format!("{s}"));
                }
            });
            (self.params.width, self.params.height) = (size, size);
            let mut ss = self.params.supersample;
            egui::ComboBox::from_label("Antialiasing").selected_text(format!("{ss}x")).show_ui(ui, |ui| {
                for s in [1, 2, 4] {
                    ui.selectable_value(&mut ss, s, format!("{s}x"));
                }
            });
            self.params.supersample = ss;
            ui.add(egui::Slider::new(&mut self.params.padding, 0..=32).text("Padding (px)"));

            ui.add_space(4.0);
            let has_colors = model.source.primitives.iter().any(|p| p.colors.is_some());
            for kind in MapKind::ALL {
                let enabled = kind != MapKind::VertexColor || has_colors;
                let mut on = self.params.maps.contains(&kind);
                if ui.add_enabled(enabled, egui::Checkbox::new(&mut on, kind.label())).changed() {
                    self.params.maps.retain(|k| *k != kind);
                    if on {
                        self.params.maps.push(kind);
                    }
                }
            }

            let maps = &self.params.maps;
            if maps.iter().any(|k| k.is_heavy()) {
                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut self.params.ao_rays, 8..=1024).logarithmic(true).text("Rays per texel"));
                if maps.contains(&MapKind::AmbientOcclusion) {
                    ui.add(egui::Slider::new(&mut self.params.ao_distance, 0.01..=1.0).logarithmic(true).text("Occluder distance"))
                        .on_hover_text("As a share of the model's size");
                }
                if maps.contains(&MapKind::Thickness) {
                    ui.add(egui::Slider::new(&mut self.params.thickness_distance, 0.01..=1.0).logarithmic(true).text("Thickness reach"))
                        .on_hover_text("As a share of the model's size");
                }
            }
            if maps.contains(&MapKind::Curvature) {
                ui.add(egui::Slider::new(&mut self.params.curvature_radius, 0.001..=0.1).logarithmic(true).text("Curvature radius"))
                    .on_hover_text("As a share of the model's size");
                ui.add(egui::Slider::new(&mut self.params.curvature_strength, 0.1..=10.0).logarithmic(true).text("Curvature strength"));
            }
        });

        ui.add_space(6.0);
        if let Some(job) = &self.job {
            ui.horizontal(|ui| {
                ui.add(egui::ProgressBar::new(job.progress.fraction()).desired_width(150.0).show_percentage());
                if ui.button("Cancel").clicked() {
                    job.progress.cancel.store(true, Ordering::Relaxed);
                }
            });
            ui.weak(format!("Baking {}…", job.material));
        } else if ui.add_enabled(!self.params.maps.is_empty(), egui::Button::new("Bake")).clicked() {
            self.start(model);
        }
        if let Some((message, error)) = &self.message {
            let color = if *error { egui::Color32::LIGHT_RED } else { egui::Color32::LIGHT_GRAY };
            ui.colored_label(color, message);
        }
        self.results_ui(ui, model, gl);
    }

    fn results_ui(&mut self, ui: &mut egui::Ui, model: &mut Model, gl: &glow::Context) {
        let Some(result) = self.results.iter().find(|r| r.material == self.material) else {
            return;
        };
        ui.add_space(8.0);
        ui.strong("Baked maps");
        ui.weak(format!(
            "{}x{}, islands cover {:.0}% - {:.1} s",
            result.maps[0].width,
            result.maps[0].height,
            result.coverage * 100.0,
            result.seconds
        ));
        for w in &result.warnings {
            ui.colored_label(egui::Color32::from_rgb(220, 170, 60), format!("⚠ {w}"));
        }
        let (mut view, mut stop, mut save) = (None, false, None);
        for map in &result.maps {
            ui.horizontal(|ui| {
                ui.label(map.kind.label());
                let showing = self.viewing == Some((result.material, map.kind));
                if ui.selectable_label(showing, "View").clicked() {
                    if showing {
                        stop = true;
                    } else {
                        view = Some(map);
                    }
                }
                if ui.button("Save…").clicked() {
                    save = Some(map);
                }
            });
        }
        ui.checkbox(&mut self.sixteen_bit, "16-bit PNG");
        let mut save_all = false;
        ui.horizontal(|ui| {
            save_all = ui.button("Save all…").clicked();
            if self.viewing.is_some() && ui.button("Stop viewing").clicked() {
                stop = true;
            }
        });

        let material_name = model.material_info[result.material].name.clone();
        let stem = format!("{}_{}", sanitize(&model.name), sanitize(&material_name));
        let sixteen = self.sixteen_bit;
        let mut message = None;
        if let Some(map) = view {
            model.set_preview(gl, result.material, map);
            self.viewing = Some((result.material, map.kind));
        }
        if let Some(map) = save
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG", &["png"])
                .set_file_name(format!("{stem}_{}.png", map.kind.suffix()))
                .save_file()
        {
            message = Some(match map.save_png(&path, sixteen) {
                Ok(()) => (format!("Saved {}", path.display()), false),
                Err(e) => (e, true),
            });
        }
        if save_all && let Some(dir) = rfd::FileDialog::new().pick_folder() {
            message = Some(save_all_maps(&result.maps, &dir, &stem, sixteen));
        }
        if stop {
            model.clear_preview(gl);
            self.viewing = None;
        }
        if message.is_some() {
            self.message = message;
        }
    }
}

fn save_all_maps(maps: &[bake::BakedMap], dir: &Path, stem: &str, sixteen: bool) -> (String, bool) {
    let mut written: Vec<PathBuf> = Vec::new();
    for map in maps {
        let path = dir.join(format!("{stem}_{}.png", map.kind.suffix()));
        if let Err(e) = map.save_png(&path, sixteen) {
            return (e, true);
        }
        written.push(path);
    }
    (format!("Saved {} maps to {}", written.len(), dir.display()), false)
}

/// A name that is safe in a file name.
fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
    if s.is_empty() { "model".into() } else { s }
}

/// The view mode to draw with: a baked map is shown as it is.
pub fn effective_mode(baker: &Baker, chosen: ViewMode) -> ViewMode {
    if baker.viewing.is_some() { ViewMode::BaseColor } else { chosen }
}
