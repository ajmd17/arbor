//! Exporting from the panel.
//!
//! On the desktop, trees are written into a folder, as many of them as asked for, off
//! the UI thread. The web has neither a thread to spare nor a folder to write to, so
//! there the tree on screen is packed as a `.glb` and handed to the browser as a
//! download.

use eframe::egui;

use arbor_core::gltf::{self, Format};
use arbor_core::SpeciesTemplate;

use crate::assets::Maps;
use crate::presets;

/// Outcome of the last thing the panel did, and whether it was a failure.
pub type Status = Option<(String, bool)>;

#[cfg(not(target_arch = "wasm32"))]
pub use desktop::Exports;
#[cfg(all(test, not(target_arch = "wasm32")))]
pub use desktop::export_trees;
#[cfg(target_arch = "wasm32")]
pub use web::Exports;

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;
    use arbor_core::gltf::ExportOptions;
    use arbor_core::textures::TEXTURE_DIR;

    /// Where exports go until another folder is chosen, relative to the repository root.
    const EXPORT_DIR: &str = "exports";
    /// Most trees one export may grow.
    const MAX_VARIATIONS: u32 = 1000;

    /// Grows `count` trees of `params` — its own seed, then each one after, every range
    /// landing afresh for each — and writes them into `dir` named after `name`,
    /// returning the line the panel reports it with.
    /// `progress` hears how many are done after each; returning false stops the batch.
    pub fn export_trees(
        dir: &Path,
        name: &str,
        format: Format,
        params: &SpeciesTemplate,
        count: u32,
        options: &ExportOptions,
        progress: impl FnMut(u32) -> bool,
    ) -> Result<String, String> {
        let report = gltf::export_batch(dir, name, format, params, count, options, progress)?;
        let size = report.bytes as f64 / 1e6;
        let done = report.trees.len();
        let mut message = if report.stopped {
            format!("stopped after {done} of {count} trees in {} ({size:.1} MB)", dir.display())
        } else if count == 1 {
            format!("exported {} ({size:.1} MB)", report.trees[0].display())
        } else {
            format!("exported {done} trees to {} ({size:.1} MB)", dir.display())
        };
        for warning in report.warnings {
            message.push_str("; ");
            message.push_str(&warning);
        }
        Ok(message)
    }

    /// What an export running on its own thread says as it goes.
    enum ExportEvent {
        /// This many trees are written.
        Progress(u32),
        Finished(Result<String, String>),
    }

    /// An export running on its own thread.
    struct ExportJob {
        events: std::sync::mpsc::Receiver<ExportEvent>,
        /// Set to ask it to stop after the tree it is on.
        stop: Arc<AtomicBool>,
        done: u32,
        total: u32,
    }

    pub struct Exports {
        /// Folder exports are written to, as typed or picked.
        dir: String,
        /// How many trees an export grows, one per seed from the tree's own up.
        variations: u32,
        /// Whether an export carries the wind data.
        wind: bool,
        /// Browse was clicked; the folder picker opens once the panel is drawn, where
        /// the window it belongs to is to hand.
        browse_requested: bool,
        job: Option<ExportJob>,
    }

    impl Default for Exports {
        fn default() -> Self {
            Self {
                dir: EXPORT_DIR.to_string(),
                variations: 1,
                wind: false,
                browse_requested: false,
                job: None,
            }
        }
    }

    impl Exports {
        /// Picks up how far an export has got and its outcome once it is done, and opens
        /// the folder picker if it was asked for.
        pub fn poll(&mut self, frame: &eframe::Frame, status: &mut Status, _maps: &Maps) {
            if std::mem::take(&mut self.browse_requested) {
                self.browse(frame);
            }
            let Some(job) = &mut self.job else { return };
            let outcome = loop {
                match job.events.try_recv() {
                    Ok(ExportEvent::Progress(done)) => job.done = done,
                    Ok(ExportEvent::Finished(outcome)) => break outcome,
                    Err(std::sync::mpsc::TryRecvError::Empty) => return,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        break Err("the export stopped without finishing".to_string());
                    }
                }
            };
            *status = Some(match outcome {
                Ok(message) => (message, false),
                Err(why) => (format!("export failed: {why}"), true),
            });
            self.job = None;
        }

        /// Where an export would go and what its files would be called, for the buttons
        /// to say before they are pressed, or why it cannot go anywhere.
        fn target(
            &self,
            format: Format,
            params: &SpeciesTemplate,
            save_name: &str,
        ) -> Result<(PathBuf, String), String> {
            let name = presets::key_for(save_name)?;
            let dir = self.dir.trim();
            if dir.is_empty() {
                return Err("choose a folder to export to".to_string());
            }
            let count = self.variations.clamp(1, MAX_VARIATIONS);
            let first = params.seed;
            let files = if count == 1 {
                gltf::batch_file(&name, format, first, 1)
            } else {
                format!(
                    "{count} trees, {} to {}",
                    gltf::batch_file(&name, format, first, count),
                    gltf::batch_file(&name, format, first.wrapping_add(u64::from(count) - 1), count)
                )
            };
            Ok((PathBuf::from(dir), files))
        }

        /// Writes the tree as it stands, and as many more of the species as the panel
        /// asks for, off the UI thread: growing a batch, baking the leaf atlas and
        /// encoding the textures takes a while. Each tree is grown again there from a
        /// copy of the settings, which gives the same tree, since growth is
        /// deterministic, and cannot be a frame behind the panel.
        fn start(&mut self, format: Format, params: &SpeciesTemplate, save_name: &str, status: &mut Status) {
            let (dir, files) = match self.target(format, params, save_name) {
                Ok(target) => target,
                Err(why) => {
                    *status = Some((why, true));
                    return;
                }
            };
            let name = presets::key_for(save_name).expect("checked by target");
            let mut params = params.clone();
            params.name = save_name.trim().to_string();
            let count = self.variations.clamp(1, MAX_VARIATIONS);
            let options = ExportOptions {
                textures: Some(TEXTURE_DIR.into()),
                wind: self.wind,
            };
            *status = Some((format!("exporting {files} to {}", dir.display()), false));
            let stop = Arc::new(AtomicBool::new(false));
            let asked_to_stop = Arc::clone(&stop);
            let (tx, events) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let progress = tx.clone();
                let outcome = export_trees(&dir, &name, format, &params, count, &options, |done| {
                    let _ = progress.send(ExportEvent::Progress(done));
                    !asked_to_stop.load(Ordering::Relaxed)
                });
                let _ = tx.send(ExportEvent::Finished(outcome));
            });
            self.job = Some(ExportJob {
                events,
                stop,
                done: 0,
                total: count,
            });
        }

        /// The system's folder picker, opened over the viewer's window at the folder the
        /// field names when there is one.
        fn browse(&mut self, frame: &eframe::Frame) {
            let typed = Path::new(self.dir.trim());
            let start = std::path::absolute(typed)
                .ok()
                .filter(|p| p.is_dir())
                .or_else(|| std::env::current_dir().ok());
            let mut dialog = rfd::FileDialog::new()
                .set_title("Export to")
                .set_parent(frame);
            if let Some(start) = start {
                dialog = dialog.set_directory(start);
            }
            if let Some(dir) = dialog.pick_folder() {
                self.dir = dir.display().to_string();
            }
        }

        /// Where exports go, how many trees they grow, and the buttons that start them.
        pub fn ui(
            &mut self,
            ui: &mut egui::Ui,
            params: &SpeciesTemplate,
            save_name: &str,
            status: &mut Status,
        ) {
            let idle = self.job.is_none();
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Export to");
                ui.add_enabled(
                    idle,
                    egui::TextEdit::singleline(&mut self.dir)
                        .desired_width(160.0)
                        .hint_text("folder"),
                )
                .on_hover_text(
                    "Folder the exports are written to, created if it is not there. A relative \
                     path is taken from the folder the viewer was started in.",
                );
                if ui
                    .add_enabled(idle, egui::Button::new("Browse…"))
                    .on_hover_text("Pick the folder.")
                    .clicked()
                {
                    self.browse_requested = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Variations");
                ui.add_enabled(
                    idle,
                    egui::DragValue::new(&mut self.variations)
                        .range(1..=MAX_VARIATIONS)
                        .speed(0.2),
                )
                .on_hover_text(
                    "How many trees to export: this one, then the same species grown from each \
                     seed after its own. With more than one, each file is named for its seed \
                     (name_seed7.glb), so any of them can be grown again.",
                );
                wind_checkbox(ui, &mut self.wind);
            });
            ui.horizontal(|ui| {
                for (label, format, what) in [
                    ("Export GLB", Format::Glb, "one self-contained file per tree, textures inside"),
                    ("Export glTF", Format::Gltf, "JSON per tree, with its buffer and the textures as files beside it"),
                ] {
                    let target = self.target(format, params, save_name);
                    let button = ui.add_enabled(idle && target.is_ok(), egui::Button::new(label));
                    let button = match &target {
                        Ok((dir, files)) => {
                            button.on_hover_text(format!("Write {files} to {}: {what}.", dir.display()))
                        }
                        Err(why) => button.on_disabled_hover_text(why.as_str()),
                    };
                    if button.clicked() {
                        self.start(format, params, save_name, status);
                    }
                }
                if let Some(job) = &self.job {
                    ui.spinner();
                    if job.total > 1 {
                        ui.label(format!("{} / {}", job.done, job.total));
                        if ui
                            .button("Stop")
                            .on_hover_text("Stop after the tree being written now. What is written stays.")
                            .clicked()
                        {
                            job.stop.store(true, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use arbor_core::gltf::{Exporter, Textures};
    use arbor_core::textures;
    use arbor_core::{build_leaves, build_mesh, grow};
    use eframe::wasm_bindgen::closure::Closure;
    use eframe::wasm_bindgen::{JsCast as _, JsValue};

    /// A download asked for this frame, and packed at the start of the next, so the
    /// panel has said what it is doing before the page stops for it.
    struct Pending {
        params: SpeciesTemplate,
        file: String,
    }

    #[derive(Default)]
    pub struct Exports {
        /// Whether the download carries the wind data.
        wind: bool,
        pending: Option<Pending>,
    }

    impl Exports {
        /// Packs and hands over a download asked for last frame.
        pub fn poll(&mut self, _frame: &eframe::Frame, status: &mut Status, maps: &Maps) {
            let Some(pending) = self.pending.take() else { return };
            *status = Some(match self.pack(&pending, maps) {
                Ok(message) => (message, false),
                Err(why) => (format!("download failed: {why}"), true),
            });
        }

        /// Grows the tree again from the panel's settings, which gives the one on
        /// screen, and packs it with the maps already fetched to draw it.
        fn pack(&self, pending: &Pending, maps: &Maps) -> Result<String, String> {
            let tree = pending.params.instance();
            let needed: Vec<String> = [&tree.mesh.bark_texture, &tree.leaves.texture]
                .into_iter()
                .flat_map(|name| textures::map_files(name))
                .collect();
            if !maps.ready(&needed) {
                return Err("the textures are still on their way; try again in a moment".to_string());
            }
            let skeleton = grow(&tree);
            let (mesh, leaves) = (build_mesh(&skeleton, &tree), build_leaves(&skeleton, &tree));
            let textures = Textures::load(&tree, maps)?;
            let warnings = textures.warnings.clone();
            let bytes = Exporter::with_textures(textures, self.wind).glb(&skeleton, &mesh, &leaves, &tree);
            save_file(&pending.file, &bytes, "model/gltf-binary").map_err(|e| format!("{e:?}"))?;
            let mut message = format!("downloaded {} ({:.1} MB)", pending.file, bytes.len() as f64 / 1e6);
            for warning in warnings {
                message.push_str("; ");
                message.push_str(&warning);
            }
            Ok(message)
        }

        /// The download button, and whether it carries the wind data.
        pub fn ui(
            &mut self,
            ui: &mut egui::Ui,
            params: &SpeciesTemplate,
            save_name: &str,
            status: &mut Status,
        ) {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let file = presets::key_for(save_name).map(|name| gltf::batch_file(&name, Format::Glb, params.seed, 1));
                let button = ui.add_enabled(
                    self.pending.is_none() && file.is_ok(),
                    egui::Button::new("Download GLB"),
                );
                let button = match &file {
                    Ok(file) => button.on_hover_text(format!(
                        "Download the tree on screen as {file}: one self-contained file, textures inside."
                    )),
                    Err(why) => button.on_disabled_hover_text(why.as_str()),
                };
                if button.clicked()
                    && let Ok(file) = file
                {
                    let mut params = params.clone();
                    params.name = save_name.trim().to_string();
                    *status = Some((format!("packing {file}…"), false));
                    self.pending = Some(Pending { params, file });
                }
                wind_checkbox(ui, &mut self.wind);
            });
        }
    }

    /// Hands `bytes` to the browser as a file called `name`, the way a link with a
    /// `download` attribute would.
    fn save_file(name: &str, bytes: &[u8], mime: &str) -> Result<(), JsValue> {
        let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes));
        let options = web_sys::BlobPropertyBag::new();
        options.set_type(mime);
        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)?;
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let body = document.body().ok_or("no body")?;
        let link = document.create_element("a")?.dyn_into::<web_sys::HtmlAnchorElement>()?;
        link.set_href(&url);
        link.set_download(name);
        body.append_child(&link)?;
        link.click();
        link.remove();
        // Let go of the file once the browser has had time to start saving it.
        let revoke = Closure::once_into_js(move || {
            let _ = web_sys::Url::revoke_object_url(&url);
        });
        window.set_timeout_with_callback_and_timeout_and_arguments_0(revoke.unchecked_ref(), 60_000)?;
        Ok(())
    }
}

fn wind_checkbox(ui: &mut egui::Ui, wind: &mut bool) {
    ui.checkbox(wind, "Wind data").on_hover_text(
        "Also write what an engine needs to sway the tree as the viewer does: a table \
         of the stems it bends as, and where each vertex sits on them (the \
         ARBOR_tree_wind extension).",
    );
}
