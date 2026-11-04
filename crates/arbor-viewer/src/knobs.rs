//! Draws the species controls in the side panel. The tables themselves live in
//! `arbor_core::knobs`, shared with the engine plugins; this walks them with egui.
//!
//! Every number here may also be a range: the ± beside a slider splits it into two
//! handles, and each tree lands somewhere between them, where being decided by its
//! seed. The panel edits the species, ranges and all, and says under each range where
//! the tree on screen landed.

pub use arbor_core::knobs::*;
use arbor_core::species::ChildPattern;
use arbor_core::{Ranged, SpeciesTemplate};
use eframe::egui;

// ---------------------------------------------------------------------------------
// Drawing

fn slider<'a>(value: &'a mut f32, knob_range: (f32, f32), log: bool) -> egui::Slider<'a> {
    egui::Slider::new(value, knob_range.0..=knob_range.1)
        .logarithmic(log)
        // Keep a preset's own value even when it lies outside the range: the default
        // clamps it into range as it draws and writes that back over the species.
        .clamping(egui::SliderClamping::Edits)
}

/// A number short enough to sit in a label, to about three figures.
fn short(v: f32) -> String {
    let decimals = match v.abs() {
        m if m >= 100.0 => 0,
        m if m >= 10.0 => 1,
        m if m >= 1.0 => 2,
        m if m >= 0.1 => 3,
        _ => 4,
    };
    format!("{v:.decimals$}")
}

/// Draws one number that may be a range, reporting whether the user changed it.
///
/// The ± in front turns a single value into a range: a second handle appears under
/// the first, and each tree lands somewhere between the two, where being decided by
/// its seed. `landed` is where the tree on screen landed, which the second handle's
/// label reports. Pressing ± again folds the range back to that one value, so the tree
/// on screen stays as it is.
pub fn ranged_ui(
    ui: &mut egui::Ui,
    value: &mut Ranged,
    landed: f32,
    knob_range: (f32, f32),
    log: bool,
    label: &str,
    help: &str,
) -> bool {
    let mut changed = false;
    let mut indent = 0.0;
    ui.horizontal(|ui| {
        let ranged = value.is_range();
        let toggle = ui.selectable_label(ranged, "±").on_hover_text(if ranged {
            "Fold the range back to one value: the one this tree landed on."
        } else {
            "Give this a range. Every tree lands somewhere in it, decided by its seed."
        });
        indent = toggle.rect.width() + ui.spacing().item_spacing.x;
        if toggle.clicked() {
            *value = match *value {
                Ranged::Fixed(v) => Ranged::Between(v, v),
                Ranged::Between(..) => Ranged::Fixed(landed),
            };
            changed = true;
        }
        let first = match value {
            Ranged::Fixed(v) | Ranged::Between(v, _) => v,
        };
        changed |= ui.add(slider(first, knob_range, log).text(label)).on_hover_text(help).changed();
    });
    if let Ranged::Between(_, hi) = value {
        ui.horizontal(|ui| {
            ui.add_space(indent);
            changed |= ui
                .add(slider(hi, knob_range, log).text(format!("to · this tree {}", short(landed))))
                .on_hover_text(help)
                .changed();
        });
    }
    changed
}

/// Draws one continuous control, reporting whether the user changed it. `landed` is
/// the same kind of thing as `target` with every number where this tree landed.
pub fn knob_ui<T>(ui: &mut egui::Ui, target: &mut T, landed: &mut T, knob: &Knob<T>) -> bool {
    let at = (knob.get)(landed).lo();
    ranged_ui(ui, (knob.get)(target), at, knob.range, knob.log, knob.label, knob.help)
}

pub fn count_ui<T>(ui: &mut egui::Ui, target: &mut T, count: &Count<T>) -> bool {
    let value = (count.get)(target);
    ui.add(
        egui::Slider::new(value, count.range.0..=count.range.1)
            .clamping(egui::SliderClamping::Edits)
            .text(count.label),
    )
    .on_hover_text(count.help)
    .changed()
}

pub fn knobs_ui<T>(ui: &mut egui::Ui, target: &mut T, landed: &mut T, knobs: &[Knob<T>]) -> bool {
    let mut changed = false;
    for knob in knobs {
        changed |= knob_ui(ui, target, landed, knob);
    }
    changed
}

/// Draws a group of controls under its own collapsing header. `salt` keeps the
/// header's open state apart from the same group drawn for another level.
pub fn group_ui<T>(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash,
    target: &mut T,
    landed: &mut T,
    group: &Group<T>,
) -> bool {
    let mut changed = false;
    egui::CollapsingHeader::new(group.title)
        .id_salt((salt, group.title))
        .show(ui, |ui| {
            changed |= knobs_ui(ui, target, landed, group.knobs);
            for count in group.counts {
                changed |= count_ui(ui, target, count);
            }
        });
    changed
}

/// How children are laid along their parent: not at all, in whorls, or at a steady
/// rate per metre. Density is the main lever on how bushy a level comes out.
pub fn pattern_ui(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash,
    pattern: &mut ChildPattern<Ranged>,
    landed: &ChildPattern<Ranged>,
) -> bool {
    let mut changed = false;
    let current = match pattern {
        ChildPattern::None => 0,
        ChildPattern::Whorl { .. } => 1,
        ChildPattern::Continuous { .. } => 2,
    };
    let mut picked = current;
    egui::ComboBox::from_id_salt((salt, "pattern"))
        .selected_text(["None", "Whorls", "Continuous"][current])
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut picked, 0, "None");
            ui.selectable_value(&mut picked, 1, "Whorls");
            ui.selectable_value(&mut picked, 2, "Continuous");
        })
        .response
        .on_hover_text("How children are laid along the parent: in rings at intervals, or at a steady rate per metre.");
    if picked != current {
        *pattern = match picked {
            1 => ChildPattern::Whorl { every: 2, count: 4 },
            2 => ChildPattern::Continuous { density: Ranged::Fixed(2.0) },
            _ => ChildPattern::None,
        };
        changed = true;
    }
    match pattern {
        ChildPattern::None => {}
        ChildPattern::Whorl { every, count } => {
            changed |= ui
                .add(
                    egui::Slider::new(every, WHORL_EVERY.0..=WHORL_EVERY.1)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Whorl every"),
                )
                .on_hover_text("A whorl on every this-many segments of the parent.")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(count, WHORL_COUNT.0..=WHORL_COUNT.1)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Per whorl"),
                )
                .on_hover_text("Branches in each whorl.")
                .changed();
        }
        ChildPattern::Continuous { density } => {
            // Just switched to, the tree on screen has no density of its own yet.
            let at = match landed {
                ChildPattern::Continuous { density: d } => d.lo(),
                _ => density.lo(),
            };
            changed |= ranged_ui(
                ui,
                density,
                at,
                DENSITY,
                true,
                "Density",
                "Children per metre of the parent.",
            );
        }
    }
    changed
}

/// Everything about one level: how it is born off its parent, then how it grows.
/// `landed` must be the same species with every number where this tree landed.
pub fn level_ui(
    ui: &mut egui::Ui,
    params: &mut SpeciesTemplate,
    landed: &mut SpeciesTemplate,
    level: usize,
) -> bool {
    let mut changed = false;
    if let (Some(spawn), Some(spawn_at)) = (spawn_mut(params, level), spawn_mut(landed, level)) {
        let parent = if level == 1 { "trunk".to_string() } else { format!("level {}", level - 1) };
        egui::CollapsingHeader::new(format!("Spawning off {parent}"))
            .id_salt((level, "spawning"))
            .default_open(true)
            .show(ui, |ui| {
                changed |= pattern_ui(ui, level, &mut spawn.pattern, &spawn_at.pattern);
                for group in CHILDREN {
                    changed |= group_ui(ui, (level, "spawn"), spawn, spawn_at, group);
                }
            });
    }
    if let (Some(stem), Some(stem_at)) = (stem_mut(params, level), stem_mut(landed, level)) {
        for group in STEM {
            changed |= group_ui(ui, (level, "stem"), stem, stem_at, group);
        }
    }
    changed
}
