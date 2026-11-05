//! The controls of a species, a clump of ground cover or a rock as one flat list, in the
//! order the viewer's panel lays them out, for an engine's editor to build its own panel
//! from. Each entry reads and writes its own field of the template, so the editor never
//! needs to know the shape of the params.
//!
//! The list follows the template: a tree grows a section per stem level it declares, a
//! vine has its climbing controls, and a level spawned in whorls offers whorl controls
//! where one spawned continuously offers a density. Build it again after a change, since
//! a change can move the controls after it.

use crate::cover::CoverTemplate;
use crate::knobs::{self, Count, Group, Knob};
use crate::rocks::RockTemplate;
use crate::species::ChildPattern;
use crate::{Ranged, SpeciesTemplate};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// A number that may be a range, with the span its slider offers.
    Ranged { range: (f32, f32), log: bool },
    Count { range: (u32, u32) },
    Choice { options: &'static [&'static str] },
    Toggle,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Ranged(Ranged),
    Count(u32),
    /// An index into the control's options.
    Choice(u32),
    Toggle(bool),
}

pub struct Control<T> {
    /// Stable across builds of the list for the same template: section, group and label,
    /// lowercased, e.g. `level_2.forks.fork_chance`.
    pub id: String,
    /// The heading it sits under: `Shape`, `Trunk`, `Level 2`, `Foliage` and so on.
    pub section: String,
    /// The collapsing group inside that section, empty for the section's own controls.
    pub group: String,
    pub label: &'static str,
    pub help: &'static str,
    pub kind: Kind,
    /// False for what only the shaders read, the wind response, which changes nothing
    /// grown: an editor may update its materials rather than grow the whole thing again.
    pub regrows: bool,
    get: Box<dyn Fn(&mut T) -> Option<Value> + Send + Sync>,
    set: Box<dyn Fn(&mut T, Value) -> bool + Send + Sync>,
}

impl<T> Control<T> {
    /// Takes `&mut` only because the tables reach their fields that way.
    pub fn get(&self, target: &mut T) -> Option<Value> {
        (self.get)(target)
    }

    /// Writes `value` into `target`, reporting whether it was the right kind of value
    /// for this control and `target` has the field.
    pub fn set(&self, target: &mut T, value: Value) -> bool {
        (self.set)(target, value)
    }
}

/// Reaches the part of `T` a table edits, at a stem level where that matters.
type Lens<T, S> = fn(&mut T, usize) -> Option<&mut S>;

struct List<T> {
    out: Vec<Control<T>>,
    section: String,
    group: String,
    regrows: bool,
}

impl<T: 'static> List<T> {
    fn new() -> Self {
        Self { out: Vec::new(), section: String::new(), group: String::new(), regrows: true }
    }

    fn section(&mut self, section: impl Into<String>) {
        self.section = section.into();
        self.group.clear();
        self.regrows = true;
    }

    fn push(
        &mut self,
        label: &'static str,
        help: &'static str,
        kind: Kind,
        get: impl Fn(&mut T) -> Option<Value> + Send + Sync + 'static,
        set: impl Fn(&mut T, Value) -> bool + Send + Sync + 'static,
    ) {
        let id = [self.section.as_str(), self.group.as_str(), label]
            .iter()
            .filter(|part| !part.is_empty())
            .map(|part| slug(part))
            .collect::<Vec<_>>()
            .join(".");
        self.out.push(Control {
            id,
            section: self.section.clone(),
            group: self.group.clone(),
            label,
            help,
            kind,
            regrows: self.regrows,
            get: Box::new(get),
            set: Box::new(set),
        });
    }

    fn knobs<S: 'static>(&mut self, lens: Lens<T, S>, level: usize, knobs: &'static [Knob<S>]) {
        for knob in knobs {
            self.push(
                knob.label,
                knob.help,
                Kind::Ranged { range: knob.range, log: knob.log },
                move |t| lens(t, level).map(|s| Value::Ranged(*(knob.get)(s))),
                move |t, v| match (lens(t, level), v) {
                    (Some(s), Value::Ranged(r)) => {
                        *(knob.get)(s) = r;
                        true
                    }
                    _ => false,
                },
            );
        }
    }

    fn counts<S: 'static>(&mut self, lens: Lens<T, S>, level: usize, counts: &'static [Count<S>]) {
        for count in counts {
            self.push(
                count.label,
                count.help,
                Kind::Count { range: count.range },
                move |t| lens(t, level).map(|s| Value::Count(*(count.get)(s))),
                move |t, v| match (lens(t, level), v) {
                    (Some(s), Value::Count(n)) => {
                        *(count.get)(s) = n;
                        true
                    }
                    _ => false,
                },
            );
        }
    }

    /// A table's group under its own name, after `prefix` when there is one.
    fn group<S: 'static>(&mut self, prefix: &str, lens: Lens<T, S>, level: usize, group: &'static Group<S>) {
        self.group = if prefix.is_empty() { group.title.to_string() } else { format!("{prefix} · {}", group.title) };
        self.knobs(lens, level, group.knobs);
        self.counts(lens, level, group.counts);
    }
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}

const PATTERNS: &[&str] = &["None", "Whorls", "Continuous"];

/// Every control of a tree, shrub or vine species.
pub fn tree(template: &SpeciesTemplate) -> Vec<Control<SpeciesTemplate>> {
    type T = SpeciesTemplate;
    let mut list = List::<T>::new();

    list.section("Shape");
    list.knobs(|t, _| Some(t), 0, knobs::SHAPE);
    list.push(
        "Branch levels",
        "Levels grown, counting the trunk. Capped by how many levels the species declares.",
        Kind::Count { range: knobs::BRANCH_LEVELS },
        |t: &mut T| Some(Value::Count(u32::from(t.max_levels))),
        |t: &mut T, v| match v {
            Value::Count(n) => {
                t.max_levels = n.min(u32::from(u8::MAX)) as u8;
                true
            }
            _ => false,
        },
    );
    list.counts(|t, _| Some(t), 0, std::slice::from_ref(&knobs::SPLIT_DEPTH));
    list.group("", |t, _| Some(&mut t.envelope), 0, &knobs::ENVELOPE);
    list.group("", |t, _| Some(&mut t.basal), 0, &knobs::BASAL);
    if template.vine.is_some() {
        list.group("", |t, _| t.vine.as_mut(), 0, &knobs::VINE);
    }

    for level in 0..knobs::level_count(template) {
        list.section(if level == 0 { "Trunk".to_string() } else { format!("Level {level}") });
        let mut probe = template.clone();
        if let Some(spawn) = knobs::spawn_mut(&mut probe, level) {
            let parent = if level == 1 { "trunk".to_string() } else { format!("level {}", level - 1) };
            let spawning = format!("Spawning off {parent}");
            list.group = spawning.clone();
            pattern_controls(&mut list, level, &spawn.pattern);
            for group in knobs::CHILDREN {
                list.group(&spawning, |t, l| knobs::spawn_mut(t, l), level, group);
            }
        }
        for group in knobs::STEM {
            list.group("", |t, l| knobs::stem_mut(t, l), level, group);
        }
    }

    list.section("Wind");
    list.regrows = false;
    list.group("", |t, _| Some(&mut t.wind), 0, &knobs::WIND);

    list.section("Foliage");
    list.push(
        "Generate leaves",
        "Grow foliage at all.",
        Kind::Toggle,
        |t: &mut T| Some(Value::Toggle(t.leaves.enabled)),
        |t: &mut T, v| match v {
            Value::Toggle(on) => {
                t.leaves.enabled = on;
                true
            }
            _ => false,
        },
    );
    list.knobs(|t, _| Some(&mut t.leaves), 0, knobs::FOLIAGE);
    list.push(
        "Leaf from level",
        "Leaves grow on stems at this level and deeper, so the canopy sits on twigs rather than limbs.",
        Kind::Count { range: knobs::LEAF_MIN_LEVEL },
        |t: &mut T| Some(Value::Count(u32::from(t.leaves.min_level))),
        |t: &mut T, v| match v {
            Value::Count(n) => {
                t.leaves.min_level = n.min(u32::from(u8::MAX)) as u8;
                true
            }
            _ => false,
        },
    );
    for group in knobs::FOLIAGE_MORE {
        list.group("", |t, _| Some(&mut t.leaves), 0, group);
    }

    list.section("Bark & roots");
    for group in knobs::BARK {
        list.group("", |t, _| Some(&mut t.mesh), 0, group);
    }
    for group in knobs::IRREGULARITY {
        list.group("", |t, _| Some(&mut t.mesh.irregularity), 0, group);
    }
    list.out
}

/// How a level is laid along its parent, and the controls of the pattern it has now.
fn pattern_controls(list: &mut List<SpeciesTemplate>, level: usize, pattern: &ChildPattern<Ranged>) {
    type T = SpeciesTemplate;
    fn pattern_at(t: &mut T, level: usize) -> Option<&mut ChildPattern<Ranged>> {
        knobs::spawn_mut(t, level).map(|s| &mut s.pattern)
    }
    list.push(
        "Pattern",
        "How children are laid along the parent: in rings at intervals, or at a steady rate per metre.",
        Kind::Choice { options: PATTERNS },
        move |t: &mut T| {
            pattern_at(t, level).map(|p| {
                Value::Choice(match p {
                    ChildPattern::None => 0,
                    ChildPattern::Whorl { .. } => 1,
                    ChildPattern::Continuous { .. } => 2,
                })
            })
        },
        move |t: &mut T, v| {
            let (Some(p), Value::Choice(picked)) = (pattern_at(t, level), v) else { return false };
            let current = match p {
                ChildPattern::None => 0,
                ChildPattern::Whorl { .. } => 1,
                ChildPattern::Continuous { .. } => 2,
            };
            // The same defaults the viewer switches to, and nothing lost by picking the
            // pattern a level already has.
            if picked != current {
                *p = match picked {
                    1 => ChildPattern::Whorl { every: 2, count: 4 },
                    2 => ChildPattern::Continuous { density: Ranged::Fixed(2.0) },
                    _ => ChildPattern::None,
                };
            }
            true
        },
    );
    match pattern {
        ChildPattern::None => {}
        ChildPattern::Whorl { .. } => {
            for (label, help, range, which) in [
                ("Whorl every", "A whorl on every this-many segments of the parent.", knobs::WHORL_EVERY, 0),
                ("Per whorl", "Branches in each whorl.", knobs::WHORL_COUNT, 1),
            ] {
                list.push(
                    label,
                    help,
                    Kind::Count { range },
                    move |t: &mut T| match pattern_at(t, level) {
                        Some(ChildPattern::Whorl { every, count }) => Some(Value::Count([*every, *count][which])),
                        _ => None,
                    },
                    move |t: &mut T, v| match (pattern_at(t, level), v) {
                        (Some(ChildPattern::Whorl { every, count }), Value::Count(n)) => {
                            *[every, count][which] = n;
                            true
                        }
                        _ => false,
                    },
                );
            }
        }
        ChildPattern::Continuous { .. } => {
            list.push(
                "Density",
                "Children per metre of the parent.",
                Kind::Ranged { range: knobs::DENSITY, log: true },
                move |t: &mut T| match pattern_at(t, level) {
                    Some(ChildPattern::Continuous { density }) => Some(Value::Ranged(*density)),
                    _ => None,
                },
                move |t: &mut T, v| match (pattern_at(t, level), v) {
                    (Some(ChildPattern::Continuous { density }), Value::Ranged(r)) => {
                        *density = r;
                        true
                    }
                    _ => false,
                },
            );
        }
    }
}

/// Every control of a clump of ground cover.
pub fn cover(template: &CoverTemplate) -> Vec<Control<CoverTemplate>> {
    let mut list = List::<CoverTemplate>::new();
    list.section("Clump");
    for group in knobs::cover::GROUPS {
        list.regrows = !std::ptr::eq(group, &knobs::cover::WIND);
        list.group("", |t, _| Some(t), 0, group);
    }
    list.regrows = true;
    let lods = template.lod.len().min(knobs::cover::LODS.len());
    for group in &knobs::cover::LODS[..lods] {
        list.group("", |t, _| Some(t), 0, group);
    }
    list.out
}

/// Every control of a rock.
pub fn rock(template: &RockTemplate) -> Vec<Control<RockTemplate>> {
    let mut list = List::<RockTemplate>::new();
    list.section("Rock");
    for group in knobs::rock::GROUPS {
        list.group("", |t, _| Some(t), 0, group);
    }
    if template.face.is_some() {
        list.group("", |t, _| t.face.as_mut(), 0, &knobs::rock::FACE);
    }
    list.out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cover::{builtin_cover_presets, parse_cover_template};
    use crate::rocks::{builtin_rock_presets, parse_rock_template};
    use crate::species::{builtin_presets, parse_template};
    use std::collections::HashSet;

    fn check<T: Clone>(name: &str, template: &T, controls: &[Control<T>]) {
        let mut ids = HashSet::new();
        for c in controls {
            assert!(ids.insert(c.id.clone()), "{name}: two controls are {}", c.id);
            let mut t = template.clone();
            let v = c.get(&mut t).unwrap_or_else(|| panic!("{name}: {} reads nothing", c.id));
            assert!(c.set(&mut t, v), "{name}: {} would not take back its own value", c.id);
        }
    }

    #[test]
    fn every_preset_s_controls_read_and_write_and_have_ids_of_their_own() {
        for (name, text) in builtin_presets() {
            let t = parse_template(text).unwrap();
            check(name, &t, &tree(&t));
        }
        for (name, text) in builtin_cover_presets() {
            let t = parse_cover_template(text).unwrap();
            check(name, &t, &cover(&t));
        }
        for (name, text) in builtin_rock_presets() {
            let t = parse_rock_template(text).unwrap();
            check(name, &t, &rock(&t));
        }
    }

    #[test]
    fn a_control_writes_the_field_it_names() {
        let (_, text) = builtin_presets().into_iter().find(|(n, _)| *n == "oak").unwrap();
        let mut t = parse_template(text).unwrap();
        let controls = tree(&t);
        let length = controls.iter().find(|c| c.id == "level_1.length_thickness.length").unwrap();
        assert!(length.set(&mut t, Value::Ranged(Ranged::Between(3.0, 5.0))));
        assert_eq!(t.branch_levels[0].length, Ranged::Between(3.0, 5.0));
        assert!(!length.set(&mut t, Value::Count(2)), "a count is not a length");
    }

    #[test]
    fn switching_a_level_to_whorls_offers_whorl_controls() {
        let (_, text) = builtin_presets().into_iter().find(|(n, _)| *n == "oak").unwrap();
        let mut t = parse_template(text).unwrap();
        let pattern = tree(&t).into_iter().find(|c| c.id == "level_1.spawning_off_trunk.pattern").unwrap();
        assert!(pattern.set(&mut t, Value::Choice(1)));
        let after = tree(&t);
        assert!(after.iter().any(|c| c.id == "level_1.spawning_off_trunk.per_whorl"));
        assert!(!after.iter().any(|c| c.id == "level_1.spawning_off_trunk.density"));
    }
}
