//! A C API over arbor-core, for an engine that grows Arbor's plants and draws them
//! itself. `include/arbor.h` declares it.
//!
//! A recipe is one preset with its seed: a species of tree, shrub or vine, a clump of
//! ground cover, or a rock. Its controls are the ones the viewer's panel offers, as a
//! flat list (`arbor_core::controls`), and generating it gives a `.glb` with the
//! textures in it and, for trees and cover, the `ARBOR_tree_wind` sway data.
//!
//! Nothing here keeps state between recipes but the texture folder. A recipe may be
//! used from any thread, one at a time; clone it to grow it on another thread while
//! the first is being edited. Clones share the textures already prepared, which are
//! most of what a tree's export costs.
//!
//! Every call catches panics. A call that fails returns false or null and leaves the
//! reason in `arbor_last_error`.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arbor_core::controls::{self, Control, Kind, Value};
use arbor_core::cover::{CoverTemplate, build_cover, builtin_cover_presets, parse_cover_template};
use arbor_core::gltf::{self, CoverTextures, ExportOptions, Exporter, RockTextures};
use arbor_core::rocks::{RockTemplate, build_rock, builtin_rock_presets, parse_rock_template};
use arbor_core::species::{SpeciesTemplate, builtin_presets, parse_template};
use arbor_core::{Ranged, build_leaves, build_mesh, grow};

pub const ARBOR_TREE: u32 = 0;
pub const ARBOR_COVER: u32 = 1;
pub const ARBOR_ROCK: u32 = 2;

pub const ARBOR_CONTROL_RANGED: u32 = 0;
pub const ARBOR_CONTROL_COUNT: u32 = 1;
pub const ARBOR_CONTROL_CHOICE: u32 = 2;
pub const ARBOR_CONTROL_TOGGLE: u32 = 3;

#[repr(C)]
pub struct ArborBytes {
    pub data: *mut u8,
    pub len: usize,
}

/// One control, described. The strings belong to the recipe and last until its
/// controls next change shape or it is freed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArborControlInfo {
    pub id: *const c_char,
    pub section: *const c_char,
    pub group: *const c_char,
    pub label: *const c_char,
    pub help: *const c_char,
    pub kind: u32,
    /// The slider's span, for a ranged control.
    pub min: f32,
    pub max: f32,
    pub log: u32,
    /// The slider's span, for a count.
    pub count_min: u32,
    pub count_max: u32,
    /// Options of a choice, read with `arbor_control_option`.
    pub option_count: u32,
    /// Zero for what only the shaders read: changing it grows nothing new.
    pub regrows: u32,
}

/// A control's value. A ranged one uses `lo`, `hi` and `is_range`; a count, a choice
/// and a toggle use `integer`. `landed` is read-only: where this seed lands a range,
/// which is the value to keep when a range is folded back to one number.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ArborValue {
    pub is_range: u32,
    pub lo: f32,
    pub hi: f32,
    pub landed: f32,
    pub integer: u32,
}

// ---------------------------------------------------------------------------------
// The three kinds behind one handle

/// What a recipe needs of its template, for each kind of plant.
trait Template: Clone + Send + 'static {
    fn controls(&self) -> Vec<Control<Self>>;
    fn landed(&self) -> Self;
    fn seed(&self) -> u64;
    fn set_seed(&mut self, seed: u64);
    fn name(&self) -> &str;
    fn to_ron(&self) -> Result<String, String>;
    fn glb(&self, cache: &Cache) -> Result<Vec<u8>, String>;
}

/// Textures prepared for the last thing generated, with what they were prepared from.
#[derive(Default)]
enum Prepared {
    #[default]
    None,
    Tree(String, Arc<Exporter>),
    Cover(String, Arc<CoverTextures>),
}

type Cache = Arc<Mutex<Prepared>>;

static TEXTURE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

fn texture_dir() -> Option<PathBuf> {
    TEXTURE_DIR.lock().ok().and_then(|d| d.clone())
}

impl Template for SpeciesTemplate {
    fn controls(&self) -> Vec<Control<Self>> {
        controls::tree(self)
    }
    fn landed(&self) -> Self {
        self.instance().template()
    }
    fn seed(&self) -> u64 {
        self.seed
    }
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn to_ron(&self) -> Result<String, String> {
        ron::ser::to_string_pretty(self, Default::default()).map_err(|e| e.to_string())
    }
    fn glb(&self, cache: &Cache) -> Result<Vec<u8>, String> {
        let params = self.instance();
        let skeleton = grow(&params);
        let mesh = build_mesh(&skeleton, &params);
        let leaves = build_leaves(&skeleton, &params);

        // Everything `gltf::Textures::load` bakes into the maps: the bark and leaf art,
        // the dead-wood bleach, the leaf atlas, and the tint ceiling.
        let (mp, lp) = (&params.mesh, &params.leaves);
        let key = format!(
            "{:?} {:?}",
            (texture_dir(), &mp.bark_texture, mp.dead_wood_weathering, mp.bark_tint, mp.dead_wood_color),
            (
                lp.enabled,
                &lp.texture,
                lp.atlas_cols,
                lp.atlas_rows,
                &lp.cluster,
                lp.saturation,
                lp.transmission,
                arbor_core::leaves::tint_ceiling(&params),
            )
        );
        let exporter = {
            let mut prepared = cache.lock().map_err(|_| "texture cache poisoned".to_string())?;
            match &*prepared {
                Prepared::Tree(k, e) if *k == key => e.clone(),
                _ => {
                    let options = ExportOptions { textures: texture_dir(), wind: true };
                    let e = Arc::new(Exporter::new(&params, &options)?);
                    *prepared = Prepared::Tree(key, e.clone());
                    e
                }
            }
        };
        Ok(exporter.glb(&skeleton, &mesh, &leaves, &params))
    }
}

impl Template for CoverTemplate {
    fn controls(&self) -> Vec<Control<Self>> {
        controls::cover(self)
    }
    fn landed(&self) -> Self {
        self.instance().template()
    }
    fn seed(&self) -> u64 {
        self.seed
    }
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn to_ron(&self) -> Result<String, String> {
        ron::ser::to_string_pretty(self, Default::default()).map_err(|e| e.to_string())
    }
    fn glb(&self, cache: &Cache) -> Result<Vec<u8>, String> {
        let params = self.instance();
        let mesh = build_cover(&params);
        let key = format!("{:?}", params.atlas);
        let textures = {
            let mut prepared = cache.lock().map_err(|_| "texture cache poisoned".to_string())?;
            match &*prepared {
                Prepared::Cover(k, t) if *k == key => t.clone(),
                _ => {
                    let t = Arc::new(CoverTextures::bake(&params)?);
                    *prepared = Prepared::Cover(key, t.clone());
                    t
                }
            }
        };
        Ok(gltf::cover_glb(&mesh, &params, &textures))
    }
}

impl Template for RockTemplate {
    fn controls(&self) -> Vec<Control<Self>> {
        controls::rock(self)
    }
    fn landed(&self) -> Self {
        self.instance().template()
    }
    fn seed(&self) -> u64 {
        self.seed
    }
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn to_ron(&self) -> Result<String, String> {
        ron::ser::to_string_pretty(self, Default::default()).map_err(|e| e.to_string())
    }
    // Every rock has maps of its own: they are baked from its mesh.
    fn glb(&self, _cache: &Cache) -> Result<Vec<u8>, String> {
        let params = self.instance();
        let mesh = build_rock(&params);
        let textures = RockTextures::bake(&params, &mesh)?;
        Ok(gltf::rock_glb(&mesh, &params, &textures))
    }
}

/// A control's strings, kept alive for C.
struct Described {
    id: CString,
    section: CString,
    group: CString,
    label: CString,
    help: CString,
    options: Vec<CString>,
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap_or_default()
}

struct Held<T: Template> {
    template: T,
    controls: Option<(Vec<Control<T>>, Vec<Described>)>,
    landed: Option<T>,
}

impl<T: Template> Held<T> {
    fn new(template: T) -> Self {
        Self { template, controls: None, landed: None }
    }

    fn controls(&mut self) -> &(Vec<Control<T>>, Vec<Described>) {
        self.controls.get_or_insert_with(|| {
            let list = self.template.controls();
            let described = list
                .iter()
                .map(|c| Described {
                    id: c_string(&c.id),
                    section: c_string(&c.section),
                    group: c_string(&c.group),
                    label: c_string(c.label),
                    help: c_string(c.help),
                    options: match c.kind {
                        Kind::Choice { options } => options.iter().map(|o| c_string(o)).collect(),
                        _ => Vec::new(),
                    },
                })
                .collect();
            (list, described)
        })
    }

    /// Forgets what was worked out from the template after a change to it. The controls
    /// are kept unless the change moved them, so their strings stay where C has them.
    fn changed(&mut self) {
        self.landed = None;
        let ids_now: Vec<String> = self.template.controls().into_iter().map(|c| c.id).collect();
        if let Some((list, _)) = &self.controls
            && (list.len() != ids_now.len() || list.iter().zip(&ids_now).any(|(c, id)| c.id != *id))
        {
            self.controls = None;
        }
    }
}

/// One recipe behind the C handle, whatever its kind.
trait Recipe: Send {
    fn kind(&self) -> u32;
    fn clone_box(&self) -> Box<dyn Recipe>;
    fn control_count(&mut self) -> usize;
    fn control_info(&mut self, i: usize) -> Option<ArborControlInfo>;
    fn control_option(&mut self, i: usize, option: usize) -> *const c_char;
    fn control_get(&mut self, i: usize) -> Option<ArborValue>;
    /// Whether the value was taken, and whether the control list changed shape.
    fn control_set(&mut self, i: usize, value: &ArborValue) -> Option<bool>;
    fn seed(&self) -> u64;
    fn set_seed(&mut self, seed: u64);
    fn name(&self) -> &str;
    fn to_ron(&self) -> Result<String, String>;
    fn glb(&self, cache: &Cache) -> Result<Vec<u8>, String>;
}

impl<T: Template> Recipe for Held<T> {
    fn kind(&self) -> u32 {
        let any: &dyn std::any::Any = &self.template;
        if any.is::<SpeciesTemplate>() {
            ARBOR_TREE
        } else if any.is::<CoverTemplate>() {
            ARBOR_COVER
        } else {
            ARBOR_ROCK
        }
    }

    fn clone_box(&self) -> Box<dyn Recipe> {
        Box::new(Held::new(self.template.clone()))
    }

    fn control_count(&mut self) -> usize {
        self.controls().0.len()
    }

    fn control_info(&mut self, i: usize) -> Option<ArborControlInfo> {
        let (list, described) = self.controls();
        let (c, d) = (list.get(i)?, &described[i]);
        let mut info = ArborControlInfo {
            id: d.id.as_ptr(),
            section: d.section.as_ptr(),
            group: d.group.as_ptr(),
            label: d.label.as_ptr(),
            help: d.help.as_ptr(),
            kind: 0,
            min: 0.0,
            max: 0.0,
            log: 0,
            count_min: 0,
            count_max: 0,
            option_count: 0,
            regrows: u32::from(c.regrows),
        };
        match c.kind {
            Kind::Ranged { range, log } => {
                info.kind = ARBOR_CONTROL_RANGED;
                (info.min, info.max, info.log) = (range.0, range.1, u32::from(log));
            }
            Kind::Count { range } => {
                info.kind = ARBOR_CONTROL_COUNT;
                (info.count_min, info.count_max) = range;
            }
            Kind::Choice { options } => {
                info.kind = ARBOR_CONTROL_CHOICE;
                info.option_count = options.len() as u32;
            }
            Kind::Toggle => info.kind = ARBOR_CONTROL_TOGGLE,
        }
        Some(info)
    }

    fn control_option(&mut self, i: usize, option: usize) -> *const c_char {
        let (_, described) = self.controls();
        described.get(i).and_then(|d| d.options.get(option)).map_or(std::ptr::null(), |o| o.as_ptr())
    }

    fn control_get(&mut self, i: usize) -> Option<ArborValue> {
        self.controls();
        let landed = self.landed.get_or_insert_with(|| self.template.landed());
        let (list, _) = self.controls.as_ref()?;
        let c = list.get(i)?;
        let value = c.get(&mut self.template)?;
        Some(match value {
            Value::Ranged(r) => ArborValue {
                is_range: u32::from(r.is_range()),
                lo: r.lo(),
                hi: r.hi(),
                landed: match c.get(landed) {
                    Some(Value::Ranged(at)) => at.lo(),
                    _ => r.lo(),
                },
                integer: 0,
            },
            Value::Count(n) | Value::Choice(n) => ArborValue { integer: n, ..Default::default() },
            Value::Toggle(on) => ArborValue { integer: u32::from(on), ..Default::default() },
        })
    }

    fn control_set(&mut self, i: usize, value: &ArborValue) -> Option<bool> {
        self.controls();
        let (list, _) = self.controls.as_ref()?;
        let c = list.get(i)?;
        let value = match c.kind {
            Kind::Ranged { .. } if value.is_range != 0 => Value::Ranged(Ranged::Between(value.lo, value.hi)),
            Kind::Ranged { .. } => Value::Ranged(Ranged::Fixed(value.lo)),
            Kind::Count { .. } => Value::Count(value.integer),
            Kind::Choice { options } if (value.integer as usize) < options.len() => Value::Choice(value.integer),
            Kind::Choice { .. } => return None,
            Kind::Toggle => Value::Toggle(value.integer != 0),
        };
        if !c.set(&mut self.template, value) {
            return None;
        }
        self.changed();
        Some(self.controls.is_none())
    }

    fn seed(&self) -> u64 {
        self.template.seed()
    }

    fn set_seed(&mut self, seed: u64) {
        self.template.set_seed(seed);
        self.landed = None;
    }

    fn name(&self) -> &str {
        self.template.name()
    }

    fn to_ron(&self) -> Result<String, String> {
        self.template.to_ron()
    }

    fn glb(&self, cache: &Cache) -> Result<Vec<u8>, String> {
        self.template.glb(cache)
    }
}

/// The C handle.
pub struct ArborRecipe {
    recipe: Box<dyn Recipe>,
    cache: Cache,
    /// The last string handed out by `arbor_recipe_name` or `arbor_recipe_to_ron`.
    text: CString,
}

impl ArborRecipe {
    fn new(recipe: Box<dyn Recipe>) -> *mut ArborRecipe {
        Box::into_raw(Box::new(ArborRecipe { recipe, cache: Cache::default(), text: CString::default() }))
    }
}

// ---------------------------------------------------------------------------------
// Errors and panics

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(message: impl AsRef<str>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = c_string(message.as_ref()));
}

/// Runs `f`, turning a panic into `otherwise` and an error message.
fn guard<R>(otherwise: R, f: impl FnOnce() -> Result<R, String>) -> R {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            fail(e);
            otherwise
        }
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked".to_string());
            fail(format!("arbor panicked: {message}"));
            otherwise
        }
    }
}

/// # Safety
/// `s` is null or a NUL-terminated string.
unsafe fn read_str<'a>(s: *const c_char) -> Result<&'a str, String> {
    if s.is_null() {
        return Err("null string".to_string());
    }
    unsafe { CStr::from_ptr(s) }.to_str().map_err(|e| e.to_string())
}

/// # Safety
/// `r` is null or a live handle.
unsafe fn recipe<'a>(r: *mut ArborRecipe) -> Result<&'a mut ArborRecipe, String> {
    unsafe { r.as_mut() }.ok_or_else(|| "null recipe".to_string())
}

fn presets(kind: u32) -> Vec<(&'static str, &'static str)> {
    match kind {
        ARBOR_TREE => builtin_presets(),
        ARBOR_COVER => builtin_cover_presets(),
        ARBOR_ROCK => builtin_rock_presets(),
        _ => Vec::new(),
    }
}

fn parse(kind: u32, ron: &str) -> Result<Box<dyn Recipe>, String> {
    Ok(match kind {
        ARBOR_TREE => Box::new(Held::new(parse_template(ron)?)),
        ARBOR_COVER => Box::new(Held::new(parse_cover_template(ron)?)),
        ARBOR_ROCK => Box::new(Held::new(parse_rock_template(ron)?)),
        _ => return Err(format!("no kind {kind}")),
    })
}

// ---------------------------------------------------------------------------------
// The API

/// Why the last call on this thread that failed did. Never null.
#[unsafe(no_mangle)]
pub extern "C" fn arbor_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Where tree textures are read from (`<name>_albedo.png` and so on). Until this is
/// called trees are exported untextured. Cover and rocks bake their own maps.
///
/// # Safety
/// `dir` is null, to forget the folder, or a NUL-terminated UTF-8 path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_set_texture_dir(dir: *const c_char) -> bool {
    guard(false, || {
        let path = if dir.is_null() { None } else { Some(PathBuf::from(unsafe { read_str(dir) }?)) };
        *TEXTURE_DIR.lock().map_err(|_| "texture folder poisoned".to_string())? = path;
        Ok(true)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn arbor_preset_count(kind: u32) -> u32 {
    guard(0, || Ok(presets(kind).len() as u32))
}

/// The built-in preset's name, static. Null past the end.
#[unsafe(no_mangle)]
pub extern "C" fn arbor_preset_name(kind: u32, i: u32) -> *const c_char {
    static NAMES: Mutex<Vec<(u32, Vec<CString>)>> = Mutex::new(Vec::new());
    guard(std::ptr::null(), || {
        let mut names = NAMES.lock().map_err(|_| "preset names poisoned".to_string())?;
        if !names.iter().any(|(k, _)| *k == kind) {
            names.push((kind, presets(kind).iter().map(|(n, _)| c_string(n)).collect()));
        }
        let (_, list) = names.iter().find(|(k, _)| *k == kind).expect("just added");
        // The CStrings live in the static and are never dropped or moved out.
        Ok(list.get(i as usize).map_or(std::ptr::null(), |n| n.as_ptr()))
    })
}

/// A built-in preset by name. Free it with `arbor_recipe_free`.
///
/// # Safety
/// `name` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_from_preset(kind: u32, name: *const c_char) -> *mut ArborRecipe {
    guard(std::ptr::null_mut(), || {
        let name = unsafe { read_str(name) }?;
        let (_, text) = presets(kind)
            .into_iter()
            .find(|(n, _)| *n == name)
            .ok_or_else(|| format!("no built-in preset {name}"))?;
        Ok(ArborRecipe::new(parse(kind, text)?))
    })
}

/// A recipe from preset text, as `arbor_recipe_to_ron` or a saved preset gives it.
///
/// # Safety
/// `ron` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_from_ron(kind: u32, ron: *const c_char) -> *mut ArborRecipe {
    guard(std::ptr::null_mut(), || Ok(ArborRecipe::new(parse(kind, unsafe { read_str(ron) }?)?)))
}

/// A copy to grow on another thread. It shares the textures already prepared.
///
/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_clone(r: *mut ArborRecipe) -> *mut ArborRecipe {
    guard(std::ptr::null_mut(), || {
        let r = unsafe { recipe(r) }?;
        Ok(Box::into_raw(Box::new(ArborRecipe {
            recipe: r.recipe.clone_box(),
            cache: r.cache.clone(),
            text: CString::default(),
        })))
    })
}

/// # Safety
/// `r` is null or a live handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_free(r: *mut ArborRecipe) {
    if !r.is_null() {
        drop(unsafe { Box::from_raw(r) });
    }
}

/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_kind(r: *mut ArborRecipe) -> u32 {
    guard(u32::MAX, || Ok(unsafe { recipe(r) }?.recipe.kind()))
}

/// The preset's own name. Lasts until the next call that returns text from `r`.
///
/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_name(r: *mut ArborRecipe) -> *const c_char {
    guard(std::ptr::null(), || {
        let r = unsafe { recipe(r) }?;
        r.text = c_string(r.recipe.name());
        Ok(r.text.as_ptr())
    })
}

/// The whole recipe as preset text, ranges and all, to store and grow again. Lasts
/// until the next call that returns text from `r`.
///
/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_recipe_to_ron(r: *mut ArborRecipe) -> *const c_char {
    guard(std::ptr::null(), || {
        let r = unsafe { recipe(r) }?;
        r.text = c_string(&r.recipe.to_ron()?);
        Ok(r.text.as_ptr())
    })
}

/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_get_seed(r: *mut ArborRecipe) -> u64 {
    guard(0, || Ok(unsafe { recipe(r) }?.recipe.seed()))
}

/// Sets the seed, which decides where every range lands as well as how the plant grows.
///
/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_set_seed(r: *mut ArborRecipe, seed: u64) -> bool {
    guard(false, || {
        unsafe { recipe(r) }?.recipe.set_seed(seed);
        Ok(true)
    })
}

/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_control_count(r: *mut ArborRecipe) -> u32 {
    guard(0, || Ok(unsafe { recipe(r) }?.recipe.control_count() as u32))
}

/// # Safety
/// `r` is a live handle and `out` points at an `ArborControlInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_control_info(r: *mut ArborRecipe, i: u32, out: *mut ArborControlInfo) -> bool {
    guard(false, || {
        let info = unsafe { recipe(r) }?.recipe.control_info(i as usize).ok_or_else(|| format!("no control {i}"))?;
        *unsafe { out.as_mut() }.ok_or("null out")? = info;
        Ok(true)
    })
}

/// The name of a choice's option. Lasts as `arbor_control_info`'s strings do.
///
/// # Safety
/// `r` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_control_option(r: *mut ArborRecipe, i: u32, option: u32) -> *const c_char {
    guard(std::ptr::null(), || Ok(unsafe { recipe(r) }?.recipe.control_option(i as usize, option as usize)))
}

/// # Safety
/// `r` is a live handle and `out` points at an `ArborValue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_control_get(r: *mut ArborRecipe, i: u32, out: *mut ArborValue) -> bool {
    guard(false, || {
        let value = unsafe { recipe(r) }?.recipe.control_get(i as usize).ok_or_else(|| format!("no control {i}"))?;
        *unsafe { out.as_mut() }.ok_or("null out")? = value;
        Ok(true)
    })
}

/// Sets a control. `reshaped`, when not null, says whether the control list changed
/// shape — a level's pattern switched, a level added — so it has to be read again,
/// and the strings of the old one are gone.
///
/// # Safety
/// `r` is a live handle, `value` points at an `ArborValue`, `reshaped` is null or
/// points at a bool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_control_set(
    r: *mut ArborRecipe,
    i: u32,
    value: *const ArborValue,
    reshaped: *mut bool,
) -> bool {
    guard(false, || {
        let value = unsafe { value.as_ref() }.ok_or("null value")?;
        let moved = unsafe { recipe(r) }?
            .recipe
            .control_set(i as usize, value)
            .ok_or_else(|| format!("control {i} does not take that value"))?;
        if let Some(out) = unsafe { reshaped.as_mut() } {
            *out = moved;
        }
        Ok(true)
    })
}

/// Grows the recipe and writes it as one `.glb`, textures inside. Free the bytes with
/// `arbor_bytes_free`.
///
/// # Safety
/// `r` is a live handle and `out` points at an `ArborBytes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_generate_glb(r: *mut ArborRecipe, out: *mut ArborBytes) -> bool {
    guard(false, || {
        let r = unsafe { recipe(r) }?;
        let out = unsafe { out.as_mut() }.ok_or("null out")?;
        let bytes = r.recipe.glb(&r.cache)?.into_boxed_slice();
        out.len = bytes.len();
        out.data = Box::into_raw(bytes) as *mut u8;
        Ok(true)
    })
}

/// # Safety
/// `bytes` came from `arbor_generate_glb` and is not freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbor_bytes_free(bytes: *mut ArborBytes) {
    if let Some(b) = unsafe { bytes.as_mut() }
        && !b.data.is_null()
    {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.data, b.len)) });
        b.data = std::ptr::null_mut();
        b.len = 0;
    }
}
