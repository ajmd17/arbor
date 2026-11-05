//! Drives the C API the way an engine would, through the extern functions.

use std::ffi::{CStr, CString};

use arbor_ffi::*;

fn text(p: *const std::ffi::c_char) -> String {
    assert!(!p.is_null(), "null: {}", text(arbor_last_error()));
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn preset(kind: u32, name: &str) -> *mut ArborRecipe {
    let name = CString::new(name).unwrap();
    let r = unsafe { arbor_recipe_from_preset(kind, name.as_ptr()) };
    assert!(!r.is_null(), "{}", text(arbor_last_error()));
    r
}

fn glb(r: *mut ArborRecipe) -> Vec<u8> {
    let mut bytes = ArborBytes { data: std::ptr::null_mut(), len: 0 };
    assert!(unsafe { arbor_generate_glb(r, &mut bytes) }, "{}", text(arbor_last_error()));
    let out = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
    unsafe { arbor_bytes_free(&mut bytes) };
    assert!(bytes.data.is_null());
    out
}

/// The JSON chunk of a `.glb`.
fn json(glb: &[u8]) -> String {
    assert_eq!(&glb[..4], b"glTF");
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    assert_eq!(&glb[16..20], b"JSON");
    String::from_utf8(glb[20..20 + len].to_vec()).unwrap()
}

fn find(r: *mut ArborRecipe, id: &str) -> u32 {
    let mut info = std::mem::MaybeUninit::<ArborControlInfo>::uninit();
    (0..unsafe { arbor_control_count(r) })
        .find(|&i| {
            assert!(unsafe { arbor_control_info(r, i, info.as_mut_ptr()) });
            text(unsafe { info.assume_init() }.id) == id
        })
        .unwrap_or_else(|| panic!("no control {id}"))
}

#[test]
fn every_kind_has_presets_and_grows_one_into_a_glb() {
    let dir = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/textures")).unwrap();
    assert!(unsafe { arbor_set_texture_dir(dir.as_ptr()) });
    for (kind, name, extension) in [
        (ARBOR_TREE, "hazel", "ARBOR_tree_wind"),
        (ARBOR_COVER, "short_grass", "ARBOR_ground_cover"),
        (ARBOR_ROCK, "stone", "ARBOR_rock"),
    ] {
        let names: Vec<String> = (0..arbor_preset_count(kind)).map(|i| text(arbor_preset_name(kind, i))).collect();
        assert!(names.iter().any(|n| n == name), "{name} not among {names:?}");
        assert!(arbor_preset_name(kind, names.len() as u32).is_null());

        let r = preset(kind, name);
        assert_eq!(unsafe { arbor_recipe_kind(r) }, kind);
        let doc = json(&glb(r));
        assert!(doc.contains(extension), "{name}'s glb has no {extension}");
        assert!(doc.contains("\"images\""), "{name}'s glb has no textures");
        unsafe { arbor_recipe_free(r) };
    }
}

#[test]
fn a_control_set_changes_the_tree_and_survives_preset_text() {
    let r = preset(ARBOR_TREE, "hazel");
    let before = glb(r);

    let i = find(r, "level_1.length_thickness.length");
    let mut v = ArborValue::default();
    assert!(unsafe { arbor_control_get(r, i, &mut v) });
    let value = ArborValue { is_range: 0, lo: v.lo * 0.5, ..v };
    let mut reshaped = true;
    assert!(unsafe { arbor_control_set(r, i, &value, &mut reshaped) }, "{}", text(arbor_last_error()));
    assert!(!reshaped, "a length moves no controls");
    assert_ne!(glb(r), before, "halving the limbs changed nothing");

    let ron = text(unsafe { arbor_recipe_to_ron(r) });
    let ron_c = CString::new(ron.clone()).unwrap();
    let again = unsafe { arbor_recipe_from_ron(ARBOR_TREE, ron_c.as_ptr()) };
    assert!(!again.is_null(), "{}", text(arbor_last_error()));
    assert_eq!(text(unsafe { arbor_recipe_to_ron(again) }), ron);
    let mut w = ArborValue::default();
    assert!(unsafe { arbor_control_get(again, find(again, "level_1.length_thickness.length"), &mut w) });
    assert_eq!(w.lo, value.lo);
    unsafe {
        arbor_recipe_free(again);
        arbor_recipe_free(r);
    }
}

#[test]
fn switching_a_pattern_reshapes_the_controls() {
    let r = preset(ARBOR_TREE, "oak");
    let i = find(r, "level_1.spawning_off_trunk.pattern");
    let mut v = ArborValue::default();
    assert!(unsafe { arbor_control_get(r, i, &mut v) });
    let other = ArborValue { integer: if v.integer == 1 { 2 } else { 1 }, ..v };
    let mut reshaped = false;
    assert!(unsafe { arbor_control_set(r, i, &other, &mut reshaped) });
    assert!(reshaped);
    let bad = ArborValue { integer: 9, ..v };
    assert!(!unsafe { arbor_control_set(r, i, &bad, std::ptr::null_mut()) }, "there is no ninth pattern");
    assert!(!text(arbor_last_error()).is_empty());
    unsafe { arbor_recipe_free(r) };
}

#[test]
fn a_clone_grows_the_same_tree_and_the_seed_moves_it() {
    let r = preset(ARBOR_TREE, "juniper");
    let copy = unsafe { arbor_recipe_clone(r) };
    assert_eq!(glb(r), glb(copy));
    assert!(unsafe { arbor_set_seed(copy, arbor_get_seed(r) + 1) });
    assert_ne!(glb(r), glb(copy));
    unsafe {
        arbor_recipe_free(copy);
        arbor_recipe_free(r);
    }
}

#[test]
fn nonsense_fails_with_a_reason() {
    let name = CString::new("no_such_tree").unwrap();
    assert!(unsafe { arbor_recipe_from_preset(ARBOR_TREE, name.as_ptr()) }.is_null());
    assert!(text(arbor_last_error()).contains("no_such_tree"));
    let junk = CString::new("(((").unwrap();
    assert!(unsafe { arbor_recipe_from_ron(ARBOR_ROCK, junk.as_ptr()) }.is_null());
    assert!(unsafe { arbor_recipe_from_preset(7, name.as_ptr()) }.is_null());
}

#[test]
fn trees_take_their_maps_from_a_reader_when_one_is_set() {
    use std::sync::Mutex;

    static ASKED: Mutex<Vec<String>> = Mutex::new(Vec::new());

    unsafe extern "C" fn reader(_: *mut std::ffi::c_void, file: *const std::ffi::c_char, sink: *mut ArborMapSink) -> bool {
        let file = unsafe { CStr::from_ptr(file) }.to_string_lossy().into_owned();
        ASKED.lock().unwrap().push(file.clone());
        let pixels = [128u8; 8 * 8 * 4];
        file.contains("_albedo") && unsafe { arbor_map_sink_put(sink, 8, 8, pixels.as_ptr()) }
    }

    assert!(unsafe { arbor_set_map_reader(Some(reader), std::ptr::null_mut()) });
    let r = preset(ARBOR_TREE, "hazel");
    let out = glb(r);
    unsafe { arbor_recipe_free(r) };
    assert!(unsafe { arbor_set_map_reader(None, std::ptr::null_mut()) });

    let asked = ASKED.lock().unwrap();
    assert!(asked.iter().any(|f| f.ends_with("_albedo.png")), "asked for {asked:?}");
    assert!(json(&out).contains("\"images\""), "no images in the glb");
}
