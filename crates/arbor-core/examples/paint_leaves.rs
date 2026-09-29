//! Paints the procedural leaf sets into `assets/textures`: ivy, hazel, box and the
//! heart-shaped climber's leaf. See `leaf_art`.
//!
//! ```text
//! cargo run --release -p arbor-core --example paint_leaves [name ...]
//! ```
//!
//! With no names it paints every set.

use std::path::Path;

use arbor_core::leaf_art::LeafArt;
use arbor_core::textures::{encode_png, map_file, TEXTURE_DIR};

fn main() {
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(TEXTURE_DIR);
    for art in LeafArt::all() {
        if !wanted.is_empty() && !wanted.iter().any(|w| w == art.name) {
            continue;
        }
        let maps = art.paint();
        let sets = [
            ("albedo", Some(&maps.albedo)),
            ("normal", maps.normal.as_ref()),
            ("roughness", maps.roughness.as_ref()),
        ];
        for (map, bitmap) in sets {
            let Some(bitmap) = bitmap else { continue };
            let path = dir.join(map_file(art.name, map));
            let png = encode_png(bitmap).unwrap_or_else(|e| panic!("{e}"));
            std::fs::write(&path, png).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
            println!("wrote {}", path.display());
        }
    }
}
