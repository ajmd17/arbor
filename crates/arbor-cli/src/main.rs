use arbor_core::gltf::{self, ExportOptions};
use arbor_core::species::{builtin_presets, parse_species, CUSTOM_PRESET_DIR};
use arbor_core::textures::TEXTURE_DIR;
use arbor_core::{build_leaves, build_mesh, grow, LeafMesh, Mesh, SpeciesParams};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        std::process::exit(2);
    }

    let mut species_src: Option<String> = None;
    let mut seed_override: Option<u64> = None;
    let mut obj_out: Option<String> = None;
    let mut gltf_out: Vec<String> = Vec::new();
    let mut texture_dir: Option<String> = Some(TEXTURE_DIR.to_string());
    let mut wind_data = false;
    let mut variations: u32 = 1;
    let mut no_leaves = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed_override = args.get(i).and_then(|s| s.parse().ok());
            }
            "--obj" => {
                i += 1;
                obj_out = args.get(i).cloned();
            }
            // Both name the file to write; the extension is what decides the format,
            // so either flag takes either, and both may be given.
            "--glb" | "--gltf" => {
                i += 1;
                match args.get(i) {
                    Some(path) => gltf_out.push(path.clone()),
                    None => {
                        eprintln!("{} needs a file to write", args[i - 1]);
                        std::process::exit(2);
                    }
                }
            }
            "--textures" => {
                i += 1;
                texture_dir = args.get(i).cloned();
            }
            "--no-textures" => texture_dir = None,
            "--wind-data" => wind_data = true,
            "--variations" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<u32>().ok()) {
                    Some(n) if n >= 1 => variations = n,
                    _ => {
                        eprintln!("--variations needs a count of one or more");
                        std::process::exit(2);
                    }
                }
            }
            "--no-leaves" => no_leaves = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => species_src = Some(other.to_string()),
        }
        i += 1;
    }

    let src = species_src.unwrap_or_else(|| "pine".to_string());
    // A built-in by name, then a preset saved from the viewer by name, then a path.
    let saved = std::path::Path::new(CUSTOM_PRESET_DIR).join(format!("{src}.ron"));
    let ron_text = match builtin_presets().into_iter().find(|(n, _)| *n == src) {
        Some((_, text)) => text.to_string(),
        None if saved.is_file() => std::fs::read_to_string(&saved).unwrap_or_else(|e| {
            eprintln!("cannot read {}: {e}", saved.display());
            std::process::exit(1);
        }),
        None => std::fs::read_to_string(&src).unwrap_or_else(|e| {
            eprintln!("cannot read species '{src}': {e}");
            std::process::exit(1);
        }),
    };

    let mut params = parse_species(&ron_text).unwrap_or_else(|e| {
        eprintln!("species parse error: {e}");
        std::process::exit(1);
    });
    if let Some(seed) = seed_override {
        params.seed = seed;
    }
    if no_leaves {
        params.leaves.enabled = false;
    }

    let t = std::time::Instant::now();
    let skeleton = grow(&params);
    let mesh = build_mesh(&skeleton, &params);
    let leaves = build_leaves(&skeleton, &params);
    let dt = t.elapsed();
    let stats = skeleton.stats();

    println!("species: {}", params.name);
    println!("seed:    {}", params.seed);
    println!("{stats:#?}");
    println!("verts:   {}", mesh.vertex_count());
    println!("tris:    {}", mesh.triangle_count());
    println!("leaves:  {}", leaves.leaf_count());
    println!("l.tris:  {}", leaves.triangle_count());
    println!("gen:     {:.3} ms", dt.as_secs_f64() * 1000.0);

    if let Some(path) = obj_out {
        write_obj(&mesh, &leaves, &params, &path).unwrap_or_else(|e| {
            eprintln!("obj write failed: {e}");
            std::process::exit(1);
        });
        println!("obj:     {path}");
    }

    let options = ExportOptions {
        textures: texture_dir.map(Into::into),
        wind: wind_data,
    };
    for path in gltf_out {
        let t = std::time::Instant::now();
        let path = std::path::Path::new(&path);
        let Some(format) = gltf::Format::from_path(path) else {
            eprintln!("{}: export to a .glb or a .gltf", path.display());
            std::process::exit(1);
        };
        let dir = path.parent().unwrap_or(std::path::Path::new(""));
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("tree");
        // The batch grows each tree again, which for the one already grown above is the
        // same tree: growth is deterministic.
        let report = gltf::export_batch(dir, stem, format, &params, variations, &options, |done| {
            if variations > 1 {
                eprint!("\rexporting {done}/{variations}");
            }
            true
        });
        if variations > 1 {
            eprintln!();
        }
        match report {
            Ok(report) => {
                for warning in &report.warnings {
                    eprintln!("warning: {warning}");
                }
                let what = match report.trees.as_slice() {
                    [one] => one.display().to_string(),
                    trees => format!(
                        "{} trees, {} to {}",
                        trees.len(),
                        trees[0].display(),
                        trees[trees.len() - 1].display()
                    ),
                };
                println!(
                    "gltf:    {what} ({:.1} MB, {:.0} ms)",
                    report.bytes as f64 / 1e6,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                eprintln!("gltf export failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn write_obj(
    mesh: &Mesh,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    path: &str,
) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::from("# arbor tree mesh\n");
    let emit = |out: &mut String,
                group: &str,
                positions: &[[f32; 3]],
                uvs: &[[f32; 2]],
                normals: &[[f32; 3]],
                indices: &[u32],
                base: u32| {
        let _ = writeln!(out, "g {group}");
        for p in positions {
            let _ = writeln!(out, "v {} {} {}", p[0], p[1], p[2]);
        }
        for uv in uvs {
            let _ = writeln!(out, "vt {} {}", uv[0], uv[1]);
        }
        for n in normals {
            let _ = writeln!(out, "vn {} {} {}", n[0], n[1], n[2]);
        }
        for tri in indices.chunks_exact(3) {
            let (a, b, c) = (
                tri[0] + base + 1,
                tri[1] + base + 1,
                tri[2] + base + 1,
            );
            let _ = writeln!(out, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
        }
    };

    emit(
        &mut out,
        "bark",
        &mesh.positions,
        &mesh.uvs,
        &mesh.normals,
        &mesh.indices,
        0,
    );

    if !leaves.is_empty() {
        // Leaf UVs are card-local so the viewer can flip a card to its back cell.
        // An exported mesh has no such shader, so bake the front cell in.
        let lp = &params.leaves;
        let (cols, rows) = (lp.atlas_cols.max(1), lp.atlas_rows.max(1));
        let cell = lp.atlas_front.min(cols * rows - 1);
        let origin = [
            (cell % cols) as f32 / cols as f32,
            (cell / cols) as f32 / rows as f32,
        ];
        let scale = [1.0 / cols as f32, 1.0 / rows as f32];
        let uvs: Vec<[f32; 2]> = leaves
            .uvs
            .iter()
            .map(|uv| [origin[0] + uv[0] * scale[0], origin[1] + uv[1] * scale[1]])
            .collect();
        emit(
            &mut out,
            "leaves",
            &leaves.positions,
            &uvs,
            &leaves.normals,
            &leaves.indices,
            mesh.positions.len() as u32,
        );
    }

    std::fs::write(path, out)
}

fn print_usage() {
    // Listed from the presets themselves, so adding a species does not leave the
    // usage line quietly out of date.
    let names: Vec<&str> = builtin_presets().into_iter().map(|(n, _)| n).collect();
    println!(
        "arbor-cli <{}|saved-preset|path/to/species.ron> [--seed N] [--no-leaves]",
        names.join("|")
    );
    println!(
        "          [--obj out.obj] [--glb out.glb] [--gltf out.gltf] \
         [--textures DIR | --no-textures] [--wind-data] [--variations N]"
    );
    println!("Grows a tree, builds bark and leaf meshes, prints stats.");
    println!("A saved preset is one saved from the viewer, found by name in {CUSTOM_PRESET_DIR}.");
    println!("--obj writes a triangle OBJ with separate `bark` and `leaves` groups.");
    println!("--glb writes one self-contained glTF binary, textures and all.");
    println!("--gltf writes glTF JSON, with its buffer and textures as files beside it.");
    println!("  Textures are read from {TEXTURE_DIR} unless --textures says otherwise;");
    println!("  --no-textures leaves them out. --wind-data adds each vertex's sway pivots");
    println!("  and weights as custom attributes, for driving wind in an engine.");
    println!("  --variations N writes N trees, from the seed and each one after it, each");
    println!("  named for its seed: out_seed7.glb, out_seed8.glb, and so on.");
}
