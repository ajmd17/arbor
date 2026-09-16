use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_mesh, grow, Mesh};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        std::process::exit(2);
    }

    let mut species_src: Option<String> = None;
    let mut seed_override: Option<u64> = None;
    let mut obj_out: Option<String> = None;

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
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => species_src = Some(other.to_string()),
        }
        i += 1;
    }

    let src = species_src.unwrap_or_else(|| "pine".to_string());
    let ron_text = match builtin_presets().into_iter().find(|(n, _)| *n == src) {
        Some((_, text)) => text.to_string(),
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

    let t = std::time::Instant::now();
    let skeleton = grow(&params);
    let mesh = build_mesh(&skeleton, &params);
    let dt = t.elapsed();
    let stats = skeleton.stats();

    println!("species: {}", params.name);
    println!("seed:    {}", params.seed);
    println!("{stats:#?}");
    println!("verts:   {}", mesh.vertex_count());
    println!("tris:    {}", mesh.triangle_count());
    println!("gen:     {:.3} ms", dt.as_secs_f64() * 1000.0);

    if let Some(path) = obj_out {
        write_obj(&mesh, &path).unwrap_or_else(|e| {
            eprintln!("obj write failed: {e}");
            std::process::exit(1);
        });
        println!("obj:     {path}");
    }
}

fn write_obj(mesh: &Mesh, path: &str) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::from("# arbor tree mesh\n");
    for p in &mesh.positions {
        let _ = writeln!(out, "v {} {} {}", p[0], p[1], p[2]);
    }
    for uv in &mesh.uvs {
        let _ = writeln!(out, "vt {} {}", uv[0], uv[1]);
    }
    for n in &mesh.normals {
        let _ = writeln!(out, "vn {} {} {}", n[0], n[1], n[2]);
    }
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize + 1, tri[1] as usize + 1, tri[2] as usize + 1);
        let _ = writeln!(out, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
    }
    std::fs::write(path, out)
}

fn print_usage() {
    println!("arbor-cli <pine|oak|path/to/species.ron> [--seed N] [--obj out.obj]");
    println!("Grows a tree, builds the mesh, prints stats. Writes triangle OBJ with --obj.");
}
