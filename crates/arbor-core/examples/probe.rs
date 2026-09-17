//! TEMP probe: triangle budget by species. Delete when done.
use arbor_core::cluster::{bake_cluster, Bitmap, LeafMaps};
use arbor_core::species::{builtin_presets, parse_species};
use arbor_core::{build_leaves, build_mesh, grow};

fn load(p: &str) -> Bitmap {
    let i = image::open(p).unwrap().to_rgba8();
    Bitmap { width: i.width(), height: i.height(), pixels: i.into_raw() }
}
fn cov(b: &Bitmap) -> f32 {
    b.pixels.chunks_exact(4).filter(|q| q[3] > 89).count() as f32 / (b.pixels.len() / 4) as f32
}

fn main() {
    println!(
        "{:<8} {:>7} {:>8} {:>8} {:>8} | {:>6} {:>5} {:>6} {:>5} {:>6} {:>6}",
        "species", "height", "bark", "leaf", "TOTAL", "cards", "card", "dens", "clus", "ovlap", "covidx"
    );
    for a in std::env::args().skip(1) {
        let t = match builtin_presets().into_iter().find(|(n, _)| *n == a) {
            Some((_, s)) => s.to_string(),
            None => std::fs::read_to_string(&a).unwrap(),
        };
        let p = parse_species(&t).unwrap();
        let lp = &p.leaves;
        let sk = grow(&p);
        let mesh = build_mesh(&sk, &p);
        let l = build_leaves(&sk, &p);
        let src = load(&format!("assets/textures/{}_albedo.png", lp.texture));
        let c = match &lp.cluster {
            Some(cp) => cov(&bake_cluster(cp, 1, 1, LeafMaps { albedo: &src, normal: None, roughness: None }).albedo),
            None => cov(&src),
        };
        let bark = mesh.triangle_count();
        let leaf = l.triangle_count();
        println!(
            "{:<8} {:>6.1}m {:>8} {:>8} {:>8} | {:>6} {:>5.2} {:>6.2} {:>4.0}% {:>6.1} {:>6.3}",
            a.rsplit('/').next().unwrap().trim_end_matches(".ron"),
            sk.stats().height, bark, leaf, bark + leaf,
            l.leaf_count(), lp.card_length, lp.density, c * 100.0,
            lp.card_length * lp.density,
            lp.density * lp.card_length * lp.card_length * c
        );
    }
}
