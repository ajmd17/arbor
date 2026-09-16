//! Prints alpha coverage down a plain mip chain versus a coverage-preserving one, for
//! whatever leaf texture is passed in. Coverage is the share of texels that survive the
//! alpha test, so it tracks directly with how much of the canopy stays on screen.
//!
//! cargo run --release -p arbor-viewer --example alpha_coverage_report -- assets/textures/leaf_oak_albedo.png [cutoff]

#[path = "../src/mipmap.rs"]
mod mipmap;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/textures/leaf_oak_albedo.png".into());
    let cutoff: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.35);
    let img = image::open(&path).expect("open texture").to_rgba8();
    let (w, h) = (img.width(), img.height());
    let src = img.into_raw();

    let fixed = mipmap::coverage_preserving_chain(&src, w, h, cutoff);
    let mut plain: Vec<(Vec<u8>, u32, u32)> = vec![(src.clone(), w, h)];
    while plain.last().map(|(_, w, h)| *w > 1 || *h > 1) == Some(true) {
        let (s, sw, sh) = plain.last().unwrap().clone();
        plain.push(mipmap::halve(&s, sw, sh));
    }

    let base = mipmap::alpha_coverage(&src, cutoff, 1.0);
    println!("{path}  {w}x{h}  cutoff {cutoff}");
    println!("level    size   plain     fixed    (level 0 = {base:.4})");
    for (i, (px, lw, lh)) in plain.iter().enumerate() {
        let p = mipmap::alpha_coverage(px, cutoff, 1.0);
        let f = mipmap::alpha_coverage(&fixed[i].0, cutoff, 1.0);
        let stats = |px: &[u8]| {
            let a: Vec<f32> = px.chunks_exact(4).map(|q| q[3] as f32 / 255.0).collect();
            let max = a.iter().cloned().fold(0.0f32, f32::max);
            let mean = a.iter().sum::<f32>() / a.len() as f32;
            (mean, max)
        };
        let (pm, px_max) = stats(px);
        let (fm, fx_max) = stats(&fixed[i].0);
        println!(
            "{i:>3}  {lw:>5}x{lh:<5} {p:.4}  {f:.4}   keeps {:>5.1}%   plain a(mean/max) {pm:.3}/{px_max:.3}   fixed {fm:.3}/{fx_max:.3}",
            100.0 * p / base.max(1e-6)
        );
    }
}
