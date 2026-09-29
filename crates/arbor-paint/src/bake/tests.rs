use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};

use super::*;
use crate::import::{Image, ImportedModel, Instance, Material, Primitive, Slot};

/// A quad from four corners and their UVs (none for a mesh that cannot be baked), with the
/// one normal it is given. Splits into two triangles.
fn quad(corners: [Vec3; 4], uvs: Option<[Vec2; 4]>, normal: Vec3, material: usize) -> Primitive {
    Primitive {
        name: "quad".into(),
        positions: corners.map(|c| c.to_array()).to_vec(),
        normals: vec![normal.to_array(); 4],
        uvs: uvs.map(|u| u.map(|v| v.to_array()).to_vec()),
        tangents: None,
        colors: None,
        indices: vec![0, 1, 2, 0, 2, 3],
        material,
    }
}

fn model(prims: Vec<Primitive>, materials: Vec<Material>, images: Vec<Option<Image>>) -> Arc<ImportedModel> {
    let instances = (0..prims.len())
        .map(|i| Instance { name: format!("p{i}"), primitive: i, transform: Mat4::IDENTITY })
        .collect();
    Arc::new(ImportedModel { name: "test".into(), primitives: prims, materials, images, instances, warnings: vec![] })
}

fn scene(m: Arc<ImportedModel>) -> Scene {
    let visible = vec![true; m.instances.len()];
    Scene::build(m, &visible).unwrap()
}

fn params(size: u32, maps: Vec<MapKind>) -> BakeParams {
    BakeParams { width: size, height: size, padding: 0, maps, ao_rays: 64, ..Default::default() }
}

fn run(scene: &Scene, p: &BakeParams) -> BakeResult {
    bake(scene, 0, p, &Progress::default()).unwrap()
}

fn map_of(r: &BakeResult, kind: MapKind) -> &BakedMap {
    r.maps.iter().find(|m| m.kind == kind).unwrap()
}

fn at(m: &BakedMap, x: usize, y: usize) -> [f32; 4] {
    m.pixels[y * m.width as usize + x]
}

const V: fn(f32, f32, f32) -> Vec3 = Vec3::new;

/// The unit square in x and z, facing up, with u along x and v along z.
fn floor(material: usize, with_uvs: bool) -> Primitive {
    quad(
        [V(0.0, 0.0, 0.0), V(0.0, 0.0, 1.0), V(1.0, 0.0, 1.0), V(1.0, 0.0, 0.0)],
        with_uvs.then_some([Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0)]),
        Vec3::Y,
        material,
    )
}

/// The same square lifted to `height`, facing down, with no UVs so it is only an occluder.
fn ceiling(height: f32, material: usize) -> Primitive {
    let mut q = floor(material, false);
    q.positions.iter_mut().for_each(|p| p[1] = height);
    q.normals = vec![[0.0, -1.0, 0.0]; 4];
    q
}

fn plain() -> Vec<Material> {
    vec![Material::plain("floor"), Material::plain("cover")]
}

#[test]
fn a_flat_quad_bakes_its_normal_and_position() {
    let s = scene(model(vec![floor(0, true)], plain(), vec![]));
    let r = run(&s, &params(16, vec![MapKind::NormalWorld, MapKind::Position]));
    assert!(r.coverage > 0.99, "{}", r.coverage);
    let n = at(map_of(&r, MapKind::NormalWorld), 8, 8);
    assert!((n[0] - 0.5).abs() < 1e-4 && (n[1] - 1.0).abs() < 1e-4 && (n[2] - 0.5).abs() < 1e-4, "{n:?}");
    // Texel (12, 4) is at u = 12.5/16 and v = 4.5/16, which are x and z.
    let p = at(map_of(&r, MapKind::Position), 12, 4);
    assert!((p[0] - 12.5 / 16.0).abs() < 1e-3 && (p[2] - 4.5 / 16.0).abs() < 1e-3, "{p:?}");
}

#[test]
fn an_open_surface_is_unoccluded_and_a_covered_one_is_dark() {
    let open = scene(model(vec![floor(0, true)], plain(), vec![]));
    let r = run(&open, &params(8, vec![MapKind::AmbientOcclusion]));
    assert!(at(map_of(&r, MapKind::AmbientOcclusion), 4, 4)[0] > 0.999);

    // A lid 5 cm over a metre-wide floor blocks all but the flattest rays.
    let covered = scene(model(vec![floor(0, true), ceiling(0.05, 1)], plain(), vec![]));
    let r = run(&covered, &params(8, vec![MapKind::AmbientOcclusion]));
    let ao = at(map_of(&r, MapKind::AmbientOcclusion), 4, 4)[0];
    assert!(ao < 0.1, "{ao}");
}

#[test]
fn a_cutout_lets_rays_through_its_holes() {
    let mut cover = Material::plain("cover");
    cover.mask = true;
    cover.textures = vec![(Slot::BaseColor, 0)];
    let mut lid = ceiling(0.05, 1);
    lid.uvs = Some(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
    let bake_with = |alpha: u8| {
        let image = Image { width: 1, height: 1, rgba: vec![255, 255, 255, alpha] };
        let m = model(vec![floor(0, true), lid.clone_prim()], vec![Material::plain("floor"), cover_clone(&cover)], vec![Some(image)]);
        let r = run(&scene(m), &params(8, vec![MapKind::AmbientOcclusion]));
        at(map_of(&r, MapKind::AmbientOcclusion), 4, 4)[0]
    };
    assert!(bake_with(255) < 0.1, "an opaque lid shades");
    assert!(bake_with(0) > 0.999, "a fully transparent one does not");
}

fn cover_clone(m: &Material) -> Material {
    Material { textures: m.textures.clone(), name: m.name.clone(), ..Material::plain("cover") }
        .with_mask()
}

impl Material {
    fn with_mask(mut self) -> Self {
        self.mask = true;
        self
    }
}

impl Primitive {
    fn clone_prim(&self) -> Primitive {
        Primitive {
            name: self.name.clone(),
            positions: self.positions.clone(),
            normals: self.normals.clone(),
            uvs: self.uvs.clone(),
            tangents: self.tangents.clone(),
            colors: self.colors.clone(),
            indices: self.indices.clone(),
            material: self.material,
        }
    }
}

#[test]
fn a_thicker_slab_bakes_whiter() {
    let thickness = |depth: f32| {
        let mut bottom = ceiling(-depth, 1);
        bottom.normals = vec![[0.0, -1.0, 0.0]; 4];
        let s = scene(model(vec![floor(0, true), bottom], plain(), vec![]));
        let r = run(&s, &params(8, vec![MapKind::Thickness]));
        at(map_of(&r, MapKind::Thickness), 4, 4)[0]
    };
    let (thin, thick) = (thickness(0.02), thickness(0.15));
    assert!(thin > 0.0 && thin < thick && thick < 1.0, "{thin} {thick}");
}

/// Two quads meeting along z = the ridge at x = 0. The right one falls away by `drop`
/// (a ridge is convex when it is positive, a valley concave when it is negative). u runs
/// 0 to 1 across both, so the fold is at the middle of the texture.
fn fold(drop: f32) -> Arc<ImportedModel> {
    let left = quad(
        [V(-1.0, 0.0, 0.0), V(-1.0, 0.0, 1.0), V(0.0, 0.0, 1.0), V(0.0, 0.0, 0.0)],
        Some([Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(0.5, 1.0), Vec2::new(0.5, 0.0)]),
        Vec3::Y,
        0,
    );
    let slope = V(1.0, -drop, 0.0);
    let right = quad(
        [V(0.0, 0.0, 0.0), V(0.0, 0.0, 1.0), V(1.0, -drop, 1.0), V(1.0, -drop, 0.0)],
        Some([Vec2::new(0.5, 0.0), Vec2::new(0.5, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0)]),
        V(-slope.y, slope.x, 0.0).normalize(),
        0,
    );
    model(vec![left, right], plain(), vec![])
}

#[test]
fn curvature_is_flat_on_planes_and_signed_at_folds() {
    let bake_curvature = |m: Arc<ImportedModel>| {
        // Two texels either side: 0.0255 of the fold model's 2.45 diagonal.
        let p = BakeParams { curvature_radius: 0.0255, ..params(64, vec![MapKind::Curvature]) };
        run(&scene(m), &p)
    };
    let flat = bake_curvature(model(vec![floor(0, true)], plain(), vec![]));
    let c = map_of(&flat, MapKind::Curvature);
    assert!((at(c, 20, 20)[0] - 0.5).abs() < 1e-4);

    let ridge = bake_curvature(fold(1.0));
    let c = map_of(&ridge, MapKind::Curvature);
    assert!(at(c, 32, 30)[0] > 0.55, "convex at the ridge: {}", at(c, 32, 30)[0]);
    assert!((at(c, 6, 30)[0] - 0.5).abs() < 1e-3, "flat away from it");

    let valley = bake_curvature(fold(-1.0));
    let c = map_of(&valley, MapKind::Curvature);
    assert!(at(c, 32, 30)[0] < 0.45, "concave in the valley: {}", at(c, 32, 30)[0]);
}

#[test]
fn padding_grows_the_islands_and_supersampling_keeps_them_in_place() {
    // Half the texture is covered.
    let half = quad(
        [V(0.0, 0.0, 0.0), V(0.0, 0.0, 1.0), V(1.0, 0.0, 1.0), V(1.0, 0.0, 0.0)],
        Some([Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(0.5, 1.0), Vec2::new(0.5, 0.0)]),
        Vec3::Y,
        0,
    );
    let s = scene(model(vec![half], plain(), vec![]));
    let mut p = params(16, vec![MapKind::NormalWorld]);
    p.padding = 3;
    let r = run(&s, &p);
    let n = map_of(&r, MapKind::NormalWorld);
    // Three texels past the island's edge (x = 8) carry its normal; the rest is background.
    assert!((at(n, 10, 8)[1] - 1.0).abs() < 1e-4);
    assert!((at(n, 14, 8)[1] - 0.5).abs() < 1e-4);

    p.supersample = 2;
    p.padding = 0;
    let r = run(&s, &p);
    let n = map_of(&r, MapKind::NormalWorld);
    assert!((at(n, 3, 8)[1] - 1.0).abs() < 1e-4 && (at(n, 12, 8)[1] - 0.5).abs() < 1e-4);
}

#[test]
fn bad_requests_are_errors_and_a_cancelled_bake_stops() {
    let s = scene(model(vec![floor(1, true)], plain(), vec![]));
    // Material 0 has no triangles.
    assert!(bake(&s, 0, &params(8, vec![MapKind::NormalWorld]), &Progress::default()).is_err());
    assert!(bake(&s, 1, &params(8, vec![]), &Progress::default()).is_err());
    let progress = Progress::default();
    progress.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(bake(&s, 1, &params(8, vec![MapKind::AmbientOcclusion]), &progress).is_err());
}

#[test]
fn overlapping_islands_are_reported() {
    let s = scene(model(vec![floor(0, true), floor(0, true)], plain(), vec![]));
    let r = run(&s, &params(8, vec![MapKind::NormalWorld]));
    assert!(r.warnings.iter().any(|w| w.contains("100% of the texels")), "{:?}", r.warnings);
}

#[test]
fn maps_export_as_eight_and_sixteen_bit_pngs() {
    let s = scene(model(vec![floor(0, true)], plain(), vec![]));
    let r = run(&s, &params(8, vec![MapKind::Position]));
    let map = map_of(&r, MapKind::Position);
    let dir = std::env::temp_dir().join("arbor-paint-export-test");
    std::fs::create_dir_all(&dir).unwrap();
    for (sixteen, name) in [(false, "p8.png"), (true, "p16.png")] {
        let path = dir.join(name);
        map.save_png(&path, sixteen).unwrap();
        let img = image::open(&path).unwrap();
        assert_eq!((img.width(), img.height()), (8, 8));
        assert_eq!(img.color().bits_per_pixel(), if sixteen { 64 } else { 32 });
    }
}

/// Times a bake of a real file, for tuning: `ARBOR_BENCH=model.glb cargo test --release
/// bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_real_model() {
    let Ok(path) = std::env::var("ARBOR_BENCH") else { return };
    let m = Arc::new(crate::import::load(std::path::Path::new(&path)).unwrap());
    let t = std::time::Instant::now();
    let visible = vec![true; m.instances.len()];
    let s = Scene::build(m.clone(), &visible).unwrap();
    println!("scene: {} triangles, built in {:?}", s.tris.len(), t.elapsed());
    let size = std::env::var("ARBOR_BENCH_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let rays = std::env::var("ARBOR_BENCH_RAYS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    for (i, mat) in m.materials.iter().enumerate().take(1) {
        for kind in [MapKind::NormalWorld, MapKind::AmbientOcclusion, MapKind::Thickness, MapKind::Curvature] {
            let p = BakeParams { width: size, height: size, ao_rays: rays, maps: vec![kind], ..Default::default() };
            let t = std::time::Instant::now();
            match bake(&s, i, &p, &Progress::default()) {
                Ok(r) => println!("{} / {kind:?}: {:?} (coverage {:.0}%)", mat.name, t.elapsed(), r.coverage * 100.0),
                Err(e) => println!("{} / {kind:?}: {e}", mat.name),
            }
        }
    }
}
