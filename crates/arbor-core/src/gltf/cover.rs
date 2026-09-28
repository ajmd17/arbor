//! Ground cover as glTF: one clump, its LODs, its baked blade atlas, and what an engine
//! needs to plant it and sway it.
//!
//! The scene is one node, named for the preset, carrying LOD0. The coarser LODs are
//! nodes outside the scene, named `<name>_LOD1` and on, and tied to it with the
//! standard `MSFT_lod` extension, with `MSFT_screencoverage` in its extras — so a tool
//! that knows nothing of LODs opens LOD0 alone, and one that does finds the rest.
//!
//! Every primitive carries two extensions:
//!
//! - `ARBOR_tree_wind`, as trees write it, with a card for a stem: one row in
//!   `branches` per card, its root the pivot and its length the reach, order 0, carried
//!   by nothing. `TEXCOORD_1` names each vertex's card and how far up it the vertex is,
//!   so an engine bends each card along its length with the same cantilever it bends a
//!   limb with. `leafOrigins` is each card's root, so flutter pivots at the ground.
//!   Shading normals are already bent toward the sky by `normalBlend`, and
//!   `backfaceVolume` is 1 so the back of a card keeps that lean rather than facing
//!   the ground.
//! - `ARBOR_ground_cover`: `footprint`, `height`, `rootBlend`, `groundNormalBlend` and
//!   `lods`, the screen size each LOD takes over at, LOD0 first.
//!
//! Metres, Y up, the clump's origin on the ground at the middle of its footprint.

use std::path::{Path, PathBuf};

use super::*;
use crate::blades::bake_blades;
use crate::cover::{build_cover, CoverMesh, CoverParams, CoverTemplate};

/// The extension carrying what an engine needs to plant and blend a clump.
pub const GROUND_COVER_EXTENSION: &str = "ARBOR_ground_cover";
/// The standard LOD extension, and the extras key it keeps its switch sizes under.
const LOD_EXTENSION: &str = "MSFT_lod";

/// A clump's atlas, baked and encoded once for any number of seeds.
pub struct CoverTextures {
    albedo: Vec<u8>,
    normal: Option<Vec<u8>>,
    roughness: Option<Vec<u8>>,
    /// Mean alpha of each atlas cell.
    pub coverage: Vec<f32>,
}

impl CoverTextures {
    pub fn bake(params: &CoverParams) -> Result<Self, String> {
        let baked = bake_blades(&params.atlas);
        Ok(Self {
            albedo: textures::encode_png(&baked.maps.albedo)?,
            normal: baked.maps.normal.as_ref().map(textures::encode_png).transpose()?,
            roughness: baked.maps.roughness.as_ref().map(|r| textures::encode_png(&leaf_roughness(r))).transpose()?,
            coverage: baked.coverage,
        })
    }
}

/// One clump as a `.glb`, in memory.
pub fn cover_glb(mesh: &CoverMesh, params: &CoverParams, textures: &CoverTextures) -> Vec<u8> {
    build_cover_doc(Format::Glb, "cover", "cover", mesh, params, textures).glb()
}

/// Writes one clump to `path`, as a `.glb` or a `.gltf` by its extension. `images`
/// names a `.gltf`'s image files, so a batch shares one set.
pub fn write_cover(
    path: &Path,
    images: Option<&str>,
    mesh: &CoverMesh,
    params: &CoverParams,
    textures: &CoverTextures,
) -> Result<ExportReport, String> {
    let format = Format::from_path(path).ok_or_else(|| format!("{}: export to a .glb or a .gltf", path.display()))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).filter(|s| !s.is_empty()).unwrap_or("cover").to_string();
    let dir = path.parent().unwrap_or(Path::new(""));
    if !dir.as_os_str().is_empty() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let doc = build_cover_doc(format, &stem, images.unwrap_or(&stem), mesh, params, textures);
    let mut report = ExportReport::default();
    let mut write = |path: PathBuf, bytes: &[u8]| -> Result<(), String> {
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        report.bytes += bytes.len() as u64;
        report.files.push(path);
        Ok(())
    };
    match format {
        Format::Glb => write(path.to_path_buf(), &doc.glb())?,
        Format::Gltf => {
            write(path.to_path_buf(), doc.json.as_bytes())?;
            write(dir.join(format!("{stem}.bin")), &doc.bin)?;
            for (name, bytes) in &doc.files {
                write(dir.join(name), bytes)?;
            }
        }
    }
    Ok(report)
}

/// Builds and writes `count` clumps of one preset into `dir`, from its seed and each one
/// after it, named as tree batches are. The atlas is baked once for all of them.
pub fn export_cover_batch(
    dir: &Path,
    stem: &str,
    format: Format,
    template: &CoverTemplate,
    count: u32,
    mut progress: impl FnMut(u32) -> bool,
) -> Result<BatchReport, String> {
    let count = count.max(1);
    let textures = CoverTextures::bake(&template.instance())?;
    let mut report = BatchReport::default();
    for i in 0..count {
        let mut at_seed = template.clone();
        at_seed.seed = template.seed.wrapping_add(u64::from(i));
        let params = at_seed.instance();
        let mesh = build_cover(&params);
        let path = dir.join(batch_file(stem, format, params.seed, count));
        let written = write_cover(&path, (count > 1).then_some(stem), &mesh, &params, &textures)?;
        report.bytes += written.bytes;
        report.trees.push(path);
        if !progress(i + 1) && i + 1 < count {
            report.stopped = true;
            break;
        }
    }
    Ok(report)
}

fn build_cover_doc(
    format: Format,
    stem: &str,
    images: &str,
    mesh: &CoverMesh,
    params: &CoverParams,
    textures: &CoverTextures,
) -> Document {
    let mut doc = Builder::new(format, stem, images);
    let material = cover_material(&mut doc, textures);

    // One row per card, shared by every LOD: a coarse LOD's cards are a subset of the
    // fine one's, and keep their rows.
    let mut table = vec![[0.0f32; 4]; 2 * (mesh.cards.len() + 1)];
    for (i, c) in mesh.cards.iter().enumerate() {
        let at = 2 * (i + 1);
        table[at] = [c.root[0], c.root[1], c.root[2], c.height];
    }
    let branches = doc.data(&flat(&table), 4);
    doc.uses(TREE_WIND_EXTENSION);
    doc.uses(GROUND_COVER_EXTENSION);

    let w = &params.wind;
    let blend = params.normals.ground_normal_blend.clamp(0.0, 1.0);
    let lods: Vec<f32> = mesh.lods.iter().map(|l| l.screen_size).collect();
    let cover_ext = format!(
        r#""{GROUND_COVER_EXTENSION}":{{"footprint":{},"height":{},"rootBlend":{},"groundNormalBlend":{},"lods":{}}}"#,
        num(params.footprint.size),
        num(mesh.height),
        num(params.normals.root_blend.max(0.0)),
        num(blend),
        floats(&lods)
    );

    let mut lod_nodes = Vec::new();
    for (level, lod) in mesh.lods.iter().enumerate() {
        if lod.positions.is_empty() {
            continue;
        }
        let n = lod.positions.len();
        let places: Vec<[f32; 2]> = lod.sway.iter().map(|&(card, t)| [(card + 1) as f32, t]).collect();
        let attrs = [
            ("POSITION", doc.floats(&flat(&lod.positions), 3, true)),
            ("NORMAL", doc.floats(&flat(&lod.normals), 3, false)),
            ("TANGENT", doc.floats(&flat(&lod.tangents), 4, false)),
            ("TEXCOORD_0", doc.floats(&flat(&lod.uvs), 2, false)),
            ("TEXCOORD_1", doc.floats(&flat(&places), 2, false)),
        ];
        let origins = doc.data(&flat(&lod.origins), 3);
        let wind_ext = format!(
            r#""{TREE_WIND_EXTENSION}":{{"branches":{branches},"leafOrigins":{origins},"normalBlend":{},"backfaceVolume":1,"flexibility":{},"frequency":{},"flutter":{},"height":{}}}"#,
            num(blend),
            floats(&w.flexibility),
            num(w.frequency),
            num(w.flutter),
            num(mesh.height)
        );
        let idx = doc.indices(&lod.indices, n);
        let primitive = format!(
            r#"{{"attributes":{},"indices":{idx},"material":{material},"extensions":{{{wind_ext},{cover_ext}}}}}"#,
            attributes(&attrs)
        );
        let name = format!("{}_LOD{level}", params.name);
        let m = doc.mesh(&name, &[primitive]);
        lod_nodes.push((name, m, lod.screen_size));
    }

    let root_name = json_str(&params.name);
    let mut fields = format!(r#""name":{root_name}"#);
    let mut extras = format!(r#""arbor":{{"cover":{root_name},"seed":{}}}"#, params.seed);
    if let Some(((_, lod0, _), coarser)) = lod_nodes.split_first() {
        let _ = write!(fields, r#","mesh":{lod0}"#);
        if !coarser.is_empty() {
            // The coarser levels go in outside the scene, for LOD0 to name.
            let ids: Vec<String> = coarser
                .iter()
                .map(|(name, m, _)| doc.node(&format!(r#""name":{}"#, json_str(name)), *m).to_string())
                .collect();
            doc.uses(LOD_EXTENSION);
            let _ = write!(fields, r#","extensions":{{"{LOD_EXTENSION}":{{"ids":[{}]}}}}"#, ids.join(","));
            // The smallest share of the screen each level is drawn at, LOD0 first; the
            // last is the size below which nothing is drawn.
            let mut coverage: Vec<f32> = coarser.iter().map(|l| l.2).collect();
            coverage.push(0.0);
            let _ = write!(extras, r#","MSFT_screencoverage":{}"#, floats(&coverage));
        }
    }
    let root = doc.nodes.len();
    doc.nodes.push(format!(r#"{{{fields},"extras":{{{extras}}}}}"#));
    doc.finish(format!(r#"{{"name":{root_name},"nodes":[{root}]}}"#))
}

/// Cut out at the leaf cards' alpha test and drawn from both sides, with the atlas's
/// normal and roughness maps.
fn cover_material(doc: &mut Builder, textures: &CoverTextures) -> usize {
    let clamp = doc.sampler(CLAMP_TO_EDGE);
    let albedo = doc.image("grass_albedo", &textures.albedo);
    let albedo = doc.texture(albedo, clamp);
    let normal = textures.normal.as_ref().map(|png| {
        let image = doc.image("grass_normal", png);
        doc.texture(image, clamp)
    });
    let (rough, factor) = match &textures.roughness {
        Some(png) => {
            let image = doc.image("grass_roughness", png);
            (Some(doc.texture(image, clamp)), 1.0)
        }
        None => (None, LEAF_ROUGHNESS_DEFAULT),
    };
    let mask = format!(r#","alphaMode":"MASK","alphaCutoff":{},"doubleSided":true"#, num(LEAF_ALPHA_CUTOFF));
    doc.material(&pbr_material("grass", [1.0, 1.0, 1.0, 1.0], Some(albedo), rough, factor, normal, &mask))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{chunks, get, int, seq, text, validate};
    use super::*;
    use crate::cover::{parse_cover_template, MEADOW_GRASS_RON};

    fn small_meadow() -> CoverParams {
        let mut p = parse_cover_template(MEADOW_GRASS_RON).unwrap().instance();
        p.atlas.cell_height = 64;
        p
    }

    #[test]
    fn a_clump_exports_as_a_valid_glb_with_every_lod() {
        let p = small_meadow();
        let mesh = build_cover(&p);
        let glb = cover_glb(&mesh, &p, &CoverTextures::bake(&p).unwrap());
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        assert_eq!(meshes.len(), mesh.lods.len());
        for (i, (name, _)) in meshes.iter().enumerate() {
            assert_eq!(name, &format!("meadow_grass_LOD{i}"));
        }
        // Only LOD0 is in the scene; the rest hang off it by MSFT_lod.
        let scene = &seq(get(&doc, "scenes"))[0];
        let roots = seq(get(scene, "nodes"));
        assert_eq!(roots.len(), 1);
        let root = &seq(get(&doc, "nodes"))[int(&roots[0])];
        let ids = seq(get(get(get(root, "extensions"), "MSFT_lod"), "ids"));
        assert_eq!(ids.len(), mesh.lods.len() - 1);
        let used: Vec<String> = seq(get(&doc, "extensionsUsed")).iter().map(text).collect();
        for e in [TREE_WIND_EXTENSION, GROUND_COVER_EXTENSION, "MSFT_lod"] {
            assert!(used.iter().any(|u| u == e), "{e} not declared");
        }
    }

    #[test]
    fn every_primitive_carries_the_cover_and_wind_extensions() {
        let p = small_meadow();
        let mesh = build_cover(&p);
        let glb = cover_glb(&mesh, &p, &CoverTextures::bake(&p).unwrap());
        let (doc, bin) = chunks(&glb);
        for (_, prims) in validate(&doc, bin) {
            for prim in prims {
                let ext = get(&prim, "extensions");
                let cover = get(ext, GROUND_COVER_EXTENSION);
                assert_eq!(seq(get(cover, "lods")).len(), mesh.lods.len());
                let wind = get(ext, TREE_WIND_EXTENSION);
                let branches = &seq(get(&doc, "accessors"))[int(get(wind, "branches"))];
                assert_eq!(int(get(branches, "count")), 2 * (mesh.cards.len() + 1));
                assert!(int(get(wind, "leafOrigins")) < seq(get(&doc, "accessors")).len());
            }
        }
        let mat = &seq(get(&doc, "materials"))[0];
        assert_eq!(text(get(mat, "alphaMode")), "MASK");
        assert!(matches!(get(mat, "normalTexture"), ron::Value::Map(_)));
    }
}
