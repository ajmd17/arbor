//! Rocks as glTF: one rock, its LODs and its baked maps.
//!
//! The scene is one node, named for the preset, carrying LOD0; the coarser LODs hang off
//! it by `MSFT_lod`, as a clump of ground cover's do. The material is plain metallic
//! roughness, with occlusion packed into the red of the roughness map as glTF's ORM
//! convention has it, so one image serves both slots.
//!
//! Every primitive carries `ARBOR_rock`: `bury`, the depth below the origin the rock was
//! built sunk to, `height` above it, and `lods`, the screen size each LOD takes over at.
//!
//! Metres, Y up, the origin on the ground under the rock.

use std::path::{Path, PathBuf};

use super::*;
use crate::rocks::{bake_rock, build_rock, RockMaps, RockMesh, RockParams, RockTemplate};

/// The extension carrying what an engine needs to place a rock.
pub const ROCK_EXTENSION: &str = "ARBOR_rock";
const LOD_EXTENSION: &str = "MSFT_lod";

/// A rock's maps, encoded.
pub struct RockTextures {
    albedo: Vec<u8>,
    normal: Vec<u8>,
    orm: Vec<u8>,
}

impl RockTextures {
    pub fn bake(params: &RockParams, mesh: &RockMesh) -> Result<Self, String> {
        Self::encode(&bake_rock(params, mesh))
    }

    pub fn encode(maps: &RockMaps) -> Result<Self, String> {
        Ok(Self {
            albedo: textures::encode_png(&maps.albedo)?,
            normal: textures::encode_png(&maps.normal)?,
            orm: textures::encode_png(&maps.orm)?,
        })
    }
}

/// One rock as a `.glb`, in memory.
pub fn rock_glb(mesh: &RockMesh, params: &RockParams, textures: &RockTextures) -> Vec<u8> {
    build_rock_doc(Format::Glb, "rock", "rock", mesh, params, textures).glb()
}

/// Writes one rock to `path`, as a `.glb` or a `.gltf` by its extension.
pub fn write_rock(path: &Path, mesh: &RockMesh, params: &RockParams, textures: &RockTextures) -> Result<ExportReport, String> {
    let format = Format::from_path(path).ok_or_else(|| format!("{}: export to a .glb or a .gltf", path.display()))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).filter(|s| !s.is_empty()).unwrap_or("rock").to_string();
    let dir = path.parent().unwrap_or(Path::new(""));
    if !dir.as_os_str().is_empty() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let doc = build_rock_doc(format, &stem, &stem, mesh, params, textures);
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

/// Builds and writes `count` rocks of one preset into `dir`, from its seed and each one
/// after it, named as tree batches are. Every rock has maps of its own, since every seed
/// is its own stone.
pub fn export_rock_batch(
    dir: &Path,
    stem: &str,
    format: Format,
    template: &RockTemplate,
    count: u32,
    mut progress: impl FnMut(u32) -> bool,
) -> Result<BatchReport, String> {
    let count = count.max(1);
    let mut report = BatchReport::default();
    for i in 0..count {
        let mut at_seed = template.clone();
        at_seed.seed = template.seed.wrapping_add(u64::from(i));
        let params = at_seed.instance();
        let mesh = build_rock(&params);
        let textures = RockTextures::bake(&params, &mesh)?;
        let path = dir.join(batch_file(stem, format, params.seed, count));
        let written = write_rock(&path, &mesh, &params, &textures)?;
        report.bytes += written.bytes;
        report.trees.push(path);
        if !progress(i + 1) && i + 1 < count {
            report.stopped = true;
            break;
        }
    }
    Ok(report)
}

fn build_rock_doc(format: Format, stem: &str, images: &str, mesh: &RockMesh, params: &RockParams, textures: &RockTextures) -> Document {
    let mut doc = Builder::new(format, stem, images);
    let clamp = doc.sampler(CLAMP_TO_EDGE);
    let albedo = doc.image("rock_albedo", &textures.albedo);
    let albedo = doc.texture(albedo, clamp);
    let normal = doc.image("rock_normal", &textures.normal);
    let normal = doc.texture(normal, clamp);
    let orm = doc.image("rock_orm", &textures.orm);
    let orm = doc.texture(orm, clamp);
    let occlusion = format!(r#","occlusionTexture":{{"index":{orm}}}"#);
    let material = doc.material(&pbr_material(&params.name, [1.0, 1.0, 1.0, 1.0], Some(albedo), Some(orm), 1.0, Some(normal), &occlusion));
    doc.uses(ROCK_EXTENSION);

    let lods: Vec<f32> = mesh.lods.iter().map(|l| l.screen_size).collect();
    let rock_ext = format!(
        r#""{ROCK_EXTENSION}":{{"bury":{},"height":{},"lods":{}}}"#,
        num(-mesh.min[1].min(0.0)),
        num(mesh.max[1]),
        floats(&lods)
    );
    let mut lod_nodes = Vec::new();
    for (level, lod) in mesh.lods.iter().enumerate() {
        if lod.positions.is_empty() {
            continue;
        }
        let n = lod.positions.len();
        let attrs = [
            ("POSITION", doc.floats(&flat(&lod.positions), 3, true)),
            ("NORMAL", doc.floats(&flat(&lod.normals), 3, false)),
            ("TANGENT", doc.floats(&flat(&lod.tangents), 4, false)),
            ("TEXCOORD_0", doc.floats(&flat(&lod.uvs), 2, false)),
        ];
        let idx = doc.indices(&lod.indices, n);
        let primitive = format!(
            r#"{{"attributes":{},"indices":{idx},"material":{material},"extensions":{{{rock_ext}}}}}"#,
            attributes(&attrs)
        );
        let name = format!("{}_LOD{level}", params.name);
        let m = doc.mesh(&name, &[primitive]);
        lod_nodes.push((name, m, lod.screen_size));
    }

    let root_name = json_str(&params.name);
    let mut fields = format!(r#""name":{root_name}"#);
    let mut extras = format!(r#""arbor":{{"rock":{root_name},"seed":{}}}"#, params.seed);
    if let Some(((_, lod0, _), coarser)) = lod_nodes.split_first() {
        let _ = write!(fields, r#","mesh":{lod0}"#);
        if !coarser.is_empty() {
            let ids: Vec<String> = coarser
                .iter()
                .map(|(name, m, _)| doc.node(&format!(r#""name":{}"#, json_str(name)), *m).to_string())
                .collect();
            doc.uses(LOD_EXTENSION);
            let _ = write!(fields, r#","extensions":{{"{LOD_EXTENSION}":{{"ids":[{}]}}}}"#, ids.join(","));
            let mut coverage: Vec<f32> = coarser.iter().map(|l| l.2).collect();
            coverage.push(0.0);
            let _ = write!(extras, r#","MSFT_screencoverage":{}"#, floats(&coverage));
        }
    }
    let root = doc.nodes.len();
    doc.nodes.push(format!(r#"{{{fields},"extras":{{{extras}}}}}"#));
    doc.finish(format!(r#"{{"name":{root_name},"nodes":[{root}]}}"#))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{chunks, get, int, seq, text, validate};
    use super::*;
    use crate::rocks::{parse_rock_template, BOULDER_RON};

    #[test]
    fn a_rock_exports_as_a_valid_glb_with_every_lod() {
        let mut p = parse_rock_template(BOULDER_RON).unwrap().instance();
        p.texture.size = 64;
        let mesh = build_rock(&p);
        let glb = rock_glb(&mesh, &p, &RockTextures::bake(&p, &mesh).unwrap());
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        assert_eq!(meshes.len(), mesh.lods.len());
        for (i, (name, prims)) in meshes.iter().enumerate() {
            assert_eq!(name, &format!("boulder_LOD{i}"));
            let ext = get(get(&prims[0], "extensions"), ROCK_EXTENSION);
            assert_eq!(seq(get(ext, "lods")).len(), mesh.lods.len());
        }
        let scene = &seq(get(&doc, "scenes"))[0];
        assert_eq!(seq(get(scene, "nodes")).len(), 1);
        let mat = &seq(get(&doc, "materials"))[0];
        let orm = int(get(get(get(mat, "pbrMetallicRoughness"), "metallicRoughnessTexture"), "index"));
        assert_eq!(int(get(get(mat, "occlusionTexture"), "index")), orm, "occlusion and roughness share one image");
        let used: Vec<String> = seq(get(&doc, "extensionsUsed")).iter().map(text).collect();
        assert!(used.iter().any(|u| u == ROCK_EXTENSION));
    }
}
