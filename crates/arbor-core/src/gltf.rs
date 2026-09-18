//! glTF 2.0 export: a grown tree as a scene any engine or DCC tool can open, either as
//! one self-contained `.glb` or as a `.gltf` beside its buffer and textures.
//!
//! The scene is one node named for the species, carrying a `bark` node and a `leaves`
//! node. Materials are the standard metallic-roughness model, set up to match what the
//! viewer draws as closely as that model allows:
//!
//! - Bark carries its species tint as the base colour factor, and the darkening toward
//!   the foot of the trunk as a vertex colour. Dead wood is its own material, with the
//!   viewer's bleaching baked into a copy of the bark albedo.
//! - Leaves are alpha-masked at the viewer's cutoff, with each card's tint and its
//!   crown-depth shade as a vertex colour. A species whose cards show a different cell
//!   from behind gets a second, back-facing copy of every card reading that cell,
//!   since glTF has no way to pick a texture by which side is seen; one whose cells are
//!   the same gets a single double-sided copy instead.
//!
//! What does not survive: moss (drawn from noise in the viewer's shader), light
//! through the leaves, and the coverage-preserving mip chain that keeps a distant
//! canopy from thinning — an engine builds its own mips.
//!
//! With `wind` set, each vertex also carries the wood it hangs off, as the custom
//! attributes `_WIND_1` to `_WIND_3` (one per branch order: the pivot that order bends
//! about in xyz, and how far a point there swings for a unit of flexibility in w, in
//! metres), and every leaf vertex `_LEAF_ORIGIN`, the twig point its card hangs from.
//! That is what the viewer's wind shader reads, so an engine can sway the tree the same
//! way. The species' wind settings go in the root node's `extras`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use glam::Vec3;

use crate::cluster::Bitmap;
use crate::leaves::LeafMesh;
use crate::math::ortho_of;
use crate::mesh::Mesh;
use crate::species::SpeciesParams;
use crate::textures;

/// The viewer's alpha test on leaf cards.
pub const LEAF_ALPHA_CUTOFF: f32 = 0.35;
/// Roughest the viewer lets a leaf look glossy: below this, with nothing occluding the
/// sky, a canopy washes out to the colour of the sky behind it.
const LEAF_ROUGHNESS_FLOOR: f32 = 0.55;
/// A leaf with no roughness map is drawn at this.
const LEAF_ROUGHNESS_DEFAULT: f32 = 200.0 / 255.0;
/// Bark with no texture at all, in linear RGB: a plain mid brown.
const BARK_FALLBACK: [f32; 3] = [0.155, 0.091, 0.051];
/// How quickly the darkening at the foot of the trunk dies away with height, per metre.
/// The same number the viewer's bark shader uses.
const DARKEN_FALLOFF: f32 = 0.35;

const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_SHORT: u32 = 5123;
const UNSIGNED_INT: u32 = 5125;
const LINEAR: u32 = 9729;
const LINEAR_MIPMAP_LINEAR: u32 = 9987;
const REPEAT: u32 = 10497;
const CLAMP_TO_EDGE: u32 = 33071;

/// What goes into an export besides the geometry.
#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    /// Where the texture maps are read from. `None` writes the materials without any.
    pub textures: Option<PathBuf>,
    /// Also write each vertex's wind data as custom attributes.
    pub wind: bool,
}

/// The two ways a scene is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// One binary file with the buffer and the textures inside it.
    Glb,
    /// JSON, with the buffer and each texture as files beside it.
    Gltf,
}

impl Format {
    /// The format a path's extension asks for.
    pub fn from_path(path: &Path) -> Option<Format> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "glb" => Some(Format::Glb),
            "gltf" => Some(Format::Gltf),
            _ => None,
        }
    }
}

/// What an export wrote, and anything it had to leave out.
#[derive(Clone, Debug, Default)]
pub struct ExportReport {
    pub files: Vec<PathBuf>,
    pub bytes: u64,
    pub warnings: Vec<String>,
}

/// Writes the tree to `path`, as a `.glb` or a `.gltf` by its extension.
pub fn export(
    path: &Path,
    mesh: &Mesh,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    options: &ExportOptions,
) -> Result<ExportReport, String> {
    let format = Format::from_path(path)
        .ok_or_else(|| format!("{}: export to a .glb or a .gltf", path.display()))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("tree")
        .to_string();
    let dir = path.parent().unwrap_or(Path::new(""));
    if !dir.as_os_str().is_empty() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let (doc, warnings) = build(format, &stem, mesh, leaves, params, options)?;
    let mut report = ExportReport {
        warnings,
        ..Default::default()
    };
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

/// The tree as one `.glb`, in memory.
pub fn to_glb(
    mesh: &Mesh,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    Ok(build(Format::Glb, "tree", mesh, leaves, params, options)?.0.glb())
}

/// A finished document: the JSON, the binary buffer it indexes into, and, for a
/// `.gltf`, the image files it names.
struct Document {
    json: String,
    bin: Vec<u8>,
    files: Vec<(String, Vec<u8>)>,
}

impl Document {
    /// The binary container: a header, the JSON chunk padded with spaces, and the
    /// buffer chunk padded with zeros, every chunk on a four-byte boundary.
    fn glb(&self) -> Vec<u8> {
        let mut json = self.json.clone().into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bin = self.bin.clone();
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + if bin.is_empty() { 0 } else { 8 + bin.len() };
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json);
        if !bin.is_empty() {
            out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
            out.extend_from_slice(b"BIN\0");
            out.extend_from_slice(&bin);
        }
        out
    }
}

fn build(
    format: Format,
    stem: &str,
    mesh: &Mesh,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    options: &ExportOptions,
) -> Result<(Document, Vec<String>), String> {
    let mut doc = Builder::new(format, stem);
    let mut warnings = Vec::new();
    let texture_dir = options.textures.as_deref();

    let mut children = Vec::new();
    if !mesh.positions.is_empty() {
        let dead = mesh.weathering.iter().any(|&w| w > 0.5);
        let materials = bark_materials(&mut doc, params, texture_dir, dead, &mut warnings)?;
        let m = bark_mesh(&mut doc, mesh, params, materials, options.wind);
        children.push(doc.node(r#""name":"bark""#, m));
    }
    if !leaves.is_empty() {
        let lp = &params.leaves;
        let (cols, rows) = (lp.atlas_cols.max(1), lp.atlas_rows.max(1));
        let last = cols * rows - 1;
        let two_faced = lp.atlas_front.min(last) != lp.atlas_back.min(last);
        let material = leaf_material(&mut doc, params, texture_dir, two_faced, &mut warnings)?;
        let m = leaf_mesh(&mut doc, leaves, params, material, two_faced, options.wind);
        children.push(doc.node(r#""name":"leaves""#, m));
    }

    let mut extras = format!(
        r#""arbor":{{"species":{},"seed":{}"#,
        json_str(&params.name),
        params.seed
    );
    if options.wind {
        let w = &params.wind;
        let _ = write!(
            extras,
            r#","wind":{{"flexibility":{},"frequency":{},"flutter":{},"attributes":{{"_WIND_1":"limbs","_WIND_2":"branches","_WIND_3":"twigs and finer","xyz":"pivot the order bends about","w":"metres a point there swings per unit of flexibility"}}}}"#,
            floats(&w.flexibility),
            num(w.frequency),
            num(w.flutter)
        );
    }
    extras.push('}');
    let child_list = children.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    let root = doc.nodes.len();
    doc.nodes.push(format!(
        r#"{{"name":{},"children":[{child_list}],"extras":{{{extras}}}}}"#,
        json_str(&params.name)
    ));
    let scene = format!(r#"{{"name":{},"nodes":[{root}]}}"#, json_str(&params.name));
    Ok((doc.finish(scene), warnings))
}

/// The living bark's material and, when the tree has dead wood the species bleaches,
/// the dead wood's.
fn bark_materials(
    doc: &mut Builder,
    params: &SpeciesParams,
    texture_dir: Option<&Path>,
    has_dead: bool,
    warnings: &mut Vec<String>,
) -> Result<(usize, Option<usize>), String> {
    let mp = &params.mesh;
    let tint = mp.bark_tint;
    let bleach = mp.dead_wood_weathering.clamp(0.0, 1.0);
    let wants_dead = has_dead && bleach > 0.0;
    let name = &mp.bark_texture;

    // The maps go in as they are on disk, byte for byte, where nothing needs changing.
    let read = |map: &str| {
        texture_dir.and_then(|dir| std::fs::read(textures::map_path(dir, name, map)).ok())
    };
    let (albedo, normal, roughness) = (read("albedo"), read("normal"), read("roughness"));
    if texture_dir.is_some() && albedo.is_none() {
        warnings.push(format!("no {name}_albedo.png: the bark is exported untextured"));
    }
    let repeat = doc.sampler(REPEAT);
    let texture = |doc: &mut Builder, label: &str, png: Vec<u8>| {
        let image = doc.image(label, png);
        doc.texture(image, repeat)
    };
    let albedo_tex = albedo.map(|png| texture(doc, "bark_albedo", png));
    let normal_tex = normal.map(|png| texture(doc, "bark_normal", png));
    let rough_tex = roughness.map(|png| texture(doc, "bark_roughness", png));

    let living_color = match albedo_tex {
        Some(_) => tint,
        None => mul3(tint, BARK_FALLBACK),
    };
    let living = doc.material(&pbr_material(
        "bark",
        [living_color[0], living_color[1], living_color[2], 1.0],
        albedo_tex,
        rough_tex,
        1.0,
        normal_tex,
        "",
    ));

    let dead = if wants_dead {
        // The viewer blends dead wood toward one silver-grey keeping the grain of the
        // living bark; baking that into a copy of the albedo is the same picture.
        let bleached = texture_dir
            .and_then(|dir| textures::load_bitmap(&textures::map_path(dir, name, "albedo")))
            .map(|bitmap| bleach_bark(&bitmap, tint, mp.dead_wood_color, bleach));
        let (color, tex) = match bleached {
            Some(bitmap) => {
                let png = textures::encode_png(&bitmap)?;
                ([1.0, 1.0, 1.0], Some(texture(doc, "dead_wood_albedo", png)))
            }
            None => (lerp3(mul3(tint, BARK_FALLBACK), mp.dead_wood_color, bleach), None),
        };
        Some(doc.material(&pbr_material(
            "dead_wood",
            [color[0], color[1], color[2], 1.0],
            tex,
            rough_tex,
            1.0,
            normal_tex,
            "",
        )))
    } else {
        None
    };
    Ok((living, dead))
}

/// The leaf cards' material: the art, or the cluster atlas grown from it, cut out at the
/// viewer's alpha test.
fn leaf_material(
    doc: &mut Builder,
    params: &SpeciesParams,
    texture_dir: Option<&Path>,
    two_faced: bool,
    warnings: &mut Vec<String>,
) -> Result<usize, String> {
    let lp = &params.leaves;
    let maps = texture_dir.and_then(|dir| textures::load_leaf_maps(dir, lp));
    if texture_dir.is_some() && maps.is_none() {
        warnings.push(format!(
            "no {}_albedo.png: the leaf cards are exported untextured",
            lp.texture
        ));
    }
    // A card's own side is the only thing that tells the viewer which cell it reads,
    // so with two cells each side is its own card; with one, a card shows from both.
    let sided = if two_faced { "" } else { r#","doubleSided":true"# };
    let mask = format!(r#","alphaMode":"MASK","alphaCutoff":{}{sided}"#, num(LEAF_ALPHA_CUTOFF));
    let Some(maps) = maps else {
        return Ok(doc.material(&pbr_material(
            "leaves",
            [0.12, 0.22, 0.06, 1.0],
            None,
            None,
            LEAF_ROUGHNESS_DEFAULT,
            None,
            &mask,
        )));
    };
    let clamp = doc.sampler(CLAMP_TO_EDGE);
    let albedo = doc.image("leaf_albedo", textures::encode_png(&maps.albedo)?);
    let albedo = doc.texture(albedo, clamp);
    let (rough_tex, rough_factor) = match &maps.roughness {
        Some(r) => {
            let packed = leaf_roughness(r);
            let image = doc.image("leaf_roughness", textures::encode_png(&packed)?);
            (Some(doc.texture(image, clamp)), 1.0)
        }
        None => (None, LEAF_ROUGHNESS_DEFAULT),
    };
    Ok(doc.material(&pbr_material(
        "leaves",
        [1.0, 1.0, 1.0, 1.0],
        Some(albedo),
        rough_tex,
        rough_factor,
        None,
        &mask,
    )))
}

/// Bark, split into living and dead wood where the dead wood has its own material.
fn bark_mesh(
    doc: &mut Builder,
    mesh: &Mesh,
    params: &SpeciesParams,
    (living, dead): (usize, Option<usize>),
    wind: bool,
) -> usize {
    let n = mesh.positions.len();
    let normals: Vec<[f32; 3]> = mesh.normals.iter().map(|&v| unit_or(v, [0.0, 1.0, 0.0])).collect();
    let tangents: Vec<[f32; 4]> = mesh
        .tangents
        .iter()
        .zip(&normals)
        .map(|(t, nrm)| {
            let t3 = Vec3::new(t[0], t[1], t[2]);
            let t3 = t3.try_normalize().unwrap_or_else(|| ortho_of(Vec3::from(*nrm)));
            [t3.x, t3.y, t3.z, if t[3] < 0.0 { -1.0 } else { 1.0 }]
        })
        .collect();

    let mut attrs = vec![
        ("POSITION", doc.floats(&flat(&mesh.positions), 3, true)),
        ("NORMAL", doc.floats(&flat(&normals), 3, false)),
        ("TANGENT", doc.floats(&flat(&tangents), 4, false)),
        ("TEXCOORD_0", doc.floats(&flat(&mesh.uvs), 2, false)),
    ];
    let darken = params.mesh.bark_darken_low.clamp(0.0, 1.0);
    if darken > 0.0 {
        let shade: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .map(|p| {
                let k = (1.0 - darken * (-p[1].max(0.0) * DARKEN_FALLOFF).exp()).clamp(0.0, 1.0);
                [k, k, k]
            })
            .collect();
        attrs.push(("COLOR_0", doc.floats(&flat(&shade), 3, false)));
    }
    if wind && mesh.sway.len() == n {
        for (o, name) in ["_WIND_1", "_WIND_2", "_WIND_3"].into_iter().enumerate() {
            let order: Vec<[f32; 4]> = mesh.sway.iter().map(|s| s[o]).collect();
            attrs.push((name, doc.floats(&flat(&order), 4, false)));
        }
    }
    let attrs = attributes(&attrs);

    let (mut alive, mut gone) = (Vec::new(), Vec::new());
    for tri in mesh.indices.chunks_exact(3) {
        let is_dead = dead.is_some() && mesh.weathering.get(tri[0] as usize).is_some_and(|&w| w > 0.5);
        (if is_dead { &mut gone } else { &mut alive }).extend_from_slice(tri);
    }
    let mut primitives = Vec::new();
    for (indices, material) in [(alive, Some(living)), (gone, dead)] {
        if let (false, Some(material)) = (indices.is_empty(), material) {
            let idx = doc.indices(&indices, n);
            primitives.push(format!(r#"{{"attributes":{attrs},"indices":{idx},"material":{material}}}"#));
        }
    }
    doc.mesh("bark", &primitives)
}

/// The leaf cards, their card-local uvs mapped into the atlas cell each side reads.
fn leaf_mesh(
    doc: &mut Builder,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    material: usize,
    two_faced: bool,
    wind: bool,
) -> usize {
    let lp = &params.leaves;
    let n = leaves.positions.len();
    let (cols, rows) = (lp.atlas_cols.max(1), lp.atlas_rows.max(1));
    // The same mapping the viewer's leaf shader makes: clustering stacks its variants
    // down the sheet, and each card carries the offset to its own.
    let variants = lp.cluster.as_ref().map_or(1, |c| c.variants.max(1));
    let tall = rows * variants;
    let scale = [1.0 / cols as f32, 1.0 / tall as f32];
    let cell = |index: u32| {
        let i = index.min(cols * rows - 1);
        [(i % cols) as f32 * scale[0], (i / cols) as f32 * scale[1]]
    };
    let uvs_for = |c: [f32; 2]| -> Vec<[f32; 2]> {
        leaves
            .uvs
            .iter()
            .zip(&leaves.atlas_v)
            .map(|(uv, v)| [c[0] + uv[0] * scale[0], c[1] + v + uv[1] * scale[1]])
            .collect()
    };
    let normals: Vec<[f32; 3]> = leaves.normals.iter().map(|&v| unit_or(v, [0.0, 1.0, 0.0])).collect();
    // The per-card colour and the shade of how deep in the crown it sits, which the
    // viewer applies to everything the card reflects.
    let tints: Vec<[f32; 3]> = leaves
        .tints
        .iter()
        .map(|t| [0, 1, 2].map(|i| (t[i] * t[3]).clamp(0.0, 1.0)))
        .collect();

    let position = doc.floats(&flat(&leaves.positions), 3, true);
    let color = doc.floats(&flat(&tints), 3, false);
    let mut shared = vec![("POSITION", position), ("COLOR_0", color)];
    if wind && leaves.sway.len() == n && leaves.origins.len() == n {
        for (o, name) in ["_WIND_1", "_WIND_2", "_WIND_3"].into_iter().enumerate() {
            let order: Vec<[f32; 4]> = leaves.sway.iter().map(|s| s[o]).collect();
            shared.push((name, doc.floats(&flat(&order), 4, false)));
        }
        shared.push(("_LEAF_ORIGIN", doc.floats(&flat(&leaves.origins), 3, false)));
    }

    let mut primitives = Vec::new();
    let mut side = |doc: &mut Builder, normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]| {
        let mut attrs = shared.clone();
        attrs.push(("NORMAL", doc.floats(&flat(normals), 3, false)));
        attrs.push(("TEXCOORD_0", doc.floats(&flat(uvs), 2, false)));
        let idx = doc.indices(indices, n);
        primitives.push(format!(
            r#"{{"attributes":{},"indices":{idx},"material":{material}}}"#,
            attributes(&attrs)
        ));
    };
    side(doc, &normals, &uvs_for(cell(lp.atlas_front)), &leaves.indices);
    if two_faced {
        let flipped: Vec<[f32; 3]> = normals.iter().map(|v| v.map(|c| -c)).collect();
        let reversed: Vec<u32> = leaves
            .indices
            .chunks_exact(3)
            .flat_map(|t| [t[0], t[2], t[1]])
            .collect();
        side(doc, &flipped, &uvs_for(cell(lp.atlas_back)), &reversed);
    }
    doc.mesh("leaves", &primitives)
}

/// Dead wood as the viewer draws it at full weathering: toward `dead` in hue and tone,
/// keeping the living bark's light and dark as a modulation so the grain survives.
fn bleach_bark(albedo: &Bitmap, tint: [f32; 3], dead: [f32; 3], amount: f32) -> Bitmap {
    let lut: Vec<f32> = (0..256).map(|i| srgb_to_linear(i as f32 / 255.0)).collect();
    let mut out = albedo.clone();
    for px in out.pixels.chunks_exact_mut(4) {
        let lin = [0, 1, 2].map(|i| lut[px[i] as usize] * tint[i]);
        let grain = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
        let k = (0.55 + 1.8 * grain).clamp(0.35, 1.6);
        for i in 0..3 {
            let c = lin[i] + (dead[i] * k - lin[i]) * amount;
            px[i] = (linear_to_srgb(c.clamp(0.0, 1.0)) * 255.0).round() as u8;
        }
    }
    out
}

/// A leaf roughness map as glTF wants it — roughness in green, metalness in blue — with
/// the viewer's floor on how glossy a leaf may look.
fn leaf_roughness(map: &Bitmap) -> Bitmap {
    let floor = (LEAF_ROUGHNESS_FLOOR * 255.0).round() as u8;
    let mut out = map.clone();
    for px in out.pixels.chunks_exact_mut(4) {
        let r = px[0].max(floor);
        px.copy_from_slice(&[255, r, 0, 255]);
    }
    out
}

fn pbr_material(
    name: &str,
    color: [f32; 4],
    albedo: Option<usize>,
    roughness: Option<usize>,
    roughness_factor: f32,
    normal: Option<usize>,
    extra: &str,
) -> String {
    let mut pbr = format!(
        r#""baseColorFactor":{},"metallicFactor":0,"roughnessFactor":{}"#,
        floats(&color),
        num(roughness_factor)
    );
    if let Some(t) = albedo {
        let _ = write!(pbr, r#","baseColorTexture":{{"index":{t}}}"#);
    }
    if let Some(t) = roughness {
        let _ = write!(pbr, r#","metallicRoughnessTexture":{{"index":{t}}}"#);
    }
    let mut out = format!(r#"{{"name":{},"pbrMetallicRoughness":{{{pbr}}}"#, json_str(name));
    if let Some(t) = normal {
        let _ = write!(out, r#","normalTexture":{{"index":{t}}}"#);
    }
    out.push_str(extra);
    out.push('}');
    out
}

/// The document as it is put together: every array glTF indexes into, and the binary
/// buffer the accessors read.
struct Builder {
    format: Format,
    stem: String,
    bin: Vec<u8>,
    views: Vec<String>,
    accessors: Vec<String>,
    images: Vec<String>,
    samplers: Vec<String>,
    textures: Vec<String>,
    materials: Vec<String>,
    meshes: Vec<String>,
    nodes: Vec<String>,
    files: Vec<(String, Vec<u8>)>,
}

impl Builder {
    fn new(format: Format, stem: &str) -> Self {
        Self {
            format,
            stem: stem.to_string(),
            bin: Vec::new(),
            views: Vec::new(),
            accessors: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            files: Vec::new(),
        }
    }

    /// Appends bytes to the buffer on a four-byte boundary, which every accessor here
    /// needs, and returns the view onto them.
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let target = target.map_or(String::new(), |t| format!(r#","target":{t}"#));
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{}{target}}}"#,
            bytes.len()
        ));
        self.views.len() - 1
    }

    /// A float attribute of `width` components per vertex. Positions need their bounds.
    fn floats(&mut self, data: &[f32], width: usize, bounds: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(ARRAY_BUFFER));
        let count = data.len() / width;
        let kind = match width {
            1 => "SCALAR",
            2 => "VEC2",
            3 => "VEC3",
            _ => "VEC4",
        };
        let mut minmax = String::new();
        if bounds && count > 0 {
            let mut lo = vec![f32::INFINITY; width];
            let mut hi = vec![f32::NEG_INFINITY; width];
            for v in data.chunks_exact(width) {
                for i in 0..width {
                    lo[i] = lo[i].min(v[i]);
                    hi[i] = hi[i].max(v[i]);
                }
            }
            minmax = format!(r#","min":{},"max":{}"#, floats(&lo), floats(&hi));
        }
        self.accessors.push(format!(
            r#"{{"bufferView":{view},"componentType":{FLOAT},"count":{count},"type":"{kind}"{minmax}}}"#
        ));
        self.accessors.len() - 1
    }

    /// Triangle indices, as 16-bit when every vertex fits.
    fn indices(&mut self, indices: &[u32], vertices: usize) -> usize {
        let (bytes, kind): (Vec<u8>, u32) = if vertices <= u16::MAX as usize {
            (indices.iter().flat_map(|&i| (i as u16).to_le_bytes()).collect(), UNSIGNED_SHORT)
        } else {
            (indices.iter().flat_map(|i| i.to_le_bytes()).collect(), UNSIGNED_INT)
        };
        let view = self.view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
        self.accessors.push(format!(
            r#"{{"bufferView":{view},"componentType":{kind},"count":{},"type":"SCALAR"}}"#,
            indices.len()
        ));
        self.accessors.len() - 1
    }

    /// A PNG, inside the buffer for a `.glb` and as a file beside a `.gltf`.
    fn image(&mut self, label: &str, png: Vec<u8>) -> usize {
        let entry = match self.format {
            Format::Glb => {
                let view = self.view(&png, None);
                format!(r#"{{"name":{},"bufferView":{view},"mimeType":"image/png"}}"#, json_str(label))
            }
            Format::Gltf => {
                let file = format!("{}_{label}.png", self.stem);
                let entry = format!(r#"{{"name":{},"uri":{}}}"#, json_str(label), json_str(&uri(&file)));
                self.files.push((file, png));
                entry
            }
        };
        self.images.push(entry);
        self.images.len() - 1
    }

    fn sampler(&mut self, wrap: u32) -> usize {
        self.samplers.push(format!(
            r#"{{"magFilter":{LINEAR},"minFilter":{LINEAR_MIPMAP_LINEAR},"wrapS":{wrap},"wrapT":{wrap}}}"#
        ));
        self.samplers.len() - 1
    }

    fn texture(&mut self, image: usize, sampler: usize) -> usize {
        self.textures.push(format!(r#"{{"source":{image},"sampler":{sampler}}}"#));
        self.textures.len() - 1
    }

    fn material(&mut self, json: &str) -> usize {
        self.materials.push(json.to_string());
        self.materials.len() - 1
    }

    fn mesh(&mut self, name: &str, primitives: &[String]) -> usize {
        self.meshes.push(format!(
            r#"{{"name":{},"primitives":[{}]}}"#,
            json_str(name),
            primitives.join(",")
        ));
        self.meshes.len() - 1
    }

    fn node(&mut self, fields: &str, mesh: usize) -> usize {
        self.nodes.push(format!(r#"{{{fields},"mesh":{mesh}}}"#));
        self.nodes.len() - 1
    }

    fn finish(self, scene: String) -> Document {
        let buffer = match self.format {
            Format::Glb => format!(r#"{{"byteLength":{}}}"#, self.bin.len()),
            Format::Gltf => format!(
                r#"{{"uri":{},"byteLength":{}}}"#,
                json_str(&uri(&format!("{}.bin", self.stem))),
                self.bin.len()
            ),
        };
        let mut json = String::from(
            r#"{"asset":{"version":"2.0","generator":"arbor"},"scene":0,"scenes":["#,
        );
        json.push_str(&scene);
        json.push(']');
        for (key, items) in [
            ("nodes", &self.nodes),
            ("meshes", &self.meshes),
            ("materials", &self.materials),
            ("textures", &self.textures),
            ("samplers", &self.samplers),
            ("images", &self.images),
            ("accessors", &self.accessors),
            ("bufferViews", &self.views),
        ] {
            if !items.is_empty() {
                let _ = write!(json, r#","{key}":[{}]"#, items.join(","));
            }
        }
        if !self.bin.is_empty() {
            let _ = write!(json, r#","buffers":[{buffer}]"#);
        }
        json.push('}');
        Document {
            json,
            bin: self.bin,
            files: self.files,
        }
    }
}

fn attributes(attrs: &[(&str, usize)]) -> String {
    let fields: Vec<String> = attrs.iter().map(|(k, v)| format!(r#""{k}":{v}"#)).collect();
    format!("{{{}}}", fields.join(","))
}

fn flat<const N: usize>(v: &[[f32; N]]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
}

fn unit_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    Vec3::from(v).try_normalize().map_or(fallback, |u| u.to_array())
}

fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// A JSON number. glTF has no place for a non-finite one.
fn num(v: f32) -> String {
    if v.is_finite() { format!("{v}") } else { "0".to_string() }
}

fn floats(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|&x| num(x)).collect();
    format!("[{}]", items.join(","))
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A relative URI for a file name: anything outside the unreserved set escaped.
fn uri(name: &str) -> String {
    let mut out = String::new();
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, OAK_RON, PINE_RON};
    use ron::Value;

    fn grown(src: &str) -> (Mesh, LeafMesh, SpeciesParams) {
        let params = parse_species(src).unwrap();
        let sk = crate::grow(&params);
        (crate::build_mesh(&sk, &params), crate::build_leaves(&sk, &params), params)
    }

    /// The JSON chunk and the buffer chunk of a `.glb`, having checked the container.
    fn chunks(glb: &[u8]) -> (Value, &[u8]) {
        let word = |at: usize| u32::from_le_bytes(glb[at..at + 4].try_into().unwrap()) as usize;
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(word(4), 2, "glTF version");
        assert_eq!(word(8), glb.len(), "header length");
        let json_len = word(12);
        assert_eq!(&glb[16..20], b"JSON");
        assert_eq!(json_len % 4, 0, "JSON chunk is padded to four bytes");
        let json = std::str::from_utf8(&glb[20..20 + json_len]).unwrap();
        let bin_at = 20 + json_len;
        let bin_len = word(bin_at);
        assert_eq!(&glb[bin_at + 4..bin_at + 8], b"BIN\0");
        assert_eq!(bin_len % 4, 0);
        assert_eq!(bin_at + 8 + bin_len, glb.len());
        // What is written is plain JSON with nothing JSON has and RON lacks, so RON
        // reads it, which saves pulling in a JSON parser for the tests alone.
        let doc: Value = ron::from_str(json.trim_end()).expect("the JSON chunk parses");
        (doc, &glb[bin_at + 8..])
    }

    fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
        match v {
            Value::Map(m) => m
                .get(&Value::String(key.to_string()))
                .unwrap_or_else(|| panic!("no {key}")),
            _ => panic!("{key}: not an object"),
        }
    }

    fn has(v: &Value, key: &str) -> bool {
        matches!(v, Value::Map(m) if m.get(&Value::String(key.to_string())).is_some())
    }

    fn seq(v: &Value) -> &[Value] {
        match v {
            Value::Seq(s) => s,
            _ => panic!("not an array"),
        }
    }

    fn int(v: &Value) -> usize {
        match v {
            Value::Number(n) => n.into_f64() as usize,
            _ => panic!("not a number"),
        }
    }

    fn text(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            _ => panic!("not a string"),
        }
    }

    /// Checks every accessor against the buffer it reads and every primitive against
    /// its own accessors — what a validator would complain about first — and returns
    /// the primitives of each mesh by name.
    fn validate(doc: &Value, bin: &[u8]) -> Vec<(String, Vec<Value>)> {
        let views = seq(get(doc, "bufferViews"));
        let accessors = seq(get(doc, "accessors"));
        assert_eq!(int(get(&seq(get(doc, "buffers"))[0], "byteLength")), bin.len());
        for view in views {
            let (at, len) = (int(get(view, "byteOffset")), int(get(view, "byteLength")));
            assert!(at + len <= bin.len(), "a buffer view runs past the buffer");
        }
        let width = |kind: &Value| match text(kind).as_str() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            other => panic!("accessor type {other}"),
        };
        let size = |component: usize| match component {
            5123 => 2,
            5125 | 5126 => 4,
            other => panic!("component type {other}"),
        };
        for a in accessors {
            let view = &views[int(get(a, "bufferView"))];
            let need = int(get(a, "count")) * width(get(a, "type")) * size(int(get(a, "componentType")));
            assert!(need <= int(get(view, "byteLength")), "an accessor reads past its view");
            assert_eq!(int(get(view, "byteOffset")) % 4, 0, "an accessor is misaligned");
        }
        let count = |i: usize| int(get(&accessors[i], "count"));
        let mut out = Vec::new();
        for mesh in seq(get(doc, "meshes")) {
            let name = text(get(mesh, "name"));
            let prims = seq(get(mesh, "primitives")).to_vec();
            for p in &prims {
                let Value::Map(attrs) = get(p, "attributes") else { panic!("attributes") };
                let position = int(get(get(p, "attributes"), "POSITION"));
                let vertices = count(position);
                assert!(has(&accessors[position], "min") && has(&accessors[position], "max"));
                for (_, a) in attrs.iter() {
                    assert_eq!(count(int(a)), vertices, "{name}: attributes of unequal length");
                }
                // Every index names a vertex that exists.
                let idx = &accessors[int(get(p, "indices"))];
                let at = int(get(&views[int(get(idx, "bufferView"))], "byteOffset"));
                let wide = int(get(idx, "componentType")) == 5125;
                let n = int(get(idx, "count"));
                assert_eq!(n % 3, 0);
                let max = (0..n)
                    .map(|i| {
                        if wide {
                            u32::from_le_bytes(bin[at + i * 4..at + i * 4 + 4].try_into().unwrap()) as usize
                        } else {
                            u16::from_le_bytes(bin[at + i * 2..at + i * 2 + 2].try_into().unwrap()) as usize
                        }
                    })
                    .max()
                    .unwrap_or(0);
                assert!(max < vertices, "{name}: an index past the last vertex");
            }
            out.push((name, prims));
        }
        out
    }

    fn material_names(doc: &Value, prims: &[Value]) -> Vec<String> {
        let materials = seq(get(doc, "materials"));
        prims
            .iter()
            .map(|p| text(get(&materials[int(get(p, "material"))], "name")))
            .collect()
    }

    #[test]
    fn a_glb_is_one_well_formed_scene_of_bark_and_leaves() {
        let (mesh, leaves, params) = grown(PINE_RON);
        let glb = to_glb(&mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        let names: Vec<&str> = meshes.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["bark", "leaves"]);
        // Every triangle of both made it across.
        let accessors = seq(get(&doc, "accessors"));
        let triangles = |prims: &[Value]| -> usize {
            prims
                .iter()
                .map(|p| int(get(&accessors[int(get(p, "indices"))], "count")) / 3)
                .sum()
        };
        assert_eq!(triangles(&meshes[0].1), mesh.triangle_count());
        assert_eq!(triangles(&meshes[1].1), leaves.triangle_count());
    }

    #[test]
    fn dead_wood_is_its_own_material() {
        // The pine's dead band is bleached silver in the viewer; a single bark material
        // would export it as living bark.
        let (mesh, leaves, params) = grown(PINE_RON);
        assert!(params.mesh.dead_wood_weathering > 0.0);
        let glb = to_glb(&mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        assert_eq!(material_names(&doc, &meshes[0].1), ["bark", "dead_wood"]);
    }

    #[test]
    fn a_card_with_a_different_underside_is_exported_from_both_sides() {
        // The oak shows the pale underside of its leaf from behind. glTF cannot pick a
        // texture by which face is seen, so it takes a back-facing copy of every card.
        let (mesh, leaves, params) = grown(OAK_RON);
        let glb = to_glb(&mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        let materials = seq(get(&doc, "materials"));
        let (_, prims) = meshes.iter().find(|(n, _)| n == "leaves").unwrap();
        assert_eq!(prims.len(), 2, "the oak's cards need a back face");
        assert!(!has(&materials[int(get(&prims[0], "material"))], "doubleSided"));

        // The pine's cards read one cell from either side, so they stay single.
        let (mesh, leaves, params) = grown(PINE_RON);
        let glb = to_glb(&mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        let materials = seq(get(&doc, "materials"));
        let (_, prims) = meshes.iter().find(|(n, _)| n == "leaves").unwrap();
        assert_eq!(prims.len(), 1);
        assert!(has(&materials[int(get(&prims[0], "material"))], "doubleSided"));
    }

    #[test]
    fn wind_data_goes_in_only_when_asked_for() {
        let (mesh, leaves, params) = grown(PINE_RON);
        let attribute_names = |wind: bool| -> Vec<String> {
            let options = ExportOptions {
                wind,
                ..Default::default()
            };
            let glb = to_glb(&mesh, &leaves, &params, &options).unwrap();
            let (doc, bin) = chunks(&glb);
            let mut names = Vec::new();
            for (_, prims) in validate(&doc, bin) {
                for p in prims {
                    if let Value::Map(m) = get(&p, "attributes") {
                        names.extend(m.keys().map(text));
                    }
                }
            }
            names
        };
        let without = attribute_names(false);
        assert!(without.iter().all(|a| !a.starts_with('_')), "{without:?}");
        let with = attribute_names(true);
        for a in ["_WIND_1", "_WIND_2", "_WIND_3", "_LEAF_ORIGIN"] {
            assert!(with.iter().any(|w| w == a), "no {a} in {with:?}");
        }
    }

    #[test]
    fn a_gltf_names_its_buffer_and_textures_as_files_beside_it() {
        let (mesh, leaves, params) = grown(PINE_RON);
        let dir = std::env::temp_dir().join(format!("arbor-gltf-test-{}", std::process::id()));
        let path = dir.join("my pine.gltf");
        let options = ExportOptions {
            textures: Some(PathBuf::from("../../assets/textures")),
            wind: false,
        };
        let report = export(&path, &mesh, &leaves, &params, &options).unwrap();
        let doc: Value = ron::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(text(get(&seq(get(&doc, "buffers"))[0], "uri")), "my%20pine.bin");
        let images = seq(get(&doc, "images"));
        assert!(!images.is_empty(), "no textures were written");
        for image in images {
            let file = text(get(image, "uri")).replace("%20", " ");
            assert!(dir.join(&file).is_file(), "{file} is not beside the .gltf");
        }
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(report.files.len(), 2 + images.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
