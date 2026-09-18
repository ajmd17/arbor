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
//! With `wind` set, every primitive also carries the `ARBOR_tree_wind` extension, which
//! is enough for an engine to sway the tree as the viewer does: a table of the stems the
//! tree bends as (`branches`, two VEC4s a stem: its pivot and reach, then the row of the
//! stem carrying it, how far out along that one it leaves, and its order; row 0 is the
//! trunk and empty), the species' `flexibility` per order, `frequency`, `flutter` and the
//! tree's `height`, and on leaves `leafOrigins`, the twig point each card hangs from.
//! Each vertex names its row and how far out along it it sits in `TEXCOORD_1`. The tree's
//! foot is at the origin, which the trunk's bend is reckoned from.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use glam::Vec3;

use crate::cluster::Bitmap;
use crate::leaves::LeafMesh;
use crate::math::ortho_of;
use crate::mesh::Mesh;
use crate::skeleton::Skeleton;
use crate::species::{SpeciesParams, SpeciesTemplate};
use crate::textures;
use crate::wind::{SwayAt, SwayField, SwayStem};

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

/// The extension the wind data goes in.
pub const TREE_WIND_EXTENSION: &str = "ARBOR_tree_wind";

/// What goes into an export besides the geometry.
#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    /// Where the texture maps are read from. `None` writes the materials without any.
    pub textures: Option<PathBuf>,
    /// Also write what an engine needs to sway the tree, in the `ARBOR_tree_wind`
    /// extension.
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

    pub fn extension(self) -> &'static str {
        match self {
            Format::Glb => "glb",
            Format::Gltf => "gltf",
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

/// A species' textures, read, baked and encoded once, ready to go into any number of
/// exports of it. The leaf atlas of a clustering species is grown here, and every map
/// is encoded here, which is most of what an export costs; a batch pays it once.
#[derive(Default)]
pub struct Textures {
    bark_albedo: Option<Vec<u8>>,
    bark_normal: Option<Vec<u8>>,
    bark_roughness: Option<Vec<u8>>,
    /// The bark albedo with the viewer's dead-wood bleaching baked in, for a species
    /// that bleaches its dead wood.
    dead_albedo: Option<Vec<u8>>,
    leaf_albedo: Option<Vec<u8>>,
    /// The leaf roughness, packed as glTF wants it.
    leaf_roughness: Option<Vec<u8>>,
    /// Maps that were looked for and not found.
    pub warnings: Vec<String>,
}

impl Textures {
    /// Reads the species' maps from `dir`. A missing map is reported in `warnings` and
    /// left out, and the material it belongs to is exported without it.
    pub fn load(params: &SpeciesParams, dir: &Path) -> Result<Self, String> {
        let mut out = Textures::default();
        let mp = &params.mesh;
        let name = &mp.bark_texture;
        // Bark maps go in as they are on disk, byte for byte: nothing needs changing.
        let read = |map: &str| std::fs::read(textures::map_path(dir, name, map)).ok();
        out.bark_albedo = read("albedo");
        out.bark_normal = read("normal");
        out.bark_roughness = read("roughness");
        if out.bark_albedo.is_none() {
            out.warnings.push(format!("no {name}_albedo.png: the bark is exported untextured"));
        }
        let bleach = mp.dead_wood_weathering.clamp(0.0, 1.0);
        if bleach > 0.0
            && let Some(bitmap) = textures::load_bitmap(&textures::map_path(dir, name, "albedo"))
        {
            let bleached = bleach_bark(&bitmap, mp.bark_tint, mp.dead_wood_color, bleach);
            out.dead_albedo = Some(textures::encode_png(&bleached)?);
        }

        // A tree grown without leaves has no use for them, and growing a cluster atlas
        // is the slowest thing here.
        let lp = &params.leaves;
        if lp.enabled {
            match textures::load_leaf_maps(dir, lp) {
                Some(maps) => {
                    out.leaf_albedo = Some(textures::encode_png(&maps.albedo)?);
                    if let Some(r) = &maps.roughness {
                        out.leaf_roughness = Some(textures::encode_png(&leaf_roughness(r))?);
                    }
                }
                None => out.warnings.push(format!(
                    "no {}_albedo.png: the leaf cards are exported untextured",
                    lp.texture
                )),
            }
        }
        Ok(out)
    }
}

/// Writes trees of one species, preparing its textures once however many are written.
pub struct Exporter {
    textures: Textures,
    wind: bool,
    /// What a `.gltf`'s image files are named after, in place of the file's own name,
    /// so a batch of trees shares one set.
    shared_images: Option<String>,
    /// Image files this exporter has already written, which are not written again.
    written: std::collections::HashSet<PathBuf>,
}

impl Exporter {
    pub fn new(params: &SpeciesParams, options: &ExportOptions) -> Result<Self, String> {
        let textures = match &options.textures {
            Some(dir) => Textures::load(params, dir)?,
            None => Textures::default(),
        };
        Ok(Self {
            textures,
            wind: options.wind,
            shared_images: None,
            written: Default::default(),
        })
    }

    /// Maps that were looked for and not found.
    pub fn warnings(&self) -> &[String] {
        &self.textures.warnings
    }

    /// Names the image files of every `.gltf` written from here on `<name>_*.png`,
    /// rather than after each `.gltf`, so trees written into one folder share them.
    pub fn share_images(&mut self, name: &str) {
        self.shared_images = Some(name.to_string());
    }

    /// Writes one tree to `path`, as a `.glb` or a `.gltf` by its extension. `params`
    /// has to be the species the exporter was made for; only the seed may differ. The
    /// mesh and leaves have to be the ones built from `skeleton`.
    pub fn write(
        &mut self,
        path: &Path,
        skeleton: &Skeleton,
        mesh: &Mesh,
        leaves: &LeafMesh,
        params: &SpeciesParams,
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
        let images = self.shared_images.clone().unwrap_or_else(|| stem.clone());
        let tree = Tree { skeleton, mesh, leaves, params };
        let doc = build(format, &stem, &images, &tree, &self.textures, self.wind);
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
                    let file = dir.join(name);
                    if self.written.insert(file.clone()) {
                        write(file, bytes)?;
                    }
                }
            }
        }
        Ok(report)
    }
}

/// Writes the tree to `path`, as a `.glb` or a `.gltf` by its extension.
pub fn export(
    path: &Path,
    skeleton: &Skeleton,
    mesh: &Mesh,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    options: &ExportOptions,
) -> Result<ExportReport, String> {
    let mut exporter = Exporter::new(params, options)?;
    let mut report = exporter.write(path, skeleton, mesh, leaves, params)?;
    report.warnings = exporter.textures.warnings;
    Ok(report)
}

/// The tree as one `.glb`, in memory.
pub fn to_glb(
    skeleton: &Skeleton,
    mesh: &Mesh,
    leaves: &LeafMesh,
    params: &SpeciesParams,
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let exporter = Exporter::new(params, options)?;
    let tree = Tree { skeleton, mesh, leaves, params };
    let doc = build(Format::Glb, "tree", "tree", &tree, &exporter.textures, options.wind);
    Ok(doc.glb())
}

/// What a batch wrote.
#[derive(Clone, Debug, Default)]
pub struct BatchReport {
    /// The tree files, one per tree, in seed order. A `.gltf`'s buffer and images are
    /// counted in `bytes` but not listed.
    pub trees: Vec<PathBuf>,
    pub bytes: u64,
    pub warnings: Vec<String>,
    /// Whether `progress` asked for it to stop before every tree was written.
    pub stopped: bool,
}

/// The file a batch writes the tree grown from `seed` to: `<stem>.<ext>` when it is the
/// only one, and `<stem>_seed<seed>.<ext>` when there are several, so any of them can
/// be grown again from its name.
pub fn batch_file(stem: &str, format: Format, seed: u64, count: u32) -> String {
    if count <= 1 {
        format!("{stem}.{}", format.extension())
    } else {
        format!("{stem}_seed{seed}.{}", format.extension())
    }
}

/// Grows and writes `count` trees of one species into `dir`: the one `species` grows at
/// its own seed, and then one from each seed after it, every range in the species
/// landing afresh for each. The textures are prepared once, from the first tree, and a
/// batch of `.gltf` shares one set of image files, `<stem>_*.png` — so a range on a
/// colour that is baked into an image, the bark tint and the dead-wood colour, varies
/// between batches but not within one.
///
/// `progress` hears how many trees are done after each one; returning false from it
/// stops the batch there.
pub fn export_batch(
    dir: &Path,
    stem: &str,
    format: Format,
    species: &SpeciesTemplate,
    count: u32,
    options: &ExportOptions,
    mut progress: impl FnMut(u32) -> bool,
) -> Result<BatchReport, String> {
    let count = count.max(1);
    let mut exporter = Exporter::new(&species.instance(), options)?;
    if count > 1 {
        exporter.share_images(stem);
    }
    let mut report = BatchReport {
        warnings: exporter.warnings().to_vec(),
        ..Default::default()
    };
    for i in 0..count {
        let mut at_seed = species.clone();
        at_seed.seed = species.seed.wrapping_add(u64::from(i));
        let tree = at_seed.instance();
        let skeleton = crate::grow(&tree);
        let mesh = crate::build_mesh(&skeleton, &tree);
        let leaves = crate::build_leaves(&skeleton, &tree);
        let path = dir.join(batch_file(stem, format, tree.seed, count));
        let written = exporter.write(&path, &skeleton, &mesh, &leaves, &tree)?;
        report.bytes += written.bytes;
        report.trees.push(path);
        if !progress(i + 1) && i + 1 < count {
            report.stopped = true;
            break;
        }
    }
    Ok(report)
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

/// One grown tree, everything an export reads.
struct Tree<'a> {
    skeleton: &'a Skeleton,
    mesh: &'a Mesh,
    leaves: &'a LeafMesh,
    params: &'a SpeciesParams,
}

/// The document for one tree. `stem` names a `.gltf`'s buffer file and `images` its
/// image files.
fn build(format: Format, stem: &str, images: &str, tree: &Tree, textures: &Textures, wind: bool) -> Document {
    let Tree { mesh, leaves, params, .. } = *tree;
    let mut doc = Builder::new(format, stem, images);
    let wind = wind.then(|| TreeWind::new(&mut doc, tree));

    let mut children = Vec::new();
    if !mesh.positions.is_empty() {
        let dead = mesh.weathering.iter().any(|&w| w > 0.5);
        let materials = bark_materials(&mut doc, params, textures, dead);
        let m = bark_mesh(&mut doc, mesh, params, materials, wind.as_ref());
        children.push(doc.node(r#""name":"bark""#, m));
    }
    if !leaves.is_empty() {
        let lp = &params.leaves;
        let (cols, rows) = (lp.atlas_cols.max(1), lp.atlas_rows.max(1));
        let last = cols * rows - 1;
        let two_faced = lp.atlas_front.min(last) != lp.atlas_back.min(last);
        let material = leaf_material(&mut doc, textures, two_faced);
        let m = leaf_mesh(&mut doc, leaves, params, material, two_faced, wind.as_ref());
        children.push(doc.node(r#""name":"leaves""#, m));
    }

    let mut extras = format!(
        r#""arbor":{{"species":{},"seed":{}"#,
        json_str(&params.name),
        params.seed
    );
    extras.push('}');
    let child_list = children.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    let root = doc.nodes.len();
    doc.nodes.push(format!(
        r#"{{"name":{},"children":[{child_list}],"extras":{{{extras}}}}}"#,
        json_str(&params.name)
    ));
    let scene = format!(r#"{{"name":{},"nodes":[{root}]}}"#, json_str(&params.name));
    doc.finish(scene)
}

/// The living bark's material and, when the tree has dead wood the species bleaches,
/// the dead wood's.
fn bark_materials(
    doc: &mut Builder,
    params: &SpeciesParams,
    textures: &Textures,
    has_dead: bool,
) -> (usize, Option<usize>) {
    let mp = &params.mesh;
    let tint = mp.bark_tint;
    let bleach = mp.dead_wood_weathering.clamp(0.0, 1.0);
    let wants_dead = has_dead && bleach > 0.0;

    let repeat = doc.sampler(REPEAT);
    let texture = |doc: &mut Builder, label: &str, png: &Option<Vec<u8>>| {
        png.as_ref().map(|png| {
            let image = doc.image(label, png);
            doc.texture(image, repeat)
        })
    };
    let albedo_tex = texture(doc, "bark_albedo", &textures.bark_albedo);
    let normal_tex = texture(doc, "bark_normal", &textures.bark_normal);
    let rough_tex = texture(doc, "bark_roughness", &textures.bark_roughness);

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
        let (color, tex) = match texture(doc, "dead_wood_albedo", &textures.dead_albedo) {
            Some(tex) => ([1.0, 1.0, 1.0], Some(tex)),
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
    (living, dead)
}

/// The leaf cards' material: the art, or the cluster atlas grown from it, cut out at the
/// viewer's alpha test.
fn leaf_material(doc: &mut Builder, textures: &Textures, two_faced: bool) -> usize {
    // A card's own side is the only thing that tells the viewer which cell it reads,
    // so with two cells each side is its own card; with one, a card shows from both.
    let sided = if two_faced { "" } else { r#","doubleSided":true"# };
    let mask = format!(r#","alphaMode":"MASK","alphaCutoff":{}{sided}"#, num(LEAF_ALPHA_CUTOFF));
    let Some(albedo) = &textures.leaf_albedo else {
        return doc.material(&pbr_material(
            "leaves",
            [0.12, 0.22, 0.06, 1.0],
            None,
            None,
            LEAF_ROUGHNESS_DEFAULT,
            None,
            &mask,
        ));
    };
    let clamp = doc.sampler(CLAMP_TO_EDGE);
    let albedo = doc.image("leaf_albedo", albedo);
    let albedo = doc.texture(albedo, clamp);
    let (rough_tex, rough_factor) = match &textures.leaf_roughness {
        Some(png) => {
            let image = doc.image("leaf_roughness", png);
            (Some(doc.texture(image, clamp)), 1.0)
        }
        None => (None, LEAF_ROUGHNESS_DEFAULT),
    };
    doc.material(&pbr_material(
        "leaves",
        [1.0, 1.0, 1.0, 1.0],
        Some(albedo),
        rough_tex,
        rough_factor,
        None,
        &mask,
    ))
}

/// Bark, split into living and dead wood where the dead wood has its own material.
fn bark_mesh(
    doc: &mut Builder,
    mesh: &Mesh,
    params: &SpeciesParams,
    (living, dead): (usize, Option<usize>),
    wind: Option<&TreeWind>,
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
    let wind = wind.filter(|_| mesh.sway_at.len() == n);
    if let Some(wind) = wind {
        attrs.push(("TEXCOORD_1", wind.places(doc, &mesh.sway_at)));
    }
    let attrs = attributes(&attrs);
    let extension = wind.map_or(String::new(), |w| w.extension(None));

    let (mut alive, mut gone) = (Vec::new(), Vec::new());
    for tri in mesh.indices.chunks_exact(3) {
        let is_dead = dead.is_some() && mesh.weathering.get(tri[0] as usize).is_some_and(|&w| w > 0.5);
        (if is_dead { &mut gone } else { &mut alive }).extend_from_slice(tri);
    }
    let mut primitives = Vec::new();
    for (indices, material) in [(alive, Some(living)), (gone, dead)] {
        if let (false, Some(material)) = (indices.is_empty(), material) {
            let idx = doc.indices(&indices, n);
            primitives.push(format!(
                r#"{{"attributes":{attrs},"indices":{idx},"material":{material}{extension}}}"#
            ));
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
    wind: Option<&TreeWind>,
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
    let mut extension = String::new();
    if let Some(wind) = wind.filter(|_| leaves.sway_at.len() == n && leaves.origins.len() == n) {
        shared.push(("TEXCOORD_1", wind.places(doc, &leaves.sway_at)));
        let origins = doc.data(&flat(&leaves.origins), 3);
        extension = wind.extension(Some(origins));
    }

    let mut primitives = Vec::new();
    let mut side = |doc: &mut Builder, normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]| {
        let mut attrs = shared.clone();
        attrs.push(("NORMAL", doc.floats(&flat(normals), 3, false)));
        attrs.push(("TEXCOORD_0", doc.floats(&flat(uvs), 2, false)));
        let idx = doc.indices(indices, n);
        primitives.push(format!(
            r#"{{"attributes":{},"indices":{idx},"material":{material}{extension}}}"#,
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

/// The tree's `ARBOR_tree_wind` data, shared by every primitive: the table of stems,
/// written once, and the species' wind settings.
struct TreeWind {
    /// The accessor holding the table.
    branches: usize,
    /// Row of the table each stem is in, by its first node.
    rows: std::collections::HashMap<u32, u32>,
    settings: String,
}

impl TreeWind {
    /// Writes the table of every stem a vertex of the tree bends with, and every stem
    /// carrying those, into the document.
    fn new(doc: &mut Builder, tree: &Tree) -> Self {
        let stems: std::collections::HashMap<u32, SwayStem> = SwayField::new(tree.skeleton)
            .stems()
            .into_iter()
            .map(|s| (s.first, s))
            .collect();
        let mut used = std::collections::BTreeSet::new();
        for at in tree.mesh.sway_at.iter().chain(&tree.leaves.sway_at) {
            let mut stem = at.stem;
            while let Some(first) = stem.filter(|&f| used.insert(f)) {
                stem = stems.get(&first).and_then(|s| s.carrier.stem);
            }
        }
        // First nodes come in growth order, so a stem's carrier always has the lower row.
        let rows: std::collections::HashMap<u32, u32> =
            used.iter().filter(|f| stems.contains_key(f)).zip(1u32..).map(|(&f, row)| (f, row)).collect();
        let mut table = vec![[0.0f32; 4]; 2 * (rows.len() + 1)];
        for (first, &row) in &rows {
            let stem = &stems[first];
            let carrier = stem.carrier.stem.and_then(|c| rows.get(&c)).copied().unwrap_or(0);
            let at = 2 * row as usize;
            table[at] = [stem.pivot.x, stem.pivot.y, stem.pivot.z, stem.reach];
            table[at + 1] = [carrier as f32, stem.carrier.t, f32::from(stem.order), 0.0];
        }
        let branches = doc.data(&flat(&table), 4);

        let w = &tree.params.wind;
        let settings = format!(
            r#""flexibility":{},"frequency":{},"flutter":{},"height":{}"#,
            floats(&w.flexibility),
            num(w.frequency),
            num(w.flutter),
            num(tree.skeleton.stats().height)
        );
        doc.uses(TREE_WIND_EXTENSION);
        Self { branches, rows, settings }
    }

    /// Each vertex's row in the table and how far out along that stem it sits, as a
    /// `TEXCOORD_1` accessor.
    fn places(&self, doc: &mut Builder, at: &[SwayAt]) -> usize {
        let places: Vec<[f32; 2]> = at
            .iter()
            .map(|a| match a.stem.and_then(|s| self.rows.get(&s)) {
                Some(&row) => [row as f32, a.t],
                None => [0.0, 0.0],
            })
            .collect();
        doc.floats(&flat(&places), 2, false)
    }

    /// The extension as it goes on a primitive, with a leaf primitive's card origins.
    fn extension(&self, leaf_origins: Option<usize>) -> String {
        let origins = leaf_origins.map_or(String::new(), |a| format!(r#","leafOrigins":{a}"#));
        format!(
            r#","extensions":{{"{TREE_WIND_EXTENSION}":{{"branches":{}{origins},{}}}}}"#,
            self.branches, self.settings
        )
    }
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
    /// What a `.gltf`'s buffer file is named after.
    stem: String,
    /// What a `.gltf`'s image files are named after.
    images_stem: String,
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
    extensions: Vec<&'static str>,
}

impl Builder {
    fn new(format: Format, stem: &str, images_stem: &str) -> Self {
        Self {
            format,
            stem: stem.to_string(),
            images_stem: images_stem.to_string(),
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
            extensions: Vec::new(),
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
        self.accessor(data, width, bounds, Some(ARRAY_BUFFER))
    }

    /// Floats an extension reads, rather than the vertices.
    fn data(&mut self, data: &[f32], width: usize) -> usize {
        self.accessor(data, width, false, None)
    }

    fn accessor(&mut self, data: &[f32], width: usize, bounds: bool, target: Option<u32>) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = self.view(&bytes, target);
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
    fn image(&mut self, label: &str, png: &[u8]) -> usize {
        let entry = match self.format {
            Format::Glb => {
                let view = self.view(png, None);
                format!(r#"{{"name":{},"bufferView":{view},"mimeType":"image/png"}}"#, json_str(label))
            }
            Format::Gltf => {
                let file = format!("{}_{label}.png", self.images_stem);
                let entry = format!(r#"{{"name":{},"uri":{}}}"#, json_str(label), json_str(&uri(&file)));
                self.files.push((file, png.to_vec()));
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

    /// Names an extension in `extensionsUsed`.
    fn uses(&mut self, extension: &'static str) {
        if !self.extensions.contains(&extension) {
            self.extensions.push(extension);
        }
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
        let mut json = String::from(r#"{"asset":{"version":"2.0","generator":"arbor"}"#);
        if !self.extensions.is_empty() {
            let used: Vec<String> = self.extensions.iter().map(|e| json_str(e)).collect();
            let _ = write!(json, r#","extensionsUsed":[{}]"#, used.join(","));
        }
        json.push_str(r#","scene":0,"scenes":["#);
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

    fn grown(src: &str) -> (Skeleton, Mesh, LeafMesh, SpeciesParams) {
        let params = parse_species(src).unwrap();
        let sk = crate::grow(&params);
        let (mesh, leaves) = (crate::build_mesh(&sk, &params), crate::build_leaves(&sk, &params));
        (sk, mesh, leaves, params)
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

    /// The floats an accessor holds.
    fn read_floats(doc: &Value, bin: &[u8], accessor: usize) -> Vec<f32> {
        let a = &seq(get(doc, "accessors"))[accessor];
        let view = &seq(get(doc, "bufferViews"))[int(get(a, "bufferView"))];
        let at = int(get(view, "byteOffset"));
        let width = match text(get(a, "type")).as_str() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            _ => 4,
        };
        (0..int(get(a, "count")) * width)
            .map(|i| f32::from_le_bytes(bin[at + i * 4..at + i * 4 + 4].try_into().unwrap()))
            .collect()
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
        let (sk, mesh, leaves, params) = grown(PINE_RON);
        let glb = to_glb(&sk, &mesh, &leaves, &params, &ExportOptions::default()).unwrap();
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
        let (sk, mesh, leaves, params) = grown(PINE_RON);
        assert!(params.mesh.dead_wood_weathering > 0.0);
        let glb = to_glb(&sk, &mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        assert_eq!(material_names(&doc, &meshes[0].1), ["bark", "dead_wood"]);
    }

    #[test]
    fn a_card_with_a_different_underside_is_exported_from_both_sides() {
        // The oak shows the pale underside of its leaf from behind. glTF cannot pick a
        // texture by which face is seen, so it takes a back-facing copy of every card.
        let (sk, mesh, leaves, params) = grown(OAK_RON);
        let glb = to_glb(&sk, &mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        let materials = seq(get(&doc, "materials"));
        let (_, prims) = meshes.iter().find(|(n, _)| n == "leaves").unwrap();
        assert_eq!(prims.len(), 2, "the oak's cards need a back face");
        assert!(!has(&materials[int(get(&prims[0], "material"))], "doubleSided"));

        // The pine's cards read one cell from either side, so they stay single.
        let (sk, mesh, leaves, params) = grown(PINE_RON);
        let glb = to_glb(&sk, &mesh, &leaves, &params, &ExportOptions::default()).unwrap();
        let (doc, bin) = chunks(&glb);
        let meshes = validate(&doc, bin);
        let materials = seq(get(&doc, "materials"));
        let (_, prims) = meshes.iter().find(|(n, _)| n == "leaves").unwrap();
        assert_eq!(prims.len(), 1);
        assert!(has(&materials[int(get(&prims[0], "material"))], "doubleSided"));
    }

    #[test]
    fn wind_data_goes_in_only_when_asked_for() {
        let (sk, mesh, leaves, params) = grown(PINE_RON);
        let export = |wind: bool| {
            let options = ExportOptions {
                wind,
                ..Default::default()
            };
            let glb = to_glb(&sk, &mesh, &leaves, &params, &options).unwrap();
            let (doc, bin) = chunks(&glb);
            let prims: Vec<Value> = validate(&doc, bin).into_iter().flat_map(|(_, p)| p).collect();
            (doc, prims)
        };
        let (doc, prims) = export(false);
        assert!(!has(&doc, "extensionsUsed"));
        for p in &prims {
            assert!(!has(p, "extensions") && !has(get(p, "attributes"), "TEXCOORD_1"));
        }
        let (doc, prims) = export(true);
        assert!(seq(get(&doc, "extensionsUsed")).iter().any(|e| text(e) == TREE_WIND_EXTENSION));
        for p in &prims {
            assert!(has(get(p, "attributes"), "TEXCOORD_1"));
            let wind = get(get(p, "extensions"), TREE_WIND_EXTENSION);
            for key in ["branches", "flexibility", "frequency", "flutter", "height"] {
                assert!(has(wind, key), "no {key}");
            }
        }
    }

    #[test]
    fn the_wind_table_gives_back_every_vertex_s_sway() {
        // What an engine reading the extension does: walk each vertex up the table of
        // stems. It has to come to the sway the viewer draws the vertex with.
        use crate::wind::{cantilever, SWAY_ORDERS};
        let (sk, mesh, leaves, params) = grown(OAK_RON);
        let options = ExportOptions {
            wind: true,
            ..Default::default()
        };
        let glb = to_glb(&sk, &mesh, &leaves, &params, &options).unwrap();
        let (doc, bin) = chunks(&glb);
        for (name, prims) in validate(&doc, bin) {
            let expected = if name == "bark" { &mesh.sway } else { &leaves.sway };
            for p in &prims {
                let wind = get(get(p, "extensions"), TREE_WIND_EXTENSION);
                let table = read_floats(&doc, bin, int(get(wind, "branches")));
                assert!(table[..8].iter().all(|&v| v == 0.0), "row 0 is the trunk's, and empty");
                let places = read_floats(&doc, bin, int(get(get(p, "attributes"), "TEXCOORD_1")));
                assert_eq!(places.len(), expected.len() * 2);
                for (v, sway) in expected.iter().enumerate() {
                    let mut walked = [[0.0f32; 4]; SWAY_ORDERS];
                    let (mut row, mut t) = (places[v * 2] as usize, places[v * 2 + 1]);
                    for _ in 0..SWAY_ORDERS {
                        if row == 0 {
                            break;
                        }
                        let stem = &table[row * 8..row * 8 + 8];
                        walked[stem[6] as usize] = [stem[0], stem[1], stem[2], stem[3] * cantilever(t)];
                        (row, t) = (stem[4] as usize, stem[5]);
                    }
                    assert_eq!(row, 0, "{name} vertex {v}: the walk never reached the trunk");
                    for o in 0..SWAY_ORDERS {
                        assert!(
                            (walked[o][3] - sway[o][3]).abs() < 1e-4,
                            "{name} vertex {v}, order {o}: swings {} by the table and {} in the viewer",
                            walked[o][3],
                            sway[o][3]
                        );
                        if sway[o][3] > 0.0 {
                            assert_eq!(walked[o][..3], sway[o][..3], "{name} vertex {v} bends order {o} about another pivot");
                        }
                    }
                }
                if name == "leaves" {
                    let origins = read_floats(&doc, bin, int(get(wind, "leafOrigins")));
                    assert_eq!(origins, flat(&leaves.origins));
                }
            }
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("arbor-gltf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn fir_open() -> SpeciesTemplate {
        let (_, src) = crate::species::builtin_presets()
            .into_iter()
            .find(|(name, _)| *name == "fir_open")
            .expect("the open-grown fir is a built-in");
        crate::species::parse_template(src).unwrap()
    }

    #[test]
    fn a_batch_grows_one_tree_per_seed_and_names_each_by_it() {
        let dir = scratch("batch");
        let mut params = fir_open();
        params.seed = 41;
        let report =
            export_batch(&dir, "fir", Format::Glb, &params, 3, &ExportOptions::default(), |_| true)
                .unwrap();
        let names: Vec<String> = report
            .trees
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["fir_seed41.glb", "fir_seed42.glb", "fir_seed43.glb"]);
        assert!(!report.stopped);
        // Each is its own tree: the seed it names is the one it was grown from, and no
        // two came out the same.
        let mut buffers = Vec::new();
        for (path, seed) in report.trees.iter().zip(41u64..) {
            let glb = std::fs::read(path).unwrap();
            let (doc, bin) = chunks(&glb);
            validate(&doc, bin);
            let root = seq(get(&doc, "nodes")).last().unwrap();
            assert_eq!(int(get(get(get(root, "extras"), "arbor"), "seed")), seed as usize);
            buffers.push(bin.to_vec());
        }
        assert!(buffers[0] != buffers[1] && buffers[1] != buffers[2]);

        // One tree is just the name.
        let one = export_batch(&dir, "one", Format::Glb, &params, 1, &ExportOptions::default(), |_| true)
            .unwrap();
        assert_eq!(one.trees, [dir.join("one.glb")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_gltf_batch_shares_one_set_of_images() {
        // The textures are the species', not the tree's, so twenty trees should not
        // write twenty copies of the leaf atlas.
        let dir = scratch("shared");
        let options = ExportOptions {
            textures: Some(PathBuf::from("../../assets/textures")),
            wind: false,
        };
        let report = export_batch(&dir, "fir", Format::Gltf, &fir_open(), 2, &options, |_| true).unwrap();
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let mut uris = Vec::new();
        for tree in &report.trees {
            let doc: Value = ron::from_str(&std::fs::read_to_string(tree).unwrap()).unwrap();
            let images: Vec<String> = seq(get(&doc, "images")).iter().map(|i| text(get(i, "uri"))).collect();
            assert!(images.iter().all(|u| u.starts_with("fir_") && dir.join(u).is_file()));
            uris.push(images);
        }
        assert_eq!(uris[0], uris[1], "the two trees name different images");
        let pngs = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "png"))
            .count();
        assert_eq!(pngs, uris[0].len(), "images were written once per tree");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_batch_stops_when_asked() {
        let dir = scratch("stop");
        let report = export_batch(&dir, "fir", Format::Glb, &fir_open(), 5, &ExportOptions::default(), |done| {
            done < 2
        })
        .unwrap();
        assert!(report.stopped);
        assert_eq!(report.trees.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_gltf_names_its_buffer_and_textures_as_files_beside_it() {
        let (sk, mesh, leaves, params) = grown(PINE_RON);
        let dir = std::env::temp_dir().join(format!("arbor-gltf-test-{}", std::process::id()));
        let path = dir.join("my pine.gltf");
        let options = ExportOptions {
            textures: Some(PathBuf::from("../../assets/textures")),
            wind: false,
        };
        let report = export(&path, &sk, &mesh, &leaves, &params, &options).unwrap();
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
