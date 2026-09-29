//! Reads a glTF or GLB file into plain data: no GL, so it can run on a worker thread and
//! be tested without a window.
//!
//! What is kept: triangle meshes, their metallic-roughness materials and textures, and the
//! scene's node transforms, flattened so every node that shows a mesh becomes one
//! [`Instance`]. Images are PNG, JPEG or KTX2 (see [`crate::ktx`]), the last also through
//! `KHR_texture_basisu`. What is not, and is reported in [`ImportedModel::warnings`] when a file
//! uses it: skinning and morph targets (drawn in the rest pose), animation, cameras, lights,
//! texture-coordinate sets other than the first, texture transforms, and sampler wrap modes.

use std::collections::HashSet;
use std::path::Path;

use glam::{Mat4, Vec3};

/// One drawable piece of geometry, a glTF primitive.
pub struct Primitive {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub indices: Vec<u32>,
    /// Index into [`ImportedModel::materials`].
    pub material: usize,
}

/// A decoded image, always eight-bit RGBA.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Which slot of a material a texture sits in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    BaseColor,
    Normal,
    MetallicRoughness,
    Occlusion,
    Emissive,
}

impl Slot {
    /// Base colour and emissive hold colours, so are sRGB-encoded; the rest are data.
    pub fn srgb(self) -> bool {
        matches!(self, Self::BaseColor | Self::Emissive)
    }
}

pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub mask: bool,
    pub blend: bool,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// An index into [`ImportedModel::images`] for each slot that has a texture.
    pub textures: Vec<(Slot, usize)>,
}

impl Material {
    /// White, fully rough-and-metal by glTF's defaults, untextured, and opaque.
    pub fn plain(name: &str) -> Self {
        Self {
            name: name.into(),
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            mask: false,
            blend: false,
            alpha_cutoff: 0.5,
            double_sided: false,
            textures: Vec::new(),
        }
    }
}

/// A primitive placed in the world by the node that shows it.
pub struct Instance {
    pub name: String,
    pub primitive: usize,
    pub transform: Mat4,
}

pub struct ImportedModel {
    pub name: String,
    pub primitives: Vec<Primitive>,
    pub materials: Vec<Material>,
    /// `None` for an image no material uses, which is not decoded.
    pub images: Vec<Option<Image>>,
    pub instances: Vec<Instance>,
    pub warnings: Vec<String>,
}

pub fn load(path: &Path) -> Result<ImportedModel, String> {
    let fail = |e: gltf::Error| format!("{}: {e}", path.display());
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let gltf::Gltf { document, blob } = gltf::Gltf::open(path).map_err(fail)?;
    let buffers = gltf::import_buffers(&document, Some(base), blob).map_err(fail)?;
    let name = path.file_stem().map_or("model".into(), |s| s.to_string_lossy().to_string());
    build(name, &document, &buffers, Some(base))
}

fn build(
    name: String,
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    base: Option<&Path>,
) -> Result<ImportedModel, String> {
    let mut warnings = Vec::new();
    for ext in doc.extensions_used() {
        if ext == "KHR_texture_transform" {
            warnings.push("KHR_texture_transform is ignored: textures may be offset or scaled wrongly".into());
        }
    }

    // Materials, then the images they name. Only those are read, and one that cannot be
    // is dropped from its material with a warning rather than failing the file.
    let image_count = doc.images().len();
    let mut used_images = HashSet::new();
    let mut materials: Vec<Material> =
        doc.materials().map(|m| material(&m, image_count, &mut used_images, &mut warnings)).collect();
    let default_material = materials.len();
    materials.push(Material::plain("Default"));
    let mut images: Vec<Option<Image>> = Vec::with_capacity(image_count);
    for (i, image) in doc.images().enumerate() {
        images.push(if used_images.contains(&i) {
            decode_image(image, base, buffers)
                .map_err(|e| warnings.push(format!("image {i} could not be read: {e}")))
                .ok()
        } else {
            None
        });
    }
    for m in &mut materials {
        m.textures.retain(|&(_, i)| images.get(i).is_some_and(Option::is_some));
    }

    // Primitives, one list per mesh so nodes can share them.
    let mut primitives = Vec::new();
    let mut mesh_primitives: Vec<Vec<usize>> = Vec::new();
    let mut unpaintable = 0;
    for mesh in doc.meshes() {
        let mut list = Vec::new();
        for (n, prim) in mesh.primitives().enumerate() {
            let label = format!("{}#{n}", mesh.name().map_or_else(|| format!("mesh{}", mesh.index()), str::to_string));
            match primitive(&prim, buffers, label, prim.material().index().unwrap_or(default_material)) {
                Ok(p) => {
                    unpaintable += usize::from(p.uvs.is_none());
                    list.push(primitives.len());
                    primitives.push(p);
                }
                Err(why) => warnings.push(why),
            }
        }
        mesh_primitives.push(list);
    }
    if unpaintable > 0 {
        warnings.push(format!("{unpaintable} primitive(s) have no UVs, so cannot be painted"));
    }
    // Exporters often write every material as BLEND whatever it is. One whose alpha is
    // solid everywhere is opaque, and drawn as blending it would only lose its shadow and
    // let its inside show through.
    let mut demoted = 0;
    for (index, m) in materials.iter_mut().enumerate() {
        if !m.blend {
            continue;
        }
        let texture_solid = m
            .textures
            .iter()
            .find(|(slot, _)| *slot == Slot::BaseColor)
            .and_then(|&(_, i)| images[i].as_ref())
            .is_none_or(|img| img.rgba.chunks_exact(4).all(|px| px[3] >= 253));
        let vertices_solid = primitives
            .iter()
            .filter(|p| p.material == index)
            .all(|p| p.colors.as_ref().is_none_or(|c| c.iter().all(|c| c[3] >= 0.99)));
        if m.base_color[3] >= 0.999 && texture_solid && vertices_solid {
            m.blend = false;
            demoted += 1;
        }
    }
    if demoted > 0 {
        warnings.push(format!("{demoted} material(s) are marked BLEND but are fully opaque, so they are drawn opaque"));
    }
    for p in &primitives {
        let share = wound_against_normals(p);
        if share > 0.5 {
            warnings.push(format!(
                "{}: {:.0}% of its triangles are wound against their normals, so it is inside out wherever back faces are culled",
                p.name,
                share * 100.0
            ));
        }
    }

    // Nodes.
    let mut instances = Vec::new();
    let mut skinned = false;
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    match scene {
        Some(scene) => {
            for node in scene.nodes() {
                walk(&node, Mat4::IDENTITY, &mesh_primitives, &primitives, &mut instances, &mut skinned);
            }
        }
        None => {
            for &p in mesh_primitives.iter().flatten() {
                instances.push(Instance { name: primitives[p].name.clone(), primitive: p, transform: Mat4::IDENTITY });
            }
        }
    }
    if skinned {
        warnings.push("skinned meshes are drawn in their rest pose".into());
    }
    if instances.is_empty() {
        return Err("the file has no triangle meshes in its scene".into());
    }
    Ok(ImportedModel { name, primitives, materials, images, instances, warnings })
}

fn walk(
    node: &gltf::Node,
    parent: Mat4,
    mesh_primitives: &[Vec<usize>],
    primitives: &[Primitive],
    out: &mut Vec<Instance>,
    skinned: &mut bool,
) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        *skinned |= node.skin().is_some();
        for &p in &mesh_primitives[mesh.index()] {
            let base = node.name().or(mesh.name()).unwrap_or("node");
            let name = if mesh_primitives[mesh.index()].len() > 1 {
                primitives[p].name.clone()
            } else {
                base.to_string()
            };
            out.push(Instance { name, primitive: p, transform: world });
        }
    }
    for child in node.children() {
        walk(&child, world, mesh_primitives, primitives, out, skinned);
    }
}

fn material(m: &gltf::Material, image_count: usize, used: &mut HashSet<usize>, warnings: &mut Vec<String>) -> Material {
    use gltf::material::AlphaMode;
    let pbr = m.pbr_metallic_roughness();
    let name = m.name().map_or_else(|| format!("material{}", m.index().unwrap_or(0)), str::to_string);
    let mut textures = Vec::new();
    let mut add = |slot: Slot, tex: gltf::Texture, tex_coord: u32| {
        if tex_coord != 0 {
            warnings.push(format!("{name}: {slot:?} uses UV set {tex_coord}, which is not supported"));
        }
        let Some(image) = image_index(&tex, image_count) else {
            warnings.push(format!("{name}: a {slot:?} texture names no image that exists"));
            return;
        };
        used.insert(image);
        textures.push((slot, image));
    };
    if let Some(t) = pbr.base_color_texture() {
        add(Slot::BaseColor, t.texture(), t.tex_coord());
    }
    if let Some(t) = pbr.metallic_roughness_texture() {
        add(Slot::MetallicRoughness, t.texture(), t.tex_coord());
    }
    if let Some(t) = m.normal_texture() {
        add(Slot::Normal, t.texture(), t.tex_coord());
    }
    if let Some(t) = m.occlusion_texture() {
        add(Slot::Occlusion, t.texture(), t.tex_coord());
    }
    if let Some(t) = m.emissive_texture() {
        add(Slot::Emissive, t.texture(), t.tex_coord());
    }
    let strength = m.emissive_strength().unwrap_or(1.0);
    Material {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: m.emissive_factor().map(|c| c * strength),
        normal_scale: m.normal_texture().map_or(1.0, |t| t.scale()),
        occlusion_strength: m.occlusion_texture().map_or(1.0, |t| t.strength()),
        mask: m.alpha_mode() == AlphaMode::Mask,
        blend: m.alpha_mode() == AlphaMode::Blend,
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided(),
        textures,
        name,
    }
}

/// The image a texture shows. A texture with `KHR_texture_basisu` names its KTX2 image
/// in the extension, and may have no ordinary `source` at all; the extension is preferred
/// over a fallback when both are there.
fn image_index(tex: &gltf::Texture, image_count: usize) -> Option<usize> {
    let basisu = tex
        .extension_value("KHR_texture_basisu")
        .and_then(|v| v.get("source"))
        .and_then(|s| s.as_u64())
        .map(|s| s as usize);
    basisu
        .filter(|&i| i < image_count)
        .or_else(|| tex.source().map(|i| i.index()).filter(|&i| i < image_count))
}

/// An image file's contents, decoded to eight-bit RGBA: KTX2 by its magic number, which
/// is more trustworthy than a mime type, anything else by the `image` crate.
fn decode_image(image: gltf::Image, base: Option<&Path>, buffers: &[gltf::buffer::Data]) -> Result<Image, String> {
    use std::borrow::Cow;
    let bytes: Cow<[u8]> = match image.source() {
        gltf::image::Source::View { view, .. } => {
            let data = &buffers[view.buffer().index()].0;
            let range = view.offset()..view.offset() + view.length();
            Cow::Borrowed(data.get(range).ok_or("its buffer view runs past the end of the buffer")?)
        }
        gltf::image::Source::Uri { uri, .. } => Cow::Owned(read_uri(uri, base)?),
    };
    if crate::ktx::is_ktx2(&bytes) {
        return crate::ktx::decode(&bytes);
    }
    let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?.to_rgba8();
    Ok(Image { width: img.width(), height: img.height(), rgba: img.into_raw() })
}

/// What a glTF URI points at: a `data:` URI, or a file beside the glTF.
fn read_uri(uri: &str, base: Option<&Path>) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    if let Some(rest) = uri.strip_prefix("data:") {
        let (head, payload) = rest.split_once(',').ok_or("a malformed data URI")?;
        return if head.ends_with(";base64") {
            base64::engine::general_purpose::STANDARD.decode(payload).map_err(|e| format!("bad base64: {e}"))
        } else {
            Ok(urlencoding::decode_binary(payload.as_bytes()).into_owned())
        };
    }
    if uri.contains("://") {
        return Err(format!("{uri} is a remote file, which is not fetched"));
    }
    let name = urlencoding::decode(uri).map_err(|e| e.to_string())?;
    let path = base.map_or_else(|| Path::new(name.as_ref()).to_path_buf(), |b| b.join(name.as_ref()));
    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The share of a primitive's triangles wound against their own vertex normals. A
/// well-made mesh has none; one with most is inside out, and looks it wherever back faces
/// are culled. Measured on the mesh itself: a node that mirrors it reverses the winding
/// and the normals together, so it makes no difference.
pub fn wound_against_normals(prim: &Primitive) -> f32 {
    let (mut against, mut total) = (0usize, 0usize);
    for tri in prim.indices.chunks(3) {
        let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(prim.positions[tri[k] as usize]));
        let face = (b - a).cross(c - a);
        let normal: Vec3 = [0, 1, 2].iter().map(|&k| Vec3::from(prim.normals[tri[k] as usize])).sum();
        if face.length_squared() < 1e-24 || normal.length_squared() < 1e-12 {
            continue;
        }
        total += 1;
        against += usize::from(face.dot(normal) < 0.0);
    }
    if total == 0 { 0.0 } else { against as f32 / total as f32 }
}

fn primitive(
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    name: String,
    material: usize,
) -> Result<Primitive, String> {
    let reader = prim.reader(|b| Some(&buffers[b.index()].0[..]));
    let positions: Vec<[f32; 3]> = reader.read_positions().ok_or_else(|| format!("{name}: no positions"))?.collect();
    let n = positions.len();
    let sequential = || (0..n as u32).collect::<Vec<u32>>();
    let raw: Vec<u32> = reader.read_indices().map_or_else(sequential, |i| i.into_u32().collect());
    let indices = to_triangles(prim.mode(), raw).ok_or_else(|| format!("{name}: {:?} primitives are not drawn", prim.mode()))?;
    if indices.len() % 3 != 0 || indices.is_empty() {
        return Err(format!("{name}: {} indices do not make triangles", indices.len()));
    }
    if let Some(&bad) = indices.iter().find(|&&i| i as usize >= n) {
        return Err(format!("{name}: index {bad} is past the {n} vertices"));
    }
    let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().collect());
    let colors: Option<Vec<[f32; 4]>> = reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|v| v.collect());

    let mut p = Primitive {
        name,
        positions,
        normals: normals.clone().unwrap_or_default(),
        uvs,
        tangents,
        colors,
        indices,
        material,
    };
    // Without normals glTF asks for flat shading, which needs a vertex per corner.
    if normals.is_none() {
        unweld(&mut p);
        flat_normals(&mut p);
    }
    if p.tangents.is_none() && p.uvs.is_some() {
        p.tangents = Some(generate_tangents(&p.positions, &p.normals, p.uvs.as_ref().unwrap(), &p.indices));
    }
    Ok(p)
}

/// The triangle list a primitive draws, or `None` for points and lines.
fn to_triangles(mode: gltf::mesh::Mode, raw: Vec<u32>) -> Option<Vec<u32>> {
    use gltf::mesh::Mode;
    Some(match mode {
        Mode::Triangles => raw,
        // Every other triangle is flipped to keep the winding.
        Mode::TriangleStrip => raw
            .windows(3)
            .enumerate()
            .flat_map(|(i, w)| if i % 2 == 0 { [w[0], w[1], w[2]] } else { [w[1], w[0], w[2]] })
            .collect(),
        Mode::TriangleFan => raw.windows(2).skip(1).flat_map(|w| [raw[0], w[0], w[1]]).collect(),
        _ => return None,
    })
}

/// Gives every triangle corner a vertex of its own.
fn unweld(p: &mut Primitive) {
    fn pick<T: Copy>(v: &[T], idx: &[u32]) -> Vec<T> {
        idx.iter().map(|&i| v[i as usize]).collect()
    }
    let idx = std::mem::take(&mut p.indices);
    p.positions = pick(&p.positions, &idx);
    p.uvs = p.uvs.take().map(|v| pick(&v, &idx));
    p.colors = p.colors.take().map(|v| pick(&v, &idx));
    p.tangents = p.tangents.take().map(|v| pick(&v, &idx));
    p.indices = (0..idx.len() as u32).collect();
}

/// Face normals for an unwelded primitive.
fn flat_normals(p: &mut Primitive) {
    p.normals = vec![[0.0, 1.0, 0.0]; p.positions.len()];
    for tri in p.indices.chunks(3) {
        let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(p.positions[tri[k] as usize]));
        let n = (b - a).cross(c - a).normalize_or(Vec3::Y);
        for &i in tri {
            p.normals[i as usize] = n.to_array();
        }
    }
}

/// Tangents from the UVs, accumulated over the triangles at each vertex (Lengyel's
/// method; close to, but not, MikkTSpace). The fourth component is the bitangent's sign,
/// glTF's convention: the bitangent is `cross(normal, tangent) * w`, and points toward
/// decreasing `v`, which is up the image, because the UV origin is the top left.
pub fn generate_tangents(positions: &[[f32; 3]], normals: &[[f32; 3]], uvs: &[[f32; 2]], indices: &[u32]) -> Vec<[f32; 4]> {
    let n = positions.len();
    let (mut tan, mut bit) = (vec![Vec3::ZERO; n], vec![Vec3::ZERO; n]);
    for tri in indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let (p0, p1, p2) = (Vec3::from(positions[i0]), Vec3::from(positions[i1]), Vec3::from(positions[i2]));
        let (e1, e2) = (p1 - p0, p2 - p0);
        // v is flipped so that "up the image" is positive.
        let d = |a: usize, b: usize| [uvs[b][0] - uvs[a][0], -(uvs[b][1] - uvs[a][1])];
        let (d1, d2) = (d(i0, i1), d(i0, i2));
        let det = d1[0] * d2[1] - d2[0] * d1[1];
        if det.abs() < 1e-20 {
            continue;
        }
        let r = 1.0 / det;
        let t = (e1 * d2[1] - e2 * d1[1]) * r;
        let b = (e2 * d1[0] - e1 * d2[0]) * r;
        for i in [i0, i1, i2] {
            tan[i] += t;
            bit[i] += b;
        }
    }
    (0..n)
        .map(|i| {
            let nrm = Vec3::from(normals[i]);
            let t = tan[i] - nrm * nrm.dot(tan[i]);
            // A vertex whose triangles have no usable UVs gets no tangent, which the
            // shader reads as "leave the normal alone".
            if t.length_squared() < 1e-16 {
                return [0.0; 4];
            }
            let t = t.normalize();
            let w = if nrm.cross(t).dot(bit[i]) < 0.0 { -1.0 } else { 1.0 };
            [t.x, t.y, t.z, w]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-triangle .gltf with its buffer inline, no normals, on a node moved along x.
    fn triangle_gltf(with_uvs: bool) -> String {
        // Positions (0,0,0) (1,0,0) (0,1,0) then UVs (0,1) (1,1) (0,0), as f32 little endian.
        let mut bytes: Vec<u8> = Vec::new();
        for f in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        for f in [0.0f32, 1.0, 1.0, 1.0, 0.0, 0.0] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let b64 = base64(&bytes);
        let (uv_attr, uv_view, uv_acc) = if with_uvs {
            (
                r#","TEXCOORD_0":1"#,
                r#",{"buffer":0,"byteOffset":36,"byteLength":24}"#,
                r#",{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}"#,
            )
        } else {
            ("", "", "")
        };
        format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
            "nodes":[{{"mesh":0,"translation":[5,0,0],"name":"tri"}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0{uv_attr}}}}}]}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}}{uv_acc}],
            "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}}{uv_view}],
            "buffers":[{{"byteLength":60,"uri":"data:application/octet-stream;base64,{b64}"}}]}}"#
        )
    }

    fn base64(bytes: &[u8]) -> String {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in bytes.chunks(3) {
            let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            for i in 0..4 {
                if i <= c.len() {
                    out.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    fn import(json: &str) -> Result<ImportedModel, String> {
        // Tests run side by side, so each gets a folder of its own.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("arbor-paint-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tri.gltf");
        std::fs::write(&path, json).unwrap();
        load(&path)
    }

    #[test]
    fn a_triangle_without_normals_is_flat_shaded_and_placed() {
        let m = import(&triangle_gltf(true)).unwrap();
        assert_eq!(m.instances.len(), 1);
        assert_eq!(m.instances[0].name, "tri");
        assert_eq!(m.instances[0].transform.w_axis.truncate(), Vec3::new(5.0, 0.0, 0.0));
        let p = &m.primitives[0];
        assert_eq!(p.indices, vec![0, 1, 2]);
        // Counter-clockwise seen from +z, so the face normal is +z.
        assert_eq!(p.normals[0], [0.0, 0.0, 1.0]);
        // No material in the file: the default one is used.
        assert_eq!(m.materials[p.material].name, "Default");
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
    }

    #[test]
    fn tangents_follow_u_and_the_bitangent_points_up_the_image() {
        let m = import(&triangle_gltf(true)).unwrap();
        let t = m.primitives[0].tangents.as_ref().unwrap()[0];
        // u grows along +x. v grows down the image, which is toward -y, so the bitangent
        // (decreasing v) is +y, and cross(+z, +x) = +y agrees: w is +1.
        assert!((Vec3::new(t[0], t[1], t[2]) - Vec3::X).length() < 1e-5, "{t:?}");
        assert_eq!(t[3], 1.0);
    }

    #[test]
    fn a_mesh_without_uvs_is_reported_as_unpaintable() {
        let m = import(&triangle_gltf(false)).unwrap();
        assert!(m.primitives[0].uvs.is_none());
        assert!(m.warnings.iter().any(|w| w.contains("no UVs")), "{:?}", m.warnings);
    }

    #[test]
    fn a_file_with_no_meshes_is_an_error() {
        let json = r#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{}]}"#;
        assert!(import(json).is_err());
    }

    #[test]
    fn strips_and_fans_become_triangles() {
        use gltf::mesh::Mode;
        assert_eq!(to_triangles(Mode::TriangleStrip, vec![0, 1, 2, 3]), Some(vec![0, 1, 2, 2, 1, 3]));
        assert_eq!(to_triangles(Mode::TriangleFan, vec![0, 1, 2, 3]), Some(vec![0, 1, 2, 0, 2, 3]));
        assert_eq!(to_triangles(Mode::Lines, vec![0, 1]), None);
    }

    #[test]
    fn a_basisu_texture_with_no_source_is_found_through_its_extension() {
        let ktx = base64(include_bytes!("../tests/fixtures/uastc.ktx2"));
        let mut bytes: Vec<u8> = Vec::new();
        for f in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
            "extensionsUsed":["KHR_texture_basisu"],"extensionsRequired":["KHR_texture_basisu"],
            "nodes":[{{"mesh":0}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1}},"material":0}}]}}],
            "materials":[{{"name":"ktx","pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],
            "textures":[{{"extensions":{{"KHR_texture_basisu":{{"source":0}}}}}}],
            "images":[{{"mimeType":"image/ktx2","uri":"data:image/ktx2;base64,{ktx}"}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
                         {{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}}],
            "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":24}}],
            "buffers":[{{"byteLength":60,"uri":"data:application/octet-stream;base64,{}"}}]}}"#,
            base64(&bytes)
        );
        let m = import(&json).unwrap();
        assert!(m.warnings.is_empty(), "{:?}", m.warnings);
        let mat = &m.materials[m.primitives[0].material];
        assert_eq!(mat.name, "ktx");
        let &(slot, image) = mat.textures.first().expect("the KTX2 texture should be on the material");
        assert_eq!(slot, Slot::BaseColor);
        let img = m.images[image].as_ref().expect("and decoded");
        assert_eq!((img.width, img.height), (64, 64));
    }

    #[test]
    fn an_image_that_cannot_be_read_is_dropped_with_a_warning() {
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
            "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}],
            "textures":[{"source":0}],"images":[{"uri":"data:image/png;base64,AAAA"}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
            "bufferViews":[{"buffer":0,"byteLength":36}],
            "buffers":[{"byteLength":36,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"}]}"#;
        let m = import(json).unwrap();
        assert!(m.materials[0].textures.is_empty());
        assert!(m.warnings.iter().any(|w| w.contains("image 0 could not be read")), "{:?}", m.warnings);
    }

    /// `ARBOR_WINDING=model.glb cargo test winding -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn winding_of_a_real_model() {
        let Ok(path) = std::env::var("ARBOR_WINDING") else { return };
        let m = load(std::path::Path::new(&path)).unwrap();
        for inst in &m.instances {
            let share = wound_against_normals(&m.primitives[inst.primitive]);
            let det = inst.transform.determinant();
            println!("{:<28} against normals: {:>5.1}%  det {det:+.3}  double-sided: {}", inst.name, share * 100.0, m.materials[m.primitives[inst.primitive].material].double_sided);
        }
    }

    #[test]
    fn an_inside_out_mesh_is_reported_and_a_mirrored_node_is_not() {
        // The triangle from `triangle_gltf` faces +z; flipping two indices turns it
        // against its (generated) normals only if the normals were given, so build one with
        // explicit normals both ways.
        let tri = |indices: [u32; 3]| Primitive {
            name: "t".into(),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            uvs: None,
            tangents: None,
            colors: None,
            indices: indices.to_vec(),
            material: 0,
        };
        assert_eq!(wound_against_normals(&tri([0, 1, 2])), 0.0);
        assert_eq!(wound_against_normals(&tri([0, 2, 1])), 1.0);
    }
}
