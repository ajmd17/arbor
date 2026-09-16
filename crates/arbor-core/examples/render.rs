// Temporary offline rasterizer used to eyeball generation quality. Writes a BMP.
use arbor_core::species::parse_species;
use arbor_core::{build_mesh, grow, Mesh};
use glam::{Mat4, Vec3, Vec4Swizzles};

const W: usize = 900;
const H: usize = 900;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().cloned().unwrap_or_else(|| "oak".to_string());
    let out = args.get(1).cloned().unwrap_or_else(|| "out.bmp".to_string());
    let yaw: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let zoom: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let src = arbor_core::species::builtin_presets()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s.to_string())
        .unwrap_or_else(|| std::fs::read_to_string(&name).unwrap());
    let params = parse_species(&src).unwrap();
    let sk = grow(&params);
    let mesh = build_mesh(&sk, &params);

    let (min, max) = mesh.aabb();
    let center = Vec3::new((min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5);
    let extent = (Vec3::from(max) - Vec3::from(min)).length().max(1.0);
    let dist = extent * 0.95 / zoom;
    let eye = center + Vec3::new(yaw.sin() * dist, extent * 0.10, yaw.cos() * dist);
    let view = Mat4::look_at_rh(eye, center, Vec3::Y);
    let proj = Mat4::perspective_rh(50f32.to_radians(), 1.0, 0.05, dist * 6.0);
    let vp = proj * view;

    let mut color = vec![[0.10f32, 0.12, 0.16]; W * H];
    let mut depth = vec![f32::INFINITY; W * H];
    let sun = Vec3::new(0.4, 0.8, 0.45).normalize();

    raster(&mesh, vp, eye, sun, &mut color, &mut depth);
    write_bmp(&out, &color);
    println!(
        "{name}: verts={} tris={} -> {out}",
        mesh.vertex_count(),
        mesh.triangle_count()
    );
}

fn raster(mesh: &Mesh, vp: Mat4, eye: Vec3, sun: Vec3, color: &mut [[f32; 3]], depth: &mut [f32]) {
    for tri in mesh.indices.chunks_exact(3) {
        let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let wp: Vec<Vec3> = idx.iter().map(|&i| Vec3::from(mesh.positions[i])).collect();
        let nrm: Vec<Vec3> = idx.iter().map(|&i| Vec3::from(mesh.normals[i])).collect();
        let mut clip = [Vec3::ZERO; 3];
        let mut ok = true;
        for k in 0..3 {
            let c = vp * wp[k].extend(1.0);
            if c.w <= 1e-4 {
                ok = false;
                break;
            }
            let ndc = c.xyz() / c.w;
            clip[k] = Vec3::new(
                (ndc.x * 0.5 + 0.5) * W as f32,
                (1.0 - (ndc.y * 0.5 + 0.5)) * H as f32,
                c.w,
            );
        }
        if !ok {
            continue;
        }
        let min_x = clip.iter().map(|p| p.x).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let max_x = (clip.iter().map(|p| p.x).fold(f32::MIN, f32::max).ceil()).min(W as f32 - 1.0);
        let min_y = clip.iter().map(|p| p.y).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let max_y = (clip.iter().map(|p| p.y).fold(f32::MIN, f32::max).ceil()).min(H as f32 - 1.0);
        if max_x < 0.0 || max_y < 0.0 {
            continue;
        }
        let area = edge(clip[0], clip[1], clip[2]);
        if area.abs() < 1e-7 {
            continue;
        }
        for y in min_y..=(max_y as usize) {
            for x in min_x..=(max_x as usize) {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let w0 = edge(clip[1], clip[2], p) / area;
                let w1 = edge(clip[2], clip[0], p) / area;
                let w2 = edge(clip[0], clip[1], p) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * clip[0].z + w1 * clip[1].z + w2 * clip[2].z;
                let di = y * W + x;
                if z >= depth[di] {
                    continue;
                }
                depth[di] = z;
                let mut n = (nrm[0] * w0 + nrm[1] * w1 + nrm[2] * w2).normalize_or_zero();
                let pos = wp[0] * w0 + wp[1] * w1 + wp[2] * w2;
                let v = (eye - pos).normalize_or_zero();
                if n.dot(v) < 0.0 {
                    n = -n;
                }
                let ndl = n.dot(sun).max(0.0);
                let amb = 0.22 + 0.18 * (n.y * 0.5 + 0.5);
                let base = Vec3::new(0.46, 0.33, 0.22);
                let h = (v + sun).normalize_or_zero();
                let spec = n.dot(h).max(0.0).powf(24.0) * 0.12;
                let c = base * (amb + ndl * 0.95) + Vec3::splat(spec);
                color[di] = [
                    c.x.powf(1.0 / 2.2).min(1.0),
                    c.y.powf(1.0 / 2.2).min(1.0),
                    c.z.powf(1.0 / 2.2).min(1.0),
                ];
            }
        }
    }
}

fn edge(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn write_bmp(path: &str, color: &[[f32; 3]]) {
    let row_bytes = W * 3;
    let pad = (4 - row_bytes % 4) % 4;
    let data_size = (row_bytes + pad) * H;
    let mut out = Vec::with_capacity(54 + data_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&((54 + data_size) as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(W as i32).to_le_bytes());
    out.extend_from_slice(&(H as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    for _ in 0..6 {
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    for y in (0..H).rev() {
        for x in 0..W {
            let c = color[y * W + x];
            out.push((c[2] * 255.0) as u8);
            out.push((c[1] * 255.0) as u8);
            out.push((c[0] * 255.0) as u8);
        }
        out.extend(std::iter::repeat(0u8).take(pad));
    }
    std::fs::write(path, out).unwrap();
}
