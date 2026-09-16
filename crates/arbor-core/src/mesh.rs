use glam::Vec3;
use std::f32::consts::TAU;

use crate::math::{norm_or_zero, ortho_of, ortho_unit, transport};
use crate::skeleton::Skeleton;
use crate::species::{MeshParams, SpeciesParams};

/// Angular step used for the finite-difference normal around a ring.
const NORMAL_DA: f32 = 0.01;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &self.positions {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        (min, max)
    }
}

fn seed_phases(seed: u64) -> (f32, f32) {
    let mut z = seed ^ 0x9E3779B97F4A7C15;
    let step = |z: &mut u64| {
        *z = z.wrapping_add(0x9E3779B97F4A7C15);
        let mut x = *z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    };
    let a = step(&mut z);
    let b = step(&mut z);
    (
        (a as f32 / u64::MAX as f32) * TAU,
        (b as f32 / u64::MAX as f32) * TAU,
    )
}

struct MeshSink {
    mesh: Mesh,
}

impl MeshSink {
    fn push_vertex(&mut self, pos: Vec3, normal: Vec3, tangent: Vec3, uv: [f32; 2]) {
        self.mesh.positions.push(pos.to_array());
        self.mesh.normals.push(normal.to_array());
        let t = norm_or_zero(tangent - normal * tangent.dot(normal));
        self.mesh.tangents.push([t.x, t.y, t.z, 1.0]);
        self.mesh.uvs.push(uv);
    }

    fn vertex_offset(&self) -> u32 {
        self.mesh.positions.len() as u32
    }
}

/// One stem laid out as a chain of ring centres, ready to be swept.
struct StemPath {
    /// Ring centres. For every stem but the trunk, the first entry sits on the parent
    /// centreline so the tube starts inside the parent instead of floating beside it.
    points: Vec<Vec3>,
    /// Axis direction at each ring, from a centred difference.
    dirs: Vec<Vec3>,
    /// Skeleton radius at each ring, before socket and root flare.
    radii: Vec<f32>,
    /// Extra widening at the foot of a branch so the junction reads as a socket.
    socket: Vec<f32>,
    /// Distance travelled along the stem, used for the V coordinate.
    arc: Vec<f32>,
    /// True for the stem that starts at the root node, which gets the buttress flare
    /// and the ground cap.
    is_trunk: bool,
}

impl StemPath {
    fn build(sk: &Skeleton, stem: &[u32], mp: &MeshParams) -> Option<StemPath> {
        let first = *stem.first()? as usize;
        let is_trunk = sk.nodes[first].parent.is_none();

        let mut points = Vec::with_capacity(stem.len() + 1);
        let mut radii = Vec::with_capacity(stem.len() + 1);
        if let Some(p) = sk.nodes[first].parent {
            // Anchor the stem on its parent so the two tubes overlap at the junction.
            let anchor = sk.nodes[p as usize].position;
            if (anchor - sk.nodes[first].position).length() > 1e-5 {
                points.push(anchor);
                radii.push(sk.nodes[first].radius);
            }
        }
        for &i in stem {
            let node = &sk.nodes[i as usize];
            // Skip duplicate positions: a zero-length segment gives no usable axis.
            if points
                .last()
                .is_some_and(|p: &Vec3| (*p - node.position).length() < 1e-5)
            {
                continue;
            }
            points.push(node.position);
            radii.push(node.radius);
        }
        if points.len() < 2 {
            return None;
        }

        let last = points.len() - 1;
        let mut dirs: Vec<Vec3> = Vec::with_capacity(points.len());
        for i in 0..points.len() {
            let prev = if i == 0 {
                points[0] * 2.0 - points[1]
            } else {
                points[i - 1]
            };
            let next = if i < last {
                points[i + 1]
            } else {
                points[i] * 2.0 - prev
            };
            let d = norm_or_zero(next - prev);
            dirs.push(if d == Vec3::ZERO {
                *dirs.last().unwrap_or(&Vec3::Y)
            } else {
                d
            });
        }

        let mut arc = Vec::with_capacity(points.len());
        let mut travelled = 0.0;
        for i in 0..points.len() {
            if i > 0 {
                travelled += (points[i] - points[i - 1]).length();
            }
            arc.push(travelled);
        }

        // The socket fades out over a few base radii, so the flare is sized by the
        // branch rather than by however finely the stem happens to be segmented.
        let socket_len = (radii[0] * 3.0).max(1e-4);
        let socket = arc
            .iter()
            .map(|&s| {
                if is_trunk {
                    1.0
                } else {
                    let t = (s / socket_len).clamp(0.0, 1.0);
                    1.0 + mp.socket_flare * (1.0 - t) * (1.0 - t)
                }
            })
            .collect();

        Some(StemPath {
            points,
            dirs,
            radii,
            socket,
            arc,
            is_trunk,
        })
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    /// Surface radius of ring `i` at angle `a`, including socket and root flare.
    fn radius_at(&self, i: usize, a: f32, mp: &MeshParams, phase: (f32, f32)) -> f32 {
        let base = self.radii[i] * self.socket[i];
        let y = self.points[i].y;
        if self.is_trunk && y < mp.flare_height && mp.flare_height > 1e-4 {
            let t = (y / mp.flare_height).clamp(0.0, 1.0);
            let k = (1.0 - t) * (1.0 - t);
            let lobe = 0.7 + 0.3 * (a * 3.0 + phase.0).sin() + 0.2 * (a * 7.0 + phase.1).sin();
            base * (1.0 + mp.root_flare * k * lobe.max(0.2))
        } else {
            base
        }
    }
}

pub fn build_mesh(sk: &Skeleton, params: &SpeciesParams) -> Mesh {
    let mp: &MeshParams = &params.mesh;
    let mut sink = MeshSink {
        mesh: Mesh::default(),
    };
    let phase = seed_phases(params.seed);

    for stem in sk.stem_runs() {
        let Some(path) = StemPath::build(sk, &stem, mp) else {
            continue;
        };
        emit_stem(&mut sink, &path, mp, phase);
    }

    sink.mesh
}

fn emit_stem(sink: &mut MeshSink, path: &StemPath, mp: &MeshParams, phase: (f32, f32)) {
    let last = path.len() - 1;
    // Resolution follows the thickest ring so a tapering stem keeps its silhouette
    // all the way down instead of being sized by its average.
    let r_max = path.radii.iter().copied().fold(0.0f32, f32::max) * path.socket[0];
    let radial = ((mp.radial_per_meter * r_max).round() as i32)
        .clamp(mp.min_radial.max(3) as i32, mp.max_radial.max(3) as i32) as u32;

    let mut ring_bases: Vec<u32> = Vec::with_capacity(path.len());
    let mut n = ortho_of(path.dirs[0]);
    // Ring 0 frame, kept so the ground cap can be stitched from the same vertices.
    let mut first_frame = (n, path.dirs[0].cross(n));

    for i in 0..path.len() {
        let d = path.dirs[i];
        if i > 0 {
            n = transport(path.dirs[i - 1], d, n);
        }
        n = ortho_unit(n, d);
        let b = d.cross(n);
        if i == 0 {
            first_frame = (n, b);
        }

        // Neighbours used for the axial slope of the radius, so the normal follows
        // taper and flare instead of pointing straight out of the axis.
        let i_prev = i.saturating_sub(1);
        let i_next = (i + 1).min(last);
        let ds = path.arc[i_next] - path.arc[i_prev];

        let base = sink.vertex_offset();
        for j in 0..=radial {
            let a = j as f32 / radial as f32 * TAU;
            let e_r = n * a.cos() + b * a.sin();
            let e_a = b * a.cos() - n * a.sin();
            let r = path.radius_at(i, a, mp, phase);

            let dr_da = (path.radius_at(i, a + NORMAL_DA, mp, phase)
                - path.radius_at(i, a - NORMAL_DA, mp, phase))
                / (2.0 * NORMAL_DA);
            let dr_ds = if ds > 1e-6 {
                (path.radius_at(i_next, a, mp, phase) - path.radius_at(i_prev, a, mp, phase)) / ds
            } else {
                0.0
            };
            let normal = norm_or_zero(e_r - e_a * (dr_da / r.max(1e-5)) - d * dr_ds);
            let normal = if normal == Vec3::ZERO { e_r } else { normal };

            let uv = [
                j as f32 / radial as f32 * path.radii[i] * TAU / mp.uv_scale.max(1e-4),
                path.arc[i] / mp.uv_scale.max(1e-4),
            ];
            sink.push_vertex(path.points[i] + e_r * r, normal, e_a, uv);
        }
        ring_bases.push(base);
    }

    for w in 0..ring_bases.len() - 1 {
        let bi = ring_bases[w];
        let bn = ring_bases[w + 1];
        for j in 0..radial {
            let a0 = bi + j;
            let a1 = bi + j + 1;
            let b0 = bn + j;
            let b1 = bn + j + 1;
            sink.mesh
                .indices
                .extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
        }
    }

    if path.is_trunk {
        // Only the trunk meets the ground, so it is the only stem that needs a disc.
        // The rim is duplicated with the cap normal to keep the edge crisp.
        let cap_n = -path.dirs[0];
        let (n0, b0) = first_frame;
        let rim = sink.vertex_offset();
        for j in 0..=radial {
            let a = j as f32 / radial as f32 * TAU;
            let e_r = n0 * a.cos() + b0 * a.sin();
            let pos = path.points[0] + e_r * path.radius_at(0, a, mp, phase);
            sink.push_vertex(pos, cap_n, ortho_of(cap_n), [0.0, 0.0]);
        }
        let center = sink.vertex_offset();
        sink.push_vertex(path.points[0], cap_n, ortho_of(cap_n), [0.0, 0.0]);
        for j in 0..radial {
            sink.mesh
                .indices
                .extend_from_slice(&[center, rim + j + 1, rim + j]);
        }
    }

    // Every stem closes with a short cone. Interior ends are usually buried under
    // their own children; visible ends get a tapered twig tip rather than a disc.
    let end_dir = path.dirs[last];
    let r_end = path.radii[last] * path.socket[last];
    let tip = path.points[last] + end_dir * mp.tip_length.max(r_end * 1.2);
    let apex = sink.vertex_offset();
    sink.push_vertex(tip, end_dir, ortho_of(end_dir), [0.0, path.arc[last]]);
    let base = ring_bases[last];
    for j in 0..radial {
        sink.mesh
            .indices
            .extend_from_slice(&[base + j, base + j + 1, apex]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::species::{parse_species, PINE_RON};

    fn mesh_aabb_width_at_height(mesh: &Mesh, max_y: f32) -> f32 {
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for (p, n) in mesh.positions.iter().zip(mesh.normals.iter()) {
            if p[1] <= max_y && n[1].abs() < 0.9 {
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
                min_z = min_z.min(p[2]);
                max_z = max_z.max(p[2]);
            }
        }
        (max_x - min_x).max(max_z - min_z)
    }

    #[test]
    fn mesh_is_deterministic() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let a = build_mesh(&sk, &params);
        let b = build_mesh(&sk, &params);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_meshes() {
        let mut a = parse_species(PINE_RON).unwrap();
        let mut b = parse_species(PINE_RON).unwrap();
        a.seed = 10;
        b.seed = 11;
        let ma = build_mesh(&crate::grow(&a), &a);
        let mb = build_mesh(&crate::grow(&b), &b);
        assert_ne!(ma.positions, mb.positions);
    }

    #[test]
    fn indices_in_range_and_normals_unit() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        let count = mesh.vertex_count() as u32;
        assert!(count > 100, "mesh too small: {count}");
        assert!(mesh.triangle_count() > 1000, "expected a substantial mesh");
        for &i in &mesh.indices {
            assert!(i < count, "index {i} out of range {count}");
        }
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 0.02, "normal length {len}");
        }
        for (n, t) in mesh.normals.iter().zip(mesh.tangents.iter()) {
            let dot = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
            assert!(dot.abs() < 0.02, "tangent not orthogonal: dot={dot}");
        }
    }

    #[test]
    fn uvs_are_sane() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        for uv in &mesh.uvs {
            assert!(uv[0].is_finite() && uv[1].is_finite());
            assert!(uv[0] >= -1e-4 && uv[1] >= -1e-4);
        }
    }

    #[test]
    fn root_flare_widens_base() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        let base_width = mesh_aabb_width_at_height(&mesh, 0.25);
        assert!(
            base_width > params.trunk.radius * 2.0 + 0.15,
            "base width {base_width} suggests no flare"
        );
    }

    #[test]
    fn mesh_sits_on_ground() {
        let params = parse_species(PINE_RON).unwrap();
        let sk = crate::grow(&params);
        let mesh = build_mesh(&sk, &params);
        let (min, max) = mesh.aabb();
        assert!(min[1] > -0.3, "mesh dips below ground: {}", min[1]);
        assert!(max[1] > 5.0, "mesh too short: {}", max[1]);
    }
}
