//! Built-in shapes, for when no model is loaded.

use arbor_render::pbr::PbrMeshData;

/// Owned vertex data, lent out as [`PbrMeshData`].
pub struct Shape {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl Shape {
    pub fn data(&self) -> PbrMeshData<'_> {
        PbrMeshData {
            positions: &self.positions,
            normals: &self.normals,
            uvs: Some(&self.uvs),
            tangents: Some(&self.tangents),
            colors: None,
            indices: &self.indices,
        }
    }
}

/// A unit sphere, `segments` rings by twice as many columns, with one UV set wrapped once
/// round it.
pub fn sphere(segments: u32) -> Shape {
    let mut s = Shape { positions: vec![], normals: vec![], uvs: vec![], tangents: vec![], indices: vec![] };
    for j in 0..=segments {
        let v = j as f32 / segments as f32;
        let phi = v * std::f32::consts::PI;
        for i in 0..=segments * 2 {
            let u = i as f32 / (segments * 2) as f32;
            let theta = u * std::f32::consts::TAU;
            let d = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            s.positions.push(d);
            s.normals.push(d);
            s.uvs.push([u, v]);
            s.tangents.push([-theta.sin(), 0.0, theta.cos(), -1.0]);
        }
    }
    let row = segments * 2 + 1;
    for j in 0..segments {
        for i in 0..segments * 2 {
            let a = j * row + i;
            s.indices.extend_from_slice(&[a, a + 1, a + row, a + 1, a + row + 1, a + row]);
        }
    }
    s
}

/// A flat square of side 2 on the ground, facing up.
pub fn floor() -> Shape {
    Shape {
        positions: vec![[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        normals: vec![[0.0, 1.0, 0.0]; 4],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        tangents: vec![[1.0, 0.0, 0.0, -1.0]; 4],
        indices: vec![0, 2, 1, 0, 3, 2],
    }
}
