//! Small frame helpers shared by growth and meshing. Both need to carry a reference
//! vector along a curving stem, and both need it to stay perpendicular to the stem.

use glam::{Quat, Vec3};

pub fn norm_or_zero(v: Vec3) -> Vec3 {
    let len = v.length();
    if len > 1e-8 { v / len } else { Vec3::ZERO }
}

/// An arbitrary but stable unit vector perpendicular to `d`.
pub fn ortho_of(d: Vec3) -> Vec3 {
    let reference = if d.dot(Vec3::Y).abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let c = d.cross(reference);
    if c.length_squared() > 1e-12 {
        c.normalize()
    } else {
        Vec3::X
    }
}

/// A unit vector perpendicular to `axis`, as close to `v` as possible.
pub fn ortho_unit(v: Vec3, axis: Vec3) -> Vec3 {
    let projected = v - axis * v.dot(axis);
    if projected.length_squared() > 1e-12 {
        projected.normalize()
    } else {
        ortho_of(axis)
    }
}

/// Rotates `v` by the shortest rotation taking `from` onto `to`. Carrying a frame
/// this way keeps it from spinning about the stem axis as the stem curves, which is
/// what stops ring vertices from twisting between segments.
pub fn transport(from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
    let axis = from.cross(to);
    let s = axis.length();
    if s < 1e-6 {
        return if from.dot(to) < 0.0 { -v } else { v };
    }
    let angle = from.dot(to).clamp(-1.0, 1.0).acos();
    Quat::from_axis_angle(axis / s, angle) * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_of_is_perpendicular_and_unit() {
        for d in [Vec3::Y, Vec3::X, Vec3::new(0.3, 0.9, -0.2).normalize(), -Vec3::Y] {
            let o = ortho_of(d);
            assert!((o.length() - 1.0).abs() < 1e-5, "not unit for {d:?}");
            assert!(o.dot(d).abs() < 1e-5, "not perpendicular for {d:?}");
        }
    }

    #[test]
    fn ortho_unit_falls_back_when_degenerate() {
        let d = Vec3::Y;
        let o = ortho_unit(Vec3::Y * 3.0, d);
        assert!((o.length() - 1.0).abs() < 1e-5);
        assert!(o.dot(d).abs() < 1e-5);
    }

    #[test]
    fn transport_keeps_the_frame_perpendicular() {
        let from = Vec3::Y;
        let to = Vec3::new(0.4, 1.0, 0.1).normalize();
        let v = ortho_of(from);
        let moved = transport(from, to, v);
        assert!((moved.length() - 1.0).abs() < 1e-5);
        assert!(moved.dot(to).abs() < 1e-5);
    }

    #[test]
    fn transport_is_identity_for_equal_directions() {
        let d = Vec3::new(0.2, 0.9, 0.3).normalize();
        let v = ortho_of(d);
        assert!((transport(d, d, v) - v).length() < 1e-6);
    }
}
