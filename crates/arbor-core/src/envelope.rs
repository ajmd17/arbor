use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::ranged::{key, Scalar};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeVolume {
    Ellipsoid {
        center: [f32; 3],
        radii: [f32; 3],
    },
    Cone {
        base_y: f32,
        apex_y: f32,
        base_radius: f32,
        apex_radius: f32,
    },
    Cylinder {
        base_y: f32,
        top_y: f32,
        radius: f32,
    },
}

impl EnvelopeVolume {
    fn normalized_distance(&self, p: Vec3) -> Option<f32> {
        match self {
            EnvelopeVolume::Ellipsoid { center, radii } => {
                let c = Vec3::from(*center);
                let r = Vec3::from(*radii).max(Vec3::splat(1e-4));
                Some(((p - c) / r).length())
            }
            EnvelopeVolume::Cone {
                base_y,
                apex_y,
                base_radius,
                apex_radius,
            } => {
                let span = apex_y - base_y;
                if span.abs() < 1e-4 {
                    return None;
                }
                let t = (p.y - base_y) / span;
                if !(0.0..=1.0).contains(&t) {
                    return None;
                }
                let expected = base_radius + (apex_radius - base_radius) * t;
                let radial = (p.x * p.x + p.z * p.z).sqrt();
                Some(radial / expected.max(1e-3))
            }
            EnvelopeVolume::Cylinder {
                base_y,
                top_y,
                radius,
            } => {
                let span = top_y - base_y;
                if span.abs() < 1e-4 {
                    return None;
                }
                let t = (p.y - base_y) / span;
                if !(0.0..=1.0).contains(&t) {
                    return None;
                }
                let radial = (p.x * p.x + p.z * p.z).sqrt();
                Some(radial / radius.max(1e-3))
            }
        }
    }

    fn steer_target(&self, p: Vec3) -> Option<Vec3> {
        match self {
            EnvelopeVolume::Ellipsoid { center, .. } => Some(Vec3::from(*center)),
            EnvelopeVolume::Cone { base_y, apex_y, .. } => {
                Some(Vec3::new(0.0, p.y.clamp(*base_y, *apex_y), 0.0))
            }
            EnvelopeVolume::Cylinder { base_y, top_y, .. } => {
                Some(Vec3::new(0.0, p.y.clamp(*base_y, *top_y), 0.0))
            }
        }
    }

    /// The volume stretched along the trunk by `y`, its widths untouched.
    fn stretched(&self, y: f32) -> Self {
        let y = y.max(1e-3);
        match self {
            EnvelopeVolume::Ellipsoid { center, radii } => EnvelopeVolume::Ellipsoid {
                center: [center[0], center[1] * y, center[2]],
                radii: [radii[0], radii[1] * y, radii[2]],
            },
            EnvelopeVolume::Cone {
                base_y,
                apex_y,
                base_radius,
                apex_radius,
            } => EnvelopeVolume::Cone {
                base_y: base_y * y,
                apex_y: apex_y * y,
                base_radius: *base_radius,
                apex_radius: *apex_radius,
            },
            EnvelopeVolume::Cylinder {
                base_y,
                top_y,
                radius,
            } => EnvelopeVolume::Cylinder {
                base_y: base_y * y,
                top_y: top_y * y,
                radius: *radius,
            },
        }
    }

    fn scaled(&self, s: f32) -> Self {
        let s = s.max(1e-3);
        match self {
            EnvelopeVolume::Ellipsoid { center, radii } => EnvelopeVolume::Ellipsoid {
                center: [center[0] * s, center[1] * s, center[2] * s],
                radii: [radii[0] * s, radii[1] * s, radii[2] * s],
            },
            EnvelopeVolume::Cone {
                base_y,
                apex_y,
                base_radius,
                apex_radius,
            } => EnvelopeVolume::Cone {
                base_y: base_y * s,
                apex_y: apex_y * s,
                base_radius: base_radius * s,
                apex_radius: (apex_radius * s).max(1e-3),
            },
            EnvelopeVolume::Cylinder {
                base_y,
                top_y,
                radius,
            } => EnvelopeVolume::Cylinder {
                base_y: base_y * s,
                top_y: top_y * s,
                radius: radius * s,
            },
        }
    }
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "EnvelopeParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct EnvelopeParams<V = f32> {
    pub volumes: Vec<EnvelopeVolume>,
    pub falloff: V,
    pub kill_threshold: V,
    /// How hard the crown turns a stem that is on its way out of it, in radians per
    /// metre grown. Per metre rather than per segment, so a level with short segments
    /// is not steered several times harder than one with long segments for the same
    /// setting. Only a stem heading outward is turned at all: a pull that keeps
    /// acting once the stem has come back round is a fixed force toward a fixed
    /// point, which curls the stem into a loop rather than settling it.
    pub pull_strength: V,
    /// The trunk length the volumes were drawn for, in metres. Zero means they are
    /// absolute and stay where they are whatever the trunk does.
    ///
    /// The volumes are authored in metres around a trunk of one particular length.
    /// Declared longer, the trunk climbs straight out of the top of them and every
    /// branch born up there is pruned at birth, so a tree made taller in the viewer
    /// came out as its old crown with a bare spike on top. With this set, the volumes
    /// stretch along the trunk by the ratio of the two lengths and the crown keeps its
    /// place on the tree. Only the heights stretch — a taller tree is not, by that
    /// alone, a wider one — so the width stays with `envelope_scale`.
    pub for_trunk_length: V,
}

impl Default for EnvelopeParams {
    fn default() -> Self {
        Self {
            volumes: vec![EnvelopeVolume::Ellipsoid {
                center: [0.0, 5.0, 0.0],
                radii: [4.0, 4.0, 4.0],
            }],
            falloff: 0.25,
            kill_threshold: 0.03,
            pull_strength: 0.6,
            for_trunk_length: 0.0,
        }
    }
}

impl<V: Scalar> EnvelopeParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        EnvelopeParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> EnvelopeParams<W> {
        EnvelopeParams {
            volumes: self.volumes.clone(),
            falloff: f(&key(at, "falloff"), self.falloff),
            kill_threshold: f(&key(at, "kill_threshold"), self.kill_threshold),
            pull_strength: f(&key(at, "pull_strength"), self.pull_strength),
            for_trunk_length: f(&key(at, "for_trunk_length"), self.for_trunk_length),
        }
    }
}

impl EnvelopeParams {
    pub fn density(&self, p: Vec3) -> f32 {
        let mut best = 0.0f32;
        let falloff = self.falloff.max(1e-4);
        for v in &self.volumes {
            if let Some(q) = v.normalized_distance(p) {
                let d = ((1.0 - q) / falloff).clamp(0.0, 1.0);
                best = best.max(smoothstep(d));
            }
        }
        best
    }

    pub fn steer_target(&self, p: Vec3) -> Vec3 {
        let mut best_density = f32::NEG_INFINITY;
        let mut best_target = Vec3::ZERO;
        for v in &self.volumes {
            let d = v
                .normalized_distance(p)
                .map(|q| 1.0 - q)
                .unwrap_or(f32::NEG_INFINITY);
            if d > best_density {
                best_density = d;
                best_target = v.steer_target(p).unwrap_or(Vec3::ZERO);
            }
        }
        best_target
    }

    pub fn scaled(&self, s: f32) -> Self {
        Self {
            volumes: self.volumes.iter().map(|v| v.scaled(s)).collect(),
            ..self.clone()
        }
    }

    /// The envelope stretched along the trunk by `y`, its widths untouched.
    pub fn stretched(&self, y: f32) -> Self {
        Self {
            volumes: self.volumes.iter().map(|v| v.stretched(y)).collect(),
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cone() -> EnvelopeParams {
        EnvelopeParams {
            volumes: vec![EnvelopeVolume::Cone {
                base_y: 1.0,
                apex_y: 15.0,
                base_radius: 5.0,
                apex_radius: 0.5,
            }],
            falloff: 0.2,
            kill_threshold: 0.05,
            pull_strength: 0.6,
            for_trunk_length: 0.0,
        }
    }

    #[test]
    fn density_inside_center_is_full() {
        let env = cone();
        assert!(env.density(Vec3::new(0.0, 8.0, 0.0)) > 0.99);
    }

    #[test]
    fn density_outside_is_zero() {
        let env = cone();
        assert!(env.density(Vec3::new(20.0, 8.0, 0.0)) < 1e-5);
        assert!(env.density(Vec3::new(0.0, 20.0, 0.0)) < 1e-5);
    }

    #[test]
    fn density_falls_off_near_boundary() {
        let env = cone();
        let mid = env.density(Vec3::new(2.5, 8.0, 0.0));
        assert!(mid < 0.99 && mid > 0.0, "boundary density was {mid}");
    }

    #[test]
    fn steer_target_points_at_axis() {
        let env = cone();
        let t = env.steer_target(Vec3::new(4.0, 8.0, 3.0));
        assert!((t.x.abs()) < 1e-5 && (t.z.abs()) < 1e-5);
        assert!((t.y - 8.0).abs() < 1e-4);
    }

    #[test]
    fn scaling_scales_volumes() {
        let env = cone().scaled(2.0);
        assert!(env.density(Vec3::new(0.0, 16.0, 0.0)) > 0.99);
        assert!(env.density(Vec3::new(0.0, 35.0, 0.0)) < 1e-5);
    }
}
