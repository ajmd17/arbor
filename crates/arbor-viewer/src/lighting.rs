//! The lighting model, kept free of any GL so the real renderer and the offline
//! preview shade from one description rather than two that drift apart.

use glam::{Mat4, Vec3};

/// A sky dome and the sun in front of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyParams {
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground_bounce: Vec3,
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
}

// The viewer sends these numbers to the shaders rather than evaluating them on the
// CPU, so from the binary alone the evaluators look unused; the offline preview in
// examples/preview.rs shades with them, and the tests below check them.
#[allow(dead_code)]
impl SkyParams {
    /// Low warm sun under a cold dome, with a warm bounce coming back off the
    /// ground: the light of the first half hour after sunrise.
    pub fn dawn() -> Self {
        Self {
            zenith: Vec3::new(0.055, 0.085, 0.20),
            horizon: Vec3::new(0.62, 0.34, 0.22),
            ground_bounce: Vec3::new(0.085, 0.065, 0.055),
            sun_dir: Vec3::new(0.0, 0.22, 1.0).normalize(),
            // Irradiance, not radiance: every surface divides by PI on its way to
            // Lambert, so this reads high for what is a dim, very warm sun.
            sun_color: Vec3::new(11.0, 5.4, 2.25),
        }
    }

    /// Colour of the dome looking along `dir`, without the sun itself.
    pub fn dome(&self, dir: Vec3) -> Vec3 {
        let h = dir.y.clamp(-1.0, 1.0);
        let dome = self.horizon.lerp(self.zenith, h.max(0.0).powf(0.42));
        let t = ((h + 0.25) / 0.28).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        self.ground_bounce.lerp(dome, t)
    }

    /// Ambient arriving at a surface facing `n`.
    pub fn ambient(&self, n: Vec3) -> Vec3 {
        self.dome(n) * 0.55 + self.horizon * 0.12
    }

    /// The dome plus the sun disc and the wash around it, for the background.
    pub fn background(&self, dir: Vec3) -> Vec3 {
        let d = dir.dot(self.sun_dir).max(0.0);
        self.dome(dir)
            + self.sun_color * 0.22 * d.powf(900.0)
            + self.sun_color * 0.09 * d.powf(18.0)
            + self.horizon * 0.35 * d.powi(3)
    }
}

/// Shadow transform, plus the world size of one shadow texel.
///
/// The frustum is fitted to the tree together with where its shadow lands on the
/// ground. A low sun throws a shadow several times the height of the tree, and a box
/// sized to the tree alone simply cuts it off partway along.
pub fn light_view_proj(
    aabb: ([f32; 3], [f32; 3]),
    sun_dir: Vec3,
    shadow_size: i32,
) -> (Mat4, f32) {
    let min = Vec3::from(aabb.0);
    let max = Vec3::from(aabb.1);
    let sun = sun_dir.normalize_or(Vec3::Y);

    let mut points: Vec<Vec3> = Vec::with_capacity(16);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        points.push(corner);
        // Where that corner lands on the ground, following the light down.
        if sun.y > 0.08 && corner.y > 0.0 {
            points.push(corner - sun * (corner.y / sun.y));
        }
    }

    let centre = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let radius = points
        .iter()
        .map(|p| (*p - centre).length())
        .fold(0.0f32, f32::max)
        + 1.0;
    let up = if sun.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_at_rh(centre + sun * radius, centre, up);

    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for p in &points {
        let q = view.transform_point3(*p);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    let proj = Mat4::orthographic_rh(
        lo.x,
        hi.x,
        lo.y,
        hi.y,
        (-hi.z - 1.0).max(0.01),
        -lo.z + 1.0,
    );
    let texel = (hi.x - lo.x).max(hi.y - lo.y) / shadow_size.max(1) as f32;
    (proj * view, texel)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_aabb() -> ([f32; 3], [f32; 3]) {
        ([-4.0, 0.0, -4.0], [4.0, 18.0, 4.0])
    }

    fn sun_at(elevation_deg: f32) -> Vec3 {
        let e = elevation_deg.to_radians();
        Vec3::new(e.cos(), e.sin(), 0.0).normalize()
    }

    /// Everything the shadow map has to cover should land inside clip space.
    fn covers(m: Mat4, p: Vec3) -> bool {
        let c = m * p.extend(1.0);
        let n = c.truncate() / c.w;
        n.x.abs() <= 1.001 && n.y.abs() <= 1.001 && (0.0..=1.001).contains(&n.z)
    }

    #[test]
    fn the_frustum_reaches_the_far_end_of_a_low_sun_shadow() {
        let (min, max) = tree_aabb();
        let sun = sun_at(12.0);
        let (m, _) = light_view_proj((min, max), sun, 4096);

        // The top of the tree casts to here, tens of metres away.
        let top = Vec3::new(0.0, max[1], 0.0);
        let landing = top - sun * (top.y / sun.y);
        assert!(
            landing.length() > 4.0 * max[1],
            "a 12 degree sun should throw a long shadow, got {}",
            landing.length()
        );
        assert!(
            covers(m, landing),
            "the far end of the shadow at {landing:?} falls outside the shadow map"
        );
        assert!(covers(m, top), "the tree itself is not covered");
    }

    #[test]
    fn a_long_shadow_does_not_cost_shadow_resolution() {
        // The shadow a low sun throws runs away from the light, so it eats depth
        // range rather than ground area. Fitting the frustum to the light axis keeps
        // the texel footprint roughly the size of the tree however low the sun gets,
        // which a frustum sized to the shadow on the ground would not.
        let aabb = tree_aabb();
        let (_, low) = light_view_proj(aabb, sun_at(8.0), 2048);
        let (_, high) = light_view_proj(aabb, sun_at(70.0), 2048);
        let height = tree_aabb().1[1];
        assert!(
            low < 2.0 * height / 2048.0,
            "a low sun should not blow up the texel footprint: {low}"
        );
        assert!(
            low < high * 3.0,
            "resolution should hold up across sun angles: {low} against {high}"
        );
    }

    #[test]
    fn texel_size_halves_with_a_bigger_shadow_map() {
        let aabb = tree_aabb();
        let (_, coarse) = light_view_proj(aabb, sun_at(10.0), 2048);
        let (_, fine) = light_view_proj(aabb, sun_at(10.0), 4096);
        assert!(
            (fine - coarse * 0.5).abs() < coarse * 0.02,
            "{fine} should be half of {coarse}"
        );
    }

    #[test]
    fn a_sun_on_the_horizon_still_produces_a_usable_frustum() {
        // Below the cut-off the ground projection is skipped rather than exploding.
        let (m, texel) = light_view_proj(tree_aabb(), sun_at(1.0), 2048);
        assert!(texel.is_finite() && texel > 0.0);
        assert!(covers(m, Vec3::new(0.0, 18.0, 0.0)));
    }

    #[test]
    fn the_dome_is_warm_low_and_cold_high() {
        let sky = SkyParams::dawn();
        let low = sky.dome(Vec3::new(1.0, 0.02, 0.0).normalize());
        let high = sky.dome(Vec3::Y);
        assert!(low.x > low.z, "the horizon should be warm: {low:?}");
        assert!(high.z > high.x, "the zenith should be cold: {high:?}");
    }

    #[test]
    fn the_sun_is_the_brightest_thing_in_the_sky() {
        let sky = SkyParams::dawn();
        let at_sun = sky.background(sky.sun_dir);
        let away = sky.background(-sky.sun_dir);
        assert!(at_sun.length() > away.length() * 3.0);
    }
}
