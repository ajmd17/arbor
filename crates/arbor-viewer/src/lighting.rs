//! The lighting model, kept free of any GL so the real renderer and the offline
//! preview shade from one description rather than two that drift apart.
//!
//! The sky is built from the sun rather than picked by hand: sunlight crossing more
//! atmosphere loses its blue first, which is what makes a low sun red, the horizon
//! around it warm, and the zenith stay blue until late. Ambient then comes from that
//! sky by projecting the dome into spherical harmonics, so it follows the time of day
//! on its own instead of being a second set of numbers to keep in step.

use glam::{Mat4, Vec3};

/// Rayleigh optical depth at sea level, per channel at roughly 615/535/465 nm. Blue
/// scatters an order of magnitude more than red, which is the whole reason a sunset
/// is a sunset.
const TAU_RAYLEIGH: Vec3 = Vec3::new(0.0722, 0.1345, 0.2934);
/// Aerosol optical depth. Haze scatters far more evenly than air does, but not quite
/// evenly, so it reddens a low sun a little further.
const TAU_AEROSOL: Vec3 = Vec3::new(0.087, 0.105, 0.132);
/// Sunlight above the atmosphere. Irradiance, not radiance: every surface divides by
/// PI on its way to Lambert.
const SUN_TOP: Vec3 = Vec3::new(45.0, 43.5, 42.0);
/// How bright the dome is against the sun that lights it.
const SKY_GAIN: f32 = 0.85;
/// What the sky is once the sun is well down: starlight and airglow, cold and dim.
const NIGHT_ZENITH: Vec3 = Vec3::new(0.0010, 0.0014, 0.0030);
const NIGHT_HORIZON: Vec3 = Vec3::new(0.0014, 0.0018, 0.0034);
/// Reflectance of the ground the dome bounces off.
const GROUND_ALBEDO: Vec3 = Vec3::new(0.26, 0.19, 0.13);

/// A sky dome and the sun in front of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyParams {
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground_bounce: Vec3,
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn exp3(v: Vec3) -> Vec3 {
    Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

/// Kasten-Young relative air mass: how many atmospheres' worth of air the light
/// crosses on its way in. One straight overhead, near forty on the horizon, and it is
/// this number rather than the angle that decides the colour.
fn air_mass(sin_elev: f32) -> f32 {
    // Clamped at the horizon. The fit is only defined for a sun that is up, and fed a
    // negative elevation it turns back down and claims a set sun is crossing less air
    // than one overhead, which lights the dusk sky like noon.
    let sin_elev = sin_elev.max(0.0);
    let deg = sin_elev.clamp(0.0, 1.0).asin().to_degrees();
    let denom = sin_elev + 0.50572 * (deg + 6.07995).powf(-1.6364);
    (1.0 / denom.max(1e-4)).min(40.0)
}

fn transmit(m: f32) -> Vec3 {
    exp3(-(TAU_RAYLEIGH + TAU_AEROSOL) * m)
}

// The viewer sends these numbers to the shaders rather than evaluating them on the
// CPU, so from the binary alone the evaluators look unused; the offline preview in
// examples/preview.rs shades with them, and the tests below check them.
#[allow(dead_code)]
impl SkyParams {
    /// Low warm sun under a cold dome: the light of the first half hour after sunrise.
    pub fn dawn() -> Self {
        Self::for_sun(Vec3::new(0.0, 0.105, 1.0))
    }

    /// The whole sky that goes with a sun in this direction.
    pub fn for_sun(sun_dir: Vec3) -> Self {
        let sun = sun_dir.normalize_or(Vec3::Y);
        let sin_elev = sun.y;
        let m = air_mass(sin_elev);

        // Civil twilight: the sun keeps lighting the dome from below the horizon for a
        // while after it sets, so the day fades out rather than switching off.
        let day = smoothstep(-0.21, 0.10, sin_elev);

        // How much sunlight there is to scatter at all. Illuminance falls away far
        // faster than the sun's height does, which is why the minute after sunrise is
        // hundreds of times dimmer than noon rather than a little dimmer.
        let sky_light = day * (0.06 + 0.94 * sin_elev.max(0.0).powf(0.6));

        // The dome is sunlight scattered once on its way in. Light scattered high
        // overhead has crossed less air than light scattered near the horizon, so a
        // low sun can leave the zenith blue while the horizon has gone red. The path
        // to the zenith stops lengthening once the sun is low, because that light is
        // scattered high in the atmosphere and has little air left to cross on the way
        // down; without the cap a sunset turns the whole sky red, which is not what a
        // sunset looks like.
        let zenith_light = transmit(m.min(4.0) * 0.30);
        // The horizon is the one place single scattering is badly wrong: it predicts a
        // dark red band, where a real horizon is bright and pale because most of the
        // light arriving there has bounced several times and kept little of the
        // reddening. Mixing a lightly attenuated term in for that is what stops the
        // horizon coming out darker than the zenith, which never happens in daylight.
        let single = transmit(m * 1.25);
        let multi = transmit(m * 0.5);
        let horizon_light = single * 0.35 + multi * 0.65;
        // Rayleigh scattering is what makes the sky blue; aerosol scattering is close
        // to neutral and is what makes the horizon pale rather than deep blue.
        let rayleigh = TAU_RAYLEIGH / TAU_RAYLEIGH.z;

        let zenith = (rayleigh * 0.92 + Vec3::splat(0.10)) * zenith_light * sky_light * SKY_GAIN;
        let horizon =
            (rayleigh * 0.35 + Vec3::splat(0.90)) * horizon_light * sky_light * SKY_GAIN * 1.5;

        let zenith = zenith + NIGHT_ZENITH;
        let horizon = horizon + NIGHT_HORIZON;

        // What comes back up off the ground: the dome it can see, plus whatever the
        // sun lays on it.
        let sun_color = SUN_TOP * transmit(m) * day;
        let onto_ground = (zenith + horizon) * 0.5 + sun_color * sin_elev.max(0.0) / PI;
        let ground_bounce = onto_ground * GROUND_ALBEDO;

        Self {
            zenith,
            horizon,
            ground_bounce,
            sun_dir: sun,
            sun_color,
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

    /// The dome plus the sun disc and the wash around it, for the background.
    pub fn background(&self, dir: Vec3) -> Vec3 {
        let d = dir.dot(self.sun_dir).max(0.0);
        self.dome(dir)
            + self.sun_color * 0.22 * d.powf(900.0)
            + self.sun_color * 0.09 * d.powf(18.0)
            + self.horizon * 0.35 * d.powi(3)
    }

    /// Exposure that fits a whole day into a display without flattening it.
    ///
    /// Noon is some thirty times brighter than the minute after sunrise. Adapting all
    /// of that away would make every hour look alike, so this adapts most of it and
    /// leaves the rest, the way an eye does.
    pub fn exposure(&self) -> f32 {
        const ADAPT: f32 = 0.82;
        const KEY: f32 = 2.2;
        let key = luminance(self.sun_color * self.sun_dir.y.max(0.0) / PI)
            + luminance(self.ambient_key());
        // Clamped at the top so night stays night. Without it, adaptation would keep
        // opening up until starlight read as daylight.
        (KEY / key.max(1e-4)).powf(ADAPT).clamp(0.05, 14.0)
    }

    fn ambient_key(&self) -> Vec3 {
        (self.zenith + self.horizon) * 0.5
    }

    /// The dome projected into nine spherical-harmonic coefficients.
    ///
    /// This is what makes ambient follow the sky: instead of a hand-picked fraction of
    /// whatever the surface happens to face, every surface gets the whole dome weighted
    /// by how much of it it can see, ground bounce included.
    pub fn sh9(&self) -> [Vec3; 9] {
        // A Fibonacci sphere covers evenly, without the pole clustering that a
        // latitude-longitude grid would bias the low-order terms with.
        const N: usize = 4096;
        let golden = PI * (3.0 - 5.0f32.sqrt());
        let mut sh = [Vec3::ZERO; 9];
        for i in 0..N {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / N as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let a = golden * i as f32;
            let dir = Vec3::new(r * a.cos(), z, r * a.sin());
            let radiance = self.dome(dir);
            for (k, b) in sh_basis(dir).into_iter().enumerate() {
                sh[k] += radiance * b;
            }
        }
        let weight = 4.0 * PI / N as f32;
        sh.map(|c| c * weight)
    }
}

const PI: f32 = std::f32::consts::PI;

fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// The nine real spherical harmonics up to second order, in the order the irradiance
/// evaluation below expects.
fn sh_basis(d: Vec3) -> [f32; 9] {
    [
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    ]
}

/// The dome's harmonics for a sky, remembered between calls.
///
/// Projecting the dome is not expensive, but the renderer asks for the same sky once
/// per shader program per frame, and a sky only changes when the sun moves. This is a
/// memo of a pure function, nothing more.
pub fn sh9_cached(sky: &SkyParams) -> [Vec3; 9] {
    use std::cell::RefCell;
    thread_local! {
        static LAST: RefCell<Option<(SkyParams, [Vec3; 9])>> = const { RefCell::new(None) };
    }
    LAST.with(|cell| {
        let mut slot = cell.borrow_mut();
        if let Some((cached, sh)) = slot.as_ref()
            && cached == sky
        {
            return *sh;
        }
        let sh = sky.sh9();
        *slot = Some((*sky, sh));
        sh
    })
}

/// A sky's ambient, ready to evaluate per surface.
#[derive(Clone, Copy, Debug)]
pub struct SkyIrradiance {
    pub sh: [Vec3; 9],
}

#[allow(dead_code)]
impl SkyIrradiance {
    pub fn new(sky: &SkyParams) -> Self {
        Self {
            sh: sh9_cached(sky),
        }
    }

    /// Irradiance arriving at a surface facing `n`, divided by PI.
    ///
    /// The division is what the Lambert convention here expects: a surface multiplies
    /// this by its albedo and nothing else, and a uniform dome of radiance L returns L.
    pub fn eval(&self, n: Vec3) -> Vec3 {
        // Ramamoorthi and Hanrahan's closed form for the cosine-convolved dome, with
        // the 1/PI folded into the constants.
        const C1: f32 = 0.429043 / PI;
        const C2: f32 = 0.511664 / PI;
        const C3: f32 = 0.743125 / PI;
        const C4: f32 = 0.886227 / PI;
        const C5: f32 = 0.247708 / PI;
        let n = n.normalize_or(Vec3::Y);
        let (x, y, z) = (n.x, n.y, n.z);
        let e = self.sh[0] * C4 - self.sh[6] * C5
            + (self.sh[3] * x + self.sh[1] * y + self.sh[2] * z) * (2.0 * C2)
            + self.sh[6] * (C3 * z * z)
            + (self.sh[4] * x * y + self.sh[7] * x * z + self.sh[5] * y * z) * (2.0 * C1)
            + self.sh[8] * (C1 * (x * x - y * y));
        e.max(Vec3::ZERO)
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

    fn sky_at(elevation_deg: f32) -> SkyParams {
        SkyParams::for_sun(sun_at(elevation_deg))
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

    #[test]
    fn a_low_sun_is_red_and_a_high_one_is_not() {
        // The whole point of driving the sky from air mass: the same light source
        // changes colour with how much atmosphere it is coming through.
        let dawn = sky_at(4.0).sun_color;
        let noon = sky_at(80.0).sun_color;
        let warmth = |c: Vec3| c.x / c.z.max(1e-4);
        assert!(
            warmth(dawn) > 6.0,
            "a 4 degree sun should be deeply red: {dawn:?}"
        );
        assert!(
            warmth(noon) < 1.5,
            "an 80 degree sun should be near white: {noon:?}"
        );
        assert!(
            noon.length() > dawn.length() * 3.0,
            "noon should be far brighter: {noon:?} against {dawn:?}"
        );
    }

    #[test]
    fn the_zenith_keeps_its_blue_while_the_horizon_reddens() {
        // A sunset is not the whole sky turning red. Light scattered overhead has
        // crossed much less air than light scattered at the horizon.
        let sky = sky_at(3.0);
        assert!(
            sky.zenith.z > sky.zenith.x,
            "zenith should still be blue at sunset: {:?}",
            sky.zenith
        );
        assert!(
            sky.horizon.x > sky.horizon.z,
            "horizon should be warm at sunset: {:?}",
            sky.horizon
        );
    }

    #[test]
    fn night_is_dark_and_cold_but_not_black() {
        let night = sky_at(-20.0);
        assert!(
            luminance(night.zenith) < 0.03,
            "night should be dark: {:?}",
            night.zenith
        );
        assert!(
            luminance(night.zenith) > 0.0,
            "night should not be pure black"
        );
        assert!(
            night.zenith.z > night.zenith.x,
            "night should be cold: {:?}",
            night.zenith
        );
        assert!(
            luminance(night.sun_color) < 0.01,
            "the sun should be off: {:?}",
            night.sun_color
        );
    }

    #[test]
    fn a_set_sun_is_still_behind_a_whole_atmosphere() {
        // The air mass fit is only defined above the horizon; below it, it turns back
        // down. Left alone it reports a sun six degrees under the horizon as being
        // less attenuated than one overhead, and dusk comes out lit like noon.
        let horizon = air_mass(0.0);
        assert!(
            horizon > 30.0,
            "a sun on the horizon crosses tens of atmospheres, got {horizon}"
        );
        for elev in [-1.0f32, -6.0, -20.0, -60.0] {
            let below = air_mass(elev.to_radians().sin());
            assert!(
                below >= horizon - 1e-3,
                "a sun {elev} degrees down reported {below} against {horizon} at the horizon"
            );
        }
        // And the sky it produces has to keep getting darker, never brighter.
        let mut last = f32::MAX;
        for elev in [6.0f32, 2.0, 0.0, -3.0, -6.0, -12.0] {
            let sky = SkyParams::for_sun(sun_at(elev));
            let lum = luminance(sky.horizon);
            assert!(
                lum < last,
                "dusk brightened at {elev} degrees: {lum} against {last}"
            );
            last = lum;
        }
    }

    #[test]
    fn ambient_reproduces_a_uniform_dome_exactly() {
        // The calibration the whole ambient path rests on: a dome of uniform radiance
        // L has to come back out as L, whatever the surface faces.
        let flat = SkyParams {
            zenith: Vec3::splat(0.5),
            horizon: Vec3::splat(0.5),
            ground_bounce: Vec3::splat(0.5),
            sun_dir: Vec3::Y,
            sun_color: Vec3::ZERO,
        };
        let irr = SkyIrradiance::new(&flat);
        for n in [Vec3::Y, -Vec3::Y, Vec3::X, Vec3::Z, Vec3::new(1.0, 1.0, 1.0)] {
            let got = irr.eval(n.normalize());
            assert!(
                (got.x - 0.5).abs() < 0.01,
                "facing {n:?} gave {got:?}, wanted 0.5"
            );
        }
    }

    #[test]
    fn ambient_follows_the_sky_rather_than_a_fixed_fraction_of_it() {
        // Up sees the dome, down sees the ground bounce. Hand-picking a fraction of
        // whatever the surface faces cannot tell those apart the way an integral does.
        let sky = sky_at(8.0);
        let irr = SkyIrradiance::new(&sky);
        let up = irr.eval(Vec3::Y);
        let down = irr.eval(-Vec3::Y);
        assert!(
            luminance(up) > luminance(down),
            "up should catch more light than down: {up:?} against {down:?}"
        );
        assert!(
            down.x > down.z,
            "light coming off warm ground should be warm: {down:?}"
        );
        assert!(
            up.z > up.x,
            "light coming off a blue dome should be cold: {up:?}"
        );
    }

    #[test]
    fn ambient_brightens_with_the_day() {
        let dawn = SkyIrradiance::new(&sky_at(4.0)).eval(Vec3::Y);
        let noon = SkyIrradiance::new(&sky_at(80.0)).eval(Vec3::Y);
        assert!(
            luminance(noon) > luminance(dawn) * 3.0,
            "midday ambient should dwarf dawn: {noon:?} against {dawn:?}"
        );
    }

    #[test]
    fn exposure_holds_the_day_inside_a_display() {
        // Adapted, but not adapted flat: noon still has to read brighter than dawn.
        let key = |e: f32| {
            let sky = sky_at(e);
            luminance(sky.sun_color * sky.sun_dir.y.max(0.0) / PI) * sky.exposure()
        };
        let dawn = key(4.0);
        let noon = key(80.0);
        assert!(noon < 4.0, "noon should not be blown out: {noon}");
        assert!(dawn > 0.02, "dawn should not be crushed: {dawn}");
        assert!(
            noon > dawn * 1.2,
            "the day should still read brighter than dawn: {noon} against {dawn}"
        );
    }
}
