//! Photographed environments: equirectangular HDR images, and what the renderer needs
//! out of one before it can light a tree with it. Kept free of GL, like the sky model,
//! so the import tool and the tests read an environment exactly as the viewer does.
//!
//! A photographed sky carries its sun as a handful of pixels tens of thousands of times
//! brighter than the rest. Left in the image, that sun is smeared into the ambient by
//! the low-order projection and lights everything from everywhere, and it casts no
//! shadow. So it is taken out: the pixels that stand clear of the sky around them are
//! clamped back to it, and what was removed becomes a directional light of the same
//! energy, which does cast shadows and does light a leaf through from behind.

use glam::Vec3;

use crate::lighting::{luminance, sh_basis};

const PI: f32 = std::f32::consts::PI;

/// Where environments are read from, relative to the working directory.
pub const HDRI_DIR: &str = "assets/hdri";

/// The environments bundled with the repository, in the order the viewer offers them.
/// On the desktop any other `.hdr` dropped into [HDRI_DIR] is offered after these.
pub const BUNDLED: &[&str] = &[
    "spruit_sunrise",
    "pretville_street",
    "belfast_sunset_puresky",
    "blue_photo_studio",
];

/// How a tree is set into a photograph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Staging {
    /// Metres above the ground the photograph is taken to have been shot from.
    pub shot_from: f32,
    /// Metres off its surroundings are taken to stand: the radius of the dome its
    /// ground is laid under the tree on.
    pub radius: f32,
    /// Stops to expose it by. Published photographs are meant to be seen as they are,
    /// but not every one is saved as bright as the next.
    pub exposure: f32,
    /// Whether the ground in it is worth standing the tree on. A sky-only photograph's
    /// floor is painted in, and a plain ground does better.
    pub has_ground: bool,
}

/// How the photograph `name` is staged. Guessed for the bundled ones from what is in
/// them; anything else gets a middling open field.
pub fn staging(name: &str) -> Staging {
    let (shot_from, radius, exposure, has_ground) = match name {
        // Open field, trees and a fence a long way off.
        "spruit_sunrise" => (10.0, 150.0, 0.0, true),
        // A car park between low shops.
        "pretville_street" => (6.0, 45.0, 0.0, true),
        // Nothing but sky over a painted-in floor, saved two stops brighter than the rest.
        "belfast_sunset_puresky" => (10.0, 400.0, -1.7, false),
        // A room some fifteen metres across.
        "blue_photo_studio" => (3.0, 14.0, -0.3, true),
        _ => (10.0, 100.0, 0.0, true),
    };
    Staging { shot_from, radius, exposure, has_ground }
}

/// Every environment the viewer can offer, by name: the bundled ones, then on the
/// desktop any other `.hdr` in [HDRI_DIR]. The web cannot list a folder, and offers
/// the bundled ones alone.
pub fn available() -> Vec<String> {
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut names: Vec<String> = BUNDLED.iter().map(|s| s.to_string()).collect();
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(entries) = std::fs::read_dir(HDRI_DIR) {
        let mut extra: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("hdr")))
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .filter(|n| !names.contains(n))
            .collect();
        extra.sort();
        names.extend(extra);
    }
    names
}

/// An equirectangular image: longitude across, the pole straight up at the top row.
#[derive(Clone)]
pub struct Equirect {
    pub width: usize,
    pub height: usize,
    /// Linear radiance, row by row from the top.
    pub pixels: Vec<Vec3>,
}

/// The sun, once it has been lifted out of a photograph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// Toward the sun, in the environment's own frame.
    pub dir: Vec3,
    /// Irradiance on a surface square to it: the same convention as the procedural
    /// sky's `sun_color`, so every shader reads either without knowing which it is.
    pub irradiance: Vec3,
    /// Angular radius of what was taken out, disc and glare together. The shadows take
    /// their softness from it.
    pub angular_radius: f32,
}

/// Everything the renderer takes from a photographed environment.
pub struct Analysed {
    /// The image with its sun clamped out: what the ambient and the reflections see.
    pub image: Equirect,
    pub sun: Option<Sun>,
    /// Nine spherical-harmonic coefficients of the sunless image, in the order
    /// [sh_basis] gives them.
    pub sh: [Vec3; 9],
}

impl Equirect {
    /// Reads a Radiance `.hdr`.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .map_err(|e| format!("not a readable .hdr: {e}"))?
            .into_rgb32f();
        let (width, height) = (img.width() as usize, img.height() as usize);
        let pixels = img
            .pixels()
            .map(|p| Vec3::new(p[0], p[1], p[2]).max(Vec3::ZERO))
            .collect();
        Ok(Self { width, height, pixels })
    }

    /// Writes a Radiance `.hdr`. The import tool's; the viewer only reads.
    #[allow(dead_code)]
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let rgb: Vec<image::Rgb<f32>> = self.pixels.iter().map(|p| image::Rgb(p.to_array())).collect();
        let mut out = Vec::new();
        image::codecs::hdr::HdrEncoder::new(&mut out)
            .encode(&rgb, self.width, self.height)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// Box-filtered down by whole factors until it is no wider than `max_width`. A box
    /// keeps the energy where it was, which is what matters for a light source.
    pub fn shrink_to(&self, max_width: usize) -> Self {
        let mut img = self.clone();
        while img.width > max_width && img.width.is_multiple_of(2) && img.height.is_multiple_of(2) {
            let (w, h) = (img.width / 2, img.height / 2);
            let mut px = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let at = |dx: usize, dy: usize| img.pixels[(2 * y + dy) * img.width + 2 * x + dx];
                    px.push((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25);
                }
            }
            img = Self { width: w, height: h, pixels: px };
        }
        img
    }

    /// The direction through the centre of pixel (x, y).
    ///
    /// The middle of the image looks down -Z and the image runs rightward toward +X,
    /// so a panorama seen from inside reads the way it was shot rather than mirrored.
    /// The shaders use the same mapping.
    pub fn direction(&self, x: usize, y: usize) -> Vec3 {
        let u = (x as f32 + 0.5) / self.width as f32;
        let v = (y as f32 + 0.5) / self.height as f32;
        dir_from_uv(u, v)
    }

    /// Solid angle one pixel in row `y` covers.
    pub fn texel_solid_angle(&self, y: usize) -> f32 {
        let theta = PI * (y as f32 + 0.5) / self.height as f32;
        (2.0 * PI / self.width as f32) * (PI / self.height as f32) * theta.sin()
    }

    /// Radiance looking along `dir`, nearest pixel. The shaders do their own lookups;
    /// this is for checking what they should find.
    #[allow(dead_code)]
    pub fn sample(&self, dir: Vec3) -> Vec3 {
        let (u, v) = uv_from_dir(dir);
        let x = ((u * self.width as f32) as usize).min(self.width - 1);
        let y = ((v * self.height as f32) as usize).min(self.height - 1);
        self.pixels[y * self.width + x]
    }

    /// The image projected into nine spherical harmonics.
    pub fn sh9(&self) -> [Vec3; 9] {
        let mut sh = [Vec3::ZERO; 9];
        for y in 0..self.height {
            let d_omega = self.texel_solid_angle(y);
            for x in 0..self.width {
                let radiance = self.pixels[y * self.width + x] * d_omega;
                for (k, b) in sh_basis(self.direction(x, y)).into_iter().enumerate() {
                    sh[k] += radiance * b;
                }
            }
        }
        sh
    }

    /// Mean radiance over the whole sphere.
    pub fn mean(&self) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for y in 0..self.height {
            let d_omega = self.texel_solid_angle(y);
            for x in 0..self.width {
                sum += self.pixels[y * self.width + x] * d_omega;
            }
        }
        sum / (4.0 * PI)
    }

    /// Takes the sun out of the image, if it has one, and hands it back as a light.
    ///
    /// Anything within a few degrees of the brightest pixel that stands well clear of
    /// the sky just around it is sun: disc, corona and lens glare alike, since all of it
    /// arrives from that one direction as far as a tree is concerned. Those pixels are
    /// clamped down to the threshold and the energy above it is summed into the light.
    /// An overcast sky or a room lit evenly has nothing that stands clear, and gives
    /// back `None` with the image untouched.
    pub fn extract_sun(&mut self) -> Option<Sun> {
        const REGION: f32 = 9.0; // degrees from the peak that may count as sun
        const RING: (f32, f32) = (12.0, 20.0); // where the sky around it is measured
        const STANDS_OUT: f32 = 12.0; // how many times brighter than that sky a peak must be
        const CLAMP: f32 = 3.0; // the level, against that sky, the sun is clamped to
        // How far the sun's irradiance has to outdo the whole sky's before it is a sun.
        // Clear suns in the bundled photographs come out at 14 to 29 times; windows,
        // lamps and a sunset sun behind cloud at 2.5 or less.
        const SUN_OVER_SKY: f32 = 3.0;

        let (peak_index, peak_lum) = self
            .pixels
            .iter()
            .map(|p| luminance(*p))
            .enumerate()
            .fold((0, 0.0f32), |best, (i, l)| if l > best.1 { (i, l) } else { best });
        let peak = self.direction(peak_index % self.width, peak_index / self.width);

        let cos_region = REGION.to_radians().cos();
        let (cos_ring_in, cos_ring_out) = (RING.0.to_radians().cos(), RING.1.to_radians().cos());
        let mut ring: Vec<f32> = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let c = self.direction(x, y).dot(peak);
                if c <= cos_ring_in && c >= cos_ring_out {
                    ring.push(luminance(self.pixels[y * self.width + x]));
                }
            }
        }
        if ring.is_empty() {
            return None;
        }
        ring.sort_by(f32::total_cmp);
        let sky = ring[ring.len() / 2].max(1e-6);
        if peak_lum < sky * STANDS_OUT {
            return None;
        }

        let threshold = sky * CLAMP;
        let mut irradiance = Vec3::ZERO;
        let mut removed = Vec3::ZERO;
        let mut toward = Vec3::ZERO;
        let mut weight = 0.0f32;
        let mut spread = 0.0f32;
        let mut clamped = Vec::new();
        for y in 0..self.height {
            let d_omega = self.texel_solid_angle(y);
            for x in 0..self.width {
                let dir = self.direction(x, y);
                let c = dir.dot(peak);
                if c < cos_region {
                    continue;
                }
                let i = y * self.width + x;
                let lum = luminance(self.pixels[i]);
                if lum <= threshold {
                    continue;
                }
                let kept = self.pixels[i] * (threshold / lum);
                let taken = (self.pixels[i] - kept) * d_omega;
                clamped.push((i, kept));
                removed += taken;
                // Irradiance on a surface square to the sun, so each part of the
                // removed light is weighted by how squarely it would arrive.
                irradiance += taken * c;
                let w = luminance(taken);
                toward += dir * w;
                weight += w;
                spread += w * c.clamp(-1.0, 1.0).acos().powi(2);
            }
        }
        // A window, a lamp or a sun low in cloud stands out from the sky beside it
        // without outshining the sky as a whole. Lifted out, it would cast a hard
        // shadow the photograph never had, so it stays part of the ambient.
        let sky_irradiance = luminance(self.mean() - removed / (4.0 * PI)) * PI;
        if weight <= 0.0 || luminance(irradiance) < SUN_OVER_SKY * sky_irradiance {
            return None;
        }
        for (i, kept) in clamped {
            self.pixels[i] = kept;
        }
        Some(Sun {
            dir: toward.normalize_or(peak),
            irradiance,
            // An RMS spread under-reads a disc's edge; 1.4 puts it back near the rim.
            angular_radius: ((spread / weight).sqrt() * 1.4).max(0.0047),
        })
    }

    /// Sun out, harmonics taken: the environment ready to light with.
    pub fn analyse(mut self) -> Analysed {
        let sun = self.extract_sun();
        // The harmonics only carry the broad shape of the light, so a small copy says
        // the same as the full image in a fraction of the time.
        let sh = self.shrink_to(256).sh9();
        Analysed { image: self, sun, sh }
    }
}

/// The equirectangular mapping, `u` across and `v` down from the zenith.
pub fn dir_from_uv(u: f32, v: f32) -> Vec3 {
    let phi = 2.0 * PI * (u - 0.5);
    let theta = PI * v;
    Vec3::new(theta.sin() * phi.sin(), theta.cos(), -theta.sin() * phi.cos())
}

#[allow(dead_code)]
pub fn uv_from_dir(dir: Vec3) -> (f32, f32) {
    let d = dir.normalize_or(Vec3::Y);
    let u = 0.5 + d.x.atan2(-d.z) / (2.0 * PI);
    let v = d.y.clamp(-1.0, 1.0).acos() / PI;
    (u.rem_euclid(1.0), v)
}

/// A world direction turned into the environment's frame, for an environment turned
/// `rotation` radians about the vertical. The shaders turn every lookup the same way.
pub fn to_env(dir: Vec3, rotation: f32) -> Vec3 {
    let (s, c) = rotation.sin_cos();
    Vec3::new(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z)
}

/// The environment's frame back into the world's.
pub fn from_env(dir: Vec3, rotation: f32) -> Vec3 {
    to_env(dir, -rotation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: usize, value: Vec3) -> Equirect {
        Equirect { width, height: width / 2, pixels: vec![value; width * width / 2] }
    }

    /// A pale sky with a small, very bright sun painted into it at `sun`.
    fn sky_with_sun(sun: Vec3, sun_radiance: f32) -> Equirect {
        let mut img = flat(512, Vec3::new(0.4, 0.5, 0.7));
        let cos_disc = 0.8f32.to_radians().cos();
        for y in 0..img.height {
            for x in 0..img.width {
                if img.direction(x, y).dot(sun) > cos_disc {
                    img.pixels[y * img.width + x] = Vec3::splat(sun_radiance);
                }
            }
        }
        img
    }

    #[test]
    fn the_mapping_turns_around_and_back() {
        for dir in [Vec3::X, -Vec3::Z, Vec3::new(0.3, 0.8, -0.5), Vec3::new(-0.6, -0.2, 0.7)] {
            let dir = dir.normalize();
            let (u, v) = uv_from_dir(dir);
            assert!(dir_from_uv(u, v).distance(dir) < 1e-4, "{dir:?} came back wrong");
        }
        // The middle of the image looks down -Z, and a quarter further on looks at +X.
        assert!(dir_from_uv(0.5, 0.5).distance(-Vec3::Z) < 1e-5);
        assert!(dir_from_uv(0.75, 0.5).distance(Vec3::X) < 1e-5);
        assert!(dir_from_uv(0.5, 0.0).distance(Vec3::Y) < 1e-5);
    }

    #[test]
    fn rotation_goes_there_and_back() {
        let d = Vec3::new(0.3, 0.4, -0.8).normalize();
        assert!(from_env(to_env(d, 1.1), 1.1).distance(d) < 1e-5);
        // A quarter turn carries -Z onto the environment's +X or -X, not upward.
        assert!(to_env(-Vec3::Z, PI / 2.0).y.abs() < 1e-6);
    }

    #[test]
    fn solid_angles_cover_the_sphere() {
        let img = flat(256, Vec3::ONE);
        let total: f32 = (0..img.height).map(|y| img.texel_solid_angle(y) * img.width as f32).sum();
        assert!((total - 4.0 * PI).abs() < 0.01, "{total}");
        assert!((img.mean() - Vec3::ONE).length() < 1e-3);
    }

    #[test]
    fn a_uniform_environment_projects_to_a_constant() {
        // Only the constant band survives, at L * sqrt(4 pi).
        let sh = flat(256, Vec3::splat(2.0)).sh9();
        assert!((sh[0].x - 2.0 * (4.0 * PI).sqrt()).abs() < 0.02, "{:?}", sh[0]);
        for c in &sh[1..] {
            assert!(c.length() < 0.02, "{c:?}");
        }
    }

    #[test]
    fn the_sun_comes_out_where_it_was_and_with_its_energy() {
        let toward = Vec3::new(0.5, 0.6, -0.4).normalize();
        let radiance = 20000.0;
        let mut img = sky_with_sun(toward, radiance);
        let before = img.mean();
        let sun = img.extract_sun().expect("a clear sun should be found");
        assert!(sun.dir.angle_between(toward).to_degrees() < 0.3, "{:?}", sun.dir);

        // A disc of radius r holds about pi r^2 of solid angle.
        let disc = PI * 0.8f32.to_radians().powi(2);
        let expected = radiance * disc;
        let got = luminance(sun.irradiance);
        assert!((got / expected - 1.0).abs() < 0.2, "irradiance {got} against {expected}");

        // Whatever was taken out is gone from the image, and the sky is left as it was.
        let sky = luminance(Vec3::new(0.4, 0.5, 0.7));
        let after = luminance(img.mean());
        assert!(luminance(before) > sky * 2.0, "the sun should dominate the image first");
        assert!((after / sky - 1.0).abs() < 0.01, "{after} against a sky of {sky}");
        let away = img.sample(-toward);
        assert!((away - Vec3::new(0.4, 0.5, 0.7)).length() < 1e-5);
        assert!(sun.angular_radius.to_degrees() < 2.0, "{}", sun.angular_radius.to_degrees());
    }

    #[test]
    fn an_even_sky_has_no_sun() {
        let mut img = flat(256, Vec3::new(0.6, 0.6, 0.65));
        // A gentle brightening toward the zenith, as an overcast sky has.
        for y in 0..img.height {
            for x in 0..img.width {
                let up = img.direction(x, y).y.max(0.0);
                img.pixels[y * img.width + x] *= 1.0 + up;
            }
        }
        let untouched = img.pixels.clone();
        assert!(img.extract_sun().is_none());
        assert!(img.pixels == untouched);
    }

    #[test]
    fn a_bright_spot_that_does_not_outshine_the_sky_stays_in_it() {
        // A lamp: far brighter than the sky beside it, but small and not bright enough
        // to light the scene more than the sky does.
        let mut img = sky_with_sun(Vec3::new(0.3, 0.2, -0.9).normalize(), 400.0);
        let untouched = img.pixels.clone();
        assert!(img.extract_sun().is_none());
        assert!(img.pixels == untouched);
    }

    #[test]
    fn shrinking_keeps_the_energy() {
        let img = sky_with_sun(Vec3::new(0.2, 0.9, 0.1).normalize(), 5000.0);
        let small = img.shrink_to(128);
        assert_eq!((small.width, small.height), (128, 64));
        let (a, b) = (luminance(img.mean()), luminance(small.mean()));
        assert!((a / b - 1.0).abs() < 0.02, "{a} against {b}");
    }
}
