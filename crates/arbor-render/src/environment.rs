//! The environment a scene is lit by, and the settings that shape how it lights.
//!
//! Either the procedural sky or a photograph from [`crate::hdri`]. This keeps track of
//! which one the GPU maps were last built from, so asking for the same one again costs
//! nothing, and turns it plus a few sliders into the [`Lighting`] every lit pass takes.

use glam::Vec3;

use crate::hdri::{self, Equirect, Sun};
use crate::ibl::EnvironmentMaps;
use crate::lighting::SkyParams;
use crate::render::Lighting;

/// What the maps on the GPU were built from.
#[derive(Clone, Debug, PartialEq)]
enum Built {
    Nothing,
    Sky(SkyParams),
    Photo(String),
}

/// What is known about the photograph on the GPU.
#[derive(Clone, Copy)]
struct Photo {
    sun: Option<Sun>,
    sh: [Vec3; 9],
}

/// The sliders that shape the lighting.
#[derive(Clone, Copy, Debug)]
pub struct EnvironmentSettings {
    /// The procedural sky, used when no photograph is loaded. Its sun colour already
    /// carries the sun's strength.
    pub sky: SkyParams,
    /// Degrees the photograph is turned about the vertical.
    pub rotation_deg: f32,
    /// Scales the photograph's light, sky and sun both.
    pub intensity: f32,
    /// Scales the sun found in a photograph, on top of `intensity`.
    pub photo_sun: f32,
    /// Stops of exposure, on top of what the sky or photograph asks for.
    pub exposure_ev: f32,
    /// Angular diameter of the sun disc, in degrees.
    pub sun_size_deg: f32,
    /// How far the shadows soften against what a sun of that size gives.
    pub shadow_softness: f32,
}

impl EnvironmentSettings {
    pub fn new(sky: SkyParams) -> Self {
        Self {
            sky,
            rotation_deg: 0.0,
            intensity: 1.0,
            photo_sun: 1.0,
            exposure_ev: 0.0,
            sun_size_deg: 0.53,
            shadow_softness: 1.0,
        }
    }
}

pub struct Environment {
    built: Built,
    photo: Option<Photo>,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        Self { built: Built::Nothing, photo: None }
    }

    /// The name of the photograph the maps are built from, if they are.
    pub fn photo_name(&self) -> Option<&str> {
        match &self.built {
            Built::Photo(name) => Some(name),
            _ => None,
        }
    }

    /// Builds the maps from the procedural sky, unless they already are.
    pub fn load_sky(&mut self, gl: &glow::Context, maps: &mut EnvironmentMaps, sky: &SkyParams) {
        if self.built == Built::Sky(*sky) {
            return;
        }
        maps.load_sky(gl, sky);
        self.built = Built::Sky(*sky);
        self.photo = None;
    }

    /// Builds the maps from the `.hdr` file `bytes`, called `name`, unless they already
    /// are. On failure the maps are left as they were.
    pub fn load_photo(
        &mut self,
        gl: &glow::Context,
        maps: &mut EnvironmentMaps,
        name: &str,
        bytes: &[u8],
    ) -> Result<(), String> {
        if self.photo_name() == Some(name) {
            return Ok(());
        }
        let shown = Equirect::decode(bytes)?;
        let analysed = shown.clone().analyse();
        maps.load_photo(gl, &analysed.image, &shown);
        self.photo = Some(Photo { sun: analysed.sun, sh: analysed.sh });
        self.built = Built::Photo(name.to_string());
        Ok(())
    }

    /// Whether the photograph on the GPU has a sun in it.
    pub fn photo_has_sun(&self) -> bool {
        self.photo.is_some_and(|p| p.sun.is_some())
    }

    /// The lighting for the camera at `cam_pos`. Its shadow and occlusion fields are
    /// left for the frame to fill in.
    pub fn lighting(&self, s: &EnvironmentSettings, cam_pos: Vec3) -> Lighting {
        let ev = 2f32.powf(s.exposure_ev);
        let base = Lighting {
            sun_radius: (s.sun_size_deg * 0.5).to_radians(),
            shadow_softness: s.shadow_softness,
            exposure: s.sky.exposure() * ev,
            ..Lighting::procedural(s.sky, cam_pos, 0.0)
        };
        let (Some(photo), Built::Photo(name)) = (&self.photo, &self.built) else {
            return base;
        };
        let rotation = s.rotation_deg.to_radians();
        let (sun_dir, sun_color, sun_radius) = match photo.sun {
            Some(sun) => (
                hdri::from_env(sun.dir, rotation),
                sun.irradiance * (s.intensity * s.photo_sun),
                sun.angular_radius.clamp(0.0046, 0.03),
            ),
            None => (Vec3::Y, Vec3::ZERO, 0.0046),
        };
        Lighting {
            photo: true,
            sh: photo.sh,
            env_rotation: rotation,
            env_intensity: s.intensity,
            sun_dir,
            sun_color,
            sun_radius,
            // A photograph comes already exposed, and is shown as it is rather than
            // metered on the scene, which would blow a bright sky out.
            exposure: ev * 2f32.powf(hdri::staging(name).exposure),
            ..base
        }
    }
}

/// The sun's direction from its elevation and azimuth in degrees, y up.
pub fn sun_direction(elevation_deg: f32, azimuth_deg: f32) -> Vec3 {
    let (el, az) = (elevation_deg.to_radians(), azimuth_deg.to_radians());
    Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos())
}
