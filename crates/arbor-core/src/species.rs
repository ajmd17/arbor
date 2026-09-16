use serde::{Deserialize, Serialize};

use crate::envelope::EnvelopeParams;

pub const PINE_RON: &str = include_str!("../../../assets/species/pine.ron");
pub const OAK_RON: &str = include_str!("../../../assets/species/oak.ron");

pub fn builtin_presets() -> Vec<(&'static str, &'static str)> {
    vec![("pine", PINE_RON), ("oak", OAK_RON)]
}

pub fn parse_species(ron_src: &str) -> Result<SpeciesParams, String> {
    ron::from_str(ron_src).map_err(|e| e.to_string())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeciesParams {
    pub name: String,
    pub seed: u64,
    pub max_levels: u8,
    pub max_split_depth: u32,
    pub trunk: StemParams,
    pub branch_levels: Vec<StemParams>,
    pub gravity_multiplier: f32,
    pub phototropism_multiplier: f32,
    pub envelope_scale: f32,
    pub leaves: LeafParams,
    pub wind: WindParams,
    pub mesh: MeshParams,
    pub envelope: EnvelopeParams,
}

impl Default for SpeciesParams {
    fn default() -> Self {
        Self {
            name: "generic".to_string(),
            seed: 1,
            max_levels: 4,
            max_split_depth: 2,
            trunk: StemParams::default(),
            branch_levels: vec![StemParams::branch_default(1)],
            gravity_multiplier: 1.0,
            phototropism_multiplier: 1.0,
            envelope_scale: 1.0,
            leaves: LeafParams::default(),
            wind: WindParams::default(),
            mesh: MeshParams::default(),
            envelope: EnvelopeParams::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeshParams {
    pub uv_scale: f32,
    pub radial_per_meter: f32,
    pub min_radial: u32,
    pub max_radial: u32,
    pub root_flare: f32,
    pub flare_height: f32,
    pub tip_length: f32,
    pub socket_flare: f32,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            uv_scale: 1.2,
            radial_per_meter: 44.0,
            min_radial: 6,
            max_radial: 24,
            root_flare: 0.9,
            flare_height: 1.4,
            tip_length: 0.06,
            socket_flare: 0.35,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StemParams {
    pub length: f32,
    pub length_variance: f32,
    pub radius: f32,
    pub radius_ratio: f32,
    pub taper: f32,
    pub da_vinci_exponent: f32,
    pub segment_length: f32,
    pub curvature: f32,
    pub phototropism: f32,
    pub gravity: f32,
    pub vigor_falloff: f32,
    pub split_probability: f32,
    pub children: ChildParams,
}

impl StemParams {
    pub fn branch_default(level: u8) -> Self {
        let scale = 0.6f32.powi(level as i32);
        Self {
            length: 4.0 * scale.max(0.15),
            length_variance: 0.25,
            radius: 0.1,
            radius_ratio: 0.6,
            taper: 0.25,
            da_vinci_exponent: 2.3,
            segment_length: 0.4,
            curvature: 0.03,
            phototropism: 0.015,
            gravity: 0.012,
            vigor_falloff: 0.5,
            split_probability: 0.04,
            children: ChildParams::default(),
        }
    }
}

impl Default for StemParams {
    fn default() -> Self {
        Self {
            length: 10.0,
            length_variance: 0.12,
            radius: 0.3,
            radius_ratio: 0.62,
            taper: 0.3,
            da_vinci_exponent: 2.4,
            segment_length: 0.55,
            curvature: 0.015,
            phototropism: 0.03,
            gravity: 0.005,
            vigor_falloff: 0.35,
            split_probability: 0.02,
            children: ChildParams::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChildParams {
    pub pattern: ChildPattern,
    pub start_fraction: f32,
    pub end_fraction: f32,
    pub crotch_angle_deg: f32,
    pub crotch_variance_deg: f32,
    pub roll_variance_deg: f32,
    pub phyllotaxis_deg: f32,
    pub scale: f32,
}

impl Default for ChildParams {
    fn default() -> Self {
        Self {
            pattern: ChildPattern::None,
            start_fraction: 0.2,
            end_fraction: 0.95,
            crotch_angle_deg: 45.0,
            crotch_variance_deg: 8.0,
            roll_variance_deg: 10.0,
            phyllotaxis_deg: 137.5,
            scale: 0.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChildPattern {
    None,
    Whorl { every: u32, count: u32 },
    Continuous { density: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LeafParams {
    pub enabled: bool,
    pub card_size: f32,
    pub density: f32,
    pub hue_variance: f32,
    pub atlas_col: u32,
    pub atlas_row: u32,
}

impl Default for LeafParams {
    fn default() -> Self {
        Self {
            enabled: false,
            card_size: 0.35,
            density: 0.7,
            hue_variance: 0.1,
            atlas_col: 0,
            atlas_row: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindParams {
    pub strength: f32,
    pub gustiness: f32,
    pub flutter: f32,
    pub flexibility: [f32; 4],
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            strength: 0.35,
            gustiness: 0.4,
            flutter: 0.5,
            flexibility: [0.05, 0.15, 0.35, 0.8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse() {
        for (_, src) in builtin_presets() {
            parse_species(src).unwrap_or_else(|e| panic!("preset failed: {e}"));
        }
    }

    #[test]
    fn ron_roundtrip_is_lossless() {
        for (_, src) in builtin_presets() {
            let params = parse_species(src).unwrap();
            let serial =
                ron::ser::to_string_pretty(&params, ron::ser::PrettyConfig::default()).unwrap();
            let reparsed =
                parse_species(&serial).unwrap_or_else(|e| panic!("reparse failed: {e}\n{serial}"));
            assert_eq!(params, reparsed);
        }
    }

    #[test]
    fn partial_ron_uses_defaults() {
        let params = parse_species("(name: \"tiny\", seed: 7)").unwrap();
        assert_eq!(params.name, "tiny");
        assert_eq!(params.seed, 7);
        assert_eq!(params.max_levels, 4);
    }

    #[test]
    fn wind_array_parses() {
        let w: WindParams =
            ron::from_str("(strength: 0.35, flexibility: (0.1, 0.2, 0.3, 0.4))").unwrap();
        assert_eq!(w.flexibility[1], 0.2);
    }
}
