use serde::{Deserialize, Serialize};

use crate::envelope::EnvelopeParams;
use crate::ranged::{key, map_array, Ranged, Scalar};

pub const PINE_RON: &str = include_str!("../../../assets/species/pine.ron");
pub const OAK_RON: &str = include_str!("../../../assets/species/oak.ron");
pub const BIRCH_RON: &str = include_str!("../../../assets/species/birch.ron");
pub const SPRUCE_RON: &str = include_str!("../../../assets/species/spruce.ron");
pub const DOUGLAS_FIR_RON: &str = include_str!("../../../assets/species/douglas_fir.ron");
pub const DOUGLAS_FIR_OPEN_RON: &str =
    include_str!("../../../assets/species/douglas_fir_open.ron");

/// Where presets saved from the viewer are kept, relative to the repository root —
/// which is where the viewer and the CLI both expect to be run from. Unlike the
/// built-in presets these are read from disk every time they are picked, so an edit to
/// one takes effect without a rebuild.
pub const CUSTOM_PRESET_DIR: &str = "assets/species/custom";

pub fn builtin_presets() -> Vec<(&'static str, &'static str)> {
    vec![
        ("pine", PINE_RON),
        ("oak", OAK_RON),
        ("birch", BIRCH_RON),
        ("spruce", SPRUCE_RON),
        ("fir", DOUGLAS_FIR_RON),
        ("fir_open", DOUGLAS_FIR_OPEN_RON),
    ]
}

/// A species as its file describes it: a kind of tree, any of whose numbers may be a
/// range for each tree grown from it to land in. `instance` gives the one tree its
/// seed lands on. See `ranged`.
pub type SpeciesTemplate = SpeciesParams<Ranged>;

/// A species file, ranges and all.
pub fn parse_template(ron_src: &str) -> Result<SpeciesTemplate, String> {
    ron::from_str(ron_src).map_err(|e| e.to_string())
}

/// The tree a species file grows at its own seed.
///
/// Any range in the file has already landed here, at the seed the file names. To grow
/// the species at another seed, set the seed on `parse_template` and take its
/// `instance`: changing `seed` on what this returns grows a different tree from the
/// numbers the file's own seed landed on, which is not a tree the species would grow.
pub fn parse_species(ron_src: &str) -> Result<SpeciesParams, String> {
    parse_template(ron_src).map(|t| t.instance())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "SpeciesParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct SpeciesParams<V = f32> {
    pub name: String,
    pub seed: u64,
    pub max_levels: u8,
    pub max_split_depth: u32,
    pub trunk: StemParams<V>,
    pub branch_levels: Vec<StemParams<V>>,
    pub gravity_multiplier: V,
    pub phototropism_multiplier: V,
    pub envelope_scale: V,
    pub leaves: LeafParams<V>,
    pub wind: WindParams<V>,
    pub mesh: MeshParams<V>,
    pub envelope: EnvelopeParams<V>,
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

impl<V: Scalar> SpeciesParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        SpeciesParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> SpeciesParams<W> {
        SpeciesParams {
            name: self.name.clone(),
            seed: self.seed,
            max_levels: self.max_levels,
            max_split_depth: self.max_split_depth,
            trunk: self.trunk.map(&key(at, "trunk"), f),
            branch_levels: self
                .branch_levels
                .iter()
                .enumerate()
                .map(|(i, level)| level.map(&key(at, &format!("branch_levels.{i}")), f))
                .collect(),
            gravity_multiplier: f(&key(at, "gravity_multiplier"), self.gravity_multiplier),
            phototropism_multiplier: f(
                &key(at, "phototropism_multiplier"),
                self.phototropism_multiplier,
            ),
            envelope_scale: f(&key(at, "envelope_scale"), self.envelope_scale),
            leaves: self.leaves.map(&key(at, "leaves"), f),
            wind: self.wind.map(&key(at, "wind"), f),
            mesh: self.mesh.map(&key(at, "mesh"), f),
            envelope: self.envelope.map(&key(at, "envelope"), f),
        }
    }
}

impl SpeciesTemplate {
    /// The one tree this species grows at its seed: every range pinned to wherever the
    /// seed lands in it, every fixed number as it is.
    pub fn instance(&self) -> SpeciesParams {
        let seed = self.seed;
        self.map("", &mut |key, v| v.land(seed, key))
    }

    /// Every number given as a range, by key — `trunk.length`, `branch_levels.0.droop` —
    /// with where this species' seed lands in it.
    pub fn landings(&self) -> Vec<(String, Ranged, f32)> {
        let seed = self.seed;
        let mut found = Vec::new();
        self.map("", &mut |key, v| {
            if v.is_range() {
                found.push((key.to_string(), v, v.land(seed, key)));
            }
        });
        found
    }
}

impl SpeciesParams {
    /// A species that grows exactly these numbers, whatever its seed.
    pub fn template(&self) -> SpeciesTemplate {
        self.map("", &mut |_, v| Ranged::Fixed(v))
    }

    /// The crown envelope as the tree is grown against it: scaled by `envelope_scale`,
    /// and stretched along the trunk to follow it when the volumes were drawn for a
    /// trunk of a different length.
    pub fn grown_envelope(&self) -> EnvelopeParams {
        let env = self.envelope.scaled(self.envelope_scale);
        if self.envelope.for_trunk_length > 0.0 {
            env.stretched(self.trunk.length / self.envelope.for_trunk_length)
        } else {
            env
        }
    }
}

/// What stops a trunk from being a cylinder.
///
/// Real boles are fluted rather than round, swell and waist along their length, and
/// carry the odd burl. All of it is shaped here as smooth functions of the angle
/// around the stem and the distance along it, because the mesher takes its normals
/// from finite differences of the radius: anything smooth gets correct shading for
/// free, and anything that is not shows up as faceting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "BarkIrregularity::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct BarkIrregularity<V = f32> {
    /// Depth of the flutes running up the stem, as a fraction of its radius.
    pub flute_depth: V,
    /// Roughly how many flutes go round. Several frequencies are mixed around this,
    /// so the cross-section does not come out as a tidy cog.
    pub flute_waves: V,
    /// How far the flutes wind around the stem, in turns per metre.
    pub flute_twist: V,
    /// Slow swelling and waisting along the length, as a fraction of the radius.
    pub swell_depth: V,
    /// Length of the longest swelling, in metres.
    pub swell_period: V,
    /// Burls per metre of stem thick enough to carry them.
    pub burl_density: V,
    /// How far a burl stands out, as a fraction of the radius.
    pub burl_depth: V,
    /// Width of a burl in metres, before it is scaled to the stem.
    pub burl_size: V,
    /// Knots per metre of stem thick enough to carry them.
    ///
    /// Where a limb was lost the bark grows over it, leaving a dimple inside a raised
    /// collar. It is the single most recognisable mark on an old bole, and nothing
    /// else in this model makes a hollow rather than a bump.
    pub knot_density: V,
    /// How deep the dimple runs, as a fraction of the stem's radius.
    pub knot_depth: V,
    /// Width of a knot in metres, before it is scaled to the stem.
    pub knot_size: V,
    /// How far below the lowest branch a stem still carries, as a multiple of its own
    /// radius, a knot may sit.
    ///
    /// A knot is the scar of a branch the tree lost, so one can only be where a branch
    /// could have been. The lowest living branch marks the bottom of that zone, and
    /// the wood a little under it is where the ones already shed used to be. Below
    /// that is clean bole that never carried a limb — which on a trunk is the stretch
    /// at eye level that anyone standing by the tree looks at hardest.
    pub knot_reach: V,
    /// Smallest stem that may carry a knot, as a multiple of the knot's own width.
    ///
    /// Growing over a lost branch takes years of wood laid on around it, so a stem no
    /// thicker than the scar is younger than the scar it would be wearing. It also
    /// keeps a knot from wrapping most of the way round a thin stem, which reads as a
    /// bite taken out of it rather than as a scar on it.
    pub knot_min_stem: V,
    /// Height of the branch bark ridge: the raised seam that runs up the parent from
    /// a crotch, where the bark of the two stems meets and is pushed out. It is the
    /// most recognisable mark a living junction leaves, and nothing else here makes it.
    pub bark_ridge: V,
    /// Swelling where a branch leaves, as a fraction of the child's radius. A real
    /// trunk thickens into every limb it carries rather than meeting it at a seam.
    pub collar_depth: V,
    /// Stems thinner than this stay clean: a twig has no room for any of it, and
    /// paying for the rings to describe it would be waste.
    pub min_radius: V,
    /// Rings per metre on stems that do carry the detail, before the ones that are
    /// not earning their place are dropped again. The skeleton is segmented for
    /// growth, far too coarsely to show a burl.
    pub rings_per_meter: V,
    /// How far the surface may move when a ring is dropped, in metres.
    ///
    /// Rings are laid down densely and then thinned against this, so a smooth stretch
    /// of bole costs what a smooth stretch should and the rings end up where the shape
    /// actually needs them. Capped at `silhouette_tolerance`, so the surface is held to
    /// one budget along the stem and around it rather than two that disagree.
    pub ring_tolerance: V,
}

impl Default for BarkIrregularity {
    fn default() -> Self {
        Self {
            flute_depth: 0.115,
            flute_waves: 4.0,
            flute_twist: 0.07,
            swell_depth: 0.085,
            swell_period: 3.2,
            burl_density: 0.4,
            burl_depth: 0.46,
            burl_size: 0.42,
            knot_density: 0.0,
            knot_depth: 0.12,
            knot_size: 0.3,
            knot_reach: 3.0,
            knot_min_stem: 1.2,
            bark_ridge: 0.0,
            collar_depth: 0.55,
            min_radius: 0.045,
            rings_per_meter: 14.0,
            ring_tolerance: 0.010,
        }
    }
}

impl<V: Scalar> BarkIrregularity<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        BarkIrregularity::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> BarkIrregularity<W> {
        BarkIrregularity {
            flute_depth: f(&key(at, "flute_depth"), self.flute_depth),
            flute_waves: f(&key(at, "flute_waves"), self.flute_waves),
            flute_twist: f(&key(at, "flute_twist"), self.flute_twist),
            swell_depth: f(&key(at, "swell_depth"), self.swell_depth),
            swell_period: f(&key(at, "swell_period"), self.swell_period),
            burl_density: f(&key(at, "burl_density"), self.burl_density),
            burl_depth: f(&key(at, "burl_depth"), self.burl_depth),
            burl_size: f(&key(at, "burl_size"), self.burl_size),
            knot_density: f(&key(at, "knot_density"), self.knot_density),
            knot_depth: f(&key(at, "knot_depth"), self.knot_depth),
            knot_size: f(&key(at, "knot_size"), self.knot_size),
            knot_reach: f(&key(at, "knot_reach"), self.knot_reach),
            knot_min_stem: f(&key(at, "knot_min_stem"), self.knot_min_stem),
            bark_ridge: f(&key(at, "bark_ridge"), self.bark_ridge),
            collar_depth: f(&key(at, "collar_depth"), self.collar_depth),
            min_radius: f(&key(at, "min_radius"), self.min_radius),
            rings_per_meter: f(&key(at, "rings_per_meter"), self.rings_per_meter),
            ring_tolerance: f(&key(at, "ring_tolerance"), self.ring_tolerance),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "MeshParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct MeshParams<V = f32> {
    pub uv_scale: V,
    /// How far a swept tube may cut the corner off the circle it stands for, in metres.
    ///
    /// A stem sweept at `n` sides misses its true radius by `r * (1 - cos(pi/n))` at
    /// every corner, so this decides the sides directly: thick stems earn more of them
    /// and twigs earn fewer, which is what a fixed count per metre of radius could not
    /// express. A quarter of a centimetre is about a pixel on a trunk filling a screen.
    pub silhouette_tolerance: V,
    pub min_radial: u32,
    /// Stems thinner than this get no bark at all.
    ///
    /// The finest twigs are a third of the triangles and two thirds of the vertices,
    /// and each is a sliver a centimetre across carrying leaf cards many times its own
    /// size. Under foliage they cannot be seen at all; bare, a few millimetres costs
    /// only the finest hairs. Zero keeps every one of them.
    pub min_bark_radius: V,
    /// Shallowest level the bark cutoffs may cull from. Stems above it are always
    /// swept, however thin.
    ///
    /// The cutoffs are for twigs: wood buried in foliage that no one can see. A thin
    /// limb is not that. It is part of the structure the crown hangs on, and dropping
    /// it leaves its foliage floating where the gaps in a crown show it. Zero lets the
    /// cutoff reach any level.
    pub cull_from_level: u32,
    pub max_radial: u32,
    /// How much wider than the bole the buttress gets where it meets the ground.
    pub root_flare: V,
    /// How far up the bole the buttress reaches.
    pub flare_height: V,
    /// Buttress roots around the foot of the trunk. A mature broadleaf stands on a
    /// handful of distinct ridges running down into the ground, not on a cone.
    pub root_count: u32,
    /// How peaked those ridges are. At 1 they are a smooth wave; higher narrows each
    /// root and opens the hollow between them, which is what reads as buttressing.
    pub root_sharpness: V,
    /// How quickly the buttress dies away with height. Higher keeps it to the foot.
    pub root_taper: V,
    /// How much the buttress lobes narrow into separate arms as they near the ground.
    /// At zero the flare stays an unbroken skirt all the way down.
    pub root_split: V,
    /// How far the roots carry on below the ground, in metres.
    ///
    /// A trunk that stops dead at the ground plane is a cut cylinder. Carrying it a
    /// little way under lets the ground hide the cap, and the roots read as going into
    /// the soil rather than being sawn off level with it.
    pub root_depth: V,
    /// Depth of the finer grooves running down each buttress root.
    pub root_grooves: V,
    pub tip_length: V,
    pub socket_flare: V,
    /// How tightly the socket flare gathers at the very foot of a branch.
    ///
    /// A limb does not widen evenly into its parent, it trumpets: nearly all of the
    /// extra girth is in the last few centimetres before the bark of the two meet.
    /// Higher values pull the flare into that last stretch, which is what stands in
    /// for a fillet where two swept tubes just intersect.
    pub socket_power: V,
    /// How much more the socket flares on the underside of a limb than on top, where
    /// a branch lays down extra wood to carry its own weight.
    pub socket_bias: V,
    /// Base name of the bark texture set under `assets/textures`.
    pub bark_texture: String,
    /// Colour of the moss and lichen that grows on the bark.
    pub moss_color: [V; 3],
    /// How far up the tree moss reaches, in metres.
    pub moss_height: V,
    /// How much of the bark it takes at its thickest. Zero is bare bark.
    pub moss_amount: V,
    /// How much darker the bark is at the foot of the tree than high in the crown.
    /// Old bark low down weathers and holds damp; new wood above it does not.
    pub bark_darken_low: V,
    /// Colour multiplier over the bark texture, for pulling a bark set to the tone a
    /// species wants without re-authoring the art.
    pub bark_tint: [V; 3],
    /// Colour dead wood weathers toward.
    ///
    /// A branch that has lost its bark, or kept it and been bleached for years, goes
    /// silver-grey whatever the living bark was, and on a conifer that is what makes
    /// the dead lower limbs read as dead at a glance rather than as bare living ones.
    pub dead_wood_color: [V; 3],
    /// How far dead wood has gone toward `dead_wood_color`, from 0 to 1. The grain
    /// of the bark is kept and only its colour is pulled across.
    pub dead_wood_weathering: V,
    /// Bare dead wood thinner than this gets no bark, in metres. Zero follows
    /// `min_bark_radius`.
    ///
    /// The cull there exists because the finest twigs are buried in foliage, and bare
    /// dead wood has none near it: the dead twigs under a pine's crown are the finest
    /// wood on the tree and among the most visible, so they want a lower threshold than
    /// the living twigs do. Bare means reaching the trunk through nothing but dead
    /// wood. A dead twig on a living limb is inside the crown, hidden as well as any
    /// living one, and takes `min_bark_radius` like them.
    pub dead_bark_radius: V,
    pub irregularity: BarkIrregularity<V>,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            uv_scale: 1.2,
            silhouette_tolerance: 0.010,
            min_radial: 3,
            min_bark_radius: 0.006,
            cull_from_level: 0,
            max_radial: 24,
            root_flare: 0.9,
            flare_height: 1.4,
            root_count: 3,
            root_sharpness: 1.0,
            root_taper: 2.0,
            root_split: 0.0,
            root_depth: 0.0,
            root_grooves: 0.0,
            tip_length: 0.06,
            socket_flare: 0.35,
            socket_power: 2.0,
            socket_bias: 0.0,
            bark_texture: "bark".to_string(),
            moss_color: [0.20, 0.27, 0.13],
            moss_height: 0.0,
            moss_amount: 0.0,
            bark_darken_low: 0.0,
            bark_tint: [1.0, 1.0, 1.0],
            dead_wood_color: [0.62, 0.61, 0.58],
            dead_wood_weathering: 0.0,
            dead_bark_radius: 0.0,
            irregularity: BarkIrregularity::default(),
        }
    }
}

impl<V: Scalar> MeshParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        MeshParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> MeshParams<W> {
        MeshParams {
            uv_scale: f(&key(at, "uv_scale"), self.uv_scale),
            silhouette_tolerance: f(&key(at, "silhouette_tolerance"), self.silhouette_tolerance),
            min_radial: self.min_radial,
            min_bark_radius: f(&key(at, "min_bark_radius"), self.min_bark_radius),
            cull_from_level: self.cull_from_level,
            max_radial: self.max_radial,
            root_flare: f(&key(at, "root_flare"), self.root_flare),
            flare_height: f(&key(at, "flare_height"), self.flare_height),
            root_count: self.root_count,
            root_sharpness: f(&key(at, "root_sharpness"), self.root_sharpness),
            root_taper: f(&key(at, "root_taper"), self.root_taper),
            root_split: f(&key(at, "root_split"), self.root_split),
            root_depth: f(&key(at, "root_depth"), self.root_depth),
            root_grooves: f(&key(at, "root_grooves"), self.root_grooves),
            tip_length: f(&key(at, "tip_length"), self.tip_length),
            socket_flare: f(&key(at, "socket_flare"), self.socket_flare),
            socket_power: f(&key(at, "socket_power"), self.socket_power),
            socket_bias: f(&key(at, "socket_bias"), self.socket_bias),
            bark_texture: self.bark_texture.clone(),
            moss_color: map_array(&key(at, "moss_color"), self.moss_color, f),
            moss_height: f(&key(at, "moss_height"), self.moss_height),
            moss_amount: f(&key(at, "moss_amount"), self.moss_amount),
            bark_darken_low: f(&key(at, "bark_darken_low"), self.bark_darken_low),
            bark_tint: map_array(&key(at, "bark_tint"), self.bark_tint, f),
            dead_wood_color: map_array(&key(at, "dead_wood_color"), self.dead_wood_color, f),
            dead_wood_weathering: f(&key(at, "dead_wood_weathering"), self.dead_wood_weathering),
            dead_bark_radius: f(&key(at, "dead_bark_radius"), self.dead_bark_radius),
            irregularity: self.irregularity.map(&key(at, "irregularity"), f),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "StemParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct StemParams<V = f32> {
    pub length: V,
    pub length_variance: V,
    /// Base radius of a stem at this level growing at full vigor. The trunk uses it
    /// directly; a branch takes the smaller of this and `radius_ratio` of whatever
    /// its parent measures where it attaches.
    pub radius: V,
    /// Ceiling on a stem base radius as a fraction of its parent at the attachment
    /// point. Applied once per stem, never per segment.
    pub radius_ratio: V,
    pub taper: V,
    pub da_vinci_exponent: V,
    pub segment_length: V,
    pub curvature: V,
    pub phototropism: V,
    pub gravity: V,
    /// How much further a stem sags toward its tip than at its base.
    ///
    /// Gravity on its own bends a stem evenly, which is not how a limb behaves: the
    /// bending moment accumulates along it while the wood thins, so the last part of a
    /// long branch droops far more than the first. Zero keeps the even bend.
    pub droop: V,
    /// How far a stem jogs sideways at each node, in degrees.
    ///
    /// A shoot is not one smooth curve. The terminal bud aborts at the end of each
    /// season and a lateral takes over the axis, so the stem is a chain of short
    /// straight runs meeting at slight angles. The jog alternates sides, so it adds no
    /// net turn and a stem still goes where the rest of the model sends it; what it
    /// changes is that the wood stops reading as extruded. It matters most on the fine
    /// levels, which are short enough that nothing else bends them measurably.
    pub zigzag_deg: V,
    pub vigor_falloff: V,
    pub split_probability: V,
    /// How far a co-dominant fork leans away from the stem it splits from.
    pub split_angle_deg: V,
    /// Turning a stem of this level may bank, in radians per metre of its own length.
    /// Zero takes the model's own figure.
    ///
    /// The bank is what stops the crown pull settling into an orbit — there is always a
    /// radius at which a fixed pull supplies exactly the turn a circle needs, and a stem
    /// with turning to spare will ride it round. A limb long enough against its crown
    /// has to come back on itself to stay inside, and the result is a shepherd's crook.
    /// A species whose limbs are stiff, or long against the crown they grow in, wants
    /// less than the default.
    pub turn_bank: V,
    /// How evenly a fork divides the drive of the stem it leaves.
    ///
    /// At 0 the fork is a side branch: it takes the smaller share and the original
    /// carries on as the leader, which is what a conifer does for its whole life. At 1
    /// the two come away equal and neither is the trunk any more. A mature broadleaf
    /// does exactly that — it gives up its leader partway up and builds the crown out
    /// of three or four co-dominant limbs — and that one difference is most of what
    /// separates a rounded oak from a conical spruce.
    pub split_evenness: V,
    /// Length left behind when the crown prunes a stem on its very first segment.
    /// Those are branches born outside the crown, which on a real conifer are the
    /// dead stubs along the bare lower trunk. Zero removes them entirely.
    pub dead_stub_length: V,
    /// Share of stems at this level that the tree has lost.
    ///
    /// Every mature broadleaf carries dead wood: branches shaded out by their own
    /// neighbours that never shed. They keep their bark, carry no leaves, and end in a
    /// break. The weakest go first, so this is weighted by vigor rather than drawn
    /// evenly.
    pub dieback: V,
    /// Radius at which dead wood still stands, in metres.
    ///
    /// A dead limb thicker than this keeps its length; anything thinner snaps back
    /// toward its base, and the thinner it is the less of it is left. Without this a
    /// dead twig stands intact above the crown for ever, which is the one thing dead
    /// wood never does.
    pub snap_radius: V,
    /// Earliest point along a stem, as a fraction of its length, where it may fork.
    /// Without it a trunk can split at ground level and grow a second pole flush
    /// against the first.
    pub split_start_fraction: V,
    pub children: ChildParams<V>,
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
            droop: 0.0,
            zigzag_deg: 0.0,
            vigor_falloff: 0.5,
            split_probability: 0.04,
            split_angle_deg: 22.0,
            turn_bank: 0.0,
            split_evenness: 0.0,
            dead_stub_length: 0.0,
            dieback: 0.0,
            snap_radius: 0.0,
            split_start_fraction: 0.3,
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
            droop: 0.0,
            zigzag_deg: 0.0,
            vigor_falloff: 0.35,
            split_probability: 0.02,
            split_angle_deg: 25.0,
            turn_bank: 0.0,
            split_evenness: 0.0,
            dead_stub_length: 0.0,
            dieback: 0.0,
            snap_radius: 0.0,
            split_start_fraction: 0.35,
            children: ChildParams::default(),
        }
    }
}

impl<V: Scalar> StemParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        StemParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> StemParams<W> {
        StemParams {
            length: f(&key(at, "length"), self.length),
            length_variance: f(&key(at, "length_variance"), self.length_variance),
            radius: f(&key(at, "radius"), self.radius),
            radius_ratio: f(&key(at, "radius_ratio"), self.radius_ratio),
            taper: f(&key(at, "taper"), self.taper),
            da_vinci_exponent: f(&key(at, "da_vinci_exponent"), self.da_vinci_exponent),
            segment_length: f(&key(at, "segment_length"), self.segment_length),
            curvature: f(&key(at, "curvature"), self.curvature),
            phototropism: f(&key(at, "phototropism"), self.phototropism),
            gravity: f(&key(at, "gravity"), self.gravity),
            droop: f(&key(at, "droop"), self.droop),
            zigzag_deg: f(&key(at, "zigzag_deg"), self.zigzag_deg),
            vigor_falloff: f(&key(at, "vigor_falloff"), self.vigor_falloff),
            split_probability: f(&key(at, "split_probability"), self.split_probability),
            split_angle_deg: f(&key(at, "split_angle_deg"), self.split_angle_deg),
            turn_bank: f(&key(at, "turn_bank"), self.turn_bank),
            split_evenness: f(&key(at, "split_evenness"), self.split_evenness),
            dead_stub_length: f(&key(at, "dead_stub_length"), self.dead_stub_length),
            dieback: f(&key(at, "dieback"), self.dieback),
            snap_radius: f(&key(at, "snap_radius"), self.snap_radius),
            split_start_fraction: f(&key(at, "split_start_fraction"), self.split_start_fraction),
            children: self.children.map(&key(at, "children"), f),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "ChildParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct ChildParams<V = f32> {
    pub pattern: ChildPattern<V>,
    pub start_fraction: V,
    pub end_fraction: V,
    pub crotch_angle_deg: V,
    /// Angle to use at the very tip of the parent. Conifer branches stand more
    /// upright the nearer the leader they are, which is what draws the crown to a
    /// spire; one angle for the whole stem gives a pincushion instead. The blend is
    /// weighted hard toward the tip so the body of the crown keeps the angle
    /// `crotch_angle_deg` asks for. Leave the two equal for a stem whose children
    /// all leave at one angle.
    pub crotch_angle_tip_deg: V,
    pub crotch_variance_deg: V,
    pub roll_variance_deg: V,
    pub phyllotaxis_deg: V,
    pub scale: V,
    /// Spread of `scale` between siblings. Without it every branch in a whorl gets
    /// the same drive and so the same length, which reads as a wheel spoke pattern
    /// rather than a tree.
    pub scale_variance: V,
    /// How unequally siblings share the drive going into them.
    ///
    /// `scale_variance` spreads them symmetrically, which keeps every branch close to
    /// the mean and gives a crown of near-clones — a bottle brush. A real crown is a
    /// few limbs that won and a great many that were suppressed, so the draw here is
    /// skewed: at 1 most children come away well under the mean and a handful come
    /// away at several times it, while the average is unchanged. That hierarchy is
    /// what the eye reads as a tree having competed for its shape.
    pub dominance: V,
    /// How much of a parent's drive goes to the children near its tip rather than its
    /// base, from -1 to 1.
    ///
    /// Temperate broadleaves are acrotonic: the strongest shoots of a season form at
    /// the distal end of what grew last season, which is what carries a crown outward
    /// and leaves the inside of it open. At 0 the drive is spread evenly down the
    /// parent and the foliage comes out as a band along every limb instead. Negative
    /// favours the base, which is what a stem whose lower branches have had the most
    /// years to grow wants.
    pub acrotony: V,
    /// How far children are pulled into the flat plane of the limb carrying them.
    /// Conifer branchlets grow in a plane, and the flat sprays that makes are most
    /// of what gives a fir its layered silhouette; at 0 they spiral around the limb
    /// instead. The first branch off the trunk sets the plane, everything deeper on
    /// that limb shares it.
    pub planarity: V,
    /// Whorls vary by up to this many branches either way, and may come out empty,
    /// which is what breaks the ladder rhythm of a whorl on every single node.
    pub count_variance: u32,
    /// Share of the parent, from its base, whose children the crown has shaded out.
    ///
    /// A tree grows its lowest limbs first and then overtops them: the crown above
    /// takes their light and they die where they stand, keeping their wood and their
    /// twigs but none of their foliage. On the trunk this is the band of dead limbs
    /// under the living crown of a pine; on a limb it is the bare, dead-twigged inside
    /// of the limb with the foliage carried out at its end. Measured along the length
    /// the parent actually grew, not the length it set out to, because a limb the
    /// envelope cut short is shaded along what it has. Zero kills nothing.
    pub shade_line: V,
    /// Depth of the transition under `shade_line`, as a share of the parent. At the
    /// shade line every child still lives; this far below it every child is dead, and
    /// in between the odds run evenly from one to the other, so the lowest living limb
    /// is not a ruled line across the tree. Zero is a hard line.
    pub shade_blend: V,
    /// How much of its length a shaded-out child keeps, at the parent's base.
    ///
    /// A limb that died years ago stopped growing when it died, and its thin outer end
    /// has broken off since, so the dead limbs of a forest pine are shorter than the
    /// living ones above them — and the oldest, lowest ones shortest of all. This is the
    /// share kept by a child at the very base of the parent; it runs up to all of it
    /// at the shade line, where the limbs have only just died. At 1 nothing is cut back
    /// and the dead band is as long as the tree grew it.
    pub shade_keep: V,
    /// How far a shaded-out child may settle down about its base once it has died, in
    /// degrees.
    ///
    /// A living limb is held up by the wood it lays down each year against its own
    /// weight; a dead one lays down nothing and settles. The band of dead limbs under a
    /// pine's crown is not the tidy radiating rack the crown's live limbs make — some
    /// stand where they grew, some hang, a few have dropped to hang almost against the
    /// trunk. Each dead limb tilts rigidly about its attachment by a share of this,
    /// skewed so most settle a little and a few settle hard. Zero leaves them as grown.
    pub dead_sag_deg: V,
    /// Longest a branch leaving this stem may grow, as a multiple of the stem still to
    /// come past the point it leaves. Holds side branches and forks alike.
    ///
    /// A side shoot is no older than the length its parent went on to grow past it — the
    /// bud was set when the tip was there — and grows no faster than the axis it comes
    /// off, so it cannot be longer than what lies ahead of it. It is the same from the
    /// other side: out towards the tip there is less and less branch to carry a child's
    /// weight, and a long limb hanging off the last metre of a branch reads as one the
    /// branch could never have held up. At 1 nothing outgrows the stem ahead of it;
    /// higher lets children near the tip run longer, lower draws them in further.
    ///
    /// Zero uses the model's default: 1 off a branch, and no limit off the trunk, whose
    /// limbs the crown envelope shapes instead — a broadleaf's limbs rightly outreach the
    /// leader above them.
    pub tip_reach: V,
}

impl Default for ChildParams {
    fn default() -> Self {
        Self {
            pattern: ChildPattern::None,
            start_fraction: 0.2,
            end_fraction: 0.95,
            crotch_angle_deg: 45.0,
            crotch_angle_tip_deg: 45.0,
            crotch_variance_deg: 8.0,
            roll_variance_deg: 10.0,
            phyllotaxis_deg: 137.5,
            scale: 0.5,
            scale_variance: 0.0,
            dominance: 0.0,
            acrotony: 0.0,
            planarity: 0.0,
            count_variance: 0,
            shade_line: 0.0,
            shade_blend: 0.0,
            shade_keep: 1.0,
            dead_sag_deg: 0.0,
            tip_reach: 0.0,
        }
    }
}

impl<V: Scalar> ChildParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        ChildParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> ChildParams<W> {
        ChildParams {
            pattern: self.pattern.map(&key(at, "pattern"), f),
            start_fraction: f(&key(at, "start_fraction"), self.start_fraction),
            end_fraction: f(&key(at, "end_fraction"), self.end_fraction),
            crotch_angle_deg: f(&key(at, "crotch_angle_deg"), self.crotch_angle_deg),
            crotch_angle_tip_deg: f(&key(at, "crotch_angle_tip_deg"), self.crotch_angle_tip_deg),
            crotch_variance_deg: f(&key(at, "crotch_variance_deg"), self.crotch_variance_deg),
            roll_variance_deg: f(&key(at, "roll_variance_deg"), self.roll_variance_deg),
            phyllotaxis_deg: f(&key(at, "phyllotaxis_deg"), self.phyllotaxis_deg),
            scale: f(&key(at, "scale"), self.scale),
            scale_variance: f(&key(at, "scale_variance"), self.scale_variance),
            dominance: f(&key(at, "dominance"), self.dominance),
            acrotony: f(&key(at, "acrotony"), self.acrotony),
            planarity: f(&key(at, "planarity"), self.planarity),
            count_variance: self.count_variance,
            shade_line: f(&key(at, "shade_line"), self.shade_line),
            shade_blend: f(&key(at, "shade_blend"), self.shade_blend),
            shade_keep: f(&key(at, "shade_keep"), self.shade_keep),
            dead_sag_deg: f(&key(at, "dead_sag_deg"), self.dead_sag_deg),
            tip_reach: f(&key(at, "tip_reach"), self.tip_reach),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChildPattern<V = f32> {
    None,
    Whorl { every: u32, count: u32 },
    /// Children per metre of the parent stem.
    ///
    /// A rate, not a chance: 3.0 really is three children to the metre, and changing a
    /// level's `segment_length` no longer changes how much it ramifies. It used to be
    /// the probability that one segment carried one child, which saturated at 1.0 —
    /// every value at or above that was the same value — and that ceiling was what
    /// held the presets to two or three orders of branching.
    Continuous { density: V },
}

impl<V: Scalar> ChildPattern<V> {
    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> ChildPattern<W> {
        match *self {
            ChildPattern::None => ChildPattern::None,
            ChildPattern::Whorl { every, count } => ChildPattern::Whorl { every, count },
            ChildPattern::Continuous { density } => ChildPattern::Continuous {
                density: f(&key(at, "density"), density),
            },
        }
    }
}

/// How a single-leaf texture is grown into a leaf-cluster one.
///
/// Leaves are laid down a shoot, alternating sides, splayed wide at the base and
/// closing toward the tip. Angles are measured off the length of the cell, and
/// positions are fractions of it, so a cluster describes itself in the same terms
/// whatever resolution it is generated at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LeafClusterParams {
    /// Leaves composited into one cell. More of them fills the card without costing
    /// a single triangle, which is the whole trade clustering exists to make.
    pub count: u32,
    /// Length of a leaf at the base of the shoot, as a fraction of the cell. Together
    /// with `card_length` this is what fixes the real-world size of a leaf.
    pub leaf_length: f32,
    /// Angle off the shoot at its base, and at its tip.
    pub base_angle_deg: f32,
    pub tip_angle_deg: f32,
    /// Size of a tip leaf against a base one.
    pub tip_scale: f32,
    /// Width of a placed leaf against its natural width. A conifer has no needle art
    /// of its own: below about a quarter, a broad blade narrows into one, which at
    /// card scale is all a needle is.
    pub leaf_narrow: f32,
    pub angle_variance_deg: f32,
    /// Roll around the shoot between one leaf and the next. A shoot puts its leaves
    /// on a spiral, not in two flat rows, and that spiral is what the depth terms
    /// below have to work with: at 180 degrees every leaf lands in the plane of the
    /// card and the cluster comes out as a ladder of clones.
    pub roll_deg: f32,
    pub roll_variance_deg: f32,
    /// How much a leaf is drawn as though it really pointed out of the card.
    ///
    /// The arrangement is worked out in three dimensions and then flattened, so a
    /// leaf pointing out of the cell foreshortens along its length and a blade turned
    /// edge-on narrows to a sliver. This is what gives one source leaf a whole range
    /// of shapes and the cell a ragged outline. At 0 every leaf lies flat in the card
    /// at full size, which is the ladder again.
    pub depth: f32,
    /// How far a leaf may twist about its own stalk. Blades that all lie in one plane
    /// narrow together as the shoot rolls; spreading the twist keeps some of them
    /// broad wherever they point.
    pub blade_twist_deg: f32,
    /// Darkening of the leaves that point away from the viewer, which is the only
    /// depth cue a flat card has once the arrangement is flattened into it.
    pub depth_shade: f32,
    /// Spread of leaf size within one cluster, as a fraction.
    pub size_variance: f32,
    /// How irregularly the leaves are spaced along the shoot, in steps: at 0.5 a leaf
    /// may sit halfway toward either neighbour.
    pub spacing_variance: f32,
    /// How far the shoot leans across the cell by its tip, as a fraction of the cell.
    pub shoot_curve: f32,
    /// Separate arrangements baked one under another, for the renderer to pick
    /// between per card. One cluster repeated over a whole canopy is visible as a
    /// motif however the cards are turned; a handful of them is not.
    pub variants: u32,
    /// Shoots the cell is built from, side by side.
    ///
    /// One shoot can only ever fill a strip as wide as twice a leaf is long, so a
    /// species whose leaves are small against its card gets a narrow column of
    /// overlapping foliage down the middle and empty corners — a solid, scalloped
    /// slab rather than a spray. Standing two or three shoots across the cell is what
    /// a card of small leaves needs, and it is what an artist drawing one would do.
    pub shoots: u32,
    /// Where along the cell the first and last leaves attach, 1.0 being the base.
    pub shoot_base: f32,
    pub shoot_tip: f32,
    /// Width against length of the card the source leaf texture was authored for.
    /// Leaves are rotated in pixels, so a cell whose pixels are not square in world
    /// terms has to be stretched first or every rotated leaf comes out sheared.
    pub source_aspect: f32,
    /// Alternative leaves stacked down the source, one `atlas_cols` x `atlas_rows` grid
    /// each, and every leaf placed in the cluster picks one of them.
    ///
    /// A photographed set comes as several different sprays rather than one leaf, and a
    /// tuft built from one of them repeated reads as that spray stamped round a point.
    /// Drawing each leaf from the whole set is what makes a cluster look gathered
    /// rather than cloned. One is a single source, as before.
    pub sources: u32,
    /// Resolution of one generated cell. Cells are square, so a species that clusters
    /// wants a square card as well.
    pub cell_size: u32,
    pub seed: u64,
}

impl Default for LeafClusterParams {
    fn default() -> Self {
        Self {
            count: 24,
            leaf_length: 0.45,
            base_angle_deg: 72.0,
            tip_angle_deg: 26.0,
            tip_scale: 0.55,
            leaf_narrow: 1.0,
            angle_variance_deg: 11.0,
            roll_deg: 137.5,
            roll_variance_deg: 25.0,
            depth: 0.8,
            blade_twist_deg: 55.0,
            depth_shade: 0.45,
            size_variance: 0.28,
            spacing_variance: 0.45,
            shoot_curve: 0.06,
            shoots: 1,
            variants: 1,
            shoot_base: 0.99,
            shoot_tip: 0.26,
            source_aspect: 1.0,
            sources: 1,
            cell_size: 1024,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "LeafParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct LeafParams<V = f32> {
    pub enabled: bool,
    /// Leaves grow on stems at this level and deeper, so the canopy sits on twigs
    /// rather than on structural limbs.
    pub min_level: u8,
    /// Nothing thicker than this carries leaves, whatever its level.
    pub max_twig_radius: V,
    /// Cluster anchors per metre of twig. Each anchor carries `cluster_size` cards.
    pub density: V,
    /// How irregular the gaps between anchors are, as a fraction of the mean gap.
    /// Evenly spaced anchors read as a pinstripe along every twig and give the whole
    /// canopy one grain; scattering them is what lets the cards bunch and leave holes.
    pub spacing_variance: V,
    /// Metres back from the tip of a twig that carry leaves, 0 for all of it.
    ///
    /// A tree bears its leaves on the shoots it grew this year, so the foliage is a
    /// shell over bare branchwork rather than a solid volume. Without this the crown
    /// fills in solid to the trunk: every limb is buried, the silhouette is one dome,
    /// and none of the light and shade that comes of masses standing apart survives.
    pub leafy_length: V,
    /// Leaves emitted together at one point on a twig. Real foliage grows in tufts,
    /// and clumping the cards gives a canopy of masses and gaps instead of a uniform
    /// spray, for the same number of triangles.
    pub cluster_size: u32,
    /// How far the cards in one cluster fan out from its shared direction.
    pub cluster_spread_deg: V,
    pub card_length: V,
    pub card_width: V,
    pub size_variance: V,
    /// Angle between the leaf and the twig it grows from.
    pub crotch_angle_deg: V,
    pub crotch_variance_deg: V,
    /// Roll between successive leaves around the twig.
    pub phyllotaxis_deg: V,
    /// How far the blade hangs under its own weight.
    pub droop_deg: V,
    /// Random roll of the blade about its own length.
    pub twist_deg: V,
    /// How evenly the cards of one cluster are rolled about their own length, 0 to 1.
    ///
    /// At 0 each card takes its own random roll within `twist_deg`. At 1 they are
    /// spread evenly round instead, 180 degrees over the cluster, with the whole cluster
    /// turned to a random angle: two cards make a cross, three a star. It is what lets a
    /// tuft of needles be two cards rather than four — two rolled at random line up
    /// often enough that the tuft goes thin seen from one side.
    pub even_roll: V,
    /// How far the shading normal leans from the flat card toward the outward
    /// direction of the crown. This is what makes a pile of quads light like a
    /// canopy; at 0 every leaf shades as the flat plane it really is.
    pub normal_blend: V,
    /// Bend applied to the normal across the width of a card, for a rounded blade.
    pub curvature: V,
    /// Flat colour multiplier for the whole canopy, for pulling a leaf texture to
    /// the colour a species wants without re-authoring the art.
    pub tint: [V; 3],
    pub hue_variance: V,
    /// Darkening applied to leaves deep inside the crown.
    pub interior_shade: V,
    /// How crisp the cutout edge stays as a card shrinks on screen, from 0 to 1.
    ///
    /// At 0 the filtered alpha is handed straight to alpha-to-coverage, which is soft:
    /// a broad leaf keeps a clean outline and overlapping cards blend into a mass. Fine
    /// art — needles a couple of texels wide — does not survive that. A few mip levels
    /// down each needle has been averaged into the air around it and the foliage comes
    /// out as smudges. At 1 the alpha is rescaled by how fast it changes across the
    /// screen, so every edge is resolved to about a pixel whatever the mip, and the
    /// strands stay strands.
    pub edge_sharpness: V,
    /// How far a card seen from behind keeps the crown's outward shading, from 0 to 1.
    ///
    /// A card is one quad drawn from both sides, and seen from behind its shading normal
    /// is turned round with it. For a broad leaf that is right: its underside faces
    /// into the crown. But the normal has been leaned toward the outside of the crown
    /// (`normal_blend`), and turning all of it round leans it inward instead, so every
    /// card seen from its back shades as though buried. Half a canopy of randomly
    /// rolled cards goes near black that way. A tuft of needles has no front and back —
    /// it is a volume — so at 1 only the card's own flat share of the normal is turned
    /// and the outward share stays put.
    pub backface_volume: V,
    /// How dark foliage in shadow goes, from 0 to 1.
    ///
    /// The shadow map is a hard yes or no, and a card behind other foliage gets the
    /// no: all of the sun taken away, and only the sky left. A broad leaf really does
    /// throw a shadow that dense. A crown of needles does not — light comes through it
    /// in a thousand gaps — and holding every card behind another to the sky alone
    /// makes the shaded side of the crown go black where it should stay green. Below 1
    /// that much of the sun is let through the shadow.
    pub self_shadow: V,
    /// When set, the leaf texture named below is a single leaf, and the atlas the
    /// renderer actually samples is generated from it at load. `atlas_cols` and
    /// `atlas_rows` describe both, since clustering maps each source cell to one
    /// cluster cell.
    pub cluster: Option<LeafClusterParams>,
    pub atlas_cols: u32,
    pub atlas_rows: u32,
    /// Atlas cells, counted left to right then top to bottom, for the lit face and
    /// the underside of a leaf.
    pub atlas_front: u32,
    pub atlas_back: u32,
    /// Base name of the texture set under `assets/textures`.
    pub texture: String,
}

impl Default for LeafParams {
    fn default() -> Self {
        Self {
            enabled: false,
            min_level: 2,
            max_twig_radius: 0.08,
            density: 14.0,
            spacing_variance: 0.6,
            leafy_length: 0.0,
            cluster_size: 3,
            cluster_spread_deg: 38.0,
            card_length: 0.22,
            card_width: 0.16,
            size_variance: 0.25,
            crotch_angle_deg: 55.0,
            crotch_variance_deg: 18.0,
            phyllotaxis_deg: 137.5,
            droop_deg: 22.0,
            twist_deg: 35.0,
            even_roll: 0.0,
            normal_blend: 0.55,
            curvature: 0.35,
            tint: [1.0, 1.0, 1.0],
            hue_variance: 0.12,
            interior_shade: 0.35,
            edge_sharpness: 0.0,
            backface_volume: 0.0,
            self_shadow: 1.0,
            cluster: None,
            atlas_cols: 1,
            atlas_rows: 1,
            atlas_front: 0,
            atlas_back: 0,
            texture: "leaf".to_string(),
        }
    }
}

impl<V: Scalar> LeafParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        LeafParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> LeafParams<W> {
        LeafParams {
            enabled: self.enabled,
            min_level: self.min_level,
            max_twig_radius: f(&key(at, "max_twig_radius"), self.max_twig_radius),
            density: f(&key(at, "density"), self.density),
            spacing_variance: f(&key(at, "spacing_variance"), self.spacing_variance),
            leafy_length: f(&key(at, "leafy_length"), self.leafy_length),
            cluster_size: self.cluster_size,
            cluster_spread_deg: f(&key(at, "cluster_spread_deg"), self.cluster_spread_deg),
            card_length: f(&key(at, "card_length"), self.card_length),
            card_width: f(&key(at, "card_width"), self.card_width),
            size_variance: f(&key(at, "size_variance"), self.size_variance),
            crotch_angle_deg: f(&key(at, "crotch_angle_deg"), self.crotch_angle_deg),
            crotch_variance_deg: f(&key(at, "crotch_variance_deg"), self.crotch_variance_deg),
            phyllotaxis_deg: f(&key(at, "phyllotaxis_deg"), self.phyllotaxis_deg),
            droop_deg: f(&key(at, "droop_deg"), self.droop_deg),
            twist_deg: f(&key(at, "twist_deg"), self.twist_deg),
            even_roll: f(&key(at, "even_roll"), self.even_roll),
            normal_blend: f(&key(at, "normal_blend"), self.normal_blend),
            curvature: f(&key(at, "curvature"), self.curvature),
            tint: map_array(&key(at, "tint"), self.tint, f),
            hue_variance: f(&key(at, "hue_variance"), self.hue_variance),
            interior_shade: f(&key(at, "interior_shade"), self.interior_shade),
            edge_sharpness: f(&key(at, "edge_sharpness"), self.edge_sharpness),
            backface_volume: f(&key(at, "backface_volume"), self.backface_volume),
            self_shadow: f(&key(at, "self_shadow"), self.self_shadow),
            cluster: self.cluster.clone(),
            atlas_cols: self.atlas_cols,
            atlas_rows: self.atlas_rows,
            atlas_front: self.atlas_front,
            atlas_back: self.atlas_back,
            texture: self.texture.clone(),
        }
    }
}

/// How a species gives to the wind.
///
/// Only the tree's side of it lives here. How hard the wind blows, from where and how
/// gustily is the weather, which belongs to the scene the tree stands in rather than to
/// the tree, and the renderer sets it. A species file that still carries `strength` or
/// `gustiness` from before the two were split parses as it always did; the fields are
/// ignored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    default = "WindParams::defaults",
    bound(deserialize = "V: Scalar + Deserialize<'de>")
)]
pub struct WindParams<V = f32> {
    /// How far each order of wood bends in a full gale, in radians: the trunk, the limbs
    /// off it, the branches off those, and every finer twig as one. Finer wood is
    /// whippier, so these climb.
    pub flexibility: [V; 4],
    /// How far a leaf card flutters about where it hangs from its twig, in radians in a
    /// full gale.
    pub flutter: V,
    /// How fast the trunk sways, in hertz. Each finer order swings faster than the one
    /// carrying it. A tall conifer is slow — a fifty-metre fir goes back and forth about
    /// once every five seconds — and a birch several times quicker.
    pub frequency: V,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            flexibility: [0.05, 0.15, 0.35, 0.8],
            flutter: 0.5,
            frequency: 0.4,
        }
    }
}

impl<V: Scalar> WindParams<V> {
    /// The defaults, as either kind of number.
    pub fn defaults() -> Self {
        WindParams::default().map("", &mut |_, v| V::fixed(v))
    }

    /// Every number in this passed through `f`, which is told the key each is known by.
    pub fn map<W>(&self, at: &str, f: &mut impl FnMut(&str, V) -> W) -> WindParams<W> {
        WindParams {
            flexibility: map_array(&key(at, "flexibility"), self.flexibility, f),
            flutter: f(&key(at, "flutter"), self.flutter),
            frequency: f(&key(at, "frequency"), self.frequency),
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
        // As the viewer saves them: the species, ranges and all.
        for (_, src) in builtin_presets() {
            let params = parse_template(src).unwrap();
            let serial =
                ron::ser::to_string_pretty(&params, ron::ser::PrettyConfig::default()).unwrap();
            let reparsed =
                parse_template(&serial).unwrap_or_else(|e| panic!("reparse failed: {e}\n{serial}"));
            assert_eq!(params, reparsed);
        }
    }

    #[test]
    fn a_species_without_ranges_is_the_tree_it_always_was() {
        // Nothing lands anywhere, so every seed grows from the numbers in the file.
        for (name, src) in builtin_presets() {
            let mut species = parse_template(src).unwrap();
            let own = species.instance();
            assert_eq!(own.template().instance(), own, "{name}");
            species.seed = 12345;
            let other = species.instance();
            assert_eq!(SpeciesParams { seed: own.seed, ..other }, own, "{name}");
        }
    }

    #[test]
    fn any_number_in_a_species_may_be_a_range() {
        // Every number in a preset turned into a range, written out and read back: the
        // file format has to carry one wherever a number goes, nested levels, arrays
        // and the child pattern included.
        let mut n = 0;
        let spread: SpeciesTemplate = parse_species(OAK_RON).unwrap().map("", &mut |_, v| {
            n += 1;
            Ranged::Between(v, v + 1.0)
        });
        let text = ron::ser::to_string_pretty(&spread, ron::ser::PrettyConfig::default()).unwrap();
        let back = parse_template(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(back, spread);
        assert_eq!(back.landings().len(), n);
        // And each is drawn on its own: no two numbers share a key.
        let mut keys: Vec<String> = back.landings().into_iter().map(|(k, _, _)| k).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), n);
        assert!(keys.contains(&"trunk.length".to_string()));
        assert!(keys.contains(&"branch_levels.0.children.pattern.density".to_string()));
        assert!(keys.contains(&"leaves.tint.1".to_string()));
    }

    #[test]
    fn a_range_lands_by_seed_and_leaves_the_rest_alone() {
        let src = "(seed: 3, trunk: (length: (12.0, 18.0), radius: 0.4))";
        let mut species = parse_template(src).unwrap();
        let heights: Vec<f32> = (0..16)
            .map(|seed| {
                species.seed = seed;
                let tree = species.instance();
                assert_eq!(tree.trunk.radius, 0.4);
                tree.trunk.length
            })
            .collect();
        assert!(heights.iter().all(|h| (12.0..=18.0).contains(h)), "{heights:?}");
        let (lo, hi) = heights.iter().fold((f32::MAX, f32::MIN), |(a, b), &h| (a.min(h), b.max(h)));
        assert!(hi - lo > 3.0, "sixteen seeds all landed between {lo} and {hi}");
        // parse_species is the file's own seed.
        let own = parse_species(src).unwrap();
        species.seed = 3;
        assert_eq!(own.trunk.length, species.instance().trunk.length);
        assert_eq!(species.landings()[0].0, "trunk.length");
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
            ron::from_str("(frequency: 0.3, flexibility: (0.1, 0.2, 0.3, 0.4))").unwrap();
        assert_eq!(w.flexibility[1], 0.2);
        assert_eq!(w.frequency, 0.3);
    }

    #[test]
    fn a_species_saved_before_the_weather_moved_out_still_parses() {
        // The wind's strength and gustiness used to be written into every species, and
        // a preset saved then must keep loading.
        let w: WindParams = ron::from_str(
            "(strength: 0.35, gustiness: 0.4, flutter: 0.7, flexibility: (0.1, 0.2, 0.3, 0.4))",
        )
        .unwrap();
        assert_eq!(w.flutter, 0.7);
        assert_eq!(w.frequency, WindParams::default().frequency);
    }
}
