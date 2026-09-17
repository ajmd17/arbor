use serde::{Deserialize, Serialize};

use crate::envelope::EnvelopeParams;

pub const PINE_RON: &str = include_str!("../../../assets/species/pine.ron");
pub const OAK_RON: &str = include_str!("../../../assets/species/oak.ron");
pub const BIRCH_RON: &str = include_str!("../../../assets/species/birch.ron");

pub fn builtin_presets() -> Vec<(&'static str, &'static str)> {
    vec![("pine", PINE_RON), ("oak", OAK_RON), ("birch", BIRCH_RON)]
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

/// What stops a trunk from being a cylinder.
///
/// Real boles are fluted rather than round, swell and waist along their length, and
/// carry the odd burl. All of it is shaped here as smooth functions of the angle
/// around the stem and the distance along it, because the mesher takes its normals
/// from finite differences of the radius: anything smooth gets correct shading for
/// free, and anything that is not shows up as faceting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BarkIrregularity {
    /// Depth of the flutes running up the stem, as a fraction of its radius.
    pub flute_depth: f32,
    /// Roughly how many flutes go round. Several frequencies are mixed around this,
    /// so the cross-section does not come out as a tidy cog.
    pub flute_waves: f32,
    /// How far the flutes wind around the stem, in turns per metre.
    pub flute_twist: f32,
    /// Slow swelling and waisting along the length, as a fraction of the radius.
    pub swell_depth: f32,
    /// Length of the longest swelling, in metres.
    pub swell_period: f32,
    /// Burls per metre of stem thick enough to carry them.
    pub burl_density: f32,
    /// How far a burl stands out, as a fraction of the radius.
    pub burl_depth: f32,
    /// Width of a burl in metres, before it is scaled to the stem.
    pub burl_size: f32,
    /// Knots per metre of stem thick enough to carry them.
    ///
    /// Where a limb was lost the bark grows over it, leaving a dimple inside a raised
    /// collar. It is the single most recognisable mark on an old bole, and nothing
    /// else in this model makes a hollow rather than a bump.
    pub knot_density: f32,
    /// How deep the dimple runs, as a fraction of the stem's radius.
    pub knot_depth: f32,
    /// Width of a knot in metres, before it is scaled to the stem.
    pub knot_size: f32,
    /// How far below the lowest branch a stem still carries, as a multiple of its own
    /// radius, a knot may sit.
    ///
    /// A knot is the scar of a branch the tree lost, so one can only be where a branch
    /// could have been. The lowest living branch marks the bottom of that zone, and
    /// the wood a little under it is where the ones already shed used to be. Below
    /// that is clean bole that never carried a limb — which on a trunk is the stretch
    /// at eye level that anyone standing by the tree looks at hardest.
    pub knot_reach: f32,
    /// Smallest stem that may carry a knot, as a multiple of the knot's own width.
    ///
    /// Growing over a lost branch takes years of wood laid on around it, so a stem no
    /// thicker than the scar is younger than the scar it would be wearing. It also
    /// keeps a knot from wrapping most of the way round a thin stem, which reads as a
    /// bite taken out of it rather than as a scar on it.
    pub knot_min_stem: f32,
    /// Height of the branch bark ridge: the raised seam that runs up the parent from
    /// a crotch, where the bark of the two stems meets and is pushed out. It is the
    /// most recognisable mark a living junction leaves, and nothing else here makes it.
    pub bark_ridge: f32,
    /// Swelling where a branch leaves, as a fraction of the child's radius. A real
    /// trunk thickens into every limb it carries rather than meeting it at a seam.
    pub collar_depth: f32,
    /// Stems thinner than this stay clean: a twig has no room for any of it, and
    /// paying for the rings to describe it would be waste.
    pub min_radius: f32,
    /// Rings per metre on stems that do carry the detail, before the ones that are
    /// not earning their place are dropped again. The skeleton is segmented for
    /// growth, far too coarsely to show a burl.
    pub rings_per_meter: f32,
    /// How far the surface may move when a ring is dropped, in metres.
    ///
    /// Rings are laid down densely and then thinned against this, so a smooth stretch
    /// of bole costs what a smooth stretch should and the rings end up where the shape
    /// actually needs them. Capped at `silhouette_tolerance`, so the surface is held to
    /// one budget along the stem and around it rather than two that disagree.
    pub ring_tolerance: f32,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeshParams {
    pub uv_scale: f32,
    /// How far a swept tube may cut the corner off the circle it stands for, in metres.
    ///
    /// A stem sweept at `n` sides misses its true radius by `r * (1 - cos(pi/n))` at
    /// every corner, so this decides the sides directly: thick stems earn more of them
    /// and twigs earn fewer, which is what a fixed count per metre of radius could not
    /// express. A quarter of a centimetre is about a pixel on a trunk filling a screen.
    pub silhouette_tolerance: f32,
    pub min_radial: u32,
    /// Stems thinner than this get no bark at all.
    ///
    /// The finest twigs are a third of the triangles and two thirds of the vertices,
    /// and each is a sliver a centimetre across carrying leaf cards many times its own
    /// size. Under foliage they cannot be seen at all; bare, a few millimetres costs
    /// only the finest hairs. Zero keeps every one of them.
    pub min_bark_radius: f32,
    pub max_radial: u32,
    /// How much wider than the bole the buttress gets where it meets the ground.
    pub root_flare: f32,
    /// How far up the bole the buttress reaches.
    pub flare_height: f32,
    /// Buttress roots around the foot of the trunk. A mature broadleaf stands on a
    /// handful of distinct ridges running down into the ground, not on a cone.
    pub root_count: u32,
    /// How peaked those ridges are. At 1 they are a smooth wave; higher narrows each
    /// root and opens the hollow between them, which is what reads as buttressing.
    pub root_sharpness: f32,
    /// How quickly the buttress dies away with height. Higher keeps it to the foot.
    pub root_taper: f32,
    /// How much the buttress lobes narrow into separate arms as they near the ground.
    /// At zero the flare stays an unbroken skirt all the way down.
    pub root_split: f32,
    /// How far the roots carry on below the ground, in metres.
    ///
    /// A trunk that stops dead at the ground plane is a cut cylinder. Carrying it a
    /// little way under lets the ground hide the cap, and the roots read as going into
    /// the soil rather than being sawn off level with it.
    pub root_depth: f32,
    /// Depth of the finer grooves running down each buttress root.
    pub root_grooves: f32,
    pub tip_length: f32,
    pub socket_flare: f32,
    /// How tightly the socket flare gathers at the very foot of a branch.
    ///
    /// A limb does not widen evenly into its parent, it trumpets: nearly all of the
    /// extra girth is in the last few centimetres before the bark of the two meet.
    /// Higher values pull the flare into that last stretch, which is what stands in
    /// for a fillet where two swept tubes just intersect.
    pub socket_power: f32,
    /// How much more the socket flares on the underside of a limb than on top, where
    /// a branch lays down extra wood to carry its own weight.
    pub socket_bias: f32,
    /// Base name of the bark texture set under `assets/textures`.
    pub bark_texture: String,
    /// Colour of the moss and lichen that grows on the bark.
    pub moss_color: [f32; 3],
    /// How far up the tree moss reaches, in metres.
    pub moss_height: f32,
    /// How much of the bark it takes at its thickest. Zero is bare bark.
    pub moss_amount: f32,
    /// How much darker the bark is at the foot of the tree than high in the crown.
    /// Old bark low down weathers and holds damp; new wood above it does not.
    pub bark_darken_low: f32,
    pub irregularity: BarkIrregularity,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            uv_scale: 1.2,
            silhouette_tolerance: 0.010,
            min_radial: 3,
            min_bark_radius: 0.006,
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
            irregularity: BarkIrregularity::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StemParams {
    pub length: f32,
    pub length_variance: f32,
    /// Base radius of a stem at this level growing at full vigor. The trunk uses it
    /// directly; a branch takes the smaller of this and `radius_ratio` of whatever
    /// its parent measures where it attaches.
    pub radius: f32,
    /// Ceiling on a stem base radius as a fraction of its parent at the attachment
    /// point. Applied once per stem, never per segment.
    pub radius_ratio: f32,
    pub taper: f32,
    pub da_vinci_exponent: f32,
    pub segment_length: f32,
    pub curvature: f32,
    pub phototropism: f32,
    pub gravity: f32,
    /// How much further a stem sags toward its tip than at its base.
    ///
    /// Gravity on its own bends a stem evenly, which is not how a limb behaves: the
    /// bending moment accumulates along it while the wood thins, so the last part of a
    /// long branch droops far more than the first. Zero keeps the even bend.
    pub droop: f32,
    /// How far a stem jogs sideways at each node, in degrees.
    ///
    /// A shoot is not one smooth curve. The terminal bud aborts at the end of each
    /// season and a lateral takes over the axis, so the stem is a chain of short
    /// straight runs meeting at slight angles. The jog alternates sides, so it adds no
    /// net turn and a stem still goes where the rest of the model sends it; what it
    /// changes is that the wood stops reading as extruded. It matters most on the fine
    /// levels, which are short enough that nothing else bends them measurably.
    pub zigzag_deg: f32,
    pub vigor_falloff: f32,
    pub split_probability: f32,
    /// How far a co-dominant fork leans away from the stem it splits from.
    pub split_angle_deg: f32,
    /// Turning a stem of this level may bank, in radians per metre of its own length.
    /// Zero takes the model's own figure.
    ///
    /// The bank is what stops the crown pull settling into an orbit — there is always a
    /// radius at which a fixed pull supplies exactly the turn a circle needs, and a stem
    /// with turning to spare will ride it round. A limb long enough against its crown
    /// has to come back on itself to stay inside, and the result is a shepherd's crook.
    /// A species whose limbs are stiff, or long against the crown they grow in, wants
    /// less than the default.
    pub turn_bank: f32,
    /// How evenly a fork divides the drive of the stem it leaves.
    ///
    /// At 0 the fork is a side branch: it takes the smaller share and the original
    /// carries on as the leader, which is what a conifer does for its whole life. At 1
    /// the two come away equal and neither is the trunk any more. A mature broadleaf
    /// does exactly that — it gives up its leader partway up and builds the crown out
    /// of three or four co-dominant limbs — and that one difference is most of what
    /// separates a rounded oak from a conical spruce.
    pub split_evenness: f32,
    /// Length left behind when the crown prunes a stem on its very first segment.
    /// Those are branches born outside the crown, which on a real conifer are the
    /// dead stubs along the bare lower trunk. Zero removes them entirely.
    pub dead_stub_length: f32,
    /// Share of stems at this level that the tree has lost.
    ///
    /// Every mature broadleaf carries dead wood: branches shaded out by their own
    /// neighbours that never shed. They keep their bark, carry no leaves, and end in a
    /// break. The weakest go first, so this is weighted by vigor rather than drawn
    /// evenly.
    pub dieback: f32,
    /// Radius at which dead wood still stands, in metres.
    ///
    /// A dead limb thicker than this keeps its length; anything thinner snaps back
    /// toward its base, and the thinner it is the less of it is left. Without this a
    /// dead twig stands intact above the crown for ever, which is the one thing dead
    /// wood never does.
    pub snap_radius: f32,
    /// Earliest point along a stem, as a fraction of its length, where it may fork.
    /// Without it a trunk can split at ground level and grow a second pole flush
    /// against the first.
    pub split_start_fraction: f32,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChildParams {
    pub pattern: ChildPattern,
    pub start_fraction: f32,
    pub end_fraction: f32,
    pub crotch_angle_deg: f32,
    /// Angle to use at the very tip of the parent. Conifer branches stand more
    /// upright the nearer the leader they are, which is what draws the crown to a
    /// spire; one angle for the whole stem gives a pincushion instead. The blend is
    /// weighted hard toward the tip so the body of the crown keeps the angle
    /// `crotch_angle_deg` asks for. Leave the two equal for a stem whose children
    /// all leave at one angle.
    pub crotch_angle_tip_deg: f32,
    pub crotch_variance_deg: f32,
    pub roll_variance_deg: f32,
    pub phyllotaxis_deg: f32,
    pub scale: f32,
    /// Spread of `scale` between siblings. Without it every branch in a whorl gets
    /// the same drive and so the same length, which reads as a wheel spoke pattern
    /// rather than a tree.
    pub scale_variance: f32,
    /// How unequally siblings share the drive going into them.
    ///
    /// `scale_variance` spreads them symmetrically, which keeps every branch close to
    /// the mean and gives a crown of near-clones — a bottle brush. A real crown is a
    /// few limbs that won and a great many that were suppressed, so the draw here is
    /// skewed: at 1 most children come away well under the mean and a handful come
    /// away at several times it, while the average is unchanged. That hierarchy is
    /// what the eye reads as a tree having competed for its shape.
    pub dominance: f32,
    /// How much of a parent's drive goes to the children near its tip rather than its
    /// base, from -1 to 1.
    ///
    /// Temperate broadleaves are acrotonic: the strongest shoots of a season form at
    /// the distal end of what grew last season, which is what carries a crown outward
    /// and leaves the inside of it open. At 0 the drive is spread evenly down the
    /// parent and the foliage comes out as a band along every limb instead. Negative
    /// favours the base, which is what a stem whose lower branches have had the most
    /// years to grow wants.
    pub acrotony: f32,
    /// How far children are pulled into the flat plane of the limb carrying them.
    /// Conifer branchlets grow in a plane, and the flat sprays that makes are most
    /// of what gives a fir its layered silhouette; at 0 they spiral around the limb
    /// instead. The first branch off the trunk sets the plane, everything deeper on
    /// that limb shares it.
    pub planarity: f32,
    /// Whorls vary by up to this many branches either way, and may come out empty,
    /// which is what breaks the ladder rhythm of a whorl on every single node.
    pub count_variance: u32,
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
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChildPattern {
    None,
    Whorl { every: u32, count: u32 },
    /// Children per metre of the parent stem.
    ///
    /// A rate, not a chance: 3.0 really is three children to the metre, and changing a
    /// level's `segment_length` no longer changes how much it ramifies. It used to be
    /// the probability that one segment carried one child, which saturated at 1.0 —
    /// every value at or above that was the same value — and that ceiling was what
    /// held the presets to two or three orders of branching.
    Continuous { density: f32 },
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
            cell_size: 1024,
            seed: 7,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LeafParams {
    pub enabled: bool,
    /// Leaves grow on stems at this level and deeper, so the canopy sits on twigs
    /// rather than on structural limbs.
    pub min_level: u8,
    /// Nothing thicker than this carries leaves, whatever its level.
    pub max_twig_radius: f32,
    /// Cluster anchors per metre of twig. Each anchor carries `cluster_size` cards.
    pub density: f32,
    /// How irregular the gaps between anchors are, as a fraction of the mean gap.
    /// Evenly spaced anchors read as a pinstripe along every twig and give the whole
    /// canopy one grain; scattering them is what lets the cards bunch and leave holes.
    pub spacing_variance: f32,
    /// Metres back from the tip of a twig that carry leaves, 0 for all of it.
    ///
    /// A tree bears its leaves on the shoots it grew this year, so the foliage is a
    /// shell over bare branchwork rather than a solid volume. Without this the crown
    /// fills in solid to the trunk: every limb is buried, the silhouette is one dome,
    /// and none of the light and shade that comes of masses standing apart survives.
    pub leafy_length: f32,
    /// Leaves emitted together at one point on a twig. Real foliage grows in tufts,
    /// and clumping the cards gives a canopy of masses and gaps instead of a uniform
    /// spray, for the same number of triangles.
    pub cluster_size: u32,
    /// How far the cards in one cluster fan out from its shared direction.
    pub cluster_spread_deg: f32,
    pub card_length: f32,
    pub card_width: f32,
    pub size_variance: f32,
    /// Angle between the leaf and the twig it grows from.
    pub crotch_angle_deg: f32,
    pub crotch_variance_deg: f32,
    /// Roll between successive leaves around the twig.
    pub phyllotaxis_deg: f32,
    /// How far the blade hangs under its own weight.
    pub droop_deg: f32,
    /// Random roll of the blade about its own length.
    pub twist_deg: f32,
    /// How far the shading normal leans from the flat card toward the outward
    /// direction of the crown. This is what makes a pile of quads light like a
    /// canopy; at 0 every leaf shades as the flat plane it really is.
    pub normal_blend: f32,
    /// Bend applied to the normal across the width of a card, for a rounded blade.
    pub curvature: f32,
    /// Flat colour multiplier for the whole canopy, for pulling a leaf texture to
    /// the colour a species wants without re-authoring the art.
    pub tint: [f32; 3],
    pub hue_variance: f32,
    /// Darkening applied to leaves deep inside the crown.
    pub interior_shade: f32,
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
            normal_blend: 0.55,
            curvature: 0.35,
            tint: [1.0, 1.0, 1.0],
            hue_variance: 0.12,
            interior_shade: 0.35,
            cluster: None,
            atlas_cols: 1,
            atlas_rows: 1,
            atlas_front: 0,
            atlas_back: 0,
            texture: "leaf".to_string(),
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
