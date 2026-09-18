//! The species controls in the side panel, kept as tables rather than as code.
//!
//! Every number the panel can edit is described once here — its label, the range its
//! slider offers, whether that range is swept logarithmically, and a line of help — and
//! the panel is drawn by walking these tables. `species_values_fit_their_sliders` walks
//! the same ones against every preset, so a control added here is under that test the
//! moment it exists rather than whenever someone remembers to add it.
//!
//! The ranges are still load-bearing, though less than they were. Every slider here is
//! drawn with `SliderClamping::Edits`, so a preset carrying a value outside its range
//! keeps it rather than having the clamped number written back over the species as the
//! panel draws — which is what egui's default does, and what once grew a 52 m douglas
//! fir as a 32 m one under a crown based at 25 m. What the range still decides is
//! whether the handle can reach the preset's own value, so it has to cover every one.

use arbor_core::species::{BarkIrregularity, ChildParams, ChildPattern, LeafParams, MeshParams, StemParams};
use arbor_core::{EnvelopeParams, SpeciesParams};
use eframe::egui;

/// One continuous control over a field of `T`.
pub struct Knob<T> {
    pub label: &'static str,
    pub range: (f32, f32),
    /// Sweep the range logarithmically, for quantities that matter by ratio rather than
    /// by difference: a radius of 2 mm against 4 mm is as big a change as 2 m against 4.
    pub log: bool,
    pub help: &'static str,
    pub get: fn(&mut T) -> &mut f32,
}

/// One whole-number control over a field of `T`.
pub struct Count<T> {
    pub label: &'static str,
    pub range: (u32, u32),
    pub help: &'static str,
    pub get: fn(&mut T) -> &mut u32,
}

pub struct Group<T: 'static> {
    pub title: &'static str,
    pub knobs: &'static [Knob<T>],
    pub counts: &'static [Count<T>],
}

const fn knob<T>(
    label: &'static str,
    range: (f32, f32),
    help: &'static str,
    get: fn(&mut T) -> &mut f32,
) -> Knob<T> {
    Knob { label, range, log: false, help, get }
}

const fn log_knob<T>(
    label: &'static str,
    range: (f32, f32),
    help: &'static str,
    get: fn(&mut T) -> &mut f32,
) -> Knob<T> {
    Knob { label, range, log: true, help, get }
}

// ---------------------------------------------------------------------------------
// Whole tree

/// Crown shape and the multipliers that act on every level at once. These are the
/// quick controls, shown open at the top of the panel.
pub static SHAPE: &[Knob<SpeciesParams>] = &[
    knob("Trunk length", (2.0, 60.0), "Declared length of the trunk, in metres. The crown envelope does not follow it, so a big change here wants Envelope scale moved with it.", |p| &mut p.trunk.length),
    knob("Envelope scale", (0.4, 2.5), "Scales the crown envelope that steers and prunes every branch.", |p| &mut p.envelope_scale),
    knob("Gravity", (0.0, 4.0), "Multiplies every level's gravity at once.", |p| &mut p.gravity_multiplier),
    knob("Phototropism", (0.0, 4.0), "Multiplies every level's pull toward the light at once.", |p| &mut p.phototropism_multiplier),
];

pub const BRANCH_LEVELS: (u32, u32) = (1, 6);

pub static SPLIT_DEPTH: Count<SpeciesParams> = Count {
    label: "Fork depth",
    range: (0, 5),
    help: "How many times a stem may fork and its forks fork again. Zero turns forking off everywhere.",
    get: |p| &mut p.max_split_depth,
};

pub static ENVELOPE: Group<EnvelopeParams> = Group {
    title: "Crown envelope",
    knobs: &[
        log_knob("Falloff", (0.01, 1.0), "How soft the envelope's edge is. Small is a hard wall; large lets branches thin out gradually toward it.", |e| &mut e.falloff),
        knob("Kill threshold", (0.0, 0.5), "Envelope density below which a growing stem is pruned. Higher prunes earlier, inside the edge.", |e| &mut e.kill_threshold),
        knob("Pull strength", (0.0, 5.0), "How hard the crown turns a stem heading out of it, in radians per metre grown.", |e| &mut e.pull_strength),
    ],
    counts: &[],
};

// ---------------------------------------------------------------------------------
// One level's growth

pub static STEM: &[Group<StemParams>] = &[
    Group {
        title: "Length & thickness",
        knobs: &[
            log_knob("Length", (0.05, 80.0), "Length a stem at this level grows at full drive, in metres. Drive and the crown envelope usually cut it shorter.", |s| &mut s.length),
            knob("Length variance", (0.0, 1.0), "Random spread of length between stems, as a fraction either way.", |s| &mut s.length_variance),
            log_knob("Radius", (0.002, 3.0), "Base radius of a stem at full drive, in metres.", |s| &mut s.radius),
            knob("Radius ratio", (0.05, 0.95), "Most a stem may measure at its base, as a share of its parent's radius where it leaves.", |s| &mut s.radius_ratio),
            knob("Taper", (0.0, 1.0), "Radius left at the tip, as a share of the base. Low tapers to a point.", |s| &mut s.taper),
            knob("Da Vinci exponent", (1.0, 4.0), "How cross-section is conserved through a junction. 2 keeps total area; higher lets children be thicker for their parent.", |s| &mut s.da_vinci_exponent),
            log_knob("Segment length", (0.02, 3.0), "Length of one growth step, in metres. Shorter bends more smoothly and costs more nodes.", |s| &mut s.segment_length),
        ],
        counts: &[],
    },
    Group {
        title: "Bending",
        knobs: &[
            knob("Curvature", (0.0, 0.5), "Random wander per segment.", |s| &mut s.curvature),
            knob("Phototropism", (0.0, 0.5), "Pull toward the light per segment. Lifts tips.", |s| &mut s.phototropism),
            knob("Gravity", (-0.2, 0.5), "Pull downward per segment, evenly along the stem. Negative lifts.", |s| &mut s.gravity),
            knob("Droop", (0.0, 10.0), "How much harder gravity bends the tip than the base. Zero is an even bend.", |s| &mut s.droop),
            knob("Zigzag", (0.0, 40.0), "Alternating jog at each node, in degrees. Breaks the extruded look without changing where the stem goes.", |s| &mut s.zigzag_deg),
            knob("Turn bank", (0.0, 2.0), "Total turning a stem may do, in radians per metre of its length. Zero uses the model's default; lower stiffens long limbs.", |s| &mut s.turn_bank),
        ],
        counts: &[],
    },
    Group {
        title: "Vigor & dieback",
        knobs: &[
            knob("Vigor falloff", (0.0, 1.0), "Drive lost from base to tip. Higher leaves the outer children weaker than the inner ones.", |s| &mut s.vigor_falloff),
            knob("Dieback", (0.0, 1.0), "Share of stems at this level that are dead wood, weakest first.", |s| &mut s.dieback),
            knob("Snap radius", (0.0, 0.3), "Dead wood thinner than this breaks back toward its base, in metres.", |s| &mut s.snap_radius),
            knob("Dead stub length", (0.0, 3.0), "Stub left where the crown prunes a stem at birth, in metres. The dead stubs on a bare lower bole.", |s| &mut s.dead_stub_length),
        ],
        counts: &[],
    },
    Group {
        title: "Forks",
        knobs: &[
            knob("Fork chance", (0.0, 1.0), "Chance per segment that the stem forks.", |s| &mut s.split_probability),
            knob("Fork angle", (0.0, 90.0), "How far a fork leans off the stem, in degrees.", |s| &mut s.split_angle_deg),
            knob("Fork evenness", (0.0, 1.0), "0: the fork is a side branch and the stem stays the leader. 1: two equal co-dominant stems.", |s| &mut s.split_evenness),
            knob("Fork start", (0.0, 1.0), "Earliest point along the stem a fork may leave, as a fraction of its length.", |s| &mut s.split_start_fraction),
        ],
        counts: &[],
    },
];

// ---------------------------------------------------------------------------------
// How one level is spawned off the level above it

pub const DENSITY: (f32, f32) = (0.05, 40.0);
pub const WHORL_EVERY: (u32, u32) = (1, 12);
pub const WHORL_COUNT: (u32, u32) = (1, 16);

pub static CHILDREN: &[Group<ChildParams>] = &[
    Group {
        title: "Placement",
        knobs: &[
            knob("Start", (0.0, 1.0), "Where along the parent children begin, as a fraction of its length.", |c| &mut c.start_fraction),
            knob("End", (0.0, 1.0), "Where along the parent children stop.", |c| &mut c.end_fraction),
            knob("Phyllotaxis", (0.0, 360.0), "Roll around the parent from one child to the next, in degrees. 137.5 spirals; 180 makes a flat two-sided spray.", |c| &mut c.phyllotaxis_deg),
            knob("Roll variance", (0.0, 90.0), "Random spread on that roll, in degrees.", |c| &mut c.roll_variance_deg),
            knob("Planarity", (0.0, 1.0), "How hard children are pulled into the flat plane of the limb carrying them. The layered spray of a fir.", |c| &mut c.planarity),
            knob("Shade line", (0.0, 1.0), "Share of the parent, from its base, whose children are dead: shaded out by the crown above. The dead lower limbs of a pine, or the bare inside of a limb.", |c| &mut c.shade_line),
            knob("Shade blend", (0.0, 1.0), "How far under the shade line the odds run from all alive to all dead, so the lowest living limb is not a ruled line.", |c| &mut c.shade_blend),
            knob("Shade keep", (0.0, 1.0), "How much of its length a shaded-out child keeps at the parent's base, rising to all of it at the shade line. Lower reels the dead band in: old dead limbs stopped growing and lost their ends.", |c| &mut c.shade_keep),
        ],
        counts: &[Count {
            label: "Count variance",
            range: (0, 8),
            help: "Whorls vary by up to this many branches either way, and may come out empty.",
            get: |c| &mut c.count_variance,
        }],
    },
    Group {
        title: "Angles",
        knobs: &[
            knob("Crotch angle", (0.0, 180.0), "Angle off the parent, in degrees. Past 90 the child leaves pointing back down.", |c| &mut c.crotch_angle_deg),
            knob("Crotch at tip", (0.0, 180.0), "Angle for children near the parent's tip. Blended in hard toward the tip, so a conifer can go from sloping limbs to an upright spire.", |c| &mut c.crotch_angle_tip_deg),
            knob("Crotch variance", (0.0, 60.0), "Random spread on the crotch angle, in degrees.", |c| &mut c.crotch_variance_deg),
        ],
        counts: &[],
    },
    Group {
        title: "Drive",
        knobs: &[
            knob("Scale", (0.01, 1.5), "Share of the parent's drive a child starts with. Sets how much smaller each level is.", |c| &mut c.scale),
            knob("Scale variance", (0.0, 1.0), "Symmetric spread of that share between siblings.", |c| &mut c.scale_variance),
            knob("Dominance", (0.0, 1.0), "How unequally siblings share the drive. High gives a few long winners and many suppressed stems.", |c| &mut c.dominance),
            knob("Acrotony", (-1.0, 1.0), "Positive favours children near the parent's tip; negative favours those near its base.", |c| &mut c.acrotony),
        ],
        counts: &[],
    },
];

// ---------------------------------------------------------------------------------
// Foliage

pub static FOLIAGE: &[Knob<LeafParams>] = &[
    log_knob("Leaves per metre", (0.5, 60.0), "Cluster anchors per metre of twig.", |l| &mut l.density),
    log_knob("Leaf length", (0.03, 2.0), "Card length, in metres.", |l| &mut l.card_length),
    log_knob("Leaf width", (0.02, 2.0), "Card width, in metres.", |l| &mut l.card_width),
    knob("Normal blend", (0.0, 1.0), "How far the shading normal leans from the flat card toward the outside of the crown.", |l| &mut l.normal_blend),
    knob("Leaf droop", (-40.0, 70.0), "How far a card hangs under its own weight, in degrees.", |l| &mut l.droop_deg),
];

pub const LEAF_MIN_LEVEL: (u32, u32) = (0, 6);

pub static FOLIAGE_MORE: &[Group<LeafParams>] = &[
    Group {
        title: "Placement",
        knobs: &[
            knob("Spacing variance", (0.0, 1.5), "How irregular the gaps between anchors are. Even spacing reads as a pinstripe.", |l| &mut l.spacing_variance),
            knob("Leafy length", (0.0, 5.0), "Metres back from each twig tip that carry leaves. Zero leafs the whole twig.", |l| &mut l.leafy_length),
            log_knob("Max twig radius", (0.001, 0.5), "Nothing thicker than this carries leaves, in metres.", |l| &mut l.max_twig_radius),
            knob("Cluster spread", (0.0, 120.0), "How far the cards of one cluster fan out, in degrees.", |l| &mut l.cluster_spread_deg),
        ],
        counts: &[Count {
            label: "Cluster size",
            range: (1, 12),
            help: "Cards emitted together at one anchor.",
            get: |l| &mut l.cluster_size,
        }],
    },
    Group {
        title: "Cards",
        knobs: &[
            knob("Size variance", (0.0, 1.0), "Random spread of card size.", |l| &mut l.size_variance),
            knob("Crotch angle", (0.0, 120.0), "Angle between a card and its twig, in degrees.", |l| &mut l.crotch_angle_deg),
            knob("Crotch variance", (0.0, 60.0), "Random spread on that angle.", |l| &mut l.crotch_variance_deg),
            knob("Phyllotaxis", (0.0, 360.0), "Roll around the twig between one card and the next, in degrees. 180 keeps a flat spray flat.", |l| &mut l.phyllotaxis_deg),
            knob("Twist", (0.0, 180.0), "Random roll of a card about its own length, in degrees.", |l| &mut l.twist_deg),
            knob("Even roll", (0.0, 1.0), "How evenly a cluster's cards are rolled round their axis. 1 spreads them 180 degrees over the cluster: two cards make a cross, three a star, and no tuft goes thin from one side.", |l| &mut l.even_roll),
        ],
        counts: &[],
    },
    Group {
        title: "Shading",
        knobs: &[
            knob("Curvature", (0.0, 1.5), "Bend of the normal across a card's width, for a rounded blade.", |l| &mut l.curvature),
            knob("Hue variance", (0.0, 1.0), "Per-card colour spread.", |l| &mut l.hue_variance),
            knob("Interior shade", (0.0, 1.0), "Darkening of cards deep inside the crown.", |l| &mut l.interior_shade),
            knob("Backface volume", (0.0, 1.0), "How far a card seen from behind keeps the crown's outward shading. 0 turns the whole normal round, right for broad leaves; 1 turns only the card's own share, for needle tufts that have no back.", |l| &mut l.backface_volume),
            knob("Self shadow", (0.0, 1.0), "How much of the sun foliage in shadow loses. 1 is all of it, right for broad leaves; a needle crown lets light through and wants less.", |l| &mut l.self_shadow),
            knob("Edge sharpness", (0.0, 1.0), "How crisp the cutout edge stays as a card shrinks. 0 is soft coverage, which blends cards into a mass; 1 keeps fine needles as strands.", |l| &mut l.edge_sharpness),
            knob("Tint red", (0.0, 1.5), "Colour multiplier for the whole canopy.", |l| &mut l.tint[0]),
            knob("Tint green", (0.0, 1.5), "Colour multiplier for the whole canopy.", |l| &mut l.tint[1]),
            knob("Tint blue", (0.0, 1.5), "Colour multiplier for the whole canopy.", |l| &mut l.tint[2]),
        ],
        counts: &[],
    },
];

// ---------------------------------------------------------------------------------
// Bark and roots

pub static BARK: &[Group<MeshParams>] = &[
    Group {
        title: "Roots & flare",
        knobs: &[
            knob("Root flare", (0.0, 4.0), "How much wider than the bole the foot gets.", |m| &mut m.root_flare),
            knob("Flare height", (0.0, 6.0), "How far up the bole the flare reaches, in metres.", |m| &mut m.flare_height),
            knob("Root sharpness", (0.5, 8.0), "How peaked the buttress ridges are. 1 is a smooth wave.", |m| &mut m.root_sharpness),
            knob("Root taper", (0.1, 8.0), "How quickly the buttress dies away with height.", |m| &mut m.root_taper),
            knob("Root split", (0.0, 6.0), "How much the buttress lobes separate into arms near the ground. Zero is an unbroken skirt.", |m| &mut m.root_split),
            knob("Root depth", (0.0, 3.0), "How far the roots carry on below the ground, in metres.", |m| &mut m.root_depth),
            knob("Root grooves", (0.0, 1.0), "Depth of the finer grooves down each root.", |m| &mut m.root_grooves),
        ],
        counts: &[Count {
            label: "Root count",
            range: (0, 12),
            help: "Buttress roots around the foot of the trunk.",
            get: |m| &mut m.root_count,
        }],
    },
    Group {
        title: "Junctions",
        knobs: &[
            knob("Socket flare", (0.0, 2.0), "How much a branch widens where it meets its parent.", |m| &mut m.socket_flare),
            knob("Socket power", (0.5, 8.0), "How tightly that flare gathers at the very foot of the branch.", |m| &mut m.socket_power),
            knob("Socket bias", (0.0, 1.0), "Extra flare on the underside of a limb.", |m| &mut m.socket_bias),
        ],
        counts: &[],
    },
    Group {
        title: "Surface",
        knobs: &[
            log_knob("UV scale", (0.1, 10.0), "Bark texture repeats per metre.", |m| &mut m.uv_scale),
            knob("Moss height", (0.0, 20.0), "How far up the tree moss reaches, in metres.", |m| &mut m.moss_height),
            knob("Moss amount", (0.0, 1.0), "How much of the bark moss takes at its thickest.", |m| &mut m.moss_amount),
            knob("Darken low", (0.0, 1.0), "How much darker the bark is at the foot than high in the crown.", |m| &mut m.bark_darken_low),
            knob("Tint red", (0.0, 2.0), "Colour multiplier over the bark texture.", |m| &mut m.bark_tint[0]),
            knob("Tint green", (0.0, 2.0), "Colour multiplier over the bark texture.", |m| &mut m.bark_tint[1]),
            knob("Tint blue", (0.0, 2.0), "Colour multiplier over the bark texture.", |m| &mut m.bark_tint[2]),
        ],
        counts: &[],
    },
    Group {
        title: "Dead wood",
        knobs: &[
            knob("Weathering", (0.0, 1.0), "How far dead wood has bleached toward the dead-wood colour. The grain is kept.", |m| &mut m.dead_wood_weathering),
            knob("Colour red", (0.0, 1.5), "Colour dead wood weathers toward.", |m| &mut m.dead_wood_color[0]),
            knob("Colour green", (0.0, 1.5), "Colour dead wood weathers toward.", |m| &mut m.dead_wood_color[1]),
            knob("Colour blue", (0.0, 1.5), "Colour dead wood weathers toward.", |m| &mut m.dead_wood_color[2]),
            log_knob("Min living bark", (0.0005, 0.1), "Living stems thinner than this get no bark, in metres. Foliage hides them.", |m| &mut m.min_bark_radius),
            knob("Min dead bark", (0.0, 0.05), "Bare dead stems — the dead band, reaching the trunk through dead wood only — thinner than this get no bark, in metres. Zero follows the living figure. Dead twigs inside the crown take the living one.", |m| &mut m.dead_bark_radius),
        ],
        counts: &[Count {
            label: "Cull from level",
            range: (0, 6),
            help: "Shallowest level the bark cutoffs may cull. Stems above it are always swept, so a thin limb is never dropped and its foliage never left floating. Zero lets the cutoff reach any level.",
            get: |m| &mut m.cull_from_level,
        }],
    },
];

pub static IRREGULARITY: &[Group<BarkIrregularity>] = &[
    Group {
        title: "Flutes & swelling",
        knobs: &[
            knob("Flute depth", (0.0, 0.6), "Depth of the flutes up the stem, as a share of its radius.", |b| &mut b.flute_depth),
            knob("Flute waves", (0.0, 16.0), "Roughly how many flutes go round.", |b| &mut b.flute_waves),
            knob("Flute twist", (0.0, 1.0), "How far the flutes wind round, in turns per metre.", |b| &mut b.flute_twist),
            knob("Swell depth", (0.0, 0.5), "Slow swelling and waisting along the stem, as a share of radius.", |b| &mut b.swell_depth),
            knob("Swell period", (0.1, 20.0), "Length of the longest swelling, in metres.", |b| &mut b.swell_period),
        ],
        counts: &[],
    },
    Group {
        title: "Burls, knots & ridges",
        knobs: &[
            knob("Burl density", (0.0, 5.0), "Burls per metre of stem thick enough to carry them.", |b| &mut b.burl_density),
            knob("Burl depth", (0.0, 2.0), "How far a burl stands out, as a share of radius.", |b| &mut b.burl_depth),
            knob("Burl size", (0.02, 3.0), "Width of a burl in metres.", |b| &mut b.burl_size),
            knob("Knot density", (0.0, 5.0), "Healed-over branch scars per metre.", |b| &mut b.knot_density),
            knob("Knot depth", (0.0, 1.0), "How deep a knot's dimple runs, as a share of radius.", |b| &mut b.knot_depth),
            knob("Knot size", (0.02, 3.0), "Width of a knot in metres.", |b| &mut b.knot_size),
            knob("Bark ridge", (0.0, 1.0), "Height of the raised seam running up the parent from each crotch.", |b| &mut b.bark_ridge),
            knob("Collar depth", (0.0, 2.0), "Swelling where a branch leaves, as a share of its radius.", |b| &mut b.collar_depth),
        ],
        counts: &[],
    },
];

// ---------------------------------------------------------------------------------
// Drawing

fn slider<'a>(value: &'a mut f32, knob_range: (f32, f32), log: bool) -> egui::Slider<'a> {
    egui::Slider::new(value, knob_range.0..=knob_range.1)
        .logarithmic(log)
        // Keep a preset's own value even when it lies outside the range: the default
        // clamps it into range as it draws and writes that back over the species.
        .clamping(egui::SliderClamping::Edits)
}

/// Draws one continuous control, reporting whether the user changed it.
pub fn knob_ui<T>(ui: &mut egui::Ui, target: &mut T, knob: &Knob<T>) -> bool {
    let value = (knob.get)(target);
    ui.add(slider(value, knob.range, knob.log).text(knob.label))
        .on_hover_text(knob.help)
        .changed()
}

pub fn count_ui<T>(ui: &mut egui::Ui, target: &mut T, count: &Count<T>) -> bool {
    let value = (count.get)(target);
    ui.add(
        egui::Slider::new(value, count.range.0..=count.range.1)
            .clamping(egui::SliderClamping::Edits)
            .text(count.label),
    )
    .on_hover_text(count.help)
    .changed()
}

pub fn knobs_ui<T>(ui: &mut egui::Ui, target: &mut T, knobs: &[Knob<T>]) -> bool {
    let mut changed = false;
    for knob in knobs {
        changed |= knob_ui(ui, target, knob);
    }
    changed
}

/// Draws a group of controls under its own collapsing header. `salt` keeps the
/// header's open state apart from the same group drawn for another level.
pub fn group_ui<T>(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash,
    target: &mut T,
    group: &Group<T>,
) -> bool {
    let mut changed = false;
    egui::CollapsingHeader::new(group.title)
        .id_salt((salt, group.title))
        .show(ui, |ui| {
            changed |= knobs_ui(ui, target, group.knobs);
            for count in group.counts {
                changed |= count_ui(ui, target, count);
            }
        });
    changed
}

/// How children are laid along their parent: not at all, in whorls, or at a steady
/// rate per metre. Density is the main lever on how bushy a level comes out.
pub fn pattern_ui(ui: &mut egui::Ui, salt: impl std::hash::Hash, pattern: &mut ChildPattern) -> bool {
    let mut changed = false;
    let current = match pattern {
        ChildPattern::None => 0,
        ChildPattern::Whorl { .. } => 1,
        ChildPattern::Continuous { .. } => 2,
    };
    let mut picked = current;
    egui::ComboBox::from_id_salt((salt, "pattern"))
        .selected_text(["None", "Whorls", "Continuous"][current])
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut picked, 0, "None");
            ui.selectable_value(&mut picked, 1, "Whorls");
            ui.selectable_value(&mut picked, 2, "Continuous");
        })
        .response
        .on_hover_text("How children are laid along the parent: in rings at intervals, or at a steady rate per metre.");
    if picked != current {
        *pattern = match picked {
            1 => ChildPattern::Whorl { every: 2, count: 4 },
            2 => ChildPattern::Continuous { density: 2.0 },
            _ => ChildPattern::None,
        };
        changed = true;
    }
    match pattern {
        ChildPattern::None => {}
        ChildPattern::Whorl { every, count } => {
            changed |= ui
                .add(
                    egui::Slider::new(every, WHORL_EVERY.0..=WHORL_EVERY.1)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Whorl every"),
                )
                .on_hover_text("A whorl on every this-many segments of the parent.")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(count, WHORL_COUNT.0..=WHORL_COUNT.1)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Per whorl"),
                )
                .on_hover_text("Branches in each whorl.")
                .changed();
        }
        ChildPattern::Continuous { density } => {
            changed |= ui
                .add(slider(density, DENSITY, true).text("Density"))
                .on_hover_text("Children per metre of the parent.")
                .changed();
        }
    }
    changed
}

/// The growth parameters of stem level `level`, 0 being the trunk.
pub fn stem_mut(params: &mut SpeciesParams, level: usize) -> Option<&mut StemParams> {
    if level == 0 {
        Some(&mut params.trunk)
    } else {
        params.branch_levels.get_mut(level - 1)
    }
}

/// The parameters that spawn stem level `level` off the one above it. The trunk is
/// not spawned by anything.
pub fn spawn_mut(params: &mut SpeciesParams, level: usize) -> Option<&mut ChildParams> {
    if level == 0 {
        None
    } else {
        stem_mut(params, level - 1).map(|s| &mut s.children)
    }
}

/// Stem levels the panel offers: the trunk and every branch level the species declares.
pub fn level_count(params: &SpeciesParams) -> usize {
    params.branch_levels.len() + 1
}

/// Everything about one level: how it is born off its parent, then how it grows.
pub fn level_ui(ui: &mut egui::Ui, params: &mut SpeciesParams, level: usize) -> bool {
    let mut changed = false;
    if let Some(spawn) = spawn_mut(params, level) {
        let parent = if level == 1 { "trunk".to_string() } else { format!("level {}", level - 1) };
        egui::CollapsingHeader::new(format!("Spawning off {parent}"))
            .id_salt((level, "spawning"))
            .default_open(true)
            .show(ui, |ui| {
                changed |= pattern_ui(ui, level, &mut spawn.pattern);
                for group in CHILDREN {
                    changed |= group_ui(ui, (level, "spawn"), spawn, group);
                }
            });
    }
    if let Some(stem) = stem_mut(params, level) {
        for group in STEM {
            changed |= group_ui(ui, (level, "stem"), stem, group);
        }
    }
    changed
}
