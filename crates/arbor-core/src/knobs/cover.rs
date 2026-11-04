//! Controls for a clump of ground cover.

use super::{knob, Group};
use crate::CoverTemplate;

pub static FOOTPRINT: Group<CoverTemplate> = Group {
    title: "Footprint",
    knobs: &[
        knob("Size", (0.5, 4.0), "Metres across the square one clump covers. The engine plants clumps this far apart.", |p| &mut p.footprint.size),
        knob("Tufts", (1.0, 200.0), "Tufts in one clump, spread over its footprint.", |p| &mut p.footprint.tufts),
        knob("Overhang", (0.0, 0.5), "How far tufts run past the edge of the square, thinning out, so clumps planted at a random turn overlap rather than leaving bare corners.", |p| &mut p.footprint.overhang),
        knob("Jitter", (0.0, 1.0), "How far a tuft strays from its own grid square. 0 plants a grid; 1 anywhere in the square.", |p| &mut p.footprint.jitter),
    ],
    counts: &[],
};

pub static TUFT: Group<CoverTemplate> = Group {
    title: "Tuft",
    knobs: &[
        knob("Cards", (1.0, 16.0), "Cards fanned round each tuft.", |p| &mut p.tuft.cards),
        knob("Radius", (0.0, 0.3), "Metres the cards' roots spread over.", |p| &mut p.tuft.radius),
        knob("Height", (0.05, 1.5), "Height of a card, in metres.", |p| &mut p.tuft.height),
        knob("Height variance", (0.0, 0.8), "Spread of height from tuft to tuft, as a share.", |p| &mut p.tuft.height_variance),
        knob("Card height variance", (0.0, 0.8), "Spread of height from card to card within a tuft.", |p| &mut p.tuft.card_height_variance),
        knob("Lean", (0.0, 60.0), "Degrees a card leans out from its tuft at the root.", |p| &mut p.tuft.lean_deg),
        knob("Lean variance", (0.0, 40.0), "Spread of that lean, in degrees.", |p| &mut p.tuft.lean_variance_deg),
        knob("Curl", (-30.0, 90.0), "Degrees a card bends further by its tip: the arch of long grass.", |p| &mut p.tuft.curl_deg),
        knob("Twist", (0.0, 90.0), "Degrees a card turns about its length from root to tip.", |p| &mut p.tuft.twist_deg),
        knob("Understory", (0.0, 1.0), "Share of cards drawn short: the low layer of leaves under the tall ones, which hides the ground.", |p| &mut p.tuft.understory),
        knob("Understory height", (0.05, 1.0), "Height of those short cards against the rest.", |p| &mut p.tuft.understory_height),
        knob("Rosette", (0.0, 1.0), "0 fans a tuft's cards as grass; 1 spreads them evenly round the middle, each facing along its own arch, as a fern's fronds are.", |p| &mut p.tuft.rosette),
    ],
    counts: &[],
};

pub static CARD: Group<CoverTemplate> = Group {
    title: "Card & colour",
    knobs: &[
        knob("Width", (0.3, 2.0), "Card width against the width that keeps atlas pixels square.", |p| &mut p.card.width),
        knob("Width at top", (0.5, 2.0), "Width at the tip against width at the root.", |p| &mut p.card.width_top),
        knob("Fold", (0.0, 1.0), "How far a card's edges stand up from its middle, as a share of its half width: the shallow V along a frond's rachis. 0 is flat.", |p| &mut p.card.fold),
        knob("Dry tufts", (0.0, 1.0), "Share of tufts that draw from the dry cells.", |p| &mut p.colour.dry_fraction),
        knob("Mix", (0.0, 1.0), "Share of a tuft's cards that stray to the other kind of cell.", |p| &mut p.colour.mix),
        knob("Flowering tufts", (0.0, 1.0), "Share of tufts drawing from the cells with wildflowers in them.", |p| &mut p.colour.flower_fraction),
        knob("Flower drift", (0.0, 3.0), "Metres across the drifts flowering tufts gather in. 0 scatters them tuft by tuft.", |p| &mut p.colour.flower_patch),
        knob("Ground normal blend", (0.0, 1.0), "How far shading normals lean from the card to straight up. High reads as one lit surface; low as speckle.", |p| &mut p.normals.ground_normal_blend),
        knob("Root blend", (0.0, 0.5), "Metres above the root the engine blends into the terrain colour. Exported only; the viewer does not draw it.", |p| &mut p.normals.root_blend),
    ],
    counts: &[],
};

pub static WIND: Group<CoverTemplate> = Group {
    title: "Wind response",
    knobs: &[
        knob("Bend", (0.0, 2.0), "How far a card bends along its length in a full gale.", |p| &mut p.wind.flexibility[1]),
        knob("Flutter", (0.0, 1.0), "How far a card flutters about its root, in radians.", |p| &mut p.wind.flutter),
        knob("Frequency", (0.2, 4.0), "How fast the cards sway, in hertz.", |p| &mut p.wind.frequency),
    ],
    counts: &[],
};

macro_rules! lod_group {
    ($name:ident, $title:literal, $i:literal) => {
        pub static $name: Group<CoverTemplate> = Group {
            title: $title,
            knobs: &[
                knob("Cards kept", (0.0, 1.0), "Share of the clump's cards this LOD keeps. Always a subset of the finer LOD's.", |p| &mut p.lod[$i].cards),
                knob("Segments", (1.0, 8.0), "Segments up each card.", |p| &mut p.lod[$i].segments),
                knob("Width", (0.5, 3.0), "Card width against LOD0's, to make up the coverage the dropped cards took.", |p| &mut p.lod[$i].width),
                knob("Screen size", (0.0, 0.3), "Share of the screen's height below which this LOD takes over.", |p| &mut p.lod[$i].screen_size),
            ],
            counts: &[],
        };
    };
}
lod_group!(LOD0, "LOD0", 0);
lod_group!(LOD1, "LOD1", 1);
lod_group!(LOD2, "LOD2", 2);
lod_group!(LOD3, "LOD3", 3);
pub static LODS: [&Group<CoverTemplate>; 4] = [&LOD0, &LOD1, &LOD2, &LOD3];

pub static GROUPS: [&Group<CoverTemplate>; 4] = [&FOOTPRINT, &TUFT, &CARD, &WIND];
