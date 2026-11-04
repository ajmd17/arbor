//! Controls for a rock.

use super::{knob, Group};
use crate::rocks::RockTemplate;

pub static SHAPE: Group<RockTemplate> = Group {
    title: "Shape",
    knobs: &[
        knob("Width", (0.1, 6.0), "Metres across, before noise.", |p| &mut p.shape.width),
        knob("Height", (0.1, 4.0), "Metres high, before noise and the flat bottom.", |p| &mut p.shape.height),
        knob("Depth", (0.1, 6.0), "Metres deep, before noise.", |p| &mut p.shape.depth),
        knob("Facets", (0.0, 32.0), "Planes the rock is cut by: its broad faces.", |p| &mut p.shape.facets),
        knob("Facet depth", (0.0, 0.8), "How deep those planes cut, as a share of how far the rock reaches that way.", |p| &mut p.shape.facet_depth),
        knob("Sharpness", (2.0, 80.0), "How crisply the faces meet: low rounds the edges like a river stone, high leaves them like fresh rubble.", |p| &mut p.shape.sharpness),
        knob("Wear", (0.0, 1.0), "How unevenly the edges are worn: at 1 some stay four times as crisp as others.", |p| &mut p.shape.wear),
        knob("Bulge", (0.0, 0.3), "Swelling and denting of the whole, as a share of the radius.", |p| &mut p.shape.bulge),
        knob("Bulge scale", (0.2, 4.0), "How many swells fit round the rock.", |p| &mut p.shape.bulge_scale),
        knob("Undulation", (0.0, 0.1), "Broad rolls and hollows over the faces, as a share of the radius.", |p| &mut p.shape.undulation),
        knob("Chips", (0.0, 0.12), "How deep chips are struck off the crisper edges, as a share of the radius.", |p| &mut p.shape.chips),
        knob("Chip size", (0.05, 1.0), "How far each chip runs along its edge, as a share of the radius.", |p| &mut p.shape.chip_size),
        knob("Flaking", (0.0, 1.0), "Share of the rock broken into plates flaking off it, stacked a step above one another.", |p| &mut p.shape.flaking),
        knob("Flake step", (0.0, 0.05), "Height of each plate's step, as a share of the radius.", |p| &mut p.shape.flake_step),
        knob("Flake size", (0.05, 2.0), "Size of a plate, as a share of the radius.", |p| &mut p.shape.flake_size),
        knob("Bedded", (0.0, 1.0), "How far the plates lie along the bedding: shells following the surface at 0, layers of slate at 1.", |p| &mut p.shape.bedded),
        knob("Fracture", (0.0, 0.1), "Stepped ledges where the stone has split along its bedding.", |p| &mut p.shape.fracture),
        knob("Flat top", (0.0, 1.0), "Where the top is cut flat along the bedding, as a share of the half height: 0 for none.", |p| &mut p.shape.flat_top),
        knob("Sides", (0.0, 1.0), "Four upright faces square to the width and depth, as a share of how far the rock reaches that way: 0 for none.", |p| &mut p.shape.sides),
        knob("Flat bottom", (0.0, 0.9), "Share of the height cut flat where it sits on the ground.", |p| &mut p.shape.flat_bottom),
        knob("Bury", (0.0, 0.9), "Share of the height below the origin, so a rock set on the ground is sunk into it.", |p| &mut p.shape.bury),
    ],
    counts: &[],
};

pub static SURFACE: Group<RockTemplate> = Group {
    title: "Surface",
    knobs: &[
        knob("Grain", (0.0, 0.006), "Height of the fine unevenness of the stone, metres. Maps only.", |p| &mut p.surface.grain),
        knob("Pitting", (0.0, 1.0), "Share of the stone pitted with small holes. Maps only.", |p| &mut p.surface.pitting),
        knob("Cracks", (0.0, 8.0), "Joint cracks through the stone, showing as near straight lines. Maps only.", |p| &mut p.surface.cracks),
        knob("Crack width", (0.0, 0.02), "How wide the cracks open, metres.", |p| &mut p.surface.crack_width),
        knob("Strata", (0.0, 1.5), "Bands of bedding across the stone, in colour and in relief.", |p| &mut p.surface.strata),
        knob("Cavity", (0.0, 2.0), "How strongly hollows and cracks are darkened and occluded.", |p| &mut p.surface.cavity),
        knob("Crunch", (0.0, 0.012), "Height of the knobbly, broken relief a few centimetres across, metres. Maps only.", |p| &mut p.surface.crunch),
        knob("Foliation", (0.0, 1.0), "How far the grain is drawn out into parallel lines, as in gneiss. Maps only.", |p| &mut p.surface.foliation),
    ],
    counts: &[],
};

pub static MOSS: Group<RockTemplate> = Group {
    title: "Moss",
    knobs: &[
        knob("Amount", (-1.0, 1.0), "From none at -1, through tops only at 0, to nearly all of it at 1.", |p| &mut p.moss.amount),
        knob("Upward", (0.0, 2.0), "How far moss keeps to what faces up, against creeping up from the ground.", |p| &mut p.moss.upward),
        knob("Patchiness", (0.05, 3.0), "How broken up the moss is.", |p| &mut p.moss.patchiness),
        knob("Softness", (0.01, 1.0), "How soft the edge of a patch is.", |p| &mut p.moss.softness),
        knob("Crevices", (0.0, 1.0), "How far moss gathers in hollows, cracks and along ledges.", |p| &mut p.moss.crevices),
        knob("Dry", (0.0, 1.0), "Share of the moss dried to a paler olive.", |p| &mut p.moss.dry),
    ],
    counts: &[],
};

pub static GROUPS: [&Group<RockTemplate>; 3] = [&SHAPE, &SURFACE, &MOSS];
