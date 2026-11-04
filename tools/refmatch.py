"""Scores an arbor capture against studio photographs of the same species.

    python tools/refmatch.py birch out/iter/birch_s1.png [more captures] [--debug out/iter/dbg.png]

The capture is taken with `--backdrop 1,0,1 --ground none` so the tree can be cut out
against the key colour; the references are cut out against their white studio sweep.
Both are then set on the same footing, the trunk foot at the bottom middle and the
tree's height as the unit, and compared on what makes a tree read as its species from
across a field: the crown's shape and how its mass is spread, how much sky shows
through it, how fine its clumps are, its colour, and its contrast. Each comes out
0..1, and the score is their weighted mean. It is a yardstick for iterating, not a
substitute for looking at the two side by side.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "out", "refs", "megaplants")
GRID = 96
WEIGHTS = {"shape": 0.30, "porosity": 0.12, "grain": 0.13, "colour": 0.30, "contrast": 0.15}


def ref_mask(img, shadow_row=None):
    """Tree pixels in a studio shot: anything off the smooth sweep behind it."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    u, v = xs.ravel() / w, ys.ravel() / h
    basis = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    bright = f.min(2).ravel()
    sat = (f.max(2) - f.min(2)).ravel()
    sel = (bright > np.percentile(bright, 40)) & (sat < 12)
    bg = np.empty_like(f)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(basis[sel], f[..., c].ravel()[sel], rcond=None)
        bg[..., c] = (basis @ coef).reshape(h, w)
    dev = np.abs(f - bg).max(2)
    m = dev > 22
    if shadow_row is not None:
        m[shadow_row:] &= dev[shadow_row:] > 70
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m.astype(bool)


def key_mask(img, key=(231, 0, 231)):
    d = np.abs(img.astype(np.int32) - np.array(key[::-1])).max(2)
    return d > 40


def normalise(img, mask):
    """Crops to the tree with the foot at the bottom middle, one tree-height square."""
    rows = np.where(mask.sum(1) > 0)[0]
    top, bot = rows[0], rows[-1]
    hgt = bot - top + 1
    base_rows = mask[bot - max(2, hgt // 30): bot + 1]
    bx = int(np.median(np.where(base_rows)[1]))
    half = hgt // 2 + 1
    pad = ((half, half), (half, half))
    m = np.pad(mask, pad)
    im = np.pad(img, pad + ((0, 0),), constant_values=0)
    x0, y0 = bx - half + half, top + half
    sl = (slice(y0, y0 + hgt), slice(x0, x0 + 2 * half))
    return im[sl], m[sl]


def lab_pixels(img, mask):
    """Colours inside the tree, not at its edges: a pixel half leaf and half sweep says
    more about the sweep than the leaf, and the two sweeps are not the same grey."""
    solid = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if solid.sum() > 200:
        mask = solid
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[mask]
    px[:, 0] *= 100 / 255
    px[:, 1:] -= 128
    return px


def features(img, mask):
    img, mask = normalise(img, mask)
    cov = cv2.resize(mask.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA)
    # Porosity: of the crown's row-by-row span, how much is filled.
    spans, filled = 0, 0
    for r in range(GRID):
        c = np.where(cov[r] > 0.02)[0]
        if len(c):
            spans += c[-1] - c[0] + 1
            filled += cov[r, c[0]: c[-1] + 1].sum()
    porosity = 1 - filled / max(spans, 1)
    # Grain: edge length per covered area at a fixed size, how broken up the mass is.
    m512 = cv2.resize(mask.astype(np.uint8) * 255, (512, 512), interpolation=cv2.INTER_AREA) > 127
    edges = np.logical_xor(m512, cv2.erode(m512.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    grain = edges.sum() / max(m512.sum(), 1)
    px = lab_pixels(img, mask)
    return {"cov": cov, "porosity": porosity, "grain": grain, "lab": px,
            "aspect": float((cov.sum(0) > 0.5).sum()) / GRID}


def hist(px):
    h, _ = np.histogramdd(px, bins=(10, 12, 12), range=((0, 100), (-40, 20), (-10, 50)))
    return h / max(h.sum(), 1)


def compare(a, b):
    ca = cv2.GaussianBlur(a["cov"], (0, 0), 2)
    cb = cv2.GaussianBlur(b["cov"], (0, 0), 2)
    shape = np.minimum(ca, cb).sum() / np.maximum(ca, cb).sum()
    porosity = 1 - abs(a["porosity"] - b["porosity"]) / max(a["porosity"], b["porosity"], 1e-3)
    grain = min(a["grain"], b["grain"]) / max(a["grain"], b["grain"])
    ha, hb = hist(a["lab"]), hist(b["lab"])
    inter = np.minimum(ha, hb).sum()
    de = np.linalg.norm(a["lab"].mean(0) - b["lab"].mean(0))
    colour = 0.5 * inter + 0.5 * np.exp(-de / 15)
    sa, sb = a["lab"][:, 0].std(), b["lab"][:, 0].std()
    contrast = min(sa, sb) / max(sa, sb)
    parts = {"shape": shape, "porosity": porosity, "grain": grain, "colour": colour, "contrast": contrast}
    parts["score"] = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    return parts


def load_refs(species):
    with open(os.path.join(ROOT, "tools", "refs.json")) as f:
        entries = json.load(f)[species]
    out = []
    for e in entries:
        img = cv2.imread(os.path.join(REF_DIR, e["file"]))
        x0, y0, x1, y1 = e["box"]
        crop = img[y0:y1, x0:x1]
        shadow = e.get("shadow_y")
        mask = ref_mask(crop, None if shadow is None else shadow - y0)
        for ex0, ey0, ex1, ey1 in e.get("exclude", []):
            mask[max(ey0 - y0, 0): ey1 - y0, max(ex0 - x0, 0): ex1 - x0] = False
        out.append((e, crop, mask))
    return out


def load_capture(path, panel=400):
    """The keyed capture gives the mask. Its colours are the look capture's beside it,
    when there is one: against the key, every leaf edge is part magenta, where against
    the grey sweep it is part grey, as the photographs' edges are part white."""
    img = cv2.imread(path)[:, panel:]
    mask = key_mask(img)
    look = path[:-4] + "_look.png"
    if os.path.exists(look):
        img = cv2.imread(look)[:, panel:]
    return img, mask


def debug_sheet(path, pairs):
    tiles = []
    for img, mask in pairs:
        img, mask = normalise(img, mask)
        im = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mask.astype(np.uint8) * 255, (360, 360), interpolation=cv2.INTER_AREA)
        tiles.append(np.vstack([im, cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)]))
    cv2.imwrite(path, np.hstack(tiles))


def main():
    args = sys.argv[1:]
    dbg = None
    if "--debug" in args:
        i = args.index("--debug")
        dbg = args[i + 1]
        del args[i:i + 2]
    species, captures = args[0], args[1:]
    caps = [load_capture(c) for c in captures]
    fcs = [features(img, mask) for img, mask in caps]
    refs = load_refs(species)
    fr = [features(img, mask) for _, img, mask in refs]
    keys = list(WEIGHTS) + ["score"]
    # Every capture against every reference: seeds are individuals as much as the
    # photographs are, and one lucky or unlucky tree says little about the species.
    rows = [compare(fc, f) for fc in fcs for f in fr]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print(f"{os.path.basename(captures[0])} +{len(captures) - 1} vs {species} ({len(fr)} refs)")
    print("  arbor      " + "  ".join(f"{k} {mean[k]:.3f}" for k in keys))
    # The ceiling: how well the references match one another. Two birches are not one
    # birch, so a capture is judged against what a second tree of the kind would get.
    if len(fr) > 1:
        pairs = [compare(fr[i], fr[j]) for i in range(len(fr)) for j in range(len(fr)) if i != j]
        cm = {k: float(np.mean([r[k] for r in pairs])) for k in keys}
        print("  ref-vs-ref " + "  ".join(f"{k} {cm[k]:.3f}" for k in keys))
        print(f"  relative {mean['score'] / cm['score']:.3f}")
    avg = lambda fs, k: np.mean([f[k] for f in fs])
    print(f"  porosity arbor {avg(fcs, 'porosity'):.3f} refs {avg(fr, 'porosity'):.3f}"
          f"  grain arbor {avg(fcs, 'grain'):.3f} refs {avg(fr, 'grain'):.3f}"
          f"  aspect arbor {avg(fcs, 'aspect'):.2f} refs {avg(fr, 'aspect'):.2f}")
    la = np.mean([f["lab"].mean(0) for f in fcs], 0)
    lr = np.mean([f["lab"].mean(0) for f in fr], 0)
    print(f"  Lab mean arbor {np.round(la, 1)} refs {np.round(lr, 1)}"
          f"  L std arbor {np.mean([f['lab'][:, 0].std() for f in fcs]):.1f}"
          f" refs {np.mean([f['lab'][:, 0].std() for f in fr]):.1f}")
    if dbg:
        debug_sheet(dbg, caps + [(img, mask) for _, img, mask in refs])


if __name__ == "__main__":
    main()
