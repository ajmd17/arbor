"""Fits a species' leaf colour to its references the way fitcolour.py does, but holds the
leaf's albedo inside the range real foliage has. Studio photographs cannot say how bright
a leaf really is: how bright it comes out depends on the studio's lights and exposure as
much as on the leaf, and fitting lightness alone drove some presets to leaves twice as
reflective as any real one, and pale and grey with it. So lightness is fitted only as
far as the range allows, and hue (the references' a* and b*) is fitted inside it.

    python tools/fitalbedo.py <preset.ron> <species-name> <refs-key> <kind> [rounds]
    python tools/fitalbedo.py <preset.ron> - - <kind>      (only report the albedo)

`kind` is broadleaf, conifer or glaucous (a conifer with a waxy bloom, Scots pine).

Three numbers: an overall gain, and red and blue against green, written as the tint
(gain * red, gain, gain * blue). `saturation` is held, at no less than SATURATION_FLOOR:
it greys every texel alike, so pulling it down and the tint's chroma back up flattens
the art's own variation (brown twig, green needle) into one colour while landing on the
same mean. Red fits a*, blue fits b*. The albedo checked against the range is
the mean, in linear light, of the art's opaque texels after the loader's saturation and
times the tint, which is what the shader multiplies the light by."""
import os, re, subprocess, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm

# Mean linear albedo of the drawn leaf. A green leaf reflects about 0.10-0.18 of the
# light around 550 nm; a conifer shoot, needles shading needles in the scan, sits lower.
# Chlorophyll takes red to about half of green and blue further still, so a leaf whose
# red or blue reaches its green reads olive, grey or mint rather than green.
# The wax on a glaucous needle scatters blue back before the chlorophyll gets it, which
# is what makes a Scots pine blue-green, so blue may run much closer to green there.
GREEN = {"broadleaf": (0.08, 0.16), "conifer": (0.05, 0.11), "glaucous": (0.05, 0.11)}
RED_OVER_GREEN = (0.40, 0.75)
BLUE_OVER_GREEN = {"broadleaf": (0.15, 0.45), "conifer": (0.15, 0.45), "glaucous": (0.15, 0.85)}
SATURATION_FLOOR = 0.7

path, name, key, kind = sys.argv[1:5]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 6
TINT = re.compile(r"^(        tint: \()([0-9.]+), ([0-9.]+), ([0-9.]+)(\),)", re.M)
SAT = re.compile(r"^(        saturation: )([0-9.]+)(,)", re.M)
TEX = re.compile(r'^        texture: "([^"]+)"', re.M)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


src = open(path, encoding="utf-8", newline="").read()
art = np.asarray(Image.open(os.path.join(ROOT, "assets", "textures", TEX.search(src).group(1) + "_albedo.png"))
                 .convert("RGBA")).astype(np.float32)
art = art[..., :3][art[..., 3] > 128]


def art_mean(sat):
    """The art's mean linear colour after `textures.rs::saturate`, which works in sRGB."""
    grey = art @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    s = np.clip(np.round(grey[:, None] + (art - grey[:, None]) * sat), 0, 255)
    return linear(s).mean(0)


def project(gain, red, blue, sat):
    """Brings the drawn albedo into range, moving each number only as far as it must."""
    m = art_mean(sat)
    lo, hi = GREEN[kind]
    gain = float(np.clip(gain, lo / m[1], hi / m[1]))
    red = float(np.clip(red, RED_OVER_GREEN[0] * m[1] / m[0], RED_OVER_GREEN[1] * m[1] / m[0]))
    b_lo, b_hi = BLUE_OVER_GREEN[kind]
    blue = float(np.clip(blue, b_lo * m[1] / m[2], b_hi * m[1] / m[2]))
    return gain, red, blue


def albedo(gain, red, blue, sat):
    return art_mean(sat) * np.array([gain * red, gain, gain * blue])


def write(gain, red, blue, sat):
    s = open(path, encoding="utf-8", newline="").read()
    s = TINT.sub(lambda m: f"{m.group(1)}{gain * red:.3f}, {gain:.3f}, {gain * blue:.3f}{m.group(5)}", s, count=1)
    s = SAT.sub(lambda m: f"{m.group(1)}{sat:.3f}{m.group(3)}", s, count=1)
    open(path, "w", encoding="utf-8", newline="").write(s)


def measure():
    subprocess.run(["sh", "tools/cap.sh", name, key, "fit"], env={**os.environ, "SEEDS": "1"},
                   stdout=subprocess.DEVNULL, check=True)
    img, m = rm.load_capture("out/iter/fit_s1.png")
    return rm.features(img, m)["lab"].mean(0)


if not SAT.search(src):
    raise SystemExit("add a `saturation:` line under the tint first")
t = [float(TINT.search(src).group(i)) for i in (2, 3, 4)]
sat = float(SAT.search(src).group(2))
gain, red, blue = t[1], t[0] / t[1], t[2] / t[1]
print(f"start: albedo {np.round(albedo(gain, red, blue, sat), 3)}")
if name == "-":
    a = albedo(gain, red, blue, sat)
    print(f"green {a[1]:.3f} in {GREEN[kind]}, red/green {a[0] / a[1]:.2f} in {RED_OVER_GREEN}, "
          f"blue/green {a[2] / a[1]:.2f} in {BLUE_OVER_GREEN[kind]}")
    raise SystemExit
target = np.mean([rm.features(i, m)["lab"].mean(0) for _, i, m in rm.load_refs(key)], 0)
sat = max(sat, SATURATION_FLOOR)
gain, red, blue = project(gain, red, blue, sat)
write(gain, red, blue, sat)
for r in range(rounds):
    lab = measure()
    dl, da, db = target - lab
    print(f"round {r}: tint {np.round([gain * red, gain, gain * blue], 3)} saturation {sat:.3f} "
          f"albedo {np.round(albedo(gain, red, blue, sat), 3)} Lab {np.round(lab, 1)} target {np.round(target, 1)}")
    if max(abs(da), abs(db)) < 0.8 and abs(dl) < 0.8:
        break
    gain *= ((lab[0] + dl) / max(lab[0], 1.0)) ** 2.4
    # Damped, as in fitcolour.py: red and blue each move both axes a little.
    red *= np.exp(da / 45.0)
    blue *= np.exp(-db / 45.0)
    gain, red, blue = project(gain, red, blue, sat)
    write(gain, red, blue, sat)
print(f"tint ({gain * red:.3f}, {gain:.3f}, {gain * blue:.3f}) saturation {sat:.3f} "
      f"albedo {np.round(albedo(gain, red, blue, sat), 3)}")
