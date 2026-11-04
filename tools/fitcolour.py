"""Fits a species' leaf colour to its references under the comparison light, a few
captures at a time, and writes it back into the preset as it goes. Three numbers, one
for each axis of the references' mean Lab: an overall gain on `tint` for lightness,
`saturation` for greenness, and blue against red and green in `tint` for yellowness.
The tint starts from grey, so it shifts the art's hue without adding chroma of its own.
    python tools/fitcolour.py <preset.ron> <species-name> <refs-key> [rounds]"""
import os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm

path, name, key = sys.argv[1:4]
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
TINT = re.compile(r"^(        tint: \()([0-9.]+), ([0-9.]+), ([0-9.]+)(\),)", re.M)
SAT = re.compile(r"^(        saturation: )([0-9.]+)(,)", re.M)


def write(gain, blue, sat):
    s = open(path, encoding="utf-8", newline="").read()
    s = TINT.sub(lambda m: f"{m.group(1)}{gain:.3f}, {gain:.3f}, {gain * blue:.3f}{m.group(5)}", s, count=1)
    s = SAT.sub(lambda m: f"{m.group(1)}{sat:.3f}{m.group(3)}", s, count=1)
    open(path, "w", encoding="utf-8", newline="").write(s)


def measure():
    subprocess.run(["sh", "tools/cap.sh", name, key, "fit"], env={**os.environ, "SEEDS": "1"},
                   stdout=subprocess.DEVNULL, check=True)
    img, m = rm.load_capture("out/iter/fit_s1.png")
    return rm.features(img, m)["lab"].mean(0)


target = np.mean([rm.features(i, m)["lab"].mean(0) for _, i, m in rm.load_refs(key)], 0)
s = open(path, encoding="utf-8", newline="").read()
t = [float(TINT.search(s).group(i)) for i in (2, 3, 4)]
gain, blue = (t[0] + t[1]) / 2, t[2] / ((t[0] + t[1]) / 2)
sat = float(SAT.search(s).group(2)) if SAT.search(s) else 1.0
if not SAT.search(s):
    raise SystemExit("add a `saturation:` line under the tint first")
write(gain, blue, sat)
for r in range(rounds):
    lab = measure()
    dl, da, db = target - lab
    print(f"round {r}: gain {gain:.3f} blue {blue:.3f} saturation {sat:.3f} Lab {np.round(lab, 1)} target {np.round(target, 1)}")
    if max(abs(dl), abs(da), abs(db)) < 0.8:
        break
    gain *= ((lab[0] + dl) / max(lab[0], 1.0)) ** 2.4
    sat = float(np.clip(sat * np.exp((lab[1] - target[1]) * 0.05), 0.1, 1.5))
    # Damped: blue and saturation both move yellowness, and undamped they chase.
    blue *= np.exp(-db / 45.0)
    write(gain, blue, sat)
print(f"gain {gain:.3f} blue {blue:.3f} saturation {sat:.3f}")
