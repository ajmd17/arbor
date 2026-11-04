"""Tunes a few of a preset's leaf numbers against the whole reference score rather than
the mean colour alone. `fitcolour.py` lands the crown's average Lab on the photographs',
but an average can be met by the wrong spread: a crown bluer than the reference in its
highlights and yellower in its shade has the right mean and the wrong histogram. This
captures, scores, and walks each number up and down in turn, keeping what helps and
halving the step when nothing does. It writes the best values back as it goes.
Give it four seeds or more: one seed to the next moves the score by about 0.02, more
than most single steps here, and on two seeds the walk mostly fits the noise.
    SEEDS="1 2 3 4" python tools/fitscore.py <preset.ron> <species-name> <refs-key> \
        field[=step] ... [--rounds N]
A field is a number in the preset's `leaves` block (`saturation`, `canopy_scatter`,
`hue_variance`, ...), or `gain` and `blue`, which move `tint` as fitcolour does."""
import os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm

args = sys.argv[1:]
rounds = 3
if "--rounds" in args:
    i = args.index("--rounds")
    rounds = int(args[i + 1])
    del args[i:i + 2]
path, name, key = args[:3]
fields = []
for f in args[3:]:
    n, _, st = f.partition("=")
    fields.append([n, float(st) if st else None])
seeds = os.environ.get("SEEDS", "1 2 3 4").split()
tag = f"fs_{name}"
TINT = re.compile(r"^(        tint: \()([0-9.]+), ([0-9.]+), ([0-9.]+)(\),)", re.M)


def leaves_span(s):
    a = s.index("    leaves: (")
    return a, s.index("\n    wind: (", a)


def get(s, n):
    if n in ("gain", "blue"):
        t = [float(TINT.search(s).group(i)) for i in (2, 3, 4)]
        g = (t[0] + t[1]) / 2
        return g if n == "gain" else t[2] / g
    a, b = leaves_span(s)
    m = re.search(rf"^        {n}: (-?[0-9.]+),", s[a:b], re.M)
    if m is None:
        raise SystemExit(f"no `{n}:` in the leaves block of {path}")
    return float(m.group(1))


def put(s, n, v):
    if n in ("gain", "blue"):
        g, bl = get(s, "gain"), get(s, "blue")
        g, bl = (v, bl) if n == "gain" else (g, v)
        return TINT.sub(lambda m: f"{m.group(1)}{g:.3f}, {g:.3f}, {g * bl:.3f}{m.group(5)}", s, count=1)
    a, b = leaves_span(s)
    body = re.sub(rf"^(        {n}: )-?[0-9.]+(,)", lambda m: f"{m.group(1)}{v:.3f}{m.group(2)}",
                  s[a:b], count=1, flags=re.M)
    return s[:a] + body + s[b:]


refs = [rm.features(i, m) for _, i, m in rm.load_refs(key)]
pairs = [rm.compare(refs[i], refs[j]) for i in range(len(refs)) for j in range(len(refs)) if i != j]
ceiling = np.mean([p["score"] for p in pairs])


def score(s):
    open(path, "w", encoding="utf-8", newline="").write(s)
    subprocess.run(["sh", "tools/cap.sh", name, key, tag], env={**os.environ, "SEEDS": " ".join(seeds)},
                   stdout=subprocess.DEVNULL, check=True)
    fcs = [rm.features(*rm.load_capture(f"out/iter/{tag}_s{sd}.png")) for sd in seeds]
    rows = [rm.compare(fc, f) for fc in fcs for f in refs]
    return np.mean([r["score"] for r in rows]) / ceiling, np.mean([r["colour"] for r in rows])


best = open(path, encoding="utf-8", newline="").read()
for f in fields:
    if f[1] is None:
        v = get(best, f[0])
        f[1] = max(abs(v) * 0.15, 0.05)
cur, col = score(best)
print(f"start {cur:.4f} (colour {col:.3f})", flush=True)
for r in range(rounds):
    moved = False
    for f in fields:
        n, step = f
        v = get(best, n)
        for d in (step, -step):
            nv = max(v + d, 0.0)
            if nv == v:
                continue
            cand = put(best, n, nv)
            sc, col = score(cand)
            print(f"  round {r} {n} {v:.3f} -> {nv:.3f}: {sc:.4f} (colour {col:.3f})", flush=True)
            if sc > cur + 1e-4:
                best, cur, moved = cand, sc, True
                break
        else:
            f[1] = step / 2
    open(path, "w", encoding="utf-8", newline="").write(best)
    print(f"round {r}: {cur:.4f} " + " ".join(f"{n} {get(best, n):.3f}" for n, _ in fields), flush=True)
    if not moved and all(s < 0.01 for _, s in fields):
        break
open(path, "w", encoding="utf-8", newline="").write(best)
print(f"best {cur:.4f}")
