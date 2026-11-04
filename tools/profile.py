"""Width, mass and fill of a crown band by band, captures averaged against references
averaged.   python tools/profile.py <refs-key> <capture.png>..."""
import sys, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm
key, caps = sys.argv[1], sys.argv[2:]
blur = lambda c: cv2.GaussianBlur(c["cov"], (0, 0), 2)
a = np.mean([blur(rm.features(*rm.load_capture(c))) for c in caps], 0)
r = np.mean([blur(rm.features(i, m)) for _, i, m in rm.load_refs(key)], 0)
for i in range(0, rm.GRID, 12):
    sa, sr = a[i:i + 12], r[i:i + 12]
    print(f"{i / rm.GRID:.2f}  width {(sa > 0.05).sum(1).mean() / rm.GRID:.2f} / {(sr > 0.05).sum(1).mean() / rm.GRID:.2f}"
          f"  mass {sa.sum() / a.sum():.2f} / {sr.sum() / r.sum():.2f}"
          f"  fill {sa[sa > 0.05].mean():.2f} / {sr[sr > 0.05].mean():.2f}")
