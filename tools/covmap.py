"""Where an arbor crown's mass sits against its references': arbor in red, the mean of
the references in green, both on the tree-height square refmatch scores in.
    python tools/covmap.py <species> <capture.png> <out.png>"""
import sys, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm
species, cap, out = sys.argv[1:4]
img, m = rm.load_capture(cap)
a = cv2.GaussianBlur(rm.features(img, m)["cov"], (0, 0), 2)
refs = [cv2.GaussianBlur(rm.features(i, mm)["cov"], (0, 0), 2) for _, i, mm in rm.load_refs(species)]
r = np.mean(refs, 0)
k = 255 / max(a.max(), r.max())
viz = np.dstack([np.zeros_like(a), r * k, a * k]).astype(np.uint8)
viz = cv2.resize(viz, (480, 480), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(out, viz)
rows = lambda c: (c > 0.05).sum(1)
print("width by height band (fraction of tree height), top to bottom:")
for i in range(0, rm.GRID, 12):
    print(f"  {i / rm.GRID:.2f}  arbor {rows(a)[i:i+12].mean() / rm.GRID:.2f}  ref {rows(r)[i:i+12].mean() / rm.GRID:.2f}"
          f"   mass arbor {a[i:i+12].sum() / a.sum():.2f} ref {r[i:i+12].sum() / r.sum():.2f}")
