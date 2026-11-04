"""Percentiles of L, a and b inside the tree, a capture against its references.
    python tools/colourdist.py <refs-key> <capture.png>..."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm
key, caps = sys.argv[1], sys.argv[2:]
P = [5, 25, 50, 75, 95]
def show(name, pxs):
    px = np.concatenate(pxs)
    q = np.percentile(px, P, axis=0)
    print(f"{name:6s} " + "  ".join(f"{c} {' '.join(f'{v:6.1f}' for v in q[:, i])}" for i, c in enumerate("Lab")))
show("arbor", [rm.features(*rm.load_capture(c))["lab"] for c in caps])
show("refs", [rm.features(i, m)["lab"] for _, i, m in rm.load_refs(key)])
