"""Side by side: an arbor look capture next to a reference, both cut to the tree and set
to the same height.   python tools/side.py <species> <look.png> <out.jpg> [ref index]"""
import sys, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm
species, look, out = sys.argv[1:4]
idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
H = 1000
key = cv2.imread(look.replace("_look", ""))[:, 400:]
img = cv2.imread(look)[:, 400:]
m = rm.key_mask(key)
ys, xs = np.where(m)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
a = img[y0:y1 + 1, x0:x1 + 1]
e, rimg, rmask = rm.load_refs(species)[idx]
ys, xs = np.where(rmask)
b = rimg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
def fit(im): return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA)
a, b = fit(a), fit(b)
gap = np.full((H, 20, 3), 255, np.uint8)
cv2.imwrite(out, np.hstack([a, gap, b]))
