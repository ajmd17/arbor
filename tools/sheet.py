"""A labelled sheet of arbor captures beside their references, one row per species:
before, after and the photograph, each cut to the tree and set to the same height.
    python tools/sheet.py out.jpg <label> <refs-key> <ref-index> <before_look.png|-> <after_look.png> ..."""
import sys, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refmatch as rm
H = 520
def cut_capture(look):
    key = cv2.imread(look.replace("_look", ""))[:, 400:]
    img = cv2.imread(look)[:, 400:]
    ys, xs = np.where(rm.key_mask(key))
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
def fit(im, label):
    im = cv2.resize(im, (max(1, int(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
    pad = max(0, 300 - im.shape[1])
    im = cv2.copyMakeBorder(im, 30, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=(235, 235, 235))
    cv2.putText(im, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    return im
out, args = sys.argv[1], sys.argv[2:]
rows = []
while args:
    label, key, idx, before, after, args = args[0], args[1], int(args[2]), args[3], args[4], args[5:]
    _, rimg, rmask = rm.load_refs(key)[idx]
    ys, xs = np.where(rmask)
    tiles = []
    if before != "-":
        tiles.append(fit(cut_capture(before), f"{label}: before"))
    tiles.append(fit(cut_capture(after), f"{label}: after"))
    tiles.append(fit(rimg[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "Megaplants reference"))
    row = np.hstack([np.hstack([t, np.full((t.shape[0], 12, 3), 255, np.uint8)]) for t in tiles])
    rows.append(row)
w = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 12, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
