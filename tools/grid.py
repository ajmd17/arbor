"""Two reference images side by side with a pixel grid, for boxing trees in refs.json.
    python tools/grid.py a.jpg b.jpg out.jpg"""
import sys
import cv2, numpy as np
a, b = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
s = cv2.resize(np.hstack([a, b]), (1920, 540))
for x in range(0, 1920, 96):
    cv2.line(s, (x, 0), (x, 540), (0, 0, 255), 1)
    cv2.putText(s, str((x * 2) % 1920), (x + 2, 12), 0, 0.35, (0, 0, 255), 1)
for y in range(0, 540, 50):
    cv2.line(s, (0, y), (1920, y), (255, 0, 0), 1)
    cv2.putText(s, str(y * 2), (2, y + 12), 0, 0.35, (255, 0, 0), 1)
cv2.imwrite(sys.argv[3], s)
