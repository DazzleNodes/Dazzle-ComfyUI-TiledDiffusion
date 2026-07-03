"""Crop-ambiguity curve (slicing DWP): how 'generic' are reference crops of a
given size? For each crop size (linear fraction of scene width), sample many
pairs of crops at DIFFERENT positions and measure how alike they look
(correlation of 32x32 luminance). High lookalike-rate = a tile can't infer its
scene position from its ref slice = collage risk.

Deterministic sampling grid (no RNG -- reproducible). CPU-only.
Usage: python ref_crop_ambiguity.py [image]  (default: the m4 base render)
"""
import sys
import numpy as np
from PIL import Image

IMG = sys.argv[1] if len(sys.argv) > 1 else r"C:\code\ComfyUI_experiment\output\ref_tiled\base_1mp_00002_.png"
img = np.asarray(Image.open(IMG).convert("L"), dtype=np.float64)
H, W = img.shape

def thumb(a):
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((32, 32), Image.BILINEAR), dtype=np.float64)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum() / d) if d > 0 else 1.0

print(f"image {W}x{H}   (render-geometry markers: 45-tile crop = 0.127 linear, 15-tile crop = 0.254)")
print(f"{'crop frac':>9} {'crop px':>8} {'pairs':>6} {'mean corr':>10} {'lookalike>0.5':>14}")
for frac in (0.08, 0.127, 0.18, 0.254, 0.35, 0.5):
    cw = int(W * frac); chh = int(H * frac * (W / H) * (H / W))  # square-ish in scene terms
    chh = int(H * frac)
    xs = np.linspace(0, W - cw, 6, dtype=int); ys = np.linspace(0, H - chh, 4, dtype=int)
    crops = [thumb(img[y:y+chh, x:x+cw]) for y in ys for x in xs]
    cs = []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            cs.append(corr(crops[i], crops[j]))
    cs = np.array(cs)
    print(f"{frac:>9.3f} {cw:>4}x{chh:<4} {len(cs):>5} {cs.mean():>10.3f} {(cs > 0.5).mean()*100:>13.0f}%")
