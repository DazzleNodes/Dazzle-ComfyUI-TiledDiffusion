"""Token-cost table for reference-context designs (slicing DWP Stage 3).

Pure arithmetic, no GPU. For each canvas/tile geometry, compares per-forward
token counts (image tokens + reference tokens) for:
  slice        -- per-tile aligned slice (current default)
  margin+50%   -- slice grown 50% rightward/downward (RoPE-safe direction only)
  +thumb 1/N   -- slice + whole-scene thumbnail at 1/N canvas scale (2nd ref)
  broadcast    -- whole canvas-res ref per tile (TD_REF_NO_SLICE)
Flux.2: 1 token per latent cell (packed 16px/cell).
"""

def grid(canvas_cells, tile_cells, overlap_cells):
    stride = tile_cells - overlap_cells
    n = 1
    pos = 0
    while pos + tile_cells < canvas_cells:
        pos = min(pos + stride, canvas_cells - tile_cells)
        n += 1
    return n

CASES = [  # (label, canvas px WxH, tile px, overlap px)
    ("2304x1296 tile512",  (144, 81),  32, 4),
    ("4032x2304 tile512",  (252, 144), 32, 4),
    ("4032x2304 tile1024", (252, 144), 64, 8),
]
print(f"{'geometry':<22} {'tiles':>5} | {'slice':>7} {'margin+50%':>11} {'+thumb 1/4':>11} {'+thumb 1/3':>11} {'broadcast':>10}")
for label, (cw, ch), t, ov in CASES:
    nx, ny = grid(cw, t, ov), grid(ch, t, ov)
    img = t * t
    slice_ = img          # aligned slice == tile size
    margin = int(t*1.5) * int(t*1.5)                      # grown 50% right+down (interior tile)
    th4 = (cw // 4) * (ch // 4)
    th3 = (cw // 3) * (ch // 3)
    bcast = cw * ch
    def tot(ref): return img + ref
    print(f"{label:<22} {nx*ny:>5} | {tot(slice_):>7,} {tot(margin):>11,} {tot(slice_+th4):>11,} {tot(slice_+th3):>11,} {tot(bcast):>10,}")
print()
print("Per-forward tokens (image + reference). Attention cost scales ~quadratically;")
print("linear layers scale linearly. slice=1.0x baseline per row:")
for label, (cw, ch), t, ov in CASES:
    img = t*t; base = 2*img
    m = img + int(t*1.5)**2; t4 = 2*img + (cw//4)*(ch//4); t3 = 2*img + (cw//3)*(ch//3); b = img + cw*ch
    print(f"{label:<22}  margin {m/base:.2f}x | thumb1/4 {t4/base:.2f}x | thumb1/3 {t3/base:.2f}x | broadcast {b/base:.1f}x")
