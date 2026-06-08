"""
Empirical harness: how does a reference latent at a DIFFERENT resolution than the
canvas behave under tiling, across three handling strategies?

This is Phase F1 of the Flux/Flux.2 ref-latent tiling design
(private/claude/2026-06-07__23-04-33__flux-ref-latent-tiling-resolution-mismatch.md).

We are NOT running Flux weights here. The question F1 answers is geometric and is
fully determined by ComfyUI's process_img() RoPE construction
(comfy/ldm/flux/model.py:314-342), which we reimplement faithfully:

    img_ids[..., 1] = linspace(h_offset, h_len - 1 + h_offset, steps=patch_count_h)
    img_ids[..., 2] = linspace(w_offset, w_len - 1 + w_offset, steps=patch_count_w)

The RoPE *coordinate span* of any token grid is therefore [offset, offset + P - 1]
where P = patch count along that axis. Alignment between an image tile and its
reference requires their coordinate spans to COINCIDE. That is the crux of the
design decision, and this harness measures it numerically for three strategies:

  (broadcast) current behavior  -- full low-res ref handed to every tile
  (slice)     Solution A        -- low-res ref sliced to each tile's region
  (resample)  Solution C        -- ref resampled to canvas res, then sliced

For each it reports, per tile:
  - ref token count added to that tile's forward (memory proxy)
  - the ref's RoPE coordinate span vs the image tile's span (alignment proxy)
  - an OVERLAP/COVERAGE figure: what fraction of the image tile's coordinate
    range the reference actually covers (1.0 == fully aligned; <1.0 == ref only
    informs a sub-region; >1.0 or shifted == misaligned)

It also reports peak host-tensor bytes for the resampled ref (Solution C's
transient cost) so the memory tradeoff is grounded in a measurement, not a guess.

Usage (CPU is fine -- this is geometry, not GPU compute):
  /c/code/ComfyUI_experiment/venv/Scripts/python.exe \
      tests/one-offs/reflatent_tiling_geometry.py

  # custom geometry:
  python tests/one-offs/reflatent_tiling_geometry.py \
      --canvas-latent 384 384 --ref-latent 128 128 \
      --tile-latent 176 176 --overlap-latent 48 --patch-size 1

Defaults model Adreitz's case on Flux.2:
  canvas 9MP  -> ~384x384 latent ; ref 1MP -> ~128x128 latent
  tile 176x176 latent ; overlap 48 ; patch_size 1 (Flux.2)
"""

import argparse


def patch_count(n_latent, patch_size):
    """Mirror process_img: h_len = (h + patch_size//2) // patch_size."""
    return (n_latent + (patch_size // 2)) // patch_size


def span(offset, p):
    """RoPE coordinate span [first, last] for a token grid of p patches at offset.
    process_img uses linspace(offset, p-1+offset, steps=p) -> coords offset..offset+p-1.
    """
    if p <= 0:
        return (offset, offset)
    return (offset, offset + p - 1)


def grid_tiles(canvas, tile, overlap):
    """Return list of (start, size) bbox starts along one axis (latent units),
    mirroring split_bboxes' stride = tile - overlap walk with a clamped last tile."""
    stride = max(1, tile - overlap)
    starts = []
    pos = 0
    if tile >= canvas:
        return [(0, canvas)]
    while pos + tile <= canvas:
        starts.append((pos, tile))
        pos += stride
    # clamp a final tile flush to the edge if there's a remainder
    if starts and starts[-1][0] + tile < canvas:
        starts.append((canvas - tile, tile))
    if not starts:
        starts.append((0, min(tile, canvas)))
    return starts


def overlap_fraction(img_span, ref_span):
    """Fraction of the image tile's coordinate range that the ref covers."""
    lo = max(img_span[0], ref_span[0])
    hi = min(img_span[1], ref_span[1])
    inter = max(0.0, hi - lo)
    img_range = max(1e-9, img_span[1] - img_span[0])
    return inter / img_range


def analyze_axis(name, canvas_l, ref_l, tile_l, overlap_l, patch_size):
    print(f"\n=== axis: {name} ===")
    print(f"canvas_latent={canvas_l} ref_latent={ref_l} tile_latent={tile_l} "
          f"overlap_latent={overlap_l} patch_size={patch_size}")

    cf = canvas_l / ref_l if ref_l else 0
    print(f"canvas/ref resolution ratio (cf) = {cf:.3f}"
          f"  ({'clean integer' if abs(cf - round(cf)) < 1e-6 else 'NON-integer'})")

    img_tiles = grid_tiles(canvas_l, tile_l, overlap_l)
    print(f"image tiles along this axis: {len(img_tiles)}  starts/sizes={img_tiles}")

    # image tile patch count + span (with per-tile rope shift = tile start, in patches)
    print(f"\n{'tile':>4} | {'strategy':<10} | {'ref_patches':>11} | "
          f"{'img_span':>14} | {'ref_span':>16} | {'coverage':>8}")
    print("-" * 78)

    for i, (start, size) in enumerate(img_tiles):
        img_p = patch_count(size, patch_size)
        # our rope_patch shifts the image tile to its absolute canvas position,
        # in patch units: shift = start // patch_size (approx; process_img divides offset)
        shift = (start + (patch_size // 2)) // patch_size
        img_span = span(shift, img_p)

        # --- broadcast (current behavior): full ref handed to the tile, shifted by `shift`
        ref_p_bc = patch_count(ref_l, patch_size)
        ref_span_bc = span(shift, ref_p_bc)
        cov_bc = overlap_fraction(img_span, ref_span_bc)

        # --- slice (Solution A): ref sliced to the tile-corresponding region at REF res
        # corresponding ref region size in latent units:
        ref_region = max(1, round(size / cf)) if cf else size
        ref_p_sl = patch_count(ref_region, patch_size)
        ref_span_sl = span(shift, ref_p_sl)   # shifted to same origin as image tile
        cov_sl = overlap_fraction(img_span, ref_span_sl)

        # --- resample (Solution C): ref upsampled to canvas res, then sliced exactly
        # like the image tile -> identical patch grid -> identical span
        ref_p_rs = img_p
        ref_span_rs = img_span
        cov_rs = overlap_fraction(img_span, ref_span_rs)

        print(f"{i:>4} | {'broadcast':<10} | {ref_p_bc:>11} | {str(img_span):>14} | "
              f"{str(ref_span_bc):>16} | {cov_bc:>7.2f}")
        print(f"{'':>4} | {'slice (A)':<10} | {ref_p_sl:>11} | {str(img_span):>14} | "
              f"{str(ref_span_sl):>16} | {cov_sl:>7.2f}")
        print(f"{'':>4} | {'resamp (C)':<10} | {ref_p_rs:>11} | {str(img_span):>14} | "
              f"{str(ref_span_rs):>16} | {cov_rs:>7.2f}")
    return img_tiles


def token_and_memory_summary(canvas_hw, ref_hw, tile_hw, overlap_hw, patch_size):
    ch, cw = canvas_hw
    rh, rw = ref_hw
    th, tw = tile_hw
    oh, ow = overlap_hw

    img_tiles_h = grid_tiles(ch, th, oh)
    img_tiles_w = grid_tiles(cw, tw, ow)
    n_tiles = len(img_tiles_h) * len(img_tiles_w)

    img_tile_tokens = patch_count(th, patch_size) * patch_count(tw, patch_size)

    ref_tokens_broadcast = patch_count(rh, patch_size) * patch_count(rw, patch_size)
    cf_h = ch / rh
    cf_w = cw / rw
    ref_region_h = max(1, round(th / cf_h))
    ref_region_w = max(1, round(tw / cf_w))
    ref_tokens_slice = patch_count(ref_region_h, patch_size) * patch_count(ref_region_w, patch_size)
    ref_tokens_resample = img_tile_tokens  # ref slice == image tile grid

    print("\n\n========== TOKEN / MEMORY SUMMARY (both axes) ==========")
    print(f"canvas_latent={canvas_hw} ref_latent={ref_hw} tile_latent={tile_hw} "
          f"patch_size={patch_size}")
    print(f"tiles: {len(img_tiles_h)} x {len(img_tiles_w)} = {n_tiles}")
    print(f"image tokens per tile           : {img_tile_tokens:>8}")
    print(f"ref tokens per tile [broadcast] : {ref_tokens_broadcast:>8}  "
          f"(+{100*ref_tokens_broadcast/img_tile_tokens:5.1f}% of tile)")
    print(f"ref tokens per tile [slice A]   : {ref_tokens_slice:>8}  "
          f"(+{100*ref_tokens_slice/img_tile_tokens:5.1f}% of tile)")
    print(f"ref tokens per tile [resamp C]  : {ref_tokens_resample:>8}  "
          f"(+{100*ref_tokens_resample/img_tile_tokens:5.1f}% of tile)")

    # total ref tokens processed across the whole render (per step), a compute proxy
    print(f"\nTotal ref tokens across all tiles (per forward over the grid):")
    print(f"  broadcast : {ref_tokens_broadcast * n_tiles:>10}")
    print(f"  slice (A) : {ref_tokens_slice * n_tiles:>10}")
    print(f"  resamp (C): {ref_tokens_resample * n_tiles:>10}")

    # Solution C transient: one canvas-sized ref latent resident (16 channels, fp16)
    canvas_ref_elems = 16 * ch * cw
    print(f"\nSolution C transient: one canvas-res ref latent resident")
    print(f"  ~{canvas_ref_elems:,} elems  (~{canvas_ref_elems*2/1024**2:.1f} MB fp16, "
          f"16-ch latent {ch}x{cw})")
    print(f"  (held once per render, sliced per tile -- NOT per-tile resident)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--canvas-latent', type=int, nargs=2, default=[384, 384],
                   metavar=('H', 'W'), help='canvas latent H W (default 384 384 ~ 9MP/8)')
    p.add_argument('--ref-latent', type=int, nargs=2, default=[128, 128],
                   metavar=('H', 'W'), help='reference latent H W (default 128 128 ~ 1MP/8)')
    p.add_argument('--tile-latent', type=int, nargs=2, default=[176, 176],
                   metavar=('H', 'W'), help='tile latent H W (default 176 176)')
    p.add_argument('--overlap-latent', type=int, default=48,
                   help='tile overlap in latent units (default 48)')
    p.add_argument('--patch-size', type=int, default=1,
                   help='1 for Flux.2, 2 for Flux.1 (default 1)')
    args = p.parse_args()

    print("Reference-latent tiling geometry probe (Phase F1)")
    print("Mirrors comfy/ldm/flux/model.py process_img() RoPE coordinate construction.")
    print("coverage 1.00 == ref fully aligns with the image tile's coordinate range.")
    print("coverage < 1.00 == ref only informs a SUB-REGION of the tile (misaligned).")

    analyze_axis('H', args.canvas_latent[0], args.ref_latent[0],
                 args.tile_latent[0], args.overlap_latent, args.patch_size)
    analyze_axis('W', args.canvas_latent[1], args.ref_latent[1],
                 args.tile_latent[1], args.overlap_latent, args.patch_size)

    token_and_memory_summary(
        tuple(args.canvas_latent), tuple(args.ref_latent),
        tuple(args.tile_latent), (args.overlap_latent, args.overlap_latent),
        args.patch_size)

    print("\nDecision read:")
    print("  - If 'slice (A)' coverage < 1.00, naive slicing misaligns the ref (sub-region).")
    print("  - 'resamp (C)' coverage should be 1.00 by construction (patch-grid parity).")
    print("  - Compare ref-token columns for the memory tradeoff (correctness aside).")


if __name__ == '__main__':
    raise SystemExit(main())
