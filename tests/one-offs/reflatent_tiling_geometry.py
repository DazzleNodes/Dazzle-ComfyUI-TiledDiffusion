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

        # --- Option 2: keep the low-res slice (cheap, ref_p_sl patches) but RESCALE
        # its RoPE coordinates so it spans the whole tile. process_img builds the
        # span as linspace(offset, h_len-1+offset) with h_len = (h_len-1)*scale+1,
        # so the span endpoint is offset + (ref_p-1)*scale. We compute that endpoint
        # FROM the scale formula (not assert 1.0) to verify the formula is right.
        ref_p_o2 = ref_p_sl
        if ref_p_o2 > 1:
            scale_o2 = (img_p - 1) / (ref_p_o2 - 1)
        else:
            scale_o2 = 1.0  # degenerate: 1-patch ref can't be stretched meaningfully
        ref_span_o2 = (float(shift), shift + (ref_p_o2 - 1) * scale_o2)
        cov_o2 = overlap_fraction(img_span, ref_span_o2)

        print(f"{i:>4} | {'broadcast':<10} | {ref_p_bc:>11} | {str(img_span):>14} | "
              f"{str(ref_span_bc):>16} | {cov_bc:>7.2f}")
        print(f"{'':>4} | {'slice (A)':<10} | {ref_p_sl:>11} | {str(img_span):>14} | "
              f"{str(ref_span_sl):>16} | {cov_sl:>7.2f}")
        print(f"{'':>4} | {'resamp (C)':<10} | {ref_p_rs:>11} | {str(img_span):>14} | "
              f"{str(ref_span_rs):>16} | {cov_rs:>7.2f}")
        print(f"{'':>4} | {'option2':<10} | {ref_p_o2:>11} | {str(img_span):>14} | "
              f"{'('+str(int(ref_span_o2[0]))+', '+f'{ref_span_o2[1]:.0f}'+')':>16} | "
              f"{cov_o2:>7.2f}   scale={scale_o2:.2f}")
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
    # Option 2 uses the SAME tiny slice as 'slice A' (cheap), but aligned via RoPE
    # rescale -> coverage 1.0. So its per-tile token cost == slice A's.
    ref_tokens_option2 = ref_tokens_slice
    print(f"ref tokens per tile [option2]   : {ref_tokens_option2:>8}  "
          f"(+{100*ref_tokens_option2/img_tile_tokens:5.1f}% of tile)  "
          f"<- aligned (coverage 1.0) AND cheap")
    if ref_tokens_option2 > 0:
        print(f"  option2 vs resamp C: {ref_tokens_resample/ref_tokens_option2:.1f}x fewer ref tokens/tile")

    # total ref tokens processed across the whole render (per step), a compute proxy
    print(f"\nTotal ref tokens across all tiles (per forward over the grid):")
    print(f"  broadcast : {ref_tokens_broadcast * n_tiles:>10}")
    print(f"  slice (A) : {ref_tokens_slice * n_tiles:>10}")
    print(f"  resamp (C): {ref_tokens_resample * n_tiles:>10}")
    print(f"  option2   : {ref_tokens_option2 * n_tiles:>10}  (same as slice A, but aligned)")

    # Solution C transient: one canvas-sized ref latent resident (16 channels, fp16)
    canvas_ref_elems = 16 * ch * cw
    print(f"\nSolution C transient: one canvas-res ref latent resident")
    print(f"  ~{canvas_ref_elems:,} elems  (~{canvas_ref_elems*2/1024**2:.1f} MB fp16, "
          f"16-ch latent {ch}x{cw})")
    print(f"  (held once per render, sliced per tile -- NOT per-tile resident)")


# Per-model latent geometry (verified from ComfyUI source):
#   Flux.1 / Qwen-Image: patch_size 2, VAE 8x downscale, 16-ch latent
#   Flux.2:              patch_size 1, VAE 16x downscale, 128-ch latent
# (Qwen is 5D [B,16,T,H,W] with T=1 for images, so spatial geometry == Flux.1.)
MODELS = {
    'flux1': {'patch': 2, 'downscale': 8,  'channels': 16},
    'qwen':  {'patch': 2, 'downscale': 8,  'channels': 16},
    'flux2': {'patch': 1, 'downscale': 16, 'channels': 128},
}


def run_one(canvas_l, ref_l, tile_l, overlap_l, patch_size):
    analyze_axis('H', canvas_l[0], ref_l[0], tile_l[0], overlap_l, patch_size)
    analyze_axis('W', canvas_l[1], ref_l[1], tile_l[1], overlap_l, patch_size)
    token_and_memory_summary(tuple(canvas_l), tuple(ref_l), tuple(tile_l),
                             (overlap_l, overlap_l), patch_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(MODELS), default=None,
                   help='Set patch/downscale per model and interpret --*-px as PIXELS.')
    p.add_argument('--canvas-px', type=int, nargs=2, metavar=('H', 'W'),
                   help='canvas pixels H W (with --model)')
    p.add_argument('--ref-px', type=int, nargs=2, metavar=('H', 'W'),
                   help='reference pixels H W (with --model)')
    p.add_argument('--tile-px', type=int, default=512, help='square tile pixels (with --model)')
    p.add_argument('--overlap-px', type=int, default=64, help='tile overlap pixels (with --model)')
    # legacy latent-space interface
    p.add_argument('--canvas-latent', type=int, nargs=2, default=[384, 384], metavar=('H', 'W'))
    p.add_argument('--ref-latent', type=int, nargs=2, default=[128, 128], metavar=('H', 'W'))
    p.add_argument('--tile-latent', type=int, nargs=2, default=[176, 176], metavar=('H', 'W'))
    p.add_argument('--overlap-latent', type=int, default=48)
    p.add_argument('--patch-size', type=int, default=1)
    p.add_argument('--adreitz-all', action='store_true',
                   help="Run Adreitz's scenario (4032x2304 canvas, 1344x768 ref, 512px tiles) "
                        "across flux1/qwen and flux2.")
    args = p.parse_args()

    print("Reference-latent tiling geometry probe (G1: Option-2 verification)")
    print("Mirrors process_img() RoPE coordinate construction (Flux + Qwen-Image).")
    print("coverage 1.00 == ref fully spans the image tile's coordinate range.")
    print("'option2' = low-res slice (cheap) + RoPE rescale -> coverage 1.00 by formula.\n")

    if args.adreitz_all:
        scen = {'canvas_px': (2304, 4032), 'ref_px': (768, 1344), 'tile_px': 512, 'overlap_px': 64}
        for mdl in ('flux1', 'qwen', 'flux2'):
            m = MODELS[mdl]; ds = m['downscale']; ps = m['patch']
            cl = [scen['canvas_px'][0] // ds, scen['canvas_px'][1] // ds]
            rl = [scen['ref_px'][0] // ds,    scen['ref_px'][1] // ds]
            tl = [scen['tile_px'] // ds,      scen['tile_px'] // ds]
            ol = scen['overlap_px'] // ds
            print("\n" + "#" * 80)
            print(f"# MODEL: {mdl}  (patch {ps}, {ds}x downscale, {m['channels']}ch) "
                  f"-- canvas {cl} ref {rl} tile {tl} overlap {ol} latent")
            print("#" * 80)
            run_one(cl, rl, tl, ol, ps)
        return 0

    if args.model:
        m = MODELS[args.model]; ds = m['downscale']; ps = m['patch']
        if not (args.canvas_px and args.ref_px):
            print("--model requires --canvas-px and --ref-px"); return 1
        cl = [args.canvas_px[0] // ds, args.canvas_px[1] // ds]
        rl = [args.ref_px[0] // ds, args.ref_px[1] // ds]
        tl = [args.tile_px // ds, args.tile_px // ds]
        ol = args.overlap_px // ds
        print(f"MODEL {args.model}: patch {ps}, {ds}x downscale -> canvas {cl} ref {rl} tile {tl} latent")
        run_one(cl, rl, tl, ol, ps)
        return 0

    run_one(args.canvas_latent, args.ref_latent, args.tile_latent,
            args.overlap_latent, args.patch_size)

    print("\nDecision read:")
    print("  - 'slice (A)' coverage < 1.00 -> naive slicing misaligns (corner only).")
    print("  - 'resamp (C)' coverage 1.00 by patch-grid parity, but full tile tokens.")
    print("  - 'option2' coverage 1.00 (by the rescale formula) AND slice-A token cost.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
