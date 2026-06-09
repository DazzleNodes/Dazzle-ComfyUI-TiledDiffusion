"""
Forward-pass GPU memory probe for Flux.1 / Flux.2 / Qwen-Image.

Unlike mod_buffer_memory_probe.py (which measures only the TiledDiffusion node's
resident buffers), this LOADS a real diffusion model and runs ONE forward at a
chosen resolution, reporting peak GPU memory. It answers the actual question:
where does the GB-scale memory go -- direct (full canvas) vs per-tile, and how
much a reference latent adds.

Scope / honesty notes:
  - Measures the DiT forward (activations + weights). It does NOT run the text
    encoder or VAE -- text conditioning is synthetic zeroed tensors of the right
    shape (content is irrelevant to memory; only shapes/dtypes matter).
  - "Tiled" peak is approximated by running the forward at TILE resolution: in
    real tiling, tiles run sequentially, so peak memory ~= a single tile's
    forward. (This script does not invoke the node's grid logic.)
  - CUDA only. On MPS the absolute numbers differ; run it there for your box.

Bypassing the sampler means we call BaseModel.apply_model directly:
  apply_model(x, t, c_crossattn=context, **kwargs)  [comfy/model_base.py:188-233]
  -> c_crossattn becomes `context`; remaining kwargs flow to diffusion_model.
  Forward kwargs (comfy/ldm/flux/model.py:344, qwen_image/model.py:417):
    Flux.1: y=[B,vec], guidance=[B]      Flux.2: guidance=[B] (no y)
    Qwen  : attention_mask (optional)
  Reference: pass ref_latents=[tensor] (the forward's param name -- we bypass the
  extra_conds() rename of reference_latents -> ref_latents).

Run (ComfyUI venv; CLOSE the ComfyUI server first so VRAM is free):
  Windows:     C:\\code\\ComfyUI_experiment\\venv\\Scripts\\python.exe tests/one-offs/forward_memory_probe.py <model> [opts]
  Linux/macOS: <comfyui>/venv/bin/python tests/one-offs/forward_memory_probe.py <model> [opts]

Examples:
  ... forward_memory_probe.py C:/.../models/diffusion_models/flux2-dev.safetensors --canvas 3072 --tile 1024 --reference
  ... forward_memory_probe.py C:/.../models/diffusion_models/flux1-dev.sft        --canvas 2048 --tile 1024
  ... forward_memory_probe.py C:/.../flux-2-klein-9b-Q8_0.gguf --gguf --canvas 2048
"""
import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so _probe_device imports


def _find_comfy():
    """Locate ComfyUI (needs comfy/utils.py). Set COMFY_PATH to override; otherwise
    walk up from this file (works when run from an installed node under custom_nodes/)."""
    cands = [os.environ.get("COMFY_PATH")]
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(7):
        cands.append(d)
        d = os.path.dirname(d)
    cands.append(r"C:\code\ComfyUI_experiment")   # local fallback
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "comfy", "utils.py")):
            return c
    raise SystemExit("Could not locate ComfyUI (needs comfy/utils.py). Set COMFY_PATH=/path/to/ComfyUI.")


COMFY = _find_comfy()
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)

import torch
from _probe_device import dev
import comfy.sd
import comfy.model_management as mm


def gb(x):
    return x / 1024 ** 3


def load_model(path, is_gguf):
    """Return a ModelPatcher from a .safetensors or .gguf diffusion model."""
    if is_gguf:
        gguf_dir = os.path.join(COMFY, "custom_nodes", "ComfyUI-GGUF")
        spec = importlib.util.spec_from_file_location(
            "comfyui_gguf_loader", os.path.join(gguf_dir, "loader.py"))
        loader = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_gguf_loader"] = loader
        spec.loader.exec_module(loader)
        ops_spec = importlib.util.spec_from_file_location(
            "comfyui_gguf_ops", os.path.join(gguf_dir, "ops.py"))
        gops = importlib.util.module_from_spec(ops_spec)
        sys.modules["comfyui_gguf_ops"] = gops
        ops_spec.loader.exec_module(gops)
        sd = loader.gguf_sd_loader(path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": gops.GGMLOps()})
    else:
        model = comfy.sd.load_diffusion_model(path, model_options={})
    if model is None:
        raise RuntimeError(f"Failed to load model: {path}")
    return model


def build_conditioning(model, batch, seq_len, device, dtype):
    """Synthetic zeroed conditioning sized to the loaded model.
    Returns (c_crossattn, extra_kwargs). Dims are introspected where possible."""
    dm = model.diffusion_model
    cls = model.__class__.__name__
    # text feature dim: Flux/Qwen both expose txt_in (Linear: in_features = ctx dim)
    txt_dim = getattr(getattr(dm, "txt_in", None), "in_features", None)
    if txt_dim is None:
        txt_dim = {"Flux": 4096, "Flux2": 3840, "QwenImage": 3584}.get(cls, 4096)
    context = torch.zeros(batch, seq_len, txt_dim, device=device, dtype=dtype)

    extra = {}
    if cls in ("Flux", "Flux2"):
        # pooled vector (Flux.1 has vector_in; Flux.2 does not)
        vec_in = getattr(dm, "vector_in", None)
        vec_dim = getattr(getattr(vec_in, "in_layer", vec_in), "in_features", None)
        if vec_in is not None and vec_dim:
            extra["y"] = torch.zeros(batch, vec_dim, device=device, dtype=dtype)
        # guidance scalar (dev models use guidance_embed)
        extra["guidance"] = torch.full((batch,), 3.5, device=device, dtype=dtype)
    elif cls == "QwenImage":
        extra["attention_mask"] = torch.ones(batch, 1, seq_len, device=device, dtype=dtype)
    return context, extra, txt_dim


def measure(model, latent_hw, channels, batch, seq_len, device, dtype, ref_hw=None):
    """Run one forward at the given LATENT (h,w); ref_hw=(h,w) adds a reference of
    that latent size (None = no reference). Returns (alloc_gb, reserved_gb, secs)
    or (errstr, '', secs)."""
    lh, lw = latent_hw
    x = torch.zeros(batch, channels, lh, lw, device=device, dtype=dtype)
    t = torch.tensor([1.0] * batch, device=device, dtype=torch.float32)
    context, extra, _ = build_conditioning(model, batch, seq_len, device, dtype)
    if ref_hw is not None:
        rh, rw = ref_hw
        extra["ref_latents"] = [torch.zeros(batch, channels, rh, rw, device=device, dtype=dtype)]

    dev.empty_cache()
    dev.reset_peak()
    dev.synchronize()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            model.apply_model(x, t, c_crossattn=context, transformer_options={}, **extra)
        dev.synchronize()
        secs = time.perf_counter() - t0
        return gb(dev.peak_allocated()), gb(dev.peak_reserved()), secs
    except dev.oom_errors:
        dev.empty_cache()
        return "OOM", "", time.perf_counter() - t0
    except Exception as e:  # shape/interface mismatch etc -- surface it
        return f"ERR: {type(e).__name__}: {str(e)[:120]}", "", time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to .safetensors / .sft / .gguf diffusion model")
    ap.add_argument("--gguf", action="store_true", help="model is a GGUF file")
    ap.add_argument("--canvas", type=int, default=2048, help="full canvas size in PIXELS (square)")
    ap.add_argument("--tile", type=int, default=1024, help="tile size in PIXELS (square) for the tiled-peak approximation")
    ap.add_argument("--batch", type=int, default=1, help="batch (use 2 for cond+uncond worst case)")
    ap.add_argument("--text-seq-len", type=int, default=256, help="synthetic text token count")
    ap.add_argument("--reference", action="store_true", help="also measure WITH a reference latent")
    ap.add_argument("--reference-px", type=int, default=1024, help="reference latent size in PIXELS (square); ~1MP=1024 is realistic vs a same-size worst case")
    ap.add_argument("--vram-guard", type=float, default=0.95, help="skip a forward whose estimated peak exceeds this fraction of total VRAM (a spilling forward hangs and ignores Ctrl+C)")
    args = ap.parse_args()

    if not dev.available():
        print("No CUDA or MPS device available.")
        return 1
    device = dev.device

    note = "" if dev.peak_is_true else "  (MPS: 'peak' = driver pool high-water, not a true per-op peak)"
    print(f"Device : {dev.name()} [{dev.kind}]{note}")
    print(f"Loading: {args.model}{' (gguf)' if args.gguf else ''}")
    model = load_model(args.model, args.gguf)
    mm.load_models_gpu([model])
    cls = model.model.__class__.__name__
    lf = model.model.latent_format
    ch, ds = lf.latent_channels, lf.spacial_downscale_ratio
    dtype = model.model.get_dtype_inference()
    _, _, txt_dim = build_conditioning(model.model, args.batch, args.text_seq_len, device, dtype)
    print(f"Model  : {cls}  | latent {ch}ch / {ds}x downscale | infer dtype {dtype} | txt dim {txt_dim}")
    print(f"Config : batch {args.batch}, text_seq_len {args.text_seq_len}")
    print()

    def latent_hw(px):
        return (px // ds, px // ds)

    def tokens(hw):
        p = 1 if cls == "Flux2" else 2   # Flux.2 patch_size 1; Flux.1/Qwen patch 2
        return (hw[0] // p) * (hw[1] // p)

    ref_hw = latent_hw(args.reference_px)
    total_gb, resident_gb = gb(dev.mem_total()), gb(dev.mem_used())
    guard_gb = args.vram_guard * total_gb
    print(f"VRAM   : {resident_gb:.1f} GB resident after load / {total_gb:.1f} GB total"
          f"  (skip-guard {guard_gb:.1f} GB)")
    print()

    cfgs = [{"label": "DIRECT (full canvas)", "hw": latent_hw(args.canvas), "ref": None}]
    if args.reference:
        cfgs.append({"label": f"DIRECT + {args.reference_px}px ref", "hw": latent_hw(args.canvas), "ref": ref_hw})
    cfgs.append({"label": "TILED approx (1 tile)", "hw": latent_hw(args.tile), "ref": None})
    if args.reference:
        cfgs.append({"label": f"TILED approx + {args.reference_px}px ref", "hw": latent_hw(args.tile), "ref": ref_hw})
    for i, c in enumerate(cfgs):
        c["order"], c["tok"] = i, tokens(c["hw"]) + (tokens(c["ref"]) if c["ref"] else 0)

    # Run smallest-first so the cheap tile rows calibrate per-token activation, then
    # SKIP any row whose estimated peak would exceed the guard. A forward that spills
    # to host RAM hangs and ignores Ctrl+C, so we refuse to start it. The slope MUST
    # come from the delta between two measured rows -- (peak - resident)/tokens reads
    # ~0 at tile scale (weights dominate) and badly under-estimates the full canvas.
    pts = []  # successful (tokens, reserved_gb), accumulated smallest-first
    DEFAULT_SLOPE = 0.00025  # GB/token, conservative fallback until a 2-point slope exists
    interrupted = False
    for c in sorted(cfgs, key=lambda c: c["tok"]):
        if interrupted:
            c["result"] = ("--", "", 0.0)
            continue
        est = None
        if len(pts) >= 2:
            (t0, p0), (t1, p1) = pts[-2], pts[-1]
            slope = max(0.0, (p1 - p0) / max(1, t1 - t0))     # real GB/token from two points
            est = p1 + (c["tok"] - t1) * slope
        elif len(pts) == 1:
            est = pts[0][1] + (c["tok"] - pts[0][0]) * DEFAULT_SLOPE
        if est is not None and est > guard_gb:
            c["result"] = (f"SKIP ~{est:.0f}GB est", "", 0.0)
            continue
        try:
            c["result"] = measure(model.model, c["hw"], ch, args.batch, args.text_seq_len, device, dtype, c["ref"])
        except KeyboardInterrupt:
            dev.empty_cache()
            c["result"], interrupted = ("INTERRUPTED", "", 0.0), True
            print("\n[Ctrl+C -- stopping after this row; partial results below]")
            continue
        if isinstance(c["result"][1], float):     # track RESERVED (physical footprint), not alloc
            pts.append((c["tok"], c["result"][1]))

    print(f"{'case':<26s} {'latent hxw':<12s} {'tokens':>9s} {'peak alloc':>11s} {'reserved':>10s} {'secs':>7s}")
    print("-" * 80)
    for c in sorted(cfgs, key=lambda c: c["order"]):
        alloc, reserved, secs = c["result"]
        hw = c["hw"]
        a = f"{alloc:.2f} GB" if isinstance(alloc, float) else alloc
        r = f"{reserved:.2f} GB" if isinstance(reserved, float) else (reserved or "")
        print(f"{c['label']:<26s} {f'{hw[0]}x{hw[1]}':<12s} {c['tok']:>9d} {a:>11s} {r:>10s} {secs:>6.1f}s")
    print()
    print("Reads: 'DIRECT' = forward over the whole canvas (what OOMs at high res).")
    print("'TILED approx' = forward at one tile; real tiling peaks ~here since tiles")
    print("run sequentially. The reference rows show the ref-latent memory delta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
