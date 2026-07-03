"""
A/B/C reference render matrix for the Flux.2 packed-resample bug (issue #4).

Reproduces, end-to-end against a RUNNING ComfyUI server, the experiment that
verified the packed-latent resample corruption and its fix:

  A: control  -- canvas-resolution ref (pixel-space 3x upscale of the base, re-encoded)
  B: bug      -- legacy PACKED-space bilinear resample of the low-res base latent
  C: fix      -- UNPACKED-space resample (unpack 2x2 -> bilinear at 32ch/8x -> repack)

Design note: all three conditions feed a CANVAS-RESOLUTION reference latent, so
the TiledDiffusion node's own resample fast-paths identically in every condition
-- the ONLY variable is the reference tensor's content. B's tensor is produced by
the exact legacy math, C's by the fixed math, offline in this script.

Observed result (RTX 5090, flux2-dev fp8, 2304x1296 canvas, 768x432 ref, cf=3.0,
tile 512/overlap 64, seed 7, 23 steps res_multistep/beta, 2026-07-02):
  cfg 1: A, B, C all coherent (corruption tolerated at low CFG).
  cfg 6: B degrades into repeating high-frequency block garbage (the reported
         symptom); C coherent; A clean. Single-variable isolation => the legacy
         packed-space resample was the cause, cross-platform.

Usage (server must be running; models present):
  python flux2_ref_matrix.py step0     # base 768x432 render -> image + latent
  python flux2_ref_matrix.py refs      # build ref_B / ref_C .latent into ComfyUI input/
  python flux2_ref_matrix.py A|B|C     # one tiled condition at cfg 1
  python flux2_ref_matrix.py A6|B6|C6  # same at cfg 6 + negative prompt (the decisive pair)

Environment overrides (defaults in parentheses):
  M4_HOST   ComfyUI server               (http://127.0.0.1:8188)
  M4_COMFY  ComfyUI root for input/output dirs (C:\\code\\ComfyUI_experiment)
  M4_UNET   diffusion model filename     (flux2_dev_fp8mixed.safetensors)
  M4_UNET_LOADER  loader node class      (UNETLoader; use UnetLoaderGGUF for .gguf)
  M4_CLIP   text encoder filename        (mistral_3_small_flux2_fp8.safetensors)
  M4_VAE    VAE filename                 (flux2-vae.safetensors)
"""
import json
import os
import sys
import time
import urllib.request

HOST = os.environ.get("M4_HOST", "http://127.0.0.1:8188")
COMFY = os.environ.get("M4_COMFY", r"C:\code\ComfyUI_experiment")
COMFY_INPUT = os.path.join(COMFY, "input")
COMFY_OUTPUT = os.path.join(COMFY, "output")

UNET = os.environ.get("M4_UNET", "flux2_dev_fp8mixed.safetensors")
UNET_LOADER = os.environ.get("M4_UNET_LOADER", "UNETLoader")
CLIPN = os.environ.get("M4_CLIP", "mistral_3_small_flux2_fp8.safetensors")
VAEN = os.environ.get("M4_VAE", "flux2-vae.safetensors")

PROMPT = ("aerial photograph of a mediterranean coastal town at golden hour, "
          "red tiled rooftops, narrow winding streets, small harbor with fishing "
          "boats, stone breakwater, detailed architecture, sharp focus")
NEG6 = "blurry, low quality, watermark, jpeg artifacts, oversaturated"
SEED_BASE = 42
SEED_TILED = 7
CANVAS_W, CANVAS_H = 2304, 1296     # packed latent 144 x 81
REF_W, REF_H = 768, 432             # packed latent 48 x 27  (cf = 3.0)


def q(graph):
    req = urllib.request.Request(HOST + "/prompt",
                                 data=json.dumps({"prompt": graph}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    if out.get("node_errors"):
        print(json.dumps(out, indent=1))
        sys.exit(1)
    return out["prompt_id"]


def wait(pid, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{HOST}/history/{pid}") as r:
            h = json.load(r)
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                return h[pid]["outputs"]
            if st.get("status_str") == "error":
                print(json.dumps(h[pid].get("status"), indent=1)[:3000])
                sys.exit(1)
        time.sleep(3)
    print("TIMEOUT")
    sys.exit(1)


def loaders(g, neg=""):
    if UNET_LOADER == "UNETLoader":
        g["u"] = {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}}
    else:  # e.g. UnetLoaderGGUF (ComfyUI-GGUF)
        g["u"] = {"class_type": UNET_LOADER, "inputs": {"unet_name": UNET}}
    g["c"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIPN, "type": "flux2", "device": "default"}}
    g["v"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAEN}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["c", 0]}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["c", 0]}}
    return g


def show(outs):
    for _nid, o in outs.items():
        for kind in ("images", "latents"):
            for f in o.get(kind, []):
                print(kind, "->", f.get("subfolder", ""), f["filename"])


def step0():
    g = loaders({})
    g["fg"] = {"class_type": "FluxGuidance", "inputs": {"guidance": 4.0, "conditioning": ["pos", 0]}}
    g["lat"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": REF_W, "height": REF_H, "batch_size": 1}}
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": ["u", 0], "positive": ["fg", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
        "seed": SEED_BASE, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}}
    g["img"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "m4/flux2_ref_base"}}
    g["sl"] = {"class_type": "SaveLatent", "inputs": {"samples": ["ks", 0], "filename_prefix": "m4/flux2_ref_lowres"}}
    pid = q(g)
    print("queued step0", pid)
    show(wait(pid))


def refs():
    import glob
    import torch
    import safetensors.torch
    sys.path.insert(0, COMFY)
    from comfy.utils import common_upscale

    src = sorted(glob.glob(os.path.join(COMFY_OUTPUT, "m4", "flux2_ref_lowres_*.latent")))[-1]
    d = safetensors.torch.load_file(src)
    lat = d["latent_tensor"].float()          # [1,128,27,48]
    print("source latent:", tuple(lat.shape), "from", os.path.basename(src))
    ch, cw = CANVAS_H // 16, CANVAS_W // 16   # 81, 144

    # B: the legacy path -- bilinear on the PACKED tensor (the bug)
    ref_b = common_upscale(lat, cw, ch, "bilinear", "disabled")
    # C: the fixed path -- unpack 2x2 -> bilinear at 32ch/8x -> repack
    b, _, h, w = lat.shape
    u = lat.reshape(b, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, 32, h * 2, w * 2)
    u = common_upscale(u, cw * 2, ch * 2, "bilinear", "disabled")
    ref_c = u.reshape(b, 32, ch, 2, cw, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, 128, ch, cw)

    diff = (ref_b - ref_c).abs().mean().item()
    print(f"ref_B {tuple(ref_b.shape)} vs ref_C {tuple(ref_c.shape)}  mean|diff|={diff:.4f} (must be >0)")
    for name, t in (("m4_ref_B_legacy.latent", ref_b), ("m4_ref_C_fixed.latent", ref_c)):
        safetensors.torch.save_file(
            {"latent_tensor": t.contiguous(), "latent_format_version_0": torch.tensor([])},
            os.path.join(COMFY_INPUT, name))
        print("wrote input/", name)


def tiled(cond, cfg=1.0, neg=""):
    g = loaders({}, neg=neg)
    if cond == "A":
        import glob
        import shutil
        img = sorted(glob.glob(os.path.join(COMFY_OUTPUT, "m4", "flux2_ref_base_*.png")))[-1]
        shutil.copyfile(img, os.path.join(COMFY_INPUT, "m4_ref_base.png"))  # LoadImage reads input/
        g["li"] = {"class_type": "LoadImage", "inputs": {"image": "m4_ref_base.png"}}
        g["up"] = {"class_type": "ImageScale", "inputs": {"image": ["li", 0], "upscale_method": "bilinear",
                                                          "width": CANVAS_W, "height": CANVAS_H, "crop": "disabled"}}
        g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["up", 0], "vae": ["v", 0]}}
        ref = ["enc", 0]
    else:
        g["ll"] = {"class_type": "LoadLatent",
                   "inputs": {"latent": f"m4_ref_{'B_legacy' if cond == 'B' else 'C_fixed'}.latent"}}
        ref = ["ll", 0]
    g["rl"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["pos", 0], "latent": ref}}
    g["fg"] = {"class_type": "FluxGuidance", "inputs": {"guidance": 4.0, "conditioning": ["rl", 0]}}
    g["td"] = {"class_type": "TiledDiffusion", "inputs": {"model": ["u", 0], "method": "Mixture of Diffusers",
               "tile_width": 512, "tile_height": 512, "tile_overlap": 64, "tile_batch_size": 4}}
    g["lat"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": CANVAS_W, "height": CANVAS_H, "batch_size": 1}}
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": ["td", 0], "positive": ["fg", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
        "seed": SEED_TILED, "steps": 23, "cfg": cfg, "sampler_name": "res_multistep", "scheduler": "beta", "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["v", 0]}}
    g["img"] = {"class_type": "SaveImage", "inputs": {
        "images": ["dec", 0], "filename_prefix": f"m4/matrix_{cond}{'_cfg%g' % cfg if cfg != 1.0 else ''}"}}
    pid = q(g)
    print(f"queued condition {cond} (cfg {cfg})", pid)
    show(wait(pid))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: flux2_ref_matrix.py step0|refs|A|B|C|A6|B6|C6  (see module docstring)")
    cmd = sys.argv[1]
    if cmd == "step0":
        step0()
    elif cmd == "refs":
        refs()
    elif cmd in ("A", "B", "C"):
        tiled(cmd)
    elif cmd in ("A6", "B6", "C6"):
        tiled(cmd[0], cfg=6.0, neg=NEG6)
    else:
        raise SystemExit(f"unknown command {cmd!r} (step0|refs|A|B|C|A6|B6|C6)")
