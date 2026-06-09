"""
Probe: how much GPU memory does the MoD tiled-diffusion node ITSELF hold
resident (its own buffers), independent of the model forward?

SCOPE / WHAT THIS DOES AND DOES NOT MEASURE (read before citing numbers):
  - It reconstructs ONLY the node's resident buffers (from tiled_diffusion.py):
    x_buffer = zeros_like(x_in); weights = zeros(1,1,H,W, fp32);
    rescale_factor = 1/weights; tile_weights = gaussian(tile). Per-tile tensors
    are del'd each iteration, so they are NOT resident across tiles.
  - It does NOT load any model, does NOT run the model forward, and does NOT
    model reference/Kontext tokens. So it canNOT measure the Flux.2 forward
    activations or the reference-doubling memory -- that is the model/framework
    (and on MPS, the allocator), which this probe says nothing about.

What it DOES establish: the node's own resident footprint is latent-space and
small (tens of MB at a 9MP-class canvas), so the node is not holding large
redundant copies. Note it is NOT patch-independent: for the SAME pixel canvas a
Flux.2 latent (128ch / 16x) has ~2x the elements of a Flux.1 latent (16ch / 8x)
  Flux.2: 128 * (H/16)(W/16) = HW/2 elements
  Flux.1:  16 * (H/8)(W/8)   = HW/4 elements
so the node's x_buffer is ~2x on Flux.2 -- still tens of MB, still negligible
next to the GB-scale model forward.

Run (ComfyUI venv; uses only a few MB so it coexists with a running server):
  Windows:     C:\\code\\ComfyUI_experiment\\venv\\Scripts\\python.exe tests/one-offs/mod_buffer_memory_probe.py
  Linux/macOS: <comfyui>/venv/bin/python tests/one-offs/mod_buffer_memory_probe.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so _probe_device imports
import torch
from _probe_device import dev


def mb(x):
    return x / 1024**2


def gaussian_weights(tw, th, device, dtype=torch.float32):
    # mirrors utils.gaussian_weights shape (1,1,th,tw); values irrelevant to size
    return torch.ones((1, 1, th, tw), device=device, dtype=dtype)


def node_resident_bytes(canvas_h, canvas_w, tile_h, tile_w, channels, batch,
                        latent_dtype, device):
    """Allocate exactly what MoD holds resident and return (held, measured_total).
    canvas_*/tile_* are LATENT dims."""
    dev.synchronize()
    base = dev.current_allocated()
    x_in = torch.zeros(batch, channels, canvas_h, canvas_w, device=device, dtype=latent_dtype)
    # --- the node's resident state ---
    x_buffer = torch.zeros_like(x_in)                                              # latent-dtype
    weights = torch.zeros((1, 1, canvas_h, canvas_w), device=device, dtype=torch.float32)
    weights += 1.0
    rescale_factor = 1.0 / weights                                                 # fp32
    tile_weights = gaussian_weights(tile_w, tile_h, device)                        # fp32, tile-sized
    dev.synchronize()
    total = dev.current_allocated() - base
    held = {
        'x_in (canvas, the sampler owns this anyway)': x_in.numel() * x_in.element_size(),
        'x_buffer (node)': x_buffer.numel() * x_buffer.element_size(),
        'weights (node, fp32)': weights.numel() * weights.element_size(),
        'rescale_factor (node, fp32)': rescale_factor.numel() * rescale_factor.element_size(),
        'tile_weights (node, fp32)': tile_weights.numel() * tile_weights.element_size(),
    }
    del x_in, x_buffer, weights, rescale_factor, tile_weights
    dev.empty_cache()
    return held, total


def main():
    if not dev.available():
        print('No CUDA or MPS device available.')
        return 1
    device = dev.device
    print(f'Device: {dev.name()} [{dev.kind}]')
    print('NOTE: node-buffer footprint only. Does NOT load a model or measure the')
    print('      forward / reference tokens (that is where the GB-scale memory is).')
    print()

    # (label, canvas_h, canvas_w, tile_h, tile_w, channels, batch, dtype) -- LATENT dims.
    # 512px tile: Flux.1 -> 64 latent (512/8); Flux.2 -> 32 latent (512/16).
    configs = [
        ('Flux.1  ~9MP (3072x3072px), 512px tile, bf16, batch 2',  384, 384, 64, 64,  16, 2, torch.bfloat16),
        ('Flux.2  ~9MP (3072x3072px), 512px tile, bf16, batch 2',  192, 192, 32, 32, 128, 2, torch.bfloat16),
        ('Flux.1  ~1MP (1024x1024px), 512px tile, bf16, batch 2',  128, 128, 64, 64,  16, 2, torch.bfloat16),
        ('Flux.2  ~1MP (1024x1024px), 512px tile, bf16, batch 2',   64,  64, 32, 32, 128, 2, torch.bfloat16),
    ]
    for label, ch_h, ch_w, th, tw, c, b, dt in configs:
        held, total = node_resident_bytes(ch_h, ch_w, th, tw, c, b, dt, device)
        node_only = sum(v for k, v in held.items() if '(node' in k)
        print(f'=== {label} ===')
        for k, v in held.items():
            print(f'   {k:<48s} {mb(v):8.2f} MB')
        print(f'   {"NODE-OWNED resident (excl. x_in)":<48s} {mb(node_only):8.2f} MB')
        print(f'   {"measured allocator delta":<48s} {mb(total):8.2f} MB')
        print()

    print('Takeaway: node-owned resident memory is tens of MB, latent-space.')
    print('Flux.2 is ~2x Flux.1 for the same pixel canvas (128ch/16x packs more),')
    print('but both are negligible vs the model-forward activations + framework')
    print('(MPS/GGUF), which this probe does not measure.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
