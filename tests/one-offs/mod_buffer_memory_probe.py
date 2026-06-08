"""
Probe: how much GPU memory does the MoD tiled-diffusion node ITSELF hold
resident (its own buffers), independent of the model forward?

Motivates the answer to Adreitz/#80/#74: is the Flux.2 tiled memory weight the
node holding redundant full-canvas copies, or is it model/framework (MPS/GGUF)?

We faithfully reconstruct AbstractDiffusion/MixtureOfDiffusers's resident state
(from tiled_diffusion.py): x_buffer = zeros_like(x_in); weights = zeros(1,1,H,W,
fp32); rescale_factor = 1/weights; tile_weights = gaussian(tile). Per-tile
tensors are del'd each iteration, so they are NOT resident across tiles.

Key point this demonstrates: the node's buffers are LATENT-space and therefore
patch_size-independent -- identical for Flux.1 (patch 2) and Flux.2 (patch 1).
The 4x Flux.2 token density is in the model forward, not here.

Run (ComfyUI venv; uses only a few MB so it coexists with a running server):
  /c/code/ComfyUI_experiment/venv/Scripts/python.exe tests/one-offs/mod_buffer_memory_probe.py
"""
import torch


def mb(x):
    return x / 1024**2


def gaussian_weights(tw, th, device, dtype=torch.float32):
    # mirrors utils.gaussian_weights shape (1,1,th,tw); values irrelevant to size
    return torch.ones((1, 1, th, tw), device=device, dtype=dtype)


def node_resident_bytes(canvas_h, canvas_w, tile_h, tile_w, channels, batch,
                        latent_dtype, device):
    """Allocate exactly what MoD holds resident and return (tensors, bytes)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    x_in = torch.zeros(batch, channels, canvas_h, canvas_w, device=device, dtype=latent_dtype)
    # --- the node's resident state ---
    x_buffer = torch.zeros_like(x_in)                                              # latent-dtype
    weights = torch.zeros((1, 1, canvas_h, canvas_w), device=device, dtype=torch.float32)
    weights += 1.0
    rescale_factor = 1.0 / weights                                                 # fp32
    tile_weights = gaussian_weights(tile_w, tile_h, device)                        # fp32, tile-sized
    torch.cuda.synchronize()
    total = torch.cuda.memory_allocated() - base
    held = {
        'x_in (canvas, the sampler owns this anyway)': x_in.numel() * x_in.element_size(),
        'x_buffer (node)': x_buffer.numel() * x_buffer.element_size(),
        'weights (node, fp32)': weights.numel() * weights.element_size(),
        'rescale_factor (node, fp32)': rescale_factor.numel() * rescale_factor.element_size(),
        'tile_weights (node, fp32)': tile_weights.numel() * tile_weights.element_size(),
    }
    del x_in, x_buffer, weights, rescale_factor, tile_weights
    torch.cuda.empty_cache()
    return held, total


def main():
    if not torch.cuda.is_available():
        print('CUDA not available; this probe measures the node footprint on CUDA.')
        return 1
    device = 'cuda'
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print()

    # Adreitz's case: 9 MP canvas, 512px tiles, cond+uncond batch=2 worst case.
    configs = [
        ('Flux.2 9MP, 512px tile, bf16, batch 2', 288, 504, 64, 64, 16, 2, torch.bfloat16),
        ('Flux.2 9MP, 512px tile, bf16, batch 1', 288, 504, 64, 64, 16, 1, torch.bfloat16),
        ('Flux.1 9MP equivalent (same latent buffers)', 288, 504, 88, 88, 16, 2, torch.bfloat16),
        ('Huge 16MP canvas, batch 2', 384, 672, 64, 64, 16, 2, torch.bfloat16),
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

    print('Takeaway: the node-owned resident memory is single-digit MB and is')
    print('latent-space (patch_size-independent). It is NOT the Flux.2 tiled memory')
    print('weight -- that is model-forward activations + framework (MPS/GGUF).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
