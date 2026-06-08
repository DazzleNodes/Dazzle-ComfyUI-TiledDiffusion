"""
Synthetic benchmark: does batching N tiles in 1 forward beat N sequential
forwards at Qwen-Image-ish model scale?

This is a quick-and-dirty probe to validate the hypothesis behind Phase S4
of the TiledDiffusion speed optimisation plan (see
private/claude/2026-05-02__09-29-18__tiled-diffusion-speed-optimization.md).

We are NOT loading Qwen weights or running real diffusion forwards. We're
just measuring whether the GPU utilisation pattern at the relevant tensor
shapes favours batching enough to justify the per-tile-RoPE surgery
(~150-250 LOC of monkey-patch extension). Cheap synthetic answer first;
real-Qwen verification later only if this says batching helps.

Usage (with ComfyUI's venv):
  /c/code/ComfyUI_experiment/venv/Scripts/python.exe \
      tests/one-offs/benchmark_tile_batching.py

Defaults approximate Qwen-Image scale:
  hidden_dim=3072, num_heads=24, num_blocks=30, dtype=bf16, num_tiles=4,
  seq_len=7744 (= (176/2)**2 patches per 176-latent tile at patch_size=2).

Decision rule:
  speedup >= 2.0x  -> Phase S4 strongly justified
  1.5x <= speedup < 2.0x  -> Phase S4 worth pursuing; payoff exists
  1.2x <= speedup < 1.5x  -> Marginal; favour smaller wins (c_in caching) first
  speedup < 1.2x  -> Don't pursue Phase S4; lift comes from elsewhere
"""

import argparse
import time

import torch
import torch.nn as nn


def make_block(hidden_dim, num_heads, dtype):
    """Approximation of one Qwen-Image transformer block.

    Real Qwen-Image uses DoubleStream / SingleStream blocks with both
    self-attn and cross-attn to text. For this benchmark we approximate
    with self-attn + FFN since our question is just "does batching help"
    not "what is the absolute speed of Qwen-Image".
    """
    return nn.ModuleDict({
        'attn':  nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dtype=dtype),
        'ffn1':  nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype),
        'ffn2':  nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype),
        'norm1': nn.LayerNorm(hidden_dim, dtype=dtype),
        'norm2': nn.LayerNorm(hidden_dim, dtype=dtype),
    })


class TileMockModel(nn.Module):
    def __init__(self, hidden_dim=3072, num_heads=24, num_blocks=30, dtype=torch.bfloat16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [make_block(hidden_dim, num_heads, dtype) for _ in range(num_blocks)]
        )

    @torch.inference_mode()
    def forward(self, x):
        for b in self.blocks:
            h, _ = b['attn'](x, x, x, need_weights=False)
            x = b['norm1'](x + h)
            h = b['ffn2'](nn.functional.gelu(b['ffn1'](x)))
            x = b['norm2'](x + h)
        return x


def benchmark_sequential(model, tiles):
    """tiles: list of (1, T, D) tensors. Run each through the model independently."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for tile in tiles:
        _ = model(tile)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def benchmark_batched(model, tiles):
    """tiles: list of (1, T, D) tensors. Concatenate to (N, T, D) and run one forward."""
    batched = torch.cat(tiles, dim=0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = model(batched)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden-dim', type=int, default=3072)
    p.add_argument('--num-heads', type=int, default=24)
    p.add_argument('--num-blocks', type=int, default=30)
    p.add_argument('--num-tiles', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=7744,
                   help='Tokens per tile (default = (176/2)**2 for Qwen-Image 176-latent tile, patch_size=2)')
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--scan', action='store_true',
                   help='Run a sweep over (num_tiles, seq_len) to characterise speedup')
    args = p.parse_args()

    if not torch.cuda.is_available():
        print('CUDA not available -- this benchmark is meaningless on CPU.')
        return 1

    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    device = 'cuda'

    print(f'Device: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'GPU memory free: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB')
    print()

    print(f'Model: hidden_dim={args.hidden_dim}, num_heads={args.num_heads}, '
          f'num_blocks={args.num_blocks}, dtype={args.dtype}')
    model = TileMockModel(args.hidden_dim, args.num_heads, args.num_blocks, dtype).to(device).eval()

    if args.scan:
        configs = [
            ('4 tiles, 7744 tokens (Qwen 176-latent)', 4, 7744),
            ('4 tiles, 4096 tokens (Qwen 128-latent)', 4, 4096),
            ('4 tiles, 2304 tokens (smaller tile)',    4, 2304),
            ('2 tiles, 7744 tokens',                    2, 7744),
            ('8 tiles, 7744 tokens',                    8, 7744),
        ]
    else:
        configs = [
            (f'{args.num_tiles} tiles, {args.seq_len} tokens', args.num_tiles, args.seq_len),
        ]

    print()
    print(f'{"Config":<48s}{"Seq (ms)":>14s}{"Batched (ms)":>16s}{"Speedup":>12s}')
    print('-' * 90)

    for label, n_tiles, seq_len in configs:
        tiles = [torch.randn(1, seq_len, args.hidden_dim, device=device, dtype=dtype)
                 for _ in range(n_tiles)]

        for _ in range(args.warmup):
            benchmark_sequential(model, tiles)
            benchmark_batched(model, tiles)

        seq_times = [benchmark_sequential(model, tiles) for _ in range(args.runs)]
        bat_times = [benchmark_batched(model, tiles) for _ in range(args.runs)]

        seq_min = min(seq_times) * 1000
        bat_min = min(bat_times) * 1000
        speedup = seq_min / bat_min

        print(f'{label:<48s}{seq_min:>14.1f}{bat_min:>16.1f}{speedup:>11.2f}x')

        del tiles
        torch.cuda.empty_cache()

    print()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f'Peak GPU memory allocated this run: {peak_gb:.2f} GB')
    print()
    print('Decision rule:')
    print('  speedup >= 2.0x  -> Phase S4 strongly justified')
    print('  1.5x - 2.0x       -> Phase S4 worth pursuing')
    print('  1.2x - 1.5x       -> Marginal; favour c_in caching first')
    print('  < 1.2x            -> Skip Phase S4')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
