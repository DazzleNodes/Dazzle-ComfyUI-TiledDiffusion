# Tiled Diffusion & VAE for ComfyUI (DazzleNodes fork)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/v/release/DazzleNodes/Dazzle-ComfyUI-TiledDiffusion?include_prereleases&label=version)](https://github.com/DazzleNodes/Dazzle-ComfyUI-TiledDiffusion/releases)
[![License](https://img.shields.io/badge/License-GPLv3%20%2B%20CC--BY--NC--SA--4.0-blue.svg)](#license)

Large-image drawing & upscaling with limited VRAM — extended for the DiT era. A fork of [shiimizu/ComfyUI-TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion) with per-tile global RoPE, packed-latent-aware Flux.2 support, reference-latent tiling, and Wan-family ControlNet slicing. Part of the [DazzleNodes](https://github.com/DazzleNodes) collection.

<div align="center">
  <img width="500" alt="Tiled_Diffusion" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/7cb897a3-a645-426f-8742-d6ba5cf04b64">
</div>

## Nodes

- **Tiled Diffusion** — patches a MODEL so sampling runs tile-by-tile ([MultiDiffusion](https://arxiv.org/abs/2302.08113), [Mixture of Diffusers](https://arxiv.org/abs/2302.02412), or [SpotDiffusion](https://arxiv.org/abs/2407.15507))
- **Tiled VAE Encode / Decode** — pkuliyi2015 & Kahsolt's Tiled VAE algorithm

## Features

- **Supported models**: SD1.x / SD2.x / SDXL / SD3, FLUX, **FLUX.2 / Klein** (T2I + I2I with reference latents), Qwen-Image-Edit, Qwen-Image base (see the [Hi-Res Fix recipe](#quick-start))
- **Per-tile global RoPE** (`rope_patch`) — fixes the seams DiT models produce when every tile restarts its positions at (0,0); auto-detected for Flux-family and Qwen-Image-Edit
- **Reference-latent tiling** — each tile attends only to its spatially-aligned slice of the reference; non-canvas references are resampled correctly (packed-latent-aware on Flux.2)
- **Correct tile sizes on every model** — widget pixels convert via the model's own latent format (Flux.2 is 16 px/cell; earlier builds assumed 8 everywhere, silently making Flux.2 tiles 4x heavier)
- **ControlNet support** incl. Wan-family VAEs (5-D latent hints, tuple `downscale_ratio`)
- Img2img upscale, ultra-large generation, `structure_latent` experimental input

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DazzleNodes/Dazzle-ComfyUI-TiledDiffusion.git
```

(If you have the upstream `ComfyUI-TiledDiffusion` installed, remove or disable it first — the node names overlap.)

Verify the install (prints your code version and runs the regression tests): `python tests/verify_install.py` (use your ComfyUI venv's python).

## Quick Start

**Which workflow?** Want your image upscaled *faithfully* — same picture, more detail? Use the **Hi-Res Fix** recipe below. Want a *new, bigger* image in the same spirit as a reference (Flux.2 / Kontext / Qwen-Edit)? Use the **reference** workflow. A reference guides *what kind* of image to make (subject, colors, lighting), not where everything goes; the Hi-Res Fix keeps the layout locked.

**Upscale anything (Hi-Res Fix recipe — recommended):** low-res pass at the model's training resolution → pixel-space upscale to target → `VAE Encode` → `KSampler` at **partial denoise** with the `TiledDiffusion`-patched model. The `denoise` value is the lever: `~0.025` pure polish (composition untouched), `~0.18` fine detail (structure locked), `~0.28-0.42` more inventive, above `~0.6-0.8` tiles start forgetting the structure. Ready-to-load graph: [docs/examples/hires-fix-tiled-refine.json](docs/examples/hires-fix-tiled-refine.json).

**Reference-guided tiling (Flux.2 / Kontext / Qwen-Image-Edit):** wire your reference through `ReferenceLatent` as usual and sample full-denoise with the patched model — each tile sees its aligned slice of the reference. Ready-to-load graph: [docs/examples/flux2-ref-tiled-upscale.json](docs/examples/flux2-ref-tiled-upscale.json).

> [!TIP]
> * Set `tile_overlap` to 0 and `denoise` to 1 to see the tile seams, then adjust to your needs.
> * Increase `tile_batch_size` to increase speed (if your machine can handle it). `rope_patch` forces it to 1.
> * Use the [colorfix node](https://github.com/gameltb/Comfyui-StableSR) if your colors look off.

## Tiled Diffusion Options

| Name              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `method`          | Tiling [strategy](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/fbb24736c9bc374c7f098f82b575fcd14a73936a/scripts/tilediffusion.py#L39-L46). |
| `tile_width` / `tile_height` | Tile size in pixels (converted via the model's latent format — on Flux.2, 512 means a true 512px tile as of v0.2.0). |
| `tile_overlap`    | Tile overlap in pixels.                                      |
| `tile_batch_size` | Tiles processed per forward.                                 |
| `rope_patch`      | *(optional, DiT)* `auto` \| `enable` \| `disable`. Per-tile global RoPE rewrite; `auto` enables for supported DiT models. Forces `tile_batch_size=1`. |
| `rope_scale`      | *(optional, Flux only)* DyPE-style RoPE frequency scale for rendering above training resolution. `1.0` = off. |
| `structure_latent`| *(optional, experimental)* Canvas-sized structural prior injected per-tile via `ref_latents`. For ref-trained models only — see [docs/technical.md](docs/technical.md). |

**Tile arrangement:** divide the input's pixel dimensions by the number of columns/rows you want and feed the results to `tile_width`/`tile_height` (a [Math Expression](https://github.com/pythongosssss/ComfyUI-Custom-Scripts#math-expression) node works well).

## Flux.2 users — note on v0.2.0

Tile/overlap widget values now mean **true pixels** on Flux.2 (earlier builds silently doubled them: a "512" tile was physically 1024, costing 4x the memory). Keep your values for a large memory reduction, or double them to reproduce your previous effective tiling. A one-time console line reports the conversion. Non-canvas reference latents are also now resampled without corrupting Flux.2's packed latent format. Details, receipts, and measurements: [docs/technical.md](docs/technical.md).

## Tiled VAE

<div align="center">
  <img width="900" alt="Tiled_VAE" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/b5850e03-2cac-49ce-b1fe-a67906bf4c9d">
</div>

Recommended tile sizes are suggested on node creation based on available VRAM.

| Name        | Description |
|-------------|-------------|
| `tile_size` | The image is split into tiles, padded with 11/32 pixels in the decoder/encoder. |
| `fast`      | Skips the tile-by-tile GroupNorm accounting by estimating it from a downsampled pass — faster, may shift contrast/brightness slightly. Full mechanics in [docs/technical.md](docs/technical.md#tiled-vae-internals). |
| `color_fix` | Encoder-only semi-fast mode; can restore colors when tiles are small. |

## Documentation

- [docs/technical.md](docs/technical.md) — DiT/RoPE mechanics, Flux.2 packed-latent semantics (with ComfyUI source receipts), reference-latent alignment, Qwen-Image coherence notes, Tiled VAE internals
- [docs/examples/](docs/examples/) — ready-to-load workflow JSONs
- [CHANGELOG.md](CHANGELOG.md)

## Workflows

The following upstream images can also be loaded directly in ComfyUI:

<div align="center">
  <img alt="ComfyUI_07501_" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/c3713cfb-e083-4df4-a310-9467827ee666">
  <p>Simple upscale.</p>
</div>

<div align="center">
  <img alt="ComfyUI_07503_" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/b681b617-4bb1-49e5-b85a-ef5a0f6e4830">
  <p>4x upscale. 3 passes.</p>
</div>

## Background

This fork extends [shiimizu/ComfyUI-TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion) (which reproduced the SOTA methods from the [SD-WebUI extension](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/) by pkuliyi2015 & Kahsolt) with fixes and features for Diffusion-Transformer models, developed while the upstream project was inactive. Reference-latent groundwork drew on [enternalsaga's RefLatent branch](https://github.com/enternalsaga/ComfyUI-TiledDiffusion-RefLatent).

## License

Great thanks to all the contributors! 🎉🎉🎉
The implementation of MultiDiffusion, Mixture of Diffusers, and Tiled VAE code is currently under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) since it was borrowed from the wonderful [SD-WebUI extension](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/). Anything else GPLv3.

## Citation

```bibtex
@article{jimenez2023mixtureofdiffusers,
  title={Mixture of Diffusers for scene composition and high resolution image generation},
  author={Álvaro Barbero Jiménez},
  journal={arXiv preprint arXiv:2302.02412},
  year={2023}
}
```

```bibtex
@article{bar2023multidiffusion,
  title={MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation},
  author={Bar-Tal, Omer and Yariv, Lior and Lipman, Yaron and Dekel, Tali},
  journal={arXiv preprint arXiv:2302.08113},
  year={2023}
}
```
