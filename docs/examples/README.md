# Example Workflows

Three minimal workflows covering the node's two canonical patterns. Both use Flux.2 loaders — swap the three loader nodes for any supported model family (the Hi-Res Fix pattern is model-agnostic; drop `FluxGuidance` for non-Flux models).

| File | Pattern | For |
|---|---|---|
| `hires-fix-tiled-refine.json` | Structure-in-latent, partial denoise | Everything, incl. base models — the recommended upscale recipe |
| `flux2-ref-tiled-upscale-pixelspace.json` | Reference-guided, full denoise, **pixel-space ref (best quality)** | Edit/ref-trained models (Flux.2, Flux Kontext, Qwen-Image-Edit) |
| `flux2-ref-tiled-upscale.json` | Reference-guided, full denoise, latent-resampled ref | Same — doubles as a smoke test of the v0.2.0 packed resample |

## hires-fix-tiled-refine.json

The Hi-Res Fix recipe as a graph: `LoadImage` (~1MP) → `ImageScale` to 2304x1296 (pixel-space upscale) → `VAE Encode` → `KSampler` at partial denoise with a `TiledDiffusion`-patched model. The structure lives in the init latent, so tiles refine an already-coherent canvas.

**Set the positive prompt to describe YOUR input image** (it ships with a generic quality prompt, which is fine for low-denoise polish).

**Good test source:** a detail-rich image ships at [`tests/assets/avernus_cdb_cover_1024x1024.jpg`](../../tests/assets/avernus_cdb_cover_1024x1024.jpg) — copy it into `ComfyUI/input/` and point `LoadImage` at it. Its smooth dark background and fine linework reveal tile artifacts that flat images (like the default `example.png`) hide.

**`denoise` is the lever** (the KSampler ships at `0.18`): `~0.025` pure polish, composition untouched; `~0.18` fine detail, structure locked; `~0.28-0.42` more inventive; above `~0.6-0.8` tiles increasingly forget the structure.

## flux2-ref-tiled-upscale.json

A self-contained reference-latent tiled upscale — the distilled form of the issue #4 workflow, matching its exact geometry (1344x768 base, 3x canvas). Two stages sharing one prompt:

1. **Base pass** — a normal 1344x768 (~1MP) generation (fixed seed).
2. **Tiled pass** — the base *latent* is wired through `ReferenceLatent`, and a full-denoise render runs on a 4032x2304 canvas with the `TiledDiffusion`-patched model. The workflow ships the verified coherence recipe: `seam_bias_y 0.5` with cfg 4 (see the note below on why full-denoise tiling wants a tie-breaker).

Both stages save images so you can see exactly what the reference carries. **Set expectations correctly for denoise 1.0:** the tiled render is a *new image guided by* the reference — subject, palette, lighting, and scene character follow the base, but composition is NOT locked (a reference is strong image-prompting, not img2img). If you want the upscale to preserve the base's exact composition, that is the other workflow's job (`hires-fix-tiled-refine.json`, structure in the init latent) — or combine both: feed the upscaled base as the `latent_image` at partial denoise *and* keep the reference attached. (For a lighter run, lower both canvases together, e.g. 768x432 base / 2304x1296 tiled.)

The reference is intentionally NOT canvas resolution: it routes through the packed-aware resample this fork fixed in v0.2.0, so the workflow doubles as a smoke test of that path. With `TD_FLUX2_PACKED_RESAMPLE=0` (env var) you can reproduce the pre-fix corruption at cfg 6 for comparison.

**Known limitation of upscaling the reference in latent space (as this workflow does, 3x):** it is like enlarging a small photo instead of retaking it — smooth areas (sky, water, rooftops) enlarge fine, but busy textures (foliage, distant fine detail) come out blurry, and at high CFG the model fights that blur and can produce a shimmering crosshatch in exactly those areas. This is inherent to enlarging latents, not a tiling bug — the pixel-space variant below avoids it entirely. (Measurements in [docs/technical.md](../technical.md).)

## flux2-ref-tiled-upscale-pixelspace.json

The same two-stage graph, but the reference is upscaled as an IMAGE before re-encoding: base -> `VAEDecode` -> `ImageScale` to canvas size -> `VAEEncode` -> `ReferenceLatent`. Re-encoding the enlarged image gives the model the kind of reference it was trained on (a real photo at full resolution), so the blur-fighting crosshatch above cannot happen.

> [!IMPORTANT]
> **If a full-denoise reference render breaks into a tile collage:** the v0.2.1 blend-weight correction removed an accidental asymmetry that acted as a tile tie-breaker, and perfectly symmetric blending can deadlock weakly-anchored tiles (A/B verified 2026-07-04). Three verified rescues, pick one: set `seam_bias_y` to `0.5` (free; pairs best with cfg ~4 for clean detail), raise `tile_overlap` to ~128 (no knob, ~1.6x slower), or switch method to SpotDiffusion (accepts some composition drift). The Hi-Res Fix workflow is unaffected (its coherence comes from the init latent).

**Tile size is the lever that keeps the picture together.** Each tile only sees its own postage-stamp piece of the reference. If the pieces are too small, no tile knows what the whole picture is — every tile paints its own complete little scene and you get a collage. Bigger tiles = bigger pieces = one coherent image. (A/B verified at 4032x2304 with identical seeds: `tile 512` / 45 tiles collaged; `tile 1024` / 15 tiles — the shipped setting — rendered one coherent scene with no artifacts.) Rule of thumb: **keep the tile count around ~15 or fewer**, and use CFG 4-6 to tune how hard the model listens to the prompt and reference. **Use this variant with large tiles for best quality; use the latent-resample variant to exercise/verify the resample path.**

**Console lines you should see in the ref workflows** (v0.2.0+; if missing you are running stale files — restart ComfyUI after updating the node, it caches the module at startup). The first two print in both variants; the `Reference latent != canvas; resampling ... Tip:` line prints only in the latent-resample variant:

```
[TiledDiffusion] 16 px per latent cell (from model latent_format; earlier builds assumed 8): tile 512x512px -> 32x32 latent cells, overlap 64px -> 4 cells.
[TiledDiffusion] ref_latents: 1 reference(s) per tile [1x128x48x84] (packed 2x2 latent; unpacked-space resample)
```

**Models required** (edit the three loaders to your filenames):

| Loader | Example file |
|---|---|
| `UNETLoader` | `flux2_dev_fp8mixed.safetensors` (any Flux.2 dev; for `.gguf` swap in a GGUF UNet loader) |
| `CLIPLoader` (type `flux2`) | `mistral_3_small_flux2_fp8.safetensors` |
| `VAELoader` | `flux2-vae.safetensors` |
