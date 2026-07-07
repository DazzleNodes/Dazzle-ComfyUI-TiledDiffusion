# Changelog

All notable changes to this fork are documented here. Versioning begins at 0.2.0 (2026-07-02); the fork's earlier work — per-tile global RoPE for Flux/Qwen-Image-Edit, list-of-tensor reference-latent conditioning, Wan-family-VAE-aware ControlNet hint slicing, the reference resample-to-canvas fix, profiling tooling — predates versioning and is treated as the implicit 0.1.x line; see `git log` for that history.

## [0.2.6] - 2026-07-07

### Changed

- **`TD_DIAG` CUDA snapshots now include driver-level numbers** (`driver free=X GB of Y GB` via `torch.cuda.mem_get_info`). ComfyUI enables the `cudaMallocAsync` allocator backend by default on supported GPUs (`cuda_malloc.py`), which makes `torch.cuda.memory_allocated/reserved` blind to most real usage -- field log showed 0.45 GB reported while ~23 GB of model weights were actively sampling. The driver-level query sees every allocator, so weights-class retention (the pinned-model-copy bug class from issue #4) stays visible on CUDA. The MPS branch already reported driver-level numbers.

## [0.2.5] - 2026-07-07

### Fixed

- **`TD_DIAG` memory line now prints at every run start.** The snapshot previously lived at grid init, which only executes on a resolution change -- so the documented "run twice and compare the memory lines" protocol produced no line on the second run. A run-boundary detector at the sampling wrapper (sigmas only decrease within a run; an increase marks a new run) now emits one `[TD-DIAG] run-start` line per run. Pinned by 3 tests (`tests/test_td_diag_run_tick.py`): silent when unset, one line per run, robust to partial-denoise starts and repeated equal sigmas.

## [0.2.4] - 2026-07-06

### Fixed

- **Model weights pinned across runs (issue #4, the run-2 memory step on MPS):** the per-tile RoPE machinery stored a strong reference to the diffusion model module on the tiling impl, which survives between runs inside ComfyUI's loaded-model registry (via the unet-function wrapper). With unload nodes in the workflow (UnloadModel / UnloadAllModels), the next run reloads the model while the old copy stays pinned -- two resident copies, ~30 GB for Flux.2, exactly one stale copy at a time (replaced per run, hence flat-after-run-2). Now a weakref: unload actually frees; sampling is unaffected (the model is strongly held by ComfyUI during a run). Matches the reporter's signature: +25-33 GB at run 2 then flat, clean on pre-RoPE builds.
- **Mid-run canvas refresh dropped the RoPE configuration** (pre-existing since the April RoPE work): `reset()` now preserves rope flavour/scale, patch size, the model weakref, and structure_latent alongside the tile settings.
- **Tiled VAE decode crashed on Flux.2's VAE** (`give_pre_end` attribute missing on the 32ch decoder): SD-era attributes are now getattr-guarded. Measured on a real 4032x2304 refine: our Tiled VAE Decode deviates from a single-pass decode by mean 1.3/255 (p99 5) vs ComfyUI core `VAEDecodeTiled`'s 4.8/255 (p99 23) -- 3.6x less of the per-tile color/brightness drift that reads as ghosting/halos on flat regions.

### Added

- **`TD_DIAG=1` env flag:** one-block diagnostic report -- node file hash, method/tile settings, latent_format (class/channels/ratio), computed compression, packed flag, rope state, seam biases, torch version, plus memory snapshots (cuda allocated/reserved or mps current/driver) at apply and at every run's grid init. Run a workflow twice and the pasted console shows cross-run retention directly. Restart ComfyUI after setting/unsetting (read at import).

## [0.2.3] - 2026-07-04

### Added

- **`seam_bias_y` / `seam_bias_x` widgets (experimental, Mixture of Diffusers)**: shift each tile's blend-weight peak down/right by N latent cells (default 0.0 = the mathematically centered #5-correct weights, byte-identical, pinned by test). Background: the v0.2.1 weights correction removed an accidental asymmetry that had been acting as a tile tie-breaker, and perfectly symmetric blending can deadlock weakly-anchored tiles into a collage in full-denoise reference workflows (A/B verified against an archived render: same config coherent on pre-fix weights, collage on corrected weights). `seam_bias_y=0.5` reproduces the historical blend behavior exactly on square tiles (pinned); recommended full-denoise ref recipe: `seam_bias_y 0.5` + cfg ~4. Verified alternatives needing no knob: `tile_overlap` ~128 (~1.6x slower) or the SpotDiffusion method (some composition drift). Ladder of six A/B renders in the v0.2.3 checklist.
- **Coherence gate** (`tests/checklists/v0.2.3__Feature__seam-bias-and-coherence-gate.md`): a required render-verification matrix for any future blend-weight change — unit pins guarantee weight shapes, not render outcomes, and this class of regression is invisible to them.

## [0.2.2] - 2026-07-04

### Fixed

- **Reference-resample memory churn** (issue #4, MPS batch-over-batch growth): `_resample_ref_to_canvas` ran per tile per step -- 1,000+ identical recomputations per run at a 3x geometry, ~28 GB of transient alloc/free churn that CUDA's allocator recycles invisibly but the MPS allocator reportedly retains/fragments. The resampled reference is now cached once per run (fingerprint-keyed so a recycled `data_ptr` or in-place edit cannot serve a stale tensor; capped at 8 entries; byte-identical output pinned by test). Regression tests in `tests/test_ref_resample_cache.py`; suite wired into `verify_install.py`.

### Added

- `tests/assets/avernus_cdb_cover_1024x1024.jpg` — detail-rich render-verification image (dark gradients + fine linework expose tile artifacts the default `example.png` hides), referenced from the examples README and checklists. Artwork (c) Dustin Darcy (*Avernus Cube*), test/demo use only, not under the code licenses.
- `tests/verify_install.py` — one-command install/consistency check: prints the exact code version (git commit + content hash, so zip installs are identifiable) and runs the regression suites on the default torch device (doubles as an MPS math check on Apple Silicon).

## [0.2.1] - 2026-07-03

### Fixed

- **Mixture-of-Diffusers gaussian blend weights** (fix by Adreitz, #5; the per-axis variance defect was independently caught upstream in [shiimizu#77](https://github.com/shiimizu/ComfyUI-TiledDiffusion/pull/77) by pfpb): the y-axis spread was computed from `tile_w` (non-square tiles got a near-flat y-gaussian), the y midpoint sat half a cell low (a 4.5x top-vs-bottom edge-weight asymmetry at tile 32, biasing vertical-seam blending toward the upper tile), and the distribution was unnormalized (cosmetic — MoD blending is scale-invariant). Each axis now uses its own dimension with symmetric `(n-1)/2` midpoints. Inherited from upstream's initial commit; verified numerically before adoption. Regression tests (`tests/test_gaussian_weights.py`) and a human checklist included.

## [0.2.0] - 2026-07-02

### Fixed

- **Mixture-of-Diffusers weight/bbox mismatch for alternate-resolution conditions**: the gaussian weight was built from the unclamped scaled tile size while bbox generation clamps to the condition tensor, crashing when the scaled tile exceeded the condition's dims (clamp issue identified in upstream PR shiimizu#79 by xmarre).
- **Flux.2 tile sizes were silently 4x.** The widget-px -> latent-cell conversion assumed 8 px/cell for every model; ComfyUI's Flux.2 latent is a 2x2-packed 32ch/8x VAE latent (16 px/cell), so a "512px" tile was physically 1024x1024 — 4x the tokens and attention memory per forward. Compression is now derived from the model's own `latent_format.spacial_downscale_ratio` (also corrects ChromaRadiance and video formats; Cascade unchanged).
- **Resampling a packed Flux.2 latent corrupted references.** Interpolating the packed 128-channel tensor resamples the four 2x2 sub-position planes on misaligned grids, injecting a high-frequency comb (~16px period, up to ~40x the correct high-frequency floor at 3x ref-to-canvas) that degrades renders at meaningful CFG. Non-canvas references on packed-latent models are now unpacked to the 32ch/8x grid, resampled, and repacked. `TD_FLUX2_PACKED_RESAMPLE=0` restores the legacy path for A/B diagnosis. Verified end-to-end on CUDA: at cfg 6 the legacy path reproduces the reported degradation (issue #4) and the fixed path renders coherently under identical settings.

### Changed

- **Flux.2 tile/overlap widget semantics** (consequence of the first fix): widget values now mean true pixels on Flux.2. Existing Flux.2 setups get tiles half the previous linear size at the same values — keep them for a ~4x per-forward memory reduction, or double tile and overlap values to reproduce the previous effective tiling. A one-time console line reports the conversion whenever a model's latent format is not the classic 8 px/cell.
- One-time console log of reference-latent count/shapes per tile.
- The resample console note now adds a quality tip at scale factors >= 2: upscaling a reference in latent space softens fine detail (measured ~10x below natural-latent levels at 3x) and can moire texture-dense regions at high CFG — prepare the reference in pixel space instead (decode -> image upscale -> re-encode).
- Reference-tiling guidance (A/B render-verified at 4032x2304): tile size is the coherence lever — keep the tile count near or below ~15 so each tile's reference crop carries a meaningful fraction of the scene; at 45 tiles every tile paints its own miniature scene. Documented in `docs/examples/`.
- `TD_REF_NO_SLICE=1` env var (diagnostic, default off): hands every tile the whole canvas-resolution reference instead of its per-tile slice, for slice-vs-broadcast A/B testing.

### Added

- `pyproject.toml` (versioning begins; ComfyUI Registry metadata).
- `docs/technical.md` — the deep DiT/Flux/Qwen material moved out of the README (which is now a concise DazzleNodes-style overview).
- `docs/examples/` — three minimal ready-to-load workflows: `hires-fix-tiled-refine.json` (the recommended partial-denoise upscale recipe), `flux2-ref-tiled-upscale-pixelspace.json` (reference-guided tiling with a pixel-space canvas-resolution reference — the best-quality route, shipped at the render-verified tile 1024 / overlap 128), and `flux2-ref-tiled-upscale.json` (latent-resampled reference; doubles as a smoke test of the packed resample and its console guard lines).
- `tests/test_flux2_packed_resample.py` — 5 deterministic regression tests for the packed resample (roundtrip, equivalence, comb regression, non-packed byte-identity, layout probe).
- `tests/one-offs/flux2_packed_resample_probe.py` — comb/phase measurement probe behind the numbers above.
- `tests/one-offs/flux2_ref_matrix.py` — A/B/C render-matrix driver (control / legacy / fixed reference) against a running ComfyUI server.
- `tests/checklists/v0.2.0__Fix__flux2-packed-latent-fixes.md` — human test checklist for the semantics change and fixes.
- README: "Flux.2 tile sizes — semantics fixed" section; corrected the token-density note (Flux.1 and Flux.2 are both one token per 16x16 px); empirical `denoise` ladder for the Hi-Res Fix recipe.

[0.2.0]: https://github.com/DazzleNodes/Dazzle-ComfyUI-TiledDiffusion/releases/tag/v0.2.0
