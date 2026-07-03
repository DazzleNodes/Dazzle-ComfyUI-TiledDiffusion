# Changelog

All notable changes to this fork are documented here. Versioning begins at
0.2.0 (2026-07-02); the fork's earlier work — per-tile global RoPE for
Flux/Qwen-Image-Edit, list-of-tensor reference-latent conditioning,
Wan-family-VAE-aware ControlNet hint slicing, the reference resample-to-canvas
fix, profiling tooling — predates versioning and is treated as the implicit
0.1.x line; see `git log` for that history.

## [0.2.0] - 2026-07-02

### Fixed

- **Flux.2 tile sizes were silently 4x.** The widget-px -> latent-cell
  conversion assumed 8 px/cell for every model; ComfyUI's Flux.2 latent is a
  2x2-packed 32ch/8x VAE latent (16 px/cell), so a "512px" tile was physically
  1024x1024 — 4x the tokens and attention memory per forward. Compression is
  now derived from the model's own `latent_format.spacial_downscale_ratio`
  (also corrects ChromaRadiance and video formats; Cascade unchanged).
- **Resampling a packed Flux.2 latent corrupted references.** Interpolating
  the packed 128-channel tensor resamples the four 2x2 sub-position planes on
  misaligned grids, injecting a high-frequency comb (~16px period, up to ~40x
  the correct high-frequency floor at 3x ref-to-canvas) that degrades renders
  at meaningful CFG. Non-canvas references on packed-latent models are now
  unpacked to the 32ch/8x grid, resampled, and repacked.
  `TD_FLUX2_PACKED_RESAMPLE=0` restores the legacy path for A/B diagnosis.
  Verified end-to-end on CUDA: at cfg 6 the legacy path reproduces the
  reported degradation (issue #4) and the fixed path renders coherently under
  identical settings.

### Changed

- **Flux.2 tile/overlap widget semantics** (consequence of the first fix):
  widget values now mean true pixels on Flux.2. Existing Flux.2 setups get
  tiles half the previous linear size at the same values — keep them for a
  ~4x per-forward memory reduction, or double tile and overlap values to
  reproduce the previous effective tiling. A one-time console line reports
  the conversion whenever a model's latent format is not the classic
  8 px/cell.
- One-time console log of reference-latent count/shapes per tile.

### Added

- `pyproject.toml` (versioning begins; ComfyUI Registry metadata).
- `tests/test_flux2_packed_resample.py` — 5 deterministic regression tests
  for the packed resample (roundtrip, equivalence, comb regression,
  non-packed byte-identity, layout probe).
- `tests/one-offs/flux2_packed_resample_probe.py` — comb/phase measurement
  probe behind the numbers above.
- `tests/one-offs/flux2_ref_matrix.py` — A/B/C render-matrix driver
  (control / legacy / fixed reference) against a running ComfyUI server.
- `tests/checklists/v0.2.0__Fix__flux2-packed-latent-fixes.md` — human test
  checklist for the semantics change and fixes.
- README: "Flux.2 tile sizes — semantics fixed" section; corrected the
  token-density note (Flux.1 and Flux.2 are both one token per 16x16 px);
  empirical `denoise` ladder for the Hi-Res Fix recipe.

[0.2.0]: https://github.com/DazzleNodes/Dazzle-ComfyUI-TiledDiffusion/releases/tag/v0.2.0
