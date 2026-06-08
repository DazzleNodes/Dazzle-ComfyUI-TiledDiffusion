"""
Unit test for AbstractDiffusion._resample_ref_to_canvas (Solution C, Phase F4).

Loads the REAL method from tiled_diffusion.py (faking the package so the
`from .utils import store` relative import resolves) and calls it against a stub
`self`, since the method only touches self._ref_resample_warned plus the
module-level common_upscale/torch. This exercises the shipped code, not a copy.

Verifies:
  1. Canvas-match ref            -> returned unchanged (identity, fast path).
  2. Non-canvas, matching aspect -> resampled to canvas H/W, channels preserved (4D).
  3. Same for a 5D latent (Wan/Qwen [B,C,T,H,W]).
  4. Aspect mismatch             -> returned unchanged (edit/Kontext guard).
  5. ndim mismatch / non-tensor  -> returned unchanged.
  6. Warning emitted once per distinct (ref_hw, canvas_hw).
  7. After resample, a canvas bbox slice has full per-tile coverage (the point).

Run (ComfyUI venv):
  /c/code/ComfyUI_experiment/venv/Scripts/python.exe tests/one-offs/test_resample_ref_to_canvas.py
"""

import os
import sys
import types
import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY = r"C:\code\ComfyUI_experiment"
sys.path.insert(0, COMFY)

import torch


def load_real_module():
    pkgname = "td_test_pkg"
    pkg = types.ModuleType(pkgname)
    pkg.__path__ = [REPO]
    sys.modules[pkgname] = pkg

    spec_u = importlib.util.spec_from_file_location(
        pkgname + ".utils", os.path.join(REPO, "utils.py"))
    utils = importlib.util.module_from_spec(spec_u)
    sys.modules[pkgname + ".utils"] = utils
    spec_u.loader.exec_module(utils)

    spec_td = importlib.util.spec_from_file_location(
        pkgname + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(spec_td)
    sys.modules[pkgname + ".tiled_diffusion"] = td
    spec_td.loader.exec_module(td)
    return td


def make_stub(td):
    # The method is unbound; give it a self with just the attribute it reads.
    return types.SimpleNamespace(
        _ref_resample_warned=set(),
        _resample_ref_to_canvas=td.AbstractDiffusion._resample_ref_to_canvas.__get__(
            types.SimpleNamespace()),
    )


def call(td, stub, e, x_in):
    return td.AbstractDiffusion._resample_ref_to_canvas(stub, e, x_in)


def main():
    td = load_real_module()
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # canvas latent: 384x384, 16 channels
    x_in = torch.zeros(1, 16, 384, 384)

    print("1. canvas-match ref -> identity")
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    e = torch.randn(1, 16, 384, 384)
    out = call(td, stub, e, x_in)
    check("returned the same object (no copy)", out is e)

    print("2. non-canvas matching-aspect ref (128x128) -> resampled to 384x384")
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    e = torch.randn(1, 16, 128, 128)
    out = call(td, stub, e, x_in)
    check("spatial dims now match canvas", tuple(out.shape[-2:]) == (384, 384))
    check("channel count preserved (16)", out.shape[1] == 16)
    check("batch preserved", out.shape[0] == 1)
    check("warned once", len(stub._ref_resample_warned) == 1)

    print("3. 5D latent [1,16,1,128,128] -> [1,16,1,384,384]")
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    x_in5 = torch.zeros(1, 16, 1, 384, 384)
    e5 = torch.randn(1, 16, 1, 128, 128)
    out5 = call(td, stub, e5, x_in5)
    check("5D spatial dims match canvas", tuple(out5.shape[-2:]) == (384, 384))
    check("5D rank preserved", out5.dim() == 5)
    check("5D channels preserved", out5.shape[1] == 16)

    print("4. aspect mismatch (128x256 vs 384x384) -> identity + guard warn")
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    e = torch.randn(1, 16, 128, 256)
    out = call(td, stub, e, x_in)
    check("returned unchanged (edit/Kontext guard)", out is e)
    check("guard warned once", len(stub._ref_resample_warned) == 1)

    print("5. ndim mismatch / non-tensor -> identity")
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    e3 = torch.randn(16, 128, 128)  # rank 3 vs canvas rank 4
    check("rank mismatch -> unchanged", call(td, stub, e3, x_in) is e3)
    sentinel = ["not a tensor"]
    check("non-tensor -> unchanged", call(td, stub, sentinel, x_in) is sentinel)

    print("6. warning de-dup: same mismatch twice -> one warning")
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    e = torch.randn(1, 16, 128, 128)
    call(td, stub, e, x_in)
    call(td, stub, e, x_in)
    check("warned exactly once for repeated mismatch", len(stub._ref_resample_warned) == 1)

    print("7. post-resample per-tile coverage is full")
    # After resampling to canvas, a canvas bbox slice yields a tile-sized ref slice
    # whose patch grid == the image tile's -> coverage 1.00 (the whole point).
    stub = types.SimpleNamespace(_ref_resample_warned=set())
    e = torch.randn(1, 16, 128, 128)
    resampled = call(td, stub, e, x_in)
    bbox_slicer = (Ellipsis, slice(0, 176), slice(0, 176))  # a 176x176 tile
    ref_slice = resampled[bbox_slicer]
    img_slice = x_in[bbox_slicer]
    check("ref slice spatial dims == image tile slice dims",
          tuple(ref_slice.shape[-2:]) == tuple(img_slice.shape[-2:]) == (176, 176))

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
