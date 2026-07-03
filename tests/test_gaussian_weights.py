"""
Regression tests for the gaussian_weights fix (issue #5, reported with the
corrected formula by Adreitz).

Upstream (shiimizu 37fbe89, verbatim from the A1111 extension's collapse of
albarji/mixture-of-diffusers' two per-axis comprehensions; the half-cell
midpoint predates both -- albarji's own tiling.py lacks the "-1" its
canvas.py twin has, a duplication Adreitz flagged in #5) had three defects:
  1. y-axis variance driven by tile_w (non-square tiles got the wrong spread)
  2. y-midpoint tile_h/2 vs x-midpoint (tile_w-1)/2 -- half-cell asymmetry,
     a 4.5x top-vs-bottom edge-weight ratio at tile 32
  3. missing 1/tile_dim normalization factor (cosmetic: MoD blending divides
     by accumulated weights, so global scale cancels -- pinned by test 4)

Runnable standalone (python tests/test_gaussian_weights.py) or via pytest.
Needs ComfyUI importable (COMFY_PATH env to override discovery).
"""
import importlib.util
import math
import os
import sys
import types

def _find_repo():
    env = os.environ.get("TD_REPO")
    if env and os.path.isfile(os.path.join(env, "tiled_diffusion.py")):
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if (os.path.isfile(os.path.join(d, "tiled_diffusion.py"))
                and os.path.isfile(os.path.join(d, "utils.py"))):
            return d
        d = os.path.dirname(d)
    raise SystemExit("Cannot locate the TiledDiffusion repo; set TD_REPO.")

def _find_comfy(repo):
    cands = [os.environ.get("COMFY_PATH")]
    d = repo
    for _ in range(6):
        cands.append(d)
        d = os.path.dirname(d)
    cands.append(r"C:\code\ComfyUI_experiment")
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "comfy", "utils.py")):
            return c
    raise SystemExit("Cannot locate ComfyUI (needs comfy/utils.py); set COMFY_PATH.")

REPO = _find_repo()
sys.path.insert(0, _find_comfy(REPO))

import numpy as np
import torch


def _load_td():
    pkg = "td_gw_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()

def W(tw, th):
    return TD.gaussian_weights(tw, th).cpu().double().numpy()

def _old_weights(tw, th):
    """The pre-fix upstream formula, kept here to PIN the bug (test 1 asserts
    the old code fails the symmetry the fix guarantees)."""
    f = lambda x, mid, var=0.01: math.exp(-(x-mid)*(x-mid) / (tw*tw) / (2*var)) / math.sqrt(2*math.pi*var)
    xs = [f(x, (tw-1)/2) for x in range(tw)]
    ys = [f(y,  th   /2) for y in range(th)]
    return np.outer(ys, xs)


def test_1_y_symmetry():
    """Weights must be mirror-symmetric vertically (old code: 4.5x edge ratio)."""
    for tw, th in ((32, 32), (32, 8), (16, 48)):
        w = W(tw, th)
        assert np.allclose(w, w[::-1, :], rtol=1e-12), f"y-asymmetric at {tw}x{th}"
    o = _old_weights(32, 32)
    assert not np.allclose(o, o[::-1, :], rtol=1e-3), \
        "old formula unexpectedly symmetric -- bug pin is stale"


def test_2_x_symmetry():
    for tw, th in ((32, 32), (32, 8)):
        w = W(tw, th)
        assert np.allclose(w, w[:, ::-1], rtol=1e-12), f"x-asymmetric at {tw}x{th}"


def test_3_per_axis_spread():
    """y falloff must be governed by tile_h: the normalized y-profile of a
    32x8 tile must equal the normalized x-profile of an 8x32 tile."""
    y_prof = W(32, 8)[:, 16]
    x_prof = W(8, 32)[16, :]
    assert np.allclose(y_prof / y_prof.max(), x_prof / x_prof.max(), rtol=1e-9), \
        "y spread not governed by tile_h"
    # and the old code's 32x8 y-profile is nearly flat by comparison
    o = _old_weights(32, 8)
    assert o[0, 16] / o[3, 16] > 0.4, "old-code flat-y pin is stale"


def test_4_blend_scale_invariance():
    """MoD blends as w1*a+w2*b / (w1+w2): global weight scale must not change
    blend fractions (proves the normalization change is cosmetic)."""
    w = W(32, 32)
    w1, w2 = w[28:32, 16], w[0:4, 16]
    frac = w1 / (w1 + w2)
    ws = w * 987.654
    frac_s = ws[28:32, 16] / (ws[28:32, 16] + ws[0:4, 16])
    assert np.allclose(frac, frac_s, atol=1e-12)


def test_5_x_profile_shape_unchanged():
    """The fix is y-only plus a global scale: the new x-profile times tile_w
    must equal the old x-profile exactly."""
    new_x = W(32, 32)[16, :] / W(32, 32)[16, 16]
    old = _old_weights(32, 32)
    old_x = old[16, :] / old[16, 16]
    # rtol bounded by the module's float32 cast (measured deviation ~4e-8);
    # any real shape change is orders of magnitude larger.
    assert np.allclose(new_x, old_x, rtol=1e-6), "x-axis behavior changed -- fix is not y-only"


if __name__ == "__main__":
    fails = 0
    for name in sorted(k for k in globals() if k.startswith("test_")):
        try:
            globals()[name]()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            fails += 1
    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    raise SystemExit(1 if fails else 0)
