"""
Regression test for the Mixture-of-Diffusers scaled-weight clamp.

For alternate-resolution 4D conditioning tensors, per-tile bboxes are generated
at a scaled tile size. get_grid_bbox() clamps that size to the tensor's dims
internally, but the gaussian weight callback used to be built from the
UNCLAMPED size -- crashing with a shape mismatch whenever the scaled tile
exceeded the condition tensor (clamp issue identified in upstream PR
shiimizu#79 by xmarre; fixed at the MixtureOfDiffusers call site).

Two tests against the REAL MixtureOfDiffusers:
  1. The old (unclamped) call pattern raises -- documents the bug class so a
     revert to it fails loudly.
  2. The shipped (clamped) pattern succeeds on the same inputs.

Runnable standalone (python tests/test_grid_bbox_weight_clamp.py) or via pytest.
Needs ComfyUI importable (COMFY_PATH env to override discovery).
"""
import importlib.util
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

import torch


def _load_td():
    pkg = "td_clamp_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Scenario: the alternate-res cond (32x32) is SMALLER than the scaled tile (64x64).
SCALED, COND = 64, 32


def _mod():
    return TD.MixtureOfDiffusers()


def test_1_unclamped_weight_raises():
    """The pre-fix call pattern must fail (bbox clamped internally, weight not)."""
    mod = _mod()
    try:
        mod.get_grid_bbox(SCALED, SCALED, 8, 1, COND, COND, DEV,
                          lambda: mod.get_weight(SCALED, SCALED))
    except RuntimeError:
        return
    raise AssertionError("unclamped weight pattern no longer raises -- "
                         "either get_grid_bbox semantics changed or this test is stale")


def test_2_clamped_weight_succeeds():
    """The shipped pattern: clamp BEFORE building the weight."""
    mod = _mod()
    tw, th = min(SCALED, COND), min(SCALED, COND)
    batches = mod.get_grid_bbox(tw, th, 8, 1, COND, COND, DEV,
                                lambda: mod.get_weight(tw, th))
    n = sum(len(b) for b in batches)
    assert n >= 1, f"expected at least one bbox, got {n}"


if __name__ == "__main__":
    fails = 0
    for name in ("test_1_unclamped_weight_raises", "test_2_clamped_weight_succeeds"):
        try:
            globals()[name]()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            fails += 1
    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    raise SystemExit(1 if fails else 0)
