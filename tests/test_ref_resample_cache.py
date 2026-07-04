"""
Regression tests for the reference-resample cache (issue #4, MPS memory churn).

_resample_ref_to_canvas is called per tile per step (1,000+ times per run on a
large canvas) while its input reference is constant for the whole run. The
cache resamples once and returns the stored result, keyed on
(data_ptr, shape, dtype, device, canvas) PLUS a 16-sample content fingerprint
so a recycled data_ptr or an in-place edit cannot serve a stale tensor.

Runnable standalone (python tests/test_ref_resample_cache.py) or via pytest.
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
    pkg = "td_cache_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _impl(packed=True):
    impl = TD.MixtureOfDiffusers()
    impl.latent_is_packed_2x2 = packed
    return impl

def _tensors(packed=True):
    ch = 128 if packed else 16
    torch.manual_seed(7)
    ref = torch.randn(1, ch, 48, 84, device=DEV, dtype=torch.float16)
    x_in = torch.zeros(1, ch, 144, 252, device=DEV, dtype=torch.float16)
    return ref, x_in


def test_1_cache_hit_returns_same_tensor():
    """Second call with the same ref must return the stored object, not recompute."""
    impl = _impl()
    ref, x_in = _tensors()
    a = impl._resample_ref_to_canvas(ref, x_in)
    b = impl._resample_ref_to_canvas(ref, x_in)
    assert a is b, "second call recomputed instead of hitting the cache"
    assert len(impl._ref_resample_cache) == 1


def test_2_cached_result_matches_uncached():
    """The cache must be behavior-invisible: byte-identical to a fresh compute."""
    ref, x_in = _tensors()
    a = _impl()._resample_ref_to_canvas(ref, x_in)
    b = _impl()._resample_ref_to_canvas(ref.clone(), x_in)
    assert torch.equal(a, b), "cached path output differs from fresh compute"


def test_3_inplace_change_busts_cache():
    """Same data_ptr, changed content -> fingerprint must force a recompute."""
    impl = _impl()
    ref, x_in = _tensors()
    a = impl._resample_ref_to_canvas(ref, x_in)
    ref.mul_(2.0)  # same tensor object, same data_ptr, new content
    b = impl._resample_ref_to_canvas(ref, x_in)
    assert a is not b, "stale cache hit after in-place content change"
    fresh = _impl()._resample_ref_to_canvas(ref.clone(), x_in)
    assert torch.equal(b, fresh), "recomputed result incorrect after cache bust"


def test_4_distinct_canvases_cached_separately():
    impl = _impl()
    ref, x_in = _tensors()
    x_in2 = torch.zeros(1, 128, 96, 168, device=DEV, dtype=torch.float16)
    a1 = impl._resample_ref_to_canvas(ref, x_in)
    b1 = impl._resample_ref_to_canvas(ref, x_in2)
    assert a1.shape[-2:] != b1.shape[-2:]
    assert impl._resample_ref_to_canvas(ref, x_in) is a1
    assert impl._resample_ref_to_canvas(ref, x_in2) is b1


def test_5_cache_is_capped():
    """A stress loop of distinct refs must not grow the cache beyond the cap."""
    impl = _impl()
    _, x_in = _tensors()
    for i in range(20):
        torch.manual_seed(i)
        r = torch.randn(1, 128, 48, 84, device=DEV, dtype=torch.float16)
        impl._resample_ref_to_canvas(r, x_in)
    assert len(impl._ref_resample_cache) <= 8, \
        f"cache grew to {len(impl._ref_resample_cache)} entries"


def test_6_nonpacked_path_cached_too():
    impl = _impl(packed=False)
    ref, x_in = _tensors(packed=False)
    a = impl._resample_ref_to_canvas(ref, x_in)
    b = impl._resample_ref_to_canvas(ref, x_in)
    assert a is b


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
