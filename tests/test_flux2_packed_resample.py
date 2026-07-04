"""
Regression tests for the Flux.2 packed-latent reference resample fix.

ComfyUI's Flux.2 sampler-facing latent [B,128,H/16,W/16] is a 2x2 pixel-shuffle
packing of the 32ch/8x VAE latent (pack: comfy/ldm/models/autoencoder.py encode
tail; unpack: comfy/latent_formats.py Flux2.latent_rgb_factors_reshape).
Bilinearly resampling the PACKED tensor interpolates the 2x2 sub-position
planes independently, phase-shifting them against each other by (1 - 1/s) VAE
pixels -- a comb at the latent Nyquist (~16px image period). The fix resamples
in UNPACKED space (tiled_diffusion._resample_ref_to_canvas packed branch;
TD_FLUX2_PACKED_RESAMPLE=0 forces the legacy path for A/B).

Five tests (deterministic, CPU):
  1. pack(unpack(x)) == x exact roundtrip (locks the layout).
  2. Fixed path == interpolate-in-unpacked-space, exact.
  3. Regression: legacy packed path comb >= 5x the fixed path at s=3 (smooth).
  4. Non-packed (16ch) input byte-identical to plain common_upscale.
  5. Known-position layout probe: packed channel c -> (c//4, dy=(c%4)//2, dx=c%2).

Runnable standalone (python tests/test_flux2_packed_resample.py) or via pytest.
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
from comfy.utils import common_upscale


def _load_td():
    pkg = "td_packed_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()


def unpack(t):
    """[B,128,h,w] -> [B,32,2h,2w]; mirrors latent_formats.py Flux2 preview lambda."""
    b, _, h, w = t.shape
    return t.reshape(b, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, 32, h * 2, w * 2)

def pack(t):
    """[B,32,2H,2W] -> [B,128,H,W]; inverse of unpack."""
    b, _, hh, ww = t.shape
    H, W = hh // 2, ww // 2
    return t.reshape(b, 32, H, 2, W, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, 128, H, W)


def _stub(packed=True):
    s = types.SimpleNamespace(latent_is_packed_2x2=packed, _ref_resample_warned=set(),
                              _ref_resample_cache={},
                              _refs_logged=True)
    s._resample_ref_to_canvas = TD.AbstractDiffusion._resample_ref_to_canvas.__get__(s)
    return s

def _resample(e, x_in, packed=True, legacy=False):
    old = os.environ.get("TD_FLUX2_PACKED_RESAMPLE")
    try:
        if legacy:
            os.environ["TD_FLUX2_PACKED_RESAMPLE"] = "0"
        else:
            os.environ.pop("TD_FLUX2_PACKED_RESAMPLE", None)
        return _stub(packed)._resample_ref_to_canvas(e, x_in)
    finally:
        if old is None:
            os.environ.pop("TD_FLUX2_PACKED_RESAMPLE", None)
        else:
            os.environ["TD_FLUX2_PACKED_RESAMPLE"] = old

def _smooth_packed(h, w, seed=0):
    """Smooth unpacked latent (bicubic-upsampled coarse noise), packed."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.randn(1, 32, max(2, h // 2), max(2, w // 2), generator=g)
    u = torch.nn.functional.interpolate(coarse, size=(h * 2, w * 2), mode="bicubic", align_corners=False)
    return pack(u)

def _comb(t):
    """Mean deviation of odd rows from interpolation of adjacent even rows (Nyquist detector)."""
    even, odd = t[..., 0:-2:2, :], t[..., 1:-1:2, :]
    even2 = t[..., 2::2, :]
    return (odd - 0.5 * (even + even2)).abs().mean().item()


# canvas 144x252 packed (Adreitz geometry), ref at s=3
CAN_H, CAN_W = 144, 252
REF = _smooth_packed(48, 84)
X_IN = torch.zeros(1, 128, CAN_H, CAN_W)


def test_1_pack_unpack_roundtrip():
    x = torch.randn(1, 128, 48, 84, generator=torch.Generator().manual_seed(1))
    assert torch.equal(pack(unpack(x)), x), "pack(unpack(x)) != x -- layout drifted"

def test_2_fixed_path_equals_unpacked_interp():
    got = _resample(REF, X_IN)
    want = pack(common_upscale(unpack(REF), CAN_W * 2, CAN_H * 2, "bilinear", "disabled"))
    assert got.shape == (1, 128, CAN_H, CAN_W)
    assert torch.equal(got, want), "packed branch != unpack->interp->repack"

def test_3_legacy_comb_regression():
    fixed = unpack(_resample(REF, X_IN))
    legacy = unpack(_resample(REF, X_IN, legacy=True))
    ratio = _comb(legacy) / max(_comb(fixed), 1e-9)
    assert ratio >= 5.0, f"legacy/fixed comb ratio {ratio:.1f} < 5 -- regression detector broken"

def test_4_non_packed_byte_identical():
    e16 = torch.randn(1, 16, 48, 84, generator=torch.Generator().manual_seed(2))
    x16 = torch.zeros(1, 16, CAN_H, CAN_W)
    got = _resample(e16, x16, packed=False)
    want = common_upscale(e16, CAN_W, CAN_H, "bilinear", "disabled")
    assert torch.equal(got, want), "non-packed path changed -- must stay byte-identical"

def test_5_known_position_layout():
    # place a spike at packed channel c, cell (y,x); it must appear at
    # unpacked channel c//4, position (2y + (c%4)//2, 2x + c%2).
    for c, y, x in [(0, 3, 5), (1, 3, 5), (2, 7, 2), (3, 7, 2), (37, 0, 0), (126, 10, 20)]:
        t = torch.zeros(1, 128, 12, 24)
        t[0, c, y, x] = 1.0
        u = unpack(t)
        cy, cx = 2 * y + (c % 4) // 2, 2 * x + c % 2
        assert u[0, c // 4, cy, cx] == 1.0 and u.sum() == 1.0, f"layout mismatch at c={c}"


if __name__ == "__main__":
    fails = 0
    for name in ["test_1_pack_unpack_roundtrip", "test_2_fixed_path_equals_unpacked_interp",
                 "test_3_legacy_comb_regression", "test_4_non_packed_byte_identical",
                 "test_5_known_position_layout"]:
        try:
            globals()[name]()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            fails += 1
    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    raise SystemExit(1 if fails else 0)
