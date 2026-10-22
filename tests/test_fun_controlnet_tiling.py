"""Qwen-Image 2.1 Fun ControlNet under tiling (v0.3.1).

The core patch builds its control latent at whatever size the model is called
with, so per tile it would squash the whole control image into the tile. The
node encodes once at full canvas and hands each tile its crop. These pin the
collector (every patch through a .previous chain, nothing else) and the crop
(one encode, each tile's own region, SpotDiffusion's roll followed). Real
behaviour is checked with renders.

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
    pkg = "td_fun_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()


class FakeFunPatch:
    """Mimics QwenImage21FunControlPatch's contract: prepare(h, w) returns early
    when control already has shape (h, w), else builds it at (h, w)."""
    def __init__(self, seed):
        self.control, self.seed, self.encodes, self.active = None, seed, 0, False

    def prepare(self, h, w):
        if self.control is not None and self.control.shape[-2:] == (h, w):
            return
        self.encodes += 1
        g = torch.Generator().manual_seed(self.seed)
        self.control = torch.randn(1, 129, h, w, generator=g)

    def diffusion_model_wrapper(self, *a, **k):
        pass


class Block:
    def __init__(self, control_patch, previous=None):
        self.control_patch, self.previous = control_patch, previous


def test_1_collector():
    a, b, c = FakeFunPatch(1), FakeFunPatch(2), FakeFunPatch(3)
    other = types.SimpleNamespace(previous=None)   # some unrelated dit patch
    # the cascade: three applies chained via .previous on two injection layers
    dit = {("single_block", 0): Block(c, Block(b, Block(a))),
           ("single_block", 8): Block(c, Block(b, Block(a))),
           ("single_block", 16): other}
    found = TD._collect_fun_control_patches({"patches_replace": {"dit": dit}})
    assert found == [c, b, a], f"expected every distinct patch once, got {found}"
    assert TD._collect_fun_control_patches({}) == []
    print("[PASS] test_1_collector")


def test_2_crop_per_tile():
    impl = TD.MixtureOfDiffusers()
    impl.h, impl.w = 12, 16
    p = FakeFunPatch(7)
    impl._fun_full, impl._fun_patches = {}, [p]
    p.prepare(12, 16)
    full = p.control.clone()
    p.control, p.encodes = None, 0

    tiles = [TD.BBox(0, 0, 8, 6), TD.BBox(8, 6, 8, 6)]
    for bbox in tiles:
        impl._inject_fun_controlnet([bbox])
        assert torch.equal(p.control, full[bbox.slicer]), "tile must get its own region of the full-canvas control"
        p.prepare(6, 8)   # what the core wrapper does with the tile's shape: must NOT re-encode
    assert p.encodes == 1, f"expected one full-canvas encode per run, got {p.encodes}"

    roll = lambda t: t.roll(shifts=3, dims=-1)      # SpotDiffusion shift
    impl._inject_fun_controlnet([tiles[0]], roll=roll)
    assert torch.equal(p.control, roll(full)[tiles[0].slicer]), "crop must follow SpotDiffusion's roll"
    print("[PASS] test_2_crop_per_tile")


if __name__ == "__main__":
    test_1_collector()
    test_2_crop_per_tile()
    print("ALL PASS")
