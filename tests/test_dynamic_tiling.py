"""dynamic_tiling decision rule (v0.2.8).

Single untiled pass only when BOTH gates pass: the canvas is at most N tiles'
worth of area, and ComfyUI's own fit check (memory_required * 1.5 < free)
says it fits. Off at 0. Real behaviour is checked with renders; this pins
the rule itself.

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


def _load_td():
    pkg = "td_dyn_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()
GB = 2 ** 30
decide = TD._dynamic_single_pass


def test_1_decision_rule():
    # Qwen-Image 2.1, 1792x1200 at 1200 px tiles: 112x75 canvas vs 75x75 tile = 1.5 tiles' worth
    canvas, tile = 112 * 75, 75 * 75
    assert decide(0.0, canvas, tile, 1 * GB, 20 * GB)[0] is False, "0 must mean off"
    assert decide(2.0, canvas, tile, 1 * GB, 20 * GB)[0] is True, "small canvas + fits -> single pass"
    assert decide(1.0, canvas, tile, 1 * GB, 20 * GB)[0] is False, "1.5 tiles' worth > limit 1 -> tile"
    # memory gate uses ComfyUI's 1.5x margin: 10 GB needed -> 15 GB with margin
    assert decide(2.0, canvas, tile, 10 * GB, 14 * GB)[0] is False, "doesn't fit with the margin -> tile"
    assert decide(2.0, canvas, tile, 10 * GB, 16 * GB)[0] is True
    assert decide(2.0, canvas, tile, None, None)[0] is False, "no estimate -> tile"
    # Flux.2 4032x2304 at 512 px tiles: 252x144 vs 32x32 = ~35 tiles' worth -> stays tiled even if it "fits"
    assert decide(2.0, 252 * 144, 32 * 32, 1 * GB, 30 * GB)[0] is False
    print("[PASS] test_1_decision_rule")


if __name__ == "__main__":
    test_1_decision_rule()
    print("ALL PASS")
