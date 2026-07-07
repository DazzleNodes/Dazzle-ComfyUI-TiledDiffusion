"""TD_DIAG run-boundary detector tests (v0.2.5, refs #4).

The remote-debugging protocol is "run the workflow twice, compare the two
[TD-DIAG] run-start memory lines". That promise depends on the tick firing
exactly once per sampling run: sigmas only decrease within a run, so a sigma
increase vs the previous wrapper call marks a new run. These tests pin that
logic (and that the whole thing is silent unless TD_DIAG=1).

Needs ComfyUI importable (COMFY_PATH env to override discovery).
"""

import contextlib
import importlib.util
import io
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


def _load_td(diag: bool):
    """Load a fresh module instance with TD_DIAG set/unset (read at import)."""
    if diag:
        os.environ["TD_DIAG"] = "1"
    else:
        os.environ.pop("TD_DIAG", None)
    pkg = f"td_diag_test_pkg_{int(diag)}"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td


class _Host:
    """Minimal object carrying the tick method's state."""
    def __init__(self, td):
        self._tick = td.AbstractDiffusion._td_diag_run_tick
    def tick(self, sigma):
        self._tick(self, torch.tensor([sigma]))


def _run(td, sigmas):
    host = _Host(td)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for s in sigmas:
            host.tick(s)
    return buf.getvalue().count("[TD-DIAG] run-start")


def test_1_silent_when_flag_off():
    td = _load_td(diag=False)
    n = _run(td, [10.0, 5.0, 1.0, 9.0, 4.0])
    assert n == 0, f"expected silence with TD_DIAG unset, got {n} lines"
    print("[PASS] test_1_silent_when_flag_off")


def test_2_one_line_per_run():
    td = _load_td(diag=True)
    # run 1: 10 -> 1 (decreasing); run 2 starts when sigma jumps back up to 9
    n = _run(td, [10.0, 7.0, 4.0, 1.0, 9.0, 6.0, 2.0])
    assert n == 2, f"expected 2 run-start lines (one per run), got {n}"
    print("[PASS] test_2_one_line_per_run")


def test_3_partial_denoise_and_equal_sigmas():
    td = _load_td(diag=True)
    # partial-denoise run starts mid-schedule (5.0 < the previous run's start);
    # repeated equal sigmas (uncond/cond calls at the same step) must not retrigger
    n = _run(td, [10.0, 10.0, 7.0, 7.0, 1.0, 5.0, 5.0, 2.0])
    assert n == 2, f"expected 2 run-start lines, got {n}"
    print("[PASS] test_3_partial_denoise_and_equal_sigmas")


def test_4_mem_snapshot_driver_level():
    td = _load_td(diag=True)
    snap = td._td_mem_snapshot()
    assert isinstance(snap, str) and snap, f"snapshot should be a non-empty string, got {snap!r}"
    if torch.cuda.is_available():
        # driver-level free/total must be present (sees allocators the torch
        # counters miss, e.g. ComfyUI's default cudaMallocAsync backend)
        assert "driver free=" in snap and " of " in snap, snap
    print("[PASS] test_4_mem_snapshot_driver_level")


if __name__ == "__main__":
    test_1_silent_when_flag_off()
    test_2_one_line_per_run()
    test_3_partial_denoise_and_equal_sigmas()
    test_4_mem_snapshot_driver_level()
    print("ALL PASS")
