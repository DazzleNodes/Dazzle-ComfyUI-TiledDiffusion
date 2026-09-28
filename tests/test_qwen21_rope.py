"""Qwen-Image 2.1 per-tile global RoPE tests.

Qwen-Image 2.1 (comfy/ldm/qwen_image21/model.py) has no process_img: its
build_sequence() centres every image grid's positions on 0, so a tile gets
positions centred on the TILE. _install_qwen21_rope_patch moves them to where
the same cells sit on the full canvas.

The invariant pinned here, against the REAL model class (tiny config, CPU):
a tile's position encodings must equal the full-canvas encodings for the
same cells. Plus: tile-shaped reference slices move with the tile,
whole-canvas references and text rows do not, no state = untouched output,
and an upstream layout change falls back instead of guessing.

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
import comfy.ops
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel


def _load_td():
    pkg = "td_q21_test_pkg"
    m = types.ModuleType(pkg); m.__path__ = [REPO]; sys.modules[pkg] = m
    su = importlib.util.spec_from_file_location(pkg + ".utils", os.path.join(REPO, "utils.py"))
    u = importlib.util.module_from_spec(su); sys.modules[pkg + ".utils"] = u; su.loader.exec_module(u)
    st = importlib.util.spec_from_file_location(pkg + ".tiled_diffusion", os.path.join(REPO, "tiled_diffusion.py"))
    td = importlib.util.module_from_spec(st); sys.modules[pkg + ".tiled_diffusion"] = td; st.loader.exec_module(td)
    return td

TD = _load_td()
CH, CTX, TXT = 4, 8, 5          # latent channels, text feature dim, text tokens


def _model():
    # tiny but real: head_dim 8 split across the 3 rope axes (2 + 2 + 4)
    return QwenImage21Transformer2DModel(
        in_channels=CH, out_channels=CH, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=CTX, axes_dims_rope=(2, 2, 4),
        operations=comfy.ops.disable_weight_init)


def _record_ids(dm):
    """Instance-level recorder on pe_embedder; 'last' holds the final ids fed in."""
    rec = {}
    fwd = dm.pe_embedder.forward
    def recorder(ids):
        rec['last'] = ids.clone()
        return fwd(ids)
    dm.pe_embedder.forward = recorder
    return rec


def _build(dm, x, refs=(), state=None):
    dm._td_tile_state = state
    ctx = torch.zeros(1, TXT, CTX)
    with torch.no_grad():
        return dm.build_sequence(x, ctx, list(refs), [])


def _state(y0, x0, H, W):
    return {'h_offset_pixels': y0, 'w_offset_pixels': x0,
            'canvas_h_len': H, 'canvas_w_len': W}


def _target_rows(pe, h, w):
    # target image is always last in the sequence, row-major
    return pe[:, -h * w:].reshape(pe.shape[0], h, w, *pe.shape[2:])


def test_1_no_state_is_untouched():
    dm_ref, dm = _model(), _model()
    TD._install_qwen21_rope_patch(dm)
    x = torch.zeros(1, CH, 7, 9)
    _, pe_ref, _ = _build(dm_ref, x)
    _, pe, _ = _build(dm, x, state=None)
    assert torch.equal(pe, pe_ref), "no tile state must leave build_sequence output untouched"
    print("[PASS] test_1_no_state_is_untouched")


def test_2_tile_matches_full_canvas():
    # the core invariant, across odd/even canvas and tile sizes and edge/interior tiles
    cases = [(12, 16, 6, 8), (13, 17, 7, 9), (13, 16, 6, 9), (12, 17, 7, 8)]
    for H, W, h, w in cases:
        dm_full = _model()
        dm = _model()
        TD._install_qwen21_rope_patch(dm)
        _, pe_full, _ = _build(dm_full, torch.zeros(1, CH, H, W))
        full = _target_rows(pe_full, H, W)
        for y0 in (0, (H - h) // 2, H - h):
            for x0 in (0, (W - w) // 2, W - w):
                _, pe, _ = _build(dm, torch.zeros(1, CH, h, w), state=_state(y0, x0, H, W))
                tile = _target_rows(pe, h, w)
                assert torch.equal(tile, full[:, y0:y0 + h, x0:x0 + w]), \
                    f"canvas {H}x{W} tile {h}x{w} at ({y0},{x0}): positions differ from full canvas"
    print("[PASS] test_2_tile_matches_full_canvas")


def test_3_refs_and_text():
    H, W, h, w, y0, x0 = 13, 17, 6, 8, 5, 9
    dm_native, dm = _model(), _model()
    TD._install_qwen21_rope_patch(dm)
    x = torch.zeros(1, CH, h, w)
    ref_tile = torch.zeros(1, CH, h, w)      # a slice our node cut to match the tile
    ref_whole = torch.zeros(1, CH, H, W)     # a whole-canvas reference
    refs = [ref_tile, ref_whole]

    rec_n = _record_ids(dm_native)
    _, _, segs = _build(dm_native, x, refs)
    native = rec_n['last'][0]
    rec = _record_ids(dm)
    _build(dm, x, refs, state=_state(y0, x0, H, W))
    patched = rec['last'][0]

    ceil = lambda n: n - n // 2
    dy, dx = y0 - ceil(H) + ceil(h), x0 - ceil(W) + ceil(w)
    img = [s for s in segs if s[2] is None]           # [ref_tile, ref_whole, target]
    txt = [s for s in segs if s[2] is not None]
    (a0, a1, _), (b0, b1, _), (t0, t1, _) = img

    assert torch.equal(patched[t0:t1, 1], native[t0:t1, 1] + dy), "target rows must shift in y"
    assert torch.equal(patched[t0:t1, 2], native[t0:t1, 2] + dx), "target rows must shift in x"
    assert torch.equal(patched[a0:a1, 1:], native[a0:a1, 1:] + torch.tensor([dy, dx], dtype=native.dtype)), \
        "tile-shaped reference slice must move with the tile"
    assert torch.equal(patched[b0:b1], native[b0:b1]), "whole-canvas reference must stay canvas-centred"
    for s0, s1, _ in txt:
        assert torch.equal(patched[s0:s1], native[s0:s1]), "text rows must be untouched"
    assert torch.equal(patched[:, 0], native[:, 0]), "the sequence-position axis must be untouched"
    print("[PASS] test_3_refs_and_text")


def test_5_upstream_layout_change_falls_back():
    dm = _model()
    native_build = dm.build_sequence
    # simulate an upstream change: pe no longer equals raw.transpose(1, 2)
    dm.build_sequence = lambda x, c, r, s: (lambda o: (o[0], o[1] * 2, o[2]))(native_build(x, c, r, s))
    TD._install_qwen21_rope_patch(dm)
    x = torch.zeros(1, CH, 6, 8)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, pe, _ = _build(dm, x, state=_state(2, 3, 12, 16))
    _, pe_native, _ = _build(_model(), x)
    assert torch.equal(pe, pe_native * 2), "fallback must return the model's own (unshifted) result"
    assert "unexpected position layout" in buf.getvalue()
    assert getattr(dm, '_td_qwen21_layout_bad', False)
    print("[PASS] test_5_upstream_layout_change_falls_back")


def _stand_in(module, **methods):
    # routing reads only the module name and which methods exist
    cls = type("StandIn", (), {k: (lambda self, *a, **k: None) for k in methods})
    cls.__module__ = module
    return cls()


def test_6_routing():
    # the 2.1 crash: a "comfy.ldm.qwen_image" prefix match sent 2.1 to the
    # process_img patch (AttributeError). Routing must pick the 2.1 patch.
    flavour, ok = TD._detect_rope_flavour(_model())
    assert (flavour, ok) == ("qwen21", True), f"Qwen-Image 2.1 routed to {flavour!r}"
    assert TD._detect_rope_flavour(_stand_in("comfy.ldm.qwen_image.model", process_img=1)) == ("qwen", True), \
        "Qwen-Image must keep the process_img patch"
    assert TD._detect_rope_flavour(_stand_in("comfy.ldm.flux.model", process_img=1)) == ("flux", True), \
        "Flux must keep the rope_options path"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        unknown = TD._detect_rope_flavour(_stand_in("comfy.ldm.qwen_image99.model"))
    assert unknown == ("flux", False), "an unknown Qwen variant must not auto-enable any patch"
    assert "Unrecognised Qwen-Image variant" in buf.getvalue()
    assert TD._detect_rope_flavour(None) == ("flux", False)
    print("[PASS] test_6_routing")


def test_7_prefix_cache_off_for_tiling():
    # ComfyUI's prefix cache crashes on tiled access order (hit on a non-first
    # slot; repro in tests/one-offs/qwen21_prefix_cache_select_repro.py).
    # The node turns it off via the model's own switch -- verify the model
    # actually honours what we set, and that a user-chosen dtype survives.
    dm = _model()
    opts = {'transformer_options': {'qwen_image21_cache': {'device': 'auto', 'dtype': 'int8'}}}
    assert TD._disable_qwen21_prefix_cache(opts, dm) is True
    cache_opts = opts['transformer_options']['qwen_image21_cache']
    assert cache_opts == {'device': 'off', 'dtype': 'int8'}, cache_opts
    assert dm.select_prefix_cache(torch.zeros(1, 8), 1, torch.device('cpu'), cache_opts) == (None, False), \
        "the model must treat our setting as cache-off"
    fresh = {}
    assert TD._disable_qwen21_prefix_cache(fresh, dm) is True
    assert fresh['transformer_options']['qwen_image21_cache']['device'] == 'off'
    other = {'transformer_options': {}}
    assert TD._disable_qwen21_prefix_cache(other, _stand_in("comfy.ldm.flux.model", process_img=1)) is False
    assert other == {'transformer_options': {}}, "non-2.1 models must be left alone"
    print("[PASS] test_7_prefix_cache_off_for_tiling")


if __name__ == "__main__":
    test_1_no_state_is_untouched()
    test_2_tile_matches_full_canvas()
    test_3_refs_and_text()
    test_5_upstream_layout_change_falls_back()
    test_6_routing()
    test_7_prefix_cache_off_for_tiling()
    print("ALL PASS")
