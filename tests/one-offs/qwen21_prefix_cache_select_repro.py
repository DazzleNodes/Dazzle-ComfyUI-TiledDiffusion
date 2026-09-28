"""Repro: ComfyUI PoseBranchCache.select() crashes on a hit that is not the first slot.

select() does self.slots.remove(s). list.remove compares s with == against each
earlier slot; slots are dicts holding tensor keys, so the comparison is tensor
== tensor. Keys of different lengths (cond vs uncond prompts) raise a shape
error; equal lengths raise "ambiguous truth value".

Untiled sampling alternates cond, uncond, so every hit is on slot 0 and it never
fires. Tiled sampling runs cond for every tile, then uncond for every tile, so
the second uncond hit is on slot 1 -> crash on the first step (Qwen-Image 2.1,
2026-09-28, user log). Run with the ComfyUI venv.
"""
import os, sys
sys.path.insert(0, os.environ.get("COMFY_PATH", r"C:\code\ComfyUI_experiment"))
import torch
from comfy.ldm.wan.model_animate2 import PoseBranchCache

cond = torch.zeros(1, 4 + 86 * 4)     # stand-ins for the user's 86- and 23-token prompt keys
uncond = torch.zeros(1, 4 + 23 * 4)


def run(order):
    cache = PoseBranchCache(store_device="cpu")
    for name, key in order:
        try:
            if not cache.select(key, create=False):
                cache.select(key)          # miss: create the slot
        except RuntimeError as e:
            return f"CRASH on {name}: {e}"
    return "ok"


print("untiled  (cond, uncond, cond, uncond):", run([("cond", cond), ("uncond", uncond)] * 2))
print("tiled x2 (cond, cond, uncond, uncond):", run([("cond", cond), ("cond", cond), ("uncond", uncond), ("uncond", uncond)]))
