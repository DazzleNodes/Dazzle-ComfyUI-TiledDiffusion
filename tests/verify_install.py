"""
One-command install & consistency check for Dazzle-ComfyUI-TiledDiffusion.

    python tests/verify_install.py

Run it with the SAME python that runs ComfyUI (your ComfyUI venv). It prints
the exact code version you are running (git commit if available, plus a
content hash of tiled_diffusion.py that identifies the version even for zip
downloads), your torch device, and then runs the regression suites. The
tensor tests run on your default device, so on Apple Silicon this also
exercises the MPS math paths. Paste the whole output block when reporting.
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

def main():
    print("=" * 64)
    print("Dazzle-ComfyUI-TiledDiffusion install check")
    print("=" * 64)

    # Version: git if present, content hash always
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        print(f"git commit : {sha or 'unknown'}{' (LOCAL MODIFICATIONS)' if dirty else ''}")
    except Exception:
        print("git commit : (git not available -- zip install?)")
    with open(os.path.join(REPO, "tiled_diffusion.py"), "rb") as fh:
        print(f"node hash  : tiled_diffusion.py sha256:{hashlib.sha256(fh.read()).hexdigest()[:16]}")

    print(f"python     : {sys.executable}")

    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
              and torch.backends.mps.is_available() else "cpu")
        print(f"torch      : {torch.__version__}  default test device: {dev}")
    except Exception as e:
        print(f"torch      : NOT IMPORTABLE ({e}) -- run with your ComfyUI venv's python")
        return 1

    suites = ["test_gaussian_weights.py", "test_flux2_packed_resample.py",
              "test_grid_bbox_weight_clamp.py"]
    failures = 0
    import_error = None
    for name in suites:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            print(f"[MISSING] {name}  <-- your checkout predates this suite; pull main")
            failures += 1
            continue
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0
        err_text = r.stdout + r.stderr
        if not ok and ("ModuleNotFoundError" in err_text or "ImportError" in err_text):
            # Wrong-interpreter signature: summarize once at the end instead
            # of printing three walls of identical tracebacks.
            for line in err_text.strip().splitlines()[::-1]:
                if "Error" in line:
                    import_error = line.strip()
                    break
            failures += 1
            print(f"[FAIL] {name}: {import_error}")
            continue
        verdict = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {verdict}")
        if not ok:
            failures += 1
            for line in err_text.strip().splitlines()[-6:]:
                print(f"         {line}")

    print("-" * 64)
    if import_error:
        print(f"ENVIRONMENT PROBLEM (not a node problem): {import_error}")
        print("This python cannot import ComfyUI's dependencies. Run this script")
        print("with the SAME python that runs ComfyUI:")
        print("    <your-comfyui-python> tests/verify_install.py")
        print("Note for Windows: invoking the script bare uses the .py file")
        print("association and IGNORES your activated venv. Go through")
        print("    python tests/verify_install.py")
    print("ALL CHECKS PASS" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
