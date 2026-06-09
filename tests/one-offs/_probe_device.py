"""
Tiny CUDA/MPS device shim so the memory probes share one callsite.

CUDA exposes true peak alloc/reserved + mem_get_info. MPS (torch 2.10) exposes
current_allocated_memory / driver_allocated_memory / recommended_max_memory but
NO per-op peak stats and NO mem_get_info -- so on MPS we proxy:
  - footprint / "peak"  -> driver_allocated_memory() : the Metal pool high-water.
    It grows to the forward's peak and is retained, so it is the meaningful number.
  - total VRAM          -> recommended_max_memory()
  - reset-peak          -> empty_cache() (drops the pool back toward live tensors)
MPS has no dedicated OOM class, so an OOM surfaces as RuntimeError.

Every MPS call is getattr-guarded so an older torch degrades gracefully (returns 0)
instead of crashing. NOTE: the MPS path is UNTESTED on real Apple hardware from a
CUDA box -- if you run it on MPS, please report whether the numbers look sane.
"""
import torch


def _mps_ok():
    b = getattr(torch.backends, "mps", None)
    return b is not None and b.is_available()


def _mps(fn, default=0):
    return getattr(torch.mps, fn, lambda: default)()


class _Dev:
    def __init__(self):
        if torch.cuda.is_available():
            self.kind, self.device = "cuda", "cuda"
        elif _mps_ok():
            self.kind, self.device = "mps", "mps"
        else:
            self.kind, self.device = None, "cpu"
        self.peak_is_true = self.kind == "cuda"     # MPS values are pool high-water, not true peak

    def available(self):
        return self.kind is not None

    def name(self):
        if self.kind == "cuda":
            return torch.cuda.get_device_name(0)
        if self.kind == "mps":
            import platform
            return f"Apple MPS ({platform.machine()})"
        return "CPU (no GPU detected)"

    def synchronize(self):
        if self.kind == "cuda":
            torch.cuda.synchronize()
        elif self.kind == "mps":
            getattr(torch.mps, "synchronize", lambda: None)()

    def empty_cache(self):
        if self.kind == "cuda":
            torch.cuda.empty_cache()
        elif self.kind == "mps":
            getattr(torch.mps, "empty_cache", lambda: None)()

    def reset_peak(self):
        if self.kind == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif self.kind == "mps":
            getattr(torch.mps, "empty_cache", lambda: None)()   # no peak stats; reset the pool baseline

    def peak_allocated(self):
        if self.kind == "cuda":
            return torch.cuda.max_memory_allocated()
        if self.kind == "mps":
            return _mps("driver_allocated_memory")              # no true peak; pool high-water is the proxy
        return 0

    def peak_reserved(self):
        if self.kind == "cuda":
            return torch.cuda.max_memory_reserved()
        if self.kind == "mps":
            return _mps("driver_allocated_memory")
        return 0

    def current_allocated(self):
        if self.kind == "cuda":
            return torch.cuda.memory_allocated()
        if self.kind == "mps":
            return _mps("current_allocated_memory")
        return 0

    def mem_total(self):
        if self.kind == "cuda":
            return torch.cuda.mem_get_info()[1]
        if self.kind == "mps":
            return _mps("recommended_max_memory")
        return 0

    def mem_used(self):
        if self.kind == "cuda":
            free, total = torch.cuda.mem_get_info()
            return total - free
        if self.kind == "mps":
            return _mps("driver_allocated_memory")
        return 0

    @property
    def oom_errors(self):
        if self.kind == "cuda":
            return (torch.cuda.OutOfMemoryError,)
        return (RuntimeError,)   # MPS has no dedicated OOM exception


dev = _Dev()
