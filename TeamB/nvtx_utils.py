"""
nvtx_utils.py
─────────────────────────────────────────────────────────────────────────────
NVTX annotations for Nsight Systems / Nsight Compute.

API note: the nvtx PyPI package (>=0.2) uses push_range/pop_range, NOT the
old start()/end() API which was removed in that version. This file handles
both versions and falls back to torch.cuda.nvtx when the package is absent.
─────────────────────────────────────────────────────────────────────────────
"""

import contextlib
from typing import Optional

# ── torch.cuda.nvtx — always available with CUDA PyTorch ────────────────────
try:
    import torch.cuda.nvtx as _torch_nvtx
    _TORCH_NVTX = True
except ImportError:
    _torch_nvtx = None
    _TORCH_NVTX = False

# ── nvtx PyPI package (optional) ─────────────────────────────────────────────
# Current API (>=0.2): push_range / pop_range
# Old API     (<0.2):  start / end  [removed — do not use]
_NVTX_PKG     = False
_NVTX_PKG_NEW = False
_nvtx_pkg     = None

try:
    import nvtx as _nvtx_pkg
    if hasattr(_nvtx_pkg, "push_range"):
        _NVTX_PKG = True
        _NVTX_PKG_NEW = True
    elif hasattr(_nvtx_pkg, "start"):
        _NVTX_PKG = True
        _NVTX_PKG_NEW = False
    else:
        _nvtx_pkg = None  # unknown version — ignore
except ImportError:
    _nvtx_pkg = None

NVTX_AVAILABLE: bool = _TORCH_NVTX or _NVTX_PKG
DOMAIN = "UAI_PROFILER"


class Color:
    GREEN  = 0x00FF00
    YELLOW = 0xFFFF00
    CYAN   = 0x00FFFF
    BLUE   = 0x0000FF
    RED    = 0xFF0000
    WHITE  = 0xFFFFFF


@contextlib.contextmanager
def NvtxRange(message: str, color: Optional[int] = None, domain: str = DOMAIN):
    """Push/pop an NVTX range. Falls back gracefully across backends."""
    if _NVTX_PKG_NEW:
        # Current nvtx API (>=0.2)
        try:
            _nvtx_pkg.push_range(message=message, color=color, domain=domain)
            try:
                yield
            finally:
                _nvtx_pkg.pop_range(domain=domain)
        except TypeError:
            # Some builds don't accept domain kwarg — retry without it
            _nvtx_pkg.push_range(message=message, color=color)
            try:
                yield
            finally:
                _nvtx_pkg.pop_range()
    elif _NVTX_PKG and not _NVTX_PKG_NEW:
        # Old nvtx API (<0.2)
        try:
            rng = _nvtx_pkg.start(message)
            try:
                yield
            finally:
                _nvtx_pkg.end(rng)
        except Exception:
            yield
    elif _TORCH_NVTX:
        # torch.cuda.nvtx — no color/domain but always reliable
        _torch_nvtx.range_push(message)
        try:
            yield
        finally:
            _torch_nvtx.range_pop()
    else:
        yield  # no-op


@contextlib.contextmanager
def warmup_range(step: int):
    with NvtxRange(f"warmup/step_{step}", color=Color.YELLOW):
        yield


@contextlib.contextmanager
def profiling_region():
    """Outer NVTX range. nsys/ncu capture-range targets this label."""
    with NvtxRange("profiling_region", color=Color.CYAN):
        yield


@contextlib.contextmanager
def profile_step_range(step: int):
    with NvtxRange(f"profile_step_{step}", color=Color.BLUE):
        yield


@contextlib.contextmanager
def generate_range(step: int, n_tokens: int):
    with NvtxRange(f"generate/step_{step}/tokens_{n_tokens}", color=Color.RED):
        yield


def probe_nvtx() -> dict:
    """Check NVTX availability. Returns dict with available, backend, message."""
    if not NVTX_AVAILABLE:
        return {"available": False, "backend": "none",
                "message": "No NVTX backend. pip install nvtx or use CUDA PyTorch."}
    if _NVTX_PKG_NEW:
        backend = "nvtx pkg (push_range/pop_range)"
    elif _NVTX_PKG:
        backend = "nvtx pkg (legacy start/end)"
    else:
        backend = "torch.cuda.nvtx"
    try:
        with NvtxRange("probe", color=Color.WHITE):
            pass
        return {"available": True, "backend": backend,
                "message": f"NVTX OK via {backend}"}
    except Exception as e:
        return {"available": False, "backend": backend,
                "message": f"NVTX probe failed: {e}"}
