"""Optional ctypes bridge for the C attention kernel."""

import ctypes
import sys
from functools import cache
from pathlib import Path

import numpy as np


def _library_path() -> Path:
    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    return Path(__file__).with_name("_microgpt_attention" + suffix)


@cache
def _load_kernel():
    try:
        library = ctypes.CDLL(str(_library_path()))
        kernel = library.microgpt_causal_attention
    except (OSError, AttributeError):
        return None

    array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags=("C_CONTIGUOUS", "ALIGNED"))
    scratch = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
    kernel.argtypes = [
        array, array, array, array, scratch,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]
    kernel.restype = None
    return kernel


def is_available() -> bool:
    return _load_kernel() is not None


def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the C kernel to float32 arrays ordered by head, token and channel."""
    for tensor in (q, k, v):
        if (
            not isinstance(tensor, np.ndarray)
            or tensor.dtype != np.float32
            or tensor.ndim != 3
            or not tensor.flags.c_contiguous
            or not tensor.flags.aligned
        ):
            raise ValueError("q, k and v must be aligned C-contiguous float32 arrays of rank 3")
    if q.shape != k.shape or q.shape != v.shape or not all(q.shape):
        raise ValueError("q, k and v must have the same nonempty shape")

    kernel = _load_kernel()
    if kernel is None:
        raise ValueError("C attention backend is unavailable; build it with python -m microgpt.build_native")
    output = np.empty_like(q)
    scratch = np.empty(q.shape[1], dtype=np.float64)
    kernel(q, k, v, output, scratch, *q.shape)
    return output
