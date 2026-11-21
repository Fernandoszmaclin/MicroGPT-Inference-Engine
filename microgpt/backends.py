"""Causal scaled dot-product attention and backend selection."""

from collections.abc import Callable
from math import sqrt

import numpy as np


Attention = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def causal_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute attention for tensors laid out as (heads, tokens, head_dim)."""
    scores = (q @ k.swapaxes(-1, -2)) / np.float32(sqrt(q.shape[-1]))
    future = np.triu(np.ones(scores.shape[-2:], dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def resolve_attention(backend: str) -> tuple[str, Attention]:
    if backend not in ("numpy", "c", "auto"):
        raise ValueError(f"unknown attention backend: {backend!r}")
    if backend != "numpy":
        try:
            from . import native
        except ImportError:
            native = None
        if native is not None and native.is_available():
            return "c", native.causal_attention
        if backend == "c":
            raise ValueError("C attention backend is unavailable; use numpy or auto")
    return "numpy", causal_attention_numpy
