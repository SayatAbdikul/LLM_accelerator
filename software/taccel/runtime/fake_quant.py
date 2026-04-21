"""Small fake-quant helpers used by Stage 3 runtime tests."""
from __future__ import annotations

import numpy as np


def quantize_int8(x, scale: float) -> np.ndarray:
    """Quantize with the golden-model round-to-nearest-even and int8 saturation."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    q = np.rint(np.asarray(x, dtype=np.float32) / np.float32(scale))
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_int8(x, scale: float) -> np.ndarray:
    return np.asarray(x, dtype=np.int8).astype(np.float32) * np.float32(scale)


def cosine_similarity(a, b) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom == 0.0:
        return 1.0 if np.array_equal(lhs, rhs) else 0.0
    return float(np.dot(lhs, rhs) / denom)


def p99_abs_error_lsb(actual, expected, scale: float = 1.0) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    err = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    return float(np.percentile(err / np.float32(scale), 99))
