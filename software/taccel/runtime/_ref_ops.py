"""Shared NumPy primitives for the Python reference implementations.

`fp32_reference.py` (torch ground-truth), `w8a16_simulator_reference.py`
(W8A16 like-for-like dynamic-INT8-activation reference), and
`fake_quant_reference.py` (W8A8 per-tensor INT8 reference) each used to carry
their own copy of LayerNorm / GELU / softmax math. The two numpy references
now import from here so the math lives in one place.

The simulator reference passes ``fp16_storage=True`` to add the FP32→FP16→FP32
storage round-trip that mirrors hardware ABUF read/write semantics. The
fake-quant reference omits the round-trip (its activations are pure FP32
between INT8 QDQ boundaries). Every helper preserves the exact arithmetic
of the two original implementations so existing FP16-ULP tests keep passing.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.special import erf as _scipy_erf
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is an optional dep
    _HAS_SCIPY = False


__all__ = [
    "cast_fp16",
    "layernorm",
    "gelu_tanh",
    "gelu_erf",
    "softmax_causal",
    "softmax_masked",
]


def cast_fp16(x: np.ndarray) -> np.ndarray:
    """FP32 → FP16 → FP32 storage round-trip.

    Mirrors `mem.write_fp16_tile` followed by `mem.read_fp16_tile` in the
    simulator (FP16 in-memory, widened to FP32 on next read).
    """
    return x.astype(np.float16).astype(np.float32)


def layernorm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    *,
    fp16_storage: bool = False,
) -> np.ndarray:
    """Row-wise LayerNorm with FP32 internal reduction.

    ``fp16_storage=True`` casts input and output through FP16 (simulator
    semantics, matches `_exec_layernorm_fp32`). ``False`` keeps FP32
    throughout (fake-quant reference semantics matching sfu.py's Welford
    reduction).
    """
    if fp16_storage:
        x = cast_fp16(x)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        out = ((x - mean) / np.sqrt(var + eps)) * gamma.reshape(1, -1) + beta.reshape(1, -1)
        return cast_fp16(out)
    mean = x.mean(axis=-1, keepdims=True).astype(np.float32)
    var = x.var(axis=-1, keepdims=True).astype(np.float32)
    return (x - mean) / np.sqrt(var + np.float32(eps)) * gamma + beta


def gelu_tanh(x: np.ndarray, *, fp16_storage: bool = False) -> np.ndarray:
    """gelu_new (tanh approximation, used by GPT-2 / `gelu_new` / `gelu_fast`).

    ``fp16_storage=True`` round-trips input and output through FP16 (matches
    the hardware FP16 sub-layer contract). ``False`` keeps FP32 throughout.
    """
    if fp16_storage:
        x = cast_fp16(x)
        out = 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))
        return cast_fp16(out)
    xf = x.astype(np.float32)
    return xf * np.float32(0.5) * (
        np.float32(1.0) + np.tanh(
            np.float32(np.sqrt(2.0 / np.pi)) * (xf + np.float32(0.044715) * xf ** 3)
        )
    )


def gelu_erf(x: np.ndarray) -> np.ndarray:
    """GELU via erf — matches sfu.py when scipy is available; uses the
    Abramowitz & Stegun 7.1.26 polynomial fallback otherwise.

    Only the W8A8 fake-quant reference uses this variant (no fp16_storage
    round-trip needed; the simulator path is always tanh-approximation).
    """
    xf = x.astype(np.float32)
    if _HAS_SCIPY:
        return xf * np.float32(0.5) * (np.float32(1.0) + _scipy_erf(xf / np.sqrt(np.float32(2.0))))
    sgn = np.sign(xf)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * np.abs(xf))
    poly = t * (
        np.float32(0.254829592) + t * (
            np.float32(-0.284496736) + t * (
                np.float32(1.421413741) + t * (
                    np.float32(-1.453152027) + t * np.float32(1.061405429)
                )
            )
        )
    )
    erf_approx = sgn * (np.float32(1.0) - poly * np.exp(-(xf ** 2)))
    return xf * np.float32(0.5) * (np.float32(1.0) + erf_approx)


def softmax_causal(x: np.ndarray) -> np.ndarray:
    """Row-wise causal softmax for x[seq, seq] — matches execute_masked_softmax.

    Columns j > i (upper triangle) are masked to -inf before the max-subtract
    step. Used by the prefill / full-sequence fake-quant reference path.
    """
    seq = x.shape[0]
    out = np.empty_like(x, dtype=np.float32)
    for i in range(seq):
        row = x[i].astype(np.float32).copy()
        row[i + 1:] = -np.inf
        valid = row[: i + 1]
        row_max = float(valid.max())
        exp_row = np.exp(row - row_max)
        exp_row[i + 1:] = 0.0
        out[i] = exp_row / float(exp_row.sum())
    return out


def softmax_masked(
    scores: np.ndarray,
    valid_kv_len: int,
    *,
    fp16_storage: bool = False,
) -> np.ndarray:
    """Last-axis softmax with uniform "first valid_kv_len columns" mask.

    Used by the W8A16 simulator reference for single-token decode where
    every row attends to the same valid prefix. ``fp16_storage=True`` adds
    the simulator's FP16 storage round-trip on input and output.
    """
    if fp16_storage:
        scores = cast_fp16(scores)
    masked = scores.astype(np.float32).copy()
    if valid_kv_len < masked.shape[-1]:
        masked[..., valid_kv_len:] = -np.inf
    masked -= np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked)
    out = exp / np.sum(exp, axis=-1, keepdims=True)
    return cast_fp16(out) if fp16_storage else out
