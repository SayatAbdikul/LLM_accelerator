"""Lightweight Hessian-guided scoring helpers for PTQ experiments."""

from __future__ import annotations

import numpy as np


def find_hessian_gelu_scale(
    gelu_all: np.ndarray,
    fc2_weight: np.ndarray,
    *,
    n_steps: int = 200,
) -> float:
    """Return the INT8 scale for GELU that minimises H-weighted quantisation error.

    Searches clipping thresholds from the 50th to 100th percentile of
    abs(gelu_all) and picks the scale s (FP16-rounded, matching hardware)
    that minimises mean(H * (qdq(gelu, s) - gelu)^2) where H is the
    diagonal Hessian proxy for the GELU -> FC2 output path.
    """
    g = np.asarray(gelu_all, dtype=np.float32)
    H = gelu_fc2_hessian_diag(g, fc2_weight)
    abs_flat = np.abs(g).ravel()
    lo = float(np.percentile(abs_flat, 50.0))
    hi = float(abs_flat.max())
    if hi <= 0.0:
        return 1.0 / 127.0

    best_scale = float(np.float16(hi / 127.0))
    best_error = float("inf")

    for clip_val in np.linspace(lo, hi, n_steps):
        if clip_val <= 0.0:
            continue
        s = float(np.float16(np.float32(clip_val) / np.float32(127.0)))
        if s <= 0.0:
            continue
        q = np.clip(np.round(g / s), -128.0, 127.0).astype(np.float32) * s
        err = weighted_quant_error_score(g, q, H)
        if err < best_error:
            best_error = err
            best_scale = s

    return best_scale


def weighted_quant_error_score(
    reference: np.ndarray,
    candidate: np.ndarray,
    hessian_diag: np.ndarray,
) -> float:
    """Return mean(H * (candidate - reference)^2) with broadcast support."""

    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    diag = np.asarray(hessian_diag, dtype=np.float32)
    diff = cand - ref
    return float(np.mean(diag * diff * diff))


def softmax_attn_v_hessian_diag(softmax: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Diagonal Hessian proxy for softmax -> attn@V under local squared loss."""

    soft = np.asarray(softmax, dtype=np.float32)
    val = np.asarray(value, dtype=np.float32)
    col_norm_sq = np.sum(val * val, axis=-1, dtype=np.float32)
    return np.broadcast_to(col_norm_sq.reshape(1, -1), soft.shape).astype(np.float32)


def gelu_fc2_hessian_diag(gelu: np.ndarray, fc2_weight: np.ndarray) -> np.ndarray:
    """Diagonal Hessian proxy for GELU -> FC2 under local squared loss."""

    gelu_arr = np.asarray(gelu, dtype=np.float32)
    weight = np.asarray(fc2_weight, dtype=np.float32)
    col_norm_sq = np.sum(weight * weight, axis=0, dtype=np.float32)
    return np.broadcast_to(col_norm_sq.reshape(1, -1), gelu_arr.shape).astype(np.float32)

