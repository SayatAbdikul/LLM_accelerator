"""Lightweight Hessian-guided scoring helpers for PTQ experiments."""

from __future__ import annotations

import numpy as np


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

