"""PTQ4ViT-style twin-uniform quantization helpers.

These helpers implement software-only quantize-dequant emulation for the
post-softmax and post-GELU tensors discussed in PTQ4ViT. They do not change the
accelerator ISA or the stored tensor format; they are intended for search,
replay, and fake-quant experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _levels(bits: int) -> int:
    if bits < 2:
        raise ValueError("twin-uniform quantization requires at least 2 bits")
    return 1 << (bits - 1)


def _qmax(bits: int) -> int:
    return _levels(bits) - 1


def _qdq_unsigned(x: np.ndarray, delta: float, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    if delta <= 0:
        q = np.zeros_like(x, dtype=np.int32)
        return q.astype(np.float32), q
    q = np.clip(np.round(x / np.float32(delta)), 0, _qmax(bits)).astype(np.int32)
    return q.astype(np.float32) * np.float32(delta), q


def quantize_dequant_softmax_twin(
    tensor: np.ndarray,
    range1_max: float,
    *,
    bits: int = 8,
    return_metadata: bool = False,
):
    """Apply PTQ4ViT-style twin-uniform quantize-dequant to a softmax tensor.

    The low range uses a searched split ``range1_max``; the high range keeps the
    paper-fixed resolution of ``1 / 2^(bits-1)``.
    """

    levels = _levels(bits)
    x = np.clip(np.asarray(tensor, dtype=np.float32), 0.0, 1.0)
    split = float(np.clip(range1_max, 1e-8, 1.0))
    delta_lo = np.float32(split / levels)
    delta_hi = np.float32(1.0 / levels)

    mask_lo = x < np.float32(split)
    qdq_lo, q_lo = _qdq_unsigned(x, float(delta_lo), bits)
    qdq_hi, q_hi = _qdq_unsigned(x, float(delta_hi), bits)
    qdq = np.where(mask_lo, qdq_lo, qdq_hi).astype(np.float32)

    if not return_metadata:
        return qdq

    q = np.where(mask_lo, q_lo, q_hi)
    sat = np.where(mask_lo, q_lo == _qmax(bits), q_hi == _qmax(bits))
    meta: Dict[str, float] = {
        "mode": "softmax",
        "split": split,
        "delta_lo": float(delta_lo),
        "delta_hi": float(delta_hi),
        "low_fraction": float(np.mean(mask_lo)) if mask_lo.size else 0.0,
        "high_fraction": float(np.mean(~mask_lo)) if mask_lo.size else 0.0,
        "saturation_rate": float(np.mean(sat)) if sat.size else 0.0,
        "zero_fraction": float(np.mean(q == 0)) if q.size else 1.0,
    }
    return qdq, meta


def quantize_dequant_gelu_twin(
    tensor: np.ndarray,
    positive_range_max: float,
    *,
    negative_extent: float | None = None,
    bits: int = 8,
    return_metadata: bool = False,
):
    """Apply PTQ4ViT-style twin-uniform quantize-dequant to a GELU tensor.

    The negative range is fixed to cover the observed negative extent, while the
    positive range uses the searched split ``positive_range_max``.
    """

    levels = _levels(bits)
    x = np.asarray(tensor, dtype=np.float32)
    neg_extent = (
        float(negative_extent)
        if negative_extent is not None
        else float(max(-np.min(x), 0.0))
    )
    neg_extent = max(neg_extent, 1e-8)
    pos_extent = max(float(positive_range_max), 1e-8)
    delta_neg = np.float32(neg_extent / levels)
    delta_pos = np.float32(pos_extent / levels)

    neg_mag = np.maximum(-x, 0.0)
    pos_mag = np.maximum(x, 0.0)
    qdq_neg_mag, q_neg = _qdq_unsigned(neg_mag, float(delta_neg), bits)
    qdq_pos_mag, q_pos = _qdq_unsigned(pos_mag, float(delta_pos), bits)
    qdq = np.where(x < 0.0, -qdq_neg_mag, qdq_pos_mag).astype(np.float32)

    if not return_metadata:
        return qdq

    q = np.where(x < 0.0, q_neg, q_pos)
    sat = np.where(x < 0.0, q_neg == _qmax(bits), q_pos == _qmax(bits))
    meta: Dict[str, float] = {
        "mode": "gelu",
        "negative_extent": neg_extent,
        "positive_range_max": pos_extent,
        "delta_neg": float(delta_neg),
        "delta_pos": float(delta_pos),
        "negative_fraction": float(np.mean(x < 0.0)) if x.size else 0.0,
        "positive_fraction": float(np.mean(x >= 0.0)) if x.size else 0.0,
        "saturation_rate": float(np.mean(sat)) if sat.size else 0.0,
        "zero_fraction": float(np.mean(q == 0)) if q.size else 1.0,
    }
    return qdq, meta

