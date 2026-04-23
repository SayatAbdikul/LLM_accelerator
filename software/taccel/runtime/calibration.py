"""Calibration helpers for Stage 3 tiny nanoGPT verification.

Runs FP32 forward passes on Shakespeare calibration sequences extracted from
the fixture payload and returns per-node max-abs scales that both the
compiler (via calibration_scales) and the NanoGPTFQReference consume.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .fake_quant_reference import _fp32_forward


_DEFAULT_SCALES = 6.0 / 127.0
_SFU_DEFAULT_SCALES = 1.0 / 127.0


def _tokenize_text(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def build_calibration_seqs(
    payload: dict,
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
) -> List[List[int]]:
    """Extract short token sequences from the fixture's embedded Shakespeare text."""
    text = str(payload.get("text", ""))
    stoi = payload.get("stoi", {})
    if not text or not stoi:
        return [[0] * seq_len]
    tokens = _tokenize_text(text, stoi)
    if len(tokens) < seq_len:
        tokens = tokens * ((seq_len // max(len(tokens), 1)) + 2)
    seqs = []
    step = max(1, (len(tokens) - seq_len) // max(n_seqs, 1))
    for i in range(n_seqs):
        start = (i * step) % (len(tokens) - seq_len + 1)
        seqs.append(tokens[start: start + seq_len])
    return seqs


def build_calibration_seqs_from_token_ids(
    token_ids: Sequence[int],
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
) -> List[List[int]]:
    """Extract deterministic calibration windows from already-tokenized text."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    tokens = [int(tok) for tok in token_ids]
    if not tokens:
        return [[0] * seq_len]
    if len(tokens) < seq_len:
        tokens = tokens * ((seq_len // max(len(tokens), 1)) + 2)
    seqs: List[List[int]] = []
    step = max(1, (len(tokens) - seq_len) // max(n_seqs, 1))
    for i in range(n_seqs):
        start = (i * step) % (len(tokens) - seq_len + 1)
        seqs.append(tokens[start: start + seq_len])
    return seqs


def build_calibration_scales(
    payload: dict,
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
    percentile: float = 99.9,
) -> Dict[str, float]:
    """Return per-node INT8 scales derived from FP32 calibration runs.

    Runs ``_fp32_forward`` on ``n_seqs`` calibration sequences and sets
    each node's scale to ``percentile``-th percentile of |activation| / 127.
    Falls back to a small default for nodes that produce no data.
    """
    model_args = payload["model_args"]
    state_dict = payload["state_dict"]
    seqs = build_calibration_seqs(payload, n_seqs=n_seqs, seq_len=seq_len)

    # per-node max-abs accumulator
    accum: Dict[str, List[float]] = {}

    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name, arr in node_outputs.items():
            arr_f = np.asarray(arr, dtype=np.float32).ravel()
            if arr_f.size == 0:
                continue
            p = float(
                np.percentile(np.abs(arr_f), percentile)
                if percentile < 100.0
                else float(np.abs(arr_f).max())
            )
            accum.setdefault(name, []).append(p)

    scales: Dict[str, float] = {}
    for name, vals in accum.items():
        max_abs = float(np.max(vals))
        scales[name] = max(max_abs, 1e-8) / 127.0

    # Nodes that calibration didn't observe → keep the compiler's defaults
    _fill_defaults(scales, model_args)
    return scales


def build_calibration_scales_from_token_ids(
    payload: dict,
    token_ids: Sequence[int],
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
    percentile: float = 99.9,
) -> Dict[str, float]:
    """Return per-node INT8 scales from tokenized calibration text."""
    model_args = payload["model_args"]
    state_dict = payload["state_dict"]
    seqs = build_calibration_seqs_from_token_ids(
        token_ids,
        n_seqs=n_seqs,
        seq_len=seq_len,
    )

    accum: Dict[str, List[float]] = {}
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name, arr in node_outputs.items():
            arr_f = np.asarray(arr, dtype=np.float32).ravel()
            if arr_f.size == 0:
                continue
            p = float(
                np.percentile(np.abs(arr_f), percentile)
                if percentile < 100.0
                else float(np.abs(arr_f).max())
            )
            accum.setdefault(name, []).append(p)

    scales: Dict[str, float] = {}
    for name, vals in accum.items():
        max_abs = float(np.max(vals))
        scales[name] = max(max_abs, 1e-8) / 127.0

    _fill_defaults(scales, model_args)
    return scales


def _fill_defaults(scales: Dict[str, float], model_args: dict) -> None:
    """Ensure every compiler calibration_scales key has an entry."""
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])

    def _d(name: str, default: float = _DEFAULT_SCALES) -> None:
        scales.setdefault(name, default)

    _d("tok_embed")
    _d("pos_embed")
    _d("tok_pos_add")
    for L in range(n_layer):
        _d(f"block{L}_ln1")
        for H in range(n_head):
            _d(f"block{L}_head{H}_query")
            _d(f"block{L}_head{H}_key")
            _d(f"block{L}_head{H}_value")
            _d(f"block{L}_head{H}_softmax", _SFU_DEFAULT_SCALES)
            _d(f"block{L}_head{H}_attn_v")
            # kv_load nodes in the decode graph substitute for key/value matmul
            # outputs.  They carry the same INT8 values so must use the same scale.
            scales.setdefault(
                f"block{L}_head{H}_key_kv_load",
                scales.get(f"block{L}_head{H}_key", _DEFAULT_SCALES),
            )
            scales.setdefault(
                f"block{L}_head{H}_value_kv_load",
                scales.get(f"block{L}_head{H}_value", _DEFAULT_SCALES),
            )
        _d(f"block{L}_concat")
        _d(f"block{L}_out_proj")
        _d(f"block{L}_residual1")
        _d(f"block{L}_ln2")
        _d(f"block{L}_fc1")
        _d(f"block{L}_gelu", _SFU_DEFAULT_SCALES)
        _d(f"block{L}_fc2")
        _d(f"block{L}_residual2")
    _d("ln_f")
    _d("lm_head")
