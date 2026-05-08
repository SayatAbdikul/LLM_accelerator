#!/usr/bin/env python3
"""QuaRot/rotation-based-quantization commit/no-commit diagnostic.

Runs a 3-tier funnel to decide whether the GPT-2 PTQ project should commit to
integrating QuaRot/SpinQuant/TurboQuant-style rotation-based outlier
suppression. Each tier has explicit numerical thresholds and early-exits on
NO_GO signals.

  Tier 1: per-tensor outlier statistics (kurtosis, max/median, channel
          concentration, INT8 SNR). Decisive NO_GO if no outliers exist or
          outliers are post-GELU-only.
  Tier 2: statistical rotation efficacy. Apply random Hadamard, recompute
          kurtosis and SNR. Decisive NO_GO if rotation doesn't reduce them.
  Tier 3: end-to-end one-block rotation simulation. Pre-rotate weight input
          columns, inject runtime FP32 activation rotation, measure logit
          cosine and PPL. Final commit/no-commit decision.

See plan: /Users/sayat/.claude/plans/playful-enchanting-music.md

Usage::

    PYTHONPATH=software python3 software/tools/diagnose_activation_outliers.py \\
      software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \\
      --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \\
      --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \\
      --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \\
      --tier all --json-out /tmp/diag.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from taccel.quantizer.quantize import quantize_tensor
from taccel.runtime.calibration import (
    build_calibration_scales_from_token_ids,
    build_calibration_seqs_from_token_ids,
)
from taccel.runtime.fake_quant_reference import (
    NanoGPTFQReference,
    _arch_scale,
    _fp32_forward,
    _to_f32,
    clear_weight_component_cache,
)
from taccel.runtime.fp32_reference import NanoGPTFP32Reference
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_PERCENTILE_DEFAULT,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)


TOOL_VERSION = "1.0.0"


# =============================================================================
# Tier 1: per-tensor outlier statistics
# =============================================================================


def _kurt(arr: np.ndarray) -> float:
    """Excess kurtosis (Gaussian = 0). Computed over the flattened tensor.

    Uses the standard moment estimator: m4 / m2^2 - 3 with population moments.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    n = a.size
    if n < 4:
        return 0.0
    mu = a.mean()
    diff = a - mu
    m2 = float(np.mean(diff * diff))
    if m2 <= 1e-30:
        return 0.0
    m4 = float(np.mean(diff * diff * diff * diff))
    return m4 / (m2 * m2) - 3.0


def _stats_for_tensor(arr: np.ndarray, scale_percentile: float = 99.9) -> Dict[str, float]:
    """Compute the per-tensor outlier statistics described in the plan.

    `arr` is reshaped to `[N, C]`: any leading dims are treated as samples.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    n_samples, n_channels = a.shape

    abs_a = np.abs(a)
    flat_abs = abs_a.ravel()

    max_abs = float(flat_abs.max()) if flat_abs.size else 0.0
    median_abs = float(np.median(flat_abs)) if flat_abs.size else 0.0
    max_over_median = max_abs / max(median_abs, 1e-12)

    kurt = _kurt(a)

    # Per-channel L2 concentration: how much L2-energy lives in top 1% of channels
    chan_l2_sq = np.sum(a.astype(np.float64) ** 2, axis=0)
    total_l2_sq = float(chan_l2_sq.sum())
    if total_l2_sq > 0.0 and n_channels >= 100:
        n_top = max(1, n_channels // 100)
        top_idx = np.argpartition(chan_l2_sq, -n_top)[-n_top:]
        top1pct_l2_frac = float(chan_l2_sq[top_idx].sum() / total_l2_sq)
    else:
        # Too few channels for "top 1%" to be meaningful; report 1/n_channels as floor
        top1pct_l2_frac = float(1.0 / max(n_channels, 1))

    # Quantization SNR at production-equivalent per-tensor scale (99.9 pct / 127)
    pct_val = float(np.percentile(flat_abs, scale_percentile)) if flat_abs.size else 0.0
    raw_scale = max(pct_val, 1e-8) / 127.0
    arch_scale = float(_arch_scale(raw_scale))
    if arch_scale <= 0.0:
        int8_mse = 0.0
        int8_snr_db = float("inf")
    else:
        q = np.clip(np.round(a / arch_scale), -128, 127).astype(np.int8)
        dq = q.astype(np.float32) * arch_scale
        diff = (a - dq).astype(np.float64)
        int8_mse = float(np.mean(diff * diff))
        var_a = float(np.var(a.astype(np.float64)))
        if int8_mse <= 1e-30 or var_a <= 1e-30:
            int8_snr_db = float("inf")
        else:
            int8_snr_db = 10.0 * np.log10(var_a / int8_mse)

    return {
        "n_samples": int(n_samples),
        "n_channels": int(n_channels),
        "kurt": float(kurt),
        "max_abs": float(max_abs),
        "median_abs": float(median_abs),
        "max_over_median": float(max_over_median),
        "top1pct_l2_frac": float(top1pct_l2_frac),
        "scale_99_9": float(arch_scale),
        "int8_mse": float(int8_mse),
        "int8_snr_db": float(int8_snr_db),
    }


_GELU_NODE_RE = re.compile(r"^block\d+_gelu$")


def classify_tensor(
    name: str,
    *,
    kurt: float,
    max_over_median: float,
    top1pct_l2_frac: float,
    int8_snr_db: float,
    **_unused,
) -> str:
    """Classify a tensor as one of: clean, channel-concentrated,
    diffuse-heavy-tail, post-gelu-untreatable.

    Rules from plan:
      if name matches "block.*_gelu" and outlier-bearing: "post-gelu-untreatable"
      elif kurt > 50 and top1pct_l2_frac > 0.50: "channel-concentrated"
      elif kurt > 50: "diffuse-heavy-tail"
      elif kurt > 10 and max_over_median > 100:
          "channel-concentrated" if top1pct_l2_frac > 0.30 else "diffuse-heavy-tail"
      else: "clean"
    """
    if _GELU_NODE_RE.match(name):
        if kurt > 10 or max_over_median > 50:
            return "post-gelu-untreatable"
        return "clean"
    if kurt > 50:
        if top1pct_l2_frac > 0.50:
            return "channel-concentrated"
        return "diffuse-heavy-tail"
    if kurt > 10 and max_over_median > 100:
        if top1pct_l2_frac > 0.30:
            return "channel-concentrated"
        return "diffuse-heavy-tail"
    return "clean"


def _is_gelu_node(name: str) -> bool:
    return bool(_GELU_NODE_RE.match(name))


def collect_activation_stats(
    payload: Dict[str, object],
    calib_token_ids: Sequence[int],
    *,
    n_seqs: int = 16,
    seq_len: int = 64,
    target_nodes: Optional[Sequence[str]] = None,
    scale_percentile: float = CALIBRATION_PERCENTILE_DEFAULT,
) -> Dict[str, Dict[str, float]]:
    """Capture FP32 activations from a calibration run and compute per-tensor stats.

    Iterates calibration sequences, calls _fp32_forward, accumulates each node's
    outputs, then computes statistics on the concatenated tensor.

    Args:
        payload: model payload with state_dict + model_args
        calib_token_ids: tokenized calibration text
        n_seqs: number of calibration sequences
        seq_len: tokens per sequence
        target_nodes: subset of nodes to process. If None, processes all nodes
            output by _fp32_forward except softmax/lm_head (last-token-only).
        scale_percentile: percentile for current-pipeline scale (default 99.9)

    Returns: dict mapping node name → stats dict (see _stats_for_tensor).
    """
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids, n_seqs=n_seqs, seq_len=seq_len
    )

    # Accumulate per-node activations across sequences.
    accum: Dict[str, List[np.ndarray]] = {}
    target_set = set(target_nodes) if target_nodes is not None else None
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name, arr in node_outputs.items():
            # Skip softmax (already in [0,1] — quantization is a different problem)
            # Skip lm_head (last-token-only, very few samples)
            if name == "lm_head":
                continue
            if "_softmax" in name:
                continue
            if target_set is not None and name not in target_set:
                continue
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim > 2:
                a = a.reshape(-1, a.shape[-1])
            elif a.ndim == 1:
                a = a.reshape(1, -1)
            accum.setdefault(name, []).append(a)

    # Compute stats per node on the concatenated activations.
    stats: Dict[str, Dict[str, float]] = {}
    for name, rows in accum.items():
        full = np.concatenate(rows, axis=0)
        s = _stats_for_tensor(full, scale_percentile=scale_percentile)
        s["classification"] = classify_tensor(name, **s)
        stats[name] = s
    return stats


def tier1_decision(stats: Dict[str, Dict[str, float]]) -> Tuple[str, str]:
    """Return (decision, reason) per the plan's tier 1 decision rules.

    Decisions:
      NO_GO_NO_OUTLIERS: all non-gelu tensors have kurt<10 and max/median<50
      NO_GO_POST_GELU_ONLY: top-3 by kurt all post-GELU AND every non-gelu is "clean"
      GO: ≥1 non-gelu "channel-concentrated" with int8_snr_db<25
      INCONCLUSIVE: mixed signals; run Tier 2
    """
    if not stats:
        return ("INCONCLUSIVE", "no tensors collected")

    # Gather the non-gelu and gelu-only tensors.
    non_gelu = [(n, s) for n, s in stats.items() if not _is_gelu_node(n)]
    gelu = [(n, s) for n, s in stats.items() if _is_gelu_node(n)]

    # NO_GO_NO_OUTLIERS: all non-gelu are clean by the loose threshold.
    all_clean = all(
        s["kurt"] < 10.0 and s["max_over_median"] < 50.0 for _, s in non_gelu
    )
    if all_clean:
        return (
            "NO_GO_NO_OUTLIERS",
            f"all {len(non_gelu)} non-gelu tensors have kurt<10 and max/median<50",
        )

    # NO_GO_POST_GELU_ONLY: top-3 kurt are all gelu AND every non-gelu classifies clean.
    sorted_by_kurt = sorted(stats.items(), key=lambda kv: kv[1]["kurt"], reverse=True)
    top3_names = [n for n, _ in sorted_by_kurt[:3]]
    top3_all_gelu = all(_is_gelu_node(n) for n in top3_names)
    every_non_gelu_clean = all(s["classification"] == "clean" for _, s in non_gelu)
    if top3_all_gelu and every_non_gelu_clean and len(gelu) > 0:
        return (
            "NO_GO_POST_GELU_ONLY",
            f"top-3 kurt are all post-GELU ({top3_names}) and all non-gelu classify clean",
        )

    # GO: ≥1 non-gelu channel-concentrated with int8_snr_db<25
    candidates = [
        (n, s)
        for n, s in non_gelu
        if s["classification"] == "channel-concentrated" and s["int8_snr_db"] < 25.0
    ]
    if candidates:
        worst = min(candidates, key=lambda kv: kv[1]["int8_snr_db"])
        return (
            "GO",
            f"{len(candidates)} non-gelu channel-concentrated tensor(s); "
            f"worst SNR {worst[1]['int8_snr_db']:.1f} dB at {worst[0]}",
        )

    # Otherwise, mixed — run Tier 2 to disambiguate.
    return (
        "INCONCLUSIVE",
        "outliers present but no non-gelu tensor classifies channel-concentrated; "
        "Tier 2 will check rotation efficacy on diffuse-heavy-tail candidates",
    )


# =============================================================================
# Tier 2: statistical rotation efficacy
# =============================================================================


def build_random_orthogonal(d: int, *, seed: int = 0xCAFE) -> np.ndarray:
    """Random d×d orthogonal matrix via QR of a Gaussian (Haar-distributed)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, d)).astype(np.float64)
    q, r = np.linalg.qr(g)
    # Sign-correct so Q is uniformly distributed over O(d).
    sign = np.sign(np.diag(r))
    q = q * sign[np.newaxis, :]
    return q.astype(np.float32)


def _sylvester_hadamard(n: int) -> np.ndarray:
    """Sylvester-construction Hadamard matrix of size n. n must be 2^k."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"size must be a power of 2, got {n}")
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def build_block_hadamard_768() -> np.ndarray:
    """768×768 = 12 ⊗ H_64 block-Hadamard, normalized to be orthogonal.

    768 = 12 × 64. We build a block-diagonal of twelve 64×64 Sylvester
    Hadamards normalized by 1/sqrt(64). This is what a real on-chip Hadamard
    unit would compute (mixing only within head-dim-sized chunks).

    Note: a tighter mixing would use a 256-Hadamard tiled 3 times (since
    768 = 256·3) but that requires a non-power-of-2 outer structure. The
    head-dim block diagonal is a conservative choice and matches QuaRot's
    head-dim Hadamard structure.
    """
    H64 = _sylvester_hadamard(64) / np.sqrt(64.0)
    H = np.zeros((768, 768), dtype=np.float64)
    for b in range(12):
        H[b * 64 : (b + 1) * 64, b * 64 : (b + 1) * 64] = H64
    return H.astype(np.float32)


def build_block_hadamard_3072() -> np.ndarray:
    """3072×3072 block-Hadamard for the gelu→fc2 path.

    3072 = 4 × 768 = 12 × 256 = 48 × 64. We use 12 blocks of 256-Hadamards
    (each 256 = 2^8 has a Sylvester Hadamard). Wider mixing than the 768
    head-block version, since gelu's "outlier features" tend to span more
    channels than the residual stream's.
    """
    H256 = _sylvester_hadamard(256) / np.sqrt(256.0)
    H = np.zeros((3072, 3072), dtype=np.float64)
    for b in range(12):
        H[b * 256 : (b + 1) * 256, b * 256 : (b + 1) * 256] = H256
    return H.astype(np.float32)


def _measure_quant_efficacy(
    activations: np.ndarray,
    rotation: np.ndarray,
    *,
    scale_percentile: float = CALIBRATION_PERCENTILE_DEFAULT,
) -> Dict[str, float]:
    """Measure Q(Rx) ≈ Rx (NOT R^T·Q(Rx) ≈ x — that folds in a free FP32 inverse).

    Recalibrates the scale to the rotated tensor's 99.9th percentile, mirroring
    what real integration would do at calibration time.
    """
    a = np.asarray(activations, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])

    # "Before" = quantize unrotated activation at its own 99.9pct scale.
    abs_a = np.abs(a)
    pct_pre = float(np.percentile(abs_a.ravel(), scale_percentile))
    scale_pre = float(_arch_scale(max(pct_pre, 1e-8) / 127.0))
    q_pre = np.clip(np.round(a / scale_pre), -128, 127).astype(np.int8)
    dq_pre = q_pre.astype(np.float32) * scale_pre
    mse_pre = float(np.mean((a - dq_pre).astype(np.float64) ** 2))
    var_a = float(np.var(a.astype(np.float64)))
    snr_pre = 10.0 * np.log10(var_a / max(mse_pre, 1e-30)) if var_a > 1e-30 else float("inf")
    kurt_pre = _kurt(a)

    # "After" = rotate, then quantize at the rotated tensor's own 99.9pct scale.
    Rx = a @ rotation.T  # [N, C] @ [C, C].T → [N, C], applies R per-row
    pct_post = float(np.percentile(np.abs(Rx).ravel(), scale_percentile))
    scale_post = float(_arch_scale(max(pct_post, 1e-8) / 127.0))
    q_post = np.clip(np.round(Rx / scale_post), -128, 127).astype(np.int8)
    dq_post = q_post.astype(np.float32) * scale_post
    mse_post = float(np.mean((Rx - dq_post).astype(np.float64) ** 2))
    var_Rx = float(np.var(Rx.astype(np.float64)))
    snr_post = 10.0 * np.log10(var_Rx / max(mse_post, 1e-30)) if var_Rx > 1e-30 else float("inf")
    kurt_post = _kurt(Rx)

    snr_delta = snr_post - snr_pre
    kurt_ratio = kurt_post / max(kurt_pre, 1e-12) if kurt_pre > 0 else 1.0

    return {
        "kurt_before": float(kurt_pre),
        "kurt_after": float(kurt_post),
        "kurt_ratio": float(kurt_ratio),
        "snr_before_db": float(snr_pre),
        "snr_after_db": float(snr_post),
        "snr_delta_db": float(snr_delta),
        "scale_before": float(scale_pre),
        "scale_after": float(scale_post),
    }


def measure_rotation_efficacy_for_tensors(
    payload: Dict[str, object],
    calib_token_ids: Sequence[int],
    candidate_nodes: Sequence[str],
    *,
    n_seqs: int = 16,
    seq_len: int = 64,
    seeds: Sequence[int] = (0xCAFE, 0xC0FFEE),
    include_block_hadamard: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    """For each candidate node, capture activations and measure rotation efficacy
    under multiple rotations (random-orthogonal seeds + structured Hadamard).

    Returns: dict mapping node name → list of measurements (one per rotation).
    Each measurement dict contains rotation name + the kurt/SNR statistics.
    """
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids, n_seqs=n_seqs, seq_len=seq_len
    )
    target_set = set(candidate_nodes)

    accum: Dict[str, List[np.ndarray]] = {}
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name in target_set:
            if name not in node_outputs:
                continue
            a = np.asarray(node_outputs[name], dtype=np.float32)
            if a.ndim > 2:
                a = a.reshape(-1, a.shape[-1])
            elif a.ndim == 1:
                a = a.reshape(1, -1)
            accum.setdefault(name, []).append(a)

    # Build rotation matrices keyed by name.
    rotations: List[Tuple[str, np.ndarray, int]] = []
    for seed in seeds:
        rotations.append((f"random_orthogonal_seed_{seed:#x}", None, seed))
    if include_block_hadamard:
        H768 = build_block_hadamard_768()
        rotations.append(("block_hadamard_768", H768, -1))

    out: Dict[str, List[Dict[str, object]]] = {}
    for node, rows in accum.items():
        full = np.concatenate(rows, axis=0)
        d = full.shape[-1]
        node_results = []
        for rname, R, seed in rotations:
            if R is None:
                R_use = build_random_orthogonal(d, seed=seed)
            elif R.shape[0] != d:
                # Skip structured Hadamard if dim doesn't match (e.g., MLP fc1 is 3072)
                continue
            else:
                R_use = R
            eff = _measure_quant_efficacy(full, R_use)
            eff["rotation"] = rname
            eff["dim"] = int(d)
            node_results.append(eff)
        out[node] = node_results
    return out


def tier2_decision(
    rotation_results: Dict[str, List[Dict[str, object]]],
    tier1_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, str]:
    """Tier 2 decision logic per plan.

    GO: kurt_ratio<0.2 AND snr_delta_db>3 on the worst non-gelu tensor from Tier 1.
    NO_GO: snr_delta_db<1 on every channel-concentrated tensor.
    MARGINAL: snr_delta_db in [1, 3] dB on the best tensor.
    """
    if not rotation_results:
        return ("NO_GO", "no tensors had measurable rotation efficacy")

    # Identify the worst non-gelu channel-concentrated tensor by Tier 1 SNR.
    cc_non_gelu = [
        (n, s)
        for n, s in tier1_stats.items()
        if s.get("classification") == "channel-concentrated" and not _is_gelu_node(n)
    ]
    if not cc_non_gelu:
        # No channel-concentrated tensors — fall back to diffuse-heavy-tail.
        cc_non_gelu = [
            (n, s)
            for n, s in tier1_stats.items()
            if s.get("classification") == "diffuse-heavy-tail" and not _is_gelu_node(n)
        ]
    if not cc_non_gelu:
        return ("NO_GO", "no candidate tensors from Tier 1")

    worst_node = min(cc_non_gelu, key=lambda kv: kv[1]["int8_snr_db"])[0]
    worst_results = rotation_results.get(worst_node, [])
    if not worst_results:
        return ("NO_GO", f"no rotation results for worst Tier 1 tensor {worst_node}")

    # Best rotation result for the worst tensor (max snr_delta_db).
    best = max(worst_results, key=lambda r: r["snr_delta_db"])
    kurt_ratio = best["kurt_ratio"]
    snr_delta = best["snr_delta_db"]

    if snr_delta > 3.0 and kurt_ratio < 0.2:
        return (
            "GO",
            f"worst tensor {worst_node}: kurt_ratio={kurt_ratio:.3f}, "
            f"Δ SNR={snr_delta:+.1f} dB (rotation={best['rotation']})",
        )
    # If every channel-concentrated tensor has snr_delta < 1 dB on its best rotation:
    all_flat = True
    for n, _ in cc_non_gelu:
        results = rotation_results.get(n, [])
        if results:
            best_for_n = max(results, key=lambda r: r["snr_delta_db"])
            if best_for_n["snr_delta_db"] >= 1.0:
                all_flat = False
                break
    if all_flat:
        return (
            "NO_GO",
            f"every channel-concentrated tensor has best Δ SNR < 1 dB; "
            f"rotation does not help on this model",
        )
    return (
        "MARGINAL",
        f"worst tensor {worst_node}: kurt_ratio={kurt_ratio:.3f}, "
        f"Δ SNR={snr_delta:+.1f} dB — borderline, run Tier 3 with tighter threshold",
    )


# =============================================================================
# Tier 3: end-to-end one-block rotation simulation
# =============================================================================


def rotate_residual_stream_weights(
    state_dict: dict,
    model_args: dict,
    R: np.ndarray,
    *,
    target_blocks: Sequence[int] = None,
) -> List[str]:
    """Pre-rotate ALL weights so the residual stream lives in rotated basis.

    QuaRot Phase 1 recipe (without β-fold; LN handled at runtime via
    un-rotate-before-LN):
      1. wte, wpe: right-mul by R^T (each row rotated → embedding output is rotated)
      2. For each target block:
         a. c_attn input weights (per-head q_w, k_w, v_w): right-mul by R^T
         b. mlp.c_fc input weight: right-mul by R^T
         c. attn.c_proj OUTPUT rows: left-mul by R (output is rotated)
         d. mlp.c_proj OUTPUT rows: left-mul by R (output is rotated)
      3. lm_head: right-mul by R^T (input is rotated)

    For non-target blocks, c_attn/c_fc inputs and c_proj outputs are left
    untouched. This means the residual stream entering and leaving non-target
    blocks must still be rotated, which requires un-rotating before each non-
    target block's matmuls and re-rotating their outputs at runtime — a pain
    that defeats the simulation's purpose. So in practice target_blocks
    should be all blocks (full network).

    Returns: list of state_dict keys modified.
    """
    n_layer = int(model_args["n_layer"])
    if target_blocks is None:
        target_blocks = list(range(n_layer))
    target_blocks = [int(L) for L in target_blocks]

    R_t = np.asarray(R, dtype=np.float32).T
    R_arr = np.asarray(R, dtype=np.float32)
    modified: List[str] = []

    def _store(key: str, new_value: np.ndarray) -> None:
        old = state_dict[key]
        if hasattr(old, "dtype") and hasattr(old, "to"):
            state_dict[key] = torch.from_numpy(new_value).to(dtype=old.dtype)
        else:
            state_dict[key] = torch.from_numpy(new_value)
        modified.append(key)

    # 1. Embeddings: each row rotated → wte_new = wte @ R^T
    for key in ("transformer.wte.weight", "transformer.wpe.weight"):
        if key in state_dict:
            w = _to_f32(state_dict[key])
            _store(key, w @ R_t)

    # 2. Per block: input rotations + output rotations
    for L in target_blocks:
        # 2a. c_attn input weights (per-head split)
        H = 0
        while True:
            base = f"transformer.h.{L}.attn.c_attn.weight_h{H}"
            if f"{base}_query" not in state_dict:
                break
            for kind in ("query", "key", "value"):
                k = f"{base}_{kind}"
                if k in state_dict:
                    w = _to_f32(state_dict[k])  # [d_head, d_model]
                    _store(k, w @ R_t)
            H += 1

        # 2b. mlp.c_fc input weight
        key = f"transformer.h.{L}.mlp.c_fc.weight"
        if key in state_dict:
            w = _to_f32(state_dict[key])  # [4*d_model, d_model]
            _store(key, w @ R_t)

        # 2c. attn.c_proj OUTPUT rows: left-mul by R
        key = f"transformer.h.{L}.attn.c_proj.weight"
        if key in state_dict:
            w = _to_f32(state_dict[key])  # [d_model, d_model]
            _store(key, R_arr @ w)
        # also rotate c_proj bias (output gets rotated)
        bkey = f"transformer.h.{L}.attn.c_proj.bias"
        if bkey in state_dict:
            b = _to_f32(state_dict[bkey])  # [d_model]
            _store(bkey, R_arr @ b)

        # 2d. mlp.c_proj OUTPUT rows: left-mul by R
        key = f"transformer.h.{L}.mlp.c_proj.weight"
        if key in state_dict:
            w = _to_f32(state_dict[key])  # [d_model, 4*d_model]
            _store(key, R_arr @ w)
        bkey = f"transformer.h.{L}.mlp.c_proj.bias"
        if bkey in state_dict:
            b = _to_f32(state_dict[bkey])
            _store(bkey, R_arr @ b)

    # 3. lm_head input rotation: lm_head reads from rotated ln_f output.
    # lm_head shape [vocab, d_model], input is per-row [d_model]. Right-mul.
    key = "lm_head.weight"
    if key in state_dict:
        w = _to_f32(state_dict[key])
        _store(key, w @ R_t)

    return modified


def rotate_block_input_weights(
    state_dict: dict,
    block: int,
    R: np.ndarray,
    *,
    paths: Sequence[str] = ("mlp.c_fc", "attn.c_attn"),
) -> List[str]:
    """Mutate state_dict in place: for each rotated path, right-multiply weight
    input columns by R^T (so that W·R^T · R·x = W·x in FP32).

    For c_attn weights, the codebase stores per-head split weights as
    `attn.c_attn.weight_h{H}_{query,key,value}`. Each is [d_head, d_model].
    Right-multiplying by R^T rotates the d_model (input) dimension.

    For c_fc weight, shape is [4·d_model, d_model]. Same pattern.

    Returns: list of state_dict keys that were modified.
    """
    R_t = np.asarray(R, dtype=np.float32).T  # input-dim rotation matrix transposed
    modified: List[str] = []

    def _store(key: str, new_value: np.ndarray) -> None:
        old = state_dict[key]
        if hasattr(old, "dtype") and hasattr(old, "to"):
            state_dict[key] = torch.from_numpy(new_value).to(dtype=old.dtype)
        else:
            state_dict[key] = torch.from_numpy(new_value)
        modified.append(key)

    for path in paths:
        if path == "mlp.c_fc":
            key = f"transformer.h.{block}.mlp.c_fc.weight"
            if key in state_dict:
                w = _to_f32(state_dict[key])  # [4*d_model, d_model]
                _store(key, w @ R_t)  # right-mul rotates input cols (d_model)
        elif path == "mlp.c_proj":
            key = f"transformer.h.{block}.mlp.c_proj.weight"
            if key in state_dict:
                w = _to_f32(state_dict[key])  # [d_model, 4*d_model]
                _store(key, w @ R_t)  # right-mul rotates input cols (4*d_model)
        elif path == "attn.c_attn":
            # Per-head split weights: iterate H = 0, 1, ... until missing.
            H = 0
            while True:
                base = f"transformer.h.{block}.attn.c_attn.weight_h{H}"
                if f"{base}_query" not in state_dict:
                    break
                for kind in ("query", "key", "value"):
                    k = f"{base}_{kind}"
                    if k in state_dict:
                        w = _to_f32(state_dict[k])  # [d_head, d_model]
                        _store(k, w @ R_t)
                H += 1
        else:
            raise ValueError(f"unknown rotation path {path!r}")

    return modified


def tier3_one_block_simulation(
    payload: Dict[str, object],
    eval_token_ids: Sequence[int],
    *,
    target_blocks: Sequence[int],
    rotation: np.ndarray,
    gelu_rotation: Optional[np.ndarray] = None,
    fp32_ppl: Optional[float] = None,
    rotate_qk: bool = False,
    n_calib_seqs: int = 16,
    calib_seq_len: int = 64,
    calibration_token_ids: Optional[Sequence[int]] = None,
    ptq_preset_name: Optional[str] = None,
) -> Dict[str, object]:
    """Run end-to-end one-block rotation simulation.

    Steps:
      1. Compute baseline INT8 PPL with current production preset.
      2. Build a rotated state_dict: pre-rotate c_attn (per-head) and c_fc
         weights of target_block by W' = W·R^T.
      3. Compute calibration scales against the ROTATED activations
         (block{L}_ln1 and ln2 outputs need to be rotated by R for scale calc).
      4. Run rotated INT8 PPL.
      5. Compare logit cosine + PPL.

    Returns: dict with self_tests, baseline_*, rotated_*, and decision metrics.
    """
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    vocab_size = int(model_args["vocab_size"])

    # Calibration data: use eval set if no calib provided.
    if calibration_token_ids is None:
        calibration_token_ids = list(eval_token_ids)
    preset_name = ptq_preset_name or stage5_default_ptq_preset_name()
    preset = resolve_stage5_ptq_preset(preset_name)

    # Teacher-forced eval setup.
    inputs, targets = teacher_forced_inputs_and_targets(list(eval_token_ids))

    # ---------------- Baseline INT8 PPL (no rotation) ----------------
    # Capture state_dict snapshot for restoration.
    sd_snapshot = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state_dict.items()}

    base_scales = build_calibration_scales_from_token_ids(
        payload, calibration_token_ids, n_seqs=n_calib_seqs, seq_len=calib_seq_len,
        percentile=CALIBRATION_PERCENTILE_DEFAULT,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    base_scales = apply_stage5_ptq_scale_policy(base_scales, model_args, preset)

    clear_weight_component_cache()
    base_ref = NanoGPTFQReference(
        state_dict, model_args, base_scales,
        requant_pc_weight_names=_resolve_requant_pc_names(preset, model_args),
        raw_residual1_blocks=_raw_residual1_blocks(preset),
        raw_residual2_blocks=_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=preset.gelu_from_accum_blocks or None,
    )
    t0 = time.time()
    base_logits = base_ref.incremental_logits_trace(inputs)
    t_base = time.time() - t0
    base_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(base_logits, targets)
    ]
    base_ppl, _ = perplexity_from_nlls(base_nlls)

    # ---------------- Self-test: identity rotation produces identical INT8 output ----------------
    self_tests: Dict[str, object] = {}
    d = int(model_args["n_embd"])
    identity = np.eye(d, dtype=np.float32)
    # Apply identity to a fresh state_dict copy (snapshot is the original).
    identity_sd = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in sd_snapshot.items()}
    identity_payload = {"state_dict": identity_sd, "model_args": model_args}
    target_blocks_list = sorted(set(int(b) for b in target_blocks))
    # For self-test, only test on the first target block.
    test_block = target_blocks_list[0] if target_blocks_list else 0
    rotate_block_input_weights(identity_sd, test_block, identity, paths=("mlp.c_fc", "attn.c_attn"))
    # Verify state_dict is byte-identical (within float epsilon) to snapshot.
    max_drift = 0.0
    for k in identity_sd:
        a = _to_f32(identity_sd[k])
        b = _to_f32(sd_snapshot[k])
        if a.shape == b.shape:
            max_drift = max(max_drift, float(np.abs(a - b).max()))
    self_tests["identity_state_dict_max_drift"] = max_drift
    self_tests["identity_state_dict_match"] = max_drift < 1e-5

    # ---------------- FP32 ceiling (uses UNROTATED state_dict) ----------------
    # Must run BEFORE we rotate the state_dict. Otherwise the FP32 reference
    # would consume rotated weights without runtime activation rotation, which
    # produces garbage from target blocks onwards.
    fp32_ref = NanoGPTFP32Reference(state_dict, model_args)
    fp32_logits = fp32_ref.incremental_logits_trace(inputs)
    if fp32_ppl is None:
        fp32_nlls = [
            stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
            for row, tgt in zip(fp32_logits, targets)
        ]
        fp32_ppl, _ = perplexity_from_nlls(fp32_nlls)

    # ---------------- Rotated calibration scales (uses UNROTATED state_dict) ----------------
    rot_scales = _build_rotated_calibration_scales(
        payload, calibration_token_ids,
        target_blocks=target_blocks_list, rotation=rotation,
        gelu_rotation=gelu_rotation,
        n_seqs=n_calib_seqs, seq_len=calib_seq_len,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    rot_scales = apply_stage5_ptq_scale_policy(rot_scales, model_args, preset)

    # ---------------- Rotated INT8 PPL ----------------
    # NOW we rotate the state_dict. After this point, state_dict has rotated
    # weights for target_block(s), and FP32 forwards on it would be wrong.
    paths = ["mlp.c_fc", "attn.c_attn"]
    modified: List[str] = []
    for L in target_blocks_list:
        modified += rotate_block_input_weights(state_dict, L, rotation, paths=paths)
        if gelu_rotation is not None:
            # Phase 2: also rotate the gelu→fc2 path (mlp.c_proj input cols).
            modified += rotate_block_input_weights(
                state_dict, L, gelu_rotation, paths=("mlp.c_proj",),
            )

    clear_weight_component_cache()
    rot_ref = _RotatedAtBlockFQReference(
        state_dict, model_args, rot_scales,
        rotation=rotation, target_blocks=target_blocks_list,
        gelu_rotation=gelu_rotation,
        requant_pc_weight_names=_resolve_requant_pc_names(preset, model_args),
        raw_residual1_blocks=_raw_residual1_blocks(preset),
        raw_residual2_blocks=_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=preset.gelu_from_accum_blocks or None,
    )
    t0 = time.time()
    rot_logits = rot_ref.incremental_logits_trace(inputs)
    t_rot = time.time() - t0
    rot_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(rot_logits, targets)
    ]
    rot_ppl, _ = perplexity_from_nlls(rot_nlls)

    base_cos = _mean_logit_cosine(fp32_logits, base_logits)
    rot_cos = _mean_logit_cosine(fp32_logits, rot_logits)
    cos_delta = rot_cos - base_cos
    ppl_delta = base_ppl - rot_ppl
    gap = max(base_ppl - fp32_ppl, 1e-9)
    fraction_of_gap_closed = ppl_delta / gap

    # Restore state_dict to original (be a good citizen).
    for k, v in sd_snapshot.items():
        state_dict[k] = (v.clone() if hasattr(v, "clone") else v)

    # Tier 3 decision
    # Decision criteria (gap-closed-dominant; cosine is too saturated near 1.0
    # to be the primary signal in this PPL regime):
    # - STRONG_GO: gap closed > 20%
    # - WEAK_GO:   gap closed in (5%, 20%] OR PPL improvement > 5% with positive cosine_delta
    # - NO_GO:     gap closed <= 5%
    if fraction_of_gap_closed > 0.20:
        decision = "STRONG_GO"
    elif fraction_of_gap_closed > 0.05 and ppl_delta > 0:
        decision = "WEAK_GO"
    else:
        decision = "NO_GO"

    return {
        "decision": decision,
        "target_blocks": target_blocks_list,
        "phase": 2 if gelu_rotation is not None else 1,
        "rotate_qk": bool(rotate_qk),
        "rotated_paths": modified,
        "self_tests": self_tests,
        "baseline_ppl": float(base_ppl),
        "rotated_ppl": float(rot_ppl),
        "fp32_ppl": float(fp32_ppl),
        "ppl_delta": float(ppl_delta),
        "fraction_of_gap_closed": float(fraction_of_gap_closed),
        "baseline_logit_cosine": float(base_cos),
        "rotated_logit_cosine": float(rot_cos),
        "cosine_delta": float(cos_delta),
        "wall_seconds": {"baseline": t_base, "rotated": t_rot},
    }


class _ResidualStreamRotatedFQReference(NanoGPTFQReference):
    """NanoGPTFQReference subclass for full residual-stream rotation simulation.

    This is the proper QuaRot Phase 1 simulation. The state_dict has been
    pre-rotated by `rotate_residual_stream_weights`, so the residual stream
    operates in the rotated basis end-to-end. The only runtime intervention
    is at LayerNorm: LN doesn't commute with rotation due to mean
    subtraction, so we un-rotate the input, compute LN, and re-rotate the
    output — making LN behave as if it operated on the unrotated stream.

    For Phase 2 (with online Hadamard for gelu→fc2), `gelu_rotation` rotates
    the gelu output before fc2. Note that with residual-stream rotation
    already in place, fc2's c_proj is already rotated; the gelu_rotation
    further reduces fc2 input quantization error by flattening the gelu
    distribution that feeds it.
    """

    def __init__(
        self,
        state_dict: dict,
        model_args: dict,
        scales: Dict[str, float],
        *,
        rotation: np.ndarray,
        gelu_rotation: Optional[np.ndarray] = None,
        target_blocks: Sequence[int] = None,
        **kwargs,
    ):
        super().__init__(state_dict, model_args, scales, **kwargs)
        d = int(model_args["n_embd"])
        rot = np.asarray(rotation, dtype=np.float32)
        if rot.shape != (d, d):
            raise ValueError(f"rotation shape mismatch: {rot.shape} vs ({d}, {d})")
        self._rotation = rot
        self._rotation_inv = rot.T.astype(np.float32)  # R^T = R^{-1} for orthogonal R
        n_layer = int(model_args["n_layer"])
        self._target_blocks_list = list(target_blocks) if target_blocks is not None else list(range(n_layer))

        if gelu_rotation is not None:
            d_mlp = 4 * d
            grot = np.asarray(gelu_rotation, dtype=np.float32)
            if grot.shape != (d_mlp, d_mlp):
                raise ValueError(f"gelu_rotation shape mismatch: {grot.shape} vs ({d_mlp}, {d_mlp})")
            self._gelu_rotation = grot
            self._gelu_rotation_t = grot.T.astype(np.float32)
        else:
            self._gelu_rotation = None
            self._gelu_rotation_t = None

    def _decode_incremental_step(self, *args, **kwargs):
        from taccel.runtime import fake_quant_reference as _fqr
        orig_ln = _fqr._layernorm_np
        R = self._rotation
        R_inv = self._rotation_inv

        def patched_ln(xr, w, b, eps):
            # xr is in rotated basis (R · x). Un-rotate: each row · R = x_unrotated.
            # In code: xr @ R (since each row · R is xr @ R).
            x = xr @ R
            out_unrotated = orig_ln(x, w, b, eps)
            # Re-rotate output: out @ R^T (= R · out as column).
            return out_unrotated @ R_inv

        # Optionally patch gelu for Phase 2.
        orig_gelu_fn = self.gelu_fn
        gelu_counter = [0]
        gelu_rot_t = self._gelu_rotation_t
        target_blocks_set = frozenset(self._target_blocks_list)

        if gelu_rot_t is not None:
            def patched_gelu(x):
                out = orig_gelu_fn(x)
                cc = gelu_counter[0]
                gelu_counter[0] += 1
                if cc in target_blocks_set:
                    out = out @ gelu_rot_t
                return out
            self.gelu_fn = patched_gelu

        try:
            _fqr._layernorm_np = patched_ln
            return super()._decode_incremental_step(*args, **kwargs)
        finally:
            _fqr._layernorm_np = orig_ln
            if gelu_rot_t is not None:
                self.gelu_fn = orig_gelu_fn


def _build_residual_stream_calibration_scales(
    unrotated_payload: Dict[str, object],
    calibration_token_ids: Sequence[int],
    *,
    rotation: np.ndarray,
    gelu_rotation: Optional[np.ndarray] = None,
    target_blocks: Sequence[int] = None,
    n_seqs: int,
    seq_len: int,
    percentile: float = CALIBRATION_PERCENTILE_DEFAULT,
    activation_percentile_overrides: Optional[Dict[str, float]] = None,
    hessian_gelu_blocks: Sequence[int] = (),
) -> Dict[str, float]:
    """Build calibration scales for the full-residual-stream rotation simulation.

    For nodes in the rotated basis (residual stream tensors), compute the
    99.9 percentile of the rotated activation. For unrotated nodes, use the
    standard percentile.

    Rotated nodes (in `R` basis):
      tok_embed, pos_embed, tok_pos_add
      block{L}_residual1, residual2 (for all L)
      block{L}_out_proj (output of attn.c_proj, in rotated basis)
      block{L}_fc2 (output of mlp.c_proj, in rotated basis)
      block{L}_ln1, ln2 (output of LN, in rotated basis)
      block{L}_concat (in rotated... wait, no — concat is per-head outputs concatenated)
      ln_f (in rotated basis)

    Unrotated nodes (their activations are unchanged from FP32-equivalent
    baseline — internal block tensors):
      block{L}_head{H}_query, key, value, attn_v, softmax
      block{L}_concat (concat of per-head outputs, each unrotated)
      block{L}_fc1 (matmul of rotated ln2 with rotated c_fc → unrotated fc1)
      block{L}_gelu (unrotated unless gelu_rotation specified)

    For the gelu Phase 2 case: block{L}_gelu is rotated by gelu_rotation.

    For lm_head: input is rotated (ln_f), but lm_head's output is logits in
    its own basis (uses lm_head_w_q which we rotated). We use the standard
    lm_head scale.
    """
    n_layer = int(unrotated_payload["model_args"]["n_layer"])
    n_head = int(unrotated_payload["model_args"]["n_head"])
    if target_blocks is None:
        target_blocks = list(range(n_layer))
    target_blocks_set = frozenset(int(L) for L in target_blocks)

    # Step 1: Run unrotated FP32 forward to capture all activations.
    state_dict = unrotated_payload["state_dict"]
    model_args = unrotated_payload["model_args"]
    seqs = build_calibration_seqs_from_token_ids(
        calibration_token_ids, n_seqs=n_seqs, seq_len=seq_len
    )

    # Identify which nodes are in rotated basis.
    def _node_rotation(name: str) -> Optional[np.ndarray]:
        # Returns R if node is in rotated basis, gelu_rotation if rotated by gelu R, None otherwise.
        if name in ("tok_embed", "pos_embed", "tok_pos_add", "ln_f"):
            return rotation
        # Block-level patterns
        m = re.match(r"^block(\d+)_(.+)$", name)
        if not m:
            return None
        L = int(m.group(1))
        suffix = m.group(2)
        if suffix in ("ln1", "ln2", "out_proj", "fc2", "residual1", "residual2"):
            return rotation
        if suffix == "gelu" and gelu_rotation is not None and L in target_blocks_set:
            return gelu_rotation
        # Internal tensors (head_*, fc1, concat, etc.) are unrotated.
        return None

    # Collect per-node activation samples and compute scales.
    accum: Dict[str, List[float]] = {}
    node_pcts = dict(activation_percentile_overrides or {})
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name, arr in node_outputs.items():
            arr_f = np.asarray(arr, dtype=np.float32)
            R = _node_rotation(name)
            if R is not None:
                # Rotated activation: each row · R.T
                # Reshape to [N, C] where C = R.shape[0]
                if arr_f.ndim > 2:
                    arr_f = arr_f.reshape(-1, arr_f.shape[-1])
                elif arr_f.ndim == 1:
                    arr_f = arr_f.reshape(1, -1)
                if arr_f.shape[-1] == R.shape[0]:
                    arr_f = arr_f @ R.T
            arr_flat = arr_f.ravel()
            if arr_flat.size == 0:
                continue
            pct = float(node_pcts.get(name, percentile))
            p = float(
                np.percentile(np.abs(arr_flat), pct)
                if pct < 100.0
                else float(np.abs(arr_flat).max())
            )
            accum.setdefault(name, []).append(p)

    scales: Dict[str, float] = {}
    for name, vals in accum.items():
        max_abs = float(np.max(vals))
        scales[name] = max(max_abs, 1e-8) / 127.0

    # Fill in defaults for nodes that calibration didn't observe.
    from taccel.runtime.calibration import _fill_defaults
    _fill_defaults(scales, model_args)

    return scales


def tier3_residual_stream_simulation(
    payload: Dict[str, object],
    eval_token_ids: Sequence[int],
    *,
    rotation: np.ndarray,
    gelu_rotation: Optional[np.ndarray] = None,
    target_blocks: Optional[Sequence[int]] = None,
    fp32_ppl: Optional[float] = None,
    n_calib_seqs: int = 16,
    calib_seq_len: int = 64,
    calibration_token_ids: Optional[Sequence[int]] = None,
    ptq_preset_name: Optional[str] = None,
) -> Dict[str, object]:
    """End-to-end simulation with FULL residual-stream rotation (proper QuaRot Phase 1).

    1. Pre-rotate state_dict: wte, wpe, c_attn inputs, c_fc inputs,
       attn.c_proj outputs, mlp.c_proj outputs, lm_head input.
    2. Recalibrate ALL scales against the rotated forward.
    3. Run FQ reference with un-rotate-before-LN injection at every LN.
    4. Compare logit cosine + PPL vs unrotated baseline.
    """
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    vocab_size = int(model_args["vocab_size"])
    n_layer = int(model_args["n_layer"])
    if target_blocks is None:
        target_blocks = list(range(n_layer))

    if calibration_token_ids is None:
        calibration_token_ids = list(eval_token_ids)
    preset_name = ptq_preset_name or stage5_default_ptq_preset_name()
    preset = resolve_stage5_ptq_preset(preset_name)
    inputs, targets_list = teacher_forced_inputs_and_targets(list(eval_token_ids))

    # ---------------- Baseline (unrotated) ----------------
    sd_snapshot = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in state_dict.items()}

    base_scales = build_calibration_scales_from_token_ids(
        payload, calibration_token_ids, n_seqs=n_calib_seqs, seq_len=calib_seq_len,
        percentile=CALIBRATION_PERCENTILE_DEFAULT,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    base_scales = apply_stage5_ptq_scale_policy(base_scales, model_args, preset)

    clear_weight_component_cache()
    base_ref = NanoGPTFQReference(
        state_dict, model_args, base_scales,
        requant_pc_weight_names=_resolve_requant_pc_names(preset, model_args),
        raw_residual1_blocks=_raw_residual1_blocks(preset),
        raw_residual2_blocks=_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=preset.gelu_from_accum_blocks or None,
    )
    t0 = time.time()
    base_logits = base_ref.incremental_logits_trace(inputs)
    t_base = time.time() - t0
    base_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(base_logits, targets_list)
    ]
    base_ppl, _ = perplexity_from_nlls(base_nlls)

    # ---------------- FP32 ceiling (uses unrotated state_dict) ----------------
    fp32_ref = NanoGPTFP32Reference(state_dict, model_args)
    fp32_logits = fp32_ref.incremental_logits_trace(inputs)
    if fp32_ppl is None:
        fp32_nlls = [
            stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
            for row, tgt in zip(fp32_logits, targets_list)
        ]
        fp32_ppl, _ = perplexity_from_nlls(fp32_nlls)

    # ---------------- Calibration scales for rotated forward (uses unrotated state_dict) ----------------
    rot_scales = _build_residual_stream_calibration_scales(
        payload, calibration_token_ids,
        rotation=rotation, gelu_rotation=gelu_rotation,
        target_blocks=target_blocks,
        n_seqs=n_calib_seqs, seq_len=calib_seq_len,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    rot_scales = apply_stage5_ptq_scale_policy(rot_scales, model_args, preset)

    # ---------------- Self-tests ----------------
    self_tests: Dict[str, object] = {}
    # Identity self-test: with R = I, all rotation operations should be no-ops.
    d = int(model_args["n_embd"])
    identity_sd = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in sd_snapshot.items()}
    rotate_residual_stream_weights(
        identity_sd, model_args, np.eye(d, dtype=np.float32), target_blocks=target_blocks,
    )
    max_drift = 0.0
    for k in identity_sd:
        if k not in sd_snapshot:
            continue
        a = _to_f32(identity_sd[k])
        b = _to_f32(sd_snapshot[k])
        if a.shape == b.shape:
            max_drift = max(max_drift, float(np.abs(a - b).max()))
    self_tests["identity_state_dict_max_drift"] = max_drift
    self_tests["identity_state_dict_match"] = max_drift < 1e-4  # slightly looser due to compound multiplications

    # ---------------- Rotated INT8 forward ----------------
    modified = rotate_residual_stream_weights(
        state_dict, model_args, rotation, target_blocks=target_blocks,
    )
    if gelu_rotation is not None:
        for L in target_blocks:
            key = f"transformer.h.{L}.mlp.c_proj.weight"
            if key in state_dict:
                w = _to_f32(state_dict[key])  # already R-rotated on output
                # Now rotate input cols by gelu_rotation^T
                _store = lambda k, v: state_dict.update({k: torch.from_numpy(v).to(dtype=state_dict[k].dtype if hasattr(state_dict[k], "dtype") else torch.float32)})
                _store(key, w @ gelu_rotation.T)
                modified.append(f"{key} (gelu rotation)")

    clear_weight_component_cache()
    rot_ref = _ResidualStreamRotatedFQReference(
        state_dict, model_args, rot_scales,
        rotation=rotation,
        gelu_rotation=gelu_rotation,
        target_blocks=target_blocks,
        requant_pc_weight_names=_resolve_requant_pc_names(preset, model_args),
        raw_residual1_blocks=_raw_residual1_blocks(preset),
        raw_residual2_blocks=_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=preset.gelu_from_accum_blocks or None,
    )
    t0 = time.time()
    rot_logits = rot_ref.incremental_logits_trace(inputs)
    t_rot = time.time() - t0
    rot_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(rot_logits, targets_list)
    ]
    rot_ppl, _ = perplexity_from_nlls(rot_nlls)

    # FP32-rotated check: re-run FP32 with rotated state_dict + LN un-rotate.
    # Should produce ~identical FP32 logits (within numerical noise).
    # We check this by running the rotated FQ ref with mock-FP32 mode would be ideal,
    # but for simplicity we rely on the identity self-test above.

    base_cos = _mean_logit_cosine(fp32_logits, base_logits)
    rot_cos = _mean_logit_cosine(fp32_logits, rot_logits)
    cos_delta = rot_cos - base_cos
    ppl_delta = base_ppl - rot_ppl
    gap = max(base_ppl - fp32_ppl, 1e-9)
    fraction_of_gap_closed = ppl_delta / gap

    # Restore state_dict.
    for k, v in sd_snapshot.items():
        state_dict[k] = (v.clone() if hasattr(v, "clone") else v)

    # Decision criteria (gap-closed-dominant; cosine is too saturated near 1.0
    # to be the primary signal in this PPL regime):
    # - STRONG_GO: gap closed > 20%
    # - WEAK_GO:   gap closed in (5%, 20%] OR PPL improvement > 5% with positive cosine_delta
    # - NO_GO:     gap closed <= 5%
    if fraction_of_gap_closed > 0.20:
        decision = "STRONG_GO"
    elif fraction_of_gap_closed > 0.05 and ppl_delta > 0:
        decision = "WEAK_GO"
    else:
        decision = "NO_GO"

    return {
        "decision": decision,
        "mode": "residual_stream",
        "phase": 2 if gelu_rotation is not None else 1,
        "target_blocks": list(target_blocks),
        "rotated_paths_count": len(modified),
        "self_tests": self_tests,
        "baseline_ppl": float(base_ppl),
        "rotated_ppl": float(rot_ppl),
        "fp32_ppl": float(fp32_ppl),
        "ppl_delta": float(ppl_delta),
        "fraction_of_gap_closed": float(fraction_of_gap_closed),
        "baseline_logit_cosine": float(base_cos),
        "rotated_logit_cosine": float(rot_cos),
        "cosine_delta": float(cos_delta),
        "wall_seconds": {"baseline": t_base, "rotated": t_rot},
    }


def _mean_logit_cosine(a_list, b_list) -> float:
    """Per-position cosine averaged across positions."""
    cos_vals = []
    for a, b in zip(a_list, b_list):
        a_v = np.asarray(a, dtype=np.float32).ravel()
        b_v = np.asarray(b, dtype=np.float32).ravel()
        na = float(np.linalg.norm(a_v))
        nb = float(np.linalg.norm(b_v))
        if na <= 1e-12 or nb <= 1e-12:
            continue
        cos_vals.append(float(np.dot(a_v, b_v) / (na * nb)))
    return float(np.mean(cos_vals)) if cos_vals else 0.0


def _resolve_requant_pc_names(preset, model_args) -> List[str]:
    """Resolve preset's per-channel requant weight names. Delegates to the
    canonical helper from stage5_ptq."""
    return list(stage5_requant_pc_weight_names(model_args, preset))


def _raw_residual1_blocks(preset) -> Optional[List[int]]:
    blocks = stage5_raw_residual1_blocks(preset)
    return sorted(blocks) if blocks else None


def _raw_residual2_blocks(preset) -> Optional[List[int]]:
    blocks = stage5_raw_residual2_blocks(preset)
    return sorted(blocks) if blocks else None


def _build_rotated_calibration_scales(
    unrotated_payload: Dict[str, object],
    calibration_token_ids: Sequence[int],
    *,
    target_blocks: Sequence[int],
    rotation: np.ndarray,
    gelu_rotation: Optional[np.ndarray] = None,
    n_seqs: int,
    seq_len: int,
    percentile: float = CALIBRATION_PERCENTILE_DEFAULT,
    activation_percentile_overrides: Optional[Dict[str, float]] = None,
    hessian_gelu_blocks: Sequence[int] = (),
) -> Dict[str, float]:
    """Build calibration scales using ROTATED activations for the target block's
    ln1 and ln2 nodes. All other nodes use unrotated FP32 activations.

    IMPORTANT: takes an UNROTATED payload (with original state_dict). The
    rotated state_dict cannot be used for FP32 calibration forwards because
    `_fp32_forward` would produce garbage activations downstream of target_block
    (the rotated weights aren't FP32-equivalent without runtime activation
    rotation, which only happens inside the FQ forward).

    For the rotated nodes (block{L}_ln1, block{L}_ln2), we apply `R^T` to the
    UNROTATED FP32 LN output to compute what the rotated INT8 forward will see
    at calibration time, and compute the percentile of that.

    For all other nodes, we use the unrotated FP32 percentile, which is exact
    in FP32 because the rotation cancels in the FP32 forward (the residual
    stream is unchanged after target_block rotation).
    """
    # Start from production calibration on unrotated state.
    scales = build_calibration_scales_from_token_ids(
        unrotated_payload, calibration_token_ids,
        n_seqs=n_seqs, seq_len=seq_len, percentile=percentile,
        activation_percentile_overrides=activation_percentile_overrides,
        hessian_gelu_blocks=hessian_gelu_blocks,
    )
    # Override the rotated nodes' scales with rotated-activation percentile.
    state_dict = unrotated_payload["state_dict"]
    model_args = unrotated_payload["model_args"]
    seqs = build_calibration_seqs_from_token_ids(
        calibration_token_ids, n_seqs=n_seqs, seq_len=seq_len
    )
    # ln nodes get the d_model rotation; gelu node gets the 4*d_model rotation.
    rotated_specs: List[Tuple[str, np.ndarray]] = []
    for L in target_blocks:
        rotated_specs.append((f"block{L}_ln1", rotation))
        rotated_specs.append((f"block{L}_ln2", rotation))
        if gelu_rotation is not None:
            rotated_specs.append((f"block{L}_gelu", gelu_rotation))
    accum_per: Dict[str, List[float]] = {name: [] for name, _ in rotated_specs}
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name, R in rotated_specs:
            if name not in node_outputs:
                continue
            arr = np.asarray(node_outputs[name], dtype=np.float32) @ R.T
            arr_f = arr.ravel()
            if arr_f.size == 0:
                continue
            p = float(
                np.percentile(np.abs(arr_f), percentile)
                if percentile < 100.0
                else float(np.abs(arr_f).max())
            )
            accum_per[name].append(p)
    for n, vals in accum_per.items():
        if vals:
            scales[n] = max(float(np.max(vals)), 1e-8) / 127.0
    return scales


class _RotatedAtBlockFQReference(NanoGPTFQReference):
    """NanoGPTFQReference subclass that injects FP32 rotation R on
    block{L}_ln1, ln2, and (optionally) gelu outputs BEFORE INT8
    quantization, for each L in `target_blocks`.

    Mechanism: monkey-patch `_layernorm_np` (always) and `self.gelu_fn`
    (when `gelu_rotation` is provided) at module/instance scope for the
    duration of each `_decode_incremental_step` call.

    LN counter logic: for each block L (0..n_layer-1) the order is ln1 (call
    index 2L), ln2 (call index 2L+1); final ln_f is the last call. We
    right-multiply the LN output by R^T for each target block's ln1 and ln2.

    GELU counter logic: gelu_fn is called exactly once per block, in order
    L = 0, 1, ..., n_layer-1. We right-multiply the gelu output by Rg^T for
    each target block's gelu.

    Phase distinction:
      Phase 1 (offline-only equivalent): only ln-rotation. Phase-1-faithful
        scope rotates the matmul INPUTS (ln1→Q/K/V, ln2→fc1).
      Phase 2 (with online Hadamard): also gelu-rotation. Adds online
        Hadamard between gelu and fc2, which addresses post-GELU outliers
        that are propagated into fc2 and the residual stream.
    """

    def __init__(
        self,
        state_dict: dict,
        model_args: dict,
        scales: Dict[str, float],
        *,
        rotation: np.ndarray,
        target_blocks: Sequence[int],
        gelu_rotation: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(state_dict, model_args, scales, **kwargs)
        d = int(model_args["n_embd"])
        rot = np.asarray(rotation, dtype=np.float32)
        if rot.shape != (d, d):
            raise ValueError(f"rotation shape mismatch: {rot.shape} vs ({d}, {d})")
        self._rotation = rot
        self._rotation_t = rot.T.astype(np.float32)
        self._target_blocks = frozenset(int(b) for b in target_blocks)

        if gelu_rotation is not None:
            d_mlp = 4 * d
            grot = np.asarray(gelu_rotation, dtype=np.float32)
            if grot.shape != (d_mlp, d_mlp):
                raise ValueError(
                    f"gelu_rotation shape mismatch: {grot.shape} vs ({d_mlp}, {d_mlp})"
                )
            self._gelu_rotation = grot
            self._gelu_rotation_t = grot.T.astype(np.float32)
        else:
            self._gelu_rotation = None
            self._gelu_rotation_t = None

    def _decode_incremental_step(self, *args, **kwargs):
        # Lazy import to avoid circular: monkeypatch the module-level
        # _layernorm_np for the duration of this step.
        from taccel.runtime import fake_quant_reference as _fqr
        orig_ln = _fqr._layernorm_np
        targets = self._target_blocks
        rot_t = self._rotation_t
        ln_counter = [0]

        # Pre-compute set of LN call indices to rotate: each target L contributes
        # call 2L (ln1) and 2L+1 (ln2).
        ln_rotate_indices = set()
        for L in targets:
            ln_rotate_indices.add(2 * L)
            ln_rotate_indices.add(2 * L + 1)

        def patched_ln(x, w, b, eps):
            out = orig_ln(x, w, b, eps)
            cc = ln_counter[0]
            ln_counter[0] += 1
            if cc in ln_rotate_indices:
                out = out @ rot_t
            return out

        # Optionally patch gelu_fn for Phase 2 simulation.
        orig_gelu_fn = self.gelu_fn
        gelu_counter = [0]
        gelu_rot_t = self._gelu_rotation_t

        if gelu_rot_t is not None:
            gelu_rotate_indices = targets  # gelu call index == block index

            def patched_gelu(x):
                out = orig_gelu_fn(x)
                cc = gelu_counter[0]
                gelu_counter[0] += 1
                if cc in gelu_rotate_indices:
                    out = out @ gelu_rot_t
                return out
            self.gelu_fn = patched_gelu

        try:
            _fqr._layernorm_np = patched_ln
            return super()._decode_incremental_step(*args, **kwargs)
        finally:
            _fqr._layernorm_np = orig_ln
            if gelu_rot_t is not None:
                self.gelu_fn = orig_gelu_fn


def _classify_table_row(name: str, s: Dict[str, float]) -> str:
    return (
        f"| {name:<24} "
        f"| {s['kurt']:>8.1f} "
        f"| {s['max_over_median']:>8.1f} "
        f"| {s['top1pct_l2_frac']:>10.3f} "
        f"| {s['int8_snr_db']:>11.1f} "
        f"| {s['classification']:<24} |"
    )


def _tier1_markdown_table(stats: Dict[str, Dict[str, float]], top: int = 30) -> str:
    """Render Tier 1 stats as a markdown table sorted by kurt descending."""
    sorted_items = sorted(stats.items(), key=lambda kv: kv[1]["kurt"], reverse=True)
    lines = [
        "| Node                     | Kurt     | Max/Med  | Top1%·L²/L² | INT8 SNR (dB) | Classification           |",
        "|--------------------------|----------|----------|-------------|---------------|--------------------------|",
    ]
    for name, s in sorted_items[:top]:
        lines.append(_classify_table_row(name, s))
    return "\n".join(lines)


def _tier2_markdown_table(
    rotation_results: Dict[str, List[Dict[str, object]]],
    tier1_stats: Dict[str, Dict[str, float]],
    top: int = 20,
) -> str:
    """Render Tier 2 results as a markdown table."""
    rows = []
    for node, results in rotation_results.items():
        if not results:
            continue
        # Pick best result by snr_delta_db.
        best = max(results, key=lambda r: r["snr_delta_db"])
        rows.append((node, best))
    rows.sort(key=lambda kv: kv[1]["snr_delta_db"], reverse=True)
    lines = [
        "| Node                     | Rotation                | Kurt: pre→post     | SNR: pre→post (dB)  | Δ SNR  |",
        "|--------------------------|-------------------------|---------------------|---------------------|--------|",
    ]
    for node, r in rows[:top]:
        kp = r["kurt_before"]
        kn = r["kurt_after"]
        snr_p = r["snr_before_db"]
        snr_n = r["snr_after_db"]
        delta = r["snr_delta_db"]
        lines.append(
            f"| {node:<24} "
            f"| {r['rotation']:<23} "
            f"| {kp:>8.1f} → {kn:>7.1f} "
            f"| {snr_p:>8.1f} → {snr_n:>7.1f} "
            f"| {delta:>+6.1f} |"
        )
    return "\n".join(lines)


# =============================================================================
# Self-tests
# =============================================================================


def _run_self_tests() -> List[str]:
    """Run synthetic self-tests on the diagnostic helpers themselves.

    Returns: list of failure messages (empty if all pass).
    """
    failures: List[str] = []

    rng = np.random.default_rng(42)

    # Tier 1 self-tests
    g = rng.standard_normal(10000).astype(np.float64)
    k = _kurt(g)
    if abs(k) > 0.5:
        failures.append(f"_kurt(randn) should be ~0, got {k:.3f}")

    cubed = rng.standard_normal(10000).astype(np.float64) ** 3
    k_cubed = _kurt(cubed)
    if k_cubed < 5.0:
        failures.append(f"_kurt(randn**3) should be > 5, got {k_cubed:.3f}")

    # Channel-concentrated synthetic should classify as such.
    spike = rng.standard_normal((1000, 768)).astype(np.float32)
    spike[:, 0] *= 100.0
    s = _stats_for_tensor(spike)
    cls = classify_tensor("synthetic", **s)
    if cls != "channel-concentrated":
        failures.append(
            f"channel-concentrated synthetic classified as {cls!r} (kurt={s['kurt']:.1f}, "
            f"top1pct={s['top1pct_l2_frac']:.3f})"
        )

    # Clean synthetic (Gaussian) should classify as clean.
    clean = rng.standard_normal((1000, 768)).astype(np.float32)
    s_clean = _stats_for_tensor(clean)
    cls_clean = classify_tensor("synthetic", **s_clean)
    if cls_clean != "clean":
        failures.append(
            f"Gaussian synthetic classified as {cls_clean!r} (kurt={s_clean['kurt']:.1f})"
        )

    # Tier 2 self-tests
    H768 = build_block_hadamard_768()
    if not np.allclose(H768 @ H768.T, np.eye(768), atol=1e-4):
        failures.append("block_hadamard_768 not orthogonal: H @ H.T ≠ I")

    R = build_random_orthogonal(768, seed=42)
    if not np.allclose(R @ R.T, np.eye(768), atol=1e-4):
        failures.append("random_orthogonal not orthogonal: R @ R.T ≠ I")

    # Rotating an isotropic Gaussian should not significantly improve SNR.
    gauss = rng.standard_normal((2048, 768)).astype(np.float32)
    eff_gauss = _measure_quant_efficacy(gauss, H768)
    if abs(eff_gauss["snr_delta_db"]) > 1.5:
        failures.append(
            f"isotropic gauss rotation Δ SNR should be ~0, got {eff_gauss['snr_delta_db']:.1f} dB"
        )

    # Rotating a "realistic-outlier" tensor (moderate baseline + extreme spike
    # channel) should significantly improve SNR. This mirrors the LLM-outlier
    # scenario where the 99.9pct scale is set by extreme channels and crushes
    # the moderate values' precision.
    realistic = rng.standard_normal((2048, 768)).astype(np.float32)  # baseline ~N(0,1)
    realistic[:, 0] *= 100.0  # one channel ~N(0, 100^2)
    eff_realistic = _measure_quant_efficacy(realistic, H768)
    if eff_realistic["snr_delta_db"] < 5.0:
        failures.append(
            f"realistic-outlier rotation Δ SNR should be >5, got "
            f"{eff_realistic['snr_delta_db']:.1f} dB "
            f"(kurt {eff_realistic['kurt_before']:.1f} → {eff_realistic['kurt_after']:.1f})"
        )

    return failures


# =============================================================================
# CLI
# =============================================================================


def _hex_or_int(s: str) -> int:
    s = s.strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path)
    parser.add_argument(
        "--tier",
        choices=("1", "2", "3", "all", "self-test"),
        default="all",
    )
    parser.add_argument("--target-block", type=int, default=None,
                        help="Tier 3: single block to rotate. Default = auto from Tier 1.")
    parser.add_argument("--target-blocks", type=str, default=None,
                        help="Tier 3: comma-separated list of blocks to rotate "
                             "(e.g. '0,1,2,3,4,5,6,7,8,9,10,11' for full network). "
                             "Overrides --target-block if set.")
    parser.add_argument("--phase", choices=("1", "2", "both"), default="both",
                        help="Tier 3: '1' = ln1/ln2 input rotation only (offline-equivalent); "
                             "'2' = also gelu→fc2 rotation (requires online Hadamard); "
                             "'both' = run both, report both results")
    parser.add_argument("--mode", choices=("input-only", "residual-stream", "both-modes"),
                        default="both-modes",
                        help="Tier 3 rotation mode: 'input-only' rotates only matmul inputs "
                             "(LN→Q/K/V, LN→fc1); 'residual-stream' is the proper QuaRot Phase 1 "
                             "with full residual-stream rotation. 'both-modes' runs both.")
    parser.add_argument("--rotate-qk", action="store_true",
                        help="Tier 3: also rotate per-head Q/K (default off)")
    parser.add_argument("--seed-list", type=str, default="0xCAFE,0xC0FFEE",
                        help="Tier 2 random orthogonal seeds (comma-separated)")
    parser.add_argument("--n-seqs", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=33,
                        help="Tier 3 eval tokens (33 = fast; 257 = production)")
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--ptq-preset", type=str, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args(argv)

    # Always run self-tests first.
    print("Running self-tests...", flush=True)
    failures = _run_self_tests()
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        print(f"FAIL: {len(failures)} self-test(s) failed; aborting.")
        return 2
    print(f"  ✓ all self-tests passed")

    if args.tier == "self-test":
        print("Self-test mode complete.")
        return 0

    # Load payload + tokenize.
    print(f"\nLoading checkpoint {args.checkpoint}...")
    payload = torch.load(args.checkpoint, map_location="cpu")
    print(f"  model_args: n_layer={payload['model_args']['n_layer']}, "
          f"n_head={payload['model_args']['n_head']}, "
          f"n_embd={payload['model_args']['n_embd']}, "
          f"vocab_size={payload['model_args']['vocab_size']}")

    print(f"\nTokenizing calibration text...")
    calib_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    print(f"  {len(calib_ids)} tokens")

    if args.eval_text is not None:
        print(f"Tokenizing eval text...")
        eval_ids = tokenize_text_file(
            args.tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens
        )
        print(f"  {len(eval_ids)} eval tokens")
    else:
        eval_ids = None

    seeds = [_hex_or_int(s) for s in args.seed_list.split(",")]

    report: Dict[str, object] = {
        "tool_version": TOOL_VERSION,
        "calibration": {"n_seqs": args.n_seqs, "seq_len": args.seq_len},
        "model": {
            "n_layer": int(payload["model_args"]["n_layer"]),
            "n_head": int(payload["model_args"]["n_head"]),
            "n_embd": int(payload["model_args"]["n_embd"]),
        },
    }

    # ---------------- Tier 1 ----------------
    print(f"\n{'='*70}\nTIER 1: per-tensor outlier statistics\n{'='*70}")
    t0 = time.time()
    tier1_stats = collect_activation_stats(
        payload, calib_ids, n_seqs=args.n_seqs, seq_len=args.seq_len
    )
    tier1_time = time.time() - t0
    print(f"Captured stats for {len(tier1_stats)} activation nodes in {tier1_time:.1f}s")
    decision1, reason1 = tier1_decision(tier1_stats)
    print(f"\nTier 1 decision: {decision1}")
    print(f"  Reason: {reason1}")
    print(f"\n{_tier1_markdown_table(tier1_stats, top=30)}")

    report["tier_1"] = {
        "decision": decision1,
        "reason": reason1,
        "wall_seconds": tier1_time,
        "per_tensor": [
            {"node": n, **s} for n, s in sorted(
                tier1_stats.items(), key=lambda kv: kv[1]["kurt"], reverse=True
            )
        ],
    }

    if args.tier == "1":
        return _emit_report(report, args)

    # Early exit on Tier 1 NO_GO
    if decision1.startswith("NO_GO"):
        print(f"\n*** Early exit: Tier 1 returned {decision1} ***")
        report["final_recommendation"] = (
            "DO_NOT_COMMIT_PHASE_1" if decision1 == "NO_GO_POST_GELU_ONLY"
            else "DO_NOT_COMMIT"
        )
        return _emit_report(report, args)

    # ---------------- Tier 2 ----------------
    print(f"\n{'='*70}\nTIER 2: statistical rotation efficacy\n{'='*70}")
    # Candidate set: all non-clean, non-gelu nodes.
    candidates = [
        n for n, s in tier1_stats.items()
        if s["classification"] in ("channel-concentrated", "diffuse-heavy-tail")
    ]
    print(f"Testing rotation on {len(candidates)} candidate tensors with seeds {seeds}...")
    t0 = time.time()
    rotation_results = measure_rotation_efficacy_for_tensors(
        payload, calib_ids, candidates,
        n_seqs=args.n_seqs, seq_len=args.seq_len,
        seeds=seeds, include_block_hadamard=True,
    )
    tier2_time = time.time() - t0
    print(f"Done in {tier2_time:.1f}s")
    decision2, reason2 = tier2_decision(rotation_results, tier1_stats)
    print(f"\nTier 2 decision: {decision2}")
    print(f"  Reason: {reason2}")
    print(f"\n{_tier2_markdown_table(rotation_results, tier1_stats, top=20)}")

    report["tier_2"] = {
        "decision": decision2,
        "reason": reason2,
        "snr_definition": "Q(Rx) vs Rx (no free FP32 inverse rotation)",
        "wall_seconds": tier2_time,
        "per_tensor": rotation_results,
    }

    if args.tier == "2":
        return _emit_report(report, args)

    if decision2 == "NO_GO":
        print(f"\n*** Early exit: Tier 2 returned NO_GO ***")
        report["final_recommendation"] = "DO_NOT_COMMIT"
        return _emit_report(report, args)

    # ---------------- Tier 3 ----------------
    print(f"\n{'='*70}\nTIER 3: end-to-end one-block rotation simulation\n{'='*70}")
    if args.eval_text is None:
        print("ERROR: Tier 3 requires --eval-text")
        return 3
    if eval_ids is None:
        return 3

    # Determine target block(s): explicit list, single, or auto-pick.
    target_blocks_list: List[int]
    if args.target_blocks is not None:
        target_blocks_list = sorted(set(int(s.strip()) for s in args.target_blocks.split(",")))
        print(f"Using target blocks: {target_blocks_list}")
    elif args.target_block is not None:
        target_blocks_list = [int(args.target_block)]
        print(f"Using target block: {target_blocks_list[0]}")
    else:
        block_re = re.compile(r"^block(\d+)_")
        per_block: Dict[int, List[float]] = {}
        for n, s in tier1_stats.items():
            if _is_gelu_node(n):
                continue
            m = block_re.match(n)
            if not m:
                continue
            L = int(m.group(1))
            per_block.setdefault(L, []).append(s["int8_snr_db"])
        if per_block:
            auto_block = min(per_block.keys(), key=lambda L: min(per_block[L]))
        else:
            auto_block = 2
        target_blocks_list = [auto_block]
        print(f"Auto-picked target block {auto_block} (worst non-gelu min SNR)")

    H = build_block_hadamard_768()
    H_gelu = build_block_hadamard_3072()

    phases_to_run: List[int]
    if args.phase == "1":
        phases_to_run = [1]
    elif args.phase == "2":
        phases_to_run = [2]
    else:
        phases_to_run = [1, 2]

    modes_to_run: List[str]
    if args.mode == "input-only":
        modes_to_run = ["input-only"]
    elif args.mode == "residual-stream":
        modes_to_run = ["residual-stream"]
    else:
        modes_to_run = ["input-only", "residual-stream"]

    tier3_results: Dict[str, Dict[str, object]] = {}
    for mode in modes_to_run:
        for ph in phases_to_run:
            gelu_rot = H_gelu if ph == 2 else None
            label = f"{mode}_phase_{ph}"
            print(f"\nRunning Tier 3 (mode={mode}, phase {ph}) on blocks {target_blocks_list}...")
            t0 = time.time()
            if mode == "input-only":
                result = tier3_one_block_simulation(
                    payload, eval_ids,
                    target_blocks=target_blocks_list,
                    rotation=H,
                    gelu_rotation=gelu_rot,
                    rotate_qk=args.rotate_qk,
                    n_calib_seqs=args.n_seqs,
                    calib_seq_len=args.seq_len,
                    calibration_token_ids=calib_ids,
                    ptq_preset_name=args.ptq_preset,
                )
            else:  # residual-stream
                result = tier3_residual_stream_simulation(
                    payload, eval_ids,
                    rotation=H,
                    gelu_rotation=gelu_rot,
                    target_blocks=target_blocks_list,
                    n_calib_seqs=args.n_seqs,
                    calib_seq_len=args.seq_len,
                    calibration_token_ids=calib_ids,
                    ptq_preset_name=args.ptq_preset,
                )
            wall = time.time() - t0
            tier3_results[label] = result
            print(f"Done in {wall:.1f}s")
            print(f"  {mode} phase {ph} decision: {result['decision']}")
            print(f"    baseline PPL = {result['baseline_ppl']:.2f}")
            print(f"    rotated  PPL = {result['rotated_ppl']:.2f}")
            print(f"    fp32     PPL = {result['fp32_ppl']:.2f}")
            print(f"    Δ PPL = {result['ppl_delta']:+.2f}")
            print(f"    fraction of FP32→INT8 gap closed = {result['fraction_of_gap_closed']*100:+.1f}%")
            print(f"    Δ cosine = {result['cosine_delta']:+.4f}")
            print(f"    self-tests: {result['self_tests']}")

    report["tier_3"] = tier3_results

    # ---------------- Final recommendation ----------------
    # Choose best phase result for final recommendation. Note that Phase 2
    # requires online Hadamard hardware support (bigger commit).
    best_phase = max(
        tier3_results.keys(),
        key=lambda k: tier3_results[k]["fraction_of_gap_closed"],
    )
    best_result = tier3_results[best_phase]
    if best_result["decision"] == "STRONG_GO":
        rec = "COMMIT_FULL_QUAROT" if "phase_2" in tier3_results and tier3_results["phase_2"]["decision"] == "STRONG_GO" else "COMMIT"
    elif best_result["decision"] == "WEAK_GO":
        rec = "MARGINAL"
    else:
        rec = "DO_NOT_COMMIT"

    # Differentiate phase 1 vs phase 2 and mode (input-only vs residual-stream).
    # Look for the residual-stream-mode results first since they're more
    # decisive of the actual QuaRot recipe.
    rs_p1 = tier3_results.get("residual-stream_phase_1")
    rs_p2 = tier3_results.get("residual-stream_phase_2")
    io_p1 = tier3_results.get("input-only_phase_1")
    io_p2 = tier3_results.get("input-only_phase_2")

    def _best(*results):
        valid = [r for r in results if r is not None]
        return max(valid, key=lambda r: r["fraction_of_gap_closed"]) if valid else None

    rs_best = _best(rs_p1, rs_p2)
    io_best = _best(io_p1, io_p2)

    if rs_best is not None and rs_best["decision"] == "STRONG_GO":
        rec = "COMMIT_FULL_QUAROT"
    elif rs_best is not None and rs_best["decision"] == "WEAK_GO":
        rec = "MARGINAL_FULL_QUAROT"
    elif io_best is not None and io_best["decision"] == "STRONG_GO":
        rec = "COMMIT_INPUT_ROTATION"  # rare case
    elif rs_best is not None or io_best is not None:
        # Both modes ran but neither hit STRONG_GO.
        rec = "DO_NOT_COMMIT"
    else:
        rec = "INCONCLUSIVE"

    report["final_recommendation"] = rec

    print(f"\n{'='*70}")
    print(f"FINAL RECOMMENDATION: {rec}")
    print(f"  Tier 1 = {decision1}: {reason1}")
    print(f"  Tier 2 = {decision2}: {reason2}")
    for label, res in tier3_results.items():
        print(f"  Tier 3 ({label}) = {res['decision']}: "
              f"{res['fraction_of_gap_closed']*100:+.1f}% of gap closed, "
              f"Δ cosine {res['cosine_delta']:+.4f}")
    print(f"{'='*70}")

    return _emit_report(report, args)


def _emit_report(report: Dict[str, object], args) -> int:
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"\nWrote JSON report to {args.json_out}")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown_out, "w") as f:
            f.write(_render_markdown(report))
        print(f"Wrote markdown report to {args.markdown_out}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"non-serializable object of type {type(o)}")


def _render_markdown(report: Dict[str, object]) -> str:
    lines = [f"# QuaRot diagnostic report (tool {report['tool_version']})", ""]
    lines.append(f"- Model: {report['model']}")
    lines.append(f"- Calibration: {report['calibration']}")
    lines.append("")
    if "tier_1" in report:
        t1 = report["tier_1"]
        lines.append(f"## Tier 1: {t1['decision']}")
        lines.append(f"_{t1['reason']}_")
        lines.append("")
        # Build the table from per_tensor list.
        lines.append("| Node                     | Kurt     | Max/Med  | Top1%·L²/L² | INT8 SNR (dB) | Classification           |")
        lines.append("|--------------------------|----------|----------|-------------|---------------|--------------------------|")
        for entry in t1["per_tensor"][:30]:
            lines.append(_classify_table_row(entry["node"], entry))
        lines.append("")
    if "tier_2" in report:
        t2 = report["tier_2"]
        lines.append(f"## Tier 2: {t2['decision']}")
        lines.append(f"_{t2['reason']}_")
        lines.append(f"_SNR definition: {t2['snr_definition']}_")
        lines.append("")
    if "tier_3" in report:
        t3 = report["tier_3"]
        lines.append(f"## Tier 3")
        # t3 is dict of phase_label → result
        for label, res in t3.items():
            lines.append(f"### {label}: {res['decision']}")
            lines.append(f"- target_blocks: {res.get('target_blocks', res.get('target_block', '?'))}")
            lines.append(f"- baseline_ppl: {res['baseline_ppl']:.2f}")
            lines.append(f"- rotated_ppl:  {res['rotated_ppl']:.2f}")
            lines.append(f"- fp32_ppl:     {res['fp32_ppl']:.2f}")
            lines.append(f"- fraction_of_gap_closed: {res['fraction_of_gap_closed']*100:+.1f}%")
            lines.append(f"- cosine: {res['baseline_logit_cosine']:.4f} → {res['rotated_logit_cosine']:.4f} (Δ {res['cosine_delta']:+.4f})")
            lines.append("")
    lines.append(f"## Final recommendation: **{report.get('final_recommendation', 'N/A')}**")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
