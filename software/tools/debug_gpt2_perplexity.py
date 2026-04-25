#!/usr/bin/env python3
"""Diagnose the Stage 5 GPT-2 perplexity gap.

The tool is intentionally read-only.  It compares the same token slice across
golden INT8, compiler-matched fake-quant, true FP32, direct FP32-logit
quantization, and the local HuggingFace GPT-2 model when available.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from taccel.runtime.calibration import build_calibration_scales_from_token_ids
from taccel.runtime.fake_quant import cosine_similarity
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.fp32_reference import NanoGPTFP32Reference
from taccel.runtime.gpt2_perplexity import (
    file_sha256,
    load_gpt2_tokenizer,
    perplexity_from_nlls,
    run_fake_quant_teacher_forced_logits,
    run_golden_teacher_forced_logits,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    choose_stage5_ptq_promotion,
    choose_stage5_ptq_winner,
    rank_stage5_ptq_rows,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
    stage5_raw_residual1_blocks,
    stage5_requant_pc_weight_names,
    STAGE5_PTQ_PRESETS,
    apply_stage5_ptq_scale_policy,
)


SWEEP_CONFIGS = (
    (8, 32, 99.9),
    (32, 64, 99.9),
    (64, 128, 99.9),
    (64, 128, 100.0),
)

ABLATION_GROUPS = (
    "embeddings",
    "qkv",
    "softmax",
    "attn_v",
    "out_proj",
    "residual_vadd",
    "mlp",
    "ln_f",
    "lm_head",
)


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _quantize_logits(logits: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        return np.zeros_like(logits, dtype=np.int8)
    return np.clip(np.round(logits.astype(np.float32) / np.float32(scale)), -128, 127).astype(np.int8)


def _int8_stats(values: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.int16).reshape(-1)
    if arr.size == 0:
        return {"unique_count": 0, "saturation_rate": 0.0, "min": 0, "max": 0}
    return {
        "unique_count": int(np.unique(arr).size),
        "saturation_rate": float(np.mean(np.logical_or(arr <= -128, arr >= 127))),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def _topk_ids(logits: np.ndarray, *, k: int, vocab_size: int) -> List[int]:
    active = np.asarray(logits, dtype=np.float32)[:vocab_size]
    k = min(int(k), active.size)
    if k <= 0:
        return []
    # Stable token-id order for ties keeps reports deterministic.
    return [int(idx) for idx in np.lexsort((np.arange(active.size), -active))[:k]]


def _rank_of_target(logits: np.ndarray, target: int, *, vocab_size: int) -> int:
    active = np.asarray(logits, dtype=np.float32)[:vocab_size]
    target_i = int(target)
    target_logit = active[target_i]
    return int(1 + np.sum(active > target_logit))


def _embedding_add_scale_diagnostics(
    payload: dict,
    token_ids: Sequence[int],
    *,
    active_scale: float,
    percentile: float,
) -> Dict[str, object]:
    sd = payload["state_dict"]
    wte = np.asarray(sd["transformer.wte.weight"], dtype=np.float32)
    wpe = np.asarray(sd["transformer.wpe.weight"], dtype=np.float32)
    tokens = [int(tok) for tok in token_ids]
    if not tokens:
        return {}
    pos = list(range(len(tokens)))
    fp32 = wte[tokens] + wpe[pos]
    legacy_abs = np.abs(fp32).reshape(-1)
    raw_vadd_abs = (np.abs(wte[tokens]) + np.abs(wpe[pos])).reshape(-1)
    legacy_scale = max(
        float(np.percentile(legacy_abs, percentile) if percentile < 100.0 else np.max(legacy_abs)),
        1e-8,
    ) / 127.0
    raw_vadd_safe_scale = max(
        float(np.percentile(raw_vadd_abs, percentile) if percentile < 100.0 else np.max(raw_vadd_abs)),
        1e-8,
    ) / 127.0

    def metrics(scale: float) -> Dict[str, object]:
        q_tok = np.clip(np.round(wte[tokens] / np.float32(scale)), -128, 127).astype(np.int8)
        q_pos = np.clip(np.round(wpe[pos] / np.float32(scale)), -128, 127).astype(np.int8)
        q_sum = np.clip(q_tok.astype(np.int16) + q_pos.astype(np.int16), -128, 127).astype(np.int8)
        deq = q_sum.astype(np.float32) * np.float32(scale)
        err = np.abs(deq.reshape(-1) - fp32.reshape(-1))
        p99 = float(np.percentile(err, 99.0)) if err.size else 0.0
        return {
            "scale": float(scale),
            "raw_cosine": cosine_similarity(deq, fp32),
            "centered_cosine": _centered_cosine(deq, fp32),
            "p99_abs_error": p99,
            "p99_error_lsb": float(p99 / max(float(scale), 1e-12)),
            "int8": _int8_stats(q_sum),
        }

    return {
        "legacy_sum_scale": float(legacy_scale),
        "raw_vadd_safe_scale": float(raw_vadd_safe_scale),
        "active_scale": float(active_scale),
        "legacy": metrics(legacy_scale),
        "active": metrics(active_scale),
    }


def _decoded_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(int(token_id))


def _logit_pair_summary(lhs: Sequence[np.ndarray], rhs: Sequence[np.ndarray], *, vocab_size: int) -> Dict[str, object]:
    cosines = []
    top10 = []
    p99 = []
    for a, b in zip(lhs, rhs):
        af = np.asarray(a, dtype=np.float32)[:vocab_size]
        bf = np.asarray(b, dtype=np.float32)[:vocab_size]
        cosines.append(cosine_similarity(af, bf))
        top10.append(len(set(_topk_ids(af, k=10, vocab_size=vocab_size)) & set(_topk_ids(bf, k=10, vocab_size=vocab_size))))
        p99.append(float(np.percentile(np.abs(af - bf), 99.0)))
    if not cosines:
        return {"steps": 0, "min_cosine": 1.0, "mean_cosine": 1.0, "min_top10_overlap": 10, "max_p99_abs_error": 0.0}
    return {
        "steps": len(cosines),
        "min_cosine": float(min(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "min_top10_overlap": int(min(top10)),
        "mean_top10_overlap": float(np.mean(top10)),
        "max_p99_abs_error": float(max(p99)),
    }


def _build_scales_for_preset(payload: dict, calibration_ids: Sequence[int], args, preset_name: str) -> Dict[str, float]:
    preset = resolve_stage5_ptq_preset(preset_name)
    scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_ids,
        n_seqs=args.calibration_n_seqs,
        seq_len=args.calibration_seq_len,
        percentile=args.calibration_percentile,
        activation_percentile_overrides=(
            preset.activation_percentile_nodes or None
        ),
    )
    return apply_stage5_ptq_scale_policy(scales, payload["model_args"], preset)


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    return cosine_similarity(lhs - np.float32(lhs.mean()), rhs - np.float32(rhs.mean()))


def _trace_entry_value(entry: Dict[str, object]) -> np.ndarray:
    return np.asarray(entry.get("value", []), dtype=np.float32)


def _trace_node_metrics(
    name: str,
    fake_entry: Dict[str, object],
    fp32_entry: Dict[str, object],
    *,
    target: int | None,
    vocab_size: int,
) -> Dict[str, object]:
    fake_value = _trace_entry_value(fake_entry)
    fp32_value = _trace_entry_value(fp32_entry)
    scale = fake_entry.get("scale")
    int8_values = fake_entry.get("int8")
    abs_err = np.abs(fake_value.reshape(-1) - fp32_value.reshape(-1))
    p99 = float(np.percentile(abs_err, 99.0)) if abs_err.size else 0.0
    metrics: Dict[str, object] = {
        "node": name,
        "shape": list(fake_value.shape),
        "raw_cosine": cosine_similarity(fake_value, fp32_value),
        "centered_cosine": _centered_cosine(fake_value, fp32_value),
        "p99_abs_error": p99,
        "scale": None if scale is None else float(scale),
        "p99_error_lsb": None if scale in (None, 0) else float(p99 / float(scale)),
        "int8": None if int8_values is None else _int8_stats(np.asarray(int8_values)),
        "top10_overlap": None,
        "target_rank": None,
    }
    if name == "lm_head":
        fake_active = fake_value.reshape(-1)[:vocab_size]
        fp32_active = fp32_value.reshape(-1)[:vocab_size]
        metrics["top10_overlap"] = len(
            set(_topk_ids(fake_active, k=10, vocab_size=vocab_size))
            & set(_topk_ids(fp32_active, k=10, vocab_size=vocab_size))
        )
        if target is not None:
            metrics["target_rank"] = {
                "fake_quant": _rank_of_target(fake_active, target, vocab_size=vocab_size),
                "fp32": _rank_of_target(fp32_active, target, vocab_size=vocab_size),
            }
    return metrics


def _node_order(model_args: Dict[str, object]) -> List[str]:
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    names = ["tok_pos_add"]
    for layer in range(n_layer):
        names.append(f"block{layer}_ln1")
        for head in range(n_head):
            names.extend([
                f"block{layer}_head{head}_query",
                f"block{layer}_head{head}_key",
                f"block{layer}_head{head}_value",
                f"block{layer}_head{head}_softmax",
                f"block{layer}_head{head}_attn_v",
            ])
        names.extend([
            f"block{layer}_concat",
            f"block{layer}_out_proj",
            f"block{layer}_residual1",
            f"block{layer}_ln2",
            f"block{layer}_fc1",
            f"block{layer}_gelu",
            f"block{layer}_fc2",
            f"block{layer}_residual2",
        ])
    names.extend(["ln_f", "lm_head"])
    return names


def _divergence_reasons(metrics: Dict[str, object]) -> List[str]:
    reasons: List[str] = []
    centered = float(metrics["centered_cosine"])
    if centered < 0.95:
        reasons.append("centered_cosine<0.95")
    p99_lsb = metrics.get("p99_error_lsb")
    if p99_lsb is not None and float(p99_lsb) > 2.0:
        reasons.append("p99_error_lsb>2")
    top10 = metrics.get("top10_overlap")
    if top10 is not None and int(top10) < 7:
        reasons.append("lm_head_top10_overlap<7")
    int8_stats = metrics.get("int8")
    if isinstance(int8_stats, dict) and float(int8_stats.get("saturation_rate", 0.0)) > 0.05:
        reasons.append("saturation_rate>5%")
    return reasons


def _trace_report(
    *,
    fake_trace: Dict[str, Dict[str, object]],
    fp32_trace: Dict[str, Dict[str, object]],
    model_args: Dict[str, object],
    target: int | None,
    vocab_size: int,
    trace_node: str,
    include_first_divergence: bool,
) -> Dict[str, object]:
    selected = []
    first = None
    requested_all = trace_node == "all"
    for name in _node_order(model_args):
        if name not in fake_trace or name not in fp32_trace:
            continue
        if not requested_all and name != trace_node:
            continue
        metrics = _trace_node_metrics(
            name,
            fake_trace[name],
            fp32_trace[name],
            target=target,
            vocab_size=vocab_size,
        )
        reasons = _divergence_reasons(metrics)
        metrics["divergence_reasons"] = reasons
        selected.append(metrics)
        if include_first_divergence and first is None and reasons:
            first = {
                "node": name,
                "reasons": reasons,
                "metrics": metrics,
            }
    return {
        "trace_node": trace_node,
        "nodes": selected,
        "first_divergence": first,
    }


def _collapse_summary(nodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for node in nodes:
        stats = node.get("int8")
        if not isinstance(stats, dict):
            continue
        rows.append({
            "node": node["node"],
            "unique_count": int(stats.get("unique_count", 0)),
            "saturation_rate": float(stats.get("saturation_rate", 0.0)),
            "centered_cosine": float(node.get("centered_cosine", 1.0)),
            "p99_error_lsb": node.get("p99_error_lsb"),
        })
    rows.sort(key=lambda item: (-float(item["saturation_rate"]), int(item["unique_count"]), float(item["centered_cosine"])))
    return rows


def _ppl_from_logits(logits: Sequence[np.ndarray], targets: Sequence[int], *, vocab_size: int) -> Dict[str, object]:
    nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size)
        for row, target in zip(logits, targets)
    ]
    ppl, mean_nll = perplexity_from_nlls(nlls)
    return {
        "perplexity": ppl,
        "mean_nll": mean_nll,
        "nlls": [float(v) for v in nlls],
    }


def _per_step_rows(
    *,
    tokenizer,
    tokens: Sequence[int],
    targets: Sequence[int],
    vocab_size: int,
    lm_head_scale: float,
    golden_i8: Sequence[np.ndarray],
    fake_i8: Sequence[np.ndarray],
    fp32_logits: Sequence[np.ndarray],
    direct_quant_i8: Sequence[np.ndarray],
) -> List[Dict[str, object]]:
    rows = []
    for step, (input_id, target, golden, fake, fp32, direct_q) in enumerate(zip(tokens[:-1], targets, golden_i8, fake_i8, fp32_logits, direct_quant_i8)):
        golden_deq = np.asarray(golden, dtype=np.float32)[:vocab_size] * np.float32(lm_head_scale)
        fake_deq = np.asarray(fake, dtype=np.float32)[:vocab_size] * np.float32(lm_head_scale)
        direct_deq = np.asarray(direct_q, dtype=np.float32)[:vocab_size] * np.float32(lm_head_scale)
        fp32_active = np.asarray(fp32, dtype=np.float32)[:vocab_size]
        rows.append({
            "step": int(step),
            "input_token_id": int(input_id),
            "input_token": _decoded_token(tokenizer, int(input_id)),
            "target_token_id": int(target),
            "target_token": _decoded_token(tokenizer, int(target)),
            "target_rank": {
                "fp32": _rank_of_target(fp32_active, target, vocab_size=vocab_size),
                "direct_quant_fp32": _rank_of_target(direct_deq, target, vocab_size=vocab_size),
                "fake_quant": _rank_of_target(fake_deq, target, vocab_size=vocab_size),
                "golden": _rank_of_target(golden_deq, target, vocab_size=vocab_size),
            },
            "target_nll": {
                "fp32": stable_cross_entropy(fp32_active, target, vocab_size=vocab_size),
                "direct_quant_fp32": stable_cross_entropy(direct_deq, target, vocab_size=vocab_size),
                "fake_quant": stable_cross_entropy(fake_deq, target, vocab_size=vocab_size),
                "golden": stable_cross_entropy(golden_deq, target, vocab_size=vocab_size),
            },
            "top10_overlap": {
                "golden_vs_fake": len(set(_topk_ids(golden_deq, k=10, vocab_size=vocab_size)) & set(_topk_ids(fake_deq, k=10, vocab_size=vocab_size))),
                "fake_vs_fp32": len(set(_topk_ids(fake_deq, k=10, vocab_size=vocab_size)) & set(_topk_ids(fp32_active, k=10, vocab_size=vocab_size))),
                "direct_quant_vs_fp32": len(set(_topk_ids(direct_deq, k=10, vocab_size=vocab_size)) & set(_topk_ids(fp32_active, k=10, vocab_size=vocab_size))),
            },
            "cosine": {
                "golden_vs_fake": cosine_similarity(np.asarray(golden, dtype=np.float32)[:vocab_size], np.asarray(fake, dtype=np.float32)[:vocab_size]),
                "fake_vs_fp32": cosine_similarity(fake_deq, fp32_active),
                "direct_quant_vs_fp32": cosine_similarity(direct_deq, fp32_active),
            },
            "int8": {
                "golden": _int8_stats(np.asarray(golden)[:vocab_size]),
                "fake_quant": _int8_stats(np.asarray(fake)[:vocab_size]),
                "direct_quant_fp32": _int8_stats(np.asarray(direct_q)[:vocab_size]),
            },
            "top10_tokens": {
                "fp32": _topk_ids(fp32_active, k=10, vocab_size=vocab_size),
                "direct_quant_fp32": _topk_ids(direct_deq, k=10, vocab_size=vocab_size),
                "fake_quant": _topk_ids(fake_deq, k=10, vocab_size=vocab_size),
                "golden": _topk_ids(golden_deq, k=10, vocab_size=vocab_size),
            },
        })
    return rows


def _fp32_incremental_and_full(ref: NanoGPTFP32Reference, tokens: Sequence[int]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    inputs, _ = teacher_forced_inputs_and_targets(tokens)
    incremental = ref.incremental_logits_trace(inputs)
    full = [ref.forward(inputs[: idx + 1]) for idx in range(len(inputs))]
    return incremental, full


def _dequantize_fake_logits(row: np.ndarray, lm_scale: float, *, group: str | None = None) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float32)
    if group == "lm_head":
        return arr
    return arr * np.float32(lm_scale)


def _fake_incremental_and_full(ref: NanoGPTFQReference, tokens: Sequence[int]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    inputs, _ = teacher_forced_inputs_and_targets(tokens)
    incremental = ref.incremental_logits_trace(inputs)
    full = [ref.forward(inputs[: idx + 1]) for idx in range(len(inputs))]
    return incremental, full


def _ablation_sweep(
    *,
    payload: dict,
    scales: Dict[str, float],
    inputs: Sequence[int],
    targets: Sequence[int],
    fp32_logits: Sequence[np.ndarray],
    golden_i8: Sequence[np.ndarray],
    fake_i8: Sequence[np.ndarray],
    vocab_size: int,
    lm_scale: float,
) -> List[Dict[str, object]]:
    baseline_fake_deq = [
        _dequantize_fake_logits(row, lm_scale)[:vocab_size]
        for row in fake_i8
    ]
    baseline_ppl = _ppl_from_logits(baseline_fake_deq, targets, vocab_size=vocab_size)
    baseline_summary = _logit_pair_summary(
        [np.asarray(row, dtype=np.float32)[:vocab_size] for row in golden_i8],
        [np.asarray(row, dtype=np.float32)[:vocab_size] for row in fake_i8],
        vocab_size=vocab_size,
    )
    rows = []
    for group in ABLATION_GROUPS:
        ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], scales)
        logits = ref.incremental_logits_trace(inputs, fp32_groups={group})
        deq = [
            _dequantize_fake_logits(row, lm_scale, group=group)[:vocab_size]
            for row in logits
        ]
        ppl = _ppl_from_logits(deq, targets, vocab_size=vocab_size)
        pair = _logit_pair_summary(deq, fp32_logits, vocab_size=vocab_size)
        rows.append({
            "group": group,
            "perplexity": ppl["perplexity"],
            "mean_nll": ppl["mean_nll"],
            "nll_improvement_vs_baseline": float(baseline_ppl["mean_nll"] - ppl["mean_nll"]),
            "min_top10_overlap_vs_fp32": pair["min_top10_overlap"],
            "min_cosine_vs_fp32": pair["min_cosine"],
            "mean_cosine_vs_fp32": pair["mean_cosine"],
            "baseline_golden_vs_fake_min_cosine": baseline_summary["min_cosine"],
            "baseline_golden_vs_fake_min_top10_overlap": baseline_summary["min_top10_overlap"],
        })
    return rows


def _best_ablation(ablation_rows: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    if not ablation_rows:
        return None
    return max(
        ablation_rows,
        key=lambda row: (
            float(row.get("nll_improvement_vs_baseline", 0.0)),
            int(row.get("min_top10_overlap_vs_fp32", 0)),
            float(row.get("min_cosine_vs_fp32", 0.0)),
        ),
    )


def _preset_sweep(
    *,
    payload: dict,
    calibration_ids: Sequence[int],
    eval_tokens: Sequence[int],
    inputs: Sequence[int],
    targets: Sequence[int],
    fp32_logits: Sequence[np.ndarray],
    vocab_size: int,
    args,
) -> Dict[str, object]:
    rows: list[dict[str, object]] = []
    for preset_name in STAGE5_PTQ_PRESETS:
        preset = resolve_stage5_ptq_preset(preset_name)
        scales = _build_scales_for_preset(payload, calibration_ids, args, preset.name)
        lm_scale = float(scales.get("lm_head", 1.0))
        golden_i8 = run_golden_teacher_forced_logits(
            payload,
            eval_tokens,
            scales,
            ptq_preset=preset,
        )
        fake_i8 = run_fake_quant_teacher_forced_logits(
            payload,
            eval_tokens,
            scales,
            ptq_preset=preset,
        )
        golden_deq = [np.asarray(row, dtype=np.float32)[:vocab_size] * np.float32(lm_scale) for row in golden_i8]
        fake_deq = [np.asarray(row, dtype=np.float32)[:vocab_size] * np.float32(lm_scale) for row in fake_i8]
        golden_ppl = _ppl_from_logits(golden_deq, targets, vocab_size=vocab_size)
        fake_ppl = _ppl_from_logits(fake_deq, targets, vocab_size=vocab_size)
        pair_fake_fp32 = _logit_pair_summary(fake_deq, fp32_logits, vocab_size=vocab_size)
        pair_golden_fake = _logit_pair_summary(golden_i8, fake_i8, vocab_size=vocab_size)
        expected_top10 = min(10, int(vocab_size))
        rows.append({
            "name": preset.name,
            "fake_quant_perplexity": float(fake_ppl["perplexity"]),
            "golden_perplexity": float(golden_ppl["perplexity"]),
            "relative_delta": abs(golden_ppl["perplexity"] - fake_ppl["perplexity"]) / max(abs(fake_ppl["perplexity"]), 1e-12),
            "mean_target_nll": float(fake_ppl["mean_nll"]),
            "min_top10_overlap_vs_fp32": int(pair_fake_fp32["min_top10_overlap"]),
            "mean_top10_overlap_vs_fp32": float(pair_fake_fp32["mean_top10_overlap"]),
            "min_cosine_vs_fp32": float(pair_fake_fp32["min_cosine"]),
            "mean_cosine_vs_fp32": float(pair_fake_fp32["mean_cosine"]),
            "golden_vs_fake_min_cosine": float(pair_golden_fake["min_cosine"]),
            "golden_vs_fake_min_top10_overlap": int(pair_golden_fake["min_top10_overlap"]),
            "logits_gate_passed": bool(
                pair_golden_fake["min_cosine"] >= 0.995
                and pair_golden_fake["min_top10_overlap"] >= expected_top10
            ),
        })

    ranked = rank_stage5_ptq_rows(rows)
    winner = choose_stage5_ptq_winner(rows)
    proposed = choose_stage5_ptq_promotion(
        rows,
        gate_passed=bool(winner and winner.get("logits_gate_passed")),
    )
    promoted_default = stage5_default_ptq_preset_name()
    return {
        "rows": ranked,
        "winner": winner,
        "proposed_promotion": proposed,
        "promoted_default": promoted_default,
        "winner_promoted": bool(winner and winner["name"] == promoted_default),
    }


def _load_hf_logits(tokenizer_dir: Path, inputs: Sequence[int]) -> tuple[List[np.ndarray] | None, Dict[str, object]]:
    try:
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(str(tokenizer_dir), local_files_only=True)
        model.eval()
        with torch.no_grad():
            ids = torch.tensor([list(map(int, inputs))], dtype=torch.long)
            out = model(input_ids=ids)
            logits = out.logits[0].detach().cpu().numpy().astype(np.float32)
        return [logits[idx].copy() for idx in range(len(inputs))], {"available": True, "source": str(tokenizer_dir)}
    except Exception as exc:
        return None, {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _calibration_sweep(
    payload: dict,
    *,
    calibration_ids: Sequence[int],
    eval_tokens: Sequence[int],
    vocab_size: int,
    tokenizer_dir: Path,
    args,
    ptq_preset: str,
) -> List[Dict[str, object]]:
    rows = []
    _, targets = teacher_forced_inputs_and_targets(eval_tokens)
    for n_seqs, seq_len, percentile in SWEEP_CONFIGS:
        preset = resolve_stage5_ptq_preset(ptq_preset)
        scales = build_calibration_scales_from_token_ids(
            payload,
            calibration_ids,
            n_seqs=n_seqs,
            seq_len=seq_len,
            percentile=percentile,
            activation_percentile_overrides=(
                preset.activation_percentile_nodes or None
            ),
        )
        scales = apply_stage5_ptq_scale_policy(scales, payload["model_args"], preset)
        lm_scale = float(scales.get("lm_head", 1.0))
        golden = run_golden_teacher_forced_logits(payload, eval_tokens, scales, ptq_preset=preset)
        fake = run_fake_quant_teacher_forced_logits(payload, eval_tokens, scales, ptq_preset=preset)
        golden_deq = [np.asarray(row, dtype=np.float32)[:vocab_size] * np.float32(lm_scale) for row in golden]
        fake_deq = [np.asarray(row, dtype=np.float32)[:vocab_size] * np.float32(lm_scale) for row in fake]
        golden_ppl = _ppl_from_logits(golden_deq, targets, vocab_size=vocab_size)
        fake_ppl = _ppl_from_logits(fake_deq, targets, vocab_size=vocab_size)
        stats = [_int8_stats(np.asarray(row)[:vocab_size]) for row in golden]
        rows.append({
            "n_seqs": int(n_seqs),
            "seq_len": int(seq_len),
            "percentile": float(percentile),
            "ptq_preset": preset.name,
            "lm_head_scale": lm_scale,
            "golden_perplexity": golden_ppl["perplexity"],
            "fake_quant_perplexity": fake_ppl["perplexity"],
            "relative_delta": abs(golden_ppl["perplexity"] - fake_ppl["perplexity"]) / max(abs(fake_ppl["perplexity"]), 1e-12),
            "min_golden_vs_fake_cosine": _logit_pair_summary(golden, fake, vocab_size=vocab_size)["min_cosine"],
            "mean_saturation_rate": float(np.mean([s["saturation_rate"] for s in stats])),
            "min_unique_count": int(min(s["unique_count"] for s in stats)),
            "tokenizer_dir": str(tokenizer_dir),
        })
    return rows


def choose_primary_suspect(report: Dict[str, object]) -> str:
    hf = report["suspects"]["converter_bias_layout"]
    if hf.get("available") and (
        float(hf.get("min_cosine_vs_hf", 1.0)) < 0.999
        or float(hf.get("max_p99_abs_error_vs_hf", 0.0)) > 1e-3
        or int(hf.get("min_top10_overlap_vs_hf", 10)) < 10
    ):
        return "converter/bias/layout mismatch"

    fp32_ppl = float(report["suspects"]["fp32_baseline"]["incremental"]["perplexity"])
    direct_ppl = float(report["suspects"]["lm_head_quantization"]["direct_quant_fp32"]["perplexity"])
    fake_ppl = float(report["suspects"]["perplexity"]["fake_quant_perplexity"])
    fp32_full_inc = report["suspects"]["shared_decode_semantics"]["fp32_full_vs_incremental"]
    fake_full_inc = report["suspects"]["shared_decode_semantics"]["fake_full_vs_incremental"]

    if fp32_ppl > 1.0e6:
        return "eval slice/context issue"
    if direct_ppl > max(fp32_ppl * 10.0, 1.0e6):
        return "lm_head quantization/scale issue"
    if fake_ppl > direct_ppl * 10.0:
        return "upstream activation/calibration issue"
    if (
        float(fp32_full_inc.get("min_cosine", 1.0)) < 0.999
        or float(fake_full_inc.get("min_cosine", 1.0)) < 0.995
    ):
        return "decode/KV/position semantics issue"
    if hf.get("available") is False:
        return "converter/bias/layout sanity unavailable"
    return "no single dominant suspect"


def _dominant_loss_point(report: Dict[str, object]) -> Dict[str, object]:
    trace = report.get("node_trace")
    if isinstance(trace, dict) and trace.get("first_divergence"):
        first = trace["first_divergence"]
        return {
            "kind": "first_node_divergence",
            "node": first["node"],
            "reasons": first["reasons"],
        }
    best = _best_ablation(report.get("ablation_sweep", []))
    if best is not None:
        return {
            "kind": "best_ablation",
            "group": best["group"],
            "nll_improvement_vs_baseline": best["nll_improvement_vs_baseline"],
            "min_top10_overlap_vs_fp32": best["min_top10_overlap_vs_fp32"],
        }
    return {"kind": "not_computed"}


def run_report(args) -> Dict[str, object]:
    checkpoint = Path(args.checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    payload = torch.load(checkpoint, map_location="cpu")
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)
    calibration_ids = tokenize_text_file(tokenizer_dir, args.calibration_text)
    eval_ids = tokenize_text_file(tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens)
    token_budget = min(int(args.max_eval_tokens), int(args.context_len) + 1)
    eval_tokens = [int(tok) for tok in eval_ids[:token_budget]]
    inputs, targets = teacher_forced_inputs_and_targets(eval_tokens)
    vocab_size = int(payload["model_args"]["vocab_size"])
    ptq_preset = stage5_default_ptq_preset_name() if args.ptq_preset is None else args.ptq_preset
    resolved_preset = resolve_stage5_ptq_preset(ptq_preset)

    scales = _build_scales_for_preset(payload, calibration_ids, args, resolved_preset.name)
    lm_scale = float(scales.get("lm_head", 1.0))

    fp32_ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    fp32_incr, fp32_full = _fp32_incremental_and_full(fp32_ref, eval_tokens)
    fake_ref = NanoGPTFQReference(
        payload["state_dict"],
        payload["model_args"],
        scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(payload["model_args"], resolved_preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(resolved_preset),
    )
    fake_incr, fake_full = _fake_incremental_and_full(fake_ref, eval_tokens)
    golden_i8 = run_golden_teacher_forced_logits(payload, eval_tokens, scales, ptq_preset=resolved_preset)
    direct_q = [_quantize_logits(row, lm_scale) for row in fp32_incr]

    fp32_ppl = _ppl_from_logits(fp32_incr, targets, vocab_size=vocab_size)
    direct_deq = [row.astype(np.float32) * np.float32(lm_scale) for row in direct_q]
    direct_ppl = _ppl_from_logits(direct_deq, targets, vocab_size=vocab_size)
    fake_deq = [row.astype(np.float32)[:vocab_size] * np.float32(lm_scale) for row in fake_incr]
    golden_deq = [row.astype(np.float32)[:vocab_size] * np.float32(lm_scale) for row in golden_i8]
    fake_ppl = _ppl_from_logits(fake_deq, targets, vocab_size=vocab_size)
    golden_ppl = _ppl_from_logits(golden_deq, targets, vocab_size=vocab_size)

    hf_logits, hf_status = _load_hf_logits(tokenizer_dir, inputs)
    if hf_logits is not None:
        hf_summary = _logit_pair_summary(fp32_full, hf_logits, vocab_size=vocab_size)
        hf_status.update({
            "min_cosine_vs_hf": hf_summary["min_cosine"],
            "mean_cosine_vs_hf": hf_summary["mean_cosine"],
            "min_top10_overlap_vs_hf": hf_summary["min_top10_overlap"],
            "max_p99_abs_error_vs_hf": hf_summary["max_p99_abs_error"],
        })

    report = {
        "checkpoint": str(checkpoint),
        "tokenizer_dir": str(tokenizer_dir),
        "calibration_text": str(args.calibration_text),
        "eval_text": str(args.eval_text),
        "calibration_sha256": file_sha256(args.calibration_text),
        "eval_sha256": file_sha256(args.eval_text),
        "ptq_preset": resolved_preset.name,
        "vocab_size": vocab_size,
        "lm_head_scale": lm_scale,
        "token_count": len(eval_tokens),
        "target_count": len(targets),
        "tokens": [
            {
                "index": idx,
                "id": int(tok),
                "text": _decoded_token(tokenizer, int(tok)),
            }
            for idx, tok in enumerate(eval_tokens)
        ],
        "per_step": _per_step_rows(
            tokenizer=tokenizer,
            tokens=eval_tokens,
            targets=targets,
            vocab_size=vocab_size,
            lm_head_scale=lm_scale,
            golden_i8=golden_i8,
            fake_i8=fake_incr,
            fp32_logits=fp32_incr,
            direct_quant_i8=direct_q,
        ),
        "suspects": {
            "perplexity": {
                "fp32_perplexity": fp32_ppl["perplexity"],
                "fp32_nll": fp32_ppl["mean_nll"],
                "golden_perplexity": golden_ppl["perplexity"],
                "fake_quant_perplexity": fake_ppl["perplexity"],
                "relative_delta": abs(golden_ppl["perplexity"] - fake_ppl["perplexity"]) / max(abs(fake_ppl["perplexity"]), 1e-12),
                "golden_vs_fp32_gap": (golden_ppl["perplexity"] / max(fp32_ppl["perplexity"], 1e-12)) - 1.0,
                "golden_nll": golden_ppl["mean_nll"],
                "fake_quant_nll": fake_ppl["mean_nll"],
            },
            "fp32_baseline": {
                "incremental": {
                    "perplexity": fp32_ppl["perplexity"],
                    "mean_nll": fp32_ppl["mean_nll"],
                },
                "full_prefix_vs_incremental": _logit_pair_summary(fp32_full, fp32_incr, vocab_size=vocab_size),
            },
            "lm_head_quantization": {
                "direct_quant_fp32": {
                    "perplexity": direct_ppl["perplexity"],
                    "mean_nll": direct_ppl["mean_nll"],
                    "vs_fp32": _logit_pair_summary(direct_deq, fp32_incr, vocab_size=vocab_size),
                    "mean_saturation_rate": float(np.mean([_int8_stats(row)["saturation_rate"] for row in direct_q])),
                    "min_unique_count": int(min(_int8_stats(row)["unique_count"] for row in direct_q)),
                },
            },
            "embedding_add_scale": _embedding_add_scale_diagnostics(
                payload,
                inputs,
                active_scale=float(scales.get("tok_pos_add", 6.0 / 127.0)),
                percentile=args.calibration_percentile,
            ),
            "calibration_sensitivity": _calibration_sweep(
                payload,
                calibration_ids=calibration_ids,
                eval_tokens=eval_tokens,
                vocab_size=vocab_size,
                tokenizer_dir=tokenizer_dir,
                args=args,
                ptq_preset=resolved_preset.name,
            ),
            "shared_decode_semantics": {
                "fp32_full_vs_incremental": _logit_pair_summary(fp32_full, fp32_incr, vocab_size=vocab_size),
                "fake_full_vs_incremental": _logit_pair_summary(
                    [row.astype(np.float32) * np.float32(lm_scale) for row in fake_full],
                    fake_deq,
                    vocab_size=vocab_size,
                ),
            },
            "converter_bias_layout": hf_status,
        },
    }
    report["primary_suspect"] = choose_primary_suspect(report)
    trace_step = getattr(args, "trace_step", None)
    if trace_step is not None:
        step = int(trace_step)
        if step < 0 or step >= len(inputs):
            raise ValueError(f"trace-step {step} is outside available input steps 0..{len(inputs) - 1}")
        prefix = inputs[: step + 1]
        target = targets[step] if step < len(targets) else None
        fake_trace = NanoGPTFQReference(
            payload["state_dict"],
            payload["model_args"],
            scales,
            requant_pc_weight_names=stage5_requant_pc_weight_names(payload["model_args"], resolved_preset),
            raw_residual1_blocks=stage5_raw_residual1_blocks(resolved_preset),
        ).incremental_node_trace(prefix)[-1]
        fp32_trace = NanoGPTFP32Reference(
            payload["state_dict"],
            payload["model_args"],
        ).incremental_node_trace(prefix)[-1]
        trace = _trace_report(
            fake_trace=fake_trace,
            fp32_trace=fp32_trace,
            model_args=payload["model_args"],
            target=target,
            vocab_size=vocab_size,
            trace_node=getattr(args, "trace_node", "all"),
            include_first_divergence=bool(getattr(args, "first_divergence", False)),
        )
        trace["step"] = step
        trace["input_token_id"] = int(inputs[step])
        if target is not None:
            trace["target_token_id"] = int(target)
        report["node_trace"] = trace
        report["collapse_summary"] = _collapse_summary(trace["nodes"])
    if bool(getattr(args, "ablation_sweep", False)):
        report["ablation_sweep"] = _ablation_sweep(
            payload=payload,
            scales=scales,
            inputs=inputs,
            targets=targets,
            fp32_logits=fp32_incr,
            golden_i8=golden_i8,
            fake_i8=fake_incr,
            vocab_size=vocab_size,
            lm_scale=lm_scale,
        )
    if bool(getattr(args, "preset_sweep", False)):
        report["preset_sweep"] = _preset_sweep(
            payload=payload,
            calibration_ids=calibration_ids,
            eval_tokens=eval_tokens,
            inputs=inputs,
            targets=targets,
            fp32_logits=fp32_incr,
            vocab_size=vocab_size,
            args=args,
        )
    if "node_trace" in report or "ablation_sweep" in report:
        report["dominant_loss_point"] = _dominant_loss_point(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=33)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--calibration-seq-len", type=int, default=32)
    parser.add_argument("--calibration-n-seqs", type=int, default=8)
    parser.add_argument("--calibration-percentile", type=float, default=99.9)
    parser.add_argument("--ptq-preset", default=None)
    parser.add_argument("--trace-step", type=int)
    parser.add_argument("--trace-node", default="all")
    parser.add_argument("--first-divergence", action="store_true")
    parser.add_argument("--ablation-sweep", action="store_true")
    parser.add_argument("--preset-sweep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_report(args)
    if args.json:
        print(json.dumps(_jsonify(report), indent=2, sort_keys=True))
    else:
        ppl = report["suspects"]["perplexity"]
        fp32_gap = ppl.get("golden_vs_fp32_gap")
        print(json.dumps(_jsonify({
            "primary_suspect": report["primary_suspect"],
            "perplexity": ppl,
            "fp32_gap_diagnostic": {
                "fp32_perplexity": ppl.get("fp32_perplexity"),
                "golden_perplexity": ppl.get("golden_perplexity"),
                "golden_vs_fp32_gap": fp32_gap,
                "golden_vs_fp32_gap_pct": None if fp32_gap is None else round(float(fp32_gap) * 100.0, 2),
            },
            "fp32_baseline": report["suspects"]["fp32_baseline"],
            "lm_head_quantization": report["suspects"]["lm_head_quantization"],
            "converter_bias_layout": report["suspects"]["converter_bias_layout"],
            "ptq_preset": report["ptq_preset"],
            "preset_sweep": report.get("preset_sweep"),
            "dominant_loss_point": report.get("dominant_loss_point"),
            "ablation_sweep": report.get("ablation_sweep"),
            "node_trace": report.get("node_trace"),
        }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
