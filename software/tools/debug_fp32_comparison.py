#!/usr/bin/env python3
"""Diagnose golden INT8 nanoGPT logits against true FP32 inference.

This is a read-only debug tool.  It builds the same ProgramBundle path used by
the trained nanoGPT tests, then compares golden, fake-quant, and FP32 logits
under several prefix modes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.fake_quant import cosine_similarity
from taccel.runtime.fake_quant_reference import NanoGPTFQReference, _fp32_forward
from taccel.runtime.fp32_reference import NanoGPTFP32Reference
from taccel.runtime.tiny_fixture import (
    build_stage3_tiny_decoder_bundle,
    run_tiny_decode_trace,
)


TOOL_PATH = Path(__file__).resolve().parent / "train_tiny_fixture.py"


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonify(v) for v in value)
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _topk_set(logits: Sequence[float], *, vocab_size: int, k: int) -> set[int]:
    active = np.asarray(logits, dtype=np.float32)[:vocab_size]
    k = min(int(k), int(vocab_size))
    if k <= 0 or active.size == 0:
        return set()
    threshold = np.sort(active)[-k]
    return set(np.where(active >= threshold)[0].tolist())


def _argmax_set(logits: Sequence[float], *, vocab_size: int) -> set[int]:
    active = np.asarray(logits, dtype=np.float32)[:vocab_size]
    if active.size == 0:
        return set()
    return set(np.where(active == np.max(active))[0].tolist())


def _centered_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    lhs = np.asarray(a, dtype=np.float32)
    rhs = np.asarray(b, dtype=np.float32)
    return cosine_similarity(lhs - np.mean(lhs), rhs - np.mean(rhs))


def _rank_metrics(lhs: Sequence[float], rhs: Sequence[float], *, vocab_size: int) -> Dict[str, object]:
    lhs_active = np.asarray(lhs, dtype=np.float32)[:vocab_size]
    rhs_active = np.asarray(rhs, dtype=np.float32)[:vocab_size]
    lhs_top5 = _topk_set(lhs_active, vocab_size=vocab_size, k=5)
    rhs_top5 = _topk_set(rhs_active, vocab_size=vocab_size, k=5)
    lhs_top10 = _topk_set(lhs_active, vocab_size=vocab_size, k=10)
    rhs_top10 = _topk_set(rhs_active, vocab_size=vocab_size, k=10)
    lhs_argmax = _argmax_set(lhs_active, vocab_size=vocab_size)
    rhs_argmax = _argmax_set(rhs_active, vocab_size=vocab_size)
    return {
        "top5_overlap": len(lhs_top5 & rhs_top5),
        "top10_overlap": len(lhs_top10 & rhs_top10),
        "lhs_argmax_in_rhs_top10": bool(lhs_argmax & rhs_top10),
        "rhs_argmax_in_lhs_top10": bool(rhs_argmax & lhs_top10),
        "exact_argmax_overlap": bool(lhs_argmax & rhs_argmax),
        "raw_cosine": cosine_similarity(lhs_active, rhs_active),
        "centered_cosine": _centered_cosine(lhs_active, rhs_active),
        "lhs_argmax": sorted(lhs_argmax),
        "rhs_argmax": sorted(rhs_argmax),
        "lhs_top10": sorted(lhs_top10),
        "rhs_top10": sorted(rhs_top10),
    }


def _int8_stats(logits: Sequence[int], *, vocab_size: int) -> Dict[str, object]:
    active = np.asarray(logits, dtype=np.int16)[:vocab_size]
    if active.size == 0:
        return {
            "unique_count": 0,
            "saturation_rate": 0.0,
            "min": 0,
            "max": 0,
        }
    saturated = np.logical_or(active <= -128, active >= 127)
    return {
        "unique_count": int(np.unique(active).size),
        "saturation_rate": float(np.mean(saturated)),
        "min": int(np.min(active)),
        "max": int(np.max(active)),
    }


def _dequant(logits: Sequence[int], scale: float, *, vocab_size: int) -> np.ndarray:
    return np.asarray(logits, dtype=np.int8)[:vocab_size].astype(np.float32) * np.float32(scale)


def _quantize_logits(logits: Sequence[float], scale: float) -> np.ndarray:
    if scale <= 0.0:
        return np.zeros_like(np.asarray(logits), dtype=np.int8)
    return np.clip(np.round(np.asarray(logits, dtype=np.float32) / np.float32(scale)), -128, 127).astype(np.int8)


def _run_fake_free(ref: NanoGPTFQReference, prompt_ids: Sequence[int], *,
                   max_new_tokens: int, vocab_size: int) -> Dict[str, object]:
    generated = [int(tok) for tok in prompt_ids]
    logits_trace: List[np.ndarray] = []
    logits = ref.forward_incremental(generated)
    logits_trace.append(logits)
    next_token = int(np.argmax(logits[:vocab_size]))
    for _ in range(max_new_tokens):
        generated.append(next_token)
        logits = ref.forward_incremental(generated)
        logits_trace.append(logits)
        next_token = int(np.argmax(logits[:vocab_size]))
    return {"generated": generated, "logits": logits_trace}


def _run_golden_forced(tiny, token_ids: Sequence[int]) -> Dict[str, object]:
    if not token_ids:
        raise ValueError("token_ids must be non-empty")
    runner = __import__("taccel.runtime.host_runner", fromlist=["HostRunner"]).HostRunner(
        tiny.build.bundle,
        logits_dtype=np.int8,
    )
    logits_trace = [runner.run_prefill([int(token_ids[0])])]
    for pos, token in enumerate(token_ids[1:], start=1):
        logits_trace.append(runner.run_decode_step(int(token), pos))
    return {"generated": [int(tok) for tok in token_ids], "logits": logits_trace}


def _run_fake_forced(ref: NanoGPTFQReference, token_ids: Sequence[int]) -> Dict[str, object]:
    return {
        "generated": [int(tok) for tok in token_ids],
        "logits": ref.incremental_logits_trace(token_ids),
    }


def _eval_tokens(payload: dict, metadata: dict | None, *, length: int, prompt_id: int | None) -> List[int]:
    text = str(payload.get("text", ""))
    stoi = payload.get("stoi", {})
    if not text or not stoi:
        base = [0] * max(1, length)
    else:
        if metadata is None:
            tool = _fixture_tool()
            ranges = tool.split_ranges(text, calibration_limit=2048)
            start, end = ranges["evaluation_bytes"]
        else:
            span = metadata["ranges"]["evaluation_bytes"]
            start, end = int(span["start"]), int(span["end"])
        eval_text = text.encode("utf-8")[start:end].decode("utf-8")
        base = [int(stoi[ch]) for ch in eval_text if ch in stoi]
    if not base:
        base = [0]
    if prompt_id is not None:
        tokens = [int(prompt_id)]
        rest = [tok for tok in base if tok != int(prompt_id)]
        tokens.extend(rest)
    else:
        tokens = list(base)
    repeats = (length // len(tokens)) + 1
    return (tokens * repeats)[:length]


def _load_metadata_for_fixture(fixture: Path) -> dict | None:
    metadata_path = fixture.with_suffix(fixture.suffix + ".json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return None


def _load_payload(fixture: Path) -> dict:
    if not fixture.exists():
        raise FileNotFoundError(fixture)
    return torch.load(fixture, map_location="cpu")


def _build_traces(payload: dict, *, mode: str, prompt_id: int | None,
                  max_new_tokens: int, calibration_seq_len: int,
                  calibration_percentile: float, metadata: dict | None):
    calibration_scales = build_calibration_scales(
        payload,
        seq_len=calibration_seq_len,
        percentile=calibration_percentile,
    )
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max_new_tokens,
        calibration_scales=calibration_scales,
    )
    vocab_size = int(payload["model_args"]["vocab_size"])
    prompt = [int(prompt_id if prompt_id is not None else 0)]
    fake_ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], calibration_scales)
    fp32_ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])

    if mode == "free_running":
        golden = run_tiny_decode_trace(tiny, prompt, max_new_tokens=max_new_tokens)
        fake = _run_fake_free(fake_ref, prompt, max_new_tokens=max_new_tokens, vocab_size=vocab_size)
        fp32 = fp32_ref.greedy_decode_trace(prompt, max_new_tokens=max_new_tokens)
        return calibration_scales, {
            "golden": {"generated": golden.generated, "logits": golden.logits},
            "fake": fake,
            "fp32": {"generated": fp32.generated, "logits": fp32.logits},
        }

    if mode == "same_prefix_golden":
        golden = run_tiny_decode_trace(tiny, prompt, max_new_tokens=max_new_tokens)
        fake = _run_fake_forced(fake_ref, golden.generated)
        fp32_logits = fp32_ref.incremental_logits_trace(golden.generated)
        return calibration_scales, {
            "golden": {"generated": golden.generated, "logits": golden.logits},
            "fake": fake,
            "fp32": {"generated": list(golden.generated), "logits": fp32_logits},
        }

    if mode == "teacher_forced_eval":
        tokens = _eval_tokens(
            payload,
            metadata,
            length=max_new_tokens + 1,
            prompt_id=prompt_id,
        )
        golden = _run_golden_forced(tiny, tokens)
        fake = _run_fake_forced(fake_ref, tokens)
        fp32_logits = fp32_ref.incremental_logits_trace(tokens)
        return calibration_scales, {
            "golden": golden,
            "fake": fake,
            "fp32": {"generated": list(tokens), "logits": fp32_logits},
        }

    raise ValueError(f"unknown mode: {mode}")


def _pair_summary(lhs_logits: Sequence[np.ndarray], rhs_logits: Sequence[np.ndarray], *,
                  vocab_size: int, lhs_scale: float | None = None) -> Dict[str, object]:
    metrics = []
    for lhs, rhs in zip(lhs_logits, rhs_logits):
        lhs_cmp = _dequant(lhs, lhs_scale, vocab_size=vocab_size) if lhs_scale is not None else np.asarray(lhs)[:vocab_size]
        rhs_cmp = np.asarray(rhs)[:vocab_size]
        metrics.append(_rank_metrics(lhs_cmp, rhs_cmp, vocab_size=vocab_size))
    if not metrics:
        return {
            "steps": 0,
            "min_top10_overlap": 0,
            "lhs_argmax_in_rhs_top10_rate": 0.0,
            "rhs_argmax_in_lhs_top10_rate": 0.0,
            "exact_argmax_match_rate": 0.0,
            "min_raw_cosine": 1.0,
            "mean_raw_cosine": 1.0,
            "min_centered_cosine": 1.0,
            "mean_centered_cosine": 1.0,
        }
    return {
        "steps": len(metrics),
        "min_top10_overlap": int(min(m["top10_overlap"] for m in metrics)),
        "min_top5_overlap": int(min(m["top5_overlap"] for m in metrics)),
        "lhs_argmax_in_rhs_top10_rate": float(np.mean([m["lhs_argmax_in_rhs_top10"] for m in metrics])),
        "rhs_argmax_in_lhs_top10_rate": float(np.mean([m["rhs_argmax_in_lhs_top10"] for m in metrics])),
        "exact_argmax_match_rate": float(np.mean([m["exact_argmax_overlap"] for m in metrics])),
        "min_raw_cosine": float(min(m["raw_cosine"] for m in metrics)),
        "mean_raw_cosine": float(np.mean([m["raw_cosine"] for m in metrics])),
        "min_centered_cosine": float(min(m["centered_cosine"] for m in metrics)),
        "mean_centered_cosine": float(np.mean([m["centered_cosine"] for m in metrics])),
    }


def _step_metrics(traces: dict, *, vocab_size: int, lm_head_scale: float) -> List[Dict[str, object]]:
    out = []
    for step, (golden, fake, fp32) in enumerate(zip(
        traces["golden"]["logits"],
        traces["fake"]["logits"],
        traces["fp32"]["logits"],
    )):
        golden_dequant = _dequant(golden, lm_head_scale, vocab_size=vocab_size)
        fake_dequant = _dequant(fake, lm_head_scale, vocab_size=vocab_size)
        fp32_active = np.asarray(fp32, dtype=np.float32)[:vocab_size]
        out.append({
            "step": step,
            "golden_token": int(traces["golden"]["generated"][step]) if step < len(traces["golden"]["generated"]) else None,
            "fake_token": int(traces["fake"]["generated"][step]) if step < len(traces["fake"]["generated"]) else None,
            "fp32_token": int(traces["fp32"]["generated"][step]) if step < len(traces["fp32"]["generated"]) else None,
            "golden_int8": _int8_stats(golden, vocab_size=vocab_size),
            "golden_vs_fp32": _rank_metrics(golden_dequant, fp32_active, vocab_size=vocab_size),
            "golden_vs_fake": _rank_metrics(golden, fake, vocab_size=vocab_size),
            "fake_vs_fp32": _rank_metrics(fake_dequant, fp32_active, vocab_size=vocab_size),
            "fp32_logit_min": float(np.min(fp32_active)),
            "fp32_logit_max": float(np.max(fp32_active)),
            "lm_head_scale": float(lm_head_scale),
        })
    return out


def _quantization_experiment(logits_by_step: Iterable[np.ndarray], *, vocab_size: int,
                             current_scale: float) -> Dict[str, object]:
    fp32 = [np.asarray(step, dtype=np.float32)[:vocab_size] for step in logits_by_step]
    if not fp32:
        return {}
    flat_abs = np.concatenate([np.abs(step).reshape(-1) for step in fp32])

    def eval_scale(name: str, scale: float, *, centered: bool = False) -> Dict[str, object]:
        q_logits = []
        for step in fp32:
            if centered:
                mean = np.float32(np.mean(step))
                q = _quantize_logits(step - mean, scale)
                deq = q.astype(np.float32) * np.float32(scale) + mean
            else:
                q = _quantize_logits(step, scale)
                deq = q.astype(np.float32) * np.float32(scale)
            q_logits.append((q, deq))
        metrics = [
            _rank_metrics(deq, step, vocab_size=vocab_size)
            for (q, deq), step in zip(q_logits, fp32)
        ]
        saturation = [
            _int8_stats(q, vocab_size=vocab_size)["saturation_rate"]
            for q, _ in q_logits
        ]
        unique_counts = [
            _int8_stats(q, vocab_size=vocab_size)["unique_count"]
            for q, _ in q_logits
        ]
        return {
            "name": name,
            "scale": float(scale),
            "centered": bool(centered),
            "min_top10_overlap": int(min(m["top10_overlap"] for m in metrics)),
            "fp32_argmax_in_quantized_top10_rate": float(np.mean([m["rhs_argmax_in_lhs_top10"] for m in metrics])),
            "quantized_argmax_in_fp32_top10_rate": float(np.mean([m["lhs_argmax_in_rhs_top10"] for m in metrics])),
            "exact_argmax_match_rate": float(np.mean([m["exact_argmax_overlap"] for m in metrics])),
            "mean_saturation_rate": float(np.mean(saturation)),
            "min_unique_count": int(min(unique_counts)),
            "mean_centered_cosine": float(np.mean([m["centered_cosine"] for m in metrics])),
        }

    experiments = {
        "scalar_current": eval_scale("scalar_current", current_scale),
        "centered_current": eval_scale("centered_current", current_scale, centered=True),
    }
    for pct in (99.0, 99.5, 99.9, 100.0):
        scale = max(float(np.percentile(flat_abs, pct)), 1e-8) / 127.0
        experiments[f"scalar_p{pct:g}"] = eval_scale(f"scalar_p{pct:g}", scale)

    oracle_logits = []
    for step in fp32:
        # Diagnostic upper bound: each vocab entry gets its own exact scale.
        deq = step.astype(np.float32).copy()
        q = np.where(step >= 0.0, 127, -127).astype(np.int8)
        q[np.isclose(step, 0.0)] = 0
        oracle_logits.append((q, deq))
    oracle_metrics = [
        _rank_metrics(deq, step, vocab_size=vocab_size)
        for (q, deq), step in zip(oracle_logits, fp32)
    ]
    experiments["per_token_oracle"] = {
        "name": "per_token_oracle",
        "scale": "per-token absmax",
        "centered": False,
        "min_top10_overlap": int(min(m["top10_overlap"] for m in oracle_metrics)),
        "fp32_argmax_in_quantized_top10_rate": float(np.mean([m["rhs_argmax_in_lhs_top10"] for m in oracle_metrics])),
        "quantized_argmax_in_fp32_top10_rate": float(np.mean([m["lhs_argmax_in_rhs_top10"] for m in oracle_metrics])),
        "exact_argmax_match_rate": float(np.mean([m["exact_argmax_overlap"] for m in oracle_metrics])),
        "mean_saturation_rate": 1.0,
        "min_unique_count": int(min(_int8_stats(q, vocab_size=vocab_size)["unique_count"] for q, _ in oracle_logits)),
        "mean_centered_cosine": float(np.mean([m["centered_cosine"] for m in oracle_metrics])),
    }
    return experiments


def _trace_value(entry: object) -> np.ndarray:
    if isinstance(entry, dict) and "value" in entry:
        return np.asarray(entry["value"], dtype=np.float32)
    return np.asarray(entry, dtype=np.float32)


def _trace_int8(entry: object) -> Optional[np.ndarray]:
    if isinstance(entry, dict):
        value = entry.get("int8")
        if value is not None:
            return np.asarray(value, dtype=np.int8)
    return None


def _trace_scale(entry: object) -> Optional[float]:
    if isinstance(entry, dict):
        scale = entry.get("scale")
        if scale is not None:
            return float(scale)
    return None


def _node_pair_metrics(fake_entry: object, fp32_entry: object) -> Dict[str, object]:
    fake = _trace_value(fake_entry).astype(np.float32).reshape(-1)
    fp32 = _trace_value(fp32_entry).astype(np.float32).reshape(-1)
    n = int(min(fake.size, fp32.size))
    if n == 0 or fake.size != fp32.size:
        return {
            "shape_match": bool(fake.size == fp32.size),
            "fake_shape": list(_trace_value(fake_entry).shape),
            "fp32_shape": list(_trace_value(fp32_entry).shape),
            "top10_overlap": 0,
            "centered_cosine": 0.0,
            "raw_cosine": 0.0,
            "p99_abs_error": None,
            "p99_abs_error_lsb": None,
            "fake_int8": _int8_stats(_trace_int8(fake_entry), vocab_size=0)
            if _trace_int8(fake_entry) is not None else None,
        }
    fake = fake[:n]
    fp32 = fp32[:n]
    p99_abs = float(np.percentile(np.abs(fake - fp32), 99.0))
    scale = _trace_scale(fake_entry)
    fake_i8 = _trace_int8(fake_entry)
    stats = None
    if fake_i8 is not None:
        stats = _int8_stats(fake_i8.reshape(-1), vocab_size=fake_i8.size)
    rank = _rank_metrics(fake, fp32, vocab_size=n)
    return {
        "shape_match": True,
        "shape": list(_trace_value(fake_entry).shape),
        "numel": n,
        "scale": scale,
        "p99_abs_error": p99_abs,
        "p99_abs_error_lsb": None if not scale or scale <= 0.0 else float(p99_abs / scale),
        "fake_int8": stats,
        **rank,
    }


def _node_is_divergent(metrics: Dict[str, object]) -> bool:
    if not metrics.get("shape_match", False):
        return True
    if float(metrics.get("centered_cosine", 1.0)) < 0.95:
        return True
    p99_lsb = metrics.get("p99_abs_error_lsb")
    if p99_lsb is not None and float(p99_lsb) > 2.0:
        return True
    if int(metrics.get("numel", 0)) >= 7 and int(metrics.get("top10_overlap", 10)) < 7:
        return True
    stats = metrics.get("fake_int8")
    if isinstance(stats, dict) and float(stats.get("saturation_rate", 0.0)) > 0.05:
        return True
    return False


def _token_prefix_for_step(traces: dict, step: int) -> List[int]:
    generated = [int(tok) for tok in traces["golden"]["generated"]]
    if not generated:
        return [0]
    step = max(0, min(int(step), len(generated) - 1))
    return generated[: step + 1]


def _node_trace_report(payload: dict, traces: dict, calibration_scales: Dict[str, float],
                       *, step: int, trace_node: str) -> Dict[str, object]:
    tokens = _token_prefix_for_step(traces, step)
    fake_ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], calibration_scales)
    fp32_ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    fake_steps = fake_ref.incremental_node_trace(tokens)
    fp32_steps = fp32_ref.incremental_node_trace(tokens)
    selected_step = len(tokens) - 1
    fake_nodes = fake_steps[selected_step]
    fp32_nodes = fp32_steps[selected_step]

    if trace_node == "all":
        names = [name for name in fake_nodes if name in fp32_nodes]
    else:
        names = [trace_node] if trace_node in fake_nodes and trace_node in fp32_nodes else []

    node_metrics = {
        name: _node_pair_metrics(fake_nodes[name], fp32_nodes[name])
        for name in names
    }
    first = None
    for name in [name for name in fake_nodes if name in fp32_nodes]:
        metrics = _node_pair_metrics(fake_nodes[name], fp32_nodes[name])
        if _node_is_divergent(metrics):
            first = {"node": name, "metrics": metrics}
            break
    return {
        "step": int(selected_step),
        "prefix_tokens": tokens,
        "node_metrics": node_metrics,
        "first_divergence": first,
    }


def _trace_lm_head(payload: dict, traces: dict, *, vocab_size: int, lm_head_scale: float) -> Dict[str, object]:
    failing_step = 0
    for step, (golden, fp32) in enumerate(zip(traces["golden"]["logits"], traces["fp32"]["logits"])):
        golden_dequant = _dequant(golden, lm_head_scale, vocab_size=vocab_size)
        metrics = _rank_metrics(golden_dequant, fp32, vocab_size=vocab_size)
        if not metrics["rhs_argmax_in_lhs_top10"] or not metrics["lhs_argmax_in_rhs_top10"]:
            failing_step = step
            break
    prefix = traces["golden"]["generated"][: failing_step + 1]
    node_outputs = _fp32_forward(payload["state_dict"], payload["model_args"], prefix)
    n_layer = int(payload["model_args"]["n_layer"])
    final_hidden = node_outputs[f"block{n_layer - 1}_residual2"][-1]
    ln_f = node_outputs["ln_f"][-1]
    lm_head = node_outputs["lm_head"][0][:vocab_size]
    q_current = _quantize_logits(lm_head, lm_head_scale)
    deq_current = q_current.astype(np.float32) * np.float32(lm_head_scale)
    centered_mean = np.float32(np.mean(lm_head))
    q_centered = _quantize_logits(lm_head - centered_mean, lm_head_scale)
    deq_centered = q_centered.astype(np.float32) * np.float32(lm_head_scale) + centered_mean
    return {
        "step": int(failing_step),
        "prefix_tokens": [int(tok) for tok in prefix],
        "final_hidden_before_ln_f": {
            "shape": list(np.asarray(final_hidden).shape),
            "min": float(np.min(final_hidden)),
            "max": float(np.max(final_hidden)),
            "mean": float(np.mean(final_hidden)),
            "std": float(np.std(final_hidden)),
        },
        "ln_f": {
            "shape": list(np.asarray(ln_f).shape),
            "min": float(np.min(ln_f)),
            "max": float(np.max(ln_f)),
            "mean": float(np.mean(ln_f)),
            "std": float(np.std(ln_f)),
        },
        "lm_head_fp32_pre_quant": {
            "min": float(np.min(lm_head)),
            "max": float(np.max(lm_head)),
            "mean": float(np.mean(lm_head)),
            "std": float(np.std(lm_head)),
            "top10": sorted(_topk_set(lm_head, vocab_size=vocab_size, k=10)),
        },
        "lm_head_current_int8": {
            **_int8_stats(q_current, vocab_size=vocab_size),
            "top10": sorted(_topk_set(q_current, vocab_size=vocab_size, k=10)),
        },
        "lm_head_current_dequant": {
            "top10": sorted(_topk_set(deq_current, vocab_size=vocab_size, k=10)),
            "rank_metrics_vs_fp32": _rank_metrics(deq_current, lm_head, vocab_size=vocab_size),
        },
        "lm_head_centered_dequant": {
            "top10": sorted(_topk_set(deq_centered, vocab_size=vocab_size, k=10)),
            "rank_metrics_vs_fp32": _rank_metrics(deq_centered, lm_head, vocab_size=vocab_size),
        },
    }


def _compare_logits_to_fp32(logits_by_step: Sequence[np.ndarray], fp32_logits: Sequence[np.ndarray], *,
                            vocab_size: int, lm_head_scale: float) -> Dict[str, object]:
    metrics = []
    saturation = []
    unique_counts = []
    for lhs, rhs in zip(logits_by_step, fp32_logits):
        lhs_arr = np.asarray(lhs)
        if np.issubdtype(lhs_arr.dtype, np.integer):
            lhs_cmp = _dequant(lhs_arr, lm_head_scale, vocab_size=vocab_size)
            stats = _int8_stats(lhs_arr, vocab_size=vocab_size)
            saturation.append(stats["saturation_rate"])
            unique_counts.append(stats["unique_count"])
        else:
            lhs_cmp = lhs_arr.astype(np.float32)[:vocab_size]
        metrics.append(_rank_metrics(lhs_cmp, np.asarray(rhs, dtype=np.float32)[:vocab_size], vocab_size=vocab_size))
    if not metrics:
        return {"steps": 0}
    return {
        "steps": len(metrics),
        "min_top10_overlap": int(min(m["top10_overlap"] for m in metrics)),
        "fp32_argmax_in_top10_rate": float(np.mean([m["rhs_argmax_in_lhs_top10"] for m in metrics])),
        "fake_argmax_in_fp32_top10_rate": float(np.mean([m["lhs_argmax_in_rhs_top10"] for m in metrics])),
        "exact_argmax_match_rate": float(np.mean([m["exact_argmax_overlap"] for m in metrics])),
        "mean_centered_cosine": float(np.mean([m["centered_cosine"] for m in metrics])),
        "min_centered_cosine": float(min(m["centered_cosine"] for m in metrics)),
        "mean_saturation_rate": None if not saturation else float(np.mean(saturation)),
        "min_unique_count": None if not unique_counts else int(min(unique_counts)),
    }


def _ablation_experiments(payload: dict, traces: dict, calibration_scales: Dict[str, float],
                          *, vocab_size: int, lm_head_scale: float) -> Dict[str, object]:
    tokens = [int(tok) for tok in traces["golden"]["generated"]]
    groups = ("embeddings", "qkv", "softmax", "attn_v", "out_proj", "residual_vadd", "mlp", "ln_f", "lm_head")
    out = {}
    for group in groups:
        ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], calibration_scales)
        logits = ref.incremental_logits_trace(tokens, fp32_groups={group})
        out[group] = _compare_logits_to_fp32(
            logits,
            traces["fp32"]["logits"],
            vocab_size=vocab_size,
            lm_head_scale=lm_head_scale,
        )
    return out


def _targeted_override_sweep(payload: dict, traces: dict, calibration_scales: Dict[str, float],
                             *, vocab_size: int, lm_head_scale: float) -> List[Dict[str, object]]:
    tokens = [int(tok) for tok in traces["golden"]["generated"]]
    targets = {
        "residual": lambda key: "residual" in key,
        "fc": lambda key: key.endswith("_fc1") or key.endswith("_fc2"),
        "gelu": lambda key: key.endswith("_gelu"),
        "ln_f": lambda key: key == "ln_f",
        "lm_head": lambda key: key == "lm_head",
    }
    rows = []
    for target, predicate in targets.items():
        for factor in (0.5, 0.75, 1.25, 1.5, 2.0):
            scales = dict(calibration_scales)
            changed = 0
            for key, value in list(scales.items()):
                if predicate(key):
                    scales[key] = float(value) * factor
                    changed += 1
            ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], scales)
            logits = ref.incremental_logits_trace(tokens)
            summary = _compare_logits_to_fp32(
                logits,
                traces["fp32"]["logits"],
                vocab_size=vocab_size,
                lm_head_scale=float(scales.get("lm_head", lm_head_scale)),
            )
            rows.append({
                "target": target,
                "factor": float(factor),
                "changed_scales": changed,
                **summary,
            })
    return rows


def _calibration_sweep(payload: dict, args, metadata: dict | None) -> List[Dict[str, object]]:
    rows = []
    for seq_len in (16, 32, 64, 128):
        for percentile in (99.0, 99.5, 99.9, 100.0):
            scales, traces = _build_traces(
                payload,
                mode=args.mode,
                prompt_id=args.prompt_id,
                max_new_tokens=args.max_new_tokens,
                calibration_seq_len=seq_len,
                calibration_percentile=percentile,
                metadata=metadata,
            )
            vocab_size = int(payload["model_args"]["vocab_size"])
            lm_head_scale = float(scales.get("lm_head", 1.0))
            bridge = {
                "golden_vs_fake": _pair_summary(
                    traces["golden"]["logits"],
                    traces["fake"]["logits"],
                    vocab_size=vocab_size,
                ),
                "golden_vs_fp32": _pair_summary(
                    traces["golden"]["logits"],
                    traces["fp32"]["logits"],
                    vocab_size=vocab_size,
                    lhs_scale=lm_head_scale,
                ),
            }
            step_metrics = _step_metrics(traces, vocab_size=vocab_size, lm_head_scale=lm_head_scale)
            rows.append({
                "seq_len": int(seq_len),
                "percentile": float(percentile),
                "lm_head_scale": lm_head_scale,
                "golden_vs_fake_min_top5": bridge["golden_vs_fake"]["min_top5_overlap"],
                "golden_vs_fake_mean_cosine": bridge["golden_vs_fake"]["mean_raw_cosine"],
                "golden_vs_fp32_min_top10": bridge["golden_vs_fp32"]["min_top10_overlap"],
                "golden_argmax_in_fp32_top10_rate": bridge["golden_vs_fp32"]["lhs_argmax_in_rhs_top10_rate"],
                "fp32_argmax_in_golden_top10_rate": bridge["golden_vs_fp32"]["rhs_argmax_in_lhs_top10_rate"],
                "mean_saturation_rate": float(np.mean([
                    step["golden_int8"]["saturation_rate"] for step in step_metrics
                ])),
                "min_unique_count": int(min(step["golden_int8"]["unique_count"] for step in step_metrics)),
            })
    return rows


def run_debug(args) -> Dict[str, object]:
    fixture = Path(args.fixture)
    payload = _load_payload(fixture)
    metadata = _load_metadata_for_fixture(fixture)
    calibration_scales, traces = _build_traces(
        payload,
        mode=args.mode,
        prompt_id=args.prompt_id,
        max_new_tokens=args.max_new_tokens,
        calibration_seq_len=args.calibration_seq_len,
        calibration_percentile=args.calibration_percentile,
        metadata=metadata,
    )
    vocab_size = int(payload["model_args"]["vocab_size"])
    lm_head_scale = float(calibration_scales.get("lm_head", 1.0))
    step_metrics = _step_metrics(traces, vocab_size=vocab_size, lm_head_scale=lm_head_scale)
    bridge = {
        "golden_vs_fake": _pair_summary(
            traces["golden"]["logits"],
            traces["fake"]["logits"],
            vocab_size=vocab_size,
        ),
        "golden_vs_fp32": _pair_summary(
            traces["golden"]["logits"],
            traces["fp32"]["logits"],
            vocab_size=vocab_size,
            lhs_scale=lm_head_scale,
        ),
        "fake_vs_fp32": _pair_summary(
            traces["fake"]["logits"],
            traces["fp32"]["logits"],
            vocab_size=vocab_size,
            lhs_scale=lm_head_scale,
        ),
    }
    out = {
        "fixture": str(fixture),
        "mode": args.mode,
        "prompt_id": args.prompt_id,
        "max_new_tokens": int(args.max_new_tokens),
        "vocab_size": vocab_size,
        "lm_head_scale": lm_head_scale,
        "calibration": {
            "seq_len": int(args.calibration_seq_len),
            "percentile": float(args.calibration_percentile),
        },
        "generated": {
            "golden": [int(tok) for tok in traces["golden"]["generated"]],
            "fake": [int(tok) for tok in traces["fake"]["generated"]],
            "fp32": [int(tok) for tok in traces["fp32"]["generated"]],
        },
        "summary": bridge["golden_vs_fp32"],
        "bridge_summary": bridge,
        "steps": step_metrics,
        "logit_quantization_experiments": _quantization_experiment(
            traces["fp32"]["logits"],
            vocab_size=vocab_size,
            current_scale=lm_head_scale,
        ),
        "ablation_experiments": _ablation_experiments(
            payload,
            traces,
            calibration_scales,
            vocab_size=vocab_size,
            lm_head_scale=lm_head_scale,
        ),
    }
    if args.trace_node != "none":
        trace_report = _node_trace_report(
            payload,
            traces,
            calibration_scales,
            step=args.step if args.step is not None else 0,
            trace_node=args.trace_node,
        )
        out["trace"] = trace_report
        if args.trace_node == "lm_head":
            out["trace"]["lm_head"] = _trace_lm_head(
                payload,
                traces,
                vocab_size=vocab_size,
                lm_head_scale=lm_head_scale,
            )
    if args.calibration_sweep:
        out["calibration_sweep"] = _calibration_sweep(payload, args, metadata)
        out["targeted_scale_override_sweep"] = _targeted_override_sweep(
            payload,
            traces,
            calibration_scales,
            vocab_size=vocab_size,
            lm_head_scale=lm_head_scale,
        )
    return out


def main(argv=None) -> int:
    tool = _fixture_tool()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=tool.DEFAULT_TRAINED_D128_FIXTURE)
    parser.add_argument(
        "--mode",
        choices=("free_running", "same_prefix_golden", "teacher_forced_eval"),
        default="free_running",
    )
    parser.add_argument("--prompt-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--node", "--trace-node", dest="trace_node", default="none")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--first-divergence", action="store_true")
    parser.add_argument("--calibration-sweep", action="store_true")
    parser.add_argument("--calibration-seq-len", type=int, default=16)
    parser.add_argument("--calibration-percentile", type=float, default=99.9)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_debug(args)
    if args.json:
        print(json.dumps(_jsonify(result), indent=2, sort_keys=True))
    else:
        print(json.dumps(_jsonify({
            "mode": result["mode"],
            "generated": result["generated"],
            "summary": result["summary"],
            "bridge_summary": result["bridge_summary"],
        }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
