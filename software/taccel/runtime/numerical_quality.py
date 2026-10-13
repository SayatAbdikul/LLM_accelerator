"""Fail-closed, teacher-forced W8A16 reference and FP32 quality checks.

Reference agreement uses the existing 2% PPL, 0.995 cosine and 4-of-5
ranking contracts. The separate FP32-relative budget is supplied explicitly
by the caller; it is never inferred from a measured discrepancy.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .gpt2_perplexity import perplexity_from_nlls, stable_cross_entropy


def _checked_rows(name, logits, count: int, vocab_size: int):
    rows = list(logits)
    if len(rows) != count:
        raise ValueError(f"{name}: expected {count} scored rows, got {len(rows)}")
    checked = []
    for step, row in enumerate(rows):
        values = np.asarray(row)
        # Validate before conversion: casting a complex row to FP32 silently
        # discards its imaginary component, and object/string/bool arrays can
        # otherwise be coerced into apparently valid numerical evidence.
        if values.dtype.kind not in "fiu":
            raise ValueError(f"{name} step {step}: invalid logits dtype {values.dtype}")
        if values.ndim != 1 or values.size < vocab_size:
            raise ValueError(f"{name} step {step}: invalid live-vocabulary row {values.shape}")
        values = values[:vocab_size].astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} step {step}: nonfinite live logits")
        checked.append(values)
    return checked


def teacher_forced_quality_report(
    *,
    token_ids: Sequence[int],
    vocab_size: int,
    compiled_logits,
    reference_logits,
    fp32_logits,
    max_fp32_ppl_increase: float,
) -> dict:
    """Check identical teacher-forced prefixes and return an explicit verdict.

    ``max_fp32_ppl_increase`` is a fraction (0.10 means at most 10%). All
    live rows on all three rails must exist and be finite. Padding columns
    are deliberately excluded. A metric failure produces ``passed=False``;
    malformed/missing data raises instead of being silently skipped.
    """
    if (isinstance(vocab_size, bool) or not isinstance(vocab_size, int)
            or vocab_size < 1):
        raise ValueError("vocab_size must be a positive integer")
    tokens = list(token_ids)
    if len(tokens) < 2:
        raise ValueError("quality evaluation requires at least two tokens")
    if any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
           or not 0 <= int(t) < vocab_size for t in tokens):
        raise ValueError("all token IDs must be integers inside the live vocabulary")
    if isinstance(max_fp32_ppl_increase, (bool, np.bool_)):
        raise ValueError("FP32-relative budget must not be boolean")
    budget = float(max_fp32_ppl_increase)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("FP32-relative budget must be finite and nonnegative")
    targets = [int(t) for t in tokens[1:]]
    count = len(targets)
    rails = {
        "compiled": _checked_rows("compiled", compiled_logits, count, vocab_size),
        "reference": _checked_rows("reference", reference_logits, count, vocab_size),
        "fp32": _checked_rows("fp32", fp32_logits, count, vocab_size),
    }
    model_metrics, all_nlls = {}, {}
    for name, rows in rails.items():
        nlls = [stable_cross_entropy(row, targets[i], vocab_size=vocab_size)
                for i, row in enumerate(rows)]
        if not np.isfinite(nlls).all():
            raise ValueError(f"{name}: nonfinite per-token NLL")
        try:
            ppl, mean_nll = perplexity_from_nlls(nlls)
        except OverflowError as exc:
            raise ValueError(f"{name}: perplexity overflow") from exc
        if not math.isfinite(ppl) or ppl <= 0 or not math.isfinite(mean_nll):
            raise ValueError(f"{name}: invalid perplexity")
        model_metrics[name] = {"perplexity": ppl, "mean_nll": mean_nll}
        all_nlls[name] = nlls
    cosines, overlaps = [], []
    k = min(5, vocab_size)
    min_overlap = min(4, k)
    for actual, expected in zip(rails["compiled"], rails["reference"]):
        # FP64 metric arithmetic avoids norm overflow without altering logits.
        a, b = actual.astype(np.float64), expected.astype(np.float64)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = (float(np.dot(a, b) / denominator) if denominator
                  else (1.0 if np.array_equal(a, b) else 0.0))
        if not math.isfinite(cosine):
            raise ValueError("nonfinite reference cosine")
        cosines.append(min(1.0, max(-1.0, cosine)))
        # Stable token-ID ordering makes quantized ties deterministic.
        top_a = set(np.argsort(-a, kind="stable")[:k].tolist())
        top_b = set(np.argsort(-b, kind="stable")[:k].tolist())
        overlaps.append(len(top_a & top_b))
    actual_ppl = model_metrics["compiled"]["perplexity"]
    reference_ppl = model_metrics["reference"]["perplexity"]
    fp32_ppl = model_metrics["fp32"]["perplexity"]
    reference_delta = abs(actual_ppl - reference_ppl) / reference_ppl
    fp32_increase = actual_ppl / fp32_ppl - 1.0
    if not math.isfinite(reference_delta) or not math.isfinite(fp32_increase):
        raise ValueError("nonfinite perplexity comparison")
    violations = []
    if reference_delta > 0.02:
        violations.append("reference_perplexity")
    if min(cosines) < 0.995:
        violations.append("reference_cosine")
    if min(overlaps) < min_overlap:
        violations.append("reference_top5_overlap")
    if fp32_increase > budget:
        violations.append("fp32_perplexity")
    return {
        "passed": not violations,
        "violations": violations,
        "token_count": len(tokens),
        "target_count": count,
        "vocab_size": vocab_size,
        "thresholds": {"max_reference_ppl_relative_delta": 0.02,
                       "min_reference_cosine": 0.995,
                       "min_reference_top5_overlap": min_overlap,
                       "max_fp32_ppl_increase": budget},
        "models": model_metrics,
        "reference_ppl_relative_delta": reference_delta,
        "fp32_ppl_increase": fp32_increase,
        "min_reference_cosine": min(cosines),
        "min_reference_top5_overlap": min(overlaps),
        "per_step": [{"target": targets[i], "cosine": cosines[i],
                      "top5_overlap": overlaps[i],
                      "nll": {name: values[i] for name, values in all_nlls.items()}}
                     for i in range(count)],
    }
