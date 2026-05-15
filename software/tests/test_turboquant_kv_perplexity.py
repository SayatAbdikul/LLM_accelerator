"""Level-3 (end-to-end PPL) + Level-4 (per-position non-compounding)
verification of TurboQuant KV-cache quantization.

Anchored to the **weight_only_int8_quarot reference** baseline (kv_quant=None)
— the near-FP32 winner, NOT the 53.42 FP32 ceiling. Single-config success is
structural (anchor reproduced, finite); "is there a quality-neutral config"
is the sweep's job.

L4 — the distinctive deep check. TurboQuant is per-vector / data-oblivious,
so the KV-quant perturbation per position must NOT systematically *grow*
with sequence position (that would mean cache error compounds with context
— an instability red flag). Measured on the **absolute** per-position NLL
perturbation |Δnll| (signed Δ oscillates around ~0 for a near-neutral
config and makes a signed-trend metric meaningless / vacuously passable).

Reference-only (skips the compiled golden bundle); `prepare()` once per base
preset, `ppl_for()` per kv_quant config.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from taccel.quantizer.turboquant import TurboQuantKV
from taccel.runtime._turboquant_eval import prepare, ppl_for

FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")
SEED = 20260515
D_HEAD = 64  # GPT-2 124M


def _skip_if_no_fixtures():
    miss = [str(p) for p in (FIXTURE, TOKENIZER_DIR, EVAL_TEXT) if not p.exists()]
    if miss:
        pytest.skip(f"fixtures missing: {miss}")
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch unavailable")


def _abs_growth(delta_nll: np.ndarray) -> tuple[float, float]:
    """Non-compounding metrics on |Δnll| (robust to sign oscillation and
    near-zero neutral configs). Returns:
      norm_slope = (slope·n) / mean|Δ|  — total magnitude trend vs its scale
      half_ratio = mean|Δ|[2nd half] / mean|Δ|[1st half]  — growth across seq
    Both ≈ flat (slope→0, ratio→1) when error does not compound."""
    a = np.abs(np.asarray(delta_nll, dtype=np.float64))
    n = len(a)
    if n < 4 or float(np.mean(a)) < 1e-9:
        return 0.0, 1.0  # no perturbation → trivially non-compounding
    t = np.arange(n, dtype=np.float64)
    slope = np.polyfit(t, a, 1)[0]
    norm_slope = float(slope * n / (np.mean(a) + 1e-12))
    h = n // 2
    half_ratio = float((np.mean(a[h:]) + 1e-12) / (np.mean(a[:h]) + 1e-12))
    return norm_slope, half_ratio


def _run(max_tokens: int):
    prep = prepare(
        FIXTURE, TOKENIZER_DIR, EVAL_TEXT, max_tokens=max_tokens,
        ptq_preset="weight_only_int8_quarot", calibration_text=CALIB_TEXT,
    )
    anchor = ppl_for(prep, kv_quant=None)
    anchor2 = ppl_for(prep, kv_quant=None)
    tq = TurboQuantKV(d=D_HEAD, bits=4.0, variant="mse", target="kv", seed=SEED)
    test = ppl_for(prep, kv_quant=tq)

    da = np.asarray(test.nll_per_position) - np.asarray(anchor.nll_per_position)
    norm_slope, half_ratio = _abs_growth(da)
    rel = (test.perplexity - anchor.perplexity) / anchor.perplexity
    print(
        f"\n[TurboQuant-KV {max_tokens}-tok] anchor(quarot,kv=None)="
        f"{anchor.perplexity:.4f}  {tq}={test.perplexity:.4f}  "
        f"Δ={100*rel:+.1f}%  | L4 |Δ|norm_slope={norm_slope:.3f} "
        f"half_ratio={half_ratio:.3f}  mean|Δnll|={np.mean(np.abs(da)):.4f}"
    )
    # L3 structural
    assert np.isfinite(anchor.perplexity) and anchor.perplexity > 1.0
    assert np.isfinite(test.perplexity)
    assert anchor.preset == "weight_only_int8_quarot"
    # kv_quant=None is a true no-op (deterministic, baseline-preserving)
    assert anchor.perplexity == anchor2.perplexity
    # L4 hard assert: |Δ| must not blow up with position (non-compounding).
    assert norm_slope < 1.5, f"per-position |Δ| compounds: norm_slope={norm_slope}"
    assert half_ratio < 3.0, f"2nd-half |Δ| >> 1st-half: {half_ratio}"


def test_turboquant_kv_33_token():
    _skip_if_no_fixtures()
    _run(33)


def test_turboquant_kv_257_token():
    if os.environ.get("PYTEST_SLOW") != "1":
        pytest.skip("set PYTEST_SLOW=1 for the 257-tok TurboQuant-KV gate")
    _skip_if_no_fixtures()
    _run(257)
