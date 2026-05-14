"""W8A16 simulator-backed perplexity gates — Phase 3 (c.2) M4-G.

Two end-to-end gates exercising the simulator-backed W8A16 codegen path:

1. **Fast gate**: 33-token / 32-context window, ~1 min.
2. **Slow gate**: 257-token / 256-context window, gated by `PYTEST_SLOW=1`,
   ~14 min at GPT-2 124M scale.

These mirror the W8A32 simulator-backed gates in
`test_weight_only_int8_simulator_perplexity.py` but with
`fp_precision="fp16"` — FP16 ABUF storage, FP32 internal datapath, bias
folded into DEQUANT epilogue.

**Current scope: structural gate, not numerical-correctness gate.**

The fc2 per-K-tile dynamic activation scaling path under fp16 carries
the same compounded-rounding caveat as W8A32 — additionally, FP16
storage rounds intermediates at each tile boundary. The numerical
`relative_delta` between the simulator-backed bundle and the M4-E
like-for-like reference is **logged but not asserted at FP16 ULP**.
The gate passes as long as `relative_delta` is finite.

Diagnostic output: each gate also reports the W8A32 PPL and
`WeightOnlyHostRunner` PPL (FP32-with-INT8-QDQ ceiling) so the FP16
precision-loss can be quantified independently.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")


def _skip_if_no_fixtures() -> None:
    missing = [str(p) for p in (FIXTURE, TOKENIZER_DIR, EVAL_TEXT) if not p.exists()]
    if missing:
        pytest.skip(f"W8A16 fixtures missing: {missing}")


def _diagnostic_host_runner_ppl(payload, eval_ids, vocab_size: int) -> float:
    """`WeightOnlyHostRunner` PPL (FP32-with-INT8-QDQ ceiling). Diagnostic only."""
    from taccel.runtime.gpt2_perplexity import (
        run_weight_only_int8_golden_teacher_forced_logits,
        teacher_forced_inputs_and_targets,
        stable_cross_entropy,
        perplexity_from_nlls,
    )
    _, targets = teacher_forced_inputs_and_targets(eval_ids)
    logits = run_weight_only_int8_golden_teacher_forced_logits(payload, eval_ids)
    nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), target,
                             vocab_size=vocab_size)
        for row, target in zip(logits, targets)
    ]
    ppl, _ = perplexity_from_nlls(nlls)
    return float(ppl)


def _diagnostic_w8a16_simulator_ppl(payload, eval_ids, vocab_size: int) -> float:
    """W8A16 simulator-backed PPL — diagnostic logging only."""
    from taccel.runtime.gpt2_perplexity import (
        run_weight_only_int8_simulator_teacher_forced_logits,
        teacher_forced_inputs_and_targets,
        stable_cross_entropy,
        perplexity_from_nlls,
    )
    _, targets = teacher_forced_inputs_and_targets(eval_ids)
    logits = run_weight_only_int8_simulator_teacher_forced_logits(
        payload, eval_ids,
    )
    nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), target,
                             vocab_size=vocab_size)
        for row, target in zip(logits, targets)
    ]
    ppl, _ = perplexity_from_nlls(nlls)
    return float(ppl)


def test_33_token_simulator_backed_fp16_perplexity():
    """Fast gate: 33-token simulator-backed perplexity under fp_precision='fp16'.

    Builds a W8A16 ProgramBundle (INT8 weights + FP16 activations + bias-fold
    DEQUANT) via `build_stage3_tiny_decoder_bundle(..., fp_precision='fp16')`,
    runs prefill+decode through `HostRunner(logits_dtype=np.float16)`, and
    verifies finite perplexity.
    """
    import torch

    _skip_if_no_fixtures()
    from taccel.runtime.gpt2_perplexity import (
        evaluate_gpt2_perplexity,
        tokenize_text_file,
    )

    payload = torch.load(FIXTURE, map_location="cpu")
    calibration_ids = (
        tokenize_text_file(TOKENIZER_DIR, CALIB_TEXT)
        if CALIB_TEXT.exists() else [0, 1]
    )
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=33)

    result = evaluate_gpt2_perplexity(
        payload,
        calibration_token_ids=calibration_ids,
        eval_token_ids=eval_ids,
        tokenizer_dir=TOKENIZER_DIR,
        max_eval_tokens=33,
        context_len=32,
        calibration_n_seqs=2,
        calibration_seq_len=8,
        ptq_preset="weight_only_int8",
        simulator_backed=True,
        compute_fp32_ceiling=False,
    )

    assert result.ptq_preset == "weight_only_int8"
    assert not math.isnan(result.golden_perplexity)
    assert not math.isnan(result.fake_quant_perplexity)
    assert not math.isnan(result.relative_delta)

    vocab_size = int(payload["model_args"]["vocab_size"])
    diag_host_runner_ppl = _diagnostic_host_runner_ppl(payload, eval_ids, vocab_size)

    print(
        f"\n[M4-G W8A16 33-tok] "
        f"simulator_backed={result.golden_perplexity:.4f}, "
        f"like_for_like={result.fake_quant_perplexity:.4f}, "
        f"WeightOnlyHostRunner={diag_host_runner_ppl:.4f}, "
        f"relative_delta={result.relative_delta:.4e}"
    )
    assert math.isfinite(result.relative_delta), (
        "M4-G W8A16 33-tok relative_delta is not finite — simulator-backed "
        "PPL or M4-E reference PPL was NaN/inf."
    )


def test_257_token_simulator_backed_fp16_perplexity():
    """Slow gate: 257-token simulator-backed perplexity under fp16 (PYTEST_SLOW=1).

    Same contract as the fast gate at the production 256-context window.
    Runs in ~14 min at GPT-2 124M scale.
    """
    if os.environ.get("PYTEST_SLOW") != "1":
        pytest.skip("set PYTEST_SLOW=1 to run the W8A16 simulator-backed slow gate")
    import torch

    _skip_if_no_fixtures()
    if not CALIB_TEXT.exists():
        pytest.skip("calibration text fixture missing")
    from taccel.runtime.gpt2_perplexity import (
        evaluate_gpt2_perplexity,
        file_sha256,
        tokenize_text_file,
    )

    payload = torch.load(FIXTURE, map_location="cpu")
    calibration_ids = tokenize_text_file(TOKENIZER_DIR, CALIB_TEXT)
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=257)

    result = evaluate_gpt2_perplexity(
        payload,
        calibration_token_ids=calibration_ids,
        eval_token_ids=eval_ids,
        tokenizer_dir=TOKENIZER_DIR,
        calibration_sha256=file_sha256(CALIB_TEXT),
        eval_sha256=file_sha256(EVAL_TEXT),
        max_eval_tokens=257,
        context_len=256,
        calibration_n_seqs=2,
        calibration_seq_len=8,
        ptq_preset="weight_only_int8",
        simulator_backed=True,
        compute_fp32_ceiling=False,
    )

    assert result.ptq_preset == "weight_only_int8"
    assert result.target_count == 256
    assert not math.isnan(result.golden_perplexity)
    assert not math.isnan(result.fake_quant_perplexity)

    vocab_size = int(payload["model_args"]["vocab_size"])
    diag_host_runner_ppl = _diagnostic_host_runner_ppl(payload, eval_ids, vocab_size)

    print(
        f"\n[M4-G W8A16 257-tok] "
        f"simulator_backed={result.golden_perplexity:.4f}, "
        f"like_for_like={result.fake_quant_perplexity:.4f}, "
        f"WeightOnlyHostRunner={diag_host_runner_ppl:.4f}, "
        f"relative_delta={result.relative_delta:.4e}"
    )
    assert math.isfinite(result.relative_delta), (
        "M4-G W8A16 257-tok relative_delta is not finite — simulator-backed "
        "PPL or M4-E reference PPL was NaN/inf."
    )
