"""W8A32 (Phase 1, plan `w8a32-weights-only-plan.md`) evaluation gate.

Covers:

1. `weight_only_int8` preset construction rejects every W8A8 transform
   (rotation, AWQ, BC, output-aware searches, per-channel requant) at
   `_preset(...)` time. No silent compositions.
2. `evaluate_gpt2_perplexity(..., ptq_preset="weight_only_int8")` returns
   a `GPT2PerplexityResult` with `golden_perplexity` / `relative_delta`
   as NaN and `fake_quant_perplexity` ≤ 1 PPL above the FP32 ceiling.
3. The new pipeline path produces **bit-identical** logits to the
   diagnostic at `software/tools/diagnose_weight_only_qdq_ceiling.py`
   for the same fixture and eval window. A floating-point tolerance
   would let a real divergence (e.g. lm_head padding leaking through)
   silently pass; bit-identity catches it.

Slow-gate at the 257-tok / 256-ctx production window is gated by
`PYTEST_SLOW=1`; the fast path keeps to a small eval window so the
test fits inside CI budgets.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from taccel.runtime.fp32_reference import build_weight_only_int8_reference
from taccel.runtime.gpt2_perplexity import (
    evaluate_gpt2_perplexity,
    file_sha256,
    run_weight_only_int8_golden_teacher_forced_logits,
    run_weight_only_int8_teacher_forced_logits,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    STAGE5_PTQ_PRESETS,
    _preset,
    resolve_stage5_ptq_preset,
)
from taccel.runtime.weight_only_host_runner import WeightOnlyHostRunner


FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")


# ---------------------------------------------------------------------------
# Preset construction validation
# ---------------------------------------------------------------------------

def test_weight_only_int8_preset_registered():
    assert "weight_only_int8" in STAGE5_PTQ_PRESETS
    preset = resolve_stage5_ptq_preset("weight_only_int8")
    assert preset.weight_only_int8 is True
    # And every W8A8 field is empty / disabled.
    assert preset.bias_correction_blocks == ()
    assert preset.requant_pc_out_proj_blocks == ()
    assert preset.requant_pc_fc1_blocks == ()
    assert preset.requant_pc_fc2_blocks == ()
    assert preset.output_aware_mlp_blocks == ()
    assert preset.output_aware_attn_blocks == ()
    assert preset.output_aware_lm_head is False
    assert preset.quarot_enabled is False
    assert preset.awq_enabled is False


@pytest.mark.parametrize(
    "kwarg,value",
    [
        ("bias_correction_blocks", (0, 1)),
        ("requant_pc_out_proj_blocks", (0,)),
        ("requant_pc_fc1_blocks", (0,)),
        ("requant_pc_fc2_blocks", (0,)),
        ("output_aware_mlp_blocks", (0,)),
        ("output_aware_attn_blocks", (0,)),
        ("output_aware_gelu_blocks", (0,)),
        ("output_aware_lm_head", True),
        ("hessian_gelu_blocks", (0,)),
        ("fc2_aware_gelu_blocks", (0,)),
        ("gelu_from_accum_blocks", (0,)),
        ("quarot_enabled", True),
        ("awq_enabled", True),
        ("activation_percentile_nodes", {"ln_f": 99.8}),
    ],
)
def test_weight_only_int8_rejects_w8a8_field(kwarg, value):
    """Every W8A8 transform must be rejected at preset-construction time
    when paired with weight_only_int8=True. The W8A32 path has no
    calibration / no per-tensor activation scales for those transforms to
    operate on, and silently ignoring them would mask configuration bugs.
    """
    with pytest.raises(ValueError, match="weight_only_int8=True is incompatible"):
        _preset("bad_combo", weight_only_int8=True, **{kwarg: value})


# ---------------------------------------------------------------------------
# End-to-end W8A32 evaluation
# ---------------------------------------------------------------------------

def _skip_if_no_fixtures():
    missing = [str(p) for p in (FIXTURE, TOKENIZER_DIR, EVAL_TEXT) if not p.exists()]
    if missing:
        pytest.skip(f"W8A32 fixtures missing: {missing}")


def test_evaluate_gpt2_perplexity_weight_only_int8_short_window():
    """Fast path: a 33-tok / 32-ctx eval finishes in seconds and exercises
    the full pipeline. We allow generous PPL headroom here because the
    short eval window is noisier than the 257-tok slow gate.

    Phase 3 option (b) addition: `golden_perplexity` is now populated
    by `WeightOnlyHostRunner` (instead of the Phase 1 NaN), and
    `relative_delta` is ~0 because both paths wrap the same QDQ helper.
    """
    _skip_if_no_fixtures()

    payload = torch.load(FIXTURE, map_location="cpu")
    calibration_ids = tokenize_text_file(TOKENIZER_DIR, CALIB_TEXT) if CALIB_TEXT.exists() else [0, 1]
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=33)
    result = evaluate_gpt2_perplexity(
        payload,
        calibration_token_ids=calibration_ids,
        eval_token_ids=eval_ids,
        tokenizer_dir=TOKENIZER_DIR,
        max_eval_tokens=33,
        context_len=32,
        # Calibration knobs are irrelevant for W8A32 but the API still
        # requires positional values; pass small numbers.
        calibration_n_seqs=2,
        calibration_seq_len=8,
        ptq_preset="weight_only_int8",
    )
    assert result.ptq_preset == "weight_only_int8"
    # Phase 3 option (b): golden / golden_nll / relative_delta are now
    # real numbers populated by WeightOnlyHostRunner.
    assert not math.isnan(result.golden_perplexity)
    assert not math.isnan(result.golden_nll)
    assert not math.isnan(result.relative_delta)
    assert not math.isnan(result.fake_quant_perplexity)
    # Both paths wrap the same numpy reference, so the contract is
    # tight: golden vs fake_quant must agree within 1e-9.
    assert result.relative_delta < 1e-9, (
        f"W8A32 golden vs fake_quant divergence at short window: "
        f"golden={result.golden_perplexity:.6f} "
        f"fake_quant={result.fake_quant_perplexity:.6f} "
        f"relative_delta={result.relative_delta:.6e}"
    )
    # FP32 ceiling captured pre-mutation; sanity-bound the W8A32 PPL to
    # at most 1.5× the FP32 ceiling.
    assert not math.isnan(result.fp32_perplexity)
    assert result.fake_quant_perplexity <= 1.5 * result.fp32_perplexity, (
        f"W8A32 fake_quant_ppl={result.fake_quant_perplexity:.4f} exceeded "
        f"1.5x fp32_ppl={result.fp32_perplexity:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase 3 option (b) — WeightOnlyHostRunner contract
# ---------------------------------------------------------------------------

def test_weight_only_host_runner_matches_reference_bit_identical():
    """Phase 3 (b) contract: the host runner produces bit-identical
    logits to the Phase 1 numpy reference helper.

    Both paths wrap `build_weight_only_int8_reference`; the runner just
    interposes a `HostRunner`-style prefill/decode API. There must be
    no numerical drift between the two paths or the W8A32 golden gate
    would be unreliable.
    """
    _skip_if_no_fixtures()

    payload = torch.load(FIXTURE, map_location="cpu")
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=33)
    eval_tokens = [int(tok) for tok in eval_ids[:33]]
    inputs, _ = teacher_forced_inputs_and_targets(eval_tokens)

    reference_logits = run_weight_only_int8_teacher_forced_logits(payload, eval_tokens)
    runner_logits = run_weight_only_int8_golden_teacher_forced_logits(payload, eval_tokens)

    assert len(reference_logits) == len(runner_logits)
    for idx, (a, b) in enumerate(zip(reference_logits, runner_logits)):
        np.testing.assert_array_equal(
            a,
            b,
            err_msg=(
                f"WeightOnlyHostRunner vs reference divergence at step {idx}: "
                f"max |Δ|={float(np.max(np.abs(a - b))):.6g}"
            ),
        )


def test_weight_only_host_runner_prefill_decode_consistency():
    """Prefill of N tokens + decode_step(N+1) must equal teacher-forced
    of (N+1) tokens at the same step. This is the contract HostRunner
    has for the deployed bundle; the W8A32 runner must satisfy it too.
    """
    _skip_if_no_fixtures()

    payload = torch.load(FIXTURE, map_location="cpu")
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=9)
    toks = [int(t) for t in eval_ids[:9]]

    # Reference: teacher-force all 9 tokens; the last position's logits
    # are what a decode step after prefill on the first 8 should produce.
    runner_a = WeightOnlyHostRunner(payload)
    teacher = runner_a.run_teacher_forced(toks)

    # Test: prefill on toks[:8], then decode_step(toks[8], 8).
    runner_b = WeightOnlyHostRunner(payload)
    _ = runner_b.run_prefill(toks[:8])
    decoded = runner_b.run_decode_step(int(toks[8]), 8)

    np.testing.assert_array_equal(
        teacher[-1],
        decoded,
        err_msg="WeightOnlyHostRunner prefill+decode does not match teacher-forced",
    )


def test_weight_only_host_runner_rejects_out_of_sequence_decode():
    """The runner maintains an internal KV cache cursor; out-of-sequence
    decode calls (position != cursor) must raise rather than silently
    produce garbage logits."""
    _skip_if_no_fixtures()

    payload = torch.load(FIXTURE, map_location="cpu")
    runner = WeightOnlyHostRunner(payload)
    runner.run_prefill([1, 2, 3])
    # Position 0 already consumed; calling with position=0 should raise.
    with pytest.raises(ValueError, match="does not match internal cursor"):
        runner.run_decode_step(4, 0)


def test_weight_only_int8_logits_match_diagnostic_bit_identical():
    """Bit-identical equivalence between the pipeline's W8A32 path and the
    diagnostic at `software/tools/diagnose_weight_only_qdq_ceiling.py`.

    Both call `incremental_logits_trace` on `NanoGPTFP32Reference` with
    QDQ weights produced by the same helper — same numpy → same bits.
    A `1e-5` tolerance would let a real divergence (lm_head padding,
    mode-string drift, or accidental float64 promotion) silently pass.
    """
    _skip_if_no_fixtures()

    payload = torch.load(FIXTURE, map_location="cpu")
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=33)
    eval_tokens = [int(tok) for tok in eval_ids[:33]]
    inputs, _ = teacher_forced_inputs_and_targets(eval_tokens)

    # Path A: pipeline helper (what evaluate_gpt2_perplexity uses).
    pipeline_logits = run_weight_only_int8_teacher_forced_logits(payload, eval_tokens)

    # Path B: diagnostic-style construction via the shared helper.
    diag_ref = build_weight_only_int8_reference(payload, weight_mode="per_channel")
    diag_logits = diag_ref.incremental_logits_trace(inputs)

    assert len(pipeline_logits) == len(diag_logits)
    for idx, (a, b) in enumerate(zip(pipeline_logits, diag_logits)):
        np.testing.assert_array_equal(
            a,
            b,
            err_msg=(
                f"W8A32 pipeline vs diagnostic divergence at step {idx}: "
                f"max |Δ|={float(np.max(np.abs(a - b))):.6g}"
            ),
        )


# ---------------------------------------------------------------------------
# Slow gate at the 257-tok / 256-ctx production window
# ---------------------------------------------------------------------------

def test_weight_only_int8_slow_gate_at_production_window():
    """Production-window W8A32 gate. Acceptance: fake_quant_perplexity is
    within 1 PPL of the FP32 ceiling, matching the diagnostic at
    `software/tools/diagnose_weight_only_qdq_ceiling.py` to a tight
    headroom. The 1-PPL bound is consistent with the bit-identical
    numpy paths: any drift would be a real regression, not noise.

    Phase 3 option (b): also asserts `golden_perplexity` matches
    `fake_quant_perplexity` within 1e-9 (both wrap the same QDQ helper).
    """
    if os.environ.get("PYTEST_SLOW") != "1":
        pytest.skip("set PYTEST_SLOW=1 to run the W8A32 production slow gate")
    _skip_if_no_fixtures()
    if not CALIB_TEXT.exists():
        pytest.skip("calibration text fixture missing")

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
        # Irrelevant for W8A32 but kept small so the constructor short-circuit
        # doesn't burn time on per-token activation sweeps.
        calibration_n_seqs=2,
        calibration_seq_len=8,
        ptq_preset="weight_only_int8",
    )
    assert result.ptq_preset == "weight_only_int8"
    assert result.target_count == 256
    # Phase 3 (b): golden is populated and matches fake_quant tightly.
    assert not math.isnan(result.golden_perplexity)
    assert not math.isnan(result.relative_delta)
    assert result.relative_delta < 1e-9, (
        f"W8A32 golden vs fake_quant divergence at 257-tok slow gate: "
        f"golden={result.golden_perplexity:.6f} "
        f"fake_quant={result.fake_quant_perplexity:.6f} "
        f"relative_delta={result.relative_delta:.6e}"
    )
    assert not math.isnan(result.fp32_perplexity)
    # Diagnostic recorded 53.4212 at this window; per-channel weight QDQ
    # is actually slightly better than the un-rotated FP32 reference
    # because the rounding noise is symmetric. Allow +1 PPL headroom over
    # FP32 to keep the gate tight (any larger drift = real regression).
    assert result.fake_quant_perplexity <= result.fp32_perplexity + 1.0, (
        f"W8A32 slow-gate regression: "
        f"fake_quant_ppl={result.fake_quant_perplexity:.4f} vs "
        f"fp32_ppl={result.fp32_perplexity:.4f} "
        f"(plan target: ≤54 PPL)"
    )
    assert result.golden_perplexity <= result.fp32_perplexity + 1.0, (
        f"W8A32 golden_ppl={result.golden_perplexity:.4f} exceeded "
        f"fp32_ppl + 1: fp32={result.fp32_perplexity:.4f}"
    )
    # And confirm we're nowhere near the W8A8 6,174 baseline.
    assert result.fake_quant_perplexity < 100.0, (
        f"W8A32 fake_quant_ppl={result.fake_quant_perplexity:.4f} "
        f"is implausibly high — expected ≈53.42"
    )
