"""Stage 5 GPT-2 perplexity gate."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from taccel.runtime import gpt2_perplexity as gpt2_ppl_mod
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_SEQ_LEN_LARGE,
    evaluate_gpt2_perplexity,
    file_sha256,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import stage5_default_ptq_preset_name


FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")


def test_stable_cross_entropy_matches_manual_log_softmax():
    logits = np.asarray([1.0, 2.0, -1.0], dtype=np.float32)
    nll = stable_cross_entropy(logits, 1, vocab_size=3)
    exp = np.exp(logits - logits.max())
    expected = -float(np.log(exp[1] / exp.sum()))
    assert nll == pytest.approx(expected)
    ppl, mean_nll = perplexity_from_nlls([nll, nll])
    assert mean_nll == pytest.approx(nll)
    assert ppl == pytest.approx(float(np.exp(nll)))


def test_teacher_forced_alignment_scores_next_token():
    inputs, targets = teacher_forced_inputs_and_targets([10, 20, 30, 40])
    assert inputs == [10, 20, 30]
    assert targets == [20, 30, 40]


def test_teacher_forced_alignment_requires_two_tokens():
    with pytest.raises(ValueError, match="at least two tokens"):
        teacher_forced_inputs_and_targets([7])
    with pytest.raises(ValueError, match="at least one NLL"):
        perplexity_from_nlls([])


def test_gpt2_default_ptq_preset_tracks_stage5_default():
    assert gpt2_ppl_mod.GPT2_DEFAULT_PTQ_PRESET == stage5_default_ptq_preset_name()


def test_evaluate_gpt2_perplexity_runs_quarot_before_calibration(monkeypatch):
    """When the preset has quarot_enabled=True, the rotation step must run
    BEFORE the first build_calibration_scales call so that calibration sees
    the rotated activation distribution."""
    call_order: list[str] = []

    def fake_quarot(*args, **kwargs):
        call_order.append("quarot")
        return [], {}

    def fake_build_scales(*args, **kwargs):
        call_order.append("build_scales")
        return {"lm_head": 1.0}

    def fake_policy(scales, *args, **kwargs):
        call_order.append("policy")
        return dict(scales)

    def fake_logits(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    monkeypatch.setattr(gpt2_ppl_mod, "apply_quarot_rotation_from_token_ids", fake_quarot)
    monkeypatch.setattr(gpt2_ppl_mod, "build_calibration_scales_from_token_ids", fake_build_scales)
    monkeypatch.setattr(gpt2_ppl_mod, "apply_stage5_ptq_scale_policy", fake_policy)
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_mlp_scale_search_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_lm_head_scale_search_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_bias_correction_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), []),
    )
    monkeypatch.setattr(gpt2_ppl_mod, "run_golden_teacher_forced_logits", fake_logits)
    monkeypatch.setattr(gpt2_ppl_mod, "run_fake_quant_teacher_forced_logits", fake_logits)

    gpt2_ppl_mod.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 4, "n_layer": 12, "n_embd": 768}, "state_dict": {}},
        calibration_token_ids=[0, 1, 2],
        eval_token_ids=[0, 1],
        tokenizer_dir=Path("."),
        ptq_preset="quarot_baseline",
        compute_fp32_ceiling=False,
    )

    # First call must be quarot, then build_scales+policy (calibration).
    assert call_order[0] == "quarot", (
        f"quarot must run first, got call_order={call_order}"
    )
    # build_scales should appear at least once after quarot.
    assert "build_scales" in call_order[1:], (
        f"build_calibration_scales not called after quarot; order={call_order}"
    )


def test_evaluate_gpt2_perplexity_skips_quarot_when_disabled(monkeypatch):
    """When the preset has quarot_enabled=False (the default preset path),
    apply_quarot_rotation_from_token_ids must NOT be called."""
    quarot_call_count = [0]

    def fake_quarot(*args, **kwargs):
        quarot_call_count[0] += 1
        return [], {}

    def fake_build_scales(*args, **kwargs):
        return {"lm_head": 1.0}

    def fake_policy(scales, *args, **kwargs):
        return dict(scales)

    def fake_logits(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    monkeypatch.setattr(gpt2_ppl_mod, "apply_quarot_rotation_from_token_ids", fake_quarot)
    monkeypatch.setattr(gpt2_ppl_mod, "build_calibration_scales_from_token_ids", fake_build_scales)
    monkeypatch.setattr(gpt2_ppl_mod, "apply_stage5_ptq_scale_policy", fake_policy)
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_mlp_scale_search_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_lm_head_scale_search_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_bias_correction_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), []),
    )
    monkeypatch.setattr(gpt2_ppl_mod, "run_golden_teacher_forced_logits", fake_logits)
    monkeypatch.setattr(gpt2_ppl_mod, "run_fake_quant_teacher_forced_logits", fake_logits)

    # Use the current default preset, which has quarot_enabled=False.
    gpt2_ppl_mod.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 4, "n_layer": 12, "n_embd": 768}, "state_dict": {}},
        calibration_token_ids=[0, 1, 2],
        eval_token_ids=[0, 1],
        tokenizer_dir=Path("."),
        compute_fp32_ceiling=False,
    )

    assert quarot_call_count[0] == 0, (
        f"apply_quarot_rotation_from_token_ids should not be called for "
        f"non-quarot preset; got {quarot_call_count[0]} calls"
    )


def test_evaluate_gpt2_perplexity_forwards_mlp_search_caps(monkeypatch):
    captured = {}

    def fake_build_scales(*args, **kwargs):
        return {"lm_head": 1.0}

    def fake_policy(scales, *args, **kwargs):
        return dict(scales)

    def fake_mlp_search(*args, **kwargs):
        captured["search_n_seqs_max"] = kwargs.get("search_n_seqs_max")
        captured["search_seq_len_max"] = kwargs.get("search_seq_len_max")
        return dict(args[2]), {}

    def fake_logits(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    monkeypatch.setattr(gpt2_ppl_mod, "build_calibration_scales_from_token_ids", fake_build_scales)
    monkeypatch.setattr(gpt2_ppl_mod, "apply_stage5_ptq_scale_policy", fake_policy)
    monkeypatch.setattr(gpt2_ppl_mod, "apply_output_aware_mlp_scale_search_from_token_ids", fake_mlp_search)
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_lm_head_scale_search_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_bias_correction_from_token_ids",
        lambda *args, **kwargs: (dict(args[2]), []),
    )
    monkeypatch.setattr(gpt2_ppl_mod, "run_golden_teacher_forced_logits", fake_logits)
    monkeypatch.setattr(gpt2_ppl_mod, "run_fake_quant_teacher_forced_logits", fake_logits)

    result = gpt2_ppl_mod.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 4}, "state_dict": {}},
        calibration_token_ids=[0, 1, 2],
        eval_token_ids=[0, 1],
        tokenizer_dir=Path("."),
        output_aware_search_n_seqs=4,
        output_aware_search_seq_len=64,
        compute_fp32_ceiling=False,
    )

    assert result.ptq_preset == stage5_default_ptq_preset_name()
    assert captured == {"search_n_seqs_max": 4, "search_seq_len_max": 64}


def test_gpt2_perplexity_gate_against_fake_quant_reference():
    missing = [
        str(path)
        for path in (FIXTURE, TOKENIZER_DIR, CALIB_TEXT, EVAL_TEXT)
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            "Stage 5 GPT-2 perplexity inputs missing: "
            f"{missing}. Provide local calibration/eval text files and run "
            "PYTHONPATH=software python software/tools/evaluate_gpt2_perplexity.py "
            "software/tests/fixtures/generated/gpt2_converted_nanogpt.pt "
            "--tokenizer-dir software/tests/fixtures/generated/hf_gpt2 "
            "--calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt "
            "--eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt "
            "--max-eval-tokens 257 --context-len 256 --json"
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
        calibration_n_seqs=8,
        calibration_seq_len=32,
    )

    assert result.target_count == 256
    assert result.ptq_preset == stage5_default_ptq_preset_name()
    assert result.relative_delta <= 0.02, (
        f"golden_ppl={result.golden_perplexity:.6f}, "
        f"fake_quant_ppl={result.fake_quant_perplexity:.6f}, "
        f"relative_delta={result.relative_delta:.6%}, "
        f"token_count={result.token_count}, "
        f"tokenizer={result.tokenizer_dir}, "
        f"calibration_sha256={result.calibration_sha256}, "
        f"eval_sha256={result.eval_sha256}"
    )


def test_gpt2_perplexity_production_calibration_slow_gate():
    if os.environ.get("PYTEST_SLOW") != "1":
        pytest.skip("set PYTEST_SLOW=1 to run the production calibration GPT-2 gate")
    missing = [
        str(path)
        for path in (FIXTURE, TOKENIZER_DIR, CALIB_TEXT, EVAL_TEXT)
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"Stage 5 GPT-2 perplexity inputs missing: {missing}")

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
        calibration_n_seqs=CALIBRATION_N_SEQS_LARGE,
        calibration_seq_len=CALIBRATION_SEQ_LEN_LARGE,
    )

    assert result.target_count == 256
    assert result.ptq_preset == stage5_default_ptq_preset_name()
    assert result.relative_delta <= 0.02, (
        f"golden_ppl={result.golden_perplexity:.6f}, "
        f"fake_quant_ppl={result.fake_quant_perplexity:.6f}, "
        f"relative_delta={result.relative_delta:.6%}"
    )


# ----------------------------------------------------------------------------
# Phase 0A: FP32 ceiling field on GPT2PerplexityResult
# ----------------------------------------------------------------------------

def test_gpt2_perplexity_result_has_fp32_ceiling_fields_with_nan_default():
    """Phase 0A: the result dataclass exposes fp32_perplexity / fp32_nll
    with NaN defaults so callers can opt in/out without breaking back-compat."""
    from taccel.runtime.gpt2_perplexity import GPT2PerplexityResult
    import math

    r = GPT2PerplexityResult(
        golden_perplexity=1.0,
        fake_quant_perplexity=2.0,
        relative_delta=0.5,
        token_count=10,
        target_count=9,
        golden_nll=0.1,
        fake_quant_nll=0.2,
        tokenizer_dir="",
        calibration_sha256="",
        eval_sha256="",
        ptq_preset="x",
    )
    assert math.isnan(r.fp32_perplexity)
    assert math.isnan(r.fp32_nll)


def test_evaluate_gpt2_perplexity_skip_fp32_ceiling_returns_nan(monkeypatch):
    """Phase 0A: when compute_fp32_ceiling=False the FP32 helper must NOT be
    called and the result fields stay NaN. This lets ablation runs skip the
    ~30s FP32 forward when only fake_quant numbers are needed."""
    import math

    fp32_called = [0]

    def fake_fp32(*args, **kwargs):
        fp32_called[0] += 1
        return [np.zeros(4, dtype=np.float32)]

    def fake_logits(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    monkeypatch.setattr(gpt2_ppl_mod, "run_fp32_teacher_forced_logits", fake_fp32)
    monkeypatch.setattr(gpt2_ppl_mod, "build_calibration_scales_from_token_ids", lambda *a, **k: {"lm_head": 1.0})
    monkeypatch.setattr(gpt2_ppl_mod, "apply_stage5_ptq_scale_policy", lambda s, *a, **k: dict(s))
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_mlp_scale_search_from_token_ids",
        lambda *a, **k: (dict(a[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_lm_head_scale_search_from_token_ids",
        lambda *a, **k: (dict(a[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_bias_correction_from_token_ids",
        lambda *a, **k: (dict(a[2]), []),
    )
    monkeypatch.setattr(gpt2_ppl_mod, "run_golden_teacher_forced_logits", fake_logits)
    monkeypatch.setattr(gpt2_ppl_mod, "run_fake_quant_teacher_forced_logits", fake_logits)

    result = gpt2_ppl_mod.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 4, "n_layer": 12, "n_embd": 768}, "state_dict": {}},
        calibration_token_ids=[0, 1, 2],
        eval_token_ids=[0, 1],
        tokenizer_dir=Path("."),
        compute_fp32_ceiling=False,
    )

    assert fp32_called[0] == 0, "FP32 helper must not be invoked when skipped"
    assert math.isnan(result.fp32_perplexity)
    assert math.isnan(result.fp32_nll)


def test_evaluate_gpt2_perplexity_compute_fp32_ceiling_default_true(monkeypatch):
    """Phase 0A: compute_fp32_ceiling defaults to True; the FP32 helper is
    invoked with the eval_tokens and its logits drive the fp32_perplexity field."""
    fp32_called = [0]

    def fake_fp32(payload, eval_tokens, *args, **kwargs):
        fp32_called[0] += 1
        # one row per (n_eval - 1) targets
        return [np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32) for _ in range(len(eval_tokens) - 1)]

    def fake_logits(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    monkeypatch.setattr(gpt2_ppl_mod, "run_fp32_teacher_forced_logits", fake_fp32)
    monkeypatch.setattr(gpt2_ppl_mod, "build_calibration_scales_from_token_ids", lambda *a, **k: {"lm_head": 1.0})
    monkeypatch.setattr(gpt2_ppl_mod, "apply_stage5_ptq_scale_policy", lambda s, *a, **k: dict(s))
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_mlp_scale_search_from_token_ids",
        lambda *a, **k: (dict(a[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_lm_head_scale_search_from_token_ids",
        lambda *a, **k: (dict(a[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_bias_correction_from_token_ids",
        lambda *a, **k: (dict(a[2]), []),
    )
    monkeypatch.setattr(gpt2_ppl_mod, "run_golden_teacher_forced_logits", fake_logits)
    monkeypatch.setattr(gpt2_ppl_mod, "run_fake_quant_teacher_forced_logits", fake_logits)

    result = gpt2_ppl_mod.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 4, "n_layer": 12, "n_embd": 768}, "state_dict": {}},
        calibration_token_ids=[0, 1, 2],
        eval_token_ids=[0, 1],
        tokenizer_dir=Path("."),
        # NOTE: compute_fp32_ceiling defaults to True
    )

    assert fp32_called[0] == 1, "FP32 helper should be invoked exactly once by default"
    # Logits = [0, 1, 0, 0]; target=1 → softmax probability of target = e/(1+3*1) = ~0.475
    # NLL = -log(0.475) ≈ 0.745. Just check it's a real, non-NaN finite value.
    import math
    assert not math.isnan(result.fp32_perplexity)
    assert not math.isnan(result.fp32_nll)
    assert result.fp32_perplexity > 0


def test_evaluate_gpt2_perplexity_fp32_ceiling_runs_before_state_mutations(monkeypatch):
    """Phase 0A: FP32 ceiling MUST be computed BEFORE any state_dict-mutating
    step (rotation, BC) so the returned number is the true pre-quantization
    perplexity. This test verifies the call ordering."""
    call_order: list[str] = []

    def fake_fp32(*args, **kwargs):
        call_order.append("fp32")
        return [np.zeros(4, dtype=np.float32)]

    def fake_quarot(*args, **kwargs):
        call_order.append("quarot")
        return [], {}

    def fake_bc(*args, **kwargs):
        call_order.append("bc")
        return (dict(args[2]), [])

    def fake_logits(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    monkeypatch.setattr(gpt2_ppl_mod, "run_fp32_teacher_forced_logits", fake_fp32)
    monkeypatch.setattr(gpt2_ppl_mod, "apply_quarot_rotation_from_token_ids", fake_quarot)
    monkeypatch.setattr(gpt2_ppl_mod, "apply_bias_correction_from_token_ids", fake_bc)
    monkeypatch.setattr(gpt2_ppl_mod, "build_calibration_scales_from_token_ids", lambda *a, **k: {"lm_head": 1.0})
    monkeypatch.setattr(gpt2_ppl_mod, "apply_stage5_ptq_scale_policy", lambda s, *a, **k: dict(s))
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_mlp_scale_search_from_token_ids",
        lambda *a, **k: (dict(a[2]), {}),
    )
    monkeypatch.setattr(
        gpt2_ppl_mod,
        "apply_output_aware_lm_head_scale_search_from_token_ids",
        lambda *a, **k: (dict(a[2]), {}),
    )
    monkeypatch.setattr(gpt2_ppl_mod, "run_golden_teacher_forced_logits", fake_logits)
    monkeypatch.setattr(gpt2_ppl_mod, "run_fake_quant_teacher_forced_logits", fake_logits)

    gpt2_ppl_mod.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 4, "n_layer": 12, "n_embd": 768}, "state_dict": {}},
        calibration_token_ids=[0, 1, 2],
        eval_token_ids=[0, 1],
        tokenizer_dir=Path("."),
        ptq_preset="quarot_with_bc",  # has both quarot and BC
    )

    fp32_idx = call_order.index("fp32")
    quarot_idx = call_order.index("quarot")
    bc_idx = call_order.index("bc")
    assert fp32_idx < quarot_idx, f"fp32 must precede quarot; got {call_order}"
    assert fp32_idx < bc_idx, f"fp32 must precede bias correction; got {call_order}"


# ----------------------------------------------------------------------------
# Phase 0B: KV cache FP32 toggle
# ----------------------------------------------------------------------------

def test_nanogpt_fq_reference_keep_kv_cache_fp32_toggle_changes_logits():
    """Phase 0B: when keep_kv_cache_fp32=True, the K/V cache stores FP32
    instead of INT8. Logits should differ from the default INT8-K/V path
    (sanity check that the toggle is wired through, not just stored as an
    attribute)."""
    from taccel.runtime.fake_quant_reference import NanoGPTFQReference
    from taccel.runtime.calibration import build_calibration_scales

    fixture = Path("software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2_trained.pt")
    if not fixture.exists():
        pytest.skip(f"fixture missing: {fixture}")

    payload = torch.load(fixture, map_location="cpu")
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]

    # Tiny calibration: 2 sequences of 8 tokens
    rng = np.random.default_rng(0xC0FFEE)
    vocab = int(model_args["vocab_size"])
    scales = build_calibration_scales(payload, n_seqs=2, seq_len=8)

    ref_int8 = NanoGPTFQReference(
        state_dict, model_args, scales, keep_kv_cache_fp32=False
    )
    ref_fp32_kv = NanoGPTFQReference(
        state_dict, model_args, scales, keep_kv_cache_fp32=True
    )

    eval_tokens = rng.integers(0, vocab, size=10).tolist()
    logits_int8 = ref_int8.incremental_logits_trace(eval_tokens)
    logits_fp32_kv = ref_fp32_kv.incremental_logits_trace(eval_tokens)

    assert len(logits_int8) == len(logits_fp32_kv) == len(eval_tokens)

    # The toggle should produce DIFFERENT logits (the K/V cache is computed
    # at higher precision). If they're identical the toggle did nothing.
    any_diff = False
    for a, b in zip(logits_int8, logits_fp32_kv):
        if not np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)):
            any_diff = True
            break
    assert any_diff, (
        "keep_kv_cache_fp32=True must change logits vs the INT8-K/V default "
        "for this fixture; got identical traces"
    )


def test_nanogpt_fq_reference_keep_kv_cache_fp32_default_is_int8():
    """Phase 0B: keep_kv_cache_fp32 defaults to False so production code
    paths (slow gate, deployed bundle reference) are unchanged."""
    from taccel.runtime.fake_quant_reference import NanoGPTFQReference
    import inspect

    sig = inspect.signature(NanoGPTFQReference.__init__)
    assert "keep_kv_cache_fp32" in sig.parameters
    assert sig.parameters["keep_kv_cache_fp32"].default is False


# ----------------------------------------------------------------------------
# Phase 1 Branch B: FP32 residual stream toggle
# ----------------------------------------------------------------------------

def test_nanogpt_fq_reference_fp32_residual_stream_toggle_changes_logits():
    """Phase 1 Branch B: when fp32_residual_stream=True, the residual stream
    (block_residual1/2 and LN outputs) is computed in FP32 instead of INT8.
    Logits should differ from the default INT8-residual path."""
    from taccel.runtime.fake_quant_reference import NanoGPTFQReference
    from taccel.runtime.calibration import build_calibration_scales

    fixture = Path("software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2_trained.pt")
    if not fixture.exists():
        pytest.skip(f"fixture missing: {fixture}")

    payload = torch.load(fixture, map_location="cpu")
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    scales = build_calibration_scales(payload, n_seqs=2, seq_len=8)

    ref_default = NanoGPTFQReference(state_dict, model_args, scales, fp32_residual_stream=False)
    ref_fp32_res = NanoGPTFQReference(state_dict, model_args, scales, fp32_residual_stream=True)

    rng = np.random.default_rng(0xBEEF)
    vocab = int(model_args["vocab_size"])
    eval_tokens = rng.integers(0, vocab, size=10).tolist()

    logits_default = ref_default.incremental_logits_trace(eval_tokens)
    logits_fp32_res = ref_fp32_res.incremental_logits_trace(eval_tokens)

    assert len(logits_default) == len(logits_fp32_res) == len(eval_tokens)

    any_diff = False
    for a, b in zip(logits_default, logits_fp32_res):
        if not np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)):
            any_diff = True
            break
    assert any_diff, (
        "fp32_residual_stream=True must change logits vs the INT8-residual default; "
        "got identical traces"
    )


def test_nanogpt_fq_reference_fp32_residual_stream_default_is_int8():
    """Phase 1 Branch B: fp32_residual_stream defaults to False so the
    deployed bundle reference is unchanged."""
    from taccel.runtime.fake_quant_reference import NanoGPTFQReference
    import inspect

    sig = inspect.signature(NanoGPTFQReference.__init__)
    assert "fp32_residual_stream" in sig.parameters
    assert sig.parameters["fp32_residual_stream"].default is False
