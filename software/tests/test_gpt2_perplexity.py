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
