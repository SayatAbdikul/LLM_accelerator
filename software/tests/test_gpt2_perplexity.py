"""Stage 5 GPT-2 perplexity gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from taccel.runtime.gpt2_perplexity import (
    evaluate_gpt2_perplexity,
    file_sha256,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)


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
            "--eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt --max-eval-tokens 33 --json"
        )

    payload = torch.load(FIXTURE, map_location="cpu")
    calibration_ids = tokenize_text_file(TOKENIZER_DIR, CALIB_TEXT)
    eval_ids = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=33)
    result = evaluate_gpt2_perplexity(
        payload,
        calibration_token_ids=calibration_ids,
        eval_token_ids=eval_ids,
        tokenizer_dir=TOKENIZER_DIR,
        calibration_sha256=file_sha256(CALIB_TEXT),
        eval_sha256=file_sha256(EVAL_TEXT),
        max_eval_tokens=33,
        context_len=32,
        calibration_n_seqs=8,
        calibration_seq_len=32,
    )

    assert result.target_count == 32
    assert result.ptq_preset == "control"
    assert result.relative_delta <= 0.02, (
        f"golden_ppl={result.golden_perplexity:.6f}, "
        f"fake_quant_ppl={result.fake_quant_perplexity:.6f}, "
        f"relative_delta={result.relative_delta:.6%}, "
        f"token_count={result.token_count}, "
        f"tokenizer={result.tokenizer_dir}, "
        f"calibration_sha256={result.calibration_sha256}, "
        f"eval_sha256={result.eval_sha256}"
    )
