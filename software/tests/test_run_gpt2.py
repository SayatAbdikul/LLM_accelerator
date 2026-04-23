"""Smoke test for the Stage 5 GPT-2 runner CLI."""
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"
RUNNER = Path(__file__).resolve().parents[1] / "run_gpt2.py"
GPT2_FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_gpt2_accepts_nanogpt_payload_smoke(tmp_path):
    tool = _fixture_tool()
    checkpoint = tmp_path / "stage5_smoke_nanogpt.pt"
    metadata = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_stage4_fixture(checkpoint, metadata)

    env = os.environ.copy()
    env["PYTHONPATH"] = "software"
    result = subprocess.run(
        [
            "/tmp/llm_accelerator_stage3_venv/bin/python",
            str(RUNNER),
            str(checkpoint),
            "--prompt-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "generated_ids" in result.stdout
    assert "min_cosine" in result.stdout


def test_run_gpt2_perplexity_json_smoke(tmp_path):
    if not GPT2_FIXTURE.exists() or not TOKENIZER_DIR.exists():
        pytest.skip(
            "converted GPT-2 fixture/tokenizer missing; run the HF converter before "
            "the run_gpt2 perplexity smoke"
        )
    calibration = tmp_path / "calibration.txt"
    evaluation = tmp_path / "eval.txt"
    calibration.write_text("The quick brown fox jumps over the lazy dog.\n", encoding="utf-8")
    evaluation.write_text("The quick brown fox jumps again.\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = "software"
    result = subprocess.run(
        [
            "/tmp/llm_accelerator_stage3_venv/bin/python",
            str(RUNNER),
            str(GPT2_FIXTURE),
            "--prompt-ids",
            "0",
            "--max-new-tokens",
            "0",
            "--perplexity-text",
            str(evaluation),
            "--calibration-text",
            str(calibration),
            "--tokenizer-dir",
            str(TOKENIZER_DIR),
            "--max-eval-tokens",
            "3",
            "--context-len",
            "2",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "perplexity_golden" in result.stdout
    assert "perplexity_fake_quant" in result.stdout
    assert "perplexity_relative_delta" in result.stdout
