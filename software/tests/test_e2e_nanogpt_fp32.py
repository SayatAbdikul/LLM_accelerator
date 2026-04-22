"""Rank-based golden-vs-FP32 nanoGPT e2e diagnostics."""
import importlib.util
from pathlib import Path

import pytest
import torch

from taccel.runtime.tiny_fixture import run_nanogpt_fp32_e2e


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_fp32_rank_gate(result):
    assert result.min_top5_overlap >= 3
    assert result.fp32_top1_in_golden_top5_all
    assert result.golden_top1_in_fp32_top5_all


def test_stage3_d128_golden_generation_matches_fp32_rank_gate():
    tool = _fixture_tool()
    if not tool.DEFAULT_FIXTURE.exists():
        pytest.skip(
            "Stage 3 d=128 fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )
    tool.validate_fixture_metadata(tool.DEFAULT_FIXTURE, tool.DEFAULT_METADATA)
    payload = torch.load(tool.DEFAULT_FIXTURE, map_location="cpu")

    result = run_nanogpt_fp32_e2e(payload, prompt_ids=[0], max_new_tokens=32)

    assert len(result.generated) == 33
    assert len(result.fp32_generated) == 33
    _assert_fp32_rank_gate(result)


def test_stage4_d384_golden_generation_matches_fp32_rank_gate():
    tool = _fixture_tool()
    if not tool.DEFAULT_STAGE4_FIXTURE.exists():
        pytest.skip(
            "Stage 4 d=384 fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py --stage4-d384"
        )
    tool.validate_fixture_metadata(tool.DEFAULT_STAGE4_FIXTURE, tool.DEFAULT_STAGE4_METADATA)
    payload = torch.load(tool.DEFAULT_STAGE4_FIXTURE, map_location="cpu")
    vocab = int(payload["model_args"]["vocab_size"])
    prompts = [[idx % vocab] for idx in range(5)]

    for prompt in prompts:
        result = run_nanogpt_fp32_e2e(payload, prompt_ids=prompt, max_new_tokens=64)
        assert len(result.generated) == 65
        assert len(result.fp32_generated) == 65
        _assert_fp32_rank_gate(result)
