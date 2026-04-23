"""Stage 5 deterministic GPT-2-class logits gate.

This test is active when a converted GPT-2 nanoGPT-format checkpoint exists
locally.  The converter intentionally writes ignored `.pt` artifacts, so CI and
fresh clones skip with an exact generation command.
"""
from pathlib import Path

import pytest
import torch

from taccel.runtime.tiny_fixture import run_stage3_tiny_e2e


FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")


def test_converted_gpt2_logits_match_fake_quant_reference():
    if not FIXTURE.exists():
        pytest.skip(
            "converted GPT-2 fixture missing; run "
            "PYTHONPATH=software python software/tools/convert_hf_gpt2_to_nanogpt.py "
            "gpt2 --output software/tests/fixtures/generated/gpt2_converted_nanogpt.pt"
        )
    payload = torch.load(FIXTURE, map_location="cpu")
    result = run_stage3_tiny_e2e(payload, prompt_ids=[0], max_new_tokens=2)

    assert result.min_cosine >= 0.995
    assert min(result.top5_overlap_per_step) >= 4
    assert result.generated == result.reference_generated
