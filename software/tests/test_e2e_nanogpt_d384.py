"""Stage 4 d=384 nanoGPT e2e gate."""
import importlib.util
from pathlib import Path

import pytest
import torch

from taccel.runtime.tiny_fixture import run_stage3_tiny_e2e


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_d384_nanogpt_fixture_available_for_stage4_e2e():
    tool = _fixture_tool()
    if not tool.DEFAULT_STAGE4_FIXTURE.exists():
        pytest.skip(
            "Stage 4 d=384 fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py --stage4-d384"
        )
    metadata = tool.validate_fixture_metadata(tool.DEFAULT_STAGE4_FIXTURE, tool.DEFAULT_STAGE4_METADATA)
    if not metadata.get("stage4_d384_ready", False):
        pytest.skip("fixture metadata exists but is marked stage4_d384_ready=false")

    payload = torch.load(tool.DEFAULT_STAGE4_FIXTURE, map_location="cpu")
    vocab = int(payload["model_args"]["vocab_size"])
    prompts = [[idx % vocab] for idx in range(5)]

    for prompt in prompts:
        result = run_stage3_tiny_e2e(payload, prompt_ids=prompt, max_new_tokens=64)
        assert result.min_cosine >= 0.995
        assert min(result.top5_overlap_per_step) >= 4
