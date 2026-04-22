"""Real trained d128 nanoGPT golden-vs-fake-quant gate."""
import importlib.util
from pathlib import Path

import pytest

from taccel.runtime.tiny_fixture import run_stage3_tiny_e2e


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trained_d128_golden_matches_fake_quant_reference():
    torch = pytest.importorskip("torch")
    tool = _fixture_tool()
    if not tool.DEFAULT_TRAINED_D128_FIXTURE.exists():
        pytest.skip(
            "real trained d128 nanoGPT fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py --trained-d128"
        )
    metadata = tool.validate_trained_fixture_metadata(
        tool.DEFAULT_TRAINED_D128_FIXTURE,
        tool.DEFAULT_TRAINED_D128_METADATA,
    )
    if not metadata.get("real_trained_d128_ready", False):
        pytest.skip("trained fixture metadata exists but is marked real_trained_d128_ready=false")

    payload = torch.load(tool.DEFAULT_TRAINED_D128_FIXTURE, map_location="cpu")
    result = run_stage3_tiny_e2e(payload, prompt_ids=[0], max_new_tokens=32)

    assert len(result.generated) == 33
    assert result.generated == result.reference_generated
    assert result.min_cosine >= 0.995
    assert min(result.top5_overlap_per_step) >= 4
