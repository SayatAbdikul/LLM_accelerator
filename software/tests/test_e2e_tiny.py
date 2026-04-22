"""Stage 3 tiny nanoGPT e2e gate.

The trained checkpoint is generated locally and ignored by git.  Until it
exists, this test skips with the exact command needed to create metadata.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.tiny_fixture import run_stage3_tiny_e2e


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_nanogpt_fixture_available_for_stage3_e2e():
    tool = _load_tool()
    if not tool.DEFAULT_FIXTURE.exists():
        pytest.skip(
            "tiny nanoGPT fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )

    metadata = tool.validate_fixture_metadata(tool.DEFAULT_FIXTURE, tool.DEFAULT_METADATA)
    if not metadata.get("stage3c_e2e_ready", False):
        pytest.skip("fixture metadata exists but is marked stage3c_e2e_ready=false")

    torch = pytest.importorskip("torch")
    payload = torch.load(tool.DEFAULT_FIXTURE, map_location="cpu")

    result = run_stage3_tiny_e2e(payload, prompt_ids=[0], max_new_tokens=32)

    assert len(result.generated) == 33
    assert result.generated == result.reference_generated
    assert len(result.logits) == 33
    assert len(result.reference_logits) == 33
    assert result.min_cosine >= 0.995
    assert all(overlap >= 4 for overlap in result.top5_overlap_per_step)


def test_tmp_generated_tiny_nanogpt_fixture_runs_stage3_e2e(tmp_path):
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    metadata = tool.write_fixture(checkpoint, metadata_path)
    assert metadata["stage3c_e2e_ready"] is True

    payload = torch.load(checkpoint, map_location="cpu")
    result = run_stage3_tiny_e2e(payload, prompt_ids=[0], max_new_tokens=32)

    assert len(result.generated) == 33
    assert result.generated == result.reference_generated
    assert result.min_cosine >= 0.99


def test_incremental_fake_quant_matches_full_sequence_reference(tmp_path):
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_fixture(checkpoint, metadata_path)

    payload = torch.load(checkpoint, map_location="cpu")
    scales = build_calibration_scales(payload)
    ref = NanoGPTFQReference(payload["state_dict"], payload["model_args"], scales)
    token_ids = [0, 0, 1, 2, 3, 4, 5, 6]

    incremental = ref.incremental_logits_trace(token_ids)

    for idx, logits in enumerate(incremental):
        full = ref.forward(token_ids[:idx + 1])
        diff = logits.astype(np.int16) - full.astype(np.int16)
        assert np.percentile(np.abs(diff), 99) <= 1
