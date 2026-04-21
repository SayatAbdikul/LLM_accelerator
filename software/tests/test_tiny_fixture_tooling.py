"""Tests for Stage 3 tiny fixture metadata tooling."""
import importlib.util
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_metadata_roundtrip_and_checksum(tmp_path):
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")

    metadata = tool.write_fixture(checkpoint, metadata_path)
    loaded = tool.validate_fixture_metadata(checkpoint, metadata_path)

    assert loaded["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert loaded["seed"] == 1337
    assert loaded["tokenizer"]["kind"] == "character"
    assert "alphabet" in loaded["tokenizer"]
    assert "calibration_bytes" in loaded["ranges"]
    assert "evaluation_bytes" in loaded["ranges"]
    assert loaded["source_snapshot"]
    assert loaded["stage3e_logits_smoke_ready"] is True
    assert loaded["stage3f_full_graph_smoke_ready"] is True
    assert loaded["stage3c_e2e_ready"] is True


def test_default_fixture_skip_message_when_absent():
    tool = _load_tool()
    if tool.DEFAULT_FIXTURE.exists():
        tool.validate_fixture_metadata(tool.DEFAULT_FIXTURE, tool.DEFAULT_METADATA)
        return

    with pytest.raises(FileNotFoundError):
        tool.validate_fixture_metadata(tool.DEFAULT_FIXTURE, tool.DEFAULT_METADATA)
