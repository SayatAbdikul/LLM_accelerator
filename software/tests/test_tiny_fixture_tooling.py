"""Tests for Stage 3 tiny fixture metadata tooling."""
import importlib.util
import json
import subprocess
import sys
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


def test_trained_d128_fixture_cli_smoke_and_validation(tmp_path):
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2_trained.pt"
    metadata = checkpoint.with_suffix(checkpoint.suffix + ".json")

    subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--trained-d128",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--train-seq-len",
            "16",
            "--output",
            str(checkpoint),
            "--metadata",
            str(metadata),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    loaded = tool.validate_trained_fixture_metadata(checkpoint, metadata)
    assert loaded["real_trained_d128_ready"] is True
    assert loaded["training"]["steps"] == 2
    assert loaded["learning_rate"] == loaded["training"]["learning_rate"]
    assert "initial_train_loss" in loaded
    assert "final_train_loss" in loaded
    assert "training_quality" in loaded
    payload = torch.load(checkpoint, map_location="cpu")
    tool.validate_trained_payload(payload)


def test_trained_validation_rejects_missing_ready_flag(tmp_path):
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2_trained.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_fixture(checkpoint, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("real_trained_d128_ready", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="real_trained_d128_ready"):
        tool.validate_trained_fixture_metadata(checkpoint, metadata_path)


def test_trained_validation_rejects_all_zero_transformer_weights(tmp_path):
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2_trained.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_fixture(checkpoint, metadata_path)
    payload = torch.load(checkpoint, map_location="cpu")

    with pytest.raises(ValueError, match="not dense enough"):
        tool.validate_trained_payload(payload, require_quality=False)


def test_trained_quality_validation_rejects_unimproved_training():
    tool = _load_tool()
    payload = tool._train_d128_payload(steps=0, batch_size=2, train_seq_len=16)
    payload["training"]["steps"] = 600
    payload["training"]["final_train_loss"] = payload["training"]["initial_train_loss"]
    payload["training"]["train_loss"] = payload["training"]["initial_train_loss"]

    with pytest.raises(ValueError, match="loss did not improve"):
        tool.validate_trained_payload(payload, require_quality=True)
