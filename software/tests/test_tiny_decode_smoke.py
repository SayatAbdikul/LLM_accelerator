"""Stage 3 tiny fixture smoke test for HostRunner logits plumbing."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_full_graph_smoke(payload):
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2)
    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)

    prefill_a = runner.run_prefill([0])
    decode_a = runner.run_decode_step(1, 1)
    decode_repeat = runner.run_decode_step(1, 1)

    runner_b = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
    runner_b.run_prefill([0])
    decode_b = runner_b.run_decode_step(2, 1)

    assert prefill_a.shape == (tiny.logits_size,)
    assert decode_a.shape == (tiny.logits_size,)
    assert decode_b.shape == (tiny.logits_size,)
    assert np.any(prefill_a)
    assert np.any(decode_a)
    assert np.any(decode_b)
    assert np.array_equal(decode_a, decode_repeat)
    assert not np.array_equal(decode_a, decode_b)


def test_generated_tiny_fixture_runs_full_graph_smoke_when_available():
    tool = _load_tool()
    if not tool.DEFAULT_FIXTURE.exists():
        pytest.skip(
            "tiny nanoGPT fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )
    metadata = tool.validate_fixture_metadata(tool.DEFAULT_FIXTURE, tool.DEFAULT_METADATA)
    if not metadata.get("stage3f_full_graph_smoke_ready", False):
        pytest.skip("fixture metadata exists but is not marked stage3f_full_graph_smoke_ready=true")

    torch = pytest.importorskip("torch")
    payload = torch.load(tool.DEFAULT_FIXTURE, map_location="cpu")
    _run_full_graph_smoke(payload)


def test_tmp_generated_fixture_runs_full_graph_smoke(tmp_path):
    pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    metadata = tool.write_fixture(checkpoint, metadata_path)
    assert metadata["stage3f_full_graph_smoke_ready"] is True

    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    _run_full_graph_smoke(payload)
