"""Stage 3 bitwise determinism check for the tiny nanoGPT golden model.

Two independent compile + run paths must produce bit-identical INT8 logits at
every decode step.  This is a weaker property than the correctness gate in
test_e2e_tiny.py (which compares against a fake-quant reference), but it
catches non-determinism in the simulator, assembler, or calibration path.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import (
    build_stage3_tiny_decoder_bundle,
    run_tiny_decode_trace,
)


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_nanogpt_decode_is_bitwise_deterministic():
    """Two fresh compiles from the same fixture must produce bit-identical logits."""
    tool = _load_tool()
    if not tool.DEFAULT_FIXTURE.exists():
        pytest.skip(
            "tiny nanoGPT fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )

    torch = pytest.importorskip("torch")
    payload = torch.load(tool.DEFAULT_FIXTURE, map_location="cpu")

    MAX_NEW_TOKENS = 8
    PROMPT_IDS = [0]

    scales = build_calibration_scales(payload)

    tiny_a = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=MAX_NEW_TOKENS, calibration_scales=scales
    )
    tiny_b = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=MAX_NEW_TOKENS, calibration_scales=scales
    )

    trace_a = run_tiny_decode_trace(tiny_a, PROMPT_IDS, max_new_tokens=MAX_NEW_TOKENS)
    trace_b = run_tiny_decode_trace(tiny_b, PROMPT_IDS, max_new_tokens=MAX_NEW_TOKENS)

    assert trace_a.generated == trace_b.generated, (
        f"token sequences diverged: {trace_a.generated} vs {trace_b.generated}"
    )
    for step, (la, lb) in enumerate(zip(trace_a.logits, trace_b.logits)):
        assert np.array_equal(la, lb), (
            f"logits differ at step {step}: max |diff| = {np.abs(la.astype(np.int32) - lb.astype(np.int32)).max()}"
        )


def test_tmp_tiny_nanogpt_decode_is_bitwise_deterministic(tmp_path):
    """Same determinism check using a freshly generated fixture."""
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_fixture(checkpoint, metadata_path)

    payload = torch.load(checkpoint, map_location="cpu")

    MAX_NEW_TOKENS = 4
    PROMPT_IDS = [0]

    scales = build_calibration_scales(payload)

    tiny_a = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=MAX_NEW_TOKENS, calibration_scales=scales
    )
    tiny_b = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=MAX_NEW_TOKENS, calibration_scales=scales
    )

    trace_a = run_tiny_decode_trace(tiny_a, PROMPT_IDS, max_new_tokens=MAX_NEW_TOKENS)
    trace_b = run_tiny_decode_trace(tiny_b, PROMPT_IDS, max_new_tokens=MAX_NEW_TOKENS)

    assert trace_a.generated == trace_b.generated
    for step, (la, lb) in enumerate(zip(trace_a.logits, trace_b.logits)):
        assert np.array_equal(la, lb), f"logits differ at step {step}"
