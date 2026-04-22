"""Tests for true FP32 nanoGPT reference and QKT scale metadata."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from taccel.compiler.frontend.nanogpt_adapter import load_nanogpt
from taccel.runtime.fp32_reference import NanoGPTFP32Reference


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _fixture_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(*, d_model: int, n_head: int, n_layer: int = 2, block_size: int = 16):
    return SimpleNamespace(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=d_model,
        block_size=block_size,
        vocab_size=32,
        bias=True,
        layer_norm_epsilon=1e-5,
    )


def test_fp32_incremental_logits_match_full_sequence_logits(tmp_path):
    torch = pytest.importorskip("torch")
    tool = _fixture_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_fixture(checkpoint, metadata)
    payload = torch.load(checkpoint, map_location="cpu")

    ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    token_ids = [0, 0, 1, 2, 3, 4]
    incremental = ref.incremental_logits_trace(token_ids)

    for idx, logits in enumerate(incremental):
        full = ref.forward(token_ids[:idx + 1])
        np.testing.assert_allclose(logits, full, rtol=1e-5, atol=1e-5)


def test_fp32_reference_instantiates_for_available_default_fixtures():
    torch = pytest.importorskip("torch")
    tool = _fixture_tool()
    paths = [
        tool.DEFAULT_FIXTURE,
        tool.DEFAULT_STAGE4_FIXTURE,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"generated fixtures missing: {missing}")

    for path in paths:
        payload = torch.load(path, map_location="cpu")
        ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
        logits = ref.forward([0])
        assert logits.shape == (int(payload["model_args"]["vocab_size"]),)
        assert np.isfinite(logits).all()


def test_nanogpt_adapter_sets_qkt_scale_for_d128_and_d384():
    cases = [
        _config(d_model=128, n_head=4, n_layer=2, block_size=128),
        _config(d_model=384, n_head=6, n_layer=6, block_size=256),
    ]
    for cfg in cases:
        result = load_nanogpt(config=cfg, variant="forward_1token")
        expected = (cfg.n_embd // cfg.n_head) ** -0.5
        qkt_nodes = [node for node in result.graph.nodes if node.op == "matmul_qkt"]
        assert qkt_nodes
        assert all(node.attrs["scale"] == expected for node in qkt_nodes)
