"""Phase 2 (2a) tests: lockstep batched decode graph builds + runs in golden.

The batched decode stream carries 16 query rows through the shared
non-attention path (embeddings/LN/FFN/quant/logits are all M-shaped and batch
for free). Attention is emitted dense here so the decode graph references the
same weights as the single-token prefill graph — they share one data blob.
Per-stream block-diagonal attention + the byte-exact RTL leg arrive in 2b.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from taccel.compiler.frontend import load_frontend
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_nanogpt_config():
    return {
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 16,
        "vocab_size": 65,
        "bias": True,
    }


def _decode_sites(bundle, kind):
    return [s for s in bundle.runtime_patch_sites if s.kind == kind and s.stream == "decode"]


def test_forward_batch16_frontend_shape():
    """The batched variant is a 16-row graph that still emits attention."""
    result = load_frontend("nanogpt", config=_tiny_nanogpt_config(), variant="forward_batch16")
    ops = {node.op for node in result.graph.nodes}
    assert "matmul_qkt" in ops and "softmax" in ops and "matmul_attn_v" in ops
    assert result.graph.get_node("tok_embed").attrs["token_ids"] == [0] * 16
    assert result.graph.nodes[-1].name == "lm_head"
    assert result.graph.nodes[-1].output_shape[0] == 16
    assert all(node.output_shape[0] == 16 for node in result.graph.nodes if node.output_shape)


def test_build_stage3_rejects_bad_batch():
    with pytest.raises(ValueError):
        build_stage3_tiny_decoder_bundle({}, batch=4)


def test_batched_decode_graph_is_per_stream():
    """The decode graph must expand attention block-diagonally per stream."""
    from taccel.compiler.frontend.nanogpt_adapter import load_nanogpt_batched_decode
    import collections

    n_layer, n_head, n_streams = 2, 4, 16
    fe = load_nanogpt_batched_decode(config=_tiny_nanogpt_config(), key_len=3, n_streams=n_streams)
    c = collections.Counter(n.op for n in fe.graph.nodes)
    per_stream = n_layer * n_head * n_streams
    assert c["matmul_qkt"] == per_stream
    assert c["matmul_attn_v"] == per_stream
    assert c["softmax"] == per_stream
    assert c["row_copy"] == per_stream            # per-stream query extract
    assert c["kv_store"] == 2 * per_stream        # key + value per stream
    assert c["kv_load"] == 2 * per_stream
    assert c["gather_rows"] == n_layer * n_head   # one gather per head
    # Every KV node carries a stream tag spanning all 16 streams.
    kv = [n for n in fe.graph.nodes if n.op in ("kv_store", "kv_load")]
    assert all("stream" in n.attrs for n in kv)
    assert sorted({n.attrs["stream"] for n in kv}) == list(range(n_streams))


def _build_payload(tmp_path):
    pytest.importorskip("torch")
    tool = _load_tool()
    checkpoint = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    tool.write_fixture(checkpoint, metadata_path)
    import torch

    return torch.load(checkpoint, map_location="cpu")


def test_batched_bundle_builds_and_has_16_embedding_sites(tmp_path):
    payload = _build_payload(tmp_path)

    base = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2, batch=1)
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2, batch=16)

    # Batch dimension flows through the embedding lookups: one runtime patch
    # site per row (16), vs a single site in the single-token decoder.
    assert len(_decode_sites(base.build.bundle, "token_embed")) == 1
    assert len(_decode_sites(tiny.build.bundle, "token_embed")) == 16
    assert len(_decode_sites(tiny.build.bundle, "pos_embed")) == 16

    # The single-token bundle's prefill has one embed site; the batched
    # bundle reuses the batched graph for its prefill slot (16 sites) so the
    # two streams share one weight/scale data blob.
    assert len([s for s in base.build.bundle.runtime_patch_sites
                if s.kind == "token_embed" and s.stream == "prefill"]) == 1
    assert len([s for s in tiny.build.bundle.runtime_patch_sites
                if s.kind == "token_embed" and s.stream == "prefill"]) == 16


def test_batched_decode_runs_clean_in_golden(tmp_path):
    payload = _build_payload(tmp_path)
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2, batch=16)

    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
    # Batched bundle's prefill slot is the lockstep batched graph (16 rows).
    vocab = _tiny_nanogpt_config()["vocab_size"]
    runner.run_prefill([(3 + s) % vocab for s in range(16)])
    tokens16 = [(1 + s) % vocab for s in range(16)]
    out = runner.run_decode_step_batch(tokens16, position=1)

    assert out.shape == (tiny.logits_size,)
    assert np.any(out)


def test_batch1_decode_still_matches_single_token_path(tmp_path):
    """The batch knob must leave the single-token decoder byte-identical."""
    payload = _build_payload(tmp_path)
    default_bundle = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2)
    batch1_bundle = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2, batch=1)

    assert bytes(default_bundle.build.bundle.decode_instrs) == bytes(batch1_bundle.build.bundle.decode_instrs)
    assert bytes(default_bundle.build.bundle.prefill_instrs) == bytes(batch1_bundle.build.bundle.prefill_instrs)
    assert bytes(default_bundle.build.bundle.shared_data) == bytes(batch1_bundle.build.bundle.shared_data)
