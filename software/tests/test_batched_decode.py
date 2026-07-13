"""Phase 2 tests: lockstep batched (B=16) decode — graph, golden, RTL byte-match.

The batched decode stream carries 16 query rows through the shared
non-attention path (embeddings/LN/FFN/quant/logits are all M-shaped and batch
for free); attention is per-stream block-diagonal (each stream runs the
single-token query_len=1 QK^T/softmax/AV against its own KV cache). The final
test is the bit-exact contract: the mode-1 synth RTL and the mode-0 golden
model produce byte-identical logits for the batched decode program.
"""
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from taccel.assembler.assembler import ProgramBinary
from taccel.compiler.frontend import load_frontend
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

RTL_SYNTH_BINARY = (
    Path(__file__).resolve().parents[2]
    / "rtl" / "verilator" / "build" / "run_program_synth" / "Vtaccel_top"
)


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"

# Batched tests run at the PRODUCTION attention depth: key_len = 1 + 63 = 64, so
# Kseq_pad = 64 and the per-head attn_v tiles spill to DRAM-temp exactly as they
# do on GPT-2 124M. That spill is what bounds ABUF — without it the
# n_head x n_streams live attn_v tiles are 4 x 32 x 1 KB = the whole 128 KB ABUF,
# and B=32 cannot even build. (A shallow Kseq_pad < 64 is a toy regime that never
# occurs in a real decode.)
BATCHED_SMOKE = 63


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


@pytest.mark.parametrize("bad", [4, 8, 64])
def test_build_stage3_rejects_bad_batch(bad):
    # 64 must STAY rejected: fc2's 512-col streaming N-tile costs M_pad*512*4 of
    # ACCUM, which is exactly ACCUM_SIZE (64 KB) at M_pad=32 and would be 2x
    # over at 64. Lifting the cap past 32 requires the Stage-4 tile plan to take
    # the real M_pad (it assumes a 16-row strip, `codegen.STAGE4_M_TILE`).
    with pytest.raises(ValueError):
        build_stage3_tiny_decoder_bundle({}, batch=bad)


def test_batched_decode_graph_is_per_stream():
    """The decode graph must expand attention block-diagonally per stream."""
    from taccel.compiler.frontend.nanogpt_adapter import load_nanogpt_batched_decode
    import collections

    n_layer, n_head, n_streams = 2, 4, 16
    fe = load_nanogpt_batched_decode(config=_tiny_nanogpt_config(), key_len=3, n_streams=n_streams)
    c = collections.Counter(n.op for n in fe.graph.nodes)
    per_stream = n_layer * n_head * n_streams
    # Lever B: the 12-per-head QK^T matmuls collapse into ONE block-diagonal
    # packed matmul per (layer, stream); each head's scores are split back out
    # by a per-head dequant. Q is quantized straight from the batched projection
    # (no separate row_copy). softmax / attn_v stay per-head, byte-identical.
    assert c["matmul_qkt"] == 0
    assert c["packed_qkt_matmul"] == n_layer * n_streams   # one per (layer, stream)
    assert c["qkt_dequant"] == per_stream                  # one per (layer, head, stream)
    assert c["matmul_attn_v"] == per_stream
    assert c["softmax"] == per_stream
    assert c["row_copy"] == 0                      # query packed straight from projection
    assert c["kv_store"] == 2 * per_stream         # key + value per stream
    # Streaming K (lever B): K caches are loaded INSIDE the packed matmul
    # emitter (no standalone kv_load node), so only the per-head V loads remain
    # as graph nodes. Each packed node carries stream_k + its global k_heads.
    assert c["kv_load"] == per_stream              # value loads only
    packed = [n for n in fe.graph.nodes if n.op == "packed_qkt_matmul"]
    assert all(n.attrs.get("stream_k") for n in packed)
    assert all(n.attrs["k_heads"] == list(range(n_head)) for n in packed)
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


@pytest.mark.parametrize("batch", [16, 32])
def test_batched_bundle_builds_and_has_n_embedding_sites(tmp_path, batch):
    payload = _build_payload(tmp_path)

    base = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=BATCHED_SMOKE, batch=1)
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=BATCHED_SMOKE, batch=batch)

    # Batch dimension flows through the embedding lookups: one runtime patch
    # site per row (N), vs a single site in the single-token decoder.
    assert len(_decode_sites(base.build.bundle, "token_embed")) == 1
    assert len(_decode_sites(tiny.build.bundle, "token_embed")) == batch
    assert len(_decode_sites(tiny.build.bundle, "pos_embed")) == batch

    # The single-token bundle's prefill has one embed site; the batched
    # bundle reuses the batched graph for its prefill slot (N sites) so the
    # two streams share one weight/scale data blob.
    assert len([s for s in base.build.bundle.runtime_patch_sites
                if s.kind == "token_embed" and s.stream == "prefill"]) == 1
    assert len([s for s in tiny.build.bundle.runtime_patch_sites
                if s.kind == "token_embed" and s.stream == "prefill"]) == batch


@pytest.mark.parametrize("batch", [16, 32])
def test_batched_decode_runs_clean_in_golden(tmp_path, batch):
    payload = _build_payload(tmp_path)
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=BATCHED_SMOKE, batch=batch)

    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
    # Batched bundle's prefill slot is the lockstep batched graph (N rows).
    vocab = _tiny_nanogpt_config()["vocab_size"]
    runner.run_prefill([(3 + s) % vocab for s in range(batch)])
    tokens = [(1 + s) % vocab for s in range(batch)]
    out = runner.run_decode_step_batch(tokens, position=1)

    # Lever I-a (logits×N): the region spans all N per-stream logits rows, not
    # just row 0. `out` is the flat int8 view of the whole logits_size region;
    # reshape to per-stream rows and confirm every stream is populated.
    assert out.shape == (tiny.logits_size,)
    per_stream = out.reshape(batch, -1)
    assert per_stream.shape[1] == tiny.logits_size // batch
    assert np.all(per_stream.any(axis=1)), "a per-stream logits row is empty (store_rows<N?)"
    # The N streams decode distinct tokens, so their logits rows must differ —
    # proves the store captured per-stream data, not row 0 replicated N×.
    assert len({row.tobytes() for row in per_stream}) > 1, "per-stream rows are all identical"


@pytest.mark.parametrize("batch", [16, 32])
def test_batched_decode_rtl_matches_golden_bytes(tmp_path, batch):
    """Bit-exact contract: mode-1 synth RTL == mode-0 golden on the batched
    decode program (byte-identical logits), for every batched stream's row.

    NOTE this is the TINY model on purpose: RTL-vs-golden *byte*-match is only
    well-posed below GPT-2 124M's first fp16 overflow (block0_out_proj →
    ±65504/NaN), past which the golden itself saturates and conformance becomes
    logits-level (argmax/cosine/perplexity) — see rtl_cosim.py #109.
    """
    if not RTL_SYNTH_BINARY.exists():
        pytest.skip(f"synth RTL binary not built: {RTL_SYNTH_BINARY}")
    payload = _build_payload(tmp_path)
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=BATCHED_SMOKE, batch=batch)
    bundle = tiny.build.bundle
    pos = 1
    tokens16 = [(3 + s) % 64 for s in range(batch)]

    # Golden (mode-0 DPI). Decode-ONLY, to mirror the RTL side below, which runs
    # the extracted decode program on a fresh image. Running a prefill here (and
    # not on the RTL) would compare two different machine states.
    runner = HostRunner(bundle, logits_dtype=np.int8)
    golden = runner.run_decode_step_batch(tokens16, pos)

    # Extract the standalone decode program and run it on the synth RTL,
    # dumping the logits region for a byte compare.
    patcher = HostRunner(bundle, simulator=None)
    patcher._patch_embeddings("decode", tokens16, [pos] * batch)
    patcher._patch_kv_bases(pos)
    patcher._patch_decode_attention_context(pos)
    image = bundle.materialize(reset_runtime=False)
    data_base = int(bundle.data_base)
    decode_only = bytes(image[int(bundle.decode_instrs_offset):data_base])
    pb = ProgramBinary(
        instructions=decode_only, data=bytes(image[data_base:]), entry_point=0,
        insn_count=len(decode_only) // 8, data_base=data_base,
        input_offset=0, pos_embed_patch_dram_offset=0, pos_embed_cls_dram_offset=0,
        cls_token_dram_offset=0, trace_manifest={}, compiler_manifest={},
    )
    td = Path(tempfile.mkdtemp())
    (td / "p.bin").write_bytes(pb.to_bytes())
    logits_off, logits_size = int(bundle.decode_logits_offset), int(bundle.logits_size)
    argv = [str(RTL_SYNTH_BINARY), "--program", str(td / "p.bin"),
            "--json-out", str(td / "s.json"), "--fast-beats", "--max-cycles", "20000000",
            "--dram-dump-offset", str(logits_off), "--dram-dump-size", str(logits_size),
            "--dram-dump-out", str(td / "dram.bin")]
    cp = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    assert (td / "s.json").exists(), f"RTL run failed: {cp.stderr[-800:]}"
    summ = json.loads((td / "s.json").read_text())
    assert summ.get("status") == "halted" and not summ.get("fault"), summ
    rtl = np.frombuffer((td / "dram.bin").read_bytes(), dtype=np.int8)[:golden.shape[0]]

    assert np.array_equal(golden, rtl), (
        f"RTL != golden: {int(np.sum(golden != rtl))}/{golden.shape[0]} bytes differ"
    )


def test_batch1_decode_still_matches_single_token_path(tmp_path):
    """The batch knob must leave the single-token decoder byte-identical."""
    payload = _build_payload(tmp_path)
    default_bundle = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2)
    batch1_bundle = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=2, batch=1)

    assert bytes(default_bundle.build.bundle.decode_instrs) == bytes(batch1_bundle.build.bundle.decode_instrs)
    assert bytes(default_bundle.build.bundle.prefill_instrs) == bytes(batch1_bundle.build.bundle.prefill_instrs)
    assert bytes(default_bundle.build.bundle.shared_data) == bytes(batch1_bundle.build.bundle.shared_data)
