"""Phase 2 (2b-kv): per-stream KV store/load isolation in the golden sim.

Stores 16 distinct rows of a batched K projection into 16 independent
per-stream caches (static ``s * stream_span`` offset), then loads each stream
back and asserts stream ``s`` round-trips its own row — no cross-stream
contamination. The per-stream offset is compile-time; the runtime ``kv_base``
patch (token position) is applied on top and here pinned to position 0.
"""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.golden_model.simulator import Simulator
from taccel.isa.opcodes import BUF_ABUF


N_STREAMS = 16
D_HEAD = 16


def _config():
    return ModelConfig(
        name="kv-stream",
        model_kind="decoder",
        n_layer=1,
        n_head=1,
        d_model=D_HEAD,
        d_head=D_HEAD,
        mlp_dim=D_HEAD,
        vocab_size=8,
        max_seq_len=4,
        embedding_kind="token_pos",
    )


def _decode_graph():
    g = IRGraph()
    # Per-stream stores: row s of the batched K projection (src tile at ABUF
    # offset 0) -> stream s's cache at the runtime position (kv_base).
    for s in range(N_STREAMS):
        g.add_node(IRNode(
            op="kv_store", name=f"store_s{s}", inputs=[], output_shape=(),
            attrs={
                "layer": 0, "kind": "key", "head": 0, "tokens": 1,
                "src_buf": BUF_ABUF, "src_off_units": 0, "src_row": s,
                "stream": s, "decode": True,
            },
        ))
    # Per-stream loads: stream s's cache -> a distinct ABUF destination.
    for s in range(N_STREAMS):
        g.add_node(IRNode(
            op="kv_load", name=f"load_s{s}", output_shape=(1, D_HEAD),
            attrs={
                "layer": 0, "kind": "key", "head": 0, "tokens": 1,
                "dst_buf": BUF_ABUF, "dst_off_units": 100 + s * 2,
                "stream": s, "decode": True,
            },
        ))
    return g


def _build():
    return build_decoder_program_bundle(
        prefill_graph=IRGraph(),
        decode_graph=_decode_graph(),
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=_config(),
        n_streams=N_STREAMS,
    )


def test_per_stream_store_load_round_trips_without_contamination():
    build = _build()
    bundle = build.bundle
    sim = Simulator()
    sim.load_bundle(bundle)

    # Row s of the batched projection is 32 bytes (16 FP16) all equal to s.
    row_bytes = D_HEAD * 2
    src = np.zeros((N_STREAMS, row_bytes), dtype=np.uint8)
    for s in range(N_STREAMS):
        src[s, :] = s
    sim.state.abuf[:N_STREAMS * row_bytes] = src.tobytes()

    # Pin every kv_base site to position 0 (per-stream separation is static).
    for site in bundle.runtime_patch_sites:
        if site.kind == "kv_base":
            bundle.patch_runtime_site(site, 0)

    sim.run_program(bundle, "decode")

    for s in range(N_STREAMS):
        off = (100 + s * 2) * 16
        got = bytes(sim.state.abuf[off:off + row_bytes])
        assert got == bytes([s]) * row_bytes, (
            f"stream {s} loaded {got[:4]!r}..., expected all {s}"
        )


def test_kv_cache_is_sized_for_all_streams():
    build = _build()
    # 1 layer * 2 kinds * 1 head * (16 streams * seq_len(4) * d_head(16) * 2B)
    assert build.bundle.kv_cache_size == 1 * 2 * 1 * (N_STREAMS * 4 * D_HEAD * 2)
