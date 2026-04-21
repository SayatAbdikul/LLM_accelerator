"""Stage 3 KV-cache scale-contract smoke tests."""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.golden_model.simulator import Simulator
from taccel.isa.opcodes import BUF_ABUF


def _config():
    return ModelConfig(
        name="kv-scale",
        model_kind="decoder",
        n_layer=1,
        n_head=1,
        d_model=16,
        d_head=16,
        mlp_dim=16,
        vocab_size=8,
        max_seq_len=4,
        embedding_kind="token_pos",
    )


def _prefill_graph(kind: str):
    graph = IRGraph()
    graph.add_node(IRNode(
        op="kv_store",
        name=f"store_{kind}",
        inputs=[f"{kind}_src"],
        attrs={
            "layer": 0,
            "kind": kind,
            "head": 0,
            "seq_len": 2,
            "src_buf": BUF_ABUF,
            "src_off_units": 0 if kind == "key" else 2,
            "decode": False,
        },
    ))
    return graph


def _decode_graph(kind: str):
    graph = IRGraph()
    graph.add_node(IRNode(
        op="kv_load",
        name=f"load_{kind}",
        attrs={
            "layer": 0,
            "kind": kind,
            "head": 0,
            "tokens": 1,
            "dst_buf": BUF_ABUF,
            "dst_off_units": 4 if kind == "key" else 5,
            "decode": True,
        },
    ))
    return graph


def _softmax(x):
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def test_cached_kv_uses_projection_native_quantized_bytes_without_requant():
    key_rows = (np.arange(32, dtype=np.int16) - 8).astype(np.int8).reshape(2, 16)
    value_rows = (np.arange(32, dtype=np.int16) + 3).astype(np.int8).reshape(2, 16)

    key_build = build_decoder_program_bundle(
        prefill_graph=_prefill_graph("key"),
        decode_graph=_decode_graph("key"),
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=_config(),
    )
    value_build = build_decoder_program_bundle(
        prefill_graph=_prefill_graph("value"),
        decode_graph=_decode_graph("value"),
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=_config(),
    )

    sim = Simulator()
    sim.load_bundle(key_build.bundle)
    sim.state.abuf[:32] = key_rows.tobytes()
    sim.run_program(key_build.bundle, "prefill")
    key_site = [site for site in key_build.bundle.runtime_patch_sites if site.kind == "kv_base"][0]
    key_build.bundle.patch_runtime_site(key_site, 16)
    sim.run_program(key_build.bundle, "decode")
    cached_key = np.frombuffer(bytes(sim.state.abuf[64:80]), dtype=np.int8)
    np.testing.assert_array_equal(cached_key, key_rows[1])

    sim = Simulator()
    sim.load_bundle(value_build.bundle)
    sim.state.abuf[32:64] = value_rows.tobytes()
    sim.run_program(value_build.bundle, "prefill")
    value_site = [site for site in value_build.bundle.runtime_patch_sites if site.kind == "kv_base"][0]
    value_build.bundle.patch_runtime_site(value_site, 16)
    sim.run_program(value_build.bundle, "decode")
    cached_value = np.frombuffer(bytes(sim.state.abuf[80:96]), dtype=np.int8)
    np.testing.assert_array_equal(cached_value, value_rows[1])

    q = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
    key_scale = np.float32(0.03125)
    value_scale = np.float32(0.0625)
    scores_from_cache = key_rows.astype(np.float32) * key_scale @ q
    scores_from_full = np.vstack([key_rows[0], cached_key]).astype(np.float32) * key_scale @ q
    np.testing.assert_allclose(scores_from_cache, scores_from_full, rtol=0, atol=0)

    probs = _softmax(scores_from_cache)
    out_from_cache = probs @ (value_rows.astype(np.float32) * value_scale)
    out_from_full = probs @ (np.vstack([value_rows[0], cached_value]).astype(np.float32) * value_scale)
    np.testing.assert_allclose(out_from_cache, out_from_full, rtol=0, atol=0)
