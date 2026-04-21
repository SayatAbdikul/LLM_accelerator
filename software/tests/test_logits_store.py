"""Stage 3 tests for decoder logits_store lowering."""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.golden_model.simulator import Simulator
from taccel.isa.opcodes import BUF_ABUF
from taccel.runtime.host_runner import HostRunner


def _config(vocab_size=8):
    return ModelConfig(
        name="logits-store-test",
        model_kind="decoder",
        n_layer=1,
        n_head=1,
        d_model=16,
        d_head=16,
        mlp_dim=16,
        vocab_size=vocab_size,
        max_seq_len=16,
        embedding_kind="token_pos",
    )


def _weights(vocab_size=8):
    token = (np.arange(vocab_size * 16, dtype=np.int16).reshape(vocab_size, 16) - 32).astype(np.int8)
    pos = (np.arange(16 * 16, dtype=np.int16).reshape(16, 16) + 17).astype(np.int8)
    return {
        "transformer.wte.weight": (token, None),
        "transformer.wpe.weight": (pos, None),
    }


def _fixed_logits_graph(token_id):
    del token_id
    graph = IRGraph()
    graph.add_node(IRNode(
        op="logits_store",
        name="store_logits",
        inputs=["manual_logits"],
        output_shape=(1, 16),
        attrs={"src_buf": BUF_ABUF, "src_off_units": 0},
    ))
    return graph


def _dynamic_logits_graph():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        output_shape=(1, 16),
        attrs={"table": "transformer.wte.weight", "runtime_patch": True},
    ))
    graph.add_node(IRNode(
        op="logits_store",
        name="store_logits",
        inputs=["tok_embed"],
        output_shape=(1, 16),
    ))
    return graph


def test_codegen_logits_store_writes_prefill_and_decode_regions():
    build = build_decoder_program_bundle(
        prefill_graph=_fixed_logits_graph(2),
        decode_graph=_fixed_logits_graph(5),
        weight_data=_weights(),
        calibration_scales={},
        prescaled_biases={},
        model_config=_config(),
        logits_size=16,
    )
    bundle = build.bundle
    sim = Simulator()
    sim.load_bundle(bundle)
    token_table = _weights()["transformer.wte.weight"][0]
    shared_before = bytes(sim.state.dram[bundle.data_base:bundle.temp_base])
    kv_before = bytes(sim.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes])

    sim.state.abuf[:16] = token_table[2].tobytes()
    sim.run_program(bundle, "prefill")
    assert bytes(sim.state.dram[bundle.prefill_logits_offset:bundle.prefill_logits_offset + 16]) == (
        token_table[2].tobytes()
    )

    sim.state.abuf[:16] = token_table[5].tobytes()
    sim.run_program(bundle, "decode")
    assert bytes(sim.state.dram[bundle.decode_logits_offset:bundle.decode_logits_offset + 16]) == (
        token_table[5].tobytes()
    )
    assert bytes(sim.state.dram[bundle.data_base:bundle.temp_base]) == shared_before
    assert bytes(sim.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes]) == kv_before


def test_host_runner_reads_codegen_logits_from_dynamic_embedding_sites():
    config = _config()
    build = build_decoder_program_bundle(
        prefill_graph=_dynamic_logits_graph(),
        decode_graph=_dynamic_logits_graph(),
        weight_data=_weights(),
        calibration_scales={},
        prescaled_biases={},
        model_config=config,
        logits_size=16,
    )
    runner = HostRunner(build.bundle, logits_dtype=np.int8)
    token_table = _weights()["transformer.wte.weight"][0]

    prefill_logits = runner.run_prefill([3])
    decode_logits = runner.run_decode_step(6, 1)

    assert np.array_equal(prefill_logits, token_table[3])
    assert np.array_equal(decode_logits, token_table[6])
    assert [site.kind for site in build.prefill_codegen.runtime_patch_sites] == ["token_embed"]
    assert [site.kind for site in build.decode_codegen.runtime_patch_sites] == ["token_embed"]
