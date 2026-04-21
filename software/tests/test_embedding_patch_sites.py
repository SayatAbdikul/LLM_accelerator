"""Stage 3 tests for dynamic token/position embedding patch sites."""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.compiler.codegen import CodeGenerator
from taccel.golden_model.simulator import Simulator


def _config():
    return ModelConfig(
        name="embed-test",
        model_kind="decoder",
        n_layer=1,
        n_head=1,
        d_model=16,
        d_head=16,
        mlp_dim=16,
        vocab_size=8,
        max_seq_len=16,
        embedding_kind="token_pos",
    )


def _weights():
    token = np.arange(8 * 16, dtype=np.int8).reshape(8, 16)
    pos = (np.arange(16 * 16, dtype=np.int16).reshape(16, 16) - 64).astype(np.int8)
    return {
        "transformer.wte.weight": (token, None),
        "transformer.wpe.weight": (pos, None),
    }


def _dynamic_embedding_graph():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        output_shape=(1, 16),
        attrs={"table": "transformer.wte.weight", "runtime_patch": True},
    ))
    graph.add_node(IRNode(
        op="pos_embed_lookup",
        name="pos_embed",
        output_shape=(1, 16),
        attrs={"table": "transformer.wpe.weight", "runtime_patch": True},
    ))
    return graph


def test_dynamic_embedding_patch_sites_load_selected_rows():
    config = _config()
    build = build_decoder_program_bundle(
        prefill_graph=_dynamic_embedding_graph(),
        decode_graph=_dynamic_embedding_graph(),
        weight_data=_weights(),
        calibration_scales={},
        prescaled_biases={},
        model_config=config,
    )
    bundle = build.bundle
    sim = Simulator()
    sim.load_bundle(bundle)

    bundle.patch_runtime_site("token_embed", 3 * config.d_model, stream="decode")
    bundle.patch_runtime_site("pos_embed", 5 * config.d_model, stream="decode")
    sim.run_program(bundle, "decode")

    token_table = _weights()["transformer.wte.weight"][0]
    pos_table = _weights()["transformer.wpe.weight"][0]
    pos_off = build.decode_codegen.mem.abuf.get("pos_embed").offset_units * 16
    assert bytes(sim.state.abuf[:16]) == token_table[3].tobytes()
    assert bytes(sim.state.abuf[pos_off:pos_off + 16]) == pos_table[5].tobytes()


def test_fixed_embedding_lookup_does_not_record_runtime_patch_sites():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        output_shape=(1, 16),
        attrs={"table": "transformer.wte.weight", "token_ids": [2]},
    ))
    codegen = CodeGenerator(_weights(), {}, {}, model_config=_config())

    codegen.generate(graph)

    assert codegen.runtime_patch_sites == []


def test_fixed_embedding_lookup_preserves_row_addend_in_program_bundle():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        output_shape=(1, 16),
        attrs={"table": "transformer.wte.weight", "token_ids": [2]},
    ))
    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,
        weight_data=_weights(),
        calibration_scales={},
        prescaled_biases={},
        model_config=_config(),
    )
    sim = Simulator()
    sim.load_bundle(build.bundle)

    sim.run_program(build.bundle, "prefill")

    token_table = _weights()["transformer.wte.weight"][0]
    assert bytes(sim.state.abuf[:16]) == token_table[2].tobytes()


def test_runtime_embedding_lookup_records_one_site_per_row():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        output_shape=(3, 16),
        attrs={"table": "transformer.wte.weight", "runtime_patch": True},
    ))
    codegen = CodeGenerator(_weights(), {}, {}, model_config=_config(), stream_name="prefill")

    codegen.generate(graph)

    sites = [site for site in codegen.runtime_patch_sites if site.kind == "token_embed"]
    assert len(sites) == 3
    assert [site.local_lo_pc for site in sites] == [0, 4, 8]
