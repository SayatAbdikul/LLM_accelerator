"""Stage 3 synthetic ProgramBundle tests for KV STORE/LOAD."""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.golden_model.simulator import Simulator
from taccel.isa.opcodes import BUF_ABUF


def _config():
    return ModelConfig(
        name="kv-store-load",
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


def _prefill_graph():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="kv_store",
        name="store_key_prefill",
        inputs=["key_src"],
        output_shape=(),
        attrs={
            "layer": 0,
            "kind": "key",
            "head": 0,
            "seq_len": 2,
            "src_buf": BUF_ABUF,
            "src_off_units": 0,
            "decode": False,
        },
    ))
    return graph


def _decode_graph():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="kv_load",
        name="load_key_decode",
        output_shape=(1, 16),
        attrs={
            "layer": 0,
            "kind": "key",
            "head": 0,
            "tokens": 1,
            "dst_buf": BUF_ABUF,
            "dst_off_units": 4,
            "decode": True,
        },
    ))
    return graph


def _build():
    return build_decoder_program_bundle(
        prefill_graph=_prefill_graph(),
        decode_graph=_decode_graph(),
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=_config(),
    )


def _kv_site(bundle):
    sites = [site for site in bundle.runtime_patch_sites if site.kind == "kv_base"]
    assert len(sites) == 1
    return sites[0]


def test_prefill_store_and_decode_load_persist_kv_cache():
    build = _build()
    bundle = build.bundle
    sim = Simulator()
    sim.load_bundle(bundle)
    rows = np.arange(32, dtype=np.int8)
    sim.state.abuf[:32] = rows.tobytes()

    sim.run_program(bundle, "prefill")
    assert bytes(sim.state.dram[bundle.kv_cache_base:bundle.kv_cache_base + 32]) == rows.tobytes()

    sim.state.abuf[64:80] = bytes(16)
    bundle.patch_runtime_site(_kv_site(bundle), 0)
    sim.run_program(bundle, "decode")
    assert bytes(sim.state.abuf[64:80]) == rows[:16].tobytes()

    sim.state.abuf[64:80] = bytes(16)
    bundle.patch_runtime_site(_kv_site(bundle), 16)
    sim.run_program(bundle, "decode")
    assert bytes(sim.state.abuf[64:80]) == rows[16:32].tobytes()


def test_decode_runtime_kv_patch_does_not_clobber_data_or_cache():
    build = _build()
    bundle = build.bundle
    sim = Simulator()
    sim.load_bundle(bundle)
    sim.state.abuf[:32] = bytes(range(32))
    sim.run_program(bundle, "prefill")
    shared_before = bytes(sim.state.dram[bundle.data_base:bundle.temp_base])
    cache_before = bytes(sim.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes])
    decode_before = bundle.stream_bytes("decode")

    bundle.patch_runtime_site(_kv_site(bundle), 16)
    sim.run_program(bundle, "decode")

    assert bundle.stream_bytes("decode") != decode_before
    assert bytes(sim.state.dram[bundle.data_base:bundle.temp_base]) == shared_before
    assert bytes(sim.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes]) == cache_before


def test_prefill_kv_base_is_static_relocation_not_runtime_patch():
    build = _build()

    assert [site.symbol for site in build.prefill_codegen.relocation_sites] == ["kv_bank0"]
    assert build.prefill_codegen.runtime_patch_sites == []
    assert [site.kind for site in build.decode_codegen.runtime_patch_sites] == ["kv_base"]
