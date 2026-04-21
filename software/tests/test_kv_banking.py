"""Directed Stage 3 tests for deterministic multi-bank KV cache behavior."""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.kv_cache import M_DRAM_OFF_REACH_BYTES, build_kv_cache_layout
from taccel.compiler.model_config import ModelConfig
from taccel.golden_model.simulator import Simulator
from taccel.isa.opcodes import BUF_ABUF


def _multibank_config():
    return ModelConfig(
        name="kv-multibank",
        model_kind="decoder",
        n_layer=2,
        n_head=64,
        d_model=1024,
        d_head=16,
        mlp_dim=1024,
        vocab_size=64,
        max_seq_len=1024,
        embedding_kind="token_pos",
    )


def test_multibank_layout_rule_and_cross_bank_decode_load():
    config = _multibank_config()
    layout = build_kv_cache_layout(config)
    assert len(layout.banks) >= 2
    assert [bank.bank_id for bank in layout.banks] == list(range(len(layout.banks)))

    for entry in layout.entries:
        assert entry.bank_offset <= M_DRAM_OFF_REACH_BYTES
        assert entry.dram_off_units <= 0xFFFF

    first_cross_bank_entry = next(entry for entry in layout.entries if entry.bank_id == 1)
    assert first_cross_bank_entry.byte_offset == layout.banks[1].base_offset

    prefill_graph = IRGraph()
    decode_graph = IRGraph()
    decode_graph.add_node(IRNode(
        op="kv_load",
        name="cross_bank_load",
        output_shape=(1, config.d_head),
        attrs={
            "layer": first_cross_bank_entry.layer,
            "kind": first_cross_bank_entry.kind,
            "head": first_cross_bank_entry.head,
            "tokens": 1,
            "dst_buf": BUF_ABUF,
            "dst_off_units": 3,
            "decode": True,
        },
    ))
    build = build_decoder_program_bundle(
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=config,
    )
    bundle = build.bundle
    site = next(site for site in bundle.runtime_patch_sites if site.kind == "kv_base")
    assert site.base_symbol == first_cross_bank_entry.base_symbol

    sim = Simulator()
    sim.load_bundle(bundle)
    position = 7
    expected = np.arange(config.d_head, dtype=np.int8).tobytes()
    address = (
        bundle.symbol_address(first_cross_bank_entry.base_symbol)
        + first_cross_bank_entry.dram_off_units * 16
        + position * config.d_head
    )
    sim.state.dram[address:address + config.d_head] = expected

    bundle.patch_runtime_site(site, position * config.d_head)
    sim.run_program(bundle, "decode")

    dst = 3 * 16
    assert bytes(sim.state.abuf[dst:dst + config.d_head]) == expected
