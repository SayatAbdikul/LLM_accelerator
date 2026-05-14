"""Tests for narrow decoder bundle graph helpers."""
import numpy as np

from taccel.compiler.decoder_bundle import (
    build_decoder_program_bundle,
    inject_kv_cache_nodes,
    mark_runtime_embedding_lookups,
)
from taccel.compiler.frontend.nanogpt_adapter import load_nanogpt
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.isa.instructions import LoadInsn


def _config():
    return {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "vocab_size": 32,
        "block_size": 16,
        "bias": True,
    }


def _weight_data():
    return {
        "transformer.wte.weight": (np.zeros((32, 16), dtype=np.int8), None),
        "transformer.wpe.weight": (np.zeros((16, 16), dtype=np.int8), None),
        "transformer.h.0.ln_1.weight": (np.ones(16, dtype=np.float16), None),
        "transformer.h.0.ln_1.bias": (np.zeros(16, dtype=np.float16), None),
        "transformer.h.0.ln_2.weight": (np.ones(16, dtype=np.float16), None),
        "transformer.h.0.ln_2.bias": (np.zeros(16, dtype=np.float16), None),
        "transformer.ln_f.weight": (np.ones(16, dtype=np.float16), None),
        "transformer.ln_f.bias": (np.zeros(16, dtype=np.float16), None),
        "transformer.h.0.attn.c_attn.weight_h0_query": (
            np.zeros((16, 16), dtype=np.int8),
            np.ones(16, dtype=np.float16),
        ),
        "transformer.h.0.attn.c_attn.weight_h0_key": (
            np.zeros((16, 16), dtype=np.int8),
            np.ones(16, dtype=np.float16),
        ),
        "transformer.h.0.attn.c_attn.weight_h0_value": (
            np.zeros((16, 16), dtype=np.int8),
            np.ones(16, dtype=np.float16),
        ),
        "transformer.h.0.attn.c_proj.weight": (
            np.zeros((16, 16), dtype=np.int8),
            np.ones(16, dtype=np.float16),
        ),
        "transformer.h.0.mlp.c_fc.weight": (
            np.zeros((16, 64), dtype=np.int8),
            np.ones(64, dtype=np.float16),
        ),
        "transformer.h.0.mlp.c_proj.weight": (
            np.zeros((64, 16), dtype=np.int8),
            np.ones(16, dtype=np.float16),
        ),
        "lm_head.weight": (np.zeros((16, 32), dtype=np.int8), np.ones(32, dtype=np.float16)),
    }


def _prescaled_biases():
    return {
        "transformer.h.0.attn.c_proj.bias": np.zeros(16, dtype=np.int32),
        "transformer.h.0.mlp.c_fc.bias": np.zeros(64, dtype=np.int32),
        "transformer.h.0.mlp.c_proj.bias": np.zeros(16, dtype=np.int32),
    }


def test_inject_kv_cache_nodes_prefill_stores_without_runtime_loads():
    result = load_nanogpt(config=_config(), variant="forward_1token")

    graph = inject_kv_cache_nodes(result.graph, result.config, decode=False, seq_len=1)
    ops = [node.op for node in graph.nodes]

    assert ops.count("kv_store") == 2
    assert "kv_load" not in ops
    assert all(not node.attrs.get("decode", False) for node in graph.nodes if node.op == "kv_store")


def test_inject_kv_cache_nodes_decode_loads_before_attention_consumers():
    result = load_nanogpt(config=_config(), variant="forward_1token")

    graph = inject_kv_cache_nodes(result.graph, result.config, decode=True, seq_len=1)
    ops = [node.op for node in graph.nodes]
    qkt = graph.get_node("block0_head0_qkt")
    attn_v = graph.get_node("block0_head0_attn_v")

    assert ops.count("kv_store") == 2
    assert ops.count("kv_load") == 2
    assert qkt.inputs[1] == "block0_head0_key_kv_load"
    assert attn_v.inputs[1] == "block0_head0_value_kv_load"
    assert graph.nodes.index(graph.get_node("block0_head0_key_kv_load")) < graph.nodes.index(qkt)
    assert graph.nodes.index(graph.get_node("block0_head0_value_kv_load")) < graph.nodes.index(attn_v)


def test_decoder_bundle_appends_logits_store_for_lm_head_graphs():
    result = load_nanogpt(config=_config(), variant="forward_1token")

    build = build_decoder_program_bundle(
        prefill_graph=result.graph,
        decode_graph=result.graph,
        weight_data=_weight_data(),
        calibration_scales={},
        prescaled_biases=_prescaled_biases(),
        model_config=result.config,
    )

    assert build.bundle.logits_size == 32
    assert [site.symbol for site in build.prefill_codegen.relocation_sites].count("prefill_logits_offset") == 1
    assert [site.symbol for site in build.decode_codegen.relocation_sites].count("decode_logits_offset") == 1


def test_full_decoder_bundle_records_kv_runtime_config_and_logits_sites():
    result = load_nanogpt(config=_config(), variant="forward_1token")
    runtime_graph = mark_runtime_embedding_lookups(result.graph)
    prefill_graph = inject_kv_cache_nodes(runtime_graph, result.config, decode=False, seq_len=1)
    decode_graph = inject_kv_cache_nodes(runtime_graph, result.config, decode=True, seq_len=3)

    build = build_decoder_program_bundle(
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        weight_data=_weight_data(),
        calibration_scales={},
        prescaled_biases=_prescaled_biases(),
        model_config=result.config,
    )

    prefill_ops = [node.op for node in prefill_graph.nodes]
    decode_ops = [node.op for node in decode_graph.nodes]
    assert prefill_ops.count("kv_store") == 2
    assert "kv_load" not in prefill_ops
    assert decode_ops.count("kv_store") == 2
    assert decode_ops.count("kv_load") == 2
    assert build.decode_codegen.runtime_config_attn_sites
    assert {site.kind for site in build.decode_codegen.runtime_patch_sites} == {
        "token_embed",
        "pos_embed",
        "kv_base",
    }
    assert [site.symbol for site in build.prefill_codegen.relocation_sites].count("prefill_logits_offset") == 1
    assert [site.symbol for site in build.decode_codegen.relocation_sites].count("decode_logits_offset") == 1


def test_decode_kv_load_allocates_padded_tensor_footprint():
    """Each `kv_load` allocation must reserve `M_pad × N_pad × elem_bytes`
    bytes, not just the runtime DMA `xfer_bytes`. Without this, a
    subsequent QUANT_FP32_INT8 could first-fit-pick into the K tile's
    padding-row region and corrupt the kv_load result. See the M2-W8A16
    fix comment in `codegen._emit_kv_load`.

    Verified by intercepting `MemoryAllocator.alloc` and checking the
    sizes recorded for the two `kv_load` allocations. The original
    non-overlap check was too strict — the allocator correctly re-uses
    K's ABUF slot for V once K has been BUF_COPYed to WBUF, so the two
    `LoadInsn.sram_off` values can be equal even when each alloc reserves
    the full padded footprint.
    """
    from taccel.compiler.memory_alloc import BufferAllocator

    result = load_nanogpt(config=_config(), variant="forward_1token")
    runtime_graph = mark_runtime_embedding_lookups(result.graph)
    prefill_graph = inject_kv_cache_nodes(runtime_graph, result.config, decode=False, seq_len=1)
    decode_graph = inject_kv_cache_nodes(runtime_graph, result.config, decode=True, seq_len=3)

    captured: dict[str, int] = {}
    original_alloc = BufferAllocator.alloc

    def alloc_wrapper(self, name, size_bytes, evictable=True):
        if "kv_load" in name and name not in captured:
            captured[name] = int(size_bytes)
        return original_alloc(self, name, size_bytes, evictable)

    BufferAllocator.alloc = alloc_wrapper
    try:
        build = build_decoder_program_bundle(
            prefill_graph=prefill_graph,
            decode_graph=decode_graph,
            weight_data=_weight_data(),
            calibration_scales={},
            prescaled_biases=_prescaled_biases(),
            model_config=result.config,
        )
    finally:
        BufferAllocator.alloc = original_alloc

    kv_loads = [
        insn for insn in build.decode_codegen.instructions
        if isinstance(insn, LoadInsn) and insn.addr_reg == 2 and insn.xfer_len == 3
    ]
    assert len(kv_loads) == 2

    # config has n_embd=16, n_head=1 → d_head=16. seq_len=3 padded to 16
    # rows of d_head=16 cols × 1 byte (INT8) = 256 bytes per padded tile.
    padded_tile_bytes = 16 * 16 * 1
    assert captured["block0_head0_key_kv_load"] >= padded_tile_bytes, (
        f"K kv_load alloc {captured['block0_head0_key_kv_load']} < "
        f"padded {padded_tile_bytes}; padding rows may be corrupted by a "
        f"first-fit-picked downstream INT8 write."
    )
    assert captured["block0_head0_value_kv_load"] >= padded_tile_bytes, (
        f"V kv_load alloc {captured['block0_head0_value_kv_load']} < "
        f"padded {padded_tile_bytes}"
    )


def test_decoder_bundle_reserves_codegen_temp_before_logits_and_kv_cache():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul",
        name="large_fc1",
        inputs=["act", "large.weight"],
        output_shape=(1, 1536),
    ))
    config = ModelConfig(
        name="stage4-temp-layout-test",
        model_kind="decoder",
        n_layer=1,
        n_head=6,
        d_model=384,
        d_head=64,
        mlp_dim=1536,
        vocab_size=32,
        max_seq_len=256,
        embedding_kind="token_pos",
    )

    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,
        weight_data={
            "large.weight": (
                np.zeros((384, 1536), dtype=np.int8),
                np.ones(1536, dtype=np.float16),
            ),
        },
        calibration_scales={"act": 0.125, "large_fc1": 0.25},
        prescaled_biases={},
        model_config=config,
    )

    generated_temp = max(
        build.prefill_codegen.mem.dram_temp_total,
        build.decode_codegen.mem.dram_temp_total,
    )
    assert generated_temp > 0
    assert build.bundle.temp_size == generated_temp
    assert build.bundle.logits_base >= build.bundle.temp_base + build.bundle.temp_size
    assert build.bundle.kv_cache_base >= build.bundle.logits_base + build.bundle.logits_size
