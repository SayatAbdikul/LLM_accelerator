"""Numerical equivalence and resource counts for independently gated schedules."""
from dataclasses import replace

import numpy as np
import pytest

from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import deit_tiny_config
from taccel.compiler.options import CompilerOptions
from taccel.golden_model.simulator import Simulator
from taccel.golden_model.state import MachineState
from taccel.isa.encoding import encode
from taccel.isa.opcodes import Opcode


def _projection_graph(rows=1, heads=2):
    rng = np.random.default_rng(91)
    graph, weights = IRGraph(), {}
    for head in range(heads):
        for projection in ("query", "key", "value"):
            name = f"h{head}_{projection}"
            weights[name] = (rng.integers(-16, 17, (32, 16), dtype=np.int8),
                             np.full(16, .03125, dtype=np.float16))
            graph.add_node(IRNode("matmul", name, ["x", name + "_weight"], (rows, 16),
                                  {"projection": projection}))
            weights[name + "_weight"] = weights.pop(name)
    return graph, weights


def _compile(graph, weights, options=CompilerOptions(), **kwargs):
    cg = CodeGenerator(weights, {}, {}, model_config=deit_tiny_config(), options=options, **kwargs)
    cg.generate(graph)
    return cg


def _execute(cg, rows):
    state = MachineState()
    state.abuf[:] = b"\x35" * len(state.abuf)
    state.dram[:len(cg.dram_blob)] = cg.dram_blob
    alloc = cg.mem.abuf.get("x")
    data = np.random.default_rng(42).normal(size=(rows, 32)).astype(np.float16)
    start = alloc.offset_units * 16
    state.abuf[start:start + data.nbytes] = data.tobytes()
    sim = Simulator(state)
    outputs, scales = [], []
    for insn in cg.instructions:
        sim._execute(insn)
        if insn.opcode == Opcode.MAX_ABS_REDUCE_FP32:
            scales.append(state.scale_regs[insn.sreg:insn.sreg + 2].copy())
        if insn.opcode == Opcode.DEQUANT_ACCUM_FP32_SCALED:
            start = insn.dst_off * 16
            outputs.append(bytes(state.abuf[start:start + rows * 16 * 2]))
    return outputs, scales


@pytest.mark.parametrize("rows", [1, 16, 32])
def test_qkv_cache_keeps_outputs_and_scale_bits(rows):
    graph, weights = _projection_graph(rows)
    baseline = _compile(graph, weights)
    candidate = _compile(graph, weights, CompilerOptions(reuse_qkv_quantization=True))
    expected, old_scales = _execute(baseline, rows)
    actual, new_scales = _execute(candidate, rows)
    assert actual == expected
    assert len(old_scales) == 6 and len(new_scales) == 1
    for scale in old_scales:
        np.testing.assert_array_equal(scale, new_scales[0])
    assert candidate.compilation_metrics()["opcodes"]["QUANT_FP32_INT8"] == 1
    assert not candidate.quantization_cache and not candidate.reserved_sregs
    assert candidate.mem.abuf.high_water_units * 16 <= candidate.mem.abuf.capacity


def test_indexed_liveness_preserves_instruction_and_data_bytes():
    graph, weights = _projection_graph()
    graph.add_node(IRNode("vadd", "sum", ["h0_query", "h0_key"], (1, 16)))
    baseline = _compile(graph, weights)
    candidate = _compile(graph, weights, CompilerOptions(indexed_liveness=True))
    assert [encode(i) for i in candidate.instructions] == [encode(i) for i in baseline.instructions]
    assert candidate.dram_blob == baseline.dram_blob


def test_unknown_operation_and_missing_weight_fail_before_emission():
    graph = IRGraph()
    graph.add_node(IRNode("typo", "bad", [], (1, 16)))
    with pytest.raises(ValueError, match="Unsupported IR"):
        _compile(graph, {})
    graph.nodes[0] = IRNode("matmul", "bad", ["x", "missing"], (1, 16))
    with pytest.raises(KeyError, match="missing matmul weight"):
        _compile(graph, {})


def test_duplicate_nodes_and_retired_options_are_rejected():
    graph = IRGraph()
    graph.add_node(IRNode("gelu", "x", ["external"], (1, 16)))
    with pytest.raises(ValueError, match="Duplicate"):
        graph.add_node(IRNode("gelu", "x", ["external"], (1, 16)))
    with pytest.raises(ValueError, match="gelu_from_accum"):
        _compile(graph, {}, gelu_from_accum=True)


def test_hardware_target_rejects_research_only_features():
    graph = IRGraph()
    graph.add_node(IRNode("softmax", "softmax", ["x"], (1, 16)))
    with pytest.raises(ValueError, match="noncausal"):
        _compile(graph, {}, target="rtl-gen2")
    with pytest.raises(ValueError, match="INT4"):
        _compile(graph, {}, target="rtl-gen2", weight_dtypes={"w": "int4"})


def test_embedding_dtype_mismatch_is_rejected():
    config = replace(deit_tiny_config(), model_kind="decoder", embedding_kind="token_pos")
    graph = IRGraph()
    graph.add_node(IRNode("embed_lookup", "emb", [], (1, config.d_model)))
    cg = CodeGenerator({"transformer.wte.weight": (np.zeros((16, config.d_model), np.int8), None)},
                       {}, {}, model_config=config)
    with pytest.raises(ValueError, match="float16"):
        cg.generate(graph)


def test_hardware_graph_requires_explicit_external_activation_binding():
    graph, weights = _projection_graph()
    with pytest.raises(ValueError, match="undeclared external activation 'x'"):
        _compile(graph, weights, target="rtl-gen2")
    # A parameter with the same name cannot initialize an activation buffer.
    weights["x"] = (np.zeros((16, 32), np.int8), np.ones(32, np.float16))
    with pytest.raises(ValueError, match="undeclared external activation 'x'"):
        _compile(graph, weights, target="rtl-gen2")
    research = _compile(graph, weights)
    explicit = _compile(graph, weights, target="rtl-gen2", external_inputs={"x"})
    assert research.dram_blob == explicit.dram_blob
    assert [encode(i) for i in research.instructions] == [encode(i) for i in explicit.instructions]
    with pytest.raises(ValueError, match="undeclared external"):
        _compile(graph, weights, external_inputs=set())
    for invalid in ("x", ["x"], {1}, {""}):
        with pytest.raises(ValueError, match="external_inputs must"):
            _compile(graph, weights, external_inputs=invalid)


def test_public_decoder_rejects_misspelled_activation_before_weight_layout(monkeypatch):
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    graph, weights = _projection_graph()

    def forbidden_layout(*args):
        pytest.fail("invalid graph reached weight staging")

    monkeypatch.setattr(CodeGenerator, "_layout_weights", forbidden_layout)
    with pytest.raises(ValueError, match="undeclared external activation"):
        build_decoder_program_bundle(
            prefill_graph=graph, decode_graph=graph, weight_data=weights,
            calibration_scales={}, prescaled_biases={},
            model_config=replace(deit_tiny_config(), model_kind="decoder", embedding_kind="token_pos"))


@pytest.mark.parametrize("op", ["kv_store", "logits_store"])
def test_complete_physical_store_source_binding_is_an_explicit_input(op):
    from taccel.compiler.validation import validate_graph
    from taccel.isa.opcodes import BUF_ABUF
    cg = CodeGenerator({}, {}, {}, model_config=deit_tiny_config(), target="rtl-gen2")
    for attrs in ({}, {"src_buf": BUF_ABUF}, {"src_off_units": 0}):
        graph = IRGraph()
        graph.add_node(IRNode(op, "store", ["manual_source"], (1, 16), attrs))
        with pytest.raises(ValueError, match="undeclared external activation 'manual_source'"):
            validate_graph(cg, graph)
    graph = IRGraph()
    graph.add_node(IRNode(op, "store", ["manual_source"], (1, 16),
                          {"src_buf": BUF_ABUF, "src_off_units": 0}))
    validate_graph(cg, graph)
    # Only inputs[0] is physically bound. Other missing dependencies cannot
    # hide behind the attributes, nor can an unrelated emitter use them.
    graph.nodes[0].inputs.append("unexpected_dependency")
    with pytest.raises(ValueError, match="undeclared external activation 'unexpected_dependency'"):
        validate_graph(cg, graph)
    graph.nodes[0] = IRNode("gelu", "output", ["manual_source"], (1, 16),
                            {"src_buf": BUF_ABUF, "src_off_units": 0})
    with pytest.raises(ValueError, match="undeclared external activation 'manual_source'"):
        validate_graph(cg, graph)


@pytest.mark.parametrize("shape", [(True, 16), (1, True), (np.bool_(True), 16)])
def test_boolean_shape_dimensions_are_not_integer_dimensions(shape):
    graph = IRGraph()
    graph.add_node(IRNode("gelu", "output", ["external"], shape))
    with pytest.raises(ValueError, match="positive 2D"):
        _compile(graph, {})


def test_bias_must_cover_logical_not_padded_columns():
    graph = IRGraph()
    graph.add_node(IRNode("matmul", "output", ["external", "w"], (1, 17), {"bias": "b"}))
    weights = {"w": (np.zeros((16, 32), np.int8), np.ones(32, np.float16))}
    with pytest.raises(ValueError, match="logical output columns"):
        _compile(graph, weights, biases={"b": np.zeros(16, np.float32)})
    _compile(graph, weights, biases={"b": np.zeros(17, np.float32)})
    # Special values remain valid research conversion inputs, not sanitised.
    _compile(graph, weights, biases={"b": np.full(17, np.inf, np.float32)})


def test_mutated_node_list_uses_current_producer_shape():
    graph = IRGraph()
    graph.add_node(IRNode("gelu", "input", ["external"], (1, 16)))
    graph.add_node(IRNode("matmul", "output", ["input", "w"], (1, 16)))
    graph.nodes[0] = IRNode("gelu", "input", ["external"], (1, 32))
    weights = {"w": (np.zeros((16, 16), np.int8), np.ones(16, np.float16))}
    with pytest.raises(ValueError, match="activation and weight shapes"):
        _compile(graph, weights)


def _streamed_fc2(options, rows=1, K=512, N=1024):
    from taccel.compiler.w8a16_emit.matmul import emit_matmul_w8a16
    from taccel.isa.instructions import HaltInsn
    from taccel.compiler.tiler import pad_dim

    rng = np.random.default_rng(170)
    graph = IRGraph()
    node = graph.add_node(IRNode("matmul", "fc2", ["x", "w"], (rows, N), {"bias": "bias"}))
    weights = {"w": (rng.integers(-8, 9, (K, N), dtype=np.int8),
                     np.full(N, 0.015625, dtype=np.float16))}
    cg = CodeGenerator(weights, {}, {}, model_config=deit_tiny_config(), options=options,
                       biases={"bias": np.linspace(-.25, .25, N, dtype=np.float32)})
    cg._layout_weights(graph)
    x = rng.normal(size=(pad_dim(rows), K)).astype(np.float16)
    # Deliberately dirty padding verifies that real-row loading still clears it.
    x[rows:] = 1000
    input_offset = len(cg.dram_blob)
    cg.dram_temp_outputs["x"] = input_offset
    cg.dram_temp_fp32_outputs["x"] = x.nbytes
    cg.dram_temp_start = input_offset + x.nbytes
    emit_matmul_w8a16(cg, node)
    cg._emit(HaltInsn())
    state = MachineState()
    state.abuf[:] = b"\x35" * len(state.abuf)
    state.dram[:len(cg.dram_blob)] = cg.dram_blob
    state.dram[input_offset:input_offset + x.nbytes] = x.tobytes()
    sim = Simulator(state)
    scales = []
    for insn in cg.instructions:
        sim._execute(insn)
        if insn.opcode == Opcode.MAX_ABS_REDUCE_FP32:
            scales.append(state.scale_regs[insn.sreg:insn.sreg + 2].copy())
    start = cg.dram_temp_outputs["fc2"]
    return cg, bytes(state.dram[start:start + rows * N * 2]), scales


@pytest.mark.parametrize("rows", [1, 16])
def test_fc2_real_rows_and_quantization_reuse_preserve_partial_order(rows):
    baseline, output, old_scales = _streamed_fc2(CompilerOptions(), rows)
    row_only, row_output, _ = _streamed_fc2(CompilerOptions(real_row_fc2_loads=True), rows)
    candidate, actual, new_scales = _streamed_fc2(
        CompilerOptions(real_row_fc2_loads=True, reuse_fc2_quantization=True), rows)
    assert actual == row_output == output
    assert len(old_scales) == 4 and len(new_scales) == 1
    for i, scale in enumerate(old_scales):
        np.testing.assert_array_equal(scale, new_scales[0])
    assert candidate.compilation_metrics()["dma_load_bytes"] < baseline.compilation_metrics()["dma_load_bytes"]
    if rows == 1:
        assert row_only.compilation_metrics()["dma_load_bytes"] < baseline.compilation_metrics()["dma_load_bytes"]
    assert not candidate.reserved_sregs
    assert candidate.mem.abuf.high_water_units * 16 <= candidate.mem.abuf.capacity


@pytest.mark.parametrize("rows,retained", [(1, 6), (16, 6), (32, 5)])
def test_fc2_partial_cache_preserves_multi_k_rounding_and_single_bias(rows, retained):
    baseline, output, scales = _streamed_fc2(CompilerOptions(), rows=rows, K=3072, N=768)
    candidate, actual, new_scales = _streamed_fc2(
        CompilerOptions(real_row_fc2_loads=True, reuse_fc2_quantization=True), rows=rows, K=3072, N=768)
    assert actual == output
    assert len(scales) == 24 and len(new_scales) == 24 - retained
    # The first N group visits every K partition; subsequent groups omit only
    # the retained prefix. Runtime scale bits and remaining order are exact.
    for expected, observed in zip(scales[:12] + scales[12 + retained:], new_scales):
        np.testing.assert_array_equal(expected, observed)
    admission = [event for event in candidate.optimization_events if event["kind"] == "fc2_cache_admission"]
    assert len(admission) == 1 and admission[0]["cached_partitions"] == retained
    assert admission[0]["working_bound_bytes"] <= admission[0]["largest_free_bytes"]
    assert sum(event["kind"] == "fc2_quantization_hit" for event in candidate.optimization_events) == retained
    assert not any("fallback" in event["kind"] for event in candidate.optimization_events)
    assert candidate.compilation_metrics()["dma_load_bytes"] < baseline.compilation_metrics()["dma_load_bytes"]
    assert not candidate.reserved_sregs


def test_pruning_unreferenced_weight_representations_preserves_results():
    baseline, output, _ = _streamed_fc2(CompilerOptions())
    candidate, actual, _ = _streamed_fc2(CompilerOptions(deduplicate_data=True))
    assert actual == output
    assert "w" in baseline.dram_layout and "w" not in candidate.dram_layout
    assert len(candidate.dram_blob) < len(baseline.dram_blob) - 512 * 1024


def test_exact_wbuf_weight_falls_back_to_safe_tiling_with_identical_output():
    from taccel.isa.opcodes import WBUF_SIZE
    baseline, expected, _ = _streamed_fc2(CompilerOptions(), K=512, N=512)
    candidate, actual, _ = _streamed_fc2(
        CompilerOptions(real_row_fc2_loads=True, reuse_fc2_quantization=True,
                        deduplicate_data=True), K=512, N=512)
    assert actual == expected
    assert "w" not in candidate.dram_layout
    for cg in (baseline, candidate):
        assert any("__stage4_tile_" in name for name in cg.dram_layout)
        assert cg.mem.wbuf.high_water_units * 16 <= WBUF_SIZE


def test_wbuf_epilogue_budget_and_malformed_weight_columns():
    from taccel.compiler.codegen import matmul_wbuf_bytes
    from taccel.isa.opcodes import WBUF_SIZE
    assert matmul_wbuf_bytes(512 * 512, 512) > WBUF_SIZE
    assert matmul_wbuf_bytes(512 * 496, 496) <= WBUF_SIZE
    graph = IRGraph()
    graph.add_node(IRNode("matmul", "y", ["x", "w"], (1, 496)))
    weights = {"w": (np.zeros((512, 496), np.int8), np.ones(496, np.float16))}
    cg = _compile(graph, weights)
    assert not any("__stage4_tile_" in name for name in cg.dram_layout)
    assert cg.mem.wbuf.high_water_units * 16 == matmul_wbuf_bytes(512 * 496, 496)
    weights["w"] = (np.zeros((512, 512), np.int8), np.ones(512, np.float16))
    with pytest.raises(ValueError, match="weight columns"):
        _compile(graph, weights)


def test_bundle_shared_layout_is_built_once_with_identical_bytes(monkeypatch):
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    config = replace(deit_tiny_config(), model_kind="decoder", embedding_kind="token_pos",
                     d_model=32, n_head=2, d_head=16)
    graph, weights = _projection_graph()
    closed = IRGraph()
    closed.add_node(IRNode("embed_lookup", "x", [], (1, 32)))
    for node in graph.nodes:
        closed.add_node(node)
    graph = closed
    weights["transformer.wte.weight"] = (np.zeros((16, 32), np.float16), None)
    kwargs = dict(prefill_graph=graph, decode_graph=graph, weight_data=weights,
                  calibration_scales={}, prescaled_biases={}, model_config=config)
    baseline = build_decoder_program_bundle(**kwargs)
    calls = []
    original = CodeGenerator._layout_weights

    def counted(self, g):
        calls.append(self.stream_name)
        original(self, g)

    monkeypatch.setattr(CodeGenerator, "_layout_weights", counted)
    candidate = build_decoder_program_bundle(**kwargs, compiler_options=CompilerOptions(shared_data_layout=True))
    assert calls == ["prefill"]
    assert candidate.bundle.prefill_instrs == baseline.bundle.prefill_instrs
    assert candidate.bundle.decode_instrs == baseline.bundle.decode_instrs
    assert candidate.bundle.shared_data == baseline.bundle.shared_data
    assert candidate.prefill_codegen.dram_blob is candidate.decode_codegen.dram_blob


def test_data_pruning_retains_canonical_weight_used_by_either_stream():
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    config = replace(deit_tiny_config(), model_kind="decoder", embedding_kind="token_pos",
                     d_model=256, n_head=16, d_head=16)
    prefill, decode = IRGraph(), IRGraph()
    prefill.add_node(IRNode("embed_lookup", "x", [], (64, 256)))
    decode.add_node(IRNode("embed_lookup", "x", [], (1, 256)))
    prefill.add_node(IRNode("matmul", "projection", ["x", "w"], (64, 512)))
    decode.add_node(IRNode("matmul", "projection", ["x", "w"], (1, 512)))
    weights = {"w": (np.zeros((256, 512), np.int8), np.ones(512, np.float16))}
    weights["transformer.wte.weight"] = (np.zeros((16, 256), np.float16), None)
    result = build_decoder_program_bundle(
        prefill_graph=prefill, decode_graph=decode, weight_data=weights,
        calibration_scales={}, prescaled_biases={}, model_config=config,
        compiler_options=CompilerOptions(shared_data_layout=True, deduplicate_data=True),
    )
    assert "w" in result.prefill_codegen.dram_layout
    assert "w" in result.decode_codegen.dram_layout
    assert any("__stage4_tile_" in name for name in result.prefill_codegen.dram_layout)


def test_scale_register_reservation_survives_interleaved_allocators():
    graph, weights = _projection_graph()
    cg = _compile(graph, weights)
    cg.reserved_sregs.update({14, 15})
    assert all(cg._alloc_sreg_quad() in {0, 4, 8} for _ in range(16))
    cg.reserved_sregs.update(range(0, 12))
    assert cg._alloc_sreg_pair() == 12


def test_qkv_cache_falls_back_under_interleaved_matmul_pressure():
    graph = IRGraph()
    weights = {}
    for name, source, K, N, projection in (
        ("q", "x", 1024, 16, "query"), ("middle", "z", 1792, 64, None),
        ("k", "x", 1024, 16, "key"),
    ):
        graph.add_node(IRNode("matmul", name, [source, "w" + name], (16, N),
                              {"projection": projection} if projection else {}))
        weights["w" + name] = (np.zeros((K, N), np.int8), np.ones(N, np.float16))
    baseline = _compile(graph, weights)
    candidate = _compile(graph, weights, CompilerOptions(reuse_qkv_quantization=True))
    assert baseline.compilation_metrics()["opcodes"]["QUANT_FP32_INT8"] == 3
    assert candidate.compilation_metrics()["opcodes"]["QUANT_FP32_INT8"] == 3
    assert [encode(i) for i in candidate.instructions] == [encode(i) for i in baseline.instructions]
    assert any(e["kind"] == "quantization_reuse_fallback" for e in candidate.optimization_events)
    assert not candidate.quantization_cache and not candidate.reserved_sregs


def test_fc2_cache_accounts_for_uncached_scratch_and_falls_back():
    from taccel.compiler.w8a16_emit.matmul import emit_matmul_w8a16
    graph = IRGraph()
    node = graph.add_node(IRNode("matmul", "fc2", ["x", "w"], (16, 384)))
    weights = {"w": (np.zeros((5376, 384), np.int8), np.ones(384, np.float16))}
    results = []
    for enabled in (False, True):
        cg = CodeGenerator(weights, {}, {}, model_config=deit_tiny_config(), stage4_m_pad=80,
                           options=CompilerOptions(reuse_fc2_quantization=enabled))
        cg._layout_weights(graph)
        # Baseline work fits the remaining 48 KiB, but even a one-partition
        # cache plus its explicit uncached scratch requires 54 KiB.
        cg.mem.abuf.alloc("other_live", 80 * 1024)
        cg.dram_temp_outputs["x"] = len(cg.dram_blob)
        cg.dram_temp_fp32_outputs["x"] = 16 * 5376 * 2
        cg.dram_temp_start += 16 * 5376 * 2
        emit_matmul_w8a16(cg, node)
        results.append(cg)
    assert [encode(i) for i in results[0].instructions] == [encode(i) for i in results[1].instructions]
    assert any(e["kind"] == "fc2_reuse_fallback" for e in results[1].optimization_events)
    assert not results[1].reserved_sregs


def test_fc2_cache_falls_back_for_fragmented_free_space():
    from taccel.compiler.w8a16_emit.matmul import emit_matmul_w8a16
    graph = IRGraph()
    node = graph.add_node(IRNode("matmul", "fc2", ["x", "w"], (16, 768)))
    weights = {"w": (np.zeros((3072, 768), np.int8), np.ones(768, np.float16))}
    results = []
    for enabled in (False, True):
        cg = CodeGenerator(weights, {}, {}, model_config=deit_tiny_config(),
                           options=CompilerOptions(reuse_fc2_quantization=enabled))
        cg._layout_weights(graph)
        for index, size_kib in enumerate((20, 20, 20, 8)):
            cg.mem.abuf.alloc(f"hole{index}", size_kib * 1024)
            cg.mem.abuf.alloc(f"live{index}", 8 * 1024)
        cg.mem.abuf.alloc("hole4", 4 * 1024)
        cg.mem.abuf.alloc("tail", 24 * 1024)
        for index in range(5):
            cg.mem.abuf.free(f"hole{index}")
        assert cg.mem.abuf.free_bytes == 72 * 1024
        assert cg.mem.abuf.largest_free_bytes == 20 * 1024
        cg.dram_temp_outputs["x"] = len(cg.dram_blob)
        cg.dram_temp_fp32_outputs["x"] = 16 * 3072 * 2
        cg.dram_temp_start += 16 * 3072 * 2
        emit_matmul_w8a16(cg, node)
        results.append(cg)
    assert [encode(i) for i in results[0].instructions] == [encode(i) for i in results[1].instructions]
    assert any(e["kind"] == "fc2_reuse_fallback" for e in results[1].optimization_events)


def test_direct_hardware_emitter_rejects_unsupported_flags():
    from taccel.isa.instructions import SetScaleInsn, SoftmaxFp32Insn
    cg = CodeGenerator({}, {}, {}, model_config=deit_tiny_config(), target="rtl-gen2")
    with pytest.raises(ValueError, match="immediate mode"):
        cg._emit(SetScaleInsn(src_mode=1))
    with pytest.raises(ValueError, match="SOFTMAX_FP32"):
        cg._emit(SoftmaxFp32Insn(flags=1))
