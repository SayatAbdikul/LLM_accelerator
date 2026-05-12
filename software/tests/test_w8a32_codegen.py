"""W8A32 codegen lowering tests (Phase 3 (c.1), milestone M2).

Scope of M2 (kept narrow — see `software/taccel/compiler/w8a32_emit.py`
docstring):

  - Sub-layer ops emit FP32 variants when `CodeGenerator(w8a32_enabled=True)`:
      LAYERNORM_FP32, GELU_FP32, SOFTMAX_FP32, MASKED_SOFTMAX_FP32, VADD_FP32
  - W8A8 optimization blocks (dequant_add residual, requant_pc,
    gelu_from_accum, fused_softmax_attnv) are force-disabled in W8A32 mode.
  - `evaluate_gpt2_perplexity`'s W8A32 preset (`weight_only_int8`) does NOT
    yet dispatch to the new codegen — its golden path still goes through
    `WeightOnlyHostRunner` (Phase 3 (b)). End-to-end golden via the
    simulator-backed bundle is M3+M4.

What this file does NOT test (deferred to M2.5):

  - Matmul-output dequant (REQUANT_PC -> DEQUANT_ACCUM_FP32). Matmul
    in W8A32 mode currently still emits the INT8 REQUANT epilogue;
    the simulator will execute that correctly but the activations
    will be INT8 between layers, not FP32. M2.5 will fix this by
    inserting DEQUANT_ACCUM_FP32 after each matmul + QUANT_FP32_INT8
    before each next-matmul input.
"""
from __future__ import annotations

import pytest

from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import deit_tiny_config
from taccel.compiler.w8a32_emit import (
    FP32_BYTES_PER_ELEM,
    emit_gelu_fp32,
    emit_layernorm_fp32,
    emit_softmax_fp32,
    emit_vadd_fp32,
)
from taccel.isa.opcodes import Opcode


def _fresh_codegen(*, w8a32: bool = False) -> CodeGenerator:
    """Build a bare CodeGenerator with no weights/scales for isolated tests."""
    return CodeGenerator(
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=w8a32,
    )


def _reset_cg(cg: CodeGenerator) -> None:
    """Clear emitted instructions and free all ABUF/WBUF allocations.

    Used between sub-tests so each test gets a clean slate without
    rebuilding the CodeGenerator (which is cheap but noisy).
    """
    cg.instructions.clear()
    cg.mem.abuf.allocations.clear()
    cg.mem.abuf._free = [(0, cg.mem.abuf.capacity_units)]
    cg.mem.wbuf.allocations.clear()
    cg.mem.wbuf._free = [(0, cg.mem.wbuf.capacity_units)]


# ---------------------------------------------------------------------------
# CodeGenerator constructor: w8a32_enabled flag + force-disable W8A8 opts
# ---------------------------------------------------------------------------


def test_w8a32_enabled_default_is_false():
    cg = _fresh_codegen()
    assert cg.w8a32_enabled is False


def test_w8a32_enabled_true_force_disables_w8a8_opts():
    """In W8A32 mode all W8A8 optimization blocks must be empty/disabled.

    These opts (dequant_add, requant_pc, gelu_from_accum, fused
    softmax_attnv) are optimizations of the INT8 round-trip that W8A32
    doesn't have. Letting them remain set would route emission into
    W8A8-optimized branches that produce broken bundles in W8A32 mode.
    """
    cg = CodeGenerator(
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        stream_name="prefill",
        # Try to set every W8A8 optimization simultaneously.
        gelu_from_accum=True,
        gelu_from_accum_blocks={0, 1, 2},
        dequant_add_residual1_blocks={0, 1},
        dequant_add_residual2_blocks={0, 1},
        fused_softmax_attnv_blocks={0},
        fused_softmax_attnv_accum_out_proj_blocks={0},
        requant_pc_weight_names={"some_weight"},
        requant_pc_scale_tables={"some_weight": None},
        w8a32_enabled=True,
    )
    assert cg.w8a32_enabled is True
    assert cg.gelu_from_accum is False
    assert cg.gelu_from_accum_blocks == set()
    assert cg.dequant_add_residual1_blocks == set()
    assert cg.dequant_add_residual2_blocks == set()
    assert cg.fused_softmax_attnv_blocks == set()
    assert cg.fused_softmax_attnv_accum_out_proj_blocks == set()
    assert cg.requant_pc_weight_names == set()
    assert cg.requant_pc_scale_tables == {}


# ---------------------------------------------------------------------------
# Per-op emission helpers
# ---------------------------------------------------------------------------


def test_emit_gelu_fp32_emits_gelu_fp32_opcode():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="gelu", name="gelu_n", inputs=["gelu_in"], output_shape=(16, 16), attrs={}
    )
    emit_gelu_fp32(cg, node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.GELU_FP32 in opcodes
    # No INT8 GELU should appear.
    assert Opcode.GELU not in opcodes


def test_emit_vadd_fp32_emits_vadd_fp32_opcode():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="vadd", name="vadd_n", inputs=["a", "b"], output_shape=(16, 16), attrs={}
    )
    emit_vadd_fp32(cg, node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.VADD_FP32 in opcodes
    assert Opcode.VADD not in opcodes
    assert Opcode.DEQUANT_ADD not in opcodes


def test_emit_softmax_fp32_non_masked():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="softmax", name="sm", inputs=["sm_in"], output_shape=(16, 16), attrs={}
    )
    emit_softmax_fp32(cg, node, masked=False)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.SOFTMAX_FP32 in opcodes
    assert Opcode.MASKED_SOFTMAX_FP32 not in opcodes
    assert Opcode.SOFTMAX not in opcodes


def test_emit_softmax_fp32_masked():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="softmax",
        name="msm",
        inputs=["msm_in"],
        output_shape=(16, 16),
        attrs={"causal": True},
    )
    emit_softmax_fp32(cg, node, masked=True)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.MASKED_SOFTMAX_FP32 in opcodes
    assert Opcode.SOFTMAX_FP32 not in opcodes
    assert Opcode.MASKED_SOFTMAX not in opcodes


def test_emit_fp32_helpers_allocate_4x_int8_size():
    """FP32 tile allocations must use 4 bytes/element (vs 1 for INT8).

    Verifies the M1 design contract holds end-to-end at the codegen
    level: a [16, 16] FP32 tile occupies 16*16*4 = 1024 bytes = 64
    16-byte addressing units.
    """
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="gelu", name="gelu_test", inputs=["gin"], output_shape=(16, 16), attrs={}
    )
    emit_gelu_fp32(cg, node)
    # The output node's allocation must be 64 units (=1024B, FP32 16x16).
    out_alloc = cg.mem.abuf.allocations["gelu_test"]
    assert out_alloc.size_bytes == 16 * 16 * FP32_BYTES_PER_ELEM == 1024
    assert out_alloc.size_units == 64


# ---------------------------------------------------------------------------
# CodeGenerator._emit_node dispatch: w8a32_enabled flag routes correctly
# ---------------------------------------------------------------------------


def test_emit_node_layernorm_dispatches_to_fp32_when_w8a32():
    """Calling the main CodeGenerator's `_emit_layernorm` in W8A32 mode
    must produce LAYERNORM_FP32, not the INT8 LAYERNORM."""
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="layernorm",
        name="ln_test",
        # LN expects (input, gamma, beta) inputs but we don't load real
        # weight_data in this test — the helper skips DMA emission when
        # gamma_data is None, leaving just CONFIG_TILE + LAYERNORM_FP32 + SYNC.
        inputs=["ln_in", "ln_gamma", "ln_beta"],
        output_shape=(16, 16),
        attrs={},
    )
    cg._emit_layernorm(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.LAYERNORM_FP32 in opcodes
    assert Opcode.LAYERNORM not in opcodes


def test_emit_node_gelu_dispatches_to_fp32_when_w8a32():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(op="gelu", name="gelu_test", inputs=["gin"], output_shape=(16, 16), attrs={})
    cg._emit_gelu(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.GELU_FP32 in opcodes
    assert Opcode.GELU not in opcodes


def test_emit_node_vadd_dispatches_to_fp32_when_w8a32():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(op="vadd", name="vadd_test", inputs=["a", "b"], output_shape=(16, 16), attrs={})
    cg._emit_vadd(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.VADD_FP32 in opcodes
    assert Opcode.VADD not in opcodes


def test_emit_node_softmax_dispatches_to_fp32_when_w8a32():
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(op="softmax", name="sm_test", inputs=["sm_in"], output_shape=(16, 16), attrs={})
    cg._emit_softmax(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.SOFTMAX_FP32 in opcodes
    assert Opcode.SOFTMAX not in opcodes


def test_emit_node_softmax_dispatches_to_masked_fp32_for_causal():
    """When the node has `attrs={"causal": True}` the W8A32 dispatch
    must use MASKED_SOFTMAX_FP32 (not the plain variant)."""
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="softmax",
        name="csm_test",
        inputs=["csm_in"],
        output_shape=(16, 16),
        attrs={"causal": True},
    )
    cg._emit_softmax(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.MASKED_SOFTMAX_FP32 in opcodes
    assert Opcode.SOFTMAX_FP32 not in opcodes


# ---------------------------------------------------------------------------
# Regression: when w8a32_enabled=False, the existing INT8 path is unchanged
# ---------------------------------------------------------------------------


def test_int8_layernorm_emission_unchanged_when_w8a32_disabled():
    """Without w8a32_enabled, _emit_layernorm must still emit LAYERNORM
    (INT8), not LAYERNORM_FP32. Guards against the dispatch hook
    accidentally activating in W8A8 mode."""
    cg = _fresh_codegen(w8a32=False)
    node = IRNode(
        op="layernorm",
        name="ln_int8",
        inputs=["ln_in", "ln_gamma", "ln_beta"],
        output_shape=(16, 16),
        attrs={},
    )
    cg._emit_layernorm(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.LAYERNORM in opcodes
    assert Opcode.LAYERNORM_FP32 not in opcodes


def test_int8_gelu_emission_unchanged_when_w8a32_disabled():
    cg = _fresh_codegen(w8a32=False)
    node = IRNode(op="gelu", name="gelu_int8", inputs=["gin"], output_shape=(16, 16), attrs={})
    cg._emit_gelu(node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.GELU in opcodes
    assert Opcode.GELU_FP32 not in opcodes


# ---------------------------------------------------------------------------
# decoder_bundle plumbing
# ---------------------------------------------------------------------------


def test_w8a32_generate_refuses_graph_with_matmul_until_m2_5():
    """M2 guardrail: a W8A32 codegen must NotImplementedError when asked
    to emit a graph that contains matmul nodes, because the matmul
    epilogue still emits INT8 REQUANT_PC (M2.5 work) while the
    sub-layer ops now read FP32. That combination produces a silently
    malformed bundle; loud-fail catches it at codegen time."""
    cg = _fresh_codegen(w8a32=True)
    graph = IRGraph()
    graph.add_node(
        IRNode(
            op="matmul",
            name="m",
            inputs=["x", "w"],
            output_shape=(16, 16),
            attrs={},
        )
    )
    with pytest.raises(NotImplementedError, match="M2.5"):
        cg.generate(graph)


def test_w8a32_generate_accepts_graph_without_matmul():
    """A W8A32 codegen must successfully generate a graph composed of
    only sub-layer ops (LN/GELU/softmax/VADD/embedding/etc.). This is
    a degenerate case — real transformer graphs always have matmuls —
    but it exercises the guardrail's negative case so the test stays
    honest about what's exempted vs blocked."""
    cg = _fresh_codegen(w8a32=True)
    graph = IRGraph()
    graph.add_node(
        IRNode(
            op="gelu",
            name="standalone_gelu",
            inputs=["input"],
            output_shape=(16, 16),
            attrs={},
        )
    )
    insns, _ = cg.generate(graph)
    opcodes = [insn.opcode for insn in insns]
    assert Opcode.GELU_FP32 in opcodes


def test_decoder_bundle_w8a32_flag_plumbed_to_codegen():
    """`build_decoder_program_bundle(w8a32_enabled=True)` must propagate
    the flag to both prefill and decode CodeGenerators. We don't try to
    actually build a working bundle here (that requires full weights
    + frontend graph machinery, which is M3 territory) — just verify
    the keyword argument is accepted."""
    from inspect import signature

    from taccel.compiler.decoder_bundle import build_decoder_program_bundle

    sig = signature(build_decoder_program_bundle)
    assert "w8a32_enabled" in sig.parameters
    assert sig.parameters["w8a32_enabled"].default is False
