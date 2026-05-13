"""W8A32 codegen lowering tests (Phase 3 (c.1), milestones M2 + M2.5-B).

Scope (kept narrow — see `software/taccel/compiler/w8a32_emit.py` docstring):

  - Sub-layer ops emit FP32 variants when `CodeGenerator(w8a32_enabled=True)`:
      LAYERNORM_FP32, GELU_FP32, SOFTMAX_FP32, MASKED_SOFTMAX_FP32, VADD_FP32
  - `matmul` op (M2.5-B) lowers to MAX_ABS_REDUCE_FP32 + QUANT_FP32_INT8 +
    MATMUL + DEQUANT_ACCUM_FP32_SCALED via `emit_matmul_w8a32`. The
    `matmul_qkt` and `matmul_attn_v` (attention internals) are still
    blocked by the `generate()` guardrail — those are M3.
  - W8A8 optimization blocks (dequant_add residual, requant_pc,
    gelu_from_accum, fused_softmax_attnv) are force-disabled in W8A32 mode.
  - `evaluate_gpt2_perplexity`'s W8A32 preset (`weight_only_int8`) does NOT
    yet dispatch to the new codegen — its golden path still goes through
    `WeightOnlyHostRunner` (Phase 3 (b)). End-to-end golden via the
    simulator-backed bundle is M3+M4.

What this file does NOT test (deferred to M3):

  - Strip-mined / large-weight-tiled / fused-out-proj-accum matmul
    variants in W8A32 mode. `emit_matmul_w8a32` raises
    NotImplementedError if the sizing thresholds force a non-simple
    lowering.
  - Embedding → FP32 boundary handling. Tests of `emit_matmul_w8a32`
    seed the input ABUF tile directly (FP32) rather than feeding it
    from the INT8 embedding lookup.
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


@pytest.mark.parametrize("blocked_op", ["matmul_qkt", "matmul_attn_v"])
def test_w8a32_generate_refuses_attention_matmuls_until_m3(blocked_op):
    """M2.5-B guardrail: attention-internal matmul ops still raise. The
    plain `matmul` op was unblocked in M2.5-B (lowered through
    `emit_matmul_w8a32`); QKT and ATTN_V remain INT8-output until M3."""
    cg = _fresh_codegen(w8a32=True)
    graph = IRGraph()
    graph.add_node(
        IRNode(
            op=blocked_op,
            name=blocked_op,
            inputs=["x", "w"],
            output_shape=(16, 16),
            attrs={},
        )
    )
    with pytest.raises(NotImplementedError, match="M3"):
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


# ---------------------------------------------------------------------------
# M2.5-B: emit_matmul_w8a32 — dynamic-scale prelude + DEQUANT epilogue
# ---------------------------------------------------------------------------


import numpy as np  # noqa: E402  - introduced for M2.5-B test fixtures

from taccel.compiler.w8a32_emit import emit_matmul_w8a32  # noqa: E402


def _matmul_w8a32_codegen(
    *,
    weight_name: str = "w0",
    weight_shape=(16, 16),
    bias_name: str | None = None,
    bias_vec: np.ndarray | None = None,
) -> tuple[CodeGenerator, IRGraph, IRNode]:
    """Build a CodeGenerator + 1-node IR graph for an isolated matmul.

    Stages a fake INT8 weight + its per-channel FP16 scales, plus
    optional FP32 bias. The codegen's DRAM-staging loop runs inside
    `generate()`, so callers should invoke `cg.generate(graph)` to
    materialize the DRAM symbols before any direct `emit_matmul_w8a32`
    call. For unit tests that only inspect emitted opcodes (not
    actually execute on the simulator), the staging is sufficient.
    """
    K, N = weight_shape
    rng = np.random.default_rng(42)
    w_int8 = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    w_scales = (np.abs(rng.standard_normal(N)) * 0.001 + 1e-4).astype(np.float16)
    weight_data = {weight_name: (w_int8, w_scales)}

    fp32_biases: dict[str, np.ndarray] = {}
    if bias_name is not None:
        fp32_biases[bias_name] = (
            bias_vec
            if bias_vec is not None
            else rng.standard_normal(N).astype(np.float32)
        )

    cg = CodeGenerator(
        weight_data=weight_data,
        calibration_scales={"x_fp32": 1.0, "m_out": 1.0},
        prescaled_biases={},
        fp32_biases=fp32_biases,
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=True,
    )

    graph = IRGraph()
    attrs: dict = {}
    if bias_name is not None:
        attrs["bias"] = bias_name
    node = IRNode(
        op="matmul",
        name="m_out",
        inputs=["x_fp32", weight_name],
        output_shape=(K, N),
        attrs=attrs,
    )
    graph.add_node(node)
    return cg, graph, node


def test_emit_matmul_w8a32_emits_full_prelude_and_epilogue():
    """The M2.5-B lowering must emit MAX_ABS_REDUCE_FP32 → QUANT_FP32_INT8
    → MATMUL → DEQUANT_ACCUM_FP32_SCALED, in that order, with the M1
    INT8 REQUANT/REQUANT_PC opcodes absent."""
    cg, graph, _ = _matmul_w8a32_codegen()
    insns, _ = cg.generate(graph)
    opcodes = [insn.opcode for insn in insns]

    # Positive: every M2.5-A primitive is present.
    assert Opcode.MAX_ABS_REDUCE_FP32 in opcodes
    assert Opcode.QUANT_FP32_INT8 in opcodes
    assert Opcode.MATMUL in opcodes
    assert Opcode.DEQUANT_ACCUM_FP32_SCALED in opcodes

    # Negative: the INT8 REQUANT epilogue and M1's plain DEQUANT do not appear.
    assert Opcode.REQUANT not in opcodes
    assert Opcode.REQUANT_PC not in opcodes
    assert Opcode.DEQUANT_ACCUM_FP32 not in opcodes

    # Ordering: prelude before MATMUL before DEQUANT_SCALED.
    max_abs_pc = next(i for i, op in enumerate(opcodes) if op == Opcode.MAX_ABS_REDUCE_FP32)
    quant_pc = next(i for i, op in enumerate(opcodes) if op == Opcode.QUANT_FP32_INT8)
    matmul_pc = next(i for i, op in enumerate(opcodes) if op == Opcode.MATMUL)
    dequant_pc = next(i for i, op in enumerate(opcodes) if op == Opcode.DEQUANT_ACCUM_FP32_SCALED)
    assert max_abs_pc < quant_pc < matmul_pc < dequant_pc


def test_emit_matmul_w8a32_dequant_scaled_consumes_max_abs_sreg_pair():
    """The DEQUANT_ACCUM_FP32_SCALED's sreg must equal the
    MAX_ABS_REDUCE_FP32's sreg + 1 — that's the explicit M2.5-A pair
    contract (S[s] = 127/max, S[s+1] = max/127)."""
    cg, graph, _ = _matmul_w8a32_codegen()
    insns, _ = cg.generate(graph)
    max_abs = next(insn for insn in insns if insn.opcode == Opcode.MAX_ABS_REDUCE_FP32)
    quant = next(insn for insn in insns if insn.opcode == Opcode.QUANT_FP32_INT8)
    dequant = next(insn for insn in insns if insn.opcode == Opcode.DEQUANT_ACCUM_FP32_SCALED)

    # QUANT uses S[s] (the forward inverse scale).
    assert quant.sreg == max_abs.sreg
    # DEQUANT uses S[s+1] (the forward dequant scale).
    assert dequant.sreg == max_abs.sreg + 1


def test_emit_matmul_w8a32_stages_fp16_pc_scales_to_dram():
    """The codegen must stage every weight's PC FP16 scale vector at DRAM
    symbol `f"{weight_name}__w8a32_pc_scale"` so the emit helper can
    DMA-load it into WBUF for the DEQUANT epilogue."""
    cg, graph, _ = _matmul_w8a32_codegen(weight_name="w_test", weight_shape=(16, 32))
    cg.generate(graph)
    sym = "w_test__w8a32_pc_scale"
    assert sym in cg.dram_layout

    # The staged blob is N_pad * 2 bytes (FP16), and the first N values
    # equal the supplied w_scales (cast to FP16).
    pc_off = cg.dram_layout[sym]
    pc_bytes = cg.dram_blob[pc_off:pc_off + 32 * 2]
    pc_vec = np.frombuffer(pc_bytes, dtype=np.float16)
    assert pc_vec.shape == (32,)
    # Non-trivial scales — at least one entry is the staged value.
    expected = cg.weight_data["w_test"][1].astype(np.float16)
    np.testing.assert_array_equal(pc_vec[:len(expected)], expected)


def test_emit_matmul_w8a32_no_bias_omits_bias_vadd():
    """Without `node.attrs['bias']`, no VADD_FP32 is emitted (the lowering
    has no place to inject a bias)."""
    cg, graph, _ = _matmul_w8a32_codegen()  # no bias
    insns, _ = cg.generate(graph)
    opcodes = [insn.opcode for insn in insns]
    assert Opcode.VADD_FP32 not in opcodes
    assert Opcode.VADD not in opcodes


def test_emit_matmul_w8a32_with_bias_appends_fp32_vadd():
    """With a registered bias the lowering must (a) stage the FP32 bias
    in DRAM under `f"{bias_name}__fp32"`, (b) DMA-load it into ABUF, and
    (c) end with VADD_FP32 over the FP32 output tile."""
    bias_vec = np.linspace(-0.5, 0.5, num=16, dtype=np.float32)
    cg, graph, _ = _matmul_w8a32_codegen(bias_name="m_out.bias", bias_vec=bias_vec)
    insns, _ = cg.generate(graph)
    opcodes = [insn.opcode for insn in insns]

    assert Opcode.VADD_FP32 in opcodes
    # VADD_FP32 must come AFTER DEQUANT_ACCUM_FP32_SCALED (bias is applied
    # to the FP32 output, not the INT32 accumulator).
    dequant_pc = next(i for i, op in enumerate(opcodes) if op == Opcode.DEQUANT_ACCUM_FP32_SCALED)
    vadd_pc = next(i for i, op in enumerate(opcodes) if op == Opcode.VADD_FP32)
    assert dequant_pc < vadd_pc

    # DRAM has the FP32 bias staged.
    assert "m_out.bias__fp32" in cg.dram_layout
    bias_off = cg.dram_layout["m_out.bias__fp32"]
    bias_bytes = cg.dram_blob[bias_off:bias_off + 16 * 4]
    staged = np.frombuffer(bias_bytes, dtype=np.float32)
    np.testing.assert_array_equal(staged, bias_vec)


def test_emit_matmul_w8a32_missing_fp32_bias_raises():
    """A matmul node that references a bias name not in `fp32_biases`
    must raise a clear error at emit time — not silently produce a bundle
    that DMA-loads garbage."""
    cg = CodeGenerator(
        weight_data={"w": (
            np.zeros((16, 16), dtype=np.int8),
            np.full(16, 0.01, dtype=np.float16),
        )},
        calibration_scales={"x_fp32": 1.0, "m_out": 1.0},
        prescaled_biases={},
        fp32_biases={},  # intentionally empty
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=True,
    )
    graph = IRGraph()
    graph.add_node(
        IRNode(
            op="matmul",
            name="m_out",
            inputs=["x_fp32", "w"],
            output_shape=(16, 16),
            attrs={"bias": "missing_bias"},
        )
    )
    with pytest.raises(KeyError, match="no FP32 bias was staged"):
        cg.generate(graph)


def test_emit_matmul_w8a32_rejects_dequant_add_residual():
    """Fused DEQUANT_ADD residual conflicts with M2.5-A's dynamic act
    scale — the prescaled-INT32 skip path assumed a static scale.
    `emit_matmul_w8a32` raises NotImplementedError rather than emitting
    a malformed bundle. (The W8A32 init also force-disables these blocks
    on entry; we patch directly to simulate the broken contract.)"""
    cg, graph, _ = _matmul_w8a32_codegen()
    # Bypass init-time force-disable by reactivating the residual block.
    # The lowering helper must catch this regardless.
    cg.dequant_add_residual1_blocks = {0}
    # Make node match the residual1 naming convention.
    graph.nodes[0].name = "block0_out_proj"

    with pytest.raises(NotImplementedError, match="DEQUANT_ADD"):
        cg.generate(graph)


def test_emit_matmul_w8a32_rejects_strip_mine_threshold():
    """If the FP32 output exceeds ABUF_SIZE (or weight exceeds WBUF), the
    lowering raises NotImplementedError instead of silently bailing into
    a strip-mined path. Use an M, K so the FP32 output tile would exceed
    ABUF (128 KB)."""
    # M_pad * N_pad * 4 > 128 KB. With N_pad=16: M > 128KB/(16*4) = 2048.
    cg, graph, _ = _matmul_w8a32_codegen(weight_shape=(16, 16))
    graph.nodes[0].output_shape = (4096, 16)  # FP32 output: 4096*16*4 = 256KB > ABUF
    with pytest.raises(NotImplementedError, match="strip-mined"):
        cg.generate(graph)


def test_emit_matmul_w8a32_end_to_end_on_simulator():
    """End-to-end smoke: compile a 1-matmul IR fragment, run the
    codegen-emitted instructions directly on the simulator with a
    seeded FP32 input, and verify the FP32 output matches the FP32
    reference within INT8 quantization noise. Exercises the full
    M2.5-B path: prelude + INT8 MATMUL + DEQUANT epilogue + optional
    bias VADD."""
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    K, N = 16, 16
    rng = np.random.default_rng(7)
    fp32_input = rng.standard_normal((K, N)).astype(np.float32) * 1.5
    weight_fp32 = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    fp32_bias = rng.standard_normal(N).astype(np.float32) * 0.05

    # Quantize the weight per-column to INT8 (matches quantize_tensor's
    # output: scales shape (N,), INT8 stored [K, N]).
    max_per_col = np.maximum(np.max(np.abs(weight_fp32), axis=0), 1e-8)
    w_scales = (max_per_col / 127.0).astype(np.float16)
    w_int8 = np.clip(
        np.round(weight_fp32 / w_scales.astype(np.float32).reshape(1, -1)),
        -128, 127,
    ).astype(np.int8)

    cg = CodeGenerator(
        weight_data={"w_e2e": (w_int8, w_scales)},
        calibration_scales={"x_e2e": 1.0, "y_e2e": 1.0},
        prescaled_biases={},
        fp32_biases={"b_e2e": fp32_bias},
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=True,
    )
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul",
        name="y_e2e",
        inputs=["x_e2e", "w_e2e"],
        output_shape=(K, N),
        attrs={"bias": "b_e2e"},
    ))
    insns, dram_blob = cg.generate(graph)

    # SET_ADDR_LO/HI emits byte offsets into `dram_blob` (no data_base
    # relocation yet). Mounting dram_blob at DRAM[0] makes those
    # offsets DRAM-absolute. The unified Assembler would relocate them
    # for a real bundle, but for a unit-level integration we skip that.
    sim = Simulator(MachineState(dram_data=bytes(dram_blob)))

    # Seed the FP32 input tile at the input alloc offset (set during
    # codegen's emit_matmul_w8a32 call).
    in_alloc = cg.mem.abuf.allocations["x_e2e"]
    mem.write_fp32_tile(sim.state, BUF_ABUF, in_alloc.offset_units, fp32_input)

    for insn in insns:
        sim._execute(insn)

    out_alloc = cg.mem.abuf.allocations["y_e2e"]
    out_fp32 = mem.read_fp32_tile(sim.state, BUF_ABUF, out_alloc.offset_units, K, N)

    expected_fp32 = (fp32_input @ weight_fp32 + fp32_bias).astype(np.float32)
    err = np.abs(out_fp32 - expected_fp32)

    # Error bound: dominated by INT8 quantization on both activation
    # (dynamic max_abs/127 step) and weight (per-column step). Use a
    # generous bound — the array_equal-tight reference is in
    # test_w8a32_simulator.py's composition test; here we just confirm
    # the codegen path doesn't introduce extra error beyond INT8 noise.
    act_step = float(np.abs(fp32_input).max()) / 127.0
    weight_step = float(w_scales.astype(np.float32).max())
    # Each output entry sums K=16 partial products, each with INT8
    # rounding error bounded by 0.5 * (|x|*weight_step + |w|*act_step).
    # Plus the FP16 round-off on inv_fp32 * fwd_fp32 ≈ 1.0
    # (relative epsilon ≈ 2^-10).
    per_term = (
        float(np.abs(fp32_input).max()) * weight_step
        + float(np.abs(weight_fp32).max()) * act_step
    )
    fp16_rel = 2.0 ** -10
    bound = (
        K * per_term * 1.0  # sum of |errors| <= K * per_term
        + float(np.abs(expected_fp32).max()) * fp16_rel
        + 1e-3
    )
    assert err.max() <= bound, (
        f"max e2e error {err.max():.6f} exceeds bound {bound:.6f}; "
        f"out_fp32[0,0]={out_fp32[0,0]}, expected[0,0]={expected_fp32[0,0]}"
    )


# ---------------------------------------------------------------------------
# M2.5-C: compiler-side fp32_biases population + plumbing
# ---------------------------------------------------------------------------


def _one_layer_gpt2_payload():
    """Minimal one-layer GPT-2 fixture sized for tiny end-to-end tests.

    Mirrors `tests.test_stage5_ptq_presets._one_layer_payload` but duplicated
    locally to avoid cross-test imports."""
    d_model = 16
    mlp_dim = 4 * d_model
    vocab = 16
    block = 16
    state = {
        "transformer.wte.weight": np.linspace(-0.2, 0.2, vocab * d_model, dtype=np.float32).reshape(vocab, d_model),
        "transformer.wpe.weight": np.linspace(0.05, 0.25, block * d_model, dtype=np.float32).reshape(block, d_model),
        "transformer.ln_f.weight": np.ones(d_model, dtype=np.float32),
        "transformer.ln_f.bias": np.zeros(d_model, dtype=np.float32),
        "lm_head.weight": np.linspace(-0.3, 0.4, vocab * d_model, dtype=np.float32).reshape(vocab, d_model),
    }
    for ln in ("ln_1", "ln_2"):
        state[f"transformer.h.0.{ln}.weight"] = np.ones(d_model, dtype=np.float32)
        state[f"transformer.h.0.{ln}.bias"] = np.zeros(d_model, dtype=np.float32)
    for proj in ("query", "key", "value"):
        state[f"transformer.h.0.attn.c_attn.weight_h0_{proj}"] = np.linspace(
            -0.4, 0.4, d_model * d_model, dtype=np.float32
        ).reshape(d_model, d_model)
        state[f"transformer.h.0.attn.c_attn.bias_h0_{proj}"] = np.linspace(
            -0.1, 0.1, d_model, dtype=np.float32
        )
    state["transformer.h.0.attn.c_proj.weight"] = np.linspace(
        -0.5, 0.5, d_model * d_model, dtype=np.float32
    ).reshape(d_model, d_model)
    state["transformer.h.0.attn.c_proj.bias"] = np.linspace(-0.2, 0.2, d_model, dtype=np.float32)
    fc_rows = [
        np.linspace(-0.05 * (idx + 1), 0.05 * (idx + 1), d_model, dtype=np.float32)
        for idx in range(mlp_dim)
    ]
    state["transformer.h.0.mlp.c_fc.weight"] = np.stack(fc_rows, axis=0)
    state["transformer.h.0.mlp.c_fc.bias"] = np.linspace(-0.3, 0.3, mlp_dim, dtype=np.float32)
    state["transformer.h.0.mlp.c_proj.weight"] = np.linspace(
        -0.3, 0.3, d_model * mlp_dim, dtype=np.float32
    ).reshape(d_model, mlp_dim)
    state["transformer.h.0.mlp.c_proj.bias"] = np.linspace(-0.15, 0.15, d_model, dtype=np.float32)
    return {
        "model_args": {
            "n_layer": 1,
            "n_head": 1,
            "n_embd": d_model,
            "block_size": block,
            "vocab_size": vocab,
            "layer_norm_epsilon": 1e-5,
        },
        "state_dict": state,
    }


def test_quantize_fixture_payload_stages_fp32_biases_for_non_attention_matmuls():
    """M2.5-C + M3-prep: `quantize_fixture_payload` returns an
    `fp32_biases` dict for every matmul bias the W8A32 lowering will
    eventually consume.

    M2.5-C staged the four non-attention bias families that
    `emit_matmul_w8a32` reaches today (`mlp.c_fc.bias`,
    `mlp.c_proj.bias`, `attn.c_proj.bias`, and optional `lm_head.bias`).
    M3-prep extends this to also stage per-head Q/K/V biases
    (`c_attn.bias_h{H}_{proj}`) — those sit behind the
    matmul_qkt/matmul_attn_v guardrail until M3 lifts it, but staging
    them now removes a hidden M3 blocker. The per-head Q/K/V matmul
    nodes themselves are `op="matmul"` and would be lowered by
    `emit_matmul_w8a32` once the graph-level guardrail comes down.
    """
    from taccel.runtime.tiny_fixture import quantize_fixture_payload

    payload = _one_layer_gpt2_payload()
    weight_data, prescaled, fp32_biases, _, config, _ = quantize_fixture_payload(payload)

    # 1) Tuple shape: 6 elements (was 5 pre-M2.5-C).
    #    (verified by the unpack succeeding above)

    # 2) Non-attention bias families (M2.5-C).
    expected_non_attention = {
        "transformer.h.0.mlp.c_fc.bias",
        "transformer.h.0.mlp.c_proj.bias",
        "transformer.h.0.attn.c_proj.bias",
    }
    assert expected_non_attention.issubset(fp32_biases.keys()), (
        f"missing FP32 bias entries: {expected_non_attention - fp32_biases.keys()}"
    )

    # 3) Per-head Q/K/V biases now ARE staged (M3-prep). The INT8-path
    #    prescaled version is also expected (kept for parity with the
    #    INT8 path's c_attn split).
    for proj in ("query", "key", "value"):
        head_bias = f"transformer.h.0.attn.c_attn.bias_h0_{proj}"
        assert head_bias in fp32_biases, (
            f"per-head bias '{head_bias}' should be staged (M3-prep)"
        )
        assert head_bias in prescaled

    # 4) Every fp32_biases entry is FP32 and padded to N_pad.
    for name, arr in fp32_biases.items():
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
        # N_pad is a multiple of 16.
        assert arr.shape[0] % 16 == 0, f"{name}: length {arr.shape[0]} not 16-padded"

    # 5) The FP32 bias's leading region equals the raw state_dict tensor.
    fc1_bias = fp32_biases["transformer.h.0.mlp.c_fc.bias"]
    fc1_expected = payload["state_dict"]["transformer.h.0.mlp.c_fc.bias"]
    np.testing.assert_array_equal(fc1_bias[: len(fc1_expected)], fc1_expected.astype(np.float32))


def test_quantize_fixture_payload_stages_lm_head_bias_when_present():
    """When `lm_head.bias` is in the state_dict (the QuaRot β-fold
    path), it must show up in fp32_biases too. The default fixture has
    no `lm_head.bias` — we add one explicitly."""
    from taccel.runtime.tiny_fixture import quantize_fixture_payload

    payload = _one_layer_gpt2_payload()
    payload["state_dict"]["lm_head.bias"] = np.linspace(
        -0.1, 0.1, payload["model_args"]["vocab_size"], dtype=np.float32
    )
    _, prescaled, fp32_biases, _, _, _ = quantize_fixture_payload(payload)

    assert "lm_head.bias" in fp32_biases
    assert "lm_head.bias" in prescaled
    assert fp32_biases["lm_head.bias"].dtype == np.float32


def test_quantize_fixture_payload_omits_lm_head_bias_when_absent():
    """Standard GPT-2 has no `lm_head.bias`; `quantize_fixture_payload`
    must omit it from both `prescaled_biases` and `fp32_biases` rather
    than staging zeros."""
    from taccel.runtime.tiny_fixture import quantize_fixture_payload

    payload = _one_layer_gpt2_payload()
    assert "lm_head.bias" not in payload["state_dict"]
    _, prescaled, fp32_biases, _, _, _ = quantize_fixture_payload(payload)

    assert "lm_head.bias" not in fp32_biases
    assert "lm_head.bias" not in prescaled


def test_build_decoder_program_bundle_accepts_fp32_biases():
    """M2.5-C plumbing: the new `fp32_biases` parameter on
    `build_decoder_program_bundle` must be in the signature and default
    to None (preserves all existing INT8 callers)."""
    from inspect import signature

    from taccel.compiler.decoder_bundle import build_decoder_program_bundle

    sig = signature(build_decoder_program_bundle)
    assert "fp32_biases" in sig.parameters
    assert sig.parameters["fp32_biases"].default is None


def test_build_decoder_program_bundle_w8a32_ffn_only_graph_succeeds():
    """M2.5-C end-to-end plumbing: with a graph containing only
    sub-layer ops + `matmul` (no `matmul_qkt`/`matmul_attn_v`), a
    W8A32-enabled bundle build must succeed and emit the M2.5-A
    primitives in both prefill and decode streams. The required FP32
    biases are supplied via the new parameter.

    This is contrived (real GPT-2 graphs always contain attention),
    but it's the only way to exercise the full populate→forward→consume
    chain without M3."""
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    from taccel.compiler.model_config import ModelConfig

    seq_len, d_model, mlp_dim = 16, 16, 64
    rng = np.random.default_rng(11)

    graph = IRGraph()
    graph.add_node(IRNode(
        op="layernorm", name="ln",
        inputs=["x_in", "ln_gamma", "ln_beta"],
        output_shape=(seq_len, d_model),
        attrs={"block_idx": 0, "epsilon": 1e-5},
    ))
    graph.add_node(IRNode(
        op="matmul", name="fc1",
        inputs=["ln", "fc1_w"],
        output_shape=(seq_len, mlp_dim),
        attrs={"bias": "fc1_b", "weight_name": "fc1_w"},
        weight_name="fc1_w",
    ))
    graph.add_node(IRNode(
        op="gelu", name="gelu_out",
        inputs=["fc1"],
        output_shape=(seq_len, mlp_dim),
        attrs={"block_idx": 0},
    ))
    graph.add_node(IRNode(
        op="matmul", name="fc2",
        inputs=["gelu_out", "fc2_w"],
        output_shape=(seq_len, d_model),
        attrs={"bias": "fc2_b", "weight_name": "fc2_w"},
        weight_name="fc2_w",
    ))

    weight_data = {
        "fc1_w": (
            rng.integers(-127, 128, size=(d_model, mlp_dim), dtype=np.int8),
            (np.abs(rng.standard_normal(mlp_dim)) * 0.001 + 1e-4).astype(np.float16),
        ),
        "fc2_w": (
            rng.integers(-127, 128, size=(mlp_dim, d_model), dtype=np.int8),
            (np.abs(rng.standard_normal(d_model)) * 0.001 + 1e-4).astype(np.float16),
        ),
        "ln_gamma": (np.ones(d_model, dtype=np.float16), None),
        "ln_beta": (np.zeros(d_model, dtype=np.float16), None),
    }
    fp32_biases = {
        "fc1_b": rng.standard_normal(mlp_dim).astype(np.float32),
        "fc2_b": rng.standard_normal(d_model).astype(np.float32),
    }

    config = ModelConfig(
        model_kind="decoder",
        n_layer=1, n_head=1, d_model=d_model, d_head=d_model,
        mlp_dim=mlp_dim, vocab_size=16, max_seq_len=seq_len,
        embedding_kind="token_pos",
    )

    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,  # same fragment for both streams
        weight_data=weight_data,
        calibration_scales={
            "x_in": 1.0, "ln": 1.0, "fc1": 1.0,
            "gelu_out": 1.0, "fc2": 1.0,
        },
        prescaled_biases={},
        fp32_biases=fp32_biases,
        model_config=config,
        max_seq_len=seq_len,
        logits_size=d_model,
        w8a32_enabled=True,
    )

    # The prefill codegen's emitted instructions must contain the W8A32
    # prelude+epilogue primitives (MAX_ABS_REDUCE_FP32 /
    # DEQUANT_ACCUM_FP32_SCALED) plus the FP32 sub-layer ops
    # (LAYERNORM_FP32, GELU_FP32) — confirming the full
    # M2 + M2.5-A + M2.5-B + M2.5-C chain composed correctly.
    prefill_opcodes = {insn.opcode for insn in build.prefill_codegen.instructions}
    assert Opcode.LAYERNORM_FP32 in prefill_opcodes
    assert Opcode.GELU_FP32 in prefill_opcodes
    assert Opcode.MAX_ABS_REDUCE_FP32 in prefill_opcodes
    assert Opcode.QUANT_FP32_INT8 in prefill_opcodes
    assert Opcode.MATMUL in prefill_opcodes
    assert Opcode.DEQUANT_ACCUM_FP32_SCALED in prefill_opcodes
    assert Opcode.VADD_FP32 in prefill_opcodes  # FP32 bias adds
    # And the INT8 epilogue is absent.
    assert Opcode.REQUANT not in prefill_opcodes
    assert Opcode.REQUANT_PC not in prefill_opcodes
    # The decode codegen must mirror the prefill structure (the test
    # passes the same graph for both streams).
    decode_opcodes = {insn.opcode for insn in build.decode_codegen.instructions}
    assert Opcode.DEQUANT_ACCUM_FP32_SCALED in decode_opcodes


# ---------------------------------------------------------------------------
# M2.5-D: FP32 trace-event extraction in Simulator._capture_trace_events
# ---------------------------------------------------------------------------


def test_capture_trace_events_extracts_fp32_tile_for_w8a32_nodes():
    """M2.5-D: when a trace event has `dtype="fp32"` the simulator
    reads from `mem.read_fp32_tile` and stores the values as-is in
    `trace_tensors[node]` (no spurious `* scale` dequant — the FP32
    tile already holds real-units values).

    Pre-M2.5-D this would fall into the INT8 default branch and read
    `mem.read_int8_tile` from a 4-bytes-per-elem FP32 region, then
    multiply by the calibration scale — silently producing garbage
    for any tool consuming W8A32 trace_tensors.

    The check: run the canonical W8A32 FFN-only fragment from the
    M2.5-C plumbing test on the simulator with tracing enabled,
    then assert `trace_tensors[node]` equals the FP32 view of the
    output ABUF region (bit-for-bit `array_equal`).
    """
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    from taccel.compiler.model_config import ModelConfig
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    seq_len, d_model, mlp_dim = 16, 16, 64
    rng = np.random.default_rng(23)

    graph = IRGraph()
    graph.add_node(IRNode(
        op="layernorm", name="ln",
        inputs=["x_in", "ln_gamma", "ln_beta"],
        output_shape=(seq_len, d_model),
        attrs={"block_idx": 0, "epsilon": 1e-5},
    ))
    graph.add_node(IRNode(
        op="matmul", name="fc1",
        inputs=["ln", "fc1_w"],
        output_shape=(seq_len, mlp_dim),
        attrs={"bias": "fc1_b", "weight_name": "fc1_w"},
        weight_name="fc1_w",
    ))
    graph.add_node(IRNode(
        op="gelu", name="gelu_out",
        inputs=["fc1"],
        output_shape=(seq_len, mlp_dim),
        attrs={"block_idx": 0},
    ))
    graph.add_node(IRNode(
        op="matmul", name="fc2",
        inputs=["gelu_out", "fc2_w"],
        output_shape=(seq_len, d_model),
        attrs={"bias": "fc2_b", "weight_name": "fc2_w"},
        weight_name="fc2_w",
    ))

    weight_data = {
        "fc1_w": (
            rng.integers(-127, 128, size=(d_model, mlp_dim), dtype=np.int8),
            (np.abs(rng.standard_normal(mlp_dim)) * 0.001 + 1e-4).astype(np.float16),
        ),
        "fc2_w": (
            rng.integers(-127, 128, size=(mlp_dim, d_model), dtype=np.int8),
            (np.abs(rng.standard_normal(d_model)) * 0.001 + 1e-4).astype(np.float16),
        ),
        "ln_gamma": (np.ones(d_model, dtype=np.float16), None),
        "ln_beta": (np.zeros(d_model, dtype=np.float16), None),
    }
    fp32_biases = {
        "fc1_b": rng.standard_normal(mlp_dim).astype(np.float32),
        "fc2_b": rng.standard_normal(d_model).astype(np.float32),
    }
    config = ModelConfig(
        model_kind="decoder",
        n_layer=1, n_head=1, d_model=d_model, d_head=d_model,
        mlp_dim=mlp_dim, vocab_size=16, max_seq_len=seq_len,
        embedding_kind="token_pos",
    )

    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,
        weight_data=weight_data,
        calibration_scales={
            "x_in": 1.0, "ln": 1.0, "fc1": 1.0,
            "gelu_out": 1.0, "fc2": 1.0,
        },
        prescaled_biases={},
        fp32_biases=fp32_biases,
        model_config=config,
        max_seq_len=seq_len,
        logits_size=d_model,
        w8a32_enabled=True,
    )

    cg = build.prefill_codegen
    instructions = cg.instructions

    # Seed the simulator with the codegen's DRAM blob mounted at offset 0
    # (matches the un-relocated codegen output — same mechanism as the
    # M2.5-B end-to-end test).
    sim = Simulator(MachineState(dram_data=bytes(cg.dram_blob)))
    in_alloc = cg.mem.abuf.allocations["x_in"]
    fp32_input = rng.standard_normal((seq_len, d_model)).astype(np.float32) * 1.0
    mem.write_fp32_tile(sim.state, BUF_ABUF, in_alloc.offset_units, fp32_input)

    # Wire the trace_manifest into the simulator and enable tracing for
    # every node that has FP32 trace events.
    sim.trace_manifest = cg.trace_manifest
    fp32_node_names = sorted({
        event["node_name"]
        for events in cg.trace_manifest.values()
        for event in events
        if event["dtype"] == "fp32"
    })
    # The four sub-layer ops in the FFN fragment must all emit FP32 events.
    assert {"ln", "fc1", "gelu_out", "fc2"}.issubset(set(fp32_node_names)), (
        f"missing FP32 trace events; got {fp32_node_names}"
    )
    sim.enable_trace(fp32_node_names)

    # Execute instruction-by-instruction with explicit pc, since we
    # bypassed `load_program` (that requires the full ProgramBundle
    # plumbing — overkill here). `_capture_trace_events(pc)` does the
    # work we're testing.
    #
    # As we go, we also snapshot the FP32 tile at the same pc using
    # `mem.read_fp32_tile` ourselves — this is the ground-truth view
    # of what the trace capture *should* contain. We have to do this
    # in the same loop because the codegen's last-use allocator frees
    # node outputs once they're no longer live (e.g. `fc1`'s ABUF
    # region gets reclaimed once `gelu_out` is emitted), so a post-loop
    # read would see overwritten bytes.
    ground_truth: dict[str, np.ndarray] = {}
    for pc, insn in enumerate(instructions):
        sim._execute(insn)
        sim._capture_trace_events(pc)
        for event in cg.trace_manifest.get(pc, []):
            if event["dtype"] != "fp32":
                continue
            if event["node_name"] not in fp32_node_names:
                continue
            live = mem.read_fp32_tile(
                sim.state,
                event["buf_id"],
                event["offset_units"],
                event["mem_rows"],
                event["mem_cols"],
            )[: event["logical_rows"], : event["logical_cols"]]
            ground_truth[event["node_name"]] = live.copy()

    payload = sim.get_trace_payload()
    captured = payload["tensors"]

    # M2.5-D contract: for every FP32-traced node the captured trace
    # tensor equals the ground-truth FP32 tile read at the same pc —
    # bit-for-bit `array_equal`. Pre-M2.5-D, `_capture_trace_events`
    # would have fallen into the INT8 branch and read int8_tile from
    # the same offset, producing values 4× compressed and rescaled by
    # a calibration scale — the assertion would have failed.
    for node_name, expected in ground_truth.items():
        rows, cols = expected.shape
        captured_view = captured[node_name][:rows, :cols]
        np.testing.assert_array_equal(
            captured_view,
            expected,
            err_msg=f"trace_tensors[{node_name}] disagrees with ground-truth FP32 tile",
        )

    # Sanity: the trace meta records dtype="fp32" for the captured nodes.
    for node_name in fp32_node_names:
        assert payload["meta"][node_name]["dtype"] == "fp32"


def test_capture_trace_events_fp32_raw_tensors_dtype_is_float32():
    """The `trace_raw_tensors` array dtype for FP32 events must be
    float32 — not int8 (the pre-M2.5-D default) or int32. Downstream
    diagnostic tooling discriminates on `raw_tensors[node].dtype` and
    would mis-classify FP32 nodes as INT8 with the old fallthrough."""
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    from taccel.compiler.model_config import ModelConfig
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    # Minimal one-op fragment: just a GELU_FP32. Smaller surface than the
    # full FFN test; pure regression coverage for the raw-dtype branch.
    seq_len, d_model = 16, 16
    graph = IRGraph()
    graph.add_node(IRNode(
        op="gelu", name="gelu_solo",
        inputs=["x_in"],
        output_shape=(seq_len, d_model),
        attrs={"block_idx": 0},
    ))
    config = ModelConfig(
        model_kind="decoder",
        n_layer=1, n_head=1, d_model=d_model, d_head=d_model,
        mlp_dim=d_model, vocab_size=16, max_seq_len=seq_len,
        embedding_kind="token_pos",
    )
    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,
        weight_data={},
        calibration_scales={"x_in": 1.0, "gelu_solo": 1.0},
        prescaled_biases={},
        fp32_biases={},
        model_config=config,
        max_seq_len=seq_len,
        logits_size=d_model,
        w8a32_enabled=True,
    )

    cg = build.prefill_codegen
    sim = Simulator(MachineState(dram_data=bytes(cg.dram_blob)))
    in_alloc = cg.mem.abuf.allocations["x_in"]
    rng = np.random.default_rng(31)
    mem.write_fp32_tile(
        sim.state, BUF_ABUF, in_alloc.offset_units,
        rng.standard_normal((seq_len, d_model)).astype(np.float32),
    )
    sim.trace_manifest = cg.trace_manifest
    sim.enable_trace(["gelu_solo"])
    for pc, insn in enumerate(cg.instructions):
        sim._execute(insn)
        sim._capture_trace_events(pc)

    payload = sim.get_trace_payload()
    assert "gelu_solo" in payload["raw_tensors"]
    assert payload["raw_tensors"]["gelu_solo"].dtype == np.float32, (
        f"expected raw_tensors[gelu_solo].dtype == float32, "
        f"got {payload['raw_tensors']['gelu_solo'].dtype}"
    )
    # And no saturation stats (those are INT8-specific).
    assert "gelu_solo" not in payload["stats"]


# ---------------------------------------------------------------------------
# M3-prep: embedding → FP32 boundary + per-head Q/K/V FP32 bias staging
# ---------------------------------------------------------------------------


def test_quantize_fixture_payload_stages_per_head_qkv_fp32_biases():
    """M3-prep: `fp32_biases` must include per-head Q/K/V entries so
    the M3 attention lowering can consume them without extra staging
    work. These are dead today (guardrailed by matmul_qkt/matmul_attn_v)
    but their absence would be a silent M3 blocker if not fixed now.

    Also assert that the per-head Q/K/V *weights* are staged alongside
    the biases — `emit_matmul_w8a32` needs both the weight in `weight_data`
    and a corresponding `f"{name}__w8a32_pc_scale"` entry in the codegen's
    DRAM layout (the latter is set up by `codegen.generate()` over the
    full `weight_data` dict, so staging the weight here is sufficient)."""
    from taccel.runtime.tiny_fixture import quantize_fixture_payload

    payload = _one_layer_gpt2_payload()
    weight_data, prescaled, fp32_biases, _, config, _ = quantize_fixture_payload(payload)

    for head in range(config.n_head):
        for proj in ("query", "key", "value"):
            key = f"transformer.h.0.attn.c_attn.bias_h{head}_{proj}"
            assert key in fp32_biases, (
                f"missing per-head FP32 bias '{key}' — M3 attention lowering needs this"
            )
            arr = fp32_biases[key]
            assert arr.dtype == np.float32
            # Padded to N_pad (d_head padded to a multiple of TILE=16).
            assert arr.shape[0] % 16 == 0
            # Leading region matches the raw state_dict tensor.
            raw = payload["state_dict"][key].astype(np.float32)
            np.testing.assert_array_equal(arr[: len(raw)], raw)
            # INT32 prescaled sibling kept for INT8-path parity.
            assert key in prescaled

            # Per-head weight + scale sibling: emit_matmul_w8a32 will look
            # both up. We assert weight presence here; the `__w8a32_pc_scale`
            # DRAM symbol is staged automatically by `codegen.generate()` for
            # any 2-D weight in `weight_data` with non-None scales.
            wname = f"transformer.h.0.attn.c_attn.weight_h{head}_{proj}"
            assert wname in weight_data, (
                f"missing per-head weight '{wname}' — emit_matmul_w8a32 needs this"
            )
            w_int8, w_scales = weight_data[wname]
            assert w_int8.dtype == np.int8
            assert w_scales is not None
            assert w_scales.dtype == np.float16


def test_quantize_fixture_payload_w8a32_stores_embedding_tables_as_fp32():
    """M3-prep: when `w8a32_enabled=True`, `quantize_fixture_payload`
    stores the token and position embedding tables as raw FP32 (4
    bytes/elem) instead of INT8. The codegen DMAs `d_model_pad × 4`
    bytes per row in W8A32 mode so the next sub-layer op reads real
    units. INT8 mode is unchanged."""
    from taccel.runtime.tiny_fixture import quantize_fixture_payload

    payload = _one_layer_gpt2_payload()

    # INT8 path: embeddings stored as int8.
    w8a8_weights, _, _, _, _, _ = quantize_fixture_payload(payload, w8a32_enabled=False)
    assert w8a8_weights["transformer.wte.weight"][0].dtype == np.int8
    assert w8a8_weights["transformer.wpe.weight"][0].dtype == np.int8

    # W8A32 path: embeddings stored as float32 with values matching the
    # raw state_dict tensors (no scaling).
    w8a32_weights, _, _, _, _, _ = quantize_fixture_payload(payload, w8a32_enabled=True)
    wte = w8a32_weights["transformer.wte.weight"][0]
    wpe = w8a32_weights["transformer.wpe.weight"][0]
    assert wte.dtype == np.float32
    assert wpe.dtype == np.float32
    np.testing.assert_array_equal(
        wte, payload["state_dict"]["transformer.wte.weight"].astype(np.float32)
    )
    np.testing.assert_array_equal(
        wpe, payload["state_dict"]["transformer.wpe.weight"].astype(np.float32)
    )


def test_emit_embedding_lookup_w8a32_dmas_fp32_rows_and_records_fp32_trace():
    """M3-prep: in W8A32 mode the embedding emitter must (a) DMA
    `d_model_pad × 4` bytes per row, (b) allocate `M_pad × N_pad × 4`
    bytes in ABUF for the output tile, and (c) record a `dtype="fp32"`
    trace event. This is the codegen-side hookup that pairs with the
    FP32 DRAM storage in `quantize_fixture_payload`.

    INT8 mode is unchanged — verified by a parallel codegen run with
    `w8a32_enabled=False`."""
    seq_len, d_model = 16, 16
    vocab = 16

    # Build a minimal token-embed-only graph fragment.
    def _build(w8a32: bool):
        from taccel.compiler.model_config import ModelConfig

        config = ModelConfig(
            model_kind="decoder", n_layer=1, n_head=1, d_model=d_model,
            d_head=d_model, mlp_dim=d_model, vocab_size=vocab,
            max_seq_len=seq_len, embedding_kind="token_pos",
        )
        # Fake weight_data: store the embedding table at the right
        # shape (FP32 if w8a32 else INT8) so the codegen's DRAM staging
        # uses the right byte width.
        rng = np.random.default_rng(2026)
        if w8a32:
            wte = rng.standard_normal((vocab, d_model)).astype(np.float32)
        else:
            wte = rng.integers(-128, 128, size=(vocab, d_model), dtype=np.int8)
        weight_data = {"transformer.wte.weight": (wte, None)}

        cg = CodeGenerator(
            weight_data=weight_data,
            calibration_scales={"tok_embed": 1.0 / 127.0},
            prescaled_biases={},
            model_config=config,
            stream_name="prefill",
            w8a32_enabled=w8a32,
        )

        graph = IRGraph()
        graph.add_node(IRNode(
            op="embed_lookup",
            name="tok_embed",
            inputs=[],
            output_shape=(seq_len, d_model),
            attrs={
                "table": "transformer.wte.weight",
                "token_ids": [0] * seq_len,
                "seq_len": seq_len,
            },
        ))
        instructions, _ = cg.generate(graph)
        return cg, instructions

    cg_int8, insns_int8 = _build(w8a32=False)
    cg_fp32, insns_fp32 = _build(w8a32=True)

    # (a) DMA xfer_len: the FP32 path's LOAD instructions transfer 4×
    #     more bytes per row than the INT8 path.
    from taccel.isa.opcodes import Opcode

    def _load_xfer_units(insns):
        return [insn.xfer_len for insn in insns if insn.opcode == Opcode.LOAD]

    int8_loads = _load_xfer_units(insns_int8)
    fp32_loads = _load_xfer_units(insns_fp32)
    assert len(int8_loads) == len(fp32_loads), (
        f"different LOAD count: int8={len(int8_loads)}, fp32={len(fp32_loads)}"
    )
    # INT8: d_model_pad bytes / UNIT (16) = d_model_pad / 16 units per row.
    # FP32: d_model_pad * 4 bytes / 16 units = d_model_pad / 4 units per row.
    # Ratio: 4×.
    assert all(f == i * 4 for f, i in zip(fp32_loads, int8_loads)), (
        f"expected FP32 LOAD units = 4× INT8 LOAD units; got int8={int8_loads}, fp32={fp32_loads}"
    )

    # (b) ABUF allocation: 4× more bytes in W8A32.
    int8_alloc = cg_int8.mem.abuf.allocations["tok_embed"]
    fp32_alloc = cg_fp32.mem.abuf.allocations["tok_embed"]
    assert fp32_alloc.size_bytes == int8_alloc.size_bytes * 4

    # (c) Trace event dtype.
    def _embed_trace_dtype(cg):
        for events in cg.trace_manifest.values():
            for event in events:
                if event["node_name"] == "tok_embed":
                    return event["dtype"]
        return None

    assert _embed_trace_dtype(cg_int8) == "int8"
    assert _embed_trace_dtype(cg_fp32) == "fp32"


def test_emit_embedding_lookup_w8a32_end_to_end_matches_state_dict():
    """M3-prep correctness signal: compile a `embed_lookup +
    pos_embed_lookup + tok_pos_add` fragment in W8A32 mode, execute on
    the simulator, and assert the post-vadd FP32 tile in ABUF equals
    the row-wise sum of the raw FP32 state-dict embedding tables
    (array_equal). This is the test that proves the embedding →
    sub-layer FP32 boundary works."""
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    from taccel.compiler.model_config import ModelConfig
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    rng = np.random.default_rng(2026)
    seq_len, d_model = 16, 16
    vocab, max_pos = 32, 16

    wte_fp32 = rng.standard_normal((vocab, d_model)).astype(np.float32)
    wpe_fp32 = rng.standard_normal((max_pos, d_model)).astype(np.float32)

    config = ModelConfig(
        model_kind="decoder", n_layer=1, n_head=1, d_model=d_model,
        d_head=d_model, mlp_dim=d_model, vocab_size=vocab,
        max_seq_len=seq_len, embedding_kind="token_pos",
    )
    token_ids = [3, 7, 11, 0, 1, 2, 5, 9, 14, 6, 4, 10, 12, 15, 8, 13]
    pos_ids = list(range(seq_len))

    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        inputs=[],
        output_shape=(seq_len, d_model),
        attrs={
            "table": "transformer.wte.weight",
            "token_ids": token_ids,
            "seq_len": seq_len,
        },
    ))
    graph.add_node(IRNode(
        op="pos_embed_lookup",
        name="pos_embed",
        inputs=[],
        output_shape=(seq_len, d_model),
        attrs={
            "table": "transformer.wpe.weight",
            "position_ids": pos_ids,
            "seq_len": seq_len,
        },
    ))
    graph.add_node(IRNode(
        op="vadd", name="tok_pos_add",
        inputs=["tok_embed", "pos_embed"],
        output_shape=(seq_len, d_model),
        attrs={},
    ))

    weight_data = {
        "transformer.wte.weight": (wte_fp32, None),
        "transformer.wpe.weight": (wpe_fp32, None),
    }

    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,
        weight_data=weight_data,
        calibration_scales={
            "tok_embed": 1.0,
            "pos_embed": 1.0,
            "tok_pos_add": 1.0,
        },
        prescaled_biases={},
        fp32_biases={},
        model_config=config,
        max_seq_len=seq_len,
        logits_size=d_model,
        w8a32_enabled=True,
    )

    cg = build.prefill_codegen
    sim = Simulator(MachineState(dram_data=bytes(cg.dram_blob)))
    for insn in cg.instructions:
        sim._execute(insn)

    # The post-VADD output tile in ABUF must equal token + position
    # rows from the raw FP32 tables, exactly.
    out_alloc = cg.mem.abuf.allocations["tok_pos_add"]
    out_fp32 = mem.read_fp32_tile(sim.state, BUF_ABUF, out_alloc.offset_units, seq_len, d_model)

    expected = (
        wte_fp32[token_ids, :].astype(np.float32)
        + wpe_fp32[pos_ids, :].astype(np.float32)
    )
    np.testing.assert_array_equal(out_fp32, expected)
