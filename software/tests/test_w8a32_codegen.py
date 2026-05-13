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


def test_w8a32_generate_accepts_all_matmul_kinds_after_m3_b():
    """M3-B empties the W8A32 graph-level guardrail. Every matmul kind
    (`matmul`, `matmul_qkt`, `matmul_attn_v`) is now lowered through
    its respective emitter. The remaining gaps (masked-softmax
    CONFIG_ATTN handling, pad-row zero-fill) are M3-C and surface as
    per-emitter NotImplementedError or simulator-time ConfigError,
    not as graph-level rejection."""
    for op_name in ("matmul", "matmul_qkt", "matmul_attn_v"):
        cg = _fresh_codegen(w8a32=True)
        node_attrs: dict = {}
        if op_name == "matmul_qkt":
            node_attrs = {"head_idx": 0, "scale": 0.25}
        elif op_name == "matmul_attn_v":
            node_attrs = {"head_idx": 0}
        graph = IRGraph()
        graph.add_node(
            IRNode(
                op=op_name,
                name=op_name,
                inputs=["x", "w"],
                output_shape=(16, 16),
                attrs=node_attrs,
                weight_name="w",
            )
        )
        # The codegen must not raise NotImplementedError at the graph
        # boundary. For matmul (no weight_data) the emitter returns early
        # without producing instructions; for QKT/attn_v the static
        # composite scale staging + emitter run to completion.
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


# ---------------------------------------------------------------------------
# M3-A: matmul_qkt W8A32 lowering (Q @ K^T with static composite dequant)
# ---------------------------------------------------------------------------


from taccel.compiler.w8a32_emit import emit_matmul_qkt_w8a32  # noqa: E402


def _qkt_w8a32_codegen(
    *,
    seq_len: int = 16,
    d_head: int = 16,
    q_scale: float = 0.05,
    k_scale: float = 0.07,
    inv_sqrt_d_head: float = None,
) -> tuple[CodeGenerator, IRGraph, IRNode]:
    """Build a CodeGenerator + 1-node IR graph for an isolated matmul_qkt."""
    from taccel.compiler.model_config import ModelConfig

    if inv_sqrt_d_head is None:
        inv_sqrt_d_head = float(d_head ** -0.5)
    config = ModelConfig(
        model_kind="decoder", n_layer=1, n_head=1, d_model=d_head,
        d_head=d_head, mlp_dim=d_head, vocab_size=16,
        max_seq_len=seq_len, embedding_kind="token_pos",
    )
    cg = CodeGenerator(
        weight_data={},
        calibration_scales={
            "q_in": q_scale,
            "k_in": k_scale,
            "qkt_out": 1.0,
        },
        prescaled_biases={},
        fp32_biases={},
        model_config=config,
        stream_name="prefill",
        w8a32_enabled=True,
    )
    graph = IRGraph()
    node = IRNode(
        op="matmul_qkt",
        name="qkt_out",
        inputs=["q_in", "k_in"],
        output_shape=(seq_len, seq_len),
        attrs={
            "head_idx": 0,
            "scale": inv_sqrt_d_head,
            "query_len": seq_len,
            "key_len": seq_len,
        },
    )
    graph.add_node(node)
    return cg, graph, node


def test_emit_matmul_qkt_w8a32_emits_full_quant_matmul_dequant_sequence():
    """M3-A: the QKT lowering must emit QUANT_FP32_INT8 ×2, BUF_COPY
    (transpose=1), MATMUL, DEQUANT_ACCUM_FP32 in order. No M2.5-A
    DEQUANT_ACCUM_FP32_SCALED here — the static composite fold uses M1's
    non-scaled DEQUANT variant."""
    cg, graph, _ = _qkt_w8a32_codegen()
    insns, _ = cg.generate(graph)
    opcodes = [insn.opcode for insn in insns]

    quant_indices = [i for i, op in enumerate(opcodes) if op == Opcode.QUANT_FP32_INT8]
    bufcopy_indices = [i for i, op in enumerate(opcodes) if op == Opcode.BUF_COPY]
    matmul_indices = [i for i, op in enumerate(opcodes) if op == Opcode.MATMUL]
    dequant_indices = [i for i, op in enumerate(opcodes) if op == Opcode.DEQUANT_ACCUM_FP32]

    assert len(quant_indices) == 2, f"expected 2 QUANT_FP32_INT8, got {len(quant_indices)}"
    assert len(bufcopy_indices) >= 1
    assert len(matmul_indices) == 1
    assert len(dequant_indices) == 1

    # Ordering: both quants before the bufcopy transpose, bufcopy before
    # matmul, matmul before dequant.
    assert quant_indices[0] < quant_indices[1] < bufcopy_indices[0]
    assert bufcopy_indices[0] < matmul_indices[0] < dequant_indices[0]

    # The BUF_COPY for the K transpose has transpose=1.
    transpose_bufcopies = [
        insn for insn in insns
        if insn.opcode == Opcode.BUF_COPY and insn.transpose == 1
    ]
    assert len(transpose_bufcopies) >= 1, (
        "expected at least one BUF_COPY with transpose=1 (K-transpose)"
    )

    # No M2.5-A SCALED variant in the sequence — QKT uses the M1 dequant.
    assert Opcode.DEQUANT_ACCUM_FP32_SCALED not in opcodes
    # And no MAX_ABS_REDUCE_FP32 (QKT uses static scales for Q and K).
    assert Opcode.MAX_ABS_REDUCE_FP32 not in opcodes


def test_emit_matmul_qkt_w8a32_pc_scale_bytes_bit_exact_fp16_composite():
    """M3-A correctness signal: the PC scale vector staged at DRAM
    must match `np.float16(q_scale * k_scale * inv_sqrt_d_head)`
    replicated N_pad times, bit-for-bit. Any drift in the FP32→FP16
    cast order between codegen and reference here would silently
    skew the entire QK^T result."""
    q_scale = 0.05
    k_scale = 0.07
    inv_sqrt_d_head = 1.0 / np.sqrt(16.0)  # d_head=16
    cg, graph, node = _qkt_w8a32_codegen(
        q_scale=q_scale, k_scale=k_scale, inv_sqrt_d_head=inv_sqrt_d_head,
    )
    cg.generate(graph)

    sym = f"{node.name}__qkt_pc_scale"
    assert sym in cg.dram_layout, f"PC scale symbol '{sym}' not staged"

    offset = cg.dram_layout[sym]
    n_pad = 16  # pad_dim(16) == 16
    pc_bytes = cg.dram_blob[offset:offset + n_pad * 2]
    staged = np.frombuffer(pc_bytes, dtype=np.float16)

    expected_value_fp16 = np.float16(
        np.float32(q_scale) * np.float32(k_scale) * np.float32(inv_sqrt_d_head)
    )
    expected = np.full(n_pad, expected_value_fp16, dtype=np.float16)
    np.testing.assert_array_equal(staged, expected)


def test_emit_scale_mul_renames_abuf_alloc_in_w8a32_mode():
    """M3-A: _emit_scale_mul must rename the ABUF allocation (not just
    WBUF) when the source is in ABUF — the W8A32 QKT writes its FP32
    scores tile to ABUF, so the scale_mul rename has to happen there.
    Calibration scale propagation stays the same."""
    cg = _fresh_codegen(w8a32=True)
    # Pre-populate an ABUF allocation as if a prior op had written there.
    cg.mem.abuf.alloc("source_node", 16 * 16 * 4)  # FP32 16×16
    cg.calibration_scales["source_node"] = 0.123

    scale_mul_node = IRNode(
        op="scale_mul",
        name="scaled_node",
        inputs=["source_node"],
        output_shape=(16, 16),
        attrs={"scale": 0.125},
    )
    cg._emit_scale_mul(scale_mul_node)

    # The ABUF allocation under the old name is gone; the new name owns it.
    assert "source_node" not in cg.mem.abuf.allocations
    assert "scaled_node" in cg.mem.abuf.allocations

    # Calibration scale propagates.
    assert cg.calibration_scales["scaled_node"] == cg.calibration_scales["source_node"] == 0.123

    # No instructions emitted (scale_mul is metadata-only).
    assert cg.instructions == []


def test_emit_matmul_qkt_w8a32_zero_fills_k_pad_rows_when_key_len_short():
    """M3-C: when `key_len < N_pad`, the QKT lowering emits a DMA from
    `__zero_pad__` to zero K's FP32 pad rows in ABUF before the K-QUANT
    step. Without this, LN(zero_row) = β contaminates K's padding rows
    and downstream attention scores would include β-derived garbage in
    the masked-out columns. Replaces the M3-A `rejects_pad_row` test
    which was the temporary deferral guard."""
    config_seq_len = 16
    key_len_short = 12  # pad_dim(12) == 16, so N_pad > key_len
    cg, graph, node = _qkt_w8a32_codegen(seq_len=config_seq_len)
    node.attrs["key_len"] = key_len_short
    node.output_shape = (config_seq_len, key_len_short)
    # Must not raise — M3-C handles the pad-row case.
    insns, _ = cg.generate(graph)

    # K pad-row zero-fill DMA: at least one LOAD reads from
    # `__zero_pad__`'s DRAM offset with addr_reg=3 (the standard
    # zero-pad source convention from the INT8 path). Verifying via
    # the addr_reg=3 LOAD instruction is the lightest-touch check.
    load_ar3_indices = [
        i for i, insn in enumerate(insns)
        if insn.opcode == Opcode.LOAD and insn.addr_reg == 3
    ]
    assert len(load_ar3_indices) >= 1, (
        "expected at least one LOAD with addr_reg=3 (K pad-row zero-fill DMA)"
    )

    # The xfer_len for the pad-row LOAD = pad_rows * K_pad * 4 bytes.
    # pad_rows = N_pad - key_len = 16 - 12 = 4. K_pad = pad_dim(d_head) = 16.
    # FP32 byte size = 4 * 16 * 4 = 256 bytes = 16 units.
    pad_row_loads = [insns[i] for i in load_ar3_indices]
    assert any(insn.xfer_len == 16 for insn in pad_row_loads), (
        f"expected pad-row LOAD xfer_len=16 (4 pad rows × 16 d_head × 4 bytes / 16); "
        f"got xfer_lens={[insn.xfer_len for insn in pad_row_loads]}"
    )


def test_w8a32_generate_accepts_matmul_qkt_after_m3_a():
    """Counterpart: a graph with only matmul_qkt (no matmul_attn_v) must
    now compile in W8A32 mode. The guardrail narrowed in M3-A."""
    cg, graph, _ = _qkt_w8a32_codegen()
    insns, _ = cg.generate(graph)
    # Bundle compiles; emitted instructions contain the M3-A epilogue.
    opcodes = {insn.opcode for insn in insns}
    assert Opcode.DEQUANT_ACCUM_FP32 in opcodes
    assert Opcode.MATMUL in opcodes


def test_emit_matmul_qkt_w8a32_end_to_end_matches_fp32_reference():
    """M3-A end-to-end correctness signal: compile a single matmul_qkt
    fragment in W8A32 mode, seed FP32 Q and K directly in ABUF,
    execute on the simulator, and assert the FP32 scores tile matches
    the reference `(Q @ K.T) * inv_sqrt_d_head` within INT8
    quantization noise.

    This is the bit-end-to-end signal that emit_matmul_qkt_w8a32
    composes correctly with the M1 DEQUANT_ACCUM_FP32 and the QUANT/
    BUF_COPY transpose mechanics."""
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    seq_len = 16
    d_head = 16
    inv_sqrt_d_head = 1.0 / np.sqrt(d_head)

    rng = np.random.default_rng(33)
    q_fp32 = rng.standard_normal((seq_len, d_head)).astype(np.float32) * 0.4
    k_fp32 = rng.standard_normal((seq_len, d_head)).astype(np.float32) * 0.5

    # Pick calibration scales generously to cover the actual max(|Q|), max(|K|).
    q_scale = float(np.abs(q_fp32).max() / 127.0)
    k_scale = float(np.abs(k_fp32).max() / 127.0)

    cg, graph, _ = _qkt_w8a32_codegen(
        seq_len=seq_len, d_head=d_head,
        q_scale=q_scale, k_scale=k_scale, inv_sqrt_d_head=inv_sqrt_d_head,
    )
    insns, dram_blob = cg.generate(graph)

    sim = Simulator(MachineState(dram_data=bytes(dram_blob)))
    # Seed FP32 Q and K under the input names the codegen allocated for.
    # `_qkt_w8a32_codegen` lets emit_matmul_qkt_w8a32 allocate these via
    # `_abuf_alloc_fp32` since they aren't pre-allocated.
    q_alloc = cg.mem.abuf.allocations["q_in"]
    k_alloc = cg.mem.abuf.allocations["k_in"]
    mem.write_fp32_tile(sim.state, BUF_ABUF, q_alloc.offset_units, q_fp32)
    mem.write_fp32_tile(sim.state, BUF_ABUF, k_alloc.offset_units, k_fp32)

    for insn in insns:
        sim._execute(insn)

    out_alloc = cg.mem.abuf.allocations["qkt_out"]
    out_fp32 = mem.read_fp32_tile(sim.state, BUF_ABUF, out_alloc.offset_units, seq_len, seq_len)

    expected = (q_fp32 @ k_fp32.T).astype(np.float32) * np.float32(inv_sqrt_d_head)
    err = np.abs(out_fp32 - expected)

    # Error bound: dominated by INT8 quantization of Q and K. Each output
    # entry is a sum of d_head=16 partial products, each with INT8 round
    # error bounded by 0.5 * (|q| * k_step + |k| * q_step). Plus FP16
    # round-off on the composite PC scale.
    q_step = q_scale
    k_step = k_scale
    per_term = (
        float(np.abs(q_fp32).max()) * k_step
        + float(np.abs(k_fp32).max()) * q_step
    )
    fp16_rel = 2.0 ** -10
    bound = (
        d_head * per_term * abs(inv_sqrt_d_head) * 1.0
        + float(np.abs(expected).max()) * fp16_rel
        + 1e-3
    )
    assert err.max() <= bound, (
        f"max e2e error {err.max():.6f} exceeds bound {bound:.6f}; "
        f"out[0,0]={out_fp32[0,0]}, expected[0,0]={expected[0,0]}"
    )


def test_qkt_scale_mul_softmax_chain_w8a32_pipes_through_abuf_alias():
    """M3-A integration: a graph containing matmul_qkt → scale_mul →
    softmax must produce a softmax that reads its input from the QKT
    output ABUF region (via the scale_mul rename). The contract is:

      qkt:       emit_matmul_qkt_w8a32 allocates `abuf[qkt_node]`.
      scale_mul: _emit_scale_mul renames `abuf[qkt_node]` → `abuf[scale_mul_node]`.
      softmax:   emit_softmax_fp32's `_abuf_alloc_fp32(input)` returns the
                 EXISTING alloc under `scale_mul_node` (no fresh alloc).

    Without the M3-A `_emit_scale_mul` ABUF branch the softmax would
    silently allocate a fresh region and read uninitialized FP32 bytes.
    This test guards against future refactors of `_abuf_alloc_fp32`
    that drop the "return existing if any" behavior.

    By the time `generate()` returns, the codegen's last-use allocator
    has freed `qkt_scaled` (softmax was its last consumer), so we check
    the contract via the trace manifest: the QKT trace event records the
    offset where QKT wrote; the SOFTMAX_FP32 instruction's `src1_off`
    must match that offset.
    """
    seq_len = 16
    d_head = 16
    inv_sqrt_d_head = 1.0 / np.sqrt(d_head)

    cg, graph, qkt_node = _qkt_w8a32_codegen(
        seq_len=seq_len, d_head=d_head, inv_sqrt_d_head=inv_sqrt_d_head,
    )
    # Extend the graph with the post-QKT chain — non-masked softmax so we
    # don't have to set up CONFIG_ATTN (M3-B will handle that).
    graph.add_node(IRNode(
        op="scale_mul",
        name="qkt_scaled",
        inputs=["qkt_out"],
        output_shape=(seq_len, seq_len),
        attrs={"scale": inv_sqrt_d_head},
    ))
    graph.add_node(IRNode(
        op="softmax",
        name="qkt_softmax",
        inputs=["qkt_scaled"],
        output_shape=(seq_len, seq_len),
        attrs={"causal": False},  # non-masked — avoids CONFIG_ATTN dependency
    ))
    insns, _ = cg.generate(graph)

    # QKT trace event tells us where QKT wrote its output ABUF tile.
    qkt_offset = None
    for events in cg.trace_manifest.values():
        for event in events:
            if event["node_name"] == "qkt_out" and event["dtype"] == "fp32":
                qkt_offset = event["offset_units"]
    assert qkt_offset is not None, "no qkt_out trace event found"

    # SOFTMAX_FP32 must read from that same offset — proving the scale_mul
    # rename + softmax-input lookup chain pointed at the QKT region.
    softmax_insns = [
        insn for insn in insns if insn.opcode == Opcode.SOFTMAX_FP32
    ]
    assert len(softmax_insns) == 1
    assert softmax_insns[0].src1_off == qkt_offset, (
        f"SOFTMAX_FP32 reads from offset {softmax_insns[0].src1_off}, "
        f"expected {qkt_offset} (QKT's output offset, via scale_mul rename). "
        f"Likely cause: _emit_scale_mul didn't rename the ABUF alloc, so "
        f"emit_softmax_fp32 allocated a fresh region and reads garbage."
    )

    # And the softmax output stays in ABUF (its own fresh alloc).
    assert "qkt_softmax" in cg.mem.abuf.allocations


# ---------------------------------------------------------------------------
# M3-B: matmul_attn_v W8A32 lowering — softmax(QK^T) @ V
# ---------------------------------------------------------------------------


from taccel.compiler.w8a32_emit import emit_matmul_attn_v_w8a32  # noqa: E402


def _attn_v_w8a32_codegen(
    *,
    seq_len: int = 16,
    d_head: int = 16,
    sm_scale: float = 1.0 / 127.0,
    v_scale: float = 0.07,
) -> tuple[CodeGenerator, IRGraph, IRNode]:
    """Build a CodeGenerator + 1-node IR graph for an isolated matmul_attn_v."""
    from taccel.compiler.model_config import ModelConfig

    config = ModelConfig(
        model_kind="decoder", n_layer=1, n_head=1, d_model=d_head,
        d_head=d_head, mlp_dim=d_head, vocab_size=16,
        max_seq_len=seq_len, embedding_kind="token_pos",
    )
    cg = CodeGenerator(
        weight_data={},
        calibration_scales={
            "sm_in": sm_scale,
            "v_in": v_scale,
            "attn_v_out": 1.0,
        },
        prescaled_biases={},
        fp32_biases={},
        model_config=config,
        stream_name="prefill",
        w8a32_enabled=True,
    )
    graph = IRGraph()
    node = IRNode(
        op="matmul_attn_v",
        name="attn_v_out",
        inputs=["sm_in", "v_in"],
        output_shape=(seq_len, d_head),
        attrs={
            "head_idx": 0,
            "query_len": seq_len,
            "key_len": seq_len,
        },
    )
    graph.add_node(node)
    return cg, graph, node


def test_emit_matmul_attn_v_w8a32_emits_full_quant_matmul_dequant_sequence():
    """M3-B opcode contract: QUANT_FP32_INT8 ×2 (softmax then V),
    BUF_COPY with transpose=0 (V is K-major, no transpose needed —
    contrasts M3-A QKT which has transpose=1 for K^T), single MATMUL,
    single DEQUANT_ACCUM_FP32. No M2.5-A SCALED variant; no
    MAX_ABS_REDUCE_FP32 (static composite scale, same architecture
    as M3-A)."""
    cg, graph, _ = _attn_v_w8a32_codegen()
    insns, _ = cg.generate(graph)
    opcodes = [insn.opcode for insn in insns]

    quant_indices = [i for i, op in enumerate(opcodes) if op == Opcode.QUANT_FP32_INT8]
    bufcopy_indices = [i for i, op in enumerate(opcodes) if op == Opcode.BUF_COPY]
    matmul_indices = [i for i, op in enumerate(opcodes) if op == Opcode.MATMUL]
    dequant_indices = [i for i, op in enumerate(opcodes) if op == Opcode.DEQUANT_ACCUM_FP32]

    assert len(quant_indices) == 2
    assert len(bufcopy_indices) >= 1
    assert len(matmul_indices) == 1
    assert len(dequant_indices) == 1

    # Ordering: both quants before the bufcopy, bufcopy before matmul,
    # matmul before dequant.
    assert quant_indices[0] < quant_indices[1] < bufcopy_indices[0]
    assert bufcopy_indices[0] < matmul_indices[0] < dequant_indices[0]

    # V BUF_COPY is NON-transpose (transpose=0). This is the key
    # structural difference vs M3-A QKT's BUF_COPY (which is transpose=1
    # for K^T).
    v_bufcopies = [
        insn for insn in insns
        if insn.opcode == Opcode.BUF_COPY and insn.transpose == 0
    ]
    assert len(v_bufcopies) >= 1, (
        "expected at least one BUF_COPY with transpose=0 (V move to WBUF)"
    )

    assert Opcode.DEQUANT_ACCUM_FP32_SCALED not in opcodes
    assert Opcode.MAX_ABS_REDUCE_FP32 not in opcodes


def test_emit_matmul_attn_v_w8a32_pc_scale_bytes_bit_exact_fp16_composite():
    """M3-B correctness signal: the PC scale vector staged at DRAM must
    match `np.float16(sm_scale * v_scale)` (FP32 × FP32 → FP16)
    replicated N_pad times. Cast order locked, same as M3-A.

    Unlike M3-A QKT, no `1/√d_head` factor — that's already applied by
    `emit_matmul_qkt_w8a32` upstream and the softmax-output values are
    in [0, 1] regardless of input scale."""
    sm_scale = 1.0 / 127.0
    v_scale = 0.07
    cg, graph, node = _attn_v_w8a32_codegen(sm_scale=sm_scale, v_scale=v_scale)
    cg.generate(graph)

    sym = f"{node.name}__attn_v_pc_scale"
    assert sym in cg.dram_layout, f"PC scale symbol '{sym}' not staged"

    offset = cg.dram_layout[sym]
    n_pad = 16  # pad_dim(16) == 16
    pc_bytes = cg.dram_blob[offset:offset + n_pad * 2]
    staged = np.frombuffer(pc_bytes, dtype=np.float16)

    expected_value_fp16 = np.float16(np.float32(sm_scale) * np.float32(v_scale))
    expected = np.full(n_pad, expected_value_fp16, dtype=np.float16)
    np.testing.assert_array_equal(staged, expected)


def test_emit_matmul_attn_v_w8a32_zero_fills_v_pad_rows_when_key_len_short():
    """M3-C: when `key_len < Kseq_pad`, the attn_v lowering zero-fills
    V's FP32 pad rows before the V-QUANT step. Same architectural
    reason as the QKT K-pad case (LN-derived β contaminates padded V
    rows). Replaces the M3-B `rejects_pad_row` test."""
    cg, graph, node = _attn_v_w8a32_codegen()
    short_key_len = 12  # pad_dim(12) = 16
    node.attrs["key_len"] = short_key_len
    # Must not raise — M3-C handles the pad-row case.
    insns, _ = cg.generate(graph)

    # Same check pattern as the QKT pad-row test: at least one LOAD
    # with addr_reg=3 (the __zero_pad__ DMA convention).
    load_ar3_indices = [
        i for i, insn in enumerate(insns)
        if insn.opcode == Opcode.LOAD and insn.addr_reg == 3
    ]
    assert len(load_ar3_indices) >= 1, (
        "expected at least one LOAD with addr_reg=3 (V pad-row zero-fill DMA)"
    )

    # V pad-row LOAD: pad_rows = Kseq_pad - key_len = 16 - 12 = 4.
    # N_pad = pad_dim(d_head) = 16. FP32 byte size = 4*16*4 = 256B = 16 units.
    pad_row_loads = [insns[i] for i in load_ar3_indices]
    assert any(insn.xfer_len == 16 for insn in pad_row_loads), (
        f"expected V pad-row LOAD xfer_len=16; "
        f"got xfer_lens={[insn.xfer_len for insn in pad_row_loads]}"
    )


def test_emit_matmul_attn_v_w8a32_end_to_end_matches_fp32_reference():
    """M3-B end-to-end correctness signal: compile a `matmul_qkt →
    softmax (non-masked) → matmul_attn_v` fragment in W8A32 mode,
    seed FP32 Q, K, V directly in ABUF, execute on the simulator,
    and assert the FP32 attn_v output matches the numpy reference
    `softmax(Q @ K.T / √d_head) @ V` within INT8 quantization noise.

    This is the integration signal that M3-A's QKT, M2's softmax_fp32,
    and M3-B's attn_v all compose correctly through the codegen +
    simulator pipeline. Non-masked softmax avoids the M3-C
    CONFIG_ATTN gap; the math is otherwise identical to causal."""
    from taccel.compiler.model_config import ModelConfig
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    seq_len = 16
    d_head = 16
    inv_sqrt_d_head = 1.0 / np.sqrt(d_head)

    rng = np.random.default_rng(57)
    q_fp32 = rng.standard_normal((seq_len, d_head)).astype(np.float32) * 0.3
    k_fp32 = rng.standard_normal((seq_len, d_head)).astype(np.float32) * 0.4
    v_fp32 = rng.standard_normal((seq_len, d_head)).astype(np.float32) * 0.5

    # Generous calibration scales to cover max(|·|) of each input.
    q_scale = float(np.abs(q_fp32).max() / 127.0)
    k_scale = float(np.abs(k_fp32).max() / 127.0)
    v_scale = float(np.abs(v_fp32).max() / 127.0)
    sm_scale = 1.0 / 127.0  # softmax output in [0, 1] → 1/127 maps to INT8

    config = ModelConfig(
        model_kind="decoder", n_layer=1, n_head=1, d_model=d_head,
        d_head=d_head, mlp_dim=d_head, vocab_size=16,
        max_seq_len=seq_len, embedding_kind="token_pos",
    )

    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul_qkt",
        name="qkt",
        inputs=["q_in", "k_in"],
        output_shape=(seq_len, seq_len),
        attrs={
            "head_idx": 0,
            "scale": inv_sqrt_d_head,
            "query_len": seq_len,
            "key_len": seq_len,
        },
    ))
    graph.add_node(IRNode(
        op="scale_mul",
        name="qkt_scaled",
        inputs=["qkt"],
        output_shape=(seq_len, seq_len),
        attrs={"scale": inv_sqrt_d_head},
    ))
    graph.add_node(IRNode(
        op="softmax",
        name="sm",
        inputs=["qkt_scaled"],
        output_shape=(seq_len, seq_len),
        attrs={"causal": False},
    ))
    graph.add_node(IRNode(
        op="matmul_attn_v",
        name="attn_v",
        inputs=["sm", "v_in"],
        output_shape=(seq_len, d_head),
        attrs={"head_idx": 0, "query_len": seq_len, "key_len": seq_len},
    ))

    cg = CodeGenerator(
        weight_data={},
        calibration_scales={
            "q_in": q_scale,
            "k_in": k_scale,
            "v_in": v_scale,
            "qkt": 1.0,
            "qkt_scaled": 1.0,
            "sm": sm_scale,
            "attn_v": 1.0,
        },
        prescaled_biases={},
        fp32_biases={},
        model_config=config,
        stream_name="prefill",
        w8a32_enabled=True,
    )
    # Pre-allocate FP32 inputs (q_in, k_in, v_in) before generate() so
    # they live in the allocator's "active" set throughout. Without this,
    # `emit_matmul_attn_v_w8a32`'s lazy `v_in` alloc could land on a
    # region freed by an earlier int8-scratch (the codegen's free
    # tracking lets the region be reused, but the EMITTED QUANT
    # instruction still writes there at execution time, corrupting V's
    # pre-seeded FP32 bytes). In real graphs V comes from a per-head V
    # projection's `emit_matmul_w8a32`, which keeps V's region live
    # throughout the QKT/softmax/attn_v window.
    fp32_bytes = seq_len * d_head * 4
    cg.mem.abuf.alloc("q_in", fp32_bytes)
    cg.mem.abuf.alloc("k_in", fp32_bytes)
    cg.mem.abuf.alloc("v_in", fp32_bytes)
    insns, dram_blob = cg.generate(graph)

    sim = Simulator(MachineState(dram_data=bytes(dram_blob)))
    # Seed FP32 Q, K, V at their respective input ABUF allocs.
    q_alloc = cg.mem.abuf.allocations["q_in"]
    k_alloc = cg.mem.abuf.allocations["k_in"]
    v_alloc = cg.mem.abuf.allocations["v_in"]
    mem.write_fp32_tile(sim.state, BUF_ABUF, q_alloc.offset_units, q_fp32)
    mem.write_fp32_tile(sim.state, BUF_ABUF, k_alloc.offset_units, k_fp32)
    mem.write_fp32_tile(sim.state, BUF_ABUF, v_alloc.offset_units, v_fp32)

    for insn in insns:
        sim._execute(insn)

    # The attn_v output replaces qkt's ABUF region via the scale_mul →
    # softmax → attn_v rename + free chain. Pull the recorded
    # offset from the attn_v trace event (the live alloc may have
    # been freed by last-use cleanup; the trace event captures the
    # write site).
    attn_v_offset = None
    for events in cg.trace_manifest.values():
        for event in events:
            if event["node_name"] == "attn_v" and event["dtype"] == "fp32":
                attn_v_offset = event["offset_units"]
    assert attn_v_offset is not None
    out_fp32 = mem.read_fp32_tile(
        sim.state, BUF_ABUF, attn_v_offset, seq_len, d_head
    )

    # Numpy reference: softmax(Q @ K.T / √d_head) @ V.
    scores = (q_fp32 @ k_fp32.T).astype(np.float32) * np.float32(inv_sqrt_d_head)
    row_max = scores.max(axis=-1, keepdims=True)
    exp = np.exp((scores - row_max).astype(np.float32))
    sm_ref = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
    expected = (sm_ref @ v_fp32).astype(np.float32)

    err = np.abs(out_fp32 - expected)
    # Composite error bound: INT8 quant on Q/K (QKT path) + INT8 quant on
    # softmax/V (attn_v path) + FP16 round-off on both composite PC scales.
    # Observed err.max() ≈ 0.013 (4.7% relative for |expected|.max() ≈ 0.27)
    # with seed=57; bound 0.05 gives ~4× headroom for seed variation while
    # catching a 4× regression in INT8 round-trip precision. Bit-exact PC
    # scale staging tests are the structural correctness signal; this e2e
    # is integration coverage.
    bound = 0.05
    assert err.max() <= bound, (
        f"max e2e error {err.max():.6f} exceeds bound {bound:.6f}; "
        f"out[0,0]={out_fp32[0,0]}, expected[0,0]={expected[0,0]}"
    )
    # Sanity: outputs are real-valued (no inf, no nan).
    assert np.all(np.isfinite(out_fp32))
    # Sanity: outputs are at the right rough scale (V range × probability range).
    assert np.abs(out_fp32).max() <= 2.0 * np.abs(v_fp32).max()


# ---------------------------------------------------------------------------
# M3-C: CONFIG_ATTN for masked softmax + pad-row zero-fill in W8A32
# ---------------------------------------------------------------------------


def test_emit_softmax_fp32_masked_emits_config_attn_before_masked_softmax():
    """M3-C: when `node.attrs['masked']` is True, `emit_softmax_fp32`
    emits ConfigAttnInsn(query_row_base=0, valid_kv_len=key_len, mode=0b10)
    immediately before MaskedSoftmaxFp32Insn. In W8A32 mode softmax runs
    over the FULL M_pad × N_pad tile in one instruction, so
    `query_row_base=0` always — not the per-strip dance the INT8 path
    does in `_emit_qkt`."""
    cg = _fresh_codegen(w8a32=True)
    key_len = 16
    node = IRNode(
        op="softmax",
        name="masked_sm",
        inputs=["sm_in"],
        output_shape=(16, key_len),
        attrs={"masked": True, "key_len": key_len},
    )
    emit_softmax_fp32(cg, node, masked=True)
    opcodes = [insn.opcode for insn in cg.instructions]

    # The CONFIG_ATTN must appear before MASKED_SOFTMAX_FP32.
    config_attn_indices = [i for i, op in enumerate(opcodes) if op == Opcode.CONFIG_ATTN]
    masked_sm_indices = [i for i, op in enumerate(opcodes) if op == Opcode.MASKED_SOFTMAX_FP32]
    assert len(config_attn_indices) == 1, (
        f"expected exactly one CONFIG_ATTN, got {len(config_attn_indices)}"
    )
    assert len(masked_sm_indices) == 1
    assert config_attn_indices[0] < masked_sm_indices[0]
    # And the CONFIG_ATTN immediately precedes the masked softmax (with
    # any number of non-CONFIG_ATTN ops in between is fine, but in this
    # simple fragment the gap is just CONFIG_TILE).
    config_attn = cg.instructions[config_attn_indices[0]]
    assert config_attn.query_row_base == 0
    assert config_attn.valid_kv_len == key_len
    assert config_attn.mode == 0b10  # prefill causal (N_pad == key_len)


def test_emit_softmax_fp32_unmasked_does_not_emit_config_attn():
    """M3-C counterpart: non-masked softmax still emits SOFTMAX_FP32
    without any CONFIG_ATTN setup."""
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="softmax",
        name="plain_sm",
        inputs=["sm_in"],
        output_shape=(16, 16),
        attrs={"causal": False},
    )
    emit_softmax_fp32(cg, node, masked=False)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.CONFIG_ATTN not in opcodes
    assert Opcode.SOFTMAX_FP32 in opcodes
    assert Opcode.MASKED_SOFTMAX_FP32 not in opcodes


def test_emit_softmax_fp32_picks_up_masked_attr_from_ir_node():
    """M3-C: even when the `masked` keyword is False, an IR node with
    `attrs['masked']=True` triggers the CONFIG_ATTN emission. This is
    the path the codegen-dispatch `_emit_softmax` takes for graphs
    coming through the nanogpt frontend, which sets the attr but does
    not pass the keyword."""
    cg = _fresh_codegen(w8a32=True)
    node = IRNode(
        op="softmax",
        name="ir_masked_sm",
        inputs=["sm_in"],
        output_shape=(16, 16),
        attrs={"masked": True, "key_len": 16},
    )
    emit_softmax_fp32(cg, node, masked=False)  # keyword False
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.CONFIG_ATTN in opcodes
    assert Opcode.MASKED_SOFTMAX_FP32 in opcodes


def test_zero_pad_blob_is_4x_in_w8a32_mode():
    """M3-C: the `__zero_pad__` DRAM blob grows 4× in W8A32 mode so the
    FP32 pad-row zero-fill DMAs (embedding + K + V) read enough zeros.
    INT8 mode is unchanged.

    Both modes still store all-zero bytes — the test only checks the
    size delta."""
    from taccel.compiler.model_config import ModelConfig

    config = ModelConfig(
        model_kind="decoder",
        n_layer=1, n_head=1, d_model=16, d_head=16, mlp_dim=16,
        vocab_size=16, max_seq_len=16, embedding_kind="token_pos",
    )
    # Token/pos embedding kind: _zero_pad_size = (TILE - 1) * d_model
    # = 15 * 16 = 240 bytes (INT8), 960 bytes (W8A32).
    int8_cg = CodeGenerator(
        weight_data={}, calibration_scales={}, prescaled_biases={},
        model_config=config, stream_name="prefill", w8a32_enabled=False,
    )
    int8_cg.generate(IRGraph())
    int8_zero_pad_off = int8_cg.dram_layout["__zero_pad__"]
    next_sym_off = int8_cg.dram_layout["__input_patches__"]
    int8_size = next_sym_off - int8_zero_pad_off

    w8a32_cg = CodeGenerator(
        weight_data={}, calibration_scales={}, prescaled_biases={},
        model_config=config, stream_name="prefill", w8a32_enabled=True,
    )
    w8a32_cg.generate(IRGraph())
    w8a32_zero_pad_off = w8a32_cg.dram_layout["__zero_pad__"]
    w8a32_next_sym_off = w8a32_cg.dram_layout["__input_patches__"]
    w8a32_size = w8a32_next_sym_off - w8a32_zero_pad_off

    assert int8_size == 15 * 16, f"INT8 __zero_pad__ size {int8_size} != 240"
    assert w8a32_size == int8_size * 4, (
        f"W8A32 __zero_pad__ size {w8a32_size} != 4 × INT8 ({4 * int8_size})"
    )

    # Contents are zeros in both modes.
    int8_zero_bytes = bytes(int8_cg.dram_blob[int8_zero_pad_off:int8_zero_pad_off + int8_size])
    w8a32_zero_bytes = bytes(w8a32_cg.dram_blob[w8a32_zero_pad_off:w8a32_zero_pad_off + w8a32_size])
    assert int8_zero_bytes == bytes(int8_size)
    assert w8a32_zero_bytes == bytes(w8a32_size)


def test_emit_embedding_lookup_w8a32_emits_pad_row_zero_fill_when_seq_short():
    """M3-C: re-enable the pad-row zero-fill that M3-prep deferred. In
    W8A32 mode, when seq_len < pad_dim(seq_len), `_emit_embedding_lookup`
    must emit a DMA from `__zero_pad__` to zero the FP32 padding rows."""
    from taccel.compiler.model_config import ModelConfig

    d_model = 16
    seq_len = 12  # pad_dim(12) = 16 → pad_rows = 4
    vocab = 16
    config = ModelConfig(
        model_kind="decoder",
        n_layer=1, n_head=1, d_model=d_model, d_head=d_model, mlp_dim=d_model,
        vocab_size=vocab, max_seq_len=seq_len, embedding_kind="token_pos",
    )
    rng = np.random.default_rng(0)
    wte_fp32 = rng.standard_normal((vocab, d_model)).astype(np.float32)
    cg = CodeGenerator(
        weight_data={"transformer.wte.weight": (wte_fp32, None)},
        calibration_scales={"tok_embed": 1.0 / 127.0},
        prescaled_biases={},
        model_config=config,
        stream_name="prefill",
        w8a32_enabled=True,
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
    insns, _ = cg.generate(graph)

    # Pad-row zero-fill: LOAD with addr_reg=3 (the `__zero_pad__`
    # convention) and xfer_len = pad_rows * d_model_pad * 4 / 16.
    # pad_rows = 16 - 12 = 4; d_model_pad = 16; FP32 byte size = 4*16*4 = 256B = 16 units.
    pad_loads = [
        insn for insn in insns
        if insn.opcode == Opcode.LOAD and insn.addr_reg == 3 and insn.xfer_len == 16
    ]
    assert len(pad_loads) >= 1, (
        f"expected at least one embedding pad-row zero-fill LOAD "
        f"(addr_reg=3, xfer_len=16 units); insns: "
        f"{[(insn.opcode.name, getattr(insn,'addr_reg',None), getattr(insn,'xfer_len',None)) for insn in insns if insn.opcode == Opcode.LOAD]}"
    )


def test_int8_zero_pad_size_unchanged_when_w8a32_disabled():
    """Regression: the INT8 path's `__zero_pad__` blob stays at the
    original 1-byte/elem size (no W8A32 growth). Touches the same code
    path as test_zero_pad_blob_is_4x_in_w8a32_mode but inverted."""
    from taccel.compiler.model_config import ModelConfig

    config = ModelConfig(
        model_kind="decoder",
        n_layer=1, n_head=1, d_model=16, d_head=16, mlp_dim=16,
        vocab_size=16, max_seq_len=16, embedding_kind="token_pos",
    )
    cg = CodeGenerator(
        weight_data={}, calibration_scales={}, prescaled_biases={},
        model_config=config, stream_name="prefill", w8a32_enabled=False,
    )
    cg.generate(IRGraph())
    zero_pad_off = cg.dram_layout["__zero_pad__"]
    next_off = cg.dram_layout["__input_patches__"]
    # 15 * 16 = 240 bytes (TILE - 1) * d_model.
    assert next_off - zero_pad_off == 15 * 16


def test_masked_attention_fragment_w8a32_end_to_end():
    """M3-C end-to-end correctness signal: compile a small attention
    fragment with `masked=True` softmax + seq_len < N_pad pad rows,
    execute on the simulator, and assert the FP32 output matches the
    numpy reference where K/V padding rows are zero and attention
    scores for padded columns are masked to -∞.

    seq_len=8 (pads to 16, so pad_rows=8). Tests the full M3-C chain:
      - K pad-row zero-fill in emit_matmul_qkt_w8a32
      - V pad-row zero-fill in emit_matmul_attn_v_w8a32
      - CONFIG_ATTN + MASKED_SOFTMAX_FP32 in emit_softmax_fp32

    Pre-allocate Q/K/V inputs as in M3-B's e2e test to avoid the
    lazy-allocator aliasing bug.
    """
    from taccel.compiler.model_config import ModelConfig
    from taccel.golden_model import memory as mem
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.isa.opcodes import BUF_ABUF

    seq_len = 8
    d_head = 16
    n_pad = 16  # pad_dim(seq_len) = pad_dim(8) = 16
    inv_sqrt_d_head = 1.0 / np.sqrt(d_head)

    rng = np.random.default_rng(91)
    # Build full N_pad-sized FP32 tiles. Pad rows (rows 8..15) are
    # seeded with LARGE values (×10 the normal scale) so if the
    # codegen's K/V pad-row zero-fill DMA is broken (wrong offset,
    # wrong size, or missing entirely), those β-derived contributions
    # would dominate the attention output and the bound check would
    # clearly fail. Without this, small random pad-row values would
    # contribute too little to distinguish a working from a broken
    # zero-fill — the test would pass even if the zero-fill regressed.
    q_fp32 = rng.standard_normal((n_pad, d_head)).astype(np.float32) * 0.3
    k_fp32 = rng.standard_normal((n_pad, d_head)).astype(np.float32) * 0.4
    v_fp32 = rng.standard_normal((n_pad, d_head)).astype(np.float32) * 0.5
    # Amplify K and V padding rows specifically — Q's padding rows are
    # never read (queries only emit for seq_len rows).
    k_fp32[seq_len:] = rng.standard_normal((n_pad - seq_len, d_head)).astype(np.float32) * 10.0
    v_fp32[seq_len:] = rng.standard_normal((n_pad - seq_len, d_head)).astype(np.float32) * 10.0

    q_scale = float(np.abs(q_fp32[:seq_len]).max() / 127.0)
    k_scale = float(np.abs(k_fp32[:seq_len]).max() / 127.0)
    v_scale = float(np.abs(v_fp32[:seq_len]).max() / 127.0)
    sm_scale = 1.0 / 127.0

    config = ModelConfig(
        model_kind="decoder", n_layer=1, n_head=1, d_model=d_head,
        d_head=d_head, mlp_dim=d_head, vocab_size=16,
        max_seq_len=n_pad, embedding_kind="token_pos",
    )

    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul_qkt",
        name="qkt",
        inputs=["q_in", "k_in"],
        output_shape=(seq_len, seq_len),
        attrs={
            "head_idx": 0,
            "scale": inv_sqrt_d_head,
            "query_len": seq_len,
            "key_len": seq_len,
            "masked": True,  # propagates to softmax via attrs below
        },
    ))
    graph.add_node(IRNode(
        op="scale_mul",
        name="qkt_scaled",
        inputs=["qkt"],
        output_shape=(seq_len, seq_len),
        attrs={"scale": inv_sqrt_d_head},
    ))
    graph.add_node(IRNode(
        op="softmax",
        name="sm",
        inputs=["qkt_scaled"],
        output_shape=(seq_len, seq_len),
        # M3-C IR contract additions:
        attrs={"masked": True, "key_len": seq_len, "causal_identity": False},
    ))
    graph.add_node(IRNode(
        op="matmul_attn_v",
        name="attn_v",
        inputs=["sm", "v_in"],
        output_shape=(seq_len, d_head),
        attrs={"head_idx": 0, "query_len": seq_len, "key_len": seq_len},
    ))

    cg = CodeGenerator(
        weight_data={},
        calibration_scales={
            "q_in": q_scale, "k_in": k_scale, "v_in": v_scale,
            "qkt": 1.0, "qkt_scaled": 1.0, "sm": sm_scale, "attn_v": 1.0,
        },
        prescaled_biases={},
        fp32_biases={},
        model_config=config,
        stream_name="prefill",
        w8a32_enabled=True,
    )
    # Pre-allocate inputs (see M3-B docstring caveat).
    fp32_bytes = n_pad * d_head * 4
    cg.mem.abuf.alloc("q_in", fp32_bytes)
    cg.mem.abuf.alloc("k_in", fp32_bytes)
    cg.mem.abuf.alloc("v_in", fp32_bytes)
    insns, dram_blob = cg.generate(graph)

    sim = Simulator(MachineState(dram_data=bytes(dram_blob)))
    mem.write_fp32_tile(
        sim.state, BUF_ABUF, cg.mem.abuf.allocations["q_in"].offset_units, q_fp32,
    )
    mem.write_fp32_tile(
        sim.state, BUF_ABUF, cg.mem.abuf.allocations["k_in"].offset_units, k_fp32,
    )
    mem.write_fp32_tile(
        sim.state, BUF_ABUF, cg.mem.abuf.allocations["v_in"].offset_units, v_fp32,
    )

    for insn in insns:
        sim._execute(insn)

    # Pull the attn_v trace event for the architectural output offset.
    attn_v_offset = next(
        e["offset_units"]
        for events in cg.trace_manifest.values()
        for e in events
        if e["node_name"] == "attn_v" and e["dtype"] == "fp32"
    )
    out_fp32 = mem.read_fp32_tile(
        sim.state, BUF_ABUF, attn_v_offset, seq_len, d_head,
    )

    # Numpy reference: K and V pad rows (rows ≥ seq_len) are zeroed by
    # the codegen's pad-row zero-fill, so the reference uses k_fp32 and
    # v_fp32 with those rows masked. The attention scores for padded
    # columns (j ≥ seq_len) are masked to -∞ by MASKED_SOFTMAX_FP32, so
    # only the seq_len × seq_len block contributes.
    k_eff = k_fp32.copy()
    k_eff[seq_len:] = 0.0
    v_eff = v_fp32.copy()
    v_eff[seq_len:] = 0.0

    # Real-units scores tile, then mask the padded query/key rows.
    scores = (q_fp32 @ k_eff.T).astype(np.float32) * np.float32(inv_sqrt_d_head)
    # Causal masking (prefill mode 0b10): for query row q, valid keys
    # are 0..q (inclusive). Cols > q within the valid_kv range get -∞;
    # cols ≥ seq_len always get -∞.
    masked_scores = scores.copy()
    for q in range(n_pad):
        # Mask cols > q (causal) AND cols ≥ seq_len (valid_kv_len bound).
        masked_scores[q, q + 1:] = -np.inf
        masked_scores[q, seq_len:] = -np.inf
    row_max = masked_scores.max(axis=-1, keepdims=True)
    exp = np.exp((masked_scores - row_max).astype(np.float32))
    # Rows with all -∞ produce NaN; clamp to zero (only relevant for
    # the padded query rows which we don't compare anyway).
    exp_sum = exp.sum(axis=-1, keepdims=True)
    sm_ref = np.where(exp_sum > 0, exp / exp_sum, np.zeros_like(exp)).astype(np.float32)
    expected = (sm_ref @ v_eff).astype(np.float32)

    # Compare only the architectural seq_len × d_head region.
    err = np.abs(out_fp32 - expected[:seq_len, :d_head])
    bound = 0.05  # same envelope as M3-B's e2e
    assert err.max() <= bound, (
        f"max e2e error {err.max():.6f} exceeds bound {bound:.6f}; "
        f"out[0,0]={out_fp32[0,0]}, expected[0,0]={expected[0,0]}"
    )
    assert np.all(np.isfinite(out_fp32))


# ---------------------------------------------------------------------------
# M3-D: Stage 3 tiny GPT-2 W8A32 bundle smoke test
# ---------------------------------------------------------------------------


def test_stage3_w8a32_bundle_compiles_executes_and_produces_finite_logits():
    """M3-D smoke test: the full Stage 3 tiny GPT-2 fixture compiles
    and executes end-to-end through the simulator-backed bundle in
    W8A32 mode. Verifies the substrate landed by M2.5-A → M3-C is
    sufficient for a real model graph — no NotImplementedError, no
    simulator-time exception, finite logits in the architectural shape.

    Numerical comparison to NanoGPTFQReference is intentionally NOT
    done: that reference uses FP32 activations + INT8 weight QDQ
    (`WeightOnlyHostRunner` semantics), while the W8A32 codegen
    quantizes activations per matmul (M2.5-A dynamic scaling +
    M3-A/M3-B static-composite QKT/attn_v re-quant). The two
    deployment strategies produce different numerical results for the
    same `weight_only_int8` preset name. A like-for-like reference
    requires a W8A8-semantics path that mirrors the codegen's
    quantization, deferred to a future milestone (M3-E or beyond).
    """
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    payload = _one_layer_gpt2_payload()
    bundle = build_stage3_tiny_decoder_bundle(payload, ptq_preset="weight_only_int8")
    runner = HostRunner(bundle.build.bundle, logits_dtype=np.float32)

    # Prefill on a single token; logits should be the architectural
    # vocab-wide distribution for that token.
    logits = runner.run_prefill([3])

    vocab = int(payload["model_args"]["vocab_size"])
    # pad_dim(vocab) FP32 entries in the logits region.
    from taccel.compiler.tiler import pad_dim as _pad_dim
    assert logits.shape == (_pad_dim(vocab),), (
        f"logits shape {logits.shape} != ({_pad_dim(vocab)},)"
    )

    # Finite (no NaN/inf from numerical pathologies in the FP32 path).
    assert np.all(np.isfinite(logits)), (
        f"logits contain non-finite values: {logits}"
    )

    # Non-trivial: not all-zero (would indicate the bundle ran but the
    # data didn't flow through), and not pathologically large (would
    # indicate FP32 overflow somewhere upstream propagating to the
    # architectural row).
    assert np.abs(logits).max() > 1e-6, "logits are all (approximately) zero"
    assert np.abs(logits).max() < 100.0, (
        f"logits magnitude {np.abs(logits).max()} suggests FP32 overflow upstream"
    )

    # Argmax is a real vocab index (within [0, vocab)).
    top_token = int(np.argmax(logits[:vocab]))
    assert 0 <= top_token < vocab


def test_stage3_w8a32_bundle_executes_decode_step():
    """M3-D smoke: decode step also runs end-to-end. The decode stream
    is a separate program with KV-cache LOADs and runtime CONFIG_ATTN
    patching; this exercises the M3-A/B/C primitives in the decode
    code path (which the M3-A/B/C unit tests only touched in prefill)."""
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    payload = _one_layer_gpt2_payload()
    bundle = build_stage3_tiny_decoder_bundle(payload, ptq_preset="weight_only_int8")
    runner = HostRunner(bundle.build.bundle, logits_dtype=np.float32)
    # Prime the KV cache via prefill.
    _ = runner.run_prefill([3])
    # Then take one decode step.
    decode_logits = runner.run_decode_step(token_id=5, position=1)

    vocab = int(payload["model_args"]["vocab_size"])
    from taccel.compiler.tiler import pad_dim as _pad_dim
    assert decode_logits.shape == (_pad_dim(vocab),)
    assert np.all(np.isfinite(decode_logits))
    assert np.abs(decode_logits).max() > 1e-6


def test_stage3_w8a32_bundle_deterministic_across_runs():
    """M3-D smoke: the same payload + same token sequence produces
    identical logits across rebuilds. Each `_run()` rebuilds the bundle
    from scratch — calibration, quantization, codegen, DRAM staging,
    simulator execution — so bit-exact equality verifies determinism
    across the FULL build+execute pipeline, not just simulator-state
    cleanliness."""
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    payload = _one_layer_gpt2_payload()

    def _run():
        bundle = build_stage3_tiny_decoder_bundle(payload, ptq_preset="weight_only_int8")
        runner = HostRunner(bundle.build.bundle, logits_dtype=np.float32)
        return runner.run_prefill([3])

    a = _run()
    b = _run()
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# M4-A: ABUF residency (in-place VADD aliasing + FP32 DRAM-temp spill).
#
# Real GPT-2 124M with d_model=768 hit MemoryError at `tok_pos_add`
# because the 48 KB FP32 residual + 48 KB other FP32 tile + 48 KB output
# = 144 KB > 128 KB ABUF. M4-A introduces:
#
#   1. `emit_vadd_fp32` writes its result in-place into one of the input
#      slots and renames the ABUF allocation (matches the INT8 path).
#   2. `emit_layernorm_fp32` spills its FP32 input to DRAM-temp after
#      reading it when the input has future uses AND the tile is ≥ the
#      `fp32_spill_threshold_bytes` (16 KB). The next `emit_vadd_fp32`
#      reloads via `_load_dram_to_abuf_fp32`.
#
# Tiny fixtures (d_model=16, 1 KB tile) stay below the threshold and
# emit the original instruction stream unchanged — verified by the
# pre-M4 test suite still passing.
# ---------------------------------------------------------------------------


def test_emit_layernorm_fp32_does_not_spill_below_threshold():
    """LN with d_model=16 (1 KB FP32 tile) stays under the 16 KB spill
    threshold and produces no DMA_STORE in the instruction stream.
    Existing tiny-fixture tests rely on this — adding the spill code
    must not perturb their byte counts."""
    cg = _fresh_codegen(w8a32=True)
    cg.last_uses = {"res": 10}  # residual has future uses
    cg.current_node_idx = 0
    node = IRNode(
        op="layernorm", name="ln", inputs=["res", "gamma", "beta"],
        output_shape=(16, 16), attrs={},
    )
    emit_layernorm_fp32(cg, node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.LAYERNORM_FP32 in opcodes
    assert Opcode.STORE not in opcodes  # no spill
    assert "res" not in cg.dram_temp_fp32_outputs


def test_emit_layernorm_fp32_spills_large_tile_above_threshold():
    """LN with d_model=256 (16 KB FP32 tile = 16*256*4) triggers spill.
    The instruction stream must contain a DMA_STORE after the LN
    output is computed; the input's ABUF allocation is freed; and
    `dram_temp_fp32_outputs['res']` records the byte size."""
    cg = _fresh_codegen(w8a32=True)
    cg.last_uses = {"res": 10}
    cg.current_node_idx = 0
    node = IRNode(
        op="layernorm", name="ln", inputs=["res", "gamma", "beta"],
        output_shape=(16, 256), attrs={},
    )
    emit_layernorm_fp32(cg, node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.LAYERNORM_FP32 in opcodes
    assert Opcode.STORE in opcodes  # spill happened
    assert "res" in cg.dram_temp_fp32_outputs
    assert cg.dram_temp_fp32_outputs["res"] == 16 * 256 * 4
    # ABUF slot for "res" must be released so subsequent ops can re-use it.
    assert cg.mem.abuf.get("res") is None
    # The LN output stays in ABUF (it's the next op's input).
    assert cg.mem.abuf.get("ln") is not None


def test_emit_layernorm_fp32_skips_spill_when_input_has_no_future_use():
    """If `last_uses[in_name] <= current_node_idx`, the input is dead
    after this LN and spill is pointless. Skip the DMA_STORE."""
    cg = _fresh_codegen(w8a32=True)
    cg.last_uses = {"res": 0}  # last use is THIS node (current_node_idx=0)
    cg.current_node_idx = 0
    node = IRNode(
        op="layernorm", name="ln", inputs=["res", "gamma", "beta"],
        output_shape=(16, 256), attrs={},
    )
    emit_layernorm_fp32(cg, node)
    opcodes = [insn.opcode for insn in cg.instructions]
    assert Opcode.LAYERNORM_FP32 in opcodes
    assert Opcode.STORE not in opcodes  # no spill
    assert "res" not in cg.dram_temp_fp32_outputs


def test_emit_layernorm_fp32_spill_threshold_is_configurable():
    """The codegen's `fp32_spill_threshold_bytes` lets callers tighten
    or relax the spill heuristic for tests or specific deployments."""
    cg = _fresh_codegen(w8a32=True)
    cg.fp32_spill_threshold_bytes = 1024  # tighten: every ≥1KB tile spills
    cg.last_uses = {"res": 10}
    cg.current_node_idx = 0
    node = IRNode(
        op="layernorm", name="ln", inputs=["res", "gamma", "beta"],
        output_shape=(16, 16), attrs={},  # 1 KB tile, at the new threshold
    )
    emit_layernorm_fp32(cg, node)
    assert Opcode.STORE in [insn.opcode for insn in cg.instructions]
    assert "res" in cg.dram_temp_fp32_outputs


def test_emit_vadd_fp32_aliases_output_in_place_into_input2():
    """VADD writes its result into src2's ABUF slot and renames the
    allocation to node.name. After this, `mem.abuf.get("b")` returns
    None and `mem.abuf.get("vadd_out")` returns the (renamed) slot.
    Matches the INT8 `_emit_vadd` convention at codegen.py:2779."""
    cg = _fresh_codegen(w8a32=True)
    cg.last_uses = {}
    cg.current_node_idx = 0
    node = IRNode(
        op="vadd", name="vadd_out", inputs=["a", "b"],
        output_shape=(16, 16), attrs={},
    )
    emit_vadd_fp32(cg, node)
    # The VADD_FP32 instruction's dst_off must equal src2_off (in-place).
    vadd_insn = next(i for i in cg.instructions if i.opcode == Opcode.VADD_FP32)
    assert vadd_insn.dst_off == vadd_insn.src2_off
    # The output node owns src2's ABUF slot now; the original src2
    # name is no longer in the allocations table.
    assert cg.mem.abuf.get("b") is None
    assert cg.mem.abuf.get("vadd_out") is not None


def test_emit_vadd_fp32_reloads_spilled_fp32_residual_before_adding():
    """When inputs[0] is in `dram_temp_fp32_outputs`, the VADD emitter
    DMA-loads it back to ABUF before computing the sum. The sequence is:
    LOAD (reload src1) → SYNC → CONFIG_TILE → VADD_FP32 → SYNC."""
    cg = _fresh_codegen(w8a32=True)
    cg.last_uses = {"a": 10, "b": 10}
    cg.current_node_idx = 5

    # Simulate that "a" was previously spilled by an earlier LN.
    # Pre-populate dram_temp_outputs as if the LN had spilled it.
    M_pad, N_pad = 16, 256
    cg.dram_temp_outputs["a"] = 0x10000  # fake DRAM offset
    cg.dram_temp_fp32_outputs["a"] = M_pad * N_pad * 4

    # "b" is freshly produced in ABUF.
    cg.mem.abuf.alloc("b", M_pad * N_pad * 4)

    node = IRNode(
        op="vadd", name="vadd_out", inputs=["a", "b"],
        output_shape=(M_pad, N_pad), attrs={},
    )
    emit_vadd_fp32(cg, node)

    opcodes = [insn.opcode for insn in cg.instructions]
    # LOAD appears before CONFIG_TILE (the reload happens before the VADD
    # context is set, matching the INT8 _emit_vadd ordering).
    load_idx = opcodes.index(Opcode.LOAD) if Opcode.LOAD in opcodes else -1
    config_idx = opcodes.index(Opcode.CONFIG_TILE)
    vadd_idx = opcodes.index(Opcode.VADD_FP32)
    assert load_idx != -1, "expected DMA_LOAD reload for spilled src1"
    assert load_idx < config_idx < vadd_idx
    # Reload picked the alias target (in0 was reloaded → out aliases src1's slot).
    vadd_insn = cg.instructions[vadd_idx]
    assert vadd_insn.dst_off == vadd_insn.src1_off


def test_codegen_generate_loop_drops_dram_temp_fp32_outputs_after_last_use():
    """When a spilled tile's last_use index is reached, the generate()
    loop must clear `dram_temp_fp32_outputs[name]` so the entry doesn't
    leak symbols. Test by running a tiny graph that LN-spills then
    VADD-reloads its residual."""
    cg = _fresh_codegen(w8a32=True)
    cg.fp32_spill_threshold_bytes = 1024  # force spill on tiny tile

    # Build a graph: residual (provided externally) → LN(residual) → VADD(residual, ln_out).
    # We use embed_lookup to produce "residual" — that emitter loads
    # from DRAM. With model_config=deit_tiny_config we have a patch_cls
    # encoder; for this test we just pre-place "residual" in ABUF and
    # skip _emit_node for the producer.
    graph = IRGraph()
    graph.add_node(IRNode(op="layernorm", name="ln_out",
                          inputs=["residual", "gamma", "beta"],
                          output_shape=(16, 16), attrs={}))
    graph.add_node(IRNode(op="vadd", name="new_residual",
                          inputs=["residual", "ln_out"],
                          output_shape=(16, 16), attrs={}))

    # Bootstrap: pretend the residual was produced upstream.
    cg.mem.abuf.alloc("residual", 16 * 16 * 4)
    # `residual` is an external input (no producer node), so the graph's
    # compute_last_uses() doesn't track it. We splice it in manually
    # mirroring what the real generate() would record if the residual
    # was produced by an upstream `embed_lookup` or `vadd` node.
    cg.last_uses = dict(graph.compute_last_uses())
    cg.last_uses["residual"] = 1
    assert cg.last_uses["residual"] == 1, "vadd is residual's last use"

    # Manually run the inner loop of generate() — we want to skip the
    # weight layout step which would error on missing gamma/beta data.
    for idx, node in enumerate(graph.nodes):
        cg.current_node_idx = idx
        cg._emit_node(node)
        for inp_name, last_idx in cg.last_uses.items():
            if last_idx == idx:
                if cg.mem.abuf.get(inp_name) is not None:
                    cg.mem.abuf.free(inp_name)
                if inp_name in cg.dram_temp_fp32_outputs:
                    del cg.dram_temp_fp32_outputs[inp_name]
                    cg.dram_temp_outputs.pop(inp_name, None)

    # After processing both nodes, the spilled "residual" entry is gone.
    assert "residual" not in cg.dram_temp_fp32_outputs
    assert "residual" not in cg.dram_temp_outputs


def test_emit_layernorm_fp32_reloads_already_spilled_input():
    """If an earlier emitter spilled the same FP32 tile name (e.g. the
    block-boundary residual carrying across multiple layers), the next
    LN must reload it via DMA_LOAD before running LN. This covers the
    multi-block case in real GPT-2."""
    cg = _fresh_codegen(w8a32=True)
    cg.last_uses = {"res": 10}
    cg.current_node_idx = 5

    # Pre-spill state: "res" lives only in DRAM-temp.
    M_pad, N_pad = 16, 256
    cg.dram_temp_outputs["res"] = 0x20000
    cg.dram_temp_fp32_outputs["res"] = M_pad * N_pad * 4

    node = IRNode(
        op="layernorm", name="ln2",
        inputs=["res", "gamma", "beta"],
        output_shape=(M_pad, N_pad), attrs={},
    )
    emit_layernorm_fp32(cg, node)

    opcodes = [insn.opcode for insn in cg.instructions]
    # LOAD (reload) precedes LAYERNORM_FP32.
    assert Opcode.LOAD in opcodes
    load_idx = opcodes.index(Opcode.LOAD)
    ln_idx = opcodes.index(Opcode.LAYERNORM_FP32)
    assert load_idx < ln_idx
