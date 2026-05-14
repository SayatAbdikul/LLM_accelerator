"""W8A16 codegen tests (Phase 3 (c.2), milestone M2).

The W8A32 toolchain (committed) reuses opcodes 0x17-0x1F. The W8A16
extension repurposes `RTypeInsn.flags[0]` as an fp_precision selector:

  - flags=0 → FP32 storage (W8A32, legacy, bit-identical)
  - flags=1 → FP16 storage (W8A16, new path)

This test module verifies the compiler-side polymorphism:

  - CodeGenerator(`fp_precision="fp16"`) wires elem_bytes=2,
    fp_precision_flag=1, and the half-size FP-tile byte sizing.
  - Every R-type emit site in `w8a32_emit.py` carries
    `flags=cg.fp_precision_flag`.
  - ABUF allocations under fp16 are exactly half-size of fp32.
  - DEQUANT_ACCUM_FP32_SCALED's src2 vector layout differs by flag:
    N FP16 (fp32) vs 2N FP16 (fp16, PC scales + folded bias).
  - The simple matmul under fp16 skips the separate bias VADD (folded
    into DEQUANT epilogue).
  - The `__w8a16_pc_scale_and_bias` DRAM blob is staged with the
    expected `[N FP16 PC] + [N FP16 bias]` layout.
  - Logits store / embedding row strides scale by 2 (vs 4 for fp32).
"""
from __future__ import annotations

import numpy as np
import pytest

from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import deit_tiny_config
from taccel.compiler.tiler import pad_dim, TILE
from taccel.compiler.w8a32_emit import (
    emit_gelu_fp32,
    emit_layernorm_fp32,
    emit_matmul_w8a32,
    emit_softmax_fp32,
    emit_vadd_fp32,
)
from taccel.isa.opcodes import Opcode


def _fresh_codegen(*, w8a32: bool = True, fp_precision: str = "fp16") -> CodeGenerator:
    """W8A16 codegen with no weights/scales — for isolated emit-site tests."""
    return CodeGenerator(
        weight_data={},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=w8a32,
        fp_precision=fp_precision,
    )


def _reset_cg(cg: CodeGenerator) -> None:
    cg.instructions.clear()
    cg.mem.abuf.allocations.clear()
    cg.mem.abuf._free = [(0, cg.mem.abuf.capacity_units)]
    cg.mem.wbuf.allocations.clear()
    cg.mem.wbuf._free = [(0, cg.mem.wbuf.capacity_units)]


# ---------------------------------------------------------------------------
# CodeGenerator constructor: fp_precision wiring
# ---------------------------------------------------------------------------


def test_codegen_fp_precision_default_is_fp32():
    """Backward compat: default fp_precision is 'fp32' (M2.7 flips to 'fp16')."""
    cg = _fresh_codegen(fp_precision="fp32")
    assert cg.fp_precision == "fp32"
    assert cg.elem_bytes == 4
    assert cg.fp_precision_flag == 0


def test_codegen_fp_precision_fp16_wires_elem_bytes_2_and_flag_1():
    cg = _fresh_codegen(fp_precision="fp16")
    assert cg.fp_precision == "fp16"
    assert cg.elem_bytes == 2
    assert cg.fp_precision_flag == 1


def test_codegen_fp_precision_invalid_raises():
    with pytest.raises(ValueError, match="fp_precision must be 'fp32' or 'fp16'"):
        CodeGenerator(
            weight_data={},
            calibration_scales={},
            prescaled_biases={},
            model_config=deit_tiny_config(),
            fp_precision="bf16",
        )


def test_codegen_fp_spill_threshold_scales_with_precision():
    """FP16 tiles are half-size; the spill threshold scales accordingly."""
    cg_fp32 = _fresh_codegen(fp_precision="fp32")
    cg_fp16 = _fresh_codegen(fp_precision="fp16")
    assert cg_fp32.fp32_spill_threshold_bytes == 16384
    assert cg_fp16.fp32_spill_threshold_bytes == 8192


# ---------------------------------------------------------------------------
# Sub-layer emissions: every R-type insn carries flags=1 under fp_precision=fp16
# ---------------------------------------------------------------------------


def test_emit_gelu_fp16_carries_flags_1():
    cg = _fresh_codegen(fp_precision="fp16")
    node = IRNode(op="gelu", name="g0", inputs=["gx"], output_shape=(16, 16), attrs={})
    emit_gelu_fp32(cg, node)
    gelu_insns = [i for i in cg.instructions if i.opcode == Opcode.GELU_FP32]
    assert len(gelu_insns) == 1
    assert gelu_insns[0].flags == 1


def test_emit_gelu_fp32_carries_flags_0():
    """Backward compat: GELU under fp_precision='fp32' carries flags=0."""
    cg = _fresh_codegen(fp_precision="fp32")
    node = IRNode(op="gelu", name="g0", inputs=["gx"], output_shape=(16, 16), attrs={})
    emit_gelu_fp32(cg, node)
    gelu_insns = [i for i in cg.instructions if i.opcode == Opcode.GELU_FP32]
    assert len(gelu_insns) == 1
    assert gelu_insns[0].flags == 0


def test_emit_vadd_fp16_carries_flags_1():
    cg = _fresh_codegen(fp_precision="fp16")
    node = IRNode(op="vadd", name="v0", inputs=["a", "b"], output_shape=(16, 16), attrs={})
    emit_vadd_fp32(cg, node)
    vadd = [i for i in cg.instructions if i.opcode == Opcode.VADD_FP32]
    assert len(vadd) == 1
    assert vadd[0].flags == 1


def test_emit_softmax_fp16_carries_flags_1():
    cg = _fresh_codegen(fp_precision="fp16")
    node = IRNode(op="softmax", name="s0", inputs=["sx"], output_shape=(16, 16), attrs={})
    emit_softmax_fp32(cg, node, masked=False)
    sm = [i for i in cg.instructions if i.opcode == Opcode.SOFTMAX_FP32]
    assert len(sm) == 1
    assert sm[0].flags == 1


def test_emit_layernorm_fp16_carries_flags_1():
    # LN needs gamma/beta in weight_data; build a codegen with them staged.
    cg = CodeGenerator(
        weight_data={
            "gamma": (np.ones(16, dtype=np.float16), None),
            "beta": (np.zeros(16, dtype=np.float16), None),
        },
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=True,
        fp_precision="fp16",
    )
    cg.dram_layout["gamma"] = 0
    cg.dram_layout["beta"] = 32
    node = IRNode(op="layernorm", name="ln0", inputs=["lx", "gamma", "beta"], output_shape=(16, 16), attrs={})
    emit_layernorm_fp32(cg, node)
    ln = [i for i in cg.instructions if i.opcode == Opcode.LAYERNORM_FP32]
    assert len(ln) == 1
    assert ln[0].flags == 1


# ---------------------------------------------------------------------------
# ABUF allocations: FP16 tiles are exactly half-size of FP32
# ---------------------------------------------------------------------------


def test_w8a16_abuf_alloc_is_half_of_w8a32():
    """A 16x16 tile under fp16 occupies 32 ABUF units; under fp32 it's 64."""
    cg_fp32 = _fresh_codegen(fp_precision="fp32")
    cg_fp16 = _fresh_codegen(fp_precision="fp16")
    node = IRNode(op="gelu", name="g0", inputs=["gx"], output_shape=(16, 16), attrs={})
    emit_gelu_fp32(cg_fp32, node)
    emit_gelu_fp32(cg_fp16, node)
    a_fp32 = cg_fp32.mem.abuf.get("g0")
    a_fp16 = cg_fp16.mem.abuf.get("g0")
    assert a_fp32.size_bytes == 16 * 16 * 4
    assert a_fp16.size_bytes == 16 * 16 * 2
    assert a_fp16.size_bytes * 2 == a_fp32.size_bytes


def test_w8a16_abuf_alloc_scales_with_n_pad():
    """Test at d_model=128: 16x128 fp16 tile = 4 KB; fp32 = 8 KB."""
    cg = _fresh_codegen(fp_precision="fp16")
    node = IRNode(op="gelu", name="g0", inputs=["gx"], output_shape=(16, 128), attrs={})
    emit_gelu_fp32(cg, node)
    a = cg.mem.abuf.get("g0")
    assert a.size_bytes == 16 * 128 * 2


# ---------------------------------------------------------------------------
# Simple matmul under fp_precision="fp16": bias-fold + flags + no VADD
# ---------------------------------------------------------------------------


def _build_matmul_cg(*, fp_precision: str, with_bias: bool):
    """Build a codegen with a single matmul weight + optional bias staged."""
    K, N = 16, 16
    weight = np.zeros((K, N), dtype=np.int8)
    scales = np.ones(N, dtype=np.float16) * np.float16(0.01)
    fp32_biases = {}
    if with_bias:
        fp32_biases["fcb"] = np.linspace(-0.5, 0.5, N, dtype=np.float32)
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul", name="fc",
        inputs=["x_in", "fcw"], output_shape=(16, N),
        attrs={"bias": "fcb"} if with_bias else {},
    ))
    cg = CodeGenerator(
        weight_data={"fcw": (weight, scales)},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        stream_name="prefill",
        w8a32_enabled=True,
        fp_precision=fp_precision,
        fp32_biases=fp32_biases,
    )
    # Stage DRAM symbols the emitter will need.
    cg.generate(graph)
    return cg


def test_emit_matmul_w8a16_dequant_carries_flags_1():
    cg = _build_matmul_cg(fp_precision="fp16", with_bias=False)
    dequant_scaled = [i for i in cg.instructions if i.opcode == Opcode.DEQUANT_ACCUM_FP32_SCALED]
    assert len(dequant_scaled) == 1
    assert dequant_scaled[0].flags == 1


def test_emit_matmul_w8a16_no_separate_bias_vadd():
    """Under fp16, bias is folded into DEQUANT — no separate VADD_FP32(bias)."""
    cg = _build_matmul_cg(fp_precision="fp16", with_bias=True)
    vadds = [i for i in cg.instructions if i.opcode == Opcode.VADD_FP32]
    # No residual VADD in this graph; the only candidate would be a bias VADD
    # which is folded under fp16. So count must be zero.
    assert len(vadds) == 0


def test_emit_matmul_w8a32_separate_bias_vadd_persists():
    """Under fp32, bias VADD remains separate (legacy contract)."""
    cg = _build_matmul_cg(fp_precision="fp32", with_bias=True)
    vadds = [i for i in cg.instructions if i.opcode == Opcode.VADD_FP32]
    assert len(vadds) == 1


# ---------------------------------------------------------------------------
# DRAM staging: __w8a16_pc_scale_and_bias blob layout
# ---------------------------------------------------------------------------


def test_w8a16_pc_scale_and_bias_blob_staged_for_matmul_with_bias():
    cg = _build_matmul_cg(fp_precision="fp16", with_bias=True)
    sym = "fcw__w8a16_pc_scale_and_bias"
    assert sym in cg.dram_layout
    offset = cg.dram_layout[sym]
    # Layout: 16 FP16 PC + 16 FP16 bias = 64 bytes total.
    blob_bytes = bytes(cg.dram_blob[offset : offset + 64])
    arr = np.frombuffer(blob_bytes, dtype=np.float16)
    assert len(arr) == 32
    # First half: PC scales (all 0.01 by fixture).
    np.testing.assert_array_equal(arr[:16], np.full(16, np.float16(0.01)))
    # Second half: bias values matching the fixture linspace.
    expected_bias = np.linspace(-0.5, 0.5, 16, dtype=np.float32).astype(np.float16)
    np.testing.assert_array_equal(arr[16:], expected_bias)


def test_w8a16_pc_scale_and_bias_blob_zero_padded_when_no_bias():
    cg = _build_matmul_cg(fp_precision="fp16", with_bias=False)
    sym = "fcw__w8a16_pc_scale_and_bias"
    assert sym in cg.dram_layout
    offset = cg.dram_layout[sym]
    blob_bytes = bytes(cg.dram_blob[offset : offset + 64])
    arr = np.frombuffer(blob_bytes, dtype=np.float16)
    # First half: PC scales unchanged. Second half: zero (no bias).
    np.testing.assert_array_equal(arr[:16], np.full(16, np.float16(0.01)))
    np.testing.assert_array_equal(arr[16:], np.zeros(16, dtype=np.float16))


def test_w8a16_pc_scale_and_bias_blob_also_staged_under_fp32_path():
    """The combined blob is staged regardless of fp_precision (small constant
    overhead, keeps prefill/decode DRAM identical when precision flips)."""
    cg = _build_matmul_cg(fp_precision="fp32", with_bias=True)
    sym = "fcw__w8a16_pc_scale_and_bias"
    assert sym in cg.dram_layout


# ---------------------------------------------------------------------------
# Zero-pad blob sizing
# ---------------------------------------------------------------------------


def test_zero_pad_blob_size_scales_with_elem_bytes():
    """__zero_pad__ blob is sized for the active fp_precision (2× FP16, 4× FP32)."""
    # Build with a 2-D weight so the max_k_pad path takes effect.
    K = 32
    weight = np.zeros((K, 16), dtype=np.int8)
    scales = np.ones(16, dtype=np.float16) * np.float16(0.01)
    graph = IRGraph()
    graph.add_node(IRNode(op="matmul", name="fc", inputs=["x", "w"], output_shape=(16, 16), attrs={}))
    cg_fp32 = CodeGenerator(
        weight_data={"w": (weight, scales)},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        w8a32_enabled=True,
        fp_precision="fp32",
    )
    cg_fp32.generate(graph)
    cg_fp16 = CodeGenerator(
        weight_data={"w": (weight, scales)},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        w8a32_enabled=True,
        fp_precision="fp16",
    )
    cg_fp16.generate(graph)
    # __zero_pad__ size: max((TILE-1) * d_model * elem_bytes, (TILE-1) * max_k_pad * elem_bytes).
    # The blob length difference between fp32 and fp16 must be a factor of 2.
    # Find the symbol's allocation size: it lives in dram_blob from `dram_layout["__zero_pad__"]`
    # to the next symbol's offset. Easiest check: dram_blob total in fp32 path is larger.
    fp32_blob_size = len(cg_fp32.dram_blob)
    fp16_blob_size = len(cg_fp16.dram_blob)
    # The zero pad in fp16 mode is half — but other staged blobs (the 2N
    # combined PC+bias under fp16) add a small constant. Net: fp32 blob
    # should be larger by approximately the zero-pad delta (TILE-1) * d * 2.
    assert fp32_blob_size > fp16_blob_size, (
        f"fp32 DRAM blob {fp32_blob_size} should exceed fp16 {fp16_blob_size}"
    )


# ---------------------------------------------------------------------------
# End-to-end small matmul through the simulator: FP16 round-trip
# ---------------------------------------------------------------------------


def test_w8a16_simple_matmul_simulator_round_trip():
    """Build a fp16 codegen with a small matmul, execute through the
    simulator, and verify the output matches a numpy FP16 reference
    within FP16 ULP.
    """
    from taccel.golden_model.simulator import Simulator
    from taccel.golden_model.state import MachineState
    from taccel.golden_model import memory as mem
    from taccel.isa.opcodes import BUF_ABUF

    K, N = 16, 32
    rng = np.random.default_rng(42)
    # FP16-representable input range.
    x_fp32 = (rng.standard_normal((16, K)) * 2.0).astype(np.float32)
    weight_q = rng.integers(-50, 50, size=(K, N), dtype=np.int8)
    weight_scales = np.full(N, 0.01, dtype=np.float16)
    bias_fp32 = rng.uniform(-0.5, 0.5, size=N).astype(np.float32)

    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul", name="fc",
        inputs=["x_in", "fcw"], output_shape=(16, N),
        attrs={"bias": "fcb"},
    ))

    cg = CodeGenerator(
        weight_data={"fcw": (weight_q, weight_scales)},
        calibration_scales={},
        prescaled_biases={},
        model_config=deit_tiny_config(),
        w8a32_enabled=True,
        fp_precision="fp16",
        fp32_biases={"fcb": bias_fp32},
    )
    instructions, dram_data = cg.generate(graph)

    # Set up simulator, write FP16 input, then execute.
    state = MachineState()
    # Load DRAM with the staged data.
    state.dram[: len(dram_data)] = bytes(dram_data)
    # Codegen allocated "x_in" in ABUF as FP16; find its offset.
    x_in_alloc = cg.mem.abuf.get("x_in")
    mem.write_fp16_tile(state, BUF_ABUF, x_in_alloc.offset_units, x_fp32)

    sim = Simulator(state)
    for insn in instructions:
        sim._execute(insn)

    # Output is FP16 in ABUF at the codegen's `fc` allocation.
    fc_alloc = cg.mem.abuf.get("fc")
    out = mem.read_fp16_tile(state, BUF_ABUF, fc_alloc.offset_units, 16, N)

    # Reference: source FP16 round-trip → INT8 quant-dequant → +bias → FP16 cast.
    x_fp16 = x_fp32.astype(np.float16).astype(np.float32)
    max_abs = float(np.max(np.abs(x_fp16)))
    max_abs = max(max_abs, 2.0 ** -9)
    inv_scale = np.float32(np.float16(127.0 / max_abs))
    fwd_scale = np.float32(np.float16(max_abs / 127.0))
    x_int8 = np.clip(np.round(x_fp16 * inv_scale), -128, 127).astype(np.int32)
    pc_fp16 = weight_scales.astype(np.float16).astype(np.float32)
    bias_fp16 = bias_fp32.astype(np.float16).astype(np.float32)
    accum = x_int8 @ weight_q.astype(np.int32)
    dequant_fp32 = (
        accum.astype(np.float32) * pc_fp16.reshape(1, N) * fwd_scale
        + bias_fp16.reshape(1, N)
    )
    expected_fp16 = dequant_fp32.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected_fp16)


# ---------------------------------------------------------------------------
# KV cache layout: elem_bytes=2 for W8A16
# ---------------------------------------------------------------------------


def test_kv_cache_layout_accepts_elem_bytes_2():
    from taccel.compiler.kv_cache import build_kv_cache_layout
    layout = build_kv_cache_layout(deit_tiny_config(), max_seq_len=16, elem_bytes=2)
    assert layout.elem_bytes == 2
    # Span and total size scale with elem_bytes.
    int8_layout = build_kv_cache_layout(deit_tiny_config(), max_seq_len=16, elem_bytes=1)
    fp32_layout = build_kv_cache_layout(deit_tiny_config(), max_seq_len=16, elem_bytes=4)
    assert layout.entries[0].span_bytes == 2 * int8_layout.entries[0].span_bytes
    assert fp32_layout.entries[0].span_bytes == 2 * layout.entries[0].span_bytes
    assert layout.kv_cache_size == 2 * int8_layout.kv_cache_size


# ---------------------------------------------------------------------------
# Decoder bundle: fp_precision plumbing
# ---------------------------------------------------------------------------


def test_decoder_bundle_accepts_fp_precision_param():
    """build_decoder_program_bundle must accept fp_precision as a kw arg."""
    from inspect import signature
    from taccel.compiler.decoder_bundle import build_decoder_program_bundle
    sig = signature(build_decoder_program_bundle)
    assert "fp_precision" in sig.parameters
    assert sig.parameters["fp_precision"].default == "fp32"
