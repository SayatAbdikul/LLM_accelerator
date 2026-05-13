"""W8A32 simulator dispatch tests (Phase 3 (c.1), milestone M1).

Each new R-type opcode dispatches through the simulator and produces
results matching a numpy reference. Tests are intentionally small (16x16
tiles) so they execute in milliseconds and can run in normal CI.

Layout convention used by the W8A32 path:
  - ABUF bytes reinterpreted as FP32 (4 bytes/element) when accessed
    by the W8A32 opcodes. A 16x16 FP32 tile occupies 1024 bytes = 64
    16-byte addressing units.
  - WBUF holds FP16 vectors for LN gamma/beta and per-channel dequant
    scales (2 bytes/element).
  - ACCUM stays INT32 — the MXU's contract is unchanged.

Tests use disjoint ABUF regions for source and destination tiles so a
buggy in-place implementation doesn't silently appear correct.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from taccel.golden_model import memory as mem
from taccel.golden_model.simulator import (
    ConfigError,
    IllegalBufferError,
    Simulator,
)
from taccel.golden_model.state import MachineState
from taccel.isa.instructions import (
    ConfigAttnInsn,
    ConfigTileInsn,
    DequantAccumFp32Insn,
    DequantAccumFp32ScaledInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    MaxAbsReduceFp32Insn,
    QuantFp32Int8Insn,
    SoftmaxFp32Insn,
    VaddFp32Insn,
)
from taccel.isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF


# Buffer offsets (in 16-byte units). 16x16 FP32 tiles occupy 64 units.
SRC1_OFF = 0
SRC2_OFF = 128       # disjoint from SRC1 (FP32 16x16 = 64 units away)
DST_OFF = 256
SCALE_OFF = 512      # WBUF region for FP16 scale tables / gamma+beta
GAMMA_BETA_OFF = 640


def _make_sim_with_tile(M_tiles: int = 1, N_tiles: int = 1) -> Simulator:
    """Build a fresh simulator with CONFIG_TILE set to M_tiles x N_tiles."""
    sim = Simulator(MachineState())
    # CONFIG_TILE is 0-based encoded: value V means V+1 tiles
    sim._execute(ConfigTileInsn(M=M_tiles - 1, N=N_tiles - 1, K=0))
    return sim


def _write_fp32_to_abuf(sim: Simulator, offset_units: int, data: np.ndarray) -> None:
    mem.write_fp32_tile(sim.state, BUF_ABUF, offset_units, data)


def _read_fp32_from_abuf(sim: Simulator, offset_units: int, rows: int, cols: int) -> np.ndarray:
    return mem.read_fp32_tile(sim.state, BUF_ABUF, offset_units, rows, cols)


# ---------------------------------------------------------------------------
# memory.read_fp32_tile / write_fp32_tile round-trip
# ---------------------------------------------------------------------------


def test_fp32_tile_round_trip_in_abuf():
    """write_fp32_tile then read_fp32_tile must reproduce the input exactly
    (FP32 storage is byte-exact)."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(1)
    src = rng.standard_normal((16, 16)).astype(np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, src)
    out = _read_fp32_from_abuf(sim, SRC1_OFF, 16, 16)
    np.testing.assert_array_equal(out, src)


def test_fp32_tile_rejects_accum_buffer():
    """ACCUM stays INT32-only; FP32 read/write helpers must reject it."""
    sim = _make_sim_with_tile()
    with pytest.raises(ValueError, match="ACCUM buffer is INT32-only"):
        mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, 16, 16)
    with pytest.raises(ValueError, match="ACCUM buffer is INT32-only"):
        mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, np.zeros((16, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# DEQUANT_ACCUM_FP32
# ---------------------------------------------------------------------------


def test_dequant_accum_fp32_per_channel():
    """ACCUM[INT32] x per-col FP16 scale -> ABUF[FP32]."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(2)
    accum = rng.integers(-10_000, 10_000, size=(16, 16), dtype=np.int32)
    scales_fp32 = rng.uniform(1e-4, 1e-2, size=16).astype(np.float32)

    # Write ACCUM (offset = 0 in the int32 array)
    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    # Write per-col FP16 scales to WBUF
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, scales_fp32)

    sim._execute(
        DequantAccumFp32Insn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    # Reference: dequant uses FP16-widened-to-FP32 scales (same as REQUANT_PC).
    scales_fp16_widened = scales_fp32.astype(np.float16).astype(np.float32)
    expected = (accum.astype(np.float32) * scales_fp16_widened.reshape(1, 16)).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_dequant_accum_fp32_rejects_non_accum_src1():
    """src1 must be ACCUM (the MXU output); ABUF/WBUF src1 must raise."""
    sim = _make_sim_with_tile()
    with pytest.raises(IllegalBufferError):
        sim._execute(
            DequantAccumFp32Insn(
                src1_buf=BUF_ABUF,
                src1_off=0,
                src2_buf=BUF_WBUF,
                src2_off=SCALE_OFF,
                dst_buf=BUF_ABUF,
                dst_off=DST_OFF,
            )
        )


def test_dequant_accum_fp32_rejects_accum_dst():
    sim = _make_sim_with_tile()
    with pytest.raises(IllegalBufferError):
        sim._execute(
            DequantAccumFp32Insn(
                src1_buf=BUF_ACCUM,
                src1_off=0,
                src2_buf=BUF_WBUF,
                src2_off=SCALE_OFF,
                dst_buf=BUF_ACCUM,
                dst_off=0,
            )
        )


def test_dequant_accum_fp32_requires_config_tile():
    sim = Simulator(MachineState())
    with pytest.raises(ConfigError, match="CONFIG_TILE"):
        sim._execute(
            DequantAccumFp32Insn(
                src1_buf=BUF_ACCUM,
                src1_off=0,
                src2_buf=BUF_WBUF,
                src2_off=SCALE_OFF,
                dst_buf=BUF_ABUF,
                dst_off=DST_OFF,
            )
        )


# ---------------------------------------------------------------------------
# QUANT_FP32_INT8
# ---------------------------------------------------------------------------


def test_quant_fp32_int8_uses_sreg_scale():
    """FP32 -> INT8 using sreg-held FP16 scale. clip(round(x * scale))."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(3)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 5.0
    scale = np.float32(127.0 / 6.0)  # roughly typical activation quant

    _write_fp32_to_abuf(sim, SRC1_OFF, src)
    sim.state.scale_regs[7] = np.float16(scale)

    sim._execute(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
            sreg=7,
        )
    )

    out = mem.read_int8_tile(sim.state, BUF_ABUF, DST_OFF, 16, 16)
    expected = np.clip(
        np.round(src * np.float32(np.float16(scale))), -128, 127
    ).astype(np.int8)
    np.testing.assert_array_equal(out, expected)


def test_quant_fp32_int8_saturates_to_int8_range():
    """Large inputs must saturate to [-128, 127] not wrap."""
    sim = _make_sim_with_tile()
    src = np.full((16, 16), 1e6, dtype=np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, src)
    sim.state.scale_regs[0] = np.float16(1.0)

    sim._execute(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
            sreg=0,
        )
    )

    out = mem.read_int8_tile(sim.state, BUF_ABUF, DST_OFF, 16, 16)
    assert (out == 127).all()


# ---------------------------------------------------------------------------
# VADD_FP32
# ---------------------------------------------------------------------------


def test_vadd_fp32_element_wise_add():
    """FP32 + FP32 with no clipping (the W8A32 residual stream contract)."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(4)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, a)
    _write_fp32_to_abuf(sim, SRC2_OFF, b)

    sim._execute(
        VaddFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=SRC2_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    np.testing.assert_array_equal(out, (a + b).astype(np.float32))


def test_vadd_fp32_does_not_saturate():
    """Unlike INT8 VADD, FP32 VADD must not saturate or clip."""
    sim = _make_sim_with_tile()
    a = np.full((16, 16), 1e10, dtype=np.float32)
    b = np.full((16, 16), 1e10, dtype=np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, a)
    _write_fp32_to_abuf(sim, SRC2_OFF, b)
    sim._execute(
        VaddFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=SRC2_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )
    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    assert (out == np.float32(2e10)).all()


# ---------------------------------------------------------------------------
# LAYERNORM_FP32
# ---------------------------------------------------------------------------


def test_layernorm_fp32_matches_numpy_reference():
    """Welford-style LN with eps=1e-6, FP16 gamma+beta widened to FP32."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(5)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 2.0
    gamma = rng.uniform(0.5, 1.5, size=16).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=16).astype(np.float32)

    _write_fp32_to_abuf(sim, SRC1_OFF, src)
    # gamma followed by beta (2N FP16 values)
    mem.write_fp16_vector(sim.state, BUF_WBUF, GAMMA_BETA_OFF, np.concatenate([gamma, beta]))

    sim._execute(
        LayernormFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_WBUF,
            src2_off=GAMMA_BETA_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    # Reference: eps=1e-6, gamma/beta widened through FP16.
    gamma_fp16 = gamma.astype(np.float16).astype(np.float32)
    beta_fp16 = beta.astype(np.float16).astype(np.float32)
    mean = src.mean(axis=-1, keepdims=True).astype(np.float32)
    var = src.var(axis=-1, keepdims=True).astype(np.float32)
    eps = np.float32(1e-6)
    expected = ((src - mean) / np.sqrt(var + eps) * gamma_fp16 + beta_fp16).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# GELU_FP32
# ---------------------------------------------------------------------------


def test_gelu_fp32_matches_gelu_new_reference():
    """tanh-based GELU (gelu_new) matching the GPT-2 production checkpoint."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(6)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 3.0
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        GeluFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    sqrt_2_over_pi = np.float32(np.sqrt(2.0 / np.pi))
    inner = sqrt_2_over_pi * (src + np.float32(0.044715) * src ** 3)
    expected = (src * np.float32(0.5) * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# SOFTMAX_FP32
# ---------------------------------------------------------------------------


def test_softmax_fp32_rows_sum_to_one():
    """Each row's softmax must sum to 1.0 within FP32 ulp."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(7)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 4.0
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        SoftmaxFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    row_sums = out.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=1e-6)
    # And matches reference exactly.
    row_max = src.max(axis=-1, keepdims=True)
    exp_row = np.exp((src - row_max).astype(np.float32))
    expected = (exp_row / exp_row.sum(axis=-1, keepdims=True)).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# MASKED_SOFTMAX_FP32
# ---------------------------------------------------------------------------


def test_masked_softmax_fp32_causal_masking():
    """Row i should attend to keys 0..i only (causal). Beyond i the
    probabilities must be exactly 0; valid columns must sum to 1.
    """
    sim = _make_sim_with_tile()
    # Configure attention context: query_row_base=0 (start of sequence),
    # valid_kv_len=16 (matches the N=16 key columns), mode=1 (prefill causal).
    sim._execute(ConfigAttnInsn(query_row_base=0, valid_kv_len=16, mode=1))

    rng = np.random.default_rng(8)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 2.0
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        MaskedSoftmaxFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    # Every row should be causal:
    for i in range(16):
        # Columns > i are masked to zero.
        assert (out[i, i + 1:] == 0.0).all(), f"row {i}: non-causal entries"
        # Valid columns (0..i) sum to 1.
        np.testing.assert_allclose(out[i, : i + 1].sum(), 1.0, rtol=0, atol=1e-6)


def test_masked_softmax_fp32_requires_config_attn():
    """Without CONFIG_ATTN set, masked softmax must raise — matches the
    contract of the existing INT8 MASKED_SOFTMAX op."""
    sim = _make_sim_with_tile()
    # CONFIG_TILE is set, CONFIG_ATTN is not.
    with pytest.raises(ConfigError, match="CONFIG_ATTN"):
        sim._execute(
            MaskedSoftmaxFp32Insn(
                src1_buf=BUF_ABUF,
                src1_off=SRC1_OFF,
                src2_buf=BUF_ABUF,
                src2_off=0,
                dst_buf=BUF_ABUF,
                dst_off=DST_OFF,
            )
        )


# ---------------------------------------------------------------------------
# Composition smoke: DEQUANT_ACCUM_FP32 -> VADD_FP32 -> QUANT_FP32_INT8
# ---------------------------------------------------------------------------


def test_w8a32_op_composition_smoke():
    """Smoke-test the canonical W8A32 instruction sequence end-to-end:
    INT32 accumulator -> dequant to FP32 -> residual add -> back to INT8
    for the next matmul. This is what the codegen will emit on every
    matmul output node in the W8A32 lowering.
    """
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(9)

    # Stage 1: ACCUM holds a matmul result; per-col scales in WBUF.
    accum = rng.integers(-1_000, 1_000, size=(16, 16), dtype=np.int32)
    scales = rng.uniform(1e-3, 5e-3, size=16).astype(np.float32)
    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, scales)

    sim._execute(
        DequantAccumFp32Insn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )

    # Stage 2: residual stream FP32 add. Use the dequant output as src1,
    # write a known FP32 residual to SRC2_OFF.
    residual = rng.standard_normal((16, 16)).astype(np.float32) * 0.5
    _write_fp32_to_abuf(sim, SRC2_OFF, residual)
    sim._execute(
        VaddFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=DST_OFF,
            src2_buf=BUF_ABUF,
            src2_off=SRC2_OFF,
            dst_buf=BUF_ABUF,
            dst_off=SRC1_OFF,  # write the result somewhere disjoint
        )
    )

    # Stage 3: feed the FP32 result back into INT8 ABUF via the dynamic
    # activation quant. Use a synthetic scale.
    sim.state.scale_regs[3] = np.float16(127.0 / 4.0)
    INT8_DST = DST_OFF + 64  # disjoint from FP32 tiles
    sim._execute(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=INT8_DST,
            sreg=3,
        )
    )

    # Reference end-to-end:
    scales_widened = scales.astype(np.float16).astype(np.float32)
    fp32_after_dequant = (accum.astype(np.float32) * scales_widened.reshape(1, 16)).astype(np.float32)
    fp32_after_add = (fp32_after_dequant + residual).astype(np.float32)
    scale_widened = np.float32(np.float16(127.0 / 4.0))
    expected_int8 = np.clip(
        np.round(fp32_after_add * scale_widened), -128, 127
    ).astype(np.int8)

    out_int8 = mem.read_int8_tile(sim.state, BUF_ABUF, INT8_DST, 16, 16)
    np.testing.assert_array_equal(out_int8, expected_int8)


# ===========================================================================
# M2.5-A — dynamic per-matmul activation scaling primitives
# ===========================================================================
#
# The M1 ops above assume the activation quant scale is known at compile time
# (it lives in a scale register loaded by SET_SCALE before the matmul). M2.5-A
# adds two opcodes that let the matmul prelude compute the activation scale
# at runtime from the actual FP32 tile:
#
#   MAX_ABS_REDUCE_FP32:
#     scan an FP32 ABUF tile, compute max(|x|), write a derived FP16 scale
#     pair to scale_regs[sreg], scale_regs[sreg+1]:
#       scale_regs[sreg]   = 127 / max_abs  (multiplier for QUANT_FP32_INT8)
#       scale_regs[sreg+1] = max_abs / 127  (factor for DEQUANT_ACCUM_FP32_SCALED)
#     Eps-clamped (1e-10) for the all-zero tile so the inverse stays finite.
#     `sreg+1 <= 15` enforced at execution time (sreg in 0..14).
#
#   DEQUANT_ACCUM_FP32_SCALED:
#     M1's DEQUANT_ACCUM_FP32 math, plus an extra `* scale_regs[sreg]`
#     factor. Used to undo the dynamic activation quant on the matmul
#     output. M1's 0x17 stays bit-identical — this is a separate opcode
#     (0x1F) that adds the scalar multiply.


# ---------------------------------------------------------------------------
# MAX_ABS_REDUCE_FP32
# ---------------------------------------------------------------------------


def test_max_abs_reduce_fp32_writes_inverse_scale_pair():
    """MAX_ABS_REDUCE_FP32 over a known FP32 tile writes both
    `127/max_abs` and `max_abs/127` as FP16 to scale_regs[sreg] and
    scale_regs[sreg+1] respectively. The pair convention matches
    DEQUANT_ADD's two-sreg contract.
    """
    sim = _make_sim_with_tile()
    src = np.zeros((16, 16), dtype=np.float32)
    # One known maximum so the expected scale is bit-exact under FP16.
    src[3, 7] = 2.5
    src[10, 4] = -4.0  # max(|x|) = 4.0
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=0,
            sreg=5,
        )
    )

    expected_inv = np.float16(127.0 / 4.0)   # quant input scale
    expected_fwd = np.float16(4.0 / 127.0)   # dequant output scale
    assert sim.state.scale_regs[5] == expected_inv
    assert sim.state.scale_regs[6] == expected_fwd


def test_max_abs_reduce_fp32_eps_clamps_all_zero_tile():
    """An all-zero tile would make `127/0 = inf`. The eps clamp ensures
    both `127/eps` and `eps/127` are FP16-representable. The clamp
    constant `2**-9 = 1/512` gives `inv = 127*512 = 65024`, just under
    FP16 max 65504. Downstream QUANT_FP32_INT8 on the same all-zero
    source still produces zeros, so the clamp value does not affect
    correctness in this corner — only the finiteness invariant matters.
    """
    sim = _make_sim_with_tile()
    src = np.zeros((16, 16), dtype=np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=0,
            sreg=0,
        )
    )

    # 1) Both scales must be finite (no inf, no nan) — the eps clamp's
    #    only architectural responsibility.
    assert np.isfinite(sim.state.scale_regs[0])
    assert np.isfinite(sim.state.scale_regs[1])

    # 2) Specific clamped values: eps = 2**-9 → inv = 127*512 = 65024,
    #    fwd = 1/(127*512) ≈ 1.54e-5 (FP16 subnormal).
    expected_inv = np.float16(127.0 / (2.0 ** -9))
    expected_fwd = np.float16((2.0 ** -9) / 127.0)
    assert sim.state.scale_regs[0] == expected_inv
    assert sim.state.scale_regs[1] == expected_fwd


def test_max_abs_reduce_fp32_accepts_sreg_14():
    """`sreg=14` writes scale_regs[14] and scale_regs[15] — the last legal
    pair. Verify the simulator accepts this boundary value."""
    sim = _make_sim_with_tile()
    src = np.full((16, 16), 1.0, dtype=np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=0,
            sreg=14,
        )
    )

    assert sim.state.scale_regs[14] == np.float16(127.0)
    assert sim.state.scale_regs[15] == np.float16(1.0 / 127.0)


def test_max_abs_reduce_fp32_rejects_sreg_15():
    """`sreg=15` would require scale_regs[16] for the forward scale, which
    doesn't exist. The simulator must raise (matches DEQUANT_ADD's
    sreg-pair convention)."""
    sim = _make_sim_with_tile()
    src = np.ones((16, 16), dtype=np.float32)
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    with pytest.raises(ConfigError, match="MAX_ABS_REDUCE_FP32 sreg"):
        sim._execute(
            MaxAbsReduceFp32Insn(
                src1_buf=BUF_ABUF,
                src1_off=SRC1_OFF,
                src2_buf=BUF_ABUF,
                src2_off=0,
                dst_buf=BUF_ABUF,
                dst_off=0,
                sreg=15,
            )
        )


def test_max_abs_reduce_fp32_rejects_accum_src1():
    """ACCUM is INT32-only; the FP32 reduce must reject ACCUM src1."""
    sim = _make_sim_with_tile()
    with pytest.raises(IllegalBufferError):
        sim._execute(
            MaxAbsReduceFp32Insn(
                src1_buf=BUF_ACCUM,
                src1_off=0,
                src2_buf=BUF_ABUF,
                src2_off=0,
                dst_buf=BUF_ABUF,
                dst_off=0,
                sreg=0,
            )
        )


def test_max_abs_reduce_fp32_requires_config_tile():
    sim = Simulator(MachineState())
    with pytest.raises(ConfigError, match="CONFIG_TILE"):
        sim._execute(
            MaxAbsReduceFp32Insn(
                src1_buf=BUF_ABUF,
                src1_off=SRC1_OFF,
                src2_buf=BUF_ABUF,
                src2_off=0,
                dst_buf=BUF_ABUF,
                dst_off=0,
                sreg=0,
            )
        )


# ---------------------------------------------------------------------------
# DEQUANT_ACCUM_FP32_SCALED
# ---------------------------------------------------------------------------


def test_dequant_accum_fp32_scaled_per_channel_times_activation_scale():
    """ACCUM[INT32] × per-col FP16 weight scale × FP16 activation scale
    -> ABUF[FP32]. Matches M1's DEQUANT_ACCUM_FP32 but with the extra
    activation-scale factor.
    """
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(11)
    accum = rng.integers(-10_000, 10_000, size=(16, 16), dtype=np.int32)
    wt_scales_fp32 = rng.uniform(1e-4, 1e-2, size=16).astype(np.float32)
    act_scale = np.float32(0.0625)  # exact in FP16

    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, wt_scales_fp32)
    sim.state.scale_regs[2] = np.float16(act_scale)

    sim._execute(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
            sreg=2,
        )
    )

    out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)
    wt_widened = wt_scales_fp32.astype(np.float16).astype(np.float32)
    act_widened = np.float32(np.float16(act_scale))
    expected = (
        accum.astype(np.float32) * wt_widened.reshape(1, 16) * act_widened
    ).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_dequant_accum_fp32_scaled_with_unit_scale_matches_m1():
    """With activation scale = 1.0 (FP16), DEQUANT_ACCUM_FP32_SCALED must
    produce the same output as M1's DEQUANT_ACCUM_FP32 — the M2.5-A op
    is a strict superset, not a replacement."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(12)
    accum = rng.integers(-5_000, 5_000, size=(16, 16), dtype=np.int32)
    wt_scales = rng.uniform(1e-4, 1e-2, size=16).astype(np.float32)
    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, wt_scales)

    # Run M1 op
    sim._execute(
        DequantAccumFp32Insn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF,
        )
    )
    m1_out = _read_fp32_from_abuf(sim, DST_OFF, 16, 16)

    # Run M2.5-A op with activation scale = 1.0
    sim.state.scale_regs[3] = np.float16(1.0)
    M25_DST = DST_OFF + 64
    sim._execute(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF,
            dst_off=M25_DST,
            sreg=3,
        )
    )
    m25_out = _read_fp32_from_abuf(sim, M25_DST, 16, 16)

    # With act_scale = 1.0, outputs must be bit-identical.
    np.testing.assert_array_equal(m1_out, m25_out)


def test_dequant_accum_fp32_scaled_rejects_non_accum_src1():
    sim = _make_sim_with_tile()
    sim.state.scale_regs[0] = np.float16(1.0)
    with pytest.raises(IllegalBufferError):
        sim._execute(
            DequantAccumFp32ScaledInsn(
                src1_buf=BUF_ABUF,
                src1_off=0,
                src2_buf=BUF_WBUF,
                src2_off=SCALE_OFF,
                dst_buf=BUF_ABUF,
                dst_off=DST_OFF,
                sreg=0,
            )
        )


def test_dequant_accum_fp32_scaled_rejects_accum_dst():
    sim = _make_sim_with_tile()
    sim.state.scale_regs[0] = np.float16(1.0)
    with pytest.raises(IllegalBufferError):
        sim._execute(
            DequantAccumFp32ScaledInsn(
                src1_buf=BUF_ACCUM,
                src1_off=0,
                src2_buf=BUF_WBUF,
                src2_off=SCALE_OFF,
                dst_buf=BUF_ACCUM,
                dst_off=0,
                sreg=0,
            )
        )


def test_dequant_accum_fp32_scaled_requires_config_tile():
    sim = Simulator(MachineState())
    sim.state.scale_regs[0] = np.float16(1.0)
    with pytest.raises(ConfigError, match="CONFIG_TILE"):
        sim._execute(
            DequantAccumFp32ScaledInsn(
                src1_buf=BUF_ACCUM,
                src1_off=0,
                src2_buf=BUF_WBUF,
                src2_off=SCALE_OFF,
                dst_buf=BUF_ABUF,
                dst_off=DST_OFF,
                sreg=0,
            )
        )


# ---------------------------------------------------------------------------
# Composition: matmul prelude
#   MAX_ABS_REDUCE_FP32 → QUANT_FP32_INT8 → (eye-matmul stub) → DEQUANT_ACCUM_FP32_SCALED
#
# This is the canonical M2.5-A use case: dynamically rescale the FP32
# activation tile so its INT8 quant uses the full [-127, 127] range, then
# undo that scaling on the matmul output. The reference computation is
# bit-exact thanks to FP16 widening on both the inverse and forward scales.
# ---------------------------------------------------------------------------


def test_matmul_prelude_composition_round_trips_fp32_via_int8():
    """End-to-end M2.5-A flow against an identity weight (eye(16), per-channel
    scale = 1.0):

        FP32 → MAX_ABS_REDUCE → QUANT (with S[s]=127/max)
             → INT8
             → INT8 matmul against eye(16) → INT32 = INT8 promoted
             → DEQUANT_SCALED (with weights=1.0_fp16, S[s+1]=max/127)
             → FP32

    Result must equal `int8.astype(fp32) * fp16(max/127)` — bit-exact under
    `array_equal`. Any drift from this reference would indicate the
    simulator's FP16 widening contract changed.

    We skip executing the actual MATMUL opcode (it requires WBUF layout
    setup that's irrelevant to M2.5-A correctness) and stage the matmul
    result into ACCUM directly. The matmul output for `x @ eye(16)` is
    `x` promoted to INT32 — that's exactly what we write.
    """
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(13)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 3.5
    _write_fp32_to_abuf(sim, SRC1_OFF, src)

    # Per-channel weight scales = 1.0 (FP16). For eye(16) weights, this
    # makes the matmul output a pure identity in the dequant path.
    wt_scales = np.ones(16, dtype=np.float32)
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, wt_scales)

    # Stage 1: MAX_ABS_REDUCE writes 127/max to S[s], max/127 to S[s+1].
    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=0,
            sreg=4,
        )
    )

    # Stage 2: QUANT using S[s] (127/max).
    INT8_DST = DST_OFF
    sim._execute(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF,
            src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=INT8_DST,
            sreg=4,
        )
    )
    int8_out = mem.read_int8_tile(sim.state, BUF_ABUF, INT8_DST, 16, 16)

    # Stage 3: simulated matmul. For w = eye(16), the matmul result in
    # ACCUM is exactly x_int8 widened to INT32 (column j of x · eye(16) = x_j).
    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, int8_out.astype(np.int32))

    # Stage 4: DEQUANT_SCALED with weight scales = 1.0 and S[s+1] = max/127.
    sim._execute(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM,
            src1_off=0,
            src2_buf=BUF_WBUF,
            src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF,
            dst_off=DST_OFF + 128,  # disjoint from the INT8 staging tile
            sreg=5,                  # S[s+1] = S[5]
        )
    )

    out_fp32 = _read_fp32_from_abuf(sim, DST_OFF + 128, 16, 16)

    # Reference end-to-end. Match the simulator's FP16 widening exactly so
    # the assertion is array_equal, not allclose.
    max_abs = float(np.max(np.abs(src.astype(np.float32))))
    max_abs_eps = max(max_abs, 1e-10)
    inv_fp32 = np.float32(np.float16(127.0 / max_abs_eps))
    fwd_fp32 = np.float32(np.float16(max_abs_eps / 127.0))
    wt_widened = np.float32(np.float16(1.0))
    x_int8_ref = np.clip(
        np.round(src * inv_fp32), -128, 127
    ).astype(np.int8)
    expected = (
        x_int8_ref.astype(np.int32).astype(np.float32) * wt_widened * fwd_fp32
    ).astype(np.float32)

    # The QUANT stage's INT8 output must match the reference too.
    np.testing.assert_array_equal(int8_out, x_int8_ref)
    np.testing.assert_array_equal(out_fp32, expected)

    # And the dynamic round-trip preserves the FP32 input within INT8
    # quantization noise. Per-element error <= 0.5*fwd_fp32 (rounding step)
    # plus |x|_max * FP16_EPS (the FP16 round-off on inv_fp32*fwd_fp32 ≠ 1
    # exactly). FP16 has 10 mantissa bits → relative epsilon ≈ 2^-10.
    err = np.abs(out_fp32 - src)
    fp16_relative_eps = 2.0 ** -10
    bound = 0.5 * fwd_fp32 + float(np.abs(src).max()) * fp16_relative_eps + 1e-6
    assert err.max() <= bound, (
        f"max round-trip error {err.max()} exceeds bound {bound} "
        f"(0.5*step={0.5 * fwd_fp32}, |x|max*fp16_eps={np.abs(src).max() * fp16_relative_eps})"
    )
