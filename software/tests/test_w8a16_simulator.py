"""W8A16 simulator dispatch tests (Phase 3 (c.2), milestone M1).

The 9 W8A32 R-type opcodes (0x17-0x1F) are polymorphic on `RTypeInsn.flags[0]`:

  - flags=0 → FP32 storage (W8A32, existing, bit-identical)
  - flags=1 → FP16 storage (W8A16, new default)

Internal datapath stays FP32 (LN variance, softmax exp, GELU x^3 all in FP32);
the FP16 helpers widen on read and downcast on write. For
DEQUANT_ACCUM_FP32_SCALED specifically, `flags=1` also changes src2 layout:
src2 = 2N FP16 (N PC scales + N bias values), and the dequant epilogue folds
bias before casting to FP16 once — avoiding FP16 double-rounding.

Tests use 16x16 tiles and disjoint ABUF regions so a buggy in-place
implementation doesn't silently appear correct.
"""
from __future__ import annotations

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


# Buffer offsets (in 16-byte units). 16x16 FP16 tiles occupy 32 units (half of FP32).
SRC1_OFF = 0
SRC2_OFF = 64        # disjoint from SRC1 (FP16 16x16 = 32 units away)
DST_OFF = 128
SCALE_OFF = 512      # WBUF region for FP16 scale tables (PC scales / gamma+beta)
GAMMA_BETA_OFF = 640


def _make_sim_with_tile(M_tiles: int = 1, N_tiles: int = 1) -> Simulator:
    """Build a fresh simulator with CONFIG_TILE set to M_tiles x N_tiles."""
    sim = Simulator(MachineState())
    sim._execute(ConfigTileInsn(M=M_tiles - 1, N=N_tiles - 1, K=0))
    return sim


def _write_fp16_to_abuf(sim: Simulator, offset_units: int, data: np.ndarray) -> None:
    mem.write_fp16_tile(sim.state, BUF_ABUF, offset_units, data)


def _read_fp16_from_abuf(sim: Simulator, offset_units: int, rows: int, cols: int) -> np.ndarray:
    return mem.read_fp16_tile(sim.state, BUF_ABUF, offset_units, rows, cols)


# ---------------------------------------------------------------------------
# memory.read_fp16_tile / write_fp16_tile round-trip
# ---------------------------------------------------------------------------


def test_fp16_tile_round_trip_in_abuf():
    """write_fp16_tile then read_fp16_tile must reproduce input up to FP16
    precision (FP16 storage truncates mantissa)."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(1)
    src = rng.standard_normal((16, 16)).astype(np.float32)
    _write_fp16_to_abuf(sim, SRC1_OFF, src)
    out = _read_fp16_from_abuf(sim, SRC1_OFF, 16, 16)
    # FP16 storage round-trip — value must equal source-cast-to-FP16.
    np.testing.assert_array_equal(out, src.astype(np.float16).astype(np.float32))


def test_fp16_tile_storage_is_half_of_fp32():
    """A 16x16 FP16 tile must occupy exactly 16*16*2 = 512 bytes (half of FP32)."""
    sim = _make_sim_with_tile()
    src = np.ones((16, 16), dtype=np.float32)
    _write_fp16_to_abuf(sim, SRC1_OFF, src)
    # Reading 32 16-byte units = 512 bytes from ABUF should contain the FP16 tile.
    raw = mem.read_bytes(sim.state, BUF_ABUF, SRC1_OFF, 16 * 16 * 2)
    assert len(raw) == 512
    # Reading the next 32 units should be unaffected (zero-initialized).
    next_raw = mem.read_bytes(sim.state, BUF_ABUF, SRC1_OFF + 32, 16 * 16 * 2)
    assert next_raw == b"\x00" * 512


def test_fp16_tile_rejects_accum_buffer():
    """ACCUM stays INT32-only; FP16 read/write helpers must reject it."""
    sim = _make_sim_with_tile()
    with pytest.raises(ValueError, match="ACCUM buffer is INT32-only"):
        mem.read_fp16_tile(sim.state, BUF_ACCUM, 0, 16, 16)
    with pytest.raises(ValueError, match="ACCUM buffer is INT32-only"):
        mem.write_fp16_tile(sim.state, BUF_ACCUM, 0, np.zeros((16, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# DEQUANT_ACCUM_FP32 (flags=1) — INT32 ACCUM × FP16 PC scale → FP16 ABUF
# ---------------------------------------------------------------------------


def test_dequant_accum_fp32_flags1_writes_fp16_output():
    """flags=1: INT32 ACCUM × per-col FP16 scale → FP16 ABUF (downcast on store)."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(2)
    accum = rng.integers(-1_000, 1_000, size=(16, 16), dtype=np.int32)
    scales_fp32 = rng.uniform(1e-3, 1e-1, size=16).astype(np.float32)

    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, scales_fp32)

    sim._execute(
        DequantAccumFp32Insn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    scales_fp16 = scales_fp32.astype(np.float16).astype(np.float32)
    expected_fp32 = (accum.astype(np.float32) * scales_fp16.reshape(1, 16)).astype(np.float32)
    expected_fp16 = expected_fp32.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected_fp16)


# ---------------------------------------------------------------------------
# QUANT_FP32_INT8 (flags=1) — FP16 source → INT8 output
# ---------------------------------------------------------------------------


def test_quant_fp32_int8_flags1_reads_fp16_source():
    """flags=1: FP16 ABUF source → INT8 output using sreg-held FP16 scale."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(4)
    # Constrain source so FP16-representable values dominate.
    src = (rng.standard_normal((16, 16)) * 4.0).astype(np.float32)
    scale = np.float32(127.0 / 6.0)

    _write_fp16_to_abuf(sim, SRC1_OFF, src)
    sim.state.scale_regs[7] = np.float16(scale)

    sim._execute(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            sreg=7,
            flags=1,
        )
    )

    out = mem.read_int8_tile(sim.state, BUF_ABUF, DST_OFF, 16, 16)
    # Reference: source widened from FP16 to FP32, then quantized.
    src_fp16 = src.astype(np.float16).astype(np.float32)
    expected = np.clip(
        np.round(src_fp16 * np.float32(np.float16(scale))), -128, 127
    ).astype(np.int8)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# VADD_FP32 (flags=1) — FP16+FP16 → FP16
# ---------------------------------------------------------------------------


def test_vadd_fp32_flags1_fp16_storage_with_fp32_internal_sum():
    """FP16 sources widened to FP32 for the sum, FP16 store on output."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(5)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    _write_fp16_to_abuf(sim, SRC1_OFF, a)
    _write_fp16_to_abuf(sim, SRC2_OFF, b)

    sim._execute(
        VaddFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=SRC2_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    # Reference: source widened FP16→FP32, sum in FP32, store FP16.
    a_fp16 = a.astype(np.float16).astype(np.float32)
    b_fp16 = b.astype(np.float16).astype(np.float32)
    expected = (a_fp16 + b_fp16).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_vadd_fp32_flags1_saturates_to_fp16_inf():
    """FP16 stores must saturate to +inf when sum exceeds FP16 max (~65504).

    No clip in the simulator — the natural FP16 cast produces +inf, which is
    the correct hardware contract for FP16 overflow.
    """
    sim = _make_sim_with_tile()
    a = np.full((16, 16), 50000.0, dtype=np.float32)
    b = np.full((16, 16), 50000.0, dtype=np.float32)
    _write_fp16_to_abuf(sim, SRC1_OFF, a)
    _write_fp16_to_abuf(sim, SRC2_OFF, b)
    sim._execute(
        VaddFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=SRC2_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )
    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    # 50000 + 50000 = 100000 > FP16 max (65504); FP16 cast → +inf.
    assert np.isinf(out).all() and (out > 0).all()


# ---------------------------------------------------------------------------
# LAYERNORM_FP32 (flags=1) — FP16 I/O with FP32-internal variance
# ---------------------------------------------------------------------------


def test_layernorm_fp32_flags1_matches_fp32_reference_downcast():
    """FP16 LN must match an FP32-internal reference downcast to FP16."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(6)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 2.0
    gamma = rng.uniform(0.5, 1.5, size=16).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=16).astype(np.float32)

    _write_fp16_to_abuf(sim, SRC1_OFF, src)
    mem.write_fp16_vector(sim.state, BUF_WBUF, GAMMA_BETA_OFF, np.concatenate([gamma, beta]))

    sim._execute(
        LayernormFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_WBUF, src2_off=GAMMA_BETA_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    # Reference: FP16 source widened, variance/normalize in FP32, FP16 store.
    src_fp16 = src.astype(np.float16).astype(np.float32)
    gamma_fp16 = gamma.astype(np.float16).astype(np.float32)
    beta_fp16 = beta.astype(np.float16).astype(np.float32)
    mean = src_fp16.mean(axis=-1, keepdims=True).astype(np.float32)
    var = src_fp16.var(axis=-1, keepdims=True).astype(np.float32)
    eps = np.float32(1e-5)  # matches simulator _exec_layernorm_fp32 / GPT-2 checkpoint
    expected_fp32 = ((src_fp16 - mean) / np.sqrt(var + eps) * gamma_fp16 + beta_fp16).astype(np.float32)
    expected_fp16 = expected_fp32.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected_fp16)


def test_layernorm_fp32_flags1_variance_upcast_no_overflow():
    """Stress test: |x| in [200, 300] would overflow naive FP16 variance
    reduction (x^2 > FP16 max for |x| > 256). Simulator MUST upcast to FP32
    internally for the reduction so the result stays finite.
    """
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(7)
    # All values in [200, 300] — squaring 300 = 90000 > FP16 max.
    src = (200.0 + 100.0 * rng.uniform(size=(16, 16))).astype(np.float32)
    gamma = np.ones(16, dtype=np.float32)
    beta = np.zeros(16, dtype=np.float32)

    _write_fp16_to_abuf(sim, SRC1_OFF, src)
    mem.write_fp16_vector(sim.state, BUF_WBUF, GAMMA_BETA_OFF, np.concatenate([gamma, beta]))

    sim._execute(
        LayernormFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_WBUF, src2_off=GAMMA_BETA_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )
    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    assert np.isfinite(out).all(), "LN FP16 must not overflow when |x| > 256"
    # Normalized outputs should be in [-3, 3] approximately (3-sigma range).
    assert (np.abs(out) < 10.0).all(), f"LN output out of expected range: max abs = {np.abs(out).max()}"


# ---------------------------------------------------------------------------
# GELU_FP32 (flags=1) — FP16 I/O with FP32-internal polynomial
# ---------------------------------------------------------------------------


def test_gelu_fp32_flags1_matches_fp32_reference_downcast():
    """FP16 GELU must match FP32-internal gelu_new downcast to FP16."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(8)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 3.0
    _write_fp16_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        GeluFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    # Reference: source widened, gelu_new in FP32, FP16 store.
    xf = src.astype(np.float16).astype(np.float32)
    sqrt_2_over_pi = np.float32(np.sqrt(2.0 / np.pi))
    inner = sqrt_2_over_pi * (xf + np.float32(0.044715) * xf ** 3)
    expected_fp32 = (xf * np.float32(0.5) * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)
    expected_fp16 = expected_fp32.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected_fp16)


def test_gelu_fp32_flags1_x_cubed_upcast_no_overflow():
    """Stress test: |x| in [30, 50]. Naive FP16 x^3 overflows at |x| > ~40
    (40^3 = 64000 near FP16 max). The FP32-internal compute path must keep
    the cube finite.
    """
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(9)
    src = (30.0 + 20.0 * rng.uniform(size=(16, 16))).astype(np.float32)
    _write_fp16_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        GeluFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )
    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    assert np.isfinite(out).all(), "GELU FP16 must not overflow when |x| > 40"
    # For large positive x, gelu(x) ≈ x. Output magnitudes should be in [30, 50].
    assert (out > 0).all()


# ---------------------------------------------------------------------------
# SOFTMAX_FP32 (flags=1)
# ---------------------------------------------------------------------------


def test_softmax_fp32_flags1_rows_sum_to_one_within_fp16_ulp():
    """FP16 softmax: row sums should equal 1.0 within FP16 ULP (~1e-3)."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(10)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 4.0
    _write_fp16_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        SoftmaxFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    row_sums = out.sum(axis=-1)
    # FP16 ULP at value 1.0 is ~9.77e-4. Allow up to 16x that for the sum.
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=2e-2)


def test_softmax_fp32_flags1_large_inputs_safe():
    """Max-subtraction must keep softmax safe even for FP16 inputs with
    |x| approaching FP16 max — exp((x-max)) is in [0, 1] always.
    """
    sim = _make_sim_with_tile()
    src = np.zeros((16, 16), dtype=np.float32)
    src[:, 0] = 60000.0  # Near FP16 max; max-subtraction renormalizes.
    _write_fp16_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        SoftmaxFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )
    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    # First col dominates; should be ~1.0, others ~0.
    np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-3)
    np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# MASKED_SOFTMAX_FP32 (flags=1) — causal masking with CONFIG_ATTN
# ---------------------------------------------------------------------------


def test_masked_softmax_fp32_flags1_causal_masking():
    """Row i attends to keys 0..i only (causal). Beyond i: exact 0. Sum=1."""
    sim = _make_sim_with_tile()
    sim._execute(ConfigAttnInsn(query_row_base=0, valid_kv_len=16, mode=1))

    rng = np.random.default_rng(11)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 2.0
    _write_fp16_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        MaskedSoftmaxFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    for i in range(16):
        # Columns > i should be exact zero.
        assert (out[i, i + 1:] == 0.0).all(), f"row {i}: cols {i+1}: not zero"
        # Sum of valid columns should be ~1 within FP16 ULP.
        row_sum = float(out[i, :i + 1].sum())
        assert abs(row_sum - 1.0) < 2e-2, f"row {i}: sum={row_sum}, expected 1.0"


# ---------------------------------------------------------------------------
# MAX_ABS_REDUCE_FP32 (flags=1) — FP16 source widened to FP32 for reduction
# ---------------------------------------------------------------------------


def test_max_abs_reduce_fp32_flags1_over_fp16_source():
    """FP16 source widened to FP32; max(|x|) computed in FP32; scales FP16."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(12)
    src = rng.standard_normal((16, 16)).astype(np.float32) * 6.0
    _write_fp16_to_abuf(sim, SRC1_OFF, src)

    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=0,
            sreg=4,
            flags=1,
        )
    )

    # Source after FP16 round-trip:
    src_fp16 = src.astype(np.float16).astype(np.float32)
    max_abs = float(np.max(np.abs(src_fp16)))
    expected_inv = np.float16(127.0 / max(max_abs, 2.0 ** -9))
    expected_fwd = np.float16(max(max_abs, 2.0 ** -9) / 127.0)
    assert sim.state.scale_regs[4] == expected_inv
    assert sim.state.scale_regs[5] == expected_fwd


def test_max_abs_reduce_fp32_flags1_zero_tile_clamps():
    """All-zero FP16 tile must produce finite scales via the eps clamp."""
    sim = _make_sim_with_tile()
    _write_fp16_to_abuf(sim, SRC1_OFF, np.zeros((16, 16), dtype=np.float32))
    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC1_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=0,
            sreg=0,
            flags=1,
        )
    )
    inv = float(sim.state.scale_regs[0])
    fwd = float(sim.state.scale_regs[1])
    assert np.isfinite(inv) and np.isfinite(fwd)
    # 127 / (2**-9) = 65024, FP16-representable.
    assert abs(inv - 65024.0) < 100.0


# ---------------------------------------------------------------------------
# DEQUANT_ACCUM_FP32_SCALED — bias-fold contract (flags=1)
# ---------------------------------------------------------------------------


def test_dequant_accum_fp32_scaled_flags1_folds_bias_via_2n_src2():
    """flags=1: src2 = 2N FP16 (PC scales + bias). Output = (int32 × pc × act_scale + bias).astype(fp16)."""
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(13)
    accum = rng.integers(-1_000, 1_000, size=(16, 16), dtype=np.int32)
    wt_scales = rng.uniform(1e-3, 1e-2, size=16).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, size=16).astype(np.float32)
    act_scale = np.float32(0.0625)

    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    # Pack PC scales + bias as 2N FP16 at SCALE_OFF.
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, np.concatenate([wt_scales, bias]))
    sim.state.scale_regs[2] = np.float16(act_scale)

    sim._execute(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            sreg=2,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    pc_fp16 = wt_scales.astype(np.float16).astype(np.float32)
    bias_fp16 = bias.astype(np.float16).astype(np.float32)
    act_widened = np.float32(np.float16(act_scale))
    expected_fp32 = (
        accum.astype(np.float32) * pc_fp16.reshape(1, 16) * act_widened
        + bias_fp16.reshape(1, 16)
    ).astype(np.float32)
    expected_fp16 = expected_fp32.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected_fp16)


def test_dequant_accum_fp32_scaled_flags1_no_double_rounding():
    """The bias-fold contract: bias added in FP32 BEFORE the FP16 cast.

    Construct a case where the dequant intermediate has more mantissa
    precision than FP16 can hold, AND the bias would round differently
    if cast first. Verify the simulator produces (X + b).astype(fp16),
    NOT X.astype(fp16) + b.astype(fp16).
    """
    sim = _make_sim_with_tile()
    # Construct: dequant intermediate = X (FP32 value with FP16-ULP mantissa).
    # pc=1.0, act=1.0, so dequant = accum.astype(fp32). Pick accum values so
    # dequant value sits between two FP16-representable numbers.
    # FP16 ULP at 2048 is 2.0; so X=2049 rounds to 2048 in FP16. b=0.6 added
    # post-cast loses precision; added pre-cast gives 2049.6 → 2050 in FP16.
    accum = np.full((16, 16), 2049, dtype=np.int32)
    wt_scales = np.ones(16, dtype=np.float32)
    bias = np.full(16, 0.6, dtype=np.float32)

    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_vector(sim.state, BUF_WBUF, SCALE_OFF, np.concatenate([wt_scales, bias]))
    sim.state.scale_regs[3] = np.float16(1.0)

    sim._execute(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=SCALE_OFF,
            dst_buf=BUF_ABUF, dst_off=DST_OFF,
            sreg=3,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DST_OFF, 16, 16)
    # Correct (no double-round): (2049 + 0.6).astype(fp16) = (2049.6).astype(fp16) = 2050.
    correct = np.float16(2049.0 + np.float32(np.float16(0.6))).astype(np.float32)
    # WRONG (double-round): 2049.astype(fp16) + 0.6.astype(fp16) = 2048 + 0.6 ≈ 2048.6 → 2048.
    wrong = (np.float16(2049.0) + np.float16(np.float16(0.6))).astype(np.float16).astype(np.float32)
    assert (out == correct).all(), (
        f"bias-fold contract violated: out={out[0,0]}, correct={correct}, wrong={wrong}"
    )
    assert correct != wrong, "test fixture invalid — must distinguish bias-fold vs double-round"


# ---------------------------------------------------------------------------
# Composition: matmul prelude in FP16
#   MAX_ABS_REDUCE → QUANT → (eye-matmul stub) → DEQUANT_SCALED+bias
# ---------------------------------------------------------------------------


def test_matmul_prelude_composition_w8a16_round_trips_with_bias_fold():
    """End-to-end W8A16 matmul prelude flow against an identity weight:

        FP16 source → MAX_ABS_REDUCE → QUANT (S[s]=127/max)
                   → INT8
                   → INT8 matmul against eye(16) [stub: copy INT8 to ACCUM]
                   → DEQUANT_SCALED (PC=1.0 fp16, S[s+1]=max/127, +bias)
                   → FP16

    With identity weight and unit PC scales, the output should equal
    `(source + bias).astype(fp16)` within FP16 ULP — bias folded in FP32
    before the final cast.
    """
    sim = _make_sim_with_tile()
    rng = np.random.default_rng(16)
    # Source values constrained to FP16 dynamic range so the round-trip is meaningful.
    src = rng.standard_normal((16, 16)).astype(np.float32) * 4.0
    bias = rng.uniform(-0.3, 0.3, size=16).astype(np.float32)

    SRC_OFF = 0
    INT8_OFF = 64
    DEQUANT_OUT_OFF = 128
    PC_BIAS_OFF = 800  # WBUF

    _write_fp16_to_abuf(sim, SRC_OFF, src)
    # PC scales = 1.0 (identity weight) + bias values (2N FP16 layout).
    pc_scales = np.ones(16, dtype=np.float32)
    mem.write_fp16_vector(sim.state, BUF_WBUF, PC_BIAS_OFF, np.concatenate([pc_scales, bias]))

    # 1. MAX_ABS_REDUCE on FP16 source.
    sim._execute(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF, src1_off=SRC_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=0,
            sreg=6,
            flags=1,
        )
    )

    # 2. QUANT FP16 → INT8 using sreg=6 (127/max).
    sim._execute(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=SRC_OFF,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=INT8_OFF,
            sreg=6,
            flags=1,
        )
    )

    # 3. Eye-matmul stub: load INT8 directly into ACCUM as INT32.
    # (Skip the real systolic: with weights=identity, accum = int8 sign-extended.)
    int8_data = mem.read_int8_tile(sim.state, BUF_ABUF, INT8_OFF, 16, 16)
    mem.write_int32_tile(sim.state, BUF_ACCUM, 0, int8_data.astype(np.int32))

    # 4. DEQUANT_SCALED with bias fold (sreg=7 = max/127, src2 = 2N FP16).
    sim._execute(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=PC_BIAS_OFF,
            dst_buf=BUF_ABUF, dst_off=DEQUANT_OUT_OFF,
            sreg=7,
            flags=1,
        )
    )

    out = _read_fp16_from_abuf(sim, DEQUANT_OUT_OFF, 16, 16)
    # Reference: source widened FP16→FP32, then INT8 quant-dequant round-trip
    # (lossy but bounded), then bias added in FP32, then cast to FP16.
    src_fp16 = src.astype(np.float16).astype(np.float32)
    max_abs = float(np.max(np.abs(src_fp16)))
    max_abs = max(max_abs, 2.0 ** -9)
    inv_scale = np.float32(np.float16(127.0 / max_abs))
    fwd_scale = np.float32(np.float16(max_abs / 127.0))
    int8_round = np.clip(np.round(src_fp16 * inv_scale), -128, 127).astype(np.int32)
    dequant_fp32 = int8_round.astype(np.float32) * 1.0 * fwd_scale
    with_bias_fp32 = (dequant_fp32 + bias.astype(np.float16).astype(np.float32).reshape(1, 16)).astype(np.float32)
    expected_fp16 = with_bias_fp32.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected_fp16)


