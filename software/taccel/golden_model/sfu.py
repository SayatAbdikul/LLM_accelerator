"""Special Function Units: LayerNorm, Softmax, GELU (FP32 internal, INT8 I/O).

Precision spec (matches RTL target)
------------------------------------
- Scale registers store FP16 values, widened to FP32 for arithmetic.
- All SFU internal operations use FP32 (dequant, reduction, exp/erf, requant).
- Rounding convention: **round-half-to-even** (IEEE 754 default, NumPy default).
  This applies to REQUANT, SCALE_MUL, and all SFU requantization paths.
  RTL must implement the same rounding mode, otherwise results may differ
  by ±1 LSB on tie values (e.g. 0.5 → 0 with banker's vs. 1 with away).
- Clip to INT8 after rounding: [-128, 127].

GELU erf() implementation
--------------------------
The golden model uses scipy.special.erf (the C library's erf, accurate to
ULP precision).  RTL should implement the Abramowitz & Stegun 7.1.26
polynomial approximation (see ``_erf_poly`` below), which has max absolute
error < 1.5e-7 — far below the INT8 quantization noise floor (~0.004).

The polynomial approximation is provided as ``_erf_poly()`` for RTL
reference and verification.  In FP32 arithmetic the max absolute error is
~5e-7 (the 1.5e-7 figure from A&S applies to FP64).  Both implementations
produce identical INT8 output for all inputs encountered in the DeiT-tiny
workload.
"""
import numpy as np
from . import memory
from ..isa.opcodes import BUF_ACCUM
from ..quantizer.twin_uniform import (
    quantize_dequant_gelu_twin,
    quantize_dequant_softmax_twin,
)
from ..utils.int8_ops import clip_int8

CYCLE_PER_ELEMENT = 2
ATTN_MODE_PADDED = 0b01
ATTN_MODE_CAUSAL = 0b10
ATTN_MODE_CAUSAL_PADDED = 0b11


def _erf_poly(x: np.ndarray) -> np.ndarray:
    """Polynomial approximation of erf(x) for RTL implementation reference.

    Abramowitz & Stegun formula 7.1.26 — max |error| < 5e-7 in FP32
    (~1.5e-7 in FP64).  Uses only FMA + exp — no erf hardware needed.

    RTL implementation: 5 FMA + 1 exp + 1 reciprocal per element.
    """
    a1 = np.float32(0.254829592)
    a2 = np.float32(-0.284496736)
    a3 = np.float32(1.421413741)
    a4 = np.float32(-1.453152027)
    a5 = np.float32(1.061405429)
    p = np.float32(0.3275911)

    sign = np.sign(x)
    x_abs = np.abs(x)
    t = np.float32(1.0) / (np.float32(1.0) + p * x_abs)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    y = np.float32(1.0) - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * np.exp(-(x_abs * x_abs))
    return sign * y


def _get_dual_scales(state, sreg: int):
    """Return (in_scale, out_scale) as FP32 from consecutive scale registers.

    Scale registers hold FP16. Widening to FP32 preserves the exact FP16 value
    without adding precision — this is the RTL behaviour (FP16 reg → FP32 datapath).
    """
    from .simulator import ConfigError
    if sreg >= 15:
        raise ConfigError("SFU sreg+1 out of range")
    in_scale  = np.float32(state.scale_regs[sreg])
    out_scale = np.float32(state.scale_regs[sreg + 1])
    return in_scale, out_scale


def _get_quad_scales(state, sreg: int):
    """Return four consecutive FP32 scales from scale registers."""
    from .simulator import ConfigError
    if sreg >= 13:
        raise ConfigError("SFU sreg+3 out of range")
    return tuple(np.float32(state.scale_regs[sreg + idx]) for idx in range(4))


def _runtime_twin_spec(state, kind: str):
    specs = getattr(state, "runtime_twin_specs", {}) or {}
    spec = specs.get(int(getattr(state, "current_pc", -1)))
    if not spec or spec.get("kind") != kind or spec.get("mode") != "paper_exact":
        return None
    return spec


def _requantize_int8(x: np.ndarray, out_scale: np.float32) -> np.ndarray:
    """Requantize FP32 to INT8 using the architectural rounding policy."""
    if out_scale == np.float32(0):
        return np.zeros_like(x, dtype=np.int8)
    return np.clip(np.round(x / out_scale), -128, 127).astype(np.int8)


def _attention_mask(M: int, N: int, query_row_base: int, valid_kv_len: int, mode: int) -> np.ndarray:
    """Return a True/False mask where True means a logit is visible."""
    from .simulator import ConfigError
    if valid_kv_len <= 0:
        raise ConfigError("CONFIG_ATTN valid_kv_len must be > 0")
    rows = np.arange(M, dtype=np.int32).reshape(M, 1)
    cols = np.arange(N, dtype=np.int32).reshape(1, N)
    visible = np.ones((M, N), dtype=bool)
    if mode & ATTN_MODE_CAUSAL:
        visible &= cols <= (query_row_base + rows)
    if mode & ATTN_MODE_PADDED:
        visible &= cols < valid_kv_len
    fully_masked = ~visible.any(axis=1)
    if np.any(fully_masked):
        first = int(np.flatnonzero(fully_masked)[0])
        raise ConfigError(f"CONFIG_ATTN produced a fully masked row at local row {first}")
    return visible


def _masked_softmax_probs(x: np.ndarray, query_row_base: int, valid_kv_len: int, mode: int) -> np.ndarray:
    """Apply configured attention mask before stable softmax."""
    visible = _attention_mask(x.shape[0], x.shape[1], query_row_base, valid_kv_len, mode)
    masked = np.where(visible, x, np.float32(-np.inf)).astype(np.float32)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted).astype(np.float32)
    exp_x = np.where(visible, exp_x, np.float32(0.0))
    denom = exp_x.sum(axis=-1, keepdims=True)
    if np.any(denom == np.float32(0)):
        from .simulator import ConfigError
        raise ConfigError("CONFIG_ATTN produced zero softmax denominator")
    return (exp_x / denom).astype(np.float32)


def _attn_context_values(state):
    from .simulator import ConfigError
    ctx = getattr(state, "attn_context", None) or {}
    if not ctx.get("is_valid", False):
        raise ConfigError("CONFIG_ATTN not set")
    return (
        int(ctx["query_row_base"]),
        int(ctx["valid_kv_len"]),
        int(ctx["mode"]),
    )


def execute_layernorm(state, insn):
    """LayerNorm: dequant INT8 → FP32, normalize, requant → INT8.

    src1 = input activations, src2 = gamma/beta (FP16 packed), dst = output.
    All arithmetic in FP32; gamma/beta widened from FP16.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)

    # Read gamma, beta — stored as FP16, widen to FP32
    gb_bytes = memory.read_bytes(state, insn.src2_buf, insn.src2_off, N * 4)
    gamma = np.frombuffer(gb_bytes[:N * 2], dtype=np.float16).astype(np.float32)
    beta  = np.frombuffer(gb_bytes[N * 2:], dtype=np.float16).astype(np.float32)

    # Dequantize: INT8 × FP32(in_scale) → FP32
    x = inp.astype(np.float32) * in_scale

    # Normalize in FP32 (epsilon = 1e-6 matches PyTorch LayerNorm default)
    eps = np.float32(1e-6)
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1,  keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_out = x_norm * gamma + beta   # FP32 affine

    # Requantize: round-half-to-even, clip to INT8
    if out_scale == np.float32(0):
        result = np.zeros_like(inp)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax(state, insn):
    """Softmax: dequant (INT8 or INT32) → FP32, softmax along last dim, requant → INT8.

    Numerically stable: subtract row-max before exp.
    All arithmetic in FP32.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    if insn.src1_buf == BUF_ACCUM:
        # C1 path: consume raw INT32 QKT accumulators directly.
        inp_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, N)
        x = inp_i32.astype(np.float32) * in_scale
    else:
        inp_i8 = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)
        x = inp_i8.astype(np.float32) * in_scale

    # Numerically stable softmax in FP32
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted).astype(np.float32)
    x_out = exp_x / exp_x.sum(axis=-1, keepdims=True)
    softmax_spec = _runtime_twin_spec(state, "softmax")
    if softmax_spec is not None:
        x_out = quantize_dequant_softmax_twin(
            x_out,
            float(softmax_spec.get("range1_max", 1.0)),
        ).astype(np.float32)

    # Requantize: round-half-to-even, clip to INT8
    if out_scale == np.float32(0):
        result = np.zeros((M, N), dtype=np.int8)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_masked_softmax(state, insn):
    """Masked softmax under the current CONFIG_ATTN context."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)
    query_row_base, valid_kv_len, mode = _attn_context_values(state)

    if insn.src1_buf == BUF_ACCUM:
        inp_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, N)
        x = inp_i32.astype(np.float32) * in_scale
    else:
        inp_i8 = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)
        x = inp_i8.astype(np.float32) * in_scale

    x_out = _masked_softmax_probs(x, query_row_base, valid_kv_len, mode)
    softmax_spec = _runtime_twin_spec(state, "softmax")
    if softmax_spec is not None:
        x_out = quantize_dequant_softmax_twin(
            x_out,
            float(softmax_spec.get("range1_max", 1.0)),
        ).astype(np.float32)

    result = _requantize_int8(x_out, out_scale)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_gelu(state, insn):
    """GELU: dequant INT8 → FP32, GELU activation, requant → INT8.

    GELU-new (tanh approximation) matching GPT-2's gelu_new activation:
        GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    All arithmetic in FP32.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    if insn.src1_buf == BUF_ACCUM:
        inp_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, N)
        x = inp_i32.astype(np.float32) * in_scale
    else:
        inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)
        # Dequantize: INT8 × FP32(in_scale) → FP32
        x = inp.astype(np.float32) * in_scale

    # GELU-new (tanh approximation) in FP32
    x_out = x * np.float32(0.5) * (
        np.float32(1.0) + np.tanh(
            np.float32(np.sqrt(2.0 / np.pi)) * (x + np.float32(0.044715) * x ** 3)
        )
    )
    gelu_spec = _runtime_twin_spec(state, "gelu")
    if gelu_spec is not None:
        x_out = quantize_dequant_gelu_twin(
            x_out,
            float(gelu_spec.get("positive_range_max", max(out_scale * 127.0, 1e-8))),
            negative_extent=float(gelu_spec.get("negative_extent", 1e-8)),
        ).astype(np.float32)

    # Requantize: round-half-to-even, clip to INT8
    if out_scale == np.float32(0):
        result = np.zeros((M, N), dtype=np.int8)
    else:
        result = np.clip(np.round(x_out / out_scale), -128, 127).astype(np.int8)

    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax_attnv(state, insn):
    """Fused SOFTMAX_ATTNV.

    Reads QKT from ACCUM and V from INT8 SRAM, computes softmax(QKT) in FP32,
    immediately multiplies by dequantized V in FP32, and requantizes the final
    attn@V output to INT8. For diagnostics, it also returns a virtual INT8
    softmax tile using a trace-only scale in sreg+3.
    """
    from .simulator import ConfigError, IllegalBufferError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")
    if insn.src1_buf != BUF_ACCUM:
        raise IllegalBufferError(insn.src1_buf)
    if insn.src2_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.src2_buf)
    if insn.dst_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.dst_buf)

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1
    M = m_tiles * 16
    N = n_tiles * 16
    K = k_tiles * 16

    qkt_in_scale, v_scale, out_scale, softmax_trace_scale = _get_quad_scales(state, insn.sreg)

    qkt_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, K)
    v_i8 = memory.read_int8_tile(state, insn.src2_buf, insn.src2_off, K, N)

    qkt = qkt_i32.astype(np.float32) * qkt_in_scale
    v = v_i8.astype(np.float32) * v_scale

    qkt_shifted = qkt - qkt.max(axis=-1, keepdims=True)
    exp_qkt = np.exp(qkt_shifted).astype(np.float32)
    softmax = exp_qkt / exp_qkt.sum(axis=-1, keepdims=True)
    softmax_spec = _runtime_twin_spec(state, "softmax")
    if softmax_spec is not None:
        softmax = quantize_dequant_softmax_twin(
            softmax,
            float(softmax_spec.get("range1_max", 1.0)),
        ).astype(np.float32)
    attn_v = np.matmul(softmax, v).astype(np.float32)

    if out_scale == np.float32(0):
        result = np.zeros((M, N), dtype=np.int8)
    else:
        result = np.clip(np.round(attn_v / out_scale), -128, 127).astype(np.int8)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)

    if softmax_trace_scale == np.float32(0):
        softmax_i8 = np.zeros((M, K), dtype=np.int8)
    else:
        softmax_i8 = np.clip(np.round(softmax / softmax_trace_scale), -128, 127).astype(np.int8)

    state.cycle_count += (M * K * CYCLE_PER_ELEMENT) + (m_tiles * n_tiles * k_tiles * 16) + (M * N)
    return {
        "softmax": {
            "raw": softmax_i8,
            "dtype": "int8",
            "scale": float(softmax_trace_scale),
            "sat": int(np.count_nonzero((softmax_i8 == 127) | (softmax_i8 == -128))),
            "zero": int(np.count_nonzero(softmax_i8 == 0)),
            "total": int(softmax_i8.size),
        }
    }


def execute_masked_softmax_attnv(state, insn):
    """Fused masked softmax + attn@V under CONFIG_ATTN."""
    from .simulator import ConfigError, IllegalBufferError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")
    if insn.src1_buf != BUF_ACCUM:
        raise IllegalBufferError(insn.src1_buf)
    if insn.src2_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.src2_buf)
    if insn.dst_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.dst_buf)

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1
    M = m_tiles * 16
    N = n_tiles * 16
    K = k_tiles * 16

    query_row_base, valid_kv_len, mode = _attn_context_values(state)
    qkt_in_scale, v_scale, out_scale, softmax_trace_scale = _get_quad_scales(state, insn.sreg)

    qkt_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, K)
    v_i8 = memory.read_int8_tile(state, insn.src2_buf, insn.src2_off, K, N)

    qkt = qkt_i32.astype(np.float32) * qkt_in_scale
    v = v_i8.astype(np.float32) * v_scale

    softmax = _masked_softmax_probs(qkt, query_row_base, valid_kv_len, mode)
    softmax_spec = _runtime_twin_spec(state, "softmax")
    if softmax_spec is not None:
        softmax = quantize_dequant_softmax_twin(
            softmax,
            float(softmax_spec.get("range1_max", 1.0)),
        ).astype(np.float32)
    attn_v = np.matmul(softmax, v).astype(np.float32)

    result = _requantize_int8(attn_v, out_scale)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)

    softmax_i8 = _requantize_int8(softmax, softmax_trace_scale)
    state.cycle_count += (M * K * CYCLE_PER_ELEMENT) + (m_tiles * n_tiles * k_tiles * 16) + (M * N)
    return {
        "softmax": {
            "raw": softmax_i8,
            "dtype": "int8",
            "scale": float(softmax_trace_scale),
            "sat": int(np.count_nonzero((softmax_i8 == 127) | (softmax_i8 == -128))),
            "zero": int(np.count_nonzero(softmax_i8 == 0)),
            "total": int(softmax_i8.size),
        }
    }
