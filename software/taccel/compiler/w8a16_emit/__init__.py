"""W8A16 codegen helpers — per-op lowering functions for the W8A16 path.

The main `CodeGenerator` (`codegen.py`) dispatches to these helpers at
each sub-layer emission site when `use_fp16_activations=True`. Each
helper writes instructions and ABUF/WBUF allocations on the codegen
instance; no return value.

Sub-modules:
  - `sublayer`   — LAYERNORM_FP32, GELU_FP32, SOFTMAX_FP32 /
                   MASKED_SOFTMAX_FP32, VADD_FP32
  - `matmul`     — matmul (simple + large-weight-tiled +
                   large-input-streaming)
  - `attention`  — matmul_qkt (Q @ K^T), matmul_attn_v (softmax · V)
  - `_common`    — shared free functions (FP16 cast, padding-row
                   zero-fill, ABUF alloc) + the `UNIT = 16` constant

Numerical contract (mirrored in `w8a16_simulator_reference.py`):
  - ABUF FP tiles are FP16 (2 bytes/element); compute upcasts to FP32.
  - WBUF holds FP16 gamma/beta and per-channel weight scales.
  - DEQUANT_ACCUM_FP32_SCALED reads `2N FP16` src2 (PC scales + bias)
    so bias folds into the dequant epilogue (no double-rounding).

Re-entrancy: helpers must not call back into the CodeGenerator dispatch
methods (`_emit_layernorm`, `_emit_gelu`, …). Those are the W8A16 entry
points; calling them from inside a helper infinitely recurses. Use
`cg._emit(...)`, `cg.mem.*`, and low-level helpers (`cg._emit_dma_load`,
`cg._record_trace_event`, etc.) instead.
"""
from .attention import emit_matmul_attn_v_w8a16, emit_matmul_qkt_w8a16
from .matmul import emit_matmul_w8a16, emit_matmul_w8a16_large_weight_tiled
from .sublayer import (
    emit_gelu_fp32,
    emit_layernorm_fp32,
    emit_softmax_fp32,
    emit_vadd_fp32,
)

__all__ = [
    "emit_gelu_fp32",
    "emit_layernorm_fp32",
    "emit_matmul_attn_v_w8a16",
    "emit_matmul_qkt_w8a16",
    "emit_matmul_w8a16",
    "emit_matmul_w8a16_large_weight_tiled",
    "emit_softmax_fp32",
    "emit_vadd_fp32",
]
