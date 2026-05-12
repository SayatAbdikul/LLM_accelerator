"""W8A32 codegen helpers (Phase 3 (c.1), milestone M2).

This module contains the per-op lowering functions for the W8A32 path.
The main `CodeGenerator` (`codegen.py`) dispatches to these helpers at
each sub-layer emission site when its `w8a32_enabled` flag is True.

Scope (M2)
----------

  - LAYERNORM_FP32:  drop-in replacement for the INT8 LAYERNORM emission
  - GELU_FP32:       drop-in replacement for the INT8 GELU emission
  - SOFTMAX_FP32 / MASKED_SOFTMAX_FP32: drop-in for INT8 softmax variants
  - VADD_FP32:       drop-in replacement for the INT8 VADD residual emission

Out of scope (M2.5)
-------------------

  - Matmul-output dequant: replacing REQUANT/REQUANT_PC with
    DEQUANT_ACCUM_FP32 inside `_emit_matmul_simple` /
    `_emit_matmul_strip_mined`. Those methods are 200+ LOC each and
    have many cross-cutting concerns (gelu-from-accum, fused
    softmax-attnv, dequant_add residual) — splitting that work into
    its own milestone lets M2 ship a clean, testable surface.
  - Pre-matmul QUANT_FP32_INT8 prelude.

Design contract from M1 (do not break)
--------------------------------------

  - FP32 ABUF tiles use 4 bytes per element. A 16x16 FP32 tile occupies
    1024 bytes = 64 16-byte addressing units (vs 16 units for INT8).
    Callers must size their `mem.abuf.alloc(..., size_bytes=M_pad * N_pad * 4)`
    accordingly — the size is in bytes, not elements.
  - FP16 stays the WBUF convention for LN gamma+beta vectors. Codegen
    emits them as `N * 2` bytes (gamma) immediately followed by `N * 2`
    bytes (beta), matching `_exec_layernorm_fp32` which calls
    `mem.read_fp16_vector(src2_buf, src2_off, 2 * N)`.
  - GELU is `gelu_new` (tanh approximation) by hardware contract.

Re-entrancy contract
--------------------

These helpers must **not** call back into the patched dispatch methods
on `CodeGenerator` (`_emit_layernorm`, `_emit_gelu`, `_emit_softmax`,
`_emit_vadd`). Those methods are the W8A32 entry points; calling them
from inside a helper would infinitely recurse into the W8A32 branch.
Use `cg._emit(...)`, `cg.mem.*`, and the low-level helpers
(`cg._emit_dma_load`, `cg._record_trace_event`, etc.) instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..isa.instructions import (
    ConfigTileInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    SoftmaxFp32Insn,
    SyncInsn,
    VaddFp32Insn,
)
from ..isa.opcodes import BUF_ABUF, BUF_WBUF
from .tiler import TILE, pad_dim

if TYPE_CHECKING:
    from .codegen import CodeGenerator
    from .ir import IRNode


# Each FP32 element is 4 bytes; the codegen allocates in bytes through
# `mem.abuf.alloc(name, size_bytes)`, so we multiply M*N by this.
FP32_BYTES_PER_ELEM = 4


def _abuf_alloc_fp32(cg: "CodeGenerator", name: str, M_pad: int, N_pad: int):
    """Allocate an ABUF region sized for an [M_pad, N_pad] FP32 tile.

    Returns the existing allocation if one is already keyed by `name`
    (i.e. an inplace re-use), otherwise allocates fresh. The 4× sizing
    accounts for FP32 being 4 bytes/element vs INT8's 1 byte/element.
    """
    existing = cg.mem.abuf.get(name)
    if existing is not None:
        return existing
    return cg.mem.abuf.alloc(name, M_pad * N_pad * FP32_BYTES_PER_ELEM)


def emit_layernorm_fp32(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 LAYERNORM_FP32 lowering.

    Reads FP32 input from ABUF, FP16 gamma+beta from WBUF, writes FP32
    output to ABUF. No per-tensor activation scales involved — LN does
    not consume `in_scale` or `out_scale` SET_SCALE instructions for
    the W8A32 path (its dynamic range is preserved end-to-end in FP32).
    """
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    # Load gamma/beta as FP16 into WBUF (2 bytes/elem). The simulator
    # handler reads `2 * N_pad` FP16 values from src2_off (gamma then beta).
    gamma_name = node.inputs[1]
    beta_name = node.inputs[2]
    gamma_data = cg.weight_data.get(gamma_name)
    beta_data = cg.weight_data.get(beta_name)
    gb_alloc = None
    if gamma_data is not None and beta_data is not None:
        from ..isa.opcodes import BUF_WBUF as _BUF_WBUF
        gamma_dram = cg._dram_offset_required(
            gamma_name, f"loading W8A32 layernorm gamma for '{node.name}'"
        )
        beta_dram = cg._dram_offset_required(
            beta_name, f"loading W8A32 layernorm beta for '{node.name}'"
        )
        gb_bytes = N_pad * 4  # gamma N FP16 + beta N FP16
        gb_alloc = cg.mem.wbuf.alloc(f"gb_{node.name}", gb_bytes)
        cg._emit_dma_load(BUF_WBUF, gb_alloc.offset_units, N_pad * 2, 1, gamma_dram)
        cg._emit(SyncInsn(resource_mask=0b001))
        beta_off_units = gb_alloc.offset_units + (N_pad * 2) // 16
        cg._emit_dma_load(BUF_WBUF, beta_off_units, N_pad * 2, 1, beta_dram)
        cg._emit(SyncInsn(resource_mask=0b001))

    in_alloc = _abuf_alloc_fp32(cg, node.inputs[0], M_pad, N_pad)
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    cg._emit(
        LayernormFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=in_alloc.offset_units,
            src2_buf=BUF_WBUF,
            src2_off=gb_alloc.offset_units if gb_alloc else 0,
            dst_buf=BUF_ABUF,
            dst_off=out_alloc.offset_units,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "fp32",
        out_scale,
    )

    if gb_alloc is not None:
        cg.mem.wbuf.free(f"gb_{node.name}")


def emit_gelu_fp32(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 GELU_FP32 lowering (tanh approximation, matching gelu_new)."""
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    in_alloc = _abuf_alloc_fp32(cg, node.inputs[0], M_pad, N_pad)
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    cg._emit(
        GeluFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=in_alloc.offset_units,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=out_alloc.offset_units,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "fp32",
        out_scale,
    )


def emit_softmax_fp32(cg: "CodeGenerator", node: "IRNode", *, masked: bool = False) -> None:
    """W8A32 SOFTMAX_FP32 / MASKED_SOFTMAX_FP32 lowering.

    For non-masked softmax the W8A32 path is straightforward. For the
    causal-masked variant the CONFIG_ATTN context must already be set
    by the caller (typically through the same path that emits
    MASKED_SOFTMAX in the INT8 codegen).
    """
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    in_alloc = _abuf_alloc_fp32(cg, node.inputs[0], M_pad, N_pad)
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    cls = MaskedSoftmaxFp32Insn if masked else SoftmaxFp32Insn
    cg._emit(
        cls(
            src1_buf=BUF_ABUF,
            src1_off=in_alloc.offset_units,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=out_alloc.offset_units,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "fp32",
        out_scale,
    )


def emit_vadd_fp32(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 VADD_FP32 lowering for the residual stream.

    Two FP32 sources read from ABUF, FP32 destination written to ABUF.
    Unlike the INT8 VADD, FP32 add does not saturate and does not
    consume per-tensor activation scales.
    """
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    src1_alloc = _abuf_alloc_fp32(cg, node.inputs[0], M_pad, N_pad)
    src2_alloc = _abuf_alloc_fp32(cg, node.inputs[1], M_pad, N_pad)
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    cg._emit(
        VaddFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=src1_alloc.offset_units,
            src2_buf=BUF_ABUF,
            src2_off=src2_alloc.offset_units,
            dst_buf=BUF_ABUF,
            dst_off=out_alloc.offset_units,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "fp32",
        out_scale,
    )
