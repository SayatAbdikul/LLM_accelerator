"""SFU op emit helpers: scale_mul rename + layernorm/softmax/gelu/vadd.

Free-function migrations of the original `_emit_scale_mul`,
`_emit_softmax`, `_emit_gelu`, `_emit_gelu_from_dram_temp`,
`_emit_layernorm`, and `_emit_vadd` methods on `CodeGenerator`.

In W8A16 mode each entry point (other than `_emit_scale_mul`, which is
a metadata-only rename) early-dispatches to the matching helper in
`compiler/w8a16_emit/`. The legacy INT8 paths preserved below are kept
for source parity with the original method bodies; they are unreachable
under the current `use_fp16_activations = True` hardcode.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...isa.instructions import (
    ConfigTileInsn,
    DequantAddInsn,
    GeluInsn,
    LayernormInsn,
    SetScaleInsn,
    SyncInsn,
    VaddInsn,
)
from ...isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF
from ..ir import IRNode
from ..tiler import TILE, pad_dim
from ._common import UNIT, _fp16_to_uint16

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def emit_scale_mul(cg: "CodeGenerator", node: IRNode) -> None:
    """C1: scale_mul is metadata-only; scaling is folded into the QKT
    epilogue (INT8 path: via SOFTMAX's input scale; W8A32 path
    (M3-A): via DEQUANT_ACCUM_FP32's composite PC scale vector).
    Either way this node is a rename — propagate the allocation
    from `node.inputs[0]` to `node.name`."""
    # INT8 path: QKT writes its softmax output to WBUF (INT8).
    in_wbuf = cg.mem.wbuf.get(node.inputs[0])
    if in_wbuf is not None:
        # Rename in-place: pop the old key and re-insert under the new name
        # without touching the free list (free() would double-book the region).
        alloc = cg.mem.wbuf.allocations.pop(node.inputs[0])
        cg.mem.wbuf.allocations[node.name] = alloc
    else:
        # W8A32 path (M3-A): QKT writes its FP32 scores tile to ABUF.
        # Same rename mechanism, just in the ABUF allocator.
        in_abuf = cg.mem.abuf.get(node.inputs[0])
        if in_abuf is not None:
            alloc = cg.mem.abuf.allocations.pop(node.inputs[0])
            cg.mem.abuf.allocations[node.name] = alloc

    # Propagate scale metadata for downstream rename nodes.
    cg.calibration_scales[node.name] = cg.calibration_scales.get(
        node.inputs[0], 6.0 / 127.0
    )


def emit_softmax(cg: "CodeGenerator", node: IRNode) -> None:
    """Softmax (W8A16): dispatches to the FP32 lowering helper.

    Causal detection from node.attrs — MASKED_SOFTMAX_FP32 for causal,
    SOFTMAX_FP32 (0x1C, non-causal) for the unmasked case (the latter is
    currently illegal in RTL; non-causal models are out-of-target).
    """
    from ..w8a16_emit import emit_softmax_fp32
    masked = bool(node.attrs.get("causal", False)) or bool(node.attrs.get("masked", False))
    emit_softmax_fp32(cg, node, masked=masked)


def emit_gelu(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit GELU SFU instruction (W8A16: dispatches to FP32 lowering helper).

    W8A16 force-disables gelu_from_accum + strip-mined inlining, so the
    simple GELU path is what we want here.
    """
    from ..w8a16_emit import emit_gelu_fp32
    emit_gelu_fp32(cg, node)


def emit_gelu_from_dram_temp(cg: "CodeGenerator", node: IRNode) -> None:
    """Apply GELU strip-by-strip to a DRAM-temp-resident tensor."""
    input_name = node.inputs[0]
    input_dram = cg.dram_temp_outputs[input_name]
    M = int(node.output_shape[0])
    N = int(node.output_shape[1])
    M_pad = pad_dim(M)
    N_pad = pad_dim(N)
    strip_rows = TILE
    out_dram = cg.dram_temp_start + cg.mem.alloc_dram_temp(
        f"{node.name}_temp", M_pad * N_pad
    )
    in_scale = cg.calibration_scales.get(input_name, 1.0 / 127.0)
    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg.mem.abuf.free(input_name)

    for row_start in range(0, M_pad, strip_rows):
        logical_rows = max(0, min(strip_rows, M - row_start))
        strip_alloc = cg.mem.abuf.alloc(f"{node.name}_strip{row_start}", strip_rows * N_pad)
        cg._emit_dma_load(
            BUF_ABUF,
            strip_alloc.offset_units,
            strip_rows * N_pad,
            1,
            input_dram + row_start * N_pad,
        )
        cg._emit(SyncInsn(resource_mask=0b001))
        cg._emit(ConfigTileInsn(M=0, N=N_pad // TILE - 1, K=0))
        sreg = cg._alloc_sreg_pair()
        cg._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(in_scale)))
        cg._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(out_scale)))
        cg._emit(GeluInsn(
            src1_buf=BUF_ABUF,
            src1_off=strip_alloc.offset_units,
            dst_buf=BUF_ABUF,
            dst_off=strip_alloc.offset_units,
            sreg=sreg,
        ))
        cg._emit(SyncInsn(resource_mask=0b100))
        cg._record_trace_event(
            node.name,
            BUF_ABUF,
            strip_alloc.offset_units,
            strip_rows,
            N_pad,
            logical_rows,
            N,
            "int8",
            out_scale,
            row_start=row_start,
            full_rows=M,
            full_cols=N,
        )
        cg._emit_dma_store(
            BUF_ABUF,
            strip_alloc.offset_units,
            strip_rows * N_pad,
            2,
            out_dram + row_start * N_pad,
        )
        cg._emit(SyncInsn(resource_mask=0b001))
        cg.mem.abuf.free(strip_alloc.name)

    cg.dram_temp_outputs[node.name] = out_dram
    placeholder = cg.mem.abuf.alloc(node.name, strip_rows * N_pad)
    placeholder.size_bytes = M_pad * N_pad


def emit_layernorm(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit LAYERNORM SFU instruction (W8A16: dispatches to FP32 lowering helper)."""
    from ..w8a16_emit import emit_layernorm_fp32
    emit_layernorm_fp32(cg, node)


def emit_vadd(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit VADD for residual connection (W8A16: non-saturating VADD_FP32).

    dequant_add residual paths are force-disabled (gen-1-only); the W8A16
    path uses the FP32 lowering helper.
    """
    if node.name in cg.precomputed_nodes:
        return
    from ..w8a16_emit import emit_vadd_fp32
    emit_vadd_fp32(cg, node)
