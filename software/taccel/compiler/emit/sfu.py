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
    """C1: softmax already emitted per-strip in _emit_qkt; this node is a rename."""
    if cg.use_fp16_activations:
        # Phase 3 (c.1) M2: softmax in W8A32 emits SOFTMAX_FP32 (or
        # MASKED_SOFTMAX_FP32 for causal). Causal detection comes
        # from the node's attrs — same convention as the INT8 path.
        from ..w8a16_emit import emit_softmax_fp32
        masked = bool(node.attrs.get("causal", False)) or bool(node.attrs.get("masked", False))
        emit_softmax_fp32(cg, node, masked=masked)
        return
    in_alloc = cg.mem.wbuf.get(node.inputs[0])
    if in_alloc is not None and node.inputs[0] in cg.mem.wbuf.allocations:
        alloc = cg.mem.wbuf.allocations.pop(node.inputs[0])
        cg.mem.wbuf.allocations[node.name] = alloc
    cg.calibration_scales[node.name] = cg.calibration_scales.get(
        node.inputs[0], 1.0 / 127.0
    )


def emit_gelu(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit GELU SFU instruction (no-op if inlined with a strip-mined matmul)."""
    if cg.use_fp16_activations:
        # Phase 3 (c.1) M2: GELU dispatches to the FP32 lowering helper.
        # W8A32 mode force-disables gelu_from_accum + strip-mined inlining,
        # so the simple GELU path is what we want here.
        from ..w8a16_emit import emit_gelu_fp32
        emit_gelu_fp32(cg, node)
        return
    if node.attrs.get("inline_with"):
        # GELU was applied inline in the strip-mined FC1 loop.
        # Propagate DRAM temp tracking and rename the ABUF placeholder.
        fc1_name = node.inputs[0]
        if fc1_name in cg.dram_temp_outputs:
            cg.dram_temp_outputs[node.name] = cg.dram_temp_outputs[fc1_name]
        # Rename fc1's allocation to the gelu node name (transfer ownership).
        # Do NOT create a second Allocation pointing at the same bytes — that
        # would cause a double-free when the generate loop frees both fc1 and
        # gelu at their respective last-use indices.
        fc1_alloc = cg.mem.abuf.allocations.pop(fc1_name, None)
        if fc1_alloc is not None:
            fc1_alloc.name = node.name
            cg.mem.abuf.allocations[node.name] = fc1_alloc
        return
    if node.inputs[0] in cg.dram_temp_outputs:
        emit_gelu_from_dram_temp(cg, node)
        return
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    sreg = cg._alloc_sreg_pair()
    in_scale = cg.calibration_scales.get(node.inputs[0], 1.0 / 127.0)
    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(in_scale)))
    cg._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(out_scale)))

    in_alloc = cg.mem.abuf.get(node.inputs[0]) or \
               cg.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)
    out_alloc = cg.mem.abuf.alloc(node.name, M_pad * N_pad)
    cg._emit(GeluInsn(
        src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
        dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
        sreg=sreg,
    ))
    cg._emit(SyncInsn(resource_mask=0b100))
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "int8",
        out_scale,
    )


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
    """Emit LAYERNORM SFU instruction."""
    if cg.use_fp16_activations:
        # Phase 3 (c.1) M2: LN dispatches to the FP32 lowering helper.
        from ..w8a16_emit import emit_layernorm_fp32
        emit_layernorm_fp32(cg, node)
        return
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    sreg = cg._alloc_sreg_pair()
    in_scale = cg.calibration_scales.get(node.inputs[0], 1.0 / 127.0)
    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(in_scale)))
    cg._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(out_scale)))

    # Load gamma/beta to WBUF
    gamma_name = node.inputs[1]
    beta_name = node.inputs[2]
    gamma_data = cg.weight_data.get(gamma_name)
    beta_data = cg.weight_data.get(beta_name)

    if gamma_data is not None and beta_data is not None:
        gamma_dram = cg._dram_offset_required(gamma_name, f"loading layernorm gamma for '{node.name}'")
        beta_dram = cg._dram_offset_required(beta_name, f"loading layernorm beta for '{node.name}'")
        # Pack gamma then beta in WBUF
        gb_bytes = N_pad * 4  # gamma[N] FP16 + beta[N] FP16 = N*4 bytes
        gb_alloc = cg.mem.wbuf.alloc(f"gb_{node.name}", gb_bytes)
        cg._emit_dma_load(BUF_WBUF, gb_alloc.offset_units, N_pad * 2, 1, gamma_dram)
        cg._emit(SyncInsn(resource_mask=0b001))
        # Load beta right after gamma
        beta_off = gb_alloc.offset_units + (N_pad * 2) // UNIT
        cg._emit_dma_load(BUF_WBUF, beta_off, N_pad * 2, 1, beta_dram)
        cg._emit(SyncInsn(resource_mask=0b001))

    in_alloc = cg.mem.abuf.get(node.inputs[0]) or \
               cg.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)
    gb_alloc = cg.mem.wbuf.get(f"gb_{node.name}")
    gb_off = gb_alloc.offset_units if gb_alloc else 0
    trace_ln1_padding = cg._should_trace_ln1_padding_debug(node.name)

    if trace_ln1_padding:
        cg._record_trace_event(
            f"{node.name}__input_padded",
            BUF_ABUF,
            in_alloc.offset_units,
            M_pad,
            N_pad,
            M_pad,
            N_pad,
            "int8",
            in_scale,
        )

    out_alloc = cg.mem.abuf.alloc(node.name, M_pad * N_pad)
    cg._emit(LayernormInsn(
        src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
        src2_buf=BUF_WBUF, src2_off=gb_off,
        dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
        sreg=sreg,
    ))
    cg._emit(SyncInsn(resource_mask=0b100))
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "int8",
        out_scale,
    )
    if trace_ln1_padding:
        cg._record_trace_event(
            f"{node.name}__output_padded",
            BUF_ABUF,
            out_alloc.offset_units,
            M_pad,
            N_pad,
            M_pad,
            N_pad,
            "int8",
            out_scale,
        )

    if gb_alloc:
        cg.mem.wbuf.free(f"gb_{node.name}")


def emit_vadd(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit VADD for residual connection (INT8 saturating add).

    Handles the case where one input is DRAM-resident (strip-mined output)
    by loading it into ABUF first.
    """
    if cg.use_fp16_activations:
        # Phase 3 (c.1) M2: residual stream VADD in W8A32 emits the
        # non-saturating VADD_FP32. dequant_add residual paths are
        # force-disabled in the W8A32 constructor, so we won't hit
        # the DequantAddInsn branch below in this mode.
        if node.name in cg.precomputed_nodes:
            return
        from ..w8a16_emit import emit_vadd_fp32
        emit_vadd_fp32(cg, node)
        return
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE

    if node.name in cg.precomputed_nodes:
        return

    pending_name = None
    skip_name = None
    if cg._dequant_add_enabled_for_residual(node.name):
        for input_name in node.inputs:
            if input_name in cg.pending_accum_outputs:
                pending_name = input_name
                break
        if pending_name is not None:
            skip_candidates = [name for name in node.inputs if name != pending_name]
            if len(skip_candidates) != 1:
                raise ValueError(f"{node.name} expected one skip input, got {node.inputs}")
            skip_name = skip_candidates[0]

    if pending_name is not None and skip_name is not None:
        if skip_name in cg.dram_temp_outputs:
            skip_alloc = cg._load_dram_to_abuf(skip_name, M_pad, N_pad)
        else:
            skip_alloc = cg.mem.abuf.get(skip_name) or \
                        cg.mem.abuf.alloc(skip_name, M_pad * N_pad)

        pending = cg.pending_accum_outputs.pop(pending_name)
        output_scale = cg.calibration_scales.get(node.name, 6.0 / 127.0)
        skip_scale = cg.calibration_scales.get(skip_name, 6.0 / 127.0)
        accum_rescale = pending["accum_real_scale"] / max(output_scale, 1e-12)
        skip_rescale = skip_scale / max(output_scale, 1e-12)
        sreg = cg._alloc_sreg_pair()
        cg._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(accum_rescale)))
        cg._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(skip_rescale)))
        cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))
        cg._emit(DequantAddInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=skip_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=skip_alloc.offset_units,
            sreg=sreg,
        ))
        cg._record_trace_event(
            node.name,
            BUF_ABUF,
            skip_alloc.offset_units,
            M_pad,
            N_pad,
            node.output_shape[0],
            node.output_shape[1],
            "int8",
            output_scale,
        )
        alloc = cg.mem.abuf.allocations.pop(skip_name, None)
        if alloc is not None:
            alloc.name = node.name
            cg.mem.abuf.allocations[node.name] = alloc
        return

    # Resolve src1 — load from DRAM if needed
    if node.inputs[0] in cg.dram_temp_outputs:
        src1_alloc = cg._load_dram_to_abuf(node.inputs[0], M_pad, N_pad)
        free_src1 = True
    else:
        src1_alloc = cg.mem.abuf.get(node.inputs[0]) or \
                     cg.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)
        free_src1 = False

    # Resolve src2 — load from DRAM if needed
    if node.inputs[1] in cg.dram_temp_outputs:
        src2_alloc = cg._load_dram_to_abuf(node.inputs[1], M_pad, N_pad)
        free_src2 = True
    else:
        src2_alloc = cg.mem.abuf.get(node.inputs[1]) or \
                     cg.mem.abuf.alloc(node.inputs[1], M_pad * N_pad)
        free_src2 = False

    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    # Write result in-place into src2's slot to avoid a third ABUF allocation.
    # (src2 is residual1 whose last use is this VADD, so overwriting is safe.)
    cg._emit(VaddInsn(
        src1_buf=BUF_ABUF, src1_off=src1_alloc.offset_units,
        src2_buf=BUF_ABUF, src2_off=src2_alloc.offset_units,
        dst_buf=BUF_ABUF, dst_off=src2_alloc.offset_units,
    ))
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        src2_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "int8",
        cg.calibration_scales.get(node.name, 6.0 / 127.0),
    )

    # Free temporary ABUF slot used for the DRAM-loaded src1
    if free_src1:
        cg.mem.abuf.free(node.inputs[0])

    # Rename src2's allocation to the output node name
    alloc = cg.mem.abuf.allocations.pop(node.inputs[1], None)
    if alloc is not None:
        alloc.name = node.name
        cg.mem.abuf.allocations[node.name] = alloc
