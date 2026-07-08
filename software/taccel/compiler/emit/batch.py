"""Phase 2 batched-decode row-movement helpers.

Two BUF_COPY-backed ops route rows between ABUF tiles for lockstep batched
decode's per-stream attention:

- ``row_copy``   : copy ``num_rows`` rows out of one source tile (starting at
                   ``src_row``) into a fresh destination tile at ``dst_row``.
                   Used to extract stream ``s``'s single query row out of the
                   batched (16, d_head) Q projection before its per-stream QK^T.
- ``gather_rows``: copy row ``src_row`` (default 0) of each of N source tiles
                   into rows 0..N-1 of one destination tile. Used to collect
                   the 16 per-stream attention outputs back into one (16,
                   d_head) tile for concat_heads.

Destination tiles are padded to ``pad_dim`` rows so downstream matmul/QK^T
consumers (which read ``pad_dim(query_len)`` rows) stay in bounds; only the
copied rows carry meaningful data, exactly as the single-token decode path
tolerates non-zero query pad rows (query_len=1 keeps output row 0 only).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...isa.instructions import BufCopyInsn, SyncInsn
from ...isa.opcodes import BUF_ABUF
from ..ir import IRNode
from ..tiler import pad_dim
from ._common import UNIT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def _row_units(cg: "CodeGenerator", cols: int) -> int:
    return (pad_dim(int(cols)) * cg.elem_bytes) // UNIT


def emit_row_copy(cg: "CodeGenerator", node: IRNode) -> None:
    if not node.inputs:
        raise ValueError(f"{node.name} requires one source tensor input")
    if len(node.output_shape) != 2:
        raise ValueError(f"{node.name} requires a 2D output shape, got {node.output_shape}")
    src_alloc = cg.mem.abuf.get(node.inputs[0])
    if src_alloc is None:
        raise KeyError(f"Missing ABUF allocation '{node.inputs[0]}' for {node.name}")

    num_rows = int(node.output_shape[0])
    cols = int(node.output_shape[1])
    src_row = int(node.attrs.get("src_row", 0))
    dst_row = int(node.attrs.get("dst_row", 0))
    row_units = _row_units(cg, cols)

    # Pad the destination to pad_dim(num_rows) rows so a downstream QK^T /
    # matmul consumer reading pad_dim(query_len) rows stays in bounds.
    dst_rows_pad = pad_dim(num_rows)
    out_alloc = cg.mem.abuf.alloc(node.name, dst_rows_pad * pad_dim(cols) * cg.elem_bytes)

    cg._emit(BufCopyInsn(
        src_buf=BUF_ABUF, src_off=src_alloc.offset_units + src_row * row_units,
        dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units + dst_row * row_units,
        length=num_rows * row_units,
    ))
    cg._emit(SyncInsn(resource_mask=0b001))
    cg._record_trace_event(
        node.name, BUF_ABUF, out_alloc.offset_units,
        dst_rows_pad, pad_dim(cols), num_rows, cols, "fp16",
        cg.calibration_scales.get(node.name, 1.0),
    )


def emit_gather_rows(cg: "CodeGenerator", node: IRNode) -> None:
    if not node.inputs:
        raise ValueError(f"{node.name} requires at least one source tensor input")
    if len(node.output_shape) != 2:
        raise ValueError(f"{node.name} requires a 2D output shape, got {node.output_shape}")
    n = len(node.inputs)
    rows = int(node.output_shape[0])
    if rows != n:
        raise ValueError(
            f"{node.name} output rows {rows} must equal the number of source "
            f"tiles {n}"
        )
    cols = int(node.output_shape[1])
    row_units = _row_units(cg, cols)
    src_row = int(node.attrs.get("src_row", 0))

    out_alloc = cg.mem.abuf.alloc(node.name, pad_dim(rows) * pad_dim(cols) * cg.elem_bytes)
    src_M_pad = pad_dim(1)  # each source is a query_len=1 attention output tile
    src_N_pad = pad_dim(cols)
    for i, src_name in enumerate(node.inputs):
        # Per-stream attn_v outputs are spilled to DRAM-temp at production
        # decode scale (Kseq_pad >= 64). Reload before copying, exactly like
        # concat_heads, then free the reloaded ABUF tile so only one source
        # is resident at a time (bounds ABUF pressure across the 16 streams).
        if src_name in cg.dram_temp_fp32_outputs and cg.mem.abuf.get(src_name) is None:
            cg._load_dram_to_abuf_fp(src_name, src_M_pad, src_N_pad)
        src_alloc = cg.mem.abuf.get(src_name)
        if src_alloc is None:
            raise KeyError(f"Missing ABUF allocation '{src_name}' for {node.name}")
        cg._emit(BufCopyInsn(
            src_buf=BUF_ABUF, src_off=src_alloc.offset_units + src_row * row_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units + i * row_units,
            length=row_units,
        ))
        cg._emit(SyncInsn(resource_mask=0b001))
        cg.mem.abuf.free(src_name)
    cg._record_trace_event(
        node.name, BUF_ABUF, out_alloc.offset_units,
        pad_dim(rows), pad_dim(cols), rows, cols, "fp16",
        cg.calibration_scales.get(node.name, 1.0),
    )
