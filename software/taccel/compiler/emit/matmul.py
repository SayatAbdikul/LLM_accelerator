"""Matmul op-family emit helpers — Q@K^T, softmax·V, and head concat.

Free-function migrations of the original `_emit_qkt`, `_emit_attn_v`,
and `_emit_concat_heads` methods on `CodeGenerator`. The simple
linear matmul (`_emit_matmul`) is already a 2-line dispatcher to
`compiler/w8a16_emit/matmul.emit_matmul_w8a16` and stays in
`codegen.py`.

In W8A16 mode each entry point early-dispatches to its matching
`compiler/w8a16_emit/` helper. The legacy INT8 paths preserved below
are kept for source parity with the original method bodies; they are
unreachable under the current `use_fp16_activations = True` hardcode.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...isa.instructions import (
    BufCopyInsn,
    ConfigTileInsn,
    MaskedSoftmaxAttnVInsn,
    MaskedSoftmaxInsn,
    MatmulInsn,
    RequantInsn,
    SetScaleInsn,
    SoftmaxAttnVInsn,
    SoftmaxInsn,
    SyncInsn,
)
from ...isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF
from ..ir import IRNode
from ..tiler import TILE, pad_dim
from ._common import UNIT, _fp16_to_uint16

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def emit_qkt(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit Q@K^T attention matmul, strip-mined over Q's M dimension.

    Full [208,208] INT32 would need 173KB in ACCUM (only 64KB available).
    Instead process 16-row strips: each [16,208] INT32 = 13KB ≤ 64KB.
    SOFTMAX each strip from ACCUM directly to INT8 → WBUF immediately.
    After all strips, WBUF holds [208,208] INT8 = 43KB for downstream softmax.

    W8A32 mode (M3-A): dispatches to `emit_matmul_qkt_w8a16` which
    re-quantizes the FP32 Q and K (produced by the upstream per-head
    Q/K projection matmuls under `emit_matmul_w8a16`) to INT8 with
    static calibration scales, does INT8 MATMUL, and dequants to FP32
    scores via DEQUANT_ACCUM_FP32 with a composite PC scale that folds
    `1/√d_head` into the dequant factor. The downstream `scale_mul`
    IR node becomes a no-op (rename) and `softmax` flows through
    MASKED_SOFTMAX_FP32 (M2 path).
    """
    from ..w8a16_emit import emit_matmul_qkt_w8a16
    emit_matmul_qkt_w8a16(cg, node)


def emit_attn_v(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit attention @ V matmul.

    attn scores are in WBUF (from softmax in-place).
    V_h is in ABUF. MATMUL src1=WBUF[attn], src2=ABUF[V].
    After matmul, free both attn (WBUF) and V (ABUF via last-use).

    W8A32 mode (M3-B): dispatches to `emit_matmul_attn_v_w8a16`.
    Both inputs are FP32 in ABUF by that point (softmax output FP32
    from emit_softmax_fp32; V FP32 from the per-head V projection's
    emit_matmul_w8a16). Re-quant statically, INT8 MATMUL, DEQUANT
    to FP32 attn_v tile. No transpose (V is already K-major).
    """
    from ..w8a16_emit import emit_matmul_attn_v_w8a16
    emit_matmul_attn_v_w8a16(cg, node)


def emit_concat_heads(cg: "CodeGenerator", node: IRNode) -> None:
    """BUF_COPY per-head outputs from WBUF (INT8) or ABUF (W8A32 FP32)
    into a contiguous ABUF region.

    INT8 path: each head's attn_v output [M_pad, head_dim] is in WBUF.
    We interleave them into ABUF as [M_pad, num_heads * head_dim] INT8.

    W8A32 path (M3-B+): each head's attn_v output is FP32 in ABUF
    (4 bytes/elem). We interleave them into a contiguous FP32 ABUF
    region with 4× the byte stride.

    The matmul reads activations as row-major [M, K], so each output row t
    must contain all heads' data for that token:
        tile[t, :] = [h0[t], h1[t], ..., hH[t]]
    """
    if cg._fused_softmax_attnv_accum_out_proj_enabled_for(node.name):
        return

    head_dim = cg.config.d_head
    seq_len = node.output_shape[0]
    M_pad = pad_dim(seq_len)
    N_pad = pad_dim(head_dim)
    num_heads = len(node.inputs)
    # M4-debug (concat_heads W8A32 fix) + M2-W8A16: when use_fp16_activations,
    # per-head attn_v outputs live in ABUF as FP-precision (FP32 or
    # FP16). The pre-M4-debug emitter unconditionally read from WBUF;
    # for W8A32 with n_head>=1 this produced an unfilled out_alloc
    # whose bytes were stale INT8 scratch (read as FP32 → 0x7F7F7F7F ≈
    # 3.4e38). W8A16 needs elem_bytes=2 (FP16 stride) not 4.
    elem_bytes = cg.elem_bytes
    per_head_bytes = N_pad * elem_bytes
    total_out_bytes_per_row = num_heads * per_head_bytes
    # M4-debug (concat_heads ABUF fragmentation): at GPT-2 scale
    # (n_head=12, d_head=64, FP32) the per-head loop fragments ABUF.
    # Need 48 KB contiguous for the concat tile; compaction packs
    # live allocations leftward to expose a large free block.
    cg._compact_abuf()
    out_alloc = cg.mem.abuf.alloc(node.name, M_pad * total_out_bytes_per_row)

    row_units = per_head_bytes // UNIT
    out_row_units = total_out_bytes_per_row // UNIT

    # W8A32: optimize n_head=1 with a rename (no copy). For
    # n_head>1, emit BufCopy from ABUF (the head's FP32 region)
    # into the concat slot, M_pad rows per head.
    if num_heads == 1:
        inp_name = node.inputs[0]
        src_alloc = cg.mem.abuf.get(inp_name)
        if src_alloc is not None:
            # Free the just-allocated out_alloc and rename the
            # head's existing allocation to the concat node name.
            cg.mem.abuf.free(node.name)
            src_alloc = cg.mem.abuf.allocations.pop(inp_name, None)
            if src_alloc is not None:
                src_alloc.name = node.name
                cg.mem.abuf.allocations[node.name] = src_alloc
                out_alloc = src_alloc
    else:
        # M4-debug: per-head attn_v outputs may have been spilled
        # to DRAM-temp by emit_matmul_attn_v_w8a16 (production
        # decode scale Kseq_pad >= 64). Reload each one before
        # BufCopying to its concat slot, then free the reloaded
        # ABUF tile.
        for h, inp_name in enumerate(node.inputs):
            if (
                inp_name in cg.dram_temp_fp32_outputs
                and cg.mem.abuf.get(inp_name) is None
            ):
                cg._load_dram_to_abuf_fp(inp_name, M_pad, N_pad)
            src_alloc = cg.mem.abuf.get(inp_name)
            if src_alloc is None:
                continue
            for t in range(M_pad):
                src_off = src_alloc.offset_units + t * row_units
                dst_off = out_alloc.offset_units + t * out_row_units + h * row_units
                cg._emit(BufCopyInsn(
                    src_buf=BUF_ABUF, src_off=src_off,
                    dst_buf=BUF_ABUF, dst_off=dst_off,
                    length=row_units,
                ))
                cg._emit(SyncInsn(resource_mask=0b001))
            cg.mem.abuf.free(inp_name)
    cg._record_trace_event(
        node.name, BUF_ABUF, out_alloc.offset_units,
        M_pad, total_out_bytes_per_row // elem_bytes,  # elements per row
        seq_len, node.output_shape[1],
        "fp16",
        cg.calibration_scales.get(node.name, 1.0 / 127.0),
    )
