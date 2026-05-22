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
    if cg.use_fp16_activations:
        from ..w8a16_emit import emit_matmul_qkt_w8a16
        emit_matmul_qkt_w8a16(cg, node)
        return
    head_idx = node.attrs["head_idx"]
    query_len = int(node.attrs.get("query_len", node.output_shape[0]))
    key_len = int(node.attrs.get("key_len", node.output_shape[1] if len(node.output_shape) > 1 else query_len))
    head_dim = cg.config.d_head
    M_pad = pad_dim(query_len)
    N_pad = pad_dim(key_len)
    K_pad = pad_dim(head_dim)
    num_strips = M_pad // TILE  # 13 strips of 16 rows
    trace_qkt_debug = re.match(r"block\d+_head\d+_qkt$", node.name) is not None
    act_scale_q = cg.calibration_scales.get(node.inputs[0], 6.0 / 127.0)
    act_scale_k = cg.calibration_scales.get(node.inputs[1], 6.0 / 127.0)

    # BUF_COPY K_h → WBUF (transpose) to get K^T [64,208]
    k_alloc = cg.mem.abuf.get(node.inputs[1])
    if k_alloc is None:
        k_alloc = cg.mem.abuf.alloc(node.inputs[1], N_pad * K_pad)

    # Zero out K rows for padding positions (197-207).
    # LN(zero_row) = layernorm_beta (non-zero), so K[padding] = W_k @ beta + b_k.
    # Zeroing removes this contribution so padding columns don't steer attention.
    real_key_len = key_len
    if N_pad > real_key_len:
        pad_rows = N_pad - real_key_len
        k_pad_units = k_alloc.offset_units + (real_key_len * K_pad) // UNIT
        zero_pad_dram = cg._dram_offset_required("__zero_pad__", "loading K padding mask")
        cg._emit_dma_load(BUF_ABUF, k_pad_units, pad_rows * K_pad, 3,
                          zero_pad_dram)
        cg._emit(SyncInsn(resource_mask=0b001))

    src_rows = N_pad // TILE
    length_units = (N_pad * K_pad) // UNIT
    kt_wbuf = cg.mem.wbuf.alloc(f"kt_head{head_idx}", K_pad * N_pad)
    cg._emit(BufCopyInsn(
        src_buf=BUF_ABUF, src_off=k_alloc.offset_units,
        dst_buf=BUF_WBUF, dst_off=kt_wbuf.offset_units,
        length=length_units,
        src_rows=src_rows,
        transpose=1,
    ))
    key_transpose_pc = len(cg.instructions) - 1
    if trace_qkt_debug:
        # Snapshot the exact padded K tensor consumed by BUF_COPY and the
        # transposed WBUF tensor it produces. If the first divergence moves
        # to one of these traces we know whether the bug is in K
        # preparation or later in the Q x K^T path.
        cg._record_trace_event(
            f"{node.name}__key_padded_input",
            BUF_ABUF,
            k_alloc.offset_units,
            N_pad,
            K_pad,
            N_pad,
            K_pad,
            "int8",
            act_scale_k,
            full_rows=N_pad,
            full_cols=K_pad,
            pc=key_transpose_pc,
        )
        cg._record_trace_event(
            f"{node.name}__key_transposed",
            BUF_WBUF,
            kt_wbuf.offset_units,
            K_pad,
            N_pad,
            K_pad,
            N_pad,
            "int8",
            act_scale_k,
            full_rows=K_pad,
            full_cols=N_pad,
            pc=key_transpose_pc,
        )
    cg._emit(SyncInsn(resource_mask=0b001))

    q_alloc = cg.mem.abuf.get(node.inputs[0])
    if q_alloc is None:
        q_alloc = cg.mem.abuf.alloc(node.inputs[0], M_pad * K_pad)

    attn_mode = cg._attention_mask_mode_for_qkt(node, N_pad)
    fused_softmax_attnv = cg._block_selected(node.name, cg.fused_softmax_attnv_blocks)
    softmax_name = node.name.replace("_qkt", "_softmax")
    attn_v_name = node.name.replace("_qkt", "_attn_v")
    value_name = node.name.replace("_qkt", "_value")

    n_tiles = N_pad // TILE
    k_tiles = K_pad // TILE
    # C1: softmax consumes raw ACCUM values with this dequant scale.
    # qkt_in_scale = q_scale * k_scale * (1/sqrt(d_head)).
    # Do NOT look up node.name here: _emit_qkt itself writes that key
    # (calibration_scales[node.name] = softmax_out_scale) so that
    # _emit_attn_v can recover the softmax output scale.  If prefill
    # and decode codegens share the same dict, the decode codegen would
    # read back softmax_out_scale and use it as qkt_in_scale, which is
    # ~1000x too large and collapses softmax to a degenerate distribution.
    qkt_in_scale = act_scale_q * act_scale_k * node.attrs.get("scale", 0.125)
    softmax_out_scale = cg.calibration_scales.get(softmax_name, 1.0 / 127.0)
    if fused_softmax_attnv:
        v_alloc = cg.mem.abuf.get(value_name)
        if v_alloc is None:
            v_alloc = cg.mem.abuf.alloc(value_name, N_pad * K_pad)
        if N_pad > real_key_len:
            pad_rows = N_pad - real_key_len
            v_pad_units = v_alloc.offset_units + (real_key_len * K_pad) // UNIT
            zero_pad_dram = cg._dram_offset_required("__zero_pad__", "loading V padding mask")
            cg._emit_dma_load(BUF_ABUF, v_pad_units, pad_rows * K_pad, 3, zero_pad_dram)
            cg._emit(SyncInsn(resource_mask=0b001))
        target_act_scale = cg.calibration_scales.get(attn_v_name, 6.0 / 127.0)
        attn_v_alloc = cg.mem.wbuf.alloc(attn_v_name, M_pad * K_pad)
        v_scale = cg.calibration_scales.get(value_name, 6.0 / 127.0)
    else:
        # Output: full [208,208] INT8 softmax probabilities in WBUF
        qkt_wbuf = cg.mem.wbuf.alloc(node.name, M_pad * N_pad)

    for s in range(num_strips):
        row_start = s * TILE
        logical_rows = max(0, min(TILE, query_len - row_start))
        # CONFIG_TILE: M=1 strip (16 rows), N=full, K=head_dim
        cg._emit(ConfigTileInsn(M=0, N=n_tiles - 1, K=k_tiles - 1))
        qkt_config_pc = len(cg.instructions) - 1
        if attn_mode is not None and not fused_softmax_attnv:
            cg._emit_config_attn_for_qkt(
                node,
                row_start=row_start,
                valid_kv_len=key_len,
                mode=attn_mode,
            )
        if trace_qkt_debug:
            # Snapshot ACCUM immediately before the QK^T MATMUL. CONFIG_TILE
            # itself does not mutate SRAM, so tracing at this PC gives us the
            # architectural pre-state without adding a new "before" semantic
            # to the trace manifest.
            cg._record_trace_event(
                f"{node.name}__accum_pre_matmul",
                BUF_ACCUM,
                0,
                TILE,
                N_pad,
                logical_rows,
                key_len,
                "int32",
                qkt_in_scale,
                row_start=row_start,
                full_rows=query_len,
                full_cols=key_len,
                pc=qkt_config_pc,
            )
            cg._record_trace_event(
                f"{node.name}__accum_pre_matmul_next",
                BUF_ACCUM,
                0,
                TILE,
                N_pad,
                logical_rows,
                key_len,
                "int32",
                qkt_in_scale,
                row_start=row_start,
                full_rows=query_len,
                full_cols=key_len,
                pc=qkt_config_pc,
                capture_phase="retire_plus_1",
            )

        # Q strip offset: s * 16 rows * K_pad cols
        q_strip_off = q_alloc.offset_units + (s * TILE * K_pad) // UNIT
        cg._emit(MatmulInsn(
            src1_buf=BUF_ABUF, src1_off=q_strip_off,
            src2_buf=BUF_WBUF, src2_off=kt_wbuf.offset_units,
            dst_buf=BUF_ACCUM, dst_off=0,
            flags=0,
        ))
        qkt_matmul_pc = len(cg.instructions) - 1
        if trace_qkt_debug:
            cg._record_trace_event(
                f"{node.name}__query_input",
                BUF_ABUF,
                q_strip_off,
                TILE,
                K_pad,
                logical_rows,
                head_dim,
                "int8",
                act_scale_q,
                row_start=row_start,
                full_rows=query_len,
                full_cols=head_dim,
                pc=qkt_matmul_pc,
            )
        cg._emit(SyncInsn(resource_mask=0b010))
        cg._record_trace_event(
            node.name,
            BUF_ACCUM,
            0,
            TILE,
            N_pad,
            logical_rows,
            key_len,
            "int32",
            qkt_in_scale,
            row_start=row_start,
            full_rows=query_len,
            full_cols=key_len,
        )

        if fused_softmax_attnv:
            cg._emit(ConfigTileInsn(M=0, N=k_tiles - 1, K=n_tiles - 1))
            if attn_mode is not None:
                cg._emit_config_attn_for_qkt(
                    node,
                    row_start=row_start,
                    valid_kv_len=key_len,
                    mode=attn_mode,
                )
            sreg = cg._alloc_sreg_quad()
            cg._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(qkt_in_scale)))
            cg._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0, imm16=_fp16_to_uint16(v_scale)))
            cg._emit(SetScaleInsn(sreg=sreg + 2, src_mode=0, imm16=_fp16_to_uint16(target_act_scale)))
            cg._emit(SetScaleInsn(sreg=sreg + 3, src_mode=0, imm16=_fp16_to_uint16(softmax_out_scale)))
            strip_out_off = attn_v_alloc.offset_units + (s * TILE * K_pad) // UNIT
            fused_cls = MaskedSoftmaxAttnVInsn if attn_mode is not None else SoftmaxAttnVInsn
            cg._emit(fused_cls(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_ABUF, src2_off=v_alloc.offset_units,
                dst_buf=BUF_WBUF, dst_off=strip_out_off,
                sreg=sreg,
            ))
            fused_pc = len(cg.instructions) - 1
            cg._emit(SyncInsn(resource_mask=0b100))
            cg._record_trace_event(
                softmax_name,
                BUF_WBUF,
                0,
                TILE,
                N_pad,
                logical_rows,
                key_len,
                "int8",
                softmax_out_scale,
                row_start=row_start,
                full_rows=query_len,
                full_cols=key_len,
                pc=fused_pc,
            )
            cg.trace_manifest.setdefault(fused_pc, [])[-1]["source"] = "virtual"
            cg._record_trace_event(
                attn_v_name,
                BUF_WBUF,
                strip_out_off,
                TILE,
                K_pad,
                logical_rows,
                head_dim,
                "int8",
                target_act_scale,
                row_start=row_start,
                full_rows=query_len,
                full_cols=head_dim,
                pc=fused_pc,
            )
        else:
            # C1: SOFTMAX directly from ACCUM to avoid QKT INT8 bottleneck.
            # in_scale dequants INT32 accumulators; out_scale quantizes probabilities.
            sreg = cg._alloc_sreg_pair()
            cg._emit(SetScaleInsn(sreg=sreg, src_mode=0,
                                  imm16=_fp16_to_uint16(qkt_in_scale)))
            cg._emit(SetScaleInsn(sreg=sreg + 1, src_mode=0,
                                  imm16=_fp16_to_uint16(softmax_out_scale)))
            strip_wbuf_off = qkt_wbuf.offset_units + (s * TILE * N_pad) // UNIT
            softmax_cls = MaskedSoftmaxInsn if attn_mode is not None else SoftmaxInsn
            cg._emit(softmax_cls(
                src1_buf=BUF_ACCUM, src1_off=0,
                dst_buf=BUF_WBUF, dst_off=strip_wbuf_off,
                sreg=sreg,
            ))
            softmax_pc = len(cg.instructions) - 1
            if trace_qkt_debug:
                cg._record_trace_event(
                    f"{node.name}__accum_pre_softmax",
                    BUF_ACCUM,
                    0,
                    TILE,
                    N_pad,
                    logical_rows,
                    key_len,
                    "int32",
                    qkt_in_scale,
                    row_start=row_start,
                    full_rows=query_len,
                    full_cols=key_len,
                    pc=softmax_pc,
                )
                cg._record_trace_event(
                    f"{node.name}__accum_pre_softmax_next",
                    BUF_ACCUM,
                    0,
                    TILE,
                    N_pad,
                    logical_rows,
                    key_len,
                    "int32",
                    qkt_in_scale,
                    row_start=row_start,
                    full_rows=query_len,
                    full_cols=key_len,
                    pc=softmax_pc,
                    capture_phase="retire_plus_1",
                )
            cg._emit(SyncInsn(resource_mask=0b100))
            cg._record_trace_event(
                softmax_name,
                BUF_WBUF,
                strip_wbuf_off,
                TILE,
                N_pad,
                logical_rows,
                key_len,
                "int8",
                softmax_out_scale,
                row_start=row_start,
                full_rows=query_len,
                full_cols=key_len,
            )

    cg.mem.wbuf.free(f"kt_head{head_idx}")
    # Metadata now reflects softmax-quantized output in node.name allocation.
    cg.calibration_scales[node.name] = softmax_out_scale


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
    if cg.use_fp16_activations:
        from ..w8a16_emit import emit_matmul_attn_v_w8a16
        emit_matmul_attn_v_w8a16(cg, node)
        return
    if cg._block_selected(node.name, cg.fused_softmax_attnv_blocks):
        return

    head_idx = node.attrs["head_idx"]
    query_len = int(node.attrs.get("query_len", node.output_shape[0]))
    key_len = int(node.attrs.get("key_len", node.attrs.get("attn_key_len", query_len)))
    head_dim = node.output_shape[1]
    M_pad = pad_dim(query_len)
    Kseq_pad = pad_dim(key_len)
    N_pad = pad_dim(head_dim)

    # attn scores in WBUF under the softmax node name
    attn_alloc = cg.mem.wbuf.get(node.inputs[0])
    if attn_alloc is None:
        attn_alloc = cg.mem.wbuf.alloc(node.inputs[0], M_pad * Kseq_pad)

    # V_h is the per-head ABUF allocation
    v_alloc = cg.mem.abuf.get(node.inputs[1])
    if v_alloc is None:
        v_alloc = cg.mem.abuf.alloc(node.inputs[1], Kseq_pad * N_pad)

    # Zero out V rows for padding positions (197-207).
    # Same reason as K: LN(zero_row) = beta propagates non-zero values into V.
    # Zeroing V ensures padding positions contribute nothing to attn@V output.
    if Kseq_pad > key_len:
        pad_rows = Kseq_pad - key_len
        v_pad_units = v_alloc.offset_units + (key_len * N_pad) // UNIT
        zero_pad_dram = cg._dram_offset_required("__zero_pad__", "loading V padding mask")
        cg._emit_dma_load(BUF_ABUF, v_pad_units, pad_rows * N_pad, 3,
                          zero_pad_dram)
        cg._emit(SyncInsn(resource_mask=0b001))

    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    k_tiles = Kseq_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=k_tiles - 1))

    # MATMUL: attn(WBUF) @ V(ABUF) → ACCUM
    cg._emit(MatmulInsn(
        src1_buf=BUF_WBUF, src1_off=attn_alloc.offset_units,
        src2_buf=BUF_ABUF, src2_off=v_alloc.offset_units,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=0,
    ))
    cg._emit(SyncInsn(resource_mask=0b010))

    # Free attn scores from WBUF
    cg.mem.wbuf.free(node.inputs[0])
    # Also free the scale_mul intermediate if still present
    for inp in node.inputs:
        cg.mem.wbuf.free(inp)

    # Requantize: attn (INT8 softmax output) @ V (INT8 activation)  → INT32
    # requant_scale = attn_scale * v_scale / target_act_scale
    # attn_scale is the calibrated softmax output scale (max_prob/127 per head).
    # Using 1/127 would overestimate by 1/max_prob (up to 4×), causing heavy clipping.
    attn_scale = cg.calibration_scales.get(node.inputs[0], 1.0 / 127.0)
    v_scale = cg.calibration_scales.get(node.inputs[1], 6.0 / 127.0)
    target_act_scale = cg.calibration_scales.get(node.name, 6.0 / 127.0)
    requant_scale_f = attn_scale * v_scale / max(target_act_scale, 1e-12)
    sreg = cg._alloc_sreg()
    cg._emit(SetScaleInsn(sreg=sreg, src_mode=0, imm16=_fp16_to_uint16(requant_scale_f)))
    out_alloc = cg.mem.wbuf.alloc(node.name, M_pad * N_pad)
    cg._emit(RequantInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        dst_buf=BUF_WBUF, dst_off=out_alloc.offset_units,
        sreg=sreg,
    ))
    cg._record_trace_event(
        node.name,
        BUF_WBUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        query_len,
        head_dim,
        "int8",
        target_act_scale,
    )


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
    elem_bytes = cg.elem_bytes if cg.use_fp16_activations else 1
    per_head_bytes = N_pad * elem_bytes
    total_out_bytes_per_row = num_heads * per_head_bytes
    # M4-debug (concat_heads ABUF fragmentation): at GPT-2 scale
    # (n_head=12, d_head=64, FP32) the per-head loop fragments ABUF.
    # Need 48 KB contiguous for the concat tile; compaction packs
    # live allocations leftward to expose a large free block.
    if cg.use_fp16_activations:
        cg._compact_abuf()
    out_alloc = cg.mem.abuf.alloc(node.name, M_pad * total_out_bytes_per_row)

    row_units = per_head_bytes // UNIT
    out_row_units = total_out_bytes_per_row // UNIT

    if cg.use_fp16_activations:
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
        return

    # Legacy INT8 path: per-head outputs are in WBUF.
    for h, inp_name in enumerate(node.inputs):
        src_alloc = cg.mem.wbuf.get(inp_name)
        if src_alloc is None:
            continue
        for t in range(M_pad):
            src_off = src_alloc.offset_units + t * row_units
            dst_off = out_alloc.offset_units + t * out_row_units + h * row_units
            cg._emit(BufCopyInsn(
                src_buf=BUF_WBUF, src_off=src_off,
                dst_buf=BUF_ABUF, dst_off=dst_off,
                length=row_units,
            ))
            cg._emit(SyncInsn(resource_mask=0b001))
        cg.mem.wbuf.free(inp_name)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        total_out_bytes_per_row,
        seq_len,
        node.output_shape[1],
        "int8",
        cg.calibration_scales.get(node.name, 6.0 / 127.0),
    )
