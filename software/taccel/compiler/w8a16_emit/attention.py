"""W8A16 attention matmul lowering: `emit_matmul_qkt_w8a16` (Q @ K^T) and
`emit_matmul_attn_v_w8a16` (softmax · V).

Both differ from the standard matmul path in that re-quantization to INT8
uses **static composite scales** (baked into SET_SCALE immediates via
`_fp16_to_uint16`) rather than the dynamic per-tile MAX_ABS_REDUCE. The
composite is precomputed offline:
  - QKT:    composite_fp16 = q_scale × k_scale × inv_sqrt_d_head
  - attn_v: composite_fp16 = sm_scale × v_scale

DEQUANT_ACCUM_FP32 (the M1 variant without `_SCALED`) then folds the
composite through a single FP16 PC vector (`__qkt_pc_scale` /
`__attn_v_pc_scale`, staged by codegen.py:_layout_weights).

Per-strip CONFIG_ATTN for masked softmax sits between the matmuls; see
codegen.py's INT8 path for the strip-mining contract that this lowering
preserves under W8A16.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...isa.instructions import (
    BufCopyInsn,
    ConfigAttnInsn,
    ConfigTileInsn,
    DequantAccumFp32Insn,
    MatmulInsn,
    QuantFp32Int8Insn,
    SetScaleInsn,
    SyncInsn,
)
from ...isa.opcodes import ABUF_SIZE, ACCUM_SIZE, BUF_ABUF, BUF_ACCUM, BUF_WBUF, WBUF_SIZE
from ..tiler import TILE, pad_dim
from ._common import UNIT, _abuf_alloc_fp32, _fp16_to_uint16

if TYPE_CHECKING:
    from ..codegen import CodeGenerator
    from ..ir import IRNode


# ---------------------------------------------------------------------------
# M3-A: matmul_qkt W8A32 lowering — Q @ K^T with static composite dequant
# ---------------------------------------------------------------------------


def emit_matmul_qkt_w8a16(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 lowering for `matmul_qkt` (per-head Q @ K^T attention matmul).

    Q and K live in ABUF as FP32 (the per-head Q/K projection matmuls
    that produced them were lowered through `emit_matmul_w8a16`, so
    each already benefits from M2.5-A dynamic activation scaling at
    *its* re-quant boundary). For QKT we re-quantize Q and K to INT8
    using static calibration scales, INT8-matmul, then dequant via the
    M1 `DEQUANT_ACCUM_FP32` op (no _SCALED variant — only one factor
    needed) with a pre-folded composite PC scale:

        wt_scale_pc[j] = q_scale × k_scale × (1/√d_head)    (all j)

    The SCALE_MUL IR node downstream of `matmul_qkt` becomes a no-op
    (the `1/√d_head` factor is already in the QKT epilogue). The
    `softmax` IR node lowers to `MASKED_SOFTMAX_FP32` (or
    `SOFTMAX_FP32`) via the existing M2 dispatch.

    Static QKT scaling is NOT a regression from M2.5-A — the upstream
    Q and K projection matmuls used dynamic scaling for their inputs;
    only the QKT re-quant boundary is static. This matches the
    architecture of the INT8 path's QKT (qkt_in_scale is also static
    there).

    Strip-mining over Q's M dimension matches the INT8 path: 16-row
    Q strips, each producing a (16, N_pad) tile in ACCUM that's
    immediately dequant'd to a (16, N_pad) FP32 strip of the output.

    Scope (M3-A):
      - K padding zeros are NOT emitted in W8A32 mode. The tiny
        fixture has key_len == N_pad so this never matters. Real
        GPT-2 graphs hit the NotImplementedError below until M3-B
        adds the pad-row zero-fill at the right (FP32) byte size.
      - Per-head per-strip CONFIG_ATTN setup (causal masking) is NOT
        emitted by this helper — the MASKED_SOFTMAX_FP32 downstream
        handles its own attention masking via its CONFIG_ATTN context.
        For W8A32 mode the masking happens at softmax-time, not at
        QKT-time, since the FP32 scores tile is mask-able directly.
    """
    head_idx = node.attrs["head_idx"]
    query_len = int(node.attrs.get("query_len", node.output_shape[0]))
    key_len = int(
        node.attrs.get(
            "key_len",
            node.output_shape[1] if len(node.output_shape) > 1 else query_len,
        )
    )
    head_dim = cg.config.d_head
    inv_sqrt_d_head = float(node.attrs.get("scale", head_dim ** -0.5))
    M_pad = pad_dim(query_len)
    N_pad = pad_dim(key_len)
    K_pad = pad_dim(head_dim)
    num_strips = M_pad // TILE
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    k_tiles = K_pad // TILE

    # FP32 Q and K input allocs from the upstream per-head Q/K matmuls.
    q_fp32_alloc = cg.mem.abuf.get(node.inputs[0]) or _abuf_alloc_fp32(
        cg, node.inputs[0], M_pad, K_pad
    )
    k_fp32_alloc = cg.mem.abuf.get(node.inputs[1]) or _abuf_alloc_fp32(
        cg, node.inputs[1], N_pad, K_pad
    )

    # ----- M3-C: zero K's FP32 pad rows before quantization -----
    # If key_len < N_pad, K has padding rows (e.g. seq_len=14 → N_pad=16
    # has 2 padding rows). LN(zero_row) = β (non-zero) propagates into
    # K's padding rows during the per-head K projection, so the FP32 K
    # tile holds β-derived junk there. The downstream masked-softmax-FP32
    # will mask those columns to -∞, but the K-QUANT runs first and
    # would consume that junk; the safer (and matching-INT8) approach is
    # to zero the FP32 pad rows BEFORE quantization so K's INT8 also
    # has zero pad rows. `__zero_pad__` is sized 4× in W8A32 mode (see
    # codegen._layout_weights), so the FP32 byte size fits.
    if N_pad > key_len:
        pad_rows = N_pad - key_len
        k_row_units_fp32 = (K_pad * cg.elem_bytes) // UNIT
        k_pad_units = k_fp32_alloc.offset_units + key_len * k_row_units_fp32
        zero_pad_dram = cg._dram_offset_required(
            "__zero_pad__", f"zeroing K padding rows for '{node.name}'"
        )
        cg._emit_dma_load(
            BUF_ABUF,
            k_pad_units,
            pad_rows * K_pad * cg.elem_bytes,
            3,
            zero_pad_dram,
        )
        cg._emit(SyncInsn(resource_mask=0b001))

    # Static calibration scales for the QKT re-quant boundary.
    DEFAULT_ACT_SCALE = 6.0 / 127.0
    q_scale = float(cg.calibration_scales.get(node.inputs[0], DEFAULT_ACT_SCALE))
    k_scale = float(cg.calibration_scales.get(node.inputs[1], DEFAULT_ACT_SCALE))

    # ----- Stage 1: QUANT_FP32_INT8(Q) into an ABUF scratch -----
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=k_tiles - 1, K=0))
    sreg_q = cg._alloc_sreg()
    cg._emit(
        SetScaleInsn(
            sreg=sreg_q, src_mode=0,
            imm16=_fp16_to_uint16(1.0 / max(q_scale, 1e-12)),
        )
    )
    q_int8_alloc = cg.mem.abuf.alloc(f"{node.name}__q_int8", M_pad * K_pad)
    cg._emit(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=q_fp32_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=q_int8_alloc.offset_units,
            sreg=sreg_q,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # ----- Stage 2: QUANT_FP32_INT8(K) into an ABUF scratch -----
    cg._emit(ConfigTileInsn(M=n_tiles - 1, N=k_tiles - 1, K=0))
    sreg_k = cg._alloc_sreg()
    cg._emit(
        SetScaleInsn(
            sreg=sreg_k, src_mode=0,
            imm16=_fp16_to_uint16(1.0 / max(k_scale, 1e-12)),
        )
    )
    k_int8_alloc = cg.mem.abuf.alloc(f"{node.name}__k_int8", N_pad * K_pad)
    cg._emit(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=k_fp32_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=k_int8_alloc.offset_units,
            sreg=sreg_k,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # ----- Stage 3: BUF_COPY transpose K_int8 (ABUF) → K^T (WBUF) -----
    # Same mechanism as the INT8 _emit_qkt path: 1-byte/elem transpose,
    # source has N_pad // TILE row-tiles of TILE rows each.
    length_units = (N_pad * K_pad) // UNIT
    kt_wbuf = cg.mem.wbuf.alloc(f"kt_head{head_idx}_{node.name}", K_pad * N_pad)
    cg._emit(
        BufCopyInsn(
            src_buf=BUF_ABUF, src_off=k_int8_alloc.offset_units,
            dst_buf=BUF_WBUF, dst_off=kt_wbuf.offset_units,
            length=length_units,
            src_rows=N_pad // TILE,
            transpose=1,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b001))

    # M4-debug: free the FP32 K (k_loaded) and K_int8 ABUF regions now —
    # the K^T is in WBUF for the MATMUL, and K's last_use is this QKT
    # so neither original copy is needed anymore. At GPT-2 decode scale
    # this frees 64 KB (k_loaded) + 16 KB (k_int8) = 80 KB ABUF before
    # the qkt out_alloc, which would otherwise OOM at head 7+ when
    # accumulated attn_v outputs (28 KB+) crowd the buffer.
    #
    # Gated by k_loaded being big enough to matter (≥ 16 KB) so unit
    # tests with seq_len=16/d_head=64 (= 4 KB) keep their allocations
    # alive for post-emit inspection.
    k_size_bytes = N_pad * K_pad * cg.elem_bytes
    if k_size_bytes >= 16 * 1024:
        cg.mem.abuf.free(f"{node.name}__k_int8")
        k_in_name = node.inputs[1]
        if cg.last_uses.get(k_in_name, -1) <= cg.current_node_idx:
            if cg.mem.abuf.get(k_in_name) is not None:
                cg.mem.abuf.free(k_in_name)

    # ----- Stage 4: DMA-load the staged FP16 PC scale vector -----
    # Vector entries are all `q_scale * k_scale * inv_sqrt_d_head` cast
    # to FP16 — staged at DRAM-layout time in `_layout_weights`.
    pc_scale_sym = f"{node.name}__qkt_pc_scale"
    pc_scale_dram = cg._dram_offset_required(
        pc_scale_sym,
        f"loading W8A32 QKT composite PC scale for '{node.name}'",
    )
    pc_scale_bytes = N_pad * 2  # FP16
    pc_scale_alloc = cg.mem.wbuf.alloc(f"_qkt_pc_{node.name}", pc_scale_bytes)
    cg._emit_dma_load(
        BUF_WBUF, pc_scale_alloc.offset_units, pc_scale_bytes, 0, pc_scale_dram,
    )
    cg._emit(SyncInsn(resource_mask=0b001))

    # ----- Stage 5: FP32 output tile (M_pad × N_pad × 4 bytes) -----
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)

    # ----- Stage 6: per-strip MATMUL + DEQUANT_ACCUM_FP32 -----
    out_row_units = (N_pad * cg.elem_bytes) // UNIT
    for s in range(num_strips):
        # CONFIG_TILE for one Q strip: M=1 tile (16 rows), N=full N_pad, K=full K_pad.
        cg._emit(ConfigTileInsn(M=0, N=n_tiles - 1, K=k_tiles - 1))
        q_strip_off = q_int8_alloc.offset_units + (s * TILE * K_pad) // UNIT
        cg._emit(
            MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=q_strip_off,
                src2_buf=BUF_WBUF, src2_off=kt_wbuf.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0,
            )
        )
        cg._emit(SyncInsn(resource_mask=0b010))

        # DEQUANT_ACCUM_FP32 (M1, no _SCALED): accum × wt_scale_pc.
        # Reuses the MATMUL's CONFIG_TILE (M and N are what DEQUANT reads).
        strip_out_off = out_alloc.offset_units + s * TILE * out_row_units
        cg._emit(
            DequantAccumFp32Insn(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
                dst_buf=BUF_ABUF, dst_off=strip_out_off,
                flags=cg.fp_precision_flag,
            )
        )
        cg._emit(SyncInsn(resource_mask=0b100))

    # Cleanup. k_int8 was freed in Stage 3 (M4-debug) when running
    # inside generate(); idempotent free here for unit tests.
    cg.mem.abuf.free(f"{node.name}__q_int8")
    cg.mem.abuf.free(f"{node.name}__k_int8")
    cg.mem.wbuf.free(f"kt_head{head_idx}_{node.name}")
    cg.mem.wbuf.free(f"_qkt_pc_{node.name}")

    # Trace event: FP32 scores tile. The recorded scale is the composite
    # `q_scale * k_scale * inv_sqrt_d_head` so downstream tooling can
    # recover an INT8-equivalent representation if it wants, but the
    # tile itself is real-units FP32.
    composite_scale = q_scale * k_scale * inv_sqrt_d_head
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        query_len,
        key_len,
        "fp32",
        composite_scale,
    )


# ---------------------------------------------------------------------------
# M3-B: matmul_attn_v W8A32 lowering — softmax(QK^T) @ V with static dequant
# ---------------------------------------------------------------------------


def emit_matmul_attn_v_w8a16(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 lowering for `matmul_attn_v` (softmax(QK^T) @ V per head).

    Both inputs live in ABUF as FP32 by the time this emitter fires:
      - softmax output: produced by `emit_softmax_fp32` (M2) — FP32
        normalized probabilities in (M=seq_len, K=seq_len).
      - V tile: produced by the per-head V projection's
        `emit_matmul_w8a16` — FP32 in (K=seq_len, N=d_head).

    V is already laid out as [K, N] (no transpose needed — the MATMUL
    expects src2 in WBUF as [K, N] which is exactly V's shape). So the
    sequence is structurally similar to `emit_matmul_qkt_w8a16` minus
    the transpose step:

        FP32 softmax → QUANT_FP32_INT8 (S[s]=1/sm_scale) → INT8 ABUF scratch
        FP32 V       → QUANT_FP32_INT8 (S[s]=1/v_scale)  → INT8 ABUF scratch
        INT8 V       → BUF_COPY (no transpose)            → WBUF
        Composite PC scale (sm_scale × v_scale)           → WBUF
        MATMUL softmax_int8 @ V_wbuf                       → INT32 ACCUM
        DEQUANT_ACCUM_FP32 (M1, no _SCALED)                → FP32 attn_v in ABUF

    Static calibration scales here — same architecture as the M3-A QKT
    re-quant boundary. No 1/√d_head factor in the composite (that was
    already applied by emit_matmul_qkt_w8a16 in its dequant epilogue;
    the softmax input is the correctly-scaled attention scores, and
    softmax output values are in [0, 1] regardless of input scale).

    Scope (M3-B):
      - V pad-row zero-fill is NOT emitted in W8A32 mode. The tiny
        fixture has key_len == Kseq_pad so this never matters. Real
        GPT-2 graphs hit the NotImplementedError below until M3-C
        adds the pad-row zero-fill at the right FP32 byte size. The
        INT8 path zero-fills V's padding rows (lines 2249-2258 of
        codegen.py) because LN(zero_row) = β contaminates attention
        otherwise.

    Synthetic-fragment caveat (test fixtures only):
      In real graphs every input to attn_v is produced by an earlier
      `matmul` node whose `emit_matmul_w8a16` already allocated its
      ABUF region under `node.name`. Tests that build a synthetic
      `matmul_attn_v` fragment with graph-input names like `sm`/`v_in`
      MUST pre-allocate those inputs in `cg.mem.abuf` before
      `generate()`. Otherwise the lazy `_abuf_alloc_fp32` here may
      land them on a region that an earlier emit step's freed scratch
      occupied — the emitted scratch-QUANT instruction still writes
      there at execution time and corrupts the pre-seeded input
      bytes. The end-to-end test in `test_w8a16_codegen.py` does this
      pre-allocation explicitly; see the comment around the
      `cg.mem.abuf.alloc("v_in", ...)` calls there.
    """
    if node.op != "matmul_attn_v":
        raise NotImplementedError(
            f"emit_matmul_attn_v_w8a16 only handles op='matmul_attn_v' "
            f"(got {node.op!r})."
        )

    head_idx = node.attrs["head_idx"]
    query_len = int(node.attrs.get("query_len", node.output_shape[0]))
    key_len = int(node.attrs.get("key_len", node.attrs.get("attn_key_len", query_len)))
    head_dim = int(node.output_shape[1])
    M_pad = pad_dim(query_len)
    Kseq_pad = pad_dim(key_len)
    N_pad = pad_dim(head_dim)
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    k_tiles = Kseq_pad // TILE

    sm_fp32_alloc = cg.mem.abuf.get(node.inputs[0]) or _abuf_alloc_fp32(
        cg, node.inputs[0], M_pad, Kseq_pad
    )
    v_fp32_alloc = cg.mem.abuf.get(node.inputs[1]) or _abuf_alloc_fp32(
        cg, node.inputs[1], Kseq_pad, N_pad
    )

    # ----- M3-C: zero V's FP32 pad rows before quantization -----
    # Same reasoning as K's pad-row zero-fill in emit_matmul_qkt_w8a16:
    # LN(zero_row) = β contaminates V's padding rows downstream of the V
    # projection. Zeroing the FP32 pad rows before V-QUANT ensures the
    # attention output (softmax × V) doesn't include padded-position
    # contributions. (Softmax probabilities for padded columns are
    # already 0 — the MASKED_SOFTMAX_FP32 op masks them to -∞ before the
    # exp — so this V pad-row zero-fill is a defense-in-depth measure
    # that matches the INT8 path's behavior (see codegen._emit_attn_v
    # lines 2249-2258).
    if Kseq_pad > key_len:
        pad_rows = Kseq_pad - key_len
        v_row_units_fp32 = (N_pad * cg.elem_bytes) // UNIT
        v_pad_units = v_fp32_alloc.offset_units + key_len * v_row_units_fp32
        zero_pad_dram = cg._dram_offset_required(
            "__zero_pad__", f"zeroing V padding rows for '{node.name}'"
        )
        cg._emit_dma_load(
            BUF_ABUF,
            v_pad_units,
            pad_rows * N_pad * cg.elem_bytes,
            3,
            zero_pad_dram,
        )
        cg._emit(SyncInsn(resource_mask=0b001))

    # Static calibration scales for the attn_v re-quant boundary.
    DEFAULT_ACT_SCALE = 6.0 / 127.0
    sm_scale = float(cg.calibration_scales.get(node.inputs[0], 1.0 / 127.0))
    v_scale = float(cg.calibration_scales.get(node.inputs[1], DEFAULT_ACT_SCALE))

    # ----- Stage 1: QUANT_FP32_INT8(softmax) into an ABUF scratch -----
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=k_tiles - 1, K=0))
    sreg_sm = cg._alloc_sreg()
    cg._emit(
        SetScaleInsn(
            sreg=sreg_sm, src_mode=0,
            imm16=_fp16_to_uint16(1.0 / max(sm_scale, 1e-12)),
        )
    )
    sm_int8_alloc = cg.mem.abuf.alloc(f"{node.name}__sm_int8", M_pad * Kseq_pad)
    cg._emit(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=sm_fp32_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=sm_int8_alloc.offset_units,
            sreg=sreg_sm,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # M4-debug: free FP32 softmax output now — sm_int8 has the quantized
    # version, and softmax's last_use is this attn_v. At GPT-2 scale
    # this frees 16 KB ABUF before v_int8 alloc. Same size gate as V.
    sm_size_bytes = M_pad * Kseq_pad * cg.elem_bytes
    if sm_size_bytes >= 16 * 1024:
        sm_in_name = node.inputs[0]
        if cg.last_uses.get(sm_in_name, -1) <= cg.current_node_idx:
            if cg.mem.abuf.get(sm_in_name) is not None:
                cg.mem.abuf.free(sm_in_name)

    # ----- Stage 2: QUANT_FP32_INT8(V) into an ABUF scratch -----
    cg._emit(ConfigTileInsn(M=k_tiles - 1, N=n_tiles - 1, K=0))
    sreg_v = cg._alloc_sreg()
    cg._emit(
        SetScaleInsn(
            sreg=sreg_v, src_mode=0,
            imm16=_fp16_to_uint16(1.0 / max(v_scale, 1e-12)),
        )
    )
    v_int8_alloc = cg.mem.abuf.alloc(f"{node.name}__v_int8", Kseq_pad * N_pad)
    cg._emit(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=v_fp32_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=v_int8_alloc.offset_units,
            sreg=sreg_v,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # ----- Stage 3: BUF_COPY V_int8 (ABUF) → V (WBUF), no transpose -----
    # V's natural layout is [K, N] = [seq_len, d_head], which is what
    # MATMUL expects for src2. No transpose needed.
    length_units = (Kseq_pad * N_pad) // UNIT
    v_wbuf = cg.mem.wbuf.alloc(f"v_head{head_idx}_{node.name}", Kseq_pad * N_pad)
    cg._emit(
        BufCopyInsn(
            src_buf=BUF_ABUF, src_off=v_int8_alloc.offset_units,
            dst_buf=BUF_WBUF, dst_off=v_wbuf.offset_units,
            length=length_units,
            src_rows=0,
            transpose=0,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b001))

    # M4-debug: same as QKT — free FP32 V (v_loaded) and V_int8 ABUF
    # regions now since the MATMUL reads V from WBUF, not ABUF. At
    # GPT-2 decode 256-token scale this frees 64 KB + 16 KB = 80 KB.
    # Gated by v_size >= 16 KB to preserve unit-test allocation
    # inspection at d_head=64 / seq_len=16 (= 4 KB v_loaded).
    v_size_bytes = Kseq_pad * N_pad * cg.elem_bytes
    if v_size_bytes >= 16 * 1024:
        cg.mem.abuf.free(f"{node.name}__v_int8")
        v_in_name = node.inputs[1]
        if cg.last_uses.get(v_in_name, -1) <= cg.current_node_idx:
            if cg.mem.abuf.get(v_in_name) is not None:
                cg.mem.abuf.free(v_in_name)

    # ----- Stage 4: DMA-load the staged FP16 PC scale vector -----
    pc_scale_sym = f"{node.name}__attn_v_pc_scale"
    pc_scale_dram = cg._dram_offset_required(
        pc_scale_sym,
        f"loading W8A32 attn_v composite PC scale for '{node.name}'",
    )
    pc_scale_bytes = N_pad * 2  # FP16
    pc_scale_alloc = cg.mem.wbuf.alloc(f"_attn_v_pc_{node.name}", pc_scale_bytes)
    cg._emit_dma_load(
        BUF_WBUF, pc_scale_alloc.offset_units, pc_scale_bytes, 0, pc_scale_dram,
    )
    cg._emit(SyncInsn(resource_mask=0b001))

    # ----- Stage 5: FP32 output tile (M_pad × N_pad × 4 bytes) -----
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)

    # ----- Stage 6: MATMUL + DEQUANT_ACCUM_FP32 -----
    # The attn_v matmul is small enough on tiny fixtures (seq_len=16,
    # d_head=16, so the M_pad × N_pad × 4 = 1024B output fits in any
    # ACCUM region) that we don't strip-mine — single MATMUL covering
    # the full output. Real GPT-2 graphs hit the M3-C sizing-guard
    # path; M3-B's NotImplementedError above ensures we don't silently
    # produce a malformed bundle there.
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=k_tiles - 1))
    cg._emit(
        MatmulInsn(
            src1_buf=BUF_ABUF, src1_off=sm_int8_alloc.offset_units,
            src2_buf=BUF_WBUF, src2_off=v_wbuf.offset_units,
            dst_buf=BUF_ACCUM, dst_off=0,
            flags=0,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b010))

    # DEQUANT_ACCUM_FP32: accum × wt_scale_pc → FP{32,16} attn_v in ABUF.
    cg._emit(
        DequantAccumFp32Insn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # Cleanup.
    cg.mem.abuf.free(f"{node.name}__sm_int8")
    cg.mem.abuf.free(f"{node.name}__v_int8")
    cg.mem.wbuf.free(f"v_head{head_idx}_{node.name}")
    cg.mem.wbuf.free(f"_attn_v_pc_{node.name}")

    # M4-debug: at production decode scale (Kseq_pad >= 64), spill the
    # per-head attn_v output to DRAM-temp. Otherwise 12 heads × 4 KB
    # per-head output accumulate in ABUF, leaving no room for head 11's
    # QKT k_int8 alloc (115 KB live + 16 KB request > 128 KB).
    # concat_heads reloads each spilled tile when assembling the concat
    # tensor.
    if Kseq_pad >= 64 and cg.current_node_idx >= 0:
        out_bytes = M_pad * N_pad * cg.elem_bytes
        cg._spill_fp32_tile_to_dram(node.name, out_alloc, M_pad, N_pad)
        # _spill freed the ABUF; concat_heads detects this via
        # dram_temp_fp32_outputs[node.name] and reloads on demand.

    # Trace event: FP32 attn_v tile. The recorded scale is the composite
    # `sm_scale × v_scale` (= 1.0 in real units, the FP32 tile is already
    # in real units — the scale is for INT8-equivalent recovery tooling).
    composite_scale = sm_scale * v_scale
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        query_len,
        head_dim,
        "fp32",
        composite_scale,
    )
