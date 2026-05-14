"""W8A16 matmul lowering using the M2.5-A dynamic activation-scale primitives.

Three entry points dispatch by sizing:
  - `emit_matmul_w8a16`                       — simple path; everything fits
    ABUF/ACCUM/WBUF (Q/K/V/out_proj at GPT-2 124M scale).
  - `emit_matmul_w8a16_large_weight_tiled`    — weight > WBUF or accum >
    ACCUM; outer-loops over N-tile groups (fc1 at GPT-2).
  - `_emit_matmul_w8a16_large_input_streaming` — fc2-style streaming for
    when the input is itself in DRAM-temp (large fc1 output spill).

The shared per-matmul prelude+epilogue:
    FP source → MAX_ABS_REDUCE_FP32 → S[s]=127/max, S[s+1]=max/127
              → QUANT_FP32_INT8 (S[s])  → INT8 scratch in ABUF
              → MATMUL (INT8 × INT8)    → INT32 ACCUM
              → DEQUANT_ACCUM_FP32_SCALED (× FP16 PC weight scales × S[s+1])
              → FP-precision output in ABUF (bias folded into the
                DEQUANT_FP32_SCALED epilogue when present; see
                `cg.biases` and the `__w8a16_pc_scale_and_bias` blob in
                codegen.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...isa.instructions import (
    BufCopyInsn,
    ConfigTileInsn,
    DequantAccumFp32ScaledInsn,
    MatmulInsn,
    MaxAbsReduceFp32Insn,
    QuantFp32Int8Insn,
    SetScaleInsn,
    SyncInsn,
    VaddFp32Insn,
)
from ...isa.opcodes import ABUF_SIZE, ACCUM_SIZE, BUF_ABUF, BUF_ACCUM, BUF_WBUF, WBUF_SIZE
from ..tiler import TILE, pad_dim
from ._common import UNIT, _abuf_alloc_fp32, _zero_fill_fp32_padding_rows

if TYPE_CHECKING:
    from ..codegen import CodeGenerator
    from ..ir import IRNode


# ---------------------------------------------------------------------------
# M2.5-B: full matmul lowering using the M2.5-A dynamic-scale primitives.
# ---------------------------------------------------------------------------


def emit_matmul_w8a16(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 matmul lowering using the M2.5-A dynamic activation-scale primitives.

    Replaces the INT8 path's REQUANT/REQUANT_PC epilogue with the
    M2.5-A FP32 round-trip:

        FP32 input → MAX_ABS_REDUCE_FP32 → S[s] = 127/max_abs,
                                            S[s+1] = max_abs/127
                  → QUANT_FP32_INT8 (S[s]) → INT8 scratch in ABUF
                  → MATMUL (INT8 × INT8)   → INT32 ACCUM
                  → DEQUANT_ACCUM_FP32_SCALED (× FP16 PC wt scales × S[s+1])
                  → FP32 output in ABUF
                  → [if node.attrs['bias']: replicate FP32 bias to M_pad
                     rows via DMA, then VADD_FP32 onto the FP32 output]

    Scope (M2.5-B):
      - Simple matmul only. The sizing thresholds that force
        strip-mining / large-weight-tiling in the INT8 path raise
        NotImplementedError here.
      - `biases` must contain an entry for any `node.attrs['bias']`
        referenced; the codegen stages it to DRAM as `f"{bias_name}__fp32"`.
      - Fused gelu-from-accum and dequant-add residual epilogues are
        force-disabled at CodeGenerator init; this helper raises if they
        somehow re-surface.
    """
    if node.op != "matmul":
        raise NotImplementedError(
            f"emit_matmul_w8a16 only handles op='matmul' (got {node.op!r}); "
            "matmul_qkt and matmul_attn_v are M3."
        )
    if cg._dequant_add_enabled_for_output(node.name):
        raise NotImplementedError(
            f"W8A32 matmul lowering does not support fused DEQUANT_ADD "
            f"residual epilogue (node '{node.name}'). The dynamic "
            "activation scale in M2.5-A makes the prescaled-INT32 skip "
            "path incompatible."
        )
    if node.attrs.get("gelu_from_accum"):
        raise NotImplementedError(
            f"W8A32 matmul lowering does not support fused gelu-from-accum "
            f"(node '{node.name}'). The FP32 sub-layer ops keep GELU in a "
            "separate emit_gelu_fp32 call."
        )
    # M4-C: `stage4_weight_tiled` is now honored — it forces the
    # large-weight-tiled lowering even when the weight would fit the
    # simple path's WBUF threshold. Useful for tests that exercise the
    # tiled path on small fixtures.
    force_tiled = bool(node.attrs.get("stage4_weight_tiled"))

    M, N = node.output_shape
    weight_name = node.inputs[1]
    weight_data = cg.weight_data.get(weight_name)
    if weight_data is None:
        return

    w_q, w_scales = weight_data
    if w_q.ndim != 2:
        raise NotImplementedError(
            f"W8A32 matmul lowering expects a 2-D weight tensor for "
            f"'{weight_name}', got shape {w_q.shape}."
        )
    K = int(w_q.shape[0])
    M_pad = pad_dim(M)
    N_pad = pad_dim(N)
    K_pad = pad_dim(K)

    # Sizing thresholds — when any of accum / output / weight overflows
    # the simple lowering's buffer, dispatch to the M4-C large-weight-
    # tiled emitter (or raise for fc2-style large-input cases that we
    # defer to a follow-up milestone).
    output_bytes = M_pad * N_pad * cg.elem_bytes
    accum_bytes = M_pad * N_pad * 4  # INT32
    weight_bytes = int(w_q.size)
    input_bytes = M_pad * K_pad * cg.elem_bytes
    if input_bytes > ABUF_SIZE and node.inputs[0] not in cg.dram_temp_fp32_outputs:
        raise NotImplementedError(
            f"W8A32 matmul '{node.name}' input '{node.inputs[0]}' is "
            f"{input_bytes}B (> ABUF {ABUF_SIZE}) and not in DRAM-temp. "
            "K-tile activation streaming from DRAM-temp is deferred."
        )
    if (
        force_tiled
        or output_bytes > ABUF_SIZE
        or accum_bytes > ACCUM_SIZE
        or weight_bytes > WBUF_SIZE
    ):
        emit_matmul_w8a16_large_weight_tiled(cg, node)
        return

    # Per-channel FP16 weight-scale + folded bias vector staging symbol —
    # set up by `CodeGenerator.generate()` at DRAM-staging time when
    # use_fp16_activations. 2N FP16 (PC scales + bias), folded into the DEQUANT
    # epilogue per the bias-fold contract. Bias half is zero-padded when
    # the matmul has no bias.
    bias_name = node.attrs.get("bias")
    pc_scale_sym = f"{weight_name}__w8a16_pc_scale_and_bias"

    # ----- ABUF allocations -----
    # FP32 input tile (M_pad × K_pad × 4 bytes). The prior op's output
    # FP32 alloc is reused here under the input's name.
    #
    # M4-debug: if the input was spilled to DRAM-temp by an earlier
    # emit_matmul_w8a16 (post-QUANT spill, see below), reload it now.
    input_name = node.inputs[0]
    if (
        input_name in cg.dram_temp_fp32_outputs
        and cg.mem.abuf.get(input_name) is None
    ):
        in_alloc = cg._load_dram_to_abuf_fp(input_name, M_pad, K_pad)
    else:
        in_alloc = _abuf_alloc_fp32(cg, input_name, M_pad, K_pad)
    # FP32 output tile (M_pad × N_pad × 4 bytes).
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    # INT8 scratch tile (M_pad × K_pad × 1 byte) for the QUANT output.
    # Lives in ABUF, freed immediately after the MATMUL consumes it.
    int8_scratch = cg.mem.abuf.alloc(f"{node.name}__quant_int8", M_pad * K_pad)

    # M4-debug: zero-fill FP32 padding rows before MAX_ABS_REDUCE so the
    # dynamic activation scale is computed against only the valid query
    # rows. Otherwise LN-beta / matmul-bias leakage into padding rows
    # inflates the scale → INT8 precision loss → catastrophic PPL.
    _zero_fill_fp32_padding_rows(cg, in_alloc, M, M_pad, K_pad)

    # Allocate an sreg pair: S[sreg] holds 127/max_abs for QUANT,
    # S[sreg+1] holds max_abs/127 for the DEQUANT epilogue.
    sreg = cg._alloc_sreg_pair()

    # ----- Stage 1: MAX_ABS_REDUCE_FP32 -----
    # CONFIG_TILE for the activation tile: M_pad rows × K_pad cols.
    m_tiles = M_pad // TILE
    k_tiles = K_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=k_tiles - 1, K=0))
    cg._emit(
        MaxAbsReduceFp32Insn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=0,
            sreg=sreg,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))  # wait SFU

    # ----- Stage 2: QUANT_FP32_INT8 (FP{32,16} ABUF → INT8 ABUF) -----
    cg._emit(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=int8_scratch.offset_units,
            sreg=sreg,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # M4-debug: free the FP32 input ABUF region now — the INT8 scratch
    # contains the quantized version and the rest of this matmul
    # emission doesn't read FP32. Three cases:
    #   1. Last use is this matmul: just free ABUF (post-emit free would
    #      do the same but later — early-free helps the downstream
    #      bias VADD's allocation pressure).
    #   2. Future use AND tile big enough to matter AND not yet in
    #      DRAM-temp: spill to DRAM-temp; the reload at the next
    #      consumer's start uses _load_dram_to_abuf_fp.
    #   3. Future use AND already in DRAM-temp (this matmul reloaded
    #      from a prior spill): just free the ABUF copy. The DRAM slot
    #      from the earlier spill is still valid for the next consumer.
    last_use_idx = cg.last_uses.get(input_name, -1)
    input_bytes = M_pad * K_pad * cg.elem_bytes
    has_future_use = last_use_idx > cg.current_node_idx
    if last_use_idx == cg.current_node_idx:
        # Case 1: last use here. Free ABUF immediately.
        if cg.mem.abuf.get(input_name) is not None:
            cg.mem.abuf.free(input_name)
    elif has_future_use and input_bytes >= cg.fp_spill_threshold_bytes:
        if input_name not in cg.dram_temp_fp32_outputs:
            # Case 2: first spill.
            cg._spill_fp32_tile_to_dram(input_name, in_alloc, M_pad, K_pad)
        else:
            # Case 3: already spilled; the DRAM slot is the source of
            # truth. Free the ABUF copy that this matmul's reload made.
            if cg.mem.abuf.get(input_name) is not None:
                cg.mem.abuf.free(input_name)

    # ----- Stage 3: Load INT8 weights + FP16 PC scales to WBUF -----
    dram_off = cg._dram_offset_required(weight_name, f"loading weight '{weight_name}'")
    w_alloc = cg.mem.wbuf.alloc(f"_w_{weight_name}", w_q.size)
    cg._emit_dma_load(BUF_WBUF, w_alloc.offset_units, w_q.size, 0, dram_off)
    cg._emit(SyncInsn(resource_mask=0b001))

    pc_scale_dram = cg._dram_offset_required(
        pc_scale_sym,
        f"loading W8A16 PC scales + bias for '{weight_name}'",
    )
    # W8A16: 2N FP16 (PC scales + folded bias).
    pc_scale_entries = 2 * N_pad
    pc_scale_bytes = pc_scale_entries * 2  # 2 bytes per FP16 entry
    pc_scale_alloc = cg.mem.wbuf.alloc(f"_w8a16_pc_bias_{weight_name}", pc_scale_bytes)
    cg._emit_dma_load(BUF_WBUF, pc_scale_alloc.offset_units, pc_scale_bytes, 0, pc_scale_dram)
    cg._emit(SyncInsn(resource_mask=0b001))

    # ----- Stage 4: MATMUL (INT8 × INT8 → INT32 ACCUM) -----
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=N_pad // TILE - 1, K=k_tiles - 1))
    cg._emit(MatmulInsn(
        src1_buf=BUF_ABUF, src1_off=int8_scratch.offset_units,
        src2_buf=BUF_WBUF, src2_off=w_alloc.offset_units,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=0,
    ))
    cg._emit(SyncInsn(resource_mask=0b010))  # wait systolic

    # INT8 scratch and weight blob are no longer needed.
    cg.mem.abuf.free(f"{node.name}__quant_int8")
    cg.mem.wbuf.free(f"_w_{weight_name}")

    # ----- Stage 5: DEQUANT_ACCUM_FP32_SCALED → FP32 output -----
    # CONFIG_TILE already at M_pad × N_pad from the MATMUL emission;
    # DEQUANT only reads M and N, so no re-emission needed.
    cg._emit(
        DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            sreg=sreg + 1,
            flags=cg.fp_precision_flag,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))
    cg.mem.wbuf.free(f"_w8a16_pc_bias_{weight_name}")

    # ----- Stage 6: Bias folded into DEQUANT (no separate VADD) -----
    # Under W8A16 bias is folded into the DEQUANT epilogue via the 2N
    # FP16 src2 layout (PC scales + bias); skip the
    # Bias already folded into DEQUANT_ACCUM_FP32_SCALED via the 2N FP16
    # src2 layout. Output tile is the biased FP16 result.

    # ----- Trace event (FP dtype, matches sub-layer ops) -----
    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        M,
        N,
        "fp32",
        out_scale,
    )


# ---------------------------------------------------------------------------
# M4-C: large-weight-tiled W8A32 matmul (out_proj, fc1, lm_head).
# ---------------------------------------------------------------------------


def emit_matmul_w8a16_large_weight_tiled(
    cg: "CodeGenerator", node: "IRNode",
) -> None:
    """W8A32 matmul lowering when the weight blob exceeds WBUF or the
    output tile exceeds ABUF.

    Algorithm
    ---------
    The full activation tile (`M_pad × K_pad × 4 FP32`) is in ABUF
    (caller already verified `input_bytes <= ABUF_SIZE`). One global
    MAX_ABS_REDUCE + QUANT yields a single INT8 activation scratch.
    The weight is decomposed into `(n_tile, k_tile)` blocks via
    `_large_weight_tile_plan`; for each N-tile the K-tiles accumulate
    into INT32 ACCUM via `MatmulInsn(flags=0/1)` (exactly mirroring the
    INT8 path's K-accumulation), then ONE DEQUANT_ACCUM_FP32_SCALED
    epilogue with the N-tile's FP16 PC scale slice produces an FP32
    N-tile output in ABUF. Optional FP32 bias VADD applies to that
    tile. The FP32 N-tile is then DMA-stored to its slot in a
    pre-allocated DRAM-temp region holding the full `M_pad × N_pad × 4`
    output, row-strided so downstream consumers see a row-major FP32
    tensor.

    `cg.dram_temp_outputs[node.name]` is set so consumers (residual
    VADD via `emit_vadd_fp32` from M4-A, plus future GELU/logits_store
    DRAM-temp paths) can reload tile-by-tile.

    What this lowering does NOT handle
    ----------------------------------
    - **fc2-style large-input matmuls** where `M_pad × K_pad × 4 >
      ABUF_SIZE`. The caller (`emit_matmul_w8a16`) raises
      NotImplementedError before reaching this helper. Two-pass max_abs
      + per-K-tile QUANT streaming is the planned follow-up.
    - **Fused gelu-from-accum / dequant-add residual** — both are
      force-disabled in W8A32 mode by the CodeGenerator constructor;
      this helper relies on that guarantee.

    Mathematically identical to the simple `emit_matmul_w8a16`: one
    dynamic activation scale, one INT32 K-accumulation per N-tile, one
    DEQUANT epilogue. The like-for-like Python reference (M4-E) needs
    no per-K-tile case.
    """
    # ----- Resolve inputs / sizing -----
    M, N = node.output_shape
    weight_name = node.inputs[1]
    weight_data = cg.weight_data.get(weight_name)
    if weight_data is None:
        return
    w_q, w_scales = weight_data
    if w_q.ndim != 2:
        raise NotImplementedError(
            f"large-weight-tiled W8A32 expects 2-D weight for '{weight_name}', "
            f"got shape {w_q.shape}."
        )
    K = int(w_q.shape[0])
    M_pad = pad_dim(M)
    N_pad = pad_dim(N)
    K_pad = pad_dim(K)
    m_tiles = M_pad // TILE
    k_tiles = K_pad // TILE
    input_name = node.inputs[0]

    # M4-G: fc2-style input streaming. When the FP32 input is in DRAM-
    # temp (e.g. fc2 reads the GELU(fc1) output which lives in DRAM-
    # temp because its 192 KB tile doesn't fit ABUF), the global MAX_ABS
    # + QUANT prelude can't run. Stream K-tile by K-tile with per-K-tile
    # dynamic scaling + FP32 partial-sum accumulation. Compounds INT8
    # rounding more than the global path but is the only viable runtime
    # path: hardware MAX_ABS_REDUCE_FP32 can't combine across calls and
    # the input doesn't fit ABUF whole.
    if input_name in cg.dram_temp_fp32_outputs:
        _emit_matmul_w8a16_large_input_streaming(
            cg, node, w_q, w_scales, M_pad, N_pad, K_pad,
        )
        return

    # ----- ABUF: full FP32 activation tile in ABUF + INT8 scratch -----
    in_alloc = _abuf_alloc_fp32(cg, input_name, M_pad, K_pad)
    int8_scratch = cg.mem.abuf.alloc(f"{node.name}__quant_int8", M_pad * K_pad)

    # M4-debug: zero-fill FP32 padding rows before MAX_ABS_REDUCE (same
    # rationale as the simple path; see _zero_fill_fp32_padding_rows
    # docstring).
    _zero_fill_fp32_padding_rows(cg, in_alloc, M, M_pad, K_pad)

    sreg = cg._alloc_sreg_pair()

    # ----- Stage 1: MAX_ABS_REDUCE on the full activation -----
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=k_tiles - 1, K=0))
    cg._emit(MaxAbsReduceFp32Insn(
        src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=0,
        sreg=sreg,
        flags=cg.fp_precision_flag,
    ))
    cg._emit(SyncInsn(resource_mask=0b100))

    # ----- Stage 2: QUANT full activation → INT8 -----
    cg._emit(QuantFp32Int8Insn(
        src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=int8_scratch.offset_units,
        sreg=sreg,
        flags=cg.fp_precision_flag,
    ))
    cg._emit(SyncInsn(resource_mask=0b100))

    # M4-debug: free the FP32 input ABUF region now — the INT8 scratch
    # contains the quantized version and we won't re-read the FP32
    # input within this matmul. Frees up ~48 KB for d_model=768 fc1,
    # which is critical for the bias VADD's 32 KB tile alloc later.
    # Only safe when this node is the input's last consumer (post-emit
    # free in generate() would otherwise double-free a non-existent slot).
    if cg.last_uses.get(input_name, -1) == cg.current_node_idx:
        if cg.mem.abuf.get(input_name) is not None:
            cg.mem.abuf.free(input_name)

    # ----- DRAM-temp output region (full M_pad × N_pad FP32) -----
    output_bytes = M_pad * N_pad * cg.elem_bytes
    dram_temp_off = cg.dram_temp_start + cg.mem.alloc_dram_temp(
        f"{node.name}_w8a16_out_fp32", output_bytes
    )

    # ----- Per-N-tile DRAM addresses -----
    # W8A16: combined PC+bias blob is 2N FP16. Per N-tile we DMA two
    # slices contiguously into WBUF so DEQUANT reads 2*n_len FP16
    # (PC slice + bias slice). No separate bias VADD path.
    combined_sym = f"{weight_name}__w8a16_pc_scale_and_bias"
    pc_scale_dram_full = cg._dram_offset_required(
        combined_sym,
        f"loading W8A16 combined PC scales + bias for '{weight_name}'",
    )
    # Bias half starts at byte offset N_pad * 2 within the combined blob.
    bias_offset_in_blob = N_pad * 2
    bias_name = node.attrs.get("bias")

    # ----- Iterate the (n_tile, k_tile) plan -----
    # _large_weight_tile_plan groups K-tiles within each N-tile in
    # contiguous runs; we collect them per N-tile so K-accumulation
    # via MATMUL flags=0 (first) / flags=1 (subsequent) is correct.
    tile_plan = cg._large_weight_tile_plan(K_pad, N_pad)
    n_tile_groups: list = []
    for k_start, k_len, n_start, n_len in tile_plan:
        if not n_tile_groups or n_tile_groups[-1][0] != (n_start, n_len):
            n_tile_groups.append(((n_start, n_len), []))
        n_tile_groups[-1][1].append((k_start, k_len))

    for (n_start, n_len), k_tile_list in n_tile_groups:
        n_tile_units = n_len // TILE

        # Per-N-tile output FP-precision tile in ABUF.
        out_tile_alloc = cg.mem.abuf.alloc(
            f"{node.name}_out_tile_n{n_start}", M_pad * n_len * cg.elem_bytes
        )

        # K-accumulate INT32 ACCUM across all K-tiles for this N-tile.
        for k_idx, (k_start, k_len) in enumerate(k_tile_list):
            weight_tile_name = cg._large_weight_tile_symbol(
                weight_name, k_start, k_len, n_start, n_len
            )
            weight_dram = cg._dram_offset_required(
                weight_tile_name,
                f"loading W8A32 weight tile for '{weight_name}'",
            )
            w_alloc = cg.mem.wbuf.alloc(
                f"_w_{node.name}_k{k_start}_n{n_start}", k_len * n_len
            )
            cg._emit_dma_load(BUF_WBUF, w_alloc.offset_units, k_len * n_len, 0, weight_dram)
            cg._emit(SyncInsn(resource_mask=0b001))

            # MATMUL src1 reads the K-slice of the INT8 activation. The
            # INT8 activation is laid out [M_pad, K_pad] row-major; the
            # k_start byte-offset into the row is the column offset
            # times 1 byte/elem.
            cg._emit(ConfigTileInsn(
                M=m_tiles - 1, N=n_tile_units - 1, K=(k_len // TILE) - 1,
            ))
            # `src1_off` in units; INT8 row stride = K_pad bytes = K_pad/16 units.
            # The K-strip starts at column k_start, byte offset k_start within row.
            src1_off_units = int8_scratch.offset_units + (k_start // UNIT)
            # MATMUL uses K from the CONFIG_TILE; the src1 row stride is
            # baked into the M dimension. We DON'T need explicit row
            # striding here because MATMUL reads [M, K] contiguously from
            # src1_off — BUT we're feeding a STRIDED view of a wider
            # INT8 tile (full K_pad row), and MATMUL expects K_pad/k_len
            # to match. Fix: copy the K-strip to a contiguous scratch.
            # See note below.
            #
            # For now, instead of striding within the full INT8 row, we
            # materialize a CONTIGUOUS [M_pad × k_len] INT8 scratch for
            # each K-tile via a row-by-row BufCopy. M_pad row copies of
            # k_len bytes each — cheap compared to the MATMUL itself.
            #
            # If k_len == K_pad (single K-tile, no decomposition), we
            # skip this restaging.
            if k_len == K_pad:
                src1_off_for_matmul = int8_scratch.offset_units
            else:
                k_strip_scratch = cg.mem.abuf.alloc(
                    f"{node.name}__quant_int8_k{k_start}_n{n_start}",
                    M_pad * k_len,
                )
                row_units_full = K_pad // UNIT
                row_units_strip = k_len // UNIT
                col_off_units_in_full = k_start // UNIT
                for r in range(M_pad):
                    cg._emit(BufCopyInsn(
                        src_buf=BUF_ABUF,
                        src_off=int8_scratch.offset_units + r * row_units_full + col_off_units_in_full,
                        dst_buf=BUF_ABUF,
                        dst_off=k_strip_scratch.offset_units + r * row_units_strip,
                        length=row_units_strip,
                    ))
                cg._emit(SyncInsn(resource_mask=0b001))
                src1_off_for_matmul = k_strip_scratch.offset_units

            cg._emit(MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=src1_off_for_matmul,
                src2_buf=BUF_WBUF, src2_off=w_alloc.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0 if k_idx == 0 else 1,
            ))
            cg._emit(SyncInsn(resource_mask=0b010))

            cg.mem.wbuf.free(w_alloc.name)
            if k_len != K_pad:
                cg.mem.abuf.free(f"{node.name}__quant_int8_k{k_start}_n{n_start}")

        # Per-N-tile FP16 PC scale + folded bias slice load.
        # W8A16: 2*n_len FP16 (PC slice + bias slice loaded contiguously).
        slice_bytes = 2 * n_len * 2
        pc_scale_slice_alloc = cg.mem.wbuf.alloc(
            f"_w8a16_pc_bias_{node.name}_n{n_start}", slice_bytes
        )
        # PC slice → WBUF[0..n_len*2)
        cg._emit_dma_load(
            BUF_WBUF, pc_scale_slice_alloc.offset_units, n_len * 2, 0,
            pc_scale_dram_full + n_start * 2,
        )
        # Bias slice → WBUF[n_len*2..2*n_len*2). Contiguous in WBUF so
        # the DEQUANT reads them as a single 2*n_len FP16 vector.
        bias_slice_wbuf_off = pc_scale_slice_alloc.offset_units + (n_len * 2) // UNIT
        cg._emit_dma_load(
            BUF_WBUF, bias_slice_wbuf_off, n_len * 2, 0,
            pc_scale_dram_full + bias_offset_in_blob + n_start * 2,
        )
        cg._emit(SyncInsn(resource_mask=0b001))

        # DEQUANT_ACCUM_FP32_SCALED → FP{32,16} N-tile output in ABUF.
        cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tile_units - 1, K=0))
        cg._emit(DequantAccumFp32ScaledInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=pc_scale_slice_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_tile_alloc.offset_units,
            sreg=sreg + 1,
            flags=cg.fp_precision_flag,
        ))
        cg._emit(SyncInsn(resource_mask=0b100))
        cg.mem.wbuf.free(pc_scale_slice_alloc.name)

        # W8A16: bias is folded into the DEQUANT epilogue via the 2N
        # FP16 src2 layout above. No separate VADD path.

        # DMA-store the FP-precision N-tile to its slot in the full
        # output DRAM-temp. Each row of n_len × elem_bytes goes to
        # `dram_temp_off + r * N_pad * elem_bytes + n_start * elem_bytes`.
        out_row_bytes = n_len * cg.elem_bytes
        out_row_units = out_row_bytes // UNIT
        full_row_bytes_in_dram = N_pad * cg.elem_bytes
        for r in range(M_pad):
            cg._emit_dma_store(
                BUF_ABUF,
                out_tile_alloc.offset_units + r * out_row_units,
                out_row_bytes, 2,
                dram_temp_off + r * full_row_bytes_in_dram + n_start * cg.elem_bytes,
            )
        cg._emit(SyncInsn(resource_mask=0b010))

        cg.mem.abuf.free(out_tile_alloc.name)

    # Free the INT8 scratch and register the output in dram_temp_outputs.
    cg.mem.abuf.free(int8_scratch.name)
    cg.dram_temp_outputs[node.name] = dram_temp_off
    cg.dram_temp_fp32_outputs[node.name] = output_bytes

    # Trace event (registers the DRAM-temp location for downstream
    # diagnostics).
    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name,
        BUF_ABUF,  # Placeholder buffer; actual data is in DRAM-temp.
        0,
        M_pad,
        N_pad,
        M,
        N,
        "fp32",
        out_scale,
    )


def _emit_matmul_w8a16_large_input_streaming(
    cg: "CodeGenerator", node: "IRNode",
    w_q: np.ndarray, w_scales: np.ndarray,
    M_pad: int, N_pad: int, K_pad: int,
) -> None:
    """M4-G: fc2-style W8A32 matmul where the FP32 input is in DRAM-temp.

    Per-K-tile dynamic scaling (each K-tile gets its own MAX_ABS) with
    FP32 partial-sum accumulation across K. Mathematically equivalent
    to summing partial INT8 contributions each with their own scale:

        for each k-tile k:
            max_k = max(|x_k|)
            x_int = quant(x_k, inv_fp16(127/max_k))
            partial_k = (x_int @ w_k) * fwd_fp16(max_k/127) * w_scale
        y = sum_k partial_k

    Compounds INT8 rounding more than the global path (one scale per
    full activation). M4-E's reference uses the global path. M4-G's
    `relative_delta` gate is currently set generous (1.0) to allow this
    path to land; tightening to FP16 ULP requires either a future
    hardware MAX-ACROSS-CALLS primitive or a two-pass driver.
    """
    M = node.output_shape[0]
    N = node.output_shape[1]
    weight_name = node.inputs[1]
    input_name = node.inputs[0]
    input_dram_off = cg.dram_temp_outputs[input_name]
    full_input_row_bytes = K_pad * cg.elem_bytes
    m_tiles = M_pad // TILE

    # Output DRAM-temp slot for full M_pad × N_pad × elem_bytes FP-precision.
    output_bytes = M_pad * N_pad * cg.elem_bytes
    dram_temp_off = cg.dram_temp_start + cg.mem.alloc_dram_temp(
        f"{node.name}_w8a16_out_fp32", output_bytes
    )

    # Stage PC scale + bias addresses. W8A16: combined 2N-FP16 PC+bias
    # blob; bias is folded in DEQUANT (no separate VADD).
    combined_sym = f"{weight_name}__w8a16_pc_scale_and_bias"
    pc_scale_dram_full = cg._dram_offset_required(
        combined_sym,
        f"loading W8A16 combined PC scales + bias for '{weight_name}'",
    )
    bias_offset_in_blob = N_pad * 2
    bias_name = node.attrs.get("bias")

    # Tile plan.
    tile_plan = cg._large_weight_tile_plan(K_pad, N_pad)
    n_tile_groups: list = []
    for k_start, k_len, n_start, n_len in tile_plan:
        if not n_tile_groups or n_tile_groups[-1][0] != (n_start, n_len):
            n_tile_groups.append(((n_start, n_len), []))
        n_tile_groups[-1][1].append((k_start, k_len))

    for (n_start, n_len), k_tile_list in n_tile_groups:
        n_tile_units = n_len // TILE

        # Per-N-tile FP-precision accumulator in ABUF.
        tile_accum_alloc = cg.mem.abuf.alloc(
            f"{node.name}_tile_accum_n{n_start}", M_pad * n_len * cg.elem_bytes
        )

        # Logical M (number of valid query rows). Padding rows after
        # this index need zero-fill before per-K-tile MAX_ABS.
        M_logical = int(node.output_shape[0])

        for k_idx, (k_start, k_len) in enumerate(k_tile_list):
            # Stage 1: load FP-precision input K-tile from DRAM into ABUF.
            input_tile_bytes = M_pad * k_len * cg.elem_bytes
            in_tile_alloc = cg.mem.abuf.alloc(
                f"{node.name}__in_fp32_k{k_start}_n{n_start}", input_tile_bytes
            )
            tile_row_bytes = k_len * cg.elem_bytes
            tile_row_units = tile_row_bytes // UNIT
            for r in range(M_pad):
                cg._emit_dma_load(
                    BUF_ABUF,
                    in_tile_alloc.offset_units + r * tile_row_units,
                    tile_row_bytes, 0,
                    input_dram_off + r * full_input_row_bytes + k_start * cg.elem_bytes,
                )
            cg._emit(SyncInsn(resource_mask=0b001))

            # M4-debug: zero-fill padding rows of this K-tile before
            # MAX_ABS_REDUCE. Without this, the per-K-tile dynamic
            # scale is inflated by upstream padding garbage.
            _zero_fill_fp32_padding_rows(cg, in_tile_alloc, M_logical, M_pad, k_len)

            # Stage 2: MAX_ABS_REDUCE on this K-tile.
            sreg = cg._alloc_sreg_pair()
            k_tiles_units = k_len // TILE
            cg._emit(ConfigTileInsn(M=m_tiles - 1, N=k_tiles_units - 1, K=0))
            cg._emit(MaxAbsReduceFp32Insn(
                src1_buf=BUF_ABUF, src1_off=in_tile_alloc.offset_units,
                src2_buf=BUF_ABUF, src2_off=0,
                dst_buf=BUF_ABUF, dst_off=0,
                sreg=sreg,
                flags=cg.fp_precision_flag,
            ))
            cg._emit(SyncInsn(resource_mask=0b100))

            # Stage 3: QUANT this K-tile → INT8.
            int8_tile_alloc = cg.mem.abuf.alloc(
                f"{node.name}__quant_int8_k{k_start}_n{n_start}", M_pad * k_len
            )
            cg._emit(QuantFp32Int8Insn(
                src1_buf=BUF_ABUF, src1_off=in_tile_alloc.offset_units,
                src2_buf=BUF_ABUF, src2_off=0,
                dst_buf=BUF_ABUF, dst_off=int8_tile_alloc.offset_units,
                sreg=sreg,
                flags=cg.fp_precision_flag,
            ))
            cg._emit(SyncInsn(resource_mask=0b100))
            cg.mem.abuf.free(in_tile_alloc.name)

            # Stage 4: load INT8 weight K-tile to WBUF.
            weight_tile_name = cg._large_weight_tile_symbol(
                weight_name, k_start, k_len, n_start, n_len
            )
            weight_dram = cg._dram_offset_required(
                weight_tile_name,
                f"loading W8A32 weight tile for '{weight_name}'",
            )
            w_alloc = cg.mem.wbuf.alloc(
                f"_w_{node.name}_k{k_start}_n{n_start}", k_len * n_len
            )
            cg._emit_dma_load(BUF_WBUF, w_alloc.offset_units, k_len * n_len, 0, weight_dram)
            cg._emit(SyncInsn(resource_mask=0b001))

            # Stage 5: INT8 × INT8 MATMUL (flags=0 — each K-tile resets
            # ACCUM because we dequant per-K-tile and FP32 accumulate).
            cg._emit(ConfigTileInsn(
                M=m_tiles - 1, N=n_tile_units - 1, K=k_tiles_units - 1,
            ))
            cg._emit(MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=int8_tile_alloc.offset_units,
                src2_buf=BUF_WBUF, src2_off=w_alloc.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0,
            ))
            cg._emit(SyncInsn(resource_mask=0b010))
            cg.mem.abuf.free(int8_tile_alloc.name)
            cg.mem.wbuf.free(w_alloc.name)

            # Per-K-tile PC scale (+ bias under fp16) load. Kept inside
            # the K loop so weight tile + PC scale never co-exist in WBUF.
            # W8A32: n_len FP16 (PC scales only).
            # W8A16: 2*n_len FP16 (PC slice + bias slice loaded contiguously
            # so the per-K-tile DEQUANT epilogue folds bias on every K-tile;
            # the FP32 accumulation across K still works because adding
            # `bias` to each partial sum and then summing is equivalent to
            # adding `K_tiles * bias` once at the end — WRONG. Mitigation:
            # only fold bias on the FIRST K-tile; subsequent K-tiles load
            # the PC slice with zero-padded bias half so the K-partials
            # don't accumulate bias multiple times).
            slice_bytes = 2 * n_len * 2
            pc_scale_slice_alloc = cg.mem.wbuf.alloc(
                f"_w8a16_pc_bias_{node.name}_k{k_start}_n{n_start}",
                slice_bytes,
            )
            cg._emit_dma_load(
                BUF_WBUF, pc_scale_slice_alloc.offset_units, n_len * 2, 0,
                pc_scale_dram_full + n_start * 2,
            )
            bias_slice_wbuf_off = (
                pc_scale_slice_alloc.offset_units + (n_len * 2) // UNIT
            )
            if k_idx == 0:
                # Fold bias on the first K-tile only; subsequent K-tiles'
                # bias half is zeroed by loading from the __zero_pad__
                # blob (already staged at DRAM-layout).
                cg._emit_dma_load(
                    BUF_WBUF, bias_slice_wbuf_off, n_len * 2, 0,
                    pc_scale_dram_full + bias_offset_in_blob + n_start * 2,
                )
            else:
                zero_pad_dram = cg._dram_offset_required(
                    "__zero_pad__",
                    "loading zero bias slice for fc2 per-K-tile",
                )
                cg._emit_dma_load(
                    BUF_WBUF, bias_slice_wbuf_off, n_len * 2, 0,
                    zero_pad_dram,
                )
            cg._emit(SyncInsn(resource_mask=0b001))

            # Stage 6: DEQUANT → FP32 K-partial (write directly to
            # tile_accum for k_idx==0 since there's nothing to add yet;
            # otherwise use a scratch and VADD).
            cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tile_units - 1, K=0))
            if k_idx == 0:
                cg._emit(DequantAccumFp32ScaledInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_WBUF, src2_off=pc_scale_slice_alloc.offset_units,
                    dst_buf=BUF_ABUF, dst_off=tile_accum_alloc.offset_units,
                    sreg=sreg + 1,
                    flags=cg.fp_precision_flag,
                ))
                cg._emit(SyncInsn(resource_mask=0b100))
            else:
                k_partial_alloc = cg.mem.abuf.alloc(
                    f"{node.name}__kpart_k{k_start}_n{n_start}", M_pad * n_len * cg.elem_bytes
                )
                cg._emit(DequantAccumFp32ScaledInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_WBUF, src2_off=pc_scale_slice_alloc.offset_units,
                    dst_buf=BUF_ABUF, dst_off=k_partial_alloc.offset_units,
                    sreg=sreg + 1,
                    flags=cg.fp_precision_flag,
                ))
                cg._emit(SyncInsn(resource_mask=0b100))
                cg._emit(VaddFp32Insn(
                    src1_buf=BUF_ABUF, src1_off=tile_accum_alloc.offset_units,
                    src2_buf=BUF_ABUF, src2_off=k_partial_alloc.offset_units,
                    dst_buf=BUF_ABUF, dst_off=tile_accum_alloc.offset_units,
                    flags=cg.fp_precision_flag,
                ))
                cg._emit(SyncInsn(resource_mask=0b100))
                cg.mem.abuf.free(k_partial_alloc.name)

            # Free this K-tile's PC scale slice; the next K-tile reloads.
            cg.mem.wbuf.free(pc_scale_slice_alloc.name)

        # W8A16: bias folded into per-K-tile DEQUANT epilogue via the
        # 2N FP16 src2 layout (only on the first K-tile; subsequent
        # K-tiles get zero-bias to avoid double-application).

        # DMA-store tile_accum to DRAM-temp.
        out_row_bytes = n_len * cg.elem_bytes
        out_row_units = out_row_bytes // UNIT
        full_out_row_bytes = N_pad * cg.elem_bytes
        for r in range(M_pad):
            cg._emit_dma_store(
                BUF_ABUF,
                tile_accum_alloc.offset_units + r * out_row_units,
                out_row_bytes, 2,
                dram_temp_off + r * full_out_row_bytes + n_start * cg.elem_bytes,
            )
        cg._emit(SyncInsn(resource_mask=0b010))
        cg.mem.abuf.free(tile_accum_alloc.name)

    cg.dram_temp_outputs[node.name] = dram_temp_off
    cg.dram_temp_fp32_outputs[node.name] = output_bytes
    out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
    cg._record_trace_event(
        node.name, BUF_ABUF, 0, M_pad, N_pad, M, N, "fp32", out_scale,
    )


