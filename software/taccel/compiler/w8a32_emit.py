"""W8A32 codegen helpers (Phase 3 (c.1), milestones M2 + M2.5-B).

This module contains the per-op lowering functions for the W8A32 path.
The main `CodeGenerator` (`codegen.py`) dispatches to these helpers at
each sub-layer emission site when its `w8a32_enabled` flag is True.

Scope (M2 + M2.5-B)
-------------------

  - LAYERNORM_FP32  (M2):       FP32-I/O drop-in for INT8 LAYERNORM
  - GELU_FP32       (M2):       FP32-I/O drop-in for INT8 GELU
  - SOFTMAX_FP32 / MASKED_SOFTMAX_FP32 (M2): drop-in for INT8 softmax
  - VADD_FP32       (M2):       FP32-I/O drop-in for INT8 VADD residual
  - emit_matmul_w8a32 (M2.5-B): full prelude+epilogue lowering of
    `matmul` IR nodes using the M2.5-A dynamic activation-scale
    primitives. Sequence:
      FP32 input → MAX_ABS_REDUCE_FP32 → S[s] (127/max), S[s+1] (max/127)
                → QUANT_FP32_INT8 (S[s])  → INT8 scratch in ABUF
                → MATMUL (INT8 × INT8)    → INT32 ACCUM
                → DEQUANT_ACCUM_FP32_SCALED (× FP16 PC weight scales × S[s+1])
                → FP32 output in ABUF
                → [optional FP32 bias VADD_FP32]

Out of scope (M3)
-----------------

  - `matmul_qkt` / `matmul_attn_v` (attention internals).
  - Strip-mined / large-weight-tiled / fused-out-proj-accum matmul
    variants. emit_matmul_w8a32 raises NotImplementedError when any of
    the sizing thresholds force a non-simple lowering.
  - Fused gelu-from-accum and dequant-add residual epilogues.
  - Embedding → FP32 boundary handling (the IR fragments tested in
    M2.5-B start from an FP32 input already in ABUF).

Design contract from M1 + M2.5-A (do not break)
-----------------------------------------------

  - FP32 ABUF tiles use 4 bytes per element. A 16x16 FP32 tile occupies
    1024 bytes = 64 16-byte addressing units (vs 16 units for INT8).
    Callers must size their `mem.abuf.alloc(..., size_bytes=M_pad * N_pad * 4)`
    accordingly — the size is in bytes, not elements.
  - FP16 stays the WBUF convention for LN gamma+beta and per-channel
    weight scales (2 bytes/element).
  - GELU is `gelu_new` (tanh approximation) by hardware contract.
  - MAX_ABS_REDUCE_FP32 consumes a `(M, N)` FP32 tile and writes the
    paired scales `S[sreg], S[sreg+1]`; sreg must be ≤ 14.
  - DEQUANT_ACCUM_FP32_SCALED reads INT32 ACCUM × FP16 per-channel
    weight scale × FP32(S[sreg]) → FP32 ABUF.

Re-entrancy contract
--------------------

These helpers must **not** call back into the patched dispatch methods
on `CodeGenerator` (`_emit_layernorm`, `_emit_gelu`, `_emit_softmax`,
`_emit_vadd`, `_emit_matmul`). Those methods are the W8A32 entry points;
calling them from inside a helper would infinitely recurse into the
W8A32 branch. Use `cg._emit(...)`, `cg.mem.*`, and the low-level helpers
(`cg._emit_dma_load`, `cg._record_trace_event`, etc.) instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..isa.instructions import (
    ConfigTileInsn,
    DequantAccumFp32ScaledInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    MatmulInsn,
    MaxAbsReduceFp32Insn,
    QuantFp32Int8Insn,
    SoftmaxFp32Insn,
    SyncInsn,
    VaddFp32Insn,
)
from ..isa.opcodes import ABUF_SIZE, ACCUM_SIZE, BUF_ABUF, BUF_ACCUM, BUF_WBUF, WBUF_SIZE
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


# ---------------------------------------------------------------------------
# M2.5-B: full matmul lowering using the M2.5-A dynamic-scale primitives.
# ---------------------------------------------------------------------------


def emit_matmul_w8a32(cg: "CodeGenerator", node: "IRNode") -> None:
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
      - `fp32_biases` must contain an entry for any `node.attrs['bias']`
        referenced; the codegen stages it to DRAM as `f"{bias_name}__fp32"`.
      - Fused gelu-from-accum and dequant-add residual epilogues are
        force-disabled at CodeGenerator init; this helper raises if they
        somehow re-surface.
    """
    if node.op != "matmul":
        raise NotImplementedError(
            f"emit_matmul_w8a32 only handles op='matmul' (got {node.op!r}); "
            "matmul_qkt and matmul_attn_v are M3."
        )
    if cg._dequant_add_enabled_for_output(node.name):
        raise NotImplementedError(
            f"W8A32 matmul lowering does not support fused DEQUANT_ADD "
            f"residual epilogue (node '{node.name}'). The dynamic "
            "activation scale in M2.5-A makes the prescaled-INT32 skip "
            "path incompatible."
        )
    if node.attrs.get("gelu_from_accum") or node.attrs.get("stage4_weight_tiled"):
        raise NotImplementedError(
            f"W8A32 matmul lowering does not support fused gelu-from-accum "
            f"or Stage 4 weight tiling (node '{node.name}')."
        )

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

    # Sizing guardrail — mirror the same thresholds used by _emit_matmul
    # to decide between simple/strip-mined/large-weight-tiled. If we'd
    # need a non-simple lowering, raise rather than silently producing a
    # bad bundle.
    output_bytes = M_pad * N_pad * FP32_BYTES_PER_ELEM
    accum_bytes = M_pad * N_pad * 4  # INT32
    weight_bytes = int(w_q.size)
    if output_bytes > ABUF_SIZE or accum_bytes > ACCUM_SIZE or weight_bytes > WBUF_SIZE:
        raise NotImplementedError(
            f"W8A32 matmul '{node.name}' exceeds the simple-matmul "
            f"sizing thresholds (output {output_bytes}B vs ABUF "
            f"{ABUF_SIZE}, accum {accum_bytes}B vs ACCUM {ACCUM_SIZE}, "
            f"weight {weight_bytes}B vs WBUF {WBUF_SIZE}); strip-mined "
            "and large-weight-tiled W8A32 lowerings are M3."
        )

    # Per-channel FP16 weight-scale vector staging symbol — set up by
    # `CodeGenerator.generate()` at DRAM-staging time when w8a32_enabled.
    pc_scale_sym = f"{weight_name}__w8a32_pc_scale"

    # ----- ABUF allocations -----
    # FP32 input tile (M_pad × K_pad × 4 bytes). The prior op's output
    # FP32 alloc is reused here under the input's name.
    in_alloc = _abuf_alloc_fp32(cg, node.inputs[0], M_pad, K_pad)
    # FP32 output tile (M_pad × N_pad × 4 bytes).
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    # INT8 scratch tile (M_pad × K_pad × 1 byte) for the QUANT output.
    # Lives in ABUF, freed immediately after the MATMUL consumes it.
    int8_scratch = cg.mem.abuf.alloc(f"{node.name}__quant_int8", M_pad * K_pad)

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
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))  # wait SFU

    # ----- Stage 2: QUANT_FP32_INT8 (FP32 ABUF → INT8 ABUF) -----
    cg._emit(
        QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=int8_scratch.offset_units,
            sreg=sreg,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # ----- Stage 3: Load INT8 weights + FP16 PC scales to WBUF -----
    dram_off = cg._dram_offset_required(weight_name, f"loading weight '{weight_name}'")
    w_alloc = cg.mem.wbuf.alloc(f"_w_{weight_name}", w_q.size)
    cg._emit_dma_load(BUF_WBUF, w_alloc.offset_units, w_q.size, 0, dram_off)
    cg._emit(SyncInsn(resource_mask=0b001))

    pc_scale_dram = cg._dram_offset_required(
        pc_scale_sym,
        f"loading W8A32 per-channel weight scales for '{weight_name}'",
    )
    pc_scale_bytes = N_pad * 2  # FP16
    pc_scale_alloc = cg.mem.wbuf.alloc(f"_w8a32_pc_{weight_name}", pc_scale_bytes)
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
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))
    cg.mem.wbuf.free(f"_w8a32_pc_{weight_name}")

    # ----- Stage 6: Optional FP32 bias VADD -----
    # VADD_FP32 does not broadcast — it reads M_pad × N_pad from both
    # sources. We replicate the bias's (1, N_pad) vector to (M_pad, N_pad)
    # in ABUF via M_pad DMA loads, each pulling the same N_pad × 4 bytes
    # from DRAM into a distinct row of the bias scratch tile.
    bias_name = node.attrs.get("bias")
    if bias_name is not None:
        if bias_name not in cg.fp32_biases:
            raise KeyError(
                f"W8A32 matmul '{node.name}' references bias '{bias_name}' "
                "but no FP32 bias was staged. Pass `fp32_biases={"
                f"'{bias_name}': <fp32 ndarray>" "}` to CodeGenerator."
            )
        bias_sym = f"{bias_name}__fp32"
        bias_dram = cg._dram_offset_required(
            bias_sym, f"loading W8A32 FP32 bias for '{node.name}'"
        )
        bias_alloc = _abuf_alloc_fp32(cg, f"{node.name}__bias_fp32", M_pad, N_pad)
        # row_units: how far in 16-byte units to advance per row.
        row_units = (N_pad * FP32_BYTES_PER_ELEM) // 16
        for row in range(M_pad):
            cg._emit_dma_load(
                BUF_ABUF,
                bias_alloc.offset_units + row * row_units,
                N_pad * FP32_BYTES_PER_ELEM,
                0,
                bias_dram,
            )
        cg._emit(SyncInsn(resource_mask=0b001))

        # VADD_FP32 in-place onto the output tile (src1 == dst is fine).
        cg._emit(ConfigTileInsn(M=m_tiles - 1, N=N_pad // TILE - 1, K=0))
        cg._emit(
            VaddFp32Insn(
                src1_buf=BUF_ABUF, src1_off=out_alloc.offset_units,
                src2_buf=BUF_ABUF, src2_off=bias_alloc.offset_units,
                dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            )
        )
        cg._emit(SyncInsn(resource_mask=0b100))
        cg.mem.abuf.free(f"{node.name}__bias_fp32")

    # ----- Trace event (FP32 dtype, matches sub-layer ops) -----
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
