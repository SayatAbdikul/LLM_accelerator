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
    BufCopyInsn,
    ConfigAttnInsn,
    ConfigTileInsn,
    DequantAccumFp32Insn,
    DequantAccumFp32ScaledInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    MatmulInsn,
    MaxAbsReduceFp32Insn,
    QuantFp32Int8Insn,
    SetScaleInsn,
    SoftmaxFp32Insn,
    SyncInsn,
    VaddFp32Insn,
)
from ..isa.opcodes import ABUF_SIZE, ACCUM_SIZE, BUF_ABUF, BUF_ACCUM, BUF_WBUF, WBUF_SIZE
from .tiler import TILE, pad_dim


def _fp16_to_uint16(val) -> int:
    """Convert FP32/python float to its FP16 bit pattern (uint16, little-endian).

    Mirror of `codegen._fp16_to_uint16` — duplicated to keep w8a32_emit a
    self-contained import surface (codegen imports w8a32_emit, not the
    other way around).
    """
    fp16 = np.float16(val)
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


# Size of one 16-byte addressing unit (matches `taccel/isa/opcodes.UNIT`).
UNIT = 16

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

    in_name = node.inputs[0]
    # M4-A: if this LN input was previously spilled to DRAM-temp (by an
    # earlier LN at a block boundary), reload it via the FP32 helper so
    # the LN reads it directly from ABUF.
    if in_name in cg.dram_temp_fp32_outputs and cg.mem.abuf.get(in_name) is None:
        in_alloc = cg._load_dram_to_abuf_fp32(in_name, M_pad, N_pad)
    else:
        in_alloc = _abuf_alloc_fp32(cg, in_name, M_pad, N_pad)
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

    # M4-A: spill the LN input (residual) to DRAM-temp if it has future
    # uses and is big enough to matter. This frees ABUF for the per-head
    # attention sub-block. The consumer (next residual VADD) reloads it
    # via `_load_dram_to_abuf_fp32`. Skip for tiny fixtures (≤ 16 KB
    # tiles) so existing tests' instruction counts stay unchanged.
    tile_bytes = M_pad * N_pad * FP32_BYTES_PER_ELEM
    last_use_idx = cg.last_uses.get(in_name, -1)
    has_future_use = last_use_idx > cg.current_node_idx
    if (
        has_future_use
        and tile_bytes >= cg.fp32_spill_threshold_bytes
        and in_name not in cg.dram_temp_fp32_outputs
    ):
        cg._spill_fp32_tile_to_dram(in_name, in_alloc, M_pad, N_pad)


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

    Non-masked variant: emits CONFIG_TILE + SOFTMAX_FP32.

    Masked variant (M3-C): emits CONFIG_TILE + ConfigAttnInsn +
    MASKED_SOFTMAX_FP32. The CONFIG_ATTN context is set ONCE with
    `query_row_base=0` because in W8A32 mode the softmax runs over
    the FULL M_pad × N_pad attention scores tile in a single
    instruction — NOT per-strip like the INT8 `_emit_qkt` path.
    `valid_kv_len` comes from `node.attrs["key_len"]` (M3-C IR contract
    addition); `mode = 0b10` for the standard prefill-causal case
    (key_pad == key_len), `mode = 0b11` for runtime-config-attn graphs.

    The `masked` keyword (legacy from M2's INT8-mode dispatch) stays
    as a convenience. When `node.attrs.get("masked")` is set, this
    helper picks up the IR-level value automatically; the keyword arg
    is OR'd with it. The `key_len` defaults to `node.output_shape[1]`
    when not explicitly set (matches the IR contract from
    `nanogpt_adapter.py`).
    """
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    # Pick up the IR-level `masked` attr; OR with the keyword for
    # direct-caller convenience (the M2 dispatch in `_emit_softmax`
    # passes `masked=True` explicitly via the keyword).
    ir_masked = bool(node.attrs.get("masked", False))
    is_masked = bool(masked) or ir_masked

    if is_masked:
        # CONFIG_ATTN context for the masked-softmax. In W8A32 mode the
        # softmax processes all M_pad query rows in one go, so
        # query_row_base=0 always — contrast the INT8 path's per-strip
        # CONFIG_ATTN emission inside _emit_qkt.
        key_len = int(node.attrs.get("key_len", node.output_shape[1]))
        runtime_config_attn = bool(node.attrs.get("runtime_config_attn", False))
        # Mode picking mirrors codegen._attention_mask_mode_for_qkt's
        # logic from the softmax's perspective: prefill-causal when
        # N_pad == key_len (no key padding); runtime-config-attn for
        # decode streams that patch the context at runtime.
        if runtime_config_attn:
            # Decode-stream graphs that patch the context at runtime.
            attn_mode = 0b11
        elif N_pad == key_len:
            # Standard prefill-causal: no key padding.
            attn_mode = 0b10
        else:
            # Static-but-padded key_len (N_pad > key_len): mode 0b11
            # with a static `valid_kv_len` matches the INT8 path's
            # convention (see codegen._attention_mask_mode_for_qkt).
            # Not "runtime-patched" in spite of the mode bits; the
            # simulator's masked-softmax handles static valid_kv_len
            # correctly under either 0b10 or 0b11 — 0b11 just signals
            # that key_pad > valid_kv_len.
            attn_mode = 0b11
        cg._emit(
            ConfigAttnInsn(
                query_row_base=0,
                valid_kv_len=key_len,
                mode=attn_mode,
            )
        )

    in_alloc = _abuf_alloc_fp32(cg, node.inputs[0], M_pad, N_pad)
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    cls = MaskedSoftmaxFp32Insn if is_masked else SoftmaxFp32Insn
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

    M4-A
    ----
    Two ABUF residency optimizations:

    1. **Reload from FP32 DRAM-temp.** If either input was spilled by
       an earlier `emit_layernorm_fp32` (block boundary residual), DMA-
       load it back into ABUF first. The reload uses the FP32 stride
       (4 bytes/elem) — `_load_dram_to_abuf_fp32` does this.

    2. **In-place output aliasing.** Write the FP32 sum into one of the
       input slots (either is safe since both inputs' last use is this
       node by the residual-VADD IR contract). The simulator's
       `_exec_vadd_fp32` reads both sources fully before writing dst,
       so `dst_off == src{1,2}_off` is well-defined.

       Selection rule: prefer the input whose last use is at this node
       AND was reloaded from DRAM (since that slot is freshly allocated
       and dead after this op). Fall back to inputs[1] (matches the INT8
       `_emit_vadd` convention — see codegen.py:2779). This keeps the
       INT8 and W8A32 paths' ABUF layouts isomorphic for downstream ops.

    Without (1) the 48 KB residual reload would silently miss the spilled
    DRAM bytes; without (2) the peak ABUF would still exceed 128 KB at
    `block0_residual1` (tok_pos_add 48 KB + block0_attn_out 48 KB + new
    block0_residual1 48 KB = 144 KB).
    """
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE

    in0_name = node.inputs[0]
    in1_name = node.inputs[1]
    # Reload spilled FP32 inputs first. CONFIG_TILE goes AFTER the loads
    # so the simulator's DMA path runs without an active tile context
    # (matches the INT8 VADD pattern at codegen.py:2776-2785).
    in0_reloaded = False
    in1_reloaded = False
    if in0_name in cg.dram_temp_fp32_outputs and cg.mem.abuf.get(in0_name) is None:
        cg._load_dram_to_abuf_fp32(in0_name, M_pad, N_pad)
        in0_reloaded = True
    if in1_name in cg.dram_temp_fp32_outputs and cg.mem.abuf.get(in1_name) is None:
        cg._load_dram_to_abuf_fp32(in1_name, M_pad, N_pad)
        in1_reloaded = True

    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    src1_alloc = _abuf_alloc_fp32(cg, in0_name, M_pad, N_pad)
    src2_alloc = _abuf_alloc_fp32(cg, in1_name, M_pad, N_pad)

    # In-place aliasing: rewrite the result into one of the inputs' slots.
    # Pick inputs[0] when it was just reloaded from DRAM (its slot is
    # freshly allocated and dead after this VADD), else fall back to
    # inputs[1] matching the INT8 convention.
    alias_input_name = in0_name if in0_reloaded else in1_name
    alias_input_alloc = src1_alloc if in0_reloaded else src2_alloc

    cg._emit(
        VaddFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=src1_alloc.offset_units,
            src2_buf=BUF_ABUF,
            src2_off=src2_alloc.offset_units,
            dst_buf=BUF_ABUF,
            dst_off=alias_input_alloc.offset_units,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # Rename the aliased input's ABUF allocation to the output node name.
    # After this, `cg.mem.abuf.get(node.name)` returns the (renamed)
    # allocation, and `cg.mem.abuf.get(alias_input_name)` returns None.
    # The post-emit free in generate() then skips alias_input_name (no
    # double-free) and frees the *other* input normally.
    aliased = cg.mem.abuf.allocations.pop(alias_input_name, None)
    if aliased is not None:
        aliased.name = node.name
        cg.mem.abuf.allocations[node.name] = aliased
        out_alloc = aliased
    else:
        # Defensive: alias_input_name wasn't actually in ABUF (e.g. the
        # name was already aliased upstream). Fall back to allocating a
        # fresh slot for node.name and writing the result there. This
        # branch shouldn't trigger under the current IR contract but
        # keeps the helper robust against future graph rewriters.
        out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)

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


# ---------------------------------------------------------------------------
# M3-A: matmul_qkt W8A32 lowering — Q @ K^T with static composite dequant
# ---------------------------------------------------------------------------


def emit_matmul_qkt_w8a32(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 lowering for `matmul_qkt` (per-head Q @ K^T attention matmul).

    Q and K live in ABUF as FP32 (the per-head Q/K projection matmuls
    that produced them were lowered through `emit_matmul_w8a32`, so
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
        k_row_units_fp32 = (K_pad * FP32_BYTES_PER_ELEM) // UNIT
        k_pad_units = k_fp32_alloc.offset_units + key_len * k_row_units_fp32
        zero_pad_dram = cg._dram_offset_required(
            "__zero_pad__", f"zeroing K padding rows for '{node.name}'"
        )
        cg._emit_dma_load(
            BUF_ABUF,
            k_pad_units,
            pad_rows * K_pad * FP32_BYTES_PER_ELEM,
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
    out_row_units = (N_pad * FP32_BYTES_PER_ELEM) // UNIT
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
            )
        )
        cg._emit(SyncInsn(resource_mask=0b100))

    # Cleanup.
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


def emit_matmul_attn_v_w8a32(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 lowering for `matmul_attn_v` (softmax(QK^T) @ V per head).

    Both inputs live in ABUF as FP32 by the time this emitter fires:
      - softmax output: produced by `emit_softmax_fp32` (M2) — FP32
        normalized probabilities in (M=seq_len, K=seq_len).
      - V tile: produced by the per-head V projection's
        `emit_matmul_w8a32` — FP32 in (K=seq_len, N=d_head).

    V is already laid out as [K, N] (no transpose needed — the MATMUL
    expects src2 in WBUF as [K, N] which is exactly V's shape). So the
    sequence is structurally similar to `emit_matmul_qkt_w8a32` minus
    the transpose step:

        FP32 softmax → QUANT_FP32_INT8 (S[s]=1/sm_scale) → INT8 ABUF scratch
        FP32 V       → QUANT_FP32_INT8 (S[s]=1/v_scale)  → INT8 ABUF scratch
        INT8 V       → BUF_COPY (no transpose)            → WBUF
        Composite PC scale (sm_scale × v_scale)           → WBUF
        MATMUL softmax_int8 @ V_wbuf                       → INT32 ACCUM
        DEQUANT_ACCUM_FP32 (M1, no _SCALED)                → FP32 attn_v in ABUF

    Static calibration scales here — same architecture as the M3-A QKT
    re-quant boundary. No 1/√d_head factor in the composite (that was
    already applied by emit_matmul_qkt_w8a32 in its dequant epilogue;
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
      `matmul` node whose `emit_matmul_w8a32` already allocated its
      ABUF region under `node.name`. Tests that build a synthetic
      `matmul_attn_v` fragment with graph-input names like `sm`/`v_in`
      MUST pre-allocate those inputs in `cg.mem.abuf` before
      `generate()`. Otherwise the lazy `_abuf_alloc_fp32` here may
      land them on a region that an earlier emit step's freed scratch
      occupied — the emitted scratch-QUANT instruction still writes
      there at execution time and corrupts the pre-seeded input
      bytes. The end-to-end test in `test_w8a32_codegen.py` does this
      pre-allocation explicitly; see the comment around the
      `cg.mem.abuf.alloc("v_in", ...)` calls there.
    """
    if node.op != "matmul_attn_v":
        raise NotImplementedError(
            f"emit_matmul_attn_v_w8a32 only handles op='matmul_attn_v' "
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
    # Same reasoning as K's pad-row zero-fill in emit_matmul_qkt_w8a32:
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
        v_row_units_fp32 = (N_pad * FP32_BYTES_PER_ELEM) // UNIT
        v_pad_units = v_fp32_alloc.offset_units + key_len * v_row_units_fp32
        zero_pad_dram = cg._dram_offset_required(
            "__zero_pad__", f"zeroing V padding rows for '{node.name}'"
        )
        cg._emit_dma_load(
            BUF_ABUF,
            v_pad_units,
            pad_rows * N_pad * FP32_BYTES_PER_ELEM,
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
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

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

    # DEQUANT_ACCUM_FP32: accum × wt_scale_pc → FP32 attn_v in ABUF.
    cg._emit(
        DequantAccumFp32Insn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=pc_scale_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
        )
    )
    cg._emit(SyncInsn(resource_mask=0b100))

    # Cleanup.
    cg.mem.abuf.free(f"{node.name}__sm_int8")
    cg.mem.abuf.free(f"{node.name}__v_int8")
    cg.mem.wbuf.free(f"v_head{head_idx}_{node.name}")
    cg.mem.wbuf.free(f"_attn_v_pc_{node.name}")

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
