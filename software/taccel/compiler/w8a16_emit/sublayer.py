"""W8A16 sub-layer op lowering: LAYERNORM_FP32, GELU_FP32, SOFTMAX_FP32 /
MASKED_SOFTMAX_FP32, VADD_FP32.

Each helper reads FP{16,32}-storage source(s) from ABUF (FP16 gamma/beta
from WBUF where applicable), emits the matching ISA op, and writes the
FP-precision result back to ABUF. None of these consume per-tensor
activation scales — sub-layer compute stays in the FP datapath.

`emit_vadd_fp32` (residual stream) and `emit_layernorm_fp32` cooperate
with the M4-A ABUF-residency optimization: large LN inputs spill to
DRAM-temp on emit; the residual VADD reloads them via
`cg._load_dram_to_abuf_fp` and aliases the output back into an input
slot to keep peak ABUF below 128 KB.

`emit_gelu_fp32` has a DRAM-streamed path for when the input is too
large for ABUF (fc1 → GELU → fc2 at GPT-2 scale, 192 KB FP16 tile).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...isa.instructions import (
    ConfigAttnInsn,
    ConfigTileInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    SoftmaxFp32Insn,
    SyncInsn,
    VaddFp32Insn,
)
from ...isa.opcodes import BUF_ABUF, BUF_WBUF
from ..tiler import TILE, pad_dim
from ._common import UNIT, _abuf_alloc_fp32

if TYPE_CHECKING:
    from ..codegen import CodeGenerator
    from ..ir import IRNode


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
        from ...isa.opcodes import BUF_WBUF as _BUF_WBUF
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
        in_alloc = cg._load_dram_to_abuf_fp(in_name, M_pad, N_pad)
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
            flags=cg.fp_precision_flag,
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
    # via `_load_dram_to_abuf_fp`. Skip for tiny fixtures (≤ 16 KB
    # tiles) so existing tests' instruction counts stay unchanged.
    tile_bytes = M_pad * N_pad * cg.elem_bytes
    last_use_idx = cg.last_uses.get(in_name, -1)
    has_future_use = last_use_idx > cg.current_node_idx
    if (
        has_future_use
        and tile_bytes >= cg.fp_spill_threshold_bytes
        and in_name not in cg.dram_temp_fp32_outputs
    ):
        cg._spill_fp32_tile_to_dram(in_name, in_alloc, M_pad, N_pad)


def emit_gelu_fp32(cg: "CodeGenerator", node: "IRNode") -> None:
    """W8A32 GELU_FP32 lowering (tanh approximation, matching gelu_new).

    M4-G extension: when the input is in DRAM-temp (e.g. produced by
    `emit_matmul_w8a16_large_weight_tiled` for fc1 → GELU → fc2 at
    GPT-2 scale where the 192 KB fc1 output doesn't fit ABUF), stream
    the GELU tile-by-tile through DRAM: load N-tile to ABUF → GELU →
    store back to DRAM. The output node lives in the same DRAM-temp
    slot (GELU is element-wise) so downstream consumers see the right
    bytes via `dram_temp_outputs[node.name]`.
    """
    M_pad = pad_dim(node.output_shape[0])
    N_pad = pad_dim(node.output_shape[1])
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE

    in_name = node.inputs[0]
    if in_name in cg.dram_temp_fp32_outputs:
        # Stream tile-by-tile. The full input lives at
        # `cg.dram_temp_outputs[in_name]` as a row-major M_pad × N_pad ×
        # cg.elem_bytes region (FP32=4 or FP16=2). Allocate a new DRAM-
        # temp slot for the GELU output (same size as input).
        input_dram_off = cg.dram_temp_outputs[in_name]
        full_bytes = M_pad * N_pad * cg.elem_bytes
        output_dram_off = cg.dram_temp_start + cg.mem.alloc_dram_temp(
            f"{node.name}_w8a16_out_fp32", full_bytes
        )

        # Pick an N-tile size that fits ABUF together with some slack
        # for the GELU's I/O. M_pad * n_tile * elem_bytes ≤ 32 KB →
        # FP32: n_tile ≤ 32K/(16*4) = 512. FP16: 1024.
        n_tile = min(N_pad, 32768 // (16 * cg.elem_bytes))
        n_tile = max(TILE, (n_tile // TILE) * TILE)
        for n_start in range(0, N_pad, n_tile):
            n_len = min(n_tile, N_pad - n_start)
            tile_alloc = cg.mem.abuf.alloc(
                f"{node.name}_gelu_tile_n{n_start}", M_pad * n_len * cg.elem_bytes
            )
            tile_row_bytes = n_len * cg.elem_bytes
            tile_row_units = tile_row_bytes // UNIT
            full_row_bytes_in_dram = N_pad * cg.elem_bytes
            # Load M_pad rows of (n_len × elem_bytes bytes) from DRAM into ABUF.
            for r in range(M_pad):
                cg._emit_dma_load(
                    BUF_ABUF,
                    tile_alloc.offset_units + r * tile_row_units,
                    tile_row_bytes, 0,
                    input_dram_off + r * full_row_bytes_in_dram + n_start * cg.elem_bytes,
                )
            cg._emit(SyncInsn(resource_mask=0b001))

            # GELU_FP32 in-place over this N-tile.
            tile_n_tiles = n_len // TILE
            cg._emit(ConfigTileInsn(M=m_tiles - 1, N=tile_n_tiles - 1, K=0))
            cg._emit(GeluFp32Insn(
                src1_buf=BUF_ABUF, src1_off=tile_alloc.offset_units,
                src2_buf=BUF_ABUF, src2_off=0,
                dst_buf=BUF_ABUF, dst_off=tile_alloc.offset_units,
                flags=cg.fp_precision_flag,
            ))
            cg._emit(SyncInsn(resource_mask=0b100))

            # Store back to DRAM-temp.
            for r in range(M_pad):
                cg._emit_dma_store(
                    BUF_ABUF,
                    tile_alloc.offset_units + r * tile_row_units,
                    tile_row_bytes, 2,
                    output_dram_off + r * full_row_bytes_in_dram + n_start * cg.elem_bytes,
                )
            cg._emit(SyncInsn(resource_mask=0b010))
            cg.mem.abuf.free(tile_alloc.name)

        # Register the GELU output in DRAM-temp.
        cg.dram_temp_outputs[node.name] = output_dram_off
        cg.dram_temp_fp32_outputs[node.name] = full_bytes
        out_scale = cg.calibration_scales.get(node.name, 1.0 / 127.0)
        cg._record_trace_event(
            node.name, BUF_ABUF, 0, M_pad, N_pad,
            node.output_shape[0], node.output_shape[1], "fp32", out_scale,
        )
        return

    # ABUF-resident input: simple one-shot GELU.
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    in_alloc = _abuf_alloc_fp32(cg, in_name, M_pad, N_pad)
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    cg._emit(
        GeluFp32Insn(
            src1_buf=BUF_ABUF,
            src1_off=in_alloc.offset_units,
            src2_buf=BUF_ABUF,
            src2_off=0,
            dst_buf=BUF_ABUF,
            dst_off=out_alloc.offset_units,
            flags=cg.fp_precision_flag,
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
        # M4-D: when this softmax sits in a decode stream that patches
        # CONFIG_ATTN at runtime, capture the PC and register a
        # RuntimeConfigAttnSite so HostRunner._patch_decode_attention_
        # context(position=N) can rewrite (query_row_base, valid_kv_len)
        # per decode step. Mirrors codegen._emit_config_attn_for_qkt.
        if runtime_config_attn:
            from ...assembler.assembler import RuntimeConfigAttnSite
            pc = len(cg.instructions)
            cg._emit(ConfigAttnInsn(
                query_row_base=0,
                valid_kv_len=1,
                mode=attn_mode,
            ))
            cg.runtime_config_attn_sites.append(RuntimeConfigAttnSite(
                stream=cg.stream_name,
                local_pc=pc,
                absolute_pc=0,
                mode=attn_mode,
            ))
        else:
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
            flags=cg.fp_precision_flag,
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
       (4 bytes/elem) — `_load_dram_to_abuf_fp` does this.

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
        cg._load_dram_to_abuf_fp(in0_name, M_pad, N_pad)
        in0_reloaded = True
    if in1_name in cg.dram_temp_fp32_outputs and cg.mem.abuf.get(in1_name) is None:
        cg._load_dram_to_abuf_fp(in1_name, M_pad, N_pad)
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
            flags=cg.fp_precision_flag,
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


