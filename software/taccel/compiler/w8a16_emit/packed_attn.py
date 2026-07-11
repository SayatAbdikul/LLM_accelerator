"""Lever B — packed QK^T attention core (block-diagonal, batched decode only).

Consolidates the 12 per-head QK^T systolic matmuls of one (layer, stream) into
ONE block-diagonal packed matmul, then splits the packed INT32 result back into
the 12 per-head FP16 score tiles with 12 dequants. Everything downstream
(masked-softmax, softmax-quant, attn_v, gather, concat) is UNCHANGED — the score
tiles this emitter produces are byte-identical to the per-head path.

Two ops:
  - ``packed_qkt_matmul`` (one per layer,stream): builds the block-diagonal INT8
    Q_pack (12 QUANTs, head h's query at row h, cols [d_head*h, d_head*h+d_head)),
    assembles K_all^T in WBUF (12 transposes, head h at rows [d_head*h, ...)), and
    runs one MATMUL Q_pack @ K_all^T -> ACCUM (n_head_pad, key_len). The result is
    LEFT IN ACCUM (DEQUANT_ACCUM_FP32 can only read ACCUM); the following
    ``qkt_dequant`` nodes MUST run before any AV matmul clobbers it.
  - ``qkt_dequant`` (one per layer,head,stream): DEQUANT_ACCUM_FP32 of ACCUM row
    ``head_idx`` (m_exact=1) through the composite PC vector -> the per-head
    ``blockL_headH_sS_qkt`` FP16 score tile (row 0 = head h's scores), exactly the
    tile the per-head ``emit_matmul_qkt_w8a16`` produced.

Byte-exactness: Q_pack row h holds head h's INT8 query (same 6/127 static scale,
same source bytes) in its diagonal block and zeros elsewhere; K_all^T rows
[d_head*h, ...) hold head h's transposed INT8 K. The INT8 matmul row h therefore
sums head h's 64 real products plus 704 exact-zero products = head h's INT32
scores, identical to the per-head ACCUM. Dequant uses the identical composite
(q_scale*k_scale*inv_sqrt, all 6/127 defaults on the batched path). See
scratchpad/leverB_design.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...isa.instructions import (
    BufCopyInsn,
    ConfigTileInsn,
    DequantAccumFp32Insn,
    MatmulInsn,
    QuantFp32Int8Insn,
    SetScaleInsn,
    SyncInsn,
)
from ...isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF, WBUF_SIZE
from ..kv_cache import normalize_kv_kind
from ..tiler import TILE, pad_dim
from ._common import UNIT, _abuf_alloc_fp32, _fp16_to_uint16

if TYPE_CHECKING:
    from ..codegen import CodeGenerator
    from ..ir import IRNode

DEFAULT_ACT_SCALE = 6.0 / 127.0


def emit_packed_qkt_matmul(cg: "CodeGenerator", node: "IRNode") -> None:
    """Block-diagonal packed Q @ K^T for all heads of one (layer, stream).

    Two K-residency modes (byte-identical K_all^T, so identical scores):
      - pre-loaded: inputs = [qb_0..qb_{H-1}, kload_0..kload_{H-1}] (batched
        query projections + per-(head,stream) INT8 K caches already in ABUF).
      - ``stream_k`` (default on the batched path): inputs = [qb_0..qb_{H-1}]
        only; the emitter loads each head's INT8 K cache into a REUSED ABUF
        scratch (load -> transpose -> free), so at most ONE cache is resident.
        The global head index per WBUF block comes from ``attrs['k_heads']``.
        Streaming lifts the co-resident-K-cache ABUF cap that otherwise pins the
        group size, letting more heads pack into one matmul.
    Leaves ACCUM = (n_head_pad, key_len) INT32.
    """
    from ..emit.kv import kv_int8_full_rows  # lazy: import cycle w/ emit.kv

    n_head = int(node.attrs["n_head"])
    d_head = int(node.attrs["d_head"])
    d_model = int(node.attrs["d_model"])
    key_len = int(node.attrs["key_len"])
    stream = int(node.attrs["stream"])
    q_scale_keys = node.attrs["q_scale_keys"]

    if d_head % TILE != 0:
        raise NotImplementedError(
            f"packed_qkt requires d_head % {TILE} == 0 (got {d_head})"
        )
    if n_head * d_head != d_model:
        raise NotImplementedError(
            f"packed_qkt requires n_head*d_head == d_model "
            f"({n_head}*{d_head} != {d_model})"
        )

    M_pad = pad_dim(n_head)           # heads packed into the row dimension
    N_pad = pad_dim(key_len)          # key positions
    Kc_pad = d_model                  # packed contraction dim (already 16-mult)
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    k_tiles = Kc_pad // TILE

    qb_allocs = [cg.mem.abuf.get(node.inputs[h]) for h in range(n_head)]
    for h, a in enumerate(qb_allocs):
        if a is None:
            raise KeyError(
                f"Missing ABUF alloc '{node.inputs[h]}' (batched query) for {node.name}"
            )

    # ----- Q_pack: zero (M_pad, d_model) INT8, then 12 diagonal QUANTs -----
    q_row_units_fp16 = (d_head * cg.elem_bytes) // UNIT
    qpack = cg.mem.abuf.alloc(f"{node.name}__qpack", M_pad * d_model)  # INT8
    zero_dram = cg._dram_offset_required(
        "__zero_pad__", f"zeroing Q_pack for '{node.name}'"
    )
    cg._emit_dma_load(BUF_ABUF, qpack.offset_units, M_pad * d_model, 3, zero_dram)
    cg._emit(SyncInsn(resource_mask=0b001))

    for h in range(n_head):
        q_scale = float(cg.calibration_scales.get(q_scale_keys[h], DEFAULT_ACT_SCALE))
        cg._emit(ConfigTileInsn(M=0, N=d_head // TILE - 1, K=0, m_exact=1))
        sreg_q = cg._alloc_sreg()
        cg._emit(SetScaleInsn(
            sreg=sreg_q, src_mode=0,
            imm16=_fp16_to_uint16(1.0 / max(q_scale, 1e-12)),
        ))
        src_off = qb_allocs[h].offset_units + stream * q_row_units_fp16
        dst_off = qpack.offset_units + (h * d_model + d_head * h) // UNIT
        cg._emit(QuantFp32Int8Insn(
            src1_buf=BUF_ABUF, src1_off=src_off,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=dst_off,
            sreg=sreg_q, flags=cg.fp_precision_flag,
        ))
        cg._emit(SyncInsn(resource_mask=0b100))

    # ----- K_all^T: per-head INT8 transposes into one WBUF tile -----
    kt_wbuf = cg.mem.wbuf.alloc(f"{node.name}__ktall", Kc_pad * N_pad)  # INT8
    if Kc_pad * N_pad > WBUF_SIZE:
        raise NotImplementedError(
            f"packed K_all^T {Kc_pad}x{N_pad} INT8 = {Kc_pad * N_pad} B exceeds "
            f"WBUF {WBUF_SIZE} B — needs the B2 N-split (not yet implemented)"
        )
    loaded_rows = kv_int8_full_rows(cg, key_len)
    # Two K-residency modes with byte-identical K_all^T bytes:
    #   - pre-loaded (grouped path): each head's INT8 K cache is already an ABUF
    #     kv_load input (inputs[n_head+h]); transpose it in place.
    #   - streamed (`stream_k`, full/wide pack): load one head's INT8 K cache
    #     into a reused ABUF scratch, transpose into the WBUF block, free the
    #     scratch — so at most ONE K cache is resident (not `group`). This lifts
    #     the co-resident-K-cache ABUF cap that otherwise pins the group size,
    #     without touching the transpose bytes or the downstream dequant/AV.
    stream_k = bool(node.attrs.get("stream_k"))
    for h in range(n_head):
        dst_off = kt_wbuf.offset_units + (d_head * h * N_pad) // UNIT
        if stream_k:
            head_g = int(node.attrs["k_heads"][h])
            layer = int(node.attrs["block_idx"])
            entry = cg.kv_layout.entry(
                layer, normalize_kv_kind("key"), head_g, stream
            )
            kscratch = cg.mem.abuf.alloc(
                f"{node.name}__kstream", N_pad * d_head
            )  # INT8, reused per head
            # Full-context int8 load from position 0 (mirrors emit_kv_load's
            # decode `tokens>1` branch: base-relocated, no kv_base patch).
            cg._emit_dma_load(
                BUF_ABUF, kscratch.offset_units, loaded_rows * d_head, 2, 0,
                dram_off_units=entry.dram_off_units,
                relocation_symbol=entry.base_symbol,
            )
            cg._emit(SyncInsn(resource_mask=0b001))
            k_src_off = kscratch.offset_units
        else:
            k_alloc = cg.mem.abuf.get(node.inputs[n_head + h])
            if k_alloc is None:
                raise KeyError(
                    f"Missing ABUF alloc '{node.inputs[n_head + h]}' "
                    f"(int8 K cache) for {node.name}"
                )
            k_src_off = k_alloc.offset_units
        if loaded_rows < N_pad:
            # Zero the INT8 K pad rows in the cache tile (matches per-head path).
            zpad = cg._dram_offset_required(
                "__zero_pad__", f"zeroing int8 K pad rows for '{node.name}' h{h}"
            )
            cg._emit_dma_load(
                BUF_ABUF,
                k_src_off + (loaded_rows * d_head) // UNIT,
                (N_pad - loaded_rows) * d_head, 3, zpad,
            )
            cg._emit(SyncInsn(resource_mask=0b001))
        # transpose (N_pad, d_head) INT8 -> (d_head, N_pad) at row d_head*h
        cg._emit(BufCopyInsn(
            src_buf=BUF_ABUF, src_off=k_src_off,
            dst_buf=BUF_WBUF, dst_off=dst_off,
            length=(N_pad * d_head) // UNIT,
            src_rows=N_pad // TILE,
            transpose=1,
        ))
        cg._emit(SyncInsn(resource_mask=0b001))
        if stream_k:
            cg.mem.abuf.free(f"{node.name}__kstream")

    # ----- one packed MATMUL Q_pack @ K_all^T -> ACCUM -----
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=k_tiles - 1))
    cg._emit(MatmulInsn(
        src1_buf=BUF_ABUF, src1_off=qpack.offset_units,
        src2_buf=BUF_WBUF, src2_off=kt_wbuf.offset_units,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=0,
    ))
    cg._emit(SyncInsn(resource_mask=0b010))

    cg.mem.abuf.free(f"{node.name}__qpack")
    cg.mem.wbuf.free(f"{node.name}__ktall")
    # kload_h ABUF tiles are freed by the generate() last-use sweep.


def emit_qkt_dequant(cg: "CodeGenerator", node: "IRNode") -> None:
    """DEQUANT_ACCUM_FP32 of ACCUM row head_idx -> per-head FP16 score tile.

    Reads the packed matmul's ACCUM (must be intact — no AV matmul between). The
    composite PC vector (blockL_headH_sS_qkt__qkt_pc_scale) is staged by codegen.
    """
    query_len = int(node.attrs.get("query_len", 1))
    key_len = int(node.attrs["key_len"])
    # `accum_row` = this head's row within its group's ACCUM tile (0-based
    # local index); falls back to head_idx for the whole-layer (single-group)
    # case. Decode has query_len==1, so a group's ACCUM is (group_pad, N_pad).
    accum_row = int(node.attrs.get("accum_row", node.attrs["head_idx"]))
    N_pad = pad_dim(key_len)
    n_tiles = N_pad // TILE

    # PC scale vector -> WBUF (staged at DRAM-layout time for this node).
    pc_sym = f"{node.name}__qkt_pc_scale"
    pc_dram = cg._dram_offset_required(
        pc_sym, f"loading packed QKT composite PC scale for '{node.name}'"
    )
    pc_bytes = N_pad * 2  # FP16
    pc_alloc = cg.mem.wbuf.alloc(f"_pqkt_pc_{node.name}", pc_bytes)
    cg._emit_dma_load(BUF_WBUF, pc_alloc.offset_units, pc_bytes, 0, pc_dram)
    cg._emit(SyncInsn(resource_mask=0b001))

    # Same (M_pad, N_pad) FP16 score tile the per-head path produced (row 0 =
    # this head's scores). ABUF is bounded by the builder interleaving each
    # group's softmax/AV so only `group` score tiles are live at once.
    M_pad = pad_dim(query_len)
    m_tiles = M_pad // TILE
    out_alloc = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)

    # ACCUM holds (group_pad, N_pad) INT32; row accum_row = this head's scores.
    accum_row_units = (accum_row * N_pad * 4) // UNIT
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0, m_exact=1))
    cg._emit(DequantAccumFp32Insn(
        src1_buf=BUF_ACCUM, src1_off=accum_row_units,
        src2_buf=BUF_WBUF, src2_off=pc_alloc.offset_units,
        dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
        flags=cg.fp_precision_flag,
    ))
    cg._emit(SyncInsn(resource_mask=0b100))
    cg.mem.wbuf.free(f"_pqkt_pc_{node.name}")

    q_scale = float(cg.calibration_scales.get(node.attrs["q_scale_key"], DEFAULT_ACT_SCALE))
    k_scale = float(cg.calibration_scales.get(node.attrs["k_scale_key"], DEFAULT_ACT_SCALE))
    inv_sqrt = float(node.attrs.get("scale", int(node.attrs["d_head"]) ** -0.5))
    cg._record_trace_event(
        node.name, BUF_ABUF, out_alloc.offset_units, M_pad, N_pad,
        query_len, key_len, "fp32", q_scale * k_scale * inv_sqrt,
    )
