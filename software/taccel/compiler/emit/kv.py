"""KV-cache emit helpers (store, load) + logits-store.

Free-function migrations of the original `_kv_*` and `_emit_kv_*` and
`_emit_logits_store` methods on `CodeGenerator`. Semantics are
byte-for-byte preserved; only the `self`→`cg` rename and the lift to
module-scope changes.

The KV-cache layout (`self.kv_layout`) is required for every kv_load /
kv_store; logits_store uses ABUF as a staging area for DRAM-temp
sources and chunk-streams when the row would not fit ABUF whole.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ...isa.instructions import SyncInsn
from ...isa.opcodes import ABUF_SIZE, BUF_ABUF
from ..ir import IRNode
from ..kv_cache import normalize_kv_kind
from ..tiler import pad_dim
from ._common import UNIT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def kv_entry_for_node(cg: "CodeGenerator", node: IRNode):
    if cg.kv_layout is None:
        raise ValueError("kv_layout is required for kv_load/kv_store nodes")
    return cg.kv_layout.entry(
        int(node.attrs["layer"]),
        normalize_kv_kind(node.attrs["kind"]),
        int(node.attrs["head"]),
    )


def kv_transfer_bytes(cg: "CodeGenerator", node: IRNode, *,
                      decode_default: bool) -> int:
    if "xfer_bytes" in node.attrs:
        return int(node.attrs["xfer_bytes"])
    tokens = int(node.attrs.get("tokens", 1 if decode_default else node.attrs.get("seq_len", 1)))
    # M4-B: W8A32 stores FP32 K/V tiles (4 bytes/elem) in the KV cache.
    # The KV cache layout's `elem_bytes` is the source of truth; falling
    # back to `use_fp16_activations` keeps decoder bundles whose `kv_layout`
    # wasn't built with explicit `elem_bytes` working as expected.
    elem_bytes = 1
    if cg.kv_layout is not None:
        elem_bytes = int(cg.kv_layout.elem_bytes)
    else:
        elem_bytes = 4
    return tokens * cg.config.d_head * elem_bytes


def kv_source_location(cg: "CodeGenerator", node: IRNode) -> Tuple[int, int]:
    if "src_buf" in node.attrs and "src_off_units" in node.attrs:
        return int(node.attrs["src_buf"]), int(node.attrs["src_off_units"])
    if not node.inputs:
        raise ValueError(f"{node.name} requires an input allocation or src_buf/src_off_units attrs")
    alloc = cg.mem.abuf.get(node.inputs[0])
    if alloc is None:
        raise KeyError(f"Missing ABUF allocation '{node.inputs[0]}' for {node.name}")
    return alloc.buf_id, alloc.offset_units


def emit_kv_store(cg: "CodeGenerator", node: IRNode) -> None:
    entry = kv_entry_for_node(cg, node)
    src_buf, src_off = kv_source_location(cg, node)
    decode_mode = bool(node.attrs.get("decode", cg.stream_name == "decode"))
    xfer_bytes = kv_transfer_bytes(cg, node, decode_default=decode_mode)
    addr_reg = int(node.attrs.get("addr_reg", 2))
    if decode_mode:
        cg._emit_dma_store(
            src_buf,
            src_off,
            xfer_bytes,
            addr_reg,
            0,
            dram_off_units=entry.dram_off_units,
            runtime_patch_kind="kv_base",
            runtime_base_symbol=entry.base_symbol,
        )
    else:
        cg._emit_dma_store(
            src_buf,
            src_off,
            xfer_bytes,
            addr_reg,
            0,
            dram_off_units=entry.dram_off_units,
            relocation_symbol=entry.base_symbol,
        )
    cg._emit(SyncInsn(resource_mask=0b001))


def emit_kv_load(cg: "CodeGenerator", node: IRNode) -> None:
    entry = kv_entry_for_node(cg, node)
    decode_mode = bool(node.attrs.get("decode", cg.stream_name == "decode"))
    xfer_bytes = kv_transfer_bytes(cg, node, decode_default=decode_mode)
    addr_reg = int(node.attrs.get("addr_reg", 2))
    dst_buf = int(node.attrs.get("dst_buf", BUF_ABUF))
    if "dst_off_units" in node.attrs:
        dst_off = int(node.attrs["dst_off_units"])
    else:
        alloc_bytes = xfer_bytes
        if len(node.output_shape) == 2:
            rows = pad_dim(int(node.output_shape[0]))
            cols = pad_dim(int(node.output_shape[1]))
            # M2-W8A16 fix: in W8A{32,16} mode the K/V cache stores FP-precision
            # elements (4 bytes for FP32, 2 bytes for FP16). The ABUF alloc must
            # hold rows*cols*elem_bytes, not rows*cols (the INT8 size). Without
            # this multiplier the FP16 K tile got 1024B instead of 2048B; the
            # 1024B hole between [128..192] was first-fit-picked as the next
            # QUANT's INT8 destination, corrupting the FP16 K source at offsets
            # [192..256] and producing NaN-decoded INT8 byte pairs (e.g. 0x09
            # 0xfc -> FP16 NaN). See tools/debug_w8a16_nan.py for the trace.
            tile_elem_bytes = cg.elem_bytes
            alloc_bytes = max(alloc_bytes, rows * cols * tile_elem_bytes)
        # M4-debug: large W8A32 kv_load tiles (256-token K/V FP32 cache
        # at GPT-2 scale = 64 KB) fragment under first-fit. Compact
        # before the alloc to expose a contiguous free region.
        if dst_buf == BUF_ABUF and alloc_bytes > 16 * 1024:
            cg._compact_abuf()
        alloc = cg.mem.abuf.alloc(node.name, alloc_bytes)
        dst_off = alloc.offset_units
    tokens = int(node.attrs.get("tokens", 1))
    if decode_mode and tokens > 1:
        # Full-context kv_load (tokens = seq_len): must always read from
        # position 0 so the QKT sees K[0..seq_len-1].  kv_store uses
        # kv_base to write the *current* token at the right position;
        # kv_load must NOT inherit that offset or it would skip position 0
        # and feed stale garbage columns to the QKT.
        cg._emit_dma_load(
            dst_buf,
            dst_off,
            xfer_bytes,
            addr_reg,
            0,
            dram_off_units=entry.dram_off_units,
            relocation_symbol=entry.base_symbol,
        )
    elif decode_mode:
        # Single-token kv_load (tokens = 1): reads one specific token at
        # the position indicated by kv_base (position-indexed access).
        cg._emit_dma_load(
            dst_buf,
            dst_off,
            xfer_bytes,
            addr_reg,
            0,
            dram_off_units=entry.dram_off_units,
            runtime_patch_kind="kv_base",
            runtime_base_symbol=entry.base_symbol,
        )
    else:
        cg._emit_dma_load(
            dst_buf,
            dst_off,
            xfer_bytes,
            addr_reg,
            0,
            dram_off_units=entry.dram_off_units,
            relocation_symbol=entry.base_symbol,
        )
    cg._emit(SyncInsn(resource_mask=0b001))


def emit_logits_store(cg: "CodeGenerator", node: IRNode) -> None:
    """Store a logits tensor to the ProgramBundle logits region.

    INT8 path: 1 byte/elem, store_rows × cols_pad bytes.

    W8A32 path (M3-D): lm_head produces FP32 (4 bytes/elem), so the
    DMA store moves 4× more bytes and row strides are 4× wider.
    Detected via `cg.use_fp16_activations`; the IR contract is unchanged
    (logits_store always describes the logical shape).
    """
    if not node.inputs:
        raise ValueError(f"{node.name} requires a source tensor input")
    source_name = node.inputs[0]
    source_shape = tuple(node.attrs.get("source_shape", node.output_shape))
    if len(source_shape) != 2:
        raise ValueError(f"{node.name} requires a 2D source_shape/output_shape, got {source_shape}")

    logical_rows = int(source_shape[0])
    logical_cols = int(source_shape[1])
    if logical_rows <= 0 or logical_cols <= 0:
        raise ValueError(f"{node.name} source shape must be positive, got {source_shape}")
    cols_pad = pad_dim(logical_cols)
    default_row = logical_rows - 1 if cg.stream_name == "prefill" else 0
    row_index = int(node.attrs.get("row_index", default_row))
    store_rows = int(node.attrs.get("store_rows", 1))
    if row_index < 0 or row_index + store_rows > logical_rows:
        raise ValueError(
            f"{node.name} row range [{row_index}, {row_index + store_rows}) "
            f"exceeds logical row count {logical_rows}"
        )
    if store_rows <= 0:
        raise ValueError(f"{node.name} store_rows must be positive")

    # M3-D + M2-W8A16: W8A{32,16} lm_head output is FP-precision
    # (FP32=4 or FP16=2 bytes/elem) — both the DMA xfer size and
    # the row stride scale by self.elem_bytes.
    elem_bytes = cg.elem_bytes
    size_bytes = int(node.attrs.get(
        "xfer_bytes", store_rows * cols_pad * elem_bytes,
    ))
    addr_reg = int(node.attrs.get("addr_reg", 3))
    symbol = str(node.attrs.get("symbol", f"{cg.stream_name}_logits_offset"))
    row_byte_offset = row_index * cols_pad * elem_bytes

    staging_name = None
    if "src_buf" in node.attrs and "src_off_units" in node.attrs:
        src_buf = int(node.attrs["src_buf"])
        src_off = int(node.attrs["src_off_units"]) + row_byte_offset // UNIT
    elif source_name in cg.dram_temp_outputs:
        # Strip-mined / W8A32-tiled producers leave their full output in
        # DRAM temp. Load the requested row window back into ABUF before
        # storing to the logits region.
        #
        # M4-G: for W8A32 lm_head at GPT-2 vocab=50257, one row is
        # 50272*4 ≈ 200 KB, which exceeds ABUF (128 KB). Stream the
        # row in column chunks small enough to fit. Each chunk:
        # load FP32 chunk from DRAM-temp into ABUF, DMA_STORE to its
        # slot in the logits region, free the staging slot.
        if size_bytes > ABUF_SIZE:
            # Pick a chunk size ≤ 32 KB (8192 FP32 cols at most;
            # corresponds to a quarter of ABUF for breathing room).
            chunk_bytes = 32 * 1024
            if elem_bytes > 1:
                chunk_bytes = max(chunk_bytes - (chunk_bytes % elem_bytes), elem_bytes)
            offset = 0
            while offset < size_bytes:
                cur = min(chunk_bytes, size_bytes - offset)
                chunk_name = f"{node.name}_staging_off{offset}"
                chunk_alloc = cg.mem.abuf.alloc(chunk_name, cur)
                cg._emit_dma_load(
                    BUF_ABUF,
                    chunk_alloc.offset_units,
                    cur,
                    addr_reg,
                    cg.dram_temp_outputs[source_name] + row_byte_offset + offset,
                )
                cg._emit(SyncInsn(resource_mask=0b001))
                cg._emit_dma_store(
                    BUF_ABUF,
                    chunk_alloc.offset_units,
                    cur,
                    addr_reg,
                    offset,
                    relocation_symbol=symbol,
                )
                cg._emit(SyncInsn(resource_mask=0b001))
                cg.mem.abuf.free(chunk_name)
                offset += cur
            return
        staging_name = f"{node.name}_staging"
        staging = cg.mem.abuf.alloc(staging_name, size_bytes)
        cg._emit_dma_load(
            BUF_ABUF,
            staging.offset_units,
            size_bytes,
            addr_reg,
            cg.dram_temp_outputs[source_name] + row_byte_offset,
        )
        cg._emit(SyncInsn(resource_mask=0b001))
        src_buf = BUF_ABUF
        src_off = staging.offset_units
    else:
        alloc = cg.mem.abuf.get(source_name)
        if alloc is None:
            raise KeyError(f"Missing ABUF allocation '{source_name}' for {node.name}")
        src_buf = alloc.buf_id
        src_off = alloc.offset_units + row_byte_offset // UNIT

    cg._emit_dma_store(
        src_buf,
        src_off,
        size_bytes,
        addr_reg,
        0,
        relocation_symbol=symbol,
    )
    cg._emit(SyncInsn(resource_mask=0b001))
    if staging_name is not None:
        cg.mem.abuf.free(staging_name)
