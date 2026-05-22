"""Embedding emit helpers: token/position lookups + CLS prepend/extract.

Free-function migrations of the original `_emit_embedding_lookup`,
`_emit_cls_prepend`, `_emit_pos_embed_add`, and `_emit_cls_extract`
methods on `CodeGenerator`. Semantics are byte-for-byte preserved;
only the `self`→`cg` rename and the lift to module-scope changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...isa.instructions import BufCopyInsn, ConfigTileInsn, SyncInsn, VaddInsn
from ...isa.opcodes import BUF_ABUF, BUF_WBUF
from ..ir import IRNode
from ..tiler import TILE, pad_dim
from ._common import UNIT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def emit_embedding_lookup(cg: "CodeGenerator", node: IRNode, *,
                          default_table: str) -> None:
    """Emit fixed-row token/position embedding loads for Stage 1 tests.

    In W8A32 mode (M3-prep), token/position embedding tables are
    stored as raw FP32 in DRAM (see `tiny_fixture._fp32_embedding`)
    and this emitter loads `d_model_pad × 4` bytes per row so the
    next op (typically `tok_pos_add` lowered as VADD_FP32, then
    `ln1` lowered as LAYERNORM_FP32) reads real-units FP32 from
    ABUF. The pad-row zero-fill is skipped in W8A32 mode — its
    only consumer is the M3 attention QKT path which the W8A32
    guardrail still blocks; sub-layer ops produce row-independent
    outputs so the garbage stays out of valid rows.
    """
    table_name = node.attrs.get("table", default_table)
    if table_name not in cg.dram_layout:
        return
    seq_len, d_model = node.output_shape
    d_model_pad = pad_dim(d_model)
    if d_model_pad != d_model:
        raise ValueError("Stage 1 embedding lookup requires d_model to be 16-aligned")

    # M3-prep + M2-W8A16: token/position embeddings emit at FP precision
    # in W8A{32,16} mode (`self.elem_bytes` = 4 for fp32, 2 for fp16).
    # The DeiT path uses embedding_kind="patch_cls" and never sets
    # use_fp16_activations today (CodeGenerator init force-disables W8A8
    # opts for W8A32 but doesn't touch the DeiT compiler entry);
    # the embedding-kind guard makes the intent explicit.
    use_fp = (
        cg.use_fp16_activations
        and cg.config.embedding_kind == "token_pos"
    )
    elem_bytes = cg.elem_bytes if use_fp else 1
    row_bytes = d_model_pad * elem_bytes

    runtime_patch = bool(node.attrs.get("runtime_patch", False))
    if runtime_patch:
        row_indices = [0] * seq_len
    elif "row_indices" in node.attrs:
        row_indices = list(node.attrs["row_indices"])
    elif "token_ids" in node.attrs:
        row_indices = list(node.attrs["token_ids"])
    elif "position_ids" in node.attrs:
        row_indices = list(node.attrs["position_ids"])
    else:
        row_indices = list(range(seq_len)) if node.op == "pos_embed_lookup" else [0] * seq_len
    if len(row_indices) != seq_len:
        raise ValueError(f"{node.name} expected {seq_len} row indices, got {len(row_indices)}")

    out_alloc = cg.mem.abuf.alloc(node.name, pad_dim(seq_len) * row_bytes)
    table_dram = cg._dram_offset_required(table_name, f"loading embedding table '{table_name}'")
    row_units = row_bytes // UNIT
    for row_idx, table_row in enumerate(row_indices):
        runtime_kind = None
        runtime_base = None
        dram_addr = table_dram + int(table_row) * row_bytes
        if runtime_patch:
            runtime_kind = "token_embed" if node.op == "embed_lookup" else "pos_embed"
            runtime_base = table_name
            dram_addr = 0
        cg._emit_dma_load(
            BUF_ABUF,
            out_alloc.offset_units + row_idx * row_units,
            row_bytes,
            0,
            dram_addr,
            runtime_patch_kind=runtime_kind,
            runtime_base_symbol=runtime_base,
        )
        cg._emit(SyncInsn(resource_mask=0b001))
    pad_rows = pad_dim(seq_len) - seq_len
    # Both the INT8 and W8A32 paths now zero-fill pad rows (M3-prep
    # initially skipped this for W8A32 mode because attention was
    # still guardrailed; M3-C re-enables it because masked softmax in
    # the FP32 path requires K/V padding rows to be zero so
    # LN(zero_row)=β doesn't contaminate attention scores). The
    # `__zero_pad__` blob is sized 4× in W8A32 mode (see
    # `_layout_weights`), so `pad_rows * row_bytes` always fits.
    if pad_rows:
        zero_pad_dram = cg._dram_offset_required("__zero_pad__", "zeroing embedding padding rows")
        cg._emit_dma_load(
            BUF_ABUF,
            out_alloc.offset_units + seq_len * row_units,
            pad_rows * row_bytes,
            3,
            zero_pad_dram,
        )
        cg._emit(SyncInsn(resource_mask=0b001))
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        pad_dim(seq_len),
        d_model_pad,
        seq_len,
        d_model,
        "fp16" if use_fp else "int8",
        cg.calibration_scales.get(node.name, 6.0 / 127.0),
    )


def emit_cls_prepend(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit CLS token prepend: load CLS to ABUF[0], then DMA patches to rows 1-196."""
    if cg.config.embedding_kind != "patch_cls":
        raise ValueError("cls_prepend is only valid for patch_cls embeddings")
    cls_name = node.inputs[1]
    cls_dram = cg._dram_offset_required(cls_name, "loading cls token")
    # Load CLS token [1, d_model] to ABUF row 0.
    cg._emit_dma_load(BUF_ABUF, 0, cg.config.d_model, 0, cls_dram)
    cg._emit(SyncInsn(resource_mask=0b001))
    # DMA input patches from DRAM to ABUF rows 1-196.
    # Host writes INT8 patch embeddings [196, 192] to DRAM[input_offset] before run.
    # Row 1 starts at byte offset d_model in ABUF.
    patches_dram = cg.dram_layout["__input_patches__"]
    patches_bytes = (cg.config.max_seq_len - 1) * cg.config.d_model
    cg._emit_dma_load(BUF_ABUF, cg.config.d_model // UNIT, patches_bytes, 1, patches_dram)
    cg._emit(SyncInsn(resource_mask=0b001))
    # Mark allocation for the full [208, 192] padded sequence (rows 197-207 stay zero)
    cg.mem.abuf.alloc(node.name, pad_dim(cg.config.max_seq_len) * cg.config.d_model, evictable=False)


def emit_pos_embed_add(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit position embedding add."""
    if cg.config.embedding_kind != "patch_cls":
        raise ValueError("pos_embed_add is only valid for patch_cls embeddings")
    pos_name = node.inputs[1]
    pos_dram = cg._dram_offset_required(pos_name, "loading position embeddings")
    M_pad = pad_dim(cg.config.max_seq_len)
    N = cg.config.d_model
    N_pad = pad_dim(N)

    # Load pos_embed to WBUF [208, 192] (pre-padded at compile time)
    pos_bytes = M_pad * N_pad
    pos_alloc = cg.mem.wbuf.alloc("pos_embed", pos_bytes)
    cg._emit_dma_load(BUF_WBUF, pos_alloc.offset_units, pos_bytes, 0, pos_dram)
    cg._emit(SyncInsn(resource_mask=0b001))

    # CONFIG_TILE for VADD
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    cg._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

    act_alloc = cg.mem.abuf.get(node.inputs[0]) or \
                cg.mem.abuf.alloc(node.inputs[0], M_pad * N_pad)

    trace_scale = cg.calibration_scales.get(node.name, 14.0 / 127.0)
    # Trace both inputs at the pre-VADD PC so the first-divergence harness
    # can tell whether the bug is in runtime placement or in the helper op.
    cg._record_trace_event(
        f"{node.name}__act_input",
        BUF_ABUF,
        act_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "int8",
        trace_scale,
    )
    cg._record_trace_event(
        f"{node.name}__pos_input",
        BUF_WBUF,
        pos_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "int8",
        trace_scale,
    )

    # VADD: activations + pos_embed (both INT8, same scale)
    out_alloc = cg.mem.abuf.alloc(node.name, M_pad * N_pad)
    cg._emit(VaddInsn(
        src1_buf=BUF_ABUF, src1_off=act_alloc.offset_units,
        src2_buf=BUF_WBUF, src2_off=pos_alloc.offset_units,
        dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
    ))
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        M_pad,
        N_pad,
        node.output_shape[0],
        node.output_shape[1],
        "int8",
        trace_scale,
    )

    cg.mem.wbuf.free("pos_embed")


def emit_cls_extract(cg: "CodeGenerator", node: IRNode) -> None:
    """Extract CLS token (row 0) via BUF_COPY."""
    if cg.config.embedding_kind != "patch_cls":
        raise ValueError("cls_extract is only valid for patch_cls embeddings")
    N = cg.config.d_model
    in_alloc = cg.mem.abuf.get(node.inputs[0]) or \
               cg.mem.abuf.alloc(node.inputs[0], pad_dim(cg.config.max_seq_len) * pad_dim(N))
    out_alloc = cg.mem.abuf.alloc(node.name, pad_dim(N))
    # Copy 192 bytes = 12 × 16-byte units
    cg._emit(BufCopyInsn(
        src_buf=BUF_ABUF, src_off=in_alloc.offset_units,
        dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
        length=N // UNIT,
    ))
    cg._record_trace_event(
        node.name,
        BUF_ABUF,
        out_alloc.offset_units,
        1,
        N,
        1,
        N,
        "int8",
        cg.calibration_scales.get(node.name, 6.0 / 127.0),
    )
