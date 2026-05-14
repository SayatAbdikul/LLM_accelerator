"""Shared helpers for the W8A16 codegen package.

Three free functions plus the `UNIT = 16` constant. Used by every
sibling module (`sublayer`, `matmul`, `attention`):

  - `_fp16_to_uint16`  — convert FP32/python float to its FP16 bit pattern
    (uint16, little-endian). Used by attention-emit sites that bake
    SET_SCALE immediates for static composite scales.
  - `_zero_fill_fp32_padding_rows` — zero out FP{16,32} padding rows of
    an ABUF tile before MAX_ABS_REDUCE. Without it, padding rows pick
    up non-zero values from upstream (LN(zero)=beta, matmul broadcasts
    bias to every row) and inflate the dynamic activation scale,
    catastrophically degrading INT8 quantization for the valid rows.
  - `_abuf_alloc_fp32` — allocate (or reuse, by name) an ABUF region
    sized for an [M_pad × N_pad] FP-tile (cg.elem_bytes per element).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...isa.instructions import SyncInsn
from ...isa.opcodes import BUF_ABUF

if TYPE_CHECKING:
    from ..codegen import CodeGenerator


# Size of one 16-byte addressing unit (matches `taccel/isa/opcodes.UNIT`).
UNIT = 16


def _fp16_to_uint16(val) -> int:
    """Convert FP32/python float to its FP16 bit pattern (uint16, little-endian).

    Mirror of `codegen._fp16_to_uint16` — duplicated to keep w8a16_emit a
    self-contained import surface (codegen imports w8a16_emit, not the
    other way around).
    """
    fp16 = np.float16(val)
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


def _zero_fill_fp32_padding_rows(
    cg: "CodeGenerator",
    in_alloc,
    M: int,
    M_pad: int,
    K_pad: int,
) -> None:
    """M4-debug: zero out FP padding rows [M, M_pad) × K_pad of an ABUF tile.

    Why
    ---
    Hardware MAX_ABS_REDUCE_FP32 reads `M_pad × N_pad` elements based on
    CONFIG_TILE. CONFIG_TILE's M is in 16-row units, so we can't reduce
    below M=16. For decode/single-token prefill the valid row is just
    row 0 and rows 1..15 are padding. Without zero-fill, those padding
    rows accumulate non-zero values upstream:

      - emit_embedding_lookup zero-fills padding rows. ✓
      - LN(zero_row) = beta (the LN bias). ✗ output padding rows = beta.
      - VADD residual = matmul_output + previous_residual.
        matmul_output's padding rows = matmul_bias (because input is 0,
        but bias is added to every row). ✗
      - Subsequent matmul reads `MaxAbsReduce` over the full padded
        tile → max_abs is inflated by padding-row magnitudes → INT8
        quantization scale is wrong → valid row uses fewer INT8 levels
        → output noisy. Compounds across 12 layers → catastrophic PPL.

    Fix: at every dynamic-quant matmul (Q/K/V projections, out_proj,
    fc1, fc2, lm_head), zero-fill input padding rows just before
    MAX_ABS_REDUCE so the max is over the valid query rows only.

    DMA source is the enlarged `__zero_pad__` blob (codegen.py
    `_layout_weights` enlarges it for W8A16 to (TILE-1) × max_k_pad × 2
    bytes — covers the worst case across all weights).

    Skips when `M >= M_pad` (no padding to fill).
    """
    if M >= M_pad:
        return
    rows_to_zero = M_pad - M
    row_bytes = K_pad * cg.elem_bytes
    bytes_to_zero = rows_to_zero * row_bytes
    padding_start_unit = in_alloc.offset_units + (M * row_bytes) // UNIT
    zero_dram = cg._dram_offset_required(
        "__zero_pad__",
        "loading FP32 zero blob for padding-row mask before MAX_ABS_REDUCE",
    )
    cg._emit_dma_load(BUF_ABUF, padding_start_unit, bytes_to_zero, 0, zero_dram)
    cg._emit(SyncInsn(resource_mask=0b001))


def _abuf_alloc_fp32(cg: "CodeGenerator", name: str, M_pad: int, N_pad: int):
    """Allocate an ABUF region sized for an [M_pad, N_pad] FP-precision tile.

    Returns the existing allocation if one is already keyed by `name`
    (i.e. an inplace re-use), otherwise allocates fresh. Sizing uses
    `cg.elem_bytes` (W8A16 = 2 bytes/element; legacy W8A32 = 4).
    """
    existing = cg.mem.abuf.get(name)
    if existing is not None:
        return existing
    return cg.mem.abuf.alloc(name, M_pad * N_pad * cg.elem_bytes)
