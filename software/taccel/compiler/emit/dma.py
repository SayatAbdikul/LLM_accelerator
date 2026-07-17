"""DMA-emit helpers: SET_ADDR + LOAD / STORE sequences.

Both functions are pure delegations of the old `CodeGenerator._emit_dma_load`
and `CodeGenerator._emit_dma_store` methods; signatures and semantics are
preserved byte-for-byte. They are called via the thin dispatcher methods
in `codegen.py` (`self._emit_dma_load(...)` / `self._emit_dma_store(...)`)
so existing callers — internal and external — are unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...isa.instructions import LoadInsn, StoreInsn
from ._common import UNIT, _set_addr

# M-type XFER_LEN is a 16-bit field. `MTypeInsn.__post_init__` validates
# 0 <= xfer_len <= 0xFFFF and raises — but both emit sites used to pre-clamp
# with `min(xfer_units, 0xFFFF)` BEFORE constructing the instruction, which made
# that validator structurally unreachable (the same blind-gate class the project
# has been bitten by twice). An over-long transfer then became a SILENT PARTIAL
# transfer, leaving the tail of the destination buffer stale, with nothing
# raising anywhere. Raise instead: an oversized DMA is a compiler bug.
#
# Not reachable today — a DMA is bounded by its buffer (WBUF 256 KB = 16384
# units << 65535) and the only >1 MB candidate, the b16/b32 124M logits store
# (1,608,704 B), takes the chunked branch in emit/kv.py. Contrast `dram_off`,
# which was always passed unclamped and correctly raises on overflow.
_MAX_XFER_UNITS = 0xFFFF


def _checked_xfer_units(size_bytes: int, kind: str) -> int:
    """Size in 16-byte units, or raise if it cannot be encoded."""
    xfer_units = (size_bytes + UNIT - 1) // UNIT
    if xfer_units > _MAX_XFER_UNITS:
        raise ValueError(
            f"{kind} transfer of {size_bytes} bytes needs xfer_len={xfer_units} "
            f"units, which exceeds the 16-bit M-type field ({_MAX_XFER_UNITS}). "
            f"Split the transfer (see emit/kv.py's chunked logits store) — "
            f"silently truncating it would leave the buffer tail stale."
        )
    return xfer_units

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def emit_dma_load(cg: "CodeGenerator", buf_id: int, sram_off_units: int,
                  size_bytes: int, addr_reg: int, dram_byte_offset: int, *,
                  dram_off_units: int = 0,
                  relocation_symbol: Optional[str] = None,
                  runtime_patch_kind: Optional[str] = None,
                  runtime_base_symbol: Optional[str] = None,
                  transpose: int = 0,
                  cols_log2: int = 0) -> None:
    """Emit SET_ADDR + LOAD sequence.

    Lever D: with ``transpose=1`` the LOAD reads the contiguous (R, C) INT8
    region and writes its (C, R) transpose to SRAM (``C = 16 << cols_log2``),
    replacing a separate BUF_COPY(transpose=1) helper pass.
    """
    lo_pc = len(cg.instructions)
    cg.instructions.extend(_set_addr(addr_reg, dram_byte_offset))
    if relocation_symbol is None and runtime_patch_kind is None:
        relocation_symbol = "data_base"
    cg._record_addr_site(
        lo_pc,
        lo_pc + 1,
        addr_reg,
        relocation_symbol=relocation_symbol,
        runtime_patch_kind=runtime_patch_kind,
        runtime_base_symbol=runtime_base_symbol,
    )
    xfer_units = _checked_xfer_units(size_bytes, "LOAD")
    cg._emit(LoadInsn(
        buf_id=buf_id,
        sram_off=sram_off_units,
        xfer_len=xfer_units,
        addr_reg=addr_reg,
        dram_off=dram_off_units,
        transpose=transpose,
        cols_log2=cols_log2,
    ))


def emit_dma_store(cg: "CodeGenerator", buf_id: int, sram_off_units: int,
                   size_bytes: int, addr_reg: int, dram_byte_offset: int, *,
                   dram_off_units: int = 0,
                   relocation_symbol: Optional[str] = None,
                   runtime_patch_kind: Optional[str] = None,
                   runtime_base_symbol: Optional[str] = None) -> None:
    """Emit SET_ADDR + STORE sequence."""
    lo_pc = len(cg.instructions)
    cg.instructions.extend(_set_addr(addr_reg, dram_byte_offset))
    if relocation_symbol is None and runtime_patch_kind is None:
        relocation_symbol = "data_base"
    cg._record_addr_site(
        lo_pc,
        lo_pc + 1,
        addr_reg,
        relocation_symbol=relocation_symbol,
        runtime_patch_kind=runtime_patch_kind,
        runtime_base_symbol=runtime_base_symbol,
    )
    xfer_units = _checked_xfer_units(size_bytes, "STORE")
    cg._emit(StoreInsn(
        buf_id=buf_id,
        sram_off=sram_off_units,
        xfer_len=xfer_units,
        addr_reg=addr_reg,
        dram_off=dram_off_units,
    ))
