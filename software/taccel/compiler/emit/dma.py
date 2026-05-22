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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def emit_dma_load(cg: "CodeGenerator", buf_id: int, sram_off_units: int,
                  size_bytes: int, addr_reg: int, dram_byte_offset: int, *,
                  dram_off_units: int = 0,
                  relocation_symbol: Optional[str] = None,
                  runtime_patch_kind: Optional[str] = None,
                  runtime_base_symbol: Optional[str] = None) -> None:
    """Emit SET_ADDR + LOAD sequence."""
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
    xfer_units = (size_bytes + UNIT - 1) // UNIT
    cg._emit(LoadInsn(
        buf_id=buf_id,
        sram_off=sram_off_units,
        xfer_len=min(xfer_units, 0xFFFF),
        addr_reg=addr_reg,
        dram_off=dram_off_units,
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
    xfer_units = (size_bytes + UNIT - 1) // UNIT
    cg._emit(StoreInsn(
        buf_id=buf_id,
        sram_off=sram_off_units,
        xfer_len=min(xfer_units, 0xFFFF),
        addr_reg=addr_reg,
        dram_off=dram_off_units,
    ))
