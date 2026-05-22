"""Shared helpers for the per-op emit package.

This module mirrors the role of `compiler/w8a16_emit/_common.py`:
free-function utilities, a `UNIT = 16` constant, and TYPE_CHECKING-only
imports of `CodeGenerator` so per-op emit modules can be split out of
`codegen.py` without circular-import gymnastics.

The constants/helpers below were previously defined at module scope in
`codegen.py`. They live here now because every sibling emit module
(`dma`, `matmul`, `sfu`, `embedding`, `kv`, `attn`) needs at least one
of them; `codegen.py` re-exports them for backward compatibility with
any external consumer that imported them by path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from ...isa.instructions import Instruction, SetAddrLoInsn, SetAddrHiInsn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator  # noqa: F401


# Size of one 16-byte addressing unit. Matches the W8A16 sibling
# (`compiler/w8a16_emit/_common.UNIT`) and the LOAD/STORE wire-format
# `xfer_len` units used by every DMA-emitting site.
UNIT = 16


def _fp16_to_uint16(val: float) -> int:
    """Convert FP32 value to FP16 bit pattern as uint16 (little-endian)."""
    fp16 = np.float16(val)
    # tobytes() on little-endian system gives LE bytes; interpret as uint16
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


def _set_addr(addr_reg: int, byte_addr: int) -> List[Instruction]:
    """Emit SET_ADDR_LO + SET_ADDR_HI to set a 56-bit DRAM address."""
    lo = byte_addr & 0xFFFFFFF
    hi = (byte_addr >> 28) & 0xFFFFFFF
    return [
        SetAddrLoInsn(addr_reg=addr_reg, imm28=lo),
        SetAddrHiInsn(addr_reg=addr_reg, imm28=hi),
    ]
