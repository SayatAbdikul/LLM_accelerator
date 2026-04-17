"""Instruction dataclasses for all ISA instruction types."""
from dataclasses import dataclass, field
from typing import Optional
from .opcodes import (
    Opcode, BUFFER_MAX_OFF, BUF_ABUF, BUF_WBUF, BUF_ACCUM, BUF_RESERVED,
)


def _validate_buf(buf_id: int, name: str = "buf"):
    if buf_id not in (BUF_ABUF, BUF_WBUF, BUF_ACCUM):
        raise ValueError(f"{name} must be 0 (ABUF), 1 (WBUF), or 2 (ACCUM), got {buf_id}")


def _validate_offset(buf_id: int, offset: int, name: str = "offset"):
    if offset < 0:
        raise ValueError(f"{name} must be non-negative, got {offset}")
    max_off = BUFFER_MAX_OFF.get(buf_id)
    if max_off is not None and offset > max_off:
        raise ValueError(f"{name}={offset} exceeds max {max_off} for buffer {buf_id}")


@dataclass
class Instruction:
    opcode: Opcode


# --- R-type instructions ---
@dataclass
class RTypeInsn(Instruction):
    src1_buf: int = 0
    src1_off: int = 0
    src2_buf: int = 0
    src2_off: int = 0
    dst_buf: int = 0
    dst_off: int = 0
    sreg: int = 0
    flags: int = 0

    def __post_init__(self):
        _validate_buf(self.src1_buf, "src1_buf")
        _validate_offset(self.src1_buf, self.src1_off, "src1_off")
        _validate_buf(self.src2_buf, "src2_buf")
        _validate_offset(self.src2_buf, self.src2_off, "src2_off")
        _validate_buf(self.dst_buf, "dst_buf")
        _validate_offset(self.dst_buf, self.dst_off, "dst_off")
        if not (0 <= self.sreg <= 15):
            raise ValueError(f"sreg must be 0-15, got {self.sreg}")
        if not (0 <= self.flags <= 1):
            raise ValueError(f"flags must be 0 or 1, got {self.flags}")


@dataclass
class MatmulInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.MATMUL, init=False)


@dataclass
class RequantInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.REQUANT, init=False)


@dataclass
class RequantPcInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.REQUANT_PC, init=False)


@dataclass
class ScaleMulInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.SCALE_MUL, init=False)


@dataclass
class VaddInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.VADD, init=False)


@dataclass
class SoftmaxInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.SOFTMAX, init=False)


@dataclass
class LayernormInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.LAYERNORM, init=False)


@dataclass
class GeluInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.GELU, init=False)


@dataclass
class SoftmaxAttnVInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.SOFTMAX_ATTNV, init=False)


@dataclass
class DequantAddInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.DEQUANT_ADD, init=False)


# --- M-type instructions ---
@dataclass
class MTypeInsn(Instruction):
    buf_id: int = 0
    sram_off: int = 0
    xfer_len: int = 0
    addr_reg: int = 0
    dram_off: int = 0
    stride_log2: int = 0
    flags: int = 0

    def __post_init__(self):
        _validate_buf(self.buf_id, "buf_id")
        _validate_offset(self.buf_id, self.sram_off, "sram_off")
        if not (0 <= self.xfer_len <= 0xFFFF):
            raise ValueError(f"xfer_len must be 0-65535, got {self.xfer_len}")
        if not (0 <= self.addr_reg <= 3):
            raise ValueError(f"addr_reg must be 0-3, got {self.addr_reg}")
        if not (0 <= self.dram_off <= 0xFFFF):
            raise ValueError(f"dram_off must be 0-65535, got {self.dram_off}")
        if not (0 <= self.stride_log2 <= 15):
            raise ValueError(f"stride_log2 must be 0-15, got {self.stride_log2}")
        if not (0 <= self.flags <= 7):
            raise ValueError(f"flags must be 0-7, got {self.flags}")


@dataclass
class LoadInsn(MTypeInsn):
    opcode: Opcode = field(default=Opcode.LOAD, init=False)


@dataclass
class StoreInsn(MTypeInsn):
    opcode: Opcode = field(default=Opcode.STORE, init=False)


# --- B-type instruction ---
@dataclass
class BufCopyInsn(Instruction):
    opcode: Opcode = field(default=Opcode.BUF_COPY, init=False)
    src_buf: int = 0
    src_off: int = 0
    dst_buf: int = 0
    dst_off: int = 0
    length: int = 0
    src_rows: int = 0
    transpose: int = 0

    def __post_init__(self):
        _validate_buf(self.src_buf, "src_buf")
        _validate_offset(self.src_buf, self.src_off, "src_off")
        _validate_buf(self.dst_buf, "dst_buf")
        _validate_offset(self.dst_buf, self.dst_off, "dst_off")
        if not (0 <= self.length <= 0xFFFF):
            raise ValueError(f"length must be 0-65535, got {self.length}")
        if not (0 <= self.src_rows <= 63):
            raise ValueError(f"src_rows must be 0-63, got {self.src_rows}")
        if not (0 <= self.transpose <= 1):
            raise ValueError(f"transpose must be 0 or 1, got {self.transpose}")


# --- A-type instructions ---
@dataclass
class ATypeInsn(Instruction):
    addr_reg: int = 0
    imm28: int = 0

    def __post_init__(self):
        if not (0 <= self.addr_reg <= 3):
            raise ValueError(f"addr_reg must be 0-3, got {self.addr_reg}")
        if not (0 <= self.imm28 <= 0xFFFFFFF):
            raise ValueError(f"imm28 must be 0-0xFFFFFFF, got {self.imm28}")


@dataclass
class SetAddrLoInsn(ATypeInsn):
    opcode: Opcode = field(default=Opcode.SET_ADDR_LO, init=False)


@dataclass
class SetAddrHiInsn(ATypeInsn):
    opcode: Opcode = field(default=Opcode.SET_ADDR_HI, init=False)


# --- C-type instruction ---
@dataclass
class ConfigTileInsn(Instruction):
    opcode: Opcode = field(default=Opcode.CONFIG_TILE, init=False)
    M: int = 0  # tile count (0-based encoded: value V means V+1 tiles)
    N: int = 0
    K: int = 0

    def __post_init__(self):
        for name, val in [("M", self.M), ("N", self.N), ("K", self.K)]:
            if not (0 <= val <= 1023):
                raise ValueError(f"{name} must be 0-1023 (encoded), got {val}")


# --- S-type instructions ---
@dataclass
class SetScaleInsn(Instruction):
    opcode: Opcode = field(default=Opcode.SET_SCALE, init=False)
    sreg: int = 0
    src_mode: int = 0  # 00=imm, 01=ABUF, 10=WBUF, 11=ACCUM
    imm16: int = 0     # FP16 immediate or buffer offset

    def __post_init__(self):
        if not (0 <= self.sreg <= 15):
            raise ValueError(f"sreg must be 0-15, got {self.sreg}")
        if not (0 <= self.src_mode <= 3):
            raise ValueError(f"src_mode must be 0-3, got {self.src_mode}")
        if not (0 <= self.imm16 <= 0xFFFF):
            raise ValueError(f"imm16 must be 0-0xFFFF, got {self.imm16}")


@dataclass
class SyncInsn(Instruction):
    opcode: Opcode = field(default=Opcode.SYNC, init=False)
    resource_mask: int = 0  # 3-bit: bit0=DMA, bit1=systolic, bit2=SFU

    def __post_init__(self):
        if not (0 <= self.resource_mask <= 7):
            raise ValueError(f"resource_mask must be 0-7, got {self.resource_mask}")


@dataclass
class NopInsn(Instruction):
    opcode: Opcode = field(default=Opcode.NOP, init=False)


@dataclass
class HaltInsn(Instruction):
    opcode: Opcode = field(default=Opcode.HALT, init=False)
