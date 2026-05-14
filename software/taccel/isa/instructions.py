"""Instruction dataclasses for all ISA instruction types."""
from dataclasses import dataclass, field
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
class MaskedSoftmaxInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.MASKED_SOFTMAX, init=False)


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
class MaskedSoftmaxAttnVInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.MASKED_SOFTMAX_ATTNV, init=False)


@dataclass
class DequantAddInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.DEQUANT_ADD, init=False)


# --- W8A32 R-type instructions (Phase 3 (c.1), milestone M1) ---
#
# These mirror the existing R-type contract (same field layout, same
# encoding). The only difference is dtype interpretation:
#   - DEQUANT_ACCUM_FP32: src1=ACCUM (INT32), src2=WBUF or sreg (FP16
#     per-channel scale table or scalar), dst=ABUF (FP32 written, 4 bytes/elem)
#   - QUANT_FP32_INT8:    src1=ABUF (FP32 read), src2=unused, dst=ABUF (INT8 written)
#   - VADD_FP32:          src1=ABUF (FP32), src2=ABUF (FP32), dst=ABUF (FP32)
#   - LAYERNORM_FP32:     src1=ABUF (FP32 input), src2=WBUF (FP16 gamma+beta),
#                          dst=ABUF (FP32 output)
#   - GELU_FP32:          src1=ABUF (FP32), src2=unused, dst=ABUF (FP32)
#   - SOFTMAX_FP32:       src1=ABUF (FP32), src2=unused, dst=ABUF (FP32)
#   - MASKED_SOFTMAX_FP32: src1=ABUF (FP32), src2=unused, dst=ABUF (FP32);
#                          consumes CONFIG_ATTN context like MASKED_SOFTMAX
#
# W8A16 extension (Phase 3 (c.2), milestone M1, 2026-05-14)
# ---------------------------------------------------------
# The 9 W8A32 R-type opcodes (0x17-0x1F) are reused with `flags[0]` as
# an fp_precision selector:
#   - flags=0  -> FP32 storage (existing W8A32 behavior, bit-identical)
#   - flags=1  -> FP16 storage (new W8A16 default)
# Under flags=1 the ABUF I/O is 2 bytes/elem (vs 4 for FP32); internal
# datapath stays FP32 for numerically sensitive math (LN variance,
# softmax exp, GELU x^3). For DEQUANT_ACCUM_FP32_SCALED specifically,
# flags=1 also changes src2 layout: it carries `2N FP16` (N PC scales
# followed by N bias values), and the epilogue computes
# `fp32 = int32 * pc * act_scale + bias` then casts to FP16 once.
# This avoids FP16 double-rounding bias through a separate VADD. Under
# flags=0 the src2 layout is unchanged (N FP16 PC scales only).


@dataclass
class DequantAccumFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.DEQUANT_ACCUM_FP32, init=False)


@dataclass
class QuantFp32Int8Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.QUANT_FP32_INT8, init=False)


@dataclass
class VaddFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.VADD_FP32, init=False)


@dataclass
class LayernormFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.LAYERNORM_FP32, init=False)


@dataclass
class GeluFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.GELU_FP32, init=False)


@dataclass
class SoftmaxFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.SOFTMAX_FP32, init=False)


@dataclass
class MaskedSoftmaxFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.MASKED_SOFTMAX_FP32, init=False)


# --- W8A32 R-type instructions (M2.5-A, dynamic activation scale primitives) ---
#
# DEQUANT_ACCUM_FP32_SCALED is the M2.5 variant of DEQUANT_ACCUM_FP32 — same
# inputs but additionally multiplies by an FP16 scalar from scale_regs[sreg].
# The scalar is the activation scale (max_abs/127) emitted by the matmul
# prelude's MAX_ABS_REDUCE_FP32. M1's DEQUANT_ACCUM_FP32 (0x17) stays
# bit-identical to its shipped contract — this is a NEW opcode (0x1E), not
# a modification of the existing one.
#
# MAX_ABS_REDUCE_FP32 scans an FP32 ABUF tile, computes max(|x|), and
# writes derived FP16 scales to a register pair:
#   scale_regs[sreg]   = 127.0 / max_abs       (QUANT input scale)
#   scale_regs[sreg+1] = max_abs / 127.0       (DEQUANT activation scale)
# Eps-guarded: if max_abs == 0, the eps clamp (1e-10) keeps scale_regs[sreg]
# finite. The sreg+1 store requires `0 <= sreg <= 14` (i.e. sreg+1 <= 15),
# enforced at execution time like DEQUANT_ADD's sreg-pair contract.


@dataclass
class DequantAccumFp32ScaledInsn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.DEQUANT_ACCUM_FP32_SCALED, init=False)


@dataclass
class MaxAbsReduceFp32Insn(RTypeInsn):
    opcode: Opcode = field(default=Opcode.MAX_ABS_REDUCE_FP32, init=False)


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


# --- ATTN-type instruction ---
@dataclass
class ConfigAttnInsn(Instruction):
    opcode: Opcode = field(default=Opcode.CONFIG_ATTN, init=False)
    query_row_base: int = 0
    valid_kv_len: int = 0
    mode: int = 0

    def __post_init__(self):
        if not (0 <= self.query_row_base <= 0xFFF):
            raise ValueError(f"query_row_base must be 0-4095, got {self.query_row_base}")
        if not (0 <= self.valid_kv_len <= 0xFFF):
            raise ValueError(f"valid_kv_len must be 0-4095, got {self.valid_kv_len}")
        if not (0 <= self.mode <= 0x3):
            raise ValueError(f"mode must be 0-3, got {self.mode}")


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
