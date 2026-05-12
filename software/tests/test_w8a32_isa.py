"""W8A32 ISA extension (Phase 3 (c.1), milestone M1) — encoding tests.

The W8A32 path adds 7 new R-type opcodes (0x17-0x1D) for FP32-I/O
sub-layer ops alongside the existing INT8 ones. This test suite covers:

  1. Each new opcode is registered in `OPCODE_FORMAT` as R_TYPE.
  2. Each new instruction class encodes to 8 bytes and decodes back to
     the same class with the same field values (round-trip).
  3. The dispatch class table in `_R_TYPE_CLASSES` covers every new
     opcode (regression: decoding a valid opcode must not return an
     unrelated class).
  4. The reserved range narrowed correctly — opcodes 0x1E and 0x1F
     still raise illegal-opcode at decode time.

The simulator-level dispatch and FP32 memory helpers are exercised in
`test_w8a32_simulator.py`.
"""
from __future__ import annotations

import pytest

from taccel.isa.encoding import _R_TYPE_CLASSES, decode, encode
from taccel.isa.instructions import (
    DequantAccumFp32Insn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    QuantFp32Int8Insn,
    SoftmaxFp32Insn,
    VaddFp32Insn,
)
from taccel.isa.opcodes import (
    BUF_ABUF,
    BUF_ACCUM,
    BUF_WBUF,
    InsnFormat,
    OPCODE_FORMAT,
    Opcode,
)


W8A32_OPCODES = (
    Opcode.DEQUANT_ACCUM_FP32,
    Opcode.QUANT_FP32_INT8,
    Opcode.VADD_FP32,
    Opcode.LAYERNORM_FP32,
    Opcode.GELU_FP32,
    Opcode.SOFTMAX_FP32,
    Opcode.MASKED_SOFTMAX_FP32,
)


W8A32_CLASSES = (
    (Opcode.DEQUANT_ACCUM_FP32, DequantAccumFp32Insn),
    (Opcode.QUANT_FP32_INT8, QuantFp32Int8Insn),
    (Opcode.VADD_FP32, VaddFp32Insn),
    (Opcode.LAYERNORM_FP32, LayernormFp32Insn),
    (Opcode.GELU_FP32, GeluFp32Insn),
    (Opcode.SOFTMAX_FP32, SoftmaxFp32Insn),
    (Opcode.MASKED_SOFTMAX_FP32, MaskedSoftmaxFp32Insn),
)


def test_w8a32_opcodes_are_contiguous_0x17_to_0x1d():
    """The seven new W8A32 opcodes consume reserved slots 0x17-0x1D in
    contiguous order, with 0x1E-0x1F left reserved."""
    expected_values = [0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D]
    actual_values = [int(op) for op in W8A32_OPCODES]
    assert actual_values == expected_values


@pytest.mark.parametrize("op", W8A32_OPCODES)
def test_w8a32_opcode_is_r_type(op):
    assert OPCODE_FORMAT[op] == InsnFormat.R_TYPE


@pytest.mark.parametrize("op,cls", W8A32_CLASSES)
def test_w8a32_class_registered_in_decoder(op, cls):
    assert _R_TYPE_CLASSES[op] is cls


@pytest.mark.parametrize("cls", [pair[1] for pair in W8A32_CLASSES])
def test_w8a32_instruction_encode_decode_roundtrip(cls):
    """Encode an instruction with non-default field values, decode it
    back, and verify all fields survive the round-trip exactly."""
    insn = cls(
        src1_buf=BUF_ABUF,
        src1_off=37,
        src2_buf=BUF_WBUF,
        src2_off=1234,
        dst_buf=BUF_ABUF,
        dst_off=5678,
        sreg=11,
        flags=1,
    )
    blob = encode(insn)
    assert len(blob) == 8
    decoded = decode(blob)
    assert type(decoded) is cls
    assert decoded.src1_buf == BUF_ABUF
    assert decoded.src1_off == 37
    assert decoded.src2_buf == BUF_WBUF
    assert decoded.src2_off == 1234
    assert decoded.dst_buf == BUF_ABUF
    assert decoded.dst_off == 5678
    assert decoded.sreg == 11
    assert decoded.flags == 1


def test_dequant_accum_fp32_with_accum_src1_encodes_and_decodes():
    """DEQUANT_ACCUM_FP32 specifically requires src1=ACCUM; verify that's
    a valid round-trip (the R-type encoding accepts any of the three
    legal buffer IDs; ACCUM enforcement happens in the simulator)."""
    insn = DequantAccumFp32Insn(
        src1_buf=BUF_ACCUM,
        src1_off=0,
        src2_buf=BUF_WBUF,
        src2_off=42,
        dst_buf=BUF_ABUF,
        dst_off=99,
        sreg=0,
        flags=0,
    )
    decoded = decode(encode(insn))
    assert decoded.src1_buf == BUF_ACCUM
    assert decoded.src2_buf == BUF_WBUF
    assert decoded.dst_buf == BUF_ABUF


@pytest.mark.parametrize("reserved_opcode_value", [0x1E, 0x1F])
def test_reserved_opcodes_after_m1_still_illegal(reserved_opcode_value):
    """After M1 the reserved range narrows from 0x17-0x1F to 0x1E-0x1F.
    Decoding either remaining reserved opcode must still raise."""
    # Build a raw 8-byte instruction with opcode at bits [63:59].
    word = reserved_opcode_value << 59
    raw = word.to_bytes(8, byteorder="big")
    with pytest.raises(ValueError):
        decode(raw)


def test_pre_m1_opcodes_unchanged():
    """Regression — every pre-M1 opcode (0x00-0x16) still maps to the
    same InsnFormat as before this milestone, so we didn't accidentally
    shift the table."""
    pre_m1_expected = {
        Opcode.NOP: InsnFormat.S_TYPE,
        Opcode.HALT: InsnFormat.S_TYPE,
        Opcode.SYNC: InsnFormat.S_TYPE,
        Opcode.CONFIG_TILE: InsnFormat.C_TYPE,
        Opcode.SET_SCALE: InsnFormat.S_TYPE,
        Opcode.SET_ADDR_LO: InsnFormat.A_TYPE,
        Opcode.SET_ADDR_HI: InsnFormat.A_TYPE,
        Opcode.LOAD: InsnFormat.M_TYPE,
        Opcode.STORE: InsnFormat.M_TYPE,
        Opcode.BUF_COPY: InsnFormat.B_TYPE,
        Opcode.MATMUL: InsnFormat.R_TYPE,
        Opcode.REQUANT: InsnFormat.R_TYPE,
        Opcode.SCALE_MUL: InsnFormat.R_TYPE,
        Opcode.VADD: InsnFormat.R_TYPE,
        Opcode.SOFTMAX: InsnFormat.R_TYPE,
        Opcode.LAYERNORM: InsnFormat.R_TYPE,
        Opcode.GELU: InsnFormat.R_TYPE,
        Opcode.REQUANT_PC: InsnFormat.R_TYPE,
        Opcode.SOFTMAX_ATTNV: InsnFormat.R_TYPE,
        Opcode.DEQUANT_ADD: InsnFormat.R_TYPE,
        Opcode.CONFIG_ATTN: InsnFormat.ATTN_TYPE,
        Opcode.MASKED_SOFTMAX: InsnFormat.R_TYPE,
        Opcode.MASKED_SOFTMAX_ATTNV: InsnFormat.R_TYPE,
    }
    for op, fmt in pre_m1_expected.items():
        assert OPCODE_FORMAT[op] == fmt, f"{op.name}: format changed"
