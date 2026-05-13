"""W8A32 ISA extension (Phase 3 (c.1), milestones M1 + M2.5-A) — encoding tests.

The W8A32 path adds 9 new R-type opcodes that consume the previously
reserved slots 0x17-0x1F:

  M1 (0x17-0x1D): FP32-I/O sub-layer ops (dequant, quant, vadd, LN, GELU,
  softmax, masked softmax) alongside the existing INT8 ones.
  M2.5-A (0x1E-0x1F): dynamic per-matmul activation scaling primitives
  (DEQUANT_ACCUM_FP32_SCALED and MAX_ABS_REDUCE_FP32). After M2.5-A the
  entire 5-bit opcode space is in use — no reserved slots remain.

This test suite covers:

  1. Each new opcode is registered in `OPCODE_FORMAT` as R_TYPE.
  2. Each new instruction class encodes to 8 bytes and decodes back to
     the same class with the same field values (round-trip).
  3. The dispatch class table in `_R_TYPE_CLASSES` covers every new
     opcode (regression: decoding a valid opcode must not return an
     unrelated class).
  4. The 5-bit opcode space is fully assigned: every value 0x00-0x1F
     decodes to an `Opcode` whose value matches the encoded bits.

The simulator-level dispatch and FP32 memory helpers are exercised in
`test_w8a32_simulator.py`.
"""
from __future__ import annotations

import pytest

from taccel.isa.encoding import _R_TYPE_CLASSES, decode, encode
from taccel.isa.instructions import (
    DequantAccumFp32Insn,
    DequantAccumFp32ScaledInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    MaxAbsReduceFp32Insn,
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


# --- M1 opcodes (0x17-0x1D) ------------------------------------------------
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


# --- M2.5-A opcodes (0x1E-0x1F) --------------------------------------------
M2_5A_OPCODES = (
    Opcode.DEQUANT_ACCUM_FP32_SCALED,
    Opcode.MAX_ABS_REDUCE_FP32,
)


M2_5A_CLASSES = (
    (Opcode.DEQUANT_ACCUM_FP32_SCALED, DequantAccumFp32ScaledInsn),
    (Opcode.MAX_ABS_REDUCE_FP32, MaxAbsReduceFp32Insn),
)


# Convenience aggregate for tests that don't care which milestone a given
# opcode came from.
ALL_NEW_OPCODES = W8A32_OPCODES + M2_5A_OPCODES
ALL_NEW_CLASSES = W8A32_CLASSES + M2_5A_CLASSES


def test_w8a32_opcodes_are_contiguous_0x17_to_0x1d():
    """The seven M1 W8A32 opcodes consume reserved slots 0x17-0x1D in
    contiguous order. (M2.5-A extends to 0x1E-0x1F — covered by its own
    test below.)"""
    expected_values = [0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D]
    actual_values = [int(op) for op in W8A32_OPCODES]
    assert actual_values == expected_values


def test_m2_5a_opcodes_are_contiguous_0x1e_to_0x1f():
    """The two M2.5-A opcodes consume the last reserved slots 0x1E-0x1F.
    After this commit the entire 5-bit opcode space is in use."""
    expected_values = [0x1E, 0x1F]
    actual_values = [int(op) for op in M2_5A_OPCODES]
    assert actual_values == expected_values


@pytest.mark.parametrize("op", ALL_NEW_OPCODES)
def test_w8a32_opcode_is_r_type(op):
    assert OPCODE_FORMAT[op] == InsnFormat.R_TYPE


@pytest.mark.parametrize("op,cls", ALL_NEW_CLASSES)
def test_w8a32_class_registered_in_decoder(op, cls):
    assert _R_TYPE_CLASSES[op] is cls


@pytest.mark.parametrize("cls", [pair[1] for pair in ALL_NEW_CLASSES])
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


@pytest.mark.parametrize("opcode_value", list(range(32)))
def test_every_5_bit_opcode_decodes_after_m2_5a(opcode_value):
    """After M2.5-A no opcodes are reserved — every value 0x00-0x1F must
    decode to an `Opcode` instance whose `.opcode` attribute matches the
    encoded bits. This is the positive replacement for the post-M1
    reserved-range test (`test_reserved_opcodes_after_m1_still_illegal`,
    deleted now that 0x1E and 0x1F are real opcodes).

    Constructing a raw word with `opcode << 59` and all other bits zero
    is a valid encoding for every format family: R-TYPE buffers all
    decode to BUF_ABUF=0 with zero offsets, CONFIG_ATTN's reserved-bits
    check passes (all zeros), S-TYPE NOP/HALT/SYNC/SET_SCALE have no
    decode-time constraints, etc.
    """
    # 1) IntEnum lookup — every 5-bit value must be a known Opcode member.
    op = Opcode(opcode_value)
    assert int(op) == opcode_value

    # 2) Every Opcode must have a registered InsnFormat.
    assert op in OPCODE_FORMAT, (
        f"Opcode {op.name} (0x{opcode_value:02X}) is not in OPCODE_FORMAT"
    )

    # 3) The full decode pipeline must succeed and return an instruction
    #    whose `.opcode` field matches the encoded bits — the post-M2.5-A
    #    "no reserved slots" invariant.
    raw = (opcode_value << 59).to_bytes(8, byteorder="big")
    decoded = decode(raw)
    assert int(decoded.opcode) == opcode_value, (
        f"decode produced opcode {int(decoded.opcode):#x} for raw 0x{opcode_value:02X}"
    )


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
