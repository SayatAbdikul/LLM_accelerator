"""W8A16 ISA extension (Phase 3 (c.2), milestone M1) — flag-bit encoding tests.

The W8A16 path reuses the 9 W8A32 R-type opcodes (0x17-0x1F) with
`RTypeInsn.flags[0]` repurposed as an fp_precision selector:

  - flags=0  → FP32 storage (W8A32, existing, bit-identical behavior)
  - flags=1  → FP16 storage (W8A16, new default)

No new opcodes are added; the 5-bit opcode field stays full. This test
suite verifies:

  1. Each of the 9 W8A32 R-type instruction classes round-trips with
     `flags=1` (encode → decode → same class, same fields, flags=1
     preserved).
  2. The `flags` field at bit 0 of the 64-bit instruction word is the
     LSB and round-trips faithfully for both 0 and 1.
  3. `RTypeInsn._validate` rejects `flags >= 2` — only 0 and 1 are legal.
  4. Setting `flags=1` does NOT change the opcode value (the dispatch
     class table still maps the same opcode to the same class).

The simulator-level FP16 behavior is exercised in `test_w8a16_simulator.py`.
"""
from __future__ import annotations

import pytest

from taccel.isa.encoding import decode, encode
from taccel.isa.instructions import (
    DequantAccumFp32Insn,
    DequantAccumFp32ScaledInsn,
    GeluFp32Insn,
    LayernormFp32Insn,
    MaskedSoftmaxFp32Insn,
    MaxAbsReduceFp32Insn,
    QuantFp32Int8Insn,
    RTypeInsn,
    SoftmaxFp32Insn,
    VaddFp32Insn,
)
from taccel.isa.opcodes import (
    BUF_ABUF,
    BUF_ACCUM,
    BUF_WBUF,
    Opcode,
)


# All 9 W8A32 R-type classes — under W8A16 these become the polymorphic ops.
W8A16_CLASSES = (
    DequantAccumFp32Insn,
    QuantFp32Int8Insn,
    VaddFp32Insn,
    LayernormFp32Insn,
    GeluFp32Insn,
    SoftmaxFp32Insn,
    MaskedSoftmaxFp32Insn,
    DequantAccumFp32ScaledInsn,
    MaxAbsReduceFp32Insn,
)


def _make_insn(cls, flags: int):
    """Construct an Insn with stable buffer fields and the given flags bit.

    All 9 classes share the same RTypeInsn field layout — buffer IDs and
    offsets are chosen to satisfy `_validate_buf` for both ACCUM-source ops
    (DEQUANT_ACCUM_FP32 / DEQUANT_ACCUM_FP32_SCALED require src1=ACCUM)
    and the others.
    """
    src1_buf = BUF_ACCUM if cls in (DequantAccumFp32Insn, DequantAccumFp32ScaledInsn) else BUF_ABUF
    return cls(
        src1_buf=src1_buf,
        src1_off=16,
        src2_buf=BUF_WBUF,
        src2_off=8,
        dst_buf=BUF_ABUF,
        dst_off=32,
        sreg=3,
        flags=flags,
    )


@pytest.mark.parametrize("cls", W8A16_CLASSES, ids=lambda c: c.__name__)
def test_flag_bit_1_round_trips_through_encode_decode(cls):
    """flags=1 (FP16 storage) must survive encode/decode for every W8A16 class."""
    insn = _make_insn(cls, flags=1)
    encoded = encode(insn)
    assert len(encoded) == 8  # 64-bit instruction word

    decoded = decode(encoded)
    assert type(decoded) is cls, f"flags=1 must not change dispatch class for {cls.__name__}"
    assert decoded.flags == 1, f"flags=1 lost in round-trip for {cls.__name__}"
    # Every other field must round-trip too — flag bit must not alias other fields.
    assert decoded.src1_buf == insn.src1_buf
    assert decoded.src1_off == insn.src1_off
    assert decoded.src2_buf == insn.src2_buf
    assert decoded.src2_off == insn.src2_off
    assert decoded.dst_buf == insn.dst_buf
    assert decoded.dst_off == insn.dst_off
    assert decoded.sreg == insn.sreg


@pytest.mark.parametrize("cls", W8A16_CLASSES, ids=lambda c: c.__name__)
def test_flag_bit_0_still_round_trips(cls):
    """flags=0 (FP32 legacy) must remain bit-identical to the W8A32 contract."""
    insn = _make_insn(cls, flags=0)
    encoded = encode(insn)
    decoded = decode(encoded)
    assert type(decoded) is cls
    assert decoded.flags == 0


@pytest.mark.parametrize("cls", W8A16_CLASSES, ids=lambda c: c.__name__)
def test_flags_at_bit_zero_of_encoded_word(cls):
    """The flag bit must be at bit 0 of the 64-bit instruction word.

    Encoded difference between flags=0 and flags=1 must be exactly the LSB.
    """
    insn0 = _make_insn(cls, flags=0)
    insn1 = _make_insn(cls, flags=1)
    word0 = int.from_bytes(encode(insn0), byteorder="big")
    word1 = int.from_bytes(encode(insn1), byteorder="big")
    assert (word0 ^ word1) == 1, f"flag bit not at LSB for {cls.__name__}"


@pytest.mark.parametrize("cls", W8A16_CLASSES, ids=lambda c: c.__name__)
def test_flags_above_one_rejected_at_construction(cls):
    """flags must be 0 or 1; values >= 2 raise ValueError at __post_init__."""
    with pytest.raises(ValueError, match="flags must be 0 or 1"):
        _make_insn(cls, flags=2)


def test_rtype_validate_rejects_negative_flags():
    """flags=-1 also rejected (defensive — type system doesn't prevent it)."""
    with pytest.raises(ValueError, match="flags must be 0 or 1"):
        RTypeInsn(
            opcode=Opcode.VADD_FP32,
            src1_buf=BUF_ABUF, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=0,
            sreg=0,
            flags=-1,
        )


def test_opcode_value_unchanged_by_flag_bit():
    """Setting flags=1 must NOT shift the opcode field. Decoded opcode preserved."""
    for cls in W8A16_CLASSES:
        insn1 = _make_insn(cls, flags=1)
        decoded1 = decode(encode(insn1))
        insn0 = _make_insn(cls, flags=0)
        decoded0 = decode(encode(insn0))
        assert decoded0.opcode == decoded1.opcode == insn0.opcode, (
            f"opcode changed by flag bit for {cls.__name__}"
        )
