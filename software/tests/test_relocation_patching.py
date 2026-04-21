"""Stage 3 tests for paired SET_ADDR relocation helpers."""
import pytest

from taccel.assembler.assembler import (
    ProgramBinary,
    patch_set_addr_pair,
    relocate_set_addr_pairs,
)
from taccel.isa.encoding import decode, encode
from taccel.isa.instructions import HaltInsn, NopInsn, SetAddrHiInsn, SetAddrLoInsn


def _bytes(*insns) -> bytearray:
    out = bytearray()
    for insn in insns:
        out.extend(encode(insn))
    return out


def _addr_pair(words: bytes, lo_pc: int = 0, hi_pc: int = 1):
    lo = decode(words[lo_pc * 8:lo_pc * 8 + 8])
    hi = decode(words[hi_pc * 8:hi_pc * 8 + 8])
    return lo, hi, (hi.imm28 << 28) | lo.imm28


def test_patch_set_addr_pair_carries_into_hi_half():
    words = _bytes(SetAddrLoInsn(addr_reg=2, imm28=0), SetAddrHiInsn(addr_reg=2, imm28=0))

    patch_set_addr_pair(words, local_lo_pc=0, local_hi_pc=1, addr_reg=2, byte_addr=(1 << 28) + 0x34)

    lo, hi, addr = _addr_pair(words)
    assert lo.addr_reg == 2
    assert hi.addr_reg == 2
    assert lo.imm28 == 0x34
    assert hi.imm28 == 1
    assert addr == (1 << 28) + 0x34


def test_relocate_set_addr_pairs_adds_delta_with_carry():
    old = (1 << 28) - 16
    words = _bytes(
        SetAddrLoInsn(addr_reg=0, imm28=old & 0x0FFFFFFF),
        SetAddrHiInsn(addr_reg=0, imm28=old >> 28),
        HaltInsn(),
    )

    patched = relocate_set_addr_pairs(words, 32)

    _, _, addr = _addr_pair(patched)
    assert addr == (1 << 28) + 16


@pytest.mark.parametrize(
    "words, message",
    [
        (_bytes(SetAddrLoInsn(addr_reg=0, imm28=0)), "missing paired"),
        (_bytes(SetAddrLoInsn(addr_reg=0, imm28=0), NopInsn()), "not followed"),
        (
            _bytes(SetAddrLoInsn(addr_reg=0, imm28=0), SetAddrHiInsn(addr_reg=1, imm28=0)),
            "register mismatch",
        ),
    ],
)
def test_relocate_set_addr_pairs_rejects_malformed_pairs(words, message):
    with pytest.raises(ValueError, match=message):
        relocate_set_addr_pairs(words, 16)


def test_patch_set_addr_pair_validates_expected_register():
    words = _bytes(SetAddrLoInsn(addr_reg=1, imm28=0), SetAddrHiInsn(addr_reg=1, imm28=0))

    with pytest.raises(ValueError, match="expected R2"):
        patch_set_addr_pair(words, 0, 1, addr_reg=2, byte_addr=64)


def test_program_binary_dram_image_layout_is_unchanged():
    program = ProgramBinary(
        instructions=bytes(_bytes(HaltInsn())),
        data=b"abcd",
        data_base=16,
        insn_count=1,
    )

    assert program.to_dram_image() == bytes(_bytes(HaltInsn())) + bytes(8) + b"abcd"
