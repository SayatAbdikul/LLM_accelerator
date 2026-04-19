"""Tests for ISA encoding/decoding round-trips."""
import pytest
import struct
import numpy as np
from taccel.isa import encode, decode
from taccel.isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from taccel.isa.instructions import (
    NopInsn, HaltInsn, SyncInsn, ConfigTileInsn, SetScaleInsn,
    SetAddrLoInsn, SetAddrHiInsn, LoadInsn, StoreInsn, BufCopyInsn,
    MatmulInsn, RequantInsn, RequantPcInsn, ScaleMulInsn, VaddInsn,
    SoftmaxInsn, LayernormInsn, GeluInsn, SoftmaxAttnVInsn,
    ConfigAttnInsn, MaskedSoftmaxInsn, MaskedSoftmaxAttnVInsn, DequantAddInsn,
)


def round_trip(insn):
    """Encode then decode an instruction and check equality."""
    enc = encode(insn)
    assert len(enc) == 8, f"Expected 8 bytes, got {len(enc)}"
    dec = decode(enc)
    assert type(dec) == type(insn), f"Type mismatch: {type(dec)} != {type(insn)}"
    # Re-encode and check binary equality
    enc2 = encode(dec)
    assert enc == enc2, f"Binary mismatch for {type(insn).__name__}"
    return dec


class TestSystemInstructions:
    def test_nop(self):
        dec = round_trip(NopInsn())
        assert dec.opcode == Opcode.NOP

    def test_halt(self):
        dec = round_trip(HaltInsn())
        assert dec.opcode == Opcode.HALT

    def test_sync_all(self):
        for mask in range(8):
            dec = round_trip(SyncInsn(resource_mask=mask))
            assert dec.resource_mask == mask

    def test_sync_dma_only(self):
        dec = round_trip(SyncInsn(resource_mask=0b001))
        assert dec.resource_mask == 1

    def test_set_scale_immediate(self):
        dec = round_trip(SetScaleInsn(sreg=4, src_mode=0, imm16=0x3000))
        assert dec.sreg == 4
        assert dec.src_mode == 0
        assert dec.imm16 == 0x3000

    def test_set_scale_all_sregs(self):
        for sreg in range(16):
            dec = round_trip(SetScaleInsn(sreg=sreg, src_mode=0, imm16=0x1234))
            assert dec.sreg == sreg

    def test_set_scale_from_buffer(self):
        dec = round_trip(SetScaleInsn(sreg=2, src_mode=1, imm16=128))
        assert dec.sreg == 2
        assert dec.src_mode == 1
        assert dec.imm16 == 128


class TestConfigInstructions:
    def test_config_tile_basic(self):
        dec = round_trip(ConfigTileInsn(M=12, N=12, K=3))
        assert dec.M == 12 and dec.N == 12 and dec.K == 3

    def test_config_tile_max(self):
        dec = round_trip(ConfigTileInsn(M=1023, N=1023, K=1023))
        assert dec.M == 1023 and dec.N == 1023 and dec.K == 1023

    def test_config_tile_zero(self):
        dec = round_trip(ConfigTileInsn(M=0, N=0, K=0))
        assert dec.M == 0 and dec.N == 0 and dec.K == 0

    def test_config_attn_roundtrip_boundaries(self):
        for mode in range(4):
            dec = round_trip(ConfigAttnInsn(
                query_row_base=0xFFF,
                valid_kv_len=0xFFF,
                mode=mode,
            ))
            assert dec.query_row_base == 0xFFF
            assert dec.valid_kv_len == 0xFFF
            assert dec.mode == mode

    def test_config_attn_reserved_bits_fault(self):
        word = (Opcode.CONFIG_ATTN << 59) | 1
        with pytest.raises(ValueError, match="reserved bits"):
            decode(struct.pack(">Q", word))


class TestAddressInstructions:
    def test_set_addr_lo(self):
        dec = round_trip(SetAddrLoInsn(addr_reg=0, imm28=0x1234567))
        assert dec.addr_reg == 0
        assert dec.imm28 == 0x1234567

    def test_set_addr_hi(self):
        dec = round_trip(SetAddrHiInsn(addr_reg=3, imm28=0xFFFFFFF))
        assert dec.addr_reg == 3
        assert dec.imm28 == 0xFFFFFFF

    def test_all_addr_regs(self):
        for reg in range(4):
            lo = round_trip(SetAddrLoInsn(addr_reg=reg, imm28=reg * 100))
            assert lo.addr_reg == reg


class TestDMAInstructions:
    def test_load_basic(self):
        dec = round_trip(LoadInsn(buf_id=BUF_WBUF, sram_off=0, xfer_len=9216,
                                   addr_reg=0, dram_off=0))
        assert dec.buf_id == BUF_WBUF
        assert dec.xfer_len == 9216

    def test_store_basic(self):
        dec = round_trip(StoreInsn(buf_id=BUF_ABUF, sram_off=100, xfer_len=768,
                                    addr_reg=2, dram_off=500))
        assert dec.buf_id == BUF_ABUF
        assert dec.sram_off == 100
        assert dec.xfer_len == 768
        assert dec.addr_reg == 2
        assert dec.dram_off == 500

    def test_load_with_stride(self):
        dec = round_trip(LoadInsn(buf_id=BUF_ABUF, sram_off=0, xfer_len=16,
                                   addr_reg=1, dram_off=32, stride_log2=4, flags=1))
        assert dec.stride_log2 == 4
        assert dec.flags == 1

    def test_max_xfer_len(self):
        dec = round_trip(LoadInsn(buf_id=BUF_WBUF, sram_off=0, xfer_len=0xFFFF,
                                   addr_reg=0, dram_off=0))
        assert dec.xfer_len == 0xFFFF


class TestBufCopyInstruction:
    def test_flat_copy(self):
        dec = round_trip(BufCopyInsn(src_buf=BUF_ABUF, src_off=0, dst_buf=BUF_WBUF,
                                      dst_off=0, length=100, transpose=0))
        assert dec.src_buf == BUF_ABUF and dec.dst_buf == BUF_WBUF
        assert dec.length == 100 and dec.transpose == 0

    def test_transpose_copy(self):
        dec = round_trip(BufCopyInsn(src_buf=BUF_ABUF, src_off=50, dst_buf=BUF_WBUF,
                                      dst_off=0, length=832, src_rows=13, transpose=1))
        assert dec.src_rows == 13
        assert dec.transpose == 1
        assert dec.length == 832
        assert dec.src_off == 50

    def test_cls_extract(self):
        """CLS extraction: copy 12 units (192 bytes)."""
        dec = round_trip(BufCopyInsn(src_buf=BUF_ABUF, src_off=0, dst_buf=BUF_ABUF,
                                      dst_off=8191, length=12, transpose=0))
        assert dec.length == 12

    def test_max_src_rows(self):
        dec = round_trip(BufCopyInsn(src_buf=0, src_off=0, dst_buf=1, dst_off=0,
                                      length=0, src_rows=63, transpose=1))
        assert dec.src_rows == 63


class TestRTypeInstructions:
    def test_matmul_basic(self):
        dec = round_trip(MatmulInsn(src1_buf=BUF_ABUF, src1_off=256,
                                     src2_buf=BUF_WBUF, src2_off=0,
                                     dst_buf=BUF_ACCUM, dst_off=0,
                                     sreg=0, flags=0))
        assert dec.src1_buf == BUF_ABUF
        assert dec.src1_off == 256
        assert dec.src2_buf == BUF_WBUF
        assert dec.dst_buf == BUF_ACCUM

    def test_matmul_accumulate(self):
        dec = round_trip(MatmulInsn(src1_buf=0, src1_off=0, src2_buf=1, src2_off=0,
                                     dst_buf=2, dst_off=0, sreg=0, flags=1))
        assert dec.flags == 1

    def test_requant(self):
        dec = round_trip(RequantInsn(src1_buf=BUF_ACCUM, src1_off=0,
                                      dst_buf=BUF_ABUF, dst_off=512,
                                      sreg=5))
        assert dec.src1_buf == BUF_ACCUM
        assert dec.dst_buf == BUF_ABUF
        assert dec.dst_off == 512
        assert dec.sreg == 5

    def test_requant_pc(self):
        dec = round_trip(RequantPcInsn(src1_buf=BUF_ACCUM, src1_off=0,
                                        src2_buf=BUF_WBUF, src2_off=64,
                                        dst_buf=BUF_ABUF, dst_off=512,
                                        sreg=0))
        assert dec.src1_buf == BUF_ACCUM
        assert dec.src2_buf == BUF_WBUF
        assert dec.src2_off == 64
        assert dec.dst_buf == BUF_ABUF
        assert dec.dst_off == 512

    def test_scale_mul(self):
        dec = round_trip(ScaleMulInsn(src1_buf=BUF_ACCUM, src1_off=0,
                                       dst_buf=BUF_ACCUM, dst_off=0, sreg=0))
        assert dec.src1_buf == BUF_ACCUM and dec.dst_buf == BUF_ACCUM

    def test_vadd_int8(self):
        dec = round_trip(VaddInsn(src1_buf=BUF_ABUF, src1_off=0,
                                   src2_buf=BUF_ABUF, src2_off=100,
                                   dst_buf=BUF_ABUF, dst_off=200))
        assert dec.src1_buf == BUF_ABUF

    def test_vadd_int32(self):
        dec = round_trip(VaddInsn(src1_buf=BUF_ACCUM, src1_off=0,
                                   src2_buf=BUF_WBUF, src2_off=50,
                                   dst_buf=BUF_ACCUM, dst_off=0))
        assert dec.src1_buf == BUF_ACCUM

    def test_softmax(self):
        dec = round_trip(SoftmaxInsn(src1_buf=BUF_ABUF, src1_off=0,
                                      dst_buf=BUF_ABUF, dst_off=0, sreg=6))
        assert dec.sreg == 6

    def test_masked_softmax(self):
        dec = round_trip(MaskedSoftmaxInsn(src1_buf=BUF_ACCUM, src1_off=0,
                                           dst_buf=BUF_WBUF, dst_off=64, sreg=6))
        assert dec.src1_buf == BUF_ACCUM
        assert dec.dst_buf == BUF_WBUF
        assert dec.sreg == 6

    def test_layernorm(self):
        dec = round_trip(LayernormInsn(src1_buf=BUF_ABUF, src1_off=0,
                                        src2_buf=BUF_WBUF, src2_off=0,
                                        dst_buf=BUF_ABUF, dst_off=0, sreg=4))
        assert dec.src2_buf == BUF_WBUF

    def test_gelu(self):
        dec = round_trip(GeluInsn(src1_buf=BUF_ABUF, src1_off=0,
                                   dst_buf=BUF_ABUF, dst_off=0, sreg=2))
        assert dec.sreg == 2

    def test_softmax_attnv(self):
        dec = round_trip(SoftmaxAttnVInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=64,
            dst_buf=BUF_WBUF, dst_off=128, sreg=4,
        ))
        assert dec.src1_buf == BUF_ACCUM
        assert dec.src2_buf == BUF_ABUF
        assert dec.dst_buf == BUF_WBUF
        assert dec.sreg == 4

    def test_masked_softmax_attnv(self):
        dec = round_trip(MaskedSoftmaxAttnVInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=64,
            dst_buf=BUF_WBUF, dst_off=128, sreg=4,
        ))
        assert dec.src1_buf == BUF_ACCUM
        assert dec.src2_buf == BUF_ABUF
        assert dec.dst_buf == BUF_WBUF
        assert dec.sreg == 4

    def test_dequant_add(self):
        dec = round_trip(DequantAddInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=64,
            dst_buf=BUF_ABUF, dst_off=128, sreg=6,
        ))
        assert dec.src1_buf == BUF_ACCUM
        assert dec.src2_buf == BUF_ABUF
        assert dec.dst_buf == BUF_ABUF
        assert dec.sreg == 6

    def test_max_offset_abuf(self):
        """Test max valid ABUF offset."""
        dec = round_trip(MatmulInsn(src1_buf=BUF_ABUF, src1_off=8191,
                                     src2_buf=BUF_WBUF, src2_off=0,
                                     dst_buf=BUF_ACCUM, dst_off=0))
        assert dec.src1_off == 8191

    def test_max_offset_wbuf(self):
        """Test max valid WBUF offset."""
        dec = round_trip(MatmulInsn(src1_buf=BUF_ABUF, src1_off=0,
                                     src2_buf=BUF_WBUF, src2_off=16383,
                                     dst_buf=BUF_ACCUM, dst_off=0))
        assert dec.src2_off == 16383

    def test_all_20_instruction_types(self):
        """Ensure all 20 instruction types can be round-tripped."""
        insns = [
            NopInsn(), HaltInsn(), SyncInsn(resource_mask=7),
            ConfigTileInsn(M=0, N=0, K=0),
            SetScaleInsn(sreg=0, src_mode=0, imm16=0),
            SetAddrLoInsn(addr_reg=0, imm28=0),
            SetAddrHiInsn(addr_reg=0, imm28=0),
            LoadInsn(buf_id=0, sram_off=0, xfer_len=0, addr_reg=0, dram_off=0),
            StoreInsn(buf_id=0, sram_off=0, xfer_len=0, addr_reg=0, dram_off=0),
            BufCopyInsn(src_buf=0, src_off=0, dst_buf=1, dst_off=0, length=0),
            MatmulInsn(src1_buf=0, src1_off=0, src2_buf=1, src2_off=0, dst_buf=2, dst_off=0),
            RequantInsn(src1_buf=2, src1_off=0, dst_buf=0, dst_off=0, sreg=0),
            RequantPcInsn(src1_buf=2, src1_off=0, src2_buf=1, src2_off=0, dst_buf=0, dst_off=0, sreg=0),
            ScaleMulInsn(src1_buf=2, src1_off=0, dst_buf=2, dst_off=0, sreg=0),
            VaddInsn(src1_buf=0, src1_off=0, src2_buf=0, src2_off=0, dst_buf=0, dst_off=0),
            SoftmaxInsn(src1_buf=0, src1_off=0, dst_buf=0, dst_off=0, sreg=0),
            SoftmaxAttnVInsn(src1_buf=2, src1_off=0, src2_buf=0, src2_off=0, dst_buf=1, dst_off=0, sreg=0),
            DequantAddInsn(src1_buf=2, src1_off=0, src2_buf=0, src2_off=0, dst_buf=0, dst_off=0, sreg=0),
            LayernormInsn(src1_buf=0, src1_off=0, src2_buf=1, src2_off=0, dst_buf=0, dst_off=0, sreg=0),
            GeluInsn(src1_buf=0, src1_off=0, dst_buf=0, dst_off=0, sreg=0),
        ]
        assert len(insns) == 20, f"Expected 20 instructions, got {len(insns)}"
        for insn in insns:
            round_trip(insn)


class TestValidation:
    def test_invalid_buf_raises(self):
        with pytest.raises(ValueError):
            MatmulInsn(src1_buf=3, src1_off=0, src2_buf=1, src2_off=0,
                       dst_buf=2, dst_off=0)

    def test_offset_exceeds_abuf_max(self):
        with pytest.raises(ValueError):
            MatmulInsn(src1_buf=BUF_ABUF, src1_off=8192,
                       src2_buf=BUF_WBUF, src2_off=0, dst_buf=BUF_ACCUM, dst_off=0)

    def test_invalid_sreg(self):
        with pytest.raises(ValueError):
            SetScaleInsn(sreg=16, src_mode=0, imm16=0)

    def test_config_tile_oob(self):
        with pytest.raises(ValueError):
            ConfigTileInsn(M=1024, N=0, K=0)
