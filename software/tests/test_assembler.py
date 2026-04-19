"""Tests for assembler/disassembler."""
import pytest
import struct
from taccel.assembler import Assembler, Disassembler
from taccel.assembler.assembler import ProgramBinary, MAGIC, VERSION, LEGACY_VERSION
from taccel.isa.encoding import decode


SAMPLE_ASM = """\
; Sample program
NOP
SYNC 0b001
CONFIG_TILE M=13, N=13, K=4
SET_SCALE S0, imm=0x3000
SET_ADDR_LO R0, 0x0000000
LOAD buf_id=WBUF, sram_off=0, xfer_len=9216, addr_reg=0, dram_off=0
MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0
SYNC 0b010
REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[512], sreg=0, flags=0
REQUANT_PC src1=ACCUM[0], src2=WBUF[0], dst=ABUF[528], sreg=0, flags=0
HALT
"""


class TestAssembler:
    def setup_method(self):
        self.asm = Assembler()
        self.disasm = Disassembler()

    def test_basic_assemble(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        assert prog.insn_count == 11
        assert len(prog.instructions) == 88  # 11 * 8 bytes

    def test_program_binary_magic(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        raw = prog.to_bytes()
        import struct
        magic = struct.unpack(">I", raw[:4])[0]
        assert magic == MAGIC

    def test_program_binary_version(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        raw = prog.to_bytes()
        import struct
        version = struct.unpack(">H", raw[4:6])[0]
        assert version == VERSION

    def test_program_binary_roundtrip(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        raw = prog.to_bytes()
        prog2 = ProgramBinary.from_bytes(raw)
        assert prog2.insn_count == prog.insn_count
        assert prog2.instructions == prog.instructions
        assert prog2.entry_point == prog.entry_point

    def test_program_binary_roundtrip_preserves_runtime_metadata(self):
        prog = ProgramBinary(
            instructions=b"\x00" * 16,
            data=b"\x01\x02\x03\x04",
            entry_point=7,
            insn_count=2,
            data_base=96,
            input_offset=128,
            pos_embed_patch_dram_offset=256,
            pos_embed_cls_dram_offset=64,
            cls_token_dram_offset=48,
        )
        raw = prog.to_bytes()
        prog2 = ProgramBinary.from_bytes(raw)

        assert prog2.data_base == 96
        assert prog2.input_offset == 128
        assert prog2.pos_embed_patch_dram_offset == 256
        assert prog2.pos_embed_cls_dram_offset == 64
        assert prog2.cls_token_dram_offset == 48

    def test_program_binary_roundtrip_preserves_trace_and_compiler_manifest(self):
        prog = ProgramBinary(
            instructions=b"\x00" * 8,
            data=b"\x11\x22",
            entry_point=0,
            insn_count=1,
            trace_manifest={
                3: [{"node_name": "block0_ln1", "dtype": "int8", "logical_rows": 197}],
            },
            compiler_manifest={
                "compiler": {"options": {"gelu_from_accum": False}},
                "program_layout": {"data_base": 16},
            },
        )

        prog2 = ProgramBinary.from_bytes(prog.to_bytes())

        assert prog2.trace_manifest == prog.trace_manifest
        assert prog2.compiler_manifest == prog.compiler_manifest

    def test_program_binary_from_legacy_bytes_defaults_runtime_metadata(self):
        instructions = b"\x00" * 16
        data = b"\xAA\xBB"
        legacy_header_fmt = ">IHHIIIIQ"
        data_offset = struct.calcsize(legacy_header_fmt) + len(instructions)
        raw = struct.pack(
            legacy_header_fmt,
            MAGIC,
            LEGACY_VERSION,
            0,
            len(instructions) // 8,
            data_offset,
            len(data),
            0,
            0,
        ) + instructions + data

        prog = ProgramBinary.from_bytes(raw)

        assert prog.instructions == instructions
        assert prog.data == data
        assert prog.data_base == 0
        assert prog.input_offset == 0
        assert prog.pos_embed_patch_dram_offset == 0
        assert prog.pos_embed_cls_dram_offset == 0
        assert prog.cls_token_dram_offset == 0

    def test_disassemble(self):
        prog = self.asm.assemble(SAMPLE_ASM)
        text = self.disasm.disassemble(prog)
        lines = text.strip().split('\n')
        assert len(lines) == 11
        assert 'NOP' in lines[0]
        assert 'SYNC' in lines[1]
        assert 'CONFIG_TILE' in lines[2]
        assert 'HALT' in lines[-1]

    def test_assemble_disassemble_assemble_identity(self):
        """Assemble → disassemble → re-assemble produces identical binary."""
        prog1 = self.asm.assemble(SAMPLE_ASM)

        text = self.disasm.disassemble(prog1)

        # Convert disassembled text to assembly that can be re-assembled
        # The disassembler uses [0xXXXX] prefix, strip it
        lines = []
        for line in text.split('\n'):
            if '] ' in line:
                lines.append(line.split('] ', 1)[1])
            else:
                lines.append(line)

        prog2 = self.asm.assemble('\n'.join(lines))
        assert prog1.instructions == prog2.instructions, \
            f"Binary mismatch after round-trip"

    def test_nop_encoding(self):
        prog = self.asm.assemble("NOP")
        raw_insn = prog.get_instruction_bytes(0)
        insn = decode(raw_insn)
        from taccel.isa.instructions import NopInsn
        assert isinstance(insn, NopInsn)

    def test_requant_pc_roundtrip(self):
        prog = self.asm.assemble(
            "REQUANT_PC src1=ACCUM[0], src2=WBUF[4], dst=ABUF[8], sreg=0, flags=0"
        )
        insn = decode(prog.get_instruction_bytes(0))
        assert insn.opcode.name == "REQUANT_PC"
        assert insn.src1_buf == 2
        assert insn.src2_buf == 1
        assert insn.src2_off == 4
        assert insn.dst_off == 8

    def test_config_tile_tile_count_convention(self):
        """Assembly uses tile counts (1-based), encoding is 0-based."""
        prog = self.asm.assemble("CONFIG_TILE M=13, N=13, K=4")
        insn = decode(prog.get_instruction_bytes(0))
        # Encoded as M-1=12, N-1=12, K-1=3
        assert insn.M == 12 and insn.N == 12 and insn.K == 3

    def test_data_section(self):
        import os
        data = os.urandom(64)
        prog = self.asm.assemble("NOP\nHALT", data=data)
        raw = prog.to_bytes()
        prog2 = ProgramBinary.from_bytes(raw)
        assert prog2.data == data

    def test_empty_program(self):
        prog = self.asm.assemble("")
        assert prog.insn_count == 0


class TestDisassembler:
    def test_config_tile_display(self):
        """Disassembler shows tile counts (1-based)."""
        asm = Assembler()
        prog = asm.assemble("CONFIG_TILE M=13, N=13, K=4")
        disasm = Disassembler()
        text = disasm.disassemble(prog)
        assert "M=13" in text and "N=13" in text and "K=4" in text

    def test_buf_copy_transpose_shown(self):
        prog = Assembler().assemble(
            "BUF_COPY src_buf=ABUF, src_off=0, dst_buf=WBUF, dst_off=0, "
            "length=832, src_rows=13, transpose=1"
        )
        text = Disassembler().disassemble(prog)
        assert "transpose=1" in text
        assert "src_rows=13" in text

    def test_requant_pc_shown(self):
        prog = Assembler().assemble(
            "REQUANT_PC src1=ACCUM[0], src2=WBUF[4], dst=ABUF[8], sreg=0, flags=0"
        )
        text = Disassembler().disassemble(prog)
        assert "REQUANT_PC" in text
        assert "WBUF[0x0004]" in text

    def test_softmax_attnv_roundtrip(self):
        prog = Assembler().assemble(
            "SOFTMAX_ATTNV src1=ACCUM[0], src2=ABUF[4], dst=WBUF[8], sreg=4, flags=0"
        )
        insn = decode(prog.get_instruction_bytes(0))
        assert insn.opcode.name == "SOFTMAX_ATTNV"
        text = Disassembler().disassemble(prog)
        assert "SOFTMAX_ATTNV" in text
        assert "ABUF[0x0004]" in text

    def test_config_attn_roundtrip(self):
        prog = Assembler().assemble(
            "CONFIG_ATTN query_row_base=16, valid_kv_len=31, mode=0b11"
        )
        insn = decode(prog.get_instruction_bytes(0))
        assert insn.opcode.name == "CONFIG_ATTN"
        assert insn.query_row_base == 16
        assert insn.valid_kv_len == 31
        assert insn.mode == 0b11
        text = Disassembler().disassemble(prog)
        assert "CONFIG_ATTN" in text
        assert "query_row_base=16" in text
        assert "valid_kv_len=31" in text
        assert "mode=0b11" in text

    def test_masked_softmax_assembly_roundtrip(self):
        source = "\n".join([
            "CONFIG_ATTN query_row_base=0, valid_kv_len=16, mode=0b10",
            "MASKED_SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=WBUF[16], sreg=2, flags=0",
            "MASKED_SOFTMAX_ATTNV src1=ACCUM[0], src2=ABUF[4], dst=WBUF[8], sreg=4, flags=0",
        ])
        prog1 = Assembler().assemble(source)
        text = Disassembler().disassemble(prog1)
        lines = [line.split('] ', 1)[1] for line in text.split('\n')]
        prog2 = Assembler().assemble('\n'.join(lines))
        assert prog1.instructions == prog2.instructions
        assert "MASKED_SOFTMAX" in text
        assert "MASKED_SOFTMAX_ATTNV" in text

    def test_dequant_add_roundtrip(self):
        prog = Assembler().assemble(
            "DEQUANT_ADD src1=ACCUM[0], src2=ABUF[4], dst=ABUF[8], sreg=6, flags=0"
        )
        insn = decode(prog.get_instruction_bytes(0))
        assert insn.opcode.name == "DEQUANT_ADD"
        text = Disassembler().disassemble(prog)
        assert "DEQUANT_ADD" in text
        assert "ABUF[0x0008]" in text
