"""Disassembler: ProgramBinary → annotated text assembly."""
from ..isa.encoding import decode
from ..isa.opcodes import Opcode, BUFFER_NAMES, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from ..isa.instructions import (
    RTypeInsn, MTypeInsn, BufCopyInsn, ATypeInsn, ConfigTileInsn,
    ConfigAttnInsn, SetScaleInsn, SyncInsn, NopInsn, HaltInsn,
)
from .assembler import ProgramBinary


def _buf_name(buf_id: int) -> str:
    return BUFFER_NAMES.get(buf_id, f"BUF{buf_id}")


def _buf_ref(buf_id: int, offset: int) -> str:
    return f"{_buf_name(buf_id)}[0x{offset:04x}]"


class Disassembler:
    """Convert ProgramBinary to annotated assembly text."""

    def disassemble(self, program: ProgramBinary) -> str:
        lines = []
        for pc in range(program.insn_count):
            raw = program.get_instruction_bytes(pc)
            insn = decode(raw)
            asm = self._format_insn(insn)
            lines.append(f"[0x{pc:04x}] {asm}")
        return '\n'.join(lines)

    def _format_insn(self, insn) -> str:
        op = insn.opcode
        name = op.name

        if isinstance(insn, NopInsn):
            return "NOP"
        elif isinstance(insn, HaltInsn):
            return "HALT"
        elif isinstance(insn, SyncInsn):
            return f"SYNC 0b{insn.resource_mask:03b}"
        elif isinstance(insn, ConfigTileInsn):
            # Display tile counts (add 1 back from encoded value)
            return f"CONFIG_TILE M={insn.M + 1}, N={insn.N + 1}, K={insn.K + 1}"
        elif isinstance(insn, ConfigAttnInsn):
            return (
                "CONFIG_ATTN "
                f"query_row_base={insn.query_row_base}, "
                f"valid_kv_len={insn.valid_kv_len}, "
                f"mode=0b{insn.mode:02b}"
            )
        elif isinstance(insn, SetScaleInsn):
            if insn.src_mode == 0:
                return f"SET_SCALE S{insn.sreg}, imm=0x{insn.imm16:04x}"
            else:
                src_buf = {1: "ABUF", 2: "WBUF", 3: "ACCUM"}[insn.src_mode]
                return f"SET_SCALE S{insn.sreg}, {src_buf}[0x{insn.imm16:04x}]"
        elif isinstance(insn, ATypeInsn):
            return f"{name} R{insn.addr_reg}, 0x{insn.imm28:07x}"
        elif isinstance(insn, MTypeInsn):
            parts = [
                f"buf_id={_buf_name(insn.buf_id)}",
                f"sram_off={insn.sram_off}",
                f"xfer_len={insn.xfer_len}",
                f"addr_reg={insn.addr_reg}",
                f"dram_off={insn.dram_off}",
            ]
            if insn.stride_log2 > 0 or insn.flags & 1:
                parts.append(f"stride_log2={insn.stride_log2}")
                parts.append(f"flags={insn.flags}")
            return f"{name} {', '.join(parts)}"
        elif isinstance(insn, BufCopyInsn):
            parts = [
                f"src_buf={_buf_name(insn.src_buf)}",
                f"src_off={insn.src_off}",
                f"dst_buf={_buf_name(insn.dst_buf)}",
                f"dst_off={insn.dst_off}",
                f"length={insn.length}",
            ]
            if insn.transpose:
                parts.append(f"src_rows={insn.src_rows}")
                parts.append(f"transpose={insn.transpose}")
            return f"BUF_COPY {', '.join(parts)}"
        elif isinstance(insn, RTypeInsn):
            src1 = _buf_ref(insn.src1_buf, insn.src1_off)
            src2 = _buf_ref(insn.src2_buf, insn.src2_off)
            dst = _buf_ref(insn.dst_buf, insn.dst_off)
            parts = [src1, src2, dst]
            if insn.sreg > 0 or op in (Opcode.REQUANT, Opcode.SCALE_MUL,
                                        Opcode.SOFTMAX, Opcode.LAYERNORM, Opcode.GELU,
                                        Opcode.SOFTMAX_ATTNV, Opcode.DEQUANT_ADD):
                parts.append(f"S{insn.sreg}")
            if insn.flags:
                parts.append(f"acc={insn.flags}")
            return f"{name} {', '.join(parts)}"

        return f"{name} ???"
