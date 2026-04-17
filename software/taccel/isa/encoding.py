"""Encode instructions to bytes and decode bytes to instructions."""
import struct
from .opcodes import (
    Opcode, InsnFormat, OPCODE_FORMAT,
    OPCODE_SHIFT, OPCODE_MASK,
    R_SRC1_BUF_SHIFT, R_SRC1_OFF_SHIFT, R_SRC2_BUF_SHIFT, R_SRC2_OFF_SHIFT,
    R_DST_BUF_SHIFT, R_DST_OFF_SHIFT, R_SREG_SHIFT, R_FLAGS_SHIFT,
    M_BUF_ID_SHIFT, M_SRAM_OFF_SHIFT, M_XFER_LEN_SHIFT, M_ADDR_REG_SHIFT,
    M_DRAM_OFF_SHIFT, M_STRIDE_LOG2_SHIFT, M_FLAGS_SHIFT,
    B_SRC_BUF_SHIFT, B_SRC_OFF_SHIFT, B_DST_BUF_SHIFT, B_DST_OFF_SHIFT,
    B_LENGTH_SHIFT, B_SRC_ROWS_SHIFT, B_TRANSPOSE_SHIFT,
    A_ADDR_REG_SHIFT, A_IMM28_SHIFT,
    C_M_SHIFT, C_N_SHIFT, C_K_SHIFT,
    SS_SREG_SHIFT, SS_SRC_MODE_SHIFT, SS_IMM16_SHIFT,
    SYNC_RESOURCE_MASK_SHIFT,
    MASK_2BIT, MASK_3BIT, MASK_4BIT, MASK_6BIT, MASK_10BIT, MASK_16BIT, MASK_28BIT,
)
from .instructions import (
    Instruction, RTypeInsn, MTypeInsn, BufCopyInsn, ATypeInsn, ConfigTileInsn,
    SetScaleInsn, SyncInsn, NopInsn, HaltInsn,
    MatmulInsn, RequantInsn, RequantPcInsn, ScaleMulInsn, VaddInsn, SoftmaxInsn, LayernormInsn, GeluInsn,
    SoftmaxAttnVInsn, DequantAddInsn,
    LoadInsn, StoreInsn, SetAddrLoInsn, SetAddrHiInsn,
)

# Map opcode to concrete R-type class
_R_TYPE_CLASSES = {
    Opcode.MATMUL: MatmulInsn,
    Opcode.REQUANT: RequantInsn,
    Opcode.REQUANT_PC: RequantPcInsn,
    Opcode.SCALE_MUL: ScaleMulInsn,
    Opcode.VADD: VaddInsn,
    Opcode.SOFTMAX: SoftmaxInsn,
    Opcode.LAYERNORM: LayernormInsn,
    Opcode.GELU: GeluInsn,
    Opcode.SOFTMAX_ATTNV: SoftmaxAttnVInsn,
    Opcode.DEQUANT_ADD: DequantAddInsn,
}

_M_TYPE_CLASSES = {
    Opcode.LOAD: LoadInsn,
    Opcode.STORE: StoreInsn,
}

_A_TYPE_CLASSES = {
    Opcode.SET_ADDR_LO: SetAddrLoInsn,
    Opcode.SET_ADDR_HI: SetAddrHiInsn,
}


def encode(insn: Instruction) -> bytes:
    """Encode an instruction to 8 bytes (big-endian)."""
    word = insn.opcode << OPCODE_SHIFT
    fmt = OPCODE_FORMAT[insn.opcode]

    if fmt == InsnFormat.R_TYPE:
        word |= (insn.src1_buf & MASK_2BIT) << R_SRC1_BUF_SHIFT
        word |= (insn.src1_off & MASK_16BIT) << R_SRC1_OFF_SHIFT
        word |= (insn.src2_buf & MASK_2BIT) << R_SRC2_BUF_SHIFT
        word |= (insn.src2_off & MASK_16BIT) << R_SRC2_OFF_SHIFT
        word |= (insn.dst_buf & MASK_2BIT) << R_DST_BUF_SHIFT
        word |= (insn.dst_off & MASK_16BIT) << R_DST_OFF_SHIFT
        word |= (insn.sreg & MASK_4BIT) << R_SREG_SHIFT
        word |= (insn.flags & 0x1) << R_FLAGS_SHIFT

    elif fmt == InsnFormat.M_TYPE:
        word |= (insn.buf_id & MASK_2BIT) << M_BUF_ID_SHIFT
        word |= (insn.sram_off & MASK_16BIT) << M_SRAM_OFF_SHIFT
        word |= (insn.xfer_len & MASK_16BIT) << M_XFER_LEN_SHIFT
        word |= (insn.addr_reg & MASK_2BIT) << M_ADDR_REG_SHIFT
        word |= (insn.dram_off & MASK_16BIT) << M_DRAM_OFF_SHIFT
        word |= (insn.stride_log2 & MASK_4BIT) << M_STRIDE_LOG2_SHIFT
        word |= (insn.flags & MASK_3BIT) << M_FLAGS_SHIFT

    elif fmt == InsnFormat.B_TYPE:
        word |= (insn.src_buf & MASK_2BIT) << B_SRC_BUF_SHIFT
        word |= (insn.src_off & MASK_16BIT) << B_SRC_OFF_SHIFT
        word |= (insn.dst_buf & MASK_2BIT) << B_DST_BUF_SHIFT
        word |= (insn.dst_off & MASK_16BIT) << B_DST_OFF_SHIFT
        word |= (insn.length & MASK_16BIT) << B_LENGTH_SHIFT
        word |= (insn.src_rows & MASK_6BIT) << B_SRC_ROWS_SHIFT
        word |= (insn.transpose & 0x1) << B_TRANSPOSE_SHIFT

    elif fmt == InsnFormat.A_TYPE:
        word |= (insn.addr_reg & MASK_2BIT) << A_ADDR_REG_SHIFT
        word |= (insn.imm28 & MASK_28BIT) << A_IMM28_SHIFT

    elif fmt == InsnFormat.C_TYPE:
        word |= (insn.M & MASK_10BIT) << C_M_SHIFT
        word |= (insn.N & MASK_10BIT) << C_N_SHIFT
        word |= (insn.K & MASK_10BIT) << C_K_SHIFT

    elif fmt == InsnFormat.S_TYPE:
        if insn.opcode == Opcode.SET_SCALE:
            word |= (insn.sreg & MASK_4BIT) << SS_SREG_SHIFT
            word |= (insn.src_mode & MASK_2BIT) << SS_SRC_MODE_SHIFT
            word |= (insn.imm16 & MASK_16BIT) << SS_IMM16_SHIFT
        elif insn.opcode == Opcode.SYNC:
            word |= (insn.resource_mask & MASK_3BIT) << SYNC_RESOURCE_MASK_SHIFT
        # NOP and HALT have no payload

    return struct.pack(">Q", word)


def decode(data: bytes) -> Instruction:
    """Decode 8 bytes (big-endian) to an instruction."""
    if len(data) != 8:
        raise ValueError(f"Expected 8 bytes, got {len(data)}")
    word = struct.unpack(">Q", data)[0]
    opcode_val = (word >> OPCODE_SHIFT) & OPCODE_MASK
    opcode = Opcode(opcode_val)
    fmt = OPCODE_FORMAT[opcode]

    if fmt == InsnFormat.R_TYPE:
        cls = _R_TYPE_CLASSES[opcode]
        return cls(
            src1_buf=(word >> R_SRC1_BUF_SHIFT) & MASK_2BIT,
            src1_off=(word >> R_SRC1_OFF_SHIFT) & MASK_16BIT,
            src2_buf=(word >> R_SRC2_BUF_SHIFT) & MASK_2BIT,
            src2_off=(word >> R_SRC2_OFF_SHIFT) & MASK_16BIT,
            dst_buf=(word >> R_DST_BUF_SHIFT) & MASK_2BIT,
            dst_off=(word >> R_DST_OFF_SHIFT) & MASK_16BIT,
            sreg=(word >> R_SREG_SHIFT) & MASK_4BIT,
            flags=(word >> R_FLAGS_SHIFT) & 0x1,
        )

    elif fmt == InsnFormat.M_TYPE:
        cls = _M_TYPE_CLASSES[opcode]
        return cls(
            buf_id=(word >> M_BUF_ID_SHIFT) & MASK_2BIT,
            sram_off=(word >> M_SRAM_OFF_SHIFT) & MASK_16BIT,
            xfer_len=(word >> M_XFER_LEN_SHIFT) & MASK_16BIT,
            addr_reg=(word >> M_ADDR_REG_SHIFT) & MASK_2BIT,
            dram_off=(word >> M_DRAM_OFF_SHIFT) & MASK_16BIT,
            stride_log2=(word >> M_STRIDE_LOG2_SHIFT) & MASK_4BIT,
            flags=(word >> M_FLAGS_SHIFT) & MASK_3BIT,
        )

    elif fmt == InsnFormat.B_TYPE:
        return BufCopyInsn(
            src_buf=(word >> B_SRC_BUF_SHIFT) & MASK_2BIT,
            src_off=(word >> B_SRC_OFF_SHIFT) & MASK_16BIT,
            dst_buf=(word >> B_DST_BUF_SHIFT) & MASK_2BIT,
            dst_off=(word >> B_DST_OFF_SHIFT) & MASK_16BIT,
            length=(word >> B_LENGTH_SHIFT) & MASK_16BIT,
            src_rows=(word >> B_SRC_ROWS_SHIFT) & MASK_6BIT,
            transpose=(word >> B_TRANSPOSE_SHIFT) & 0x1,
        )

    elif fmt == InsnFormat.A_TYPE:
        cls = _A_TYPE_CLASSES[opcode]
        return cls(
            addr_reg=(word >> A_ADDR_REG_SHIFT) & MASK_2BIT,
            imm28=(word >> A_IMM28_SHIFT) & MASK_28BIT,
        )

    elif fmt == InsnFormat.C_TYPE:
        return ConfigTileInsn(
            M=(word >> C_M_SHIFT) & MASK_10BIT,
            N=(word >> C_N_SHIFT) & MASK_10BIT,
            K=(word >> C_K_SHIFT) & MASK_10BIT,
        )

    elif fmt == InsnFormat.S_TYPE:
        if opcode == Opcode.SET_SCALE:
            return SetScaleInsn(
                sreg=(word >> SS_SREG_SHIFT) & MASK_4BIT,
                src_mode=(word >> SS_SRC_MODE_SHIFT) & MASK_2BIT,
                imm16=(word >> SS_IMM16_SHIFT) & MASK_16BIT,
            )
        elif opcode == Opcode.SYNC:
            return SyncInsn(
                resource_mask=(word >> SYNC_RESOURCE_MASK_SHIFT) & MASK_3BIT,
            )
        elif opcode == Opcode.NOP:
            return NopInsn()
        elif opcode == Opcode.HALT:
            return HaltInsn()

    raise ValueError(f"Unknown opcode {opcode_val:#x}")
