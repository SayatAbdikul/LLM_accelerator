"""ISA opcode definitions, instruction formats, and field constants.

TACCEL ISA v1 — 64-bit fixed-width instructions, big-endian encoding.

Architecture overview
---------------------
Three execution units operate behind an in-order issue stage:
  - DMA   : LOAD / STORE (DRAM ↔ SRAM)
  - Systolic : MATMUL (INT8×INT8 → INT32, 16×16 tiled)
  - SFU   : SOFTMAX / LAYERNORM / GELU (FP32 datapath, INT8 I/O)

The programmer inserts SYNC instructions with a 3-bit resource mask to
enforce ordering between units.  Without SYNC, the hardware may overlap
execution of independent units (e.g. a LOAD can overlap a MATMUL).

Reserved fields / opcodes
-------------------------
- Opcodes 0x14–0x1F are reserved.  Decoding a reserved opcode raises an
  illegal-instruction fault and the processor halts.
- M-TYPE stride_log2 [6:3] is reserved and must be zero.
- M-TYPE flags [2:0] are reserved and must be zero.
"""
from enum import IntEnum


class Opcode(IntEnum):
    NOP = 0x00
    HALT = 0x01
    SYNC = 0x02
    CONFIG_TILE = 0x03
    SET_SCALE = 0x04
    SET_ADDR_LO = 0x05
    SET_ADDR_HI = 0x06
    LOAD = 0x07
    STORE = 0x08
    BUF_COPY = 0x09
    MATMUL = 0x0A
    REQUANT = 0x0B
    SCALE_MUL = 0x0C
    VADD = 0x0D
    SOFTMAX = 0x0E
    LAYERNORM = 0x0F
    GELU = 0x10
    REQUANT_PC = 0x11
    SOFTMAX_ATTNV = 0x12
    DEQUANT_ADD = 0x13


class InsnFormat(IntEnum):
    R_TYPE = 0
    M_TYPE = 1
    B_TYPE = 2
    A_TYPE = 3
    C_TYPE = 4
    S_TYPE = 5


OPCODE_FORMAT = {
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
}

# Buffer IDs (2-bit, shared across R-type, M-type, B-type)
BUF_ABUF = 0b00      # Activation buffer (128 KB, INT8)
BUF_WBUF = 0b01      # Weight buffer     (256 KB, INT8)
BUF_ACCUM = 0b10     # Accumulator       ( 64 KB, INT32, little-endian)
BUF_RESERVED = 0b11  # Reserved — raises illegal-buffer fault

BUFFER_NAMES = {BUF_ABUF: "ABUF", BUF_WBUF: "WBUF", BUF_ACCUM: "ACCUM"}

# Per-buffer max offset (in 16-byte units)
ABUF_MAX_OFF = 8191    # 128KB / 16 = 8192 slots, 0-indexed
WBUF_MAX_OFF = 16383   # 256KB / 16
ACCUM_MAX_OFF = 4095   # 64KB / 16

BUFFER_MAX_OFF = {
    BUF_ABUF: ABUF_MAX_OFF,
    BUF_WBUF: WBUF_MAX_OFF,
    BUF_ACCUM: ACCUM_MAX_OFF,
}

# Buffer sizes in bytes
ABUF_SIZE = 128 * 1024
WBUF_SIZE = 256 * 1024
ACCUM_SIZE = 64 * 1024

# Systolic array dimensions
SYSTOLIC_DIM = 16

# --- Bit field positions (from MSB, bit 63 is MSB) ---
# All formats: opcode at [63:59]
OPCODE_SHIFT = 59
OPCODE_MASK = 0x1F

# R-type fields
R_SRC1_BUF_SHIFT = 57
R_SRC1_OFF_SHIFT = 41
R_SRC2_BUF_SHIFT = 39
R_SRC2_OFF_SHIFT = 23
R_DST_BUF_SHIFT = 21
R_DST_OFF_SHIFT = 5
R_SREG_SHIFT = 1
R_FLAGS_SHIFT = 0

# M-type fields (LOAD / STORE)
# Effective DRAM byte address = addr_regs[ADDR_REG] + DRAM_OFF × 16
M_BUF_ID_SHIFT = 57
M_SRAM_OFF_SHIFT = 41
M_XFER_LEN_SHIFT = 25
M_ADDR_REG_SHIFT = 23
M_DRAM_OFF_SHIFT = 7
M_STRIDE_LOG2_SHIFT = 3  # Reserved — must be 0
M_FLAGS_SHIFT = 0         # Reserved — must be 0

# B-type fields
B_SRC_BUF_SHIFT = 57
B_SRC_OFF_SHIFT = 41
B_DST_BUF_SHIFT = 39
B_DST_OFF_SHIFT = 23
B_LENGTH_SHIFT = 7
B_SRC_ROWS_SHIFT = 1
B_TRANSPOSE_SHIFT = 0

# A-type fields
A_ADDR_REG_SHIFT = 57
A_IMM28_SHIFT = 29

# C-type fields
C_M_SHIFT = 49
C_N_SHIFT = 39
C_K_SHIFT = 29

# S-type SET_SCALE fields
SS_SREG_SHIFT = 55
SS_SRC_MODE_SHIFT = 53
SS_IMM16_SHIFT = 37

# S-type SYNC fields
SYNC_RESOURCE_MASK_SHIFT = 56

# Field widths / masks
MASK_2BIT = 0x3
MASK_3BIT = 0x7
MASK_4BIT = 0xF
MASK_5BIT = 0x1F
MASK_6BIT = 0x3F
MASK_10BIT = 0x3FF
MASK_16BIT = 0xFFFF
MASK_28BIT = 0xFFFFFFF
