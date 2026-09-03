"""Current TACCEL opcode allocation, instruction formats, and field constants.

Instructions are fixed-width 64-bit words serialized big-endian, with the
5-bit opcode in bits [63:59]. Four resources sit behind the in-order control
plane: DMA, the 16×16 INT8 systolic array, the generation-2 SFU, and the
blocking helper engine. SYNC masks cover DMA, systolic, and SFU; the helper is
blocking.

The maintained hardware target is W8A16. Generation-2 R-type opcodes
0x17..0x1F use FP32 internal arithmetic and require flags[0]=1 in RTL.
SOFTMAX_FP32 (0x1C) remains Python/golden-only; causal frontends use
MASKED_SOFTMAX_FP32 (0x1D). The flags=0 W8A32-storage form is likewise a
Python research path.

Six generation-1 SFU opcode numbers (0x0E, 0x0F, 0x10, 0x12, 0x15, 0x16)
remain named only so old binaries receive clear retirement diagnostics. The
integer compatibility helpers at 0x0B, 0x0C, 0x0D, 0x11, and 0x13 remain
implemented.

Current extensions inside formerly reserved fields are:
  - M-type [6:3] ``cols_log2`` and [0] ``transpose`` for transposed LOAD;
  - CONFIG_TILE [27:16] ``m_exact`` for the exact SFU row count;
  - CONFIG_TILE [28] ``weight_int4`` in Python only. RTL ignores bit [28], so
    hardware-target programs must leave it zero.

See ``software/docs/isa_spec.md`` for the complete support and field matrix.
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
    CONFIG_ATTN = 0x14
    MASKED_SOFTMAX = 0x15
    MASKED_SOFTMAX_ATTNV = 0x16
    # W8A32 extension (M1, 2026-05-12): FP32-I/O variants for inter-layer
    # activations. All R-type; ABUF bytes reinterpreted as FP32.
    DEQUANT_ACCUM_FP32 = 0x17
    QUANT_FP32_INT8 = 0x18
    VADD_FP32 = 0x19
    LAYERNORM_FP32 = 0x1A
    GELU_FP32 = 0x1B
    SOFTMAX_FP32 = 0x1C
    MASKED_SOFTMAX_FP32 = 0x1D
    # W8A32 extension (M2.5-A, 2026-05-12): dynamic per-matmul activation
    # scaling primitives. M1's 0x17 stays bit-identical to its shipped
    # contract; 0x1E is the separate "scaled" variant.
    DEQUANT_ACCUM_FP32_SCALED = 0x1E
    MAX_ABS_REDUCE_FP32 = 0x1F


RETIRED_OPCODES = frozenset({
    Opcode.SOFTMAX,
    Opcode.LAYERNORM,
    Opcode.GELU,
    Opcode.SOFTMAX_ATTNV,
    Opcode.MASKED_SOFTMAX,
    Opcode.MASKED_SOFTMAX_ATTNV,
})


class InsnFormat(IntEnum):
    R_TYPE = 0
    M_TYPE = 1
    B_TYPE = 2
    A_TYPE = 3
    C_TYPE = 4
    S_TYPE = 5
    ATTN_TYPE = 6


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
    Opcode.REQUANT_PC: InsnFormat.R_TYPE,
    Opcode.DEQUANT_ADD: InsnFormat.R_TYPE,
    Opcode.CONFIG_ATTN: InsnFormat.ATTN_TYPE,
    # W8A32 R-type extension (M1)
    Opcode.DEQUANT_ACCUM_FP32: InsnFormat.R_TYPE,
    Opcode.QUANT_FP32_INT8: InsnFormat.R_TYPE,
    Opcode.VADD_FP32: InsnFormat.R_TYPE,
    Opcode.LAYERNORM_FP32: InsnFormat.R_TYPE,
    Opcode.GELU_FP32: InsnFormat.R_TYPE,
    Opcode.SOFTMAX_FP32: InsnFormat.R_TYPE,
    Opcode.MASKED_SOFTMAX_FP32: InsnFormat.R_TYPE,
    # W8A32 R-type extension (M2.5-A)
    Opcode.DEQUANT_ACCUM_FP32_SCALED: InsnFormat.R_TYPE,
    Opcode.MAX_ABS_REDUCE_FP32: InsnFormat.R_TYPE,
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
M_STRIDE_LOG2_SHIFT = 3  # cols_log2 for transposed LOAD; 0 for plain transfer
M_FLAGS_SHIFT = 0         # transpose for LOAD; 0 for canonical STORE

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
# W4A16 plan Phase 2 (2026-05-24). Bit [28] of a CONFIG_TILE instruction
# selects packed W4 weight interpretation in Python bundle/golden paths. RTL
# currently ignores this bit and still reads INT8 lanes, so it must remain zero
# in hardware-target programs. Bits [27:16] are assigned to m_exact below.
# The bit lives inside the existing C-type field allocation (M/N/K
# occupy [58:29], leaving [28:0] free) so adding it is encoder-additive
# and does NOT widen the instruction word. Default 0 keeps every existing
# W8 CONFIG_TILE bit-identical. See plan §3.1 + [[w4a16-phase1-quality]].
C_WEIGHT_INT4_SHIFT = 28
# m_exact (freeze §6 rev 2026-07-10). Bits [27:16] of CONFIG_TILE carry an
# exact SFU row count: 0 = full tiles ((M+1)*16 rows, legacy — every
# pre-existing bundle encodes 0), k>0 = the SFU row loops walk exactly k
# rows. Consumed ONLY by the SFU engine (RTL: sfu_engine dispatch_m_rows_w
# mux; golden: the SFU _exec_* row bounds). Systolic MATMUL and the helper
# engine keep the tile-quantized (M+1)*16. Encoder-additive inside the free
# C-type bits — default 0 keeps every existing CONFIG_TILE bit-identical.
C_M_EXACT_SHIFT = 16

# ATTN-type CONFIG_ATTN fields
ATTN_QUERY_ROW_BASE_SHIFT = 47
ATTN_VALID_KV_LEN_SHIFT = 35
ATTN_MODE_SHIFT = 33
ATTN_RESERVED_MASK = (1 << 33) - 1

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
MASK_12BIT = 0xFFF
MASK_16BIT = 0xFFFF
MASK_28BIT = 0xFFFFFFF
