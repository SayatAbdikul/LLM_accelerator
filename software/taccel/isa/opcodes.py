"""ISA opcode definitions, instruction formats, and field constants.

TACCEL ISA v1 — 64-bit fixed-width instructions, big-endian encoding.

Architecture overview
---------------------
Three execution units operate behind an in-order issue stage:
  - DMA   : LOAD / STORE (DRAM ↔ SRAM)
  - Systolic : MATMUL (INT8×INT8 → INT32, 16×16 tiled)
  - SFU   : SOFTMAX / LAYERNORM / GELU (FP32 datapath, INT8 I/O *and* FP32 I/O)

The programmer inserts SYNC instructions with a 3-bit resource mask to
enforce ordering between units.  Without SYNC, the hardware may overlap
execution of independent units (e.g. a LOAD can overlap a MATMUL).

W8A32 extension (Phase 3 (c.1), milestones M1 + M2.5-A, 2026-05-12)
------------------------------------------------------------------
Nine new R-type opcodes (0x17–0x1F) extend the ISA to support FP32
inter-layer activations + dynamic per-matmul activation scaling while
preserving INT8 MXU matmul:

  M1 (commit `47141fb`):
    0x17 DEQUANT_ACCUM_FP32:    per-channel dequant of INT32 ACCUM → FP32 ABUF
    0x18 QUANT_FP32_INT8:       per-tensor INT8 quant of FP32 ABUF → INT8 ABUF
    0x19 VADD_FP32:             element-wise FP32 add (residual stream)
    0x1A LAYERNORM_FP32:        LayerNorm with FP32 I/O
    0x1B GELU_FP32:             GELU with FP32 I/O
    0x1C SOFTMAX_FP32:          row-wise softmax with FP32 I/O
    0x1D MASKED_SOFTMAX_FP32:   causal softmax with FP32 I/O

  M2.5-A (this commit):
    0x1E DEQUANT_ACCUM_FP32_SCALED:
       like DEQUANT_ACCUM_FP32 but additionally multiplies by an FP16
       scalar from a scale register. Used by W8A32 matmul-output
       lowering to apply the dynamic per-matmul activation scale
       (max_abs/127) on top of the static per-channel weight scales.
       M1's 0x17 op stays bit-identical to its M1 contract — 0x1E is
       a separate opcode that adds the scalar multiply.
    0x1F MAX_ABS_REDUCE_FP32:
       scans an FP32 ABUF tile, computes max(|x|), and writes derived
       FP16 scales to a register pair: scale_regs[sreg] = 127/max_abs
       (for QUANT_FP32_INT8 input scaling), scale_regs[sreg+1] =
       max_abs/127 (for DEQUANT_ACCUM_FP32_SCALED output scaling).
       Eps-guarded for all-zero tiles.

There is no buffer dtype tag — ABUF bytes are reinterpreted as INT8
(1 byte/elem) or FP32 (4 bytes/elem) based on the opcode. Codegen owns
the dtype layout. A 16x16 FP32 tile occupies 64 16-byte buffer units
versus 16 units for the corresponding INT8 tile.

Reserved fields / opcodes
-------------------------
- After M2.5-A, the entire 5-bit opcode space (0x00–0x1F) is in use.
  No opcodes are reserved. Future ISA extensions must reuse a slot
  via a CONFIG-style prefix, expand the opcode field, or relocate an
  existing rarely-used encoding.
- CONFIG_ATTN reserved bits [32:0] must be zero.
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
    Opcode.SOFTMAX: InsnFormat.R_TYPE,
    Opcode.LAYERNORM: InsnFormat.R_TYPE,
    Opcode.GELU: InsnFormat.R_TYPE,
    Opcode.REQUANT_PC: InsnFormat.R_TYPE,
    Opcode.SOFTMAX_ATTNV: InsnFormat.R_TYPE,
    Opcode.DEQUANT_ADD: InsnFormat.R_TYPE,
    Opcode.CONFIG_ATTN: InsnFormat.ATTN_TYPE,
    Opcode.MASKED_SOFTMAX: InsnFormat.R_TYPE,
    Opcode.MASKED_SOFTMAX_ATTNV: InsnFormat.R_TYPE,
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
