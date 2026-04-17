// TACCEL ISA v1 -- shared types, parameters, and constants.
//
// This package is the single RTL-side contract for:
//   - instruction opcodes and field layouts
//   - architectural buffer sizes
//   - global bus / array sizing
//   - architectural fault codes
//
// All hardware imports this package. It mirrors the Python constants in
// software/taccel/isa/opcodes.py, so changes here and in software must stay
// synchronized.

`ifndef TACCEL_PKG_SV
`define TACCEL_PKG_SV

package taccel_pkg;

  // -------------------------------------------------------------------------
  // Opcodes (5-bit, instruction bits [63:59])
  // -------------------------------------------------------------------------
  typedef enum logic [4:0] {
    OP_NOP         = 5'h00,
    OP_HALT        = 5'h01,
    OP_SYNC        = 5'h02,
    OP_CONFIG_TILE = 5'h03,
    OP_SET_SCALE   = 5'h04,
    OP_SET_ADDR_LO = 5'h05,
    OP_SET_ADDR_HI = 5'h06,
    OP_LOAD        = 5'h07,
    OP_STORE       = 5'h08,
    OP_BUF_COPY    = 5'h09,
    OP_MATMUL      = 5'h0A,
    OP_REQUANT     = 5'h0B,
    OP_SCALE_MUL   = 5'h0C,
    OP_VADD        = 5'h0D,
    OP_SOFTMAX     = 5'h0E,
    OP_LAYERNORM   = 5'h0F,
    OP_GELU        = 5'h10,
    OP_REQUANT_PC  = 5'h11,
    OP_SOFTMAX_ATTNV = 5'h12,
    OP_DEQUANT_ADD = 5'h13
    // 5'h14–5'h1F: reserved — illegal instruction fault
  } opcode_t;

  // -------------------------------------------------------------------------
  // Buffer IDs (2-bit)
  // -------------------------------------------------------------------------
  typedef enum logic [1:0] {
    BUF_ABUF  = 2'b00,   // Activation buffer: 128 KB INT8
    BUF_WBUF  = 2'b01,   // Weight buffer:     256 KB INT8
    BUF_ACCUM = 2'b10    // Accumulator:        64 KB INT32, little-endian
    // 2'b11 = reserved — illegal buffer fault
  } buf_id_t;

  // -------------------------------------------------------------------------
  // Memory parameters  (1 row = 16 bytes, matching the 16-byte DMA unit)
  // -------------------------------------------------------------------------
  parameter int ABUF_ROWS  = 8192;    // 128 KB / 16 B
  parameter int WBUF_ROWS  = 16384;   // 256 KB / 16 B
  parameter int ACCUM_ROWS = 4096;    //  64 KB / 16 B

  parameter int ABUF_BYTES  = 131072;
  parameter int WBUF_BYTES  = 262144;
  parameter int ACCUM_BYTES =  65536;

  // -------------------------------------------------------------------------
  // Architecture parameters
  // -------------------------------------------------------------------------
  parameter int SYS_DIM        = 16;   // systolic array dimension
  // Systolic architecture modes used during migration.
  parameter int SYS_MODE_BROADCAST = 0;
  parameter int SYS_MODE_CHAINED   = 1;
  // Chained mode is now the default systolic architecture.
  parameter int SYS_MODE_DEFAULT   = SYS_MODE_CHAINED;
  parameter int NUM_SCALE_REGS = 16;   // FP16 scale registers S0–S15
  parameter int NUM_ADDR_REGS  = 4;    // 56-bit DRAM address registers R0–R3
  parameter int ADDR_WIDTH     = 56;   // DRAM address bits
  parameter int AXI_DATA_W     = 128;  // AXI data bus width (bits, 16 bytes/beat)
  parameter int AXI_ADDR_W     = 56;   // AXI address width (bits)

  // -------------------------------------------------------------------------
  // Fault codes
  // -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    FAULT_NONE       = 4'h0,
    FAULT_ILLEGAL_OP = 4'h1,   // reserved opcode 0x14–0x1F
    FAULT_DRAM_OOB   = 4'h2,   // DRAM address out of bounds
    FAULT_SRAM_OOB   = 4'h3,   // SRAM offset out of bounds
    FAULT_NO_CONFIG  = 4'h4,   // compute instruction without CONFIG_TILE
    FAULT_BAD_BUF    = 4'h5,   // buffer ID 0b11 (reserved)
    FAULT_UNSUPPORTED_OP = 4'h6 // legal ISA op or parameter not yet implemented
  } fault_code_t;

  // -------------------------------------------------------------------------
  // Internal-only fault source tags used by the Verilator runner and trace
  // hooks. These are intentionally not exposed through the ISA.
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    OBS_FAULT_SRC_NONE    = 3'd0,
    OBS_FAULT_SRC_FETCH   = 3'd1,
    OBS_FAULT_SRC_DMA     = 3'd2,
    OBS_FAULT_SRC_HELPER  = 3'd3,
    OBS_FAULT_SRC_SFU     = 3'd4,
    OBS_FAULT_SRC_SRAM    = 3'd5,
    OBS_FAULT_SRC_CONTROL = 3'd6
  } obs_fault_src_t;

  // -------------------------------------------------------------------------
  // Decoded instruction struct
  //
  // All format fields are decoded in parallel; only those matching the current
  // opcode's format are architecturally meaningful. This lets decode stay
  // purely combinational and keeps control-unit format selection simple.
  //
  // Bit positions match software/taccel/isa/encoding.py exactly.
  // -------------------------------------------------------------------------
  typedef struct packed {
    // ---- Common ----
    logic [4:0]  opcode;
    logic        illegal;      // 1 = reserved opcode OR reserved buffer ID

    // ---- R-TYPE: MATMUL, REQUANT, SCALE_MUL, VADD, SOFTMAX, LAYERNORM, GELU,
    //              REQUANT_PC, SOFTMAX_ATTNV, DEQUANT_ADD ----
    // Bits [58:57] = src1_buf, [56:41] = src1_off, [40:39] = src2_buf,
    //      [38:23] = src2_off, [22:21] = dst_buf,  [20:5]  = dst_off,
    //      [4:1]   = sreg,     [0]     = flags
    logic [1:0]  src1_buf;
    logic [15:0] src1_off;
    logic [1:0]  src2_buf;
    logic [15:0] src2_off;
    logic [1:0]  dst_buf;
    logic [15:0] dst_off;
    logic [3:0]  sreg;
    logic        flags;

    // ---- M-TYPE: LOAD, STORE ----
    // Bits [58:57]=buf_id, [56:41]=sram_off, [40:25]=xfer_len,
    //      [24:23]=addr_reg, [22:7]=dram_off
    logic [1:0]  m_buf_id;
    logic [15:0] m_sram_off;
    logic [15:0] m_xfer_len;
    logic [1:0]  m_addr_reg;
    logic [15:0] m_dram_off;

    // ---- B-TYPE: BUF_COPY ----
    // Bits [58:57]=src_buf, [56:41]=src_off, [40:39]=dst_buf,
    //      [38:23]=dst_off, [22:7]=length, [6:1]=src_rows, [0]=transpose
    logic [1:0]  b_src_buf;
    logic [15:0] b_src_off;
    logic [1:0]  b_dst_buf;
    logic [15:0] b_dst_off;
    logic [15:0] b_length;
    logic [5:0]  b_src_rows;
    logic        b_transpose;

    // ---- A-TYPE: SET_ADDR_LO, SET_ADDR_HI ----
    // Bits [58:57]=addr_reg, [56:29]=imm28
    logic [1:0]  a_addr_reg;
    logic [27:0] a_imm28;

    // ---- C-TYPE: CONFIG_TILE ----
    // Bits [58:49]=M, [48:39]=N, [38:29]=K  (0-based; actual = field + 1)
    logic [9:0]  c_tile_m;
    logic [9:0]  c_tile_n;
    logic [9:0]  c_tile_k;

    // ---- S-TYPE: SET_SCALE ----
    // Bits [58:55]=sreg, [54:53]=src_mode, [52:37]=imm16
    logic [3:0]  s_sreg;
    logic [1:0]  s_src_mode;   // 0=imm, 1=ABUF, 2=WBUF, 3=ACCUM
    logic [15:0] s_imm16;

    // ---- S-TYPE: SYNC ----
    // Bits [58:56]=resource_mask  (bit0=DMA, bit1=Systolic, bit2=SFU)
    logic [2:0]  sync_mask;

  } decoded_insn_t;

  // -------------------------------------------------------------------------
  // SYNC resource mask bit positions
  // -------------------------------------------------------------------------
  parameter int SYNC_DMA_BIT = 0;
  parameter int SYNC_SYS_BIT = 1;
  parameter int SYNC_SFU_BIT = 2;

  // -------------------------------------------------------------------------
  // Erf polynomial coefficients (Abramowitz & Stegun 7.1.26) as FP32 bits
  // Max absolute error < 5e-7 in FP32 — far below INT8 noise floor ~4e-3.
  // Used by sfu_gelu in Phase 5.
  // -------------------------------------------------------------------------
  parameter logic [31:0] ERF_A1 = 32'h3E827906;  //  0.254829592
  parameter logic [31:0] ERF_A2 = 32'hBE91A98E;  // -0.284496736
  parameter logic [31:0] ERF_A3 = 32'h3FB5D78E;  //  1.421413741
  parameter logic [31:0] ERF_A4 = 32'hBFBA0005;  // -1.453152027
  parameter logic [31:0] ERF_A5 = 32'h3F87DC22;  //  1.061405429
  parameter logic [31:0] ERF_P  = 32'h3EA7B9D2;  //  0.3275911

endpackage

`endif // TACCEL_PKG_SV
