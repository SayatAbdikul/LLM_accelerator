// Testbench wrapper for decode_unit: exposes every field as a named scalar port
// so Verilator C++ tests can access them directly without bit-twiddling.
// Only used in simulation; not part of synthesis netlist.

`ifndef TB_DECODE_UNIT_SV
`define TB_DECODE_UNIT_SV

`include "taccel_pkg.sv"

module tb_decode_unit
  import taccel_pkg::*;
(
  input  logic [63:0] insn_data,

  // Common
  output logic [4:0]  opcode,
  output logic        illegal,

  // R-type
  output logic [1:0]  src1_buf,
  output logic [15:0] src1_off,
  output logic [1:0]  src2_buf,
  output logic [15:0] src2_off,
  output logic [1:0]  dst_buf,
  output logic [15:0] dst_off,
  output logic [3:0]  sreg,
  output logic        flags,

  // M-type
  output logic [1:0]  m_buf_id,
  output logic [15:0] m_sram_off,
  output logic [15:0] m_xfer_len,
  output logic [1:0]  m_addr_reg,
  output logic [15:0] m_dram_off,

  // B-type
  output logic [1:0]  b_src_buf,
  output logic [15:0] b_src_off,
  output logic [1:0]  b_dst_buf,
  output logic [15:0] b_dst_off,
  output logic [15:0] b_length,
  output logic [5:0]  b_src_rows,
  output logic        b_transpose,

  // A-type
  output logic [1:0]  a_addr_reg,
  output logic [27:0] a_imm28,

  // C-type
  output logic [9:0]  c_tile_m,
  output logic [9:0]  c_tile_n,
  output logic [9:0]  c_tile_k,

  // S-type SET_SCALE
  output logic [3:0]  s_sreg,
  output logic [1:0]  s_src_mode,
  output logic [15:0] s_imm16,

  // S-type SYNC
  output logic [2:0]  sync_mask
);

  decoded_insn_t dec;

  decode_unit u_decode (
    .insn_data (insn_data),
    .insn      (dec)
  );

  // Break out all fields
  assign opcode      = dec.opcode;
  assign illegal     = dec.illegal;
  assign src1_buf    = dec.src1_buf;
  assign src1_off    = dec.src1_off;
  assign src2_buf    = dec.src2_buf;
  assign src2_off    = dec.src2_off;
  assign dst_buf     = dec.dst_buf;
  assign dst_off     = dec.dst_off;
  assign sreg        = dec.sreg;
  assign flags       = dec.flags;
  assign m_buf_id    = dec.m_buf_id;
  assign m_sram_off  = dec.m_sram_off;
  assign m_xfer_len  = dec.m_xfer_len;
  assign m_addr_reg  = dec.m_addr_reg;
  assign m_dram_off  = dec.m_dram_off;
  assign b_src_buf   = dec.b_src_buf;
  assign b_src_off   = dec.b_src_off;
  assign b_dst_buf   = dec.b_dst_buf;
  assign b_dst_off   = dec.b_dst_off;
  assign b_length    = dec.b_length;
  assign b_src_rows  = dec.b_src_rows;
  assign b_transpose = dec.b_transpose;
  assign a_addr_reg  = dec.a_addr_reg;
  assign a_imm28     = dec.a_imm28;
  assign c_tile_m    = dec.c_tile_m;
  assign c_tile_n    = dec.c_tile_n;
  assign c_tile_k    = dec.c_tile_k;
  assign s_sreg      = dec.s_sreg;
  assign s_src_mode  = dec.s_src_mode;
  assign s_imm16     = dec.s_imm16;
  assign sync_mask   = dec.sync_mask;

endmodule

`endif // TB_DECODE_UNIT_SV
