// Blackbox stubs for the two non-synthesizable modules (sfu_engine,
// blocking_helper_engine). Used ONLY by the Phase-0 control-plane diagnostic
// (not by the normal `make synth-check` gate, which reads the real modules
// and is correctly RED until those are rewritten as synthesizable RTL).
//
// Port lists mirror the real modules; (* blackbox *) tells yosys to treat
// them as opaque IP (no implementation needed). This lets us prove the
// control plane synthesizes cleanly in isolation, even while SFU/helper
// still contain real/DPI behavioral code.

(* blackbox *)
module sfu_engine (
  input  wire         clk,
  input  wire         rst_n,
  input  wire         dispatch,
  input  wire [4:0]   opcode,
  input  wire [1:0]   src1_buf,
  input  wire [15:0]  src1_off,
  input  wire [1:0]   src2_buf,
  input  wire [15:0]  src2_off,
  input  wire [1:0]   dst_buf,
  input  wire [15:0]  dst_off,
  input  wire [3:0]   sreg,
  input  wire [9:0]   tile_m,
  input  wire [9:0]   tile_n,
  input  wire [9:0]   tile_k,
  input  wire         attn_valid,
  input  wire [11:0]  attn_query_row_base,
  input  wire [11:0]  attn_valid_kv_len,
  input  wire [1:0]   attn_mode,
  input  wire [15:0]  scale0_data,
  input  wire [15:0]  scale1_data,
  input  wire [15:0]  scale2_data,
  input  wire [15:0]  scale3_data,
  output wire         sfu_busy,
  output wire         sfu_fault,
  output wire [3:0]   sfu_fault_code,
  output wire         sram_a_en,
  output wire         sram_a_we,
  output wire [1:0]   sram_a_buf,
  output wire [15:0]  sram_a_row,
  output wire [127:0] sram_a_wdata,
  input  wire         sram_a_fault,
  output wire         sram_b_en,
  output wire [1:0]   sram_b_buf,
  output wire [15:0]  sram_b_row,
  input  wire [127:0] sram_b_rdata,
  input  wire         sram_b_fault,
  output wire         sfu_scale_we,
  output wire [3:0]   sfu_scale_waddr,
  output wire [15:0]  sfu_scale_wdata
);
endmodule

(* blackbox *)
module blocking_helper_engine (
  input  wire         clk,
  input  wire         rst_n,
  input  wire         dispatch,
  input  wire [4:0]   opcode,
  input  wire [1:0]   src1_buf,
  input  wire [15:0]  src1_off,
  input  wire [1:0]   src2_buf,
  input  wire [15:0]  src2_off,
  input  wire [1:0]   dst_buf,
  input  wire [15:0]  dst_off,
  input  wire [3:0]   sreg,
  input  wire [15:0]  b_length,
  input  wire [5:0]   b_src_rows,
  input  wire         b_transpose,
  input  wire [9:0]   tile_m,
  input  wire [9:0]   tile_n,
  input  wire [15:0]  scale0_data,
  input  wire [15:0]  scale1_data,
  output wire         helper_busy,
  output wire         helper_fault,
  output wire [3:0]   helper_fault_code,
  output wire         sram_a_en,
  output wire         sram_a_we,
  output wire [1:0]   sram_a_buf,
  output wire [15:0]  sram_a_row,
  output wire [127:0] sram_a_wdata,
  input  wire [127:0] sram_a_rdata,
  input  wire         sram_a_fault,
  output wire         sram_b_en,
  output wire [1:0]   sram_b_buf,
  output wire [15:0]  sram_b_row,
  input  wire [127:0] sram_b_rdata,
  input  wire         sram_b_fault
);
endmodule
