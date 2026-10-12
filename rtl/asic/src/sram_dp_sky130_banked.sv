// EXPERIMENT ONLY: bank/lane adapter for OpenRAM's SKY130 2-KiB 1RW1R macro.
// This is deliberately NOT named sram_dp_macro and is not in the ASIC filelist.
// A one-entry delayed-write overlay preserves the shared inferred memory's
// write-first A / old-data B semantics without issuing the macro's undefined
// same-row write/read collision. It is not yet a production SRAM binding.
//
// One logical row spans DATA_W/32 macros. Each bank holds 512 logical rows.
// Requests are sampled on the rising edge. The vendor model produces reads
// after the falling edge plus DELAY; they must settle before the next rising
// edge, where core consumers sample them. This model test is not timing signoff.
// Port A writes replace the pending entry, committing its prior contents. A
// reads defer commit; idle A cycles commit and empty the entry. Reads of the
// pending row forward its PRE-EDGE value, including when a new write replaces
// it. Output holds prevent disabled macros' X outputs from leaking.
//
// Like the inferred SRAM, this interface has no reset and initial contents /
// outputs are unspecified. No initialized valid bit is required: arbitrary
// initial macro memory plus an arbitrary valid pending entry describes an
// arbitrary initial logical memory. The first A write defines its row, and
// every subsequent access preserves that logical memory view. Do not assume
// physical macro contents alone include the most recent pending write.
`timescale 1ns/1ps

module sram_dp_sky130_banked #(
  parameter integer DATA_W = 128,
  parameter integer DEPTH = 1024
)(
  input wire clk,
  input wire a_en,
  input wire a_we,
  input wire [$clog2(DEPTH)-1:0] a_addr,
  input wire [DATA_W-1:0] a_wdata,
  output wire [DATA_W-1:0] a_rdata,
  input wire b_en,
  input wire [$clog2(DEPTH)-1:0] b_addr,
  output wire [DATA_W-1:0] b_rdata
);
  localparam integer BANKS = DEPTH / 512;
  localparam integer LANES = DATA_W / 32;
  localparam integer BANK_W = (BANKS > 1) ? $clog2(BANKS) : 1;
  localparam integer ADDR_W = $clog2(DEPTH);

  reg pending_valid_q;
  reg [ADDR_W-1:0] pending_addr_q;
  reg [DATA_W-1:0] pending_data_q;
  wire a_read_w = a_en && !a_we;
  wire a_forward_w = a_read_w && pending_valid_q && a_addr == pending_addr_q;
  wire b_forward_w = b_en && pending_valid_q && b_addr == pending_addr_q;
  wire commit_w = pending_valid_q && !a_read_w;
  wire macro_a_en_w = commit_w || (a_read_w && !a_forward_w);
  wire [ADDR_W-1:0] macro_a_addr_w = commit_w ? pending_addr_q : a_addr;
  // Forwarding also disables B during every physical commit collision.
  wire macro_b_en_w = b_en && !b_forward_w;

  wire [DATA_W-1:0] macro_a [0:BANKS-1];
  wire [DATA_W-1:0] macro_b [0:BANKS-1];
  reg [BANK_W-1:0] a_bank_q, b_bank_q;
  reg a_read_q, a_write_q, b_read_q;
  reg a_forward_q, b_forward_q;
  reg [DATA_W-1:0] a_forward_data_q, b_forward_data_q;
  reg [DATA_W-1:0] a_write_data_q, a_hold_q, b_hold_q;
  wire [DATA_W-1:0] selected_a = macro_a[a_bank_q];
  wire [DATA_W-1:0] selected_b = macro_b[b_bank_q];
  wire [DATA_W-1:0] a_read_data_w = a_forward_q ? a_forward_data_q : selected_a;
  wire [DATA_W-1:0] b_read_data_w = b_forward_q ? b_forward_data_q : selected_b;

  assign a_rdata = a_write_q ? a_write_data_q : (a_read_q ? a_read_data_w : a_hold_q);
  assign b_rdata = b_read_q ? b_read_data_w : b_hold_q;

  always @(posedge clk) begin
    if (a_read_q) a_hold_q <= a_read_data_w;
    else if (a_write_q) a_hold_q <= a_write_data_q;
    if (b_read_q) b_hold_q <= b_read_data_w;
    a_read_q <= a_read_w;
    a_write_q <= a_en && a_we;
    b_read_q <= b_en;
    a_forward_q <= a_forward_w;
    b_forward_q <= b_forward_w;
    // Nonblocking capture retains the previous entry for old-data B, even
    // when the current A write replaces that same row with different data.
    if (a_forward_w) a_forward_data_q <= pending_data_q;
    if (b_forward_w) b_forward_data_q <= pending_data_q;
    if (a_en) a_bank_q <= BANK_W'(a_addr >> 9);
    if (b_en) b_bank_q <= BANK_W'(b_addr >> 9);
    if (a_en && a_we) begin
      a_write_data_q <= a_wdata;
      pending_valid_q <= 1'b1;
      pending_addr_q <= a_addr;
      pending_data_q <= a_wdata;
    end else if (!a_read_w) begin
      pending_valid_q <= 1'b0;
    end
  end

  genvar bank, lane;
  generate
    for (bank = 0; bank < BANKS; bank = bank + 1) begin : banks
      for (lane = 0; lane < LANES; lane = lane + 1) begin : lanes
        sky130_sram_2kbyte_1rw1r_32x512_8
`ifndef SYNTHESIS
          #(.VERBOSE(0))
`endif
        u_macro (
          .clk0(clk),
          .csb0(!(macro_a_en_w && ((macro_a_addr_w >> 9) == bank))),
          .web0(!commit_w),
          .wmask0(4'hf),
          .addr0(macro_a_addr_w[8:0]),
          .din0(pending_data_q[lane*32 +: 32]),
          .dout0(macro_a[bank][lane*32 +: 32]),
          .clk1(clk),
          .csb1(!(macro_b_en_w && ((b_addr >> 9) == bank))),
          .addr1(b_addr[8:0]),
          .dout1(macro_b[bank][lane*32 +: 32])
        );
      end
    end
  endgenerate

`ifndef SYNTHESIS
  initial begin
    if (DATA_W < 32 || DATA_W % 32 != 0 || DEPTH < 512 ||
        DEPTH % 512 != 0 || (DEPTH & (DEPTH - 1)) != 0)
      $fatal(1, "SKY130 experiment requires width multiple of 32 and power-of-two depth >=512");
  end
  always @(posedge clk) begin
    if (commit_w && macro_b_en_w && pending_addr_q == b_addr)
      $fatal(1, "SKY130 adapter issued undefined physical write/read collision");
  end
`endif
endmodule
