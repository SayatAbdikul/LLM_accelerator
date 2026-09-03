// SKY130 PDK SRAM macro binding for the `sram_dp_macro` slot.
//
// Declares `module sram_dp_macro` with a BEHAVIORAL register-array body so the
// ASIC wrapper elaborates cleanly. Despite the filename, this file does not
// instantiate a SKY130 SRAM macro. A physical implementation must replace it
// with characterized banks matching {DATA_W, DEPTH} and the required port
// behavior.
//
// Bank-target sizes:
//   ABUF  : DATA_W=128, DEPTH=8192  → 128 KB
//   WBUF  : DATA_W=128, DEPTH=16384 → 256 KB
//   ACCUM : DATA_W=128, DEPTH=4096  → 64  KB
//
// These logical capacities have not been floorplanned or proven to fit a
// particular open-PDK shuttle or die.

`ifndef SRAM_DP_SKY130_SV
`define SRAM_DP_SKY130_SV

module sram_dp_macro #(
  parameter int DATA_W = 128,
  parameter int DEPTH  = 8192
)(
  input  logic                          clk,

  // Port A — read/write
  input  logic                          a_en,
  input  logic                          a_we,
  input  logic [$clog2(DEPTH)-1:0]      a_addr,
  input  logic [DATA_W-1:0]             a_wdata,
  output logic [DATA_W-1:0]             a_rdata,

  // Port B — read only
  input  logic                          b_en,
  input  logic [$clog2(DEPTH)-1:0]      b_addr,
  output logic [DATA_W-1:0]             b_rdata
);

  // Behavioral stub — replace with characterized macro banks before physical
  // implementation. Behavior matches the inferred body for elaboration.
  logic [DATA_W-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (a_en) begin
      if (a_we)
        mem[a_addr] <= a_wdata;
      a_rdata <= a_we ? a_wdata : mem[a_addr];
    end
  end

  always_ff @(posedge clk) begin
    if (b_en)
      b_rdata <= mem[b_addr];
  end

endmodule

`endif // SRAM_DP_SKY130_SV
