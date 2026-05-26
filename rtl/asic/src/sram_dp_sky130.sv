// SKY130 PDK SRAM macro binding for the `sram_dp_macro` slot.
//
// Step E (2026-05-26 RTL restructure): declares `module sram_dp_macro`
// with a BEHAVIORAL stub body (equivalent to sram_dp_inferred) so the
// ASIC wrapper elaborates cleanly. The real macro instantiations land
// when OpenLane is wired up — at that point this body becomes a bank of
// `sky130_sram_1rw1r_*` macros composed to match {DATA_W, DEPTH}.
//
// Bank-target sizes:
//   ABUF  : DATA_W=128, DEPTH=8192  → 128 KB
//   WBUF  : DATA_W=128, DEPTH=16384 → 256 KB
//   ACCUM : DATA_W=128, DEPTH=4096  → 64  KB
//
// At ~1 mm²/2 KB on sky130A this is way more SRAM than fits in a Caravel
// user-area slot; see top-level README's tape-out strategy section for
// the substrate-IP vs full-model trade-off.

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

  // Behavioral stub — to be replaced with banked sky130_sram_* macro
  // instantiations once OpenLane is set up. Behavior matches the inferred
  // body (sram_dp_inferred) so the dispatch wrapper's two branches are
  // logically equivalent for the elaboration smoke gate.
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
