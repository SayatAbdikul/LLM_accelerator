// Inferred dual-port SRAM body — used by Verilator simulation, the Yosys
// generic synth-check gate, and FPGA build targets. The (* ram_style = "block" *)
// attribute is consumed by FPGA synth tools (Vivado, Yosys+nextpnr) to map
// this array to block RAM. The body was extracted verbatim from sram_dp.sv
// during the Step B RTL restructure (target-dispatch wrapper introduction)
// so its behavior is bit-identical to the pre-restructure baseline.
//
// Module name `sram_dp_inferred` is intentionally distinct from `sram_dp`;
// the public-facing module name `sram_dp` lives in the dispatch wrapper
// (sram_dp.sv) which selects this module for SIM/FPGA targets and the
// ASIC PDK SRAM macro (sram_dp_macro) for TARGET_ASIC.

`ifndef SRAM_DP_INFERRED_SV
`define SRAM_DP_INFERRED_SV

module sram_dp_inferred #(
  parameter int DATA_W = 128,     // bits per row (must be 128 for TACCEL)
  parameter int DEPTH  = 8192     // number of rows
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

  // Shared storage array. The wrapper modules above this one decide which
  // architectural buffer is being addressed and suppress OOB accesses.
  (* ram_style = "block" *)
  logic [DATA_W-1:0] mem [0:DEPTH-1];

  // Port A: synchronous read/write
  always_ff @(posedge clk) begin
    if (a_en) begin
      if (a_we)
        mem[a_addr] <= a_wdata;
      a_rdata <= a_we ? a_wdata : mem[a_addr];
    end
  end

  // Port B: synchronous read only
  always_ff @(posedge clk) begin
    if (b_en)
      b_rdata <= mem[b_addr];
  end

endmodule

`endif // SRAM_DP_INFERRED_SV
