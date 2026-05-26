// Target-dispatch wrapper for the dual-port SRAM macro family.
//
// Port A: read/write (DMA, BUF_COPY, REQUANT write-back)
// Port B: read-only  (Systolic array, SFU read)
//
// Both ports are synchronous on the rising edge of clk. Write-first
// semantics on port A: if A writes and reads the same address in the
// same cycle, the new (written) data appears on a_rdata next cycle.
// Port B is always read-only — no write path.
//
// Data width: 128 bits (16 bytes) per row, matching the 16-byte DMA unit.
//
// Target dispatch:
//   - TARGET_ASIC : binds to `sram_dp_macro` (PDK SRAM macro wrapper),
//                   defined in rtl/asic/src/sram_dp_<pdk>.sv.
//   - default     : binds to `sram_dp_inferred` (inferred BRAM body with
//                   `ram_style = "block"` attribute). Used by Verilator
//                   simulation, the Yosys generic synth-check gate, and
//                   FPGA build targets.
//
// The module name `sram_dp` and its port set are preserved across this
// wrapper so sram_subsystem.sv (and any other instantiator) continues to
// work unchanged.

`ifndef SRAM_DP_SV
`define SRAM_DP_SV

module sram_dp #(
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

`ifdef TARGET_ASIC
  sram_dp_macro    #(.DATA_W(DATA_W), .DEPTH(DEPTH)) u_impl (.*);
`else  // TARGET_FPGA or TARGET_SIM (default for Verilator + synth-check)
  sram_dp_inferred #(.DATA_W(DATA_W), .DEPTH(DEPTH)) u_impl (.*);
`endif

endmodule

`endif // SRAM_DP_SV
