// Parameterized synchronous dual-port SRAM.
//
// Port A: read/write (DMA, BUF_COPY, REQUANT write-back)
// Port B: read-only  (Systolic array, SFU read)
//
// Both ports are synchronous on the rising edge of clk.
// Write-first semantics on port A: if A writes and reads the same address
// in the same cycle, the new (written) data appears on a_rdata next cycle.
// Port B is always read-only — no write path.
//
// Data width: 128 bits (16 bytes) per row, matching the 16-byte DMA unit.
// FPGA synthesis: infer block RAM with (* ram_style = "block" *).

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

`endif // SRAM_DP_SV
