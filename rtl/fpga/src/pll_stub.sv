// PLL placeholder — passes the input clock straight through.
//
// Replace with vendor PLL/MMCM IP when the target FPGA is picked:
//   - Xilinx Vivado: MMCME2_ADV / PLLE2_ADV (or Clocking Wizard IP)
//   - Intel/Altera Quartus: ALTPLL / IOPLL
//   - Lattice nextpnr / Yosys: EHXPLLL (ECP5) etc.
//
// Stub-only so the FPGA wrapper elaborates without a vendor IP environment.
// Does not provide clock multiplication, phase shift, locking, or jitter
// cleanup.

`ifndef PLL_STUB_SV
`define PLL_STUB_SV

module pll_stub (
  input  logic clk_in,
  output logic clk_out
);
  assign clk_out = clk_in;
endmodule

`endif // PLL_STUB_SV
