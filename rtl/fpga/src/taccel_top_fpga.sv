// FPGA top wrapper around the verified `taccel_top` core.
//
// Research integration wrapper exposing board-level clock/reset/control pins
// and routing the core's AXI master to a placeholder DDR controller. There is
// no selected device, board constraint set, real memory controller, or
// bitstream flow. See rtl/TESTBENCHES.md for current conformance coverage.
//
// When a target FPGA part is picked, replace pll_stub / iobuf_stub /
// ddr_axi_stub with vendor IP (MMCM/MIG/etc.) — the core instantiation
// here doesn't change.

`ifndef TACCEL_TOP_FPGA_SV
`define TACCEL_TOP_FPGA_SV

// Belt-and-suspenders guard. The FPGA build flow MUST set
// SFU_SYNTH_NO_DPI; synthesis cannot consume DPI imports.
`ifndef SFU_SYNTH_NO_DPI
  `error "TARGET_FPGA requires SFU_SYNTH_NO_DPI; FPGA synthesis cannot consume DPI imports."
`endif

`include "taccel_pkg.sv"

module taccel_top_fpga
  import taccel_pkg::*;
#(
  parameter int SYSTOLIC_ARCH_MODE = SYS_MODE_DEFAULT,
  parameter int DRAM_SIZE          = 1 << 24,
  // FPGA target always uses the synth-friendly compute paths.
  parameter int SFU_SYNTH_MODE     = 1,
  parameter int HELPER_SYNTH_MODE  = 1
)(
  // Board-level pins (placeholder; widened/replaced when a part is picked)
  input  logic        clk_pin,
  input  logic        rst_n_pin,
  input  logic        start,
  output logic        done,
  output logic        fault,
  output logic [3:0]  fault_code
);

  // ----------------------------------------------------------------------
  // Clock + reset distribution (vendor IP placeholders)
  // ----------------------------------------------------------------------
  logic clk, rst_n;

  pll_stub u_pll (
    .clk_in (clk_pin),
    .clk_out(clk)
  );

  iobuf_stub u_rst_sync (
    .pin(rst_n_pin),
    .clk(clk),
    .out(rst_n)
  );

  // ----------------------------------------------------------------------
  // Internal AXI4 master signals between the core and the DDR stub
  // ----------------------------------------------------------------------
  logic [AXI_ADDR_W-1:0] m_axi_ar_addr;
  logic                  m_axi_ar_valid;
  logic [7:0]            m_axi_ar_len;
  logic [2:0]            m_axi_ar_size;
  logic [1:0]            m_axi_ar_burst;
  logic                  m_axi_ar_ready;

  logic [AXI_DATA_W-1:0] m_axi_r_data;
  logic [1:0]            m_axi_r_resp;
  logic                  m_axi_r_valid;
  logic                  m_axi_r_last;
  logic                  m_axi_r_ready;

  logic [AXI_ADDR_W-1:0] m_axi_aw_addr;
  logic [7:0]            m_axi_aw_len;
  logic [2:0]            m_axi_aw_size;
  logic [1:0]            m_axi_aw_burst;
  logic                  m_axi_aw_valid;
  logic                  m_axi_aw_ready;

  logic [AXI_DATA_W-1:0] m_axi_w_data;
  logic [15:0]           m_axi_w_strb;
  logic                  m_axi_w_valid;
  logic                  m_axi_w_last;
  logic                  m_axi_w_ready;

  logic [1:0]            m_axi_b_resp;
  logic                  m_axi_b_valid;
  logic                  m_axi_b_ready;

  // ----------------------------------------------------------------------
  // DDR controller stub (replace with vendor IP when a part is picked)
  // ----------------------------------------------------------------------
  ddr_axi_stub #(.DRAM_SIZE(DRAM_SIZE)) u_ddr (
    .clk           (clk),
    .rst_n         (rst_n),
    .m_axi_ar_addr (m_axi_ar_addr),
    .m_axi_ar_valid(m_axi_ar_valid),
    .m_axi_ar_len  (m_axi_ar_len),
    .m_axi_ar_size (m_axi_ar_size),
    .m_axi_ar_burst(m_axi_ar_burst),
    .m_axi_ar_ready(m_axi_ar_ready),
    .m_axi_r_data  (m_axi_r_data),
    .m_axi_r_resp  (m_axi_r_resp),
    .m_axi_r_valid (m_axi_r_valid),
    .m_axi_r_last  (m_axi_r_last),
    .m_axi_r_ready (m_axi_r_ready),
    .m_axi_aw_addr (m_axi_aw_addr),
    .m_axi_aw_len  (m_axi_aw_len),
    .m_axi_aw_size (m_axi_aw_size),
    .m_axi_aw_burst(m_axi_aw_burst),
    .m_axi_aw_valid(m_axi_aw_valid),
    .m_axi_aw_ready(m_axi_aw_ready),
    .m_axi_w_data  (m_axi_w_data),
    .m_axi_w_strb  (m_axi_w_strb),
    .m_axi_w_valid (m_axi_w_valid),
    .m_axi_w_last  (m_axi_w_last),
    .m_axi_w_ready (m_axi_w_ready),
    .m_axi_b_resp  (m_axi_b_resp),
    .m_axi_b_valid (m_axi_b_valid),
    .m_axi_b_ready (m_axi_b_ready)
  );

  // ----------------------------------------------------------------------
  // The verified compute core. Same module name + ports that the freeze
  // cosim gate, the Verilator regression matrix, and the Yosys synth-check
  // all consume — this wrapper is purely target glue.
  // ----------------------------------------------------------------------
  taccel_top #(
    .SYSTOLIC_ARCH_MODE(SYSTOLIC_ARCH_MODE),
    .DRAM_SIZE         (DRAM_SIZE),
    .SFU_SYNTH_MODE    (SFU_SYNTH_MODE),
    .HELPER_SYNTH_MODE (HELPER_SYNTH_MODE)
  ) u_core (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (start),
    .done          (done),
    .fault         (fault),
    .fault_code    (fault_code),
    .m_axi_ar_addr (m_axi_ar_addr),
    .m_axi_ar_valid(m_axi_ar_valid),
    .m_axi_ar_len  (m_axi_ar_len),
    .m_axi_ar_size (m_axi_ar_size),
    .m_axi_ar_burst(m_axi_ar_burst),
    .m_axi_ar_ready(m_axi_ar_ready),
    .m_axi_r_data  (m_axi_r_data),
    .m_axi_r_resp  (m_axi_r_resp),
    .m_axi_r_valid (m_axi_r_valid),
    .m_axi_r_last  (m_axi_r_last),
    .m_axi_r_ready (m_axi_r_ready),
    .m_axi_aw_addr (m_axi_aw_addr),
    .m_axi_aw_len  (m_axi_aw_len),
    .m_axi_aw_size (m_axi_aw_size),
    .m_axi_aw_burst(m_axi_aw_burst),
    .m_axi_aw_valid(m_axi_aw_valid),
    .m_axi_aw_ready(m_axi_aw_ready),
    .m_axi_w_data  (m_axi_w_data),
    .m_axi_w_strb  (m_axi_w_strb),
    .m_axi_w_valid (m_axi_w_valid),
    .m_axi_w_last  (m_axi_w_last),
    .m_axi_w_ready (m_axi_w_ready),
    .m_axi_b_resp  (m_axi_b_resp),
    .m_axi_b_valid (m_axi_b_valid),
    .m_axi_b_ready (m_axi_b_ready)
  );

endmodule

`endif // TACCEL_TOP_FPGA_SV
