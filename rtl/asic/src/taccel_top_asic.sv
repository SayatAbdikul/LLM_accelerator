// ASIC top wrapper around the verified `taccel_top` core.
//
// Research integration wrapper exposing clock/reset/control pins and the AXI
// master through placeholder pad logic. This is sufficient for structural
// elaboration, but it is not a tape-out-ready top: the pad ring, SRAM macros,
// physical configuration, and sign-off flow remain unimplemented.
//
// When integrated into eFabless Caravel, this wrapper's external AXI is
// re-routed through Caravel's Wishbone bridge instead of off-chip pads,
// and the pad ring stub is replaced by Caravel's `user_project_wrapper`
// interface. The compute core itself does not change.

`ifndef TACCEL_TOP_ASIC_SV
`define TACCEL_TOP_ASIC_SV

// Belt-and-suspenders guards. ASIC synthesis cannot consume DPI, and
// the sram_dp dispatch wrapper only binds to sram_dp_macro under
// TARGET_ASIC.
`ifndef SFU_SYNTH_NO_DPI
  `error "TARGET_ASIC requires SFU_SYNTH_NO_DPI; ASIC synthesis cannot consume DPI imports."
`endif

`ifndef TARGET_ASIC
  `error "taccel_top_asic requires -DTARGET_ASIC (selects sram_dp_macro binding in the common SRAM dispatch wrapper)."
`endif

`include "taccel_pkg.sv"

module taccel_top_asic
  import taccel_pkg::*;
#(
  parameter int SYSTOLIC_ARCH_MODE = SYS_MODE_DEFAULT,
  parameter int DRAM_SIZE          = 1 << 24,
  // ASIC target always uses the synth-friendly compute paths.
  parameter int SFU_SYNTH_MODE     = 1,
  parameter int HELPER_SYNTH_MODE  = 1
)(
  // Off-chip pads — placeholder pin list, real pin order set by
  // a physical-flow pin-order file once a floorplan is defined.
  input  logic        clk_pad,
  input  logic        rst_n_pad,
  input  logic        start,
  output logic        done,
  output logic        fault,
  output logic [3:0]  fault_code,

  // External AXI4 master to off-chip memory (e.g. HyperRAM via external
  // controller chip, or DDR via a companion FPGA). When integrated into
  // Caravel, these become Wishbone bridge ports inside the harness.
  output logic [AXI_ADDR_W-1:0] m_axi_ar_addr,
  output logic                  m_axi_ar_valid,
  output logic [7:0]            m_axi_ar_len,
  output logic [2:0]            m_axi_ar_size,
  output logic [1:0]            m_axi_ar_burst,
  input  logic                  m_axi_ar_ready,

  input  logic [AXI_DATA_W-1:0] m_axi_r_data,
  input  logic [1:0]            m_axi_r_resp,
  input  logic                  m_axi_r_valid,
  input  logic                  m_axi_r_last,
  output logic                  m_axi_r_ready,

  output logic [AXI_ADDR_W-1:0] m_axi_aw_addr,
  output logic [7:0]            m_axi_aw_len,
  output logic [2:0]            m_axi_aw_size,
  output logic [1:0]            m_axi_aw_burst,
  output logic                  m_axi_aw_valid,
  input  logic                  m_axi_aw_ready,

  output logic [AXI_DATA_W-1:0] m_axi_w_data,
  output logic [15:0]           m_axi_w_strb,
  output logic                  m_axi_w_valid,
  output logic                  m_axi_w_last,
  input  logic                  m_axi_w_ready,

  input  logic [1:0]            m_axi_b_resp,
  input  logic                  m_axi_b_valid,
  output logic                  m_axi_b_ready
);

  // ----------------------------------------------------------------------
  // Pad ring → core clock/reset
  // ----------------------------------------------------------------------
  logic clk, rst_n;
  pad_ring_stub u_pads (
    .clk_pad   (clk_pad),
    .rst_n_pad (rst_n_pad),
    .clk_core  (clk),
    .rst_n_core(rst_n)
  );

  // ----------------------------------------------------------------------
  // The verified compute core. Same module name + ports that the freeze
  // cosim, the Verilator regression matrix, and the Yosys synth-check
  // all consume — this wrapper is purely target glue.
  //
  // Under TARGET_ASIC, the core's `sram_dp` dispatch wrapper resolves
  // to `sram_dp_macro` (declared in rtl/asic/src/sram_dp_sky130.sv).
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

`endif // TACCEL_TOP_ASIC_SV
