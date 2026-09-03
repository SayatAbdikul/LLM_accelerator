// Minimal AXI4 slave stub — accepts all requests, drives constant outputs.
//
// PURPOSE: lets `taccel_top_fpga` elaborate cleanly without a vendor DDR
// controller IP. NOT functional DRAM — does not actually store/return data.
// Replace with vendor DDR controller IP when the target FPGA is picked:
//   - Xilinx: MIG / DDR4 / HBM controller
//   - Intel: UniPHY / DDR4-EMIF
//   - Open source: LiteDRAM
//
// Elaboration-only placeholder. It accepts requests but never asserts RVALID
// or BVALID, so any real read or write transaction stalls permanently.

`ifndef DDR_AXI_STUB_SV
`define DDR_AXI_STUB_SV

`include "taccel_pkg.sv"

module ddr_axi_stub
  import taccel_pkg::*;
#(
  parameter int DRAM_SIZE = 1 << 24
)(
  input  logic                  clk,
  input  logic                  rst_n,

  // AR channel (taccel_top is master; stub is slave)
  input  logic [AXI_ADDR_W-1:0] m_axi_ar_addr,
  input  logic                  m_axi_ar_valid,
  input  logic [7:0]            m_axi_ar_len,
  input  logic [2:0]            m_axi_ar_size,
  input  logic [1:0]            m_axi_ar_burst,
  output logic                  m_axi_ar_ready,

  // R channel
  output logic [AXI_DATA_W-1:0] m_axi_r_data,
  output logic [1:0]            m_axi_r_resp,
  output logic                  m_axi_r_valid,
  output logic                  m_axi_r_last,
  input  logic                  m_axi_r_ready,

  // AW channel
  input  logic [AXI_ADDR_W-1:0] m_axi_aw_addr,
  input  logic [7:0]            m_axi_aw_len,
  input  logic [2:0]            m_axi_aw_size,
  input  logic [1:0]            m_axi_aw_burst,
  input  logic                  m_axi_aw_valid,
  output logic                  m_axi_aw_ready,

  // W channel
  input  logic [AXI_DATA_W-1:0] m_axi_w_data,
  input  logic [15:0]           m_axi_w_strb,
  input  logic                  m_axi_w_valid,
  input  logic                  m_axi_w_last,
  output logic                  m_axi_w_ready,

  // B channel
  output logic [1:0]            m_axi_b_resp,
  output logic                  m_axi_b_valid,
  input  logic                  m_axi_b_ready
);

  // Always-ready handshake; never produces valid response.
  // This deadlocks any actual transaction, but is sufficient for
  // structural elaboration.
  assign m_axi_ar_ready = 1'b1;
  assign m_axi_aw_ready = 1'b1;
  assign m_axi_w_ready  = 1'b1;

  assign m_axi_r_data  = '0;
  assign m_axi_r_resp  = 2'b00;
  assign m_axi_r_valid = 1'b0;
  assign m_axi_r_last  = 1'b0;

  assign m_axi_b_resp  = 2'b00;
  assign m_axi_b_valid = 1'b0;

  // Suppress unused-signal warnings for elaboration-only stub.
  logic _unused;
  assign _unused = &{1'b0, clk, rst_n,
                     m_axi_ar_addr, m_axi_ar_valid, m_axi_ar_len,
                     m_axi_ar_size, m_axi_ar_burst,
                     m_axi_r_ready,
                     m_axi_aw_addr, m_axi_aw_len, m_axi_aw_size,
                     m_axi_aw_burst, m_axi_aw_valid,
                     m_axi_w_data, m_axi_w_strb, m_axi_w_valid, m_axi_w_last,
                     m_axi_b_ready};
endmodule

`endif // DDR_AXI_STUB_SV
