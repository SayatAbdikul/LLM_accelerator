// Register file: scale registers, DRAM address registers, tile config.
//
// Scale registers (S0–S15): 16 × FP16
//   Written by SET_SCALE (imm or from buffer).
//   Read by all compute instructions that reference sreg.
//
// Address registers (R0–R3): 4 × 56-bit
//   SET_ADDR_LO writes bits [27:0] leaving [55:28] unchanged.
//   SET_ADDR_HI writes bits [55:28] leaving [27:0] unchanged.
//   Read by LOAD/STORE for effective DRAM address computation.
//
// Tile config: tile_m, tile_n, tile_k (10-bit each, 0-based encoded)
//   Written by CONFIG_TILE.
//   tile_valid asserted after first CONFIG_TILE; cleared only on reset.
//   Compute instructions must check tile_valid and fault if not set.
//
// This module is intentionally simple storage. Instruction legality and
// sequencing are enforced in control, not here.

`ifndef REGISTER_FILE_SV
`define REGISTER_FILE_SV

`include "taccel_pkg.sv"

module register_file
  import taccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // --- Scale register write ---
  input  logic        scale_we,
  input  logic [3:0]  scale_waddr,    // S0–S15
  input  logic [15:0] scale_wdata,    // FP16

  // --- Scale register read (up to 4 simultaneous, for SFU quad-scale) ---
  input  logic [3:0]  scale_raddr0,
  output logic [15:0] scale_rdata0,
  input  logic [3:0]  scale_raddr1,
  output logic [15:0] scale_rdata1,
  input  logic [3:0]  scale_raddr2,
  output logic [15:0] scale_rdata2,
  input  logic [3:0]  scale_raddr3,
  output logic [15:0] scale_rdata3,

  // --- DRAM address register write ---
  input  logic        addr_lo_we,     // SET_ADDR_LO: write bits [27:0]
  input  logic        addr_hi_we,     // SET_ADDR_HI: write bits [55:28]
  input  logic [1:0]  addr_wsel,      // R0–R3
  input  logic [27:0] addr_imm28,     // immediate value

  // --- DRAM address register read ---
  input  logic [1:0]  addr_rsel,
  output logic [55:0] addr_rdata,

  // --- Tile config write (CONFIG_TILE) ---
  input  logic        tile_we,
  input  logic [9:0]  tile_m_in,
  input  logic [9:0]  tile_n_in,
  input  logic [9:0]  tile_k_in,

  // --- Tile config read ---
  output logic [9:0]  tile_m,
  output logic [9:0]  tile_n,
  output logic [9:0]  tile_k,
  output logic        tile_valid      // 0 until first CONFIG_TILE
);

  // -------------------------------------------------------------------------
  // Scale registers: 16 × FP16 (16 bits each)
  // -------------------------------------------------------------------------
  logic [15:0] scale_regs [0:NUM_SCALE_REGS-1];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_SCALE_REGS; i++)
        scale_regs[i] <= 16'h0;
    end else if (scale_we) begin
      scale_regs[scale_waddr] <= scale_wdata;
    end
  end

  assign scale_rdata0 = scale_regs[scale_raddr0];
  assign scale_rdata1 = scale_regs[scale_raddr1];
  assign scale_rdata2 = scale_regs[scale_raddr2];
  assign scale_rdata3 = scale_regs[scale_raddr3];

  // -------------------------------------------------------------------------
  // DRAM address registers: 4 × 56-bit
  // -------------------------------------------------------------------------
  logic [55:0] addr_regs [0:NUM_ADDR_REGS-1];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_ADDR_REGS; i++)
        addr_regs[i] <= 56'h0;
    end else begin
      if (addr_lo_we)
        addr_regs[addr_wsel][27:0]  <= addr_imm28;
      if (addr_hi_we)
        addr_regs[addr_wsel][55:28] <= addr_imm28;
    end
  end

  assign addr_rdata = addr_regs[addr_rsel];

  // -------------------------------------------------------------------------
  // Tile configuration.
  // The fields store the encoded values directly; downstream logic applies the
  // architectural +1 when it needs an actual tile count.
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tile_m     <= 10'h0;
      tile_n     <= 10'h0;
      tile_k     <= 10'h0;
      tile_valid <= 1'b0;
    end else if (tile_we) begin
      tile_m     <= tile_m_in;
      tile_n     <= tile_n_in;
      tile_k     <= tile_k_in;
      tile_valid <= 1'b1;
    end
  end

endmodule

`endif // REGISTER_FILE_SV
