// AXI4 slave behavioral model for simulation testbenches.
//
// Models a DRAM slave with parameterizable latency.
// Supports read-only in Phase 1 (instruction fetch).
// Write channel responses are stubbed.
//
// Byte ordering convention (matches TACCEL spec):
//   - Data bytes are stored at increasing DRAM byte addresses
//   - rdata[7:0] = DRAM byte at the transfer's base address (byte 0)
//   - rdata[15:8] = DRAM byte at base+1, etc.
//   - Instructions are stored big-endian at PC*8:
//       DRAM[PC*8+0] = instruction MSByte (bits [63:56])
//       DRAM[PC*8+7] = instruction LSByte (bits [7:0])
//
// Latency: configurable via READ_LATENCY parameter (in cycles after AR handshake).

`ifndef AXI4_SLAVE_MODEL_SV
`define AXI4_SLAVE_MODEL_SV

`include "taccel_pkg.sv"

module axi4_slave_model
  import taccel_pkg::*;
#(
  parameter int    DRAM_SIZE    = 16 * 1024 * 1024,  // 16 MB
  parameter int    READ_LATENCY = 2                   // cycles from AR accept to R valid
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- AXI4 read channels ---
  input  logic [AXI_ADDR_W-1:0] s_axi_ar_addr,
  input  logic                   s_axi_ar_valid,
  output logic                   s_axi_ar_ready,

  output logic [AXI_DATA_W-1:0] s_axi_r_data,
  output logic [1:0]             s_axi_r_resp,
  output logic                   s_axi_r_valid,
  output logic                   s_axi_r_last,
  input  logic                   s_axi_r_ready,

  // --- AXI4 write channels (stub) ---
  input  logic [AXI_ADDR_W-1:0] s_axi_aw_addr,
  input  logic                   s_axi_aw_valid,
  output logic                   s_axi_aw_ready,
  input  logic [AXI_DATA_W-1:0] s_axi_w_data,
  input  logic [15:0]            s_axi_w_strb,
  input  logic                   s_axi_w_valid,
  input  logic                   s_axi_w_last,
  output logic                   s_axi_w_ready,
  output logic [1:0]             s_axi_b_resp,
  output logic                   s_axi_b_valid,
  input  logic                   s_axi_b_ready
);

  // -------------------------------------------------------------------------
  // DRAM array (byte-addressable)
  // -------------------------------------------------------------------------
  logic [7:0] mem [0:DRAM_SIZE-1];

  // -------------------------------------------------------------------------
  // Read pipeline: latency FIFO
  // -------------------------------------------------------------------------
  typedef struct packed {
    logic [AXI_ADDR_W-1:0] addr;
    logic                   valid;
    int                     timer;
  } read_req_t;

  read_req_t pending;

  // AR channel: always ready (accept immediately)
  assign s_axi_ar_ready = 1'b1;

  // -------------------------------------------------------------------------
  // Read response logic
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pending.valid     <= 1'b0;
      pending.timer     <= 0;
      s_axi_r_valid     <= 1'b0;
      s_axi_r_data      <= '0;
      s_axi_r_resp      <= 2'b00;
      s_axi_r_last      <= 1'b0;
    end else begin

      // Accept new AR transaction
      if (s_axi_ar_valid && s_axi_ar_ready && !pending.valid) begin
        pending.valid <= 1'b1;
        pending.addr  <= s_axi_ar_addr & ~{56'h0, 4'hF};  // align to 16 bytes
        pending.timer <= READ_LATENCY;
      end

      // Count down latency, then present data
      if (pending.valid) begin
        if (pending.timer > 0) begin
          pending.timer <= pending.timer - 1;
        end else if (!s_axi_r_valid || s_axi_r_ready) begin
          // Build the 128-bit (16-byte) read response
          for (int b = 0; b < 16; b++) begin
            automatic logic [AXI_ADDR_W-1:0] baddr = pending.addr + b;
            if (baddr < AXI_ADDR_W'(DRAM_SIZE))
              s_axi_r_data[b*8 +: 8] <= mem[baddr];
            else
              s_axi_r_data[b*8 +: 8] <= 8'h0;
          end
          s_axi_r_valid   <= 1'b1;
          s_axi_r_resp    <= 2'b00;  // OKAY
          s_axi_r_last    <= 1'b1;   // single-beat burst
          pending.valid   <= 1'b0;
        end
      end

      // Clear valid when master accepts
      if (s_axi_r_valid && s_axi_r_ready) begin
        s_axi_r_valid <= 1'b0;
        s_axi_r_last  <= 1'b0;
      end

    end
  end

  // -------------------------------------------------------------------------
  // Write channel: accept writes (Phase 2 DMA STORE support)
  // -------------------------------------------------------------------------
  assign s_axi_aw_ready = 1'b1;
  assign s_axi_w_ready  = 1'b1;
  assign s_axi_b_resp   = 2'b00;
  assign s_axi_b_valid  = 1'b0;  // simplified: no write response in Phase 1

  always_ff @(posedge clk) begin
    if (s_axi_w_valid && s_axi_w_ready) begin
      // Phase 2+ will implement proper address-tracked writes
      // For now just silently discard
    end
  end

  // -------------------------------------------------------------------------
  // Task: load a byte array into DRAM at a given byte offset (for testbenches)
  // -------------------------------------------------------------------------
  task automatic load_bytes(
    input logic [AXI_ADDR_W-1:0] base_addr,
    input logic [7:0]             data[],
    input int                     len
  );
    for (int i = 0; i < len; i++) begin
      if (base_addr + i < DRAM_SIZE)
        mem[base_addr + i] = data[i];
    end
  endtask

  // Write a 64-bit big-endian instruction at instruction index pc_idx
  task automatic write_insn(input int pc_idx, input logic [63:0] word);
    automatic logic [55:0] base = 56'(pc_idx) * 56'h8;
    mem[base+0] = word[63:56];
    mem[base+1] = word[55:48];
    mem[base+2] = word[47:40];
    mem[base+3] = word[39:32];
    mem[base+4] = word[31:24];
    mem[base+5] = word[23:16];
    mem[base+6] = word[15:8];
    mem[base+7] = word[7:0];
  endtask

endmodule

`endif // AXI4_SLAVE_MODEL_SV
