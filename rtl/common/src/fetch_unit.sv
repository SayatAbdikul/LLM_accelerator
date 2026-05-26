// Instruction fetch unit.
//
// Reads 8-byte instructions from DRAM via an AXI4 read-only master port
// (128-bit data bus, 16-byte aligned transfers).
//
// DRAM layout: instructions are packed at byte address PC × 8, big-endian.
// Since the AXI bus transfers 16 bytes per beat, each beat contains two
// consecutive instructions:
//   PC even: instruction is in rdata[63:0]   after byte-swap
//   PC odd:  instruction is in rdata[127:64] after byte-swap
//
// Byte swap: AXI transfers byte 0 in rdata[7:0] (little-endian ordering),
// but TACCEL instructions are big-endian (opcode in MSByte).  The fetch
// unit byte-swaps the relevant 8 bytes before presenting insn_data.
//
// Timing (non-pipelined, 1 request in flight at a time):
//   Cycle 0: ar_valid asserted, aligned address driven
//   Cycle 1: ar_ready assumed; address accepted
//   Cycle N: r_valid from slave; instruction captured, insn_valid pulsed
//
// The fetch unit is intentionally tiny: one request in flight, one returned
// instruction beat, then back to idle. Faults are sticky until reset so the
// control unit cannot miss them.

`ifndef FETCH_UNIT_SV
`define FETCH_UNIT_SV

`include "taccel_pkg.sv"

module fetch_unit
  import taccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // --- From control unit ---
  input  logic [55:0] pc,             // current program counter (instruction index)
  input  logic        fetch_req,      // pulse: request fetch of insn at PC
  output logic        insn_valid,     // pulse: insn_data holds valid instruction
  output logic [63:0] insn_data,      // fetched instruction (big-endian word)
  output logic        fetch_fault,    // sticky fault until reset
  output logic [3:0]  fetch_fault_code,

  // --- AXI4 read channels (AR + R) ---
  // Address channel
  output logic [AXI_ADDR_W-1:0] m_axi_ar_addr,
  output logic                   m_axi_ar_valid,
  output logic [7:0]             m_axi_ar_len,    // 0 = single beat
  output logic [2:0]             m_axi_ar_size,   // 3'b100 = 16 bytes/beat
  output logic [1:0]             m_axi_ar_burst,  // 2'b01 = INCR
  input  logic                   m_axi_ar_ready,

  // Read data channel
  input  logic [AXI_DATA_W-1:0] m_axi_r_data,
  input  logic [1:0]             m_axi_r_resp,
  input  logic                   m_axi_r_valid,
  input  logic                   m_axi_r_last,
  output logic                   m_axi_r_ready
);

  // -------------------------------------------------------------------------
  // AXI burst parameters (fixed for instruction fetch)
  // -------------------------------------------------------------------------
  assign m_axi_ar_len   = 8'h00;    // single beat
  assign m_axi_ar_size  = 3'b100;   // 16 bytes per beat (2^4)
  assign m_axi_ar_burst = 2'b01;    // INCR (required even for single beat)

  // -------------------------------------------------------------------------
  // Address computation
  // PC is instruction index.  Byte address = PC × 8.
  // Aligned 16-byte address = (PC × 8) & ~15 = (PC >> 1) × 16
  // -------------------------------------------------------------------------
  logic [55:0] byte_addr;
  logic [55:0] aligned_addr;
  logic        pc_odd;              // selects which half of the 16-byte beat

  assign byte_addr    = pc << 3;               // PC * 8
  assign aligned_addr = {byte_addr[55:4], 4'b0};  // clear lower 4 bits
  assign pc_odd       = pc[0];                 // odd PC → upper 8 bytes

  // -------------------------------------------------------------------------
  // Byte-swap helper: AXI LE byte ordering → BE instruction word
  //   rdata[7:0] holds the lowest DRAM byte (instruction MSByte for even PC)
  // -------------------------------------------------------------------------
  function automatic [63:0] bswap64(input [63:0] x);
    return {x[7:0], x[15:8], x[23:16], x[31:24],
            x[39:32], x[47:40], x[55:48], x[63:56]};
  endfunction

  // -------------------------------------------------------------------------
  // FSM: IDLE → AR_REQ → WAIT_R / FAULT
  // -------------------------------------------------------------------------
  typedef enum logic [1:0] {
    S_IDLE   = 2'd0,
    S_AR_REQ = 2'd1,   // driving AR channel, waiting for ar_ready
    S_WAIT_R = 2'd2,   // ar accepted, waiting for R data
    S_FAULT  = 2'd3
  } fetch_state_t;

  fetch_state_t state, next_state;
  logic [3:0]   fault_code_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      fault_code_r <= 4'(FAULT_NONE);
    end else begin
      state <= next_state;
      if (state == S_WAIT_R && m_axi_r_valid &&
          ((m_axi_r_resp != 2'b00) || !m_axi_r_last))
        fault_code_r <= 4'(FAULT_DRAM_OOB);
    end
  end

  // Latch the lane select at AR acceptance time because PC may already have
  // advanced by the time the R beat comes back.
  logic pc_odd_q;
  always_ff @(posedge clk) begin
    if (state == S_AR_REQ && m_axi_ar_ready && m_axi_ar_valid)
      pc_odd_q <= pc_odd;
  end

  always_comb begin
    next_state        = state;
    m_axi_ar_valid    = 1'b0;
    m_axi_ar_addr     = aligned_addr;
    m_axi_r_ready     = 1'b0;
    insn_valid        = 1'b0;
    insn_data         = 64'h0;
    fetch_fault       = 1'b0;
    fetch_fault_code  = fault_code_r;

    case (state)
      S_IDLE: begin
        if (fetch_req)
          next_state = S_AR_REQ;
      end

      S_AR_REQ: begin
        m_axi_ar_valid = 1'b1;
        m_axi_ar_addr  = aligned_addr;
        if (m_axi_ar_ready)
          next_state = S_WAIT_R;
      end

      S_WAIT_R: begin
        m_axi_r_ready = 1'b1;
        if (m_axi_r_valid) begin
          if ((m_axi_r_resp != 2'b00) || !m_axi_r_last) begin
            next_state = S_FAULT;
          end else begin
            // Extract the relevant 8 bytes and byte-swap to get BE word
            if (pc_odd_q)
              insn_data = bswap64(m_axi_r_data[127:64]);
            else
              insn_data = bswap64(m_axi_r_data[63:0]);
            insn_valid = 1'b1;
            next_state = S_IDLE;
          end
        end
      end

      S_FAULT: begin
        fetch_fault = 1'b1;
      end

      default: next_state = S_IDLE;
    endcase
  end

endmodule

`endif // FETCH_UNIT_SV
