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
// One request may be in flight. Optional one-beat buffering retains both
// instruction halves. `invalidate` bypasses the buffer and poisons any pending
// refill; the caller holds it during DMA stores and pulses it on a new run.
// Faults are sticky until reset so the control unit cannot miss them.

`ifndef FETCH_UNIT_SV
`define FETCH_UNIT_SV

`include "taccel_pkg.sv"

module fetch_unit
  import taccel_pkg::*;
#(
  parameter bit FETCH_BUFFER_ENABLE = 1'b0
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- From control unit ---
  input  logic [55:0] pc,             // current program counter (instruction index)
  input  logic        fetch_req,      // pulse: request fetch of insn at PC
  input  logic        invalidate,     // bypass and invalidate retained beat
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
  logic [55:0] request_addr_q;
  logic        pc_odd_q;
  logic [55:0] buffer_addr_q;
  logic [127:0] buffer_data_q;
  logic        buffer_valid_q;
  logic        refill_allowed_q;
  logic        buffer_hit_w;

  assign buffer_hit_w = FETCH_BUFFER_ENABLE && buffer_valid_q && !invalidate &&
                        (buffer_addr_q == aligned_addr);
  // VALID/payload depend only on the registered request, not READY. Keep
  // these outside the next-state block so arbitration has no combinational
  // dependency through its own ready feedback.
  assign m_axi_ar_valid = (state == S_AR_REQ);
  assign m_axi_ar_addr = request_addr_q;
  assign m_axi_r_ready = (state == S_WAIT_R);
  assign fetch_fault = (state == S_FAULT);
  assign fetch_fault_code = fault_code_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      fault_code_r <= 4'(FAULT_NONE);
      request_addr_q <= 56'h0;
      pc_odd_q <= 1'b0;
      buffer_addr_q <= 56'h0;
      buffer_data_q <= 128'h0;
      buffer_valid_q <= 1'b0;
      refill_allowed_q <= 1'b0;
    end else begin
      state <= next_state;
      if (state == S_IDLE && fetch_req && !buffer_hit_w) begin
        // Hold the requested address and half across AR backpressure and R
        // latency. A changing upstream PC must not change an outstanding AR.
        request_addr_q <= aligned_addr;
        pc_odd_q <= pc_odd;
        refill_allowed_q <= !invalidate;
      end
      if (FETCH_BUFFER_ENABLE && state == S_WAIT_R && m_axi_r_valid &&
          m_axi_r_resp == 2'b00 && m_axi_r_last && refill_allowed_q) begin
        buffer_addr_q <= request_addr_q;
        buffer_data_q <= m_axi_r_data;
        buffer_valid_q <= 1'b1;
      end
      // Invalidation wins over a same-cycle refill, and remains effective
      // when it occurs earlier than a delayed response. The response still
      // completes the original read, exactly as in the unbuffered machine.
      if (invalidate) begin
        buffer_valid_q <= 1'b0;
        refill_allowed_q <= 1'b0;
      end
      if (state == S_WAIT_R && m_axi_r_valid &&
          ((m_axi_r_resp != 2'b00) || !m_axi_r_last))
        fault_code_r <= 4'(FAULT_DRAM_OOB);
    end
  end

  always_comb begin
    next_state        = state;
    insn_valid        = 1'b0;
    insn_data         = 64'h0;

    case (state)
      S_IDLE: begin
        if (fetch_req) begin
          if (buffer_hit_w) begin
            insn_data = pc_odd ? bswap64(buffer_data_q[127:64])
                               : bswap64(buffer_data_q[63:0]);
            insn_valid = 1'b1;
          end else begin
            next_state = S_AR_REQ;
          end
        end
      end

      S_AR_REQ: begin
        if (m_axi_ar_ready)
          next_state = S_WAIT_R;
      end

      S_WAIT_R: begin
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

      S_FAULT: ;

      default: next_state = S_IDLE;
    endcase
  end

endmodule

`endif // FETCH_UNIT_SV
