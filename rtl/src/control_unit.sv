// Control unit -- in-order issue stage FSM.
//
// Orchestrates instruction fetch, decode (via external decode_unit), and
// execution.  All issue-stage instructions execute here; long-latency
// instructions are dispatched to execution units via one-cycle dispatch pulses.
//
// FSM states:
//   IDLE        — awaiting start pulse
//   FETCH       — fetch_req asserted, waiting for insn_valid pulse
//   ISSUE       — decode + execute one instruction (1 cycle)
//   SYNC_WAIT   — barrier: stall until selected units drain
//   DISP_WAIT   -- waiting for a blocking helper op to complete
//   HALT        — terminal: done=1
//   FAULT       — terminal: fault=1
//
// Stage D: DMA LOAD/STORE, systolic MATMUL, blocking helper ops, and the first
// asynchronous SFU ops (SOFTMAX / LAYERNORM / GELU) are supported.
// Remaining legal-but-unimplemented helper/SFU instructions fault as
// FAULT_UNSUPPORTED_OP.

`ifndef CONTROL_UNIT_SV
`define CONTROL_UNIT_SV

`include "taccel_pkg.sv"

module control_unit
  import taccel_pkg::*;
(
  input  logic          clk,
  input  logic          rst_n,
  input  logic          start,

  // --- Fetch interface ---
  output logic [55:0]   pc,
  output logic          fetch_req,   // combinational, held in S_FETCH
  input  logic          insn_valid,  // 1-cycle pulse when instruction ready

  // --- Decoded instruction (stable after insn_valid; registered in top) ---
  input  decoded_insn_t insn,

  // --- Register file write ports ---
  output logic          scale_we,
  output logic [3:0]    scale_waddr,
  output logic [15:0]   scale_wdata,

  output logic          addr_lo_we,
  output logic          addr_hi_we,
  output logic [1:0]    addr_wsel,
  output logic [27:0]   addr_imm28,

  output logic          tile_we,
  output logic [9:0]    tile_m_in,
  output logic [9:0]    tile_n_in,
  output logic [9:0]    tile_k_in,

  // --- Register file read-back ---
  input  logic          tile_valid,

  // --- Dispatch to execution units (1-cycle pulses) ---
  output logic          dma_dispatch,
  output logic          sys_dispatch,
  output logic          sfu_dispatch,
  output logic          helper_dispatch,

  // --- Execution unit status ---
  input  logic          dma_busy,
  input  logic          sys_busy,
  input  logic          sfu_busy,
  input  logic          helper_busy,

  // --- External fault path from fetch / DMA / top-level memory plumbing ---
  input  logic          ext_fault,
  input  logic [3:0]    ext_fault_code,

  // --- Status outputs ---
  output logic          done,
  output logic          fault,
  output logic [3:0]    fault_code,

  // --- Internal-only observability hooks for the Verilator program runner ---
  output logic          obs_retire_pulse,
  output logic [55:0]   obs_retire_pc,
  output logic [4:0]    obs_retire_opcode,
  output logic          obs_ctrl_fault_pulse,
  output logic [3:0]    obs_ctrl_fault_code,
  output logic [55:0]   obs_ctrl_fault_pc,
  output logic [4:0]    obs_ctrl_fault_opcode,
  output logic          obs_sync_wait_dma,
  output logic          obs_sync_wait_sys,
  output logic          obs_sync_wait_sfu
);

  // -------------------------------------------------------------------------
  // FSM state encoding
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_IDLE      = 3'd0,
    S_FETCH     = 3'd1,
    S_ISSUE     = 3'd2,
    S_SYNC_WAIT = 3'd3,
    S_DISP_WAIT = 3'd4,
    S_HALT      = 3'd5,
    S_FAULT     = 3'd6
  } ctrl_state_t;

  ctrl_state_t  state;
  logic [55:0]  pc_reg;
  logic [2:0]   sync_mask_q;  // latched resource_mask for SYNC_WAIT
  logic [3:0]   fault_code_r; // latched fault code

  assign pc         = pc_reg;
  assign fault_code = fault_code_r;

  // -------------------------------------------------------------------------
  // Pre-computed combinational helpers
  // -------------------------------------------------------------------------
  // Checks whether the units specified in mask are all idle. SYNC only tracks
  // architecturally asynchronous resources; Phase C helpers stay blocking and
  // therefore do not participate in the mask.
  logic sync_clear_q;   // for SYNC_WAIT: uses latched mask
  logic sync_clear_now; // for ISSUE: uses current insn mask

  assign sync_clear_q   = ~|(sync_mask_q       & {sfu_busy, sys_busy, dma_busy});
  assign sync_clear_now = ~|(insn.sync_mask & {sfu_busy, sys_busy, dma_busy});

  // -------------------------------------------------------------------------
  // Helper: legal ISA operations and parameterizations that are not yet
  // implemented in the current RTL.
  // -------------------------------------------------------------------------
  function automatic logic unsupported_op(
    input logic [4:0]  op,
    input logic [1:0]  s_src_mode
  );
    begin
      case (op)
        OP_SET_SCALE:
          unsupported_op = (s_src_mode != 2'b00);

        default:
          unsupported_op = 1'b0;
      endcase
    end
  endfunction

  logic unsupported_now;
  assign unsupported_now = unsupported_op(insn.opcode, insn.s_src_mode);

  // -------------------------------------------------------------------------
  // Sequential FSM + registers
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      pc_reg       <= 56'h0;
      sync_mask_q  <= 3'h0;
      fault_code_r <= 4'h0;
      obs_retire_pulse     <= 1'b0;
      obs_retire_pc        <= 56'h0;
      obs_retire_opcode    <= 5'h0;
      obs_ctrl_fault_pulse <= 1'b0;
      obs_ctrl_fault_code  <= 4'h0;
      obs_ctrl_fault_pc    <= 56'h0;
      obs_ctrl_fault_opcode <= 5'h0;

    end else begin
      obs_retire_pulse     <= 1'b0;
      obs_ctrl_fault_pulse <= 1'b0;

      case (state)

        // ------------------------------------------------------------------
        S_IDLE:
          if (start) begin
            pc_reg <= 56'h0;
            state  <= S_FETCH;
          end

        // ------------------------------------------------------------------
        // Stay in FETCH until insn_valid pulse.  The instruction is already
        // registered into insn_data_q in taccel_top on this same posedge, so
        // insn is valid on the NEXT cycle (S_ISSUE).
        S_FETCH:
          if (ext_fault) begin
            fault_code_r <= ext_fault_code;
            state        <= S_FAULT;
          end else if (insn_valid)
            state <= S_ISSUE;

        // ------------------------------------------------------------------
        S_ISSUE: begin
          if (ext_fault) begin
            fault_code_r <= ext_fault_code;
            state        <= S_FAULT;
          end else if (insn.illegal) begin
            fault_code_r <= (insn.opcode > 5'h13) ?
                             4'(FAULT_ILLEGAL_OP) : 4'(FAULT_BAD_BUF);
            obs_ctrl_fault_pulse  <= 1'b1;
            obs_ctrl_fault_code   <= (insn.opcode > 5'h13) ?
                                     4'(FAULT_ILLEGAL_OP) : 4'(FAULT_BAD_BUF);
            obs_ctrl_fault_pc     <= pc_reg;
            obs_ctrl_fault_opcode <= insn.opcode;
            state        <= S_FAULT;
          end else if (unsupported_now) begin
            fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
            obs_ctrl_fault_pulse  <= 1'b1;
            obs_ctrl_fault_code   <= 4'(FAULT_UNSUPPORTED_OP);
            obs_ctrl_fault_pc     <= pc_reg;
            obs_ctrl_fault_opcode <= insn.opcode;
            state        <= S_FAULT;
          end else begin
            case (insn.opcode)
              OP_NOP: begin
                obs_retire_pulse  <= 1'b1;
                obs_retire_pc     <= pc_reg;
                obs_retire_opcode <= insn.opcode;
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_HALT: begin
                obs_retire_pulse  <= 1'b1;
                obs_retire_pc     <= pc_reg;
                obs_retire_opcode <= insn.opcode;
                state <= S_HALT;
              end

              OP_SYNC: begin
                if (sync_clear_now) begin
                  obs_retire_pulse  <= 1'b1;
                  obs_retire_pc     <= pc_reg;
                  obs_retire_opcode <= insn.opcode;
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end else begin
                  sync_mask_q <= insn.sync_mask;
                  state       <= S_SYNC_WAIT;
                end
              end

              OP_CONFIG_TILE: begin
                obs_retire_pulse  <= 1'b1;
                obs_retire_pc     <= pc_reg;
                obs_retire_opcode <= insn.opcode;
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_SET_SCALE: begin
                obs_retire_pulse  <= 1'b1;
                obs_retire_pc     <= pc_reg;
                obs_retire_opcode <= insn.opcode;
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_SET_ADDR_LO: begin
                obs_retire_pulse  <= 1'b1;
                obs_retire_pc     <= pc_reg;
                obs_retire_opcode <= insn.opcode;
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              OP_SET_ADDR_HI: begin
                obs_retire_pulse  <= 1'b1;
                obs_retire_pc     <= pc_reg;
                obs_retire_opcode <= insn.opcode;
                pc_reg <= pc_reg + 56'h1;
                state  <= S_FETCH;
              end

              // Blocking helper: do not advance PC until the helper drops busy.
              OP_BUF_COPY: begin
                if (dma_busy || sys_busy || helper_busy || sfu_busy) begin
                  state <= S_ISSUE;
                end else begin
                  state <= S_DISP_WAIT;
                end
              end

              // DMA is architecturally asynchronous. Once the dispatch pulse is
              // emitted, fetch/issue may continue and SYNC(001) becomes the
              // ordering point.
              OP_LOAD, OP_STORE: begin
                if (sfu_busy) begin
                  state <= S_ISSUE;
                end else begin
                  obs_retire_pulse  <= 1'b1;
                  obs_retire_pc     <= pc_reg;
                  obs_retire_opcode <= insn.opcode;
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              OP_REQUANT,
              OP_REQUANT_PC,
              OP_SCALE_MUL,
              OP_VADD,
              OP_DEQUANT_ADD: begin
                if (!tile_valid) begin
                  fault_code_r <= 4'(FAULT_NO_CONFIG);
                  obs_ctrl_fault_pulse  <= 1'b1;
                  obs_ctrl_fault_code   <= 4'(FAULT_NO_CONFIG);
                  obs_ctrl_fault_pc     <= pc_reg;
                  obs_ctrl_fault_opcode <= insn.opcode;
                  state        <= S_FAULT;
                end else if (dma_busy || sys_busy || helper_busy || sfu_busy) begin
                  state <= S_ISSUE;
                end else begin
                  state <= S_DISP_WAIT;
                end
              end

              // Systolic matmul is also asynchronous from the control FSM.
              // DST_CLEAR uses SRAM port A; the helper engine holds higher SRAM
              // priority, so MATMUL must not be dispatched while helper is busy
              // or DST_CLEAR writes would be silently dropped.
              OP_MATMUL: begin
                if (!tile_valid) begin
                  fault_code_r <= 4'(FAULT_NO_CONFIG);
                  obs_ctrl_fault_pulse  <= 1'b1;
                  obs_ctrl_fault_code   <= 4'(FAULT_NO_CONFIG);
                  obs_ctrl_fault_pc     <= pc_reg;
                  obs_ctrl_fault_opcode <= insn.opcode;
                  state        <= S_FAULT;
                end else if (sfu_busy || helper_busy) begin
                  state <= S_ISSUE;
                end else begin
                  obs_retire_pulse  <= 1'b1;
                  obs_retire_pc     <= pc_reg;
                  obs_retire_opcode <= insn.opcode;
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              // Stage D SFU ops are architecturally asynchronous. Dispatch once
              // the serialized Stage D resource slot is clear, then advance PC
              // immediately and rely on SYNC(100) for ordering.
              OP_SOFTMAX,
              OP_SOFTMAX_ATTNV,
              OP_LAYERNORM,
              OP_GELU: begin
                if (!tile_valid) begin
                  fault_code_r <= 4'(FAULT_NO_CONFIG);
                  obs_ctrl_fault_pulse  <= 1'b1;
                  obs_ctrl_fault_code   <= 4'(FAULT_NO_CONFIG);
                  obs_ctrl_fault_pc     <= pc_reg;
                  obs_ctrl_fault_opcode <= insn.opcode;
                  state        <= S_FAULT;
                end else if (dma_busy || sys_busy || helper_busy || sfu_busy) begin
                  state <= S_ISSUE;
                end else begin
                  obs_retire_pulse  <= 1'b1;
                  obs_retire_pc     <= pc_reg;
                  obs_retire_opcode <= insn.opcode;
                  pc_reg <= pc_reg + 56'h1;
                  state  <= S_FETCH;
                end
              end

              default: begin
                fault_code_r <= 4'(FAULT_ILLEGAL_OP);
                obs_ctrl_fault_pulse  <= 1'b1;
                obs_ctrl_fault_code   <= 4'(FAULT_ILLEGAL_OP);
                obs_ctrl_fault_pc     <= pc_reg;
                obs_ctrl_fault_opcode <= insn.opcode;
                state        <= S_FAULT;
              end
            endcase
          end
        end

        // ------------------------------------------------------------------
        S_SYNC_WAIT: begin
          // A helper fault is delivered through ext_fault while the helper is
          // still busy. A clean completion is observed when helper_busy drops.
          if (ext_fault) begin
            fault_code_r <= ext_fault_code;
            state        <= S_FAULT;
          end else if (sync_clear_q) begin
            obs_retire_pulse  <= 1'b1;
            obs_retire_pc     <= pc_reg;
            obs_retire_opcode <= 5'(OP_SYNC);
            pc_reg <= pc_reg + 56'h1;
            state  <= S_FETCH;
          end
        end

        // ------------------------------------------------------------------
        S_DISP_WAIT: begin
          if (ext_fault) begin
            fault_code_r <= ext_fault_code;
            state        <= S_FAULT;
          end else if (!helper_busy) begin
            obs_retire_pulse  <= 1'b1;
            obs_retire_pc     <= pc_reg;
            obs_retire_opcode <= insn.opcode;
            pc_reg <= pc_reg + 56'h1;
            state  <= S_FETCH;
          end
        end

        // ------------------------------------------------------------------
        S_HALT, S_FAULT: ;  // terminal — held by done/fault outputs

        default: state <= S_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Combinational output logic
  // -------------------------------------------------------------------------
  always_comb begin
    logic helper_ready_now;
    logic sfu_ready_now;

    // Stage D keeps the SFU serialized against the SRAM-using helper, DMA, and
    // systolic paths. Helper dispatch is also blocked while an SFU op is active.
    helper_ready_now = !dma_busy && !sys_busy && !helper_busy && !sfu_busy;
    sfu_ready_now    = !dma_busy && !sys_busy && !helper_busy && !sfu_busy;

    // Defaults
    fetch_req    = 1'b0;
    done         = 1'b0;
    fault        = 1'b0;

    scale_we     = 1'b0;
    scale_waddr  = insn.s_sreg;
    scale_wdata  = insn.s_imm16;

    addr_lo_we   = 1'b0;
    addr_hi_we   = 1'b0;
    addr_wsel    = insn.a_addr_reg;
    addr_imm28   = insn.a_imm28;

    tile_we      = 1'b0;
    tile_m_in    = insn.c_tile_m;
    tile_n_in    = insn.c_tile_n;
    tile_k_in    = insn.c_tile_k;

    dma_dispatch = 1'b0;
    sys_dispatch = 1'b0;
    sfu_dispatch = 1'b0;
    helper_dispatch = 1'b0;

    case (state)
      S_FETCH: begin
        fetch_req = 1'b1;
      end

      S_ISSUE: begin
        if (!insn.illegal && !ext_fault && !unsupported_now) begin
          case (insn.opcode)
            OP_CONFIG_TILE:
              tile_we = 1'b1;

            OP_SET_SCALE:
              scale_we = 1'b1;

            OP_SET_ADDR_LO:
              addr_lo_we = 1'b1;

            OP_SET_ADDR_HI:
              addr_hi_we = 1'b1;

            OP_LOAD, OP_STORE:
              dma_dispatch = !sfu_busy;

            OP_MATMUL:
              sys_dispatch = tile_valid && !sfu_busy && !helper_busy;

            OP_BUF_COPY:
              helper_dispatch = helper_ready_now;

            OP_REQUANT, OP_REQUANT_PC, OP_SCALE_MUL, OP_VADD, OP_DEQUANT_ADD:
              helper_dispatch = tile_valid && helper_ready_now;

            OP_SOFTMAX, OP_SOFTMAX_ATTNV, OP_LAYERNORM, OP_GELU:
              sfu_dispatch = tile_valid && sfu_ready_now;

            default: ;
          endcase
        end
      end

      S_HALT:
        done = 1'b1;

      S_FAULT:
        fault = 1'b1;

      default: ;
    endcase
  end

  // The program runner samples these wait-cause bits to attribute SYNC stalls
  // to specific asynchronous engines.
  assign obs_sync_wait_dma = (state == S_SYNC_WAIT) &&
                             sync_mask_q[SYNC_DMA_BIT] && dma_busy;
  assign obs_sync_wait_sys = (state == S_SYNC_WAIT) &&
                             sync_mask_q[SYNC_SYS_BIT] && sys_busy;
  assign obs_sync_wait_sfu = (state == S_SYNC_WAIT) &&
                             sync_mask_q[SYNC_SFU_BIT] && sfu_busy;

endmodule

`endif // CONTROL_UNIT_SV
