// DMA Engine -- LOAD (DRAM->SRAM) and STORE (SRAM->DRAM) via AXI4 master.
//
// LOAD:  Issues sequential AXI4 read bursts of up to 256 beats each.
//        Each accepted 16-byte beat is written to SRAM port A at consecutive rows.
//
// STORE: Issues sequential AXI4 write bursts of up to 256 beats each.
//        SRAM port A is read 1 cycle ahead per beat (registered SRAM output).
//
// Effective DRAM byte address = base_addr + (dram_off × 16).
// Whole-transfer prevalidation is performed before the first AXI request:
//   - DRAM end address must remain within DRAM_SIZE
//   - SRAM row range must remain within the selected buffer
//
// xfer_len=0 is a legal no-op after prevalidation. All parameters are latched
// on the dispatch pulse; insn fields may change after.

`ifndef DMA_ENGINE_SV
`define DMA_ENGINE_SV

`include "taccel_pkg.sv"

module dma_engine
  import taccel_pkg::*;
#(
  parameter int DRAM_SIZE = 1 << 24   // 16 MB default
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Dispatch (1-cycle pulse from control_unit) ---
  input  logic        dispatch,
  input  logic        is_store,        // 0=LOAD, 1=STORE
  input  logic [1:0]  buf_id,
  input  logic [15:0] sram_off,        // SRAM start row (16-byte units)
  input  logic [15:0] xfer_len,        // number of 16-byte beats (0..65535)
  input  logic [55:0] base_addr,       // DRAM base address (from addr_reg)
  input  logic [15:0] dram_off,        // DRAM row offset (×16 = byte offset)

  // --- Status ---
  output logic        dma_busy,        // asserted while any state != IDLE
  output logic        dma_rd_busy,     // asserted while a DMA read burst is pending/in flight
  output logic        dma_fault,
  output logic [3:0]  dma_fault_code,

  // --- SRAM Port A ---
  output logic         sram_en,
  output logic         sram_we,
  output logic [1:0]   sram_buf,
  output logic [15:0]  sram_row,
  output logic [127:0] sram_wdata,
  input  logic [127:0] sram_rdata,     // valid 1 cycle after sram_en && !sram_we
  input  logic         sram_fault,     // OOB or reserved buffer on the selected row

  // --- AXI4 read channels (LOAD) ---
  output logic [AXI_ADDR_W-1:0] dma_ar_addr,
  output logic [7:0]             dma_ar_len,
  output logic                   dma_ar_valid,
  input  logic                   dma_ar_ready,
  input  logic [AXI_DATA_W-1:0]  dma_r_data,
  input  logic [1:0]             dma_r_resp,
  input  logic                   dma_r_valid,
  input  logic                   dma_r_last,
  output logic                   dma_r_ready,

  // --- AXI4 write channels (STORE) ---
  output logic [AXI_ADDR_W-1:0]  dma_aw_addr,
  output logic [7:0]             dma_aw_len,
  output logic                   dma_aw_valid,
  input  logic                   dma_aw_ready,
  output logic [AXI_DATA_W-1:0]  dma_w_data,
  output logic [15:0]            dma_w_strb,
  output logic                   dma_w_valid,
  output logic                   dma_w_last,
  input  logic                   dma_w_ready,
  input  logic [1:0]             dma_b_resp,
  input  logic                   dma_b_valid,
  output logic                   dma_b_ready
);

  // -------------------------------------------------------------------------
  // FSM.
  // LOAD walks bursts as AR -> R.
  // STORE walks bursts as AW -> SRAM pre-read -> W -> B because port A is a
  // synchronous single read/write port.
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    D_IDLE           = 3'd0,
    D_LOAD_AR        = 3'd1,
    D_LOAD_R         = 3'd2,
    D_STORE_AW       = 3'd3,
    D_STORE_SRAM_PRE = 3'd4,
    D_STORE_W        = 3'd5,
    D_STORE_B        = 3'd6,
    D_FAULT          = 3'd7
  } dma_state_t;

  dma_state_t  state;

  // Latched whole-transfer state. The engine reuses these registers across many
  // AXI bursts, updating them only at burst boundaries.
  logic        is_store_q;
  logic [1:0]  buf_id_q;
  logic [15:0] curr_sram_row_q;
  logic [15:0] beats_remaining_q;
  logic [15:0] burst_beats_q;
  logic [15:0] burst_beat_idx_q;
  logic [55:0] curr_dram_addr_q;
  logic [3:0]  fault_code_r;

  logic [55:0] burst_bytes_w;
  logic [15:0] remaining_after_burst_w;
  logic [15:0] next_burst_beats_w;
  logic [55:0] dram_addr_after_burst_w;
  logic [15:0] sram_row_after_burst_w;
  logic        burst_last_beat_w;
  logic        transfer_last_burst_w;
  logic        load_beat_fault_w;
  logic        load_beat_accept_w;

  logic [56:0] dispatch_dram_byte_addr_w;
  logic [56:0] dispatch_dram_end_w;
  logic [15:0] dispatch_buf_rows_w;
  logic [16:0] dispatch_sram_end_w;
  logic        dispatch_dram_oob_w;
  logic        dispatch_sram_oob_w;

  function automatic logic [15:0] buf_rows(input logic [1:0] bid);
    begin
      case (bid)
        BUF_ABUF:  buf_rows = 16'(ABUF_ROWS);
        BUF_WBUF:  buf_rows = 16'(WBUF_ROWS);
        BUF_ACCUM: buf_rows = 16'(ACCUM_ROWS);
        default:   buf_rows = 16'h0;
      endcase
    end
  endfunction

  function automatic logic [15:0] burst_beats(input logic [15:0] remaining);
    begin
      if (remaining > 16'd256)
        burst_beats = 16'd256;
      else
        burst_beats = remaining;
    end
  endfunction

  assign burst_bytes_w           = {36'h0, burst_beats_q, 4'b0};
  assign remaining_after_burst_w = beats_remaining_q - burst_beats_q;
  assign next_burst_beats_w      = burst_beats(remaining_after_burst_w);
  assign dram_addr_after_burst_w = curr_dram_addr_q + burst_bytes_w;
  assign sram_row_after_burst_w  = curr_sram_row_q + burst_beats_q;
  assign burst_last_beat_w       = (burst_beat_idx_q == (burst_beats_q - 16'h1));
  assign transfer_last_burst_w   = (beats_remaining_q == burst_beats_q);

  // For LOAD, AXI protocol correctness and SRAM validity are checked per beat.
  // The final beat of each burst must be the only beat that asserts RLAST.
  assign load_beat_fault_w =
      sram_fault |
      (dma_r_resp != 2'b00) |
      (dma_r_last != burst_last_beat_w);
  // Gate SRAM writes only on AXI protocol validity; the SRAM itself suppresses
  // writes on a_fault, which avoids a combinational loop back through sram_fault.
  assign load_beat_accept_w =
      dma_r_valid &
      (dma_r_resp == 2'b00) &
      (dma_r_last == burst_last_beat_w);

  assign dispatch_dram_byte_addr_w = {1'b0, base_addr} + {37'h0, dram_off, 4'b0};
  assign dispatch_dram_end_w       = dispatch_dram_byte_addr_w + {37'h0, xfer_len, 4'b0};
  assign dispatch_buf_rows_w       = buf_rows(buf_id);
  assign dispatch_sram_end_w       = {1'b0, sram_off} + {1'b0, xfer_len};
  assign dispatch_dram_oob_w       = (dispatch_dram_end_w > 57'(DRAM_SIZE));
  assign dispatch_sram_oob_w =
      (dispatch_buf_rows_w == 16'h0) |
      ((xfer_len == 16'h0) ? (sram_off >= dispatch_buf_rows_w)
                           : (dispatch_sram_end_w > {1'b0, dispatch_buf_rows_w}));

  // -------------------------------------------------------------------------
  // Sequential FSM.
  // Whole-transfer OOB checks happen once in D_IDLE before any side effects.
  // Mid-transfer faults are terminal; completed beats are not rolled back.
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state             <= D_IDLE;
      is_store_q        <= 1'b0;
      buf_id_q          <= 2'b00;
      curr_sram_row_q   <= 16'h0;
      beats_remaining_q <= 16'h0;
      burst_beats_q     <= 16'h0;
      burst_beat_idx_q  <= 16'h0;
      curr_dram_addr_q  <= 56'h0;
      fault_code_r      <= 4'h0;
    end else begin
      case (state)
        D_IDLE: begin
          if (dispatch) begin
            is_store_q        <= is_store;
            buf_id_q          <= buf_id;
            curr_sram_row_q   <= sram_off;
            beats_remaining_q <= xfer_len;
            burst_beats_q     <= burst_beats(xfer_len);
            burst_beat_idx_q  <= 16'h0;
            curr_dram_addr_q  <= dispatch_dram_byte_addr_w[55:0];

            if (dispatch_dram_oob_w) begin
              fault_code_r <= 4'(FAULT_DRAM_OOB);
              state        <= D_FAULT;
            end else if (dispatch_sram_oob_w) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= D_FAULT;
            end else if (xfer_len != 16'h0) begin
              state <= is_store ? D_STORE_AW : D_LOAD_AR;
            end
          end
        end

        D_LOAD_AR: begin
          if (dma_ar_ready)
            state <= D_LOAD_R;
        end

        D_LOAD_R: begin
          if (dma_r_valid) begin
            if (sram_fault) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= D_FAULT;
            end else if (dma_r_resp != 2'b00) begin
              fault_code_r <= 4'(FAULT_DRAM_OOB);
              state        <= D_FAULT;
            end else if (dma_r_last != burst_last_beat_w) begin
              fault_code_r <= 4'(FAULT_DRAM_OOB);
              state        <= D_FAULT;
            end else if (burst_last_beat_w) begin
              if (transfer_last_burst_w) begin
                state <= D_IDLE;
              end else begin
                curr_dram_addr_q  <= dram_addr_after_burst_w;
                curr_sram_row_q   <= sram_row_after_burst_w;
                beats_remaining_q <= remaining_after_burst_w;
                burst_beats_q     <= next_burst_beats_w;
                burst_beat_idx_q  <= 16'h0;
                state             <= D_LOAD_AR;
              end
            end else begin
              burst_beat_idx_q <= burst_beat_idx_q + 16'h1;
            end
          end
        end

        D_STORE_AW: begin
          if (dma_aw_ready) begin
            burst_beat_idx_q <= 16'h0;
            state            <= D_STORE_SRAM_PRE;
          end
        end

        D_STORE_SRAM_PRE: begin
          if (sram_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= D_FAULT;
          end else begin
            state <= D_STORE_W;
          end
        end

        D_STORE_W: begin
          if (dma_w_ready) begin
            if (burst_last_beat_w) begin
              state <= D_STORE_B;
            end else begin
              burst_beat_idx_q <= burst_beat_idx_q + 16'h1;
              state            <= D_STORE_SRAM_PRE;
            end
          end
        end

        D_STORE_B: begin
          if (dma_b_valid) begin
            if (dma_b_resp != 2'b00) begin
              fault_code_r <= 4'(FAULT_DRAM_OOB);
              state        <= D_FAULT;
            end else if (transfer_last_burst_w) begin
              state <= D_IDLE;
            end else begin
              curr_dram_addr_q  <= dram_addr_after_burst_w;
              curr_sram_row_q   <= sram_row_after_burst_w;
              beats_remaining_q <= remaining_after_burst_w;
              burst_beats_q     <= next_burst_beats_w;
              burst_beat_idx_q  <= 16'h0;
              state             <= D_STORE_AW;
            end
          end
        end

        D_FAULT: ;  // terminal — cleared only by reset

        default: state <= D_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Combinational outputs.
  // `dma_rd_busy` is intentionally burst-scoped rather than transfer-scoped so
  // fetch can slip in between accepted DMA read bursts.
  // -------------------------------------------------------------------------
  always_comb begin
    dma_busy       = (state != D_IDLE);
    dma_rd_busy    = (state == D_LOAD_AR || state == D_LOAD_R);
    dma_fault      = (state == D_FAULT);
    dma_fault_code = fault_code_r;

    // AXI read defaults
    dma_ar_addr  = curr_dram_addr_q;
    dma_ar_len   = (burst_beats_q == 16'h0) ? 8'h00 : 8'(burst_beats_q - 16'h1);
    dma_ar_valid = 1'b0;
    dma_r_ready  = 1'b0;

    // AXI write defaults
    dma_aw_addr  = curr_dram_addr_q;
    dma_aw_len   = (burst_beats_q == 16'h0) ? 8'h00 : 8'(burst_beats_q - 16'h1);
    dma_aw_valid = 1'b0;
    dma_w_data   = 128'h0;
    dma_w_strb   = 16'hFFFF;
    dma_w_valid  = 1'b0;
    dma_w_last   = 1'b0;
    dma_b_ready  = 1'b0;

    // SRAM defaults
    sram_en    = 1'b0;
    sram_we    = 1'b0;
    sram_buf   = buf_id_q;
    sram_row   = curr_sram_row_q + burst_beat_idx_q;
    sram_wdata = dma_r_data;

    case (state)
      D_LOAD_AR: begin
        dma_ar_valid = 1'b1;
      end

      D_LOAD_R: begin
        dma_r_ready = 1'b1;
        if (load_beat_accept_w) begin
          sram_en    = 1'b1;
          sram_we    = 1'b1;
          sram_wdata = dma_r_data;
        end
      end

      D_STORE_AW: begin
        dma_aw_valid = 1'b1;
      end

      D_STORE_SRAM_PRE: begin
        // Prime the synchronous SRAM so its row appears on sram_rdata during
        // the following D_STORE_W cycle.
        sram_en = 1'b1;
        sram_we = 1'b0;
      end

      D_STORE_W: begin
        dma_w_valid = 1'b1;
        dma_w_data  = sram_rdata;
        dma_w_last  = burst_last_beat_w;
      end

      D_STORE_B: begin
        dma_b_ready = 1'b1;
      end

      default: ;
    endcase
  end

endmodule

`endif // DMA_ENGINE_SV
