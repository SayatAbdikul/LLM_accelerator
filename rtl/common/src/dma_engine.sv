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
  // Lever D: transposed LOAD. When is_transpose=1 (LOAD only), the contiguous
  // (R, C) INT8 read is written to SRAM as its (C, R) transpose, where
  // C = 16 << cols_log2. Ignored for STORE. See the D_T_* states below.
  input  logic        is_transpose,
  input  logic [3:0]  cols_log2,

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
  //
  // Pipelining (2026-05-27): the dispatch-time 57-bit add + OOB compare was the
  // critical path (7.49 ns on sky130_fd_sc_hd TT). It is now split across
  // D_IDLE (just the add latched into curr_dram_addr_q) and D_DISPATCH_CHECK
  // (the end+compare on the latched value). +1 cycle of dispatch latency.
  // The burst-boundary 56-bit add is similarly precomputed every cycle into
  // next_*_q registers, so the burst-done transition is flop->mux->flop with
  // no in-path adder.
  // -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    D_IDLE           = 4'd0,
    D_DISPATCH_CHECK = 4'd1,
    D_LOAD_AR        = 4'd2,
    D_LOAD_R         = 4'd3,
    D_STORE_AW       = 4'd4,
    D_STORE_SRAM_PRE = 4'd5,
    D_STORE_W        = 4'd6,
    D_STORE_B        = 4'd7,
    D_FAULT          = 4'd8,
    // Lever D transposed-load states: read one 16-row stripe into `tbuf`,
    // then write its C transposed columns to strided SRAM rows.
    D_T_AR           = 4'd9,
    D_T_R            = 4'd10,
    D_T_WRITE        = 4'd11
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

  // ---- Lever D transposed-load state ----
  // A transposed LOAD walks `t_rows_tiles` stripes. Each stripe reads
  // C=t_csize contiguous beats (16 input rows × C/16 beats) into `tbuf`, then
  // writes C transposed output beats to SRAM rows `base + col*t_rows_tiles + s`.
  // C == stripe_beats == 16<<cols_log2, so one reg (t_csize_q) serves both.
  localparam int T_MAX_COLS = 64;              // max C (d_head) supported
  logic [7:0]  tbuf [0:15][0:T_MAX_COLS-1];    // 16 rows × C bytes (1 KB)
  logic        is_transpose_q;
  logic [3:0]  cols_log2_q;
  logic [15:0] t_csize_q;        // C = stripe_beats = output-row count
  logic [15:0] t_rows_tiles_q;   // stripe count = output-row stride (beats)
  logic [15:0] t_stripe_idx_q;   // current stripe 0..t_rows_tiles-1
  logic [15:0] t_beat_idx_q;     // current input beat within stripe 0..C-1
  logic [15:0] t_col_idx_q;      // current output column during write 0..C-1
  logic [15:0] t_out_row_q;      // running SRAM row = base + col*rows_tiles + s
  logic [15:0] t_base_row_q;     // SRAM base row (sram_off), constant per xfer

  logic [15:0] t_cols_beats_w;   // C/16 = 1 << cols_log2
  logic [15:0] t_row_in_w;       // input row of current beat = beat >> cols_log2
  logic [15:0] t_colgrp_w;       // input beat's 16-col group = beat & (C/16-1)
  logic [15:0] t_stripe_bytes_w; // bytes per stripe = C*16
  logic        t_read_last_w;    // last beat of the stripe read
  logic        t_write_last_w;   // last column of the stripe write
  logic        t_stripe_last_w;  // last stripe of the transfer
  logic [6:0]  t_wr_col_base_w;  // tbuf byte-col base for the beat = colgrp*16
  logic [127:0] t_out_beat_w;    // gathered transposed output beat (column)

  assign t_cols_beats_w  = 16'h1 << cols_log2_q;
  assign t_row_in_w      = t_beat_idx_q >> cols_log2_q;
  assign t_colgrp_w      = t_beat_idx_q & (t_cols_beats_w - 16'h1);
  assign t_wr_col_base_w = t_colgrp_w[2:0] << 4;   // 0,16,32,48 (C≤64)
  assign t_stripe_bytes_w= {t_csize_q[11:0], 4'b0};   // C beats × 16 bytes
  assign t_read_last_w   = (t_beat_idx_q == (t_csize_q - 16'h1));
  assign t_write_last_w  = (t_col_idx_q == (t_csize_q - 16'h1));
  assign t_stripe_last_w = (t_stripe_idx_q == (t_rows_tiles_q - 16'h1));

  // Transposed output beat for the current column: byte i = tbuf[i][col].
  always_comb begin
    for (int i = 0; i < 16; i++)
      t_out_beat_w[i*8 +: 8] = tbuf[i][t_col_idx_q[5:0]];
  end

  // Precomputed next-burst params, registered one cycle ahead of the
  // burst-done transition. Removes the curr_dram_addr_q + burst_bytes_w
  // adder + control-mux tail from the burst-boundary critical path.
  logic [55:0] next_dram_addr_q;
  logic [15:0] next_sram_row_q;
  logic [15:0] next_beats_remaining_q;
  logic [15:0] next_burst_beats_q;

  logic [55:0] burst_bytes_w;
  logic [15:0] remaining_after_burst_w;
  logic [15:0] next_burst_beats_w;
  logic [55:0] dram_addr_after_burst_w;
  logic [15:0] sram_row_after_burst_w;
  logic        burst_last_beat_w;
  logic        transfer_last_burst_w;
  logic        load_beat_fault_w;
  logic        load_beat_accept_w;

  // Dispatch-cycle: only the start-address add is left here. The end-of-
  // transfer compare moved to D_DISPATCH_CHECK and operates on latched regs.
  logic [56:0] dispatch_dram_byte_addr_w;

  // D_DISPATCH_CHECK-cycle: OOB checks on the latched dispatch params.
  logic [56:0] latched_dram_end_w;
  logic        latched_dram_oob_w;
  logic [15:0] latched_buf_rows_w;
  logic [16:0] latched_sram_end_w;
  logic        latched_sram_oob_w;

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

  // D_IDLE-cycle start address. Cut here: the result is registered into
  // curr_dram_addr_q, and the end-compare is done in D_DISPATCH_CHECK using
  // that registered value (see latched_* below).
  assign dispatch_dram_byte_addr_w = {1'b0, base_addr} + {37'h0, dram_off, 4'b0};

  // D_DISPATCH_CHECK-cycle OOB on the post-latch state. Equivalent to the
  // previous combinational chain but the inputs are registered so the path
  // is short.
  assign latched_dram_end_w  = {1'b0, curr_dram_addr_q}
                             + {37'h0, beats_remaining_q, 4'b0};
  assign latched_dram_oob_w  = (latched_dram_end_w > 57'(DRAM_SIZE));
  assign latched_buf_rows_w  = buf_rows(buf_id_q);
  assign latched_sram_end_w  = {1'b0, curr_sram_row_q} + {1'b0, beats_remaining_q};
  assign latched_sram_oob_w  =
      (latched_buf_rows_w == 16'h0) |
      ((beats_remaining_q == 16'h0) ? (curr_sram_row_q >= latched_buf_rows_w)
                                    : (latched_sram_end_w > {1'b0, latched_buf_rows_w}));

  // -------------------------------------------------------------------------
  // Sequential FSM.
  // Whole-transfer OOB checks happen once in D_IDLE before any side effects.
  // Mid-transfer faults are terminal; completed beats are not rolled back.
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state                  <= D_IDLE;
      is_store_q             <= 1'b0;
      buf_id_q               <= 2'b00;
      curr_sram_row_q        <= 16'h0;
      beats_remaining_q      <= 16'h0;
      burst_beats_q          <= 16'h0;
      burst_beat_idx_q       <= 16'h0;
      curr_dram_addr_q       <= 56'h0;
      fault_code_r           <= 4'h0;
      next_dram_addr_q       <= 56'h0;
      next_sram_row_q        <= 16'h0;
      next_beats_remaining_q <= 16'h0;
      next_burst_beats_q     <= 16'h0;
      is_transpose_q         <= 1'b0;
      cols_log2_q            <= 4'h0;
      t_csize_q              <= 16'h0;
      t_rows_tiles_q         <= 16'h0;
      t_stripe_idx_q         <= 16'h0;
      t_beat_idx_q           <= 16'h0;
      t_col_idx_q            <= 16'h0;
      t_out_row_q            <= 16'h0;
      t_base_row_q           <= 16'h0;
    end else begin
      // Burst-boundary precompute: registers the (current burst +1) values
      // every cycle so the burst-done transition only needs a flop->mux->flop
      // path, no in-path adder.
      next_dram_addr_q       <= dram_addr_after_burst_w;
      next_sram_row_q        <= sram_row_after_burst_w;
      next_beats_remaining_q <= remaining_after_burst_w;
      next_burst_beats_q     <= next_burst_beats_w;

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
            // Lever D: latch transpose mode + geometry. Only honored for LOAD.
            is_transpose_q    <= is_transpose && !is_store;
            cols_log2_q       <= cols_log2;
            t_base_row_q      <= sram_off;
            // OOB check moved to D_DISPATCH_CHECK so this cycle's critical
            // path is only the 57-bit add (no compare/select tail).
            state             <= D_DISPATCH_CHECK;
          end
        end

        D_DISPATCH_CHECK: begin
          if (latched_dram_oob_w) begin
            fault_code_r <= 4'(FAULT_DRAM_OOB);
            state        <= D_FAULT;
          end else if (latched_sram_oob_w) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= D_FAULT;
          end else if (beats_remaining_q == 16'h0) begin
            // Legal no-op dispatch (xfer_len=0, in-range sram_off).
            state <= D_IDLE;
          end else if (is_transpose_q) begin
            // Lever D: derive stripe geometry (shifts only). C = 16<<cols_log2
            // is both the stripe beat count and the output-column count;
            // rows_tiles = xfer_len / (16*C/16) = xfer_len >> (4+cols_log2).
            t_csize_q      <= 16'h10 << cols_log2_q;
            t_rows_tiles_q <= beats_remaining_q >> (5'(cols_log2_q) + 5'd4);
            t_stripe_idx_q <= 16'h0;
            t_beat_idx_q   <= 16'h0;
            state          <= D_T_AR;
          end else begin
            state <= is_store_q ? D_STORE_AW : D_LOAD_AR;
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
                // Use precomputed next-burst regs (one cycle stale, which is
                // exactly the burst transition cadence — correct by construction).
                curr_dram_addr_q  <= next_dram_addr_q;
                curr_sram_row_q   <= next_sram_row_q;
                beats_remaining_q <= next_beats_remaining_q;
                burst_beats_q     <= next_burst_beats_q;
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
              curr_dram_addr_q  <= next_dram_addr_q;
              curr_sram_row_q   <= next_sram_row_q;
              beats_remaining_q <= next_beats_remaining_q;
              burst_beats_q     <= next_burst_beats_q;
              burst_beat_idx_q  <= 16'h0;
              state             <= D_STORE_AW;
            end
          end
        end

        // ---- Lever D transposed-load stripe loop ----
        D_T_AR: begin
          if (dma_ar_ready) begin
            t_beat_idx_q <= 16'h0;
            state        <= D_T_R;
          end
        end

        D_T_R: begin
          if (dma_r_valid) begin
            if (dma_r_resp != 2'b00) begin
              fault_code_r <= 4'(FAULT_DRAM_OOB);
              state        <= D_FAULT;
            end else if (dma_r_last != t_read_last_w) begin
              fault_code_r <= 4'(FAULT_DRAM_OOB);
              state        <= D_FAULT;
            end else begin
              // Route this beat into tbuf row (beat>>cols_log2), 16-col group
              // (beat & (C/16-1)). Registered; the read-out gathers columns.
              for (int j = 0; j < 16; j++)
                tbuf[t_row_in_w[3:0]][t_wr_col_base_w[5:0] + 6'(j)] <= dma_r_data[j*8 +: 8];
              if (t_read_last_w) begin
                t_col_idx_q <= 16'h0;
                t_out_row_q <= t_base_row_q + t_stripe_idx_q;  // col 0 row
                state       <= D_T_WRITE;
              end else begin
                t_beat_idx_q <= t_beat_idx_q + 16'h1;
              end
            end
          end
        end

        D_T_WRITE: begin
          // One transposed output beat per cycle (SRAM write in the comb block).
          if (t_write_last_w) begin
            if (t_stripe_last_w) begin
              state <= D_IDLE;
            end else begin
              curr_dram_addr_q <= curr_dram_addr_q + {40'h0, t_stripe_bytes_w};
              t_stripe_idx_q   <= t_stripe_idx_q + 16'h1;
              t_beat_idx_q     <= 16'h0;
              state            <= D_T_AR;
            end
          end else begin
            t_col_idx_q <= t_col_idx_q + 16'h1;
            t_out_row_q <= t_out_row_q + t_rows_tiles_q;  // next output row
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
    dma_rd_busy    = (state == D_LOAD_AR || state == D_LOAD_R ||
                      state == D_T_AR    || state == D_T_R);
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

      // ---- Lever D transposed-load ----
      D_T_AR: begin
        dma_ar_addr  = curr_dram_addr_q;
        dma_ar_len   = 8'(t_csize_q - 16'h1);   // C beats (≤64) per stripe
        dma_ar_valid = 1'b1;
      end

      D_T_R: begin
        dma_r_ready = 1'b1;                      // beats routed to tbuf (seq)
      end

      D_T_WRITE: begin
        sram_en    = 1'b1;
        sram_we    = 1'b1;
        sram_buf   = buf_id_q;
        sram_row   = t_out_row_q;
        sram_wdata = t_out_beat_w;
      end

      default: ;
    endcase
  end

endmodule

`endif // DMA_ENGINE_SV
