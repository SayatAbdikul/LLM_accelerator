`ifndef SYSTOLIC_CONTROLLER_SV
`define SYSTOLIC_CONTROLLER_SV

`include "taccel_pkg.sv"

// Systolic MATMUL controller.
//
// Responsibilities:
//   - latch one MATMUL instruction on dispatch
//   - walk the logical M/N/K tile grid one 16x16 systolic tile at a time
//   - stream src1/src2 rows into the array
//   - drain the 16x16 INT32 accumulator tile back to SRAM
//
// The controller is asynchronous from the main control FSM: control emits a
// one-cycle dispatch pulse, then later observes `sys_busy` dropping to know the
// operation has completed.

module systolic_controller
  import taccel_pkg::*;
#(
  parameter int SYSTOLIC_ARCH_MODE = SYS_MODE_DEFAULT
)
(
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 dispatch,

  input  logic [9:0]           tile_m,
  input  logic [9:0]           tile_n,
  input  logic [9:0]           tile_k,

  input  logic [1:0]           src1_buf,
  input  logic [15:0]          src1_off,
  input  logic [1:0]           src2_buf,
  input  logic [15:0]          src2_off,
  input  logic [1:0]           dst_buf,
  input  logic [15:0]          dst_off,
  input  logic                 flags_accumulate,

  output logic                 sys_busy,

  // SRAM port A (read/write): src2 read and dst writeback
  output logic                 sram_a_en,
  output logic                 sram_a_we,
  output logic [1:0]           sram_a_buf,
  output logic [15:0]          sram_a_row,
  output logic [127:0]         sram_a_wdata,
  input  logic [127:0]         sram_a_rdata,

  // SRAM port B (read-only): src1 read
  output logic                 sram_b_en,
  output logic [1:0]           sram_b_buf,
  output logic [15:0]          sram_b_row,
  input  logic [127:0]         sram_b_rdata
);

  // READ_REQ issues synchronous SRAM reads, READ_USE consumes the returned rows
  // one cycle later, and DRAIN_WR writes the accumulated 16x16 tile back out as
  // 64 rows of 4xINT32 each.
  typedef enum logic [3:0] {
    ST_IDLE       = 4'd0,
    ST_INIT_TILE  = 4'd1,
    ST_READ_REQ   = 4'd2,
    ST_READ_USE   = 4'd3,
    ST_DRAIN_PREP = 4'd4,
    ST_DRAIN_WR   = 4'd5,
    ST_A_LOAD_REQ = 4'd6,
    ST_A_LOAD_LATCH = 4'd7,
    ST_DST_CLEAR_PREP = 4'd8,
    ST_DST_CLEAR_WR   = 4'd9
  } state_t;

  state_t state;

  logic [1:0]  src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0] src1_off_q, src2_off_q, dst_off_q;
  logic        flags_accumulate_q;

  logic [10:0] m_tiles_q, n_tiles_q, k_tiles_q;
  logic [10:0] mtile_q, ntile_q, ktile_q;
  logic [5:0]  lane_q;
  logic [4:0]  a_load_row_q;
  logic [4:0]  drain_row_q;
  logic [1:0]  drain_grp_q;
  logic [31:0] dst_clear_row_idx_q;
  logic [31:0] dst_clear_total_rows_q;
  logic [7:0]  a_tile_scratch [0:SYS_DIM-1][0:SYS_DIM-1];

  // Row-major drain address tracking.
  // tile_drain_base_q = dst_off + mtile * n_tiles * 64 (advances by n_tiles*64 per M-tile).
  // drain_row_addr_q  = tile_drain_base_q + ntile*4 + drain_row*(n_tiles*4);
  //   advances by n_tiles*4 on each drain_row increment.
  // sram_a_row = drain_row_addr_q + drain_grp_q  (in ST_DRAIN_WR)
  logic [15:0] tile_drain_base_q;
  logic [15:0] drain_row_addr_q;

  logic step_en;
  logic clear_acc;
  logic       inject_zero_data;
  logic [15:0] lane_row_idx;
  logic [127:0] a_row_data_q, b_row_data_q;
  logic [SYS_DIM*SYS_DIM*32-1:0] acc_flat;

  localparam int CHAIN_FLUSH_CYCLES = (2 * (SYS_DIM - 1));
  localparam int CHAIN_TOTAL_STEPS  = SYS_DIM + CHAIN_FLUSH_CYCLES;

  systolic_array #(
    .SYSTOLIC_ARCH_MODE(SYSTOLIC_ARCH_MODE)
  ) u_array (
    .clk      (clk),
    .rst_n    (rst_n),
    .step_en  (step_en),
    .clear_acc(clear_acc),
    .a_row_data(a_row_data_q),
    .b_row_data(b_row_data_q),
    .acc_flat (acc_flat)
  );

  function automatic logic [31:0] acc_at(
    input logic [4:0] r,
    input logic [4:0] c
  );
    int idx;
    begin
      idx = ((int'(r) * SYS_DIM) + int'(c)) * 32;
      acc_at = acc_flat[idx +: 32];
    end
  endfunction

  // Source SRAM is presented to MATMUL in compiler-visible row-major form:
  //   src1 = A[M,K] with K/16 row units per logical row
  //   src2 = B[K,N] with N/16 row units per logical row
  // The controller stages one 16x16 A tile locally, then streams its columns
  // into the array while reading the matching B rows directly from SRAM.
  logic [31:0] src1_row_units, src2_row_units;
  logic [31:0] src1_logical_row, src2_logical_row;
  logic [31:0] src1_load_row_addr, src2_stream_row_addr;
  logic [31:0] dispatch_m_tiles_w, dispatch_n_tiles_w, dispatch_clear_rows_w;
  logic        needs_dst_preclear_w;
  always_comb begin
    src1_row_units = {21'h0, k_tiles_q};
    src2_row_units = {21'h0, n_tiles_q};

    dispatch_m_tiles_w = {22'h0, tile_m} + 32'd1;
    dispatch_n_tiles_w = {22'h0, tile_n} + 32'd1;
    dispatch_clear_rows_w = (dispatch_m_tiles_w * dispatch_n_tiles_w) << 6;
    needs_dst_preclear_w = !flags_accumulate && (dst_buf == BUF_ACCUM);

    src1_logical_row = ({21'h0, mtile_q} << 4) + {27'h0, a_load_row_q};
    src2_logical_row = ({21'h0, ktile_q} << 4) + {26'h0, lane_q};

    src1_load_row_addr = {16'h0, src1_off_q} + (src1_logical_row * src1_row_units) + {21'h0, ktile_q};
    src2_stream_row_addr = {16'h0, src2_off_q} + (src2_logical_row * src2_row_units) + {21'h0, ntile_q};
  end

  // Tile-walking FSM. One MATMUL dispatch can cover many logical 16x16 tiles
  // when CONFIG_TILE programmed M/N/K larger than zero.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      src1_buf_q <= 2'b0;
      src2_buf_q <= 2'b0;
      dst_buf_q <= 2'b0;
      src1_off_q <= 16'h0;
      src2_off_q <= 16'h0;
      dst_off_q <= 16'h0;
      flags_accumulate_q <= 1'b0;
      m_tiles_q <= 11'd0;
      n_tiles_q <= 11'd0;
      k_tiles_q <= 11'd0;
      mtile_q <= 11'd0;
      ntile_q <= 11'd0;
      ktile_q <= 11'd0;
      lane_q <= 6'd0;
      a_load_row_q <= 5'd0;
      drain_row_q <= 5'd0;
      drain_grp_q <= 2'd0;
      dst_clear_row_idx_q <= 32'd0;
      dst_clear_total_rows_q <= 32'd0;
      tile_drain_base_q <= 16'h0;
      drain_row_addr_q <= 16'h0;
      for (int row_idx = 0; row_idx < SYS_DIM; row_idx = row_idx + 1)
        for (int col_idx = 0; col_idx < SYS_DIM; col_idx = col_idx + 1)
          a_tile_scratch[row_idx][col_idx] <= 8'h0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (dispatch) begin
            src1_buf_q <= src1_buf;
            src2_buf_q <= src2_buf;
            dst_buf_q <= dst_buf;
            src1_off_q <= src1_off;
            src2_off_q <= src2_off;
            dst_off_q <= dst_off;
            flags_accumulate_q <= flags_accumulate;
            tile_drain_base_q <= dst_off;
            m_tiles_q <= {1'b0, tile_m} + 11'd1;
            n_tiles_q <= {1'b0, tile_n} + 11'd1;
            k_tiles_q <= {1'b0, tile_k} + 11'd1;
            mtile_q <= 11'd0;
            ntile_q <= 11'd0;
            ktile_q <= 11'd0;
            lane_q <= 6'd0;
            a_load_row_q <= 5'd0;
            drain_row_q <= 5'd0;
            drain_grp_q <= 2'd0;
            dst_clear_row_idx_q <= 32'd0;
            dst_clear_total_rows_q <= dispatch_clear_rows_w;
            tile_drain_base_q <= dst_off;
            drain_row_addr_q <= dst_off;
            state <= needs_dst_preclear_w ? ST_DST_CLEAR_PREP : ST_INIT_TILE;
          end
        end

        ST_DST_CLEAR_PREP: begin
          dst_clear_row_idx_q <= 32'd0;
          if (dst_clear_total_rows_q == 32'd0)
            state <= ST_INIT_TILE;
          else
            state <= ST_DST_CLEAR_WR;
        end

        ST_DST_CLEAR_WR: begin
          if (dst_clear_row_idx_q + 32'd1 >= dst_clear_total_rows_q) begin
            dst_clear_row_idx_q <= 32'd0;
            state <= ST_INIT_TILE;
          end else begin
            dst_clear_row_idx_q <= dst_clear_row_idx_q + 32'd1;
          end
        end

        ST_INIT_TILE: begin
          lane_q <= 6'd0;
          a_load_row_q <= 5'd0;
          state <= ST_A_LOAD_REQ;
        end

        ST_A_LOAD_REQ: begin
          state <= ST_A_LOAD_LATCH;
        end

        ST_A_LOAD_LATCH: begin
          for (int col_idx = 0; col_idx < SYS_DIM; col_idx = col_idx + 1)
            a_tile_scratch[a_load_row_q[3:0]][col_idx] <= sram_b_rdata[(col_idx * 8) +: 8];

          if (a_load_row_q == 5'd15) begin
            lane_q <= 6'd0;
            state <= ST_READ_REQ;
          end else begin
            a_load_row_q <= a_load_row_q + 5'd1;
            state <= ST_A_LOAD_REQ;
          end
        end

        ST_READ_REQ: begin
          state <= ST_READ_USE;
        end

        ST_READ_USE: begin
          if (int'(lane_q) == ((SYSTOLIC_ARCH_MODE == SYS_MODE_CHAINED) ? (CHAIN_TOTAL_STEPS - 1) : (SYS_DIM - 1))) begin
            lane_q <= 6'd0;
            if (ktile_q + 11'd1 < k_tiles_q) begin
              ktile_q <= ktile_q + 11'd1;
              a_load_row_q <= 5'd0;
              state <= ST_A_LOAD_REQ;
            end else begin
              drain_row_q <= 5'd0;
              drain_grp_q <= 2'd0;
              state <= ST_DRAIN_PREP;
            end
          end else begin
            lane_q <= lane_q + 6'd1;
            state <= ST_READ_REQ;
          end
        end

        ST_DRAIN_PREP: begin
          drain_row_addr_q <= tile_drain_base_q + ({5'h0, ntile_q} << 2);
          state <= ST_DRAIN_WR;
        end

        ST_DRAIN_WR: begin
          if (drain_grp_q == 2'd3) begin
            drain_grp_q <= 2'd0;
            if (drain_row_q == 5'd15) begin
              // Tile complete. Move to next n/m tile.
              ktile_q <= 11'd0;
              if (ntile_q + 11'd1 < n_tiles_q) begin
                ntile_q <= ntile_q + 11'd1;
                state <= ST_INIT_TILE;
              end else if (mtile_q + 11'd1 < m_tiles_q) begin
                mtile_q <= mtile_q + 11'd1;
                ntile_q <= 11'd0;
                tile_drain_base_q <= tile_drain_base_q + ({5'h0, n_tiles_q} << 6);
                state <= ST_INIT_TILE;
              end else begin
                state <= ST_IDLE;
              end
            end else begin
              drain_row_q <= drain_row_q + 5'd1;
              drain_row_addr_q <= drain_row_addr_q + ({5'h0, n_tiles_q} << 2);
            end
          end else begin
            drain_grp_q <= drain_grp_q + 2'd1;
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  // Drive SRAM and array control for the current state.
  always_comb begin
    sys_busy = (state != ST_IDLE);

    // During chained flush cycles, inject zeros so only in-flight operands
    // continue propagating through the PE mesh.
    inject_zero_data = (SYSTOLIC_ARCH_MODE == SYS_MODE_CHAINED)
                    && (int'(lane_q) >= SYS_DIM)
                    && ((state == ST_READ_REQ) || (state == ST_READ_USE));
    lane_row_idx = (SYSTOLIC_ARCH_MODE == SYS_MODE_CHAINED)
               ? {10'h0, lane_q[5:0]}
               : {10'h0, lane_q[5:0]};
    a_row_data_q = 128'h0;
    for (int row_idx = 0; row_idx < SYS_DIM; row_idx = row_idx + 1) begin
      if (!inject_zero_data && (lane_q < 6'd16))
        a_row_data_q[(row_idx * 8) +: 8] = a_tile_scratch[row_idx][lane_q[3:0]];
    end
    b_row_data_q = inject_zero_data ? 128'h0 : sram_a_rdata;

    sram_a_en = 1'b0;
    sram_a_we = 1'b0;
    sram_a_buf = src2_buf_q;
    sram_a_row = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en = 1'b0;
    sram_b_buf = src1_buf_q;
    sram_b_row = 16'h0;

    step_en = 1'b0;
    // Clear accumulator at tile start unless accumulate mode is requested.
    clear_acc = (state == ST_INIT_TILE) && !flags_accumulate_q;

    case (state)
      ST_A_LOAD_REQ: begin
        sram_b_en = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = src1_load_row_addr[15:0];
      end

      ST_READ_REQ: begin
        // Issue both source reads together. Their rows appear one cycle later
        // in ST_READ_USE, which then advances the systolic mesh by one step.
        if (inject_zero_data) begin
          sram_b_en = 1'b0;
          sram_a_en = 1'b0;
        end else begin
          sram_a_en = 1'b1;
          sram_a_we = 1'b0;
          sram_a_buf = src2_buf_q;
          sram_a_row = src2_stream_row_addr[15:0];
        end
      end

      ST_READ_USE: begin
        step_en = 1'b1;
      end

      ST_DRAIN_WR: begin
        // Pack four neighboring INT32 accumulators into one 128-bit SRAM row.
        sram_a_en = 1'b1;
        sram_a_we = 1'b1;
        sram_a_buf = dst_buf_q;
        sram_a_row = drain_row_addr_q + {14'h0, drain_grp_q};
        sram_a_wdata[31:0]   = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00});
        sram_a_wdata[63:32]  = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd1);
        sram_a_wdata[95:64]  = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd2);
        sram_a_wdata[127:96] = acc_at(drain_row_q, {1'b0, drain_grp_q, 2'b00} + 5'd3);
      end

      ST_DST_CLEAR_WR: begin
        // Non-accumulating MATMUL starts from a clean architectural
        // destination span. Clear the full padded ACCUM tile region once
        // before the first compute tile begins.
        sram_a_en = 1'b1;
        sram_a_we = 1'b1;
        sram_a_buf = dst_buf_q;
        sram_a_row = dst_off_q + dst_clear_row_idx_q[15:0];
        sram_a_wdata = 128'h0;
      end

      default: ;
    endcase
  end

endmodule

`endif // SYSTOLIC_CONTROLLER_SV
