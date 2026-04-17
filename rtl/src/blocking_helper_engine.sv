// Blocking helper engine for Phase C baseline helper instructions.
//
// Supported operations:
//   - BUF_COPY     : flat copy + distinct-buffer transpose
//   - VADD         : INT8 saturating add and ACCUM+WBUF bias add
//   - REQUANT      : ACCUM INT32 -> ABUF/WBUF INT8 via exact FP16 scale
//   - REQUANT_PC   : ACCUM INT32 -> ABUF/WBUF INT8 via per-column FP16 scales
//   - SCALE_MUL    : width-preserving tile-scale multiply for INT8 / INT32
//   - DEQUANT_ADD  : ACCUM + INT8 skip path fused back to INT8
//
// The helper engine is architecturally blocking. Control dispatches it through
// helper_dispatch and waits in S_DISP_WAIT until helper_busy drops.
//
// It owns both SRAM ports while active:
//   - port A for reads and writes
//   - port B for a second read stream when an operation needs two sources
//
// The implementation is intentionally microcoded as a small FSM rather than
// deeply pipelined. Phase C prioritizes correctness and clear SRAM sequencing
// over maximum overlap.

`ifndef BLOCKING_HELPER_ENGINE_SV
`define BLOCKING_HELPER_ENGINE_SV

`include "taccel_pkg.sv"

module blocking_helper_engine
  import taccel_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,

  input  logic         dispatch,
  input  logic [4:0]   opcode,
  input  logic [1:0]   src1_buf,
  input  logic [15:0]  src1_off,
  input  logic [1:0]   src2_buf,
  input  logic [15:0]  src2_off,
  input  logic [1:0]   dst_buf,
  input  logic [15:0]  dst_off,
  input  logic [3:0]   sreg,
  input  logic [15:0]  b_length,
  input  logic [5:0]   b_src_rows,
  input  logic         b_transpose,
  input  logic [9:0]   tile_m,
  input  logic [9:0]   tile_n,
  input  logic [15:0]  scale0_data,
  input  logic [15:0]  scale1_data,

  output logic         helper_busy,
  output logic         helper_fault,
  output logic [3:0]   helper_fault_code,

  output logic         sram_a_en,
  output logic         sram_a_we,
  output logic [1:0]   sram_a_buf,
  output logic [15:0]  sram_a_row,
  output logic [127:0] sram_a_wdata,
  input  logic [127:0] sram_a_rdata,
  input  logic         sram_a_fault,

  output logic         sram_b_en,
  output logic [1:0]   sram_b_buf,
  output logic [15:0]  sram_b_row,
  input  logic [127:0] sram_b_rdata,
  input  logic         sram_b_fault
);

  import "DPI-C" function real sfu_fp32_round(input real value_r);
  import "DPI-C" function real sfu_fp32_mul(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_add(input real lhs_r, input real rhs_r);
  import "DPI-C" function int  sfu_fp32_quantize_i8(input real value_r, input real out_scale_r);

  // State groups:
  //   H_FLAT_*   : BUF_COPY flat copy / memmove
  //   H_TSRC_*   : BUF_COPY transpose source gather
  //   H_TDST_*   : BUF_COPY transpose destination update
  //   H_V8_*     : INT8 residual VADD
  //   H_VB_*     : load and reuse one INT32 bias row chunk
  //   H_VACC_*   : INT32 ACCUM bias add
  //   H_RQ_*       : REQUANT four ACCUM rows into one INT8 row
  //   H_SM_*       : SCALE_MUL row-wise read / write
  //   H_RQPC_*     : REQUANT_PC scale-table load + ACCUM gather + write
  //   H_DQ_*       : DEQUANT_ADD skip read + ACCUM gather + write
  typedef enum logic [5:0] {
    H_IDLE            = 6'd0,
    H_FLAT_READ       = 6'd1,
    H_FLAT_WRITE      = 6'd2,
    H_TSRC_REQ1       = 6'd3,
    H_TSRC_CAP1       = 6'd4,
    H_TSRC_REQ2       = 6'd5,
    H_TSRC_CAP2       = 6'd6,
    H_TDST_REQ        = 6'd7,
    H_TDST_WRITE      = 6'd8,
    H_V8_READ         = 6'd9,
    H_V8_WRITE        = 6'd10,
    H_VB_REQ          = 6'd11,
    H_VB_LATCH        = 6'd12,
    H_VACC_READ       = 6'd13,
    H_VACC_WRITE      = 6'd14,
    H_RQ_REQ          = 6'd15,
    H_RQ_LATCH        = 6'd16,
    H_RQ_WRITE        = 6'd17,
    H_SM_REQ          = 6'd18,
    H_SM_WRITE        = 6'd19,
    H_RQPC_SCALE0_REQ = 6'd20,
    H_RQPC_SCALE0_LATCH = 6'd21,
    H_RQPC_SCALE1_REQ = 6'd22,
    H_RQPC_SCALE1_LATCH = 6'd23,
    H_RQPC_REQ        = 6'd24,
    H_RQPC_LATCH      = 6'd25,
    H_RQPC_WRITE      = 6'd26,
    H_DQ_SKIP_REQ     = 6'd27,
    H_DQ_SKIP_LATCH   = 6'd28,
    H_DQ_REQ          = 6'd29,
    H_DQ_LATCH        = 6'd30,
    H_DQ_WRITE        = 6'd31,
    H_FAULT           = 6'd32
  } helper_state_t;

  helper_state_t state;

  // Latched instruction parameters and loop counters.
  logic [4:0]   opcode_q;
  logic [1:0]   src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0]  src1_off_q, src2_off_q, dst_off_q;
  logic [3:0]   sreg_q;
  logic [15:0]  b_length_q;
  logic [5:0]   b_src_rows_q;
  logic         b_transpose_q;
  logic [15:0]  scale0_q, scale1_q;
  logic [14:0]  m_rows_q;
  logic [10:0]  n_tiles_q;
  logic [12:0]  n_chunks_i32_q;
  logic [3:0]   fault_code_r;

  logic [31:0]  step_idx_q;
  logic         flat_backward_q;

  logic [15:0]  trans_row_count_q;
  logic [15:0]  trans_cols_q;
  logic [15:0]  trans_rbase_q;
  logic [15:0]  trans_cbase_q;
  logic [4:0]   trans_height_q;
  logic [4:0]   trans_width_q;
  logic [4:0]   trans_src_row_idx_q;
  logic [4:0]   trans_dst_row_idx_q;
  logic [127:0] trans_first_row_q;
  logic [127:0] trans_scratch_q [0:15];

  logic [15:0]  bias_chunk_q;
  logic [14:0]  bias_row_idx_q;
  logic [127:0] bias_data_q;

  logic [14:0]  rq_row_idx_q;
  logic [10:0]  rq_col_chunk_q;
  logic [1:0]   rq_part_q;
  logic [127:0] rq_row0_q, rq_row1_q, rq_row2_q, rq_row3_q;
  logic [127:0] skip_row_q;
  logic [15:0]  pc_scale_chunk_q [0:15];

  // Small helpers used by multiple helper ops.
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

  function automatic logic [4:0] block_span(
    input logic [15:0] total,
    input logic [15:0] base
  );
    logic [15:0] rem;
    begin
      rem = total - base;
      if (rem > 16)
        block_span = 5'd16;
      else
        block_span = rem[4:0];
    end
  endfunction

  function automatic logic [7:0] get_byte(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_byte = row[(idx * 8) +: 8];
    end
  endfunction

  function automatic logic [15:0] get_u16(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_u16 = row[(idx * 16) +: 16];
    end
  endfunction

  // Saturating INT8 lane-wise add for residual connections.
  function automatic logic [127:0] sat_add_int8_row(
    input logic [127:0] a_row,
    input logic [127:0] b_row
  );
    logic signed [8:0] sum;
    logic signed [7:0] a_i8, b_i8;
    logic [127:0] out_row;
    integer i;
    begin
      out_row = 128'h0;
      for (i = 0; i < 16; i++) begin
        a_i8 = a_row[(i * 8) +: 8];
        b_i8 = b_row[(i * 8) +: 8];
        sum  = $signed(a_i8) + $signed(b_i8);
        if (sum > 9'sd127)
          out_row[(i * 8) +: 8] = 8'h7F;
        else if (sum < -9'sd128)
          out_row[(i * 8) +: 8] = 8'h80;
        else
          out_row[(i * 8) +: 8] = sum[7:0];
      end
      sat_add_int8_row = out_row;
    end
  endfunction

  // Wraparound INT32 lane-wise add to match numpy int32 bias-add behavior.
  function automatic logic [127:0] add_wrap_int32_row(
    input logic [127:0] a_row,
    input logic [127:0] b_row
  );
    logic signed [31:0] a_i32, b_i32, sum_i32;
    logic [127:0] out_row;
    integer i;
    begin
      out_row = 128'h0;
      for (i = 0; i < 4; i++) begin
        a_i32 = a_row[(i * 32) +: 32];
        b_i32 = b_row[(i * 32) +: 32];
        sum_i32 = $signed(a_i32) + $signed(b_i32);
        out_row[(i * 32) +: 32] = sum_i32;
      end
      add_wrap_int32_row = out_row;
    end
  endfunction

  // Exact INT32 x FP16 scaling with round-half-to-even on right shifts.
  // This is written as a reusable scalar primitive for later helper/SFU work.
  function automatic logic signed [63:0] fp16_mul_round_even(
    input logic signed [31:0] src_val,
    input logic [15:0]        fp16_val
  );
    logic        sign_h;
    logic [4:0]  exp_h;
    logic [9:0]  frac_h;
    logic signed [12:0] mant;
    integer      shift_amt;
    logic signed [63:0] prod;
    logic signed [63:0] abs_prod;
    logic signed [63:0] quot;
    logic signed [63:0] rem;
    logic signed [63:0] half;
    begin
      sign_h = fp16_val[15];
      exp_h  = fp16_val[14:10];
      frac_h = fp16_val[9:0];

      if ((exp_h == 5'h0) && (frac_h == 10'h0)) begin
        fp16_mul_round_even = 64'sd0;
      end else begin
        if (exp_h == 5'h0) begin
          mant      = $signed({3'b000, frac_h});
          shift_amt = -24;
        end else begin
          mant      = $signed({2'b00, 1'b1, frac_h});
          shift_amt = integer'(exp_h) - 25;
        end

        prod = $signed(src_val) * $signed(mant);
        if (sign_h)
          prod = -prod;

        if (shift_amt >= 0) begin
          fp16_mul_round_even = prod <<< shift_amt;
        end else begin
          abs_prod = (prod < 0) ? -prod : prod;
          quot     = abs_prod >>> (-shift_amt);
          rem      = abs_prod & ((64'sd1 <<< (-shift_amt)) - 64'sd1);
          half     = 64'sd1 <<< ((-shift_amt) - 1);
          if ((rem > half) || ((rem == half) && quot[0]))
            quot = quot + 64'sd1;
          fp16_mul_round_even = (prod < 0) ? -quot : quot;
        end
      end
    end
  endfunction

  function automatic real pow2_int(input integer exp_i);
    real v;
    integer j;
    begin
      v = 1.0;
      if (exp_i >= 0) begin
        for (j = 0; j < exp_i; j++)
          v = v * 2.0;
      end else begin
        for (j = 0; j < -exp_i; j++)
          v = v * 0.5;
      end
      pow2_int = v;
    end
  endfunction

  function automatic real fp16_to_real(input logic [15:0] bits);
    logic sign_bit;
    logic [4:0] exp_bits;
    logic [9:0] frac_bits;
    real sign_r;
    begin
      sign_bit = bits[15];
      exp_bits = bits[14:10];
      frac_bits = bits[9:0];
      sign_r = sign_bit ? -1.0 : 1.0;

      if ((exp_bits == 5'h0) && (frac_bits == 10'h0)) begin
        fp16_to_real = 0.0;
      end else if (exp_bits == 5'h0) begin
        fp16_to_real = sign_r * (real'(frac_bits) / 1024.0) * pow2_int(-14);
      end else if (exp_bits == 5'h1F) begin
        fp16_to_real = sign_r * 65504.0;
      end else begin
        fp16_to_real = sign_r *
                       (1.0 + (real'(frac_bits) / 1024.0)) *
                       pow2_int(integer'(exp_bits) - 15);
      end
      fp16_to_real = sfu_fp32_round(fp16_to_real);
    end
  endfunction

  function automatic integer round_half_even(input real value_r);
    integer floor_i;
    real frac_r;
    begin
      floor_i = integer'($floor(value_r));
      frac_r = value_r - real'(floor_i);
      if (frac_r > 0.5)
        round_half_even = floor_i + 1;
      else if (frac_r < 0.5)
        round_half_even = floor_i;
      else if (floor_i[0])
        round_half_even = floor_i + 1;
      else
        round_half_even = floor_i;
    end
  endfunction

  function automatic logic [127:0] scale_mul_i8_row(
    input logic [127:0] row,
    input logic [15:0]  scale_val
  );
    logic signed [7:0] src_i8;
    logic signed [63:0] scaled;
    logic [127:0] out_row;
    integer i;
    begin
      out_row = 128'h0;
      for (i = 0; i < 16; i++) begin
        src_i8 = row[(i * 8) +: 8];
        scaled = fp16_mul_round_even({{24{src_i8[7]}}, src_i8}, scale_val);
        if (scaled > 64'sd127)
          out_row[(i * 8) +: 8] = 8'h7F;
        else if (scaled < -64'sd128)
          out_row[(i * 8) +: 8] = 8'h80;
        else
          out_row[(i * 8) +: 8] = scaled[7:0];
      end
      scale_mul_i8_row = out_row;
    end
  endfunction

  function automatic logic [127:0] scale_mul_i32_row(
    input logic [127:0] row,
    input logic [15:0]  scale_val
  );
    logic signed [31:0] src_i32;
    logic signed [63:0] scaled;
    logic [127:0] out_row;
    integer i;
    begin
      out_row = 128'h0;
      for (i = 0; i < 4; i++) begin
        src_i32 = row[(i * 32) +: 32];
        scaled = fp16_mul_round_even(src_i32, scale_val);
        if (scaled > 64'sd2147483647)
          out_row[(i * 32) +: 32] = 32'h7FFF_FFFF;
        else if (scaled < -64'sd2147483648)
          out_row[(i * 32) +: 32] = 32'h8000_0000;
        else
          out_row[(i * 32) +: 32] = scaled[31:0];
      end
      scale_mul_i32_row = out_row;
    end
  endfunction

  function automatic logic [127:0] dequant_add_pack(
    input logic [127:0] row0,
    input logic [127:0] row1,
    input logic [127:0] row2,
    input logic [127:0] row3,
    input logic [127:0] skip_row,
    input logic [15:0]  scale0_val,
    input logic [15:0]  scale1_val
  );
    logic signed [31:0] src_i32;
    logic signed [7:0]  skip_i8;
    logic [127:0] out_row;
    real accum_scale_r;
    real skip_scale_r;
    real sum_r;
    integer q_i;
    integer idx;
    begin
      accum_scale_r = fp16_to_real(scale0_val);
      skip_scale_r  = fp16_to_real(scale1_val);
      out_row = 128'h0;
      for (idx = 0; idx < 16; idx++) begin
        case (idx[3:2])
          2'd0: src_i32 = row0[(idx[1:0] * 32) +: 32];
          2'd1: src_i32 = row1[(idx[1:0] * 32) +: 32];
          2'd2: src_i32 = row2[(idx[1:0] * 32) +: 32];
          default: src_i32 = row3[(idx[1:0] * 32) +: 32];
        endcase
        skip_i8 = skip_row[(idx * 8) +: 8];
        sum_r = sfu_fp32_add(
            sfu_fp32_mul(sfu_fp32_round(real'(src_i32)), accum_scale_r),
            sfu_fp32_mul(sfu_fp32_round(real'(skip_i8)), skip_scale_r)
        );
        q_i = integer'(sfu_fp32_quantize_i8(sum_r, 1.0));
        if (q_i > 127)
          out_row[(idx * 8) +: 8] = 8'h7F;
        else if (q_i < -128)
          out_row[(idx * 8) +: 8] = 8'h80;
        else
          out_row[(idx * 8) +: 8] = q_i[7:0];
      end
      dequant_add_pack = out_row;
    end
  endfunction

  // Pack four ACCUM rows (4 x 4 x INT32) into one INT8 destination row.
  function automatic logic [127:0] requant_pack(
    input logic [127:0] row0,
    input logic [127:0] row1,
    input logic [127:0] row2,
    input logic [127:0] row3,
    input logic [15:0]  scale_val
  );
    logic signed [31:0] src_i32;
    logic signed [63:0] scaled;
    logic [127:0] out_row;
    integer i;
    begin
      out_row = 128'h0;
      for (i = 0; i < 4; i++) begin
        src_i32 = row0[(i * 32) +: 32];
        scaled  = fp16_mul_round_even(src_i32, scale_val);
        if (scaled > 64'sd127)
          out_row[(i * 8) +: 8] = 8'h7F;
        else if (scaled < -64'sd128)
          out_row[(i * 8) +: 8] = 8'h80;
        else
          out_row[(i * 8) +: 8] = scaled[7:0];

        src_i32 = row1[(i * 32) +: 32];
        scaled  = fp16_mul_round_even(src_i32, scale_val);
        if (scaled > 64'sd127)
          out_row[((i + 4) * 8) +: 8] = 8'h7F;
        else if (scaled < -64'sd128)
          out_row[((i + 4) * 8) +: 8] = 8'h80;
        else
          out_row[((i + 4) * 8) +: 8] = scaled[7:0];

        src_i32 = row2[(i * 32) +: 32];
        scaled  = fp16_mul_round_even(src_i32, scale_val);
        if (scaled > 64'sd127)
          out_row[((i + 8) * 8) +: 8] = 8'h7F;
        else if (scaled < -64'sd128)
          out_row[((i + 8) * 8) +: 8] = 8'h80;
        else
          out_row[((i + 8) * 8) +: 8] = scaled[7:0];

        src_i32 = row3[(i * 32) +: 32];
        scaled  = fp16_mul_round_even(src_i32, scale_val);
        if (scaled > 64'sd127)
          out_row[((i + 12) * 8) +: 8] = 8'h7F;
        else if (scaled < -64'sd128)
          out_row[((i + 12) * 8) +: 8] = 8'h80;
        else
          out_row[((i + 12) * 8) +: 8] = scaled[7:0];
      end
      requant_pack = out_row;
    end
  endfunction

  // Transpose path helper: extract up to 16 contiguous bytes starting at an
  // arbitrary byte position, spanning two source rows when needed.
  function automatic logic [127:0] extract_window(
    input logic [127:0] row0,
    input logic [127:0] row1,
    input logic [3:0]   start_byte,
    input logic [4:0]   width
  );
    logic [127:0] out_row;
    integer i;
    integer src_idx;
    begin
      out_row = 128'h0;
      for (i = 0; i < 16; i++) begin
        if (i < integer'(width)) begin
          src_idx = integer'(start_byte) + i;
          if (src_idx < 16)
            out_row[(i * 8) +: 8] = get_byte(row0, src_idx);
          else
            out_row[(i * 8) +: 8] = get_byte(row1, src_idx - 16);
        end
      end
      extract_window = out_row;
    end
  endfunction

  // Geometry derived from CONFIG_TILE and the dispatched helper instruction.
  logic [14:0] dispatch_m_rows_w;
  logic [10:0] dispatch_n_tiles_w;
  logic [12:0] dispatch_n_chunks_i32_w;
  logic [31:0] dispatch_int8_units_w;
  logic [31:0] dispatch_int32_units_w;
  logic [31:0] dispatch_scale_rows_w;
  logic [31:0] dispatch_copy_units_w;
  logic [15:0] dispatch_src_rows_w;
  logic [15:0] dispatch_trans_cols_w;
  logic [15:0] dispatch_src_buf_rows_w;
  logic [15:0] dispatch_src2_buf_rows_w;
  logic [15:0] dispatch_dst_buf_rows_w;
  logic        dispatch_unsupported_w;
  logic        dispatch_sram_oob_w;
  logic        dispatch_is_vadd_int8_w;
  logic        dispatch_is_vadd_bias_w;
  logic        dispatch_is_scale_mul_int8_w;
  logic        dispatch_is_scale_mul_int32_w;
  logic        dispatch_is_requant_pc_w;
  logic        dispatch_is_dequant_add_w;
  logic        dispatch_same_buf_overlap_w;

  logic [15:0] flat_src_row_w;
  logic [15:0] flat_dst_row_w;

  logic [31:0] trans_src_byte_addr_w;
  logic [15:0] trans_src_row0_w;
  logic [15:0] trans_src_row1_w;
  logic [3:0]  trans_src_lane_w;
  logic        trans_need_row1_w;
  logic [15:0] trans_dst_row_w;
  logic [127:0] trans_dst_data_w;
  logic [127:0] trans_dst_merge_w;

  logic [15:0] v8_src1_row_w;
  logic [15:0] v8_src2_row_w;
  logic [15:0] v8_dst_row_w;

  logic [15:0] vbias_row_w;
  logic [15:0] vacc_row_w;

  logic [15:0] rq_src_row_w;
  logic [15:0] rq_dst_row_w;
  logic [15:0] rqpc_scale_row_w;
  logic [15:0] dq_skip_row_w;

  logic [127:0] trans_col_data_w;
  logic [127:0] trans_partial_row_w;
  logic [31:0] rq_src_row_full_w;
  logic [31:0] rq_dst_row_full_w;
  logic [127:0] rqpc_write_data_w;
  logic [127:0] dq_write_data_w;
  logic [127:0] scale_mul_write_data_w;

  assign dispatch_m_rows_w      = ({5'h0, tile_m} + 15'd1) << 4;
  assign dispatch_n_tiles_w     = {1'b0, tile_n} + 11'd1;
  assign dispatch_n_chunks_i32_w = dispatch_n_tiles_w << 2;
  assign dispatch_int8_units_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
  assign dispatch_int32_units_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
  assign dispatch_scale_rows_w  = {20'h0, dispatch_n_tiles_w, 1'b0};
  assign dispatch_copy_units_w  = {16'h0, b_length};
  assign dispatch_src_rows_w    = {6'h0, b_src_rows, 4'h0};
  assign dispatch_trans_cols_w  = (b_src_rows == 6'h0) ? 16'h0 : (b_length / {10'h0, b_src_rows});
  assign dispatch_src_buf_rows_w = buf_rows(src1_buf);
  assign dispatch_src2_buf_rows_w = buf_rows(src2_buf);
  assign dispatch_dst_buf_rows_w = buf_rows(dst_buf);
  assign dispatch_is_vadd_int8_w = (src1_buf == BUF_ABUF) &&
                                   ((src2_buf == BUF_ABUF) || (src2_buf == BUF_WBUF)) &&
                                   (dst_buf == BUF_ABUF);
  assign dispatch_is_vadd_bias_w = (src1_buf == BUF_ACCUM) &&
                                   (src2_buf == BUF_WBUF) &&
                                   (dst_buf == BUF_ACCUM);
  assign dispatch_is_scale_mul_int8_w = (src1_buf != BUF_ACCUM) && (dst_buf != BUF_ACCUM);
  assign dispatch_is_scale_mul_int32_w = (src1_buf == BUF_ACCUM) && (dst_buf == BUF_ACCUM);
  assign dispatch_is_requant_pc_w = (src1_buf == BUF_ACCUM) &&
                                    (src2_buf != BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_is_dequant_add_w = (src1_buf == BUF_ACCUM) &&
                                     (src2_buf != BUF_ACCUM) &&
                                     (dst_buf != BUF_ACCUM);
  assign dispatch_same_buf_overlap_w =
      (src1_buf == dst_buf) &&
      ({16'h0, src1_off} < ({16'h0, dst_off} + dispatch_copy_units_w)) &&
      ({16'h0, dst_off} < ({16'h0, src1_off} + dispatch_copy_units_w));

  // Dispatch-time validation.
  // Reject unsupported legal mode combinations before touching SRAM, and also
  // reject whole-operation OOB ranges up front.
  always_comb begin
    dispatch_unsupported_w = 1'b0;
    dispatch_sram_oob_w    = 1'b0;

    case (opcode)
      OP_BUF_COPY: begin
        dispatch_sram_oob_w =
            ({1'b0, src1_off} + {1'b0, b_length} > {1'b0, dispatch_src_buf_rows_w}) ||
            ({1'b0, dst_off}  + {1'b0, b_length} > {1'b0, dispatch_dst_buf_rows_w});

        if (b_transpose) begin
          if ((b_length != 16'h0) &&
              ((b_src_rows == 6'h0) || ((b_length % {10'h0, b_src_rows}) != 16'h0) || (src1_buf == dst_buf)))
            dispatch_unsupported_w = 1'b1;
        end
      end

      OP_VADD: begin
        if (dispatch_is_vadd_int8_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int8_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, src2_off} + dispatch_int8_units_w > {16'h0, dispatch_src2_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int8_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else if (dispatch_is_vadd_bias_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, src2_off} + {19'h0, dispatch_n_chunks_i32_w} > {16'h0, dispatch_src2_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int32_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else begin
          dispatch_unsupported_w = 1'b1;
        end
      end

      OP_REQUANT: begin
        if ((src1_buf != BUF_ACCUM) || (dst_buf == BUF_ACCUM))
          dispatch_unsupported_w = 1'b1;
        else
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int8_units_w  > {16'h0, dispatch_dst_buf_rows_w});
      end

      OP_REQUANT_PC: begin
        if (!dispatch_is_requant_pc_w)
          dispatch_unsupported_w = 1'b1;
        else
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, src2_off} + dispatch_scale_rows_w > {16'h0, dispatch_src2_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int8_units_w > {16'h0, dispatch_dst_buf_rows_w});
      end

      OP_SCALE_MUL: begin
        if (dispatch_is_scale_mul_int32_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int32_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else if (dispatch_is_scale_mul_int8_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int8_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int8_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else begin
          dispatch_unsupported_w = 1'b1;
        end
      end

      OP_DEQUANT_ADD: begin
        if (!dispatch_is_dequant_add_w || (sreg == 4'hF))
          dispatch_unsupported_w = 1'b1;
        else
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_int32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, src2_off} + dispatch_int8_units_w > {16'h0, dispatch_src2_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_int8_units_w > {16'h0, dispatch_dst_buf_rows_w});
      end

      default:
        dispatch_unsupported_w = 1'b1;
    endcase
  end

  assign flat_src_row_w =
      flat_backward_q ? (src1_off_q + b_length_q - 16'(step_idx_q) - 16'h1)
                      : (src1_off_q + 16'(step_idx_q));
  assign flat_dst_row_w =
      flat_backward_q ? (dst_off_q + b_length_q - 16'(step_idx_q) - 16'h1)
                      : (dst_off_q + 16'(step_idx_q));

  assign trans_src_byte_addr_w =
      ({16'h0, src1_off_q} << 4) +
      (({16'h0, trans_rbase_q} + {27'h0, trans_src_row_idx_q}) * {16'h0, trans_cols_q}) +
      {16'h0, trans_cbase_q};
  assign trans_src_row0_w  = trans_src_byte_addr_w[19:4];
  assign trans_src_row1_w  = trans_src_row0_w + 16'h1;
  assign trans_src_lane_w  = trans_src_byte_addr_w[3:0];
  assign trans_need_row1_w = ({1'b0, trans_src_lane_w} + {1'b0, trans_width_q} > 6'd16);
  assign trans_dst_row_w =
      dst_off_q +
      ((trans_cbase_q + {11'h0, trans_dst_row_idx_q}) * {10'h0, b_src_rows_q}) +
      (trans_rbase_q >> 4);

  assign v8_src1_row_w = src1_off_q + 16'(step_idx_q);
  assign v8_src2_row_w = src2_off_q + 16'(step_idx_q);
  assign v8_dst_row_w  = dst_off_q + 16'(step_idx_q);

  assign vbias_row_w = src2_off_q + bias_chunk_q;
  assign vacc_row_w  = src1_off_q + (bias_row_idx_q * n_chunks_i32_q) + bias_chunk_q;

  assign rq_src_row_full_w = {16'h0, src1_off_q} +
                             ({17'h0, rq_row_idx_q} * {19'h0, n_chunks_i32_q}) +
                             ({19'h0, rq_col_chunk_q} << 2) +
                             {30'h0, rq_part_q};
  assign rq_dst_row_full_w = {16'h0, dst_off_q} +
                             ({17'h0, rq_row_idx_q} * {21'h0, n_tiles_q}) +
                             {21'h0, rq_col_chunk_q};
  assign rq_src_row_w = rq_src_row_full_w[15:0];
  assign rq_dst_row_w = rq_dst_row_full_w[15:0];
  assign rqpc_scale_row_w = src2_off_q + ({4'h0, rq_col_chunk_q} << 1) + {15'h0, rq_part_q[0]};
  assign dq_skip_row_w = src2_off_q + (rq_row_idx_q * n_tiles_q) + {5'h0, rq_col_chunk_q};

  // Assemble one transposed destination column from the scratch tile and merge
  // it with a partially covered destination row when the transpose edge is not
  // a full 16-byte row.
  always_comb begin
    trans_col_data_w = 128'h0;
    for (int j = 0; j < 16; j++) begin
      if (j < integer'(trans_height_q))
        trans_col_data_w[(j * 8) +: 8] =
            trans_scratch_q[j[3:0]][(integer'(trans_dst_row_idx_q) * 8) +: 8];
    end

    trans_partial_row_w = sram_a_rdata;
    for (int j = 0; j < 16; j++) begin
      if (j < integer'(trans_height_q))
        trans_partial_row_w[(j * 8) +: 8] = trans_col_data_w[(j * 8) +: 8];
    end
  end

  assign trans_dst_data_w  = trans_col_data_w;
  assign trans_dst_merge_w = trans_partial_row_w;

  always_comb begin
    rqpc_write_data_w = 128'h0;
    dq_write_data_w = 128'h0;
    scale_mul_write_data_w = 128'h0;

    for (int lane = 0; lane < 16; lane++) begin
      logic signed [31:0] src_i32;
      logic signed [63:0] scaled;
      case (lane[3:2])
        2'd0: src_i32 = rq_row0_q[(lane[1:0] * 32) +: 32];
        2'd1: src_i32 = rq_row1_q[(lane[1:0] * 32) +: 32];
        2'd2: src_i32 = rq_row2_q[(lane[1:0] * 32) +: 32];
        default: src_i32 = rq_row3_q[(lane[1:0] * 32) +: 32];
      endcase
      scaled = fp16_mul_round_even(src_i32, pc_scale_chunk_q[lane]);
      if (scaled > 64'sd127)
        rqpc_write_data_w[(lane * 8) +: 8] = 8'h7F;
      else if (scaled < -64'sd128)
        rqpc_write_data_w[(lane * 8) +: 8] = 8'h80;
      else
        rqpc_write_data_w[(lane * 8) +: 8] = scaled[7:0];
    end

    dq_write_data_w = dequant_add_pack(rq_row0_q, rq_row1_q, rq_row2_q, rq_row3_q,
                                       skip_row_q, scale0_q, scale1_q);

    if (src1_buf_q == BUF_ACCUM)
      scale_mul_write_data_w = scale_mul_i32_row(sram_b_rdata, scale0_q);
    else
      scale_mul_write_data_w = scale_mul_i8_row(sram_b_rdata, scale0_q);
  end

  // Main helper FSM. SRAM is synchronous, so read states issue a request and
  // the following state consumes the returned row.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state             <= H_IDLE;
      opcode_q          <= 5'h0;
      src1_buf_q        <= 2'b0;
      src2_buf_q        <= 2'b0;
      dst_buf_q         <= 2'b0;
      src1_off_q        <= 16'h0;
      src2_off_q        <= 16'h0;
      dst_off_q         <= 16'h0;
      sreg_q            <= 4'h0;
      b_length_q        <= 16'h0;
      b_src_rows_q      <= 6'h0;
      b_transpose_q     <= 1'b0;
      scale0_q          <= 16'h0;
      scale1_q          <= 16'h0;
      m_rows_q          <= 15'h0;
      n_tiles_q         <= 11'h0;
      n_chunks_i32_q    <= 13'h0;
      fault_code_r      <= 4'(FAULT_NONE);
      step_idx_q        <= 32'h0;
      flat_backward_q   <= 1'b0;
      trans_row_count_q <= 16'h0;
      trans_cols_q      <= 16'h0;
      trans_rbase_q     <= 16'h0;
      trans_cbase_q     <= 16'h0;
      trans_height_q    <= 5'h0;
      trans_width_q     <= 5'h0;
      trans_src_row_idx_q <= 5'h0;
      trans_dst_row_idx_q <= 5'h0;
      trans_first_row_q <= 128'h0;
      bias_chunk_q      <= 16'h0;
      bias_row_idx_q    <= 15'h0;
      bias_data_q       <= 128'h0;
      rq_row_idx_q      <= 15'h0;
      rq_col_chunk_q    <= 11'h0;
      rq_part_q         <= 2'h0;
      rq_row0_q         <= 128'h0;
      rq_row1_q         <= 128'h0;
      rq_row2_q         <= 128'h0;
      rq_row3_q         <= 128'h0;
      skip_row_q        <= 128'h0;
      for (int i = 0; i < 16; i++)
        pc_scale_chunk_q[i] <= 16'h0;
      for (int j = 0; j < 16; j++)
        trans_scratch_q[j] <= 128'h0;
    end else begin
      case (state)
        H_IDLE: begin
          if (dispatch) begin
            opcode_q       <= opcode;
            src1_buf_q     <= src1_buf;
            src2_buf_q     <= src2_buf;
            dst_buf_q      <= dst_buf;
            src1_off_q     <= src1_off;
            src2_off_q     <= src2_off;
            dst_off_q      <= dst_off;
            sreg_q         <= sreg;
            b_length_q     <= b_length;
            b_src_rows_q   <= b_src_rows;
            b_transpose_q  <= b_transpose;
            scale0_q       <= scale0_data;
            scale1_q       <= scale1_data;
            m_rows_q       <= dispatch_m_rows_w;
            n_tiles_q      <= dispatch_n_tiles_w;
            n_chunks_i32_q <= dispatch_n_chunks_i32_w;

            if (dispatch_unsupported_w) begin
              fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
              state        <= H_FAULT;
            end else if (dispatch_sram_oob_w) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= H_FAULT;
            end else begin
              case (opcode)
                OP_BUF_COPY: begin
                  if (b_length == 16'h0) begin
                    state <= H_IDLE;
                  end else if (b_transpose) begin
                    trans_row_count_q   <= dispatch_src_rows_w;
                    trans_cols_q        <= dispatch_trans_cols_w;
                    trans_rbase_q       <= 16'h0;
                    trans_cbase_q       <= 16'h0;
                    trans_height_q      <= block_span(dispatch_src_rows_w, 16'h0);
                    trans_width_q       <= block_span(dispatch_trans_cols_w, 16'h0);
                    trans_src_row_idx_q <= 5'h0;
                    trans_dst_row_idx_q <= 5'h0;
                    state               <= H_TSRC_REQ1;
                  end else begin
                    step_idx_q      <= 32'h0;
                    flat_backward_q <= dispatch_same_buf_overlap_w && (dst_off > src1_off);
                    state           <= H_FLAT_READ;
                  end
                end

                OP_VADD: begin
                  if (dispatch_is_vadd_int8_w) begin
                    step_idx_q <= 32'h0;
                    state      <= H_V8_READ;
                  end else begin
                    bias_chunk_q   <= 16'h0;
                    bias_row_idx_q <= 15'h0;
                    state          <= H_VB_REQ;
                  end
                end

                OP_REQUANT: begin
                  rq_row_idx_q   <= 15'h0;
                  rq_col_chunk_q <= 11'h0;
                  rq_part_q      <= 2'h0;
                  state          <= H_RQ_REQ;
                end

                OP_REQUANT_PC: begin
                  rq_row_idx_q   <= 15'h0;
                  rq_col_chunk_q <= 11'h0;
                  rq_part_q      <= 2'h0;
                  state          <= H_RQPC_SCALE0_REQ;
                end

                OP_SCALE_MUL: begin
                  step_idx_q <= 32'h0;
                  state      <= H_SM_REQ;
                end

                OP_DEQUANT_ADD: begin
                  rq_row_idx_q   <= 15'h0;
                  rq_col_chunk_q <= 11'h0;
                  rq_part_q      <= 2'h0;
                  state          <= H_DQ_SKIP_REQ;
                end

                default: begin
                  fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
                  state        <= H_FAULT;
                end
              endcase
            end
          end
        end

        H_FLAT_READ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_FLAT_WRITE;
          end
        end

        H_FLAT_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= {16'h0, b_length_q}) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            state      <= H_FLAT_READ;
          end
        end

        H_TSRC_REQ1: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_TSRC_CAP1;
          end
        end

        H_TSRC_CAP1: begin
          // Gather one source row slice into the 16x16 transpose scratch tile.
          if (trans_need_row1_w) begin
            trans_first_row_q <= sram_a_rdata;
            state             <= H_TSRC_REQ2;
          end else begin
            trans_scratch_q[trans_src_row_idx_q[3:0]] <= extract_window(
                sram_a_rdata, 128'h0, trans_src_lane_w, trans_width_q);
            if (trans_src_row_idx_q + 5'd1 >= trans_height_q) begin
              trans_dst_row_idx_q <= 5'h0;
              if (trans_height_q == 5'd16)
                state <= H_TDST_WRITE;
              else
                state <= H_TDST_REQ;
            end else begin
              trans_src_row_idx_q <= trans_src_row_idx_q + 5'd1;
              state               <= H_TSRC_REQ1;
            end
          end
        end

        H_TSRC_REQ2: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_TSRC_CAP2;
          end
        end

        H_TSRC_CAP2: begin
          trans_scratch_q[trans_src_row_idx_q[3:0]] <= extract_window(
              trans_first_row_q, sram_a_rdata, trans_src_lane_w, trans_width_q);
          if (trans_src_row_idx_q + 5'd1 >= trans_height_q) begin
            trans_dst_row_idx_q <= 5'h0;
            if (trans_height_q == 5'd16)
              state <= H_TDST_WRITE;
            else
              state <= H_TDST_REQ;
          end else begin
            trans_src_row_idx_q <= trans_src_row_idx_q + 5'd1;
            state               <= H_TSRC_REQ1;
          end
        end

        H_TDST_REQ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_TDST_WRITE;
          end
        end

        H_TDST_WRITE: begin
          // Write one destination row of the transposed tile. Partial rows use
          // read-modify-write through H_TDST_REQ.
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (trans_dst_row_idx_q + 5'd1 < trans_width_q) begin
            trans_dst_row_idx_q <= trans_dst_row_idx_q + 5'd1;
            if (trans_height_q == 5'd16)
              state <= H_TDST_WRITE;
            else
              state <= H_TDST_REQ;
          end else if ({16'h0, trans_cbase_q} + {27'h0, trans_width_q} < {16'h0, trans_cols_q}) begin
            trans_cbase_q       <= trans_cbase_q + 16'd16;
            trans_width_q       <= block_span(trans_cols_q, trans_cbase_q + 16'd16);
            trans_src_row_idx_q <= 5'h0;
            state               <= H_TSRC_REQ1;
          end else if ({16'h0, trans_rbase_q} + {27'h0, trans_height_q} < {16'h0, trans_row_count_q}) begin
            trans_rbase_q       <= trans_rbase_q + 16'd16;
            trans_cbase_q       <= 16'h0;
            trans_height_q      <= block_span(trans_row_count_q, trans_rbase_q + 16'd16);
            trans_width_q       <= block_span(trans_cols_q, 16'h0);
            trans_src_row_idx_q <= 5'h0;
            state               <= H_TSRC_REQ1;
          end else begin
            state <= H_IDLE;
          end
        end

        H_V8_READ: begin
          if (sram_a_fault || sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_V8_WRITE;
          end
        end

        H_V8_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= (m_rows_q * n_tiles_q)) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            state      <= H_V8_READ;
          end
        end

        H_VB_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_VB_LATCH;
          end
        end

        H_VB_LATCH: begin
          // Reuse the same 1xN bias chunk across every output row before
          // advancing to the next chunk.
          bias_data_q    <= sram_b_rdata;
          bias_row_idx_q <= 15'h0;
          state          <= H_VACC_READ;
        end

        H_VACC_READ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_VACC_WRITE;
          end
        end

        H_VACC_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (bias_row_idx_q + 15'd1 < m_rows_q) begin
            bias_row_idx_q <= bias_row_idx_q + 15'd1;
            state          <= H_VACC_READ;
          end else if ({16'h0, bias_chunk_q} + 32'd1 < {19'h0, n_chunks_i32_q}) begin
            bias_chunk_q <= bias_chunk_q + 16'd1;
            state       <= H_VB_REQ;
          end else begin
            state <= H_IDLE;
          end
        end

        H_RQ_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_RQ_LATCH;
          end
        end

        H_RQ_LATCH: begin
          // Four ACCUM rows are collected before one packed INT8 row is written.
          case (rq_part_q)
            2'd0: rq_row0_q <= sram_b_rdata;
            2'd1: rq_row1_q <= sram_b_rdata;
            2'd2: rq_row2_q <= sram_b_rdata;
            default: rq_row3_q <= sram_b_rdata;
          endcase

          if (rq_part_q == 2'd3) begin
            state <= H_RQ_WRITE;
          end else begin
            rq_part_q <= rq_part_q + 2'd1;
            state     <= H_RQ_REQ;
          end
        end

        H_RQ_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (rq_col_chunk_q + 11'd1 < n_tiles_q) begin
            rq_col_chunk_q <= rq_col_chunk_q + 11'd1;
            rq_part_q      <= 2'd0;
            state          <= H_RQ_REQ;
          end else if (rq_row_idx_q + 15'd1 < m_rows_q) begin
            rq_row_idx_q   <= rq_row_idx_q + 15'd1;
            rq_col_chunk_q <= 11'd0;
            rq_part_q      <= 2'd0;
            state          <= H_RQ_REQ;
          end else begin
            state <= H_IDLE;
          end
        end

        H_SM_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_SM_WRITE;
          end
        end

        H_SM_WRITE: begin
          logic [31:0] total_rows_w;
          total_rows_w = (src1_buf_q == BUF_ACCUM) ? (m_rows_q * n_chunks_i32_q)
                                                   : (m_rows_q * n_tiles_q);
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= total_rows_w) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            state      <= H_SM_REQ;
          end
        end

        H_RQPC_SCALE0_REQ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_RQPC_SCALE0_LATCH;
          end
        end

        H_RQPC_SCALE0_LATCH: begin
          for (int lane = 0; lane < 8; lane++)
            pc_scale_chunk_q[lane] <= get_u16(sram_a_rdata, lane);
          rq_part_q <= 2'd1;
          state     <= H_RQPC_SCALE1_REQ;
        end

        H_RQPC_SCALE1_REQ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_RQPC_SCALE1_LATCH;
          end
        end

        H_RQPC_SCALE1_LATCH: begin
          for (int lane = 0; lane < 8; lane++)
            pc_scale_chunk_q[lane + 8] <= get_u16(sram_a_rdata, lane);
          rq_row_idx_q <= 15'h0;
          rq_part_q    <= 2'd0;
          state        <= H_RQPC_REQ;
        end

        H_RQPC_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_RQPC_LATCH;
          end
        end

        H_RQPC_LATCH: begin
          case (rq_part_q)
            2'd0: rq_row0_q <= sram_b_rdata;
            2'd1: rq_row1_q <= sram_b_rdata;
            2'd2: rq_row2_q <= sram_b_rdata;
            default: rq_row3_q <= sram_b_rdata;
          endcase

          if (rq_part_q == 2'd3) begin
            state <= H_RQPC_WRITE;
          end else begin
            rq_part_q <= rq_part_q + 2'd1;
            state     <= H_RQPC_REQ;
          end
        end

        H_RQPC_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (rq_row_idx_q + 15'd1 < m_rows_q) begin
            rq_row_idx_q <= rq_row_idx_q + 15'd1;
            rq_part_q    <= 2'd0;
            state        <= H_RQPC_REQ;
          end else if (rq_col_chunk_q + 11'd1 < n_tiles_q) begin
            rq_col_chunk_q <= rq_col_chunk_q + 11'd1;
            rq_part_q      <= 2'd0;
            state          <= H_RQPC_SCALE0_REQ;
          end else begin
            state <= H_IDLE;
          end
        end

        H_DQ_SKIP_REQ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_DQ_SKIP_LATCH;
          end
        end

        H_DQ_SKIP_LATCH: begin
          skip_row_q <= sram_a_rdata;
          rq_part_q  <= 2'd0;
          state      <= H_DQ_REQ;
        end

        H_DQ_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_DQ_LATCH;
          end
        end

        H_DQ_LATCH: begin
          case (rq_part_q)
            2'd0: rq_row0_q <= sram_b_rdata;
            2'd1: rq_row1_q <= sram_b_rdata;
            2'd2: rq_row2_q <= sram_b_rdata;
            default: rq_row3_q <= sram_b_rdata;
          endcase

          if (rq_part_q == 2'd3) begin
            state <= H_DQ_WRITE;
          end else begin
            rq_part_q <= rq_part_q + 2'd1;
            state     <= H_DQ_REQ;
          end
        end

        H_DQ_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (rq_col_chunk_q + 11'd1 < n_tiles_q) begin
            rq_col_chunk_q <= rq_col_chunk_q + 11'd1;
            state          <= H_DQ_SKIP_REQ;
          end else if (rq_row_idx_q + 15'd1 < m_rows_q) begin
            rq_row_idx_q   <= rq_row_idx_q + 15'd1;
            rq_col_chunk_q <= 11'd0;
            state          <= H_DQ_SKIP_REQ;
          end else begin
            state <= H_IDLE;
          end
        end

        H_FAULT: ;

        default:
          state <= H_IDLE;
      endcase
    end
  end

  // State-to-SRAM decode.
  // Port A handles all writes and most reads; port B is only used for the
  // second source stream in VADD / REQUANT / bias load.
  always_comb begin
    helper_busy       = (state != H_IDLE) && (state != H_FAULT);
    helper_fault      = (state == H_FAULT);
    helper_fault_code = fault_code_r;

    sram_a_en    = 1'b0;
    sram_a_we    = 1'b0;
    sram_a_buf   = src1_buf_q;
    sram_a_row   = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en    = 1'b0;
    sram_b_buf   = src1_buf_q;
    sram_b_row   = 16'h0;

    case (state)
      H_FLAT_READ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = flat_src_row_w;
      end

      H_FLAT_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = flat_dst_row_w;
        sram_a_wdata = sram_a_rdata;
      end

      H_TSRC_REQ1: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = trans_src_row0_w;
      end

      H_TSRC_REQ2: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = trans_src_row1_w;
      end

      H_TDST_REQ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = dst_buf_q;
        sram_a_row = trans_dst_row_w;
      end

      H_TDST_WRITE: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b1;
        sram_a_buf = dst_buf_q;
        sram_a_row = trans_dst_row_w;
        if (trans_height_q == 5'd16)
          sram_a_wdata = trans_dst_data_w;
        else
          sram_a_wdata = trans_dst_merge_w;
      end

      H_V8_READ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src2_buf_q;
        sram_a_row = v8_src2_row_w;
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = v8_src1_row_w;
      end

      H_V8_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = v8_dst_row_w;
        sram_a_wdata = sat_add_int8_row(sram_b_rdata, sram_a_rdata);
      end

      H_VB_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        sram_b_row = vbias_row_w;
      end

      H_VACC_READ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = vacc_row_w;
      end

      H_VACC_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = dst_off_q + (bias_row_idx_q * n_chunks_i32_q) + bias_chunk_q;
        sram_a_wdata = add_wrap_int32_row(sram_a_rdata, bias_data_q);
      end

      H_RQ_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = rq_src_row_w;
      end

      H_RQ_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = rq_dst_row_w;
        sram_a_wdata = requant_pack(rq_row0_q, rq_row1_q, rq_row2_q, rq_row3_q, scale0_q);
      end

      H_SM_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = src1_off_q + 16'(step_idx_q);
      end

      H_SM_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = dst_off_q + 16'(step_idx_q);
        sram_a_wdata = scale_mul_write_data_w;
      end

      H_RQPC_SCALE0_REQ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src2_buf_q;
        sram_a_row = rqpc_scale_row_w;
      end

      H_RQPC_SCALE1_REQ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src2_buf_q;
        sram_a_row = rqpc_scale_row_w;
      end

      H_RQPC_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = rq_src_row_w;
      end

      H_RQPC_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = rq_dst_row_w;
        sram_a_wdata = rqpc_write_data_w;
      end

      H_DQ_SKIP_REQ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src2_buf_q;
        sram_a_row = dq_skip_row_w;
      end

      H_DQ_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = rq_src_row_w;
      end

      H_DQ_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = rq_dst_row_w;
        sram_a_wdata = dq_write_data_w;
      end

      default: ;
    endcase
  end

endmodule

`endif // BLOCKING_HELPER_ENGINE_SV
