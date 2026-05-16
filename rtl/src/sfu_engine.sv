// Special-function-unit engine for Stage D numerical parity.
//
// Supported operations:
//   - SOFTMAX   : INT8/INT32 input, INT8 output, row-wise across full logical N
//   - LAYERNORM : ABUF INT8 input, WBUF FP16 gamma/beta, INT8 output
//   - GELU      : ABUF INT8 or ACCUM INT32 input, INT8 output
//   - SOFTMAX_ATTNV : fused softmax(QK^T) @ V with INT8 output
//   - MASKED_SOFTMAX / MASKED_SOFTMAX_ATTNV : attention-mask variants
//
// Architectural contract:
//   - dispatched asynchronously through sfu_dispatch / sfu_busy
//   - serialized against DMA / helper / systolic at control level in Stage D
//   - faults propagate asynchronously through sfu_fault / sfu_fault_code
//
// Implementation note:
//   Stage D prioritizes functional parity with the software golden model over
//   synthesis-oriented microarchitecture. The engine therefore uses real-valued
//   intermediate storage plus explicit FP32 rounding helpers to preserve the
//   architectural "all SFU internal operations use FP32" contract under the
//   current simulator, which does not model shortreal arithmetic directly.

`ifndef SFU_ENGINE_SV
`define SFU_ENGINE_SV

`include "taccel_pkg.sv"

module sfu_engine
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
  input  logic [9:0]   tile_m,
  input  logic [9:0]   tile_n,
  input  logic [9:0]   tile_k,
  input  logic         attn_valid,
  input  logic [11:0]  attn_query_row_base,
  input  logic [11:0]  attn_valid_kv_len,
  input  logic [1:0]   attn_mode,
  input  logic [15:0]  scale0_data,
  input  logic [15:0]  scale1_data,
  input  logic [15:0]  scale2_data,
  input  logic [15:0]  scale3_data,

  output logic         sfu_busy,
  output logic         sfu_fault,
  output logic [3:0]   sfu_fault_code,

  output logic         sram_a_en,
  output logic         sram_a_we,
  output logic [1:0]   sram_a_buf,
  output logic [15:0]  sram_a_row,
  output logic [127:0] sram_a_wdata,
  input  logic         sram_a_fault,

  output logic         sram_b_en,
  output logic [1:0]   sram_b_buf,
  output logic [15:0]  sram_b_row,
  input  logic [127:0] sram_b_rdata,
  input  logic         sram_b_fault
);

  import "DPI-C" function real sfu_fp32_round(input real value_r);
  import "DPI-C" function real sfu_fp32_add(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_sub(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_mul(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_div(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_exp(input real value_r);
  import "DPI-C" function real sfu_fp32_sqrt(input real value_r);
  import "DPI-C" function real sfu_fp32_gelu(input real value_r);
  import "DPI-C" function int sfu_fp32_quantize_i8(input real value_r, input real out_scale_r);
  // gen-2 FP32 opcodes (frozen ISA): exact IEEE-754 half<->fp32 (numpy
  // float16 semantics) + tanh gelu_new. NOT the gen-1 erf sfu_fp32_gelu.
  import "DPI-C" function real sfu_fp16_bits_to_fp32(input int bits);
  import "DPI-C" function int  sfu_fp32_to_fp16_bits(input real value_r);
  import "DPI-C" function real sfu_fp32_gelu_new(input real value_r);

  localparam int SFU_MAX_ROW_ELEMS = 1024;
  localparam real LN_EPS = 1.0e-6;
  // gen-2 LAYERNORM_FP32 (0x1A) eps — the GPT-2 / golden value (1e-5).
  // Distinct from the gen-1 INT8 LN_EPS (1e-6) above; do NOT reuse it.
  localparam real LN_FP32_EPS = 1.0e-5;

  typedef enum logic [4:0] {
    F_IDLE          = 5'd0,
    F_LN_PARAM_REQ  = 5'd1,
    F_LN_PARAM_LATCH= 5'd2,
    F_ROW_I8_REQ    = 5'd3,
    F_ROW_I8_LATCH  = 5'd4,
    F_ROW_I32_REQ   = 5'd5,
    F_ROW_I32_LATCH = 5'd6,
    F_ROW_COMPUTE   = 5'd7,
    F_ROW_PACK      = 5'd8,
    F_ROW_WRITE     = 5'd9,
    F_GELU_I8_REQ   = 5'd10,
    F_GELU_I8_LATCH = 5'd11,
    F_GELU_I8_WRITE = 5'd12,
    F_GELU_I32_REQ  = 5'd13,
    F_GELU_I32_LATCH= 5'd14,
    F_GELU_I32_WRITE= 5'd15,
    F_ATTN_QKT_REQ  = 5'd16,
    F_ATTN_QKT_LATCH= 5'd17,
    F_ATTN_PREP     = 5'd18,
    F_ATTN_V_REQ    = 5'd19,
    F_ATTN_V_LATCH  = 5'd20,
    F_ATTN_WRITE    = 5'd21,
    F_FAULT         = 5'd22,
    // gen-2 FP32 shared datapath (0x19 VADD / 0x1A LN / 0x1B GELU).
    // FP16 storage (8 elems / 16-byte row), FP32 internal.
    F_G2_S1_REQ     = 5'd23,
    F_G2_S1_LATCH   = 5'd24,
    F_G2_S2_REQ     = 5'd25,
    F_G2_S2_LATCH   = 5'd26,
    F_G2_COMPUTE    = 5'd27,
    F_G2_PACK       = 5'd28,
    F_G2_WRITE      = 5'd29
  } sfu_state_t;

  sfu_state_t state;

  logic [4:0]   opcode_q;
  logic [1:0]   src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0]  src1_off_q, src2_off_q, dst_off_q;
  logic [3:0]   sreg_q;
  logic [14:0]  m_rows_q;
  logic [10:0]  n_tiles_q;
  logic [10:0]  k_tiles_q;
  logic [12:0]  n_chunks_i32_q;
  logic [12:0]  k_chunks_i32_q;
  logic [15:0]  n_elems_q;
  logic [15:0]  k_elems_q;
  logic [15:0]  ln_gamma_rows_q;
  logic [15:0]  ln_param_rows_q;
  logic         attn_valid_q;
  logic [11:0] attn_query_row_base_q;
  logic [11:0] attn_valid_kv_len_q;
  logic [1:0]  attn_mode_q;
  logic [3:0]   fault_code_r;

  logic [14:0]  row_idx_q;
  logic [12:0]  read_idx_q;
  logic [10:0]  write_chunk_q;
  logic [1:0]   gelu_part_q;
  logic [15:0]  attn_k_idx_q;

  logic [127:0] gelu_i8_row_q;
  logic [127:0] gelu_row0_q, gelu_row1_q, gelu_row2_q, gelu_row3_q;

  real scale0_q /* verilator public_flat_rd */, scale1_q /* verilator public_flat_rd */,
       scale2_q /* verilator public_flat_rd */, scale3_q /* verilator public_flat_rd */;
  real row_data_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  real attn_accum_q [0:SFU_MAX_ROW_ELEMS-1];
  real gamma_q    [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  real beta_q     [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [7:0] out_bytes_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  // gen-2: FP16 result bit-patterns + FP16-rows-per-logical-row (=2*n_tiles).
  logic [15:0] out_h_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [12:0] g2_rows_q;
  real attn_row_max_q;
  real attn_exp_sum_q;
  real ln_debug_mean_q /* verilator public_flat_rd */;
  real ln_debug_var_q /* verilator public_flat_rd */;
  real ln_debug_denom_q /* verilator public_flat_rd */;
  real ln_debug_y_q [0:15] /* verilator public_flat_rd */;

  logic [14:0] dispatch_m_rows_w;
  logic [10:0] dispatch_n_tiles_w;
  logic [10:0] dispatch_k_tiles_w;
  logic [12:0] dispatch_n_chunks_i32_w;
  logic [12:0] dispatch_k_chunks_i32_w;
  logic [15:0] dispatch_n_elems_w;
  logic [15:0] dispatch_k_elems_w;
  logic [15:0] dispatch_ln_gamma_rows_w;
  logic [15:0] dispatch_ln_param_rows_w;
  logic [15:0] dispatch_src1_rows_w;
  logic [15:0] dispatch_src2_rows_w;
  logic [15:0] dispatch_dst_rows_w;
  logic        dispatch_softmax_accum_w;
  logic        dispatch_softmax_int8_w;
  logic        dispatch_layernorm_w;
  logic        dispatch_gelu_accum_w;
  logic        dispatch_gelu_int8_w;
  logic        dispatch_softmax_attnv_w;
  logic        dispatch_masked_softmax_w;
  logic        dispatch_masked_softmax_attnv_w;
  logic        dispatch_attn_context_bad_w;
  logic        dispatch_unsupported_w;
  logic        dispatch_sram_oob_w;

  logic [31:0] dispatch_src1_need_rows_w;
  logic [31:0] dispatch_src2_need_rows_w;
  logic [31:0] dispatch_dst_need_rows_w;
  logic [15:0] dispatch_attn_key_cols_w;

  logic [31:0] row_i8_addr_w;
  logic [31:0] row_i32_addr_w;
  logic [31:0] row_dst_addr_w;
  logic [31:0] ln_param_addr_w;
  logic [31:0] gelu_i8_addr_w;
  logic [31:0] gelu_acc_addr_w;
  logic [31:0] gelu_dst_addr_w;
  logic [31:0] attn_qkt_addr_w;
  logic [31:0] attn_v_addr_w;

  logic [127:0] row_write_data_w;
  logic [127:0] row_write_q;
  logic [127:0] gelu_i8_write_data_w;
  logic [127:0] gelu_i32_write_data_w;
  logic [127:0] attn_write_data_w;
  logic [127:0] g2_write_data_w;

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

  function automatic logic signed [7:0] get_i8(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_i8 = row[(idx * 8) +: 8];
    end
  endfunction

  function automatic logic signed [31:0] get_i32(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_i32 = row[(idx * 32) +: 32];
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

  function automatic logic [7:0] quantize_to_i8(
    input real value_r,
    input real out_scale_r
  );
    int q_i;
    begin
      if (out_scale_r == 0.0) begin
        quantize_to_i8 = 8'h00;
      end else begin
        q_i = sfu_fp32_quantize_i8(value_r, out_scale_r);
        if (q_i > 127)
          quantize_to_i8 = 8'h7F;
        else if (q_i < -128)
          quantize_to_i8 = 8'h80;
        else
          quantize_to_i8 = q_i[7:0];
      end
    end
  endfunction

  function automatic real gelu_real(input real x_r);
    begin
      gelu_real = sfu_fp32_gelu(x_r);
    end
  endfunction

  function automatic logic attn_visible(
    input logic [14:0] row_idx,
    input integer      col_idx
  );
    integer abs_query_row;
    begin
      abs_query_row = integer'(attn_query_row_base_q) + integer'(row_idx);
      attn_visible = 1'b1;
      if (attn_mode_q[1])
        attn_visible = attn_visible && (col_idx <= abs_query_row);
      if (attn_mode_q[0])
        attn_visible = attn_visible && (col_idx < integer'(attn_valid_kv_len_q));
    end
  endfunction

  assign dispatch_m_rows_w        = ({5'h0, tile_m} + 15'd1) << 4;
  assign dispatch_n_tiles_w       = {1'b0, tile_n} + 11'd1;
  assign dispatch_k_tiles_w       = {1'b0, tile_k} + 11'd1;
  assign dispatch_n_chunks_i32_w  = dispatch_n_tiles_w << 2;
  assign dispatch_k_chunks_i32_w  = dispatch_k_tiles_w << 2;
  assign dispatch_n_elems_w       = {1'b0, dispatch_n_tiles_w, 4'h0};
  assign dispatch_k_elems_w       = {1'b0, dispatch_k_tiles_w, 4'h0};
  assign dispatch_ln_gamma_rows_w = ({5'h0, dispatch_n_tiles_w}) << 1;
  assign dispatch_ln_param_rows_w = ({5'h0, dispatch_n_tiles_w}) << 2;
  assign dispatch_src1_rows_w     = buf_rows(src1_buf);
  assign dispatch_src2_rows_w     = buf_rows(src2_buf);
  assign dispatch_dst_rows_w      = buf_rows(dst_buf);

  assign dispatch_softmax_accum_w = ((opcode == OP_SOFTMAX) ||
                                     (opcode == OP_MASKED_SOFTMAX)) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_softmax_int8_w  = ((opcode == OP_SOFTMAX) ||
                                     (opcode == OP_MASKED_SOFTMAX)) &&
                                    (src1_buf != BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_layernorm_w     = (opcode == OP_LAYERNORM) &&
                                    (src1_buf == BUF_ABUF) &&
                                    (src2_buf == BUF_WBUF) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_gelu_accum_w    = (opcode == OP_GELU) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_gelu_int8_w     = (opcode == OP_GELU) &&
                                    (src1_buf == BUF_ABUF) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_softmax_attnv_w = (opcode == OP_SOFTMAX_ATTNV) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (src2_buf != BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_masked_softmax_w = (opcode == OP_MASKED_SOFTMAX);
  assign dispatch_masked_softmax_attnv_w = (opcode == OP_MASKED_SOFTMAX_ATTNV) &&
                                           (src1_buf == BUF_ACCUM) &&
                                           (src2_buf != BUF_ACCUM) &&
                                           (dst_buf != BUF_ACCUM);
  assign dispatch_attn_key_cols_w = (opcode == OP_MASKED_SOFTMAX_ATTNV) ?
                                    dispatch_k_elems_w : dispatch_n_elems_w;

  // gen-2 FP32 shared datapath detection (FP16 storage, ABUF I/O).
  logic        dispatch_g2_vadd_w;
  logic        dispatch_g2_ln_w;
  logic        dispatch_g2_gelu_w;
  logic [12:0] dispatch_g2_rows_w;   // FP16 rows per logical row = 2*n_tiles
  assign dispatch_g2_rows_w = {1'b0, dispatch_n_tiles_w} + {1'b0, dispatch_n_tiles_w};
  assign dispatch_g2_vadd_w = (opcode == OP_VADD_FP32) &&
                              (src1_buf == BUF_ABUF) &&
                              (src2_buf == BUF_ABUF) &&
                              (dst_buf  == BUF_ABUF);
  assign dispatch_g2_ln_w   = (opcode == OP_LAYERNORM_FP32) &&
                              (src1_buf == BUF_ABUF) &&
                              (src2_buf == BUF_WBUF) &&
                              (dst_buf  == BUF_ABUF);
  assign dispatch_g2_gelu_w = (opcode == OP_GELU_FP32) &&
                              (src1_buf == BUF_ABUF) &&
                              (dst_buf  == BUF_ABUF);
  logic dispatch_g2_dq_w;   // 0x17 DEQUANT_ACCUM_FP32
  logic dispatch_g2_q_w;    // 0x18 QUANT_FP32_INT8
  assign dispatch_g2_dq_w = (opcode == OP_DEQUANT_ACCUM_FP32) &&
                            (src1_buf == BUF_ACCUM) &&
                            (src2_buf == BUF_WBUF) &&
                            (dst_buf  == BUF_ABUF);
  assign dispatch_g2_q_w  = (opcode == OP_QUANT_FP32_INT8) &&
                            (src1_buf == BUF_ABUF) &&
                            (dst_buf  == BUF_ABUF);

  always_comb begin
    dispatch_attn_context_bad_w = 1'b0;
    if ((opcode == OP_MASKED_SOFTMAX) ||
        (opcode == OP_MASKED_SOFTMAX_ATTNV)) begin
      dispatch_attn_context_bad_w = !attn_valid ||
                                    (attn_mode == 2'b00) ||
                                    (attn_valid_kv_len == 12'h000);
      if (!dispatch_attn_context_bad_w) begin
        if (attn_mode == 2'b10)
          dispatch_attn_context_bad_w =
              ({4'h0, dispatch_attn_key_cols_w} != {8'h00, attn_valid_kv_len});
        else if (attn_mode[0])
          dispatch_attn_context_bad_w =
              ({4'h0, dispatch_attn_key_cols_w} < {8'h00, attn_valid_kv_len});
      end
    end
  end

  always_comb begin
    dispatch_unsupported_w = 1'b0;
    dispatch_sram_oob_w    = 1'b0;
    dispatch_src1_need_rows_w = 32'd0;
    dispatch_src2_need_rows_w = 32'd0;
    dispatch_dst_need_rows_w  = 32'd0;

    case (opcode)
      OP_SOFTMAX, OP_MASKED_SOFTMAX: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!(dispatch_softmax_accum_w || dispatch_softmax_int8_w))
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;

        if (dispatch_softmax_accum_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        else if (dispatch_softmax_int8_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_LAYERNORM: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!dispatch_layernorm_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;

        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_GELU: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!(dispatch_gelu_accum_w || dispatch_gelu_int8_w))
          dispatch_unsupported_w = 1'b1;

        if (dispatch_gelu_accum_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        else if (dispatch_gelu_int8_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_SOFTMAX_ATTNV: begin
        if (sreg > 4'd12)
          dispatch_unsupported_w = 1'b1;
        if (!dispatch_softmax_attnv_w)
          dispatch_unsupported_w = 1'b1;
        if ((integer'(dispatch_k_elems_w) > SFU_MAX_ROW_ELEMS) ||
            (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS))
          dispatch_unsupported_w = 1'b1;

        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_k_chunks_i32_w;
        dispatch_src2_need_rows_w = dispatch_k_elems_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_MASKED_SOFTMAX_ATTNV: begin
        if (sreg > 4'd12)
          dispatch_unsupported_w = 1'b1;
        if (!dispatch_masked_softmax_attnv_w)
          dispatch_unsupported_w = 1'b1;
        if ((integer'(dispatch_k_elems_w) > SFU_MAX_ROW_ELEMS) ||
            (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS))
          dispatch_unsupported_w = 1'b1;

        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_k_chunks_i32_w;
        dispatch_src2_need_rows_w = dispatch_k_elems_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_VADD_FP32: begin
        if (!dispatch_g2_vadd_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_src2_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_LAYERNORM_FP32: begin
        if (!dispatch_g2_ln_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_GELU_FP32: begin
        if (!dispatch_g2_gelu_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_DEQUANT_ACCUM_FP32: begin
        if (!dispatch_g2_dq_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        dispatch_src2_need_rows_w = {19'h0, dispatch_g2_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_QUANT_FP32_INT8: begin
        if (!dispatch_g2_q_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      default:
        dispatch_unsupported_w = 1'b1;
    endcase

    dispatch_sram_oob_w =
        ({16'h0, src1_off} + dispatch_src1_need_rows_w > {16'h0, dispatch_src1_rows_w}) ||
        ({16'h0, src2_off} + dispatch_src2_need_rows_w > {16'h0, dispatch_src2_rows_w}) ||
        ({16'h0, dst_off}  + dispatch_dst_need_rows_w  > {16'h0, dispatch_dst_rows_w});
  end

  assign row_i8_addr_w  = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {19'h0, read_idx_q};
  assign row_i32_addr_w = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                          {19'h0, read_idx_q};
  assign row_dst_addr_w = {16'h0, dst_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {21'h0, write_chunk_q};
  assign ln_param_addr_w = {16'h0, src2_off_q} + {19'h0, read_idx_q};
  assign gelu_i8_addr_w = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {21'h0, write_chunk_q};
  assign gelu_acc_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                           ({21'h0, write_chunk_q} << 2) +
                           {30'h0, gelu_part_q};
  assign gelu_dst_addr_w = {16'h0, dst_off_q} +
                           ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                           {21'h0, write_chunk_q};
  assign attn_qkt_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, k_chunks_i32_q}) +
                           {19'h0, read_idx_q};
  assign attn_v_addr_w = {16'h0, src2_off_q} +
                         ({16'h0, attn_k_idx_q} * {21'h0, n_tiles_q}) +
                         {19'h0, read_idx_q};

  // gen-2 FP16-tile addressing (8 elems / 16-byte row, g2_rows_q per row).
  logic [31:0] g2_s1_addr_w;
  logic [31:0] g2_s2_addr_w;
  logic [31:0] g2_lnp_addr_w;
  logic [31:0] g2_dst_addr_w;
  assign g2_s1_addr_w  = {16'h0, src1_off_q} +
                         ({17'h0, row_idx_q} * {19'h0, g2_rows_q}) +
                         {19'h0, read_idx_q};
  assign g2_s2_addr_w  = {16'h0, src2_off_q} +
                         ({17'h0, row_idx_q} * {19'h0, g2_rows_q}) +
                         {19'h0, read_idx_q};
  assign g2_lnp_addr_w = {16'h0, src2_off_q} + {19'h0, read_idx_q};
  assign g2_dst_addr_w = {16'h0, dst_off_q} +
                         ({17'h0, row_idx_q} * {19'h0, g2_rows_q}) +
                         {21'h0, write_chunk_q};

  always_comb begin
    row_write_data_w = 128'h0;
    gelu_i8_write_data_w = 128'h0;
    gelu_i32_write_data_w = 128'h0;
    attn_write_data_w = 128'h0;
    g2_write_data_w = 128'h0;

    // gen-2: pack 8 FP16 results (16-bit each) for the current write chunk.
    for (int g2l = 0; g2l < 8; g2l++) begin
      int g2idx;
      g2idx = integer'(write_chunk_q) * 8 + g2l;
      if (g2idx < integer'(n_elems_q))
        g2_write_data_w[(g2l * 16) +: 16] = out_h_q[g2idx];
    end

    for (int lane = 0; lane < 16; lane++) begin
      int idx;
      real x_r;
      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        row_write_data_w[(lane * 8) +: 8] = out_bytes_q[idx];

      x_r = sfu_fp32_mul(real'(get_i8(gelu_i8_row_q, lane)), scale0_q);
      gelu_i8_write_data_w[(lane * 8) +: 8] = quantize_to_i8(gelu_real(x_r), scale1_q);

      if (lane < 4) begin
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row0_q, lane)), scale0_q);
        gelu_i32_write_data_w[(lane * 8) +: 8] = quantize_to_i8(gelu_real(x_r), scale1_q);
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row1_q, lane)), scale0_q);
        gelu_i32_write_data_w[((lane + 4) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), scale1_q);
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row2_q, lane)), scale0_q);
        gelu_i32_write_data_w[((lane + 8) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), scale1_q);
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row3_q, lane)), scale0_q);
        gelu_i32_write_data_w[((lane + 12) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), scale1_q);
      end

      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        attn_write_data_w[(lane * 8) +: 8] = quantize_to_i8(attn_accum_q[idx], scale2_q);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= F_IDLE;
      opcode_q       <= 5'h0;
      src1_buf_q     <= 2'b0;
      src2_buf_q     <= 2'b0;
      dst_buf_q      <= 2'b0;
      src1_off_q     <= 16'h0;
      src2_off_q     <= 16'h0;
      dst_off_q      <= 16'h0;
      sreg_q         <= 4'h0;
      m_rows_q       <= 15'h0;
      n_tiles_q      <= 11'h0;
      k_tiles_q      <= 11'h0;
      n_chunks_i32_q <= 13'h0;
      k_chunks_i32_q <= 13'h0;
      n_elems_q      <= 16'h0;
      k_elems_q      <= 16'h0;
      ln_gamma_rows_q<= 16'h0;
      ln_param_rows_q<= 16'h0;
      attn_valid_q   <= 1'b0;
      attn_query_row_base_q <= 12'h0;
      attn_valid_kv_len_q   <= 12'h0;
      attn_mode_q           <= 2'b00;
      fault_code_r   <= 4'(FAULT_NONE);
      row_idx_q      <= 15'h0;
      read_idx_q     <= 13'h0;
      write_chunk_q  <= 11'h0;
      gelu_part_q    <= 2'h0;
      attn_k_idx_q   <= 16'h0;
      g2_rows_q      <= 13'h0;
      gelu_i8_row_q  <= 128'h0;
      gelu_row0_q    <= 128'h0;
      gelu_row1_q    <= 128'h0;
      gelu_row2_q    <= 128'h0;
      gelu_row3_q    <= 128'h0;
      row_write_q    <= 128'h0;
      scale0_q       <= 0.0;
      scale1_q       <= 0.0;
      scale2_q       <= 0.0;
      scale3_q       <= 0.0;
      attn_row_max_q <= 0.0;
      attn_exp_sum_q <= 0.0;
      ln_debug_mean_q <= 0.0;
      ln_debug_var_q <= 0.0;
      ln_debug_denom_q <= 0.0;
      for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
        row_data_q[i] <= 0.0;
        attn_accum_q[i] <= 0.0;
        gamma_q[i]    <= 0.0;
        beta_q[i]     <= 0.0;
        out_bytes_q[i] <= 8'h00;
        out_h_q[i]    <= 16'h0;
      end
      for (int i = 0; i < 16; i++)
        ln_debug_y_q[i] <= 0.0;
    end else begin
      case (state)
        F_IDLE: begin
          if (dispatch) begin
            opcode_q        <= opcode;
            src1_buf_q      <= src1_buf;
            src2_buf_q      <= src2_buf;
            dst_buf_q       <= dst_buf;
            src1_off_q      <= src1_off;
            src2_off_q      <= src2_off;
            dst_off_q       <= dst_off;
            sreg_q          <= sreg;
            m_rows_q        <= dispatch_m_rows_w;
            n_tiles_q       <= dispatch_n_tiles_w;
            k_tiles_q       <= dispatch_k_tiles_w;
            n_chunks_i32_q  <= dispatch_n_chunks_i32_w;
            k_chunks_i32_q  <= dispatch_k_chunks_i32_w;
            n_elems_q       <= dispatch_n_elems_w;
            k_elems_q       <= dispatch_k_elems_w;
            ln_gamma_rows_q <= dispatch_ln_gamma_rows_w;
            ln_param_rows_q <= dispatch_ln_param_rows_w;
            g2_rows_q       <= dispatch_g2_rows_w;
            attn_valid_q    <= attn_valid;
            attn_query_row_base_q <= attn_query_row_base;
            attn_valid_kv_len_q   <= attn_valid_kv_len;
            attn_mode_q           <= attn_mode;
            scale0_q        <= fp16_to_real(scale0_data);
            scale1_q        <= fp16_to_real(scale1_data);
            scale2_q        <= fp16_to_real(scale2_data);
            scale3_q        <= fp16_to_real(scale3_data);
            ln_debug_mean_q <= 0.0;
            ln_debug_var_q <= 0.0;
            ln_debug_denom_q <= 0.0;
            for (int i = 0; i < 16; i++)
              ln_debug_y_q[i] <= 0.0;
            row_idx_q       <= 15'h0;
            read_idx_q      <= 13'h0;
            write_chunk_q   <= 11'h0;
            gelu_part_q     <= 2'h0;
            attn_k_idx_q    <= 16'h0;

            if (dispatch_unsupported_w) begin
              fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
              state        <= F_FAULT;
            end else if (dispatch_attn_context_bad_w) begin
              fault_code_r <= 4'(FAULT_NO_CONFIG);
              state        <= F_FAULT;
            end else if (dispatch_sram_oob_w) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= F_FAULT;
            end else begin
              case (opcode)
                OP_SOFTMAX, OP_MASKED_SOFTMAX: begin
                  if (src1_buf == BUF_ACCUM)
                    state <= F_ROW_I32_REQ;
                  else
                    state <= F_ROW_I8_REQ;
                end

                OP_LAYERNORM:
                  state <= F_LN_PARAM_REQ;

                OP_GELU: begin
                  if (src1_buf == BUF_ACCUM)
                    state <= F_GELU_I32_REQ;
                  else
                    state <= F_GELU_I8_REQ;
                end

                OP_SOFTMAX_ATTNV, OP_MASKED_SOFTMAX_ATTNV:
                  state <= F_ATTN_QKT_REQ;

                OP_VADD_FP32, OP_LAYERNORM_FP32, OP_GELU_FP32,
                OP_DEQUANT_ACCUM_FP32, OP_QUANT_FP32_INT8:
                  state <= F_G2_S1_REQ;

                default: begin
                  fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
                  state        <= F_FAULT;
                end
              endcase
            end
          end
        end

        F_LN_PARAM_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_LN_PARAM_LATCH;
          end
        end

        F_LN_PARAM_LATCH: begin
          integer base_idx;
          base_idx = (integer'(read_idx_q) < integer'(ln_gamma_rows_q)) ?
                     (integer'(read_idx_q) * 8) :
                     ((integer'(read_idx_q) - integer'(ln_gamma_rows_q)) * 8);
          for (int lane = 0; lane < 8; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q)) begin
              if (integer'(read_idx_q) < integer'(ln_gamma_rows_q))
                gamma_q[base_idx + lane] <= fp16_to_real(get_u16(sram_b_rdata, lane));
              else
                beta_q[base_idx + lane] <= fp16_to_real(get_u16(sram_b_rdata, lane));
            end
          end

          if ((integer'(read_idx_q) + 1) < integer'(ln_param_rows_q)) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_LN_PARAM_REQ;
          end else begin
            read_idx_q <= 13'h0;
            state      <= F_ROW_I8_REQ;
          end
        end

        F_ROW_I8_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ROW_I8_LATCH;
          end
        end

        F_ROW_I8_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 16;
          for (int lane = 0; lane < 16; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  sfu_fp32_mul(real'(get_i8(sram_b_rdata, lane)), scale0_q);
          end

          if (read_idx_q + 13'd1 < {2'h0, n_tiles_q}) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_I8_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ROW_COMPUTE;
          end
        end

        F_ROW_I32_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ROW_I32_LATCH;
          end
        end

        F_ROW_I32_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 4;
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  sfu_fp32_mul(real'(get_i32(sram_b_rdata, lane)), scale0_q);
          end

          if (read_idx_q + 13'd1 < n_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_I32_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ROW_COMPUTE;
          end
        end

        F_ROW_COMPUTE: begin
          if ((opcode_q == OP_SOFTMAX) || (opcode_q == OP_MASKED_SOFTMAX)) begin
            real row_max_r;
            real exp_sum_r;
            real exp_r;
            logic have_visible;
            have_visible = 1'b0;
            row_max_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if ((i < integer'(n_elems_q)) &&
                  ((opcode_q == OP_SOFTMAX) || attn_visible(row_idx_q, i))) begin
                if (!have_visible || (row_data_q[i] > row_max_r))
                  row_max_r = row_data_q[i];
                have_visible = 1'b1;
              end
            end

            if (!have_visible) begin
              fault_code_r <= 4'(FAULT_NO_CONFIG);
              state        <= F_FAULT;
            end else begin
              exp_sum_r = 0.0;
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if ((i < integer'(n_elems_q)) &&
                    ((opcode_q == OP_SOFTMAX) || attn_visible(row_idx_q, i))) begin
                  exp_r = sfu_fp32_exp(sfu_fp32_sub(row_data_q[i], row_max_r));
                  exp_sum_r = sfu_fp32_add(exp_sum_r, exp_r);
                end
              end

              if (exp_sum_r == 0.0) begin
                fault_code_r <= 4'(FAULT_NO_CONFIG);
                state        <= F_FAULT;
              end else begin
                for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                  if (i < integer'(n_elems_q)) begin
                    if ((opcode_q == OP_MASKED_SOFTMAX) && !attn_visible(row_idx_q, i)) begin
                      out_bytes_q[i] <= 8'h00;
                    end else begin
                      exp_r = sfu_fp32_exp(sfu_fp32_sub(row_data_q[i], row_max_r));
                      out_bytes_q[i] <= quantize_to_i8(sfu_fp32_div(exp_r, exp_sum_r), scale1_q);
                    end
                  end
                end
                state <= F_ROW_PACK;
              end
            end
          end else begin
            real sum_r;
            real mean_r;
            real var_r;
            real denom_r;
            sum_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                sum_r = sfu_fp32_add(sum_r, row_data_q[i]);
            end
            mean_r = sfu_fp32_div(sum_r, real'(n_elems_q));

            var_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                real diff_r;
                diff_r = sfu_fp32_sub(row_data_q[i], mean_r);
                var_r = sfu_fp32_add(var_r, sfu_fp32_mul(diff_r, diff_r));
              end
            end
            var_r = sfu_fp32_div(var_r, real'(n_elems_q));
            denom_r = sfu_fp32_sqrt(sfu_fp32_add(var_r, LN_EPS));
            ln_debug_mean_q <= mean_r;
            ln_debug_var_q <= var_r;
            ln_debug_denom_q <= denom_r;

            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              real y_r;
              if (i < integer'(n_elems_q)) begin
                y_r = sfu_fp32_add(
                    sfu_fp32_mul(
                        sfu_fp32_div(sfu_fp32_sub(row_data_q[i], mean_r), denom_r),
                        gamma_q[i]),
                    beta_q[i]);
                out_bytes_q[i] <= quantize_to_i8(y_r, scale1_q);
                if (i < 16)
                  ln_debug_y_q[i] <= y_r;
              end else if (i < 16) begin
                ln_debug_y_q[i] <= 0.0;
              end
            end
            state <= F_ROW_PACK;
          end
        end

        F_ROW_PACK: begin
          row_write_q <= row_write_data_w;
          state <= F_ROW_WRITE;
        end

        F_ROW_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_ROW_PACK;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            if (opcode_q == OP_QUANT_FP32_INT8)
              state <= F_G2_S1_REQ;          // gen-2 0x18 next-row FP16 read
            else if (((opcode_q == OP_SOFTMAX) || (opcode_q == OP_MASKED_SOFTMAX)) &&
                (src1_buf_q == BUF_ACCUM))
              state <= F_ROW_I32_REQ;
            else
              state <= F_ROW_I8_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_GELU_I8_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_GELU_I8_LATCH;
          end
        end

        F_GELU_I8_LATCH: begin
          gelu_i8_row_q <= sram_b_rdata;
          state         <= F_GELU_I8_WRITE;
        end

        F_GELU_I8_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_GELU_I8_REQ;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            write_chunk_q <= 11'h0;
            state         <= F_GELU_I8_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_GELU_I32_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_GELU_I32_LATCH;
          end
        end

        F_GELU_I32_LATCH: begin
          case (gelu_part_q)
            2'd0: gelu_row0_q <= sram_b_rdata;
            2'd1: gelu_row1_q <= sram_b_rdata;
            2'd2: gelu_row2_q <= sram_b_rdata;
            default: gelu_row3_q <= sram_b_rdata;
          endcase

          if (gelu_part_q == 2'd3) begin
            state <= F_GELU_I32_WRITE;
          end else begin
            gelu_part_q <= gelu_part_q + 2'd1;
            state       <= F_GELU_I32_REQ;
          end
        end

        F_GELU_I32_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            gelu_part_q   <= 2'h0;
            state         <= F_GELU_I32_REQ;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            write_chunk_q <= 11'h0;
            gelu_part_q   <= 2'h0;
            state         <= F_GELU_I32_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_ATTN_QKT_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ATTN_QKT_LATCH;
          end
        end

        F_ATTN_QKT_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 4;
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(k_elems_q))
              row_data_q[base_idx + lane] <=
                  sfu_fp32_mul(real'(get_i32(sram_b_rdata, lane)), scale0_q);
          end

          if (read_idx_q + 13'd1 < k_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ATTN_QKT_REQ;
          end else begin
            state <= F_ATTN_PREP;
          end
        end

        F_ATTN_PREP: begin
          real row_max_r;
          real exp_sum_r;
          logic have_visible;
          have_visible = 1'b0;
          row_max_r = 0.0;
          for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
            if ((i < integer'(k_elems_q)) &&
                ((opcode_q == OP_SOFTMAX_ATTNV) || attn_visible(row_idx_q, i))) begin
              if (!have_visible || (row_data_q[i] > row_max_r))
                row_max_r = row_data_q[i];
              have_visible = 1'b1;
            end
          end

          exp_sum_r = 0.0;
          for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
            if ((i < integer'(k_elems_q)) &&
                ((opcode_q == OP_SOFTMAX_ATTNV) || attn_visible(row_idx_q, i)))
              exp_sum_r = sfu_fp32_add(
                  exp_sum_r, sfu_fp32_exp(sfu_fp32_sub(row_data_q[i], row_max_r)));
            if (i < integer'(n_elems_q))
              attn_accum_q[i] <= 0.0;
          end

          if (!have_visible || (exp_sum_r == 0.0)) begin
            fault_code_r <= 4'(FAULT_NO_CONFIG);
            state        <= F_FAULT;
          end else begin
            attn_row_max_q <= row_max_r;
            attn_exp_sum_q <= exp_sum_r;
            attn_k_idx_q   <= 16'h0;
            read_idx_q     <= 13'h0;
            write_chunk_q  <= 11'h0;
            state          <= F_ATTN_V_REQ;
          end
        end

        F_ATTN_V_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ATTN_V_LATCH;
          end
        end

        F_ATTN_V_LATCH: begin
          real weight_r;
          if ((opcode_q == OP_MASKED_SOFTMAX_ATTNV) &&
              !attn_visible(row_idx_q, integer'(attn_k_idx_q))) begin
            weight_r = 0.0;
          end else begin
            weight_r = sfu_fp32_div(
                sfu_fp32_exp(sfu_fp32_sub(row_data_q[integer'(attn_k_idx_q)], attn_row_max_q)),
                attn_exp_sum_q);
          end
          for (int lane = 0; lane < 16; lane++) begin
            integer idx;
            idx = integer'(read_idx_q) * 16 + lane;
            if (idx < integer'(n_elems_q))
              attn_accum_q[idx] <= sfu_fp32_add(
                  attn_accum_q[idx],
                  sfu_fp32_mul(
                      sfu_fp32_mul(weight_r, real'(get_i8(sram_b_rdata, lane))),
                      scale1_q));
          end

          if (read_idx_q + 13'd1 < {2'h0, n_tiles_q}) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ATTN_V_REQ;
          end else if (attn_k_idx_q + 16'd1 < k_elems_q) begin
            attn_k_idx_q <= attn_k_idx_q + 16'd1;
            read_idx_q   <= 13'h0;
            state        <= F_ATTN_V_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ATTN_WRITE;
          end
        end

        F_ATTN_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_ATTN_WRITE;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            state         <= F_ATTN_QKT_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        // ----------------------------------------------------------------
        // gen-2 FP32 shared datapath (0x19 VADD / 0x1A LN / 0x1B GELU).
        // FP16 storage (8 elems / 16-byte row), FP32 internal. src1 is an
        // ABUF FP16 tile; src2 is an ABUF FP16 tile (VADD) or 2N FP16
        // gamma||beta in WBUF (LN); GELU has no src2. Output FP16 to ABUF.
        // ----------------------------------------------------------------
        F_G2_S1_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_G2_S1_LATCH;
          end
        end

        F_G2_S1_LATCH: begin
          integer base_idx;
          if (opcode_q == OP_DEQUANT_ACCUM_FP32) begin
            // 0x17: src1 = ACCUM INT32, 4 int32 / 16-byte row, raw -> real
            // (per-column FP16 scale applied later in F_G2_COMPUTE).
            base_idx = integer'(read_idx_q) * 4;
            for (int lane = 0; lane < 4; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q))
                row_data_q[base_idx + lane] <=
                    real'(get_i32(sram_b_rdata, lane));
            end
            if (read_idx_q + 13'd1 < n_chunks_i32_q) begin
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_REQ;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              state         <= F_G2_S2_REQ;   // N FP16 per-col scales
            end
          end else begin
            // FP16 src1 tile, 8 elems / 16-byte row.
            base_idx = integer'(read_idx_q) * 8;
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q))
                row_data_q[base_idx + lane] <=
                    sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
            end
            if (read_idx_q + 13'd1 < {2'h0, g2_rows_q[10:0]}) begin
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_REQ;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              // GELU / QUANT_FP32_INT8 have no src2 -> straight to compute.
              if (opcode_q == OP_GELU_FP32 ||
                  opcode_q == OP_QUANT_FP32_INT8)
                state <= F_G2_COMPUTE;
              else
                state <= F_G2_S2_REQ;
            end
          end
        end

        F_G2_S2_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_G2_S2_LATCH;
          end
        end

        F_G2_S2_LATCH: begin
          integer base_idx;
          if (opcode_q == OP_LAYERNORM_FP32) begin
            // src2 = 2N FP16 (N gamma then N beta), 8 per 16-byte row.
            base_idx = (integer'(read_idx_q) < integer'(ln_gamma_rows_q)) ?
                       (integer'(read_idx_q) * 8) :
                       ((integer'(read_idx_q) - integer'(ln_gamma_rows_q)) * 8);
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (integer'(read_idx_q) < integer'(ln_gamma_rows_q))
                  gamma_q[base_idx + lane] <=
                      sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
                else
                  beta_q[base_idx + lane] <=
                      sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
              end
            end
            if (read_idx_q + 13'd1 < {1'b0, ln_param_rows_q[11:0]}) begin
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S2_REQ;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              state         <= F_G2_COMPUTE;
            end
          end else begin
            // VADD: src2 is an ABUF FP16 tile (2nd operand).
            base_idx = integer'(read_idx_q) * 8;
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q))
                attn_accum_q[base_idx + lane] <=
                    sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
            end
            if (read_idx_q + 13'd1 < {2'h0, g2_rows_q[10:0]}) begin
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S2_REQ;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              state         <= F_G2_COMPUTE;
            end
          end
        end

        F_G2_COMPUTE: begin
          if (opcode_q == OP_VADD_FP32) begin
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                    sfu_fp32_add(row_data_q[i], attn_accum_q[i])));
            end
            state <= F_G2_PACK;
          end else if (opcode_q == OP_GELU_FP32) begin
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                    sfu_fp32_gelu_new(row_data_q[i])));
            end
            state <= F_G2_PACK;
          end else if (opcode_q == OP_DEQUANT_ACCUM_FP32) begin
            // 0x17: FP16 = fp32(INT32) * per-column FP16 scale.
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                    sfu_fp32_mul(row_data_q[i], attn_accum_q[i])));
            end
            state <= F_G2_PACK;
          end else if (opcode_q == OP_QUANT_FP32_INT8) begin
            // 0x18: INT8 = clip(round_half_even(FP16 * scale_regs[sreg])).
            // quantize_to_i8(v, 1.0) == clamp(round_half_even(v), -128,127).
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                out_bytes_q[i] <= quantize_to_i8(
                    sfu_fp32_mul(row_data_q[i], scale0_q), 1.0);
            end
            state <= F_ROW_PACK;          // gen-1 INT8 pack (16 / row)
          end else begin
            // LAYERNORM_FP32: mean / var (population) / eps=1e-5 / gamma,beta.
            real sum_r;
            real mean_r;
            real var_r;
            real denom_r;
            sum_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                sum_r = sfu_fp32_add(sum_r, row_data_q[i]);
            end
            mean_r = sfu_fp32_div(sum_r, real'(n_elems_q));
            var_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                real diff_r;
                diff_r = sfu_fp32_sub(row_data_q[i], mean_r);
                var_r = sfu_fp32_add(var_r, sfu_fp32_mul(diff_r, diff_r));
              end
            end
            var_r = sfu_fp32_div(var_r, real'(n_elems_q));
            denom_r = sfu_fp32_sqrt(sfu_fp32_add(var_r, LN_FP32_EPS));
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                    sfu_fp32_add(
                        sfu_fp32_mul(
                            sfu_fp32_div(
                                sfu_fp32_sub(row_data_q[i], mean_r),
                                denom_r),
                            gamma_q[i]),
                        beta_q[i])));
            end
            state <= F_G2_PACK;
          end
        end

        F_G2_PACK: begin
          row_write_q <= g2_write_data_w;
          state       <= F_G2_WRITE;
        end

        F_G2_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < g2_rows_q[10:0]) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_G2_PACK;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            state         <= F_G2_S1_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_FAULT: ;

        default:
          state <= F_IDLE;
      endcase
    end
  end

  always_comb begin
    sfu_busy       = (state != F_IDLE) && (state != F_FAULT);
    sfu_fault      = (state == F_FAULT);
    sfu_fault_code = fault_code_r;

    sram_a_en    = 1'b0;
    sram_a_we    = 1'b0;
    sram_a_buf   = dst_buf_q;
    sram_a_row   = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en    = 1'b0;
    sram_b_buf   = src1_buf_q;
    sram_b_row   = 16'h0;

    case (state)
      F_LN_PARAM_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        sram_b_row = ln_param_addr_w[15:0];
      end

      F_ROW_I8_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = row_i8_addr_w[15:0];
      end

      F_ROW_I32_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = row_i32_addr_w[15:0];
      end

      F_ROW_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = row_dst_addr_w[15:0];
        sram_a_wdata = row_write_q;
      end

      F_GELU_I8_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = gelu_i8_addr_w[15:0];
      end

      F_GELU_I8_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = gelu_dst_addr_w[15:0];
        sram_a_wdata = gelu_i8_write_data_w;
      end

      F_GELU_I32_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = gelu_acc_addr_w[15:0];
      end

      F_GELU_I32_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = gelu_dst_addr_w[15:0];
        sram_a_wdata = gelu_i32_write_data_w;
      end

      F_ATTN_QKT_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = attn_qkt_addr_w[15:0];
      end

      F_ATTN_V_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        sram_b_row = attn_v_addr_w[15:0];
      end

      F_ATTN_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = row_dst_addr_w[15:0];
        sram_a_wdata = attn_write_data_w;
      end

      F_G2_S1_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        // 0x17 reads INT32 ACCUM (4 / row); others read an FP16 tile.
        sram_b_row = (opcode_q == OP_DEQUANT_ACCUM_FP32) ?
                     row_i32_addr_w[15:0] : g2_s1_addr_w[15:0];
      end

      F_G2_S2_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        // LN gamma/beta and 0x17 per-col scales are row-independent
        // (src2_off + read_idx); VADD's src2 is a full per-row tile.
        sram_b_row = (opcode_q == OP_VADD_FP32) ?
                     g2_s2_addr_w[15:0] : g2_lnp_addr_w[15:0];
      end

      F_G2_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = g2_dst_addr_w[15:0];
        sram_a_wdata = row_write_q;
      end

      default: ;
    endcase
  end

endmodule

`endif // SFU_ENGINE_SV
