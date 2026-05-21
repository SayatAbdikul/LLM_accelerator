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
// Synthesizable fp32 primitives (fp32_add, fp32_to_fp16, fp16_to_fp32) are
// added to CONTROL_SV in the Makefile so they elaborate alongside this
// module. They're used by the SFU_SYNTH_MODE=1 op paths; in mode=0 they
// remain present but their outputs are never sampled (synth folds them).

module sfu_engine
  import taccel_pkg::*;
#(
  // Phase-2 migration toggle: 0 = behavioral real+DPI path (the cosim-pinned
  // default), 1 = synthesizable RTL path using the Phase-1 primitives. Set
  // per-op-at-a-time as ops are migrated; the gate is `SFU_SYNTH_MODE==1 &&
  // opcode_q==<op>`. Until all ops are migrated, mode=1 still falls back to
  // the DPI path for unmigrated opcodes — both paths coexist.
  parameter int SFU_SYNTH_MODE = 0
) (
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
  input  logic         sram_b_fault,

  // --- Scale-register write-back (MAX_ABS_REDUCE_FP32 0x1F) ---
  output logic         sfu_scale_we,
  output logic [3:0]   sfu_scale_waddr,
  output logic [15:0]  sfu_scale_wdata
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
  import "DPI-C" function int  sfu_fp64_to_fp16_bits(input real value_r);
  import "DPI-C" function real sfu_fp32_gelu_new(input real value_r);

  localparam int SFU_MAX_ROW_ELEMS = 1024;
  localparam real LN_EPS = 1.0e-6;
  // gen-2 LAYERNORM_FP32 (0x1A) eps — the GPT-2 / golden value (1e-5).
  // Distinct from the gen-1 INT8 LN_EPS (1e-6) above; do NOT reuse it.
  localparam real LN_FP32_EPS = 1.0e-5;

  // 6-bit enum: Phase-2 LN sub-FSM (5'd32..) requires the extra state slot
  // beyond the original 5'd0..5'd31 range.
  typedef enum logic [5:0] {
    F_IDLE          = 6'd0,
    F_LN_PARAM_REQ  = 6'd1,
    F_LN_PARAM_LATCH= 6'd2,
    F_ROW_I8_REQ    = 6'd3,
    F_ROW_I8_LATCH  = 6'd4,
    F_ROW_I32_REQ   = 6'd5,
    F_ROW_I32_LATCH = 6'd6,
    F_ROW_COMPUTE   = 6'd7,
    F_ROW_PACK      = 6'd8,
    F_ROW_WRITE     = 6'd9,
    F_GELU_I8_REQ   = 6'd10,
    F_GELU_I8_LATCH = 6'd11,
    F_GELU_I8_WRITE = 6'd12,
    F_GELU_I32_REQ  = 6'd13,
    F_GELU_I32_LATCH= 6'd14,
    F_GELU_I32_WRITE= 6'd15,
    F_ATTN_QKT_REQ  = 6'd16,
    F_ATTN_QKT_LATCH= 6'd17,
    F_ATTN_PREP     = 6'd18,
    F_ATTN_V_REQ    = 6'd19,
    F_ATTN_V_LATCH  = 6'd20,
    F_ATTN_WRITE    = 6'd21,
    F_FAULT         = 6'd22,
    // gen-2 FP32 shared datapath (0x19 VADD / 0x1A LN / 0x1B GELU).
    // FP16 storage (8 elems / 16-byte row), FP32 internal.
    F_G2_S1_REQ     = 6'd23,
    F_G2_S1_LATCH   = 6'd24,
    F_G2_S2_REQ     = 6'd25,
    F_G2_S2_LATCH   = 6'd26,
    F_G2_COMPUTE    = 6'd27,
    F_G2_PACK       = 6'd28,
    F_G2_WRITE      = 6'd29,
    F_G2_SCALE_WR   = 6'd30,  // 0x1F: 2-cycle scale-reg write-back
    // Phase-2 synth-mode iterator state (SFU_SYNTH_MODE=1): serializes the
    // 1024-parallel combinational compute loops into one element / cycle
    // through the shared synthesizable primitives. The op-code (opcode_q)
    // multiplexes which primitive's output is sampled.  Currently handles:
    //   0x19 OP_VADD_FP32, 0x17 OP_DEQUANT_ACCUM_FP32,
    //   0x18 OP_QUANT_FP32_INT8, 0x1E OP_DEQUANT_ACCUM_FP32_SCALED.
    F_G2_SYNTH_ITER = 6'd31,
    // Phase-2 LAYERNORM_FP32 (0x1A) synth sub-FSM. 3 reduction passes over
    // the row (sum -> mean; var -> denom; output) plus 2 single-cycle math
    // steps. State sequence:
    //   F_G2_LN_SUM      : iter; sum_acc_q += row_data_q[iter]
    //   F_G2_LN_MEAN     : 1 cycle; mean_q = sum_acc_q / n_elems_fp32; reset var_acc
    //   F_G2_LN_VAR      : iter; var_acc_q += (row[iter] - mean)^2
    //   F_G2_LN_DENOM    : 1 cycle; denom_q = sqrt(var_acc/n + LN_FP32_EPS)
    //   F_G2_LN_OUT      : iter; out_h_q[iter] = f2h((row-mean)/denom*gamma + beta)
    F_G2_LN_SUM     = 6'd32,
    F_G2_LN_MEAN    = 6'd33,
    F_G2_LN_VAR     = 6'd34,
    F_G2_LN_DENOM   = 6'd35,
    F_G2_LN_OUT     = 6'd36,
    // Phase-2 MASKED_SOFTMAX_FP32 (0x1D) synth sub-FSM. 3 passes:
    //   F_G2_SM_MAX     : iter; track row_max over visible elements
    //   F_G2_SM_EXPSUM  : iter; exp_sum += exp(row[i] - row_max) (visible only)
    //   F_G2_SM_OUT     : iter; out[i] = f2h(exp(row[i] - row_max) / exp_sum)
    //                     for visible elements (masked -> 16'h0).
    //   Banded — bounded by `fp32_exp` accuracy (Phase-3 minimax pending).
    F_G2_SM_MAX     = 6'd37,
    F_G2_SM_EXPSUM  = 6'd38,
    F_G2_SM_OUT     = 6'd39
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
  // Phase-2 synth-mode (SFU_SYNTH_MODE=1) per-element iterator. 11 bits ->
  // covers SFU_MAX_ROW_ELEMS=1024. Unused when SFU_SYNTH_MODE=0.
  logic [10:0]  iter_idx_q;
  // Phase-2 synth-mode LN (0x1A) reduction accumulators / cached results.
  // Stored as fp32 bit-patterns (the primitives' native I/O).
  logic [31:0]  ln_sum_acc_q;
  logic [31:0]  ln_var_acc_q;
  logic [31:0]  ln_mean_q;
  logic [31:0]  ln_denom_q;
  // Phase-2 synth-mode SOFTMAX (0x1D) reduction state.
  logic [31:0]  sm_row_max_q;       // running fp32 row_max
  logic [31:0]  sm_exp_sum_q;       // running fp32 exp_sum
  logic         sm_have_vis_q;      // any visible element seen
  logic signed [15:0] sm_keep_through_q;
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
  // 0x1F MAX_ABS_REDUCE_FP32: running global max|x| + 2-cycle write phase.
  real         g2_maxabs_q;
  logic        g2_wr_phase_q;
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
      // Option B non-finite requant contract (isa_generation_freeze.md §7
      // item 8, P6g/#110): NaN -> 0, +inf -> +127, -inf -> -128,
      // finite-overflow -> saturate. Explicit & deterministic — matches
      // the golden np.where(isnan,0,np.clip(...)) semantics. Threshold
      // 1e40 unambiguously separates +-inf from any finite operand on
      // this datapath (fp32 max 3.4e38; fp16-sourced |x| <= 65504).
      if (out_scale_r == 0.0) begin
        quantize_to_i8 = 8'h00;
      end else if (value_r != value_r) begin
        quantize_to_i8 = 8'h00;            // NaN -> 0
      end else if (value_r > 1.0e40) begin
        quantize_to_i8 = 8'h7F;            // +inf -> +127
      end else if (value_r < -1.0e40) begin
        quantize_to_i8 = 8'h80;            // -inf -> -128
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

  // 0x1F: clamp max|x| to [2^-9, 65504*127/2] (golden MAX_ABS_REDUCE eps).
  function automatic real g2_clamp_eps(input real m);
    real e;
    begin
      e = m;
      if (e < 0.001953125) e = 0.001953125;   // 2^-9
      if (e > 4159004.0)   e = 4159004.0;      // 65504.0*127.0/2.0
      g2_clamp_eps = e;
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
  logic dispatch_g2_ms_w;   // 0x1D MASKED_SOFTMAX_FP32
  assign dispatch_g2_ms_w = (opcode == OP_MASKED_SOFTMAX_FP32) &&
                            (src1_buf == BUF_ABUF) &&
                            (dst_buf  == BUF_ABUF);
  logic dispatch_g2_ds_w;   // 0x1E DEQUANT_ACCUM_FP32_SCALED
  logic dispatch_g2_mar_w;  // 0x1F MAX_ABS_REDUCE_FP32
  assign dispatch_g2_ds_w = (opcode == OP_DEQUANT_ACCUM_FP32_SCALED) &&
                            (src1_buf == BUF_ACCUM) &&
                            (src2_buf == BUF_WBUF) &&
                            (dst_buf  == BUF_ABUF);
  assign dispatch_g2_mar_w = (opcode == OP_MAX_ABS_REDUCE_FP32) &&
                             (src1_buf == BUF_ABUF) &&
                             (sreg <= 4'd14);   // sreg+1 must be valid

  // ===================================================================
  // Phase-2 synth-mode (SFU_SYNTH_MODE=1) primitive instances.
  // ===================================================================
  // Shared single-issue compute pipeline driven by iter_idx_q one element
  // per cycle (the F_G2_VADD_ITER state). For 0x19 VADD_FP32 the dataflow
  // is: row_data_q[iter] + attn_accum_q[iter] -> fp16. Inputs come from
  // `real` (= double) storage; the stored value is always single-precision
  // (the DPI always casts to float) so the IEEE-754 double<->fp32 mapping
  // below is lossless.
  //
  // Instantiated unconditionally; the synth path is only entered when
  // SFU_SYNTH_MODE==1 (the F_G2_VADD_ITER dispatch). In mode=0 the
  // primitives compute but their outputs are never sampled, and synth
  // tools fold the dead logic.

  // fp32 bit-pattern -> real (IEEE-754 double). Inverse of real_to_fp32_bits.
  // Used by the synth-mode latch states to store fp32 results into the
  // existing `real` storage arrays without changing their type.  fp32->fp64
  // is exact widening (fp32 is a strict subset of fp64).
  function automatic real fp32_bits_to_real(input logic [31:0] bits);
    logic               s;
    logic [7:0]         e_f;
    logic [22:0]        m_f;
    logic signed [11:0] e_unb;
    logic [10:0]        e_d;
    logic [63:0]        db;
    begin
      s   = bits[31];
      e_f = bits[30:23];
      m_f = bits[22:0];
      if (e_f == 8'd0 && m_f == 23'd0) begin
        db = {s, 63'd0};
      end else if (e_f == 8'd255) begin
        // ±inf (m_f==0) or NaN (m_f!=0; force quiet by setting MSB of mant).
        db = (m_f == 23'd0) ? {s, 11'h7FF, 52'd0}
                            : {s, 11'h7FF, 1'b1, m_f[21:0], 29'd0};
      end else if (e_f == 8'd0) begin
        // fp32 subnormal: value = m_f * 2^-149. For our use cases (fp16->fp32
        // results and arithmetic outputs) subnormal fp32 is rare; safe
        // approximation since the value is < 2^-126 (far below all fp16
        // representable values and most LN/softmax intermediates).
        db = {s, 63'd0};
      end else begin
        // Normal: rebias 127 -> 1023; widen mantissa 23 -> 52 (zero-pad LSBs).
        e_unb = $signed({4'b0, e_f}) - 12'sd127;
        e_d   = 11'(e_unb + 12'sd1023);
        db    = {s, e_d, m_f, 29'd0};
      end
      return $bitstoreal(db);
    end
  endfunction

  // real (IEEE-754 double) -> fp32 bit-pattern. The stored values come
  // from sfu_fp16_bits_to_fp32 / arithmetic DPIs that all cast to (float)
  // internally, so the input double exactly represents a single-precision
  // value and this mapping is round-trip exact.
  function automatic logic [31:0] real_to_fp32_bits(input real r);
    logic [63:0]        db;
    logic               s;
    logic [10:0]        e_d;
    logic [51:0]        m_d;
    logic signed [11:0] e_unb;
    logic [7:0]         e32;
    begin
      db    = $realtobits(r);
      s     = db[63];
      e_d   = db[62:52];
      m_d   = db[51:0];
      if (r == 0.0)                  return {s, 31'd0};
      e_unb = $signed({1'b0, e_d}) - 12'sd1023;
      if (e_unb >  12'sd127)         return {s, 8'd255, 23'd0};  // overflow -> inf
      if (e_unb < -12'sd126)         return {s, 8'd0,   23'd0};  // underflow -> 0
      e32   = 8'(e_unb + 12'sd127);
      return {s, e32, m_d[51:29]};
    end
  endfunction

  // Per-element fp32 operands fetched from the `real` storage via the
  // bit-pattern coercion (lossless because the stored values are always
  // single-precision floats).
  logic [31:0] synth_a_bits;
  logic [31:0] synth_b_bits;
  assign synth_a_bits = real_to_fp32_bits(row_data_q[iter_idx_q[9:0]]);
  assign synth_b_bits = real_to_fp32_bits(attn_accum_q[iter_idx_q[9:0]]);

  // Op-specific b-operand mux: 0x18 QUANT broadcasts the scalar scale0_q;
  // VADD/DEQUANT-AC use the per-element attn_accum_q. Selected by opcode_q.
  logic [31:0] synth_b_bits_eff;
  always_comb begin
    case (opcode_q)
      OP_QUANT_FP32_INT8: synth_b_bits_eff = real_to_fp32_bits(scale0_q);
      default:            synth_b_bits_eff = synth_b_bits;
    endcase
  end

  // Per-op arithmetic primitives, all combinational and always elaborated.
  logic [31:0] synth_add_out;
  logic [31:0] synth_mul_out;
  fp32_add u_synth_add (.a(synth_a_bits), .b(synth_b_bits),     .y(synth_add_out));
  fp32_mul u_synth_mul (.a(synth_a_bits), .b(synth_b_bits_eff), .y(synth_mul_out));

  // Per-op compute output (fp32) -> shared fp32_to_fp16 (for fp16-out ops).
  logic [31:0] synth_compute_out;
  logic [15:0] synth_out_bits;
  always_comb begin
    case (opcode_q)
      OP_VADD_FP32:                 synth_compute_out = synth_add_out;
      OP_DEQUANT_ACCUM_FP32:        synth_compute_out = synth_mul_out;
      OP_DEQUANT_ACCUM_FP32_SCALED: synth_compute_out = synth_scaled_add;
      default:                      synth_compute_out = 32'd0;
    endcase
  end
  fp32_to_fp16 u_synth_f2h (.a(synth_compute_out), .y(synth_out_bits));

  // For 0x18 QUANT_FP32_INT8 the mul result goes through fp32_quantize_i8
  // (Option-B: NaN->0, ±inf->±127/-128, finite RNE+clamp). out is int8.
  logic signed [7:0] synth_quant_out;
  fp32_quantize_i8 u_synth_quant (.a(synth_mul_out), .y(synth_quant_out));

  // For 0x1E DEQUANT_ACCUM_FP32_SCALED chain:
  //   out = ((row_data_q * gamma_q) * scale0_q) + beta_q  -> f2h
  // Three combinational stages, then through the shared fp32_to_fp16.
  logic [31:0] synth_gamma_bits;
  logic [31:0] synth_beta_bits;
  logic [31:0] synth_scale0_bits;
  logic [31:0] synth_scaled_mul1;
  logic [31:0] synth_scaled_mul2;
  logic [31:0] synth_scaled_add;
  assign synth_gamma_bits  = real_to_fp32_bits(gamma_q[iter_idx_q[9:0]]);
  assign synth_beta_bits   = real_to_fp32_bits(beta_q[iter_idx_q[9:0]]);
  assign synth_scale0_bits = real_to_fp32_bits(scale0_q);
  fp32_mul u_synth_scaled_mul1 (
    .a(synth_a_bits),     .b(synth_gamma_bits),  .y(synth_scaled_mul1));
  fp32_mul u_synth_scaled_mul2 (
    .a(synth_scaled_mul1), .b(synth_scale0_bits), .y(synth_scaled_mul2));
  fp32_add u_synth_scaled_add  (
    .a(synth_scaled_mul2), .b(synth_beta_bits),   .y(synth_scaled_add));

  // For 0x1F MAX_ABS_REDUCE F_G2_SCALE_WR phase: replace the DPI fp64 path
  // (127.0/eps and eps/127.0 in fp64, then fp16 cast) with fp32 div + cvt.
  // The g2_clamp_eps() helper is reused (still a `real` function, but the
  // value it returns is single-precision-representable so the bit-pattern
  // coercion is lossless).
  localparam logic [31:0] C_127_FP32 = 32'h42FE_0000;  // 127.0
  logic [31:0] synth_clamp_eps_bits;
  logic [31:0] synth_inv_eps;
  logic [31:0] synth_eps_inv127;
  logic [15:0] synth_inv_eps_fp16;
  logic [15:0] synth_eps_inv127_fp16;
  assign synth_clamp_eps_bits = real_to_fp32_bits(g2_clamp_eps(g2_maxabs_q));
  fp32_div u_synth_inv_eps    (.a(C_127_FP32),          .b(synth_clamp_eps_bits), .y(synth_inv_eps));
  fp32_div u_synth_eps_inv127 (.a(synth_clamp_eps_bits), .b(C_127_FP32),          .y(synth_eps_inv127));
  fp32_to_fp16 u_synth_inv_eps_h    (.a(synth_inv_eps),    .y(synth_inv_eps_fp16));
  fp32_to_fp16 u_synth_eps_inv127_h (.a(synth_eps_inv127), .y(synth_eps_inv127_fp16));

  // ===================================================================
  // 0x1A LAYERNORM_FP32 synth sub-FSM combinational primitives.
  //   row_bits = real_to_fp32_bits(row_data_q[iter]) = synth_a_bits (reused)
  //   neg_mean = ln_mean_q ^ sign-bit  (fp32 negate via XOR)
  // Reduction primitives:
  //   ln_sum_add = fp32_add(ln_sum_acc_q, synth_a_bits)
  //   ln_diff    = fp32_add(synth_a_bits, neg_mean)        // row - mean
  //   ln_diff_sq = fp32_mul(ln_diff, ln_diff)
  //   ln_var_add = fp32_add(ln_var_acc_q, ln_diff_sq)
  //   ln_n_fp32  = real_to_fp32_bits(real'(n_elems_q))
  //   ln_var_norm= fp32_div(ln_var_acc_q, ln_n_fp32)
  //   ln_var_eps = fp32_add(ln_var_norm, C_LN_FP32_EPS)
  //   ln_denom_w = fp32_sqrt(ln_var_eps)                   // computed denom
  //   ln_norm    = fp32_div(ln_diff, ln_denom_q)
  //   ln_norm_g  = fp32_mul(ln_norm, synth_gamma_bits)
  //   ln_norm_gb = fp32_add(ln_norm_g, synth_beta_bits)
  //   ln_out_h   = fp32_to_fp16(ln_norm_gb)
  localparam logic [31:0] C_LN_FP32_EPS = 32'h3727_C5AC;  // 1.0e-5 fp32
  logic [31:0] ln_neg_mean;
  logic [31:0] ln_n_fp32;
  logic [31:0] ln_sum_add_w;
  logic [31:0] ln_mean_div_w;
  logic [31:0] ln_diff_w;
  logic [31:0] ln_diff_sq_w;
  logic [31:0] ln_var_add_w;
  logic [31:0] ln_var_norm_w;
  logic [31:0] ln_var_eps_w;
  logic [31:0] ln_denom_w;
  logic [31:0] ln_norm_w;
  logic [31:0] ln_norm_g_w;
  logic [31:0] ln_norm_gb_w;
  logic [15:0] ln_out_h_w;
  assign ln_neg_mean = ln_mean_q ^ 32'h8000_0000;
  assign ln_n_fp32   = real_to_fp32_bits(real'(n_elems_q));

  fp32_add  u_ln_sum_add (.a(ln_sum_acc_q), .b(synth_a_bits), .y(ln_sum_add_w));
  fp32_div  u_ln_mean    (.a(ln_sum_acc_q), .b(ln_n_fp32),    .y(ln_mean_div_w));
  fp32_add  u_ln_diff    (.a(synth_a_bits), .b(ln_neg_mean),  .y(ln_diff_w));
  fp32_mul  u_ln_diff_sq (.a(ln_diff_w),    .b(ln_diff_w),    .y(ln_diff_sq_w));
  fp32_add  u_ln_var_add (.a(ln_var_acc_q), .b(ln_diff_sq_w), .y(ln_var_add_w));
  fp32_div  u_ln_var_norm(.a(ln_var_acc_q), .b(ln_n_fp32),    .y(ln_var_norm_w));
  fp32_add  u_ln_var_eps (.a(ln_var_norm_w),.b(C_LN_FP32_EPS),.y(ln_var_eps_w));
  fp32_sqrt u_ln_sqrt    (.a(ln_var_eps_w),                    .y(ln_denom_w));
  fp32_div  u_ln_norm    (.a(ln_diff_w),    .b(ln_denom_q),   .y(ln_norm_w));
  fp32_mul  u_ln_norm_g  (.a(ln_norm_w),    .b(synth_gamma_bits), .y(ln_norm_g_w));
  fp32_add  u_ln_norm_gb (.a(ln_norm_g_w),  .b(synth_beta_bits),  .y(ln_norm_gb_w));
  fp32_to_fp16 u_ln_out_h(.a(ln_norm_gb_w),                        .y(ln_out_h_w));

  // ===================================================================
  // 0x1D MASKED_SOFTMAX_FP32 synth sub-FSM combinational primitives.
  //   neg_max  = sm_row_max_q ^ sign-bit
  //   diff     = row[iter] - row_max               (fp32_add with neg)
  //   exp_v    = exp(diff)                         (fp32_exp, BANDED)
  //   sum_add  = exp_sum_q + exp_v                 (fp32_add)
  //   norm     = exp_v / exp_sum_q                 (fp32_div)
  //   out_h    = fp32_to_fp16(norm)
  // Visibility: F_G2_SM_MAX/EXPSUM/OUT each check
  //   (iter < n_elems_q) && (iter_signed <= sm_keep_through_q)
  logic [31:0] sm_neg_max;
  logic [31:0] sm_diff_w;
  logic [31:0] sm_exp_w;
  logic [31:0] sm_sum_add_w;
  logic [31:0] sm_norm_w;
  logic [15:0] sm_out_h_w;
  assign sm_neg_max = sm_row_max_q ^ 32'h8000_0000;
  fp32_add     u_sm_diff   (.a(synth_a_bits), .b(sm_neg_max),    .y(sm_diff_w));
  fp32_exp     u_sm_exp    (.a(sm_diff_w),                        .y(sm_exp_w));
  fp32_add     u_sm_sum_add(.a(sm_exp_sum_q), .b(sm_exp_w),       .y(sm_sum_add_w));
  fp32_div     u_sm_div    (.a(sm_exp_w),     .b(sm_exp_sum_q),   .y(sm_norm_w));
  fp32_to_fp16 u_sm_out_h  (.a(sm_norm_w),                        .y(sm_out_h_w));
  // SOFTMAX max-update predicate: `diff` MSB (=sign): 1 -> row < max
  // (no update); 0 -> row >= max (update only if strictly > or first-vis).
  // Strictly > requires diff != 0; equal (diff == 0) is no-op either way.
  logic sm_row_gt_max;
  assign sm_row_gt_max = (sm_diff_w[31] == 1'b0) && (sm_diff_w[30:0] != 31'd0);
  // ===================================================================

  // ===================================================================
  // Phase-2 synth-mode shared latch lanes (SFU_SYNTH_MODE=1).
  // ===================================================================
  // Eight parallel fp16->fp32 primitives convert the sram_b_rdata row
  // (8 × fp16 = 128 bits) to fp32 bit-patterns. Used by both the gen-2
  // latch states (F_G2_S1_LATCH FP16-src1 / F_G2_S2_LATCH VADD / LN /
  // SCALED) and the 0x1F MAX_ABS_REDUCE absolute-value reduction.
  logic [31:0] synth_lat_h2f [0:7];
  genvar g_lj;
  generate
    for (g_lj = 0; g_lj < 8; g_lj = g_lj + 1) begin : g_synth_lat
      fp16_to_fp32 u_h2f (
        .a(sram_b_rdata[g_lj*16 +: 16]),
        .y(synth_lat_h2f[g_lj])
      );
    end
  endgenerate

  // 0x1F MAX_ABS_REDUCE running-max reduction. abs() via sign-bit clear;
  // positive-fp32 magnitudes compare correctly as unsigned ints (IEEE-754).
  // Per-lane candidate: lane abs if active (base_idx+lane < n_elems_q),
  // else fall back to the current max (no-op contribution).
  logic [31:0] mar_lane_abs [0:7];
  logic [31:0] mar_curr_bits;
  logic [31:0] mar_cand     [0:7];
  logic [31:0] mar_new_max;
  // base_idx in the F_G2_S1_LATCH 0x1F branch = read_idx_q * 8.
  logic [15:0] mar_base_idx;
  assign mar_base_idx = {3'h0, read_idx_q[12:0]} * 16'd8;
  assign mar_curr_bits = real_to_fp32_bits(g2_maxabs_q) & 32'h7FFF_FFFF;
  generate
    for (g_lj = 0; g_lj < 8; g_lj = g_lj + 1) begin : g_mar
      assign mar_lane_abs[g_lj] = synth_lat_h2f[g_lj] & 32'h7FFF_FFFF;
    end
  endgenerate
  always_comb begin
    for (int i = 0; i < 8; i = i + 1) begin
      if ((mar_base_idx + 16'(i)) < n_elems_q)
        mar_cand[i] = mar_lane_abs[i];
      else
        mar_cand[i] = mar_curr_bits;
    end
  end
  always_comb begin
    logic [31:0] m;
    m = mar_curr_bits;
    for (int i = 0; i < 8; i = i + 1)
      if (mar_cand[i] > m) m = mar_cand[i];
    mar_new_max = m;
  end
  // ===================================================================

  // ===================================================================

  always_comb begin
    dispatch_attn_context_bad_w = 1'b0;
    if ((opcode == OP_MASKED_SOFTMAX) ||
        (opcode == OP_MASKED_SOFTMAX_ATTNV) ||
        (opcode == OP_MASKED_SOFTMAX_FP32)) begin
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

      OP_MASKED_SOFTMAX_FP32: begin
        if (!dispatch_g2_ms_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_DEQUANT_ACCUM_FP32_SCALED: begin
        if (!dispatch_g2_ds_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_MAX_ABS_REDUCE_FP32: begin
        if (!dispatch_g2_mar_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        // no src2, no tile dst (writes 2 scale regs).
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
      iter_idx_q     <= 11'h0;
      ln_sum_acc_q   <= 32'h0;
      ln_var_acc_q   <= 32'h0;
      ln_mean_q      <= 32'h0;
      ln_denom_q     <= 32'h0;
      sm_row_max_q   <= 32'h0;
      sm_exp_sum_q   <= 32'h0;
      sm_have_vis_q  <= 1'b0;
      sm_keep_through_q <= 16'sh0;
      write_chunk_q  <= 11'h0;
      gelu_part_q    <= 2'h0;
      attn_k_idx_q   <= 16'h0;
      g2_rows_q      <= 13'h0;
      g2_maxabs_q    <= 0.0;
      g2_wr_phase_q  <= 1'b0;
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
            g2_maxabs_q     <= 0.0;
            g2_wr_phase_q   <= 1'b0;
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
                OP_DEQUANT_ACCUM_FP32, OP_QUANT_FP32_INT8,
                OP_MASKED_SOFTMAX_FP32, OP_DEQUANT_ACCUM_FP32_SCALED,
                OP_MAX_ABS_REDUCE_FP32:
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
          if (opcode_q == OP_DEQUANT_ACCUM_FP32 ||
              opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
            // 0x17 / 0x1E: src1 = ACCUM INT32, 4 int32 / 16-byte row, raw
            // -> real (scales/bias applied later in F_G2_COMPUTE).
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
              state         <= F_G2_S2_REQ;   // src2: scales (+bias)
            end
          end else if (opcode_q == OP_MAX_ABS_REDUCE_FP32) begin
            // 0x1F: FP16 src1; accumulate the GLOBAL max|x| over the whole
            // M*N tile (own row loop, no per-row output).
            base_idx = integer'(read_idx_q) * 8;
            if (SFU_SYNTH_MODE == 1) begin
              // Synth: max-reduce the 8 fp32-bit-abs lanes against the
              // current g2_maxabs_q (computed combinationally at module
              // scope as `mar_new_max`); store back via fp32_bits_to_real.
              g2_maxabs_q <= fp32_bits_to_real(mar_new_max);
            end else begin
              // DPI path (default; cosim-pinned).
              real m;
              real v;
              real av;
              m = g2_maxabs_q;
              for (int lane = 0; lane < 8; lane++) begin
                if ((base_idx + lane) < integer'(n_elems_q)) begin
                  v  = sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
                  av = (v < 0.0) ? -v : v;
                  if (av > m) m = av;
                end
              end
              g2_maxabs_q <= m;
            end
            if (read_idx_q + 13'd1 < {2'h0, g2_rows_q[10:0]}) begin
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_REQ;
            end else if (row_idx_q + 15'd1 < m_rows_q) begin
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;
            end else begin
              state <= F_G2_SCALE_WR;          // all elements seen
            end
          end else begin
            // FP16 src1 tile, 8 elems / 16-byte row.
            base_idx = integer'(read_idx_q) * 8;
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (SFU_SYNTH_MODE == 1)
                  row_data_q[base_idx + lane] <=
                      fp32_bits_to_real(synth_lat_h2f[lane]);
                else
                  row_data_q[base_idx + lane] <=
                      sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
              end
            end
            if (read_idx_q + 13'd1 < {2'h0, g2_rows_q[10:0]}) begin
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_REQ;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              // GELU / QUANT_FP32_INT8 / MASKED_SOFTMAX_FP32 have no src2.
              if (opcode_q == OP_GELU_FP32 ||
                  opcode_q == OP_QUANT_FP32_INT8 ||
                  opcode_q == OP_MASKED_SOFTMAX_FP32)
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
          if (opcode_q == OP_LAYERNORM_FP32 ||
              opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
            // LN: src2 = 2N FP16 (N gamma || N beta). 0x1E: identical
            // layout, N wt-scales (-> gamma_q) || N bias (-> beta_q).
            base_idx = (integer'(read_idx_q) < integer'(ln_gamma_rows_q)) ?
                       (integer'(read_idx_q) * 8) :
                       ((integer'(read_idx_q) - integer'(ln_gamma_rows_q)) * 8);
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (integer'(read_idx_q) < integer'(ln_gamma_rows_q)) begin
                  if (SFU_SYNTH_MODE == 1)
                    gamma_q[base_idx + lane] <=
                        fp32_bits_to_real(synth_lat_h2f[lane]);
                  else
                    gamma_q[base_idx + lane] <=
                        sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
                end else begin
                  if (SFU_SYNTH_MODE == 1)
                    beta_q[base_idx + lane] <=
                        fp32_bits_to_real(synth_lat_h2f[lane]);
                  else
                    beta_q[base_idx + lane] <=
                        sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
                end
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
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (SFU_SYNTH_MODE == 1)
                  attn_accum_q[base_idx + lane] <=
                      fp32_bits_to_real(synth_lat_h2f[lane]);
                else
                  attn_accum_q[base_idx + lane] <=
                      sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
              end
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
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: serialize via F_G2_SYNTH_ITER (fp32_add + cvt).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
            end else begin
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_add(row_data_q[i], attn_accum_q[i])));
              end
              state <= F_G2_PACK;
            end
          end else if (opcode_q == OP_GELU_FP32) begin
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                    sfu_fp32_gelu_new(row_data_q[i])));
            end
            state <= F_G2_PACK;
          end else if (opcode_q == OP_DEQUANT_ACCUM_FP32) begin
            // 0x17: FP16 = fp32(INT32) * per-column FP16 scale.
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: serialize via F_G2_SYNTH_ITER (fp32_mul + cvt).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
            end else begin
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_mul(row_data_q[i], attn_accum_q[i])));
              end
              state <= F_G2_PACK;
            end
          end else if (opcode_q == OP_QUANT_FP32_INT8) begin
            // 0x18: INT8 = clip(round_half_even(FP16 * scale_regs[sreg])).
            // quantize_to_i8(v, 1.0) == clamp(round_half_even(v), -128,127).
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: serialize via F_G2_SYNTH_ITER (fp32_mul + quant_i8).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
            end else begin
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_bytes_q[i] <= quantize_to_i8(
                      sfu_fp32_mul(row_data_q[i], scale0_q), 1.0);
              end
              state <= F_ROW_PACK;          // gen-1 INT8 pack (16 / row)
            end
          end else if (opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
            // 0x1E: FP16 = int32 * wt_scale[col] * act_scale + bias[col].
            // row_data_q=int32(real); gamma_q=wt-scales; beta_q=bias;
            // scale0_q = scale_regs[sreg] (the fwd act-scale from 0x1F).
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: 3-stage combinational chain (mul, mul, add)
              // through F_G2_SYNTH_ITER, then cvt fp16.
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
            end else begin
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_add(
                          sfu_fp32_mul(
                              sfu_fp32_mul(row_data_q[i], gamma_q[i]),
                              scale0_q),
                          beta_q[i])));
              end
              state <= F_G2_PACK;
            end
          end else if (opcode_q == OP_MASKED_SOFTMAX_FP32) begin
            // 0x1D causal masked softmax. Golden is mode-independent:
            // keep_through = min(row + query_row_base, valid_kv_len - 1);
            // cols 0..keep_through visible, rest -> 0. FP32 internal.
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: 3-pass sub-FSM through F_G2_SM_{MAX,EXPSUM,OUT}.
              // BANDED — bounded by fp32_exp accuracy (Phase-3 minimax-tune
              // pending). Compute keep_through here, then iter through passes.
              automatic logic signed [16:0] qrow_s;
              automatic logic signed [16:0] kt_s;
              qrow_s = $signed({5'b0, attn_query_row_base_q}) +
                       $signed({2'b0, row_idx_q[14:0]});
              kt_s = $signed({5'b0, attn_valid_kv_len_q}) - 17'sd1;
              if (qrow_s < kt_s)
                sm_keep_through_q <= 16'(qrow_s);
              else
                sm_keep_through_q <= 16'(kt_s);
              iter_idx_q   <= 11'h0;
              sm_row_max_q <= 32'h0;
              sm_exp_sum_q <= 32'h0;
              sm_have_vis_q <= 1'b0;
              state        <= F_G2_SM_MAX;
            end else begin
              // DPI path (default; cosim-pinned).
            integer qrow;
            integer keep_through;
            real row_max_r;
            real exp_sum_r;
            logic have_vis;
            qrow = integer'(attn_query_row_base_q) + integer'(row_idx_q);
            keep_through = (qrow < (integer'(attn_valid_kv_len_q) - 1)) ?
                           qrow : (integer'(attn_valid_kv_len_q) - 1);
            have_vis  = 1'b0;
            row_max_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if ((i < integer'(n_elems_q)) && (i <= keep_through)) begin
                if (!have_vis || (row_data_q[i] > row_max_r))
                  row_max_r = row_data_q[i];
                have_vis = 1'b1;
              end
            end
            exp_sum_r = 0.0;
            if (have_vis) begin
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if ((i < integer'(n_elems_q)) && (i <= keep_through))
                  exp_sum_r = sfu_fp32_add(exp_sum_r,
                      sfu_fp32_exp(sfu_fp32_sub(row_data_q[i], row_max_r)));
              end
            end
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                if (have_vis && (i <= keep_through) && (exp_sum_r != 0.0))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_div(
                          sfu_fp32_exp(
                              sfu_fp32_sub(row_data_q[i], row_max_r)),
                          exp_sum_r)));
                else
                  out_h_q[i] <= 16'h0;
              end
            end
            state <= F_G2_PACK;
            end  // SFU_SYNTH_MODE==0 else branch
          end else begin
            // LAYERNORM_FP32: mean / var (population) / eps=1e-5 / gamma,beta.
            if (SFU_SYNTH_MODE == 1 && opcode_q == OP_LAYERNORM_FP32) begin
              // Synth path: 3-pass sub-FSM through F_G2_LN_{SUM,MEAN,VAR,
              // DENOM,OUT}. Init iter + sum accumulator (+0 == 32'h0).
              iter_idx_q   <= 11'h0;
              ln_sum_acc_q <= 32'h0;
              state        <= F_G2_LN_SUM;
            end else begin
              // DPI path (default; cosim-pinned).
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
        end

        // Phase-2 SFU_SYNTH_MODE=1 synth-path element iterator. Drives the
        // shared module-scope primitives one element / cycle, advancing
        // iter_idx_q from 0 up to n_elems_q-1. Op-mux at module scope picks
        // which primitive output is written, and which row-buffer/next-state
        // the FSM advances to:
        //   0x19 VADD       -> synth_add_out -> f2h -> out_h_q   -> F_G2_PACK
        //   0x17 DEQUANT_AC -> synth_mul_out -> f2h -> out_h_q   -> F_G2_PACK
        //   0x18 QUANT      -> synth_mul_out -> qi8 -> out_bytes -> F_ROW_PACK
        F_G2_SYNTH_ITER: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            if (opcode_q == OP_QUANT_FP32_INT8)
              out_bytes_q[iter_idx_q[9:0]] <= synth_quant_out;
            else
              out_h_q[iter_idx_q[9:0]]     <= synth_out_bits;
            iter_idx_q                     <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= (opcode_q == OP_QUANT_FP32_INT8) ? F_ROW_PACK
                                                           : F_G2_PACK;
          end
        end

        // 0x1A LAYERNORM_FP32 synth sub-FSM (Phase-2):
        //   1) Sum reduction: sum_acc += row[iter]
        //   2) Mean: mean = sum_acc / n; reset var_acc
        //   3) Variance: var_acc += (row[iter] - mean)^2
        //   4) Denom: denom = sqrt(var_acc / n + LN_EPS)
        //   5) Output: out[iter] = f2h((row[iter] - mean) / denom * gamma + beta)
        F_G2_LN_SUM: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            ln_sum_acc_q <= ln_sum_add_w;
            iter_idx_q   <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q   <= 11'h0;
            state        <= F_G2_LN_MEAN;
          end
        end

        F_G2_LN_MEAN: begin
          // mean = sum_acc / n_elems_fp32   (combinational chain reuses
          // u_ln_var_norm with operands sum_acc / n -- but that unit's a
          // is tied to ln_var_acc_q.  Add a dedicated div instead.)
          // Instead, compute mean via the same fp32_div module: re-instance
          // not feasible inline, so do the divide here using
          // u_ln_var_norm's port -- impractical. Use the dedicated below.
          ln_mean_q    <= ln_mean_div_w;
          ln_var_acc_q <= 32'h0;
          state        <= F_G2_LN_VAR;
        end

        F_G2_LN_VAR: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            ln_var_acc_q <= ln_var_add_w;
            iter_idx_q   <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q   <= 11'h0;
            state        <= F_G2_LN_DENOM;
          end
        end

        F_G2_LN_DENOM: begin
          ln_denom_q <= ln_denom_w;
          state      <= F_G2_LN_OUT;
        end

        F_G2_LN_OUT: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            out_h_q[iter_idx_q[9:0]] <= ln_out_h_w;
            iter_idx_q               <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_G2_PACK;
          end
        end

        // 0x1D MASKED_SOFTMAX_FP32 synth sub-FSM (Phase-2; BANDED):
        //   F_G2_SM_MAX:  iterate, update row_max if visible & row>max (or
        //                 first visible). visible = iter<n_elems && iter<=kt.
        //   F_G2_SM_EXPSUM: iterate; if visible, exp_sum += exp(row - max).
        //   F_G2_SM_OUT:  iterate; out[iter] = f2h(exp(row-max)/exp_sum) if
        //                 visible & have_vis & exp_sum!=0, else 0.
        F_G2_SM_MAX: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            // Visibility: iter_signed <= sm_keep_through_q (signed compare)
            // — for kt<0, no iter passes since iter is unsigned ≥ 0.
            if ($signed({6'b0, iter_idx_q}) <= $signed({1'b0, sm_keep_through_q[15:0]})) begin
              if (!sm_have_vis_q || sm_row_gt_max)
                sm_row_max_q <= synth_a_bits;
              sm_have_vis_q  <= 1'b1;
            end
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q   <= 11'h0;
            state        <= F_G2_SM_EXPSUM;
          end
        end

        F_G2_SM_EXPSUM: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            if (sm_have_vis_q &&
                ($signed({6'b0, iter_idx_q}) <= $signed({1'b0, sm_keep_through_q[15:0]})))
              sm_exp_sum_q <= sm_sum_add_w;  // exp_sum += exp(row - row_max)
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_G2_SM_OUT;
          end
        end

        F_G2_SM_OUT: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            if (sm_have_vis_q &&
                ($signed({6'b0, iter_idx_q}) <= $signed({1'b0, sm_keep_through_q[15:0]})) &&
                (sm_exp_sum_q != 32'h0))
              out_h_q[iter_idx_q[9:0]] <= sm_out_h_w;
            else
              out_h_q[iter_idx_q[9:0]] <= 16'h0;
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_G2_PACK;
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

        // 0x1F: write scale_regs[sreg]=127/eps (phase 0), then
        // scale_regs[sreg+1]=eps/127 (phase 1). Writes driven in the
        // combinational block; here we just sequence the two phases.
        F_G2_SCALE_WR: begin
          if (g2_wr_phase_q == 1'b0)
            g2_wr_phase_q <= 1'b1;
          else
            state <= F_IDLE;
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

    sfu_scale_we    = 1'b0;
    sfu_scale_waddr = 4'h0;
    sfu_scale_wdata = 16'h0;

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
        // 0x17 / 0x1E read INT32 ACCUM (4 / row); others read FP16 tiles.
        sram_b_row = ((opcode_q == OP_DEQUANT_ACCUM_FP32) ||
                      (opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED)) ?
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

      // 0x1F MAX_ABS_REDUCE_FP32 scale write-back. Golden:
      //   eps = clamp(max|x|, 2^-9, 65504*127/2)
      //   scale_regs[sreg]   = float16(127/eps)   (phase 0)
      //   scale_regs[sreg+1] = float16(eps/127)   (phase 1)
      // float16() is a single round of the float64 quotient.
      F_G2_SCALE_WR: begin
        sfu_scale_we = 1'b1;
        if (g2_wr_phase_q == 1'b0) begin
          sfu_scale_waddr = sreg_q;
          if (SFU_SYNTH_MODE == 1)
            sfu_scale_wdata = synth_inv_eps_fp16;
          else
            sfu_scale_wdata = 16'(sfu_fp64_to_fp16_bits(
                127.0 / g2_clamp_eps(g2_maxabs_q)));
        end else begin
          sfu_scale_waddr = sreg_q + 4'd1;
          if (SFU_SYNTH_MODE == 1)
            sfu_scale_wdata = synth_eps_inv127_fp16;
          else
            sfu_scale_wdata = 16'(sfu_fp64_to_fp16_bits(
                g2_clamp_eps(g2_maxabs_q) / 127.0));
        end
      end

      default: ;
    endcase
  end

endmodule

`endif // SFU_ENGINE_SV
