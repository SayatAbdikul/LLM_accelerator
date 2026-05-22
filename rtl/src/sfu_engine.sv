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

`include "sfu_dpi_helpers.svh"

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
    F_G2_SM_OUT     = 6'd39,
    // Phase-3.B gen-1 GELU synth-mode iterator (uses fp32_gelu_new tanh-poly
    // primitive as the synth approximation of gen-1 erf-GELU; the int8
    // quantization at out_bytes_q absorbs the tanh-vs-erf difference for
    // typical fixture inputs).
    F_GELU_SYNTH_I8_ITER  = 6'd40,
    F_GELU_SYNTH_I32_ITER = 6'd41
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

  // Phase-3.D storage cascade (2026-05-21): persistent storage moved from
  // `real` to `logic [31:0]` (fp32 bit-pattern). DPI mode wraps writes with
  // `real_to_fp32_bits(...)` and reads with `fp32_bits_to_real(...)`; synth
  // mode reads/writes the bits directly. The helpers are lossless
  // (single-precision in `real` double).
  logic [31:0] scale0_q /* verilator public_flat_rd */, scale1_q /* verilator public_flat_rd */,
               scale2_q /* verilator public_flat_rd */, scale3_q /* verilator public_flat_rd */;
  logic [31:0] row_data_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [31:0] attn_accum_q [0:SFU_MAX_ROW_ELEMS-1];
  logic [31:0] gamma_q    [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [31:0] beta_q     [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [7:0] out_bytes_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  // gen-2: FP16 result bit-patterns + FP16-rows-per-logical-row (=2*n_tiles).
  logic [15:0] out_h_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [12:0] g2_rows_q;
  // 0x1F MAX_ABS_REDUCE_FP32: running global max|x| + 2-cycle write phase.
  logic [31:0] g2_maxabs_q;
  logic        g2_wr_phase_q;
  logic [31:0] attn_row_max_q;
  logic [31:0] attn_exp_sum_q;
  logic [31:0] ln_debug_mean_q  /* verilator public_flat_rd */;
  logic [31:0] ln_debug_var_q   /* verilator public_flat_rd */;
  logic [31:0] ln_debug_denom_q /* verilator public_flat_rd */;
  logic [31:0] ln_debug_y_q     [0:15] /* verilator public_flat_rd */;

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

`include "sfu_synth_datapath.svh"
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
      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        row_write_data_w[(lane * 8) +: 8] = out_bytes_q[idx];
    end
`ifndef SFU_SYNTH_NO_DPI
    for (int lane = 0; lane < 16; lane++) begin
      int idx;
      real x_r;
      x_r = sfu_fp32_mul(real'(get_i8(gelu_i8_row_q, lane)), fp32_bits_to_real(scale0_q));
      gelu_i8_write_data_w[(lane * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));

      if (lane < 4) begin
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row0_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[(lane * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row1_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[((lane + 4) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row2_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[((lane + 8) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row3_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[((lane + 12) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
      end

      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        attn_write_data_w[(lane * 8) +: 8] = quantize_to_i8(fp32_bits_to_real(attn_accum_q[idx]), fp32_bits_to_real(scale2_q));
    end
`endif
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
`ifndef SFU_SYNTH_NO_DPI
            scale0_q        <= real_to_fp32_bits(fp16_to_real(scale0_data));
            scale1_q        <= real_to_fp32_bits(fp16_to_real(scale1_data));
            scale2_q        <= real_to_fp32_bits(fp16_to_real(scale2_data));
            scale3_q        <= real_to_fp32_bits(fp16_to_real(scale3_data));
`else
            scale0_q        <= {16'h0, scale0_data};
            scale1_q        <= {16'h0, scale1_data};
            scale2_q        <= {16'h0, scale2_data};
            scale3_q        <= {16'h0, scale3_data};
`endif
            ln_debug_mean_q <= 32'h0;
            ln_debug_var_q <= 32'h0;
            ln_debug_denom_q <= 32'h0;
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
`ifndef SFU_SYNTH_NO_DPI
          for (int lane = 0; lane < 8; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q)) begin
              if (integer'(read_idx_q) < integer'(ln_gamma_rows_q))
                gamma_q[base_idx + lane] <= real_to_fp32_bits(fp16_to_real(get_u16(sram_b_rdata, lane)));
              else
                beta_q[base_idx + lane] <= real_to_fp32_bits(fp16_to_real(get_u16(sram_b_rdata, lane)));
            end
          end
`endif

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
`ifndef SFU_SYNTH_NO_DPI
          for (int lane = 0; lane < 16; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  real_to_fp32_bits(sfu_fp32_mul(real'(get_i8(sram_b_rdata, lane)), fp32_bits_to_real(scale0_q)));
          end
`endif

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
`ifndef SFU_SYNTH_NO_DPI
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  real_to_fp32_bits(sfu_fp32_mul(real'(get_i32(sram_b_rdata, lane)), fp32_bits_to_real(scale0_q)));
          end
`endif

          if (read_idx_q + 13'd1 < n_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_I32_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ROW_COMPUTE;
          end
        end

`include "sfu_g1_compute.svh"
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
          if (SFU_SYNTH_MODE == 1) begin
            iter_idx_q <= 11'h0;
            state      <= F_GELU_SYNTH_I8_ITER;
          end else begin
            state      <= F_GELU_I8_WRITE;
          end
        end

        // Phase-3.B gen-1 GELU INT8 synth iterator. 16 cycles per chunk:
        // each cycle picks one lane from gelu_i8_row_q (via gelu_g1_in_pick
        // mux) through the shared synth datapath (cvt → mul scale0 → gelu_new
        // → div scale1 → quantize_i8) and writes out_bytes_q[chunk*16+lane].
        // After 16 lanes done, exits to F_GELU_I8_WRITE (which in synth mode
        // sources sram_a_wdata from row_write_data_w packing out_bytes_q).
        F_GELU_SYNTH_I8_ITER: begin
          if (iter_idx_q < 11'd16) begin
            int byte_idx;
            byte_idx = integer'(write_chunk_q) * 16 + integer'(iter_idx_q[3:0]);
            if (byte_idx < integer'(n_elems_q))
              out_bytes_q[byte_idx[9:0]] <= gelu_g1_quant_w;
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_GELU_I8_WRITE;
          end
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
            if (SFU_SYNTH_MODE == 1) begin
              iter_idx_q <= 11'h0;
              state      <= F_GELU_SYNTH_I32_ITER;
            end else begin
              state      <= F_GELU_I32_WRITE;
            end
          end else begin
            gelu_part_q <= gelu_part_q + 2'd1;
            state       <= F_GELU_I32_REQ;
          end
        end

        // Phase-3.B gen-1 GELU INT32 synth iterator. 16 iterations:
        // iter_idx_q[3:2] = lane (0..3), iter_idx_q[1:0] = row (0..3).
        // Matches the always_comb packing: byte_idx = lane + row * 4
        // within the 16-byte chunk.
        F_GELU_SYNTH_I32_ITER: begin
          if (iter_idx_q < 11'd16) begin
            int byte_idx;
            int lane;
            int row;
            lane = integer'(iter_idx_q[3:2]);
            row  = integer'(iter_idx_q[1:0]);
            byte_idx = integer'(write_chunk_q) * 16 + lane + row * 4;
            if (byte_idx < integer'(n_elems_q))
              out_bytes_q[byte_idx[9:0]] <= gelu_g1_quant_w;
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_GELU_I32_WRITE;
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

`include "sfu_attn.svh"
        F_ATTN_V_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ATTN_V_LATCH;
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
`ifndef SFU_SYNTH_NO_DPI
            for (int lane = 0; lane < 4; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q))
                row_data_q[base_idx + lane] <=
                    real_to_fp32_bits(real'(get_i32(sram_b_rdata, lane)));
            end
`endif
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
              g2_maxabs_q <= mar_new_max;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              real m;
              real v;
              real av;
              m = fp32_bits_to_real(g2_maxabs_q);
              for (int lane = 0; lane < 8; lane++) begin
                if ((base_idx + lane) < integer'(n_elems_q)) begin
                  v  = sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
                  av = (v < 0.0) ? -v : v;
                  if (av > m) m = av;
                end
              end
              g2_maxabs_q <= real_to_fp32_bits(m);
`endif
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
                      synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                else
                  row_data_q[base_idx + lane] <=
                      real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
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
                        synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                  else
                    gamma_q[base_idx + lane] <=
                        real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
                end else begin
                  if (SFU_SYNTH_MODE == 1)
                    beta_q[base_idx + lane] <=
                        synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                  else
                    beta_q[base_idx + lane] <=
                        real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
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
                      synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                else
                  attn_accum_q[base_idx + lane] <=
                      real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
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

`include "sfu_g2_compute.svh"
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
        // Phase-3.B: synth mode sources from row_write_data_w (packed view
        // of out_bytes_q written by F_GELU_SYNTH_I8_ITER); DPI mode uses
        // the always_comb-computed gelu_i8_write_data_w.
        sram_a_wdata = (SFU_SYNTH_MODE == 1) ? row_write_data_w : gelu_i8_write_data_w;
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
        sram_a_wdata = (SFU_SYNTH_MODE == 1) ? row_write_data_w : gelu_i32_write_data_w;
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
`ifndef SFU_SYNTH_NO_DPI
          else
            sfu_scale_wdata = 16'(sfu_fp64_to_fp16_bits(
                127.0 / g2_clamp_eps(fp32_bits_to_real(g2_maxabs_q))));
`endif
        end else begin
          sfu_scale_waddr = sreg_q + 4'd1;
          if (SFU_SYNTH_MODE == 1)
            sfu_scale_wdata = synth_eps_inv127_fp16;
`ifndef SFU_SYNTH_NO_DPI
          else
            sfu_scale_wdata = 16'(sfu_fp64_to_fp16_bits(
                g2_clamp_eps(fp32_bits_to_real(g2_maxabs_q)) / 127.0));
`endif
        end
      end

      default: ;
    endcase
  end

endmodule

`endif // SFU_ENGINE_SV
