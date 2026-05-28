// Synthesizable primitive instances for sfu_engine.sv.
//
// R6 (2026-05-23): extracted from sfu_engine.sv L531-984. Contents:
//   * fp32 bit-pattern <-> real conversion helpers
//     (fp32_bits_to_real, real_to_fp32_bits) — DPI-guarded
//   * Phase-2 G2 shared compute primitive instances
//     (synth_a_bits / synth_b_bits / u_synth_add / u_synth_mul /
//      u_synth_quant / u_synth_gelu / u_synth_f2h, etc.)
//   * 0x1A LAYERNORM_FP32 synth combinational primitives (ln_*)
//   * 0x1D MASKED_SOFTMAX_FP32 synth combinational primitives (sm_*)
//   * Phase-3.B gen-1 GELU synth datapath (gelu_g1_*)
//   * Phase-3.B ATTN V_LATCH parallel 16-lane synth datapath
//   * Phase-2 synth-mode shared latch lanes + 0x1F MAX_ABS_REDUCE
//
// Always elaborated; outputs are unsampled when SFU_SYNTH_MODE=0
// (synth folds the dead logic).

// ---- G2/LN/SM/GELU-g1/ATTN/MAR synth primitives (was sfu_engine.sv L531-L984) ----
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
`ifndef SFU_SYNTH_NO_DPI
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
`endif

  // Per-element fp32 operands fetched from the `real` storage via the
  // bit-pattern coercion (lossless because the stored values are always
  // single-precision floats).
  logic [31:0] synth_a_bits;
  logic [31:0] synth_b_bits;
  assign synth_a_bits = row_data_q[iter_idx_q[9:0]];
  assign synth_b_bits = attn_accum_q[iter_idx_q[9:0]];

  // Op-specific b-operand mux: 0x18 QUANT broadcasts the scalar scale0_q;
  // VADD/DEQUANT-AC use the per-element attn_accum_q. Selected by opcode_q.
  logic [31:0] synth_b_bits_eff;
  always_comb begin
    case (opcode_q)
      OP_QUANT_FP32_INT8: synth_b_bits_eff = scale0_q;
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
      OP_GELU_FP32:                 synth_compute_out = synth_gelu_out;
      default:                      synth_compute_out = 32'd0;
    endcase
  end
  fp32_to_fp16 u_synth_f2h (.a(synth_compute_out), .y(synth_out_bits));

  // For 0x18 QUANT_FP32_INT8 the mul result goes through fp32_quantize_i8
  // (Option-B: NaN->0, ±inf->±127/-128, finite RNE+clamp). out is int8.
  logic signed [7:0] synth_quant_out;
  fp32_quantize_i8 u_synth_quant (.a(synth_mul_out), .y(synth_quant_out));

  // For 0x1B GELU_FP32 the synth datapath is the tanh-poly gen-2 GELU
  // (fp32_gelu_new): depth-9 combinational chain (2*mul + add + mul + mul +
  // exp + add + div + sub + add + mul). MEASURED-BAND op — bounded by the
  // internal fp32_exp scaffold accuracy. SOFTMAX-precedent (86-ULP exp ->
  // 0 fp16 ULP at output via fp16-quant absorption) is the bet.
  logic [31:0] synth_gelu_out;
  fp32_gelu_new u_synth_gelu (.a(synth_a_bits), .y(synth_gelu_out));

  // For 0x1E DEQUANT_ACCUM_FP32_SCALED chain:
  //   out = ((row_data_q * gamma_q) * scale0_q) + beta_q  -> f2h
  // Three combinational stages, then through the shared fp32_to_fp16.
  logic [31:0] synth_gamma_bits;
  logic [31:0] synth_beta_bits;
  logic [31:0] synth_scale0_bits;
  logic [31:0] synth_scaled_mul1;
  logic [31:0] synth_scaled_mul2;
  logic [31:0] synth_scaled_add;
  assign synth_gamma_bits  = gamma_q[iter_idx_q[9:0]];
  assign synth_beta_bits   = beta_q[iter_idx_q[9:0]];
  assign synth_scale0_bits = scale0_q;
  fp32_mul u_synth_scaled_mul1 (
    .a(synth_a_bits),     .b(synth_gamma_bits),  .y(synth_scaled_mul1));
  fp32_mul u_synth_scaled_mul2 (
    .a(synth_scaled_mul1), .b(synth_scale0_bits), .y(synth_scaled_mul2));
  fp32_add u_synth_scaled_add  (
    .a(synth_scaled_mul2), .b(synth_beta_bits),   .y(synth_scaled_add));

  // For 0x1F MAX_ABS_REDUCE F_G2_SCALE_WR phase: replace the DPI fp64 path
  // (127.0/eps and eps/127.0 in fp64, then fp16 cast) with fp32 div + cvt.
  // Phase-3.E (2026-05-21): g2_clamp_eps performed via bit-level magnitude
  // compare on the positive fp32 (synth-safe). Magnitudes of positive fp32
  // numbers compare numerically as unsigned int (IEEE-754 monotonic).
  localparam logic [31:0] C_127_FP32       = 32'h42FE_0000;  // 127.0
  localparam logic [31:0] C_CLAMP_MIN_FP32 = 32'h3B00_0000;  // 2^-9   = 0.001953125
  localparam logic [31:0] C_CLAMP_MAX_FP32 = 32'h4A7D_DC00;  // 4159504.0 = 65504.0*127.0/2.0
  logic [31:0] synth_clamp_eps_bits;
  logic [31:0] synth_inv_eps;
  logic [31:0] synth_eps_inv127;
  logic [15:0] synth_inv_eps_fp16;
  logic [15:0] synth_eps_inv127_fp16;
  assign synth_clamp_eps_bits = (g2_maxabs_q < C_CLAMP_MIN_FP32) ? C_CLAMP_MIN_FP32
                              : (g2_maxabs_q > C_CLAMP_MAX_FP32) ? C_CLAMP_MAX_FP32
                              : g2_maxabs_q;
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
  localparam logic [31:0] C_LN_FP32_EPS = 32'h3727_C5AC;  // 1.0e-5 fp32 (gen-2)
  localparam logic [31:0] C_LN_EPS_G1   = 32'h358D_3F3F;  // 1.0e-6 fp32 (gen-1)
  logic [31:0] ln_neg_mean;
  logic [31:0] ln_n_fp32;
  logic [31:0] ln_sum_add_w;
  logic [31:0] ln_mean_div_w;
  logic [31:0] ln_diff_w;
  logic [31:0] ln_diff_sq_w;
  logic [31:0] ln_var_add_w;
  logic [31:0] ln_var_norm_w;
  logic [31:0] ln_var_eps_w;
  logic [31:0] ln_eps_sel_w;
  logic [31:0] ln_denom_w;
  logic [31:0] ln_norm_w;
  logic [31:0] ln_norm_g_w;
  logic [31:0] ln_norm_gb_w;
  logic [15:0] ln_out_h_w;
  // Phase-3.B gen-1 LN output path: y/scale1 then quantize_i8 (matches DPI
  // sfu_fp32_quantize_i8 contract — DIVIDE by scale, RNE round, clamp).
  logic [31:0]       ln_g1_scaled_w;
  logic signed [7:0] ln_g1_quant_w;
  assign ln_neg_mean = ln_mean_q ^ 32'h8000_0000;
  // n_elems_q (16-bit unsigned) → fp32 via i32_to_fp32 primitive (synth-safe).
  i32_to_fp32 u_ln_n_cvt (.a({16'h0, n_elems_q}), .y(ln_n_fp32));
  // Gen-1 OP_LAYERNORM is illegal at decode_unit; only gen-2 LAYERNORM_FP32
  // reaches here. Drop the mux and pin to C_LN_FP32_EPS (1e-5).
  assign ln_eps_sel_w = C_LN_FP32_EPS;

  fp32_add  u_ln_sum_add (.a(ln_sum_acc_q), .b(synth_a_bits), .y(ln_sum_add_w));
  fp32_div  u_ln_mean    (.a(ln_sum_acc_q), .b(ln_n_fp32),    .y(ln_mean_div_w));
  fp32_add  u_ln_diff    (.a(synth_a_bits), .b(ln_neg_mean),  .y(ln_diff_w));
  fp32_mul  u_ln_diff_sq (.a(ln_diff_w),    .b(ln_diff_w),    .y(ln_diff_sq_w));
  fp32_add  u_ln_var_add (.a(ln_var_acc_q), .b(ln_diff_sq_w), .y(ln_var_add_w));
  fp32_div  u_ln_var_norm(.a(ln_var_acc_q), .b(ln_n_fp32),    .y(ln_var_norm_w));
  fp32_add  u_ln_var_eps (.a(ln_var_norm_w),.b(ln_eps_sel_w), .y(ln_var_eps_w));
  // Phase-4 pipeline cut: sqrt reads the registered ln_var_eps_q (latched in
  // F_G2_LN_DENOM_PRE one cycle earlier), not the combinational ln_var_eps_w.
  fp32_sqrt u_ln_sqrt    (.a(ln_var_eps_q),                    .y(ln_denom_w));
  fp32_div  u_ln_norm    (.a(ln_diff_w),    .b(ln_denom_q),   .y(ln_norm_w));
  // Phase-4 pipeline cut: ln_norm_g multiplier reads the registered ln_norm_q
  // (latched in F_G2_LN_OUT_NORM), not the combinational ln_norm_w.
  fp32_mul  u_ln_norm_g  (.a(ln_norm_q),    .b(synth_gamma_bits), .y(ln_norm_g_w));
  fp32_add  u_ln_norm_gb (.a(ln_norm_g_w),  .b(synth_beta_bits),  .y(ln_norm_gb_w));
  fp32_to_fp16 u_ln_out_h(.a(ln_norm_gb_w),                        .y(ln_out_h_w));
  fp32_div  u_ln_g1_scale(.a(ln_norm_gb_w), .b(synth_scale1_bits), .y(ln_g1_scaled_w));
  fp32_quantize_i8 u_ln_g1_quant (.a(ln_g1_scaled_w),               .y(ln_g1_quant_w));

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

  // Phase-3.B: gen-1 SOFTMAX/MASKED_SOFTMAX synth output path.
  //   gen-2 0x1D MASKED_SOFTMAX_FP32 writes out_h_q[i] = fp16(norm).
  //   gen-1 0x0E SOFTMAX / 0x0F MASKED_SOFTMAX writes
  //         out_bytes_q[i] = quantize_i8(norm / scale1_q).
  // DPI golden (testbench.h `sfu_fp32_quantize_i8`) computes
  //   q = round_half_even((float)value / (float)out_scale)
  // — DIVIDE by scale, not multiply. The synth chain mirrors that exactly:
  // fp32_div then fp32_quantize_i8. Note: matches per-rounding-step boundary
  // behavior only if we don't introduce an intermediate fp32 round between
  // the divide and the int8 quant; both are bit-exact RNE primitives so the
  // composition is byte-identical when the upstream sm_norm_w matches.
  logic [31:0]       synth_scale1_bits;
  logic [31:0]       sm_g1_scaled_w;
  logic signed [7:0] sm_g1_quant_w;
  assign synth_scale1_bits = scale1_q;
  fp32_div         u_sm_g1_scale (.a(sm_norm_w),       .b(synth_scale1_bits), .y(sm_g1_scaled_w));
  fp32_quantize_i8 u_sm_g1_quant (.a(sm_g1_scaled_w),                          .y(sm_g1_quant_w));

  // Phase-3.B gen-1 GELU synth datapath. Sequential ITER over 16 lanes per
  // chunk (or 4 i32-lanes × 4 rows for the i32 GELU). Computes:
  //   x = (sign_ext(i8_lane) -> fp32) * scale0_q
  //   y = fp32_gelu_new(x)              (tanh-poly approximation of erf-GELU)
  //   q = quantize_i8(y / scale1_q)
  // Approximation note: gen-1 DPI uses erf-GELU (sfu_fp32_gelu). The synth
  // path uses gelu_new (tanh-poly) because that's the only synthesizable
  // GELU primitive available (Phase-1 unit #10 fp32_gelu_erf isn't built
  // yet). For the byte-quantized output the int8 round absorbs the tanh-vs-
  // erf difference on the fixture inputs (verified by test_sfu_synth
  // gelu_*_roundtrip).
  logic signed [7:0] gelu_g1_i8_sel;
  logic [31:0]       gelu_g1_i32_sel;
  logic [31:0]       gelu_g1_in_fp32;
  logic [31:0]       gelu_g1_x_bits;
  logic [31:0]       gelu_g1_y_bits;
  logic [31:0]       gelu_g1_scaled_w;
  logic signed [7:0] gelu_g1_quant_w;
  logic [31:0]       gelu_g1_in_pick;
  // synth_scale0_bits is already declared module-scope (DEQUANT_ACCUM_SCALED
  // block). Reuse it here.
  // INT8 source: lane comes from gelu_i8_row_q indexed by iter_idx_q[3:0].
  assign gelu_g1_i8_sel  = $signed(gelu_i8_row_q[8*iter_idx_q[3:0] +: 8]);
  // INT32 source: lane select by iter_idx_q[3:2] (0..3), row by iter_idx_q[1:0]
  // (0..3). Matches the original always_comb packing where (lane, row)
  // packs to byte index (lane + row*4) in gelu_i32_write_data_w.
  always_comb begin
    case (iter_idx_q[1:0])
      2'd0: gelu_g1_i32_sel = gelu_row0_q[32*iter_idx_q[3:2] +: 32];
      2'd1: gelu_g1_i32_sel = gelu_row1_q[32*iter_idx_q[3:2] +: 32];
      2'd2: gelu_g1_i32_sel = gelu_row2_q[32*iter_idx_q[3:2] +: 32];
      default: gelu_g1_i32_sel = gelu_row3_q[32*iter_idx_q[3:2] +: 32];
    endcase
  end
  // Combine the two source selections — the active one is decided by the
  // state (F_GELU_SYNTH_I8_ITER vs F_GELU_SYNTH_I32_ITER).
  assign gelu_g1_in_pick = (state == F_GELU_SYNTH_I32_ITER) ? gelu_g1_i32_sel
                                                            : {{24{gelu_g1_i8_sel[7]}}, gelu_g1_i8_sel};
  i32_to_fp32      u_gelu_g1_cvt  (.a(gelu_g1_in_pick),                                  .y(gelu_g1_in_fp32));
  fp32_mul         u_gelu_g1_x    (.a(gelu_g1_in_fp32), .b(synth_scale0_bits),           .y(gelu_g1_x_bits));
  fp32_gelu_new    u_gelu_g1_y    (.a(gelu_g1_x_bits),                                   .y(gelu_g1_y_bits));
  fp32_div         u_gelu_g1_s    (.a(gelu_g1_y_bits),  .b(synth_scale1_bits),           .y(gelu_g1_scaled_w));
  fp32_quantize_i8 u_gelu_g1_q    (.a(gelu_g1_scaled_w),                                 .y(gelu_g1_quant_w));

  // Opcode-aware SOFTMAX visibility predicate. The F_G2_SM_* sub-FSM tests
  // sm_visible_w in every state, replacing the inline kt-comparison so the
  // same sub-FSM handles all softmax-family opcodes (gen-1 + gen-2 + ATTN).
  //   OP_SOFTMAX:               all iters visible (gen-1, unmasked)
  //   OP_MASKED_SOFTMAX:        attn_visible(row, iter) (gen-1, gen-1 mask)
  //   OP_MASKED_SOFTMAX_FP32:   iter_s <= sm_keep_through_q (gen-2 causal)
  //   OP_SOFTMAX_ATTNV:         all iters visible (gen-1 ATTN unmasked)
  //   OP_MASKED_SOFTMAX_ATTNV:  attn_visible(row, iter) (gen-1 ATTN mask)
  logic sm_visible_w;
  always_comb begin
    case (opcode_q)
      OP_SOFTMAX:                sm_visible_w = 1'b1;
      OP_MASKED_SOFTMAX:         sm_visible_w = attn_visible(row_idx_q, integer'(iter_idx_q));
      OP_MASKED_SOFTMAX_FP32:    sm_visible_w =
          ($signed({6'b0, iter_idx_q}) <= $signed({1'b0, sm_keep_through_q[15:0]}));
      OP_SOFTMAX_ATTNV:          sm_visible_w = 1'b1;
      OP_MASKED_SOFTMAX_ATTNV:   sm_visible_w = attn_visible(row_idx_q, integer'(iter_idx_q));
      default:                   sm_visible_w = 1'b0;
    endcase
  end
  // Phase-3.B ATTN: bound is k_elems_q (column count) instead of n_elems_q.
  // Bound mux keeps the F_G2_SM_* sub-FSM iteration logic single-source.
  logic [15:0] sm_iter_bound_w;
  always_comb begin
    case (opcode_q)
      OP_SOFTMAX_ATTNV, OP_MASKED_SOFTMAX_ATTNV: sm_iter_bound_w = k_elems_q;
      default:                                   sm_iter_bound_w = n_elems_q;
    endcase
  end

  // Phase-3.B ATTN V_LATCH parallel 16-lane synth datapath. Each cycle:
  //   weight   = exp(row_data_q[k_idx] - row_max) / exp_sum   (visible only)
  //   per lane: attn_accum_q[idx] += weight * sign_ext(v_lane) * scale1_q
  // Per-K weight compute is shared; 16 lanes parallel for the V-multiply
  // and accumulate. Mirrors the DPI V_LATCH (1 cycle per chunk) — no
  // new state needed; F_ATTN_V_LATCH muxes synth vs DPI on SFU_SYNTH_MODE.
  logic [31:0] attn_row_at_k_bits;
  logic [31:0] attn_diff_w;
  logic [31:0] attn_exp_w;
  logic [31:0] attn_weight_w;
  logic [31:0] attn_weight_eff_w;
  logic        attn_vis_at_k;
  assign attn_row_at_k_bits = row_data_q[attn_k_idx_q[9:0]];
  fp32_add u_attn_diff (.a(attn_row_at_k_bits), .b(sm_neg_max),   .y(attn_diff_w));
  fp32_exp u_attn_exp  (.a(attn_diff_w),                          .y(attn_exp_w));
  fp32_div u_attn_div  (.a(attn_exp_w), .b(sm_exp_sum_q),         .y(attn_weight_w));
  assign attn_vis_at_k = (opcode_q == OP_SOFTMAX_ATTNV) ||
                         attn_visible(row_idx_q, integer'(attn_k_idx_q));
  assign attn_weight_eff_w = attn_vis_at_k ? attn_weight_w : 32'h0;

  logic [31:0] attn_v_lane_fp32    [0:15];
  logic [31:0] attn_v_weighted     [0:15];
  logic [31:0] attn_v_scaled       [0:15];
  logic [31:0] attn_acc_old_bits   [0:15];
  logic [31:0] attn_acc_new_bits   [0:15];
  genvar gv_lane;
  generate
    for (gv_lane = 0; gv_lane < 16; gv_lane++) begin : v_lane
      logic signed [7:0] byte_sel;
      logic [31:0]       byte_sx;
      assign byte_sel = $signed(sram_b_rdata[8*gv_lane +: 8]);
      assign byte_sx  = {{24{byte_sel[7]}}, byte_sel};
      i32_to_fp32 u_v_cvt (.a(byte_sx),                .y(attn_v_lane_fp32[gv_lane]));
      fp32_mul    u_v_w   (.a(attn_weight_eff_w),
                           .b(attn_v_lane_fp32[gv_lane]), .y(attn_v_weighted[gv_lane]));
      fp32_mul    u_v_s   (.a(attn_v_weighted[gv_lane]),
                           .b(synth_scale1_bits),         .y(attn_v_scaled[gv_lane]));
    end
  endgenerate
  always_comb begin
    for (int li = 0; li < 16; li++) begin
      automatic int idx_li;
      idx_li = integer'(read_idx_q) * 16 + li;
      attn_acc_old_bits[li] = attn_accum_q[idx_li[9:0]];
    end
  end
  generate
    for (gv_lane = 0; gv_lane < 16; gv_lane++) begin : v_acc
      fp32_add u_v_add (.a(attn_acc_old_bits[gv_lane]),
                        .b(attn_v_scaled[gv_lane]),
                        .y(attn_acc_new_bits[gv_lane]));
    end
  endgenerate
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
  assign mar_curr_bits = g2_maxabs_q & 32'h7FFF_FFFF;
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
