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
      // 0x17 / 0x1E read the PIPELINED results (fmax phase 0e) — both are
      // handled only by F_G2_DQL, which indexes out_h_q by its collect pointer.
      OP_DEQUANT_ACCUM_FP32:        synth_compute_out = synth_mul_d3_q;
      OP_DEQUANT_ACCUM_FP32_SCALED: synth_compute_out = synth_sc_s_q;
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
  // (2*mul + add + mul + mul + exp + add + div + mul). MEASURED-BAND op —
  // bounded by the internal fp32_exp scaffold accuracy. SOFTMAX-precedent
  // (86-ULP exp -> 0 fp16 ULP at output via fp16-quant absorption) is the bet.
  //
  // 2026-07-21 (fmax phase 0d): this was `fp32_gelu_new`, a ~700 ns SINGLE-
  // CYCLE combinational cloud — the largest in the SFU, containing a whole
  // fp32_exp and a whole fp32_div, ×8 lanes, and never once in a timing report.
  // It is now the 33-stage `fp32_gelu_p33`, a pure retiming of that same core
  // (bit-identical, gated by test_fp32_gelu_p33's 8.5M-vector zero-diff run).
  // CONSUMER CONTRACT: this output is now REGISTERED and arrives LATENCY=33
  // cycles after its operand was presented. The only state that drives it is
  // F_G2_GLC, which feeds `synth_a_bits` at iter_idx_q and collects at the
  // lagging gelu_coll_q, gated by synth_gelu_vo — never by a hardcoded 33.
  // All 8 lanes share one valid_in, so lane 0's valid_out speaks for all of
  // them; the replicas sink theirs. F_G2_SYNTH_ITER must NOT be used for
  // OP_GELU_FP32 any more (it reads the op-mux in-cycle); its one dispatch
  // site is mode-1-dead and now faults — see sfu_g2_compute.svh F_G2_COMPUTE.
  logic [31:0] synth_gelu_out;
  logic        synth_gelu_vo;
  fp32_gelu_p33 u_synth_gelu (
    .clk(clk), .rst_n(rst_n), .valid_in(gelu_feed_en_w),
    .a(synth_a_bits), .valid_out(synth_gelu_vo), .y(synth_gelu_out));

  // For 0x1E DEQUANT_ACCUM_FP32_SCALED chain:
  //   out = ((row_data_q * gamma_q) * scale0_q) + beta_q  -> f2h
  //
  // 2026-07-21 (fmax phase 0e): this was THREE combinational stages in ONE
  // cycle, then f2h — measured 86.59 ns standalone, ~3x the primitive floor,
  // LIVE on every b16 step (it is the DEQUANT_SCALED that follows each matmul).
  // Now cut so each stage carries one primitive:
  //   T   mul1 = a * gamma          -> synth_sc_m1_q
  //   T+1 mul2 = m1 * scale0        -> synth_sc_m2_q
  //   T+2 add  = m2 + beta(delayed) -> synth_sc_s_q
  //   T+3 the shared f2h, at the COLLECT index in F_G2_DQL
  //
  // BETA MUST BE DELAYED BY 2. gamma and `a` are read at the FEED index, but
  // the add fires two cycles later, when iter_idx_q has already moved on —
  // beta_q[iter_idx_q] at that moment belongs to a DIFFERENT element. This is
  // the same alignment hazard as fp32_gelu_p33's x delay line, and it is silent:
  // it would corrupt every element by a fixed index skew while still producing
  // plausible-looking numbers. scale0_q needs no delay (scalar, constant for
  // the instruction).
  logic [31:0] synth_gamma_bits;
  logic [31:0] synth_beta_bits;
  logic [31:0] synth_scale0_bits;
  logic [31:0] synth_scaled_mul1;
  logic [31:0] synth_scaled_mul2;
  logic [31:0] synth_scaled_add;
  logic [31:0] synth_sc_m1_q, synth_sc_m2_q, synth_sc_s_q;
  logic [31:0] synth_beta_d1_q, synth_beta_d2_q;
  assign synth_gamma_bits  = gamma_q[iter_idx_q[9:0]];
  assign synth_beta_bits   = beta_q[iter_idx_q[9:0]];
  assign synth_scale0_bits = scale0_q;
  fp32_mul u_synth_scaled_mul1 (
    .a(synth_a_bits),      .b(synth_gamma_bits),  .y(synth_scaled_mul1));
  fp32_mul u_synth_scaled_mul2 (
    .a(synth_sc_m1_q),     .b(synth_scale0_bits), .y(synth_scaled_mul2));
  fp32_add u_synth_scaled_add  (
    .a(synth_sc_m2_q),     .b(synth_beta_d2_q),   .y(synth_scaled_add));

  // For 0x17 DEQUANT_ACCUM_FP32 (the other op F_G2_DQL serves): its compute is
  // a single fp32_mul and needs no pipelining, but it shares F_G2_DQL's collect
  // pointer, so its result is delayed to the SAME depth. Cheaper than giving
  // the state two write indices, and far less error-prone.
  logic [31:0] synth_mul_d1_q, synth_mul_d2_q, synth_mul_d3_q;

  // Free-running (validity comes from F_G2_DQL's valid chain, see dq_vld_q).
  always_ff @(posedge clk) begin
    synth_sc_m1_q   <= synth_scaled_mul1;
    synth_sc_m2_q   <= synth_scaled_mul2;
    synth_sc_s_q    <= synth_scaled_add;
    synth_beta_d1_q <= synth_beta_bits;
    synth_beta_d2_q <= synth_beta_d1_q;
    synth_mul_d1_q  <= synth_mul_out;
    synth_mul_d2_q  <= synth_mul_d1_q;
    synth_mul_d3_q  <= synth_mul_d2_q;
  end

  // ===================================================================
  // 8-wide SIMD widening of the F_G2_SYNTH_ITER elementwise loop.
  // ===================================================================
  // Lanes 1..7 replicate the lane-0 compute (add / mul / quant / gelu /
  // scaled-chain / f2h) at element index (iter_idx_q + lane). Lane 0 is
  // kept as the existing synth_out_bits / synth_quant_out above (also
  // reused by the LN/SM datapath via synth_a_bits), so only lanes 1..7
  // are added here. Each lane's compute is identical & independent, so
  // the 8-wide writeback in F_G2_SYNTH_ITER is bit-exact vs. the old
  // 1/cycle sequential loop. Per-lane writes are gated by the FSM on
  // (iter_idx_q + lane) < n_elems_q, so out-of-range lanes never write.
  //
  // synth_out_bits_lane[0] / synth_quant_out_lane[0] alias the existing
  // lane-0 outputs so the FSM can index lanes 0..7 uniformly.
  logic [15:0]       synth_out_bits_lane  [0:7];
  logic signed [7:0] synth_quant_out_lane [0:7];
  assign synth_out_bits_lane[0]  = synth_out_bits;
  assign synth_quant_out_lane[0] = synth_quant_out;

  genvar gv_simd;
  generate
    for (gv_simd = 1; gv_simd < 8; gv_simd = gv_simd + 1) begin : g_simd_lane
      // Element index for this lane (wraps in 11-bit; the FSM guard keeps
      // only valid lanes, and iter_idx_q+lane <= 1024+7 fits 11 bits).
      logic [10:0] lane_idx;
      assign lane_idx = iter_idx_q + 11'(gv_simd);

      // Per-lane operands (same coercion as lane 0).
      logic [31:0] a_bits;
      logic [31:0] b_bits;
      logic [31:0] b_bits_eff;
      logic [31:0] gamma_bits;
      logic [31:0] beta_bits;
      assign a_bits     = row_data_q[lane_idx[9:0]];
      assign b_bits     = attn_accum_q[lane_idx[9:0]];
      assign gamma_bits = gamma_q[lane_idx[9:0]];
      assign beta_bits  = beta_q[lane_idx[9:0]];
      always_comb begin
        case (opcode_q)
          OP_QUANT_FP32_INT8: b_bits_eff = scale0_q;
          default:            b_bits_eff = b_bits;
        endcase
      end

      // Arithmetic primitives (replicas of u_synth_add / u_synth_mul).
      logic [31:0] add_out;
      logic [31:0] mul_out;
      fp32_add u_add (.a(a_bits), .b(b_bits),     .y(add_out));
      fp32_mul u_mul (.a(a_bits), .b(b_bits_eff), .y(mul_out));

      // DEQUANT_ACCUM_FP32_SCALED chain (replica of the scaled chain), and the
      // 0x17 mul delay — both pipelined in lockstep with lane 0 (fmax phase
      // 0e). Note beta rides a 2-deep delay so it stays aligned with the add,
      // which fires two cycles after gamma/a were read; see the lane-0 comment.
      logic [31:0] scaled_mul1;
      logic [31:0] scaled_mul2;
      logic [31:0] scaled_add;
      logic [31:0] sc_m1_q, sc_m2_q, sc_s_q;
      logic [31:0] beta_d1_q, beta_d2_q;
      logic [31:0] mul_d1_q, mul_d2_q, mul_d3_q;
      fp32_mul u_scaled_mul1 (.a(a_bits),   .b(gamma_bits),  .y(scaled_mul1));
      fp32_mul u_scaled_mul2 (.a(sc_m1_q),  .b(scale0_q),    .y(scaled_mul2));
      fp32_add u_scaled_add  (.a(sc_m2_q),  .b(beta_d2_q),   .y(scaled_add));
      always_ff @(posedge clk) begin
        sc_m1_q   <= scaled_mul1;
        sc_m2_q   <= scaled_mul2;
        sc_s_q    <= scaled_add;
        beta_d1_q <= beta_bits;
        beta_d2_q <= beta_d1_q;
        mul_d1_q  <= mul_out;
        mul_d2_q  <= mul_d1_q;
        mul_d3_q  <= mul_d2_q;
      end

      // GELU replica (identical to lane-0 u_synth_gelu). 2026-07-08: widened
      // to 8 lanes. The old "area hog" guess was falsified by the 2026-05-30
      // trim measurement (removing 7 gelu replicas saved only 0.5% of SFU
      // cells, 1,287,144 -> 1,280,022; the real area went to the scaled-chain
      // replicas), so GELU strides 8 like every other elementwise op
      // (~-560K mode-1 cyc = +2.3% tok/s; docs/perf_roadmap_2026-07-08.md #4).
      // 2026-07-21 (fmax phase 0d): pipelined to fp32_gelu_p33 in lockstep
      // with lane 0 — same clk, same valid_in, so all 8 lanes present element
      // (gelu_coll_q + lane) on the same cycle. Each lane is still an
      // identical, independent core, so this stays bit-exact vs both the
      // combinational 8-wide version and the original lane-0-only 1/cycle loop.
      // valid_out is sunk here; lane 0's synth_gelu_vo is the shared collect
      // strobe (an unused-but-driven net per lane would be a phase-0a dead
      // cone, so the FSM deliberately reads ONE of them, not eight).
      logic [31:0] gelu_out;
      logic        gelu_vo_unused;
      fp32_gelu_p33 u_gelu (
        .clk(clk), .rst_n(rst_n), .valid_in(gelu_feed_en_w),
        .a(a_bits), .valid_out(gelu_vo_unused), .y(gelu_out));

      // Compute-output op-mux -> shared f2h (replica of synth_compute_out).
      logic [31:0] compute_out;
      always_comb begin
        case (opcode_q)
          OP_VADD_FP32:                 compute_out = add_out;
          OP_DEQUANT_ACCUM_FP32:        compute_out = mul_d3_q;
          OP_DEQUANT_ACCUM_FP32_SCALED: compute_out = sc_s_q;
          OP_GELU_FP32:                 compute_out = gelu_out;
          default:                      compute_out = 32'd0;
        endcase
      end
      fp32_to_fp16 u_f2h (.a(compute_out), .y(synth_out_bits_lane[gv_simd]));

      // QUANT_FP32_INT8 path (replica of u_synth_quant).
      fp32_quantize_i8 u_quant (.a(mul_out), .y(synth_quant_out_lane[gv_simd]));
    end
  endgenerate

  // For 0x1F MAX_ABS_REDUCE F_G2_SCALE_WR phase: replace the DPI fp64 path
  // (127.0/eps and eps/127.0 in fp64, then fp16 cast) with fp32 div + cvt.
  // Phase-3.E (2026-05-21): g2_clamp_eps performed via bit-level magnitude
  // compare on the positive fp32 (synth-safe). Magnitudes of positive fp32
  // numbers compare numerically as unsigned int (IEEE-754 monotonic).
  //
  // 2026-07-24 (fmax MAX_ABS fix): these two divides were plain combinational
  // `fp32_div` — a full 29-iteration restoring divide in ONE cycle, measured
  // 175.33 ns on the mode-1 whole-lane netlist (reg -> clamp -> div -> f2h ->
  // sfu_scale_wdata port -> top-level scale regfile). It never appeared in any
  // reg-to-reg min-period report because the path ends at a PORT. Now:
  //   cycle 1 (SCALE_WR entry): synth_eps_q <= clamp(g2_maxabs_q)  [final max]
  //   cycle 2: mar_feed_q one-shot -> both fp32_div_p6 (bit-identical divide)
  //   cycle 8: valid_out -> collect regs; F_G2_SCALE_WR writes on mar_vld_q.
  // The clamp is registered so div stage 1 keeps its standalone depth, and the
  // writes read f2h off the COLLECT regs (reg -> f2h -> port ≈ short). The op
  // is rare (745 ops per 124M decode step at BOTH batch shapes — counted in
  // the decode instr stream), so the +8 cycles/op tail is +5,960 cyc/step =
  // +0.012% b16 / +0.033% b1, measured exact (Δtotal == Δsfu_busy == 8x745). Sequencing lives in F_G2_SCALE_WR
  // (sfu_engine.sv): mar_arm_q -> mar_feed_q -> mar_vld_q, gated on the
  // pipe's valid chain, never a hardcoded latency.
  localparam logic [31:0] C_127_FP32       = 32'h42FE_0000;  // 127.0
  localparam logic [31:0] C_CLAMP_MIN_FP32 = 32'h3B00_0000;  // 2^-9   = 0.001953125
  localparam logic [31:0] C_CLAMP_MAX_FP32 = 32'h4A7D_DC00;  // 4159504.0 = 65504.0*127.0/2.0
  logic [31:0] synth_clamp_eps_bits;
  logic [31:0] synth_eps_q;
  logic [31:0] synth_inv_eps;
  logic [31:0] synth_eps_inv127;
  logic [31:0] synth_inv_eps_q;
  logic [31:0] synth_eps_inv127_q;
  logic        synth_mar_div_vo;
  logic        synth_mar_div_vo2;
  logic [15:0] synth_inv_eps_fp16;
  logic [15:0] synth_eps_inv127_fp16;
  assign synth_clamp_eps_bits = (g2_maxabs_q < C_CLAMP_MIN_FP32) ? C_CLAMP_MIN_FP32
                              : (g2_maxabs_q > C_CLAMP_MAX_FP32) ? C_CLAMP_MAX_FP32
                              : g2_maxabs_q;
  fp32_div_p6 u_synth_inv_eps (
    .clk(clk), .rst_n(rst_n), .valid_in(mar_feed_q),
    .a(C_127_FP32), .b(synth_eps_q),
    .valid_out(synth_mar_div_vo), .y(synth_inv_eps));
  fp32_div_p6 u_synth_eps_inv127 (
    .clk(clk), .rst_n(rst_n), .valid_in(mar_feed_q),
    .a(synth_eps_q), .b(C_127_FP32),
    .valid_out(synth_mar_div_vo2), .y(synth_eps_inv127));
  // Free-running: eps tracks the clamp one cycle behind g2_maxabs_q (stable by
  // the feed cycle); each collect reg samples y exactly when its pipe says so.
  always_ff @(posedge clk) begin
    synth_eps_q <= synth_clamp_eps_bits;
    if (synth_mar_div_vo)  synth_inv_eps_q    <= synth_inv_eps;
    if (synth_mar_div_vo2) synth_eps_inv127_q <= synth_eps_inv127;
  end
  fp32_to_fp16 u_synth_inv_eps_h    (.a(synth_inv_eps_q),    .y(synth_inv_eps_fp16));
  fp32_to_fp16 u_synth_eps_inv127_h (.a(synth_eps_inv127_q), .y(synth_eps_inv127_fp16));

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
  assign ln_neg_mean = ln_mean_q ^ 32'h8000_0000;
  // n_elems_q (16-bit unsigned) → fp32 via i32_to_fp32 primitive (synth-safe).
  i32_to_fp32 u_ln_n_cvt (.a({16'h0, n_elems_q}), .y(ln_n_fp32));
  // Gen-1 OP_LAYERNORM is illegal at decode_unit; only gen-2 LAYERNORM_FP32
  // reaches here. Drop the mux and pin to C_LN_FP32_EPS (1e-5).
  assign ln_eps_sel_w = C_LN_FP32_EPS;

  // valid_out of the pipelined dividers is unused (fixed-LATENCY=6 use; the
  // FSM samples y a fixed number of cycles after presenting operands). Named
  // dummy sinks keep PINCONNECTEMPTY / sv2v / yosys quiet.
  logic ln_mean_vo, ln_var_norm_vo, ln_norm_vo, sm_div_vo, ln_sqrt_vo;
  fp32_add  u_ln_sum_add (.a(ln_sum_acc_q), .b(synth_a_bits), .y(ln_sum_add_w));
  // 2026-07-12 (lever E): 6-stage pipelined divider (LATENCY=6, fp32_div_p6). FSM
  // presents the registered ln_sum_acc_q and samples ln_mean_div_w 6 cycles later
  // (F_G2_LN_MEAN -> _W -> _W2 -> _W3 -> _W4 -> _W5 -> _S). valid_in tied high.
  fp32_div_p6 u_ln_mean  (.clk(clk), .rst_n(rst_n), .valid_in(1'b1),
                          .a(ln_sum_acc_q), .b(ln_n_fp32),
                          .valid_out(ln_mean_vo), .y(ln_mean_div_w));
  fp32_add  u_ln_diff    (.a(synth_a_bits), .b(ln_neg_mean),  .y(ln_diff_w));
  fp32_mul  u_ln_diff_sq (.a(ln_diff_w),    .b(ln_diff_w),    .y(ln_diff_sq_w));
  // 2026-05-31: LN_VAR pipelined. The accumulate reads the REGISTERED square
  // (ln_dsq_q, latched in F_G2_LN_VAR from ln_diff_sq_w) instead of the
  // combinational ln_diff_sq_w, so sub+mul (-> ln_dsq_q) and the loop-carried
  // accumulate (-> ln_var_acc_q) sit in separate pipeline stages. Breaks the
  // ~42 ns sub->mul->add SFU floor into ~28 ns + ~14 ns. Bit-exact.
  fp32_add  u_ln_var_add (.a(ln_var_acc_q), .b(ln_dsq_q),     .y(ln_var_add_w));
  // 2026-07-12 (lever E): 6-stage divider (LATENCY=6, fp32_div_p6). var_acc/n
  // sampled 6 cycles after the registered ln_var_acc_q is presented
  // (F_G2_LN_DENOM_PRE -> _W -> _W2 -> _W3 -> _W4 -> _W5 -> _S); the +eps add then
  // sits after the divider output register.
  fp32_div_p6 u_ln_var_norm(.clk(clk), .rst_n(rst_n), .valid_in(1'b1),
                            .a(ln_var_acc_q), .b(ln_n_fp32),
                            .valid_out(ln_var_norm_vo), .y(ln_var_norm_w));
  fp32_add  u_ln_var_eps (.a(ln_var_norm_w),.b(ln_eps_sel_w), .y(ln_var_eps_w));
  // 2026-06-01: 6-stage pipelined sqrt. After sqrt_p4 + div_p5, the floor was
  // sqrt_p4 STAGE-3 (u_ln_sqrt.rC_r, iters 7..4, 31.54 ns) with STAGE-1 (unpack +
  // iters 24..12) co-binding at 31.07 ns — a tail-only split couldn't beat the
  // pinned stage 1. fp32_sqrt_p6 ISOLATES the unpack into its own stage and
  // splits the iter tail across 5 stages (12/10/7/4 split, ~32 ns standalone
  // floor, the iters-24..12 stage 2 being the pinned limiter), dropping the sqrt
  // to the div_p5 tier (~28 ns PNR). Reads the registered ln_var_eps_q (latched
  // in F_G2_LN_DENOM_PRE_S); ln_denom_w (= sqrt y) is valid in F_G2_LN_DENOM_S,
  // 6 cycles after F_G2_LN_DENOM presents it. valid_in tied high; valid_out unused.
  fp32_sqrt_p6 u_ln_sqrt (.clk(clk), .rst_n(rst_n), .valid_in(1'b1),
                          .a(ln_var_eps_q),
                          .valid_out(ln_sqrt_vo), .y(ln_denom_w));
  // 2026-07-12 (lever E): 6-stage pipelined divider (LATENCY=6, fp32_div_p6).
  // Consumes the registered ln_diff_q in the software-pipelined F_G2_LN_OUT_DIFF
  // drain; the collect pointer ln_coll_q = iter_idx_q - 7 (1 ln_diff_q feed reg +
  // 6 div stages). The downstream gamma multiply reads ln_norm_w directly.
  fp32_div_p6 u_ln_norm  (.clk(clk), .rst_n(rst_n), .valid_in(1'b1),
                          .a(ln_diff_q),    .b(ln_denom_q),
                          .valid_out(ln_norm_vo), .y(ln_norm_w));
  // 2026-05-31: LN_OUT divider-drain pipelining. The gamma/beta applied to a
  // given divider output (ln_norm_w) belong to the COLLECT element (ln_coll_q =
  // iter_idx_q-7 with div_p6), not the feed element (iter_idx_q). Index them by ln_coll_q so the multiply
  // -add matches the element emerging from u_ln_norm this cycle. (synth_gamma/
  // beta_bits stay iter_idx_q-indexed for the other ops that use them.)
  logic [31:0] ln_gamma_coll_w, ln_beta_coll_w;
  assign ln_gamma_coll_w = gamma_q[ln_coll_q[9:0]];
  assign ln_beta_coll_w  = beta_q [ln_coll_q[9:0]];
  // 2026-07-21 (fmax phase 0e): the collect side was mul -> add -> f2h in ONE
  // cycle behind the divider's registered quotient — 61.07 ns standalone, the
  // last un-timed LIVE cloud after the SCALED chain. Now one primitive per
  // stage, so the collect walks THREE cycles behind the divider output:
  //   c   mul  = ln_norm_w * gamma[ln_coll_q]   -> ln_g_q
  //   c+1 add  = ln_g_q    + beta(delayed 1)    -> ln_gb_q
  //   c+2 f2h  = ln_gb_q                        -> out_h_q[ln_wr_q]
  //
  // BETA DELAY, same hazard as the SCALED chain: gamma is read when the
  // quotient emerges, but the add fires a cycle later, by which time ln_coll_q
  // has advanced and beta_q[ln_coll_q] is a DIFFERENT element's bias. gamma
  // needs no delay (it is consumed in the same cycle it is read); beta does.
  logic [31:0] ln_g_q, ln_gb_q, ln_beta_d1_q;
  fp32_mul  u_ln_norm_g  (.a(ln_norm_w), .b(ln_gamma_coll_w), .y(ln_norm_g_w));
  fp32_add  u_ln_norm_gb (.a(ln_g_q),    .b(ln_beta_d1_q),    .y(ln_norm_gb_w));
  fp32_to_fp16 u_ln_out_h(.a(ln_gb_q),                        .y(ln_out_h_w));
  always_ff @(posedge clk) begin
    ln_g_q       <= ln_norm_g_w;
    ln_beta_d1_q <= ln_beta_coll_w;
    ln_gb_q      <= ln_norm_gb_w;
  end

  // ===================================================================
  // 0x1D MASKED_SOFTMAX_FP32 synth sub-FSM primitives.
  //   neg_max  = sm_row_max_q ^ sign-bit
  //   diff     = row[iter] - row_max               (fp32_add with neg)
  //   exp_v    = exp(diff)                         (fp32_exp_p18, LATENCY=18)
  //   sum_add  = exp_sum_q + exp_v                 (fp32_add)
  //   norm     = exp_v / exp_sum_q                 (fp32_div_p6, LATENCY=6)
  //   out_h    = fp32_to_fp16(norm)
  // Visibility: F_G2_SM_MAX/EXPSUM/OUT each check
  //   (iter < n_elems_q) && (iter_signed <= sm_keep_through_q)
  //
  // 2026-07-21 (fmax phase 0b): `u_sm_exp` was the COMBINATIONAL fp32_exp, which
  // put the whole ~412 ns exp cloud inside one cycle on TWO reg-to-reg paths:
  //   EXPSUM   sm_exp_sum_q -> add -> exp -> add -> sm_exp_sum_q   (~490 ns)
  //   OUT_NORM row_data_q   -> add -> exp -> sm_exp_q              (~440 ns)
  // Neither was ever in a timing report (the full-SFU flatten OOMs), so the
  // committed 34.41 MHz — a div/sqrt-PRIMITIVE number — structurally excluded
  // them. It is now the 18-stage `fp32_exp_p18` (bit-IDENTICAL pure retiming,
  // 9M-vector zero-diff gate `make test_fp32_exp_p18`), fed one element/cycle
  // and collected LATENCY later, exactly like the lever-E divider drain.
  // The two register boundaries that make every path single-primitive:
  //   sm_diff_q : row_data_q -> u_sm_diff -> REG          (one fp32_add)
  //               -> exp_p18 stage 1, so the subtract never chains into exp.
  //   sm_exp_q  : exp_p18 s17 -> s18 output glue -> REG   (glue only)
  //               and is then the SOLE dividend/addend source, so the
  //               accumulate is REG -> u_sm_sum_add -> REG (one fp32_add) and
  //               the divide is REG -> div_p6 stage 1. (sm_exp_q already
  //               existed as the divider's registered dividend; reusing it for
  //               the accumulate costs one pipe stage, not a new register.)
  // Byte-exact: identical primitives, identical operands, and the accumulate
  // still walks elements in ascending index order (fp add is non-associative —
  // the collect pointer preserves the order, it does not reorder it).
  logic [31:0] sm_neg_max;
  logic [31:0] sm_diff_w;
  logic [31:0] sm_exp_w;
  logic [31:0] sm_sum_add_w;
  logic [31:0] sm_norm_w;
  logic [15:0] sm_out_h_w;
  logic        sm_exp_vo;
  assign sm_neg_max = sm_row_max_q ^ 32'h8000_0000;
  fp32_add     u_sm_diff   (.a(synth_a_bits), .b(sm_neg_max),    .y(sm_diff_w));
  fp32_exp_p18 u_sm_exp    (.clk(clk), .rst_n(rst_n), .valid_in(1'b1),
                            .a(sm_diff_q),
                            .valid_out(sm_exp_vo), .y(sm_exp_w));
  fp32_add     u_sm_sum_add(.a(sm_exp_sum_q), .b(sm_exp_q),       .y(sm_sum_add_w));
  // 2026-07-12 (lever E): 6-stage pipelined divider (LATENCY=6, fp32_div_p6).
  // Dividend is the REGISTERED sm_exp_q (latched in F_G2_SM_OUT_NORM). The
  // software-pipelined F_G2_SM_OUT_NORM drain collects at sm_coll_q =
  // iter_idx_q - 26 (1 sm_diff_q + 18 exp_p18 + 1 sm_exp_q + 6 div stages);
  // f2h reads sm_norm_w directly.
  fp32_div_p6  u_sm_div    (.clk(clk), .rst_n(rst_n), .valid_in(1'b1),
                            .a(sm_exp_q),     .b(sm_exp_sum_q),
                            .valid_out(sm_div_vo), .y(sm_norm_w));
  fp32_to_fp16 u_sm_out_h  (.a(sm_norm_w),                        .y(sm_out_h_w));
  // SOFTMAX max-update predicate: `diff` MSB (=sign): 1 -> row < max
  // (no update); 0 -> row >= max (update only if strictly > or first-vis).
  // Strictly > requires diff != 0; equal (diff == 0) is no-op either way.
  logic sm_row_gt_max;
  assign sm_row_gt_max = (sm_diff_w[31] == 1'b0) && (sm_diff_w[30:0] != 31'd0);

  // 2026-07-21 (fmax phase 0a): the gen-1 SOFTMAX and gen-1 GELU synth output
  // cones were DELETED here. Both were driven but had no reader — their
  // terminal nets (sm_g1_quant_w, gelu_g1_quant_w) were read nowhere in the
  // tree, and their driving states (F_GELU_SYNTH_I8_ITER / _I32_ITER) are
  // never entered (gen-1 OP_SOFTMAX/OP_GELU are illegal at decode). They cost
  // a full fp32_gelu_new (exp+div, ~700 ns) plus three fp32_div/quantize
  // chains of FALSE critical path in any full-SFU STA. `synth_scale1_bits`
  // went with them (its last consumer was the deleted ATTN V cone below).

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
  // Collect-indexed visibility for the software-pipelined F_G2_SM_OUT_NORM: the
  // element whose quotient emerges on sm_norm_w is sm_coll_q (= iter_idx_q-7),
  // NOT the feed element iter_idx_q, so its output mask must be evaluated at
  // sm_coll_q. Mirrors sm_visible_w exactly with sm_coll_q. Only gen-2
  // OP_MASKED_SOFTMAX_FP32 reaches mode-1 here; the gen-1 arms are decode-illegal
  // (dead) but kept for elaboration parity with sm_visible_w.
  logic sm_visible_coll_w;
  always_comb begin
    case (opcode_q)
      OP_SOFTMAX:                sm_visible_coll_w = 1'b1;
      OP_MASKED_SOFTMAX:         sm_visible_coll_w = attn_visible(row_idx_q, integer'(sm_coll_q));
      OP_MASKED_SOFTMAX_FP32:    sm_visible_coll_w =
          ($signed({6'b0, sm_coll_q}) <= $signed({1'b0, sm_keep_through_q[15:0]}));
      OP_SOFTMAX_ATTNV:          sm_visible_coll_w = 1'b1;
      OP_MASKED_SOFTMAX_ATTNV:   sm_visible_coll_w = attn_visible(row_idx_q, integer'(sm_coll_q));
      default:                   sm_visible_coll_w = 1'b0;
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

  // Runtime-bounded MAX/EXPSUM walk for gen-2 causal masked softmax. For
  // OP_MASKED_SOFTMAX_FP32 the visible set is EXACTLY the contiguous prefix
  // iter in [0, sm_keep_through_q] (sm_visible_w = iter <= keep_through), so the
  // MAX and EXPSUM reductions — which accumulate ONLY visible elements, in
  // element order — reach their final value at iter == keep_through; the
  // invisible tail (keep_through, sm_iter_bound_w) contributes nothing
  // (excluded from the max; exp gated to 0 in expsum). Bounding those two walks
  // to keep_through+1 is therefore bit-exact (identical values & accumulation
  // order) and skips the masked tail — a position-dependent decode win, since
  // decode's keep_through = position grows with the sequence while the compiled
  // key width sm_iter_bound_w stays at the context budget. keep_through is
  // treated as the same non-negative value sm_visible_w uses ({1'b0, kt}), so
  // the bound matches the mask exactly. The OUT pass keeps the full
  // sm_iter_bound_w walk (it must write 0 for every masked column, consumed by
  // the AV matmul). All other softmax opcodes keep the full bound (their
  // visible set is not this keep_through prefix), so they are unchanged.
  logic [16:0] sm_kt_p1_w;
  assign sm_kt_p1_w = {1'b0, sm_keep_through_q[15:0]} + 17'd1;  // keep_through + 1
  logic [15:0] sm_eff_bound_w;
  always_comb begin
    if ((opcode_q == OP_MASKED_SOFTMAX_FP32) &&
        (sm_kt_p1_w < {1'b0, sm_iter_bound_w}))
      sm_eff_bound_w = sm_kt_p1_w[15:0];
    else
      sm_eff_bound_w = sm_iter_bound_w;
  end

  // 2026-07-21 (fmax phase 0a): the Phase-3.B ATTN V_LATCH 16-lane synth cone
  // was DELETED here. It was the single largest dead cone in the SFU: a shared
  // add→exp→div weight chain plus 16 lanes of (i32_to_fp32, fp32_mul, fp32_mul,
  // fp32_add) accumulating into attn_acc_new_bits — a net with NO reader. Its
  // only consumer state, F_ATTN_V_LATCH (7'd20), has no case arm and nothing
  // transitions into it, so the whole cone was unreachable. Note attn_accum_q
  // itself is LIVE (the VADD / DEQUANT_ACCUM synth path reads it); only this
  // 16-lane accumulate cone is gone.
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
  // 2026-05-31: streamed 0x1F load. The chunk being reduced this cycle is the
  // CAPTURED one on the bus (ld_cap_q), not the issue pointer (read_idx_q, which
  // now runs one chunk ahead). base_idx in the F_G2_S1_LATCH 0x1F branch =
  // ld_cap_q * 8 to match.
  logic [15:0] mar_base_idx;
  assign mar_base_idx = {3'h0, ld_cap_q[12:0]} * 16'd8;
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
