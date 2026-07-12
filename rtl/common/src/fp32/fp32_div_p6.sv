// 6-stage pipelined IEEE-754 binary32 divide — bit-exact to the combinational
// `fp32_div` (and to fp32_div_p2/p3/p4/p5, and the DPI golden `sfu_fp32_div`).
//
// WHY THIS EXISTS:
//   After fp32_div_p5 + fp32_sqrt_p6, the SFU post-PNR fmax floor is a 3-way
//   primitive cluster: div_p5 STAGE 2 (rB_P, iters 23..18) and STAGE 4 (rD_P,
//   iters 11..6) co-bind with sqrt_p6 STAGE 2 (rB_r). The div_p5 middle stages
//   each carry SIX 25-bit restoring-divide iterations (~29 ns PNR). A SIXTH stage
//   recuts the 29-iteration restoring divide into ~5-iter pieces, dropping every
//   divider stage below the cluster. Must land together with the sqrt-Mpad
//   restructure (they co-bind — cutting one alone leaves the other as the floor).
//
// EQUIVALENCE:
//   Identical arithmetic to fp32_div_p5 — same unpack, subnormal normalize,
//   MSB-aligned restoring division (P preloaded = sig_a, iterate k=28..0), RNE
//   rounding, subnormal/underflow path, non-finite contract, pack. The only
//   change is WHERE the iteration chain is cut: 5 pipeline registers split it
//   six ways (p5 had 4 registers / five ways). The restoring divide is
//   value-identical regardless of where it is cut (each iteration's
//   compare/subtract/shift is unchanged), so q_full / remainder are bit-identical.
//   test_fp32_div_p6.cpp (built against this module with LATENCY=6) is the
//   byte-exact gate.
//
//   SPLIT POINTS (parameterized for STA rebalancing): stage 1 runs iters
//   28..SPLIT1, stage 2 SPLIT1-1..SPLIT2, stage 3 SPLIT2-1..SPLIT3, stage 4
//   SPLIT3-1..SPLIT4, stage 5 SPLIT4-1..SPLIT5, stage 6 SPLIT5-1..0 plus
//   round/pack. Divider iterations are ~uniform cost (compare is always 25-bit),
//   so the split is near-even; stage 1 gets fewer iters (carries unpack), stage 6
//   fewer (carries round/pack).
//   SPLIT1=24,SPLIT2=19,SPLIT3=14,SPLIT4=9,SPLIT5=4 -> 5/5/5/5/5/4 iters.
//
// INTERFACE: fixed LATENCY=6. Present (a,b,valid_in) on cycle N; (y,valid_out)
//   are valid on cycle N+6. Fully pipelined: accepts a new pair every cycle.

`ifndef FP32_DIV_P6_SV
`define FP32_DIV_P6_SV

module fp32_div_p6 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        valid_out,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;
  // Iteration split points. Stage1: 28..SPLIT1 ; Stage2: SPLIT1-1..SPLIT2 ;
  // Stage3: SPLIT2-1..SPLIT3 ; Stage4: SPLIT3-1..SPLIT4 ; Stage5: SPLIT4-1..SPLIT5 ;
  // Stage6: SPLIT5-1..0.
  // SPLIT1=25,SPLIT2=20,SPLIT3=15,SPLIT4=10,SPLIT5=5 -> 4 / 5 / 5 / 5 / 5 / 5 iters.
  // (STA-tuned: stage 1 carries unpack, so it gets 4 iters; the five middles get
  //  5 each. Standalone sky130 synth+STA floor 28.85 ns vs 30.45 for 5/5/5/5/5/4.)
  localparam int SPLIT1 = 25;
  localparam int SPLIT2 = 20;
  localparam int SPLIT3 = 15;
  localparam int SPLIT4 = 10;
  localparam int SPLIT5 = 5;

  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // =====================================================================
  // STAGE 1 (combinational): unpack, normalize sigs, division iters 28..SPLIT1.
  // =====================================================================
  logic        sa, sb;
  logic [7:0]  ea, eb;
  logic [22:0] ma, mb;
  assign sa = a[31];  assign ea = a[30:23];  assign ma = a[22:0];
  assign sb = b[31];  assign eb = b[30:23];  assign mb = b[22:0];

  logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan, a_sub, b_sub;
  assign a_zero = (ea == 8'd0)   && (ma == 23'd0);
  assign b_zero = (eb == 8'd0)   && (mb == 23'd0);
  assign a_sub  = (ea == 8'd0)   && (ma != 23'd0);
  assign b_sub  = (eb == 8'd0)   && (mb != 23'd0);
  assign a_inf  = (ea == 8'd255) && (ma == 23'd0);
  assign b_inf  = (eb == 8'd255) && (mb == 23'd0);
  assign a_nan  = (ea == 8'd255) && (ma != 23'd0);
  assign b_nan  = (eb == 8'd255) && (mb != 23'd0);

  int unsigned mp_a, mp_b;
  logic [23:0] sig_a, sig_b;
  logic signed [9:0] exp_a, exp_b;
  always_comb begin
    mp_a = a_sub ? msb23(ma) : 0;
    mp_b = b_sub ? msb23(mb) : 0;
    if (a_sub) begin
      sig_a = {1'b0, ma} << (5'd23 - mp_a[4:0]);
      exp_a = -10'sd126 - {{5{1'b0}}, (5'd23 - mp_a[4:0])};
    end else begin
      sig_a = {1'b1, ma};
      exp_a = $signed({2'b0, ea}) - 10'sd127;
    end
    if (b_sub) begin
      sig_b = {1'b0, mb} << (5'd23 - mp_b[4:0]);
      exp_b = -10'sd126 - {{5{1'b0}}, (5'd23 - mp_b[4:0])};
    end else begin
      sig_b = {1'b1, mb};
      exp_b = $signed({2'b0, eb}) - 10'sd127;
    end
  end

  logic [24:0] s1_P;
  logic [28:0] s1_q;
  integer k1;
  always_comb begin
    s1_P = {1'b0, sig_a};        // 25-bit, < 2^24
    s1_q = 29'd0;
    for (k1 = 28; k1 >= SPLIT1; k1 = k1 - 1) begin
      if (s1_P >= {1'b0, sig_b}) begin
        s1_q[k1] = 1'b1;
        s1_P = s1_P - {1'b0, sig_b};
      end
      s1_P = s1_P << 1;          // k1 >= SPLIT1 > 0, always shift
    end
  end

  // ---- pipeline register A (stage1 -> stage2) ----
  logic        rA_valid;
  logic [24:0] rA_P;
  logic [28:0] rA_q;
  logic [23:0] rA_sigb;
  logic        rA_sign, rA_agb;
  logic signed [9:0] rA_expa, rA_expb;
  logic        rA_anan, rA_bnan, rA_ainf, rA_binf, rA_azero, rA_bzero;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rA_valid <= 1'b0;
    end else begin
      rA_valid <= valid_in;
      rA_P     <= s1_P;
      rA_q     <= s1_q;
      rA_sigb  <= sig_b;
      rA_sign  <= sa ^ sb;
      rA_agb   <= (sig_a >= sig_b);
      rA_expa  <= exp_a;
      rA_expb  <= exp_b;
      rA_anan  <= a_nan;  rA_bnan  <= b_nan;
      rA_ainf  <= a_inf;  rA_binf  <= b_inf;
      rA_azero <= a_zero; rA_bzero <= b_zero;
    end
  end

  // =====================================================================
  // STAGE 2 (combinational): division iters SPLIT1-1 .. SPLIT2.
  // =====================================================================
  logic [24:0] s2_P;
  logic [28:0] s2_q;
  integer k2;
  always_comb begin
    s2_P = rA_P;
    s2_q = rA_q;
    for (k2 = SPLIT1 - 1; k2 >= SPLIT2; k2 = k2 - 1) begin
      if (s2_P >= {1'b0, rA_sigb}) begin
        s2_q[k2] = 1'b1;
        s2_P = s2_P - {1'b0, rA_sigb};
      end
      s2_P = s2_P << 1;          // k2 >= SPLIT2 > 0, always shift
    end
  end

  // ---- pipeline register B (stage2 -> stage3) ----
  logic        rB_valid;
  logic [24:0] rB_P;
  logic [28:0] rB_q;
  logic [23:0] rB_sigb;
  logic        rB_sign, rB_agb;
  logic signed [9:0] rB_expa, rB_expb;
  logic        rB_anan, rB_bnan, rB_ainf, rB_binf, rB_azero, rB_bzero;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rB_valid <= 1'b0;
    end else begin
      rB_valid <= rA_valid;
      rB_P     <= s2_P;
      rB_q     <= s2_q;
      rB_sigb  <= rA_sigb;
      rB_sign  <= rA_sign;
      rB_agb   <= rA_agb;
      rB_expa  <= rA_expa;
      rB_expb  <= rA_expb;
      rB_anan  <= rA_anan;  rB_bnan  <= rA_bnan;
      rB_ainf  <= rA_ainf;  rB_binf  <= rA_binf;
      rB_azero <= rA_azero; rB_bzero <= rA_bzero;
    end
  end

  // =====================================================================
  // STAGE 3 (combinational): division iters SPLIT2-1 .. SPLIT3.
  // =====================================================================
  logic [24:0] s3_P;
  logic [28:0] s3_q;
  integer k3;
  always_comb begin
    s3_P = rB_P;
    s3_q = rB_q;
    for (k3 = SPLIT2 - 1; k3 >= SPLIT3; k3 = k3 - 1) begin
      if (s3_P >= {1'b0, rB_sigb}) begin
        s3_q[k3] = 1'b1;
        s3_P = s3_P - {1'b0, rB_sigb};
      end
      s3_P = s3_P << 1;          // k3 >= SPLIT3 > 0, always shift
    end
  end

  // ---- pipeline register C (stage3 -> stage4) ----
  logic        rC_valid;
  logic [24:0] rC_P;
  logic [28:0] rC_q;
  logic [23:0] rC_sigb;
  logic        rC_sign, rC_agb;
  logic signed [9:0] rC_expa, rC_expb;
  logic        rC_anan, rC_bnan, rC_ainf, rC_binf, rC_azero, rC_bzero;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rC_valid <= 1'b0;
    end else begin
      rC_valid <= rB_valid;
      rC_P     <= s3_P;
      rC_q     <= s3_q;
      rC_sigb  <= rB_sigb;
      rC_sign  <= rB_sign;
      rC_agb   <= rB_agb;
      rC_expa  <= rB_expa;
      rC_expb  <= rB_expb;
      rC_anan  <= rB_anan;  rC_bnan  <= rB_bnan;
      rC_ainf  <= rB_ainf;  rC_binf  <= rB_binf;
      rC_azero <= rB_azero; rC_bzero <= rB_bzero;
    end
  end

  // =====================================================================
  // STAGE 4 (combinational): division iters SPLIT3-1 .. SPLIT4.
  // =====================================================================
  logic [24:0] s4_P;
  logic [28:0] s4_q;
  integer k4;
  always_comb begin
    s4_P = rC_P;
    s4_q = rC_q;
    for (k4 = SPLIT3 - 1; k4 >= SPLIT4; k4 = k4 - 1) begin
      if (s4_P >= {1'b0, rC_sigb}) begin
        s4_q[k4] = 1'b1;
        s4_P = s4_P - {1'b0, rC_sigb};
      end
      s4_P = s4_P << 1;          // k4 >= SPLIT4 > 0, always shift
    end
  end

  // ---- pipeline register D (stage4 -> stage5) ----
  logic        rD_valid;
  logic [24:0] rD_P;
  logic [28:0] rD_q;
  logic [23:0] rD_sigb;
  logic        rD_sign, rD_agb;
  logic signed [9:0] rD_expa, rD_expb;
  logic        rD_anan, rD_bnan, rD_ainf, rD_binf, rD_azero, rD_bzero;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rD_valid <= 1'b0;
    end else begin
      rD_valid <= rC_valid;
      rD_P     <= s4_P;
      rD_q     <= s4_q;
      rD_sigb  <= rC_sigb;
      rD_sign  <= rC_sign;
      rD_agb   <= rC_agb;
      rD_expa  <= rC_expa;
      rD_expb  <= rC_expb;
      rD_anan  <= rC_anan;  rD_bnan  <= rC_bnan;
      rD_ainf  <= rC_ainf;  rD_binf  <= rC_binf;
      rD_azero <= rC_azero; rD_bzero <= rC_bzero;
    end
  end

  // =====================================================================
  // STAGE 5 (combinational): division iters SPLIT4-1 .. SPLIT5.
  // =====================================================================
  logic [24:0] s5_P;
  logic [28:0] s5_q;
  integer k5;
  always_comb begin
    s5_P = rD_P;
    s5_q = rD_q;
    for (k5 = SPLIT4 - 1; k5 >= SPLIT5; k5 = k5 - 1) begin
      if (s5_P >= {1'b0, rD_sigb}) begin
        s5_q[k5] = 1'b1;
        s5_P = s5_P - {1'b0, rD_sigb};
      end
      s5_P = s5_P << 1;          // k5 >= SPLIT5 > 0, always shift
    end
  end

  // ---- pipeline register E (stage5 -> stage6) ----
  logic        rE_valid;
  logic [24:0] rE_P;
  logic [28:0] rE_q;
  logic [23:0] rE_sigb;
  logic        rE_sign, rE_agb;
  logic signed [9:0] rE_expa, rE_expb;
  logic        rE_anan, rE_bnan, rE_ainf, rE_binf, rE_azero, rE_bzero;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rE_valid <= 1'b0;
    end else begin
      rE_valid <= rD_valid;
      rE_P     <= s5_P;
      rE_q     <= s5_q;
      rE_sigb  <= rD_sigb;
      rE_sign  <= rD_sign;
      rE_agb   <= rD_agb;
      rE_expa  <= rD_expa;
      rE_expb  <= rD_expb;
      rE_anan  <= rD_anan;  rE_bnan  <= rD_bnan;
      rE_ainf  <= rD_ainf;  rE_binf  <= rD_binf;
      rE_azero <= rD_azero; rE_bzero <= rD_bzero;
    end
  end

  // =====================================================================
  // STAGE 6 (combinational): division iters SPLIT5-1 .. 0, RNE round, pack.
  // =====================================================================
  logic [24:0] s6_P;
  logic [28:0] q_full;
  logic [24:0] remainder;        // only |remainder| (sticky) is used
  integer k6;
  always_comb begin
    s6_P   = rE_P;
    q_full = rE_q;
    for (k6 = SPLIT5 - 1; k6 >= 0; k6 = k6 - 1) begin
      if (s6_P >= {1'b0, rE_sigb}) begin
        q_full[k6] = 1'b1;
        s6_P = s6_P - {1'b0, rE_sigb};
      end
      if (k6 > 0) s6_P = s6_P << 1;
    end
    remainder = s6_P;            // final partial remainder = dividend mod sig_b
  end

  // --- extract mantissa + RNE on the regime ULP (verbatim from fp32_div_p5) ---
  logic [23:0] mant24_pre;
  logic        rb, st, ru;
  logic [24:0] mant24_rnd;
  logic signed [10:0] exp_y_unb;
  always_comb begin
    if (rE_agb) begin
      mant24_pre = q_full[28:5];
      rb         = q_full[4];
      st         = (|q_full[3:0]) | (|remainder);
      exp_y_unb  = rE_expa - rE_expb;
    end else begin
      mant24_pre = q_full[27:4];
      rb         = q_full[3];
      st         = (|q_full[2:0]) | (|remainder);
      exp_y_unb  = rE_expa - rE_expb - 11'sd1;
    end
    ru         = rb & (st | mant24_pre[0]);
    mant24_rnd = {1'b0, mant24_pre} + {24'd0, ru};
  end

  logic               carry;
  logic signed [10:0] exp_y_final_unb;
  logic [7:0]         exp_y_biased;
  always_comb begin
    carry           = mant24_rnd[24];
    exp_y_final_unb = exp_y_unb + (carry ? 11'sd1 : 11'sd0);
    exp_y_biased    = 8'(exp_y_final_unb + 11'sd127);
  end

  // Subnormal/underflow path (verbatim from fp32_div_p5).
  logic signed [11:0] k_lsb_s;
  logic [4:0]         k_lsb;
  logic [28:0]        q_shifted_sub;
  logic [22:0]        mant_sub_pre;
  logic               sub_rb, sub_st, sub_ru;
  logic [23:0]        mant_sub_rnd;
  always_comb begin
    k_lsb_s       = -{{2{rE_expa[9]}}, rE_expa} + {{2{rE_expb[9]}}, rE_expb} - 12'sd121;
    k_lsb         = (k_lsb_s > 12'sd29) ? 5'd29 : k_lsb_s[4:0];
    q_shifted_sub = q_full >> k_lsb;
    mant_sub_pre  = q_shifted_sub[22:0];

    if (k_lsb_s >= 12'sd1 && k_lsb_s <= 12'sd29)
      sub_rb = q_full[k_lsb - 5'd1];
    else
      sub_rb = 1'b0;

    if (k_lsb_s > 12'sd29)
      sub_st = (|q_full) | (|remainder);
    else if (k_lsb_s >= 12'sd2)
      sub_st = (|(q_full & ((29'd1 << (k_lsb - 5'd1)) - 29'd1))) | (|remainder);
    else if (k_lsb_s == 12'sd1)
      sub_st = (|remainder);
    else
      sub_st = 1'b0;

    sub_ru       = sub_rb & (sub_st | mant_sub_pre[0]);
    mant_sub_rnd = {1'b0, mant_sub_pre} + {23'd0, sub_ru};
  end

  logic [31:0] y_comb;
  always_comb begin
    if (rE_anan || rE_bnan) begin
      y_comb = QNAN;
    end else if (rE_ainf && rE_binf) begin
      y_comb = QNAN;
    end else if (rE_azero && rE_bzero) begin
      y_comb = QNAN;
    end else if (rE_ainf) begin
      y_comb = {rE_sign, 8'd255, 23'd0};
    end else if (rE_binf) begin
      y_comb = {rE_sign, 8'd0,   23'd0};
    end else if (rE_bzero) begin
      y_comb = {rE_sign, 8'd255, 23'd0};
    end else if (rE_azero) begin
      y_comb = {rE_sign, 8'd0,   23'd0};
    end else if (exp_y_final_unb >= 11'sd128) begin
      y_comb = {rE_sign, 8'd255, 23'd0};
    end else if (exp_y_final_unb < -11'sd126) begin
      if (mant_sub_rnd[23])
        y_comb = {rE_sign, 8'd1, 23'd0};
      else
        y_comb = {rE_sign, 8'd0, mant_sub_rnd[22:0]};
    end else if (carry) begin
      y_comb = {rE_sign, exp_y_biased, 23'd0};
    end else begin
      y_comb = {rE_sign, exp_y_biased, mant24_rnd[22:0]};
    end
  end

  // ---- output register (stage-6 path ends at a register) ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
    end else begin
      valid_out <= rE_valid;
      y         <= y_comb;
    end
  end

endmodule

`endif // FP32_DIV_P6_SV
