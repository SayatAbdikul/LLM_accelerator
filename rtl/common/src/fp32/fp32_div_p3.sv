// 3-stage pipelined IEEE-754 binary32 divide — bit-exact to the combinational
// `fp32_div` (and to the 2-stage `fp32_div_p2`, and the DPI golden
// `sfu_fp32_div`).
//
// WHY THIS EXISTS:
//   After fp32_div_p2 (2-stage) + fp32_sqrt_p2, the SFU synth fmax floor is the
//   divider's STAGE 1 (~88 ns: unpack/normalize + the front half of the
//   29-iteration restoring divide). Splitting the divide into THREE stages
//   drops each stage to ~60-70 ns, moving the floor down another tier.
//
// EQUIVALENCE:
//   Identical arithmetic to fp32_div_p2 — same unpack, subnormal normalize,
//   MSB-aligned restoring division (P preloaded = sig_a, iterate k=28..0),
//   RNE rounding, subnormal/underflow path, non-finite contract, pack. The
//   only change is WHERE the iteration chain is cut: 2 pipeline registers
//   split it three ways. The restoring divide is value-identical regardless of
//   where it is cut (each iteration's compare/subtract/shift is unchanged), so
//   q_full/remainder are bit-identical. test_fp32_div.cpp (built against this
//   module with LATENCY=3) is the byte-exact gate.
//
//   SPLIT POINTS (parameterized for STA rebalancing): stage 1 runs iters
//   28..SPLIT1, stage 2 runs SPLIT1-1..SPLIT2, stage 3 runs SPLIT2-1..0 plus
//   round/pack. Divider iterations are ~uniform cost (the compare is always
//   25-bit, unlike sqrt's widening q), so the split is near-even; stage 1 gets
//   fewer iters because it also carries the unpack/normalize, stage 3 fewer
//   because it carries round/pack.
//
// INTERFACE: fixed LATENCY=3. Present (a,b,valid_in) on cycle N; (y,valid_out)
//   are valid on cycle N+3. Fully pipelined: accepts a new pair every cycle.

`ifndef FP32_DIV_P3_SV
`define FP32_DIV_P3_SV

module fp32_div_p3 (
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
  // Stage3: SPLIT2-1..0. SPLIT1=21,SPLIT2=9 -> 8 / 12 / 9 iters.
  localparam int SPLIT1 = 21;
  localparam int SPLIT2 = 9;

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
  // STAGE 3 (combinational): division iters SPLIT2-1 .. 0, RNE round, pack.
  // =====================================================================
  logic [24:0] s3_P;
  logic [28:0] q_full;
  logic [24:0] remainder;        // only |remainder| (sticky) is used
  integer k3;
  always_comb begin
    s3_P   = rB_P;
    q_full = rB_q;
    for (k3 = SPLIT2 - 1; k3 >= 0; k3 = k3 - 1) begin
      if (s3_P >= {1'b0, rB_sigb}) begin
        q_full[k3] = 1'b1;
        s3_P = s3_P - {1'b0, rB_sigb};
      end
      if (k3 > 0) s3_P = s3_P << 1;
    end
    remainder = s3_P;            // final partial remainder = dividend mod sig_b
  end

  // --- extract mantissa + RNE on the regime ULP (verbatim from fp32_div_p2) ---
  logic [23:0] mant24_pre;
  logic        rb, st, ru;
  logic [24:0] mant24_rnd;
  logic signed [10:0] exp_y_unb;
  always_comb begin
    if (rB_agb) begin
      mant24_pre = q_full[28:5];
      rb         = q_full[4];
      st         = (|q_full[3:0]) | (|remainder);
      exp_y_unb  = rB_expa - rB_expb;
    end else begin
      mant24_pre = q_full[27:4];
      rb         = q_full[3];
      st         = (|q_full[2:0]) | (|remainder);
      exp_y_unb  = rB_expa - rB_expb - 11'sd1;
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

  // Subnormal/underflow path (verbatim from fp32_div_p2).
  logic signed [11:0] k_lsb_s;
  logic [4:0]         k_lsb;
  logic [28:0]        q_shifted_sub;
  logic [22:0]        mant_sub_pre;
  logic               sub_rb, sub_st, sub_ru;
  logic [23:0]        mant_sub_rnd;
  always_comb begin
    k_lsb_s       = -{{2{rB_expa[9]}}, rB_expa} + {{2{rB_expb[9]}}, rB_expb} - 12'sd121;
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
    if (rB_anan || rB_bnan) begin
      y_comb = QNAN;
    end else if (rB_ainf && rB_binf) begin
      y_comb = QNAN;
    end else if (rB_azero && rB_bzero) begin
      y_comb = QNAN;
    end else if (rB_ainf) begin
      y_comb = {rB_sign, 8'd255, 23'd0};
    end else if (rB_binf) begin
      y_comb = {rB_sign, 8'd0,   23'd0};
    end else if (rB_bzero) begin
      y_comb = {rB_sign, 8'd255, 23'd0};
    end else if (rB_azero) begin
      y_comb = {rB_sign, 8'd0,   23'd0};
    end else if (exp_y_final_unb >= 11'sd128) begin
      y_comb = {rB_sign, 8'd255, 23'd0};
    end else if (exp_y_final_unb < -11'sd126) begin
      if (mant_sub_rnd[23])
        y_comb = {rB_sign, 8'd1, 23'd0};
      else
        y_comb = {rB_sign, 8'd0, mant_sub_rnd[22:0]};
    end else if (carry) begin
      y_comb = {rB_sign, exp_y_biased, 23'd0};
    end else begin
      y_comb = {rB_sign, exp_y_biased, mant24_rnd[22:0]};
    end
  end

  // ---- output register (stage-3 path ends at a register) ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
    end else begin
      valid_out <= rB_valid;
      y         <= y_comb;
    end
  end

endmodule

`endif // FP32_DIV_P3_SV
