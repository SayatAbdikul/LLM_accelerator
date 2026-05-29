// 2-stage pipelined IEEE-754 binary32 square root — bit-exact to the
// combinational `fp32_sqrt` (and thus to the DPI golden `sfu_fp32_sqrt`).
//
// WHY THIS EXISTS:
//   With the 4 binding dividers pipelined (fp32_div_p2), the combinational
//   `fp32_sqrt` 25-iteration digit-recurrence (the SFU LN denom path,
//   ln_denom_q) became the new post-PNR fmax floor (~96.9 ns). This module
//   cuts the path INSIDE the recurrence: the 25 restoring-sqrt iterations are
//   split 12 / 13 across a pipeline register, roughly halving the critical
//   path. (As with fp32_div_p2, registering only the OUTPUT would add latency
//   with ZERO fmax benefit — the cut must be inside the iteration chain.)
//
// EQUIVALENCE:
//   Everything except where the iteration chain is split is identical to
//   fp32_sqrt.sv (unpack, subnormal normalize, even-exponent radicand build,
//   restoring radix-2 recurrence, RNE rounding, non-finite contract, pack).
//   The array-indexed recurrence r[25..0]/q[25..0] is rewritten as an
//   equivalent forward-running (s_r, s_q) accumulation — same arithmetic,
//   same per-iteration trial/subtract — so q_final/r_final are bit-identical.
//   test_fp32_sqrt.cpp is the byte-exact streaming gate.
//
// SPLIT POINT (iter 8, not the midpoint): the recurrence cost is NOT uniform.
//   Early iterations have a narrow q (leading-zero-folded by synthesis, ~2.5
//   ns/iter); later iterations have a wide q -> full 52-bit compare/subtract
//   (~7.4 ns/iter). STA on a naive 12/13 midpoint split measured 30 ns for
//   stage 1 vs 102 ns for stage 2 (the entire wide-q tail). Balancing the
//   *delay* (not the iter count) puts the cut at iter 8: stage 1 runs the 17
//   cheap+moderate iters 24..8 (~67 ns), stage 2 the 8 widest iters 7..0 plus
//   round/pack (~65 ns) — both under the ~87 ns softmax-divider floor, so the
//   sqrt leaves the critical path. Stage 2's blocks (M_pad[15:0]) are all zero
//   (sig_a sits in M_pad's top bits; the one low set bit M_pad[25] is consumed
//   by iter 12, in stage 1), so stage 2 needs no M_pad register.
//
// INTERFACE: fixed LATENCY=2. Present (a,valid_in) on cycle N; (y,valid_out)
//   are valid on cycle N+2. Fully pipelined: accepts a new operand every cycle.

`ifndef FP32_SQRT_P2_SV
`define FP32_SQRT_P2_SV

module fp32_sqrt_p2 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  output logic        valid_out,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // =====================================================================
  // STAGE 1 (combinational): unpack, normalize, build radicand, iters 24..13.
  // =====================================================================
  logic        s;
  logic [7:0]  e;
  logic [22:0] m;
  assign s = a[31];
  assign e = a[30:23];
  assign m = a[22:0];

  logic a_zero, a_inf, a_nan, a_sub, a_neg_finite;
  assign a_zero       = (e == 8'd0)   && (m == 23'd0);
  assign a_sub        = (e == 8'd0)   && (m != 23'd0);
  assign a_inf        = (e == 8'd255) && (m == 23'd0);
  assign a_nan        = (e == 8'd255) && (m != 23'd0);
  assign a_neg_finite = s && !a_zero && !a_inf && !a_nan;

  int unsigned        mp;
  logic [23:0]        sig_a;
  logic signed [9:0]  exp_a;
  always_comb begin
    if (a_sub) begin
      mp    = msb23(m);
      sig_a = {1'b0, m} << (5'd23 - mp[4:0]);
      exp_a = -10'sd126 - {{5{1'b0}}, (5'd23 - mp[4:0])};
    end else begin
      mp    = 0;
      sig_a = {1'b1, m};
      exp_a = $signed({2'b0, e}) - 10'sd127;
    end
  end

  // Build even-exponent 50-bit radicand M_pad and result exponent R_exp.
  logic              exp_a_odd;
  assign exp_a_odd = exp_a[0];

  logic [49:0]       M_pad;
  logic signed [9:0] R_exp;
  always_comb begin
    if (exp_a_odd) begin
      M_pad = {sig_a, 26'd0};            // top bit at 49
      R_exp = (exp_a - 10'sd1) >>> 1;
    end else begin
      M_pad = {1'b0, sig_a, 25'd0};      // top bit at 48
      R_exp = exp_a >>> 1;
    end
  end

  // Restoring radix-2 sqrt as a forward-running accumulation (equivalent to
  // the array recurrence in fp32_sqrt.sv). STAGE 1 runs iters ii = 24..8.
  logic [51:0] s1_r;
  logic [24:0] s1_q;
  integer i1;
  always_comb begin
    s1_r = 52'd0;
    s1_q = 25'd0;
    for (i1 = 24; i1 >= 8; i1 = i1 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      logic [1:0]  blk;
      blk = M_pad[2*i1 +: 2];
      rn  = (s1_r << 2) | {50'd0, blk};
      tr  = ({27'd0, s1_q} << 2) | 52'd1;
      if (rn >= tr) begin
        s1_r = rn - tr;
        s1_q = (s1_q << 1) | 25'd1;
      end else begin
        s1_r = rn;
        s1_q = (s1_q << 1);
      end
    end
  end

  // ---- pipeline register ----
  // No M_pad bits needed: stage-2 iters 7..0 consume M_pad[15:0], always zero.
  logic              r_valid;
  logic [51:0]       r_r;
  logic [24:0]       r_q;
  logic signed [9:0] r_Rexp;
  logic              r_s, r_anan, r_negf, r_azero, r_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_valid <= 1'b0;
    end else begin
      r_valid <= valid_in;
      r_r     <= s1_r;
      r_q     <= s1_q;
      r_Rexp  <= R_exp;
      r_s     <= s;
      r_anan  <= a_nan;
      r_negf  <= a_neg_finite;
      r_azero <= a_zero;
      r_ainf  <= a_inf;
    end
  end

  // =====================================================================
  // STAGE 2 (combinational): iters 7..0 (zero blocks), RNE round, pack.
  // =====================================================================
  logic [51:0] s2_r;
  logic [24:0] q_final;
  integer i2;
  always_comb begin
    s2_r    = r_r;
    q_final = r_q;
    for (i2 = 7; i2 >= 0; i2 = i2 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s2_r << 2);               // block = M_pad[2*i2+:2] = 0 for i2 in 7..0
      tr  = ({27'd0, q_final} << 2) | 52'd1;
      if (rn >= tr) begin
        s2_r    = rn - tr;
        q_final = (q_final << 1) | 25'd1;
      end else begin
        s2_r    = rn;
        q_final = (q_final << 1);
      end
    end
  end

  logic        sticky;
  assign sticky = (s2_r != 52'd0);

  // RNE on (round=q_final[0], sticky) — verbatim from fp32_sqrt.
  logic        rb_sq, ru_sq;
  logic [23:0] mant_pre;
  assign rb_sq    = q_final[0];
  assign mant_pre = q_final[24:1];
  assign ru_sq    = rb_sq & (sticky | mant_pre[0]);

  logic [24:0] mant24_rnd;
  assign mant24_rnd = {1'b0, mant_pre} + {24'd0, ru_sq};

  logic               carry;
  logic signed [9:0]  exp_y_unb;
  logic [7:0]         exp_y_biased;
  always_comb begin
    carry        = mant24_rnd[24];
    exp_y_unb    = r_Rexp + (carry ? 10'sd1 : 10'sd0);
    exp_y_biased = 8'(exp_y_unb + 10'sd127);
  end

  logic [31:0] y_comb;
  always_comb begin
    if (r_anan) begin
      y_comb = QNAN;
    end else if (r_negf) begin
      y_comb = QNAN;                                   // sqrt(neg) -> NaN
    end else if (r_azero) begin
      y_comb = {r_s, 8'd0, 23'd0};                     // ±0 -> ±0
    end else if (r_ainf) begin
      y_comb = r_s ? QNAN : {1'b0, 8'd255, 23'd0};     // -inf -> NaN, +inf -> +inf
    end else if (exp_y_unb >= 10'sd128) begin
      y_comb = {1'b0, 8'd255, 23'd0};                  // overflow (essentially impossible)
    end else if (exp_y_unb < -10'sd126) begin
      y_comb = {1'b0, 8'd0, 23'd0};                    // underflow (also basically impossible)
    end else if (carry) begin
      y_comb = {1'b0, exp_y_biased, 23'd0};
    end else begin
      y_comb = {1'b0, exp_y_biased, mant24_rnd[22:0]};
    end
  end

  // ---- output register (stage-2 path ends at a register) ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
    end else begin
      valid_out <= r_valid;
      y         <= y_comb;
    end
  end

endmodule

`endif // FP32_SQRT_P2_SV
