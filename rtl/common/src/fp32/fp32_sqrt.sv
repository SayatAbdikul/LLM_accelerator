// Synthesizable IEEE-754 binary32 square root — bit-exact to the DPI golden
// `sfu_fp32_sqrt` (testbench.h): (float)std::sqrt((float)x), with NaN
// canonicalized to 0x7FC00000 and sqrt(-x for x>0) -> qNaN.
//
// Algorithm: non-restoring radix-2 digit-recurrence sqrt, fully unrolled
// (25 iterations -> 25-bit integer sqrt = 24-bit mantissa + 1 round bit).
// The partial remainder after the last iteration provides the sticky bit
// (exactly: remainder != 0 means input was not a perfect square).
//
// Exponent / radicand normalization:
//   value(a) = sig_a * 2^(exp_a - 23). sqrt(value) = sqrt(sig_a) * 2^((exp_a-23)/2).
//   If (exp_a-23) is odd, shift sig_a left by 1 (still <= 25 bits) and use
//   exp_eff = exp_a - 24 (now even). Effective radicand M in [2^23, 2^25).
//   M expanded to 50 bits (M << 26) gives an integer sqrt of 25 bits.
//
// Non-finite contract:
//   NaN     -> qNaN
//   -finite (a != ±0) -> qNaN  (sqrt of negative)
//   -0      -> -0  (IEEE special, matches libm)
//   +0      -> +0
//   +inf    -> +inf

`ifndef FP32_SQRT_SV
`define FP32_SQRT_SV

module fp32_sqrt (
  input  logic [31:0] a,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // most-significant set-bit index of a 23-bit value (used to normalize fp32 subnormals)
  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // --- unpack ---
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

  // Effective 24-bit significand (handle subnormals via leading-1 normalization).
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

  // --- normalize exponent to even, building 25-bit radicand M ---
  // We want value = M * 2^(2 * R_exp), so result = sqrt(M) * 2^R_exp.
  // If exp_a is even: shift sig_a left by 1 so the radicand has the proper
  // parity (M = sig_a << 1, top bit at 24); exp_eff = exp_a - 1; R_exp = exp_eff/2 + (-23/2 round).
  // Simpler concrete formula: compute exp_a_even = (exp_a even ? exp_a : exp_a - 1);
  //   M = (exp_a even ? sig_a << 1 : sig_a)
  // Then result_exp_unbiased = exp_a_even / 2  (treating sig_a*2^(exp_a-23) as
  // (M << shift) * 2^(exp_a_even - 23 - 1)... hmm easier to think in terms of
  // the integer sqrt below.

  // Simplified: regardless of parity, build an integer R such that
  //   value = R * 2^P  with P even, and R has its top bit at position 47 or 48.
  // sqrt(R) gives 24 or 25 bits.
  logic       exp_a_odd;
  assign exp_a_odd = exp_a[0];

  // Expand sig_a to 50-bit value: if exp_a odd, sig_a is at bit position 47 (top);
  //   if exp_a even, shift left by 1 -> top at bit 48.
  // Combined with 2-bit-per-iter sqrt, we want the radicand top bit at an
  // EVEN position (bits 48 for largest). Let's place sig_a so that
  // value_int_part = sig_a (top at bit 23 normally) and integer-sqrt gives
  // a 12-bit result -- but we need 24+ bits result.
  // To get 25-bit result sqrt, the radicand must be 50-bit; M[49:0]:
  //   sqrt(M) in [2^24, 2^25).  M in [2^48, 2^50).
  // Place sig_a (24-bit) at the top of M:
  //   if exp_a odd:  M = {sig_a, 26'd0}; top bit at 49.  hmm but that's odd
  //     -> sqrt(M) in [2^24.5, ?)
  //   We need M's top bit at an even-numbered position for the digit-recurrence
  //   sqrt to be naturally well-aligned.
  // Use this convention: M is 50-bit, with sig_a placed such that bit-49 is set
  // iff sig_a >= 2^23 AND exp_a is even (so adding 1 shift makes top bit 49).
  // Equivalent: M = sig_a << shift, where shift = exp_a_odd ? 25 : 26.
  //   exp_a even -> shift=26 -> sig_a at bits [49:26], top=49.
  //   exp_a odd  -> shift=25 -> sig_a at bits [48:25], top=48.
  // sqrt(M):
  //   M top at 49 (even=> for sqrt regime): sqrt in [2^24.5, 2^25)? Hmm we want
  //   result top bit at index 24 (a 25-bit value).
  // Easier approach: stay analytical, just shift sig_a appropriately.

  logic [49:0] M_pad;
  logic signed [9:0] R_exp;
  always_comb begin
    if (exp_a_odd) begin
      M_pad = {sig_a, 26'd0};            // top bit at 49 (since sig_a[23]=1, normal)
      R_exp = (exp_a - 10'sd1) >>> 1;    // (exp_a-1)/2 (signed shift)
    end else begin
      M_pad = {1'b0, sig_a, 25'd0};      // top bit at 48
      R_exp = exp_a >>> 1;               // exp_a/2 (signed)
    end
  end

  // --- non-restoring radix-2 digit-recurrence sqrt over M_pad ---
  // 25 iterations produce a 25-bit q. The remainder R after the last iteration
  // is exactly M_pad - q*q; for perfect squares R == 0.
  //
  // Convention: r is 52-bit signed partial remainder; q is 25-bit unsigned
  // building result.
  //
  // Step i (for i = 24 down to 0):
  //   r <- (r << 2) - 2-bit block of M_pad at position (2*i+1, 2*i)
  //   ... non-restoring variant uses r as signed.
  // For simplicity, use the RESTORING variant:
  //   r <- (r << 2) | mp_block
  //   trial = (q << 2) | 1
  //   if r >= trial: r = r - trial; q = (q << 1) | 1
  //   else:                            q =  q << 1
  logic [51:0] r [0:25];
  logic [24:0] q [0:25];
  logic [51:0] trial;
  integer ii;
  always_comb begin
    r[25] = 52'd0;
    q[25] = 25'd0;
    for (ii = 24; ii >= 0; ii = ii - 1) begin
      // Move ii-th 2-bit block from top of M_pad into r.
      // Note: this is the "next iteration" trial; the variables are indexed
      // from the FINAL state backwards (r[0],q[0] is the LAST iteration's result).
      logic [51:0] r_next0;
      logic [51:0] trial0;
      logic [1:0]  block;
      block   = M_pad[2*ii +: 2];
      r_next0 = (r[ii + 1] << 2) | {50'd0, block};
      trial0  = ({27'd0, q[ii + 1]} << 2) | 52'd1;  // (q<<2) | 1, widened to 52
      if (r_next0 >= trial0) begin
        r[ii] = r_next0 - trial0;
        q[ii] = (q[ii + 1] << 1) | 25'd1;
      end else begin
        r[ii] = r_next0;
        q[ii] = (q[ii + 1] << 1);
      end
    end
  end

  // Final q[0] has 25 bits (top bit at 24 = implicit leading 1; mantissa
  // bits 23..1; round bit at 0). Final r[0] is the remainder (sticky if != 0).
  logic [24:0] q_final;
  logic [51:0] r_final;
  logic        sticky;
  assign q_final = q[0];
  assign r_final = r[0];
  assign sticky  = (r_final != 52'd0);

  // RNE on (round=q_final[0], sticky)
  logic        rb_sq, ru_sq;
  logic [23:0] mant_pre, mant_rnd;
  assign rb_sq    = q_final[0];
  assign mant_pre = q_final[24:1];                   // 24-bit (incl. implicit leading 1)
  assign ru_sq    = rb_sq & (sticky | mant_pre[0]);

  logic [24:0] mant24_rnd;
  assign mant24_rnd = {1'b0, mant_pre} + {24'd0, ru_sq};

  // Result exponent: result_value = q_final / 2^(?), with sqrt(M_pad) producing
  // q with top bit at 24 -> sqrt-value at bit 24. Need to map to fp32:
  //   value = q_final / 2^? * 2^R_exp -- need to add a precise offset.
  // Empirical alignment by R_exp formula above: the result mantissa MSB
  // (bit 24 of q_final) corresponds to weight 2^R_exp. So exp_y_unb = R_exp.
  // (If rounding carries into bit 24+1 - but mant24_rnd is 25-bit and carry is
  // tracked by mant24_rnd[24].)
  logic               carry;
  logic signed [9:0]  exp_y_unb;
  logic [7:0]         exp_y_biased;
  always_comb begin
    carry        = mant24_rnd[24];
    exp_y_unb    = R_exp + (carry ? 10'sd1 : 10'sd0);
    exp_y_biased = 8'(exp_y_unb + 10'sd127);
  end

  always_comb begin
    if (a_nan) begin
      y = QNAN;
    end else if (a_neg_finite) begin
      y = QNAN;                                       // sqrt(neg) -> NaN
    end else if (a_zero) begin
      y = {s, 8'd0, 23'd0};                           // ±0 -> ±0
    end else if (a_inf) begin
      y = s ? QNAN : {1'b0, 8'd255, 23'd0};           // -inf -> NaN, +inf -> +inf
    end else if (exp_y_unb >= 10'sd128) begin
      y = {1'b0, 8'd255, 23'd0};                      // overflow (essentially impossible for sqrt)
    end else if (exp_y_unb < -10'sd126) begin
      y = {1'b0, 8'd0, 23'd0};                        // underflow (also basically impossible)
    end else if (carry) begin
      y = {1'b0, exp_y_biased, 23'd0};
    end else begin
      y = {1'b0, exp_y_biased, mant24_rnd[22:0]};
    end
  end

endmodule

`endif // FP32_SQRT_SV
