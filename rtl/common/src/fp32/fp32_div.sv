// Synthesizable IEEE-754 binary32 divide — bit-exact to the DPI golden
// `sfu_fp32_div` (testbench.h): (float)((float)a / (float)b), with NaN
// canonicalized to 0x7FC00000.
//
// Algorithm (exact-by-construction long division):
//   1. Normalize subnormal inputs (find leading 1 of mantissa, shift to bit
//      23, adjust unbiased exponent).
//   2. Compute  dividend = sig_a << 28,  quotient = dividend / sig_b,
//                                       remainder = dividend - quotient*sig_b.
//      sig_a, sig_b are 24-bit normalized in [2^23, 2^24).  dividend is 52-bit.
//      sig_a / sig_b in [0.5, 2):
//        - sig_a >= sig_b -> q_full in [2^27, 2^28), MSB at bit 27.
//        - sig_a <  sig_b -> q_full in [2^26, 2^27), MSB at bit 26.
//   3. Result mantissa = q_full[MSB:MSB-23] (24 bits incl. hidden);
//      round bit = q_full[MSB-24], sticky = |q_full[MSB-25:0] | (remainder!=0).
//   4. RNE round-to-nearest-ties-to-even on (round, sticky); carry bumps the
//      exponent by 1 (and may overflow to inf).
//   5. Result exponent: exp_a - exp_b + (sig_a >= sig_b ? 0 : -1) + 127 bias.
//      Overflow (exp_y_unbiased >= 128) -> ±inf.
//      Underflow (exp_y_unbiased < -126) -> subnormal/zero with regime-correct
//      rounding from the same q_full bits.
//
// Non-finite contract:
//   NaN involved   -> qNaN 0x7FC00000
//   inf / inf      -> qNaN
//   inf / finite   -> ±inf (sign = s_a^s_b)
//   finite / inf   -> ±0
//   0 / 0          -> qNaN
//   x!=0 / 0       -> ±inf (sign = s_a^s_b)
//   0 / x!=0       -> ±0   (sign = s_a^s_b)

`ifndef FP32_DIV_SV
`define FP32_DIV_SV

module fp32_div (
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // most-significant set-bit index of a 23-bit value (-1 if zero, but caller
  // only invokes for nonzero subnormal m where MSB is in [0, 22]).
  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // --- unpack ---
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

  // --- normalize sigs (handle subnormals) ---
  int unsigned mp_a, mp_b;
  logic [23:0] sig_a, sig_b;
  logic signed [9:0] exp_a, exp_b;
  always_comb begin
    mp_a = a_sub ? msb23(ma) : 0;
    mp_b = b_sub ? msb23(mb) : 0;
    if (a_sub) begin
      sig_a = {1'b0, ma} << (5'd23 - mp_a[4:0]);  // leading 1 lands at bit 23
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

  // --- long divide: q_full = (sig_a << 28) / sig_b, exact ---
  // q_full range: sig_a/sig_b in [0.5, 2) -> (sig_a<<28)/sig_b in [2^27, 2^29).
  //   sig_a >= sig_b -> [2^28, 2^29), MSB at bit 28.
  //   sig_a <  sig_b -> [2^27, 2^28), MSB at bit 27.
  logic [51:0] dividend;
  logic [28:0] q_full;
  logic [23:0] remainder;
  assign dividend  = {sig_a, 28'd0};
  assign q_full    = 29'(dividend / {28'd0, sig_b});
  assign remainder = 24'(dividend - {23'd0, q_full} * {28'd0, sig_b});

  logic a_ge_b;
  assign a_ge_b = (sig_a >= sig_b);

  // --- extract mantissa + RNE on the regime ULP ---
  logic [23:0] mant24_pre;     // 24-bit mantissa pre-rounding (incl. hidden 1)
  logic        rb, st, ru;
  logic [24:0] mant24_rnd;     // 25-bit to capture rounding carry
  logic signed [10:0] exp_y_unb;
  always_comb begin
    if (a_ge_b) begin
      // q_full MSB at 28 -> mant24 = q_full[28:5], rb=q_full[4], st=|q_full[3:0]|remainder
      mant24_pre = q_full[28:5];
      rb         = q_full[4];
      st         = (|q_full[3:0]) | (|remainder);
      exp_y_unb  = exp_a - exp_b;
    end else begin
      // q_full MSB at 27 -> mant24 = q_full[27:4], rb=q_full[3], st=|q_full[2:0]|remainder
      mant24_pre = q_full[27:4];
      rb         = q_full[3];
      st         = (|q_full[2:0]) | (|remainder);
      exp_y_unb  = exp_a - exp_b - 11'sd1;
    end
    ru         = rb & (st | mant24_pre[0]);
    mant24_rnd = {1'b0, mant24_pre} + {24'd0, ru};
  end

  // --- pack ---
  logic               carry;
  logic signed [10:0] exp_y_final_unb;
  logic [7:0]         exp_y_biased;
  always_comb begin
    carry           = mant24_rnd[24];
    exp_y_final_unb = exp_y_unb + (carry ? 11'sd1 : 11'sd0);
    exp_y_biased    = 8'(exp_y_final_unb + 11'sd127);
  end

  // Subnormal/underflow path: compute mantissa + round + sticky FRESH at the
  // subnormal regime ULP (avoiding double-rounding through the normal-regime
  // mant24_rnd). The subnormal mant LSB sits at q_full bit position
  //   k_lsb = -(exp_a - exp_b) - 121 = exp_b - exp_a - 121
  // (a_ge_b agnostic — the formula maps q_full-bit-weight to result-value-weight
  //  2^-149 directly). For k_lsb > 28: result is below smallest subnormal (only
  // round-up to ±smallest-sub possible based on sticky from all of q_full).
  logic signed [11:0] k_lsb_s;
  logic [4:0]         k_lsb;
  logic [28:0]        q_shifted_sub;
  logic [22:0]        mant_sub_pre;
  logic               sub_rb, sub_st, sub_ru;
  logic [23:0]        mant_sub_rnd;
  always_comb begin
    k_lsb_s       = -{{2{exp_a[9]}}, exp_a} + {{2{exp_b[9]}}, exp_b} - 12'sd121;
    // Saturate shift amount to 29 (one above q_full's MSB) so q_shifted_sub
    // becomes 0 for any deeper underflow. Wider shifts are equivalent in mant.
    k_lsb         = (k_lsb_s > 12'sd29) ? 5'd29 : k_lsb_s[4:0];
    q_shifted_sub = q_full >> k_lsb;
    mant_sub_pre  = q_shifted_sub[22:0];

    // Round bit at q_full[k_lsb-1]; absent (=0) when k_lsb=0 or above q_full's MSB.
    if (k_lsb_s >= 12'sd1 && k_lsb_s <= 12'sd29)
      sub_rb = q_full[k_lsb - 5'd1];
    else
      sub_rb = 1'b0;

    // Sticky: OR of q_full bits BELOW the round bit + |remainder.
    //   k_lsb_s > 29: round bit absent -> sticky = all of q_full | remainder.
    //   k_lsb_s >= 2:  sticky = q_full[k_lsb-2:0] | remainder
    //   k_lsb_s == 1:  no q_full bits below round; sticky only from remainder.
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

  always_comb begin
    // Non-finite contract
    if (a_nan || b_nan) begin
      y = QNAN;
    end else if (a_inf && b_inf) begin
      y = QNAN;
    end else if (a_zero && b_zero) begin
      y = QNAN;
    end else if (a_inf) begin
      y = {sa ^ sb, 8'd255, 23'd0};                    // inf/finite
    end else if (b_inf) begin
      y = {sa ^ sb, 8'd0,   23'd0};                    // finite/inf -> ±0
    end else if (b_zero) begin
      y = {sa ^ sb, 8'd255, 23'd0};                    // x/0 -> ±inf
    end else if (a_zero) begin
      y = {sa ^ sb, 8'd0,   23'd0};                    // 0/x -> ±0
    end else if (exp_y_final_unb >= 11'sd128) begin
      y = {sa ^ sb, 8'd255, 23'd0};                    // overflow -> ±inf
    end else if (exp_y_final_unb < -11'sd126) begin
      // Subnormal / underflow: mant_sub_rnd[23] set means rounding carried into
      // the implicit bit -> result is smallest normal {s, 1, 0}.
      if (mant_sub_rnd[23])
        y = {sa ^ sb, 8'd1, 23'd0};
      else
        y = {sa ^ sb, 8'd0, mant_sub_rnd[22:0]};
    end else if (carry) begin
      // Carry: mantissa becomes 2^24 (was 2^24, now after carry is 2^24 with
      // bit 24 set). The result is 1.0 * 2^(exp+1), so mantissa = 0.
      y = {sa ^ sb, exp_y_biased, 23'd0};
    end else begin
      y = {sa ^ sb, exp_y_biased, mant24_rnd[22:0]};
    end
  end

endmodule

`endif // FP32_DIV_SV
