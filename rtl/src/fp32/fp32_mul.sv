// Synthesizable IEEE-754 binary32 multiplier — bit-exact to the DPI golden
// `sfu_fp32_mul` (testbench.h):
//   (float)( (float)a * (float)b )  with NaN canonicalized to 0x7FC00000.
//
// Design — same single-rounding exact-alignment pattern as fp32_add.sv:
//   value(op)  = (-1)^s * sig * 2^(e-23), sig 24-bit (hidden bit explicit),
//                e = (field==0 ? 1 : field) - 127.
//   prod_value = (-1)^(sa^sb) * (sig_a * sig_b) * 2^(ea + eb - 46).
//   Let raw = sig_a * sig_b (48-bit unsigned, no alignment needed — multiply
//   is naturally exact).  Let p = MSB(raw); the result's unbiased exponent
//   is fixed by raw:  exp0 = p + ea + eb - 46  (rounding only bumps it on a
//   significand carry).  Single round at the regime ULP:
//     normal   -> bit p - 23 of raw
//     subnormal-> bit -(103 + ea + eb) of raw  (== the 2^-149 step).
//
// Non-finite contract (matches DPI / Option-B SFU philosophy):
//   any NaN involved -> qNaN 0x7FC00000
//   inf * 0          -> qNaN 0x7FC00000
//   inf * finite!=0  -> ±inf  (sign = sa^sb)
//   0   * finite     -> ±0    (sign = sa^sb; RNE preserves -0 only with both)
//
// Pure combinational. Standalone-gated by test_fp32_mul (and via the
// pipelined shell fp32_mul_pipe) bit-exact vs host float across directed
// edges + millions of randoms incl. subnormal/inf/NaN.

`ifndef FP32_MUL_SV
`define FP32_MUL_SV

module fp32_mul (
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // most-significant set-bit index of a 48-bit value (-1 if zero)
  function automatic int unsigned msb48(input logic [47:0] v);
    int i;
    begin
      for (i = 47; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0; // v==0 handled by caller (res_zero short-circuit)
    end
  endfunction

  // --- unpack ---
  logic        sa, sb;
  logic [7:0]  ea, eb;
  logic [22:0] ma, mb;
  assign sa = a[31];  assign ea = a[30:23];  assign ma = a[22:0];
  assign sb = b[31];  assign eb = b[30:23];  assign mb = b[22:0];

  logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
  assign a_zero = (ea == 8'd0)   && (ma == 23'd0);
  assign b_zero = (eb == 8'd0)   && (mb == 23'd0);
  assign a_inf  = (ea == 8'd255) && (ma == 23'd0);
  assign b_inf  = (eb == 8'd255) && (mb == 23'd0);
  assign a_nan  = (ea == 8'd255) && (ma != 23'd0);
  assign b_nan  = (eb == 8'd255) && (mb != 23'd0);

  logic [23:0]        sig_a, sig_b;
  logic signed [10:0] exp_a, exp_b, exp_sum;
  assign sig_a = (ea == 8'd0) ? {1'b0, ma} : {1'b1, ma};
  assign sig_b = (eb == 8'd0) ? {1'b0, mb} : {1'b1, mb};
  assign exp_a = (ea == 8'd0) ? -11'sd126
                              : ($signed({3'b0, ea}) - 11'sd127);
  assign exp_b = (eb == 8'd0) ? -11'sd126
                              : ($signed({3'b0, eb}) - 11'sd127);
  assign exp_sum = exp_a + exp_b;  // signed sum, fits in 11 bits.

  // --- multiply: 24 x 24 = 48-bit unsigned, exact ---
  logic [47:0] raw;
  assign raw = sig_a * sig_b;

  logic s_y;
  assign s_y = sa ^ sb;

  // --- exponent fixed by raw: exp0 = p + (ea + eb) - 46.  Round once at
  //     the regime ULP (normal: bit p-23 of raw; subnormal: bit -(103+ex)
  //     == the 2^-149 step). ---
  logic                 res_zero;
  int unsigned          p;
  logic signed [11:0]   exp0;
  logic signed [11:0]   rpos;
  logic [49:0]          sig;          // rounded significand (incl. carry bit)
  logic                 g, st, ru;
  logic signed [11:0]   fexp;         // biased exponent (signed pre-clamp)
  always_comb begin
    res_zero = (raw == 48'd0);
    p    = msb48(raw);
    exp0 = $signed({1'b0, p[10:0]}) + {{1{exp_sum[10]}}, exp_sum} - 12'sd46;

    if (exp0 >= -12'sd126)
      rpos = $signed({1'b0, p[10:0]}) - 12'sd23;
    else
      rpos = -12'sd103 - {{1{exp_sum[10]}}, exp_sum};

    if (rpos <= 12'sd0) begin
      // rpos<=0 cannot happen for fp32_mul in practice (the subnormal regime
      // has rpos >= 24 by the exp0<-126 constraint, and the normal regime has
      // rpos = p-23 with p>=23 since one input must be normal). Defensive
      // pass-through: no shift, no rounding.
      sig = {2'b00, raw};
      g   = 1'b0;
      st  = 1'b0;
    end else if (rpos >= 12'sd49) begin
      // Result LSB sits strictly ABOVE raw's MSB: kept mantissa is 0; round
      // bit is also 0 (above raw); sticky is set iff raw is nonzero.
      sig = 50'd0;
      g   = 1'b0;
      st  = (raw != 48'd0);
    end else if (rpos == 12'sd48) begin
      // Round bit is exactly raw's MSB (bit 47); sticky is raw[46:0].
      sig = 50'd0;
      g   = raw[47];
      st  = |raw[46:0];
    end else begin
      // rpos in [1, 47]: full-width shift count fits in 6 bits.
      sig = {2'b00, (raw >> rpos[5:0])};
      g   = raw[rpos[5:0] - 6'd1];
      st  = (rpos > 12'sd1)
              ? (|(raw & ((48'd1 << (rpos[5:0] - 6'd1)) - 48'd1)))
              : 1'b0;
    end
    ru  = g & (st | sig[0]);
    sig = sig + {49'd0, ru};

    if (exp0 >= -12'sd126)
      fexp = (sig[24] ? (exp0 + 12'sd1) : exp0) + 12'sd127;
    else
      fexp = sig[23] ? 12'sd1 : 12'sd0;               // 0 == subnormal field
  end

  // --- assemble ---
  always_comb begin
    if (a_nan || b_nan) begin
      y = QNAN;
    end else if ((a_inf && b_zero) || (a_zero && b_inf)) begin
      y = QNAN;                                       // inf * 0 = NaN
    end else if (a_inf || b_inf) begin
      y = {s_y, 8'd255, 23'd0};                       // sign(inf*finite) = sa^sb
    end else if (a_zero || b_zero) begin
      y = {s_y, 8'd0, 23'd0};                         // sign(0*finite) = sa^sb
    end else if (res_zero) begin
      y = {s_y, 8'd0, 23'd0};                         // (no finite*finite hits this)
    end else if (fexp >= 12'sd255) begin
      y = {s_y, 8'd255, 23'd0};                       // overflow -> inf
    end else if (exp0 >= -12'sd126) begin
      y = sig[24] ? {s_y, fexp[7:0], 23'd0}
                  : {s_y, fexp[7:0], sig[22:0]};
    end else begin
      y = sig[23] ? {s_y, 8'd1, 23'd0}                // carry to smallest normal
                  : {s_y, 8'd0, sig[22:0]};
    end
  end

endmodule

`endif // FP32_MUL_SV
