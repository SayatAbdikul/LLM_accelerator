// Synthesizable fp32 -> IEEE half (fp16) converter — RNE rounding from
// 23-bit mantissa down to 10-bit (or shifted into the fp16 subnormal regime).
// Bit-exact to the DPI golden `sfu_fp32_to_fp16_bits` (testbench.h):
//   _Float16 h = (_Float16)((float)value_r); return *(uint16_t*)&h;
// i.e. one IEEE-754 RNE rounding from fp32 to fp16 (NaN canonicalized to fp16
// qNaN 0x7E00, the standard `_Float16` cast result on a NaN input).
//
// Algorithm (single rounding):
//   E = e - 127 (unbiased fp32 exponent for a normal input).
//   E > 15     -> ±inf
//   E in [-14, 15] -> fp16 normal: exp_h = E + 15; mant_h = RNE(m23, drop 13 bits)
//                    + carry handling (mant_h overflow -> exp_h++, possibly ±inf)
//   E in [-25, -15] -> fp16 subnormal: shift the 24-bit significand right by
//                    (shamt = -E - 1) in [14, 24]; round half-to-even on the
//                    dropped bits; carry -> smallest fp16 normal {s,1,0}.
//   E < -25    -> ±0 (no possible round-up; |value| < 2^-25, half-ulp of
//                 smallest subnormal, tie-to-even resolves to 0).
//   fp32 subnormal input -> ±0 (max ~2^-126 is far below 2^-25).
//   fp32 ±inf  -> fp16 ±inf
//   fp32 NaN   -> fp16 qNaN 0x7E00 (canonical).
//
// Pure combinational.

`ifndef FP32_TO_FP16_SV
`define FP32_TO_FP16_SV

module fp32_to_fp16 (
  input  logic [31:0] a,
  output logic [15:0] y
);
  localparam logic [15:0] QNAN_H = 16'h7E00;

  // --- unpack ---
  logic        s;
  logic [7:0]  e;
  logic [22:0] m;
  assign s = a[31];
  assign e = a[30:23];
  assign m = a[22:0];

  logic        a_zero, a_inf, a_nan, a_sub;
  assign a_zero = (e == 8'd0)   && (m == 23'd0);
  assign a_sub  = (e == 8'd0)   && (m != 23'd0);
  assign a_inf  = (e == 8'd255) && (m == 23'd0);
  assign a_nan  = (e == 8'd255) && (m != 23'd0);

  // Unbiased exponent (signed). Range for normals: [-126, 127].
  logic signed [9:0] E;
  assign E = $signed({2'b0, e}) - 10'sd127;

  // --- normal -> normal (E in [-14, 15]) ---
  logic [4:0]  exp_h_n;     // biased fp16 exp pre-carry
  logic [9:0]  mant_h_n;    // fp16 mant pre-carry
  logic        rb_n, st_n, ru_n;
  logic [10:0] mant_n_rnd;  // 11-bit to capture carry
  logic [4:0]  exp_n_final;
  logic [9:0]  mant_n_final;
  logic        ovfl_n;
  always_comb begin
    exp_h_n    = 5'(E + 10'sd15);          // E in [-14,15] -> [1,30]
    mant_h_n   = m[22:13];
    rb_n       = m[12];
    st_n       = |m[11:0];
    ru_n       = rb_n & (st_n | mant_h_n[0]);
    mant_n_rnd = {1'b0, mant_h_n} + {10'd0, ru_n};
    // Carry into bit 10: mantissa becomes 2^10 -> mant=0, exp++.
    if (mant_n_rnd[10]) begin
      exp_n_final  = exp_h_n + 5'd1;
      mant_n_final = 10'd0;
      // Detect overflow to inf when exp_h was already 30 (so +1 -> 31).
      ovfl_n       = (exp_h_n == 5'd30);
    end else begin
      exp_n_final  = exp_h_n;
      mant_n_final = mant_n_rnd[9:0];
      ovfl_n       = 1'b0;
    end
  end

  // --- normal -> subnormal (E in [-25, -15]) ---
  // shamt = -E - 1 in [14, 24]. Right-shift the 24-bit significand;
  // RNE on the dropped bits.
  logic [4:0]  shamt;
  logic [23:0] full_sig;
  logic [23:0] sub_shifted;
  logic [9:0]  mant_s_pre;
  logic        rb_s, st_s, ru_s;
  logic [10:0] mant_s_rnd;
  logic [4:0]  exp_s_final;
  logic [9:0]  mant_s_final;
  always_comb begin
    full_sig    = {1'b1, m};
    shamt       = 5'(-E - 10'sd1);                // 14..24
    sub_shifted = full_sig >> shamt;
    mant_s_pre  = sub_shifted[9:0];

    // Round bit at position (shamt-1) of full_sig.
    rb_s = full_sig[shamt - 5'd1];
    // Sticky: any bit below the round bit.
    if (shamt >= 5'd2)
      st_s = |(full_sig & ((24'd1 << (shamt - 5'd1)) - 24'd1));
    else
      st_s = 1'b0;

    ru_s       = rb_s & (st_s | mant_s_pre[0]);
    mant_s_rnd = {1'b0, mant_s_pre} + {10'd0, ru_s};
    // Carry into bit 10: subnormal mantissa overflowed -> smallest normal {s,1,0}.
    if (mant_s_rnd[10]) begin
      exp_s_final  = 5'd1;
      mant_s_final = 10'd0;
    end else begin
      exp_s_final  = 5'd0;
      mant_s_final = mant_s_rnd[9:0];
    end
  end

  // --- assemble ---
  always_comb begin
    if (a_nan) begin
      y = QNAN_H;
    end else if (a_inf) begin
      y = {s, 5'd31, 10'd0};
    end else if (a_zero || a_sub) begin
      y = {s, 5'd0, 10'd0};                       // fp32 ±0 or subnormal -> fp16 ±0
    end else if (E > 10'sd15) begin
      y = {s, 5'd31, 10'd0};                      // overflow -> ±inf
    end else if (E < -10'sd25) begin
      y = {s, 5'd0, 10'd0};                       // far underflow -> ±0
    end else if (E >= -10'sd14) begin             // normal-out
      y = ovfl_n ? {s, 5'd31, 10'd0}
                 : {s, exp_n_final, mant_n_final};
    end else begin                                // subnormal-out (E in [-25,-15])
      y = {s, exp_s_final, mant_s_final};
    end
  end

endmodule

`endif // FP32_TO_FP16_SV
