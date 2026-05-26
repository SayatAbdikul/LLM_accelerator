// Synthesizable fp32 -> int8 quantizer (round-half-to-even + clamp +
// Option-B non-finite contract). Bit-exact to `sfu_fp32_quantize_i8` of
// `testbench.h` when out_scale=1.0 (the SFU integration calls fp32_div
// for value/out_scale first, then this primitive on the divided result).
//
// Option-B non-finite contract (freeze §7 item 8, P6g/#110):
//   NaN     -> 0
//   +inf    -> +127
//   -inf    -> -128
//   finite  -> RNE round to int, saturate(int, -128, 127)
//
// Round-half-to-even (banker's rounding) on the fraction bits of the fp32:
//   tie (exactly 0.5 ulp of result) -> nearest even integer.

`ifndef FP32_QUANTIZE_I8_SV
`define FP32_QUANTIZE_I8_SV

module fp32_quantize_i8 (
  input  logic [31:0]      a,
  output logic signed [7:0] y
);
  // --- unpack ---
  logic        s;
  logic [7:0]  e;
  logic [22:0] m;
  assign s = a[31];
  assign e = a[30:23];
  assign m = a[22:0];

  logic is_zero, is_inf, is_nan;
  assign is_zero = (e == 8'd0)   && (m == 23'd0);  // ±0 (subnormal -> finite-tiny, rounds to 0 below)
  assign is_inf  = (e == 8'd255) && (m == 23'd0);
  assign is_nan  = (e == 8'd255) && (m != 23'd0);

  // Unbiased exponent. For finite-rounding-to-int we only care if
  // |value| >= 0.5 (need rounding) or < 0.5 (rounds to 0).
  logic signed [9:0] E;
  assign E = $signed({2'b0, e}) - 10'sd127;

  // Reconstruct 24-bit significand (hidden 1 + 23-bit mantissa) for normal;
  // for subnormal, |value| < 2^-126 << 0.5, so output is 0 anyway.
  logic [23:0] sig24;
  assign sig24 = {1'b1, m};

  // Decision: |value| < 0.5 iff E < -1 (incl. fp32 subnormal which has E=-126).
  //           round_to_nearest_int producing nonzero requires E >= -1.
  //           At E = -1, |value| in [0.5, 1): tie at exact 0.5 (m==0).
  //           At E >= 23, |value| >= 2^23 -> all integer (no fraction).

  // Compute |int part| and round bits.
  // The fp32 value = sig24 * 2^(E - 23). For integer result:
  //   shamt = 23 - E (in [-?, 24])  (shift right of sig24 by shamt to land at integer position)
  //   shamt <= 0: |x| = sig24 << (-shamt) -- already integer, no rounding.
  //   shamt >  0: |x| = sig24 >> shamt; round bit at sig24[shamt-1]; sticky below.
  // For int8 saturation, we only need to know if |int| > 127 (after rounding).

  // Saturation detection: |value| >= 128 (= 2^7) <-> E >= 7.
  //   E >= 7 in finite normal: |intg| >= 128 -> saturate to ±127/-128.
  //   Edge case at E == 7 sign==1 mant==0: -128 exactly -> output -128 (not saturated).
  //   At E == 7 sign==0: 128 or more -> 127 sat.
  //   At E == 7 sign==1 mant!=0: < -128 -> -128 sat (RNE may also round to -128 in tie).

  logic [4:0] shamt;        // 5-bit unsigned shift, range [0, 24]
  logic [31:0] sig_shifted; // 32-bit holding the shifted integer (max value ~2^24)
  logic        rb, st, ru;
  logic [31:0] abs_int;     // pre-clamp signed magnitude (could be 0..128)

  always_comb begin
    // Default: |value| < 0.5 -> abs_int = 0.
    shamt       = 5'd0;
    sig_shifted = 32'd0;
    rb          = 1'b0;
    st          = 1'b0;
    abs_int     = 32'd0;

    if (E >= -10'sd1 && !is_zero && !is_inf && !is_nan && (e != 8'd0)) begin
      // Finite normal with |value| >= 0.5.
      if (E >= 10'sd23) begin
        // |value| >= 2^23 -> already integer; will saturate to ±127/-128 below
        // since 2^23 > 127. Just set abs_int huge.
        abs_int = 32'h7FFF_FFFF;
        rb      = 1'b0;
        st      = 1'b0;
      end else begin
        // shamt = 23 - E, range [0, 24].
        shamt       = 5'(10'sd23 - E);
        sig_shifted = {8'd0, sig24} >> shamt;
        abs_int     = sig_shifted;
        if (shamt >= 5'd1) begin
          rb = sig24[shamt - 5'd1];
          if (shamt >= 5'd2)
            st = |(sig24 & ((24'd1 << (shamt - 5'd1)) - 24'd1));
          else
            st = 1'b0;
        end else begin
          rb = 1'b0;
          st = 1'b0;
        end
      end
    end

    // RNE: round up iff round_bit && (sticky || lsb_of_abs_int)
    ru = rb & (st | abs_int[0]);
  end

  // Apply rounding then sign, then clamp to int8.
  logic [31:0] abs_rnd;
  logic signed [31:0] signed_val;
  always_comb begin
    abs_rnd    = abs_int + {31'd0, ru};
    signed_val = s ? -$signed(abs_rnd) : $signed(abs_rnd);
  end

  always_comb begin
    if (is_nan) begin
      y = 8'sd0;
    end else if (is_inf) begin
      y = s ? -8'sd128 : 8'sd127;
    end else if (signed_val > 32'sd127) begin
      y = 8'sd127;
    end else if (signed_val < -32'sd128) begin
      y = -8'sd128;
    end else begin
      y = signed_val[7:0];
    end
  end

endmodule

`endif // FP32_QUANTIZE_I8_SV
