// Synthesizable signed-int32 -> fp32 conversion (round-to-nearest-ties-to-even).
// Bit-exact to the natural C/Verilator cast `(float)int32`. This replaces the
// SV-`real`-typed `sfu_fp32_round(real'(int32))` idiom used in blocking_helper
// (i32-from-ACCUM scaling) and any int->fp32 staging in the SFU rewrite.
//
// Algorithm (single rounding, no double-rounding):
//   1. Take |a| as a 32-bit unsigned magnitude (special-case INT32_MIN).
//   2. p = MSB(|a|) in [0,31]; result value = |a| * 2^0; in fp32 = 1.mant *
//      2^p, so unbiased exp = p, biased exp = 127 + p.
//   3. If p >= 23: shift right by (p-23) to form 24-bit significand
//      [bit-23 hidden=1, bit-22:0 mantissa]; the bits below are guard+sticky.
//   4. If p < 23:  shift left by (23-p); no rounding.
//   5. Round-half-to-even; significand carry bumps exp by 1.

`ifndef I32_TO_FP32_SV
`define I32_TO_FP32_SV

module i32_to_fp32 (
  input  logic signed [31:0] a,
  output logic [31:0]        y
);

  function automatic int unsigned msb32(input logic [31:0] v);
    int i;
    begin
      for (i = 31; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  logic        s;
  logic [31:0] abs_a;
  assign s = a[31];
  // |INT32_MIN| does not fit in int32 (2's complement); its bit-pattern
  // 0x80000000 already represents 2^31 as unsigned, which is correct.
  assign abs_a = a[31] ? (~a + 32'd1) : a;

  int unsigned p;
  logic [7:0]  exp_y;
  logic [23:0] sig24;
  logic        g, st, ru;
  logic [24:0] sig25;
  logic        is_zero;
  logic [31:0] shifted_r, shifted_l;

  always_comb begin
    is_zero   = (abs_a == 32'd0);
    p         = msb32(abs_a);
    exp_y     = 8'd127 + {3'd0, p[4:0]};
    shifted_r = abs_a >> (p[4:0] - 5'd23);
    shifted_l = {8'd0, abs_a[23:0]} << (5'd23 - p[4:0]);

    if (p > 23) begin
      sig24 = shifted_r[23:0];
      g     = abs_a[p[4:0] - 5'd24];
      st    = (p > 24)
                ? (|(abs_a & ((32'd1 << (p[4:0] - 5'd24)) - 32'd1)))
                : 1'b0;
    end else begin
      sig24 = shifted_l[23:0];
      g     = 1'b0;
      st    = 1'b0;
    end

    ru    = g & (st | sig24[0]);
    sig25 = {1'b0, sig24} + {24'd0, ru};
  end

  always_comb begin
    if (is_zero)        y = {s, 8'd0, 23'd0};
    else if (sig25[24]) y = {s, (exp_y + 8'd1), sig25[23:1]};
    else                y = {s, exp_y, sig25[22:0]};
  end

endmodule

`endif // I32_TO_FP32_SV
