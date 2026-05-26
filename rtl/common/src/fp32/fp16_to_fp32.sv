// Synthesizable IEEE half (fp16) -> IEEE single (fp32) converter — every
// fp16 value has an EXACT fp32 representation, so no rounding is needed.
// Bit-exact to the DPI golden `sfu_fp16_bits_to_fp32` (testbench.h):
//   numpy half bit-pattern -> Python float32 (via _Float16).
//
// Layout:
//   fp16:  [s:1 | e:5 | m:10]   bias=15
//   fp32:  [s:1 | e:8 | m:23]   bias=127
//
// Cases:
//   exp==0, mant==0  -> ±0
//   exp==0, mant!=0  -> subnormal fp16: value = mant * 2^-24
//                       Normalize: find MSB of mant in [0,9]; result is a
//                       normal fp32 with exp32 = msb_pos + 103, mant23 =
//                       (mant << (14 - msb_pos))[22:0] (zero-pad to 23 bits).
//   exp in [1,30]    -> normal: exp32 = e + 112; mant23 = mant << 13
//   exp==31, mant==0 -> ±inf
//   exp==31, mant!=0 -> qNaN canonicalized to 0x7FC00000
//
// Pure combinational.

`ifndef FP16_TO_FP32_SV
`define FP16_TO_FP32_SV

module fp16_to_fp32 (
  input  logic [15:0] a,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // most-significant set-bit index of a 10-bit value (-1 if zero)
  function automatic int unsigned msb10(input logic [9:0] v);
    int i;
    begin
      for (i = 9; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  logic        s;
  logic [4:0]  e;
  logic [9:0]  m;
  assign s = a[15];
  assign e = a[14:10];
  assign m = a[9:0];

  logic        is_zero, is_inf, is_nan, is_sub;
  assign is_zero = (e == 5'd0)  && (m == 10'd0);
  assign is_sub  = (e == 5'd0)  && (m != 10'd0);
  assign is_inf  = (e == 5'd31) && (m == 10'd0);
  assign is_nan  = (e == 5'd31) && (m != 10'd0);

  // Normalize subnormal: find leading 1 in mant (msb_p in [0,9]).
  // value = mant * 2^-24 = (1.frac) * 2^(msb_p - 24)
  // -> exp32 = msb_p + 103,  mant23 = frac bits at top of 23-bit field.
  // Shift amount = 23 - msb_p (in [14, 23]); after shifting m to bit 23,
  // the implicit leading 1 lands at bit 23 and is dropped by the [22:0] slice.
  int unsigned msb_p;
  logic [4:0]  shamt_u;
  logic [23:0] m_shifted;
  logic [22:0] sub_mant23;
  logic [7:0]  sub_exp32;
  always_comb begin
    msb_p      = msb10(m);
    shamt_u    = 5'd23 - msb_p[4:0];
    m_shifted  = {14'd0, m} << shamt_u;
    sub_mant23 = m_shifted[22:0];
    sub_exp32  = 8'd103 + {3'd0, msb_p[4:0]};
  end

  always_comb begin
    if (is_zero)      y = {s, 8'd0,   23'd0};
    else if (is_inf)  y = {s, 8'd255, 23'd0};
    else if (is_nan)  y = QNAN;
    else if (is_sub)  y = {s, sub_exp32, sub_mant23};
    else /* normal */ y = {s, (8'(e) + 8'd112), m, 13'd0};
  end

endmodule

`endif // FP16_TO_FP32_SV
