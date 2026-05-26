// Synthesizable IEEE-754 binary32 adder — first brick of the synthesizable
// SFU primitive library (FPGA-demo roadmap Phase 2).
//
// Bit-exact target: the DPI golden `sfu_fp32_add` (testbench.h):
//   (double)(float)( (float)a + (float)b )
// i.e. IEEE-754 single-precision add, round-to-nearest-ties-to-even, gradual
// underflow (subnormals), overflow -> inf. NaN is canonicalized to the quiet
// NaN 0x7FC00000 (any NaN-involving add / inf-inf); the host produces a qNaN
// and the SFU 0-ULP fixtures are finite, so finite/inf/zero is the
// bit-exact-critical domain — NaN canonicalization matches the Option-B
// "designed, deterministic" non-finite philosophy the freeze adopted.
//
// Design (single-rounding, exact-alignment — double-rounding-free):
//   * value(op) = (-1)^s * sig * 2^(e-23), sig 24-bit (hidden bit explicit),
//     e = ((field==0)?1:field) - 127  (subnormal == normal exp=1, hidden 0).
//   * Operands ordered by magnitude (lexicographic (e,sig)).
//   * Significand placed as ext = sig<<27 (27 exact guard bits). The low 27
//     bits of ext are zero, so a right-shift by shamt<=27 loses NOTHING:
//     alignment is EXACT and no alignment sticky/borrow logic is needed.
//   * shamt>27  => the smaller operand is strictly < 0.5 ulp of the result
//     (operand1 is then necessarily normal): it cannot change the RNE result
//     nor create a tie, so it is dropped (result == operand1).
//   * value = (+/-) raw * 2^(e1-50). One round at lsb_pos =
//     max(p-23 [normal ulp], -99-e1 [subnormal ulp 2^-149]) — the coarser of
//     the two regimes, giving a single correctly-rounded result for both.
//
// Pure combinational. Standalone-gated by test_fp32_add (bit-exact vs host
// float over directed + millions random incl. subnormals/inf/NaN) before any
// sfu_engine.sv integration — zero risk to the proven byte-exact cosim.

`ifndef FP32_ADD_SV
`define FP32_ADD_SV

module fp32_add (
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // most-significant set-bit index of a 64-bit value (-1 if zero)
  function automatic int unsigned msb64(input logic [63:0] v);
    int i;
    begin
      for (i = 63; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0; // v==0 handled by caller (raw==0 -> +0)
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

  logic [23:0]       sig_a, sig_b;
  logic signed [9:0] exp_a, exp_b;
  assign sig_a = (ea == 8'd0) ? {1'b0, ma} : {1'b1, ma};
  assign sig_b = (eb == 8'd0) ? {1'b0, mb} : {1'b1, mb};
  assign exp_a = (ea == 8'd0) ? -10'sd126
                              : ($signed({2'b0, ea}) - 10'sd127);
  assign exp_b = (eb == 8'd0) ? -10'sd126
                              : ($signed({2'b0, eb}) - 10'sd127);

  // order so OP1 has the >= magnitude (lexicographic (exp, sig))
  logic a_ge;
  assign a_ge = (exp_a > exp_b) ||
                ((exp_a == exp_b) && (sig_a >= sig_b));

  logic        s1, s2;
  logic signed [9:0] e1, e2;
  logic [23:0] sg1, sg2;
  always_comb begin
    if (a_ge) begin
      s1 = sa; e1 = exp_a; sg1 = sig_a;
      s2 = sb; e2 = exp_b; sg2 = sig_b;
    end else begin
      s1 = sb; e1 = exp_b; sg1 = sig_b;
      s2 = sa; e2 = exp_a; sg2 = sig_a;
    end
  end

  logic eff_sub;
  assign eff_sub = (s1 ^ s2);

  // --- align (exact for shamt<=27; drop op2 for shamt>27) ---
  logic signed [10:0] shamt;
  assign shamt = e1 - e2;                       // >= 0 (op1 is larger)

  logic [63:0] ext1, ext2_al, raw;
  always_comb begin
    ext1 = {40'd0, sg1} << 27;
    if (shamt > 11'sd27)
      ext2_al = 64'd0;                          // op2 < 0.5 ulp -> negligible
    else
      ext2_al = ({40'd0, sg2} << 27) >> shamt[5:0];   // exact (low 27 bits 0)
    raw = eff_sub ? (ext1 - ext2_al) : (ext1 + ext2_al);
  end

  // --- exponent of |value| is fixed by raw (rounding only bumps it on a
  //     significand carry): exp0 = p + e1 - 50.  Round once at the regime
  //     ULP (normal: bit p-23; subnormal: bit -99-e1 == the 2^-149 step). ---
  logic                 res_zero;
  int unsigned          p;
  logic signed [11:0]   exp0;
  logic signed [11:0]   rpos;          // raw bit index of the result LSB
  logic [64:0]          sig;           // rounded significand (25-bit incl carry)
  logic                 g, st, ru;
  logic signed [11:0]   fexp;          // biased exponent field (signed pre-clamp)
  always_comb begin
    res_zero = (raw == 64'd0);
    p    = msb64(raw);
    exp0 = $signed({1'b0, p[10:0]}) + {{2{e1[9]}}, e1} - 12'sd50;

    // round position in raw: normal -> p-23; subnormal -> -99-e1.
    if (exp0 >= -12'sd126)
      rpos = $signed({1'b0, p[10:0]}) - 12'sd23;
    else
      rpos = -12'sd99 - {{2{e1[9]}}, e1};

    if (rpos <= 12'sd0) begin
      sig = {1'b0, raw} << (-rpos[5:0]);              // exact (low bits zero)
      g   = 1'b0;
      st  = 1'b0;
    end else begin
      sig = {1'b0, (raw >> rpos[5:0])};
      g   = raw[rpos[5:0] - 6'd1];
      st  = (rpos > 12'sd1)
              ? (|(raw & ((64'd1 << (rpos[5:0] - 6'd1)) - 64'd1)))
              : 1'b0;
    end
    ru  = g & (st | sig[0]);
    sig = sig + {64'd0, ru};

    // significand carry (2^24 normal / 2^23 subnormal) bumps the exponent.
    if (exp0 >= -12'sd126)
      fexp = (sig[24] ? (exp0 + 12'sd1) : exp0) + 12'sd127;
    else
      fexp = sig[23] ? 12'sd1 : 12'sd0;               // 0 == subnormal field
  end

  // --- assemble ---
  always_comb begin
    if (a_nan || b_nan) begin
      y = QNAN;
    end else if (a_inf && b_inf) begin
      y = (sa == sb) ? {sa, 8'd255, 23'd0} : QNAN;
    end else if (a_inf) begin
      y = {sa, 8'd255, 23'd0};
    end else if (b_inf) begin
      y = {sb, 8'd255, 23'd0};
    end else if (a_zero && b_zero) begin
      y = {(sa & sb), 8'd0, 23'd0};              // -0 only if both -0 (RNE)
    end else if (res_zero) begin
      y = 32'd0;                                 // exact cancellation -> +0
    end else if (fexp >= 12'sd255) begin
      y = {s1, 8'd255, 23'd0};                   // overflow -> inf
    end else if (exp0 >= -12'sd126) begin
      // normal: 24-bit significand sig[23:0] (carry -> sig[24], mant 0).
      y = sig[24] ? {s1, fexp[7:0], 23'd0}
                  : {s1, fexp[7:0], sig[22:0]};
    end else begin
      // subnormal: value == sig * 2^-149 (carry to 2^23 -> smallest normal).
      y = sig[23] ? {s1, 8'd1, 23'd0}
                  : {s1, 8'd0, sig[22:0]};
    end
  end

endmodule

`endif // FP32_ADD_SV
