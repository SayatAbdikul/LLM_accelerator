// Synthesizable fp32 exp(x) — measured-band approximation (freeze §7).
// Compared against the DPI golden `sfu_fp32_exp` ((float)std::exp((float)x))
// over a representative input range; the resulting ULP histogram becomes
// the committed band for this op.
//
// Algorithm: range reduction + degree-5 Taylor polynomial.
//   k = round(x * log2_e)         (nearest integer)
//   r = x - k * ln2                (|r| <= ln2/2 ≈ 0.347)
//   exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120   (Horner)
//   exp(x)  = exp(r) * 2^k         (apply 2^k via exponent-field add)
//
// Overflow/underflow:
//   x >  88.7 (~ln(FLT_MAX)) -> +inf
//   x < -87.3 (~ln(min_normal)) -> +0 (graceful underflow approximation)
//   NaN -> qNaN. -inf -> +0. +inf -> +inf.
//
// Instantiates the proven fp32_add and fp32_mul cores for the
// polynomial evaluation (combinational chain). fmax depends on the
// resulting critical path; deeper pipelining is Phase 2.8 work.

`ifndef FP32_EXP_SV
`define FP32_EXP_SV

`include "fp32_add.sv"
`include "fp32_mul.sv"

module fp32_exp (
  input  logic [31:0] a,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN     = 32'h7FC0_0000;
  localparam logic [31:0] POS_INF  = 32'h7F80_0000;
  localparam logic [31:0] POS_ZERO = 32'h0000_0000;
  // Constants:
  localparam logic [31:0] C_LOG2E  = 32'h3FB8_AA3B;  // log2(e)
  localparam logic [31:0] C_LN2    = 32'h3F31_7218;  // ln(2)
  localparam logic [31:0] C_ONE    = 32'h3F80_0000;  // 1.0
  localparam logic [31:0] C_HALF   = 32'h3F00_0000;  // 1/2
  localparam logic [31:0] C_1_6    = 32'h3E2A_AAAB;  // 1/6
  localparam logic [31:0] C_1_24   = 32'h3D2A_AAAB;  // 1/24
  localparam logic [31:0] C_1_120  = 32'h3C08_8889;  // 1/120

  // Classify input
  logic        sa;
  logic [7:0]  ea;
  logic [22:0] ma;
  assign sa = a[31];
  assign ea = a[30:23];
  assign ma = a[22:0];

  logic a_zero, a_inf, a_nan;
  assign a_zero = (ea == 8'd0)   && (ma == 23'd0);
  assign a_inf  = (ea == 8'd255) && (ma == 23'd0);
  assign a_nan  = (ea == 8'd255) && (ma != 23'd0);

  // Saturation thresholds in fp32 bit-patterns.
  //   Overflow boundary: exp(x) > FLT_MAX  -> +inf at x ≈  88.72   (bits 0x42B17218).
  //   Underflow boundary: exp(x) < 2^-149  ->  0   at x ≈ -103.97  (bits 0x42CFF1B5).
  // Between -103.97 and -87.34 the result is a subnormal — let the polynomial
  // path produce it; do NOT pre-saturate to 0.
  logic a_overflow, a_underflow;
  assign a_overflow  = !sa && (a >  32'h42B1_7218);
  assign a_underflow =  sa && ((a & 32'h7FFF_FFFF) > 32'h42CF_F1B5);

  // --- Step 1: k_f = a * log2_e (fp32) ---
  logic [31:0] k_f;
  fp32_mul u_mul1 (.a(a), .b(C_LOG2E), .y(k_f));

  // --- Step 2: k_int = round(k_f) to integer (in fp32 form), as signed int ---
  // For |x| < ~88, k_int is in roughly [-127, 127] which fits in 8 signed bits.
  // Use the existing fp32_quantize-style logic inline: unpack k_f, find integer.
  logic        ksgn;
  logic [7:0]  kexp;
  logic [22:0] kmant;
  logic signed [9:0] k_int;
  assign ksgn  = k_f[31];
  assign kexp  = k_f[30:23];
  assign kmant = k_f[22:0];
  always_comb begin
    // For values |k_f| < 0.5 (kexp < 126): k_int = 0.
    // For values |k_f| in [0.5, 256): kexp in [126, 134]; extract integer with RNE.
    if (kexp < 8'd126) begin
      k_int = 10'sd0;
    end else begin
      // shamt = 150 - kexp (number of bits to drop from 24-bit sig)
      automatic logic [4:0] kshamt;
      automatic logic [23:0] ksig24;
      automatic logic [9:0]  kabs_int;
      automatic logic        kg, kst, kru;
      ksig24  = {1'b1, kmant};
      kshamt  = 5'(8'd150 - kexp);   // for kexp in [126,150], kshamt in [0,24]
      if (kshamt == 5'd0) begin
        kabs_int = ksig24[9:0];  // can't actually happen for k in [-127,127]
        kg = 1'b0; kst = 1'b0;
      end else begin
        kabs_int = 10'((ksig24 >> kshamt));
        kg       = ksig24[kshamt - 5'd1];
        if (kshamt >= 5'd2)
          kst = |(ksig24 & ((24'd1 << (kshamt - 5'd1)) - 24'd1));
        else
          kst = 1'b0;
      end
      kru = kg & (kst | kabs_int[0]);
      kabs_int = kabs_int + {9'd0, kru};
      k_int = ksgn ? 10'(-$signed({1'b0, kabs_int})) : 10'($signed({1'b0, kabs_int}));
    end
  end

  // --- Step 3: r = a - k_int * ln2 ---
  // Convert k_int back to fp32 (via i32_to_fp32 inline logic) and multiply by ln2.
  // For simplicity: compute k_fp = (float)k_int directly (10-bit signed → fp32 is small).
  logic [31:0] k_fp32;
  logic [9:0]  kabs;
  logic        kabs_sign;
  logic [3:0]  kp;
  logic [22:0] km23;
  logic [7:0]  ke8;
  logic [22:0] km23_shifted;
  always_comb begin
    kabs_sign = (k_int < 10'sd0);
    kabs      = kabs_sign ? 10'(-k_int) : 10'(k_int);
    // Find MSB of kabs (10-bit unsigned)
    kp = 4'd0;
    for (int j = 9; j >= 0; j = j - 1)
      if (kabs[j]) begin
        kp = j[3:0];
        break;
      end
    ke8          = 8'd127 + {4'd0, kp};
    km23_shifted = ({13'd0, kabs} << (5'd23 - {1'b0, kp}));  // 23-bit shift result
    km23         = km23_shifted & 23'h7F_FFFF;
    if (kabs == 10'd0)
      k_fp32 = 32'd0;
    else
      k_fp32 = {kabs_sign, ke8, km23};
  end

  logic [31:0] k_ln2;
  fp32_mul u_mul2 (.a(k_fp32), .b(C_LN2), .y(k_ln2));
  logic [31:0] r;
  fp32_add u_sub1 (.a(a), .b({~k_ln2[31], k_ln2[30:0]}), .y(r));

  // --- Step 4: Polynomial p(r) = ((((r/120 + 1/24) r + 1/6) r + 1/2) r + 1) r + 1 ---
  // Horner from innermost:
  //   t0 = r * (1/120)
  //   t1 = (t0 + 1/24) * r
  //   t2 = (t1 + 1/6)  * r
  //   t3 = (t2 + 1/2)  * r
  //   t4 = (t3 + 1)    * r
  //   exp_r = t4 + 1
  logic [31:0] t0a, t1s, t1m, t2s, t2m, t3s, t3m, t4s, t4m, exp_r;
  fp32_mul m0  (.a(r),   .b(C_1_120), .y(t0a));
  fp32_add a1  (.a(t0a), .b(C_1_24),  .y(t1s));
  fp32_mul m1  (.a(t1s), .b(r),       .y(t1m));
  fp32_add a2  (.a(t1m), .b(C_1_6),   .y(t2s));
  fp32_mul m2  (.a(t2s), .b(r),       .y(t2m));
  fp32_add a3  (.a(t2m), .b(C_HALF),  .y(t3s));
  fp32_mul m3  (.a(t3s), .b(r),       .y(t3m));
  fp32_add a4  (.a(t3m), .b(C_ONE),   .y(t4s));
  fp32_mul m4  (.a(t4s), .b(r),       .y(t4m));
  fp32_add a5  (.a(t4m), .b(C_ONE),   .y(exp_r));

  // --- Step 5: exp(x) = exp_r * 2^k_int (apply k_int to exponent field) ---
  logic               s_er;
  logic [7:0]         e_er;
  logic [22:0]        m_er;
  logic signed [10:0] e_scaled;
  assign s_er = exp_r[31];
  assign e_er = exp_r[30:23];
  assign m_er = exp_r[22:0];
  assign e_scaled = $signed({3'b0, e_er}) + {{1{k_int[9]}}, k_int};

  // Subnormal output path: when e_scaled <= 0, shift the 24-bit significand
  // right by (1 - e_scaled) bits to fit into a 23-bit subnormal mantissa.
  logic [4:0]  sub_shamt;
  logic [23:0] sub_sig_in;
  logic [23:0] sub_sig_shifted;
  logic [22:0] sub_mant;
  always_comb begin
    sub_shamt       = 5'(11'sd1 - e_scaled);          // in [1, 23] for valid subnormal
    sub_sig_in      = {1'b1, m_er};
    sub_sig_shifted = sub_sig_in >> sub_shamt;
    sub_mant        = sub_sig_shifted[22:0];
  end

  always_comb begin
    if (a_nan) begin
      y = QNAN;
    end else if (a_inf) begin
      y = sa ? POS_ZERO : POS_INF;                 // exp(-inf)=0, exp(+inf)=+inf
    end else if (a_zero) begin
      y = C_ONE;                                   // exp(0) = 1
    end else if (a_overflow) begin
      y = POS_INF;
    end else if (a_underflow) begin
      y = POS_ZERO;
    end else if (e_scaled >= 11'sd255) begin
      y = POS_INF;
    end else if (e_scaled <= -11'sd22) begin
      y = POS_ZERO;                                // below smallest subnormal
    end else if (e_scaled <= 11'sd0) begin
      // Subnormal output (graceful underflow). Result sign is positive
      // (exp result is always positive in the polynomial regime).
      y = {1'b0, 8'd0, sub_mant};
    end else begin
      y = {s_er, e_scaled[7:0], m_er};
    end
  end

endmodule

`endif // FP32_EXP_SV
