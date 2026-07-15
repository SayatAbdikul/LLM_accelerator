// 18-stage pipelined fp32 exp(x) — BIT-EXACT to the combinational `fp32_exp`
// (and to the DPI golden band it defines). Pure retiming: identical unpack,
// Cody-Waite range reduction, degree-6 Horner polynomial, and 2^k exponent
// scale — the SAME fp32_add / fp32_mul cores and the SAME integer glue as
// fp32_exp.sv, with pipeline registers inserted between the (already-present)
// combinational primitives. No arithmetic changes ⇒ the output is bit-IDENTICAL
// to fp32_exp, not merely within-band. test_fp32_exp_p18.cpp is the zero-diff
// gate (drives the combinational fp32_exp and this module with the same stream
// and asserts equality on the LATENCY=18 deque).
//
// WHY THIS EXISTS (T3 step 0, docs/t1_measured_redirect.md / T0.3 audit):
//   fp32_exp is a ~16-serial-fp-op combinational cloud (~412 ns). It sits on
//   THREE single-cycle SFU reg-to-reg paths — the softmax EXPSUM accumulate
//   (`sfu_synth_datapath.svh:400`, add→exp→add), the SOFTMAX_ATTNV weight
//   (`:568`, add→exp→div), and GELU (`fp32_gelu_new.sv:65`, …→exp→add→div→…).
//   Each is ~490–520 ns, so the full-SFU critical path is exp, NOT the div/sqrt
//   primitives the committed 34.41 MHz was measured on (the full-SFU PNR OOMs,
//   so exp was never in a timing report). Pipelining exp closes that latent
//   hole; each stage below carries at most ONE fp primitive (fp32_add 28.49 ns /
//   fp32_mul 27.34 ns < the 29.06 ns @34.41 MHz budget). Consumers feed one
//   element/cycle and collect LATENCY later (same software-pipeline as the
//   lever-E fp32_div_p6 softmax-OUT drain), preserving accumulate ORDER ⇒
//   byte-exact. Full-SFU fmax closure remains PNR-gated (≥24 GB); because this
//   changes datapath STRUCTURE (not primitive internals like lever E), the
//   standalone per-stage STA is a weaker proxy than lever E's was.
//
// INTERFACE: fixed LATENCY=18, mirrors fp32_div_p6. Present (a, valid_in) on
//   cycle N; (y, valid_out) are valid on cycle N+18. Fully pipelined — accepts a
//   new input every cycle.
//
// STAGE MAP (one fp primitive per stage; cheap integer glue rides an fp stage or
//   its own where it must precede one):
//   s0  unpack + exception flags + mul1 (a*log2e = k_f)
//   s1  k_int = round(k_f)                        [integer]
//   s2  k_fp32 = (float)k_int                     [integer]
//   s3  mul2_hi (k_fp32*ln2_hi), mul2_lo (k_fp32*ln2_lo)   [parallel, 1 stage]
//   s4  sub_hi (a - k_ln2_hi = r_hi)
//   s5  sub_lo (r_hi - k_ln2_lo = r)
//   s6  m0 (r*1/720)      s7  a1(+1/120)   s8  m1(*r)   s9  a2(+1/24)
//   s10 m2(*r)   s11 a3(+1/6)   s12 m3(*r)   s13 a4(+1/2)   s14 m4(*r)
//   s15 a5(+1)   s16 m5(*r)   s17 a6(+1) = exp_r
//   s18 (combinational, no reg) exponent scale + subnormal + output mux = y
//   ⇒ 18 register stages (after s0..s17), s18 drives y combinationally.

`ifndef FP32_EXP_P18_SV
`define FP32_EXP_P18_SV

// Dependencies fp32_add / fp32_mul are read in order by the parent build
// (FP32_PRIMS) or the standalone gate's -I, exactly like fp32_exp.sv.

module fp32_exp_p18 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  output logic        valid_out,
  output logic [31:0] y
);
  localparam int LATENCY = 18;

  localparam logic [31:0] QNAN     = 32'h7FC0_0000;
  localparam logic [31:0] POS_INF  = 32'h7F80_0000;
  localparam logic [31:0] POS_ZERO = 32'h0000_0000;
  localparam logic [31:0] C_LOG2E  = 32'h3FB8_AA3B;  // log2(e)
  localparam logic [31:0] C_LN2_HI = 32'h3F31_7200;
  localparam logic [31:0] C_LN2_LO = 32'h35BF_BE8E;
  localparam logic [31:0] C_ONE    = 32'h3F80_0000;
  localparam logic [31:0] C_HALF   = 32'h3F00_0000;
  localparam logic [31:0] C_1_6    = 32'h3E2A_AAAB;
  localparam logic [31:0] C_1_24   = 32'h3D2A_AAAB;
  localparam logic [31:0] C_1_120  = 32'h3C08_8889;
  localparam logic [31:0] C_1_720  = 32'h3AB6_0B61;

  // Exception flags carried s0->s18. `sa` (input sign) is needed only for the
  // a_inf case; the normal path's sign is exp_r's sign, computed at s18.
  typedef struct packed {
    logic nan_f;
    logic inf_f;
    logic zero_f;
    logic ovf_f;
    logic unf_f;
    logic sa;
  } flags_t;

  // valid shift register: valid_in at cycle N -> valid_out at N+LATENCY.
  logic [LATENCY-1:0] vld_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) vld_q <= '0;
    else        vld_q <= {vld_q[LATENCY-2:0], valid_in};
  end
  assign valid_out = vld_q[LATENCY-1];

  // ================= STAGE 0 (comb): unpack + flags + mul1 ================
  logic        s0_sa;
  logic [7:0]  s0_ea;
  logic [22:0] s0_ma;
  assign s0_sa = a[31];
  assign s0_ea = a[30:23];
  assign s0_ma = a[22:0];
  flags_t s0_flags;
  assign s0_flags.zero_f = (s0_ea == 8'd0)   && (s0_ma == 23'd0);
  assign s0_flags.inf_f  = (s0_ea == 8'd255) && (s0_ma == 23'd0);
  assign s0_flags.nan_f  = (s0_ea == 8'd255) && (s0_ma != 23'd0);
  assign s0_flags.ovf_f  = !s0_sa && (a >  32'h42B1_7218);
  assign s0_flags.unf_f  =  s0_sa && ((a & 32'h7FFF_FFFF) > 32'h42CF_F1B5);
  assign s0_flags.sa     = s0_sa;
  logic [31:0] s0_kf_w;
  fp32_mul u_mul1 (.a(a), .b(C_LOG2E), .y(s0_kf_w));

  logic [31:0] r0_a, r0_kf;
  flags_t      r0_flags;
  always_ff @(posedge clk) begin
    r0_a     <= a;
    r0_kf    <= s0_kf_w;
    r0_flags <= s0_flags;
  end

  // ================= STAGE 1 (comb): k_int = round(k_f) ===================
  logic        s1_ksgn;
  logic [7:0]  s1_kexp;
  logic [22:0] s1_kmant;
  logic signed [9:0] s1_kint_w;
  assign s1_ksgn  = r0_kf[31];
  assign s1_kexp  = r0_kf[30:23];
  assign s1_kmant = r0_kf[22:0];
  always_comb begin
    if (s1_kexp < 8'd126) begin
      s1_kint_w = 10'sd0;
    end else begin
      automatic logic [4:0]  kshamt;
      automatic logic [23:0] ksig24;
      automatic logic [9:0]  kabs_int;
      automatic logic        kg, kst, kru;
      ksig24 = {1'b1, s1_kmant};
      kshamt = 5'(8'd150 - s1_kexp);
      if (kshamt == 5'd0) begin
        kabs_int = ksig24[9:0];
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
      s1_kint_w = s1_ksgn ? 10'(-$signed({1'b0, kabs_int})) : 10'($signed({1'b0, kabs_int}));
    end
  end

  logic [31:0]       r1_a;
  flags_t            r1_flags;
  logic signed [9:0] r1_kint;
  always_ff @(posedge clk) begin
    r1_a     <= r0_a;
    r1_flags <= r0_flags;
    r1_kint  <= s1_kint_w;
  end

  // ================= STAGE 2 (comb): k_fp32 = (float)k_int ================
  logic [9:0]  s2_kabs;
  logic        s2_kabs_sign;
  logic [3:0]  s2_kp;
  logic [22:0] s2_km23;
  logic [7:0]  s2_ke8;
  logic [22:0] s2_km23_shifted;
  logic [31:0] s2_kfp32_w;
  always_comb begin
    s2_kabs_sign = (r1_kint < 10'sd0);
    s2_kabs      = s2_kabs_sign ? 10'(-r1_kint) : 10'(r1_kint);
    s2_kp = 4'd0;
    for (int j = 9; j >= 0; j = j - 1)
      if (s2_kabs[j]) begin
        s2_kp = j[3:0];
        break;
      end
    s2_ke8          = 8'd127 + {4'd0, s2_kp};
    s2_km23_shifted = ({13'd0, s2_kabs} << (5'd23 - {1'b0, s2_kp}));
    s2_km23         = s2_km23_shifted & 23'h7F_FFFF;
    if (s2_kabs == 10'd0)
      s2_kfp32_w = 32'd0;
    else
      s2_kfp32_w = {s2_kabs_sign, s2_ke8, s2_km23};
  end

  logic [31:0]       r2_a, r2_kfp32;
  flags_t            r2_flags;
  logic signed [9:0] r2_kint;
  always_ff @(posedge clk) begin
    r2_a     <= r1_a;
    r2_flags <= r1_flags;
    r2_kint  <= r1_kint;
    r2_kfp32 <= s2_kfp32_w;
  end

  // ================= STAGE 3 (comb): k*ln2_hi, k*ln2_lo ===================
  logic [31:0] s3_klnhi_w, s3_klnlo_w;
  fp32_mul u_mul2_hi (.a(r2_kfp32), .b(C_LN2_HI), .y(s3_klnhi_w));
  fp32_mul u_mul2_lo (.a(r2_kfp32), .b(C_LN2_LO), .y(s3_klnlo_w));

  logic [31:0]       r3_a, r3_klnhi, r3_klnlo;
  flags_t            r3_flags;
  logic signed [9:0] r3_kint;
  always_ff @(posedge clk) begin
    r3_a     <= r2_a;
    r3_flags <= r2_flags;
    r3_kint  <= r2_kint;
    r3_klnhi <= s3_klnhi_w;
    r3_klnlo <= s3_klnlo_w;
  end

  // ================= STAGE 4 (comb): r_hi = a - k*ln2_hi ==================
  logic [31:0] s4_rhi_w;
  fp32_add u_sub_hi (.a(r3_a), .b({~r3_klnhi[31], r3_klnhi[30:0]}), .y(s4_rhi_w));

  logic [31:0]       r4_klnlo, r4_rhi;
  flags_t            r4_flags;
  logic signed [9:0] r4_kint;
  always_ff @(posedge clk) begin
    r4_flags <= r3_flags;
    r4_kint  <= r3_kint;
    r4_klnlo <= r3_klnlo;
    r4_rhi   <= s4_rhi_w;
  end

  // ================= STAGE 5 (comb): r = r_hi - k*ln2_lo ==================
  logic [31:0] s5_r_w;
  fp32_add u_sub_lo (.a(r4_rhi), .b({~r4_klnlo[31], r4_klnlo[30:0]}), .y(s5_r_w));

  logic [31:0]       r5_r;
  flags_t            r5_flags;
  logic signed [9:0] r5_kint;
  always_ff @(posedge clk) begin
    r5_flags <= r4_flags;
    r5_kint  <= r4_kint;
    r5_r     <= s5_r_w;
  end

  // ================= STAGES 6..17: degree-6 Horner poly ===================
  // Each stage = one fp primitive; `r` and (flags,kint) ride along until used.
  // s6 m0
  logic [31:0] s6_t0a_w;  fp32_mul m0 (.a(r5_r), .b(C_1_720), .y(s6_t0a_w));
  logic [31:0] r6_r, r6_t0a;  flags_t r6_flags;  logic signed [9:0] r6_kint;
  always_ff @(posedge clk) begin
    r6_flags <= r5_flags; r6_kint <= r5_kint; r6_r <= r5_r; r6_t0a <= s6_t0a_w;
  end
  // s7 a1
  logic [31:0] s7_t1s_w;  fp32_add a1 (.a(r6_t0a), .b(C_1_120), .y(s7_t1s_w));
  logic [31:0] r7_r, r7_t1s;  flags_t r7_flags;  logic signed [9:0] r7_kint;
  always_ff @(posedge clk) begin
    r7_flags <= r6_flags; r7_kint <= r6_kint; r7_r <= r6_r; r7_t1s <= s7_t1s_w;
  end
  // s8 m1
  logic [31:0] s8_t1m_w;  fp32_mul m1 (.a(r7_t1s), .b(r7_r), .y(s8_t1m_w));
  logic [31:0] r8_r, r8_t1m;  flags_t r8_flags;  logic signed [9:0] r8_kint;
  always_ff @(posedge clk) begin
    r8_flags <= r7_flags; r8_kint <= r7_kint; r8_r <= r7_r; r8_t1m <= s8_t1m_w;
  end
  // s9 a2
  logic [31:0] s9_t2s_w;  fp32_add a2 (.a(r8_t1m), .b(C_1_24), .y(s9_t2s_w));
  logic [31:0] r9_r, r9_t2s;  flags_t r9_flags;  logic signed [9:0] r9_kint;
  always_ff @(posedge clk) begin
    r9_flags <= r8_flags; r9_kint <= r8_kint; r9_r <= r8_r; r9_t2s <= s9_t2s_w;
  end
  // s10 m2
  logic [31:0] s10_t2m_w;  fp32_mul m2 (.a(r9_t2s), .b(r9_r), .y(s10_t2m_w));
  logic [31:0] r10_r, r10_t2m;  flags_t r10_flags;  logic signed [9:0] r10_kint;
  always_ff @(posedge clk) begin
    r10_flags <= r9_flags; r10_kint <= r9_kint; r10_r <= r9_r; r10_t2m <= s10_t2m_w;
  end
  // s11 a3
  logic [31:0] s11_t3s_w;  fp32_add a3 (.a(r10_t2m), .b(C_1_6), .y(s11_t3s_w));
  logic [31:0] r11_r, r11_t3s;  flags_t r11_flags;  logic signed [9:0] r11_kint;
  always_ff @(posedge clk) begin
    r11_flags <= r10_flags; r11_kint <= r10_kint; r11_r <= r10_r; r11_t3s <= s11_t3s_w;
  end
  // s12 m3
  logic [31:0] s12_t3m_w;  fp32_mul m3 (.a(r11_t3s), .b(r11_r), .y(s12_t3m_w));
  logic [31:0] r12_r, r12_t3m;  flags_t r12_flags;  logic signed [9:0] r12_kint;
  always_ff @(posedge clk) begin
    r12_flags <= r11_flags; r12_kint <= r11_kint; r12_r <= r11_r; r12_t3m <= s12_t3m_w;
  end
  // s13 a4
  logic [31:0] s13_t4s_w;  fp32_add a4 (.a(r12_t3m), .b(C_HALF), .y(s13_t4s_w));
  logic [31:0] r13_r, r13_t4s;  flags_t r13_flags;  logic signed [9:0] r13_kint;
  always_ff @(posedge clk) begin
    r13_flags <= r12_flags; r13_kint <= r12_kint; r13_r <= r12_r; r13_t4s <= s13_t4s_w;
  end
  // s14 m4
  logic [31:0] s14_t4m_w;  fp32_mul m4 (.a(r13_t4s), .b(r13_r), .y(s14_t4m_w));
  logic [31:0] r14_r, r14_t4m;  flags_t r14_flags;  logic signed [9:0] r14_kint;
  always_ff @(posedge clk) begin
    r14_flags <= r13_flags; r14_kint <= r13_kint; r14_r <= r13_r; r14_t4m <= s14_t4m_w;
  end
  // s15 a5
  logic [31:0] s15_t5s_w;  fp32_add a5 (.a(r14_t4m), .b(C_ONE), .y(s15_t5s_w));
  logic [31:0] r15_r, r15_t5s;  flags_t r15_flags;  logic signed [9:0] r15_kint;
  always_ff @(posedge clk) begin
    r15_flags <= r14_flags; r15_kint <= r14_kint; r15_r <= r14_r; r15_t5s <= s15_t5s_w;
  end
  // s16 m5 (last use of r)
  logic [31:0] s16_t5m_w;  fp32_mul m5 (.a(r15_t5s), .b(r15_r), .y(s16_t5m_w));
  logic [31:0] r16_t5m;  flags_t r16_flags;  logic signed [9:0] r16_kint;
  always_ff @(posedge clk) begin
    r16_flags <= r15_flags; r16_kint <= r15_kint; r16_t5m <= s16_t5m_w;
  end
  // s17 a6 -> exp_r
  logic [31:0] s17_expr_w;  fp32_add a6 (.a(r16_t5m), .b(C_ONE), .y(s17_expr_w));
  logic [31:0] r17_expr;  flags_t r17_flags;  logic signed [9:0] r17_kint;
  always_ff @(posedge clk) begin
    r17_flags <= r16_flags; r17_kint <= r16_kint; r17_expr <= s17_expr_w;
  end

  // ============ STAGE 18 (comb): exp_r * 2^k_int + output mux ============
  logic               s18_s_er;
  logic [7:0]         s18_e_er;
  logic [22:0]        s18_m_er;
  logic signed [10:0] s18_e_scaled;
  assign s18_s_er = r17_expr[31];
  assign s18_e_er = r17_expr[30:23];
  assign s18_m_er = r17_expr[22:0];
  assign s18_e_scaled = $signed({3'b0, s18_e_er}) + {{1{r17_kint[9]}}, r17_kint};

  logic [4:0]  s18_sub_shamt;
  logic [23:0] s18_sub_sig_in;
  logic [23:0] s18_sub_sig_shifted;
  logic [22:0] s18_sub_mant;
  always_comb begin
    s18_sub_shamt       = 5'(11'sd1 - s18_e_scaled);
    s18_sub_sig_in      = {1'b1, s18_m_er};
    s18_sub_sig_shifted = s18_sub_sig_in >> s18_sub_shamt;
    s18_sub_mant        = s18_sub_sig_shifted[22:0];
  end

  always_comb begin
    if (r17_flags.nan_f) begin
      y = QNAN;
    end else if (r17_flags.inf_f) begin
      y = r17_flags.sa ? POS_ZERO : POS_INF;
    end else if (r17_flags.zero_f) begin
      y = C_ONE;
    end else if (r17_flags.ovf_f) begin
      y = POS_INF;
    end else if (r17_flags.unf_f) begin
      y = POS_ZERO;
    end else if (s18_e_scaled >= 11'sd255) begin
      y = POS_INF;
    end else if (s18_e_scaled <= -11'sd22) begin
      y = POS_ZERO;
    end else if (s18_e_scaled <= 11'sd0) begin
      y = {1'b0, 8'd0, s18_sub_mant};
    end else begin
      y = {s18_s_er, s18_e_scaled[7:0], s18_m_er};
    end
  end

endmodule

`endif // FP32_EXP_P18_SV
