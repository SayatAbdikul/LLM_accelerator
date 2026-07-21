// 33-stage pipelined gen-2 GELU — BIT-EXACT to the combinational
// `fp32_gelu_new`. Pure retiming: the SAME ten fp primitives, in the SAME
// order, with the SAME operand port assignments; only pipeline registers are
// inserted between them (and `fp32_exp` / `fp32_div` are swapped for their
// already-proven bit-exact pipelined twins `fp32_exp_p18` / `fp32_div_p6`).
// No arithmetic changes ⇒ the output is bit-IDENTICAL to fp32_gelu_new, not
// merely within-band. test_fp32_gelu_p33.cpp is the zero-diff gate (drives the
// combinational core and this module with the same stream and asserts equality
// on the LATENCY=33 deque).
//
// WHY THIS EXISTS (fmax campaign rung 0, phase 0d):
//   fp32_gelu_new is the single largest combinational cloud in the SFU: a
//   10-deep chain of fp primitives that CONTAINS a full fp32_exp (~412 ns) and
//   a full fp32_div, replicated across all 8 elementwise lanes. Standalone STA
//   put it at ~700+ ns — i.e. a ~1.4 MHz path sitting in the netlist, live on
//   every mode-1 GELU. It has never appeared in a timing report (the full-SFU
//   flatten OOMs on this box), which is exactly why the quoted 34.41 MHz is a
//   div/sqrt-PRIMITIVE number and not a chip number. Phase 0b closed the same
//   hole on the softmax side; this closes the bigger one. After this, every
//   stage below carries at most ONE fp primitive (fp32_add 28.49 ns /
//   fp32_mul 27.34 ns), so the binder returns to the primitive floor.
//
// STAGE MAP (cycle N = the cycle during which that stage's logic evaluates;
//   its result is available in a register at cycle N+1):
//   s0   x_sq   = x * x                              [fp32_mul]
//   s1   x_cb   = x_sq * x(-1)                       [fp32_mul]
//   s2   c_x_cb = 0.044715 * x_cb                    [fp32_mul]
//   s3   inner  = x(-3) + c_x_cb                     [fp32_add]
//   s4   z      = K * inner                          [fp32_mul]
//   s5   z2     = z * 2.0                            [fp32_mul]
//   s6..s23    exp_2z = exp(z2)                      [fp32_exp_p18, 18 stages]
//   s24  (exp_p18's y is COMBINATIONAL off its r17 — its s18 output glue is an
//        11-bit add + 24-bit variable shift + priority mux, so it gets its own
//        capture stage rather than being chained into the next fp32_add. This
//        is the same call made in phase 0b for the softmax EXPSUM accumulate.)
//   s25  denom  = exp_2z + 1.0                       [fp32_add]  (+ exp_2z copy)
//   s26..s31   ratio = exp_2z / denom                [fp32_div_p6, 6 stages]
//   s32  y      = x(-32) * ratio                     [fp32_mul]  -> output reg
//   ⇒ LATENCY = 33.
//
// ALIGNMENT INVARIANT (the thing that makes stalls free): every datapath
// register here FREE-RUNS — no clock enables, no conditional shifts. Only the
// valid bits are gated. A consumer that feeds a bubble simply feeds valid=0;
// x and all of its dependents shift in lockstep regardless, so the delay-line
// taps stay aligned with the arithmetic under arbitrary stall patterns. Do NOT
// add an enable to a subset of these registers.
//
// INTERFACE: fixed LATENCY=33, mirrors fp32_exp_p18 / fp32_div_p6. Present
//   (a, valid_in) on cycle N; (y, valid_out) are valid on cycle N+33. Fully
//   pipelined — accepts a new input every cycle.

`ifndef FP32_GELU_P33_SV
`define FP32_GELU_P33_SV

// Dependencies (fp32_add, fp32_mul, fp32_exp_p18, fp32_div_p6) are read in
// order by the parent build (FP32_PRIMS in rtl/verilator/Makefile, or the
// standalone gate rule with -I), exactly like fp32_gelu_new.sv. Local
// `\`include` directives are intentionally absent so the CONTROL_SV path
// (which has no -I to fp32/) elaborates cleanly.

module fp32_gelu_p33 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,        // x
  output logic        valid_out,
  output logic [31:0] y         // gelu_new(x), LATENCY cycles later
);
  localparam int LATENCY = 33;

  localparam logic [31:0] C_ONE       = 32'h3F80_0000;   // 1.0
  localparam logic [31:0] C_TWO       = 32'h4000_0000;   // 2.0
  localparam logic [31:0] C_K_SQRT2PI = 32'h3F4C_4229;   // sqrt(2/pi)
  localparam logic [31:0] C_044715    = 32'h3D37_2713;   // 0.044715

  // ---- valid shift register: valid_in at cycle N -> valid_out at N+LATENCY --
  // The ONLY reset state in the module. Datapath regs free-run (see the
  // ALIGNMENT INVARIANT above); their contents are meaningless where the
  // matching valid bit is 0.
  logic [LATENCY-1:0] vld_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) vld_q <= '0;
    else        vld_q <= {vld_q[LATENCY-2:0], valid_in};
  end
  assign valid_out = vld_q[LATENCY-1];

  // ---- x delay line -------------------------------------------------------
  // x_dly[i] holds x delayed by (i+1) cycles. Taps used:
  //   x_dly[0]  (delay 1)  -> s1  x_sq * x
  //   x_dly[2]  (delay 3)  -> s3  x + c_x_cb
  //   x_dly[31] (delay 32) -> s32 x * ratio
  logic [31:0] x_dly [0:31];
  always_ff @(posedge clk) begin
    x_dly[0] <= a;
    for (int i = 1; i < 32; i++) x_dly[i] <= x_dly[i-1];
  end

  // ================= STAGE 0: x_sq = x * x ================================
  logic [31:0] s0_xsq_w;
  fp32_mul m_xx (.a(a), .b(a), .y(s0_xsq_w));
  logic [31:0] r0_xsq;
  always_ff @(posedge clk) r0_xsq <= s0_xsq_w;

  // ================= STAGE 1: x_cb = x_sq * x =============================
  logic [31:0] s1_xcb_w;
  fp32_mul m_x3 (.a(r0_xsq), .b(x_dly[0]), .y(s1_xcb_w));
  logic [31:0] r1_xcb;
  always_ff @(posedge clk) r1_xcb <= s1_xcb_w;

  // ================= STAGE 2: c_x_cb = 0.044715 * x_cb ====================
  logic [31:0] s2_cxcb_w;
  fp32_mul m_c (.a(C_044715), .b(r1_xcb), .y(s2_cxcb_w));
  logic [31:0] r2_cxcb;
  always_ff @(posedge clk) r2_cxcb <= s2_cxcb_w;

  // ================= STAGE 3: inner = x + c_x_cb ==========================
  logic [31:0] s3_inner_w;
  fp32_add a_in (.a(x_dly[2]), .b(r2_cxcb), .y(s3_inner_w));
  logic [31:0] r3_inner;
  always_ff @(posedge clk) r3_inner <= s3_inner_w;

  // ================= STAGE 4: z = K * inner ===============================
  logic [31:0] s4_z_w;
  fp32_mul m_z (.a(C_K_SQRT2PI), .b(r3_inner), .y(s4_z_w));
  logic [31:0] r4_z;
  always_ff @(posedge clk) r4_z <= s4_z_w;

  // ================= STAGE 5: z2 = z * 2 ==================================
  logic [31:0] s5_z2_w;
  fp32_mul m_z2 (.a(r4_z), .b(C_TWO), .y(s5_z2_w));
  logic [31:0] r5_z2;
  always_ff @(posedge clk) r5_z2 <= s5_z2_w;

  // ============ STAGES 6..23: exp_2z = exp(z2) [fp32_exp_p18] =============
  // valid_in is tied high and valid_out left unconnected: this module's own
  // vld_q is the single authority on validity (mirrors the sfu_synth_datapath
  // convention for shared pipelined primitives). The sub-pipe's datapath
  // free-runs like every other register here.
  logic [31:0] s24_e2z_w;
  logic        exp_vo_unused;
  fp32_exp_p18 e_2z (
    .clk(clk), .rst_n(rst_n), .valid_in(1'b1),
    .a(r5_z2), .valid_out(exp_vo_unused), .y(s24_e2z_w));

  // ================= STAGE 24: capture exp_p18's combinational s18 glue ====
  logic [31:0] r24_e2z;
  always_ff @(posedge clk) r24_e2z <= s24_e2z_w;

  // ================= STAGE 25: denom = exp_2z + 1 =========================
  logic [31:0] s25_denom_w;
  fp32_add a_dn (.a(r24_e2z), .b(C_ONE), .y(s25_denom_w));
  logic [31:0] r25_denom, r25_e2z;
  always_ff @(posedge clk) begin
    r25_denom <= s25_denom_w;
    r25_e2z   <= r24_e2z;      // dividend rides alongside its own +1
  end

  // ======== STAGES 26..31: ratio = exp_2z / denom [fp32_div_p6] ===========
  // Stable: ratio is in (0, 1) with no near-cancellation (see fp32_gelu_new).
  logic [31:0] s32_ratio_w;
  logic        div_vo_unused;
  fp32_div_p6 d_r (
    .clk(clk), .rst_n(rst_n), .valid_in(1'b1),
    .a(r25_e2z), .b(r25_denom), .valid_out(div_vo_unused), .y(s32_ratio_w));

  // ================= STAGE 32: y = x * ratio ==============================
  // fp32_div_p6 registers its own output, so s32_ratio_w is already a register
  // output; this stage is a single fp32_mul between two registers.
  logic [31:0] s32_y_w;
  fp32_mul m_out (.a(x_dly[31]), .b(s32_ratio_w), .y(s32_y_w));
  always_ff @(posedge clk) y <= s32_y_w;

endmodule

`endif // FP32_GELU_P33_SV
