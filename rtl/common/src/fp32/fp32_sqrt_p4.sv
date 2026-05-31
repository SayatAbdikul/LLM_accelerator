// 4-stage pipelined IEEE-754 binary32 square root — bit-exact to the
// combinational `fp32_sqrt` (and thus to the DPI golden `sfu_fp32_sqrt`).
//
// WHY THIS EXISTS:
//   After the div_p4 + sqrt_p3 campaign + LN_VAR-accumulate pipelining, the
//   sole post-PNR fmax floor became fp32_sqrt_p3 STAGE-2 (the LN denom path
//   u_ln_sqrt.rB_r cluster, 38.67 ns at PNR — the 5 mid digit-recurrence iters
//   9..5, all in one stage). 3-stage rebalancing can't beat ~37 ns (stages are
//   already delay-balanced). This module adds a FOURTH pipeline cut so the wide
//   recurrence tail is split across three stages instead of two, dropping the
//   per-stage delay toward the next floor cluster.
//
// EQUIVALENCE:
//   Identical arithmetic to fp32_sqrt / fp32_sqrt_p2 / fp32_sqrt_p3 (unpack,
//   subnormal normalize, even-exponent radicand build, restoring radix-2
//   recurrence, RNE rounding, non-finite contract, pack). Only the placement of
//   pipeline registers differs. test_fp32_sqrt_p4.cpp is the byte-exact gate.
//
// SPLIT POINTS (balance by DELAY, not iter count — the recurrence cost is NOT
//   uniform: early iters have a narrow q [leading-zero-folded, cheap], later
//   iters a wide q -> full 52-bit compare/subtract, expensive; cost rises as the
//   iter index falls because q accumulates one bit per iter):
//     Stage 1: iters 24..S1_LO. MUST run through iter 12 — the lowest set
//       M_pad block (bit 25, even-exp case) is consumed at iter 12, so with
//       S1_LO<=12 every real (nonzero) block is retired in stage 1 and stages
//       2/3/4 see only zero blocks (no M_pad register needed past stage 1).
//     Stage 2: iters S1_LO-1 .. S2_LO   (zero blocks)
//     Stage 3: iters S2_LO-1 .. S3_LO   (zero blocks)
//     Stage 4: iters S3_LO-1 .. 0       (zero blocks) + RNE round + pack
//   The S1_LO / S2_LO / S3_LO constants below are tuned from per-stage STA;
//   adjust them (re-measure) if the binding stage shifts.
//
// INTERFACE: fixed LATENCY=4. Present (a,valid_in) on cycle N; (y,valid_out)
//   are valid on cycle N+4. Fully pipelined: accepts a new operand every cycle.

`ifndef FP32_SQRT_P4_SV
`define FP32_SQRT_P4_SV

module fp32_sqrt_p4 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  output logic        valid_out,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // Stage boundaries (iters run high->low; stage N covers HI..LO inclusive).
  // S1_LO must be <= 12 so all nonzero M_pad blocks retire in stage 1.
  localparam int S1_LO = 12;   // stage 1: iters 24 .. S1_LO
  localparam int S2_LO = 8;    // stage 2: iters S1_LO-1 .. S2_LO
  localparam int S3_LO = 4;    // stage 3: iters S2_LO-1 .. S3_LO

  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // =====================================================================
  // STAGE 1 (combinational): unpack, normalize, build radicand, iters 24..S1_LO.
  // =====================================================================
  logic        s;
  logic [7:0]  e;
  logic [22:0] m;
  assign s = a[31];
  assign e = a[30:23];
  assign m = a[22:0];

  logic a_zero, a_inf, a_nan, a_sub, a_neg_finite;
  assign a_zero       = (e == 8'd0)   && (m == 23'd0);
  assign a_sub        = (e == 8'd0)   && (m != 23'd0);
  assign a_inf        = (e == 8'd255) && (m == 23'd0);
  assign a_nan        = (e == 8'd255) && (m != 23'd0);
  assign a_neg_finite = s && !a_zero && !a_inf && !a_nan;

  int unsigned        mp;
  logic [23:0]        sig_a;
  logic signed [9:0]  exp_a;
  always_comb begin
    if (a_sub) begin
      mp    = msb23(m);
      sig_a = {1'b0, m} << (5'd23 - mp[4:0]);
      exp_a = -10'sd126 - {{5{1'b0}}, (5'd23 - mp[4:0])};
    end else begin
      mp    = 0;
      sig_a = {1'b1, m};
      exp_a = $signed({2'b0, e}) - 10'sd127;
    end
  end

  // Build even-exponent 50-bit radicand M_pad and result exponent R_exp.
  logic              exp_a_odd;
  assign exp_a_odd = exp_a[0];

  logic [49:0]       M_pad;
  logic signed [9:0] R_exp;
  always_comb begin
    if (exp_a_odd) begin
      M_pad = {sig_a, 26'd0};            // top bit at 49
      R_exp = (exp_a - 10'sd1) >>> 1;
    end else begin
      M_pad = {1'b0, sig_a, 25'd0};      // top bit at 48
      R_exp = exp_a >>> 1;
    end
  end

  // Restoring radix-2 sqrt, forward-running accumulation. STAGE 1: iters 24..S1_LO.
  logic [51:0] s1_r;
  logic [24:0] s1_q;
  integer i1;
  always_comb begin
    s1_r = 52'd0;
    s1_q = 25'd0;
    for (i1 = 24; i1 >= S1_LO; i1 = i1 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      logic [1:0]  blk;
      blk = M_pad[2*i1 +: 2];
      rn  = (s1_r << 2) | {50'd0, blk};
      tr  = ({27'd0, s1_q} << 2) | 52'd1;
      if (rn >= tr) begin
        s1_r = rn - tr;
        s1_q = (s1_q << 1) | 25'd1;
      end else begin
        s1_r = rn;
        s1_q = (s1_q << 1);
      end
    end
  end

  // ---- pipeline register A (stage 1 -> stage 2) ----
  // No M_pad bits needed: stages 2/3/4 consume M_pad[2*S1_LO-1:0], all zero.
  logic              rA_valid;
  logic [51:0]       rA_r;
  logic [24:0]       rA_q;
  logic signed [9:0] rA_Rexp;
  logic              rA_s, rA_anan, rA_negf, rA_azero, rA_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rA_valid <= 1'b0;
    end else begin
      rA_valid <= valid_in;
      rA_r     <= s1_r;
      rA_q     <= s1_q;
      rA_Rexp  <= R_exp;
      rA_s     <= s;
      rA_anan  <= a_nan;
      rA_negf  <= a_neg_finite;
      rA_azero <= a_zero;
      rA_ainf  <= a_inf;
    end
  end

  // =====================================================================
  // STAGE 2 (combinational): iters S1_LO-1 .. S2_LO (zero blocks).
  // =====================================================================
  logic [51:0] s2_r;
  logic [24:0] s2_q;
  integer i2;
  always_comb begin
    s2_r = rA_r;
    s2_q = rA_q;
    for (i2 = S1_LO-1; i2 >= S2_LO; i2 = i2 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s2_r << 2);               // block = M_pad[2*i2+:2] = 0
      tr  = ({27'd0, s2_q} << 2) | 52'd1;
      if (rn >= tr) begin
        s2_r = rn - tr;
        s2_q = (s2_q << 1) | 25'd1;
      end else begin
        s2_r = rn;
        s2_q = (s2_q << 1);
      end
    end
  end

  // ---- pipeline register B (stage 2 -> stage 3) ----
  logic              rB_valid;
  logic [51:0]       rB_r;
  logic [24:0]       rB_q;
  logic signed [9:0] rB_Rexp;
  logic              rB_s, rB_anan, rB_negf, rB_azero, rB_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rB_valid <= 1'b0;
    end else begin
      rB_valid <= rA_valid;
      rB_r     <= s2_r;
      rB_q     <= s2_q;
      rB_Rexp  <= rA_Rexp;
      rB_s     <= rA_s;
      rB_anan  <= rA_anan;
      rB_negf  <= rA_negf;
      rB_azero <= rA_azero;
      rB_ainf  <= rA_ainf;
    end
  end

  // =====================================================================
  // STAGE 3 (combinational): iters S2_LO-1 .. S3_LO (zero blocks).
  // =====================================================================
  logic [51:0] s3_r;
  logic [24:0] s3_q;
  integer i3;
  always_comb begin
    s3_r = rB_r;
    s3_q = rB_q;
    for (i3 = S2_LO-1; i3 >= S3_LO; i3 = i3 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s3_r << 2);               // block = M_pad[2*i3+:2] = 0
      tr  = ({27'd0, s3_q} << 2) | 52'd1;
      if (rn >= tr) begin
        s3_r = rn - tr;
        s3_q = (s3_q << 1) | 25'd1;
      end else begin
        s3_r = rn;
        s3_q = (s3_q << 1);
      end
    end
  end

  // ---- pipeline register C (stage 3 -> stage 4) ----
  logic              rC_valid;
  logic [51:0]       rC_r;
  logic [24:0]       rC_q;
  logic signed [9:0] rC_Rexp;
  logic              rC_s, rC_anan, rC_negf, rC_azero, rC_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rC_valid <= 1'b0;
    end else begin
      rC_valid <= rB_valid;
      rC_r     <= s3_r;
      rC_q     <= s3_q;
      rC_Rexp  <= rB_Rexp;
      rC_s     <= rB_s;
      rC_anan  <= rB_anan;
      rC_negf  <= rB_negf;
      rC_azero <= rB_azero;
      rC_ainf  <= rB_ainf;
    end
  end

  // =====================================================================
  // STAGE 4 (combinational): iters S3_LO-1 .. 0 (zero blocks), RNE, pack.
  // =====================================================================
  logic [51:0] s4_r;
  logic [24:0] q_final;
  integer i4;
  always_comb begin
    s4_r    = rC_r;
    q_final = rC_q;
    for (i4 = S3_LO-1; i4 >= 0; i4 = i4 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s4_r << 2);               // block = M_pad[2*i4+:2] = 0
      tr  = ({27'd0, q_final} << 2) | 52'd1;
      if (rn >= tr) begin
        s4_r    = rn - tr;
        q_final = (q_final << 1) | 25'd1;
      end else begin
        s4_r    = rn;
        q_final = (q_final << 1);
      end
    end
  end

  logic        sticky;
  assign sticky = (s4_r != 52'd0);

  // RNE on (round=q_final[0], sticky) — verbatim from fp32_sqrt.
  logic        rb_sq, ru_sq;
  logic [23:0] mant_pre;
  assign rb_sq    = q_final[0];
  assign mant_pre = q_final[24:1];
  assign ru_sq    = rb_sq & (sticky | mant_pre[0]);

  logic [24:0] mant24_rnd;
  assign mant24_rnd = {1'b0, mant_pre} + {24'd0, ru_sq};

  logic               carry;
  logic signed [9:0]  exp_y_unb;
  logic [7:0]         exp_y_biased;
  always_comb begin
    carry        = mant24_rnd[24];
    exp_y_unb    = rC_Rexp + (carry ? 10'sd1 : 10'sd0);
    exp_y_biased = 8'(exp_y_unb + 10'sd127);
  end

  logic [31:0] y_comb;
  always_comb begin
    if (rC_anan) begin
      y_comb = QNAN;
    end else if (rC_negf) begin
      y_comb = QNAN;                                   // sqrt(neg) -> NaN
    end else if (rC_azero) begin
      y_comb = {rC_s, 8'd0, 23'd0};                    // ±0 -> ±0
    end else if (rC_ainf) begin
      y_comb = rC_s ? QNAN : {1'b0, 8'd255, 23'd0};    // -inf -> NaN, +inf -> +inf
    end else if (exp_y_unb >= 10'sd128) begin
      y_comb = {1'b0, 8'd255, 23'd0};                  // overflow (essentially impossible)
    end else if (exp_y_unb < -10'sd126) begin
      y_comb = {1'b0, 8'd0, 23'd0};                    // underflow (also basically impossible)
    end else if (carry) begin
      y_comb = {1'b0, exp_y_biased, 23'd0};
    end else begin
      y_comb = {1'b0, exp_y_biased, mant24_rnd[22:0]};
    end
  end

  // ---- output register (stage-4 path ends at a register) ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
    end else begin
      valid_out <= rC_valid;
      y         <= y_comb;
    end
  end

endmodule

`endif // FP32_SQRT_P4_SV
