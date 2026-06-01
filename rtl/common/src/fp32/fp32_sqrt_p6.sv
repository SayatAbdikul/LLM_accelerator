// 6-stage pipelined IEEE-754 binary32 square root — bit-exact to the
// combinational `fp32_sqrt` (and the DPI golden `sfu_fp32_sqrt`).
//
// WHY THIS EXISTS:
//   After fp32_sqrt_p4 + fp32_div_p5, the post-PNR fmax floor is fp32_sqrt_p4
//   STAGE-3 (u_ln_sqrt.rC_r, iters 7..4, 31.54 ns) with STAGE-1 (rA_r = unpack +
//   iters 24..12) co-binding at 31.07 ns. A tail-only split (sqrt_p5) can't beat
//   31.07 because stage 1 is pinned: S1_LO<=12 forces it to carry the unpack AND
//   iters 24..12 (the nonzero-M_pad region). This module:
//     (1) ISOLATES the unpack/normalize into its own stage (register sig_a/exp_a
//         + non-finite flags before any iteration), and
//     (2) splits the expensive iter tail (each late iter ~7.5 ns -> <=3 per stage)
//   so every stage drops below the div_p5 tier (~27.5 ns PNR), moving the floor
//   onto the divider.
//
// EQUIVALENCE:
//   Identical arithmetic to fp32_sqrt / _p2 / _p3 / _p4 (unpack, subnormal
//   normalize, even-exponent radicand build, restoring radix-2 recurrence, RNE
//   rounding, non-finite contract, pack). Only pipeline-register PLACEMENT
//   differs: 5 registers split it six ways. The M_pad radicand is rebuilt in
//   stage 2 from the registered sig_a/exp_a (cheap shifts) so the 50-bit M_pad
//   needs no pipeline register. test_fp32_sqrt_p6.cpp is the byte-exact gate.
//
// STAGE LAYOUT:
//   Stage 1: unpack + subnormal normalize -> sig_a, exp_a, flags        (reg A)
//   Stage 2: build M_pad/R_exp; iters 24..S2_LO (the nonzero-block region; must
//            run through iter 12 so every nonzero M_pad block retires here)(reg B)
//   Stage 3: iters S2_LO-1 .. S3_LO   (zero blocks)                      (reg C)
//   Stage 4: iters S3_LO-1 .. S4_LO   (zero blocks)                      (reg D)
//   Stage 5: iters S4_LO-1 .. S5_LO   (zero blocks)                      (reg E)
//   Stage 6: iters S5_LO-1 .. 0       (zero blocks) + RNE round + pack
//   S2_LO must be <= 12 (iter 12 consumes the lowest set M_pad block, bit 25 in
//   the even-exp case). Constants tuned by per-stage STA.
//
// INTERFACE: fixed LATENCY=6. Present (a,valid_in) on cycle N; (y,valid_out) are
//   valid on cycle N+6. Fully pipelined: accepts a new operand every cycle.

`ifndef FP32_SQRT_P6_SV
`define FP32_SQRT_P6_SV

module fp32_sqrt_p6 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  output logic        valid_out,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  // Stage boundaries (iters run high->low; stage covers HI..LO inclusive).
  // S2_LO must be <= 12 so all nonzero M_pad blocks retire in stage 2.
  localparam int S2_LO = 12;   // stage 2: iters 24 .. S2_LO
  localparam int S3_LO = 10;    // stage 3: iters S2_LO-1 .. S3_LO
  localparam int S4_LO = 7;    // stage 4: iters S3_LO-1 .. S4_LO
  localparam int S5_LO = 4;    // stage 5: iters S4_LO-1 .. S5_LO

  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // =====================================================================
  // STAGE 1 (combinational): unpack, subnormal normalize -> sig_a, exp_a, flags.
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

  // ---- pipeline register A (stage 1 -> stage 2) ----
  logic              rA_valid;
  logic [23:0]       rA_sig;
  logic signed [9:0] rA_expa;
  logic              rA_s, rA_anan, rA_negf, rA_azero, rA_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rA_valid <= 1'b0;
    end else begin
      rA_valid <= valid_in;
      rA_sig   <= sig_a;
      rA_expa  <= exp_a;
      rA_s     <= s;
      rA_anan  <= a_nan;
      rA_negf  <= a_neg_finite;
      rA_azero <= a_zero;
      rA_ainf  <= a_inf;
    end
  end

  // =====================================================================
  // STAGE 2 (combinational): build even-exp radicand M_pad / R_exp from the
  // registered sig_a/exp_a, then iters 24..S2_LO (the nonzero-block region).
  // =====================================================================
  logic              exp_a_odd;
  assign exp_a_odd = rA_expa[0];

  logic [49:0]       M_pad;
  logic signed [9:0] R_exp;
  always_comb begin
    if (exp_a_odd) begin
      M_pad = {rA_sig, 26'd0};            // top bit at 49
      R_exp = (rA_expa - 10'sd1) >>> 1;
    end else begin
      M_pad = {1'b0, rA_sig, 25'd0};      // top bit at 48
      R_exp = rA_expa >>> 1;
    end
  end

  logic [51:0] s2_r;
  logic [24:0] s2_q;
  integer i2;
  always_comb begin
    s2_r = 52'd0;
    s2_q = 25'd0;
    for (i2 = 24; i2 >= S2_LO; i2 = i2 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      logic [1:0]  blk;
      blk = M_pad[2*i2 +: 2];
      rn  = (s2_r << 2) | {50'd0, blk};
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
  // No M_pad bits needed past here: stages 3..6 consume M_pad[2*S2_LO-1:0]=0.
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
      rB_Rexp  <= R_exp;
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
      rn  = (s3_r << 2);               // block = 0
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
  // STAGE 4 (combinational): iters S3_LO-1 .. S4_LO (zero blocks).
  // =====================================================================
  logic [51:0] s4_r;
  logic [24:0] s4_q;
  integer i4;
  always_comb begin
    s4_r = rC_r;
    s4_q = rC_q;
    for (i4 = S3_LO-1; i4 >= S4_LO; i4 = i4 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s4_r << 2);
      tr  = ({27'd0, s4_q} << 2) | 52'd1;
      if (rn >= tr) begin
        s4_r = rn - tr;
        s4_q = (s4_q << 1) | 25'd1;
      end else begin
        s4_r = rn;
        s4_q = (s4_q << 1);
      end
    end
  end

  // ---- pipeline register D (stage 4 -> stage 5) ----
  logic              rD_valid;
  logic [51:0]       rD_r;
  logic [24:0]       rD_q;
  logic signed [9:0] rD_Rexp;
  logic              rD_s, rD_anan, rD_negf, rD_azero, rD_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rD_valid <= 1'b0;
    end else begin
      rD_valid <= rC_valid;
      rD_r     <= s4_r;
      rD_q     <= s4_q;
      rD_Rexp  <= rC_Rexp;
      rD_s     <= rC_s;
      rD_anan  <= rC_anan;
      rD_negf  <= rC_negf;
      rD_azero <= rC_azero;
      rD_ainf  <= rC_ainf;
    end
  end

  // =====================================================================
  // STAGE 5 (combinational): iters S4_LO-1 .. S5_LO (zero blocks).
  // =====================================================================
  logic [51:0] s5_r;
  logic [24:0] s5_q;
  integer i5;
  always_comb begin
    s5_r = rD_r;
    s5_q = rD_q;
    for (i5 = S4_LO-1; i5 >= S5_LO; i5 = i5 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s5_r << 2);
      tr  = ({27'd0, s5_q} << 2) | 52'd1;
      if (rn >= tr) begin
        s5_r = rn - tr;
        s5_q = (s5_q << 1) | 25'd1;
      end else begin
        s5_r = rn;
        s5_q = (s5_q << 1);
      end
    end
  end

  // ---- pipeline register E (stage 5 -> stage 6) ----
  logic              rE_valid;
  logic [51:0]       rE_r;
  logic [24:0]       rE_q;
  logic signed [9:0] rE_Rexp;
  logic              rE_s, rE_anan, rE_negf, rE_azero, rE_ainf;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rE_valid <= 1'b0;
    end else begin
      rE_valid <= rD_valid;
      rE_r     <= s5_r;
      rE_q     <= s5_q;
      rE_Rexp  <= rD_Rexp;
      rE_s     <= rD_s;
      rE_anan  <= rD_anan;
      rE_negf  <= rD_negf;
      rE_azero <= rD_azero;
      rE_ainf  <= rD_ainf;
    end
  end

  // =====================================================================
  // STAGE 6 (combinational): iters S5_LO-1 .. 0 (zero blocks), RNE, pack.
  // =====================================================================
  logic [51:0] s6_r;
  logic [24:0] q_final;
  integer i6;
  always_comb begin
    s6_r    = rE_r;
    q_final = rE_q;
    for (i6 = S5_LO-1; i6 >= 0; i6 = i6 - 1) begin
      logic [51:0] rn;
      logic [51:0] tr;
      rn  = (s6_r << 2);
      tr  = ({27'd0, q_final} << 2) | 52'd1;
      if (rn >= tr) begin
        s6_r    = rn - tr;
        q_final = (q_final << 1) | 25'd1;
      end else begin
        s6_r    = rn;
        q_final = (q_final << 1);
      end
    end
  end

  logic        sticky;
  assign sticky = (s6_r != 52'd0);

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
    exp_y_unb    = rE_Rexp + (carry ? 10'sd1 : 10'sd0);
    exp_y_biased = 8'(exp_y_unb + 10'sd127);
  end

  logic [31:0] y_comb;
  always_comb begin
    if (rE_anan) begin
      y_comb = QNAN;
    end else if (rE_negf) begin
      y_comb = QNAN;                                   // sqrt(neg) -> NaN
    end else if (rE_azero) begin
      y_comb = {rE_s, 8'd0, 23'd0};                    // ±0 -> ±0
    end else if (rE_ainf) begin
      y_comb = rE_s ? QNAN : {1'b0, 8'd255, 23'd0};    // -inf -> NaN, +inf -> +inf
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

  // ---- output register (stage-6 path ends at a register) ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
    end else begin
      valid_out <= rE_valid;
      y         <= y_comb;
    end
  end

endmodule

`endif // FP32_SQRT_P6_SV
