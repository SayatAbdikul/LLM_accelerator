// 2-stage pipelined IEEE-754 binary32 divide — bit-exact to the combinational
// `fp32_div` (and thus to the DPI golden `sfu_fp32_div`).
//
// WHY THIS EXISTS:
//   The combinational `fp32_div` long-division (q_full = (sig_a<<28)/sig_b) is
//   the post-PNR fmax floor of the whole chip (144.56 ns binding path in the
//   SFU LN/softmax normalize states). The earlier `fp32_div_pipe` shell merely
//   registered the OUTPUT of the same combinational core (y_pipe <= y_comb) —
//   latency without cutting the path, so ZERO fmax. This module instead cuts
//   the path INSIDE the division: the 29-iteration restoring divide is split
//   15 / 14 across a pipeline register, roughly halving the critical path.
//
// EQUIVALENCE:
//   Everything except the division is identical to fp32_div.sv (unpack,
//   subnormal normalize, RNE rounding, subnormal/underflow path, non-finite
//   contract, pack). The behavioral `dividend / sig_b` is replaced by an
//   explicit MSB-aligned restoring division proven bit-identical to floor
//   division over the operand domain (both sigs in [2^23, 2^24)); see the
//   Python gate in the commit and test_fp32_div.cpp for the byte-exact gate.
//
// INTERFACE: fixed LATENCY=2. Present (a,b,valid_in) on cycle N; (y,valid_out)
//   are valid on cycle N+2. Fully pipelined: accepts a new pair every cycle.

`ifndef FP32_DIV_P2_SV
`define FP32_DIV_P2_SV

module fp32_div_p2 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        valid_out,
  output logic [31:0] y
);
  localparam logic [31:0] QNAN = 32'h7FC0_0000;

  function automatic int unsigned msb23(input logic [22:0] v);
    int i;
    begin
      for (i = 22; i >= 0; i = i - 1)
        if (v[i]) return i;
      return 0;
    end
  endfunction

  // =====================================================================
  // STAGE 1 (combinational): unpack, normalize sigs, division iters 28..14.
  // =====================================================================
  logic        sa, sb;
  logic [7:0]  ea, eb;
  logic [22:0] ma, mb;
  assign sa = a[31];  assign ea = a[30:23];  assign ma = a[22:0];
  assign sb = b[31];  assign eb = b[30:23];  assign mb = b[22:0];

  logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan, a_sub, b_sub;
  assign a_zero = (ea == 8'd0)   && (ma == 23'd0);
  assign b_zero = (eb == 8'd0)   && (mb == 23'd0);
  assign a_sub  = (ea == 8'd0)   && (ma != 23'd0);
  assign b_sub  = (eb == 8'd0)   && (mb != 23'd0);
  assign a_inf  = (ea == 8'd255) && (ma == 23'd0);
  assign b_inf  = (eb == 8'd255) && (mb == 23'd0);
  assign a_nan  = (ea == 8'd255) && (ma != 23'd0);
  assign b_nan  = (eb == 8'd255) && (mb != 23'd0);

  int unsigned mp_a, mp_b;
  logic [23:0] sig_a, sig_b;
  logic signed [9:0] exp_a, exp_b;
  always_comb begin
    mp_a = a_sub ? msb23(ma) : 0;
    mp_b = b_sub ? msb23(mb) : 0;
    if (a_sub) begin
      sig_a = {1'b0, ma} << (5'd23 - mp_a[4:0]);
      exp_a = -10'sd126 - {{5{1'b0}}, (5'd23 - mp_a[4:0])};
    end else begin
      sig_a = {1'b1, ma};
      exp_a = $signed({2'b0, ea}) - 10'sd127;
    end
    if (b_sub) begin
      sig_b = {1'b0, mb} << (5'd23 - mp_b[4:0]);
      exp_b = -10'sd126 - {{5{1'b0}}, (5'd23 - mp_b[4:0])};
    end else begin
      sig_b = {1'b1, mb};
      exp_b = $signed({2'b0, eb}) - 10'sd127;
    end
  end

  // Explicit MSB-aligned restoring division (replaces behavioral `/`).
  // P preloaded = dividend[51:28] = sig_a; iters k=28..0 produce q_full[28:0],
  // shifting in dividend low bits (all zero, since dividend = sig_a<<28).
  // STAGE 1 runs iters 28..14 (15 iters); the k=14 post-shift lands P in the
  // k=13 position so STAGE 2 resumes cleanly.
  logic [24:0] s1_P;
  logic [28:0] s1_q;
  integer k1;
  always_comb begin
    s1_P = {1'b0, sig_a};        // 25-bit, < 2^24
    s1_q = 29'd0;
    for (k1 = 28; k1 >= 14; k1 = k1 - 1) begin
      if (s1_P >= {1'b0, sig_b}) begin
        s1_q[k1] = 1'b1;
        s1_P = s1_P - {1'b0, sig_b};
      end
      s1_P = s1_P << 1;          // k1 >= 14 > 0, always shift
    end
  end

  // ---- pipeline register ----
  logic        r_valid;
  logic [24:0] r_P;
  logic [28:0] r_q;
  logic [23:0] r_sigb;
  logic        r_sign, r_agb;
  logic signed [9:0] r_expa, r_expb;
  logic        r_anan, r_bnan, r_ainf, r_binf, r_azero, r_bzero;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_valid <= 1'b0;
    end else begin
      r_valid <= valid_in;
      r_P     <= s1_P;
      r_q     <= s1_q;
      r_sigb  <= sig_b;
      r_sign  <= sa ^ sb;
      r_agb   <= (sig_a >= sig_b);
      r_expa  <= exp_a;
      r_expb  <= exp_b;
      r_anan  <= a_nan;  r_bnan  <= b_nan;
      r_ainf  <= a_inf;  r_binf  <= b_inf;
      r_azero <= a_zero; r_bzero <= b_zero;
    end
  end

  // =====================================================================
  // STAGE 2 (combinational): division iters 13..0, RNE round, pack.
  // =====================================================================
  logic [24:0] s2_P;
  logic [28:0] q_full;
  logic [24:0] remainder;        // only |remainder| (sticky) is used
  integer k2;
  always_comb begin
    s2_P   = r_P;
    q_full = r_q;
    for (k2 = 13; k2 >= 0; k2 = k2 - 1) begin
      if (s2_P >= {1'b0, r_sigb}) begin
        q_full[k2] = 1'b1;
        s2_P = s2_P - {1'b0, r_sigb};
      end
      if (k2 > 0) s2_P = s2_P << 1;
    end
    remainder = s2_P;            // final partial remainder = dividend mod sig_b
  end

  // --- extract mantissa + RNE on the regime ULP (verbatim from fp32_div) ---
  logic [23:0] mant24_pre;
  logic        rb, st, ru;
  logic [24:0] mant24_rnd;
  logic signed [10:0] exp_y_unb;
  always_comb begin
    if (r_agb) begin
      mant24_pre = q_full[28:5];
      rb         = q_full[4];
      st         = (|q_full[3:0]) | (|remainder);
      exp_y_unb  = r_expa - r_expb;
    end else begin
      mant24_pre = q_full[27:4];
      rb         = q_full[3];
      st         = (|q_full[2:0]) | (|remainder);
      exp_y_unb  = r_expa - r_expb - 11'sd1;
    end
    ru         = rb & (st | mant24_pre[0]);
    mant24_rnd = {1'b0, mant24_pre} + {24'd0, ru};
  end

  logic               carry;
  logic signed [10:0] exp_y_final_unb;
  logic [7:0]         exp_y_biased;
  always_comb begin
    carry           = mant24_rnd[24];
    exp_y_final_unb = exp_y_unb + (carry ? 11'sd1 : 11'sd0);
    exp_y_biased    = 8'(exp_y_final_unb + 11'sd127);
  end

  // Subnormal/underflow path (verbatim from fp32_div).
  logic signed [11:0] k_lsb_s;
  logic [4:0]         k_lsb;
  logic [28:0]        q_shifted_sub;
  logic [22:0]        mant_sub_pre;
  logic               sub_rb, sub_st, sub_ru;
  logic [23:0]        mant_sub_rnd;
  always_comb begin
    k_lsb_s       = -{{2{r_expa[9]}}, r_expa} + {{2{r_expb[9]}}, r_expb} - 12'sd121;
    k_lsb         = (k_lsb_s > 12'sd29) ? 5'd29 : k_lsb_s[4:0];
    q_shifted_sub = q_full >> k_lsb;
    mant_sub_pre  = q_shifted_sub[22:0];

    if (k_lsb_s >= 12'sd1 && k_lsb_s <= 12'sd29)
      sub_rb = q_full[k_lsb - 5'd1];
    else
      sub_rb = 1'b0;

    if (k_lsb_s > 12'sd29)
      sub_st = (|q_full) | (|remainder);
    else if (k_lsb_s >= 12'sd2)
      sub_st = (|(q_full & ((29'd1 << (k_lsb - 5'd1)) - 29'd1))) | (|remainder);
    else if (k_lsb_s == 12'sd1)
      sub_st = (|remainder);
    else
      sub_st = 1'b0;

    sub_ru       = sub_rb & (sub_st | mant_sub_pre[0]);
    mant_sub_rnd = {1'b0, mant_sub_pre} + {23'd0, sub_ru};
  end

  logic [31:0] y_comb;
  always_comb begin
    if (r_anan || r_bnan) begin
      y_comb = QNAN;
    end else if (r_ainf && r_binf) begin
      y_comb = QNAN;
    end else if (r_azero && r_bzero) begin
      y_comb = QNAN;
    end else if (r_ainf) begin
      y_comb = {r_sign, 8'd255, 23'd0};
    end else if (r_binf) begin
      y_comb = {r_sign, 8'd0,   23'd0};
    end else if (r_bzero) begin
      y_comb = {r_sign, 8'd255, 23'd0};
    end else if (r_azero) begin
      y_comb = {r_sign, 8'd0,   23'd0};
    end else if (exp_y_final_unb >= 11'sd128) begin
      y_comb = {r_sign, 8'd255, 23'd0};
    end else if (exp_y_final_unb < -11'sd126) begin
      if (mant_sub_rnd[23])
        y_comb = {r_sign, 8'd1, 23'd0};
      else
        y_comb = {r_sign, 8'd0, mant_sub_rnd[22:0]};
    end else if (carry) begin
      y_comb = {r_sign, exp_y_biased, 23'd0};
    end else begin
      y_comb = {r_sign, exp_y_biased, mant24_rnd[22:0]};
    end
  end

  // ---- output register (stage-2 path ends at a register) ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
    end else begin
      valid_out <= r_valid;
      y         <= y_comb;
    end
  end

endmodule

`endif // FP32_DIV_P2_SV
