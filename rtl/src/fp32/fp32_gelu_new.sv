// Synthesizable fp32 gelu_new(x) — gen-2 GELU (tanh-poly variant), the
// frozen-bundle target. Measured-band approximation (freeze §7).
//
// Definition (matches DPI golden sfu_fp32_gelu_new in testbench.h):
//   gelu_new(x) = x * 0.5 * (1 + tanh(K * (x + 0.044715 * x^3)))
//                 where K = sqrt(2/pi) ≈ 0.7978845608.
//
// Stable algebraic form (avoids catastrophic cancellation in (1 + tanh) for
// large negative z, where tanh → -1):
//   1 + tanh(z) = 1 + (1 - 2/(exp(2z)+1)) = 2 - 2/(exp(2z)+1)
//               = 2 * (exp(2z) + 1 - 1)/(exp(2z) + 1)
//               = 2 * exp(2z) / (exp(2z) + 1)
//   gelu(x)    = x * 0.5 * (1 + tanh(z)) = x * exp(2z) / (exp(2z) + 1).
// Saves 1 mul + 2 adds vs the naive (1 + tanh) form AND eliminates the
// near-zero subtraction that lost ~9 mantissa bits for x ≲ -3.
//
// Combinational chain instantiates the proven fp32_add, fp32_mul, fp32_div,
// fp32_exp cores. With Cody-Waite + degree-6 fp32_exp (≤3 ULP) and this
// stable form, the GELU output band sits within freeze §7's ≤3 fp16 ULP.

`ifndef FP32_GELU_NEW_SV
`define FP32_GELU_NEW_SV

// Dependencies (fp32_add, fp32_mul, fp32_div, fp32_exp) are read in order by
// the parent build (FP32_PRIMS in rtl/verilator/Makefile, or the standalone
// gate rule with -I). Local `\`include` directives are intentionally absent
// so the CONTROL_SV path (which has no -I to fp32/) elaborates cleanly.

module fp32_gelu_new (
  input  logic [31:0] a,    // x
  output logic [31:0] y     // gelu_new(x)
);
  localparam logic [31:0] C_HALF      = 32'h3F00_0000;   // 0.5
  localparam logic [31:0] C_ONE       = 32'h3F80_0000;   // 1.0
  localparam logic [31:0] C_TWO       = 32'h4000_0000;   // 2.0
  localparam logic [31:0] C_K_SQRT2PI = 32'h3F4C_4229;   // sqrt(2/pi) ≈ 0.7978845608
  localparam logic [31:0] C_044715    = 32'h3D37_2713;   // 0.044715

  // x^2 = x * x
  logic [31:0] x_sq;
  fp32_mul m_xx (.a(a), .b(a), .y(x_sq));

  // x^3 = x^2 * x
  logic [31:0] x_cb;
  fp32_mul m_x3 (.a(x_sq), .b(a), .y(x_cb));

  // 0.044715 * x^3
  logic [31:0] c_x_cb;
  fp32_mul m_c (.a(C_044715), .b(x_cb), .y(c_x_cb));

  // x + 0.044715 * x^3
  logic [31:0] inner_add;
  fp32_add a_in (.a(a), .b(c_x_cb), .y(inner_add));

  // K * (x + ...) = z
  logic [31:0] z;
  fp32_mul m_z (.a(C_K_SQRT2PI), .b(inner_add), .y(z));

  // 2z
  logic [31:0] z2;
  fp32_mul m_z2 (.a(z), .b(C_TWO), .y(z2));

  // exp(2z)
  logic [31:0] exp_2z;
  fp32_exp e_2z (.a(z2), .y(exp_2z));

  // exp(2z) + 1
  logic [31:0] denom;
  fp32_add a_dn (.a(exp_2z), .b(C_ONE), .y(denom));

  // ratio = exp(2z) / (exp(2z) + 1)  — i.e. 0.5 * (1 + tanh(z))
  // Stable: ratio is in (0, 1) with no near-cancellation; for large negative
  // z the result is exp(2z)/(1 + tiny) ≈ exp(2z); for large positive z it's
  // big/(big + 1) ≈ 1. No 1 + tanh subtraction; ~9 bits of precision saved.
  logic [31:0] ratio;
  fp32_div d_r (.a(exp_2z), .b(denom), .y(ratio));

  // x * ratio = x * 0.5 * (1 + tanh(z)) = gelu_new(x)
  fp32_mul m_out (.a(a), .b(ratio), .y(y));

endmodule

`endif // FP32_GELU_NEW_SV
