// Synthesizable fp32 gelu_new(x) — gen-2 GELU (tanh-poly variant), the
// frozen-bundle target. Measured-band approximation (freeze §7).
//
// Definition (matches DPI golden sfu_fp32_gelu_new in testbench.h):
//   gelu_new(x) = x * 0.5 * (1 + tanh(K * (x + 0.044715 * x^3)))
//                 where K = sqrt(2/pi) ≈ 0.7978845608.
//
// tanh implementation via exp:
//   tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
//   Equivalent and cheaper: tanh(z) = 1 - 2 / (exp(2z) + 1).
// We use the second form: 1 mul (2z), 1 exp(2z), 1 add(+1), 1 div(2/...),
// 1 sub(1 - ...).
//
// Combinational chain instantiates the proven fp32_add, fp32_mul, fp32_div,
// fp32_exp cores. ULP error accumulates across the chain; the resulting
// band is measured against the libm golden.

`ifndef FP32_GELU_NEW_SV
`define FP32_GELU_NEW_SV

`include "fp32_add.sv"
`include "fp32_mul.sv"
`include "fp32_div.sv"
`include "fp32_exp.sv"

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

  // 2 / (exp(2z) + 1)
  logic [31:0] two_over_denom;
  fp32_div d_q (.a(C_TWO), .b(denom), .y(two_over_denom));

  // 1 - 2/(exp(2z)+1) = tanh(z)
  logic [31:0] neg_t;
  logic [31:0] one_plus_tanh;
  fp32_add a_1mt (.a(C_ONE), .b({~two_over_denom[31], two_over_denom[30:0]}), .y(neg_t));
  // neg_t = 1 - 2/(denom) = tanh(z); 1 + tanh = 1 + tanh:
  fp32_add a_1pt (.a(C_ONE), .b(neg_t), .y(one_plus_tanh));

  // 0.5 * (1 + tanh)
  logic [31:0] half_term;
  fp32_mul m_half (.a(C_HALF), .b(one_plus_tanh), .y(half_term));

  // x * half_term
  fp32_mul m_out (.a(a), .b(half_term), .y(y));

endmodule

`endif // FP32_GELU_NEW_SV
