// Bit-exact equivalence harness for fp32_gelu_p33 vs the combinational
// fp32_gelu_new. Instantiates BOTH from the same input `a`; the C++
// (test_fp32_gelu_p33.cpp) drives a stream and asserts y_pipe (LATENCY=33
// later) == the y_comb that was presented 33 cycles earlier. Pure retiming
// (same ten primitives, same order, same operand ports; fp32_exp/fp32_div
// swapped for their bit-exact pipelined twins) ⇒ ZERO diff expected.
module fp32_gelu_p33_tb (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  output logic [31:0] y_comb,
  output logic        valid_out,
  output logic [31:0] y_pipe
);
  fp32_gelu_new u_comb (.a(a), .y(y_comb));
  fp32_gelu_p33 u_pipe (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                        .a(a), .valid_out(valid_out), .y(y_pipe));
endmodule
