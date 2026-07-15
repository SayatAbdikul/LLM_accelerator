// Bit-exact equivalence harness for fp32_exp_p18 vs the combinational fp32_exp.
// Instantiates BOTH from the same input `a`; the C++ (test_fp32_exp_p18.cpp)
// drives a stream and asserts y_pipe (LATENCY=18 later) == the y_comb that was
// presented 18 cycles earlier. Pure retiming ⇒ ZERO diff expected.
module fp32_exp_p18_tb (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  output logic [31:0] y_comb,
  output logic        valid_out,
  output logic [31:0] y_pipe
);
  fp32_exp     u_comb (.a(a), .y(y_comb));
  fp32_exp_p18 u_pipe (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                       .a(a), .valid_out(valid_out), .y(y_pipe));
endmodule
