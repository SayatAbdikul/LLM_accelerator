// fp32_sub_pipe — IEEE-754 binary32 subtract via the proven fp32_add core.
// Identity: a - b == a + (-b), and IEEE negation is the sign-bit XOR
// (preserves the bit-pattern for NaN/inf/zero/subnormal). The DPI golden
// `sfu_fp32_sub` (testbench.h) returns (float)((float)a-(float)b), which is
// exactly what `fp32_add(a, b ^ 0x80000000)` computes — bit-exact, no
// rounding artifacts (single-rounded by the proven add core).
//
// Pipelined-first contract: same uniform fixed-latency handshake as
// fp32_add_pipe (LATENCY param, valid_in/valid_out, 1 op/cycle throughput).

`ifndef FP32_SUB_PIPE_SV
`define FP32_SUB_PIPE_SV

`include "fp32_add_pipe.sv"

module fp32_sub_pipe #(
  parameter int unsigned LATENCY = 1
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        valid_out,
  output logic [31:0] y
);
  // Negate b by flipping its IEEE-754 sign bit (works for all classes
  // including NaN, ±inf, ±0, subnormal — the bit-pattern semantics).
  logic [31:0] neg_b;
  assign neg_b = b ^ 32'h8000_0000;

  fp32_add_pipe #(.LATENCY(LATENCY)) u_add (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_in (valid_in),
    .a        (a),
    .b        (neg_b),
    .valid_out(valid_out),
    .y        (y)
  );

endmodule

`endif // FP32_SUB_PIPE_SV
