`ifndef FP32_GELU_NEW_PIPE_SV
`define FP32_GELU_NEW_PIPE_SV
`include "fp32_gelu_new.sv"
module fp32_gelu_new_pipe #(parameter int unsigned LATENCY = 1) (
  input  logic clk, input logic rst_n,
  input  logic valid_in, input logic [31:0] a,
  output logic valid_out, output logic [31:0] y);
  logic [31:0] y_comb;
  fp32_gelu_new u_core (.a(a), .y(y_comb));
  logic [31:0] y_pipe [0:LATENCY-1];
  logic        v_pipe [0:LATENCY-1];
  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < LATENCY; i = i + 1) begin v_pipe[i]<=1'b0; y_pipe[i]<=32'd0; end
    end else begin
      v_pipe[0] <= valid_in; y_pipe[0] <= y_comb;
      for (i = 1; i < LATENCY; i = i + 1) begin v_pipe[i]<=v_pipe[i-1]; y_pipe[i]<=y_pipe[i-1]; end
    end
  end
  assign valid_out = v_pipe[LATENCY-1];
  assign y         = y_pipe[LATENCY-1];
endmodule
`endif
