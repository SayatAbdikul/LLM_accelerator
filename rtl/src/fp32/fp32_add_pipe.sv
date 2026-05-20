// fp32_add_pipe — pipelined shell over the proven fp32_add combinational
// core. Establishes the uniform fixed-latency valid handshake every
// synthesizable fp32 primitive uses, so the SFU FSM (Phase 2) can drop in
// any primitive with a deterministic LATENCY-cycle wait — no elastic
// backpressure, no FSM redesign.
//
// Contract: at any cycle with valid_in==1, the unit captures {a,b}; the
// result y is valid LATENCY cycles later, with valid_out==1 in that cycle.
// One operation can be issued per cycle (throughput 1 op/cycle).
//
// The combinational fp32_add core (bit-exact vs sfu_fp32_add, 6M/0) is
// instantiated unchanged — `wrap, do not rewrite` (plan Phase 1).

`ifndef FP32_ADD_PIPE_SV
`define FP32_ADD_PIPE_SV

`include "fp32_add.sv"

module fp32_add_pipe #(
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
  // Combinational arithmetic core (proven bit-exact).
  logic [31:0] y_comb;
  fp32_add u_core (.a(a), .b(b), .y(y_comb));

  // Fixed-latency pipeline of (valid, result). LATENCY must be >= 1; the
  // shell's purpose is to register the combinational output so downstream
  // FSMs see a clean clk-aligned valid handshake. Deeper LATENCY allows
  // breaking the combinational path for fmax (Phase 2.8 perf work).
  logic [31:0] y_pipe [0:LATENCY-1];
  logic        v_pipe [0:LATENCY-1];

  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < LATENCY; i = i + 1) begin
        v_pipe[i] <= 1'b0;
        y_pipe[i] <= 32'd0;
      end
    end else begin
      v_pipe[0] <= valid_in;
      y_pipe[0] <= y_comb;
      for (i = 1; i < LATENCY; i = i + 1) begin
        v_pipe[i] <= v_pipe[i-1];
        y_pipe[i] <= y_pipe[i-1];
      end
    end
  end

  assign valid_out = v_pipe[LATENCY-1];
  assign y         = y_pipe[LATENCY-1];

endmodule

`endif // FP32_ADD_PIPE_SV
