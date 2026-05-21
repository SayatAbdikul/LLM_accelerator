// Shared synthesizable fp32 ALU — composes the Phase-1 primitives into the
// uniform fixed-latency valid handshake the SFU FSM (Phase 2) will use.
// Single-issue: one op per cycle, LATENCY-cycle result. This is the canonical
// drop-in for the sfu_engine.sv DPI->synth migration.
//
// Each op routes through its dedicated primitive instance (`fp32_add`,
// `fp32_mul`, ...) on the combinational path; the unit's output mux selects
// based on the issued op-code, and a single shared output-register pipeline
// gives the uniform LATENCY contract. Unused primitives' inputs are tied to
// 0/1 so they idle without spurious switching.
//
// Op encoding (4-bit):
//   0x0  ADD       y = fp32_add(a, b)
//   0x1  SUB       y = fp32_add(a, b ^ 0x80000000)
//   0x2  MUL       y = fp32_mul(a, b)
//   0x3  DIV       y = fp32_div(a, b)
//   0x4  SQRT      y = fp32_sqrt(a)
//   0x5  CVT_H2F   y = fp16_to_fp32(a[15:0])
//   0x6  CVT_F2H   y[15:0] = fp32_to_fp16(a); y[31:16]=0
//   0x7  CVT_I2F   y = i32_to_fp32(signed a)
//   0x8  QUANT_I8  y[7:0] = fp32_quantize_i8(a); y[31:8]=sign-ext
//   0x9..0xF reserved (EXP / GELU_NEW are NOT in this minimum ALU;
//            they instantiate separately in sfu_engine.sv where used —
//            their compositions are far heavier than the basic 1-2-mul ops
//            here and Phase 2.8 will share or sequence them differently).
//
// Output semantics:
//   y is 32-bit; for CVT_F2H the fp16 result is in y[15:0]; for QUANT_I8 the
//   int8 result is sign-extended to 32 bits. The FSM extracts the right bits
//   based on the op it issued.

`ifndef FP32_ALU_PIPE_SV
`define FP32_ALU_PIPE_SV

`include "fp32_add.sv"
`include "fp32_mul.sv"
`include "fp32_div.sv"
`include "fp32_sqrt.sv"
`include "fp16_to_fp32.sv"
`include "fp32_to_fp16.sv"
`include "i32_to_fp32.sv"
`include "fp32_quantize_i8.sv"

module fp32_alu_pipe #(
  parameter int unsigned LATENCY = 1
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [3:0]  op,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        valid_out,
  output logic [3:0]  op_out,         // forwarded so the FSM knows the result type
  output logic [31:0] y
);
  localparam logic [3:0] OP_ADD     = 4'h0;
  localparam logic [3:0] OP_SUB     = 4'h1;
  localparam logic [3:0] OP_MUL     = 4'h2;
  localparam logic [3:0] OP_DIV     = 4'h3;
  localparam logic [3:0] OP_SQRT    = 4'h4;
  localparam logic [3:0] OP_CVT_H2F = 4'h5;
  localparam logic [3:0] OP_CVT_F2H = 4'h6;
  localparam logic [3:0] OP_CVT_I2F = 4'h7;
  localparam logic [3:0] OP_QUANT_I8= 4'h8;

  // --- primitive instances (parallel; mux selects the active one) ---
  logic [31:0] y_add, y_sub, y_mul, y_div, y_sqrt, y_h2f, y_i2f;
  logic [15:0] y_f2h;
  logic signed [7:0] y_qi8;
  // sub uses the existing add core with b's sign-bit flipped (cheap; reuse)
  logic [31:0] sub_b;
  assign sub_b = b ^ 32'h8000_0000;

  fp32_add        u_add    (.a(a),       .b(b),      .y(y_add));
  fp32_add        u_sub    (.a(a),       .b(sub_b),  .y(y_sub));
  fp32_mul        u_mul    (.a(a),       .b(b),      .y(y_mul));
  fp32_div        u_div    (.a(a),       .b(b),      .y(y_div));
  fp32_sqrt       u_sqrt   (.a(a),                   .y(y_sqrt));
  fp16_to_fp32    u_h2f    (.a(a[15:0]),             .y(y_h2f));
  fp32_to_fp16    u_f2h    (.a(a),                   .y(y_f2h));
  i32_to_fp32     u_i2f    (.a($signed(a)),          .y(y_i2f));
  fp32_quantize_i8 u_qi8   (.a(a),                   .y(y_qi8));

  // --- combinational op-mux ---
  logic [31:0] y_comb;
  always_comb begin
    case (op)
      OP_ADD:      y_comb = y_add;
      OP_SUB:      y_comb = y_sub;
      OP_MUL:      y_comb = y_mul;
      OP_DIV:      y_comb = y_div;
      OP_SQRT:     y_comb = y_sqrt;
      OP_CVT_H2F:  y_comb = y_h2f;
      OP_CVT_F2H:  y_comb = {16'd0, y_f2h};
      OP_CVT_I2F:  y_comb = y_i2f;
      OP_QUANT_I8: y_comb = {{24{y_qi8[7]}}, y_qi8};  // sign-extend
      default:     y_comb = 32'd0;
    endcase
  end

  // --- fixed-latency pipeline ---
  logic [31:0] y_pipe  [0:LATENCY-1];
  logic [3:0]  op_pipe [0:LATENCY-1];
  logic        v_pipe  [0:LATENCY-1];

  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < LATENCY; i = i + 1) begin
        v_pipe[i]  <= 1'b0;
        y_pipe[i]  <= 32'd0;
        op_pipe[i] <= 4'd0;
      end
    end else begin
      v_pipe[0]  <= valid_in;
      y_pipe[0]  <= y_comb;
      op_pipe[0] <= op;
      for (i = 1; i < LATENCY; i = i + 1) begin
        v_pipe[i]  <= v_pipe[i-1];
        y_pipe[i]  <= y_pipe[i-1];
        op_pipe[i] <= op_pipe[i-1];
      end
    end
  end

  assign valid_out = v_pipe[LATENCY-1];
  assign op_out    = op_pipe[LATENCY-1];
  assign y         = y_pipe[LATENCY-1];

endmodule

`endif // FP32_ALU_PIPE_SV
