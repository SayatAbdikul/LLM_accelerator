`ifndef SYSTOLIC_PE_SV
`define SYSTOLIC_PE_SV

// One processing element in the 16x16 systolic mesh.
// Each cycle it forwards A/B to its east/south neighbors and accumulates
// signed INT8 x INT8 products into a 32-bit accumulator.

module systolic_pe (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        en,
  input  logic        acc_clear,
  input  logic [7:0]  a_in,
  input  logic [7:0]  b_in,
  output logic [7:0]  a_out,
  output logic [7:0]  b_out,
  output logic [31:0] acc
);

  logic signed [7:0]  a_s;
  logic signed [7:0]  b_s;
  logic signed [15:0] prod_s;

  assign a_s = a_in;
  assign b_s = b_in;
  assign prod_s = a_s * b_s;

  // Forwarding and accumulation happen only on `en`, so the controller can
  // treat the array as a lockstep pipeline that advances once per READ_USE
  // cycle. When acc_clear asserts at a tile boundary we also clear the
  // forwarded A/B registers; chained mode depends on those delay registers
  // starting from zero rather than whatever stale SRAM row was last observed.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 8'h0;
      b_out <= 8'h0;
      acc   <= 32'h0;
    end else begin
      if (acc_clear) begin
        a_out <= 8'h0;
        b_out <= 8'h0;
        acc <= 32'h0;
      end else if (en) begin
        a_out <= a_in;
        b_out <= b_in;
        acc <= $signed(acc) + $signed({{16{prod_s[15]}}, prod_s});
      end
    end
  end

endmodule

`endif // SYSTOLIC_PE_SV
