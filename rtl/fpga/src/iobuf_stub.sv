// IO buffer + 2-FF reset synchronizer.
//
// Replace with vendor IBUF + ASYNC_REG / dont_touch synchronizer FFs when
// the target FPGA is picked. This stub provides minimal metastability
// hardening for an asynchronous reset pin.
//
// Step D (2026-05-26 RTL restructure): stub-only so the FPGA wrapper
// elaborates without vendor IBUF cells.

`ifndef IOBUF_STUB_SV
`define IOBUF_STUB_SV

module iobuf_stub (
  input  logic pin,
  input  logic clk,
  output logic out
);
  logic [1:0] sync_q;
  always_ff @(posedge clk) sync_q <= {sync_q[0], pin};
  assign out = sync_q[1];
endmodule

`endif // IOBUF_STUB_SV
