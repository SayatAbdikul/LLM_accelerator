// Placeholder ASIC pad ring.
//
// Passes the clock pad through and samples reset through two flops. This is
// elaboration-only: it has no real IO cells, ESD protection, vendor attributes,
// or explicit power-on initialization. Replace it with the integration
// target's characterized IO/reset solution.

`ifndef PAD_RING_STUB_SV
`define PAD_RING_STUB_SV

module pad_ring_stub (
  input  logic clk_pad,
  input  logic rst_n_pad,
  output logic clk_core,
  output logic rst_n_core
);

  assign clk_core = clk_pad;

  // Two-flop sampling on the core clock. Both assertion and deassertion are
  // synchronous in this placeholder implementation.
  logic [1:0] rst_sync_q;
  always_ff @(posedge clk_core) rst_sync_q <= {rst_sync_q[0], rst_n_pad};
  assign rst_n_core = rst_sync_q[1];

endmodule

`endif // PAD_RING_STUB_SV
