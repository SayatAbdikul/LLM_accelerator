// Placeholder ASIC pad ring.
//
// Provides clock/reset synchronization between off-chip pads and the
// on-chip core domain. Step E (2026-05-26 RTL restructure): stub-only,
// no real IO cells or ESD protection. Replace with the PDK's IO library
// (sky130_fd_io_*, etc.) when OpenLane is wired up. For Caravel
// integration the pad ring lives inside the harness, not here.

`ifndef PAD_RING_STUB_SV
`define PAD_RING_STUB_SV

module pad_ring_stub (
  input  logic clk_pad,
  input  logic rst_n_pad,
  output logic clk_core,
  output logic rst_n_core
);

  assign clk_core = clk_pad;

  // 2-FF reset synchronizer (asynchronous reset assertion, synchronous
  // deassertion on the core clock).
  logic [1:0] rst_sync_q;
  always_ff @(posedge clk_core) rst_sync_q <= {rst_sync_q[0], rst_n_pad};
  assign rst_n_core = rst_sync_q[1];

endmodule

`endif // PAD_RING_STUB_SV
