`ifndef SYSTOLIC_ARRAY_SV
`define SYSTOLIC_ARRAY_SV

`include "taccel_pkg.sv"

// 16x16 systolic mesh wrapper.
//
// Input rows arrive as packed 16-byte vectors. The wrapper unpacks them into
// lane vectors, optionally applies boundary skew for chained mode, and wires
// the PE mesh in either:
  //   - broadcast mode: each row/column sees the same lane value
  //   - chained mode: operands flow across the mesh one PE per cycle

module systolic_array
  import taccel_pkg::*;
#(
  parameter int SYSTOLIC_ARCH_MODE = SYS_MODE_DEFAULT
)
(
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         step_en,
  input  logic                         clear_acc,
  input  logic [AXI_DATA_W-1:0]        a_row_data,
  input  logic [AXI_DATA_W-1:0]        b_row_data,
  output logic [SYS_DIM*SYS_DIM*32-1:0] acc_flat
);

  logic [7:0] a_vec [0:SYS_DIM-1];
  logic [7:0] b_vec [0:SYS_DIM-1];
  logic [7:0] a_edge_vec [0:SYS_DIM-1];
  logic [7:0] b_edge_vec [0:SYS_DIM-1];
  logic [7:0] a_skew [0:SYS_DIM-1][0:SYS_DIM-2];
  logic [7:0] b_skew [0:SYS_DIM-1][0:SYS_DIM-2];

  // PE-local state and interconnect signals.
  logic [31:0] pe_acc [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [7:0] pe_a_in  [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [7:0] pe_b_in  [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [7:0] pe_a_out [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [7:0] pe_b_out [0:SYS_DIM-1][0:SYS_DIM-1];

  genvar i, j;
  // Unpack the incoming 128-bit rows into 16 signed INT8 lanes and select the
  // edge-fed values used in chained mode after skew insertion.
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_A_B
      assign a_vec[i] = a_row_data[i*8 +: 8];
      assign b_vec[i] = b_row_data[i*8 +: 8];

      if (i == 0) begin : GEN_EDGE_NO_DELAY
        assign a_edge_vec[i] = a_vec[i];
        assign b_edge_vec[i] = b_vec[i];
      end else begin : GEN_EDGE_DELAYED
        assign a_edge_vec[i] = a_skew[i][i-1];
        assign b_edge_vec[i] = b_skew[i][i-1];
      end
    end
  endgenerate

  // Chained systolic mode requires boundary skew so A/B operands that belong
  // to the same k arrive at each PE on the same cycle.
  always_ff @(posedge clk or negedge rst_n) begin : SKew_PIPE
    int r, s;
    if (!rst_n) begin
      for (r = 0; r < SYS_DIM; r++) begin
        for (s = 0; s < SYS_DIM-1; s++) begin
          a_skew[r][s] <= 8'h00;
          b_skew[r][s] <= 8'h00;
        end
      end
    end else if (clear_acc) begin
      for (r = 0; r < SYS_DIM; r++) begin
        for (s = 0; s < SYS_DIM-1; s++) begin
          a_skew[r][s] <= 8'h00;
          b_skew[r][s] <= 8'h00;
        end
      end
    end else if (step_en) begin
      for (r = 0; r < SYS_DIM; r++) begin
        a_skew[r][0] <= a_vec[r];
        b_skew[r][0] <= b_vec[r];
        for (s = 1; s < SYS_DIM-1; s++) begin
          a_skew[r][s] <= a_skew[r][s-1];
          b_skew[r][s] <= b_skew[r][s-1];
        end
      end
    end
  end

  // Dual-mode routing scaffold:
  // - Broadcast mode: all PEs in row/col see same a_vec/b_vec lane.
  // - Chained mode: left/top edge injects a_vec/b_vec and interior PEs consume
  //   neighbor outputs (west/east for A, north/south for B).
  // Route either broadcast inputs or neighbor-forwarded chained inputs into
  // each PE. This keeps the PE itself oblivious to the global architecture.
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_ROUTE_ROW
      for (j = 0; j < SYS_DIM; j++) begin : GEN_ROUTE_COL
        always_comb begin
          if (SYSTOLIC_ARCH_MODE == SYS_MODE_CHAINED) begin
            pe_a_in[i][j] = (j == 0) ? a_edge_vec[i] : pe_a_out[i][j-1];
            pe_b_in[i][j] = (i == 0) ? b_edge_vec[j] : pe_b_out[i-1][j];
          end else begin
            pe_a_in[i][j] = a_vec[i];
            pe_b_in[i][j] = b_vec[j];
          end
        end
      end
    end
  endgenerate

  // Instantiate the full mesh and flatten the accumulator matrix for the
  // controller's writeback logic.
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_ROW
      for (j = 0; j < SYS_DIM; j++) begin : GEN_COL
        systolic_pe u_pe (
          .clk       (clk),
          .rst_n     (rst_n),
          .en        (step_en),
          .acc_clear (clear_acc),
          .a_in      (pe_a_in[i][j]),
          .b_in      (pe_b_in[i][j]),
          .a_out     (pe_a_out[i][j]),
          .b_out     (pe_b_out[i][j]),
          .acc       (pe_acc[i][j])
        );

        localparam int FLAT = (i * SYS_DIM + j) * 32;
        assign acc_flat[FLAT +: 32] = pe_acc[i][j];
      end
    end
  endgenerate

endmodule

`endif // SYSTOLIC_ARRAY_SV
