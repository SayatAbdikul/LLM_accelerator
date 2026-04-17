// SRAM subsystem: ABUF, WBUF, ACCUM -- three dual-port SRAMs.
//
// Each SRAM row is 128 bits (16 bytes), matching the 16-byte DMA transfer unit.
//
//   ABUF : 8192 rows × 16 B = 128 KB  (INT8 activations)
//   WBUF : 16384 rows × 16 B = 256 KB  (INT8 weights / FP16 params / INT32 bias)
//   ACCUM: 4096 rows × 16 B =  64 KB  (INT32, 4 elements per row, little-endian)
//
// Port A (read/write): DMA engine, BUF_COPY, issue-stage write-back
// Port B (read only):  Systolic array, SFU
//
// Address decoding: the caller passes a buffer ID (2-bit) and a row offset
// (16-bit, in 16-byte units).  Muxing to the correct SRAM is done here.
//
// OOB check: asserts sram_fault if the offset exceeds the buffer's max row.
// The SRAMs themselves are not touched when a fault is detected; callers use
// the fault bit to convert the access into an architectural error.

`ifndef SRAM_SUBSYSTEM_SV
`define SRAM_SUBSYSTEM_SV

`include "taccel_pkg.sv"

module sram_subsystem
  import taccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // --- Port A (read/write) ---
  input  logic        a_en,
  input  logic        a_we,
  input  logic [1:0]  a_buf,          // BUF_ABUF / BUF_WBUF / BUF_ACCUM
  input  logic [15:0]  a_row,         // row offset in 16-byte units
  input  logic [127:0] a_wdata,
  output logic [127:0] a_rdata,
  output logic         a_fault,       // 1 = OOB or reserved buf

  // --- Port B (read only) ---
  input  logic        b_en,
  input  logic [1:0]  b_buf,
  input  logic [15:0]  b_row,
  output logic [127:0] b_rdata,
  output logic         b_fault
);

  // -------------------------------------------------------------------------
  // Bounds check helper
  // -------------------------------------------------------------------------
  function automatic logic oob_check(
    input logic [1:0]  bid,
    input logic [15:0] row
  );
    case (bid)
      BUF_ABUF:  return (row >= 16'(ABUF_ROWS));
      BUF_WBUF:  return (row >= 16'(WBUF_ROWS));
      BUF_ACCUM: return (row >= 16'(ACCUM_ROWS));
      default:   return 1'b1;  // reserved buffer ID → fault
    endcase
  endfunction

  assign a_fault = oob_check(a_buf, a_row);
  assign b_fault = oob_check(b_buf, b_row);

  // -------------------------------------------------------------------------
  // ABUF instance and enable decode.
  // -------------------------------------------------------------------------
  logic [127:0] abuf_a_rdata, abuf_b_rdata;
  logic         abuf_a_en, abuf_b_en, abuf_a_we;

  assign abuf_a_en = a_en && (a_buf == BUF_ABUF) && !a_fault;
  assign abuf_a_we = a_we;
  assign abuf_b_en = b_en && (b_buf == BUF_ABUF) && !b_fault;

  sram_dp #(.DATA_W(128), .DEPTH(ABUF_ROWS)) u_abuf (
    .clk    (clk),
    .a_en   (abuf_a_en),
    .a_we   (abuf_a_we),
    .a_addr (a_row[$clog2(ABUF_ROWS)-1:0]),
    .a_wdata(a_wdata),
    .a_rdata(abuf_a_rdata),
    .b_en   (abuf_b_en),
    .b_addr (b_row[$clog2(ABUF_ROWS)-1:0]),
    .b_rdata(abuf_b_rdata)
  );

  // -------------------------------------------------------------------------
  // WBUF instance and enable decode.
  // -------------------------------------------------------------------------
  logic [127:0] wbuf_a_rdata, wbuf_b_rdata;
  logic         wbuf_a_en, wbuf_b_en, wbuf_a_we;

  assign wbuf_a_en = a_en && (a_buf == BUF_WBUF) && !a_fault;
  assign wbuf_a_we = a_we;
  assign wbuf_b_en = b_en && (b_buf == BUF_WBUF) && !b_fault;

  sram_dp #(.DATA_W(128), .DEPTH(WBUF_ROWS)) u_wbuf (
    .clk    (clk),
    .a_en   (wbuf_a_en),
    .a_we   (wbuf_a_we),
    .a_addr (a_row[$clog2(WBUF_ROWS)-1:0]),
    .a_wdata(a_wdata),
    .a_rdata(wbuf_a_rdata),
    .b_en   (wbuf_b_en),
    .b_addr (b_row[$clog2(WBUF_ROWS)-1:0]),
    .b_rdata(wbuf_b_rdata)
  );

  // -------------------------------------------------------------------------
  // ACCUM instance and enable decode.
  // -------------------------------------------------------------------------
  logic [127:0] accum_a_rdata, accum_b_rdata;
  logic         accum_a_en, accum_b_en, accum_a_we;

  assign accum_a_en = a_en && (a_buf == BUF_ACCUM) && !a_fault;
  assign accum_a_we = a_we;
  assign accum_b_en = b_en && (b_buf == BUF_ACCUM) && !b_fault;

  sram_dp #(.DATA_W(128), .DEPTH(ACCUM_ROWS)) u_accum (
    .clk    (clk),
    .a_en   (accum_a_en),
    .a_we   (accum_a_we),
    .a_addr (a_row[$clog2(ACCUM_ROWS)-1:0]),
    .a_wdata(a_wdata),
    .a_rdata(accum_a_rdata),
    .b_en   (accum_b_en),
    .b_addr (b_row[$clog2(ACCUM_ROWS)-1:0]),
    .b_rdata(accum_b_rdata)
  );

  // -------------------------------------------------------------------------
  // Read-data mux.
  // `sram_dp` returns data one cycle after the request, so we register the
  // selected buffer ID and use it to choose the returning row.
  // -------------------------------------------------------------------------
  logic [1:0] a_buf_q, b_buf_q;
  always_ff @(posedge clk) begin
    a_buf_q <= a_buf;
    b_buf_q <= b_buf;
  end

  always_comb begin
    case (a_buf_q)
      BUF_ABUF:  a_rdata = abuf_a_rdata;
      BUF_WBUF:  a_rdata = wbuf_a_rdata;
      BUF_ACCUM: a_rdata = accum_a_rdata;
      default:   a_rdata = '0;
    endcase
  end

  always_comb begin
    case (b_buf_q)
      BUF_ABUF:  b_rdata = abuf_b_rdata;
      BUF_WBUF:  b_rdata = wbuf_b_rdata;
      BUF_ACCUM: b_rdata = accum_b_rdata;
      default:   b_rdata = '0;
    endcase
  end

endmodule

`endif // SRAM_SUBSYSTEM_SV
