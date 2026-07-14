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

  // --- Port S (read/write, systolic-dedicated) ---
  // The SECOND Port-A-class channel. ABUF / WBUF / ACCUM are three separate
  // dual-port macros, each with its own physical Port A, so a request that
  // targets a DIFFERENT buffer than Port A's can proceed in the SAME cycle —
  // it costs zero new SRAM ports, only a per-buffer fan-out of the request.
  //
  // Before this existed, one shared Port-A bus fed all three macros through a
  // fixed-priority mux with NO backpressure, so the systolic's ACCUM drain write
  // was SILENTLY DROPPED whenever the DMA wanted WBUF in the same cycle —
  // 257,406 lost writes per 124M decode step, corrupting every matmul. The two
  // engines were never actually contending for memory; they were contending for
  // a wire. See docs/phase0_measurement.md.
  //
  // Port A keeps priority. If S targets the SAME buffer as A in the same cycle,
  // S is denied and `s_collision` fires — the top level turns that into an
  // architectural FAULT. That case cannot silently corrupt: it halts.
  input  logic        s_en,
  input  logic        s_we,
  input  logic [1:0]  s_buf,
  input  logic [15:0]  s_row,
  input  logic [127:0] s_wdata,
  output logic [127:0] s_rdata,
  output logic         s_fault,        // 1 = OOB or reserved buf on the S channel
  output logic         s_collision,    // 1 = S and A wanted the same buffer

  // --- Port B (read only) ---
  input  logic        b_en,
  input  logic [1:0]  b_buf,
  input  logic [15:0]  b_row,
  output logic [127:0] b_rdata,
  output logic         b_fault,

  // --- Port W (read only, WBUF-dedicated) ---
  // A second read path hardwired to WBUF's physical Port B, so the systolic
  // array can stream weights from WBUF while the DMA engine writes the next
  // weight tile through the shared Port A (WBUF.PortA) — the two ports of the
  // dual-port WBUF, targeting double-buffered regions, run concurrently.
  // WBUF.PortB is otherwise unused (shared Port B only ever targets ABUF for
  // activations / src1), so this is a free extra read port for weights.
  input  logic        w_en,
  input  logic [15:0] w_row,
  output logic [127:0] w_rdata,
  output logic        w_fault
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
  assign w_fault = oob_check(BUF_WBUF, w_row);
  assign s_fault = s_en && oob_check(s_buf, s_row);

  // -------------------------------------------------------------------------
  // Per-buffer Port-A arbitration (the bus split).
  //
  // For each macro X, exactly one of the two channels drives its physical Port A:
  //   A owns X  <=  A is enabled and targeting X
  //   S owns X  <=  S is enabled and targeting X, and A is NOT targeting X
  //
  // Different buffers => both channels proceed CONCURRENTLY, which is the whole
  // point. Same buffer => A wins and s_collision fires (=> FAULT upstream), so a
  // denied S access can never pass silently.
  // -------------------------------------------------------------------------
  logic a_req_abuf, a_req_wbuf, a_req_accum;
  logic s_req_abuf, s_req_wbuf, s_req_accum;
  logic s_own_abuf, s_own_wbuf, s_own_accum;

  assign a_req_abuf  = a_en && (a_buf == BUF_ABUF)  && !a_fault;
  assign a_req_wbuf  = a_en && (a_buf == BUF_WBUF)  && !a_fault;
  assign a_req_accum = a_en && (a_buf == BUF_ACCUM) && !a_fault;

  assign s_req_abuf  = s_en && (s_buf == BUF_ABUF)  && !s_fault;
  assign s_req_wbuf  = s_en && (s_buf == BUF_WBUF)  && !s_fault;
  assign s_req_accum = s_en && (s_buf == BUF_ACCUM) && !s_fault;

  assign s_own_abuf  = s_req_abuf  && !a_req_abuf;
  assign s_own_wbuf  = s_req_wbuf  && !a_req_wbuf;
  assign s_own_accum = s_req_accum && !a_req_accum;

  // Same-buffer contention. Deliberately compares the RAW requests (not the
  // fault-masked ones): a collision is a scheduling error worth reporting even
  // if the loser was also going to fault.
  assign s_collision = a_en && s_en && (a_buf == s_buf);

  // -------------------------------------------------------------------------
  // ABUF instance and enable decode.
  // -------------------------------------------------------------------------
  logic [127:0] abuf_a_rdata, abuf_b_rdata;
  logic         abuf_a_en, abuf_b_en, abuf_a_we;
  logic [15:0]  abuf_a_row;
  logic [127:0] abuf_a_wdata;

  assign abuf_a_en    = a_req_abuf || s_own_abuf;
  assign abuf_a_we    = s_own_abuf ? s_we    : a_we;
  assign abuf_a_row   = s_own_abuf ? s_row   : a_row;
  assign abuf_a_wdata = s_own_abuf ? s_wdata : a_wdata;
  assign abuf_b_en    = b_en && (b_buf == BUF_ABUF) && !b_fault;

  sram_dp #(.DATA_W(128), .DEPTH(ABUF_ROWS)) u_abuf (
    .clk    (clk),
    .a_en   (abuf_a_en),
    .a_we   (abuf_a_we),
    .a_addr (abuf_a_row[$clog2(ABUF_ROWS)-1:0]),
    .a_wdata(abuf_a_wdata),
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
  logic [15:0]  wbuf_b_row;

  logic [15:0]  wbuf_a_row;
  logic [127:0] wbuf_a_wdata;

  assign wbuf_a_en    = a_req_wbuf || s_own_wbuf;
  assign wbuf_a_we    = s_own_wbuf ? s_we    : a_we;
  assign wbuf_a_row   = s_own_wbuf ? s_row   : a_row;
  assign wbuf_a_wdata = s_own_wbuf ? s_wdata : a_wdata;
  // WBUF Port B is driven by the dedicated W port (weight streaming) when
  // active, else the shared Port B (when it targets WBUF). The two are
  // mutually exclusive — W is asserted only during systolic streaming, and
  // shared-B-to-WBUF only by SFU/helper, which are serialized against the
  // systolic array by the forbidden-overlap invariant. W takes priority.
  assign wbuf_b_en  = (w_en && !w_fault) || (b_en && (b_buf == BUF_WBUF) && !b_fault);
  assign wbuf_b_row = w_en ? w_row : b_row;

  sram_dp #(.DATA_W(128), .DEPTH(WBUF_ROWS)) u_wbuf (
    .clk    (clk),
    .a_en   (wbuf_a_en),
    .a_we   (wbuf_a_we),
    .a_addr (wbuf_a_row[$clog2(WBUF_ROWS)-1:0]),
    .a_wdata(wbuf_a_wdata),
    .a_rdata(wbuf_a_rdata),
    .b_en   (wbuf_b_en),
    .b_addr (wbuf_b_row[$clog2(WBUF_ROWS)-1:0]),
    .b_rdata(wbuf_b_rdata)
  );

  // W port always reads WBUF.PortB, so its returning row is wbuf_b_rdata
  // (registered 1 cycle after the request, matching the shared ports).
  assign w_rdata = wbuf_b_rdata;

  // -------------------------------------------------------------------------
  // ACCUM instance and enable decode.
  // -------------------------------------------------------------------------
  logic [127:0] accum_a_rdata, accum_b_rdata;
  logic         accum_a_en, accum_b_en, accum_a_we;
  logic [15:0]  accum_a_row;
  logic [127:0] accum_a_wdata;

  // This is the one that was being dropped: the systolic's ACCUM drain, losing
  // the shared bus to a DMA weight prefetch bound for a DIFFERENT macro.
  assign accum_a_en    = a_req_accum || s_own_accum;
  assign accum_a_we    = s_own_accum ? s_we    : a_we;
  assign accum_a_row   = s_own_accum ? s_row   : a_row;
  assign accum_a_wdata = s_own_accum ? s_wdata : a_wdata;
  assign accum_b_en    = b_en && (b_buf == BUF_ACCUM) && !b_fault;

  sram_dp #(.DATA_W(128), .DEPTH(ACCUM_ROWS)) u_accum (
    .clk    (clk),
    .a_en   (accum_a_en),
    .a_we   (accum_a_we),
    .a_addr (accum_a_row[$clog2(ACCUM_ROWS)-1:0]),
    .a_wdata(accum_a_wdata),
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
  logic [1:0] a_buf_q, b_buf_q, s_buf_q;
  always_ff @(posedge clk) begin
    a_buf_q <= a_buf;
    b_buf_q <= b_buf;
    s_buf_q <= s_buf;
  end

  always_comb begin
    case (a_buf_q)
      BUF_ABUF:  a_rdata = abuf_a_rdata;
      BUF_WBUF:  a_rdata = wbuf_a_rdata;
      BUF_ACCUM: a_rdata = accum_a_rdata;
      default:   a_rdata = '0;
    endcase
  end

  // The S channel's read return. It reads the SAME physical Port A of whichever
  // macro it owned, so it selects from the same three `*_a_rdata` outputs — and
  // because S only ever owns a macro that A did NOT take that cycle, the two
  // selects can never name the same macro, so both return their own data.
  //
  // This exists for generality, not for the shipped schedule: TODAY the systolic
  // never reads on this channel (ST_DRAIN_RD needs flags_accumulate=1, of which
  // the 124M bundle contains ZERO, and attention's src2 lives in WBUF, which
  // routes to the dedicated W port). Wiring the return only when the current
  // schedule happens to need it is exactly the "correct by schedule, not by
  // construction" mistake that produced the dropped-write bug in the first
  // place. tb_systolic covers the flags=1 read path under DMA overlap.
  always_comb begin
    case (s_buf_q)
      BUF_ABUF:  s_rdata = abuf_a_rdata;
      BUF_WBUF:  s_rdata = wbuf_a_rdata;
      BUF_ACCUM: s_rdata = accum_a_rdata;
      default:   s_rdata = '0;
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
