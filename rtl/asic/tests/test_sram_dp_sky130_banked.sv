// Compare the isolated adapter with the actual shared inferred SRAM using the
// pinned upstream macro model. Logical collisions are REQUIRED coverage;
// physical macro collisions are forbidden. No timing signoff is implied.
`timescale 1ns/1ps
module sram_bank_equivalence_case #(
  parameter integer DATA_W = 128,
  parameter integer DEPTH = 1024,
  // 0: uninitialized registers; 1: arbitrary invalid pending entry;
  // 2/3/4: arbitrary valid pending entry, first access read/idle/write;
  // 5: uninitialized registers, first access read (not write).
  parameter integer STARTUP = 0
)(output reg done = 0);
  localparam integer ADDR_W = $clog2(DEPTH);
  localparam integer START_ROW = DEPTH - 17;
  reg clk = 0;
  reg a_en = 0, a_we = 0, b_en = 0;
  reg [ADDR_W-1:0] a_addr = 0, b_addr = 0;
  reg [DATA_W-1:0] a_wdata = 0;
  wire [DATA_W-1:0] a_rdata, b_rdata, ref_a_rdata, ref_b_rdata;
  integer checks = 0, collisions = 0, commit_forwards = 0;
  integer a_forwards = 0, b_forwards = 0, deferred = 0;
  integer i, aa, ba;
  reg [31:0] rng = 32'h1874abcd ^ DATA_W ^ DEPTH ^ STARTUP;

  sram_dp_sky130_banked #(.DATA_W(DATA_W), .DEPTH(DEPTH)) dut (.*);
  sram_dp_inferred #(.DATA_W(DATA_W), .DEPTH(DEPTH)) reference (
    .clk(clk), .a_en(a_en), .a_we(a_we), .a_addr(a_addr), .a_wdata(a_wdata),
    .a_rdata(ref_a_rdata), .b_en(b_en), .b_addr(b_addr), .b_rdata(ref_b_rdata)
  );

  function automatic [DATA_W-1:0] pattern(input integer row, input integer epoch);
    integer lane;
    reg [31:0] value;
    begin
      for (lane = 0; lane < DATA_W/32; lane = lane + 1) begin
        value = 32'(row) ^ (32'(epoch) * 32'h9e3779b9) ^ (32'(lane) * 32'h85ebca6b);
        value = (value ^ (value >> 16)) * 32'h7feb352d;
        value = (value ^ (value >> 15)) * 32'h846ca68b;
        pattern[lane*32 +: 32] = value ^ (value >> 16);
      end
    end
  endfunction

  // Check actual macro input pins, independent of the adapter's own assertion.
  genvar bank, lane;
  generate
    for (bank = 0; bank < DEPTH/512; bank = bank + 1) begin : check_banks
      for (lane = 0; lane < DATA_W/32; lane = lane + 1) begin : check_lanes
        always @(posedge clk) begin
          if (dut.banks[bank].lanes[lane].u_macro.csb0 === 1'b0 &&
              dut.banks[bank].lanes[lane].u_macro.web0 === 1'b0 &&
              dut.banks[bank].lanes[lane].u_macro.csb1 === 1'b0 &&
              dut.banks[bank].lanes[lane].u_macro.addr0 ==
                dut.banks[bank].lanes[lane].u_macro.addr1)
            $fatal(1, "undefined physical macro collision at bank %0d lane %0d", bank, lane);
        end
      end
    end
  endgenerate

  // Rising->falling is 10ns; check 9ns after falling, beyond the model's
  // arbitrary 3ns delay. Disabled outputs are compared as well as active ones.
  task automatic cycle(input bit ae, input bit we, input integer addr_a,
                       input [DATA_W-1:0] wd, input bit be, input integer addr_b);
    reg saved_valid;
    reg [ADDR_W-1:0] saved_addr;
    reg [DATA_W-1:0] saved_data;
    begin
      a_en = ae; a_we = we; a_addr = ADDR_W'(addr_a); a_wdata = wd;
      b_en = be; b_addr = ADDR_W'(addr_b);
      #1;
      saved_valid = dut.pending_valid_q;
      saved_addr = dut.pending_addr_q;
      saved_data = dut.pending_data_q;
      if (ae && we && be && addr_a == addr_b) collisions = collisions + 1;
      if (dut.commit_w && dut.b_forward_w) commit_forwards = commit_forwards + 1;
      if (dut.a_forward_w) a_forwards = a_forwards + 1;
      if (dut.b_forward_w) b_forwards = b_forwards + 1;
      if (ae && !we && dut.pending_valid_q) deferred = deferred + 1;
      clk = 1;
      #10 clk = 0;
      #9;
      if (a_rdata !== ref_a_rdata)
        $fatal(1, "%0dx%0d startup%0d cycle%0d A addr%0d got %h expected %h",
               DATA_W, DEPTH, STARTUP, checks, addr_a, a_rdata, ref_a_rdata);
      if (b_rdata !== ref_b_rdata)
        $fatal(1, "%0dx%0d startup%0d cycle%0d B addr%0d got %h expected %h",
               DATA_W, DEPTH, STARTUP, checks, addr_b, b_rdata, ref_b_rdata);
      if (ae && !we && (dut.pending_valid_q !== saved_valid ||
                       dut.pending_addr_q !== saved_addr ||
                       dut.pending_data_q !== saved_data))
        $fatal(1, "A read changed the deferred pending entry");
      checks = checks + 1;
    end
  endtask

  initial begin
    // No reset or idle priming cycle. All physical locations start unknown.
    // An arbitrary valid startup entry is part of the arbitrary logical
    // initial memory, so only that overlay row is seeded in the reference.
    // These assignments are test stimuli, never hardware initialization.
    if (STARTUP >= 1 && STARTUP <= 4) begin
      dut.pending_valid_q = (STARTUP >= 2);
      dut.pending_addr_q = ADDR_W'(START_ROW);
      dut.pending_data_q = pattern(START_ROW, 938 + DATA_W);
    end
    if (STARTUP >= 2 && STARTUP <= 4) begin
      reference.mem[START_ROW] = pattern(START_ROW, 938 + DATA_W);
      if (STARTUP == 3)
        cycle(0, 0, 0, 0, 1, START_ROW); // initial commit must forward B
      if (STARTUP == 4)
        cycle(1, 1, START_ROW, pattern(START_ROW, 777), 1, START_ROW);
      cycle(1, 0, START_ROW, 0, 1, START_ROW);
      cycle(1, 0, 3, 0, 1, 4); // other initial rows remain unspecified
      cycle(1, 0, START_ROW, 0, 1, START_ROW);
    end
    if (STARTUP == 5)
      cycle(1, 0, 0, 0, 1, 0); // arbitrary initial contents may be unknown
    cycle(1, 1, 0, pattern(0, 0), 1, 0); // first write; B sees unknown old value
    cycle(1, 0, 0, 0, 1, 0); // both ports must forward the pending first write
    for (i = 1; i < DEPTH; i = i + 1)
      cycle(1, 1, i, pattern(i, 0), 1, i - 1);
    // Final initialization write is still pending; dual-read runs may defer
    // it indefinitely without stalling either port or losing its latest value.
    for (i = 0; i < 128; i = i + 1)
      cycle(1, 0, (i % 3 == 0) ? DEPTH-1 : i, 0,
            1, (i % 2 == 0) ? DEPTH-1 : DEPTH-2-i);
    cycle(0, 0, 7, 0, 1, DEPTH-1); // commit+forward old pending B
    cycle(0, 1, 33, pattern(33, 2), 0, 0); // disabled A_WE must not write
    cycle(0, 0, DEPTH-1, 0, 0, DEPTH-1); // both outputs hold through macro X

    // Logical same-row collisions: B sees the previous logical value, never
    // the current write or the older physical value hidden by the overlay.
    for (i = 1; i <= 32; i = i + 1)
      cycle(1, 1, 511, pattern(511, i), 1, 511);
    cycle(1, 0, 511, 0, 1, 511);
    cycle(1, 1, 0, pattern(0, 99), 1, 511); // replace row, B sees prior pending
    cycle(1, 0, 0, 0, 1, 511);
    // Both bank/lane boundaries and queue replacement across banks.
    cycle(1, 1, DEPTH-1, pattern(DEPTH-1, 101), 1, 0);
    cycle(1, 1, 0, pattern(0, 102), 1, DEPTH-1);
    cycle(1, 0, DEPTH-1, 0, 1, 0);
    cycle(1, 0, 0, 0, 1, DEPTH-1);
    for (i = 0; i < DEPTH/512; i = i + 1) begin
      cycle(1, 1, i*512, pattern(i*512, 103), 1, i*512);
      cycle(1, 0, i*512, 0, 1, i*512+511);
      cycle(0, 0, 0, 0, 1, i*512);
    end

    for (i = 0; i < 10000; i = i + 1) begin
      rng = rng ^ (rng << 13); rng = rng ^ (rng >> 17); rng = rng ^ (rng << 5);
      aa = rng % DEPTH;
      ba = (rng >> 13) % DEPTH;
      // Deliberately bias collisions/forwarding; uniform random addresses
      // alone would rarely stress the one-entry lifetime and ordering.
      case (i % 6)
        0: ba = aa;
        1: if (dut.pending_valid_q) ba = dut.pending_addr_q;
        2: if (dut.pending_valid_q) aa = dut.pending_addr_q;
        3: if (dut.pending_valid_q) begin aa = dut.pending_addr_q; ba = aa; end
        default: begin end
      endcase
      cycle(rng[28], rng[29], aa, pattern(aa, i + 1000), rng[30], ba);
    end
    cycle(0, 0, 0, 0, 0, 0); // flush pending while preserving output holds
    // Exhaustive post-stress readback proves no row was lost during commits.
    for (i = 0; i < DEPTH; i = i + 1)
      cycle(1, 0, i, 0, 1, DEPTH-1-i);
    cycle(0, 0, 0, 0, 0, 0);
    if (collisions < 100 || commit_forwards < 100 ||
        a_forwards < 100 || b_forwards < 100 || deferred < 128)
      $fatal(1, "insufficient pending/collision/defer coverage");
    $display("PASS: SKY130 %0dx%0d startup%0d: %0d equivalent cycles; logical collisions=%0d commit+B forwards=%0d A/B forwards=%0d/%0d deferred=%0d",
             DATA_W, DEPTH, STARTUP, checks, collisions, commit_forwards,
             a_forwards, b_forwards, deferred);
    done = 1;
  end
endmodule

module test_sram_dp_sky130_banked;
  wire [7:0] done;
  sram_bank_equivalence_case #(.DATA_W(32), .DEPTH(512), .STARTUP(0)) c0(done[0]);
  sram_bank_equivalence_case #(.DATA_W(128), .DEPTH(1024), .STARTUP(0)) c1(done[1]);
  sram_bank_equivalence_case #(.DATA_W(128), .DEPTH(1024), .STARTUP(1)) c2(done[2]);
  sram_bank_equivalence_case #(.DATA_W(64), .DEPTH(1024), .STARTUP(2)) c3(done[3]);
  sram_bank_equivalence_case #(.DATA_W(128), .DEPTH(2048), .STARTUP(2)) c4(done[4]);
  sram_bank_equivalence_case #(.DATA_W(128), .DEPTH(1024), .STARTUP(3)) c5(done[5]);
  sram_bank_equivalence_case #(.DATA_W(32), .DEPTH(512), .STARTUP(4)) c6(done[6]);
  sram_bank_equivalence_case #(.DATA_W(32), .DEPTH(512), .STARTUP(5)) c7(done[7]);
  initial begin
    wait (&done);
    $display("PASS: all eight SRAM adapter geometries/startup cases preserve inferred old-data semantics; zero physical collisions");
    $finish;
  end
  initial begin
    #1000000;
    $fatal(1, "SRAM equivalence test timeout");
  end
endmodule
