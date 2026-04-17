// TACCEL top-level: fetch -> decode -> issue -> execute.
//
// Phase 2 additions:
//   - DMA engine instantiated (LOAD/STORE via AXI4 master)
//   - SRAM Port A connected to DMA engine
//   - AXI write channels (AW/W/B) connected to DMA engine
//   - New ports: m_axi_aw_len, m_axi_aw_size, m_axi_aw_burst
//
// Phase 3 additions:
//   - Systolic controller instantiated
//   - sys_busy now driven by systolic controller (not stubbed)
//   - SRAM port arbitration between DMA and systolic controller
//
// Phase B DMA productionization:
//   - explicit AR grant tracking for fetch vs DMA
//   - burst-boundary fetch interleave between DMA read bursts
//   - R channel ownership based on the accepted AR request
//
// Phase C helper engine:
//   - blocking SRAM-local helper path for BUF_COPY / VADD / REQUANT
//   - helper owns SRAM ahead of DMA / systolic while active
//
// Stage D SFU engine:
//   - asynchronous SOFTMAX / LAYERNORM / GELU path
//   - serialized against helper / DMA / systolic at dispatch time
//   - SFU owns SRAM ahead of DMA / systolic while active
//
// This module is mostly glue:
//   - arbitrate shared AXI read access between fetch and DMA
//   - arbitrate shared SRAM ports between helper, DMA, and systolic
//   - fan out decoded instruction fields to the right execution unit
//   - collapse submodule faults into one architectural fault path

`ifndef TACCEL_TOP_SV
`define TACCEL_TOP_SV

`include "taccel_pkg.sv"

module taccel_top
  import taccel_pkg::*;
#(
  parameter int SYSTOLIC_ARCH_MODE = SYS_MODE_DEFAULT
)
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        done,
  output logic        fault,
  output logic [3:0]  fault_code,

  // --- AXI4 master (128-bit) ---
  // Read address channel
  output logic [AXI_ADDR_W-1:0] m_axi_ar_addr,
  output logic                   m_axi_ar_valid,
  output logic [7:0]             m_axi_ar_len,
  output logic [2:0]             m_axi_ar_size,
  output logic [1:0]             m_axi_ar_burst,
  input  logic                   m_axi_ar_ready,

  // Read data channel
  input  logic [AXI_DATA_W-1:0] m_axi_r_data,
  input  logic [1:0]             m_axi_r_resp,
  input  logic                   m_axi_r_valid,
  input  logic                   m_axi_r_last,
  output logic                   m_axi_r_ready,

  // Write address channel
  output logic [AXI_ADDR_W-1:0] m_axi_aw_addr,
  output logic [7:0]             m_axi_aw_len,
  output logic [2:0]             m_axi_aw_size,
  output logic [1:0]             m_axi_aw_burst,
  output logic                   m_axi_aw_valid,
  input  logic                   m_axi_aw_ready,

  // Write data channel
  output logic [AXI_DATA_W-1:0] m_axi_w_data,
  output logic [15:0]            m_axi_w_strb,
  output logic                   m_axi_w_valid,
  output logic                   m_axi_w_last,
  input  logic                   m_axi_w_ready,

  // Write response channel
  input  logic [1:0]             m_axi_b_resp,
  input  logic                   m_axi_b_valid,
  output logic                   m_axi_b_ready
);

  // =========================================================================
  // Internal wires
  // =========================================================================

  // --- Fetch unit AXI connections (before arbitration mux) ---
  logic [AXI_ADDR_W-1:0] fetch_ar_addr;
  logic                   fetch_ar_valid;
  logic                   fetch_ar_ready;
  logic                   fetch_r_valid;
  logic                   fetch_r_ready;

  // --- DMA engine AXI connections ---
  logic [AXI_ADDR_W-1:0] dma_ar_addr;
  logic [7:0]             dma_ar_len;
  logic                   dma_ar_valid;
  logic                   dma_ar_ready;
  logic                   dma_r_valid;
  logic                   dma_r_ready;
  logic                   dma_r_owner_q;
  logic                   rd_inflight_q;
  logic                   prefer_fetch_after_dma_q;
  logic                   select_dma_ar_w;
  logic                   select_fetch_ar_w;

  // AR arbitration: only one read request may be in flight at a time. DMA gets
  // default priority, but after each completed DMA burst we allow one pending
  // fetch read to slip in before the next DMA burst is launched.
  logic dma_rd_busy;

  always_comb begin
    select_dma_ar_w   = 1'b0;
    select_fetch_ar_w = 1'b0;

    if (!rd_inflight_q) begin
      if (dma_ar_valid && fetch_ar_valid) begin
        if (prefer_fetch_after_dma_q)
          select_fetch_ar_w = 1'b1;
        else
          select_dma_ar_w = 1'b1;
      end else if (dma_ar_valid) begin
        select_dma_ar_w = 1'b1;
      end else if (fetch_ar_valid) begin
        select_fetch_ar_w = 1'b1;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dma_r_owner_q            <= 1'b0;
      rd_inflight_q            <= 1'b0;
      prefer_fetch_after_dma_q <= 1'b0;
    end else begin
      if (m_axi_ar_valid && m_axi_ar_ready) begin
        dma_r_owner_q <= select_dma_ar_w;
        rd_inflight_q <= 1'b1;
        if (select_fetch_ar_w)
          prefer_fetch_after_dma_q <= 1'b0;
      end

      if (rd_inflight_q && m_axi_r_valid && m_axi_r_ready && m_axi_r_last) begin
        rd_inflight_q <= 1'b0;
        if (dma_r_owner_q)
          prefer_fetch_after_dma_q <= 1'b1;
      end
    end
  end

  assign m_axi_ar_addr  = select_dma_ar_w ? dma_ar_addr : fetch_ar_addr;
  assign m_axi_ar_len   = select_dma_ar_w ? dma_ar_len  : 8'h00;
  assign m_axi_ar_size  = 3'b100;   // 16 bytes/beat
  assign m_axi_ar_burst = 2'b01;    // INCR
  assign m_axi_ar_valid = select_dma_ar_w | select_fetch_ar_w;

  assign dma_ar_ready   = m_axi_ar_ready & ~rd_inflight_q &
                          (~fetch_ar_valid | ~prefer_fetch_after_dma_q);
  assign fetch_ar_ready = m_axi_ar_ready & ~rd_inflight_q &
                          (~dma_ar_valid | prefer_fetch_after_dma_q);

  // R channel routing follows the owner captured when the winning AR request is
  // accepted, which keeps fetch and DMA from inferring ownership heuristically.
  assign dma_r_valid    = m_axi_r_valid &  rd_inflight_q &  dma_r_owner_q;
  assign fetch_r_valid  = m_axi_r_valid &  rd_inflight_q & ~dma_r_owner_q;
  assign m_axi_r_ready  = rd_inflight_q ? (dma_r_owner_q ? dma_r_ready : fetch_r_ready)
                                        : 1'b0;

  // AXI write: fixed protocol fields (DMA always uses 16B/INCR)
  assign m_axi_aw_size  = 3'b100;
  assign m_axi_aw_burst = 2'b01;

  // --- Instruction pipeline ---
  logic [55:0]    pc;
  logic           fetch_req;
  logic           insn_valid_w;
  logic [63:0]    insn_data_w;
  logic [63:0]    insn_data_q;
  decoded_insn_t  insn;
  logic           fetch_fault_w;
  logic [3:0]     fetch_fault_code_w;

  // Register file control signals
  logic        scale_we;
  logic [3:0]  scale_waddr;
  logic [15:0] scale_wdata;
  logic        addr_lo_we, addr_hi_we;
  logic [1:0]  addr_wsel;
  logic [27:0] addr_imm28;
  logic        tile_we;
  logic [9:0]  tile_m_in, tile_n_in, tile_k_in;
  logic [9:0]  tile_m, tile_n, tile_k;
  logic        tile_valid;
  logic [15:0] scale_rdata0, scale_rdata1, scale_rdata2, scale_rdata3;
  logic [55:0] addr_rdata;
  logic [1:0]  helper_src1_buf_w, helper_src2_buf_w, helper_dst_buf_w;
  logic [15:0] helper_src1_off_w, helper_src2_off_w, helper_dst_off_w;

  // Dispatch signals
  logic dma_dispatch, sys_dispatch, sfu_dispatch, helper_dispatch;

  // Explicit intermediate: Icarus fails to evaluate complex expressions in port
  // connections (e.g. enum comparisons), so we pull is_store out to a named wire.
  logic dma_is_store;
  assign dma_is_store = (insn_data_q[63:59] == 5'(OP_STORE));
  // BUF_COPY uses the B-type field aliases instead of the R-type aliases used
  // by VADD / REQUANT, so remap the helper inputs here once.
  assign helper_src1_buf_w = (insn.opcode == OP_BUF_COPY) ? insn.b_src_buf : insn.src1_buf;
  assign helper_src1_off_w = (insn.opcode == OP_BUF_COPY) ? insn.b_src_off : insn.src1_off;
  assign helper_src2_buf_w = (insn.opcode == OP_BUF_COPY) ? 2'b00 : insn.src2_buf;
  assign helper_src2_off_w = (insn.opcode == OP_BUF_COPY) ? 16'h0 : insn.src2_off;
  assign helper_dst_buf_w  = (insn.opcode == OP_BUF_COPY) ? insn.b_dst_buf : insn.dst_buf;
  assign helper_dst_off_w  = (insn.opcode == OP_BUF_COPY) ? insn.b_dst_off : insn.dst_off;

  // DMA engine status
  logic       dma_busy;
  logic       dma_fault_w;
  logic [3:0] dma_fault_code_w;
  logic       dma_sram_fault_w;

  logic       ext_fault_w;
  logic [3:0] ext_fault_code_w;
  logic       helper_fault_w;
  logic [3:0] helper_fault_code_w;
  logic       sfu_fault_w;
  logic [3:0] sfu_fault_code_w;
  logic       sys_sram_fault_now;
  logic       sys_sram_fault_latched;

  logic sys_busy, sfu_busy, helper_busy;

  // -----------------------------------------------------------------------
  // Internal-only observability state used by the Verilator program runner.
  // These signals intentionally stay off the architectural interface.
  // -----------------------------------------------------------------------
  logic         obs_retire_pulse_w /* verilator public_flat_rd */;
  logic [55:0]  obs_retire_pc_w /* verilator public_flat_rd */;
  logic [4:0]   obs_retire_opcode_w /* verilator public_flat_rd */;
  logic         obs_ctrl_fault_pulse_w /* verilator public_flat_rd */;
  logic [3:0]   obs_ctrl_fault_code_w /* verilator public_flat_rd */;
  logic [55:0]  obs_ctrl_fault_pc_w /* verilator public_flat_rd */;
  logic [4:0]   obs_ctrl_fault_opcode_w /* verilator public_flat_rd */;
  logic         obs_sync_wait_dma_w /* verilator public_flat_rd */;
  logic         obs_sync_wait_sys_w /* verilator public_flat_rd */;
  logic         obs_sync_wait_sfu_w /* verilator public_flat_rd */;

  logic         obs_run_active_q /* verilator public_flat_rd */;
  logic [63:0]  obs_cycle_count_q /* verilator public_flat_rd */;
  logic [63:0]  obs_retired_insn_count_q /* verilator public_flat_rd */;
  logic [63:0]  obs_sync_wait_dma_cycles_q /* verilator public_flat_rd */;
  logic [63:0]  obs_sync_wait_sys_cycles_q /* verilator public_flat_rd */;
  logic [63:0]  obs_sync_wait_sfu_cycles_q /* verilator public_flat_rd */;
  logic [63:0]  obs_dma_burst_count_q /* verilator public_flat_rd */;
  logic [63:0]  obs_dma_beat_count_q /* verilator public_flat_rd */;
  logic [63:0]  obs_helper_busy_cycles_q /* verilator public_flat_rd */;
  logic [63:0]  obs_sfu_busy_cycles_q /* verilator public_flat_rd */;
  logic [63:0]  obs_sys_busy_cycles_q /* verilator public_flat_rd */;
  logic         obs_fault_valid_q /* verilator public_flat_rd */;
  logic [55:0]  obs_fault_pc_q /* verilator public_flat_rd */;
  logic [4:0]   obs_fault_opcode_q /* verilator public_flat_rd */;
  logic         obs_fault_opcode_valid_q /* verilator public_flat_rd */;
  logic [2:0]   obs_fault_source_q /* verilator public_flat_rd */;
  logic [3:0]   obs_fault_code_q /* verilator public_flat_rd */;
  logic         obs_forbidden_overlap_violation_q /* verilator public_flat_rd */;
  logic [55:0]  obs_dma_issue_pc_q /* verilator public_flat_rd */;
  logic [4:0]   obs_dma_issue_opcode_q /* verilator public_flat_rd */;
  logic [55:0]  obs_sys_issue_pc_q /* verilator public_flat_rd */;
  logic [4:0]   obs_sys_issue_opcode_q /* verilator public_flat_rd */;
  logic [55:0]  obs_helper_issue_pc_q /* verilator public_flat_rd */;
  logic [4:0]   obs_helper_issue_opcode_q /* verilator public_flat_rd */;
  logic [55:0]  obs_sfu_issue_pc_q /* verilator public_flat_rd */;
  logic [4:0]   obs_sfu_issue_opcode_q /* verilator public_flat_rd */;

  logic         obs_dma_burst_fire_w;
  logic         obs_dma_beat_fire_w;
  logic         obs_terminal_event_w;

  // SRAM Port A requests from helper / SFU / DMA / Systolic
  logic         helper_sram_a_en,  helper_sram_a_we;
  logic [1:0]   helper_sram_a_buf;
  logic [15:0]  helper_sram_a_row;
  logic [127:0] helper_sram_a_wdata, helper_sram_a_rdata;

  // SRAM Port B requests from helper / SFU / Systolic
  logic         helper_sram_b_en;
  logic [1:0]   helper_sram_b_buf;
  logic [15:0]  helper_sram_b_row;
  logic [127:0] helper_sram_b_rdata;

  logic         sfu_sram_a_en,  sfu_sram_a_we;
  logic [1:0]   sfu_sram_a_buf;
  logic [15:0]  sfu_sram_a_row;
  logic [127:0] sfu_sram_a_wdata;

  logic         sfu_sram_b_en;
  logic [1:0]   sfu_sram_b_buf;
  logic [15:0]  sfu_sram_b_row;
  logic [127:0] sfu_sram_b_rdata;

  logic         dma_sram_en,  dma_sram_we;
  logic [1:0]   dma_sram_buf;
  logic [15:0]  dma_sram_row;
  logic [127:0] dma_sram_wdata, dma_sram_rdata;

  logic         sys_sram_a_en,  sys_sram_a_we;
  logic [1:0]   sys_sram_a_buf;
  logic [15:0]  sys_sram_a_row;
  logic [127:0] sys_sram_a_wdata, sys_sram_a_rdata;

  // SRAM Port B request from Systolic
  logic         sys_sram_b_en;
  logic [1:0]   sys_sram_b_buf;
  logic [15:0]  sys_sram_b_row;
  logic [127:0] sys_sram_b_rdata;

  // Arbitrated SRAM wires
  logic         sram_a_en, sram_a_we;
  logic [1:0]   sram_a_buf;
  logic [15:0]  sram_a_row;
  logic [127:0] sram_a_wdata, sram_a_rdata;
  logic         sram_a_fault;
  logic         sram_b_en;
  logic [1:0]   sram_b_buf;
  logic [15:0]  sram_b_row;
  logic [127:0] sram_b_rdata;
  logic         sram_b_fault;

  // Helper gets first access, then SFU, then DMA, then systolic. Stage D keeps
  // enough serialization at control level that fixed priority is sufficient.
  assign sram_a_en    = helper_sram_a_en ? helper_sram_a_en
                      : sfu_sram_a_en    ? sfu_sram_a_en
                      : dma_sram_en      ? dma_sram_en
                                         : sys_sram_a_en;
  assign sram_a_we    = helper_sram_a_en ? helper_sram_a_we
                      : sfu_sram_a_en    ? sfu_sram_a_we
                      : dma_sram_en      ? dma_sram_we
                                         : sys_sram_a_we;
  assign sram_a_buf   = helper_sram_a_en ? helper_sram_a_buf
                      : sfu_sram_a_en    ? sfu_sram_a_buf
                      : dma_sram_en      ? dma_sram_buf
                                         : sys_sram_a_buf;
  assign sram_a_row   = helper_sram_a_en ? helper_sram_a_row
                      : sfu_sram_a_en    ? sfu_sram_a_row
                      : dma_sram_en      ? dma_sram_row
                                         : sys_sram_a_row;
  assign sram_a_wdata = helper_sram_a_en ? helper_sram_a_wdata
                      : sfu_sram_a_en    ? sfu_sram_a_wdata
                      : dma_sram_en      ? dma_sram_wdata
                                         : sys_sram_a_wdata;

  assign sram_b_en    = helper_sram_b_en ? helper_sram_b_en
                      : sfu_sram_b_en    ? sfu_sram_b_en
                                         : sys_sram_b_en;
  assign sram_b_buf   = helper_sram_b_en ? helper_sram_b_buf
                      : sfu_sram_b_en    ? sfu_sram_b_buf
                                         : sys_sram_b_buf;
  assign sram_b_row   = helper_sram_b_en ? helper_sram_b_row
                      : sfu_sram_b_en    ? sfu_sram_b_row
                                         : sys_sram_b_row;

  assign helper_sram_a_rdata = sram_a_rdata;
  assign dma_sram_rdata      = sram_a_rdata;
  assign sys_sram_a_rdata    = sram_a_rdata;
  assign helper_sram_b_rdata = sram_b_rdata;
  assign sfu_sram_b_rdata    = sram_b_rdata;
  assign sys_sram_b_rdata    = sram_b_rdata;
  assign dma_sram_fault_w = dma_sram_en & sram_a_fault;
  assign sys_sram_fault_now = (sys_sram_b_en & ~helper_sram_b_en & sram_b_fault) |
                              (sys_sram_a_en & ~helper_sram_a_en & ~sfu_sram_a_en & ~dma_sram_en & sram_a_fault);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      sys_sram_fault_latched <= 1'b0;
    else if (sys_sram_fault_now)
      sys_sram_fault_latched <= 1'b1;
  end

  // Merge all asynchronous or submodule-originated failures into the single
  // external fault input consumed by the control FSM.
  assign ext_fault_w = fetch_fault_w | dma_fault_w | helper_fault_w |
                       sfu_fault_w |
                       sys_sram_fault_now | sys_sram_fault_latched;

  always_comb begin
    if (fetch_fault_w)
      ext_fault_code_w = fetch_fault_code_w;
    else if (dma_fault_w)
      ext_fault_code_w = dma_fault_code_w;
    else if (helper_fault_w)
      ext_fault_code_w = helper_fault_code_w;
    else if (sfu_fault_w)
      ext_fault_code_w = sfu_fault_code_w;
    else if (sys_sram_fault_now || sys_sram_fault_latched)
      ext_fault_code_w = 4'(FAULT_SRAM_OOB);
    else
      ext_fault_code_w = 4'(FAULT_NONE);
  end

  assign obs_dma_burst_fire_w = (dma_ar_valid && dma_ar_ready) ||
                                (m_axi_aw_valid && m_axi_aw_ready);
  assign obs_dma_beat_fire_w  = (dma_r_valid && dma_r_ready) ||
                                (m_axi_w_valid && m_axi_w_ready);
  assign obs_terminal_event_w = obs_ctrl_fault_pulse_w |
                                fetch_fault_w |
                                dma_fault_w |
                                helper_fault_w |
                                sfu_fault_w |
                                sys_sram_fault_now |
                                sys_sram_fault_latched |
                                (obs_retire_pulse_w &&
                                 (obs_retire_opcode_w == 5'(OP_HALT)));

  // Keep run-level counters and latched fault context local to the RTL. The
  // native program runner reads these signals hierarchically after each run
  // without introducing any new architectural debug interface.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      obs_run_active_q                 <= 1'b0;
      obs_cycle_count_q                <= 64'h0;
      obs_retired_insn_count_q         <= 64'h0;
      obs_sync_wait_dma_cycles_q       <= 64'h0;
      obs_sync_wait_sys_cycles_q       <= 64'h0;
      obs_sync_wait_sfu_cycles_q       <= 64'h0;
      obs_dma_burst_count_q            <= 64'h0;
      obs_dma_beat_count_q             <= 64'h0;
      obs_helper_busy_cycles_q         <= 64'h0;
      obs_sfu_busy_cycles_q            <= 64'h0;
      obs_sys_busy_cycles_q            <= 64'h0;
      obs_fault_valid_q                <= 1'b0;
      obs_fault_pc_q                   <= 56'h0;
      obs_fault_opcode_q               <= 5'h0;
      obs_fault_opcode_valid_q         <= 1'b0;
      obs_fault_source_q               <= 3'(OBS_FAULT_SRC_NONE);
      obs_fault_code_q                 <= 4'(FAULT_NONE);
      obs_forbidden_overlap_violation_q <= 1'b0;
      obs_dma_issue_pc_q               <= 56'h0;
      obs_dma_issue_opcode_q           <= 5'h0;
      obs_sys_issue_pc_q               <= 56'h0;
      obs_sys_issue_opcode_q           <= 5'h0;
      obs_helper_issue_pc_q            <= 56'h0;
      obs_helper_issue_opcode_q        <= 5'h0;
      obs_sfu_issue_pc_q               <= 56'h0;
      obs_sfu_issue_opcode_q           <= 5'h0;
    end else if (start && !obs_run_active_q && !done && !fault) begin
      obs_run_active_q                 <= 1'b1;
      obs_cycle_count_q                <= 64'h0;
      obs_retired_insn_count_q         <= 64'h0;
      obs_sync_wait_dma_cycles_q       <= 64'h0;
      obs_sync_wait_sys_cycles_q       <= 64'h0;
      obs_sync_wait_sfu_cycles_q       <= 64'h0;
      obs_dma_burst_count_q            <= 64'h0;
      obs_dma_beat_count_q             <= 64'h0;
      obs_helper_busy_cycles_q         <= 64'h0;
      obs_sfu_busy_cycles_q            <= 64'h0;
      obs_sys_busy_cycles_q            <= 64'h0;
      obs_fault_valid_q                <= 1'b0;
      obs_fault_pc_q                   <= 56'h0;
      obs_fault_opcode_q               <= 5'h0;
      obs_fault_opcode_valid_q         <= 1'b0;
      obs_fault_source_q               <= 3'(OBS_FAULT_SRC_NONE);
      obs_fault_code_q                 <= 4'(FAULT_NONE);
      obs_forbidden_overlap_violation_q <= 1'b0;
      obs_dma_issue_pc_q               <= 56'h0;
      obs_dma_issue_opcode_q           <= 5'h0;
      obs_sys_issue_pc_q               <= 56'h0;
      obs_sys_issue_opcode_q           <= 5'h0;
      obs_helper_issue_pc_q            <= 56'h0;
      obs_helper_issue_opcode_q        <= 5'h0;
      obs_sfu_issue_pc_q               <= 56'h0;
      obs_sfu_issue_opcode_q           <= 5'h0;
    end else begin
      if (obs_run_active_q) begin
        obs_cycle_count_q <= obs_cycle_count_q + 64'd1;

        if (obs_sync_wait_dma_w)
          obs_sync_wait_dma_cycles_q <= obs_sync_wait_dma_cycles_q + 64'd1;
        if (obs_sync_wait_sys_w)
          obs_sync_wait_sys_cycles_q <= obs_sync_wait_sys_cycles_q + 64'd1;
        if (obs_sync_wait_sfu_w)
          obs_sync_wait_sfu_cycles_q <= obs_sync_wait_sfu_cycles_q + 64'd1;
        if (helper_busy)
          obs_helper_busy_cycles_q <= obs_helper_busy_cycles_q + 64'd1;
        if (sfu_busy)
          obs_sfu_busy_cycles_q <= obs_sfu_busy_cycles_q + 64'd1;
        if (sys_busy)
          obs_sys_busy_cycles_q <= obs_sys_busy_cycles_q + 64'd1;
      end

      if (obs_retire_pulse_w)
        obs_retired_insn_count_q <= obs_retired_insn_count_q + 64'd1;

      if (obs_dma_burst_fire_w)
        obs_dma_burst_count_q <= obs_dma_burst_count_q + 64'd1;

      if (obs_dma_beat_fire_w)
        obs_dma_beat_count_q <= obs_dma_beat_count_q + 64'd1;

      if (dma_dispatch) begin
        obs_dma_issue_pc_q     <= pc;
        obs_dma_issue_opcode_q <= insn.opcode;
      end

      if (sys_dispatch) begin
        obs_sys_issue_pc_q     <= pc;
        obs_sys_issue_opcode_q <= insn.opcode;
      end

      if (helper_dispatch) begin
        obs_helper_issue_pc_q     <= pc;
        obs_helper_issue_opcode_q <= insn.opcode;
      end

      if (sfu_dispatch) begin
        obs_sfu_issue_pc_q     <= pc;
        obs_sfu_issue_opcode_q <= insn.opcode;
      end

      if (helper_busy && (dma_busy || sys_busy || sfu_busy))
        obs_forbidden_overlap_violation_q <= 1'b1;
      if (sfu_busy && (dma_busy || sys_busy || helper_busy))
        obs_forbidden_overlap_violation_q <= 1'b1;

      if (!obs_fault_valid_q) begin
        if (obs_ctrl_fault_pulse_w) begin
          obs_fault_valid_q        <= 1'b1;
          obs_fault_pc_q           <= obs_ctrl_fault_pc_w;
          obs_fault_opcode_q       <= obs_ctrl_fault_opcode_w;
          obs_fault_opcode_valid_q <= 1'b1;
          obs_fault_source_q       <= 3'(OBS_FAULT_SRC_CONTROL);
          obs_fault_code_q         <= obs_ctrl_fault_code_w;
        end else if (fetch_fault_w) begin
          obs_fault_valid_q        <= 1'b1;
          obs_fault_pc_q           <= pc;
          obs_fault_opcode_q       <= 5'h0;
          obs_fault_opcode_valid_q <= 1'b0;
          obs_fault_source_q       <= 3'(OBS_FAULT_SRC_FETCH);
          obs_fault_code_q         <= fetch_fault_code_w;
        end else if (dma_fault_w) begin
          obs_fault_valid_q        <= 1'b1;
          obs_fault_pc_q           <= obs_dma_issue_pc_q;
          obs_fault_opcode_q       <= obs_dma_issue_opcode_q;
          obs_fault_opcode_valid_q <= 1'b1;
          obs_fault_source_q       <= 3'(OBS_FAULT_SRC_DMA);
          obs_fault_code_q         <= dma_fault_code_w;
        end else if (helper_fault_w) begin
          obs_fault_valid_q        <= 1'b1;
          obs_fault_pc_q           <= obs_helper_issue_pc_q;
          obs_fault_opcode_q       <= obs_helper_issue_opcode_q;
          obs_fault_opcode_valid_q <= 1'b1;
          obs_fault_source_q       <= 3'(OBS_FAULT_SRC_HELPER);
          obs_fault_code_q         <= helper_fault_code_w;
        end else if (sfu_fault_w) begin
          obs_fault_valid_q        <= 1'b1;
          obs_fault_pc_q           <= obs_sfu_issue_pc_q;
          obs_fault_opcode_q       <= obs_sfu_issue_opcode_q;
          obs_fault_opcode_valid_q <= 1'b1;
          obs_fault_source_q       <= 3'(OBS_FAULT_SRC_SFU);
          obs_fault_code_q         <= sfu_fault_code_w;
        end else if (sys_sram_fault_now || sys_sram_fault_latched) begin
          obs_fault_valid_q        <= 1'b1;
          obs_fault_pc_q           <= obs_sys_issue_pc_q;
          obs_fault_opcode_q       <= obs_sys_issue_opcode_q;
          obs_fault_opcode_valid_q <= 1'b1;
          obs_fault_source_q       <= 3'(OBS_FAULT_SRC_SRAM);
          obs_fault_code_q         <= 4'(FAULT_SRAM_OOB);
        end
      end

      if (obs_run_active_q && obs_terminal_event_w)
        obs_run_active_q <= 1'b0;
    end
  end

  // =========================================================================
  // Instruction register.
  // Fetch and decode are separated by one register so the control FSM sees a
  // stable decoded instruction for the whole ISSUE cycle.
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      insn_data_q <= 64'h0;
    else if (insn_valid_w)
      insn_data_q <= insn_data_w;
  end

  // =========================================================================
  // Submodule instantiation
  // =========================================================================

  fetch_unit u_fetch (
    .clk            (clk),
    .rst_n          (rst_n),
    .pc             (pc),
    .fetch_req      (fetch_req),
    .insn_valid     (insn_valid_w),
    .insn_data      (insn_data_w),
    .fetch_fault    (fetch_fault_w),
    .fetch_fault_code(fetch_fault_code_w),
    .m_axi_ar_addr  (fetch_ar_addr),
    .m_axi_ar_valid (fetch_ar_valid),
    /* verilator lint_off PINCONNECTEMPTY */
    .m_axi_ar_len   (),                // always 0; mux uses dma_ar_len / 0
    .m_axi_ar_size  (),                // always 3'b100; fixed in mux
    .m_axi_ar_burst (),                // always 2'b01; fixed in mux
    /* verilator lint_on PINCONNECTEMPTY */
    .m_axi_ar_ready (fetch_ar_ready),
    .m_axi_r_data   (m_axi_r_data),
    .m_axi_r_resp   (m_axi_r_resp),
    .m_axi_r_valid  (fetch_r_valid),
    .m_axi_r_last   (m_axi_r_last),
    .m_axi_r_ready  (fetch_r_ready)
  );

  decode_unit u_decode (
    .insn_data (insn_data_q),
    .insn      (insn)
  );

  control_unit u_ctrl (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (start),
    .pc             (pc),
    .fetch_req      (fetch_req),
    .insn_valid     (insn_valid_w),
    .insn           (insn),
    .scale_we       (scale_we),
    .scale_waddr    (scale_waddr),
    .scale_wdata    (scale_wdata),
    .addr_lo_we     (addr_lo_we),
    .addr_hi_we     (addr_hi_we),
    .addr_wsel      (addr_wsel),
    .addr_imm28     (addr_imm28),
    .tile_we        (tile_we),
    .tile_m_in      (tile_m_in),
    .tile_n_in      (tile_n_in),
    .tile_k_in      (tile_k_in),
    .tile_valid     (tile_valid),
    .dma_dispatch   (dma_dispatch),
    .sys_dispatch   (sys_dispatch),
    .sfu_dispatch   (sfu_dispatch),
    .helper_dispatch(helper_dispatch),
    .dma_busy       (dma_busy),
    .sys_busy       (sys_busy),
    .sfu_busy       (sfu_busy),
    .helper_busy    (helper_busy),
    .ext_fault      (ext_fault_w),
    .ext_fault_code (ext_fault_code_w),
    .done           (done),
    .fault          (fault),
    .fault_code     (fault_code),
    .obs_retire_pulse    (obs_retire_pulse_w),
    .obs_retire_pc       (obs_retire_pc_w),
    .obs_retire_opcode   (obs_retire_opcode_w),
    .obs_ctrl_fault_pulse(obs_ctrl_fault_pulse_w),
    .obs_ctrl_fault_code (obs_ctrl_fault_code_w),
    .obs_ctrl_fault_pc   (obs_ctrl_fault_pc_w),
    .obs_ctrl_fault_opcode(obs_ctrl_fault_opcode_w),
    .obs_sync_wait_dma   (obs_sync_wait_dma_w),
    .obs_sync_wait_sys   (obs_sync_wait_sys_w),
    .obs_sync_wait_sfu   (obs_sync_wait_sfu_w)
  );

  register_file u_regfile (
    .clk          (clk),
    .rst_n        (rst_n),
    .scale_we     (scale_we),
    .scale_waddr  (scale_waddr),
    .scale_wdata  (scale_wdata),
    .scale_raddr0 (insn.sreg),
    .scale_rdata0 (scale_rdata0),
    .scale_raddr1 (insn.sreg + 4'd1),
    .scale_rdata1 (scale_rdata1),
    .scale_raddr2 (insn.sreg + 4'd2),
    .scale_rdata2 (scale_rdata2),
    .scale_raddr3 (insn.sreg + 4'd3),
    .scale_rdata3 (scale_rdata3),
    .addr_lo_we   (addr_lo_we),
    .addr_hi_we   (addr_hi_we),
    .addr_wsel    (addr_wsel),
    .addr_imm28   (addr_imm28),
    .addr_rsel    (insn.m_addr_reg),
    .addr_rdata   (addr_rdata),
    .tile_we      (tile_we),
    .tile_m_in    (tile_m_in),
    .tile_n_in    (tile_n_in),
    .tile_k_in    (tile_k_in),
    .tile_m       (tile_m),
    .tile_n       (tile_n),
    .tile_k       (tile_k),
    .tile_valid   (tile_valid)
  );

  blocking_helper_engine u_helper (
    .clk            (clk),
    .rst_n          (rst_n),
    .dispatch       (helper_dispatch),
    .opcode         (insn.opcode),
    .src1_buf       (helper_src1_buf_w),
    .src1_off       (helper_src1_off_w),
    .src2_buf       (helper_src2_buf_w),
    .src2_off       (helper_src2_off_w),
    .dst_buf        (helper_dst_buf_w),
    .dst_off        (helper_dst_off_w),
    .sreg           (insn.sreg),
    .b_length       (insn.b_length),
    .b_src_rows     (insn.b_src_rows),
    .b_transpose    (insn.b_transpose),
    .tile_m         (tile_m),
    .tile_n         (tile_n),
    .scale0_data    (scale_rdata0),
    .scale1_data    (scale_rdata1),
    .helper_busy       (helper_busy),
    .helper_fault      (helper_fault_w),
    .helper_fault_code (helper_fault_code_w),
    .sram_a_en      (helper_sram_a_en),
    .sram_a_we      (helper_sram_a_we),
    .sram_a_buf     (helper_sram_a_buf),
    .sram_a_row     (helper_sram_a_row),
    .sram_a_wdata   (helper_sram_a_wdata),
    .sram_a_rdata   (helper_sram_a_rdata),
    .sram_a_fault   (helper_sram_a_en & sram_a_fault),
    .sram_b_en      (helper_sram_b_en),
    .sram_b_buf     (helper_sram_b_buf),
    .sram_b_row     (helper_sram_b_row),
    .sram_b_rdata   (helper_sram_b_rdata),
    .sram_b_fault   (helper_sram_b_en & sram_b_fault)
  );

  sfu_engine u_sfu (
    .clk            (clk),
    .rst_n          (rst_n),
    .dispatch       (sfu_dispatch),
    .opcode         (insn.opcode),
    .src1_buf       (insn.src1_buf),
    .src1_off       (insn.src1_off),
    .src2_buf       (insn.src2_buf),
    .src2_off       (insn.src2_off),
    .dst_buf        (insn.dst_buf),
    .dst_off        (insn.dst_off),
    .sreg           (insn.sreg),
    .tile_m         (tile_m),
    .tile_n         (tile_n),
    .tile_k         (tile_k),
    .scale0_data    (scale_rdata0),
    .scale1_data    (scale_rdata1),
    .scale2_data    (scale_rdata2),
    .scale3_data    (scale_rdata3),
    .sfu_busy       (sfu_busy),
    .sfu_fault      (sfu_fault_w),
    .sfu_fault_code (sfu_fault_code_w),
    .sram_a_en      (sfu_sram_a_en),
    .sram_a_we      (sfu_sram_a_we),
    .sram_a_buf     (sfu_sram_a_buf),
    .sram_a_row     (sfu_sram_a_row),
    .sram_a_wdata   (sfu_sram_a_wdata),
    .sram_a_fault   (sfu_sram_a_en & sram_a_fault),
    .sram_b_en      (sfu_sram_b_en),
    .sram_b_buf     (sfu_sram_b_buf),
    .sram_b_row     (sfu_sram_b_row),
    .sram_b_rdata   (sfu_sram_b_rdata),
    .sram_b_fault   (sfu_sram_b_en & sram_b_fault)
  );

  dma_engine u_dma (
    .clk             (clk),
    .rst_n           (rst_n),
    // Dispatch
    .dispatch        (dma_dispatch),
    .is_store        (dma_is_store),
    .buf_id          (insn.m_buf_id),
    .sram_off        (insn.m_sram_off),
    .xfer_len        (insn.m_xfer_len),
    .base_addr       (addr_rdata),
    .dram_off        (insn.m_dram_off),
    // Status
    .dma_busy        (dma_busy),
    .dma_rd_busy     (dma_rd_busy),
    .dma_fault       (dma_fault_w),
    .dma_fault_code  (dma_fault_code_w),
    // SRAM Port A
    .sram_en         (dma_sram_en),
    .sram_we         (dma_sram_we),
    .sram_buf        (dma_sram_buf),
    .sram_row        (dma_sram_row),
    .sram_wdata      (dma_sram_wdata),
    .sram_rdata      (dma_sram_rdata),
    .sram_fault      (dma_sram_fault_w),
    // AXI read
    .dma_ar_addr     (dma_ar_addr),
    .dma_ar_len      (dma_ar_len),
    .dma_ar_valid    (dma_ar_valid),
    .dma_ar_ready    (dma_ar_ready),
    .dma_r_data      (m_axi_r_data),
    .dma_r_resp      (m_axi_r_resp),
    .dma_r_valid     (dma_r_valid),
    .dma_r_last      (m_axi_r_last),
    .dma_r_ready     (dma_r_ready),
    // AXI write
    .dma_aw_addr     (m_axi_aw_addr),
    .dma_aw_len      (m_axi_aw_len),
    .dma_aw_valid    (m_axi_aw_valid),
    .dma_aw_ready    (m_axi_aw_ready),
    .dma_w_data      (m_axi_w_data),
    .dma_w_strb      (m_axi_w_strb),
    .dma_w_valid     (m_axi_w_valid),
    .dma_w_last      (m_axi_w_last),
    .dma_w_ready     (m_axi_w_ready),
    .dma_b_resp      (m_axi_b_resp),
    .dma_b_valid     (m_axi_b_valid),
    .dma_b_ready     (m_axi_b_ready)
  );

  systolic_controller #(
    .SYSTOLIC_ARCH_MODE(SYSTOLIC_ARCH_MODE)
  ) u_systolic (
    .clk             (clk),
    .rst_n           (rst_n),
    .dispatch        (sys_dispatch),
    .tile_m          (tile_m),
    .tile_n          (tile_n),
    .tile_k          (tile_k),
    .src1_buf        (insn.src1_buf),
    .src1_off        (insn.src1_off),
    .src2_buf        (insn.src2_buf),
    .src2_off        (insn.src2_off),
    .dst_buf         (insn.dst_buf),
    .dst_off         (insn.dst_off),
    .flags_accumulate(insn.flags),
    .sys_busy        (sys_busy),
    .sram_a_en       (sys_sram_a_en),
    .sram_a_we       (sys_sram_a_we),
    .sram_a_buf      (sys_sram_a_buf),
    .sram_a_row      (sys_sram_a_row),
    .sram_a_wdata    (sys_sram_a_wdata),
    .sram_a_rdata    (sys_sram_a_rdata),
    .sram_b_en       (sys_sram_b_en),
    .sram_b_buf      (sys_sram_b_buf),
    .sram_b_row      (sys_sram_b_row),
    .sram_b_rdata    (sys_sram_b_rdata)
  );

  sram_subsystem u_sram (
    .clk     (clk),
    .rst_n   (rst_n),
    // Port A: DMA/Systolic (arbitrated)
    .a_en    (sram_a_en),
    .a_we    (sram_a_we),
    .a_buf   (sram_a_buf),
    .a_row   (sram_a_row),
    .a_wdata (sram_a_wdata),
    .a_rdata (sram_a_rdata),
    .a_fault (sram_a_fault),
    // Port B: systolic source reads
    .b_en    (sram_b_en),
    .b_buf   (sram_b_buf),
    .b_row   (sram_b_row),
    .b_rdata (sram_b_rdata),
    .b_fault (sram_b_fault)
  );

endmodule

`endif // TACCEL_TOP_SV
