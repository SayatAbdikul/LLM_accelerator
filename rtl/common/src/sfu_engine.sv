// Special-function-unit engine for Stage D numerical parity.
//
// Supported operations:
//   - SOFTMAX   : INT8/INT32 input, INT8 output, row-wise across full logical N
//   - LAYERNORM : ABUF INT8 input, WBUF FP16 gamma/beta, INT8 output
//   - GELU      : ABUF INT8 or ACCUM INT32 input, INT8 output
//   - SOFTMAX_ATTNV : fused softmax(QK^T) @ V with INT8 output
//   - MASKED_SOFTMAX / MASKED_SOFTMAX_ATTNV : attention-mask variants
//
// Architectural contract:
//   - dispatched asynchronously through sfu_dispatch / sfu_busy
//   - serialized against DMA / helper / systolic at control level in Stage D
//   - faults propagate asynchronously through sfu_fault / sfu_fault_code
//
// Implementation note:
//   Stage D prioritizes functional parity with the software golden model over
//   synthesis-oriented microarchitecture. The engine therefore uses real-valued
//   intermediate storage plus explicit FP32 rounding helpers to preserve the
//   architectural "all SFU internal operations use FP32" contract under the
//   current simulator, which does not model shortreal arithmetic directly.

`ifndef SFU_ENGINE_SV
`define SFU_ENGINE_SV

`include "taccel_pkg.sv"
// Synthesizable fp32 primitives (fp32_add, fp32_to_fp16, fp16_to_fp32) are
// added to CONTROL_SV in the Makefile so they elaborate alongside this
// module. They're used by the SFU_SYNTH_MODE=1 op paths; in mode=0 they
// remain present but their outputs are never sampled (synth folds them).

module sfu_engine
  import taccel_pkg::*;
#(
  // Phase-2 migration toggle: 0 = behavioral real+DPI path (the cosim-pinned
  // default), 1 = synthesizable RTL path using the Phase-1 primitives. Set
  // per-op-at-a-time as ops are migrated; the gate is `SFU_SYNTH_MODE==1 &&
  // opcode_q==<op>`. Until all ops are migrated, mode=1 still falls back to
  // the DPI path for unmigrated opcodes — both paths coexist.
  parameter int SFU_SYNTH_MODE = 0
) (
  input  logic         clk,
  input  logic         rst_n,

  input  logic         dispatch,
  input  logic [4:0]   opcode,
  input  logic [1:0]   src1_buf,
  input  logic [15:0]  src1_off,
  input  logic [1:0]   src2_buf,
  input  logic [15:0]  src2_off,
  input  logic [1:0]   dst_buf,
  input  logic [15:0]  dst_off,
  input  logic [3:0]   sreg,
  input  logic [9:0]   tile_m,
  input  logic [9:0]   tile_n,
  input  logic [9:0]   tile_k,
  // m_exact (freeze §6 rev 2026-07-10): exact row count for the SFU row
  // loops; 0 = full tiles ((tile_m+1)*16, legacy behaviour).
  input  logic [11:0]  m_exact,
  input  logic         attn_valid,
  input  logic [11:0]  attn_query_row_base,
  input  logic [11:0]  attn_valid_kv_len,
  input  logic [1:0]   attn_mode,
  input  logic [15:0]  scale0_data,
  input  logic [15:0]  scale1_data,
  input  logic [15:0]  scale2_data,
  input  logic [15:0]  scale3_data,

  output logic         sfu_busy,
  output logic         sfu_fault,
  output logic [3:0]   sfu_fault_code,

  output logic         sram_a_en,
  output logic         sram_a_we,
  output logic [1:0]   sram_a_buf,
  output logic [15:0]  sram_a_row,
  output logic [127:0] sram_a_wdata,
  input  logic         sram_a_fault,

  output logic         sram_b_en,
  output logic [1:0]   sram_b_buf,
  output logic [15:0]  sram_b_row,
  input  logic [127:0] sram_b_rdata,
  input  logic         sram_b_fault,

  // --- Scale-register write-back (MAX_ABS_REDUCE_FP32 0x1F) ---
  output logic         sfu_scale_we,
  output logic [3:0]   sfu_scale_waddr,
  output logic [15:0]  sfu_scale_wdata
);

`include "sfu_dpi_helpers.svh"

  localparam int SFU_MAX_ROW_ELEMS = 1024;
  localparam real LN_EPS = 1.0e-6;
  // gen-2 LAYERNORM_FP32 (0x1A) eps — the GPT-2 / golden value (1e-5).
  // Distinct from the gen-1 INT8 LN_EPS (1e-6) above; do NOT reuse it.
  localparam real LN_FP32_EPS = 1.0e-5;

  // 6-bit enum: Phase-2 LN sub-FSM (5'd32..) requires the extra state slot
  // beyond the original 5'd0..5'd31 range.
  typedef enum logic [6:0] {
    F_IDLE          = 7'd0,
    F_LN_PARAM_REQ  = 7'd1,
    F_LN_PARAM_LATCH= 7'd2,
    F_ROW_I8_REQ    = 7'd3,
    F_ROW_I8_LATCH  = 7'd4,
    F_ROW_I32_REQ   = 7'd5,
    F_ROW_I32_LATCH = 7'd6,
    F_ROW_COMPUTE   = 7'd7,
    F_ROW_PACK      = 7'd8,
    F_ROW_WRITE     = 7'd9,
    F_GELU_I8_REQ   = 7'd10,
    F_GELU_I8_LATCH = 7'd11,
    F_GELU_I8_WRITE = 7'd12,
    F_GELU_I32_REQ  = 7'd13,
    F_GELU_I32_LATCH= 7'd14,
    F_GELU_I32_WRITE= 7'd15,
    F_ATTN_QKT_REQ  = 7'd16,
    F_ATTN_QKT_LATCH= 7'd17,
    F_ATTN_PREP     = 7'd18,
    F_ATTN_V_REQ    = 7'd19,
    F_ATTN_V_LATCH  = 7'd20,
    F_ATTN_WRITE    = 7'd21,
    F_FAULT         = 7'd22,
    // gen-2 FP32 shared datapath (0x19 VADD / 0x1A LN / 0x1B GELU).
    // FP16 storage (8 elems / 16-byte row), FP32 internal.
    F_G2_S1_REQ     = 7'd23,
    F_G2_S1_LATCH   = 7'd24,
    F_G2_S2_REQ     = 7'd25,
    F_G2_S2_LATCH   = 7'd26,
    F_G2_COMPUTE    = 7'd27,
    F_G2_PACK       = 7'd28,
    F_G2_WRITE      = 7'd29,
    F_G2_SCALE_WR   = 7'd30,  // 0x1F: 2-cycle scale-reg write-back
    // Phase-2 synth-mode iterator state (SFU_SYNTH_MODE=1): serializes the
    // 1024-parallel combinational compute loops into one element / cycle
    // through the shared synthesizable primitives. The op-code (opcode_q)
    // multiplexes which primitive's output is sampled.  Currently handles:
    //   0x19 OP_VADD_FP32, 0x17 OP_DEQUANT_ACCUM_FP32,
    //   0x18 OP_QUANT_FP32_INT8, 0x1E OP_DEQUANT_ACCUM_FP32_SCALED.
    F_G2_SYNTH_ITER = 7'd31,
    // Phase-2 LAYERNORM_FP32 (0x1A) synth sub-FSM. 3 reduction passes over
    // the row (sum -> mean; var -> denom; output) plus 2 single-cycle math
    // steps. State sequence:
    //   F_G2_LN_SUM      : iter; sum_acc_q += row_data_q[iter]
    //   F_G2_LN_MEAN     : 1 cycle; mean_q = sum_acc_q / n_elems_fp32; reset var_acc
    //   F_G2_LN_VAR      : iter; var_acc_q += (row[iter] - mean)^2
    //   F_G2_LN_DENOM    : 1 cycle; denom_q = sqrt(var_acc/n + LN_FP32_EPS)
    //   F_G2_LN_OUT      : iter; out_h_q[iter] = f2h((row-mean)/denom*gamma + beta)
    F_G2_LN_SUM     = 7'd32,
    F_G2_LN_MEAN    = 7'd33,
    F_G2_LN_VAR     = 7'd34,
    F_G2_LN_DENOM   = 7'd35,
    F_G2_LN_OUT     = 7'd36,
    // Phase-2 MASKED_SOFTMAX_FP32 (0x1D) synth sub-FSM. 3 passes:
    //   F_G2_SM_MAX     : iter; track row_max over visible elements
    //   F_G2_SM_EXPSUM  : iter; exp_sum += exp(row[i] - row_max) (visible only)
    //   F_G2_SM_OUT     : iter; out[i] = f2h(exp(row[i] - row_max) / exp_sum)
    //                     for visible elements (masked -> 16'h0).
    //   Banded — bounded by `fp32_exp` accuracy (Phase-3 minimax pending).
    F_G2_SM_MAX     = 7'd37,
    F_G2_SM_EXPSUM  = 7'd38,
    F_G2_SM_OUT     = 7'd39,
    // Phase-3.B gen-1 GELU synth-mode iterator (uses fp32_gelu_new tanh-poly
    // primitive as the synth approximation of gen-1 erf-GELU; the int8
    // quantization at out_bytes_q absorbs the tanh-vs-erf difference for
    // typical fixture inputs).
    F_GELU_SYNTH_I8_ITER  = 7'd40,
    F_GELU_SYNTH_I32_ITER = 7'd41,
    // Phase-4 (2026-05-28): split F_G2_LN_DENOM into two cycles so the
    // (var_acc/n + eps) and sqrt() ops live on separate clock periods.
    // The pre-stage latches ln_var_eps_q; F_G2_LN_DENOM reads it via u_ln_sqrt.
    F_G2_LN_DENOM_PRE = 7'd42,
    // Phase-4: split F_G2_LN_OUT — the 5-op fp32 chain
    //   (row - mean) -> /denom -> *gamma -> +beta -> f2h -> out_h_q
    // ran in one cycle (~190 ns). NORM latches ln_norm_q = (row-mean)/denom
    // for the current iter; the OUT cycle takes (norm*gamma + beta) -> f2h.
    F_G2_LN_OUT_NORM  = 7'd43,
    // Phase-5 (2026-05-28): further split — ln_diff_q latches (row - mean) in
    // F_G2_LN_OUT_DIFF so u_ln_norm's input is registered, removing the
    // serial fp32_add + fp32_div chain from the SFU critical path. The path
    // ln_mean_q -> ln_norm_q was the post-PNR worst path at 184 ns.
    F_G2_LN_OUT_DIFF  = 7'd44,
    // Phase-6 (2026-05-29): split F_G2_SM_OUT — sm_norm_q latches the
    // fp32_div(exp, exp_sum) result so the subsequent fp32_to_fp16 sits in
    // its own cycle. Post Phase-5 the worst path was sm_exp_sum_q ->
    // fp32_div -> fp32_to_fp16 -> out_h_q at 149 ns; this cut isolates
    // fp32_div from f2h.
    F_G2_SM_OUT_NORM  = 7'd45,
    // 2026-05-29 (new session): fp32_div_p2 pipelined divider (LATENCY=2)
    // replaces the combinational fp32_div in the 4 STA-binding sites
    // (ln_norm 180 ns, sm_div 157 ns, ln_var_norm 134 ns, ln_mean 122 ns).
    // Each div site gains wait state(s) so the FSM samples y 2 cycles after
    // the registered operands are presented. Moves the SFU fmax floor from
    // the divider down to fp32_sqrt (ln_denom, ~113 ns). See [[dma_floor]].
    F_G2_LN_MEAN_W      = 7'd46,  // ln_mean div stage2
    F_G2_LN_MEAN_S      = 7'd47,  // ln_mean div y valid -> sample ln_mean_q
    F_G2_LN_DENOM_PRE_W = 7'd48,  // ln_var_norm div stage2
    F_G2_LN_DENOM_PRE_S = 7'd49,  // div y valid -> +eps -> sample ln_var_eps_q
    F_G2_LN_OUT_W       = 7'd50,  // ln_norm div stage2 (NORM presented, OUT uses y)
    F_G2_SM_OUT_DIV     = 7'd51,  // sm div stage1 (exp registered in OUT_NORM)
    F_G2_SM_OUT_W       = 7'd52,  // sm div stage2 (OUT uses y)
    // 2026-05-29: fp32_sqrt_p2 pipelined sqrt (LATENCY=2) replaces the
    // combinational fp32_sqrt at the LN denom site — the new ~97 ns fmax
    // floor after the dividers were pipelined. F_G2_LN_DENOM presents the
    // registered ln_var_eps_q; sample ln_denom_q 2 cycles later. See [[dma_floor]].
    F_G2_LN_DENOM_W     = 7'd53,  // ln_denom sqrt stage2
    F_G2_LN_DENOM_S     = 7'd54,  // sqrt y valid -> sample ln_denom_q
    // 2026-05-29: fp32_div_p3 (3-stage, LATENCY=3) replaces fp32_div_p2 at the
    // 4 binding divider sites — the divider stage-1 was the ~88 ns synth floor
    // after the sqrt was pipelined. Each site gets ONE extra wait state vs the
    // LATENCY=2 integration so the FSM samples y 3 cycles after the registered
    // operands are presented. See [[dma_floor]].
    F_G2_LN_MEAN_W2     = 7'd55,  // ln_mean div 3rd stage
    F_G2_LN_DENOM_PRE_W2= 7'd56,  // ln_var_norm div 3rd stage
    F_G2_LN_OUT_W2      = 7'd57,  // ln_norm div 3rd stage
    F_G2_SM_OUT_W2      = 7'd58,  // sm_div 3rd stage
    // 2026-05-29: fp32_sqrt_p3 (3-stage, LATENCY=3) replaces fp32_sqrt_p2 at the
    // ln_denom site — the sqrt stage-2 (iters 7..0 + pack) was the ~57.8 ns
    // post-PNR fmax floor. One extra wait state vs the LATENCY=2 integration so
    // the FSM samples ln_denom_w 3 cycles after F_G2_LN_DENOM presents it.
    F_G2_LN_DENOM_W2    = 7'd59,  // ln_denom sqrt 3rd stage
    // 2026-05-30: fp32_div_p4 (4-stage, LATENCY=4) replaces fp32_div_p3 at the 4
    // divider sites — the div_p3 stage-2 (12-iter middle) was the ~53.8 ns
    // post-PNR floor (binding at u_ln_var_norm + u_sm_div). One MORE wait state
    // each (W3) so the FSM samples y 4 cycles after the operands are presented.
    F_G2_LN_MEAN_W3     = 7'd60,  // ln_mean div 4th stage
    F_G2_LN_DENOM_PRE_W3= 7'd61,  // ln_var_norm div 4th stage
    F_G2_LN_OUT_W3      = 7'd62,  // ln_norm div 4th stage
    F_G2_SM_OUT_W3      = 7'd63,  // sm_div 4th stage
    F_G2_LN_DENOM_W3    = 7'd64,  // ln_sqrt p4 4th stage (LATENCY=4)
    F_G2_LN_MEAN_W4     = 7'd65,  // ln_mean div_p6 5th stage (was div_p5, lever E)
    F_G2_LN_DENOM_PRE_W4= 7'd66,  // ln_var_norm div_p6 5th stage (was div_p5, lever E)
    F_G2_SM_OUT_W4      = 7'd67,  // sm_div 5th stage (retired serial drain; streaming now)
    F_G2_LN_DENOM_W4    = 7'd68,  // ln_sqrt p6 5th stage (LATENCY=6)
    F_G2_LN_DENOM_W5    = 7'd69,  // ln_sqrt p6 6th stage (LATENCY=6)
    F_G2_CW             = 7'd70,  // fused compute+write (FP16 elementwise ops)
    F_G2_QLC            = 7'd71,  // QUANT fused load+compute (FP16 in / INT8 out)
    F_G2_DQL            = 7'd72,  // DEQUANT fused int32-load+compute+write
    F_G2_VLC            = 7'd73,  // VADD fused src2-load+compute+write
    F_G2_GLC            = 7'd74,  // GELU fused load+compute+write (FP16 in / FP16 out)
    // 2026-07-12 (lever E): fp32_div_p6 (LATENCY=6) replaces div_p5 at the scalar
    // divider sites. ONE more wait state (W5) each so ln_mean / ln_var_norm sample
    // y 6 cycles after the operands are presented. (The streaming ln_norm/sm_div
    // drains re-tune via the collect threshold iter>=7, no new state.)
    F_G2_LN_MEAN_W5     = 7'd75,  // ln_mean div_p6 6th stage (LATENCY=6)
    F_G2_LN_DENOM_PRE_W5= 7'd76   // ln_var_norm div_p6 6th stage (LATENCY=6)
  } sfu_state_t;

  sfu_state_t state;

  logic [4:0]   opcode_q;
  logic [1:0]   src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0]  src1_off_q, src2_off_q, dst_off_q;
  logic [3:0]   sreg_q;
  logic [14:0]  m_rows_q;
  logic [10:0]  n_tiles_q;
  logic [10:0]  k_tiles_q;
  logic [12:0]  n_chunks_i32_q;
  logic [12:0]  k_chunks_i32_q;
  logic [15:0]  n_elems_q;
  logic [15:0]  k_elems_q;
  logic [15:0]  ln_gamma_rows_q;
  logic [15:0]  ln_param_rows_q;
  logic         attn_valid_q;
  logic [11:0] attn_query_row_base_q;
  logic [11:0] attn_valid_kv_len_q;
  logic [1:0]  attn_mode_q;
  logic [3:0]   fault_code_r;

  logic [14:0]  row_idx_q;
  logic [12:0]  read_idx_q;
  // Phase-2 synth-mode (SFU_SYNTH_MODE=1) per-element iterator. 11 bits ->
  // covers SFU_MAX_ROW_ELEMS=1024. Unused when SFU_SYNTH_MODE=0.
  logic [10:0]  iter_idx_q;
  // Phase-2 synth-mode LN (0x1A) reduction accumulators / cached results.
  // Stored as fp32 bit-patterns (the primitives' native I/O).
  logic [31:0]  ln_sum_acc_q;
  logic [31:0]  ln_var_acc_q;
  logic [31:0]  ln_mean_q;
  logic [31:0]  ln_denom_q;
  // Phase-4 (2026-05-28): pipeline cut between (var_acc/n + eps) and sqrt.
  // ln_var_eps_q is latched in F_G2_LN_DENOM_PRE, consumed by u_ln_sqrt next cycle.
  logic [31:0]  ln_var_eps_q;
  // Phase-5: latches (row - mean) in F_G2_LN_OUT_DIFF so the divider sees a
  // registered operand. (The Phase-4 ln_norm_q intermediate latch was removed
  // 2026-05-29: u_ln_norm is now fp32_div_p6 (lever E, 2026-07-12), whose output register replaces it
  // — F_G2_LN_OUT reads ln_norm_w directly.)
  logic [31:0]  ln_diff_q;
  // 2026-05-31: LN_OUT divider-drain software-pipelining. The fp32_div_p4
  // u_ln_norm is fully pipelined (accepts a dividend every cycle), but the old
  // DIFF/NORM/W/W2/W3/OUT loop fed one element then idled 4 cycles draining it
  // (6 cyc/elem). The pipelined F_G2_LN_OUT_DIFF runs one state: iter_idx_q is
  // the master/feed counter (presents row[iter]-mean to ln_diff_q each cycle);
  // ln_coll_q is the lagging collect pointer (= iter_idx_q - 7: one ln_diff_q
  // reg + 6 divider stages), indexing gamma/beta as the matching divider output
  // (ln_norm_w) emerges. ~6x on the OUT pass; bit-exact (same ops/operands/order,
  // same divider instance) and fmax-neutral (the feed and collect combinational
  // paths already existed as the DIFF and OUT critical paths; they now just run
  // concurrently, and don't chain).
  logic [10:0]  ln_coll_q;
  // 2026-05-31: LN_VAR accumulator pipelining (fmax lever). The variance pass
  // computed (row-mean)^2 AND accumulated it in ONE cycle: synth_a_bits ->
  // fp32_add(sub) -> fp32_mul(square) -> fp32_add(accumulate) -> ln_var_acc_q,
  // a ~42 ns chain = the post-div_p4 SFU PNR floor (= whole-chip clock). diff
  // and square are NOT loop-carried (only the final accumulate is), so register
  // the square here and accumulate it the next cycle: the binding path splits
  // into sub+mul (~28 ns, fed) and add (~14 ns, loop-carried). Software-
  // pipelined like F_G2_LN_OUT_DIFF — feed element i's square while accumulating
  // element i-1's — so throughput stays ~1 elem/cycle (+1 drain/row). Bit-exact
  // (same sub/mul/add, same operands, same accumulation order).
  logic [31:0]  ln_dsq_q;
  // 2026-07-21 (fmax phase 0b): feed register for the pipelined fp32_exp_p18.
  // Holds (row[iter_idx_q] - row_max) so the fp32_add subtract never chains
  // into exp's stage 1. Written every cycle of F_G2_SM_EXPSUM and
  // F_G2_SM_OUT_NORM; entries fed past the row bound are simply never
  // collected. Reset value is irrelevant to results (no collect can reach it).
  logic [31:0]  sm_diff_q;
  // 2026-05-29: registers exp(row-max) so the pipelined fp32_div_p6 sees a
  // registered dividend. 2026-07-21: it is now ALSO the exp_p18 collect
  // register — the sole source for both the EXPSUM accumulate and the OUT_NORM
  // divide — so exp's s18 output glue never chains into an fp32_add or into
  // div stage 1. Holds exp(element iter_idx_q - 20).
  logic [31:0]  sm_exp_q;
  // Phase-2 synth-mode SOFTMAX (0x1D) reduction state.
  logic [31:0]  sm_row_max_q;       // running fp32 row_max
  logic [31:0]  sm_exp_sum_q;       // running fp32 exp_sum
  logic         sm_have_vis_q;      // any visible element seen
  logic signed [15:0] sm_keep_through_q;
  // Lagging collect pointer, shared by BOTH softmax drain states (the element
  // emerging from the pipe this cycle; the feed pointer is iter_idx_q):
  //   F_G2_SM_EXPSUM   sm_coll_q = iter_idx_q - 20  (1 sm_diff_q + 18 exp_p18
  //                                                  + 1 sm_exp_q)
  //   F_G2_SM_OUT_NORM sm_coll_q = iter_idx_q - 26  (the above + 6 div_p6)
  // Feeds sm_visible_coll_w, the EXPSUM accumulate gate, and the out_h_q write
  // target, so every mask is evaluated against the DRAINING element rather than
  // the feed element. Analogous to ln_coll_q in the LN output pipeline.
  logic [10:0]  sm_coll_q;
  logic [10:0]  write_chunk_q;
  // Streamed-writeback lagging address pointer: the chunk index whose packed
  // data is currently held in row_write_q (write_chunk_q is the leading pack
  // pointer). Lets F_G2_WRITE / F_ROW_WRITE issue one SRAM write per cycle
  // while the pack mux registers the next chunk — 2 cyc/chunk -> 1, with the
  // row_write_q register boundary (hence SFU fmax) preserved.
  logic [10:0]  g2_wr_addr_q;
  // F_G2_CW (fused compute+write) "staged-chunk valid" flag: set the cycle
  // after a chunk is packed into row_write_q, gating the comb SRAM-A write so
  // the prime/drain cycles issue no spurious write. See F_G2_CW in
  // sfu_g2_compute.svh.
  logic         cw_have_q;
  // F_G2_QLC (QUANT fused load+compute): qlc_load_q = a chunk is still being
  // streamed in on port B; qlc_vis_q = count of FP16 chunks already captured &
  // visible in row_data_q (the compute pointer iter_idx_q may quantize any chunk
  // < qlc_vis_q). Lets the FP16 load and the 8-wide quant run concurrently.
  logic         qlc_load_q;
  logic [12:0]  qlc_vis_q;
  // F_G2_DQL (DEQUANT 0x17/0x1E fused int32-load+compute+write, mode-1): the
  // int32 src1 load (4 elem/cyc, half the 8-wide compute rate) is overlapped
  // with the 8-wide dequant compute + FP16 writeback in one pass. To do that the
  // scales/bias (src2) must already be in registers, so DEQUANT reorders src2
  // ahead of src1: dq_params_done_q sequences the two F_G2_S1_REQ visits (0 -> go
  // load params via S2; 1 -> prime int32 + enter F_G2_DQL). dq_load_q = int32
  // chunks still streaming on port B; dq_vis_q = count of int32 chunks captured &
  // visible in row_data_q (compute chunk iter ready when iter+8 <= dq_vis_q*4, or
  // the load has finished for the partial tail).
  logic         dq_params_done_q;
  logic         dq_load_q;
  logic [12:0]  dq_vis_q;
  // F_G2_VLC (VADD 0x19 fused src2-load+compute+write, mode-1): VADD adds two
  // FP16 tiles (src1 -> row_data_q, src2 -> attn_accum_q). src1 loads in the
  // normal F_G2_S1 pass; then F_G2_VLC streams src2 (port B, 8 elem/cyc, SAME
  // rate as the 8-wide add) WHILE the add+writeback run concurrently -> the
  // compute+write pass is hidden behind the src2 load (op cost ~2*g2_rows: the
  // two irreducible port-B operand loads, since both operands share port B).
  // vlc_load_q = src2 chunks still streaming; vlc_vis_q = count of src2 chunks
  // captured & visible in attn_accum_q (compute chunk iter ready when
  // iter>>3 < vlc_vis_q; row_data_q is already fully loaded).
  logic         vlc_load_q;
  logic [12:0]  vlc_vis_q;
  // F_G2_GLC (GELU 0x1B fused load+compute+write, mode-1): GELU is a single-
  // operand FP16-in/FP16-out elementwise op (like QUANT but FP16-out). Its
  // operand streams into row_data_q on port B (8 elem/cyc, SAME rate as the
  // 8-wide gelu compute) WHILE the compute + FP16 writeback run as concurrent
  // trailing tracks -> the compute+write pass is hidden behind the load (op
  // cost ~g2_rows: the single irreducible port-B operand load). glc_load_q =
  // FP16 chunks still streaming on port B; glc_vis_q = count of chunks captured
  // & visible in row_data_q (compute chunk iter ready when iter>>3 < glc_vis_q).
  // Load track == F_G2_QLC's; compute+write track == F_G2_VLC's (FP16 out).
  logic         glc_load_q;
  logic [12:0]  glc_vis_q;
  // 2026-07-21 (fmax phase 0d): the 8 GELU lane cores are the 33-stage
  // fp32_gelu_p33, so F_G2_GLC's compute track became a software pipeline —
  // iter_idx_q stays the FEED pointer and gelu_coll_q is the lagging COLLECT
  // pointer that indexes the out_h_q writeback. gelu_coll_q advances ONLY on
  // the pipe's own valid_out (synth_gelu_vo), never on a hand-counted offset,
  // so an arbitrary stall pattern on the feed side (a chunk not yet visible)
  // costs exactly its own bubbles and nothing more. The WRITE track keys off
  // gelu_coll_q for the same reason it used to key off iter_idx_q: a chunk is
  // packable once the pointer has moved PAST it, i.e. it landed in out_h_q on
  // an earlier cycle. Cross-row contamination is structurally absent — feed
  // stops at iter_idx_q >= n_elems_q so every slot behind the last real
  // element carries valid=0, and the row cannot exit until write_chunk_q has
  // drained every chunk, which cannot happen until collect has produced them.
  logic [10:0]  gelu_coll_q;
  logic         gelu_feed_en_w;
  // 2026-07-21 (fmax phase 0e): F_G2_DQL's compute track got the same
  // feed/collect split, for the same reason — its 0x1E chain (mul->mul->add
  // ->f2h) was one 86.59 ns cycle. dq_coll_q lags iter_idx_q by the pipe depth
  // 3; dq_vld_q is the depth-3 valid shift that says when a fed chunk has
  // emerged (the inline pipe has no valid_out of its own, unlike gelu_p33).
  // Both of F_G2_DQL's ops (0x17 and 0x1E) are delayed to the SAME depth so
  // the state has ONE write index rather than one per opcode.
  logic [10:0]  dq_coll_q;
  logic [2:0]   dq_vld_q;
  logic         dq_feed_en_w;
  // Streamed-load lagging capture pointer: the chunk index whose SRAM read
  // data is on the bus this cycle (read_idx_q is the leading issue pointer,
  // running one chunk ahead). Lets the F_G2_S1/S2 load loops issue one SRAM
  // read per cycle and capture the prior cycle's data — 2 cyc/chunk -> 1.
  // Used by every streamed src1/src2 load, including (2026-05-31) the
  // MAX_ABS_REDUCE running-max reduction (its mask mar_base_idx = ld_cap_q*8).
  logic [12:0]  ld_cap_q;
  logic [1:0]   gelu_part_q;
  logic [15:0]  attn_k_idx_q;

  logic [127:0] gelu_i8_row_q;
  logic [127:0] gelu_row0_q, gelu_row1_q, gelu_row2_q, gelu_row3_q;

  // Phase-3.D storage cascade (2026-05-21): persistent storage moved from
  // `real` to `logic [31:0]` (fp32 bit-pattern). DPI mode wraps writes with
  // `real_to_fp32_bits(...)` and reads with `fp32_bits_to_real(...)`; synth
  // mode reads/writes the bits directly. The helpers are lossless
  // (single-precision in `real` double).
  logic [31:0] scale0_q /* verilator public_flat_rd */, scale1_q /* verilator public_flat_rd */,
               scale2_q /* verilator public_flat_rd */, scale3_q /* verilator public_flat_rd */;
  logic [31:0] row_data_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [31:0] attn_accum_q [0:SFU_MAX_ROW_ELEMS-1];
  logic [31:0] gamma_q    [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [31:0] beta_q     [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [7:0] out_bytes_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  // gen-2: FP16 result bit-patterns + FP16-rows-per-logical-row (=2*n_tiles).
  logic [15:0] out_h_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [12:0] g2_rows_q;
  // 0x1F MAX_ABS_REDUCE_FP32: running global max|x| + 2-cycle write phase.
  logic [31:0] g2_maxabs_q;
  logic        g2_wr_phase_q;
  logic [31:0] attn_row_max_q;
  logic [31:0] attn_exp_sum_q;
  // R9 (2026-05-23): gen-1 LAYERNORM intermediate probes. Cell-cost only
  // useful when test_sfu.cpp `test_layernorm_replay_probe` runs (which is
  // env-gated on RTL_QKT_REPLAY_DIR); synth-check builds drop them so the
  // gate's cell count reflects the deployable design.
`ifdef SFU_DEBUG_LN
  logic [31:0] ln_debug_mean_q  /* verilator public_flat_rd */;
  logic [31:0] ln_debug_var_q   /* verilator public_flat_rd */;
  logic [31:0] ln_debug_denom_q /* verilator public_flat_rd */;
  logic [31:0] ln_debug_y_q     [0:15] /* verilator public_flat_rd */;
`endif

  logic [14:0] dispatch_m_rows_w;
  logic [10:0] dispatch_n_tiles_w;
  logic [10:0] dispatch_k_tiles_w;
  logic [12:0] dispatch_n_chunks_i32_w;
  logic [12:0] dispatch_k_chunks_i32_w;
  logic [15:0] dispatch_n_elems_w;
  logic [15:0] dispatch_k_elems_w;
  logic [15:0] dispatch_ln_gamma_rows_w;
  logic [15:0] dispatch_ln_param_rows_w;
  logic [15:0] dispatch_src1_rows_w;
  logic [15:0] dispatch_src2_rows_w;
  logic [15:0] dispatch_dst_rows_w;
  // 2026-05-23 Phase B: gen-1 dispatch flags (dispatch_softmax_*,
  // dispatch_layernorm_w, dispatch_gelu_*, dispatch_*softmax_attnv*)
  // stripped along with their consumers.
  logic        dispatch_attn_context_bad_w;
  logic        dispatch_unsupported_w;
  logic        dispatch_sram_oob_w;

  logic [31:0] dispatch_src1_need_rows_w;
  logic [31:0] dispatch_src2_need_rows_w;
  logic [31:0] dispatch_dst_need_rows_w;
  logic [15:0] dispatch_attn_key_cols_w;

  logic [31:0] row_i8_addr_w;
  logic [31:0] row_i32_addr_w;
  logic [31:0] row_dst_addr_w;
  logic [31:0] ln_param_addr_w;
  logic [31:0] gelu_i8_addr_w;
  logic [31:0] gelu_acc_addr_w;
  logic [31:0] gelu_dst_addr_w;
  logic [31:0] attn_qkt_addr_w;
  logic [31:0] attn_v_addr_w;

  logic [127:0] row_write_data_w;
  logic [127:0] row_write_q;
  logic [127:0] gelu_i8_write_data_w;
  logic [127:0] gelu_i32_write_data_w;
  logic [127:0] attn_write_data_w;
  logic [127:0] g2_write_data_w;

  function automatic logic [15:0] buf_rows(input logic [1:0] bid);
    begin
      case (bid)
        BUF_ABUF:  buf_rows = 16'(ABUF_ROWS);
        BUF_WBUF:  buf_rows = 16'(WBUF_ROWS);
        BUF_ACCUM: buf_rows = 16'(ACCUM_ROWS);
        default:   buf_rows = 16'h0;
      endcase
    end
  endfunction


  function automatic logic signed [7:0] get_i8(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_i8 = row[(idx * 8) +: 8];
    end
  endfunction

  function automatic logic signed [31:0] get_i32(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_i32 = row[(idx * 32) +: 32];
    end
  endfunction

  function automatic logic [15:0] get_u16(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_u16 = row[(idx * 16) +: 16];
    end
  endfunction


  function automatic logic attn_visible(
    input logic [14:0] row_idx,
    input integer      col_idx
  );
    integer abs_query_row;
    begin
      abs_query_row = integer'(attn_query_row_base_q) + integer'(row_idx);
      attn_visible = 1'b1;
      if (attn_mode_q[1])
        attn_visible = attn_visible && (col_idx <= abs_query_row);
      if (attn_mode_q[0])
        attn_visible = attn_visible && (col_idx < integer'(attn_valid_kv_len_q));
    end
  endfunction

  // m_exact != 0 overrides the tile-quantized row count. Dispatch-stage
  // logic only (latched into m_rows_q at op dispatch) — no change to the
  // per-row loop compare paths or any fp32 primitive.
  assign dispatch_m_rows_w        = (m_exact != 12'd0)
                                    ? {3'h0, m_exact}
                                    : (({5'h0, tile_m} + 15'd1) << 4);
  assign dispatch_n_tiles_w       = {1'b0, tile_n} + 11'd1;
  assign dispatch_k_tiles_w       = {1'b0, tile_k} + 11'd1;
  assign dispatch_n_chunks_i32_w  = dispatch_n_tiles_w << 2;
  assign dispatch_k_chunks_i32_w  = dispatch_k_tiles_w << 2;
  assign dispatch_n_elems_w       = {1'b0, dispatch_n_tiles_w, 4'h0};
  assign dispatch_k_elems_w       = {1'b0, dispatch_k_tiles_w, 4'h0};
  assign dispatch_ln_gamma_rows_w = ({5'h0, dispatch_n_tiles_w}) << 1;
  assign dispatch_ln_param_rows_w = ({5'h0, dispatch_n_tiles_w}) << 2;
  assign dispatch_src1_rows_w     = buf_rows(src1_buf);
  assign dispatch_src2_rows_w     = buf_rows(src2_buf);
  assign dispatch_dst_rows_w      = buf_rows(dst_buf);

  // 2026-05-23 Phase B: gen-1 dispatch flag assigns stripped along with
  // their consumers. attn_key_cols simplifies — only OP_MASKED_SOFTMAX_FP32
  // (the `else` branch of the old ternary) remains as a consumer.
  assign dispatch_attn_key_cols_w = dispatch_n_elems_w;

  // gen-2 FP32 shared datapath detection (FP16 storage, ABUF I/O).
  logic        dispatch_g2_vadd_w;
  logic        dispatch_g2_ln_w;
  logic        dispatch_g2_gelu_w;
  logic [12:0] dispatch_g2_rows_w;   // FP16 rows per logical row = 2*n_tiles
  assign dispatch_g2_rows_w = {1'b0, dispatch_n_tiles_w} + {1'b0, dispatch_n_tiles_w};
  assign dispatch_g2_vadd_w = (opcode == OP_VADD_FP32) &&
                              (src1_buf == BUF_ABUF) &&
                              (src2_buf == BUF_ABUF) &&
                              (dst_buf  == BUF_ABUF);
  assign dispatch_g2_ln_w   = (opcode == OP_LAYERNORM_FP32) &&
                              (src1_buf == BUF_ABUF) &&
                              (src2_buf == BUF_WBUF) &&
                              (dst_buf  == BUF_ABUF);
  assign dispatch_g2_gelu_w = (opcode == OP_GELU_FP32) &&
                              (src1_buf == BUF_ABUF) &&
                              (dst_buf  == BUF_ABUF);
  logic dispatch_g2_dq_w;   // 0x17 DEQUANT_ACCUM_FP32
  logic dispatch_g2_q_w;    // 0x18 QUANT_FP32_INT8
  assign dispatch_g2_dq_w = (opcode == OP_DEQUANT_ACCUM_FP32) &&
                            (src1_buf == BUF_ACCUM) &&
                            (src2_buf == BUF_WBUF) &&
                            (dst_buf  == BUF_ABUF);
  assign dispatch_g2_q_w  = (opcode == OP_QUANT_FP32_INT8) &&
                            (src1_buf == BUF_ABUF) &&
                            (dst_buf  == BUF_ABUF);
  logic dispatch_g2_ms_w;   // 0x1D MASKED_SOFTMAX_FP32
  assign dispatch_g2_ms_w = (opcode == OP_MASKED_SOFTMAX_FP32) &&
                            (src1_buf == BUF_ABUF) &&
                            (dst_buf  == BUF_ABUF);
  logic dispatch_g2_ds_w;   // 0x1E DEQUANT_ACCUM_FP32_SCALED
  logic dispatch_g2_mar_w;  // 0x1F MAX_ABS_REDUCE_FP32
  assign dispatch_g2_ds_w = (opcode == OP_DEQUANT_ACCUM_FP32_SCALED) &&
                            (src1_buf == BUF_ACCUM) &&
                            (src2_buf == BUF_WBUF) &&
                            (dst_buf  == BUF_ABUF);
  assign dispatch_g2_mar_w = (opcode == OP_MAX_ABS_REDUCE_FP32) &&
                             (src1_buf == BUF_ABUF) &&
                             (sreg <= 4'd14);   // sreg+1 must be valid

  // GELU pipeline feed strobe (fmax phase 0d). This is the SINGLE definition
  // of "a chunk is entering the gelu pipe this cycle": F_G2_GLC's compute
  // track below is written as `if (gelu_feed_en_w)`, so the FSM's pointer
  // advance and the pipe's valid_in cannot drift apart by construction. The
  // fault guards mirror F_G2_GLC's leading branches — feeding a doomed row
  // would be harmless (it is never collected), but a feed condition that is
  // "the same except for the corners" is exactly how these two ever diverge.
  assign gelu_feed_en_w =
      (state == F_G2_GLC) &&
      !(cw_have_q  && sram_a_fault) &&
      !(glc_load_q && sram_b_fault) &&
      (write_chunk_q < g2_rows_q[10:0]) &&
      ({5'h0, iter_idx_q[10:3]} < glc_vis_q) &&
      ({5'h0, iter_idx_q} < n_elems_q);

  // DEQUANT (0x17/0x1E) pipeline feed strobe — the single definition of "a
  // chunk enters the DQL pipe this cycle", mirroring gelu_feed_en_w. F_G2_DQL's
  // compute track is written as `if (dq_feed_en_w)` so the pointer advance and
  // the valid chain cannot drift apart.
  assign dq_feed_en_w =
      (state == F_G2_DQL) &&
      !(cw_have_q && sram_a_fault) &&
      !(dq_load_q && sram_b_fault) &&
      (write_chunk_q < g2_rows_q[10:0]) &&
      ({5'h0, iter_idx_q} < n_elems_q) &&
      ((({5'h0, iter_idx_q} + 16'd8) <= {1'b0, dq_vis_q, 2'b00}) || !dq_load_q);

  // Depth-3 valid chain for the DQL pipe. Free-running like the datapath
  // registers it tracks; reset only clears validity, never data.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) dq_vld_q <= 3'h0;
    else        dq_vld_q <= {dq_vld_q[1:0], dq_feed_en_w};
  end

`include "sfu_synth_datapath.svh"
  // ===================================================================

  // ===================================================================

  always_comb begin
    dispatch_attn_context_bad_w = 1'b0;
    // 2026-05-23 Phase B: only gen-2 OP_MASKED_SOFTMAX_FP32 remains in the
    // attn-context-bad check; gen-1 OP_MASKED_SOFTMAX/OP_MASKED_SOFTMAX_ATTNV
    // are now illegal at decode and never reach SFU dispatch.
    if (opcode == OP_MASKED_SOFTMAX_FP32) begin
      dispatch_attn_context_bad_w = !attn_valid ||
                                    (attn_mode == 2'b00) ||
                                    (attn_valid_kv_len == 12'h000);
      if (!dispatch_attn_context_bad_w) begin
        if (attn_mode == 2'b10)
          dispatch_attn_context_bad_w =
              ({4'h0, dispatch_attn_key_cols_w} != {8'h00, attn_valid_kv_len});
        else if (attn_mode[0])
          dispatch_attn_context_bad_w =
              ({4'h0, dispatch_attn_key_cols_w} < {8'h00, attn_valid_kv_len});
      end
    end
  end

  always_comb begin
    dispatch_unsupported_w = 1'b0;
    dispatch_sram_oob_w    = 1'b0;
    dispatch_src1_need_rows_w = 32'd0;
    dispatch_src2_need_rows_w = 32'd0;
    dispatch_dst_need_rows_w  = 32'd0;

    // 2026-05-23 Phase B: gen-1 SFU dispatch_unsupported case arms
    // (OP_SOFTMAX, OP_MASKED_SOFTMAX, OP_LAYERNORM, OP_GELU, OP_SOFTMAX_ATTNV,
    // OP_MASKED_SOFTMAX_ATTNV) stripped — those opcodes are now illegal at
    // decode and never reach SFU dispatch.
    case (opcode)
      OP_VADD_FP32: begin
        if (!dispatch_g2_vadd_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_src2_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_LAYERNORM_FP32: begin
        if (!dispatch_g2_ln_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_GELU_FP32: begin
        if (!dispatch_g2_gelu_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_DEQUANT_ACCUM_FP32: begin
        if (!dispatch_g2_dq_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        dispatch_src2_need_rows_w = {19'h0, dispatch_g2_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_QUANT_FP32_INT8: begin
        if (!dispatch_g2_q_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_MASKED_SOFTMAX_FP32: begin
        if (!dispatch_g2_ms_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_DEQUANT_ACCUM_FP32_SCALED: begin
        if (!dispatch_g2_ds_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_g2_rows_w;
      end

      OP_MAX_ABS_REDUCE_FP32: begin
        if (!dispatch_g2_mar_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;
        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_g2_rows_w;
        // no src2, no tile dst (writes 2 scale regs).
      end

      default:
        dispatch_unsupported_w = 1'b1;
    endcase

    dispatch_sram_oob_w =
        ({16'h0, src1_off} + dispatch_src1_need_rows_w > {16'h0, dispatch_src1_rows_w}) ||
        ({16'h0, src2_off} + dispatch_src2_need_rows_w > {16'h0, dispatch_src2_rows_w}) ||
        ({16'h0, dst_off}  + dispatch_dst_need_rows_w  > {16'h0, dispatch_dst_rows_w});
  end

  assign row_i8_addr_w  = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {19'h0, read_idx_q};
  assign row_i32_addr_w = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                          {19'h0, read_idx_q};
  assign row_dst_addr_w = {16'h0, dst_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {21'h0, g2_wr_addr_q};
  assign ln_param_addr_w = {16'h0, src2_off_q} + {19'h0, read_idx_q};
  assign gelu_i8_addr_w = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {21'h0, write_chunk_q};
  assign gelu_acc_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                           ({21'h0, write_chunk_q} << 2) +
                           {30'h0, gelu_part_q};
  assign gelu_dst_addr_w = {16'h0, dst_off_q} +
                           ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                           {21'h0, write_chunk_q};
  assign attn_qkt_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, k_chunks_i32_q}) +
                           {19'h0, read_idx_q};
  assign attn_v_addr_w = {16'h0, src2_off_q} +
                         ({16'h0, attn_k_idx_q} * {21'h0, n_tiles_q}) +
                         {19'h0, read_idx_q};

  // gen-2 FP16-tile addressing (8 elems / 16-byte row, g2_rows_q per row).
  logic [31:0] g2_s1_addr_w;
  logic [31:0] g2_s2_addr_w;
  logic [31:0] g2_lnp_addr_w;
  logic [31:0] g2_dst_addr_w;
  assign g2_s1_addr_w  = {16'h0, src1_off_q} +
                         ({17'h0, row_idx_q} * {19'h0, g2_rows_q}) +
                         {19'h0, read_idx_q};
  assign g2_s2_addr_w  = {16'h0, src2_off_q} +
                         ({17'h0, row_idx_q} * {19'h0, g2_rows_q}) +
                         {19'h0, read_idx_q};
  assign g2_lnp_addr_w = {16'h0, src2_off_q} + {19'h0, read_idx_q};
  assign g2_dst_addr_w = {16'h0, dst_off_q} +
                         ({17'h0, row_idx_q} * {19'h0, g2_rows_q}) +
                         {21'h0, g2_wr_addr_q};

  always_comb begin
    row_write_data_w = 128'h0;
    gelu_i8_write_data_w = 128'h0;
    gelu_i32_write_data_w = 128'h0;
    attn_write_data_w = 128'h0;
    g2_write_data_w = 128'h0;

    // gen-2: pack 8 FP16 results (16-bit each) for the current write chunk.
    for (int g2l = 0; g2l < 8; g2l++) begin
      int g2idx;
      g2idx = integer'(write_chunk_q) * 8 + g2l;
      if (g2idx < integer'(n_elems_q))
        g2_write_data_w[(g2l * 16) +: 16] = out_h_q[g2idx];
    end

    for (int lane = 0; lane < 16; lane++) begin
      int idx;
      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        row_write_data_w[(lane * 8) +: 8] = out_bytes_q[idx];
    end
`ifndef SFU_SYNTH_NO_DPI
    for (int lane = 0; lane < 16; lane++) begin
      int idx;
      real x_r;
      x_r = sfu_fp32_mul(real'(get_i8(gelu_i8_row_q, lane)), fp32_bits_to_real(scale0_q));
      gelu_i8_write_data_w[(lane * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));

      if (lane < 4) begin
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row0_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[(lane * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row1_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[((lane + 4) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row2_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[((lane + 8) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
        x_r = sfu_fp32_mul(real'(get_i32(gelu_row3_q, lane)), fp32_bits_to_real(scale0_q));
        gelu_i32_write_data_w[((lane + 12) * 8) +: 8] = quantize_to_i8(gelu_real(x_r), fp32_bits_to_real(scale1_q));
      end

      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        attn_write_data_w[(lane * 8) +: 8] = quantize_to_i8(fp32_bits_to_real(attn_accum_q[idx]), fp32_bits_to_real(scale2_q));
    end
`endif
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= F_IDLE;
      opcode_q       <= 5'h0;
      src1_buf_q     <= 2'b0;
      src2_buf_q     <= 2'b0;
      dst_buf_q      <= 2'b0;
      src1_off_q     <= 16'h0;
      src2_off_q     <= 16'h0;
      dst_off_q      <= 16'h0;
      sreg_q         <= 4'h0;
      m_rows_q       <= 15'h0;
      n_tiles_q      <= 11'h0;
      k_tiles_q      <= 11'h0;
      n_chunks_i32_q <= 13'h0;
      k_chunks_i32_q <= 13'h0;
      n_elems_q      <= 16'h0;
      k_elems_q      <= 16'h0;
      ln_gamma_rows_q<= 16'h0;
      ln_param_rows_q<= 16'h0;
      attn_valid_q   <= 1'b0;
      attn_query_row_base_q <= 12'h0;
      attn_valid_kv_len_q   <= 12'h0;
      attn_mode_q           <= 2'b00;
      fault_code_r   <= 4'(FAULT_NONE);
      row_idx_q      <= 15'h0;
      read_idx_q     <= 13'h0;
      iter_idx_q     <= 11'h0;
      ln_sum_acc_q   <= 32'h0;
      ln_var_acc_q   <= 32'h0;
      ln_mean_q      <= 32'h0;
      ln_denom_q     <= 32'h0;
      ln_var_eps_q   <= 32'h0;
      ln_diff_q      <= 32'h0;
      ln_coll_q      <= 11'h0;
      ln_dsq_q       <= 32'h0;
      sm_diff_q      <= 32'h0;
      sm_exp_q       <= 32'h0;
      sm_row_max_q   <= 32'h0;
      sm_exp_sum_q   <= 32'h0;
      sm_have_vis_q  <= 1'b0;
      sm_keep_through_q <= 16'sh0;
      sm_coll_q      <= 11'h0;
      write_chunk_q  <= 11'h0;
      g2_wr_addr_q   <= 11'h0;
      cw_have_q      <= 1'b0;
      qlc_load_q     <= 1'b0;
      qlc_vis_q      <= 13'h0;
      dq_params_done_q <= 1'b0;
      dq_load_q      <= 1'b0;
      dq_vis_q       <= 13'h0;
      dq_coll_q      <= 11'h0;
      vlc_load_q     <= 1'b0;
      vlc_vis_q      <= 13'h0;
      glc_load_q     <= 1'b0;
      glc_vis_q      <= 13'h0;
      gelu_coll_q    <= 11'h0;
      ld_cap_q       <= 13'h0;
      gelu_part_q    <= 2'h0;
      attn_k_idx_q   <= 16'h0;
      g2_rows_q      <= 13'h0;
      g2_maxabs_q    <= 0.0;
      g2_wr_phase_q  <= 1'b0;
      gelu_i8_row_q  <= 128'h0;
      gelu_row0_q    <= 128'h0;
      gelu_row1_q    <= 128'h0;
      gelu_row2_q    <= 128'h0;
      gelu_row3_q    <= 128'h0;
      row_write_q    <= 128'h0;
      scale0_q       <= 0.0;
      scale1_q       <= 0.0;
      scale2_q       <= 0.0;
      scale3_q       <= 0.0;
      attn_row_max_q <= 0.0;
      attn_exp_sum_q <= 0.0;
`ifdef SFU_DEBUG_LN
      ln_debug_mean_q <= 0.0;
      ln_debug_var_q <= 0.0;
      ln_debug_denom_q <= 0.0;
`endif
      for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
        row_data_q[i] <= 0.0;
        attn_accum_q[i] <= 0.0;
        gamma_q[i]    <= 0.0;
        beta_q[i]     <= 0.0;
        out_bytes_q[i] <= 8'h00;
        out_h_q[i]    <= 16'h0;
      end
`ifdef SFU_DEBUG_LN
      for (int i = 0; i < 16; i++)
        ln_debug_y_q[i] <= 0.0;
`endif
    end else begin
      case (state)
        F_IDLE: begin
          if (dispatch) begin
            opcode_q        <= opcode;
            src1_buf_q      <= src1_buf;
            src2_buf_q      <= src2_buf;
            dst_buf_q       <= dst_buf;
            src1_off_q      <= src1_off;
            src2_off_q      <= src2_off;
            dst_off_q       <= dst_off;
            sreg_q          <= sreg;
            m_rows_q        <= dispatch_m_rows_w;
            n_tiles_q       <= dispatch_n_tiles_w;
            k_tiles_q       <= dispatch_k_tiles_w;
            n_chunks_i32_q  <= dispatch_n_chunks_i32_w;
            k_chunks_i32_q  <= dispatch_k_chunks_i32_w;
            n_elems_q       <= dispatch_n_elems_w;
            k_elems_q       <= dispatch_k_elems_w;
            ln_gamma_rows_q <= dispatch_ln_gamma_rows_w;
            ln_param_rows_q <= dispatch_ln_param_rows_w;
            g2_rows_q       <= dispatch_g2_rows_w;
            g2_maxabs_q     <= 0.0;
            g2_wr_phase_q   <= 1'b0;
            attn_valid_q    <= attn_valid;
            attn_query_row_base_q <= attn_query_row_base;
            attn_valid_kv_len_q   <= attn_valid_kv_len;
            attn_mode_q           <= attn_mode;
`ifndef SFU_SYNTH_NO_DPI
            scale0_q        <= real_to_fp32_bits(fp16_to_real(scale0_data));
            scale1_q        <= real_to_fp32_bits(fp16_to_real(scale1_data));
            scale2_q        <= real_to_fp32_bits(fp16_to_real(scale2_data));
            scale3_q        <= real_to_fp32_bits(fp16_to_real(scale3_data));
`else
            scale0_q        <= {16'h0, scale0_data};
            scale1_q        <= {16'h0, scale1_data};
            scale2_q        <= {16'h0, scale2_data};
            scale3_q        <= {16'h0, scale3_data};
`endif
`ifdef SFU_DEBUG_LN
            ln_debug_mean_q <= 32'h0;
            ln_debug_var_q <= 32'h0;
            ln_debug_denom_q <= 32'h0;
            for (int i = 0; i < 16; i++)
              ln_debug_y_q[i] <= 0.0;
`endif
            row_idx_q       <= 15'h0;
            read_idx_q      <= 13'h0;
            write_chunk_q   <= 11'h0;
            dq_params_done_q <= 1'b0;
            gelu_part_q     <= 2'h0;
            attn_k_idx_q    <= 16'h0;

            if (dispatch_unsupported_w) begin
              fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
              state        <= F_FAULT;
            end else if (dispatch_attn_context_bad_w) begin
              fault_code_r <= 4'(FAULT_NO_CONFIG);
              state        <= F_FAULT;
            end else if (dispatch_sram_oob_w) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= F_FAULT;
            end else begin
              case (opcode)
                // 2026-05-23 Phase B: gen-1 SFU dispatch arms stripped.
                // Gen-1 opcodes (0x0E/0x0F/0x10/0x12/0x15/0x16) are now
                // illegal at decode_unit; they never reach this case.
                OP_VADD_FP32, OP_LAYERNORM_FP32, OP_GELU_FP32,
                OP_DEQUANT_ACCUM_FP32, OP_QUANT_FP32_INT8,
                OP_MASKED_SOFTMAX_FP32, OP_DEQUANT_ACCUM_FP32_SCALED,
                OP_MAX_ABS_REDUCE_FP32:
                  state <= F_G2_S1_REQ;

                default: begin
                  fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
                  state        <= F_FAULT;
                end
              endcase
            end
          end
        end

        // 2026-05-23 Phase B: gen-1 LN_PARAM / ROW_I8 / ROW_I32 state bodies
        // stripped (OP_LAYERNORM/SOFTMAX/MASKED_SOFTMAX illegal at decode).
        // The `include "sfu_g1_compute.svh"` (gen-1 F_ROW_COMPUTE body) is
        // also removed. F_ROW_PACK / F_ROW_WRITE stay — gen-2 0x18
        // QUANT_FP32_INT8 uses them too (int8 pack format).
        // Streamed writeback (INT8 output, 0x18 QUANT). Same pipeline as
        // F_G2_PACK/WRITE: F_ROW_PACK primes chunk 0; F_ROW_WRITE issues one
        // SRAM write per cycle (row_write_q @ row_dst_addr_w(g2_wr_addr_q))
        // while row_write_data_w packs the next chunk. 1 cyc/chunk vs 2.
        F_ROW_PACK: begin
          row_write_q   <= row_write_data_w;       // packs write_chunk_q (==0)
          g2_wr_addr_q  <= write_chunk_q;
          write_chunk_q <= write_chunk_q + 11'd1;
          state         <= F_ROW_WRITE;
        end

        F_ROW_WRITE: begin
          // The comb block writes row_write_q every F_ROW_WRITE cycle, so the
          // chunk g2_wr_addr_q is committed this cycle regardless of branch.
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q < n_tiles_q) begin
            row_write_q   <= row_write_data_w;      // pack next chunk
            g2_wr_addr_q  <= write_chunk_q;
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_ROW_WRITE;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            // 2026-05-23 Phase B: only gen-2 OP_QUANT_FP32_INT8 (0x18) reaches
            // F_ROW_WRITE now; the gen-1 SOFTMAX/MASKED_SOFTMAX next-row arms
            // (-> F_ROW_I32_REQ / F_ROW_I8_REQ) were stripped along with their
            // states. Loop directly back to F_G2_S1_REQ for the next FP16 row.
            state <= F_G2_S1_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        // 2026-05-23 Phase B: gen-1 GELU INT8/INT32 state bodies stripped
        // (OP_GELU 0x10 illegal at decode). The F_GELU_SYNTH_I8_ITER and
        // F_GELU_SYNTH_I32_ITER synth-mode iterators went with them.
        // 2026-05-23 Phase B: gen-1 ATTN_QKT/PREP/V/WRITE state bodies
        // stripped (OP_SOFTMAX_ATTNV 0x12 / OP_MASKED_SOFTMAX_ATTNV 0x16
        // illegal at decode). The `include "sfu_attn.svh"` is also removed.
        // ----------------------------------------------------------------
        // gen-2 FP32 shared datapath (0x19 VADD / 0x1A LN / 0x1B GELU).
        // FP16 storage (8 elems / 16-byte row), FP32 internal. src1 is an
        // ABUF FP16 tile; src2 is an ABUF FP16 tile (VADD) or 2N FP16
        // gamma||beta in WBUF (LN); GELU has no src2. Output FP16 to ABUF.
        // ----------------------------------------------------------------
        // Streamed src1 load. F_G2_S1_REQ primes (the comb block issues chunk
        // read_idx_q==0), then F_G2_S1_LATCH captures the prior chunk (ld_cap_q)
        // while the comb block issues the next (read_idx_q). 2026-05-31:
        // MAX_ABS_REDUCE now streams too — its running-max reduction is a single
        // -cycle R-M-W on g2_maxabs_q, so it keeps the divider-free 1-cyc/chunk
        // pace; the reduction mask just follows the captured chunk ld_cap_q.
        F_G2_S1_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          // QUANT (mode-1): fuse the FP16 load with the 8-wide quant compute
          // (same 8-elem/cyc rate) via F_G2_QLC instead of the sequential
          // F_G2_S1_LATCH -> F_G2_SYNTH_ITER. Mode-0/DPI keeps the old path.
          end else if (SFU_SYNTH_MODE == 1 && opcode_q == OP_QUANT_FP32_INT8) begin
            ld_cap_q      <= read_idx_q;          // chunk on the bus next cycle (==0)
            read_idx_q    <= read_idx_q + 13'd1;  // issue pointer -> chunk 1
            iter_idx_q    <= 11'h0;
            qlc_vis_q     <= 13'h0;
            qlc_load_q    <= 1'b1;
            write_chunk_q <= 11'h0;
            cw_have_q     <= 1'b0;
            state         <= F_G2_QLC;
          // GELU (mode-1, 0x1B): single-operand FP16-in/FP16-out; fuse the FP16
          // load with the 8-wide gelu compute + FP16 writeback (F_G2_GLC) instead
          // of the sequential F_G2_S1_LATCH -> F_G2_SYNTH_ITER -> F_G2_PACK/WRITE.
          // Same FP16 8-elem/cyc load as QUANT (F_G2_QLC), FP16 output like VADD.
          // Mode-0/DPI keeps the old F_G2_COMPUTE path (this branch is mode-1-gated).
          end else if (SFU_SYNTH_MODE == 1 && opcode_q == OP_GELU_FP32) begin
            ld_cap_q      <= read_idx_q;          // chunk on the bus next cycle (==0)
            read_idx_q    <= read_idx_q + 13'd1;  // issue pointer -> chunk 1
            iter_idx_q    <= 11'h0;               // gelu pipe FEED pointer
            gelu_coll_q   <= 11'h0;               // gelu pipe COLLECT pointer
            glc_vis_q     <= 13'h0;
            glc_load_q    <= 1'b1;
            write_chunk_q <= 11'h0;
            cw_have_q     <= 1'b0;
            state         <= F_G2_GLC;
          // DEQUANT (mode-1, 0x17/0x1E): reorder src2 (per-col scales/bias) ahead
          // of the int32 src1 so the src1 load fuses with compute+write in
          // F_G2_DQL. dq_params_done_q sequences the two visits here: 0 -> bounce
          // to S2 to load params (read_idx_q stays 0 for the S2 prime); 1 -> prime
          // int32 chunk 0 + enter F_G2_DQL. Mode-0/DPI keeps the old
          // S1_LATCH -> S2 -> F_G2_COMPUTE path (this branch is mode-1-gated).
          end else if (SFU_SYNTH_MODE == 1 &&
                       (opcode_q == OP_DEQUANT_ACCUM_FP32 ||
                        opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED)) begin
            if (!dq_params_done_q) begin
              state <= F_G2_S2_REQ;                  // load src2 params first
            end else begin
              ld_cap_q      <= read_idx_q;           // int32 chunk on bus next cyc (==0)
              read_idx_q    <= read_idx_q + 13'd1;   // issue pointer -> chunk 1
              iter_idx_q    <= 11'h0;                // dequant pipe FEED pointer
              dq_coll_q     <= 11'h0;                // dequant pipe COLLECT pointer
              write_chunk_q <= 11'h0;
              cw_have_q     <= 1'b0;
              dq_vis_q      <= 13'h0;
              dq_load_q     <= 1'b1;
              state         <= F_G2_DQL;
            end
          end else begin
            ld_cap_q   <= read_idx_q;          // chunk on the bus next cycle (==0)
            read_idx_q <= read_idx_q + 13'd1;  // issue pointer -> chunk 1
            state      <= F_G2_S1_LATCH;
          end
        end

        F_G2_S1_LATCH: begin
          integer base_idx;
          if (opcode_q == OP_DEQUANT_ACCUM_FP32 ||
              opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
            // 0x17 / 0x1E: src1 = ACCUM INT32, 4 int32 / 16-byte row, raw
            // -> real (scales/bias applied later in F_G2_COMPUTE). Streamed:
            // capture chunk ld_cap_q; the comb block already issued read_idx_q.
            base_idx = integer'(ld_cap_q) * 4;
`ifndef SFU_SYNTH_NO_DPI
            for (int lane = 0; lane < 4; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q))
                row_data_q[base_idx + lane] <=
                    real_to_fp32_bits(real'(get_i32(sram_b_rdata, lane)));
            end
`endif
            if (read_idx_q < n_chunks_i32_q) begin
              ld_cap_q   <= read_idx_q;          // chunk issued this cycle
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_LATCH;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              state         <= F_G2_S2_REQ;   // src2: scales (+bias)
            end
          end else if (opcode_q == OP_MAX_ABS_REDUCE_FP32) begin
            // 0x1F: FP16 src1; accumulate the GLOBAL max|x| over the whole
            // M*N tile (own row loop, no per-row output). Streamed: reduce the
            // CAPTURED chunk ld_cap_q (on the bus now); the comb block already
            // issued the next chunk read_idx_q.
            base_idx = integer'(ld_cap_q) * 8;
            if (SFU_SYNTH_MODE == 1) begin
              // Synth: max-reduce the 8 fp32-bit-abs lanes against the
              // current g2_maxabs_q (computed combinationally at module
              // scope as `mar_new_max`, masked by mar_base_idx = ld_cap_q*8);
              // store back via fp32_bits_to_real.
              g2_maxabs_q <= mar_new_max;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned). max is order-independent, so
              // streaming the chunk captures leaves the global result identical.
              real m;
              real v;
              real av;
              m = fp32_bits_to_real(g2_maxabs_q);
              for (int lane = 0; lane < 8; lane++) begin
                if ((base_idx + lane) < integer'(n_elems_q)) begin
                  v  = sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)});
                  av = (v < 0.0) ? -v : v;
                  if (av > m) m = av;
                end
              end
              g2_maxabs_q <= real_to_fp32_bits(m);
`endif
            end
            if (read_idx_q < {2'h0, g2_rows_q[10:0]}) begin
              // more chunks issued for this row: capture next, keep streaming.
              ld_cap_q   <= read_idx_q;
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_LATCH;
            end else if (row_idx_q + 15'd1 < m_rows_q) begin
              // row complete (captured its last chunk); advance row + re-prime.
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;
            end else begin
              state <= F_G2_SCALE_WR;          // all elements seen
            end
          end else begin
            // FP16 src1 tile, 8 elems / 16-byte row. Streamed: capture chunk
            // ld_cap_q; the comb block already issued read_idx_q.
            base_idx = integer'(ld_cap_q) * 8;
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (SFU_SYNTH_MODE == 1)
                  row_data_q[base_idx + lane] <=
                      synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                else
                  row_data_q[base_idx + lane] <=
                      real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
              end
            end
            if (read_idx_q < {2'h0, g2_rows_q[10:0]}) begin
              ld_cap_q   <= read_idx_q;
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S1_LATCH;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              // GELU / QUANT_FP32_INT8 / MASKED_SOFTMAX_FP32 have no src2.
              if (opcode_q == OP_GELU_FP32 ||
                  opcode_q == OP_QUANT_FP32_INT8 ||
                  opcode_q == OP_MASKED_SOFTMAX_FP32)
                state <= F_G2_COMPUTE;
              else
                state <= F_G2_S2_REQ;
            end
          end
        end

        // 2026-06-01: QUANT (0x18) FULLY-FUSED load+compute+write (mode-1). FP16
        // load, 8-wide quantize, and INT8 writeback all run as concurrent tracks
        // in one pass (vs the old load -> compute -> F_ROW_PACK/WRITE three-pass
        // flow), so QUANT is ~load-bound. Three trailing pointers:
        //   * ld_cap_q / qlc_vis_q : FP16 load capture (port B, 8 elem/cyc).
        //   * iter_idx_q           : quant compute (8 elem/cyc, trails the load by
        //                            >=1 chunk: out_bytes_q[iter] needs
        //                            row_data_q[iter..] registered & visible).
        //   * write_chunk_q        : INT8 write chunk (16 elem, so 1 per 2 compute
        //                            cyc; trails compute). row_write_q registers
        //                            the F_ROW pack (row_write_data_w from
        //                            out_bytes_q); the comb writes it to port A
        //                            (row_dst_addr_w(g2_wr_addr_q)) the next cycle;
        //                            cw_have_q marks a staged write. Reuses the
        //                            F_ROW_WRITE int8 pack/addr + multi-row tail,
        //                            so F_ROW_PACK/F_ROW_WRITE are bypassed here.
        // Load is port B, write is port A, compute is registers -> no port clash.
        // Bit-exact: identical captures (synth_lat_h2f) + identical 8-wide quant
        // (synth_quant_out_lane) + identical INT8 pack (row_write_data_w) &
        // addresses in chunk order; only overlapped. fmax-neutral: load->row_data_q,
        // row_data_q->compute->out_bytes_q and out_bytes_q->row_write_q boundaries
        // are all preserved; the tracks are independent same-cycle paths that never
        // chain. Mode-0/DPI untouched (QUANT there: F_G2_S1_LATCH -> F_G2_COMPUTE
        // -> F_ROW_PACK; F_G2_QLC is gated SFU_SYNTH_MODE==1 in F_G2_S1_REQ).
        F_G2_QLC: begin
          integer cap_base;
          if (cw_have_q && sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // INT8 write OOB
            state        <= F_FAULT;
          end else if (qlc_load_q && sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // FP16 load OOB
            state        <= F_FAULT;
          end else if (write_chunk_q >= n_tiles_q) begin
            // Every INT8 chunk written (the comb committed the last staged chunk
            // this cycle when cw_have_q). Advance row / finish (== F_ROW_WRITE tail).
            cw_have_q     <= 1'b0;
            iter_idx_q    <= 11'h0;
            write_chunk_q <= 11'h0;
            qlc_vis_q     <= 13'h0;
            qlc_load_q    <= 1'b0;
            if (row_idx_q + 15'd1 < m_rows_q) begin
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;
            end else begin
              state <= F_IDLE;
            end
          end else begin
            // LOAD track: capture the FP16 chunk on the bus into row_data_q.
            if (qlc_load_q) begin
              cap_base = integer'(ld_cap_q) * 8;
              for (int lane = 0; lane < 8; lane++)
                if ((cap_base + lane) < integer'(n_elems_q))
                  row_data_q[cap_base + lane] <= synth_lat_h2f[lane];
              qlc_vis_q <= qlc_vis_q + 13'd1;          // chunk ld_cap_q visible next cycle
              if (read_idx_q < {2'h0, g2_rows_q[10:0]}) begin
                ld_cap_q   <= read_idx_q;
                read_idx_q <= read_idx_q + 13'd1;
              end else begin
                qlc_load_q <= 1'b0;                    // last chunk captured
              end
            end
            // COMPUTE track: quantize chunk iter_idx_q once captured & visible
            // (iter_idx_q>>3 < qlc_vis_q) -> out_bytes_q, advance by 8.
            if (({5'h0, iter_idx_q[10:3]} < qlc_vis_q) &&
                ({5'h0, iter_idx_q} < n_elems_q)) begin
              for (int lane = 0; lane < 8; lane++) begin
                automatic logic [10:0] wr_idx = iter_idx_q + 11'(lane);
                if (({5'h0, iter_idx_q} + 16'(lane)) < n_elems_q)
                  out_bytes_q[wr_idx[9:0]] <= synth_quant_out_lane[lane];
              end
              iter_idx_q <= iter_idx_q + 11'd8;
            end
            // WRITE track: stage INT8 chunk write_chunk_q (16 elems) once compute
            // has produced it -- 2 compute-chunks past it ({iter>>4} > write_chunk_q)
            // or compute finished (iter >= n_elems, for the partial last chunk).
            if ((write_chunk_q < n_tiles_q) &&
                (({4'h0, iter_idx_q[10:4]} > write_chunk_q) ||
                 ({5'h0, iter_idx_q} >= n_elems_q))) begin
              row_write_q   <= row_write_data_w;       // packs out_bytes_q[16*write_chunk_q..]
              g2_wr_addr_q  <= write_chunk_q;
              write_chunk_q <= write_chunk_q + 11'd1;
              cw_have_q     <= 1'b1;
            end else begin
              cw_have_q     <= 1'b0;
            end
          end
        end

        // 2026-06-01: DEQUANT (0x17/0x1E) FULLY-FUSED int32-load+compute+write
        // (mode-1). The int32 src1 load (4 elem/cyc, half the 8-wide compute rate),
        // the 8-wide dequant compute, and the FP16 writeback all run as concurrent
        // tracks in one pass (vs the old int32-load -> compute -> F_G2_CW three-pass
        // flow), so DEQUANT is ~int32-load-bound: its compute+write pass is hidden
        // behind the longer int32 load. Reached only AFTER src2 (per-col scales/bias)
        // is already registered -- the src2-before-src1 reorder in F_G2_S1_REQ -- so
        // compute can consume row_data_q as the int32 stream lands. Three pointers:
        //   * ld_cap_q / dq_vis_q : int32 load capture (port B, 4 elem/cyc). The
        //                           int32->fp32 convert is the SAME DPI-guarded
        //                           capture as F_G2_S1_LATCH (compiled OUT of the
        //                           synth netlist, so this track is fmax-neutral).
        //   * iter_idx_q          : 8-wide dequant compute -> out_h_q, trailing the
        //                           load: chunk iter needs row_data_q[iter..iter+7]
        //                           captured (iter+8 <= dq_vis_q*4), or the load
        //                           finished (the partial last chunk).
        //   * write_chunk_q       : FP16 write chunk (8 elem), trailing compute.
        //                           row_write_q registers the g2_write_data_w pack of
        //                           out_h_q[write_chunk_q]; the comb writes it to
        //                           port A (g2_dst_addr_w) next cycle; cw_have_q marks
        //                           a staged write. Same pack/addr/order as F_G2_CW.
        // Load is port B, write is port A, compute is registers -> no port clash.
        // Bit-exact vs the unfused path: identical int32 capture + identical 8-wide
        // dequant (synth_out_bits_lane, the F_G2_CW datapath) + identical FP16 pack
        // (g2_write_data_w) & addresses in chunk order; only overlapped. fmax-neutral:
        // load->row_data_q, row_data_q->compute->out_h_q and out_h_q->row_write_q
        // boundaries are all preserved; the tracks never chain. Mode-0/DPI never
        // enters here (DEQUANT there: S1_LATCH -> S2 -> F_G2_COMPUTE -> F_G2_PACK).
        F_G2_DQL: begin
          integer cap_base;
          if (cw_have_q && sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // FP16 write OOB
            state        <= F_FAULT;
          end else if (dq_load_q && sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // INT32 load OOB
            state        <= F_FAULT;
          end else if (write_chunk_q >= g2_rows_q[10:0]) begin
            // Every FP16 chunk written (the comb committed the last staged chunk
            // this cycle when cw_have_q). Advance row / finish.
            cw_have_q        <= 1'b0;
            iter_idx_q       <= 11'h0;
            dq_coll_q        <= 11'h0;
            write_chunk_q    <= 11'h0;
            dq_vis_q         <= 13'h0;
            dq_load_q        <= 1'b0;
            dq_params_done_q <= 1'b0;   // next row reloads the (row-independent) params
            if (row_idx_q + 15'd1 < m_rows_q) begin
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;
            end else begin
              state <= F_IDLE;
            end
          end else begin
            // LOAD track: capture int32 chunk ld_cap_q (4 elems) -> row_data_q as
            // fp32 (identical to the F_G2_S1_LATCH int32 capture; DPI-guarded).
            if (dq_load_q) begin
              cap_base = integer'(ld_cap_q) * 4;
`ifndef SFU_SYNTH_NO_DPI
              for (int lane = 0; lane < 4; lane++) begin
                if ((cap_base + lane) < integer'(n_elems_q))
                  row_data_q[cap_base + lane] <=
                      real_to_fp32_bits(real'(get_i32(sram_b_rdata, lane)));
              end
`endif
              dq_vis_q <= dq_vis_q + 13'd1;          // chunk ld_cap_q visible next cycle
              if (read_idx_q < n_chunks_i32_q) begin
                ld_cap_q   <= read_idx_q;
                read_idx_q <= read_idx_q + 13'd1;
              end else begin
                dq_load_q <= 1'b0;                   // last int32 chunk captured
              end
            end
            // FEED track: present chunk iter_idx_q to the dequant pipe once its 8
            // elements are loaded & visible (iter+8 <= dq_vis_q*4), or the load has
            // finished (partial last chunk). dq_feed_en_w IS that condition.
            if (dq_feed_en_w) begin
              iter_idx_q <= iter_idx_q + 11'd8;
            end
            // COLLECT track: chunk dq_coll_q emerges 3 cycles after it was fed;
            // dq_vld_q[2] says when. synth_out_bits_lane is the same op-mux ->
            // f2h shell as before, now fed from the pipeline's last register
            // (synth_sc_s_q for 0x1E, synth_mul_d3_q for 0x17).
            if (dq_vld_q[2]) begin
              for (int lane = 0; lane < 8; lane++) begin
                automatic logic [10:0] wr_idx = dq_coll_q + 11'(lane);
                if (({5'h0, dq_coll_q} + 16'(lane)) < n_elems_q)
                  out_h_q[wr_idx[9:0]] <= synth_out_bits_lane[lane];
              end
              dq_coll_q <= dq_coll_q + 11'd8;
            end
            // WRITE track: stage FP16 chunk write_chunk_q once COLLECT produced it
            // ({coll>>3} > write_chunk_q -> out_h_q[write_chunk_q] registered) or
            // collect finished (coll >= n_elems, for the partial last chunk).
            if ((write_chunk_q < g2_rows_q[10:0]) &&
                (({3'h0, dq_coll_q[10:3]} > write_chunk_q) ||
                 ({5'h0, dq_coll_q} >= n_elems_q))) begin
              row_write_q   <= g2_write_data_w;       // packs out_h_q[8*write_chunk_q..]
              g2_wr_addr_q  <= write_chunk_q;
              write_chunk_q <= write_chunk_q + 11'd1;
              cw_have_q     <= 1'b1;
            end else begin
              cw_have_q     <= 1'b0;
            end
          end
        end

        // 2026-06-01: VADD (0x19) FULLY-FUSED src2-load+compute+write (mode-1).
        // VADD adds two FP16 tiles; src1 is already fully in row_data_q (loaded by
        // the normal F_G2_S1 pass). F_G2_VLC streams src2 into attn_accum_q (port B,
        // FP16 8 elem/cyc) WHILE the 8-wide add + FP16 writeback run as concurrent
        // trailing tracks (vs the old src2-load -> compute -> F_G2_CW three passes),
        // so the compute+write pass is hidden behind the src2 load. Op cost drops
        // ~3*g2_rows -> ~2*g2_rows (the two irreducible port-B operand loads: both
        // src1 and src2 are FP16 tiles on the single port B, so 2*g2_rows of reads
        // is the floor once compute+write overlap). Three pointers:
        //   * ld_cap_q / vlc_vis_q : src2 FP16 load capture -> attn_accum_q (8/cyc,
        //                            SAME rate as compute, like F_G2_QLC).
        //   * iter_idx_q           : 8-wide add -> out_h_q, trailing the load: chunk
        //                            iter needs attn_accum_q[iter..iter+7] captured
        //                            (iter>>3 < vlc_vis_q); row_data_q is pre-loaded.
        //   * write_chunk_q        : FP16 write chunk (8 elem), trailing compute
        //                            (== F_G2_CW / F_G2_DQL pack/addr/order).
        // Load=portB, write=portA, compute=regs -> no clash. Bit-exact vs the unfused
        // path: identical src2 capture (synth_lat_h2f -> attn_accum_q), identical
        // 8-wide add (synth_out_bits_lane, the F_G2_CW datapath), identical FP16 pack
        // (g2_write_data_w) + addresses in chunk order; only overlapped. fmax-neutral:
        // load->attn_accum_q, ->compute->out_h_q, ->row_write_q boundaries preserved;
        // tracks never chain. Mode-0/DPI never enters here (VADD there keeps
        // S2_LATCH -> F_G2_COMPUTE -> F_G2_PACK; F_G2_VLC is gated in F_G2_S2_REQ).
        F_G2_VLC: begin
          integer cap_base;
          if (cw_have_q && sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // FP16 write OOB
            state        <= F_FAULT;
          end else if (vlc_load_q && sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // src2 load OOB
            state        <= F_FAULT;
          end else if (write_chunk_q >= g2_rows_q[10:0]) begin
            // Every FP16 chunk written (comb committed the last staged chunk this
            // cycle when cw_have_q). Advance row / finish.
            cw_have_q     <= 1'b0;
            iter_idx_q    <= 11'h0;
            write_chunk_q <= 11'h0;
            vlc_vis_q     <= 13'h0;
            vlc_load_q    <= 1'b0;
            if (row_idx_q + 15'd1 < m_rows_q) begin
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;   // reload next row's src1 (then S2_REQ -> VLC)
            end else begin
              state <= F_IDLE;
            end
          end else begin
            // LOAD track: capture the src2 FP16 chunk ld_cap_q -> attn_accum_q
            // (synth_lat_h2f, the synthesizable FP16->fp32 of the port-B bus; ==
            // F_G2_S2_LATCH VADD branch). Same-rate with compute (8 elem/cyc).
            if (vlc_load_q) begin
              cap_base = integer'(ld_cap_q) * 8;
              for (int lane = 0; lane < 8; lane++)
                if ((cap_base + lane) < integer'(n_elems_q))
                  attn_accum_q[cap_base + lane] <= synth_lat_h2f[lane];
              vlc_vis_q <= vlc_vis_q + 13'd1;          // chunk ld_cap_q visible next cycle
              if (read_idx_q < {2'h0, g2_rows_q[10:0]}) begin
                ld_cap_q   <= read_idx_q;
                read_idx_q <= read_idx_q + 13'd1;
              end else begin
                vlc_load_q <= 1'b0;                    // last src2 chunk captured
              end
            end
            // COMPUTE track: 8-wide add chunk iter_idx_q -> out_h_q once its src2
            // chunk is captured & visible (iter>>3 < vlc_vis_q). Same datapath as
            // F_G2_CW (op-mux selects synth_add_out for OP_VADD_FP32).
            if (({5'h0, iter_idx_q[10:3]} < vlc_vis_q) &&
                ({5'h0, iter_idx_q} < n_elems_q)) begin
              for (int lane = 0; lane < 8; lane++) begin
                automatic logic [10:0] wr_idx = iter_idx_q + 11'(lane);
                if (({5'h0, iter_idx_q} + 16'(lane)) < n_elems_q)
                  out_h_q[wr_idx[9:0]] <= synth_out_bits_lane[lane];
              end
              iter_idx_q <= iter_idx_q + 11'd8;
            end
            // WRITE track: stage FP16 chunk write_chunk_q once compute produced it
            // ({iter>>3} > write_chunk_q) or compute finished (iter >= n_elems).
            if ((write_chunk_q < g2_rows_q[10:0]) &&
                (({3'h0, iter_idx_q[10:3]} > write_chunk_q) ||
                 ({5'h0, iter_idx_q} >= n_elems_q))) begin
              row_write_q   <= g2_write_data_w;       // packs out_h_q[8*write_chunk_q..]
              g2_wr_addr_q  <= write_chunk_q;
              write_chunk_q <= write_chunk_q + 11'd1;
              cw_have_q     <= 1'b1;
            end else begin
              cw_have_q     <= 1'b0;
            end
          end
        end

        // 2026-07-08: GELU (0x1B) FULLY-FUSED load+compute+write (mode-1). GELU
        // is a single-operand FP16-in/FP16-out elementwise op; after the 8-wide
        // widening (1a) its compute is 8 elem/cyc, the SAME rate as the FP16 load.
        // F_G2_GLC streams the operand into row_data_q (port B, 8/cyc, == F_G2_QLC
        // load) WHILE the 8-wide gelu + FP16 writeback run as concurrent trailing
        // tracks (== F_G2_VLC compute/write), vs the old load -> F_G2_SYNTH_ITER
        // compute -> F_G2_PACK/WRITE three passes. Op cost ~3*g2_rows -> ~g2_rows
        // (the single irreducible port-B operand load; compute+write hide behind
        // it). FOUR pointers (2026-07-21, fmax phase 0d — the compute track
        // split in two when the gelu cores became the 33-stage fp32_gelu_p33):
        //   * ld_cap_q / glc_vis_q : FP16 load capture -> row_data_q (8/cyc).
        //   * iter_idx_q           : gelu pipe FEED, trailing the load
        //                            (chunk iter ready when iter>>3 < glc_vis_q).
        //   * gelu_coll_q          : gelu pipe COLLECT -> out_h_q, LATENCY=33
        //                            behind the feed, strobed by synth_gelu_vo.
        //   * write_chunk_q        : FP16 write chunk (== F_G2_CW/VLC pack/addr).
        // Load=portB, write=portA, compute=regs -> no clash.
        //
        // STILL bit-exact vs the unfused path, for the SAME reasons as before
        // (identical FP16 capture, identical per-element gelu arithmetic,
        // identical FP16 pack + addresses in chunk order) plus one new one: the
        // pipe is a pure retiming, and elements are fed AND collected strictly
        // in index order, so no value and no ordering changes — only WHEN each
        // result lands. Cost: +LATENCY cycles once per row (the drain), nothing
        // per element, because the pipe is II=1 and the feed rate is unchanged.
        //
        // NOT fmax-neutral, and that is the entire point: pre-0d the compute
        // track was row_data_q -> fp32_gelu_new -> f2h -> out_h_q in ONE cycle,
        // a ~700 ns combinational cloud (a whole fp32_exp + fp32_div) x8 lanes
        // that had never appeared in a timing report. It is now feed
        // (row_data_q -> one fp32_mul -> reg, 28.9 ns) and collect (reg -> f2h
        // -> out_h_q). Load->row_data_q and out_h_q->row_write_q boundaries are
        // preserved and the tracks still never chain.
        //
        // Mode-0/DPI never enters here (GELU there: S1_LATCH -> F_G2_COMPUTE
        // DPI whole-row). Mode-1 GELU reaches F_G2_GLC and ONLY F_G2_GLC —
        // F_G2_SYNTH_ITER reads the op-mux in-cycle and would sample the pipe
        // mid-flight, so its one (unreachable) dispatch site now faults.
        F_G2_GLC: begin
          integer cap_base;
          if (cw_have_q && sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // FP16 write OOB
            state        <= F_FAULT;
          end else if (glc_load_q && sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);       // FP16 load OOB
            state        <= F_FAULT;
          end else if (write_chunk_q >= g2_rows_q[10:0]) begin
            // Every FP16 chunk written (the comb committed the last staged chunk
            // this cycle when cw_have_q). Advance row / finish.
            cw_have_q     <= 1'b0;
            iter_idx_q    <= 11'h0;
            gelu_coll_q   <= 11'h0;
            write_chunk_q <= 11'h0;
            glc_vis_q     <= 13'h0;
            glc_load_q    <= 1'b0;
            if (row_idx_q + 15'd1 < m_rows_q) begin
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;
            end else begin
              state <= F_IDLE;
            end
          end else begin
            // LOAD track: capture the FP16 chunk ld_cap_q -> row_data_q
            // (synth_lat_h2f, the synthesizable FP16->fp32 of the port-B bus;
            // == F_G2_QLC load / F_G2_S1_LATCH FP16 branch). Same rate as compute.
            if (glc_load_q) begin
              cap_base = integer'(ld_cap_q) * 8;
              for (int lane = 0; lane < 8; lane++)
                if ((cap_base + lane) < integer'(n_elems_q))
                  row_data_q[cap_base + lane] <= synth_lat_h2f[lane];
              glc_vis_q <= glc_vis_q + 13'd1;          // chunk ld_cap_q visible next cycle
              if (read_idx_q < {2'h0, g2_rows_q[10:0]}) begin
                ld_cap_q   <= read_idx_q;
                read_idx_q <= read_idx_q + 13'd1;
              end else begin
                glc_load_q <= 1'b0;                    // last chunk captured
              end
            end
            // FEED track: present chunk iter_idx_q to the 8 gelu pipes once its
            // chunk is captured & visible (iter>>3 < glc_vis_q). gelu_feed_en_w
            // IS that condition and is what drives the pipes' valid_in.
            if (gelu_feed_en_w) begin
              iter_idx_q <= iter_idx_q + 11'd8;
            end
            // COLLECT track: chunk gelu_coll_q emerges from the pipes 33 cycles
            // after it was fed; synth_gelu_vo (lane 0's valid_out, all 8 lanes
            // in lockstep) says when. synth_out_bits_lane is the SAME op-mux ->
            // f2h shell as before — only its input is now a register output
            // instead of a ~700 ns combinational cloud, and only the write
            // index moved from the feed pointer to the collect pointer.
            if (synth_gelu_vo) begin
              for (int lane = 0; lane < 8; lane++) begin
                automatic logic [10:0] wr_idx = gelu_coll_q + 11'(lane);
                if (({5'h0, gelu_coll_q} + 16'(lane)) < n_elems_q)
                  out_h_q[wr_idx[9:0]] <= synth_out_bits_lane[lane];
              end
              gelu_coll_q <= gelu_coll_q + 11'd8;
            end
            // WRITE track: stage FP16 chunk write_chunk_q once COLLECT produced
            // it ({coll>>3} > write_chunk_q) or collect finished (coll >=
            // n_elems). Identical structure to the pre-0d version with the feed
            // pointer swapped for the collect pointer: the invariant is "the
            // pointer has moved past this chunk", which is what guarantees the
            // chunk's 8 out_h_q entries were registered on an earlier cycle.
            if ((write_chunk_q < g2_rows_q[10:0]) &&
                (({3'h0, gelu_coll_q[10:3]} > write_chunk_q) ||
                 ({5'h0, gelu_coll_q} >= n_elems_q))) begin
              row_write_q   <= g2_write_data_w;       // packs out_h_q[8*write_chunk_q..]
              g2_wr_addr_q  <= write_chunk_q;
              write_chunk_q <= write_chunk_q + 11'd1;
              cw_have_q     <= 1'b1;
            end else begin
              cw_have_q     <= 1'b0;
            end
          end
        end

        // Streamed src2 load (LN/0x1E gamma||beta, or VADD 2nd operand). Same
        // pipeline as src1: F_G2_S2_REQ primes chunk 0; F_G2_S2_LATCH captures
        // chunk ld_cap_q while the comb block issues read_idx_q.
        F_G2_S2_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            ld_cap_q   <= read_idx_q;          // chunk on the bus next cycle (==0)
            read_idx_q <= read_idx_q + 13'd1;  // issue pointer -> chunk 1
            // VADD (mode-1): src1 is now fully in row_data_q; fuse the src2 load
            // with the 8-wide add+writeback via F_G2_VLC (same 8-elem/cyc rate)
            // instead of the sequential F_G2_S2_LATCH -> F_G2_COMPUTE -> F_G2_CW.
            // This S2_REQ cycle already primed src2 chunk 0 (comb issues
            // g2_s2_addr_w for read_idx_q==0). Mode-0/DPI keeps the old path.
            if (SFU_SYNTH_MODE == 1 && opcode_q == OP_VADD_FP32) begin
              iter_idx_q    <= 11'h0;
              vlc_vis_q     <= 13'h0;
              vlc_load_q    <= 1'b1;
              write_chunk_q <= 11'h0;
              cw_have_q     <= 1'b0;
              state         <= F_G2_VLC;
            end else
              state      <= F_G2_S2_LATCH;
          end
        end

        F_G2_S2_LATCH: begin
          integer base_idx;
          if (opcode_q == OP_LAYERNORM_FP32 ||
              opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
            // LN: src2 = 2N FP16 (N gamma || N beta). 0x1E: identical
            // layout, N wt-scales (-> gamma_q) || N bias (-> beta_q). The
            // gamma/beta split is keyed on the captured chunk ld_cap_q.
            base_idx = (integer'(ld_cap_q) < integer'(ln_gamma_rows_q)) ?
                       (integer'(ld_cap_q) * 8) :
                       ((integer'(ld_cap_q) - integer'(ln_gamma_rows_q)) * 8);
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (integer'(ld_cap_q) < integer'(ln_gamma_rows_q)) begin
                  if (SFU_SYNTH_MODE == 1)
                    gamma_q[base_idx + lane] <=
                        synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                  else
                    gamma_q[base_idx + lane] <=
                        real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
                end else begin
                  if (SFU_SYNTH_MODE == 1)
                    beta_q[base_idx + lane] <=
                        synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                  else
                    beta_q[base_idx + lane] <=
                        real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
                end
              end
            end
            if (read_idx_q < {1'b0, ln_param_rows_q[11:0]}) begin
              ld_cap_q   <= read_idx_q;
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S2_LATCH;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              // DEQUANT_SCALED (mode-1): wt-scales/bias now registered -> re-enter
              // S1_REQ to prime the int32 src1 and run the fused F_G2_DQL. LN keeps
              // the COMPUTE path (this reroute is mode-1 + 0x1E gated).
              if (SFU_SYNTH_MODE == 1 &&
                  opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
                dq_params_done_q <= 1'b1;
                state            <= F_G2_S1_REQ;
              end else
                state         <= F_G2_COMPUTE;
            end
          end else begin
            // VADD: src2 is an ABUF FP16 tile (2nd operand).
            base_idx = integer'(ld_cap_q) * 8;
            for (int lane = 0; lane < 8; lane++) begin
              if ((base_idx + lane) < integer'(n_elems_q)) begin
                if (SFU_SYNTH_MODE == 1)
                  attn_accum_q[base_idx + lane] <=
                      synth_lat_h2f[lane];
`ifndef SFU_SYNTH_NO_DPI
                else
                  attn_accum_q[base_idx + lane] <=
                      real_to_fp32_bits(sfu_fp16_bits_to_fp32({16'h0, get_u16(sram_b_rdata, lane)}));
`endif
              end
            end
            if (read_idx_q < {2'h0, g2_rows_q[10:0]}) begin
              ld_cap_q   <= read_idx_q;
              read_idx_q <= read_idx_q + 13'd1;
              state      <= F_G2_S2_LATCH;
            end else begin
              read_idx_q    <= 13'h0;
              write_chunk_q <= 11'h0;
              // DEQUANT_ACCUM (mode-1): per-col scales now in attn_accum_q ->
              // re-enter S1_REQ to prime the int32 src1 and run the fused
              // F_G2_DQL. VADD keeps the COMPUTE -> F_G2_CW path (its 2-operand
              // load fusion is a separate lever; this reroute is mode-1 + 0x17 gated).
              if (SFU_SYNTH_MODE == 1 && opcode_q == OP_DEQUANT_ACCUM_FP32) begin
                dq_params_done_q <= 1'b1;
                state            <= F_G2_S1_REQ;
              end else
                state         <= F_G2_COMPUTE;
            end
          end
        end

`include "sfu_g2_compute.svh"
        // Streamed writeback (FP16 output). F_G2_PACK primes the pipeline by
        // registering chunk 0's packed data; F_G2_WRITE then issues one SRAM
        // write per cycle (row_write_q @ g2_dst_addr_w(g2_wr_addr_q)) while the
        // pack mux registers the next chunk. write_chunk_q leads (pack ptr);
        // g2_wr_addr_q lags (the chunk now in row_write_q). 1 cyc/chunk vs 2,
        // row_write_q register boundary preserved so SFU fmax is unchanged.
        F_G2_PACK: begin
          row_write_q   <= g2_write_data_w;       // packs write_chunk_q (==0)
          g2_wr_addr_q  <= write_chunk_q;          // row_write_q holds this chunk
          write_chunk_q <= write_chunk_q + 11'd1;  // advance pack pointer
          state         <= F_G2_WRITE;
        end

        F_G2_WRITE: begin
          // The comb block writes row_write_q every F_G2_WRITE cycle, so the
          // chunk g2_wr_addr_q is committed this cycle regardless of branch.
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q < g2_rows_q[10:0]) begin
            row_write_q   <= g2_write_data_w;       // pack next chunk
            g2_wr_addr_q  <= write_chunk_q;
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_G2_WRITE;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            state         <= F_G2_S1_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        // 0x1F: write scale_regs[sreg]=127/eps (phase 0), then
        // scale_regs[sreg+1]=eps/127 (phase 1). Writes driven in the
        // combinational block; here we just sequence the two phases.
        F_G2_SCALE_WR: begin
          if (g2_wr_phase_q == 1'b0)
            g2_wr_phase_q <= 1'b1;
          else
            state <= F_IDLE;
        end

        F_FAULT: ;

        default:
          state <= F_IDLE;
      endcase
    end
  end

  always_comb begin
    sfu_busy       = (state != F_IDLE) && (state != F_FAULT);
    sfu_fault      = (state == F_FAULT);
    sfu_fault_code = fault_code_r;

    sram_a_en    = 1'b0;
    sram_a_we    = 1'b0;
    sram_a_buf   = dst_buf_q;
    sram_a_row   = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en    = 1'b0;
    sram_b_buf   = src1_buf_q;
    sram_b_row   = 16'h0;

    sfu_scale_we    = 1'b0;
    sfu_scale_waddr = 4'h0;
    sfu_scale_wdata = 16'h0;

    case (state)
      // 2026-05-23 Phase B: gen-1 SRAM-port arbiter arms stripped
      // (F_LN_PARAM_REQ, F_ROW_I8_REQ, F_ROW_I32_REQ, F_GELU_I8_REQ/WRITE,
      // F_GELU_I32_REQ/WRITE, F_ATTN_QKT_REQ, F_ATTN_V_REQ, F_ATTN_WRITE).
      // The states stay in the enum (unreachable; reduced cell-count from
      // missing arm logic) so dead refs in sub-FSM gen-1-aware branches
      // still elaborate. F_ROW_WRITE arm stays — gen-2 0x18 reaches it.
      F_ROW_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = row_dst_addr_w[15:0];
        sram_a_wdata = row_write_q;
      end

      F_G2_S1_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        // 0x17 / 0x1E read INT32 ACCUM (4 / row); others read FP16 tiles.
        sram_b_row = ((opcode_q == OP_DEQUANT_ACCUM_FP32) ||
                      (opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED)) ?
                     row_i32_addr_w[15:0] : g2_s1_addr_w[15:0];
      end

      // Streamed src1: keep issuing the next chunk (read_idx_q) every cycle
      // while F_G2_S1_LATCH captures the prior one. en drops once the issue
      // pointer reaches the chunk count. 2026-05-31: MAX_ABS_REDUCE streams too
      // (FP16-tile addressing, g2_rows_q chunks/row); en follows the same bound.
      F_G2_S1_LATCH: begin
        sram_b_buf = src1_buf_q;
        sram_b_row = ((opcode_q == OP_DEQUANT_ACCUM_FP32) ||
                      (opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED)) ?
                     row_i32_addr_w[15:0] : g2_s1_addr_w[15:0];
        sram_b_en  = ((opcode_q == OP_DEQUANT_ACCUM_FP32) ||
                      (opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED)) ?
                       (read_idx_q < n_chunks_i32_q) :
                       (read_idx_q < {2'h0, g2_rows_q[10:0]});
      end

      // QUANT fused load+compute+write: FP16 src1 read on port B (issue
      // read_idx_q until all g2_rows_q chunks fetched) AND, concurrently, the
      // staged INT8 chunk written on port A (== F_ROW_WRITE arm) when cw_have_q.
      F_G2_QLC: begin
        sram_b_buf = src1_buf_q;
        sram_b_row = g2_s1_addr_w[15:0];
        sram_b_en  = (read_idx_q < {2'h0, g2_rows_q[10:0]});
        if (cw_have_q) begin
          sram_a_en    = 1'b1;
          sram_a_we    = 1'b1;
          sram_a_buf   = dst_buf_q;
          sram_a_row   = row_dst_addr_w[15:0];
          sram_a_wdata = row_write_q;
        end
      end

      // DEQUANT fused int32-load+compute+write: stream INT32 src1 on port B
      // (row_i32_addr_w, == F_G2_S1_LATCH int32 arm) while the staged FP16 chunk
      // is written on port A (g2_dst_addr_w, == F_G2_CW / F_G2_WRITE) when cw_have_q.
      F_G2_DQL: begin
        sram_b_buf = src1_buf_q;
        sram_b_row = row_i32_addr_w[15:0];
        sram_b_en  = (read_idx_q < n_chunks_i32_q);
        if (cw_have_q) begin
          sram_a_en    = 1'b1;
          sram_a_we    = 1'b1;
          sram_a_buf   = dst_buf_q;
          sram_a_row   = g2_dst_addr_w[15:0];
          sram_a_wdata = row_write_q;
        end
      end

      // VADD fused src2-load+compute+write: stream FP16 src2 on port B
      // (g2_s2_addr_w, == F_G2_S2 VADD arm) while the staged FP16 output chunk
      // is written on port A (g2_dst_addr_w, == F_G2_CW / F_G2_WRITE) when cw_have_q.
      F_G2_VLC: begin
        sram_b_buf = src2_buf_q;
        sram_b_row = g2_s2_addr_w[15:0];
        sram_b_en  = (read_idx_q < {2'h0, g2_rows_q[10:0]});
        if (cw_have_q) begin
          sram_a_en    = 1'b1;
          sram_a_we    = 1'b1;
          sram_a_buf   = dst_buf_q;
          sram_a_row   = g2_dst_addr_w[15:0];
          sram_a_wdata = row_write_q;
        end
      end

      // GELU fused load+compute+write: FP16 src1 read on port B (g2_s1_addr_w,
      // == F_G2_QLC / F_G2_S1_LATCH FP16 arm) while the staged FP16 output chunk
      // is written on port A (g2_dst_addr_w, == F_G2_CW / F_G2_VLC) when cw_have_q.
      F_G2_GLC: begin
        sram_b_buf = src1_buf_q;
        sram_b_row = g2_s1_addr_w[15:0];
        sram_b_en  = (read_idx_q < {2'h0, g2_rows_q[10:0]});
        if (cw_have_q) begin
          sram_a_en    = 1'b1;
          sram_a_we    = 1'b1;
          sram_a_buf   = dst_buf_q;
          sram_a_row   = g2_dst_addr_w[15:0];
          sram_a_wdata = row_write_q;
        end
      end

      F_G2_S2_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        // LN gamma/beta and 0x17 per-col scales are row-independent
        // (src2_off + read_idx); VADD's src2 is a full per-row tile.
        sram_b_row = (opcode_q == OP_VADD_FP32) ?
                     g2_s2_addr_w[15:0] : g2_lnp_addr_w[15:0];
      end

      // Streamed src2: same as src1 LATCH (no MAX_ABS here).
      F_G2_S2_LATCH: begin
        sram_b_buf = src2_buf_q;
        sram_b_row = (opcode_q == OP_VADD_FP32) ?
                     g2_s2_addr_w[15:0] : g2_lnp_addr_w[15:0];
        sram_b_en  = (opcode_q == OP_VADD_FP32) ?
                     (read_idx_q < {2'h0, g2_rows_q[10:0]}) :
                     (read_idx_q < {1'b0, ln_param_rows_q[11:0]});
      end

      F_G2_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = g2_dst_addr_w[15:0];
        sram_a_wdata = row_write_q;
      end

      // Fused compute+write: stream the staged chunk (row_write_q) to SRAM A
      // while F_G2_CW's compute track fills out_h_q. Same port/addr/data as
      // F_G2_WRITE; cw_have_q gates en/we so prime + drain cycles don't write.
      F_G2_CW: begin
        if (cw_have_q) begin
          sram_a_en    = 1'b1;
          sram_a_we    = 1'b1;
          sram_a_buf   = dst_buf_q;
          sram_a_row   = g2_dst_addr_w[15:0];
          sram_a_wdata = row_write_q;
        end
      end

      // 0x1F MAX_ABS_REDUCE_FP32 scale write-back. Golden:
      //   eps = clamp(max|x|, 2^-9, 65504*127/2)
      //   scale_regs[sreg]   = float16(127/eps)   (phase 0)
      //   scale_regs[sreg+1] = float16(eps/127)   (phase 1)
      // float16() is a single round of the float64 quotient.
      F_G2_SCALE_WR: begin
        sfu_scale_we = 1'b1;
        if (g2_wr_phase_q == 1'b0) begin
          sfu_scale_waddr = sreg_q;
          if (SFU_SYNTH_MODE == 1)
            sfu_scale_wdata = synth_inv_eps_fp16;
`ifndef SFU_SYNTH_NO_DPI
          else
            sfu_scale_wdata = 16'(sfu_fp64_to_fp16_bits(
                127.0 / g2_clamp_eps(fp32_bits_to_real(g2_maxabs_q))));
`endif
        end else begin
          sfu_scale_waddr = sreg_q + 4'd1;
          if (SFU_SYNTH_MODE == 1)
            sfu_scale_wdata = synth_eps_inv127_fp16;
`ifndef SFU_SYNTH_NO_DPI
          else
            sfu_scale_wdata = 16'(sfu_fp64_to_fp16_bits(
                g2_clamp_eps(fp32_bits_to_real(g2_maxabs_q)) / 127.0));
`endif
        end
      end

      default: ;
    endcase
  end

endmodule

`endif // SFU_ENGINE_SV
