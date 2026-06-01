// F_G2_COMPUTE through F_G2_SM_OUT state bodies for sfu_engine.sv
// main always_ff.
//
// R6 (2026-05-23): extracted from sfu_engine.sv L2117-2504. Includes:
//   * F_G2_COMPUTE (gen-2 dispatch hub + DPI path for non-iter ops)
//   * F_G2_SYNTH_ITER (per-element synth loop for VADD/DEQUANT/
//     QUANT/SCALED)
//   * F_G2_LN_{SUM,MEAN,VAR,DENOM,OUT} (gen-2 LAYERNORM_FP32 synth
//     sub-FSM)
//   * F_G2_SM_{MAX,EXPSUM,OUT} (gen-2 MASKED_SOFTMAX_FP32 synth
//     sub-FSM)
// State arm bodies only; case labels and outer begin/end live in
// sfu_engine.sv.

// ---- F_G2_COMPUTE..F_G2_SM_OUT state bodies (was sfu_engine.sv L2117-L2504) ----
        F_G2_COMPUTE: begin
          if (opcode_q == OP_VADD_FP32) begin
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: 8-wide compute (fp32_add + cvt) fused with the
              // writeback (F_G2_CW) — no separate F_G2_PACK/WRITE drain pass.
              iter_idx_q    <= 11'h0;
              write_chunk_q <= 11'h0;
              cw_have_q     <= 1'b0;
              state         <= F_G2_CW;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_add(fp32_bits_to_real(row_data_q[i]), fp32_bits_to_real(attn_accum_q[i]))));
              end
              state <= F_G2_PACK;
`endif
            end
          end else if (opcode_q == OP_GELU_FP32) begin
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: serialize via F_G2_SYNTH_ITER. The fp32_gelu_new
              // combinational core (tanh-poly) drives synth_gelu_out; the
              // shared synth_compute_out -> f2h shell writes out_h_q.
              // MEASURED-BAND — freeze §7 (≤3 fp16 ULP, anchor).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_gelu_new(fp32_bits_to_real(row_data_q[i]))));
              end
              state <= F_G2_PACK;
`endif
            end
          end else if (opcode_q == OP_DEQUANT_ACCUM_FP32) begin
            // 0x17: FP16 = fp32(INT32) * per-column FP16 scale.
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: 8-wide compute (fp32_mul + cvt) fused with the
              // writeback (F_G2_CW) — no separate F_G2_PACK/WRITE drain pass.
              iter_idx_q    <= 11'h0;
              write_chunk_q <= 11'h0;
              cw_have_q     <= 1'b0;
              state         <= F_G2_CW;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_mul(fp32_bits_to_real(row_data_q[i]), fp32_bits_to_real(attn_accum_q[i]))));
              end
              state <= F_G2_PACK;
`endif
            end
          end else if (opcode_q == OP_QUANT_FP32_INT8) begin
            // 0x18: INT8 = clip(round_half_even(FP16 * scale_regs[sreg])).
            // quantize_to_i8(v, 1.0) == clamp(round_half_even(v), -128,127).
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: serialize via F_G2_SYNTH_ITER (fp32_mul + quant_i8).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_bytes_q[i] <= quantize_to_i8(
                      sfu_fp32_mul(fp32_bits_to_real(row_data_q[i]), fp32_bits_to_real(scale0_q)), 1.0);
              end
              state <= F_ROW_PACK;          // gen-1 INT8 pack (16 / row)
`endif
            end
          end else if (opcode_q == OP_DEQUANT_ACCUM_FP32_SCALED) begin
            // 0x1E: FP16 = int32 * wt_scale[col] * act_scale + bias[col].
            // row_data_q=int32(real); gamma_q=wt-scales; beta_q=bias;
            // scale0_q = scale_regs[sreg] (the fwd act-scale from 0x1F).
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: 3-stage combinational chain (mul, mul, add) + cvt
              // fp16, 8-wide, fused with the writeback (F_G2_CW) — no separate
              // F_G2_PACK/WRITE drain pass.
              iter_idx_q    <= 11'h0;
              write_chunk_q <= 11'h0;
              cw_have_q     <= 1'b0;
              state         <= F_G2_CW;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_add(
                          sfu_fp32_mul(
                              sfu_fp32_mul(fp32_bits_to_real(row_data_q[i]), fp32_bits_to_real(gamma_q[i])),
                              fp32_bits_to_real(scale0_q)),
                          fp32_bits_to_real(beta_q[i]))));
              end
              state <= F_G2_PACK;
`endif
            end
          end else if (opcode_q == OP_MASKED_SOFTMAX_FP32) begin
            // 0x1D causal masked softmax. Golden is mode-independent:
            // keep_through = min(row + query_row_base, valid_kv_len - 1);
            // cols 0..keep_through visible, rest -> 0. FP32 internal.
            if (SFU_SYNTH_MODE == 1) begin
              // Synth path: 3-pass sub-FSM through F_G2_SM_{MAX,EXPSUM,OUT}.
              // BANDED — bounded by fp32_exp accuracy (Phase-3 minimax-tune
              // pending). Compute keep_through here, then iter through passes.
              automatic logic signed [16:0] qrow_s;
              automatic logic signed [16:0] kt_s;
              qrow_s = $signed({5'b0, attn_query_row_base_q}) +
                       $signed({2'b0, row_idx_q[14:0]});
              kt_s = $signed({5'b0, attn_valid_kv_len_q}) - 17'sd1;
              if (qrow_s < kt_s)
                sm_keep_through_q <= 16'(qrow_s);
              else
                sm_keep_through_q <= 16'(kt_s);
              iter_idx_q   <= 11'h0;
              sm_row_max_q <= 32'h0;
              sm_exp_sum_q <= 32'h0;
              sm_have_vis_q <= 1'b0;
              state        <= F_G2_SM_MAX;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
            integer qrow;
            integer keep_through;
            real row_max_r;
            real exp_sum_r;
            logic have_vis;
            qrow = integer'(attn_query_row_base_q) + integer'(row_idx_q);
            keep_through = (qrow < (integer'(attn_valid_kv_len_q) - 1)) ?
                           qrow : (integer'(attn_valid_kv_len_q) - 1);
            have_vis  = 1'b0;
            row_max_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if ((i < integer'(n_elems_q)) && (i <= keep_through)) begin
                if (!have_vis || (fp32_bits_to_real(row_data_q[i]) > row_max_r))
                  row_max_r = fp32_bits_to_real(row_data_q[i]);
                have_vis = 1'b1;
              end
            end
            exp_sum_r = 0.0;
            if (have_vis) begin
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if ((i < integer'(n_elems_q)) && (i <= keep_through))
                  exp_sum_r = sfu_fp32_add(exp_sum_r,
                      sfu_fp32_exp(sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), row_max_r)));
              end
            end
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                if (have_vis && (i <= keep_through) && (exp_sum_r != 0.0))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_div(
                          sfu_fp32_exp(
                              sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), row_max_r)),
                          exp_sum_r)));
                else
                  out_h_q[i] <= 16'h0;
              end
            end
            state <= F_G2_PACK;
`endif
            end  // SFU_SYNTH_MODE==0 else branch
          end else begin
            // LAYERNORM_FP32: mean / var (population) / eps=1e-5 / gamma,beta.
            if (SFU_SYNTH_MODE == 1 && opcode_q == OP_LAYERNORM_FP32) begin
              // Synth path: 3-pass sub-FSM through F_G2_LN_{SUM,MEAN,VAR,
              // DENOM,OUT}. Init iter + sum accumulator (+0 == 32'h0).
              iter_idx_q   <= 11'h0;
              ln_sum_acc_q <= 32'h0;
              state        <= F_G2_LN_SUM;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
              // DPI path (default; cosim-pinned).
              real sum_r;
              real mean_r;
              real var_r;
              real denom_r;
              sum_r = 0.0;
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  sum_r = sfu_fp32_add(sum_r, fp32_bits_to_real(row_data_q[i]));
              end
              mean_r = sfu_fp32_div(sum_r, real'(n_elems_q));
              var_r = 0.0;
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q)) begin
                  real diff_r;
                  diff_r = sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), mean_r);
                  var_r = sfu_fp32_add(var_r, sfu_fp32_mul(diff_r, diff_r));
                end
              end
              var_r = sfu_fp32_div(var_r, real'(n_elems_q));
              denom_r = sfu_fp32_sqrt(sfu_fp32_add(var_r, LN_FP32_EPS));
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if (i < integer'(n_elems_q))
                  out_h_q[i] <= 16'(sfu_fp32_to_fp16_bits(
                      sfu_fp32_add(
                          sfu_fp32_mul(
                              sfu_fp32_div(
                                  sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), mean_r),
                                  denom_r),
                              fp32_bits_to_real(gamma_q[i])),
                          fp32_bits_to_real(beta_q[i]))));
              end
              state <= F_G2_PACK;
`endif
            end
          end
        end

        // Phase-2 SFU_SYNTH_MODE=1 synth-path element iterator. Drives the
        // shared module-scope primitives one element / cycle, advancing
        // iter_idx_q from 0 up to n_elems_q-1. Op-mux at module scope picks
        // which primitive output is written, and which row-buffer/next-state
        // the FSM advances to:
        //   0x19 VADD       -> synth_add_out -> f2h -> out_h_q   -> F_G2_PACK
        //   0x17 DEQUANT_AC -> synth_mul_out -> f2h -> out_h_q   -> F_G2_PACK
        //   0x18 QUANT      -> synth_mul_out -> qi8 -> out_bytes -> F_ROW_PACK
        F_G2_SYNTH_ITER: begin
          // 8-wide SIMD: process up to 8 elements / cycle. Lanes 0..7 read
          // element (iter_idx_q + lane) via the replicated datapath
          // (synth_out_bits_lane / synth_quant_out_lane in
          // sfu_synth_datapath.svh; lane 0 == the original synth_out_bits /
          // synth_quant_out). Each lane is gated by (iter_idx_q + lane) <
          // n_elems_q so partial final groups never write beyond the row.
          // Bit-exact vs. the old 1/cycle loop (independent per-element ops).
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            // GELU is lane-0-only (fp32_gelu_new not replicated across lanes
            // 1..7 — see sfu_synth_datapath.svh), so it strides by 1 and writes
            // only lane 0; all other elementwise ops process 8 lanes/cycle.
            for (int lane = 0; lane < 8; lane++) begin
              automatic logic [10:0] wr_idx = iter_idx_q + 11'(lane);
              if (((lane == 0) || (opcode_q != OP_GELU_FP32)) &&
                  (({5'h0, iter_idx_q} + 16'(lane)) < n_elems_q)) begin
                if (opcode_q == OP_QUANT_FP32_INT8)
                  out_bytes_q[wr_idx[9:0]] <= synth_quant_out_lane[lane];
                else
                  out_h_q[wr_idx[9:0]]     <= synth_out_bits_lane[lane];
              end
            end
            iter_idx_q <= iter_idx_q +
                          ((opcode_q == OP_GELU_FP32) ? 11'd1 : 11'd8);
          end else begin
            iter_idx_q <= 11'h0;
            state      <= (opcode_q == OP_QUANT_FP32_INT8) ? F_ROW_PACK
                                                           : F_G2_PACK;
          end
        end

        // 2026-06-01: FUSED compute+write (mode-1) for the FP16-output stride-8
        // elementwise ops (VADD / DEQUANT_ACCUM / DEQUANT_ACCUM_SCALED). Folds
        // the separate F_G2_PACK/F_G2_WRITE drain pass INTO the 8-wide compute
        // loop, so an op costs load + (compute || write) instead of
        // load + compute + write — the whole ~g2_rows_q-cycle write pass is
        // hidden behind compute. Two pointers run concurrently in this one state:
        //   * iter_idx_q  — COMPUTE pointer (stride 8): each cycle computes one
        //     8-element chunk into out_h_q, exactly as F_G2_SYNTH_ITER (same
        //     datapath, so the row_data_q->compute->f2h->out_h_q register
        //     boundary is unchanged -> SFU fmax is unaffected).
        //   * write_chunk_q — lagging PACK pointer: once compute has produced a
        //     chunk (write_chunk_q < iter_idx_q>>3, i.e. out_h_q[write_chunk_q]
        //     is registered & visible), pack it via the shared g2_write_data_w
        //     into row_write_q; the comb block streams row_write_q to SRAM port A
        //     (idle during compute) the next cycle — identical to F_G2_WRITE.
        // Bit-exact vs the unfused path: identical out_h_q values, identical
        // g2_write_data_w packing (lanes >= n_elems_q zeroed), identical
        // g2_dst_addr_w addresses, written in ascending chunk order — only the
        // write is overlapped with compute. cw_have_q marks a staged-but-unwritten
        // chunk (gates the comb write so prime/drain cycles stay quiet). Mode-0
        // (DPI) never enters here (F_G2_COMPUTE routes it straight to F_G2_PACK),
        // so the cosim byte-match is untouched. GELU (stride 1) and QUANT (INT8
        // pack, F_ROW_*) keep the unfused F_G2_SYNTH_ITER path.
        F_G2_CW: begin
          if (cw_have_q && sram_a_fault) begin
            // Fault on the chunk the comb block is writing this cycle.
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q >= g2_rows_q[10:0]) begin
            // All chunks packed; the comb block commits the final staged chunk
            // THIS cycle (when cw_have_q). Drain complete -> next row / idle.
            cw_have_q     <= 1'b0;
            iter_idx_q    <= 11'h0;
            write_chunk_q <= 11'h0;
            if (row_idx_q + 15'd1 < m_rows_q) begin
              row_idx_q  <= row_idx_q + 15'd1;
              read_idx_q <= 13'h0;
              state      <= F_G2_S1_REQ;
            end else begin
              state <= F_IDLE;
            end
          end else begin
            // WRITE track: pack the next chunk once compute has produced it.
            if ({3'h0, iter_idx_q[10:3]} > write_chunk_q) begin
              row_write_q   <= g2_write_data_w;   // packs out_h_q[write_chunk_q]
              g2_wr_addr_q  <= write_chunk_q;      // its SRAM chunk address
              write_chunk_q <= write_chunk_q + 11'd1;
              cw_have_q     <= 1'b1;
            end else begin
              cw_have_q     <= 1'b0;               // prime cycle: nothing ready
            end
            // COMPUTE track: one 8-element chunk into out_h_q (datapath ==
            // F_G2_SYNTH_ITER; these ops never take the GELU stride-1 branch).
            if ({5'h0, iter_idx_q} < n_elems_q) begin
              for (int lane = 0; lane < 8; lane++) begin
                automatic logic [10:0] wr_idx = iter_idx_q + 11'(lane);
                if (({5'h0, iter_idx_q} + 16'(lane)) < n_elems_q)
                  out_h_q[wr_idx[9:0]] <= synth_out_bits_lane[lane];
              end
              iter_idx_q <= iter_idx_q + 11'd8;
            end
          end
        end

        // 0x1A LAYERNORM_FP32 synth sub-FSM (Phase-2):
        //   1) Sum reduction: sum_acc += row[iter]
        //   2) Mean: mean = sum_acc / n; reset var_acc
        //   3) Variance: var_acc += (row[iter] - mean)^2
        //   4) Denom: denom = sqrt(var_acc / n + LN_EPS)
        //   5) Output: out[iter] = f2h((row[iter] - mean) / denom * gamma + beta)
        F_G2_LN_SUM: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            ln_sum_acc_q <= ln_sum_add_w;
            iter_idx_q   <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q   <= 11'h0;
            state        <= F_G2_LN_MEAN;
          end
        end

        // mean = sum_acc / n_elems_fp32 via the pipelined fp32_div_p2 u_ln_mean
        // (LATENCY=2). ln_sum_acc_q is registered & stable on entry; sample its
        // quotient 2 cycles later. Once-per-row so the +2 cycles are negligible.
        F_G2_LN_MEAN: begin
          state <= F_G2_LN_MEAN_W;
        end
        F_G2_LN_MEAN_W: begin
          state <= F_G2_LN_MEAN_W2;
        end
        F_G2_LN_MEAN_W2: begin    // div_p5 3rd stage
          state <= F_G2_LN_MEAN_W3;
        end
        F_G2_LN_MEAN_W3: begin    // div_p5 4th stage
          state <= F_G2_LN_MEAN_W4;
        end
        F_G2_LN_MEAN_W4: begin    // div_p5 5th stage (LATENCY=5)
          state <= F_G2_LN_MEAN_S;
        end
        F_G2_LN_MEAN_S: begin
          ln_mean_q    <= ln_mean_div_w;   // divider y now valid
          ln_var_acc_q <= 32'h0;
          state        <= F_G2_LN_VAR;
        end

        // 2026-05-31: pipelined to break the ~42 ns sub->mul->add SFU floor.
        // ln_dsq_q holds element (iter-1)'s (row-mean)^2 (registered last cycle);
        // accumulate it (ln_var_add_w = ln_var_acc_q + ln_dsq_q, a lone add)
        // while simultaneously computing element iter's square into ln_dsq_q
        // (the sub+mul feed). Same accumulation order as the old 1-cycle loop
        // (((0+dsq0)+dsq1)+...), so bit-exact; costs +1 drain cycle per row.
        F_G2_LN_VAR: begin
          // ACCUMULATE element (iter-1)'s square, once the pipe has filled.
          if (iter_idx_q >= 11'd1)
            ln_var_acc_q <= ln_var_add_w;
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            // FEED: register element iter's (row[iter]-mean)^2.
            ln_dsq_q   <= ln_diff_sq_w;
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            // iter==n_elems: the final square (element n-1) was just accumulated
            // by the branch above (iter>=1). Variance sum complete.
            iter_idx_q <= 11'h0;
            state       <= F_G2_LN_DENOM_PRE;
          end
        end

        // Phase-4 pipeline cut: F_G2_LN_DENOM was a 4-op fp32 chain
        //   var_acc/n -> +eps -> sqrt -> ln_denom_q
        // ~250 ns combinationally on sky130. Now split: PRE latches the
        // (var_acc/n + eps) sum into ln_var_eps_q; the next cycle takes
        // sqrt of that registered value. Same bit-exact output, +1 cycle.
        // var_norm = var_acc / n via pipelined u_ln_var_norm (LATENCY=2), then
        // +eps. ln_var_acc_q is registered & stable on entry; the +eps add sits
        // after the divider output register. Once-per-row.
        F_G2_LN_DENOM_PRE: begin
          state <= F_G2_LN_DENOM_PRE_W;
        end
        F_G2_LN_DENOM_PRE_W: begin
          state <= F_G2_LN_DENOM_PRE_W2;
        end
        F_G2_LN_DENOM_PRE_W2: begin   // div_p5 3rd stage
          state <= F_G2_LN_DENOM_PRE_W3;
        end
        F_G2_LN_DENOM_PRE_W3: begin   // div_p5 4th stage
          state <= F_G2_LN_DENOM_PRE_W4;
        end
        F_G2_LN_DENOM_PRE_W4: begin   // div_p5 5th stage (LATENCY=5)
          state <= F_G2_LN_DENOM_PRE_S;
        end
        F_G2_LN_DENOM_PRE_S: begin
          ln_var_eps_q <= ln_var_eps_w;   // add(divider y, eps); y now valid
          state        <= F_G2_LN_DENOM;
        end

        // denom = sqrt(ln_var_eps_q) via the pipelined fp32_sqrt_p6 u_ln_sqrt
        // (LATENCY=6). ln_var_eps_q was latched in F_G2_LN_DENOM_PRE_S and is
        // stable on entry; F_G2_LN_DENOM presents it, _W.._W5 are the sqrt's
        // 2nd..6th pipeline stages, _S samples the now-valid ln_denom_w.
        // Once-per-row so the +6 cycles are negligible.
        F_G2_LN_DENOM: begin
          state <= F_G2_LN_DENOM_W;
        end
        F_G2_LN_DENOM_W: begin
          state <= F_G2_LN_DENOM_W2;
        end
        F_G2_LN_DENOM_W2: begin
          state <= F_G2_LN_DENOM_W3;
        end
        F_G2_LN_DENOM_W3: begin
          state <= F_G2_LN_DENOM_W4;
        end
        F_G2_LN_DENOM_W4: begin
          state <= F_G2_LN_DENOM_W5;
        end
        F_G2_LN_DENOM_W5: begin           // sqrt p6 6th stage (LATENCY=6)
          state <= F_G2_LN_DENOM_S;
        end
        F_G2_LN_DENOM_S: begin
          ln_denom_q <= ln_denom_w;        // sqrt y now valid
          ln_coll_q  <= 11'h0;             // collect pointer for the pipelined OUT
          state      <= F_G2_LN_OUT_DIFF;  // iter_idx_q is already 0 (reset in VAR)
        end

        // 2026-05-31: divider-drain SOFTWARE-PIPELINED LN output. The old
        // DIFF/NORM/W/W2/W3/OUT chain processed ONE element per 6 cycles —
        // it fed a single (row-mean) into the fully-pipelined fp32_div_p5
        // u_ln_norm, then idled draining it. Since u_ln_norm accepts a
        // new dividend EVERY cycle, this single state instead keeps the divider
        // full: each cycle it feeds row[iter_idx_q]-mean into ln_diff_q (the
        // FEED/master pointer) and, once the pipe has filled (iter>=6: 1 ln_diff_q
        // reg + 5 divider stages), collects the quotient now emerging on
        // ln_norm_w for element ln_coll_q (= iter_idx_q-6), applying that
        // element's gamma/beta (ln_gamma/beta_coll_w, indexed by ln_coll_q in
        // the datapath) and writing out_h_q[ln_coll_q]. The phase costs
        // n_elems+6 cycles instead of 6*n_elems (~6x on the OUT pass).
        //   Bit-exact: identical fp32 ops, same operands (gamma/beta now tracked
        //   to the collect element), same RNE rounding, same in-order out_h_q[]
        //   writes — only the per-element latency is overlapped. fmax-neutral:
        //   the feed path (row-mean -> ln_diff_q) and the collect path (ln_norm_w
        //   -> *gamma -> +beta -> f2h -> out_h_q) already existed as the DIFF and
        //   OUT critical paths; they now run concurrently but do not chain.
        //   Gen-1 OP_LAYERNORM (0x0F) is illegal at decode_unit (decode_unit.sv
        //   L45); only gen-2 0x1A LAYERNORM_FP32 reaches here, writing FP16.
        F_G2_LN_OUT_DIFF: begin
          if ({5'h0, ln_coll_q} >= n_elems_q) begin
            // every element collected (also the n_elems==0 degenerate case).
            iter_idx_q <= 11'h0;
            ln_coll_q  <= 11'h0;
            state      <= F_G2_PACK;
          end else begin
            // FEED: present row[iter_idx_q]-mean to the divider input register.
            if ({5'h0, iter_idx_q} < n_elems_q)
              ln_diff_q <= ln_diff_w;
            // COLLECT: after the 6-deep pipe fills (1 ln_diff_q reg + 5 div_p5
            // stages), ln_norm_w holds element ln_coll_q's quotient; finalize
            // and write it (in element order).
            if (iter_idx_q >= 11'd6) begin
              out_h_q[ln_coll_q[9:0]] <= ln_out_h_w;
              ln_coll_q               <= ln_coll_q + 11'd1;
            end
            iter_idx_q <= iter_idx_q + 11'd1;
          end
        end

        // Retired serial LN-OUT drain states — replaced by the pipelined
        // F_G2_LN_OUT_DIFF above. Unreachable; kept as enum-valid no-ops.
        F_G2_LN_OUT_NORM, F_G2_LN_OUT_W, F_G2_LN_OUT_W2,
        F_G2_LN_OUT_W3, F_G2_LN_OUT: begin
          state <= F_G2_LN_OUT_DIFF;
        end

        // 0x1D MASKED_SOFTMAX_FP32 synth sub-FSM (Phase-2; BANDED):
        //   F_G2_SM_MAX:  iterate, update row_max if visible & row>max (or
        //                 first visible). visible = iter<n_elems && iter<=kt.
        //   F_G2_SM_EXPSUM: iterate; if visible, exp_sum += exp(row - max).
        //   F_G2_SM_OUT:  iterate; out[iter] = f2h(exp(row-max)/exp_sum) if
        //                 visible & have_vis & exp_sum!=0, else 0.
        F_G2_SM_MAX: begin
          if ({5'h0, iter_idx_q} < sm_iter_bound_w) begin
            // Opcode-aware visibility via sm_visible_w (gen-1, gen-2, and
            // ATTN share this sub-FSM). For kt<0 in gen-2, no iter passes
            // since iter is unsigned ≥ 0; gen-1 unmasked is always visible.
            if (sm_visible_w) begin
              if (!sm_have_vis_q || sm_row_gt_max)
                sm_row_max_q <= synth_a_bits;
              sm_have_vis_q  <= 1'b1;
            end
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q   <= 11'h0;
            // Gen-1 SOFTMAX/ATTNV all illegal at decode; their no-visible
            // FAULT path is unreachable. Gen-2 just falls through to EXPSUM.
            state <= F_G2_SM_EXPSUM;
          end
        end

        F_G2_SM_EXPSUM: begin
          if ({5'h0, iter_idx_q} < sm_iter_bound_w) begin
            if (sm_have_vis_q && sm_visible_w)
              sm_exp_sum_q <= sm_sum_add_w;  // exp_sum += exp(row - row_max)
            iter_idx_q <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            // Gen-1 SOFTMAX/MASKED_SOFTMAX/{ATTNV,MASKED_ATTNV} all illegal at
            // decode_unit (0x0E/0x15/0x12/0x16). Only gen-2 OP_MASKED_SOFTMAX_FP32
            // reaches here and proceeds to F_G2_SM_OUT_NORM.
            state <= F_G2_SM_OUT_NORM;
          end
        end

        // Phase-6 pipeline cut: F_G2_SM_OUT was a chain
        //   exp(row-max) -> /exp_sum -> f2h -> out_h_q
        // at 149 ns post-PNR. NORM latches sm_norm_q = fp32_div(sm_exp_w,
        // sm_exp_sum_q) so OUT just does f2h. Each element costs 2 cycles.
        // exp(row-max)/exp_sum via pipelined u_sm_div (LATENCY=2, STA #2 path
        // at 157 ns). NORM registers the combinational exp into sm_exp_q so the
        // divider sees a registered dividend (isolating fp32_exp from div
        // stage-1); DIV/_W are the divider's two pipeline stages; SM_OUT reads
        // the y (sm_norm_w) directly. iter_idx_q is held across DIV/_W so the
        // SM_OUT visibility check and write target stay on the same element.
        F_G2_SM_OUT_NORM: begin
          if ({5'h0, iter_idx_q} < sm_iter_bound_w) begin
            sm_exp_q <= sm_exp_w;        // exp(row[iter]-max); divider sees it next cycle
            state    <= F_G2_SM_OUT_DIV;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_G2_PACK;
          end
        end
        F_G2_SM_OUT_DIV: begin
          state <= F_G2_SM_OUT_W;
        end
        F_G2_SM_OUT_W: begin
          state <= F_G2_SM_OUT_W2;
        end
        F_G2_SM_OUT_W2: begin     // div_p5 3rd stage
          state <= F_G2_SM_OUT_W3;
        end
        F_G2_SM_OUT_W3: begin     // div_p5 4th stage
          state <= F_G2_SM_OUT_W4;
        end
        F_G2_SM_OUT_W4: begin     // div_p5 5th stage (LATENCY=5)
          state <= F_G2_SM_OUT;
        end

        F_G2_SM_OUT: begin
          // Gen-1 OP_SOFTMAX (0x0E), OP_MASKED_SOFTMAX (0x15),
          // OP_SOFTMAX_ATTNV (0x12), OP_MASKED_SOFTMAX_ATTNV (0x16) are all
          // illegal at decode_unit (decode_unit.sv L45). The INT8 quantize
          // writeback path through sm_g1_quant_w is unreachable; only gen-2
          // OP_MASKED_SOFTMAX_FP32 (0x1D) reaches here, writing FP16.
          if (sm_have_vis_q && sm_visible_w && (sm_exp_sum_q != 32'h0))
            out_h_q[iter_idx_q[9:0]] <= sm_out_h_w;
          else
            out_h_q[iter_idx_q[9:0]] <= 16'h0;
          iter_idx_q <= iter_idx_q + 11'd1;
          state      <= F_G2_SM_OUT_NORM;
        end
