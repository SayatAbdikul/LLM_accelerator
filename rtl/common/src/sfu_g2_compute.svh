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
              // Synth path: serialize via F_G2_SYNTH_ITER (fp32_add + cvt).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
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
              // Synth path: serialize via F_G2_SYNTH_ITER (fp32_mul + cvt).
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
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
              // Synth path: 3-stage combinational chain (mul, mul, add)
              // through F_G2_SYNTH_ITER, then cvt fp16.
              iter_idx_q <= 11'h0;
              state      <= F_G2_SYNTH_ITER;
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
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            if (opcode_q == OP_QUANT_FP32_INT8)
              out_bytes_q[iter_idx_q[9:0]] <= synth_quant_out;
            else
              out_h_q[iter_idx_q[9:0]]     <= synth_out_bits;
            iter_idx_q                     <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= (opcode_q == OP_QUANT_FP32_INT8) ? F_ROW_PACK
                                                           : F_G2_PACK;
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
        F_G2_LN_MEAN_W2: begin    // div_p4 3rd stage
          state <= F_G2_LN_MEAN_W3;
        end
        F_G2_LN_MEAN_W3: begin    // div_p4 4th stage (LATENCY=4)
          state <= F_G2_LN_MEAN_S;
        end
        F_G2_LN_MEAN_S: begin
          ln_mean_q    <= ln_mean_div_w;   // divider y now valid
          ln_var_acc_q <= 32'h0;
          state        <= F_G2_LN_VAR;
        end

        F_G2_LN_VAR: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            ln_var_acc_q <= ln_var_add_w;
            iter_idx_q   <= iter_idx_q + 11'd1;
          end else begin
            iter_idx_q   <= 11'h0;
            state        <= F_G2_LN_DENOM_PRE;
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
        F_G2_LN_DENOM_PRE_W2: begin   // div_p4 3rd stage
          state <= F_G2_LN_DENOM_PRE_W3;
        end
        F_G2_LN_DENOM_PRE_W3: begin   // div_p4 4th stage (LATENCY=4)
          state <= F_G2_LN_DENOM_PRE_S;
        end
        F_G2_LN_DENOM_PRE_S: begin
          ln_var_eps_q <= ln_var_eps_w;   // add(divider y, eps); y now valid
          state        <= F_G2_LN_DENOM;
        end

        // denom = sqrt(ln_var_eps_q) via the pipelined fp32_sqrt_p3 u_ln_sqrt
        // (LATENCY=3). ln_var_eps_q was latched in F_G2_LN_DENOM_PRE_S and is
        // stable on entry; F_G2_LN_DENOM presents it, _W / _W2 are the sqrt's
        // 2nd / 3rd pipeline stages, _S samples the now-valid ln_denom_w.
        // Once-per-row so the +3 cycles are negligible.
        F_G2_LN_DENOM: begin
          state <= F_G2_LN_DENOM_W;
        end
        F_G2_LN_DENOM_W: begin
          state <= F_G2_LN_DENOM_W2;
        end
        F_G2_LN_DENOM_W2: begin           // sqrt p3 3rd stage (LATENCY=3)
          state <= F_G2_LN_DENOM_S;
        end
        F_G2_LN_DENOM_S: begin
          ln_denom_q <= ln_denom_w;        // sqrt y now valid
          state      <= F_G2_LN_OUT_DIFF;
        end

        // Phase-4/5 pipeline cuts: F_G2_LN_OUT was a 5-op fp32 chain
        //   (row - mean) -> /denom -> *gamma -> +beta -> f2h -> out_h_q
        // (~187 ns combinationally on sky130). Now split across 3 cycles:
        //   DIFF: ln_diff_q  <- row - mean                 (fp32_add only)
        //   NORM: ln_norm_q  <- ln_diff_q / denom          (fp32_div only)
        //   OUT : out_h_q[i] <- f2h(norm*gamma + beta);    iter++
        // The DIFF cut removed the worst-case 184 ns ln_mean_q -> ln_norm_q
        // path. Each element now costs 3 cycles instead of 2.
        F_G2_LN_OUT_DIFF: begin
          if ({5'h0, iter_idx_q} < n_elems_q) begin
            ln_diff_q <= ln_diff_w;
            state     <= F_G2_LN_OUT_NORM;
          end else begin
            iter_idx_q <= 11'h0;
            state      <= F_G2_PACK;
          end
        end

        // (row-mean)/denom via pipelined u_ln_norm (LATENCY=2, the 180 ns STA
        // worst path). ln_diff_q was registered in F_G2_LN_OUT_DIFF; the divider
        // y (ln_norm_w) is valid in F_G2_LN_OUT, where the gamma multiply reads
        // it directly (no intermediate ln_norm_q latch). NORM just presents;
        // _W is the divider's 2nd pipeline stage. Net +1 cycle/element.
        F_G2_LN_OUT_NORM: begin
          state <= F_G2_LN_OUT_W;
        end
        F_G2_LN_OUT_W: begin
          state <= F_G2_LN_OUT_W2;
        end
        F_G2_LN_OUT_W2: begin     // div_p4 3rd stage
          state <= F_G2_LN_OUT_W3;
        end
        F_G2_LN_OUT_W3: begin     // div_p4 4th stage (LATENCY=4)
          state <= F_G2_LN_OUT;
        end

        F_G2_LN_OUT: begin
          // Gen-1 OP_LAYERNORM (0x0F) is illegal at decode_unit (see
          // decode_unit.sv L45). The gen-1 INT8 write path is dead — only
          // gen-2 0x1A LAYERNORM_FP32 reaches here, writing FP16.
          out_h_q[iter_idx_q[9:0]] <= ln_out_h_w;
          iter_idx_q               <= iter_idx_q + 11'd1;
          state                    <= F_G2_LN_OUT_DIFF;
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
        F_G2_SM_OUT_W2: begin     // div_p4 3rd stage
          state <= F_G2_SM_OUT_W3;
        end
        F_G2_SM_OUT_W3: begin     // div_p4 4th stage (LATENCY=4)
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
