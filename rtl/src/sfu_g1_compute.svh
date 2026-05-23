// F_ROW_COMPUTE state body for sfu_engine.sv main always_ff.
//
// R6 (2026-05-23): extracted from sfu_engine.sv L1515-1629. Implements
// the gen-1 row-wise compute pass (SOFTMAX/LAYERNORM) — both the
// SFU_SYNTH_MODE=1 synth path (reuses F_G2_SM_*/F_G2_LN_* sub-FSMs)
// and the DPI fallback. State arm body only; the case label and
// outer begin/end live in sfu_engine.sv.

// ---- F_ROW_COMPUTE state body (was sfu_engine.sv L1515-L1629) ----
        F_ROW_COMPUTE: begin
          if ((opcode_q == OP_SOFTMAX) || (opcode_q == OP_MASKED_SOFTMAX)) begin
            if (SFU_SYNTH_MODE == 1) begin
              // Phase-3.B: gen-1 SOFTMAX/MASKED_SOFTMAX synth path. Reuses
              // the gen-2 F_G2_SM_{MAX,EXPSUM,OUT} 3-pass sub-FSM with
              // opcode-aware sm_visible_w and writeback (int8 + scale1_q).
              iter_idx_q   <= 11'h0;
              sm_row_max_q <= 32'h0;
              sm_exp_sum_q <= 32'h0;
              sm_have_vis_q <= 1'b0;
              state        <= F_G2_SM_MAX;
            end else begin
`ifndef SFU_SYNTH_NO_DPI
            real row_max_r;
            real exp_sum_r;
            real exp_r;
            logic have_visible;
            have_visible = 1'b0;
            row_max_r = 0.0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if ((i < integer'(n_elems_q)) &&
                  ((opcode_q == OP_SOFTMAX) || attn_visible(row_idx_q, i))) begin
                if (!have_visible || (fp32_bits_to_real(row_data_q[i]) > row_max_r))
                  row_max_r = fp32_bits_to_real(row_data_q[i]);
                have_visible = 1'b1;
              end
            end

            if (!have_visible) begin
              fault_code_r <= 4'(FAULT_NO_CONFIG);
              state        <= F_FAULT;
            end else begin
              exp_sum_r = 0.0;
              for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                if ((i < integer'(n_elems_q)) &&
                    ((opcode_q == OP_SOFTMAX) || attn_visible(row_idx_q, i))) begin
                  exp_r = sfu_fp32_exp(sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), row_max_r));
                  exp_sum_r = sfu_fp32_add(exp_sum_r, exp_r);
                end
              end

              if (exp_sum_r == 0.0) begin
                fault_code_r <= 4'(FAULT_NO_CONFIG);
                state        <= F_FAULT;
              end else begin
                for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
                  if (i < integer'(n_elems_q)) begin
                    if ((opcode_q == OP_MASKED_SOFTMAX) && !attn_visible(row_idx_q, i)) begin
                      out_bytes_q[i] <= 8'h00;
                    end else begin
                      exp_r = sfu_fp32_exp(sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), row_max_r));
                      out_bytes_q[i] <= quantize_to_i8(sfu_fp32_div(exp_r, exp_sum_r), fp32_bits_to_real(scale1_q));
                    end
                  end
                end
                state <= F_ROW_PACK;
              end
            end
`endif
            end  // end SFU_SYNTH_MODE==0 (DPI) SOFTMAX path
          end else if (SFU_SYNTH_MODE == 1) begin
            // Phase-3.B: gen-1 LAYERNORM synth path. Reuses the gen-2
            // F_G2_LN_{SUM,MEAN,VAR,DENOM,OUT} 5-pass sub-FSM with opcode-
            // aware epsilon (gen-1: 1e-6) and writeback (int8 + scale1_q).
            iter_idx_q   <= 11'h0;
            ln_sum_acc_q <= 32'h0;
            ln_var_acc_q <= 32'h0;
            state        <= F_G2_LN_SUM;
          end else begin
`ifndef SFU_SYNTH_NO_DPI
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
            denom_r = sfu_fp32_sqrt(sfu_fp32_add(var_r, LN_EPS));
`ifdef SFU_DEBUG_LN
            ln_debug_mean_q <= real_to_fp32_bits(mean_r);
            ln_debug_var_q <= real_to_fp32_bits(var_r);
            ln_debug_denom_q <= real_to_fp32_bits(denom_r);
`endif

            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              real y_r;
              if (i < integer'(n_elems_q)) begin
                y_r = sfu_fp32_add(
                    sfu_fp32_mul(
                        sfu_fp32_div(sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), mean_r), denom_r),
                        fp32_bits_to_real(gamma_q[i])),
                    fp32_bits_to_real(beta_q[i]));
                out_bytes_q[i] <= quantize_to_i8(y_r, fp32_bits_to_real(scale1_q));
`ifdef SFU_DEBUG_LN
                if (i < 16)
                  ln_debug_y_q[i] <= real_to_fp32_bits(y_r);
`endif
              end
`ifdef SFU_DEBUG_LN
              else if (i < 16) begin
                ln_debug_y_q[i] <= 32'h0;
              end
`endif
            end
            state <= F_ROW_PACK;
`endif
          end
        end
