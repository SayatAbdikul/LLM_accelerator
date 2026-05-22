// F_ATTN_QKT_LATCH / F_ATTN_PREP / F_ATTN_V_LATCH state bodies for
// sfu_engine.sv main always_ff.
//
// R6 (2026-05-23): extracted from sfu_engine.sv L1790-1919
// (intermediate F_ATTN_V_REQ at L1863-1871 stays inline in
// sfu_engine.sv since it's a 9-line SRAM-port-enable arm).
// The other small ATTN arms (F_ATTN_QKT_REQ, F_ATTN_WRITE) also
// remain inline. State arm bodies only; case labels and outer
// begin/end live in sfu_engine.sv.

// ---- F_ATTN_QKT_LATCH state body (was sfu_engine.sv L1790-L1808) ----
        F_ATTN_QKT_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 4;
`ifndef SFU_SYNTH_NO_DPI
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(k_elems_q))
              row_data_q[base_idx + lane] <=
                  real_to_fp32_bits(sfu_fp32_mul(real'(get_i32(sram_b_rdata, lane)), fp32_bits_to_real(scale0_q)));
          end
`endif

          if (read_idx_q + 13'd1 < k_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ATTN_QKT_REQ;
          end else begin
            state <= F_ATTN_PREP;
          end
        end


// ---- F_ATTN_PREP state body (was sfu_engine.sv L1809-L1862) ----
        F_ATTN_PREP: begin
          if (SFU_SYNTH_MODE == 1) begin
            // Phase-3.B gen-1 ATTN PREP synth path. Reuses F_G2_SM_*
            // sub-FSM via sm_visible_w + sm_iter_bound_w (k_elems_q). On
            // EXPSUM exit, syncs sm_* → attn_* and zeros attn_accum_q,
            // then transitions to F_ATTN_V_REQ (the V_LATCH DPI path
            // consumes attn_*). V_LATCH itself stays on DPI in synth mode
            // pending a future per-K weighted-V synth pipeline.
            iter_idx_q   <= 11'h0;
            sm_row_max_q <= 32'h0;
            sm_exp_sum_q <= 32'h0;
            sm_have_vis_q <= 1'b0;
            state        <= F_G2_SM_MAX;
          end else begin
`ifndef SFU_SYNTH_NO_DPI
          real row_max_r;
          real exp_sum_r;
          logic have_visible;
          have_visible = 1'b0;
          row_max_r = 0.0;
          for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
            if ((i < integer'(k_elems_q)) &&
                ((opcode_q == OP_SOFTMAX_ATTNV) || attn_visible(row_idx_q, i))) begin
              if (!have_visible || (fp32_bits_to_real(row_data_q[i]) > row_max_r))
                row_max_r = fp32_bits_to_real(row_data_q[i]);
              have_visible = 1'b1;
            end
          end

          exp_sum_r = 0.0;
          for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
            if ((i < integer'(k_elems_q)) &&
                ((opcode_q == OP_SOFTMAX_ATTNV) || attn_visible(row_idx_q, i)))
              exp_sum_r = sfu_fp32_add(
                  exp_sum_r, sfu_fp32_exp(sfu_fp32_sub(fp32_bits_to_real(row_data_q[i]), row_max_r)));
            if (i < integer'(n_elems_q))
              attn_accum_q[i] <= 32'h0;
          end

          if (!have_visible || (exp_sum_r == 0.0)) begin
            fault_code_r <= 4'(FAULT_NO_CONFIG);
            state        <= F_FAULT;
          end else begin
            attn_row_max_q <= real_to_fp32_bits(row_max_r);
            attn_exp_sum_q <= real_to_fp32_bits(exp_sum_r);
            attn_k_idx_q   <= 16'h0;
            read_idx_q     <= 13'h0;
            write_chunk_q  <= 11'h0;
            state          <= F_ATTN_V_REQ;
          end
`endif
          end  // end SFU_SYNTH_MODE==0 (DPI) F_ATTN_PREP path
        end


// ---- F_ATTN_V_LATCH state body (was sfu_engine.sv L1872-L1919) ----
        F_ATTN_V_LATCH: begin
          if (SFU_SYNTH_MODE == 1) begin
            // Phase-3.B gen-1 ATTN V_LATCH synth path. Parallel 16-lane
            // accumulate via attn_acc_new_bits (combinationally computed
            // from sram_b_rdata, attn_weight_eff_w, sm_exp_sum_q).
            for (int lane = 0; lane < 16; lane++) begin
              automatic int idx;
              idx = integer'(read_idx_q) * 16 + lane;
              if (idx < integer'(n_elems_q))
                attn_accum_q[idx[9:0]] <= attn_acc_new_bits[lane];
            end
          end else begin
`ifndef SFU_SYNTH_NO_DPI
          real weight_r;
          if ((opcode_q == OP_MASKED_SOFTMAX_ATTNV) &&
              !attn_visible(row_idx_q, integer'(attn_k_idx_q))) begin
            weight_r = 0.0;
          end else begin
            weight_r = sfu_fp32_div(
                sfu_fp32_exp(sfu_fp32_sub(fp32_bits_to_real(row_data_q[integer'(attn_k_idx_q)]), fp32_bits_to_real(attn_row_max_q))),
                fp32_bits_to_real(attn_exp_sum_q));
          end
          for (int lane = 0; lane < 16; lane++) begin
            integer idx;
            idx = integer'(read_idx_q) * 16 + lane;
            if (idx < integer'(n_elems_q))
              attn_accum_q[idx] <= real_to_fp32_bits(sfu_fp32_add(
                  fp32_bits_to_real(attn_accum_q[idx]),
                  sfu_fp32_mul(
                      sfu_fp32_mul(weight_r, real'(get_i8(sram_b_rdata, lane))),
                      fp32_bits_to_real(scale1_q))));
          end
`endif
          end

          if (read_idx_q + 13'd1 < {2'h0, n_tiles_q}) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ATTN_V_REQ;
          end else if (attn_k_idx_q + 16'd1 < k_elems_q) begin
            attn_k_idx_q <= attn_k_idx_q + 16'd1;
            read_idx_q   <= 13'h0;
            state        <= F_ATTN_V_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ATTN_WRITE;
          end
        end
