// Behavioral counterpart of the synthesizable LN reduction sequencer.
// Keep each add/subtract/multiply individually rounded to FP32. This is a
// bounded iterative traversal rather than recursive SV (portable to Verilator).
// Not included in no-DPI synthesis; no golden-model policy lives here.
`ifndef SFU_SYNTH_NO_DPI
  function automatic real ln_pairwise_reduce(input bit squares, input real mean_r);
    real partial [0:7];
    real left_value [0:2];
    real value_r, result_r;
    integer right_start [0:2];
    integer right_len [0:2];
    bit right_phase [0:2];
    integer first, count, depth, half_count;
    bit need_leaf, finished;
    begin
      first = 0;
      count = integer'(n_elems_q);
      depth = 0;
      need_leaf = 1;
      finished = 0;
      result_r = 0.0;
      while (!finished) begin
        if (need_leaf) begin
          while (count > 128) begin
            half_count = (count / 2) & ~7;
            right_start[depth] = first + half_count;
            right_len[depth] = count - half_count;
            right_phase[depth] = 0;
            depth = depth + 1;
            count = half_count;
          end
          for (int i = 0; i < count; i++) begin
            value_r = fp32_bits_to_real(row_data_q[first + i]);
            if (squares) begin
              value_r = sfu_fp32_sub(value_r, mean_r);
              value_r = sfu_fp32_mul(value_r, value_r);
            end
            if (i < 8) partial[i] = value_r;
            else partial[i % 8] = sfu_fp32_add(partial[i % 8], value_r);
          end
          result_r = sfu_fp32_add(
              sfu_fp32_add(sfu_fp32_add(partial[0], partial[1]),
                           sfu_fp32_add(partial[2], partial[3])),
              sfu_fp32_add(sfu_fp32_add(partial[4], partial[5]),
                           sfu_fp32_add(partial[6], partial[7])));
          need_leaf = 0;
        end else if (depth == 0) begin
          finished = 1;
        end else if (!right_phase[depth - 1]) begin
          left_value[depth - 1] = result_r;
          right_phase[depth - 1] = 1;
          first = right_start[depth - 1];
          count = right_len[depth - 1];
          need_leaf = 1;
        end else begin
          result_r = sfu_fp32_add(left_value[depth - 1], result_r);
          depth = depth - 1;
        end
      end
      // The ufunc's outer accumulator has identity +0, even if the tree
      // itself returns -0 for an all-negative-zero leaf.
      ln_pairwise_reduce = sfu_fp32_add(0.0, result_r);
    end
  endfunction
`endif
