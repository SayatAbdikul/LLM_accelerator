// DPI imports + behavioral-real helper functions for sfu_engine.sv.
//
// R6 (2026-05-23): extracted from sfu_engine.sv L88-104, L311-353,
// L382-431. All content is inside a single `ifndef SFU_SYNTH_NO_DPI
// block — when SFU_SYNTH_MODE=1 with `-DSFU_SYNTH_NO_DPI`, this entire
// file evaluates to nothing. SV allows forward function references
// within a module, so consolidating these three pre-existing blocks
// into one earlier-in-file location is equivalent at elaboration.
//
// R8 plan: blocking_helper_engine.sv `\`include`s this same file to
// deduplicate `pow2_int` and `fp16_to_real`. The DPI imports here
// declare a superset of blocking_helper's; the extras are harmless
// (DPI imports must be declared before use; they bind to C
// implementations at link time).

// ---- DPI imports (was sfu_engine.sv L88-L104) ----
`ifndef SFU_SYNTH_NO_DPI
  import "DPI-C" function real sfu_fp32_round(input real value_r);
  import "DPI-C" function real sfu_fp32_add(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_sub(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_mul(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_div(input real lhs_r, input real rhs_r);
  import "DPI-C" function real sfu_fp32_exp(input real value_r);
  import "DPI-C" function real sfu_fp32_sqrt(input real value_r);
  import "DPI-C" function real sfu_fp32_gelu(input real value_r);
  import "DPI-C" function int sfu_fp32_quantize_i8(input real value_r, input real out_scale_r);
  // gen-2 FP32 opcodes (frozen ISA): exact IEEE-754 half<->fp32 (numpy
  // float16 semantics) + tanh gelu_new. NOT the gen-1 erf sfu_fp32_gelu.
  import "DPI-C" function real sfu_fp16_bits_to_fp32(input int bits);
  import "DPI-C" function int  sfu_fp32_to_fp16_bits(input real value_r);
  import "DPI-C" function int  sfu_fp64_to_fp16_bits(input real value_r);
  import "DPI-C" function real sfu_fp32_gelu_new(input real value_r);
`endif

// ---- pow2_int, fp16_to_real (was sfu_engine.sv L311-L353) ----
`ifndef SFU_SYNTH_NO_DPI
  function automatic real pow2_int(input integer exp_i);
    real v;
    integer j;
    begin
      v = 1.0;
      if (exp_i >= 0) begin
        for (j = 0; j < exp_i; j++)
          v = v * 2.0;
      end else begin
        for (j = 0; j < -exp_i; j++)
          v = v * 0.5;
      end
      pow2_int = v;
    end
  endfunction

  function automatic real fp16_to_real(input logic [15:0] bits);
    logic sign_bit;
    logic [4:0] exp_bits;
    logic [9:0] frac_bits;
    real sign_r;
    begin
      sign_bit = bits[15];
      exp_bits = bits[14:10];
      frac_bits = bits[9:0];
      sign_r = sign_bit ? -1.0 : 1.0;

      if ((exp_bits == 5'h0) && (frac_bits == 10'h0)) begin
        fp16_to_real = 0.0;
      end else if (exp_bits == 5'h0) begin
        fp16_to_real = sign_r * (real'(frac_bits) / 1024.0) * pow2_int(-14);
      end else if (exp_bits == 5'h1F) begin
        fp16_to_real = sign_r * 65504.0;
      end else begin
        fp16_to_real = sign_r *
                       (1.0 + (real'(frac_bits) / 1024.0)) *
                       pow2_int(integer'(exp_bits) - 15);
      end
      fp16_to_real = sfu_fp32_round(fp16_to_real);
    end
  endfunction
`endif

// ---- quantize_to_i8, gelu_real, g2_clamp_eps (was sfu_engine.sv L382-L431) ----
`ifndef SFU_SYNTH_NO_DPI
  function automatic logic [7:0] quantize_to_i8(
    input real value_r,
    input real out_scale_r
  );
    int q_i;
    begin
      // Option B non-finite requant contract (isa_generation_freeze.md §7
      // item 8, P6g/#110): NaN -> 0, +inf -> +127, -inf -> -128,
      // finite-overflow -> saturate. Explicit & deterministic — matches
      // the golden np.where(isnan,0,np.clip(...)) semantics. Threshold
      // 1e40 unambiguously separates +-inf from any finite operand on
      // this datapath (fp32 max 3.4e38; fp16-sourced |x| <= 65504).
      if (out_scale_r == 0.0) begin
        quantize_to_i8 = 8'h00;
      end else if (value_r != value_r) begin
        quantize_to_i8 = 8'h00;            // NaN -> 0
      end else if (value_r > 1.0e40) begin
        quantize_to_i8 = 8'h7F;            // +inf -> +127
      end else if (value_r < -1.0e40) begin
        quantize_to_i8 = 8'h80;            // -inf -> -128
      end else begin
        q_i = sfu_fp32_quantize_i8(value_r, out_scale_r);
        if (q_i > 127)
          quantize_to_i8 = 8'h7F;
        else if (q_i < -128)
          quantize_to_i8 = 8'h80;
        else
          quantize_to_i8 = q_i[7:0];
      end
    end
  endfunction

  function automatic real gelu_real(input real x_r);
    begin
      gelu_real = sfu_fp32_gelu(x_r);
    end
  endfunction

  // 0x1F: clamp max|x| to [2^-9, 65504*127/2] (golden MAX_ABS_REDUCE eps).
  function automatic real g2_clamp_eps(input real m);
    real e;
    begin
      e = m;
      if (e < 0.001953125) e = 0.001953125;   // 2^-9
      if (e > 4159004.0)   e = 4159004.0;      // 65504.0*127.0/2.0
      g2_clamp_eps = e;
    end
  endfunction
`endif
