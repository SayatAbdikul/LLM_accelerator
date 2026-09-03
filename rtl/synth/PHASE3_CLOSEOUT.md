# Phase 3 close-out — synthesizable-RTL definition of done MET

Status as of 2026-05-21. Supersedes `PHASE2_INTEGRATION.md`.

> **Historical close-out, reconciled 2026-09-03.** The RTL later gained the
> Port-S split, transpose DMA, `m_exact`, K-split RMW drain, and integrated
> pipelined div/sqrt/exp/GELU paths; six legacy gen-1 SFU opcodes were then
> removed. Some commands and file lists below were deleted during cleanup.
> The live hierarchy/check/stat procedure and current metrics are in
> [`BASELINE.md`](BASELINE.md); physical-flow limitations are in
> [`../asic/README.md`](../asic/README.md).

## Definition of done (FPGA-demo roadmap Phase-2)

At this historical milestone, `make synth-check` returned **0** on the selected
design (no FPGA part). Every module in the then-current file list elaborated
through sv2v + yosys to a
synthesizable construct set: zero `real`-typed storage, zero
`import "DPI-C"`, zero system-tasks, zero unbounded loops.

## Phase 3 sub-target ledger

| Sub-target | State | Mechanism |
|---|---|---|
| 3.A | ✅ DONE | 0x1B GELU_FP32 synth migration + fp32_exp Cody-Waite ln2-split + degree-6 Taylor (86→3 ULP); fp32_gelu_new stable algebraic form `x · exp(2z)/(exp(2z)+1)` (eliminates 1+tanh cancellation). Measured: §7 ≤3 fp16 ULP. |
| 3.B | ✅ DONE | All 5 gen-1 compute paths migrated to synth-mode byte-exact: SOFTMAX / MASKED_SOFTMAX via F_G2_SM_* (opcode-aware `sm_visible_w`, fp32_div not mul for int8 output); LAYERNORM via F_G2_LN_* (epsilon mux 1e-6/1e-5); GELU INT8 + INT32 via new F_GELU_SYNTH_I8/I32_ITER + `fp32_gelu_new` tanh-poly (absorbed by int8 quantize); ATTN PREP via F_G2_SM_* with `sm_iter_bound_w` k_elems mux + sync-to-attn_*; ATTN V_LATCH via 16-lane parallel synth datapath. |
| 3.C | ✅ DONE (Phase-2 base) | Helper `dequant_add_pack` synth path via 16-lane parallel primitives (i32_to_fp32 + mul + add + quantize_i8) gated by HELPER_SYNTH_MODE=1. DPI version wrapped in `\`ifndef SFU_SYNTH_NO_DPI`. |
| 3.D | ✅ DONE | Storage cascade `real` → `logic [31:0]` (fp32 bit-pattern) for row_data_q, attn_accum_q, gamma_q, beta_q, scale0_q-scale3_q, g2_maxabs_q, attn_row_max_q, attn_exp_sum_q, ln_debug_*_q. DPI mode wraps assignments with `real_to_fp32_bits(...)` and reads with `fp32_bits_to_real(...)`. Synth mode reads/writes bits directly. `g2_clamp_eps` replaced with bit-level magnitude compare (positive fp32 monotonic as unsigned int). `ln_n_fp32` replaced with `i32_to_fp32` primitive. |
| 3.E | ✅ DONE | All DPI imports + real-using helper functions + DPI call-site else branches wrapped in `\`ifndef SFU_SYNTH_NO_DPI`. `synth-check` Makefile target adds `-DSFU_SYNTH_NO_DPI` to the sv2v invocation. Cosim builds (no flag) retain DPI for byte-exact regression; synth-check builds (flag set) see only the synth-mode datapaths. |
| 3.F | ✅ DONE | `make synth-check` whole-design rc=0 (proc + opt -fast + check -assert + stat). |
| 3.G | ✅ DONE | First whole-design yosys area `stat` captured (2026-05-21): **38,174 cells**, 565,527 public wire bits, 3 memories (3,670,016 memory bits — the SRAM models), 9 submodules. Cell breakdown: 2,437 $add, 2,292 $sub, 698 $mul, 2,294 $mux, 14,730 $lt, 4,335 $eq, plus DFFs/logic ops. End-to-end yosys time 34.76 s (peak 2.6 GB). Compare to control-plane 5,111 cells (sfu/helper blackboxed); whole-design adds the gen-2 SFU sub-FSMs + parallel synth datapaths + 11 fp32 primitive instances. |
| 3.H | ✅ DONE | `BASELINE.md` updated GREEN. `PHASE3_CLOSEOUT.md` (this file) supersedes `PHASE2_INTEGRATION.md`. §7 ledger draft for user-owned freeze §5 commit prepared below. |

## §7 ULP bands — draft for user-owned freeze §5 commit

Below is the band ledger draft. All gen-2 frozen-bundle measurements
unchanged from §7 REVISION 2026-05-19, plus new synth-path measurements
that match within the existing bands. Golden SHA pin
(`131d3ef1a6009519976cf99baf9157a434e67f6f`) unchanged — only the RTL
moved; the golden simulator stayed pinned.

| Op | Band (REVISION 2026-05-21) | Notes |
|---|---|---|
| 0x17 DEQUANT_ACCUM_FP32 | 0 fp16 ULP (bit-exact) | synth = DPI (mech) |
| 0x18 QUANT_FP32_INT8 | 0 (Option-B non-finite) | synth = DPI (mech) |
| 0x19 VADD_FP32 | 0 fp16 ULP | synth = DPI (mech) |
| 0x1A LAYERNORM_FP32 | Fixture 0 / Real-data ≤1 fp16 ULP | synth = DPI on fixture (mech LN); real-data band unchanged from #109 REV 2026-05-19 |
| 0x1B GELU_FP32 (gelu_new) | ≤3 fp16 ULP | synth via Cody-Waite + deg-6 fp32_exp + stable algebraic form; 3 ULP max on fixture (Phase 3.A this session) |
| 0x1D MASKED_SOFTMAX_FP32 | 0 fp16 ULP (fixture) | synth via F_G2_SM_*; fp32_exp 3-ULP max scaffold absorbed by fp16 normalize+round |
| 0x1E DEQUANT_ACCUM_FP32_SCALED | 0 fp16 ULP | synth = DPI (mech) |
| 0x1F MAX_ABS_REDUCE_FP32 | 0 fp16 ULP | synth path: fp32_div + fp32→fp16 (replaces DPI fp64); `g2_clamp_eps` bit-level compare |

**Why bands stayed within previous discipline:** the SOFTMAX precedent
(fp32_exp accuracy absorbed by fp16 quantize) covered LN; the stable
algebraic-form GELU rewrite eliminated the (1+tanh) cancellation that
was the SOFTMAX-precedent-blocker; the Cody-Waite + degree-6 fp32_exp
tightening (86 ULP → 3 ULP max) was the structural ingredient.

## What's intentionally still informational

- **Gen-1 INT8 paths** (SOFTMAX / MASKED_SOFTMAX / LAYERNORM / GELU /
  ATTN) — byte-exact on existing test_sfu fixtures, but informational
  (not in the freeze §7 ledger). Synth-mode equivalence confirmed via
  test_sfu_synth 21/21 PASS.
- **Synthesizable transcendentals** (fp32_exp 3 ULP max, fp32_gelu_new
  measured-band) — these are FPGA-demo bands, not fp32-bit-exact to
  libm. Documented in freeze §6/§7.
- **Whole-design yosys area** — captured at proc/opt-fast level (skips
  flatten). FPGA-mapped area (BRAM/URAM/DSP58 inference) is the natural
  follow-on once an FPGA part is selected.

## Verification (final, 2026-05-21)

| Gate | Result |
|---|---|
| `make synth-check` (whole design, -DSFU_SYNTH_NO_DPI) | rc=0 |
| `make synth-check-ctrl` (control plane, blackboxed) | rc=0, 5,111 cells |
| `make test_sfu` (DPI default, 21 fixtures) | 21/21 PASS, byte-exact |
| `make test_sfu_synth` (synth mode, 21 fixtures) | 21/21 PASS, byte-exact |
| `make test_helpers` / `make test_helpers_synth` | 19/19 each |
| `make test_fp32_*` (11 primitive standalone gates) | All PASS within measured bands |
| `pytest software/tests/test_compare_rtl_golden.py` | 6 passed, 1 skipped (freeze cosim byte-identical) |

## Next (FPGA-demo roadmap Phase-3)

Out of scope for this RTL-only plan: FPGA-part decision, vendor toolchain
(Vivado / Quartus), DRAM controller (MIG / DDR4), host link (PCIe / UART),
BRAM/URAM/DSP58 inference + timing closure, board top, bitstream
bring-up. Roadmap Phase-3 in `docs/accelerator_completion_review.md`.
