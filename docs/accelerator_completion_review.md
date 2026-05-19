# TACCEL Accelerator — Comprehensive Review & FPGA-Demo Roadmap

**Date:** 2026-05-19
**Target end-state (user-confirmed):** running on a real FPGA board — tokens/sec on silicon.
**Exactness policy (user-confirmed):** synthesizable approx units are allowed; each
transcendental gets its own characterized ULP band via the freeze §6/§7 revision
mechanism (golden SHA unchanged). This is the lowest-risk path to hardware.

---

## Part I — Where the project stands

### The strategic headline

The RTL is a **byte-exact behavioral model, not synthesizable hardware.** Every one
of the 8 gen-2 SFU ops (`0x17–0x1F`) dispatches through a datapath built from
`real`/`shortreal` storage (`sfu_engine.sv:165–168` `real scale0_q`, `real
row_data_q[]`, `real attn_accum_q[]`) and `import "DPI-C"` C functions
(`sfu_fp32_add/mul/div/exp/sqrt/gelu_new/...`, `sfu_engine.sv:77–91`). There is
**no synthesis, place-and-route, or timing-constraint flow anywhere in the repo**
(verified: no `*.tcl/*.xdc/*.sdc/*.qsf/*.xpr/synth*` outside `.venv/.git`). The
freeze itself names this gap, §4.4:

> "the gen-2 FP32 sub-layer datapath is expected to cost more SFU area than the
> gen-1 ops it supersedes. No synthesis number exists. This is acknowledged, not
> blocking the freeze, but should be measured before RTL sign-off."

So the enormous, well-executed byte-exact-vs-golden effort has produced a
*functionally complete and verified reference implementation*. It will **not**
pass through yosys/Vivado as written. "Completing the accelerator as a whole" is
overwhelmingly about closing the gap from this behavioral model to synthesizable,
timing-closed, board-deployed hardware — not about more correctness work.

### What is mature and working

| Layer | Status |
|---|---|
| **Software compiler** | nanoGPT/GPT-2/DeiT frontends → IR → codegen → ProgramBundle. Strip-mining, KV-cache, runtime patch sites. Mature. |
| **Quantizer** | W8A16, QuaRot (data-free rotation), AWQ, bias-correction, SmoothQuant, TurboQuant-KV, Hessian-guided. GPT-2 124M `weight_only_int8_quarot` = **55.76 PPL** at the 53.42 FP32 ceiling. |
| **ISA** | gen-2 frozen: 19 emitted opcodes, 8 normative gen-2 (`0x17–0x1F`, `0x1C` reserved). Content-pinned golden (`simulator.py` blob `131d3ef1…`). |
| **Golden model** | Cycle-faithful Python simulator, content-pinned, the conformance arbiter. |
| **RTL (functional)** | All 19 opcodes implemented. fetch/decode/control/DMA/16×16-systolic/SFU/helper. ~7k lines RTL. |
| **Cosim + gate** | `test_compare_rtl_golden.py` **5/5 passing** (SHA pin, determinism, prefill token 0/5, prefill+2-decode multi-step P6m). |
| **SW test suite** | ~356 passing (Stage-5 baseline) + 5/5 freeze gate. Only known failures are dataset-discovery (DeiT calibration), not datapath. |

### Correctness posture (what's verified, what's open)

- **Byte-exact (0 fp16 ULP) RTL↔golden:** tiny fixture full prefill+decode (P6m);
  GPT-2 124M single-token prefill **through `block0_out_proj`** (the full
  pre-fp16-overflow datapath).
- **BUG1 (#108) FIXED** — dispatch-drop: `dma_dispatch` lacked `!dma_busy`;
  LOAD silently dropped during a STORE DMA. Fixed in `control_unit.sv`, reproducer
  `test_addr_raw_hazard.cpp`.
- **BUG2 (#115) FIXED** — `flags=1` K-split multitile miscompute, worked around in
  codegen (`_large_weight_tile_plan` prefers full-K, 0 flags=1 emitted).
- **Open correctness items:**
  - **#109** — post-fp16-overflow logits metric + 257-tok 124M decode conformance.
    The freeze §5 definition-of-done is **formally open** until this lands.
  - **#116 (R-min)** — the *latent RTL* `flags=1`-multitile HW bug is unfixed; only
    the codegen workaround makes the frozen bundle safe. **FPGA-generality blocker:**
    any future bundle that emits `flags=1` will miscompute on real hardware.
  - **#114** — ACCUM-snapshot readback is unreliable; a verification-tooling debt.

### The verification asymmetry

The DPI-C `sfu_fp32_*` functions are the SFU's *de facto* numeric reference. The
gen-2 fixtures (`rtl/verilator/fixtures/gen2/*`) and the §7 ULP-band machinery
already exist and already characterize per-op behavior. **This is exactly the
harness needed to band-characterize synthesizable replacement units** — the DPI
path does not get deleted; it becomes the per-op golden each synthesizable unit is
measured against. The hardest verification scaffolding for Phase 1 is already built.

---

## Part II — Gap analysis to "FPGA demo"

Four gap classes. The first two are **coupled**, which is the core of the plan.

### A. Correctness closure (scoped, near-term)
`#109` (logits metric + 257-tok 124M decode), `#116` (RTL `flags=1` HW fix so
hardware is generally correct, not just for the frozen bundle), `#114` (ACCUM
snapshot). Plus the freeze §5 hard item: **the user must commit the spec files**
(`isa/opcodes.py`, `isa/instructions.py`, `isa/encoding.py`,
`golden_model/simulator.py`, `tests/test_compare_rtl_golden.py`) — the freeze is
not real until these are the committed baseline.

### B. Synthesizable SFU + helper (the dominant effort)
`sfu_engine.sv` (1725 lines) and `blocking_helper_engine.sv` (1511 lines) must be
re-implemented as synthesizable RTL:
- `real`/`shortreal` storage → fixed IEEE-754 fp32 register/SRAM storage.
- DPI-C → real RTL arithmetic units:
  - **Mechanical:** fp32 add/sub/mul (pipelined), fp16↔fp32 convert, round,
    int8 quantize/saturate, the NaN/±inf Option-B contract.
  - **Hard (transcendental):** `exp` (masked-softmax), `tanh`/`gelu_new` (GELU),
    `1/sqrt(var+eps)` / `div` (LayerNorm). CORDIC or range-reduction+polynomial,
    each band-characterized against the existing DPI reference.
- Note: *all* 8 ops currently route through `real`+DPI, so even the
  non-transcendental ops (VADD/DEQUANT/QUANT/MAX_ABS) need the storage/arith
  rewrite — but they have no approximation risk, so they land first and cheaply.

### B↔A coupling — the central design fact
A synthesizable `exp`/`tanh`/`rsqrt` **will not be byte-exact** to numpy libm.
This *re-opens* the correctness gate for those ops. The bridge — confirmed by the
user — is the **freeze §7 per-op-class ULP band** mechanism, already in place for
`gelu_new` (≤3 ULP). Each synthesizable transcendental gets a new *measured* band
via a §6 revision; the golden SHA is unchanged; the cosim gate compares
within-band, not 0-ULP. **This means Phase B output changes the conformance
contract — it must be sequenced with A, not after it.** Treat "synthesizable
transcendental + its characterized band" as one atomic deliverable per op.

### C. FPGA platform integration
- **Memory:** `axi4_slave_model.sv` is a testbench ideal slave. Real board needs a
  DDR4/HBM controller (Xilinx MIG / Altera EMIF). The compiler assumes ≥1 GiB DRAM
  for 124M (`run_program` faults at the 16 MB Makefile default).
- **On-chip SRAM:** 128 KB ABUF + 256 KB WBUF + 64 KB ACCUM = **448 KB** behavioral
  `sram_dp.sv` → must map to FPGA BRAM/URAM (a real capacity/banking budget on
  mid-range parts).
- **Host link:** no CPU↔accelerator path exists (PCIe/XDMA, or UART/JTAG for a
  first demo) to stream the ProgramBinary in and logits out.
- **Top-level:** `taccel_top.sv` is sim glue (start/done/fault + AXI). Needs a real
  reset/clock/control wrapper and the host-facing register/DMA interface.

### D. Performance characterization (no target exists yet)
The **only** measured datapoint: ~1600 cycles/instruction on 124M, ≈100 M cycles
for a 124M single-token prefill. **No fmax target, no tokens/sec figure** — and
none can be honestly stated without a clock target and a synthesized critical
path. Setting the perf target and deriving the model is **Phase 0 work, not a
late-stage measurement.**

---

## Part III — Roadmap (FPGA-demo target)

Ordering: **Phase 0 → (Phase 1 ∥ Phase 2) → Phase 3 → Phase 4.** Phases 1 and 2
run in parallel and share the §7 band framework; the linear "finish correctness
then synthesize" framing is wrong because synthesis re-opens correctness.

### Phase 0 — Lock the foundation (weeks, not months)
1. **Close the correctness freeze.** Land #109 (logits metric + 257-tok 124M
   decode — the multi-step `run_cosim_sequence` from P6m is the scaffold; extend
   to logits-level conformance past the overflow boundary). Then commit the §5
   spec files (user-owned) so the freeze becomes the real baseline.
2. **Pick the FPGA target.** Concrete part + board (e.g. an UltraScale+ with
   ≥1 GB DDR4 and ≥1 MB usable URAM/BRAM after the 448 KB SRAM budget). This
   choice sets the synthesis toolchain (Vivado vs yosys+nextpnr) and the memory
   controller for Phase 3.
3. **Set the perf target.** Choose an fmax goal for the chosen part; define the
   tokens/sec success metric for the demo. Build the analytic model now
   (cyc/token × target clock); refine with real numbers in Phase 4.

### Phase 1 — RTL correctness generality (parallel)
4. **#116** — fix the RTL `flags=1`-multitile HW bug (controller per-tile
   `clear_acc` + DRAIN read-modify-write). Flip `test_ksplit_accumulate_diagnostic`
   to a hard assertion. Without this, hardware is correct only for the one frozen
   bundle.
5. **#114** — make ACCUM-snapshot readback trustworthy (or formally retire it as a
   verification path). Needed so Phase 2 can verify the synthesizable systolic/SFU
   ACCUM path.

### Phase 2 — Synthesizable datapath (parallel, the long pole)
6. **Non-transcendental SFU first** (no approximation risk): rewrite `real`/DPI
   storage+arith for VADD/DEQUANT_ACCUM/QUANT_FP32_INT8/DEQUANT_ACCUM_FP32_SCALED/
   MAX_ABS_REDUCE as synthesizable fp32. Gate: still **0 ULP** vs DPI reference
   (these must stay exact).
7. **Transcendental units, one op at a time**, each as an atomic
   "unit + measured band + §6 revision" deliverable:
   - `gelu_new` (tanh) → ≤? ULP (already has a ≤3 behavioral band to anchor to)
   - `masked_softmax` (exp)
   - `layernorm` (rsqrt/div)
   Reuse the existing gen-2 fixtures + `expect_fp16_ulp` as the characterization
   harness; the DPI function is the per-op golden.
8. **First synthesis pass.** Stand up the toolchain (from Phase 0's part choice).
   Produce the **first area/fmax numbers** — the deliverable freeze §4.4 demands.
   Iterate microarchitecture (pipeline depth, LUT vs CORDIC) against fmax.
9. **Re-run the full cosim gate** with synthesizable units behind the *banded*
   comparison. Update the freeze (§6 revision per banded op) — user commits.

### Phase 3 — FPGA platform bring-up
10. **Memory controller**: integrate MIG/EMIF; replace the ideal AXI slave; verify
    the DMA engine against real DDR latency/refresh (the BUG1-class hazards were
    sensitive to DMA timing — re-validate on realistic latency).
11. **SRAM mapping**: ABUF/WBUF/ACCUM → BRAM/URAM with correct banking; close
    timing on the SRAM ports.
12. **Host interface**: minimal path to load a ProgramBinary and read back
    logits/summary (XDMA/PCIe, or UART+JTAG for a first light-up).
13. **Synthesizable top wrapper**: real clock/reset/control; collapse the
    sim-only observability into debug registers or an ILA.

### Phase 4 — On-hardware validation & performance
14. **On-hardware cosim**: run the frozen tiny fixture, then GPT-2 124M, on the
    board; compare logits to the pinned golden within the Phase-2 bands. This is
    the real "definition of done" for the demo.
15. **Measure tokens/sec**; profile the bottleneck (likely SFU latency or DMA
    bandwidth given ~1600 cyc/insn); optimize (SFU pipelining, SRAM
    double-buffering, systolic-array sizing) against the Phase-0 target.

---

## Part IV — Risks, sequencing, immediate next actions

### Top risks
- **R1 — Synthesis re-opens correctness silently.** Mitigation: the atomic
  "unit+band+§6-revision" deliverable in Phase 2.7; never merge a synthesizable
  transcendental without its measured band committed.
- **R2 — `real`-based byte-exactness is partly a Verilator double-precision
  artifact.** The current 0-ULP results on transcendentals do *not* predict
  synthesizable-unit accuracy. Budget Phase 2 against *new* characterization, not
  the existing 0-ULP numbers.
- **R3 — #116 latent HW bug.** Until fixed, the design is correct only for the
  frozen bundle's tiling. Any microarchitecture/tiling change in Phase 2–4 can
  expose it. Fix early (Phase 1).
- **R4 — SRAM/DRAM budget on the chosen part.** 448 KB on-chip + ≥1 GB DRAM is
  not free; validate against the real part in Phase 0, not Phase 3.
- **R5 — DMA timing hazards under real DRAM.** BUG1 was a DMA-concurrency hazard
  found only with realistic timing. Expect more when the ideal slave is replaced;
  keep the `test_addr_raw_hazard.cpp` discipline.

### Immediate next actions (this/next session)
1. **#109** — extend `run_cosim_sequence` (the P6m scaffold) to a logits-level
   conformance metric past the fp16-overflow boundary; this is the last gate on
   the freeze §5 definition-of-done and unblocks the user's §5 commit.
2. **#116** — scope the RTL `clear_acc`-per-tile + DRAIN-RMW fix; it is the only
   *generality* blocker and should land before any synthesis microarchitecture
   work depends on the systolic accumulator path.
3. **Phase 0 board/part decision** — needed before any synthesis toolchain or
   memory-controller work can start; gate for all of Phase 2.8 onward.

### One-line summary
The hard correctness war is essentially won; the unbuilt accelerator is the
**synthesizable SFU + FPGA platform**, and the only safe way to build it is to
treat each synthesizable transcendental as an atomic "unit + characterized §7
band + §6 revision" deliverable, sequenced *with* (not after) the remaining
correctness closure.
