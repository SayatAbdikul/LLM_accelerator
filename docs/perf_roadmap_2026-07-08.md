# Performance roadmap — tokens/sec levers (2026-07-08)

Ranked plan for the next tok/s improvements, grounded in a fresh full-chip
profile at HEAD `0aeb597` ("sfu: software-pipeline the masked-softmax output
divider-drain"). All numbers: GPT-2 124M, 1-token prefill bundle
(`/tmp/p124m.bin`), mode-1 (SFU_SYNTH_MODE=1 = real-chip RTL), honest-BW DMA
model (`--fast-beats`, 1 beat/cycle), post-PNR fmax 34.41 MHz.

**Current: 23,934,723 cycles/token → ~1.438 tok/s.**

tok/s = fmax / cycles_per_token. The two axes multiply; every cycle lever
below stacks with every fmax lever.

---

## 1. Where the cycles go (measured)

Per-opcode retire-gap attribution over the full run. Attribution is airtight:
trace-gap total = 23,934,717 vs direct-run 23,934,723 (6-cycle match); MATMUL
gap matches systolic busy-counter to 0.01%, SFU gap to 0.06%.
Raw table: `/tmp/sfu_profile_m1.txt` (regen: `/tmp/sfu_profile_m1.py`).

| block | cycles | share | state |
|---|---:|---:|---|
| MATMUL (systolic) | 10,949,508 | 45.7% | ~93% is M-padding waste (see §2.1) |
| DMA LOAD | 5,354,825 | 22.4% | weight streaming; ~4.5M of 9.9M beats already hidden behind systolic (163f0cb prefetch) |
| SFU | 6,438,802 | 26.9% | fully serialized (sync_wait_sfu == sfu_busy); split below |
| control (SET_ADDR/SYNC/CONFIG) | ~1,125,000 | 4.7% | mostly dispatch hazard-waits (§3.6) |
| STORE + helper | ~185,000 | 0.8% | negligible |

SFU decomposition (6.44M):

| op | cycles | % of SFU | state |
|---|---:|---:|---|
| DEQUANT_ACCUM_FP32_SCALED (0x1E) | 2,004,928 | 31.1% | int32-load-bound (F_G2_DQL) — at port floor |
| LAYERNORM_FP32 (0x1A) | 1,087,319 | 16.9% | 25 ops @ 43.5K cyc; 3 serial fp-add passes/row — structural under bit-exact |
| QUANT_FP32_INT8 (0x18) | 994,531 | 15.4% | FP16-load-bound (F_G2_QLC) — at port floor |
| MAX_ABS_REDUCE (0x1F) | 863,002 | 13.4% | streams 1 chunk/cyc — floor as a standalone op |
| GELU_FP32 (0x1B) | 739,140 | 11.5% | **still 1 elem/cycle (lane-0 only)** — see §2.4 |
| VADD_FP32 (0x19) | 505,385 | 7.8% | 2-operand port-B-bound (F_G2_VLC) — at port floor |
| MASKED_SOFTMAX (0x1D) | 147,504 | 2.3% | OUT pass pipelined (0aeb597); MAX/EXPSUM serial reductions |
| DEQUANT_ACCUM (0x17) | 96,993 | 1.5% | load-bound |

---

## 2. Ranked improvements

| # | lever | tok/s ceiling | effort | risk |
|---|---|---|---|---|
| 1 | Batching B=16 | ~2.5–3.0× (ctx≈0) / ~2.1× @ ctx=1024 | multi-week, mostly compiler | medium |
| 2 | SFU‖DMA overlap relax | 1.29× standalone (shrinks to ~+4% after batching) | 1–2 weeks | high (golden + cosim contract) |
| 3 | fmax cluster: div_p6 + sqrt M_pad-restructure | +12–18%, multiplies everything | multi-day, EDA-heavy | medium — **blocked** (§3.1) |
| 4 | GELU 8-wide | +2.3% | days | low |
| 5 | MAX_ABS producer-fusion | up to +3.6% | ~1 week | medium-high (ISA freeze) |
| 6 | Dispatch-stall investigation | 0 to +3% | ~1 day | low |
| 7 | Multi-clock domains | ~2.2× single-stream / ~3.4× with batching | multi-week | highest (CDC) |
| 8 | LN 2-row software-pipeline | +1.5% | medium | medium |

### 2.1 Batching B=16 — the headline lever (~2.5–3×)

The compiler already pads decode's M=1 to **M_pad=16 through the entire
pipeline** (`w8a16_emit/matmul.py` `pad_dim` + `_zero_fill_fp32_padding_rows`;
CONFIG_TILE m=16; the 16×16 array streams 16-row tiles regardless). So 15/16
of every systolic pass, all weight DMA, and most control (21.7K SET_ADDR pairs
are weight-load addressing) is spent on rows that are **zero padding**.
Batching = fill those 15 rows with 15 more decode streams:

- systolic 10.95M → ~684K/token; weight DMA 5.35M → ~334K/token;
  control ~1.1M → ~0.3M/token; SFU 6.44M stays (elementwise work is
  per-token). Per-token ≈ 7.9M → **~3.0 tok/s (ctx≈0)**.
- At ctx=1024, per-stream attention + KV traffic (~+2.4M/token) does NOT
  batch (each stream has its own KV) → **~2.1×** there.
- Mostly compiler + golden-harness work; near-zero RTL (datapath is already
  16-row-shaped end-to-end).
- Bit-exact per-stream is verifiable: rows are independent through
  matmul (per-PE-row int32 accum), LN, softmax — each stream's output must
  byte-match its single-stream run.
- Note: this is a *throughput* metric (16 concurrent streams), not
  single-stream latency.
- Post-batching the chip is ~80% SFU → #3/#4/#5 become the follow-on levers
  and #2 shrinks to noise.

### 2.2 SFU‖DMA overlap relax (1.29× standalone)

The `forbidden_overlap` invariant (`taccel_top.sv:560`) walls the SFU (26.9%,
fully serialized) off from everything. Letting next-layer weight DMA run under
SFU work hides up to 5.35M behind 6.44M → 23.9M → ~18.6M. Requires: real port
arbitration with backpressure (today DMA outranks and steals shared port A),
dropping the SFU‖DMA arm of the checker, golden-model scheduling semantics,
and re-baselining every cosim test. **Decision fork:** worth it only if
single-stream latency matters or batching is deferred — after batching the
hideable DMA is ~334K (~+4%).

### 2.3 fmax cluster (+12–18%) — blocked on EDA box

The 29.06 ns floor is a tight 3-way primitive cluster: div_p5 stage2/4
(`u_ln_var_norm.rB_P/rD_P`) + sqrt_p6 stage2 (`u_ln_sqrt.rB_r`). Both must be
restructured together (they co-bind): div_p6/p7 AND the sqrt M_pad-in-stage-1
restructure. ~25–28 ns asymptote after. **Currently unverifiable on this
box** — see §3.1. Post-batching this is the main remaining lever (SFU-bound
chip, and the SFU is the 34.41 MHz block).

### 2.4 GELU 8-wide (+2.3%, best contained win)

GELU still strides 1 elem/cycle (lane-0 only) and burns 739K (11.5% of SFU).
The lane-0-only decision rested on "fp32_gelu_new is the area hog (~45%)" —
**falsified by our own measurement**: the trim experiment showed removing the
7 GELU replicas saved only 0.5% of SFU cells (1,287,144 → 1,280,022). The real
area went to the scaled-chain replicas (already paid). Widening GELU populates
`compute_out` for lanes 1..7 in the existing genvar block and reuses the
existing 8-wide writeback mux (no new fanout), FSM stride 1→8 for
OP_GELU_FP32. ~739K → ~180K. Cycle gain measurable immediately; the fmax
stamp needs the blocked PNR (risk low: parallel lanes, boundaries unchanged).
Fix the stale area-hog comment in `sfu_synth_datapath.svh` when doing this.

### 2.5 MAX_ABS producer-fusion (up to +3.6%)

MAX_ABS re-reads whole tiles (863K) to compute a scalar max of values some
SFU op just wrote. fp max is exactly order-independent (selects, never
rounds), so the producer op can maintain a running max|x| over its writeback
for free; MAX_ABS collapses to a register read. Requires a new op/flag →
touches the ISA freeze, golden model, compiler. Standalone MAX_ABS is already
load-bound, so fusion is the only way to reclaim this.

### 2.6 Dispatch-stall investigation (0 to +3%, 1 day)

SET_ADDR_HI retires at 31.2 cyc/op and CONFIG_TILE at 40.9 (baseline simple
ops: 4.0) ≈ 844K total. Hypothesis: the *following* engine op can't dispatch
while that engine is busy, and the wait lands in the preceding ctl op's
retire gap — i.e., mostly the same serialization story, not fetch
inefficiency. Worth one day to confirm; any part that's a genuine dispatch
bubble is a cheap fix.

### 2.7 Multi-clock domains (~2.2× single-stream)

Only the SFU needs 34.41 MHz; per-block PNR: dma 159 MHz, helper 109 MHz,
systolic 875 MHz (synth). Separate clock domains (SFU slow, everything else
fast) ≈ 302 ms/token → ~3.3 tok/s without batching; ~3.4× combined with it.
The byte-match contract is timing-agnostic so cosim survives, but this is a
CDC redesign of SRAM arbitration + dispatch/sync — the heaviest verification
burden on this list, and per-block chip-level fmax at target is unproven.
The only lever besides batching worth >2×, and the only >2× *latency* lever.

### 2.8 LN 2-row software-pipeline (+1.5%)

LN's 3 serial passes/row (SUM, VAR, OUT) use different primitives — row i+1's
SUM could overlap row i's OUT (~360K reclaimable). Awkward: `row_data_q`
holds one row, so it needs double-buffering. Only worth it when squeezing the
post-batching SFU tail.

---

## Exhausted / rejected — do not revisit

- **SFU elementwise load-fusion** — mined out (6 commits, 2026-06/07):
  QUANT/DEQUANT/VADD are all load-bound at the single-port-B floor;
  LN/softmax OUT passes divider-pipelined; MAX_ABS streams.
- **DMA‖systolic overlap** — done (163f0cb), capped at −6.5% by the WBUF
  double-buffer tax (halved tile budget doubled N-groups, +51% insns).
- **Further SFU FSM cuts** — the fmax floor is primitive-internal now.
- **Testbench 1-beat/cycle model** — already the honest-BW metric; a
  reporting decision, not a chip change.
- **Reduced-precision fp32 / w4 weights** — breaks the bit-exact contract
  (or changes model quality). Standing constraint: byte-match cosim.

## Recommended sequence

1. Unblock the EDA box (§3.1 — trivial once swap is added).
2. GELU 8-wide (#4): days, +2.3%, lowest risk.
3. Batching (#1): the big one; compiler-dominant.
4. fmax cluster (#3): now worth more (post-batching chip is SFU-bound).
5. MAX_ABS fusion (#5) / LN pipeline (#8): the SFU tail.

(#2 SFU‖DMA and #7 multi-clock are the alternates if single-stream latency —
not batched throughput — is the target metric.)

---

## 3. Current caveats / problems — RESOLUTION STATUS (2026-07-08 evening)

1. **EDA validation: CLOSED AS HARDWARE-BLOCKED — do not retry on this box.**
   yosys `synth -flatten`+abc on the current SFU netlist was OOM-killed on
   **5/5 attempts, including a swap-backed run** (8 GB swapfile active,
   swap 8 GB free at kill time, no cgroup/ulimit caps): abc's working set is
   hot (`anon-rss ≈ total-vm` at kill — nothing swapped out) and its
   allocation burst outruns swap-out on the 92%-full NVMe. The empirical
   fmax stamp for the 6-fusion stack needs a machine with **≥24 GB RAM**;
   `/tmp/pnr_final.sh` is self-contained for that. Neutrality remains
   architecturally solid (no fp32 primitive modified; fusions reuse
   registered boundaries, no new arithmetic near the 29 ns cluster).
2. **Unpushed commits**: RESOLVED — user pushed `fbee703..0aeb597`; the
   caveat-fix commits (`81ad53f..`) await the next push.
3. **Benchmark shape**: RESOLVED — `software/tools/bench_decode_cycles.py`
   measures the true decode shape (see §4 below). Headline: decode compiled
   for a ctx-512 budget costs **34,605,975 cyc/token = 0.994 tok/s** — the
   1-token prefill number (1.438) overstates deployment decode by ~45%.
4. **Pre-existing test debt**: RESOLVED — SHA pin re-pinned per freeze §6
   (`81ad53f`; cosim file fully green, no deselection); systolic unit-test
   family repaired (`ccd604b`; 5/5 targets, 7/7 chained); d384-on-RTL
   root-caused and fixed (`6cab6ea`; act-quant row-split, byte-identical;
   d384 now runs to halt on RTL).
5. **Stale GELU comment**: RESOLVED (`9f233c5`).
6. **Dirty tree**: RESOLVED (`9f233c5`) — artifact roots gitignored,
   `design_asic.v` restored, fixture metadata refreshed; `git status` clean.

## 4. Decode-shape benchmark (NEW standard measurement)

`software/tools/bench_decode_cycles.py` builds the decoder ProgramBundle,
patches the decode stream to a target position (the exact
`HostRunner.run_decode_step` patch set), and measures one standalone step on
the mode-1 RTL. Cycle counts are data-independent, so zero-filled KV is
valid and no warm-up steps are needed.

Measured at HEAD `0aeb597` + fixes (mode-1 honest-BW, 34.41 MHz), decode
program compiled for a **ctx-512 budget** — cycles are **exactly
position-invariant** (attention is statically shaped for the compiled
window and runtime-masked):

| shape | cycles/tok | sfu_busy | sys_busy | dma beats | tok/s |
|---|---:|---:|---:|---:|---:|
| 1-token prefill (ctx≈0) | 23,934,723 | 6.44M | 10.95M | 9.92M | 1.438 |
| decode @ ctx-512 budget | 34,605,975 | 12.63M | 12.39M | 11.14M | **0.994** |

The +10.7M delta is dominated by **SFU masked-softmax over the full
compiled window (+6.2M)**, then QK^T/AV systolic (+1.4M) and KV DMA
(+1.2M beats). Two roadmap consequences:
- SFU share in the deployment shape is **36.5%** (vs 26.9% at ctx≈0) —
  strengthens every SFU lever in §2.
- **New decode-specific lever — runtime-bounded softmax reductions:** the
  SM MAX/EXPSUM passes walk all n_elems with a visibility gate instead of
  stopping at `keep_through` (invisible elements contribute nothing —
  bounding the walk is bit-exact; the OUT pass must still write zeros).
  Averaged over a 512-token generation (mean valid_kv_len ≈ 256) this
  reclaims roughly half of the MAX/EXPSUM walk, ~2M cyc/token ≈ +6% decode
  tok/s. Makes decode cycles position-dependent (benchmark must then sweep
  positions). Low-risk, contained — slots between #4 and #5 in §2.

## Appendix: measurement recipe

- Build mode-1 binary: `/tmp/build_mode1_cw.sh` (derives sources from
  `core.f`; `rm -rf` first — stale-binary gotcha). Runner: `/tmp/run_mode1.py`
  (both DMA models).
- Cycle totals: trust a DIRECT `run_program_synth --fast-beats` run (the
  profiler's absolute total went stale once; its per-op shares are fine).
- Per-op profile: `/tmp/sfu_profile_m1.py` → `/tmp/sfu_profile_m1.txt`.
- Bit-exact gates: `make -C rtl/verilator test_sfu_synth` (11/11) + `pytest
  software/tests/test_compare_rtl_golden.py -k "byte_match or multistep_tiny
  or frozen_bundle"`.
- Never run two yosys synths, or synth + PNR, concurrently (OOM at
  ~14/15 GB).
