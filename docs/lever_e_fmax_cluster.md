# Lever E — fmax cluster (break the 3-way SFU primitive floor)

> **Historical timing experiment, reconciled 2026-09-03.** The pipelined
> div/sqrt work described below remains in the design, and the later
> `fp32_exp_p18` and `fp32_gelu_p33` pipelines are also integrated. The reported
> 29–32 ns paths and 34.41 MHz peg came from standalone/partial physical runs;
> they are not a current full-chip clock limit. A later full-lane run exceeded
> the available 15 GB host memory. See [current project status](project_status.md)
> for the sign-off boundary and [documentation index](README.md) for authority.

**Goal:** lower the post-PNR SFU critical path (the div/sqrt-primitive fmax floor,
29.06 ns = 34.41 MHz) by recutting the two primitives that co-bind it. Multiplicative
with every cycle lever. Byte-exact (timing-only change; golden model untouched).

## The floor (from [[dma_floor]])

Post-PNR the SFU fmax is a tight 3-way primitive cluster:
- `fp32_div_p5` STAGE 2 (`u_ln_var_norm.rB_P`) and STAGE 4 (`rD_P`) — the 6-iter
  restoring-divide middles.
- `fp32_sqrt_p6` STAGE 2 (`u_ln_sqrt.rB_r`) — the M_pad-build + 13-iter nonzero
  radicand region.

They co-bind: cutting one alone leaves the other as the floor, so both must land
together (memory: "next gain needs BOTH div_p6 AND sqrt-Mpad-restructure").

## Measurement method (15 GB box — no full-SFU PNR needed)

The prior fmax stamps used a full-SFU synth+OpenSTA flow (`sfu_fmax_flow.sh`).
That flow now OOMs: the cycle-fusion campaign grew `sfu_engine.sv` to ~92 KB and
the `synth -flatten` no longer fits 15 GB (full-chip PNR needs ≥24 GB).

But the floor is **primitive-internal** (register-to-register inside the divider /
sqrt; inputs and outputs are registered at the SFU boundary). So a **standalone
per-primitive** synth+STA measures it faithfully and fits trivially:
`/tmp/prim_fmax.sh <module> <sv>` = sv2v → `yosys synth -flatten -noshare` →
`abc -D 5000` (sky130 tt_025C_1v80) → OpenSTA `report_clock_min_period`.

Absolute numbers run ~3 ns higher than post-PNR (no placement / different abc
target), but the flow is a valid **relative** tool — verified monotonic:
div_p4=41.97 → div_p5=31.17 ns; sqrt_p4=36.05 → sqrt_p6=32.10 ns (deeper pipe =
faster, as expected). Old vs new measured identically ⇒ the floor drop transfers.

## The two changes

### 1. `fp32_sqrt_p6` — M_pad build moved to stage 1 (latency-neutral)

The even-exp radicand build (`exp_a_odd` mux + 50-bit `{sig_a,0}` assembly) was
done combinationally at the top of stage 2, ON the `rB_r` critical path. Moved it
into stage 1 (which only did unpack and had slack) and REGISTERED it (`rA_Mpad`,
50 bits). Stage 2 is now pure iteration. **Still 6 stages — LATENCY unchanged, so
zero SFU FSM change.** Arithmetic identical (same formula, one stage earlier).
- Standalone floor **32.10 → 29.64 ns** (−2.46 ns). Cost: one 50-bit register.
- Bit-exact: `test_fp32_sqrt_p6` (10,004,130 checks, 0 mismatches).

### 2. `fp32_div_p6` — 6-stage divider (new module)

Recuts the 29-iteration restoring divide from p5's ~6-iter stages into ~5-iter
stages. STA-tuned split `SPLIT=25/20/15/10/5` → **4 / 5 / 5 / 5 / 5 / 5 iters**
(stage 1 gets 4 since it also carries unpack; the five register-bounded middles
get 5 each). Same restoring algorithm as p5, one more pipeline register — value-
identical regardless of where the iteration chain is cut.
- Standalone floor **31.17 → 28.85 ns** (`24/19/14/9/4` measured 30.45; giving
  stage 1 fewer iters was the STA win).
- Bit-exact: `test_fp32_div_p6` (10,000,960 checks, 0 mismatches).
- The divider is now iteration-chain-bound with a large fixed per-stage cost;
  6→5 iters/stage bought only ~0.7 ns raw, then the stage-1-lightening tune got
  the rest. Deeper (7–8 stages) is deep diminishing — stopped at 6.

### SFU integration (div LATENCY 5 → 6)

`sfu_synth_datapath.svh`: the 4 pipelined dividers (`u_ln_mean`,
`u_ln_var_norm`, `u_ln_norm`, `u_sm_div`) → `fp32_div_p6`. `u_ln_sqrt` unchanged
(same module, restructured internally — drop-in).

`sfu_g2_compute.svh` + `sfu_engine.sv` re-tune for the extra divider stage:
- **Scalar divides** (mean, var_norm): one more wait state each — new enum states
  `F_G2_LN_MEAN_W5` (7'd75) / `F_G2_LN_DENOM_PRE_W5` (7'd76); sample y 6 cycles
  after the operand instead of 5.
- **Streaming drains** (LN-OUT `F_G2_LN_OUT_DIFF`, softmax-OUT `F_G2_SM_OUT_NORM`):
  pipe depth 1 feed-reg + 6 div stages = 7, so the collect threshold `iter>=6`
  becomes `iter>=7` (collect pointer = `iter_idx_q - 7`). No new state.

Cycle cost: +1 drain per scalar/streaming divide invocation — a few hundred cyc
per token, ~0.0004% of a b16 step. Effectively cycle-neutral.

## Result — fmax floor (standalone sky130 synth+STA)

| primitive        | baseline | lever E | Δ |
|------------------|---------:|--------:|---|
| fp32_div (min)   | 31.17 ns | 28.85 ns | −2.32 |
| fp32_sqrt (min)  | 32.10 ns | 29.64 ns | −2.46 |
| **cluster floor**| **32.10 ns** | **29.64 ns** | **−7.7% period / +8.3% fmax** |

## Result — cycles (measured, clean same-position A/B, b16 pos-511 ctx-512)

Baseline binary = git-stash of the lever-E tree (div_p5), rebuilt; both run
through the identical `bench_decode_cycles.py`:

| metric      | baseline (div_p5) | lever E (div_p6) | Δ |
|-------------|------------------:|-----------------:|---|
| step cyc    | 57,201,711 | 57,205,215 | **+3,504 (+0.006%)** |
| sfu_busy    | 10,125,202 | 10,128,706 | +3,504 (= step Δ exactly) |
| sys_busy    | 25,017,789 | 25,017,789 | **0 — byte-identical** |
| dma_beats   | 19,775,920 | 19,775,920 | **0 — byte-identical** |

sys_busy + dma_beats byte-identical ⇒ the change is **provably SFU-only**. Step Δ
== sfu Δ ⇒ the SFU serializes (only sys‖DMA overlap is legal), so the sole cycle
cost is the div-latency drains — **effectively cycle-neutral** (+0.006%).

## tok/s

Cycle-neutral ⇒ the whole gain is the +8.3% fmax multiplier (position-independent,
a pure clock scalar on a fixed cycle count):
- **pos-511** (measured here): 9.624 → **~10.42 tok/s**.
- **pos-510** (canonical waterfall reference, lever D 9.779): **9.779 → ~10.59
  tok/s** (+8.3%, ≈+280% over the 2.79 base).
- single-stream decode rides along ~×1.083 (≈1.63 → ~1.77).

SFU post-PNR fmax projects **34.41 → ~37.3 MHz** (×1.083). The full-chip PNR
re-stamp confirming the absolute MHz (and that no other SFU path becomes the new
binder) is deferred to a ≥24 GB box — this box OOMs the full-SFU flatten. The
relative floor drop IS measured here, and dma_floor establishes the SFU floor is
primitive-bound (the entire top-50 PNR endpoint tier was div/sqrt), so the drop
transfers. Same posture the memory records for the HW-blocked 6-fusion PNR stamp.

## Gate (all passed)

- `test_fp32_div_p6` 10.0M checks / 0 mismatch; `test_fp32_sqrt_p6` 10.0M / 0.
- `test_sfu_synth` 11/11 — LN `max_ulp=0`, masked-softmax `max_ulp=0`.
- `test_batched_decode` 7/7 incl. `test_batched_decode_rtl_matches_golden_bytes`
  (mode-1 synth RTL == mode-0 golden, byte-identical) — rebuilt run_program_synth.
- Golden model untouched ⇒ golden logits invariant (timing-only lever); 124M
  byte-exactness follows from the SFU-unit + tiny-cosim byte-match on the identical
  SFU RTL (no 124M-specific path is touched).
- 124M b16 same-position A/B: cycles +0.006% (sys/dma byte-identical) — no regression.

## Build note

`fp32_div_p6.sv` added to `rtl/common/filelists/core.f`. Rebuild the mode-1
binary with the core.f-derived source list (`-GSFU_SYNTH_MODE=1
-GDRAM_SIZE=1073741824`, `rm -rf build/run_program_synth` first — stale-binary
gotcha). No ISA change (lever E is RTL + primitives only).
