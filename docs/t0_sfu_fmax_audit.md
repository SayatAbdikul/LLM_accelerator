# T0.3 — SFU combinational-cloud STA audit (the T3 fmax gate)

**Date: 2026-07-15. Standalone sky130 (tt_025C_1v80) synth + OpenSTA, reg-to-reg
delay of flop-in/flop-out shells around the un-pipelined SFU clouds.**

## Why this audit exists

The chip's fmax is quoted at **34.41 MHz (29.06 ns)**, and the lever-E "29.64 ns
floor" was measured on the **div/sqrt primitives in isolation**. But the SFU has
combinational clouds that were **never in any timing report** — the full-SFU flatten
OOMs on this 15 GB box, so the softmax EXPSUM path (`add → exp → add` in ONE cycle,
`sfu_synth_datapath.svh:399-401`) has no STA number. The T3 plan (deepen div/sqrt to
reach 70–90 MHz single-domain) is only valid if div/sqrt is actually the binder.
This audit checks that.

## Flow is CALIBRATED (this is what makes the numbers trustworthy)

My flow — `sv2v → synth -flatten -noshare → abc -liberty -D 5000 → OpenSTA
report_clock_min_period`, reg-to-reg — reproduces the known committed number:

| target | my flow | committed | verdict |
|---|---:|---:|---|
| **fp32_div_p6** (worst pipeline stage) | **28.52 ns** | ~29.64 ns / 34.41 MHz | **matches** (synth-only, slightly optimistic vs PNR) |

So the flow is honest, and the numbers below are real, not artifacts. (`synth`'s SHARE
pass HANGS on fp32_add's many shifter/subtractor share candidates — the documented
`synth_share_oom` gotcha; `-noshare` is required. `abc -fast` is too pessimistic —
38 ns for the add; `abc -D 5000` delay-opt is the calibrated setting.)

## Measured

| cloud | reg-to-reg period | fmax | meaning |
|---|---:|---:|---|
| fp32_div_p6 (**calibration**) | 28.52 ns | 35.1 MHz | matches committed ~29.64 ns floor |
| **fp32_add** | **28.49 ns** (delay-opt) | 35.1 MHz | one FP add ≈ the ENTIRE 29 ns budget |
| **fp32_mul** | **27.34 ns** (delay-opt) | 36.6 MHz | one FP mul, same |
| **fp32_exp** | **~412 ns** (556 ns via -fast / 1.35 pessimism) | **~2.4 MHz** | ~16 serial fp ops (Cody-Waite + Horner) |
| **EXPSUM** (`add→exp→add`, **1 cycle**) | **~490 ns** (658 ns via -fast / 1.35) | **~2 MHz** | the softmax accumulate; single-cycle in the synth datapath |

(delay-opt `abc -D 5000` completed for add/mul/div_p6; it TIMED OUT at 400 s on the
big exp/expsum clouds, so those are `abc -fast` numbers divided by the 1.35 pessimism
factor measured on the add — a real estimate, not a bound. Direction is unambiguous.)

## THE FINDING — pipelining `fp32_exp` is the #1 fmax blocker, AHEAD of div/sqrt

The softmax **EXPSUM path is ~490 ns single-cycle** (`reg → fp32_add → fp32_exp →
fp32_add → reg`, verified in RTL at `sfu_synth_datapath.svh:399-401`), and it is
**entirely the exp**: 28 (a−max) + **~412 (exp)** + 28 (accumulate). `fp32_exp` is an
un-pipelined ~16-serial-op combinational cloud (Cody-Waite + degree-6 Horner) whose own
header flags pipelining as deferred **"Phase 2.8 work."** This path has never been in a
timing report — the full-SFU flatten OOMs (PNR deferred to ≥24 GB), and the committed
**34.41 MHz is a div/sqrt-PRIMITIVE STA that structurally excludes this cloud.**

**This is CONTINGENT, not fiction — and it re-orders T3, it does not delete it:**

1. **34.41 MHz is achievable once exp is pipelined; today, unpipelined, the softmax
   path would not close.** Pipeline `fp32_exp` into ~7–16 stages and restructure the
   EXPSUM accumulate (feed exp 1 elem/cyc, collect N later — the exact transform the
   softmax-OUT divider drain already uses, byte-exact): the pre-sub add lands in the
   first pipeline stage and the accumulate add after the last, so **they are never
   combinationally chained with exp again. The worst single-cycle path drops to ONE
   fp32_add ≈ 28.49 ns — UNDER the 29.06 ns (34.41 MHz) budget.** So the peg is
   contingent on known-deferred work, not wrong. (Cycle counts and byte-exactness are
   timing-agnostic and entirely unaffected — only the ns→tok/s peg is contingent.)

2. **div/sqrt deepening is not the wrong path — it is the path AFTER exp.** Post
   exp-pipelining the binder is `max(add 28.5, div/sqrt 29.6) = div/sqrt`. So T3
   proceeds as written, with **"pipeline fp32_exp + restructure EXPSUM" inserted as its
   prerequisite step 0.** div/sqrt deepening to 12–15 stages then does its job.

3. **The single fp32_add passes the current budget (28.49 < 29.06 ns) — no add split is
   needed at 34.41 MHz.** An add split (2–3 stage carry-select/normalize, uniform &
   bit-exact) is only for the **70–90 MHz stretch**, where the add goes co-critical with
   div/sqrt. Whether it is needed at all depends on the add's true delay (below).

## The open reconciliation (decides the STRETCH scope, not the finding)

A single fp32_add at ~28 ns is in tension with the committed **"helper 109 MHz"** — a
single-cycle fp mul→add would be ~55 ns / ~18 MHz, so either that number predates the fp
datapath, or my isolated single-primitive shell is abc-pessimistic (div_p6 calibrates
the flow for *pipelined divide* logic, not a standalone barrel-shifter add). **One
experiment settles it — synth `blocking_helper_engine` through the same flow:**
- **~55 ns** → add ≈ 28 ns confirmed, 109 MHz impeached (stale / non-fp path), and the
  add split IS needed for the 70–90 MHz stretch.
- **~9 ns** → the isolated shell is pessimistic, add ≈ 9 ns, the SFU is div/sqrt-bound
  after exp-pipelining, and T3-as-written needs no add work at all.
Either outcome leaves the core finding (exp first) intact.

## Caveat (honest scope)

Synth-only, no PNR, no wire load; the flow is calibrated on div_p6 (28.52 vs committed
29.64 ns) but that only validates it for divide logic, not the standalone add. add/mul/
div_p6 are delay-opt `abc -D 5000`; **exp (556) and expsum (658) are `abc -fast` upper
bounds ÷ the 1.35× add pessimism factor → ~412 / ~490 ns estimates, not mapped delay-opt
numbers** (those timed out at 400 s). The exact exp multiple wants the deferred ≥24 GB
full-SFU PNR; but even at 3× optimism the ~16-serial-op exp cloud is ≫29 ns single-cycle,
so "pipeline exp first" is not in doubt. Flow + shells: `scratchpad/run_sfu_sta.sh`,
`sta_shells.sv`.
