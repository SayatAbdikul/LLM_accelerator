# Phase 0 — measure + audit

Cheap, zero-risk, and it **re-ranks the whole plan**. Two results:

1. **The 38.7% "other" is 91% exposed DMA.** Not the helper engine — the helper is
   0.14% of the step. The plan's guess that a third disjoint engine was hiding in
   there is refuted.
2. **⚠️ The DMA‖Systolic lever (`163f0cb`) silently corrupts matmul results.** The
   systolic loses the shared Port-A bus to the DMA and has no way to know. Measured
   on GPT-2 124M b16 decode: **255,818 dropped `ST_DRAIN_WR` writes per step** —
   real accumulator results, gone. Nothing in the design detects it, and the
   byte-exact test gate **cannot** see it.

Everything below is measured on the mode-1 synth RTL at `--fast-beats`, GPT-2 124M,
b16 pos-510 — the exact configuration the 9.729 tok/s headline comes from.

---

## 1. Where the b16 step actually goes

The instrumented binary reproduces the known step **to the cycle** (56,588,361), and
`sfu_busy` / `sys_busy` / `dma_beats` are byte-identical to the pre-instrumentation
run — so the counters are observation-only, not a perturbation.

The core is in-order, so the cycle gap between consecutive retire events is an *exact
partition* of the run. It spans 56,588,355 of 56,588,361 cycles (99.99999%; the
remainder is pre-first-retire).

| engine | cycles | share | |
|---|---:|---:|---|
| **systolic** | 24,640,500 | **43.5%** | |
| **DMA (exposed)** | 19,844,527 | **35.1%** | ← this *is* the "other" |
| **SFU** | 10,074,569 | 17.8% | |
| control | 2,010,323 | 3.6% | SYNC + address setup + CONFIG |
| helper | 76,800 | **0.14%** | |

The plan's unexplained 38.7% decomposes as 35.1 + 3.6 + 0.03 — **it was almost
entirely exposed DMA all along.**

**The helper is a dead end.** `busy_cycles.helper` = 76,800 cycles = 0.14% of the
step, all of it `BUF_COPY`. There is nothing to win there. (`bench_decode_cycles.py`
had been dropping this counter on the floor, which is why it looked like a suspect.)

**The DMA is ~100% exposed.** 19,664,920 beats against 19,844,527 exposed cycles —
a ratio of 1.009. At `--fast-beats` (1 beat/cycle) the DMA is running flat-out and
hiding under almost nothing. Only 4,148,317 cycles (7.3% of the step) overlap the
systolic; the rest is naked. `LOAD` alone is 19,534,121 cycles across 21,405 ops
(912.6 cyc/op).

This **raises** Phase 2's ceiling: the prize is not a sliver, it is a third of the
step.

### b1 (single-user latency) — 21,075,726 cyc = 1.633 tok/s

| engine | cycles | share |
|---|---:|---:|
| **systolic** | 12,391,812 | **58.8%** |
| **DMA (exposed)** | 6,019,189 | **28.6%** |
| control | 2,015,771 | 9.6% |
| helper | 986,112 | 4.7% |
| SFU | 638,624 | **3.0%** |

Two things flip relative to b16. The systolic **dominates** (58.8%), because b1 wastes
15/16 mesh rows. And the **SFU all but vanishes** (3.0%, down from 17.8%) — lever C's
`m_exact` did its job. So **Phase 2's SFU‖DMA overlap is worth almost nothing at b1**;
what matters at b1 is the systolic and the 28.6% of exposed DMA.

### Method note: the partition is checked, not asserted

For each engine, `sum(retire gaps of its opcodes) == busy_cycles[engine] + n_ops`
(the `+1` per op is the dispatch cycle). It holds **exactly**:

| | gap total | busy counter | n_ops | |
|---|---:|---:|---:|---|
| sfu | 10,074,569 | 10,059,586 | 14,983 | exact |
| sys | 24,640,500 | 24,636,477 | 4,023 | exact |

It does *not* hold for the helper — `BUF_COPY`'s successors are non-blocking `ctl`
ops, so the helper's tail overlaps them and lands in `ctl`. For the helper the busy
counter is authoritative. `software/tools/profile_decode_step.py` enforces this
identity and fails loudly if an opcode is filed under the wrong engine. (Its
scratchpad predecessor had **no helper class at all** — every helper op was silently
counted as control.)

---

## 2. ⚠️ The Port-A hazard: DMA‖Systolic is not safe

### What the RTL actually does

`taccel_top.sv` arbitrates one shared Port-A bus with a **fixed-priority mux and no
backpressure**:

```systemverilog
assign sram_a_en = helper_sram_a_en ? helper_sram_a_en
                 : sfu_sram_a_en    ? sfu_sram_a_en
                 : dma_sram_en      ? dma_sram_en
                                    : sys_sram_a_en;   // <-- loses to everyone
```

Neither `dma_engine.sv` nor `systolic_controller.sv` has a grant / ready / stall
input. When both assert, **the DMA wins and the systolic's access is silently
dropped** — `drain_grp_q` advances anyway, so the row is never retried. The RTL
already encodes this fact one line down, in the fault mask:

```systemverilog
(sys_sram_a_en & ~helper_sram_a_en & ~sfu_sram_a_en & ~dma_sram_en & sram_a_fault)
```

`163f0cb` moved only the systolic's **weight read** to the dedicated Port W. The
systolic still drives shared Port A in six states — src2 stream reads (when src2 is
not WBUF, i.e. all of attention), the drain RMW read, the **ACCUM drain write**, and
the **ACCUM preclear write**.

`obs_forbidden_overlap_violation_q` has **no `dma_busy && sys_busy` term** — by
design, since DMA‖Systolic is the one *legal* concurrency. So the drop is invisible.

### Measured — 124M, b16, pos-510

| | cycles |
|---|---:|
| `dma_busy && sys_busy` co-busy (the denominator) | 4,148,317 (7.3% of step) |
| **systolic Port-A accesses LOST** | **631,408** |
| — of which writes | 631,408 (100%) |
| — target ABUF / WBUF | 0 / 0 |
| — target ACCUM | 631,408 |
| &nbsp;&nbsp;• **preclear** (`ST_DST_CLEAR_WR`) — harmless | 375,590 |
| &nbsp;&nbsp;• **drain** (`ST_DRAIN_WR`) — **corrupts** | **255,818** |
| `forbidden_overlap_violation` | **false** |

The split matters enormously. A dropped **preclear** write only loses zeros that
`ST_DRAIN_WR` overwrites unconditionally — harmless. A dropped **drain** write loses
a real accumulator result *permanently*: the drain runs once per (m,n) output tile
(the K-loop re-enters at `ST_A_LOAD_REQ`, never `ST_INIT_TILE`), and `flags=1`
read-modify-write is not used anywhere in this bundle.

**255,818 real results per step are dropped on the floor.**

Zero ABUF and zero WBUF losses, exactly as the structure predicts: when src2 is WBUF
the systolic reads Port W, and attention's ABUF reads happen when the DMA is idle.
So the collision is *entirely* DMA-writes-WBUF versus systolic-writes-ACCUM — two
**different** SRAMs, colliding only because they share one bus.

**b1 shows the identical counts** — 4,148,317 co-busy, 631,408 lost, 255,818 dropped
drains, to the cycle. That is not a coincidence: the collision is the *weight*
prefetch racing the ACCUM drain, and neither the weights nor the dense-matmul tile
geometry (`M_pad = 16` either way) changes with batch. **The whole 124M decode path is
corrupt at every batch size**, and b1 is proportionally worse — the same 4.15M co-busy
cycles are 19.7% of a b1 step versus 7.3% of a b16 step.

### Why every test passes: the byte-exact gate is blind

`test_batched_decode` proves mode-1 RTL == mode-0 golden **byte-for-byte** at b16, and
it passes. It cannot fail. On the **tiny** model:

```
dma||sys co-busy cycles          0
systolic Port-A LOST             0
logits RTL vs golden:  0 / 1,536 bytes differ
```

**The tiny model never overlaps the two engines at all** — zero co-busy cycles — so
the byte-match passes no matter how corrupt 124M is. And 124M byte-match is itself
ill-posed (`rtl_cosim.py` #109: past the first fp16 overflow the golden saturates
too), so it was never gated there either. The bug sat in a blind spot between the two.

This is the deeper lesson: **a zero-drop count means nothing without the co-busy
denominator.** That is why the counter ships with one.

### Proof that it changes the answer — not just the bus

A dropped-write counter is a fact about the *bus*, not yet a fact about the *model*:
the drain covers the **padded** tile region, so in principle every lost row could be
padding nobody reads. Settled by A/B — the same program binary on two RTL builds
differing in exactly one parameter (`SYS_DMA_OVERLAP`), with a validity gate proving
both executed identical work:

| | cycles | `sys_busy` | retired | dma beats | dropped drain |
|---|---:|---:|---:|---:|---:|
| **shipped** (`=1`) | 56,588,361 | 24,636,477 | 210,602 | 19,664,920 | **255,818** |
| **serialized** (`=0`) | 60,736,113 | 24,636,477 | 210,602 | 19,664,920 | 0 |

`sys_busy` / `retired` / `dma beats` are **identical**, so both machines really do run
the same program — the serialized arm only *delays* dispatch. And yet:

> **1,605,138 of 1,608,704 logits bytes differ — 99.78%.**

The dropped drain writes are live model state. **The shipped 124M b16 decode path
computes garbage.**

**The honest decode number is 9.06 tok/s, not 9.729.** The 9.729 headline is measured
on a machine whose matmul results are 99.78% wrong; a *correct* machine today costs
+7.33% (60,736,113 cycles). That 7.33% is the true value of the DMA‖Systolic lever —
it is a real win, it just has to be *earned* rather than stolen.

### The design already knew

`control_unit.sv`, immediately above `OP_MATMUL`:

> *"DST_CLEAR uses SRAM port A; the helper engine holds higher SRAM priority, so
> MATMUL must not be dispatched while helper is busy **or DST_CLEAR writes would be
> silently dropped**."*

The hazard was understood, and guarded — **for the helper**. The DMA has the *same*
priority relationship over the systolic on the *same* bus, and got no such guard when
`163f0cb` legalized DMA‖Systolic.

### A second, latent bug found on the way

`control_unit.sv` maintains the **dispatch** condition (combinational) and the
**retire / PC-advance** condition (sequential) as *two separate expressions that must
agree*. When they disagree, the instruction retires **without dispatching** — the
systolic FSM only accepts `dispatch` in `ST_IDLE`, so the pulse evaporates and the
matmul is silently skipped, with no fault, no counter, and an unchanged
`retired_instructions`.

The first `SYS_DMA_OVERLAP=0` build hit exactly this (it gated `sys_dispatch` but not
the retry) and quietly skipped matmuls — which looked, seductively, like the correct
machine being *faster*. Both conditions are now written to mirror each other, with a
comment saying so. **This is why the A/B carries a validity gate instead of trusting
the logits diff.**

---

## 3. What this does to the plan

**Phase 2 (split the Port-A bus per buffer) is no longer an optimization. It is the
fix for a live correctness bug, and it must come first.**

The good news is that the fix is exactly the one already scoped: ABUF / WBUF / ACCUM
are **already three separate dual-port SRAMs** (`sram_subsystem.sv:88-153`) behind
one shared `a_en/a_buf/a_row/a_wdata` bus. Splitting the bus per buffer costs **zero
new SRAM ports** and resolves 100% of the measured losses by construction — every one
of them is DMA→WBUF racing systolic→ACCUM, i.e. *different* SRAMs.

**Lever 1a (delete the dead ACCUM preclear) is now entangled.** It is still dead work
(`ST_DRAIN_WR` overwrites the region bijectively, the golden never precleared, and
`flags=1` count is zero), but 375,590 of the current collisions land on preclear
writes. Removing it re-phases the systolic against the DMA and changes which writes
collide. **1a must not land before the bus is fixed**, or it will silently move
corruption around.

**A `SYS_DMA_OVERLAP` parameter now exists** (`control_unit.sv`, default 1). Setting
it to 0 serializes the two engines: correct, and the honest reference for A/B-ing the
corruption. It is also the **safe-mode fallback** — a correct machine today, at
9.06 tok/s, if one is needed before the bus split lands.

### Why the fix is the one already scoped

Every single lost access is **DMA→WBUF racing systolic→ACCUM** (ABUF losses: 0, WBUF
losses: 0). Those are **different SRAMs**. They collide only because they share one
bus, and ABUF / WBUF / ACCUM are **already three separate dual-port macros**
(`sram_subsystem.sv:88-153`, where `a_en` is merely fanned out by `a_buf`).

So splitting Port A per buffer costs **zero new SRAM ports** and eliminates **100% of
the measured losses by construction** — while keeping the 7.33% the lever was trying
to buy. It is simultaneously the correctness fix and the performance lever.

### Revised ranking

| | was | now |
|---|---|---|
| **Split the Port-A bus** | Phase 2, "+15-20%" | **PHASE 1 — CORRECTNESS FIX.** Also recovers the 7.33% legitimately, and the DMA is 35% of the step, so the ceiling above that is higher than estimated |
| Delete the ACCUM preclear (1a) | Phase 1, +4.3% | **BLOCKED on the bus fix.** 375,590 of today's collisions land on preclear writes; removing it re-phases the systolic against the DMA and moves corruption around |
| Chase the helper engine | suspected 3rd engine | **DEAD — 0.14% of the step** |
| fc2 full-K (1b), A-load hiding (1c), softmax MAX 8-wide (1d) | Phase 1 | unchanged, still independent |

**Every perf number in `docs/` for 124M decode was measured on the corrupting build.**
The *cycle counts* remain valid (cycle counts are data-independent), so the levers'
relative gains still hold. The *tok/s* figures should be restated against the correct
baseline, and any model-quality claim (perplexity / argmax conformance) at 124M b16
must be re-run once the bus is fixed.
