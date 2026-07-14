# The Port-A bus split — fixing the silent matmul corruption

**Status: LANDED.** This is the correctness fix Phase 0 called for
(`docs/phase0_measurement.md`). It is not primarily a performance lever, though it
does legitimately recover the 7.33% that the DMA‖Systolic overlap was "buying" by
throwing away results.

---

## What was broken

`taccel_top` arbitrated **one** shared Port-A bus between four engines with a
fixed-priority mux and **no backpressure**:

```
helper > sfu > dma > systolic
```

The systolic sat at the bottom, and its port list has **no grant / ready / stall
input**. So when the DMA asserted Port A in the same cycle, the systolic's access was
simply *not selected* — and its FSM advanced anyway (`drain_grp_q` increments
unconditionally). The write was **discarded, permanently, with no fault and no
counter**.

Measured on GPT-2 124M, b16, pos-510:

| | |
|---|---:|
| `dma_busy && sys_busy` co-busy cycles | 4,148,317 |
| systolic Port-A accesses **lost** | **257,406** |
| — all of them writes, all to ACCUM (`ST_DRAIN_WR`) | |
| logits bytes differing vs a correct machine | **1,605,138 / 1,608,704 (99.78%)** |
| argmax agreement with the golden model | **0 / 16 streams** |

Those were real accumulator results. **Every matmul on the 124M decode path was
corrupt, and the chip emitted broken text.**

### The engines were never contending for memory — only for a wire

This is the whole point. ABUF / WBUF / ACCUM are **three separate dual-port SRAM
macros** (`sram_subsystem.sv`), each with its own physical Port A. Every single lost
access was **DMA→WBUF racing systolic→ACCUM** — *different macros*. They collided
only because one bus fed all three sets of `a_*` pins.

### Why nothing caught it

The byte-exact gate is the **tiny** model, which has **ZERO co-busy cycles** — it
never overlaps the two engines, so it is *structurally incapable* of reaching the
bug. And 124M byte-match is separately ill-posed (`rtl_cosim` #109: past the first
fp16 overflow the golden itself saturates). The bug lived in the gap between them.

> **The lesson, and it generalizes:** never accept "bit-exact" for a concurrency
> change without reporting the **co-busy denominator**. A zero loss count on a run
> where the two engines never overlapped proves exactly nothing.

---

## The audit that decided the design

A per-buffer fan-out resolves *different*-buffer contention completely. It does
**not** resolve *same*-buffer contention. So the design hinged on one number: how
many co-accesses are same-buffer?

New counters decompose DMA-vs-systolic Port-A **co-access** (both engines want the
bus, regardless of who wins) by whether they want the same macro:

| shape | co-busy | **same buf** | diff buf | lost |
|---|---:|---:|---:|---:|
| decode b16 pos-510 | 4,148,317 | **0** | 257,406 | 257,406 |
| chunked-prefill P=16 base-256 | 4,148,317 | **0** | 257,406 | 257,406 |
| chunked-prefill P=16 base-0 | 4,148,317 | **0** | 257,406 | 257,406 |

**`same_buf == 0` everywhere.** And — this is the part that makes the zero mean
something — the *marginals* prove the counter is not blind:

| shape | systolic→Port-A@ABUF | DMA→Port-A@ABUF |
|---|---:|---:|
| decode b16 | **0** | 2,276,928 |
| chunked-prefill base-256 | **0** | 1,655,256 |

The DMA drives Port-A@ABUF heavily. **The systolic never does — zero cycles.** So the
same-buffer hazard is not *avoided by luck*; it is **structurally absent**:

- the systolic's src2 stream reads route to the dedicated **W port** when src2 is in
  WBUF, and in the shipped int8-KV path (lever A) both K^T and V *are* in WBUF
  (`w8a16_emit/attention.py:245,309`);
- so the systolic's shared Port A carries **only** the ACCUM drain.

> ⚠️ **This is a property of the current configuration, not a law.** The legacy
> non-int8-KV branch (`attention.py:467`) puts V in **ABUF**, which would put the
> systolic's src2 stream back on Port A and make same-buffer contention live. That is
> exactly what the fault below exists to catch.

*(Phase 0 originally warned that the KV re-read (DMA→ABUF) races attention's src2
(systolic←ABUF). The premise was wrong — attention's src2 is in WBUF. The warning is
withdrawn, and it was withdrawn by measurement, not by assumption.)*

---

## The fix

**Fan Port A out per buffer.** Zero new SRAM ports — the macros were already there.

1. `sram_subsystem.sv` gains a second Port-A-class channel, **Port S**, dedicated to
   the systolic. Each macro's physical Port A is driven by whichever channel targets
   it:

   ```
   A owns X  <=  A is enabled and targeting X
   S owns X  <=  S is enabled and targeting X, and A is NOT targeting X
   ```

   Different buffers ⇒ **both proceed in the same cycle**. That is the fix.

2. `taccel_top.sv` removes the systolic from the shared priority mux entirely and
   wires it to Port S.

3. **Same-buffer collision raises `s_collision` ⇒ architectural FAULT.** A collision
   is *always* corruption today (a silently dropped access), so faulting can only
   convert existing corruption into a loud halt — it cannot break a path that
   currently works. It also guards the configs above: if a future schedule or a
   non-int8-KV bundle reintroduces src2-in-ABUF, the machine **halts** instead of
   quietly computing garbage.

4. **The OOB fault moves with the channel.** It used to be masked behind
   `~dma_sram_en` — an out-of-bounds systolic address was invisible *precisely when
   the DMA was active*, which was itself a symptom of the shared bus. On Port S it is
   unconditional.

5. **The Port-S read return is wired** (`s_rdata`, own registered buf-select), even
   though the systolic reads nothing on it today. `ST_DRAIN_RD` (the `flags=1`
   read-modify-write) is a real ISA feature that the 124M bundle simply never uses.
   Wiring a return path only where the current schedule happens to need it is
   *precisely* the "correct by schedule, not by construction" mistake that produced
   this bug. `test_systolic`'s `matmul_accumulate_flag` covers it.

---

## Result

**The corruption is gone, and the overlap is kept.**

| 124M b16 pos-510 | before | after |
|---|---:|---:|
| logits vs a correct (drop-free) machine | 99.78% of bytes differ | **BYTE-IDENTICAL** |
| systolic accesses lost | 257,406 | **0** |
| same-buffer collisions (faults) | — | **0** |
| **co-busy — the concurrency itself** | 4,148,317 | **4,148,317** |
| step cycles | 54,269,714 | 54,269,714 |

The third row is the one that matters for honesty: co-busy is **unchanged**, so the
two engines still overlap for 4.15M cycles per step. The fix works by letting them
*run concurrently on different macros* — **not** by quietly re-serializing them. (A
byte-identical result with collapsed co-busy would have been `SYS_DMA_OVERLAP=0` in
disguise, and would have thrown away the 7.33%.)

**A correct machine went from 58,417,466 cycles to 54,269,714** ⇒

| | tok/s @ 34.41 MHz |
|---|---|
| correct, before the split (serialized) | 9.424 |
| **correct, after the split** | **10.145** |

**+7.65%, and this time the number is real.** The old 10.145 was measured on a chip
that emitted garbage.

### fmax — measured, not argued

The split **adds** logic to the SRAM request path (per-buffer decode + the `s_own_*`
term + a 2:1 mux on `{en, we, row, wdata}`), so 1a's "only removes logic" waiver does
**not** apply. Standalone sky130 synth + STA of `sram_subsystem`'s request cone
(macros shrunk to 16 rows so STA has real endpoints; the decode/arbitration logic
under test is depth-independent):

| | min period | fmax |
|---|---:|---:|
| before | 2.56 ns | 391 MHz |
| after | 2.80 ns | 357 MHz |

**+0.24 ns — one mux level.** The chip floor is **29.64 ns**, set inside the SFU
(fp32 div/sqrt). At 2.80 ns this path sits **10.6× under the floor** and cannot
become critical. Chip fmax is unaffected.

### Verification

- **124M A/B** (`SYS_DMA_OVERLAP=1` vs `=0`): logits **byte-identical**, 1,608,704 /
  1,608,704. This is the gate that matters, and it is RTL-vs-RTL so the golden's
  fp16-saturation ill-posedness does not apply.
- Port-A audit re-run on **all three shapes**: lost = 0, collisions = 0.
- `test_batched_decode` + `test_multi_token_prefill`: **20/20** (RTL == golden on the
  tiny model).
- `test_systolic` **8/8** (incl. `matmul_accumulate_flag` = the Port-S read path),
  `test_systolic_chained` 7/7, `test_sfu_synth` 11/11, `test_dma` 29/29,
  `test_helpers` 19/19.
- `test_control` fails at `masked_softmax_without_config_attn` — **pre-existing**,
  reproduced on clean HEAD with the split stashed. Not a regression.

`SYS_DMA_OVERLAP=0` is retained as the drop-free A/B reference and safe-mode
fallback (`make run_program_noovl`).
