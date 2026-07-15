# T1 — the compiler overlap campaign, measured and re-scoped (2026-07-15)

**Bottom line: the T1 plan's four items were sized off beat counts and
schedule assumptions that the measured baselines contradict. Three of the
four are net-negative / already-done / tiny at b1. The one real lever is
item 1 (cross-group KV prefetch) and it pays at *b16*, not b1 — b1 is already
at its bandwidth floor.** Every number below is measured (`profile_decode_step`,
`--fast-beats`, pinned BW model) or read from the RTL, not estimated, except
where marked "est.".

## The b1 baseline (pos-511, 19,794,737 cyc, 1.738 tok/s — the plan's shape)

| engine | cyc | % | T1 reach |
|---|---:|---:|---|
| **sys** (systolic) | 11,110,829 | 56.1 | **FLOOR** — untouchable by T1 (`sys gap == busy + n_ops` exactly). tok/s ceiling ~3.1. Only T3 (clock) / T4 (width) move it. |
| **exposed DMA** | 5,886,116 | 29.7 | the only real prize — but see "the wall" below |
| **ctl** | 2,015,771 | 10.2 | mostly LOAD-startup latency, not a bubble (T0) |
| **sfu** | 638,624 | 3.2 | small |
| **helper** | busy 986,112 / **exposed 10,324** | 0.1 | **already hidden** |

Port-A audit: co-busy 4,148,317, lost 0. DMA beats 10,534,000 (4.15M co-busy, 6.4M exposed).

## Why each plan item is mis-sized at b1

- **Item 2a (weight prefetch): ALREADY IMPLEMENTED.** `w8a16_emit/matmul.py:605-643`
  (`pipeline_full_k`) double-buffers weight tiles for GPT-2's FC/attn layers and
  overlaps the next tile's DMA with the current MATMUL. Its own comment records the
  b1 result: *"the DMA total work + non-overlappable SFU is the floor."* **This is
  the bandwidth wall**: at M=1 the MATMUL is 16 rows — far too short to hide a full
  weight tile's DMA. The exposed 5.9M at b1 is largely **un-hideable**, not
  un-scheduled. So T1's planned b1 gain (→2.2-2.4) rests on overlap that is already
  done or physically impossible at M=1.
- **Item 3 (DMA-transpose K^T): NET-NEGATIVE at b1.** The helper K^T BUF_COPY
  transposes are 986K busy but only **10K exposed** — already hidden (`sys gap ==
  busy + n_ops` leaves no room for a helper stall). Replacing them with a DMA
  transpose-load *deletes hidden work* and *adds ~2× exposed transpose beats* (the
  stripe transpose is serial read-then-write). Strictly worse at b1. (May help b16 —
  denser schedule; TBD from the b16 trace.)
- **Item 4 (SET_ADDR dedup): CEILING ≪ plan's 0.6-0.7M.** T0 already found
  SET_ADDR_HI's 32.4 cyc/op is the *following LOAD's* startup latency, not a
  removable ctl bubble. Deduping removes the instruction fetch/dispatch (~4 cyc each)
  but the LOAD latency stays. Real ceiling is small.
- **Item 2b (fc2 input restaging): real but SMALL at b1 (~147K beats est.).** fc2 is
  the ONLY large-input-streaming matmul (K_pad=3072 can't full-K); at b1 it splits into
  **2 n-groups** (`_large_weight_tile_plan(3072,768,16)`), so the second group re-loads
  its 12 input k-tiles = ~12,288 beats/layer × 12 = ~147K beats. lm_head's 315 n-groups
  and c_attn/fc1/out_proj all use the *weight*-tiled path (input quantized once — NO
  restaging). So item 2b ≈ 0.7% of the step, not the plan's 2M.
- **The plan's "2M restaging" is actually M-PADDING waste, and it's a different lever.**
  At b1 M=1 but M_pad=16, so every M-padded *input* DMA loads 16 rows where 1 is real
  (15/16 padding). Truncating those loads to M real rows is **byte-exact** (the pad
  output rows are systolic junk that the m_exact DEQUANT already drops — same argument
  as lever C), and the activation-load share could be ~1-2M beats. It is NOT a
  concurrency change ⇒ the tiny RTL==golden gate is VALID (not blind). But it's a broad
  multi-emitter change and its *exposed* fraction is unmeasured (weights dominate b1 DMA
  at 7.75M; activation loads are the minority). Candidate, but measure exposure first.

## The port probe — item 1 is zero-RTL and port-safe (RTL-verified)

`sram_dp_sky130.sv` = `sky130_sram_1rw1r`: **Port A = read/write, Port B = read-only**,
independent. `sram_subsystem.sv`: the systolic reads src1 via shared **Port B**
(→ABUF), src2 via dedicated **Port W** (→WBUF Port B), and drains to ACCUM via
**Port S**. `s_collision = a_en && s_en && (a_buf == s_buf)` faults only when
channel A and channel S target the *same* buffer's Port A. A DMA prefetch of the
next head/group's K/V into **WBUF/ABUF (Port A)** runs concurrently with the
systolic reading the current tile (Port B/W) and draining to ACCUM (Port S) —
`a_buf ∈ {WBUF,ABUF}` vs `s_buf = ACCUM` ⇒ no collision, **serviced by the 1rw1r
macros.** Requirements: prefetch → WBUF/ABUF (not ACCUM); double-buffered addresses.
See [[t1-port-probe]], docs/porta_bus_split.md.

## Item 1 (cross-group KV prefetch) — the real lever, at b16 — CEILING CONFIRMED

Fresh b16 trace (pos-510): step 54.3M cyc / 10.145 tok/s; sys 22,321,853 (41.1%),
dma 19,844,527 (36.6%), sfu 10,074,569 (18.6%), ctl 2,010,323. **Total DMA
19,664,920 beats; exposed 15,533,729; co-busy 4,148,317.**

**The exposed-DMA decomposition (the gate the advisor required) is settled by the
b1→b16 delta:** exposed DMA grew 6.40M → 15.53M = **+9.13M**, and total DMA grew
10.53M → 19.66M = **+9.13M** — identical. So *every* extra b16 beat is exposed, and
that increment is the per-stream KV (b16 reads 16 streams' caches vs b1's 1; KV
geometry 512×64×12×2×12×16 = 9.44M). **⇒ ~9.4M of b16's exposed DMA is KV, fully
un-overlapped today. That is item 1's ceiling.** Hide it under the 22.3M systolic ⇒
**54.3M → ~45M → ~12.2 tok/s (+~20%).**

**The co-busy counter is CORRECT, not blind (resolves the b1==b16=4,148,317 alarm).**
It measures the weight-prefetch (`pipeline_full_k`) DMA‖MATMUL overlap, which is
batch-invariant (weights load once). The KV loads (`packed_attn.py:158-234`) load all
12 heads' K^T **serially, each SYNC-blocked, BEFORE the single packed MATMUL** — so
they overlap NOTHING and contribute **0 co-busy at both b1 and b16**. Adding 16× KV at
b16 therefore adds only exposed DMA, leaving co-busy pinned at 4,148,317. The counter
is a live per-cycle count (`taccel_top.sv:681`); the identity is physics, not a bug.
**Consequence: when item 1 overlaps the KV under the MATMULs, co-busy WILL rise from
4.15M toward ~13M — so it is a valid "shown-to-move" gate after all.**

**Validation (ground truth):** the item-1 win MUST show up as **total step cycles
dropping** (an independent, unimpeachable counter — re-serialization can't fake a
shorter step), plus byte-exact (tiny RTL==golden batch=16 + 124M SYS_DMA_OVERLAP A/B)
plus Port-A `lost=0` plus co-busy risen. Total-cycles is the primary metric; co-busy
is the corroborating denominator.

## What must be true before the item-1 build (the porta-scar gate)

Item 1 is a concurrency change; the tiny byte-exact gate has ZERO co-busy and is
**structurally blind** (this is exactly how the Port-A silent-corruption lived for
months — docs/porta_bus_split.md). Required gate, all three:
1. byte-exact — tiny RTL==golden (batch=16) **and** 124M SYS_DMA_OVERLAP A/B logits;
2. Port-A audit **lost = 0** (fresh);
3. co-busy **quoted and shown to MOVE** (item 1 should *increase* it — more overlap is
   the win; a byte-identical result with unchanged/collapsed co-busy is
   re-serialization in disguise). Counter is provably live (`taccel_top.sv:681`
   per-cycle on `dma_busy && sys_busy`); the b1≈b16 identity (4,148,317) is the
   batch-invariant decode schedule, not a stuck counter — VERIFY on the b16 fresh run.

## Recommended campaign order (revised)

1. **b1 is done for T1.** Its lever is T3 (clock — DMA scales with clock under the
   pinned model) or fewer model beats (W4, out of scope). Do not chase b1 overlap.
2. **Item 1 (cross-group KV prefetch) at b16** — the one real T1 win, ~+18% est.
   Build behind a flag, gate as above. This is the "2-3 week" hard core of T1.
3. Re-evaluate item 3 / item 2b at b16 *after* the b16 trace decomposition — they may
   flip positive in the denser b16 schedule.
