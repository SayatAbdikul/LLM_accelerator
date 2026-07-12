# Performance roadmap — tokens/sec levers, post-Phase-2 (2026-07-10)

Successor to `perf_roadmap_2026-07-08.md`, grounded in the Phase-2 batched-decode
measurement (commit `2d877be`) plus a fresh per-opcode retire-gap profile of the
batch-1 decode step at HEAD `7291fd5` (mode-1 synth RTL, honest-BW `--fast-beats`,
34.41 MHz). Batch-16 profile run in flight; its aggregate counters are the
committed ones. All shapes: GPT-2 124M, decode compiled for the ctx-512 budget
(`key_len=513`, `Kseq_pad=528`, `valid_kv_len` patched to 512).

**Measured state:**

| shape | cyc/step | cyc/tok | tok/s | sfu | sys | helper | dma beats |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode B=1, pos 511 | 33,868,695 | 33,868,695 | 1.016 | 11.89M | 12.39M | 1.90M | 11.14M |
| decode B=16, pos 510 | 197,138,919 | 12,321,182 | **2.792** | 106.81M | 35.51M | **28.53M** | 28.78M |

Verified this session: the rebuilt HEAD binary reproduces the b1 33,868,695
**exactly**; the b16 re-measurement lands −0.07% vs the committed 197,274,279
(248,658 vs 249,810 insns — sub-0.5% bundle-emission variation, immaterial).
The per-opcode profiles (below) match the analytic RTL cost model to a few %.

---

## 0. UPDATE (same day) — lever A LANDED (commits `931b373` A1, `4163a08` A2)

Store-time int8 KV (A1: `kv_quant` node + int8 loads, KV DRAM/DMA halved,
per-core K/V window QUANTs gone) + V kv_load direct DRAM→WBUF (A2: per-core
helper V copy gone). Bit-exact end to end: pre/post golden logits byte-diff
12/12 identical per commit (tiny + 124M, b1 + b16, 3 steps each at the bench
window shapes), mode-1 RTL == mode-0 golden byte-match on the batched program,
full suite has zero new failures vs a clean-HEAD baseline (16 pre-existing
failures reproduce identically on unmodified HEAD — fixture SHA drift,
output-aware-search `n_embd` KeyError, W4 tile_config tuple, fp16-embedding-era
synthetic tests; 3 KV-contract tests updated to the int8 contract).

**Measured after lever A (same protocol as the baseline table):**

| shape | cyc/step | cyc/tok | tok/s | Δ vs base | sfu | sys | dma beats |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode B=1, pos 511 | 30,566,487 | 30,566,487 | **1.126** | **+10.8%** | 10.13M | 12.39M | 10.53M |
| decode B=16, pos 510 | 145,048,383 | 9,065,524 | **3.796** | **+36.0%** | 78.55M | 35.51M | 19.33M |

Closure (b1): A1 −2,387,232 cyc — sys_busy unchanged to the cycle
(12,390,189), dma −609,408 beats == prediction exactly (−573,696 halved KV
loads, −34,560 deleted zero-fills, −1,152 halved stores), sfu −1.77M (288
per-core 528-row K/V QUANTs out, 288 single-tile kv_quants in). A2 −914,976
cyc with sfu/sys/dma all bit-identical to A1 — pure helper+sync: 144 V
BUF_COPYs (−608k busy) plus lighter `_compact_abuf` churn (V no longer
stages 33 KB in ABUF). Program sizes: b1 94,878→93,726 insns, b16
248,658→221,634; b16 KV 604→302 MB (bundle 971→666 MB).

---

## 0.1 UPDATE (same day) — lever C LANDED (`1bb5f51` C-2 ISA/RTL/golden, `6bc4dd7` C-3a attention core, `85e181e` C-3b shared ops)

`m_exact` (12-bit exact SFU row count in CONFIG_TILE[27:16]; 0 = full-tile
legacy): the SFU row loops walk the real rows instead of the 16-row-quantized
tile count. C-2 wired ISA/RTL/golden in lockstep — SFU-only, systolic MATMUL +
helper ignore the field, and m_exact=0 keeps every pre-existing bundle byte-
identical. C-3a stamped the per-head attention core (QK^T / masked-softmax / AV
+ store-time `kv_quant`); C-3b the per-token dynamic-quant matmuls (MAX_ABS /
QUANT / DEQUANT across all three lowering paths — simple, large-weight-tiled,
large-input-streamed — including the fc2 act-quant f=3 reshape) plus LN / GELU /
residual VADD. Bit-exact per commit: golden logits byte-diff 12/12 identical
(tiny + 124M, b1 + b16), mode-1 RTL == mode-0 golden byte-match, full suite
zero new failures vs the 16 pre-existing.

**Measured after lever C (same protocol as the baseline table):**

| shape | cyc/step | cyc/tok | tok/s | Δ vs A | Δ vs base | sfu | sys | dma beats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode B=1, pos 511 | 21,075,507 | 21,075,507 | **1.633** | **+45.0%** | **+60.7%** | 0.63M | 12.39M | 10.53M |
| decode B=16, pos 510 | 76,550,463 | 4,784,404 | **7.192** | **+89.5%** | **+157.6%** | 10.06M | 35.51M | 19.33M |

Closure: sys_busy AND dma_beats are invariant to the cycle on BOTH shapes
(systolic + DMA untouched — m_exact is SFU-only), so every delta is pure SFU
row-walk. b1 sfu 10.13M → 0.63M (−93.7%: the attention core + every per-token
dynamic-quant epilogue shed 15/16 of their walk). b16 sfu 78.55M → 10.06M
(−87.2%, entirely C-3a — C-3b is a no-op at b16 because the 16 batched streams
fill M_pad, so M==M_pad → m_exact=0 on every shared op). Program byte-count
unchanged (m_exact rides existing CONFIG_TILE immediates): b1 93,726 insns,
b16 221,634. SFU is now near-eliminated for b1; **sys_busy (systolic) is the
new floor on both shapes** → the next cycle lever is B (packed attention) or
the systolic/K^T-transpose path (D), not the SFU.

---

## 0.2 UPDATE — lever B FOUNDATION landed (`dabbeb9`), B2 pending

Block-diagonal **packed QK^T**: the 12 per-head QK^T systolic matmuls of one
(layer, stream) collapse into ONE matmul `Q_pack @ K_all^T`, where `Q_pack`
(16, d_model) is block-diagonal (head h's INT8 query at row h, cols
[d_head·h, d_head·h+d_head), zeros elsewhere) and `K_all^T` (d_model, key_len)
stacks the 12 per-head transposed K caches at block rows. Row h of the INT8
matmul = head h's INT32 scores (the 704 zero columns add exactly nothing), so
a per-head dequant of ACCUM row h reproduces the exact per-head score tile;
softmax / attn_v / gather / concat are UNCHANGED. Pure compiler change — no
ISA / RTL / golden touch. Byte-exact: golden 12/12 (tiny+124M, b1+b16), tiny
mode-1 RTL == mode-0 golden byte-match, zero new test failures.

**Two roadmap corrections proven this pass:**
- **AV packing is SYSTOLIC-NEUTRAL.** per-head AV = 4 n-tiles ×192 = 768
  tile-ops/layer; packed AV = 48 n-tiles ×16 = 768 tile-ops/layer — identical,
  and packed computes 11/12 junk off-diagonal blocks. It also needs a V
  column-interleave into (key_len, d_model) that BUF_COPY can't do (no strided
  scatter). So lever B is **QK^T-only**; AV stays per-head. The entire
  attention-systolic win (~24.7M → ~13.1M/step, the QK^T share) is in QK^T.
- **Softmax packing (12→1, m_exact=n_head) is byte-exact but NOT needed:** the
  decode PADDED mask (valid_kv_len=pos+1) makes all packed head-rows share the
  same causal window, so one MASKED_SOFTMAX == 12; but post-C softmax is
  already ~free, and keeping per-head softmax lets the per-head dequant feed it
  unchanged (byte-identical). Deferred.

**Grouped pack (`5f47277`) LANDS the ctx-512 win.** A full 12-head pack blows
the budgets at ctx-512 (K_all^T 405 KB > 256 KB WBUF; 12 K caches 396 KB >
128 KB ABUF), so pack **g heads at a time** with g the largest that fits — g=2
at ctx-512 (68 KB / 68 KB) → 6 group-matmuls/stream. Each group's K caches free
before the next group's load, and each group's softmax/attn_v consume its score
tiles immediately, so ABUF holds only `group` caches + score tiles. tiny packs
as one group (== dabbeb9); 124M ctx-512 packs group=2.

**Measured (mode-1 honest-BW, 34.41 MHz, b16 pos-510):**

| metric | pre-B (C) | grouped pack | Δ |
|---|---:|---:|---:|
| step cyc | 76,550,463 | 70,208,091 | −8.3% |
| sys_busy | 35,513,277 | 28,987,197 | **−18.4%** (the QK^T consolidation) |
| sfu_busy | 10,056,082 | 10,056,082 | 0 (systolic-only) |
| **tok/s** | **7.192** | **7.842** | **+9.0%** (+181% over base) |

Byte-exact: golden 12/12 identical, tiny mode-1 RTL==golden byte-match, zero new
failures. b1 unchanged (1.633 — inject_kv_cache path, untouched).

**Remaining stretch to ~9 (full pack, ~3× QK^T vs the grouped ~1.5-1.9×):**
N-split `K_all^T` into ≤256 KB WBUF passes + stream the per-head K-load
(load→transpose→free one 33 KB cache at a time) so all 12 heads pack in one
matmul. Code + design: `w8a16_emit/packed_attn.py` + scratchpad/leverB_design.md.

Waterfall now: 2.79 → A 3.80 → +C **7.19** → +B(grouped) **7.84** → +B(full)/D
~9-11 → +E ~12-13.

---

## 0.3 UPDATE — lever B STREAMING-K group-widening landed (group 2 → 4)

The grouped pack's group=2 cap was **not** WBUF — it was the co-resident INT8 K
caches in ABUF *during the matmul* (`group × 33.8 KB`). Those caches free before
the dequant score tiles allocate, so the matmul phase and the dequant phase never
overlap. **Streaming the K loads** — the packed emitter now loads each head's
INT8 K cache into ONE reused ABUF scratch (`load → transpose → free`) instead of
consuming `group` pre-loaded caches — collapses that term to a single 33.8 KB
scratch. The group then rises to where the *dequant phase* binds: the `group`
per-head FP16 score tiles (16×N_pad) + one softmax-output tile + the batched
query projections. At 124M ctx-512 that is **group=4 → 3 groups/stream (was 6)**;
packed QK^T matmuls 1152 → 576. `K_all^T` bytes and every downstream op are
untouched, so it is byte-exact by construction (no ISA/RTL/golden change; pure
compiler). K loads are no longer standalone `kv_load` nodes (V loads remain);
the packed node carries `stream_k` + global `k_heads`.

**Measured (mode-1 honest-BW, 34.41 MHz, b16 pos-510):**

| metric | grouped (g=2) | streaming (g=4) | Δ |
|---|---:|---:|---:|
| step cyc | 70,208,091 | 66,799,599 | −4.9% |
| sys_busy | 28,987,197 | 25,724,157 | **−11.3%** (further QK^T consolidation) |
| sfu_busy | 10,056,082 | 10,056,082 | 0 (systolic-only, exact) |
| dma_beats | 19,476,400 | 19,476,400 | 0 (same K bytes, exact) |
| **tok/s** | **7.842** | **8.242** | **+5.1%** (+195% over base) |

Byte-exact: tiny b1/b16 golden identical, tiny mode-1 RTL==golden byte-match
(`test_batched_decode` 7/7), **gpt2 124M b1/b16 golden byte-identical** to the
grouped baseline; 95 compiler/decoder tests green, zero new failures. b1 unchanged
(1.633 — inject_kv_cache path, untouched).

**Remaining stretch to ~9 (full pack, group=12, one matmul/stream):** the WBUF
`K_all^T` for 12 heads (405 KB) still exceeds 256 KB, so the group-4→12 step needs
the N-split (≤256 KB WBUF passes writing distinct ACCUM column ranges, per-pass
dequants) — ACCUM is *not* the binder (16×N_pad INT32 = 33 KB). MATMUL writes/
clears only its (M,N) tile at `dst_off` (golden `systolic.py:81`, RTL
`systolic_controller.sv:112-115`), so coexisting N-pass ACCUM regions are safe;
add a forced-small-WBUF cosim test for RTL coverage (tiny packs one pass).

---

## 0.4 UPDATE — lever B interleaved AV → group=7 landed; QK^T packing is DONE

Removed the LAST cap that pinned the group below WBUF: the per-head score tiles.
All `group` per-head QK^T dequants had to precede any AV because the AV matmul
wrote ACCUM[0], clobbering the packed scores. Fix: `matmul_attn_v` gains an
`av_accum_off` (units) that parks its output tile PAST the (16×N_pad) scores
block; the group then INTERLEAVES dequant→scale→softmax→AV per head with the
scores staying intact in ACCUM[0..] — so only ~2 score tiles are ever live, and
the group rises to the WBUF bound. AV arithmetic is unchanged (only the ACCUM
address moves) → byte-exact. 124M ctx-512 → **group=7 (2 groups/stream, 7+5),
packed QK^T matmuls 576 → 384**.

| metric | streaming g=4 | interleaved g=7 | Δ |
|---|---:|---:|---:|
| step cyc | 66,799,599 | 65,706,735 | −1.6% |
| sys_busy | 25,724,157 | 24,636,477 | −4.2% |
| sfu / dma | 10,056,082 / 19,476,400 | 10,056,082 / 19,476,400 | 0 (exact) |
| **tok/s** | **8.242** | **8.379** | **+1.66%** |

Byte-exact: tiny b1/b16 golden identical + RTL==golden (`test_batched_decode`
7/7); gpt2 124M b1/b16 golden byte-identical to the 822d350 baseline; 145 tests
green. b1 unchanged (1.633).

**FINDING — the full 12-head N-split is NOT worth doing; group=7 is the QK^T
packing ceiling at ctx-512.** The dominant systolic cost scales with the number
of QK^T *matmul instructions* (measured ~1.08M sys_busy each: g2/6→g4/3→g7/2
matmuls tracks −3.27M then −1.08M). The 256 KB WBUF holds at most a 7-head
single-pass `K_all^T` (8 heads = 270 KB > 256 KB), so ctx-512 needs **≥2 QK^T
passes no matter what** — and group=7 already achieves exactly 2. A group=12
N-split is *also* 2 passes: both do the identical 1584 tile-ops (48 k-tiles ×
33 n-tiles) in 2 matmul instructions. The only delta is ACCUM preclears (2×33=66
vs 17+16=33), worth ~405 K cyc ≈ **+0.5%** — not worth the per-pass-dequant /
coexisting-ACCUM / forced-split-test complexity. Lever B (QK^T packing) is
**mined out at ~8.4 tok/s**; the systolic floor is now the AV matmuls + the
non-packable FFN/projection systolic, and the real next block is the serialized
helper (lever D).

Waterfall now: 2.79 → A 3.80 → +C **7.19** → +B(grouped) **7.84** →
+B(streaming g=4) **8.24** → +B(interleaved g=7) **8.38** [B ceiling] →
+D ~9.5-10.5 → +E ~11-12.

---

## 0.5 UPDATE — lever D LANDED (DMA transpose-load; byte-exact)

De-serialized the ~9.4M-cycle/step helper K^T-transpose pass by folding the
transpose into the DMA **load**: a new transposed-LOAD mode reads the contiguous
(N_pad, d_head) INT8 K cache and writes its (d_head, N_pad) transpose straight to
the WBUF K_all^T block — one DMA per head, replacing the per-head ABUF scratch
load + serial BUF_COPY(transpose=1) helper op. Byte-identical K^T bytes ⇒ packed
scores unchanged. The plan's "strided-beat write" premise was **wrong** (the
transpose is byte-granularity, not beat-granularity); the real mechanism is a
16-row-stripe byte-transpose datapath in `dma_engine.sv` (buffer a stripe, write
C transposed columns to strided WBUF rows). Geometry rides the reserved M-type
bits (transpose + cols_log2) → byte-compatible encoding, no ISA freeze. Detail in
`docs/lever_d_dma_transpose.md`.

| metric | interleaved g=7 (f152b07) | +lever D | Δ |
|---|---:|---:|---:|
| step cyc | 65,706,735 | 56,299,503 | **−9,407,232 (−14.3%)** |
| sys_busy | 24,636,477 | 24,636,477 | 0 (exact) |
| sfu / dma | 10,056,082 / 19,476,400 | 10,056,082 / 19,476,400 | 0 (exact) |
| **tok/s** | **8.379** | **9.779** | **+16.7%** (+250% over base) |

The step cut equals the predicted ~9.4M helper pass exactly; sys/sfu/dma are
byte-for-byte invariant — pure helper-pass deletion. Byte-exact: tiny b16 golden
identical + RTL==golden (`test_batched_decode`, 128 transpose loads exercised);
gpt2 124M b1 (`b882d500`) and b16/window-511 (`2f37c52c`) golden byte-identical to
baseline; test_dma 29/29 (incl. 3 new transpose cases); test_isa/assembler green.

Note: the RTL cosim harness (`testbench.h`) needed a portable soft-float16
fallback to build on this box's g++-9 (`_Float16` needs GCC 12+); it is bit-exact
to numpy (validated all 65536 halves + 8.3M f32 + 6.3M f64 incl. ties) and native
on newer compilers. Rebuild `run_program_synth` with **`-GDRAM_SIZE=1073741824`**
(1<<30) — the 16 MB Makefile default faults the 124M bench (FAULT_DRAM_OOB).

Waterfall now: … +B(interleaved g=7) **8.38** → **+D 9.78** → +E ~11-13
(fmax, ≥24 GB box).

---

## 1. Where the cycles go

### 1.1 Batch-1 decode step, measured per-opcode (fresh profile, HEAD)

| op | gap cyc | % | count | cyc/op |
|---|---:|---:|---:|---:|
| MATMUL | 12,391,812 | 36.6 | 1,623 | 7,635 |
| LOAD (visible) | 6,571,193 | 19.4 | 11,033 | 596 |
| MASKED_SOFTMAX_FP32 | 3,907,728 | 11.5 | 144 | 27,137 |
| QUANT_FP32_INT8 | 2,912,105 | 8.6 | 1,321 | 2,205 |
| DEQUANT_ACCUM_FP32_SCALED | 2,006,231 | 5.9 | 1,335 | 1,503 |
| SYNC (≈ helper drain) | 1,959,361 | 5.8 | 14,499 | 135 |
| LAYERNORM_FP32 | 1,088,025 | 3.2 | 25 | 43,521 |
| MAX_ABS_REDUCE_FP32 | 863,563 | 2.5 | 745 | 1,159 |
| DEQUANT/VADD/GELU/ctl rest | ~2,169,000 | 6.4 | | |

busy counters: sfu 11.89M, sys 12.39M, **helper 1.90M** (BUF_COPY retire-gap is
4.0 — its execution is absorbed by the following SYNC's gap; use
`busy_cycles.helper`).

### 1.2 Batch-16 step, measured per-opcode (fresh profile, HEAD)

| op | gap cyc | % | count | cyc/op | note |
|---|---:|---:|---:|---:|---|
| MASKED_SOFTMAX_FP32 | 61,712,640 | 31.3 | 2,304 | 26,785 | 15/16 rows are padding |
| MATMUL | 35,519,220 | 18.0 | 5,943 | 5,977 | == sys_busy |
| QUANT_FP32_INT8 | 32,167,721 | 16.3 | 9,961 | 3,229 | 2304× each of K/V/sm/q + shared |
| SYNC | 29,335,681 | 14.9 | 61,866 | 474 | ≈ helper execution (busy 28.53M) |
| LOAD (visible) | 23,994,497 | 12.2 | 20,972 | 1,144 | of 28.78M beats — KV barely hides |
| DEQUANT_ACCUM_FP32 (0x17) | 8,409,600 | 4.3 | 4,608 | 1,825 | qkt + av dequants (pc-vector) |
| DEQUANT_SCALED / LN / MAXABS / VADD / GELU | ~4.5M | 2.3 | | | shared, B-invariant |
| ctl rest (SET_ADDR/CONFIG/STORE/…) | ~1.6M | 0.8 | | | |

Closure: SFU 106.81M (54.2%) + sys 35.51M (18.0%) + **helper 28.53M (14.5%,
serialized — K^T transposes + V→WBUF copies ≈ 19.4M, row_copy/gather/concat +
`_compact_abuf` churn ≈ 9M)** + visible DMA ~24.2M (12.3%) + ctl ~2.1M ≈
197.1M. Per (stream,head) core ×2304: softmax 26.8K + quants ~13.4K + dequants
~3.7K SFU; matmuls 10.7K sys; copies ~12.4K helper; ~63 insns; K+V fp16 loads
8,448 beats mostly unhidden (nothing legal to hide them under in the serial
SFU-heavy chain).

### 1.3 The three structural wastes (all in the attention core)

1. **16-row tile tax.** CONFIG_TILE M is in 16-row units (`sfu_engine.sv:511`
   `(tile_m+1)<<4`); every per-stream SFU op walks 16 rows for 1 real row.
   Softmax alone: 62.5M of which ~58.6M is padding rows.
2. **Per-step KV re-quantization.** K and V are cached **fp16** and re-loaded,
   re-quantized (static calibration scales!), and re-copied/transposed into
   WBUF every step for every (stream,head): ~20M SFU + ~19M helper+copy +
   double the KV DMA. Quantization with a static scale is a deterministic pure
   function — doing it once at store time is **byte-identical**.
3. **Per-(stream,head) op explosion.** 2304 attention cores × ~63 insns, each
   paying matmul fixed costs (130 cyc/tile + ACCUM pre-clear 64/tile),
   per-core SET_ADDR/SYNC/CONFIG, `_compact_abuf` churn, and 4,608 broadcast
   pc_scale loads.

---

## 2. Ranked levers

tok/s = fmax / cyc_per_token; cycle levers and fmax levers multiply.
Estimates are per-step cycles at B=16 ctx-512, from the validated cost model
(±15% on sub-splits; A/B/C interact — combined numbers are the honest ones).

| # | lever | step cyc → | tok/s | effort | risk | bit-exact? |
|---|---|---|---:|---|---|---|
| A | int8 KV cache (store-time quant) | 197M → ~154M | ~3.65 | ~1 wk, compiler+layout | low | **yes** (static scales) |
| C | `m_exact` row count (ISA ext) | +A → ~86M | ~6.5 | days RTL, ~1 wk total | low-med | yes (freeze §6 revision) |
| B | packed attention core (12 heads → 1) | +A+C → ~63M | **~9** | 2-3 wk compiler | med | yes (needs C for per-head scales) |
| D | de-serialize K^T transpose / helper | +ABC → ~53M | ~10.5-11 | RTL (DMA transpose) or layout trick | med | yes |
| E | fmax cluster div_p6+sqrt (≥24 GB box) | ×1.12–1.18 | ~12-13 | multi-day EDA | med | yes |
| F | multi-clock domains (SFU slow island) | ~×1.7-2 on top | ~18+ | multi-week CDC | highest | yes |
| G | W4 weights (RTL doesn't decode int4 yet) | −~5M beats | small post-B | RTL+PPL gate | med | **no** (PPL-gated) |
| H | B=32 (needs ABUF freed by A/B) | ~+15-20%/tok | — | small after B | low | yes |
| I | serving completeness: logits ×16, multi-token prefill | ~free / big TTFT win | — | small / 1-2 wk | low | yes |

(Empirically grounded: A's SFU cut is the measured K/V QUANT share ≈ 25.3M of
the 32.2M QUANT class; C's cut is the measured softmax 61.7M → 3.9M plus the
0x17 dequants 8.4M → ~0.6M and sm-quant; B's is the measured attention-matmul
24.7M → ~11.2M plus helper/ctl consolidation; D's is the residual ~9.4M of
serialized K^T transposes inside the 28.53M helper.)

### 2A. int8 KV cache — store-time quantization (do first) — **DELIVERED, see §0**

All four attention quant boundaries use **static calibration scales**
(`attention.py:141-163, 167-185, 398-422, 435-453`) — so quantizing K/V rows
**once at kv_store time** produces bit-identical int8 bytes to today's
per-step re-quant. Store `k_int8`/`v_int8` (layout already supports
`elem_bytes=1`, `kv_cache.py:63,123`) instead of fp16:

- kills QUANT-K/QUANT-V ×4608: **−20.3M SFU**
- kills the V ABUF→WBUF copy (DMA int8 V straight to WBUF — LOAD already
  targets WBUF, e.g. the pc_scale load `attention.py:230-234`): **−9.7M helper**
- halves KV DMA: 19.5M → 9.7M beats: **−~5M visible**
- frees the 66KB fp16 V + transients per stream → ABUF headroom (compaction
  churn drops; ctx>512 and B=32 unblock; KV DRAM halves → ctx-1024 batched
  fits the 1 GB budget: 302MB KV + 392MB weights)
- also −~2.2M on batch-1 decode (~+7% single-stream)

V stores become 12×64B per (layer,stream) (per-head rows); K keeps its
per-(layer,head,stream) row-major layout, per-step BUF_COPY-transpose to WBUF
stays (see D). New-row quant: the (16,64) K/V projection tiles are quantized
once per (layer,head) — full 16-row utilization — then rows stored.
Gate: byte-match cosim (outputs identical by construction) + kv-layout tests.

### 2C. `m_exact` — exact SFU row count (small ISA extension, big cut) — **DELIVERED, see §0.1**

Landed as designed with two refinements: the field is **12-bit** (bits
[27:16], not 5-bit — decode M can reach 1024 at prefill) and **MAX_ABS *was*
adopted** after all. The reduction-safety worry below was resolved: pad rows
are zero-filled before every MAX_ABS, so bounding to the real rows gives the
identical max (|0| ≤ any real value). Result beat the estimate: **b16 7.192
tok/s (est. ~7.1), b1 1.633 (est. ~1.16)** — b1 far exceeded because C-3b's
shared-op adoption (matmuls + LN/GELU/VADD, not just attention) near-eliminated
the batch-1 SFU (10.13M → 0.63M). Original plan text preserved below.

CONFIG_TILE bits **[28:0] are free and already ignored by the RTL decoder**
(`decode_unit.sv:158-160` extracts only [58:29]; the W4 bit [28] exists only
in software). Add a 5-bit `m_exact` (0 = full tile, backward compatible —
every existing bundle byte-identical):

- RTL: **one mux** at the dispatch latch (`sfu_engine.sv:882`) —
  `m_rows_q <= use_exact ? m_exact : dispatch_m_rows_w`. `m_rows_q` is used
  *only* as the row-loop bound at the 7 loop sites, never in addressing or the
  OOB check; mode-0 DPI and mode-1 synth share the FSM, so one change bounds
  both. fmax-neutral by construction (a 15-bit mux far from the fp32 cluster).
- Golden: `simulator.py` gen-2 handlers — loop `m_exact` rows and **partial
  write** (leave rows [m_exact,16) untouched); ~14 sites, mechanical.
- Freeze: §6 dated revision + `gen_gen2_fixtures.py` regen + SHA re-pin
  (exact W4-extension recipe, commit `81ad53f`).
- Compiler: emit `m_exact=1` on the per-stream attention SFU ops (softmax,
  qkt-dequant, sm-quant, av-dequant). MAX_ABS keeps full rows (it's a
  reduction; bounding changes semantics — padding rows are zeroed today, so
  bounded is equal-valued, but leave it full-tile for safety).

Effect at B=16: softmax 62.5M → 3.9M, qkt-dequant 5.1M → 0.3M, sm-quant
2.5M → 0.2M. **With A: step ≈ 78M → ~7.1 tok/s (2.5×).** Also worth ~+6% on
batch-1 decode (softmax 3.9M → 0.24M).

### 2B. Packed attention core — one QK^T/softmax/AV per (layer,stream)

Restructure the per-stream loop from 12 per-head cores to **one packed core**:

- **Q_pack** (16,768) int8: block-diagonal — row h holds head h's query in
  columns [64h,64h+64), zeros elsewhere (one-time zeroed region; 12 tiny
  BUF_COPY row-inserts per stream from the per-(layer,head) q_int8 tiles,
  which are quantized once for all 16 streams). Zeros contribute exactly 0 to
  int8 dot products → scores bit-identical.
- **QK^T**: one matmul (16, 768, 528) vs 12× (16,64,528). Same MACs, but the
  130-cyc/tile overhead + ACCUM pre-clear amortize: 12×7.84K → ~32.8K
  (2 WBUF passes: K^T (768,264) = 198KB ≤ 256KB WBUF). **Attn systolic
  24.7M → ~11.2M.**
- **scores** (12 real rows, 528): dequant per head-row (12 × 1-row DEQUANT
  via `m_exact`, each with that head's constant pc vector) → **one
  MASKED_SOFTMAX (m_exact=12)** ≈ 20.3K vs 12×27.1K. All rows share
  keep_through (lockstep) — the per-row causal ramp already handles rows
  identically at `query_row_base=position`.
- **probs quant**: 12 × 1-row QUANT with each head's static sm_scale
  (`m_exact=1`, ~72 cyc each) — preserves exact per-head scales, no coupling.
- **AV**: one matmul (16,528,768) against V (528,768) int8 loaded
  **directly DRAM→WBUF** (per-(layer,stream) (pos,768) int8 V layout from A;
  k-split ×2 with `flags_accumulate` since 405KB > WBUF). Output (16,768):
  diag block (row h, cols 64h..) is head h's output — dequant once with a
  per-column composite vector (sm_scale_h·v_scale_h on column block h — the
  existing per-N pc_scale mechanism, `codegen.py:468-476`, just non-uniform),
  then 12 tiny diag-extract BUF_COPYs replace gather_rows + its spills.
- **K^T**: per-head slice transposes into WBUF row offsets (same bytes as
  today, ~9.7M helper — target of D).

Instruction count: 2304 cores × ~63 → 192 × ~40 insns (−~135K insns/step);
pc_scale blob dedup falls out (192×2 vectors vs 4608 broadcasts, −2.6MB DRAM).
ABUF peak drops (no fp16 V, no 16 co-resident score tiles — the old
gather-softmax blocker is moot). **A+B+C step ≈ 52M → ~10.5 tok/s (3.8×).**
Gate: byte-match per stream (scores/probs/outputs identical by construction —
same int8 inputs, same scales, same reduction orders); `test_batched_decode`
suite; PPL spot-check unnecessary (no numeric change).

### 2D. De-serialize the K^T transpose (~10M helper, post-B the #2 block)

Helper is forbidden to overlap anything (`taccel_top.sv:558-561`), so the
per-step 9.7M-cycle K^T transpose is pure serial time. Options, cheapest
first: (i) **K^T-blocked DRAM layout** — store K^T in (64,16) column blocks;
appending a token = 64 strided bytes = 4 beats ×16-byte... requires strided
DMA (not supported) → instead 64 small stores/step/(head,stream) is worse;
skip. (ii) **DMA transpose-on-load** into WBUF (new dma_engine addressing
mode, moderate RTL, no ISA change if keyed off a CONFIG bit): kills the
helper pass entirely AND overlaps legally with systolic → **−~10M, ~13
tok/s**. (iii) relax helper∥DMA/sys overlap arms (contract change + port
arbitration — bundle with F). Note: with B, transposes run 12×192 on int8;
any of these also shrinks batch-1 decode.

### 2E/2F. fmax and clock domains (multiplicative, unchanged from 07-08 doc)

- **E**: div_p6 + sqrt-Mpad restructure: +12–18% on everything; still blocked
  on a ≥24 GB box (`/tmp/pnr_final.sh` ready).
- **F**: post-A+B+C the step is ~78% sys+helper+dma+ctl — blocks that
  synth at 109–875 MHz vs the SFU's 34.41. A slow-SFU-island CDC design
  roughly halves the remaining step again (~20 tok/s at B=16). Heaviest
  verification burden; byte-match contract is timing-agnostic so cosim
  survives. Only worth starting after A–D land.

### 2G–2I. Second tier

- **G. W4 weights**: golden/ISA carry `weight_int4` (CONFIG_TILE bit 28) but
  the **RTL never decodes it** — real RTL work (systolic B-load unpack).
  Halves weight DMA (9.9M → ~5.2M beats); mostly hidden post-B; PPL-gated
  (TurboQuant sweep suggests even 4-bit KV is tolerable, so W4 weights are
  plausible). Bigger win for prefill/single-stream than batched decode.
- **H. B=32**: graph/layout generalize (guard at `tiny_fixture.py:426` +
  bench); shared FFN/weights amortize 2×; per-stream attention scales
  linearly. Post-A/B ABUF fits it. ~+15–20% per-token.
- **I. Serving completeness**: logits ×16 is ~free (+92K beats, ~180 insns —
  finish it); **multi-token batched prefill** reuses the packed-attention
  machinery with M=16 real query rows per stream — 16× on prompt
  processing (TTFT), currently ~1 tok/s equivalent.

---

## 3. Recommended sequence

1. **A — int8 KV** (~1 wk): byte-exact, unlocks ABUF, halves KV DMA,
   −25M SFU, −10M helper. Lands alone → ~3.65 tok/s.
2. **C — m_exact** (~1 wk incl. freeze revision): 1-mux RTL + golden partial
   write + emit plumbing. With A → **~6.5 tok/s**. (Also +6-7% batch-1.)
3. **B — packed attention** (2–3 wk): the structural consolidation.
   → **~9 tok/s** batched, and it shrinks single-stream attention too.
4. **D — kill the transpose serialization** (RTL DMA-transpose preferred)
   → ~10.5–11 tok/s.
5. **E — fmax** when a ≥24 GB box appears (×1.12–1.18) → ~12–13.
6. **H, I** opportunistically; **F (multi-clock)** as the next big rock;
   **G (W4)** if/when prefill or DRAM footprint matters.

Waterfall (B=16, ctx-512, honest-BW, 34.41 MHz unless noted):
2.79 → A 3.80 → +C **7.19** → +B ~9 → +D ~10.5-11 → +E ~12-13 → +F ~18+ tok/s.
Single-stream decode rides along: 1.016 → 1.126 (A) → **1.633 (A+C)** → ~1.8+ (B,D).
**A landed (§0): b1 1.126, b16 3.796 — both beat estimate.**
**C landed (§0.1): b16 3.796 → 7.192 (beat the ~6.5-7.1 estimate); b1 1.126 →
1.633 (far beat the ~1.16 estimate — C-3b's shared-op adoption near-eliminated
batch-1 SFU). sys_busy is now the floor on both shapes → B/D are next, not SFU.**

## 4. Floors (what "done" looks like at this fmax)

Per token at B=16 ctx-512: FFN+proj systolic ~680K; attn systolic (packed)
~700K; shared SFU ~350K; softmax floor ~245K; KV int8 DMA ~600K beats;
weights ~620K beats (hides under FFN). Fully-serialized floor ≈ ~3.3M/tok
(≈ 10.4 tok/s); with legal DMA∥sys overlap fully exploited ≈ ~2.7M
(≈ 12.7 tok/s); E pushes toward ~15; past that it's F (clock domains) or
wider hardware (array/SFU lanes).

## 5. Measurement notes

- Profile scripts (this session): `scratchpad/profile_step.py` (retire-gap
  attribution incl. helper via `busy_cycles`), `build_bins.py`,
  `rebuild_mode1.sh`, `pipeline.sh` — clone of `/tmp/sfu_profile_m1.py`
  extended to decode/batched bins.
- BUF_COPY retire-gap ≈ 4.0 always — helper time lands in the following
  SYNC's gap; read `busy_cycles.helper`.
- Batch-16 empirical per-opcode profile: landed (§1.2), full results in
  `scratchpad/prof_b16_p510.txt` + `_sum.json`. Aggregates agree with commit
  `2d877be` to −0.07% (bundle-emission variation of ~1,150 insns between that
  build and this rebuild; b1 reproduces exactly).
