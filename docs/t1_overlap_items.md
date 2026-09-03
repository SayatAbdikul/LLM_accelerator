# T1 items 1+2 LANDED — KV V-prefetch (b16) + FC2 weight prefetch (b1+b16), byte-exact (2026-07-16)

> **Historical landing report, reconciled 2026-09-03.** Both compiler overlap
> mechanisms remain implemented, but later pipeline work changed the latest
> measurements to 18,318,261 cycles for b1 and 49,998,042 cycles per b16 step.
> The token rates below use the old 34.41 MHz partial-design timing peg and are
> not current full-chip sign-off. See [current project status](project_status.md)
> and the [documentation index](README.md).

**Bottom line: two compiler-only overlap levers, zero RTL. Combined: b16
54,269,714 → 49,803,314 cyc/step = 10.145 → 11.055 tok/s (+8.96%); b1
19,794,743 → 18,300,503 = 1.738 → 1.880 tok/s (+8.17%). Both byte-exact
(RTL-vs-RTL logits sha1 unchanged), both with co-busy risen ≈ exactly the cycle
cut and Port-A `lost=0`.** Item 2's measurement REFUTES the redirect doc's "b1
is done for T1" (`docs/t1_measured_redirect.md`); item 1's measurement confirms
the capacity-wall-revised ceiling, not the original +20%.

Both items are the same software-pipeline shape: **defer a DMA prefetch of the
NEXT tile (no trailing sync) so it streams under the CURRENT tile's MATMUL**
(`SYNC 0b010` is systolic-only, so the DMA keeps moving), and let the next
tile's already-emitted `SYNC 0b001` (pc_scale load) drain it before first read.
Port-safe by the shipped weight-prefetch pattern: the DMA writes WBUF/ABUF via
channel A while the systolic reads src2 via dedicated Port W (WBUF Port B) and
drains via Port S (ACCUM) — `s_collision` requires channel A and channel S on
the SAME buffer, and `a_buf ∈ {WBUF,ABUF}` vs `s_buf = ACCUM` never collides
(`sram_subsystem.sv`; see docs/porta_bus_split.md and the port probe in
docs/t1_measured_redirect.md).

## Item 1 — cross-head KV V-prefetch inside the packed attention group (b16) — commit `dde6f19`

Mechanism (increment 1 of the KV-prefetch design): within each packed group's
interleaved per-head dequant→softmax→AV walk, head j's AV emitter prefetches
head j+1's int8 V tile into WBUF **right after its pc_scale `SYNC 0b001`**,
deferred; the V DMA overlaps AV_j's MATMUL. Head j+1's `kv_load` node becomes a
`kv_prefetched` no-op (the tile is already resident and the WBUF alloc is
registered under its name); its own pc_scale sync drains the prefetch before
the read. ~58 lines across three files:

- `compiler/frontend/nanogpt_adapter.py` — the packed-group interleave loop
  stamps `v_prefetch_next_head/_name/_key_len` on each `matmul_attn_v` and
  `kv_prefetched=(j>0)` on each V `kv_load`;
- `compiler/emit/kv.py` — `emit_kv_load` early-returns on `kv_prefetched`;
- `compiler/w8a16_emit/attention.py` — emits the deferred
  `_emit_dma_load(BUF_WBUF, …)` of the next head's V after the AV pc_scale
  sync (KV-layout lookup + WBUF alloc under the next head's vload name).

**Measured** (fast_gate_b16.py, b16 pos-510, honest-BW `--fast-beats`,
34.41 MHz):

| metric | baseline | item 1 | Δ |
|---|---:|---:|---:|
| step cyc | 54,269,714 | 51,297,554 | **−2,972,160 (−5.5%)** |
| **tok/s** | **10.145** | **10.733** | **+5.8%** |
| Port-A co-busy | 4,148,317 | 7,110,877 | **+2,962,560** (Δco-busy ≈ Δcyc — clean overlap, not re-serialization) |
| Port-A sys_lost | 0 | 0 | audit clean |
| logits sha1 (RTL-vs-RTL) | 205682b6515f7e85 | 205682b6515f7e85 | **byte-exact** |

b1 is untouched (batch=1 compiles the per-head `inject_kv_cache_nodes` path,
not the packed path).

**The ceiling history (why +5.8% and not the redirect's +20%):** the b1→b16
exposed-DMA delta proved ~9.4M/step of un-overlapped KV, and hiding all of it
would be +~20%. Two advisor rounds killed that number with the **capacity
wall**: the KV working set is 32 KB per (layer,head,stream) → **12.3 MB/layer
≫ 384 KB total SRAM** (ABUF 128 K + WBUF 256 K), and KV rows are
dependency-movable but not schedule-movable across the MLP — so KV can only
overlap **attention's own systolic (~5.5M)**, which is smaller than the KV DMA
(9.44M). Realistic ceiling ≈ +8–11%; increment 1 (V only, ≈ the V share of the
phase) took +5.8% of it. **Increment 2 — prefetch the next group's K^T into
the WBUF region freed by `__ktall` at the end of `emit_packed_qkt_matmul` —
is the follow-on for the rest** (spec'd, not built).

**Re-pin:** item 1 legitimately trips the spec-dec inertness byte-pin at
batch=16 (it changes the default schedule, not the logits) —
`test_specdec_is_inert_at_the_default` golden re-pinned
`e0f9c8ca2a50d259 → fa75a8991ee385b3` (user-approved; batch=1 pin unchanged at
`172b4aa61a3de54e`; spec-dec functional suite stayed green; the opt-in guard
`prefill_store_rows == 1` still asserted).

## Item 2 — FC2 weight prefetch in the large-input-streaming path (b1+b16) — commit `9a82e34`

FC2 is the only matmul lowered through
`_emit_matmul_w8a16_large_input_streaming` (`w8a16_emit/matmul.py:814+`;
K_pad=3072 exceeds the full-K budget), and it was **fully serial** — every
stage SYNC-barriered, zero overlap — while fc1/qkv/out_proj/lm_head already
had `pipeline_full_k` weight double-buffering. The change mirrors that pattern
into the streaming path: per n-group, prologue-load weight chunk 0 (+`SYNC
0b001`), then during MATMUL(w[k]) (`SYNC 0b010`) prefetch w[k+1] deferred;
the existing pc_scale `SYNC 0b001` drains it. `cur_w`/`nxt_w` double-buffer in
WBUF — the `_large_weight_tile_plan` chunks are ≤ WBUF/2 **by construction**,
so two always fit.

**Measured** (fast_gate_b16.py, honest-BW, 34.41 MHz):

| shape | baseline (item 1) | item 2 | Δcyc | tok/s |
|---|---:|---:|---:|---|
| b1 pos-511 | 19,794,743 | 18,300,503 | **−1,494,240** | 1.738 → **1.880 (+8.17%)** |
| b16 pos-510 | 51,297,554 | 49,803,314 | **−1,494,240** | 10.733 → **11.055 (+3.0%)** |

The b1 and b16 cuts are **identical to the cycle** — FC2's weight stream is
batch-invariant, a clean closure. Gates: b1 co-busy 4,148,317 → 5,641,237
(+1,492,920 ≈ Δcyc), b16 co-busy → ~8.60M (+1.49M); Port-A `lost=0` both;
logits sha1 unchanged both (b1 `eeab004014642d14`, b16 `205682b6515f7e85`).
**No re-pin needed** (the tiny model's FC2 is K=512 → full-K path;
`test_speculative_decode` 8/8 green).

**This refutes "b1 is walled":** the redirect asserted b1's exposed DMA is
"largely un-hideable" without measuring adjacency. The measured b1 profile
says otherwise — systolic slack (sys 11.11M − co-busy 4.15M = **6.96M**)
exceeds exposed LOAD (**5.89M**), so exposed DMA is hideable wherever it sits
next to a matmul. FC2's was **un-scheduled, not un-hideable**.

## What remains on this vein

1. **item 2b — FC2 input hoist** (`matmul.py` streaming path): the input tile
   is loaded + MAX_ABS'd + QUANT'd once **per n-group** (2 n-groups at 124M ⇒
   2× redundant), and each load is M_pad-padded (16 rows where M=1 is real at
   b1). Hoist out of the n-loop + load only real rows: **byte-exact** (pad
   rows are zeroed before the matmul anyway, `matmul.py:901`) and it REDUCES
   DMA (~0.28M beats at b1) rather than hiding it; also cuts SET_ADDR count.
   Small but strict.
2. **increment 2 — next-group K^T prefetch** (b16): the rest of the item-1
   ceiling (~+3-5% est.).
3. The ~2M prologue/tail exposed DMA on the already-pipelined full-K paths
   (harder — needs cross-op motion).

b1's scheduling ceiling from here ≈ max(sys 11.11M, DMA 10.53M beats) + ctl ≈
~13–14M ⇒ **~2.5 tok/s**; the sys floor itself (M=1 leaves the 16-row mesh
15/16 empty) only moves with batching/spec-dec/clock (T3).

## The gate (standing, from the Port-A scar)

Never accept "byte-exact" alone for an overlap change — the tiny cosim gate
has ZERO co-busy cycles and is structurally blind (docs/porta_bus_split.md).
Every item above quotes, from a direct `--fast-beats` run: **total cycles
dropping** (primary, un-fakeable), **co-busy risen** (the overlap is real),
**Port-A audit `lost=0`** (no silent drops), and the **RTL-vs-RTL logits
sha1** (byte-exactness; RTL-vs-golden is ill-posed at 124M past the first fp16
overflow). Tool: `software/tools/fast_gate_b16.py` (~15–20 min/run; builds the
decode bin, runs `run_program_synth --fast-beats --json-out`, prints exactly
these five numbers; `--batch 1 --position 511` for the b1 shape). Never run
two of these (or one plus any other 124M sim/synth) concurrently on a 15 GB
box.
