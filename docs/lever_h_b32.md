# Lever H — B=32 batching (and the weight-corruption bug it exposed)

**Result: +3.45% (9.729 → 10.065 tok/s), not the roadmap's +15–20%.**
**Batching is MINED OUT.** The commit is still worth having: it fixed a
pre-existing, silent weight-corruption bug and proved the RTL correct at
`M_pad > SYSTOLIC_DIM`.

## Measured — 124M, pos-510 (ctx-511), mode-1 honest-BW, 34.41 MHz

| | b16 | b32 | ratio | **per-token Δ** |
|---|---:|---:|---:|---:|
| step cycles | 56,588,361 | 109,402,720 | 1.933× | **−3.3%** |
| **sys_busy** | 24,636,477 | 49,271,619 | **2.000×** | **−0.0%** |
| **sfu_busy** | 10,059,586 | 20,117,682 | **2.000×** | **−0.0%** |
| dma beats | 19,664,920 | 31,544,920 | 1.604× | −19.8% |
| **tok/s** | **9.729** | **10.065** | | **+3.45%** |

At the post-lever-E fmax (~37.3 MHz): b16 ~10.54 → b32 ~10.90.

(b16 here is 9.729 vs lever-D's 9.779: lever **I-a** (logits×N) costs +288,858
cyc = **+0.51%**, with sys/sfu byte-invariant — exactly the "~free" price
predicted for the 1.6 MB chunked 16-row logits store.)

## Why the roadmap was wrong

The roadmap assumed the shared FFN/projection work **amortizes** over the batch.
It does not.

The systolic mesh is **16 rows wide** (`SYSTOLIC_DIM = 16`). An `M=32` matmul is
**2 m-tiles**, each costing a full 16-row pass — the weight is re-streamed
through the mesh per m-tile. The cost model is
`mt · nt · (64 clear + 130 + 17(kt−1))`, **linear in `mt`**. So systolic time
scales **exactly 1:1 with tokens**; the SFU likewise (it walks `m_exact` rows).
Both per-token figures above are identical to three decimals — that is not a
coincidence, it is the architecture.

Only **weight DMA** amortizes (same weight tile serves both m-tiles): 1.604× for
2× the tokens = −19.8%/token. But DMA largely hides under systolic, so the net
step gain is only −3.3%.

### The deep fact

**This chip is systolic-COMPUTE-bound, not weight-bandwidth-bound.** Batching is
the classic remedy for a *bandwidth*-bound machine (GPUs at low batch). Here
`sys_busy` **is** the floor and it scales 1:1 with tokens, so batching cannot
touch it. The bandwidth amortization was **already fully captured at B=16** —
that is precisely the b1→b16 2.75× win (b1 21.1M cyc/tok → b16 3.54M).
**B=16 → 32 → 64 buys ~nothing.**

**Do not pursue B=64.** It is also physically blocked: ACCUM sits at exactly
100% at B=32 (fc2's 512-col streaming N-tile = `32·512·4` = 65,536 B =
`ACCUM_SIZE`), and DRAM is already 991.4 MB of the 1 GB budget (KV alone is
604 MB at 32 streams × max_seq_len 1024).

**B=32's real value** is 2× concurrent streams served (serving capacity) at
+3.45% throughput, paid for with 2× DRAM and 2× step latency. Keep it available;
it is not a default win.

## The bug this exposed (the actual payoff)

`HostRunner._patch_kv_bases` only ever patched the **`"decode"`** stream. But the
batched bundle reuses the batched **decode** graph for its **prefill** slot
(`tiny_fixture.py`), so the prefill stream *also* carries runtime `kv_base`
sites — and they were never patched. Unpatched they read **0**, so every prefill
KV row store addressed `0 + dram_off` and landed **inside the weight/data DRAM
region, overwriting weights**:

- **B=16: 127** such 32-byte stores — they happened to hit only bytes the decode
  never read, so it stayed latent through levers A–E.
- **B=32: 275** — one landed on the **live lm_head weight** `(k=13, n=13)`,
  flipping `127 → 0`, which zeroed logit column 13.

Fix: `_patch_kv_bases(position, stream="decode")` + `run_prefill` primes the
prefill stream at position 0. The single-token prefill graph has no `kv_base`
sites, so **b1 stays byte-identical**.

## The RTL is CORRECT at M_pad > 16

B=32 is the **first** configuration where `M_pad` (32) exceeds
`SYSTOLIC_DIM` (16) — a **two-m-tile MATMUL walk**. This looked exactly like the
known-fragile multi-tile `clear_acc` area (#115/#116), and it is **not**: the
two-m-tile walk is **byte-exact**. The golden was simply reading weights its own
prefill had corrupted; the RTL escaped only because the byte-match test runs it
**decode-only**.

**This unblocks lever I-b** (multi-token prefill with M > 16 query rows).

### The invariant that cracked it

Activation scales are **static** (`MAX_ABS` = 0 in the decode stream; all
`SET_SCALE` immediates), therefore **each stream's logits must be
batch-independent**. The RTL satisfied it (b16 row0 == b32 row0); the golden did
not (col 13: −0.088684 → 0.0). That single check identified which side was
lying. **Reuse this for any future batch work.**

## Test-methodology corrections

1. **Batched tests now run at the production attention depth**
   (`BATCHED_SMOKE = 63` → key_len 64 → `Kseq_pad` 64). This is *required* for
   B=32 to even build: at `Kseq_pad ≥ 64` the per-head attn_v tiles spill to
   DRAM-temp (`attention.py:680`) exactly as on 124M, and that spill is what
   bounds ABUF. Without it the `n_head × n_streams` live attn_v tiles are
   4 × 32 × 1 KB = the entire 128 KB ABUF. A shallow `Kseq_pad < 64` is a toy
   regime that never occurs in a real decode.
2. **The RTL byte-match test is now symmetric** — golden runs decode-only,
   mirroring the decode-only RTL run (it previously ran a prefill the RTL never
   saw, which is what let the corruption hide).

**Do NOT widen the attn_v spill trigger to relieve ABUF pressure.** A
pressure-aware `spill_output` hint was tried and it **corrupted results** —
freed-hole reuse shifts the ABUF layout (the classic W8A16 corruption class, cf.
`kv.py:233`). The spill is fine; it must not be forced into a layout it wasn't
shaped for.

**124M byte-match is ill-posed** — don't build a gate on it. Per `rtl_cosim.py`
#109: per-tensor byte-match is well-posed only *before* GPT-2 124M's first fp16
overflow (`block0_out_proj` → ±65504/NaN); past it **the golden itself
saturates**, so 124M conformance is logits-level (argmax / cosine / perplexity),
not byte-equality. The byte-exact RTL==golden gate is the **tiny** model.

## Gate

- `test_batched_decode` **12/12**, incl. `test_batched_decode_rtl_matches_golden_bytes`
  for **both b16 and b32** (mode-1 synth RTL == mode-0 golden, byte-identical
  over the full N-row logits region).
- 43 passed across logits_store / decoder_bundle / host_runner /
  runtime_patch_sites / kv_layout / tiny_decode_determinism / w8a16_codegen —
  zero new failures (the 1 failure reproduces on a clean tree).
- 124M b32 builds: 390,786 decode insns, logits_size 3,217,408
  (= 32 × pad_dim(50257) × 2), all 32 per-stream rows distinct, DRAM 991.4 MB.

## Where the throughput lever actually is

`sys_busy` is the floor and scales 1:1 with tokens. Nothing in the *compiler* can
amortize it. The remaining real lever is **F — the multi-clock SFU island**: it
attacks the **clock** on that floor (the non-SFU blocks synthesize at
109–875 MHz vs the SFU's 34.41). Lever **I-b** (multi-token prefill) remains
worthwhile for **TTFT** — and is now unblocked by the M_pad>16 result.
