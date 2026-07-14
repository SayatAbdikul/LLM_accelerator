# Lever B3 — speculative decoding: make the 16 rows carry 16 tokens

**Status: LANDED (compiler + host only — zero RTL change).**

This is the batch-1 lever. Every other b1 lever shaves cycles off a step; this one
changes how many tokens a step is worth.

---

## Why batch-1 needs it

Two measurements frame the whole problem. Both are from the b1 decode step at
ctx-511 on GPT-2 124M, HEAD, honest-BW (`profile_decode_step.py --batch 1`):

| | | |
|---|---:|---|
| step | **19,707,911 cyc** | = **1.746 tok/s** @ 34.41 MHz |
| systolic | 11,074,973 (56.2%) | 1,623 MATMULs |
| DMA (exposed) | 6,000,181 (30.4%) | |
| control | 1,988,123 (10.1%) | 13,923 SYNCs + 22,014 SET_ADDR_HI |
| SFU | 634,304 (3.2%) | lever C already killed it at b1 |
| helper | 958,464 busy | 2,581 BUF_COPYs |
| **DMA beats** | **10,515,280** | |

**1. The mesh is 15/16 idle, and rows are nearly free.** The array is 16×16. A
decode step presents ONE query row and pads the other 15. But the systolic streams
weights at the same rate regardless of M — only the A-load and the drain scale with
rows — so a 16-row pass costs barely more than a 1-row step. The waste is spatial,
and no scheduling lever can reclaim it. Only *putting real tokens in those rows* can.

**2. There is a hard DMA wall at ~3.3 tok/s.** 10,515,280 beats/token at 16 B/beat
is 10.5M cycles of DMA activity minimum — at 34.41 MHz that caps batch-1 at
**~3.27 tok/s even with perfect overlap and infinitely fast compute**. Weights are
INT8 and stay INT8 (project constraint), and ~7.75M of those beats are the weight
stream itself: the entire 124 MB model is pulled from DRAM **to produce one token**.

Shaving systolic cycles cannot beat wall 2. Only two things can: move more bytes per
cycle, or **make one weight stream produce more than one token**. That is spec-dec.

---

## The verify pass already existed

The expensive half of speculative decoding — scoring K candidate positions in one
pass — is exactly lever I-b's chunked prefill, which has been in the tree and
byte-exact since `67c7f46`:

`HostRunner.run_prefill_chunk(tokens, base_pos)` runs P causal query rows at a
runtime-patched base. Row *i* is masked to exactly the keys at or before its own
global position (`query_row_base + row_idx`, `sfu_engine.sv:505-517`), so **row i is
the model's next-token distribution after consuming chain[0..i]** — which is
precisely what verification needs, for every candidate at once.

So B3 is not a hardware feature. It is the observation that the verify primitive was
already shipped, plus the glue to read all P of its rows.

---

## What was missing (and it was small)

1. **The prefill stream stored one logits row.** Lever I-a taught the *decode* stream
   to store all N rows (`store_rows`); prefill kept a last-row-only store. Now
   `store_rows = P, row_index = 0` for the prefill stream — **opt-in**
   (`prefill_store_rows`), not inferred from the graph shape, because…
2. **…the logits region had to grow in lockstep.** Prefill and decode **share** the
   logits region (both offsets default to `logits_base`), and `kv_cache_base` sits
   immediately after it (`assembler.py`). A P-row store into a 1-row region runs
   straight off the end into the KV cache. `logits_rows = max(batch, prefill_tokens)`
   sizes it; the assembler validates `logits_size % logits_rows == 0`.
3. **Readers had to slice to their own rows.** This is the subtle one. On a
   `prefill_tokens=16` bundle the region holds 16 rows, but a decode step writes only
   row 0 — rows 1..15 still hold *the last verify pass's logits*. Reading the whole
   region and taking an argmax over it would silently return a token from a stale
   row. `_read_logits(offset, rows)` now takes an explicit row count, and every
   caller passes its own.
4. **The driver + a draft** (`taccel/runtime/speculative.py`).

---

## Exactness — a property of the accept rule, not of the draft

The accepted tokens are **identical to sequential greedy**, and that does not depend
on the draft being good:

- a candidate is accepted **only** where the model's own argmax — computed on the
  true prefix, inside the same pass — agrees with it;
- every pass then contributes one **correction token** taken straight from the model
  (the argmax of the first row that disagreed, or of the last row if all agreed).

So each emitted token is the greedy argmax given the true prefix. **A bad draft costs
cycles, never correctness.** The draft needs no accuracy guarantee, which is why a
weightless n-gram is a legitimate choice rather than a compromise.

Argmax ties are pinned to the lowest index in both the driver and the reference, so
the gate cannot become a coin flip on tied logits.

**Rejected candidates need no KV rewind.** The chain always leads with the known-good
token `cur`, so every pass rewrites `cur`'s KV row correctly. Rejected guesses and
pad rows write garbage KV *after* the accepted prefix — and the next pass, based at
`cur_pos`, overwrites every one of those rows before anything reads them. Within a
pass, a real row *i* is causally masked to `col ≤ base+i`, so it can never see a pad
row (which sits at `base+len..base+P-1`, strictly beyond it). Nothing ever reads a
speculative KV row.

---

## The draft: prompt-lookup n-gram (no weights, no second model)

`PromptLookupDraft` takes the last *n* tokens, finds the most recent earlier
occurrence of that n-gram in the context, and proposes whatever followed it (longest
n first). Zero training, zero parameters, zero extra DMA. On grounded text
(summarisation, QA over a passage, code) the continuation is often a literal repeat.

**Adaptive fallback:** when the draft proposes nothing, the driver issues a plain
1-token decode step instead of paying for a verify pass. This is what pins the
guaranteed floor at ~1.0× the baseline rather than 1/r.

---

## Economics — report the terms, not a headline

    speedup = t / r        t = tokens confirmed per pass,  r = pass cost / step cost
    break-even at t = r

`r` is a hardware property: workload-free, measured by
`software/tools/bench_specdec_cycles.py` (both programs extracted from the *same*
bundle, so there is no cross-build variation in the ratio).

`t` is a property of the draft **and the text**. It is not a constant, and quoting a
single tok/s number without it is how a spec-dec claim turns into fiction.
`SpecDecStats` reports acceptance rate and a tokens-per-pass histogram so the number
is auditable. The honest way to state the result is: **floor ~1.0× (fallback),
break-even at t = r, and a named-corpus figure for t.**

### Measured — GPT-2 124M, P=16, base_pos=496, honest-BW, 34.41 MHz

| engine | 1-token decode step | 16-row verify pass | × |
|---|---:|---:|---:|
| **systolic** | 11,109,206 | **11,109,206** | **1.00** |
| helper | 986,112 | 986,112 | 1.00 |
| DMA beats | 10,534,000 | 9,943,480 | 0.94 |
| **SFU** | 634,441 | **10,094,146** | **15.91** |
| **total** | **19,794,743** | **28,763,366** | **r = 1.4531** |

**Break-even = 1.45 accepted tokens/pass. Ceiling 16.**

| t | 2 | 3 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| tok/s | 2.39 | 3.59 | 4.79 | 7.18 | 9.57 |
| vs baseline | 1.38× | 2.06× | 2.75× | 4.13× | 5.51× |

Note t=4 (**4.79 tok/s**) already clears the **~3.27 tok/s DMA wall** that caps *any*
non-speculative b1 decode at this clock. Nothing else on the roadmap can pass it.

### The result that re-ranks the roadmap

**The systolic verifies 16 tokens for the price of one — 11,109,206 cycles, identical
to the cycle.** So does the helper; the DMA actually moves *fewer* beats. This is the
lever's thesis, measured rather than argued: mesh cost is M-independent, and the
124 MB weight stream amortises to nothing.

**Every cycle of the 1.45 is the SFU**, which scales linearly with rows (15.91×,
+9.46M cyc). So the SFU — masked softmax above all — is now the *only* thing between
us and near-free verification, and it is worth far more here than anywhere else:
driving SFU cost down pulls `r` toward 1.0, which multiplies **every** acceptance
level simultaneously. At r = 1.0, t = 4 would be 4.0× instead of 2.75×.

This revives **lever 1d** (softmax MAX reduction 8-wide, bit-exact by fp32-max
associativity, `sfu_g2_compute.svh:514-531`), which was correctly judged worthless
for a 1-row decode step (SFU = 3% there) and is now the top b1 lever *because* the
workload is a 16-row pass (SFU = 35%). Same for anything else that shrinks per-row
SFU work.

---

## Verification

- **`test_speculative_decode.py` — the gate that matters:** speculative output is
  **token-for-token identical to sequential greedy**, on both a generic and a
  repetitive prompt, with `stats.passes > 0` asserted so the test cannot pass
  vacuously by falling back every step.
- Amortisation: a hitting draft must emit more tokens than it runs passes.
- **`test_multi_token_prefill.py` still green (7/7)**, and now *stronger*: the
  RTL-vs-golden byte-match covers all 16 logits rows instead of one.
- `test_batched_decode` + `test_decoder_bundle` + determinism: green — a
  `prefill_tokens=1` bundle is byte-identical to before (`store_rows` defaults to the
  old last-row store; `logits_rows=1`).

## Follow-on

- The draft is an interface (`Draft` protocol). A Medusa-style head would fill the
  same rows at near-zero marginal systolic cost — but needs off-chip training.
- **A candidate *tree* is inexpressible on this hardware**: the causal mask is
  `query_row_base + row_idx`, a single contiguous ramp per row. Linear chains only.
  Do not attempt tree attention without an ISA change.
- `prefill_tokens > 1` requires `batch == 1`, so this is a b1 lever and does not
  compose with b16 today.
