# Lever B3 — speculative decoding: make the 16 rows carry 16 tokens

**Status: LANDED (compiler + host only — zero RTL change).**
**Measured: b1 1.738 → 5.075 tok/s (2.92×) on wikitext-2, exact-greedy, P=16.**

This is the batch-1 lever. Every other b1 lever shaves cycles off a step; this one
changes how many tokens a step is worth. It is also the only lever that gets past the
~3.27 tok/s DMA wall.

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

### t — measured on real text, not assumed

`software/tools/bench_specdec_acceptance.py` measures `t` on wikitext-2. It needs no
RTL: acceptance depends only on the model's greedy continuation, and given that
sequence the accept rule is a deterministic walk — so one torch decode reproduces
`speculative_generate` exactly, in minutes instead of hours of simulation. A tiny-model
test (`test_acceptance_bench_simulator_matches_the_shipped_driver`) pins that simulator
to the **shipped** driver's pass accounting, so the reported number cannot drift away
from the code that actually runs.

**The model must be the chip.** Default is `--model fake-quant`: `NanoGPTFQReference`
under the frozen `weight_only_int8_quarot` preset — **W8A16** (INT8 weights, FP16
activations, static calibration scales, QuaRot). `--model w8a32` (FP32 activations) and
`--model fp32` are cross-checks, *not* the accelerator; a different model gives a
different greedy sequence and hence a different `t`.

**Indicative (W8A32 cross-check, 128-token prompt, 48 generated, 3 samples,
prompt-lookup max_ngram=2):** acceptance 32.3%, **tokens/pass 5.24**, **2.92× ⇒ ~5.08
tok/s** — past the ~3.27 tok/s DMA wall.

> ⚠️ **Headline pending.** The W8A16 (true-chip) re-measurement is the number of record;
> the 2.92× above is from the W8A32 cross-check and an earlier simulator that carried an
> off-by-one in the draft context. Treat 2.92× as indicative until the fake-quant run
> replaces it here.

⚠️ **Sample size matters, and it bit me.** A 64-token prompt generating 16 tokens
measured 1.05× — the n-gram had almost no context to match against. The lever looked
worthless. Do not price a draft on a short sample.

⚠️ **Where the win comes from, stated plainly.** GPT-2 124M's greedy decoding is
degenerate — it emits long literal repeats — and long literal repeats are exactly
what a prompt-lookup draft catches (the pass histogram has repeated 11-, 15- and
16-token accepts). A less repetitive model, or sampling instead of greedy, will show
a lower `t`. The floor is ~1.0× (adaptive fallback), never a loss.

### P=16 is optimal because it exactly FILLS THE MESH (measured)

For **P ≤ 16** the systolic/DMA/helper are FLAT in rows and only the SFU scales, so the
cost ratio is a clean line — **r(P) = 1 + 0.0302·(P−1)** (measured r = 1.0903 / 1.2108 /
1.4531 at P = 4 / 8 / 16; SFU 3.99× / 7.97× / 15.91×, exactly linear).

⚠️ **That law DIES at the mesh height.** Measured **r(32) = 2.6331**, far above the 1.936
the line predicts — because `M_pad=32 > SYSTOLIC_DIM=16` walks **two m-tiles at full
price** and the systolic **doubles**: 11,109,206 → 22,218,412, *exactly* 2.00×. So
"the systolic verifies P tokens for the price of one" holds **only up to 16 rows**.

| P | r | break-even | systolic | notes |
|---|---:|---:|---:|---|
| 4 | 1.0903 | 1.09 | 1.00× | |
| 8 | 1.2108 | 1.21 | 1.00× | |
| **16** | **1.4531** | **1.45** | **1.00×** | **fills the mesh exactly — the optimum** |
| 32 | **2.6331** | 2.63 | **2.00×** | two m-tiles; tok/pass does not grow ⇒ strictly worse |

P=32 is a double loss: it pays a second m-tile *and* confirms no more tokens (the n-gram
rarely finds >15 continuation tokens). This is the **same 16-row wall** that made lever H
(B=32 batching) a dud — the mesh is 16 rows, and nothing above that is free.

**Always MEASURE r at a new P — never extrapolate the line past 16.**
A stronger draft can only move the optimum *within* [1, 16]; `prefill_tokens` is already
a bundle parameter, so retuning is free.

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

**Where the SFU goes on the verify pass** (retire-gap profile, P=16, base 496):

| SFU op | cyc | % of SFU | cyc/op |
|---|---:|---:|---:|
| MASKED_SOFTMAX_FP32 | 3,875,472 | 38.4 | 26,913 |
| DEQUANT_ACCUM_FP32_SCALED | 2,006,231 | 19.9 | 1,503 |
| QUANT_FP32_INT8 | 1,142,633 | 11.3 | 865 |
| LAYERNORM_FP32 | 1,089,225 | 10.8 | 43,569 |
| MAX_ABS_REDUCE_FP32 | 863,563 | 8.6 | 1,159 |
| rest (DEQUANT/VADD/GELU) | 1,121,205 | 11.1 | |

Softmax leads but does not dominate — and note this SFU work is now **real**, not
padding: all 16 rows are live tokens. Unlike the b16 case (where lever C deleted
15/16 of it as waste), it can only be made *faster per row*, not deleted. Softmax is
three sequential 1-elem/cycle passes (MAX, EXPSUM, OUT); only **MAX is safely
widenable** — fp32 max is associative, fp32 add is **not**, so EXPSUM cannot be
reordered bit-exactly. That caps 1d at ~29% of softmax ⇒ r 1.45 → ~1.40, worth ~+4%
on the final number. Real, but the honest ranking is: **`t` (the draft) dominates
`r` (the hardware)** — going from t=2 to t=4 doubles throughput; a heroic SFU push
buys 4%. **The next big b1 win is a better draft, not more RTL.**

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
