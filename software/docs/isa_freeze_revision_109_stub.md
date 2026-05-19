# Freeze §5/§7 Revision STUB — #109 logits-level conformance metric

> **SUPERSEDED 2026-05-19 — folded into `isa_generation_freeze.md`.** The
> `LAYERNORM_FP32` ≤1-ULP real-data band + logits-metric are now in §4 item 7
> (REVISION 2026-05-19); the §5 definition-of-done status note records #109
> DONE. This file is retained only as the measurement-provenance record; it
> is safe to delete. The text below is the original draft.

**Status: DRAFT for the user to review, fill the Measurement section, and
commit into `software/docs/isa_generation_freeze.md`.** This is a separate stub
file by design: `isa_generation_freeze.md` and `simulator.py` are content-pinned
(`test_frozen_golden_sha_pin`, blob `131d3ef1…`) and are owned by the user
(freeze §5). Nothing here edits the pinned artifacts.

Date proposed: 2026-05-19. To be dated by the user on commit.

---

## Why this revision (the well-posedness problem)

Freeze §5 defines done as an end-to-end RTL↔golden byte-match within §7 FP16-ULP
bands on the GPT-2 124M `weight_only_int8_quarot` bundle. P6c/P6k/P6l proved the
RTL is **byte-EXACT (0 ULP) to the pinned golden across the entire 124M prefill
up to the first golden fp16 overflow** (`block0_out_proj`; the W8A16 124M MLP
dynamic range genuinely saturates fp16 → ±65504/NaN; model still 55.76 PPL).

**Past that boundary, per-tensor byte-match is ill-posed**: the golden itself
carries non-finite values, so a per-tensor ULP comparison (`compare()`) is not a
meaningful conformance signal there (freeze §4.x / §8 already document this; the
first node past out_proj, `block0_residual1`, shows ~18k ULP — expected, *not* a
bug). Freeze §7 already established the discipline for exactly this situation
(the gelu_new ≤3-ULP band, the Option-B non-finite contract): **a characterized
metric proposed *with* its first measurement, via the §6 revision mechanism —
never a guessed threshold.**

## Proposed metric (the past-overflow conformance signal)

For the GPT-2 124M `weight_only_int8_quarot` teacher-forced reference, in
addition to the §7 per-tensor bands **which remain authoritative up to and
including `block0_out_proj`**, conformance past the overflow boundary is the
**logits-level metric** computed by
`software/tools/rtl_cosim.py::logits_metric` over the final logits of every
teacher-forced position:

- **argmax-agreement rate** — fraction of positions where RTL and golden pick
  the same top-1 token.
- **min finite-masked cosine** — cosine similarity of the RTL vs golden logits
  vectors over entries finite on both rails (the worst position).
- **perplexity** — golden vs RTL perplexity from per-token NLL
  (`stable_cross_entropy` / `perplexity_from_nlls`), aggregated over
  finite-NLL positions, with explicit non-finite-position counts so the regime
  is characterized, not hidden.

Rationale for argmax+cosine+ppl (not a single number): past the overflow the
absolute logit magnitudes are partly governed by the saturating W8A16 path
(shared by golden and RTL); argmax/cosine measure *decision agreement* robust to
that, while the ppl delta measures end-task-equivalent divergence. The metric is
implemented and **self-validated on the tiny fixture**
(`test_rtl_cosim_gen2_logits_metric_tiny`: no overflow ⇒ argmax 100 %, cosine
1.0, ppl_delta 0.0, 0 non-finite — proves the metric code is correct before it
is trusted on 124M).

## Conformance band — TO BE PINNED FROM THE FIRST MEASUREMENT

Per §6/§7 discipline the band is **not guessed**. It is set to the first
measured value (rounded with the same conservatism §7 used for gelu_new),
recorded below, then enforced.

### Measurement — FIRST RUN 2026-05-19 (filled)

```
date                : 2026-05-19
run_program build   : DRAM_SIZE=1073741824 (pre-#116 binary; see Finding below)
token_ids           : [464, 3290, 318]   (1 prefill + 1 decode, teacher-forced)
first per-tensor divergence node:
                      block0_ln1  (pc 31, dtype fp32/FP16-storage,
                      row 0 col 489, fp16 ULP 1 > band 0;
                      golden 0.02655029296875  rtl 0.0265655517578125)
argmax_agree_rate   : 1.0
min_cosine          : 0.9999901952619332
ppl_golden          : 319.60334007478326
ppl_rtl             : 361.03379436609964
ppl_delta           : 41.43045429131638
max_abs_nll_delta   : 0.24378180503845215
n_nonfinite_golden  : 0
n_nonfinite_rtl     : 0
per-step            : step0 argmax 198==198, g_nll==rtl_nll==8.91332 (identical);
                      step1 argmax 338==338, g_nll 2.62084 vs rtl_nll 2.86462
```

### FINDING — the boundary is `block0_ln1`, NOT `block0_out_proj` (user attention required)

The first per-tensor divergence is **`block0_ln1` at 1 fp16 ULP**, *earlier*
than the freeze's documented "byte-EXACT (0 ULP) through `block0_out_proj`"
(§4.x / P6c; also `[[gen2-rtl-conformance]]`). Assessment:

- **Not a regression / not introduced by this work.** The 124M run used the
  **pre-#116** `run_program` binary. Independently, the post-#116 frozen-bundle
  freeze gate is **6/6 byte-exact (0 ULP)** on the tiny fixture
  (`test_compare_rtl_golden.py`), and the frozen 124M bundle emits 0 `flags=1`
  ⇒ #116 is provably behavior-neutral for it. So this is a **pre-existing
  real-token characterization fact**, surfaced by #109's first measurement.
- **Most likely a real-data LayerNorm band, exactly the case §7 anticipates.**
  `block0_ln1` is `LAYERNORM_FP32` (0x1A); it contains `1/sqrt(var+eps)`. §7
  pins LayerNorm at 0 ULP **on its characterized fixture** but explicitly notes
  (for masked_softmax, same discipline) "e2e is the final arbiter — if
  real-data drift appears it gets its OWN characterized band". P6c's 0-ULP
  result used a specific token (≈token_id 0); under real tokens
  `[464,3290,318]` a single-ULP fp16 rounding difference appears at the
  layernorm output (numpy var/rsqrt vs the RTL/DPI libm path) — the same
  numpy-vs-libm class as the characterized `gelu_new` ≤3-ULP band.
- **Recommended freeze action (user, §6/§7 revision):** add a measured
  **`layernorm_fp32` real-data band of ≤1 fp16 ULP** (same mechanism and
  discipline as the `gelu_new` ≤3-ULP band), with this measurement as its
  evidence. This is a *characterized-contract* decision, not a silent relax —
  it does **not** change `simulator.py` (golden SHA untouched); it adds a
  per-op-class band in `_freeze7_band` / `expect_fp16_ulp`, mirroring gelu.
- **Discriminating run DONE (2026-05-19, post-#116 binary, token_id=0,
  single-token prefill, `run_cosim`):** clean run (halted, no fault, no
  overlap, 976/976 captures); first per-tensor divergence =
  **`block1_ln1`, pc 4320, fp16 ULP 1** (golden −0.0004210472,
  rtl −0.00042128562). Conclusions:
  1. **Token-dependence CONFIRMED.** The boundary node moves with input
     (`block0_ln1` for `[464,3290,318]`, `block1_ln1` for token 0) — but in
     **both** independent token sets the first divergence is a **LayerNorm at
     exactly 1 fp16 ULP**. The mechanism is consistent; only the position
     shifts. This strongly grounds the proposed `layernorm_fp32` ≤1-ULP
     real-data band (two independent measurements, same 1-ULP signature).
  2. **"Byte-exact through `block0_out_proj`" is token-specific.** With
     token 0 it HOLDS (first divergence `block1_ln1` is in block 1, i.e.
     *after* all of block 0 incl. out_proj — consistent with P6c). With real
     tokens `[464,3290,318]` it does NOT (first divergence `block0_ln1`
     precedes out_proj). The freeze/memory claim is true only for the P6c
     token, not for arbitrary real input — exactly what §7's "e2e is the
     final arbiter" clause exists to catch.
  3. **#116 regression-neutrality EMPIRICALLY CONFIRMED on 124M.** The token-0
     run used the **post-#116** binary; the only divergence is the expected
     ≤1-ULP LayerNorm drift — no flags=1-class corruption (which would be
     gross, not 1 ULP). Combined with the post-#116 frozen freeze gate (6/6,
     0 ULP), #116 is regression-neutral on both the tiny and 124M paths.
- **User: the recall memory needs narrowing too.** `gen2-rtl-conformance.md`
  carries the verbatim claim "GPT-2-124M prefill byte-exact THROUGH out_proj"
  (and its `description:` line). Per the discipline this file is not silently
  edited; on accepting this revision, narrow that memory line to
  *"...through out_proj **for the P6c token**; under arbitrary real tokens a
  characterized ≤1-ULP `layernorm_fp32` real-data band applies (#109)"* so the
  recalled context stays accurate.

### Reading the measurement (do not over-read the PPL)

The **argmax-agreement (1.0) and min-cosine (0.99999)** are the decision-quality
signal — they say RTL and golden make the same predictions with near-identical
logit geometry even past the overflow. The **`ppl_delta=41.4` is illustrative,
not a population statistic**: it is the compounded mean-NLL shift over a
deliberately tiny **N=2** arbitrary-token run (step0 NLL is identical to 1e-5;
the entire delta is step1's 2.62→2.86 single-sample NLL move). It is not noise,
but it is not an eval-set perplexity either — a real PPL band needs a real
teacher-forced eval text (chunked manual run), not this 2-sample feasibility
probe. Pin the band on argmax/cosine; treat PPL as directional.

### Proposed pinned band (derive from the measurement above)

- argmax-agreement rate ≥ `<measured>` (propose: == measured, i.e. 1.0 if
  measured 1.0).
- min finite-masked cosine ≥ `<measured rounded down>`.
- |ppl_golden − ppl_rtl| ≤ `<measured rounded up>` (the analogue of gelu_new's
  ≤3-ULP characterized slack).

`software/tests/test_compare_rtl_golden.py::test_rtl_cosim_gpt2_124m_logits_metric`
emits all of the above; it deliberately asserts only run-cleanliness +
non-vacuity until this band is pinned (then the user adds the band asserts).

## Scope honesty (carry into the freeze text)

- This closes the *well-posedness* gap in §5: per-tensor byte-match through
  `block0_out_proj` (already proven) **plus** the logits metric past it.
- The literal §5 "257-tok" *decode* end-to-end remains opt-in/manual: ~5 min per
  RTL run × 257 ≈ infeasible in CI. #109 delivers the metric, the reusable
  multi-step infra (`run_cosim_sequence_logits`), and the 124M
  single-token-prefill first measurement (the past-overflow signal). Longer
  teacher-forced runs are a chunked manual extension, not a CI gate.

## §5 action items unblocked by this revision (user-owned)

Once the Measurement section is filled and this is merged into
`isa_generation_freeze.md` (new dated §7 sub-item + §5 status update), the
remaining §5 spec-file commit can proceed:
`software/taccel/isa/{opcodes,instructions,encoding}.py`,
`software/taccel/golden_model/simulator.py`,
`software/tests/test_compare_rtl_golden.py`.
