# Freeze §5/§7 Revision STUB — #109 logits-level conformance metric

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
- **Independent confirmation suggested:** rerun with P6c's original token to
  confirm the boundary is token-dependent (expected: 0-ULP through out_proj
  reproduces; the ≤1-ULP block0_ln1 drift is real-token-specific).

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
