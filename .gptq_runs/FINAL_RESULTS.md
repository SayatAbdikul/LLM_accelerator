# GPTQ Experiment — Final Results & Decision

## Bottom line

**Do NOT promote GPTQ.** The 256-token win (block 11 mlp.c_proj, golden
10690 → 6147, −42.5%) **does not hold** at 512 tokens. At the longer eval
window the picture flips:

| Eval window | Unmodified RTN | GPTQ block-11-c_proj | GPTQ vs baseline |
|---|---:|---:|---:|
| 256 tokens | 10690 | **6147** | **−42.5%** (win) |
| 512 tokens | **7693** | 9213 | **+19.8%** (regression) |

Both numbers are golden-simulator PPL with `relative_delta` between
golden and fake-quant of 0.0% on the 512-token results — the ground
truth disagrees cleanly with what the 256-token gate told us.

The 256-token "win" was an eval-window artifact. The 1024-token
confirmation called for in the original ship gate is hardware-blocked
(KV-cache buffer overflows at 1024-token contexts), so 512 tokens is
the longest feasible test through the golden simulator. **At 512
tokens the ship gate fails.**

The only reasonable interpretation: the test text's first 256 tokens
happen to be a distribution where minimizing FP32 layer-output MSE on
block 11's `mlp.c_proj` correlates with lower NLL; the next 256 tokens
are not. GPTQ's optimization target is layer-MSE, not end-to-end
perplexity, and the two only align on a narrow slice of this eval text.

## Composition experiments — all regress

We additionally tested the three compositions the user selected (Phase
2 of the follow-up plan), at 256 tokens, on top of block-11 GPTQ:

| Config | golden PPL | rel_delta | vs block-11-alone (6147) |
|---|---:|---:|---:|
| Block-11 GPTQ alone | 6147 | 5.2% | baseline |
| GPTQ + bias correction | 11385 | 0.0% | +85% |
| SQ α=0.3 + GPTQ | 8532 | 0.0% | +39% |
| SQ α=0.5 + GPTQ | 20960 | 0.0% | +241% |

α=0.4, 0.7 and SQ-alone control runs were OOM-killed at 4× concurrency
before producing PPL. With the trend already monotone-bad and the
512-token reversal of the underlying winner, finishing the α-sweep would
not have changed the conclusion. Skipped Step C (three-way) and Step D
ship gate for compositions accordingly.

## All 256-token results from the original sweep

These are the raw data this plan was working from. They remain valid
data points; the only thing that changed is our interpretation of the
"winner".

| # | Config | golden PPL | rel_delta | vs unmodified (10690) |
|---|---|---:|---:|---:|
| 1 | Unmodified RTN baseline | 10690 | 0.0% | — |
| 2 | RTN-via-GPTQ-path (n=0) | 9576 | 0.0% | −10.4% (calibration artifact*) |
| 3 | block 11 mlp.c_proj | 6147 | 5.2% | −42.5% (256-tok artifact) |
| 4 | block 11 c_fc + c_proj | 9223 | 0.0% | −13.7% |
| 5 | block 11 c_proj + attn.c_proj | 10988 | 0.0% | +2.8% |
| 6 | block 11 all-3 weights | 24716 | 0.0% | +131% |
| 7 | blocks {10,11} c_proj | 7445 | 0.3% | −30.4% |
| 8 | blocks {9,10,11} c_proj | 10291 | 0.0% | −3.7% |
| 9 | blocks {8,9,10,11} c_proj | 7054 | 0.0% | −34.0% |
| 10 | blocks {6-11} c_proj | 7191 | 0.0% | −32.7% |
| 11 | blocks {0-5} c_proj | 19928 | 3.3% | +86% |
| 12 | all 12 c_proj parallel | 17558 | 0.0% | +64% |
| 13 | all 12 c_proj sequential | 36720 | 1.3% | +243% |

\* The 9576 baseline came from running the GPTQ-path with empty
calibration (RTN fallback). The activation calibration adapts to the
dq-rounded weight. This effect is real but unrelated to GPTQ.

## What we shipped

The code lands; **no preset variant** does:

- `software/taccel/quantizer/quantize.py::gptq_quantize` — Frantar-style
  per-channel symmetric INT8 algorithm. **6 unit tests, all green.**
- `software/taccel/quantizer/__init__.py` — exports `gptq_quantize`.
- `software/taccel/runtime/fake_quant_reference.py::NanoGPTFQReference.forward`
  — added optional `capture: dict | None` keyword for diagnostic node
  capture (D3 hook).
- `software/tools/eval_with_gptq.py` — full standalone driver:
  - FP32 / fake-quant Hessian capture
  - Round-trip exactness with per-row uniform rescale + drift telemetry
  - Layer-MSE telemetry (weight MSE, output MSE, max residual)
  - `--use-fakequant-activations` flag (D3)
  - `--weight-search-n-seqs 0` RTN-via-GPTQ-path (D6)
  - `--sequential` Frantar-style block-by-block re-capture
- `software/tools/eval_with_compose.py` — composition driver chaining
  `apply_smoothquant_*` → `apply_gptq` → `apply_bias_correction`. Useful
  for any future composition work.
- `software/tests/test_quantizer.py::TestGPTQ` — 6 tests covering
  signature, RTN fallback, identity-Hessian = RTN, MSE improvement on
  correlated inputs, blocksize invariance, dead-column handling.
- `software/tests/test_eval_with_compose.py` — preset-variant pin test
  that asserts the GPTQ-block-11 preset was *not* added (rollback
  guard so a future reader doesn't reintroduce it without redoing
  the longer-window confirmation).
- **No preset variant** in `stage5_ptq.py` and **no change** to
  `PROMOTED_STAGE5_PTQ_PRESET`. The default preset stays
  `output_aware_mlp_lm_head_0_11_pc_full`.

## Phase 4 (rel_delta diagnostic) — not run

The plan's parallel diagnostic for the 5.2% rel_delta gap on the
256-tok GPTQ winner was deferred. Now moot — the winner failed the
512-tok ship gate, so the rel_delta question is academic. The
investigation would still be informative for future weight-refinement
work (the gap appeared specifically when a single weight was modified
through the GPTQ-path), but isn't worth pursuing on its own.

## Why the 256-tok win didn't generalize

Three plausible explanations, in decreasing order of likelihood:

1. **GPTQ's layer-MSE objective ≠ NLL minimum.** GPTQ minimizes
   `||(W - W_q) X^T||²` on a calibration distribution. The 256-token
   eval text happens to be drawn from a distribution where this
   objective correlates well with cross-entropy on next-token
   prediction. Beyond 256 tokens the eval text shifts to a region
   where this correlation breaks down. This is a known structural
   limit of GPTQ, not a bug in the implementation.

2. **Calibration distribution mismatch.** The activation calibration
   used 64 calibration sequences × 128 tokens = 8192 tokens drawn
   from `wikitext2_stage5_calibration.txt`. The 256-token eval is
   from `wikitext2_stage5_eval.txt`, but with only 256 tokens
   sampled, the eval-window distribution overlaps the calibration
   distribution more than the 512-token version does.

3. **Block-11 c_proj is the residual stream's last entry before the
   final LayerNorm + lm_head.** Small perturbations there have an
   amplified effect on the very-late tokens (which depend on more
   accumulated context). At 256 tokens these late-token errors are
   averaged with many easy early tokens; at 512 tokens they
   dominate.

All three suggest that any "GPTQ helps" claim in this codebase needs
to be confirmed at the longest feasible context window. 33-token
diagnostics are too noisy. 256-token alone is too short to be
trustworthy. 512-token is the longest the hardware fits.

## Recommendations

**Ship now**: nothing GPTQ-specific. The implementation lands, the
default preset doesn't change.

**Do not pursue**:
- Multi-block GPTQ (already abandoned; failed at every subset).
- GPTQ + bias correction (Step A failed at 256 tokens already).
- GPTQ + SmoothQuant (Step B failed at 256 tokens already).

**Could investigate** (low priority, for future work):
- Sequential GPTQ with the *fake-quant Hessian* (i.e., `--sequential
  --use-fakequant-activations` together). The current implementation
  supports both flags but they were never combined in this experiment.
  Hypothesis: matching the eval-time activation distribution at every
  block re-capture would close the gap between 256-tok and 512-tok
  results. Risk: D3 already showed FQ-Hessian destabilizes
  golden/FQ alignment (rel_delta=13.4% at 33 tokens).
- A different objective: GPTQ-equivalent rounding refinement that
  minimizes fake-quant cross-entropy directly (rather than FP32
  layer-output MSE). The infrastructure to capture per-token NLL is
  already in `evaluate_gpt2_perplexity`.
- Bias correction's plumbing — the +85% PPL regression with GPTQ + BC
  at 256 tokens was unexpected. The bias-correction tool was tested
  in isolation and matched expectations there. The interaction with
  GPTQ deserves a closer look (likely the `_input_act_scale_key`
  default of `6.0/127.0` when `calibration_scales` doesn't have the
  key — but this should not have happened on block 11).

## Files / data products

- `.gptq_runs/baseline_256tok.log` — 10690 unmodified 256-tok
- `.gptq_runs/D4_n128_256tok.log` — 6147 GPTQ winner 256-tok
- `.gptq_runs/followup/D_winner_512tok.log` — 9213 GPTQ at 512-tok
  (failed gate)
- `.gptq_runs/followup/D_baseline_512tok.log` — 7693 unmodified 512-tok
- `.gptq_runs/followup/A_gptq_bc.log` — 11385 GPTQ + BC
- `.gptq_runs/followup/B_sq{0.3,0.5}_gptq.log` — α-sweep partial
- All Phase 1/2 logs in `.gptq_runs/` (D1-D6, S1-S2-seq, late-block
  subsets, weight-type axis) — 30+ runs documenting the full sweep.
