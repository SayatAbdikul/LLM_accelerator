# GPT-2 INT8 PTQ — Phase A Diagnostic Findings

Status: **Phase A complete (2026-05-12).** The 115× FP32 gap on the production
preset `output_aware_mlp_lm_head_0_11_pc_full_bc` (6,174 PPL at 257-tok /
256-ctx vs FP32 ceiling 53.69 PPL) **cannot be closed by pure-offline
transforms on this codebase**. Closing it requires codegen/ISA extensions.

This doc captures the empirical journey, the two distinct failure modes
diagnosed in Phase A, and the recommended Phase D priorities.

## TL;DR

Six independent attempts at pure-offline PPL improvement converge on the same
structural ceiling:

| Attempt | Approach | PPL @ 257-tok | vs 6,174 baseline |
|---|---|---:|---:|
| QuaRot Phase 1B+ | Rotate residual stream weights | 14,322 | 2.3× worse |
| Phase 1B (FP32 residual) | Skip residual stream INT8 quant | 62,916 | 10× worse |
| AWQ | Activation-aware weight scaling | 50,544 | 8× worse |
| Tier 1 fc1 PC | Per-channel dequant on fc1 | 10,872 | 1.8× worse |
| Tier 1 out_proj PC | Per-channel dequant on out_proj | 982,333 | 159× worse |
| Tier 1 fc1+out_proj PC | Both PC paths combined | 1,212,272 | 196× worse + breaks gate |

Phase A's eval-length sweep revealed **two distinct failure modes**, neither of
which has a cheap fix:

1. **Position-correlated compounding** (AWQ): multiplier 3.63× → 6.07× → 6.84×
   → 8.19× as eval length grows 33 → 257 tokens. Errors compound through
   sequence positions — attention / KV cache / residual stream noise pattern.
2. **Per-step catastrophic** (PC expansion): multiplier 211× from token 1,
   roughly constant across eval lengths. Per-channel weight dequant produces
   matmul outputs whose magnitudes mismatch the per-tensor activation
   calibration; clipping at the next activation node destroys information at
   every step.

## Phase A diagnostic infrastructure

### A1: Per-layer FP32-vs-INT8 error decomposition

`software/tools/diagnose_per_layer_error.py` (commit `3cdda28`). Runs paired
FP32 reference and fake-quant references step-by-step with `trace=...`,
tabulates per-activation-node `rel_l2`, `rel_max`, and L2 contribution.

**A1 findings:**
- Top per-step errors are concentrated in the **attention V path** at late
  layers (per-head `attn_v` nodes, blocks 7-11; out_proj outputs in those
  blocks). NOT residual stream as some prior analysis assumed.
- HOWEVER, the top error nodes are *nearly identical* across the 6,174 PPL
  baseline and the 982,333 PPL `out_proj_pc` preset. The per-step `rel_l2`
  metric does NOT discriminate good from catastrophic presets.
- "Attention dominates per-step error" is a property of GPT-2 INT8 in general,
  not a discriminator for choosing remediation strategies.

### A2: Combined-PC fake-quant vs golden divergence

`default_plus_fc1_out_proj_pc_8` produced `fake_quant_ppl = 1,212,272` vs
`golden_ppl = 50,492` — a 24× divergence (`rel_delta = 0.958`, well past the
0.02 fidelity gate).

**A2 findings (partial, not root-caused):**
- Single-PC presets (fc1 alone, out_proj alone) have `rel_delta = 0` —
  fake-quant matches golden exactly.
- PC dequant formulas are identical in both implementations:
  `requant_pc[j] = input_scale × per_channel_weight_scale[j] / output_scale`.
- Bias encoding is correctly per-channel:
  `bias_int32[j] = round(bias / (input_scale × per_channel_weight_scale[j]))`.
- FP16 precision loss on `requant_pc` is at most 0.33% on `out_proj` at late
  layers (per-channel scale spread up to 100×) — not enough to explain 24×.
- **Notable**: `golden = 50,492` is much BETTER than `fake_quant = 1,212,272`.
  The compiled bundle handles combined PC better than the fake-quant
  reference. The fake-quant may be *underestimating* what combined PC could
  achieve.

**Not root-caused.** Bounds future PC-expansion work: adding multiple PC
layers introduces failure modes that singleton PC doesn't have.

### A4: Eval-length sweep (the discriminating evidence)

Ran `awq_with_bc` and `default_plus_out_proj_pc_8` at {33, 65, 129, 257}-tok
with production calibration (64 seqs × 128 tok), compared to the existing
production-default curve.

| Eval × ctx | default | awq_with_bc | out_proj_pc |
|---|---:|---:|---:|
| 33 / 32 | 3,171 | 11,509 | 670,571 |
| 65 / 64 | 2,929 | 17,785 | 1,005,157 |
| 129 / 128 | 4,010 | 27,429 | 626,025 |
| 257 / 256 | 6,174 | 50,544 | 982,333 |

| Eval | awq multiplier vs default | out_proj_pc multiplier vs default |
|---|---:|---:|
| 33 | 3.63× | 211× |
| 65 | 6.07× | 343× |
| 129 | 6.84× | 156× |
| 257 | 8.19× | 159× |

- **AWQ**: monotonically diverging multiplier. Position-correlated.
- **out_proj_pc**: catastrophic from token 1, multiplier fluctuates but
  doesn't monotonically grow. Per-step uniform failure.

## What this rules out

- **A single "iterative re-calibration" fix** does not unlock both failure
  modes. The pipeline already re-calibrates after rotation+AWQ and after BC
  (`software/taccel/runtime/gpt2_perplexity.py:264, 295`); re-calibrating
  more aggressively addresses neither pattern.
- **A single Phase B branch** (B1-B5 in the plan) targets one layer's
  precision, not the mismatch between weight dequant granularity and
  activation calibration granularity.
- **Further preset tuning** within the existing field set. All practical
  combinations were enumerated in Phase 0C and Tier 1.

## Phase D priorities (the only remaining path)

Closing the 115×→3× FP32 gap requires codegen/ISA extensions:

1. **Per-channel ACTIVATION quant for matmul outputs.** Extend
   `requant_pc_*_blocks` infrastructure to also calibrate per-channel OUTPUT
   activation scales. Addresses the PC-expansion failure mode directly: per-
   channel weight dequant + per-channel activation quant become a matched
   pair instead of a mismatch. ~3 weeks of codegen + ISA work.

2. **Runtime un-rotate-LN op in bundle.** Unlocks general-Haar QuaRot, which
   the simulation-only diagnostic (see
   `software/tools/diagnose_activation_outliers.py`) showed delivers
   +75-86% of the FP32→INT8 gap. The QuaRot Phase 1 infrastructure
   (`software/taccel/quantizer/rotation.py`, `software/taccel/quantizer/ln_fold.py`)
   is already in place — only bundle/codegen support is missing. ~2 weeks.

3. **Per-token activation quantization for residual stream.** Each sequence
   position gets its own activation scale (currently per-tensor). Addresses
   AWQ's position-correlated failure mode by giving activation quant fine-
   grained per-position control. ~3-5 weeks of codegen + ISA.

4. **Mixed-precision residual stream (FP16 residual, INT8 matmul).**
   Architectural. Wider datapath. Last resort if (1)-(3) fall short. ~3+
   weeks.

## Reusable Phase A artifacts

All committed and ready for future use:

- `software/tools/diagnose_per_layer_error.py` — paired FP32+fake-quant
  per-node error decomposition. Reusable for any preset comparison.
- `software/tools/diagnose_weight_only_qdq_ceiling.py` — pure-FP32 forward
  with QDQ weights (per-channel and mean-scale modes). Establishes the
  upper-bound contribution of weight-quant vs activation-quant.
- `software/tools/diagnose_activation_outliers.py` — full Tier 1/2/3 outlier
  characterization, rotation simulation, and decision logic.
- `software/tools/tabulate_ablation_results.py` — markdown tabulation of
  JSON ablation results.
- Eight A-pre presets staying registered as opt-in diagnostic targets:
  `quarot_baseline`, `quarot_with_bc`, `awq_baseline`, `awq_with_bc`,
  `awq_with_bc_searches`, `default_plus_fc1_pc_8`,
  `default_plus_out_proj_pc_8`, `default_plus_fc1_out_proj_pc_8`.

## Verification protocol for future Phase D work

When implementing any Phase D priority, validate against this baseline using
existing infrastructure:

```bash
# 1. Slow gate against current default
PYTHONPATH=software python3 software/tools/evaluate_gpt2_perplexity.py \
  software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
  --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
  --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \
  --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \
  --max-eval-tokens 257 --context-len 256 \
  --calibration-n-seqs 64 --calibration-seq-len 128 \
  --ptq-preset <NEW_PRESET> --json

# Acceptance: fake_quant_ppl < 6,174 AND rel_delta <= 0.02

# 2. Eval-length sweep to confirm position-correlation behavior
for L in 33 65 129 257; do
  python3 software/tools/evaluate_gpt2_perplexity.py \
    ... --max-eval-tokens $L --context-len $((L-1)) ...
done

# 3. Per-layer error decomposition vs baseline
python3 software/tools/diagnose_per_layer_error.py \
  software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
  --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
  --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \
  --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \
  --ptq-preset <NEW_PRESET> --n-eval-steps 32
```

## Commit reference

Phase A and the broader empirical journey are captured across:

| Commit | What |
|---|---|
| (prior QuaRot) | Phase 1B+ rotation infrastructure |
| `8835408` | Phase 0 PPL diagnostics (FP32 ceiling + KV-FP32 toggle) |
| `c03f7ba` | Phase 1B v1: FP32 residual stream toggle (inconclusive) |
| `37929dd` | Phase 1B v1 verdict + weight-only QDQ ceiling diagnostic |
| `be003f9` | Phase 1 Branch C v1: AWQ infrastructure (negative result) |
| `bd8e111` | Tier 1 PC-expansion presets |
| `3cdda28` | Phase A1: per-layer error diagnostic |
| (this commit) | Phase A summary + strategic verdict |

The plan file documenting the full execution log lives at
`/Users/sayat/.claude/plans/playful-enchanting-music.md`.
