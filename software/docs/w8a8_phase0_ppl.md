# W8A8 Phase 0 — PPL gate result

> **Historical experiment, reconciled 2026-09-03.** This result rejected the
> proposed W8A8 inter-op path; the maintained hardware target remains W8A16.
> Fixture paths, exact perplexities, and tool behavior below are the record of
> the 2026-05-23 run and have not been revalidated as current release gates.
> See the current [project status](../../docs/project_status.md) and
> [software map](../CODEBASE.md).

**Date:** 2026-05-23
**Verdict:** **RED** — abort the W8A8 plan as written, pivot to mitigations
or to W4A8 weight-only.

## Setup

- Checkpoint: `software/tests/fixtures/generated/gpt2_converted_nanogpt.pt` (GPT-2 124M)
- Calibration text: `software/tests/fixtures/generated/wikitext2_stage5_calibration.txt`
- Eval text: `software/tests/fixtures/generated/wikitext2_stage5_eval.txt`
- Calibration: 64 sequences × 128 tokens, percentile 99.9
- Eval: 257 tokens, context_len 256
- Preset: `weight_only_int8_quarot` (the production W8A16 baseline preset)

## Result

| Metric | W8A16 (baseline) | W8A8 (per-tensor inter-op) |
|---|---|---|
| **257-tok perplexity** | **54.78** | **363.30** |
| NLL | 4.003 | 5.895 |
| Delta vs W8A16 | — | **+308.5 PPL (+563%)** |

Gate thresholds: ≤60 GREEN, 60–75 YELLOW, >75 RED.
**Result: 363.30 → RED.**

The W8A16 baseline (54.78) matches the documented 55.76 within calibration
noise — the harness is correct. The W8A8 number reflects the actual cost
of static per-tensor INT8 quantization at every inter-op storage point.

## Why the plan's "≤+5 PPL expected" prediction was wrong

Phase 1 orientation noted that `NanoGPTW8A16SimulatorReference` already
INT8-quantizes activations at the matmul input (`w8a16_simulator_reference.py:32`).
This is true — but the matmul-input quantization is **dynamic per-tile**
(`max_abs` recomputed per matmul → `inv_fp16 = 127 / max_abs`), so each
tile re-picks its own optimal scale. Static **per-tensor** quantization
of the inter-op storage doesn't get that adaptivity: one scale must cover
the whole feature dimension's worst-case range, so most values fall in
the low-resolution interior of the INT8 grid.

With ~16 inter-op storage points × 12 transformer blocks = ~192 quantization
points per token, the per-tensor scale's resolution loss compounds. The
"matmul input already INT8" argument only holds when the INT8 step is the
*last* lossy step before integer arithmetic; here it's a *storage* step
followed by re-dequant for downstream FP32 SFU consumption — so the
inter-op INT8 is an *additional* lossy round-trip, not a free byproduct.

## Mitigation candidates (in order of effort)

Before declaring W8A8 dead on GPT-2 124M, try (in increasing complexity):

1. **Output-aware scale search** (1-2 days). Re-run scales through
   `apply_output_aware_*_scale_search_from_token_ids` (already in
   `runtime/calibration/output_aware.py`). The output-aware search minimizes
   downstream cross-entropy rather than MSE; can recover several PPL.

2. **Per-channel (per-feature-dim) activation scales** (3-4 days). Today's
   scale is `Dict[str, float]` — one scalar per tensor. Per-channel means
   one scale per feature dim of the activation tile. Storage cost: an extra
   N FP16 scales per tile (small vs. M×N INT8 storage). The ISA already
   supports per-channel scales (`DEQUANT_ACCUM_FP32_SCALED` uses 2N FP16),
   so the inverse `DEQUANT_INT8_FP32` could too if we extend the spec.
   This is a freeze §6 spec change to `0x1C`.

3. **Dynamic per-tile inter-op quantization** (1 week + spec change). Each
   inter-op storage point computes its own max-abs (a `MAX_ABS_REDUCE` per
   storage event), then uses that scale. Trades extra runtime cycles for
   accuracy; mirrors the matmul-input dynamic strategy.

4. **Mixed-precision per-block** (3-4 days). Run W8A8 only on blocks where
   ablation shows it's quality-neutral; keep W8A16 elsewhere. Smaller
   memory savings but recoverable PPL.

5. **Different model class** (if 124M is uniquely hard). LLaMA-7B or larger
   models have wider distributions and survive per-tensor INT8 better. Not
   applicable to the freeze-pinned 124M target.

## Alternative: pivot to W4A8 weight-only

Per the prior SoTA recommendation (`accelerator-completion-direction.md`),
W4A8 (INT4 weights + INT8 activations) is the SoTA-FPGA lever that fits
the existing QuaRot+AWQ infrastructure. PPL ceiling is well-characterized
for GPT-2 (typically +1-3 PPL above the W8A16 baseline). This was always
the parallel option; with W8A8 RED, it becomes the primary path.

## Phase 0 artifacts (this commit)

- `software/tools/w8a8_ppl_gate.py` — Phase 0 gate runner (commit-worthy).
- `software/taccel/runtime/w8a8_simulator_reference.py` — W8A8 NumPy reference
  (subclass of `NanoGPTW8A16SimulatorReference`, overrides forward pass with
  per-tensor INT8 inter-op round-trip). Commit-worthy as Phase-0 measurement
  infrastructure; **not deployed in any compiler / RTL path**.
- `software/docs/w8a8_phase0_ppl.md` — this file.

## Phases 1-9 status

**BLOCKED.** Do not start Phase 1 (ISA spec for 0x1C) until either:
- A mitigation above takes W8A8 PPL ≤60 on 257-tok GPT-2 124M, or
- The user explicitly chooses to proceed with a different model target,
  or
- The user pivots to W4A8 and re-scopes the plan accordingly.

## Reproduction

```bash
PYTHONPATH=software python3 software/tools/w8a8_ppl_gate.py \
    software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
    --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
    --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \
    --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \
    --max-eval-tokens 257 --context-len 256 \
    --calibration-n-seqs 64 --calibration-seq-len 128 \
    --ptq-preset weight_only_int8_quarot --json
```

Runtime: ~3 min wall on M-series Mac (PyTorch 2.3, numpy ≥1.24).
