# Diagnostics Rerun Report — 2026-03-29

## Scope

This report reruns the current canonical diagnostics stack from the latest codebase state:

- Full test suite
- Frozen local baseline benchmark preset
- Current best experimental benchmark preset
- Fresh trace diff between baseline and current best
- Fresh late-attention path diff
- Fresh block-impact reports for both baseline and current best

All commands were run from:

- [software](/Users/sayat/Documents/GitHub/transformer_accelerator/software)

## Commands Run

```bash
venv/bin/python3 -m pytest tests -q

venv/bin/python3 compare_golden.py \
  --diagnostic-preset baseline_frozen_local \
  --output /tmp/baseline_frozen_local_rerun.json \
  --trace-output /tmp/baseline_frozen_local_trace_report.json

venv/bin/python3 compare_golden.py \
  --diagnostic-preset current_best_sq_ln2_fc1_b0_8_10 \
  --output /tmp/current_best_sq_ln2_fc1_b0_8_10_rerun.json \
  --trace-output /tmp/current_best_sq_ln2_fc1_b0_8_10_trace_report.json

venv/bin/python3 diagnose_accuracy.py \
  --trace-diff-report \
  --trace-json-a /tmp/baseline_frozen_local_rerun.json \
  --trace-json-b /tmp/current_best_sq_ln2_fc1_b0_8_10_rerun.json \
  --trace-diff-output /tmp/current_best_vs_baseline_trace_diff.json

venv/bin/python3 diagnose_accuracy.py \
  --late-attn-path-report \
  --trace-json-a /tmp/baseline_frozen_local_rerun.json \
  --trace-json-b /tmp/current_best_sq_ln2_fc1_b0_8_10_rerun.json \
  --late-attn-output /tmp/current_best_vs_baseline_late_attn_path.json

venv/bin/python3 diagnose_accuracy.py \
  --block-impact-report \
  --trace-json-a /tmp/baseline_frozen_local_rerun.json \
  --block-impact-output /tmp/baseline_block_impact_rerun.json

venv/bin/python3 diagnose_accuracy.py \
  --block-impact-report \
  --trace-json-a /tmp/current_best_sq_ln2_fc1_b0_8_10_rerun.json \
  --block-impact-output /tmp/current_best_block_impact_rerun.json
```

## Verification

- Test suite: `167 passed`

## Benchmark Results

| Preset | Mean Cosine | P10 | Min | Top-1 | Top-5 Avg | Avg Cycles |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_frozen_local` | `0.8291` | `0.7467` | `0.6823` | `70%` | `71.0%` | `25,018,260` |
| `current_best_sq_ln2_fc1_b0_8_10` | `0.8314` | `0.7671` | `0.6816` | `90%` | `70.0%` | `25,018,260` |

### Delta: Current Best vs Baseline

- Mean cosine: `+0.0023`
- P10 cosine: `+0.0204`
- Min cosine: `-0.0007`
- Top-1 agreement: `+20 points`
- Top-5 overlap: `-1 point`

## Active Best Variant

The current best preset remains:

- `SmoothQuant`
- target set: `ln2_fc1`
- `alpha = 0.50`
- blocks: `0,1,2,3,4,5,6,7,8,10`

From the embedded compile manifest:

- baseline SmoothQuant: `off`
- current best SmoothQuant groups: `10`
- trace nodes captured per run: `328`
- trace events captured per run: `1769`
- instruction count: `28905`

Reference artifacts:

- Baseline rerun: [baseline_frozen_local_rerun.json](/tmp/baseline_frozen_local_rerun.json)
- Baseline trace: [baseline_frozen_local_trace_report.json](/tmp/baseline_frozen_local_trace_report.json)
- Current best rerun: [current_best_sq_ln2_fc1_b0_8_10_rerun.json](/tmp/current_best_sq_ln2_fc1_b0_8_10_rerun.json)
- Current best trace: [current_best_sq_ln2_fc1_b0_8_10_trace_report.json](/tmp/current_best_sq_ln2_fc1_b0_8_10_trace_report.json)

## Trace Comparison

Fresh trace diff:

- [current_best_vs_baseline_trace_diff.json](/tmp/current_best_vs_baseline_trace_diff.json)

Late-attention path diff:

- [current_best_vs_baseline_late_attn_path.json](/tmp/current_best_vs_baseline_late_attn_path.json)

### Stage-Level Delta on Common Traced Images

Common traced images:

- `785`
- `2006`
- `2685`
- `5037`

Analysed late blocks:

- `6`
- `8`
- `10`
- `11`

Mean stage delta, current best minus baseline:

- `qkt`: `+0.0022`
- `residual2`: `+0.0012`
- `classifier`: `+0.0001`
- `gelu`: `-0.0006`
- `fc2`: `-0.0014`
- `attn_v`: `-0.0035`
- `softmax`: `-0.0043`
- `concat`: `-0.0046`
- `out_proj`: `-0.0055`

### Worst Fresh Node Regressions

- `block11_head1_attn_v`: `-0.0564`
- `block11_head0_softmax`: `-0.0330`
- `block11_out_proj`: `-0.0305`
- `block10_head0_softmax`: `-0.0253`
- `block11_concat`: `-0.0246`

### Late Attention Path Read

Mean late-path delta by stage:

- `value`: `+0.0021`
- `qkt`: `+0.0025`
- `softmax`: `-0.0044`
- `attn_v`: `-0.0037`

Worst late heads:

- `block11 h1`: `attn_v -0.0564`, path `-0.0160`
- `block11 h0`: `attn_v -0.0097`, path `-0.0101`
- `block10 h0`: `attn_v -0.0037`, path `-0.0061`

Interpretation:

- The current best variant improves the model broadly enough to win on mean cosine, `p10`, and top-1.
- The remaining downside is still concentrated in late attention, especially `block11`.
- The dominant fresh regressions are still `block11` `softmax -> attn_v -> concat -> out_proj`, not the MLP path.

## Block Impact

Baseline block-impact report:

- [baseline_block_impact_rerun.json](/tmp/baseline_block_impact_rerun.json)

Current best block-impact report:

- [current_best_block_impact_rerun.json](/tmp/current_best_block_impact_rerun.json)

### Baseline: Top Worsening Blocks

- `block11`: share `0.1338`, mean delta `-0.0267`
- `block04`: share `0.1030`, mean delta `-0.0235`
- `block02`: share `0.0998`, mean delta `-0.0230`
- `block03`: share `0.0991`, mean delta `-0.0226`
- `block07`: share `0.0913`, mean delta `-0.0204`

### Current Best: Top Worsening Blocks

- `block11`: share `0.1558`, mean delta `-0.0317`
- `block06`: share `0.0942`, mean delta `-0.0216`
- `block09`: share `0.0935`, mean delta `-0.0179`
- `block07`: share `0.0920`, mean delta `-0.0214`
- `block04`: share `0.0906`, mean delta `-0.0213`

### Block-Level Interpretation

- `block11` remains the dominant tail problem in both baseline and current best.
- The current best variant reduces a lot of earlier MLP damage enough to improve average metrics, but it makes `block11` an even larger share of the remaining loss budget.
- The problematic band is now narrower and later than before.

## Per-Image Read

### Baseline Mismatches

- `632`
- `1000`
- `1353`
- `2473`
- `2685`
- `5037`

### Current Best Mismatches

- `2473`
- `2685`

### Important Tail Cases

- `2685`
  - still the hardest image
  - current best final cosine: `0.6816`
  - dominant worsening block: `11`
  - strongest fresh regressions: `block11_head1_softmax`, `block11_head0_softmax`, `block11_concat`

- `2473`
  - still flips top-1 under the current best variant
  - not primarily a `block11` story in the same way as `2685`

- `785`
  - remains a strong low-cosine image
  - still shows heavy late `GELU` damage

## Bottom Line

The rerun confirms the same overall picture, but with fresh reproducible artifacts:

- The codebase is stable enough for deeper diagnostics.
- The canonical frozen local baseline is still `0.8291 / 0.7467 / 0.6823 / 70%`.
- The best current experimental path is still `LN2 -> FC1 SmoothQuant`, `alpha=0.50`, blocks `0-8,10`.
- That variant now re-confirms as the best measured tradeoff:
  - better mean cosine
  - better `p10`
  - much better top-1
  - only a tiny min regression

The remaining accuracy bottleneck is now highly localized:

- late `block11` attention
- especially `softmax`, `attn_v`, `concat`, and `out_proj`
- with `2685` still the clearest tail outlier

## Recommended Next Step

Do not reopen broad sweeps.

The next highest-signal work item is:

- a surgical `block11` tail fix layered on top of the current best SmoothQuant variant

Most promising order:

1. per-image tail diagnostics focused on `2685` and `2473`
2. block-11-only causal splice or replacement experiments

## Rollout Status Addendum

The later ISA rollout work after this rerun has completed the first two items from the residual-precision plan:

1. `DEQUANT_ADD` ISA + simulator
2. `block11 residual1` only

Everything after that in the original rollout order is still pending:

3. all `residual1`
4. `block11 residual2`
5. all `residual2`
6. combine with `SOFTMAX_ATTNV block11`
7. then FC `REQUANT_PC`
8. then revisit `GELU-from-ACCUM`

### What Was Implemented

- `DEQUANT_ADD` is now implemented end to end in the ISA, assembler/disassembler, simulator, compiler, and codegen.
- The live model path needed strip-mined `out_proj -> residual1` handling, so `block11 residual1` was implemented on the real strip-mined path, not only on the simple matmul path.
- Full verification after this work is green at `184 passed`.

### Measured Result for `block11 residual1`

Reference control on the current best SmoothQuant preset:

- [dequant_add_residual1_control_v4.json](/tmp/dequant_add_residual1_control_v4.json)
  - mean cosine `0.831394`
  - `p10 0.767090`
  - `min 0.681559`
  - top-1 `90%`
  - cycles `25,018,260`

`DEQUANT_ADD` with the same default residual scale:

- [dequant_add_residual1_block11_default_v4.json](/tmp/dequant_add_residual1_block11_default_v4.json)
  - identical accuracy
  - lower cycles: `24,973,316`

So the operator itself is safe and slightly cheaper, but not an accuracy win by itself.

### Residual1 Scale Search Result

We then tested bounded `block11` residual1 scale variants on top of `DEQUANT_ADD`:

- [dequant_add_residual1_block11_blend025_v4.json](/tmp/dequant_add_residual1_block11_blend025_v4.json)
  - scale `0.13114`
  - `0.829939 / 0.761664 / 0.667215 / 75%`
- [dequant_add_residual1_block11_blend050_v4.json](/tmp/dequant_add_residual1_block11_blend050_v4.json)
  - scale `0.15355`
  - `0.825653 / 0.755180 / 0.644217 / 80%`
- [dequant_add_residual1_block11_pct990_v4.json](/tmp/dequant_add_residual1_block11_pct990_v4.json)
  - scale `0.02340`
  - `0.823353 / 0.727792 / 0.669157 / 75%`

These all regressed versus the neutral default-scale `DEQUANT_ADD` path.

### Current Read

- `DEQUANT_ADD` is useful as a structural and performance primitive.
- Freeing `block11 residual1` scale did not improve cosine in any tested form.
- The accuracy upside from this lane does not currently come from residual1 scale search.

### Next Recommended Step

The best next step is no longer `all residual1`.

Given the neutral `DEQUANT_ADD` result and the earlier promising fused-attention result, the next highest-signal experiment is:

1. combine neutral `DEQUANT_ADD` on `block11 residual1` with `SOFTMAX_ATTNV block11`
2. benchmark that combination against the current best SmoothQuant preset
3. only if that combination shows value, consider expanding residual precision further

In short:

- completed: `1`, `2`
- ready next: `6`
- not recommended right now: `3`, `4`, `5` without a better causal signal
3. only then another narrow block-11 intervention
