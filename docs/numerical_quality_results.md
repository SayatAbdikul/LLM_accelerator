# W8A16 numerical quality results

Updated: 2026-10-13

The GPT-2 124M compiled W8A16 path passes the explicitly selected 5% maximum
perplexity increase over original FP32 on the pinned 257-token evaluation
window. Its independent W8A16 reference agrees exactly on all 256 live logit
rows. This result requires actual QuaRot preprocessing and shared calibration;
passing a preset name to the bundle builder alone does not apply the transform.

The real-data RTL LayerNorm and GELU defects are repaired without modifying
the frozen golden model or widening a tolerance. The agreed bounded hardware
gate also passes: three successive steps produce exact golden logits and
whole KV while carrying the preceding hardware state. This is separate from
the 256-target software evaluation, not full-window RTL or corpus-wide proof.

## Acceptance contract

The user selected a 5% FP32-relative perplexity limit and explicitly accepted
bounded three-step hardware continuation plus full primitive regressions,
rather than all 256 RTL targets. The software evaluation covers all 256
targets in the fixed window. Reference agreement
retains the existing 2% relative PPL difference, minimum cosine 0.995 and
at least four shared top-five tokens. The gate checks every teacher-forced
step and live vocabulary element; missing/truncated rows, invalid targets,
nonfinite logits/NLL/PPL, changed input identities and failed metrics return
nonzero. No threshold is inferred from the observed error.

PPL limits apply to each evaluation window's aggregate perplexity, not
each target's individual exponentiated NLL. Cosine, ranking and finite-value
checks apply to every scored row. The bounded hardware gate uses the same
first three inputs, full compiled attention window and corresponding original
FP32/reference prefixes; it does not claim full-window RTL quality.

The original FP32 model executes before any in-memory PTQ transformation.
Calibration and evaluation use different pinned local text files. Exact full
evaluation/selected-calibration-window overlap checks pass; corpus article
and split provenance is not independently established. This is a reproducible
local-window result, not an official corpus-wide WikiText-2 benchmark.

## Canonical software evaluation

| Rail | Perplexity | Mean next-token NLL |
|---|---:|---:|
| Original FP32 | 53.693768911 | 3.983296960 |
| Compiled W8A16 | 55.032995153 | 4.007932917 |
| Independent W8A16 reference | 55.032995153 | 4.007932917 |

FP32-relative increase is **2.494193%**, below the 5% limit. Reference PPL
difference is zero; all 256 live rows match exactly, minimum reference cosine
is effectively 1 and every top-five overlap is 5/5. Each rail has 12,865,792
finite live logits; unused padded columns are excluded.

Configuration: converted GPT-2 124M checkpoint, local GPT-2 tokenizer,
`weight_only_int8_quarot`, actual data-free random orthogonal rotation with
the preset seed, calibration 8×32 at percentile 99.9, B1 one-token prefill
followed by 255 teacher-forced decodes, full attention window 256. Compiler
optimization switches remain disabled for this numerical baseline. The two
quantized rails share one complete calibration map with identical KV aliases.
The checkpoint, metadata, tokenizer files, texts, token IDs, transforms,
scales, package/tool versions and executable sources are pinned in the report.

The final software rerun takes 720.140 s under supervision, peaking at
1,906.344 MiB sampled child-process-group RSS. All 256 compiled steps explicitly
halt before their rows are accepted. Baseline compiler options are recorded as
six false flags. Invalid row types and boolean loss budgets fail closed;
isolated negative tests also exercise the cosine and top-five policies.

All 768 raw rows across the three rails and all ten exported static/program/
golden-capture artifacts are byte-identical to the prior canonical run.
Independent FP64 rescoring gives a 2.494194% increase; the sub-micro-NLL
difference from the recorded FP32 scoring is reduction precision, not changed
logits. The prior canonical run takes 430.929 s and the 33-token pilot is
debug-only; neither replaces the final source-epoch checks.

Evidence: `build/optimization-numerical-20261013/quality-quarot-257-v2/`
and `supervision-quality-quarot-257-v2/`. The gate evaluates compiled golden
execution, not 256 hardware executions. Hardware status is reported separately.

## Corrected reference and scale setup

The previous independent reference did not model several stored operands or
streamed operations correctly. The repaired implementation independently
models weight integer bins selected with FP32 scales before FP16 scale storage,
separately rounded embedding tables, FP16 normalization parameters and biases,
FP32 GELU constants, FP16 reciprocal-based static quantization, per-K-partition
FC2 scales and ordered FP16 partial sums, INT8 KV store/load scale aliases,
and the full padded softmax reduction window.

An implicit-calibration setup bug also passed `None` to both rails with
different meanings. Calibration now resolves once and the same map is passed
to both. The default softmax composite scale is corrected to match its
probability quantizer's 1/127 fallback rather than the 6/127 activation default.
Explicitly calibrated optimization campaigns do not use that fallback.

The reference does not invoke compiler emitters or the golden simulator to
construct expected values. Two synthetic compiled two-token cases match it
exactly for logits and complete INT8 KV bytes. Negative controls expose the
old rounding, partition, scale and fail-open-gate defects.

## Repaired RTL LayerNorm

A retained original-address prefix stops after the first real GPT-2 LayerNorm
at PC 31. Hardware/golden inputs, gamma and beta match exactly. Before the
repair, serial FP32 variance bits are `3e08d606` versus NumPy's `3e08d608`.
The serial calculation reproduces every old RTL output, including column
489's `26cd` versus golden `26cc`. The independent pairwise calculation
matches every frozen-golden intermediate and all 768 outputs.

Synth RTL and DPI now use eight partial sums with an ordered, bounded tree
for both mean and variance. The shared adder and registered feed stages are
retained. Both modes pass 513 exact vectors each: eight patterns across all
64 legal widths from 16 through 1024, plus the captured real row. Signed-zero,
subnormal, cancellation and extreme finite inputs are included. All 11
original SFU tests pass in each mode. The native campaign takes 74.686 s,
peaking at 551.109 MiB sampled group RSS.

Fresh fetch-off/on mode-1 runners build in 103.832 s, peaking at 586.266 MiB.
The fixed real-prefix replay then matches all 768 golden outputs at zero ULP
in 14.424 s with 2,525.906 MiB sampled group peak RSS. All 73 frozen
source/fixture identities remain unchanged. These measurements establish
numerical behavior, not new mapped area, timing or power results.

Evidence: `first-ln-001/`, `first-ln-002-fixed/`,
`supervision-ln-native-v1/` and `supervision-fresh-runners-v1/` under
`build/optimization-numerical-20261013/`.

## Repaired RTL GELU

The original-address 6,200-instruction prefix captures 2,185 intermediate
inputs/results and final STORE ranges under unchanged default memory pacing.
Before repair, its first out-of-band result was GELU at PC 3994, column 588:
both rails received FP16 `496f` (10.8671875), but RTL produced `7e00` (NaN) instead of finite
`496f`. The GELU exponential argument is approximately 108.91607, above the
FP32 exponential overflow boundary of approximately 88.72284. The former
`exp(2z)/(exp(2z)+1)` implementation therefore evaluated infinity/infinity.
Retiming parity alone previously allowed both primitive versions to agree on
this incorrect result. Both versions now normalize the positive-infinite
exponential ratio to 1/1 before division. Other NaNs are not masked, and the
guard uses the existing pipeline stage rather than adding latency.

Both SFU modes pass 12/12 checks, including a new 32-element signed-tail
regression. The native primitive gate passes 8,526,144 bit-exact comparisons,
including 2,285,857 independent positive-tail checks and 8,514,615 finite-input
checks. These counts overlap; they are not separate test totals. The repeated
LayerNorm gate again passes all 1,026 exact vectors. Fresh buffered/unbuffered
mode-1 runners build and the whole-core mode-1 hierarchy resolves. Yosys's
28,051 pre-process diagnostic problems are not an asserted structural proof;
this is elaboration, not mapped area, timing or physical signoff. The combined
native/build/elaboration campaign takes 381.853 s with 3,019.094 MiB sampled
child-group peak RSS.

The final strengthened primitive rerun additionally passes 18,522,288 exact
33-stage valid-timing checks, including bubbles and missing/extra-output
guards, and 11,527 independent NaN-input checks. All 8,526,144 comparisons
again pass. It takes 47.321 s with 104.453 MiB sampled group peak RSS; the
bounded drain cannot hang after a missing valid pulse. Evidence is in
`supervision-gelu-timing-final-v1/`.

The complete 12-target standalone FP32 suite also passes: addition, divider
pipelines at depths 2/3/4/5/6, square-root pipelines at depths 2/3/4/6,
exponential p18 and GELU p33. This rerun takes 85.363 s with 19.438 MiB sampled
group peak RSS. Evidence is in `supervision-all-primitives-final-v1/`.

The fixed real prefix makes the captured GELU result finite and within its
unchanged 3-ULP band. All FP16-typed captures among the 2,185 snapshots are
finite. The first remaining byte difference is a reload of four FP16 outputs
from an earlier GELU, each 1–2 ULP apart. Each rail's STORE/reload reproduces
its own output exactly; subsequent quantization, matmul and dequantization
at PCs 4206, 4212 and 4222 return to exact agreement. This is propagation of
permitted GELU approximation, not another diagnosed DMA fault or full-model
acceptance. The fixed prefix takes 48.624 s with 1,929.328 MiB sampled peak RSS.

Evidence: `block0-prefix-001/`, `block0-prefix-002-fixed/` and their supervision
directories, plus `supervision-native-final-v1/`. The original probe
takes 47.390 s, peaking at 1,642.750 MiB sampled group RSS. Sources and retained
inputs are unchanged; scalar-register bits are not exported by the native
runner, so the trace claims only the observed SRAM/DRAM results.

## Hardware continuation and release scope

The repaired three-step hardware replay **passes** the agreed bounded gate.
All three padded FP16 logit rows and each complete 18,874,368-byte KV region
match golden exactly. Every live logit is finite; reference PPL difference is
zero, minimum cosine is effectively 1 and each top-five overlap is 5/5.

| Rail on the same three-target prefix | Perplexity | Mean next-token NLL |
|---|---:|---:|
| Original FP32 | 1371.070346166 | 7.223346988 |
| Mode-1 RTL | 1317.182714160 | 7.183250427 |
| Independent W8A16 reference | 1317.182714160 | 7.183250427 |

The prefix's aggregate FP32-relative increase is −3.930333%, within the 5%
limit. These short-prefix values are not comparable to the 256-target PPL
table above. Independent FP64 rescoring reproduces the verdict and fully
rehashes all actual programs/raw dumps and 178 pinned files.

Both SFU/helper modes are 1 and instruction buffering is enabled. Steps 0,
1 and 2 cleanly retire 93,198 / 92,622 / 92,622 instructions in
32,378,793 / 34,166,338 / 34,166,625 cycles. The first input KV is zero;
each next input is byte-exact to the preceding RTL whole-KV output. Immutable
weights/scales/constants and all data outside emitted STORE ranges remain
unchanged. The run takes 1,784.732 s with 2,392.469 MiB sampled group peak RSS.

Evidence: `quality-rtl-replay-v2/report.json`, its raw programs/DRAM dumps,
`independent_verification.json` and `supervision-quality-rtl-replay-v2/` under
`build/optimization-numerical-20261013/`. No golden KV supplies execution
history. Exact state agreement is an observed diagnostic in addition to the
unchanged acceptance policies.

### Retained failure evidence

The earlier failed replay completed steps 0 and 1, then stopped the third step
after the first result exposed a large mismatch.
Step 0 cleanly retired all 93,198 instructions in 32,378,793 cycles, but its
reference PPL difference was 68.09% and top-five overlap was 3/5. All 50,257 live
logits differed (maximum absolute difference 24.171875); 15,123 KV bytes differed.
Layer 0's live K/V rows were exact, with divergence starting in layer 1.
The retained first-block probe identifies the GELU overflow described above.

The initial hardware KV is zero; step 1 consumes the preceding
hardware-generated whole KV region, never golden history. Input/export/runner
hashes, immutable model data and every live FP16 output are checked. Exported
golden data is comparison-only. Completed and interrupted artifacts remain in
`quality-rtl-replay-v1/` and `supervision-quality-rtl-replay-v1/`; they are failure
evidence, not an accepted three-step or 256-target hardware campaign.

The reusable [RTL quality CLI](../software/tools/w8a16_rtl_quality_gate.py)
requires the final canonical software report, its complete ordered HALT
coverage and the matching three-step export. It rejects incomplete retirement,
wrong hardware modes, changed identities, nonfinite outputs and unintended
writes. Validation/native/timeout failures retain terminal error reports and
partial artifacts. Exact golden logits/KV are separate diagnostics; they do
not replace the declared PPL/cosine/top-five acceptance criteria.

The two retained legacy default-preset perplexity tests compare different
W8A8/W8A16 arithmetic and are not removed or reclassified as passes. Historical
637/639 results and October 11–12 performance reports remain evidence of their
original source epochs; fresh performance validation is required before applying
those measurements to the corrected numerical RTL. Default promotion,
full-corpus quality and physical signoff are separate decisions.

## Final regression coverage

The current 781-node software collection contains 537 required tests and
244 optional cases. All 537 required tests pass with zero failures or skips;
the optional cases are deselected, not called passes. The supervised run takes
93.831 s with 689.094 MiB sampled group peak RSS.

All 32 affected prefill/generator/diagnostic-helper tests also pass, plus five
subtests: eight prefill cases, six LayerNorm-generator checks and eighteen
prefix-helper controls. This takes 13.381 s with 300.969 MiB sampled peak RSS.
All 148 focused numerical checks pass as well; these overlap the required and
affected selections and must not be added to a distinct total. Deliberate
overflow/special-value warnings are retained. All 73 frozen source/fixture
identities still verify, and no numerical threshold was relaxed.

Evidence: `required-final.xml`, `affected-final.xml`, `focused-final.xml`,
`supervision-required-final-v1/` and `supervision-affected-final-v1/`.

## Reproduction

Use the installed local dependencies and a resource supervisor on the 16 GiB
laptop. Run only one heavy job at a time; the measured campaigns use sampled
8 GiB child-group and 20 GiB actual free-disk guards, not hard OS limits.

```sh
PYTHONPATH=software .venv/bin/python software/tools/w8a16_quality_gate.py \
  --checkpoint software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
  --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
  --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \
  --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \
  --ptq-preset weight_only_int8_quarot \
  --max-fp32-ppl-increase-percent 5 --max-eval-tokens 257 \
  --rtl-export-steps 3 --out build/optimization-numerical-REPEAT/quality
make -C rtl/verilator -j1 test_sfu test_sfu_synth test_ln_quality
make -C rtl/verilator -j1 \
  test_fp32_add test_fp32_div test_fp32_div_p3 test_fp32_div_p4 \
  test_fp32_div_p5 test_fp32_div_p6 test_fp32_sqrt test_fp32_sqrt_p3 \
  test_fp32_sqrt_p4 test_fp32_sqrt_p6 test_fp32_exp_p18 test_fp32_gelu_p33
make -C rtl/verilator -j1 run_program_synth
PYTHONPATH=software .venv/bin/python software/tools/w8a16_rtl_quality_gate.py \
  --quality build/optimization-numerical-REPEAT/quality \
  --runner rtl/verilator/build/run_program_synth/Vtaccel_top \
  --out build/optimization-numerical-REPEAT/rtl-replay
```

The output directory must be new. The optional real-row vector requires the
retained prefix inputs and `LN_CAPTURED_PREFIX`; a fresh clone can still run
the 512 deterministic vectors. See the [quality CLI](../software/tools/w8a16_quality_gate.py),
[fail-closed metric checks](../software/taccel/runtime/numerical_quality.py) and
[independent reference](../software/taccel/runtime/w8a16_simulator_reference.py).
