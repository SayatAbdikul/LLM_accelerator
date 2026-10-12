# October optimization implementation and validation

**Verification dates:** 2026-10-11–12  
**Base commit:** `978f126` plus the working-tree changes  
**Status:** opt-in integration validated; numerical-quality and physical gates remain open

This report implements the [optimization roadmap](optimization_roadmap.md).
It distinguishes implemented candidates, accepted behavior, measured benefits,
and unexecuted or failed promotion gates. Scheduling changes do not amend the
ISA or the frozen golden arithmetic.

## Implemented work

### Correctness and measurement foundation

- Corrected performance runners and synthesis configuration explicitly select
  both synthesizable datapaths, `SFU_SYNTH_MODE=1` and `HELPER_SYNTH_MODE=1`.
  Runner JSON reports actual parameters and instruction-buffer selection.
- DMA splits normal and transposed transfers at 4 KiB boundaries, rejects
  effective-address overflow before truncation, and exercises response faults
  and backpressure. AXI read arbitration retains its grant under stall. A
  real-top negative control reproduced unstable STORE WDATA during a legal
  same-buffer/different-row systolic read; a stalled-payload register fixes
  it without changing unstalled STORE cadence (45/42 cycles for fetch off/on).
- Hardware-target IR validation rejects unsupported W4/noncausal requests,
  unknown operations, duplicate names, missing inputs/weights, and malformed
  shapes/dtypes. `research-golden` remains a separate explicit target.
- Repaired outdated regression contracts and checkpoint provenance. The
  selected like-for-like regression reference independently models the active
  W8A16 path. The retained legacy PPL comparisons still use W8A8 and remain
  failed; existing numerical tolerances are unchanged.
- Pinned 73 golden/ISA/assembler/helper/fixture dependencies. The required
  verification profile fails skipped mandatory gates; optional omissions are
  explicitly deselected, not counted as successful verification.
- Benchmarks retain source/runner/model/calibration/program identities, actual
  FP16 DRAM logits, DMA directional bytes/beats/bursts, and systolic phase
  counters. The legacy `summary.logits` field is an INT32 ACCUM diagnostic,
  not the decoder's FP16 logits.
- Packed batched attention now conservatively admits the complete AV-output
  cohort and workspace against the available ABUF capacity before retaining
  it. Otherwise it spills from the
  first output, including short contexts that the former K≥64-only policy
  missed. Per-compilation decisions reset on regeneration and fallback;
  arithmetic, K order, scales and the ISA are unchanged.

### Compiler, host, and fetch candidates

All six compiler experiments are independent `CompilerOptions` switches;
explicit `CompilerOptions()` preserves the corrected baseline. Quantization
reuse reserves scale-register lifetimes, checks buffer capacity/fragmentation,
and falls back to a fresh uncached whole-graph schedule if admission fails.

| Candidate | Implementation and declared primary metric |
|---|---|
| Real-row FC2 loads | Transfer only valid FP16 rows, explicitly initialize padding; DMA bytes/cycles |
| One-beat instruction buffer | Retain both 64-bit halves; poison refill on restart/STORE and preserve AXI stability; reads/cycles |
| QKV quantization reuse | Reuse immutable input/version/shape/precision and exact scale bits; MAX_ABS/QUANT work |
| FC2 K-tile reuse | Preserve local scales, K order and first-K bias while retaining admitted tiles; DMA/SFU work |
| Indexed liveness | Pre-index final-use releases; compile time with identical instructions/data |
| Shared layout and data deduplication | One relocatable prefill/decode layout, consumer-union pruning; bundle bytes/compile cost |
| Skip final unused decode | `generate(..., final_token_commit=False)` uses prefill plus N−1 decodes for N positive tokens without EOS; host calls |

`final_token_commit=True` retains the explicit continuation behavior. N=0,
EOS, returned-token equivalence and continuation are covered separately.
The golden host runner also distinguishes sequential prefill from lockstep
B16/B32 priming: the latter uses position zero and a one-token causal window
for every stream. Nonzero-position-embedding regressions compare priming and
continuation logits/KV against the matrix's explicit runtime patching.
Sequential chunk APIs reject lockstep bundles before mutating their state.

Packed next-attention-group prefetch was investigated and not integrated:
the current group occupies WBUF, an ABUF destination conflicts with systolic
Port S ownership, and staging through helper copies has no demonstrated net
benefit. It remains a rejected/deferred candidate, not completed overlap work.

### Physical feasibility

The portable [leaf flow](../rtl/asic/tools/block_flow.py) accepts explicit
PDK/corner/Liberty/LEF inputs, verifies physical cell-area units, captures
resource use, and optionally runs zero-wire STA. It never downloads a PDK,
silently substitutes a library, or retries whole-SFU flattening.

An isolated SKY130 macro-bank adapter passes 96,804 cycles against the inferred
SRAM across four geometries and eight startup cases, including 2,518 same-row
A-write/B-read collisions. A one-entry delayed-write overlay forwards the
previous logical contents and prevents undefined physical macro collisions;
both deliberately broken negative controls fail. The adapter is not bound into
the chip, and macro timing collateral is not yet validated. The selected 2 KiB macros
would occupy 63.737 mm² for 448 KiB before peripheral logic; this is not a
measured full-chip area or a universal SRAM density bound. See
[ASIC flow and limitations](../rtl/asic/README.md) and the
[opcode-by-opcode SFU storage inventory](sfu_storage_inventory.md).

This is sampled-cycle functional equivalence: the vendor model responds after
the falling edge, and checks observe settled data before the next rising-edge
consumer. It is not continuous-waveform equivalence or evidence that physical
macro timing meets the core clock. The bank flow refuses mixed-process
SKY130-macro/ICSprout55-periphery combinations and reports macro footprint
separately from standard-cell area.

The 128-bit × 1,024-row (16 KiB) bank maps to the expected eight macros and
27,817.9296 µm² of standard-cell periphery. Its macro footprint is
2,276,307.792 µm², giving 2.304126 mm² of cell-plus-macro instance area, not die
area. Mapping takes 0.666 s and peaks at 54,378,496 bytes RSS; sv2v peaks at
295,190,528 bytes. All 25 peripheral cell types pass LEF area verification.
The pinned macro's 683.1 × 416.54 µm footprint is 284,538.474 µm² and matches
its Liberty area; an earlier manual 6 µm² discrepancy was an arithmetic error.
The report still records any Liberty/LEF difference instead of concealing it.
Local evidence: `rtl/asic/results/optimization-20261011/sky130hd/sram-bank-1024/report.json`.

## Verified measurements

### Functional verification

The final required Python profile, including the short-context capacity,
stored-row and host-priming fixes, passes all 395 tests, with 244 optional
tests deselected and no failures or mandatory skips. Pytest takes 89.72 s
(91.006 s wall). All eight affected prefill tests also pass
(12.01 s pytest, 13.059 s wall). The supervisor reports 671,744,000 bytes
maximum cumulative child RSS, not an aggregate process-tree peak.
There are 639 collected nodes. Warnings from intentional overflow/special-value
coverage are retained; reference payloads and numerical tolerances are unchanged.

Fresh offline official-Hugging-Face/converted-checkpoint FP32 parity also passes
both tests (5.93 s pytest, 8.253 s wall, 1,093.78 MiB sampled process-group peak
RSS). Its unchanged cosine, top-10 and p99-error gates validate conversion,
not quantized RTL model quality.

The reconciled result is 637/639 passing nodes: 395 required and 242 optional.
Both remaining failures are legacy cross-arithmetic perplexity comparisons,
not hidden skips. All 54 formerly pending nodes execute: 51 pass and three
fail, with no skips, incompletes, timeouts or resource-limit stops. That
original campaign closes at 636/639 against identical 177-file Python source
hashes. Its 48 regular nodes finish with 46 passes/two failures in 2,230.673 s,
peaking at 4,122.797 MiB sampled test-process-group RSS; the five slow nodes
finish with four passes/one failure in 1,149.071 s, peaking at 3,301.875 MiB.
Every phase verifies source, fixture and runner identities at closure.

The third original failure is a stale acceptance smoke: its generated
six-layer checkpoint was sent to a default preset targeting twelve layers.
The test now explicitly selects the compatible `control` preset. Its existing
assertions are unchanged; production defaults, arithmetic and fixtures are
not modified. A verifier permits only those two added CLI literals in the
AST, checks the one-file/177 hash delta and exact unchanged 639-node collection,
then reruns both CLI smokes and their explicit importer. All three pass in
45.49 s pytest (47.195 s supervisor), with 2,753.844 MiB sampled child-group
peak RSS. The resulting ledger combines 634 dependency-unchanged observations
with three fresh passes. It does not claim all 639 tests reran under one source
hash, and preserves the original failed logs and both unrelated PPL failures.

A separate 180-node rerun previously closed the earlier-source evidence gap
with zero failures/skips in 230.981 s and 830.56 MiB sampled group peak RSS.
The 124M test passes its finite-output characterization contract, not an
accuracy threshold; its numerical differences are recorded below.
Exact IDs, source identities, logs, JUnit and verified retention manifests
are under `build/optimization-20261011/python-final/`: `earlier-current-180/`,
`final-optional/` and `test-contract-repair/`. Original temporary evidence is
preserved under `/private/tmp/taccel-python-light-20261011.qG9iCl/`.

### Clean source reproduction

An isolated export of 428 pinned source/document files reproduces the accepted
opt-in configuration without copying prior build products. Both native runners
are freshly compiled with `SYSTOLIC_ARCH_MODE=1` and a 1 GiB DRAM range;
runtime metadata confirms SFU/helper modes 1/1 and fetch disabled/enabled.
Their SHA-256 hashes exactly match the runners used for the final performance
campaigns. Existing installed dependencies are reused: this is a clean-source
and clean-build check, not a fresh dependency installation.

| Reproduction gate | Outcome | Supervisor time (s) | Sampled process-group peak RSS (MiB) |
|---|---|---:|---:|
| Fetch-off native build | Passed | 52.614 | 574.438 |
| Fetch-on native build | Passed | 51.539 | 601.156 |
| Required Python profile | 395 passed, zero failures/errors/skips | 92.758 | 687.656 |
| Affected prefill profile | 8 passed, zero failures/errors/skips | 13.398 | 299.734 |
| Portable ASIC contracts | 39 static/mocked tests passed; no physical tool run | 4.145 | 36.125 |
| Tiny d128/L2 combined profile | 8 executions, 8 golden comparisons and 4 exact A/B pairs passed | 17.553 | 275.641 |

The tiny matrix uses the pinned trained fixture, B1 position zero and one-token
prefill followed by two teacher-forced decodes. All six compiler options are
enabled only for the candidate. The closing gate independently checks finite
live FP16 logits, complete KV, immutable weights, actual runner modes and
payload hashes; there are no fallbacks. These repeat tests do not increase the
distinct 637/639 Python count or establish trained GPT-2 quality.

All 428 original/exported source hashes remain unchanged throughout execution.
Final documentation updates occur afterward and do not modify production code
or that retained snapshot. Jobs run serially with sampled process-group RSS
and a 20 GiB free-disk guard, not instantaneous OS-enforced resource limits.
The run closes with 32.072 GiB free. Retention preserves 559 files totaling
43,323,287 bytes, including source, logs, JUnit, fixture, tiny payloads, runners
and verification helpers; every copied hash is verified. Original evidence is
unchanged. Durable evidence: `build/optimization-20261011/clean-source-final/`,
especially `proof/result.json`, `proof/identity.json` and
`retention_manifest.json`.

### Full-model reference characterization

The optional 124M test passes one exact selected node in 702.241 s supervisor
time (699.54 s pytest), with 2,850.5 MiB sampled test-process-group peak RSS.
Source, fixture and runner identities remain pinned. Both prefill and retained-KV
decode produce finite live FP16 logits on both rails; assertions also check
clean execution. Cosine, argmax and perplexity measurements are not thresholded
acceptance criteria in this test.

| Step | Golden argmax | RTL argmax | Cosine | Golden target NLL | RTL target NLL |
|---:|---:|---:|---:|---:|---:|
| 0 | 198 | 262 | 0.999596715 | 9.399083 | 9.404213 |
| 1 | 338 | 318 | 0.999885321 | 2.380519 | 1.506474 |

The snapshot comparator flags a one-FP16-ULP difference for `block0_ln1`
at PC 31, row 0/column 489, against its exact-operation band zero. Argmax
agreement is 0/2 despite high cosine similarity. Two-target perplexities are
361.333467 golden and 234.005235 RTL; these are not corpus-level quality results.
This pass must not be labeled full-model numerical conformance or used to
promote an accuracy claim. The behavior-preserving optimized/baseline RTL
comparisons remain separate from this RTL/golden reference difference.

The capture is anchored to a post-SFU `SYNC` (`0x02`), not a LOAD; static
retire/writeback ordering is consistent with the completed LayerNorm output.
The trace's `fp32` semantic label still refers to FP16-backed SRAM values.
Different FP32 reduction rounding between the serial RTL and NumPy reference
is a diagnostic lead, not an established cause. The first flagged element is
not a maximum-error bound. A further diagnosis needs retained LayerNorm inputs,
gamma/beta, intermediate reductions and write/retire traces; this test's
temporary snapshots were deleted on normal completion. No tolerance or
frozen arithmetic is changed on the strength of this measurement.

Original evidence: `/private/tmp/taccel-python-light-20261011.qG9iCl/`
`final_deferred_124m/report.json` and its
`00_test_compare_rtl_golden/pytest.log`, JUnit and exact phase records. Final
optional-test reconciliation/retention is complete under
`build/optimization-20261011/python-final/final-optional/`.

### Optional legacy perplexity comparison

The 257-token/256-target default-preset research gate fails its unchanged 2%
relative-difference assertion: compiled golden perplexity is 1,969.913069,
legacy fake-quant perplexity 45,804.510880, and relative difference 95.699303%.
The pre-PTQ FP32 baseline is 53.693769. This uses
`output_aware_mlp_lm_head_0_11_pc_full_bc`, not the optimization matrix's
`weight_only_int8_quarot` workload. Pytest takes 469.96 s; sampled module peak
RSS is 3,227.469 MiB. It is a numerical failure, not a resource omission.

No RTL executes in this test. The runtime explicitly identifies the comparison
as compiled W8A16/FP16 logits versus retained W8A8/INT8-logit fake quantization.
Their embedding quantization, LayerNorm epsilon and requantization policies
differ; output-aware search also targets that legacy reference. The failure
therefore does not establish a same-arithmetic conformance defect or invalidate
the matched RTL optimization results. It remains a failed research assertion,
and neither poor quantized perplexity supports quality promotion. No threshold
is loosened. A useful next diagnostic would compare the identical post-PTQ
weights, biases and scales with the independent active W8A16 reference before
migrating this legacy quality contract.

Evidence: the original optional campaign's
`final_deferred_default_heavy/07_test_gpt2_perplexity/pytest.log`, JUnit and
exact-node records under `/private/tmp/taccel-python-light-20261011.qG9iCl/`.

### Slow optional measurements

The production-window weight-only INT8 gate and both 33/257-token TurboQuant-KV
gates pass their existing assertions. The 257-token software-only TurboQuant
experiment records a rotated-reference anchor PPL of 55.8973 versus 57.8117
for 4-bit MSE KV quantization (about 3.4% higher). This is not a claim that
4-bit KV is implemented by the RTL target. Its pytest time is 126.09 s and
sampled module peak RSS 2,012.516 MiB.

The 257-token FP16 simulator-backed test also passes its unchanged
finite/structural contract in 413.79 s pytest (415.109 s module), peaking at
3,042.391 MiB sampled process-group RSS. It records simulator PPL 174.2368,
independent like-for-like reference PPL 231.3346 (24.68% relative gap), and
weight-only host-reference PPL 53.4212. The test reports that gap without
bounding it; its pass does not establish numerical equivalence or trained
quality. The production-calibration comparison instead fails its unchanged
2% assertion: golden PPL 2,146.373680 versus legacy fake-quant PPL 5,733.220343,
a 62.5625% relative gap; its pre-PTQ FP32 baseline is 53.693769. This uses
64×128 calibration tokens and the same 257-token/256-target evaluation, not
the regular gate's 8×32 calibration budget. Module time is 536.953 s. It is
the same explicitly cross-arithmetic default-preset research contract, not
an RTL measurement. All five slow nodes complete, with four passes/one
failure and no skips, timeouts or resource stops.
Original logs and exact node records are under
`/private/tmp/taccel-python-light-20261011.qG9iCl/final_deferred_slow/`.

### Native verification

| Native gate | Result |
|---|---|
| Unit/subsystem/chained/RAW/snapshot campaign | 212 functional checks pass |
| FP32 primitive campaign | 12 gates pass; 113,522,077 checked inputs, zero mismatches |
| Fetch off/on | 7 checks per variant pass |
| Earlier DMA off/on | 34 checks per variant pass; final affected campaign below passes 35 per variant |
| SKY130 representative bank | 96,804 equivalence cycles; 2,518 logical collisions; zero physical collisions |
| Portable leaf-flow unit tests | 34 pass, including macro structure/area/process validation |
| Legacy STA replay contract | 5 mocked/static tests pass; mode overrides precede hierarchy; area units remain explicitly unverified; no real mapping/STA in these tests |
| External QKT replay/padded data | 17 explicitly skipped; replay data is absent |

The native campaign's clean already-built rerun takes 59.17 s and peaks at
540,917,760 bytes RSS, with no swaps. The first compile-and-test execution
takes 774.26 s; its test results pass, but the sandbox blocks the outer
`time` resource query, which is why the clean resource-enabled rerun is
retained separately. Generated C++ emits nonfatal compiler warnings;
Verilator reports no `%Warning` in this campaign. No arithmetic tolerance
or reference payload was changed.

After the final accepted-start counter correction, both mode-1 program runners
are genuinely rebuilt and pass freshness and actual-parameter checks. The
affected native campaign passes 128 checks: DMA 35 per fetch variant, control
36, systolic 8 and fetch 7 per variant. Held/repeated busy starts no longer
erase instruction-read counters. Unstalled STORE remains 45/42 cycles with
fetch off/on. This final affected campaign takes 109.98 s and reports
330,039,296 bytes maximum child RSS, not an aggregate process-tree peak.
The earlier 212-check/FP32 campaign is retained separately; its primitive
datapaths are unchanged. A 64-NOP-plus-HALT smoke test records 65/33 accepted
instruction reads and 390/262 cycles for fetch off/on, with no violations.

### Final elaboration gates

All four final mode-1 hierarchy checks exit zero. Their raw-process `check`
diagnostics report 27,891 problems, confined to generated function,
memory-write and bit-selection temporaries; none is presented as a passed
whole-design structural assertion. Both wrapper runs ultimately derive mode-1
SFU/helper instances, despite also parsing unused default-mode source branches.

| Raw elaboration target | Wall time (s) | Sampled process-group peak RSS (bytes) |
|---|---:|---:|
| Shared core, fetch off | 85.55 | 2,923,626,496 |
| ASIC wrapper, fetch off | 118.57 | 3,089,743,872 |
| FPGA wrapper, fetch off | 118.77 | 2,534,703,104 |
| Shared core, fetch on | 94.26 | 3,038,625,792 |

No process-lowering, whole-chip mapping, placement, routing or power analysis
runs in these gates. The 8 GiB RSS and 20 GiB free-disk guards are not reached.
Local evidence: `build/optimization-20261011/elaboration-final/report.json`
and its hash-recorded logs. The initial sandbox-denied RSS-monitor attempt is
retained separately and is not counted as a resource-measured successful run.

### Small-model RTL A/B matrix

The deterministic nonzero d64/one-layer fixture ran B1/B16/B32 with context
positions 0/15/127, contiguous teacher-forced decode, and chunked prefill.
There were 225 finite, fault-free executions and 200 exact logical FP16
logits/full-KV comparisons, with zero lost SRAM accesses. These runs preceded
the final stalled-WDATA correction; final integration is a separate gate.

| Fixed workload mix | Corrected baseline | Fetch only | QKV + fetch | All six + fetch |
|---|---:|---:|---:|---:|
| 22 supported cases | 3,064,429 | 3,007,629 | 2,851,853 | 2,859,903 |
| Single-chunk P128 + two decodes | 722,468 | 717,699 | 672,620 | 672,436 |

The combined profile reduces these totals by 6.67% and 6.93%, respectively;
they are fixed-mix sums, not universal speedups. The d64 model does not enter
large streamed FC2, so FC2 reuse requires the separate d768 gate.

Nine two-chunk P128 profiles failed compilation identically: context width 272
requires two 69,632-byte attention buffers, exceeding 128 KiB ABUF. These failures
are retained, not converted into passes. P128 single-chunk and two P16 chunks
exercise supported prompt/continuation cases instead.

At B32/p0, QKV-only costs 271,330 cycles and the combined profile 272,674 (+0.495%,
still 7.43% below the 294,544-cycle baseline). Dedup removes 4,320 data bytes,
shifting KV V-load addresses by one page plus 224 bytes. First bursts grow from
29 to 43 beats; 96 overlapped head prefetches lose 14 cycles each, accounting for the
1,344-cycle difference. Instructions differ only in relocated address fields;
DMA traffic totals and arithmetic phases are unchanged. Smaller layouts are
therefore an intentional size/latency tradeoff, not monotonically faster.

Local evidence: `build/optimization-20261011/matrix-summary.json`,
`b32-dedup-overlap.json`, and the two matrix directories. Peak Python-process
RSS was 243 MB; retained evidence was approximately 457 MiB.

### Representative streamed-FC2 RTL A/B matrix

A deterministic nonzero d768/12-head/one-layer fixture completes 198 RTL runs
and 176 exact logical FP16-logits/full-KV comparisons across B1/B16/B32,
positions 0/15/127, two P16 prefill chunks, and two contiguous decode steps.
All outputs are finite, with zero errors, lost accesses or forbidden overlap.
This matrix includes the final STORE backpressure fix, but precedes the
observability-only accepted-start counter correction and fail-early IR checks.

| Fixed workload mix | Corrected baseline | Combined candidate | Cycle reduction |
|---|---:|---:|---:|
| B1, one-token prefill | 6,292,175 | 5,469,658 | 13.072% |
| B1, P16 prefill | 4,941,842 | 4,263,261 | 13.731% |
| B16 | 12,886,023 | 11,398,023 | 11.547% |
| B32 | 25,096,185 | 23,905,391 | 4.745% |
| All 22 cases | 49,216,225 | 45,036,333 | 8.493% |

All candidate measurements include fetch retention. Fetch-only reduces the
fixed-mix total by 2.509%; indexed liveness and shared layout have those same
inference cycles, confirming that their primary benefit is host efficiency.
Real-row, FC2, QKV and dedup profiles reduce the baseline total by 2.706%,
2.999%, 6.980% and 3.118%, respectively; these include fetch and are not
independent gains to add or multiply.

B1/B16 reuse QKV quantization with 35 hits per stream and six admitted FC2
partitions. B32's 24 KiB QKV activation exceeds its 16 KiB cache cap, so QKV
reuse is deliberately not admitted (not a pressure fallback); FC2 retains five
of twelve K partitions, records five hits per stream, and has no fallback.
Its conservative ABUF working bound reaches exactly 128 KiB. Host Python
lifetime peak RSS is 753,631,232 bytes. Local evidence:
`build/optimization-20261011/matrix-d768-l1/{report,aggregate}.json`.

### Final short-context batched attention gate

The capacity fix completes 12 fresh RTL executions for d768/H12/L1, B16/B32,
one-token prefill and two contiguous teacher-forced decodes. All six
baseline/combined comparisons and all 12 frozen-golden comparisons pass,
including finite FP16 logits, full KV and unchanged weights. Both profiles
use the fix: this is an optimization comparison against a corrected baseline,
not a speed comparison against the previously uncompilable schedule.

| Fixed three-step sequence | Corrected baseline | Combined candidate | Cycle reduction |
|---|---:|---:|---:|
| B16 | 5,044,593 | 4,422,690 | 12.328% |
| B32 | 9,565,490 | 9,400,078 | 1.729% |

Wall time is 510.83 s; sampled total-job peak RSS is 872,235,008 bytes.
The capacity regression module contains 44 tests, including an old-policy
negative control, fragmentation, poisoned-memory numerical parity and
regeneration/fallback isolation. Full-width B16/B32 boundary compilation
covers key lengths 1/3/16/17/32/33/48/49 in both profiles. This covers the
packed multi-head path, not every possible batch/head geometry.

Lockstep B16/B32 prefill intentionally stores only the last stream query's
logits into output row zero; batched decode stores every stream. Sequential
P16 prefill instead stores all 16 query rows. Earlier batched matrix metadata
incorrectly counted reserved prefill output rows as logical results. The benchmark now separates
executed token rows, stored logits rows and logical source rows. The retained
short-context sidecar rechecks all actual stored rows and all KV without
rewriting the original reports or programs. The same distinction applies to
earlier prefill matrices; their reserved zero rows are not additional computed
logits. Evidence: `build/optimization-20261011/matrix-av-short-d768/report.json`,
`stored-logits-row-clarification.json` and
`build/optimization-20261011/av-short-supervision/report.json`.

A compile-only full GPT-2 gate also verifies byte-identical program images for
all 14 previously supported B1/B16 shape cases under the old and new AV
policies. It sizes the six newly supported B16 short-sequence executions at
7,484,414,464 bytes including metadata reserve. This takes 199.11 s, with
3,244,572,672 bytes sampled total-job peak RSS; no RTL executes in that gate.
The complete 26-run plan needs 24,193,553,936 bytes including runner snapshots,
so uncompressed retention would violate the 20 GiB free-disk floor on this
host. Execution is staged; original evidence is preserved, and any compression
must pass byte-for-byte recovery checks before replacing newly generated raw
files. Evidence: `build/optimization-20261011/full-shape-av-invariance.json`
and `full-shape-av-invariance-supervision/report.json`.

### Final-source full-model shape comparisons

The fresh GPT-2 124M campaign passes 14 executions and seven exact
baseline/combined comparisons: eight B1 runs and six B16 runs. All 50,257
live FP16 logits per stream are finite; full KV and immutable weights pass
their checks. Both runners select modes
1/1, with fetch off for baseline and on for combined, at one 16-byte read
beat per core cycle.

| Batch | Position | Corrected baseline cycles | Combined cycles | Cycle reduction |
|---:|---:|---:|---:|---:|
| 1 | 0 | 17,912,427 | 16,131,401 | 9.943% |
| 1 | 63 | 17,930,570 | 16,149,545 | 9.933% |
| 1 | 255 | 17,985,866 | 16,204,841 | 9.902% |
| 1 | 511 | 18,059,594 | 16,278,569 | 9.862% |
| 16 | 63 | 47,548,997 | 44,657,798 | 6.080% |
| 16 | 255 | 48,433,733 | 45,542,534 | 5.969% |
| 16 | 510 | 49,608,789 | 46,717,574 | 5.828% |

The fixed seven-shape total falls from 217,479,976 to 201,682,262 cycles
(7.264%). B1 and B16 subtotal reductions are 9.910% and 5.957%; these are
fixed-mix sums, not universal speedups. Sampled total-job peak RSS is
3,832,004,608 bytes. B1 and B16 monotonic campaign times are 1,860.278 s and
4,562.690 s; they exclude an overnight host suspension and are not
uninterrupted wall-clock latency.

Shape cases use token ID `(position + 7*stream) % vocab_size` and zero-filled
historical KV, not a genuine continuation or a language-quality reference.
Program SHA-256 matches the
14-case compile-invariance proof; compiler files, copied runners and inputs
are pinned at execution time. Local evidence:
`build/optimization-20261011/full-gpt2-b1-shapes-final/report.json` and
`full-gpt2-b16-shapes-final/report.json`, with
`full-gpt2-shapes-final-supervision/report.json`.

A hash-pinned metadata sidecar rechecks the actual logical logits regions
and documents that no tokenizer is used. At B1/p511, the canonical decoder
estimate is 132,969,216 useful GEMM MACs: 5.83% of `256 × enabled_compute_steps`
or 3.19% of `256 × total_cycles` for the combined candidate. This excludes padding, SFU/helper work and
PTQ; it is not measured physical utilization or operations/second. Evidence:
`build/optimization-20261011/full-gpt2-shape-metrics.sidecar.json`.

All 28 newly generated shape dumps/program images are now held in verified
lossless chunk archives: 12,516,483,968 bytes become 1,753,792,146 bytes of
unique encoded payload, excluding filesystem overhead. Every original and
reconstructed stream matches its full SHA-256 and byte count before a durable
recovery manifest permits replacement. The original reports, layouts,
commands and runners are unchanged; older evidence and model fixtures are
untouched. Replaying the recorded shape commands requires restoring the
archived `program.bin` and `dram.raw` paths with those recovery checks.
Evidence: each final shape directory's `lossless_chunk_archives.json` and
`build/optimization-20261011/full-gpt2-lossless-storage-review.json`.

This funded the 12-run continuation campaign: launch free disk was
38,443,114,496 bytes, with an 11,434,732,080-byte budget including both runner
snapshots. Projected remaining space is 27,008,382,416 bytes, above the
20 GiB floor. Current free space is still checked throughout execution;
the launch projection was not a guarantee against unrelated disk activity.

At B1/p511, read traffic falls from 162,388,032 to 139,587,648 bytes (14.04%),
retired instructions from 92,886 to 73,266 (21.12%), and instruction reads
from 92,886 to 54,241 (41.60%). Systolic work is unchanged: 11,073,350 busy
cycles, or 68.02% of the combined total. Drain takes 1,289,990 cycles (7.92%);
load, K-boundary, flush and drain together take 16.97% of total cycles.
Exposed DMA synchronization wait is 2,498,662 cycles (15.35%). These counters
overlap and are not a disjoint partition of execution time.

The measurements favor a separate exact-row/B1 GEMV or drain/A-reuse
experiment next. Wider-memory work needs the read-pacing gate, resource
sharing needs per-op area evidence, and retiming needs actual critical-path
evidence. None of those architecture changes is silently included here.

### Final-source full-model continuation

The full GPT-2 124M campaign also passes 12 executions and six exact
baseline/combined comparisons: B1 and B16, one-token prefill at position zero,
then teacher-forced decodes at positions one and two. All stored live FP16
logits are finite, complete KV states match, immutable weights are unchanged,
and no faults, lost accesses or forbidden collisions occur. Hardware modes
are 1/1; baseline fetch is off and combined fetch is on.

| Batch | Step | Corrected baseline cycles | Combined cycles | Cycle reduction |
|---:|---|---:|---:|---:|
| 1 | Prefill | 15,319,674 | 13,460,685 | 12.135% |
| 1 | Decode, position 1 | 15,225,067 | 13,366,940 | 12.204% |
| 1 | Decode, position 2 | 15,225,355 | 13,367,229 | 12.204% |
| 16 | Prefill | 23,145,048 | 20,744,895 | 10.370% |
| 16 | Decode, position 1 | 23,435,141 | 21,034,694 | 10.243% |
| 16 | Decode, position 2 | 23,439,749 | 21,039,302 | 10.241% |

B1 sequence totals fall from 45,770,096 to 40,194,854 cycles (12.181%);
B16 totals fall from 70,019,938 to 62,818,891 (10.284%). The fixed six-step
mix falls from 115,790,034 to 103,013,745 cycles (11.034%). These are
workload-specific cycle reductions, not measured hardware tokens/second.
Active campaign time is 3,167.353 s, with 3,517,218,816 bytes sampled total-job
peak RSS under the 8 GiB/20 GiB resource guards.

An independent streaming binary check verifies four initially zero prefill
KV regions, eight nonzero decode input regions byte-identical to the immediate
predecessor's raw output, and 12 changed, nonzero output KV states. It pins the
actual retained program/dump bytes and original reports before archival.
This establishes whole-KV continuation, not per-row write placement, golden
accuracy, generated-text quality or a perplexity threshold. B16 prefill
executes 16 token rows but stores one last-query logits row; each B16 decode
stores all 16 rows. The finite checks cover those actual stored outputs,
not reserved padding rows.

Evidence: `build/optimization-20261011/full-gpt2-short-sequences-final/report.json`,
its `kv-carry-chain.json`, and
`build/optimization-20261011/full-sequences-final-supervision/report.json`.

All 24 newly generated continuation programs/dumps are subsequently stored
in verified recovery archives: 11,229,677,056 raw bytes become 2,292,631,546
unique encoded payload bytes. Together with the shape campaign, 52 recoverable
artifacts save 19,699,737,332 bytes, excluding filesystem overhead. Original
reports, the KV proof, layouts, commands and runners remain unchanged; older
evidence and model fixtures are untouched. Restore archived program/raw paths
and verify their hashes before replay. Evidence: the sequence directory's
`lossless_chunk_archives.json` and
`build/optimization-20261011/full-gpt2-final-storage-closure.json`.

### Read pacing and deterministic repeats

The representative d768/H12/L1, B1/B16 position-127 replay passes 12 runs and
six exact finite-logits/full-KV comparisons at read intervals 1/2/4. Actual
program, runner, input/output and report identities are pinned. All four
interval-one cycle totals exactly reproduce their retained original values;
this repeats those four cases, not the entire 14-shape full-model campaign.

| Batch | Read cycles per beat | Baseline cycles | Combined cycles | Cycle reduction |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1,058,519 | 921,451 | 12.949% |
| 1 | 2 | 2,171,433 | 1,791,883 | 17.479% |
| 1 | 4 | 3,401,917 | 2,779,119 | 18.307% |
| 16 | 1 | 2,187,186 | 1,939,196 | 11.338% |
| 16 | 2 | 3,547,232 | 3,172,112 | 10.575% |
| 16 | 4 | 5,048,784 | 4,607,492 | 8.741% |

Both profiles become slower as read pacing increases; the optimization benefit
survives every tested interval. Writes remain independent and unthrottled.
These results do not establish fixed shared GB/s, full-model bandwidth scaling
or realized external-memory/clock performance. Campaign time is 344.952 s,
with 442,843,136 bytes sampled total-job peak RSS. Evidence:
`build/optimization-20261011/pacing-d768-final/report.json` and
`pacing-final-supervision/report.json`.

### Earlier corrected full-model B1 baseline

GPT-2 124M, frozen preset name `weight_only_int8_quarot`, hardware modes 1/1, fetch off,
one 16-byte read beat/cycle, zero historical KV, token ID 0:

| Position | Cycles | Actual FP16 logits |
|---:|---:|---|
| 0 | 17,998,971 | 50,257/50,257 finite |
| 63 | 18,017,114 | 50,257/50,257 finite |
| 255 | 18,072,410 | 50,257/50,257 finite |
| 511 | 18,146,138 | 50,257/50,257 finite |

All cases retired 92,886 instructions, with 5,818,015 DMA/systolic co-busy cycles,
zero lost accesses and no forbidden overlap. At p511, systolic busy time of
11,109,206 cycles partitions into load 379,810, data 8,329,728,
K-boundary 500,618, flush 599,700 and drain 1,299,350.
The enabled compute-step count is 8,929,428;
it includes padding and is not a useful-MAC utilization estimate.

The baseline was captured before the final stalled-WDATA correction; its
source/runner identities are retained, and it is not relabeled as a final
integration run. These finite shape outputs do not establish trained-model
quality with genuine prior KV. The direct bundle benchmark uses the preset's
weight-only policy but does not apply the separate QuaRot model transform;
the preset name is not evidence that rotation was executed. Local report:
`build/optimization-20261011/baseline-b1/report.json`.

### Matched process leaf measurements

Both PDKs use identical source hashes, a 30 ns clock, a 5,000 ps ABC objective,
30% IO delay, ideal input slew and 0.05 pF output load. Area is verified against
all mapped cells' LEF sizes. Corners: SKY130HD TT 25°C/1.8V and ICSprout55
LLSC-H7CR typical TT 25°C/1.2V. Native Yosys 0.65/sv2v 0.0.13; OpenSTA runs in
the existing pinned ORFS image `sha256:493e179a…8903e1`.

The DMA leaf uses its default 16 MiB DRAM-address range, matched between PDKs
and consistent with the ASIC wrapper's default. Full-model performance runners
instead configure a 1 GiB range. These are isolated leaf comparisons, not
mapping results for that complete performance-runner configuration.

| Leaf | SKY130 cell area (µm²) | ICSprout55 cell area (µm²) | SKY setup bound (ns) | ICS setup bound (ns) |
|---|---:|---:|---:|---:|
| PE | 6,474.96 | 1,972.60 | 7.281 | 3.360 |
| DMA | 373,674.63 | 92,870.12 | 15.823 | 7.231 |
| FP32 div p6 | 49,523.75 | 13,817.44 | 30.768 | 13.808 |
| Fetch off | 3,079.20 | 909.44 | 2.512 | 1.692 |
| Fetch on | 11,116.91 | 3,247.16 | 8.426 | 7.316 |

Setup bounds are single-clock register-to-register, **zero-wire** estimates,
not post-layout clocks. The SKY divider violates the nominal 30 ns objective.
Fetch retention costs 8,037.71 µm² in SKY and 2,337.72 µm² in ICS in this isolated
mapping, and lengthens its local setup cone; this does not establish the
chip's critical path. Do not multiply leaf frequency or cell density into
chip throughput or area.

| Host resource | SKY130 | ICSprout55 |
|---|---:|---:|
| DMA mapping time | 84.04 s | 90.20 s |
| DMA mapping peak RSS | 1,446,592,512 B | 1,715,060,736 B |
| PE mapping peak RSS | 47,529,984 B | 179,830,784 B |
| Largest measured STA container RSS (DMA) | 194,715,648 B | 288,026,624 B |

sv2v peak RSS was approximately 295 MB for either PDK. The ten leaf run
directories occupy approximately 52 MiB combined, excluding PDK archives,
libraries and the 4.62 GB existing Docker image. ICS produces smaller mapped
logic here but uses more mapping/STA RAM; it is not automatically easier for
the 16 GiB laptop. Both leaf campaigns fit. Full-SFU/chip resource demand is
unmeasured in this campaign. No activity, extracted parasitics, SRAM/IO
integration, DRC/LVS or power model was evaluated, so watts and energy/token
remain unavailable. SKY130 remains the reference backend.

Raw reports/logs/netlists are under
`rtl/asic/results/optimization-20261011/{sky130hd,icsprout55}/`.

All ten retained mapped netlists also pass a separate `hierarchy -check` /
`check -assert` audit after importing the actual standard-cell Liberty port
directions. Earlier post-map logs treated those cell outputs as unknown
drivers because the blackbox interfaces had not been loaded into Yosys.
Original reports/netlists are unchanged; the post-check records their hashes
under `postmap-structural-check/`. The maintained flow now imports those
interfaces before asserting structural checks.

### Host compilation

Five repetitions with alternating baseline/candidate order on synthetic
linear graphs produced identical instruction and data SHA-256 in every case:

| Nodes | Baseline median | Indexed median | Speedup |
|---:|---:|---:|---:|
| 500 | 54.16 ms | 44.06 ms | 1.229× |
| 1,000 | 125.89 ms | 90.12 ms | 1.397× |
| 2,000 | 325.83 ms | 180.35 ms | 1.807× |

Host peak RSS was 71,204,864 bytes. It is the current Python process's
lifetime peak, not isolated compilation allocation or an RTL child-process
measurement. Raw local evidence:
`build/optimization-liveness-20261011/compile.json`.

### Static schedule diagnostics

These isolated emitter examples are not end-to-end chip speedups:

| Example | Corrected baseline | Candidate |
|---|---:|---:|
| QKV instruction count | 133 | 88 |
| QKV LOAD bytes | 9,216 | 4,416 |
| QKV MAX_ABS/QUANT pairs | 6 | 1 |
| FC2 M1/K3072/N768 instructions | 1,921 | 841 real-row; 763 with tile reuse |
| Same FC2 LOAD bytes | 2,777,088 | 2,592,768 real-row; 2,543,616 with reuse |
| Same FC2 quantization pairs | 24 | 18 with reuse |
| Example FC2 image bytes | 4,856,064 | 2,492,160 deduplicated |

FC2 reuse raises the example ABUF high-water mark from 32,768 to 65,536 bytes;
that capacity tradeoff is intentional and admission is bounded.

## Reproduction

From the repository root, with the installed Python dependencies and native
Verilator/sv2v/Yosys:

```sh
make -C rtl/verilator -j1 SYSTOLIC_ARCH_MODE=1 DRAM_SIZE_124M=1073741824 \
  run_program_synth_nofetch run_program_synth
PYTHONPATH=software .venv/bin/python -m pytest -q software/tests \
  --verification-profile required
make -C rtl/verilator test_dma test_dma_buffer test_fetch_no_buffer test_fetch_buffer
make -C rtl/asic test-sram-bank
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s rtl/asic/tests -p 'test_*.py'
PYTHONPATH=software .venv/bin/python software/tools/benchmark_indexed_liveness.py --help
PYTHONPATH=software .venv/bin/python software/tools/optimization_matrix.py --help
```

The matrix supports isolated candidates, B1/B16/B32, genuine nonzero synthetic
weights, contiguous teacher-forced decode and chunked prefill, exact logical
logits/KV comparisons, and separate shape-only context tests. A missing or
non-finite logical output fails the comparison gate. Full DRAM hashes are
recorded but not compared across relocated layouts; logical regions and
immutable weight bytes are checked instead.

Shape-only full GPT-2 cycle reports use zero-filled historical KV rows;
continuation reports instead carry actual preceding RTL KV. Neither establishes
language-model quality. `--beat-interval 1/2/4` paces the AXI read channel;
writes remain independent and unthrottled. This is not a fixed shared GB/s
memory controller. Tokens/second are unavailable without an explicitly named
clock; the old 34.41 MHz partial-design result is not reused as achieved fmax.

### Explicit optimized bundle API

`CompilerOptions()` and omitted options retain the corrected baseline.
The measured combined candidate is available through the public builder:

```python
from taccel.compiler.options import CompilerOptions
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

options = CompilerOptions(
    indexed_liveness=True, real_row_fc2_loads=True,
    reuse_qkv_quantization=True, reuse_fc2_quantization=True,
    shared_data_layout=True, deduplicate_data=True,
)
tiny = build_stage3_tiny_decoder_bundle(
    payload, smoke_decode_steps=2, batch=1, prefill_tokens=1,
    compiler_options=options, target="rtl-gen2",
)
```

Here `payload` is a loaded nanoGPT-format checkpoint. The builder does not
perform QuaRot preprocessing. `HostRunner` runs the Python golden model,
not RTL hardware; use the matrix's pinned mode-1 runners for hardware-cycle
comparisons. For single-stream generation, select FP16 logits and pass the
explicit live vocabulary. `final_token_commit=False` returns the last token
without executing its decode/KV write; `True` commits it, including EOS.
The benchmark-only combined profile is not an implicit CLI/default promotion.

Generic and wrapper smoke gates make unresolved hierarchy errors fatal, but
their pre-process-lowering `check` output is diagnostic. Unused inlined-function
temporaries can appear undriven at that stage. These gates are not asserted
whole-design structural proofs; the mapped leaf/bank flows instead lower
processes, load actual cell interfaces and run `check -assert`.

## Remaining acceptance gates

- Final shape, continuation/KV-carry and representative read-pacing/repeat
  gates are complete. Their workload and repetition scopes remain distinct.
- Final affected native, required Python and raw hierarchy-resolution gates
  are complete. All optional nodes are executed and retained; the stale CLI
  smoke repair passes, while both PPL failures remain unresolved. Clean-source
  and fresh-build reproduction also passes. This closes M5 integration for
  the accepted opt-in M2 changes, not the entire physical roadmap.
- Resolve the two retained legacy PPL failures and characterize the 124M
  RTL/golden numerical gap before claiming trained-model quality. Changing
  the reference policy or numerical contract is a separate proposal.
- Default promotion remains a separate decision. All six compiler defaults
  stay disabled, and buffered performance-runner results are not automatically
  the wrapper's default configuration.
- Validate the experimental SRAM adapter's physical timing contract before macro integration,
  full-chip mapping/PNR, IO, extraction, DRC/LVS and activity-based power.
- Select any M4 architecture experiment from measured bottlenecks. No new ISA,
  arithmetic policy, memory interface, or technology migration is implied by
  the present behavior-preserving changes.

No full-chip area, achieved frequency, watts, operations/second or energy/token
is established by this implementation batch.
