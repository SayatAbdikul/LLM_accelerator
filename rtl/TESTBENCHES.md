# RTL testbench guide

**Updated:** 2026-10-13 against the current working tree.

Native Verilator C++ benches are the supported RTL verification framework.
The former Cocotb tier was dormant, superseded, and removed in `3486c01`.
Icarus is best-effort only.

## Build modes

| Mode | Target/example | Meaning |
|---|---|---|
| Reference | `test_sfu`, `test_helpers`, `run_program` | DPI-backed behavioral SFU/helper comparison path |
| Synthesizable units | `test_sfu_synth`, `test_helpers_synth` | Mode 1 for the respective unit |
| Synthesizable program | `run_program_synth` | SFU=1, helper=1, instruction buffer enabled |
| Corrected program baseline | `run_program_synth_nofetch` | SFU=1, helper=1, instruction buffer disabled |
| No overlap | `run_program_noovl` | `SYS_DMA_OVERLAP=0` serialization reference |

The three hardware program variants use 1 GiB modeled DRAM by default
(`DRAM_SIZE_124M`) so GPT-2 124M bundles fit. `run_program_noovl` also enables
the instruction buffer. Inspect the runner JSON `configuration` fields
`sfu_synth_mode`, `helper_synth_mode`, and `fetch_buffer_enabled`; a binary's
directory name is not evidence of its datapath configuration. Historical
SFU=1/helper=0 measurements are not the corrected hardware baseline.

The Makefile tracks both shared C++ headers and included `.svh` files as
prerequisites. If a source/header changes, do not bypass Make's rebuild check
by executing an old binary directly. Python conformance and benchmark helpers
also reject runners older than relevant RTL/C++/header/filelist/Makefile inputs.

## Main aggregate

```sh
make -C rtl/verilator all
```

This runs exactly:

- `test_decode`;
- `test_control`;
- `test_dma`;
- `test_helpers`;
- `test_sfu`;
- `test_systolic`.

It does not include the mode-1, primitive, QKT, or program-level gates below.

## Focused benches

### Frontend and control

```sh
make -C rtl/verilator test_decode
make -C rtl/verilator test_control
make -C rtl/verilator test_addr_raw_hazard
make -C rtl/verilator test_fetch_no_buffer test_fetch_buffer
```

`test_decode` includes illegal checks for the six retired generation-1 SFU
opcodes and RTL-reserved `SOFTMAX_FP32` `0x1C`. `test_control` covers issue,
SYNC, configuration, dispatch, and architectural faults.

The focused fetch bench exercises both instruction-buffer configurations,
including the protocol, coherence and fault contracts. Full-core STORE-to-
instruction coherence is also covered by `test_dma_buffer`. On 2026-10-11,
both focused fetch configurations passed all seven cases, including stalled
request stability, invalidation during an outstanding refill, same-cycle
invalidation/response, response errors and sticky fault reset.

### DMA and helpers

```sh
make -C rtl/verilator test_dma
make -C rtl/verilator test_dma_buffer
make -C rtl/verilator test_helpers
make -C rtl/verilator test_helpers_synth
make -C rtl/verilator test_sfu_helper_synth
```

The DMA bench covers regular transfer, errors, transposed LOAD, 4 KiB page
boundaries under backpressure, effective-address overflow, split-transfer
response errors and instruction coherence. `test_dma_buffer` reruns it with
the instruction buffer enabled. The helper mode-1 bench exercises the
synthesizable helper datapath rather than only the DPI reference;
`test_sfu_helper_synth` runs SFU cases with both mode bits set to 1.

On 2026-10-11, `test_dma` and `test_dma_buffer` each passed **35/35** checks.
The 16 page-boundary shapes use full-address-mixed data and exact accepted
AR/AW address-and-length sequences, distinguishing transpose stripes and
successive pages rather than relying on repeating byte patterns.

The fetch-counter regression holds and reasserts `start` while a program is
busy, compares every accepted instruction AR with the observed counter, and
checks ignored post-HALT starts plus a reset-qualified new run. It fails before
the accepted-start guard and passes in both fetch configurations afterward.

The stalled-STORE regression issues a legal systolic Port-S read from a
different row of the same ABUF while DMA `WVALID` is backpressured. It failed
before the payload-hold fix in both fetch configurations because SRAM output
changes leaked onto stalled `WDATA`; after the fix, it checks stable AXI data,
the stored bytes and all concurrent MATMUL results. Its unstalled control
retains the pre-fix 45-cycle total without buffering and 42-cycle total with
buffering, with the first write beat two cycles after AW and subsequent beats
every two cycles, including a page split. These are directed bench timings,
not full-model throughput measurements.

### SFU

```sh
make -C rtl/verilator test_sfu
make -C rtl/verilator test_sfu_synth
```

`test_sfu_synth` is the generation-2 chip-path gate. It runs ten frozen
fixture vectors, a scale-write/consumer-chain test, and a 32-element signed
GELU-tail regression, covering dequant, scaled dequant, VADD, LayerNorm, GELU,
masked softmax, quantization, and MAX_ABS scale generation. On 2026-10-13,
both `test_sfu` and `test_sfu_synth` passed **12/12** checks. The added tail
case includes the captured input 10.8671875 and independently expects the
positive saturated result to equal the input and the negative result to be
negative zero.

The separate LayerNorm quality gate runs both SFU modes:

```sh
make -C rtl/verilator test_ln_quality
# Optionally include a retained real GPT-2 row:
make -C rtl/verilator test_ln_quality \
  LN_CAPTURED_PREFIX=/absolute/path/to/first-ln-001
```

`rtl/verilator/gen_ln_vectors.py` generates 512 deterministic cases: eight
patterns at every legal width from 16 through 1024 in steps of 16. The patterns
cover normal, near-constant, cancellation, wide-range, extrema, positive-zero,
negative-zero and subnormal inputs. Expected values use the frozen simulator's
NumPy FP32 mean/variance expression, not a copy of the RTL reduction traversal.
The gate compares raw FP16 output bytes, including signed zero. Generated
vectors live in ignored `rtl/verilator/build/ln_quality.bin`; the existing
73-entry frozen reference manifest and its fixtures are not regenerated.

`LN_CAPTURED_PREFIX` adds a 768-element real `block0_ln1` case only after
checking its four array hashes and reproducing its expected output with the
unchanged NumPy expression. With that case included, both SFU modes passed
**513/513** vectors on 2026-10-13. This validates the repaired pairwise mean
and variance reduction order at the existing zero-ULP LayerNorm contract;
it does not establish full-model numerical quality. `test_ln_quality` is not
part of `make all`.

### Systolic

```sh
make -C rtl/verilator test_systolic
make -C rtl/verilator test_systolic_array_chained
make -C rtl/verilator test_systolic_chained
make -C rtl/verilator test_accum_snapshot_readback
make -C rtl/verilator test_systolic_query
make -C rtl/verilator test_systolic_qkt
```

`test_systolic_qkt` aggregates:

- `test_systolic_qkt_basic`;
- `test_systolic_qkt_replay`;
- `test_systolic_qkt_padded`.

Replay and padded cases require `RTL_QKT_REPLAY_DIR`. They skip when the
external dataset is absent. The obsolete history bench was removed because it
emitted a retired opcode and every case skipped.

The main systolic bench also checks phase accounting: load, data, K-boundary,
flush, drain and other cycles must partition systolic busy time, while compute
steps equal data plus flush cycles.

### FP32 primitives

```sh
make -C rtl/verilator \
  test_fp32_add \
  test_fp32_div test_fp32_div_p3 test_fp32_div_p4 \
  test_fp32_div_p5 test_fp32_div_p6 \
  test_fp32_sqrt test_fp32_sqrt_p3 test_fp32_sqrt_p4 test_fp32_sqrt_p6 \
  test_fp32_exp_p18 test_fp32_gelu_p33
```

Pipelined primitive gates compare against their combinational parent or DPI
reference according to the individual bench contract. `fp32_exp_p18` and
`fp32_gelu_p33` are not experimental standalones: both are integrated into the
current SFU datapath.

The GELU gate also checks mathematical properties independently of the
combinational implementation: positive saturated-tail outputs equal their
inputs, and tested finite inputs produce finite outputs. Directed coverage
includes every positive finite FP16 value at least 10 and its negative partner
(26,112 inputs), the captured 10.8671875 failure, and valid bubbles. These
checks catch a defect shared by the combinational and pipelined implementations,
such as the former `Inf/Inf` in the exponential ratio. The repair handles the
positive-infinite exponential's limiting ratio before division; it does not
mask arbitrary NaNs or relax the GELU tolerance.

On 2026-10-13, `test_fp32_gelu_p33` passed **8,526,144** bit-exact comparisons
with zero failures, including **2,285,857** independent positive-tail checks
and **8,514,615** finite-input checks. The strengthened gate also passed
**18,522,288** exact 33-stage valid checks and **11,527** independent NaN-input
checks. Its supervised run took 47.321 s and peaked at 104.453 MiB sampled
test-process-group RSS. These overlapping checks are not separate test cases.
The strengthened gate's log and resource record are under
`build/optimization-numerical-20261013/supervision-gelu-timing-final-v1/`.
The earlier native SFU, LayerNorm and GELU results remain retained in
`build/optimization-numerical-20261013/supervision-native-final-v1/run.log`.

All 12 standalone FP32 suites listed above subsequently passed in a serial
run on 2026-10-13, taking 85.363 s supervised wall time and peaking at
19.438 MiB sampled process-group RSS. The log and resource record are under
`build/optimization-numerical-20261013/supervision-all-primitives-final-v1/`.

## Program-level conformance

Build the mandatory hardware runner and, for matched fetch comparisons, the
corrected no-buffer baseline:

```sh
make -C rtl/verilator run_program_synth run_program_synth_nofetch
```

Prepare the deterministic tiny fixture and run the mandatory software/RTL core:

```sh
PYTHONPATH=software .venv/bin/python software/tools/train_tiny_fixture.py
PYTHONPATH=software .venv/bin/python -m pytest -q software/tests --verification-profile required
```

The required profile selects core modules, excludes optional 124M cases and
turns a skipped mandatory test into a failure. On 2026-10-13, it passed
**537 tests** with zero failures or skips; 244 optional cases were deselected
from a 781-test collection. The supervised run took 93.831 s and peaked at
689.094 MiB sampled test-process-group RSS. Its log and resource record are
under `build/optimization-numerical-20261013/supervision-required-final-v1/`,
with exact test outcomes in
`build/optimization-numerical-20261013/required-final.xml`. This required-profile
pass does not rerun or resolve the historical optional PPL failures below.

The separate affected run passed 32 tests (eight prefill, six LayerNorm-vector
generator and 18 diagnostic-helper tests) plus five subtests in 13.381 s
supervised wall time, peaking at 300.969 MiB sampled process-group RSS. Its
log/resource record and JUnit outcomes are retained in
`build/optimization-numerical-20261013/supervision-affected-final-v1/` and
`build/optimization-numerical-20261013/affected-final.xml`. These results are
reported separately, not added to the 781-test software collection.

### Historical October 11 and 12 validation

The following suite and clean-source counts predate the October 13 numerical
repairs. They describe retained historical evidence, not a fresh full-suite
run of the current RTL and software.

On 2026-10-11 the required profile passed 395
tests with zero skips or failures; 244 optional cases were deselected from a
639-test collection. All eight affected multi-token prefill tests and both
offline Hugging Face converter-parity tests also passed after the AV-residency,
stored-row mapping and host lockstep-priming fixes.

On 2026-10-12, all 180 optional tests with only earlier-source evidence passed
a rerun against the same 177 Python source-file hashes, with no failures or
skips. The rerun took 230.981 s wall time and peaked at 830.56 MiB sampled
test-process-group RSS. That rerun established 585 distinct current-source
passes: 395 required and 190 optional. Its exact IDs, logs and source/resource
records are retained locally under
`build/optimization-20261011/python-final/earlier-current-180/`.

The separately bounded 124M RTL finite-output characterization passed on
2026-10-12 in 702.241 s
wall time, with 2850.5 MiB peak sampled test-process-group RSS and source
hashes verified at completion. The regular optional phase passed 46/48 tests
and the slow phase passed 4/5, with no skips or incomplete outcomes. The original
639-test epoch therefore ended at 636 passes and three failures, with all 177
Python source-file hashes unchanged. Its ledger, logs and identity manifest
remain unchanged under `build/optimization-20261011/python-final/final-optional/`.

Only `software/tests/test_run_gpt2.py` subsequently changed: its CLI fixtures now
select compatible explicit `--ptq-preset` arguments, with assertions and
numerical tolerances unchanged. Both module tests and its verification-contract
importer passed a fresh three-test run in 45.49 s pytest time (47.195 s supervised
wall time, 2753.844 MiB peak sampled test-process-group RSS). The reconciled
result is 637/639 passes: 395 required and 242 optional, consisting of 634
dependency-validated carried passes and three fresh passes. All 639 tests were
not rerun under one source hash. The new ledger and retention manifest are under
`build/optimization-20261011/python-final/test-contract-repair/`.

Both research PPL gates remain failed: relative golden/fake-quant PPL differences
were 95.6993% for the regular gate and 62.5625% for production calibration,
against the unchanged 2% limit. This is not a whole-suite or language-quality
pass, and the original failures remain retained. The 124M log is at
`build/optimization-20261011/python-final/final-optional/execution/124m/00_test_compare_rtl_golden/pytest.log`.

The final isolated clean-source reproduction passed 395 required tests, eight
prefill tests and 39 portable ASIC static/mocked tests. Newly compiled runners
used SFU/helper modes 1/1 with fetch disabled/enabled; all eight tiny-model
RTL/golden comparisons and four exact A/B comparisons passed for logical FP16
logits and full KV state. All 428 exported source hashes stayed unchanged during
the run. Existing installed dependencies were reused, not freshly installed.
These repeat checks do not add distinct tests to the 637/639 result or resolve
the PPL failures. Retained evidence is under
`build/optimization-20261011/clean-source-final/`.

### Current numerical quality scope

On 2026-10-13, the separate software quality gate passed a canonical
257-token window with 256 prediction targets using an actually applied QuaRot
transform. The compiled golden simulator and independent W8A16 reference
produced the same perplexity; its increase over FP32 was 2.494193%, within
the declared 5% bound. The report is retained at
`build/optimization-numerical-20261013/quality-quarot-257-v2/report.json`.
That software result does not itself establish hardware quality, supersede the
historical cross-arithmetic PPL failures or promote a default preset.

The repaired hardware replay passed on 2026-10-13 within the accepted bounded
scope: the first three teacher-forced targets of that pinned 257-token window,
using the full compiled attention window of 256. All three live-logit rows and
whole-KV outputs were bit-exact against golden; the two continuation steps used
actual preceding hardware KV dumps. RTL/reference PPL was 1317.182714 versus
original FP32 1371.070346 (-3.930333%), meeting the selected 5% FP32 increase
limit and the unchanged 2% reference PPL, 0.995 cosine and 4/5 top-five limits.
Measured cosine was 1 and every top-five overlap was 5/5. Source and output
hash checks closed successfully. The result is retained at
`build/optimization-numerical-20261013/quality-rtl-replay-v2/report.json`.
Reproduce it with the tracked
[RTL quality CLI](../software/tools/w8a16_rtl_quality_gate.py) and the
[documented prerequisites and command](../software/CODEBASE.md#tools).
The [numerical report](../docs/numerical_quality_results.md) preserves the
earlier failed replay and the exact input scope. This is not a 256-target RTL
or corpus-wide quality result and does not promote a default preset.

### Conformance contracts

The developer profile can run a smaller selection directly, but its optional
skips do not count as passes.

- `software/tools/rtl_cosim.py` serializes a bundle stream into a
  `ProgramBinary` and compares RTL memory/results with the golden simulator.
- It defaults to `--runner synth`; `--runner reference` explicitly selects
  `run_program`, which must be built separately with `make -C rtl/verilator
  run_program`. `--rtl PATH` selects a specific runner executable.
- `test_compare_rtl_golden.py` checks the simulator pin plus the complete
  reference manifest, deterministic bundles, tiny prefill, multistep KV
  continuation and finite-logits checks. GPT-2 124M is opt-in.
- `test_batched_decode.py` covers tiny decode bundles including packed B=16
  attention and the B=32 multi-M-tile walk.

The checked-in `software/tests/fixtures/gen2_reference_manifest.json` pins
73 golden/ISA/assembler/utility source, generator and raw fixture/metadata
files. Verify it without refreshing any expected values:

```sh
PYTHONPATH=software .venv/bin/python software/tools/check_reference_manifest.py
```

Per-op co-simulation uses a 3-ULP allowance for GELU and 0 ULP for other FP16
operations, retaining the native signed-zero and special-value contract.
Differing NaN/Inf encodings do not pass as numerically close. Integer captures
and the batched tiny logits contract are byte-exact. The tiny logits gate
requires finite outputs, identical argmax, near-unit cosine and zero NLL delta.
The opt-in `PYTEST_124M=1` test records the first tensor divergence and logits
metrics and rejects nonfinite outputs even if hardware status is clean. Its
October 12 pass characterized two finite steps without an accuracy threshold:
argmax differed on both steps, minimum cosine was 0.999596715, and the first
tensor divergence was a 1-FP16-ULP difference at `block0_ln1` against its frozen
0-ULP band. This is neither an exact full-model conformance pass nor a language-
quality result. Tiny-model conformance is not proof of full-model accuracy.

Compiler-side `target="rtl-gen2"` validates hardware-supported instructions
before execution. Research INT4/noncausal-softmax paths must use
`research-golden` and are not hardware conformance workloads. See
[software targets and compiler candidates](../software/CODEBASE.md#targets-and-compiler-candidates).

## Generic elaboration gate

```sh
make -C rtl/verilator synth-check
```

This is the only generic shared-core hierarchy/check/stat target. The older
`synth-check-ctrl` partial gate and its black-box stubs were removed after the
full gate became authoritative.

## Performance measurement

Build:

```sh
make -C rtl/verilator run_program_synth
```

Measure direct cycle counts:

```sh
PYTHONPATH=software .venv/bin/python software/tools/bench_decode_cycles.py \
  --positions 0,63,255,511 --batch 1 --out-dir /tmp/taccel-decode-report
PYTHONPATH=software .venv/bin/python software/tools/fast_gate_b16.py \
  --batch 16 --position 510 --out /tmp/taccel-b16-report.json
```

These commands require the converted GPT-2 fixture and measure standalone
decode shapes with zero-filled historical KV, not representative model
quality. Supply `--fixture PATH` for another compatible checkpoint.
`--compiler-optimizations` accepts comma-separated `CompilerOptions` names;
the empty default is the corrected compiler baseline. For an actual matched
gate, add `--baseline-report PATH` to `fast_gate_b16.py`. Without that input it
reports a measurement, not a passed performance comparison. The gate compares
checkpoint/calibration hashes, shape, memory pacing, compute modes, actual
logits and cycle regression; finite outputs are required on both rails.

`optimization_matrix.py` retains program images, command lines, raw logs,
runner JSON and full data-region dumps for baseline/candidate experiments. It
separates zero-KV shape cases from contiguous teacher-forced prefill/decode
sequences, checks immutable weights, and compares logical FP16 logits plus
the complete KV image. Its default candidate enables all compiler options;
that is an experiment configuration, not a change to compiler defaults.

Performance reports must include:

- source commit/dirty identity and checkpoint, calibration, program and runner hashes;
- batch and position/context;
- direct step cycles and cycles per token;
- actual SFU/helper/fetch configuration;
- read-beat pacing and its write-channel limitation;
- explicitly assumed clock, if converting to tokens/second;
- actual FP16 DRAM logits hash, finite counts and vocabulary shape;
- requested PTQ preset and separately recorded transform execution;
- DMA/systolic co-busy and Port-A lost-write/violation counters for an
  overlap change;
- instruction-read count, systolic phase counts, and accepted DMA read/write
  beats, bursts and bytes.

`summary.logits` is a legacy INT32 ACCUM diagnostic. It is not the decoder's
FP16 logits; the current tools dump the latter from its declared DRAM region
and exclude padded vocabulary columns. Matching hashes of nonfinite outputs
cannot justify performance promotion. `--require-finite-logits` also turns a
standalone shape measurement's nonfinite outputs into a failure.

The direct bundle path does **not** execute the separate QuaRot weight
rotation. `weight_only_int8_quarot` is the historical requested preset name,
not proof of a transformed checkpoint. New reports record
`ptq_transforms.quarot.applied_in_bundle_build=false`, while prior checkpoint
transform history remains unverified. Historical-report clarification
sidecars preserve the original source/runner hashes and measured values.

Logical decoder MAC estimates count canonical unpadded projection, attention,
MLP and vocabulary work. They exclude PTQ transforms, SFU/helper operations and
padding. Ratios against `256 * compute_steps` and `256 * total_cycles` require
the corresponding nonzero counters; these are algorithmic capacity ratios,
not measured array utilization or ISA operation counts.

The shared validator rejects faults, timeout/parse errors, nonempty runner
violations, lost SRAM accesses, and inconsistent phase/DMA accounting.
`instruction_read_count` measures fetch requests, not retired instructions.

`--beat-interval N` spaces read beats in core cycles. The model's write channel
is not throttled by that option, and read/write channels are independent. This
is a read-bandwidth sensitivity model, not shared fixed-total-GB/s DRAM.
No clock is inferred by default: use `--fmax-mhz` only for a labeled assumed
tokens/second conversion.

The historical 34.41 MHz number is not current full-chip timing sign-off after
the July pipeline integration. See
[`docs/project_status.md`](../docs/project_status.md).

## Adding a bench

1. Build on `include/testbench.h` and `tbutil::SimHarness`.
2. Add happy, boundary, and fault cases.
3. Cover dispatch/busy/SYNC behavior for asynchronous engines.
4. Add the source and all relevant headers to the Makefile prerequisites.
5. Negative-control the new check when practical: deliberately break the
   behavior and confirm the bench fails.
6. Add a program-level gate only when the contract crosses module boundaries.
