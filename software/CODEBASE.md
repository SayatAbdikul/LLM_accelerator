# TACCEL software codebase

**Updated:** 2026-10-13 against the current working tree.

This guide describes the current Python toolchain. The top-level
[README](../README.md) covers repository-wide setup and RTL commands; the
[ISA specification](docs/isa_spec.md) is the encoding reference.

## Package map

```text
software/
├── taccel/
│   ├── isa/                    # Opcode definitions, instruction classes, codec
│   ├── assembler/              # Assembly syntax, ProgramBinary, ProgramBundle
│   ├── compiler/
│   │   ├── frontend/           # DeiT plugin and nanoGPT/GPT-2 graph builders
│   │   ├── emit/               # Generic dispatch, DMA, KV, embedding helpers
│   │   ├── w8a16_emit/         # Active W8A16 matmul/attention/SFU lowering
│   │   ├── ir.py               # IRGraph and IRNode
│   │   ├── model_config.py     # Validated model geometry
│   │   ├── tiler.py            # 16-element padding and tile plans
│   │   ├── memory_alloc.py     # SRAM and DRAM allocation
│   │   ├── options.py          # Explicit targets and opt-in compiler candidates
│   │   ├── validation.py       # Early graph and emitted-instruction validation
│   │   ├── kv_cache.py         # KV-cache layout
│   │   ├── codegen.py          # IR-to-instruction coordinator
│   │   └── decoder_bundle.py   # Prefill/decode ProgramBundle construction
│   ├── quantizer/              # Quantization and transformation primitives
│   ├── golden_model/           # Sequential ISA simulator and memory model
│   ├── runtime/
│   │   ├── calibration/        # Scale search and calibration adapters
│   │   ├── host_runner.py      # Runtime patching and serving loop
│   │   ├── stage5_ptq.py       # GPT-2 PTQ preset registry
│   │   ├── tiny_fixture.py     # Tiny and GPT-2 bundle/test helpers
│   │   ├── numerical_quality.py # Fail-closed teacher-forced quality metrics
│   │   ├── speculative.py      # Prompt-lookup speculative decoding
│   │   └── *_reference.py      # FP32, fake-quant, W8A16/W8A8 references
│   └── utils/                  # Integer and tensor helpers
├── tools/                      # User-facing and diagnostic CLIs
├── tests/                      # Core, hardware, model-quality and research tests
├── run_gpt2.py                 # Golden-model generation/evaluation entry point
├── run_nanogpt.py              # nanoGPT-format entry point
└── chat_gpt2.py                # Interactive host-side runner
```

The old monolithic compiler façade, standalone accuracy script, and
generation-1 SFU golden module no longer exist. The W8A16 emitter and
`ProgramBundle` runtime are the supported path.

## End-to-end flow

```text
checkpoint / state_dict
        │
        ▼
frontend (DeiT or nanoGPT/GPT-2)
        │  IRGraph + ModelConfig
        ▼
quantization and calibration
        │  quantized weights, scales, biases, dtype metadata
        ▼
CodeGenerator + decoder_bundle
        │  prefill/decode instructions, shared data, relocations, patch sites
        ▼
ProgramBundle
        ├──────────────► Simulator ─────► traces / logits
        └──────────────► HostRunner ────► prefill / decode / generation
                               │
                               └────────► serialized ProgramBinary for RTL co-sim
```

## ISA layer

### `taccel/isa/opcodes.py`

Defines:

- the 5-bit opcode allocation;
- instruction formats and field shifts;
- buffer identifiers and capacities;
- the set of retired generation-1 SFU opcodes;
- `CONFIG_TILE` extensions (`weight_int4` and `m_exact`);
- transposed-LOAD fields.

The enum contains all 32 numeric names so an old binary can be diagnosed.
Presence in the enum does not imply hardware support. Consult
[the ISA matrix](docs/isa_spec.md#opcode-matrix).

### `taccel/isa/instructions.py`

Instruction dataclasses validate field ranges and buffer offsets before
encoding. Active classes cover:

- system/configuration: NOP, HALT, SYNC, CONFIG_TILE, CONFIG_ATTN, SET_SCALE,
  SET_ADDR_LO, SET_ADDR_HI;
- data movement: LOAD, STORE, BUF_COPY;
- compute/helper: MATMUL, REQUANT, REQUANT_PC, SCALE_MUL, VADD, DEQUANT_ADD;
- generation-2: DEQUANT_ACCUM_FP32, QUANT_FP32_INT8, VADD_FP32,
  LAYERNORM_FP32, GELU_FP32, SOFTMAX_FP32, MASKED_SOFTMAX_FP32,
  DEQUANT_ACCUM_FP32_SCALED, MAX_ABS_REDUCE_FP32.

Generation-1 Softmax/LayerNorm/GELU instruction classes were removed.

### `taccel/isa/encoding.py`

Encodes every instruction as exactly eight big-endian bytes. Decoding rejects
the six retired opcodes before format dispatch. The codec validates reserved
`CONFIG_ATTN` bits but cannot by itself guarantee RTL support for software-only
features such as `SOFTMAX_FP32` or `weight_int4`.

## Assembler and program containers

### Assembly

`assembler/syntax.py` maps supported mnemonics to instruction objects.
`Assembler` performs label resolution and produces a `ProgramBinary`.
`disassembler.py` emits normalized text for supported instructions. Retired
SFU mnemonics are rejected.

Useful commands:

```sh
PYTHONPATH=software .venv/bin/python software/tools/asm.py input.asm -o out.bin
PYTHONPATH=software .venv/bin/python software/tools/disasm.py out.bin
```

Use each tool's `--help` for current options.

### `ProgramBinary`

A single instruction stream plus a data section and optional trace/compiler
metadata. Binary format version 3 is current; readers retain compatibility
with legacy headers. Instruction PCs are instruction indices, while all DRAM
addresses are byte addresses assembled from 28-bit low/high halves.

### `ProgramBundle`

A relocatable decoder image containing:

- prefill and decode instruction streams;
- one shared data blob;
- temporary, logits, and KV-cache regions;
- symbol offsets and relocation sites;
- runtime SET_ADDR patch sites for token embeddings, positional embeddings,
  and KV bases;
- runtime `CONFIG_ATTN` patch sites;
- stream PCs and complete DRAM-layout metadata.

`ProgramBundle` computes aligned layout addresses and supports controlled
runtime patching without rebuilding the program.

## Compiler

### Model frontends

`compiler/frontend/nanogpt_adapter.py` builds causal GPT/nanoGPT graphs for:

- single-row prefill/decode;
- lockstep batched decode;
- packed attention groups;
- multi-row chunked prefill.

`compiler/frontend/deit_plugin.py` preserves the encoder frontend for software
research. Noncausal softmax is not supported by the `rtl-gen2` target, so an
encoder graph is not automatically a hardware-deployable program.

`ModelConfig` validates dimensions, attention geometry, maximum sequence
length, embedding type, and scale policy. `CodeGenerator` requires an explicit
model configuration; it no longer silently defaults to DeiT.

### IR and tiling

`IRGraph` is an ordered graph of `IRNode` operations. Nodes carry shapes,
inputs, weights, scales, and lowering attributes. Dimensions are padded to
multiples of 16. The compiler strip-mines tensors that exceed ABUF, WBUF, or
ACCUM capacity and stages large weights through DRAM.

Before storage allocation, `compiler/validation.py` rejects duplicate node
names, use-before-producer dependencies, unsupported operations, incompatible
matmul shapes, missing weights/biases/scales, and incorrect parameter dtypes.
Token/position embedding tables and LayerNorm parameters use FP16 storage;
matmul weights are padded 2-D INT8 arrays.

### Code generation

`compiler/codegen.py` coordinates allocation, instruction emission,
relocations, trace events, and the shared data image.

The active numeric lowering lives in `compiler/w8a16_emit/`:

- `matmul.py`: activation quantization, W8 tiled matmul, per-channel epilogue,
  bias folding, and weight-prefetch scheduling;
- `quantization.py`: bounded, opt-in reuse of activation quantization;
- `attention.py` and `packed_attn.py`: QKᵀ, causal softmax, attention-value
  matmul, grouped/batched layouts;
- `sublayer.py`: generation-2 LayerNorm, GELU, softmax, residual add, and
  quantize/dequantize operations;
- `_common.py`: precision flags, `m_exact`, and shared layout helpers.

`compiler/emit/` contains target-neutral dispatch and data movement. Legacy
W8A8 compute fallbacks were removed. `use_fp16_activations=False` and nonempty
legacy fusion/requantization requests now raise errors instead of being
silently ignored.

### Targets and compiler candidates

Decoder bundles and `build_stage3_tiny_decoder_bundle` default to
`target="rtl-gen2"`. This target rejects INT4 weights, noncausal
`SOFTMAX_FP32`, retired opcodes, non-immediate `SET_SCALE`, and generation-2
SFU instructions without FP16 storage flags. Use `target="research-golden"`
explicitly for software-only experiments. The low-level `CodeGenerator`
constructor retains `research-golden` as its default; choose `rtl-gen2`
explicitly when constructing it for hardware.

All fields of `CompilerOptions` default to `False`, preserving the corrected
baseline schedule. They are independent opt-in candidates, not blanket
performance promises:

| Option | Change |
|---|---|
| `indexed_liveness` | Index last-use release events instead of scanning them |
| `real_row_fc2_loads` | Avoid loading padded FC2 activation rows unnecessarily |
| `reuse_qkv_quantization` | Reuse eligible shared-input Q/K/V quantization |
| `reuse_fc2_quantization` | Cache a bounded set of FC2 K-tile quantizations across N groups |
| `shared_data_layout` | Reuse the prefill data layout when compiling decode |
| `deduplicate_data` | Omit unused canonical tiled weights and legacy scale/bias storage |

Pass `compiler_options=CompilerOptions(...)` to bundle builders or
`options=CompilerOptions(...)` to `CodeGenerator`. Each code generator exposes
`compilation_metrics()`: instruction/opcode counts, scheduled DMA bytes,
data-image size, SRAM high-water measurements, and optimization events. These
are static schedule/storage metrics, not measured cycles.

### Decoder bundle construction

`build_decoder_program_bundle` compiles both streams, unions stream-specific
large-weight requirements, validates identical shared data and relocation
compatibility, and returns `DecoderBundleBuild` with its KV layout and both
code generators. The baseline independently lays out both streams;
`shared_data_layout=True` builds the layout once and reuses it.

INT4 packing utilities are present and tested, but the RTL does not currently
consume `CONFIG_TILE.weight_int4`. INT4 bundles are golden/research artifacts,
not hardware-deployable programs.

## Quantization and calibration

`taccel/quantizer/` includes:

- symmetric tensor and per-channel quantization;
- AWQ and GPTQ-related transforms;
- Hessian-guided quantization;
- LayerNorm folding and residual-stream rotation;
- SmoothQuant;
- TurboQuant-KV;
- the active twin-uniform quantizer.

`runtime/stage5_ptq.py` is the source of truth for named GPT-2 presets.
The default when `--ptq-preset` is omitted is
`output_aware_mlp_lm_head_0_11_pc_full_bc`. The frozen performance/cosim bundle
uses `weight_only_int8_quarot`, which is a separate configuration.

W4 tooling stores signed values in `np.int8` during preparation and packs two
two's-complement nibbles per byte in the final data image. The LM head defaults
to W8 under W4 block weights.

## Golden model

`golden_model/state.py` owns DRAM, SRAM, scale/address registers, tile state,
attention context, faults, and the PC.

`golden_model/memory.py` implements typed ABUF/WBUF/ACCUM views and strict
bounds checking. `dma.py` implements LOAD, STORE, BUF_COPY, and transposed
LOAD. `systolic.py` implements tiled INT8×INT8 matrix multiplication.

`golden_model/simulator.py` is the instruction dispatcher and trace engine.
Execution is sequential; `SYNC` is therefore a no-op in the golden model but a
real barrier in RTL. Generation-2 SFU behavior is implemented directly in the
simulator. The existing simulator-content pin is supplemented by
`tests/fixtures/gen2_reference_manifest.json`: SHA-256 pins cover the golden,
ISA, assembler and utility Python modules, package initializer, generation-2
fixture generator, and raw fixture/metadata files. The checker detects changed,
added and removed files in that scope and never refreshes pins automatically:

```sh
PYTHONPATH=software .venv/bin/python software/tools/check_reference_manifest.py
```

A changed reference requires deliberate review and a conformance revision;
do not update expected hashes merely to make a failing test pass.

## Runtime

`HostRunner` loads a `ProgramBundle` and patches it before each invocation:

- `run_prefill` for the prefill stream;
- `run_prefill_chunk` / `run_prefill_chunk_rows` for multi-token chunks;
- `run_decode_step` for B=1;
- `run_decode_step_batch` for lockstep batches;
- `generate` for greedy generation.

Runtime patching updates embedding rows, KV-cache bases, and causal
`CONFIG_ATTN` fields. Prefill KV bases must also be patched for batched or
chunked graphs; otherwise stores can target the shared weight region.
`run_prefill` distinguishes lockstep batches by their multiple decode token-
embedding sites: all streams start at position zero with attention window
`(0, 1)`. Single-stream chunked prefill instead uses positions `0..P-1` and
window `(0, P)`. `run_prefill_chunk` rejects lockstep bundles; they advance
through `run_decode_step_batch`. A batched prefill executes every stream but
stores only its last stream's logits by default, while decode stores all
stream rows. The matrix records stored-row count and source-row mapping
separately from executed token rows and still compares the complete KV cache.

`BundlePatcher` provides the same patch operations without constructing a
simulator or loading another DRAM copy. Serialization and cycle benchmarks use
it when execution is unnecessary.

`generate(..., final_token_commit=False)` avoids executing the final sampled
token when its next-token logits will not be consumed. With at least one new
token returned, its last KV row is therefore uncommitted. To continue, execute
`run_decode_step(tokens[-1], len(tokens) - 1)` and sample the returned logits.
Set `final_token_commit=True` to execute the final token, including EOS, and
leave its KV row and next-token logits committed. This preserves the previous
non-EOS continuation state. With zero new tokens, prefill still executes.
Sampling requires an explicit live `vocab_size`, supplied to `HostRunner` or
`generate`; it is never inferred from padded row bytes. The convenience
generator supports one-row, single-stream bundles, excludes vocabulary
padding, and rejects nonfinite live logits. Chunked/batched clients select
their own logits row. Interactive golden chat uses the same finite, live-
vocabulary sampling contract with FP16 logits and skips its final unused
decode; prompt tokens are still fed sequentially. Diagnostic trace helpers
and the `run_nanogpt`/`run_gpt2` comparison paths intentionally retain the final
committed token and its additional logits; the generation shortcut is not a
speedup claim for those diagnostic paths.

The tiny trained-model gate uses the independent W8A16 reference, matching the
compiler's active FP16-storage contract; the retained W8A8 fake-quant reference
is not interchangeable with it. Fixture metadata checks retain strict file
SHA-256 validation. Newly generated fixtures also record a serialization-
independent payload hash, generator hash and framework version. The GPT-2
converter accepts local `.safetensors` with an explicit `--config` and records
source/configuration/converter provenance without requiring a download.
The legacy non-weight-only perplexity comparison still uses the W8A8 research
reference; its compiled rail now decodes FP16 logits without applying an INT8
output scale again. That cross-arithmetic comparison is distinct from the
matched W8A16 `simulator_backed` conformance path.

`runtime/w8a16_simulator_reference.py` independently models the active W8A16
storage, rounding, streamed matmul and INT8 KV-cache arithmetic. Its explicit
`attention_window` matches the compiled decode reduction width. Shared
calibration maps must include identical prefill/decode KV aliases; an implicit
map is resolved once before constructing either execution path.

`runtime/numerical_quality.py` scores identical teacher-forced prefixes across
compiled W8A16, independent W8A16 and original FP32 logits. It rejects missing
rows, invalid targets and nonfinite live logits/NLL/PPL. Reference acceptance
retains the 2% PPL, 0.995 cosine and four-of-five ranking limits; the permitted
FP32-relative PPL increase is an explicit caller decision, with no default.
See [current numerical evidence](../docs/numerical_quality_results.md) for
separate software and hardware results, rather than treating either as proof
of the other.

`runtime/speculative.py` adds opt-in prompt-lookup speculative decoding. It is
not enabled by default and is guarded by byte-inertness tests.

## Tools

| Tool | Purpose |
|---|---|
| `asm.py` / `disasm.py` | Assembly and disassembly |
| `run_golden.py` | Execute a ProgramBinary in the golden model |
| `convert_hf_gpt2_to_nanogpt.py` | Convert supported Hugging Face GPT-2 weights |
| `evaluate_gpt2_perplexity.py` | PTQ/fake-quant/golden perplexity evaluation |
| `w8a16_quality_gate.py` | Offline teacher-forced W8A16/FP32 quality gate with pinned inputs, raw logits and optional same-model RTL replay export |
| `w8a16_rtl_quality_gate.py` | Three-target hardware quality replay using the canonical export, actual RTL KV continuation and fail-closed evidence checks |
| `rtl_cosim.py` | Serialize a decoder stream and compare RTL with golden |
| `bench_decode_cycles.py` | Direct decode cycle measurement by position/batch |
| `fast_gate_b16.py` | Compare actual FP16 logits and cycles with a matched baseline report |
| `optimization_matrix.py` | Retained baseline/candidate programs, outputs and contiguous KV A/B checks |
| `benchmark_support.py` | Shared runner, provenance, counter and finite-logits checks |
| `check_reference_manifest.py` | Verify frozen executable dependencies and generation-2 fixtures |
| `profile_decode_step.py` | Detailed retire-gap and engine profile |
| `gen_gen2_fixtures.py` | Regenerate frozen generation-2 RTL fixtures |
| `audit_porta_argmax.py` | Port-A overlap/correctness audit |
| `bench_specdec_*.py` | Speculative-decoding measurements |
| `debug_*.py` | Focused quality and lowering diagnostics |

`w8a16_quality_gate.py` accepts only the three active weight-only INT8 presets
and requires `--max-fp32-ppl-increase-percent`. It captures the original FP32
trace before any rotation, records actual PTQ transforms and uses one shared
calibration map. The canonical window is 257 tokens; other explicitly selected
lengths are labeled debug/noncanonical. Output directories are exclusive, and
failed runs retain their captured logits and provenance. `--rtl-export-steps 3`
exports the first three inputs without shortening the compiled attention
window: one static data image, instruction-only ProgramBinary envelopes, and
comparison-only golden logits/whole KV. A separate hardware replay must attach
the static data and carry its own previous RTL KV dump; exporting does not run
RTL or establish hardware quality.

`w8a16_rtl_quality_gate.py` consumes a passed canonical software report with
all 256 ordered compiled steps explicitly halted, a completed three-step
export, unchanged source/export hashes, and actual QuaRot preprocessing. Its
runner must be freshly built with SFU/helper modes 1/1. Run it under an external
RSS/free-disk/time supervisor; the CLI does not build tools or download inputs.
For example, with an existing canonical export and a new output directory:

```sh
PYTHONPATH=software .venv/bin/python software/tools/w8a16_rtl_quality_gate.py \
  --quality build/numerical-quality/quality \
  --runner rtl/verilator/build/run_program_synth/Vtaccel_top \
  --out build/numerical-quality/rtl-replay
```

The first hardware step starts with zero KV; the next two consume the preceding
hardware dump, never golden KV. The gate checks complete instruction retirement,
immutable model data, permitted STORE ranges, finite live FP16 logits and
unchanged identities. It scores exactly three teacher-forced targets against
the same reference/FP32 prefixes, retaining the 2% reference PPL, 0.995 cosine,
four-of-five ranking and selected 5% FP32-relative limits. Exact golden logits
and whole-KV equality are separate diagnostics. This bounded check is not
256-target RTL or corpus-wide quality proof. Metric failures and execution or
validation errors return nonzero and retain a terminal report plus completed
and partial artifacts; existing output directories are never overwritten.

On 2026-10-13, the accepted bounded hardware replay passed for the first three
teacher-forced targets of the pinned 257-token window, retaining the compiled
attention window of 256. All three live-logit rows and whole-KV outputs were
bit-exact against golden, with two actual hardware KV continuation links.
RTL/reference PPL was 1317.182714 versus original FP32 1371.070346
(-3.930333%); cosine was 1 and every top-five overlap was 5/5. Source and output
hash checks closed successfully. Evidence is retained at
`build/optimization-numerical-20261013/quality-rtl-replay-v2/report.json`;
see [the numerical report](../docs/numerical_quality_results.md) for the pinned
inputs, reproduction details and preserved earlier failures. This accepted
scope does not promote a default preset or extend to all 256 hardware targets.

## Tests

Coverage includes:

- ISA encode/decode and assembly round-trips;
- retired-opcode rejection;
- ProgramBinary/ProgramBundle layout and relocation;
- memory allocation, tiling, KV layout, and runtime patching;
- W8A16 code generation and golden execution;
- tiny and batched RTL-versus-golden checks;
- quantizer/reference parity and PTQ policy;
- GPT-2 conversion, logits, perplexity, and deterministic decode;
- chunked prefill and speculative decoding;
- W4 and TurboQuant research paths.

Run the mandatory core, including actual mode-1 RTL conformance:

```sh
make -C rtl/verilator run_program_synth
PYTHONPATH=software .venv/bin/python software/tools/train_tiny_fixture.py
PYTHONPATH=software .venv/bin/python -m pytest -q software/tests --verification-profile required
```

`conftest.py` explicitly selects core modules, excludes optional 124M cases,
and turns a skipped mandatory test into a failure. On 2026-10-13, the current
required profile passed **537 tests**, with zero failures or skips and 244
optional cases deselected from a 781-test collection. The supervised run took
93.831 s and peaked at 689.094 MiB sampled test-process-group RSS. Its log and
resource record are under
`build/optimization-numerical-20261013/supervision-required-final-v1/`;
the exact test outcomes are in
`build/optimization-numerical-20261013/required-final.xml`.

The separate affected run passed 32 tests (eight prefill, six LayerNorm-vector
generator and 18 diagnostic-helper tests) plus five subtests in 13.381 s
supervised wall time, peaking at 300.969 MiB sampled process-group RSS. Its
log/resource record and JUnit outcomes are retained in
`build/optimization-numerical-20261013/supervision-affected-final-v1/` and
`build/optimization-numerical-20261013/affected-final.xml`. These results are
reported separately, not added to the 781-test software collection.

The native regressions separately passed 12/12 checks in each SFU mode and
513/513 LayerNorm vectors per mode. The strengthened GELU gate passed
8,526,144 bit-exact comparisons, 18,522,288 exact 33-stage valid checks and
11,527 independent NaN-input checks, with zero failures. These are overlapping
primitive checks, not additional Python tests. See
[the RTL guide](../rtl/TESTBENCHES.md#fp32-primitives) for coverage and retained
native logs. All 12 standalone FP32 primitive suites also passed in a serial
85.363 s supervised run, peaking at 19.438 MiB sampled process-group RSS;
its log and resource record are under
`build/optimization-numerical-20261013/supervision-all-primitives-final-v1/`.

### Historical October 11 and 12 validation

The following suite and clean-source counts predate the October 13 numerical
repairs. They are retained historical evidence, not a fresh full-suite run of
the current RTL and software.

On 2026-10-11 this profile passed 395 tests with zero skips or failures; 244
optional cases were deselected from a 639-test collection. All eight affected multi-token prefill
tests and both offline Hugging Face converter-parity tests also passed after
the AV-residency, stored-row mapping and host lockstep-priming fixes.

On 2026-10-12, the 180 optional tests with only earlier-source evidence were
rerun against the same 177 Python source-file hashes: all passed with no
failures or skips in 230.981 s wall time, with 830.56 MiB peak sampled
test-process-group RSS. That rerun established 585 distinct current-source
passes: 395 required and 190 optional. Its exact IDs, logs and source/resource
records are retained locally under
`build/optimization-20261011/python-final/earlier-current-180/`.

The separately bounded 124M RTL metric test then passed on 2026-10-12 in
702.241 s wall time, with 2850.5 MiB peak sampled test-process-group RSS and
source hashes verified at completion. The regular optional phase passed 46/48
tests and the slow phase passed 4/5, with no skips or incomplete outcomes.
This completed the original 639-test epoch at 636 passes and three failures,
with all 177 Python source-file hashes unchanged. Its ledger, retained logs
and identity manifest remain unchanged under
`build/optimization-20261011/python-final/final-optional/`.

A subsequent test-contract repair changed only `software/tests/test_run_gpt2.py`,
adding explicit `--ptq-preset` arguments compatible with the CLI fixtures;
assertions and numerical tolerances were unchanged. Both tests in that module
and its verification-contract importer passed a fresh three-test run in
45.49 s pytest time (47.195 s supervised wall time, 2753.844 MiB peak sampled
test-process-group RSS). The reconciled result is 637/639: 395 required and
242 optional passes, comprising 634 dependency-validated carried passes and
three fresh passes. This is not a claim that all 639 tests ran under one source
hash. The new ledger and retention manifest are under
`build/optimization-20261011/python-final/test-contract-repair/`.

Both research PPL gates remain failed: relative golden/fake-quant PPL differences
were 95.6993% for the regular gate and 62.5625% for production calibration,
against the unchanged 2% limit. Thus this is not a whole-suite or language-quality
pass. The original failures remain retained; no numerical threshold was relaxed.
The 124M log is retained at
`build/optimization-20261011/python-final/final-optional/execution/124m/00_test_compare_rtl_golden/pytest.log`.

The 124M result characterizes two finite RTL/golden logits steps, not an
accuracy threshold pass. Argmax differed on both steps, minimum cosine was
0.999596715, and the first tensor divergence was a 1-FP16-ULP difference at
`block0_ln1` against its frozen 0-ULP band. These diagnostics remain visible;
finite outputs do not establish full-model numerical equivalence or language
quality.

The final isolated clean-source reproduction passed 395 required tests, eight
prefill tests and 39 portable ASIC static/mocked tests. Newly compiled runners
used SFU/helper modes 1/1 with fetch disabled/enabled; all eight tiny-model
RTL/golden comparisons and four exact A/B comparisons passed for logical FP16
logits and full KV state. All 428 exported source hashes stayed unchanged during
the run. Existing installed dependencies were reused, not freshly installed.
These repeat checks do not add distinct tests to the 637/639 result or resolve
the PPL failures. Retained evidence is under
`build/optimization-20261011/clean-source-final/`.

The broader developer profile is `PYTHONPATH=software .venv/bin/python -m
pytest -q software/tests`. Use `--collect-only -q` to inspect its current scope.
Large checkpoints, slow PPL gates, Hugging Face parity, 124M co-simulation, and
external replay data are optional. Relevant tests skip when their prerequisite
is absent or require `PYTEST_SLOW=1` / `PYTEST_124M=1`. Stage-5 preset tests now
patch the defining calibration module. Optional skips are not successful gates.
CLI smoke tests use the current Python interpreter by default; set
`TACCEL_TEST_RUNNER_PYTHON` only when an alternate environment is needed.

### RTL and measurement contracts

`rtl_cosim.py` defaults to `--runner synth`, selecting `run_program_synth`.
`--runner reference` explicitly selects the DPI runner; `--rtl PATH` selects a
specific executable. Hardware gates require reported SFU and helper modes of
1 and reject stale runners, faults, lost SRAM accesses and inconsistent phase
or directional-DMA counters. Per-op comparisons retain GELU's 3-ULP allowance;
other operations are exact, and differing NaN/Inf encodings fail the native
special-value contract.

Benchmarks read decoder logits from their FP16 DRAM region, excluding padded
vocabulary columns. The runner's legacy `summary.logits` field is an INT32
ACCUM diagnostic, not decoder logits. Reports hash the actual outputs and
record finite counts; matched performance promotion requires finite outputs
on both runs. Standalone zero-KV decode shapes measure cycles, not model
accuracy. Full-model co-simulation rejects nonfinite logits, but its opt-in
metric test does not impose an accuracy threshold.

Reports explicitly record QuaRot transform execution. Direct bundle/shape
builders do not apply the separate rotation preprocessing step, even when the
requested preset is named `weight_only_int8_quarot`; prior transformations of
an arbitrary supplied checkpoint remain unverified. The full perplexity
pipeline applies QuaRot separately when requested. Historical reports retain
their original source/runner identities; post-run clarification sidecars do
not rewrite measured data.

The logical decoder MAC field is a canonical algorithm estimate from model
geometry, excluding PTQ transforms, SFU/helpers and padding. Its ratios to
`256 * compute_steps` or `256 * total_cycles` are only emitted for available,
nonzero counters and are not measured ISA operation counts or array utilization.

`--beat-interval` controls read-beat spacing only; writes are unthrottled by
that option, so it is not a shared fixed-total-bandwidth model. Reports retain
checkpoint/calibration/program/runner/source identities and actual hardware
configuration. Tokens/second are omitted unless `--fmax-mhz` supplies an
explicit assumed clock. See [the RTL guide](../rtl/TESTBENCHES.md) for commands.

## Change rules

1. Keep `opcodes.py`, `encoding.py`, `instructions.py`,
   `rtl/common/src/include/taccel_pkg.sv`, and `decode_unit.sv` synchronized.
2. Preserve the eight-byte instruction width and 16-byte SRAM address unit.
3. Treat a golden dependency or fixture hash update as a conformance revision.
4. Regenerate generation-2 fixtures after an intentional golden or codec
   change and confirm whether raw payloads moved.
5. Any DMA/systolic overlap change must preserve logits hashes and report the
   Port-A audit counters; byte equality on a non-overlapping tiny graph is not
   sufficient.
6. Keep hardware-target validation explicit; W4 remains a research-golden
   feature until RTL implements the complete contract.
