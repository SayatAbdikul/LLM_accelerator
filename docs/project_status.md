# TACCEL current project status

**Audited:** 2026-10-11–12
**Code basis:** `978f126` plus the optimization implementation working tree
**Primary target:** GPT-2 124M W8A16 causal-decoder inference

This file is the current state-of-truth for project capability and measured
status. Dated roadmaps under `docs/` remain experiment records.

## October implementation

See [the optimization results](optimization_results.md) for commands, current
test counts, corrected mode-1 cycle measurements, candidate acceptance, and
remaining gates. The September tables below are historical snapshots, not
fresh verification of the October changes.

Implemented candidates include real-row FC2 loading, exact QKV/FC2
quantization reuse with bounded lifetimes/fallback, indexed liveness, shared
prefill/decode data layout, data deduplication, and a one-beat instruction
buffer. They remain individually selectable through explicit compiler
options and separate buffered/unbuffered hardware runners. Hardware-target
IR validation rejects W4 and noncausal softmax before emission; the research
golden target preserves those software experiments.

Corrected DMA handling splits at 4 KiB and detects address overflow. A
directed concurrent SRAM/STORE regression additionally guards stalled AXI
write-data stability. Actual execution parameters and directional DMA,
instruction-fetch, and systolic-phase counters are recorded in JSON.

Obsolete Stage-5/embedding/KV tests and checkpoint metadata provenance were
repaired without relaxing numeric tolerances. The selected like-for-like
regression reference now models W8A16; legacy PPL comparisons still use W8A8
and remain failed. A 73-file executable-reference manifest and required
no-skip test profile now distinguish mandatory conformance from optional
large-model quality.

The final required profile passes 395 tests, plus eight affected prefill tests,
without failures or mandatory skips after short-context AV capacity and host
batch-priming corrections. Collection contains 639 nodes, with 244 optional
cases excluded from the required profile; this is not a final-source full-suite
pass. Reconciled evidence covers 637 distinct passing tests (395 required and
242 optional). All 54 formerly pending nodes execute without skips/timeouts;
their original 177-file epoch passes 51 and fails three. A test-only compatible
preset repair then passes all three affected checks, combining dependency-
unchanged evidence with fresh reruns rather than claiming one-source-hash
execution of all 639. Both legacy PPL failures remain unresolved.
The 124M finite-output metric records a
one-ULP first snapshot difference and 0/2 argmax agreement; its pass is not
a full-model accuracy or numerical-equivalence gate.
The optional default-preset 257-token comparison also fails its original
2% limit (95.699% difference); production calibration also fails (62.563%).
These compare compiled W8A16 with a retained
legacy W8A8 fake-quant policy, not matched RTL or the optimization preset;
these failed research gates and poor PPL results remain explicit.
Fresh native runners
pass the 128-check affected campaign. Four mode-1 core/wrapper hierarchy gates
resolve successfully, while their 27,891 raw-process diagnostics are not an
asserted structural proof. See the results report for identities and resources.

The final full GPT-2 campaign passes 14 shape executions/seven exact pairs
and 12 contiguous-sequence executions/six exact pairs. Stored FP16 logits
are finite and full KV matches. The sequence inputs demonstrably carry the
preceding RTL KV, rather than resetting history. The combined opt-in profile
reduces fixed-mix shape cycles by 7.264%, and three-step B1/B16 sequence cycles
by 12.181%/10.284%. These are matched cycle measurements, not trained-quality,
achieved-frequency or hardware-throughput claims.

Clean-source/fresh-build reproduction closes integration of those opt-in
changes: 395 required tests, eight affected prefill tests, 39 portable ASIC
static/mocked tests, eight tiny-model golden comparisons and four exact A/B
pairs pass. Both mode-1 runners are freshly compiled and match the performance
runner hashes; all 428 exported source hashes stay unchanged during the run.
Existing installed dependencies are reused, not freshly installed. This is
not an additional distinct-test count, full-model accuracy or physical signoff.
Source, binaries, logs and payload evidence are retained under
`build/optimization-20261011/clean-source-final/`. Compiler defaults and wrapper
fetch defaults remain unchanged.

The SKY130 macro-bank prototype is deliberately isolated: a delayed-write
overlay passes 96,804 equivalence cycles, including the inferred old-data
Port B collision contract. Macro timing is not yet validated. Real-memory
chip binding, IO, full-chip PNR, DRC/LVS and
power remain unresolved. SKY130/ICSprout55 leaf reports must not be scaled
into full-chip area, achieved frequency or energy/token.

## What works

### Software

- Fixed-width ISA encoding, validation, assembly, disassembly, relocation, and
  runtime patch sites.
- nanoGPT/GPT-2 and DeiT graph frontends.
- W8A16 code generation with tiled large weights, packed attention, INT8 KV
  cache, dynamic activation scaling, and folded bias epilogues.
- Relocatable `ProgramBundle` images with prefill and decode streams.
- Golden execution through `Simulator` and serving through `HostRunner`.
- Batched decode at B=1, B=16, and B=32; chunked multi-token prefill; optional
  prompt-lookup speculative decoding.
- PTQ research paths for W8 and W4, including QuaRot, AWQ, GPTQ, SmoothQuant,
  Hessian-guided calibration, and TurboQuant-KV.

### RTL

- In-order fetch/decode/control with architectural faults.
- 16×16 chained INT8 systolic array and INT32 accumulation.
- Burst DMA plus transposed INT8 LOAD.
- Blocking retained helper instructions.
- Generation-2 W8A16 SFU with pipelined divide, square root, exponential,
  GELU, dequantization, LayerNorm finalization, and MAX_ABS scale generation.
- ABUF/WBUF/ACCUM SRAM subsystem with separate systolic write Port S and
  collision faults.
- Native Verilator unit, subsystem, primitive, and program-level benches.
- Full shared-core generic Yosys hierarchy/check/stat gate.
- SKY130 and FPGA wrapper elaboration checks.

## Current ISA boundary

The instruction word is 64-bit big-endian with a 5-bit opcode.

- Six generation-1 SFU opcodes (`0x0E`, `0x0F`, `0x10`, `0x12`, `0x15`,
  `0x16`) are retired. Names remain only for diagnostics; the assembler,
  executable decoder, golden simulator, and RTL do not run them.
- Retained helper opcodes (`REQUANT`, `SCALE_MUL`, `VADD`, `REQUANT_PC`,
  `DEQUANT_ADD`) remain legal in RTL.
- The causal generation-2 path uses `MASKED_SOFTMAX_FP32` (`0x1D`).
- `SOFTMAX_FP32` (`0x1C`) is represented and executable in Python but is
  deliberately illegal in RTL because no supported causal frontend consumes
  it. This software/RTL asymmetry is unresolved.
- `CONFIG_TILE.weight_int4` is implemented by Python encoding, packing, and
  golden execution. RTL does not decode bit 28, so W4 is not a hardware mode.

See [the ISA specification](../software/docs/isa_spec.md) for the complete
opcode and field matrix.

## Historical September verification snapshot

The September cleanup established this post-refactor baseline:

| Gate | Result |
|---|---|
| Python module compilation | Pass |
| Complete Python test collection | 462 tests collected |
| Focused ISA/golden/W8A16 tests | 144 passed |
| Assembler, bundle, and RTL-golden tests | 43 passed, 1 optional 124M test skipped |
| Active quantizer twin | 27 passed |
| Native decode/control/SFU/chained/QKT checks | 94 passed |
| QKT replay/padded cases | 17 skipped without external replay data |
| Generation-2 fixture regeneration | 10 cases; zero raw payload drift |
| Golden simulator freeze hash | `cc1bc64f34bf7b5a53a5760bcd500dca10cb8080` |
| Full-design generic Yosys elaboration gate | Pass; 83.24 s, 1.93 GB peak, 54,998 hierarchy cells |

The last broad Python execution before cleanup collected 493 tests and exposed
pre-existing fixture, embedding, and W4 failures plus one very long test. The
cleanup removed retired tests, producing the then-current 462-test collection; all
surfaces changed by the cleanup were rerun and passed. A new broad execution
should be compared with that baseline rather than assumed to be zero-failure.

The 2026-09-03 documentation reconciliation rechecked collection, Python
compilation, ISA/assembler behavior, and the golden pin. The 71 selected
ISA/assembler/pin tests passed. A combined run that also included the Stage-5
preset module produced 80 passes and four existing failures. Those tests
monkeypatch the old `taccel.runtime.calibration` symbol after the implementation moved to
`calibration.output_aware`, so the real evaluator sees intentionally incomplete
fixtures and raises `KeyError: n_embd`. This was a test-maintenance issue,
not a documentation regression; the October contract repairs address it.

## Historical performance evidence

Direct measurements after the July pipeline campaign (not the corrected
October both-datapaths-mode-1 baseline):

| Shape | Context position | Step cycles | Cycles/token | ACCUM diagnostic SHA-1 prefix |
|---|---:|---:|---:|---|
| B=1 | 511 | 18,318,261 | 18,318,261 | `eeab004014642d14` |
| B=16 | 510 | 49,998,042 | 3,124,878 | `205682b6515f7e85` |

Conditions:

- GPT-2 124M frozen performance bundle.
- Historical `run_program_synth` selected the synthesizable SFU; helper-mode
  build provenance was not verified. The old target omitted that override.
- `--fast-beats`, one 128-bit DRAM beat per core cycle.
- Measurements include pipeline changes through `451e7df`.

The older 34.41 MHz post-PNR number was produced before all later long paths
were integrated into the measured SFU. It is useful only as a historical
conversion point. Applying it illustratively gives about 1.879 tok/s for B=1
and 11.012 aggregate tok/s for B=16, but those are not current full-chip
sign-off numbers.

Historical timing evidence is a set of standalone or partial-lane measurements:
known long combinational paths were reduced to roughly one FP primitive per
stage, generally in the 27–32 ns range depending on mapping context. The final
whole-lane post-`451e7df` placement rerun exceeded the available 15 GB memory.
Full-SFU/full-chip PNR on a larger machine remains necessary.

## Physical implementation status

### ASIC

- SKY130 wrapper, behavioral SRAM macro boundary, Yosys scripts, OpenSTA
  drivers, and direct OpenROAD block scripts exist.
- Per-block and primitive measurements exist.
- SRAM macro banking, real IO integration, full-chip placement/routing,
  DRC/LVS, and a tape-out target are unresolved.

### FPGA

- The shared core and FPGA wrapper elaborate through `sv2v + yosys`.
- Clock, reset, DDR, and IO blocks are placeholders.
- No device, board, vendor project, or constraints file is selected.

## September cleanup

Commit `3486c01` removed 7,747 lines and added 155 across 60 files:

- Removed the dormant Cocotb tier.
- Removed the obsolete control-only synthesis gate and black-box stubs.
- Removed generation-1 SFU instruction classes, assembly syntax, golden
  implementation, compiler code, and tests.
- Removed the invalid QKT-history test and its history-only helpers.
- Removed dead SFU/compiler helpers and stale imports.
- Updated current build files, fixture provenance, tests, and ISA diagnostics.

Retired numeric opcode constants remain so old binaries fail with a clear
diagnostic rather than changing the numeric allocation.

## Open decisions and work

1. Run full-SFU/full-chip physical timing on a machine with sufficient memory.
2. Select and integrate ASIC SRAM macros and IO cells, or select an FPGA board
   and replace the platform stubs.
3. Decide whether to add noncausal softmax hardware; current hardware-target
   compilation already rejects it, while research-golden remains explicit.
4. Decide whether to add W4 hardware; hardware-target compilation already
   rejects it, while research-golden retains INT4 experiments.
5. Resolve the two retained legacy PPL failures and the characterized 124M
   RTL/golden numerical gap before promoting trained-model quality; all 639
   collected Python nodes already have an explicit disposition.
6. Re-measure end-to-end fmax and throughput before publishing a new headline.
