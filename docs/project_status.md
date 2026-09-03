# TACCEL current project status

**Audited:** 2026-09-03
**Code basis:** `3486c01` plus the documentation refresh following it
**Primary target:** GPT-2 124M W8A16 causal-decoder inference

This file is the current state-of-truth for project capability and measured
status. Dated roadmaps under `docs/` remain experiment records.

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

## Verification snapshot

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
cleanup removed retired tests, producing the current 462-test collection; all
surfaces changed by the cleanup were rerun and passed. A new broad execution
should be compared with that baseline rather than assumed to be zero-failure.

The 2026-09-03 documentation reconciliation rechecked collection, Python
compilation, ISA/assembler behavior, and the golden pin. The 71 selected
ISA/assembler/pin tests passed. A combined run that also included the Stage-5
preset module produced 80 passes and four existing failures. Those tests
monkeypatch the old `taccel.runtime.calibration` symbol after the implementation moved to
`calibration.output_aware`, so the real evaluator sees intentionally incomplete
fixtures and raises `KeyError: n_embd`. This is an active test-maintenance issue,
not a documentation regression.

## Current performance evidence

Latest direct mode-1 measurements after the July pipeline campaign:

| Shape | Context position | Step cycles | Cycles/token | Logits SHA-1 prefix |
|---|---:|---:|---:|---|
| B=1 | 511 | 18,318,261 | 18,318,261 | `eeab004014642d14` |
| B=16 | 510 | 49,998,042 | 3,124,878 | `205682b6515f7e85` |

Conditions:

- GPT-2 124M frozen performance bundle.
- `run_program_synth` with synthesizable SFU/helper datapaths.
- `--fast-beats`, one 128-bit DRAM beat per core cycle.
- Measurements include pipeline changes through `451e7df`.

The older 34.41 MHz post-PNR number was produced before all later long paths
were integrated into the measured SFU. It is useful only as a historical
conversion point. Applying it illustratively gives about 1.879 tok/s for B=1
and 11.012 aggregate tok/s for B=16, but those are not current full-chip
sign-off numbers.

Current timing evidence is a set of standalone or partial-lane measurements:
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
3. Resolve `SOFTMAX_FP32` `0x1C`: implement in RTL or reject/remove it from
   compiler-facing Python APIs.
4. Resolve W4 hardware scope: implement `CONFIG_TILE.weight_int4` in RTL or
   make hardware-target compilation reject W4 bundles.
5. Re-establish a fully characterized broad Python regression baseline with
   all optional fixtures and slow gates available.
6. Re-measure end-to-end fmax and throughput before publishing a new headline.
