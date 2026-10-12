# Accelerator optimization roadmap

**Created:** 2026-10-11  
**Updated:** 2026-10-12  
**Code basis:** `978f126` and the October code audit  
**Status:** Accepted opt-in integration validated; numerical quality and physical integration remain gated  
**Primary workload:** GPT-2 124M causal inference with W8A16 and INT8 KV cache

Optimize useful work per cycle and memory traffic first, then establish physical
area and timing with real memories. Keep SKY130 as the reference backend and
evaluate ICSprout55 through matched block experiments. A process change is not
a substitute for fixing memory implementation or verification gaps.

The order is governed by acceptance gates, not calendar estimates. Preserve
numerical behavior for scheduling and storage changes; treat a numerical or ISA
change as a separate proposal. Host latency, compiler efficiency, accelerator
cycles, and physical power are separate outcomes and must be reported separately.

## Starting point

The starting October audit ran all 56 Python test modules separately: 419 tests passed,
10 failed, and 33 skipped. Six RTL suites passed 143 checks. Seven Python
failures concern outdated test contracts, two concern checkpoint provenance,
and one is an unresolved trained fake-quant reference divergence.

The generic Yosys elaboration completed in 85.3 seconds with 2.25 GiB peak
resident RAM, but selected both compute-mode parameters as zero. This is not a
measurement of the intended mode-1 mapping or physical implementation.

Historical `run_program_synth` records provide comparison context, not the
baseline for future changes. The pre-optimization target selected SFU mode 1
but left helper mode 0; the implementation now sets both modes explicitly.
Original build provenance must still be verified before assigning a complete
hardware-mode label to the historical measurements:

| Workload | Recorded step cycles | Recorded cycles per token |
|---|---:|---:|
| B=1 decode at position 511 | 18,318,261 | 18,318,261 |
| B=16 decode at position 510 | 49,998,042 | 3,124,878 |

These records use one 128-bit DRAM beat per core cycle. Current full-chip area,
frequency, and power are not established. Do not convert cycles using the old
34.41 MHz partial-design timing result and label them achieved throughput.

See [project status](project_status.md) for the historical measurements and
[the architecture tour](code_exploration_roadmap.md) for code navigation.

## Implementation status

The [optimization results](optimization_results.md) record current commands,
measurements and limitations. The milestone definitions below remain the
acceptance contract; implemented code alone does not close a promotion gate.

| Milestone | October implementation status |
|---|---|
| M0 | Complete for the declared functional scope: 395 required plus 8 affected prefill tests pass after capacity/host fixes; references unchanged, fresh mode-1 native gates pass; optional large-model quality is separate |
| M1 | Complete for measured scope: corrected B1/B16 baselines, 14 full-model shape executions/seven exact pairs (−7.26% fixed-mix cycles), 12 representative read-pacing runs/six pairs and four exact cycle repeats |
| M2 | Real-row loads, fetch retention, QKV/FC2 reuse and host/compiler candidates implemented; representative 198-run / 176-comparison matrix passes (−8.49% fixed-mix cycles); final short-context B16/B32 gate passes all golden checks; full GPT-2 continuation passes 12 runs / 6 comparisons and actual KV carry (B1 −12.18%, B16 −10.28%) |
| M3 | Delayed-write bank passes 96,804 sampled-cycle equivalence checks and maps to eight macros plus measured periphery; matched five-leaf PDK measurements complete; macro timing remains gated before chip binding |
| M4 | No larger architecture change selected; use the new phase counters and physical evidence to choose a separate experiment |
| M5 | Complete for accepted opt-in M2 changes: clean-source/fresh-build reproduction passes 395 required + 8 prefill + 39 static/mocked ASIC tests and 8 tiny golden / 4 exact A/B comparisons. All optional nodes are dispositioned, with 637/639 passes after a verified test-only repair; default promotion, 124M numerical conformance and physical integration remain separate open gates |

Next-group packed-attention prefetch was evaluated and deferred: the current
group fills WBUF, ABUF staging conflicts with systolic ownership, and helper
copy overhead has no demonstrated net benefit. It is not silently included in
the combined candidate. Smaller shared/deduplicated layouts can also change
4 KiB burst alignment and overlap; measured size/latency tradeoffs are recorded
instead of assuming every layout reduction improves execution time.

## Milestones and dependencies

| Milestone | Work | Dependency | Completion gate |
|---|---|---|---|
| M0 | Correctness and execution configuration | None | Required tests pass and intended hardware modes are verified |
| M1 | Reproducible performance baseline | M0 | Matched B1/B16 reports and a reusable comparison procedure |
| M2 | Compiler scheduling and small RTL improvements | M1 | Correct outputs and measured benefit for each accepted change |
| M3 | Memory implementation and physical feasibility | Feasibility starts during M0; implementation uses M1 gates | Verified memory banks and credible block area/timing reports |
| M4 | Larger architecture experiments | M2 measurements; M3 for PPA claims | Benefit survives functional, cycle, and physical tradeoffs |
| M5 | Integrated optimized baseline | Accepted M2–M4 changes | Clean-checkout reproduction and an updated supported-results matrix |

M2 compiler changes and the fetch-buffer experiment can be developed
independently, then measured again together. M3 macro research can proceed in
parallel, but memory-heavy tool jobs must remain serial on the laptop.

## M0 Correctness and execution configuration

1. **Select the hardware datapaths explicitly.** Set and verify both
   `SFU_SYNTH_MODE=1` and `HELPER_SYNTH_MODE=1` in generic synthesis, direct ASIC
   synthesis, and performance runners. The pre-implementation `run_program_synth`
   and `run_program_noovl` targets selected SFU mode 1 but omitted the helper
   override. Preserve the separate DPI reference configuration. Record resolved
   parameters in build/run metadata. Program-level co-simulation now defaults
   to the mode-1 `run_program_synth` runner; the DPI/reference runner remains
   an explicit diagnostic choice.
2. **Make DMA transactions legal.** Split LOAD, STORE, and transposed LOAD
   bursts at 4 KiB boundaries. Reject effective-address overflow before
   truncation. Add tests at `0x0ff0`, `0x0fe0`, and arbitrary aligned offsets,
   plus randomized backpressure, response errors, and handshake assertions.
3. **Reject invalid programs before emission.** Add hardware/research target
   profiles and IR validation. Reject unsupported W4/noncausal softmax hardware
   requests, unknown operations, duplicate names, missing weights, and invalid
   shapes or dtypes rather than silently emitting incomplete programs.
4. **Repair the regression baseline.** Update obsolete tests, restore or
   regenerate matching checkpoint/metadata pairs, and trace the trained-model
   mismatch at its first divergent token. Do not replace hashes or relax
   tolerances merely to make the tests pass.
5. **Strengthen reference integrity.** Pin the full executable golden-model
   dependency set and fixture payloads. Define mandatory and optional test
   profiles; missing mandatory inputs must fail, not skip. Add explicit
   finite-value and numerical-quality contracts before promoting full-model
   accuracy support.

Acceptance: all required profile tests pass, the new negative tests fail on the
old defective behavior, and resolved mode-1 configuration is captured in the
reports. Optional omissions have explicit reasons. Elaboration, mapped
synthesis, and routed results remain distinct categories.

Starting points: [build targets](../rtl/verilator/Makefile),
[generic gate](../rtl/synth/synth_check.ys),
[DMA](../rtl/common/src/dma_engine.sv),
[compiler](../software/taccel/compiler/codegen.py), and
[conformance tests](../software/tests/test_compare_rtl_golden.py).

## M1 Reproducible performance baseline

Rebaseline after M0: protocol splitting and corrected execution modes may
change cycles. Do not attribute those changes to later optimizations.

| Tier | Workloads | Purpose |
|---|---|---|
| Every candidate | Changed-block tests; tiny B1/B16, multistep and chunked-prefill co-simulation on the selected hardware runner | Fast correctness and scheduling feedback |
| Candidate promotion | GPT-2 B1 positions 0, 63, 255, 511; B16 positions 63, 255, 510 | Context sensitivity and full-step benefit |
| Milestone integration | B32 compatibility, prompt lengths 1/16/128, generation and continuation tests | Avoid optimizing decode at the expense of adjacent supported paths |
| Memory sensitivity | Representative B1/B16 cases at 1, 2, and 4 cycles per beat | Expose bandwidth assumptions and clock-scaling limits |

Keep shape-only cycle tests with synthetic KV contents separate from realistic
teacher-forced numerical tests. A cycle benchmark is not an accuracy gate.

Each report records commit and patch identity, tools, resolved RTL parameters,
model/tokenizer/calibration hashes, input and bundle hashes, batch, context,
memory pacing, and output hashes. Capture:

- Total cycles, cycles/token, instruction count, instruction-read count, and
  fetch AR stalls.
- DMA bytes, beats and bursts; per-engine busy cycles; DMA/systolic co-busy;
  lost-access and forbidden-collision counters.
- MAX_ABS/QUANT counts and ABUF/WBUF/ACCUM high-water marks.
- Systolic load, enabled-compute, K-boundary, flush, and drain cycles. Derive
  useful MAC utilization from logical dimensions, not enabled padded PEs.
- Compiler time, peak host RAM, bundle size, and physical-run disk use.

Use existing [decode benchmarks](../software/tools/bench_decode_cycles.py),
[B16 gate](../software/tools/fast_gate_b16.py), and
[runner counters](../rtl/verilator/run_program.cpp). Extend them where a metric
or runner selection is missing. Remove or clearly label the benchmark's
historical default frequency conversion; report cycles until timing is measured.

Acceptance: repeated deterministic runs agree; baseline artifacts are retained;
the comparison reports both local counter changes and end-to-end results.
Overlapping busy counters must not be added as disjoint parts of total time.

## M2 Compiler scheduling and small RTL improvements

Land each experiment separately so its benefit and regressions are attributable.
The effort labels describe relative scope, not promised completion times.

| Order | Experiment | Scope | Acceptance criteria |
|---|---|---|---|
| 1 | Load only real activation rows before FC2 quantization | Small compiler change | Padding remains initialized correctly; transferred bytes fall; valid output bytes are unchanged |
| 2 | Retain both instructions from one 128-bit fetch beat | Small RTL change | Sequential instruction reads approach one per pair; fault, restart and coherence tests pass; report full-step cycle effect |
| 3 | Reuse QKV activation quantization | Medium compiler change | Matching consumers reuse identical quantized bytes and scale bits; MAX_ABS/QUANT counts fall without unsafe buffer lifetimes |
| 4 | Reuse FC2 quantized K tiles across N groups | Medium compiler change | Tile scale bits and accumulation order stay unchanged; net DMA/SFU work falls after any staging overhead |
| 5 | Prefetch the next packed-attention group | Medium scheduling change | Intended overlap is exercised; no lost accesses/collisions; B16 step cycles fall with unchanged results |

### Instruction buffer

Start with a one-beat buffer, not multiple outstanding AXI transactions.
Define invalidation for reset, restart, host program patching, and any permitted
DMA store into instruction memory. Test both instruction halves, address changes,
errors, and fetch/DMA arbitration. Fewer fetch reads without fewer step cycles
is an efficiency improvement, not a throughput improvement.

Code: [fetch unit](../rtl/common/src/fetch_unit.sv) and
[top-level arbitration](../rtl/common/src/taccel_top.sv).

### Activation reuse

Represent a reusable quantized activation with tensor version, dimensions,
precision, padding, scale pair, storage location, and lifetime. Start QKV reuse
within one head, reducing three compatible quantizations to one; extend across
heads only when storage and scale-register lifetimes permit. Reusing INT8 bytes
with a subsequently overwritten scale register is incorrect.

For FC2, first evaluate retaining individual quantized K tiles against staging
them in DRAM. Preserve K partitioning, local scale rounding, partial-sum order,
and first-K-only bias application. Do not silently substitute a global scale or
reorder floating-point accumulation. Reject a schedule whose staging cost
eliminates its end-to-end benefit.

Code: [matmul emitters](../software/taccel/compiler/w8a16_emit/matmul.py),
[frontend projections](../software/taccel/compiler/frontend/nanogpt_adapter.py),
and [packed attention](../software/taccel/compiler/w8a16_emit/packed_attn.py).

### Host and compiler efficiency track

These changes may proceed after M0 but must not be advertised as faster chip
execution per decode step:

- Pre-index allocations by their final-use node. Require byte-identical
  instructions/data and measure compile time and peak RAM.
- Build one shared prefill/decode data layout and remove representations only
  when neither stream references them. Validate relocation, embedding patch
  sites, and separation of weights, scratch, logits, and KV storage.
- Avoid the final unused decode in generation only after specifying KV
  continuation behavior. With final-token commit disabled, N positive generated
  tokens without EOS should require one prefill and N−1 decode calls. Cover N=0,
  EOS, continuation, and unchanged returned tokens explicitly.

Code: [bundle builder](../software/taccel/compiler/decoder_bundle.py) and
[host runner](../software/taccel/runtime/host_runner.py).

## M3 Memory implementation and physical feasibility

### Main SRAM and SFU storage

1. Inventory the ports, latency, collision behavior, and capacity required by
   each buffer. Main storage is 128 KiB ABUF, 256 KiB WBUF, and 64 KiB ACCUM.
2. Select macros and banking that implement the actual contract: one-cycle
   reads, write-first Port A behavior, supported cross-port accesses, and the
   independent systolic routing to different buffers. Verify Liberty, LEF,
   layout, and simulation views before relying on a macro.
3. Build and verify one representative bank before integrating all 448 KiB.
   Report decoder/mux/peripheral cost separately from memory-macro area.
4. Build an opcode-by-opcode port-demand and lifetime table for the roughly
   19 KiB SFU scratch arrays. Evaluate explicit banks and synchronous reads;
   remove bulk data resets only when validity/initialization proofs and tests
   establish that unwritten contents cannot escape.
5. Confirm the intended bank structure in synthesis reports. Compare mapped
   area, critical paths, cycles, and peak synthesis RAM before expanding scope.

Code: [ASIC memory boundary](../rtl/asic/src/sram_dp_sky130.sv),
[shared memory contract](../rtl/common/src/memory/sram_dp.sv), and
[SFU engine](../rtl/common/src/sfu_engine.sv).

### Matched SKY130 and ICSprout55 experiments

Parameterize PDK and tool paths and pin revisions, cell libraries, corners,
constraints, and tool versions. First compare a PE or small array, DMA, and a
representative SFU primitive/lane. Use identical corrected RTL, parameters,
memory boundaries, load assumptions, and timing objectives.

For each backend, report standard-cell area, macro area/coverage, timing stage
(mapped or placed/routed), utilization, peak host RAM, runtime, disk use, and
unavailable physical checks. At matched frequency and workload, compare
activity-based power or energy per operation only when models and extraction
support the claim. Keep leakage, core dynamic power, SRAM, and external-memory
power separate; report energy/token only for a sufficiently complete model.

Begin SRAM availability and flow-feasibility research during M0. Switch the
primary technology only after these experiments show a usable memory solution
and credible verification/extraction coverage, not merely smaller logic area.
Public entry points: [SKY130](https://skywater-pdk.readthedocs.io/en/main/) and
[ICSprout55](https://github.com/openecos-projects/icsprout55-pdk).

### Laptop resource policy

The planning host is an M1 with 16 GiB RAM and about 53 GiB free disk at audit.
Run only one heavy mapping, place-and-route, or full-model simulation job at a
time. Initially budget approximately 8–10 GiB for a heavy job and retain 20 GiB
free disk as planning guardrails, not measured design requirements. Capture
peak RSS and avoid unbounded waveform dumps.

Do not retry known out-of-memory whole-SFU flattening unchanged. If partitioned
work still causes sustained swap pressure, retain its logs and use a larger
host sized from measured demand. Full-chip work waits for real memory bindings
and partition resource measurements. Moving from 130 nm to 55 nm does not
automatically reduce the host's netlist or routing-memory requirements.

## M4 Larger architecture experiments

Choose at most one experiment at a time from the new bottleneck measurements:

| Experiment | Entry condition | Required evidence |
|---|---|---|
| Exact-row systolic execution or a B1 GEMV mapping | Padding/load/drain overhead remains significant after M2 | Explicit target/ISA treatment of observable padding writes; B1 improvement without B16 regressions |
| Systolic A reuse or drain overlap | Phase counters identify exposed repeat work | Correct accumulator snapshots, K chaining, SRAM ownership, and reduced full-step cycles |
| SFU lane-width/resource-sharing sweep | Per-op utilization and mapped area identify costly underused units | Compare 4/8-lane or shared alternatives at equal constraints; report area, cycles, timing, and energy tradeoffs |
| Additional retiming | Current mapped/placed paths identify a real critical cone | Primitive numerical equivalence, valid/latency tests, and improved time per inference after added cycles |
| Wider memory interface or more outstanding traffic | Fixed-bandwidth tests show an exposed memory bottleneck | Realizable external interface, protocol verification, and benefit after routing/area costs |

Do not restart already-landed work: systolic double buffering, chained K
execution, weight prefetch, separate systolic SRAM routing, SFU streaming,
and pipelined exp/GELU/div/sqrt are present. W4 hardware, a larger mesh, new
clock domains, and full technology migration are separate scope decisions,
not prerequisites for this roadmap.

## M5 Integration and acceptance

For behavior-preserving changes, require unchanged logical outputs, scale bits,
KV state, and fault behavior on the defined workloads. Scheduling and storage
changes must remain byte-identical against the matched RTL baseline.
RTL-versus-golden conformance follows the frozen per-operation tolerance and
special-value policy, including existing FP16 ULP bands; it is not blanket
byte equality. At full-model non-finite boundaries, retain RTL-to-RTL checks
while separately enforcing the numerical contract from M0; matching hashes
alone does not demonstrate model quality. Layout optimizations may change
bundle bytes but must preserve relocated behavior.

An optimization is accepted when it improves its declared primary metric and
introduces no unexplained regression in required workloads. Any intentional
area/latency/energy tradeoff must be documented before promotion. Require zero
lost accesses and forbidden collisions in valid overlap tests, including cases
that actually produce simultaneous DMA and systolic activity. Keep the
serialized reference and DPI models available for diagnosis.

For timing work, compare `cycles / achieved_frequency`, not frequency alone.
Repeat with fixed external bandwidth assumptions. Compare each candidate with
its immediate parent and rerun the combined configuration; individual gains
must not be multiplied into a projected headline.

Publish one reproducible milestone report with commands, manifests, raw
counters, correctness results, resource use, and remaining limitations. Update
[project status](project_status.md) only with verified results. Missing large
fixtures or physical collateral limits the scope of completion; it does not
turn an unexecuted gate into a pass.

## First implementation batch

1. Correct and assert execution parameters and program-level runner selection.
2. Add failing protocol/overflow tests, then fix DMA boundary handling.
3. Add target/IR validation and repair the required regression profile.
4. Capture the corrected mode-1 B1/B16 baseline and buffer/counter reports.
5. Implement real-row FC2 loads and the one-beat instruction buffer as separate
   changes; measure each, then their combination.

Start SRAM feasibility research alongside steps 1–4. Begin QKV and FC2 reuse
only after their storage and scale-lifetime plans are explicit.
