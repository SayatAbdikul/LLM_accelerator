# TACCEL code exploration roadmap

**Revision:** 2026-10-11  
**Code basis:** current working tree; optimization candidates remain opt-in  
**Purpose:** learn the executable ISA, RTL microarchitecture, software-to-hardware
contract, and verification hierarchy in an order that keeps the pieces connected.

This guide is both an architecture tour and a practical reading plan. It does
not replace the bit-accurate [ISA specification](../software/docs/isa_spec.md)
or the command-oriented [RTL testbench guide](../rtl/TESTBENCHES.md). Use it to
decide what to read, what question each file answers, and what evidence to
produce before moving to the next layer.

## Outcome

After completing the roadmap, you should be able to:

1. encode an instruction by hand and identify every architectural field;
2. trace that word from Python, through DRAM fetch and decode, to an engine;
3. explain where tensors live and how their data type changes across an LLM
   sublayer;
4. reason about DMA/systolic overlap and every shared SRAM port;
5. distinguish ISA legality, software modeling, RTL support, and synthesis
   readiness;
6. select the smallest meaningful test for a change, then escalate to
   end-to-end co-simulation and performance measurement;
7. state exactly what a passing test, elaboration check, or benchmark proves.

## Architecture at a glance

```text
checkpoint / state_dict
        |
        v
frontend -> IRGraph -> tiling/allocation -> W8A16 lowering
                                            |
                                            v
                         prefill + decode ProgramBundle
                              |                 |
                              |                 +-> HostRunner patches addresses,
                              |                     embeddings, KV bases, attention
                              v
                   sequential golden Simulator
                              |
                              +------------------------+
                                                       | compare traces/results
ProgramBinary in modeled DRAM                          |
        |                                              |
        v                                              |
AXI read arbiter -> fetch -> decode -> in-order control|
        ^                            |                 |
        |             +--------------+---------------+ |
        |             |              |               | |
        +-----------> DMA         systolic          SFU/helper
                      |              |               |
                      +--------------+---------------+
                                     |
                         ABUF / WBUF / ACCUM SRAM
```

There are four contracts to keep separate:

- **ISA contract:** the bytes, state, legality, side effects, barriers, and
  faults visible to a program.
- **Microarchitecture:** how fetch, control, execution engines, ports, and
  pipelines realize the ISA cycle by cycle.
- **Reference contract:** how the sequential Python simulator defines expected
  results. It models architectural behavior, not RTL timing.
- **Verification contract:** which comparison is unit-level, byte-exact,
  numerically banded, structural, performance-only, or environment-dependent.

## Part I: ISA architecture

### Instruction and addressing model

Every instruction is one 64-bit word serialized as eight big-endian bytes.
The opcode is always bits `[63:59]`. The program counter is an instruction
index, so instruction byte address is `PC * 8`. The 128-bit AXI fetch bus
returns two adjacent instructions per beat; the fetch unit selects the half and
byte-swaps it into the architectural word.

All on-chip memory offsets and DMA lengths count 16-byte SRAM rows. A DRAM
address is:

```text
address_register[0..3] + 16 * instruction.dram_off
```

Each 56-bit address register is written in two 28-bit halves by adjacent
`SET_ADDR_LO` and `SET_ADDR_HI` instructions. Relocation code recognizes and
patches these pairs.

### Architectural state

| State | Shape | Meaning |
|---|---:|---|
| Address registers | 4 × 56 bits | Base DRAM byte addresses |
| Scale registers | 16 × FP16 bits | Quantization/dequantization parameters |
| Tile configuration | `M`, `N`, `K`, `m_exact` | Padded matmul shape plus exact SFU row count |
| Attention configuration | row base, valid KV length, mode | Persistent causal/valid-length mask context |
| ABUF | 8,192 × 16-byte rows | 128 KiB activation storage |
| WBUF | 16,384 × 16-byte rows | 256 KiB weights and parameters |
| ACCUM | 4,096 × 16-byte rows | 64 KiB, four little-endian INT32 lanes per row |
| PC/status | 56-bit PC, done/fault | In-order execution position and terminal state |

Buffer ID `0b11` is reserved. Bounds are checked in software before encoding
and again in the RTL memory path.

### Seven encoding formats

| Format | Main users | Architectural idea |
|---|---|---|
| R | MATMUL, helpers, generation-2 SFU | Three buffer operands, offsets, scale register, one flag |
| M | LOAD, STORE | SRAM range plus address register and DRAM offset; optional transposed LOAD |
| B | BUF_COPY | Local copy/transpose geometry |
| A | SET_ADDR_LO/HI | One 28-bit half of a base address |
| C | CONFIG_TILE | Zero-based 16-element tile counts and optional exact row count |
| ATTN | CONFIG_ATTN | Query origin, valid KV length, mask mode |
| S | NOP, HALT, SYNC, SET_SCALE | System payloads and immediate scale writes |

Read the exact bit layouts in
[`software/docs/isa_spec.md`](../software/docs/isa_spec.md#instruction-formats),
then compare the same fields in
[`opcodes.py`](../software/taccel/isa/opcodes.py),
[`instructions.py`](../software/taccel/isa/instructions.py),
[`encoding.py`](../software/taccel/isa/encoding.py),
[`taccel_pkg.sv`](../rtl/common/src/include/taccel_pkg.sv), and
[`decode_unit.sv`](../rtl/common/src/decode_unit.sv).

### Opcode families and scheduling

- System/configuration instructions update state or terminate immediately.
- DMA `LOAD`/`STORE` and systolic `MATMUL` are asynchronous. Control emits a
  one-cycle dispatch and may continue fetching instructions.
- `SYNC` waits on a selected mask: bit 0 DMA, bit 1 systolic, bit 2 SFU.
- DMA and systolic may overlap. This is the weight-prefetch concurrency path.
- The SFU is asynchronous, but current control serializes its dispatch against
  DMA, systolic, and helper activity.
- The helper engine is architecturally blocking. Control waits for completion,
  so it needs no SYNC bit.

The retained integer/helper instructions are `REQUANT` (`0x0B`), `SCALE_MUL`
(`0x0C`), `VADD` (`0x0D`), `REQUANT_PC` (`0x11`), and `DEQUANT_ADD` (`0x13`).
Generation-2 W8A16 work uses `0x17..0x1F`, excluding reserved `0x1C`.
The six generation-1 SFU opcodes `0x0E`, `0x0F`, `0x10`, `0x12`, `0x15`, and
`0x16` retain names for diagnostics but are illegal and have no executable
software or RTL path.

### Numeric path

The maintained decoder data path is W8A16:

```text
FP16 activation in ABUF
    -> QUANT_FP32_INT8
INT8 activation in ABUF + INT8 weight in WBUF
    -> 16x16 MATMUL
INT32 result in ACCUM
    -> DEQUANT_ACCUM_FP32[_SCALED]
FP16 activation in ABUF, with FP32 internal arithmetic
    -> VADD / LAYERNORM / GELU / MASKED_SOFTMAX
FP16 activation in ABUF
```

`CONFIG_TILE` encodes padded tile counts as zero-based values: each physical
dimension is `(field + 1) * 16`. `m_exact` lets SFU row loops ignore padded
rows without changing the matmul/helper walk. `MATMUL.flags=0` starts from a
cleared destination; `flags=1` read-modify-writes existing ACCUM values.
Generation-2 operations use `flags=1` for FP16 boundary storage; the Python
model accepts `flags=0`, but RTL reports unsupported.

### Attention model

`CONFIG_ATTN` persists until reset or replacement. Its mode combines two
predicates over key column `c` and query row `r`:

- causal: `c <= query_row_base + r`;
- valid length: `c < valid_kv_len`.

The causal RTL path uses `MASKED_SOFTMAX_FP32` (`0x1D`). Noncausal
`SOFTMAX_FP32` (`0x1C`) exists in Python/golden code but is deliberately
illegal in RTL because no supported causal frontend emits it.

### Fault model and intentional asymmetries

RTL faults are terminal until reset: illegal opcode `1`, DRAM OOB `2`, SRAM
OOB `3`, missing/inconsistent configuration `4`, reserved buffer `5`, and
legal-but-unimplemented form `6`. Fault sources from fetch, DMA, helper, SFU,
SRAM plumbing, and control merge at the top level and are also exposed to the
program runner for diagnosis.

Do not infer hardware support from a Python enum or successful golden run:

- `SOFTMAX_FP32` `0x1C` is Python/golden-only and illegal in RTL.
- `CONFIG_TILE.weight_int4` is encoded and modeled in Python, but RTL ignores
  bit 28. W4 is not a hardware deployment mode.
- `SET_SCALE.src_mode=1..3` is modeled in Python; RTL accepts immediate mode 0
  only.
- Generation-2 `flags=0` storage is modeled in Python; RTL supports W8A16
  `flags=1` only.

The compiler now enforces these boundaries for `target="rtl-gen2"`, the
decoder-bundle default. Explicit `research-golden` compilation permits
software-only experiments; it is not evidence of hardware support. Read
[`validation.py`](../software/taccel/compiler/validation.py) alongside the
ISA decoder: it also rejects malformed graphs and missing/incompatible data
before allocation.

## Part II: RTL architecture

### Compilation boundary and top level

[`core.f`](../rtl/common/filelists/core.f) is the authoritative ordered list of
shared RTL compilation units. It starts with the package and FP32 primitives,
then brings in fetch/decode/control, execution engines, registers/SRAM,
systolic blocks, DMA, and finally
[`taccel_top.sv`](../rtl/common/src/taccel_top.sv).

The top level is the integration contract. It:

- arbitrates the external AXI read channel between instruction fetch and DMA;
- registers returned instruction words before combinational decode;
- connects control dispatch pulses to DMA, systolic, helper, and SFU engines;
- arbitrates shared SRAM ports and gives the systolic dedicated paths;
- merges sticky and immediate faults;
- exposes retire, stall, overlap, collision, traffic, and engine-cycle counters
  used by the Verilator program runner.

Start architectural changes at the leaf module, but always finish review at
the top level: that is where a correct engine can still be miswired.

### Fetch, decode, and control

[`fetch_unit.sv`](../rtl/common/src/fetch_unit.sv) is a single-outstanding
`IDLE -> AR_REQ -> WAIT_R` machine. The top-level arbiter gives DMA default
priority, then lets a pending fetch slip in after a completed DMA burst. It
latches ownership at AR acceptance and routes the returning R beat by that
owner.

`FETCH_BUFFER_ENABLE=1` retains a single 128-bit beat so a matching next
instruction can use its other half without another AXI request. Requests
latch their address and half selection; invalidation clears retained data and
poisons an outstanding refill. The top level invalidates on a new run and
during DMA stores. This remains an explicit candidate: the shared core and
FPGA wrapper default to buffer disabled. `run_program_synth` opts in;
`run_program_synth_nofetch` supplies the matched hardware baseline.

[`decode_unit.sv`](../rtl/common/src/decode_unit.sv) is purely combinational.
It extracts all formats in parallel and produces a single `illegal` signal for
retired/reserved opcodes, illegal buffer IDs, and reserved ATTN bits.

[`control_unit.sv`](../rtl/common/src/control_unit.sv) is the in-order issue
machine:

```text
IDLE -> FETCH -> ISSUE -> FETCH ...
                   |       |
                   |       +-> SYNC_WAIT -> FETCH
                   +----------> DISP_WAIT -> FETCH   (blocking helper)
                   +----------> HALT or FAULT        (terminal)
```

It performs configuration checks, distinguishes illegal from unsupported
forms, retries a dispatch when the target engine is busy, and retires only
after the required architectural side effect has occurred.

[`register_file.sv`](../rtl/common/src/memory/register_file.sv) holds the four
address registers, sixteen scale registers, tile state, exact row count, and
attention state. These are configuration state, not a scalar general-purpose
register file.

### SRAM subsystem and port ownership

[`sram_subsystem.sv`](../rtl/common/src/memory/sram_subsystem.sv) instantiates
three logical dual-port memories. Reads return one cycle after request.

| Logical port | Owner/path | Use |
|---|---|---|
| A, read/write | fixed priority helper > SFU > DMA | General engine and DMA traffic |
| S, read/write | systolic only | Source-2/ACCUM read-modify-write and result drains |
| B, read-only | helper > SFU > systolic | Second source/activation reads |
| W, read-only | systolic, hardwired to WBUF Port B | Stream current weights while DMA prefetches next WBUF region |

Port A and Port S are fanned out per physical buffer. They can operate in the
same cycle when they target different SRAMs, such as DMA writing WBUF while
systolic drains ACCUM. If both target the same buffer, A wins and
`s_collision` forces an architectural fault; an access is never silently
dropped.

[`sram_dp.sv`](../rtl/common/src/memory/sram_dp.sv) is the simulation memory
boundary. [`sram_dp_inferred.sv`](../rtl/common/src/memory/sram_dp_inferred.sv)
supports inferred-memory/elaboration flows. Platform wrappers must eventually
replace these boundaries with real macro or FPGA memory mappings.

### DMA engine

[`dma_engine.sv`](../rtl/common/src/dma_engine.sv) latches a complete transfer,
checks its DRAM/SRAM extent, then walks AXI bursts.

- Normal LOAD: `AR -> R`, writing each accepted 128-bit beat into SRAM.
- STORE: `AW -> synchronous SRAM pre-read -> W -> B` for every burst.
- Transposed LOAD: read one 16-row stripe into a 1 KiB scratch array, then
  write gathered columns to strided SRAM rows.
- Burst-boundary addresses and lengths are precomputed in registers to keep
  wide add/compare logic off the transition critical path.
- LOAD, STORE and transposed-LOAD bursts are bounded by the remaining
  transfer, the AXI burst limit and the next 4 KiB page boundary.
- `dma_rd_busy` is burst-scoped, allowing instruction fetch between DMA read
  bursts.

The engine is asynchronous from control and terminally faults on bad AXI
responses, malformed `last`, or memory bounds violations.

### Systolic engine

The compute hierarchy is
[`systolic_controller.sv`](../rtl/common/src/systolic/systolic_controller.sv)
-> [`systolic_array.sv`](../rtl/common/src/systolic/systolic_array.sv) ->
[`systolic_pe.sv`](../rtl/common/src/systolic/systolic_pe.sv).

The controller walks the configured `M x N x K` grid of 16x16 tiles. It loads
an A tile into double-buffered scratch storage, streams activation/weight lanes
through the default chained array, injects flush zeros, and drains each 16x16
INT32 tile as 64 SRAM rows. It can prefetch the next A tile while the current
K tile streams. WBUF weights use dedicated Port W; other source-2 traffic and
ACCUM writeback use Port S. `flags=1` adds a per-drain ACCUM read before the
write.

The array owns arithmetic and data propagation; the controller owns tiling,
SRAM timing, clear/accumulate semantics, and the dispatch/busy contract.

### Blocking helper engine

[`blocking_helper_engine.sv`](../rtl/common/src/blocking_helper_engine.sv) is a
microcoded, correctness-oriented FSM. State families implement flat or
transposed BUF_COPY, INT8 residual add, INT32 bias add, scalar/per-channel
requantization, scale multiplication, and dequantized residual addition.
Synchronous SRAM reads are split into request/latch/write states.

`HELPER_SYNTH_MODE=0` retains the DPI-backed reference path where needed;
mode 1 selects the synthesizable DEQUANT_ADD primitive chain. Control prevents
the helper from sharing its fixed-priority ports with active compute work.

### Generation-2 SFU

[`sfu_engine.sv`](../rtl/common/src/sfu_engine.sv) implements the W8A16
generation-2 operations. FP16 values occupy the SRAM boundary; the engine
promotes them to FP32 for vector arithmetic, reductions, normalization,
activation functions, and softmax.

Its large FSM has three conceptual layers:

1. SRAM request/latch/pack/write loops for FP16, INT8, and INT32 layouts;
2. row or element iterators for VADD, dequant, quant, LayerNorm, GELU, causal
   softmax, and max-absolute reduction;
3. latency-matching wait/sample states for pipelined FP32 primitives.

The synthesizable path uses primitives under
[`fp32/`](../rtl/common/src/fp32/fp32_add.sv), including the integrated
`fp32_div_p6`, `fp32_sqrt_p6`, `fp32_exp_p18`, and `fp32_gelu_p33` pipelines.
`SFU_SYNTH_MODE=0` is the DPI/reference comparison mode; hardware cycle
measurements require both `SFU_SYNTH_MODE=1` and `HELPER_SYNTH_MODE=1`.
These are RTL parameters, not preprocessor defines. `SFU_SYNTH_NO_DPI` is the
separate preprocessing guard that removes DPI imports in synthesis flows.

### Platform and synthesis boundaries

- Native Verilator simulation is the primary functional RTL environment.
- `make -C rtl/verilator synth-check` runs `sv2v` plus fatal Yosys hierarchy
  resolution, diagnostic pre-process `check`, and statistics for the complete
  shared core. Unused inlined-function temporaries can trigger diagnostic
  undriven warnings; this is not a lowered/mapped `check -assert` proof and
  does not map cells or establish timing.
- The [ASIC flow](../rtl/asic/README.md) provides SKY130-oriented wrappers and
  research scripts, but full-chip SRAM composition, IO, PNR, DRC, and LVS are
  not complete.
- The [FPGA flow](../rtl/fpga/README.md) is a structural wrapper smoke gate.
  There is no selected board, device, memory controller, clocking IP, or final
  constraint set.
- The historical 34.41 MHz value is not a current full-chip sign-off result.
  See [current project status](project_status.md) before translating cycles to
  throughput.

## Part III: Software and reference architecture

The shortest reliable route through the Python stack is:

1. [`opcodes.py`](../software/taccel/isa/opcodes.py),
   [`instructions.py`](../software/taccel/isa/instructions.py), and
   [`encoding.py`](../software/taccel/isa/encoding.py) define words and
   validation.
2. [`assembler.py`](../software/taccel/assembler/assembler.py) and
   [`disassembler.py`](../software/taccel/assembler/disassembler.py) provide
   text/binary round trips and `ProgramBinary`/`ProgramBundle` containers.
3. [`ir.py`](../software/taccel/compiler/ir.py),
   [`tiler.py`](../software/taccel/compiler/tiler.py), and
   [`memory_alloc.py`](../software/taccel/compiler/memory_alloc.py) define the
   model-independent graph and placement constraints.
   [`validation.py`](../software/taccel/compiler/validation.py) checks graph
   inputs and hardware legality; [`options.py`](../software/taccel/compiler/options.py)
   selects explicit targets and independently measurable candidates.
4. [`codegen.py`](../software/taccel/compiler/codegen.py) coordinates emission;
   [`matmul.py`](../software/taccel/compiler/w8a16_emit/matmul.py),
   [`attention.py`](../software/taccel/compiler/w8a16_emit/attention.py),
   [`packed_attn.py`](../software/taccel/compiler/w8a16_emit/packed_attn.py),
   and [`sublayer.py`](../software/taccel/compiler/w8a16_emit/sublayer.py)
   implement the active numeric lowering.
   [`quantization.py`](../software/taccel/compiler/w8a16_emit/quantization.py)
   contains bounded QKV/FC2 quantization reuse.
5. [`decoder_bundle.py`](../software/taccel/compiler/decoder_bundle.py) joins
   prefill/decode streams, shared data, symbols, relocations, logits, and KV
   regions. The baseline builds and checks both data layouts;
   `shared_data_layout=True` reuses the first layout, while
   `deduplicate_data=True` omits unused weight/scale/bias storage.
6. [`state.py`](../software/taccel/golden_model/state.py),
   [`memory.py`](../software/taccel/golden_model/memory.py),
   [`dma.py`](../software/taccel/golden_model/dma.py),
   [`systolic.py`](../software/taccel/golden_model/systolic.py), and
   [`simulator.py`](../software/taccel/golden_model/simulator.py) execute the
   sequential reference contract. `SYNC` is a no-op here because execution is
   already serialized.
7. [`host_runner.py`](../software/taccel/runtime/host_runner.py) patches
   embeddings, addresses, KV bases, and attention context for prefill and
   decode. `BundlePatcher` can patch without allocating a simulator.
   Generation omits the unused final decode by default; use
   `final_token_commit=True` when the final sampled token's KV row and
   next-token logits must be committed for continuation.
8. [`rtl_cosim.py`](../software/tools/rtl_cosim.py) serializes the same bundle
   for the native RTL runner and compares architectural captures.

The broader package map is in the
[software codebase guide](../software/CODEBASE.md).
All six `CompilerOptions` default to `False`: indexed liveness, real-row FC2
loads, QKV/FC2 quantization reuse, shared layout and data deduplication.
Compare `compilation_metrics()` before measuring cycles. Decoder bundles
default to `rtl-gen2`, but the low-level `CodeGenerator` defaults to
`research-golden`; select its target explicitly for hardware work.

## Part IV: Verification architecture

Verification is layered. A higher layer is not a substitute for a lower one,
and a green elaboration run is not evidence of numerical correctness.

| Layer | Main evidence | What it proves |
|---|---|---|
| Python unit | Mandatory core profile plus optional model/research modules | Codec, compiler, runtime, golden, quantizer, and policy behavior |
| RTL unit/subsystem | Native C++ Verilator benches | Cycle-level handshakes, state machines, faults, and datapath behavior |
| Primitive numeric | FP32 primitive benches | Latency and numerical agreement of individual arithmetic blocks |
| Program co-sim | Python driver + mode-1 `run_program_synth` | Cross-language execution and architectural memory/trace agreement |
| Frozen generation-2 | Simulator pin + 73-file dependency/fixture manifest | Detects unreviewed reference drift and checks the chip SFU path |
| Batched decoder | tiny B=16/B=32 tests | Multi-row tiling, packed attention, and multi-M-tile execution |
| Performance | program runner counters and hashes | Cycles, overlap, stalls, traffic, and output stability under a named DRAM model |
| Hierarchy elaboration | `sv2v + yosys` hierarchy/check/stat | Complete hierarchy resolves; pre-process `check` warnings are diagnostic, not an asserted mapped-design proof |
| Physical research | SKY130 block scripts | Local implementation evidence only; not full-chip sign-off |

### Python gates

Start with codec and architectural semantics:

```sh
PYTHONPATH=software .venv/bin/python -m pytest -q \
  software/tests/test_isa_encoding.py \
  software/tests/test_assembler.py \
  software/tests/test_w8a16_isa.py \
  software/tests/test_golden_model.py \
  software/tests/test_w8a16_simulator.py
```

Then test construction and runtime mutation:

```sh
PYTHONPATH=software .venv/bin/python -m pytest -q \
  software/tests/test_decoder_bundle.py \
  software/tests/test_program_bundle.py \
  software/tests/test_relocation_patching.py \
  software/tests/test_runtime_patch_sites.py \
  software/tests/test_w8a16_codegen.py
```

Then run the mandatory core, including mode-1 program conformance:

```sh
make -C rtl/verilator run_program_synth
PYTHONPATH=software .venv/bin/python software/tools/train_tiny_fixture.py
PYTHONPATH=software .venv/bin/python software/tools/check_reference_manifest.py
PYTHONPATH=software .venv/bin/python -m pytest -q software/tests --verification-profile required
```

This profile converts skipped mandatory tests into failures. The broader
developer suite is `PYTHONPATH=software .venv/bin/python -m pytest -q software/tests`.
Some tests require generated checkpoints, tokenizer data, external replay
vectors, `PYTEST_SLOW=1`, or `PYTEST_124M=1`. Stage-5 calibration monkeypatches
have been repaired; do not dismiss new failures as that old issue. Report
optional skips and missing inputs separately from passes.

### Native RTL gates

```sh
make -C rtl/verilator all
make -C rtl/verilator test_sfu_synth test_helpers_synth
make -C rtl/verilator test_sfu_helper_synth
make -C rtl/verilator test_fetch_no_buffer test_fetch_buffer test_dma_buffer
make -C rtl/verilator test_systolic_array_chained test_systolic_qkt
make -C rtl/verilator test_fp32_div_p6 test_fp32_sqrt_p6
make -C rtl/verilator test_fp32_exp_p18 test_fp32_gelu_p33
```

`all` means decode, control, DMA, helper, SFU-reference, and systolic. It does
not include mode-1, QKT, chained-array, primitive, or program-level gates.
Replay/padded QKT cases skip without `RTL_QKT_REPLAY_DIR`.

Read the C++ benches in the same order as the engines:
[`test_control.cpp`](../rtl/verilator/test_control.cpp),
[`test_dma.cpp`](../rtl/verilator/test_dma.cpp),
[`test_systolic.cpp`](../rtl/verilator/test_systolic.cpp),
[`test_helpers.cpp`](../rtl/verilator/test_helpers.cpp), and
[`test_sfu.cpp`](../rtl/verilator/test_sfu.cpp). Shared clock, AXI, and memory
behavior lives in [`testbench.h`](../rtl/verilator/include/testbench.h).

### Program-level co-simulation

```sh
make -C rtl/verilator run_program_synth run_program_synth_nofetch
PYTHONPATH=software .venv/bin/python -m pytest -q \
  software/tests/test_compare_rtl_golden.py \
  software/tests/test_batched_decode.py
```

[`run_program.cpp`](../rtl/verilator/run_program.cpp) loads a `ProgramBinary`
into modeled DRAM, clocks the full top level, emits snapshots, and reports
retire/fault/performance counters. The generation-2 conformance test first
checks the frozen golden simulator blob
`cc1bc64f34bf7b5a53a5760bcd500dca10cb8080` and
[`gen2_reference_manifest.json`](../software/tests/fixtures/gen2_reference_manifest.json),
then deterministic compilation and RTL-versus-golden captures. The manifest
also pins executable golden/ISA/assembler/utility dependencies, the fixture
generator, and raw fixture/metadata files. Ten checked-in fixture cases under
[`fixtures/gen2`](../rtl/verilator/fixtures/gen2) cover the mode-1 SFU path.

`rtl_cosim.py` defaults to `--runner synth`; `--runner reference` explicitly
selects the separately built DPI runner. Hardware gates check actual SFU and
helper mode fields, binary freshness and execution/counter invariants.
Tiny conformance retains per-op bands: GELU allows 3 ULP, other FP16 operations
0 ULP, and differing NaN/Inf encodings cannot pass as close. The optional
GPT-2 124M gate records first divergence and logits metrics, but nonfinite
outputs fail numerical-quality eligibility. No full-model quality follows
from a clean halt, matching NaNs, or passing tiny tests.

### Performance and structural gates

```sh
make -C rtl/verilator run_program_synth
PYTHONPATH=software .venv/bin/python software/tools/bench_decode_cycles.py \
  --positions 0,63,255,511 --batch 1
PYTHONPATH=software .venv/bin/python software/tools/fast_gate_b16.py \
  --batch 16 --position 510
make -C rtl/verilator synth-check
```

A reproducible performance result names the commit, bundle/preset, batch,
position, cycles, cycles/token, actual SFU/helper/fetch modes, checkpoint and
calibration hashes, memory pacing, and overlap/collision/phase/traffic counters.
Read actual FP16 logits from their DRAM region, excluding vocabulary padding;
the runner's `summary.logits` is only a legacy INT32 ACCUM diagnostic.
Matched performance promotion requires finite, identical logical outputs.
Standalone zero-KV decode shapes measure cycles, not model quality.

`--beat-interval` paces reads only; writes are unthrottled by that option, so
this is not a shared fixed-total-bandwidth DRAM model. Throughput is omitted
unless `--fmax-mhz` supplies an explicitly assumed clock. Use
[`optimization_matrix.py`](../software/tools/optimization_matrix.py) to retain
baseline/candidate programs, raw results and contiguous KV-state comparisons.

## Part V: Ordered exploration plan

Treat each stage as a checkpoint. Do not continue merely because the files
were read; produce the listed artifact or be able to answer the exit question.

### Stage 0: establish the baseline

**Goal:** know the repository state and distinguish current docs from dated
experiment records.

**Read:** [README](../README.md), [project status](project_status.md), this
roadmap, and [`core.f`](../rtl/common/filelists/core.f).

**Run:** Python test collection, one ISA smoke group, `make -C rtl/verilator
test_decode`, and `git status --short`. Record unavailable tools or fixtures.

**Checkpoint:** a one-page baseline containing commit/worktree state, tool
versions, collected tests, executed tests, skips, known failures, and hardware
mode. Never erase or misattribute a dirty worktree.

### Stage 1: follow one instruction word

**Goal:** make encoding/decoding concrete before studying engines.

**Read:** Python ISA files, assembler/disassembler, `taccel_pkg.sv`, and
`decode_unit.sv`.

**Exercise:** choose `MATMUL`, fill its R-format fields by hand, encode it in
Python, print the eight bytes, disassemble it, and verify every RTL slice.
Repeat with an illegal buffer ID and reserved `0x1C`.

**Checkpoint:** a bit-field worksheet showing word, bytes, opcode, buffers,
offsets, scale/flag, and expected legality on Python and RTL.

### Stage 2: map state, layouts, and faults

**Goal:** understand what instructions mutate and which units own data.

**Read:** `register_file.sv`, `sram_subsystem.sv`, golden `state.py` and
`memory.py`, plus ISA sections on storage, formats, and faults.

**Exercise:** draw ABUF/WBUF/ACCUM rows for one 16x16 INT8 tile, one FP16 row,
and one 16x16 INT32 result. Translate each into instruction offsets and bytes.

**Checkpoint:** explain why one INT8 row holds 16 elements, one FP16 row holds
8, one ACCUM row holds 4, and why an offset bug can be type-dependent.

### Stage 3: trace fetch, issue, barriers, and failure

**Goal:** understand architectural sequencing and timing ownership.

**Read:** `fetch_unit.sv`, `control_unit.sv`, the AXI arbiter and fault merge in
`taccel_top.sv`, then `test_control.cpp`.

**Exercise:** trace `CONFIG_TILE -> LOAD -> MATMUL -> SYNC -> HALT` cycle by
cycle. Mark dispatch, retire, busy, PC update, and which SYNC mask bit matters.
Trace a missing-config and unsupported-mode case into terminal FAULT.

**Checkpoint:** explain why dispatch, completion, and retirement are different
events, and why a busy target causes retry rather than instruction loss.

### Stage 4: understand movement and SRAM concurrency

**Goal:** follow bytes from external DRAM into engine-visible SRAM rows.

**Read:** `dma_engine.sv`, `sram_subsystem.sv`, top-level A/S/B/W arbitration,
golden `dma.py`, and `test_dma.cpp`.

**Exercise:** trace a two-burst LOAD and a STORE including synchronous SRAM
pre-read. Then trace one transposed LOAD stripe. Finally map concurrent DMA to
WBUF and systolic drain to ACCUM through different physical memories.

**Checkpoint:** distinguish AXI read arbitration from SRAM port arbitration,
and explain why same-buffer A/S conflict faults while different-buffer access
proceeds concurrently.

### Stage 5: understand the tiled systolic dataflow

**Goal:** connect ISA tile geometry to PE-level arithmetic and ACCUM layout.

**Read:** systolic controller, array, PE, golden `systolic.py`, and systolic
benches.

**Exercise:** use a single 16x16x16 tile. Label A rows, W rows, chained data
motion, flush cycles, accumulator cells, and the 64-row drain. Expand to two K
tiles and compare `flags=0` versus `flags=1`.

**Checkpoint:** derive the destination row for an `(M-tile, N-tile, drain
row/group)` tuple and identify which controller states absorb SRAM latency.

### Stage 6: understand helpers and generation-2 numerics

**Goal:** learn layout conversion and numerical boundaries around matmul.

**Read:** helper and SFU engines, `sfu_synth_datapath.svh`, selected FP32
primitives, ISA compute contracts, and their reference/mode-1 benches.

**Exercise:** trace one ACCUM result through scaled dequant, residual add,
LayerNorm or GELU, and requantization. Mark every INT8, INT32, FP16-storage,
and FP32-internal boundary. For one pipelined primitive, count request-to-use
latency against the SFU wait states.

**Checkpoint:** explain why reference-mode parity alone does not prove the
synthesizable chip path and why mode-1 gates are separate from `make all`.

### Stage 7: trace compiler to bundle

**Goal:** see how an LLM graph becomes two relocatable instruction streams.

**Read:** IR/model config, tiler, allocator, W8A16 matmul/attention/sublayer
emitters, `validation.py`, `options.py`, quantization reuse, `codegen.py`, and
`decoder_bundle.py`.

**Exercise:** take one IR matmul or attention node and list its emitted address
setup, LOADs, tile configuration, MATMULs, barriers, epilogue, trace events,
and data symbols. Locate its prefill/decode relocation and runtime patch sites.

**Checkpoint:** explain why prefill and decode share data but need different
instruction streams, and how KV/logits/temp regions avoid overlap.
Compare one opt-in candidate against the all-false baseline; record static
instructions/DMA bytes/storage separately from actual RTL cycle measurements.

### Stage 8: compare golden semantics with RTL timing

**Goal:** separate architectural correctness from scheduling implementation.

**Read:** golden state/memory/DMA/systolic/simulator, `HostRunner`,
`rtl_cosim.py`, and `run_program.cpp`.

**Exercise:** execute the Stage 7 stream in the golden model, capture selected
buffers, then run the serialized stream in RTL and compare the same points.
Note that golden `SYNC` is a no-op while RTL may spend many cycles waiting.

**Checkpoint:** identify the first authoritative divergence location: encoded
word, architectural state before instruction, captured output bytes, or an
allowed numeric band.

### Stage 9: learn the verification ladder

**Goal:** choose tests by failure locality and claimed assurance.

**Read:** Python tests next to their implementation, native C++ benches,
`rtl/TESTBENCHES.md`, and the generation-2 freeze record.

**Exercise:** inject a temporary local defect at one layer and confirm the
nearest gate fails; revert the defect afterward. Then run the next higher
integration gate. This negative control catches stale binaries and vacuous
tests.

**Checkpoint:** for each green command, write one sentence beginning “This
proves...” and another beginning “This does not prove...”.

### Stage 10: profile performance without weakening correctness

**Goal:** attribute cycles to architecture rather than merely recording a
total.

**Read:** `bench_decode_cycles.py`, `fast_gate_b16.py`,
`profile_decode_step.py`, `audit_porta_argmax.py`, runner counters, and the
current performance section in project status.

**Exercise:** compare B=1 across context positions and one B=16 step. Record
fetch stalls, DMA/systolic co-busy cycles, SYNC waits, Port-A/S collisions,
traffic, step cycles, and logits hash. Use `run_program_noovl` for an overlap
A/B comparison when changing scheduling.

**Checkpoint:** explain whether a cycle change comes from useful overlap,
less work, a different memory model, a correctness regression, or measurement
noise.

### Stage 11: understand implementation readiness

**Goal:** know what is synthesizable and what still blocks deployment.

**Read:** generic synthesis baseline, ASIC/FPGA guides, memory wrappers, and
current open decisions.

**Exercise:** run `synth-check`, inspect hierarchy/stat output, and locate the
platform-specific memory boundary. If a target PDK/toolchain exists, run only
the documented applicable flow and record its exact scope.

**Checkpoint:** distinguish RTL elaboration, logic synthesis, STA, placed and
routed timing, DRC/LVS, FPGA implementation, and board validation. Do not call
one of these by another name.

## Golden thread: one program fragment end to end

Use this sequence as the recurring example for Stages 1–10:

```text
SET_ADDR_LO/HI R0, activation DRAM base
SET_ADDR_LO/HI R1, weight DRAM base
LOAD  R0 -> ABUF
LOAD  R1 -> WBUF
CONFIG_TILE M, N, K, m_exact
MATMUL ABUF, WBUF -> ACCUM, flags=0
SYNC  systolic
DEQUANT_ACCUM_FP32_SCALED ACCUM, WBUF params -> ABUF, flags=1
MAX_ABS_REDUCE_FP32 ABUF -> scale pair
QUANT_FP32_INT8 ABUF -> ABUF, flags=1
HALT
```

The compiler may schedule an additional WBUF LOAD after MATMUL dispatch so DMA
prefetch overlaps systolic work. Add `SYNC DMA|systolic` before consuming both
results. For every instruction, record:

- encoded word and byte order;
- architectural preconditions and side effects;
- control dispatch/retire behavior;
- engine FSM and SRAM/AXI ports used;
- golden-model function;
- closest unit test and program-level capture;
- relevant performance counters.

## Change-impact matrix

| Change | Inspect/update together | Minimum verification |
|---|---|---|
| Opcode or field layout | Python opcode/instruction/encoding, syntax/disassembler, package/decode/control, ISA spec | ISA round trip, decode, control, golden, fixture/conformance gates |
| Legality/support boundary | Python decode/classes, RTL decode/control, golden dispatcher, compiler emitters | Negative illegal/unsupported tests plus supported program co-sim |
| SRAM layout/capacity | Python constants/memory/allocator, RTL package/SRAM, all engine addressing | Bounds tests, affected engine benches, bundle layout, co-sim |
| DMA protocol/transpose | DMA RTL and golden DMA, top AXI arbitration, compiler data layout | DMA unit, transpose fixtures, program co-sim, fetch-stall counters |
| Systolic tiling/dataflow | Controller/array/PE, golden systolic, matmul emitter | Array/controller benches, QKT, batched decode, output hash |
| Helper/SFU numeric path | ISA contract, engine/datapath primitive, golden simulator, emitter | Reference and mode-1 unit tests, primitive test, gen-2 fixtures, co-sim |
| Scheduling/overlap | Control, top-level arbitration, SRAM ports, emit schedule | Control/unit tests, no-overlap A/B, Port-A audit, hashes and counters |
| Bundle/relocation/runtime patch | Assembler container, decoder bundle, HostRunner, co-sim serializer | Bundle/relocation/patch tests, prefill and decode co-sim |
| Performance optimization | Functional owner plus counters and benchmark tool | Correctness gates first; direct cycles, hashes, memory model, overlap audit |
| Platform wrapper/memory binding | shared filelist, inferred/macro memory, wrapper, constraints | Full hierarchy check plus target synthesis/STA; physical claims only if run |

## Exploration rules and common traps

1. Treat executable source as decisive when a historical report disagrees.
2. Do not use `make all` as shorthand for every RTL test; its scope is six
   main benches.
3. Rebuild through Make after changing `.svh` or shared headers. Running an old
   binary can produce a false pass.
4. Keep reference mode and synthesizable mode distinct in test reports.
5. Do not advertise W4, noncausal `0x1C`, non-immediate scale loads, or
   generation-2 `flags=0` as RTL-supported.
6. A tiny byte-exact test can miss an overlap bug if the engines never overlap.
   Check co-busy and lost-write/collision counters on a representative bundle.
7. A Yosys hierarchy/check/stat pass establishes hierarchy resolution; its
   pre-process `check` is diagnostic. It is not an asserted mapped-design
   structural proof, timing result, or physical closure.
8. Label optional/skipped tests and missing fixtures. Skips are not passes.
9. Preserve the frozen golden dependency/fixture manifest. Intentional changes
   require reviewed conformance revisions, not automatic hash refreshes.
10. Never publish throughput without direct cycles, a matching clock, and the
    named DRAM model.

## Definition of exploration complete

The codebase tour is complete when the explorer has produced:

- one annotated 64-bit instruction and Python/RTL decode comparison;
- one architectural storage/layout diagram;
- one fetch/control/engine cycle trace including a barrier and a fault;
- one DMA/SRAM port-ownership trace and one systolic tile trace;
- one numeric-path trace labeling all type conversions;
- one compiler-to-bundle emission map and runtime patch map;
- one golden-versus-RTL comparison report;
- one test-evidence matrix that states scope, skips, and known failures;
- one reproducible performance record with hashes and counters;
- one implementation-readiness statement that separates elaboration from
  physical sign-off.

At that point, a contributor can make a scoped change and know which contracts,
modules, tests, measurements, and documents must move together.
