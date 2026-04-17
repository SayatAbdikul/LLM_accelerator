# TACCEL RTL Plan

## 1. Purpose

This document replaces the original phase-only RTL plan with a software-aligned
roadmap.

The source of truth is now the current software stack:

- `software/taccel/isa`
- `software/taccel/compiler`
- `software/taccel/golden_model`
- `software/tests`
- `software/docs/isa_spec.md`

The goal is not to build a generic accelerator shell. The goal is to build RTL
that can execute compiler-generated programs for the current TACCEL software
stack, starting with baseline DeiT-tiny inference and then extending to the
experimental accuracy/performance paths already present in software.

## 2. Software-Defined Hardware Contract

### 2.1 Target workload

The current compiler and IR are hard-coded around `facebook/deit-tiny-patch16-224`.

Key dimensions:

- Sequence length: 197 tokens, padded to 208
- Embedding dimension: 192
- Attention heads: 3
- Head dimension: 64
- MLP hidden dimension: 768
- Transformer depth: 12
- Classifier output: 1000 classes

Host-side assumptions:

- Patch embedding is done on the CPU
- The accelerator starts from pre-embedded INT8 patch tokens in DRAM
- DRAM also stores weights, biases, FP16 scale tables, CLS token, and position embeddings

### 2.2 Memory contract

The software stack assumes these fixed SRAM sizes:

| Buffer | Size | Element view | Addressing unit |
|---|---:|---|---|
| ABUF | 128 KB | INT8 activations | 16 bytes |
| WBUF | 256 KB | INT8 weights / FP16 params / INT32 bias | 16 bytes |
| ACCUM | 64 KB | INT32 accumulators | 16 bytes |

Other architectural assumptions:

- DRAM is byte-addressable, little-endian for data
- Instruction words are 64-bit big-endian
- Minimum DRAM size is 16 MB
- All SRAM and DRAM offsets in the ISA are expressed in 16-byte units
- The compiler relies on software strip-mining, padding, and allocation; RTL
  does not need dynamic scheduling or dynamic SRAM management

### 2.3 Instruction set required by software

The software ISA has 20 defined instructions:

#### System and setup

- `NOP`
- `HALT`
- `SYNC`
- `CONFIG_TILE`
- `SET_SCALE`
- `SET_ADDR_LO`
- `SET_ADDR_HI`

#### Data movement

- `LOAD`
- `STORE`
- `BUF_COPY`

#### Core compute

- `MATMUL`
- `REQUANT`
- `REQUANT_PC`
- `SCALE_MUL`
- `VADD`
- `SOFTMAX`
- `LAYERNORM`
- `GELU`
- `SOFTMAX_ATTNV`
- `DEQUANT_ADD`

### 2.4 Which instructions are truly needed right now

The software codebase has three practical tiers of hardware requirements.

#### Tier 1: required for baseline compiler-generated DeiT-tiny programs

These are needed to run the default compiler flow without enabling experimental
feature flags:

- `LOAD`
- `STORE`
- `BUF_COPY`
- `MATMUL`
- `REQUANT`
- `VADD`
- `SOFTMAX`
- `LAYERNORM`
- `GELU`
- `CONFIG_TILE`
- `SET_SCALE` with immediate FP16 source
- `SET_ADDR_LO`
- `SET_ADDR_HI`
- `SYNC`
- `NOP`
- `HALT`

#### Tier 2: required for software experimental paths already present in repo

These are not always emitted by the default compiler configuration, but they are
already implemented in the ISA, golden model, compiler, and tests:

- `REQUANT_PC`
- `DEQUANT_ADD`
- `SOFTMAX_ATTNV`
- `SCALE_MUL`
- `GELU` with `ACCUM` input for `gelu_from_accum`

#### Tier 3: supported by ISA but not on the critical path today

- `SET_SCALE` with `src_mode = ABUF/WBUF/ACCUM`

The assembler and golden model support buffer-backed `SET_SCALE`, but the
current compiler emits immediate-only `SET_SCALE`.

### 2.5 Numerical requirements

The RTL must match these software-visible semantics:

- `MATMUL`: INT8 x INT8 -> INT32
- `REQUANT`, `REQUANT_PC`, `SCALE_MUL`, `SOFTMAX`, `LAYERNORM`, `GELU`,
  `DEQUANT_ADD`: round-half-to-even before clipping
- `VADD`:
  - INT8 saturating add for residual-style paths
  - INT32 add with row broadcast for bias-add paths
- `BUF_COPY transpose`:
  - source shape is `[src_rows * 16, cols]`
  - destination shape is `[cols, src_rows * 16]`
- `SOFTMAX`:
  - input is INT32 ACCUM or equivalent dequantized path
  - output is INT8 probabilities
- `LAYERNORM`:
  - `gamma` then `beta` are packed in WBUF as FP16
- `SOFTMAX_ATTNV`:
  - consumes QK^T accumulators and INT8 V
  - emits INT8 attention@V

### 2.6 Execution Model And Dispatch Contract

The software stack distinguishes between:

- instructions that are architecturally non-blocking and therefore visible to
  `SYNC` through a resource busy bit
- instructions that are architecturally blocking helpers, meaning later
  instructions may consume their results without an additional `SYNC`

The key rule for RTL is:

- `SYNC` is architecturally defined only over DMA, systolic, and SFU resources
- compiler-emitted `SYNC` after a blocking helper may be conservative and does
  not by itself force that helper onto the corresponding asynchronous unit

That matters most for `BUF_COPY` and `SCALE_MUL`. The current compiler often
emits `SYNC(001)` after `BUF_COPY` and `SYNC(100)` after emitted `SCALE_MUL`
uses, but software correctness does not require those instructions to expose a
DMA-visible or SFU-visible busy state. If they are implemented as blocking
helpers, those `SYNC`s simply retire as no-ops when the selected asynchronous
unit is idle.

| Instruction(s) | Architectural path | `SYNC` resource visibility | Contract for dependent instructions |
|---|---|---|---|
| `LOAD`, `STORE` | DMA | `0b001` | Non-blocking from issue stage; later consumers wait with `SYNC(001)` when needed |
| `MATMUL` | Systolic | `0b010` | Non-blocking from issue stage; later consumers wait with `SYNC(010)` |
| `SOFTMAX`, `LAYERNORM`, `GELU`, `SOFTMAX_ATTNV` | SFU | `0b100` | Non-blocking from issue stage; later consumers wait with `SYNC(100)` |
| `BUF_COPY` | Blocking helper / local data-movement path | None architecturally required | Results must be complete before any dependent instruction issues; compiler currently often follows with conservative `SYNC(001)` |
| `REQUANT`, `REQUANT_PC`, `VADD`, `DEQUANT_ADD` | Blocking helper / issue-stage path | None | Compiler assumes results are directly available to later instructions without extra `SYNC` |
| `SCALE_MUL` | Blocking helper by default plan | None architecturally required | Compiler currently follows emitted uses with conservative `SYNC(100)`; SFU placement is optional, not required by software correctness |
| `CONFIG_TILE`, `SET_SCALE`, `SET_ADDR_LO`, `SET_ADDR_HI` | Blocking control path | None | State update completes before the next dependent instruction |
| `SYNC` | Barrier | N/A | Waits only on DMA / systolic / SFU |
| `NOP`, `HALT` | Blocking control path | None | `NOP` retires; `HALT` terminates |

This plan therefore treats `BUF_COPY` and `SCALE_MUL` as architecturally
blocking helpers unless and until the hardware intentionally chooses to expose
them through DMA or SFU busy tracking. Either implementation strategy is
acceptable as long as the ISA-visible completion semantics above are preserved.

### 2.7 Golden Model Organization Is Not Hardware Dispatch

The Python golden model is sequential and organizes instruction behavior for
software convenience, not to define RTL unit boundaries.

In particular:

- `BUF_COPY` lives in `golden_model/dma.py` because it is a memory transform
- `SOFTMAX`, `LAYERNORM`, `GELU`, and `SOFTMAX_ATTNV` live in `golden_model/sfu.py`
- `REQUANT`, `REQUANT_PC`, `SCALE_MUL`, `VADD`, and `DEQUANT_ADD` are handled
  directly inside `golden_model/simulator.py`

RTL dispatch and busy-state design must follow the architectural contract above,
not Python file boundaries.

## 3. Current RTL State

### 3.1 Implemented and in good shape

- Fetch path
- Combinational decode
- Basic in-order control FSM
- Register file for tile config, address regs, scale regs
- SRAM wrappers
- DMA engine for contiguous transfers
- Systolic `MATMUL`
- Verilator regression for decode/control/DMA/systolic

### 3.2 Implemented but not yet software-aligned

- RTL opcode table still stops at `0x10`; software defines instructions through `0x13`
- The target reserved-opcode range is `0x14-0x1F`; any `0x11-0x1F reserved`
  comments in current RTL are outdated once the software ISA is taken as the contract
- DMA implementation assumes one AXI burst per instruction and effectively caps
  the usable transfer length at 256 beats
- Top-level fault plumbing does not yet surface SRAM OOB as an architectural fault
- `SET_SCALE` only works for immediate mode in RTL

### 3.3 Missing relative to software contract

- `BUF_COPY`
- `REQUANT`
- `REQUANT_PC`
- `SCALE_MUL`
- `VADD`
- `SOFTMAX`
- `LAYERNORM`
- `GELU`
- `SOFTMAX_ATTNV`
- `DEQUANT_ADD`

### 3.4 Known semantic mismatch to remove

The current systolic testbenches preload ABUF in a transposed layout to match
the present controller implementation. That is a temporary RTL quirk, not part
of the software ISA contract.

The software contract is:

- `MATMUL src1` is a normal logical `A[M, K]`
- `MATMUL src2` is a normal logical `B[K, N]`
- If a transpose is needed, software expresses it explicitly with `BUF_COPY`

The revised plan must therefore remove any requirement for callers to manually
transpose standard ABUF activations to satisfy the hardware.

## 4. Feature Matrix

| Feature | Needed by default compiler | Needed by software experiments | RTL now |
|---|---|---|---|
| 20-op decode table | No | Yes | No |
| Fetch/decode/control shell | Yes | Yes | Yes |
| `SET_SCALE` immediate | Yes | Yes | Yes |
| `SET_SCALE` from SRAM | No | Maybe | No |
| `LOAD` / `STORE` basic semantics | Yes | Yes | Yes |
| `LOAD` / `STORE` with large `xfer_len` | Yes | Yes | No |
| `BUF_COPY` flat | Yes | Yes | No |
| `BUF_COPY` transpose | Yes | Yes | No |
| `MATMUL` | Yes | Yes | Yes |
| `MATMUL` with software-visible layout semantics | Yes | Yes | Not yet |
| `REQUANT` | Yes | Yes | No |
| `VADD` INT8 saturating | Yes | Yes | No |
| `VADD` INT32 broadcast bias-add | Yes | Yes | No |
| `SOFTMAX` | Yes | Yes | No |
| `LAYERNORM` | Yes | Yes | No |
| `GELU` ABUF->ABUF | Yes | Yes | No |
| `REQUANT_PC` | No | Yes | No |
| `DEQUANT_ADD` | No | Yes | No |
| `SOFTMAX_ATTNV` | No | Yes | No |
| `SCALE_MUL` | No | Yes | No |
| SRAM OOB architectural fault | Yes | Yes | Partial |

## 5. Software-Driven Hardware Requirements

### 5.1 DMA must support architectural transfer lengths, not testbench lengths

This is the most important gap after opcode alignment.

The compiler emits `LOAD` and `STORE` instructions with `xfer_len` sized for the
actual tensor transfer, not for a single AXI burst. Examples already present in
software:

- Input patches: `37,632 B = 2,352` units
- Typical `192 x 192` weight matrix: `36,864 B = 2,304` units
- FC1 / FC2 weights: `147,456 B = 9,216` units

Therefore the RTL DMA engine must:

- accept full 16-bit `xfer_len`
- internally split the transfer into a sequence of legal AXI bursts
- preserve the architectural semantics of one `LOAD` / `STORE` instruction
- keep `SYNC(001)` semantics unchanged from software's point of view

The current `<= 256 beat` assumption is only a bring-up limitation and must not
survive into the software-aligned RTL.

### 5.2 The accelerator does not need hardware strip-mining

The compiler already solves these problems in software:

- FC1 / FC2 strip-mining
- QK^T strip-mining
- ABUF / WBUF allocation
- DRAM spill / reload
- padding to multiples of 16

RTL must simply implement the ISA correctly. It does not need to discover
tiles, compact buffers automatically, or infer scheduling policy on its own.

### 5.3 Baseline DeiT-tiny does not require every optional ISA path

The baseline compiler flow can avoid:

- `REQUANT_PC`
- `DEQUANT_ADD`
- `SOFTMAX_ATTNV`
- `SCALE_MUL`
- `SET_SCALE` from SRAM

That gives a clean functional milestone for full-model RTL bring-up.

However, these paths are already real parts of the software stack and should be
planned as first-class follow-on work, not as future speculation.

## 6. Revised Roadmap

### Phase A: Contract Sync And Architectural Closure

#### Goal

Make the current RTL skeleton match the current software-visible architecture
before adding more datapaths.

#### Work items

- Update `taccel_pkg`, decoder, and control logic to the 20-op ISA
- Decide which ops are implemented immediately versus trapped as unsupported
  during bring-up
- Plumb SRAM OOB and reserved-buffer faults to top-level `fault` / `fault_code`
- Clarify fetch-side fault policy for AXI response errors
- Decide whether buffer-backed `SET_SCALE` is:
  - implemented now, or
  - explicitly deferred and rejected in hardware/software configuration
- Add assertions and documentation that the ISA contract uses normal logical
  matrix layout, not the current testbench transpose workaround

#### Exit criteria

- One shared opcode table across software docs and RTL
- Top-level bad-buffer and SRAM-OOB faults are observable and tested
- The plan explicitly states which ISA paths are baseline and which are optional

### Phase B: DMA Productionization

#### Goal

Turn the current DMA into the engine the compiler actually needs.

#### Work items

- Support full 16-bit `xfer_len`
- Internally chop transfers into legal AXI bursts
- Preserve contiguous SRAM row stepping across burst boundaries
- Preserve contiguous DRAM byte addressing across burst boundaries
- Handle read and write backpressure cleanly
- Define and test AXI error handling policy
- Verify arbitration between fetch and DMA during long transfers

#### Tests

- Long `LOAD` / `STORE` transfers larger than 256 beats
- Compiler-sized weight loads
- Input-patch DMA load
- Store backpressure and BRESP handling
- DRAM OOB and boundary conditions across multi-burst transfers

#### Exit criteria

- Compiler-emitted DMA instructions no longer require software chunking

### Phase C: Blocking Helper Instructions And Local Data Movement

#### Goal

Implement the helper instructions required for baseline programs and software
dataflow manipulation.

#### Must-have work

- `BUF_COPY`
  - flat copy
  - transpose copy
  - same-buffer overlap semantics compatible with software compaction flows
  - implementation may be a blocking helper or a dedicated local copy engine;
    software does not require DMA-visible overlap for correctness
- `VADD`
  - INT8 saturating residual add
  - INT32 broadcast bias add into ACCUM
- `REQUANT`
  - INT32 -> INT8
  - round-half-to-even

#### Optional but already queued by software

- `REQUANT_PC`
- `SCALE_MUL`

#### Why this phase matters

Without these instructions the compiler cannot express:

- K -> K^T transforms for attention
- CLS extraction and head concatenation
- residual adds
- bias adds
- INT32 -> INT8 requantization after matmul
- ABUF compaction flows used by the allocator

#### Exit criteria

- Compiler-generated non-SFU baseline programs can execute up through matmul,
  copy, bias-add, residual-add, and requant paths

### Phase D: SFU Functional Parity

#### Goal

Implement the scalar-function unit needed by baseline DeiT-tiny inference.

#### Must-have work

- `SOFTMAX`
  - ACCUM input
  - INT8 output
  - row-wise semantics matching software
- `LAYERNORM`
  - INT8 input
  - FP16 gamma/beta from WBUF
  - INT8 output
- `GELU`
  - ABUF input
  - INT8 output

#### Strongly recommended in the same phase

- `GELU` from `ACCUM` input

This path is already used by the software `gelu_from_accum` option and is much
cheaper to add while the GELU datapath is fresh than as a later retrofit.

#### Implementation guidance

- Shared FP32 datapath is acceptable
- Throughput is secondary to numerical parity in the first implementation
- Match the software's round-half-to-even behavior exactly

#### Exit criteria

- A default compiler-generated DeiT-tiny program runs end-to-end in RTL
- Outputs match the golden model at the ISA-visible INT8 / INT32 boundaries

### Phase E: Experimental Software Feature Parity

#### Goal

Support the feature flags that the software stack already uses for accuracy and
performance experiments.

#### Work items

- `REQUANT_PC`
- `DEQUANT_ADD`
- `SOFTMAX_ATTNV`
- `SCALE_MUL`
- Validation of fused out-proj accumulation flows that compose:
  - `BUF_COPY`
  - `SCALE_MUL`
  - `MATMUL` accumulate mode
  - `REQUANT_PC` or `REQUANT`
  - optional `DEQUANT_ADD`

#### Notes

These features are not required for the simplest baseline compiler run, but
they are already first-class citizens in:

- compiler options
- golden-model simulator
- ISA tests
- quantization experiments

So they belong in the main roadmap, not in a speculative appendix.

#### Exit criteria

- Compiler flags for experimental paths can be enabled without manual RTL-side
  patching or custom software restrictions

### Phase F: Performance, Robustness, And Cleanups

#### Goal

Close the remaining program-level sign-off gap after ISA parity by adding an
RTL-vs-golden acceptance harness, internal observability, and cleanup work that
does not change the architectural contract.

#### Candidate work

- Verilator-first program runner for compiled `ProgramBinary` images
- Software-side RTL-vs-golden comparison driver and regression summaries
- Internal-only performance counters and fault/retire trace hooks
- Cleaner AXI response handling and fault observability
- Locale-safe and ASCII-safe regression cleanup
- Synthesis-driven cleanup and timing closure after sign-off is green
- Optional instruction cache

#### Explicitly deferred until after parity

- Any optimization that changes visible numerical behavior
- Any optimization that requires software to change data layout contracts
- Any optimization that depends on undocumented compiler assumptions

## 7. Verification Plan

Verification should follow the software stack, not just unit-local RTL intent.

### 7.1 Unit and block tests

- Decoder for all 20 opcodes
- DMA, including multi-burst transfers
- `BUF_COPY` flat and transpose
- `REQUANT`, `REQUANT_PC`, `SCALE_MUL`, `VADD`, `DEQUANT_ADD`
- `SOFTMAX`, `LAYERNORM`, `GELU`, `SOFTMAX_ATTNV`
- SRAM fault and reserved-buffer fault paths

### 7.2 Golden-model comparison

For each instruction family, compare RTL results against the software golden model:

- exact INT32 matches for `MATMUL`
- exact INT8 matches for quant helper and SFU instructions
- exact fault-code matches

### 7.3 Compiler-generated program tests

Required program-level tests:

- Default DeiT-tiny compile
- QK^T strip-mining path
- FC1 / GELU / FC2 strip-mining path
- CLS prepend / position-embedding add / CLS extract
- Bias-add and residual-add paths
- `REQUANT_PC` block subsets
- `DEQUANT_ADD` residual1 paths
- `SOFTMAX_ATTNV` selected blocks
- fused out-proj accumulation variants
- `gelu_from_accum`

### 7.4 Regression criteria

Minimum functional bar:

- Compiler-generated programs run without hand-editing emitted instructions
- Default compiler configuration is supported end-to-end
- Experimental compiler configurations either:
  - work correctly, or
  - are explicitly disabled by software configuration until their RTL exists

## 8. Definition Of Done

### Baseline done

The RTL is baseline-complete when all of the following are true:

- The default compiler-generated DeiT-tiny program runs end-to-end on RTL
- The Verilator sign-off harness compares final RTL logits against the software
  golden model without hand-editing the emitted program
- No software-side instruction rewriting is required for hardware bring-up
- DMA supports real compiler transfer sizes
- `MATMUL`, `BUF_COPY`, `REQUANT`, `VADD`, `SOFTMAX`, `LAYERNORM`, and `GELU`
  behave as defined by the software ISA and golden model
- Fault behavior is architectural, not testbench-local

### Full software-parity done

The RTL is software-parity complete when:

- All 20 ISA instructions are implemented
- Experimental compiler options run on RTL and pass the RTL-vs-golden harness
- The compiler, golden model, ISA docs, and RTL all agree on opcode meaning,
  data layout, and fault behavior

## 9. Non-Goals

These are not required for the current accelerator plan:

- On-chip patch embedding
- Strided or scatter/gather DMA
- Dynamic hardware scheduling beyond the defined issue / busy / `SYNC` model
- Hardware-managed SRAM allocation
- New ISA extensions beyond what software already defines

## 10. Immediate Priorities

If work starts from the current RTL state, the priority order should be:

1. Align RTL decode/control with the 20-op software ISA.
2. Remove the effective `xfer_len <= 256` DMA limitation.
3. Implement `BUF_COPY`, `REQUANT`, and `VADD`.
4. Fix matmul-side data layout so standard software buffers work without
   transposed preload hacks.
5. Implement `SOFTMAX`, `LAYERNORM`, and `GELU`.
6. Add `REQUANT_PC`, `DEQUANT_ADD`, `SCALE_MUL`, and `SOFTMAX_ATTNV`.

That sequence gets the hardware from "Phase 1-3 skeleton" to "compiler-usable
accelerator" with the least software backtracking.
