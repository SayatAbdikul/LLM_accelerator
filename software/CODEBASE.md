# TACCEL software codebase

**Audited:** 2026-09-03 against the code after `3486c01`.

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
│   │   ├── speculative.py      # Prompt-lookup speculative decoding
│   │   └── *_reference.py      # FP32, fake-quant, W8A16/W8A8 references
│   └── utils/                  # Integer and tensor helpers
├── tools/                      # User-facing and diagnostic CLIs
├── tests/                      # 56 pytest modules
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

`compiler/frontend/deit_plugin.py` preserves the encoder frontend. It is
supported but no longer drives the main performance campaign.

`ModelConfig` validates dimensions, attention geometry, maximum sequence
length, embedding type, and scale policy. `CodeGenerator` requires an explicit
model configuration; it no longer silently defaults to DeiT.

### IR and tiling

`IRGraph` is an ordered graph of `IRNode` operations. Nodes carry shapes,
inputs, weights, scales, and lowering attributes. Dimensions are padded to
multiples of 16. The compiler strip-mines tensors that exceed ABUF, WBUF, or
ACCUM capacity and stages large weights through DRAM.

### Code generation

`compiler/codegen.py` coordinates allocation, instruction emission,
relocations, trace events, and the shared data image.

The active numeric lowering lives in `compiler/w8a16_emit/`:

- `matmul.py`: activation quantization, W8 tiled matmul, per-channel epilogue,
  bias folding, and weight-prefetch scheduling;
- `attention.py` and `packed_attn.py`: QKᵀ, causal softmax, attention-value
  matmul, grouped/batched layouts;
- `sublayer.py`: generation-2 LayerNorm, GELU, softmax, residual add, and
  quantize/dequantize operations;
- `_common.py`: precision flags, `m_exact`, and shared layout helpers.

`compiler/emit/` contains target-neutral dispatch and data movement. Legacy
W8A8 compute fallbacks were removed; `use_fp16_activations` remains only as a
backward-compatible constructor argument and is always treated as true.

### Decoder bundle construction

`build_decoder_program_bundle` compiles both streams, unions stream-specific
large-weight requirements, lays out shared data once, validates relocation
compatibility, and returns `DecoderBundleBuild` with its KV layout and both
code generators.

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
simulator. Its content hash is frozen by
`tests/test_compare_rtl_golden.py` and must be deliberately revised when the
contract changes.

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

`runtime/speculative.py` adds opt-in prompt-lookup speculative decoding. It is
not enabled by default and is guarded by byte-inertness tests.

## Tools

| Tool | Purpose |
|---|---|
| `asm.py` / `disasm.py` | Assembly and disassembly |
| `run_golden.py` | Execute a ProgramBinary in the golden model |
| `convert_hf_gpt2_to_nanogpt.py` | Convert supported Hugging Face GPT-2 weights |
| `evaluate_gpt2_perplexity.py` | PTQ/fake-quant/golden perplexity evaluation |
| `rtl_cosim.py` | Serialize a decoder stream and compare RTL with golden |
| `bench_decode_cycles.py` | Direct decode cycle measurement by position/batch |
| `fast_gate_b16.py` | Cycle, overlap, fault, and logits-hash gate |
| `profile_decode_step.py` | Detailed retire-gap and engine profile |
| `gen_gen2_fixtures.py` | Regenerate frozen generation-2 RTL fixtures |
| `audit_porta_argmax.py` | Port-A overlap/correctness audit |
| `bench_specdec_*.py` | Speculative-decoding measurements |
| `debug_*.py` | Focused quality and lowering diagnostics |

## Tests

The 56 test modules collect 462 cases in the current default environment.
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

Run:

```sh
.venv/bin/python -m pytest -q software/tests
```

Large checkpoints, slow PPL gates, Hugging Face parity, 124M co-simulation, and
external replay data are optional. Relevant tests skip when their prerequisite
is absent or require `PYTEST_SLOW=1` / `PYTEST_124M=1`. Four Stage-5 preset
tests currently have stale monkeypatch targets after the calibration package
split; see the current [project status](../docs/project_status.md) before
interpreting a broad-suite result.

## Change rules

1. Keep `opcodes.py`, `encoding.py`, `instructions.py`,
   `rtl/common/src/include/taccel_pkg.sv`, and `decode_unit.sv` synchronized.
2. Preserve the eight-byte instruction width and 16-byte SRAM address unit.
3. Treat a golden-simulator hash update as an ISA/conformance revision.
4. Regenerate generation-2 fixtures after an intentional golden or codec
   change and confirm whether raw payloads moved.
5. Any DMA/systolic overlap change must preserve logits hashes and report the
   Port-A audit counters; byte equality on a non-overlapping tiny graph is not
   sufficient.
6. Hardware-target compilation must not silently claim W4 support until RTL
   implements or rejects `weight_int4` explicitly.
