# LLM Accelerator (TACCEL)

TACCEL is a research LLM inference accelerator with a SystemVerilog core and
a Python compiler/runtime stack. The maintained compute path is GPT-2-class
decoder inference with INT8 weights, FP16 activation storage, INT32 systolic
accumulation, and FP32 internal special-function arithmetic.

The hardware contains a 16×16 INT8 systolic array, DMA engine, blocking helper
engine, synthesizable SFU, three on-chip SRAMs, and an in-order issue unit. The
software provides the ISA, assembler, model frontends, quantization, compiler,
golden simulator, host runtime, fixtures, and RTL co-simulation tools.

> **Documentation state:** audited against commit `3486c01` on 2026-09-03.
> Start with [the current project status](docs/project_status.md) and
> [documentation index](docs/README.md). Dated plans and measurement reports
> are retained as historical evidence and are labeled accordingly.

## Current status

| Area | Current state |
|---|---|
| Primary workload | GPT-2 124M / nanoGPT-format causal decoder; the older DeiT frontend remains available but is not the active optimization target. |
| Numeric path | W8A16: INT8 weights, FP16 activation storage, INT32 accumulators, FP32 SFU internals. The KV cache is stored as INT8. |
| RTL | The full shared-core hierarchy passes the `sv2v + yosys` hierarchy/check/stat gate. FPGA and ASIC wrapper elaboration targets are also provided. These are not mapped synthesis or physical sign-off. |
| ISA | Fixed 64-bit, big-endian instructions. Six generation-1 SFU opcodes are retired. The RTL implements the causal generation-2 path; `SOFTMAX_FP32` (`0x1C`) remains software/golden-only and is illegal in RTL. |
| Verification | 462 Python tests collect. The cleanup validation passed 214 focused Python tests, 94 native RTL checks, the full-design generic elaboration gate, and fixture/hash checks. Optional large-model and replay-data tests remain environment-gated. |
| Performance | Latest measured cycles are 18,318,261 for B=1 at position 511 and 49,998,042 per B=16 step at position 510. A current full-chip post-layout fmax has not been established after the July SFU/helper pipeline work. |
| Physical target | SKY130 is the measured ASIC research path. The FPGA directory is an elaboration-ready wrapper skeleton; no board or vendor flow is selected. |

## Repository layout

```text
LLM_accelerator/
├── software/
│   ├── taccel/
│   │   ├── isa/                 # Opcodes, instruction classes, encoding
│   │   ├── assembler/           # Text assembly and ProgramBinary/ProgramBundle
│   │   ├── compiler/            # Frontends, IR, tiling, allocation, W8A16 lowering
│   │   ├── quantizer/           # W8/W4 research quantizers and calibration helpers
│   │   ├── golden_model/        # Sequential ISA simulator
│   │   └── runtime/             # HostRunner, PPL evaluation, fixtures, spec decode
│   ├── tests/                   # 56 pytest modules
│   ├── tools/                   # Co-sim, profiling, conversion, evaluation CLIs
│   └── docs/                    # ISA contract and historical quantization records
├── rtl/
│   ├── common/
│   │   ├── src/                 # Shared synthesizable accelerator RTL
│   │   └── filelists/core.f     # Source of truth for shared RTL compilation units
│   ├── verilator/               # Primary native RTL benches and program runners
│   ├── asic/                    # SKY130 wrappers and synth/STA/PNR scripts
│   ├── fpga/                    # FPGA wrapper and elaboration stubs
│   └── synth/                   # Generic synthesis gate and campaign records
└── docs/                        # Current status plus dated architecture reports
```

The dormant Cocotb suite, obsolete control-only synthesis gate, retired
generation-1 SFU implementation, and invalid QKT-history bench were removed in
`3486c01`. Native Verilator is the only supported RTL simulation framework.

## Quickstart

Create the Python environment from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r software/requirements.txt
```

RTL builds additionally require Verilator. Synthesis gates require `sv2v` and
Yosys. The ASIC timing scripts require a SKY130 installation plus OpenSTA or
OpenROAD, depending on the selected flow.

### Run the software tests

```sh
.venv/bin/python -m pytest -q software/tests
```

Useful optional gates:

```sh
PYTEST_SLOW=1 .venv/bin/python -m pytest -q software/tests
PYTEST_124M=1 .venv/bin/python -m pytest -q \
  software/tests/test_compare_rtl_golden.py
```

The full suite contains fixture- and environment-dependent tests. Use a clean
checkout on the same machine as the comparison baseline when evaluating an
unrelated change; see [project status](docs/project_status.md) for the most
recent verified scope.

### Run the native RTL gates

```sh
make -C rtl/verilator all
make -C rtl/verilator test_sfu_synth test_helpers_synth
make -C rtl/verilator test_systolic_array_chained test_systolic_qkt
make -C rtl/verilator test_fp32_div_p6 test_fp32_sqrt_p6
make -C rtl/verilator test_fp32_exp_p18 test_fp32_gelu_p33
```

`all` runs the main decode/control/DMA/helper/SFU/systolic benches. The
synthesizable-datapath, split QKT, chained-array, and FP32 primitive gates are
separate targets. QKT replay and padded cases skip when their external replay
directory is unavailable. See [the RTL testbench guide](rtl/TESTBENCHES.md).

### Run the RTL-versus-golden gates

```sh
make -C rtl/verilator run_program run_program_synth
.venv/bin/python -m pytest -q \
  software/tests/test_compare_rtl_golden.py \
  software/tests/test_batched_decode.py
```

The tiny decoder fixture is the byte-exact conformance target. GPT-2 124M
crosses an FP16 non-finite boundary, so large-model validation uses the
documented logits/argmax/perplexity metrics instead of claiming whole-program
byte equality.

### Run RTL elaboration gates

```sh
make -C rtl/verilator synth-check
make -C rtl/asic yosys-asic
make -C rtl/fpga yosys-fpga
```

The first command elaborates the complete shared core. The ASIC and FPGA
commands elaborate their wrappers and target-specific memory bindings. They
are not substitutes for placement, routing, DRC, or board bring-up.

### Run the golden-model demo

```sh
.venv/bin/python software/run_gpt2.py \
  software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
  --prompt-ids 0,1,2 --max-new-tokens 4
```

The converted checkpoint is optional repository data; commands that need it
will skip or fail clearly when it is absent.

### Evaluate perplexity

```sh
PYTHONPATH=software .venv/bin/python \
  software/tools/evaluate_gpt2_perplexity.py \
  software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \
  --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \
  --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \
  --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \
  --max-eval-tokens 257
```

Omitting `--ptq-preset` selects the evaluator default
`output_aware_mlp_lm_head_0_11_pc_full_bc`. Performance fixtures use the
separate frozen `weight_only_int8_quarot` configuration; do not compare their
results without naming the preset.

## Architecture

### RTL

- `taccel_top.sv` connects fetch, control, engines, SRAM ports, AXI, faults,
  and observability counters.
- `fetch_unit.sv` fetches fixed 8-byte instructions over a 128-bit AXI read
  channel.
- `control_unit.sv` issues in order. DMA may overlap the systolic engine;
  helper and SFU operations are serialized by the current scheduler.
- `dma_engine.sv` performs burst LOAD/STORE operations and the INT8
  transpose-LOAD used for K-cache materialization.
- `systolic/` implements the chained 16×16 INT8 array and INT32 tiled
  accumulation.
- `blocking_helper_engine.sv` implements the retained generation-1 helper
  operations: REQUANT, REQUANT_PC, SCALE_MUL, VADD, and DEQUANT_ADD.
- `sfu_engine.sv` implements the active generation-2 W8A16 path. Long
  exponential, GELU, divide, scale-write, dequant, and LayerNorm paths are
  pipelined; `fp32_exp_p18` and `fp32_gelu_p33` are integrated.
- `memory/sram_subsystem.sv` provides ABUF 128 KiB, WBUF 256 KiB, and ACCUM
  64 KiB. Port S is dedicated to systolic writes; same-buffer Port A/S
  collisions fault.

Core parameters are `SYS_DIM=16`, `AXI_DATA_W=128`, four 56-bit address
registers, and sixteen FP16 scale registers.

### Software

1. `isa/` defines the 64-bit encoding and instruction validation.
2. `assembler/` produces `ProgramBinary` and relocatable two-stream
   `ProgramBundle` images.
3. `compiler/` converts DeiT or nanoGPT/GPT-2 graphs into tiled programs,
   including KV-cache layout, runtime patch sites, packed attention, and
   chunked prefill.
4. `quantizer/` and `runtime/calibration/` provide the W8/W4 research and PTQ
   machinery.
5. `golden_model/` executes programs sequentially and provides trace tensors.
6. `runtime/HostRunner` patches embeddings, KV bases, and causal context for
   prefill, batched decode, chunked prefill, and optional speculative decode.

See [the software codebase guide](software/CODEBASE.md) and
[the current ISA specification](software/docs/isa_spec.md).

## Performance

The latest checked-in cycle measurements are:

| Shape | Position | Step cycles | Cycles/token |
|---|---:|---:|---:|
| B=1 decode | 511 | 18,318,261 | 18,318,261 |
| B=16 decode | 510 | 49,998,042 | 3,124,878 |

These measurements use `run_program_synth --fast-beats`. They include the
July pipeline changes through `451e7df` and preserve the known logits hashes.
The formerly quoted 34.41 MHz post-PNR number excluded several later-integrated
long paths and must not be presented as current full-chip closure. At 34.41 MHz
only as an illustrative conversion, the table corresponds to approximately
1.879 tok/s for B=1 and 11.012 aggregate tok/s for B=16.

For fresh measurement:

```sh
make -C rtl/verilator run_program_synth
PYTHONPATH=software .venv/bin/python software/tools/bench_decode_cycles.py \
  --positions 0,63,255,511 --batch 1
PYTHONPATH=software .venv/bin/python software/tools/fast_gate_b16.py \
  --batch 16 --position 510
```

Do not report tokens/second from a cycle result without naming the clock and
DRAM model. The pinned `--fast-beats` model supplies one 128-bit beat per core
cycle; fixed-bandwidth sensitivity is a different experiment.

## Known gaps

- A current full-SFU/full-chip post-layout timing result is still needed. The
  latest whole-lane rerun exceeded the available 15 GB host memory.
- ASIC SRAM macro composition, IO cells, full-chip PNR, DRC, and LVS are not
  complete.
- No FPGA board, clock IP, memory controller, or constraints set is selected.
- `SOFTMAX_FP32` (`0x1C`) is accepted by Python encoding/golden execution but
  rejected by RTL. Supported causal decoder graphs use
  `MASKED_SOFTMAX_FP32`; the noncausal contract still needs a product decision.
- Python supports packed INT4 metadata and golden-model execution, but RTL
  does not consume `CONFIG_TILE.weight_int4`. W4 is therefore not an RTL
  deployment mode.
- The broad Python suite still includes optional large fixtures, long
  perplexity tests, and known baseline issues, including four stale Stage-5
  preset monkeypatch tests. Current status is recorded in
  [docs/project_status.md](docs/project_status.md).

## Documentation

- [Documentation index](docs/README.md)
- [Current project status](docs/project_status.md)
- [ISA specification](software/docs/isa_spec.md)
- [ISA freeze and amendment record](software/docs/isa_generation_freeze.md)
- [Software architecture](software/CODEBASE.md)
- [RTL testbench guide](rtl/TESTBENCHES.md)
- [ASIC flow](rtl/asic/README.md)
- [FPGA flow](rtl/fpga/README.md)
- [Generic RTL elaboration gate](rtl/synth/BASELINE.md)

## License

No license file is present. Contact the repository owner before depending on
or redistributing the project.
