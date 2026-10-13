# LLM Accelerator (TACCEL)

TACCEL is a research LLM inference accelerator with a SystemVerilog core and
a Python compiler/runtime stack. The maintained compute path is GPT-2-class
decoder inference with INT8 weights, FP16 activation storage, INT32 systolic
accumulation, and FP32 internal special-function arithmetic.

The hardware contains a 16×16 INT8 systolic array, DMA engine, blocking helper
engine, synthesizable SFU, three on-chip SRAMs, and an in-order issue unit. The
software provides the ISA, assembler, model frontends, quantization, compiler,
golden simulator, host runtime, fixtures, and RTL co-simulation tools.

> **Documentation state:** October 13 numerical changes against `ffd71e8`;
> current validation is tracked in [the numerical report](docs/numerical_quality_results.md).
> October 11–12 performance evidence remains in [the milestone report](docs/optimization_results.md).
> Start with [the current project status](docs/project_status.md) and
> [documentation index](docs/README.md). Dated plans and measurement reports
> are retained as historical evidence and are labeled accordingly.

## Current status

| Area | Current state |
|---|---|
| Primary workload | GPT-2 124M / nanoGPT-format causal decoder; the older DeiT frontend remains available but is not the active optimization target. |
| Numeric path | W8A16: INT8 weights, FP16 activation storage, INT32 accumulators, FP32 SFU internals. The KV cache is stored as INT8. |
| RTL | The full mode-1 shared-core hierarchy resolves through `sv2v + yosys`; pre-process checks are diagnostic, not structural sign-off. FPGA and ASIC wrapper elaboration targets are also provided. These are not mapped synthesis or physical sign-off. |
| ISA | Fixed 64-bit, big-endian instructions. Six generation-1 SFU opcodes are retired. The RTL implements the causal generation-2 path; `SOFTMAX_FP32` (`0x1C`) remains software/golden-only and is illegal in RTL. |
| Verification | All 537 mandatory tests pass with no skips; 244 optional cases are deselected. Pinned references, native protocol/primitive tests, and opt-in large-model tests distinguish conformance from numerical quality. |
| Numerical quality | The selected QuaRot W8A16 preset passes the 5% FP32-relative limit on 256 software targets (+2.494%). The agreed three-step RTL continuation produces exact logits/whole KV. This is not corpus-wide or full-window RTL quality proof. |
| Performance | Corrected mode-1 baselines and opt-in compiler/fetch A/B reports replace ambiguous historical runner provenance. No current full-chip post-layout fmax, power, or achieved tokens/second is established. |
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
│   ├── tests/                   # Mandatory core plus optional fixture/quality tests
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

The required profile needs the deterministic tiny fixture and a freshly built
mode-1 runner. Generate a missing fixture, validate it, and build the runner:

```sh
test -f software/tests/fixtures/generated/nanogpt_shakespeare_char_d128_l2.pt || \
  PYTHONPATH=software .venv/bin/python software/tools/train_tiny_fixture.py
PYTHONPATH=software .venv/bin/python software/tools/train_tiny_fixture.py --validate-only
make -C rtl/verilator run_program_synth
```

The conditional generation command writes the ignored tiny checkpoint and
its metadata only when the checkpoint is absent. A metadata mismatch fails
validation; investigate it instead of replacing an existing payload.

```sh
PYTHONPATH=software .venv/bin/python -m pytest -q software/tests
PYTHONPATH=software .venv/bin/python -m pytest -q software/tests --verification-profile required
```

Useful optional gates:

```sh
PYTHONPATH=software PYTEST_SLOW=1 .venv/bin/python -m pytest -q software/tests
PYTHONPATH=software PYTEST_124M=1 .venv/bin/python -m pytest -q \
  software/tests/test_compare_rtl_golden.py
```

The required profile selects the release core and fails skipped mandatory
gates. The full developer suite also contains fixture- and environment-dependent
tests. Use a clean checkout on the same machine as the comparison baseline when evaluating an
unrelated change; see [project status](docs/project_status.md) for the most
recent verified scope.

### Run the native RTL gates

```sh
make -C rtl/verilator all
make -C rtl/verilator test_sfu_synth test_helpers_synth
make -C rtl/verilator test_dma_buffer test_fetch_buffer test_fetch_no_buffer
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
PYTHONPATH=software .venv/bin/python -m pytest -q \
  software/tests/test_compare_rtl_golden.py \
  software/tests/test_batched_decode.py
```

The tiny decoder fixture is the conformance target with frozen per-op FP16
ULP bands. Scheduling A/B changes require identical logical RTL outputs and KV
bytes. Historical GPT-2 124M captures reached FP16 non-finite values; the optional
quality gate requires finite logits and cannot pass merely because non-finite
hashes match. Shape-only cycle measurements are not an accuracy claim.

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

Historical cycle measurements are retained for context:

| Shape | Position | Step cycles | Cycles/token |
|---|---:|---:|---:|
| B=1 decode | 511 | 18,318,261 | 18,318,261 |
| B=16 decode | 510 | 49,998,042 | 3,124,878 |

These measurements use the historical `run_program_synth --fast-beats` build
through `451e7df`; helper-mode provenance is not verified. Their historical
"logits hashes" actually label an INT32 ACCUM diagnostic, not FP16 decoder
logits. Use [the October report](docs/optimization_results.md) for corrected
configuration, actual DRAM-output hashes, and matched candidates.
The formerly quoted 34.41 MHz post-PNR number excluded several later-integrated
long paths and must not be presented as current full-chip closure or used as
an achieved-throughput conversion.

For fresh measurement:

```sh
make -C rtl/verilator run_program_synth
PYTHONPATH=software .venv/bin/python software/tools/bench_decode_cycles.py \
  --positions 0,63,255,511 --batch 1
PYTHONPATH=software .venv/bin/python software/tools/fast_gate_b16.py \
  --batch 16 --position 510
```

Do not report tokens/second from a cycle result without naming the clock and
DRAM model. `--fast-beats --beat-interval N` supplies one 128-bit read beat
every N core cycles; writes are independently unthrottled. It is not a shared
fixed-total GB/s memory controller.

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
  does not consume `CONFIG_TILE.weight_int4`. Hardware-target compilation
  rejects W4 and noncausal `SOFTMAX_FP32`; research-golden stays explicit.
- The broad Python suite still includes optional large fixtures and long
  perplexity tests. Stale Stage-5, embedding, provenance, and trained-reference
  regressions were repaired without weakening numerical tolerances. Current status is recorded in
  [docs/project_status.md](docs/project_status.md).

## Documentation

- [Documentation index](docs/README.md)
- [Current project status](docs/project_status.md)
- [Optimization implementation and results](docs/optimization_results.md)
- [Current W8A16 numerical quality evidence](docs/numerical_quality_results.md)
- [Code exploration roadmap](docs/code_exploration_roadmap.md)
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
