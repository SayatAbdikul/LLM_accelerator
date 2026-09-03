# RTL testbench guide

**Audited:** 2026-09-03.

Native Verilator C++ benches are the supported RTL verification framework.
The former Cocotb tier was dormant, superseded, and removed in `3486c01`.
Icarus is best-effort only.

## Build modes

| Mode | Target/example | Meaning |
|---|---|---|
| Reference | `test_sfu`, `test_helpers`, `run_program` | DPI-backed behavioral SFU/helper comparison path |
| Synthesizable | `test_sfu_synth`, `test_helpers_synth`, `run_program_synth` | `SFU_SYNTH_MODE=1` / `HELPER_SYNTH_MODE=1` chip datapaths |
| No overlap | `run_program_noovl` | `SYS_DMA_OVERLAP=0` serialization reference |

`run_program_synth` uses a 1 GiB modeled DRAM so GPT-2 124M bundles fit.

The Makefile tracks both shared C++ headers and included `.svh` files as
prerequisites. If a source/header changes, do not bypass Make's rebuild check
by executing an old binary directly.

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
```

`test_decode` includes illegal checks for the six retired generation-1 SFU
opcodes and RTL-reserved `SOFTMAX_FP32` `0x1C`. `test_control` covers issue,
SYNC, configuration, dispatch, and architectural faults.

### DMA and helpers

```sh
make -C rtl/verilator test_dma
make -C rtl/verilator test_helpers
make -C rtl/verilator test_helpers_synth
make -C rtl/verilator test_sfu_helper_synth
```

The DMA bench covers regular transfer, errors, and transposed LOAD. The helper
mode-1 bench exercises the synthesizable helper datapath rather than only the
DPI reference.

### SFU

```sh
make -C rtl/verilator test_sfu
make -C rtl/verilator test_sfu_synth
```

`test_sfu_synth` is the generation-2 chip-path gate. It currently runs ten
fixture vectors plus a scale-write/consumer-chain test, covering dequant,
scaled dequant, VADD, LayerNorm, GELU, masked softmax, quantization, and
MAX_ABS scale generation.

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

## Program-level conformance

Build both runners:

```sh
make -C rtl/verilator run_program run_program_synth
```

Run the live Python-driven comparisons:

```sh
.venv/bin/python -m pytest -q \
  software/tests/test_compare_rtl_golden.py \
  software/tests/test_batched_decode.py
```

- `software/tools/rtl_cosim.py` serializes a bundle stream into a
  `ProgramBinary` and compares RTL memory/results with the golden simulator.
- `test_compare_rtl_golden.py` checks the frozen golden content hash,
  deterministic bundle construction, tiny prefill co-simulation, and optional
  124M metrics.
- `test_batched_decode.py` covers tiny decode bundles including packed B=16
  attention and the B=32 multi-M-tile walk.

Tiny-model conformance is byte-exact. The GPT-2 124M path uses logits,
argmax, and perplexity metrics after its FP16 non-finite boundary.

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
  --positions 0,63,255,511 --batch 1
PYTHONPATH=software .venv/bin/python software/tools/fast_gate_b16.py \
  --batch 16 --position 510
```

Performance reports must include:

- batch and position/context;
- direct step cycles and cycles per token;
- RTL mode;
- DRAM model (`--fast-beats` or a named beat interval);
- clock used for any tokens/second conversion;
- logits hash;
- DMA/systolic co-busy and Port-A lost-write/violation counters for an
  overlap change.

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
