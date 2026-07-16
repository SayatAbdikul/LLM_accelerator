# RTL Testbench Guide

This repo standardizes RTL verification around two complementary layers:

- Native Verilator C++ benches for fast deterministic unit and subsystem checks.
- cocotb benches for ISA-visible flows, DRAM scoreboarding, and Python reference-model comparison.

## Bench Ownership

- Front-end / control:
  - `rtl/verilator/test_decode.cpp`
  - `rtl/verilator/test_control.cpp`
  - `rtl/cocotb/test_fetch_decode.py`
- Data movement:
  - `rtl/verilator/test_dma.cpp`
  - `rtl/cocotb/test_dma.py`
- Local compute:
  - `rtl/verilator/test_helpers.cpp`
  - `rtl/verilator/test_sfu.cpp`
  - `rtl/cocotb/test_helpers.py`
  - `rtl/cocotb/test_sfu.py`
- Matrix compute:
  - `rtl/verilator/test_systolic.cpp`
  - `rtl/verilator/test_systolic_array_chained.cpp`
  - `rtl/verilator/test_systolic_chained.cpp`
  - `rtl/verilator/test_accum_snapshot_readback.cpp`
  - `rtl/verilator/test_systolic_qkt*.cpp` (attention-shape replay/padded/history)
  - `rtl/cocotb/test_systolic.py`
  - `rtl/cocotb/test_systolic_chained.py`
- Synthesizable-datapath (mode-1) gates — the chip's real datapath, vs the DPI reference:
  - `make -C rtl/verilator test_sfu_synth` (SFU fp32 datapath, 11 cases)
  - `make -C rtl/verilator test_helpers_synth`, `test_sfu_helper_synth`
- FP32 primitive bit-exactness gates (each pipelined primitive vs its
  combinational parent / DPI golden, millions of vectors, zero diffs):
  - `test_fp32_add`, `test_fp32_div` (p2), `test_fp32_div_p3/p4/p5/p6`,
    `test_fp32_sqrt` (p2 via `test_fp32_sqrt`), `test_fp32_sqrt_p3/p4/p6`,
    `test_fp32_exp_p18`
- Program-level sign-off:
  - `rtl/verilator/run_program.cpp`
  - `software/tools/compare_rtl_golden.py`
  - `software/tests/test_compare_rtl_golden.py`

## Shared Harnesses

- C++ benches should build on `rtl/verilator/include/testbench.h`.
  - Use `tbutil::SimHarness` for reset/start/run flow.
  - Use `tbutil::sram_*` helpers for direct SRAM inspection and preload.
  - Use `AXI4SlaveModel` fault injection for read/write error cases.
- cocotb benches should build on `rtl/cocotb/utils/testbench.py`.
  - Use `setup_test()` and `wait_halt()` for the standard reset/start flow.
  - Use `pattern()` and `set_addr()` for common program/data setup.
  - Use `read_accum_16x16()` / `read_accum_32x32()` for top-level MATMUL scoreboarding.

## Required Shape For New Benches

Each new RTL feature should add:

- one focused unit/subsystem bench for localized failures
- one top-level contract bench for ISA-visible behavior
- happy-path, boundary-path, and fault-path checks
- busy/dispatch/sync assertions for any asynchronous engine

When a bug is fixed, prefer the smallest regression at the lowest useful layer first, then add a top-level regression only if the failure crossed module boundaries.

## Running Tests

- Native Verilator:
  - `make -C rtl/verilator test_decode`
  - `make -C rtl/verilator test_dma`
  - `make -C rtl/verilator test_helpers`
  - `make -C rtl/verilator test_sfu`
  - `make -C rtl/verilator test_systolic`
  - `make -C rtl/verilator test_systolic_array_chained`
  - `make -C rtl/verilator test_systolic_chained`
  - `make -C rtl/verilator run_program`
- cocotb:
  - `make -C rtl/cocotb test_all SIM=verilator`
  - `make -C rtl/cocotb test_dma SIM=verilator`
  - `make -C rtl/cocotb test_sfu SIM=verilator`
  - `make -C rtl/cocotb test_systolic_chained SIM=verilator`

## Program Sign-Off

- Build the native runner with `make -C rtl/verilator run_program`.
- Compare a precompiled binary with:
  - `software/tools/compare_rtl_golden.py --summary-out out.json program --program program.bin`
- Compile and compare a model variant with:
  - `software/tools/compare_rtl_golden.py --summary-out out.json compile --scenario baseline_default --weights pytorch_model.bin --image sample.jpg`
- Failed compares automatically leave a work directory with the RTL summary and,
  when needed, golden/RTL trace artifacts for mismatch triage.

Verilator is the primary sign-off simulator. Icarus remains best-effort only.

## Performance Measurement (mode-1)

- `make -C rtl/verilator run_program_synth` builds the measurement runner:
  `SFU_SYNTH_MODE=1` (the chip's datapath, not the DPI model) and a 1 GiB
  DRAM (the 16 MB default faults GPT-2 124M). Always run with `--fast-beats`
  (the pinned honest-bandwidth model); `--beat-interval N` simulates a
  fixed-rate DRAM instead.
- `run_program_noovl` (`SYS_DMA_OVERLAP=0`) is the serialization reference
  for overlap-change A/B gates.
- The RTL exposes audit counters in `taccel_top.sv` (`obs_*`): DMA‖systolic
  co-busy, the Port-A lost-write audit (must read 0 post bus-split), and
  fetch-stall. `software/tools/fast_gate_b16.py` and
  `software/tools/profile_decode_step.py` sample them; any concurrency
  change must quote them (see the doctrine in the top-level README —
  byte-exact alone is a structurally blind gate for overlap changes).
