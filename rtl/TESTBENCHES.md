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
  - `rtl/cocotb/test_systolic.py`
  - `rtl/cocotb/test_systolic_chained.py`
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
