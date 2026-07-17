# RTL Testbench Guide

RTL verification is standardized on **native Verilator C++ benches** — fast,
deterministic unit and subsystem checks. This is the live layer; everything in
"Running Tests" below that starts `make -C rtl/verilator` is expected to pass.

> **The cocotb tier is DORMANT (assessed 2026-07-17) — do not treat it as a gate.**
> `cocotb` is not declared in `software/requirements.txt` and is not installed, so
> `make -C rtl/cocotb ...` cannot run here. It has not been touched since
> 2026-05-26 (`e150095`) and therefore predates ~20 RTL-changing commits — the
> `m_exact` CONFIG_TILE extension, the Port-A/Port-S split, the lever-D transpose
> LOAD, the ACCUM preclear deletion and `div_p6`/`sqrt_p6`. Whether these benches
> still pass is UNKNOWN, because nothing runs them.
>
> It is also fully superseded: every cocotb module has a live Verilator
> counterpart (`test_dma`, `test_decode`+`test_control`, `test_helpers`,
> `test_sfu`, `test_systolic`, `test_systolic_chained`), and the Verilator tier is
> far richer (32 targets incl. `test_fp32_exp_p18` and `test_systolic_qkt_*`).
> The cocotb references below are retained as a map of what it once covered.
> Either declare + refresh the tier, or retire it — but do not read it as coverage.

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
  - `software/tools/rtl_cosim.py` (the RTL-vs-golden prefill co-sim driver;
    replaced `compare_rtl_golden.py`, deleted in `aa08309`)
  - `software/tests/test_compare_rtl_golden.py`
  - `software/tests/test_batched_decode.py`

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

- Build the native runners with `make -C rtl/verilator run_program run_program_synth`.
- **Live RTL == golden byte-match gates — there are TWO, both real:**
  - `pytest software/tests/test_batched_decode.py` — tiny decode bundles incl. the
    packed batch-16 path and the b32 M_pad>16 two-m-tile walk.
  - `pytest software/tests/test_compare_rtl_golden.py` — the freeze §4.5 prefill
    byte-match (`test_rtl_cosim_gen2_byte_match`), plus the frozen-golden blob-SHA
    pin, which must stay green. The task-#105 bridge this leg once waited on was
    BUILT (`tools/rtl_cosim.py`); it skips only when `run_program` or the tiny
    fixture is missing.
- **RTL-vs-golden prefill co-simulation:** `software/tools/rtl_cosim.py` serializes a
  frozen decoder-bundle prefill stream into a single-shot ProgramBinary
  (`--out`, `--token`, `--cosim`).
- **Byte-exactness is a TINY-model gate only.** On 124M it is ill-posed: past the
  first FP16 overflow (`block0_out_proj`) the golden model saturates too, so both
  sides go wrong together (`rtl_cosim.py` #109). Use argmax/perplexity conformance
  there instead — see `software/tools/evaluate_gpt2_perplexity.py`.

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
