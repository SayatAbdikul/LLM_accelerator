# FPGA integration status

Updated: 2026-09-03

This directory is a target-integration skeleton, not a deployable board
project. No FPGA family or board is selected, and there are no vendor project
files, pin constraints, generated clocks, timing constraints, or real external
memory controller. The current `yosys-fpga` target verifies only that the
complete FPGA wrapper hierarchy elaborates without unresolved modules.

For the project-wide implementation and verification status, see
[`../../docs/project_status.md`](../../docs/project_status.md).

## Current files

| Path | Current role |
|---|---|
| `src/taccel_top_fpga.sv` | Wraps `taccel_top` with placeholder board pins and connects its AXI master to the DDR stub. |
| `src/pll_stub.sv` | Clock pass-through; not a PLL or generated-clock implementation. |
| `src/iobuf_stub.sv` | Reset synchronizer placeholder; not a complete board I/O binding. |
| `src/ddr_axi_stub.sv` | AXI slave placeholder that accepts request/data handshakes but never returns read or write responses; real transactions stall. |
| `src/sram_dp_fpga.sv` | Reserved for explicit BRAM/URAM bindings; it is not included in the current target file list. |
| `constraints/` | Placeholder for future XDC or SDC constraints. |
| `Makefile` | Builds the wrapper with `sv2v` and runs a Yosys hierarchy/check/stat gate. |

The common compute RTL comes from `rtl/common/filelists/core.f`; the FPGA
directory adds only target wrappers and stubs.

## Current elaboration gate

Run from the repository root:

```bash
make -C rtl/fpga yosys-fpga
```

The target defines:

```text
TARGET_FPGA
SFU_SYNTH_NO_DPI
SFU_SYNTH_MODE=1
HELPER_SYNTH_MODE=1
```

It then converts SystemVerilog with `sv2v` and runs:

```text
hierarchy -check -top taccel_top_fpga
check
stat
```

This proves source resolution and structural elaboration only. It does not run
device mapping, place-and-route, timing analysis, power analysis, or bitstream
generation, and it is not evidence of FPGA resource fit or clock frequency.

The wrapper deliberately rejects synthesis configurations that retain DPI-C.
Functional equivalence is established by the separate software/RTL conformance
tests described in [`../TESTBENCHES.md`](../TESTBENCHES.md), not by this target.

## Work required for a real board

1. Select the FPGA part, board, clock, reset, and external-memory topology.
2. Replace the PLL, I/O, and DDR placeholders with vendor or open-source IP.
3. Add pin, clock, false-path, and I/O timing constraints.
4. Decide whether inferred SRAMs are acceptable or add explicit BRAM/URAM
   bindings and validate read-during-write behavior.
5. Add a Vivado, Quartus, or nextpnr build with utilization and timing gates.
6. Add board-level reset, memory, program-loading, and end-to-end inference
   tests before describing the target as deployable.
