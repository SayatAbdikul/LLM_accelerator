# FPGA integration status

Updated: 2026-10-11

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
| `Makefile` | Converts the wrapper with `sv2v`, checks hierarchy fatally, and reports diagnostic `check` warnings and statistics. |

The common compute RTL comes from `rtl/common/filelists/core.f`; the FPGA
directory adds only target wrappers and stubs.

## Current elaboration gate

Run from the repository root:

```bash
make -C rtl/fpga yosys-fpga
```

The Makefile supplies two preprocessor defines:

```text
TARGET_FPGA
SFU_SYNTH_NO_DPI
```

`SFU_SYNTH_MODE` and `HELPER_SYNTH_MODE` are RTL parameters, not defines.
`taccel_top_fpga` defaults both to 1 and passes them explicitly to `taccel_top`.
The preprocessor guard removes DPI imports; it does not select the compute
mode by itself. The wrapper does not override `FETCH_BUFFER_ENABLE`, so the
shared-core default of 0 remains in effect. The instruction buffer is an
explicit simulation candidate (`run_program_synth` versus
`run_program_synth_nofetch`), not a promoted FPGA-wrapper default.

It then converts SystemVerilog with `sv2v` and runs:

```text
hierarchy -check -top taccel_top_fpga
check
stat
```

`hierarchy -check` rejects unresolved modules. The following `check` runs
before process lowering and is diagnostic, not `check -assert`: unused
`sv2v`-inlined function temporaries can produce undriven warnings at this
stage. A successful target therefore proves hierarchy resolution, not an
asserted structural check of a lowered or mapped design. It does not run
device mapping, place-and-route, timing analysis, power analysis, or bitstream
generation, and it is not evidence of FPGA resource fit or clock frequency.

The wrapper deliberately rejects synthesis configurations that retain DPI-C.
Functional equivalence is established by the separate software/RTL conformance
tests described in [`../TESTBENCHES.md`](../TESTBENCHES.md), not by this target.
The mandatory `--verification-profile required` selects actual mode-1 program
gates and fails skipped core checks; the frozen reference manifest pins 73
executable dependency and fixture files. Compiler `target="rtl-gen2"`
validation rejects software-only forms before deployment, but does not provide
FPGA placement, memory binding or timing closure.

For numerical checks, decoder outputs are FP16 logits in DRAM, not the native
runner's legacy INT32 ACCUM `summary.logits` diagnostic. A clean halt or tiny
conformance pass does not establish GPT-2 124M quality; nonfinite logits cannot
pass its quality gate. See the [software guide](../../software/CODEBASE.md)
for compiler candidates and runtime contracts.

## Work required for a real board

1. Select the FPGA part, board, clock, reset, and external-memory topology.
2. Replace the PLL, I/O, and DDR placeholders with vendor or open-source IP.
3. Add pin, clock, false-path, and I/O timing constraints.
4. Decide whether inferred SRAMs are acceptable or add explicit BRAM/URAM
   bindings and validate read-during-write behavior.
5. Add a Vivado, Quartus, or nextpnr build with utilization and timing gates.
6. Add board-level reset, memory, program-loading, and end-to-end inference
   tests before describing the target as deployable.
