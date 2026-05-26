# FPGA build path

Step D (2026-05-26 RTL restructure) created this skeleton. **No FPGA part
is picked yet**, so only the `yosys-fpga` elaboration smoke gate is wired
today. Full Vivado / Quartus / nextpnr integration is deferred until a
target board is chosen.

## Layout

| Path | Role |
|---|---|
| `src/taccel_top_fpga.sv` | wraps the verified `taccel_top` core with board-level pins (clk, rst, start/done/fault) and routes the AXI master to the DDR stub |
| `src/pll_stub.sv` | clock pass-through; replace with vendor MMCM/PLL |
| `src/iobuf_stub.sv` | 2-FF reset synchronizer; replace with vendor IBUF + ASYNC_REG |
| `src/ddr_axi_stub.sv` | AXI4 slave stub that always-acks but never returns data; replace with vendor DDR controller IP (Xilinx MIG, Intel UniPHY, LiteDRAM, etc.) |
| `src/sram_dp_fpga.sv` | reserved placeholder for explicit BRAM/URAM bindings; FPGA target currently uses the inferred BRAM body from `rtl/common/src/memory/sram_dp_inferred.sv` |
| `constraints/` | reserved for XDC (Vivado) / SDC files |
| `Makefile` | `yosys-fpga` smoke gate via `sv2v + yosys` |

## Target-axis defines (set by `Makefile`)

```
-DTARGET_FPGA          # selects FPGA bindings in the common RTL
-DSFU_SYNTH_NO_DPI     # elides DPI-C imports (required for synthesis)
-DSFU_SYNTH_MODE=1     # routes SFU through synthesizable fp32 primitives
-DHELPER_SYNTH_MODE=1  # routes helper engine through synthesizable chain
```

These compose with the gen-2 ISA freeze: the design synthesizes with
zero behavioral/DPI dependency, equivalent to the verified golden model
under the byte-exact freeze cosim gate (`test_compare_rtl_golden.py`).

## Wrapper `\`error` guard

`taccel_top_fpga.sv` guards against misconfigured builds:

```systemverilog
`ifndef SFU_SYNTH_NO_DPI
  `error "TARGET_FPGA requires SFU_SYNTH_NO_DPI; ..."
`endif
```

If a future build flow forgets to define `SFU_SYNTH_NO_DPI`, the FPGA
wrapper refuses to elaborate.

## Picking a part — what changes

When a target FPGA is chosen:

1. Replace `pll_stub.sv` with vendor PLL (Vivado MMCM, Intel IOPLL).
2. Replace `iobuf_stub.sv` with vendor IBUF + ASYNC_REG synchronizer.
3. Replace `ddr_axi_stub.sv` with vendor DDR controller IP (or LiteDRAM).
4. Add `constraints/<board>.xdc` (or `.sdc`) with pin assignments.
5. Add a `vivado` / `quartus` / `nextpnr` target to `Makefile`.
6. Optionally add explicit BRAM/URAM bindings in `src/sram_dp_fpga.sv`.

The shared filelist `rtl/common/filelists/core.f` remains the source of
truth for the core RTL — no churn there.
