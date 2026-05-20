# Phase-0 synth-check baseline (2026-05-19)

Empirical baseline from the new `make synth-check` gate (yosys generic synth,
no FPGA part). The gate is **structurally established** (script:
`rtl/synth/synth_check.ys`; Makefile target: `rtl/verilator/Makefile`
`synth-check`; tooling: yosys 0.65 + sv2v 0.0.13 via Homebrew). It is
currently **RED** as expected — the work list is the synthesizable-RTL effort.

## How the gate runs

```sh
sv2v -I rtl/src/include -I rtl/src/systolic <CONTROL_SV> -w build/synth/design_full.v
yosys -p "read_verilog build/synth/design_full.v" rtl/synth/synth_check.ys
```

`sv2v` adapts SystemVerilog (packages, enums, `logic`, `always_ff/comb`,
generate, `import pkg::*`) to Verilog-2005 that yosys's built-in frontend
parses. yosys then attempts `hierarchy -top taccel_top; synth; check -assert;
stat`.

## What's red (and why)

Two distinct root causes, both expected/named in the plan:

1. **DPI-C imports + `real`-typed storage in `sfu_engine.sv` /
   `blocking_helper_engine.sv`** — non-synthesizable by definition; this is
   the dominant work the rewrite eliminates. First yosys error in the full
   design hits at `design_full.v:709` (the first DPI-C import in
   `blocking_helper_engine`, `sfu_fp32_round`):

   ```
   ERROR: syntax error, unexpected TOK_ID, expecting ')' or ','
   ```

   **Closes naturally by Plan Phase 2** (DPI→pipelined fp32 primitives;
   `real`→`logic [31:0]`).

2. **2D unpacked array declarations in `systolic_*`** —
   `rtl/src/systolic/systolic_controller.sv:91`
   (`a_tile_scratch [0:15][0:15]`) and 7 declarations in
   `rtl/src/systolic/systolic_array.sv:33–41` (`a_skew/b_skew/pe_acc/
   pe_a_in/pe_b_in/pe_a_out/pe_b_out`). yosys 0.65's built-in frontend
   reports `ERROR: Insufficient number of array indices for a_tile_scratch`.
   **This is a yosys-frontend limitation, NOT an RTL defect** — Vivado /
   Quartus / `yosys-slang` (full SV frontend) handle 2D unpacked arrays
   cleanly. The control-plane diagnostic (full CONTROL_SV minus
   sfu_engine/blocking_helper_engine, those blackboxed via
   `rtl/synth/blackbox_stubs.v`) elaborates **9 of 11** modules through
   yosys (`taccel_pkg`, `decode_unit`, `fetch_unit`, `control_unit`,
   `register_file`, `sram_dp`, `sram_subsystem`, `systolic_pe`,
   `systolic_array`) and stops at this one parser gap. Two closures:

   - **Surgical edit:** rewrite the 7 declarations as packed 2D
     (`logic [15:0][15:0][7:0] arr`) — syntactic equivalence, indexing
     `arr[i][j]` unchanged, zero functional change. Must re-pass `test_systolic`
     suite to confirm no Verilator regression.
   - **Stronger frontend:** install `yosys-slang` plugin (source build,
     no homebrew formula) — leaves RTL untouched.

   Either lands in **Plan Phase 3** alongside the SFU/helper green-up;
   not on the Phase-1/2 critical path.

## Per-module verdict (from the diagnostic)

| Module | yosys+sv2v | Note |
|---|---|---|
| `taccel_pkg.sv` | ✅ parsed | package + enums + structs |
| `decode_unit.sv` | ✅ synth-clean | (after sv2v) |
| `fetch_unit.sv` | ✅ synth-clean | (after sv2v) |
| `control_unit.sv` | ✅ synth-clean | (after sv2v) |
| `register_file.sv` | ✅ synth-clean | warns "Replacing memory \\addr_regs/\\scale_regs with list of registers" — fallback, not a blocker |
| `sram_dp.sv` | ✅ synth-clean | `(* ram_style = "block" *)` BRAM-inferable |
| `sram_subsystem.sv` | ✅ synth-clean | |
| `systolic_pe.sv` | ✅ synth-clean | |
| `systolic_array.sv` | ✅ parsed (warns memory→regs) | 2D-array closure (#2 above) needed for full synth |
| `systolic_controller.sv` | ❌ 2D-unpacked-array parser gap | yosys frontend limitation (#2) |
| `dma_engine.sv` | (downstream of #2) | not reached today; audit says clean |
| `taccel_top.sv` | (downstream of #2) | not reached today; audit says clean |
| `sfu_engine.sv` | ❌ DPI/`real` (Phase 2) | the dominant work |
| `blocking_helper_engine.sv` | ❌ DPI/`real` (Phase 2) | smaller, mechanical |

## Gate exit definition

`make synth-check` returns **0** when:
- yosys completes `synth -top taccel_top; check -assert; stat`
- All modules in `$(CONTROL_SV)` parse and elaborate
- Zero `import "DPI-C"` and zero `real`-typed signals remain in `rtl/src/`

Per the plan, this is the FPGA-demo roadmap Phase-2 definition of done.
