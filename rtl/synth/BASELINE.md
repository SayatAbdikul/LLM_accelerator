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

2. **2D unpacked array declarations in `systolic_*`** — **CLOSED Phase 3**
   (2026-05-21). All 7 declarations across `systolic_array.sv` and
   `systolic_controller.sv` packed as
   `logic [SYS_DIM-1:0][...][7:0] arr` (and `[31:0]` for pe_acc).
   yosys parses the full control plane (all 12 modules) and synthesizes
   through OPT/DFF/SHARE passes into ABC tech-mapping. Verilator suites
   `test_systolic` (8/8), `test_systolic_chained` (7/7), `test_sfu`
   (21/21), and the freeze cosim (6+1) all unchanged by the refactor.
   `rtl/synth/blackbox_stubs.v` updated to parameterize the SFU and
   helper stubs (`SFU_SYNTH_MODE` / `HELPER_SYNTH_MODE`) for hierarchy
   elaboration.

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
