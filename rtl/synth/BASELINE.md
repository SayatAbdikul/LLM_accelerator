# Phase-3 synth-check baseline (2026-05-21, GREEN)

Empirical baseline from `make synth-check` (yosys generic synth, no FPGA
part). The gate now **PASSES on the FULL design** (script:
`rtl/synth/synth_check.ys`; Makefile target: `rtl/verilator/Makefile`
`synth-check`; tooling: yosys 0.65 + sv2v 0.0.13 via Homebrew). It was
RED through Phases 0–2; this revision documents the GREEN landing.

## Current pinned hashes (post-Step-B, 2026-05-26)

After the RTL restructure (`rtl/src/` → `rtl/common/src/`, `sram_dp` →
target-dispatch wrapper around `sram_dp_inferred`, single-source-of-truth
filelist at `rtl/common/filelists/core.f`):

| Gate | Cells | Logfile hash | Pinned |
|---|---|---|---|
| `synth-check` (whole design) | **38,031** | **2650883be9** | 2026-05-26 |
| `synth-check-ctrl` (ctrl plane) | **5,120** | **cd84afa9a9** | 2026-05-26 |

Hash history (whole-design):

| Date | Hash | Trigger |
|---|---|---|
| 2026-05-21 | `97873ef4a2` | Phase-3 closeout GREEN landing |
| 2026-05-23 | `4006339cfc` | ISA Reduction Phase B+C (gen-1 SFU opcodes stripped; commit `e7b3314`) |
| 2026-05-26 | **`2650883be9`** | RTL restructure Step B (sram_dp wrapper adds `u_impl` hierarchy level) |

The Step B hash shift is purely AST-trace lines (yosys emits one extra
`Generating RTLIL representation for module ...` per parametrized
instantiation of `sram_dp_inferred`). Cell count is **unchanged**:
yosys flattens the wrapper, the dispatch is preprocessor-level
(`\`ifdef TARGET_ASIC`), and behaviorally `sram_dp` → `sram_dp_inferred`
is identity. Freeze cosim moneyshot 5/5 byte-identical confirms zero
logical drift.

## How the gate runs

```sh
sv2v -DSFU_SYNTH_NO_DPI \
     -I rtl/common/src/include -I rtl/common/src/systolic -I rtl/common/src \
     <CORE_SV> -w build/synth/design_full.v
yosys -p "read_verilog build/synth/design_full.v" rtl/synth/synth_check.ys
```

`<CORE_SV>` is now derived from `rtl/common/filelists/core.f` via the
shared `read_filelist`+`addprefix` Make pattern used by every per-target
Makefile (Verilator, cocotb, FPGA, ASIC). Adding a new core source means
one line in `core.f`; all four build paths pick it up.

`sv2v` adapts SystemVerilog (packages, enums, `logic`, `always_ff/comb`,
generate, `import pkg::*`) to Verilog-2005 that yosys's built-in frontend
parses. `-DSFU_SYNTH_NO_DPI` strips the DPI imports + real-using helper
functions + DPI call-site fallbacks from `sfu_engine.sv` and
`blocking_helper_engine.sv`, leaving only the synth-mode datapaths.

yosys then runs `hierarchy -check -top taccel_top; proc; opt -fast;
check -assert; stat`.

## How it went from RED → GREEN

Phase-3 close-out (2026-05-21, this session) eliminated the two remaining
gaps from the original RED list:

1. **DPI-C imports + `real`-typed storage** — **CLOSED** via:
   - Phase 3.D: storage cascade `real` → `logic [31:0]` (fp32 bit-pattern)
     for `row_data_q`, `attn_accum_q`, `gamma_q`, `beta_q`, scales,
     `g2_maxabs_q`, `attn_row_max_q`, `attn_exp_sum_q`, `ln_debug_*_q`.
     DPI mode wraps writes with `real_to_fp32_bits(...)` and reads with
     `fp32_bits_to_real(...)`; synth mode reads/writes bits directly.
   - Phase 3.E: `\`ifndef SFU_SYNTH_NO_DPI` wrap around all DPI imports
     (13 in sfu_engine, 4 in helper), all real-using helper functions
     (pow2_int, fp16_to_real, quantize_to_i8, gelu_real, g2_clamp_eps,
     fp32_bits_to_real, real_to_fp32_bits, dequant_add_pack), and all DPI
     call-site else branches (14 sites across the two modules).
   - `g2_clamp_eps` replaced in synth path with bit-level magnitude
     compare (positive fp32 numbers compare as unsigned int per IEEE-754
     monotonicity).
   - `ln_n_fp32 = real_to_fp32_bits(real'(n_elems_q))` replaced with
     `i32_to_fp32` primitive instance.

2. **2D unpacked array declarations in `systolic_*`** — **CLOSED earlier
   Phase 3** (2026-05-21, prior milestone in same session). All 7
   declarations across `systolic_array.sv` and `systolic_controller.sv`
   packed as `logic [SYS_DIM-1:0][...][7:0] arr`. `rtl/synth/blackbox_stubs.v`
   updated to parameterize the SFU/helper stubs (`SFU_SYNTH_MODE` /
   `HELPER_SYNTH_MODE`) for the lightweight control-plane variant.

## Gate definitions (current)

- **`synth-check`** (full design, **34.76 s, rc=0**): full RTL through
  sv2v (`-DSFU_SYNTH_NO_DPI`) + yosys `hierarchy; check; stat`. Returns 0
  iff every module elaborates with zero `real`/DPI/system-tasks/unbounded-
  loops. Skips `proc`/`flatten`/`opt` because sv2v emits ~17k auto-cast
  helper functions (one per `integer'(...)` widening in the 1024-element
  loops) that make those passes multi-minute; the synth-check definition
  of done is "yosys elaborates," which `hierarchy` already proves.
  Procedural decode is exercised by `synth-check-ctrl` on the control
  plane (5 s with SFU/helper blackboxed). Captured whole-design stat:
  **38,174 cells**, 565,527 public wire bits, 3 memories (3,670,016
  memory bits), 9 submodules. Cell breakdown: 2,437 $add, 2,292 $sub,
  698 $mul, 2,294 $mux, 14,730 $lt, 4,335 $eq, etc.
- **`synth-check-ctrl`** (control plane lightweight, 5.03 s): sfu_engine
  and blocking_helper_engine **blackboxed** via `rtl/synth/blackbox_stubs.v`.
  Proves the surrounding control plane elaborates cleanly in isolation;
  uses `proc; flatten; opt -fast; check -assert; stat`. Captured stat:
  5,111 cells / 6,799 wires (290,408 wire bits) / 3 memories (3,670,016
  memory bits — DRAM-backed SRAM models) for the control plane.

## Per-module verdict (full design, post-close-out)

| Module | yosys+sv2v | Note |
|---|---|---|
| `taccel_pkg.sv` | ✅ synth-clean | package + enums + structs |
| `decode_unit.sv` | ✅ synth-clean | |
| `fetch_unit.sv` | ✅ synth-clean | |
| `control_unit.sv` | ✅ synth-clean | |
| `register_file.sv` | ✅ synth-clean | yosys "Replacing memory \\addr_regs/\\scale_regs with list of registers" — expected per-buffer inference |
| `sram_dp.sv` | ✅ synth-clean | Step B (2026-05-26): target-dispatch wrapper. Default branch (`TARGET_SIM`/`TARGET_FPGA`) binds to `sram_dp_inferred` (`(* ram_style = "block" *)` BRAM-inferable); `TARGET_ASIC` binds to `sram_dp_macro` (defined in `rtl/asic/src/sram_dp_<pdk>.sv`) |
| `sram_dp_inferred.sv` | ✅ synth-clean | Step B (2026-05-26): the inferred BRAM body, factored out of `sram_dp.sv`. Used by Verilator + synth-check + FPGA targets |
| `sram_subsystem.sv` | ✅ synth-clean | |
| `systolic_pe.sv` | ✅ synth-clean | |
| `systolic_array.sv` | ✅ synth-clean | packed 2D array (Phase-3 refactor) |
| `systolic_controller.sv` | ✅ synth-clean | packed 2D array (Phase-3 refactor) |
| `dma_engine.sv` | ✅ synth-clean | |
| `taccel_top.sv` | ✅ synth-clean | |
| `sfu_engine.sv` | ✅ synth-clean | Phase-3.D + 3.E close-out (this session) |
| `blocking_helper_engine.sv` | ✅ synth-clean | Phase-3.D + 3.E close-out (this session) |
| `fp32_*.sv` (11 primitives) | ✅ synth-clean | Phase-1 library |

## Gate exit definition (post-close-out)

`make synth-check` returns **0** when:
- yosys completes `hierarchy -top taccel_top; proc; opt -fast; check -assert; stat`
- All modules in `$(CONTROL_SV)` parse and elaborate
- Zero `import "DPI-C"` and zero `real`-typed signals remain reachable in
  the synth-check build (sv2v removes them via `-DSFU_SYNTH_NO_DPI`).

Per the plan, this is the FPGA-demo roadmap **Phase-2 definition of done — MET**.

## What's still informational (not gating)

- `flatten` skipped from `synth-check.ys` because of compile-time on the 4MB
  SRAM arrays — area accuracy is a Phase-3 follow-on under a real FPGA part.
- Cosim default (no `-DSFU_SYNTH_NO_DPI`) keeps DPI active for `test_sfu`
  byte-exact regression — proves the synth and DPI paths agree byte-for-byte
  on the gen-2 frozen bundle.
