# ASIC build path

Step E (2026-05-26 RTL restructure) created this skeleton. **Default PDK
is SKY130** (sky130A standard cells); IHP130 can be added later by dropping
a sibling `src/sram_dp_ihp130.sv` and extending the Makefile's
`PDK_SRAM_FILE_<pdk>` mapping. Full OpenLane integration is deferred
until a PDK is installed locally.

## Layout

| Path | Role |
|---|---|
| `src/taccel_top_asic.sv` | wraps the verified `taccel_top` core with off-chip pads (clk, rst, start/done/fault, AXI master) and routes through pad ring stub |
| `src/pad_ring_stub.sv` | placeholder for SKY130 IO library (sky130_fd_io_*); 2-FF reset synchronizer for now |
| `src/sram_dp_sky130.sv` | declares `module sram_dp_macro` with a BEHAVIORAL stub body; lands real `sky130_sram_*` instantiations when OpenLane is wired up |
| `openlane/` | reserved for `config.tcl`, SDC, `pin_order.cfg` |
| `libs/` | reserved for PDK liberty/lef pointers (env-var driven) |
| `Makefile` | `yosys-asic` smoke gate; stub `openlane` target |

## Target-axis defines (set by `Makefile`)

```
-DTARGET_ASIC          # selects sram_dp_macro binding in the common
                       # SRAM dispatch wrapper
-DSFU_SYNTH_NO_DPI     # elides DPI-C imports (required for synthesis)
-DSFU_SYNTH_MODE=1     # routes SFU through synthesizable fp32 primitives
-DHELPER_SYNTH_MODE=1  # routes helper engine through synthesizable chain
```

## Wrapper `\`error` guards

`taccel_top_asic.sv` carries two compile-time guards that refuse to
elaborate under misconfigured builds:

```systemverilog
`ifndef SFU_SYNTH_NO_DPI
  `error "TARGET_ASIC requires SFU_SYNTH_NO_DPI; ..."
`endif

`ifndef TARGET_ASIC
  `error "taccel_top_asic requires -DTARGET_ASIC; ..."
`endif
```

These compose with the gen-2 ISA freeze: the design synthesizes with zero
behavioral/DPI dependency, byte-exactly equivalent to the verified golden
model under `software/tests/test_compare_rtl_golden.py`.

## SRAM macro composition (deferred)

The behavioral stub in `src/sram_dp_sky130.sv` will become a bank of
`sky130_sram_*` macros when OpenLane is set up. Bank-target sizes:

| Buffer | DATA_W × DEPTH | Bytes | Macro composition |
|---|---|---|---|
| ABUF | 128 × 8192 | 128 KB | TBD |
| WBUF | 128 × 16384 | 256 KB | TBD |
| ACCUM | 128 × 4096 | 64 KB | TBD |

Note: at ~1 mm² per 2 KB on sky130A, the full 448 KB of on-chip SRAM
exceeds the eFabless Caravel user-area budget (~10 mm²) by ~20×. The
tape-out strategy (substrate-IP vs full-model trade-off) is documented in
the top-level README; the SRAM bank sizes here will likely shrink when
the final chip-scope is set.

## Wiring OpenLane (deferred)

When OpenLane is installed and a PDK is in hand:

1. Add `openlane/config.tcl` (point at `../common/filelists/core.f` +
   `../asic/src/`).
2. Add `openlane/pin_order.cfg`, `openlane/sdc/taccel.sdc`.
3. Replace the behavioral stub in `src/sram_dp_sky130.sv` with banked
   `sky130_sram_*` macro instances.
4. Add `libs/sky130/` PDK Liberty/LEF pointers (env-var driven).
5. Replace the `openlane` Makefile target with a real OpenLane runscript.
6. Run DRC/LVS/STA closure; commit GDS as tape-out-ready artifact.

The shared filelist `rtl/common/filelists/core.f` remains the source of
truth for the core RTL across all closure steps — no churn there.
