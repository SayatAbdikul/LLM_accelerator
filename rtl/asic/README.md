# ASIC build path

Step E (2026-05-26 RTL restructure) created this skeleton; it has since grown
into the project's **primary measured target**. **Default PDK is SKY130**
(sky130A standard cells, installed locally via ciel); IHP130 can be added
later by dropping a sibling `src/sram_dp_ihp130.sv` and extending the
Makefile's `PDK_SRAM_FILE_<pdk>` mapping.

**What actually runs today** (this is where every fmax number comes from):

- `build/synth_blocks/` — sky130 yosys synthesis suite: `synth_full.sh <top>`
  (full-design, `abc -D 5000` — the calibrated mapper every STA number uses),
  `synth_block.sh` (per-block, iterative delay tightening), `synth_sky130.sh`
  (macro-tuned flow), plus per-block OpenSTA drivers (`run_block_sta.sh`,
  `sta_one.sh`, `*_sta.tcl`, `*_sweep.tcl`).
- `build/openroad/` — **OpenROAD per-block PNR + STA** on sky130_fd_sc_hd:
  `block_pnr.tcl` (generic: netlist + top + period), plus tuned per-block
  scripts (`dma_pnr.tcl`, `helper_pnr.tcl`, `sfu_pnr*.tcl`). The post-PNR
  34.41 MHz figure comes from this flow.
- Memory limits: the **full-SFU flatten (and full-chip PNR) OOMs below
  ~24 GB RAM** — per-block and standalone-primitive flows are the supported
  path on smaller boxes; never run two yosys jobs concurrently on 15 GB.
  yosys's SHARE pass hangs on the SFU fp32 cones — the scripts use
  `-noshare` where needed.

## Layout

| Path | Role |
|---|---|
| `src/taccel_top_asic.sv` | wraps the verified `taccel_top` core with off-chip pads (clk, rst, start/done/fault, AXI master) and routes through pad ring stub |
| `src/pad_ring_stub.sv` | placeholder for SKY130 IO library (sky130_fd_io_*); 2-FF reset synchronizer for now |
| `src/sram_dp_sky130.sv` | declares `module sram_dp_macro` with a BEHAVIORAL stub body; lands real `sky130_sram_*` instantiations when the macro composition is chosen |
| `build/synth_blocks/`, `build/openroad/` | the working synth/STA/PNR flows (above) + their netlists and logs |
| `openlane/` | OpenLane-2 config exists for `fetch_unit` (`config.yaml` + `pin_order.cfg`); other blocks TBD — the de-facto PNR flow is OpenROAD direct |
| `libs/` | reserved for PDK liberty/lef pointers (env-var driven; the flows currently point straight at the ciel sky130A install) |
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

## SRAM macro composition (still deferred)

The behavioral stub in `src/sram_dp_sky130.sv` will become a bank of
`sky130_sram_*` macros when the tape-out scope is set. Bank-target sizes:

| Buffer | DATA_W × DEPTH | Bytes | Macro composition |
|---|---|---|---|
| ABUF | 128 × 8192 | 128 KB | TBD |
| WBUF | 128 × 16384 | 256 KB | TBD |
| ACCUM | 128 × 4096 | 64 KB | TBD |

Note: at ~1 mm² per 2 KB on sky130A, the full 448 KB of on-chip SRAM
exceeds the eFabless Caravel user-area budget (~10 mm²) by ~20×. No
tape-out strategy is decided; the SRAM bank sizes here will likely shrink
when the final chip-scope is set. (The macro's port contract — 1rw1r,
write-first Port A, read-only Port B — is load-bearing: the whole
DMA-prefetch-under-MATMUL overlap scheme and the Port-S drain channel
assume it. See `docs/porta_bus_split.md` before changing it.)

## Remaining full-chip closure steps

1. Choose the SRAM macro composition and replace the behavioral stub in
   `src/sram_dp_sky130.sv` with banked `sky130_sram_*` instances.
2. Full-chip (or at least full-SFU) PNR on a ≥24 GB machine — per-block
   PNR + calibrated standalone-primitive STA is the current evidence basis
   (`docs/t0_sfu_fmax_audit.md`, `docs/lever_e_fmax_cluster.md`).
3. Either extend the OpenLane-2 configs (`openlane/<block>/config.yaml`)
   block by block, or stay on OpenROAD direct; add SDC per block.
4. Run DRC/LVS/STA closure; commit GDS as tape-out-ready artifact.

The shared filelist `rtl/common/filelists/core.f` remains the source of
truth for the core RTL across all closure steps — no churn there.
