# ASIC build path

**Audited:** 2026-09-03
**Research PDK:** SKY130A / `sky130_fd_sc_hd`
**Status:** wrapper elaboration and per-block research flows exist; full-chip
physical closure does not

The ASIC directory is the project's measured physical-design path. It is not
a tape-out-ready project.

## What is implemented

- `src/taccel_top_asic.sv`: ASIC-facing wrapper around the shared
  `taccel_top` core and AXI master.
- `src/pad_ring_stub.sv`: placeholder clock/reset pad boundary.
- `src/sram_dp_sky130.sv`: behavioral `sram_dp_macro` boundary matching the
  shared two-port memory contract.
- `Makefile`: `sv2v + yosys` wrapper elaboration smoke gate.
- `build/synth_blocks/`: SKY130 Yosys/ABC and OpenSTA research scripts.
- `build/openroad/`: direct per-block OpenROAD placement/timing scripts.

The `openlane/` and `libs/` directories are placeholders. The Makefile
`openlane` target intentionally exits nonzero.

## Wrapper smoke gate

From the repository root:

```sh
make -C rtl/asic yosys-asic
```

This command:

1. reads `rtl/common/filelists/core.f`;
2. adds the ASIC wrapper, pad stub, and SKY130 SRAM boundary;
3. converts with `sv2v` using:

   - `TARGET_ASIC`;
   - `SFU_SYNTH_NO_DPI`;
   - `SFU_SYNTH_MODE=1`;
   - `HELPER_SYNTH_MODE=1`;

4. runs Yosys `hierarchy -check`, `check`, and `stat` on
   `taccel_top_asic`.

This is an elaboration check, not mapped synthesis or timing closure.

The wrapper contains compile-time guards for `TARGET_ASIC` and
`SFU_SYNTH_NO_DPI` so a misconfigured synthesis command fails early.

## Memory boundary

The core requests three logical memories:

| Buffer | Width × depth | Capacity |
|---|---:|---:|
| ABUF | 128 × 8,192 | 128 KiB |
| WBUF | 128 × 16,384 | 256 KiB |
| ACCUM | 128 × 4,096 | 64 KiB |

`sram_dp_sky130.sv` currently models these with registers. It does not
instantiate a real SKY130 SRAM macro.

The macro replacement must preserve:

- Port A read/write behavior;
- Port B read-only behavior;
- write-first Port A semantics;
- the latency assumed by `sram_subsystem.sv`;
- the independent logical channels used for systolic Port S routing.

The complete 448 KiB logical capacity is much larger than a small
Caravel-style SRAM budget. Banking, capacity reduction, or a different
integration substrate must be decided before physical closure.

## Technology-mapped research scripts

The scripts under `build/synth_blocks/` are separate from the Makefile smoke
gate. Examples:

```sh
rtl/asic/build/synth_blocks/synth_full.sh taccel_top
rtl/asic/build/synth_blocks/synth_block.sh fp32_div_p6
rtl/asic/build/synth_blocks/run_block_sta.sh fp32_div_p6
```

Important portability constraint: these scripts currently hard-code a Linux
Ciel SKY130 path under `/home/user/.ciel/...`, and one OpenSTA driver uses
`/tmp/gcc-shim/sta`. Update those variables/paths for the local installation
before running them. They are reproducibility artifacts, not portable build
automation.

`synth_full.sh` performs:

- `sv2v` conversion with ASIC/synth defines;
- Yosys `synth -flatten`;
- `dfflibmap`;
- ABC mapping against `sky130_fd_sc_hd__tt_025C_1v80.lib`;
- cell/area statistics.

Per-block scripts are preferred on memory-limited machines.

## OpenROAD block flow

`build/openroad/block_pnr.tcl` accepts:

```text
<netlist.v> <top> <period_ns> [utilization_percent]
```

Example shape:

```sh
openroad -no_init -exit -threads 4 \
  rtl/asic/build/openroad/block_pnr.tcl \
  <netlist.v> <top> <period_ns> <utilization>
```

Specialized `dma_pnr.tcl`, `helper_pnr.tcl`, and `sfu_pnr*.tcl` scripts
capture prior experiments. Inspect their hard-coded PDK paths and assumptions
before reuse.

## Timing status

The historical 34.41 MHz post-PNR value was obtained from an earlier
per-block/primitive configuration. Later commits integrated previously
unmeasured long exponential, GELU, dequant, LayerNorm, and scale-generation
paths. It must not be quoted as current complete-SFU or full-chip sign-off.

The latest pipeline campaign reduced known long paths to approximately one FP
primitive per stage, generally mapping in the 27–32 ns range depending on
context. The post-`451e7df` whole-lane run exceeded 15 GB and was killed, so a
current end-to-end fmax is still open. See
[the current status](../../docs/project_status.md).

## Resource constraints

- Full-SFU/full-chip flattening and PNR can exceed 15 GB.
- Yosys mapping varies materially with synthesis context; a standalone
  primitive result is a proxy, not an exact full-SFU frequency.
- Do not run multiple heavy Yosys/OpenROAD jobs concurrently on a small host.
- Some flows avoid Yosys sharing because the FP cones make it slow or
  unstable.

## Work required for physical closure

1. Parameterize PDK/tool paths instead of hard-coding the original machine.
2. Choose an SRAM macro strategy and implement `sram_dp_macro`.
3. Choose the integration target and replace the pad stub with real IO.
4. Create maintained timing constraints and power intent.
5. Run full-SFU and then full-chip mapping, placement, CTS, routing, and STA.
6. Close DRC/LVS and validate memory timing/behavior.
7. Re-measure performance using the achieved clock and a named memory model.

The shared core source list remains
`rtl/common/filelists/core.f` throughout these steps.
