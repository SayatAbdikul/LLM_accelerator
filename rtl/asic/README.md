# ASIC build path

**Audited:** 2026-10-11
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
- `tools/block_flow.py`: portable, resource-bounded leaf mapping and optional
  zero-wire OpenSTA, with source/library hashes and physical-area verification.
- `src/sram_dp_sky130_banked.sv`: isolated experimental 16 KiB bank using eight
  pinned OpenRAM 2 KiB macros; not the default memory binding.

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

4. resolves the wrapper's actual `SFU_SYNTH_MODE=1` and
   `HELPER_SYNTH_MODE=1` parameters (these are not preprocessor defines);
5. runs Yosys `hierarchy -check`, `check`, and `stat` on
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

### Representative macro-bank experiment

```sh
# Downloads only explicitly pinned views if absent; never replaces a mismatch.
python3 rtl/asic/tools/fetch_macro_views.py
make -C rtl/asic test-sram-bank
```

`macro_views.json` pins simulation, Liberty, LEF, and GDS views from
[VLSIDA's macro repository](https://github.com/VLSIDA/sky130_sram_macros).
The isolated adapter uses a one-entry delayed-write overlay to preserve
write-first A and old-data B, including simultaneous same-row writes/reads.
A reads defer committing the pending entry; a new A write or idle A cycle
commits it. Matching reads forward the pre-edge pending value, and the physical
B read is disabled during a potential macro collision. All four byte write-mask
lanes are enabled; partial logical writes are not part of this interface.

There is no reset: initial logical contents/outputs are unspecified, just as in
the inferred SRAM. Arbitrary pending state overlays arbitrary macro contents;
software/test logic must initialize known rows through logical A writes, not
by preloading only the physical macros. No initial idle cycle is required.

The test compares the actual inferred SRAM over **96,804 cycles** across eight
geometry/startup cases (32×512, 64×1,024, 128×1,024 and 128×2,048), including
80,000 deterministic randomized cycles and 2,518 logical same-row collisions.
It checks consecutive writes, both-port pending reads, long deferred commits,
disabled-output holds, bank/lane routing, startup, and exhaustive final readback.
Pin-level assertions observe zero undefined physical macro collisions.
Incorrect-forwarding and missing-collision-suppression negative controls fail.
This supports functional equivalence for tested requests, not timing or formal
signoff. The adapter remains absent from `core.f` and the default wrapper.

The selected 2 KiB macro footprint is 683.1 × 416.54 µm. Replicating it for
448 KiB requires 224 macros, totaling 63.737 mm² of macro footprints alone,
before decoders, muxes, SFU storage, logic, pads, and routing. This is a sizing
example for this macro, not a lower bound or a measured chip area. Larger
macros or a revised memory architecture can have different density.
The pinned LEF footprint is 284,538.474 µm² per macro, matching its Liberty
`area`. Physical macro footprint accounting uses LEF separately from
standard-cell area-unit verification. Upstream files and pinned hashes are
unchanged.
The macro input transition constraint also has a
[known integration issue](https://github.com/VLSIDA/OpenRAM/issues/298);
the flow does not silently relax it or claim sign-off.

## Technology-mapped research scripts

The scripts under `build/synth_blocks/` are separate from the Makefile smoke
gate. Examples:

```sh
LIB=/absolute/path/corner.lib rtl/asic/build/synth_blocks/synth_block.sh fp32_div_p6
LIB=/absolute/path/sky130hd-corner.lib STA=/absolute/path/sta \
  DRIVER_CELL=sky130_fd_sc_hd__inv_2 DRIVER_PIN=Y \
  rtl/asic/build/synth_blocks/run_block_sta.sh fp32_div_p6
```

The synthesis scripts require `LIB`. The legacy STA replay also requires
`DRIVER_CELL` and `DRIVER_PIN` from that library; `STA` defaults to `sta`.
The driver above is a SKY130HD example, not a cross-library default. STA replay
reapplies the mode-1 parameters for raw SFU/helper/core tops before hierarchy
elaboration; earlier synthesis-process parameter changes are not persisted in
the raw sv2v source. Specialized historical
scripts still contain original-machine paths and assumptions. Inspect them
before reuse. Do not launch `synth_full.sh` on the laptop until memory bindings
and partition resource measurements support it.

Legacy synthesis area printouts are unverified Liberty units, not physical
µm². Use the maintained portable flow with matching LEF verification for
physical cell-area reporting.

`synth_full.sh` performs:

- `sv2v` conversion with ASIC/synth defines;
- Yosys `synth -flatten`;
- `dfflibmap`;
- ABC mapping against `sky130_fd_sc_hd__tt_025C_1v80.lib`;
- cell/area statistics.

Per-block scripts are preferred on memory-limited machines.

### Maintained matched leaf flow

```sh
python3 rtl/asic/tools/block_flow.py \
  --block systolic_pe --pdk sky130hd --corner tt_025C_1v80 \
  --liberty /absolute/path/corner.lib --cell-lef /absolute/path/cells.lef \
  --out rtl/asic/results/my-run
```

Supported leaves: `systolic_pe`, `dma_engine`, `fetch_unit`, `fp32_div_p6`,
and `fp32_sqrt_p6`. Supply each PDK's pinned Liberty and LEF explicitly.
`--sta` uses native OpenSTA; `--docker-image` may select an existing pinned
image and never pulls. Container jobs are bounded to 8 GiB/two CPUs, and the
default guardrail retains 20 GiB of free disk. Mapping jobs are serial.

The report distinguishes raw Liberty area from µm². It claims µm² only after
every mapped cell's area agrees with its physical LEF size. It records source,
netlist, library and LEF hashes, tool versions, parameters, timing constraints,
runtime, peak RSS and output disk use. Existing reports are not overwritten
unless `--rerun` explicitly archives them. Timeout cleanup terminates mapping
descendants and the named STA container.

The separate `sram_dp_sky130_banked` experiment is mapping-only and requires
SKY130HD periphery, the pinned macro Liberty/LEF, and an explicit physical cell
LEF. A 128-bit × 1,024-row representative bank is mapped with:

```sh
python3 rtl/asic/tools/block_flow.py \
  --block sram_dp_sky130_banked --pdk sky130hd --corner tt_025C_1v80 \
  --liberty /absolute/path/sky130hd-tt.lib --cell-lef /absolute/path/sky130hd.lef \
  --macro-liberty rtl/asic/cache/sky130_sram/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib \
  --macro-lef rtl/asic/cache/sky130_sram/sky130_sram_2kbyte_1rw1r_32x512_8.lef \
  --sram-depth 1024 --out rtl/asic/results/my-bank-run
```

The report verifies the expected macro instance count, separates periphery and
macro footprint, and reports any Liberty/LEF area discrepancy explicitly.
Mixed-process macro/periphery combinations and `--sta` are refused for this
experiment. `check -assert` makes structural check failures fatal.

STA uses ideal inputs, explicit IO delay/output load, and no extracted wire
parasitics. The register-to-register setup bound is clock period minus worst
register setup slack; a missing path remains unavailable. It is not an
achieved clock or a full-chip throughput measurement. Container RSS is
reported separately from the host Docker launcher's RSS. Neither PDK has an
activity-based power or energy/token result from this flow.

For ICSprout55 use the official
[v1.10.102 release](https://github.com/openecos-projects/icsprout55-pdk/releases/tag/v1.10.102)
and matching cell views. Smaller mapped logic is not sufficient to select a
new primary backend: SRAM availability and DRC/LVS/extraction coverage remain
separate gates.

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
The general `block_pnr.tcl` accepts `TECH_LEF`, `CELL_LEF`, and `LIB` through
the environment and stops after placement/CTS with estimated parasitics. Its
historical filename does not mean detailed routing or DRC/LVS was performed.

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

1. Extend the portable leaf evidence to partitions without exceeding resources.
2. Resolve the SRAM cross-port contract and integrate a verified macro strategy.
3. Choose the integration target and replace the pad stub with real IO.
4. Create maintained timing constraints and power intent.
5. Run full-SFU and then full-chip mapping, placement, CTS, routing, and STA.
6. Close DRC/LVS and validate memory timing/behavior.
7. Re-measure performance using the achieved clock and a named memory model.

The shared core source list remains
`rtl/common/filelists/core.f` throughout these steps.
