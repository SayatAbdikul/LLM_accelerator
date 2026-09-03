# Generic RTL synthesizability gate

**Current result:** green on 2026-09-02/03 after `3486c01`
**Command:** `make -C rtl/verilator synth-check`
**Top:** `taccel_top`

This is the current shared-core `sv2v + yosys` elaboration gate. It proves that
the complete target-neutral RTL hierarchy parses without DPI/`real`
dependencies and has no unresolved modules. It is deliberately not a mapped
area/timing flow.

## Latest observed result

| Metric | Value |
|---|---:|
| Exit code | 0 |
| Yosys | 0.65 |
| Wall-reported Yosys time | 83.24 s |
| Peak memory | 1.93 GB |
| Hierarchy cells | 54,998 |
| Wires | 221,956 |
| Wire bits | 4,906,697 |
| Memories | 3 |
| Memory bits | 3,670,016 |
| Logfile hash | `c7b93d94cb` |

The run emits many known messages from generated casts, FP constant
conversion, and memory lowering. Exit status, hierarchy resolution, and the
structural check are the gate. Do not treat warning count as a stable metric
across Yosys versions.

## What runs

The Verilator Makefile:

1. reads `rtl/common/filelists/core.f`;
2. expands paths relative to `rtl/common/src/`;
3. invokes `sv2v` with `SFU_SYNTH_NO_DPI`;
4. writes `rtl/verilator/build/synth/design_full.v`;
5. asks Yosys to read that Verilog and run
   `rtl/synth/synth_check.ys`.

Equivalent outline:

```sh
sv2v -DSFU_SYNTH_NO_DPI \
  -Irtl/common/src/include \
  -Irtl/common/src/systolic \
  <sources from rtl/common/filelists/core.f> \
  -w rtl/verilator/build/synth/design_full.v

yosys -p "read_verilog rtl/verilator/build/synth/design_full.v; \
  script rtl/synth/synth_check.ys"
```

The Yosys script currently performs:

```text
hierarchy -check -top taccel_top
check
stat
```

It does not run `proc`, `flatten`, technology mapping, ABC, placement, routing,
or timing analysis. Earlier documentation called this “full synthesis”; the
accurate description is full-design elaboration/synthesizability checking.

## Definition of green

The target is green when:

- every source in `core.f` converts through `sv2v`;
- Yosys resolves the entire `taccel_top` hierarchy;
- the synth configuration contains no reachable DPI imports or `real` state;
- `check` completes without a fatal structural error;
- the command exits zero.

`SFU_SYNTH_NO_DPI` removes reference-only DPI code. The top-level synthesis
configuration selects the synthesizable SFU and helper datapaths.

## Source-of-truth rule

`rtl/common/filelists/core.f` is shared by Verilator, FPGA, and ASIC builds.
Adding a compilation unit requires one filelist update. Included `.svh` files
are tracked separately as Make prerequisites so editing them rebuilds the
generated Verilog and test binaries.

## Removed partial gate

The old `synth-check-ctrl` target excluded the SFU/helper and supplied
black-box stubs. Once the complete hierarchy became green, that partial gate
stopped proving anything stronger and drifted from the real interface. It,
`core_ctrl.f`, `blackbox_stubs.v`, and `synth_check_ctrl.ys` were removed in
`3486c01`.

## Relationship to ASIC and FPGA flows

- `make -C rtl/asic yosys-asic` elaborates `taccel_top_asic` with the SKY130
  wrapper and SRAM macro boundary.
- `make -C rtl/fpga yosys-fpga` elaborates `taccel_top_fpga` with inferred
  memory and platform stubs.
- `rtl/asic/build/synth_blocks/` contains technology-mapped SKY130 synthesis
  and OpenSTA scripts.
- `rtl/asic/build/openroad/` contains direct per-block placement/routing
  scripts.

Use the ASIC scripts for cell mapping or timing conclusions. The generic gate
does not establish fmax or physical area.

## Historical context

The gate became green during the May 2026 SFU/helper migration:

- FP state moved from host `real` values to 32-bit IEEE-754 bit patterns.
- Synthesizable add/multiply/divide/sqrt/exp/GELU/conversion primitives were
  introduced.
- DPI imports and host-only helpers were excluded from synthesis builds.
- unpacked array constructs were rewritten for `sv2v/Yosys` compatibility.

The detailed migration ledger is preserved in
[`PHASE2_INTEGRATION.md`](PHASE2_INTEGRATION.md) and
[`PHASE3_CLOSEOUT.md`](PHASE3_CLOSEOUT.md). Their old cell counts and removed
partial-gate commands are historical only.
