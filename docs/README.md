# TACCEL documentation index

This index was audited against the code after commit `3486c01` on
2026-09-03. Documents are classified by authority so a dated experiment is
not mistaken for the current architecture.

## Current operational documentation

| Document | Authority |
|---|---|
| [Top-level README](../README.md) | Setup, architecture summary, supported commands, and known gaps |
| [Current project status](project_status.md) | Latest verified state, measurements, limitations, and recent cleanup |
| [Software codebase](../software/CODEBASE.md) | Current Python package map and data flow |
| [ISA specification](../software/docs/isa_spec.md) | Current software/RTL encoding and implementation matrix |
| [ISA freeze record](../software/docs/isa_generation_freeze.md) | Frozen generation decision and dated amendments |
| [RTL testbench guide](../rtl/TESTBENCHES.md) | Supported Verilator and co-simulation gates |
| [Generic RTL elaboration gate](../rtl/synth/BASELINE.md) | Current `sv2v + yosys` hierarchy/check/stat procedure and latest result |
| [ASIC flow](../rtl/asic/README.md) | SKY130 wrapper, timing flows, and physical-closure gaps |
| [FPGA flow](../rtl/fpga/README.md) | Wrapper smoke gate and work needed after board selection |

When current documents disagree, the executable source is decisive:

- ISA software: `software/taccel/isa/`
- RTL ISA legality: `rtl/common/src/decode_unit.sv` and
  `rtl/common/src/include/taccel_pkg.sv`
- Shared RTL source list: `rtl/common/filelists/core.f`
- Test targets: `rtl/verilator/Makefile`
- PTQ defaults: `software/taccel/runtime/stage5_ptq.py`

## Architecture and performance records

These documents capture measured changes that remain useful. Their absolute
throughput values may use older cycle counts or the historical 34.41 MHz
timing proxy. Each file now begins with a current-status banner.

- [Port-A/Port-S bus split](porta_bus_split.md)
- [Phase-0 overlap audit](phase0_measurement.md)
- [T1 overlap items](t1_overlap_items.md)
- [DMA transpose-load](lever_d_dma_transpose.md)
- [SFU fmax cluster](lever_e_fmax_cluster.md)
- [B=32 batching result](lever_h_b32.md)
- [Serving and chunked prefill](lever_i_serving.md)
- [Speculative decoding](lever_b3_specdec.md)
- [SFU timing audit](t0_sfu_fmax_audit.md)
- [T1 measurement redirect](t1_measured_redirect.md)
- [July 16 performance roadmap](perf_roadmap_2026-07-16.md)

For current cycle counts and timing caveats, use
[project status](project_status.md), not a dated report.

## Superseded plans and baselines

The following files are retained for decisions, provenance, and debugging
history. They are not implementation instructions:

- [July 8 performance roadmap](perf_roadmap_2026-07-08.md)
- [July 10 performance roadmap](perf_roadmap_2026-07-10.md)
- [Accelerator completion review](accelerator_completion_review.md)
- [Original decoder ISA plan](llm_isa_plan.md)
- [Original RTL plan](rtl_plan.md)
- [First-divergence debug plan](rtl_debug_plan.md)
- [Bottom-up RTL debugging plan](rtl_debugging_plan.md)
- [Stage-5 readiness baseline](stage5_readiness_2026-04-22.md)
- [Synthesis Phase-2 migration record](../rtl/synth/PHASE2_INTEGRATION.md)
- [Synthesis Phase-3 closeout](../rtl/synth/PHASE3_CLOSEOUT.md)
- [Freeze revision draft/provenance](../software/docs/isa_freeze_revision_109_stub.md)
- [PTQ Phase-A findings](../software/docs/ptq_phase_a_findings.md)
- [W8A32 decision record](../software/docs/w8a32_deployment_scope.md)
- [W8A8 Phase-0 result](../software/docs/w8a8_phase0_ppl.md)

Historical documents may name files and test targets that were later deleted.
Their status banners identify those cases and point back to the current guides.

## Documentation maintenance rules

1. Put current commands and architecture contracts in an operational document.
2. Put benchmark inputs, commit, cycle count, clock assumption, and DRAM model
   in every measurement report.
3. Never silently revise an old measurement. Add a dated banner or addendum.
4. Update this index when adding, superseding, or deleting a document.
5. Validate relative Markdown links and stale file references before merging.
