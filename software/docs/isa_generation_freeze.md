# ISA Generation Freeze — gen-2 (W8A32 Phase 3 (c.1))

**Status: FREEZE DECISION LOCKED (2026-05-15). Becomes the effective repo
baseline on commit of the §5 dependencies** — the generation decision is
fixed and final; the repo is not yet the frozen baseline only because the
spec files in §5 are still uncommitted.

Frozen generation: **gen-2** — the FP32 sub-layer / dynamic-per-matmul-scale
generation introduced by W8A32 Phase 3 **(c.1)**, **including the in-flight
M2.5-A additions** (0x1E DEQUANT_ACCUM_FP32_SCALED, 0x1F MAX_ABS_REDUCE_FP32)
currently uncommitted in the §5 files.

This document is the normative ISA contract the RTL must implement. Until a
superseding freeze revision is published, the opcode set below is fixed; any
ISA change requires a new dated revision of this file.

## 1. Decision record

- **What:** lock the ISA to gen-2 (FP32 sub-layer ops 0x17–0x1F + the
  still-emitted gen-1 infrastructure ops). Do **not** revert to gen-1.
- **Basis:** `w8a32_deployment_scope.md` recommended option **(c.1)**;
  commits `47141fb` (Phase 3 (c.1) M1: ISA extension + simulator dispatch)
  and `5babbaa` (M2: codegen lowering for sub-layer ops) implement it; the
  toolchain and golden model already emit/dispatch gen-2. User confirmed
  "Lock gen-2 (c.1)" on 2026-05-15.
- **Why not gen-1:** reverting would undo committed Phase 3 (c.1) work and
  risk the activation-range accuracy the FP32 sub-layer path was added for
  (dynamic per-matmul scaling, `MAX_ABS_REDUCE_FP32`).

## 2. Normative opcode set (RTL MUST implement exactly these)

Ground truth = empirical histogram of a compiled GPT-2 124M bundle
(prefill + decode), **both** `weight_only_int8` and
`weight_only_int8_quarot` presets — **byte-identical** (QuaRot is data-free
weight prep, zero ISA surface). 19 opcodes are emitted:

| Opcode | Name | Gen | RTL today |
|---|---|---|---|
| 0x01 | HALT | infra | implemented |
| 0x02 | SYNC | infra | implemented |
| 0x03 | CONFIG_TILE | infra | implemented |
| 0x04 | SET_SCALE | infra | implemented |
| 0x05 | SET_ADDR_LO | infra | implemented |
| 0x06 | SET_ADDR_HI | infra | implemented |
| 0x07 | LOAD | infra | implemented |
| 0x08 | STORE | infra | implemented |
| 0x09 | BUF_COPY | infra | implemented |
| 0x0A | MATMUL | infra | implemented |
| 0x14 | CONFIG_ATTN | infra | implemented |
| 0x17 | DEQUANT_ACCUM_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x18 | QUANT_FP32_INT8 | **gen-2** | **MISSING — reserved/illegal** |
| 0x19 | VADD_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1A | LAYERNORM_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1B | GELU_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1D | MASKED_SOFTMAX_FP32 | **gen-2** | **MISSING — reserved/illegal** |
| 0x1E | DEQUANT_ACCUM_FP32_SCALED | **gen-2** | **MISSING — reserved/illegal** |
| 0x1F | MAX_ABS_REDUCE_FP32 | **gen-2** | **MISSING — reserved/illegal** |

Per-bundle emission counts (prefill, GPT-2 124M, indicative of hotness):
SYNC 11765, SET_ADDR_LO/HI 11061, BUF_COPY 8977, LOAD 6656, STORE 4405,
CONFIG_TILE 3000, MATMUL 1254, QUANT_FP32_INT8 1177, DEQUANT_ACCUM_FP32_SCALED
771, MAX_ABS_REDUCE_FP32 601, SET_SCALE 576, DEQUANT_ACCUM_FP32 288,
CONFIG_ATTN 144, MASKED_SOFTMAX_FP32 144, VADD_FP32 145, GELU_FP32 36,
LAYERNORM_FP32 25, HALT 1.

## 3. Non-normative for the gen-2 RTL target

These 13 opcodes are **not emitted** by the gen-2 toolchain. The opcode enum,
assemblers (`isa/instructions.py`), and golden model keep them for
back-compat / non-causal models, but **the gen-2 RTL is NOT required to
implement them** and MAY treat them as illegal:

- **Superseded gen-1 sub-layer ops** (replaced by their FP32 analogue):
  0x0B REQUANT, 0x0C SCALE_MUL, 0x0D VADD (→0x19), 0x0F LAYERNORM (→0x1A),
  0x10 GELU (→0x1B), 0x11 REQUANT_PC, 0x13 DEQUANT_ADD,
  0x15 MASKED_SOFTMAX (→0x1D).
- **Non-causal only** (no current frontend; GPT-2 is causal):
  0x0E SOFTMAX, 0x12 SOFTMAX_ATTNV, 0x1C SOFTMAX_FP32.
- **Unused fused-causal variant** (toolchain chose unfused 0x1D + separate
  attn·V matmul; no FP32 fused analogue defined): 0x16 MASKED_SOFTMAX_ATTNV.
- **Retained no-op:** 0x00 NOP — keep as a defined decode-to-no-op
  (conventional, zero RTL cost); not emitted but not deprecated.

Note 0x1C SOFTMAX_FP32 is gen-2-numbered but **non-normative** (no causal-LM
consumer) — RTL need not implement it; the normative gen-2 additions are
0x17–0x1B and 0x1D–0x1F (**8 opcodes**, not 9).

## 4. RTL reconciliation requirement (the blocking work item)

The RTL implements gen-1 (0x00–0x16); `taccel_pkg.sv:45` and
`decode_unit.sv:32` declare **0x17–0x1F = "reserved — illegal instruction
fault"**. A current bundle therefore illegal-faults on the first gen-2 op.
To make golden-vs-RTL cosim possible on the production path:

1. **Implement the 8 normative gen-2 opcodes in RTL:** 0x17, 0x18, 0x19,
   0x1A, 0x1B, 0x1D, 0x1E, 0x1F. Remove them from the reserved/illegal
   range in `taccel_pkg.sv` / `decode_unit.sv`. (0x1C stays reserved.)
2. **Validate the 11 already-implemented normative ops** (§2 "implemented")
   in cosim — no new RTL, but they must pass against the gen-2 golden model.
3. The SFU "all internal ops FP32" contract (`sfu_engine.sv:18-19`) is
   consistent with gen-2; the gen-2 ops are FP32-I/O, so the change is the
   instruction-level decode/datapath, not the internal precision model.
4. Area note (unmeasured): the gen-2 FP32 sub-layer datapath is expected to
   cost more SFU area than the gen-1 ops it supersedes. No synthesis number
   exists. This is acknowledged, not blocking the freeze, but should be
   measured before RTL sign-off.
5. **Conformance / definition of done.** Gen-2 RTL is conformant when
   `software/tests/test_compare_rtl_golden.py` passes an **end-to-end**
   (not block-level) byte-match within FP16 ULP on the GPT-2 W8A16
   teacher-forced reference bundle (`weight_only_int8_quarot`, 257-tok)
   against the pinned golden model below. Block-level cosim greens are
   necessary but **not** sufficient — the freeze is not satisfied until the
   end-to-end bundle byte-matches.
6. **Pinned reference golden.** RTL conformance is measured against
   `software/taccel/golden_model/simulator.py` **at the commit created by
   the §5 commit**: record here `frozen_golden_sha = <to be filled in on
   the §5 commit>`. Any later `simulator.py` change requires a new freeze
   revision — otherwise "cosim vs the gen-2 golden" is a moving target,
   the exact failure this freeze exists to close.

## 5. Actions required to *complete* the freeze (owner: user — I do not commit)

The freeze is not real until the in-flight spec is committed. As of the
session-start snapshot these are uncommitted-modified and MUST be committed
as the frozen baseline:

- `software/taccel/isa/opcodes.py`
- `software/taccel/isa/instructions.py`
- `software/taccel/isa/encoding.py`
- `software/taccel/golden_model/simulator.py`
- `software/tests/test_compare_rtl_golden.py`

After commit, any opcode change requires a new dated revision of this file
and re-opens the gen-2 RTL contract.

## 6. Cross-references

- Decision basis: `software/docs/w8a32_deployment_scope.md` (§"Decision
  required", recommends (c.1)).
- ISA opcode definitions: `software/taccel/isa/opcodes.py:86` (`Opcode`).
- RTL opcode table: `rtl/src/include/taccel_pkg.sv:22-45`.
- Cosim harness / debug state: `docs/rtl_debug_plan.md` (current first
  divergence ~`pos_embed_add`, gen-1 — predates this freeze).
- Empirical probe: `software/tools/isa_coverage_probe.py` (re-runnable;
  compiles GPT-2 via `build_stage3_tiny_decoder_bundle`, histograms
  `codegen.instructions`; basis for §2).
