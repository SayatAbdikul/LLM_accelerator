# W8A32 deployment scope

Companion to `/Users/sayat/.claude/plans/w8a32-weights-only-plan.md` (Phase 2).
Phase 1 (commit `cf59efb`) wired a first-class `weight_only_int8` preset
through `evaluate_gpt2_perplexity` that reproduces the diagnostic's
**53.4212 PPL** on `gpt2_converted_nanogpt.pt` at 257-tok / 256-ctx via
`NanoGPTFP32Reference + INT8-QDQ weights` — purely in numpy on the host.
The compiled bundle + simulator path (the "golden model") is unchanged
and remains W8A8 at the same window (6,174 PPL).

This document records what code/ISA work is needed to make the **golden
model** itself produce W8A32 inference, and asks an explicit decision
between three options.

## What "the golden model is W8A32" requires

The deployed bundle today (built by `software/taccel/compiler/decoder_bundle.py`,
executed by `software/taccel/runtime/host_runner.py:HostRunner` + the
`Simulator`) is W8A8 by construction:

- `software/taccel/isa/opcodes.py:90-92` — every SRAM buffer is fixed dtype:

  | Buffer | Size | dtype |
  |---|---:|---|
  | `BUF_ABUF` (activation buffer) | 128 KB | **INT8** |
  | `BUF_WBUF` (weight buffer) | 256 KB | **INT8** |
  | `BUF_ACCUM` (accumulator) | 64 KB | **INT32** |

- `software/taccel/isa/instructions.py:52-110` — every ALU / SFU instruction is
  INT8-I/O typed: `MatmulInsn`, `RequantInsn`, `RequantPcInsn` (INT32→INT8),
  `ScaleMulInsn`, `VaddInsn` (INT8), `SoftmaxInsn` / `MaskedSoftmaxInsn` /
  `LayernormInsn` / `GeluInsn` (INT8 I/O), `DequantAddInsn` (FP16 scale
  registers internally but INT8 in / INT8 out), `SoftmaxAttnVInsn`.
- `software/taccel/isa/opcodes.py:10` is explicit: "SFU : SOFTMAX / LAYERNORM
  / GELU (FP32 datapath, INT8 I/O)".
- `software/taccel/golden_model/simulator.py:90-92, 568, 624, 647-697`
  hard-codes the ABUF/WBUF/ACCUM dtype assumptions on every dispatch.

There is **no FP32 ABUF / FP32 op / FP32 buffer ID** in the current ISA.
Inter-layer activation storage is always INT8, every matmul output is
clipped back to INT8 via `RequantInsn` or `RequantPcInsn`, and every
sub-layer op consumes INT8 and produces INT8. This is the structural
mismatch that produces the W8A8 6,174 PPL number documented in
`software/docs/ptq_phase_a_findings.md`.

Any W8A32 deployment has to give matmul-output and inter-layer activations
a path that is *not* INT8. The three viable options below differ in
where they put that path.

## Three deployment options

### (b) Host-runtime FP32 with INT8 weight storage — memory-only win

- **Idea**: ship a bundle that stores weights as INT8 + per-channel scales
  (4× smaller than FP32 on disk and in DRAM), but bypass the INT8 MXU at
  runtime. Decompress weights to FP32 at load time (or just-in-time
  per-matmul), run inference entirely in FP32 on the host CPU/GPU using
  `NanoGPTFP32Reference`.
- **Files to add**:
  - `software/taccel/runtime/weight_only_host_runner.py` (~80 LOC) — a
    `HostRunner`-API-compatible runner wrapping
    `build_weight_only_int8_reference` from
    `software/taccel/runtime/fp32_reference.py`. Exposes
    `run_prefill(token_ids)` and `run_decode_step(token_id, position)`
    so callers can dispatch via the same interface they use for
    `HostRunner`.
- **Files to change**:
  - `software/taccel/runtime/gpt2_perplexity.py` — `evaluate_gpt2_perplexity`'s
    W8A32 branch (Phase 1) currently sets `golden_perplexity = NaN`. With
    option (b) it instead calls `WeightOnlyHostRunner` to produce
    `golden_logits` in addition to the existing fake-quant call.
    `relative_delta` becomes ~0 because both paths wrap the same
    numpy reference, but the contract (the "golden" path is a separate
    construction that *mimics deployment* — a runner that takes only
    weights + scales) is honest.
- **What it preserves**: 4× weight memory compression on disk and in DRAM,
  identical PPL to Phase 1 (~53.42).
- **What it gives up**: the INT8 MXU is not used at all. All matmul +
  LN + softmax + GELU + residual ops run in FP32 on the host. No
  accelerator throughput benefit.
- **Effort**: 1 week (~80 LOC for the runner + plumbing + tests). Fits
  inside a single session for the minimal version; a polished bundle
  format with INT8-only on-disk storage adds ~1 more week.
- **Verification**: Phase 1 slow gate gains a `golden_perplexity` field
  populated by the runner; `relative_delta ≈ 0`; `golden_perplexity ≈
  53.42` at 257-tok / 256-ctx.

### (c.1) ABUF dtype-mode FP32 — INT8 MXU preserved + FP32 inter-layer

- **Idea**: extend the ISA to support an "FP32 mode" on ABUF (and ACCUM
  dequant). The MXU stays INT8 × INT8 → INT32 (its main throughput
  win), but a new `DequantAccumToFp32Insn` writes the per-channel
  dequant of the INT32 accumulator into ABUF marked as FP32-mode. New
  FP32-mode ops (`VaddFp32`, `LayernormFp32`, `GeluFp32`, `SoftmaxFp32`)
  consume FP32 ABUF and produce FP32 ABUF; a new `QuantFp32ToInt8Insn`
  dynamically quantises a per-matmul activation tile to INT8 just before
  feeding the MXU. This matches what AWQ / llama.cpp / vLLM actually
  deploy in production: "W8 + dynamic A8 inside matmul + A32 between
  matmuls."
- **Files to change**:

  | File | What changes |
  |---|---|
  | `software/taccel/isa/opcodes.py` | New buffer dtype bit (or a per-row dtype tag), 5 new opcodes (the 4 FP32 sub-layer ops + `DequantAccumToFp32`, + `QuantFp32ToInt8`) |
  | `software/taccel/isa/instructions.py` | New instruction classes mirroring the new opcodes |
  | `software/taccel/golden_model/simulator.py` | New dispatch arms; ABUF FP32-mode storage + reads |
  | `software/taccel/compiler/codegen.py` | New REQUANT_PC_FP32 emission point on matmul output; new FP32 lowering for LN / GELU / softmax / VADD when the preset requests W8A32 |
  | `software/taccel/compiler/decoder_bundle.py` | Pass through the W8A32 emission flag |
  | `software/taccel/runtime/stage5_ptq.py` | A real W8A32 codegen preset (vs Phase 1's runner-only preset) |
  | `software/taccel/runtime/host_runner.py` | No change for runtime contract (still takes a `ProgramBundle`) but bundle now emits FP32 ops |
- **What it preserves**: INT8 MXU compute throughput; INT8 weight storage;
  KV cache compression (keys/values still INT8 if desired); roughly the
  full Phase 1 PPL win on the accelerator.
- **What it gives up**: SRAM pressure on ABUF roughly doubles for the
  segments stored FP32 (4× the bytes per activation). This is
  manageable for small models but tightens working-set constraints.
- **Effort**: 3–4 weeks. ISA bit budget audit, new instruction encodings,
  simulator extensions for FP32 ABUF semantics, codegen lowering for
  the new ops on every relevant graph node, end-to-end tests for the
  new emission patterns, and the migration of `evaluate_gpt2_perplexity`
  to dispatch to the new path.
- **Verification target**: golden W8A32 `fake_quant_ppl ≈ 53.5`,
  `relative_delta ≤ 0.02` against the runner-only Phase 1 number.

### (c.2) Sideband FP32 buffer — INT8 path unchanged, new buffer in parallel

- **Idea**: same goals as (c.1), but instead of adding a dtype mode to
  ABUF, add a new buffer type `BUF_FP32_SIDE` parallel to ABUF. The
  INT8 path is completely unchanged; FP32 ops live only in the new
  buffer. This is less invasive on the existing W8A8 codegen at the
  cost of additional SRAM.
- **Files to change**: superset of (c.1) — same instruction / simulator
  / codegen / bundle work, plus a new buffer ID, SRAM allocator
  changes (`software/taccel/compiler/memory_alloc.py`), and golden
  model state for the new buffer.
- **What it preserves**: all of (c.1)'s wins, plus zero risk of
  regressing existing W8A8 paths since they don't touch the new
  buffer.
- **What it gives up**: more total SRAM (a separate buffer that's
  active when W8A32 is, idle otherwise; or always active and
  permanently consuming budget).
- **Effort**: 4–5 weeks. (c.1) plus the buffer-introduction
  housekeeping.
- **Verification target**: same as (c.1).

## Side-by-side decision matrix

| Capability | (stop after Phase 1) | (b) host FP32 + INT8 storage | (c.1) ABUF dtype-mode | (c.2) sideband FP32 buffer |
|---|:---:|:---:|:---:|:---:|
| Phase 1 W8A32 PPL number through standard slow gate | ✅ | ✅ | ✅ | ✅ |
| `golden_perplexity` populated for the W8A32 preset | ❌ NaN | ✅ ~53.42 | ✅ ~53.5 | ✅ ~53.5 |
| INT8 weight storage compression on disk / DRAM | ❌ (FP32 checkpoint) | ✅ 4× | ✅ 4× | ✅ 4× |
| INT8 MXU compute used | ❌ | ❌ | ✅ | ✅ |
| Existing W8A8 codegen path unchanged | ✅ | ✅ | ⚠ shares files | ✅ |
| SRAM pressure | (host RAM, not SRAM) | (host RAM) | medium (ABUF wider) | high (extra buffer) |
| Effort (best case) | 0 (shipped) | ~1 week | ~3–4 weeks | ~4–5 weeks |

## Decision required

**Phase 3 cannot start until you pick one of these.** The plan was explicit
about this gate, and the options span a 5× difference in scope.

> **(stop)** / **(b)** / **(c.1)** / **(c.2)**?

Recommendation: **(c.1)** matches the strategic context (the Phase A doc's
Priority 1 was "per-channel ACTIVATION quant for matmul outputs," which
is structurally what (c.1) implements with dynamic per-matmul scaling).
But the user picks, not the doc.

If your goal is "demonstrate 53 PPL on a real accelerator run" — that's
(c.1). If your goal is "deliver the smallest deployable artifact that
records the W8A32 number through the standard gate" — that's (b). If
your goal was just "see the number" — Phase 1 already delivered that.

## Cross-references

- Phase 1 implementation: commit `cf59efb`, files
  `software/taccel/runtime/fp32_reference.py:build_weight_only_int8_reference`,
  `software/taccel/runtime/gpt2_perplexity.py:run_weight_only_int8_teacher_forced_logits`,
  `software/taccel/runtime/stage5_ptq.py` (`weight_only_int8` preset),
  `software/tests/test_weight_only_int8_perplexity.py`.
- Phase 1 artifact: `software/logs/w8a32/weight_only_int8_257tok.json`
  (`fake_quant_perplexity = 53.4212` PPL at 257-tok / 256-ctx).
- Phase A campaign context:
  `software/docs/ptq_phase_a_findings.md` (the W8A8 6,174 PPL ceiling
  and the two-failure-mode verdict).
- Plan file: `/Users/sayat/.claude/plans/w8a32-weights-only-plan.md`.
