# RTL First-Divergence Debug Plan

## Current State

- The end-to-end `baseline_default` compiler flow still fails RTL-vs-golden comparison.
- The new Phase F debug harness is working and already localizes the first bad traced tensor.
- The current first divergence is:
  - node: `pos_embed_add`
  - trace PC: `13`
  - event index: `0`
  - first differing element: row `1`, col `0`
  - golden raw value: `9`
  - RTL raw value: `20`
- The RTL run still:
  - retires the same instruction count as golden
  - halts cleanly
  - reports no architectural fault
  - reports no forbidden engine-overlap violation

## Goals

- Localize the `pos_embed_add` mismatch to one of:
  - host/runtime input placement
  - compiler-emitted early tensor preparation
  - blocking-helper execution of the local add path
- Avoid spending time on later blocks until the first divergence moves past `pos_embed_add`.
- Turn the current full-program failure into one or more small deterministic regressions.

## Scope

This debug pass is intentionally narrow.

In scope:
- folded/unfolded runtime input placement
- CLS row handling
- position-embedding row placement
- early tensor preparation before `pos_embed_add`
- `VADD` / helper-engine execution for the `pos_embed_add` step

Out of scope until the first divergence moves:
- SFU (`SOFTMAX`, `LAYERNORM`, `GELU`, `SOFTMAX_ATTNV`)
- chained systolic dataflow
- experimental Phase E ops
- later transformer blocks

## Working Hypotheses

### H1. Runtime placement mismatch

The mismatch appears at row `1`, not row `0`, which strongly suggests an early row-placement issue rather than a late numerical drift.

Highest-probability subcases:
- folded CLS / position-embedding placement mismatch between software and RTL runner
- patch rows written at the right DRAM address but reconstructed into SRAM with an off-by-one row interpretation
- patch rows and position-embedding rows aligned differently between host and program expectations

### H2. Helper-path row addressing mismatch

If both inputs feeding `pos_embed_add` match golden but the output does not, then the likely culprit is the helper path:
- wrong source offset
- wrong destination offset
- row `0` / row `1` crossing bug
- incorrect byte/row stepping for the early `VADD`

### H3. Compiler trace placement is correct, but upstream event coverage is too coarse

The current harness finds the first bad output node. To isolate the bug faster, we need traces for the two inputs that feed `pos_embed_add`, not only the output itself.

## Implementation Plan

### 1. Make the early path fully observable

- Extend the compiler trace manifest for the baseline path so the following are traced explicitly:
  - patch-embedding output before position-embedding add
  - patch position-embedding tensor as consumed by the program
  - CLS row if folded/unfolded logic affects the same buffer region
  - `pos_embed_add` output
- Keep using `ProgramBinary.trace_manifest` as the single source of truth.
- Do not add a second debug mapping.

### 2. Compare the inputs to `pos_embed_add` separately

- Use the existing first-divergence harness to capture raw RTL tensors for the two `pos_embed_add` inputs.
- Compare those inputs against golden before comparing the add output.
- Decision rule:
  - if one input is already wrong, debug placement/prep first
  - if both inputs match and output differs, debug helper execution

### 3. Audit runtime placement first

- Re-check the RTL runner and software runtime path for:
  - input patch placement
  - folded CLS behavior
  - folded position-embedding behavior
  - row ordering assumptions for patch rows vs CLS row
- Validate that:
  - row `0` is the CLS row when expected
  - row `1` is the first patch row in both software and RTL flows
  - no host-side folding step shifts only the patch region or only the positional rows

### 4. If inputs are clean, debug helper execution

- Focus on the `VADD` path in the blocking helper engine.
- Check:
  - source buffer IDs
  - source offsets
  - destination offsets
  - row/column iteration order
  - row-boundary behavior between row `0` and row `1`
  - saturating-add behavior vs numpy reference
- Do not investigate SFU or systolic blocks until this check fails to explain the mismatch.

### 5. Add focused regressions

- Add one minimal synthetic regression for the `pos_embed_add` shape:
  - at least rows `0` and `1`
  - width `192`
  - known patch row values and known position-embedding row values
  - check output rows independently
- Add one regression that isolates row `1` only, since that is where the first mismatch appears.
- Add one folded/unfolded variant if runtime placement remains a suspect after the first comparison pass.

## Debug Workflow

### Step A. Rerun baseline with expanded early-node tracing

Expected result:
- either the first bad node moves before `pos_embed_add`
- or the harness proves both `pos_embed_add` inputs are correct and the add output is wrong

### Step B. Branch on the first bad input/output

If patch input is wrong:
- debug host/runtime placement and patch-row ordering

If position-embedding input is wrong:
- debug position-embedding placement and fold logic

If both inputs are right but output is wrong:
- debug helper `VADD`

### Step C. Capture the smallest reproducer

- Once the branch is known, create the smallest native Verilator regression that reproduces the bad row/column.
- Only keep the full baseline run as the acceptance test, not as the primary development loop.

## Exit Criteria

This debug phase is complete when all of the following are true:

- the first-divergence report no longer points to `pos_embed_add`
- a focused regression exists for the bug that was found
- the baseline program is rerun and the first bad node moves later or disappears entirely

## Non-Goals

- Do not redesign the SFU.
- Do not rework chained systolic unless the first divergence moves there later.
- Do not treat final-logit mismatch as the primary debugging surface anymore.
- Do not add public debug registers or change the ISA.
