# Phase 2 — SFU/helper integration status (rolling)

The plan-mandated op-by-op migration of `sfu_engine.sv` (1725 lines) and
`blocking_helper_engine.sv` (1511 lines) from `real`+DPI behavioral to
synthesizable RTL.

## What's done

### gen-2 SFU compute paths (`SFU_SYNTH_MODE=1`)

| Op | Code | Method | Synth band | Status |
|---|---|---|---|---|
| VADD_FP32 | 0x19 | `fp32_add` + `fp32_to_fp16` | 0 ULP | ✅ MIGRATED |
| DEQUANT_ACCUM_FP32 | 0x17 | `fp32_mul` + `fp32_to_fp16` | 0 ULP | ✅ MIGRATED |
| QUANT_FP32_INT8 | 0x18 | `fp32_mul` + `fp32_quantize_i8` | 0 ULP | ✅ MIGRATED |
| DEQUANT_ACCUM_SCALED | 0x1E | 3-stage chain (mul,mul,add) + cvt | 0 ULP | ✅ MIGRATED |
| MAX_ABS_REDUCE SCALE_WR | 0x1F (part 2) | `fp32_div` + `fp32_to_fp16` (fp64→fp32 subst measured 0 ULP) | 0 ULP | ✅ MIGRATED |
| LAYERNORM_FP32 | 0x1A | 5-state sub-FSM (sum→mean→var→denom→out) | 0 ULP byte-exact | ✅ MIGRATED |
| MASKED_SOFTMAX_FP32 | 0x1D | 3-pass sub-FSM (max → exp_sum → out) via `fp32_exp` + `fp32_div` | **0 fp16 ULP byte-exact** on fixture (despite fp32_exp's 86-ULP-max scaffold band — normalization + fp16 quantization absorb the error) | ✅ MIGRATED |
| GELU_FP32 | 0x1B | needs `fp32_gelu_new` | **PHASE 3** (scaffold 882M ULP unusable for ≤3-fp16-ULP fixture) | ⏸ BLOCKED |

### gen-2 SFU latch + reduction paths (`SFU_SYNTH_MODE=1`)

| Path | Method | Status |
|---|---|---|
| `F_G2_S1_LATCH` FP16-src1 | 8-lane `fp16_to_fp32` primitives shared | ✅ MIGRATED |
| `F_G2_S2_LATCH` VADD | 8-lane `fp16_to_fp32` → `attn_accum_q` | ✅ MIGRATED |
| `F_G2_S2_LATCH` LN/SCALED | 8-lane `fp16_to_fp32` → `gamma_q`/`beta_q` | ✅ MIGRATED |
| `F_G2_S1_LATCH` 0x1F running-max | 8-lane abs reduction + max tree | ✅ MIGRATED |
| `F_G2_S1_LATCH` ACCUM-INT32 | `real'(int32)` — already value-preserving | (no change needed) |

### blocking_helper (`HELPER_SYNTH_MODE=1`)

| Op | Method | Status |
|---|---|---|
| `dequant_add_pack` | 16-lane parallel: `i32_to_fp32`+`fp32_mul`+`fp32_add`+`fp32_quantize_i8` | ✅ MIGRATED byte-exact |

### Infrastructure

- `SFU_SYNTH_MODE` parameter (default 0); propagated `taccel_top.sv` → `u_sfu`
- `HELPER_SYNTH_MODE` parameter (default 0); propagated `taccel_top.sv` → `u_helper`
- `real_to_fp32_bits` + `fp32_bits_to_real` functions in `sfu_engine.sv` (lossless transit between `real` storage and fp32 bit-pattern primitives)
- 6-bit enum widening in `sfu_engine.sv` (5→6 bits) for 5 new LN sub-FSM states
- 7-primitive `FP32_PRIMS` bundle in `Makefile` (fp32_add/mul/div/sqrt/to_fp16/i32_to_fp32/quantize_i8/fp16_to_fp32) added to `CONTROL_SV`
- Makefile targets: `test_sfu_synth`, `test_helpers_synth`, `test_sfu_helper_synth`

## Verification (this session, end-state)

| Gate | Result |
|---|---|
| `test_sfu` (default) | **21/21 PASS** |
| `test_sfu_synth` (SFU mode=1) | **21/21 PASS** |
| `test_helpers` (default) | **19/19 PASS** |
| `test_helpers_synth` (HELPER mode=1) | **19/19 PASS** |
| `test_sfu_helper_synth` (BOTH modes=1) | **21/21 PASS** (cross-mode clean) |
| `test_compare_rtl_golden.py` (freeze cosim) | **6 passed + 1 skipped** |
| `test_control` / `test_dma` / `test_decode` / `test_systolic` / `test_systolic_chained` / `test_accum_snapshot_readback` | 42+26+41+8+7+2 = **126/126 PASS** |
| Phase-1 standalone gates (12 of them) | **All PASS** |
| Transcendental scaffolds (`fp32_exp`, `fp32_gelu_new`) | PASS measured-band (Phase 3 tightening pending) |

## What remains for the strictest "Phase 2 done"

The playbook's strict definition (`make synth-check` GREEN; zero DPI/real in
`rtl/src/`) requires multi-session work beyond this session's scope:

1. **SOFTMAX/GELU synth paths** — blocked on Phase 3 minimax-tuning of
   `fp32_exp` (current 86-ULP-max scaffold) and `fp32_gelu_new` (882M-ULP
   scaffold). Wiring them up with current scaffold accuracy would regress the
   fixture's 0-ULP / ≤3-fp16-ULP bands.

2. **Gen-1 INT8 SFU paths** — SOFTMAX/LAYERNORM/GELU on the gen-1 INT8
   codepath, plus ATTN states (gen-1 attention). All `real`+DPI. Lowest
   priority per the playbook; not exercised by the frozen gen-2 bundle.

3. **DPI import removal from `sfu_engine.sv` / `blocking_helper_engine.sv`**
   — once SOFTMAX/GELU + gen-1 are migrated, the DPI imports can be
   `\`ifdef`-wrapped or removed. Currently `make synth-check` is still RED
   at the first DPI import (line ~1769 of the sv2v-converted file) because
   unmigrated ops still reference them.

4. **`real`-typed storage in `sfu_engine.sv`** — the `row_data_q`,
   `attn_accum_q`, `gamma_q`, `beta_q`, etc. are still `real` arrays. For
   yosys synth-check to be truly green, these need to become `logic [31:0]`.
   The synth-mode reads via `real_to_fp32_bits` / writes via
   `fp32_bits_to_real` make the value flow lossless, but the storage type
   itself is non-synth. A storage-type cascade touches every op.

5. **`systolic_*` 2D unpacked array yosys-frontend gap** (from
   `BASELINE.md`) — needs either packed-array refactor or `yosys-slang`.
   Independent of DPI removal.

## Phase-2 substantive completion (what IS true today)

All gen-2 SFU operations that **can** be migrated without Phase-3
transcendental tightening **are** migrated and verified byte-exact, in both
single-mode and cross-mode (SFU+HELPER simultaneously) builds. The DPI path
remains as the cosim-pinned default and is untouched by these migrations.
The freeze cosim safety net is GREEN through all changes. The migration
template (parameter toggle + iter state + module-scope primitives + opcode
mux + `real_to_fp32_bits`/`fp32_bits_to_real` lossless transit) is proven
across 6 SFU ops + 1 helper op + 8 parallel-lane reductions.

The path to full Phase-2 closure is documented; the remaining work is
**Phase-3-prerequisite** (transcendental tightening for SOFTMAX/GELU) or
**multi-session refactor scope** (real-storage cascade + gen-1 + DPI
ifdef-wrapping + yosys-slang/array-pack).
