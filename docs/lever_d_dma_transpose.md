# Lever D — DMA transpose-load (de-serialize the K^T helper pass)

**Goal:** delete the ~9.4M-cycle/step *serial* helper K^T-transpose pass by folding
the transpose into the DMA **load** (a legal systolic-overlap partner). Byte-exact.

## Why the plan's one-liner ("strided-beat DMA write") is wrong

The K^T transpose is **byte-granularity** INT8: golden `dma.py:89-92`
(`reshape(N_pad, d_head).T`) and helper RTL (`blocking_helper_engine.sv`, 16×16
byte scratch + `extract_window`). Each 16-byte DRAM beat holds 16 consecutive `d`
of ONE key position `n`; the transpose must scatter those 16 bytes to 16 DIFFERENT
WBUF rows (column `n` of each). A whole-beat strided write keeps its 16 bytes
contiguous → cannot transpose. So lever D needs a real 16-row-stripe transpose
datapath in the DMA, not an address-generator tweak.

## Mechanism (self-contained, byte-exact by construction)

Per head, replace `kv_load(int8 K) + BufCopy(transpose=1)` with ONE transposed
DMA load: reads the contiguous (N_pad, d_head) INT8 K cache from DRAM, writes the
(d_head, N_pad) transpose into the head's WBUF block.

### Geometry (shifts only — no divider)
Carried in the reserved M-type bits (byte-compatible: all existing loads/stores
keep 0 there):
- `transpose` → M_FLAGS bit[0].
- `cols_log2` → M_STRIDE_LOG2 field [6:3].  `C = 16 << cols_log2` (= d_head; for
  d_head=64 → cols_log2=2).

DMA derives:
- `cols_beats = 1 << cols_log2`      (beats per input row = C/16)
- `stripe_beats = 16 << cols_log2`   (16 input rows)
- `rows_tiles = xfer_len >> (4 + cols_log2)`  (= N_pad/16 = n_stripes = out_row_stride)

### FSM (stripe-by-stripe; one AR burst per stripe, ≤64 beats < 256)
For stripe s in 0..rows_tiles-1:
1. **D_T_AR/D_T_R**: read `stripe_beats` contiguous beats into `tbuf[16][C]`.
   beat k → row `k>>cols_log2`, colgrp `k & (cols_beats-1)`:
   `tbuf[row][colgrp*16 +: 16] = beat`.
2. **D_T_WRITE**: for output row c in 0..C-1, one SRAM beat
   `{tbuf[15][c],…,tbuf[0][c]}` (byte i = tbuf[i][c]) → WBUF row
   `sram_off + c*rows_tiles + s`. Advance dram addr by stripe bytes, next stripe.

**Buffer:** `logic [7:0] tbuf [0:15][0:63]` (16×64 B = 1 KB; MAXC=64 covers d_head≤64).

### OOB reuse
Written rows = { c*rows_tiles + s } = a bijection onto {0..xfer_len-1}. Max row =
sram_off + xfer_len - 1 — identical extent to a normal load, so the existing
DRAM/SRAM prevalidation bounds it. No new bounds logic.

### Pad
Transpose path is gated on `loaded_rows == N_pad` (always true when
max_seq_len % 16 == 0, e.g. GPT-2 block_size=1024, since key_len ≤ max_seq_len ⇒
pad_dim(key_len) ≤ max_seq_len). Otherwise the emitter falls back to
load+BufCopy. So the DMA needs a single geometry param (rows_tiles derived) and
never partial-row output.

## Touch-points
- ISA: `instructions.py` MTypeInsn +transpose/+cols_log2; `encoding.py`
  map to M_FLAGS/M_STRIDE_LOG2 (replace old stride_log2/flags encode);
  `disassembler.py`/`syntax.py` field-name follow.
- Golden: `golden_model/dma.py execute_load` — transpose branch (reshape.T).
- Emit: `emit/dma.py` + `codegen._emit_dma_load` add transpose/cols_log2 kwargs.
- RTL: `taccel_pkg.sv` (m_transpose/m_cols_log2 decoded fields),
  `decode_unit.sv` (extract bits [0],[6:3]), `taccel_top.sv` (wire to u_dma),
  `dma_engine.sv` (transpose FSM + tbuf).
- Emitter: `packed_attn.py` — swap per-head load+BufCopy for one transposed load.

## Gate
golden byte-diff (tiny+124M b1/b16) · RTL==golden byte-match (test_batched_decode)
· zero new test failures · measure b16 tok/s (expect ~10.5-11 from deleting the
serial helper pass).

## Result (measured 2026-07-12, b16 gpt2 pos-510, mode-1 honest-BW, 34.41 MHz)

| metric | baseline f152b07 | lever D | Δ |
|---|---:|---:|---:|
| step cycles | 65,706,735 | 56,299,503 | −9,407,232 (−14.3%) |
| **tok/s** | 8.379 | **9.779** | **+16.7%** |
| sys_busy | 24,636,477 | 24,636,477 | 0 (exact) |
| sfu_busy | 10,056,082 | 10,056,082 | 0 (exact) |
| dma_beats | 19,476,400 | 19,476,400 | 0 (exact) |

The step reduction (9,407,232 cyc) equals the predicted ~9.4M serial helper
K^T-transpose pass; sys/sfu/dma are byte-for-byte invariant — the win is purely
the deleted helper pass (the DMA-transpose read reuses the load DMA beats, and
its strided WBUF write phase is absorbed vs. the strictly-exclusive helper op).

## Validation (2026-07-12)

- **Golden byte-exact (new emitter == baseline, via git-stash A/B):**
  tiny-b16 `1a53237a…`, gpt2-b1 `b882d500…`, gpt2-b16(window-511) `2f37c52c…` —
  all byte-identical. (b1 gpt2 already exercises cols_log2=2 / d_head=64.)
- **Golden transpose == load+BUF_COPY(transpose):** proven byte-identical in
  numpy (transposed-load == K.T for a 512×64 tile).
- **RTL DMA datapath:** test_dma `transpose_load_c16_r32`, `_c32_r16`, `_c64_r32`
  PASS (covers cols_log2 0/1/2); all 26 pre-existing DMA tests still pass.
- **RTL == golden cosim:** `test_batched_decode_rtl_matches_golden_bytes` PASS
  against the rebuilt lever-D binary — the tiny b16 decode emits **128 transpose
  loads** (cols_log2=1), so the cosim genuinely exercises the datapath end-to-end.
- **ISA:** transpose/cols_log2 round-trip; plain loads/stores keep the reserved
  bits 0 (byte-compatible). test_isa_encoding + test_assembler green.

## Build gotcha — soft-fp16 fallback (unrelated unblock, committed with lever D)

`testbench.h`'s fp16 DPI helpers used `_Float16` (GCC 12+/clang 15+). This box has
only g++-9/10 → the RTL cosim harness could not compile. Added a portable
integer soft-float16 fallback (guarded by `__FLT16_MANT_DIG__`/`__FLT16_MAX__`;
native path unchanged on newer compilers), **bit-exact to numpy** — validated over
all 65536 halves + 8.3M float32 + 6.3M float64 incl. every tie case, and
self-checked by the cosim byte-match (golden=numpy vs RTL-DPI=fallback).

Rebuild command (there is no Makefile `run_program_synth` target): the
`run_program` verilator invocation **plus** `-GSFU_SYNTH_MODE=1
-GDRAM_SIZE=1073741824 --Mdir build/run_program_synth`. **DRAM_SIZE MUST be
1<<30 for GPT-2 124M** — the Makefile default (16 MB) fits only the tiny cosim
and faults the 124M bench with FAULT_DRAM_OOB (code 2).
