# ISA generation freeze — generation 2

**Decision locked:** 2026-05-15
**Contract status:** effective and implemented for the causal W8A16 target
**Last documentation reconciliation:** 2026-09-03

This record freezes the maintained hardware-target ISA generation and records later compatible
amendments. The exact current field layout and support matrix live in
[`isa_spec.md`](isa_spec.md).

## Frozen decision

The maintained hardware-target architecture is generation 2:

- INT8 weights and INT8 systolic inputs;
- INT32 systolic accumulation;
- FP16 inter-layer activation storage;
- FP32 internal dequantization, residual, LayerNorm, GELU, softmax, and scale
  arithmetic;
- dynamic per-matmul activation scaling;
- causal attention configured by `CONFIG_ATTN`.

Generation 2 replaced the old generation-1 INT8-I/O SFU path. Reverting to
generation 1 would invalidate the maintained W8A16 compiler, dynamic scaling,
and the validated quality path.

## Normative hardware opcode set

The frozen GPT-2 causal target requires:

### Infrastructure

- `HALT`, `SYNC`;
- `CONFIG_TILE`, `CONFIG_ATTN`, `SET_SCALE`;
- `SET_ADDR_LO`, `SET_ADDR_HI`;
- `LOAD`, `STORE`, `BUF_COPY`;
- `MATMUL`.

`NOP` remains a legal conventional no-op although compiler bundles do not
normally need it.

### Generation-2 operations

| Opcode | Name | Hardware contract |
|---:|---|---|
| `0x17` | DEQUANT_ACCUM_FP32 | INT32 ACCUM to FP16 activation storage |
| `0x18` | QUANT_FP32_INT8 | FP16 activation storage to INT8 |
| `0x19` | VADD_FP32 | FP32-internal residual add, FP16 storage |
| `0x1A` | LAYERNORM_FP32 | FP32-internal LayerNorm, FP16 storage |
| `0x1B` | GELU_FP32 | FP32-internal GELU, FP16 storage |
| `0x1D` | MASKED_SOFTMAX_FP32 | Causal softmax using `CONFIG_ATTN` |
| `0x1E` | DEQUANT_ACCUM_FP32_SCALED | Per-channel dequant × dynamic activation scale + folded bias |
| `0x1F` | MAX_ABS_REDUCE_FP32 | Produce quant/dequant scale pair |

All eight operations require R-type `flags[0]=1` in RTL. The older
FP32-storage form (`flags[0]=0`) remains modeled in Python for research but is
not part of the hardware target.

`SOFTMAX_FP32` (`0x1C`) is not normative. Python retains the class and golden
implementation for noncausal experimentation, but RTL treats it as
`FAULT_ILLEGAL_OP`. Supported GPT-2 graphs emit `MASKED_SOFTMAX_FP32`.

## Retired and compatibility opcodes

### Retired generation-1 SFU

These six opcodes are permanently retired:

- `0x0E SOFTMAX`;
- `0x0F LAYERNORM`;
- `0x10 GELU`;
- `0x12 SOFTMAX_ATTNV`;
- `0x15 MASKED_SOFTMAX`;
- `0x16 MASKED_SOFTMAX_ATTNV`.

RTL has rejected them since 2026-05-23. As of `3486c01`, Python instruction
classes, assembler syntax, compiler emission, golden execution, and legacy
tests are also removed. Their enum names remain only to preserve numeric
allocation and diagnose old binaries.

### Retained helper compatibility

The following generation-1 helper opcodes remain legal in RTL:

- `0x0B REQUANT`;
- `0x0C SCALE_MUL`;
- `0x0D VADD`;
- `0x11 REQUANT_PC`;
- `0x13 DEQUANT_ADD`.

They are non-normative for the primary W8A16 bundle but remain implemented by
`blocking_helper_engine.sv` for compatibility and selected diagnostics.

## Frozen numeric and layout contracts

### Instruction and addressing

- Instructions are fixed at 64 bits and serialized big-endian.
- The opcode is bits `[63:59]`.
- SRAM offsets and lengths use 16-byte units.
- Four address registers hold 56-bit DRAM byte addresses.
- ABUF/WBUF/ACCUM are 128/256/64 KiB respectively.

### W8A16 flag

For generation-2 R-type operations:

- `flags=1`: FP16 activation storage, FP32 internal arithmetic; normative RTL.
- `flags=0`: FP32 activation storage; Python research path only.

For `DEQUANT_ACCUM_FP32_SCALED` with `flags=1`, `src2` contains `2N` FP16
values: `N` per-channel scales followed by `N` biases. The epilogue computes

```text
int32_accum * per_channel_scale * activation_scale + bias
```

before one FP16 cast. This avoids an intermediate FP16 rounding step.

### Dynamic scale pair

`MAX_ABS_REDUCE_FP32` writes:

```text
S[sreg]   = 127 / max(max_abs, epsilon)
S[sreg+1] = max(max_abs, epsilon) / 127
```

The first scale feeds quantization; the second restores real units after the
next matmul.

### Causal mask

`CONFIG_ATTN` stores `query_row_base`, `valid_kv_len`, and two mode bits.
`MASKED_SOFTMAX_FP32` applies the selected causal and valid-length predicates.
This context is mandatory and persistent until replaced or reset.

### Exact SFU row count

The 2026-07-10 `m_exact` amendment assigns `CONFIG_TILE[27:16]`:

- zero preserves the original padded `(M+1)×16` row count;
- nonzero gives the exact SFU row count.

MATMUL and helper engines continue to use padded tile geometry. The default
zero encoding preserves old words byte-for-byte.

### Transposed LOAD

The DMA transpose amendment assigns M-type `cols_log2[6:3]` and
`transpose[0]`. A transposed LOAD reads contiguous INT8 `(R,C)` data and writes
`(C,R)` to SRAM, with `C = 16 << cols_log2`. Existing plain transfers encode
both fields as zero.

### INT4 research extension

Python assigns `CONFIG_TILE[28]` to `weight_int4` and implements packing plus
golden execution. This is not frozen hardware functionality: RTL does not
decode bit 28. Hardware-target bundles must leave it zero until a separate
hardware revision is approved.

## Conformance policy

### Golden-model pin

The authoritative content hash of
`software/taccel/golden_model/simulator.py` is:

```text
cc1bc64f34bf7b5a53a5760bcd500dca10cb8080
```

`software/tests/test_compare_rtl_golden.py::test_frozen_golden_sha_pin`
recomputes the Git blob hash and fails on drift. A simulator edit requires:

1. a dated amendment to this record;
2. an intentional pin update;
3. generation-2 fixture regeneration;
4. a report of whether raw fixture payloads changed.

The 2026-09 retired-op cleanup changed the simulator blob but regenerated all
ten generation-2 cases with zero raw payload drift.

### Tiny-model conformance

The tiny frozen decoder is the byte-exact RTL-versus-golden gate. The active
tests are:

- `software/tests/test_compare_rtl_golden.py`;
- `software/tests/test_batched_decode.py`;
- generation-2 vectors under `rtl/verilator/fixtures/gen2/`.

### GPT-2 124M conformance

Whole-program byte matching is not meaningful after the first FP16 non-finite
boundary in the large model. Large-model sign-off therefore uses:

- deterministic RTL-versus-RTL logits hashes for schedule/overlap changes;
- argmax and logits metrics;
- perplexity against the named preset and dataset window.

A report must name the checkpoint, preset, token window, context, RTL mode,
clock assumption, and DRAM model.

### Approximation bands

The synthesizable SFU uses deterministic FP32 primitives and FP16 storage.
Primitive and end-to-end gates establish operation-specific tolerances rather
than requiring every internal FP32 intermediate to match a host math library
bit-for-bit. Pipelined variants must be bit-identical to their combinational
parent where their test states that contract.

Current primitive gates cover divide, square root, `fp32_exp_p18`, and
`fp32_gelu_p33`. The active generation-2 fixture cases cover dequant,
dequant-scaled, VADD, LayerNorm, GELU, masked softmax, quantization, and
MAX_ABS reduction.

## Implementation status

The original RTL reconciliation work is complete:

- all eight normative generation-2 opcodes decode and execute in W8A16 mode;
- mode-1 SFU/helper paths are synthesizable without DPI imports;
- full shared-core `sv2v + yosys` elaboration is green;
- the long exponential, GELU, scale divide, dequant, and LayerNorm chains have
  been pipelined;
- tiny RTL-versus-golden conformance and batched decode gates are active.

Remaining work is not a generation-freeze blocker:

- full-chip physical timing and SRAM/IO integration;
- the `0x1C` software/RTL scope decision;
- the INT4 hardware-scope decision;
- optional large-model and board/tape-out closure.

## Revision log

| Date | Revision |
|---|---|
| 2026-05-15 | Generation-2 decision locked, including dynamic-scale opcodes `0x1E` and `0x1F` |
| 2026-05-16 | Operation-specific conformance/ULP policy recorded |
| 2026-05-17 | Non-finite QUANT behavior made explicit |
| 2026-05-19 | Logits-level large-model metric accepted; freeze dependencies committed |
| 2026-05-23 | Six generation-1 SFU opcodes made illegal in RTL; helper opcodes retained |
| 2026-05-24 | INT4 software/golden extension added outside the normative RTL target |
| 2026-07-08 | INT4 tile metadata reflected in the golden simulator pin |
| 2026-07-10 | `m_exact` and transposed-LOAD amendments landed |
| 2026-07-21..25 | Long SFU/helper paths pipelined without changing numerical outputs |
| 2026-09-03 | Retired generation-1 Python execution and obsolete verification paths removed; golden pin updated with zero generation-2 payload drift |

## Change procedure

Any future opcode or field change must:

1. update this record and [`isa_spec.md`](isa_spec.md);
2. update Python opcode, instruction, and codec definitions;
3. update RTL package, decode, control, and implementing engine;
4. add positive, boundary, reserved-field, and fault tests;
5. regenerate frozen fixtures;
6. review the golden hash;
7. run software, Verilator, co-simulation, and synthesis gates.

## References

- Current field and support matrix: [`isa_spec.md`](isa_spec.md)
- Current project status: [`../../docs/project_status.md`](../../docs/project_status.md)
- Python ISA: `software/taccel/isa/`
- RTL ISA constants: `rtl/common/src/include/taccel_pkg.sv`
- RTL legality: `rtl/common/src/decode_unit.sv`
- Golden pin test: `software/tests/test_compare_rtl_golden.py`
- Fixture generator: `software/tools/gen_gen2_fixtures.py`
