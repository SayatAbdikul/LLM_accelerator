# TACCEL ISA specification

**Revision:** 2026-09-03
**Applies to:** the Python and RTL tree after `3486c01`
**Instruction width:** 64 bits, serialized big-endian

This is the current implementation reference. The older
[generation-freeze document](isa_generation_freeze.md) records why generation
2 was selected and how the contract was amended.

## Architecture

TACCEL has an in-order control plane and four execution resources:

| Resource | Role | Scheduling |
|---|---|---|
| DMA | DRAM/SRAM LOAD and STORE, transposed LOAD | Asynchronous; synchronized with `SYNC[0]` |
| Systolic | 16×16 INT8 matrix multiply into INT32 ACCUM | Asynchronous; synchronized with `SYNC[1]` |
| SFU | Generation-2 W8A16 dequant, quant, residual, LayerNorm, GELU, and causal softmax | Asynchronous; synchronized with `SYNC[2]` |
| Helper | BUF_COPY and retained generation-1 integer helpers | Blocking in the current control path |

The current scheduler permits DMA and systolic work to overlap. SFU and helper
operations are serialized against other active engines.

### Architectural storage

| Buffer | ID | Capacity | Physical row | Current use |
|---|---:|---:|---:|---|
| ABUF | `0b00` | 128 KiB | 16 bytes | INT8 or FP16/FP32 bytes selected by instruction contract |
| WBUF | `0b01` | 256 KiB | 16 bytes | INT8 weights, packed metadata/parameters |
| ACCUM | `0b10` | 64 KiB | 16 bytes | Four little-endian INT32 values per row |
| Reserved | `0b11` | — | — | Illegal buffer fault |

All SRAM offsets and transfer lengths are expressed in 16-byte units. DRAM
addresses are byte addresses formed from one of four 56-bit address registers
plus `dram_off × 16`.

Other persistent state:

- 16 FP16 scale registers, `S0..S15`;
- four 56-bit DRAM address registers, `R0..R3`;
- `CONFIG_TILE` geometry;
- `CONFIG_ATTN` causal-mask context.

## Opcode matrix

The opcode field is instruction bits `[63:59]`.

| Hex | Name | Format | Python codec/golden | RTL | Purpose |
|---:|---|---|---|---|---|
| 00 | NOP | S | Yes | Yes | No operation |
| 01 | HALT | S | Yes | Yes | Stop execution |
| 02 | SYNC | S | Yes | Yes | Wait for selected asynchronous resources |
| 03 | CONFIG_TILE | C | Yes | Yes, except `weight_int4` | Set tiled dimensions and optional exact SFU row count |
| 04 | SET_SCALE | S | Yes | Immediate mode only | Write one scale register |
| 05 | SET_ADDR_LO | A | Yes | Yes | Write address bits `[27:0]` |
| 06 | SET_ADDR_HI | A | Yes | Yes | Write address bits `[55:28]` |
| 07 | LOAD | M | Yes | Yes | DRAM to SRAM; optional INT8 transpose |
| 08 | STORE | M | Yes | Yes | SRAM to DRAM; transpose fields must be zero |
| 09 | BUF_COPY | B | Yes | Yes | Blocking copy or matrix transpose between SRAM buffers |
| 0A | MATMUL | R | Yes | Yes | INT8×INT8 tiled matrix multiply into INT32 |
| 0B | REQUANT | R | Yes | Yes | Retained scalar requant helper |
| 0C | SCALE_MUL | R | Yes | Yes | Retained integer/scale helper |
| 0D | VADD | R | Yes | Yes | Retained INT8 vector add helper |
| 0E | SOFTMAX | — | Retired | Illegal | Retired generation-1 SFU |
| 0F | LAYERNORM | — | Retired | Illegal | Retired generation-1 SFU |
| 10 | GELU | — | Retired | Illegal | Retired generation-1 SFU |
| 11 | REQUANT_PC | R | Yes | Yes | Per-channel requant helper |
| 12 | SOFTMAX_ATTNV | — | Retired | Illegal | Retired fused generation-1 SFU |
| 13 | DEQUANT_ADD | R | Yes | Yes | Retained dequantized residual helper |
| 14 | CONFIG_ATTN | ATTN | Yes | Yes | Set persistent causal/valid-length context |
| 15 | MASKED_SOFTMAX | — | Retired | Illegal | Retired generation-1 SFU |
| 16 | MASKED_SOFTMAX_ATTNV | — | Retired | Illegal | Retired fused generation-1 SFU |
| 17 | DEQUANT_ACCUM_FP32 | R | Yes | W8A16 only | INT32 ACCUM to FP16/FP32 activation storage |
| 18 | QUANT_FP32_INT8 | R | Yes | W8A16 only | FP16/FP32 activation storage to INT8 |
| 19 | VADD_FP32 | R | Yes | W8A16 only | Elementwise residual add with FP32 internal sum |
| 1A | LAYERNORM_FP32 | R | Yes | W8A16 only | LayerNorm with FP32 internals |
| 1B | GELU_FP32 | R | Yes | W8A16 only | GELU with FP32 internals |
| 1C | SOFTMAX_FP32 | R | Yes | Illegal | Noncausal softmax; no supported RTL consumer |
| 1D | MASKED_SOFTMAX_FP32 | R | Yes | W8A16 only | Causal softmax using `CONFIG_ATTN` |
| 1E | DEQUANT_ACCUM_FP32_SCALED | R | Yes | W8A16 only | Per-channel dequant × activation scale + optional folded bias |
| 1F | MAX_ABS_REDUCE_FP32 | R | Yes | W8A16 only | Derive quant/dequant scale pair from maximum magnitude |

“W8A16 only” means RTL requires R-type `flags[0]=1`. Python also models
`flags[0]=0` as the older FP32-storage/W8A32 form, but RTL reports
`FAULT_UNSUPPORTED_OP` for it.

The six retired opcode names remain in the Python and RTL enums solely to
preserve numeric allocation and produce clear diagnostics. They have no
instruction classes or assembler mnemonics.

## Instruction formats

Bit 63 is the most significant bit. Unlisted payload bits should be encoded
zero unless explicitly noted.

### R-type

Used by MATMUL, retained helper operations, and generation-2 operations.

```text
[63:59] opcode
[58:57] src1_buf
[56:41] src1_off
[40:39] src2_buf
[38:23] src2_off
[22:21] dst_buf
[20:5]  dst_off
[4:1]   sreg
[0]     flags
```

`flags[0]` is opcode-specific:

- MATMUL: `0` clears/restarts the destination accumulation, `1` accumulates
  with existing ACCUM data.
- Generation-2 `0x17..0x1F`: `0` selects FP32 activation storage in Python;
  `1` selects FP16 activation storage and is the only RTL-supported mode.
- Retained helper operations: encode zero unless a helper's implementation
  explicitly defines otherwise.

Offsets are validated against each selected buffer's capacity before Python
encoding. RTL also rejects buffer ID `0b11`.

### M-type

Used by LOAD and STORE.

```text
[63:59] opcode
[58:57] buf_id
[56:41] sram_off
[40:25] xfer_len
[24:23] addr_reg
[22:7]  dram_off
[6:3]   cols_log2
[2:1]   reserved
[0]     transpose
```

Effective DRAM address:

```text
addr_regs[addr_reg] + 16 * dram_off
```

`xfer_len` is the number of 16-byte units. The DMA validates the complete
transfer range before performing it.

For a transposed LOAD (`transpose=1`), DRAM contains a contiguous INT8 matrix
`(R, C)` and SRAM receives `(C, R)`, where `C = 16 << cols_log2`. The transfer
size must match the geometry. Transpose is ignored for STORE in RTL and should
always be encoded zero for STORE. Bits `[2:1]` are currently ignored by Python
and RTL decoders; canonical encoders leave them zero.

### B-type

Used by BUF_COPY.

```text
[63:59] opcode
[58:57] src_buf
[56:41] src_off
[40:39] dst_buf
[38:23] dst_off
[22:7]  length
[6:1]   src_rows
[0]     transpose
```

With `transpose=0`, `length` 16-byte units are copied linearly. With
`transpose=1`, the source is interpreted as a byte matrix with
`src_rows × 16` rows and a column count derived from the total byte length,
then transposed to the destination.

### A-type

Used by SET_ADDR_LO and SET_ADDR_HI.

```text
[63:59] opcode
[58:57] addr_reg
[56:29] imm28
[28:0]  reserved
```

The low and high instructions update their respective half without changing
the other half. Compiler-generated address pairs are adjacent and relocation
code validates both opcodes and register IDs before patching.

### C-type

Used by CONFIG_TILE.

```text
[63:59] opcode
[58:49] M
[48:39] N
[38:29] K
[28]    weight_int4
[27:16] m_exact
[15:0]  reserved
```

`M`, `N`, and `K` are zero-based tile counts; the physical tiled dimensions
are `(field + 1) × 16`.

`m_exact` changes only the SFU row walk:

- `0`: use the padded row count `(M + 1) × 16`;
- nonzero: process exactly `m_exact` rows.

MATMUL and helper operations continue to use the padded tile count.

`weight_int4=1` tells Python bundle/golden code to interpret packed WBUF
weights as signed INT4 nibbles. Current RTL does not decode bit 28 and will
execute the same word as an INT8 MATMUL. Hardware-target tooling must therefore
keep it zero.

Bits `[15:0]` are currently ignored rather than fault-checked; canonical
encoders leave them zero.

### ATTN-type

Used by CONFIG_ATTN.

```text
[63:59] opcode
[58:47] query_row_base
[46:35] valid_kv_len
[34:33] mode
[32:0]  reserved, must be zero
```

The context persists until another CONFIG_ATTN or reset.

Mode bits are predicates applied to candidate key column `c` for query row
`r`:

- bit 1: causal predicate
  `c <= query_row_base + r`;
- bit 0: valid-length predicate
  `c < valid_kv_len`.

Thus `01` is valid-length-only, `10` is pure causal, `11` combines both, and
`00` is invalid. `valid_kv_len` must be nonzero. Mode `10` requires the
configured key width to equal `valid_kv_len`; modes `01` and `11` require the
key width to be at least `valid_kv_len`.

CONFIG_TILE must already be valid. Reserved bits `[32:0]` are checked by both
Python and RTL.

### S-type

NOP and HALT have no payload.

SET_SCALE:

```text
[63:59] opcode
[58:55] sreg
[54:53] src_mode
[52:37] imm16
[36:0]  reserved
```

`src_mode=0` stores the raw FP16 bit pattern in `imm16`. Python can load a
scale from a buffer for modes `1..3`; RTL supports immediate mode only and
reports `FAULT_UNSUPPORTED_OP` otherwise.

SYNC:

```text
[63:59] opcode
[58:56] resource_mask
[55:0]  reserved
```

Resource bits:

| Bit | Resource |
|---:|---|
| 0 | DMA |
| 1 | Systolic |
| 2 | SFU |

The helper engine is blocking and has no separate SYNC bit.

## Compute contracts

### MATMUL

CONFIG_TILE supplies padded `M×N×K` geometry. Inputs are INT8, the product and
accumulation path are INT32, and output resides in ACCUM. The default chained
array has 16×16 processing elements.

### Retained helpers

The retained helper opcodes exist for compatibility and selected internal
paths:

- REQUANT: ACCUM to INT8 using scale registers;
- REQUANT_PC: per-channel scale table;
- SCALE_MUL: apply scalar scale conversion;
- VADD: saturating INT8 elementwise add;
- DEQUANT_ADD: combine accumulator and skip paths through FP32 primitives and
  requantize;
- BUF_COPY: local linear copy or transpose.

They require valid tile configuration where applicable.

### Generation-2 W8A16 operations

For `flags=1`, ABUF activation elements occupy two bytes (FP16 storage), but
LayerNorm variance, GELU, softmax, and scaling arithmetic use FP32 internals.

- DEQUANT_ACCUM_FP32: `int32 × per-channel scale`, then one FP16 cast.
- QUANT_FP32_INT8: FP16 input promoted to FP32, multiplied by a quant scale,
  rounded/clipped to INT8.
- VADD_FP32: inputs promoted to FP32, added, and stored as FP16.
- LAYERNORM_FP32: FP32 mean/variance with epsilon, gamma, and beta; FP16
  storage at the boundary.
- GELU_FP32: pipelined FP32 approximation and FP16 output.
- MASKED_SOFTMAX_FP32: three-pass causal softmax over the CONFIG_ATTN-visible
  prefix; FP16 storage.
- DEQUANT_ACCUM_FP32_SCALED: in W8A16 mode, `src2` holds `2N` FP16 values:
  `N` per-channel scales followed by `N` biases. It computes
  `int32 × pc_scale × activation_scale + bias` and casts once to FP16.
- MAX_ABS_REDUCE_FP32: clamps the maximum magnitude by epsilon and writes
  `127/max_abs` to `sreg` and `max_abs/127` to `sreg+1`.

The exact approximation bands and freeze rationale are preserved in
[the freeze record](isa_generation_freeze.md).

## Fault behavior

RTL halts in a fault state and exposes a four-bit code:

| Code | Name | Meaning |
|---:|---|---|
| 0 | FAULT_NONE | No fault |
| 1 | FAULT_ILLEGAL_OP | Retired/reserved opcode or malformed encoding |
| 2 | FAULT_DRAM_OOB | DRAM transfer outside modeled range |
| 3 | FAULT_SRAM_OOB | SRAM access outside selected buffer |
| 4 | FAULT_NO_CONFIG | Required tile or attention context is absent/inconsistent |
| 5 | FAULT_BAD_BUF | Buffer ID `0b11` |
| 6 | FAULT_UNSUPPORTED_OP | Legal software operation/parameter not implemented in RTL |

Examples of unsupported RTL forms are generation-2 `flags=0` and non-immediate
SET_SCALE. `SOFTMAX_FP32` is classified as illegal, not unsupported.

Python raises typed/value errors instead of reproducing the exact RTL fault
state, but the legality boundary is kept aligned for supported hardware
programs.

## Program memory layout

`ProgramBinary` stores one instruction stream and optional data/metadata.
`ProgramBundle` stores prefill and decode streams plus a single aligned shared
data region, temporary region, logits region, and KV-cache region. Runtime
relocations patch adjacent SET_ADDR_LO/HI pairs and CONFIG_ATTN words.

Instructions occupy eight bytes in the file image. PCs are instruction
indices, not byte offsets. DRAM symbols and patch values are byte addresses.

## Conformance

The current conformance surfaces are:

- `software/tests/test_isa_encoding.py`;
- `software/tests/test_assembler.py`;
- `rtl/verilator/test_decode.cpp`;
- `rtl/verilator/test_control.cpp`;
- generation-2 fixtures under `rtl/verilator/fixtures/gen2/`;
- `software/tests/test_compare_rtl_golden.py`;
- `software/tests/test_batched_decode.py`.

Changes to opcode allocation, field meaning, rounding, or the golden simulator
require a dated amendment to the freeze record, synchronized Python/RTL edits,
fixture regeneration, and explicit review of the golden content hash.
