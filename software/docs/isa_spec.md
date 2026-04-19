# TACCEL ISA v1.1 Specification

This document tracks the ISA as implemented in the current software stack
(`taccel/isa`, assembler/disassembler, golden-model simulator, compiler, and
tests), not just the original baseline design.

## 1. Architecture Overview

TACCEL is a DNN inference accelerator with three parallel execution units
behind an in-order issue stage:

| Unit     | Instructions                               | Data Types                   |
|----------|--------------------------------------------|------------------------------|
| DMA      | LOAD, STORE                                | byte stream                  |
| Systolic | MATMUL                                     | INT8 in, INT32 out           |
| SFU      | SOFTMAX, MASKED_SOFTMAX, SOFTMAX_ATTNV, MASKED_SOFTMAX_ATTNV, LAYERNORM, GELU | INT8/INT32 in, FP32 internal |

Additional control/data instructions (`REQUANT`, `REQUANT_PC`, `VADD`,
`DEQUANT_ADD`, `SCALE_MUL`, `BUF_COPY`, `CONFIG_TILE`, `SET_SCALE`,
`SET_ADDR_LO/HI`, `SYNC`, `NOP`, `HALT`) execute in the issue stage.

### SRAM Buffers

| Buffer | Size    | Type  | Max Offset (16-byte units) |
|--------|---------|-------|---------------------------|
| ABUF   | 128 KB  | INT8  | 8191                      |
| WBUF   | 256 KB  | INT8  | 16383                     |
| ACCUM  |  64 KB  | INT32 | 4095                      |

Buffer ID encoding: ABUF=0b00, WBUF=0b01, ACCUM=0b10, 0b11=reserved (fault).

### DRAM

- Byte-addressable, 16-byte aligned transfers.
- 4 address registers, each 56 bits (set via SET_ADDR_LO + SET_ADDR_HI).
- Out-of-bounds access (LOAD or STORE) raises a fault; no dynamic growth.

### Byte Ordering

- **Data in SRAM/DRAM**: little-endian.
  - ACCUM INT32: byte 0 = LSB.
  - FP16 scale parameters: byte 0 = LSB (IEEE 754 binary16).
- **Instruction encoding**: big-endian 64-bit words.

---

## 2. Instruction Encoding

All instructions are 64 bits, big-endian.  Bits [63:59] hold the 5-bit opcode.

### 2.1 Opcode Table

| Mnemonic     | Opcode | Format | Description                          |
|--------------|--------|--------|--------------------------------------|
| NOP          | 0x00   | S-TYPE | No operation                         |
| HALT         | 0x01   | S-TYPE | Stop execution                       |
| SYNC         | 0x02   | S-TYPE | Barrier on resource mask             |
| CONFIG_TILE  | 0x03   | C-TYPE | Set tile dimensions (M, N, K)        |
| SET_SCALE    | 0x04   | S-TYPE | Load FP16 into scale register        |
| SET_ADDR_LO  | 0x05   | A-TYPE | Set addr_reg bits [27:0]             |
| SET_ADDR_HI  | 0x06   | A-TYPE | Set addr_reg bits [55:28]            |
| LOAD         | 0x07   | M-TYPE | DMA: DRAM → SRAM                    |
| STORE        | 0x08   | M-TYPE | DMA: SRAM → DRAM                    |
| BUF_COPY     | 0x09   | B-TYPE | SRAM ↔ SRAM (optional transpose)    |
| MATMUL       | 0x0A   | R-TYPE | INT8×INT8 → INT32 tiled matmul       |
| REQUANT      | 0x0B   | R-TYPE | INT32 → INT8 (scale + round + clip)  |
| SCALE_MUL    | 0x0C   | R-TYPE | Multiply tile by FP16 scale          |
| VADD         | 0x0D   | R-TYPE | Element-wise add (INT8 sat / INT32)  |
| SOFTMAX      | 0x0E   | R-TYPE | Row-wise softmax (FP32 internal)     |
| LAYERNORM    | 0x0F   | R-TYPE | Layer normalization (FP32 internal)  |
| GELU         | 0x10   | R-TYPE | GELU activation (FP32 internal)      |
| REQUANT_PC   | 0x11   | R-TYPE | INT32 → INT8 using per-column FP16 scales |
| SOFTMAX_ATTNV | 0x12   | R-TYPE | Fused softmax(QK^T) @ V             |
| DEQUANT_ADD  | 0x13   | R-TYPE | FP32 rescale-add of ACCUM and INT8 skip |
| CONFIG_ATTN  | 0x14   | ATTN-TYPE | Set masked-attention context       |
| MASKED_SOFTMAX | 0x15 | R-TYPE | Row-wise softmax with attention mask |
| MASKED_SOFTMAX_ATTNV | 0x16 | R-TYPE | Fused masked softmax(QK^T) @ V |
| 0x17–0x1F    | —      | —      | **Reserved** — illegal instruction fault |

### 2.2 R-TYPE (Compute)

```
[63:59]  Opcode      (5 bits)
[58:57]  SRC1_BUF    (2 bits)
[56:41]  SRC1_OFF    (16 bits, in 16-byte units)
[40:39]  SRC2_BUF    (2 bits)
[38:23]  SRC2_OFF    (16 bits)
[22:21]  DST_BUF     (2 bits)
[20:5]   DST_OFF     (16 bits)
[4:1]    SREG        (4 bits, scale register index 0–15)
[0:0]    FLAGS       (1 bit)
```

FLAGS semantics per opcode:
- MATMUL: 0=overwrite ACCUM, 1=accumulate into ACCUM.
- All others: reserved, must be 0.

SREG usage by opcode:
- `REQUANT`, `SCALE_MUL`: `S[sreg]` only.
- `SOFTMAX`, `LAYERNORM`, `GELU`, `DEQUANT_ADD`: `S[sreg]` and `S[sreg+1]`.
- `SOFTMAX_ATTNV`: `S[sreg]` through `S[sreg+3]`.
- `REQUANT_PC`: current implementation ignores `sreg`; encode `sreg=0` for
  portability and read scales from `src2`.

### 2.3 M-TYPE (Memory)

```
[63:59]  Opcode      (5 bits)
[58:57]  BUF_ID      (2 bits)
[56:41]  SRAM_OFF    (16 bits, in 16-byte units)
[40:25]  XFER_LEN    (16 bits, in 16-byte units)
[24:23]  ADDR_REG    (2 bits, address register 0–3)
[22:7]   DRAM_OFF    (16 bits, in 16-byte units)
[6:3]    STRIDE_LOG2 (4 bits) — **RESERVED, must be 0**
[2:0]    FLAGS       (3 bits) — **RESERVED, must be 0**
```

**Effective DRAM byte address** = `addr_regs[ADDR_REG] + DRAM_OFF × 16`

Transfer is contiguous.  There is no strided or scatter/gather mode.

### 2.4 B-TYPE (Buffer Copy)

```
[63:59]  Opcode      (5 bits)
[58:57]  SRC_BUF     (2 bits)
[56:41]  SRC_OFF     (16 bits)
[40:39]  DST_BUF     (2 bits)
[38:23]  DST_OFF     (16 bits)
[22:7]   LENGTH      (16 bits, total transfer in 16-byte units)
[6:1]    SRC_ROWS    (6 bits, source row count in 16-row units)
[0:0]    TRANSPOSE   (1 bit, 0=flat, 1=transpose)
```

When TRANSPOSE=1: source is read as `[SRC_ROWS×16, cols]` and written as
`[cols, SRC_ROWS×16]`, where `cols = LENGTH×16 / (SRC_ROWS×16)`.

### 2.5 A-TYPE (Address)

```
[63:59]  Opcode      (5 bits)
[58:57]  ADDR_REG    (2 bits, 0–3)
[56:29]  IMM28       (28 bits)
[28:0]   Unused
```

SET_ADDR_LO sets bits [27:0] of the address register.
SET_ADDR_HI sets bits [55:28] of the address register.
Full 56-bit address = `(HI_imm28 << 28) | LO_imm28`.

### 2.6 C-TYPE (Config)

```
[63:59]  Opcode      (5 bits)
[58:49]  M           (10 bits, 0-based: actual tiles = M+1)
[48:39]  N           (10 bits)
[38:29]  K           (10 bits)
[28:0]   Unused
```

Tile dimensions in 16-element units.  Persists until next CONFIG_TILE.
Must be set before any tiled compute instruction (`MATMUL`, `REQUANT`,
`REQUANT_PC`, `SCALE_MUL`, `VADD`, `DEQUANT_ADD`, `SOFTMAX`,
`MASKED_SOFTMAX`, `SOFTMAX_ATTNV`, `MASKED_SOFTMAX_ATTNV`, `LAYERNORM`,
`GELU`).

### 2.7 ATTN-TYPE (Attention Context)

`CONFIG_ATTN` is an opcode-specific format, not C-TYPE reuse.

```
[63:59]  Opcode = 0x14
[58:47]  QUERY_ROW_BASE   (12 bits)
[46:35]  VALID_KV_LEN     (12 bits)
[34:33]  MODE             (2 bits: 01=padded, 10=causal, 11=causal+padded)
[32:0]   Reserved, must be 0
```

Mode `0b00` is accepted by encode/decode for diagnostic roundtrips but faults
when executed. `VALID_KV_LEN` must be nonzero. The context persists until the
next `CONFIG_ATTN` or reset, and masked softmax instructions fault without a
valid context.

### 2.8 S-TYPE (System)

**SET_SCALE:**
```
[63:59]  Opcode      (5 bits)
[58:55]  SREG        (4 bits, 0–15)
[54:53]  SRC_MODE    (2 bits: 0=imm, 1=ABUF, 2=WBUF, 3=ACCUM)
[52:37]  IMM16       (16 bits: FP16 immediate or buffer offset)
[36:0]   Unused
```

**SYNC:**
```
[63:59]  Opcode      (5 bits)
[58:56]  RESOURCE_MASK (3 bits: bit0=DMA, bit1=Systolic, bit2=SFU)
[55:0]   Unused
```

**NOP / HALT:** No payload fields.

---

## 3. Execution Model

### 3.1 Pipeline

Instructions are issued in order.  Three execution units may operate in
parallel:

```
Issue → ┬→ DMA engine     (LOAD, STORE)
        ├→ Systolic array  (MATMUL)
        └→ SFU             (SOFTMAX, SOFTMAX_ATTNV, LAYERNORM, GELU)
```

Other instructions (`REQUANT`, `REQUANT_PC`, `VADD`, `DEQUANT_ADD`,
`SCALE_MUL`, `BUF_COPY`, `CONFIG_TILE`, `SET_SCALE`, `SET_ADDR`) execute in
the issue stage and complete before the next instruction issues.

### 3.2 Synchronization

SYNC stalls the issue stage until the selected units have drained:

| Mask  | Units waited       |
|-------|--------------------|
| 0b001 | DMA                |
| 0b010 | Systolic           |
| 0b100 | SFU                |
| 0b111 | All                |

Without SYNC, the hardware may reorder or overlap operations across
different units.  SYNC with mask 0b000 is a NOP.

### 3.3 Fault Handling

| Condition                    | Behaviour                        |
|------------------------------|----------------------------------|
| Illegal opcode (0x14–0x1F)  | Halt, set fault status register  |
| DRAM out of bounds           | Halt, set fault status register  |
| SRAM out of bounds           | Halt, set fault status register  |
| Missing CONFIG_TILE          | Halt, set fault status register  |
| Reserved buffer ID (0b11)   | Halt, set fault status register  |

---

## 4. Systolic Array

16×16 element systolic array.  INT8 inputs, INT32 accumulators.

```
C[i][j] += Σ_k A[i][k] × B[k][j]
```

- Tile size: 16 × 16 elements.
- Throughput: 16 cycles per tile.
- Total cycles: `(M+1) × (N+1) × (K+1) × 16`.
- FLAGS[0]=0: overwrite ACCUM.  FLAGS[0]=1: accumulate.

### Overflow

Maximum accumulator magnitude per instruction:
`M_dim × K_dim × 127² ≈ M_dim × K_dim × 16,129`

For DeiT-tiny: max(M×K) = 208 × 192 = 39,936 → max |acc| ≈ 644M.
Fits in INT32 (±2.1G).  The compiler must tile to prevent overflow.

### 4.1 Quantization and Residual Helper Instructions

All helper paths use round-half-to-even before clipping.

**REQUANT**

```
dst_int8 = clip(round(src1_int32 × S[sreg]), -128, 127)
```

**REQUANT_PC**

```
scale[j] = FP16(src2[j]) for j in 0..N_dim-1
dst[:, j] = clip(round(src1_int32[:, j] × scale[j]), -128, 127)
```

- `src1` must be `ACCUM`.
- `src2` points at `N_dim` packed FP16 scales in `ABUF` or `WBUF`.
- The same scale row is reused across all `M_dim` rows.
- `dst` must be an INT8 buffer (`ABUF` or `WBUF`).

**SCALE_MUL**

```
if src1 is ACCUM:
    dst_int32 = clip(round(src1_int32 × S[sreg]), INT32_MIN, INT32_MAX)
else:
    dst_int8 = clip(round(src1_int8 × S[sreg]), -128, 127)
```

**VADD**

Current implementation supports two modes:

```
INT8 mode:  dst = sat_int8(src1_int8 + src2_int8)
INT32 mode: dst = src1_int32 + broadcast_row(src2_int32)
```

- INT8 mode is selected when `src1_buf = ABUF`.
- INT32 mode is selected when `src1_buf = ACCUM`; `src2` is read as a single
  `1 × N_dim` row and broadcast across the tile.

**DEQUANT_ADD**

```
accum_rescale = S[sreg]
skip_rescale  = S[sreg+1]
dst_int8 = clip(round(src1_accum_int32 × accum_rescale +
                      src2_skip_int8  × skip_rescale), -128, 127)
```

- `src1` must be `ACCUM`.
- `src2` must be an INT8 SRAM buffer.
- `dst` must be an INT8 SRAM buffer.
- This is the implemented residual-add primitive used to avoid a separate
  `REQUANT` followed by `VADD`.

---

## 5. Special Function Unit (SFU)

SFU operations run with FP32 internal arithmetic and FP16 scale registers
widened to FP32.

Scale-register conventions:
- `SOFTMAX`, `LAYERNORM`, `GELU`: dual-scale form using `S[sreg]` and
  `S[sreg+1]`.
- `SOFTMAX_ATTNV` and `MASKED_SOFTMAX_ATTNV`: quad-scale form using
  `S[sreg]..S[sreg+3]`.

Source/destination conventions:
- `SOFTMAX`, `MASKED_SOFTMAX`, and `GELU` may read either INT8 SRAM or raw
  INT32 `ACCUM`.
- `LAYERNORM` reads INT8 activations plus FP16 `gamma/beta` parameters from
  `src2`.
- `SOFTMAX_ATTNV` and `MASKED_SOFTMAX_ATTNV` read `QK^T` from `ACCUM` and `V`
  from INT8 SRAM, and write the quantized `attn @ V` output to INT8 SRAM.

### 5.1 Rounding Convention

**Round-half-to-even** (IEEE 754 default / banker's rounding) for all
quantization paths: REQUANT, SCALE_MUL, and all SFU output requantization.
RTL must implement this mode.

### 5.2 SOFTMAX

Per-row softmax along the logical reduction dimension of the configured tile:

```
x_fp32 = input × in_scale           # input may be INT8 SRAM or INT32 ACCUM
x_shifted = x_fp32 - max(x_fp32, axis=N)
softmax = exp(x_shifted) / sum(exp(x_shifted))
output = clip(round(softmax / out_scale), -128, 127)
```

Row-max subtraction is required for numerical stability.

### 5.3 MASKED_SOFTMAX

`MASKED_SOFTMAX` follows the same datapath as `SOFTMAX`, but applies the active
`CONFIG_ATTN` mask before row-max and exponentiation:

- Causal mask hides columns `j > query_row_base + local_row_i`.
- Padded mask hides columns `j >= valid_kv_len`.
- Hidden logits are treated as `-inf`; their output probability is zero.
- Fully masked rows fault in the golden model.

For `MASKED_SOFTMAX`, key columns are `(CONFIG_TILE.N + 1) * 16`.

### 5.4 SOFTMAX_ATTNV

Fused attention tail:

```
qkt = ACCUM_int32 × S[sreg]         # qkt_in_scale
v   = V_int8 × S[sreg+1]            # v_scale
softmax = softmax_rows(qkt)
attn_v = softmax @ v
output = clip(round(attn_v / S[sreg+2]), -128, 127)
```

`S[sreg+3]` is used by the golden model as a trace-only softmax scale so the
software stack can materialize a virtual INT8 softmax tensor for diagnostics.
That virtual trace payload is not architectural state and does not affect the
visible destination buffer.

`MASKED_SOFTMAX_ATTNV` applies the same `CONFIG_ATTN` mask to `qkt` before
softmax. For `MASKED_SOFTMAX_ATTNV`, key columns are `(CONFIG_TILE.K + 1) * 16`.

### 5.5 LAYERNORM

Per-row normalize + affine transform:

```
x_fp32 = INT8_input × in_scale
x_norm = (x_fp32 - mean(x_fp32)) / sqrt(var(x_fp32) + 1e-6)
y = x_norm × gamma + beta          (gamma/beta are FP16, widened to FP32)
output = clip(round(y / out_scale), -128, 127)
```

Gamma and beta are packed in WBUF at src2: N×2 bytes gamma, then N×2 bytes beta.

### 5.6 GELU

```
GELU(x) = x × 0.5 × (1 + erf(x / sqrt(2)))
```

RTL implements erf via the Abramowitz & Stegun 7.1.26 polynomial:

```
erf(x) ≈ sign(x) × (1 - (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵) × exp(-x²))
where t = 1 / (1 + 0.3275911 × |x|)

a1 =  0.254829592
a2 = -0.284496736
a3 =  1.421413741
a4 = -1.453152027
a5 =  1.061405429
```

Max absolute error in FP32: ~5×10⁻⁷ (well below INT8 noise floor ~4×10⁻³).
Hardware cost: 5 FMA + 1 exp + 1 reciprocal per element.

---

## 6. Unified DRAM Memory Layout

The host loads a single contiguous binary image into DRAM:

```
Offset 0x000000  ┌─────────────────────────────┐
                 │  Instructions               │  PC fetches from DRAM[PC×8]
                 ├─────────────────────────────┤  aligned to 16 bytes
                 │  Model Parameters            │  Interleaved: each weight matrix
                 │  (INT8 weights, FP16 scales, │  followed by its scale array,
                 │   INT32 biases, FP16 LN γ/β, │  then biases, LN params,
                 │   INT8 CLS/pos_embed,        │  CLS token, pos_embed, zero-pad,
                 │   input patches placeholder)  │  input patches region
                 ├─────────────────────────────┤
                 │  Temp (strip-mine spill)     │  FC1/FC2 intermediates
                 └─────────────────────────────┘
```

Key offsets stored in ProgramBinary:
- `data_base`: byte offset of parameters (16-byte aligned after instructions).
- `input_offset`: byte offset where host writes INT8 input patches before run.

All SET_ADDR_LO immediates are DRAM-absolute (patched by the compiler to
include `data_base`).

---

## 7. Scale Registers

16 × FP16 registers (S0–S15).  Loaded via SET_SCALE.

Register usage in the current ISA:
- Single-register ops: `REQUANT`, `SCALE_MUL`
- Dual-register ops: `SOFTMAX`, `MASKED_SOFTMAX`, `LAYERNORM`, `GELU`, `DEQUANT_ADD`
- Quad-register ops: `SOFTMAX_ATTNV`, `MASKED_SOFTMAX_ATTNV`
- Memory-sourced scales: `REQUANT_PC`

Constraints:
- Dual-register ops require `sreg <= 14`
- Quad-register ops require `sreg <= 12`

FP16 → FP32 widening: the exact FP16 value is preserved (no extra precision).

---

## 8. Cycle Model

| Instruction   | Cycles                                    |
|---------------|-------------------------------------------|
| NOP           | 1                                         |
| SYNC          | 1 (+ stall time in RTL)                   |
| LOAD/STORE    | xfer_len (1 cycle per 16-byte unit)       |
| BUF_COPY      | length (1 cycle per 16-byte unit)         |
| MATMUL        | m_tiles × n_tiles × k_tiles × 16         |
| REQUANT       | M × N                                     |
| REQUANT_PC    | M × N                                     |
| SCALE_MUL     | M × N                                     |
| VADD          | M × N                                     |
| DEQUANT_ADD   | M × N                                     |
| SOFTMAX       | M × N × 2                                |
| MASKED_SOFTMAX | M × N × 2                               |
| SOFTMAX_ATTNV | M × K × 2 + m_tiles × n_tiles × k_tiles × 16 + M × N |
| MASKED_SOFTMAX_ATTNV | M × K × 2 + m_tiles × n_tiles × k_tiles × 16 + M × N |
| LAYERNORM     | M × N × 2                                |
| GELU          | M × N × 2                                |
| Others        | 0 (issue-stage only)                      |

Here `M`, `N`, and `K` denote realized element counts
(`M = (tile_M + 1) × 16`, etc.), while `m_tiles`, `n_tiles`, and `k_tiles`
denote tile counts.
