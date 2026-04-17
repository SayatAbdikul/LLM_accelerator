# TACCEL ISA Specification

## Overview

TACCEL uses a fixed-width 64-bit ISA for a transformer accelerator with three
main execution domains:

- DMA for DRAM and SRAM movement
- a systolic array for integer matmul
- an SFU path for nonlinear and normalization operations

All instructions are 8 bytes wide. Programs are issued in order. Real hardware
may overlap independent units, but the golden model executes instructions
sequentially.

## Architectural Resources

### Buffers

| Buffer | ID | Size | Element type | Typical role |
|---|---:|---:|---|---|
| `ABUF` | `0b00` | `128 KB` | `INT8` | activations |
| `WBUF` | `0b01` | `256 KB` | `INT8` | weights and tables |
| `ACCUM` | `0b10` | `64 KB` | `INT32` | widened accumulators |

All SRAM offsets are expressed in **16-byte units**.

### Scale Registers

`SET_SCALE` loads `FP16` values into scale registers. These scales are widened
to `FP32` during execution and are used by requantization, scaling, and SFU
instructions.

### Address Registers

DRAM pointers are built from two instructions:

- `SET_ADDR_LO` writes the low 28 bits
- `SET_ADDR_HI` writes the high 28 bits

DMA instructions then use an address register plus a 16-byte DRAM offset.

### Tile Configuration

`CONFIG_TILE` sets the active tile shape. The encoded values are tile counts
minus one. The golden model therefore interprets the dimensions as:

- rows = `(M + 1) * 16`
- cols = `(N + 1) * 16`
- reduction = `(K + 1) * 16`

Most compute instructions require a valid tile configuration.

## Execution Model

### Golden Model

The golden model:

- fetches one instruction
- decodes it
- executes it immediately
- increments `PC`

It does not model real pipeline overlap. Functional behavior should match the
ISA, but detailed timing only approximates hardware.

### `SYNC`

In hardware, `SYNC` is a barrier for selected execution resources. In the
golden model it is functionally a no-op because everything already runs in
program order.

### Illegal Instructions

Opcodes `0x14` through `0x1F` are reserved. Decoding them is an illegal
instruction fault. Invalid buffer IDs and missing tile state may also fault.

## Instruction Formats

### R-type

Used for compute-like operations with two sources, one destination, one scale
register selector, and flags.

Instructions:

- `MATMUL`
- `REQUANT`
- `REQUANT_PC`
- `SCALE_MUL`
- `VADD`
- `DEQUANT_ADD`
- `SOFTMAX`
- `SOFTMAX_ATTNV`
- `LAYERNORM`
- `GELU`

Fields:

- `src1_buf`, `src1_off`
- `src2_buf`, `src2_off`
- `dst_buf`, `dst_off`
- `sreg`
- `flags`

### M-type

Used for DMA:

- `LOAD`
- `STORE`

Fields:

- buffer ID
- SRAM offset
- transfer length
- address register
- DRAM offset

`stride_log2` and M-type flags are reserved in the current ISA version.

### B-type

Used for buffer-to-buffer copies:

- `BUF_COPY`

Fields include source and destination buffers, offsets, length, row count, and
transpose control.

### A-type

Used to load address-register halves:

- `SET_ADDR_LO`
- `SET_ADDR_HI`

### C-type

Used for `CONFIG_TILE`.

### S-type

Used for:

- `NOP`
- `HALT`
- `SYNC`
- `SET_SCALE`

## Instruction Reference

### `NOP`

Do nothing for one issue slot.

Use cases:

- alignment
- debugging
- patching a program without changing later addresses

### `HALT`

Stop execution.

Use cases:

- mark the end of a program
- prevent execution from falling into unrelated memory

### `SYNC`

Wait for selected resources to drain before continuing.

Why it matters:

- hardware may overlap DMA, systolic, and SFU work
- consumers sometimes must wait for producers

Golden-model note:

- functionally a no-op

### `CONFIG_TILE`

Set the active tile dimensions for later tile-based instructions.

Why it matters:

- `MATMUL`, `REQUANT`, `SOFTMAX`, `GELU`, and related instructions interpret
  buffer memory through the current tile shape

### `SET_SCALE`

Load an `FP16` scale into a scale register.

Sources:

- immediate `FP16`
- `ABUF`
- `WBUF`
- `ACCUM`

Typical uses:

- activation input scale
- activation output scale
- accumulator-to-output rescale
- dual-scale residual factors

### `SET_ADDR_LO`

Set the low 28 bits of a DRAM address register.

Typical use:

- paired with `SET_ADDR_HI` before `LOAD` or `STORE`

### `SET_ADDR_HI`

Set the high 28 bits of a DRAM address register.

Typical use:

- paired with `SET_ADDR_LO` before `LOAD` or `STORE`

### `LOAD`

Transfer data from DRAM into `ABUF`, `WBUF`, or `ACCUM`.

Conceptually:

- effective DRAM address = address register + DRAM offset
- copy a contiguous span into SRAM

Typical uses:

- load activations
- load weights
- load scale tables
- load strip-mined residual slices

### `STORE`

Transfer data from on-chip SRAM back to DRAM.

Typical uses:

- final outputs
- spilled partial tensors
- debug exports

### `BUF_COPY`

Copy data between on-chip buffers.

Typical uses:

- stage tensors for the next operation
- reshape or reposition tiles
- transpose small tiles
- build head-interleaved layouts

### `MATMUL`

Run the systolic-array multiply-accumulate kernel.

Conceptually:

- inputs are read as `INT8`
- accumulation happens in `INT32`
- output is written to `ACCUM`

Typical uses:

- linear layers
- Q, K, V projection
- attention score computation
- output projection
- MLP `FC1` and `FC2`

Important flag behavior:

- the low flag bit selects accumulate mode
- this allows multiple matmuls to accumulate into one `ACCUM` tile

### `REQUANT`

Convert an `INT32` tile into `INT8` using one scale factor.

Conceptually:

- `dst = clip(round(src * scale), -128, 127)`

Why it matters:

- this is the main boundary where data leaves high-precision accumulator space
  and returns to normal `INT8` activation storage

Golden-model note:

- rounding is round-half-to-even

### `REQUANT_PC`

Convert `INT32` to `INT8` using one scale per output column.

How it works:

- `src1` is the `ACCUM` tile
- `src2` points to a packed `FP16` scale table
- each output column gets its own scale

Why it exists:

- per-channel scaling often fits transformer weights better than one scalar
  scale for the whole tile

Typical uses:

- selective `FC1` and `FC2` output requant
- selective attention projection tuning

### `SCALE_MUL`

Multiply a tile by a scalar while staying in the same storage domain.

Behavior by source type:

- `ACCUM` input gives `INT32 -> INT32`
- `ABUF` input gives `INT8 -> INT8`

Typical uses:

- attention-score scaling
- rescaling without a full requant boundary
- small arithmetic transforms

### `VADD`

Add two `INT8` tiles elementwise and write an `INT8` result.

Typical uses:

- quantized residual adds
- combining already-quantized branches

Important limitation:

- by the time `VADD` runs, both operands have already paid any earlier
  requantization loss

Golden-model note:

- result is saturating `INT8`

### `DEQUANT_ADD`

Add a high-precision accumulator branch and a quantized residual branch in one
instruction, then write the result as `INT8`.

Conceptually:

- dequantize `ACCUM` with one scale
- dequantize `INT8` residual with another scale
- add in floating point
- requantize once to `INT8`

Why it exists:

- it removes the classic `ACCUM -> REQUANT -> VADD` bottleneck

Typical uses:

- residual add after `out_proj`
- residual add after strip-mined `FC2`

### `SOFTMAX`

Apply softmax to a quantized score tile.

Conceptually:

- dequantize input
- compute softmax in floating point
- requantize the output to `INT8`

Why it matters:

- softmax is one of the most accuracy-sensitive nonlinear steps in transformer
  inference

Typical use:

- attention probability generation after `QK^T`

### `SOFTMAX_ATTNV`

Fuse softmax and the subsequent attention-value multiply.

Conceptually:

- dequantize attention scores
- compute softmax in floating point
- multiply by value vectors
- requantize the attention output

Why it exists:

- reduces staging overhead
- avoids materializing a separate explicit softmax-result instruction stream

Typical use:

- fused late-attention experiments

Simulator note:

- it can also expose virtual trace payloads for the internal softmax output

### `LAYERNORM`

Apply layer normalization through the SFU path.

Conceptually:

- dequantize input
- normalize in floating point
- apply the required normalization parameters
- requantize back to `INT8`

Typical uses:

- transformer pre-attention normalization
- transformer pre-MLP normalization
- final normalization before classification

### `GELU`

Apply the Gaussian Error Linear Unit nonlinearity.

Conceptually:

- dequantize input
- evaluate GELU in floating point
- requantize the result to `INT8`

Why it matters:

- GELU is one of the main error amplifiers in quantized transformer inference

Typical use:

- MLP activation after `FC1`

Note:

- higher-precision `GELU-from-ACCUM` style flows are an optimization strategy,
  not a different base opcode

## Practical Programming Notes

### 1. Set tile state first

Most compute instructions assume `CONFIG_TILE` has already been issued.

### 2. Load scales explicitly

Quantized math is only meaningful if the correct scale registers were loaded
before the instruction that consumes them.

### 3. Keep `SYNC` in real programs

The golden model is sequential, but hardware is not. `SYNC` is part of the
correctness contract for hardware scheduling.

### 4. Requantization boundaries dominate accuracy loss

The biggest losses usually happen where data leaves `ACCUM` and returns to
`INT8`. Those boundaries often matter more than the matmul itself.

### 5. Reserved opcodes are faults

`0x14` through `0x1F` are currently invalid. Future extensions must update the
assembler, decoder, simulator, compiler, and this document together.

## Current Implemented Opcode Set

| Opcode | Mnemonic |
|---:|---|
| `0x00` | `NOP` |
| `0x01` | `HALT` |
| `0x02` | `SYNC` |
| `0x03` | `CONFIG_TILE` |
| `0x04` | `SET_SCALE` |
| `0x05` | `SET_ADDR_LO` |
| `0x06` | `SET_ADDR_HI` |
| `0x07` | `LOAD` |
| `0x08` | `STORE` |
| `0x09` | `BUF_COPY` |
| `0x0A` | `MATMUL` |
| `0x0B` | `REQUANT` |
| `0x0C` | `SCALE_MUL` |
| `0x0D` | `VADD` |
| `0x0E` | `SOFTMAX` |
| `0x0F` | `LAYERNORM` |
| `0x10` | `GELU` |
| `0x11` | `REQUANT_PC` |
| `0x12` | `SOFTMAX_ATTNV` |
| `0x13` | `DEQUANT_ADD` |

Reserved:

- `0x14` to `0x1F`
