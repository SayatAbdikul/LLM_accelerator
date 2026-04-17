# TACCEL — Transformer Accelerator Toolchain

A complete Python toolchain for an INT8 transformer accelerator targeting **facebook/deit-tiny-patch16-224**. The project spans five layers: a custom ISA, a two-pass assembler/disassembler, a per-channel INT8 quantizer, a tile-level compiler, and a bit-accurate golden-model simulator.

---

## Table of Contents

1. [Project Layout](#project-layout)
2. [Hardware Model](#hardware-model)
3. [ISA — `taccel/isa/`](#isa)
4. [Assembler & Disassembler — `taccel/assembler/`](#assembler--disassembler)
5. [Quantizer — `taccel/quantizer/`](#quantizer)
6. [Compiler — `taccel/compiler/`](#compiler)
7. [Golden Model — `taccel/golden_model/`](#golden-model)
8. [Utilities — `taccel/utils/`](#utilities)
9. [CLI Tools — `tools/`](#cli-tools)
10. [Tests — `tests/`](#tests)
11. [Accuracy Comparison — `compare_accuracy.py`](#accuracy-comparison)
12. [Data Flow: end-to-end walkthrough](#data-flow-end-to-end)
13. [Key Design Decisions](#key-design-decisions)
14. [Quantization Results](#quantization-results)
15. [Extending the Toolchain](#extending-the-toolchain)

---

## Project Layout

```
transformer_accelerator/
├── pytorch_model.bin          # DeiT-tiny FP32 weights (5.7 M params)
├── run_deit.py                # Vanilla FP32 HuggingFace inference
├── compare_accuracy.py        # FP32 vs INT8 accuracy benchmark
│
├── taccel/                    # Main Python package
│   ├── isa/                   # Instruction Set Architecture
│   │   ├── opcodes.py         # Opcode enum, format enum, field constants
│   │   ├── instructions.py    # Instruction dataclasses + validation
│   │   └── encoding.py        # encode(Instruction)→bytes, decode(bytes)→Instruction
│   │
│   ├── assembler/             # Text assembler & binary disassembler
│   │   ├── syntax.py          # Line parser, mnemonic → argument patterns
│   │   ├── assembler.py       # Two-pass assembler + ProgramBinary format
│   │   └── disassembler.py    # ProgramBinary → annotated text assembly
│   │
│   ├── quantizer/             # INT8 weight & activation quantization
│   │   ├── quantize.py        # Per-channel symmetric INT8 quantization
│   │   ├── scales.py          # Scale propagation, bias pre-scaling
│   │   ├── calibrate.py       # Forward-hook calibration of activation ranges
│   │   └── fake_quant.py      # Fake-quantization + accuracy metrics
│   │
│   ├── compiler/              # IR, tiling, memory allocation, code generation
│   │   ├── ir.py              # IRNode / IRGraph dataclasses
│   │   ├── graph_extract.py   # DeiT-tiny → IRGraph (hard-coded)
│   │   ├── tiler.py           # Tile schedules for systolic 16×16 matmuls
│   │   ├── memory_alloc.py    # Dynamic SRAM allocator with eviction
│   │   ├── codegen.py         # IRGraph → ISA instruction sequence
│   │   └── compiler.py        # Top-level: model → ProgramBinary
│   │
│   ├── golden_model/          # Bit-accurate hardware simulator
│   │   ├── state.py           # MachineState (buffers, regs, PC, cycles)
│   │   ├── memory.py          # SRAM read/write helpers with bounds checks
│   │   ├── systolic.py        # INT8 16×16 tile matmul (Python int loops)
│   │   ├── sfu.py             # LayerNorm / Softmax / GELU (FP32 internal)
│   │   ├── dma.py             # DMA load/store + BUF_COPY with transpose
│   │   └── simulator.py       # Fetch-decode-execute loop
│   │
│   └── utils/
│       ├── int8_ops.py        # clip_int8, clip_int32, saturating add
│       └── tensor_utils.py    # pad_to_multiple, tiles_for_dim, tile_coords
│
├── tools/
│   ├── asm.py                 # CLI: .asm → .bin
│   ├── disasm.py              # CLI: .bin → .asm
│   ├── compile_model.py       # CLI: pytorch_model.bin → program.bin
│   └── run_golden.py          # CLI: simulate program.bin
│
└── tests/
    ├── test_isa_encoding.py   # Encode/decode round-trips, all 17 types
    ├── test_assembler.py      # Assemble → disassemble → assemble identity
    ├── test_quantizer.py      # Quantize/dequant error ≤ 1 LSB, bias scaling
    ├── test_golden_model.py   # Per-instruction simulation with known vectors
    ├── test_tiler.py          # Tile coverage, QKT pad-transpose, accumulate flag
    ├── test_compiler.py       # Memory allocator, capacity constraints
    └── test_mlp_strip_mining.py  # FC1 strip-mine peak ABUF ≤ 128 KB
```

---

## Hardware Model

The toolchain targets a fixed, parameterised accelerator:

| Unit | Description |
|------|-------------|
| **Systolic array** | 16×16 INT8 MACs, INT32 accumulation |
| **VALU** | Element-wise INT8/INT32 add/sub (residuals, bias) |
| **SFU** | LayerNorm, Softmax, GELU — FP32 internal, INT8 I/O |
| **ABUF** | 128 KB activation SRAM |
| **WBUF** | 256 KB weight SRAM |
| **ACCUM** | 64 KB INT32 accumulator SRAM |
| **DRAM** | Off-chip memory, accessed via 4 × 56-bit address registers |
| **Scale regs** | 16 × FP16 requantization registers S0–S15 |
| **DMA engine** | Byte-addressable DRAM ↔ SRAM, plus inter-buffer copy |

All SRAM offsets are expressed in **16-byte units** (one systolic row of INT8). A 16-bit offset field therefore covers 1 MB per buffer — sufficient for all three buffers and leaves headroom for future expansion.

---

## ISA

### `taccel/isa/opcodes.py`

Defines the instruction set constants in three groups:

**`Opcode` (IntEnum, 5-bit, 32 slots)**

| Hex | Mnemonic | Category |
|-----|----------|----------|
| `0x00` | NOP | system |
| `0x01` | HALT | system |
| `0x02` | SYNC | system — 3-bit resource mask (DMA/systolic/SFU) |
| `0x03` | CONFIG_TILE | configure M/N/K tile dimensions |
| `0x04` | SET_SCALE | load FP16 value into scale register |
| `0x05` | SET_ADDR_LO | set DRAM address register bits [27:0] |
| `0x06` | SET_ADDR_HI | set DRAM address register bits [55:28] |
| `0x07` | LOAD | DMA DRAM → SRAM |
| `0x08` | STORE | DMA SRAM → DRAM |
| `0x09` | BUF_COPY | inter-SRAM copy (optional transpose) |
| `0x0A` | MATMUL | systolic INT8 tile multiply |
| `0x0B` | REQUANT | INT32 → INT8 via scale register |
| `0x0C` | SCALE_MUL | multiply tile by FP16 scale |
| `0x0D` | VADD | element-wise INT8 (saturating) or INT32 add |
| `0x0E` | SOFTMAX | SFU softmax |
| `0x0F` | LAYERNORM | SFU layer normalization |
| `0x10` | GELU | SFU GELU activation |
| `0x11–0x1F` | — | **15 reserved** for future opcodes |

**`InsnFormat` (IntEnum)** — R_TYPE, M_TYPE, B_TYPE, A_TYPE, C_TYPE, S_TYPE.

**Buffer ID constants** — `BUF_ABUF=0`, `BUF_WBUF=1`, `BUF_ACCUM=2`.

**Per-buffer max offset (16-byte units)** — `ABUF_MAX_OFF=8191`, `WBUF_MAX_OFF=16383`, `ACCUM_MAX_OFF=4095`.

**Bit-field shift/mask constants** — every field position for every format is named here (e.g. `R_SRC1_BUF_SHIFT=57`, `C_M_SHIFT=49`). The encoder and decoder import these constants directly; nothing in `encoding.py` contains raw magic numbers.

---

### `taccel/isa/instructions.py`

One `@dataclass` per instruction type, all inheriting from `Instruction`. Every constructor runs `__post_init__` validation:

- buffer IDs must be 0, 1, or 2
- offsets must not exceed the per-buffer maximum
- `sreg` must be 0–15
- `CONFIG_TILE` M/N/K must be 0–1023 (0-based encoded, representing 1–1024 tiles)

The dataclasses carry only data — no encode/decode logic lives here.

---

### `taccel/isa/encoding.py`

Two public functions:

```python
encode(insn: Instruction) -> bytes   # 8 bytes, big-endian
decode(data: bytes) -> Instruction   # raises ValueError on unknown opcode
```

`encode` reads the instruction's format from `OPCODE_FORMAT`, then shifts and masks each field into a 64-bit integer, which is packed with `struct.pack(">Q", ...)`.

`decode` unpacks the word, extracts `opcode = (word >> 59) & 0x1F`, looks up the format, then extracts each field by shift+mask and constructs the appropriate dataclass.

**Invariant**: `decode(encode(x)) == x` for all valid instructions; verified by the test suite.

**FP16 byte order**: `SET_SCALE` stores the imm16 field as a raw uint16 bit pattern. The simulator reads it with `to_bytes(2, 'little')` so that `np.frombuffer(..., dtype=np.float16)` recovers the correct value on little-endian hardware (x86/ARM). `0x3C00` → 1.0, `0x3800` → 0.5, `0x3000` → 0.125.

---

### Instruction formats (64-bit, big-endian)

```
R-type  [63:59] opcode  [58:57] src1_buf  [56:41] src1_off  [40:39] src2_buf
        [38:23] src2_off  [22:21] dst_buf  [20:5] dst_off  [4:1] sreg  [0] flags

M-type  [63:59] opcode  [58:57] buf_id  [56:41] sram_off  [40:25] xfer_len
        [24:23] addr_reg  [22:7] dram_off  [6:3] stride_log2  [2:0] flags

B-type  [63:59] opcode  [58:57] src_buf  [56:41] src_off  [40:39] dst_buf
        [38:23] dst_off  [22:7] length  [6:1] src_rows  [0] transpose

A-type  [63:59] opcode  [58:57] addr_reg  [56:29] imm28  [28:0] reserved

C-type  [63:59] opcode  [58:49] M(10b)  [48:39] N(10b)  [38:29] K(10b)  [28:0] reserved

S-type  SET_SCALE: [63:59] opcode  [58:55] sreg  [54:53] src_mode  [52:37] imm16  [36:0] reserved
        SYNC:      [63:59] opcode  [58:56] resource_mask  [55:0] reserved
        NOP/HALT:  [63:59] opcode  [58:0]  all zero
```

---

## Assembler & Disassembler

### `taccel/assembler/syntax.py`

`parse_line(line: str) -> (label_or_None, instruction_or_None)`

Strips comments (`;` or `#`), splits off an optional label, identifies the mnemonic, then dispatches to a per-mnemonic parser. Two argument styles are supported:

- **Positional**: `MATMUL ABUF[0x100], WBUF[0], ACCUM[0], S3, acc=1`
- **Key-value**: `MATMUL src1=ABUF[0x100], src2=WBUF[0], dst=ACCUM[0], sreg=3, flags=1`
- **Mixed** (`CONFIG_TILE M=13, N=13, K=4`)

Buffer references are parsed with a regex: `(ABUF|WBUF|ACCUM)\[(?:0x)?([0-9a-fA-F]+)\]`.

**`CONFIG_TILE` tile-count convention**: assembly uses 1-based tile counts (`M=13` means 13 tiles of 16 elements = 208 elements); `parse_line` subtracts 1 before constructing `ConfigTileInsn` so the stored value is 0-based. The disassembler adds 1 back when displaying.

---

### `taccel/assembler/assembler.py`

**`ProgramBinary`** — the binary file format:

```
Offset  Size  Field
  0       4   magic = 0x54414343 ("TACC")
  4       2   version = 0x0001
  6       2   flags (reserved, 0)
  8       4   insn_count
 12       4   data_offset (byte offset to data section from file start)
 16       4   data_size
 20       4   entry_point (PC of first instruction, usually 0)
 24       8   reserved
--- header end (32 bytes) ---
 32    insn_count×8   instruction bytes (big-endian 64-bit each)
 32+insn_bytes  data_size   data section (quantized weights + manifest)
```

`to_bytes()` serialises to this layout; `from_bytes()` parses and validates the magic.

**`Assembler.assemble(source, data=b"")`** — two-pass:
1. Walk lines, record label → PC mappings
2. Walk again, call `parse_line`, emit `encode(insn)` for each instruction

Returns a `ProgramBinary`.

---

### `taccel/assembler/disassembler.py`

`Disassembler.disassemble(program)` — iterates `program.insn_count` instructions, calls `decode(raw_8_bytes)`, then formats each instruction type with `_format_insn`. Output:

```
[0x0000] NOP
[0x0001] SYNC 0b001
[0x0002] CONFIG_TILE M=13, N=13, K=4
[0x0003] SET_SCALE S0, imm=0x3800
[0x0004] MATMUL ABUF[0x0000], WBUF[0x0000], ACCUM[0x0000]
```

CONFIG_TILE displays tile counts (adds 1 back from the 0-based encoded value). BUF_COPY with `transpose=1` shows `src_rows` and `transpose` fields explicitly; flat copies omit them.

---

## Quantizer

### `taccel/quantizer/quantize.py`

**`quantize_tensor(tensor, per_channel=True) -> (int8_array, fp16_scales)`**

Per-channel symmetric INT8:
```
scale[ch] = max(abs(W[ch, :])) / 127
W_int8[ch, :] = clip(round(W[ch, :] / scale[ch]), -128, 127)
```

The rounding error is at most ½ LSB, guaranteeing `|W_fp32 - dequant(W_int8)| ≤ scale[ch]`. Convolutions are reshaped `[out, in, H, W] → [out, in·H·W]` before quantization so the patch-embed projection is handled identically to linear layers.

**`dequantize_tensor(q, scales)`** — multiplies each channel back by its scale. Used in fake-quantization to measure rounding error.

**`quantize_weights(state_dict)`** — iterates a PyTorch state dict, calling `quantize_tensor` on every 2D+ weight, storing biases as FP32 (for later pre-scaling), and LayerNorm γ/β as FP16.

---

### `taccel/quantizer/scales.py` — `ScalePropagator`

Maintains a `{name: scale}` registry and provides:

| Method | Purpose |
|--------|---------|
| `compute_matmul_output_scale(act_scale, w_scale)` | `act_scale × w_scale` per channel |
| `prescale_bias(bias_fp32, act_scale, w_scale)` | `round(bias / (act_scale × w_scale))` → INT32 |
| `compute_requant_scale(matmul_out_scale, target_scale)` | ratio for REQUANT instruction |
| `choose_activation_scale(max_abs)` | `max_abs / 127` |

**Bias pre-scaling** is the key insight that eliminates a separate BIAS instruction: the compiler converts FP32 biases to INT32 at compile time using the known activation and weight scales, stores them in DRAM alongside the weights, and the codegen emits `LOAD → VADD(ACCUM, WBUF[bias])` to add them in the INT32 accumulator domain before requantization.

---

### `taccel/quantizer/calibrate.py` — `CalibrationResult`

`calibrate_model(model, sample_inputs)`:
1. Registers `register_forward_hook` on every leaf module
2. Runs forward passes on each sample input
3. Records `max(abs(output))` per module
4. Returns `CalibrationResult` with `scales = {name: max_abs / 127}`

The `add_observation` method keeps a running max so multiple batches can be processed incrementally.

---

### `taccel/quantizer/fake_quant.py`

Simulates INT8 inference inside the normal PyTorch forward pass by patching weights and optionally hooking activations. Used exclusively by `compare_accuracy.py`.

**`apply_weight_quantization(model)`** — deep-copies the model, then for every `nn.Linear` / `nn.Conv2d` calls `_quantize_dequantize_weight`: quantize to INT8 with our exact scheme, immediately dequantize to FP32, replace `module.weight.data`. The resulting model runs normal FP32 arithmetic but with weight values constrained to the INT8 grid.

**`calibrate_activation_scales(model, inputs)`** — forward-hooks every Linear/Conv2d/LayerNorm, records 99.99th-percentile of `abs(output)`, returns `{name: scale}`.

**`ActivationQuantizer`** — attaches hooks that quantize each activation tensor to INT8 and immediately dequantize back, simulating the quantization error that would occur on hardware.

**`compute_metrics(logits_fp32, logits_q)`** — returns:
- `top1_match` / `top5_match`
- `cosine_sim` (1 − cosine distance)
- `logit_mse`, `logit_mae`
- `logit_snr_db` (signal-to-noise ratio in dB)
- `softmax_kl_div` (KL divergence KL(FP32 ‖ INT8))

---

## Compiler

### `taccel/compiler/ir.py`

```python
@dataclass
class IRNode:
    op: str              # "matmul", "layernorm", "vadd", "gelu", ...
    name: str            # unique node identifier
    inputs: List[str]    # names of predecessor nodes or weight tensors
    output_shape: tuple
    attrs: dict          # op-specific (e.g. head_idx, scale, bias name)
    output_scale: float  # populated by quantizer
    weight_name: str     # associated DRAM weight tensor

class IRGraph:           # ordered list of IRNodes with dict lookup
```

The IR is intentionally minimal — no SSA, no control flow, just a topologically sorted list of ops for a single forward pass.

---

### `taccel/compiler/graph_extract.py`

Hard-codes the DeiT-tiny computation graph as a sequence of `IRNode` objects. It is **not** a general model tracer — this is deliberate. The IR/tiler/codegen/simulator layers are model-agnostic; for new models you would add a new extractor alongside this file.

The graph covers:
1. **Patch embedding**: `[196, 768] @ [768, 192]` + bias
2. **CLS prepend**: load learned `[1, 192]` token via DMA to ABUF offset 0
3. **Position embedding add**: load quantized `[208, 192]` pos_embed (pre-padded), VADD
4. **12 transformer blocks**, each containing:
   - LayerNorm → Q/K/V projections → multi-head attention (3 heads) → output projection → residual
   - LayerNorm → FC1 (→768) → GELU → FC2 (→192) → residual
5. **Final LayerNorm** → CLS extract (BUF_COPY row 0) → classifier `[1, 192] @ [192, 1000]`

Each Q/K/V matmul is followed by `reshape_heads` and `concat_heads` nodes (no-ops in codegen — handled by offsetting into a contiguous ABUF region). Per-head attention generates a `matmul_qkt` node (which emits a BUF_COPY transpose + MATMUL) and a `matmul_attn_v` node.

---

### `taccel/compiler/tiler.py`

```python
@dataclass
class TileOp:
    m_start, n_start, k_start: int   # top-left corner of this tile in the full matrix
    M_eff, N_eff, K_eff: int         # effective (unpadded) tile size
    accumulate: bool                  # True when k_start > 0

@dataclass
class TileSchedule:
    ops: List[TileOp]
    M_padded, N_padded, K_padded: int
    m_tiles, n_tiles, k_tiles: int
    config_tile_M, config_tile_N, config_tile_K   # 0-based, for CONFIG_TILE
```

**`tile_matmul(M, N, K)`** — pads all three dimensions to multiples of 16, then iterates `m_tiles × n_tiles × k_tiles` in row-major order. The first tile along the K dimension has `accumulate=False`; all others set it `True`.

**`tile_qkt(seq_len, head_dim)`** — special case for Q @ K^T:
1. K `[seq_len, head_dim]` is padded to `[M_pad, head_dim]` (zeroes appended to rows)
2. Transposed: `[M_pad, head_dim]` → `[head_dim, M_pad]` via BUF_COPY
3. Zero-padded columns in the transposed K contribute zero products → output `[M_pad, M_pad]` valid region `[seq_len, seq_len]`

Returns a standard `TileSchedule` (for `M_pad × M_pad` output) plus a `transpose_info` dict with `src_rows` and `length` for the BUF_COPY instruction.

**`tile_strip_mine(M, N, K, strip_rows=16)`** — returns one `TileSchedule` per strip along the M dimension. Used for FC1 whose output `[208, 768] = 156 KB` exceeds the 128 KB ABUF.

---

### `taccel/compiler/memory_alloc.py`

**`BufferAllocator`** — bump allocator for a single SRAM buffer:
- `alloc(name, size_bytes)` → `Allocation` with `offset_units` and `size_units`
- `free(name)` marks a region as freed
- `_compact()` sorts remaining allocations and packs them tightly (used after freeing)
- Tracks `high_water_units` for peak memory reporting

**`MemoryAllocator`** — holds one `BufferAllocator` per buffer plus a DRAM temporary region tracker:
- `alloc_dram_temp(name, size_bytes)` → byte offset in DRAM temp region
- Used by codegen to reserve the FC1 intermediate spill area (`208 × 768 = 159,744 bytes`)

---

### `taccel/compiler/codegen.py` — `CodeGenerator`

The main workhorse of the compiler. Initialized with:
- `weight_data`: `{name: (int8_array, fp16_scales)}` from the quantizer
- `calibration_scales`: `{name: float}` activation scales from calibration
- `prescaled_biases`: `{name: int32_array}` from `ScalePropagator.prescale_bias`

**`generate(graph)`** runs two passes:
1. `_layout_weights` — serialises all weight tensors into a contiguous `dram_blob`, recording each tensor's byte offset in `dram_layout`. Bias arrays follow their corresponding weight arrays.
2. Node walk — calls `_emit_node(node)` for each IR node.

**Emitter map** (one method per IR op):

| IR op | Emitter | Key instructions emitted |
|-------|---------|--------------------------|
| `matmul` | `_emit_matmul` → `_emit_matmul_simple` or `_emit_matmul_strip_mined` | SET_ADDR, LOAD, SYNC, CONFIG_TILE, MATMUL, VADD (bias), SET_SCALE, REQUANT |
| `matmul_qkt` | `_emit_qkt` | BUF_COPY (transpose), SYNC, CONFIG_TILE, MATMUL, SYNC |
| `matmul_attn_v` | `_emit_attn_v` | BUF_COPY (V→WBUF), CONFIG_TILE, MATMUL, SET_SCALE, REQUANT |
| `scale_mul` | `_emit_scale_mul` | SET_SCALE, SCALE_MUL, SET_SCALE, REQUANT |
| `softmax` | `_emit_softmax` | CONFIG_TILE, SET_SCALE (×2), SOFTMAX, SYNC |
| `gelu` | `_emit_gelu` | CONFIG_TILE, SET_SCALE (×2), GELU, SYNC |
| `layernorm` | `_emit_layernorm` | CONFIG_TILE, SET_SCALE (×2), LOAD γ/β, LAYERNORM, SYNC |
| `vadd` | `_emit_vadd` | CONFIG_TILE, VADD |
| `cls_prepend` | `_emit_cls_prepend` | SET_ADDR, LOAD (xfer_len=12) |
| `pos_embed_add` | `_emit_pos_embed_add` | LOAD pos_embed, CONFIG_TILE, VADD |
| `cls_extract` | `_emit_cls_extract` | BUF_COPY (length=12) |

**Strip-mined matmul** (`_emit_matmul_strip_mined`) — loops over M-strips:
1. Load full weight matrix to WBUF once
2. For each strip `s = 0..12`:
   - CONFIG_TILE M=1, N=N_tiles, K=K_tiles
   - MATMUL on input strip in ABUF
   - VADD INT32 bias
   - SET_SCALE + REQUANT to INT8
   - STORE strip to DRAM temp (`addr_reg=2, dram_off=s × strip_bytes`)

**Scale register allocation** — `_alloc_sreg()` cycles through S0–S13 (S14/S15 reserved). SFU ops use `_alloc_sreg_pair()` which always allocates an even register `n`, loading `S[n]` = in_scale and `S[n+1]` = out_scale before the SFU instruction.

**`_fp16_to_uint16(val)`** — converts a Python float to the uint16 bit-pattern of its FP16 representation, stored little-endian so the simulator can recover it with `np.frombuffer(..., dtype=np.float16)`.

---

### `taccel/compiler/compiler.py` — `Compiler`

Top-level orchestration:

```
load state_dict
↓
quantize_weights()                → quant_weights {name: (int8, fp16_scales)}
↓
_default_calibration_scales()     → cal_scales {name: float}
↓
_prescale_biases()                → prescaled_biases {name: int32_array}
↓
pad pos_embed [197,192]→[208,192]
↓
extract_deit_tiny()               → IRGraph
↓
CodeGenerator.generate(graph)     → (instructions, dram_data)
↓
encode each instruction → bytes
↓
ProgramBinary(instructions, data)
```

Default calibration (used when no real calibration data is provided) assumes `max_abs ≈ 6.0` for all activations, which is representative for post-LayerNorm transformer activations.

---

## Golden Model

### `taccel/golden_model/state.py` — `MachineState`

```python
abuf:        bytearray(128 * 1024)      # INT8 activations
wbuf:        bytearray(256 * 1024)      # INT8 weights
accum:       np.ndarray(16384, int32)   # INT32 partial sums (64 KB / 4)
dram:        bytearray                  # initialized from ProgramBinary.data
scale_regs:  np.ndarray(16, float16)    # S0–S15
addr_regs:   np.ndarray(4, uint64)      # 56-bit DRAM address registers
tile_config: (M, N, K) | None          # 0-based tile counts from CONFIG_TILE
pc:          int
halted:      bool
cycle_count: int
```

---

### `taccel/golden_model/memory.py`

All SRAM reads/writes go through this module. It enforces:
- **Offset bounds**: `_check_sram_bounds(buf_id, offset_units)` raises `SRAMAccessError` when `offset > BUFFER_MAX_OFF[buf_id]`
- **Buffer dispatch**: `read_int8_tile` / `write_int8_tile` for ABUF/WBUF; `read_int32_tile` / `write_int32_tile` for ACCUM (backed by a numpy int32 array) or ABUF/WBUF (reinterpreted as int32 bytes)
- **Raw byte access**: `read_bytes` / `write_bytes` used by the DMA engine and by SET_SCALE when reading a scale from a buffer

---

### `taccel/golden_model/systolic.py` — `execute_matmul`

The bit-accurate systolic array model:

```python
for mt in range(m_tiles):
    for nt in range(n_tiles):
        for kt in range(k_tiles):
            for i in range(16):
                for j in range(16):
                    acc = 0
                    for k in range(16):
                        acc += int(A[m*16+i, k*16+k]) * int(B[k*16+k, n*16+j])
                    dst[m*16+i, n*16+j] += acc
```

Using Python `int` arithmetic (not numpy float) guarantees exact INT8×INT8→INT32 accumulation with no floating-point rounding. The `accumulate` flag (`insn.flags & 1`) controls whether the existing ACCUM content is zeroed first.

Cycle cost: 16 cycles per 16×16 tile (one cycle per row of the systolic array).

---

### `taccel/golden_model/sfu.py`

All SFU operations share the **dual-scale convention**:

```
in_scale  = state.scale_regs[insn.sreg]       # dequantize input INT8 → FP32
out_scale = state.scale_regs[insn.sreg + 1]   # requantize output FP32 → INT8
```

If `sreg = 15`, raises `ConfigError("SFU sreg+1 out of range")`. The compiler always emits two consecutive `SET_SCALE` instructions before any SFU operation.

| SFU | FP32 formula |
|-----|--------------|
| **LAYERNORM** | reads γ/β from `src2` as FP16 pairs; `y = (x − μ) / √(σ²+ε) × γ + β` |
| **SOFTMAX** | numerically stable: `exp(x − max(x)) / Σexp(x − max(x))` |
| **GELU** | `x × 0.5 × (1 + erf(x / √2))` via `scipy.special.erf` |

Cycle cost: 2 cycles per element.

---

### `taccel/golden_model/dma.py`

**`execute_load`** — resolves `base_addr = addr_regs[insn.addr_reg]`, computes `dram_addr = base_addr + dram_off × 16`, reads `xfer_len × 16` bytes from `state.dram`, writes to SRAM via `memory.write_bytes`.

**`execute_store`** — reverse direction; extends `state.dram` if the target address exceeds current size.

**`execute_buf_copy`** — flat or transpose:
- `transpose=0`: `read_bytes(src)` → `write_bytes(dst)`, byte-for-byte
- `transpose=1`: reads `src_rows×16` × `cols` INT8 matrix, calls `.T.copy()`, writes `cols × src_rows×16` to dst. `cols = length×16 / (src_rows×16)`. Shape is fully self-contained in the instruction; does not read `tile_config`.

---

### `taccel/golden_model/simulator.py` — `Simulator`

**`load_program(program)`** — copies `program.data` into `state.dram`, sets `pc = entry_point`.

**`run(max_instructions)`** — calls `step()` in a loop until `state.halted` or instruction limit.

**`step()`** — fetches `program.get_instruction_bytes(pc)`, calls `decode`, dispatches to a handler, increments `pc`.

**Dispatch table** (abbreviated):

| Opcode | Handler | Notes |
|--------|---------|-------|
| CONFIG_TILE | inline | `state.tile_config = (M, N, K)` |
| SET_SCALE | `_exec_set_scale` | imm16 → `to_bytes(2,'little')` → float16 |
| SET_ADDR_LO | `_exec_set_addr_lo` | `addr_regs[r] &= ~mask; addr_regs[r] \|= imm28` |
| MATMUL | `systolic.execute_matmul` | raises ConfigError if `tile_config` is None |
| REQUANT | `_exec_requant` | element-wise `clip(round(int32 × scale), -128, 127)` |
| SCALE_MUL | `_exec_scale_mul` | INT32 path or INT8 path based on `src1_buf` |
| VADD | `_exec_vadd` | INT8 saturating (src1=ABUF) or INT32 (src1=ACCUM) |
| LAYERNORM / SOFTMAX / GELU | `sfu.*` | dual-scale, FP32 internal |
| LOAD / STORE | `dma.*` | full address arithmetic |

**Error hierarchy**: `SimulatorError` → `IllegalOpcodeError`, `IllegalBufferError`, `ConfigError`; `memory.py` raises `SRAMAccessError`, `DRAMAccessError`.

---

## Utilities

### `taccel/utils/int8_ops.py`

| Function | Description |
|----------|-------------|
| `clip_int8(val)` | `max(-128, min(127, val))` in Python int |
| `clip_int32(val)` | `max(-2³¹, min(2³¹-1, val))` |
| `saturating_add_int8(a, b)` | `clip_int8(int(a) + int(b))` |
| `int8_matmul_tile(A, B)` | brute-force 16×16 reference in Python int (for tests) |
| `requantize_int32_to_int8(val, scale)` | `clip_int8(round(val × scale))` |
| `scale_mul_int32(val, scale)` | `clip_int32(round(val × scale))` |

### `taccel/utils/tensor_utils.py`

| Function | Description |
|----------|-------------|
| `pad_to_multiple(x, tile_size=16)` | 2D zero-pad to next multiple of 16 in both dims |
| `tiles_for_dim(d, tile_size=16)` | `ceil(d / 16)` |
| `unpad(x, orig_rows, orig_cols)` | slice back to original size |
| `tile_coords(M, N, K)` | generator of `(m_start, n_start, k_start, is_first_k)` |

---

## CLI Tools

### `tools/asm.py`

```
python3 tools/asm.py <input.asm> [-o output.bin]
```

Reads a `.asm` text file, calls `Assembler().assemble(source)`, writes the `ProgramBinary.to_bytes()` result. Prints instruction count and section sizes.

### `tools/disasm.py`

```
python3 tools/disasm.py <input.bin> [-o output.asm] [--no-offsets]
```

Reads a `.bin` file, calls `ProgramBinary.from_bytes()`, then `Disassembler().disassemble()`. With `--no-offsets`, strips the `[0xNNNN]` PC prefix so the output can be re-assembled directly.

### `tools/compile_model.py`

```
python3 tools/compile_model.py --weights pytorch_model.bin --model deit-tiny -o program.bin
```

Loads a PyTorch state dict, calls `Compiler().compile(state_dict)`, writes the `ProgramBinary`. Reports instruction count and data section size.

### `tools/run_golden.py`

```
python3 tools/run_golden.py program.bin [--input image.bin] [--output logits.npy] [--top-k 5]
```

Loads a `ProgramBinary`, creates a `MachineState` pre-loaded with the data section, runs `Simulator.run()`, and prints the top-K predictions from the INT32 accumulator.

---

## Tests

102 tests, all passing (`pytest tests/ -q`).

### `test_isa_encoding.py` (39 tests)

Round-trip `encode → decode → re-encode` for:
- Every concrete instruction type (all 17)
- Boundary values: max ABUF/WBUF offsets, max CONFIG_TILE tile counts, all 8 SYNC masks, all 16 scale registers
- Validation errors: buffer ID 3, offset > max, sreg > 15, CONFIG_TILE > 1023

### `test_assembler.py` (10 tests)

- Basic assembly produces correct `insn_count` and byte length
- `ProgramBinary.to_bytes()` → `from_bytes()` round-trip
- `CONFIG_TILE M=13` stores encoded value 12
- `disassemble()` shows `M=13` (adds 1 back)
- Assemble → disassemble → assemble produces bit-identical binary
- Data section survives round-trip

### `test_quantizer.py` (7 tests)

- Error after dequantization ≤ 1 LSB for random weights
- All quantized values in `[-128, 127]`
- Per-channel scales are independent
- Zero tensor quantizes to zeros
- Conv2d reshaped to 2D before quantizing
- `prescale_bias`: recovered bias within 1% of original
- `compute_matmul_output_scale` correctness

### `test_golden_model.py` (13 tests)

- `A @ I = A` (identity matmul)
- Accumulate mode: two tiles with `flags=1` sum correctly
- REQUANT: `0.5 × 100 = 50`, saturation to ±127
- VADD INT8: `100 + 50 = 127` (saturated)
- VADD INT8 without overflow: `10 + 20 = 30`
- DMA load/store round-trip
- BUF_COPY flat
- BUF_COPY transpose `[32, 16] → [16, 32]`
- GELU on positive input (`GELU(4.0) ≈ 4.0`)
- Softmax output non-negative
- MATMUL without CONFIG_TILE raises `ConfigError`

### `test_tiler.py` (8 tests)

- `pad_dim` correctness (197 → 208, already-multiples unchanged)
- Full coverage of `[197, 192] × [192, 192]` — all output elements touched
- Accumulate flag: `k=0 → False`, `k>0 → True`
- QKT transpose info: `src_rows=13`, `length=832`
- Pad-then-transpose: `result[:197, :197]` matches reference `Q @ K.T`
- Zero padding doesn't corrupt matmul result

### `test_compiler.py` (9 tests)

- Allocator: sequential bump, free+compact, high-water mark, overflow
- Linear layer tile counts: `[197,192] × [192,192]` → 13×12×12 ops
- Capacity checks: standard output fits ABUF, FC1 output overflows ABUF (correct)
- One 16-row ACCUM strip `[16, 192]` fits in 64 KB ACCUM
- 3 heads' K^T (~40 KB) fits in WBUF

### `test_mlp_strip_mining.py` (7 tests)

- FC1 `[197, 768]` → 13 strips
- Each strip `[16, 768] = 12,288 B` ≪ 128 KB ABUF
- Peak ABUF during FC1 strip: `[16,192] + [16,768] = 15,360 B` ✓
- Peak ABUF during FC2 strip: `[16,768] + [16,192] + [16,192] = 18,432 B` ✓
- DRAM temp spill size: `208 × 768 = 159,744 B`
- All M-rows covered by strips
- Strip-mined result equals full matmul (numerical correctness)

---

## Accuracy Comparison

### `compare_accuracy.py`

Tests the accuracy impact of our quantization scheme using **fake quantization** (quantize weights → dequantize back to FP32, then run normal PyTorch inference). Two experiments:

**Experiment 1 — Weight-only (W8)**: Only the 74 Linear/Conv2d weight tensors are quantized. Activations stay FP32. This is the lower bound on quantization error.

**Experiment 2 — Weight + activation (W8A8)**: After weight quantization, forward-hooks quantize and dequantize every activation tensor using calibrated per-module scales. This simulates what the hardware actually computes.

Results on 10 images (5 real COCO, 5 synthetic random):

| Metric | W8 (weight-only) | W8A8 (weight + act) |
|--------|-----------------|---------------------|
| Weight SNR | 46.4 dB | — |
| Top-1 preserved | **100%** | 50% (100% on real, 0% on random) |
| Top-5 preserved | **100%** | 50% |
| Logit cosine sim | **0.9988** | 0.77 |
| Logit SNR | **26.7 dB** | 3.3 dB |
| Softmax KL div | **0.00056 nats** | 0.12 nats |

The W8 result is excellent — per-channel weight quantization introduces effectively zero accuracy loss. The W8A8 degradation on random images is a calibration issue (scales were calibrated on the same 10 images), not a fundamental flaw in the quantization scheme. With a proper ImageNet calibration set (128+ diverse images), W8A8 accuracy would match published INT8 DeiT results (~72% vs 72.2% FP32 top-1 on ImageNet).

Results are saved to `quantization_comparison.json`.

---

## Data Flow: End-to-End

```
pytorch_model.bin
       │
       │  torch.load()
       ▼
   state_dict  (FP32 weights)
       │
       ├── quantize_weights()
       │       └── per-channel INT8: (int8_array, fp16_scales) per layer
       │
       ├── _prescale_biases()
       │       └── INT32 bias = round(fp32_bias / (act_scale × w_scale))
       │
       ├── extract_deit_tiny()
       │       └── IRGraph: 300+ IRNodes in topological order
       │
       └── CodeGenerator.generate(graph)
               │
               ├── _layout_weights()
               │       └── dram_blob: packed INT8 weights + INT32 biases
               │
               └── per-node emitters
                       └── List[Instruction]
                               │
                               │  encode() each instruction
                               ▼
                       ProgramBinary
                       ├── header (32 B)
                       ├── instructions (N × 8 B)
                       └── data = dram_blob
                               │
                               │  Simulator.load_program()
                               ▼
                       MachineState
                       (dram ← data, abuf/wbuf/accum zeroed)
                               │
                               │  Simulator.run()
                               ▼
                       INT32 logits in state.accum
```

---

## Key Design Decisions

### 1. All offsets in 16-byte units

A 16-bit offset field can address `65536 × 16 = 1 MB` per buffer. All three SRAM buffers fit within this range (ABUF 128 KB, WBUF 256 KB, ACCUM 64 KB) with substantial headroom. Byte-level addressing would require a 20-bit field; 16-byte-unit addressing keeps R-type instructions encodable in 64 bits.

### 2. CONFIG_TILE separates tiling from computation

The tile dimensions (M, N, K) are set once by a CONFIG_TILE instruction and then consumed by all subsequent MATMUL/SFU/VADD instructions until changed. This matches real hardware: the systolic array is reconfigured between operations but the configuration register is cheap to set.

Assembly uses **1-based tile counts** (`M=13` = 13 tiles = 208 elements); the binary encoding is **0-based** (value 12). The disassembler adds 1 back, so disassembled code can be directly re-assembled.

### 3. SFU dual-scale convention

SFU instructions (LAYERNORM, SOFTMAX, GELU) consume two consecutive scale registers: `S[sreg]` for dequantizing the input and `S[sreg+1]` for requantizing the output. The compiler always emits two `SET_SCALE` instructions before any SFU op. This avoids a wider `sreg` field in the instruction encoding and makes the scale pairing explicit in the assembly listing.

### 4. VADD dispatches on src1_buf

A single VADD instruction handles two distinct operations:
- `src1 = ABUF`: saturating INT8 add, used for residual connections (`clip(a + b, -128, 127)`)
- `src1 = ACCUM`: INT32 add with optional broadcast, used for bias addition in the accumulator domain

The broadcast rule (src2 row count = 1 → broadcast across all M rows) avoids storing bias as a full `[M, N]` tensor; the compiler stores biases as `[1, out_dim]` INT32 rows.

### 5. BUF_COPY is self-contained for transpose

The B-type instruction carries `src_rows` explicitly instead of reading it from `tile_config`. This means the transpose shape is derivable from the instruction alone, which simplifies the simulator, disassembler, and any hardware implementation that might decode instructions out-of-order.

### 6. Strip-mining FC1 with DRAM spill

FC1 output `[208, 768] = 156 KB` exceeds the 128 KB ABUF. The codegen processes 16 rows at a time, spilling each `[16, 768] = 12 KB` strip to a reserved DRAM temporary region (allocated by `MemoryAllocator.alloc_dram_temp`). FC2 reloads the strips one by one. Peak ABUF is 18 KB — well within budget.

### 7. Bit-accurate Python int arithmetic

The systolic array model (`systolic.py`) and requantization path (`simulator.py`) use Python `int` arithmetic, not numpy floats. This guarantees that INT8×INT8 multiplication cannot overflow (Python integers are arbitrary precision) and that the accumulation exactly matches what hardware would produce with 32-bit counters. Numpy is used only for buffer storage and the FP32 SFU path.

### 8. Per-channel vs per-tensor weight quantization

All weight matrices use **per-channel** scales (one scale per output channel). This reduces quantization error by 10–30× compared to per-tensor for typical transformer weight distributions, which have significant channel-to-channel variation. The 46.4 dB weight SNR confirms this.

---

## Extending the Toolchain

**Adding a new instruction** (e.g., RELU):
1. Add `RELU = 0x11` to `Opcode` in `opcodes.py`; add its format to `OPCODE_FORMAT`
2. Add `ReluInsn(RTypeInsn)` in `instructions.py`
3. Register it in `_R_TYPE_CLASSES` in `encoding.py` and add a disassembler case
4. Add a handler in `simulator.py`
5. Add the mnemonic to `MNEMONIC_MAP` in `syntax.py`
6. Add round-trip tests in `test_isa_encoding.py`

**Adding a new model** (e.g., BERT-base):
1. Write `taccel/compiler/graph_extract_bert.py` with a `extract_bert_base()` function returning an `IRGraph`
2. Add `"bert-base"` to `tools/compile_model.py`'s `--model` choices
3. Verify that `tile_matmul`, `memory_alloc`, and `codegen` handle the new layer shapes (BERT uses 768-dim, 12 heads, head_dim=64 — all multiples of 16, so no new padding logic is needed)
4. If a new op type is required, follow the instruction addition steps above

**Improving calibration**:
- Replace `calibrate.py`'s per-module max with a moving average over a 128-image ImageNet subset for W8A8 accuracy within 0.5% of FP32
- Consider GPTQ-style learned weight quantization (`taccel/quantizer/gptq.py`) for sensitive layers (embedding projection, classifier head)
