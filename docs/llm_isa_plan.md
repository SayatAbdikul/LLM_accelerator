# Plan: Extend TACCEL ISA to Run nanoGPT (ISA v1.1)

Status: refined implementation plan, incorporating Rev 0-2 critiques plus the
Rev 3-6 correctness, runtime-contract, striping, and verification-gate reviews.

## 1. Goal

Extend the current ViT-oriented TACCEL software stack so it can run a nanoGPT /
GPT-2-style decoder end-to-end on the golden model.

Target model envelope:

- Architecture: nanoGPT / GPT-2 block
- Dimensions: `d_model = 128..384`, `n_layer = 2..6`
- Attention: MHA with learned absolute positional embeddings
- Ops: LayerNorm, Q/K/V projection, causal attention, GELU MLP, residual adds,
  final LayerNorm, `lm_head`
- Precision: INT8 weights and activations, using the current quantization model
- First end-to-end target: golden-model inference, not RTL

Non-goals for this plan:

- RMSNorm, SiLU, RoPE, GQA, MoE, INT4, speculative decoding, paged attention,
  flash attention, or LLaMA-family support
- Hardware branches or an on-device autoregressive loop
- RTL implementation of the new opcodes, except for documenting the future
  integration surface

The main architectural gap is not "transformers in general." TACCEL already has
matmul, LayerNorm, GELU, softmax, and residual operations. The real missing
pieces are decoder-specific:

- Masked attention with strip-aware row context
- DRAM-resident KV cache
- Runtime/host control for prefill and token-by-token decode
- Model/codegen parameterization beyond DeiT constants
- Larger-matrix striping for `d_model = 384` and beyond

## 2. Current Repo Constraints

The current implementation has several constraints that shape the plan:

- `software/taccel/compiler/graph_extract.py` hard-codes DeiT graph constants
  such as `EMBED_DIM`, `DEPTH`, `NUM_HEADS`, `HEAD_DIM`, and `SEQ_LEN`.
- `software/taccel/compiler/compiler.py` imports those constants directly and
  uses DeiT weight names such as `vit.encoder.layer.{i}.*`.
- `software/taccel/compiler/compiler.py` patches only `SET_ADDR_LO` at the
  `data_base` fix-up site (around line 560). That is sufficient for the current
  small DeiT image, but bundle relocation must patch `SET_ADDR_HI` as well so
  larger decoder layouts and synthetic relocation tests remain correct across
  28-bit carry boundaries.
- `software/taccel/compiler/codegen.py` still has hard-coded `head_dim = 64`
  sites and ViT-specific startup ops.
- `software/taccel/compiler/codegen.py:_emit_qkt` is the real attention
  integration point. The standalone `_emit_softmax` handler is effectively a
  rename/no-op path and is not where QK attention softmax is emitted.
- ISA opcodes `0x00..0x13` are implemented. `0x14..0x1F` are currently
  reserved.
- R-type encoding is fully packed (see `software/docs/isa_spec.md` section 2.2):

```text
5 + 2 + 16 + 2 + 16 + 2 + 16 + 4 + 1 = 64 bits
```

There is no spare R-type flag bit for "causal" or "masked" variants. New
variant behavior must use a new opcode or a preceding configuration
instruction.

- Simulator PC is instruction-index based
  (`software/taccel/golden_model/simulator.py:77` and the fetch loop around
  lines 215-219), not byte-based. Any new entry-point mechanism must respect
  this or introduce a byte-addressed fetch mode explicitly.

SRAM budget:

- ABUF: 128 KB, INT8
- WBUF: 256 KB, INT8
- ACCUM: 64 KB, INT32

That budget is enough for a tiny decoder, but only if attention and MLP paths
are tiled carefully. For example, `d_model = 384, mlp_dim = 1536` has FC1 and
FC2 weights of `384 * 1536 = 589,824` bytes each, which exceeds WBUF. The
existing codegen already spills FC1 output strips to DRAM in the DeiT path;
the Stage 4 work is to generalize that path for `d_model = 384` and to add
weight striping, not to invent a new spill mechanism.

## 3. Scope Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Control flow | Host-side runner re-invokes programs | Keeps ISA linear and avoids control-unit redesign |
| Program shape | Separate `prefill` and `decode` instruction streams in one bundle | Allows prompt processing once, then token decode while preserving KV cache |
| KV cache | DRAM-resident, persistent across `run_program` calls | Fits easily in DRAM and avoids on-chip reservation |
| Attention | Materialized per-head attention with masked softmax | Works for `seq <= 256` with existing row striping; flash attention deferred |
| Precision | INT8 only | Matches current stack; INT4 deferred |
| Frontend | Direct nanoGPT adapter first, HF GPT-2 adapter later | Avoids HuggingFace `Conv1D` layout and FX/cache-mask traps in the MVP |
| Position embeddings | Regular LOADs with host-patched token and position offsets | No embedding/gather opcode needed |
| RTL | Deferred | Golden model validates semantics first |

## 4. ISA v1.1

Use three new opcodes for the MVP and reserve one useful future opcode.

| Opcode | Mnemonic | Format | Purpose |
| --- | --- | --- | --- |
| `0x14` | `CONFIG_ATTN` | ATTN-type | Set masked-attention context for subsequent masked softmax ops |
| `0x15` | `MASKED_SOFTMAX` | R-type | Row-wise softmax with mask applied before exponentiation |
| `0x16` | `MASKED_SOFTMAX_ATTNV` | R-type | Masked variant of fused `SOFTMAX_ATTNV` |
| `0x17` | reserved for `ADDR_ADD` | A-type | Future signed address-register delta for reducing decode address setup |
| `0x18..0x1F` | reserved | - | Future RMSNorm, SiLU, RoPE, flash attention, INT4, gather, etc. |

### 4.1 `CONFIG_ATTN` Encoding

`CONFIG_ATTN` is not C-type reuse. Existing C-type decoding produces
`ConfigTileInsn(M, N, K)`, so `CONFIG_ATTN` needs its own opcode-specific
dataclass and encode/decode branch.

```text
[63:59]  opcode = 0x14            5 bits
[58:47]  query_row_base           12 bits, element index 0..4095
[46:35]  valid_kv_len             12 bits, element count 0..4095
[34:33]  mode                     2 bits, bitmask
[32:0]   reserved                 33 bits, must be 0
```

Mode is a bitmask:

| Mode | Meaning |
| --- | --- |
| `0b00` | reserved; encoder/decoder may roundtrip it for diagnostic dumps, but executing `CONFIG_ATTN` with this mode faults before state is updated |
| `0b01` | padded mask enabled |
| `0b10` | causal mask enabled |
| `0b11` | padded + causal mask enabled |

For decoder attention, codegen emits `0b11` when the active key tile is wider
than the valid key length, and `0b10` only when the active key width exactly
equals `valid_kv_len`. ViT padded-only use cases use `0b01`.

`CONFIG_TILE.N` and `CONFIG_TILE.K` below refer to the encoded 10-bit
instruction fields, which store tile count minus one. The semantic tile widths
are therefore `(encoded_field + 1) * 16` (see
`software/docs/isa_spec.md` section 2.6).

Structural invariants (enforced by the golden model):

- The active softmax key width must cover `valid_kv_len` when either `0b01` or
  `0b11` is in effect. This is opcode-specific because the standalone and fused
  instructions interpret the tile dimensions differently:
  - For `MASKED_SOFTMAX`, `softmax_key_cols = (CONFIG_TILE.N + 1) * 16`.
  - For `MASKED_SOFTMAX_ATTNV`, `softmax_key_cols = (CONFIG_TILE.K + 1) * 16`.
  - The invariant is `softmax_key_cols >= valid_kv_len`.
  Without this, the mask would declare keys valid that the current tile does
  not actually contain.
- Pure causal mode (`0b10`) is legal only when there is no padded-key overhang:
  `softmax_key_cols == valid_kv_len`. If the active tile is wider than the
  valid key range, codegen must use `0b11` so padded/stale key columns are
  masked. The golden model faults on `mode = 0b10` with
  `softmax_key_cols != valid_kv_len`.
- `valid_kv_len > 0`. A `CONFIG_ATTN` with `valid_kv_len == 0` in any mode
  raises an illegal-instruction fault; the corresponding hardware state would
  be an undefined softmax.
- Reserved bits `[32:0]` and reserved mode `0b00` are never emitted by the
  compiler. If either appears during simulator execution, the instruction faults
  and does not set `attn_context.is_valid`.
- Codegen emits `CONFIG_TILE` before `CONFIG_ATTN` in each attention strip, so
  the `CONFIG_ATTN` validator can inspect the active tile shape before masked
  softmax executes.

### 4.2 Mask Semantics

`CONFIG_ATTN` persists until the next `CONFIG_ATTN` or simulator reset.
Executing `MASKED_SOFTMAX` or `MASKED_SOFTMAX_ATTNV` without a prior valid
`CONFIG_ATTN` is a golden-model fault, matching the style of missing
`CONFIG_TILE` faults.

For a tile-local row `i` and key column `j`, with `key_col_base = 0` in ISA
v1.1:

- If `mode & 0b01`, mask columns where `j >= valid_kv_len`.
- If `mode & 0b10`, mask columns where `j > query_row_base + i`.

`MASKED_SOFTMAX` pipeline:

1. The SFU loads logits from INT8 SRAM or INT32 ACCUM and dequantizes them to
   FP32 using the same input scale convention as `SOFTMAX`.
2. The mask is applied to FP32 logits by replacing masked entries with `-inf`
   before max-subtraction and `exp`.
3. FP32 probabilities are requantized to INT8 using the output scale register,
   exactly like the existing `SOFTMAX` path.

`MASKED_SOFTMAX_ATTNV` pipeline:

1. The SFU loads QK^T logits from ACCUM and V from INT8 SRAM.
2. QK^T is dequantized to FP32 using `S[sreg]`; V is dequantized using
   `S[sreg+1]`.
3. The mask is applied to FP32 QK^T logits before stable softmax.
4. FP32 softmax probabilities are multiplied by FP32 V.
5. The final `attn @ V` result is requantized to INT8 using `S[sreg+2]`.
6. `S[sreg+3]` remains the trace-only softmax probability scale, matching the
   existing `SOFTMAX_ATTNV` convention. It is not the architectural output
   scale of the fused instruction. Changing `S[sreg+3]` must not change the
   destination SRAM written by `MASKED_SOFTMAX_ATTNV`; it may only change the
   optional virtual softmax trace payload.

Row behavior:

- A row whose only unmasked entries are a proper, nonempty subset of the tile
  proceeds normally.
- With `key_col_base = 0` and `valid_kv_len > 0`, ISA v1.1 codegen should not
  intentionally produce fully masked rows. Padded-key masking hides columns,
  not padded query rows.
- If the golden model sees a row with no unmasked entries, it raises
  `ConfigError`. Under ISA v1.1 this indicates malformed attention context or a
  simulator/codegen bug; future `key_col_base` or local-key-tile extensions may
  choose to redefine this edge case.
- A `CONFIG_ATTN` with `valid_kv_len == 0` is the only "all-masked"
  configuration treated as a programmer error and raises a fault. This is a
  compile-time invariant violation, not a runtime condition.

### 4.3 Key-Base Limitation

ISA v1.1 deliberately omits `key_col_base`. This works for:

- Full prefill: query row strips attend to keys `0..seq_len-1`.
- Standard decode: query row is position `t`, keys are cache rows `0..t`.

It does not cover:

- Chunked prefill where the key tile starts beyond column zero
- Sliding-window attention where the window start is nonzero

Future extension: add a 12-bit `key_col_base` field in the reserved
`CONFIG_ATTN` bits. No new opcode is required.

### 4.4 Deliberate Non-Additions

- No `GATHER` opcode. Token and position embeddings are regular LOADs from
  host-patched addresses.
- No `KV_APPEND` opcode. KV cache writes are regular STOREs.
- No R-type flag expansion. R-type has no spare bits.
- No `ADDR_ADD` in the MVP. It remains reserved at `0x17` because it can reduce
  future decode address setup, but it is not required for golden-model
  correctness. For a 6-layer, 6-head decoder, a naive host-patched decode step
  can touch roughly `6 layers * 6 heads * K/V` KV address sites plus token and
  position embedding sites; `ADDR_ADD` would turn many of those full LO/HI
  rewrites into small address-register deltas.

## 5. Runtime and Memory Contracts

### 5.1 ProgramBundle

Decoder compilation produces a `ProgramBundle`, not two unrelated
`ProgramBinary` objects.

Reason: current `ProgramBinary` patches instruction addresses using
`data_base = align16(len(instruction_bytes))`. Independent prefill and decode
binaries can therefore get different `data_base` values, which breaks shared
weights and KV cache assumptions.

Bundle layout:

```text
0x0000
  prefill_instrs                  read-only during normal runs
  padding to 8 bytes
  decode_instrs                   host-patched before each decode step
  padding to 16 bytes
data_base
  shared_data                     weights, scales, biases, LN params,
                                  token embedding table, position table
temp_base
  temp                            per-invocation scratch / spill region
logits_base
  logits                          reserved output buffers; excluded from temp
                                  scratch reuse
kv_cache_base
  kv_cache                        persistent across prefill/decode calls
```

Bundle metadata:

- `prefill_instrs_offset`
- `decode_instrs_offset`
- `prefill_pc = prefill_instrs_offset // 8`
- `decode_pc = decode_instrs_offset // 8`
- `data_base`
- `temp_base`
- `logits_base`
- `logits_size`
- `kv_cache_base`
- `kv_cache_size`
- `required_dram_bytes`: minimum DRAM image size required for instructions,
  shared data, temp, logits/output buffers, and KV cache
- `input_offset`: prefill token-id input buffer
- `prefill_logits_offset`: last-token logits written at end of prefill
- `decode_logits_offset`: single-token logits written at end of each decode
  step; fixed across steps; host reads these bytes after `run_program` returns
- `relocation_sites`: static bundle relocations for data-relative addresses
- `runtime_patch_sites`: per-step host patches for token/position/KV addresses

Relocation record format:

```python
@dataclass
class RelocationSite:
    stream: str               # "prefill" | "decode"; retained for
                              # serialization/debugging even when local PCs
                              # are stored in per-stream lists
    local_lo_pc: int          # stream-local SET_ADDR_LO instruction index
    local_hi_pc: int          # stream-local paired SET_ADDR_HI instruction index
    addr_reg: int             # 0..3
    symbol: str               # data-relative symbol or DRAM layout key
```

Runtime patch site record format:

```python
@dataclass
class RuntimePatchSite:
    stream: str               # usually "decode"; "prefill" allowed for prompt setup
    kind: str                 # "token_embed" | "pos_embed" | "kv_base"
    local_lo_pc: int          # stream-local SET_ADDR_LO instruction index
    local_hi_pc: int          # stream-local paired SET_ADDR_HI instruction index
    absolute_lo_pc: int       # bundle PC after layout: stream_base_pc + local_lo_pc
    absolute_hi_pc: int       # bundle PC after layout: stream_base_pc + local_hi_pc
    addr_reg: int             # 0..3
    base_symbol: str          # embedding table, position table, or kv_cache base
```

Every relocation and runtime patch site names both the LO and HI instruction.
The patcher computes `new_addr = base + offset`, splits it into
`lo = new_addr & 0x0FFFFFFF` and `hi = (new_addr >> 28) & 0x0FFFFFFF`, and
patches both fields. Codegen emits a LO/HI pair at every patchable address
site even when the original base fits in 28 bits, so the runtime path is
uniform.

`absolute_lo_pc` and `absolute_hi_pc` are not required for bytearray patching,
which uses stream-local PCs. They are kept for trace manifests, diagnostics,
disassembly, and any future DRAM-resident instruction patching path where the
program is addressed as one unified instruction window.

The simulator PC is instruction-index based today, not byte-based. Therefore
`run_program("decode")` must set PC to `decode_instrs_offset // 8`, or the
simulator must grow an explicit byte-base fetch mode. Do not mix the two.

`ProgramBinary` remains valid for existing DeiT flows. `ProgramBundle` is the
decoder-specific container.

`ProgramBundle` owns simulator sizing for decoder runs: layout code computes
`required_dram_bytes` after instructions, shared data, maximum temp scratch,
logits/output buffers, and KV cache are placed. `Simulator.load_bundle` must
grow DRAM or raise a clear error if the configured DRAM cannot hold that many
bytes; silent truncation is a simulator bug.

`ProgramBundle` stores canonical pre-relocation `prefill_instrs` and
`decode_instrs` bytearrays. `load_bundle(bundle)` is idempotent by construction:
it copies those canonical bytearrays into a fresh image, applies static
relocation once to that fresh copy, writes instructions plus shared data into
DRAM, and initializes the bundle metadata. Loading the same logical bundle again
repeats relocation on a fresh copy; it must not add `data_base` to bytes that
were already relocated. `load_bundle` also sizes or validates simulator DRAM
against `required_dram_bytes` before writing the image.

### 5.2 Host Patching

The golden-model host runner keeps `prefill_instrs` and `decode_instrs` as
Python bytearrays. Before each decode step, it patches the bytearray and writes
only the decode instruction region back into simulator DRAM.

Patch full symbolic address pairs:

- Static bundle relocation patches every `RelocationSite` once during bundle
  layout.
- Runtime host patching patches every selected `RuntimePatchSite` before a
  prefill or decode invocation.
- Patch `SET_ADDR_LO` and `SET_ADDR_HI` together at every site.
- Recompute carry into the high 28 bits after adding `data_base` or dynamic
  offsets.
- The existing compiler `data_base` fix-up site (`compiler.py` around line 560)
  must be updated to emit LO+HI patches instead of LO-only. This is a Stage 3
  prerequisite, not a Stage 4 task. Existing DeiT happens to fit under the
  28-bit low-half window, but decoder bundles, large-vocab weights, synthetic
  relocation tests, and future larger models must be correct across a 28-bit
  carry boundary.

Invariant:

- `run_program` may rewrite the selected instruction region.
- `run_program` must not rewrite `[data_base, temp_base)` shared data.
- `run_program` must not rewrite the logits/output region except through the
  program STOREs that intentionally produce logits.
- `run_program` must not rewrite the KV cache except through program STORE
  instructions.

Add tests that snapshot shared-data and KV-cache regions before and after
host-patched decode runs.

### 5.3 Program State Lifetime

`load_bundle(bundle)` initializes the simulator with the bundle's instruction
streams and shared data. `run_program(bundle, name)` starts one invocation of
either `prefill` or `decode`.

At every `run_program` entry, reset volatile execution state:

- `pc` is set to the selected stream's entry PC.
- `halted` is cleared.
- `CONFIG_TILE` state is cleared.
- `CONFIG_ATTN` state is cleared and `attn_context.is_valid = False`.
- scale registers are cleared.
- address registers are cleared.

The following state persists across `run_program` calls:

- DRAM bytes
- instruction bytes
- shared data
- temp scratch contents
- logits/output region contents
- KV cache contents
- ABUF contents
- WBUF contents
- ACCUM contents

Temp scratch is not zeroed at program entry. A program that reads temp before
writing it observes stale bytes from a previous invocation. This is intentional;
program streams must initialize every temp region they consume.

SRAM buffers are not zeroed at program entry. Program streams must initialize
every ABUF, WBUF, and ACCUM region before reading it. In particular, MATMUL
accumulate mode (`flags=1`) is legal only after the same program stream has
initialized the destination ACCUM tile, typically with a prior MATMUL
`flags=0` or another explicit producer for that ACCUM region.

Because address registers reset at entry, every program stream must begin by
issuing `SET_ADDR_LO` and `SET_ADDR_HI` for each address register it uses. A
stream must not rely on address register values left by prefill or by a previous
decode step.

### 5.4 KV Cache Layout

Use a per-layer, per-kind, per-head layout so each attention head can load
contiguous K/V rows:

```text
kv_cache[layer][kind][head][position][d_head]
```

Where:

- `kind = 0` for K
- `kind = 1` for V
- row element type is INT8
- `d_head` must be a multiple of 16 for this MVP (enforced in
  `ModelConfig.__post_init__`)

Byte offset:

```text
kv_cache_base
  + (((layer * 2 + kind) * n_head + head) * max_seq_len + position) * d_head
```

This is logically equivalent to `[seq, d_model]`, but it avoids strided loads
when codegen works one head at a time.

### 5.5 KV Cache Scale Contract

The cache stores the native quantized output of each K/V projection.

For each layer and kind (`K` or `V`):

1. The projection MATMUL + REQUANT produces INT8 output using the calibrated
   K/V projection activation scale.
2. The program STOREs those INT8 bytes directly into the KV cache.
3. Later decode LOADs those bytes directly from the KV cache.
4. QK and attention-V paths use the same K/V activation scale downstream.

There is no extra requantization on cache store or cache load.

Default scale granularity is one activation scale per layer and kind. Per-head
K/V scales can be considered later, but are not required for the MVP.

Correctness test: a cached K/V row written during prefill must produce the same
attention result during decode as the corresponding row in a full-sequence
fake-quant reference.

## 6. Software Implementation Plan

### 6.1 Model Configuration

Add `software/taccel/compiler/model_config.py` with a `ModelConfig` dataclass:

- `model_kind`: `"encoder"` or `"decoder"`
- `n_layer`
- `n_head`
- `d_model`
- `d_head`
- `mlp_dim`
- `vocab_size`
- `max_seq_len`
- `embedding_kind`: `"patch_cls"` or `"token_pos"`
- `norm_epsilon`
- `weight_name_map`
- `activation_scale_policy`: one of `"single_set_unified"`,
  `"separate_prefill_decode"`, or `"per_head_kv"`. Defined concretely below.

`__post_init__` must validate:

- `d_model == n_head * d_head`
- `d_head % 16 == 0` (systolic tile dim and KV row alignment)
- `d_model % 16 == 0`
- `mlp_dim % 16 == 0`
- `max_seq_len <= 4095` (fits in `CONFIG_ATTN.valid_kv_len`)
- `model_kind == "decoder"` implies `embedding_kind == "token_pos"`

`activation_scale_policy` semantics:

- `"single_set_unified"` (MVP default): one calibrated scale set is used for
  both the prefill and decode instruction streams. The calibration pass merges
  prefill and decode observation distributions before computing scales.
- `"separate_prefill_decode"`: two scale sets; the compiler emits distinct
  FP16 scale constants into the shared data region, and each instruction
  stream loads its own set via `SET_SCALE` at program entry.
- `"per_head_kv"`: reserved; per-head KV-projection scales. Not wired up in
  Stage 3-4; `ModelConfig.__post_init__` must reject it with
  `NotImplementedError` until the compiler and calibration stack implement the
  policy.

Then thread `ModelConfig` through compiler and codegen:

- Remove direct compiler imports of DeiT constants.
- Replace `range(12)` with `range(config.n_layer)`.
- Replace `vit.encoder.layer.{i}.*` assumptions with config-derived weight
  names.
- Replace hard-coded `head_dim = 64` with `config.d_head`.
- Gate ViT startup ops (`cls_prepend`, `pos_embed_add`, `cls_extract`) behind
  `embedding_kind == "patch_cls"`.

### 6.2 Frontends

Add a frontend package:

- `software/taccel/compiler/frontend/__init__.py`
- `software/taccel/compiler/frontend/deit_plugin.py`
- `software/taccel/compiler/frontend/nanogpt_adapter.py`

Stage 1 frontend target:

- Accept a nanoGPT `GPT` module or `state_dict + config`.
- Walk the known nanoGPT module structure directly.
- Do not use `torch.fx` for the MVP.
- Emit `IRGraph + ModelConfig`.

Deferred frontend target:

- `hf_gpt2_adapter.py`
- Remap HuggingFace GPT-2 `Conv1D` weights explicitly.
- Handle HF attention/cache/mask conventions after the direct nanoGPT path
  works.

### 6.3 IR

The existing `IRNode` / `IRGraph` structure can remain unchanged. Add op string
constants or conventions for:

- `embed_lookup`
- `pos_embed_lookup`
- `config_attn`
- `masked_softmax`
- `masked_softmax_attnv`
- `kv_load`
- `kv_store`

Use `IRNode.attrs` for metadata such as:

- `layer`
- `head`
- `query_row_base`
- `valid_kv_len`
- `mask_mode`
- `cache_kind`
- `patch_site`
- `fused_attn`: bool, consumed by codegen to select the fused vs unfused
  attention emission path (see 6.4). The frontend sets a deterministic default
  from `ModelConfig` and sequence-length policy. A single canonical IR pass,
  `select_attention_realization`, may override it based on buffer pressure.
  Individual frontend adapters must not make incompatible final decisions.

### 6.4 Codegen

Attention path selection.

The IR has two attention realization modes, decided per layer by the
`fused_attn` attribute:

- Unfused path: emit `MATMUL` (QKᵀ) into ACCUM, then `MASKED_SOFTMAX` with
  the ACCUM tile as input, write INT8 probabilities back to ABUF, then a
  second `MATMUL` for probabilities times V. The `CONFIG_ATTN` applies to the
  `MASKED_SOFTMAX` call. This is the default and matches the current ViT
  `_emit_qkt` structure.
- Fused path: emit `MASKED_SOFTMAX_ATTNV` which takes the QKᵀ ACCUM tile,
  applies the mask, softmaxes, and fuses the probabilities-times-V matmul
  into one SFU-driven operation. The `CONFIG_ATTN` applies to the fused op.
  This replaces `SOFTMAX_ATTNV` wherever it was emitted previously.

Both paths must:

- Emit `CONFIG_TILE` before `CONFIG_ATTN` in each strip. The tile must still be
  active when the masked softmax instruction executes.
- Emit one `CONFIG_ATTN` per row strip inside the existing strip loop in
  `_emit_qkt`, before the op that consumes it.
- Use `query_row_base = row_start` for the current strip.
- Use `valid_kv_len = unpadded sequence length`.
- Ensure the active tile satisfies the opcode-specific key-width invariant:
  - Before `MASKED_SOFTMAX`, `(CONFIG_TILE.N + 1) * 16 >= valid_kv_len`.
  - Before `MASKED_SOFTMAX_ATTNV`, `(CONFIG_TILE.K + 1) * 16 >= valid_kv_len`.
- Choose `mode`:
  - `0b11` when the active softmax key width is greater than `valid_kv_len`
    and padded-key columns must be masked.
  - `0b10` only when the active softmax key width equals `valid_kv_len`.
- Never emit `mode = 0b00`.

Do not implement decoder attention by mirroring `_emit_softmax`; that is the
wrong integration point.

ViT padded-column leakage:

- Existing ViT padding can leak probability mass if padded columns are present.
- Fixing this through padded-only `CONFIG_ATTN` (`mode = 0b01`) is useful, but
  should be staged as a deliberate regression experiment.
- Decoder masking is required for Stage 2. ViT padded masking is optional in
  Stage 2 and must be measured with a pre/post DeiT accuracy diff.

Embedding lookup:

- Emit LOADs from token and position embedding tables.
- Record runtime patch sites (kind `"token_embed"` or `"pos_embed"`) with
  stream-local and absolute LO/HI pc entries, as in section 5.1.
- Patch full address pairs at runtime.

KV cache:

- Emit KV STOREs after K/V projection output is quantized.
- Emit KV LOADs before QK and attention-V in decode.
- Prefill stores each `(layer, kind, head)` K/V slice as one contiguous transfer
  with `xfer_len = prompt_len * d_head` and position 0 encoded in the DRAM
  address. It does not loop over prompt positions with host-patched stores.
  Prefill KV bank bases are `RelocationSite` records resolved during bundle
  layout, not runtime patch sites.
- Use a rolling decode-position base address for KV writes and reads where
  possible: the host patches `kv_step_base = kv_cache_bank_base + t * d_head`
  once per KV bank per decode step, and codegen encodes static layer/kind/head
  offsets in M-type `DRAM_OFF`.
- The allocator must ensure every static layer/kind/head offset from its
  `kv_step_base` fits the 16-bit M-type `DRAM_OFF` field. If not, it splits the
  KV cache into banks and emits one decode-stream runtime `kv_base` patch site
  per bank. Per-store runtime patching is not part of the MVP path.
- KV banking is deterministic. Bank boundaries fall on `(layer, kind)` groups in
  natural order: increasing layer, then K before V. Within a bank, heads are
  packed by increasing head index until adding the next head or group would make
  the largest static offset exceed the unsigned M-type `DRAM_OFF` reach
  (`0xFFFF * 16` bytes from the patched base). The allocator emits one
  `RuntimePatchSite` of kind `kv_base` per bank, and codegen/host_runner both
  consume that same bank table.
- This design keeps host-side patching proportional to the number of KV banks,
  not `2 * n_layer * n_head` stores per decode step.
- For the ISA v1.1 MVP envelope, deterministic banking should avoid per-store
  runtime patching. Per-store patching is documented only as a defensive future
  fallback for extensions such as different KV layouts, chunked prefill, or
  sliding-window attention.
- Use the per-head layout from section 5.4.

Matmul striping:

- Keep Q, K, and V as separate projections. At `d_model = 384`, each
  `384 x 384` projection is 147,456 bytes and fits WBUF. A fused QKV
  `384 x 1152` weight does not.
- For FC1 (`A[M,K] x W[K,N]`, `N = mlp_dim`), support both output-column
  slicing and K-axis slicing. Output-column slicing alone is not sufficient at
  `d_model = 384`: a slice with `N_tile = 1024` satisfies ACCUM
  (`16 * 1024 * 4 = 64 KB`) but the weight slice is
  `384 * 1024 = 393,216` bytes, which exceeds the 256 KB WBUF. Codegen must
  choose tile sizes that satisfy all constraints:
  - `K_tile * N_tile <= WBUF_BYTES` for the active weight slice.
  - `M_tile * N_tile * 4 <= ACCUM_BYTES` for INT32 accumulation.
  - `M_tile * N_tile <= ABUF_BYTES` for INT8 output.
  If `K_tile < K`, partial products accumulate into ACCUM across K chunks; the
  first chunk uses MATMUL `flags=0`, and later chunks use `flags=1`.
- The existing DeiT FC1 output spill path already writes strips to DRAM temp;
  reuse it and verify it handles `seq = 256, d_model = 384, mlp_dim = 1536`.
- For FC2, consume FC1 spills as K-axis chunks and accumulate partial products
  into ACCUM with MATMUL accumulate mode.
- Ensure every accumulator tile satisfies `M_tile * N_tile * 4 <= 64 KB`.

LM head striping:

- Deferred to Stage 5.
- Slice the vocab dimension.
- Write logits chunks to a DRAM logits buffer for host-side sampling. In the
  decode stream this buffer is located at `decode_logits_offset` and has fixed
  size `vocab_size * 4` bytes (INT32); in the prefill stream it is at
  `prefill_logits_offset` and stores the last-token logits only, same size.
  Both offsets live in the reserved logits/output region, not in reusable temp
  scratch.

### 6.5 Tiler and Memory Allocator

Tiler:

- Parameterize sequence length.
- Support `prefill` mode: `M = prompt_len`, padded to 16 for tiles.
- Support `decode` mode: `M = 1`, with K/V loaded from cache.
- Keep unpadded `valid_kv_len` available for `CONFIG_ATTN`.

Memory allocator:

- Add a persistent `kv_cache` DRAM region after temp.
- Add temp spill planning for FC1/FC2 activation strips.
- Compute the maximum temp scratch requirement for the bundle before placing
  any persistent post-temp regions. This includes FC1/FC2 spills, attention
  spills, and any per-program scratch needed at the maximum supported prompt
  length for the bundle.
- After the final temp size is known, reserve a logits/output region after temp
  scratch and before KV cache.
  `decode_logits_offset` and `prefill_logits_offset` are fixed offsets inside
  this region for the life of the bundle and are excluded from scratch reuse.
- Add WBUF ping-pong allocation for weight streaming where useful.
- Keep the existing SRAM offset units and 16-byte alignment rules.

### 6.6 ISA Surface

Update all instruction surfaces together:

- `software/taccel/isa/opcodes.py`
- `software/taccel/isa/instructions.py`
- `software/taccel/isa/encoding.py`
- `software/taccel/isa/__init__.py`
- `software/taccel/assembler/syntax.py`
- `software/taccel/assembler/disassembler.py`

New classes:

- `ConfigAttnInsn`
- R-type masked softmax instruction classes, or aliases using the current
  R-type instruction structure with new opcodes

Encoding rule:

- Add `InsnFormat.ATTN_TYPE` in `software/taccel/isa/opcodes.py` and map
  `Opcode.CONFIG_ATTN` to it in `OPCODE_FORMAT`. This keeps the opcode table,
  assembler, disassembler, and docs explicit even though the decoder still
  branches opcode-specifically for the field layout.
- The `ATTN_TYPE` encoder/decoder implements the bit layout from section 4.1:
  query row base in `[58:47]`, valid KV length in `[46:35]`, mode in `[34:33]`,
  and reserved zero bits in `[32:0]`.
- Decode opcode `0x14` before the existing C-type path.
- Decode opcodes `0x15` and `0x16` through the existing R-type field layout.

### 6.7 Golden Model

State:

- Add `attn_context` to `software/taccel/golden_model/state.py`.
- Fields: `query_row_base`, `valid_kv_len`, `mode`, `is_valid`.

SFU:

- Add `masked_softmax`.
- Add `masked_softmax_attnv`.
- Apply the mask to FP32 dequantized logits before max-subtraction, per
  section 4.2.
- Reuse the existing stable-softmax path for everything downstream of masking.
- Enforce the opcode-specific `softmax_key_cols >= valid_kv_len` invariant from
  section 4.1 for modes `0b01` and `0b11`. Mismatches are golden-model faults.
- For pure causal mode `0b10`, additionally fault if
  `softmax_key_cols != valid_kv_len`; a wider active key tile requires padded
  masking via mode `0b11`.
- Raise `ConfigError` if a masked row has no unmasked entries. This should be
  unreachable for compiler-generated ISA v1.1 programs with `key_col_base = 0`
  and `valid_kv_len > 0`.

Simulator:

- Dispatch `0x14` to validate `CONFIG_ATTN` and set `attn_context`. Invalid
  reserved bits, `mode = 0b00`, or `valid_kv_len = 0` fault before state is
  updated.
- Dispatch `0x15` and `0x16` to masked SFU functions.
- Add `load_bundle(bundle)`.
- Add `run_program(bundle, name)`.
- Ensure PC is set using instruction indices, not raw byte offsets.
- Size or validate DRAM against `ProgramBundle.required_dram_bytes` during
  `load_bundle`; never silently truncate a bundle image to the current DRAM
  allocation. Default golden-model behavior is to auto-grow DRAM up to a
  configurable cap of 1 GiB; if the bundle exceeds the cap, `load_bundle`
  raises. A `strict_dram_size=True` mode disables auto-growth and raises if the
  existing DRAM is too small.
- At every `run_program` entry, reset volatile execution state as specified in
  section 5.3: tile config, attention context, scale registers, address
  registers, halted flag, and PC.
- Preserve DRAM, instruction bytes, shared data, temp scratch, logits/output
  buffers, KV cache, and SRAM buffers between program invocations.

### 6.8 Quantization and Calibration

Calibration must include:

- Attention logits before masking
- K projection outputs per layer
- V projection outputs per layer
- Prefill and decode distributions, measured separately and kept as separate
  histograms

Scale policy is controlled by `ModelConfig.activation_scale_policy`:

- `"single_set_unified"` (MVP default): merge prefill and decode histograms
  per tensor and compute one scale. The two streams share the same FP16 scale
  constants loaded from `shared_data`.
- `"separate_prefill_decode"`: compute scales from each histogram independently
  and emit two scale constant blocks; each instruction stream loads its own
  via `SET_SCALE`. No ISA changes.

The calibration choice is decided once per model at calibration time and is
not runtime-switchable.

Calibration provenance is part of the test contract. Every end-to-end test must
record the calibration corpus, split, sample count, sequence length policy, and
whether the evaluation corpus overlaps the calibration corpus. GPT-2 perplexity
tests must not silently calibrate on the same evaluation slice.

Rounding and saturation are part of the numerical contract:

- Scale registers store FP16 values and are widened to FP32 before arithmetic.
- Requantization rounds to nearest with ties to even, matching NumPy/Python
  banker rounding and the current golden-model convention.
- INT8 outputs saturate to `[-128, 127]` after rounding.
- INT32 accumulators use the existing systolic/MATMUL accumulation semantics.

Golden verification should gate against a compiler-matched PyTorch fake-quant
reference first. In this plan, "fake-quant" means INT8 weights, INT8
activations, FP16 scale constants widened to FP32, the same rounding policy, the
same activation scale policy, and the same KV-cache quantization contract as the
compiler/golden-model path. Weight-only fake quant with FP32 activations is a
diagnostic reference, not a gate for Stage 3 or later. FP32 comparisons are
useful diagnostics but should not be the primary pass/fail criterion for
multi-op or end-to-end tests.

## 7. Staged Milestones

Each stage should produce a testable artifact. Do not advance to the next stage
until the current stage passes its gates.

### Stage 1: Parameterized Codegen and nanoGPT Adapter

No ISA changes, no masked-softmax opcode, no KV cache.

Deliverables:

- `ModelConfig` is threaded through compiler/codegen, including the validation
  rules in section 6.1.
- DeiT constants are removed from generic compiler/codegen paths.
- `nanogpt_adapter.py` emits `IRGraph + ModelConfig` for a 2-layer,
  `d_model = 128` nanoGPT model.
- A shipping-path 1-token nanoGPT forward pass compiles and runs in the golden
  model. For `seq = 1`, causal masking is an identity, so the existing softmax
  path is sufficient for this smoke test.
- A `seq = 16` non-attention decoder subgraph compiles and runs:
  token/position embedding, LayerNorm, FC1, GELU, FC2, residual/LN where
  applicable, and `lm_head`. This validates parameterization and MLP tiling
  without introducing a throwaway bidirectional-attention gate.
- A bidirectional `seq = 16` full-attention variant may remain as a debug knob,
  but it is not a Stage 1 gate.

Verification:

- Existing DeiT tests pass.
- `tests/test_nanogpt_frontend.py` runs the 1-token shipping-path forward pass
  and the `seq = 16` non-attention decoder subgraph against compiler-matched
  PyTorch fake-quant references.
- Per-op and single-block Stage 1 tests use byte-equal gates against a
  TACCEL-order fake-quant reference that mirrors the golden model's reduction
  order, FP16 scale widening, and rounding policy.
- Full 1-token forward and `seq = 16` non-attention subgraph gates:
  - Gate on logit cosine >= 0.999 and p99 absolute error <= 1 LSB against
    compiler-matched fake-quant.
  - Report byte-equal status; promote byte-equal to the primary gate only after
    a TACCEL-order compound reference exists for the full forward/subgraph.
- FP32 max-abs error is reported but not a gate.
- Multi-token causal correctness is not tested here by design.

### Stage 2: `CONFIG_ATTN` and Masked Softmax

Deliverables:

- Add opcodes `0x14`, `0x15`, `0x16`.
- Add encoding/decoding, assembler syntax, disassembler support, and exports.
- Add golden-model masked softmax implementations, including the
  opcode-specific `softmax_key_cols >= valid_kv_len` assertion.
- Modify `_emit_qkt` to emit per-strip `CONFIG_ATTN` and masked softmax for
  decoder attention, covering both fused and unfused paths.

Verification:

- `tests/test_isa_encoding.py`: roundtrip `CONFIG_ATTN`, `MASKED_SOFTMAX`, and
  `MASKED_SOFTMAX_ATTNV`.
- Test every legal emitted mode value (`0b01`, `0b10`, `0b11`), both boundary
  values for `query_row_base` and `valid_kv_len`, and the `valid_kv_len == 0`
  fault.
- Test that mode `0b00` is rejected by codegen emitters but roundtrips through
  the encoder/decoder so dumps of instrumented binaries still decode.
- Golden-model fault test: mode `0b10` faults when
  `softmax_key_cols != valid_kv_len`.
- Codegen behavior test: codegen selects `0b11` whenever padded-key overhang
  exists.
- `tests/test_masked_softmax.py`: compare to PyTorch softmax with padded,
  causal, and combined masks at sequence lengths `7, 16, 17, 31, 64, 128, 255`.
- Explicitly test that strip 2 uses rows `16..31`, not rows `0..15` again.
- Explicitly test that `softmax_key_cols < valid_kv_len` faults for both
  `MASKED_SOFTMAX` and `MASKED_SOFTMAX_ATTNV`.
- Explicitly test that changing `S[sreg+3]` for `MASKED_SOFTMAX_ATTNV` changes
  only the optional virtual softmax trace payload and does not change
  destination SRAM.
- `tests/test_block_nanogpt.py`: one GPT-2 decoder block at
  `seq = 64, d_model = 128, heads = 2` versus PyTorch fake-quant.
- Gate: logit cosine >= 0.999 and p99 absolute error <= 1 LSB against
  compiler-matched fake-quant. Report byte-equal status; promote byte-equal to
  the primary gate only after a TACCEL-order block reference exists.
- FP32 comparison is diagnostic only for block-level tests.

### Stage 3: ProgramBundle, KV Cache, and Host Runner

Implementation status: complete in the software/golden-model path. Stage 3 is
implemented as an internal decoder runtime path, not yet as the stable public
`Compiler.compile(model_kind="decoder")` API. The generated checkpoint artifact
is intentionally local-only and ignored by git; tests that depend on the default
fixture skip with the exact generation command when it is absent, while temp
fixture tests exercise the full path in CI/local regression runs.

Deliverables:

- `ProgramBundle` with shared data, temp, and persistent KV cache layout.
- Static `RelocationSite` records for bundle layout preserve the emitted
  `SET_ADDR_LO/HI` addend and relocate to `symbol_address + addend`.
- `RuntimePatchSite` records cover `token_embed`, `pos_embed`, and
  decode-stream per-bank KV bases (`kv_base`); each site records both LO and HI
  pc.
- Prefill KV bank bases are static `RelocationSite` records resolved at bundle
  layout; only decode emits runtime `kv_base` patch sites.
- Update `compiler.py:560` fix-up site to patch `SET_ADDR_LO` and
  `SET_ADDR_HI` with carry; synthetic relocation tests and future larger
  decoder layouts must be correct beyond 28-bit LO reach.
- `Simulator.load_bundle` with idempotent fresh-image relocation and
  `Simulator.run_program` with explicit volatile-state reset.
- Runtime host runner for prefill and decode.
- Token and position embedding patch sites consumed at runtime.
- KV STORE/LOAD codegen.
- Runtime `CONFIG_ATTN` patch sites for decode attention update
  `query_row_base = position` and `valid_kv_len = position + 1` per decode step.
- Decoder logits are written through explicit internal `logits_store` IR nodes
  into `prefill_logits_offset` / `decode_logits_offset`; Stage 3 stores INT8
  padded-vocabulary logits.
- KV cache scale contract enforced.
- Assembler syntax and disassembler coverage for all decode-path instructions
  is asserted in tests, not just compiled.

Verification:

- `tests/test_program_bundle.py`: instruction-region patching does not touch
  shared data or KV cache; full LO+HI patching preserves static relocation
  addends and is verified against carry-boundary synthetic relocation fixtures.
- `tests/test_kv_cache_scale.py`: cached K/V from prefill produces the same
  attention output during decode as a full-sequence fake-quant reference.
- `tests/test_kv_banking.py`: synthetic decoder config deliberately exceeds one
  M-type `DRAM_OFF` reach so `n_banks >= 2`. Assert bank boundaries follow the
  deterministic `(layer, kind)` natural-order rule, every static offset within a
  bank is `<= 0xFFFF * 16` bytes, and decode attention across a cross-bank
  transition matches the full-sequence fake-quant reference.
- `tests/test_logits_store.py`: codegen-emitted `logits_store` writes prefill
  and decode INT8 logits into the bundle logits region, and `HostRunner` reads
  them back.
- `tests/test_tiny_decode_smoke.py`: builds a full 1-token tiny decoder bundle
  from a generated fixture, runs prefill plus short decode, and verifies
  deterministic non-empty logits.
- `tests/test_e2e_tiny.py`: 32 greedy decode steps from the generated
  deterministic nanoGPT-shaped Shakespeare-character checkpoint. The fixture is
  reproducible from `software/tools/train_tiny_fixture.py` and records source
  snapshot, tokenizer, seed, hyperparameters, data ranges, validation-loss
  placeholder, and SHA-256. The checkpoint remains ignored by git.
- Gate: per-step logit cosine >= 0.99 against a fresh compiler-matched replay
  reference from the same quantized bundle path. Top-10 overlap >= 7/10 is
  asserted for the current replay gate. FP32/PyTorch reference comparison and a
  trained non-placeholder checkpoint remain future quality improvements, not
  Stage 3 blockers.

### Stage 4: `d_model = 384` Weight and Activation Striping

Deliverables:

- Weight striping for FC1/FC2 where weights exceed WBUF, including two-axis
  FC1 tiling when output-column slicing alone cannot fit the active weight
  slice.
- Activation spill path for FC1 outputs that exceed ABUF (extended from the
  existing DeiT-path spill mechanism).
- FC2 consumes spilled FC1 chunks correctly.
- Q/K/V remain separate projections.
- `software/run_nanogpt.py` for pretrained nanoGPT checkpoints.

Verification:

- `tests/test_weight_striping.py`: striped and unstriped matmul match at a
  smaller dimension where both are legal; includes an FC1-style case where
  N-only slicing would exceed WBUF and K-axis accumulation is required.
- `tests/test_activation_spill.py`: FC1 output at
  `seq = 256, d_model = 384, mlp_dim = 1536` spills and is consumed by FC2.
- `tests/test_e2e_nanogpt.py`: 64-step decode from a pretrained 6-layer
  `d_model = 384` nanoGPT.
- Gate: per-step logit cosine >= 0.995 against PyTorch fake-quant, and
  top-5 overlap >= 4/5 on 5 prompts. These are tighter than Stage 3 because
  the pretrained model has well-separated rankings and we expect the INT8
  path to preserve them.

### Stage 5: Large Vocab / GPT-2-Class Scale-Up

Deliverables:

- Vocab-dimension striping for `lm_head`.
- DRAM logits buffer for host-side sampling; reuses `decode_logits_offset`.
- `software/run_gpt2.py` entry point for a nanoGPT-format GPT-2-class
  checkpoint.
- Minimal checkpoint conversion path for GPT-2 124M weights:
  `software/tools/convert_hf_gpt2_to_nanogpt.py` converts HuggingFace GPT-2
  weights into the direct nanoGPT adapter's expected state_dict layout. This is
  a narrow offline converter, not the full HF frontend.
- Converter contract:
  - Input: either a HuggingFace `from_pretrained` model name such as `gpt2` or a
    local HF checkpoint directory containing config and weights.
  - Output: a nanoGPT-format `state_dict` plus a `ModelConfig` JSON using the
    exact keys consumed by `nanogpt_adapter.py`.
  - Weight layout: explicitly transpose HuggingFace `Conv1D` matrices from
    `[in, out]` storage to the adapter's expected linear projection layout.
  - Weight tying: preserve the GPT-2/nanoGPT token-embedding-to-`lm_head` tie by
    emitting one canonical weight reference or one canonical tensor plus
    metadata, not two silently diverging copies.
  - LayerNorm epsilon: copy HuggingFace GPT-2 `layer_norm_epsilon` into
    `ModelConfig.norm_epsilon` (GPT-2 default is `1e-5`).
  - Biases: preserve the GPT-2 attention/MLP bias tensors and set the nanoGPT
    bias/config flag consistently.
  - Dependencies: `transformers` is an offline conversion dependency only; it is
    not required by the golden-model runtime.

Verification:

- Coherent continuation smoke test on at least 3 prompts.
- `tests/test_gpt2_logits.py`: deterministic selected-token logits test on a
  small prompt set. Primary gate is p99 absolute error <= 1 LSB against the
  compiler-matched fake-quant reference. Byte-equal INT8 logits are reported as
  an aspirational secondary signal when the reference mirrors every TACCEL
  ordering detail.
- `tests/test_gpt2_perplexity.py`: WikiText-2 perplexity within 2 percent of
  the PyTorch fake-quant reference. During bring-up, 5 percent may be used as a
  temporary non-final threshold, but the final Stage 5 gate is 2 percent. The
  test must disclose the calibration corpus and must not calibrate on the same
  WikiText-2 slice used for evaluation.
- Stage 5 calibration logs per-row max-absolute values for the learned position
  embedding table and warns if high-position rows exceed the representable range
  implied by the selected per-tensor scale. This is diagnostic, not an ISA
  change.
- Perplexity within 10 percent of FP32 is reported as a diagnostic and is not
  a gate.

### Stage 6: Optional Follow-Ups

- Full HuggingFace GPT-2 frontend adapter with explicit `Conv1D` weight
  remapping and HF cache/mask convention handling. This general adapter is
  separate from the narrow Stage 5 offline checkpoint converter.
- `ADDR_ADD` at opcode `0x17` to reduce per-token address setup.
- `key_col_base` field in `CONFIG_ATTN`.
- RTL implementation of attention context and masked softmax.

## 8. Critical Files

Stage 1:

- `software/taccel/compiler/model_config.py`
- `software/taccel/compiler/compiler.py`
- `software/taccel/compiler/codegen.py`
- `software/taccel/compiler/graph_extract.py`
- `software/taccel/compiler/frontend/__init__.py`
- `software/taccel/compiler/frontend/deit_plugin.py`
- `software/taccel/compiler/frontend/nanogpt_adapter.py`
- `software/taccel/compiler/ir.py`
- `software/tests/test_nanogpt_frontend.py`

Stage 2:

- `software/taccel/isa/opcodes.py`
- `software/taccel/isa/instructions.py`
- `software/taccel/isa/encoding.py`
- `software/taccel/isa/__init__.py`
- `software/taccel/assembler/syntax.py`
- `software/taccel/assembler/disassembler.py`
- `software/taccel/golden_model/sfu.py`
- `software/taccel/golden_model/state.py`
- `software/taccel/golden_model/simulator.py`
- `software/taccel/compiler/codegen.py`
- `software/docs/isa_spec.md`
- `software/tests/test_isa_encoding.py`
- `software/tests/test_masked_softmax.py`
- `software/tests/test_block_nanogpt.py`

Stage 3:

- `software/taccel/assembler/assembler.py`
- `software/taccel/assembler/syntax.py`
- `software/taccel/assembler/disassembler.py`
- `software/taccel/golden_model/simulator.py`
- `software/taccel/compiler/memory_alloc.py`
- `software/taccel/compiler/tiler.py`
- `software/taccel/compiler/codegen.py`
- `software/taccel/compiler/compiler.py`
- `software/taccel/runtime/__init__.py`
- `software/taccel/runtime/host_runner.py`
- `software/taccel/quantizer/calibrate.py`
- `software/tools/train_tiny_fixture.py`
- `software/docs/llm_decoder_flow.md`
- `software/tests/test_program_bundle.py`
- `software/tests/test_kv_cache_scale.py`
- `software/tests/test_kv_banking.py`
- `software/tests/test_assembler_roundtrip.py`
- `software/tests/test_e2e_tiny.py`

Stage 4:

- `software/taccel/compiler/codegen.py`
- `software/taccel/compiler/memory_alloc.py`
- `software/run_nanogpt.py`
- `software/tests/test_weight_striping.py`
- `software/tests/test_activation_spill.py`
- `software/tests/test_e2e_nanogpt.py`

Stage 5:

- `software/taccel/compiler/codegen.py`
- `software/run_gpt2.py`
- `software/tools/convert_hf_gpt2_to_nanogpt.py`
- `software/tests/test_gpt2_logits.py`
- `software/tests/test_gpt2_perplexity.py`

Deferred RTL:

- `rtl/src/include/taccel_pkg.sv`
- `rtl/src/decode_unit.sv`
- `rtl/src/control_unit.sv`
- `rtl/src/sfu_engine.sv`
- `rtl/verilator/run_program.cpp`

Note: existing RTL and Verilator code currently treat `0x14..0x1F` as illegal.
That is acceptable while RTL is out of scope, but any RTL-side test path must be
updated before it can run ISA v1.1 binaries. RTL/Verilator CI for ISA v1.1
programs is explicitly skipped until Stage 6 RTL decode/control/SFU support is
implemented; existing ISA v1 RTL regression tests still run unchanged.

## 9. Verification Checklist

Opcode and assembly:

- Encoding roundtrip for `0x14`, `0x15`, `0x16`
- `InsnFormat.ATTN_TYPE` exists and `OPCODE_FORMAT[CONFIG_ATTN]` maps to it
- Every compiler-emitted `CONFIG_ATTN` has reserved bits `[32:0] == 0`
- Textual assembly roundtrip for all new mnemonics
- Disassembler output for all new mnemonics
- Fault on reserved `CONFIG_ATTN` bits
- Fault on executing `CONFIG_ATTN` with mode `0b00`; encoder/decoder roundtrip
  may still accept it for diagnostic dumps
- Fault on `valid_kv_len == 0`
- Fault on opcode-specific `softmax_key_cols < valid_kv_len` for padded-mask
  modes
- Fault on pure-causal mode `0b10` when `softmax_key_cols != valid_kv_len`
- Fault on masked softmax without valid attention context
- Codegen never emits mode `0b00`; encoder/decoder still roundtrips it for
  diagnostic dumps

Masked softmax:

- Padded-only mask
- Causal-only mask
- Padded + causal mask
- Non-multiple-of-16 sequence lengths
- Multi-strip row offsets (strip 2 must use rows 16..31)
- `valid_kv_len == 0` faults; normal compiler output does not rely on
  fully-masked rows
- Fully masked rows are a `ConfigError` in ISA v1.1

Compiler/codegen:

- No generic compiler path imports DeiT constants directly
- `config.d_head` replaces hard-coded head dimensions
- `ModelConfig.__post_init__` rejects bad invariants
- `ModelConfig.__post_init__` rejects reserved `activation_scale_policy`
  values such as `"per_head_kv"` until implemented
- Decoder `_emit_qkt` emits `CONFIG_ATTN` per row strip
- Fused vs unfused attention substitutions follow section 6.4 rules
- Runtime patch sites record stream-local and absolute LO/HI pcs
- Static relocation sites are separate from runtime patch sites
- `compiler.py` fix-up site patches LO and HI with carry
- Layouts with `data_base > (1 << 28)` round-trip correctly
- Codegen emits `CONFIG_TILE` before `CONFIG_ATTN` in every masked attention
  strip and satisfies the opcode-specific key-width invariant
- KV cache codegen uses rolling per-bank decode-position bases plus static
  layer/kind/head offsets where they fit M-type `DRAM_OFF`; per-store patching
  is not the MVP path
- KV bank boundaries follow the deterministic `(layer, kind)` natural-order rule
  from section 6.4, with offsets capped by the unsigned M-type `DRAM_OFF` reach

Runtime:

- ProgramBundle uses one shared `data_base`
- PC starts use instruction indices
- `run_program` resets volatile execution state: PC, halted flag, tile config,
  attention context, scale registers, and address registers
- DRAM persists across `run_program`, including instruction bytes, shared data,
  temp scratch, logits/output buffers, and KV cache
- SRAM buffers persist across `run_program`: ABUF, WBUF, and ACCUM are not
  cleared automatically
- Program streams initialize every SRAM region before read; ACCUM
  accumulate-mode is only used after the stream initializes the destination tile
- Every stream initializes each address register it uses with LO/HI setup
- Decode instruction patching does not clobber shared data or KV cache
- KV cache persists across multiple decode invocations
- Decode logits land at `decode_logits_offset` and are readable by the host
  after each `run_program` return
- Logits/output buffers are excluded from temp scratch reuse
- `load_bundle` applies static relocation to a fresh image and never
  double-applies relocation to already relocated bytes
- `ProgramBundle.required_dram_bytes` is populated and `load_bundle` grows or
  validates DRAM before writing bundle data
- Golden-model `load_bundle` auto-grows DRAM up to the configured cap by default
  and supports `strict_dram_size=True` for no-growth validation

Numerics:

- Rounding policy is pinned: round-half-to-even, then INT8 saturation to
  `[-128, 127]`
- `S[sreg+3]` for `MASKED_SOFTMAX_ATTNV` is trace-only; changing it must not
  change destination SRAM
- Per-op and single-block Stage 1 tests use byte-equal gates against a
  TACCEL-order fake-quant reference
- Full-model Stage 1 uses byte-equal only when the reference mirrors TACCEL
  order; otherwise it gates on cosine plus p99 <= 1 LSB
- Stage 2 decoder-block test gates on cosine >= 0.999 plus p99 <= 1 LSB;
  byte-equal is reported and promoted only when a TACCEL-order block reference
  exists
- Block and end-to-end tests gate against compiler-matched PyTorch fake-quant
- FP32 comparisons are diagnostic for larger tests
- Random-init end-to-end tests are avoided; trained checkpoints used for
  top-k-sensitive gates
- Stage 3 gates are looser than Stage 4 gates, reflecting model quality
- End-to-end tests disclose calibration corpus, split, sample count, and
  overlap policy
- Stage 5 includes a deterministic logits test with p99 <= 1 LSB before the
  WikiText-2 perplexity gate
- Stage 5 calibration logs learned position-embedding row ranges and warns on
  high-position clipping risk

Regression:

- Existing DeiT compiler, golden model, quantization, and compare tests still
  pass after Stage 1 and Stage 2.
- If ViT padded-column masking is enabled, record pre/post accuracy deltas.

## 10. Risks and Open Questions

Padded-column leakage:

- Current ViT attention may allow padded zero columns to receive softmax mass.
- Decoder masking must fix this for GPT.
- Applying padded masking to ViT should be measured separately.

Calibration split:

- Prefill logits have shape `seq x seq`.
- Decode logits have shape `1 x seq`.
- `activation_scale_policy` selects between a unified scale set and separate
  prefill/decode scale sets; the ISA does not change between them.

Position embedding range:

- Learned absolute position embeddings may have wider range than token
  embeddings.
- If per-tensor quantization clips high-position rows, consider per-row scales
  or a `REQUANT_PC`-style fallback.

Instruction patching:

- Bytearray patching is a golden-model convenience.
- RTL bring-up will likely need an inference descriptor, memory-mapped address
  registers, or a small bootloader path instead.

Attention context extension:

- `key_col_base` is not needed for the MVP but will be needed for chunked
  prefill and sliding-window attention.

Large vocab:

- GPT-2 124M `lm_head` is a separate scaling problem: `50257 x 768` INT8
  weights are about 38 MB, and INT32 logits are about 196 KB per token.
- Stage 5 handles this with vocab striping and a DRAM logits buffer.

DRAM sizing:

- Simulator DRAM must be sized from the `ProgramBundle` layout rather than a
  fixed small default. GPT-2-class bundles can require well over 64 MB once
  weights, embedding tables, logits buffers, temp spill space, and KV cache are
  included.

## 11. Recommended Implementation Order

1. Land `ModelConfig` with invariants and make DeiT still pass.
2. Add the direct nanoGPT adapter, a 1-token shipping-path forward test, and a
   `seq = 16` non-attention decoder-subgraph smoke test.
3. Add `CONFIG_ATTN` and masked softmax ISA/golden-model support, including
   opcode-specific `softmax_key_cols >= valid_kv_len` enforcement.
4. Modify `_emit_qkt` for decoder masked attention, covering fused and
   unfused paths.
5. Add `ProgramBundle`, separate static relocation and runtime patch-site
   records with LO+HI pairs, and fix the `compiler.py` data_base fix-up site to
   patch both halves.
6. Add simulator `run_program` APIs with explicit volatile-state reset and DRAM
   persistence semantics, then add KV cache layout, scale contract tests, and
   the host runner.
7. Add weight/activation striping for `d_model = 384`, reusing the existing
   FC1 spill path.
8. Add large-vocab `lm_head` striping and deterministic GPT-2 logits tests.
   Require `test_gpt2_logits.py` to pass before enabling
   `test_gpt2_perplexity.py` as the Stage 5 gate.
