"""Per-op emit package.

Splits the original `codegen.py` op-emitter methods into op-family
modules. Each public `emit_*` function takes the owning
`CodeGenerator` instance as its first argument; `codegen.py` keeps
thin dispatcher methods (`_emit_dma_load`, `_emit_qkt`, ...) for
backward compatibility with external callers.

Sub-modules:
  - `_common`    — shared helpers + `UNIT` constant
  - `dma`        — SET_ADDR + LOAD / STORE
  - `attn`       — CONFIG_ATTN-mode helpers
  - `kv`         — kv_store / kv_load / logits_store
  - `embedding`  — embedding lookups + CLS prepend/extract + pos-embed-add
  - `sfu`        — scale_mul / softmax / gelu / layernorm / vadd
  - `matmul`     — Q@K^T, softmax·V, head concat
  - `batch`      — row_copy / gather_rows (per-stream row extract for
                   batched decode)
"""
from .attn import attention_mask_mode_for_qkt, emit_config_attn_for_qkt
from .dma import emit_dma_load, emit_dma_store
from .embedding import (
    emit_cls_extract,
    emit_cls_prepend,
    emit_embedding_lookup,
    emit_pos_embed_add,
)
from .kv import (
    emit_kv_load,
    emit_kv_store,
    emit_logits_store,
    kv_entry_for_node,
    kv_source_location,
    kv_transfer_bytes,
)
from .matmul import emit_attn_v, emit_concat_heads, emit_qkt
from .sfu import (
    emit_gelu,
    emit_gelu_from_dram_temp,
    emit_layernorm,
    emit_scale_mul,
    emit_softmax,
    emit_vadd,
)

__all__ = [
    "attention_mask_mode_for_qkt",
    "emit_attn_v",
    "emit_cls_extract",
    "emit_cls_prepend",
    "emit_concat_heads",
    "emit_config_attn_for_qkt",
    "emit_dma_load",
    "emit_dma_store",
    "emit_embedding_lookup",
    "emit_gelu",
    "emit_gelu_from_dram_temp",
    "emit_kv_load",
    "emit_kv_store",
    "emit_layernorm",
    "emit_logits_store",
    "emit_pos_embed_add",
    "emit_qkt",
    "emit_scale_mul",
    "emit_softmax",
    "emit_vadd",
    "kv_entry_for_node",
    "kv_source_location",
    "kv_transfer_bytes",
]
