"""Deterministic decoder KV-cache layout helpers."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .model_config import ModelConfig


M_DRAM_OFF_REACH_BYTES = 0xFFFF * 16
KV_KINDS = ("key", "value")


def normalize_kv_kind(kind: str) -> str:
    aliases = {
        "k": "key",
        "key": "key",
        "v": "value",
        "value": "value",
    }
    try:
        return aliases[str(kind).lower()]
    except KeyError as exc:
        raise ValueError("KV kind must be one of 'key', 'value', 'k', or 'v'") from exc


@dataclass(frozen=True)
class KVCacheEntry:
    bank_id: int
    base_symbol: str
    layer: int
    kind: str
    head: int
    byte_offset: int
    bank_offset: int
    dram_off_units: int
    span_bytes: int


@dataclass(frozen=True)
class KVCacheBank:
    bank_id: int
    base_symbol: str
    base_offset: int
    size_bytes: int


@dataclass(frozen=True)
class KVCacheLayout:
    max_seq_len: int
    d_head: int
    n_layer: int
    n_head: int
    kv_cache_size: int
    banks: Tuple[KVCacheBank, ...]
    entries: Tuple[KVCacheEntry, ...]
    # M4-B: per-element bytes for the K/V tile element type. 1 byte for
    # INT8 (default; legacy W8A8 path), 4 bytes for FP32 (W8A32 weight-
    # only quantization where K/V tiles are post-DEQUANT FP32). All
    # byte-stride math (head_span, kv_cache_size, per-token offset
    # `position * d_head * elem_bytes`) flows from this field.
    elem_bytes: int = 1

    def entry(self, layer: int, kind: str, head: int) -> KVCacheEntry:
        key = (int(layer), normalize_kv_kind(kind), int(head))
        for entry in self.entries:
            if (entry.layer, entry.kind, entry.head) == key:
                return entry
        raise KeyError(f"No KV cache entry for layer={layer}, kind={kind!r}, head={head}")

    @property
    def bank_symbols(self) -> Dict[str, int]:
        return {
            bank.base_symbol: bank.base_offset
            for bank in self.banks
        }

    def entries_for_bank(self, bank_id: int) -> Iterable[KVCacheEntry]:
        return (entry for entry in self.entries if entry.bank_id == bank_id)


def build_kv_cache_layout(
    config: ModelConfig,
    max_seq_len: int = None,
    *,
    elem_bytes: int = 1,
) -> KVCacheLayout:
    """Build the Stage 3 deterministic per-head KV-cache layout.

    `elem_bytes` (M4-B): 1 for INT8 K/V (default; legacy W8A8 path), 4
    for FP32 K/V (W8A32 weight-only path). All byte-stride math scales
    by this factor: each head's row stride per token is `d_head *
    elem_bytes`, each head's total span is `seq_len * d_head *
    elem_bytes`, and `kv_cache_size = n_layer * 2 * n_head * head_span`.
    The bank-reach check `M_DRAM_OFF_REACH_BYTES = 0xFFFF * 16` is
    applied to the resulting (scaled) byte offsets — so W8A32 banks
    will split sooner than INT8 banks at the same `(n_layer, n_head,
    seq_len)`. Codegen DMAs (`_kv_transfer_bytes`) and the bundle's
    `kv_step_bytes` runtime patch field both scale by `elem_bytes`.
    """
    seq_len = int(max_seq_len or config.max_seq_len)
    if seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if seq_len > config.max_seq_len:
        raise ValueError("max_seq_len cannot exceed ModelConfig.max_seq_len")
    if config.d_head % 16 != 0:
        raise ValueError("d_head must be 16-byte aligned for KV cache layout")
    if elem_bytes not in (1, 4):
        raise ValueError(f"elem_bytes must be 1 (INT8) or 4 (FP32), got {elem_bytes}")

    head_span = seq_len * config.d_head * elem_bytes
    kv_cache_size = config.n_layer * len(KV_KINDS) * config.n_head * head_span

    banks: List[KVCacheBank] = []
    entries: List[KVCacheEntry] = []
    current_bank_id = -1
    current_bank_base = 0
    current_bank_end = 0

    def start_bank(base_offset: int):
        nonlocal current_bank_id, current_bank_base, current_bank_end
        if current_bank_id >= 0:
            banks.append(KVCacheBank(
                bank_id=current_bank_id,
                base_symbol=f"kv_bank{current_bank_id}",
                base_offset=current_bank_base,
                size_bytes=current_bank_end - current_bank_base,
            ))
        current_bank_id += 1
        current_bank_base = base_offset
        current_bank_end = base_offset

    for layer in range(config.n_layer):
        for kind_idx, kind in enumerate(KV_KINDS):
            for head in range(config.n_head):
                absolute_offset = (
                    ((layer * len(KV_KINDS) + kind_idx) * config.n_head + head) * head_span
                )
                if current_bank_id < 0:
                    start_bank(absolute_offset)
                bank_offset = absolute_offset - current_bank_base
                if bank_offset > M_DRAM_OFF_REACH_BYTES:
                    start_bank(absolute_offset)
                    bank_offset = 0
                dram_off_units = bank_offset // 16
                entries.append(KVCacheEntry(
                    bank_id=current_bank_id,
                    base_symbol=f"kv_bank{current_bank_id}",
                    layer=layer,
                    kind=kind,
                    head=head,
                    byte_offset=absolute_offset,
                    bank_offset=bank_offset,
                    dram_off_units=dram_off_units,
                    span_bytes=head_span,
                ))
                current_bank_end = max(current_bank_end, absolute_offset + head_span)

    if current_bank_id >= 0:
        banks.append(KVCacheBank(
            bank_id=current_bank_id,
            base_symbol=f"kv_bank{current_bank_id}",
            base_offset=current_bank_base,
            size_bytes=current_bank_end - current_bank_base,
        ))

    return KVCacheLayout(
        max_seq_len=seq_len,
        d_head=config.d_head,
        n_layer=config.n_layer,
        n_head=config.n_head,
        kv_cache_size=kv_cache_size,
        banks=tuple(banks),
        entries=tuple(entries),
        elem_bytes=elem_bytes,
    )
