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


def build_kv_cache_layout(config: ModelConfig, max_seq_len: int = None) -> KVCacheLayout:
    """Build the Stage 3 deterministic per-head KV-cache layout."""
    seq_len = int(max_seq_len or config.max_seq_len)
    if seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if seq_len > config.max_seq_len:
        raise ValueError("max_seq_len cannot exceed ModelConfig.max_seq_len")
    if config.d_head % 16 != 0:
        raise ValueError("d_head must be 16-byte aligned for KV cache layout")

    head_span = seq_len * config.d_head
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
    )
