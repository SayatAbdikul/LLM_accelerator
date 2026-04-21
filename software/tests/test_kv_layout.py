"""Stage 3 tests for deterministic KV-cache layout."""
from taccel.compiler.kv_cache import M_DRAM_OFF_REACH_BYTES, build_kv_cache_layout
from taccel.compiler.model_config import ModelConfig


def _decoder_config(**overrides):
    params = dict(
        name="kv-test",
        model_kind="decoder",
        n_layer=1,
        n_head=2,
        d_model=32,
        d_head=16,
        mlp_dim=64,
        vocab_size=32,
        max_seq_len=8,
        embedding_kind="token_pos",
    )
    params.update(overrides)
    return ModelConfig(**params)


def test_kv_layout_size_and_offsets():
    config = _decoder_config(n_layer=2, n_head=2, d_model=32, d_head=16, max_seq_len=4)

    layout = build_kv_cache_layout(config)

    assert layout.kv_cache_size == 2 * 2 * 2 * 4 * 16
    assert len(layout.entries) == 8
    assert layout.entry(0, "key", 0).byte_offset == 0
    assert layout.entry(0, "key", 1).byte_offset == 4 * 16
    assert layout.entry(0, "value", 0).byte_offset == 2 * 4 * 16
    assert layout.entry(1, "key", 0).byte_offset == 4 * 4 * 16
    assert layout.entry(1, "value", 1).byte_offset == 7 * 4 * 16


def test_kv_layout_forces_deterministic_multibank_split():
    config = _decoder_config(
        n_layer=2,
        n_head=64,
        d_model=1024,
        d_head=16,
        mlp_dim=1024,
        max_seq_len=1024,
    )

    layout = build_kv_cache_layout(config)

    assert len(layout.banks) >= 2
    assert [bank.bank_id for bank in layout.banks] == list(range(len(layout.banks)))
    assert [bank.base_symbol for bank in layout.banks] == [
        f"kv_bank{i}" for i in range(len(layout.banks))
    ]
    for entry in layout.entries:
        assert entry.bank_offset <= M_DRAM_OFF_REACH_BYTES
        assert entry.dram_off_units <= 0xFFFF

    # Natural order is layer, then K before V, then head index.
    assert layout.entries[0].layer == 0
    assert layout.entries[0].kind == "key"
    assert layout.entries[0].head == 0
    assert layout.entries[64].layer == 0
    assert layout.entries[64].kind == "value"
    assert layout.entries[64].head == 0


def test_kv_layout_normalizes_kind_aliases():
    layout = build_kv_cache_layout(_decoder_config())

    assert layout.entry(0, "k", 0) == layout.entry(0, "key", 0)
    assert layout.entry(0, "v", 1) == layout.entry(0, "value", 1)
