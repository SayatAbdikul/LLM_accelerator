import pytest

from taccel.compiler.model_config import ModelConfig, deit_tiny_config


def _valid_decoder_config(**overrides):
    values = dict(
        model_kind="decoder",
        n_layer=2,
        n_head=2,
        d_model=128,
        d_head=64,
        mlp_dim=512,
        vocab_size=256,
        max_seq_len=128,
        embedding_kind="token_pos",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_deit_tiny_config_matches_legacy_constants():
    config = deit_tiny_config()

    assert config.model_kind == "encoder"
    assert config.embedding_kind == "patch_cls"
    assert config.n_layer == 12
    assert config.n_head == 3
    assert config.d_model == 192
    assert config.d_head == 64
    assert config.mlp_dim == 768
    assert config.vocab_size == 1000
    assert config.max_seq_len == 197


def test_rejects_invalid_d_model_product():
    with pytest.raises(ValueError, match="d_model must equal"):
        _valid_decoder_config(d_model=96, n_head=2, d_head=64)


def test_rejects_invalid_d_head_alignment():
    with pytest.raises(ValueError, match="d_head must be a multiple"):
        _valid_decoder_config(d_model=48, n_head=2, d_head=24)


def test_rejects_invalid_d_model_alignment():
    with pytest.raises(ValueError, match="d_model must be a multiple"):
        _valid_decoder_config(d_model=120, n_head=3, d_head=40)


def test_rejects_invalid_mlp_alignment():
    with pytest.raises(ValueError, match="mlp_dim must be a multiple"):
        _valid_decoder_config(mlp_dim=510)


def test_rejects_long_context_for_v1_1():
    with pytest.raises(ValueError, match="max_seq_len must be <= 4095"):
        _valid_decoder_config(max_seq_len=4096)


def test_decoder_requires_token_pos_embeddings():
    with pytest.raises(ValueError, match="decoder models require token_pos"):
        _valid_decoder_config(embedding_kind="patch_cls")


def test_per_head_kv_policy_is_deferred():
    with pytest.raises(NotImplementedError, match="per_head_kv"):
        _valid_decoder_config(activation_scale_policy="per_head_kv")


def test_separate_prefill_decode_policy_is_deferred():
    with pytest.raises(NotImplementedError, match="separate_prefill_decode"):
        _valid_decoder_config(activation_scale_policy="separate_prefill_decode")
