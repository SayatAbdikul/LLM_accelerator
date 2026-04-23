"""Tests for the narrow HuggingFace GPT-2 -> nanoGPT converter."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from software.tools.convert_hf_gpt2_to_nanogpt import convert_hf_state_dict_to_nanogpt


def _hf_state(n_layer=1, n_head=2, d_model=8, vocab=13, block=16):
    state = {
        "transformer.wte.weight": torch.arange(vocab * d_model, dtype=torch.float32).reshape(vocab, d_model),
        "transformer.wpe.weight": torch.arange(block * d_model, dtype=torch.float32).reshape(block, d_model) / 100,
        "transformer.ln_f.weight": torch.ones(d_model),
        "transformer.ln_f.bias": torch.zeros(d_model),
    }
    for layer in range(n_layer):
        p = f"transformer.h.{layer}"
        state[f"{p}.ln_1.weight"] = torch.ones(d_model)
        state[f"{p}.ln_1.bias"] = torch.arange(d_model, dtype=torch.float32)
        state[f"{p}.ln_2.weight"] = torch.ones(d_model) * 2
        state[f"{p}.ln_2.bias"] = torch.arange(d_model, dtype=torch.float32) + 10
        state[f"{p}.attn.c_attn.weight"] = torch.arange(d_model * 3 * d_model, dtype=torch.float32).reshape(d_model, 3 * d_model)
        state[f"{p}.attn.c_attn.bias"] = torch.arange(3 * d_model, dtype=torch.float32)
        state[f"{p}.attn.c_proj.weight"] = torch.arange(d_model * d_model, dtype=torch.float32).reshape(d_model, d_model) + 1000
        state[f"{p}.attn.c_proj.bias"] = torch.arange(d_model, dtype=torch.float32) + 20
        state[f"{p}.mlp.c_fc.weight"] = torch.arange(d_model * 4 * d_model, dtype=torch.float32).reshape(d_model, 4 * d_model) + 2000
        state[f"{p}.mlp.c_fc.bias"] = torch.arange(4 * d_model, dtype=torch.float32) + 30
        state[f"{p}.mlp.c_proj.weight"] = torch.arange(4 * d_model * d_model, dtype=torch.float32).reshape(4 * d_model, d_model) + 3000
        state[f"{p}.mlp.c_proj.bias"] = torch.arange(d_model, dtype=torch.float32) + 40
    cfg = SimpleNamespace(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=d_model,
        vocab_size=vocab,
        n_positions=block,
        layer_norm_epsilon=1e-5,
    )
    return state, cfg


def test_converter_transposes_conv1d_splits_qkv_and_preserves_metadata():
    hf, cfg = _hf_state()
    payload = convert_hf_state_dict_to_nanogpt(hf, cfg)
    sd = payload["state_dict"]
    args = payload["model_args"]
    d_head = args["n_embd"] // args["n_head"]

    assert args["split_qkv_bias"] is True
    assert args["layer_norm_epsilon"] == 1e-5
    assert sd["lm_head.weight"] is sd["transformer.wte.weight"]

    q0 = sd["transformer.h.0.attn.c_attn.weight_h0_query"]
    expected_q0 = hf["transformer.h.0.attn.c_attn.weight"][:, :d_head].T
    torch.testing.assert_close(q0, expected_q0)

    k1 = sd["transformer.h.0.attn.c_attn.weight_h1_key"]
    key_start = args["n_embd"]
    expected_k1 = hf["transformer.h.0.attn.c_attn.weight"][
        :, key_start + d_head:key_start + 2 * d_head
    ].T
    torch.testing.assert_close(k1, expected_k1)

    torch.testing.assert_close(
        sd["transformer.h.0.attn.c_proj.weight"],
        hf["transformer.h.0.attn.c_proj.weight"].T,
    )
    torch.testing.assert_close(
        sd["transformer.h.0.mlp.c_fc.weight"],
        hf["transformer.h.0.mlp.c_fc.weight"].T,
    )
    torch.testing.assert_close(
        sd["transformer.h.0.attn.c_attn.bias_h0_value"],
        hf["transformer.h.0.attn.c_attn.bias"][2 * args["n_embd"]:2 * args["n_embd"] + d_head],
    )


def test_local_converted_gpt2_metadata_when_present():
    meta_path = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt.json")
    if not meta_path.exists():
        pytest.skip("local converted GPT-2 metadata is not present")

    metadata = json.loads(meta_path.read_text())
    args = metadata["model_args"]
    assert metadata["stage5_gpt2_converted"] is True
    assert args["n_layer"] == 12
    assert args["n_head"] == 12
    assert args["n_embd"] == 768
    assert args["vocab_size"] == 50257
    assert args["block_size"] == 1024
    assert args["layer_norm_epsilon"] == 1e-5
    assert args["split_qkv_bias"] is True
    assert "checkpoint_sha256" in metadata
    assert "lm_head.weight aliases transformer.wte.weight" in metadata["weight_tying"]
