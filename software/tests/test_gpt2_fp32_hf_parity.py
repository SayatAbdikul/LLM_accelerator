"""FP32 parity tests for HuggingFace GPT-2 conversion."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from software.tools.convert_hf_gpt2_to_nanogpt import convert_hf_state_dict_to_nanogpt
from taccel.runtime.fake_quant import cosine_similarity
from taccel.runtime.fp32_reference import NanoGPTFP32Reference


HF_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CONVERTED = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")


def _topk(logits: np.ndarray, k: int = 10) -> set[int]:
    active = np.asarray(logits, dtype=np.float32)
    return set(int(i) for i in np.lexsort((np.arange(active.size), -active))[:k])


def _summary(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    af = np.asarray(a, dtype=np.float32)
    bf = np.asarray(b, dtype=np.float32)
    return {
        "cosine": cosine_similarity(af, bf),
        "p99": float(np.percentile(np.abs(af - bf), 99.0)),
        "top10": len(_topk(af) & _topk(bf)),
    }


def _gelu(x, activation: str):
    if activation == "gelu_new":
        return F.gelu(x, approximate="tanh")
    return F.gelu(x)


def _tiny_hf_state(*, n_layer=2, n_head=2, d_model=8, vocab=17, block=16):
    gen = torch.Generator().manual_seed(1234)

    def randn(*shape, scale=0.03):
        return torch.randn(*shape, generator=gen, dtype=torch.float32) * scale

    state = {
        "transformer.wte.weight": randn(vocab, d_model),
        "transformer.wpe.weight": randn(block, d_model),
        "transformer.ln_f.weight": torch.ones(d_model) + randn(d_model, scale=0.01),
        "transformer.ln_f.bias": randn(d_model, scale=0.01),
    }
    for layer in range(n_layer):
        p = f"transformer.h.{layer}"
        state[f"{p}.ln_1.weight"] = torch.ones(d_model) + randn(d_model, scale=0.01)
        state[f"{p}.ln_1.bias"] = randn(d_model, scale=0.02)
        state[f"{p}.ln_2.weight"] = torch.ones(d_model) + randn(d_model, scale=0.01)
        state[f"{p}.ln_2.bias"] = randn(d_model, scale=0.02)
        state[f"{p}.attn.c_attn.weight"] = randn(d_model, 3 * d_model)
        state[f"{p}.attn.c_attn.bias"] = randn(3 * d_model, scale=0.02)
        state[f"{p}.attn.c_proj.weight"] = randn(d_model, d_model)
        state[f"{p}.attn.c_proj.bias"] = randn(d_model, scale=0.02)
        state[f"{p}.mlp.c_fc.weight"] = randn(d_model, 4 * d_model)
        state[f"{p}.mlp.c_fc.bias"] = randn(4 * d_model, scale=0.02)
        state[f"{p}.mlp.c_proj.weight"] = randn(4 * d_model, d_model)
        state[f"{p}.mlp.c_proj.bias"] = randn(d_model, scale=0.02)
    cfg = SimpleNamespace(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=d_model,
        vocab_size=vocab,
        n_positions=block,
        layer_norm_epsilon=1e-5,
        activation_function="gelu_new",
    )
    return state, cfg


def _functional_hf_logits(hf_state: dict[str, torch.Tensor], cfg, token_ids: list[int]) -> np.ndarray:
    with torch.no_grad():
        x = (
            hf_state["transformer.wte.weight"][torch.tensor(token_ids, dtype=torch.long)]
            + hf_state["transformer.wpe.weight"][torch.arange(len(token_ids), dtype=torch.long)]
        )
        d_model = int(cfg.n_embd)
        d_head = d_model // int(cfg.n_head)
        attn_scale = d_head ** -0.5
        for layer in range(int(cfg.n_layer)):
            p = f"transformer.h.{layer}"
            ln1 = F.layer_norm(
                x,
                (d_model,),
                weight=hf_state[f"{p}.ln_1.weight"],
                bias=hf_state[f"{p}.ln_1.bias"],
                eps=float(cfg.layer_norm_epsilon),
            )
            qkv = ln1 @ hf_state[f"{p}.attn.c_attn.weight"] + hf_state[f"{p}.attn.c_attn.bias"]
            q_all, k_all, v_all = torch.split(qkv, d_model, dim=-1)
            head_outs = []
            for head in range(int(cfg.n_head)):
                lo = head * d_head
                hi = (head + 1) * d_head
                q = q_all[:, lo:hi]
                k = k_all[:, lo:hi]
                v = v_all[:, lo:hi]
                attn = (q @ k.T) * attn_scale
                mask = torch.triu(torch.ones(len(token_ids), len(token_ids), dtype=torch.bool), diagonal=1)
                probs = F.softmax(attn.masked_fill(mask, float("-inf")), dim=-1)
                head_outs.append(probs @ v)
            concat = torch.cat(head_outs, dim=-1)
            x = x + concat @ hf_state[f"{p}.attn.c_proj.weight"] + hf_state[f"{p}.attn.c_proj.bias"]

            ln2 = F.layer_norm(
                x,
                (d_model,),
                weight=hf_state[f"{p}.ln_2.weight"],
                bias=hf_state[f"{p}.ln_2.bias"],
                eps=float(cfg.layer_norm_epsilon),
            )
            fc = ln2 @ hf_state[f"{p}.mlp.c_fc.weight"] + hf_state[f"{p}.mlp.c_fc.bias"]
            gelu = _gelu(fc, cfg.activation_function)
            x = x + gelu @ hf_state[f"{p}.mlp.c_proj.weight"] + hf_state[f"{p}.mlp.c_proj.bias"]

        ln_f = F.layer_norm(
            x,
            (d_model,),
            weight=hf_state["transformer.ln_f.weight"],
            bias=hf_state["transformer.ln_f.bias"],
            eps=float(cfg.layer_norm_epsilon),
        )
        logits = ln_f[-1:] @ hf_state["transformer.wte.weight"].T
        return logits[0].cpu().numpy().astype(np.float32)


def test_fp32_reference_matches_synthetic_hf_conv1d_with_split_qkv_biases():
    hf, cfg = _tiny_hf_state()
    payload = convert_hf_state_dict_to_nanogpt(hf, cfg)
    ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    tokens = [0, 3, 5, 2]

    got = ref.forward(tokens)
    expected = _functional_hf_logits(hf, cfg, tokens)
    metrics = _summary(got, expected)

    assert metrics["cosine"] >= 0.99999, metrics
    assert metrics["top10"] == 10, metrics
    assert metrics["p99"] <= 1e-5, metrics


def test_converted_local_gpt2_fp32_reference_matches_huggingface_when_present():
    if not HF_DIR.exists() or not CONVERTED.exists():
        pytest.skip("local HF GPT-2 directory or converted payload is missing")
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        pytest.skip(f"transformers is unavailable: {exc}")

    payload = torch.load(CONVERTED, map_location="cpu")
    ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    hf = GPT2LMHeadModel.from_pretrained(str(HF_DIR), local_files_only=True)
    hf.eval()
    prefixes = ([0], [464, 290, 262], [383, 318, 257, 1332])

    for tokens in prefixes:
        with torch.no_grad():
            ids = torch.tensor([list(tokens)], dtype=torch.long)
            expected = hf(input_ids=ids).logits[0, -1].detach().cpu().numpy().astype(np.float32)
        got = ref.forward(tokens)
        metrics = _summary(got, expected)
        assert metrics["cosine"] >= 0.99999, (tokens, metrics)
        assert metrics["top10"] == 10, (tokens, metrics)
        assert metrics["p99"] <= 2e-3, (tokens, metrics)
