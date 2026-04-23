#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights into this repo's nanoGPT payload layout.

The converter is intentionally narrow: it prepares an offline `.pt` artifact
for the existing nanoGPT adapter/runtime rather than adding a general HF
frontend dependency to the golden-model path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def _to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).contiguous()
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()


def _cfg(config: Any, name: str, default=None):
    if isinstance(config, Mapping) and name in config:
        return config[name]
    if hasattr(config, name):
        return getattr(config, name)
    if default is not None:
        return default
    raise KeyError(name)


def _cfg_any(config: Any, *names: str, default=None):
    for name in names:
        try:
            return _cfg(config, name)
        except KeyError:
            pass
    if default is not None:
        return default
    raise KeyError(names[0])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_hf_state_dict_to_nanogpt(
    hf_state: Mapping[str, Any],
    config: Any,
    *,
    text: str = "",
    stoi: Mapping[str, int] | None = None,
    itos: Mapping[int | str, str] | None = None,
) -> dict:
    """Return a nanoGPT-format payload from an HF GPT-2 state dict/config."""
    n_layer = int(_cfg_any(config, "n_layer", "num_hidden_layers"))
    n_head = int(_cfg_any(config, "n_head", "num_attention_heads"))
    d_model = int(_cfg_any(config, "n_embd", "hidden_size"))
    vocab_size = int(_cfg(config, "vocab_size"))
    block_size = int(_cfg_any(config, "n_positions", "max_position_embeddings", default=1024))
    norm_eps = float(_cfg(config, "layer_norm_epsilon", 1e-5))
    d_head = d_model // n_head
    if d_model % n_head != 0:
        raise ValueError("GPT-2 hidden size must be divisible by n_head")

    sd: dict[str, torch.Tensor] = {}
    sd["transformer.wte.weight"] = _to_tensor(hf_state["transformer.wte.weight"])
    sd["transformer.wpe.weight"] = _to_tensor(hf_state["transformer.wpe.weight"])
    sd["transformer.ln_f.weight"] = _to_tensor(hf_state["transformer.ln_f.weight"])
    sd["transformer.ln_f.bias"] = _to_tensor(hf_state["transformer.ln_f.bias"])
    # Preserve the GPT-2/nanoGPT weight tie by pointing both keys at the same tensor.
    sd["lm_head.weight"] = sd["transformer.wte.weight"]

    for layer in range(n_layer):
        prefix = f"transformer.h.{layer}"
        sd[f"{prefix}.ln_1.weight"] = _to_tensor(hf_state[f"{prefix}.ln_1.weight"])
        sd[f"{prefix}.ln_1.bias"] = _to_tensor(hf_state[f"{prefix}.ln_1.bias"])
        sd[f"{prefix}.ln_2.weight"] = _to_tensor(hf_state[f"{prefix}.ln_2.weight"])
        sd[f"{prefix}.ln_2.bias"] = _to_tensor(hf_state[f"{prefix}.ln_2.bias"])

        c_attn_w = _to_tensor(hf_state[f"{prefix}.attn.c_attn.weight"])
        c_attn_b = _to_tensor(hf_state[f"{prefix}.attn.c_attn.bias"])
        if tuple(c_attn_w.shape) != (d_model, 3 * d_model):
            raise ValueError(f"{prefix}.attn.c_attn.weight has shape {tuple(c_attn_w.shape)}")
        q_w_hf, k_w_hf, v_w_hf = torch.split(c_attn_w, d_model, dim=1)
        q_b, k_b, v_b = torch.split(c_attn_b, d_model, dim=0)
        for head in range(n_head):
            lo = head * d_head
            hi = (head + 1) * d_head
            sd[f"{prefix}.attn.c_attn.weight_h{head}_query"] = q_w_hf[:, lo:hi].T.contiguous()
            sd[f"{prefix}.attn.c_attn.weight_h{head}_key"] = k_w_hf[:, lo:hi].T.contiguous()
            sd[f"{prefix}.attn.c_attn.weight_h{head}_value"] = v_w_hf[:, lo:hi].T.contiguous()
            sd[f"{prefix}.attn.c_attn.bias_h{head}_query"] = q_b[lo:hi].contiguous()
            sd[f"{prefix}.attn.c_attn.bias_h{head}_key"] = k_b[lo:hi].contiguous()
            sd[f"{prefix}.attn.c_attn.bias_h{head}_value"] = v_b[lo:hi].contiguous()

        sd[f"{prefix}.attn.c_proj.weight"] = _to_tensor(hf_state[f"{prefix}.attn.c_proj.weight"]).T.contiguous()
        sd[f"{prefix}.attn.c_proj.bias"] = _to_tensor(hf_state[f"{prefix}.attn.c_proj.bias"])
        sd[f"{prefix}.mlp.c_fc.weight"] = _to_tensor(hf_state[f"{prefix}.mlp.c_fc.weight"]).T.contiguous()
        sd[f"{prefix}.mlp.c_fc.bias"] = _to_tensor(hf_state[f"{prefix}.mlp.c_fc.bias"])
        sd[f"{prefix}.mlp.c_proj.weight"] = _to_tensor(hf_state[f"{prefix}.mlp.c_proj.weight"]).T.contiguous()
        sd[f"{prefix}.mlp.c_proj.bias"] = _to_tensor(hf_state[f"{prefix}.mlp.c_proj.bias"])

    model_args = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": d_model,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "bias": True,
        "split_qkv_bias": True,
        "layer_norm_epsilon": norm_eps,
    }
    return {
        "state_dict": sd,
        "model_args": model_args,
        "text": text,
        "stoi": dict(stoi or {}),
        "itos": {str(k): v for k, v in dict(itos or {}).items()},
        "source": "huggingface-gpt2",
    }


def _load_hf_model(source: str):
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit(
            "transformers is required only for conversion; install it or pass a "
            "local converted fixture. Example: pip install transformers"
        ) from exc
    return GPT2LMHeadModel.from_pretrained(source)


def write_payload(payload: dict, output: Path, *, source: str) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    metadata = {
        "stage5_gpt2_converted": True,
        "source": source,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "model_args": payload["model_args"],
        "weight_tying": "lm_head.weight aliases transformer.wte.weight in payload",
        "format": "nanogpt_split_qkv",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="HF model name such as 'gpt2' or a local HF checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    model = _load_hf_model(args.source)
    payload = convert_hf_state_dict_to_nanogpt(model.state_dict(), model.config)
    metadata = write_payload(payload, args.output, source=args.source)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
