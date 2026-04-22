"""True FP32 nanoGPT reference used for rank-based e2e diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class FP32DecodeTrace:
    generated: List[int]
    logits: List[np.ndarray]


def _as_f32_tensor(torch, tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().to(dtype=torch.float32)
    return torch.tensor(tensor, dtype=torch.float32)


class NanoGPTFP32Reference:
    """PyTorch FP32 nanoGPT reference for the repo's split-head checkpoint layout."""

    def __init__(self, state_dict: dict, model_args: dict) -> None:
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.n_layer = int(model_args["n_layer"])
        self.n_head = int(model_args["n_head"])
        self.d_model = int(model_args["n_embd"])
        self.d_head = self.d_model // self.n_head
        self.vocab_size = int(model_args["vocab_size"])
        self.block_size = int(model_args.get("block_size", 0))
        self.eps = float(model_args.get("layer_norm_epsilon", 1e-5))
        self.attn_scale = float(self.d_head ** -0.5)

        sd = state_dict
        self.wte = _as_f32_tensor(torch, sd["transformer.wte.weight"])
        self.wpe = _as_f32_tensor(torch, sd["transformer.wpe.weight"])
        self.ln_f_w = _as_f32_tensor(torch, sd["transformer.ln_f.weight"])
        self.ln_f_b = _as_f32_tensor(torch, sd["transformer.ln_f.bias"])
        self.lm_head_w = _as_f32_tensor(
            torch, sd.get("lm_head.weight", sd["transformer.wte.weight"])
        )

        self.layers = []
        for layer_idx in range(self.n_layer):
            heads = []
            for head_idx in range(self.n_head):
                heads.append((
                    _as_f32_tensor(
                        torch,
                        sd[f"transformer.h.{layer_idx}.attn.c_attn.weight_h{head_idx}_query"],
                    ),
                    _as_f32_tensor(
                        torch,
                        sd[f"transformer.h.{layer_idx}.attn.c_attn.weight_h{head_idx}_key"],
                    ),
                    _as_f32_tensor(
                        torch,
                        sd[f"transformer.h.{layer_idx}.attn.c_attn.weight_h{head_idx}_value"],
                    ),
                ))
            self.layers.append({
                "ln1_w": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.ln_1.weight"]),
                "ln1_b": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.ln_1.bias"]),
                "ln2_w": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.ln_2.weight"]),
                "ln2_b": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.ln_2.bias"]),
                "heads": heads,
                "c_proj_w": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.attn.c_proj.weight"]),
                "c_proj_b": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.attn.c_proj.bias"]),
                "fc_w": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.mlp.c_fc.weight"]),
                "fc_b": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.mlp.c_fc.bias"]),
                "proj_w": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.mlp.c_proj.weight"]),
                "proj_b": _as_f32_tensor(torch, sd[f"transformer.h.{layer_idx}.mlp.c_proj.bias"]),
            })

    def _layer_norm(self, x, w, b):
        return self.F.layer_norm(x, (self.d_model,), weight=w, bias=b, eps=self.eps)

    def _logits(self, x):
        ln_f = self._layer_norm(x, self.ln_f_w, self.ln_f_b)
        return ln_f[-1:] @ self.lm_head_w.T

    def forward(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Return FP32 logits for the last position."""
        with self.torch.no_grad():
            tids = [int(tok) for tok in token_ids]
            if not tids:
                raise ValueError("token_ids must be non-empty")
            pids = list(position_ids) if position_ids is not None else list(range(len(tids)))
            if len(pids) != len(tids):
                raise ValueError("position_ids length must match token_ids length")

            tok = self.torch.tensor(tids, dtype=self.torch.long)
            pos = self.torch.tensor(pids, dtype=self.torch.long)
            x = self.wte[tok] + self.wpe[pos]
            seq = x.shape[0]

            for layer in self.layers:
                ln1 = self._layer_norm(x, layer["ln1_w"], layer["ln1_b"])
                head_outs = []
                for q_w, k_w, v_w in layer["heads"]:
                    q = ln1 @ q_w.T
                    k = ln1 @ k_w.T
                    v = ln1 @ v_w.T
                    attn = (q @ k.T) * self.attn_scale
                    mask = self.torch.triu(
                        self.torch.ones(seq, seq, dtype=self.torch.bool),
                        diagonal=1,
                    )
                    attn = attn.masked_fill(mask, float("-inf"))
                    probs = self.F.softmax(attn, dim=-1)
                    head_outs.append(probs @ v)
                concat = self.torch.cat(head_outs, dim=-1)
                x = x + concat @ layer["c_proj_w"].T + layer["c_proj_b"]

                ln2 = self._layer_norm(x, layer["ln2_w"], layer["ln2_b"])
                fc1 = ln2 @ layer["fc_w"].T + layer["fc_b"]
                gelu = self.F.gelu(fc1)
                x = x + gelu @ layer["proj_w"].T + layer["proj_b"]

            return self._logits(x)[0].cpu().numpy().astype(np.float32)

    def incremental_logits_trace(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        """Process tokens through explicit K/V caches and return each step's logits."""
        tids = [int(tok) for tok in token_ids]
        if not tids:
            return []
        pids = list(position_ids) if position_ids is not None else list(range(len(tids)))
        if len(pids) != len(tids):
            raise ValueError("position_ids length must match token_ids length")
        caches = self._empty_caches()
        return [
            self._decode_incremental_step(tok, pos, caches)
            for tok, pos in zip(tids, pids)
        ]

    def forward_incremental(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        trace = self.incremental_logits_trace(token_ids, position_ids=position_ids)
        if not trace:
            raise ValueError("token_ids must be non-empty")
        return trace[-1]

    def greedy_decode_trace(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
    ) -> FP32DecodeTrace:
        """Greedy decode while retaining the prefill-last and per-step logits."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        prompt = [int(tok) for tok in prompt_ids]
        if not prompt:
            raise ValueError("prompt_ids must be non-empty")

        caches = self._empty_caches()
        generated = list(prompt)
        logits = None
        for pos, tok in enumerate(prompt):
            logits = self._decode_incremental_step(tok, pos, caches)
        logits_trace = [logits]
        next_token = int(np.argmax(logits[:self.vocab_size]))

        for _ in range(max_new_tokens):
            generated.append(next_token)
            position = len(generated) - 1
            logits = self._decode_incremental_step(next_token, position, caches)
            logits_trace.append(logits)
            next_token = int(np.argmax(logits[:self.vocab_size]))

        return FP32DecodeTrace(generated=generated, logits=logits_trace)

    def _empty_caches(self):
        return [
            [
                {"k": [], "v": []}
                for _ in range(self.n_head)
            ]
            for _ in range(self.n_layer)
        ]

    def _decode_incremental_step(self, token_id: int, position_id: int, caches) -> np.ndarray:
        with self.torch.no_grad():
            tok = self.torch.tensor([int(token_id)], dtype=self.torch.long)
            pos = self.torch.tensor([int(position_id)], dtype=self.torch.long)
            x = self.wte[tok] + self.wpe[pos]

            for layer_idx, layer in enumerate(self.layers):
                ln1 = self._layer_norm(x, layer["ln1_w"], layer["ln1_b"])
                head_outs = []
                for head_idx, (q_w, k_w, v_w) in enumerate(layer["heads"]):
                    q = ln1 @ q_w.T
                    k = ln1 @ k_w.T
                    v = ln1 @ v_w.T
                    caches[layer_idx][head_idx]["k"].append(k[0].clone())
                    caches[layer_idx][head_idx]["v"].append(v[0].clone())
                    k_cache = self.torch.stack(caches[layer_idx][head_idx]["k"], dim=0)
                    v_cache = self.torch.stack(caches[layer_idx][head_idx]["v"], dim=0)
                    attn = (q @ k_cache.T) * self.attn_scale
                    probs = self.F.softmax(attn, dim=-1)
                    head_outs.append(probs @ v_cache)
                concat = self.torch.cat(head_outs, dim=-1)
                x = x + concat @ layer["c_proj_w"].T + layer["c_proj_b"]

                ln2 = self._layer_norm(x, layer["ln2_w"], layer["ln2_b"])
                fc1 = ln2 @ layer["fc_w"].T + layer["fc_b"]
                gelu = self.F.gelu(fc1)
                x = x + gelu @ layer["proj_w"].T + layer["proj_b"]

            return self._logits(x)[0].cpu().numpy().astype(np.float32)
