"""True FP32 nanoGPT reference used for rank-based e2e diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class FP32DecodeTrace:
    generated: List[int]
    logits: List[np.ndarray]


def _as_f32_tensor(torch, tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().to(dtype=torch.float32)
    return torch.tensor(tensor, dtype=torch.float32)


def _optional_f32_tensor(torch, state_dict: dict, name: str, size: int):
    if name in state_dict:
        return _as_f32_tensor(torch, state_dict[name])
    return torch.zeros(int(size), dtype=torch.float32)


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
        self.activation_function = str(
            model_args.get(
                "activation_function",
                "gelu_new" if bool(model_args.get("split_qkv_bias", False)) else "gelu",
            )
        )

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
                    _optional_f32_tensor(
                        torch,
                        sd,
                        f"transformer.h.{layer_idx}.attn.c_attn.bias_h{head_idx}_query",
                        self.d_head,
                    ),
                    _optional_f32_tensor(
                        torch,
                        sd,
                        f"transformer.h.{layer_idx}.attn.c_attn.bias_h{head_idx}_key",
                        self.d_head,
                    ),
                    _optional_f32_tensor(
                        torch,
                        sd,
                        f"transformer.h.{layer_idx}.attn.c_attn.bias_h{head_idx}_value",
                        self.d_head,
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

    def _gelu(self, x):
        if self.activation_function in {"gelu_new", "gelu_fast"}:
            return self.F.gelu(x, approximate="tanh")
        if self.activation_function in {"gelu", "gelu_pytorch_tanh"}:
            approximate = "tanh" if self.activation_function == "gelu_pytorch_tanh" else "none"
            return self.F.gelu(x, approximate=approximate)
        raise ValueError(f"Unsupported activation_function={self.activation_function!r}")

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
                for q_w, k_w, v_w, q_b, k_b, v_b in layer["heads"]:
                    q = ln1 @ q_w.T + q_b
                    k = ln1 @ k_w.T + k_b
                    v = ln1 @ v_w.T + v_b
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
                gelu = self._gelu(fc1)
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

    def incremental_node_trace(
        self,
        token_ids: Sequence[int],
        position_ids: Optional[Sequence[int]] = None,
    ) -> List[dict[str, dict[str, np.ndarray]]]:
        """Process tokens with explicit K/V caches and return per-node FP32 traces."""
        tids = [int(tok) for tok in token_ids]
        if not tids:
            return []
        pids = list(position_ids) if position_ids is not None else list(range(len(tids)))
        if len(pids) != len(tids):
            raise ValueError("position_ids length must match token_ids length")
        caches = self._empty_caches()
        traces = []
        for tok, pos in zip(tids, pids):
            step_trace: dict[str, dict[str, np.ndarray]] = {}
            self._decode_incremental_step(tok, pos, caches, trace=step_trace)
            traces.append(step_trace)
        return traces

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

    def _decode_incremental_step(self, token_id: int, position_id: int, caches,
                                 trace: Optional[dict[str, dict[str, np.ndarray]]] = None) -> np.ndarray:
        with self.torch.no_grad():
            def record(name: str, value) -> None:
                if trace is None:
                    return
                trace[name] = {
                    "value": value.detach().cpu().numpy().astype(np.float32).copy()
                    if hasattr(value, "detach")
                    else np.asarray(value, dtype=np.float32).copy()
                }

            tok = self.torch.tensor([int(token_id)], dtype=self.torch.long)
            pos = self.torch.tensor([int(position_id)], dtype=self.torch.long)
            x = self.wte[tok] + self.wpe[pos]
            record("tok_pos_add", x)

            for layer_idx, layer in enumerate(self.layers):
                ln1 = self._layer_norm(x, layer["ln1_w"], layer["ln1_b"])
                record(f"block{layer_idx}_ln1", ln1)
                head_outs = []
                for head_idx, (q_w, k_w, v_w, q_b, k_b, v_b) in enumerate(layer["heads"]):
                    q = ln1 @ q_w.T + q_b
                    k = ln1 @ k_w.T + k_b
                    v = ln1 @ v_w.T + v_b
                    record(f"block{layer_idx}_head{head_idx}_query", q)
                    record(f"block{layer_idx}_head{head_idx}_key", k)
                    record(f"block{layer_idx}_head{head_idx}_value", v)
                    caches[layer_idx][head_idx]["k"].append(k[0].clone())
                    caches[layer_idx][head_idx]["v"].append(v[0].clone())
                    k_cache = self.torch.stack(caches[layer_idx][head_idx]["k"], dim=0)
                    v_cache = self.torch.stack(caches[layer_idx][head_idx]["v"], dim=0)
                    attn = (q @ k_cache.T) * self.attn_scale
                    probs = self.F.softmax(attn, dim=-1)
                    record(f"block{layer_idx}_head{head_idx}_softmax", probs)
                    head_out = probs @ v_cache
                    record(f"block{layer_idx}_head{head_idx}_attn_v", head_out)
                    head_outs.append(head_out)
                concat = self.torch.cat(head_outs, dim=-1)
                record(f"block{layer_idx}_concat", concat)
                out_proj = concat @ layer["c_proj_w"].T + layer["c_proj_b"]
                record(f"block{layer_idx}_out_proj", out_proj)
                x = x + out_proj
                record(f"block{layer_idx}_residual1", x)

                ln2 = self._layer_norm(x, layer["ln2_w"], layer["ln2_b"])
                record(f"block{layer_idx}_ln2", ln2)
                fc1 = ln2 @ layer["fc_w"].T + layer["fc_b"]
                record(f"block{layer_idx}_fc1", fc1)
                gelu = self._gelu(fc1)
                record(f"block{layer_idx}_gelu", gelu)
                fc2 = gelu @ layer["proj_w"].T + layer["proj_b"]
                record(f"block{layer_idx}_fc2", fc2)
                x = x + fc2
                record(f"block{layer_idx}_residual2", x)

            ln_f = self._layer_norm(x, self.ln_f_w, self.ln_f_b)
            record("ln_f", ln_f)
            logits = ln_f[-1:] @ self.lm_head_w.T
            record("lm_head", logits)
            return logits[0].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# W8A32 helpers — INT8 weight QDQ + FP32 activations everywhere.
#
# These were originally housed in `software/tools/diagnose_weight_only_qdq_ceiling.py`
# (commit `37929dd`, the diagnostic that established the 53.42 PPL ceiling on
# this codebase). They live here so the eval pipeline and the diagnostic share
# one source of truth and produce bit-identical logits.
# ---------------------------------------------------------------------------


def _weight_only_qdq_per_channel(arr: np.ndarray) -> np.ndarray:
    """Symmetric per-row INT8 quantise → dequantise back to FP32.

    One scale per output channel (i.e. per row). No padding, no mean-scale
    averaging — this is the path the diagnostic proved at 53.42 PPL. The
    codebase's mean-scale dequant (`_linear_components`) is the opposite
    extreme; it is catastrophic in pure FP32 (~1.3e+19 PPL) without the
    activation INT8 clipping that absorbs it in the W8A8 pipeline.
    """
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = np.maximum(np.max(np.abs(arr), axis=1, keepdims=True), 1e-10)
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return (q.astype(np.float32) * scale).astype(np.float32)


def _weight_only_qdq_mean_scale(arr: np.ndarray) -> np.ndarray:
    """Codebase's mean-scale dequant approximation, FP32-out.

    Provided only for parity with the diagnostic's `weight_mode="mean_scale"`
    branch. Not used by the W8A32 production preset — kept here so the
    diagnostic can still A/B both modes from a single helper.
    """
    # Local import to avoid pulling NanoGPTFQReference's module-load weight
    # cache into the FP32-only path.
    from .fake_quant_reference import _linear_components

    arr = np.asarray(arr, dtype=np.float32)
    return _linear_components(arr)[2].astype(np.float32)


def _weight_only_qdq_per_tensor_embedding(arr: np.ndarray) -> np.ndarray:
    """Per-tensor INT8 QDQ of an embedding table.

    Mirrors `_fq_embedding` in `fake_quant_reference.py` (the production INT8
    embedding path is per-tensor; per-channel doesn't apply because the table
    is read row-wise as activation input, not used as a matmul weight).
    """
    from .fake_quant_reference import _fq_embedding

    arr = np.asarray(arr, dtype=np.float32)
    return _fq_embedding(arr).astype(np.float32)


def _torch_qdq_per_channel(torch, t):
    arr = t.detach().cpu().to(dtype=torch.float32).numpy()
    return torch.from_numpy(_weight_only_qdq_per_channel(arr))


def _torch_qdq_mean_scale(torch, t):
    arr = t.detach().cpu().to(dtype=torch.float32).numpy()
    return torch.from_numpy(_weight_only_qdq_mean_scale(arr))


def _torch_qdq_per_tensor_embedding(torch, t):
    arr = t.detach().cpu().to(dtype=torch.float32).numpy()
    return torch.from_numpy(_weight_only_qdq_per_tensor_embedding(arr))


def build_weight_only_int8_reference(
    payload: Mapping[str, object],
    *,
    weight_mode: str = "per_channel",
) -> NanoGPTFP32Reference:
    """Build a `NanoGPTFP32Reference` whose linear weights are INT8 QDQ.

    This is the W8A32 reference path: INT8 weight storage (per-channel
    scales by default, padded to 16 wide for the matmul tile), FP32
    activations and FP32 inter-layer storage everywhere else. The
    diagnostic at `software/tools/diagnose_weight_only_qdq_ceiling.py`
    proved this achieves 53.42 PPL on `gpt2_converted_nanogpt.pt` at
    257-tok / 256-ctx, vs an FP32 ceiling of 53.69 PPL.

    Embeddings (`wte`, `wpe`) use the codebase's per-tensor INT8 path to
    match the deployed bundle's embedding scheme. Linear weights
    (`lm_head`, per-head Q/K/V, `c_proj`, `fc_w`, `proj_w`) use per-row
    symmetric INT8 quant with no mean-scale averaging.

    Args:
        payload: Loaded checkpoint dict with keys `state_dict` and
            `model_args`. Same shape as `evaluate_gpt2_perplexity` consumes.
        weight_mode: Either ``"per_channel"`` (W8A32 production path) or
            ``"mean_scale"`` (diagnostic-only; matches the catastrophic
            mean-scale approximation isolated in `_linear_components`).

    Returns:
        A `NanoGPTFP32Reference` ready for `.incremental_logits_trace(...)`.
    """
    import torch  # local import keeps the FP32 module CPU-load cheap

    if weight_mode not in ("per_channel", "mean_scale"):
        raise ValueError(
            f"weight_mode must be 'per_channel' or 'mean_scale', got {weight_mode!r}"
        )
    linear_qdq = (
        _torch_qdq_per_channel if weight_mode == "per_channel" else _torch_qdq_mean_scale
    )

    ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    ref.wte = _torch_qdq_per_tensor_embedding(torch, ref.wte)
    ref.wpe = _torch_qdq_per_tensor_embedding(torch, ref.wpe)
    ref.lm_head_w = linear_qdq(torch, ref.lm_head_w)

    for layer in ref.layers:
        new_heads = []
        for (q_w, k_w, v_w, q_b, k_b, v_b) in layer["heads"]:
            new_heads.append((
                linear_qdq(torch, q_w),
                linear_qdq(torch, k_w),
                linear_qdq(torch, v_w),
                q_b,  # biases stay FP32 — W8A32 only quantises weights
                k_b,
                v_b,
            ))
        layer["heads"] = new_heads
        layer["c_proj_w"] = linear_qdq(torch, layer["c_proj_w"])
        layer["fc_w"] = linear_qdq(torch, layer["fc_w"])
        layer["proj_w"] = linear_qdq(torch, layer["proj_w"])
        # LN weights/biases stay FP32 (small tensors, not the bottleneck).
    return ref
