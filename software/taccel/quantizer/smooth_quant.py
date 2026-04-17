"""SmoothQuant preprocessing for LayerNorm -> Linear pairs."""
from __future__ import annotations

from typing import Dict, Optional, Set

import torch

from ..compiler.graph_extract import DEPTH


def compute_smooth_factors(
    model,
    sample_inputs: list,
    alpha: float = 0.5,
    targets: str = "both",
    blocks: Optional[Set[int]] = None,
) -> Dict[str, dict]:
    """Compute SmoothQuant factors for supported DeiT-tiny LayerNorm -> Linear pairs.

    Supported targets:
    - ``ln1_qkv``: shared vector for LN1 feeding query/key/value in each block
    - ``ln2_fc1``: vector for LN2 feeding FC1 in each block
    - ``both``: both of the above
    """
    if targets not in {"ln1_qkv", "ln2_fc1", "both"}:
        raise ValueError(f"Unsupported SmoothQuant target set: {targets}")
    if blocks is not None:
        invalid = sorted(block_idx for block_idx in blocks if not (0 <= block_idx < DEPTH))
        if invalid:
            raise ValueError(f"SmoothQuant block indices out of range: {invalid}")

    act_max = {"ln1_qkv": {}, "ln2_fc1": {}}
    handles = []

    def _make_hook(bucket_name: str, block_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            ch_max = tensor.detach().abs().amax(dim=(0, 1)).to(dtype=torch.float32, device="cpu")
            prev = act_max[bucket_name].get(block_idx)
            act_max[bucket_name][block_idx] = ch_max if prev is None else torch.maximum(prev, ch_max)
        return hook

    selected_blocks = range(DEPTH) if blocks is None else sorted(blocks)
    for block_idx in selected_blocks:
        if targets in {"ln1_qkv", "both"}:
            handles.append(
                model.vit.encoder.layer[block_idx].layernorm_before.register_forward_hook(
                    _make_hook("ln1_qkv", block_idx)
                )
            )
        if targets in {"ln2_fc1", "both"}:
            handles.append(
                model.vit.encoder.layer[block_idx].layernorm_after.register_forward_hook(
                    _make_hook("ln2_fc1", block_idx)
                )
            )

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            inp_dict = dict(inp.items()) if hasattr(inp, "items") else {"pixel_values": inp}
            model(**inp_dict)

    for handle in handles:
        handle.remove()

    factors = {}
    eps = 1e-8
    for block_idx in selected_blocks:
        prefix = f"vit.encoder.layer.{block_idx}"
        if targets in {"ln1_qkv", "both"}:
            act = act_max["ln1_qkv"][block_idx].clamp_min(eps)
            weight_cols = []
            for proj in ("query", "key", "value"):
                w = model.state_dict()[f"{prefix}.attention.attention.{proj}.weight"].detach().abs().to(dtype=torch.float32, device="cpu")
                weight_cols.append(w.amax(dim=0))
            w_max = torch.stack(weight_cols, dim=0).amax(dim=0).clamp_min(eps)
            smooth = (act.pow(alpha) / w_max.pow(1.0 - alpha)).clamp_min(eps)
            factors[f"block{block_idx}_ln1_qkv"] = {
                "ln_weight": f"{prefix}.layernorm_before.weight",
                "ln_bias": f"{prefix}.layernorm_before.bias",
                "linear_weights": [
                    f"{prefix}.attention.attention.query.weight",
                    f"{prefix}.attention.attention.key.weight",
                    f"{prefix}.attention.attention.value.weight",
                ],
                "smooth": smooth,
            }

        if targets in {"ln2_fc1", "both"}:
            act = act_max["ln2_fc1"][block_idx].clamp_min(eps)
            w = model.state_dict()[f"{prefix}.intermediate.dense.weight"].detach().abs().to(dtype=torch.float32, device="cpu")
            w_max = w.amax(dim=0).clamp_min(eps)
            smooth = (act.pow(alpha) / w_max.pow(1.0 - alpha)).clamp_min(eps)
            factors[f"block{block_idx}_ln2_fc1"] = {
                "ln_weight": f"{prefix}.layernorm_after.weight",
                "ln_bias": f"{prefix}.layernorm_after.bias",
                "linear_weights": [f"{prefix}.intermediate.dense.weight"],
                "smooth": smooth,
            }

    return factors


def apply_smooth_quant(state_dict: dict, smooth_factors: Dict[str, dict]) -> dict:
    """Apply SmoothQuant factors in-place to a cloned state dict."""
    for spec in smooth_factors.values():
        smooth = spec["smooth"].to(dtype=torch.float32, device="cpu")
        ln_weight_name = spec["ln_weight"]
        ln_bias_name = spec["ln_bias"]

        ln_weight = state_dict[ln_weight_name].detach().clone().to(dtype=torch.float32, device="cpu")
        ln_bias = state_dict[ln_bias_name].detach().clone().to(dtype=torch.float32, device="cpu")
        state_dict[ln_weight_name] = (ln_weight / smooth).to(dtype=state_dict[ln_weight_name].dtype)
        state_dict[ln_bias_name] = (ln_bias / smooth).to(dtype=state_dict[ln_bias_name].dtype)

        for weight_name in spec["linear_weights"]:
            weight = state_dict[weight_name].detach().clone().to(dtype=torch.float32, device="cpu")
            state_dict[weight_name] = (weight * smooth.unsqueeze(0)).to(dtype=state_dict[weight_name].dtype)

    return state_dict
