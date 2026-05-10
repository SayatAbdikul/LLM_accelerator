"""AWQ (Activation-aware Weight Quantization) for the GPT-2 PTQ pipeline.

AWQ scales weight INPUT channels by per-channel activation-magnitude factors
and folds the inverse scale into the previous LayerNorm's `gamma`/`bias`.
Mathematically the matmul output is unchanged:

    (x[i] / s[i]) * (W[:, i] * s[i])  ==  x[i] * W[:, i]

But after per-channel INT8 weight quantization, the scaled weight has more
uniform per-channel scales (because channels reading large activations now
have larger weight values too — proportional). This reduces the codebase's
mean-scale dequant approximation error: the difference between the per-
channel scale and the mean scale shrinks, so the integer matmul's output
better approximates the FP32 reference.

Diagnostic at `software/tools/diagnose_weight_only_qdq_ceiling.py` showed
that per-channel weights are essentially free for GPT-2 PPL (53.42 vs 53.69
FP32 ceiling), but the codebase's mean-scale dequant gives 1.3e+19 PPL in
pure FP32 — only viable in production because activation INT8 clipping
absorbs the per-channel error. AWQ directly attacks this by reducing the
spread of per-channel scales.

AWQ is applied only at LayerNorm-fed matmul modules, where the inverse
scale folds cleanly into the LN's gamma and bias:

  - ln1 → c_attn (Q/K/V projections per head)
  - ln2 → c_fc  (MLP fc1 projection)
  - ln_f → lm_head

c_proj of attention (out_proj) and c_proj of MLP (fc2) cannot be AWQ'd
without changing the residual stream; they are out of scope for v1.

References:
  - Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
    Compression and Acceleration", MLSys 2024.
  - The mathematical fold is the same as SmoothQuant's, but AWQ uses a
    different scale formula focused on weight-scale uniformity rather
    than activation-distribution flattening.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


__all__ = [
    "compute_per_channel_activation_magnitudes",
    "compute_awq_scales",
    "apply_awq_to_state_dict",
]


def _to_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _store(state_dict, key: str, value: np.ndarray) -> None:
    """Write `value` into `state_dict[key]`, preserving the original dtype."""
    original = state_dict[key]
    if hasattr(original, "detach"):
        new = torch.from_numpy(value.astype(np.float32))
        state_dict[key] = new.to(dtype=original.dtype)
    else:
        state_dict[key] = value.astype(np.asarray(original).dtype)


def compute_per_channel_activation_magnitudes(
    state_dict: dict,
    model_args: dict,
    calibration_seqs: Sequence[Sequence[int]],
) -> Dict[str, np.ndarray]:
    """Run FP32 forward over `calibration_seqs` and return per-input-channel
    max-abs activation magnitudes for each AWQ fold target.

    Returns a dict keyed by the LN node name (so callers can index directly):
      - `block{L}_ln1`  →  shape (d_model,)
      - `block{L}_ln2`  →  shape (d_model,)
      - `ln_f`          →  shape (d_model,)
    """
    from ..runtime.calibration import _fp32_forward

    n_layer = int(model_args["n_layer"])
    accum: Dict[str, np.ndarray] = {}
    for tids in calibration_seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for L in range(n_layer):
            for kind in ("ln1", "ln2"):
                key = f"block{L}_{kind}"
                if key in node_outputs:
                    arr = np.asarray(node_outputs[key], dtype=np.float32)
                    # arr shape is (seq_len, d_model). Per-channel = per-input-feature.
                    max_abs = np.max(np.abs(arr), axis=0)
                    if key in accum:
                        accum[key] = np.maximum(accum[key], max_abs)
                    else:
                        accum[key] = max_abs
        if "ln_f" in node_outputs:
            arr = np.asarray(node_outputs["ln_f"], dtype=np.float32)
            max_abs = np.max(np.abs(arr), axis=0)
            if "ln_f" in accum:
                accum["ln_f"] = np.maximum(accum["ln_f"], max_abs)
            else:
                accum["ln_f"] = max_abs
    return accum


def compute_awq_scales(
    activation_magnitudes: np.ndarray,
    weight_max_abs: np.ndarray,
    *,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute per-input-channel AWQ scales.

    Standard AWQ formulation:
        s[i] = (act_mag[i] ** alpha) / (weight_mag[i] ** (1 - alpha))

    Then normalize so the GEOMETRIC MEAN of s is 1, keeping overall weight
    magnitude bounded. alpha=0.5 is the canonical setting (geometric mean of
    activation and inverse-weight magnitudes).

    Args:
        activation_magnitudes: shape (d_in,) per-input-channel max-abs of LN
            output (the matmul input).
        weight_max_abs: shape (d_in,) per-input-channel max-abs of the weight
            (max over output channels for each input channel).
        alpha: trade-off in [0, 1]; 0 = scale by 1/weight, 1 = scale by act,
            0.5 = geometric mean.

    Returns:
        np.ndarray of shape (d_in,), AWQ scale per input channel.
    """
    act = np.maximum(activation_magnitudes.astype(np.float32), eps)
    w = np.maximum(weight_max_abs.astype(np.float32), eps)
    s = (act ** alpha) / (w ** (1.0 - alpha))
    # Normalize so the geometric mean of s is 1 (keeps total weight magnitude
    # bounded; otherwise scales could blow up uniformly).
    log_s = np.log(s)
    s = s / np.exp(np.mean(log_s))
    return s


def _weight_max_abs_per_input_channel(state_dict: dict, weight_keys: Sequence[str]) -> np.ndarray:
    """Concatenate weights by output dim and return max-abs per input channel.

    Each weight has shape (out, in). We stack along output and take max-abs
    along the output axis, returning shape (in,).
    """
    rows = []
    for key in weight_keys:
        arr = _to_f32(state_dict[key])
        if arr.ndim != 2:
            raise ValueError(f"expected 2-D weight for {key}, got shape {arr.shape}")
        rows.append(np.abs(arr))
    stacked = np.concatenate(rows, axis=0)  # (sum_out, in)
    return np.max(stacked, axis=0).astype(np.float32)


def _scale_weight_input_channels(
    state_dict: dict, weight_keys: Sequence[str], scales: np.ndarray
) -> None:
    """Multiply each weight's input channels by `scales[i]` (column scaling)."""
    s_row = scales.astype(np.float32).reshape(1, -1)
    for key in weight_keys:
        arr = _to_f32(state_dict[key])
        new = arr * s_row[:, : arr.shape[1]]
        _store(state_dict, key, new)


def _scale_ln_gamma_beta(
    state_dict: dict, gamma_key: str, beta_key: str, scales: np.ndarray
) -> None:
    """Divide LN gamma and beta by `scales[i]` element-wise.

    After this fold, LN output along channel i is multiplied by `1/s[i]`
    relative to the unfolded LN, so the matmul (whose input cols were
    multiplied by `s[i]`) produces the same FP32 output.
    """
    s = scales.astype(np.float32)
    gamma = _to_f32(state_dict[gamma_key])
    if gamma.shape != s.shape:
        raise ValueError(f"gamma shape {gamma.shape} mismatched scales {s.shape}")
    _store(state_dict, gamma_key, gamma / s)
    if beta_key in state_dict:
        beta = _to_f32(state_dict[beta_key])
        _store(state_dict, beta_key, beta / s)


def apply_awq_to_state_dict(
    state_dict: dict,
    model_args: dict,
    activation_magnitudes: Dict[str, np.ndarray],
    *,
    alpha: float = 0.5,
    target_modules: Sequence[str] = ("c_attn", "c_fc", "lm_head"),
) -> List[str]:
    """Apply AWQ in place. Returns the list of mutated state_dict keys.

    For each target module:
      - "c_attn": for each block L and each head H, scale the input cols of
        c_attn.weight_h{H}_{query,key,value} by the AWQ scales derived from
        ln1's per-channel activation magnitudes and the joint Q/K/V weight
        max-abs. Fold inverse scales into ln_1.weight and ln_1.bias.
      - "c_fc": scale input cols of mlp.c_fc.weight by AWQ scales from ln2.
        Fold inverse into ln_2.
      - "lm_head": scale input cols of lm_head.weight by AWQ scales from ln_f.
        Fold inverse into transformer.ln_f.

    All scales are computed per-block (or globally for lm_head) using the
    per-channel statistics already collected by
    `compute_per_channel_activation_magnitudes`.
    """
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    mutated: List[str] = []

    if "c_attn" in target_modules:
        for L in range(n_layer):
            ln1_key = f"block{L}_ln1"
            if ln1_key not in activation_magnitudes:
                continue
            act_mags = activation_magnitudes[ln1_key]

            # Collect ALL Q/K/V weight keys for this block (per head, 3 each).
            qkv_keys = []
            for H in range(n_head):
                for kind in ("query", "key", "value"):
                    qkv_keys.append(f"transformer.h.{L}.attn.c_attn.weight_h{H}_{kind}")
            # Filter to those present (defensive).
            qkv_keys = [k for k in qkv_keys if k in state_dict]
            if not qkv_keys:
                continue
            w_mag = _weight_max_abs_per_input_channel(state_dict, qkv_keys)
            scales = compute_awq_scales(act_mags, w_mag, alpha=alpha)

            _scale_weight_input_channels(state_dict, qkv_keys, scales)
            _scale_ln_gamma_beta(
                state_dict,
                f"transformer.h.{L}.ln_1.weight",
                f"transformer.h.{L}.ln_1.bias",
                scales,
            )
            mutated.extend(qkv_keys)
            mutated.append(f"transformer.h.{L}.ln_1.weight")
            mutated.append(f"transformer.h.{L}.ln_1.bias")

    if "c_fc" in target_modules:
        for L in range(n_layer):
            ln2_key = f"block{L}_ln2"
            if ln2_key not in activation_magnitudes:
                continue
            act_mags = activation_magnitudes[ln2_key]
            fc_key = f"transformer.h.{L}.mlp.c_fc.weight"
            if fc_key not in state_dict:
                continue
            w_mag = _weight_max_abs_per_input_channel(state_dict, [fc_key])
            scales = compute_awq_scales(act_mags, w_mag, alpha=alpha)

            _scale_weight_input_channels(state_dict, [fc_key], scales)
            _scale_ln_gamma_beta(
                state_dict,
                f"transformer.h.{L}.ln_2.weight",
                f"transformer.h.{L}.ln_2.bias",
                scales,
            )
            mutated.append(fc_key)
            mutated.append(f"transformer.h.{L}.ln_2.weight")
            mutated.append(f"transformer.h.{L}.ln_2.bias")

    if "lm_head" in target_modules:
        if "ln_f" in activation_magnitudes:
            act_mags = activation_magnitudes["ln_f"]
            lm_head_key = "lm_head.weight"
            if lm_head_key in state_dict:
                w_mag = _weight_max_abs_per_input_channel(state_dict, [lm_head_key])
                scales = compute_awq_scales(act_mags, w_mag, alpha=alpha)
                _scale_weight_input_channels(state_dict, [lm_head_key], scales)
                _scale_ln_gamma_beta(
                    state_dict,
                    "transformer.ln_f.weight",
                    "transformer.ln_f.bias",
                    scales,
                )
                mutated.append(lm_head_key)
                mutated.append("transformer.ln_f.weight")
                mutated.append("transformer.ln_f.bias")

    return mutated
