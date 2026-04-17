"""Calibration-time analytical bias correction for quantized linear layers."""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from .calibrate import collect_layer_inputs
from .quantize import dequantize_tensor, quantize_tensor


LATE_BLOCKS = (9, 10, 11)


def _weight_name_sort_key(weight_name: str):
    if weight_name == "classifier.weight":
        return (-1, "", weight_name)
    match = re.fullmatch(r"vit\.encoder\.layer\.(\d+)\.(.+)\.weight", weight_name)
    if match is None:
        return (10_000, "", weight_name)
    return (int(match.group(1)), match.group(2), weight_name)


def resolve_bias_correction_targets(state_dict: dict, selector_text: str) -> List[str]:
    """Resolve a selector string into concrete weight names."""
    tokens = [token.strip() for token in (selector_text or "").split(",") if token.strip()]
    if not tokens:
        tokens = ["classifier"]

    selected = set()
    all_linear = {
        name
        for name, tensor in state_dict.items()
        if name.endswith(".weight") and hasattr(tensor, "numpy") and tensor.numpy().ndim >= 2
    }

    for token in tokens:
        if token == "classifier":
            selected.add("classifier.weight")
        elif token == "late_out_proj":
            selected.update(
                f"vit.encoder.layer.{block_idx}.attention.output.dense.weight"
                for block_idx in LATE_BLOCKS
            )
        elif token == "late_fc2":
            selected.update(
                f"vit.encoder.layer.{block_idx}.output.dense.weight"
                for block_idx in LATE_BLOCKS
            )
        elif token == "all":
            selected.update(all_linear)
        elif token.endswith(".weight"):
            selected.add(token)
        else:
            raise ValueError(f"Unknown bias-correction selector '{token}'")

    missing = sorted(name for name in selected if name not in state_dict)
    if missing:
        raise ValueError(f"Bias-correction targets not found in state_dict: {missing}")
    return sorted(selected, key=_weight_name_sort_key)


def weight_name_to_input_scale_key(weight_name: str) -> str:
    """Map a state_dict weight name to the activation-scale key used before the linear."""
    if weight_name == "classifier.weight":
        return "final_ln"

    match = re.fullmatch(r"vit\.encoder\.layer\.(\d+)\.(.+)\.weight", weight_name)
    if match is None:
        raise ValueError(f"Unsupported bias-correction weight name '{weight_name}'")

    block_idx = int(match.group(1))
    suffix = match.group(2)
    if suffix == "attention.output.dense":
        return f"block{block_idx}_concat"
    if suffix == "output.dense":
        return f"block{block_idx}_gelu"
    if suffix == "intermediate.dense":
        return f"block{block_idx}_ln2"
    if suffix.startswith("attention.attention."):
        return f"block{block_idx}_ln1"
    raise ValueError(f"Unsupported bias-correction weight suffix '{suffix}'")


def _module_name_from_weight_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected weight name, got '{weight_name}'")
    return weight_name[:-7]

def _uses_per_tensor_quantization(weight_name: str) -> bool:
    if weight_name == "classifier.weight":
        return True
    return bool(
        re.fullmatch(
            r"vit\.encoder\.layer\.\d+\.(attention\.output\.dense|intermediate\.dense|output\.dense)\.weight",
            weight_name,
        )
    )


def _quantized_weight_for_bias_correction(weight_name: str, weight_fp32: np.ndarray, quant_weights: dict):
    if _uses_per_tensor_quantization(weight_name):
        return quantize_tensor(weight_fp32, per_channel=False)
    if weight_name not in quant_weights:
        raise KeyError(f"Missing quantized weight '{weight_name}' for bias correction")
    q_weight, q_scales = quant_weights[weight_name]
    if q_scales is None:
        raise KeyError(f"Quantized weight '{weight_name}' has no scales for bias correction")
    return q_weight.astype(np.int8), q_scales.astype(np.float16)


def compute_bias_corrections(
    model,
    state_dict,
    quant_weights,
    calibration_scales,
    sample_inputs,
    target_weight_names,
) -> Dict[str, np.ndarray]:
    """Compute per-bias FP32 correction vectors from calibration activations."""
    module_names = [_module_name_from_weight_name(weight_name) for weight_name in target_weight_names]
    inputs_by_module = collect_layer_inputs(model, sample_inputs, module_names)
    corrections: Dict[str, np.ndarray] = {}

    for weight_name, module_name in zip(target_weight_names, module_names):
        bias_name = weight_name.replace(".weight", ".bias")
        if bias_name not in state_dict:
            continue

        weight_fp32 = state_dict[weight_name].detach().cpu().numpy().astype(np.float32)
        if weight_fp32.ndim > 2:
            weight_fp32 = weight_fp32.reshape(weight_fp32.shape[0], -1)
        q_weight, q_scales = _quantized_weight_for_bias_correction(weight_name, weight_fp32, quant_weights)
        dq_weight = dequantize_tensor(q_weight, q_scales)

        act_scale_key = weight_name_to_input_scale_key(weight_name)
        act_scale = max(float(calibration_scales.get(act_scale_key, 6.0 / 127.0)), 1e-12)

        err_sum = np.zeros(weight_fp32.shape[0], dtype=np.float64)
        err_count = 0
        for x_np in inputs_by_module.get(module_name, []):
            x_2d = x_np.reshape(-1, x_np.shape[-1]).astype(np.float32)
            x_q = np.clip(np.round(x_2d / act_scale), -128, 127).astype(np.int8)
            x_dq = x_q.astype(np.float32) * act_scale
            y_fp32 = x_2d @ weight_fp32.T
            y_qdq = x_dq @ dq_weight.T
            err_sum += np.sum(y_fp32 - y_qdq, axis=0, dtype=np.float64)
            err_count += x_2d.shape[0]

        if err_count == 0:
            corrections[bias_name] = np.zeros(weight_fp32.shape[0], dtype=np.float32)
        else:
            corrections[bias_name] = (err_sum / float(err_count)).astype(np.float32)

    return corrections
