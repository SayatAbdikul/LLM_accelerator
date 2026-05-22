"""Adapter helpers: QuaRot rotation, AWQ, bias correction.

Split from `runtime/calibration.py` as part of R2. Imports mirror the
original module; public symbols are re-exported via `__init__.py`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Sequence

import numpy as np

from ..fake_quant_reference import (
    NanoGPTFQReference,
    _arch_scale,
    _bias_i32,
    _fp32_forward,
    _fp32_to_int8,
    _int8_saturating_add,
    _requant_accum_pc_int8,
    _to_f32,
)
from ..stage5_ptq import (
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)
from ...quantizer.hessian_guided import find_hessian_gelu_scale
from ...quantizer.quantize import quantize_tensor


from .scales import (
    _DEFAULT_SCALES,
    _SFU_DEFAULT_SCALES,
    FC2_AWARE_GELU_MULTIPLIERS,
    OUTPUT_AWARE_GELU_MULTIPLIERS,
    OUTPUT_AWARE_MLP_MULTIPLIERS,
    OUTPUT_AWARE_SEARCH_N_SEQS_MAX,
    OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX,
    OUTPUT_AWARE_SEARCH_WORKERS_DEFAULT,
    build_calibration_scales,
    build_calibration_scales_from_token_ids,
    build_calibration_seqs,
    build_calibration_seqs_from_token_ids,
)


def _bias_correction_input_node(weight_name: str) -> str:
    """Map a state_dict weight name to the activation node feeding the layer.

    Used by apply_bias_correction_from_token_ids to look up the FP32 activations
    captured by _fp32_forward for each target weight.
    """
    if weight_name.endswith(".mlp.c_fc.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_ln2"
    if weight_name.endswith(".mlp.c_proj.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_gelu"
    if weight_name.endswith(".attn.c_proj.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_concat"
    raise ValueError(f"unsupported bias-correction weight name: {weight_name}")

def apply_quarot_rotation_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    *,
    seed: int = 0xCAFE,
    kind: str = "random_orthogonal",
) -> tuple[List[str], dict]:
    """Apply QuaRot Phase 1 to ``payload['state_dict']`` in place.

    Pipeline (executed in order):
      1. ``fold_layernorm_for_quarot(state_dict, model_args)`` — γ-fold + β-rescale.
      2. Build rotation ``R`` per ``kind`` (and ``seed``).
      3. ``rotate_residual_stream_state_dict(state_dict, model_args, R)`` —
         pre-rotates wte, wpe, every block's c_attn/c_fc input cols and
         c_proj output rows + biases, lm_head input cols, and every LN.bias.
      4. Clear the cached weight components in NanoGPTFQReference (see
         ``fake_quant_reference._WEIGHT_COMPONENT_CACHE``); the cache key uses
         ``id(state_dict)`` and would otherwise return stale unrotated
         components for any subsequent ``NanoGPTFQReference`` constructed
         from the same dict.

    The ``token_ids`` parameter is accepted for API symmetry with
    :func:`apply_bias_correction_from_token_ids` and to keep the signature
    stable when SpinQuant-style data-dependent rotations are added later.
    For ``kind="random_orthogonal"`` it is unused.

    Args:
        payload: model payload with ``state_dict`` and ``model_args``.
            ``state_dict`` is mutated in place.
        token_ids: tokenized calibration text (currently unused).
        seed: PRNG seed for ``build_random_orthogonal``.
        kind: rotation kind. Currently only ``"random_orthogonal"`` is
            supported; ``"block_hadamard_768"`` is reserved for a future PR.

    Returns:
        ``(modified_keys, diagnostics)`` where:
            * ``modified_keys``: list of state_dict keys mutated, in order.
            * ``diagnostics``: dict with ``kind``, ``seed``, ``d_model``,
              ``n_keys_folded``, ``n_keys_rotated``,
              ``rotation_orthogonality_error``.

    Raises:
        ValueError if ``kind`` is unsupported.
    """
    # Lazy imports to avoid circulars at module-load time.
    from taccel.quantizer.ln_fold import fold_layernorm_for_quarot
    from taccel.quantizer.rotation import (
        build_random_orthogonal,
        rotate_residual_stream_state_dict,
    )
    from taccel.runtime.fake_quant_reference import clear_weight_component_cache

    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    d_model = int(model_args["n_embd"])

    if kind == "random_orthogonal":
        R = build_random_orthogonal(d_model, seed=int(seed))
    else:
        raise ValueError(
            f"apply_quarot_rotation_from_token_ids: unsupported kind={kind!r}; "
            f"supported: 'random_orthogonal'"
        )

    folded_keys = fold_layernorm_for_quarot(state_dict, model_args)
    rotated_keys = rotate_residual_stream_state_dict(state_dict, model_args, R)

    # Cache invalidation MUST happen here — see docstring.
    clear_weight_component_cache()

    eye = np.eye(d_model, dtype=np.float32)
    orthogonality_error = float(np.abs(R @ R.T - eye).max())
    diagnostics = {
        "kind": kind,
        "seed": int(seed),
        "d_model": d_model,
        "n_keys_folded": len(folded_keys),
        "n_keys_rotated": len(rotated_keys),
        "rotation_orthogonality_error": orthogonality_error,
    }
    return folded_keys + rotated_keys, diagnostics

def apply_awq_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    *,
    n_seqs: int = 8,
    seq_len: int = 64,
    alpha: float = 0.5,
    target_modules: Sequence[str] = ("c_attn", "c_fc", "lm_head"),
) -> tuple[List[str], dict]:
    """Apply AWQ (Activation-aware Weight Quantization) to
    ``payload['state_dict']`` in place.

    Pipeline:
      1. Build calibration sequences from ``token_ids`` (n_seqs × seq_len).
      2. Run FP32 forward to collect per-input-channel max-abs activation
         magnitudes for ``block{L}_ln1``, ``block{L}_ln2``, and ``ln_f``.
      3. For each AWQ fold target (c_attn / c_fc / lm_head), compute per-
         input-channel scales and mutate the matmul weights + LN gamma/bias.
      4. Clear the NanoGPTFQReference weight cache.

    Args:
        payload: model payload with ``state_dict`` and ``model_args``;
            ``state_dict`` is mutated in place.
        token_ids: tokenized calibration text.
        n_seqs: number of calibration sequences.
        seq_len: tokens per sequence.
        alpha: AWQ scale exponent (0=inverse-weight, 1=activation-magnitude,
            0.5=geometric mean — canonical setting).
        target_modules: which fold targets to apply AWQ to. ``c_attn`` folds
            inverse scale into ln_1, ``c_fc`` into ln_2, ``lm_head`` into ln_f.

    Returns:
        ``(modified_keys, diagnostics)``: list of mutated state_dict keys and
        a diagnostics dict (``alpha``, ``target_modules``, ``n_keys_mutated``,
        ``n_act_stats_collected``).
    """
    from taccel.quantizer.awq import (
        apply_awq_to_state_dict,
        compute_per_channel_activation_magnitudes,
    )
    from taccel.runtime.fake_quant_reference import clear_weight_component_cache

    state_dict = payload["state_dict"]
    model_args = payload["model_args"]

    seqs = build_calibration_seqs_from_token_ids(token_ids, n_seqs=n_seqs, seq_len=seq_len)
    activation_magnitudes = compute_per_channel_activation_magnitudes(
        state_dict, model_args, seqs
    )

    mutated_keys = apply_awq_to_state_dict(
        state_dict,
        model_args,
        activation_magnitudes,
        alpha=alpha,
        target_modules=target_modules,
    )
    clear_weight_component_cache()

    diagnostics = {
        "alpha": float(alpha),
        "target_modules": tuple(target_modules),
        "n_keys_mutated": len(mutated_keys),
        "n_act_stats_collected": len(activation_magnitudes),
    }
    return mutated_keys, diagnostics

def apply_bias_correction_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    blocks: Sequence[int],
    weight_types: Sequence[str] = ("mlp.c_fc", "mlp.c_proj", "attn.c_proj"),
    n_seqs: int = 8,
    seq_len: int = 64,
) -> tuple[Dict[str, float], List[dict]]:
    """Mutate payload['state_dict'] biases in-place to absorb the per-output-channel
    mean shift introduced by INT8 quantization.

    For each target weight W with input activation X and bias b:
        err[i] = mean over samples of (X @ W.T - X_qdq @ W_qdq.T)[:, i]
        b_corrected[i] = b[i] + err[i]

    Returns the (unchanged) calibration scales and a list of per-layer diagnostics.
    """
    block_set = sorted({int(b) for b in blocks})
    if not block_set:
        return calibration_scales, []
    weight_types_norm = tuple(str(wt) for wt in weight_types)
    if not weight_types_norm:
        return calibration_scales, []

    state_dict = payload["state_dict"]
    model_args = payload["model_args"]

    target_weights: List[str] = []
    for L in block_set:
        for wt in weight_types_norm:
            target_weights.append(f"transformer.h.{L}.{wt}.weight")

    # Capture FP32 activations entering each target layer.
    target_nodes = sorted({_bias_correction_input_node(w) for w in target_weights})
    seqs = build_calibration_seqs_from_token_ids(
        token_ids, n_seqs=int(n_seqs), seq_len=int(seq_len),
    )
    accum: Dict[str, List[np.ndarray]] = {n: [] for n in target_nodes}
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for n in target_nodes:
            arr = np.asarray(node_outputs[n], dtype=np.float32)
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            accum[n].append(arr)
    activations = {n: np.concatenate(rows, axis=0) for n, rows in accum.items()}

    reports: List[dict] = []
    import torch as _torch
    for w in target_weights:
        bias_name = w.replace(".weight", ".bias")
        if bias_name not in state_dict:
            continue

        node = _bias_correction_input_node(w)
        x = activations[node]
        W = _to_f32(state_dict[w])  # [out, in]
        b = _to_f32(state_dict[bias_name])  # [out]

        # Per-channel symmetric INT8 weight quantization (matches the path used
        # elsewhere in the pipeline).
        q_w, scales_w = quantize_tensor(W, per_channel=True)
        scales_w = scales_w.astype(np.float32)
        W_dq = q_w.astype(np.float32) * scales_w.reshape(-1, 1)

        # Per-tensor input activation quantization with the calibration scale.
        x_scale = max(float(calibration_scales.get(node, _DEFAULT_SCALES)), 1e-12)
        x_q = np.clip(np.round(x / x_scale), -128, 127).astype(np.int8)
        x_dq = x_q.astype(np.float32) * np.float32(x_scale)

        y_fp32 = x @ W.T
        y_qdq = x_dq @ W_dq.T
        err_per_channel = np.mean(y_fp32 - y_qdq, axis=0).astype(np.float32)

        b_corrected = (b + err_per_channel).astype(np.float32)
        original_dtype = state_dict[bias_name].dtype if hasattr(state_dict[bias_name], "dtype") else _torch.float32
        state_dict[bias_name] = _torch.from_numpy(np.ascontiguousarray(b_corrected)).to(dtype=original_dtype)

        reports.append({
            "weight": w,
            "bias": bias_name,
            "input_node": node,
            "n_samples": int(x.shape[0]),
            "input_act_scale": float(x_scale),
            "err_abs_max": float(np.max(np.abs(err_per_channel))),
            "err_abs_mean": float(np.mean(np.abs(err_per_channel))),
        })

    return calibration_scales, reports
