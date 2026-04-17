"""Per-channel symmetric INT8 weight quantization."""
import numpy as np
from typing import Any, Dict, Optional, Tuple


def quantize_tensor(tensor: np.ndarray, per_channel: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a 2D tensor to INT8 with per-channel symmetric quantization.

    Args:
        tensor: FP32 tensor of shape [out_channels, in_features]
        per_channel: if True, compute scale per output channel

    Returns:
        (int8_tensor, scales): quantized tensor and per-channel FP16 scales
    """
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)

    if per_channel:
        # Per-channel: scale[ch] = max(abs(W[ch,:])) / 127
        max_vals = np.max(np.abs(tensor), axis=1)
        max_vals = np.maximum(max_vals, 1e-8)  # avoid division by zero
        scales = max_vals / 127.0
    else:
        # Per-tensor
        max_val = max(np.max(np.abs(tensor)), 1e-8)
        scales = np.full(tensor.shape[0], max_val / 127.0)

    # Quantize
    scales_expanded = scales.reshape(-1, 1)
    q = np.clip(np.round(tensor / scales_expanded), -128, 127).astype(np.int8)

    return q, scales.astype(np.float16)


def _flatten_calibration_inputs(calibration_inputs) -> Optional[np.ndarray]:
    if calibration_inputs is None:
        return None
    rows = []
    for sample in calibration_inputs:
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        rows.append(arr)
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def quantize_tensor_clipped(
    tensor: np.ndarray,
    calibration_inputs=None,
    *,
    per_channel: bool = True,
    n_candidates: int = 25,
    alpha_min: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a tensor with clip-search, optionally minimizing output MSE."""
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    if not (0.0 < alpha_min <= 1.0):
        raise ValueError("alpha_min must be in (0, 1]")

    alphas = np.linspace(alpha_min, 1.0, n_candidates, dtype=np.float32)
    calib_rows = _flatten_calibration_inputs(calibration_inputs)
    gram = None
    if calib_rows is not None:
        gram = (calib_rows.T @ calib_rows).astype(np.float32) / max(float(calib_rows.shape[0]), 1.0)

    if per_channel:
        max_vals = np.maximum(np.max(np.abs(tensor), axis=1), 1e-8).astype(np.float32)
        best_scores = np.full(tensor.shape[0], np.inf, dtype=np.float32)
        best_q = None
        best_scales = np.full(tensor.shape[0], max_vals / 127.0, dtype=np.float32)
        for alpha in alphas:
            scales = np.maximum(alpha * max_vals, 1e-8) / 127.0
            q = np.clip(np.round(tensor / scales.reshape(-1, 1)), -128, 127).astype(np.int8)
            dq = q.astype(np.float32) * scales.reshape(-1, 1)
            diff = dq - tensor
            if gram is not None:
                scores = np.einsum("oi,ij,oj->o", diff, gram, diff, optimize=True).astype(np.float32)
            else:
                scores = np.mean(diff ** 2, axis=1, dtype=np.float32)
            if best_q is None:
                best_q = q.copy()
            improved = scores < best_scores
            if np.any(improved):
                best_scores[improved] = scores[improved]
                best_scales[improved] = scales[improved]
                best_q[improved] = q[improved]
        return best_q.astype(np.int8), best_scales.astype(np.float16)

    max_val = max(float(np.max(np.abs(tensor))), 1e-8)
    best_score = float("inf")
    best_q = None
    best_scale = max_val / 127.0
    for alpha in alphas:
        scale = max(alpha * max_val, 1e-8) / 127.0
        q = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
        dq = q.astype(np.float32) * np.float32(scale)
        diff = dq - tensor
        if gram is not None:
            per_row = np.einsum("oi,ij,oj->o", diff, gram, diff, optimize=True).astype(np.float32)
            score = float(np.mean(per_row))
        else:
            score = float(np.mean(diff ** 2, dtype=np.float32))
        if score < best_score:
            best_score = score
            best_scale = scale
            best_q = q.copy()
    return best_q.astype(np.int8), np.full(tensor.shape[0], best_scale, dtype=np.float16)


def adaround_greedy(
    tensor: np.ndarray,
    q_init: np.ndarray,
    scales: np.ndarray,
    calibration_inputs,
    *,
    frac_lo: float = 0.3,
    frac_hi: float = 0.7,
    max_accepts_per_channel: Optional[int] = None,
) -> np.ndarray:
    """Greedily flip rounding direction for near-boundary weights.

    The search is local and calibration-aware: for each output channel we start
    from an existing quantization (`q_init`, typically from clip search), then
    consider moving each near-half-LSB weight to the alternative adjacent
    integer if doing so improves the layer's output MSE on calibration inputs.
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    q = np.asarray(q_init, dtype=np.int8).copy()
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
        q = q.reshape(1, -1)
    if tensor.shape != q.shape:
        raise ValueError("tensor and q_init must have the same shape")

    calib_rows = _flatten_calibration_inputs(calibration_inputs)
    if calib_rows is None:
        return q
    gram = (calib_rows.T @ calib_rows).astype(np.float32) / max(float(calib_rows.shape[0]), 1.0)
    gram_diag = np.diag(gram).astype(np.float32)

    scales_f32 = np.asarray(scales, dtype=np.float32)
    if scales_f32.ndim == 0:
        scales_f32 = np.full(tensor.shape[0], float(scales_f32), dtype=np.float32)
    elif scales_f32.shape[0] != tensor.shape[0]:
        raise ValueError("scales must have one value per output channel")

    for ch in range(tensor.shape[0]):
        scale = float(scales_f32[ch])
        continuous = tensor[ch] / max(scale, 1e-12)
        fractional = np.abs(continuous - np.trunc(continuous))
        candidates = np.where((fractional >= frac_lo) & (fractional <= frac_hi))[0]
        if candidates.size == 0:
            continue

        alt_q = q[ch].astype(np.int16).copy()
        delta_int = np.where(continuous > q[ch].astype(np.float32), 1, -1).astype(np.int16)
        alt_q[candidates] = np.clip(alt_q[candidates] + delta_int[candidates], -128, 127)
        candidates = candidates[alt_q[candidates] != q[ch, candidates].astype(np.int16)]
        if candidates.size == 0:
            continue

        diff = q[ch].astype(np.float32) * scale - tensor[ch]
        current_score = float(diff @ gram @ diff)
        accepted = 0
        remaining = candidates.tolist()
        while remaining:
            idxs = np.asarray(remaining, dtype=np.int32)
            step = (alt_q[idxs].astype(np.float32) - q[ch, idxs].astype(np.float32)) * scale
            # score(d + step*e_i) = score(d) + 2*step*(G_i·d) + step^2*G_ii
            g_dot_d = gram[idxs] @ diff
            deltas = 2.0 * step * g_dot_d + (step ** 2) * gram_diag[idxs]
            best_pos = int(np.argmin(deltas))
            if deltas[best_pos] >= -1e-12:
                break
            idx = int(idxs[best_pos])
            q[ch, idx] = np.int8(alt_q[idx])
            diff[idx] += float(step[best_pos])
            current_score += float(deltas[best_pos])
            remaining.remove(idx)
            accepted += 1
            if max_accepts_per_channel is not None and accepted >= max_accepts_per_channel:
                break

    return q


def dequantize_tensor(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize INT8 tensor back to FP32."""
    if q.ndim == 1:
        return q.astype(np.float32) * float(scales[0])
    scales_expanded = scales.astype(np.float32).reshape(-1, 1)
    return q.astype(np.float32) * scales_expanded


def quantize_weights(
    state_dict: dict,
    quantization_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Quantize all weight tensors in a state dict.

    Returns dict mapping weight names → (int8_tensor, scale_per_channel_fp16).
    Conv2d patch embedding is reshaped to 2D before quantizing.
    """
    result = {}
    overrides = quantization_overrides or {}
    for name, tensor in state_dict.items():
        if not hasattr(tensor, 'numpy'):
            continue
        t = tensor.numpy().astype(np.float32)

        if 'weight' in name and t.ndim >= 2:
            # Reshape conv2d [out, in, H, W] → [out, in*H*W]
            if t.ndim == 4:
                t = t.reshape(t.shape[0], -1)
            elif t.ndim > 2:
                t = t.reshape(t.shape[0], -1)
            override = overrides.get(name)
            if override is not None:
                q, scales = quantize_tensor_clipped(
                    t,
                    calibration_inputs=override.get("calibration_inputs"),
                    per_channel=bool(override.get("per_channel", True)),
                    n_candidates=int(override.get("n_candidates", 25)),
                    alpha_min=float(override.get("alpha_min", 0.5)),
                )
                if override.get("adaround"):
                    q = adaround_greedy(
                        t,
                        q,
                        scales.astype(np.float32),
                        override.get("calibration_inputs"),
                        frac_lo=float(override.get("adaround_frac_lo", 0.3)),
                        frac_hi=float(override.get("adaround_frac_hi", 0.7)),
                        max_accepts_per_channel=override.get("adaround_max_accepts_per_channel"),
                    )
            else:
                q, scales = quantize_tensor(t)
            result[name] = (q, scales)
        elif 'bias' in name and t.ndim == 1:
            # 1D biases are LayerNorm beta — store as FP16 to match gamma convention.
            # Matmul biases are also 1D but they are handled by _prescale_biases in
            # the compiler (which reads from state_dict directly, not from here).
            # The SFU (sfu.py) reads gamma then beta both as FP16, so both must be FP16.
            weight_name = name.replace('.bias', '.weight')
            if weight_name in state_dict:
                w = state_dict[weight_name]
                if hasattr(w, 'numpy') and w.numpy().ndim >= 2:
                    # Matmul/conv bias — keep as FP32 for pre-scaling
                    result[name] = (t, None)
                else:
                    # LayerNorm beta (weight is 1D) — store as FP16
                    result[name] = (t.astype(np.float16), None)
            else:
                result[name] = (t, None)
        elif t.ndim <= 2:
            # LayerNorm gamma, cls_token, pos_embed, etc. — store as FP16 for SFU
            result[name] = (t.astype(np.float16), None)

    return result
