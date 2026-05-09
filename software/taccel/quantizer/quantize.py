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


def gptq_quantize(
    tensor: np.ndarray,
    calibration_inputs,
    *,
    per_channel: bool = True,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPTQ (Frantar et al. 2022) per-channel symmetric INT8 quantization.

    Quantizes the input dimension column-by-column, propagating each column's
    rounding error into the unquantized columns via the inverse Hessian
    `H = (X^T X)`. The per-output-channel scale is computed once from the
    original `tensor` (same convention as `quantize_tensor`); the column
    propagation only modifies the per-element rounding decisions, never the
    scale, so the result round-trips through the existing `quantize_tensor`
    consumer just like the clip-search and AdaRound paths do.

    Args:
        tensor: FP32 weight, shape `[out_dim, in_dim]`.
        calibration_inputs: iterable of `[N_i, in_dim]` activation matrices
            captured by FP32-forwarding the layer on calibration tokens. The
            same shape `_flatten_calibration_inputs` accepts elsewhere.
        per_channel: if True, one scale per output channel; if False, one
            shared scale across all rows.
        percdamp: damping ratio added to the Hessian diagonal as
            `percdamp * mean(diag(H))`. The standard GPTQ default is 0.01.
        blocksize: process columns in lazy panels of this width to keep the
            running update on a small slice of the weight matrix at a time
            (perf only — does not change the result).

    Returns:
        `(int8_tensor, fp16_scales)` matching the `quantize_tensor` contract.
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError(f"gptq_quantize expects a 2D tensor, got shape {tensor.shape}")
    out_dim, in_dim = tensor.shape
    if in_dim == 0 or out_dim == 0:
        raise ValueError("gptq_quantize requires non-empty tensor")
    if percdamp <= 0.0:
        raise ValueError("percdamp must be positive")
    if blocksize <= 0:
        raise ValueError("blocksize must be positive")

    calib_rows = _flatten_calibration_inputs(calibration_inputs)
    if calib_rows is None or calib_rows.shape[0] == 0:
        # No calibration data → fall back to RTN, same as quantize_tensor.
        return quantize_tensor(tensor, per_channel=per_channel)
    if calib_rows.shape[1] != in_dim:
        raise ValueError(
            f"calibration input dim {calib_rows.shape[1]} != tensor in_dim {in_dim}"
        )

    # Per-channel scales come from the *original* W so the downstream
    # consumer (quantize_tensor on the rebuilt FP32 weight) recovers the
    # same scale even if the GPTQ-perturbed columns reduce the row maxima.
    if per_channel:
        max_vals = np.maximum(np.max(np.abs(tensor), axis=1), 1e-8).astype(np.float32)
        scales = max_vals / np.float32(127.0)
    else:
        m = max(float(np.max(np.abs(tensor))), 1e-8)
        scales = np.full(out_dim, m / 127.0, dtype=np.float32)

    # Hessian over the input dimension. Use float64 for the linear-algebra
    # step — Cholesky on a 768- or 3072-wide matrix is cheap and the extra
    # precision matters when columns are nearly collinear.
    X = calib_rows.astype(np.float64)
    H = (X.T @ X) * (2.0 / float(X.shape[0]))

    # Drop dead input columns (never excited by calibration). The standard
    # GPTQ trick: pin them to zero in the working copy and replace the
    # corresponding Hessian diagonal with 1 so the Cholesky stays PD.
    diag_idx = np.arange(in_dim)
    dead = np.diag(H) <= 0.0
    if dead.any():
        H[diag_idx[dead], diag_idx[dead]] = 1.0

    # Diagonal damping for numerical stability — proportional to the average
    # diagonal magnitude (Frantar §3.2). Without this the Cholesky often
    # fails on heavy-tailed activation distributions.
    damp = float(percdamp) * float(np.mean(np.diag(H)))
    H[diag_idx, diag_idx] += damp

    # Cholesky factor of H_inv as upper triangular U with U^T U = H_inv.
    # Compute it as L = chol(H_inv); U = L^T.
    try:
        H_inv = np.linalg.inv(H)
        L_inv = np.linalg.cholesky(H_inv)
    except np.linalg.LinAlgError:
        # Damping was insufficient — bump it and retry once.
        H[diag_idx, diag_idx] += 10.0 * damp
        H_inv = np.linalg.inv(H)
        L_inv = np.linalg.cholesky(H_inv)
    U = L_inv.T  # upper triangular, U^T U = H_inv

    W = tensor.astype(np.float64).copy()
    if dead.any():
        W[:, dead] = 0.0
    Q = np.zeros((out_dim, in_dim), dtype=np.int8)
    scales64 = scales.astype(np.float64).reshape(-1, 1)

    # Process columns in panels of `blocksize` so the inner loop touches a
    # narrow slice and the cross-panel update is one batched matmul.
    for col_start in range(0, in_dim, blocksize):
        col_end = min(col_start + blocksize, in_dim)
        panel = col_end - col_start
        # Local accumulator for the running rounding errors so we can update
        # the *rest* of the matrix in a single matmul after the panel.
        Err_panel = np.zeros((out_dim, panel), dtype=np.float64)
        U_panel = U[col_start:col_end, col_start:col_end]
        for j_local in range(panel):
            j = col_start + j_local
            w_j = W[:, j]
            d = U_panel[j_local, j_local]
            # Round-to-nearest with per-channel scale, then dequantize.
            q_int = np.clip(
                np.round(w_j / scales64[:, 0]),
                -128,
                127,
            ).astype(np.int8)
            q_dq = q_int.astype(np.float64) * scales64[:, 0]
            err = (w_j - q_dq) / d
            Err_panel[:, j_local] = err
            # In-panel propagation: shrink the remaining columns inside the
            # current panel by the new error along the diagonal of U.
            if j_local + 1 < panel:
                W[:, j + 1: col_end] -= np.outer(
                    err, U_panel[j_local, j_local + 1:]
                )
            Q[:, j] = q_int
        # Cross-panel propagation: update everything to the right of the
        # panel with the accumulated panel errors at once.
        if col_end < in_dim:
            W[:, col_end:] -= Err_panel @ U[col_start:col_end, col_end:]

    return Q, scales.astype(np.float16)


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
