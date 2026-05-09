"""QuaRot Phase 1 rotation primitives for the GPT-2 PTQ pipeline.

This module provides:
  * Orthogonal matrix builders (Haar-random, block-Hadamard).
  * `rotate_residual_stream_state_dict` — full-network state_dict mutation that
    pre-rotates wte, wpe, every block's c_attn/c_fc input cols and c_proj output
    rows, and lm_head input cols. The rotation is folded entirely into the
    weights; no runtime intervention is needed (modulo β-fold for LN, which
    lives in `taccel.quantizer.ln_fold`).

This is the production counterpart of the diagnostic functions at
`software/tools/diagnose_activation_outliers.py:330-680`. Differences from the
diagnostic:
  * No `target_blocks` knob: rotation always covers the full network. Partial
    rotation is unsound in production (see plan §0).
  * The runtime un-rotate-before-LN trick used by the diagnostic's
    `_ResidualStreamRotatedFQReference` is replaced by the offline β-fold in
    `taccel.quantizer.ln_fold`, which keeps the bundle's LN computation
    unchanged.

Algorithmic reference: QuaRot (Ashkboos et al., 2024) — rotation of the
residual stream by an orthogonal `R` makes activation distributions near-
isotropic, allowing per-tensor INT8 quantization to recover precision lost to
outlier features.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch


__all__ = [
    "build_random_orthogonal",
    "build_block_hadamard_768",
    "rotate_residual_stream_state_dict",
]


def _to_f32(x) -> np.ndarray:
    """Return an FP32 numpy view of `x` (torch tensor, numpy array, or scalar)."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Orthogonal-matrix builders
# ---------------------------------------------------------------------------


def build_random_orthogonal(d: int, *, seed: int = 0xCAFE) -> np.ndarray:
    """Return a `d × d` orthogonal matrix that PRESERVES the all-ones vector
    (`R · 1 = 1`).

    For non-RMSNorm models (like GPT-2, where LayerNorm includes mean
    subtraction), the rotation must satisfy `R · 1 = 1` (equivalently
    `R^T · 1 = 1` for orthogonal R) so that `mean(R · x) = mean(x)` for any
    `x`. Without this property, LayerNorm's mean-subtraction step does not
    commute with rotation: `LN(R · x) ≠ R · LN(x)` even after β-fold + γ-fold,
    breaking the FP32 equivalence that the QuaRot Phase 1 recipe relies on.

    Construction:
      1. v = 1 / sqrt(d) (unit-norm all-ones vector).
      2. Build orthonormal basis `Q_perp` of `perp(v)` by QR decomposition of
         a random matrix projected onto `perp(v)`.
      3. Generate a random Haar orthogonal `U_inner` in `(d-1)` dimensions
         (sign-corrected QR of a Gaussian).
      4. `R = v vᵀ + Q_perp · U_inner · Q_perpᵀ`.

    Properties:
      * `R · v = v` (hence `R · 1 = 1`).
      * `R · Rᵀ = I` (orthogonal in d dimensions).
      * Distribution: uniform over the subgroup of orthogonal matrices that
        fix `v`. Equivalent to Haar measure on `O(d-1)` lifted to `O(d)`
        through the `Q_perp` basis.

    The seed makes the output deterministic (necessary for reproducible PTQ
    runs and for the `quarot_seed` preset field).

    Returns: FP32 ndarray of shape `(d, d)`.
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if d == 1:
        # Degenerate: only [[1]] is orthogonal-and-1-preserving in 1d.
        return np.array([[1.0]], dtype=np.float32)

    rng = np.random.default_rng(seed)
    v = np.ones(d, dtype=np.float64) / np.sqrt(d)

    # Step 1-2: orthonormal basis of perp(v).
    # Generate (d × (d-1)) Gaussian, project onto perp(v), then QR-orthonormalize.
    g = rng.standard_normal((d, d - 1)).astype(np.float64)
    g = g - np.outer(v, v.T @ g)  # subtract component along v
    Q_perp, _ = np.linalg.qr(g)  # shape (d, d-1), columns span perp(v)

    # Step 3: Haar orthogonal in (d-1) dimensions.
    g_inner = rng.standard_normal((d - 1, d - 1)).astype(np.float64)
    U_inner, R_inner = np.linalg.qr(g_inner)
    sign = np.sign(np.diag(R_inner))
    U_inner = U_inner * sign[np.newaxis, :]

    # Step 4: assemble R = v v^T + Q_perp · U_inner · Q_perp^T.
    R = np.outer(v, v) + Q_perp @ U_inner @ Q_perp.T
    return R.astype(np.float32)


def _sylvester_hadamard(n: int) -> np.ndarray:
    """Sylvester-construction Hadamard matrix of size `n`. `n` must be `2**k`."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"size must be a power of 2, got {n}")
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def build_block_hadamard_768() -> np.ndarray:
    """Return a `768 × 768` block-Hadamard `I_12 ⊗ H_64`, normalized to be
    orthogonal.

    `768 = 12 × 64`. Twelve `64 × 64` Sylvester Hadamards on the diagonal,
    each scaled by `1/sqrt(64)` so the full matrix is orthogonal. This matches
    QuaRot's head-dim-aligned Hadamard structure and is the kind a compact
    on-chip Hadamard unit would compute.

    Currently reserved for future `quarot_kind="block_hadamard_768"` support;
    not exposed as a preset value in this PR (random orthogonal is the
    default).
    """
    H64 = _sylvester_hadamard(64) / np.sqrt(64.0)
    H = np.zeros((768, 768), dtype=np.float64)
    for b in range(12):
        H[b * 64 : (b + 1) * 64, b * 64 : (b + 1) * 64] = H64
    return H.astype(np.float32)


# ---------------------------------------------------------------------------
# State_dict rotation
# ---------------------------------------------------------------------------


def _store(state_dict: dict, key: str, new_value: np.ndarray) -> None:
    """Replace `state_dict[key]` with `new_value` while preserving the original
    tensor's dtype.

    Mirrors the pattern used by `apply_bias_correction_from_token_ids` in
    `taccel/runtime/calibration.py` so downstream code (bundle builder,
    `_cached_weight_components`) sees tensors with the dtype it expects.
    """
    old = state_dict[key]
    if hasattr(old, "dtype") and hasattr(old, "to"):
        state_dict[key] = torch.from_numpy(new_value).to(dtype=old.dtype)
    else:
        state_dict[key] = torch.from_numpy(new_value)


def rotate_residual_stream_state_dict(
    state_dict: dict,
    model_args: dict,
    R: np.ndarray,
) -> List[str]:
    """Mutate `state_dict` in place to pre-rotate every weight and LayerNorm
    bias that touches the residual stream.

    PRECONDITION: `taccel.quantizer.ln_fold.fold_layernorm_for_quarot` has
    already been applied to `state_dict`. After that γ-fold, every LN has
    `γ = 1` and consumer weights have `γ` absorbed into their input columns.
    LayerNorm at runtime computes `(x - mean(x)) / sqrt(var(x) + ε) + β` (with
    γ = 1), which commutes with a 1-preserving rotation R: `mean(R·x) = mean(x)`
    and `var(R·x) = var(x)`, so `LN_no_γ(R·x) = R·LN_no_γ(x) - R·β + R·β = R·LN_no_γ_no_β(x) + β`.
    To preserve full equivalence with the unrotated network, β must also be
    rotated (see step 4 below).

    Mutations applied (full network — never partial):

      1. Embeddings (rows rotated → embedding output enters residual in
         rotated basis):
           `transformer.wte.weight` ← `wte @ R^T`
           `transformer.wpe.weight` ← `wpe @ R^T`

      2. For every block `L`:

         a. Per-head c_attn input cols (consumer reads rotated LN output):
              `transformer.h.{L}.attn.c_attn.weight_h{H}_{query,key,value}`
                ← `W @ R^T`

         b. mlp.c_fc input cols:
              `transformer.h.{L}.mlp.c_fc.weight` ← `W @ R^T`

         c. attn.c_proj output rows (output rotated → enters residual in
            rotated basis):
              `transformer.h.{L}.attn.c_proj.weight` ← `R @ W`
              `transformer.h.{L}.attn.c_proj.bias`   ← `R @ b`

         d. mlp.c_proj output rows:
              `transformer.h.{L}.mlp.c_proj.weight` ← `R @ W`
              `transformer.h.{L}.mlp.c_proj.bias`   ← `R @ b`

      3. lm_head input cols (reads from rotated `ln_f` output):
           `lm_head.weight` ← `W @ R^T`

      4. LN biases (β-rotation; absorbs β-fold without needing lm_head.bias):
           `transformer.h.{L}.ln_1.bias` ← `R @ β`  (per block)
           `transformer.h.{L}.ln_2.bias` ← `R @ β`  (per block)
           `transformer.ln_f.bias`       ← `R @ β`

    Why β-rotation rather than β-fold: with 1-preserving R, rotating β by
    `β_new = R @ β` makes the LN output `R·u + R·β = R·(u + β)`, which
    composes with the rotated consumer weight `(W·γ) @ R^T` to give matmul
    output `(u + β) @ (W·γ)^T` — bit-identical to the unrotated γ-folded
    case. This avoids creating an `lm_head.bias` (which standard GPT-2 lacks
    and the codegen does not read).

    Net effect in FP32: residual stream lives in rotated basis end-to-end,
    but logits are unchanged because every "in" and "out" of the rotated
    basis cancels.

    Net effect in INT8: per-tensor activation scales on the residual stream
    are computed against the *rotated* (near-isotropic) distribution —
    dramatically tighter than the unrotated 99.9-percentile scale on the
    original outlier-laden distribution. This is the QuaRot win.

    Args:
        state_dict: `payload["state_dict"]`. Mutated in place.
        model_args: `payload["model_args"]` — used for `n_layer`.
        R: `[d_model, d_model]` orthogonal rotation matrix. Float32.
            Must satisfy `R · 1 = 1` (1-preserving). Use
            `build_random_orthogonal` which guarantees this.

    Returns:
        List of `state_dict` keys mutated, in the order they were modified.

    Raises:
        ValueError if `R` is not square or `R.shape[0] != model_args["n_embd"]`.
    """
    R_arr = np.asarray(R, dtype=np.float32)
    if R_arr.ndim != 2 or R_arr.shape[0] != R_arr.shape[1]:
        raise ValueError(f"R must be square, got shape {R_arr.shape}")
    d_model = int(model_args["n_embd"])
    if R_arr.shape[0] != d_model:
        raise ValueError(
            f"R shape {R_arr.shape} does not match n_embd={d_model}"
        )
    n_layer = int(model_args["n_layer"])

    R_t = R_arr.T
    modified: List[str] = []

    def _do(key: str, new_value: np.ndarray) -> None:
        if key in state_dict:
            _store(state_dict, key, new_value)
            modified.append(key)

    # 1. Embeddings.
    for key in ("transformer.wte.weight", "transformer.wpe.weight"):
        if key in state_dict:
            w = _to_f32(state_dict[key])
            _do(key, w @ R_t)

    # 2. Per-block input/output rotations.
    for L in range(n_layer):
        # 2a. c_attn per-head input cols.
        H = 0
        while True:
            base = f"transformer.h.{L}.attn.c_attn.weight_h{H}"
            if f"{base}_query" not in state_dict:
                break
            for kind in ("query", "key", "value"):
                k = f"{base}_{kind}"
                if k in state_dict:
                    w = _to_f32(state_dict[k])  # [d_head, d_model]
                    _do(k, w @ R_t)
            H += 1

        # 2b. mlp.c_fc input cols.
        key = f"transformer.h.{L}.mlp.c_fc.weight"
        if key in state_dict:
            w = _to_f32(state_dict[key])  # [4*d_model, d_model]
            _do(key, w @ R_t)

        # 2c. attn.c_proj output rows + bias.
        key = f"transformer.h.{L}.attn.c_proj.weight"
        if key in state_dict:
            w = _to_f32(state_dict[key])  # [d_model, d_model]
            _do(key, R_arr @ w)
        bkey = f"transformer.h.{L}.attn.c_proj.bias"
        if bkey in state_dict:
            b = _to_f32(state_dict[bkey])  # [d_model]
            _do(bkey, R_arr @ b)

        # 2d. mlp.c_proj output rows + bias.
        key = f"transformer.h.{L}.mlp.c_proj.weight"
        if key in state_dict:
            w = _to_f32(state_dict[key])  # [d_model, 4*d_model]
            _do(key, R_arr @ w)
        bkey = f"transformer.h.{L}.mlp.c_proj.bias"
        if bkey in state_dict:
            b = _to_f32(state_dict[bkey])
            _do(bkey, R_arr @ b)

    # 3. lm_head input cols.
    key = "lm_head.weight"
    if key in state_dict:
        w = _to_f32(state_dict[key])  # [vocab, d_model]
        _do(key, w @ R_t)

    # NOTE: LN biases (β) are NOT rotated. After `fold_layernorm_for_quarot`,
    # every LN.β has been zeroed (β-fold absorbs the contribution into the
    # consumer's bias, including a freshly created `lm_head.bias`). Rotating
    # zeros would still produce zeros; we omit the work.
    #
    # Consumer biases (`c_proj.bias`, `fc2.bias`) ARE rotated above (steps 2c,
    # 2d) because their outputs land in the rotated residual stream.
    # `c_attn.bias_h{H}_*` (Q/K/V) are NOT rotated — they live in unrotated
    # head/MLP-internal output basis.
    # `c_fc.bias`, `c_proj_attn.bias_h{H}_*` analogous.
    # `lm_head.bias` (created by β-fold) is NOT rotated — it's added to the
    # logit output, which lives in the unrotated logit basis.

    return modified
