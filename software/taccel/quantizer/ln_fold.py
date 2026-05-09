"""LayerNorm γ-fold + β-fold for QuaRot Phase 1.

For QuaRot Phase 1 with a 1-preserving rotation `R`, every LayerNorm needs
two transformations:

  1. **γ-fold**: fold `γ` (LN gain) into the input columns of every consumer
     weight, then set `γ = ones`.

  2. **β-fold**: fold `β` (LN bias) into every consumer's output bias:
     ``b_consumer_new = b_consumer + W_consumer_orig @ β``.
     Then set `β = zeros`.

For consumers without a pre-existing bias key (notably `lm_head`), this fold
**creates a new bias key** in `state_dict`. Callers must ensure downstream
paths (`_fp32_forward`, `NanoGPTFQReference`, codegen) read the new key.

Why β-fold instead of β-rescale (β/γ): β-rescale produces unstable values
when γ has small components — for trained GPT-2, ``ln_f.γ`` has values down
to 0.0044, producing ``|β/γ|`` magnitudes up to 224. After rotation those
large values dominate the LN output magnitude, forcing the per-tensor
INT8 calibration scale to be coarse, and the resulting quantization noise
compounds at long-eval. Empirical: β-rescale gave 35,313 PPL at 257-tok
where proper β-fold should give ~3,000–4,000 (matching the diagnostic).

Mathematical equivalence (unrotated case):
  Original: LN(x) @ W^T + b
         = (γ ⊙ x_norm + β) @ W^T + b
  After γ-fold (W_new = W * γ[None, :], γ_LN = 1) AND β-fold
  (b_new = b + W_orig @ β, β_LN = 0):
     LN_new(x) = x_norm
     LN_new(x) @ W_new^T + b_new
       = x_norm @ (W * γ[None, :])^T + b + W @ β
       = (γ ⊙ x_norm) @ W^T + b + W @ β
       = (γ ⊙ x_norm) @ W^T + β @ W^T + b   [since W @ β = β @ W^T as 1d]
       = (γ ⊙ x_norm + β) @ W^T + b = original ✓

Mathematical equivalence (rotated case, with 1-preserving R):
  After γ-fold: consumer.W_rotated = (W * γ) @ R^T.
  After β-fold: consumer.b_new = b + W_orig @ β (NOT rotated — bias is in
    the unrotated head/MLP-internal output basis).
  After ln_f β-fold: lm_head.bias_new = lm_head_orig @ β (NEW key).
  Rotated LN output (with γ=1, β=0): R·u (with 1-preserving R, mean and
  variance preserved).
  Consumer matmul:
    R·u @ ((W * γ) @ R^T)^T + b_new
      = R·u @ R @ (W * γ)^T + b + W @ β
      = u @ (W * γ)^T + b + W @ β
      = (γ ⊙ u) @ W^T + b + β @ W^T
      = (γ ⊙ u + β) @ W^T + b = original ✓

LN consumer mapping (GPT-2 / nanoGPT):
  * `transformer.h.{L}.ln_1` → 3·n_head consumers per block:
      `transformer.h.{L}.attn.c_attn.weight_h{H}_{query,key,value}`
      (each has bias_h{H}_{query,key,value} for β-fold)
  * `transformer.h.{L}.ln_2` → 1 consumer per block:
      `transformer.h.{L}.mlp.c_fc.weight` (bias mlp.c_fc.bias)
  * `transformer.ln_f` → 1 consumer global:
      `lm_head.weight` (NO PRE-EXISTING BIAS — fold creates `lm_head.bias`)
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


__all__ = [
    "fold_layernorm_for_quarot",
]


def _to_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _store(state_dict: dict, key: str, new_value: np.ndarray, *, prefer_dtype=None) -> None:
    """Replace `state_dict[key]` with `new_value`. If the key is new and no
    `prefer_dtype` given, store as FP32; if existing, preserve dtype."""
    if key in state_dict:
        old = state_dict[key]
        if hasattr(old, "dtype") and hasattr(old, "to"):
            state_dict[key] = torch.from_numpy(new_value).to(dtype=old.dtype)
            return
    if prefer_dtype is not None:
        state_dict[key] = torch.from_numpy(new_value).to(dtype=prefer_dtype)
    else:
        state_dict[key] = torch.from_numpy(new_value.astype(np.float32))


def _gamma_fold_consumer(
    state_dict: dict,
    consumer_weight_key: str,
    gamma: np.ndarray,
) -> bool:
    """Apply γ-fold to one consumer weight: ``W_new = W * γ[None, :]``.
    Returns True if the key was present and folded, False otherwise."""
    if consumer_weight_key not in state_dict:
        return False
    W = _to_f32(state_dict[consumer_weight_key])  # [d_out, d_in]
    if W.shape[-1] != gamma.shape[0]:
        raise ValueError(
            f"γ-fold dim mismatch: consumer {consumer_weight_key!r} has "
            f"in_features={W.shape[-1]}, γ has {gamma.shape[0]}"
        )
    W_new = (W * gamma[None, :]).astype(np.float32)
    _store(state_dict, consumer_weight_key, W_new)
    return True


def _beta_fold_consumer(
    state_dict: dict,
    consumer_weight_key: str,
    consumer_bias_key: str,
    beta: np.ndarray,
    *,
    create_bias_if_missing: bool = False,
    bias_dtype=None,
) -> bool:
    """Apply β-fold: ``b_new = b + W_orig @ β`` (using ORIGINAL pre-γ-fold W).

    The ORIGINAL W must be passed in via state_dict — caller must call
    `_beta_fold_consumer` BEFORE `_gamma_fold_consumer` so that the W
    here is unrotated and pre-γ-fold.

    Returns True if any modification occurred.

    If `consumer_bias_key` does not exist and `create_bias_if_missing` is
    False, the function returns False without modification.
    """
    if consumer_weight_key not in state_dict:
        return False
    W = _to_f32(state_dict[consumer_weight_key])  # [d_out, d_in]
    if W.shape[-1] != beta.shape[0]:
        raise ValueError(
            f"β-fold dim mismatch: consumer {consumer_weight_key!r} has "
            f"in_features={W.shape[-1]}, β has {beta.shape[0]}"
        )
    contribution = (W @ beta).astype(np.float32)  # [d_out]

    if consumer_bias_key in state_dict:
        b_old = _to_f32(state_dict[consumer_bias_key])
        b_new = b_old + contribution
        _store(state_dict, consumer_bias_key, b_new)
        return True

    if create_bias_if_missing:
        # Materialize a brand-new bias entry. Default dtype: same as the
        # weight, falling back to FP32.
        if bias_dtype is None and hasattr(state_dict[consumer_weight_key], "dtype"):
            bias_dtype = state_dict[consumer_weight_key].dtype
        _store(state_dict, consumer_bias_key, contribution, prefer_dtype=bias_dtype)
        return True

    return False


def _is_already_folded(gamma: np.ndarray, beta: np.ndarray | None) -> bool:
    """Detect whether an LN has already been folded: γ ≈ ones AND β ≈ zeros."""
    if not np.allclose(gamma, np.ones_like(gamma), atol=1e-6):
        return False
    if beta is not None and not np.allclose(beta, np.zeros_like(beta), atol=1e-6):
        return False
    return True


def _fold_one_layernorm(
    state_dict: dict,
    ln_weight_key: str,
    ln_bias_key: str,
    consumer_weight_keys: List[str],
    consumer_bias_keys: List[str],
    create_missing_consumer_biases: bool,
    modified: List[str],
) -> None:
    """Fold one LN: β-fold into consumer biases (using ORIGINAL W), then
    γ-fold consumer weights, then zero out LN.γ and LN.β in state_dict.

    Order matters: β-fold uses the ORIGINAL pre-γ-fold W. Doing β-fold
    first (or copying W beforehand) and γ-fold second avoids needing to
    track the γ inside β-fold.

    Idempotent: if `γ ≈ ones` AND `β ≈ zeros`, the LN has already been
    folded and this function is a no-op. This protects callers that
    may invoke `fold_layernorm_for_quarot` more than once on the same
    state_dict (e.g., test fixtures that re-fold after restoration).
    """
    if ln_weight_key not in state_dict:
        return
    gamma = _to_f32(state_dict[ln_weight_key])
    beta = _to_f32(state_dict[ln_bias_key]) if ln_bias_key in state_dict else None

    if _is_already_folded(gamma, beta):
        # Already folded — skip to avoid double-applying β-fold.
        return

    # Step 1: β-fold (uses original W).
    if beta is not None and len(consumer_weight_keys) == len(consumer_bias_keys):
        for w_key, b_key in zip(consumer_weight_keys, consumer_bias_keys):
            if _beta_fold_consumer(
                state_dict, w_key, b_key, beta,
                create_bias_if_missing=create_missing_consumer_biases,
            ):
                modified.append(b_key)

    # Step 2: γ-fold consumers (uses W AFTER β-fold's bias update — the
    # weight itself was unchanged by step 1, so γ-fold sees the original W).
    for w_key in consumer_weight_keys:
        if _gamma_fold_consumer(state_dict, w_key, gamma):
            modified.append(w_key)

    # Step 3: zero out LN's γ (set to ones) and β (set to zeros).
    ones_g = np.ones_like(gamma)
    _store(state_dict, ln_weight_key, ones_g)
    modified.append(ln_weight_key)

    if beta is not None:
        zeros_b = np.zeros_like(beta)
        _store(state_dict, ln_bias_key, zeros_b)
        modified.append(ln_bias_key)


def fold_layernorm_for_quarot(state_dict: dict, model_args: dict) -> List[str]:
    """Fold every LayerNorm's γ into consumer weights and β into consumer
    biases. Mutates `state_dict` in place. After this function:

      * Every `ln_*.weight` (γ) is `ones(d_model)`.
      * Every `ln_*.bias` (β) is `zeros(d_model)`.
      * Every consumer weight has had γ folded into its input columns.
      * Every consumer bias has had `W_orig @ β` added.
      * `lm_head.bias` has been CREATED if it didn't exist (downstream
        code must read it; see `_fp32_forward` and `NanoGPTFQReference`).

    Args:
        state_dict: `payload["state_dict"]`. Mutated in place.
        model_args: `payload["model_args"]` — used for `n_layer`.

    Returns:
        List of `state_dict` keys mutated, in order.

    Raises:
        ValueError if a γ vector's dim doesn't match a consumer weight's
            input dim (signals a corrupt or unsupported state_dict).
    """
    n_layer = int(model_args["n_layer"])
    modified: List[str] = []

    # Per-block LN_1 → c_attn (per-head Q, K, V).
    for L in range(n_layer):
        consumer_weight_keys: List[str] = []
        consumer_bias_keys: List[str] = []
        H = 0
        while True:
            base = f"transformer.h.{L}.attn.c_attn.weight_h{H}"
            if f"{base}_query" not in state_dict:
                break
            for kind in ("query", "key", "value"):
                consumer_weight_keys.append(f"{base}_{kind}")
                consumer_bias_keys.append(
                    f"transformer.h.{L}.attn.c_attn.bias_h{H}_{kind}"
                )
            H += 1
        _fold_one_layernorm(
            state_dict,
            ln_weight_key=f"transformer.h.{L}.ln_1.weight",
            ln_bias_key=f"transformer.h.{L}.ln_1.bias",
            consumer_weight_keys=consumer_weight_keys,
            consumer_bias_keys=consumer_bias_keys,
            create_missing_consumer_biases=True,  # safe; codegen reads zeros
            modified=modified,
        )

    # Per-block LN_2 → c_fc.
    for L in range(n_layer):
        _fold_one_layernorm(
            state_dict,
            ln_weight_key=f"transformer.h.{L}.ln_2.weight",
            ln_bias_key=f"transformer.h.{L}.ln_2.bias",
            consumer_weight_keys=[f"transformer.h.{L}.mlp.c_fc.weight"],
            consumer_bias_keys=[f"transformer.h.{L}.mlp.c_fc.bias"],
            create_missing_consumer_biases=False,  # c_fc.bias always exists
            modified=modified,
        )

    # Global ln_f → lm_head.
    # lm_head has NO pre-existing bias in standard GPT-2; we CREATE
    # `lm_head.bias` here so downstream paths must read it.
    _fold_one_layernorm(
        state_dict,
        ln_weight_key="transformer.ln_f.weight",
        ln_bias_key="transformer.ln_f.bias",
        consumer_weight_keys=["lm_head.weight"],
        consumer_bias_keys=["lm_head.bias"],
        create_missing_consumer_biases=True,  # creates lm_head.bias
        modified=modified,
    )

    return modified
