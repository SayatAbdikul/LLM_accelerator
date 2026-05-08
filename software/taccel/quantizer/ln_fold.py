"""LayerNorm γ-fold + β-rescale for QuaRot Phase 1.

For QuaRot Phase 1 with a 1-preserving rotation `R`, every LayerNorm needs
two transformations:

  1. **γ-fold**: fold `γ` (LN gain) into the input columns of every consumer
     weight, then set `γ = ones`.

  2. **β-rescale**: replace LN's `β` with `β / γ_orig` (elementwise division).
     This compensates for the γ-fold's effect on the β contribution to the
     consumer matmul output, without requiring an `lm_head.bias` (which
     standard GPT-2 lacks and the codegen does not read).

After this fold, LN computes `(x - mean(x)) / sqrt(var(x) + ε) + (β / γ)` at
runtime (with γ=1 baked in). The β-rotation in `rotation.py` left-multiplies
this β-rescaled value by R, producing a fully-equivalent rotated network.

Why β-rescale instead of β-fold:
  Standard QuaRot β-fold: `b_consumer ← b_consumer + W_consumer @ β`, then
  `β = 0`. This works for consumers that have biases (LN_1 → c_attn,
  LN_2 → c_fc), but fails for `lm_head` (no bias). β-rescale avoids the
  consumer-bias modification entirely, keeping both the mathematics and the
  state_dict structure clean.

Mathematical equivalence (unrotated case):
  Original: LN(x) @ W^T + b
         = (γ ⊙ x_norm + β) @ W^T + b
         = (γ ⊙ x_norm) @ W^T + β @ W^T + b
  Where x_norm = (x - μ(x)) / σ(x).

  After γ-fold (W_new = W ⊙_cols γ, set LN.γ = 1) AND β-rescale (set
  LN.β = β / γ):
    LN_new(x) = 1 ⊙ x_norm + β/γ = x_norm + β/γ
    LN_new(x) @ W_new^T + b
         = (x_norm + β/γ) @ (W * γ[None, :])^T + b
         = γ ⊙ x_norm @ W^T + (γ ⊙ β/γ) @ W^T + b
         = γ ⊙ x_norm @ W^T + β @ W^T + b
         = original ✓

Mathematical equivalence (rotated case, with 1-preserving R):
  After rotation, consumer.W_rotated = (W * γ) @ R^T. LN output is
    R·(x_unrot_norm) + R·(β/γ) = R·(x_unrot_norm + β/γ)
  Consumer matmul:
    R·(x_unrot_norm + β/γ) @ ((W * γ) @ R^T)^T
      = R·(x_unrot_norm + β/γ) @ R @ (W * γ)^T
      = (x_unrot_norm + β/γ) @ (W * γ)^T  [since R^T R = I]
      = γ ⊙ x_unrot_norm @ W^T + γ ⊙ β/γ @ W^T
      = γ ⊙ x_unrot_norm @ W^T + β @ W^T
      = original ✓

Numerical concern: β-rescale divides by γ. If any γ component is near zero,
β/γ explodes. For trained GPT-2 LayerNorms, γ values are positive and
typically in [0.5, 2.0] — well-behaved. We add a small ε guard
(γ_safe = sign(γ) · max(|γ|, 1e-6)) to avoid division-by-zero in degenerate
cases; this introduces at most O(1e-6) drift on degenerate channels.

LN consumer mapping (GPT-2 / nanoGPT):
  * `transformer.h.{L}.ln_1` → 3·n_head consumers per block:
      `transformer.h.{L}.attn.c_attn.weight_h{H}_{query,key,value}`
  * `transformer.h.{L}.ln_2` → 1 consumer per block:
      `transformer.h.{L}.mlp.c_fc.weight`
  * `transformer.ln_f` → 1 consumer global:
      `lm_head.weight`
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch


__all__ = [
    "fold_layernorm_for_quarot",
]


# Floor for γ magnitude when computing β/γ. Anything below this is clipped to
# avoid blow-up; trained LayerNorms shouldn't trigger this in practice.
_GAMMA_FLOOR = 1e-6


def _to_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _store(state_dict: dict, key: str, new_value: np.ndarray) -> None:
    """Replace `state_dict[key]` with `new_value` while preserving dtype."""
    old = state_dict[key]
    if hasattr(old, "dtype") and hasattr(old, "to"):
        state_dict[key] = torch.from_numpy(new_value).to(dtype=old.dtype)
    else:
        state_dict[key] = torch.from_numpy(new_value)


def _gamma_fold_consumer(
    state_dict: dict,
    consumer_weight_key: str,
    gamma: np.ndarray,
) -> bool:
    """Apply γ-fold to one consumer weight: `W_new = W * γ[None, :]`.

    Returns True if the key was present and folded, False otherwise.
    """
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


def _fold_one_layernorm(
    state_dict: dict,
    ln_weight_key: str,
    ln_bias_key: str,
    consumer_weight_keys: Sequence[str],
    modified: List[str],
) -> None:
    """Apply γ-fold to consumers + β-rescale (β ← β / γ) to LN itself.

    Sets `state_dict[ln_weight_key]` to ones (γ = 1) and rewrites
    `state_dict[ln_bias_key]` to `β / γ` (elementwise).
    """
    if ln_weight_key not in state_dict:
        return
    gamma = _to_f32(state_dict[ln_weight_key])
    has_bias = ln_bias_key in state_dict
    beta = _to_f32(state_dict[ln_bias_key]) if has_bias else None

    # γ-fold every consumer.
    for w_key in consumer_weight_keys:
        if _gamma_fold_consumer(state_dict, w_key, gamma):
            modified.append(w_key)

    # Set γ = ones (preserving dtype).
    ones_g = np.ones_like(gamma)
    _store(state_dict, ln_weight_key, ones_g)
    modified.append(ln_weight_key)

    # β-rescale: β_new = β / γ. Guard against near-zero γ.
    if beta is not None:
        gamma_abs = np.abs(gamma)
        gamma_safe = np.where(
            gamma_abs >= _GAMMA_FLOOR,
            gamma,
            np.where(gamma >= 0.0, _GAMMA_FLOOR, -_GAMMA_FLOOR).astype(gamma.dtype),
        ).astype(np.float32)
        beta_new = (beta / gamma_safe).astype(np.float32)
        _store(state_dict, ln_bias_key, beta_new)
        modified.append(ln_bias_key)


def fold_layernorm_for_quarot(state_dict: dict, model_args: dict) -> List[str]:
    """Fold every LayerNorm's `γ` into its consumers and rescale `β` to
    `β / γ`. Mutates `state_dict` in place.

    See module docstring for the mathematical justification. After this
    function returns:
      * Every `ln_*.weight` (γ) is `ones(d_model)`.
      * Every `ln_*.bias` (β) has been rescaled to `β_orig / γ_orig`.
      * Every consumer weight has had γ folded into its input columns.

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
        consumer_keys: List[str] = []
        H = 0
        while True:
            base = f"transformer.h.{L}.attn.c_attn.weight_h{H}"
            if f"{base}_query" not in state_dict:
                break
            for kind in ("query", "key", "value"):
                consumer_keys.append(f"{base}_{kind}")
            H += 1
        _fold_one_layernorm(
            state_dict,
            ln_weight_key=f"transformer.h.{L}.ln_1.weight",
            ln_bias_key=f"transformer.h.{L}.ln_1.bias",
            consumer_weight_keys=consumer_keys,
            modified=modified,
        )

    # Per-block LN_2 → c_fc.
    for L in range(n_layer):
        _fold_one_layernorm(
            state_dict,
            ln_weight_key=f"transformer.h.{L}.ln_2.weight",
            ln_bias_key=f"transformer.h.{L}.ln_2.bias",
            consumer_weight_keys=[f"transformer.h.{L}.mlp.c_fc.weight"],
            modified=modified,
        )

    # Global ln_f → lm_head.
    _fold_one_layernorm(
        state_dict,
        ln_weight_key="transformer.ln_f.weight",
        ln_bias_key="transformer.ln_f.bias",
        consumer_weight_keys=["lm_head.weight"],
        modified=modified,
    )

    return modified
