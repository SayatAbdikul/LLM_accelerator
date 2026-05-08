"""QuaRot Phase 1A self-tests: γ-fold, rotation primitives, and FP32 equivalence.

These tests gate every downstream phase. Each test isolates one mathematical
property of the QuaRot transformation:

    Test 1: build_random_orthogonal returns a 1-preserving orthogonal matrix.
    Test 2: γ-fold preserves FP32 forward (no rotation involved).
    Test 3: γ-fold + identity rotation is byte-identical to γ-fold alone.
    Test 4: γ-fold + random rotation preserves FP32 forward (master test).
    Test 5: rotated state_dict has expected per-key drift (sanity).
    Test 6: per-channel weight quant of rotated c_proj produces bounded QDQ
            reconstruction error.
    Test 7: idempotency — rotating by R then by R^T restores state_dict.
    Test 8: rotating ln.bias is required for correctness (control test).

The test fixture is the existing GPT-2 small checkpoint at
software/tests/fixtures/generated/gpt2_converted_nanogpt.pt — same fixture used
by the production gate. Each test is parametrized over a small token sequence
to avoid long FP32 forward runtimes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from taccel.quantizer.ln_fold import fold_layernorm_for_quarot
from taccel.quantizer.quantize import quantize_tensor
from taccel.quantizer.rotation import (
    build_random_orthogonal,
    rotate_residual_stream_state_dict,
)
from taccel.runtime.fake_quant_reference import _fp32_forward


FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")


def _load_payload():
    if not FIXTURE.exists():
        pytest.skip(f"GPT-2 fixture missing: {FIXTURE}")
    return torch.load(FIXTURE, map_location="cpu")


def _snapshot_state_dict(state_dict):
    return {
        k: (v.clone() if hasattr(v, "clone") else v)
        for k, v in state_dict.items()
    }


def _restore_state_dict(state_dict, snapshot):
    for k, v in snapshot.items():
        state_dict[k] = (v.clone() if hasattr(v, "clone") else v)


def _max_logit_drift(payload, token_seq):
    """Return max absolute drift on lm_head between current state_dict and a
    pristine snapshot run beforehand. Caller must have stashed the snapshot
    via `snapshot_logits` before mutating state_dict.

    This helper is the workhorse for FP32-equivalence tests.
    """
    fresh = _fp32_forward(payload["state_dict"], payload["model_args"], token_seq)
    return fresh["lm_head"]


# ---------------------------------------------------------------------------
# Test 1: 1-preserving orthogonal builder.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [32, 64, 256, 768])
@pytest.mark.parametrize("seed", [0xCAFE, 0xC0FFEE])
def test_build_random_orthogonal_is_1_preserving(d, seed):
    R = build_random_orthogonal(d, seed=seed)
    # Orthogonal: R^T R = I within FP32 tolerance.
    err_orth = float(np.abs(R @ R.T - np.eye(d, dtype=np.float32)).max())
    assert err_orth < 1e-4, (
        f"R@R^T deviates from I by {err_orth:.2e} (d={d}, seed={seed})"
    )
    # 1-preserving: R @ 1 = 1 (within FP32 tolerance).
    ones_d = np.ones(d, dtype=np.float32)
    R_one = R @ ones_d
    err_pres = float(np.abs(R_one - ones_d).max())
    assert err_pres < 1e-4, (
        f"R@1 deviates from 1 by {err_pres:.2e} (d={d}, seed={seed})"
    )
    # Determinism: same seed gives same R.
    R2 = build_random_orthogonal(d, seed=seed)
    assert np.array_equal(R, R2), f"build_random_orthogonal not deterministic"
    # Different seed gives different R.
    if seed != 0xC0FFEE:
        R3 = build_random_orthogonal(d, seed=seed + 1)
        assert not np.array_equal(R, R3), "different seeds gave identical R"


# ---------------------------------------------------------------------------
# Test 2: γ-fold alone preserves FP32 forward.
# ---------------------------------------------------------------------------


def test_gamma_fold_preserves_fp32_forward():
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    snapshot = _snapshot_state_dict(sd)

    # Reference forward (unrotated, unfolded).
    test_seq = list(range(16))
    ref_logits = _fp32_forward(sd, ma, test_seq)["lm_head"]

    # Apply γ-fold.
    modified = fold_layernorm_for_quarot(sd, ma)
    assert len(modified) > 0, "fold_layernorm_for_quarot reported no mutations"

    # Forward again: should match reference within FP32 tolerance.
    folded_logits = _fp32_forward(sd, ma, test_seq)["lm_head"]
    drift = float(np.abs(ref_logits - folded_logits).max())
    rel_drift = drift / max(float(np.abs(ref_logits).max()), 1e-8)
    assert rel_drift < 1e-4, (
        f"γ-fold broke FP32 equivalence: max drift {drift:.4e} "
        f"(rel {rel_drift:.4e})"
    )

    _restore_state_dict(sd, snapshot)


# ---------------------------------------------------------------------------
# Test 3: γ-fold + identity rotation matches γ-fold alone (byte equivalence).
# ---------------------------------------------------------------------------


def test_gamma_fold_plus_identity_rotation_is_equivalent():
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    d_model = int(ma["n_embd"])
    snapshot = _snapshot_state_dict(sd)

    # Apply γ-fold.
    fold_layernorm_for_quarot(sd, ma)
    folded_snapshot = _snapshot_state_dict(sd)

    # Apply identity rotation.
    I = np.eye(d_model, dtype=np.float32)
    rotate_residual_stream_state_dict(sd, ma, I)

    # Compare to folded_snapshot per key.
    max_drift = 0.0
    drift_keys = []
    for k in folded_snapshot:
        a = sd[k].detach().cpu().numpy().astype(np.float32) if hasattr(sd[k], "detach") else np.asarray(sd[k], dtype=np.float32)
        b = folded_snapshot[k].detach().cpu().numpy().astype(np.float32) if hasattr(folded_snapshot[k], "detach") else np.asarray(folded_snapshot[k], dtype=np.float32)
        if a.shape != b.shape:
            pytest.fail(f"shape mismatch on key {k!r}: {a.shape} vs {b.shape}")
        d_max = float(np.abs(a - b).max())
        if d_max > max_drift:
            max_drift = d_max
            drift_keys.append((k, d_max))
    # Identity rotation should produce ~zero drift modulo FP32 epsilon from
    # compound matrix multiplications.
    assert max_drift < 1e-4, (
        f"identity rotation drifted state_dict by {max_drift:.4e}; "
        f"top offenders: {drift_keys[-3:]}"
    )

    _restore_state_dict(sd, snapshot)


# ---------------------------------------------------------------------------
# Test 4: γ-fold + random rotation preserves FP32 forward (master test).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0xCAFE, 0xC0FFEE, 42])
def test_full_quarot_transformation_preserves_fp32_forward(seed):
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    d_model = int(ma["n_embd"])
    snapshot = _snapshot_state_dict(sd)

    # Reference forward.
    test_seq = list(range(16))
    ref_logits = _fp32_forward(sd, ma, test_seq)["lm_head"]

    # Apply γ-fold + rotation.
    fold_layernorm_for_quarot(sd, ma)
    R = build_random_orthogonal(d_model, seed=seed)
    rotate_residual_stream_state_dict(sd, ma, R)

    # Forward again: should match reference within FP32 tolerance.
    rotated_logits = _fp32_forward(sd, ma, test_seq)["lm_head"]
    drift = float(np.abs(ref_logits - rotated_logits).max())
    rel_drift = drift / max(float(np.abs(ref_logits).max()), 1e-8)
    assert rel_drift < 1e-3, (
        f"γ-fold + random rotation broke FP32 equivalence: max drift "
        f"{drift:.4e} (rel {rel_drift:.4e}); seed={seed:#x}"
    )

    _restore_state_dict(sd, snapshot)


# ---------------------------------------------------------------------------
# Test 5: state_dict drift sanity check (rotated state should differ from
# original by a measurable amount).
# ---------------------------------------------------------------------------


def test_rotation_changes_state_dict():
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    d_model = int(ma["n_embd"])
    snapshot = _snapshot_state_dict(sd)

    fold_layernorm_for_quarot(sd, ma)
    folded_snapshot = _snapshot_state_dict(sd)
    R = build_random_orthogonal(d_model, seed=0xCAFE)
    modified = rotate_residual_stream_state_dict(sd, ma, R)

    # Sanity: at least one key in `modified` actually changed substantially.
    big_drift = 0.0
    for k in modified:
        a = sd[k].detach().cpu().numpy().astype(np.float32) if hasattr(sd[k], "detach") else np.asarray(sd[k], dtype=np.float32)
        b = folded_snapshot[k].detach().cpu().numpy().astype(np.float32) if hasattr(folded_snapshot[k], "detach") else np.asarray(folded_snapshot[k], dtype=np.float32)
        d_max = float(np.abs(a - b).max())
        big_drift = max(big_drift, d_max)
    assert big_drift > 0.01, (
        f"rotation barely changed state_dict (max drift {big_drift:.4e}); "
        "may indicate R is too close to identity"
    )

    _restore_state_dict(sd, snapshot)


# ---------------------------------------------------------------------------
# Test 6: per-channel weight quant on rotated c_proj has bounded
# reconstruction error.
# ---------------------------------------------------------------------------


def test_per_channel_weight_quant_reconstruction_after_rotation():
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    d_model = int(ma["n_embd"])
    snapshot = _snapshot_state_dict(sd)

    # Rotate and check that per-channel quant on a representative c_proj
    # weight produces reasonable reconstruction error.
    fold_layernorm_for_quarot(sd, ma)
    R = build_random_orthogonal(d_model, seed=0xCAFE)
    rotate_residual_stream_state_dict(sd, ma, R)

    target_keys = [
        "transformer.h.2.mlp.c_proj.weight",   # the worst block from diagnostic
        "transformer.h.11.attn.c_proj.weight", # last-block attn
    ]
    for key in target_keys:
        if key not in sd:
            continue
        w = sd[key].detach().cpu().numpy().astype(np.float32) if hasattr(sd[key], "detach") else np.asarray(sd[key], dtype=np.float32)
        q, scales = quantize_tensor(w, per_channel=True)
        dq = q.astype(np.float32) * scales.astype(np.float32).reshape(-1, 1)
        # Per-channel reconstruction error should be bounded.
        err_per_row_max = np.abs(w - dq).max(axis=1)  # [d_out]
        scales_f32 = scales.astype(np.float32)
        # Worst-case: floor-rounding error per row is at most scales[k] / 2.
        # Allow a generous slack of 1× scale (in case some rows have weird
        # statistics post-rotation).
        for k in range(w.shape[0]):
            assert err_per_row_max[k] <= scales_f32[k] * 1.0 + 1e-6, (
                f"per-channel QDQ reconstruction error on row {k} of {key}: "
                f"max abs err {err_per_row_max[k]:.4e} > scale {scales_f32[k]:.4e}"
            )

    _restore_state_dict(sd, snapshot)


# ---------------------------------------------------------------------------
# Test 7: rotating by R then by R^T restores γ-folded state_dict.
# ---------------------------------------------------------------------------


def test_rotation_idempotency_with_inverse():
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    d_model = int(ma["n_embd"])
    snapshot = _snapshot_state_dict(sd)

    fold_layernorm_for_quarot(sd, ma)
    folded_snapshot = _snapshot_state_dict(sd)

    R = build_random_orthogonal(d_model, seed=0xC0FFEE)
    rotate_residual_stream_state_dict(sd, ma, R)
    # Rotating by R^T should undo the rotation.
    rotate_residual_stream_state_dict(sd, ma, R.T)

    # Compare to folded_snapshot.
    max_drift = 0.0
    for k in folded_snapshot:
        a = sd[k].detach().cpu().numpy().astype(np.float32) if hasattr(sd[k], "detach") else np.asarray(sd[k], dtype=np.float32)
        b = folded_snapshot[k].detach().cpu().numpy().astype(np.float32) if hasattr(folded_snapshot[k], "detach") else np.asarray(folded_snapshot[k], dtype=np.float32)
        d_max = float(np.abs(a - b).max())
        max_drift = max(max_drift, d_max)
    # Two orthogonal multiplications: float epsilon accumulates.
    assert max_drift < 5e-3, (
        f"R then R^T failed to restore state_dict: max drift {max_drift:.4e}"
    )

    _restore_state_dict(sd, snapshot)


# ---------------------------------------------------------------------------
# Test 8: control test — rotation that SKIPS LN bias breaks FP32 equivalence.
# Demonstrates that β-rotation is required.
# ---------------------------------------------------------------------------


def test_skipping_ln_bias_rotation_breaks_fp32_equivalence():
    payload = _load_payload()
    sd = payload["state_dict"]
    ma = payload["model_args"]
    d_model = int(ma["n_embd"])
    n_layer = int(ma["n_layer"])
    snapshot = _snapshot_state_dict(sd)

    test_seq = list(range(16))
    ref_logits = _fp32_forward(sd, ma, test_seq)["lm_head"]

    # γ-fold + rotation, but ZERO OUT the LN bias rotation manually to verify
    # it would have broken correctness.
    fold_layernorm_for_quarot(sd, ma)
    R = build_random_orthogonal(d_model, seed=0xCAFE)
    rotate_residual_stream_state_dict(sd, ma, R)

    # Now restore LN biases to their pre-rotation value (i.e., undo β-rotation).
    # That is, we revert ln.bias values to their (unrotated) state from the
    # γ-folded snapshot. This simulates what the old β-fold-only design would
    # have done.
    folded_snapshot = {}
    _restore_state_dict(sd, snapshot)
    fold_layernorm_for_quarot(sd, ma)
    for k in sd:
        if k.endswith(".bias") and (
            ".ln_1." in k or ".ln_2." in k or k == "transformer.ln_f.bias"
        ):
            folded_snapshot[k] = sd[k].clone() if hasattr(sd[k], "clone") else sd[k]
    # Now apply rotation, then revert ln biases to unrotated.
    rotate_residual_stream_state_dict(sd, ma, R)
    for k, v in folded_snapshot.items():
        sd[k] = v.clone() if hasattr(v, "clone") else v

    broken_logits = _fp32_forward(sd, ma, test_seq)["lm_head"]
    drift = float(np.abs(ref_logits - broken_logits).max())
    rel_drift = drift / max(float(np.abs(ref_logits).max()), 1e-8)
    # If LN bias is NOT rotated, drift should be substantial (>> 1e-3) —
    # demonstrating that β-rotation is necessary.
    assert rel_drift > 0.01, (
        f"control test failed: skipping LN bias rotation should break FP32 "
        f"equivalence, but drift was only {rel_drift:.4e}; "
        "β-rotation may not actually be necessary?"
    )

    _restore_state_dict(sd, snapshot)
