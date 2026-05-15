"""Level-1 (per-vector) verification of the TurboQuant KV quantizer.

Validates the algorithm *in isolation* before it is wired into the model, so
a quantizer bug can't masquerade as a model-accuracy finding. Three checks,
all grounded in arXiv 2504.19874:

  A. Reconstruction error decreases monotonically with bitwidth and is of the
     expected magnitude (≈ Lloyd–Max Gaussian distortion after the rotation).
  B. Self-inner-product ratio  R = E⟨x, x̂⟩ / E‖x‖²:
       - MSE variant is biased LOW by ≈ the distortion (R = 1 − D_b < 1),
         the deficit shrinking as bits↑  — the paper's known MSE bias.
       - IP variant is ≈ unbiased (R ≈ 1) and strictly closer to 1 than the
         MSE variant at the same bitwidth — the paper's whole point.
     Reproducing BOTH is the implementation-correctness gate.
  C. Determinism, shape/dtype, zero-vector handling.
"""
from __future__ import annotations

import numpy as np
import pytest

from taccel.quantizer.turboquant import TurboQuantKV

D = 64          # GPT-2 124M d_head (n_embd 768 / n_head 12)
N = 4000        # enough samples for a tight bias estimate
SEED = 20260515


def _vectors(n=N, d=D, seed=SEED):
    # Anisotropic Gaussian — closer to real K/V than isotropic, and a
    # non-trivial test of the data-oblivious rotation.
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.5, 3.0, size=d)
    return (rng.standard_normal((n, d)) * scale).astype(np.float32)


def _recon_rel_l2(x, xhat):
    num = np.linalg.norm(x - xhat, axis=-1)
    den = np.linalg.norm(x, axis=-1) + 1e-12
    return float(np.mean(num / den))


def _self_ip_ratio(x, xhat):
    # E⟨x, x̂⟩ / E‖x‖²
    return float(np.sum(x * xhat) / np.sum(x * x))


def test_recon_error_monotone_in_bits():
    x = _vectors()
    errs = {}
    for b in (2.0, 3.0, 4.0):
        tq = TurboQuantKV(d=D, bits=b, variant="mse", seed=SEED)
        errs[b] = _recon_rel_l2(x, tq.round_trip(x))
    assert errs[4.0] < errs[3.0] < errs[2.0], errs
    # Lloyd–Max Gaussian: rel-L2 ≈ sqrt(D_b); D_4≈0.0095 → ~0.10. Generous.
    assert errs[4.0] < 0.16, errs
    assert errs[2.0] < 0.55, errs


def test_mse_variant_self_inner_product_is_biased_low():
    """Paper property: the MSE-optimal quantizer shrinks ⟨x,x̂⟩ by ≈ the
    distortion (orthogonality principle ⇒ R = 1 − D_b). Deficit must be
    clearly positive and shrink as bits increase."""
    x = _vectors()
    deficits = {}
    for b in (2.0, 3.0, 4.0):
        tq = TurboQuantKV(d=D, bits=b, variant="mse", seed=SEED)
        deficits[b] = 1.0 - _self_ip_ratio(x, tq.round_trip(x))
    assert deficits[2.0] > 0.03, deficits          # clearly biased low
    assert deficits[4.0] < deficits[3.0] < deficits[2.0], deficits


def test_ip_variant_removes_the_self_inner_product_bias():
    """Paper's central claim: the inner-product variant is ~unbiased and
    strictly closer to 1 than the MSE variant at the same bitwidth."""
    x = _vectors()
    for b in (3.0, 4.0):
        mse = TurboQuantKV(d=D, bits=b, variant="mse", seed=SEED)
        ip = TurboQuantKV(d=D, bits=b, variant="ip", seed=SEED)
        r_mse = _self_ip_ratio(x, mse.round_trip(x))
        r_ip = _self_ip_ratio(x, ip.round_trip(x))
        assert abs(1.0 - r_ip) < abs(1.0 - r_mse), (b, r_mse, r_ip)
        assert abs(1.0 - r_ip) < 0.05, (b, r_ip)   # ~unbiased


def test_apply_rotation_false_disables_pi():
    """The QuaRot-redundancy ablation toggle must actually bypass Π
    (Π == I) so the sweep can isolate the two rotations."""
    tq_off = TurboQuantKV(d=D, bits=3.0, apply_rotation=False, seed=SEED)
    assert np.allclose(tq_off._pi(), np.eye(D))
    tq_on = TurboQuantKV(d=D, bits=3.0, apply_rotation=True, seed=SEED)
    assert not np.allclose(tq_on._pi(), np.eye(D))


def test_determinism_shape_dtype_and_zero_vector():
    x = _vectors(n=128)
    tq = TurboQuantKV(d=D, bits=3.0, variant="ip", seed=SEED)
    a, b = tq.round_trip(x), TurboQuantKV(d=D, bits=3.0, variant="ip",
                                          seed=SEED).round_trip(x)
    assert np.array_equal(a, b)                     # seeded → deterministic
    assert a.shape == x.shape and a.dtype == np.float32
    z = np.zeros((4, D), dtype=np.float32)
    assert np.array_equal(tq.round_trip(z), z)      # ‖x‖=0 → 0, no NaN


@pytest.mark.parametrize("bits", [2.5, 3.5])
def test_fractional_bits_between_integer_neighbors(bits):
    """Mixed-precision fractional rate must land between its integer
    neighbors in reconstruction error."""
    x = _vectors()
    lo = TurboQuantKV(d=D, bits=float(int(bits)), seed=SEED)
    hi = TurboQuantKV(d=D, bits=float(int(bits) + 1), seed=SEED)
    mid = TurboQuantKV(d=D, bits=bits, seed=SEED)
    e_lo = _recon_rel_l2(x, lo.round_trip(x))
    e_hi = _recon_rel_l2(x, hi.round_trip(x))
    e_mid = _recon_rel_l2(x, mid.round_trip(x))
    assert e_hi <= e_mid <= e_lo, (e_lo, e_mid, e_hi)
