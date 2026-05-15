"""TurboQuant (arXiv 2504.19874, ICLR 2026) data-oblivious KV-cache quantizer.

Tier-1 *reference model*: a numeric quant→dequant round-trip used to measure
the accuracy drop of KV-cache quantization. There is no physical bit-packing
here — the accuracy question (what we verify) is the numeric round-trip;
packed-DRAM layout / actual memory savings is a separate, out-of-scope concern
(see the plan).

Algorithm (per vector x ∈ ℝ^d, d = d_head), data-oblivious, seeded shared
state only — no calibration:

  Shared, seeded:
    Π  = Haar-random orthogonal d×d (sign-corrected QR of a Gaussian).
    S  = Gaussian d×d (QJL projection, inner-product variant only).
    c^(b) = Lloyd–Max optimal centroids for N(0,1), b ∈ {1..4} bits.

  MSE encode:  y = Π x ; z = y·√d/‖x‖₂ ; idx_j = argmin_k |z_j − c^(b)_k|
               store (idx, ‖x‖₂)
  MSE decode:  ŷ_j = c^(b)_{idx_j}·‖x‖₂/√d ; x̂ = Πᵀ ŷ

  Inner-product (unbiased) variant, total b bits:
    idx = MSE_encode(x, b−1) ; r = x − MSE_decode(idx)
    qjl = sign(S r)                                  (1 bit / coord)
    store (idx, qjl, ‖r‖₂)
    x̂ = MSE_decode(idx) + (√(π/2)/d)·‖r‖₂·Sᵀ qjl

The MSE quantizer is MSE-optimal but its inner-product estimate is biased by
~2/π at b=1 (paper); the inner-product variant removes that bias. Level-1
verification reproduces *both* facts as an implementation-correctness check.

Rotation note: we use a plain Haar orthogonal here, NOT
`rotation.build_random_orthogonal` (which is 1-preserving for LayerNorm
commutativity in QuaRot — irrelevant and slightly less mixing for a per-vector
cache-boundary rotation). `apply_rotation=False` disables Π entirely so the
verification sweep can test whether QuaRot's residual-stream rotation already
makes Π redundant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

_EPS = 1e-12


def _haar_orthogonal(d: int, seed: int) -> np.ndarray:
    """Sign-corrected QR of a Gaussian → Haar-distributed d×d orthogonal."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, d)).astype(np.float64)
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diag(r))[np.newaxis, :]
    return q.astype(np.float64)


def _lloyd_max_centroids(n_levels: int, *, iters: int = 64) -> np.ndarray:
    """Lloyd–Max optimal centroids for the standard normal N(0,1).

    Deterministic fixed-point iteration on a fine grid (no RNG, no scipy):
    cell boundaries = midpoints of adjacent levels; each level = the
    density-weighted centroid (conditional mean) of its cell.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    if n_levels == 1:
        return np.array([0.0])
    grid = np.linspace(-8.0, 8.0, 16384)
    pdf = np.exp(-0.5 * grid**2)
    pdf /= pdf.sum()
    # Init at equiprobable quantile midpoints of N(0,1).
    cdf = np.cumsum(pdf)
    qs = (np.arange(n_levels) + 0.5) / n_levels
    levels = grid[np.searchsorted(cdf, qs).clip(0, grid.size - 1)].astype(np.float64)
    for _ in range(iters):
        bounds = (levels[:-1] + levels[1:]) / 2.0
        edges = np.concatenate(([-np.inf], bounds, [np.inf]))
        new = levels.copy()
        for i in range(n_levels):
            m = (grid >= edges[i]) & (grid < edges[i + 1])
            w = pdf[m].sum()
            if w > _EPS:
                new[i] = (grid[m] * pdf[m]).sum() / w
        if np.allclose(new, levels, atol=1e-10):
            levels = new
            break
        levels = new
    return levels


@dataclass(frozen=True)
class TurboQuantKV:
    """Config + (lazily-built, cached) shared state for a fixed dim `d`."""

    d: int
    bits: float = 3.0
    variant: Literal["mse", "ip"] = "mse"
    target: Literal["k", "v", "kv"] = "kv"  # which of K/V to quantize
    apply_rotation: bool = True
    seed: int = 0xCAFE
    # cached shared state, keyed by the (d, seed) of this instance
    _state: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def quant_k(self) -> bool:
        return "k" in self.target

    @property
    def quant_v(self) -> bool:
        return "v" in self.target

    # -- shared state -----------------------------------------------------
    def _pi(self) -> np.ndarray:
        if "pi" not in self._state:
            self._state["pi"] = (
                _haar_orthogonal(self.d, self.seed)
                if self.apply_rotation
                else np.eye(self.d, dtype=np.float64)
            )
        return self._state["pi"]

    def _qjl(self) -> np.ndarray:
        if "qjl" not in self._state:
            self._state["qjl"] = np.random.default_rng(
                self.seed ^ 0x51_4A_4C
            ).standard_normal((self.d, self.d)).astype(np.float64)
        return self._state["qjl"]

    def _centroids(self, b: int) -> np.ndarray:
        key = f"c{b}"
        if key not in self._state:
            self._state[key] = _lloyd_max_centroids(2**b) if b >= 1 else np.zeros(1)
        return self._state[key]

    def _bit_split(self, total_bits: float) -> np.ndarray:
        """Per-coordinate integer bit allocation averaging to `total_bits`
        (mixed precision for fractional rates, e.g. 2.5 → mix of 2 and 3)."""
        lo = int(np.floor(total_bits))
        hi = lo + 1
        frac = float(total_bits) - lo
        n_hi = int(round(frac * self.d))
        alloc = np.full(self.d, lo, dtype=np.int64)
        alloc[:n_hi] = hi
        return alloc

    # -- MSE quantizer ----------------------------------------------------
    def _mse_round_trip(self, x: np.ndarray, total_bits: float) -> np.ndarray:
        """decode(encode(x)) for the MSE variant. x: (..., d) → (..., d)."""
        pi = self._pi()
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        safe = np.maximum(norm, _EPS)
        y = x @ pi.T
        z = y * np.sqrt(self.d) / safe  # standardized, coords ≈ N(0,1)
        alloc = self._bit_split(total_bits)
        zhat = np.empty_like(z)
        for b in np.unique(alloc):
            cols = alloc == b
            c = self._centroids(int(b))
            if c.size == 1:
                zhat[..., cols] = c[0]
            else:
                # nearest centroid per coordinate
                idx = np.abs(z[..., cols, None] - c).argmin(axis=-1)
                zhat[..., cols] = c[idx]
        yhat = zhat * safe / np.sqrt(self.d)
        xhat = yhat @ pi  # Πᵀ  (pi orthogonal: pi.T applied on encode, pi here)
        return np.where(norm > _EPS, xhat, 0.0)

    # -- public round-trip -----------------------------------------------
    def round_trip(self, x: np.ndarray) -> np.ndarray:
        """Lossy decode(encode(x)) used to model the quantized KV cache.

        MSE variant: reconstruction. IP variant: MSE(b−1) reconstruction +
        QJL residual correction — the unbiased inner-product estimator's
        reconstruction (Level-1 validates its unbiasedness separately).
        """
        x = np.asarray(x, dtype=np.float64)
        if self.variant == "mse":
            return self._mse_round_trip(x, self.bits).astype(np.float32)

        # inner-product (unbiased) variant
        x_mse = self._mse_round_trip(x, max(self.bits - 1.0, 1.0))
        r = x - x_mse
        rnorm = np.linalg.norm(r, axis=-1, keepdims=True)
        s = self._qjl()
        qjl = np.sign(r @ s.T)
        qjl[qjl == 0] = 1.0
        correction = (np.sqrt(np.pi / 2.0) / self.d) * rnorm * (qjl @ s)
        return (x_mse + correction).astype(np.float32)

    def __repr__(self) -> str:  # concise, state-free
        return (
            f"TurboQuantKV(d={self.d}, bits={self.bits}, variant={self.variant!r}, "
            f"apply_rotation={self.apply_rotation}, seed={hex(self.seed)})"
        )
