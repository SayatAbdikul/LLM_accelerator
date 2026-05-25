"""Disk cache for the per-(layer, source) Hessian + activation-mean vectors
used by ``apply_w4_awq_gptq``.

Background
----------
Inside ``w4_quant.apply_w4_awq_gptq``, after AWQ folds and FP32 activation
capture, we precompute one Hessian (``H = X^T X * 2/N``) and one gram
(``X^T X / N``) per (layer, source). The capture phase runs a full FP32
forward over ``n_seqs * seq_len`` tokens; the Hessian phase is a dense
``[d_in, d_in]`` matmul per source. On the Tier-1 65K-token corpus the two
together cost minutes — and are repeated every fresh process even though
they depend only on (fixture, calibration text, AWQ params, source-code
that-affects-the-capture).

Bias correction also needs the captured activations, but only through
``np.mean(X @ (W - W_dq).T, axis=0)``. By linearity that equals
``mean(X, axis=0) @ (W - W_dq).T`` — so the only per-source statistic
needed downstream is the per-input-channel mean ``x_mean[src]``, a tiny
vector. Caching ``(hessians, x_means)`` therefore lets the cache HIT
path skip activation capture entirely, including the per-source FP32
forward batch.

This module mirrors ``_prep_cache.py`` for that artifact. When a non-None
``hg_cache_path`` is passed to ``apply_w4_awq_gptq``, we look up an
``.npz`` keyed by the inputs that affect the matrices; on hit, the entire
capture + Hessian phase is skipped and only the downstream GPTQ +
AdaRound + bias-correction loop runs.

Cache key inputs
----------------
* **Fixture by path + mtime + size**: same cheap fingerprint as
  ``_prep_cache``. Editing the fixture file (touch alone is enough to
  affect mtime+size if size differs; size guards against false matches)
  invalidates.
* **Calibration text by path + mtime + size**: similarly cheap. The
  actual token list is a deterministic function of (text, tokenizer,
  ``n_seqs``, ``seq_len``); see ``calibration/scales.py``.
* **Capture + AWQ params**: ``n_seqs``, ``seq_len``, ``alpha``,
  ``awq_targets`` tuple. ``bitwidth`` is NOT included (bitwidth only
  affects GPTQ / AdaRound downstream, not the Hessian or activation mean).
* **Source-code SHA256** of files whose content affects the captured
  activations / Hessian value:
  - ``runtime/w4_quant.py`` — the capture loop and Hessian arithmetic
  - ``runtime/fake_quant_reference.py`` — ``_fp32_forward``
  - ``runtime/calibration/scales.py`` — ``build_calibration_seqs_from_token_ids``
  - ``runtime/calibration/adapters.py`` — ``apply_awq_from_token_ids``
  - ``quantizer/awq.py`` — the AWQ implementation
  - ``runtime/_hg_cache.py`` — this file, so a schema change here also
    invalidates without manual version bumps.

Note: editing the GPTQ / AdaRound / bias-correction sections of
``w4_quant.py`` will also invalidate the cache because the whole file
is hashed. If finer-grained invalidation matters, factor the
capture+Hessian phase into its own module — for now we accept the
coarse-grained invalidation because the W4 stack is stabilizing and the
common iteration pattern (downstream-of-Hessian param sweeps lives in
``quantize.py`` and elsewhere) is exactly what this cache speeds up.

Storage format
--------------
A single uncompressed ``.npz`` per cache key, with two array families:

  * ``H_<source_name>`` arrays, dtype float64, shape ``(d_in, d_in)``
  * ``X_<source_name>`` arrays, dtype float32, shape ``(d_in,)``

Schema versioning lives in the cache key (via ``_hg_cache.py``'s own
content hash in _SOURCE_FILES) — bump _hg_cache.py to invalidate.

Gram matrices are NOT stored; gram = hessian / 2.0 cast to float32 by
construction (``hessian = X^T X * 2/N``, ``gram = X^T X / N``). The
load helper derives ``grams`` on the fly to keep the on-disk footprint
manageable (Hessian alone is ~570 MB / model for the 12-layer GPT-2 124M
case; doubling to also store FP32 grams would push that to ~900 MB).

Uncompressed because Hessian matrices are dense float64 matmul outputs
that don't compress well — gzip typically halves at best, not worth the
CPU cost on a load-path that wants to mmap.

Atomicity: write to ``.tmp``, then ``os.replace``. A concurrent reader
sees either the old file (if any) or the new one, never a partial.

Limitations
-----------
* No retention policy — the user owns ``cache_dir`` cleanup.
* ``np.load`` is called with ``allow_pickle=False``; corrupt / schema-
  mismatched files are treated as misses and recomputed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# Files whose content affects the captured activations or Hessian
# computation. Add to this list if a new helper becomes load-bearing for
# the capture+Hessian phase of ``apply_w4_awq_gptq``.
_SOURCE_FILES = (
    "software/taccel/runtime/w4_quant.py",
    "software/taccel/runtime/fake_quant_reference.py",
    "software/taccel/runtime/calibration/scales.py",
    "software/taccel/runtime/calibration/adapters.py",
    "software/taccel/quantizer/awq.py",
    "software/taccel/runtime/_hg_cache.py",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "software").is_dir():
            return parent
    return Path.cwd()


def _hash_path_mtime_size(p: Optional[Path]) -> str:
    if p is None:
        return "none"
    p = Path(p)
    if not p.exists():
        return f"missing:{p}"
    st = p.stat()
    return f"{p}:{int(st.st_mtime)}:{st.st_size}"


def _hash_file_content(p: Path) -> str:
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_cache_key(
    fixture: Optional[Path],
    calibration_text: Optional[Path],
    *,
    n_seqs: int,
    seq_len: int,
    alpha: float,
    awq_targets: Sequence[str],
    repo_root: Optional[Path] = None,
) -> str:
    """Content-addressed cache key for the Hessian + activation-mean inputs."""
    rr = Path(repo_root) if repo_root is not None else _repo_root()
    h = hashlib.sha256()
    h.update(b"hg|v1|")  # cache schema version — bump on layout changes.
    h.update(f"fixture={_hash_path_mtime_size(fixture)}|".encode())
    h.update(f"calibration_text={_hash_path_mtime_size(calibration_text)}|".encode())
    h.update(f"n_seqs={int(n_seqs)}|".encode())
    h.update(f"seq_len={int(seq_len)}|".encode())
    h.update(f"alpha={float(alpha):.6f}|".encode())
    targets_sorted = ",".join(sorted(awq_targets))
    h.update(f"awq_targets={targets_sorted}|".encode())
    for relpath in _SOURCE_FILES:
        h.update(f"{relpath}={_hash_file_content(rr / relpath)}|".encode())
    return h.hexdigest()[:24]


def cache_path_for(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.npz"


def try_load(
    cache_path: Path,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """Load ``(hessians, grams, x_means)`` from disk, or None on miss/unreadable.

    Grams are derived on the fly: ``gram = hessian * 0.5`` cast to float32.

    The result is None unless at least one ``H_*`` and the matching ``X_*``
    arrays are present (a schema partial-write is a miss, not a half-success).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            keys = list(npz.files)
            hessians: Dict[str, np.ndarray] = {}
            x_means: Dict[str, np.ndarray] = {}
            for name in keys:
                if name.startswith("H_"):
                    hessians[name[len("H_"):]] = np.asarray(
                        npz[name], dtype=np.float64
                    )
                elif name.startswith("X_"):
                    x_means[name[len("X_"):]] = np.asarray(
                        npz[name], dtype=np.float32
                    )
            if not hessians:
                return None
            if set(hessians.keys()) != set(x_means.keys()):
                # Partial / schema-mismatch — recompute.
                return None
        grams: Dict[str, np.ndarray] = {
            n: (H * 0.5).astype(np.float32) for n, H in hessians.items()
        }
        return hessians, grams, x_means
    except (OSError, ValueError, EOFError, KeyError):
        return None


def save(
    hessians: Dict[str, np.ndarray],
    x_means: Dict[str, np.ndarray],
    cache_path: Path,
) -> None:
    """Atomically save ``(hessians, x_means)`` as a single ``.npz``."""
    if set(hessians.keys()) != set(x_means.keys()):
        raise ValueError(
            "hessians and x_means must have the same source-name keys; "
            f"got {sorted(hessians.keys())} vs {sorted(x_means.keys())}"
        )
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # `np.savez` auto-appends ".npz" if the path doesn't already end in it,
    # so the temp name keeps the .npz suffix to avoid a surprise double-extension.
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    payload: Dict[str, np.ndarray] = {}
    for name, H in hessians.items():
        payload[f"H_{name}"] = np.ascontiguousarray(H, dtype=np.float64)
    for name, x in x_means.items():
        payload[f"X_{name}"] = np.ascontiguousarray(x, dtype=np.float32)
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
