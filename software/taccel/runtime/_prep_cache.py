"""Disk cache for `_turboquant_eval.prepare()`.

Calibration runs can take 5–25 min on the W4 stack (AWQ + FP32 activation
capture + per-source Hessian/gram precompute + GPTQ act-order + AdaRound +
bias correction, plus optional Stage5 QKT/attn_v scale fit). This module
caches the full `Prepared` dataclass on disk keyed by all inputs that
affect the result, so re-running with identical settings unpickles in
~1 second instead of re-running the whole pipeline.

Cache key inputs
----------------
The 24-char SHA256-prefix key encodes:
  * **Fixtures by path + mtime + size** (cheap fingerprint): the GPT-2
    checkpoint, the eval-text file, and the calibration-text file. We do
    NOT content-hash the GPT-2 checkpoint (>500 MB; opening it costs more
    than re-running calibration on a cache hit) — the mtime+size combo is
    sufficient for the build artifacts we ship.
  * **Tokenizer dir path**: change here means a different tokenizer.
  * **Numeric prep args**: `max_tokens`, `calibration_n_seqs`,
    `calibration_seq_len`, `calibration_percentile`.
  * **Preset name**: changing the preset usually changes the algorithm.
  * **Source-code SHA256 of the files that govern calibration semantics**
    (full content hash): `w4_quant.py`, `quantize.py`,
    `_turboquant_eval.py`, `calibration/scales.py`,
    `calibration/adapters.py`, `stage5_ptq.py`. Editing any of these
    invalidates the cache automatically (the user does not have to bump
    a version manually).

The cache file is a gzip-compressed pickle at
`{cache_dir}/{key}.pkl.gz`. Atomicity is via temp-file rename.

Limitations
-----------
* Pickling a `Prepared` containing 500 MB of FP32 weights produces a
  ~200–300 MB gzipped file. For workflows with many distinct prep
  settings, consider `cache_dir` cleanup (or a smaller cache scope —
  TBD if this becomes a real concern).
* Unpickling depends on the running torch / numpy versions. A version
  mismatch yields a `pickle.UnpicklingError` (or similar), which we
  treat as a cache miss and recompute.
* The cache survives across git checkouts because source-file hashing
  detects content changes; mtime games don't fool us.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Optional


# Source files whose content affects calibration / prep semantics. If any
# of these change, the cache must invalidate. Listed relative to the
# project root (see `_repo_root` below). Add to this list if a new helper
# becomes load-bearing for `prepare()`.
_SOURCE_FILES = (
    "software/taccel/runtime/w4_quant.py",
    "software/taccel/runtime/_turboquant_eval.py",
    "software/taccel/runtime/stage5_ptq.py",
    "software/taccel/runtime/calibration/scales.py",
    "software/taccel/runtime/calibration/adapters.py",
    "software/taccel/quantizer/quantize.py",
    "software/taccel/quantizer/awq.py",
    "software/taccel/quantizer/rotation.py",
    "software/taccel/quantizer/ln_fold.py",
)


def _repo_root() -> Path:
    """Best-effort: walk up from this file until a `software/` dir is seen."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "software").is_dir():
            return parent
    # Fallback: cwd.
    return Path.cwd()


def _hash_path_mtime_size(p: Optional[Path]) -> str:
    """Cheap fingerprint via path + mtime + size. None / missing → sentinel."""
    if p is None:
        return "none"
    p = Path(p)
    if not p.exists():
        return f"missing:{p}"
    st = p.stat()
    return f"{p}:{int(st.st_mtime)}:{st.st_size}"


def _hash_file_content(p: Path) -> str:
    """Truncated SHA256 of file content. Used for source files where mtime
    is unreliable (git checkouts touch mtime without content change)."""
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_cache_key(
    fixture: Path,
    tokenizer_dir: Path,
    eval_text: Path,
    *,
    max_tokens: int,
    ptq_preset: str,
    calibration_text: Optional[Path],
    calibration_n_seqs: int,
    calibration_seq_len: int,
    calibration_percentile: float,
    repo_root: Optional[Path] = None,
) -> str:
    """Content-addressed cache key for the `prepare()` inputs.

    The same args produce the same key; any meaningful change (fixture
    contents via mtime+size, tokenizer path, eval/calibration text,
    numeric args, preset, OR source code) invalidates.
    """
    rr = Path(repo_root) if repo_root is not None else _repo_root()
    h = hashlib.sha256()
    h.update(b"v1|")  # cache schema version — bump on Prepared layout changes.
    h.update(f"fixture={_hash_path_mtime_size(Path(fixture))}|".encode())
    h.update(f"tokenizer_dir={Path(tokenizer_dir)}|".encode())
    h.update(f"eval_text={_hash_path_mtime_size(Path(eval_text))}|".encode())
    h.update(f"max_tokens={int(max_tokens)}|".encode())
    h.update(f"ptq_preset={ptq_preset}|".encode())
    ct = Path(calibration_text) if calibration_text is not None else None
    h.update(f"calibration_text={_hash_path_mtime_size(ct)}|".encode())
    h.update(f"calibration_n_seqs={int(calibration_n_seqs)}|".encode())
    h.update(f"calibration_seq_len={int(calibration_seq_len)}|".encode())
    # Round to a fixed precision so e.g. 99.9 vs 99.900000000001 don't
    # produce different keys due to user-side fp parsing.
    h.update(f"calibration_percentile={float(calibration_percentile):.6f}|".encode())
    for relpath in _SOURCE_FILES:
        src = rr / relpath
        h.update(f"{relpath}={_hash_file_content(src)}|".encode())
    return h.hexdigest()[:24]


def cache_path_for(cache_dir: Path, key: str) -> Path:
    """Standardized path layout: `{cache_dir}/{key}.pkl.gz`."""
    return Path(cache_dir) / f"{key}.pkl.gz"


def try_load(cache_path: Path) -> Optional[Any]:
    """Return the cached object, or None on miss / unreadable / version-skew."""
    if not Path(cache_path).exists():
        return None
    try:
        with gzip.open(cache_path, "rb") as f:
            return pickle.load(f)
    except (
        pickle.UnpicklingError,
        EOFError,
        OSError,
        ImportError,
        AttributeError,
        ModuleNotFoundError,
    ):
        # Treat unreadable / version-skewed caches as misses; we'll
        # recompute and overwrite. We do NOT delete here — a concurrent
        # writer might be mid-flight.
        return None


def save(obj: Any, cache_path: Path) -> None:
    """Pickle+gzip `obj` atomically (write to .tmp, then `os.replace`).

    The atomic rename ensures a concurrent reader never sees a partial
    file; either the old cache (if any) is in place, or the new one is.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
