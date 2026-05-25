"""Unit tests for ``software.taccel.runtime._hg_cache``.

The cache stores per-(layer, source) Hessian (float64) and activation-mean
(float32) tensors on disk for ``apply_w4_awq_gptq``. These tests cover the
save/load round trip, key-derivation behavior, and invalidation triggers
without exercising the full W4 pipeline (which has its own preset gate
in `test_stage5_ptq_presets.py` for byte-identity).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime import _hg_cache as hg


def _toy_pair(d_in: int = 8, seed: int = 0):
    """Build a small (hessians, x_means) pair with two source names."""
    rng = np.random.default_rng(seed)
    H1 = rng.standard_normal((d_in, d_in)).astype(np.float64)
    H1 = (H1 + H1.T)  # symmetric, like X.T @ X
    H2 = rng.standard_normal((d_in, d_in)).astype(np.float64)
    H2 = (H2 + H2.T)
    x1 = rng.standard_normal(d_in).astype(np.float32)
    x2 = rng.standard_normal(d_in).astype(np.float32)
    return (
        {"block0_ln1": H1, "block0_concat": H2},
        {"block0_ln1": x1, "block0_concat": x2},
    )


def test_save_load_roundtrip(tmp_path: Path):
    hessians, x_means = _toy_pair()
    cache_path = tmp_path / "k.npz"
    hg.save(hessians, x_means, cache_path)

    loaded = hg.try_load(cache_path)
    assert loaded is not None
    H_loaded, G_loaded, X_loaded = loaded
    assert set(H_loaded) == set(hessians)
    for name in hessians:
        np.testing.assert_array_equal(H_loaded[name], hessians[name])
        np.testing.assert_array_equal(X_loaded[name], x_means[name])
        # Gram is derived: gram = hessian / 2.0 in float32.
        np.testing.assert_array_equal(
            G_loaded[name], (hessians[name] * 0.5).astype(np.float32)
        )


def test_try_load_returns_none_for_missing_file(tmp_path: Path):
    assert hg.try_load(tmp_path / "nope.npz") is None


def test_try_load_returns_none_for_corrupt_file(tmp_path: Path):
    p = tmp_path / "corrupt.npz"
    p.write_bytes(b"this is not a real .npz")
    assert hg.try_load(p) is None


def test_save_rejects_mismatched_keys(tmp_path: Path):
    hessians = {"block0_ln1": np.zeros((4, 4))}
    x_means = {"block1_ln2": np.zeros(4, dtype=np.float32)}
    with pytest.raises(ValueError):
        hg.save(hessians, x_means, tmp_path / "x.npz")


def test_save_atomic_no_tmp_leftover(tmp_path: Path):
    hessians, x_means = _toy_pair()
    cache_path = tmp_path / "k.npz"
    hg.save(hessians, x_means, cache_path)
    assert cache_path.exists()
    # The .tmp must not be visible after a clean save.
    assert not (cache_path.with_suffix(".npz.tmp")).exists()


def test_cache_key_stable_for_same_inputs(tmp_path: Path):
    fixture = tmp_path / "fixture.bin"
    calib = tmp_path / "calib.txt"
    fixture.write_bytes(b"\x00" * 1024)
    calib.write_text("hello world\n")

    k1 = hg.compute_cache_key(
        fixture, calib,
        n_seqs=128, seq_len=512, alpha=0.40,
        awq_targets=("c_attn", "c_fc", "lm_head"),
    )
    k2 = hg.compute_cache_key(
        fixture, calib,
        n_seqs=128, seq_len=512, alpha=0.40,
        awq_targets=("c_attn", "c_fc", "lm_head"),
    )
    assert k1 == k2


def test_cache_key_invalidates_on_fixture_size_change(tmp_path: Path):
    fixture = tmp_path / "fixture.bin"
    calib = tmp_path / "calib.txt"
    calib.write_text("hello\n")

    fixture.write_bytes(b"\x00" * 1024)
    k_small = hg.compute_cache_key(
        fixture, calib, n_seqs=8, seq_len=8, alpha=0.4,
        awq_targets=("c_attn",),
    )
    fixture.write_bytes(b"\x00" * 2048)
    k_big = hg.compute_cache_key(
        fixture, calib, n_seqs=8, seq_len=8, alpha=0.4,
        awq_targets=("c_attn",),
    )
    assert k_small != k_big


def test_cache_key_invalidates_on_calibration_text_change(tmp_path: Path):
    fixture = tmp_path / "fixture.bin"
    calib_a = tmp_path / "a.txt"
    calib_b = tmp_path / "b.txt"
    fixture.write_bytes(b"\x00" * 16)
    calib_a.write_text("aaaa")
    calib_b.write_text("bbbbbbbb")  # different size → different mtime+size key

    ka = hg.compute_cache_key(fixture, calib_a, n_seqs=4, seq_len=4, alpha=0.4,
                              awq_targets=("c_attn",))
    kb = hg.compute_cache_key(fixture, calib_b, n_seqs=4, seq_len=4, alpha=0.4,
                              awq_targets=("c_attn",))
    assert ka != kb


@pytest.mark.parametrize(
    "kwargs_a, kwargs_b",
    [
        ({"n_seqs": 8}, {"n_seqs": 16}),
        ({"seq_len": 64}, {"seq_len": 128}),
        ({"alpha": 0.40}, {"alpha": 0.50}),
        ({"awq_targets": ("c_attn",)}, {"awq_targets": ("c_attn", "c_fc")}),
    ],
)
def test_cache_key_invalidates_on_numeric_args(
    tmp_path: Path, kwargs_a: dict, kwargs_b: dict
):
    fixture = tmp_path / "f.bin"
    calib = tmp_path / "c.txt"
    fixture.write_bytes(b"\x00" * 8)
    calib.write_text("x")

    base = dict(n_seqs=8, seq_len=64, alpha=0.40, awq_targets=("c_attn",))
    a = {**base, **kwargs_a}
    b = {**base, **kwargs_b}
    assert hg.compute_cache_key(fixture, calib, **a) != hg.compute_cache_key(
        fixture, calib, **b
    )


def test_cache_key_invalidates_on_source_file_edit(tmp_path: Path):
    """When a tracked source file's content changes, the key changes too.

    Drive this through a fake repo_root pointing at tmp_path so we can
    rewrite a tracked source without touching the real repo files.
    """
    # Lay out the tracked _SOURCE_FILES under tmp_path with stub content.
    for relpath in hg._SOURCE_FILES:
        p = tmp_path / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub v1\n")
    fixture = tmp_path / "fix.bin"
    calib = tmp_path / "cal.txt"
    fixture.write_bytes(b"\x00")
    calib.write_text("x")

    k0 = hg.compute_cache_key(
        fixture, calib, n_seqs=4, seq_len=4, alpha=0.4,
        awq_targets=("c_attn",), repo_root=tmp_path,
    )

    # Edit one tracked file.
    target = tmp_path / hg._SOURCE_FILES[0]
    target.write_text("# stub v2 — changed\n")

    k1 = hg.compute_cache_key(
        fixture, calib, n_seqs=4, seq_len=4, alpha=0.4,
        awq_targets=("c_attn",), repo_root=tmp_path,
    )
    assert k0 != k1


def test_try_load_rejects_partial_schema(tmp_path: Path):
    """If a .npz has H_* arrays but no matching X_* arrays (or vice versa),
    the load is treated as a miss so the caller recomputes both."""
    cache_path = tmp_path / "partial.npz"
    np.savez(cache_path, H_only_h=np.zeros((4, 4)))
    assert hg.try_load(cache_path) is None

    cache_path2 = tmp_path / "mismatched.npz"
    np.savez(
        cache_path2,
        H_a=np.zeros((4, 4)),
        X_b=np.zeros(4, dtype=np.float32),  # different source name from H
    )
    assert hg.try_load(cache_path2) is None


def test_cache_path_for_uses_npz_suffix(tmp_path: Path):
    p = hg.cache_path_for(tmp_path, "abcdef")
    assert p == tmp_path / "abcdef.npz"
