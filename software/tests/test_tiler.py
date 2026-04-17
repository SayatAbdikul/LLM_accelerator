"""Tests for the tiler module."""
import pytest
import numpy as np
from taccel.compiler.tiler import tile_matmul, tile_qkt, pad_dim, TILE


class TestPadDim:
    def test_already_multiple(self):
        assert pad_dim(16) == 16
        assert pad_dim(32) == 32
        assert pad_dim(208) == 208

    def test_needs_padding(self):
        assert pad_dim(197) == 208  # seq_len → 13*16
        assert pad_dim(192) == 192  # embed_dim already multiple
        assert pad_dim(1) == 16

    def test_zero(self):
        assert pad_dim(0) == 0


class TestTileMatmul:
    def test_basic_coverage(self):
        """All elements are covered by at least one tile."""
        M, N, K = 197, 192, 192
        sched = tile_matmul(M, N, K)

        covered = np.zeros((sched.M_padded, sched.N_padded), dtype=bool)
        for op in sched.ops:
            covered[op.m_start:op.m_start + TILE, op.n_start:op.n_start + TILE] = True

        assert np.all(covered), "Some output elements not covered"

    def test_tile_counts(self):
        M, N, K = 208, 192, 192
        sched = tile_matmul(M, N, K)
        assert sched.m_tiles == 13
        assert sched.n_tiles == 12
        assert sched.k_tiles == 12

    def test_accumulate_flag(self):
        """First K tile has accumulate=False, rest have accumulate=True."""
        sched = tile_matmul(32, 32, 64)
        for op in sched.ops:
            if op.k_start == 0:
                assert not op.accumulate
            else:
                assert op.accumulate

    def test_small_matmul(self):
        """1x1 tile matmul."""
        sched = tile_matmul(16, 16, 16)
        assert len(sched.ops) == 1
        assert sched.ops[0].m_start == 0

    def test_config_tile_encoding(self):
        """config_tile_M = m_tiles - 1 (0-based)."""
        sched = tile_matmul(208, 208, 64)
        assert sched.config_tile_M == 12  # 13 tiles - 1
        assert sched.config_tile_N == 12
        assert sched.config_tile_K == 3   # 4 tiles - 1


class TestQKTTiling:
    def test_output_shape(self):
        """Q@K^T for DeiT-tiny: [197, 197] valid region in [208, 208]."""
        sched, trans_info = tile_qkt(197, 64)
        assert sched.M_padded == 208
        assert sched.N_padded == 208
        assert sched.K_padded == 64  # head_dim already multiple of 16

    def test_transpose_info(self):
        """Transpose info for BUF_COPY is correct."""
        sched, trans_info = tile_qkt(197, 64)
        assert trans_info["src_rows"] == 13  # 208 / 16
        assert trans_info["length"] == 832   # 208 * 64 / 16

    def test_pad_then_transpose_correctness(self):
        """Verify that padding before transpose gives correct matmul result."""
        # K [197, 64] padded to [208, 64]
        K = np.random.randint(-127, 127, (197, 64), dtype=np.int8)
        Q = np.random.randint(-127, 127, (197, 64), dtype=np.int8)

        # Pad K
        K_pad = np.zeros((208, 64), dtype=np.int8)
        K_pad[:197] = K

        # Transpose padded K → [64, 208]
        K_T = K_pad.T  # [64, 208]

        # Pad Q
        Q_pad = np.zeros((208, 64), dtype=np.int8)
        Q_pad[:197] = Q

        # Matmul Q_pad @ K_T → [208, 208]
        result_int32 = Q_pad.astype(np.int32) @ K_T.astype(np.int32)

        # Valid region [197, 197] should match Q @ K.T
        expected = Q.astype(np.int32) @ K.T.astype(np.int32)
        np.testing.assert_array_equal(result_int32[:197, :197], expected)

    def test_padding_zeros_dont_corrupt(self):
        """Zero padding in K's M dimension contributes zero to matmul result."""
        K = np.ones((197, 64), dtype=np.int8)
        Q = np.ones((197, 64), dtype=np.int8)

        K_pad = np.zeros((208, 64), dtype=np.int8)
        K_pad[:197] = K

        result_padded = Q.astype(np.int32) @ K_pad[:197].T.astype(np.int32)
        result_ref = Q.astype(np.int32) @ K.T.astype(np.int32)
        np.testing.assert_array_equal(result_padded, result_ref)
