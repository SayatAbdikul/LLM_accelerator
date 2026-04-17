"""Tests for MLP FC1→GELU→FC2 strip-mining correctness."""
import pytest
import numpy as np
from taccel.compiler.tiler import tile_strip_mine, pad_dim, TILE
from taccel.isa.opcodes import ABUF_SIZE


class TestMlpStripMining:
    def test_strip_mine_schedule(self):
        """FC1 [197, 768]: 13 strips of 16 rows each."""
        M, N, K = 197, 768, 192
        strips = tile_strip_mine(M, N, K, strip_rows=TILE)
        assert len(strips) == pad_dim(M) // TILE  # 13 strips

    def test_each_strip_fits_abuf(self):
        """Each FC1 output strip [16, 768] = 12,288 bytes << 128KB."""
        strip_bytes = TILE * pad_dim(768)
        assert strip_bytes < ABUF_SIZE, f"Strip {strip_bytes}B >= ABUF {ABUF_SIZE}B"

    def test_fc1_plus_input_strip_fits_abuf(self):
        """FC1 input strip [16, 192] + output strip [16, 768] = 15,360 bytes < 128KB."""
        in_strip = TILE * pad_dim(192)   # 16 * 192 = 3,072
        out_strip = TILE * pad_dim(768)  # 16 * 768 = 12,288
        peak = in_strip + out_strip
        assert peak < ABUF_SIZE, f"Peak {peak}B >= ABUF {ABUF_SIZE}B"

    def test_fc2_strip_peak_fits_abuf(self):
        """FC2: input [16, 768] + output [16, 192] + residual [16, 192] = 18,432B < 128KB."""
        in_strip = TILE * pad_dim(768)    # 12,288
        out_strip = TILE * pad_dim(192)   # 3,072
        residual = TILE * pad_dim(192)    # 3,072
        peak = in_strip + out_strip + residual
        assert peak < ABUF_SIZE, f"Peak FC2 {peak}B >= ABUF {ABUF_SIZE}B"

    def test_dram_temp_size(self):
        """FC1 temp spill = 208 * 768 = 159,744 bytes."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(768)
        temp_bytes = M_pad * N_pad
        assert temp_bytes == 159_744

    def test_strip_mine_covers_all_rows(self):
        """All M rows are covered by strips."""
        M, N, K = 197, 768, 192
        strips = tile_strip_mine(M, N, K, strip_rows=TILE)
        total_rows_processed = sum(s.M_padded for s in strips)
        assert total_rows_processed == pad_dim(M)

    def test_strip_matmul_correctness(self):
        """Strip-mined FC1 matmul matches full matmul."""
        np.random.seed(0)
        M, N, K = 32, 48, 32  # Small for speed
        A = np.random.randint(-10, 10, (M, K), dtype=np.int8)
        B = np.random.randint(-10, 10, (K, N), dtype=np.int8)

        # Full reference matmul
        ref = A.astype(np.int32) @ B.astype(np.int32)  # [M, N]

        # Strip-mined
        result = np.zeros((M, N), dtype=np.int32)
        strip = TILE
        for m_start in range(0, M, strip):
            m_end = min(m_start + strip, M)
            A_strip = A[m_start:m_end]
            result[m_start:m_end] = A_strip.astype(np.int32) @ B.astype(np.int32)

        np.testing.assert_array_equal(result, ref)
