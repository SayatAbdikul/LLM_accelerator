"""Matrix tiling for 16x16 systolic array."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

TILE = 16


@dataclass
class TileOp:
    """A single tile operation in the schedule."""
    m_start: int  # row start in output
    n_start: int  # col start in output
    k_start: int  # inner dimension start
    M_eff: int    # effective rows (may be < TILE at boundary)
    N_eff: int    # effective cols
    K_eff: int    # effective inner dim
    accumulate: bool  # whether to accumulate (k > 0)


@dataclass
class TileSchedule:
    """Complete tile schedule for a matmul."""
    ops: List[TileOp]
    M_padded: int
    N_padded: int
    K_padded: int
    M_orig: int
    N_orig: int
    K_orig: int
    m_tiles: int
    n_tiles: int
    k_tiles: int

    @property
    def config_tile_M(self) -> int:
        """Encoded M for CONFIG_TILE (0-based)."""
        return self.m_tiles - 1

    @property
    def config_tile_N(self) -> int:
        return self.n_tiles - 1

    @property
    def config_tile_K(self) -> int:
        return self.k_tiles - 1


def pad_dim(d: int) -> int:
    """Pad dimension to next multiple of TILE."""
    return ((d + TILE - 1) // TILE) * TILE


def tile_matmul(M: int, N: int, K: int) -> TileSchedule:
    """Create tile schedule for C[M,N] = A[M,K] @ B[K,N].

    Pads all dimensions to multiples of 16.
    """
    M_pad = pad_dim(M)
    N_pad = pad_dim(N)
    K_pad = pad_dim(K)
    m_tiles = M_pad // TILE
    n_tiles = N_pad // TILE
    k_tiles = K_pad // TILE

    ops = []
    for m in range(m_tiles):
        for n in range(n_tiles):
            for k in range(k_tiles):
                m_start = m * TILE
                n_start = n * TILE
                k_start = k * TILE
                ops.append(TileOp(
                    m_start=m_start,
                    n_start=n_start,
                    k_start=k_start,
                    M_eff=min(TILE, M - m_start) if m_start < M else 0,
                    N_eff=min(TILE, N - n_start) if n_start < N else 0,
                    K_eff=min(TILE, K - k_start) if k_start < K else 0,
                    accumulate=(k > 0),
                ))

    return TileSchedule(
        ops=ops,
        M_padded=M_pad, N_padded=N_pad, K_padded=K_pad,
        M_orig=M, N_orig=N, K_orig=K,
        m_tiles=m_tiles, n_tiles=n_tiles, k_tiles=k_tiles,
    )


def tile_qkt(seq_len: int, head_dim: int) -> Tuple[TileSchedule, dict]:
    """Create tile schedule for Q@K^T attention matmul.

    Handles pad-then-transpose ordering:
    1. K [seq_len, head_dim] → pad to [M_pad, head_dim]
    2. Transpose: [M_pad, head_dim] → [head_dim, M_pad]
    3. Q [M_pad, head_dim] @ K^T [head_dim, M_pad] → [M_pad, M_pad]

    Returns (schedule, transpose_info).
    """
    M_pad = pad_dim(seq_len)
    K_pad = pad_dim(head_dim)

    # After transpose: K^T is [head_dim, M_pad] (already padded correctly)
    schedule = tile_matmul(M_pad, M_pad, K_pad)

    # Transpose info for BUF_COPY
    transpose_info = {
        "src_rows": M_pad // TILE,  # src_rows field in BUF_COPY (in 16-element units)
        "length": M_pad * K_pad // TILE,  # total bytes in 16-byte units
        "src_shape": (M_pad, K_pad),
        "dst_shape": (K_pad, M_pad),
    }

    return schedule, transpose_info


def tile_strip_mine(M: int, N: int, K: int, strip_rows: int = TILE) -> List[TileSchedule]:
    """Create per-strip tile schedules for strip-mined matmul.

    Used when output doesn't fit in ABUF (e.g., FC1 output [208, 768]).
    Each strip processes strip_rows of M at a time.
    """
    M_pad = pad_dim(M)
    N_pad = pad_dim(N)
    K_pad = pad_dim(K)

    strips = []
    for m_start in range(0, M_pad, strip_rows):
        m_end = min(m_start + strip_rows, M_pad)
        strip_M = m_end - m_start
        schedule = tile_matmul(strip_M, N, K)
        schedule.M_orig = min(strip_rows, M - m_start) if m_start < M else 0
        strips.append(schedule)

    return strips
