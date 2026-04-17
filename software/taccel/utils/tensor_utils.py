"""Tiling, padding, and reshape utilities."""
import numpy as np

TILE_SIZE = 16


def pad_to_multiple(x: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    """Pad a 2D array so both dimensions are multiples of tile_size."""
    rows, cols = x.shape
    pad_rows = (tile_size - rows % tile_size) % tile_size
    pad_cols = (tile_size - cols % tile_size) % tile_size
    if pad_rows == 0 and pad_cols == 0:
        return x
    return np.pad(x, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)


def tiles_for_dim(dim: int, tile_size: int = TILE_SIZE) -> int:
    """Number of tiles needed to cover a dimension."""
    return (dim + tile_size - 1) // tile_size


def unpad(x: np.ndarray, orig_rows: int, orig_cols: int) -> np.ndarray:
    """Remove padding to restore original dimensions."""
    return x[:orig_rows, :orig_cols]


def tile_coords(M: int, N: int, K: int, tile_size: int = TILE_SIZE):
    """Generate tile coordinates for tiled matmul C[M,N] = A[M,K] @ B[K,N].

    Yields (m_start, n_start, k_start, is_first_k) tuples.
    """
    m_tiles = tiles_for_dim(M, tile_size)
    n_tiles = tiles_for_dim(N, tile_size)
    k_tiles = tiles_for_dim(K, tile_size)
    for m in range(m_tiles):
        for n in range(n_tiles):
            for k in range(k_tiles):
                yield (m * tile_size, n * tile_size, k * tile_size, k == 0)
