"""Bit-accurate INT8 16×16 systolic array matmul model.

Arithmetic
----------
Each 16×16 tile computes: C[i][j] += Σ_k A[i][k] * B[k][j]
where A, B are INT8 and C is INT32.  Partial products are 16 bits;
the running sum is accumulated into 32 bits.

Overflow
--------
INT32 accumulator range is ±2,147,483,647.  Maximum possible accumulator
magnitude for a single MATMUL is M × K × 127² ≈ M × K × 16,129.
For the DeiT-tiny workload: max(M×K) = 208 × 192 = 39,936, giving
max |acc| ≈ 644 M, which fits in INT32.  However, if the compiler
generates a MATMUL with M × K > ~133,000 (e.g. M=1024, K=1024), the
accumulator wraps silently.  The compiler is responsible for tiling to
avoid this.  RTL may optionally implement saturating accumulators.

Cycle model
-----------
16 cycles per 16×16 tile.  Total = m_tiles × n_tiles × k_tiles × 16.
"""
import numpy as np
from . import memory
from ..isa.opcodes import BUF_ACCUM

TILE = 16
CYCLE_COST = 16  # cycles per 16x16 tile


def execute_matmul(state, insn):
    """Execute MATMUL instruction.

    Performs tiled matmul using CONFIG_TILE dimensions.
    src1 = activations (INT8), src2 = weights (INT8), dst = ACCUM (INT32).

    flags[0] = 0: dst = src1 @ src2       (overwrite)
    flags[0] = 1: dst += src1 @ src2      (accumulate)
    """
    from .simulator import ConfigError

    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    # Tile config: encoded as 0-based, so M_tiles = M+1, etc.
    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1

    M = m_tiles * TILE
    N = n_tiles * TILE
    K = k_tiles * TILE

    accumulate = bool(insn.flags & 1)

    # Read source tiles
    src1 = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, K)
    src2 = memory.read_int8_tile(state, insn.src2_buf, insn.src2_off, K, N)

    # Read existing accumulator if accumulating
    if accumulate:
        dst = memory.read_int32_tile(state, insn.dst_buf, insn.dst_off, M, N)
    else:
        dst = np.zeros((M, N), dtype=np.int32)

    # Vectorized matmul: same INT8→INT32 arithmetic, runs in NumPy C backend
    dst += np.matmul(src1.astype(np.int32), src2.astype(np.int32))

    # Write result
    memory.write_int32_tile(state, insn.dst_buf, insn.dst_off, dst)
    state.cycle_count += m_tiles * n_tiles * k_tiles * CYCLE_COST
