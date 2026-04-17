"""Bit-accurate INT8/INT32 arithmetic helpers."""
import numpy as np

INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def clip_int8(val: int) -> int:
    """Clip to INT8 range."""
    if val < INT8_MIN:
        return INT8_MIN
    if val > INT8_MAX:
        return INT8_MAX
    return val


def clip_int32(val: int) -> int:
    """Clip to INT32 range."""
    if val < INT32_MIN:
        return INT32_MIN
    if val > INT32_MAX:
        return INT32_MAX
    return val


def saturating_add_int8(a: int, b: int) -> int:
    """INT8 saturating addition."""
    return clip_int8(int(a) + int(b))


def int8_matmul_tile(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Bit-accurate 16x16 INT8 matmul using Python int arithmetic.

    A: [16, 16] int8, B: [16, 16] int8 → C: [16, 16] int32
    """
    assert A.shape == (16, 16) and B.shape == (16, 16)
    C = np.zeros((16, 16), dtype=np.int32)
    for i in range(16):
        for j in range(16):
            acc = 0
            for k in range(16):
                acc += int(A[i, k]) * int(B[k, j])
            C[i, j] = acc
    return C


def requantize_int32_to_int8(val_int32: int, scale_fp32: float) -> int:
    """INT32 → INT8: clip(round(val × scale))."""
    return clip_int8(int(round(float(val_int32) * scale_fp32)))


def scale_mul_int32(val: int, scale_fp32: float) -> int:
    """Multiply by scale, result as INT32."""
    return clip_int32(int(round(float(val) * scale_fp32)))
