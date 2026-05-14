"""Memory access helpers with bounds checking."""
import numpy as np
from ..isa.opcodes import (
    BUFFER_MAX_OFF, BUF_ABUF, BUF_WBUF, BUF_ACCUM,
    ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE,
)

UNIT = 16  # 16 bytes per addressing unit


class SRAMAccessError(Exception):
    def __init__(self, buf_id, offset, limit):
        self.buf_id = buf_id
        self.offset = offset
        self.limit = limit
        buf_names = {0: "ABUF", 1: "WBUF", 2: "ACCUM"}
        super().__init__(
            f"SRAM access out of bounds: {buf_names.get(buf_id, f'BUF{buf_id}')}[{offset}] "
            f"exceeds limit {limit}"
        )


class DRAMAccessError(Exception):
    def __init__(self, addr):
        self.addr = addr
        super().__init__(f"DRAM access out of bounds: address {addr:#x}")


def _check_sram_bounds(buf_id: int, offset_units: int, length_units: int = 0):
    """Check SRAM access is within bounds."""
    max_off = BUFFER_MAX_OFF.get(buf_id)
    if max_off is None:
        raise SRAMAccessError(buf_id, offset_units, 0)
    end = offset_units + length_units
    if offset_units > max_off or (length_units > 0 and end - 1 > max_off):
        raise SRAMAccessError(buf_id, offset_units, max_off)


def _buf_size(buf_id: int) -> int:
    return {BUF_ABUF: ABUF_SIZE, BUF_WBUF: WBUF_SIZE, BUF_ACCUM: ACCUM_SIZE}[buf_id]


def read_int8_tile(state, buf_id: int, offset_units: int, rows: int, cols: int) -> np.ndarray:
    """Read an INT8 tile from SRAM buffer.

    offset_units: offset in 16-byte units
    rows, cols: tile dimensions
    """
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    total_bytes = rows * cols

    if buf_id == BUF_ACCUM:
        raise ValueError("Use read_int32_tile for ACCUM buffer")

    buf = state.get_buffer(buf_id)
    end = byte_offset + total_bytes
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])

    data = np.frombuffer(buf[byte_offset:end], dtype=np.int8).copy()
    return data.reshape(rows, cols)


def write_int8_tile(state, buf_id: int, offset_units: int, data: np.ndarray):
    """Write an INT8 tile to SRAM buffer."""
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    flat = data.astype(np.int8).tobytes()

    if buf_id == BUF_ACCUM:
        raise ValueError("Use write_int32_tile for ACCUM buffer")

    buf = state.get_buffer(buf_id)
    end = byte_offset + len(flat)
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])

    buf[byte_offset:end] = flat


def read_int32_tile(state, buf_id: int, offset_units: int, rows: int, cols: int) -> np.ndarray:
    """Read an INT32 tile from a buffer.

    For ACCUM: reads directly from the int32 array.
    For ABUF/WBUF: reinterprets bytes as int32.
    """
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT

    if buf_id == BUF_ACCUM:
        # ACCUM is stored as flat int32 array
        int32_offset = byte_offset // 4
        total_ints = rows * cols
        end = int32_offset + total_ints
        if end > len(state.accum):
            raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
        return state.accum[int32_offset:end].reshape(rows, cols).copy()
    else:
        buf = state.get_buffer(buf_id)
        total_bytes = rows * cols * 4
        end = byte_offset + total_bytes
        if end > len(buf):
            raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
        return np.frombuffer(buf[byte_offset:end], dtype=np.int32).copy().reshape(rows, cols)


def write_int32_tile(state, buf_id: int, offset_units: int, data: np.ndarray):
    """Write an INT32 tile to a buffer."""
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT

    if buf_id == BUF_ACCUM:
        int32_offset = byte_offset // 4
        flat = data.astype(np.int32).flatten()
        end = int32_offset + len(flat)
        if end > len(state.accum):
            raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
        state.accum[int32_offset:end] = flat
    else:
        buf = state.get_buffer(buf_id)
        flat = data.astype(np.int32).tobytes()
        end = byte_offset + len(flat)
        if end > len(buf):
            raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
        buf[byte_offset:end] = flat


# ---------------------------------------------------------------------------
# W8A32 FP32 tile/vector helpers (Phase 3 (c.1), M1)
#
# ABUF bytes are reinterpreted as FP32 (4 bytes/element) when accessed by
# the W8A32 R-type opcodes (DEQUANT_ACCUM_FP32 / VADD_FP32 / LN_FP32 /
# GELU_FP32 / SOFTMAX_FP32 / MASKED_SOFTMAX_FP32 / QUANT_FP32_INT8 input).
# Same byte storage as INT8 — only the interpretation changes. WBUF FP16
# vectors (LN gamma+beta, per-channel dequant scales) are read with
# read_fp16_vector.
# ---------------------------------------------------------------------------


def read_fp32_tile(state, buf_id: int, offset_units: int, rows: int, cols: int) -> np.ndarray:
    """Read an FP32 tile from SRAM buffer.

    Same byte addressing as `read_int8_tile` (16-byte units) but
    interprets the underlying bytes as little-endian float32. ACCUM is
    not a valid source/destination for FP32 — its storage is fixed
    INT32 by the hardware contract.
    """
    if buf_id == BUF_ACCUM:
        raise ValueError(
            "ACCUM buffer is INT32-only; read FP32 from ABUF/WBUF via "
            "DEQUANT_ACCUM_FP32 to write to ABUF"
        )
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    total_bytes = rows * cols * 4

    buf = state.get_buffer(buf_id)
    end = byte_offset + total_bytes
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])

    data = np.frombuffer(buf[byte_offset:end], dtype=np.float32).copy()
    return data.reshape(rows, cols)


def write_fp32_tile(state, buf_id: int, offset_units: int, data: np.ndarray):
    """Write an FP32 tile to SRAM buffer (same addressing as read_fp32_tile)."""
    if buf_id == BUF_ACCUM:
        raise ValueError(
            "ACCUM buffer is INT32-only; FP32 results write to ABUF"
        )
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    flat = data.astype(np.float32).tobytes()

    buf = state.get_buffer(buf_id)
    end = byte_offset + len(flat)
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])

    buf[byte_offset:end] = flat


# ---------------------------------------------------------------------------
# W8A16 FP16 tile helpers (Phase 3 (c.2), M1)
#
# ABUF bytes are reinterpreted as FP16 (2 bytes/element) when accessed by
# the W8A32 R-type opcodes (0x17-0x1F) with `flags=1`. Reads widen to FP32
# on return (FP16-storage / FP32-datapath convention, matches
# read_fp16_vector). Writes accept FP32 and downcast to FP16 on store.
# Same byte addressing as the FP32 helpers (16-byte units); only the
# element size and dtype differ.
# ---------------------------------------------------------------------------


def read_fp16_tile(state, buf_id: int, offset_units: int, rows: int, cols: int) -> np.ndarray:
    """Read an FP16 tile from SRAM buffer, widened to FP32 on return.

    Same byte addressing as `read_fp32_tile` (16-byte units) but
    interprets the underlying bytes as little-endian float16 and
    widens to FP32 — internal datapath is FP32, storage is FP16.
    ACCUM is not a valid source/destination (INT32-only by contract).
    """
    if buf_id == BUF_ACCUM:
        raise ValueError(
            "ACCUM buffer is INT32-only; read FP16 from ABUF/WBUF via "
            "DEQUANT_ACCUM_FP32(flags=1) to write to ABUF"
        )
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    total_bytes = rows * cols * 2

    buf = state.get_buffer(buf_id)
    end = byte_offset + total_bytes
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])

    raw = np.frombuffer(buf[byte_offset:end], dtype=np.float16).copy()
    return raw.astype(np.float32).reshape(rows, cols)


def write_fp16_tile(state, buf_id: int, offset_units: int, data: np.ndarray):
    """Write an FP32 tile to SRAM downcast to FP16 (2 bytes/element)."""
    if buf_id == BUF_ACCUM:
        raise ValueError(
            "ACCUM buffer is INT32-only; FP16 results write to ABUF"
        )
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    flat = data.astype(np.float16).tobytes()

    buf = state.get_buffer(buf_id)
    end = byte_offset + len(flat)
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])

    buf[byte_offset:end] = flat


def read_fp16_vector(state, buf_id: int, offset_units: int, length: int) -> np.ndarray:
    """Read FP16 values (e.g. LN gamma/beta or per-channel dequant scales) → FP32.

    The widening to FP32 mirrors the architectural FP16-storage /
    FP32-datapath convention documented in the SFU comment in
    `opcodes.py`. ACCUM is not a valid source.
    """
    if buf_id == BUF_ACCUM:
        raise ValueError("ACCUM buffer is INT32-only; FP16 vectors live in WBUF/ABUF")
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    total_bytes = length * 2
    buf = state.get_buffer(buf_id)
    end = byte_offset + total_bytes
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
    return np.frombuffer(buf[byte_offset:end], dtype=np.float16).astype(np.float32)


def write_fp16_vector(state, buf_id: int, offset_units: int, data: np.ndarray):
    """Write FP32 values down-cast to FP16 (test/codegen helper)."""
    if buf_id == BUF_ACCUM:
        raise ValueError("ACCUM buffer is INT32-only")
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT
    flat = data.astype(np.float16).tobytes()
    buf = state.get_buffer(buf_id)
    end = byte_offset + len(flat)
    if end > len(buf):
        raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
    buf[byte_offset:end] = flat


def read_bytes(state, buf_id: int, offset_units: int, length_bytes: int) -> bytes:
    """Read raw bytes from SRAM buffer."""
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT

    if buf_id == BUF_ACCUM:
        data = state.accum.view(np.uint8)
        end = byte_offset + length_bytes
        if end > len(data):
            raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
        return bytes(data[byte_offset:end])
    else:
        buf = state.get_buffer(buf_id)
        end = byte_offset + length_bytes
        if end > len(buf):
            raise SRAMAccessError(buf_id, offset_units, BUFFER_MAX_OFF[buf_id])
        return bytes(buf[byte_offset:end])


def write_bytes(state, buf_id: int, offset_units: int, data: bytes):
    """Write raw bytes to SRAM buffer."""
    _check_sram_bounds(buf_id, offset_units)
    byte_offset = offset_units * UNIT

    if buf_id == BUF_ACCUM:
        view = state.accum.view(np.uint8)
        view[byte_offset:byte_offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
    else:
        buf = state.get_buffer(buf_id)
        buf[byte_offset:byte_offset + len(data)] = data
