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
