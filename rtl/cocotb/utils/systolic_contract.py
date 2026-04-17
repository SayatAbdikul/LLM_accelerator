"""Shared cocotb helpers for row-major MATMUL contract preparation."""

from utils.insn_builder import LOAD, SET_ADDR_HI, SET_ADDR_LO, SYNC, BUF_ABUF, BUF_WBUF


SYS_DIM = 16


def set_addr_insns(reg: int, addr: int) -> list[int]:
    return [
        SET_ADDR_LO(reg, addr & 0x0FFFFFFF),
        SET_ADDR_HI(reg, (addr >> 28) & 0x0FFFFFFF),
    ]


def append_load_sync(prog: list[int], reg: int, addr: int, buf_id: int, sram_off: int, xfer_len: int) -> None:
    prog.extend(set_addr_insns(reg, addr))
    prog.append(LOAD(buf_id, sram_off, xfer_len, reg, 0))
    prog.append(SYNC(0b001))


def flatten_16x16(mat: list[list[int]]) -> bytes:
    return bytes((row[col] & 0xFF) for row in mat for col in range(16))


def flatten_tile_32x32(mat: list[list[int]], row_base: int, col_base: int) -> bytes:
    return bytes((mat[row_base + r][col_base + c] & 0xFF) for r in range(16) for c in range(16))


def flatten_tile_16x64(mat: list[list[int]], col_base: int) -> bytes:
    return bytes((mat[r][col_base + c] & 0xFF) for r in range(16) for c in range(16))


def flatten_tile_64x16(mat: list[list[int]], row_base: int) -> bytes:
    return bytes((mat[row_base + r][c] & 0xFF) for r in range(16) for c in range(16))


def prepare_logical_16x16(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                          a_addr: int, b_addr: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    dram.write_bytes(a_addr, flatten_16x16(a))
    dram.write_bytes(b_addr, flatten_16x16(b))
    append_load_sync(prog, 0, a_addr, BUF_ABUF, abuf_off, (16 * 16) // 16)
    append_load_sync(prog, 1, b_addr, BUF_WBUF, wbuf_off, (16 * 16) // 16)


def prepare_logical_32x32(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                          a_base: int, b_base: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    a_bytes = bytes((a[r][c] & 0xFF) for r in range(32) for c in range(32))
    b_bytes = bytes((b[r][c] & 0xFF) for r in range(32) for c in range(32))
    dram.write_bytes(a_base, a_bytes)
    dram.write_bytes(b_base, b_bytes)
    append_load_sync(prog, 0, a_base, BUF_ABUF, abuf_off, (32 * 32) // 16)
    append_load_sync(prog, 1, b_base, BUF_WBUF, wbuf_off, (32 * 32) // 16)


def prepare_logical_16x64x16(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                             a_base: int, b_base: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    a_bytes = bytes((a[r][c] & 0xFF) for r in range(16) for c in range(64))
    b_bytes = bytes((b[r][c] & 0xFF) for r in range(64) for c in range(16))
    dram.write_bytes(a_base, a_bytes)
    dram.write_bytes(b_base, b_bytes)
    append_load_sync(prog, 0, a_base, BUF_ABUF, abuf_off, (16 * 64) // 16)
    append_load_sync(prog, 1, b_base, BUF_WBUF, wbuf_off, (64 * 16) // 16)
