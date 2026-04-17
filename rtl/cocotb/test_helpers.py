"""cocotb helper-engine tests for Phase C and Stage E helper ops."""

import math

import cocotb

from utils.dram_model import DramModel
from utils.insn_builder import (
    HALT, SYNC, CONFIG_TILE, SET_SCALE, SET_ADDR_LO, SET_ADDR_HI,
    LOAD, STORE, BUF_COPY, MATMUL, REQUANT, REQUANT_PC, SCALE_MUL, VADD, DEQUANT_ADD,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)
from utils.systolic_contract import prepare_logical_16x16
from utils.testbench import set_addr, setup_test, wait_halt


def _sat_add(a: int, b: int) -> int:
    s = a + b
    if s > 127:
        return 127
    if s < -128:
        return -128
    return s


def _fp16_mul_round_even(src: int, fp16: int) -> int:
    sign = (fp16 >> 15) & 1
    exp = (fp16 >> 10) & 0x1F
    frac = fp16 & 0x3FF
    if exp == 0 and frac == 0:
        return 0
    if exp == 0:
        mant = frac
        shift = -24
    else:
        mant = 1024 + frac
        shift = exp - 25
    prod = src * mant
    if sign:
        prod = -prod
    if shift >= 0:
        return prod << shift
    rshift = -shift
    abs_prod = -prod if prod < 0 else prod
    q = abs_prod >> rshift
    rem = abs_prod & ((1 << rshift) - 1)
    half = 1 << (rshift - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return -q if prod < 0 else q


def _requant(src: int, scale: int) -> int:
    scaled = _fp16_mul_round_even(src, scale)
    if scaled > 127:
        return 127
    if scaled < -128:
        return -128
    return scaled


def _fp16_to_float(bits: int) -> float:
    bits = int(bits)
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    sign_v = -1.0 if sign else 1.0
    if exp == 0 and frac == 0:
        return 0.0
    if exp == 0:
        return sign_v * (frac / 1024.0) * (2.0 ** -14)
    if exp == 31:
        return sign_v * 65504.0
    return sign_v * (1.0 + frac / 1024.0) * (2.0 ** (exp - 15))


def _round_half_even(x: float) -> int:
    floor_i = math.floor(x)
    frac = x - floor_i
    if frac > 0.5:
        return floor_i + 1
    if frac < 0.5:
        return floor_i
    return floor_i + 1 if (floor_i & 1) else floor_i


def _dequant_add(accum: int, skip: int, accum_scale: int, skip_scale: int) -> int:
    x = accum * _fp16_to_float(accum_scale) + skip * _fp16_to_float(skip_scale)
    q = _round_half_even(x)
    return max(-128, min(127, q))


def _pack_i32_le(vals):
    out = bytearray()
    for v in vals:
        out.extend(int(v & 0xFFFFFFFF).to_bytes(4, "little", signed=False))
    return bytes(out)


@cocotb.test()
async def test_buf_copy_flat_roundtrip(dut):
    src_addr = 0x10000
    dst_addr = 0x12000
    src = bytes((0x20 + 7 * i) & 0xFF for i in range(48))
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 3, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 8, 3, 0, 0),
        *set_addr(1, dst_addr),
        STORE(BUF_WBUF, 8, 3, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: src})
    await wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(src)]) == src


@cocotb.test()
async def test_buf_copy_overlap_roundtrip(dut):
    src_addr = 0x14000
    dst_addr = 0x15000
    src = bytearray((0x51 + 11 * i) & 0xFF for i in range(96))
    expected = bytearray(src)
    expected[16:64] = expected[32:80]
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 6, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 2, BUF_ABUF, 1, 3, 0, 0),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, 0, 6, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: bytes(src)})
    await wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_buf_copy_transpose_unaligned_roundtrip(dut):
    src_addr = 0x18000
    dst_addr = 0x19000
    rows, cols = 16, 18
    src = bytearray(rows * cols)
    expected = bytearray(cols * rows)
    for r in range(rows):
        for c in range(cols):
            src[r * cols + c] = (r * 19 + c * 7 + 3) & 0xFF
            expected[c * rows + r] = src[r * cols + c]
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 18, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 0, 18, 1, 1),
        *set_addr(1, dst_addr),
        STORE(BUF_WBUF, 0, 18, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: bytes(src)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_vadd_int8_roundtrip(dut):
    src_a_addr = 0x1A000
    src_b_addr = 0x1B000
    dst_addr = 0x1C000
    src_a = bytearray(256)
    src_b = bytearray(256)
    expected = bytearray(256)
    for i in range(256):
        a = 120 if i % 5 == 0 else -120 if i % 7 == 0 else (i % 17) - 8
        b = 30 if i % 5 == 0 else -30 if i % 7 == 0 else (i % 11) - 5
        src_a[i] = a & 0xFF
        src_b[i] = b & 0xFF
        expected[i] = _sat_add(a, b) & 0xFF
    prog = [
        *set_addr(0, src_a_addr),
        LOAD(BUF_ABUF, 0, 16, 0, 0),
        SYNC(0b001),
        *set_addr(1, src_b_addr),
        LOAD(BUF_WBUF, 0, 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        VADD(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ABUF, 32, 0),
        *set_addr(2, dst_addr),
        STORE(BUF_ABUF, 32, 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_a_addr: bytes(src_a), src_b_addr: bytes(src_b)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_requant_roundtrip(dut):
    src_addr = 0x1D000
    dst_addr = 0x1E000
    scale = 0x3800
    pattern = [1, 3, 5, -1, -3, -5, 255, 257, -255, -257, 300, -300, 0, 2, -2, 7]
    src = []
    expected = bytearray(256)
    for r in range(16):
        for c in range(16):
            v = pattern[c] + r
            src.append(v)
            expected[r * 16 + c] = _requant(v, scale) & 0xFF
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, 64, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(0, scale),
        REQUANT(BUF_ACCUM, 0, BUF_ABUF, 64, 0),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, 64, 16, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _pack_i32_le(src)})
    await wait_halt(dut, max_cycles=500_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_matmul_then_requant_roundtrip(dut):
    src_a_addr = 0x20000
    src_b_addr = 0x21000
    dst_addr = 0x22000
    a = [[(i * 5 + j) - 20 for j in range(16)] for i in range(16)]
    eye = [[1 if i == j else 0 for j in range(16)] for i in range(16)]

    expected = bytearray()
    for r in range(16):
        for c in range(16):
            expected.append(a[r][c] & 0xFF)

    prog = []
    dram = DramModel()
    prepare_logical_16x16(dram, prog, a, eye, src_a_addr, src_b_addr, abuf_off=128, wbuf_off=0)
    prog.extend([
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 128, BUF_WBUF, 0, BUF_ACCUM, 0, 0),
        SYNC(0b010),
        SET_SCALE(0, 0x3C00),
        REQUANT(BUF_ACCUM, 0, BUF_ABUF, 256, 0),
        *set_addr(2, dst_addr),
        STORE(BUF_ABUF, 256, 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_requant_pc_roundtrip(dut):
    src_addr = 0x23000
    scale_addr = 0x24000
    dst_addr = 0x25000
    dst_off = 96
    src = []
    scales = []
    expected = bytearray(256)
    for c in range(16):
        scales.append(0x3C00 if c % 4 == 0 else 0x3800 if c % 4 == 1 else 0xBC00 if c % 4 == 2 else 0x0000)
    for r in range(16):
        for c in range(16):
            v = (c - 6) * 37 + r * 5
            src.append(v)
            expected[r * 16 + c] = _requant(v, scales[c]) & 0xFF
    scale_bytes = b"".join(int(s & 0xFFFF).to_bytes(2, "little") for s in scales)
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, 64, 0, 0),
        SYNC(0b001),
        *set_addr(1, scale_addr),
        LOAD(BUF_WBUF, 320, 2, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        REQUANT_PC(BUF_ACCUM, 0, BUF_WBUF, 320, BUF_ABUF, dst_off, 0),
        *set_addr(2, dst_addr),
        STORE(BUF_ABUF, dst_off, 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _pack_i32_le(src), scale_addr: scale_bytes})
    await wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_scale_mul_int8_roundtrip(dut):
    src_addr = 0x26000
    dst_addr = 0x27000
    dst_off = 128
    scale = 0xB800  # -0.5
    src = bytearray(256)
    expected = bytearray(256)
    for i in range(256):
        v = ((i * 9) % 61) - 30
        src[i] = v & 0xFF
        expected[i] = _requant(v, scale) & 0xFF
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 16, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(2, scale),
        SCALE_MUL(BUF_ABUF, 0, BUF_WBUF, dst_off, 2),
        *set_addr(1, dst_addr),
        STORE(BUF_WBUF, dst_off, 16, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: bytes(src)})
    await wait_halt(dut, max_cycles=500_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_dequant_add_roundtrip(dut):
    accum_addr = 0x28000
    skip_addr = 0x29000
    dst_addr = 0x2A000
    dst_off = 192
    acc_scale = 0x2C00
    skip_scale = 0x3400
    accum = []
    skip = bytearray(256)
    expected = bytearray(256)
    for i in range(256):
        accum_v = (i - 120) * 11
        skip_v = ((i * 5) % 29) - 14
        accum.append(accum_v)
        skip[i] = skip_v & 0xFF
        expected[i] = _dequant_add(accum_v, skip_v, acc_scale, skip_scale) & 0xFF
    prog = [
        *set_addr(0, accum_addr),
        LOAD(BUF_ACCUM, 0, 64, 0, 0),
        SYNC(0b001),
        *set_addr(1, skip_addr),
        LOAD(BUF_ABUF, 0, 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(4, acc_scale),
        SET_SCALE(5, skip_scale),
        DEQUANT_ADD(BUF_ACCUM, 0, BUF_ABUF, 0, BUF_WBUF, dst_off, 4),
        *set_addr(2, dst_addr),
        STORE(BUF_WBUF, dst_off, 16, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={accum_addr: _pack_i32_le(accum), skip_addr: bytes(skip)})
    await wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)
