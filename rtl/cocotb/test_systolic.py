"""cocotb default-mode systolic contract tests."""

import random

import cocotb
from utils.dram_model import DramModel
from utils.insn_builder import HALT, SYNC, CONFIG_TILE, MATMUL, BUF_ACCUM
from utils.systolic_contract import (
    prepare_logical_16x16,
    prepare_logical_32x32,
    prepare_logical_16x64x16,
)
from utils.testbench import read_accum_16x16, read_accum_32x32, setup_test, wait_halt


def _matmul_ref(a, b):
    c = [[0 for _ in range(16)] for _ in range(16)]
    for i in range(16):
        for j in range(16):
            acc = 0
            for k in range(16):
                acc += int(a[i][k]) * int(b[k][j])
            c[i][j] = acc
    return c


def _matmul_ref_32(a, b):
    c = [[0 for _ in range(32)] for _ in range(32)]
    for i in range(32):
        for j in range(32):
            acc = 0
            for k in range(32):
                acc += int(a[i][k]) * int(b[k][j])
            c[i][j] = acc
    return c


def _matmul_ref_16x64x16(a, b):
    c = [[0 for _ in range(16)] for _ in range(16)]
    for i in range(16):
        for j in range(16):
            acc = 0
            for k in range(64):
                acc += int(a[i][k]) * int(b[k][j])
            c[i][j] = acc
    return c


def _to_i8(v):
    v &= 0xFF
    return v - 256 if v >= 128 else v



@cocotb.test()
async def test_matmul_identity(dut):
    a = [[(i * 3 + j) & 0x7F for j in range(16)] for i in range(16)]
    eye = [[1 if i == j else 0 for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, eye)

    prog = []
    dram = DramModel()
    prepare_logical_16x16(dram, prog, a, eye, 0x100000, 0x110000)
    prog.extend([
        CONFIG_TILE(1, 1, 1),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=500_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    assert read_accum_16x16(dut) == exp


@cocotb.test()
async def test_matmul_accumulate_flag(dut):
    a = [[1 for _ in range(16)] for _ in range(16)]
    b = [[2 if i == j else 0 for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, b)

    prog = []
    dram = DramModel()
    prepare_logical_16x16(dram, prog, a, b, 0x120000, 0x130000)
    prog.extend([
        CONFIG_TILE(1, 1, 1),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=1),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=600_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    exp2 = [[v * 2 for v in row] for row in exp]
    assert read_accum_16x16(dut) == exp2


@cocotb.test()
async def test_load_matmul_sync_integration(dut):
    a = [[_to_i8((i * 7 + j * 3) & 0xFF) for j in range(16)] for i in range(16)]
    b = [[_to_i8((i * 5 + j * 11) & 0xFF) for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, b)

    prog = []
    dram = DramModel()
    prepare_logical_16x16(dram, prog, a, b, 0x140000, 0x150000)
    prog.extend([
        CONFIG_TILE(1, 1, 1),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=500_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    assert read_accum_16x16(dut) == exp


@cocotb.test()
async def test_matmul_multitile_2x2x2(dut):
    a = [[((i * 7 + j * 5 + 3) % 11) - 5 for j in range(32)] for i in range(32)]
    b = [[((i * 3 + j * 9 + 1) % 13) - 6 for j in range(32)] for i in range(32)]
    exp = _matmul_ref_32(a, b)

    prog = []
    dram = DramModel()
    prepare_logical_32x32(dram, prog, a, b, 0x160000, 0x180000)
    prog.extend([
        CONFIG_TILE(2, 2, 2),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=800_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    assert read_accum_32x32(dut) == exp


@cocotb.test()
async def test_matmul_signed_extremes(dut):
    a = [[-128 if ((i + j) & 1) else 127 for j in range(16)] for i in range(16)]
    b = [[127 if ((i * 3 + j * 5) & 1) else -128 for j in range(16)] for i in range(16)]
    exp = _matmul_ref(a, b)

    prog = []
    dram = DramModel()
    prepare_logical_16x16(dram, prog, a, b, 0x1A0000, 0x1B0000)
    prog.extend([
        CONFIG_TILE(1, 1, 1),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=500_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    assert read_accum_16x16(dut) == exp


@cocotb.test()
async def test_matmul_random_regression(dut):
    rng = random.Random(12345)

    for tc in range(8):
        a = [[rng.randint(-128, 127) for _ in range(16)] for _ in range(16)]
        b = [[rng.randint(-128, 127) for _ in range(16)] for _ in range(16)]
        exp = _matmul_ref(a, b)

        prog = []
        dram = DramModel()
        prepare_logical_16x16(dram, prog, a, b, 0x1C0000 + tc * 0x4000, 0x1C2000 + tc * 0x4000)
        prog.extend([
            CONFIG_TILE(1, 1, 1),
            MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
            SYNC(0b010),
            HALT(),
        ])
        await setup_test(dut, prog, dram=dram)
        await wait_halt(dut, max_cycles=500_000)

        assert int(dut.done.value) == 1
        assert int(dut.fault.value) == 0
        assert read_accum_16x16(dut) == exp, f"random regression mismatch on case {tc}"


@cocotb.test()
async def test_matmul_k4_boundary_stress(dut):
    a = [[127 if ((i + k) & 1) else -128 for k in range(64)] for i in range(16)]
    b = [[-128 if ((k * 7 + j) & 1) else 127 for j in range(16)] for k in range(64)]
    exp = _matmul_ref_16x64x16(a, b)

    prog = []
    dram = DramModel()
    prepare_logical_16x64x16(dram, prog, a, b, 0x200000, 0x210000)
    prog.extend([
        CONFIG_TILE(1, 1, 4),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=900_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    assert read_accum_16x16(dut) == exp


@cocotb.test()
async def test_matmul_multitile_random_regression_2x2x2(dut):
    rng = random.Random(67890)

    for tc in range(4):
        a = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(32)]
        b = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(32)]
        exp = _matmul_ref_32(a, b)

        prog = []
        dram = DramModel()
        prepare_logical_32x32(dram, prog, a, b, 0x220000 + tc * 0x8000, 0x224000 + tc * 0x8000)
        prog.extend([
            CONFIG_TILE(2, 2, 2),
            MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
            SYNC(0b010),
            HALT(),
        ])
        await setup_test(dut, prog, dram=dram)
        await wait_halt(dut, max_cycles=900_000)

        assert int(dut.done.value) == 1
        assert int(dut.fault.value) == 0
        assert read_accum_32x32(dut) == exp, f"multitile random mismatch on case {tc}"
