"""Shared cocotb testbench helpers for TACCEL RTL verification.

These helpers standardize the common verification pattern used across the
owner benches:
  - start clock
  - reset/start the DUT
  - preload DRAM program/data
  - launch AXI read/write slave coroutines
  - wait for architectural halt or fault with a bounded timeout
"""

from __future__ import annotations

from typing import Iterable

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from utils.dram_model import DramModel
from utils.insn_builder import SET_ADDR_LO, SET_ADDR_HI


async def setup_test(
    dut,
    insns: Iterable[int],
    dram: DramModel | None = None,
    dram_writes: dict[int, bytes | bytearray] | None = None,
    clk_period_ns: int = 10,
    start_write_slave: bool = True,
    write_slave_kwargs: dict | None = None,
) -> DramModel:
    """Start the standard DUT harness and optionally preload DRAM bytes."""
    cocotb.start_soon(Clock(dut.clk, clk_period_ns, units="ns").start())
    if dram is None:
        dram = DramModel()
    dram.write_program(list(insns))
    if dram_writes:
        for addr, data in dram_writes.items():
            dram.write_bytes(addr, bytes(data))

    cocotb.start_soon(dram.axi_slave(dut))
    if start_write_slave:
        cocotb.start_soon(dram.axi_write_slave(dut, **(write_slave_kwargs or {})))

    dut.rst_n.value = 0
    dut.start.value = 0
    dut.m_axi_ar_ready.value = 1
    dut.m_axi_r_valid.value = 0
    dut.m_axi_r_data.value = 0
    dut.m_axi_r_resp.value = 0
    dut.m_axi_r_last.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    return dram


async def wait_halt(dut, max_cycles: int = 100_000):
    """Wait until the DUT raises done or fault, else timeout."""
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) or int(dut.fault.value):
            return
    raise TimeoutError("DUT did not halt")


async def run_program(
    dut,
    insns: Iterable[int],
    *,
    dram: DramModel | None = None,
    dram_writes: dict[int, bytes | bytearray] | None = None,
    timeout_cycles: int = 100_000,
    clk_period_ns: int = 10,
    start_write_slave: bool = True,
    write_slave_kwargs: dict | None = None,
):
    """Convenience wrapper: setup, run, then return (dram, done, fault, code)."""
    dram = await setup_test(
        dut,
        insns,
        dram=dram,
        dram_writes=dram_writes,
        clk_period_ns=clk_period_ns,
        start_write_slave=start_write_slave,
        write_slave_kwargs=write_slave_kwargs,
    )
    await wait_halt(dut, timeout_cycles)
    return dram, int(dut.done.value), int(dut.fault.value), int(dut.fault_code.value)


def set_addr(reg: int, addr: int) -> list[int]:
    """Return SET_ADDR_LO/HI pair for a 56-bit architectural address register."""
    return [
        SET_ADDR_LO(reg, addr & 0x0FFFFFFF),
        SET_ADDR_HI(reg, (addr >> 28) & 0x0FFFFFFF),
    ]


def pattern(nbytes: int, seed: int) -> bytes:
    """Deterministic byte pattern for data-movement tests."""
    return bytes((seed + 37 * i) & 0xFF for i in range(nbytes))


def accum_row_u32x4(dut, row: int) -> list[int]:
    v = int(dut.u_sram.u_accum.mem[row].value)
    return [
        (v >> 0) & 0xFFFFFFFF,
        (v >> 32) & 0xFFFFFFFF,
        (v >> 64) & 0xFFFFFFFF,
        (v >> 96) & 0xFFFFFFFF,
    ]


def read_accum_16x16(dut, dst_off: int = 0) -> list[list[int]]:
    out = [[0 for _ in range(16)] for _ in range(16)]
    for i in range(16):
        for grp in range(4):
            row = dst_off + i * 4 + grp
            lanes = accum_row_u32x4(dut, row)
            for lane in range(4):
                j = grp * 4 + lane
                u = lanes[lane]
                out[i][j] = u if u < (1 << 31) else (u - (1 << 32))
    return out


def read_accum_32x32(dut, dst_off: int = 0) -> list[list[int]]:
    out = [[0 for _ in range(32)] for _ in range(32)]
    for i in range(32):
        for j in range(32):
            grp, lane = j // 4, j % 4
            row = dst_off + i * 8 + grp
            lanes = accum_row_u32x4(dut, row)
            u = lanes[lane]
            out[i][j] = u if u < (1 << 31) else (u - (1 << 32))
    return out
