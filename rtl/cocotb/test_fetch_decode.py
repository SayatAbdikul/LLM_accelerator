"""cocotb tests for Phase 1: fetch-decode-execute pipeline.

Tests supported setup/dispatch paths plus Phase A fault behavior for illegal
opcodes, unsupported legal instructions, and fetch-side AXI read faults.

Each test:
  1. Loads instructions into the DramModel
  2. Starts the AXI slave coroutine
  3. Asserts start, runs until done/fault
  4. Checks output signals
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, First
from cocotb.result import SimTimeoutError
import sys, os

# Add cocotb utils to path
sys.path.insert(0, os.path.dirname(__file__))
from utils.dram_model import DramModel
from utils.insn_builder import (
    NOP, HALT, SYNC, CONFIG_TILE, SET_SCALE, SET_ADDR_LO, SET_ADDR_HI,
    LOAD, MATMUL, ILLEGAL_OP, BUF_COPY, REQUANT_PC, SCALE_MUL, DEQUANT_ADD, SOFTMAX_ATTNV,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM
)

CLK_PERIOD_NS = 5  # 200 MHz


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def init_dut(dut):
    """Start clock, reset DUT, return DramModel and background task."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    # Drive AXI slave stubs to safe defaults
    dut.m_axi_ar_ready.value = 1
    dut.m_axi_r_valid.value  = 0
    dut.m_axi_r_data.value   = 0
    dut.m_axi_r_resp.value   = 0
    dut.m_axi_r_last.value   = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dram = DramModel()
    slave_task = cocotb.start_soon(dram.axi_slave(dut))
    return dram, slave_task


async def run_program(dut, dram, insns, timeout_cycles=2000):
    """Load program, start DUT, return (done, fault, fault_code, cycles)."""
    dram.write_program(insns)
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for cycle in range(timeout_cycles):
        await RisingEdge(dut.clk)
        done  = int(dut.done.value)
        fault = int(dut.fault.value)
        if done or fault:
            return done, fault, int(dut.fault_code.value), cycle + 1

    raise TimeoutError(f"DUT did not halt within {timeout_cycles} cycles")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_nop_halt(dut):
    """NOP -> HALT: done=1, fault=0."""
    dram, _ = await init_dut(dut)
    done, fault, _, cycles = await run_program(dut, dram, [NOP(), HALT()])
    assert done == 1,  f"Expected done=1, got {done}"
    assert fault == 0, f"Expected fault=0, got {fault}"
    dut._log.info(f"NOP+HALT completed in {cycles} cycles")


@cocotb.test()
async def test_immediate_halt(dut):
    """HALT only: done=1 immediately."""
    dram, _ = await init_dut(dut)
    done, fault, _, _ = await run_program(dut, dram, [HALT()])
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_multi_nop_then_halt(dut):
    """5 NOPs then HALT."""
    dram, _ = await init_dut(dut)
    prog = [NOP()] * 5 + [HALT()]
    done, fault, _, cycles = await run_program(dut, dram, prog)
    assert done == 1
    assert fault == 0
    dut._log.info(f"5xNOP+HALT: {cycles} cycles")


@cocotb.test()
async def test_config_tile_then_halt(dut):
    """CONFIG_TILE M=3,N=5,K=12 -> HALT: no fault."""
    dram, _ = await init_dut(dut)
    done, fault, _, _ = await run_program(dut, dram, [CONFIG_TILE(3, 5, 12), HALT()])
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_set_scale_then_halt(dut):
    """SET_SCALE S0=0x3C00 (1.0 FP16) -> HALT: no fault."""
    dram, _ = await init_dut(dut)
    done, fault, _, _ = await run_program(dut, dram, [SET_SCALE(0, 0x3C00), HALT()])
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_set_addr_then_halt(dut):
    """SET_ADDR_LO + SET_ADDR_HI -> HALT: no fault."""
    dram, _ = await init_dut(dut)
    prog = [
        SET_ADDR_LO(0, 0x0100000),
        SET_ADDR_HI(0, 0x0000000),
        HALT()
    ]
    done, fault, _, _ = await run_program(dut, dram, prog)
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_sync_zero_mask(dut):
    """SYNC 0b000 -> immediately passes (no units to wait for)."""
    dram, _ = await init_dut(dut)
    done, fault, _, _ = await run_program(dut, dram, [SYNC(0b000), HALT()])
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_sync_all_units_idle(dut):
    """SYNC 0b111 -> passes immediately when all units are idle."""
    dram, _ = await init_dut(dut)
    done, fault, _, cycles = await run_program(dut, dram, [SYNC(0b111), HALT()])
    assert done == 1
    assert fault == 0
    dut._log.info(f"SYNC(111) with all idle: {cycles} cycles")


@cocotb.test()
async def test_config_tile_then_matmul_no_fault(dut):
    """CONFIG_TILE -> MATMUL -> SYNC -> HALT: no fault (MATMUL dispatches, SYS stub)."""
    dram, _ = await init_dut(dut)
    prog = [
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),   # wait for systolic (immediately clear, sys_busy=0)
        HALT()
    ]
    done, fault, _, _ = await run_program(dut, dram, prog)
    assert done == 1,  f"Expected done=1"
    assert fault == 0, f"No fault expected when CONFIG_TILE precedes MATMUL"


@cocotb.test()
async def test_load_dispatch_no_stall(dut):
    """LOAD dispatched (DMA stub) -> SYNC(001) clears immediately -> HALT."""
    dram, _ = await init_dut(dut)
    prog = [
        SET_ADDR_LO(0, 0x0),
        SET_ADDR_HI(0, 0x0),
        LOAD(BUF_ABUF, 0, 16, 0, 0),
        SYNC(0b001),
        HALT()
    ]
    done, fault, _, _ = await run_program(dut, dram, prog)
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_illegal_opcode_fault(dut):
    """Reserved opcode 0x14 -> fault=1, fault_code=1 (FAULT_ILLEGAL_OP)."""
    dram, _ = await init_dut(dut)
    done, fault, fault_code, _ = await run_program(dut, dram, [ILLEGAL_OP()])
    assert fault == 1,       f"Expected fault=1, got {fault}"
    assert done  == 0,       f"Expected done=0 on fault, got {done}"
    assert fault_code == 1,  f"Expected FAULT_ILLEGAL_OP=1, got {fault_code}"
    dut._log.info(f"Illegal opcode fault_code={fault_code}")


@cocotb.test()
async def test_buf_copy_same_buffer_transpose_fault(dut):
    """Same-buffer transpose remains intentionally unsupported in Phase C."""
    dram, _ = await init_dut(dut)
    done, fault, fault_code, _ = await run_program(
        dut, dram,
        [BUF_COPY(BUF_ABUF, 0, BUF_ABUF, 16, 16, 1, 1)]
    )
    assert done == 0
    assert fault == 1
    assert fault_code == 6, f"Expected FAULT_UNSUPPORTED_OP=6, got {fault_code}"


@cocotb.test()
async def test_stage_e_ops_dispatch_no_fault(dut):
    """Stage E helper/SFU instructions now execute instead of faulting unsupported."""
    dram, _ = await init_dut(dut)
    prog = [
        CONFIG_TILE(1, 1, 1),
        REQUANT_PC(BUF_ACCUM, 0, BUF_WBUF, 0, BUF_ABUF, 0, sreg=0, flags=0),
        SET_SCALE(2, 0x3800),
        SCALE_MUL(BUF_ABUF, 0, BUF_WBUF, 32, sreg=2, flags=0),
        SET_SCALE(4, 0x2C00),
        SET_SCALE(5, 0x3400),
        DEQUANT_ADD(BUF_ACCUM, 0, BUF_ABUF, 0, BUF_WBUF, 64, sreg=4, flags=0),
        SET_SCALE(8, 0x3400),
        SET_SCALE(9, 0x3400),
        SET_SCALE(10, 0x3400),
        SET_SCALE(11, 0x3000),
        SOFTMAX_ATTNV(BUF_ACCUM, 0, BUF_ABUF, 0, BUF_WBUF, 96, sreg=8, flags=0),
        SYNC(0b100),
        HALT(),
    ]
    done, fault, _, _ = await run_program(dut, dram, prog, timeout_cycles=20000)
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_set_scale_from_buffer_fault(dut):
    """SET_SCALE src_mode != 0 is intentionally rejected in Phase A."""
    dram, _ = await init_dut(dut)
    done, fault, fault_code, _ = await run_program(dut, dram, [SET_SCALE(0, 7, src_mode=1)])
    assert done == 0
    assert fault == 1
    assert fault_code == 6, f"Expected FAULT_UNSUPPORTED_OP=6, got {fault_code}"


@cocotb.test()
async def test_multiburst_load_supported(dut):
    """Phase B accepts LOAD lengths above 256 beats and completes via SYNC."""
    dram, _ = await init_dut(dut)
    prog = [
        SET_ADDR_LO(0, 0),
        SET_ADDR_HI(0, 0),
        LOAD(BUF_ABUF, 0, 257, 0, 0),
        SYNC(0b001),
        HALT(),
    ]
    done, fault, _, _ = await run_program(dut, dram, prog)
    assert done == 1
    assert fault == 0


@cocotb.test()
async def test_matmul_without_config_tile_fault(dut):
    """MATMUL without preceding CONFIG_TILE -> fault_code=4 (FAULT_NO_CONFIG)."""
    dram, _ = await init_dut(dut)
    done, fault, fault_code, _ = await run_program(
        dut, dram,
        [MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0)]
    )
    assert fault == 1,       f"Expected fault=1"
    assert fault_code == 4,  f"Expected FAULT_NO_CONFIG=4, got {fault_code}"


@cocotb.test()
async def test_fetch_rresp_fault(dut):
    """Fetch maps non-OKAY RRESP to FAULT_DRAM_OOB."""
    dram, _ = await init_dut(dut)
    dram.write_program([NOP(), HALT()])
    dram.inject_next_read(resp=2)
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(200):
        await RisingEdge(dut.clk)
        if int(dut.fault.value):
            assert int(dut.done.value) == 0
            assert int(dut.fault_code.value) == 2
            return
    raise AssertionError("Expected fetch RRESP fault")


@cocotb.test()
async def test_fetch_missing_rlast_fault(dut):
    """Single-beat fetch without RLAST is treated as malformed and faults."""
    dram, _ = await init_dut(dut)
    dram.write_program([NOP(), HALT()])
    dram.inject_next_read(resp=0, force_last=0)
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(200):
        await RisingEdge(dut.clk)
        if int(dut.fault.value):
            assert int(dut.done.value) == 0
            assert int(dut.fault_code.value) == 2
            return
    raise AssertionError("Expected missing-RLAST fetch fault")


@cocotb.test()
async def test_full_sequence_addr_scale_config_matmul(dut):
    """Full Phase 1 sequence: SET_ADDR + SET_SCALE + CONFIG_TILE + MATMUL + SYNC + HALT."""
    dram, _ = await init_dut(dut)
    prog = [
        SET_ADDR_LO(0, 0x0000000),
        SET_ADDR_HI(0, 0x0000000),
        SET_SCALE(0, 0x3C00),          # S0 = 1.0 (in_scale)
        SET_SCALE(1, 0x3C00),          # S1 = 1.0 (out_scale for SFU)
        CONFIG_TILE(1, 1, 1),
        MATMUL(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT()
    ]
    done, fault, _, cycles = await run_program(dut, dram, prog)
    assert done  == 1
    assert fault == 0
    dut._log.info(f"Full Phase 1 sequence: {cycles} cycles")


@cocotb.test()
async def test_ten_nops_halt(dut):
    """10 NOPs + HALT: all execute without fault."""
    dram, _ = await init_dut(dut)
    prog = [NOP()] * 10 + [HALT()]
    done, fault, _, cycles = await run_program(dut, dram, prog, timeout_cycles=5000)
    assert done  == 1
    assert fault == 0
    dut._log.info(f"10xNOP+HALT: {cycles} cycles")
