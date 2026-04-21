"""Stage 3 tests for Simulator.load_bundle/run_program APIs."""
import pytest

from taccel.assembler.assembler import ProgramBundle
from taccel.golden_model.simulator import Simulator, SimulatorError
from taccel.isa.encoding import encode
from taccel.isa.instructions import HaltInsn, NopInsn, SetAddrHiInsn, SetAddrLoInsn


def _bytes(*insns) -> bytes:
    out = bytearray()
    for insn in insns:
        out.extend(encode(insn))
    return bytes(out)


def test_load_bundle_auto_grows_dram_by_default():
    bundle = ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_bytes(HaltInsn()),
        shared_data=bytes(range(32)),
        kv_cache_size=128,
    )
    sim = Simulator()
    sim.state.dram = bytearray(16)

    sim.load_bundle(bundle)

    assert len(sim.state.dram) >= bundle.required_dram_bytes
    assert bytes(sim.state.dram[bundle.data_base:bundle.data_base + 32]) == bytes(range(32))


def test_load_bundle_strict_dram_size_rejects_undersized_state():
    bundle = ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_bytes(HaltInsn()),
        shared_data=bytes(range(32)),
        kv_cache_size=128,
    )
    sim = Simulator()
    sim.state.dram = bytearray(16)

    with pytest.raises(SimulatorError, match="strict DRAM size"):
        sim.load_bundle(bundle, strict_dram_size=True)


def test_load_bundle_respects_max_dram_cap():
    bundle = ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_bytes(HaltInsn()),
        shared_data=bytes(range(32)),
        kv_cache_size=128,
    )
    sim = Simulator()
    sim.state.dram = bytearray(16)

    with pytest.raises(SimulatorError, match="exceeding cap"):
        sim.load_bundle(bundle, max_dram_bytes=bundle.required_dram_bytes - 1)


def test_run_program_starts_at_selected_stream_pc():
    bundle = ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_bytes(
            NopInsn(),
            SetAddrLoInsn(addr_reg=2, imm28=0x2A),
            SetAddrHiInsn(addr_reg=2, imm28=0),
            HaltInsn(),
        ),
    )
    sim = Simulator()
    sim.load_bundle(bundle)

    count = sim.run_program(bundle, "decode")

    assert count == 4
    assert int(sim.state.addr_regs[2]) == 0x2A
    assert sim.state.current_pc == bundle.decode_pc + 3
