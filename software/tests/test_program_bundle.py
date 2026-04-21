"""Stage 3 tests for ProgramBundle layout and idempotent relocation."""
import numpy as np

from taccel.assembler.assembler import ProgramBundle, RelocationSite
from taccel.golden_model.simulator import Simulator
from taccel.isa.encoding import decode, encode
from taccel.isa.instructions import HaltInsn, LoadInsn, SetAddrHiInsn, SetAddrLoInsn, StoreInsn
from taccel.isa.opcodes import BUF_ABUF


def _bytes(*insns) -> bytes:
    out = bytearray()
    for insn in insns:
        out.extend(encode(insn))
    return bytes(out)


def _address_from_stream(stream: bytes, lo_pc: int, hi_pc: int) -> int:
    lo = decode(stream[lo_pc * 8:lo_pc * 8 + 8])
    hi = decode(stream[hi_pc * 8:hi_pc * 8 + 8])
    return (hi.imm28 << 28) | lo.imm28


def _load_stream() -> bytes:
    return _bytes(
        SetAddrLoInsn(addr_reg=0, imm28=0),
        SetAddrHiInsn(addr_reg=0, imm28=0),
        LoadInsn(buf_id=BUF_ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0),
        HaltInsn(),
    )


def _two_addend_load_stream() -> bytes:
    return _bytes(
        SetAddrLoInsn(addr_reg=0, imm28=0),
        SetAddrHiInsn(addr_reg=0, imm28=0),
        LoadInsn(buf_id=BUF_ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0),
        SetAddrLoInsn(addr_reg=1, imm28=16),
        SetAddrHiInsn(addr_reg=1, imm28=0),
        LoadInsn(buf_id=BUF_ABUF, sram_off=1, xfer_len=1, addr_reg=1, dram_off=0),
        HaltInsn(),
    )


def _store_stream() -> bytes:
    return _bytes(
        SetAddrLoInsn(addr_reg=0, imm28=0),
        SetAddrHiInsn(addr_reg=0, imm28=0),
        StoreInsn(buf_id=BUF_ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0),
        HaltInsn(),
    )


def test_program_bundle_layout_and_idempotent_materialize():
    bundle = ProgramBundle(
        prefill_instrs=_load_stream(),
        decode_instrs=_store_stream(),
        shared_data=b"abcdefghijklmnop",
        kv_cache_size=16,
        symbol_offsets={"payload": 0},
        relocation_sites=[
            RelocationSite("prefill", 0, 1, 0, "payload"),
            RelocationSite("decode", 0, 1, 0, "kv_cache_base"),
        ],
    )

    assert bundle.prefill_instrs_offset == 0
    assert bundle.decode_instrs_offset == len(_load_stream())
    assert bundle.data_base % 16 == 0
    assert bundle.temp_base % 16 == 0
    assert bundle.logits_base % 16 == 0
    assert bundle.kv_cache_base % 16 == 0
    assert bundle.temp_base == bundle.data_base + len(bundle.shared_data)
    assert bundle.logits_base == bundle.temp_base
    assert bundle.kv_cache_base == bundle.logits_base
    assert bundle.required_dram_bytes == bundle.kv_cache_base + 16

    first = bundle.materialize()
    second = bundle.materialize()

    assert first == second
    assert _address_from_stream(bundle.stream_bytes("prefill"), 0, 1) == bundle.data_base
    assert _address_from_stream(bundle.stream_bytes("decode"), 0, 1) == bundle.kv_cache_base


def test_program_bundle_relocation_preserves_static_addends():
    bundle = ProgramBundle(
        prefill_instrs=_two_addend_load_stream(),
        decode_instrs=_bytes(HaltInsn()),
        shared_data=b"abcdefghijklmnopABCDEFGHIJKLMNOP",
        relocation_sites=[
            RelocationSite("prefill", 0, 1, 0, "data_base"),
            RelocationSite("prefill", 3, 4, 1, "data_base"),
        ],
    )
    sim = Simulator()
    sim.load_bundle(bundle)

    assert _address_from_stream(bundle.stream_bytes("prefill"), 0, 1) == bundle.data_base
    assert _address_from_stream(bundle.stream_bytes("prefill"), 3, 4) == bundle.data_base + 16

    sim.run_program(bundle, "prefill")

    assert bytes(sim.state.abuf[:16]) == b"abcdefghijklmnop"
    assert bytes(sim.state.abuf[16:32]) == b"ABCDEFGHIJKLMNOP"


def test_run_program_preserves_persistent_memory_and_sram():
    bundle = ProgramBundle(
        prefill_instrs=_load_stream(),
        decode_instrs=_store_stream(),
        shared_data=b"abcdefghijklmnop",
        kv_cache_size=16,
        symbol_offsets={"payload": 0},
        relocation_sites=[
            RelocationSite("prefill", 0, 1, 0, "payload"),
            RelocationSite("decode", 0, 1, 0, "kv_cache_base"),
        ],
    )
    sim = Simulator()
    sim.load_bundle(bundle)

    shared_before = bytes(sim.state.dram[bundle.data_base:bundle.temp_base])
    sim.run_program(bundle, "prefill")
    assert bytes(sim.state.abuf[:16]) == b"abcdefghijklmnop"

    sim.run_program(bundle, "decode")
    assert bytes(sim.state.dram[bundle.data_base:bundle.temp_base]) == shared_before
    assert bytes(sim.state.dram[bundle.kv_cache_base:bundle.kv_cache_base + 16]) == b"abcdefghijklmnop"

    sim.state.abuf[:16] = b"ponmlkjihgfedcba"
    sim.run_program(bundle, "decode")
    assert bytes(sim.state.dram[bundle.kv_cache_base:bundle.kv_cache_base + 16]) == b"ponmlkjihgfedcba"


def test_run_program_resets_volatile_state_but_preserves_sram():
    bundle = ProgramBundle(prefill_instrs=_bytes(HaltInsn()), decode_instrs=_bytes(HaltInsn()))
    sim = Simulator()
    sim.load_bundle(bundle)
    sim.state.tile_config = (0, 0, 0)
    sim.state.attn_context = {"is_valid": True, "query_row_base": 7, "valid_kv_len": 8, "mode": 0b11}
    sim.state.scale_regs[:] = np.float16(3.0)
    sim.state.addr_regs[:] = 123
    sim.state.abuf[:4] = b"keep"
    sim.state.wbuf[:4] = b"stay"
    sim.state.accum[:4] = 42

    sim.run_program(bundle, "prefill")

    assert sim.state.tile_config is None
    assert sim.state.attn_context["is_valid"] is False
    assert np.all(sim.state.scale_regs == 0)
    assert np.all(sim.state.addr_regs == 0)
    assert bytes(sim.state.abuf[:4]) == b"keep"
    assert bytes(sim.state.wbuf[:4]) == b"stay"
    np.testing.assert_array_equal(sim.state.accum[:4], np.array([42, 42, 42, 42], dtype=np.int32))
