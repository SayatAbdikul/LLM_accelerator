"""Stage 3 tests for ProgramBundle runtime patch sites."""
from taccel.assembler.assembler import ProgramBundle, RuntimePatchSite
from taccel.golden_model.simulator import Simulator
from taccel.isa.encoding import encode
from taccel.isa.instructions import HaltInsn, SetAddrHiInsn, SetAddrLoInsn


def _bytes(*insns) -> bytes:
    out = bytearray()
    for insn in insns:
        out.extend(encode(insn))
    return bytes(out)


def _three_patch_stream() -> bytes:
    return _bytes(
        SetAddrLoInsn(addr_reg=0, imm28=0),
        SetAddrHiInsn(addr_reg=0, imm28=0),
        SetAddrLoInsn(addr_reg=1, imm28=0),
        SetAddrHiInsn(addr_reg=1, imm28=0),
        SetAddrLoInsn(addr_reg=2, imm28=0),
        SetAddrHiInsn(addr_reg=2, imm28=0),
        HaltInsn(),
    )


def _bundle() -> ProgramBundle:
    return ProgramBundle(
        prefill_instrs=_bytes(HaltInsn()),
        decode_instrs=_three_patch_stream(),
        shared_data=bytes(range(64)),
        temp_size=16,
        logits_size=16,
        kv_cache_size=32,
        symbol_offsets={"token_table": 16, "pos_table": 32},
        runtime_patch_sites=[
            RuntimePatchSite("decode", "token_embed", 0, 1, 0, 0, 0, "token_table"),
            RuntimePatchSite("decode", "pos_embed", 2, 3, 0, 0, 1, "pos_table"),
            RuntimePatchSite("decode", "kv_base", 4, 5, 0, 0, 2, "kv_cache_base"),
        ],
    )


def test_runtime_patch_sites_patch_selected_stream_only():
    bundle = _bundle()
    sim = Simulator()
    sim.load_bundle(bundle)
    prefill_before = bundle.stream_bytes("prefill")
    decode_before = bundle.stream_bytes("decode")
    shared_before = bytes(sim.state.dram[bundle.data_base:bundle.temp_base])
    temp_before = bytes(sim.state.dram[bundle.temp_base:bundle.logits_base])
    logits_before = bytes(sim.state.dram[bundle.logits_base:bundle.kv_cache_base])
    kv_before = bytes(sim.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes])

    bundle.patch_runtime_site("token_embed", 5, stream="decode")
    bundle.patch_runtime_site("pos_embed", 7, stream="decode")
    bundle.patch_runtime_site("kv_base", 9, stream="decode")
    sim.run_program(bundle, "decode")

    assert bundle.stream_bytes("prefill") == prefill_before
    assert bundle.stream_bytes("decode") != decode_before
    assert int(sim.state.addr_regs[0]) == bundle.symbol_address("token_table") + 5
    assert int(sim.state.addr_regs[1]) == bundle.symbol_address("pos_table") + 7
    assert int(sim.state.addr_regs[2]) == bundle.kv_cache_base + 9
    assert bytes(sim.state.dram[bundle.data_base:bundle.temp_base]) == shared_before
    assert bytes(sim.state.dram[bundle.temp_base:bundle.logits_base]) == temp_before
    assert bytes(sim.state.dram[bundle.logits_base:bundle.kv_cache_base]) == logits_before
    assert bytes(sim.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes]) == kv_before


def test_runtime_patch_site_absolute_pcs_are_derived_from_layout():
    bundle = _bundle()
    site_by_kind = {site.kind: site for site in bundle.runtime_patch_sites}

    assert site_by_kind["token_embed"].absolute_lo_pc == bundle.decode_pc
    assert site_by_kind["pos_embed"].absolute_lo_pc == bundle.decode_pc + 2
    assert site_by_kind["kv_base"].absolute_lo_pc == bundle.decode_pc + 4
