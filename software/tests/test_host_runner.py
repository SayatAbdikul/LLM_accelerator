"""Stage 3 tests for the golden-model HostRunner."""
import numpy as np

from taccel.assembler.assembler import (
    ProgramBundle,
    RelocationSite,
    RuntimeConfigAttnSite,
    RuntimePatchSite,
)
from taccel.isa.encoding import decode, encode
from taccel.isa.instructions import (
    ConfigAttnInsn,
    ConfigTileInsn,
    HaltInsn,
    LoadInsn,
    SetAddrHiInsn,
    SetAddrLoInsn,
    StoreInsn,
)
from taccel.runtime.host_runner import HostRunner


def _bytes(*insns):
    out = bytearray()
    for insn in insns:
        out.extend(encode(insn))
    return bytes(out)


def _addr_pair(addr_reg):
    return SetAddrLoInsn(addr_reg=addr_reg, imm28=0), SetAddrHiInsn(addr_reg=addr_reg, imm28=0)


def _row_with_argmax(idx):
    row = np.zeros(16, dtype=np.int8)
    row[(idx + 1) % 16] = 100
    return row


def _bundle():
    token_table = np.vstack([_row_with_argmax(i) for i in range(16)]).astype(np.int8)
    pos_table = (np.arange(16 * 16, dtype=np.int16).reshape(16, 16) - 64).astype(np.int8)
    shared_data = token_table.tobytes() + pos_table.tobytes()
    token_off = 0
    pos_off = len(token_table.tobytes())

    p_tok_lo, p_tok_hi = _addr_pair(0)
    p_pos_lo, p_pos_hi = _addr_pair(1)
    p_log_lo, p_log_hi = _addr_pair(2)
    prefill = _bytes(
        p_tok_lo, p_tok_hi, LoadInsn(buf_id=0, sram_off=0, xfer_len=1, addr_reg=0),
        p_pos_lo, p_pos_hi, LoadInsn(buf_id=0, sram_off=1, xfer_len=1, addr_reg=1),
        p_log_lo, p_log_hi, StoreInsn(buf_id=0, sram_off=0, xfer_len=1, addr_reg=2),
        HaltInsn(),
    )

    d_tok_lo, d_tok_hi = _addr_pair(0)
    d_pos_lo, d_pos_hi = _addr_pair(1)
    d_kv_lo, d_kv_hi = _addr_pair(2)
    d_log_lo, d_log_hi = _addr_pair(3)
    decode_stream = _bytes(
        d_tok_lo, d_tok_hi, LoadInsn(buf_id=0, sram_off=0, xfer_len=1, addr_reg=0),
        d_pos_lo, d_pos_hi, LoadInsn(buf_id=0, sram_off=1, xfer_len=1, addr_reg=1),
        d_kv_lo, d_kv_hi, LoadInsn(buf_id=0, sram_off=2, xfer_len=1, addr_reg=2),
        d_log_lo, d_log_hi, StoreInsn(buf_id=0, sram_off=0, xfer_len=1, addr_reg=3),
        ConfigTileInsn(M=0, N=0, K=0),
        ConfigAttnInsn(query_row_base=0, valid_kv_len=1, mode=0b11),
        HaltInsn(),
    )

    return ProgramBundle(
        prefill_instrs=prefill,
        decode_instrs=decode_stream,
        shared_data=shared_data,
        logits_size=16,
        kv_cache_size=16 * 8,
        embedding_row_bytes=16,
        kv_step_bytes=16,
        symbol_offsets={
            "transformer.wte.weight": token_off,
            "transformer.wpe.weight": pos_off,
            "kv_bank0": 0,
        },
        symbol_regions={"kv_bank0": "kv_cache"},
        relocation_sites=[
            RelocationSite("prefill", 6, 7, 2, "prefill_logits_offset"),
            RelocationSite("decode", 9, 10, 3, "decode_logits_offset"),
        ],
        runtime_patch_sites=[
            RuntimePatchSite("prefill", "token_embed", 0, 1, 0, 0, 0, "transformer.wte.weight"),
            RuntimePatchSite("prefill", "pos_embed", 3, 4, 0, 0, 1, "transformer.wpe.weight"),
            RuntimePatchSite("decode", "token_embed", 0, 1, 0, 0, 0, "transformer.wte.weight"),
            RuntimePatchSite("decode", "pos_embed", 3, 4, 0, 0, 1, "transformer.wpe.weight"),
            RuntimePatchSite("decode", "kv_base", 6, 7, 0, 0, 2, "kv_bank0"),
        ],
        runtime_config_attn_sites=[
            RuntimeConfigAttnSite("decode", local_pc=13, absolute_pc=0, mode=0b11),
        ],
    ), token_table, pos_table


def _patched_addr(stream, pc):
    lo = decode(stream[pc * 8:pc * 8 + 8])
    hi = decode(stream[(pc + 1) * 8:(pc + 2) * 8])
    return (int(hi.imm28) << 28) | int(lo.imm28)


def test_host_runner_patches_prefill_decode_and_preserves_persistent_regions():
    bundle, token_table, pos_table = _bundle()
    runner = HostRunner(bundle, logits_dtype=np.int8)
    kv_row = bytes(range(16))
    runner.simulator.state.dram[bundle.kv_cache_base + 2 * 16:bundle.kv_cache_base + 3 * 16] = kv_row
    shared_before = bytes(runner.simulator.state.dram[bundle.data_base:bundle.temp_base])
    cache_before = bytes(runner.simulator.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes])

    prefill_logits = runner.run_prefill([1])
    assert np.array_equal(prefill_logits, token_table[1])

    decode_before = bundle.stream_bytes("decode")
    decode_logits = runner.run_decode_step(3, 2)
    assert bundle.stream_bytes("decode") != decode_before
    assert np.array_equal(decode_logits, token_table[3])
    assert bytes(runner.simulator.state.abuf[16:32]) == pos_table[2].tobytes()
    assert bytes(runner.simulator.state.abuf[32:48]) == kv_row
    assert bytes(runner.simulator.state.dram[bundle.data_base:bundle.temp_base]) == shared_before
    assert bytes(runner.simulator.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes]) == cache_before

    stream = bundle.stream_bytes("decode")
    assert _patched_addr(stream, 0) == bundle.symbol_address("transformer.wte.weight") + 3 * 16
    assert _patched_addr(stream, 3) == bundle.symbol_address("transformer.wpe.weight") + 2 * 16
    assert _patched_addr(stream, 6) == bundle.symbol_address("kv_bank0") + 2 * 16
    config_attn = decode(stream[13 * 8:14 * 8])
    assert config_attn.query_row_base == 2
    assert config_attn.valid_kv_len == 3
    assert config_attn.mode == 0b11


def test_generate_greedy_returns_prompt_plus_exact_new_token_count():
    bundle, _, _ = _bundle()
    runner = HostRunner(bundle, logits_dtype=np.int8)

    generated = runner.generate([1], max_new_tokens=3)

    assert generated == [1, 2, 3, 4]
