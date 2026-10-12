"""Stage 3 tests for the golden-model HostRunner."""
from copy import deepcopy

import numpy as np
import pytest

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

    generated = runner.generate([1], max_new_tokens=3, vocab_size=16)

    assert generated == [1, 2, 3, 4]


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("commit", [False, True])
def test_generate_only_executes_consumed_tokens(count, commit, monkeypatch):
    bundle, _, _ = _bundle()
    runner = HostRunner(bundle, logits_dtype=np.int8)
    calls = []
    original = runner.run_decode_step

    def decode_step(token, position):
        calls.append((token, position))
        return original(token, position)

    monkeypatch.setattr(runner, "run_decode_step", decode_step)
    generated = runner.generate([1], count, final_token_commit=commit, vocab_size=16)
    assert generated == list(range(1, count + 2))
    executed = count if commit else max(0, count - 1)
    assert calls == [(i + 2, i + 1) for i in range(executed)]


@pytest.mark.parametrize("commit", [False, True])
def test_generate_eos_commit_is_explicit(commit, monkeypatch):
    bundle, _, _ = _bundle()
    runner = HostRunner(bundle, logits_dtype=np.int8)
    calls = []
    original = runner.run_decode_step

    def decode_step(token, position):
        calls.append((token, position))
        return original(token, position)

    monkeypatch.setattr(runner, "run_decode_step", decode_step)
    assert runner.generate([1], 8, eos_token_id=3, final_token_commit=commit, vocab_size=16) == [1, 2, 3]
    assert calls == ([(2, 1), (3, 2)] if commit else [(2, 1)])


def test_uncommitted_generation_can_continue_with_identical_machine_state():
    bundle_a, _, _ = _bundle()
    bundle_b, _, _ = _bundle()
    lazy = HostRunner(bundle_a, logits_dtype=np.int8, vocab_size=16)
    committed = HostRunner(bundle_b, logits_dtype=np.int8, vocab_size=16)
    tokens = lazy.generate([1], 3)
    assert tokens == committed.generate([1], 3, final_token_commit=True)
    next_logits = lazy.run_decode_step(tokens[-1], len(tokens) - 1)
    assert np.array_equal(next_logits, committed._read_logits(bundle_b.decode_logits_offset, rows=1))
    assert lazy.simulator.state.dram == committed.simulator.state.dram
    assert lazy.simulator.state.abuf == committed.simulator.state.abuf


def test_sampling_excludes_padding_and_rejects_nonfinite_live_logits():
    row = np.r_[np.full(33, -1, dtype=np.float16), np.zeros(15, dtype=np.float16)]
    row[7] = -0.5
    row[33:] = np.nan  # padding is neither sampled nor included in finite checks
    assert HostRunner._greedy_token(row, vocab_size=33) == 7
    for bad in (np.nan, np.inf, -np.inf):
        row[1] = bad
        with pytest.raises(ValueError, match="non-finite"):
            HostRunner._greedy_token(row, vocab_size=33)
    with pytest.raises(ValueError, match="one logits row"):
        HostRunner._greedy_token(np.zeros((2, 48)), vocab_size=33)
    with pytest.raises(ValueError, match="covering"):
        HostRunner._greedy_token(np.zeros(16), vocab_size=33)
    for size in (0, -1, True, 3.5):
        with pytest.raises(ValueError, match="positive integer"):
            HostRunner._greedy_token(np.zeros(16), vocab_size=size)


def test_generate_requires_live_vocab_and_rejects_ambiguous_rows():
    bundle, _, _ = _bundle()
    runner = HostRunner(bundle, logits_dtype=np.int8)
    with pytest.raises(ValueError, match="explicit vocab_size"):
        runner.generate([1], 1)
    assert runner.generate([1], 0) == [1]
    bundle.logits_rows = 2
    with pytest.raises(ValueError, match="single-stream"):
        runner.generate([1], 1, vocab_size=16)


def test_real_decoder_final_commit_and_continuation_preserve_written_kv():
    from tools.train_tiny_fixture import _make_checkpoint_payload
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    payload = _make_checkpoint_payload(n_layer=1, n_head=2, d_model=32,
                                       block_size=16, weight_scale=0.02)
    left = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=8)
    right = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=8)
    lazy = HostRunner(left.build.bundle, logits_dtype=np.float16, vocab_size=left.config.vocab_size)
    committed = HostRunner(right.build.bundle, logits_dtype=np.float16, vocab_size=right.config.vocab_size)
    lo, hi = left.build.bundle.kv_cache_base, left.build.bundle.required_dram_bytes
    initial_kv = bytes(lazy.simulator.state.dram[lo:hi])
    tokens = lazy.generate([0], 3)
    assert tokens == committed.generate([0], 3, final_token_commit=True)
    before_commit = bytes(lazy.simulator.state.dram[lo:hi])
    assert before_commit != initial_kv, "fixture must execute nontrivial KV stores"
    assert before_commit != bytes(committed.simulator.state.dram[lo:hi]), "last token must write a new KV row"
    resumed = lazy.run_decode_step(tokens[-1], len(tokens) - 1)
    expected = committed._read_logits(right.build.bundle.decode_logits_offset, rows=1)
    np.testing.assert_array_equal(resumed, expected)
    assert lazy.simulator.state.dram == committed.simulator.state.dram
    next_token = HostRunner._greedy_token(resumed, vocab_size=left.config.vocab_size)
    np.testing.assert_array_equal(lazy.run_decode_step(next_token, len(tokens)),
                                  committed.run_decode_step(next_token, len(tokens)))
    assert lazy.simulator.state.dram == committed.simulator.state.dram


@pytest.fixture(scope="module")
def prefill_layout_cases():
    import torch
    from tools.train_tiny_fixture import _make_checkpoint_payload
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    payload = _make_checkpoint_payload(n_layer=1, n_head=4, d_model=64,
                                       block_size=64, weight_scale=0.02)
    # The stock fixture has zero position embeddings, which would conceal
    # incorrect lockstep position IDs even if the patched bytes differ.
    payload["state_dict"]["transformer.wpe.weight"] = torch.randn(
        64, 64, generator=torch.Generator().manual_seed(73)) * 0.01
    return {(batch, chunk): build_stage3_tiny_decoder_bundle(
        payload, batch=batch, prefill_tokens=chunk, smoke_decode_steps=31,
        calibration_scales={}) for batch, chunk in ((1, 1), (16, 1), (32, 1), (1, 16))}


@pytest.mark.parametrize("batch,chunk", [(1, 1), (16, 1), (32, 1), (1, 16)])
def test_prefill_position_and_attention_patch_bytes(prefill_layout_cases, batch, chunk, monkeypatch):
    tiny = prefill_layout_cases[batch, chunk]
    runner = HostRunner(deepcopy(tiny.build.bundle), logits_dtype=np.float16)
    rows = batch if batch > 1 else chunk
    tokens = [(7 * row) % tiny.config.vocab_size for row in range(rows)]
    runner._patch_embeddings("prefill", tokens, list(range(rows)))
    old_positions = runner.bundle.stream_bytes("prefill")
    runner._patch_kv_bases(7, stream="prefill")
    runner._patch_attention_context("prefill", 7, 8)
    monkeypatch.setattr(runner.simulator, "run_program", lambda *args, **kwargs: None)
    runner.run_prefill(tokens)
    stream = runner.bundle.stream_bytes("prefill")
    expected_positions = [0] * rows if batch > 1 else list(range(rows))
    sites = runner._sites("pos_embed", "prefill")
    assert len(sites) == rows
    for site, position in zip(sites, expected_positions):
        expected = runner.bundle.symbol_address(site.base_symbol) + position * runner.bundle.embedding_row_bytes
        assert _patched_addr(stream, site.local_lo_pc) == expected
    if batch > 1:
        # Negative control: the old range(batch) priming is observably wrong.
        assert _patched_addr(old_positions, sites[1].local_lo_pc) != _patched_addr(stream, sites[1].local_lo_pc)
    for site in runner._sites("kv_base", "prefill"):
        assert _patched_addr(stream, site.local_lo_pc) == runner.bundle.symbol_address(site.base_symbol)
    for site in runner.bundle.runtime_config_attn_sites:
        if site.stream == "prefill":
            insn = decode(stream[site.local_pc * 8:(site.local_pc + 1) * 8])
            assert (insn.query_row_base, insn.valid_kv_len) == (0, 1 if batch > 1 else chunk)
            assert insn.mode == site.mode


@pytest.mark.parametrize("batch", [16, 32])
def test_lockstep_bundle_rejects_sequential_prefill_chunk_without_mutation(prefill_layout_cases, batch):
    runner = HostRunner(deepcopy(prefill_layout_cases[batch, 1].build.bundle), logits_dtype=np.float16)
    before = runner.bundle.stream_bytes("prefill")
    dram = bytes(runner.simulator.state.dram)
    with pytest.raises(ValueError, match="single-stream.*lockstep"):
        runner.run_prefill_chunk([0] * batch, 0)
    assert runner.bundle.stream_bytes("prefill") == before
    assert bytes(runner.simulator.state.dram) == dram


@pytest.mark.parametrize("batch", [16, 32])
def test_lockstep_prime_and_decode_match_explicit_matrix_patching(prefill_layout_cases, batch):
    from tools.optimization_matrix import Step, build_program, inspect_output
    from taccel.golden_model.simulator import Simulator

    tiny = deepcopy(prefill_layout_cases[batch, 1])
    runner = HostRunner(deepcopy(tiny.build.bundle), logits_dtype=np.float16)
    kv_image = None
    for position, stream in ((0, "prefill"), (1, "decode")):
        pb, layout = build_program(tiny, Step("test", stream, position, batch, True), kv_image)
        sim = Simulator()
        sim.load_program(pb)
        sim.run()
        raw = bytes(sim.state.dram[layout["data_base"]:layout["data_base"] + layout["data_bytes"]])
        _, kv_image = inspect_output(raw, layout, tiny.build.bundle.shared_data)
        output = (runner.run_prefill(layout["tokens"]) if position == 0 else
                  runner.run_decode_step_batch(layout["tokens"], position))
        start = layout["logits_offset"]
        expected = np.frombuffer(raw[start:start + layout["rows"] * layout["logits_row_bytes"]], dtype=np.float16)
        np.testing.assert_array_equal(output, expected)
        assert np.isfinite(output.reshape(layout["rows"], -1)[:, :tiny.config.vocab_size]).all()
        bundle = runner.bundle
        assert bytes(runner.simulator.state.dram[bundle.kv_cache_base:bundle.required_dram_bytes]) == kv_image
        assert any(kv_image), "nontrivial KV writes are required for this parity check"
        assert bytes(runner.simulator.state.dram[bundle.data_base:bundle.data_base + len(bundle.shared_data)]) == bundle.shared_data
        if position == 0:
            wrong = HostRunner(deepcopy(tiny.build.bundle), logits_dtype=np.float16)
            wrong._patch_embeddings("prefill", layout["tokens"], list(range(batch)))
            wrong._patch_kv_bases(0, stream="prefill")
            wrong._patch_attention_context("prefill", 0, 1)
            wrong.simulator.run_program(wrong.bundle, "prefill")
            wrong_kv = bytes(wrong.simulator.state.dram[wrong.bundle.kv_cache_base:wrong.bundle.required_dram_bytes])
            assert wrong_kv != kv_image, "nonzero position embeddings must expose old lockstep priming"


def test_sequential_prefill_equals_zero_based_chunk_and_continues(prefill_layout_cases):
    tiny = prefill_layout_cases[1, 16]
    prompt = list(range(16))
    direct = HostRunner(deepcopy(tiny.build.bundle), logits_dtype=np.float16)
    chunked = HostRunner(deepcopy(tiny.build.bundle), logits_dtype=np.float16)
    np.testing.assert_array_equal(direct.run_prefill(prompt), chunked.run_prefill_chunk(prompt, 0))
    assert direct.simulator.state.dram == chunked.simulator.state.dram
    np.testing.assert_array_equal(direct.run_decode_step(2, 16), chunked.run_decode_step(2, 16))
    assert direct.simulator.state.dram == chunked.simulator.state.dram
