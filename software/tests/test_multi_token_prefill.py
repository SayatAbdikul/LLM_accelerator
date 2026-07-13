"""Lever I-b: DENSE multi-token prefill (TTFT).

The prefill stream used to be decode-shaped — ONE token per pass. Since
`M_pad = pad_dim(M)`, a 1-token pass still occupies a full 16-row systolic
m-tile and WASTES 15/16 of the mesh. Feeding P real query rows fills the tile.
This is the same effect that made batched decode ~6x better per token than b1
(see docs/lever_h_b32.md), applied to prompt processing.

The correctness contract is strong and exact: a P-token dense prefill must
produce BYTE-IDENTICAL logits to the sequential path
`prefill(t0) + decode(t1..t_{P-1})` — matmuls are row-independent, LN/softmax are
per-row, and the KV quant scales are static. Anything else is a bug.
"""
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from taccel.assembler.assembler import ProgramBinary
from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

RTL_SYNTH_BINARY = (
    Path(__file__).resolve().parents[2]
    / "rtl" / "verilator" / "build" / "run_program_synth" / "Vtaccel_top"
)
TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"

SMOKE = 80          # decode window must cover prompt + generated tokens
VOCAB = 33          # the tiny fixture's real vocab


def _build_payload(tmp_path):
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ckpt = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    tool.write_fixture(ckpt, ckpt.with_suffix(ckpt.suffix + ".json"))
    import torch
    return torch.load(ckpt, map_location="cpu")


def _prompt(p):
    return [(7 * i + 3) % VOCAB for i in range(p)]


def test_prefill_tokens_requires_single_stream():
    # Multi-token prefill is a TTFT lever on the single-stream path. The batched
    # bundle already reuses its batched decode graph for the prefill slot.
    with pytest.raises(ValueError):
        build_stage3_tiny_decoder_bundle({}, batch=16, prefill_tokens=16)


@pytest.mark.parametrize("P", [16, 32, 64])
def test_prefill_graph_is_dense_multi_token(tmp_path, P):
    payload = _build_payload(tmp_path)
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=SMOKE, batch=1, prefill_tokens=P)
    bundle = tiny.build.bundle

    # One runtime embedding patch site per prompt row (vs 1 for a 1-token prefill).
    sites = [s for s in bundle.runtime_patch_sites
             if s.kind == "token_embed" and s.stream == "prefill"]
    assert len(sites) == P
    # The decode stream is untouched: still ONE query row.
    dec = [s for s in bundle.runtime_patch_sites
           if s.kind == "token_embed" and s.stream == "decode"]
    assert len(dec) == 1


@pytest.mark.parametrize("P", [16, 32, 64])
def test_multi_token_prefill_is_byte_identical_to_sequential(tmp_path, P):
    """The whole point: one dense P-row pass == P sequential decode-shaped passes."""
    payload = _build_payload(tmp_path)
    prompt = _prompt(P)

    seq = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=SMOKE, batch=1)
    rs = HostRunner(seq.build.bundle, logits_dtype=np.float16)
    c0 = rs.simulator.state.cycle_count
    logits_seq = rs.run_prefill([prompt[0]])
    for i in range(1, P):
        logits_seq = rs.run_decode_step(prompt[i], i)
    cyc_seq = rs.simulator.state.cycle_count - c0

    dense = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=SMOKE, batch=1, prefill_tokens=P)
    rd = HostRunner(dense.build.bundle, logits_dtype=np.float16)
    c0 = rd.simulator.state.cycle_count
    logits_dense = rd.run_prefill(prompt)
    cyc_dense = rd.simulator.state.cycle_count - c0

    assert np.array_equal(np.asarray(logits_seq)[:VOCAB],
                          np.asarray(logits_dense)[:VOCAB]), \
        "dense prefill logits differ from the sequential path"

    # The KV cache the dense pass wrote must let the decode CONTINUE correctly:
    # generate a few tokens from each and require the same sequence.
    def _gen(runner, logits, n=3):
        out, tok = [], int(np.argmax(np.asarray(logits)[:VOCAB]))
        for g in range(n):
            out.append(tok)
            lg = runner.run_decode_step(tok, P + g)
            tok = int(np.argmax(np.asarray(lg)[:VOCAB]))
        return out

    assert _gen(rs, logits_seq) == _gen(rd, logits_dense), \
        "decode diverges after the dense prefill (KV cache not written correctly)"

    # TTFT: the dense pass must be materially cheaper than P sequential passes.
    assert cyc_dense < cyc_seq / 2, f"no TTFT win: {cyc_seq} -> {cyc_dense}"


def test_multi_token_prefill_rtl_matches_golden_bytes(tmp_path):
    """Bit-exact contract: mode-1 synth RTL == mode-0 golden on the DENSE PREFILL
    program (byte-identical logits region).

    Tiny model on purpose — RTL-vs-golden byte-match is only well-posed below
    GPT-2 124M's first fp16 overflow (see rtl_cosim.py #109).
    """
    if not RTL_SYNTH_BINARY.exists():
        pytest.skip(f"synth RTL binary not built: {RTL_SYNTH_BINARY}")
    P = 16
    payload = _build_payload(tmp_path)
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=SMOKE, batch=1, prefill_tokens=P)
    bundle = tiny.build.bundle
    prompt = _prompt(P)

    # Golden (mode-0 DPI): run the prefill stream on a fresh bundle.
    runner = HostRunner(bundle, logits_dtype=np.int8)
    golden = runner.run_prefill(prompt)

    # RTL: patch the same sites, extract the standalone PREFILL program, run it
    # on a fresh image and dump the logits region. Symmetric with the golden run.
    patcher = HostRunner(bundle, simulator=None)
    patcher._patch_embeddings("prefill", prompt, list(range(P)))
    patcher._patch_kv_bases(0, stream="prefill")
    image = bundle.materialize(reset_runtime=False)
    data_base = int(bundle.data_base)
    prefill_only = bytes(image[int(bundle.prefill_instrs_offset):int(bundle.decode_instrs_offset)])
    pb = ProgramBinary(
        instructions=prefill_only, data=bytes(image[data_base:]), entry_point=0,
        insn_count=len(prefill_only) // 8, data_base=data_base,
        input_offset=0, pos_embed_patch_dram_offset=0, pos_embed_cls_dram_offset=0,
        cls_token_dram_offset=0, trace_manifest={}, compiler_manifest={},
    )
    td = Path(tempfile.mkdtemp())
    (td / "p.bin").write_bytes(pb.to_bytes())
    off, size = int(bundle.prefill_logits_offset), int(bundle.logits_size)
    argv = [str(RTL_SYNTH_BINARY), "--program", str(td / "p.bin"),
            "--json-out", str(td / "s.json"), "--fast-beats", "--max-cycles", "40000000",
            "--dram-dump-offset", str(off), "--dram-dump-size", str(size),
            "--dram-dump-out", str(td / "dram.bin")]
    cp = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    assert (td / "s.json").exists(), f"RTL run failed: {cp.stderr[-800:]}"
    summ = json.loads((td / "s.json").read_text())
    assert summ.get("status") == "halted" and not summ.get("fault"), summ
    rtl = np.frombuffer((td / "dram.bin").read_bytes(), dtype=np.int8)[:golden.shape[0]]

    assert np.array_equal(golden, rtl), (
        f"RTL != golden on the dense prefill: "
        f"{int(np.sum(golden != rtl))}/{golden.shape[0]} bytes differ"
    )
