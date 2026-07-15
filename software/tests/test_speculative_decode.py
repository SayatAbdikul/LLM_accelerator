"""Spec-dec (batch-1): the P-row verify pass must be EXACTLY greedy, only faster.

The contract is not "close to" greedy — it is byte-identical. A candidate is
accepted only where the model's own argmax (computed on the true prefix, inside
the same pass) agrees with it, and every pass contributes one correction token
straight from the model. So a bad draft costs cycles and never correctness.

That makes the decisive test cheap to state: generate the same continuation two
ways — sequential greedy on a 1-token bundle, speculative on a P=16 bundle — and
require the token sequences to be equal. It is also the test that would catch
the subtle failure mode here: the two streams SHARE the logits region, so a
decode read that forgot to slice to its own row would silently return a token
from a stale verify row, and the sequences would diverge.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime.host_runner import HostRunner
from taccel.runtime.speculative import (
    PromptLookupDraft,
    SpecDecStats,
    speculative_generate,
)
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_tiny_fixture.py"

SMOKE = 120
VOCAB = 33
P = 16


def _build_payload(tmp_path):
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ckpt = tmp_path / "nanogpt_shakespeare_char_d128_l2.pt"
    tool.write_fixture(ckpt, ckpt.with_suffix(ckpt.suffix + ".json"))
    import torch
    return torch.load(ckpt, map_location="cpu")


def _prompt(n):
    return [(7 * i + 3) % VOCAB for i in range(n)]


def _repetitive_prompt(n):
    """A prompt with a real repeated n-gram, so the lookup draft actually hits."""
    motif = [4, 11, 19, 2, 25]
    return [motif[i % len(motif)] for i in range(n)]


def test_prompt_lookup_draft_proposes_the_continuation_of_the_last_match():
    draft = PromptLookupDraft(max_ngram=3)
    # "1 2 3" occurred earlier and was followed by 7, 8, 9.
    ctx = [1, 2, 3, 7, 8, 9, 5, 5, 1, 2, 3]
    assert draft.propose(ctx, 3) == [7, 8, 9]
    # No repeat anywhere -> nothing to propose (driver then falls back).
    assert draft.propose([1, 2, 3, 4], 3) == []


@pytest.mark.parametrize("prompt_fn", [_prompt, _repetitive_prompt])
def test_speculative_matches_sequential_greedy_token_for_token(tmp_path, prompt_fn):
    payload = _build_payload(tmp_path)
    prompt = prompt_fn(24)
    n_new = 24

    # Reference: plain sequential greedy on the 1-token bundle. Its prefill graph
    # takes exactly one token, so the prompt is primed as prefill(t0) + decode(t1..)
    # -- the same way the lever I-b test builds its sequential reference.
    seq = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=SMOKE, batch=1)
    rs = HostRunner(seq.build.bundle, logits_dtype=np.float16)
    logits = rs.run_prefill([prompt[0]])
    for i in range(1, len(prompt)):
        logits = rs.run_decode_step(prompt[i], i)
    expected = list(prompt)
    for _ in range(n_new):
        tok = int(np.argmax(np.asarray(logits)[:VOCAB]))
        expected.append(tok)
        logits = rs.run_decode_step(tok, len(expected) - 1)

    # Speculative: P-row verify passes on the prefill_tokens=P bundle.
    spec = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=SMOKE, batch=1, prefill_tokens=P)
    rd = HostRunner(spec.build.bundle, logits_dtype=np.float16)
    stats = SpecDecStats()
    got = speculative_generate(rd, prompt, n_new, vocab_size=VOCAB, stats=stats)

    assert got == expected, (
        "speculative decoding diverged from sequential greedy\n"
        f"  expected: {expected}\n  got:      {got}\n  {stats.summary()}"
    )
    assert stats.tokens_emitted == n_new
    # It must actually have run verify passes, not silently fallen back to a
    # 1-token step every time -- otherwise this test proves nothing about the
    # verify path.
    assert stats.passes > 0, "no verify pass ran; the test is vacuous"


def test_speculative_costs_fewer_passes_when_the_draft_hits(tmp_path):
    """The point of the lever: a hitting draft emits more than one token per pass."""
    payload = _build_payload(tmp_path)
    spec = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=SMOKE, batch=1, prefill_tokens=P)
    rd = HostRunner(spec.build.bundle, logits_dtype=np.float16)

    stats = SpecDecStats()
    speculative_generate(rd, _repetitive_prompt(24), 24, vocab_size=VOCAB, stats=stats)

    total_passes = stats.passes + stats.fallback_steps
    assert total_passes < stats.tokens_emitted, (
        f"spec-dec used {total_passes} passes for {stats.tokens_emitted} tokens "
        f"-- no amortisation at all ({stats.summary()})"
    )


def test_acceptance_bench_simulator_matches_the_shipped_driver(tmp_path):
    """The reported speedup must come from the code that SHIPS.

    `bench_specdec_acceptance.simulate()` is a second implementation of the accept
    loop -- it walks a precomputed greedy sequence instead of driving the chip, which
    is what makes measuring `t` cheap. If it drifts from `speculative_generate`, the
    headline tok/s number describes code nobody runs. So pin them together: same
    prompt, same draft, same pass accounting.
    """
    import importlib.util as _il
    spec_mod = _il.spec_from_file_location(
        "bench_specdec_acceptance",
        Path(__file__).resolve().parents[1] / "tools" / "bench_specdec_acceptance.py")
    bench = _il.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(bench)

    payload = _build_payload(tmp_path)
    prompt = _repetitive_prompt(24)
    n_new = 24

    spec = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=SMOKE, batch=1, prefill_tokens=P)
    rd = HostRunner(spec.build.bundle, logits_dtype=np.float16)

    draft = PromptLookupDraft()
    stats = SpecDecStats()
    got = speculative_generate(rd, prompt, n_new, vocab_size=VOCAB,
                               draft=draft, stats=stats)

    # Feed the simulator the sequence the driver actually produced (which is the
    # greedy sequence -- that is the whole exactness contract) and require it to
    # reproduce the driver's pass accounting exactly.
    passes, fb, emitted, acc, prop, per_pass = bench.simulate(
        prompt, got, PromptLookupDraft(), P, 1.4531)

    assert emitted == stats.tokens_emitted
    assert passes == stats.passes, f"passes {passes} != driver {stats.passes}"
    assert fb == stats.fallback_steps
    assert acc == stats.accepted and prop == stats.proposed
    assert per_pass == stats.tokens_per_pass


def test_speculative_requires_a_multi_row_prefill_bundle(tmp_path):
    payload = _build_payload(tmp_path)
    seq = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=SMOKE, batch=1)
    rs = HostRunner(seq.build.bundle, logits_dtype=np.float16)
    with pytest.raises(ValueError, match="prefill_tokens"):
        speculative_generate(rs, _prompt(8), 4, vocab_size=VOCAB)


@pytest.mark.parametrize("batch", [1, 16])
def test_specdec_is_inert_at_the_default(tmp_path, batch):
    """A default bundle (prefill_tokens=1) must be BYTE-IDENTICAL to the pre-B3
    compiler, so speculative decoding cannot perturb an architecture/cycle
    experiment that does not explicitly opt in.

    The goal of the project is the chip; spec-dec is a non-interfering host-side
    track. If a change to the shared bundle path (logits_rows, prefill_store_rows,
    the logits-store emitter, the region layout) ever shifts a default bundle,
    this fails -- which is the signal that the opt-in boundary has leaked.
    """
    import hashlib
    payload = _build_payload(tmp_path)
    tiny = build_stage3_tiny_decoder_bundle(payload, smoke_decode_steps=8, batch=batch)
    b = tiny.build.bundle
    img = bytes(b.materialize(reset_runtime=False))
    sha = hashlib.sha256(img).hexdigest()[:16]
    # Golden hashes captured from commit daef072 (the last commit before lever B3),
    # verified by rebuilding that commit's compiler on the same fixture.
    golden = {1: "172b4aa61a3de54e", 16: "e0f9c8ca2a50d259"}
    assert b.prefill_store_rows == 1, "default bundle must not store multiple prefill rows"
    assert sha == golden[batch], (
        f"default batch={batch} bundle changed (sha {sha} != {golden[batch]}): a "
        f"spec-dec / shared-logits change has leaked into the non-spec-dec path"
    )
