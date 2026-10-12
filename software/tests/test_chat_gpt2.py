"""Bounded chat orchestration checks; no checkpoint, compile, or simulator."""
from types import SimpleNamespace

import numpy as np
import pytest

import chat_gpt2 as chat
from taccel.runtime.host_runner import HostRunner


def _mock_chat(monkeypatch, *, row_override=None):
    events = []
    builds = []
    marker = object()

    def build(payload, **kwargs):
        builds.append((payload, kwargs))
        # Deliberately misleading legacy byte budget: never infer INT8 here.
        return SimpleNamespace(build=SimpleNamespace(bundle=marker), logits_size=16)

    class FakeRunner(HostRunner):
        def __init__(self, bundle, *, logits_dtype, vocab_size):
            assert bundle is marker
            assert logits_dtype is np.float16
            assert vocab_size == 8

        def _row(self, token):
            if row_override is not None:
                return row_override.copy()
            row = np.full(16, np.nan, dtype=np.float16)
            row[:8] = -1
            row[(int(token) + 1) % 8] = -0.5
            return row

        def run_prefill(self, tokens):
            assert len(tokens) == 1, "the chat bundle has one prefill patch slot"
            events.append(("prefill", tuple(tokens)))
            return self._row(tokens[-1])

        def run_decode_step(self, token, position):
            events.append(("decode", token, position))
            return self._row(token)

    monkeypatch.setattr(chat, "build_stage3_tiny_decoder_bundle", build)
    monkeypatch.setattr(chat, "HostRunner", FakeRunner)
    return events, builds


@pytest.mark.parametrize("prompt", ([1], [1, 3, 5]))
@pytest.mark.parametrize("new_tokens", (0, 1, 4))
def test_golden_chat_skips_only_unused_final_decode(monkeypatch, prompt, new_tokens):
    events, builds = _mock_chat(monkeypatch)
    payload, preset, scales = {}, object(), {"lm_head": 0.25}
    result = chat._golden_generate(payload, prompt, max_new_tokens=new_tokens,
                                   ptq_preset=preset, calibration_scales=scales,
                                   vocab_size=8)
    sampled = [(prompt[-1] + step + 1) % 8 for step in range(new_tokens)]
    assert result == list(prompt) + sampled
    expected = [("prefill", (prompt[0],))]
    expected += [("decode", token, pos) for pos, token in enumerate(prompt[1:], 1)]
    expected += [("decode", token, len(prompt) + index)
                 for index, token in enumerate(sampled[:-1])]
    assert events == expected
    assert len(events) == len(prompt) + max(0, new_tokens - 1)
    assert builds == [(payload, {
        "smoke_decode_steps": max(0, len(prompt) + new_tokens - 1),
        "calibration_scales": scales, "ptq_preset": preset,
    })]


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_golden_chat_rejects_nonfinite_live_logits(monkeypatch, bad):
    row = np.zeros(16, dtype=np.float16)
    row[2] = bad
    events, _ = _mock_chat(monkeypatch, row_override=row)
    with pytest.raises(ValueError, match="non-finite"):
        chat._golden_generate({}, [1], max_new_tokens=1, ptq_preset=None,
                              calibration_scales={}, vocab_size=8)
    assert events == [("prefill", (1,))]


def test_zero_token_chat_does_not_sample_unused_logits(monkeypatch):
    events, _ = _mock_chat(monkeypatch, row_override=np.full(16, np.nan, dtype=np.float16))
    assert chat._golden_generate({}, [1, 3], max_new_tokens=0, ptq_preset=None,
                                 calibration_scales={}, vocab_size=8) == [1, 3]
    assert events == [("prefill", (1,)), ("decode", 3, 1)]


@pytest.mark.parametrize("prompt,new_tokens", (([], 1), ([1], -1)))
def test_golden_chat_rejects_invalid_request_before_build(monkeypatch, prompt, new_tokens):
    events, builds = _mock_chat(monkeypatch)
    with pytest.raises(ValueError):
        chat._golden_generate({}, prompt, max_new_tokens=new_tokens, ptq_preset=None,
                              calibration_scales={}, vocab_size=8)
    assert not events and not builds
