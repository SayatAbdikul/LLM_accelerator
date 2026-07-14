"""Speculative decoding for batch-1 — propose, verify in ONE pass, accept.

Why this is the batch-1 lever
-----------------------------
The systolic mesh is 16 rows. A single-token decode step presents ONE query row
and pads the other 15, so 15/16 of the array is idle — and the whole 124 MB of
INT8 weights streams from DRAM to produce that one token (~8.5M beats, which on
its own caps batch-1 at ~4 tok/s no matter how fast the compute gets).

A 16-row pass costs only ~1.2-1.3x a 1-row step, because the systolic streams
weights at the same rate either way: rows are nearly free. So if we can *guess*
the next few tokens and check them all in one pass, both the mesh waste and the
weight stream amortize over every token the pass confirms.

The verify pass already exists: it is lever I-b's chunked prefill
(`HostRunner.run_prefill_chunk_rows`) — P causal query rows at a runtime base,
each row masked to exactly the keys at or before its own global position. Row i
is therefore the model's next-token distribution after consuming chain[0..i],
which is precisely what verification needs. No RTL change; this module is host
code.

Exactness
---------
The accepted tokens are IDENTICAL to greedy sequential decoding, and this is a
property of the accept rule, not of the draft's quality: a candidate is accepted
only where the model's own argmax (computed on the true prefix, in the same
pass) agrees with it, and the pass always contributes one *correction* token
taken straight from the model. A bad draft costs cycles, never correctness — so
the draft needs no accuracy guarantee at all.

Rejected candidates need no KV cleanup. Their KV rows are overwritten by the
next pass before anything reads them, and the causal mask (`valid_kv_len`)
never exposes a position beyond the accepted prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import numpy as np


class Draft(Protocol):
    """Proposes the next tokens. Free to be wrong — see `Exactness` above."""

    def propose(self, context: Sequence[int], k: int) -> List[int]:
        ...


@dataclass
class PromptLookupDraft:
    """Prompt-lookup (n-gram) draft: no weights, no training, no second model.

    Take the last `n` tokens of the context, find the most recent EARLIER
    occurrence of that same n-gram, and propose whatever followed it. On
    grounded text (summarisation, QA over a passage, code) the continuation is
    frequently a literal repeat, so this hits often; on open-ended text it hits
    less, and the adaptive fallback below keeps that from costing anything.

    Longest match first: a 3-gram match is better evidence than a 1-gram match.
    """

    max_ngram: int = 3
    min_ngram: int = 1

    def propose(self, context: Sequence[int], k: int) -> List[int]:
        if k <= 0 or len(context) < 2:
            return []
        ctx = list(context)
        n_max = min(self.max_ngram, len(ctx) - 1)
        for n in range(n_max, self.min_ngram - 1, -1):
            tail = ctx[-n:]
            # Scan backwards for the most recent earlier occurrence; a match at
            # the very end is the tail itself and predicts nothing.
            for start in range(len(ctx) - n - 1, -1, -1):
                if ctx[start:start + n] == tail:
                    nxt = ctx[start + n:start + n + k]
                    if nxt:
                        return list(nxt)
        return []


@dataclass
class SpecDecStats:
    """Enough to audit the tok/s claim, not just assert it."""

    passes: int = 0                      # verify passes run
    fallback_steps: int = 0              # 1-token steps (draft had nothing)
    tokens_emitted: int = 0
    proposed: int = 0
    accepted: int = 0                    # draft tokens confirmed by the model
    tokens_per_pass: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def mean_tokens_per_pass(self) -> float:
        n = len(self.tokens_per_pass)
        return sum(self.tokens_per_pass) / n if n else 0.0

    def summary(self) -> str:
        return (
            f"tokens={self.tokens_emitted} passes={self.passes} "
            f"fallback_steps={self.fallback_steps} "
            f"accept_rate={self.acceptance_rate:.3f} "
            f"tokens/pass={self.mean_tokens_per_pass:.2f}"
        )


def _greedy(row: np.ndarray, vocab: int) -> int:
    """Argmax with the tie-break PINNED to the lowest index.

    The reference decoder must break ties the same way or the "identical to
    greedy" gate becomes a coin flip on tied logits. `np.argmax` already
    returns the first maximal index; this states it so it cannot drift.
    Padding columns past `vocab` are excluded — they are not tokens.
    """
    return int(np.argmax(np.asarray(row)[:vocab]))


def speculative_generate(
    runner,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    vocab_size: int,
    draft: Optional[Draft] = None,
    eos_token_id: Optional[int] = None,
    stats: Optional[SpecDecStats] = None,
) -> List[int]:
    """Greedy-equivalent generation, accelerated by a P-row verify pass.

    Returns prompt + generated tokens — byte-for-byte the sequence
    `HostRunner.generate` would produce, at a fraction of the passes.

    `runner` must be a HostRunner over a bundle built with `prefill_tokens=P`
    (so the prefill stream stores all P logits rows). The bundle's compiled KV
    window must cover the highest position reached, or the SFU faults
    FAULT_NO_CONFIG.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not prompt_ids:
        raise ValueError("speculative_generate requires at least one prompt token")

    bundle = runner.bundle
    p_rows = int(bundle.prefill_store_rows)
    if p_rows < 2:
        raise ValueError(
            "speculative_generate needs a bundle built with prefill_tokens > 1 "
            f"(prefill_store_rows={p_rows}); the verify pass is the P-row "
            "chunked-prefill program"
        )
    draft = draft or PromptLookupDraft()
    stats = stats if stats is not None else SpecDecStats()

    tokens = [int(t) for t in prompt_ids]
    prompt_len = len(tokens)

    # Prime the KV cache over the prompt with the same P-row pass, then take the
    # last prompt row's prediction as the first live token.
    logits = None
    for c in range(0, prompt_len, p_rows):
        chunk = tokens[c:c + p_rows]
        take = len(chunk) - 1  # the last real row of this chunk
        chunk = _pad_to(chunk, p_rows)
        rows = runner.run_prefill_chunk_rows(chunk, c)
        logits = rows[take]
    cur = _greedy(logits, vocab_size)
    cur_pos = prompt_len  # position `cur` will occupy once emitted

    generated: List[int] = []
    while len(generated) < max_new_tokens:
        if eos_token_id is not None and cur == eos_token_id:
            generated.append(cur)
            stats.tokens_emitted += 1
            break

        budget = max_new_tokens - len(generated)
        # The chain is [cur] + guesses. `cur` occupies row 0, so at most
        # p_rows - 1 guesses fit, and we never verify past the token budget.
        k = min(p_rows - 1, budget - 1)
        guesses = list(draft.propose(tokens + generated + [cur], k)) if k > 0 else []

        if not guesses:
            # ADAPTIVE FALLBACK: nothing to verify, so don't pay for a verify
            # pass. This is what keeps the guaranteed floor at ~1x a plain
            # decode rather than 1/r.
            row = runner.run_decode_step(cur, cur_pos)
            generated.append(cur)
            stats.tokens_emitted += 1
            stats.fallback_steps += 1
            cur = _greedy(row, vocab_size)
            cur_pos += 1
            continue

        chain = [cur] + guesses
        n_guess = len(guesses)
        rows = runner.run_prefill_chunk_rows(_pad_to(chain, p_rows), cur_pos)
        stats.passes += 1
        stats.proposed += n_guess

        # Row i predicts the token AFTER chain[0..i]. Accept guess i+1 exactly
        # where the model agrees; stop at the first disagreement.
        n_accept = 0
        for i in range(n_guess):
            if _greedy(rows[i], vocab_size) == chain[i + 1]:
                n_accept += 1
            else:
                break
        stats.accepted += n_accept

        # chain[0..n_accept] are all now known-correct; the model's own
        # prediction at row n_accept is the correction token (and when every
        # guess was accepted, it is simply the next token). Either way it comes
        # from the model, so the emitted sequence stays exactly greedy.
        emitted = chain[:n_accept + 1]
        correction = _greedy(rows[n_accept], vocab_size)

        room = max_new_tokens - len(generated)
        emitted = emitted[:room]
        generated.extend(emitted)
        stats.tokens_emitted += len(emitted)
        stats.tokens_per_pass.append(len(emitted))

        if len(generated) >= max_new_tokens:
            break
        if eos_token_id is not None and eos_token_id in emitted:
            break

        cur = correction
        cur_pos += len(emitted)

    return tokens + generated[:max_new_tokens]


def _pad_to(chain: Sequence[int], p: int) -> List[int]:
    """Pad a chain to the pass's P embedding sites.

    The program has exactly P embedding patch sites, so a short chain must be
    filled. The pad rows compute garbage and write garbage KV at positions past
    the chain — which is harmless: the next pass starts at or before the first
    pad position and overwrites those rows before any attention reads them, and
    `valid_kv_len` never exposes a position beyond the accepted prefix.
    """
    out = list(chain)
    if len(out) > p:
        raise ValueError(f"chain of {len(out)} exceeds the pass width {p}")
    while len(out) < p:
        out.append(0)
    return out
