"""Golden-model host runner for Stage 3 decoder ProgramBundles."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..assembler.assembler import ProgramBundle, RuntimePatchSite
from ..golden_model.simulator import Simulator


class HostRunner:
    """Drive prefill/decode ProgramBundle streams with runtime patch sites."""

    def __init__(self, bundle: ProgramBundle, simulator: Optional[Simulator] = None,
                 *, logits_dtype=np.int32):
        self.bundle = bundle
        self.simulator = simulator or Simulator()
        self.logits_dtype = np.dtype(logits_dtype)
        self.simulator.load_bundle(bundle)

    def _sites(self, kind: str, stream: str) -> List[RuntimePatchSite]:
        return [
            site for site in self.bundle.runtime_patch_sites
            if site.kind == kind and site.stream == stream
        ]

    def _patch_sites(self, kind: str, stream: str, offsets: Sequence[int]) -> None:
        sites = self._sites(kind, stream)
        if not sites:
            return
        if len(sites) == 1 and len(offsets) != 1:
            raise ValueError(
                f"Runtime site kind={kind!r} stream={stream!r} supports one row, "
                f"got {len(offsets)} offsets"
            )
        if len(sites) != len(offsets):
            raise ValueError(
                f"Expected {len(sites)} offsets for kind={kind!r} stream={stream!r}, "
                f"got {len(offsets)}"
            )
        for site, offset in zip(sites, offsets):
            self.bundle.patch_runtime_site(site, int(offset))

    def _patch_embeddings(self, stream: str, token_ids: Sequence[int],
                          position_ids: Sequence[int]) -> None:
        row_bytes = int(self.bundle.embedding_row_bytes)
        self._patch_sites("token_embed", stream, [int(tok) * row_bytes for tok in token_ids])
        self._patch_sites("pos_embed", stream, [int(pos) * row_bytes for pos in position_ids])

    def _patch_kv_bases(self, position: int, stream: str = "decode") -> None:
        offset = int(position) * int(self.bundle.kv_step_bytes)
        for site in self._sites("kv_base", stream):
            self.bundle.patch_runtime_site(site, offset)

    def _patch_attention_context(self, stream: str, query_row_base: int,
                                 valid_kv_len: int) -> None:
        """Set the causal window for one stream.

        The masked softmax keys its triangle off `query_row_base + row_idx`, so a
        multi-row chunk based at `query_row_base` masks each of its rows against
        exactly the keys at or before that row's GLOBAL position.
        """
        for site in self.bundle.runtime_config_attn_sites:
            if site.stream != stream:
                continue
            self.bundle.patch_config_attn_site(
                site,
                query_row_base=int(query_row_base),
                valid_kv_len=int(valid_kv_len),
            )

    def _patch_decode_attention_context(self, position: int) -> None:
        # One query row at `position`: it sees keys 0..position.
        self._patch_attention_context("decode", position, int(position) + 1)

    def _read_logits(self, offset: int, rows: int = 0) -> np.ndarray:
        """Read `rows` logits rows from `offset` (0 = the whole region).

        The prefill and decode streams SHARE the logits region, which is sized
        for whichever stores more rows. A reader must take only its OWN rows:
        on a `prefill_tokens=16` bundle a 1-row decode step writes row 0 and
        leaves rows 1..15 holding the last verify pass's logits, so reading the
        whole region and taking an argmax over it would silently return a token
        from a stale row.
        """
        size = int(self.bundle.logits_size)
        if size <= 0:
            return np.asarray([], dtype=self.logits_dtype)
        if rows > 0:
            if rows > int(self.bundle.logits_rows):
                raise ValueError(
                    f"requested {rows} logits rows but the region holds "
                    f"{self.bundle.logits_rows}"
                )
            size = rows * int(self.bundle.logits_row_bytes)
        if size % self.logits_dtype.itemsize != 0:
            raise ValueError(
                f"logits size={size} is not divisible by dtype size "
                f"{self.logits_dtype.itemsize}"
            )
        data = bytes(self.simulator.state.dram[int(offset):int(offset) + size])
        return np.frombuffer(data, dtype=self.logits_dtype).copy()

    def run_prefill(self, token_ids: Iterable[int], *,
                    max_instructions: int = 10_000_000) -> np.ndarray:
        tokens = [int(tok) for tok in token_ids]
        if not tokens:
            raise ValueError("run_prefill requires at least one token")
        self._patch_embeddings("prefill", tokens, list(range(len(tokens))))
        # The batched bundle reuses the batched DECODE graph for its prefill slot
        # (tiny_fixture), so the prefill stream carries runtime `kv_base` sites.
        # Left unpatched they read 0, and the KV row stores then address
        # `0 + dram_off` — which lands inside the weight/data region and silently
        # CORRUPTS the weights. Prime the lockstep streams at position 0.
        # (The single-token prefill graph has no kv_base sites, so this is a
        # no-op there and the b1 path stays byte-identical.)
        self._patch_kv_bases(0, stream="prefill")
        self.simulator.run_program(self.bundle, "prefill", max_instructions=max_instructions)
        return self._read_logits(self.bundle.prefill_logits_offset,
                                 rows=int(self.bundle.prefill_store_rows))

    def run_prefill_chunk(self, token_ids: Sequence[int], base_position: int, *,
                          max_instructions: int = 10_000_000) -> np.ndarray:
        """Lever I-b: run ONE prefill chunk of P tokens at positions
        [base_position, base_position + P).

        The prefill stream of a `prefill_tokens=P` bundle is the decode graph with
        P query rows: it reads the KV cache and writes its P rows back at a
        runtime-patched base, so this can be called repeatedly to walk a prompt of
        any length::

            for c in range(0, len(prompt), P):
                logits = runner.run_prefill_chunk(prompt[c:c + P], c)
            # `logits` now holds the LAST prompt row's logits = the first token.

        Each pass costs ONE decode-shaped step but consumes P tokens, because
        M_pad = pad_dim(P): a 1-token pass wastes 15/16 of the 16-row systolic
        m-tile, and P=16 fills it. The chunk must be exactly P tokens long (the
        program has P embedding patch sites).

        Returns the FLAT prefill logits region, row-major.

        On a bundle built with ``prefill_tokens=P`` the store now covers all P
        rows (``store_rows=P, row_index=0``), so row *i* is the next-token
        distribution for chunk position *i* — the last row is the prompt's next
        token, and the earlier rows are what speculative decoding verifies
        against. Use :meth:`run_prefill_chunk_rows` for the (P, cols) view.
        On a ``prefill_store_rows == 1`` bundle this is the single last-row
        store exactly as before.
        """
        tokens = [int(tok) for tok in token_ids]
        if not tokens:
            raise ValueError("run_prefill_chunk requires at least one token")
        if base_position < 0:
            raise ValueError("base_position must be non-negative")
        p = len(tokens)
        base = int(base_position)
        self._patch_embeddings("prefill", tokens, [base + i for i in range(p)])
        self._patch_kv_bases(base, stream="prefill")
        # Rows base..base+P-1 see keys 0..base+P-1; the per-row triangle inside
        # that window comes from `query_row_base + row_idx`.
        self._patch_attention_context("prefill", base, base + p)
        self.simulator.run_program(self.bundle, "prefill", max_instructions=max_instructions)
        return self._read_logits(self.bundle.prefill_logits_offset,
                                 rows=int(self.bundle.prefill_store_rows))

    def run_prefill_chunk_rows(self, token_ids: Sequence[int], base_position: int, *,
                               max_instructions: int = 10_000_000) -> np.ndarray:
        """:meth:`run_prefill_chunk` as a (prefill_store_rows, cols) 2-D view.

        Row *i* is the next-token distribution *after* chunk position *i*. This
        is the speculative-decoding verify primitive: one pass scores every
        candidate position at once.
        """
        flat = self.run_prefill_chunk(token_ids, base_position,
                                      max_instructions=max_instructions)
        rows = int(self.bundle.prefill_store_rows)
        return flat.reshape(rows, -1)

    def run_decode_step(self, token_id: int, position: int, *,
                        max_instructions: int = 10_000_000) -> np.ndarray:
        if position < 0:
            raise ValueError("position must be non-negative")
        self._patch_embeddings("decode", [int(token_id)], [int(position)])
        self._patch_kv_bases(int(position))
        self._patch_decode_attention_context(int(position))
        self.simulator.run_program(self.bundle, "decode", max_instructions=max_instructions)
        # One row only: on a spec-dec bundle the region also holds the verify
        # pass's other 15 rows, and they are stale here.
        return self._read_logits(self.bundle.decode_logits_offset, rows=1)

    def run_decode_step_batch(self, token_ids: Sequence[int], position: int, *,
                              max_instructions: int = 10_000_000) -> np.ndarray:
        """Drive one lockstep batched decode step (Phase 2).

        All ``len(token_ids)`` streams advance to the same ``position`` (single
        ``valid_kv_len`` scalar — ragged batching is out of scope). The token
        count must match the decode stream's embedding patch-site count (16 for
        a ``batch=16`` bundle, 1 for a single-token bundle).

        Lever I-a (logits×N): the decode stream stores one logits row per
        stream, so the returned flat buffer spans the whole ``logits_size``
        region — all ``len(token_ids)`` rows, contiguous and row-major.
        Reshape to per-stream logits with ``out.reshape(len(token_ids), -1)``
        (each row is ``pad_dim(vocab)`` wide; slice ``[:, :vocab]`` for the
        live logits). KV bases are patched with the shared decode base.
        """
        if position < 0:
            raise ValueError("position must be non-negative")
        tokens = [int(tok) for tok in token_ids]
        self._patch_embeddings("decode", tokens, [int(position)] * len(tokens))
        self._patch_kv_bases(int(position))
        self._patch_decode_attention_context(int(position))
        self.simulator.run_program(self.bundle, "decode", max_instructions=max_instructions)
        return self._read_logits(self.bundle.decode_logits_offset, rows=len(tokens))

    @staticmethod
    def _greedy_token(logits: np.ndarray) -> int:
        if logits.size == 0:
            return 0
        return int(np.argmax(logits))

    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int,
                 *, sampler: str = "greedy", eos_token_id: Optional[int] = None) -> List[int]:
        if sampler != "greedy":
            raise ValueError("Stage 3 HostRunner only supports greedy sampling")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not prompt_ids:
            raise ValueError("generate requires at least one prompt token")

        generated = [int(tok) for tok in prompt_ids]
        logits = self.run_prefill(generated)
        next_token = self._greedy_token(logits)

        for _ in range(max_new_tokens):
            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            position = len(generated) - 1
            logits = self.run_decode_step(next_token, position)
            next_token = self._greedy_token(logits)
        return generated
