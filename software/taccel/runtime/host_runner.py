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

    def _patch_kv_bases(self, position: int) -> None:
        offset = int(position) * int(self.bundle.kv_step_bytes)
        for site in self._sites("kv_base", "decode"):
            self.bundle.patch_runtime_site(site, offset)

    def _patch_decode_attention_context(self, position: int) -> None:
        for site in self.bundle.runtime_config_attn_sites:
            if site.stream != "decode":
                continue
            self.bundle.patch_config_attn_site(
                site,
                query_row_base=int(position),
                valid_kv_len=int(position) + 1,
            )

    def _read_logits(self, offset: int) -> np.ndarray:
        size = int(self.bundle.logits_size)
        if size <= 0:
            return np.asarray([], dtype=self.logits_dtype)
        if size % self.logits_dtype.itemsize != 0:
            raise ValueError(
                f"logits_size={size} is not divisible by dtype size "
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
        self.simulator.run_program(self.bundle, "prefill", max_instructions=max_instructions)
        return self._read_logits(self.bundle.prefill_logits_offset)

    def run_decode_step(self, token_id: int, position: int, *,
                        max_instructions: int = 10_000_000) -> np.ndarray:
        if position < 0:
            raise ValueError("position must be non-negative")
        self._patch_embeddings("decode", [int(token_id)], [int(position)])
        self._patch_kv_bases(int(position))
        self._patch_decode_attention_context(int(position))
        self.simulator.run_program(self.bundle, "decode", max_instructions=max_instructions)
        return self._read_logits(self.bundle.decode_logits_offset)

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
