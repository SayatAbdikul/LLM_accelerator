"""W8A32 host runner — Phase 3 option (b).

Companion to `HostRunner` (`software/taccel/runtime/host_runner.py`).
Where `HostRunner` drives a compiled W8A8 `ProgramBundle` through the
`Simulator`, this runner mimics the same prefill/decode API but
*bypasses the accelerator entirely* and runs FP32 inference on the host
using `NanoGPTFP32Reference` with INT8-QDQ weights.

The deployment story this represents (the scoping doc calls it option
(b), `software/docs/w8a32_deployment_scope.md`):

- Weights ship as INT8 + per-channel scales — 4x compression on disk
  and in DRAM relative to FP32.
- At load time the runner builds a `NanoGPTFP32Reference` with QDQ
  weights via `build_weight_only_int8_reference` — i.e. the per-row
  INT8 weights expanded back to FP32 for FP32 matmul.
- Everything else runs FP32 on the host CPU. The INT8 MXU is not used.
- Inter-layer activations are full FP32 throughout.

PPL on `gpt2_converted_nanogpt.pt` at 257-tok / 256-ctx: 53.4212 PPL,
matching the Phase 1 numpy reference path bit-identically by construction
(both wrap the same helper).

This file does NOT implement options (c.1) / (c.2) — those require ISA
extensions (FP32 ABUF dtype mode or a sideband FP32 buffer) and 3-5
weeks of codegen / simulator / instruction set work. See the scoping
doc for the deferred work.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from .fp32_reference import build_weight_only_int8_reference


class WeightOnlyHostRunner:
    """W8A32 host-runtime runner — INT8 weight storage + FP32 host inference.

    API-compatible with `HostRunner` for the methods Stage 5 evaluation
    cares about (`run_prefill`, `run_decode_step`, `generate`), so
    callers can dispatch to either runner without branching on type.

    The runner constructs its `NanoGPTFP32Reference` once at `__init__`
    time (via `build_weight_only_int8_reference`) and uses it for every
    incremental step thereafter. Each `run_decode_step(token, position)`
    extends an internal KV cache the same way `HostRunner` does for the
    deployed bundle.
    """

    def __init__(self, payload: dict, *, weight_mode: str = "per_channel") -> None:
        if weight_mode not in ("per_channel", "mean_scale"):
            raise ValueError(
                f"weight_mode must be 'per_channel' or 'mean_scale', got {weight_mode!r}"
            )
        # Build once; reused for every prefill / decode call. The
        # underlying `NanoGPTFP32Reference` is stateless modulo the
        # explicit `caches` argument we manage below, so the same
        # instance handles both prefill and incremental decode.
        self._ref = build_weight_only_int8_reference(payload, weight_mode=weight_mode)
        self._caches = self._ref._empty_caches()
        self._next_position: int = 0

    # ------------------------------------------------------------------
    # HostRunner-compatible API
    # ------------------------------------------------------------------

    def run_prefill(self, token_ids: Iterable[int], **_: object) -> np.ndarray:
        """Run prefill on a prompt and return the last position's FP32 logits.

        Unlike `HostRunner` (which returns INT8 logits the simulator
        wrote into DRAM), this runner returns the FP32 logits directly.
        The W8A32 contract has no INT8 logit storage step.
        """
        tokens = [int(tok) for tok in token_ids]
        if not tokens:
            raise ValueError("run_prefill requires at least one token")
        # Reset incremental state for each prefill call (matches
        # `HostRunner` semantics where each prefill is a fresh
        # invocation against an empty KV cache).
        self._caches = self._ref._empty_caches()
        self._next_position = 0
        last_logits: Optional[np.ndarray] = None
        for tok in tokens:
            last_logits = self._ref._decode_incremental_step(
                int(tok),
                self._next_position,
                self._caches,
            )
            self._next_position += 1
        assert last_logits is not None
        return np.asarray(last_logits, dtype=np.float32)

    def run_decode_step(
        self,
        token_id: int,
        position: int,
        **_: object,
    ) -> np.ndarray:
        """Run one decode step and return its FP32 logits.

        `position` must match the runner's internal cursor (i.e. equal
        to the number of tokens already processed). This mirrors the
        deployed `HostRunner` contract — callers should pass
        `position = len(generated) - 1` after appending the new token.
        """
        position = int(position)
        if position < 0:
            raise ValueError("position must be non-negative")
        if position != self._next_position:
            raise ValueError(
                f"WeightOnlyHostRunner: position {position} does not "
                f"match internal cursor {self._next_position}. The "
                f"runner maintains an internal KV cache that requires "
                f"sequential decode calls."
            )
        logits = self._ref._decode_incremental_step(
            int(token_id),
            position,
            self._caches,
        )
        self._next_position += 1
        return np.asarray(logits, dtype=np.float32)

    def run_teacher_forced(self, token_ids: Sequence[int]) -> List[np.ndarray]:
        """Convenience: produce logits for every teacher-forced position.

        Equivalent to running `run_prefill` on the first token, then
        `run_decode_step` for each subsequent token, returning the
        per-step logits as a list.
        """
        toks = [int(t) for t in token_ids]
        if not toks:
            return []
        self._caches = self._ref._empty_caches()
        self._next_position = 0
        out: List[np.ndarray] = []
        for tok in toks:
            logits = self._ref._decode_incremental_step(
                int(tok),
                self._next_position,
                self._caches,
            )
            self._next_position += 1
            out.append(np.asarray(logits, dtype=np.float32))
        return out

    @staticmethod
    def _greedy_token(logits: np.ndarray) -> int:
        if logits.size == 0:
            return 0
        return int(np.argmax(logits))

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        sampler: str = "greedy",
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Greedy decode; same contract as `HostRunner.generate`."""
        if sampler != "greedy":
            raise ValueError("WeightOnlyHostRunner only supports greedy sampling")
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
