"""Offline GPT-2 perplexity helpers for the Stage 5 golden-model gate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .calibration import build_calibration_scales_from_token_ids
from .fake_quant_reference import NanoGPTFQReference
from .host_runner import HostRunner
from .tiny_fixture import build_stage3_tiny_decoder_bundle


@dataclass
class GPT2PerplexityResult:
    golden_perplexity: float
    fake_quant_perplexity: float
    relative_delta: float
    token_count: int
    target_count: int
    golden_nll: float
    fake_quant_nll: float
    tokenizer_dir: str
    calibration_sha256: str
    eval_sha256: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gpt2_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise RuntimeError("transformers is required for offline GPT-2 tokenization") from exc
    return GPT2TokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)


def tokenize_text_file(tokenizer_dir: Path, text_path: Path, *, max_tokens: int | None = None) -> List[int]:
    if not Path(text_path).exists():
        raise FileNotFoundError(text_path)
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)
    text = Path(text_path).read_text(encoding="utf-8")
    token_ids = [int(tok) for tok in tokenizer.encode(text)]
    if max_tokens is not None:
        token_ids = token_ids[: int(max_tokens)]
    return token_ids


def teacher_forced_inputs_and_targets(token_ids: Sequence[int]) -> tuple[List[int], List[int]]:
    tokens = [int(tok) for tok in token_ids]
    if len(tokens) < 2:
        raise ValueError("perplexity evaluation requires at least two tokens")
    return tokens[:-1], tokens[1:]


def stable_cross_entropy(logits: np.ndarray, target: int, *, vocab_size: int) -> float:
    active = np.asarray(logits, dtype=np.float32)[: int(vocab_size)]
    if active.size == 0:
        raise ValueError("logits are empty")
    target_i = int(target)
    if target_i < 0 or target_i >= active.size:
        raise ValueError(f"target token {target_i} is outside vocab size {active.size}")
    row_max = float(np.max(active))
    shifted = active - np.float32(row_max)
    exp_shifted = np.exp(shifted.astype(np.float32)).astype(np.float32)
    logsumexp = row_max + float(np.log(exp_shifted.sum(dtype=np.float32)))
    return float(logsumexp - float(active[target_i]))


def perplexity_from_nlls(nlls: Sequence[float]) -> tuple[float, float]:
    if not nlls:
        raise ValueError("at least one NLL is required")
    mean_nll = float(np.mean(np.asarray(nlls, dtype=np.float64)))
    return float(math.exp(mean_nll)), mean_nll


def run_golden_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
    calibration_scales: Dict[str, float],
) -> List[np.ndarray]:
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max(0, len(inputs) - 1),
        calibration_scales=calibration_scales,
    )
    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
    logits: List[np.ndarray] = [runner.run_prefill([inputs[0]])]
    for position, token in enumerate(inputs[1:], start=1):
        logits.append(runner.run_decode_step(int(token), position))
    return logits


def run_fake_quant_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
    calibration_scales: Dict[str, float],
) -> List[np.ndarray]:
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    ref = NanoGPTFQReference(
        payload["state_dict"],
        payload["model_args"],
        calibration_scales,
    )
    return ref.incremental_logits_trace(inputs)


def evaluate_gpt2_perplexity(
    payload: Dict[str, object],
    *,
    calibration_token_ids: Sequence[int],
    eval_token_ids: Sequence[int],
    tokenizer_dir: Path,
    calibration_sha256: str = "",
    eval_sha256: str = "",
    max_eval_tokens: int = 33,
    context_len: int = 32,
    calibration_seq_len: int = 32,
    calibration_n_seqs: int = 8,
    calibration_percentile: float = 99.9,
) -> GPT2PerplexityResult:
    if context_len < 1:
        raise ValueError("context_len must be positive")
    token_budget = min(int(max_eval_tokens), int(context_len) + 1)
    if token_budget < 2:
        raise ValueError("max_eval_tokens/context_len must allow at least two tokens")
    eval_tokens = [int(tok) for tok in eval_token_ids[:token_budget]]
    if len(eval_tokens) < 2:
        raise ValueError("evaluation text produced fewer than two tokens")

    calibration_scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_token_ids,
        n_seqs=calibration_n_seqs,
        seq_len=calibration_seq_len,
        percentile=calibration_percentile,
    )
    _, targets = teacher_forced_inputs_and_targets(eval_tokens)
    vocab_size = int(payload["model_args"]["vocab_size"])
    lm_head_scale = float(calibration_scales.get("lm_head", 1.0))

    golden_logits = run_golden_teacher_forced_logits(payload, eval_tokens, calibration_scales)
    fake_logits = run_fake_quant_teacher_forced_logits(payload, eval_tokens, calibration_scales)
    if len(golden_logits) != len(targets) or len(fake_logits) != len(targets):
        raise RuntimeError("teacher-forced logits/targets length mismatch")

    golden_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32) * np.float32(lm_head_scale), target, vocab_size=vocab_size)
        for row, target in zip(golden_logits, targets)
    ]
    fake_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32) * np.float32(lm_head_scale), target, vocab_size=vocab_size)
        for row, target in zip(fake_logits, targets)
    ]
    golden_ppl, golden_nll = perplexity_from_nlls(golden_nlls)
    fake_ppl, fake_nll = perplexity_from_nlls(fake_nlls)
    rel = abs(golden_ppl - fake_ppl) / max(abs(fake_ppl), 1e-12)
    return GPT2PerplexityResult(
        golden_perplexity=golden_ppl,
        fake_quant_perplexity=fake_ppl,
        relative_delta=float(rel),
        token_count=len(eval_tokens),
        target_count=len(targets),
        golden_nll=golden_nll,
        fake_quant_nll=fake_nll,
        tokenizer_dir=str(tokenizer_dir),
        calibration_sha256=calibration_sha256,
        eval_sha256=eval_sha256,
    )
