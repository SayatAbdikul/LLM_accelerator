#!/usr/bin/env python3
"""Compute FP32 perplexity baseline on the same eval text as the golden gate.

This sets the upper-bound ceiling for any PTQ improvement: golden PPL can
never go below FP32 PPL on the same eval. Use it to decide whether further
PTQ work is worth pursuing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from taccel.runtime.fp32_reference import NanoGPTFP32Reference
from taccel.runtime.gpt2_perplexity import (
    file_sha256,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)


@dataclass
class FP32PerplexityResult:
    fp32_perplexity: float
    fp32_nll: float
    token_count: int
    target_count: int
    tokenizer_dir: str
    eval_sha256: str


def evaluate_fp32_perplexity(
    payload: dict,
    *,
    eval_token_ids,
    tokenizer_dir: Path,
    eval_sha256: str = "",
    max_eval_tokens: int = 33,
    context_len: int = 32,
) -> FP32PerplexityResult:
    if context_len < 1:
        raise ValueError("context_len must be positive")
    token_budget = min(int(max_eval_tokens), int(context_len) + 1)
    if token_budget < 2:
        raise ValueError("max_eval_tokens/context_len must allow at least two tokens")
    eval_tokens = [int(tok) for tok in eval_token_ids[:token_budget]]
    if len(eval_tokens) < 2:
        raise ValueError("evaluation text produced fewer than two tokens")

    inputs, targets = teacher_forced_inputs_and_targets(eval_tokens)
    ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    fp32_logits = ref.incremental_logits_trace(inputs)
    if len(fp32_logits) != len(targets):
        raise RuntimeError(
            f"FP32 logits length {len(fp32_logits)} != targets length {len(targets)}"
        )

    vocab_size = int(payload["model_args"]["vocab_size"])
    nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size)
        for row, target in zip(fp32_logits, targets)
    ]
    ppl, mean_nll = perplexity_from_nlls(nlls)
    return FP32PerplexityResult(
        fp32_perplexity=ppl,
        fp32_nll=mean_nll,
        token_count=len(eval_tokens),
        target_count=len(targets),
        tokenizer_dir=str(tokenizer_dir),
        eval_sha256=eval_sha256,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=33)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(args.tokenizer_dir)

    payload = torch.load(args.checkpoint, map_location="cpu")
    eval_ids = tokenize_text_file(args.tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens)
    result = evaluate_fp32_perplexity(
        payload,
        eval_token_ids=eval_ids,
        tokenizer_dir=args.tokenizer_dir,
        eval_sha256=file_sha256(args.eval_text),
        max_eval_tokens=args.max_eval_tokens,
        context_len=args.context_len,
    )
    out = asdict(result)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"fp32_perplexity: {result.fp32_perplexity:.6f}")
        print(f"fp32_nll: {result.fp32_nll:.6f}")
        print(f"target_count: {result.target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
