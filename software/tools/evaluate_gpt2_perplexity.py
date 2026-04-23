#!/usr/bin/env python3
"""Evaluate Stage 5 GPT-2 perplexity offline against fake-quant reference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from taccel.runtime.gpt2_perplexity import (
    evaluate_gpt2_perplexity,
    file_sha256,
    tokenize_text_file,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=33)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--calibration-seq-len", type=int, default=32)
    parser.add_argument("--calibration-n-seqs", type=int, default=8)
    parser.add_argument("--calibration-percentile", type=float, default=99.9)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(args.tokenizer_dir)

    payload = torch.load(args.checkpoint, map_location="cpu")
    calibration_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    eval_ids = tokenize_text_file(
        args.tokenizer_dir,
        args.eval_text,
        max_tokens=args.max_eval_tokens,
    )
    result = evaluate_gpt2_perplexity(
        payload,
        calibration_token_ids=calibration_ids,
        eval_token_ids=eval_ids,
        tokenizer_dir=args.tokenizer_dir,
        calibration_sha256=file_sha256(args.calibration_text),
        eval_sha256=file_sha256(args.eval_text),
        max_eval_tokens=args.max_eval_tokens,
        context_len=args.context_len,
        calibration_seq_len=args.calibration_seq_len,
        calibration_n_seqs=args.calibration_n_seqs,
        calibration_percentile=args.calibration_percentile,
    )
    out = asdict(result)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"golden_perplexity: {result.golden_perplexity:.6f}")
        print(f"fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
        print(f"relative_delta: {result.relative_delta:.6%}")
        print(f"target_count: {result.target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
