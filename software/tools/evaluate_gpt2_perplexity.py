#!/usr/bin/env python3
"""Evaluate Stage 5 GPT-2 perplexity offline against fake-quant reference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
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
    parser.add_argument("--calibration-seq-len", type=int, default=CALIBRATION_SEQ_LEN_LARGE)
    parser.add_argument("--calibration-n-seqs", type=int, default=CALIBRATION_N_SEQS_LARGE)
    parser.add_argument("--calibration-percentile", type=float, default=CALIBRATION_PERCENTILE_DEFAULT)
    parser.add_argument("--output-aware-search-n-seqs", type=int)
    parser.add_argument("--output-aware-search-seq-len", type=int)
    parser.add_argument("--output-aware-search-workers", type=int)
    parser.add_argument("--output-aware-include-pairs", action="store_true")
    parser.add_argument("--ptq-preset", default=None)
    parser.add_argument(
        "--skip-fp32-ceiling",
        action="store_true",
        help="Skip Phase 0A FP32-reference ceiling computation (saves ~30s/run).",
    )
    parser.add_argument(
        "--debug-fp32-kv-cache",
        action="store_true",
        help=(
            "Phase 0B diagnostic: keep K/V cache in FP32 in the fake-quant "
            "reference (Q stays INT8). Tests whether K/V cache compounding "
            "noise is the long-eval bottleneck. Reference-only — does NOT "
            "ship to the deployed bundle."
        ),
    )
    parser.add_argument(
        "--debug-fp32-residual-stream",
        action="store_true",
        help=(
            "Phase 1 Branch B diagnostic: route the residual stream through "
            "FP32 in the fake-quant reference (no INT8 quantization on "
            "block_residual1/2 or LN outputs ln1/ln2/ln_f; matmul outputs "
            "Q/K/V/fc1/fc2/lm_head still INT8). Reference-only — does NOT "
            "ship. Used to test whether residual-stream INT8 quantization "
            "is the dominant long-eval bottleneck."
        ),
    )
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
        ptq_preset=args.ptq_preset,
        output_aware_search_n_seqs=args.output_aware_search_n_seqs,
        output_aware_search_seq_len=args.output_aware_search_seq_len,
        output_aware_search_workers=args.output_aware_search_workers,
        output_aware_include_pairs=args.output_aware_include_pairs,
        compute_fp32_ceiling=(not args.skip_fp32_ceiling),
        debug_fp32_kv_cache=args.debug_fp32_kv_cache,
        debug_fp32_residual_stream=args.debug_fp32_residual_stream,
    )
    out = asdict(result)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        # W8A32 (weight_only_int8) intentionally has no golden bundle path
        # on the current ISA, so `golden_perplexity` / `relative_delta`
        # come back as NaN. Surface a W8A32-specific summary instead of
        # the standard golden/fake_quant comparison.
        is_w8a32 = result.ptq_preset == "weight_only_int8"
        print(f"fp32_perplexity: {result.fp32_perplexity:.6f}")
        if is_w8a32:
            print(f"weight_only_int8 (W8A32) fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
            if not math.isnan(result.fp32_perplexity):
                delta = result.fake_quant_perplexity - result.fp32_perplexity
                ratio = result.fake_quant_perplexity / max(result.fp32_perplexity, 1e-12)
                print(
                    f"W8A32 vs FP32: Δ {delta:+.4f} PPL ({ratio:.4f}× FP32)"
                )
            print("golden_perplexity / relative_delta: nan (no W8A32 bundle path on current ISA)")
        else:
            print(f"golden_perplexity: {result.golden_perplexity:.6f}")
            print(f"fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
            print(f"relative_delta: {result.relative_delta:.6%}")
        print(f"target_count: {result.target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
