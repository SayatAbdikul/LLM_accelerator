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
        "--simulator-backed",
        action="store_true",
        help=(
            "weight_only_int8 path: route the `golden` path through the "
            "compiled ProgramBundle + simulator (W8A16 deployment) instead "
            "of WeightOnlyHostRunner. When set, the `fake_quant` path uses "
            "the like-for-like NumPy reference "
            "(NanoGPTW8A16SimulatorReference) and `relative_delta` measures "
            "codegen correctness, not host-runner-vs-reference identity. "
            "Note: ~100× slower than WeightOnlyHostRunner; expect ~2 min "
            "for 33-token / ~14 min for 257-token at GPT-2 124M."
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
        simulator_backed=args.simulator_backed,
    )
    out = asdict(result)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        # weight_only_int8 populates `golden_perplexity` via
        # `WeightOnlyHostRunner` by default, so both golden and fake_quant
        # come back as real numbers — but they wrap the same numpy QDQ
        # helper, so `relative_delta` is ~0 by construction. Print a
        # weight_only_int8-specific banner so users don't mistake this for
        # a real golden-vs-fake-quant gap.
        is_weight_only = result.ptq_preset == "weight_only_int8"
        print(f"fp32_perplexity: {result.fp32_perplexity:.6f}")
        if is_weight_only:
            # Label the run by what the golden + fake_quant paths actually
            # represent based on --simulator-backed.
            if args.simulator_backed:
                label = "W8A16 simulator-backed"
                fake_label = "M4-E like-for-like NumPy reference"
                gate_label = f"{label} bundle vs {fake_label}"
                note = (
                    "note: simulator-backed path exercises the compiled "
                    "ProgramBundle through the golden simulator (W8A16 "
                    "deployment). relative_delta measures codegen-vs-"
                    "reference agreement (target ~FP16 ULP)."
                )
            else:
                label = "weight_only_int8 (host-runner)"
                fake_label = "weight_only_int8 (numpy QDQ)"
                gate_label = "host-runner vs QDQ reference"
                note = (
                    "note: golden path is the host-runner option — host FP32 "
                    "+ INT8 weight storage. Both paths wrap the same QDQ "
                    "helper, so relative_delta is ~0 by construction. "
                    "Pass --simulator-backed for the W8A16 simulator-backed "
                    "deployment path."
                )
            print(f"{label} fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
            print(f"{label} golden_perplexity:     {result.golden_perplexity:.6f}")
            print(f"relative_delta ({gate_label}): {result.relative_delta:.6%}")
            if not math.isnan(result.fp32_perplexity):
                delta = result.golden_perplexity - result.fp32_perplexity
                ratio = result.golden_perplexity / max(result.fp32_perplexity, 1e-12)
                print(
                    f"{label} vs FP32 ceiling: Δ {delta:+.4f} PPL ({ratio:.4f}× FP32)"
                )
            print(note)
        else:
            print(f"golden_perplexity: {result.golden_perplexity:.6f}")
            print(f"fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
            print(f"relative_delta: {result.relative_delta:.6%}")
        print(f"target_count: {result.target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
