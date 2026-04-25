#!/usr/bin/env python3
"""Run a converted GPT-2/nanoGPT-format checkpoint through the golden model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    evaluate_gpt2_perplexity,
    file_sha256,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import stage5_default_ptq_preset_name
from taccel.runtime.tiny_fixture import run_nanogpt_fp32_e2e, run_stage3_tiny_e2e


def _parse_prompt_ids(raw: str) -> List[int]:
    if not raw.strip():
        raise ValueError("--prompt-ids must not be empty")
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _prompt_to_ids(prompt: str, payload: dict) -> List[int]:
    stoi = payload.get("stoi") or {}
    if stoi:
        missing = sorted({ch for ch in prompt if ch not in stoi})
        if missing:
            raise ValueError(f"prompt contains characters absent from checkpoint tokenizer: {missing!r}")
        return [int(stoi[ch]) for ch in prompt]
    raise ValueError("text prompts require tokenizer metadata; use --prompt-ids for converted GPT-2 artifacts")


def _decode_ids(token_ids: List[int], payload: dict) -> str:
    itos = payload.get("itos") or {}
    if itos:
        return "".join(str(itos.get(str(int(tok)), itos.get(int(tok), "?"))) for tok in token_ids)
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated token ids")
    parser.add_argument("--prompt", default=None, help="Text prompt when tokenizer metadata is available")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--ptq-preset", default=None)
    parser.add_argument("--compare-fp32", action="store_true")
    parser.add_argument("--perplexity-text", type=Path, default=None)
    parser.add_argument("--calibration-text", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("software/tests/fixtures/generated/hf_gpt2"),
    )
    parser.add_argument("--max-eval-tokens", type=int, default=33)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--calibration-seq-len", type=int, default=CALIBRATION_SEQ_LEN_LARGE)
    parser.add_argument("--calibration-n-seqs", type=int, default=CALIBRATION_N_SEQS_LARGE)
    parser.add_argument("--calibration-percentile", type=float, default=CALIBRATION_PERCENTILE_DEFAULT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")

    payload = torch.load(args.checkpoint, map_location="cpu")
    if args.prompt_ids is not None:
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
    elif args.prompt is not None:
        prompt_ids = _prompt_to_ids(args.prompt, payload)
    else:
        prompt_ids = [0]
    ptq_preset = stage5_default_ptq_preset_name() if args.ptq_preset is None else args.ptq_preset

    result = run_stage3_tiny_e2e(
        payload,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        ptq_preset=ptq_preset,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "prompt_ids": prompt_ids,
        "ptq_preset": ptq_preset,
        "generated_ids": result.generated,
        "generated_text": _decode_ids(result.generated, payload),
        "max_new_tokens": args.max_new_tokens,
        "min_cosine": result.min_cosine,
        "mean_cosine": float(sum(result.cosine_per_step) / len(result.cosine_per_step)),
        "min_top5_overlap": min(result.top5_overlap_per_step),
    }
    if args.compare_fp32:
        fp32 = run_nanogpt_fp32_e2e(payload, prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens)
        summary.update({
            "fp32_generated_ids": fp32.fp32_generated,
            "fp32_generated_text": _decode_ids(fp32.fp32_generated, payload),
            "fp32_min_top5_overlap": fp32.min_top5_overlap,
            "fp32_top1_match_rate": fp32.top1_match_rate,
            "fp32_min_cosine": fp32.min_fp32_cosine,
            "fp32_mean_cosine": float(sum(fp32.fp32_cosine_per_step) / len(fp32.fp32_cosine_per_step)),
        })
    if args.perplexity_text is not None:
        if args.calibration_text is None:
            raise ValueError("--calibration-text is required when --perplexity-text is supplied")
        calibration_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
        eval_ids = tokenize_text_file(
            args.tokenizer_dir,
            args.perplexity_text,
            max_tokens=args.max_eval_tokens,
        )
        ppl = evaluate_gpt2_perplexity(
            payload,
            calibration_token_ids=calibration_ids,
            eval_token_ids=eval_ids,
            tokenizer_dir=args.tokenizer_dir,
            calibration_sha256=file_sha256(args.calibration_text),
            eval_sha256=file_sha256(args.perplexity_text),
            max_eval_tokens=args.max_eval_tokens,
            context_len=args.context_len,
            calibration_n_seqs=args.calibration_n_seqs,
            calibration_seq_len=args.calibration_seq_len,
            calibration_percentile=args.calibration_percentile,
            ptq_preset=ptq_preset,
        )
        summary.update({
            "perplexity_golden": ppl.golden_perplexity,
            "perplexity_fake_quant": ppl.fake_quant_perplexity,
            "perplexity_relative_delta": ppl.relative_delta,
            "perplexity_token_count": ppl.token_count,
            "perplexity_target_count": ppl.target_count,
            "perplexity_calibration_sha256": ppl.calibration_sha256,
            "perplexity_eval_sha256": ppl.eval_sha256,
            "perplexity_tokenizer_dir": ppl.tokenizer_dir,
            "perplexity_ptq_preset": ppl.ptq_preset,
        })

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"generated_ids: {summary['generated_ids']}")
        if summary["generated_text"]:
            print(f"generated_text: {summary['generated_text']!r}")
        print(f"min_cosine: {summary['min_cosine']:.9f}")
        print(f"mean_cosine: {summary['mean_cosine']:.9f}")
        print(f"min_top5_overlap: {summary['min_top5_overlap']}")
        if args.compare_fp32:
            print(f"fp32_generated_ids: {summary['fp32_generated_ids']}")
            print(f"fp32_top1_match_rate: {summary['fp32_top1_match_rate']:.6f}")
            print(f"fp32_min_cosine: {summary['fp32_min_cosine']:.9f}")
            print(f"fp32_mean_cosine: {summary['fp32_mean_cosine']:.9f}")
        if args.perplexity_text is not None:
            print(f"perplexity_golden: {summary['perplexity_golden']:.6f}")
            print(f"perplexity_fake_quant: {summary['perplexity_fake_quant']:.6f}")
            print(f"perplexity_relative_delta: {summary['perplexity_relative_delta']:.6%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
