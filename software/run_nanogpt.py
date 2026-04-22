#!/usr/bin/env python3
"""Run a nanoGPT-format checkpoint through the golden-model HostRunner path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from taccel.runtime.tiny_fixture import run_nanogpt_fp32_e2e, run_stage3_tiny_e2e


def _parse_prompt_ids(raw: str) -> List[int]:
    if not raw.strip():
        raise ValueError("--prompt-ids must not be empty")
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _prompt_to_ids(prompt: str, stoi: dict) -> List[int]:
    if not prompt:
        raise ValueError("--prompt must not be empty")
    missing = sorted({ch for ch in prompt if ch not in stoi})
    if missing:
        raise ValueError(f"prompt contains characters absent from checkpoint tokenizer: {missing!r}")
    return [int(stoi[ch]) for ch in prompt]


def _decode_chars(token_ids: List[int], itos: dict) -> str:
    out = []
    for tok in token_ids:
        key = str(int(tok))
        out.append(str(itos.get(key, itos.get(int(tok), "?"))))
    return "".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated token ids, e.g. 0,1,2")
    parser.add_argument("--prompt", default=None, help="Character prompt using the checkpoint tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Accepted for CLI compatibility; Stage 4 uses checkpoint block_size")
    parser.add_argument("--compare-fp32", action="store_true", help="Add rank-based comparison against true PyTorch FP32 nanoGPT")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")

    payload = torch.load(args.checkpoint, map_location="cpu")
    stoi = payload.get("stoi", {})
    itos = payload.get("itos", {})
    if args.prompt_ids is not None:
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
    elif args.prompt is not None:
        prompt_ids = _prompt_to_ids(args.prompt, stoi)
    else:
        prompt_ids = [0]

    result = run_stage3_tiny_e2e(
        payload,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "prompt_ids": prompt_ids,
        "generated_ids": result.generated,
        "generated_text": _decode_chars(result.generated, itos),
        "max_new_tokens": args.max_new_tokens,
        "min_cosine": result.min_cosine,
        "mean_cosine": float(sum(result.cosine_per_step) / len(result.cosine_per_step)),
        "min_top5_overlap": min(result.top5_overlap_per_step),
    }
    if args.compare_fp32:
        fp32 = run_nanogpt_fp32_e2e(
            payload,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
        )
        summary.update({
            "fp32_generated_ids": fp32.fp32_generated,
            "fp32_generated_text": _decode_chars(fp32.fp32_generated, itos),
            "fp32_min_top5_overlap": fp32.min_top5_overlap,
            "fp32_top1_in_golden_top5_all": fp32.fp32_top1_in_golden_top5_all,
            "golden_top1_in_fp32_top5_all": fp32.golden_top1_in_fp32_top5_all,
            "fp32_top1_match_rate": fp32.top1_match_rate,
            "fp32_min_cosine": fp32.min_fp32_cosine,
            "fp32_mean_cosine": float(sum(fp32.fp32_cosine_per_step) / len(fp32.fp32_cosine_per_step)),
        })
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"generated_ids: {summary['generated_ids']}")
        print(f"generated_text: {summary['generated_text']!r}")
        print(f"min_cosine: {summary['min_cosine']:.9f}")
        print(f"mean_cosine: {summary['mean_cosine']:.9f}")
        print(f"min_top5_overlap: {summary['min_top5_overlap']}")
        if args.compare_fp32:
            print(f"fp32_generated_ids: {summary['fp32_generated_ids']}")
            print(f"fp32_generated_text: {summary['fp32_generated_text']!r}")
            print(f"fp32_min_top5_overlap: {summary['fp32_min_top5_overlap']}")
            print(
                "fp32_top1_in_golden_top5_all: "
                f"{summary['fp32_top1_in_golden_top5_all']}"
            )
            print(
                "golden_top1_in_fp32_top5_all: "
                f"{summary['golden_top1_in_fp32_top5_all']}"
            )
            print(f"fp32_top1_match_rate: {summary['fp32_top1_match_rate']:.6f}")
            print(f"fp32_min_cosine: {summary['fp32_min_cosine']:.9f}")
            print(f"fp32_mean_cosine: {summary['fp32_mean_cosine']:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
