#!/usr/bin/env python3
"""Diagnostic: PPL with weight-only INT8 QDQ + FP32 activations everywhere.

This isolates the cost of weight quantization from activation quantization,
without calibration mismatch. It builds NanoGPTFP32Reference (pure FP32
inference) but replaces each linear weight tensor with its INT8 QDQ form
(`_linear_components(W)[2]` from fake_quant_reference). Activations stay
FP32 throughout — no _qdq calls. The resulting PPL tells us:

- ≈ FP32 ceiling (53-100):   activations dominate the FP32→INT8 gap
                              → Branch B (residual-stream activation quant) is
                                the right direction, but the substitution
                                experiment needs re-calibration to test cleanly.
- ≈ baseline INT8 (5k-6k):   weights dominate
                              → Branch C (AWQ) is the right direction.
- in between (500-2,000):    both contribute, weights probably dominate
                              → Branch C still right, more headroom available.

Usage::

    PYTHONPATH=software python3 software/tools/diagnose_weight_only_qdq_ceiling.py \\
      software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \\
      --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \\
      --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \\
      --max-eval-tokens 257
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from taccel.runtime.fp32_reference import (
    NanoGPTFP32Reference,
    build_weight_only_int8_reference,
)
from taccel.runtime.gpt2_perplexity import (
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)


# Re-exported for callers that imported the old name from this module.
build_weight_only_qdq_reference = build_weight_only_int8_reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=257)
    parser.add_argument("--context-len", type=int, default=256)
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu")
    eval_ids = tokenize_text_file(
        args.tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens
    )
    token_budget = min(int(args.max_eval_tokens), int(args.context_len) + 1)
    eval_tokens = [int(tok) for tok in eval_ids[:token_budget]]
    inputs, targets = teacher_forced_inputs_and_targets(eval_tokens)

    # Pure FP32 baseline (sanity)
    print("Building FP32 reference...")
    fp32_ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    fp32_logits = fp32_ref.incremental_logits_trace(inputs)
    vocab_size = int(payload["model_args"]["vocab_size"])
    fp32_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(fp32_logits, targets)
    ]
    fp32_ppl, _ = perplexity_from_nlls(fp32_nlls)
    print(f"FP32 (pure):                 PPL = {fp32_ppl:.4f}")

    # Weight-only QDQ (per-channel) + FP32 activations
    print("Building weight-only QDQ (per_channel) reference...")
    qdq_ref_pc = build_weight_only_qdq_reference(payload, weight_mode="per_channel")
    qdq_logits_pc = qdq_ref_pc.incremental_logits_trace(inputs)
    qdq_nlls_pc = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(qdq_logits_pc, targets)
    ]
    qdq_ppl_pc, _ = perplexity_from_nlls(qdq_nlls_pc)
    print(f"Weight-only QDQ (per-channel, FP32 acts):  PPL = {qdq_ppl_pc:.4f}")

    # Weight-only QDQ (mean-scale) — production codebase's approximation
    print("Building weight-only QDQ (mean_scale) reference...")
    qdq_ref_ms = build_weight_only_qdq_reference(payload, weight_mode="mean_scale")
    qdq_logits_ms = qdq_ref_ms.incremental_logits_trace(inputs)
    qdq_nlls_ms = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32), tgt, vocab_size=vocab_size)
        for row, tgt in zip(qdq_logits_ms, targets)
    ]
    qdq_ppl_ms, _ = perplexity_from_nlls(qdq_nlls_ms)
    print(f"Weight-only QDQ (mean-scale, FP32 acts):   PPL = {qdq_ppl_ms:.4f}")
    print()
    print(f"FP32 ceiling:                              PPL = {fp32_ppl:.4f}")
    print(f"Per-channel weight quant cost:             {qdq_ppl_pc - fp32_ppl:.2f} ({qdq_ppl_pc/fp32_ppl:.2f}× FP32)")
    print(f"Mean-scale weight approximation cost:      {qdq_ppl_ms - fp32_ppl:.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
