#!/usr/bin/env python3
"""W8A8 Phase-0 PPL gate.

Runs the W8A16 baseline simulator reference and the new W8A8 simulator
reference (which adds per-tensor INT8 quant/dequant at every inter-op
storage point) on the same GPT-2 checkpoint and tokens, reports the
PPL delta. This is the decisive kill-switch for the W8A8 implementation
plan: if PPL is GREEN (≤+5 above the W8A16 baseline of ~55.76), the
plan is worth executing; if RED, pivot to W4A8.

Usage::

    PYTHONPATH=software python software/tools/w8a8_ppl_gate.py \\
        software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \\
        --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \\
        --calibration-text software/tests/fixtures/wikitext2_stage5_calibration.txt \\
        --eval-text software/tests/fixtures/wikitext2_stage5_calibration.txt \\
        --max-eval-tokens 33 --context-len 32 \\
        --ptq-preset weight_only_int8_quarot

Tip: start with --max-eval-tokens 33 for a smoke test (~30s), then
re-run with 257 for the real gate (~5-15 min).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    file_sha256,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.calibration import (
    apply_quarot_rotation_from_token_ids,
    build_calibration_scales_from_token_ids,
)
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
)
from taccel.runtime.w8a16_simulator_reference import NanoGPTW8A16SimulatorReference
from taccel.runtime.w8a8_simulator_reference import NanoGPTW8A8SimulatorReference


def _build_calibration_scales(
    payload: dict,
    calibration_token_ids: Sequence[int],
    resolved_preset,
    *,
    n_seqs: int,
    seq_len: int,
    percentile: float,
):
    """Apply QuaRot (if preset enables it) then build per-node activation scales.

    Mirrors the weight-only branch of `evaluate_gpt2_perplexity`. QuaRot
    runs BEFORE calibration so scales reflect the rotated (near-isotropic)
    distribution — same rotate→calibrate ordering the production path uses.
    """
    if resolved_preset.quarot_enabled:
        apply_quarot_rotation_from_token_ids(
            payload,
            calibration_token_ids,
            seed=resolved_preset.quarot_seed,
            kind=resolved_preset.quarot_kind,
        )
    scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_token_ids,
        n_seqs=n_seqs,
        seq_len=seq_len,
        percentile=percentile,
        activation_percentile_overrides=(
            resolved_preset.activation_percentile_nodes or None
        ),
        hessian_gelu_blocks=resolved_preset.hessian_gelu_blocks,
    )
    scales = apply_stage5_ptq_scale_policy(
        scales, payload["model_args"], resolved_preset,
    )
    return scales


def _run_ref(ref, eval_tokens, vocab_size):
    inputs, targets = teacher_forced_inputs_and_targets(eval_tokens)
    logits = ref.run_teacher_forced(inputs)
    if len(logits) != len(targets):
        raise RuntimeError(
            f"logits/targets length mismatch: {len(logits)} vs {len(targets)}"
        )
    nlls = [
        stable_cross_entropy(
            np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size,
        )
        for row, target in zip(logits, targets)
    ]
    ppl, nll = perplexity_from_nlls(nlls)
    return ppl, nll, nlls


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
    parser.add_argument(
        "--ptq-preset",
        default="weight_only_int8_quarot",
        help="PTQ preset; default 'weight_only_int8_quarot' (the W8A16 baseline at 55.76 PPL).",
    )
    parser.add_argument("--per-block-ablation", action="store_true",
                        help="Run W8A8 on blocks 0..k and report PPL per k.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer_dir.exists():
        raise FileNotFoundError(args.tokenizer_dir)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibration_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    eval_ids = tokenize_text_file(
        args.tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens,
    )
    token_budget = min(int(args.max_eval_tokens), int(args.context_len) + 1)
    eval_tokens = [int(tok) for tok in eval_ids[:token_budget]]
    if len(eval_tokens) < 2:
        raise ValueError("need at least two eval tokens")
    vocab_size = int(payload["model_args"]["vocab_size"])

    resolved_preset = resolve_stage5_ptq_preset(args.ptq_preset)

    # Build calibration scales (single set of scales used by both refs so
    # the only difference between W8A16 and W8A8 numbers is the inter-op
    # INT8 round-trip, not the calibration).
    scales = _build_calibration_scales(
        payload,
        calibration_ids,
        resolved_preset,
        n_seqs=args.calibration_n_seqs,
        seq_len=args.calibration_seq_len,
        percentile=args.calibration_percentile,
    )

    # W8A16 baseline.
    ref16 = NanoGPTW8A16SimulatorReference(payload, calibration_scales=scales)
    ppl_w8a16, nll_w8a16, _ = _run_ref(ref16, eval_tokens, vocab_size)

    # W8A8 (all blocks).
    ref8 = NanoGPTW8A8SimulatorReference(payload, calibration_scales=scales)
    ppl_w8a8, nll_w8a8, _ = _run_ref(ref8, eval_tokens, vocab_size)

    delta_ppl = ppl_w8a8 - ppl_w8a16
    rel_pct = 100.0 * delta_ppl / max(ppl_w8a16, 1e-12)

    # Decision gate (vs W8A16 baseline of ~55.76 on production preset).
    if ppl_w8a8 <= 60.0:
        verdict = "GREEN"
        decision = "PPL ≤ 60 — proceed to Phase 1 (claim opcode 0x1C)."
    elif ppl_w8a8 <= 75.0:
        verdict = "YELLOW"
        decision = "60 < PPL ≤ 75 — try output-aware scale search before Phase 1."
    else:
        verdict = "RED"
        decision = "PPL > 75 — abort the W8A8 plan, pivot to W4A8 weight-only."

    summary = {
        "checkpoint": str(args.checkpoint),
        "ptq_preset": args.ptq_preset,
        "max_eval_tokens": args.max_eval_tokens,
        "context_len": args.context_len,
        "token_count": len(eval_tokens),
        "target_count": len(eval_tokens) - 1,
        "calibration_n_seqs": args.calibration_n_seqs,
        "calibration_seq_len": args.calibration_seq_len,
        "calibration_percentile": args.calibration_percentile,
        "calibration_sha256": file_sha256(args.calibration_text),
        "eval_sha256": file_sha256(args.eval_text),
        "w8a16_perplexity": float(ppl_w8a16),
        "w8a16_nll": float(nll_w8a16),
        "w8a8_perplexity": float(ppl_w8a8),
        "w8a8_nll": float(nll_w8a8),
        "delta_ppl": float(delta_ppl),
        "rel_delta_pct": float(rel_pct),
        "phase0_verdict": verdict,
        "phase0_decision": decision,
    }

    if args.per_block_ablation:
        # Per-block ablation: not implemented here as it requires the
        # W8A8 ref to support "INT8 only on blocks 0..k". Skipped for
        # the Phase-0 gate run; if needed, add a `quantize_only_blocks`
        # kwarg to NanoGPTW8A8SimulatorReference later.
        summary["per_block_ablation"] = "not implemented in v0"

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"checkpoint: {args.checkpoint}")
        print(f"preset:     {args.ptq_preset}")
        print(f"tokens:     {len(eval_tokens)} (targets: {len(eval_tokens)-1})")
        print(f"")
        print(f"W8A16 perplexity: {ppl_w8a16:.4f}  (baseline)")
        print(f"W8A8  perplexity: {ppl_w8a8:.4f}")
        print(f"delta:            {delta_ppl:+.4f} PPL ({rel_pct:+.2f}%)")
        print(f"")
        print(f"Phase-0 verdict:  {verdict}")
        print(f"Decision:         {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
