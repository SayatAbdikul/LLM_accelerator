#!/usr/bin/env python3
"""Locate the 180× FP32 → INT8 perplexity gap by ablating per-block per-op quantization.

Approach: take the current default preset's calibration scales, then compute
fake-quant perplexity multiple times — each time toggling ONE block's
quantization (or one operation within one block) to FP32. The resulting PPL
drop tells us how much that specific quantized op contributes to the gap.

Outputs a sorted table identifying the largest contributors.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from taccel.runtime.calibration import (
    apply_fc2_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_attn_scale_search_from_token_ids,
    apply_output_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_lm_head_scale_search_from_token_ids,
    apply_output_aware_mlp_scale_search_from_token_ids,
    build_calibration_scales_from_token_ids,
)
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)


@dataclass
class AblationRow:
    label: str
    fp32_groups: tuple[str, ...]
    perplexity: float
    nll: float
    delta_ppl: float  # baseline_ppl - this_ppl (positive = improvement)


def _build_default_scales(payload, tokenizer_dir, calib_text, n_seqs, seq_len, percentile):
    calib = tokenize_text_file(tokenizer_dir, calib_text)
    preset = resolve_stage5_ptq_preset(stage5_default_ptq_preset_name())
    scales = build_calibration_scales_from_token_ids(
        payload, calib, n_seqs=n_seqs, seq_len=seq_len, percentile=percentile,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    scales = apply_stage5_ptq_scale_policy(scales, payload["model_args"], preset)
    if preset.fc2_aware_gelu_blocks:
        scales, _ = apply_fc2_aware_gelu_scale_search_from_token_ids(
            payload, calib, scales, blocks=preset.fc2_aware_gelu_blocks,
            n_seqs=n_seqs, seq_len=seq_len,
        )
    if preset.output_aware_gelu_blocks:
        scales, _ = apply_output_aware_gelu_scale_search_from_token_ids(
            payload, calib, scales, blocks=preset.output_aware_gelu_blocks,
            ptq_preset=preset, n_seqs=n_seqs, seq_len=seq_len,
        )
    if preset.output_aware_mlp_blocks:
        scales, _ = apply_output_aware_mlp_scale_search_from_token_ids(
            payload, calib, scales, blocks=preset.output_aware_mlp_blocks,
            ptq_preset=preset, n_seqs=n_seqs, seq_len=seq_len,
            include_pair_candidates=preset.output_aware_include_pairs,
            passes=preset.output_aware_mlp_passes,
        )
    if preset.output_aware_attn_blocks:
        scales, _ = apply_output_aware_attn_scale_search_from_token_ids(
            payload, calib, scales, blocks=preset.output_aware_attn_blocks,
            ptq_preset=preset, n_seqs=n_seqs, seq_len=seq_len,
        )
    if preset.output_aware_lm_head:
        scales, _ = apply_output_aware_lm_head_scale_search_from_token_ids(
            payload, calib, scales, ptq_preset=preset,
            n_seqs=n_seqs, seq_len=seq_len,
        )
    return preset, scales


def _eval_ppl(ref, inputs, targets, vocab_size, lm_head_scale, fp32_groups):
    logits = ref.incremental_logits_trace(inputs, fp32_groups=fp32_groups)
    if len(logits) != len(targets):
        raise RuntimeError(f"logits/targets length mismatch: {len(logits)} vs {len(targets)}")
    nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32) * np.float32(lm_head_scale),
                             tgt, vocab_size=vocab_size)
        for row, tgt in zip(logits, targets)
    ]
    ppl, mean_nll = perplexity_from_nlls(nlls)
    return ppl, mean_nll


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=258)
    parser.add_argument("--context-len", type=int, default=257)
    parser.add_argument("--calibration-n-seqs", type=int, default=CALIBRATION_N_SEQS_LARGE)
    parser.add_argument("--calibration-seq-len", type=int, default=CALIBRATION_SEQ_LEN_LARGE)
    parser.add_argument("--calibration-percentile", type=float, default=CALIBRATION_PERCENTILE_DEFAULT)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    n_layer = int(payload["model_args"]["n_layer"])
    vocab_size = int(payload["model_args"]["vocab_size"])

    print(f"# Building default preset calibration scales (n_seqs={args.calibration_n_seqs}, "
          f"seq_len={args.calibration_seq_len})...")
    preset, scales = _build_default_scales(
        payload, args.tokenizer_dir, args.calibration_text,
        args.calibration_n_seqs, args.calibration_seq_len, args.calibration_percentile,
    )
    print(f"# Preset: {preset.name}")

    eval_ids = tokenize_text_file(args.tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens)
    token_budget = min(int(args.max_eval_tokens), int(args.context_len) + 1)
    eval_tokens = [int(tok) for tok in eval_ids[:token_budget]]
    inputs, targets = teacher_forced_inputs_and_targets(eval_tokens)
    lm_head_scale = float(scales.get("lm_head", 1.0))

    ref = NanoGPTFQReference(
        payload["state_dict"], payload["model_args"], scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(payload["model_args"], preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(preset),
    )

    # Baseline: all INT8 (no FP32 groups)
    print("# Computing baseline (full INT8) perplexity at "
          f"max_eval_tokens={args.max_eval_tokens}...")
    base_ppl, base_nll = _eval_ppl(ref, inputs, targets, vocab_size, lm_head_scale, fp32_groups=None)
    print(f"# baseline_ppl={base_ppl:.2f}  baseline_nll={base_nll:.4f}")

    # Ablation 1: full FP32 (everything in fp32) for sanity
    full_fp32_groups = {
        "embeddings", "qkv", "softmax", "out_proj", "mlp_full", "ln_f", "lm_head",
    }
    full_ppl, full_nll = _eval_ppl(ref, inputs, targets, vocab_size, lm_head_scale,
                                    fp32_groups=full_fp32_groups)
    print(f"# full_fp32_via_ablations_ppl={full_ppl:.2f}  (sanity check; should approach FP32 ceiling)")
    print()

    rows: list[AblationRow] = []

    def add_row(label: str, groups: set[str]) -> None:
        ppl, nll = _eval_ppl(ref, inputs, targets, vocab_size, lm_head_scale,
                              fp32_groups=groups)
        rows.append(AblationRow(label=label, fp32_groups=tuple(sorted(groups)),
                                 perplexity=ppl, nll=nll,
                                 delta_ppl=base_ppl - ppl))

    # Per-block MLP ablation
    print("# Ablating per-block MLP (full block-MLP in FP32)...")
    for L in range(n_layer):
        add_row(f"mlp_block_{L}", {f"mlp_block_{L}"})

    # Per-block attn_v ablation
    print("# Ablating per-block attn_v...")
    for L in range(n_layer):
        add_row(f"attn_v_block_{L}", {f"attn_v_block_{L}"})

    # Per-block softmax ablation
    print("# Ablating per-block softmax (all heads)...")
    for L in range(n_layer):
        head_groups = {f"softmax_block_{L}_head_{H}" for H in range(int(payload["model_args"]["n_head"]))}
        add_row(f"softmax_block_{L}", head_groups)

    # Global ablations for key components
    print("# Ablating global components...")
    for label in ("embeddings", "qkv", "softmax", "out_proj", "mlp", "ln_f", "lm_head",
                  "lm_head_weight_fp32", "lm_head_output_fp32", "lm_head_requant_fp32"):
        add_row(f"GLOBAL_{label}", {label})

    rows.sort(key=lambda r: r.delta_ppl, reverse=True)

    print()
    print(f"{'Ablation':<35}{'PPL':>14}{'NLL':>10}{'Δppl':>14}{'Δppl/baseline':>16}")
    print("-" * 90)
    for r in rows:
        frac = r.delta_ppl / max(base_ppl, 1e-9)
        print(f"{r.label:<35}{r.perplexity:>14.2f}{r.nll:>10.4f}{r.delta_ppl:>+14.2f}{frac:>15.1%}")

    print()
    print(f"baseline_ppl={base_ppl:.2f}")
    print(f"all-FP32-groups_ppl={full_ppl:.2f}  (note: ablations are not exhaustive; "
          "this is one composite check)")

    if args.json_out:
        out = {
            "preset": preset.name,
            "max_eval_tokens": args.max_eval_tokens,
            "context_len": args.context_len,
            "baseline_ppl": base_ppl,
            "baseline_nll": base_nll,
            "full_fp32_groups_ppl": full_ppl,
            "rows": [
                {"label": r.label, "fp32_groups": list(r.fp32_groups),
                 "perplexity": r.perplexity, "nll": r.nll, "delta_ppl": r.delta_ppl}
                for r in rows
            ],
        }
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"# Full results written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
