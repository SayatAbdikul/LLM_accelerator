#!/usr/bin/env python3
"""LayerNorm epsilon experiment: quantify the infer/calibrate epsilon interaction.

Three conditions:
  A: infer=1e-6, calibrate=1e-5  (current default — SFU hardcoded, model_args default)
  B: infer=1e-6, calibrate=1e-6  (align calibration to SFU — previously hurt PPL)
  C: infer=1e-5, calibrate=1e-5  (both match GPT-2 FP32 model_args)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from taccel.runtime.calibration import build_calibration_scales_from_token_ids
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    GPT2_DEFAULT_PTQ_PRESET,
    perplexity_from_nlls,
    stable_cross_entropy,
    teacher_forced_inputs_and_targets,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)


def _run_condition(
    payload: dict,
    calibration_token_ids: Sequence[int],
    eval_token_ids: Sequence[int],
    *,
    ln_eps_calibration: float,
    ln_eps_inference: float,
    calibration_n_seqs: int,
    calibration_seq_len: int,
    calibration_percentile: float,
    ptq_preset_name: str,
) -> Dict[str, float]:
    resolved = resolve_stage5_ptq_preset(ptq_preset_name)

    scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_token_ids,
        n_seqs=calibration_n_seqs,
        seq_len=calibration_seq_len,
        percentile=calibration_percentile,
        activation_percentile_overrides=(resolved.activation_percentile_nodes or None),
        hessian_gelu_blocks=resolved.hessian_gelu_blocks,
        ln_eps_calibration=ln_eps_calibration,
    )
    scales = apply_stage5_ptq_scale_policy(scales, payload["model_args"], resolved)

    eval_tokens = list(eval_token_ids)
    inputs, targets = teacher_forced_inputs_and_targets(eval_tokens)
    vocab_size = int(payload["model_args"]["vocab_size"])
    lm_head_scale = float(scales.get("lm_head", 1.0))

    ref = NanoGPTFQReference(
        state_dict=payload["state_dict"],
        model_args=payload["model_args"],
        scales=scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(payload["model_args"], resolved),
        raw_residual1_blocks=stage5_raw_residual1_blocks(resolved),
        raw_residual2_blocks=stage5_raw_residual2_blocks(resolved),
        ln_eps=ln_eps_inference,
    )
    fake_logits: List[np.ndarray] = list(ref.incremental_logits_trace(inputs))

    nlls = [
        stable_cross_entropy(
            np.asarray(row, dtype=np.float32) * np.float32(lm_head_scale),
            target,
            vocab_size=vocab_size,
        )
        for row, target in zip(fake_logits, targets)
    ]
    ppl, nll = perplexity_from_nlls(nlls)
    return {"fake_quant_perplexity": ppl, "mean_nll": nll}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=33)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--calibration-seq-len", type=int, default=CALIBRATION_SEQ_LEN_LARGE)
    parser.add_argument("--calibration-n-seqs", type=int, default=CALIBRATION_N_SEQS_LARGE)
    parser.add_argument("--calibration-percentile", type=float, default=CALIBRATION_PERCENTILE_DEFAULT)
    parser.add_argument("--ptq-preset", default=GPT2_DEFAULT_PTQ_PRESET)
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu")
    calib_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    eval_ids = tokenize_text_file(
        args.tokenizer_dir, args.eval_text,
        max_tokens=min(args.max_eval_tokens, args.context_len + 1),
    )

    conditions = [
        ("A", 1e-6, 1e-5, "infer=1e-6  calib=1e-5  (current default)"),
        ("B", 1e-6, 1e-6, "infer=1e-6  calib=1e-6  (align calib to SFU)"),
        ("C", 1e-5, 1e-5, "infer=1e-5  calib=1e-5  (match GPT-2 FP32)"),
    ]

    common = dict(
        calibration_n_seqs=args.calibration_n_seqs,
        calibration_seq_len=args.calibration_seq_len,
        calibration_percentile=args.calibration_percentile,
        ptq_preset_name=args.ptq_preset,
    )

    results = {}
    for label, ln_eps_infer, ln_eps_calib, description in conditions:
        print(f"Running condition {label}: {description} ...", flush=True)
        r = _run_condition(
            payload,
            calib_ids,
            eval_ids,
            ln_eps_calibration=ln_eps_calib,
            ln_eps_inference=ln_eps_infer,
            **common,
        )
        results[label] = (r, description)
        print(f"  fake_quant_perplexity: {r['fake_quant_perplexity']:.1f}")

    baseline_ppl = results["A"][0]["fake_quant_perplexity"]
    print()
    print(f"{'Cond':<4}  {'PPL':>8}  {'vs A':>8}  Description")
    print("-" * 70)
    for label, (r, desc) in results.items():
        ppl = r["fake_quant_perplexity"]
        delta = ppl - baseline_ppl
        sign = "+" if delta >= 0 else ""
        print(f"  {label}   {ppl:>8.1f}  {sign}{delta:>7.1f}  {desc}")

    print()
    print("Interpretation:")
    ppl_b = results["B"][0]["fake_quant_perplexity"]
    ppl_c = results["C"][0]["fake_quant_perplexity"]
    if ppl_b > baseline_ppl:
        print(f"  B > A (+{ppl_b-baseline_ppl:.0f}): calibration with 1e-6 hurts —")
        print("    the 1e-5 calib mismatch is a useful PTQ hack (wider scales = less clipping).")
    else:
        print(f"  B < A ({ppl_b-baseline_ppl:.0f}): aligning calib to SFU (1e-6) helps —")
        print("    the mismatch was accidental noise.")
    if ppl_c < baseline_ppl:
        print(f"  C < A ({ppl_c-baseline_ppl:.0f}): matching GPT-2 FP32 (both 1e-5) is best —")
        print("    but requires changing the SFU to 1e-5.")
    else:
        print(f"  C >= A: using 1e-5 for both is not better than current state.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
