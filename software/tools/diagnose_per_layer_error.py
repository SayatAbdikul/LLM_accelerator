#!/usr/bin/env python3
"""Phase A1: Per-layer error decomposition between FP32 reference and fake-quant.

For each activation node, compute the per-step error between
NanoGPTFP32Reference (true FP32 forward) and NanoGPTFQReference
(post-calibration fake-quant forward) and tabulate by total error
contribution. Identifies which layers dominate the FP32→INT8 PPL gap.

Pipeline:
  1. Load payload, tokenize calibration & eval text.
  2. Apply the same calibration pipeline as `evaluate_gpt2_perplexity`
     (rotation if enabled → AWQ if enabled → build_calibration_scales →
     scale_policy → BC if enabled → search steps). Result: same
     `calibration_scales` + mutated `state_dict` as a real run.
  3. Construct NanoGPTFP32Reference (uses ORIGINAL state_dict pre-mutation
     by separate copy) and NanoGPTFQReference (uses mutated state_dict +
     final scales).
  4. For each of N eval steps, run both `_decode_incremental_step` with
     `trace=...` and accumulate per-node error stats.
  5. Tabulate top-K nodes by L2 error contribution.

Usage:
  PYTHONPATH=software python3 software/tools/diagnose_per_layer_error.py \\
    software/tests/fixtures/generated/gpt2_converted_nanogpt.pt \\
    --tokenizer-dir software/tests/fixtures/generated/hf_gpt2 \\
    --calibration-text software/tests/fixtures/generated/wikitext2_stage5_calibration.txt \\
    --eval-text software/tests/fixtures/generated/wikitext2_stage5_eval.txt \\
    --ptq-preset output_aware_mlp_lm_head_0_11_pc_full_bc \\
    --n-eval-steps 16
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from taccel.runtime.calibration import (
    apply_awq_from_token_ids,
    apply_bias_correction_from_token_ids,
    apply_fc2_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_attn_scale_search_from_token_ids,
    apply_output_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_lm_head_scale_search_from_token_ids,
    apply_output_aware_mlp_scale_search_from_token_ids,
    apply_quarot_rotation_from_token_ids,
    build_calibration_scales_from_token_ids,
)
from taccel.runtime.fake_quant_reference import NanoGPTFQReference
from taccel.runtime.fp32_reference import NanoGPTFP32Reference
from taccel.runtime.gpt2_perplexity import tokenize_text_file
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)


def _build_pipeline_state(
    payload: dict,
    calibration_token_ids: List[int],
    preset_name: str,
    *,
    n_seqs: int,
    seq_len: int,
):
    """Run the same pipeline as evaluate_gpt2_perplexity up to (and including)
    all calibration/search steps. Mutates payload['state_dict'] in place,
    returns the final calibration_scales dict + resolved preset."""
    preset = resolve_stage5_ptq_preset(preset_name)

    if preset.quarot_enabled:
        apply_quarot_rotation_from_token_ids(
            payload, calibration_token_ids,
            seed=preset.quarot_seed, kind=preset.quarot_kind,
        )
    if preset.awq_enabled:
        apply_awq_from_token_ids(
            payload, calibration_token_ids,
            n_seqs=n_seqs, seq_len=seq_len,
            alpha=preset.awq_alpha, target_modules=preset.awq_target_modules,
        )
    cal = build_calibration_scales_from_token_ids(
        payload, calibration_token_ids,
        n_seqs=n_seqs, seq_len=seq_len, percentile=99.9,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    cal = apply_stage5_ptq_scale_policy(cal, payload["model_args"], preset)
    if preset.bias_correction_blocks:
        apply_bias_correction_from_token_ids(
            payload, calibration_token_ids, cal,
            blocks=preset.bias_correction_blocks,
            weight_types=preset.bias_correction_weight_types,
        )
        cal = build_calibration_scales_from_token_ids(
            payload, calibration_token_ids,
            n_seqs=n_seqs, seq_len=seq_len, percentile=99.9,
            activation_percentile_overrides=preset.activation_percentile_nodes or None,
            hessian_gelu_blocks=preset.hessian_gelu_blocks,
        )
        cal = apply_stage5_ptq_scale_policy(cal, payload["model_args"], preset)
    if preset.fc2_aware_gelu_blocks:
        cal, _ = apply_fc2_aware_gelu_scale_search_from_token_ids(
            payload, calibration_token_ids, cal,
            blocks=preset.fc2_aware_gelu_blocks,
            n_seqs=n_seqs, seq_len=seq_len,
        )
    if preset.output_aware_gelu_blocks:
        cal, _ = apply_output_aware_gelu_scale_search_from_token_ids(
            payload, calibration_token_ids, cal,
            blocks=preset.output_aware_gelu_blocks, ptq_preset=preset,
            n_seqs=n_seqs, seq_len=seq_len,
        )
    if preset.output_aware_mlp_blocks:
        cal, _ = apply_output_aware_mlp_scale_search_from_token_ids(
            payload, calibration_token_ids, cal,
            blocks=preset.output_aware_mlp_blocks, ptq_preset=preset,
            n_seqs=n_seqs, seq_len=seq_len,
            include_pair_candidates=preset.output_aware_include_pairs,
            passes=preset.output_aware_mlp_passes,
        )
    if preset.output_aware_attn_blocks:
        cal, _ = apply_output_aware_attn_scale_search_from_token_ids(
            payload, calibration_token_ids, cal,
            blocks=preset.output_aware_attn_blocks, ptq_preset=preset,
            n_seqs=n_seqs, seq_len=seq_len,
        )
    if preset.output_aware_lm_head:
        cal, _ = apply_output_aware_lm_head_scale_search_from_token_ids(
            payload, calibration_token_ids, cal,
            ptq_preset=preset, n_seqs=n_seqs, seq_len=seq_len,
        )

    return cal, preset


def _collect_per_node_error_stats(
    payload_fp32: dict,
    payload_fq: dict,
    cal_scales: dict,
    preset,
    eval_tokens: List[int],
    n_eval_steps: int,
):
    """Run both references step-by-step with traces, return per-node error stats."""
    ref_fp32 = NanoGPTFP32Reference(payload_fp32["state_dict"], payload_fp32["model_args"])
    ref_fq = NanoGPTFQReference(
        payload_fq["state_dict"], payload_fq["model_args"], cal_scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(payload_fq["model_args"], preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(preset),
    )

    caches_fp32 = ref_fp32._empty_caches()
    # Build fake-quant caches matching its internal structure
    caches_fq = [[{"k": [], "v": []} for _ in range(ref_fq.n_head)] for _ in range(ref_fq.n_layer)]

    n_eval_steps = min(int(n_eval_steps), len(eval_tokens))

    # Per-node accumulators
    sum_sq_err: Dict[str, float] = {}
    sum_sq_fp32: Dict[str, float] = {}
    max_abs_err: Dict[str, float] = {}
    fp32_max_abs: Dict[str, float] = {}
    count: Dict[str, int] = {}

    for step in range(n_eval_steps):
        token_id = int(eval_tokens[step])
        pos_id = step

        fp32_trace: Dict[str, dict] = {}
        fq_trace: Dict[str, dict] = {}

        ref_fp32._decode_incremental_step(token_id, pos_id, caches_fp32, trace=fp32_trace)
        ref_fq._decode_incremental_step(token_id, pos_id, caches_fq, trace=fq_trace)

        for name in fp32_trace:
            if name not in fq_trace:
                continue
            fp32_val = np.asarray(fp32_trace[name]["value"], dtype=np.float32).ravel()
            fq_val = np.asarray(fq_trace[name]["value"], dtype=np.float32).ravel()
            if fp32_val.shape != fq_val.shape:
                continue  # Skip incompatible shapes
            err = fq_val - fp32_val
            ss_err = float(np.sum(err * err))
            ss_fp32 = float(np.sum(fp32_val * fp32_val))
            mxerr = float(np.max(np.abs(err))) if err.size > 0 else 0.0
            mxfp = float(np.max(np.abs(fp32_val))) if fp32_val.size > 0 else 0.0

            sum_sq_err[name] = sum_sq_err.get(name, 0.0) + ss_err
            sum_sq_fp32[name] = sum_sq_fp32.get(name, 0.0) + ss_fp32
            max_abs_err[name] = max(max_abs_err.get(name, 0.0), mxerr)
            fp32_max_abs[name] = max(fp32_max_abs.get(name, 0.0), mxfp)
            count[name] = count.get(name, 0) + 1

    rows = []
    for name in sorted(sum_sq_err):
        rel_l2 = math.sqrt(sum_sq_err[name] / max(sum_sq_fp32[name], 1e-30))
        rel_max = max_abs_err[name] / max(fp32_max_abs[name], 1e-30)
        rows.append({
            "node": name,
            "rel_l2": rel_l2,
            "rel_max": rel_max,
            "l2_err": math.sqrt(sum_sq_err[name]),
            "fp32_l2": math.sqrt(sum_sq_fp32[name]),
            "max_abs_err": max_abs_err[name],
            "fp32_max_abs": fp32_max_abs[name],
            "steps": count[name],
        })
    return rows


def _print_table(rows: List[dict], top: int = 40):
    rows_sorted = sorted(rows, key=lambda r: r["rel_l2"], reverse=True)[:top]
    headers = ["node", "rel_l2", "rel_max", "l2_err", "fp32_l2", "max_err", "fp32_max"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows_sorted:
        print(
            f"| {r['node']:45s} | {r['rel_l2']:.4e} | {r['rel_max']:.4e} | "
            f"{r['l2_err']:.4e} | {r['fp32_l2']:.4e} | "
            f"{r['max_abs_err']:.4e} | {r['fp32_max_abs']:.4e} |"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--ptq-preset", default=None)
    parser.add_argument("--n-eval-steps", type=int, default=16)
    parser.add_argument("--calibration-n-seqs", type=int, default=16)
    parser.add_argument("--calibration-seq-len", type=int, default=64)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    preset_name = args.ptq_preset or stage5_default_ptq_preset_name()

    # Two payload copies: one stays pure FP32 (unmutated), one goes through pipeline.
    print(f"Loading checkpoint {args.checkpoint} ...")
    payload_fq = torch.load(args.checkpoint, map_location="cpu")
    payload_fp32 = {
        "state_dict": {k: v.clone() if hasattr(v, "clone") else copy.deepcopy(v)
                       for k, v in payload_fq["state_dict"].items()},
        "model_args": copy.deepcopy(payload_fq["model_args"]),
    }

    calib_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    eval_ids = tokenize_text_file(args.tokenizer_dir, args.eval_text, max_tokens=args.n_eval_steps + 1)

    print(f"Building pipeline state for preset={preset_name} ...")
    cal_scales, preset = _build_pipeline_state(
        payload_fq, calib_ids, preset_name,
        n_seqs=args.calibration_n_seqs, seq_len=args.calibration_seq_len,
    )

    print(f"Running {args.n_eval_steps} step(s) of paired FP32 + fake-quant trace ...")
    rows = _collect_per_node_error_stats(
        payload_fp32, payload_fq, cal_scales, preset, eval_ids, args.n_eval_steps
    )

    if args.json:
        out = {
            "preset": preset_name,
            "n_eval_steps": args.n_eval_steps,
            "rows": rows,
        }
        print(json.dumps(out, indent=2, sort_keys=False))
    else:
        print(f"\n## Per-node FP32 vs fake-quant error (preset = {preset_name})\n")
        _print_table(rows, top=args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
