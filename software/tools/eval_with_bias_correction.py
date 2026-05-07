#!/usr/bin/env python3
"""Apply analytical bias correction to GPT-2 quantized linear layers.

The existing taccel/quantizer/bias_correction.py implementation is hardcoded
for DeiT-tiny (ViT) module names. This script implements the same algorithm
for the converted GPT-2 layout.

Algorithm: for each target weight W (with bias b) and its input activation X,
compute the mean per-output-channel residual error introduced by quantization,
then fold that residual back into the bias:

    b_corrected[i] = b[i] + mean over samples of:
        X[s] @ W[i] - dequant(quantize(X[s])) @ dequant(quantize(W))[i]

The standard pipeline (calibration, scale search, fake-quant, golden) then
sees a state_dict with the corrected biases and naturally reproduces the
mean-shift compensation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from taccel.quantizer.quantize import quantize_tensor
from taccel.runtime.calibration import (
    build_calibration_scales_from_token_ids,
    build_calibration_seqs_from_token_ids,
)
from taccel.runtime.fake_quant_reference import _fp32_forward, _to_f32
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    evaluate_gpt2_perplexity,
    file_sha256,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
)


def _weight_to_input_node(weight_name: str) -> str:
    """Map a state_dict weight name to the activation node feeding the layer."""
    if weight_name.endswith(".mlp.c_fc.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_ln2"
    if weight_name.endswith(".mlp.c_proj.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_gelu"
    if weight_name.endswith(".attn.c_proj.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_concat"
    raise ValueError(f"Unsupported weight name: {weight_name}")


def _input_act_scale_key(weight_name: str) -> str:
    return _weight_to_input_node(weight_name)


def _capture_activations(state_dict, model_args, calib_seqs, target_node_names):
    accum: dict[str, list[np.ndarray]] = {n: [] for n in target_node_names}
    for tids in calib_seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for n in target_node_names:
            arr = np.asarray(node_outputs[n], dtype=np.float32)
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            accum[n].append(arr)
    return {n: np.concatenate(rows, axis=0) for n, rows in accum.items()}


def _expand_target_weights(target_blocks: list[int], weight_types: list[str]) -> list[str]:
    out: list[str] = []
    for L in target_blocks:
        for wt in weight_types:
            out.append(f"transformer.h.{L}.{wt}.weight")
    return out


def apply_bias_correction(
    state_dict: dict,
    model_args: dict,
    calibration_scales: dict[str, float],
    calibration_seqs,
    target_weights: list[str],
) -> list[dict]:
    """Modify state_dict biases in-place to compensate for quantization mean shift."""
    target_nodes = sorted({_weight_to_input_node(w) for w in target_weights})
    print(f"# Capturing FP32 activations for nodes: {target_nodes}")
    activations = _capture_activations(state_dict, model_args, calibration_seqs, target_nodes)

    reports: list[dict] = []
    for w in target_weights:
        bias_name = w.replace(".weight", ".bias")
        if bias_name not in state_dict:
            print(f"# WARNING: no bias for {w}, skipping")
            continue

        node = _weight_to_input_node(w)
        x = activations[node]
        W = _to_f32(state_dict[w])  # [out, in]
        b = _to_f32(state_dict[bias_name])  # [out]

        # Per-channel symmetric INT8 weight quantization (matches the path used
        # by the rest of the pipeline).
        q_w, scales_w = quantize_tensor(W, per_channel=True)
        scales_w = scales_w.astype(np.float32)
        W_dq = q_w.astype(np.float32) * scales_w.reshape(-1, 1)

        # Per-tensor input activation quantization with the calibration scale.
        x_scale = max(float(calibration_scales.get(_input_act_scale_key(w), 6.0 / 127.0)), 1e-12)
        x_q = np.clip(np.round(x / x_scale), -128, 127).astype(np.int8)
        x_dq = x_q.astype(np.float32) * np.float32(x_scale)

        y_fp32 = x @ W.T  # [N, out]
        y_qdq = x_dq @ W_dq.T  # [N, out]
        err_per_channel = np.mean(y_fp32 - y_qdq, axis=0).astype(np.float32)  # [out]

        b_corrected = b + err_per_channel
        state_dict[bias_name] = torch.from_numpy(np.ascontiguousarray(b_corrected.astype(np.float32))).to(
            dtype=state_dict[bias_name].dtype
        )

        reports.append({
            "weight": w,
            "bias": bias_name,
            "input_node": node,
            "n_samples": int(x.shape[0]),
            "input_act_scale": float(x_scale),
            "err_abs_max": float(np.max(np.abs(err_per_channel))),
            "err_abs_mean": float(np.mean(np.abs(err_per_channel))),
            "err_rms": float(np.sqrt(np.mean(err_per_channel ** 2))),
            "bias_abs_mean_before": float(np.mean(np.abs(b))),
            "bias_abs_mean_after": float(np.mean(np.abs(b_corrected))),
        })

    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, required=True)
    parser.add_argument("--max-eval-tokens", type=int, default=33)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--calibration-n-seqs", type=int, default=CALIBRATION_N_SEQS_LARGE)
    parser.add_argument("--calibration-seq-len", type=int, default=CALIBRATION_SEQ_LEN_LARGE)
    parser.add_argument("--calibration-percentile", type=float, default=CALIBRATION_PERCENTILE_DEFAULT)
    parser.add_argument("--ptq-preset", default=None)
    parser.add_argument("--target-blocks", default="0,1,2,3,4,5,6,7,8,9,10,11",
                        help="Comma-separated block indices")
    parser.add_argument("--weight-types", default="mlp.c_fc,mlp.c_proj,attn.c_proj",
                        help="Comma-separated linear suffixes")
    parser.add_argument("--bc-search-n-seqs", type=int, default=8)
    parser.add_argument("--bc-search-seq-len", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    target_blocks = [int(x.strip()) for x in args.target_blocks.split(",") if x.strip()]
    weight_types = [x.strip() for x in args.weight_types.split(",") if x.strip()]
    target_weights = _expand_target_weights(target_blocks, weight_types)
    print(f"# Bias correction targets: {len(target_weights)} weights")

    calib_token_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)

    # First, compute the activation calibration scales (so bias correction knows
    # how activations get quantized). Use the chosen preset's pipeline.
    preset_name = args.ptq_preset or stage5_default_ptq_preset_name()
    preset = resolve_stage5_ptq_preset(preset_name)
    print(f"# Using preset for activation scale baseline: {preset.name}")
    base_scales = build_calibration_scales_from_token_ids(
        payload,
        calib_token_ids,
        n_seqs=args.calibration_n_seqs,
        seq_len=args.calibration_seq_len,
        percentile=args.calibration_percentile,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    base_scales = apply_stage5_ptq_scale_policy(base_scales, payload["model_args"], preset)

    # Use a smaller set of sequences for the bias-correction pass itself.
    bc_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(args.bc_search_n_seqs),
        seq_len=int(args.bc_search_seq_len),
    )

    reports = apply_bias_correction(
        payload["state_dict"],
        payload["model_args"],
        base_scales,
        bc_seqs,
        target_weights,
    )
    print(f"# Applied bias correction to {len(reports)} layers")
    err_max_overall = max((r["err_abs_max"] for r in reports), default=0.0)
    err_mean_overall = (
        sum(r["err_abs_mean"] for r in reports) / max(len(reports), 1)
    )
    print(f"# err_abs_max overall = {err_max_overall:.4e}")
    print(f"# err_abs_mean overall = {err_mean_overall:.4e}")

    eval_ids = tokenize_text_file(args.tokenizer_dir, args.eval_text, max_tokens=args.max_eval_tokens)
    result = evaluate_gpt2_perplexity(
        payload,
        calibration_token_ids=calib_token_ids,
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
    )
    out = asdict(result)
    out["bias_correction"] = {"reports": reports}
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"\ngolden_perplexity: {result.golden_perplexity:.6f}")
        print(f"fake_quant_perplexity: {result.fake_quant_perplexity:.6f}")
        print(f"relative_delta: {result.relative_delta:.6%}")
        print(f"target_count: {result.target_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
