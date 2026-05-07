#!/usr/bin/env python3
"""Apply SmoothQuant to specified GPT-2 LN→Linear pairs and evaluate perplexity.

The existing taccel/quantizer/smooth_quant.py implementation is hardcoded for
DeiT-tiny (ViT) module names. This script implements the same algorithm for the
converted GPT-2 layout (transformer.h.{L}.ln_2 → transformer.h.{L}.mlp.c_fc),
applies it to the state_dict in-place, then runs the standard evaluation.

The smooth factor per channel is:
    s_i = act_max_i^alpha / weight_max_i^(1 - alpha)
which is divided into LN gamma+beta and multiplied into Linear weight columns.

Targets supported (per block):
- ln_2_fc1: ln_2 weight/bias and mlp.c_fc.weight (input dim 768)
- ln_1_qkv: ln_1 weight/bias and per-head q/k/v weights (input dim 768)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from taccel.runtime.fake_quant_reference import _fp32_forward, _to_f32
from taccel.runtime.calibration import build_calibration_seqs_from_token_ids
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    evaluate_gpt2_perplexity,
    file_sha256,
    tokenize_text_file,
)


def _channel_max(activations: list[np.ndarray]) -> np.ndarray:
    """Per-channel max(|x|) over a list of [N, C] arrays."""
    best = None
    for arr in activations:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim > 2:
            a = a.reshape(-1, a.shape[-1])
        elif a.ndim == 1:
            a = a.reshape(1, -1)
        m = np.max(np.abs(a), axis=0)
        best = m if best is None else np.maximum(best, m)
    if best is None:
        raise ValueError("no activations to compute channel max")
    return best


def _capture_node_activations(state_dict, model_args, calib_seqs, node_name):
    out: list[np.ndarray] = []
    for tids in calib_seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        out.append(np.asarray(node_outputs[node_name], dtype=np.float32))
    return out


def _to_torch(arr, like):
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(dtype=like.dtype)


def apply_smoothquant_ln2_fc1(state_dict, calib_acts: list[np.ndarray], block: int, alpha: float) -> dict:
    """Apply SmoothQuant to the ln_2 → mlp.c_fc pair for a single block.

    Modifies state_dict in-place. Returns a small report dict.
    """
    eps = 1e-8
    act_max = np.maximum(_channel_max(calib_acts), eps)  # [d_model]

    fc_weight_name = f"transformer.h.{block}.mlp.c_fc.weight"
    ln_w_name = f"transformer.h.{block}.ln_2.weight"
    ln_b_name = f"transformer.h.{block}.ln_2.bias"

    fc_w = _to_f32(state_dict[fc_weight_name])  # [3072, 768] (out, in)
    if fc_w.ndim != 2:
        raise ValueError(f"unexpected shape for {fc_weight_name}: {fc_w.shape}")
    weight_max = np.maximum(np.max(np.abs(fc_w), axis=0), eps)  # [768]
    if weight_max.shape[0] != act_max.shape[0]:
        raise ValueError(f"channel mismatch: weight {weight_max.shape}, act {act_max.shape}")

    smooth = np.maximum(np.power(act_max, alpha) / np.power(weight_max, 1.0 - alpha), eps)  # [768]

    ln_w = _to_f32(state_dict[ln_w_name])
    ln_b = _to_f32(state_dict[ln_b_name])
    state_dict[ln_w_name] = _to_torch(ln_w / smooth, state_dict[ln_w_name])
    state_dict[ln_b_name] = _to_torch(ln_b / smooth, state_dict[ln_b_name])

    fc_w_modified = fc_w * smooth.reshape(1, -1)  # broadcast over output dim
    state_dict[fc_weight_name] = _to_torch(fc_w_modified, state_dict[fc_weight_name])

    return {
        "block": int(block),
        "target": "ln_2_fc1",
        "alpha": float(alpha),
        "act_max_min": float(np.min(act_max)),
        "act_max_max": float(np.max(act_max)),
        "weight_max_min": float(np.min(weight_max)),
        "weight_max_max": float(np.max(weight_max)),
        "smooth_min": float(np.min(smooth)),
        "smooth_max": float(np.max(smooth)),
        "smooth_mean": float(np.mean(smooth)),
    }


def apply_smoothquant_ln1_qkv(state_dict, model_args, calib_acts: list[np.ndarray], block: int, alpha: float) -> dict:
    """Apply SmoothQuant to the ln_1 → q/k/v pair for a single block.

    Q/K/V are stored per-head as transformer.h.{L}.attn.c_attn.weight_h{H}_{query,key,value}
    each with shape [d_head, d_model]. SmoothQuant scales the input dim (d_model),
    so we update each per-head weight column-wise.
    """
    eps = 1e-8
    act_max = np.maximum(_channel_max(calib_acts), eps)  # [d_model]

    n_head = int(model_args["n_head"])
    ln_w_name = f"transformer.h.{block}.ln_1.weight"
    ln_b_name = f"transformer.h.{block}.ln_1.bias"

    # Combined weight max across all heads & projections (per input channel).
    weight_max = None
    for H in range(n_head):
        for proj in ("query", "key", "value"):
            w_name = f"transformer.h.{block}.attn.c_attn.weight_h{H}_{proj}"
            w = _to_f32(state_dict[w_name])  # [d_head, d_model]
            cur = np.max(np.abs(w), axis=0)
            weight_max = cur if weight_max is None else np.maximum(weight_max, cur)
    weight_max = np.maximum(weight_max, eps)  # [d_model]

    if weight_max.shape[0] != act_max.shape[0]:
        raise ValueError(f"channel mismatch: weight {weight_max.shape}, act {act_max.shape}")

    smooth = np.maximum(np.power(act_max, alpha) / np.power(weight_max, 1.0 - alpha), eps)

    ln_w = _to_f32(state_dict[ln_w_name])
    ln_b = _to_f32(state_dict[ln_b_name])
    state_dict[ln_w_name] = _to_torch(ln_w / smooth, state_dict[ln_w_name])
    state_dict[ln_b_name] = _to_torch(ln_b / smooth, state_dict[ln_b_name])

    for H in range(n_head):
        for proj in ("query", "key", "value"):
            w_name = f"transformer.h.{block}.attn.c_attn.weight_h{H}_{proj}"
            w = _to_f32(state_dict[w_name])
            state_dict[w_name] = _to_torch(w * smooth.reshape(1, -1), state_dict[w_name])

    return {
        "block": int(block),
        "target": "ln_1_qkv",
        "alpha": float(alpha),
        "act_max_min": float(np.min(act_max)),
        "act_max_max": float(np.max(act_max)),
        "weight_max_min": float(np.min(weight_max)),
        "weight_max_max": float(np.max(weight_max)),
        "smooth_min": float(np.min(smooth)),
        "smooth_max": float(np.max(smooth)),
        "smooth_mean": float(np.mean(smooth)),
    }


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
    parser.add_argument("--target-blocks", default="2", help="Comma-separated block indices, e.g., '2' or '2,11'")
    parser.add_argument("--targets", default="ln_2_fc1",
                        help="Comma-separated targets per block: ln_2_fc1, ln_1_qkv. Use 'ln_2_fc1,ln_1_qkv' for both.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength (0..1). 0.5 is the standard value.")
    parser.add_argument("--smooth-search-n-seqs", type=int, default=8)
    parser.add_argument("--smooth-search-seq-len", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    target_blocks = [int(x.strip()) for x in args.target_blocks.split(",") if x.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in {"ln_2_fc1", "ln_1_qkv"}:
            raise ValueError(f"unknown SmoothQuant target: {t!r}")

    calib_token_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    calib_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(args.smooth_search_n_seqs),
        seq_len=int(args.smooth_search_seq_len),
    )

    reports = []
    for block in target_blocks:
        for target in targets:
            if target == "ln_2_fc1":
                node_name = f"block{block}_ln2"
                acts = _capture_node_activations(payload["state_dict"], payload["model_args"], calib_seqs, node_name)
                reports.append(apply_smoothquant_ln2_fc1(payload["state_dict"], acts, block, float(args.alpha)))
            elif target == "ln_1_qkv":
                node_name = f"block{block}_ln1"
                acts = _capture_node_activations(payload["state_dict"], payload["model_args"], calib_seqs, node_name)
                reports.append(apply_smoothquant_ln1_qkv(
                    payload["state_dict"], payload["model_args"], acts, block, float(args.alpha)
                ))

    print(f"# Applied SmoothQuant to {len(reports)} pairs:")
    for r in reports:
        print(f"#   block{r['block']}/{r['target']}: alpha={r['alpha']}  "
              f"smooth=[{r['smooth_min']:.3e}, {r['smooth_max']:.3e}] mean={r['smooth_mean']:.3e}")

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
    out["smooth_quant"] = {"reports": reports}
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
