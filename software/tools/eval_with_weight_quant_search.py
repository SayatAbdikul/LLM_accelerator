#!/usr/bin/env python3
"""Evaluate perplexity after applying clip-search + AdaRound to selected weights.

Wires the existing `quantize_tensor_clipped` and `adaround_greedy` infrastructure
(in `taccel/quantizer/quantize.py`) into the GPT-2 path, which currently uses
plain max-abs round-to-nearest.

Approach:
1. Run FP32 forward on calibration tokens, capturing per-layer linear inputs.
2. For each target weight, optionally apply clip-search then AdaRound.
3. Replace the FP32 weight in the state_dict with `refined_q * scale` so that
   the rest of the pipeline (calibration scales, fake-quant ref, codegen) sees
   weights that round-trip back to the AdaRound INT8 values automatically.
4. Run the standard evaluate_gpt2_perplexity flow.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from taccel.quantizer.quantize import (
    adaround_greedy,
    quantize_tensor,
    quantize_tensor_clipped,
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
from taccel.runtime.calibration import build_calibration_seqs_from_token_ids


# Weight name → activation node name (input to the linear layer).
def _weight_to_input_node(weight_name: str) -> str | None:
    if weight_name.endswith(".mlp.c_fc.weight"):
        # transformer.h.{L}.mlp.c_fc.weight
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_ln2"
    if weight_name.endswith(".mlp.c_proj.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_gelu"
    if weight_name.endswith(".attn.c_proj.weight"):
        L = int(weight_name.split(".h.")[1].split(".")[0])
        return f"block{L}_concat"
    return None


def _expand_target_weights(target_blocks: list[int], weight_types: list[str]) -> list[str]:
    out: list[str] = []
    for L in target_blocks:
        for wt in weight_types:
            out.append(f"transformer.h.{L}.{wt}.weight")
    return out


def _capture_activations(state_dict, model_args, calib_seqs, target_node_names):
    """Run FP32 forward on each calibration sequence and concatenate the
    requested activations across sequences. Returns dict node_name → ndarray of
    shape [N_total_tokens, d_input]."""
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


def apply_weight_quant_search(
    state_dict: dict,
    model_args: dict,
    calibration_seqs,
    target_weights: list[str],
    *,
    use_clip_search: bool,
    use_adaround: bool,
    clip_n_candidates: int = 25,
    clip_alpha_min: float = 0.5,
    adaround_max_accepts: int | None = None,
):
    """Modify state_dict in-place: replace each target weight with W' such that
    quantize_tensor(W') reproduces the (clip-searched + AdaRounded) INT8 result.
    """
    if not (use_clip_search or use_adaround):
        return  # nothing to do

    # Resolve activation node names per target weight.
    weight_to_node = {}
    for w in target_weights:
        node = _weight_to_input_node(w)
        if node is None:
            raise ValueError(f"unsupported weight name for activation lookup: {w}")
        weight_to_node[w] = node

    target_nodes = sorted(set(weight_to_node.values()))
    print(f"# Capturing FP32 activations for nodes: {target_nodes}")
    activations = _capture_activations(state_dict, model_args, calibration_seqs, target_nodes)
    for n, arr in activations.items():
        print(f"#   {n}: shape={arr.shape}")

    for w in target_weights:
        node = weight_to_node[w]
        calib_input = activations[node]
        W = _to_f32(state_dict[w])  # [out, in]
        if W.ndim != 2:
            raise ValueError(f"expected 2D weight for {w}, got shape {W.shape}")

        # Step 1: clip search (or plain quantization to seed AdaRound).
        if use_clip_search:
            q, scales = quantize_tensor_clipped(
                W, calibration_inputs=[calib_input],
                per_channel=True, n_candidates=clip_n_candidates, alpha_min=clip_alpha_min,
            )
        else:
            q, scales = quantize_tensor(W, per_channel=True)

        # Step 2: AdaRound refinement.
        if use_adaround:
            q = adaround_greedy(
                W, q, scales.astype(np.float32),
                calibration_inputs=[calib_input],
                max_accepts_per_channel=adaround_max_accepts,
            )

        # Step 3: rebuild FP32 weight that round-trips to the same INT8 values.
        scales_f32 = scales.astype(np.float32)
        W_modified = q.astype(np.float32) * scales_f32.reshape(-1, 1)

        # Sanity check: re-quantizing should give the same q (within epsilon).
        q_check, scales_check = quantize_tensor(W_modified.astype(np.float32), per_channel=True)
        if not np.array_equal(q_check, q):
            mismatch = np.sum(q_check != q)
            print(f"# WARNING: re-quantization mismatch for {w}: {mismatch} entries differ")

        # Store back as torch tensor to preserve any downstream ".detach().cpu().numpy()" calls.
        state_dict[w] = torch.from_numpy(W_modified.astype(np.float32))
        delta_int = int(np.sum(q != quantize_tensor(W, per_channel=True)[0]))
        print(f"# {w}: AdaRound flipped {delta_int} weights, "
              f"max_scale={float(scales_f32.max()):.4e}, min_scale={float(scales_f32.min()):.4e}")


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
    parser.add_argument("--target-blocks", default="2",
                        help="Comma-separated block indices, e.g., '2' or '2,11'")
    parser.add_argument("--weight-types", default="mlp.c_fc,mlp.c_proj",
                        help="Comma-separated linear layer suffixes, e.g., "
                             "'mlp.c_fc,mlp.c_proj,attn.c_proj'")
    parser.add_argument("--clip-search", action="store_true", default=True,
                        help="Apply MSE-clipped quantization (default True)")
    parser.add_argument("--no-clip-search", dest="clip_search", action="store_false")
    parser.add_argument("--adaround", action="store_true", default=True,
                        help="Apply AdaRound greedy rounding (default True)")
    parser.add_argument("--no-adaround", dest="adaround", action="store_false")
    parser.add_argument("--clip-n-candidates", type=int, default=25)
    parser.add_argument("--clip-alpha-min", type=float, default=0.5)
    parser.add_argument("--adaround-max-accepts", type=int, default=None)
    parser.add_argument("--weight-search-n-seqs", type=int, default=8,
                        help="Number of calibration sequences to use for weight quant search.")
    parser.add_argument("--weight-search-seq-len", type=int, default=64,
                        help="Sequence length for weight quant search calibration.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    target_blocks = [int(x.strip()) for x in args.target_blocks.split(",") if x.strip()]
    weight_types = [x.strip() for x in args.weight_types.split(",") if x.strip()]
    target_weights = _expand_target_weights(target_blocks, weight_types)
    print(f"# Target weights: {target_weights}")

    calib_token_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)

    # Capture activations using a small calibration set (separate from main
    # PTQ calibration, kept small to keep the search tractable).
    calib_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(args.weight_search_n_seqs),
        seq_len=int(args.weight_search_seq_len),
    )

    apply_weight_quant_search(
        payload["state_dict"],
        payload["model_args"],
        calib_seqs,
        target_weights,
        use_clip_search=args.clip_search,
        use_adaround=args.adaround,
        clip_n_candidates=args.clip_n_candidates,
        clip_alpha_min=args.clip_alpha_min,
        adaround_max_accepts=args.adaround_max_accepts,
    )

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
    out["weight_quant_search"] = {
        "target_weights": target_weights,
        "clip_search": bool(args.clip_search),
        "adaround": bool(args.adaround),
    }
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
