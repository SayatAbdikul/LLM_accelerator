#!/usr/bin/env python3
"""Evaluate perplexity after applying GPTQ to selected weights.

Wires the new ``gptq_quantize`` (taccel/quantizer/quantize.py) into the GPT-2
path. Same plumbing as ``eval_with_weight_quant_search.py`` — captures FP32
activations on a small calibration set, refines each target weight, then
hands the refined state_dict to the standard ``evaluate_gpt2_perplexity``
flow so the Stage 5 calibration / fake-quant / golden simulator path sees
the GPTQ-rounded weights as if they were FP32 inputs.

Approach (per target weight W with input X):
1. Compute the layer-wise Hessian H = (2/N) X^T X with diagonal damping.
2. Cholesky-factor H_inv into upper-triangular U (U^T U = H_inv).
3. Quantize columns of W left-to-right, propagating each column's rounding
   error into the unquantized columns via U.
4. Replace state_dict[w] with W' = q_int * scale so the downstream
   ``quantize_tensor`` re-derives the same INT8 grid (same scale convention
   as the AdaRound tool — see WARNING handling below).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from taccel.quantizer.quantize import gptq_quantize, quantize_tensor
from taccel.runtime.fake_quant_reference import (
    NanoGPTFQReference,
    _fp32_forward,
    _to_f32,
)
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_PERCENTILE_DEFAULT,
    CALIBRATION_SEQ_LEN_LARGE,
    evaluate_gpt2_perplexity,
    file_sha256,
    tokenize_text_file,
)
from taccel.runtime.calibration import (
    build_calibration_scales_from_token_ids,
    build_calibration_seqs_from_token_ids,
)
from taccel.runtime.stage5_ptq import (
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)


# Same mapping as eval_with_weight_quant_search: weight name → activation
# node name produced by the FP32 reference's instrumented forward pass.
def _weight_to_input_node(weight_name: str) -> str | None:
    if weight_name.endswith(".mlp.c_fc.weight"):
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
    requested activations across sequences."""
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


def _capture_activations_fakequant(
    payload,
    ptq_preset,
    calibration_token_ids,
    calib_seqs,
    target_node_names,
    *,
    calibration_seq_len: int,
    calibration_n_seqs: int,
    calibration_percentile: float,
):
    """D3 path: capture activations after running each calib seq through the
    NanoGPTFQReference (the same fake-quant pipeline ``evaluate_gpt2_perplexity``
    uses), so the GPTQ Hessian is built on the distribution actually presented
    to each matmul during eval — not the FP32 distribution.

    The reference's ``forward`` was extended with a ``capture`` dict that
    receives post-qdq FP32 views of the matmul inputs (block{L}_ln1,
    block{L}_concat, block{L}_ln2, block{L}_gelu) under the same node names
    ``_fp32_forward`` uses.
    """
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    resolved_preset = resolve_stage5_ptq_preset(
        stage5_default_ptq_preset_name() if ptq_preset is None else ptq_preset
    )
    calibration_scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_token_ids,
        n_seqs=int(calibration_n_seqs),
        seq_len=int(calibration_seq_len),
        percentile=float(calibration_percentile),
        activation_percentile_overrides=(
            resolved_preset.activation_percentile_nodes or None
        ),
        hessian_gelu_blocks=resolved_preset.hessian_gelu_blocks,
    )
    ref = NanoGPTFQReference(
        state_dict,
        model_args,
        calibration_scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(model_args, resolved_preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(resolved_preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(resolved_preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(resolved_preset),
    )
    accum: dict[str, list[np.ndarray]] = {n: [] for n in target_node_names}
    for tids in calib_seqs:
        capture: dict[str, np.ndarray] = {}
        ref.forward(list(tids), return_all_logits=True, capture=capture)
        for n in target_node_names:
            if n not in capture:
                raise ValueError(
                    f"NanoGPTFQReference did not capture node {n!r}; only "
                    f"{sorted(capture)} are exposed"
                )
            arr = np.asarray(capture[n], dtype=np.float32)
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            accum[n].append(arr)
    return {n: np.concatenate(rows, axis=0) for n, rows in accum.items()}


def _block_index_of_weight(weight_name: str) -> int | None:
    """Return the transformer block index encoded in weight_name, or None."""
    if ".h." not in weight_name:
        return None
    try:
        return int(weight_name.split(".h.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def apply_gptq(
    state_dict: dict,
    model_args: dict,
    calibration_seqs,
    target_weights: list[str],
    *,
    percdamp: float = 0.01,
    blocksize: int = 128,
    use_fakequant_activations: bool = False,
    payload=None,
    calibration_token_ids=None,
    ptq_preset=None,
    calibration_seq_len: int = CALIBRATION_SEQ_LEN_LARGE,
    calibration_n_seqs: int = CALIBRATION_N_SEQS_LARGE,
    calibration_percentile: float = CALIBRATION_PERCENTILE_DEFAULT,
    sequential: bool = False,
):
    """Modify state_dict in-place: replace each target weight with W' such
    that ``quantize_tensor(W')`` reproduces the GPTQ-refined INT8 result.

    When ``sequential`` is True, weights are processed block-by-block in
    ascending block index, with activation re-capture between blocks so each
    block's GPTQ Hessian sees the FP32-forward distribution induced by *prior
    blocks' GPTQ-modified weights*. This matches the original Frantar et al.
    GPTQ paper's algorithm and prevents cross-layer error coupling that
    catastrophically regresses naive parallel GPTQ.
    """
    weight_to_node: dict[str, str] = {}
    for w in target_weights:
        node = _weight_to_input_node(w)
        if node is None:
            raise ValueError(f"unsupported weight name for activation lookup: {w}")
        weight_to_node[w] = node

    rtn_only = not calibration_seqs
    if rtn_only:
        print("# D6 path: --weight-search-n-seqs 0 → RTN fallback through GPTQ plumbing.")

    # ---- Sequential mode: group by block, recapture between blocks. ----
    if sequential and not rtn_only:
        # Sort target weights by block index so propagation flows in the
        # forward direction: weights without a block index (e.g. lm_head)
        # are processed last after every transformer block.
        keyed = []
        for w in target_weights:
            blk = _block_index_of_weight(w)
            keyed.append((blk if blk is not None else 10_000, w))
        keyed.sort()
        ordered_weights = [w for _, w in keyed]

        # Group consecutive weights that share a block index so a single
        # activation capture serves all of them (e.g. mlp.c_fc and
        # mlp.c_proj on block 5).
        groups: list[list[str]] = []
        prev_blk = object()
        for w in ordered_weights:
            blk = _block_index_of_weight(w)
            if blk != prev_blk:
                groups.append([])
                prev_blk = blk
            groups[-1].append(w)

        print(f"# Sequential GPTQ: processing {len(groups)} block groups")
        for gi, group in enumerate(groups):
            blk_label = _block_index_of_weight(group[0])
            print(f"# --- group {gi+1}/{len(groups)} (block {blk_label}): {group} ---", flush=True)
            group_nodes = sorted({weight_to_node[w] for w in group})
            if use_fakequant_activations:
                if payload is None or calibration_token_ids is None:
                    raise ValueError("--use-fakequant-activations requires payload + calibration_token_ids")
                activations = _capture_activations_fakequant(
                    payload,
                    ptq_preset,
                    calibration_token_ids,
                    calibration_seqs,
                    group_nodes,
                    calibration_seq_len=calibration_seq_len,
                    calibration_n_seqs=calibration_n_seqs,
                    calibration_percentile=calibration_percentile,
                )
            else:
                activations = _capture_activations(
                    state_dict, model_args, calibration_seqs, group_nodes
                )
            for n, arr in activations.items():
                print(f"#   {n}: shape={arr.shape}", flush=True)
            _apply_gptq_to_weights(
                state_dict,
                group,
                weight_to_node,
                activations,
                percdamp=percdamp,
                blocksize=blocksize,
            )
        return

    # ---- Parallel (non-sequential) mode — single activation capture. ----
    target_nodes = sorted(set(weight_to_node.values()))
    if rtn_only:
        activations = {n: None for n in target_nodes}
    elif use_fakequant_activations:
        if payload is None or calibration_token_ids is None:
            raise ValueError("--use-fakequant-activations requires payload + calibration_token_ids")
        print(f"# Capturing FAKE-QUANT activations for nodes: {target_nodes}")
        activations = _capture_activations_fakequant(
            payload,
            ptq_preset,
            calibration_token_ids,
            calibration_seqs,
            target_nodes,
            calibration_seq_len=calibration_seq_len,
            calibration_n_seqs=calibration_n_seqs,
            calibration_percentile=calibration_percentile,
        )
    else:
        print(f"# Capturing FP32 activations for nodes: {target_nodes}")
        activations = _capture_activations(state_dict, model_args, calibration_seqs, target_nodes)
    if not rtn_only:
        for n, arr in activations.items():
            print(f"#   {n}: shape={arr.shape}", flush=True)

    _apply_gptq_to_weights(
        state_dict,
        target_weights,
        weight_to_node,
        activations,
        percdamp=percdamp,
        blocksize=blocksize,
    )


def _apply_gptq_to_weights(
    state_dict: dict,
    target_weights: list[str],
    weight_to_node: dict[str, str],
    activations: dict[str, np.ndarray | None],
    *,
    percdamp: float,
    blocksize: int,
) -> None:
    """Run GPTQ on each weight in ``target_weights`` using ``activations``,
    install the rebuilt FP32 weight back into ``state_dict``, and print
    diagnostics. Shared by sequential and parallel paths.
    """
    for w in target_weights:
        node = weight_to_node[w]
        calib_input = activations[node]
        W = _to_f32(state_dict[w])
        if W.ndim != 2:
            raise ValueError(f"expected 2D weight for {w}, got shape {W.shape}")

        gptq_calib_inputs = None if calib_input is None else [calib_input]
        q, scales = gptq_quantize(
            W,
            gptq_calib_inputs,
            per_channel=True,
            percdamp=percdamp,
            blocksize=blocksize,
        )

        # Diff vs plain RTN, for the progress log.
        q_rtn, scales_rtn = quantize_tensor(W, per_channel=True)
        delta_int = int(np.sum(q != q_rtn))

        scales_f32 = scales.astype(np.float32).reshape(-1, 1)

        # ---- D1: Round-trip exactness with row-rescale fallback ----
        # The naive `W' = q * scale` weight drifts under `quantize_tensor`
        # whenever no entry on row ch sits at +/-127: max(|W'[ch]|) falls
        # below 127 * scale and the recomputed scale shrinks. The fix below
        # is a per-row uniform rescale that pins max(|W'[ch]|) to exactly
        # 127 * scale[ch], keeping the recomputed scale equal to the GPTQ
        # scale. This does NOT make q reproduce exactly when GPTQ kept the
        # row's max-abs strictly below 127 (rounding the smaller-magnitude
        # columns up by a fractional LSB). For those rows, the dequantized
        # weight q' * scale' ≈ q * scale to fp32 precision, so the matmul
        # output is preserved even when individual integer codes shift.
        W_modified = q.astype(np.float32) * scales_f32
        row_max_abs_int = np.max(np.abs(q.astype(np.float32)), axis=1)
        # If a row is all-zero, leave it alone — quantize_tensor's 1e-8 floor
        # recovers identity.
        nonzero_rows = row_max_abs_int > 0
        alpha = np.ones(q.shape[0], dtype=np.float32)
        alpha[nonzero_rows] = 127.0 / np.maximum(row_max_abs_int[nonzero_rows], 1.0)
        W_modified = (W_modified * alpha.reshape(-1, 1)).astype(np.float32)

        # Diagnostic: count rows that *can't* round-trip exactly (r < 127).
        rows_below_127 = int(np.sum(row_max_abs_int < 127))
        rows_total = int(q.shape[0])

        # Re-quantize and report the mismatch count and worst dq drift.
        q_check, scales_check = quantize_tensor(W_modified, per_channel=True)
        n_mismatch = int(np.sum(q_check != q))
        scales_check_f32 = scales_check.astype(np.float32).reshape(-1, 1)
        dq_intended = q.astype(np.float32) * scales_f32
        dq_recovered = q_check.astype(np.float32) * scales_check_f32
        max_dq_drift = float(np.max(np.abs(dq_intended - dq_recovered)))
        mean_dq_drift = float(np.mean(np.abs(dq_intended - dq_recovered)))

        # ---- D2: Layer-MSE telemetry ----
        # Compare GPTQ vs RTN at three levels:
        #   1. Weight Frobenius MSE: how far each method moves W in fp32.
        #   2. Output Frobenius MSE on the calibration distribution: the
        #      quantity GPTQ explicitly minimizes.
        #   3. Per-channel max-abs of the rounding residual: how big a single
        #      LSB error is in fp32 units, useful for spotting outlier rows.
        scales_rtn_f32 = scales_rtn.astype(np.float32).reshape(-1, 1)
        W_rtn_dq = q_rtn.astype(np.float32) * scales_rtn_f32
        W_gptq_dq = q.astype(np.float32) * scales_f32  # pre-row-rescale dq
        weight_mse_rtn = float(np.mean((W_rtn_dq - W) ** 2))
        weight_mse_gptq = float(np.mean((W_gptq_dq - W) ** 2))
        # Output MSE: ||(W - W_q) X^T||_F^2 / N, where X = calib_input.
        diff_rtn = W_rtn_dq - W
        diff_gptq = W_gptq_dq - W
        per_ch_resid_rtn = float(np.max(np.abs(diff_rtn)))
        per_ch_resid_gptq = float(np.max(np.abs(diff_gptq)))
        if calib_input is None:
            out_mse_rtn = float("nan")
            out_mse_gptq = float("nan")
            N_calib = 0
        else:
            X = np.asarray(calib_input, dtype=np.float32)
            if X.ndim > 2:
                X = X.reshape(-1, X.shape[-1])
            elif X.ndim == 1:
                X = X.reshape(1, -1)
            N_calib = max(int(X.shape[0]), 1)
            out_err_rtn = diff_rtn @ X.T
            out_err_gptq = diff_gptq @ X.T
            out_mse_rtn = float(np.mean(out_err_rtn ** 2))
            out_mse_gptq = float(np.mean(out_err_gptq ** 2))

        state_dict[w] = torch.from_numpy(W_modified)
        print(
            f"# {w}: GPTQ flipped {delta_int} weights vs RTN "
            f"(max_scale={float(scales_f32.max()):.4e}, "
            f"min_scale={float(scales_f32.min()):.4e})",
            flush=True,
        )
        print(
            f"#   weight_mse:  rtn={weight_mse_rtn:.6e}  gptq={weight_mse_gptq:.6e}  "
            f"ratio={weight_mse_gptq / max(weight_mse_rtn, 1e-30):.4f}",
            flush=True,
        )
        print(
            f"#   output_mse:  rtn={out_mse_rtn:.6e}  gptq={out_mse_gptq:.6e}  "
            f"ratio={out_mse_gptq / max(out_mse_rtn, 1e-30):.4f}  "
            f"(N_calib={N_calib})",
            flush=True,
        )
        print(
            f"#   max_resid:   rtn={per_ch_resid_rtn:.6e}  gptq={per_ch_resid_gptq:.6e}",
            flush=True,
        )
        print(
            f"#   round_trip:  rows_below_127={rows_below_127}/{rows_total}  "
            f"int_mismatch={n_mismatch}  max_dq_drift={max_dq_drift:.6e}  "
            f"mean_dq_drift={mean_dq_drift:.6e}",
            flush=True,
        )


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
                        help="Comma-separated block indices, e.g., '2' or '2,11'.")
    parser.add_argument("--weight-types", default="mlp.c_fc,mlp.c_proj",
                        help="Comma-separated linear layer suffixes, e.g., "
                             "'mlp.c_fc,mlp.c_proj,attn.c_proj'.")
    parser.add_argument("--gptq-percdamp", type=float, default=0.01,
                        help="Diagonal Hessian damping ratio (Frantar default 0.01).")
    parser.add_argument("--gptq-blocksize", type=int, default=128,
                        help="Column-panel size; perf only, does not change result.")
    parser.add_argument("--weight-search-n-seqs", type=int, default=8,
                        help="Number of calibration sequences for GPTQ activation capture. "
                             "Pass 0 to skip the Hessian and fall back to RTN through the "
                             "GPTQ plumbing (D6 plumbing-only check).")
    parser.add_argument("--weight-search-seq-len", type=int, default=64,
                        help="Sequence length for GPTQ activation capture.")
    parser.add_argument("--use-fakequant-activations", action="store_true",
                        help="D3: capture activations through NanoGPTFQReference instead of "
                             "the FP32 reference, so the Hessian sees the same distribution "
                             "the eval pipeline presents to each matmul.")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential GPTQ propagation per Frantar et al.: process target "
                             "weights block-by-block in ascending block index, with activation "
                             "re-capture between blocks so each block's Hessian is built on "
                             "the FP32 forward induced by prior blocks' already-GPTQ'd "
                             "weights. Prevents cross-layer error coupling.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    target_blocks = [int(x.strip()) for x in args.target_blocks.split(",") if x.strip()]
    weight_types = [x.strip() for x in args.weight_types.split(",") if x.strip()]
    target_weights = _expand_target_weights(target_blocks, weight_types)
    print(f"# GPTQ target weights ({len(target_weights)}): {target_weights}", flush=True)

    calib_token_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    calib_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(args.weight_search_n_seqs),
        seq_len=int(args.weight_search_seq_len),
    )

    apply_gptq(
        payload["state_dict"],
        payload["model_args"],
        calib_seqs,
        target_weights,
        percdamp=args.gptq_percdamp,
        blocksize=args.gptq_blocksize,
        use_fakequant_activations=args.use_fakequant_activations,
        payload=payload,
        calibration_token_ids=calib_token_ids,
        ptq_preset=args.ptq_preset,
        calibration_seq_len=args.calibration_seq_len,
        calibration_n_seqs=args.calibration_n_seqs,
        calibration_percentile=args.calibration_percentile,
        sequential=args.sequential,
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
    out["gptq"] = {
        "target_weights": target_weights,
        "percdamp": args.gptq_percdamp,
        "blocksize": args.gptq_blocksize,
        "weight_search_n_seqs": args.weight_search_n_seqs,
        "weight_search_seq_len": args.weight_search_seq_len,
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
