#!/usr/bin/env python3
"""Compose SmoothQuant → GPTQ → bias correction in one driver, then evaluate.

Phase 2 of the GPTQ follow-up plan needs all three weight refinements applied
in a fixed order on the same target set, so the only honest way to compare a
composition against its components is to share the calibration tokens, the
calibration windows, and the perplexity eval. This driver wires the existing
``apply_*`` functions from the three standalone tools and calls the standard
``evaluate_gpt2_perplexity`` at the end. No algorithm changes — every
state-dict modification reproduces what the standalone tool would do if it
were run on the same checkpoint.

Order of application (matches the plan rationale):

1. **SmoothQuant** rebalances activation/weight magnitudes (modifies LN
   weights and Linear weight columns). Run first because GPTQ's Hessian
   should be computed on the rebalanced distribution.
2. **GPTQ** refines INT8 weight rounding using the layer-wise Hessian.
   Sequential propagation is supported.
3. **Bias correction** compensates for the quantization mean shift on the
   final weight rounding. Run last because it depends on the post-GPTQ
   weight quantization.

If any of the three flag groups is omitted, that step is skipped — passing
no flag groups runs an unmodified baseline through the GPTQ-style plumbing
(a useful comparison anchor in its own right).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Reuse the apply_* functions and small helpers from the standalone tools.
# Each tool's main() is gated on __name__ == "__main__", so importing them is
# free of side-effects beyond a few argparse imports.
from tools.eval_with_smooth_quant import (
    _capture_node_activations as _capture_smooth_node,
    apply_smoothquant_ln1_qkv,
    apply_smoothquant_ln2_fc1,
)
from tools.eval_with_gptq import apply_gptq, _expand_target_weights as _gptq_expand
from tools.eval_with_bias_correction import (
    apply_bias_correction,
    _expand_target_weights as _bc_expand,
)

from taccel.runtime.calibration import (
    build_calibration_scales_from_token_ids,
    build_calibration_seqs_from_token_ids,
)
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


# ---------------------------------------------------------------------------
# Step plumbing
# ---------------------------------------------------------------------------

def _parse_int_csv(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_str_csv(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _run_smoothquant_step(
    payload: dict,
    sq_blocks: list[int],
    sq_targets: list[str],
    sq_alpha: float,
    sq_search_n_seqs: int,
    sq_search_seq_len: int,
    calib_token_ids,
) -> list[dict]:
    """Run SmoothQuant block-by-block. Mirrors eval_with_smooth_quant.main."""
    sq_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(sq_search_n_seqs),
        seq_len=int(sq_search_seq_len),
    )
    state_dict = payload["state_dict"]
    model_args = payload["model_args"]
    reports: list[dict] = []
    for block in sq_blocks:
        for target in sq_targets:
            if target == "ln_2_fc1":
                node_name = f"block{block}_ln2"
                acts = _capture_smooth_node(state_dict, model_args, sq_seqs, node_name)
                reports.append(apply_smoothquant_ln2_fc1(state_dict, acts, block, float(sq_alpha)))
            elif target == "ln_1_qkv":
                node_name = f"block{block}_ln1"
                acts = _capture_smooth_node(state_dict, model_args, sq_seqs, node_name)
                reports.append(
                    apply_smoothquant_ln1_qkv(state_dict, model_args, acts, block, float(sq_alpha))
                )
            else:
                raise ValueError(f"unknown SmoothQuant target: {target!r}")
    print(f"# Applied SmoothQuant to {len(reports)} pairs:", flush=True)
    for r in reports:
        print(
            f"#   block{r['block']}/{r['target']}: alpha={r['alpha']}  "
            f"smooth=[{r['smooth_min']:.3e}, {r['smooth_max']:.3e}] mean={r['smooth_mean']:.3e}",
            flush=True,
        )
    return reports


def _run_gptq_step(
    payload: dict,
    gptq_blocks: list[int],
    gptq_types: list[str],
    gptq_percdamp: float,
    gptq_blocksize: int,
    gptq_n_seqs: int,
    gptq_seq_len: int,
    sequential: bool,
    use_fakequant_activations: bool,
    ptq_preset: str | None,
    calibration_seq_len: int,
    calibration_n_seqs: int,
    calibration_percentile: float,
    calib_token_ids,
) -> None:
    """Run apply_gptq with the same plumbing eval_with_gptq.main uses."""
    target_weights = _gptq_expand(gptq_blocks, gptq_types)
    print(f"# GPTQ targets ({len(target_weights)}): {target_weights}", flush=True)
    calib_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(gptq_n_seqs),
        seq_len=int(gptq_seq_len),
    )
    apply_gptq(
        payload["state_dict"],
        payload["model_args"],
        calib_seqs,
        target_weights,
        percdamp=float(gptq_percdamp),
        blocksize=int(gptq_blocksize),
        use_fakequant_activations=use_fakequant_activations,
        payload=payload,
        calibration_token_ids=calib_token_ids,
        ptq_preset=ptq_preset,
        calibration_seq_len=calibration_seq_len,
        calibration_n_seqs=calibration_n_seqs,
        calibration_percentile=calibration_percentile,
        sequential=sequential,
    )


def _run_bc_step(
    payload: dict,
    bc_blocks: list[int],
    bc_types: list[str],
    bc_search_n_seqs: int,
    bc_search_seq_len: int,
    ptq_preset: str | None,
    calibration_seq_len: int,
    calibration_n_seqs: int,
    calibration_percentile: float,
    calib_token_ids,
) -> list[dict]:
    """Run bias correction. Mirrors eval_with_bias_correction.main."""
    target_weights = _bc_expand(bc_blocks, bc_types)
    preset = resolve_stage5_ptq_preset(ptq_preset or stage5_default_ptq_preset_name())
    print(f"# Bias-correction targets ({len(target_weights)}): {target_weights}", flush=True)
    print(f"# Using preset for activation scale baseline: {preset.name}", flush=True)
    base_scales = build_calibration_scales_from_token_ids(
        payload,
        calib_token_ids,
        n_seqs=int(calibration_n_seqs),
        seq_len=int(calibration_seq_len),
        percentile=float(calibration_percentile),
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    base_scales = apply_stage5_ptq_scale_policy(base_scales, payload["model_args"], preset)
    bc_seqs = build_calibration_seqs_from_token_ids(
        calib_token_ids,
        n_seqs=int(bc_search_n_seqs),
        seq_len=int(bc_search_seq_len),
    )
    reports = apply_bias_correction(
        payload["state_dict"],
        payload["model_args"],
        base_scales,
        bc_seqs,
        target_weights,
    )
    print(f"# Applied bias correction to {len(reports)} layers", flush=True)
    if reports:
        err_max = max(r["err_abs_max"] for r in reports)
        err_mean = sum(r["err_abs_mean"] for r in reports) / len(reports)
        print(f"#   err_abs_max overall = {err_max:.4e}", flush=True)
        print(f"#   err_abs_mean overall = {err_mean:.4e}", flush=True)
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    # SmoothQuant flags
    parser.add_argument("--sq-blocks", default="",
                        help="Comma-separated block indices for SmoothQuant. "
                             "Empty (default) skips the SmoothQuant step.")
    parser.add_argument("--sq-targets", default="ln_2_fc1",
                        help="Comma-separated SmoothQuant targets per block: ln_2_fc1, ln_1_qkv.")
    parser.add_argument("--sq-alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength.")
    parser.add_argument("--sq-search-n-seqs", type=int, default=8)
    parser.add_argument("--sq-search-seq-len", type=int, default=64)

    # GPTQ flags (same names as eval_with_gptq.py with --gptq- prefix where it
    # disambiguates from this driver's own flags).
    parser.add_argument("--gptq-blocks", default="",
                        help="Comma-separated block indices for GPTQ. Empty skips GPTQ.")
    parser.add_argument("--gptq-types", default="mlp.c_proj",
                        help="Comma-separated linear layer suffixes (mlp.c_fc, mlp.c_proj, attn.c_proj).")
    parser.add_argument("--gptq-percdamp", type=float, default=0.01)
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--gptq-n-seqs", type=int, default=128)
    parser.add_argument("--gptq-seq-len", type=int, default=64)
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential GPTQ propagation (Frantar-style).")
    parser.add_argument("--use-fakequant-activations", action="store_true",
                        help="GPTQ Hessian capture via fake-quant pipeline.")

    # Bias-correction flags
    parser.add_argument("--bc-blocks", default="",
                        help="Comma-separated block indices for bias correction. Empty skips BC.")
    parser.add_argument("--bc-types", default="mlp.c_proj",
                        help="Comma-separated linear layer suffixes for BC.")
    parser.add_argument("--bc-search-n-seqs", type=int, default=8)
    parser.add_argument("--bc-search-seq-len", type=int, default=64)

    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")

    sq_blocks = _parse_int_csv(args.sq_blocks)
    sq_targets = _parse_str_csv(args.sq_targets)
    gptq_blocks = _parse_int_csv(args.gptq_blocks)
    gptq_types = _parse_str_csv(args.gptq_types)
    bc_blocks = _parse_int_csv(args.bc_blocks)
    bc_types = _parse_str_csv(args.bc_types)

    print(f"# Composition steps: SQ={bool(sq_blocks)}  GPTQ={bool(gptq_blocks)}  BC={bool(bc_blocks)}",
          flush=True)

    calib_token_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)

    sq_reports: list[dict] = []
    bc_reports: list[dict] = []

    if sq_blocks:
        sq_reports = _run_smoothquant_step(
            payload,
            sq_blocks,
            sq_targets,
            args.sq_alpha,
            args.sq_search_n_seqs,
            args.sq_search_seq_len,
            calib_token_ids,
        )

    if gptq_blocks:
        _run_gptq_step(
            payload,
            gptq_blocks,
            gptq_types,
            args.gptq_percdamp,
            args.gptq_blocksize,
            args.gptq_n_seqs,
            args.gptq_seq_len,
            args.sequential,
            args.use_fakequant_activations,
            args.ptq_preset,
            args.calibration_seq_len,
            args.calibration_n_seqs,
            args.calibration_percentile,
            calib_token_ids,
        )

    if bc_blocks:
        bc_reports = _run_bc_step(
            payload,
            bc_blocks,
            bc_types,
            args.bc_search_n_seqs,
            args.bc_search_seq_len,
            args.ptq_preset,
            args.calibration_seq_len,
            args.calibration_n_seqs,
            args.calibration_percentile,
            calib_token_ids,
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
    out["compose"] = {
        "sq": {
            "blocks": sq_blocks,
            "targets": sq_targets,
            "alpha": float(args.sq_alpha),
            "reports": sq_reports,
        } if sq_blocks else None,
        "gptq": {
            "blocks": gptq_blocks,
            "types": gptq_types,
            "percdamp": float(args.gptq_percdamp),
            "blocksize": int(args.gptq_blocksize),
            "n_seqs": int(args.gptq_n_seqs),
            "seq_len": int(args.gptq_seq_len),
            "sequential": bool(args.sequential),
            "use_fakequant_activations": bool(args.use_fakequant_activations),
        } if gptq_blocks else None,
        "bc": {
            "blocks": bc_blocks,
            "types": bc_types,
            "reports": bc_reports,
        } if bc_blocks else None,
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
