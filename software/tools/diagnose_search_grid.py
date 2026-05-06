#!/usr/bin/env python3
"""Run the output-aware MLP search and analyze the chosen multipliers per block/group.

Reports:
- Which multiplier each block/group selected (0.75, 0.875, 1.0, 1.125, 1.25, 1.5)
- Whether the grid edges (0.75 or 1.5) were hit (signals grid is too narrow)
- Whether most groups selected 1.0 (signals search is saturated)
- For pair candidates: how often pair-groups won vs single-component groups
- NLL improvement per block
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from taccel.runtime.calibration import (
    OUTPUT_AWARE_MLP_MULTIPLIERS,
    apply_output_aware_mlp_scale_search_from_token_ids,
    build_calibration_scales_from_token_ids,
)
from taccel.runtime.gpt2_perplexity import (
    CALIBRATION_N_SEQS_LARGE,
    CALIBRATION_SEQ_LEN_LARGE,
    tokenize_text_file,
)
from taccel.runtime.stage5_ptq import (
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--ptq-preset", default=None)
    parser.add_argument("--calibration-n-seqs", type=int, default=CALIBRATION_N_SEQS_LARGE)
    parser.add_argument("--calibration-seq-len", type=int, default=CALIBRATION_SEQ_LEN_LARGE)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to dump full diagnostics JSON")
    args = parser.parse_args()

    preset_name = args.ptq_preset or stage5_default_ptq_preset_name()
    preset = resolve_stage5_ptq_preset(preset_name)
    print(f"# Preset: {preset.name}")
    print(f"# MLP blocks: {preset.output_aware_mlp_blocks}")
    print(f"# Pair candidates: {preset.output_aware_include_pairs}")
    print(f"# Multiplier grid: {OUTPUT_AWARE_MLP_MULTIPLIERS}")
    print()

    payload = torch.load(args.checkpoint, map_location="cpu")
    calib_ids = tokenize_text_file(args.tokenizer_dir, args.calibration_text)
    base_scales = build_calibration_scales_from_token_ids(
        payload,
        calib_ids,
        n_seqs=args.calibration_n_seqs,
        seq_len=args.calibration_seq_len,
        activation_percentile_overrides=preset.activation_percentile_nodes or None,
        hessian_gelu_blocks=preset.hessian_gelu_blocks,
    )
    base_scales = apply_stage5_ptq_scale_policy(base_scales, payload["model_args"], preset)

    if not preset.output_aware_mlp_blocks:
        print("Preset has no output_aware_mlp_blocks — nothing to diagnose.")
        return 0

    final_scales, diag = apply_output_aware_mlp_scale_search_from_token_ids(
        payload,
        calib_ids,
        base_scales,
        blocks=preset.output_aware_mlp_blocks,
        ptq_preset=preset,
        n_seqs=args.calibration_n_seqs,
        seq_len=args.calibration_seq_len,
        include_pair_candidates=preset.output_aware_include_pairs,
    )

    multiplier_counter: Counter = Counter()
    edge_hits = 0
    no_change = 0
    accepted = 0
    rejected = 0
    pair_wins = 0
    single_wins = 0
    grid_min = float(min(OUTPUT_AWARE_MLP_MULTIPLIERS))
    grid_max = float(max(OUTPUT_AWARE_MLP_MULTIPLIERS))

    print(f"{'Block':<8}{'Group':<28}{'Mult':>8}  {'Accepted':>8}  {'Δ NLL':>14}")
    print("-" * 80)
    total_nll_drop = 0.0
    for block_key, block_diag in diag.items():
        for group_name, gres in block_diag["groups"].items():
            mult = float(gres["multiplier"])
            acc = bool(gres["accepted"])
            d_nll = float(gres["selected_mean_nll"]) - float(gres["baseline_mean_nll"])
            multiplier_counter[mult] += 1
            if mult == 1.0:
                no_change += 1
            if mult == grid_min or mult == grid_max:
                edge_hits += 1
            if acc:
                accepted += 1
                if "_" in group_name:  # pair group
                    pair_wins += 1
                else:
                    single_wins += 1
            else:
                rejected += 1
            total_nll_drop += d_nll
            print(f"{block_key:<8}{group_name:<28}{mult:>8.3f}  {str(acc):>8}  {d_nll:>+14.6f}")
        print()

    print("=" * 80)
    print("Summary:")
    print(f"  Total group searches: {accepted + rejected}")
    print(f"  Accepted: {accepted}  Rejected (mult=1.0): {rejected}")
    print(f"  Of accepted: pair-groups={pair_wins}, single-component={single_wins}")
    print(f"  Multiplier distribution:")
    for m in sorted(multiplier_counter):
        bar = "#" * multiplier_counter[m]
        marker = ""
        if m == grid_min or m == grid_max:
            marker = "  <-- GRID EDGE"
        if m == 1.0:
            marker = "  <-- NO CHANGE"
        print(f"    {m:>6.3f}: {multiplier_counter[m]:>3d} {bar}{marker}")
    edge_frac = edge_hits / max(1, accepted + rejected)
    print(f"  Grid-edge hits: {edge_hits} ({edge_frac:.1%}) — high fraction signals grid is too narrow")
    print(f"  Total NLL drop across search: {total_nll_drop:+.6f}")

    if args.json_out:
        args.json_out.write_text(json.dumps(diag, indent=2))
        print(f"  Full diagnostics written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
