"""Stage 5 GPT-2 PTQ preset plumbing.

These presets are intentionally narrow and internal to the GPT-2 debug/eval
path. They let us reuse existing late-node PTQ knobs without changing the
public compiler surface or the smaller Stage 3/4 fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class Stage5PTQPreset:
    name: str
    activation_percentile_nodes: Dict[str, float]
    requant_pc_out_proj_blocks: tuple[int, ...]
    requant_pc_fc1_blocks: tuple[int, ...]
    requant_pc_fc2_blocks: tuple[int, ...]


def _preset(
    name: str,
    *,
    activation_percentile_nodes: Mapping[str, float] | None = None,
    requant_pc_out_proj_blocks: Sequence[int] = (),
    requant_pc_fc1_blocks: Sequence[int] = (),
    requant_pc_fc2_blocks: Sequence[int] = (),
) -> Stage5PTQPreset:
    return Stage5PTQPreset(
        name=name,
        activation_percentile_nodes=dict(activation_percentile_nodes or {}),
        requant_pc_out_proj_blocks=tuple(int(v) for v in requant_pc_out_proj_blocks),
        requant_pc_fc1_blocks=tuple(int(v) for v in requant_pc_fc1_blocks),
        requant_pc_fc2_blocks=tuple(int(v) for v in requant_pc_fc2_blocks),
    )


STAGE5_PTQ_PRESETS: Dict[str, Stage5PTQPreset] = {
    "control": _preset("control"),
    "final_ln_99_8": _preset(
        "final_ln_99_8",
        activation_percentile_nodes={"ln_f": 99.8},
    ),
    "block9_ln2_99_0": _preset(
        "block9_ln2_99_0",
        activation_percentile_nodes={"block9_ln2": 99.0},
    ),
    "late_ln_combo": _preset(
        "late_ln_combo",
        activation_percentile_nodes={"ln_f": 99.8, "block9_ln2": 99.0},
    ),
    "fc1_8_9": _preset(
        "fc1_8_9",
        requant_pc_fc1_blocks=(8, 9),
    ),
    "fc2_10": _preset(
        "fc2_10",
        requant_pc_fc2_blocks=(10,),
    ),
    "out_proj_11": _preset(
        "out_proj_11",
        requant_pc_out_proj_blocks=(11,),
    ),
    "late_mlp_combo": _preset(
        "late_mlp_combo",
        activation_percentile_nodes={"ln_f": 99.8, "block9_ln2": 99.0},
        requant_pc_fc1_blocks=(8, 9),
        requant_pc_fc2_blocks=(10,),
    ),
    "full_late_combo": _preset(
        "full_late_combo",
        activation_percentile_nodes={"ln_f": 99.8, "block9_ln2": 99.0},
        requant_pc_fc1_blocks=(8, 9),
        requant_pc_fc2_blocks=(10,),
        requant_pc_out_proj_blocks=(11,),
    ),
}

# Updated only after a preset wins on the real local GPT-2 checkpoint and still
# keeps the existing golden-vs-fake gates green.
PROMOTED_STAGE5_PTQ_PRESET = "control"


def stage5_default_ptq_preset_name() -> str:
    return PROMOTED_STAGE5_PTQ_PRESET


def resolve_stage5_ptq_preset(preset: str | Stage5PTQPreset | None) -> Stage5PTQPreset:
    if preset is None:
        return STAGE5_PTQ_PRESETS["control"]
    if isinstance(preset, Stage5PTQPreset):
        return preset
    try:
        return STAGE5_PTQ_PRESETS[str(preset)]
    except KeyError as exc:
        raise KeyError(
            f"unknown Stage 5 PTQ preset {preset!r}; choose one of {list(STAGE5_PTQ_PRESETS)}"
        ) from exc


def validate_stage5_ptq_preset_for_model(
    model_args_or_config,
    preset: str | Stage5PTQPreset | None,
) -> Stage5PTQPreset:
    resolved = resolve_stage5_ptq_preset(preset)
    n_layer = int(
        getattr(model_args_or_config, "n_layer", None)
        if hasattr(model_args_or_config, "n_layer")
        else model_args_or_config["n_layer"]
    )
    invalid = sorted(
        {
            idx
            for idx in (
                resolved.requant_pc_out_proj_blocks
                + resolved.requant_pc_fc1_blocks
                + resolved.requant_pc_fc2_blocks
            )
            if idx < 0 or idx >= n_layer
        }
    )
    if invalid:
        raise ValueError(
            f"Stage 5 PTQ preset {resolved.name!r} has block indices outside range for n_layer={n_layer}: {invalid}"
        )
    return resolved


def stage5_requant_pc_weight_names(
    model_args_or_config,
    preset: str | Stage5PTQPreset | None,
) -> set[str]:
    resolved = validate_stage5_ptq_preset_for_model(model_args_or_config, preset)
    names: set[str] = set()
    for block in resolved.requant_pc_out_proj_blocks:
        names.add(f"transformer.h.{block}.attn.c_proj.weight")
    for block in resolved.requant_pc_fc1_blocks:
        names.add(f"transformer.h.{block}.mlp.c_fc.weight")
    for block in resolved.requant_pc_fc2_blocks:
        names.add(f"transformer.h.{block}.mlp.c_proj.weight")
    return names


def stage5_raw_residual1_blocks(preset: str | Stage5PTQPreset | None) -> set[int]:
    resolved = resolve_stage5_ptq_preset(preset)
    return set(int(idx) for idx in resolved.requant_pc_out_proj_blocks)


def stage5_dequant_add_residual1_blocks(
    model_args_or_config,
    preset: str | Stage5PTQPreset | None,
) -> set[int]:
    resolved = validate_stage5_ptq_preset_for_model(model_args_or_config, preset)
    n_layer = int(
        getattr(model_args_or_config, "n_layer", None)
        if hasattr(model_args_or_config, "n_layer")
        else model_args_or_config["n_layer"]
    )
    raw_blocks = stage5_raw_residual1_blocks(resolved)
    return {idx for idx in range(n_layer) if idx not in raw_blocks}


def apply_stage5_ptq_scale_policy(
    calibration_scales: Mapping[str, float],
    model_args_or_config,
    preset: str | Stage5PTQPreset | None,
) -> Dict[str, float]:
    resolved = validate_stage5_ptq_preset_for_model(model_args_or_config, preset)
    scales = dict(calibration_scales)
    for block in stage5_raw_residual1_blocks(resolved):
        skip_name = "tok_pos_add" if block == 0 else f"block{block - 1}_residual2"
        shared_scale = float(scales.get(skip_name, 6.0 / 127.0))
        scales[f"block{block}_out_proj"] = shared_scale
        scales[f"block{block}_residual1"] = shared_scale
    return scales


def rank_stage5_ptq_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized = [dict(row) for row in rows]
    normalized.sort(
        key=lambda row: (
            float(row["fake_quant_perplexity"]),
            float(row["mean_target_nll"]),
            -float(row["mean_top10_overlap_vs_fp32"]),
            -float(row["mean_cosine_vs_fp32"]),
            str(row["name"]),
        )
    )
    return normalized


def choose_stage5_ptq_winner(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    ranked = rank_stage5_ptq_rows(rows)
    return ranked[0] if ranked else None


def choose_stage5_ptq_promotion(
    rows: Sequence[Mapping[str, object]],
    *,
    gate_passed: bool,
    control_name: str = "control",
) -> str:
    if not gate_passed:
        return control_name
    ranked = rank_stage5_ptq_rows(rows)
    if not ranked:
        return control_name
    best = ranked[0]
    control = next((row for row in ranked if str(row["name"]) == control_name), None)
    if control is None:
        return control_name
    if str(best["name"]) == control_name:
        return control_name
    if float(best["fake_quant_perplexity"]) >= float(control["fake_quant_perplexity"]):
        return control_name
    return str(best["name"])
