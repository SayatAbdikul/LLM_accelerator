"""Unit tests for the bounded Stage 5 GPT-2 PTQ preset plumbing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from taccel.runtime.stage5_ptq import (
    STAGE5_PTQ_PRESETS,
    Stage5PTQPreset,
    apply_stage5_ptq_scale_policy,
    choose_stage5_ptq_promotion,
    choose_stage5_ptq_winner,
    rank_stage5_ptq_rows,
    resolve_stage5_ptq_preset,
    stage5_dequant_add_residual1_blocks,
    stage5_dequant_add_residual2_blocks,
    stage5_default_ptq_preset_name,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
    validate_stage5_ptq_preset_for_model,
)


DEBUG_TOOL = Path(__file__).resolve().parents[1] / "tools" / "debug_gpt2_perplexity.py"


def _debug_module():
    spec = importlib.util.spec_from_file_location("debug_gpt2_perplexity", DEBUG_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage5_preset_registry_contains_core_presets_and_promoted_default():
    # Core presets must always be present.
    core = {
        "control", "final_ln_99_8", "block9_ln2_99_0", "late_ln_combo",
        "fc1_8_9", "fc2_10", "out_proj_11", "late_mlp_combo", "full_late_combo",
        "out_proj_11_ln_f_99_8", "gpt2_all_pc", "gpt2_all_pc_with_fc1",
        "out_proj_10_11", "out_proj_11_fc1_11", "out_proj_11_block10_ln2_99_0",
        "fc2_11_raw_vadd", "out_proj_11_fc2_11_raw_vadd", "out_proj_11_fc2_10_11_raw_vadd",
        "hessian_gelu_11", "fc2_11_fc2aware_gelu", "out_proj_11_fc2_11_fc2aware_gelu",
        "output_aware_gelu_8_to_11", "output_aware_mlp_8_to_11",
        "mlp_bias_fc2_8_to_11", "mlp_bias_resid2_8_to_11",
    }
    assert core.issubset(set(STAGE5_PTQ_PRESETS))
    assert stage5_default_ptq_preset_name() == "fc2_8_to_11_raw_vadd"
    assert resolve_stage5_ptq_preset("control").name == "control"
    with pytest.raises(KeyError, match="unknown Stage 5 PTQ preset"):
        resolve_stage5_ptq_preset("not_a_preset")


def test_stage5_preset_weight_names_and_residual_policy():
    model_args = {"n_layer": 12}
    assert stage5_requant_pc_weight_names(model_args, "fc1_8_9") == {
        "transformer.h.8.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_fc.weight",
    }
    assert stage5_requant_pc_weight_names(model_args, "fc2_10") == {
        "transformer.h.10.mlp.c_proj.weight",
    }
    assert stage5_requant_pc_weight_names(model_args, "out_proj_11") == {
        "transformer.h.11.attn.c_proj.weight",
    }
    assert stage5_raw_residual1_blocks("out_proj_11") == {11}
    assert 11 not in stage5_dequant_add_residual1_blocks(model_args, "out_proj_11")
    assert len(stage5_dequant_add_residual1_blocks(model_args, "out_proj_11")) == 11
    assert stage5_raw_residual2_blocks("fc2_11_raw_vadd") == {11}
    assert 11 not in stage5_dequant_add_residual2_blocks(model_args, "fc2_11_raw_vadd")
    assert len(stage5_dequant_add_residual2_blocks(model_args, "fc2_11_raw_vadd")) == 11
    assert stage5_raw_residual2_blocks("out_proj_11") == set()
    assert len(stage5_dequant_add_residual2_blocks(model_args, "out_proj_11")) == 12


def test_stage5_preset_rejects_unsupported_block_indices():
    with pytest.raises(ValueError, match="outside range"):
        validate_stage5_ptq_preset_for_model({"n_layer": 6}, "fc1_8_9")
    with pytest.raises(ValueError, match="outside range"):
        validate_stage5_ptq_preset_for_model({"n_layer": 6}, "fc2_11_fc2aware_gelu")
    invalid = Stage5PTQPreset(
        name="bad_fc2aware",
        activation_percentile_nodes={},
        requant_pc_out_proj_blocks=(),
        requant_pc_fc1_blocks=(),
        requant_pc_fc2_blocks=(),
        hessian_gelu_blocks=(),
        fc2_aware_gelu_blocks=(0,),
        output_aware_gelu_blocks=(),
        output_aware_mlp_blocks=(),
        mlp_bias_correction_blocks=(),
        mlp_bias_correction_target="fc2",
    )
    with pytest.raises(ValueError, match="without matching fc2 REQUANT_PC"):
        validate_stage5_ptq_preset_for_model({"n_layer": 1}, invalid)
    invalid_output_aware = Stage5PTQPreset(
        name="bad_output_aware",
        activation_percentile_nodes={},
        requant_pc_out_proj_blocks=(),
        requant_pc_fc1_blocks=(),
        requant_pc_fc2_blocks=(),
        hessian_gelu_blocks=(),
        fc2_aware_gelu_blocks=(),
        output_aware_gelu_blocks=(0,),
        output_aware_mlp_blocks=(),
        mlp_bias_correction_blocks=(),
        mlp_bias_correction_target="fc2",
    )
    with pytest.raises(ValueError, match="without matching fc2 REQUANT_PC"):
        validate_stage5_ptq_preset_for_model({"n_layer": 1}, invalid_output_aware)
    invalid_output_aware_mlp = Stage5PTQPreset(
        name="bad_output_aware_mlp",
        activation_percentile_nodes={},
        requant_pc_out_proj_blocks=(),
        requant_pc_fc1_blocks=(),
        requant_pc_fc2_blocks=(),
        hessian_gelu_blocks=(),
        fc2_aware_gelu_blocks=(),
        output_aware_gelu_blocks=(),
        output_aware_mlp_blocks=(0,),
        mlp_bias_correction_blocks=(),
        mlp_bias_correction_target="fc2",
    )
    with pytest.raises(ValueError, match="without matching fc2 REQUANT_PC"):
        validate_stage5_ptq_preset_for_model({"n_layer": 1}, invalid_output_aware_mlp)


def test_stage5_scale_policy_ties_out_proj_and_residual_for_raw_vadd_block():
    scales = {
        "block10_residual2": 0.03125,
        "block11_out_proj": 0.5,
        "block11_residual1": 0.75,
    }
    updated = apply_stage5_ptq_scale_policy(scales, {"n_layer": 12}, "out_proj_11")
    assert updated["block11_out_proj"] == pytest.approx(scales["block10_residual2"])
    assert updated["block11_residual1"] == pytest.approx(scales["block10_residual2"])
    assert scales["block11_out_proj"] == 0.5


def test_stage5_scale_policy_ties_fc2_and_residual2_for_raw_vadd_block():
    scales = {
        "block11_residual1": 0.03125,
        "block11_fc2": 0.5,
        "block11_residual2": 0.75,
    }
    updated = apply_stage5_ptq_scale_policy(scales, {"n_layer": 12}, "fc2_11_raw_vadd")
    assert updated["block11_fc2"] == pytest.approx(scales["block11_residual1"])
    assert updated["block11_residual2"] == pytest.approx(scales["block11_residual1"])
    assert scales["block11_fc2"] == 0.5


def test_stage5_scale_policy_applies_before_fc2_aware_gelu_search():
    scales = {
        "block11_residual1": 0.03125,
        "block11_fc2": 0.5,
        "block11_residual2": 0.75,
        "block11_gelu": 0.0078125,
    }
    updated = apply_stage5_ptq_scale_policy(scales, {"n_layer": 12}, "fc2_11_fc2aware_gelu")
    assert updated["block11_fc2"] == pytest.approx(scales["block11_residual1"])
    assert updated["block11_residual2"] == pytest.approx(scales["block11_residual1"])
    assert updated["block11_gelu"] == pytest.approx(scales["block11_gelu"])


def test_stage5_scale_policy_combined_preset_keeps_one_raw_vadd_scale():
    # Same-block out_proj+fc2 raw VADD needs residual1 forcing first, then
    # residual2 forcing, so residual1/fc2/residual2 all share one scale.
    scales = {
        "block10_residual2": 0.0625,
        "block11_residual1": 0.03125,
        "block11_fc2": 0.9,
        "block11_residual2": 0.8,
        "block11_out_proj": 0.7,
    }
    updated = apply_stage5_ptq_scale_policy(scales, {"n_layer": 12}, "out_proj_11_fc2_11_raw_vadd")
    assert updated["block11_out_proj"] == pytest.approx(0.0625)
    assert updated["block11_residual1"] == pytest.approx(0.0625)
    assert updated["block11_fc2"] == pytest.approx(0.0625)
    assert updated["block11_residual2"] == pytest.approx(0.0625)


def test_stage5_ranking_and_promotion_rules():
    rows = [
        {
            "name": "control",
            "fake_quant_perplexity": 100.0,
            "mean_target_nll": 5.0,
            "mean_top10_overlap_vs_fp32": 4.0,
            "mean_cosine_vs_fp32": 0.5,
        },
        {
            "name": "late_ln_combo",
            "fake_quant_perplexity": 80.0,
            "mean_target_nll": 4.0,
            "mean_top10_overlap_vs_fp32": 5.0,
            "mean_cosine_vs_fp32": 0.6,
        },
        {
            "name": "fc2_10",
            "fake_quant_perplexity": 80.0,
            "mean_target_nll": 4.0,
            "mean_top10_overlap_vs_fp32": 6.0,
            "mean_cosine_vs_fp32": 0.4,
        },
    ]

    ranked = rank_stage5_ptq_rows(rows)
    assert [row["name"] for row in ranked] == ["fc2_10", "late_ln_combo", "control"]
    assert choose_stage5_ptq_winner(rows)["name"] == "fc2_10"
    assert choose_stage5_ptq_promotion(rows, gate_passed=False) == "control"
    assert choose_stage5_ptq_promotion(rows, gate_passed=True) == "fc2_10"


def test_debug_preset_sweep_reports_all_presets_and_deterministic_winner(monkeypatch):
    from taccel.runtime.stage5_ptq import STAGE5_PTQ_PRESETS
    dbg = _debug_module()
    # Assign quality scores: presets ranked in STAGE5_PTQ_PRESETS order by default,
    # with "late_ln_combo" forced to best so winner is deterministic.
    preset_names = list(STAGE5_PTQ_PRESETS)
    quality = {name: float(len(preset_names) - i) / len(preset_names) for i, name in enumerate(preset_names)}
    quality["late_ln_combo"] = 99.0  # ensure it wins

    def fake_artifacts(payload, calibration_ids, args, preset_name):
        diagnostics = {}
        if "fc2aware" in preset_name:
            diagnostics = {"block11": {"multiplier": 1.25, "objective_mse": 0.0}}
        if "output_aware" in preset_name:
            diagnostics = {"output_aware_gelu": {"block11": {"multiplier": 1.125, "selected_mean_nll": 1.0}}}
        if "output_aware_mlp" in preset_name:
            diagnostics = {"output_aware_mlp": {"block11": {"selected_mean_nll": 1.0}}}
        if "mlp_bias" in preset_name:
            diagnostics = {"mlp_bias_correction": {"block11": {"mean_abs_correction": 0.01}}}
        return {"lm_head": 1.0}, {}, diagnostics

    def fake_fake(payload, eval_tokens, scales, *, ptq_preset=None, bias_corrections=None):
        score = quality.get(ptq_preset.name, 0.5)
        return [np.asarray([score, 2, 1, 0], dtype=np.float32)]

    def fake_golden(payload, eval_tokens, scales, *, ptq_preset=None, bias_corrections=None):
        return fake_fake(payload, eval_tokens, scales, ptq_preset=ptq_preset)

    monkeypatch.setattr(dbg, "_build_artifacts_for_preset", fake_artifacts)
    monkeypatch.setattr(dbg, "run_golden_teacher_forced_logits", fake_golden)
    monkeypatch.setattr(dbg, "run_fake_quant_teacher_forced_logits", fake_fake)

    report = dbg._preset_sweep(
        payload={"model_args": {"n_layer": 12}},
        calibration_ids=[0, 1],
        eval_tokens=[0, 1],
        inputs=[0],
        targets=[0],
        fp32_logits=[np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)],
        vocab_size=4,
        args=SimpleNamespace(),
    )

    result_names = {row["name"] for row in report["rows"]}
    assert set(preset_names).issubset(result_names)
    assert report["winner"]["name"] == "late_ln_combo"
    assert report["proposed_promotion"] == "late_ln_combo"
    assert report["promoted_default"] == "fc2_8_to_11_raw_vadd"
    assert report["default_replacement_candidate"]["baseline"] == "fc2_8_to_11_raw_vadd"
    fc2aware_rows = [row for row in report["rows"] if row["name"] == "fc2_11_fc2aware_gelu"]
    assert fc2aware_rows and "fc2_aware_gelu" in fc2aware_rows[0]
    output_aware_rows = [row for row in report["rows"] if row["name"] == "output_aware_gelu_8_to_11"]
    assert output_aware_rows and "output_aware_gelu" in output_aware_rows[0]
    output_aware_mlp_rows = [row for row in report["rows"] if row["name"] == "output_aware_mlp_8_to_11"]
    assert output_aware_mlp_rows and "output_aware_mlp" in output_aware_mlp_rows[0]
    bias_rows = [row for row in report["rows"] if row["name"] == "mlp_bias_fc2_8_to_11"]
    assert bias_rows and "mlp_bias_correction" in bias_rows[0]
