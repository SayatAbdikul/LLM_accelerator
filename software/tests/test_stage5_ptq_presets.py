"""Unit tests for the bounded Stage 5 GPT-2 PTQ preset plumbing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from taccel.runtime import calibration as calibration_mod
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
    stage5_gelu_from_accum_blocks,
    stage5_default_ptq_preset_name,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
    validate_stage5_ptq_preset_for_model,
)
from taccel.runtime.tiny_fixture import quantize_fixture_payload


DEBUG_TOOL = Path(__file__).resolve().parents[1] / "tools" / "debug_gpt2_perplexity.py"


def _one_layer_payload():
    d_model = 16
    mlp_dim = 4 * d_model
    vocab = 16
    block = 16
    state = {
        "transformer.wte.weight": np.linspace(-0.2, 0.2, vocab * d_model, dtype=np.float32).reshape(vocab, d_model),
        "transformer.wpe.weight": np.linspace(0.05, 0.25, block * d_model, dtype=np.float32).reshape(block, d_model),
        "transformer.ln_f.weight": np.ones(d_model, dtype=np.float32),
        "transformer.ln_f.bias": np.zeros(d_model, dtype=np.float32),
        "lm_head.weight": np.linspace(-0.3, 0.4, vocab * d_model, dtype=np.float32).reshape(vocab, d_model),
    }
    for ln in ("ln_1", "ln_2"):
        state[f"transformer.h.0.{ln}.weight"] = np.ones(d_model, dtype=np.float32)
        state[f"transformer.h.0.{ln}.bias"] = np.zeros(d_model, dtype=np.float32)
    for proj in ("query", "key", "value"):
        state[f"transformer.h.0.attn.c_attn.weight_h0_{proj}"] = np.linspace(
            -0.4, 0.4, d_model * d_model, dtype=np.float32
        ).reshape(d_model, d_model)
        state[f"transformer.h.0.attn.c_attn.bias_h0_{proj}"] = np.zeros(d_model, dtype=np.float32)
    state["transformer.h.0.attn.c_proj.weight"] = np.linspace(-0.5, 0.5, d_model * d_model, dtype=np.float32).reshape(d_model, d_model)
    state["transformer.h.0.attn.c_proj.bias"] = np.zeros(d_model, dtype=np.float32)
    fc_rows = [np.linspace(-0.05 * (idx + 1), 0.05 * (idx + 1), d_model, dtype=np.float32) for idx in range(mlp_dim)]
    state["transformer.h.0.mlp.c_fc.weight"] = np.stack(fc_rows, axis=0)
    state["transformer.h.0.mlp.c_fc.bias"] = np.zeros(mlp_dim, dtype=np.float32)
    state["transformer.h.0.mlp.c_proj.weight"] = np.linspace(-0.3, 0.3, d_model * mlp_dim, dtype=np.float32).reshape(d_model, mlp_dim)
    state["transformer.h.0.mlp.c_proj.bias"] = np.zeros(d_model, dtype=np.float32)
    return {
        "model_args": {
            "n_layer": 1,
            "n_head": 1,
            "n_embd": d_model,
            "block_size": block,
            "vocab_size": vocab,
            "layer_norm_epsilon": 1e-5,
        },
        "state_dict": state,
    }


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
        "output_aware_mlp_0_to_11", "output_aware_mlp_0_1_4_8_to_11",
        "output_aware_mlp_0_1_4_6_7_8_to_11",
        "output_aware_attn_all", "output_aware_mlp_attn_0_1_4_8_to_11",
        "output_aware_mlp_lm_head_0_1_4_8_to_11",
        "output_aware_mlp_lm_head_0_1_2_4_8_to_11",
        "output_aware_mlp_lm_head_0_11_pc_full",
        "output_aware_mlp_lm_head_0_11_pc_full_bc",
        "gelu_accum_8_to_11", "output_aware_mlp_gelu_accum_8_to_11",
        "quarot_baseline", "quarot_with_bc",
    }
    assert core.issubset(set(STAGE5_PTQ_PRESETS))
    assert stage5_default_ptq_preset_name() == "output_aware_mlp_lm_head_0_11_pc_full_bc"
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
    assert stage5_gelu_from_accum_blocks("gelu_accum_8_to_11") == {8, 9, 10, 11}


def test_gelu_from_accum_quantizes_fc1_with_uniform_weight_scale():
    payload = _one_layer_payload()
    scales = {
        "tok_pos_add": 0.01,
        "block0_ln1": 0.01,
        "block0_concat": 0.01,
        "block0_ln2": 0.01,
        "block0_gelu": 0.01,
    }
    per_channel_weights, _, _, _, _ = quantize_fixture_payload(payload, calibration_scales=scales)
    per_tensor_weights, _, _, _, _ = quantize_fixture_payload(
        payload,
        calibration_scales=scales,
        per_tensor_fc1_blocks={0},
    )

    per_channel_scales = per_channel_weights["transformer.h.0.mlp.c_fc.weight"][1]
    per_tensor_scales = per_tensor_weights["transformer.h.0.mlp.c_fc.weight"][1]
    assert not np.allclose(per_channel_scales, per_channel_scales[0])
    assert np.allclose(per_tensor_scales, per_tensor_scales[0])


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
        output_aware_attn_blocks=(),
        output_aware_lm_head=False,
        output_aware_include_pairs=False,
        output_aware_mlp_passes=1,
        bias_correction_blocks=(),
        bias_correction_weight_types=(),
        gelu_from_accum_blocks=(),
        quarot_enabled=False,
        quarot_seed=0xCAFE,
        quarot_kind="random_orthogonal",
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
        output_aware_attn_blocks=(),
        output_aware_lm_head=False,
        output_aware_include_pairs=False,
        output_aware_mlp_passes=1,
        bias_correction_blocks=(),
        bias_correction_weight_types=(),
        gelu_from_accum_blocks=(),
        quarot_enabled=False,
        quarot_seed=0xCAFE,
        quarot_kind="random_orthogonal",
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
        output_aware_attn_blocks=(),
        output_aware_lm_head=False,
        output_aware_include_pairs=False,
        output_aware_mlp_passes=1,
        bias_correction_blocks=(),
        bias_correction_weight_types=(),
        gelu_from_accum_blocks=(),
        quarot_enabled=False,
        quarot_seed=0xCAFE,
        quarot_kind="random_orthogonal",
    )
    with pytest.raises(ValueError, match="without matching fc2 REQUANT_PC"):
        validate_stage5_ptq_preset_for_model({"n_layer": 1}, invalid_output_aware_mlp)
    invalid_gelu_accum = Stage5PTQPreset(
        name="bad_gelu_accum_overlap",
        activation_percentile_nodes={},
        requant_pc_out_proj_blocks=(),
        requant_pc_fc1_blocks=(0,),
        requant_pc_fc2_blocks=(),
        hessian_gelu_blocks=(),
        fc2_aware_gelu_blocks=(),
        output_aware_gelu_blocks=(),
        output_aware_mlp_blocks=(),
        output_aware_attn_blocks=(),
        output_aware_lm_head=False,
        output_aware_include_pairs=False,
        output_aware_mlp_passes=1,
        bias_correction_blocks=(),
        bias_correction_weight_types=(),
        gelu_from_accum_blocks=(0,),
        quarot_enabled=False,
        quarot_seed=0xCAFE,
        quarot_kind="random_orthogonal",
    )
    with pytest.raises(ValueError, match="GELU-from-ACCUM and FC1 REQUANT_PC"):
        validate_stage5_ptq_preset_for_model({"n_layer": 1}, invalid_gelu_accum)

    invalid_quarot_kind = Stage5PTQPreset(
        name="bad_quarot_kind",
        activation_percentile_nodes={},
        requant_pc_out_proj_blocks=(),
        requant_pc_fc1_blocks=(),
        requant_pc_fc2_blocks=(),
        hessian_gelu_blocks=(),
        fc2_aware_gelu_blocks=(),
        output_aware_gelu_blocks=(),
        output_aware_mlp_blocks=(),
        output_aware_attn_blocks=(),
        output_aware_lm_head=False,
        output_aware_include_pairs=False,
        output_aware_mlp_passes=1,
        bias_correction_blocks=(),
        bias_correction_weight_types=(),
        gelu_from_accum_blocks=(),
        quarot_enabled=True,
        quarot_seed=0xCAFE,
        quarot_kind="bogus_rotation_kind",
    )
    with pytest.raises(ValueError, match="unsupported quarot_kind"):
        validate_stage5_ptq_preset_for_model({"n_layer": 1}, invalid_quarot_kind)


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


def test_output_aware_attn_blockwide_uses_one_multiplier_for_all_heads(monkeypatch):
    payload = {"model_args": {"n_layer": 1, "n_head": 2, "vocab_size": 4}, "state_dict": {}}
    scales = {
        "block0_head0_attn_v": 1.0,
        "block0_head1_attn_v": 2.0,
    }

    def fake_mean(payload, seqs, scales, *, ptq_preset):
        return float(scales["block0_head0_attn_v"] + scales["block0_head1_attn_v"])

    monkeypatch.setattr(calibration_mod, "_mean_fake_quant_target_nll", fake_mean)
    updated, diagnostics = calibration_mod.apply_output_aware_attn_scale_search_from_token_ids(
        payload,
        [0, 1, 2],
        scales,
        blocks=(0,),
        ptq_preset="control",
        multipliers=(0.75, 1.0),
    )

    assert updated["block0_head0_attn_v"] == pytest.approx(0.75)
    assert updated["block0_head1_attn_v"] == pytest.approx(1.5)
    assert tuple(diagnostics["block0"]["groups"]) == ("attn_v",)


def test_output_aware_attn_value_search_keeps_kv_load_scale_synced(monkeypatch):
    payload = {"model_args": {"n_layer": 1, "n_head": 1, "vocab_size": 4}, "state_dict": {}}
    scales = {
        "block0_head0_attn_v": 1.0,
        "block0_head0_value": 1.0,
        "block0_head0_value_kv_load": 1.0,
    }

    def fake_mean(payload, seqs, scales, *, ptq_preset):
        return float(scales["block0_head0_value"] + scales["block0_head0_value_kv_load"])

    monkeypatch.setattr(calibration_mod, "_mean_fake_quant_target_nll", fake_mean)
    updated, diagnostics = calibration_mod.apply_output_aware_attn_scale_search_from_token_ids(
        payload,
        [0, 1, 2],
        scales,
        blocks=(0,),
        ptq_preset="control",
        multipliers=(0.75, 1.0),
        include_value_search=True,
    )

    assert updated["block0_head0_value"] == pytest.approx(0.75)
    assert updated["block0_head0_value_kv_load"] == pytest.approx(0.75)
    assert diagnostics["block0"]["groups"]["value"]["key_groups"] == [
        ["block0_head0_value", "block0_head0_value_kv_load"]
    ]


def test_output_aware_lm_head_scale_search_updates_lm_head_scale(monkeypatch):
    payload = {"model_args": {"n_layer": 1, "n_head": 1, "vocab_size": 4}, "state_dict": {}}
    scales = {"lm_head": 2.0}

    def fake_mean(payload, seqs, scales, *, ptq_preset):
        return float(scales.get("lm_head", 1.0))

    monkeypatch.setattr(calibration_mod, "_mean_fake_quant_target_nll", fake_mean)
    updated, diagnostics = calibration_mod.apply_output_aware_lm_head_scale_search_from_token_ids(
        payload,
        [0, 1, 2],
        scales,
        ptq_preset="control",
        multipliers=(0.75, 1.0, 1.25),
    )

    assert updated["lm_head"] == pytest.approx(2.0 * 0.75)
    assert diagnostics["accepted"] is True
    assert diagnostics["multiplier"] == pytest.approx(0.75)
    assert diagnostics["old_scale"] == pytest.approx(2.0)
    assert diagnostics["new_scale"] == pytest.approx(2.0 * 0.75)


def test_output_aware_gelu_scale_search_accepts_search_caps(monkeypatch):
    payload = {"model_args": {"n_layer": 1, "n_head": 1, "vocab_size": 4}, "state_dict": {}}
    scales = {"block0_gelu": 2.0, "lm_head": 1.0}
    seen_lengths = []

    def fake_mean(payload, seqs, scales, *, ptq_preset):
        seen_lengths.append((len(seqs), len(seqs[0]) if seqs else 0))
        return float(scales["block0_gelu"])

    monkeypatch.setattr(calibration_mod, "_mean_fake_quant_target_nll", fake_mean)
    updated, diagnostics = calibration_mod.apply_output_aware_gelu_scale_search_from_token_ids(
        payload,
        list(range(20)),
        scales,
        blocks=(0,),
        ptq_preset="control",
        n_seqs=8,
        seq_len=16,
        multipliers=(0.75, 1.0),
        search_n_seqs_max=2,
        search_seq_len_max=4,
    )

    assert updated["block0_gelu"] == pytest.approx(1.5)
    assert diagnostics["block0"]["search_n_seqs"] == 2
    assert diagnostics["block0"]["search_seq_len"] == 4
    assert all(shape == (2, 4) for shape in seen_lengths)


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
        if "output_aware_attn" in preset_name:
            diagnostics["output_aware_attn"] = {"block0": {"selected_mean_nll": 1.0}}
        if "lm_head" in preset_name:
            diagnostics["output_aware_lm_head"] = {"accepted": True, "multiplier": 0.875}
        return {"lm_head": 1.0}, diagnostics

    def fake_fake(payload, eval_tokens, scales, *, ptq_preset=None):
        score = quality.get(ptq_preset.name, 0.5)
        return [np.asarray([score, 2, 1, 0], dtype=np.float32)]

    def fake_golden(payload, eval_tokens, scales, *, ptq_preset=None):
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
    assert report["promoted_default"] == "output_aware_mlp_lm_head_0_11_pc_full_bc"
    assert report["default_replacement_candidate"]["baseline"] == "output_aware_mlp_lm_head_0_11_pc_full_bc"
    assert report["default_replacement_candidate"]["promotion_threshold"] == pytest.approx(0.10)
    assert "passes_10pct_rule" in report["default_replacement_candidate"]
    fc2aware_rows = [row for row in report["rows"] if row["name"] == "fc2_11_fc2aware_gelu"]
    assert fc2aware_rows and "fc2_aware_gelu" in fc2aware_rows[0]
    output_aware_rows = [row for row in report["rows"] if row["name"] == "output_aware_gelu_8_to_11"]
    assert output_aware_rows and "output_aware_gelu" in output_aware_rows[0]
    output_aware_mlp_rows = [row for row in report["rows"] if row["name"] == "output_aware_mlp_lm_head_0_11_pc_full_bc"]
    assert output_aware_mlp_rows and "output_aware_mlp" in output_aware_mlp_rows[0]
    output_aware_attn_rows = [row for row in report["rows"] if row["name"] == "output_aware_attn_all"]
    assert output_aware_attn_rows and "output_aware_attn" in output_aware_attn_rows[0]
    lm_head_rows = [row for row in report["rows"] if row["name"] == "output_aware_mlp_lm_head_0_11_pc_full_bc"]
    assert lm_head_rows and "output_aware_lm_head" in lm_head_rows[0]
