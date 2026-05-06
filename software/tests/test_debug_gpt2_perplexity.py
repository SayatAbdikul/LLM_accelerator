"""Non-gating diagnostics for the Stage 5 GPT-2 perplexity gap."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from taccel.runtime.stage5_ptq import STAGE5_PTQ_PRESETS, stage5_default_ptq_preset_name


TOOL = Path(__file__).resolve().parents[1] / "tools" / "debug_gpt2_perplexity.py"
FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
CALIB_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_calibration.txt")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")


def _debug_module():
    spec = importlib.util.spec_from_file_location("debug_gpt2_perplexity", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_report():
    return {
        "suspects": {
            "converter_bias_layout": {
                "available": True,
                "min_cosine_vs_hf": 1.0,
                "max_p99_abs_error_vs_hf": 0.0,
                "min_top10_overlap_vs_hf": 10,
            },
            "fp32_baseline": {
                "incremental": {"perplexity": 50.0},
            },
            "lm_head_quantization": {
                "direct_quant_fp32": {"perplexity": 55.0},
            },
            "perplexity": {
                "fake_quant_perplexity": 60.0,
            },
            "shared_decode_semantics": {
                "fp32_full_vs_incremental": {"min_cosine": 1.0},
                "fake_full_vs_incremental": {"min_cosine": 1.0},
            },
        }
    }


def _subprocess_env():
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "software")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    return env


def _run_debug_cli(args):
    proc = subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=Path(__file__).resolve().parents[2],
        env=_subprocess_env(),
        text=True,
        capture_output=True,
    )
    if proc.returncode and "OMP: Error #179" in proc.stderr:
        pytest.skip(f"local OpenMP shared-memory blocker while running debug CLI: {proc.stderr.strip()}")
    proc.check_returncode()
    return proc


def test_primary_suspect_rules_are_deterministic():
    dbg = _debug_module()
    report = _base_report()
    assert dbg.choose_primary_suspect(report) == "no single dominant suspect"

    report = _base_report()
    report["suspects"]["converter_bias_layout"]["min_cosine_vs_hf"] = 0.8
    assert dbg.choose_primary_suspect(report) == "converter/bias/layout mismatch"

    report = _base_report()
    report["suspects"]["fp32_baseline"]["incremental"]["perplexity"] = 1.0e9
    assert dbg.choose_primary_suspect(report) == "eval slice/context issue"

    report = _base_report()
    report["suspects"]["lm_head_quantization"]["direct_quant_fp32"]["perplexity"] = 1.0e9
    assert dbg.choose_primary_suspect(report) == "lm_head quantization/scale issue"


def test_debug_gpt2_perplexity_outputs_json_sections():
    missing = [
        str(path)
        for path in (FIXTURE, TOKENIZER_DIR, CALIB_TEXT, EVAL_TEXT)
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"local GPT-2 debug inputs missing: {missing}")

    proc = _run_debug_cli(
        [
            str(FIXTURE),
            "--tokenizer-dir",
            str(TOKENIZER_DIR),
            "--calibration-text",
            str(CALIB_TEXT),
            "--eval-text",
            str(EVAL_TEXT),
            "--max-eval-tokens",
            "2",
            "--context-len",
            "1",
            "--calibration-n-seqs",
            "1",
            "--calibration-seq-len",
            "2",
            "--json",
        ]
    )
    data = json.loads(proc.stdout)
    assert "primary_suspect" in data
    assert data["ptq_preset"] == stage5_default_ptq_preset_name()
    assert {"fp32_baseline", "lm_head_quantization", "calibration_sensitivity", "shared_decode_semantics", "converter_bias_layout"}.issubset(data["suspects"])
    assert len(data["suspects"]["calibration_sensitivity"]) == 4
    assert len(data["per_step"]) == 1
    step = data["per_step"][0]
    assert "target_rank" in step
    assert "target_nll" in step
    assert "unique_count" in step["int8"]["direct_quant_fp32"]
    assert "saturation_rate" in step["int8"]["direct_quant_fp32"]


def test_debug_gpt2_perplexity_preset_sweep_json_sections():
    missing = [
        str(path)
        for path in (FIXTURE, TOKENIZER_DIR, CALIB_TEXT, EVAL_TEXT)
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"local GPT-2 debug inputs missing: {missing}")

    proc = _run_debug_cli(
        [
            str(FIXTURE),
            "--tokenizer-dir",
            str(TOKENIZER_DIR),
            "--calibration-text",
            str(CALIB_TEXT),
            "--eval-text",
            str(EVAL_TEXT),
            "--max-eval-tokens",
            "2",
            "--context-len",
            "1",
            "--calibration-n-seqs",
            "1",
            "--calibration-seq-len",
            "2",
            "--preset-sweep",
            "--json",
        ]
    )
    data = json.loads(proc.stdout)
    assert data["preset_sweep"]["promoted_default"] == stage5_default_ptq_preset_name()
    assert len(data["preset_sweep"]["rows"]) == len(STAGE5_PTQ_PRESETS)
    assert data["preset_sweep"]["rows"][0]["name"]
    assert data["preset_sweep"]["winner"]["name"]
    assert data["preset_sweep"]["default_replacement_candidate"]["baseline"] == stage5_default_ptq_preset_name()
    by_name = {row["name"]: row for row in data["preset_sweep"]["rows"]}
    assert "fc2_11_fc2aware_gelu" in by_name
    assert "fc2_aware_gelu" in by_name["fc2_11_fc2aware_gelu"]


def test_debug_gpt2_perplexity_compare_search_windows_report_sections(monkeypatch):
    dbg = _debug_module()

    class FakeTokenizer:
        def decode(self, ids):
            return f"tok{int(ids[0])}"

    def fake_base(*args, **kwargs):
        return {"lm_head": 1.0, "block0_gelu": 1.0}, {"base": True}

    def fake_apply(*args, search_n_seqs, search_seq_len, **kwargs):
        scale = 1.0 if search_n_seqs == 1 else 0.5
        return {"lm_head": scale, "block0_gelu": scale}, {"window": f"{search_n_seqs}x{search_seq_len}"}

    def fake_golden(*args, **kwargs):
        return [np.asarray([2, 1, 0, -1], dtype=np.int8)]

    def fake_fake(*args, **kwargs):
        return [np.asarray([2, 0, 1, -1], dtype=np.int8)]

    monkeypatch.setattr(dbg, "_base_scales_for_search_window_compare", fake_base)
    monkeypatch.setattr(dbg, "_apply_search_window", fake_apply)
    monkeypatch.setattr(dbg, "run_golden_teacher_forced_logits", fake_golden)
    monkeypatch.setattr(dbg, "run_fake_quant_teacher_forced_logits", fake_fake)

    comparison = dbg._compare_search_windows(
        payload={"model_args": {"vocab_size": 4}, "state_dict": {}},
        calibration_ids=[0, 1, 2],
        eval_tokens=[0, 1],
        targets=[1],
        tokenizer=FakeTokenizer(),
        vocab_size=4,
        fp32_logits=[np.asarray([4, 3, 2, 1], dtype=np.float32)],
        args=SimpleNamespace(),
        preset=SimpleNamespace(name="demo"),
    )

    assert comparison["preset"] == "demo"
    assert comparison["base_diagnostics"] == {"base": True}
    assert [row["label"] for row in comparison["windows"]] == ["1x16", "4x64"]
    assert comparison["scale_diffs_1x16_vs_4x64"]
    assert {"golden_improvement_4x64_vs_1x16", "fake_quant_improvement_4x64_vs_1x16"}.issubset(comparison)
    for row in comparison["windows"]:
        assert {"per_step", "first_per_step_divergence", "scale_diffs_vs_base"}.issubset(row)


def test_calibration_sweep_forwards_mlp_search_caps(monkeypatch):
    dbg = _debug_module()
    captured = []

    monkeypatch.setattr(dbg, "build_calibration_scales_from_token_ids", lambda *args, **kwargs: {"lm_head": 1.0})
    monkeypatch.setattr(dbg, "apply_stage5_ptq_scale_policy", lambda scales, *args, **kwargs: dict(scales))
    monkeypatch.setattr(dbg, "run_golden_teacher_forced_logits", lambda *args, **kwargs: [np.asarray([2, 1, 0, -1], dtype=np.int8)])
    monkeypatch.setattr(dbg, "run_fake_quant_teacher_forced_logits", lambda *args, **kwargs: [np.asarray([2, 1, 0, -1], dtype=np.int8)])

    def fake_mlp_search(*args, **kwargs):
        captured.append((kwargs.get("search_n_seqs_max"), kwargs.get("search_seq_len_max")))
        return dict(args[2]), {}

    monkeypatch.setattr(dbg, "apply_output_aware_mlp_scale_search_from_token_ids", fake_mlp_search)

    rows = dbg._calibration_sweep(
        payload={"model_args": {"n_layer": 12, "vocab_size": 4}, "state_dict": {}},
        calibration_ids=[0, 1, 2],
        eval_tokens=[0, 1],
        vocab_size=4,
        tokenizer_dir=Path("."),
        args=SimpleNamespace(output_aware_search_n_seqs=4, output_aware_search_seq_len=64),
        ptq_preset="output_aware_mlp_0_1_4_8_to_11",
    )

    assert len(rows) == len(dbg.SWEEP_CONFIGS)
    assert captured == [(4, 64)] * len(dbg.SWEEP_CONFIGS)


def test_calibration_sweep_gelu_search_uses_supported_kwargs(monkeypatch):
    dbg = _debug_module()
    captured = []

    monkeypatch.setattr(dbg, "build_calibration_scales_from_token_ids", lambda *args, **kwargs: {"lm_head": 1.0})
    monkeypatch.setattr(dbg, "apply_stage5_ptq_scale_policy", lambda scales, *args, **kwargs: dict(scales))
    monkeypatch.setattr(dbg, "run_golden_teacher_forced_logits", lambda *args, **kwargs: [np.asarray([2, 1, 0, -1], dtype=np.int8)])
    monkeypatch.setattr(dbg, "run_fake_quant_teacher_forced_logits", lambda *args, **kwargs: [np.asarray([2, 1, 0, -1], dtype=np.int8)])

    def fake_gelu_search(*args, **kwargs):
        assert "include_pair_candidates" not in kwargs
        captured.append((kwargs.get("search_n_seqs_max"), kwargs.get("search_seq_len_max")))
        return dict(args[2]), {}

    monkeypatch.setattr(dbg, "apply_output_aware_gelu_scale_search_from_token_ids", fake_gelu_search)

    rows = dbg._calibration_sweep(
        payload={"model_args": {"n_layer": 12, "vocab_size": 4}, "state_dict": {}},
        calibration_ids=[0, 1, 2],
        eval_tokens=[0, 1],
        vocab_size=4,
        tokenizer_dir=Path("."),
        args=SimpleNamespace(output_aware_search_n_seqs=4, output_aware_search_seq_len=64),
        ptq_preset="output_aware_gelu_8_to_11",
    )

    assert len(rows) == len(dbg.SWEEP_CONFIGS)
    assert captured == [(4, 64)] * len(dbg.SWEEP_CONFIGS)


def test_mlp_ablation_flags_report_diagnostic_rows():
    missing = [
        str(path)
        for path in (FIXTURE, TOKENIZER_DIR, CALIB_TEXT, EVAL_TEXT)
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"local GPT-2 debug inputs missing: {missing}")

    proc = _run_debug_cli(
        [
            str(FIXTURE),
            "--tokenizer-dir",
            str(TOKENIZER_DIR),
            "--calibration-text",
            str(CALIB_TEXT),
            "--eval-text",
            str(EVAL_TEXT),
            "--max-eval-tokens",
            "2",
            "--context-len",
            "1",
            "--calibration-n-seqs",
            "1",
            "--calibration-seq-len",
            "2",
            "--mlp-subablation",
            "--mlp-block-ablation",
            "--json",
        ]
    )
    data = json.loads(proc.stdout)
    subgroups = {row["group"] for row in data["mlp_subablation"]}
    assert {"mlp_fc1", "mlp_gelu", "mlp_fc2", "mlp_full"}.issubset(subgroups)
    assert len(data["mlp_block_ablation"]) == 12
    assert {"block", "nll_improvement_vs_baseline", "mean_top10_overlap_vs_fp32"}.issubset(
        data["mlp_block_ablation"][0]
    )


def test_gpt2_node_trace_reports_first_divergence():
    dbg = _debug_module()
    fake_trace = {
        "tok_pos_add": {
            "value": [[1.0, 2.0]],
            "scale": 0.5,
            "int8": [[2, 4]],
        },
        "block0_ln1": {
            "value": [[10.0, -10.0]],
            "scale": 0.25,
            "int8": [[40, -40]],
        },
        "ln_f": {
            "value": [[0.0, 1.0]],
            "scale": 0.25,
            "int8": [[0, 4]],
        },
        "lm_head": {
            "value": [[4.0, 1.0, 0.0]],
            "scale": 0.25,
            "int8": [[16, 4, 0]],
        },
    }
    fp32_trace = {
        "tok_pos_add": {"value": [[1.0, 2.0]]},
        "block0_ln1": {"value": [[-10.0, 10.0]]},
        "ln_f": {"value": [[0.0, 1.0]]},
        "lm_head": {"value": [[4.0, 1.0, 0.0]]},
    }
    report = dbg._trace_report(
        fake_trace=fake_trace,
        fp32_trace=fp32_trace,
        model_args={"n_layer": 1, "n_head": 0},
        target=0,
        vocab_size=3,
        trace_node="all",
        include_first_divergence=True,
    )

    assert report["first_divergence"]["node"] == "block0_ln1"
    assert "centered_cosine<0.95" in report["first_divergence"]["reasons"]
    first_node = report["nodes"][0]
    assert first_node["node"] == "tok_pos_add"
    assert "unique_count" in first_node["int8"]
    assert "saturation_rate" in first_node["int8"]


def test_gpt2_trace_metrics_include_scale_unique_and_saturation():
    dbg = _debug_module()
    metrics = dbg._trace_node_metrics(
        "lm_head",
        {
            "value": [[3.0, 2.0, 1.0, 0.0]],
            "scale": 0.5,
            "int8": [[127, 4, 2, -128]],
        },
        {"value": [[3.0, 1.5, 1.0, 0.0]]},
        target=1,
        vocab_size=4,
    )

    assert metrics["scale"] == 0.5
    assert metrics["p99_error_lsb"] is not None
    assert metrics["int8"]["unique_count"] == 4
    assert metrics["int8"]["saturation_rate"] == 0.5
    assert metrics["top10_overlap"] == 4
    assert metrics["target_rank"]["fake_quant"] == 2


def test_gpt2_ablation_sweep_reports_all_groups():
    dbg = _debug_module()
    rows = [
        {
            "group": group,
            "nll_improvement_vs_baseline": float(idx),
            "min_top10_overlap_vs_fp32": idx % 10,
            "min_cosine_vs_fp32": 0.5 + idx * 0.01,
        }
        for idx, group in enumerate(dbg.ABLATION_GROUPS)
    ]
    best = dbg._best_ablation(rows)

    assert [row["group"] for row in rows] == list(dbg.ABLATION_GROUPS)
    assert best["group"] == dbg.ABLATION_GROUPS[-1]


def test_attn_v_head_ablation_reports_every_head(monkeypatch):
    dbg = _debug_module()

    def fake_rows_for_groups(**kwargs):
        assert kwargs["label_key"] == "head_group"
        return [
            {
                "head_group": group,
                "perplexity": 10.0,
                "mean_nll": 1.0,
                "nll_improvement_vs_baseline": float(idx),
                "min_top10_overlap_vs_fp32": 7,
                "mean_top10_overlap_vs_fp32": 8.0,
                "min_cosine_vs_fp32": 0.9,
                "mean_cosine_vs_fp32": 0.95,
            }
            for idx, group in enumerate(kwargs["group_names"])
        ]

    monkeypatch.setattr(dbg, "_ablation_rows_for_groups", fake_rows_for_groups)
    rows = dbg._attn_v_head_ablation_sweep(
        payload={"model_args": {"n_layer": 12, "n_head": 12}},
        scales={},
        inputs=[0],
        targets=[1],
        fp32_logits=[],
        fake_i8=[],
        vocab_size=4,
        lm_scale=1.0,
        resolved_preset=None,
    )

    assert len(rows) == 144
    assert rows[0]["head_group"] == "attn_v_head_0_0"
    assert rows[0]["block"] == 0
    assert rows[0]["head"] == 0
    assert rows[-1]["head_group"] == "attn_v_head_11_11"
    assert rows[-1]["block"] == 11
    assert rows[-1]["head"] == 11


def test_attn_v_structural_sweep_reports_component_bypasses(monkeypatch):
    dbg = _debug_module()

    class FakeReference:
        def __init__(self, *args, **kwargs):
            pass

        def incremental_logits_trace(self, inputs, fp32_groups=None):
            groups = set(fp32_groups or ())
            target_logit = 2.0 + float(len(groups))
            if "attn_v_value_fp32" in groups:
                target_logit += 2.0
            return [np.asarray([target_logit, 1.0, 0.0, -1.0], dtype=np.float32)]

    monkeypatch.setattr(dbg, "NanoGPTFQReference", FakeReference)
    rows = dbg._attn_v_structural_sweep(
        payload={"state_dict": {}, "model_args": {"n_layer": 1, "n_head": 1}},
        scales={"lm_head": 1.0},
        inputs=[0],
        targets=[0],
        fp32_logits=[np.asarray([8.0, 1.0, 0.0, -1.0], dtype=np.float32)],
        fake_i8=[np.asarray([1.0, 1.0, 0.0, -1.0], dtype=np.float32)],
        vocab_size=4,
        lm_scale=1.0,
        resolved_preset=None,
    )

    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {
        "int8_product_bypass",
        "softmax_product_bypass",
        "value_product_bypass",
        "softmax_value_product_bypass",
    }
    assert by_name["softmax_value_product_bypass"]["fp32_groups"] == [
        "softmax",
        "attn_v_value_fp32",
    ]
    assert rows[0]["name"] == "softmax_value_product_bypass"
    assert rows[0]["nll_improvement_vs_baseline"] > 0.0


def test_embedding_add_diagnostics_report_before_and_active_scale():
    dbg = _debug_module()
    payload = {
        "state_dict": {
            "transformer.wte.weight": [[1.0, -1.0], [0.5, -0.5]],
            "transformer.wpe.weight": [[-1.0, 1.0], [-1.0, 1.0]],
        }
    }

    report = dbg._embedding_add_scale_diagnostics(
        payload,
        [0, 1],
        active_scale=2.0 / 127.0,
        percentile=100.0,
    )

    assert report["raw_vadd_safe_scale"] >= report["legacy_sum_scale"]
    assert "centered_cosine" in report["legacy"]
    assert "saturation_rate" in report["active"]["int8"]


def test_lm_head_structural_groups_constant_has_three_entries():
    dbg = _debug_module()
    assert len(dbg.LM_HEAD_STRUCTURAL_GROUPS) == 3
    names = {g["name"] for g in dbg.LM_HEAD_STRUCTURAL_GROUPS}
    assert {"weight_fp32_output_int8", "weight_int8_output_fp32",
            "weight_int8_requant_fp32_output_int8"}.issubset(names)
    for g in dbg.LM_HEAD_STRUCTURAL_GROUPS:
        assert "fp32_groups" in g
        assert "description" in g


def test_lm_head_structural_sweep_reports_component_bypasses(monkeypatch):
    dbg = _debug_module()

    class FakeReference:
        def __init__(self, *args, **kwargs):
            pass

        def incremental_logits_trace(self, inputs, fp32_groups=None):
            groups = set(fp32_groups or ())
            if "lm_head_weight_fp32" in groups:
                target_logit = 3.0
            elif "lm_head_output_fp32" in groups:
                target_logit = 4.0
            else:
                target_logit = 2.0
            return [np.asarray([target_logit, 1.0, 0.0, -1.0], dtype=np.float32)]

    monkeypatch.setattr(dbg, "NanoGPTFQReference", FakeReference)
    rows = dbg._lm_head_structural_sweep(
        payload={"state_dict": {}, "model_args": {"n_layer": 1, "n_head": 1}},
        scales={"lm_head": 1.0},
        inputs=[0],
        targets=[0],
        fp32_logits=[np.asarray([8.0, 1.0, 0.0, -1.0], dtype=np.float32)],
        fake_i8=[np.asarray([1.0, 1.0, 0.0, -1.0], dtype=np.float32)],
        vocab_size=4,
        lm_scale=1.0,
        resolved_preset=None,
    )

    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {
        "weight_fp32_output_int8",
        "weight_int8_requant_fp32_output_int8",
        "weight_int8_output_fp32",
    }
    for row in rows:
        assert "perplexity" in row
        assert "nll_improvement_vs_baseline" in row
        assert "description" in row
