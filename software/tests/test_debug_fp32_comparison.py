"""Non-gating diagnostics for golden-vs-FP32 nanoGPT comparison."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "debug_fp32_comparison.py"
FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "generated"
    / "nanogpt_shakespeare_char_d128_l2_trained.pt"
)


def _run_debug(*args):
    if not FIXTURE.exists():
        pytest.skip(
            "trained d128 nanoGPT fixture is not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py --trained-d128"
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--fixture",
            str(FIXTURE),
            *args,
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def test_debug_fp32_comparison_outputs_json():
    data = _run_debug("--mode", "free_running", "--prompt-id", "0", "--max-new-tokens", "1")

    assert data["mode"] == "free_running"
    assert data["vocab_size"] > 0
    assert data["lm_head_scale"] > 0.0
    assert len(data["steps"]) == 2
    assert "golden_vs_fp32" in data["bridge_summary"]
    assert "golden" in data["generated"]
    assert "fp32" in data["generated"]


def test_teacher_forced_comparison_is_deterministic():
    args = ("--mode", "teacher_forced_eval", "--max-new-tokens", "2")
    first = _run_debug(*args)
    second = _run_debug(*args)

    assert first["generated"] == second["generated"]
    assert first["summary"] == second["summary"]
    assert first["bridge_summary"] == second["bridge_summary"]


def test_lm_head_quantization_metrics_include_saturation_and_unique_counts():
    data = _run_debug(
        "--mode",
        "same_prefix_golden",
        "--prompt-id",
        "19",
        "--max-new-tokens",
        "2",
        "--node",
        "lm_head",
    )

    first_step = data["steps"][0]
    assert "unique_count" in first_step["golden_int8"]
    assert "saturation_rate" in first_step["golden_int8"]
    assert "scalar_current" in data["logit_quantization_experiments"]
    assert "centered_current" in data["logit_quantization_experiments"]
    assert "per_token_oracle" in data["logit_quantization_experiments"]
    assert "lm_head" in data["trace"]
    assert "lm_head_current_int8" in data["trace"]["lm_head"]


def test_incremental_node_trace_has_matching_fake_and_fp32_nodes():
    data = _run_debug(
        "--mode",
        "same_prefix_golden",
        "--prompt-id",
        "19",
        "--max-new-tokens",
        "1",
        "--trace-node",
        "all",
        "--step",
        "0",
    )

    trace = data["trace"]
    assert trace["step"] == 0
    assert "tok_pos_add" in trace["node_metrics"]
    assert "ln_f" in trace["node_metrics"]
    assert "lm_head" in trace["node_metrics"]
    assert trace["node_metrics"]["tok_pos_add"]["shape_match"] is True


def test_first_divergence_report_is_deterministic():
    args = (
        "--mode",
        "same_prefix_golden",
        "--prompt-id",
        "19",
        "--max-new-tokens",
        "1",
        "--trace-node",
        "all",
        "--step",
        "0",
        "--first-divergence",
    )
    first = _run_debug(*args)
    second = _run_debug(*args)

    assert first["trace"]["first_divergence"] == second["trace"]["first_divergence"]
    assert first["trace"]["first_divergence"]["node"]


def test_quantization_ablation_outputs_rank_metrics():
    data = _run_debug("--mode", "teacher_forced_eval", "--max-new-tokens", "1")

    ablations = data["ablation_experiments"]
    assert "residual_vadd" in ablations
    assert "mlp" in ablations
    assert "lm_head" in ablations
    assert "min_top10_overlap" in ablations["residual_vadd"]
    assert "mean_centered_cosine" in ablations["lm_head"]


def test_calibration_sweep_reports_scale_and_saturation_metrics():
    data = _run_debug(
        "--mode",
        "teacher_forced_eval",
        "--max-new-tokens",
        "0",
        "--calibration-sweep",
    )

    sweep = data["calibration_sweep"]
    overrides = data["targeted_scale_override_sweep"]
    assert len(sweep) == 16
    assert {"seq_len", "percentile", "lm_head_scale", "mean_saturation_rate"}.issubset(sweep[0])
    assert any(row["target"] == "residual" for row in overrides)
    assert any(row["target"] == "lm_head" for row in overrides)
