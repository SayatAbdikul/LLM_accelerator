"""Negative controls for conformance and benchmark promotion gates."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import shutil
import os
import subprocess
import sys

import numpy as np
import pytest

from tools import rtl_cosim
from tools.benchmark_support import benchmark_manifest, bundle_ptq_transform_manifest, check_runner_freshness, decoder_logits_metrics, logical_decoder_work, sha256_file, validate_runner_summary
from tools.check_reference_manifest import MANIFEST, REPO_ROOT, current_hashes, verify_reference_manifest
from tools.fast_gate_b16 import compare_report


def test_fp16_special_values_obey_native_sfu_contract():
    a = np.array([0x7e00, 0x7e00, 0x7c00, 0x7bff, 0x8000, 0x8001], dtype=np.uint16)
    b = np.array([0x7e00, 0x7e01, 0x7bff, 0x7bfe, 0x0000, 0x0001], dtype=np.uint16)
    ulp = rtl_cosim._fp16_ulp(a, b)
    assert ulp[0] == 0
    assert ulp[1] > 3 and ulp[2] > 3  # NaN payload and infinity require exact bits
    assert list(ulp[3:]) == [1, 0, 2]


def test_logits_quality_cannot_hide_nonfinite_steps_or_missing_outputs():
    clean = [np.array([1.0, 2.0]), np.array([2.0, 1.0])]
    bad = [clean[0], np.array([np.nan, 1.0])]
    result = rtl_cosim.logits_metric(clean, bad, [1, 0], 2)
    assert not result["quality_eligible"]
    assert result["n_nonfinite_rtl"] == 1
    assert result["finite_step_count_rtl"] == 1
    with pytest.raises(ValueError, match="step counts"):
        rtl_cosim.logits_metric(clean, bad[:1], [1, 0], 2)
    with pytest.raises(ValueError, match="non-empty"):
        rtl_cosim.logits_metric([], [], [], 2)


def test_hardware_gate_rejects_mixed_or_unidentified_runners():
    summary = {"status": "halted", "fault": False, "cycles": 1,
               "configuration": {"sfu_synth_mode": 1, "helper_synth_mode": 0}}
    with pytest.raises(ValueError, match="HELPER_SYNTH_MODE=1"):
        validate_runner_summary(summary)
    summary["configuration"]["helper_synth_mode"] = 1
    validate_runner_summary(summary)
    summary["port_a"] = {"sys_lost": 1}
    with pytest.raises(ValueError, match="lost SRAM"):
        validate_runner_summary(summary)


def _hardware_summary():
    return {"status": "halted", "fault": False, "cycles": 100,
            "configuration": {"sfu_synth_mode": 1, "helper_synth_mode": 1},
            "violations": [], "busy_cycles": {"systolic": 40},
            "systolic_phases": {"load_cycles": 10, "data_cycles": 15,
                                "k_boundary_cycles": 3, "flush_cycles": 5,
                                "drain_cycles": 4, "other_cycles": 3, "compute_steps": 20},
            "dma": {"beat_count": 5, "read_beat_count": 3, "write_beat_count": 2,
                    "burst_count": 2, "read_burst_count": 1, "write_burst_count": 1,
                    "accepted_bytes": 80, "read_bytes": 48, "write_bytes": 32}}


def test_hardware_summary_counters_reconcile():
    validate_runner_summary(_hardware_summary())


def _decoder_geometry():
    # Deliberately non-tile-aligned and not MLP=4D: this is logical work.
    return dict(batch=2, layers=3, d_model=5, mlp_dim=7, vocab_size=11, position=4)


def test_logical_decoder_work_counts_canonical_geometry_not_padded_work():
    result = logical_decoder_work(**_decoder_geometry(), summary=_hardware_summary())
    assert result["estimated_useful_macs"] == 1430
    assert result["component_macs"] == {
        "qkv_and_output_projections": 600, "mlp_projections": 420,
        "attention_qk_and_av": 300, "vocabulary_projection": 110,
    }
    assert result["geometry"] == _decoder_geometry()
    assert result["kind"] == "canonical_decoder_algorithm_estimate"
    assert result["measured_isa_op_count"] is False
    assert result["excludes"] == ["PTQ_transforms", "SFU_operations", "helper_operations", "padding"]
    ratios = result["capacity_ratios"]
    assert ratios["enabled_compute_steps"]["counter"] == 20
    assert ratios["enabled_compute_steps"]["value"] == pytest.approx(1430 / (256 * 20))
    assert ratios["total_cycles"]["value"] == pytest.approx(1430 / (256 * 100))
    first_position = dict(_decoder_geometry(), position=0)
    assert logical_decoder_work(**first_position)["component_macs"]["attention_qk_and_av"] == 60


@pytest.mark.parametrize("field,value", [
    ("batch", 0), ("layers", 0), ("d_model", 0), ("mlp_dim", 0), ("vocab_size", 0),
    ("batch", -1), ("layers", -1), ("d_model", -1), ("mlp_dim", -1), ("vocab_size", -1),
    ("position", -1), ("batch", True), ("d_model", 5.5), ("position", "4"),
])
def test_logical_decoder_work_rejects_invalid_geometry(field, value):
    geometry = dict(_decoder_geometry(), **{field: value})
    with pytest.raises(ValueError, match=field):
        logical_decoder_work(**geometry)


def test_logical_decoder_work_does_not_invent_missing_or_zero_counters():
    geometry = _decoder_geometry()
    assert logical_decoder_work(**geometry)["capacity_ratios"] == {}
    old_summary = {"cycles": 100, "busy_cycles": {"systolic": 40}}
    ratios = logical_decoder_work(**geometry, summary=old_summary)["capacity_ratios"]
    assert set(ratios) == {"total_cycles"}
    zero = {"cycles": 0, "systolic_phases": {"compute_steps": 0}}
    assert logical_decoder_work(**geometry, summary=zero)["capacity_ratios"] == {}
    with pytest.raises(ValueError, match="enabled_compute_steps"):
        logical_decoder_work(**geometry, summary={"systolic_phases": {"compute_steps": -1}})


@pytest.mark.parametrize("include_estimate", [False, True])
def test_decode_runner_adds_logical_work_without_rewriting_old_manifests(tmp_path, monkeypatch, include_estimate):
    from tools.bench_decode_cycles import run_one
    from types import SimpleNamespace

    binary = tmp_path / "decode.bin"
    layout_path = binary.with_suffix(".manifest.json")
    layout = dict(schema_version=1, logits_offset=0, logits_size=4,
                  batch=1, vocab_size=2, logits_row_bytes=4)
    if include_estimate:
        layout["schema_version"] = 2
        layout["logical_decoder_work"] = logical_decoder_work(
            **dict(_decoder_geometry(), batch=1, vocab_size=2))
    raw_layout = json.dumps(layout)
    layout_path.write_text(raw_layout)

    def fake_run(command, **kwargs):
        Path(command[command.index("--json-out") + 1]).write_text(json.dumps(_hardware_summary()))
        Path(command[command.index("--dram-dump-out") + 1]).write_bytes(np.array([1, 2], dtype="<f2").tobytes())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    summary = run_one(tmp_path / "runner", binary, 1000)
    assert layout_path.read_text() == raw_layout
    assert summary["program_manifest"] == layout
    assert ("logical_decoder_work" in summary) is include_estimate
    if include_estimate:
        assert set(summary["logical_decoder_work"]["capacity_ratios"]) == {"enabled_compute_steps", "total_cycles"}
        assert layout["logical_decoder_work"]["capacity_ratios"] == {}


@pytest.mark.parametrize("section,key,value,expected", [
    ("systolic_phases", "data_cycles", 14, "phase counters"),
    ("systolic_phases", "compute_steps", 19, "compute-step"),
    ("systolic_phases", "load_cycles", -1, "negative"),
    ("dma", "read_beat_count", 4, "direction counter"),
    ("dma", "write_burst_count", 2, "direction counter"),
    ("dma", "write_bytes", 16, "byte counter"),
])
def test_hardware_summary_rejects_inconsistent_counters(section, key, value, expected):
    summary = _hardware_summary()
    summary[section][key] = value
    with pytest.raises(ValueError, match=expected):
        validate_runner_summary(summary)


@pytest.mark.parametrize("key,value", [("violations", ["systolic_phase_count_mismatch"]),
                                       ("timeout", True), ("parse_error", "bad program")])
def test_hardware_summary_rejects_diagnostics_even_when_halted(key, value):
    summary = _hardware_summary()
    summary[key] = value
    with pytest.raises(ValueError, match="violations"):
        validate_runner_summary(summary)


def test_decoder_logits_guard_reads_fp16_and_excludes_padding():
    raw = np.array([[1, 2, np.nan, 0], [3, 4, np.inf, 0]], dtype="<f2").tobytes()
    result = decoder_logits_metrics(raw, rows=2, vocab_size=2, row_bytes=8)
    assert result["all_finite"] and result["element_count"] == 4
    assert result["argmax"] == [1, 1]
    assert not decoder_logits_metrics(raw, rows=2, vocab_size=3, row_bytes=8)["all_finite"]
    with pytest.raises(ValueError, match="layout"):
        decoder_logits_metrics(raw[:-1], rows=2, vocab_size=2, row_bytes=8)


def test_ptq_manifest_distinguishes_requested_rotation_from_applied_transform():
    manifest = bundle_ptq_transform_manifest("weight_only_int8_quarot")
    assert manifest["requested_preset"] == "weight_only_int8_quarot"
    assert manifest["quarot"] == {"requested_by_preset": True,
                                  "applied_in_bundle_build": False,
                                  "input_checkpoint_transform_history": "unverified"}
    assert bundle_ptq_transform_manifest("control")["quarot"]["requested_by_preset"] is False
    assert bundle_ptq_transform_manifest(None)["quarot"]["requested_by_preset"] is None


def test_legacy_golden_ppl_reads_fp16_storage_and_returns_unscaled_fp32(monkeypatch):
    from types import SimpleNamespace
    from taccel.runtime import gpt2_perplexity as ppl

    marker = object()
    monkeypatch.setattr(ppl, "build_stage3_tiny_decoder_bundle",
                        lambda *args, **kwargs: SimpleNamespace(build=SimpleNamespace(bundle=marker)))

    class FakeRunner:
        def __init__(self, bundle, *, logits_dtype):
            assert bundle is marker and logits_dtype is np.float16

        def run_prefill(self, tokens):
            assert tokens == [0]
            return np.array([1.5, -0.125], dtype=np.float16)

        def run_decode_step(self, token, position):
            assert (token, position) == (1, 1)
            return np.array([0.25, -2.25], dtype=np.float16)

    monkeypatch.setattr(ppl, "HostRunner", FakeRunner)
    result = ppl.run_golden_teacher_forced_logits({}, [0, 1, 2], {"lm_head": 0.125})
    assert all(row.dtype == np.float32 for row in result)
    np.testing.assert_array_equal(result, [[1.5, -0.125], [0.25, -2.25]])


def test_legacy_ppl_applies_output_scale_only_to_int8_research_reference(monkeypatch):
    from taccel.runtime import gpt2_perplexity as ppl

    monkeypatch.setattr(ppl, "build_calibration_scales_from_token_ids", lambda *a, **k: {"lm_head": 0.25})
    monkeypatch.setattr(ppl, "apply_stage5_ptq_scale_policy", lambda scales, *a, **k: scales)
    monkeypatch.setattr(ppl, "run_golden_teacher_forced_logits", lambda *a, **k: [np.array([0.25, 0], dtype=np.float32)])
    monkeypatch.setattr(ppl, "run_fake_quant_teacher_forced_logits", lambda *a, **k: [np.array([1, 0], dtype=np.int8)])
    result = ppl.evaluate_gpt2_perplexity(
        {"model_args": {"vocab_size": 2}}, calibration_token_ids=[0, 1],
        eval_token_ids=[0, 1], tokenizer_dir=Path("unused"), ptq_preset="control",
        compute_fp32_ceiling=False)
    assert result.golden_nll == pytest.approx(ppl.stable_cross_entropy(np.array([0.25, 0]), 1, vocab_size=2))
    assert result.relative_delta == 0


def test_manifest_checks_paired_checkpoint_and_labels_read_only_pacing(tmp_path):
    fixture, runner = tmp_path / "model.pt", tmp_path / "runner"
    fixture.write_bytes(b"model")
    runner.write_bytes(b"runner")
    metadata = fixture.with_suffix(".pt.json")
    metadata.write_text(json.dumps({"checkpoint_sha256": sha256_file(fixture)}))
    manifest = benchmark_manifest(REPO_ROOT, fixture, runner, batch=1, positions=[0],
                                  beat_interval=4, fmax_mhz=None, compiler_options={})
    assert manifest["fixture"]["metadata_sha256"] == sha256_file(metadata)
    assert manifest["memory"]["core_cycles_per_read_beat"] == 4
    assert manifest["memory"]["write_pacing"] == "unthrottled_by_beat_interval"
    assert manifest["clock_mhz"] is None and manifest["clock_kind"] == "unmeasured"
    assert manifest["schema_version"] == 2
    assert manifest["ptq_transforms"]["quarot"]["applied_in_bundle_build"] is False
    fixture.write_bytes(b"different model")
    with pytest.raises(ValueError, match="paired metadata"):
        benchmark_manifest(REPO_ROOT, fixture, runner, batch=1, positions=[0],
                           beat_interval=1, fmax_mhz=None, compiler_options={})


def test_full_reference_pin_detects_helper_and_fixture_corruption(tmp_path):
    # A copied reference tree permits a real corruption check without touching
    # the working golden model or reusing its expected hash as actual evidence.
    manifest = json.loads(MANIFEST.read_text())
    for rel in manifest["sha256"]:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, dst)
    verify_reference_manifest(tmp_path, MANIFEST)
    helper = tmp_path / "software/taccel/golden_model/memory.py"
    helper.write_bytes(helper.read_bytes() + b"\n# corruption control\n")
    with pytest.raises(ValueError, match="memory.py"):
        verify_reference_manifest(tmp_path, MANIFEST)
    raw = next(tmp_path.glob("rtl/verilator/fixtures/gen2/**/expected_out.raw"))
    raw.write_bytes(b"bad")
    with pytest.raises(ValueError, match="expected_out.raw"):
        verify_reference_manifest(tmp_path, MANIFEST)


def test_cosim_default_runner_is_hardware():
    assert rtl_cosim.RTL_BINARY.parent.name == "run_program_synth"
    assert rtl_cosim.RTL_REFERENCE_BINARY.parent.name == "run_program"


@pytest.mark.parametrize("name", ["test_run_gpt2", "test_run_nanogpt"])
def test_cli_tests_default_to_current_python_with_explicit_override(name, monkeypatch):
    def load():
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"software/tests/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.delenv("TACCEL_TEST_RUNNER_PYTHON", raising=False)
    assert load()._RUNNER_PYTHON == sys.executable
    monkeypatch.setenv("TACCEL_TEST_RUNNER_PYTHON", "/explicit/runner/python")
    assert load()._RUNNER_PYTHON == "/explicit/runner/python"


def test_runner_freshness_tracks_its_translation_unit_not_unrelated_benches(tmp_path):
    sources = tmp_path / "rtl/verilator"
    sources.mkdir(parents=True)
    runner, source, unrelated = sources / "runner", sources / "run_program.cpp", sources / "test_dma.cpp"
    for path in (runner, source, unrelated):
        path.write_bytes(b"test")
    os.utime(source, (100, 100))
    os.utime(runner, (200, 200))
    os.utime(unrelated, (300, 300))
    check_runner_freshness(tmp_path, runner)
    os.utime(source, (300, 300))
    with pytest.raises(ValueError, match="run_program.cpp"):
        check_runner_freshness(tmp_path, runner)


def test_matched_performance_gate_rejects_input_and_output_mismatches():
    baseline = {"manifest": {"batch": 1, "positions": [0], "memory": {"beat": 1},
                            "workload_kind": "shape_only_zero_kv", "fixture": {"sha256": "model"}},
                "rows": [{"cycles": 100, "program_manifest": {"calibration_sha256": "scales"},
                          "configuration": {"sfu_synth_mode": 1, "helper_synth_mode": 1},
                          "decoder_logits": {"sha256": "out", "all_finite": True}}]}
    candidate = json.loads(json.dumps(baseline))
    candidate["rows"][0]["cycles"] = 80
    assert compare_report(candidate, baseline, max_regression_percent=0) == []
    candidate["manifest"]["fixture"]["sha256"] = "other"
    candidate["rows"][0]["decoder_logits"]["sha256"] = "changed"
    failures = compare_report(candidate, baseline, max_regression_percent=0)
    assert len(failures) == 2
    candidate = json.loads(json.dumps(baseline))
    candidate["rows"][0]["cycles"] = 101
    assert "cycle regression" in compare_report(candidate, baseline, max_regression_percent=0)[0]
    assert compare_report(candidate, baseline, max_regression_percent=1) == []
    candidate["rows"][0]["decoder_logits"]["all_finite"] = False
    assert "requires finite" in compare_report(candidate, baseline, max_regression_percent=1)[0]


def test_required_profile_turns_missing_input_skip_into_failure(tmp_path):
    # Launch an isolated pytest run with the real profile hooks and a synthetic
    # mandatory missing-input test; no hardware or model download is needed.
    shutil.copyfile(REPO_ROOT / "software/tests/conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_host_runner.py").write_text(
        "import pytest\ndef test_missing_input():\n    pytest.skip('fixture absent')\n")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "software"))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tmp_path),
                          "--confcutdir", str(tmp_path), "--verification-profile", "required"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Mandatory verification gate skipped" in run.stdout


def test_shape_benchmark_writes_logits_layout_and_compiler_metrics(tmp_path):
    from tools.train_tiny_fixture import _make_checkpoint_payload
    from tools.bench_decode_cycles import build_decode_bins
    from taccel.compiler.options import CompilerOptions
    payload = _make_checkpoint_payload()
    bins = build_decode_bins(payload, [0], tmp_path, compiler_options=CompilerOptions(indexed_liveness=True))
    manifest = json.loads(bins[0].with_suffix(".manifest.json").read_text())
    assert manifest["logits_row_bytes"] >= int(payload["model_args"]["vocab_size"]) * 2
    assert manifest["logits_size"] == manifest["logits_row_bytes"]
    assert manifest["compiler_metrics"]["decode"]["instructions"] > 0
    assert manifest["schema_version"] == 2
    geometry = manifest["logical_decoder_work"]["geometry"]
    assert geometry == dict(batch=1, layers=int(payload["model_args"]["n_layer"]),
                            d_model=int(payload["model_args"]["n_embd"]),
                            mlp_dim=4 * int(payload["model_args"]["n_embd"]),
                            vocab_size=int(payload["model_args"]["vocab_size"]), position=0)
    assert manifest["logical_decoder_work"]["capacity_ratios"] == {}
    assert manifest["ptq_transforms"] == bundle_ptq_transform_manifest("weight_only_int8_quarot")
