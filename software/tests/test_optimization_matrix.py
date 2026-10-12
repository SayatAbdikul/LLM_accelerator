"""Matrix provenance, contiguous KV semantics and strict comparison tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools import optimization_matrix as matrix


def test_workloads_separate_shape_cases_from_contiguous_chunk_sequences():
    steps = matrix.workload_steps(1, 16, [0, 15], 2, 2, 128)
    assert [(s.stream, s.position, s.rows) for s in steps] == [
        ("prefill", 0, 16), ("prefill", 16, 16), ("decode", 32, 1), ("decode", 33, 1)]
    steps = matrix.workload_steps(16, 1, [0, 15], 2, 2, 128)
    assert [s.sequence for s in steps] == [False, False, True, True, True]
    assert [s.position for s in steps[-3:]] == [0, 1, 2]
    assert all(s.rows == 16 for s in steps)
    with pytest.raises(ValueError, match="context"):
        matrix.workload_steps(1, 128, [], 1, 1, 128)
    with pytest.raises(ValueError, match="single-stream"):
        matrix.workload_steps(16, 16, [], 1, 1, 128)


@pytest.mark.parametrize("batch,prefill,stream,stored_rows,source_rows", [
    (16, 1, "prefill", 1, [15]), (32, 1, "prefill", 1, [31]),
    (16, 1, "decode", 16, list(range(16))),
    (32, 1, "decode", 32, list(range(32))),
    (1, 16, "prefill", 16, list(range(16))), (1, 16, "decode", 1, [0]),
])
def test_program_layout_distinguishes_executed_tokens_from_stored_logits(
        batch, prefill, stream, stored_rows, source_rows):
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
    from tools.train_tiny_fixture import _make_checkpoint_payload
    payload = _make_checkpoint_payload(n_layer=1, n_head=4, d_model=64, block_size=64,
                                       weight_scale=.02)
    tiny = build_stage3_tiny_decoder_bundle(payload, batch=batch, prefill_tokens=prefill,
                                           smoke_decode_steps=max(2, prefill), calibration_scales={})
    token_rows = prefill if batch == 1 and stream == "prefill" else batch
    step = matrix.Step("test", stream, 0 if stream == "prefill" else prefill, token_rows, True)
    program, layout = matrix.build_program(tiny, step)
    assert layout["rows"] == stored_rows
    assert layout["logical_source_rows"] == source_rows
    assert layout["token_rows"] == token_rows == step.rows
    assert len(layout["tokens"]) == token_rows
    # Metadata must reflect the actual final STORE, not just our expectation.
    from taccel.isa.encoding import decode
    from taccel.isa.opcodes import Opcode
    stores = [decode(program.instructions[i:i + 8]) for i in range(0, len(program.instructions), 8)
              if decode(program.instructions[i:i + 8]).opcode == Opcode.STORE]
    assert stores[-1].xfer_len * 16 == stored_rows * layout["logits_row_bytes"]


def test_candidate_names_and_flags_are_explicit():
    result = matrix.candidate_options(["qkv=reuse_qkv_quantization", "control="])
    assert result["qkv"].reuse_qkv_quantization
    assert result["baseline"] == result["control"]
    for spec in ("baseline=", "../escape=", "q=unknown"):
        with pytest.raises(ValueError):
            matrix.candidate_options([spec])


def test_json_reports_are_replaced_without_partial_or_temporary_output(tmp_path):
    path = tmp_path / "report.json"
    matrix._json(path, {"status": "running"})
    matrix._json(path, {"status": "passed", "rows": [1, 2]})
    assert json.loads(path.read_text()) == {"status": "passed", "rows": [1, 2]}
    assert not list(tmp_path.glob("*.tmp"))


def test_fc2_acceptance_requires_executed_reuse_not_just_requested_flag():
    metrics = {"decode": {"optimization_events": [
        {"kind": "fc2_cache_admission", "cached_partitions": 3},
        {"kind": "fc2_quantization_hit"}, {"kind": "qkv_quantization_hit"}]}}
    coverage = matrix.cache_coverage(metrics)
    assert coverage["decode"]["fc2_hits"] == 1
    assert coverage["decode"]["qkv_hits"] == 1
    matrix.require_fc2_coverage(coverage)
    metrics["decode"]["optimization_events"].append({"kind": "fc2_reuse_fallback"})
    with pytest.raises(ValueError, match="strict FC2"):
        matrix.require_fc2_coverage(matrix.cache_coverage(metrics))
    with pytest.raises(ValueError, match="strict FC2"):
        matrix.require_fc2_coverage(matrix.cache_coverage({"decode": {"optimization_events": []}}))


def test_comparison_excludes_padding_but_catches_logits_kv_and_weights():
    weights, kv = b"weights!", b"live KV!"
    raw = weights + np.array([1, 2, 0, 0], dtype="<f2").tobytes() + kv
    layout = dict(data_bytes=len(raw), weights_bytes=8, logits_offset=8,
                  logits_row_bytes=8, rows=1, vocab_size=2, kv_offset=16, kv_bytes=8)
    baseline, _ = matrix.inspect_output(raw, layout, weights)
    padded = bytearray(raw)
    padded[12:16] = b"\xff" * 4
    candidate, _ = matrix.inspect_output(bytes(padded), layout, weights)
    assert baseline["dram_sha256"] != candidate["dram_sha256"]
    assert matrix.compare_outputs(baseline, candidate)["passed"]
    changed = bytearray(raw)
    changed[8:10] = np.float16(3).tobytes()
    candidate, _ = matrix.inspect_output(bytes(changed), layout, weights)
    assert not matrix.compare_outputs(baseline, candidate)["logical_logits_equal"]
    candidate["kv_sha256"] = "different"
    assert not matrix.compare_outputs(baseline, candidate)["kv_equal"]
    changed[0] = 0
    with pytest.raises(ValueError, match="immutable"):
        matrix.inspect_output(bytes(changed), layout, weights)
    with pytest.raises(ValueError, match="incomplete"):
        matrix.inspect_output(raw[:-1], layout, weights)


def test_matrix_compiles_once_per_shape_and_retains_contiguous_kv(tmp_path, monkeypatch):
    from taccel.runtime import tiny_fixture
    original_build = tiny_fixture.build_stage3_tiny_decoder_bundle
    calls, observed_kv = [], []

    def counted(*args, **kwargs):
        calls.append((kwargs["batch"], kwargs["prefill_tokens"], kwargs["compiler_options"]))
        return original_build(*args, **kwargs)

    def golden_as_runner(rtl, pb, layout, out, **kwargs):
        out.mkdir(parents=True)
        program = kwargs.get("program_source") or out / "program.bin"
        if "program_source" not in kwargs:
            program.write_bytes(pb.to_bytes())
        matrix._json(out / "layout.json", layout)
        input_kv = pb.data[layout["kv_offset"]:layout["kv_offset"] + layout["kv_bytes"]]
        observed_kv.append((out.name, any(input_kv)))
        raw = matrix.run_golden_preserved(pb, layout, out)
        (out / "dram.raw").write_bytes(raw)
        return {"cycles": 100, "program_sha256": matrix.sha256_file(program)}, raw

    monkeypatch.setattr(tiny_fixture, "build_stage3_tiny_decoder_bundle", counted)
    monkeypatch.setattr(matrix, "run_preserved", golden_as_runner)
    monkeypatch.setattr(matrix, "check_runner_freshness", lambda *args: None)
    monkeypatch.setattr(matrix, "source_identity", lambda repo: {"test": True})
    rtl = tmp_path / "fake-runner"
    rtl.write_bytes(b"test runner")
    out = tmp_path / "matrix"
    status = matrix.main([
        "--synthetic", "--synthetic-context", "64", "--baseline-rtl", str(rtl),
        "--rtl", str(rtl), "--batches", "1", "--positions", "0,1",
        "--prefill-tokens", "1,16", "--decode-steps", "2", "--prefill-chunks", "2",
        "--out", str(out)])
    assert status == 0
    assert len(calls) == 4  # two stream shapes, two compiler configurations
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "passed" and len(report["comparisons"]) == 9
    assert len(report["compiles"]) == 4
    assert report["host_peak_rss"]["peak_rss_bytes"] > 0
    assert report["compiles"][0]["host_peak_rss_after"]["scope"] == "current_python_process_lifetime"
    assert all(not live for name, live in observed_kv if name.startswith("shape"))
    assert all(live for name, live in observed_kv if name in ("sequence_prefill_1", "sequence_decode_0"))
    assert len(list(out.glob("**/program.bin"))) == 18
    assert len(list(out.glob("**/dram.raw"))) == 18
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["ptq_transforms"]["quarot"]["applied_in_bundle_build"] is False
    assert manifest["python"] and set(manifest["packages"]) == {"numpy", "torch", "transformers"}
    assert "available" in manifest["toolchain"]["verilator"]
    assert Path(manifest["runners"]["baseline"]["path"]).parent.name == "runner-snapshots"
    from tools import replay_optimization_matrix as replay
    replay_out = tmp_path / "replay"
    assert replay.main(["--matrix", str(out), "--baseline-rtl", str(rtl), "--rtl", str(rtl),
                        "--batches", "1", "--step", "shape_decode_p1", "--intervals", "1,2,4",
                        "--out", str(replay_out)]) == 0
    replay_report = json.loads((replay_out / "report.json").read_text())
    replay_manifest = json.loads((replay_out / "manifest.json").read_text())
    assert replay_manifest["python"] == manifest["python"]
    assert replay_manifest["packages"] == manifest["packages"]
    assert "available" in replay_manifest["toolchain"]["verilator"]
    assert len(replay_report["runs"]) == 6 and len(replay_report["comparisons"]) == 3
    assert not list(replay_out.glob("**/program.bin"))
    assert all(r["matches_original"]["passed"] for r in replay_report["runs"])
    with pytest.raises(ValueError, match="invalid retained"):
        replay.retained_case(out, report, "../baseline", "shape_decode_p1")
    bad_program = out / "b1_p1/baseline/shape_decode_p1/program.bin"
    bad_program.write_bytes(bad_program.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="program hash"):
        replay.retained_case(out, report, "b1_p1/baseline", "shape_decode_p1")
    with pytest.raises(SystemExit):
        matrix.main(["--synthetic", "--baseline-rtl", str(rtl), "--rtl", str(rtl), "--out", str(out)])


def test_indexed_liveness_microbenchmark_preserves_bytes():
    from tools.benchmark_indexed_liveness import benchmark
    result = benchmark(20, 2)
    assert result["byte_identical"]
    assert result["nodes"] == 20
    assert len(result["samples_seconds"]["indexed"]) == 2
    assert result["median_seconds"]["baseline"] > 0


@pytest.mark.parametrize("heads,width", [(0, 64), (3, 64), (4, 0)])
def test_synthetic_head_count_must_divide_positive_width(tmp_path, heads, width):
    with pytest.raises(SystemExit) as result:
        matrix.main(["--synthetic", "--synthetic-heads", str(heads),
                     "--synthetic-d-model", str(width), "--baseline-rtl", "unused",
                     "--rtl", "unused", "--out", str(tmp_path / "invalid")])
    assert result.value.code == 2
    assert not (tmp_path / "invalid").exists()


def test_disk_guard_uses_required_artifacts_plus_reserve(tmp_path, monkeypatch):
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(matrix.shutil, "disk_usage", lambda path: usage(100, 20, 80))
    assert matrix.check_disk_capacity(tmp_path, 70)["free_bytes"] == 80
    with pytest.raises(matrix.DiskCapacityError, match="disk guard"):
        matrix.check_disk_capacity(tmp_path, 81)
    with pytest.raises(matrix.DiskCapacityError, match="reserve"):
        matrix.check_disk_capacity(tmp_path, 1, min_free_gib=1)


def test_fixture_metadata_sha_is_checked_before_use(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    metadata = checkpoint.with_suffix(".pt.json")
    matrix._json(metadata, {"checkpoint_sha256": "wrong"})
    with pytest.raises(ValueError, match="paired metadata"):
        matrix.fixture_identity(checkpoint)
    matrix._json(metadata, {"checkpoint_sha256": matrix.sha256_file(checkpoint), "provenance": "test"})
    result = matrix.fixture_identity(checkpoint)
    assert result["metadata_sha256"] == matrix.sha256_file(metadata)
    assert result["provenance"] == "test"


def test_compile_only_preflight_never_runs_rtl_or_claims_numerical_pass(tmp_path, monkeypatch):
    def forbidden_run(*args, **kwargs):
        pytest.fail("preflight invoked RTL")

    monkeypatch.setattr(matrix, "run_preserved", forbidden_run)
    monkeypatch.setattr(matrix, "check_runner_freshness", lambda *args: None)
    rtl = tmp_path / "fake-runner"
    rtl.write_bytes(b"test runner")
    out = tmp_path / "preflight"
    assert matrix.main([
        "--synthetic", "--synthetic-context", "16", "--baseline-rtl", str(rtl), "--rtl", str(rtl),
        "--batches", "1", "--positions", "0,15", "--prefill-tokens", "1", "--shape-only",
        "--preflight-only", "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "preflight_passed" and not report["numerical_equivalence_executed"]
    assert len(report["planned_storage"]) == 4 and not report["runs"]
    assert report["campaign_storage"]["required_bytes"] == sum(r["required_bytes"] for r in report["planned_storage"])
    assert not list(out.glob("**/program.bin")) and not list(out.glob("**/dram.raw"))
    sequence = matrix.workload_steps(1, 1, [], 2, 1, 16)
    assert len(sequence) == 3 and all(s.sequence for s in sequence)


def test_runtime_toolchain_identity_uses_installed_metadata_and_explicit_unavailable(monkeypatch):
    from tools import benchmark_support as support
    calls = []

    def package_version(name):
        if name == "transformers":
            raise support.importlib.metadata.PackageNotFoundError(name)
        return {"numpy": "test-numpy", "torch": "test-torch"}[name]

    def installed_version(command, **kwargs):
        calls.append((command, kwargs))
        return "Verilator test-version\n"

    monkeypatch.setattr(support.importlib.metadata, "version", package_version)
    monkeypatch.setattr(support.platform, "python_version", lambda: "test-python")
    monkeypatch.setattr(support.subprocess, "check_output", installed_version)
    result = support.runtime_toolchain_identity()
    assert result["python"] == "test-python"
    assert result["packages"] == {"numpy": "test-numpy", "torch": "test-torch", "transformers": None}
    assert result["toolchain"]["verilator"]["available"] is True
    assert result["toolchain"]["verilator"]["version"] == "Verilator test-version"
    assert calls[0][0] == ["verilator", "--version"] and calls[0][1]["timeout"] == 10

    def absent(*args, **kwargs):
        raise FileNotFoundError("verilator is absent")

    monkeypatch.setattr(support.subprocess, "check_output", absent)
    unavailable = support.runtime_toolchain_identity()["toolchain"]["verilator"]
    assert unavailable["available"] is False and unavailable["version"] is None
    assert "FileNotFoundError" in unavailable["unavailable_reason"]
