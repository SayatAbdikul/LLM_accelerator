"""Tiny serialized fixtures and mocked native calls; no RTL/model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tools import w8a16_rtl_quality_gate as gate
from taccel.assembler.assembler import ProgramBinary
from taccel.isa.encoding import encode
from taccel.isa.instructions import HaltInsn, SetAddrHiInsn, SetAddrLoInsn, StoreInsn


def _json(path, value):
    path.write_text(json.dumps(value))


def _record(path):
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": gate.sha(path)}


def _program(base, position, *, store_in_weights=False):
    instructions = []
    for register, address in enumerate((base if store_in_weights else base + 32, base + 48 + position * 16)):
        instructions.extend([SetAddrLoInsn(addr_reg=register, imm28=address),
                             SetAddrHiInsn(addr_reg=register, imm28=0),
                             StoreInsn(buf_id=0, sram_off=0, xfer_len=1, addr_reg=register)])
    instructions.append(HaltInsn())
    raw = b"".join(encode(insn) for insn in instructions)
    return ProgramBinary(instructions=raw, data=b"", entry_point=0, insn_count=len(instructions), data_base=base)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    from tools import benchmark_support, check_reference_manifest
    parent, output = tmp_path / "quality", tmp_path / "replay"
    export = parent / "rtl_export"
    export.mkdir(parents=True)
    source, runner = tmp_path / "source.py", tmp_path / "runner"
    source.write_text("# fixed mock source\n")
    runner.write_text("mock binary, never executed\n")
    source_pins = {str(source): gate.sha(source)}
    monkeypatch.setattr(gate, "_verify_parent_sources", lambda report: gate._verify_pins(report["sources"]["sha256"]))
    monkeypatch.setattr(gate, "_replay_source_pins", lambda: {str(source): gate.sha(source)})
    monkeypatch.setattr(benchmark_support, "check_runner_freshness", lambda *args: None)
    monkeypatch.setattr(check_reference_manifest, "verify_reference_manifest", lambda: {"sha256": {"frozen": "fixed"}})
    tokens = [i % 8 for i in range(257)]
    row = np.arange(8, dtype=np.float16)
    arrays = {"token_ids": np.asarray(tokens, dtype=np.int64)}
    arrays.update({f"{name}_{i:04d}": row.astype(np.float32) if name != "compiled" else row
                   for name in ("compiled", "reference", "fp32") for i in range(256)})
    np.savez(parent / "logits.npz", **arrays)
    static = bytes(range(16)) + bytes(80)
    (export / "static_data.bin").write_bytes(static)
    shared_hash = hashlib.sha256(static[:16]).hexdigest()
    manifest = {
        "status": "golden_complete_rtl_not_run", "steps_requested": 3, "attention_window": 256,
        "sources": {"sha256": source_pins}, "scales_sha256": "shared-map",
        "transforms": {"quarot": {"applied": True}}, "vocab_size": 8, "token_ids": tokens[:3],
        "static_data": _record(export / "static_data.bin"), "data_base": 256, "required_dram_bytes": 352,
        "kv_offset_in_static_data": 48, "kv_cache_size": 48, "kv_cache_base": 304,
        "shared_data_bytes": 16, "shared_data_sha256": shared_hash, "logits_row_bytes": 16,
        "steps": [], "golden_steps": []}
    for i in range(3):
        program = _program(256, i)
        p = export / f"step_{i:03d}.program.bin"; p.write_bytes(program.to_bytes())
        log = export / f"golden_{i:03d}.logits.bin"; log.write_bytes(row.tobytes())
        kv = export / f"golden_{i:03d}.kv.bin"; kv.write_bytes(row.tobytes() * (i + 1) + bytes(16 * (2 - i)))
        manifest["steps"].append({"position": i, "token_id": tokens[i], "stream": "prefill" if i == 0 else "decode",
                                  "query_row_base": i, "valid_kv_len": i + 1, "program": _record(p),
                                  "insn_count": program.insn_count, "instructions_bytes": len(program.instructions),
                                  "instructions_sha256": hashlib.sha256(program.instructions).hexdigest(), "logits_offset": 288})
        manifest["golden_steps"].append({"position": i, "token_id": tokens[i], "logits": _record(log), "kv": _record(kv)})
    report = {"passed": True, "status": "passed", "exit_code": 0, "identities_unchanged": True,
              "quality": {"passed": True, "thresholds": dict(gate.THRESHOLDS), "target_count": 256, "token_count": 257, "vocab_size": 8},
              "scope": {"canonical_257_token_window": True},
              "logits_artifact": {"row_counts": dict(compiled=256, reference=256, fp32=256), "sha256": gate.sha(parent / "logits.npz")},
              "input_ids": {"evaluation": tokens, "inputs": tokens[:-1], "targets": tokens[1:]}, "model_args": {"vocab_size": 8},
              "compiled_step_execution": [{"position": i, "token_id": tokens[i], "stream": "prefill" if i == 0 else "decode", "halted": True} for i in range(256)],
              "sources": {"sha256": source_pins}, "scales_sha256": "shared-map",
              "transforms": {"quarot": {"applied": True}, "fp32_captured_before_ptq": True},
              "compiled": {"program_sha256": {"shared_data": shared_hash}}, "rtl_export": manifest}
    c = SimpleNamespace(parent=parent, export=export, out=output, source=source, runner=runner,
                        manifest=manifest, report=report, row=row, calls=[], previous=None, behavior=None)
    def save():
        _json(export / "manifest.json", manifest)
        _json(parent / "report.json", report)
    c.save = save
    save()
    def native(command, log_path):
        i = len(c.calls)
        c.calls.append(command)
        log_path.write_text("mock native output\n")
        values = {command[j]: command[j + 1] for j in range(1, len(command), 2)}
        p = ProgramBinary.from_bytes(Path(values["--program"]).read_bytes())
        data = bytearray(p.data)
        assert data[48:] == (bytes(48) if i == 0 else c.previous)
        data[32:48] = row.tobytes()
        data[48 + i * 16:64 + i * 16] = row.tobytes()
        summary = {"status": "halted", "fault": False, "timeout": False, "cycles": 123,
                   "configuration": {"sfu_synth_mode": 1, "helper_synth_mode": 1, "fetch_buffer_enabled": True},
                   "retired_instructions": p.insn_count}
        if c.behavior:
            c.behavior(i, data, summary, values)
        c.previous = bytes(data[48:])
        Path(values["--dram-dump-out"]).write_bytes(data)
        _json(Path(values["--json-out"]), summary)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(gate, "_run_native", native)
    c.argv = ["--quality", str(parent), "--runner", str(runner), "--out", str(output)]
    return c


def test_three_step_success_checks_real_carry_and_bounded_scope(campaign):
    c = campaign
    assert gate.main(c.argv) == 0
    r = json.loads((c.out / "report.json").read_text())
    assert r["passed"] and r["completed_steps"] == 3 and len(c.calls) == 3
    assert r["quality"]["target_count"] == 3 and r["quality"]["thresholds"] == gate.THRESHOLDS
    assert r["whole_kv_carry_links"] == 2 and not r["golden_kv_is_history"]
    assert r["identities_unchanged"] and r["exact_golden_state_is_diagnostic"]
    assert "not 256-target RTL" in r["scope"]
    assert all(s["golden_logits_exact"] and s["golden_kv_exact"] for s in r["steps"])
    assert r["steps"][1]["input_kv_sha256"] == r["steps"][0]["output_kv_sha256"]


@pytest.mark.parametrize("defect", ["short", "unhalted", "token", "bool_token", "position", "stream", "threshold", "export", "rows", "vocab"])
def test_parent_invalid_coverage_fails_before_native_with_report(campaign, defect):
    c = campaign
    if defect == "short": c.report["compiled_step_execution"].pop()
    elif defect == "unhalted": c.report["compiled_step_execution"][19]["halted"] = False
    elif defect == "token": c.report["compiled_step_execution"][19]["token_id"] = 7
    elif defect == "bool_token": c.report["compiled_step_execution"][0]["token_id"] = False
    elif defect == "position": c.report["compiled_step_execution"][19]["position"] = 20
    elif defect == "stream": c.report["compiled_step_execution"][19]["stream"] = "prefill"
    elif defect == "threshold": c.report["quality"]["thresholds"]["max_fp32_ppl_increase"] = 0.1
    elif defect == "rows": c.report["logits_artifact"]["row_counts"]["compiled"] = 255
    elif defect == "vocab": c.report["quality"]["vocab_size"] = 7
    else: c.report["rtl_export"] = deepcopy(c.manifest); c.report["rtl_export"]["attention_window"] = 255
    c.save()
    assert gate.main(c.argv) == 2 and not c.calls
    r = json.loads((c.out / "report.json").read_text())
    assert not r["passed"] and r["status"] == "error" and r["completed_steps"] == 0
    assert not r["identities_unchanged"]


@pytest.mark.parametrize("defect", ["retired", "mode", "nonfinite", "weights", "undeclared_store", "short_dump"])
def test_native_invalid_result_retains_partial_artifacts(campaign, defect):
    c = campaign
    def defect_at_one(i, data, summary, values):
        if i != 1: return
        if defect == "retired": summary["retired_instructions"] -= 1
        elif defect == "mode": summary["configuration"]["helper_synth_mode"] = 0
        elif defect == "nonfinite": data[32:34] = np.asarray([np.inf], dtype=np.float16).tobytes()
        elif defect == "weights": data[0] ^= 1
        elif defect == "undeclared_store": data[16] ^= 1
        else: del data[-1]
    c.behavior = defect_at_one
    assert gate.main(c.argv) == 2
    r = json.loads((c.out / "report.json").read_text())
    assert not r["passed"] and r["completed_steps"] == 1 and len(c.calls) == 2
    assert (c.out / "step_000.result.json").is_file()
    assert (c.out / "step_001.program.bin").is_file() and (c.out / "step_001.data.raw").is_file()
    assert not (c.out / "step_002.command.json").exists()


def test_golden_kv_is_only_diagnostic_and_never_hardware_history(campaign):
    c = campaign
    for golden in c.manifest["golden_steps"]:
        path = c.export / golden["kv"]["file"]
        path.write_bytes(bytes([123]) * 48)
        golden["kv"] = _record(path)
    c.save()
    assert gate.main(c.argv) == 0
    r = json.loads((c.out / "report.json").read_text())
    assert all(not s["golden_kv_exact"] for s in r["steps"])
    assert r["quality"]["passed"] and r["exact_golden_state_is_diagnostic"]


def test_metric_failure_is_nonzero_not_exact_diagnostic_pass(campaign):
    c = campaign
    def worse(i, data, summary, values):
        data[32:48] = (-np.arange(8, dtype=np.float16) * 10).tobytes()
    c.behavior = worse
    assert gate.main(c.argv) == 1
    r = json.loads((c.out / "report.json").read_text())
    assert not r["passed"] and r["status"] == "failed" and r["completed_steps"] == 3
    assert "reference_top5_overlap" in r["quality"]["violations"]


@pytest.mark.parametrize("defect", ["artifact", "source", "runner"])
def test_end_identity_drift_cannot_be_reported_as_pass(campaign, defect):
    c = campaign
    def drift(i, data, summary, values):
        if i != 2: return
        path = {"artifact": c.export / "static_data.bin", "source": c.source, "runner": c.runner}[defect]
        path.write_bytes(path.read_bytes() + b"changed")
    c.behavior = drift
    assert gate.main(c.argv) == 2
    r = json.loads((c.out / "report.json").read_text())
    assert not r["passed"] and not r["identities_unchanged"] and not r["pinned_files_unchanged"]
    assert r["completed_steps"] == 3


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("mock", 1200), KeyboardInterrupt()])
def test_timeout_or_interrupt_retains_terminal_error(campaign, monkeypatch, failure):
    def stopped(command, log):
        log.write_text("interrupted mock\n")
        raise failure
    monkeypatch.setattr(gate, "_run_native", stopped)
    assert gate.main(campaign.argv) == (130 if isinstance(failure, KeyboardInterrupt) else 2)
    r = json.loads((campaign.out / "report.json").read_text())
    assert not r["passed"] and r["completed_steps"] == 0 and r["error"]["type"] == type(failure).__name__
    assert (campaign.out / "step_000.program.bin").exists()


def test_store_into_weights_is_rejected_before_native(campaign):
    c = campaign
    p = _program(256, 0, store_in_weights=True)
    step = c.manifest["steps"][0]
    path = c.export / step["program"]["file"]
    path.write_bytes(p.to_bytes())
    step["program"] = _record(path)
    step["instructions_sha256"] = hashlib.sha256(p.instructions).hexdigest()
    c.save()
    assert gate.main(c.argv) == 2 and not c.calls
    assert "STORE overlaps" in json.loads((c.out / "report.json").read_text())["error"]["message"]


def test_existing_output_is_not_replaced(campaign):
    c = campaign
    c.out.mkdir()
    original = c.out / "report.json"; original.write_text("existing evidence")
    with pytest.raises(FileExistsError): gate.main(c.argv)
    assert original.read_text() == "existing evidence" and not c.calls


def test_missing_parent_retains_error(tmp_path):
    output = tmp_path / "out"
    assert gate.main(["--quality", str(tmp_path / "missing"), "--runner", str(tmp_path / "missing-runner"), "--out", str(output)]) == 2
    r = json.loads((output / "report.json").read_text())
    assert r["error"]["type"] == "FileNotFoundError" and not r["passed"]


def test_envelope_requires_final_halt_and_bounded_store_ranges():
    raw = _program(256, 0).instructions
    with pytest.raises(ValueError, match="terminal HALT"): gate.store_ranges(raw[:-8], 256, 352)
    with pytest.raises(ValueError, match="follow HALT"): gate.store_ranges(raw + raw[-8:], 256, 352)
    with pytest.raises(ValueError, match="escapes"): gate.store_ranges(raw, 320, 352)
