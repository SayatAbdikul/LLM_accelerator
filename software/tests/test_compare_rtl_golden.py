import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from taccel.assembler.assembler import Assembler, ProgramBinary
from taccel.isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"
COMPARE_SCRIPT = REPO_ROOT / "software" / "tools" / "compare_rtl_golden.py"
COMPARE_SPEC = importlib.util.spec_from_file_location("compare_rtl_golden", COMPARE_SCRIPT)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
compare_rtl_golden = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(compare_rtl_golden)


@pytest.fixture(scope="module", autouse=True)
def build_runner():
    subprocess.run(
        ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "run_program"],
        check=True,
    )


def _write_program(tmp_path: Path, source: str) -> Path:
    program = Assembler().assemble(source)
    path = tmp_path / "program.bin"
    path.write_bytes(program.to_bytes())
    return path


def _run_runner(program_path: Path, summary_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        str(RUNNER),
        "--program",
        str(program_path),
        "--json-out",
        str(summary_path),
        "--num-classes",
        "8",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, check=False)


def test_runner_halt_summary(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path)
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert summary["done"] is True
    assert summary["fault"] is False
    assert summary["retired_instructions"] == 2
    assert summary["violations"] == []


def test_runner_truncated_program_reports_parse_error(tmp_path: Path):
    program_path = tmp_path / "truncated.bin"
    program_path.write_bytes(b"TAC")
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path)
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 2
    assert summary["status"] == "parse_error"
    assert summary["parse_error"] is not None


def test_runner_timeout_is_reported(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path, ["--max-cycles", "0"])
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 3
    assert summary["status"] == "timeout"
    assert summary["timeout"] is True
    assert "cycle_budget_exhausted" in summary["violations"]


def test_runner_dma_fault_captures_fault_context(tmp_path: Path):
    program_path = _write_program(
        tmp_path,
        "\n".join(
            [
                "SET_ADDR_LO R0, 0x0001000",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "HALT",
            ]
        ) + "\n",
    )
    summary_path = tmp_path / "summary.json"

    proc = _run_runner(program_path, summary_path, ["--inject-rresp-addr", "0x1000:2"])
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "fault"
    assert summary["fault"] is True
    assert summary["fault_code"] == 2
    assert summary["fault_context"]["valid"] is True
    assert summary["fault_context"]["source_name"] == "dma"
    assert summary["fault_context"]["opcode"] == 7


def test_compare_program_mode_smoke_passes(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "compare_summary.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--summary-out",
            str(summary_path),
            "--runner",
            str(RUNNER),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ],
        check=False,
    )
    summary = json.loads(summary_path.read_text())

    assert proc.returncode == 0
    assert summary["pass"] is True
    assert summary["runner_exit_code"] == 0
    assert summary["mismatch_count"] == 0


def test_runner_snapshot_captures_int8_rows(tmp_path: Path):
    program = Assembler().assemble(
        "\n".join(
                [
                "SET_ADDR_LO R0, 0x0000030",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ABUF, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "HALT",
            ]
        ) + "\n",
        data=bytes(range(16)),
    )
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        # fields: pc, event_index, node_name, buf_id, offset_units,
        #         mem_rows, mem_cols, logical_rows, logical_cols,
        #         full_rows, full_cols, row_start, dtype, scale, source
        # mem_cols=16: 1 tile × 16 bytes for a 16-element INT8 row
        "3,0,trace_abuf,0,0,1,16,1,16,1,16,0,int8,1,architectural\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(snapshot_manifest.read_text())
    entry = manifest["entries"][0]
    payload = snapshot_data.read_bytes()[entry["byte_offset"]: entry["byte_offset"] + entry["byte_size"]]

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert entry["status"] == "captured"
    assert entry["capture_phase"] == "retire_cycle"
    assert payload == bytes(range(16))


def test_runner_snapshot_captures_int32_rows(tmp_path: Path):
    accum_values = np.array([10, -20, 30, -40], dtype=np.int32)
    program = Assembler().assemble(
        "\n".join(
                [
                "SET_ADDR_LO R0, 0x0000030",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ACCUM, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "HALT",
            ]
        ) + "\n",
        data=accum_values.astype("<i4").tobytes(),
    )
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        # fields: pc, event_index, node_name, buf_id, offset_units,
        #         mem_rows, mem_cols, logical_rows, logical_cols,
        #         full_rows, full_cols, row_start, dtype, scale, source
        # mem_cols=16: ceil(4/16)*16=16 (1 tile) for a 4-element INT32 row
        "3,0,trace_accum,2,0,1,16,1,4,1,4,0,int32,1,architectural\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(snapshot_manifest.read_text())
    entry = manifest["entries"][0]
    payload = snapshot_data.read_bytes()[entry["byte_offset"]: entry["byte_offset"] + entry["byte_size"]]
    observed = np.frombuffer(payload, dtype="<i4").astype(np.int32)

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert entry["status"] == "captured"
    assert entry["capture_phase"] == "retire_cycle"
    np.testing.assert_array_equal(observed, accum_values)


def test_runner_snapshot_supports_retire_plus_1_phase(tmp_path: Path):
    accum_values = np.array([10, -20, 30, -40], dtype=np.int32)
    program = Assembler().assemble(
        "\n".join(
            [
                "SET_ADDR_LO R0, 0x0000030",
                "SET_ADDR_HI R0, 0x0000000",
                "LOAD buf_id=ACCUM, sram_off=0, xfer_len=1, addr_reg=0, dram_off=0",
                "SYNC 0b001",
                "NOP",
                "HALT",
            ]
        ) + "\n",
        data=accum_values.astype("<i4").tobytes(),
    )
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        "\n".join(
            [
                "4,0,trace_accum_cycle,2,0,1,16,1,4,1,4,0,int32,1,architectural,retire_cycle",
                "4,1,trace_accum_plus1,2,0,1,16,1,4,1,4,0,int32,1,architectural,retire_plus_1",
            ]
        ) + "\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    manifest = json.loads(snapshot_manifest.read_text())
    entries = manifest["entries"]
    payload = snapshot_data.read_bytes()

    assert proc.returncode == 0
    assert [entry["capture_phase"] for entry in entries] == ["retire_cycle", "retire_plus_1"]
    first = np.frombuffer(payload[entries[0]["byte_offset"]: entries[0]["byte_offset"] + entries[0]["byte_size"]], dtype="<i4")
    second = np.frombuffer(payload[entries[1]["byte_offset"]: entries[1]["byte_offset"] + entries[1]["byte_size"]], dtype="<i4")
    np.testing.assert_array_equal(first.astype(np.int32), accum_values)
    np.testing.assert_array_equal(second.astype(np.int32), accum_values)


def test_runner_snapshot_virtual_entries_are_skipped(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"
    snapshot_request = tmp_path / "snapshot_request.csv"
    snapshot_manifest = tmp_path / "snapshot_manifest.json"
    snapshot_data = tmp_path / "snapshot_data.bin"
    snapshot_request.write_text(
        "0,0,virtual_node,0,0,1,1,1,4,1,4,0,int8,1,virtual\n"
    )

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--snapshot-request",
            str(snapshot_request),
            "--snapshot-manifest-out",
            str(snapshot_manifest),
            "--snapshot-data-out",
            str(snapshot_data),
        ],
    )
    manifest = json.loads(snapshot_manifest.read_text())
    entry = manifest["entries"][0]

    assert proc.returncode == 0
    assert entry["status"] == "skipped_virtual"
    assert snapshot_data.read_bytes() == b""


def test_runner_systolic_window_reports_unreached_window(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"
    window_path = tmp_path / "window.json"

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--systolic-window-start-pc",
            "7",
            "--systolic-window-end-pc",
            "9",
            "--systolic-window-json-out",
            str(window_path),
        ],
    )
    summary = json.loads(summary_path.read_text())
    window = json.loads(window_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert window["window_reached"] is False
    assert window["completed"] is False
    assert window["reason"] == "window_not_reached"
    assert window["records"] == []


def test_runner_systolic_window_captures_simple_matmul(tmp_path: Path):
    program_path = _write_program(
        tmp_path,
        "\n".join(
            [
                "CONFIG_TILE M=1, N=1, K=1",
                "MATMUL ABUF[0x0000], WBUF[0x0000], ACCUM[0x0000]",
                "SYNC 0b010",
                "HALT",
            ]
        ) + "\n",
    )
    summary_path = tmp_path / "summary.json"
    window_path = tmp_path / "window.json"

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--systolic-window-start-pc",
            "0",
            "--systolic-window-end-pc",
            "2",
            "--systolic-window-json-out",
            str(window_path),
            "--max-cycles",
            "20000",
        ],
    )
    summary = json.loads(summary_path.read_text())
    window = json.loads(window_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert window["window_reached"] is True
    assert window["completed"] is True
    assert window["reason"] is None
    assert len(window["records"]) > 0
    assert any(record["state"] != 0 for record in window["records"])
    assert any(record["retire_pc"] == 2 for record in window["records"])


def test_runner_accum_write_log_reports_unreached_window(tmp_path: Path):
    program_path = _write_program(tmp_path, "NOP\nHALT\n")
    summary_path = tmp_path / "summary.json"
    log_path = tmp_path / "accum_writes.json"

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--accum-write-start-pc",
            "7",
            "--accum-write-end-pc",
            "9",
            "--accum-write-json-out",
            str(log_path),
        ],
    )
    summary = json.loads(summary_path.read_text())
    log = json.loads(log_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert log["window_reached"] is False
    assert log["completed"] is False
    assert log["reason"] == "window_not_reached"
    assert log["records"] == []


def test_runner_accum_write_log_captures_simple_matmul(tmp_path: Path):
    program_path = _write_program(
        tmp_path,
        "\n".join(
            [
                "CONFIG_TILE M=1, N=1, K=1",
                "MATMUL ABUF[0x0000], WBUF[0x0000], ACCUM[0x0000]",
                "SYNC 0b010",
                "HALT",
            ]
        ) + "\n",
    )
    summary_path = tmp_path / "summary.json"
    log_path = tmp_path / "accum_writes.json"

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--accum-write-start-pc",
            "0",
            "--accum-write-end-pc",
            "2",
            "--accum-write-json-out",
            str(log_path),
            "--max-cycles",
            "20000",
        ],
    )
    summary = json.loads(summary_path.read_text())
    log = json.loads(log_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert log["window_reached"] is True
    assert log["completed"] is True
    assert len(log["records"]) > 0
    assert all(record["writer_source"] == "sys" for record in log["records"])
    assert all(len(record["row_hex"]) == 32 for record in log["records"])


def test_runner_hidden_snapshot_captures_simple_matmul(tmp_path: Path):
    program_path = _write_program(
        tmp_path,
        "\n".join(
            [
                "CONFIG_TILE M=1, N=1, K=1",
                "MATMUL ABUF[0x0000], WBUF[0x0000], ACCUM[0x0000]",
                "SYNC 0b010",
                "HALT",
            ]
        ) + "\n",
    )
    summary_path = tmp_path / "summary.json"
    snapshot_path = tmp_path / "hidden_snapshot.json"

    proc = _run_runner(
        program_path,
        summary_path,
        [
            "--systolic-hidden-snapshot-pc",
            "0",
            "--systolic-hidden-snapshot-json-out",
            str(snapshot_path),
            "--max-cycles",
            "20000",
        ],
    )
    summary = json.loads(summary_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())

    assert proc.returncode == 0
    assert summary["status"] == "halted"
    assert snapshot["captured"] is True
    assert len(snapshot["a_tile_scratch"]) == 16
    assert len(snapshot["a_skew"]) == 16
    assert len(snapshot["b_skew"]) == 16
    assert len(snapshot["pe_acc"]) == 16


def _make_trace_program() -> object:
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            },
            {
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 1,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 1,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            },
            {
                "node_name": "virtual_node",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 1.0,
                "when": "after",
                "source": "virtual",
            },
        ]
    }
    return program


def _make_golden_trace() -> dict:
    return {
        "raw_events": [
            {
                "pc": 0,
                "event_index": 0,
                "node_name": "node_a",
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "row_start": 0,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "raw_available": True,
                "raw": [1, 2, 3, 4],
            },
            {
                "pc": 0,
                "event_index": 1,
                "node_name": "node_a",
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "row_start": 1,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "raw_available": True,
                "raw": [5, 6, 7, 8],
            },
            {
                "pc": 0,
                "event_index": 2,
                "node_name": "virtual_node",
                "dtype": "int8",
                "scale": 1.0,
                "source": "virtual",
                "row_start": 0,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "raw_available": False,
            },
        ]
    }


def _write_snapshot_bundle(tmp_path: Path, *, second_fragment: bytes) -> tuple[Path, Path]:
    manifest_path = tmp_path / "snapshot_manifest.json"
    data_path = tmp_path / "snapshot_data.bin"
    first = bytes([1, 2, 3, 4])
    data_path.write_bytes(first + second_fragment)
    manifest = {
        "entries": [
            {
                "pc": 0,
                "event_index": 0,
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 1,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "status": "captured",
                "cycle": 1,
                "byte_offset": 0,
                "byte_size": 4,
            },
            {
                "pc": 0,
                "event_index": 1,
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 1,
                "mem_rows": 1,
                "mem_cols": 1,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 2,
                "full_cols": 4,
                "row_start": 1,
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "status": "captured",
                "cycle": 1,
                "byte_offset": 4,
                "byte_size": 4,
            },
        ]
    }
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, data_path


def test_compute_first_divergence_returns_none_for_exact_match(tmp_path: Path):
    program = _make_trace_program()
    golden_trace = _make_golden_trace()
    manifest_path, data_path = _write_snapshot_bundle(tmp_path, second_fragment=bytes([5, 6, 7, 8]))

    divergence = compare_rtl_golden._compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=manifest_path,
        snapshot_data_path=data_path,
        artifact_paths={"golden_trace": "golden.json", "rtl_trace": "rtl.json"},
    )

    assert divergence is None


def test_compute_first_divergence_reports_first_mismatching_fragment(tmp_path: Path):
    program = _make_trace_program()
    golden_trace = _make_golden_trace()
    manifest_path, data_path = _write_snapshot_bundle(tmp_path, second_fragment=bytes([5, 6, 7, 9]))

    divergence = compare_rtl_golden._compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=manifest_path,
        snapshot_data_path=data_path,
        artifact_paths={"golden_trace": "golden.json", "rtl_trace": "rtl.json"},
    )

    assert divergence is not None
    assert divergence["node_name"] == "node_a"
    assert divergence["mismatch_kind"] == "raw_value_mismatch"
    assert divergence["trace_pc"] == 0
    assert divergence["event_index"] == 1
    assert divergence["first_differing_element_index"] == 7
    assert divergence["raw_values"] == {"golden": 8, "rtl": 9}
    assert divergence["skipped_virtual_nodes"][0]["node_name"] == "virtual_node"


def test_compare_program_mode_writes_first_divergence_artifact_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                    "node_name": "trace_abuf",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "trace_abuf",
                                "buf_id": BUF_ABUF,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int8",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 4,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(bytes([1, 0, 0, 0]))
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert summary["first_divergence"]["node_name"] == "trace_abuf"
    assert summary["artifacts"]["program"] == str(program_path)
    assert Path(summary["artifacts"]["first_divergence"]).exists()
    assert Path(summary["artifacts"]["snapshot_manifest"]).exists()
    assert Path(summary["artifacts"]["snapshot_data"]).exists()


def test_compare_program_mode_emits_qkt_systolic_window_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "block0_head0_qkt",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)
    def fake_emit_qkt_stability_report(*, first_divergence, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_stability_report.json"
        report = {"classification": "retire_cycle_snapshot_artifact"}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_stability_report"] = str(report_path)
        return report

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "block0_head0_qkt",
                                "buf_id": BUF_ACCUM,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int32",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 16,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(np.array([1, 0, 0, 0], dtype="<i4").tobytes())
        if systolic_window_json_out is not None:
            Path(systolic_window_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": systolic_window_start_pc,
                        "window_end_pc": systolic_window_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if accum_write_json_out is not None:
            Path(accum_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": accum_write_start_pc,
                        "window_end_pc": accum_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if sram_write_json_out is not None:
            Path(sram_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": sram_write_start_pc,
                        "window_end_pc": sram_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_hidden_snapshot_json_out is not None:
            Path(systolic_hidden_snapshot_json_out).write_text(
                json.dumps(
                    {
                        "requested_pc": systolic_hidden_snapshot_pc,
                        "captured": True,
                        "reason": None,
                        "state": 0,
                        "mtile_q": 0,
                        "ntile_q": 0,
                        "ktile_q": 0,
                        "lane_q": 0,
                        "a_load_row_q": 0,
                        "drain_row_q": 0,
                        "drain_grp_q": 0,
                        "tile_drain_base_q": 0,
                        "drain_row_addr_q": 0,
                        "clear_acc": False,
                        "step_en": False,
                        "dst_clear_active": False,
                        "dst_clear_row_q": 0,
                        "dst_clear_rows_total_q": 0,
                        "a_tile_scratch": [[0] * 16 for _ in range(16)],
                        "a_skew": [[0] * 15 for _ in range(16)],
                        "b_skew": [[0] * 15 for _ in range(16)],
                        "pe_acc": [[0] * 16 for _ in range(16)],
                    }
                )
            )
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)
    monkeypatch.setattr(compare_rtl_golden, "_maybe_emit_qkt_stability_report", fake_emit_qkt_stability_report)

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert summary["first_divergence"]["node_name"] == "block0_head0_qkt"
    assert Path(summary["artifacts"]["qkt_stability_report"]).exists()
    assert Path(summary["artifacts"]["rtl_systolic_window"]).exists()


def test_compare_program_mode_emits_qkt_prestate_debug_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "block0_head0_qkt",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)

    def fake_emit_qkt_stability_report(*, first_divergence, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_stability_report.json"
        report = {"classification": "real_pre_matmul_dirty_state"}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_stability_report"] = str(report_path)
        return report

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "block0_head0_qkt",
                                "buf_id": BUF_ACCUM,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int32",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 16,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(np.array([1, 0, 0, 0], dtype="<i4").tobytes())
        if systolic_window_json_out is not None:
            Path(systolic_window_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": systolic_window_start_pc,
                        "window_end_pc": systolic_window_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if accum_write_json_out is not None:
            Path(accum_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": accum_write_start_pc,
                        "window_end_pc": accum_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if sram_write_json_out is not None:
            Path(sram_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": sram_write_start_pc,
                        "window_end_pc": sram_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_hidden_snapshot_json_out is not None:
            Path(systolic_hidden_snapshot_json_out).write_text(
                json.dumps(
                    {
                        "requested_pc": systolic_hidden_snapshot_pc,
                        "captured": True,
                        "reason": None,
                        "state": 0,
                        "mtile_q": 0,
                        "ntile_q": 0,
                        "ktile_q": 0,
                        "lane_q": 0,
                        "a_load_row_q": 0,
                        "drain_row_q": 0,
                        "drain_grp_q": 0,
                        "tile_drain_base_q": 0,
                        "drain_row_addr_q": 0,
                        "clear_acc": False,
                        "step_en": False,
                        "dst_clear_active": False,
                        "dst_clear_row_q": 0,
                        "dst_clear_rows_total_q": 0,
                        "a_tile_scratch": [[0] * 16 for _ in range(16)],
                        "a_skew": [[0] * 15 for _ in range(16)],
                        "b_skew": [[0] * 15 for _ in range(16)],
                        "pe_acc": [[0] * 16 for _ in range(16)],
                    }
                )
            )
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)
    monkeypatch.setattr(compare_rtl_golden, "_maybe_emit_qkt_stability_report", fake_emit_qkt_stability_report)

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert Path(summary["artifacts"]["rtl_accum_write_log"]).exists()
    assert Path(summary["artifacts"]["rtl_sram_write_log"]).exists()
    assert Path(summary["artifacts"]["rtl_systolic_hidden_snapshot"]).exists()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "fragment_accum_pre_matches_baseline": True,
                "fragment_qkt_output_matches_golden": True,
                "accum_write_diff_pass": True,
                "hidden_snapshot_diff_pass": True,
                "window_diff_pass": True,
            },
            "nonblocking_qkt_prestate_scratch",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": False,
                "fragment_accum_pre_matches_baseline": True,
                "fragment_qkt_output_matches_golden": False,
                "accum_write_diff_pass": True,
                "hidden_snapshot_diff_pass": True,
                "window_diff_pass": True,
            },
            "window_local_qkt_bug",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": True,
                "accum_write_diff_pass": True,
                "hidden_snapshot_diff_pass": True,
                "window_diff_pass": False,
            },
            "earlier_history_required",
        ),
        (
            {
                "fragment_artifacts_complete": False,
                "fragment_test_passed": False,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": False,
                "accum_write_diff_pass": False,
                "hidden_snapshot_diff_pass": False,
                "window_diff_pass": False,
            },
            "fragment_artifact_incomplete",
        ),
    ],
)
def test_classify_qkt_prestate_provenance(kwargs: dict[str, bool], expected: str):
    assert compare_rtl_golden.classify_qkt_prestate_provenance(**kwargs) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": True,
                "fragment_qkt_output_matches_golden": True,
                "prefix_sram_diff_pass": True,
            },
            "prefix_nonblocking_scratch",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": False,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": False,
                "prefix_sram_diff_pass": True,
            },
            "prefix_local_bug",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "pos_embed_output_matches_baseline": False,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": True,
                "prefix_sram_diff_pass": True,
            },
            "prefix_provenance_mismatch",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": True,
                "prefix_sram_diff_pass": True,
            },
            "history_earlier_than_pos_embed",
        ),
    ],
)
def test_classify_qkt_prefix_provenance(kwargs: dict[str, bool], expected: str):
    assert compare_rtl_golden.classify_qkt_prefix_provenance(**kwargs) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "pos_embed_act_input_matches_baseline": True,
                "pos_embed_pos_input_matches_baseline": True,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": True,
                "fragment_qkt_output_matches_golden": True,
                "startup_sram_diff_pass": True,
            },
            "startup_nonblocking_scratch",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": False,
                "pos_embed_act_input_matches_baseline": True,
                "pos_embed_pos_input_matches_baseline": True,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": False,
                "startup_sram_diff_pass": True,
            },
            "startup_local_bug",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "pos_embed_act_input_matches_baseline": False,
                "pos_embed_pos_input_matches_baseline": True,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": True,
                "startup_sram_diff_pass": True,
            },
            "startup_provenance_mismatch",
        ),
        (
            {
                "fragment_artifacts_complete": True,
                "fragment_test_passed": True,
                "pos_embed_act_input_matches_baseline": True,
                "pos_embed_pos_input_matches_baseline": True,
                "pos_embed_output_matches_baseline": True,
                "ln1_output_matches_baseline": True,
                "query_accum_pre_bias_matches_baseline": True,
                "fragment_accum_pre_matches_baseline": False,
                "fragment_qkt_output_matches_golden": True,
                "startup_sram_diff_pass": True,
            },
            "history_outside_program_entry",
        ),
    ],
)
def test_classify_qkt_startup_provenance(kwargs: dict[str, bool], expected: str):
    assert compare_rtl_golden.classify_qkt_startup_provenance(**kwargs) == expected


def test_compare_program_mode_sets_effective_pass_for_nonblocking_qkt_prestate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "block0_head0_qkt__accum_pre_matmul",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)

    def fake_emit_qkt_stability_report(*, first_divergence, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_stability_report.json"
        report = {"classification": "real_pre_matmul_dirty_state"}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_stability_report"] = str(report_path)
        return report

    def fake_emit_qkt_prestate_provenance_report(*, first_divergence, qkt_stability_report, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_prestate_provenance_report.json"
        report = {"classification": "nonblocking_qkt_prestate_scratch", "pass": True}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_prestate_provenance_report"] = str(report_path)
        return report

    def fake_rebase_qkt_prestate(**kwargs):
        return None

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "block0_head0_qkt__accum_pre_matmul",
                                "buf_id": BUF_ACCUM,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int32",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 16,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(np.array([1, 0, 0, 0], dtype="<i4").tobytes())
        if systolic_window_json_out is not None:
            Path(systolic_window_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": systolic_window_start_pc,
                        "window_end_pc": systolic_window_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if accum_write_json_out is not None:
            Path(accum_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": accum_write_start_pc,
                        "window_end_pc": accum_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if sram_write_json_out is not None:
            Path(sram_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": sram_write_start_pc,
                        "window_end_pc": sram_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_hidden_snapshot_json_out is not None:
            Path(systolic_hidden_snapshot_json_out).write_text(
                json.dumps(
                    {
                        "requested_pc": systolic_hidden_snapshot_pc,
                        "captured": True,
                        "reason": None,
                        "state": 0,
                        "mtile_q": 0,
                        "ntile_q": 0,
                        "ktile_q": 0,
                        "lane_q": 0,
                        "a_load_row_q": 0,
                        "drain_row_q": 0,
                        "drain_grp_q": 0,
                        "tile_drain_base_q": 0,
                        "drain_row_addr_q": 0,
                        "clear_acc": False,
                        "step_en": False,
                        "dst_clear_active": False,
                        "dst_clear_row_q": 0,
                        "dst_clear_rows_total_q": 0,
                        "a_tile_scratch": [[0] * 16 for _ in range(16)],
                        "a_skew": [[0] * 15 for _ in range(16)],
                        "b_skew": [[0] * 15 for _ in range(16)],
                        "pe_acc": [[0] * 16 for _ in range(16)],
                    }
                )
            )
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)
    monkeypatch.setattr(compare_rtl_golden, "_maybe_emit_qkt_stability_report", fake_emit_qkt_stability_report)
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_prestate_provenance_report",
        fake_emit_qkt_prestate_provenance_report,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_rebase_first_divergence_for_nonblocking_qkt_prestate",
        fake_rebase_qkt_prestate,
    )

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert summary["effective_pass"] is True
    assert summary["effective_first_divergence"] is None
    assert summary["qkt_prestate_provenance_report"]["classification"] == "nonblocking_qkt_prestate_scratch"


def test_compare_program_mode_sets_effective_pass_for_prefix_nonblocking_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "block0_head0_qkt__accum_pre_matmul",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)

    def fake_emit_qkt_stability_report(*, first_divergence, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_stability_report.json"
        report = {"classification": "real_pre_matmul_dirty_state"}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_stability_report"] = str(report_path)
        return report

    def fake_emit_qkt_prestate_provenance_report(*, first_divergence, qkt_stability_report, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_prestate_provenance_report.json"
        report = {"classification": "earlier_history_required", "pass": True}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_prestate_provenance_report"] = str(report_path)
        return report

    def fake_emit_qkt_prefix_provenance_report(*, first_divergence, qkt_prestate_provenance_report, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_prefix_provenance_report.json"
        report = {"classification": "prefix_nonblocking_scratch", "pass": True}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_prefix_provenance_report"] = str(report_path)
        return report

    def fake_rebase_qkt_prestate(**kwargs):
        return None

    def fake_rebase_qkt_prefix(**kwargs):
        return None

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "block0_head0_qkt__accum_pre_matmul",
                                "buf_id": BUF_ACCUM,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int32",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 16,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(np.array([1, 0, 0, 0], dtype="<i4").tobytes())
        if sram_write_json_out is not None:
            Path(sram_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": sram_write_start_pc,
                        "window_end_pc": sram_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_window_json_out is not None:
            Path(systolic_window_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": systolic_window_start_pc,
                        "window_end_pc": systolic_window_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if accum_write_json_out is not None:
            Path(accum_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": accum_write_start_pc,
                        "window_end_pc": accum_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_hidden_snapshot_json_out is not None:
            Path(systolic_hidden_snapshot_json_out).write_text(
                json.dumps(
                    {
                        "requested_pc": systolic_hidden_snapshot_pc,
                        "captured": True,
                        "reason": None,
                        "state": 0,
                        "mtile_q": 0,
                        "ntile_q": 0,
                        "ktile_q": 0,
                        "lane_q": 0,
                        "a_load_row_q": 0,
                        "drain_row_q": 0,
                        "drain_grp_q": 0,
                        "tile_drain_base_q": 0,
                        "drain_row_addr_q": 0,
                        "clear_acc": False,
                        "step_en": False,
                        "dst_clear_active": False,
                        "dst_clear_row_q": 0,
                        "dst_clear_rows_total_q": 0,
                        "a_tile_scratch": [[0] * 16 for _ in range(16)],
                        "a_skew": [[0] * 15 for _ in range(16)],
                        "b_skew": [[0] * 15 for _ in range(16)],
                        "pe_acc": [[0] * 16 for _ in range(16)],
                    }
                )
            )
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)
    monkeypatch.setattr(compare_rtl_golden, "_maybe_emit_qkt_stability_report", fake_emit_qkt_stability_report)
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_prestate_provenance_report",
        fake_emit_qkt_prestate_provenance_report,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_prefix_provenance_report",
        fake_emit_qkt_prefix_provenance_report,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_rebase_first_divergence_for_nonblocking_qkt_prestate",
        fake_rebase_qkt_prestate,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_rebase_first_divergence_for_nonblocking_qkt_prefix",
        fake_rebase_qkt_prefix,
    )

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert summary["effective_pass"] is True
    assert summary["effective_first_divergence"] is None
    assert summary["qkt_prefix_provenance_report"]["classification"] == "prefix_nonblocking_scratch"


def test_compare_program_mode_sets_effective_pass_for_startup_nonblocking_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "block0_head0_qkt__accum_pre_matmul",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 1.0,
                "when": "after",
            }
        ]
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())
    work_dir = tmp_path / "work"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(compare_rtl_golden, "_ensure_runner_built", lambda *args, **kwargs: None)

    def fake_emit_qkt_stability_report(*, first_divergence, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_stability_report.json"
        report = {"classification": "real_pre_matmul_dirty_state"}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_stability_report"] = str(report_path)
        return report

    def fake_emit_qkt_prestate_provenance_report(*, first_divergence, qkt_stability_report, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_prestate_provenance_report.json"
        report = {"classification": "earlier_history_required", "pass": True}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_prestate_provenance_report"] = str(report_path)
        return report

    def fake_emit_qkt_prefix_provenance_report(*, first_divergence, qkt_prestate_provenance_report, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_prefix_provenance_report.json"
        report = {"classification": "history_earlier_than_pos_embed", "pass": True}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_prefix_provenance_report"] = str(report_path)
        return report

    def fake_emit_qkt_startup_provenance_report(*, first_divergence, qkt_prefix_provenance_report, work_dir, artifacts):
        report_path = Path(work_dir) / "qkt_startup_provenance_report.json"
        report = {"classification": "startup_nonblocking_scratch", "pass": True}
        report_path.write_text(json.dumps(report))
        artifacts["qkt_startup_provenance_report"] = str(report_path)
        return report

    def fake_rebase_qkt_prestate(**kwargs):
        return None

    def fake_rebase_qkt_prefix(**kwargs):
        return None

    def fake_rebase_qkt_startup(**kwargs):
        return None

    def fake_emit_qkt_startup_debug_artifacts(**kwargs):
        return None

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        summary = {
            "status": "halted",
            "done": True,
            "fault": False,
            "fault_code": 0,
            "logits": [1, 0, 0, 0, 0, 0, 0, 0],
        }
        Path(summary_path).write_text(json.dumps(summary))
        if trace_json_out is not None:
            Path(trace_json_out).write_text(json.dumps({"events": [{"pc": 0, "opcode": 0}]}))
        if snapshot_manifest_out is not None and snapshot_data_out is not None:
            Path(snapshot_manifest_out).write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "pc": 0,
                                "event_index": 0,
                                "node_name": "block0_head0_qkt__accum_pre_matmul",
                                "buf_id": BUF_ACCUM,
                                "offset_units": 0,
                                "mem_rows": 1,
                                "mem_cols": 1,
                                "logical_rows": 1,
                                "logical_cols": 4,
                                "full_rows": 1,
                                "full_cols": 4,
                                "row_start": 0,
                                "dtype": "int32",
                                "scale": 1.0,
                                "source": "architectural",
                                "status": "captured",
                                "cycle": 1,
                                "byte_offset": 0,
                                "byte_size": 16,
                            }
                        ]
                    }
                )
            )
            Path(snapshot_data_out).write_bytes(np.array([1, 0, 0, 0], dtype="<i4").tobytes())
        if sram_write_json_out is not None:
            Path(sram_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": sram_write_start_pc,
                        "window_end_pc": sram_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_window_json_out is not None:
            Path(systolic_window_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": systolic_window_start_pc,
                        "window_end_pc": systolic_window_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if accum_write_json_out is not None:
            Path(accum_write_json_out).write_text(
                json.dumps(
                    {
                        "window_start_pc": accum_write_start_pc,
                        "window_end_pc": accum_write_end_pc,
                        "window_reached": True,
                        "completed": True,
                        "reason": None,
                        "records": [],
                    }
                )
            )
        if systolic_hidden_snapshot_json_out is not None:
            Path(systolic_hidden_snapshot_json_out).write_text(
                json.dumps(
                    {
                        "requested_pc": systolic_hidden_snapshot_pc,
                        "captured": True,
                        "reason": None,
                        "state": 0,
                        "mtile_q": 0,
                        "ntile_q": 0,
                        "ktile_q": 0,
                        "lane_q": 0,
                        "a_load_row_q": 0,
                        "drain_row_q": 0,
                        "drain_grp_q": 0,
                        "tile_drain_base_q": 0,
                        "drain_row_addr_q": 0,
                        "clear_acc": False,
                        "step_en": False,
                        "dst_clear_active": False,
                        "dst_clear_row_q": 0,
                        "dst_clear_rows_total_q": 0,
                        "a_tile_scratch": [[0] * 16 for _ in range(16)],
                        "a_skew": [[0] * 15 for _ in range(16)],
                        "b_skew": [[0] * 15 for _ in range(16)],
                        "pe_acc": [[0] * 16 for _ in range(16)],
                    }
                )
            )
        return 0, summary

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)
    monkeypatch.setattr(compare_rtl_golden, "_maybe_emit_qkt_stability_report", fake_emit_qkt_stability_report)
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_prestate_provenance_report",
        fake_emit_qkt_prestate_provenance_report,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_prefix_provenance_report",
        fake_emit_qkt_prefix_provenance_report,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_startup_provenance_report",
        fake_emit_qkt_startup_provenance_report,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_emit_qkt_startup_debug_artifacts",
        fake_emit_qkt_startup_debug_artifacts,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_rebase_first_divergence_for_nonblocking_qkt_prestate",
        fake_rebase_qkt_prestate,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_rebase_first_divergence_for_nonblocking_qkt_prefix",
        fake_rebase_qkt_prefix,
    )
    monkeypatch.setattr(
        compare_rtl_golden,
        "_maybe_rebase_first_divergence_for_nonblocking_qkt_startup",
        fake_rebase_qkt_startup,
    )

    args = compare_rtl_golden.build_parser().parse_args(
        [
            "--summary-out",
            str(summary_out),
            "--runner",
            str(RUNNER),
            "--work-dir",
            str(work_dir),
            "--num-classes",
            "8",
            "program",
            "--program",
            str(program_path),
        ]
    )
    summary = compare_rtl_golden.compare_program_mode(args)

    assert summary["pass"] is False
    assert summary["effective_pass"] is True
    assert summary["effective_first_divergence"] is None
    assert summary["qkt_startup_provenance_report"]["classification"] == "startup_nonblocking_scratch"


def test_diff_systolic_window_traces_reports_first_field_mismatch(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    fragment_path = tmp_path / "fragment.json"
    baseline_path.write_text(
        json.dumps(
            {
                "window_start_pc": 75,
                "window_end_pc": 80,
                "window_reached": True,
                "completed": True,
                "reason": None,
                "records": [
                    {"ctrl_pc": 75, "state": 2, "lane_q": 0, "sys_sram_a_row": 10, "retire_pc": None, "retire_opcode": None},
                    {"ctrl_pc": 76, "state": 3, "lane_q": 1, "sys_sram_a_row": 11, "retire_pc": 76, "retire_opcode": 9},
                ],
            }
        )
    )
    fragment_path.write_text(
        json.dumps(
            {
                "window_start_pc": 75,
                "window_end_pc": 80,
                "window_reached": True,
                "completed": True,
                "reason": None,
                "records": [
                    {"ctrl_pc": 75, "state": 2, "lane_q": 0, "sys_sram_a_row": 10, "retire_pc": None, "retire_opcode": None},
                    {"ctrl_pc": 76, "state": 4, "lane_q": 1, "sys_sram_a_row": 11, "retire_pc": 76, "retire_opcode": 9},
                ],
            }
        )
    )

    diff = compare_rtl_golden.diff_systolic_window_traces(
        baseline_trace_path=baseline_path,
        fragment_trace_path=fragment_path,
    )

    assert diff["pass"] is False
    assert diff["first_diff_cycle_index"] == 1
    assert diff["field_name"] == "state"
    assert diff["baseline_value"] == 3
    assert diff["fragment_value"] == 4


def test_diff_systolic_window_traces_trims_idle_prefix_and_ignores_pc_only_differences(tmp_path: Path):
    baseline_path = tmp_path / "baseline_trim.json"
    fragment_path = tmp_path / "fragment_trim.json"
    baseline_path.write_text(
        json.dumps(
            {
                "window_start_pc": 75,
                "window_end_pc": 80,
                "window_reached": True,
                "completed": True,
                "reason": None,
                "records": [
                    {"ctrl_pc": 75, "state": 0, "sys_busy": False, "dst_clear_active": False, "step_en": False, "clear_acc": False},
                    {"ctrl_pc": 76, "state": 0, "sys_busy": False, "dst_clear_active": False, "step_en": False, "clear_acc": False},
                    {"ctrl_pc": 77, "state": 2, "sys_busy": True, "dst_clear_active": False, "step_en": False, "clear_acc": False, "lane_q": 0},
                ],
            }
        )
    )
    fragment_path.write_text(
        json.dumps(
            {
                "window_start_pc": 11,
                "window_end_pc": 13,
                "window_reached": True,
                "completed": True,
                "reason": None,
                "records": [
                    {"ctrl_pc": 11, "state": 0, "sys_busy": False, "dst_clear_active": False, "step_en": False, "clear_acc": False},
                    {"ctrl_pc": 12, "state": 2, "sys_busy": True, "dst_clear_active": False, "step_en": False, "clear_acc": False, "lane_q": 0},
                ],
            }
        )
    )

    diff = compare_rtl_golden.diff_systolic_window_traces(
        baseline_trace_path=baseline_path,
        fragment_trace_path=fragment_path,
    )

    assert diff["pass"] is True
    assert diff["baseline_trim_start_cycle_index"] == 2
    assert diff["fragment_trim_start_cycle_index"] == 1
    assert diff["field_name"] is None


def test_diff_systolic_window_traces_ignores_idle_suffix_length_difference(tmp_path: Path):
    baseline_path = tmp_path / "baseline_suffix.json"
    fragment_path = tmp_path / "fragment_suffix.json"
    shared_active = [
        {
            "state": 2,
            "sys_busy": True,
            "dst_clear_active": False,
            "step_en": False,
            "clear_acc": False,
            "sys_sram_a_row": 12,
            "sys_sram_b_row": 34,
            "lane_q": 0,
        },
        {
            "state": 0,
            "sys_busy": False,
            "dst_clear_active": False,
            "step_en": False,
            "clear_acc": False,
            "sys_sram_a_row": 0,
            "sys_sram_b_row": 0,
            "lane_q": 0,
        },
    ]
    idle_tail = {
        "state": 0,
        "sys_busy": False,
        "dst_clear_active": False,
        "step_en": False,
        "clear_acc": False,
        "sys_sram_a_row": 0,
        "sys_sram_b_row": 0,
        "lane_q": 0,
    }
    baseline_path.write_text(json.dumps({"records": shared_active + [idle_tail, idle_tail]}))
    fragment_path.write_text(json.dumps({"records": shared_active}))

    diff = compare_rtl_golden.diff_systolic_window_traces(
        baseline_trace_path=baseline_path,
        fragment_trace_path=fragment_path,
    )

    assert diff["pass"] is True
    assert diff["field_name"] is None


def test_diff_accum_write_logs_reports_first_field_mismatch(tmp_path: Path):
    baseline_path = tmp_path / "baseline_accum.json"
    fragment_path = tmp_path / "fragment_accum.json"
    baseline_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cycle": 100,
                        "writer_source": "sys",
                        "row": 12,
                        "issue_pc": 45,
                        "issue_opcode": 10,
                        "first_word0": 1,
                        "first_word1": 2,
                        "row_hex": "00000004000000030000000200000001",
                    }
                ]
            }
        )
    )
    fragment_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cycle": 200,
                        "writer_source": "helper",
                        "row": 12,
                        "issue_pc": 45,
                        "issue_opcode": 10,
                        "first_word0": 1,
                        "first_word1": 2,
                        "row_hex": "00000004000000030000000200000001",
                    }
                ]
            }
        )
    )

    diff = compare_rtl_golden.diff_accum_write_logs(
        baseline_log_path=baseline_path,
        fragment_log_path=fragment_path,
    )

    assert diff["pass"] is False
    assert diff["first_diff_record_index"] == 0
    assert diff["field_name"] == "writer_source"
    assert diff["baseline_value"] == "sys"
    assert diff["fragment_value"] == "helper"


def test_diff_sram_write_logs_reports_first_field_mismatch(tmp_path: Path):
    baseline_path = tmp_path / "baseline_sram.json"
    fragment_path = tmp_path / "fragment_sram.json"
    baseline_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cycle": 100,
                        "writer_source": "dma",
                        "buf_id": 1,
                        "buf_name": "wbuf",
                        "row": 12,
                        "issue_pc": 31,
                        "issue_opcode": 7,
                        "first_word0": 1,
                        "first_word1": 2,
                        "row_hex": "00000004000000030000000200000001",
                    }
                ]
            }
        )
    )
    fragment_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cycle": 101,
                        "writer_source": "dma",
                        "buf_id": 0,
                        "buf_name": "abuf",
                        "row": 12,
                        "issue_pc": 31,
                        "issue_opcode": 7,
                        "first_word0": 1,
                        "first_word1": 2,
                        "row_hex": "00000004000000030000000200000001",
                    }
                ]
            }
        )
    )

    diff = compare_rtl_golden.diff_sram_write_logs(
        baseline_log_path=baseline_path,
        fragment_log_path=fragment_path,
    )

    assert diff["pass"] is False
    assert diff["first_diff_record_index"] == 0
    assert diff["field_name"] == "buf_id"
    assert diff["baseline_value"] == 1
    assert diff["fragment_value"] == 0


def test_emit_ln1_operand_report_reports_first_mismatching_beta_row(tmp_path: Path):
    replay_dir = tmp_path / "replay_payloads"
    replay_dir.mkdir()
    gamma_bytes = bytes(range(16))
    beta_bytes = bytes(range(16, 32))
    gamma_path = replay_dir / "ln1_gamma.raw"
    beta_path = replay_dir / "ln1_beta.raw"
    gamma_path.write_bytes(gamma_bytes)
    beta_path.write_bytes(beta_bytes)
    (replay_dir / "replay_metadata.json").write_text(
        json.dumps(
            {
                "ln1_gamma_path": str(gamma_path),
                "ln1_beta_path": str(beta_path),
                "ln1_gamma_beta_wbuf_offset_units": 0,
            }
        )
    )

    baseline_log = tmp_path / "rtl_sram_write_log.json"
    baseline_log.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cycle": 100,
                        "writer_source": "dma",
                        "buf_id": 1,
                        "buf_name": "wbuf",
                        "row": 1,
                        "issue_pc": 25,
                        "issue_opcode": 7,
                        "first_word0": int.from_bytes(gamma_bytes[0:4], byteorder="little", signed=False),
                        "first_word1": int.from_bytes(gamma_bytes[4:8], byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(gamma_bytes),
                    }
                ]
            }
        )
    )

    report = compare_rtl_golden.emit_ln1_operand_report_from_replay_dir(
        replay_dir=replay_dir,
        sram_log_path=baseline_log,
        out_path=tmp_path / "ln1_operand_report.json",
    )

    assert report["pass"] is False
    assert report["first_mismatch"]["row"] == 1
    assert report["first_mismatch"]["region"] == "beta"
    assert report["first_mismatch"]["ordering_hint"] == "swapped_gamma_beta"
    assert report["first_mismatch"]["expected_row_hex"] == compare_rtl_golden._sram_row_hex_from_bytes(beta_bytes)
    assert report["first_mismatch"]["observed_row_hex"] == compare_rtl_golden._sram_row_hex_from_bytes(gamma_bytes)


def test_maybe_emit_ln1_provenance_debug_artifacts_uses_fresh_ln1_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    replay_dir = work_dir / "replay_payloads"
    replay_dir.mkdir()
    gamma_bytes = bytes(range(16))
    beta_bytes = bytes(range(16, 32))
    output_bytes = bytes([3] * 32)
    gamma_path = replay_dir / "ln1_gamma.raw"
    beta_path = replay_dir / "ln1_beta.raw"
    output_path = replay_dir / "ln1_output_padded.raw"
    gamma_path.write_bytes(gamma_bytes)
    beta_path.write_bytes(beta_bytes)
    output_path.write_bytes(output_bytes)
    (replay_dir / "replay_metadata.json").write_text(
        json.dumps(
            {
                "ln1_gamma_path": str(gamma_path),
                "ln1_beta_path": str(beta_path),
                "ln1_output_padded_path": str(output_path),
                "ln1_gamma_beta_wbuf_offset_units": 0,
                "ln1_output_padded_offset_units": 64,
            }
        )
    )

    observed_calls: list[dict[str, int | str | None]] = []

    def fake_invoke_runner(
        runner_path,
        program_path,
        summary_path,
        patches_raw_path,
        patch_rows,
        patch_cols,
        cls_raw_path,
        folded_pos_embed,
        num_classes,
        max_cycles,
        trace_json_out=None,
        snapshot_request_path=None,
        snapshot_manifest_out=None,
        snapshot_data_out=None,
        systolic_window_start_pc=None,
        systolic_window_end_pc=None,
        systolic_window_json_out=None,
        accum_write_start_pc=None,
        accum_write_end_pc=None,
        accum_write_json_out=None,
        sram_write_start_pc=None,
        sram_write_end_pc=None,
        sram_write_json_out=None,
        systolic_hidden_snapshot_pc=None,
        systolic_hidden_snapshot_json_out=None,
    ):
        observed_calls.append(
            {
                "sram_write_start_pc": sram_write_start_pc,
                "sram_write_end_pc": sram_write_end_pc,
                "systolic_window_start_pc": systolic_window_start_pc,
                "systolic_window_end_pc": systolic_window_end_pc,
            }
        )
        Path(summary_path).write_text(json.dumps({"status": "halted", "fault": False, "logits": []}))
        Path(sram_write_json_out).write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "cycle": 10,
                            "writer_source": "dma",
                            "buf_id": 1,
                            "buf_name": "wbuf",
                            "row": 0,
                            "issue_pc": 21,
                            "issue_opcode": 7,
                            "first_word0": int.from_bytes(gamma_bytes[0:4], byteorder="little", signed=False),
                            "first_word1": int.from_bytes(gamma_bytes[4:8], byteorder="little", signed=False),
                            "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(gamma_bytes),
                        },
                        {
                            "cycle": 11,
                            "writer_source": "dma",
                            "buf_id": 1,
                            "buf_name": "wbuf",
                            "row": 1,
                            "issue_pc": 25,
                            "issue_opcode": 7,
                            "first_word0": int.from_bytes(beta_bytes[0:4], byteorder="little", signed=False),
                            "first_word1": int.from_bytes(beta_bytes[4:8], byteorder="little", signed=False),
                            "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(beta_bytes),
                        },
                    ]
                }
            )
        )
        Path(systolic_window_json_out).write_text(
            json.dumps(
                {
                    "window_start_pc": systolic_window_start_pc,
                    "window_end_pc": systolic_window_end_pc,
                    "window_reached": True,
                    "completed": True,
                    "reason": None,
                    "records": [],
                }
            )
        )
        return 0, {"status": "halted", "fault": False, "logits": []}

    monkeypatch.setattr(compare_rtl_golden, "_invoke_runner", fake_invoke_runner)

    artifacts = {"replay_payloads": str(replay_dir)}
    compare_rtl_golden._maybe_emit_ln1_provenance_debug_artifacts(
        first_divergence={"node_name": "block0_head0_qkt"},
        runner_path=tmp_path / "runner",
        program_path=tmp_path / "program.bin",
        rtl_summary_path=work_dir / "rtl_summary.json",
        patches_raw_path=None,
        patch_rows=None,
        patch_cols=None,
        cls_raw_path=None,
        folded_pos_embed=False,
        num_classes=8,
        max_cycles=100,
        work_dir=work_dir,
        artifacts=artifacts,
    )

    assert observed_calls == [
        {
            "sram_write_start_pc": compare_rtl_golden.LN1_SRAM_WRITE_START_PC,
            "sram_write_end_pc": compare_rtl_golden.LN1_SRAM_WRITE_END_PC,
            "systolic_window_start_pc": compare_rtl_golden.LN1_WINDOW_START_PC,
            "systolic_window_end_pc": compare_rtl_golden.LN1_WINDOW_END_PC,
        }
    ]
    assert Path(artifacts["rtl_ln1_sram_write_log"]).exists()
    assert Path(artifacts["rtl_ln1_window"]).exists()
    assert Path(artifacts["ln1_operand_report"]).exists()
    report = json.loads(Path(artifacts["ln1_operand_report"]).read_text())
    assert report["pass"] is True


def test_emit_ln1_provenance_report_detects_clipped_gamma_rows(tmp_path: Path):
    replay_dir = tmp_path / "replay_payloads"
    replay_dir.mkdir()
    gamma_rows = [bytes([idx] * 16) for idx in range(2)]
    beta_rows = [bytes([idx + 10] * 16) for idx in range(2)]
    gamma_path = replay_dir / "ln1_gamma.raw"
    beta_path = replay_dir / "ln1_beta.raw"
    output_path = replay_dir / "ln1_output_padded.raw"
    gamma_path.write_bytes(b"".join(gamma_rows))
    beta_path.write_bytes(b"".join(beta_rows))
    output_path.write_bytes(bytes([5] * 32))
    (replay_dir / "replay_metadata.json").write_text(
        json.dumps(
            {
                "ln1_gamma_path": str(gamma_path),
                "ln1_beta_path": str(beta_path),
                "ln1_output_padded_path": str(output_path),
                "ln1_gamma_beta_wbuf_offset_units": 0,
                "ln1_output_padded_offset_units": 64,
            }
        )
    )
    baseline_log = tmp_path / "baseline_ln1.json"
    fragment_log = tmp_path / "fragment_ln1.json"
    baseline_log.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cycle": 0,
                        "writer_source": "dma",
                        "buf_id": 1,
                        "buf_name": "wbuf",
                        "row": 2,
                        "issue_pc": 25,
                        "issue_opcode": 7,
                        "first_word0": int.from_bytes(beta_rows[0][0:4], byteorder="little", signed=False),
                        "first_word1": int.from_bytes(beta_rows[0][4:8], byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(beta_rows[0]),
                    },
                    {
                        "cycle": 1,
                        "writer_source": "dma",
                        "buf_id": 1,
                        "buf_name": "wbuf",
                        "row": 3,
                        "issue_pc": 25,
                        "issue_opcode": 7,
                        "first_word0": int.from_bytes(beta_rows[1][0:4], byteorder="little", signed=False),
                        "first_word1": int.from_bytes(beta_rows[1][4:8], byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(beta_rows[1]),
                    },
                ]
            }
        )
    )
    fragment_log.write_text(
        json.dumps(
            {
                "records": [
                    *[
                        {
                            "cycle": idx,
                            "writer_source": "dma",
                            "buf_id": 1,
                            "buf_name": "wbuf",
                            "row": idx,
                            "issue_pc": 5,
                            "issue_opcode": 7,
                            "first_word0": int.from_bytes(row_bytes[0:4], byteorder="little", signed=False),
                            "first_word1": int.from_bytes(row_bytes[4:8], byteorder="little", signed=False),
                            "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(row_bytes),
                        }
                        for idx, row_bytes in enumerate(gamma_rows + beta_rows)
                    ],
                ]
            }
        )
    )

    report = compare_rtl_golden.emit_ln1_provenance_report(
        replay_dir=replay_dir,
        baseline_log_path=baseline_log,
        fragment_log_path=fragment_log,
        out_path=tmp_path / "ln1_provenance_report.json",
    )

    assert report["classification"] == "capture_window_clipped"
    assert report["gamma_rows_present"] is False
    assert report["first_observed_wbuf_row"] == 2
    assert report["first_missing_expected_wbuf_row"] == 0


def test_emit_ln1_provenance_report_exonerates_matching_ln1_loading(tmp_path: Path):
    replay_dir = tmp_path / "replay_payloads"
    replay_dir.mkdir()
    gamma_rows = [bytes([idx] * 16) for idx in range(2)]
    beta_rows = [bytes([idx + 10] * 16) for idx in range(2)]
    output_rows = [bytes([20 + idx] * 16) for idx in range(2)]
    gamma_path = replay_dir / "ln1_gamma.raw"
    beta_path = replay_dir / "ln1_beta.raw"
    output_path = replay_dir / "ln1_output_padded.raw"
    gamma_path.write_bytes(b"".join(gamma_rows))
    beta_path.write_bytes(b"".join(beta_rows))
    output_path.write_bytes(b"".join(output_rows))
    (replay_dir / "replay_metadata.json").write_text(
        json.dumps(
            {
                "ln1_gamma_path": str(gamma_path),
                "ln1_beta_path": str(beta_path),
                "ln1_output_padded_path": str(output_path),
                "ln1_gamma_beta_wbuf_offset_units": 0,
                "ln1_output_padded_offset_units": 64,
            }
        )
    )

    def build_log(issue_pc: int) -> dict[str, object]:
        records = []
        for idx, row_bytes in enumerate(gamma_rows + beta_rows):
            records.append(
                {
                    "cycle": idx,
                    "writer_source": "dma",
                    "buf_id": 1,
                    "buf_name": "wbuf",
                    "row": idx,
                    "issue_pc": issue_pc,
                    "issue_opcode": 7,
                    "first_word0": int.from_bytes(row_bytes[0:4], byteorder="little", signed=False),
                    "first_word1": int.from_bytes(row_bytes[4:8], byteorder="little", signed=False),
                    "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(row_bytes),
                }
            )
        for idx, row_bytes in enumerate(output_rows):
            records.append(
                {
                    "cycle": 10 + idx,
                    "writer_source": "sfu",
                    "buf_id": 0,
                    "buf_name": "abuf",
                    "row": 64 + idx,
                    "issue_pc": issue_pc + 2,
                    "issue_opcode": 16,
                    "first_word0": int.from_bytes(row_bytes[0:4], byteorder="little", signed=False),
                    "first_word1": int.from_bytes(row_bytes[4:8], byteorder="little", signed=False),
                    "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(row_bytes),
                }
            )
        return {"records": records}

    baseline_log = tmp_path / "baseline_ln1.json"
    fragment_log = tmp_path / "fragment_ln1.json"
    baseline_log.write_text(json.dumps(build_log(21)))
    fragment_log.write_text(json.dumps(build_log(5)))

    report = compare_rtl_golden.emit_ln1_provenance_report(
        replay_dir=replay_dir,
        baseline_log_path=baseline_log,
        fragment_log_path=fragment_log,
        out_path=tmp_path / "ln1_provenance_report.json",
    )

    assert report["classification"] == "ln1_operand_loading_exonerated"
    assert report["baseline_wbuf_matches_fragment"] is True
    assert report["baseline_abuf_ln1_output_matches_fragment"] is True
    assert report["operand_report_pass"] is True


def test_emit_ln1_provenance_report_flags_abuf_divergence_after_matching_wbuf(tmp_path: Path):
    replay_dir = tmp_path / "replay_payloads"
    replay_dir.mkdir()
    gamma_rows = [bytes([idx] * 16) for idx in range(2)]
    beta_rows = [bytes([idx + 10] * 16) for idx in range(2)]
    output_rows = [bytes([20 + idx] * 16) for idx in range(2)]
    gamma_path = replay_dir / "ln1_gamma.raw"
    beta_path = replay_dir / "ln1_beta.raw"
    output_path = replay_dir / "ln1_output_padded.raw"
    gamma_path.write_bytes(b"".join(gamma_rows))
    beta_path.write_bytes(b"".join(beta_rows))
    output_path.write_bytes(b"".join(output_rows))
    (replay_dir / "replay_metadata.json").write_text(
        json.dumps(
            {
                "ln1_gamma_path": str(gamma_path),
                "ln1_beta_path": str(beta_path),
                "ln1_output_padded_path": str(output_path),
                "ln1_gamma_beta_wbuf_offset_units": 0,
                "ln1_output_padded_offset_units": 64,
            }
        )
    )

    def build_wbuf_records(issue_pc: int) -> list[dict[str, object]]:
        return [
            {
                "cycle": idx,
                "writer_source": "dma",
                "buf_id": 1,
                "buf_name": "wbuf",
                "row": idx,
                "issue_pc": issue_pc,
                "issue_opcode": 7,
                "first_word0": int.from_bytes(row_bytes[0:4], byteorder="little", signed=False),
                "first_word1": int.from_bytes(row_bytes[4:8], byteorder="little", signed=False),
                "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(row_bytes),
            }
            for idx, row_bytes in enumerate(gamma_rows + beta_rows)
        ]

    baseline_log = tmp_path / "baseline_ln1.json"
    fragment_log = tmp_path / "fragment_ln1.json"
    baseline_log.write_text(
        json.dumps(
            {
                "records": build_wbuf_records(21)
                + [
                    {
                        "cycle": 10,
                        "writer_source": "sfu",
                        "buf_id": 0,
                        "buf_name": "abuf",
                        "row": 64,
                        "issue_pc": 27,
                        "issue_opcode": 16,
                        "first_word0": int.from_bytes(bytes([99] * 4), byteorder="little", signed=False),
                        "first_word1": int.from_bytes(bytes([99] * 4), byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(bytes([99] * 16)),
                    },
                    {
                        "cycle": 11,
                        "writer_source": "sfu",
                        "buf_id": 0,
                        "buf_name": "abuf",
                        "row": 65,
                        "issue_pc": 27,
                        "issue_opcode": 16,
                        "first_word0": int.from_bytes(output_rows[1][0:4], byteorder="little", signed=False),
                        "first_word1": int.from_bytes(output_rows[1][4:8], byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(output_rows[1]),
                    },
                ]
            }
        )
    )
    fragment_log.write_text(
        json.dumps(
            {
                "records": build_wbuf_records(5)
                + [
                    {
                        "cycle": 10,
                        "writer_source": "sfu",
                        "buf_id": 0,
                        "buf_name": "abuf",
                        "row": 64,
                        "issue_pc": 11,
                        "issue_opcode": 16,
                        "first_word0": int.from_bytes(output_rows[0][0:4], byteorder="little", signed=False),
                        "first_word1": int.from_bytes(output_rows[0][4:8], byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(output_rows[0]),
                    },
                    {
                        "cycle": 11,
                        "writer_source": "sfu",
                        "buf_id": 0,
                        "buf_name": "abuf",
                        "row": 65,
                        "issue_pc": 11,
                        "issue_opcode": 16,
                        "first_word0": int.from_bytes(output_rows[1][0:4], byteorder="little", signed=False),
                        "first_word1": int.from_bytes(output_rows[1][4:8], byteorder="little", signed=False),
                        "row_hex": compare_rtl_golden._sram_row_hex_from_bytes(output_rows[1]),
                    },
                ]
            }
        )
    )

    report = compare_rtl_golden.emit_ln1_provenance_report(
        replay_dir=replay_dir,
        baseline_log_path=baseline_log,
        fragment_log_path=fragment_log,
        out_path=tmp_path / "ln1_provenance_report.json",
    )

    assert report["classification"] == "ln1_full_program_serialization_issue"
    assert report["baseline_wbuf_matches_fragment"] is True
    assert report["baseline_abuf_ln1_output_matches_fragment"] is False


def test_program_slice_helpers_distinguish_relative_and_absolute_offsets():
    program = ProgramBinary()
    program.data = bytes(range(64))
    program.data_base = 16

    assert compare_rtl_golden._read_program_data_relative_slice(program, 4, 6) == bytes(range(4, 10))
    assert compare_rtl_golden._read_program_absolute_dram_slice(program, 20, 6) == bytes(range(4, 10))


def test_validate_ln1_operand_bytes_rejects_nonfinite_preview():
    gamma = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float16).tobytes()
    beta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes()

    with pytest.raises(ValueError, match="ln1 gamma preview"):
        compare_rtl_golden._validate_ln1_operand_bytes(
            gamma_bytes=gamma,
            beta_bytes=beta,
        )


def test_diff_hidden_snapshots_reports_first_array_mismatch(tmp_path: Path):
    def snapshot_payload(pe_acc_00: int) -> dict[str, object]:
        return {
            "captured": True,
            "state": 0,
            "mtile_q": 0,
            "ntile_q": 0,
            "ktile_q": 0,
            "lane_q": 0,
            "a_load_row_q": 0,
            "drain_row_q": 0,
            "drain_grp_q": 0,
            "tile_drain_base_q": 0,
            "drain_row_addr_q": 0,
            "clear_acc": False,
            "step_en": False,
            "dst_clear_active": False,
            "dst_clear_row_q": 0,
            "dst_clear_rows_total_q": 0,
            "a_tile_scratch": [[0] * 16 for _ in range(16)],
            "a_skew": [[0] * 15 for _ in range(16)],
            "b_skew": [[0] * 15 for _ in range(16)],
            "pe_acc": [[pe_acc_00] + [0] * 15] + [[0] * 16 for _ in range(15)],
        }

    baseline_path = tmp_path / "baseline_hidden.json"
    fragment_path = tmp_path / "fragment_hidden.json"
    baseline_path.write_text(json.dumps(snapshot_payload(0)))
    fragment_path.write_text(json.dumps(snapshot_payload(9)))

    diff = compare_rtl_golden.diff_hidden_snapshots(
        baseline_snapshot_path=baseline_path,
        fragment_snapshot_path=fragment_path,
    )

    assert diff["pass"] is False
    assert diff["field_name"] == "pe_acc[0][0]"
    assert diff["baseline_value"] == 0
    assert diff["fragment_value"] == 9


def test_emit_qkt_stability_report_classifies_retire_cycle_snapshot_artifact(tmp_path: Path):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        1: [
            {
                "node_name": "block0_head0_qkt__accum_pre_matmul",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 2,
                "mem_cols": 4,
                "logical_rows": 2,
                "logical_cols": 4,
                "full_rows": 4,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 0.125,
                "when": "after",
                "capture_phase": "retire_cycle",
            },
            {
                "node_name": "block0_head0_qkt__accum_pre_matmul_next",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 2,
                "mem_cols": 4,
                "logical_rows": 2,
                "logical_cols": 4,
                "full_rows": 4,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 0.125,
                "when": "after",
                "capture_phase": "retire_plus_1",
            },
        ],
        2: [
            {
                "node_name": "block0_head0_qkt",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 2,
                "mem_cols": 4,
                "logical_rows": 2,
                "logical_cols": 4,
                "full_rows": 4,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 0.125,
                "when": "after",
            }
        ],
        3: [
            {
                "node_name": "block0_head0_qkt__accum_pre_softmax",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 2,
                "mem_cols": 4,
                "logical_rows": 2,
                "logical_cols": 4,
                "full_rows": 4,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 0.125,
                "when": "after",
                "capture_phase": "retire_cycle",
            },
            {
                "node_name": "block0_head0_qkt__accum_pre_softmax_next",
                "buf_id": BUF_ACCUM,
                "offset_units": 0,
                "mem_rows": 2,
                "mem_cols": 4,
                "logical_rows": 2,
                "logical_cols": 4,
                "full_rows": 4,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int32",
                "scale": 0.125,
                "when": "after",
                "capture_phase": "retire_plus_1",
            },
        ],
    }
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())

    golden_tensor = np.array([[101, 102, 103, 104], [105, 106, 107, 108]], dtype=np.int32)
    golden_trace_path = tmp_path / "golden_trace.json"
    golden_trace_path.write_text(
        json.dumps(
            {
                "raw_events": [
                    {
                        "pc": 1,
                        "event_index": 0,
                        "node_name": "block0_head0_qkt__accum_pre_matmul",
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_cycle",
                        "row_start": 0,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "raw_available": True,
                        "raw": np.zeros((2, 4), dtype=np.int32).tolist(),
                    },
                    {
                        "pc": 1,
                        "event_index": 1,
                        "node_name": "block0_head0_qkt__accum_pre_matmul_next",
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_plus_1",
                        "row_start": 0,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "raw_available": True,
                        "raw": np.zeros((2, 4), dtype=np.int32).tolist(),
                    },
                    {
                        "pc": 2,
                        "event_index": 0,
                        "node_name": "block0_head0_qkt",
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_cycle",
                        "row_start": 0,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "raw_available": True,
                        "raw": golden_tensor.tolist(),
                    },
                    {
                        "pc": 3,
                        "event_index": 0,
                        "node_name": "block0_head0_qkt__accum_pre_softmax",
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_cycle",
                        "row_start": 0,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "raw_available": True,
                        "raw": golden_tensor.tolist(),
                    },
                    {
                        "pc": 3,
                        "event_index": 1,
                        "node_name": "block0_head0_qkt__accum_pre_softmax_next",
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_plus_1",
                        "row_start": 0,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "raw_available": True,
                        "raw": golden_tensor.tolist(),
                    },
                ]
            }
        )
    )

    accum_pre_bad = np.array([[7, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    accum_pre_good = np.zeros((2, 4), dtype=np.int32)
    snapshot_blob = (
        accum_pre_bad.astype("<i4").tobytes()
        + accum_pre_good.astype("<i4").tobytes()
        + golden_tensor.astype("<i4").tobytes()
        + golden_tensor.astype("<i4").tobytes()
        + golden_tensor.astype("<i4").tobytes()
    )
    snapshot_data_path = tmp_path / "rtl_snapshot_data.bin"
    snapshot_data_path.write_bytes(snapshot_blob)

    size_i32 = 2 * 4 * 4
    snapshot_manifest_path = tmp_path / "rtl_snapshot_manifest.json"
    snapshot_manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "pc": 1,
                        "event_index": 0,
                        "node_name": "block0_head0_qkt__accum_pre_matmul",
                        "buf_id": BUF_ACCUM,
                        "offset_units": 0,
                        "mem_rows": 2,
                        "mem_cols": 4,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_cycle",
                        "status": "captured",
                        "cycle": 10,
                        "byte_offset": 0,
                        "byte_size": size_i32,
                    },
                    {
                        "pc": 1,
                        "event_index": 1,
                        "node_name": "block0_head0_qkt__accum_pre_matmul_next",
                        "buf_id": BUF_ACCUM,
                        "offset_units": 0,
                        "mem_rows": 2,
                        "mem_cols": 4,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_plus_1",
                        "status": "captured",
                        "cycle": 11,
                        "byte_offset": size_i32,
                        "byte_size": size_i32,
                    },
                    {
                        "pc": 2,
                        "event_index": 0,
                        "node_name": "block0_head0_qkt",
                        "buf_id": BUF_ACCUM,
                        "offset_units": 0,
                        "mem_rows": 2,
                        "mem_cols": 4,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_cycle",
                        "status": "captured",
                        "cycle": 12,
                        "byte_offset": size_i32 * 2,
                        "byte_size": size_i32,
                    },
                    {
                        "pc": 3,
                        "event_index": 0,
                        "node_name": "block0_head0_qkt__accum_pre_softmax",
                        "buf_id": BUF_ACCUM,
                        "offset_units": 0,
                        "mem_rows": 2,
                        "mem_cols": 4,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_cycle",
                        "status": "captured",
                        "cycle": 13,
                        "byte_offset": size_i32 * 3,
                        "byte_size": size_i32,
                    },
                    {
                        "pc": 3,
                        "event_index": 1,
                        "node_name": "block0_head0_qkt__accum_pre_softmax_next",
                        "buf_id": BUF_ACCUM,
                        "offset_units": 0,
                        "mem_rows": 2,
                        "mem_cols": 4,
                        "logical_rows": 2,
                        "logical_cols": 4,
                        "full_rows": 4,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int32",
                        "scale": 0.125,
                        "source": "architectural",
                        "capture_phase": "retire_plus_1",
                        "status": "captured",
                        "cycle": 14,
                        "byte_offset": size_i32 * 4,
                        "byte_size": size_i32,
                    },
                ]
            }
        )
    )

    first_divergence_path = tmp_path / "first_divergence.json"
    first_divergence_path.write_text(
        json.dumps(
            {
                "node_name": "block0_head0_qkt__accum_pre_matmul",
                "trace_pc": 1,
                "artifacts": {
                    "program": str(program_path),
                    "golden_trace": str(golden_trace_path),
                    "snapshot_manifest": str(snapshot_manifest_path),
                    "snapshot_data": str(snapshot_data_path),
                },
            }
        )
    )

    out_path = tmp_path / "qkt_stability_report.json"
    report = compare_rtl_golden.emit_qkt_stability_report(
        first_divergence_path=first_divergence_path,
        out_path=out_path,
        strip_row_start=0,
    )

    assert out_path.exists()
    assert report["classification"] == "retire_cycle_snapshot_artifact"
    assert report["checkpoints"]["accum_pre_matmul"]["samples"]["0,0"] == {"golden": 0, "rtl": 7}
    assert report["checkpoints"]["accum_pre_matmul_next"]["samples"]["0,0"] == {"golden": 0, "rtl": 0}


def test_extract_qkt_replay_payloads_writes_expected_strip_payloads(tmp_path: Path):
    program = Assembler().assemble("NOP\nHALT\n")
    def event(
        node_name: str,
        buf_id: int,
        offset_units: int,
        rows: int,
        cols: int,
        dtype: str,
        scale: float,
        *,
        pc: int,
        event_index: int,
        logical_rows: int | None = None,
        logical_cols: int | None = None,
        full_rows: int | None = None,
        full_cols: int | None = None,
        row_start: int = 0,
    ) -> tuple[dict[str, object], tuple[int, int]]:
        trace_event = {
            "node_name": node_name,
            "buf_id": buf_id,
            "offset_units": offset_units,
            "mem_rows": rows,
            "mem_cols": cols,
            "logical_rows": rows if logical_rows is None else logical_rows,
            "logical_cols": cols if logical_cols is None else logical_cols,
            "full_rows": rows if full_rows is None else full_rows,
            "full_cols": cols if full_cols is None else full_cols,
            "row_start": row_start,
            "dtype": dtype,
            "scale": scale,
            "when": "after",
        }
        return trace_event, (pc, event_index)

    trace_manifest: dict[int, list[dict[str, object]]] = {}
    snapshot_entries: list[dict[str, object]] = []
    snapshot_bytes = bytearray()

    def add_snapshot(
        trace_event: dict[str, object],
        pc: int,
        event_index: int,
        payload: bytes,
    ) -> None:
        trace_manifest.setdefault(pc, []).append(trace_event)
        snapshot_entries.append(
            {
                "pc": pc,
                "event_index": event_index,
                "node_name": trace_event["node_name"],
                "buf_id": trace_event["buf_id"],
                "offset_units": trace_event["offset_units"],
                "mem_rows": trace_event["mem_rows"],
                "mem_cols": trace_event["mem_cols"],
                "logical_rows": trace_event["logical_rows"],
                "logical_cols": trace_event["logical_cols"],
                "full_rows": trace_event["full_rows"],
                "full_cols": trace_event["full_cols"],
                "row_start": trace_event["row_start"],
                "dtype": trace_event["dtype"],
                "scale": trace_event["scale"],
                "source": "architectural",
                "status": "captured",
                "cycle": 1,
                "byte_offset": len(snapshot_bytes),
                "byte_size": len(payload),
            }
        )
        snapshot_bytes.extend(payload)

    def add_trace_only(trace_event: dict[str, object], pc: int) -> None:
        trace_manifest.setdefault(pc, []).append(trace_event)

    query_act = (np.arange(16, dtype=np.int16).reshape(4, 4) - 10).astype(np.int8)
    query_weight = (np.arange(16, dtype=np.int16).reshape(4, 4) + 20).astype(np.int8)
    query_accum_pre_bias = np.array(
        [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]],
        dtype=np.int32,
    )
    query_bias = np.array([[31, 32, 33, 34]], dtype=np.int32)
    query_accum = query_accum_pre_bias + query_bias
    query_output = (np.arange(16, dtype=np.int16).reshape(4, 4) - 30).astype(np.int8)
    query_act_padded = np.vstack([query_act, np.zeros((2, 4), dtype=np.int8)])
    query_accum_pre_bias_padded = np.vstack([query_accum_pre_bias, np.zeros((2, 4), dtype=np.int32)])
    query_accum_padded = np.vstack([query_accum, np.zeros((2, 4), dtype=np.int32)])
    query_output_padded = np.vstack([query_output, np.zeros((2, 4), dtype=np.int8)])
    pos_embed_act_input = (np.arange(16, dtype=np.int16).reshape(4, 4) - 3).astype(np.int8)
    pos_embed_pos_input = (np.arange(16, dtype=np.int16).reshape(4, 4) + 7).astype(np.int8)
    pos_embed_output = (pos_embed_act_input + pos_embed_pos_input).astype(np.int8)
    ln1_input_padded = np.vstack([pos_embed_output, np.zeros((2, 4), dtype=np.int8)])
    ln1_output_padded = np.vstack([
        query_act,
        np.array([[-1, 1, 8, 0], [-5, 10, 0, -7]], dtype=np.int8),
    ])

    key_act = (np.arange(16, dtype=np.int16).reshape(4, 4) - 8).astype(np.int8)
    key_weight = (np.arange(16, dtype=np.int16).reshape(4, 4) + 10).astype(np.int8)
    key_accum_pre_bias = np.array(
        [[41, 42, 43, 44], [45, 46, 47, 48], [49, 50, 51, 52], [53, 54, 55, 56]],
        dtype=np.int32,
    )
    key_bias = np.array([[61, 62, 63, 64]], dtype=np.int32)
    key_accum = key_accum_pre_bias + key_bias
    key_output = (np.arange(16, dtype=np.int16).reshape(4, 4) + 40).astype(np.int8)
    key_act_padded = np.vstack([key_act, np.zeros((2, 4), dtype=np.int8)])
    key_accum_pre_bias_padded = np.vstack([key_accum_pre_bias, np.zeros((2, 4), dtype=np.int32)])
    key_accum_padded = np.vstack([key_accum, np.zeros((2, 4), dtype=np.int32)])
    key_output_padded = np.vstack([key_output, np.zeros((2, 4), dtype=np.int8)])

    value_act = (np.arange(16, dtype=np.int16).reshape(4, 4) + 5).astype(np.int8)
    value_weight = (np.arange(16, dtype=np.int16).reshape(4, 4) - 20).astype(np.int8)
    value_accum_pre_bias = np.array(
        [[71, 72, 73, 74], [75, 76, 77, 78], [79, 80, 81, 82], [83, 84, 85, 86]],
        dtype=np.int32,
    )
    value_bias = np.array([[91, 92, 93, 94]], dtype=np.int32)
    value_accum = value_accum_pre_bias + value_bias
    value_output = (np.arange(16, dtype=np.int16).reshape(4, 4) - 40).astype(np.int8)
    value_act_padded = np.vstack([value_act, np.zeros((2, 4), dtype=np.int8)])
    value_accum_pre_bias_padded = np.vstack([value_accum_pre_bias, np.zeros((2, 4), dtype=np.int32)])
    value_accum_padded = np.vstack([value_accum, np.zeros((2, 4), dtype=np.int32)])
    value_output_padded = np.vstack([value_output, np.zeros((2, 4), dtype=np.int8)])

    ln1_gamma = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    ln1_beta = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float16)
    startup_cls_token = pos_embed_act_input[:1].copy()
    startup_patch_input = pos_embed_act_input[1:].copy()
    data_blob = bytearray(256)
    offset_map = {
        "vit.encoder.layer.0.layernorm_before.weight": 0,
        "vit.encoder.layer.0.layernorm_before.bias": 8,
        "vit.encoder.layer.0.attention.attention.query.weight_h0": 16,
        "vit.encoder.layer.0.attention.attention.query.bias_h0": 32,
        "vit.encoder.layer.0.attention.attention.key.weight_h0": 48,
        "vit.encoder.layer.0.attention.attention.key.bias_h0": 64,
        "vit.encoder.layer.0.attention.attention.value.weight_h0": 80,
        "vit.encoder.layer.0.attention.attention.value.bias_h0": 96,
        "__zero_pad__": 112,
        "vit.embeddings.cls_token": 128,
        "vit.embeddings.position_embeddings": 144,
    }
    data_blob[offset_map["vit.encoder.layer.0.layernorm_before.weight"]:offset_map["vit.encoder.layer.0.layernorm_before.weight"] + len(ln1_gamma.tobytes())] = ln1_gamma.tobytes()
    data_blob[offset_map["vit.encoder.layer.0.layernorm_before.bias"]:offset_map["vit.encoder.layer.0.layernorm_before.bias"] + len(ln1_beta.tobytes())] = ln1_beta.tobytes()
    data_blob[offset_map["vit.encoder.layer.0.attention.attention.query.weight_h0"]:offset_map["vit.encoder.layer.0.attention.attention.query.weight_h0"] + query_weight.nbytes] = query_weight.tobytes()
    data_blob[offset_map["vit.encoder.layer.0.attention.attention.query.bias_h0"]:offset_map["vit.encoder.layer.0.attention.attention.query.bias_h0"] + query_bias.nbytes] = query_bias.astype("<i4", copy=False).tobytes()
    data_blob[offset_map["vit.encoder.layer.0.attention.attention.key.weight_h0"]:offset_map["vit.encoder.layer.0.attention.attention.key.weight_h0"] + key_weight.nbytes] = key_weight.tobytes()
    data_blob[offset_map["vit.encoder.layer.0.attention.attention.key.bias_h0"]:offset_map["vit.encoder.layer.0.attention.attention.key.bias_h0"] + key_bias.nbytes] = key_bias.astype("<i4", copy=False).tobytes()
    data_blob[offset_map["vit.encoder.layer.0.attention.attention.value.weight_h0"]:offset_map["vit.encoder.layer.0.attention.attention.value.weight_h0"] + value_weight.nbytes] = value_weight.tobytes()
    data_blob[offset_map["vit.encoder.layer.0.attention.attention.value.bias_h0"]:offset_map["vit.encoder.layer.0.attention.attention.value.bias_h0"] + value_bias.nbytes] = value_bias.astype("<i4", copy=False).tobytes()
    data_blob[offset_map["vit.embeddings.cls_token"]:offset_map["vit.embeddings.cls_token"] + startup_cls_token.nbytes] = startup_cls_token.tobytes()
    startup_pos_input_padded = np.vstack(
        [pos_embed_pos_input, np.zeros((2, pos_embed_pos_input.shape[1]), dtype=np.int8)]
    )
    data_blob[
        offset_map["vit.embeddings.position_embeddings"]:
        offset_map["vit.embeddings.position_embeddings"] + startup_pos_input_padded.nbytes
    ] = startup_pos_input_padded.tobytes()
    program.data = bytes(data_blob)
    program.data_base = 128
    program.cls_token_dram_offset = program.data_base + offset_map["vit.embeddings.cls_token"]
    program.pos_embed_cls_dram_offset = program.data_base + offset_map["vit.embeddings.position_embeddings"]
    program.pos_embed_patch_dram_offset = program.pos_embed_cls_dram_offset + pos_embed_pos_input.shape[1]
    program.input_offset = 512
    program.compiler_manifest = {
        "program_layout": {
            "data_base": 128,
            "dram_layout": offset_map,
        },
        "weights": {
            "vit.embeddings.position_embeddings": {"dtype": "int8", "stored_shape": [6, 4]},
            "vit.encoder.layer.0.layernorm_before.weight": {"dtype": "float16", "stored_shape": [4]},
            "vit.encoder.layer.0.layernorm_before.bias": {"dtype": "float16", "stored_shape": [4]},
        },
    }

    key_t = np.arange(16, dtype=np.int8).reshape(4, 4)
    key_padded = (np.arange(16, dtype=np.int16).reshape(4, 4) + 60).astype(np.int8)
    accum_pre = np.array([[11, 12, 13, 14], [15, 16, 17, 18]], dtype=np.int32)
    query = np.array([[21, 22, 23, 24], [25, 26, 27, 28]], dtype=np.int8)

    snapshot_specs = [
        ("pos_embed_add__act_input", BUF_ABUF, 0, pos_embed_act_input, "int8", 0.125, 4, 0),
        ("pos_embed_add__pos_input", BUF_WBUF, 0, pos_embed_pos_input, "int8", 0.125, 4, 1),
        ("pos_embed_add", BUF_ABUF, 32, pos_embed_output, "int8", 0.125, 5, 0),
        ("block0_ln1__input_padded", BUF_ABUF, 30, ln1_input_padded, "int8", 0.125, 6, 0),
        ("block0_head0_query__act_input", BUF_ABUF, 6, query_act, "int8", 0.5, 0, 0),
        ("block0_head0_query__act_input_padded", BUF_ABUF, 6, query_act_padded, "int8", 0.5, 0, 1),
        ("block0_head0_query__weight_input", BUF_WBUF, 10, query_weight, "int8", 0.25, 0, 2),
        ("block0_head0_query__accum_pre_bias", BUF_ACCUM, 0, query_accum_pre_bias, "int32", 0.125, 0, 3),
        ("block0_head0_query__accum_pre_bias_padded", BUF_ACCUM, 0, query_accum_pre_bias_padded, "int32", 0.125, 0, 4),
        ("block0_head0_query__bias_input", BUF_WBUF, 18, query_bias, "int32", 0.125, 0, 5),
        ("block0_head0_query__accum", BUF_ACCUM, 0, query_accum, "int32", 0.125, 0, 6),
        ("block0_head0_query__accum_padded", BUF_ACCUM, 0, query_accum_padded, "int32", 0.125, 0, 7),
        ("block0_head0_query", BUF_ABUF, 4, query_output, "int8", 0.5, 0, 8),
        ("block0_head0_query__output_padded", BUF_ABUF, 4, query_output_padded, "int8", 0.5, 0, 9),
        ("block0_head0_key__act_input", BUF_ABUF, 8, key_act, "int8", 0.75, 1, 0),
        ("block0_head0_key__act_input_padded", BUF_ABUF, 8, key_act_padded, "int8", 0.75, 1, 1),
        ("block0_head0_key__weight_input", BUF_WBUF, 12, key_weight, "int8", 0.125, 1, 2),
        ("block0_head0_key__accum_pre_bias", BUF_ACCUM, 0, key_accum_pre_bias, "int32", 0.25, 1, 3),
        ("block0_head0_key__accum_pre_bias_padded", BUF_ACCUM, 0, key_accum_pre_bias_padded, "int32", 0.25, 1, 4),
        ("block0_head0_key__bias_input", BUF_WBUF, 20, key_bias, "int32", 0.25, 1, 5),
        ("block0_head0_key__accum", BUF_ACCUM, 0, key_accum, "int32", 0.25, 1, 6),
        ("block0_head0_key__accum_padded", BUF_ACCUM, 0, key_accum_padded, "int32", 0.25, 1, 7),
        ("block0_head0_key", BUF_ABUF, 16, key_output, "int8", 0.125, 1, 8),
        ("block0_head0_key__output_padded", BUF_ABUF, 16, key_output_padded, "int8", 0.125, 1, 9),
        ("block0_head0_value__act_input", BUF_ABUF, 24, value_act, "int8", 0.625, 2, 0),
        ("block0_head0_value__act_input_padded", BUF_ABUF, 24, value_act_padded, "int8", 0.625, 2, 1),
        ("block0_head0_value__weight_input", BUF_WBUF, 14, value_weight, "int8", 0.375, 2, 2),
        ("block0_head0_value__accum_pre_bias", BUF_ACCUM, 0, value_accum_pre_bias, "int32", 0.1875, 2, 3),
        ("block0_head0_value__accum_pre_bias_padded", BUF_ACCUM, 0, value_accum_pre_bias_padded, "int32", 0.1875, 2, 4),
        ("block0_head0_value__bias_input", BUF_WBUF, 22, value_bias, "int32", 0.1875, 2, 5),
        ("block0_head0_value__accum", BUF_ACCUM, 0, value_accum, "int32", 0.1875, 2, 6),
        ("block0_head0_value__accum_padded", BUF_ACCUM, 0, value_accum_padded, "int32", 0.1875, 2, 7),
        ("block0_head0_value", BUF_ABUF, 28, value_output, "int8", 0.375, 2, 8),
        ("block0_head0_value__output_padded", BUF_ABUF, 28, value_output_padded, "int8", 0.375, 2, 9),
        ("block0_ln1__output_padded", BUF_ABUF, 32, ln1_output_padded, "int8", 0.25, 7, 0),
        ("block0_head0_qkt__key_transposed", BUF_WBUF, 0, key_t, "int8", 0.25, 3, 0),
        ("block0_head0_qkt__key_padded_input", BUF_ABUF, 16, key_padded, "int8", 0.25, 3, 1),
        ("block0_head0_qkt__accum_pre_matmul", BUF_ACCUM, 0, accum_pre, "int32", 0.125, 3, 2),
        ("block0_head0_qkt__query_input", BUF_ABUF, 4, query, "int8", 0.5, 4, 2),
    ]

    for node_name, buf_id, offset_units, tensor, dtype, scale, pc, event_index in snapshot_specs:
        rows, cols = tensor.shape
        trace_event, _ = event(node_name, buf_id, offset_units, rows, cols, dtype, scale, pc=pc, event_index=event_index)
        payload = tensor.astype(np.int8 if dtype == "int8" else "<i4", copy=False).tobytes()
        add_snapshot(trace_event, pc, event_index, payload)

    qkt_trace_event, _ = event(
        "block0_head0_qkt",
        BUF_ACCUM,
        0,
        2,
        4,
        "int32",
        0.125,
        pc=8,
        event_index=0,
        logical_rows=2,
        logical_cols=4,
        full_rows=2,
        full_cols=4,
    )
    add_trace_only(qkt_trace_event, 8)
    program.trace_manifest = trace_manifest
    program_path = tmp_path / "program.bin"
    program_path.write_bytes(program.to_bytes())

    golden_trace_path = tmp_path / "golden_trace.json"
    golden_trace = {
        "raw_events": [
                {
                    "pc": 8,
                    "event_index": 0,
                    "node_name": "block0_head0_qkt",
                    "dtype": "int32",
                    "scale": 0.125,
                "source": "architectural",
                "row_start": 0,
                "logical_rows": 2,
                "logical_cols": 4,
                "full_rows": 4,
                "full_cols": 4,
                "raw_available": True,
                "raw": [101, 102, 103, 104, 105, 106, 107, 108],
            }
        ]
    }
    golden_trace_path.write_text(json.dumps(golden_trace))
    snapshot_data_path = tmp_path / "rtl_snapshot_data.bin"
    snapshot_data_path.write_bytes(bytes(snapshot_bytes))

    snapshot_manifest_path = tmp_path / "rtl_snapshot_manifest.json"
    snapshot_manifest = {"entries": snapshot_entries}
    snapshot_manifest_path.write_text(json.dumps(snapshot_manifest))

    first_divergence_path = tmp_path / "first_divergence.json"
    first_divergence_path.write_text(
        json.dumps(
            {
                "node_name": "block0_head0_qkt",
                "artifacts": {
                    "program": str(program_path),
                    "golden_trace": str(golden_trace_path),
                    "snapshot_manifest": str(snapshot_manifest_path),
                    "snapshot_data": str(snapshot_data_path),
                    "work_dir": str(tmp_path / "work"),
                },
            }
        )
    )

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "patches.raw").write_bytes(startup_patch_input.tobytes())

    out_dir = tmp_path / "replay_payloads"
    result = compare_rtl_golden.extract_qkt_replay_payloads(
        first_divergence_path=first_divergence_path,
        out_dir=out_dir,
        strip_row_start=0,
    )

    assert result["query_shape"] == [2, 4]
    assert result["key_transposed_shape"] == [4, 4]
    assert result["key_padded_shape"] == [4, 4]
    assert result["accum_pre_shape"] == [2, 4]
    assert result["startup_cls_token_shape"] == [1, 4]
    assert result["startup_patch_input_shape"] == [3, 4]
    assert result["startup_cls_dram_offset"] == program.cls_token_dram_offset
    assert result["startup_patch_dram_offset"] == program.input_offset
    assert result["startup_pos_dram_offset"] == program.pos_embed_cls_dram_offset
    assert result["startup_cls_dst_offset_units"] == 0
    assert result["startup_patch_dst_offset_units"] == 1
    assert result["startup_pos_wbuf_offset_units"] == 0
    assert result["startup_patch_rows"] == 3
    assert result["startup_cols"] == 4
    assert result["startup_pos_input_padded_shape"] == [6, 4]
    assert result["startup_pos_input_padded_row_units"] == 2
    assert result["startup_issue_pcs"] == [2, 6, 10, 12, 13]
    assert result["pos_embed_add_act_input_shape"] == [4, 4]
    assert result["pos_embed_add_pos_input_shape"] == [4, 4]
    assert result["pos_embed_add_shape"] == [4, 4]
    assert result["pos_embed_add_rows"] == 4
    assert result["pos_embed_add_cols"] == 4
    assert result["pos_embed_add_act_input_offset_units"] == 0
    assert result["pos_embed_add_pos_input_offset_units"] == 0
    assert result["pos_embed_add_output_offset_units"] == 32
    assert result["pos_embed_add_scale"] == pytest.approx(0.125)
    assert result["ln1_input_padded_shape"] == [6, 4]
    assert result["ln1_output_padded_shape"] == [6, 4]
    assert result["ln1_gamma_shape"] == [4]
    assert result["ln1_beta_shape"] == [4]
    assert result["ln1_gamma_beta_shape"] == [2, 4]
    assert result["query_act_input_rows"] == 4
    assert result["query_output_offset_units"] == 4
    assert result["ln1_gamma_dram_offset"] == 0
    assert result["ln1_beta_dram_offset"] == 8
    assert result["dram_layout_offsets_are_data_relative"] is True
    assert result["ln1_gamma_dram_offset_space"] == "program_data_relative"
    assert result["ln1_beta_dram_offset_space"] == "program_data_relative"
    assert result["ln1_gamma_preview"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result["ln1_beta_preview"] == pytest.approx([-1.0, -2.0, -3.0, -4.0])
    assert result["query_weight_dram_offset"] == 16
    assert result["query_bias_dram_offset"] == 32
    assert result["key_weight_dram_offset"] == 48
    assert result["value_bias_dram_offset"] == 96
    assert result["query_requant_scale_fp16"] == compare_rtl_golden._fp16_to_uint16(0.125 / 0.5)
    assert result["key_projection_act_shape"] == [4, 4]
    assert result["query_projection_weight_shape"] == [4, 4]
    assert result["key_projection_weight_shape"] == [4, 4]
    assert result["value_projection_weight_shape"] == [4, 4]
    assert result["query_act_input_padded_shape"] == [6, 4]
    assert result["query_accum_pre_bias_padded_shape"] == [6, 4]
    assert result["key_accum_post_bias_padded_shape"] == [6, 4]
    assert result["value_output_padded_shape"] == [6, 4]
    assert result["golden_qkt_shape"] == [2, 4]
    assert result["query_input_offset_units"] == 4
    assert result["key_padded_input_offset_units"] == 16
    assert result["key_projection_act_offset_units"] == 8
    assert result["query_projection_weight_offset_units"] == 10
    assert result["key_projection_weight_offset_units"] == 12
    assert result["value_projection_weight_offset_units"] == 14
    np.testing.assert_array_equal(np.fromfile(out_dir / "query_input.raw", dtype=np.int8).reshape(2, 4), query)
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "startup_cls_token.raw", dtype=np.int8).reshape(1, 4),
        startup_cls_token,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "startup_patch_input.raw", dtype=np.int8).reshape(3, 4),
        startup_patch_input,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "startup_pos_input_padded.raw", dtype=np.int8).reshape(6, 4),
        startup_pos_input_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "pos_embed_add_act_input.raw", dtype=np.int8).reshape(4, 4),
        pos_embed_act_input,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "pos_embed_add_pos_input.raw", dtype=np.int8).reshape(4, 4),
        pos_embed_pos_input,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "pos_embed_add_output.raw", dtype=np.int8).reshape(4, 4),
        pos_embed_output,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "ln1_input_padded.raw", dtype=np.int8).reshape(6, 4),
        ln1_input_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "ln1_output_padded.raw", dtype=np.int8).reshape(6, 4),
        ln1_output_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "ln1_gamma.raw", dtype=np.float16),
        ln1_gamma,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "ln1_beta.raw", dtype=np.float16),
        ln1_beta,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "ln1_gamma_beta.raw", dtype=np.float16).reshape(2, 4),
        np.stack([ln1_gamma, ln1_beta], axis=0),
    )
    np.testing.assert_array_equal(np.fromfile(out_dir / "key_transposed.raw", dtype=np.int8).reshape(4, 4), key_t)
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_padded_input.raw", dtype=np.int8).reshape(4, 4), key_padded
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "accum_pre_matmul.raw", dtype="<i4").astype(np.int32).reshape(2, 4),
        accum_pre,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_projection_act_input.raw", dtype=np.int8).reshape(4, 4),
        key_act,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_projection_weight_input.raw", dtype=np.int8).reshape(4, 4),
        query_weight,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_projection_weight_input.raw", dtype=np.int8).reshape(4, 4),
        key_weight,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_projection_weight_input.raw", dtype=np.int8).reshape(4, 4),
        value_weight,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_accum_pre_bias.raw", dtype="<i4").astype(np.int32).reshape(4, 4),
        query_accum_pre_bias,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_bias_input.raw", dtype="<i4").astype(np.int32).reshape(1, 4),
        query_bias,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_accum_post_bias.raw", dtype="<i4").astype(np.int32).reshape(4, 4),
        query_accum,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_output.raw", dtype=np.int8).reshape(4, 4),
        query_output,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_act_input_padded.raw", dtype=np.int8).reshape(6, 4),
        query_act_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_accum_pre_bias_padded.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        query_accum_pre_bias_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_accum_pre_bias_padded_golden.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        query_accum_pre_bias_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_accum_post_bias_padded.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        query_accum_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "query_output_padded.raw", dtype=np.int8).reshape(6, 4),
        query_output_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_bias_input.raw", dtype="<i4").astype(np.int32).reshape(1, 4),
        key_bias,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_output.raw", dtype=np.int8).reshape(4, 4),
        key_output,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_act_input_padded.raw", dtype=np.int8).reshape(6, 4),
        key_act_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_accum_pre_bias_padded.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        key_accum_pre_bias_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_accum_pre_bias_padded_golden.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        key_accum_pre_bias_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_accum_post_bias_padded.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        key_accum_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "key_output_padded.raw", dtype=np.int8).reshape(6, 4),
        key_output_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_projection_act_input.raw", dtype=np.int8).reshape(4, 4),
        value_act,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_bias_input.raw", dtype="<i4").astype(np.int32).reshape(1, 4),
        value_bias,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_output.raw", dtype=np.int8).reshape(4, 4),
        value_output,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_act_input_padded.raw", dtype=np.int8).reshape(6, 4),
        value_act_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_accum_pre_bias_padded.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        value_accum_pre_bias_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_accum_pre_bias_padded_golden.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        value_accum_pre_bias_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_accum_post_bias_padded.raw", dtype="<i4").astype(np.int32).reshape(6, 4),
        value_accum_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "value_output_padded.raw", dtype=np.int8).reshape(6, 4),
        value_output_padded,
    )
    np.testing.assert_array_equal(
        np.fromfile(out_dir / "golden_qkt.raw", dtype="<i4").astype(np.int32).reshape(2, 4),
        np.array([[101, 102, 103, 104], [105, 106, 107, 108]], dtype=np.int32),
    )

    projection_first_divergence_path = tmp_path / "projection_first_divergence.json"
    projection_first_divergence_path.write_text(
        json.dumps(
            {
                "node_name": "block0_head0_query__accum_pre_bias_padded",
                "artifacts": {
                    "program": str(program_path),
                    "golden_trace": str(golden_trace_path),
                    "snapshot_manifest": str(snapshot_manifest_path),
                    "snapshot_data": str(snapshot_data_path),
                },
            }
        )
    )
    projection_out_dir = tmp_path / "projection_replay_payloads"
    projection_result = compare_rtl_golden.extract_qkt_replay_payloads(
        first_divergence_path=projection_first_divergence_path,
        out_dir=projection_out_dir,
        strip_row_start=0,
    )
    assert projection_result["query_shape"] == [2, 4]
    assert projection_result["query_accum_pre_bias_padded_shape"] == [6, 4]

    ln1_first_divergence_path = tmp_path / "ln1_first_divergence.json"
    ln1_first_divergence_path.write_text(
        json.dumps(
            {
                "node_name": "block0_ln1__input_padded",
                "artifacts": {
                    "program": str(program_path),
                    "golden_trace": str(golden_trace_path),
                    "snapshot_manifest": str(snapshot_manifest_path),
                    "snapshot_data": str(snapshot_data_path),
                },
            }
        )
    )
    ln1_result = compare_rtl_golden.extract_qkt_replay_payloads(
        first_divergence_path=ln1_first_divergence_path,
        out_dir=tmp_path / "ln1_replay_payloads",
        strip_row_start=0,
    )
    assert ln1_result["block_prefix"] == "block0"
    assert ln1_result["ln1_input_padded_shape"] == [6, 4]


def test_projection_padding_report_classifies_dirty_regions(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    metadata = {
        "dtype_map": {
            "query_act_input_padded": "int8",
            "query_accum_pre_bias_padded": "int32",
            "query_accum_post_bias_padded": "int32",
            "query_output_padded": "int8",
            "key_act_input_padded": "int8",
            "key_accum_pre_bias_padded": "int32",
            "key_accum_post_bias_padded": "int32",
            "key_output_padded": "int8",
            "value_act_input_padded": "int8",
            "value_accum_pre_bias_padded": "int32",
            "value_accum_post_bias_padded": "int32",
            "value_output_padded": "int8",
        },
        "query_act_input_rows": 4,
        "query_act_input_cols": 4,
        "query_act_input_padded_rows": 6,
        "query_act_input_padded_cols": 4,
        "query_accum_post_bias_rows": 4,
        "query_accum_post_bias_cols": 4,
        "query_accum_pre_bias_padded_rows": 6,
        "query_accum_pre_bias_padded_cols": 4,
        "query_accum_post_bias_padded_rows": 6,
        "query_accum_post_bias_padded_cols": 4,
        "query_output_padded_rows": 6,
        "query_output_padded_cols": 4,
        "key_act_input_rows": 4,
        "key_act_input_cols": 4,
        "key_act_input_padded_rows": 6,
        "key_act_input_padded_cols": 4,
        "key_accum_post_bias_rows": 4,
        "key_accum_post_bias_cols": 4,
        "key_accum_pre_bias_padded_rows": 6,
        "key_accum_pre_bias_padded_cols": 4,
        "key_accum_post_bias_padded_rows": 6,
        "key_accum_post_bias_padded_cols": 4,
        "key_output_padded_rows": 6,
        "key_output_padded_cols": 4,
        "value_act_input_rows": 4,
        "value_act_input_cols": 4,
        "value_act_input_padded_rows": 6,
        "value_act_input_padded_cols": 4,
        "value_accum_post_bias_rows": 4,
        "value_accum_post_bias_cols": 4,
        "value_accum_pre_bias_padded_rows": 6,
        "value_accum_pre_bias_padded_cols": 4,
        "value_accum_post_bias_padded_rows": 6,
        "value_accum_post_bias_padded_cols": 4,
        "value_output_padded_rows": 6,
        "value_output_padded_cols": 4,
    }
    (replay_dir / "replay_metadata.json").write_text(json.dumps(metadata))

    def write_i32(name: str, arr: np.ndarray) -> None:
        (replay_dir / f"{name}.raw").write_bytes(arr.astype("<i4", copy=False).tobytes())

    def write_i8(name: str, arr: np.ndarray) -> None:
        (replay_dir / f"{name}.raw").write_bytes(arr.astype(np.int8, copy=False).tobytes())

    clean_i32 = np.zeros((6, 4), dtype=np.int32)
    clean_i8 = np.zeros((6, 4), dtype=np.int8)

    query_act = clean_i8.copy()
    query_act[4, 0] = np.int8(9)
    write_i8("query_act_input_padded", query_act)
    write_i32("query_accum_pre_bias_padded", clean_i32)
    write_i32("query_accum_post_bias_padded", clean_i32)
    write_i8("query_output_padded", clean_i8)

    key_pre = clean_i32.copy()
    key_pre[4, 1] = 7
    write_i8("key_act_input_padded", clean_i8)
    write_i32("key_accum_pre_bias_padded", key_pre)
    write_i32("key_accum_post_bias_padded", clean_i32)
    write_i8("key_output_padded", clean_i8)

    value_pre = clean_i32.copy()
    value_pre[4, 2] = 5
    write_i8("value_act_input_padded", clean_i8)
    write_i32("value_accum_pre_bias_padded", value_pre)
    write_i32("value_accum_post_bias_padded", clean_i32)
    write_i8("value_output_padded", clean_i8)

    (replay_dir / "projection_replay_results.json").write_text(
        json.dumps(
            {
                "query": {"exact_padded_match": True, "clean_padded_match": True},
                "key": {"exact_padded_match": True, "clean_padded_match": False},
                "value": {"exact_padded_match": True, "clean_padded_match": True},
            }
        )
    )

    report = compare_rtl_golden.emit_projection_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=tmp_path / "projection_padding_report.json",
    )

    assert report["overall_classification"] == "dirty_source_padding"
    assert report["projections"]["query"]["classification"] == "dirty_source_padding"
    assert report["projections"]["key"]["classification"] == "projection_matmul_or_drain_touches_padding"
    assert report["projections"]["value"]["classification"] == "full_program_sequencing_or_snapshot_gap"
    assert report["projections"]["query"]["act_input_padded"]["first_nonzero_padded_coord"] == [4, 0]
    assert report["projections"]["key"]["clean_padded_match"] is False
    assert report["projections"]["value"]["clean_padded_match"] is True


def test_projection_padding_report_classifies_clean_padding(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    metadata = {
        "dtype_map": {
            "query_act_input_padded": "int8",
            "query_accum_pre_bias_padded": "int32",
            "query_accum_post_bias_padded": "int32",
            "query_output_padded": "int8",
            "key_act_input_padded": "int8",
            "key_accum_pre_bias_padded": "int32",
            "key_accum_post_bias_padded": "int32",
            "key_output_padded": "int8",
            "value_act_input_padded": "int8",
            "value_accum_pre_bias_padded": "int32",
            "value_accum_post_bias_padded": "int32",
            "value_output_padded": "int8",
        },
        "query_act_input_rows": 4,
        "query_act_input_cols": 4,
        "query_act_input_padded_rows": 6,
        "query_act_input_padded_cols": 4,
        "query_accum_post_bias_rows": 4,
        "query_accum_post_bias_cols": 4,
        "query_accum_pre_bias_padded_rows": 6,
        "query_accum_pre_bias_padded_cols": 4,
        "query_accum_post_bias_padded_rows": 6,
        "query_accum_post_bias_padded_cols": 4,
        "query_output_padded_rows": 6,
        "query_output_padded_cols": 4,
        "key_act_input_rows": 4,
        "key_act_input_cols": 4,
        "key_act_input_padded_rows": 6,
        "key_act_input_padded_cols": 4,
        "key_accum_post_bias_rows": 4,
        "key_accum_post_bias_cols": 4,
        "key_accum_pre_bias_padded_rows": 6,
        "key_accum_pre_bias_padded_cols": 4,
        "key_accum_post_bias_padded_rows": 6,
        "key_accum_post_bias_padded_cols": 4,
        "key_output_padded_rows": 6,
        "key_output_padded_cols": 4,
        "value_act_input_rows": 4,
        "value_act_input_cols": 4,
        "value_act_input_padded_rows": 6,
        "value_act_input_padded_cols": 4,
        "value_accum_post_bias_rows": 4,
        "value_accum_post_bias_cols": 4,
        "value_accum_pre_bias_padded_rows": 6,
        "value_accum_pre_bias_padded_cols": 4,
        "value_accum_post_bias_padded_rows": 6,
        "value_accum_post_bias_padded_cols": 4,
        "value_output_padded_rows": 6,
        "value_output_padded_cols": 4,
    }
    (replay_dir / "replay_metadata.json").write_text(json.dumps(metadata))

    def write_i32(name: str, arr: np.ndarray) -> None:
        (replay_dir / f"{name}.raw").write_bytes(arr.astype("<i4", copy=False).tobytes())

    def write_i8(name: str, arr: np.ndarray) -> None:
        (replay_dir / f"{name}.raw").write_bytes(arr.astype(np.int8, copy=False).tobytes())

    clean_i32 = np.zeros((6, 4), dtype=np.int32)
    clean_i8 = np.zeros((6, 4), dtype=np.int8)
    for proj in ("query", "key", "value"):
        write_i8(f"{proj}_act_input_padded", clean_i8)
        write_i32(f"{proj}_accum_pre_bias_padded", clean_i32)
        write_i32(f"{proj}_accum_post_bias_padded", clean_i32)
        write_i8(f"{proj}_output_padded", clean_i8)

    report = compare_rtl_golden.emit_projection_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=tmp_path / "projection_padding_report_clean.json",
    )

    assert report["overall_classification"] == "padding_clean"
    assert all(report["projections"][proj]["classification"] == "padding_clean" for proj in ("query", "key", "value"))


def _write_qkv_source_replay_dir(
    replay_dir: Path,
    *,
    ln1_input: np.ndarray,
    ln1_output: np.ndarray,
    query_act: np.ndarray,
    key_act: np.ndarray,
    value_act: np.ndarray,
) -> None:
    metadata = {
        "block_prefix": "block0",
        "dtype_map": {
            "ln1_input_padded": "int8",
            "ln1_output_padded": "int8",
            "query_act_input_padded": "int8",
            "key_act_input_padded": "int8",
            "value_act_input_padded": "int8",
        },
        "ln1_input_padded_path": str(replay_dir / "ln1_input_padded.raw"),
        "ln1_input_padded_shape": list(ln1_input.shape),
        "ln1_output_padded_path": str(replay_dir / "ln1_output_padded.raw"),
        "ln1_output_padded_shape": list(ln1_output.shape),
        "query_act_input_rows": 4,
        "query_act_input_cols": 4,
        "query_act_input_padded_rows": 6,
        "query_act_input_padded_cols": 4,
        "key_act_input_padded_rows": 6,
        "key_act_input_padded_cols": 4,
        "value_act_input_padded_rows": 6,
        "value_act_input_padded_cols": 4,
    }
    (replay_dir / "replay_metadata.json").write_text(json.dumps(metadata))
    (replay_dir / "ln1_input_padded.raw").write_bytes(ln1_input.astype(np.int8, copy=False).tobytes())
    (replay_dir / "ln1_output_padded.raw").write_bytes(ln1_output.astype(np.int8, copy=False).tobytes())
    (replay_dir / "query_act_input_padded.raw").write_bytes(query_act.astype(np.int8, copy=False).tobytes())
    (replay_dir / "key_act_input_padded.raw").write_bytes(key_act.astype(np.int8, copy=False).tobytes())
    (replay_dir / "value_act_input_padded.raw").write_bytes(value_act.astype(np.int8, copy=False).tobytes())


def test_qkv_source_padding_report_classifies_layernorm_beta_expected(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    clean_prefix = np.arange(16, dtype=np.int16).reshape(4, 4).astype(np.int8)
    padded_tail = np.array([[-1, 1, 8, 0], [-5, 10, 0, -7]], dtype=np.int8)
    ln1_input = np.vstack([clean_prefix, np.zeros((2, 4), dtype=np.int8)])
    ln1_output = np.vstack([clean_prefix, padded_tail])
    _write_qkv_source_replay_dir(
        replay_dir,
        ln1_input=ln1_input,
        ln1_output=ln1_output,
        query_act=ln1_output,
        key_act=ln1_output,
        value_act=ln1_output,
    )

    report = compare_rtl_golden.emit_qkv_source_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=tmp_path / "qkv_source_padding_report.json",
    )

    assert report["classification"] == "layernorm_beta_padding_expected"
    assert report["ln1_input_padded"]["padded_rows_zero"] is True
    assert report["projections"]["query"]["matches_ln1_output_padded"] is True
    assert "block0_head0_query__act_input_padded" in report["ignored_node_names"]


def test_qkv_source_padding_report_classifies_dirty_pre_layernorm_padding(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    clean_prefix = np.arange(16, dtype=np.int16).reshape(4, 4).astype(np.int8)
    dirty_tail = np.array([[3, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    ln1_input = np.vstack([clean_prefix, dirty_tail])
    ln1_output = np.vstack([clean_prefix, dirty_tail])
    _write_qkv_source_replay_dir(
        replay_dir,
        ln1_input=ln1_input,
        ln1_output=ln1_output,
        query_act=ln1_output,
        key_act=ln1_output,
        value_act=ln1_output,
    )

    report = compare_rtl_golden.emit_qkv_source_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=tmp_path / "qkv_source_padding_report_dirty.json",
    )

    assert report["classification"] == "dirty_pre_layernorm_padding"
    assert report["ln1_input_padded"]["first_nonzero_padded_coord"] == [4, 0]


def test_qkv_source_padding_report_classifies_post_ln1_alias_or_reuse(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    clean_prefix = np.arange(16, dtype=np.int16).reshape(4, 4).astype(np.int8)
    padded_tail = np.array([[-1, 1, 8, 0], [-5, 10, 0, -7]], dtype=np.int8)
    ln1_input = np.vstack([clean_prefix, np.zeros((2, 4), dtype=np.int8)])
    ln1_output = np.vstack([clean_prefix, padded_tail])
    key_act = ln1_output.copy()
    key_act[4, 0] = np.int8(12)
    _write_qkv_source_replay_dir(
        replay_dir,
        ln1_input=ln1_input,
        ln1_output=ln1_output,
        query_act=ln1_output,
        key_act=key_act,
        value_act=ln1_output,
    )

    report = compare_rtl_golden.emit_qkv_source_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=tmp_path / "qkv_source_padding_report_alias.json",
    )

    assert report["classification"] == "post_ln1_source_alias_or_reuse"
    assert report["projections"]["key"]["matches_ln1_output_padded"] is False
    assert report["projections"]["key"]["first_diff_coord"] == [4, 0]


def test_rebase_first_divergence_only_activates_for_expected_qkv_padding(tmp_path: Path):
    program = Assembler().assemble("NOP\nHALT\n")
    program.trace_manifest = {
        0: [
            {
                "node_name": "node_a",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            }
        ],
        1: [
            {
                "node_name": "node_b",
                "buf_id": BUF_ABUF,
                "offset_units": 1,
                "mem_rows": 1,
                "mem_cols": 4,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            }
        ],
    }
    golden_trace = {
        "raw_events": [
            {
                "pc": 0,
                "event_index": 0,
                "node_name": "node_a",
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "row_start": 0,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "raw_available": True,
                "raw": [1, 2, 3, 4],
            },
            {
                "pc": 1,
                "event_index": 0,
                "node_name": "node_b",
                "dtype": "int8",
                "scale": 0.5,
                "source": "architectural",
                "row_start": 0,
                "logical_rows": 1,
                "logical_cols": 4,
                "full_rows": 1,
                "full_cols": 4,
                "raw_available": True,
                "raw": [5, 6, 7, 8],
            },
        ]
    }
    snapshot_manifest_path = tmp_path / "snapshot_manifest.json"
    snapshot_data_path = tmp_path / "snapshot_data.bin"
    snapshot_data_path.write_bytes(bytes([9, 2, 3, 4, 5, 6, 7, 9]))
    snapshot_manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "pc": 0,
                        "event_index": 0,
                        "node_name": "node_a",
                        "buf_id": BUF_ABUF,
                        "offset_units": 0,
                        "mem_rows": 1,
                        "mem_cols": 1,
                        "logical_rows": 1,
                        "logical_cols": 4,
                        "full_rows": 1,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int8",
                        "scale": 0.5,
                        "source": "architectural",
                        "status": "captured",
                        "cycle": 1,
                        "byte_offset": 0,
                        "byte_size": 4,
                    },
                    {
                        "pc": 1,
                        "event_index": 0,
                        "node_name": "node_b",
                        "buf_id": BUF_ABUF,
                        "offset_units": 1,
                        "mem_rows": 1,
                        "mem_cols": 1,
                        "logical_rows": 1,
                        "logical_cols": 4,
                        "full_rows": 1,
                        "full_cols": 4,
                        "row_start": 0,
                        "dtype": "int8",
                        "scale": 0.5,
                        "source": "architectural",
                        "status": "captured",
                        "cycle": 1,
                        "byte_offset": 4,
                        "byte_size": 4,
                    },
                ]
            }
        )
    )

    unchanged = compare_rtl_golden._maybe_rebase_first_divergence_for_expected_qkv_padding(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        work_dir=tmp_path,
        artifacts={"golden_trace": "golden.json", "rtl_trace": "rtl.json"},
        qkv_source_padding_report={"classification": "dirty_pre_layernorm_padding"},
    )
    assert unchanged is None

    rebased = compare_rtl_golden._maybe_rebase_first_divergence_for_expected_qkv_padding(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        work_dir=tmp_path,
        artifacts={"golden_trace": "golden.json", "rtl_trace": "rtl.json"},
        qkv_source_padding_report={
            "classification": "layernorm_beta_padding_expected",
            "ignored_node_names": ["node_a"],
        },
    )
    assert rebased is not None
    assert rebased["node_name"] == "node_b"
    assert rebased["ignored_node_names"] == ["node_a"]
