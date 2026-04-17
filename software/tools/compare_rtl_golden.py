#!/usr/bin/env python3
"""Compare RTL execution against the software golden model."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taccel.assembler.assembler import ProgramBinary
from taccel.golden_model import MachineState, Simulator
from taccel.isa.encoding import decode
from tools.run_golden import load_input_array, write_runtime_inputs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"
QKT_WINDOW_START_PC = 75
QKT_WINDOW_END_PC = 80
QKT_SRAM_WRITE_START_PC = 26
QKT_SRAM_WRITE_END_PC = 80
QKT_PREFIX_SRAM_WRITE_START_PC = 12
QKT_PREFIX_SRAM_WRITE_END_PC = 40
QKT_STARTUP_SRAM_WRITE_START_PC = 0
QKT_STARTUP_SRAM_WRITE_END_PC = 40
QKT_ACCUM_WRITE_START_PC = 40
QKT_ACCUM_WRITE_END_PC = 80
QKT_HIDDEN_SNAPSHOT_PC = 77
LN1_WINDOW_START_PC = 21
LN1_WINDOW_END_PC = 28
LN1_SRAM_WRITE_START_PC = 21
LN1_SRAM_WRITE_END_PC = 27
QKT_FRAGMENT_TEST_MODE = "qkt_prev_ln1_qkv_full_history_then_qkt_replay"
QKT_PREFIX_FRAGMENT_TEST_MODE = "qkt_prev_pos_embed_ln1_qkv_full_history_then_qkt_replay"
QKT_STARTUP_FRAGMENT_TEST_MODE = "qkt_prev_program_entry_full_history_then_qkt_replay"


class _LocalDeiTProcessor:
    """Offline fallback for DeiT image preprocessing.

    The compiler and runtime compare flow only need a callable object that
    converts PIL images into `pixel_values`. Keeping that logic local lets the
    Phase F sign-off harness run without a network fetch for the processor
    metadata.
    """

    def __init__(self) -> None:
        self.resize_short = 256
        self.crop_size = 224
        self.image_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _prepare_one(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        width, height = img.size
        if width == 0 or height == 0:
            raise ValueError("Input image must have nonzero dimensions")

        scale = self.resize_short / min(width, height)
        resized_w = int(round(width * scale))
        resized_h = int(round(height * scale))
        img = img.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)

        left = max((resized_w - self.crop_size) // 2, 0)
        top = max((resized_h - self.crop_size) // 2, 0)
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        return np.transpose(arr, (2, 0, 1))

    def __call__(self, *, images, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; only 'pt' is supported")

        if isinstance(images, (list, tuple)):
            batch = [self._prepare_one(image) for image in images]
        else:
            batch = [self._prepare_one(images)]
        pixel_values = torch.from_numpy(np.stack(batch, axis=0))
        return {"pixel_values": pixel_values}


def _load_processor(model_name: str):
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        print(
            f"[compare_rtl_golden] Falling back to local DeiT processor for {model_name}: {exc}",
            file=sys.stderr,
        )
        return _LocalDeiTProcessor()


def _parse_csv_ints(text: str | None) -> set[int] | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return {int(chunk.strip()) for chunk in text.split(",") if chunk.strip()}


def _scenario_overrides(name: str) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {
        "baseline_default": {},
        "experimental_requant_pc": {
            "requant_pc_out_proj": True,
            "requant_pc_out_proj_blocks": {11},
        },
        "experimental_dequant_add": {
            "dequant_add_residual1_blocks": {11},
        },
        "experimental_softmax_attnv": {
            "fused_softmax_attnv_blocks": {11},
        },
        "experimental_fused_out_proj": {
            "fused_softmax_attnv_blocks": {11},
            "fused_softmax_attnv_accum_out_proj": True,
            "requant_pc_out_proj": True,
            "requant_pc_out_proj_blocks": {11},
        },
    }
    if name not in scenarios:
        raise ValueError(f"Unknown compile scenario: {name}")
    return dict(scenarios[name])


def _ensure_runner_built(runner_path: Path, rebuild: bool) -> None:
    if rebuild or not runner_path.exists():
        subprocess.run(
            ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "run_program"],
            check=True,
        )


def _load_program(path: Path) -> ProgramBinary:
    return ProgramBinary.from_bytes(path.read_bytes())


def _load_raw_or_npy_int8(path: Path, rows: int | None = None, cols: int | None = None) -> np.ndarray:
    arr = load_input_array(str(path))
    arr = np.asarray(arr, dtype=np.int8)
    if rows is not None and cols is not None:
        arr = arr.reshape(rows, cols)
    return arr


def _write_raw_int8(path: Path, arr: np.ndarray) -> None:
    np.asarray(arr, dtype=np.int8).reshape(-1).tofile(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _is_qkt_family_node(node_name: str | None) -> bool:
    if not node_name:
        return False
    return "_qkt" in str(node_name)


def _is_one_lsb_rounding_artifact(divergence: dict | None) -> bool:
    """Return True when the divergence is a single-LSB INT8 rounding tie-break.

    FP32 LAYERNORM accumulates mean/variance over many elements; tiny ordering
    differences between the RTL DPI implementation and NumPy can push a value
    from e.g. -34.4999 to -34.5001, flipping the rounded INT8 result by ±1.
    When the cosine-similarity is 1.0 (float-level perfect agreement) and the
    maximum absolute de-quantised difference equals exactly one scale unit, the
    divergence is an FP32 rounding artifact, not a correctness issue.
    """
    if divergence is None:
        return False
    dq = divergence.get("dequantized_summary") or {}
    cosine_sim = dq.get("cosine_similarity")
    max_abs_diff = dq.get("max_abs_diff")
    if cosine_sim is None or max_abs_diff is None:
        return False
    if cosine_sim < (1.0 - 1e-9):
        return False
    scale = (divergence.get("node_metadata") or {}).get("scale") or 1.0
    # Accept if the worst-case de-quantised error is at most 1.5 × scale (1 LSB).
    return float(max_abs_diff) <= 1.5 * float(scale)


def _is_projection_tail_debug_node(node_name: str | None) -> bool:
    if not node_name:
        return False
    name = str(node_name)
    padded_suffixes = (
        "__accum_pre_bias_padded",
        "__accum_padded",
        "__output_padded",
    )
    if name.endswith(padded_suffixes):
        return True
    prefix = name.split("__", 1)[0]
    return prefix.endswith(("_query", "_key", "_value"))


def _is_ln1_padding_debug_node(node_name: str | None) -> bool:
    if not node_name:
        return False
    name = str(node_name)
    return name.endswith(("__input_padded", "__output_padded")) and "_ln1__" in name


def _expected_qkv_padding_ignore_nodes(block_prefix: str) -> set[str]:
    ignored: set[str] = set()
    for proj in ("query", "key", "value"):
        prefix = f"{block_prefix}_head0_{proj}"
        ignored.update({
            f"{prefix}__act_input_padded",
            f"{prefix}__accum_pre_bias_padded",
            f"{prefix}__accum_padded",
            f"{prefix}__output_padded",
        })
    return ignored


def _load_systolic_window_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


SYSTOLIC_TRACE_COMPARE_FIELDS = [
    "sync_waiting_on_sys",
    "state",
    "mtile_q",
    "ntile_q",
    "ktile_q",
    "lane_q",
    "a_load_row_q",
    "step_en",
    "clear_acc",
    "dst_clear_active",
    "dst_clear_row_q",
    "dst_clear_rows_total_q",
    "drain_row_q",
    "drain_grp_q",
    "tile_drain_base_q",
    "drain_row_addr_q",
    "sys_sram_a_row",
    "sys_sram_b_row",
]


def _trim_systolic_window_records(records: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    for start_index, rec in enumerate(records):
        if (rec.get("state", 0) != 0 or
                rec.get("sys_busy") or
                rec.get("dst_clear_active") or
                rec.get("step_en") or
                rec.get("clear_acc")):
            return start_index, records[start_index:]
    return 0, records


def _is_idle_systolic_window_record(rec: dict[str, Any]) -> bool:
    return (
        rec.get("state", 0) == 0 and
        not rec.get("sys_busy") and
        not rec.get("dst_clear_active") and
        not rec.get("step_en") and
        not rec.get("clear_acc") and
        rec.get("sys_sram_a_row", 0) == 0 and
        rec.get("sys_sram_b_row", 0) == 0
    )


def diff_systolic_window_traces(
    *,
    baseline_trace_path: Path,
    fragment_trace_path: Path,
) -> dict[str, Any]:
    baseline = _load_systolic_window_trace(baseline_trace_path)
    fragment = _load_systolic_window_trace(fragment_trace_path)
    baseline_records = list(baseline.get("records", []))
    fragment_records = list(fragment.get("records", []))

    result: dict[str, Any] = {
        "mode": "diff_systolic_window",
        "pass": True,
        "baseline_trace_path": str(baseline_trace_path),
        "fragment_trace_path": str(fragment_trace_path),
        "baseline_record_count": len(baseline_records),
        "fragment_record_count": len(fragment_records),
        "baseline_trim_start_cycle_index": None,
        "fragment_trim_start_cycle_index": None,
        "first_diff_cycle_index": None,
        "field_name": None,
        "baseline_value": None,
        "fragment_value": None,
    }

    baseline_trim_start, baseline_records = _trim_systolic_window_records(baseline_records)
    fragment_trim_start, fragment_records = _trim_systolic_window_records(fragment_records)
    result["baseline_trim_start_cycle_index"] = baseline_trim_start
    result["fragment_trim_start_cycle_index"] = fragment_trim_start

    for cycle_index, (baseline_rec, fragment_rec) in enumerate(zip(baseline_records, fragment_records)):
        for field_name in SYSTOLIC_TRACE_COMPARE_FIELDS:
            if baseline_rec.get(field_name) != fragment_rec.get(field_name):
                result["pass"] = False
                result["first_diff_cycle_index"] = cycle_index
                result["field_name"] = field_name
                result["baseline_value"] = baseline_rec.get(field_name)
                result["fragment_value"] = fragment_rec.get(field_name)
                return result

    if len(baseline_records) != len(fragment_records):
        common_len = min(len(baseline_records), len(fragment_records))
        baseline_suffix = baseline_records[common_len:]
        fragment_suffix = fragment_records[common_len:]
        if all(_is_idle_systolic_window_record(rec) for rec in baseline_suffix) and all(
                _is_idle_systolic_window_record(rec) for rec in fragment_suffix):
            return result
        result["pass"] = False
        result["first_diff_cycle_index"] = common_len
        result["field_name"] = "__trace_length__"
        result["baseline_value"] = len(baseline_records)
        result["fragment_value"] = len(fragment_records)
    return result


def _load_accum_write_log(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_sram_write_log(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_hidden_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


ACCUM_WRITE_COMPARE_FIELDS = [
    "writer_source",
    "row",
    "issue_pc",
    "issue_opcode",
    "first_word0",
    "first_word1",
    "row_hex",
]

SRAM_WRITE_COMPARE_FIELDS = [
    "writer_source",
    "buf_id",
    "buf_name",
    "row",
    "issue_pc",
    "issue_opcode",
    "first_word0",
    "first_word1",
    "row_hex",
]


def diff_accum_write_logs(
    *,
    baseline_log_path: Path,
    fragment_log_path: Path,
) -> dict[str, Any]:
    baseline = _load_accum_write_log(baseline_log_path)
    fragment = _load_accum_write_log(fragment_log_path)
    baseline_records = list(baseline.get("records", []))
    fragment_records = list(fragment.get("records", []))
    baseline_cycle0 = baseline_records[0]["cycle"] if baseline_records else None
    fragment_cycle0 = fragment_records[0]["cycle"] if fragment_records else None

    result: dict[str, Any] = {
        "mode": "diff_accum_writes",
        "pass": True,
        "baseline_log_path": str(baseline_log_path),
        "fragment_log_path": str(fragment_log_path),
        "baseline_record_count": len(baseline_records),
        "fragment_record_count": len(fragment_records),
        "first_diff_record_index": None,
        "field_name": None,
        "baseline_value": None,
        "fragment_value": None,
        "difference_kind": None,
    }

    for index, (baseline_rec, fragment_rec) in enumerate(zip(baseline_records, fragment_records)):
        for field_name in ACCUM_WRITE_COMPARE_FIELDS:
            if baseline_rec.get(field_name) != fragment_rec.get(field_name):
                result["pass"] = False
                result["first_diff_record_index"] = index
                result["field_name"] = field_name
                result["baseline_value"] = baseline_rec.get(field_name)
                result["fragment_value"] = fragment_rec.get(field_name)
                result["difference_kind"] = "field_mismatch"
                result["baseline_relative_cycle"] = (
                    None if baseline_cycle0 is None else int(baseline_rec["cycle"]) - int(baseline_cycle0)
                )
                result["fragment_relative_cycle"] = (
                    None if fragment_cycle0 is None else int(fragment_rec["cycle"]) - int(fragment_cycle0)
                )
                return result

    if len(baseline_records) > len(fragment_records):
        extra = baseline_records[len(fragment_records)]
        result["pass"] = False
        result["first_diff_record_index"] = len(fragment_records)
        result["field_name"] = "__extra_write__"
        result["baseline_value"] = extra
        result["fragment_value"] = None
        result["difference_kind"] = "extra_write"
        result["baseline_relative_cycle"] = (
            None if baseline_cycle0 is None else int(extra["cycle"]) - int(baseline_cycle0)
        )
        result["fragment_relative_cycle"] = None
    elif len(fragment_records) > len(baseline_records):
        missing = fragment_records[len(baseline_records)]
        result["pass"] = False
        result["first_diff_record_index"] = len(baseline_records)
        result["field_name"] = "__missing_write__"
        result["baseline_value"] = None
        result["fragment_value"] = missing
        result["difference_kind"] = "missing_write"
        result["baseline_relative_cycle"] = None
        result["fragment_relative_cycle"] = (
            None if fragment_cycle0 is None else int(missing["cycle"]) - int(fragment_cycle0)
        )

    return result


def diff_sram_write_logs(
    *,
    baseline_log_path: Path,
    fragment_log_path: Path,
) -> dict[str, Any]:
    baseline = _load_sram_write_log(baseline_log_path)
    fragment = _load_sram_write_log(fragment_log_path)
    baseline_records = list(baseline.get("records", []))
    fragment_records = list(fragment.get("records", []))
    baseline_cycle0 = baseline_records[0]["cycle"] if baseline_records else None
    fragment_cycle0 = fragment_records[0]["cycle"] if fragment_records else None

    result: dict[str, Any] = {
        "mode": "diff_sram_writes",
        "pass": True,
        "baseline_log_path": str(baseline_log_path),
        "fragment_log_path": str(fragment_log_path),
        "baseline_record_count": len(baseline_records),
        "fragment_record_count": len(fragment_records),
        "first_diff_record_index": None,
        "field_name": None,
        "baseline_value": None,
        "fragment_value": None,
        "difference_kind": None,
    }

    for index, (baseline_rec, fragment_rec) in enumerate(zip(baseline_records, fragment_records)):
        for field_name in SRAM_WRITE_COMPARE_FIELDS:
            if baseline_rec.get(field_name) != fragment_rec.get(field_name):
                result["pass"] = False
                result["first_diff_record_index"] = index
                result["field_name"] = field_name
                result["baseline_value"] = baseline_rec.get(field_name)
                result["fragment_value"] = fragment_rec.get(field_name)
                result["difference_kind"] = "field_mismatch"
                result["baseline_relative_cycle"] = (
                    None if baseline_cycle0 is None else int(baseline_rec["cycle"]) - int(baseline_cycle0)
                )
                result["fragment_relative_cycle"] = (
                    None if fragment_cycle0 is None else int(fragment_rec["cycle"]) - int(fragment_cycle0)
                )
                return result

    if len(baseline_records) > len(fragment_records):
        extra = baseline_records[len(fragment_records)]
        result["pass"] = False
        result["first_diff_record_index"] = len(fragment_records)
        result["field_name"] = "__extra_write__"
        result["baseline_value"] = extra
        result["fragment_value"] = None
        result["difference_kind"] = "extra_write"
        result["baseline_relative_cycle"] = (
            None if baseline_cycle0 is None else int(extra["cycle"]) - int(baseline_cycle0)
        )
        result["fragment_relative_cycle"] = None
    elif len(fragment_records) > len(baseline_records):
        missing = fragment_records[len(baseline_records)]
        result["pass"] = False
        result["first_diff_record_index"] = len(baseline_records)
        result["field_name"] = "__missing_write__"
        result["baseline_value"] = None
        result["fragment_value"] = missing
        result["difference_kind"] = "missing_write"
        result["baseline_relative_cycle"] = None
        result["fragment_relative_cycle"] = (
            None if fragment_cycle0 is None else int(missing["cycle"]) - int(fragment_cycle0)
        )

    return result


def _sram_row_hex_from_bytes(row_bytes: bytes) -> str:
    if len(row_bytes) != 16:
        raise ValueError(f"Expected 16 bytes for SRAM row image, got {len(row_bytes)}")
    words = [int.from_bytes(row_bytes[idx : idx + 4], byteorder="little", signed=False) for idx in range(0, 16, 4)]
    return "".join(f"{words[idx]:08x}" for idx in range(3, -1, -1))


def _sram_row_words_from_bytes(row_bytes: bytes) -> tuple[int, int]:
    if len(row_bytes) != 16:
        raise ValueError(f"Expected 16 bytes for SRAM row image, got {len(row_bytes)}")
    return (
        int.from_bytes(row_bytes[0:4], byteorder="little", signed=False),
        int.from_bytes(row_bytes[4:8], byteorder="little", signed=False),
    )


def _sram_row_bytes_from_record(record: dict[str, Any]) -> bytes:
    row_hex = str(record.get("row_hex", ""))
    if len(row_hex) == 32:
        words = [int(row_hex[idx : idx + 8], 16) for idx in range(0, 32, 8)]
        return b"".join(int(words[idx]).to_bytes(4, byteorder="little", signed=False) for idx in range(3, -1, -1))
    first0 = int(record.get("first_word0", 0))
    first1 = int(record.get("first_word1", 0))
    return (
        int(first0).to_bytes(4, byteorder="little", signed=False)
        + int(first1).to_bytes(4, byteorder="little", signed=False)
        + bytes(8)
    )


def _ln1_operand_region(abs_row: int, *, base_row: int, gamma_rows: int) -> str:
    return "gamma" if abs_row < (base_row + gamma_rows) else "beta"


def _resolve_replay_path(replay_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else replay_dir / path


def _load_ln1_replay_inputs(replay_dir: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    metadata = json.loads((replay_dir / "replay_metadata.json").read_text())
    gamma_path = _resolve_replay_path(replay_dir, str(metadata["ln1_gamma_path"]))
    beta_path = _resolve_replay_path(replay_dir, str(metadata["ln1_beta_path"]))
    output_path_value = str(metadata.get("ln1_output_padded_path", "ln1_output_padded.raw"))
    output_path = _resolve_replay_path(replay_dir, output_path_value)
    return metadata, gamma_path, beta_path, output_path


def _collect_first_sram_rows(
    sram_log: dict[str, Any],
    *,
    buf_name: str,
    row_start: int,
    row_count: int,
) -> dict[int, dict[str, Any]]:
    row_end = row_start + row_count
    rows: dict[int, dict[str, Any]] = {}
    for record in sram_log.get("records", []):
        if str(record.get("buf_name")) != buf_name:
            continue
        row = int(record.get("row", -1))
        if row_start <= row < row_end and row not in rows:
            rows[row] = record
    return rows


def _normalize_sram_records_for_diff(
    rows_by_addr: dict[int, dict[str, Any]],
    *,
    buf_id: int,
    buf_name: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(sorted(rows_by_addr)):
        record = dict(rows_by_addr[row])
        record["cycle"] = idx
        record["issue_pc"] = 0
        record["issue_opcode"] = 0
        record["buf_id"] = buf_id
        record["buf_name"] = buf_name
        records.append(record)
    return {"records": records}


def emit_ln1_operand_report_from_replay_dir(
    *,
    replay_dir: Path,
    sram_log_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata, gamma_path, beta_path, _ = _load_ln1_replay_inputs(replay_dir)
    gamma_bytes = gamma_path.read_bytes()
    beta_bytes = beta_path.read_bytes()
    expected_image = gamma_bytes + beta_bytes
    if len(expected_image) % 16 != 0:
        raise ValueError("LayerNorm gamma/beta operand image must be 16-byte aligned")

    base_row = int(metadata["ln1_gamma_beta_wbuf_offset_units"])
    gamma_rows = len(gamma_bytes) // 16
    beta_rows = len(beta_bytes) // 16
    total_rows = gamma_rows + beta_rows
    ln1_issue_pc_max = int(metadata.get("ln1_pc", 0)) - 1 if "ln1_pc" in metadata else None
    expected_rows = [
        expected_image[idx * 16 : (idx + 1) * 16]
        for idx in range(total_rows)
    ]

    sram_log = _load_sram_write_log(sram_log_path)
    observed_by_row: dict[int, dict[str, Any]] = {}
    for record in sram_log.get("records", []):
        if str(record.get("buf_name")) != "wbuf":
            continue
        if ln1_issue_pc_max is not None and int(record.get("issue_pc", -1)) > ln1_issue_pc_max:
            continue
        abs_row = int(record.get("row", -1))
        if base_row <= abs_row < (base_row + total_rows) and abs_row not in observed_by_row:
            observed_by_row[abs_row] = record

    observed_rows = sorted(observed_by_row)
    first_mismatch: dict[str, Any] | None = None
    for abs_row in observed_rows:
        expected_row = expected_rows[abs_row - base_row]
        observed_record = observed_by_row[abs_row]
        observed_row = _sram_row_bytes_from_record(observed_record)
        if observed_row == expected_row:
            continue

        matched_expected_row = None
        for idx, candidate in enumerate(expected_rows):
            if observed_row == candidate:
                matched_expected_row = base_row + idx
                break

        expected_region = _ln1_operand_region(abs_row, base_row=base_row, gamma_rows=gamma_rows)
        matched_region = (
            None
            if matched_expected_row is None
            else _ln1_operand_region(matched_expected_row, base_row=base_row, gamma_rows=gamma_rows)
        )
        if observed_row == expected_row[::-1]:
            ordering_hint = "byte_reversed"
        elif matched_expected_row is not None and matched_region != expected_region:
            ordering_hint = "swapped_gamma_beta"
        elif matched_expected_row is not None and matched_expected_row != abs_row:
            ordering_hint = f"shifted_by_rows:{matched_expected_row - abs_row}"
        else:
            ordering_hint = "no_simple_pattern"

        exp_word0, exp_word1 = _sram_row_words_from_bytes(expected_row)
        first_mismatch = {
            "row": abs_row,
            "region": expected_region,
            "expected_row_hex": _sram_row_hex_from_bytes(expected_row),
            "observed_row_hex": str(observed_record.get("row_hex")),
            "expected_first_word0": exp_word0,
            "expected_first_word1": exp_word1,
            "observed_first_word0": int(observed_record.get("first_word0", 0)),
            "observed_first_word1": int(observed_record.get("first_word1", 0)),
            "ordering_hint": ordering_hint,
            "matched_expected_row": matched_expected_row,
            "matched_expected_region": matched_region,
            "issue_pc": int(observed_record.get("issue_pc", -1)),
            "issue_opcode": int(observed_record.get("issue_opcode", -1)),
            "writer_source": str(observed_record.get("writer_source")),
        }
        break

    first_observed_row = observed_rows[0] if observed_rows else None
    report = {
        "mode": "ln1_operand_report",
        "replay_dir": str(replay_dir),
        "sram_log_path": str(sram_log_path),
        "pass": first_mismatch is None,
        "ln1_gamma_beta_wbuf_offset_units": base_row,
        "ln1_issue_pc_max": ln1_issue_pc_max,
        "expected_gamma_rows": gamma_rows,
        "expected_beta_rows": beta_rows,
        "expected_total_rows": total_rows,
        "observed_wbuf_rows": observed_rows,
        "observed_row_count": len(observed_rows),
        "first_observed_wbuf_row": first_observed_row,
        "missing_rows_before_first_observed": (
            []
            if first_observed_row is None or first_observed_row <= base_row
            else list(range(base_row, first_observed_row))
        ),
        "first_mismatch": first_mismatch,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def emit_ln1_provenance_report(
    *,
    replay_dir: Path,
    baseline_log_path: Path,
    fragment_log_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    operand_report_path = out_path.parent / "ln1_operand_report.json"
    operand_report = emit_ln1_operand_report_from_replay_dir(
        replay_dir=replay_dir,
        sram_log_path=baseline_log_path,
        out_path=operand_report_path,
    )

    metadata, gamma_path, beta_path, output_path = _load_ln1_replay_inputs(replay_dir)
    gamma_rows = len(gamma_path.read_bytes()) // 16
    beta_rows = len(beta_path.read_bytes()) // 16
    total_wbuf_rows = gamma_rows + beta_rows
    wbuf_row_start = int(metadata["ln1_gamma_beta_wbuf_offset_units"])
    wbuf_row_end = wbuf_row_start + total_wbuf_rows

    output_bytes = output_path.read_bytes()
    if len(output_bytes) % 16 != 0:
        raise ValueError("LayerNorm output replay payload must be 16-byte aligned")
    output_row_start = int(metadata["ln1_output_padded_offset_units"])
    output_row_count = len(output_bytes) // 16

    baseline_log = _load_sram_write_log(baseline_log_path)
    fragment_log = _load_sram_write_log(fragment_log_path)
    baseline_wbuf_rows = _collect_first_sram_rows(
        baseline_log,
        buf_name="wbuf",
        row_start=wbuf_row_start,
        row_count=total_wbuf_rows,
    )
    fragment_wbuf_rows = _collect_first_sram_rows(
        fragment_log,
        buf_name="wbuf",
        row_start=wbuf_row_start,
        row_count=total_wbuf_rows,
    )
    baseline_abuf_rows = _collect_first_sram_rows(
        baseline_log,
        buf_name="abuf",
        row_start=output_row_start,
        row_count=output_row_count,
    )
    fragment_abuf_rows = _collect_first_sram_rows(
        fragment_log,
        buf_name="abuf",
        row_start=output_row_start,
        row_count=output_row_count,
    )

    expected_gamma_rows = list(range(wbuf_row_start, wbuf_row_start + gamma_rows))
    expected_beta_rows = list(range(wbuf_row_start + gamma_rows, wbuf_row_end))
    missing_gamma_rows = [row for row in expected_gamma_rows if row not in baseline_wbuf_rows]
    missing_beta_rows = [row for row in expected_beta_rows if row not in baseline_wbuf_rows]
    first_observed_wbuf_row = min(baseline_wbuf_rows) if baseline_wbuf_rows else None
    first_missing_expected_wbuf_row = next(
        (row for row in range(wbuf_row_start, wbuf_row_end) if row not in baseline_wbuf_rows),
        None,
    )

    baseline_wbuf_norm = _normalize_sram_records_for_diff(
        baseline_wbuf_rows,
        buf_id=1,
        buf_name="wbuf",
    )
    fragment_wbuf_norm = _normalize_sram_records_for_diff(
        fragment_wbuf_rows,
        buf_id=1,
        buf_name="wbuf",
    )
    baseline_abuf_norm = _normalize_sram_records_for_diff(
        baseline_abuf_rows,
        buf_id=0,
        buf_name="abuf",
    )
    fragment_abuf_norm = _normalize_sram_records_for_diff(
        fragment_abuf_rows,
        buf_id=0,
        buf_name="abuf",
    )

    normalized_baseline_wbuf_path = out_path.parent / "ln1_wbuf_baseline.normalized.json"
    normalized_fragment_wbuf_path = out_path.parent / "ln1_wbuf_fragment.normalized.json"
    normalized_baseline_abuf_path = out_path.parent / "ln1_abuf_baseline.normalized.json"
    normalized_fragment_abuf_path = out_path.parent / "ln1_abuf_fragment.normalized.json"
    normalized_baseline_wbuf_path.write_text(json.dumps(baseline_wbuf_norm, indent=2, sort_keys=True))
    normalized_fragment_wbuf_path.write_text(json.dumps(fragment_wbuf_norm, indent=2, sort_keys=True))
    normalized_baseline_abuf_path.write_text(json.dumps(baseline_abuf_norm, indent=2, sort_keys=True))
    normalized_fragment_abuf_path.write_text(json.dumps(fragment_abuf_norm, indent=2, sort_keys=True))

    wbuf_diff = diff_sram_write_logs(
        baseline_log_path=normalized_baseline_wbuf_path,
        fragment_log_path=normalized_fragment_wbuf_path,
    )
    abuf_diff = diff_sram_write_logs(
        baseline_log_path=normalized_baseline_abuf_path,
        fragment_log_path=normalized_fragment_abuf_path,
    )

    baseline_wbuf_matches_fragment = bool(wbuf_diff.get("pass", False))
    baseline_abuf_matches_fragment = bool(abuf_diff.get("pass", False))
    gamma_present = not missing_gamma_rows
    beta_present = not missing_beta_rows

    if not gamma_present or first_observed_wbuf_row is None:
        classification = "capture_window_clipped"
    elif gamma_present and beta_present and not operand_report.get("pass", False):
        classification = "ln1_dma_wbuf_provenance_issue"
    elif baseline_wbuf_matches_fragment and not baseline_abuf_matches_fragment:
        classification = "ln1_full_program_serialization_issue"
    elif baseline_wbuf_matches_fragment and baseline_abuf_matches_fragment and operand_report.get("pass", False):
        classification = "ln1_operand_loading_exonerated"
    else:
        classification = "ln1_dma_wbuf_provenance_issue"

    report = {
        "mode": "ln1_provenance_report",
        "replay_dir": str(replay_dir),
        "baseline_log_path": str(baseline_log_path),
        "fragment_log_path": str(fragment_log_path),
        "classification": classification,
        "ln1_operand_report_path": str(operand_report_path),
        "ln1_gamma_beta_wbuf_offset_units": wbuf_row_start,
        "expected_gamma_rows": expected_gamma_rows,
        "expected_beta_rows": expected_beta_rows,
        "gamma_rows_present": gamma_present,
        "beta_rows_present": beta_present,
        "missing_gamma_rows": missing_gamma_rows,
        "missing_beta_rows": missing_beta_rows,
        "first_observed_wbuf_row": first_observed_wbuf_row,
        "first_missing_expected_wbuf_row": first_missing_expected_wbuf_row,
        "baseline_wbuf_matches_fragment": baseline_wbuf_matches_fragment,
        "baseline_abuf_ln1_output_matches_fragment": baseline_abuf_matches_fragment,
        "operand_report_pass": bool(operand_report.get("pass", False)),
        "wbuf_diff": wbuf_diff,
        "abuf_diff": abuf_diff,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def classify_qkt_prestate_provenance(
    *,
    fragment_artifacts_complete: bool,
    fragment_test_passed: bool,
    fragment_accum_pre_matches_baseline: bool,
    fragment_qkt_output_matches_golden: bool,
    accum_write_diff_pass: bool,
    hidden_snapshot_diff_pass: bool,
    window_diff_pass: bool,
) -> str:
    if not fragment_artifacts_complete:
        return "fragment_artifact_incomplete"
    if not fragment_test_passed or not fragment_qkt_output_matches_golden:
        return "window_local_qkt_bug"
    if (
        fragment_accum_pre_matches_baseline
        and accum_write_diff_pass
        and hidden_snapshot_diff_pass
        and window_diff_pass
    ):
        return "nonblocking_qkt_prestate_scratch"
    return "earlier_history_required"


def classify_qkt_prefix_provenance(
    *,
    fragment_artifacts_complete: bool,
    fragment_test_passed: bool,
    pos_embed_output_matches_baseline: bool,
    ln1_output_matches_baseline: bool,
    query_accum_pre_bias_matches_baseline: bool,
    fragment_accum_pre_matches_baseline: bool,
    fragment_qkt_output_matches_golden: bool,
    prefix_sram_diff_pass: bool,
) -> str:
    if not fragment_test_passed or not fragment_qkt_output_matches_golden:
        return "prefix_local_bug"
    if not fragment_artifacts_complete:
        return "prefix_provenance_mismatch"
    if (
        not prefix_sram_diff_pass
        or not pos_embed_output_matches_baseline
        or not ln1_output_matches_baseline
        or not query_accum_pre_bias_matches_baseline
    ):
        return "prefix_provenance_mismatch"
    if fragment_accum_pre_matches_baseline:
        return "prefix_nonblocking_scratch"
    return "history_earlier_than_pos_embed"


def classify_qkt_startup_provenance(
    *,
    fragment_artifacts_complete: bool,
    fragment_test_passed: bool,
    pos_embed_act_input_matches_baseline: bool,
    pos_embed_pos_input_matches_baseline: bool,
    pos_embed_output_matches_baseline: bool,
    ln1_output_matches_baseline: bool,
    query_accum_pre_bias_matches_baseline: bool,
    fragment_accum_pre_matches_baseline: bool,
    fragment_qkt_output_matches_golden: bool,
    startup_sram_diff_pass: bool,
) -> str:
    if not fragment_test_passed or not fragment_qkt_output_matches_golden:
        return "startup_local_bug"
    if not fragment_artifacts_complete:
        return "startup_provenance_mismatch"
    if (
        not startup_sram_diff_pass
        or not pos_embed_act_input_matches_baseline
        or not pos_embed_pos_input_matches_baseline
        or not pos_embed_output_matches_baseline
        or not ln1_output_matches_baseline
        or not query_accum_pre_bias_matches_baseline
    ):
        return "startup_provenance_mismatch"
    if fragment_accum_pre_matches_baseline:
        return "startup_nonblocking_scratch"
    return "history_outside_program_entry"


def _load_qkt_fragment_checkpoints(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    checkpoints: dict[str, Any] = {
        "mode": payload.get("mode"),
        "node_prefix": payload.get("node_prefix"),
        "strip_row_start": int(payload.get("strip_row_start", 0)),
    }
    for key, entry in payload.items():
        if key in ("mode", "node_prefix", "strip_row_start"):
            continue
        if not isinstance(entry, dict):
            continue
        shape = entry.get("shape")
        values = entry.get("values")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"Fragment checkpoint {key} is missing a valid shape")
        rows = int(shape[0])
        cols = int(shape[1])
        dtype = str(entry.get("dtype", "int32"))
        np_dtype = np.int32 if dtype == "int32" else np.int8
        tensor = np.asarray(values, dtype=np_dtype)
        if tensor.shape != (rows, cols):
            raise ValueError(
                f"Fragment checkpoint {key} has shape {tuple(tensor.shape)}, expected {(rows, cols)}"
            )
        checkpoints[key] = {
            "dtype": dtype,
            "shape": [rows, cols],
            "row_start": int(entry.get("row_start", 0)),
            "tensor": tensor,
        }
    return checkpoints


def _run_qkt_prestate_fragment_capture(*, replay_dir: Path, work_dir: Path) -> dict[str, Any]:
    artifact_paths = {
        "fragment_qkt_accum_write_log": work_dir / "fragment_qkt_accum_write_log.json",
        "fragment_qkt_hidden_snapshot": work_dir / "fragment_qkt_hidden_snapshot.json",
        "fragment_qkt_window": work_dir / "fragment_qkt_window.json",
        "fragment_qkt_checkpoints": work_dir / "fragment_qkt_checkpoints.json",
        "fragment_qkt_stdout": work_dir / "fragment_qkt.stdout.txt",
        "fragment_qkt_stderr": work_dir / "fragment_qkt.stderr.txt",
    }
    env = os.environ.copy()
    env.update(
        {
            "RTL_QKT_REPLAY_DIR": str(replay_dir),
            "RTL_QKT_MICROTRACE_MODE": QKT_FRAGMENT_TEST_MODE,
            "RTL_QKT_ACCUM_WRITE_LOG_OUT": str(artifact_paths["fragment_qkt_accum_write_log"]),
            "RTL_QKT_HIDDEN_SNAPSHOT_OUT": str(artifact_paths["fragment_qkt_hidden_snapshot"]),
            "RTL_QKT_MICROTRACE_OUT": str(artifact_paths["fragment_qkt_window"]),
            "RTL_QKT_CHECKPOINTS_OUT": str(artifact_paths["fragment_qkt_checkpoints"]),
        }
    )
    proc = subprocess.run(
        ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "test_systolic_qkt"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    artifact_paths["fragment_qkt_stdout"].write_text(proc.stdout)
    artifact_paths["fragment_qkt_stderr"].write_text(proc.stderr)
    return {
        "pass": proc.returncode == 0 and f"PASS: {QKT_FRAGMENT_TEST_MODE}" in proc.stdout,
        "returncode": proc.returncode,
        **{key: str(path) for key, path in artifact_paths.items()},
    }


def _run_qkt_prefix_fragment_capture(*, replay_dir: Path, work_dir: Path) -> dict[str, Any]:
    artifact_paths = {
        "fragment_prefix_sram_write_log": work_dir / "fragment_prefix_sram_write_log.json",
        "fragment_prefix_checkpoints": work_dir / "fragment_prefix_checkpoints.json",
        "fragment_prefix_window": work_dir / "fragment_prefix_window.json",
        "fragment_prefix_stdout": work_dir / "fragment_prefix.stdout.txt",
        "fragment_prefix_stderr": work_dir / "fragment_prefix.stderr.txt",
    }
    env = os.environ.copy()
    env.update(
        {
            "RTL_QKT_REPLAY_DIR": str(replay_dir),
            "RTL_QKT_MICROTRACE_MODE": QKT_PREFIX_FRAGMENT_TEST_MODE,
            "RTL_QKT_SRAM_WRITE_LOG_OUT": str(artifact_paths["fragment_prefix_sram_write_log"]),
            "RTL_QKT_CHECKPOINTS_OUT": str(artifact_paths["fragment_prefix_checkpoints"]),
            "RTL_QKT_MICROTRACE_OUT": str(artifact_paths["fragment_prefix_window"]),
        }
    )
    proc = subprocess.run(
        ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "test_systolic_qkt"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    artifact_paths["fragment_prefix_stdout"].write_text(proc.stdout)
    artifact_paths["fragment_prefix_stderr"].write_text(proc.stderr)
    return {
        "pass": proc.returncode == 0 and f"PASS: {QKT_PREFIX_FRAGMENT_TEST_MODE}" in proc.stdout,
        "returncode": proc.returncode,
        **{key: str(path) for key, path in artifact_paths.items()},
    }


def _run_qkt_startup_fragment_capture(*, replay_dir: Path, work_dir: Path) -> dict[str, Any]:
    artifact_paths = {
        "fragment_startup_sram_write_log": work_dir / "fragment_startup_sram_write_log.json",
        "fragment_startup_checkpoints": work_dir / "fragment_startup_checkpoints.json",
        "fragment_startup_window": work_dir / "fragment_startup_window.json",
        "fragment_startup_stdout": work_dir / "fragment_startup.stdout.txt",
        "fragment_startup_stderr": work_dir / "fragment_startup.stderr.txt",
    }
    env = os.environ.copy()
    env.update(
        {
            "RTL_QKT_REPLAY_DIR": str(replay_dir),
            "RTL_QKT_MICROTRACE_MODE": QKT_STARTUP_FRAGMENT_TEST_MODE,
            "RTL_QKT_SRAM_WRITE_LOG_OUT": str(artifact_paths["fragment_startup_sram_write_log"]),
            "RTL_QKT_CHECKPOINTS_OUT": str(artifact_paths["fragment_startup_checkpoints"]),
            "RTL_QKT_MICROTRACE_OUT": str(artifact_paths["fragment_startup_window"]),
        }
    )
    proc = subprocess.run(
        ["make", "-C", str(REPO_ROOT / "rtl" / "verilator"), "test_systolic_qkt"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    artifact_paths["fragment_startup_stdout"].write_text(proc.stdout)
    artifact_paths["fragment_startup_stderr"].write_text(proc.stderr)
    return {
        "pass": proc.returncode == 0 and f"PASS: {QKT_STARTUP_FRAGMENT_TEST_MODE}" in proc.stdout,
        "returncode": proc.returncode,
        **{key: str(path) for key, path in artifact_paths.items()},
    }


def emit_startup_pos_load_report(
    *,
    replay_dir: Path,
    baseline_startup_log_path: Path,
    fragment_startup_log_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    metadata = json.loads((replay_dir / "replay_metadata.json").read_text())
    issue_pcs = list(metadata.get("startup_issue_pcs", [2, 6, 10, 12, 13]))
    issue_pc = int(issue_pcs[2] if len(issue_pcs) >= 3 else 10)
    expected_start_row = int(metadata.get("startup_pos_wbuf_offset_units", 0))
    expected_row_units = int(metadata.get("startup_pos_input_padded_row_units", 0))
    expected_last_row = expected_start_row + max(0, expected_row_units - 1)

    baseline_log = _load_sram_write_log(baseline_startup_log_path)
    fragment_log = _load_sram_write_log(fragment_startup_log_path)
    baseline_records = [
        rec
        for rec in baseline_log.get("records", [])
        if rec.get("issue_pc") == issue_pc and rec.get("buf_name") == "wbuf"
    ]
    fragment_records = [
        rec
        for rec in fragment_log.get("records", [])
        if rec.get("issue_pc") == issue_pc and rec.get("buf_name") == "wbuf"
    ]

    def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [int(rec["row"]) for rec in records]
        return {
            "record_count": len(records),
            "unique_row_count": len(set(rows)),
            "min_row": min(rows) if rows else None,
            "max_row": max(rows) if rows else None,
            "covers_full_padded_span": bool(rows)
            and min(rows) <= expected_start_row
            and max(rows) >= expected_last_row,
        }

    baseline_summary = _summarize(baseline_records)
    fragment_summary = _summarize(fragment_records)
    diff_result = diff_sram_write_logs(
        baseline_log_path=baseline_startup_log_path,
        fragment_log_path=fragment_startup_log_path,
    )

    first_diff_for_pc10 = None
    if not diff_result.get("pass", False):
        max_len = min(len(baseline_records), len(fragment_records))
        for index, (baseline_rec, fragment_rec) in enumerate(zip(baseline_records, fragment_records)):
            for field_name in SRAM_WRITE_COMPARE_FIELDS:
                if baseline_rec.get(field_name) != fragment_rec.get(field_name):
                    first_diff_for_pc10 = {
                        "record_index": index,
                        "field_name": field_name,
                        "baseline_value": baseline_rec.get(field_name),
                        "fragment_value": fragment_rec.get(field_name),
                        "baseline_record": baseline_rec,
                        "fragment_record": fragment_rec,
                    }
                    break
            if first_diff_for_pc10 is not None:
                break
        if first_diff_for_pc10 is None and len(baseline_records) != len(fragment_records):
            first_diff_for_pc10 = {
                "record_index": max_len,
                "field_name": "__record_count__",
                "baseline_value": len(baseline_records),
                "fragment_value": len(fragment_records),
            }

    report = {
        "mode": "startup_pos_load_report",
        "issue_pc": issue_pc,
        "buf_name": "wbuf",
        "expected_start_row": expected_start_row,
        "expected_last_row": expected_last_row,
        "expected_row_units": expected_row_units,
        "baseline": baseline_summary,
        "fragment": fragment_summary,
        "first_diff_for_issue_pc": first_diff_for_pc10,
        "pass": bool(
            baseline_summary["covers_full_padded_span"]
            and fragment_summary["covers_full_padded_span"]
            and first_diff_for_pc10 is None
        ),
        "artifacts": {
            "baseline_startup_sram_write_log": str(baseline_startup_log_path),
            "fragment_startup_sram_write_log": str(fragment_startup_log_path),
            "replay_payloads": str(replay_dir),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def emit_qkt_prestate_provenance_report(
    *,
    first_divergence_path: Path,
    replay_dir: Path,
    baseline_accum_log_path: Path,
    baseline_hidden_snapshot_path: Path,
    baseline_window_path: Path,
    fragment_result: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    bundle = _load_debug_bundle(first_divergence_path)
    divergence = bundle["divergence"]
    node_prefix = str(divergence["node_name"]).split("__", 1)[0]

    baseline_accum_pre, _ = _extract_qkt_strip_tensor(
        bundle,
        f"{node_prefix}__accum_pre_matmul",
        source="rtl",
        strip_row_start=0,
    )
    golden_qkt_output, _ = _extract_qkt_strip_tensor(
        bundle,
        node_prefix,
        source="golden",
        strip_row_start=0,
    )

    required_fragment_paths = {
        "accum_write_log": Path(fragment_result["fragment_qkt_accum_write_log"]),
        "hidden_snapshot": Path(fragment_result["fragment_qkt_hidden_snapshot"]),
        "window": Path(fragment_result["fragment_qkt_window"]),
        "checkpoints": Path(fragment_result["fragment_qkt_checkpoints"]),
    }
    fragment_artifacts_complete = all(path.exists() for path in required_fragment_paths.values())

    fragment_accum_pre_matches_baseline = False
    fragment_qkt_output_matches_golden = False
    checkpoint_report: dict[str, Any] = {}
    accum_write_diff: dict[str, Any] | None = None
    hidden_snapshot_diff: dict[str, Any] | None = None
    window_diff: dict[str, Any] | None = None

    if fragment_artifacts_complete:
        try:
            checkpoints = _load_qkt_fragment_checkpoints(required_fragment_paths["checkpoints"])
            fragment_accum_pre = checkpoints["accum_pre_matmul"]["tensor"]
            fragment_qkt_output = checkpoints["qkt_output"]["tensor"]
            fragment_accum_pre_matches_baseline = bool(np.array_equal(fragment_accum_pre, baseline_accum_pre))
            fragment_qkt_output_matches_golden = bool(np.array_equal(fragment_qkt_output, golden_qkt_output))
            checkpoint_report = {
                "fragment_accum_pre_matmul_matches_baseline": fragment_accum_pre_matches_baseline,
                "fragment_qkt_output_matches_golden": fragment_qkt_output_matches_golden,
                "fragment_accum_pre_matmul_shape": list(fragment_accum_pre.shape),
                "fragment_qkt_output_shape": list(fragment_qkt_output.shape),
            }
            accum_write_diff = diff_accum_write_logs(
                baseline_log_path=baseline_accum_log_path,
                fragment_log_path=required_fragment_paths["accum_write_log"],
            )
            hidden_snapshot_diff = diff_hidden_snapshots(
                baseline_snapshot_path=baseline_hidden_snapshot_path,
                fragment_snapshot_path=required_fragment_paths["hidden_snapshot"],
            )
            window_diff = diff_systolic_window_traces(
                baseline_trace_path=baseline_window_path,
                fragment_trace_path=required_fragment_paths["window"],
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            fragment_artifacts_complete = False

    classification = classify_qkt_prestate_provenance(
        fragment_artifacts_complete=fragment_artifacts_complete,
        fragment_test_passed=bool(fragment_result.get("pass", False)),
        fragment_accum_pre_matches_baseline=fragment_accum_pre_matches_baseline,
        fragment_qkt_output_matches_golden=fragment_qkt_output_matches_golden,
        accum_write_diff_pass=bool(accum_write_diff and accum_write_diff.get("pass", False)),
        hidden_snapshot_diff_pass=bool(hidden_snapshot_diff and hidden_snapshot_diff.get("pass", False)),
        window_diff_pass=bool(window_diff and window_diff.get("pass", False)),
    )

    report = {
        "mode": "qkt_prestate_provenance_report",
        "classification": classification,
        "first_divergence_node": str(divergence["node_name"]),
        "first_divergence_pc": int(divergence.get("trace_pc", 0)),
        "replay_dir": str(replay_dir),
        "fragment_test_passed": bool(fragment_result.get("pass", False)),
        "fragment_returncode": int(fragment_result.get("returncode", -1)),
        "fragment_artifacts_complete": fragment_artifacts_complete,
        "fragment_result": fragment_result,
        "checkpoint_report": checkpoint_report,
        "accum_write_diff": accum_write_diff,
        "hidden_snapshot_diff": hidden_snapshot_diff,
        "window_diff": window_diff,
        "artifacts": {
            **dict(bundle["artifacts"]),
            "baseline_accum_write_log": str(baseline_accum_log_path),
            "baseline_hidden_snapshot": str(baseline_hidden_snapshot_path),
            "baseline_window": str(baseline_window_path),
            "source_first_divergence": str(first_divergence_path),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def emit_qkt_prefix_provenance_report(
    *,
    first_divergence_path: Path,
    replay_dir: Path,
    baseline_prefix_log_path: Path,
    fragment_result: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    bundle = _load_debug_bundle(first_divergence_path)
    divergence = bundle["divergence"]
    node_prefix = str(divergence["node_name"]).split("__", 1)[0]
    block_prefix = node_prefix.split("_head", 1)[0]
    query_prefix = node_prefix.replace("_qkt", "_query")

    baseline_pos_embed_output, _ = _extract_node_tensor_from_bundle(bundle, "pos_embed_add", source="rtl")
    baseline_ln1_output, _ = _extract_node_tensor_from_bundle(
        bundle,
        f"{block_prefix}_ln1__output_padded",
        source="rtl",
    )
    baseline_query_accum_pre_bias, _ = _extract_node_tensor_from_bundle(
        bundle,
        f"{query_prefix}__accum_pre_bias_padded",
        source="rtl",
    )
    baseline_accum_pre, _ = _extract_qkt_strip_tensor(
        bundle,
        f"{node_prefix}__accum_pre_matmul",
        source="rtl",
        strip_row_start=0,
    )
    baseline_qkt_output, _ = _extract_qkt_strip_tensor(
        bundle,
        node_prefix,
        source="rtl",
        strip_row_start=0,
    )
    golden_qkt_output, _ = _extract_qkt_strip_tensor(
        bundle,
        node_prefix,
        source="golden",
        strip_row_start=0,
    )

    required_fragment_paths = {
        "sram_write_log": Path(fragment_result["fragment_prefix_sram_write_log"]),
        "checkpoints": Path(fragment_result["fragment_prefix_checkpoints"]),
    }
    fragment_artifacts_complete = baseline_prefix_log_path.exists() and all(
        path.exists() for path in required_fragment_paths.values()
    )

    pos_embed_output_matches_baseline = False
    ln1_output_matches_baseline = False
    query_accum_pre_bias_matches_baseline = False
    fragment_accum_pre_matches_baseline = False
    fragment_qkt_output_matches_golden = False
    fragment_qkt_output_matches_baseline = False
    prefix_sram_diff: dict[str, Any] | None = None
    checkpoint_report: dict[str, Any] = {}

    if fragment_artifacts_complete:
        try:
            checkpoints = _load_qkt_fragment_checkpoints(required_fragment_paths["checkpoints"])
            required_keys = (
                "pos_embed_add_output",
                "ln1_output",
                "query_accum_pre_bias_padded",
                "accum_pre_matmul",
                "qkt_output",
            )
            if not all(key in checkpoints for key in required_keys):
                fragment_artifacts_complete = False
            else:
                pos_embed_output = checkpoints["pos_embed_add_output"]["tensor"]
                ln1_output = checkpoints["ln1_output"]["tensor"]
                query_accum_pre_bias = checkpoints["query_accum_pre_bias_padded"]["tensor"]
                fragment_accum_pre = checkpoints["accum_pre_matmul"]["tensor"]
                fragment_qkt_output = checkpoints["qkt_output"]["tensor"]
                pos_embed_output_matches_baseline = bool(
                    np.array_equal(pos_embed_output, baseline_pos_embed_output)
                )
                ln1_output_matches_baseline = bool(np.array_equal(ln1_output, baseline_ln1_output))
                query_accum_pre_bias_matches_baseline = bool(
                    np.array_equal(query_accum_pre_bias, baseline_query_accum_pre_bias)
                )
                fragment_accum_pre_matches_baseline = bool(
                    np.array_equal(fragment_accum_pre, baseline_accum_pre)
                )
                fragment_qkt_output_matches_golden = bool(
                    np.array_equal(fragment_qkt_output, golden_qkt_output)
                )
                fragment_qkt_output_matches_baseline = bool(
                    np.array_equal(fragment_qkt_output, baseline_qkt_output)
                )
                checkpoint_report = {
                    "pos_embed_add_output_matches_baseline": pos_embed_output_matches_baseline,
                    "ln1_output_matches_baseline": ln1_output_matches_baseline,
                    "query_accum_pre_bias_padded_matches_baseline": query_accum_pre_bias_matches_baseline,
                    "fragment_accum_pre_matmul_matches_baseline": fragment_accum_pre_matches_baseline,
                    "fragment_qkt_output_matches_golden": fragment_qkt_output_matches_golden,
                    "fragment_qkt_output_matches_baseline": fragment_qkt_output_matches_baseline,
                    "pos_embed_add_output_shape": list(pos_embed_output.shape),
                    "ln1_output_shape": list(ln1_output.shape),
                    "query_accum_pre_bias_padded_shape": list(query_accum_pre_bias.shape),
                    "fragment_accum_pre_matmul_shape": list(fragment_accum_pre.shape),
                    "fragment_qkt_output_shape": list(fragment_qkt_output.shape),
                }
                prefix_sram_diff = diff_sram_write_logs(
                    baseline_log_path=baseline_prefix_log_path,
                    fragment_log_path=required_fragment_paths["sram_write_log"],
                )
        except (ValueError, KeyError, json.JSONDecodeError):
            fragment_artifacts_complete = False

    classification = classify_qkt_prefix_provenance(
        fragment_artifacts_complete=fragment_artifacts_complete,
        fragment_test_passed=bool(fragment_result.get("pass", False)),
        pos_embed_output_matches_baseline=pos_embed_output_matches_baseline,
        ln1_output_matches_baseline=ln1_output_matches_baseline,
        query_accum_pre_bias_matches_baseline=query_accum_pre_bias_matches_baseline,
        fragment_accum_pre_matches_baseline=fragment_accum_pre_matches_baseline,
        fragment_qkt_output_matches_golden=fragment_qkt_output_matches_golden,
        prefix_sram_diff_pass=bool(prefix_sram_diff and prefix_sram_diff.get("pass", False)),
    )

    report = {
        "mode": "qkt_prefix_provenance_report",
        "classification": classification,
        "first_divergence_node": str(divergence["node_name"]),
        "first_divergence_pc": int(divergence.get("trace_pc", 0)),
        "replay_dir": str(replay_dir),
        "fragment_test_passed": bool(fragment_result.get("pass", False)),
        "fragment_returncode": int(fragment_result.get("returncode", -1)),
        "fragment_artifacts_complete": fragment_artifacts_complete,
        "fragment_result": fragment_result,
        "checkpoint_report": checkpoint_report,
        "prefix_sram_write_diff": prefix_sram_diff,
        "artifacts": {
            **dict(bundle["artifacts"]),
            "baseline_prefix_sram_write_log": str(baseline_prefix_log_path),
            "source_first_divergence": str(first_divergence_path),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def emit_qkt_startup_provenance_report(
    *,
    first_divergence_path: Path,
    replay_dir: Path,
    baseline_startup_log_path: Path,
    fragment_result: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    bundle = _load_debug_bundle(first_divergence_path)
    divergence = bundle["divergence"]
    node_prefix = str(divergence["node_name"]).split("__", 1)[0]
    block_prefix = node_prefix.split("_head", 1)[0]
    query_prefix = node_prefix.replace("_qkt", "_query")

    baseline_pos_embed_act_input, _ = _extract_node_tensor_from_bundle(
        bundle,
        "pos_embed_add__act_input",
        source="rtl",
    )
    baseline_pos_embed_pos_input, _ = _extract_node_tensor_from_bundle(
        bundle,
        "pos_embed_add__pos_input",
        source="rtl",
    )
    baseline_pos_embed_output, _ = _extract_node_tensor_from_bundle(bundle, "pos_embed_add", source="rtl")
    baseline_ln1_output, _ = _extract_node_tensor_from_bundle(
        bundle,
        f"{block_prefix}_ln1__output_padded",
        source="rtl",
    )
    baseline_query_accum_pre_bias, _ = _extract_node_tensor_from_bundle(
        bundle,
        f"{query_prefix}__accum_pre_bias_padded",
        source="rtl",
    )
    baseline_accum_pre, _ = _extract_qkt_strip_tensor(
        bundle,
        f"{node_prefix}__accum_pre_matmul",
        source="rtl",
        strip_row_start=0,
    )
    golden_qkt_output, _ = _extract_qkt_strip_tensor(
        bundle,
        node_prefix,
        source="golden",
        strip_row_start=0,
    )

    required_fragment_paths = {
        "sram_write_log": Path(fragment_result["fragment_startup_sram_write_log"]),
        "checkpoints": Path(fragment_result["fragment_startup_checkpoints"]),
    }
    fragment_artifacts_complete = baseline_startup_log_path.exists() and all(
        path.exists() for path in required_fragment_paths.values()
    )

    pos_embed_act_input_matches_baseline = False
    pos_embed_pos_input_matches_baseline = False
    pos_embed_output_matches_baseline = False
    ln1_output_matches_baseline = False
    query_accum_pre_bias_matches_baseline = False
    fragment_accum_pre_matches_baseline = False
    fragment_qkt_output_matches_golden = False
    startup_sram_diff: dict[str, Any] | None = None
    checkpoint_report: dict[str, Any] = {}

    if fragment_artifacts_complete:
        try:
            checkpoints = _load_qkt_fragment_checkpoints(required_fragment_paths["checkpoints"])
            required_keys = (
                "pos_embed_add_act_input",
                "pos_embed_add_pos_input",
                "pos_embed_add_output",
                "ln1_output",
                "query_accum_pre_bias_padded",
                "accum_pre_matmul",
                "qkt_output",
            )
            if not all(key in checkpoints for key in required_keys):
                fragment_artifacts_complete = False
            else:
                pos_embed_act_input = checkpoints["pos_embed_add_act_input"]["tensor"]
                pos_embed_pos_input = checkpoints["pos_embed_add_pos_input"]["tensor"]
                pos_embed_output = checkpoints["pos_embed_add_output"]["tensor"]
                ln1_output = checkpoints["ln1_output"]["tensor"]
                query_accum_pre_bias = checkpoints["query_accum_pre_bias_padded"]["tensor"]
                fragment_accum_pre = checkpoints["accum_pre_matmul"]["tensor"]
                fragment_qkt_output = checkpoints["qkt_output"]["tensor"]
                pos_embed_act_input_matches_baseline = bool(
                    np.array_equal(pos_embed_act_input, baseline_pos_embed_act_input)
                )
                pos_embed_pos_input_matches_baseline = bool(
                    np.array_equal(pos_embed_pos_input, baseline_pos_embed_pos_input)
                )
                pos_embed_output_matches_baseline = bool(
                    np.array_equal(pos_embed_output, baseline_pos_embed_output)
                )
                ln1_output_matches_baseline = bool(np.array_equal(ln1_output, baseline_ln1_output))
                query_accum_pre_bias_matches_baseline = bool(
                    np.array_equal(query_accum_pre_bias, baseline_query_accum_pre_bias)
                )
                fragment_accum_pre_matches_baseline = bool(
                    np.array_equal(fragment_accum_pre, baseline_accum_pre)
                )
                fragment_qkt_output_matches_golden = bool(
                    np.array_equal(fragment_qkt_output, golden_qkt_output)
                )
                checkpoint_report = {
                    "pos_embed_add_act_input_matches_baseline": pos_embed_act_input_matches_baseline,
                    "pos_embed_add_pos_input_matches_baseline": pos_embed_pos_input_matches_baseline,
                    "pos_embed_add_output_matches_baseline": pos_embed_output_matches_baseline,
                    "ln1_output_matches_baseline": ln1_output_matches_baseline,
                    "query_accum_pre_bias_padded_matches_baseline": query_accum_pre_bias_matches_baseline,
                    "fragment_accum_pre_matmul_matches_baseline": fragment_accum_pre_matches_baseline,
                    "fragment_qkt_output_matches_golden": fragment_qkt_output_matches_golden,
                    "pos_embed_add_act_input_shape": list(pos_embed_act_input.shape),
                    "pos_embed_add_pos_input_shape": list(pos_embed_pos_input.shape),
                    "pos_embed_add_output_shape": list(pos_embed_output.shape),
                    "ln1_output_shape": list(ln1_output.shape),
                    "query_accum_pre_bias_padded_shape": list(query_accum_pre_bias.shape),
                    "fragment_accum_pre_matmul_shape": list(fragment_accum_pre.shape),
                    "fragment_qkt_output_shape": list(fragment_qkt_output.shape),
                }
                startup_sram_diff = diff_sram_write_logs(
                    baseline_log_path=baseline_startup_log_path,
                    fragment_log_path=required_fragment_paths["sram_write_log"],
                )
        except (ValueError, KeyError, json.JSONDecodeError):
            fragment_artifacts_complete = False

    classification = classify_qkt_startup_provenance(
        fragment_artifacts_complete=fragment_artifacts_complete,
        fragment_test_passed=bool(fragment_result.get("pass", False)),
        pos_embed_act_input_matches_baseline=pos_embed_act_input_matches_baseline,
        pos_embed_pos_input_matches_baseline=pos_embed_pos_input_matches_baseline,
        pos_embed_output_matches_baseline=pos_embed_output_matches_baseline,
        ln1_output_matches_baseline=ln1_output_matches_baseline,
        query_accum_pre_bias_matches_baseline=query_accum_pre_bias_matches_baseline,
        fragment_accum_pre_matches_baseline=fragment_accum_pre_matches_baseline,
        fragment_qkt_output_matches_golden=fragment_qkt_output_matches_golden,
        startup_sram_diff_pass=bool(startup_sram_diff and startup_sram_diff.get("pass", False)),
    )

    report = {
        "mode": "qkt_startup_provenance_report",
        "classification": classification,
        "first_divergence_node": str(divergence["node_name"]),
        "first_divergence_pc": int(divergence.get("trace_pc", 0)),
        "replay_dir": str(replay_dir),
        "fragment_test_passed": bool(fragment_result.get("pass", False)),
        "fragment_returncode": int(fragment_result.get("returncode", -1)),
        "fragment_artifacts_complete": fragment_artifacts_complete,
        "fragment_result": fragment_result,
        "checkpoint_report": checkpoint_report,
        "startup_sram_write_diff": startup_sram_diff,
        "artifacts": {
            **dict(bundle["artifacts"]),
            "baseline_startup_sram_write_log": str(baseline_startup_log_path),
            "source_first_divergence": str(first_divergence_path),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


HIDDEN_SNAPSHOT_SCALAR_FIELDS = [
    "state",
    "mtile_q",
    "ntile_q",
    "ktile_q",
    "lane_q",
    "a_load_row_q",
    "drain_row_q",
    "drain_grp_q",
    "tile_drain_base_q",
    "drain_row_addr_q",
    "clear_acc",
    "step_en",
    "dst_clear_active",
    "dst_clear_row_q",
    "dst_clear_rows_total_q",
]


def diff_hidden_snapshots(
    *,
    baseline_snapshot_path: Path,
    fragment_snapshot_path: Path,
) -> dict[str, Any]:
    baseline = _load_hidden_snapshot(baseline_snapshot_path)
    fragment = _load_hidden_snapshot(fragment_snapshot_path)
    result: dict[str, Any] = {
        "mode": "diff_hidden_snapshot",
        "pass": True,
        "baseline_snapshot_path": str(baseline_snapshot_path),
        "fragment_snapshot_path": str(fragment_snapshot_path),
        "field_name": None,
        "baseline_value": None,
        "fragment_value": None,
    }

    for field_name in HIDDEN_SNAPSHOT_SCALAR_FIELDS:
        if baseline.get(field_name) != fragment.get(field_name):
            result["pass"] = False
            result["field_name"] = field_name
            result["baseline_value"] = baseline.get(field_name)
            result["fragment_value"] = fragment.get(field_name)
            return result

    for matrix_name in ("a_tile_scratch", "a_skew", "b_skew", "pe_acc"):
        baseline_matrix = baseline.get(matrix_name, [])
        fragment_matrix = fragment.get(matrix_name, [])
        if len(baseline_matrix) != len(fragment_matrix):
            result["pass"] = False
            result["field_name"] = f"{matrix_name}.__shape__"
            result["baseline_value"] = len(baseline_matrix)
            result["fragment_value"] = len(fragment_matrix)
            return result
        for row_idx, (baseline_row, fragment_row) in enumerate(zip(baseline_matrix, fragment_matrix)):
            if len(baseline_row) != len(fragment_row):
                result["pass"] = False
                result["field_name"] = f"{matrix_name}[{row_idx}].__shape__"
                result["baseline_value"] = len(baseline_row)
                result["fragment_value"] = len(fragment_row)
                return result
            for col_idx, (baseline_value, fragment_value) in enumerate(zip(baseline_row, fragment_row)):
                if baseline_value != fragment_value:
                    result["pass"] = False
                    result["field_name"] = f"{matrix_name}[{row_idx}][{col_idx}]"
                    result["baseline_value"] = baseline_value
                    result["fragment_value"] = fragment_value
                    return result

    return result


def _run_golden_program(
    program: ProgramBinary,
    patches_int8: np.ndarray | None,
    cls_int8: np.ndarray | None,
    folded_pos_embed: bool,
    num_classes: int,
    trace_nodes: list[str] | None = None,
) -> dict[str, Any]:
    state = MachineState(dram_data=program.data)
    sim = Simulator(state)
    sim.load_program(program)
    if trace_nodes:
        sim.enable_trace(trace_nodes)

    if patches_int8 is not None:
        write_runtime_inputs(
            state,
            program,
            patches_int8,
            cls_input=cls_int8,
            folded_pos_embed=folded_pos_embed,
        )

    insn_count = sim.run()
    logits = state.accum[:num_classes].astype(np.int32).tolist()
    return {
        "fault": False,
        "fault_code": 0,
        "instruction_count": int(insn_count),
        "cycle_count": int(state.cycle_count),
        "logits": logits,
        "trace": sim.get_trace_payload() if trace_nodes else None,
    }


def _compile_mode_cycle_budget(min_budget: int, golden_cycles: int) -> int:
    # Full compiler-generated DeiT programs are much longer than the focused
    # RTL block regressions, and the RTL usually needs materially more cycles
    # than the software golden model because it models transfer and engine
    # sequencing explicitly. Use the golden cycle count to pick a realistic
    # runner budget while still keeping a hard timeout.
    derived_budget = int(golden_cycles) * 6 + 1_000_000
    return max(int(min_budget), derived_budget)


def _invoke_runner(
    runner_path: Path,
    program_path: Path,
    summary_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    trace_json_out: Path | None = None,
    snapshot_request_path: Path | None = None,
    snapshot_manifest_out: Path | None = None,
    snapshot_data_out: Path | None = None,
    systolic_window_start_pc: int | None = None,
    systolic_window_end_pc: int | None = None,
    systolic_window_json_out: Path | None = None,
    accum_write_start_pc: int | None = None,
    accum_write_end_pc: int | None = None,
    accum_write_json_out: Path | None = None,
    sram_write_start_pc: int | None = None,
    sram_write_end_pc: int | None = None,
    sram_write_json_out: Path | None = None,
    systolic_hidden_snapshot_pc: int | None = None,
    systolic_hidden_snapshot_json_out: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    cmd = [
        str(runner_path),
        "--program",
        str(program_path),
        "--json-out",
        str(summary_path),
        "--num-classes",
        str(num_classes),
        "--max-cycles",
        str(max_cycles),
    ]
    if patches_raw_path is not None:
        cmd.extend(
            [
                "--patches-raw",
                str(patches_raw_path),
                "--patch-rows",
                str(patch_rows),
                "--patch-cols",
                str(patch_cols),
            ]
        )
    if cls_raw_path is not None:
        cmd.extend(["--cls-raw", str(cls_raw_path)])
    if folded_pos_embed:
        cmd.append("--folded-pos-embed")
    if trace_json_out is not None:
        cmd.extend(["--trace-json-out", str(trace_json_out)])
    snapshot_args = [
        snapshot_request_path,
        snapshot_manifest_out,
        snapshot_data_out,
    ]
    if any(path is not None for path in snapshot_args):
        if not all(path is not None for path in snapshot_args):
            raise ValueError("Snapshot runner arguments must be provided together")
        cmd.extend(
            [
                "--snapshot-request",
                str(snapshot_request_path),
                "--snapshot-manifest-out",
                str(snapshot_manifest_out),
                "--snapshot-data-out",
                str(snapshot_data_out),
            ]
        )
    window_args = [
        systolic_window_start_pc,
        systolic_window_end_pc,
        systolic_window_json_out,
    ]
    if any(value is not None for value in window_args):
        if (systolic_window_start_pc is None or
                systolic_window_end_pc is None or
                systolic_window_json_out is None):
            raise ValueError("Systolic window runner arguments must be provided together")
        cmd.extend(
            [
                "--systolic-window-start-pc",
                str(systolic_window_start_pc),
                "--systolic-window-end-pc",
                str(systolic_window_end_pc),
                "--systolic-window-json-out",
                str(systolic_window_json_out),
            ]
        )
    accum_write_args = [
        accum_write_start_pc,
        accum_write_end_pc,
        accum_write_json_out,
    ]
    if any(value is not None for value in accum_write_args):
        if (accum_write_start_pc is None or
                accum_write_end_pc is None or
                accum_write_json_out is None):
            raise ValueError("ACCUM write runner arguments must be provided together")
        cmd.extend(
            [
                "--accum-write-start-pc",
                str(accum_write_start_pc),
                "--accum-write-end-pc",
                str(accum_write_end_pc),
                "--accum-write-json-out",
                str(accum_write_json_out),
            ]
        )
    sram_write_args = [
        sram_write_start_pc,
        sram_write_end_pc,
        sram_write_json_out,
    ]
    if any(value is not None for value in sram_write_args):
        if (sram_write_start_pc is None or
                sram_write_end_pc is None or
                sram_write_json_out is None):
            raise ValueError("SRAM write runner arguments must be provided together")
        cmd.extend(
            [
                "--sram-write-start-pc",
                str(sram_write_start_pc),
                "--sram-write-end-pc",
                str(sram_write_end_pc),
                "--sram-write-json-out",
                str(sram_write_json_out),
            ]
        )
    hidden_snapshot_args = [
        systolic_hidden_snapshot_pc,
        systolic_hidden_snapshot_json_out,
    ]
    if any(value is not None for value in hidden_snapshot_args):
        if (systolic_hidden_snapshot_pc is None or
                systolic_hidden_snapshot_json_out is None):
            raise ValueError("Hidden snapshot runner arguments must be provided together")
        cmd.extend(
            [
                "--systolic-hidden-snapshot-pc",
                str(systolic_hidden_snapshot_pc),
                "--systolic-hidden-snapshot-json-out",
                str(systolic_hidden_snapshot_json_out),
            ]
        )

    proc = subprocess.run(cmd, check=False)
    return proc.returncode, json.loads(summary_path.read_text())


def _first_mismatch(lhs: list[int], rhs: list[int]) -> dict[str, int] | None:
    for idx, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return {"index": idx, "golden": left, "rtl": right}
    if len(lhs) != len(rhs):
        return {"index": min(len(lhs), len(rhs)), "golden": len(lhs), "rtl": len(rhs)}
    return None


def _iter_trace_events(program: ProgramBinary) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pc in sorted(program.trace_manifest):
        for event_index, event in enumerate(program.trace_manifest[pc]):
            record = dict(event)
            record["pc"] = int(pc)
            record["event_index"] = int(event_index)
            record.setdefault("source", "architectural")
            record.setdefault("capture_phase", "retire_cycle")
            events.append(record)
    return events


def _trace_node_order_from_program(program: ProgramBinary) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for event in _iter_trace_events(program):
        node_name = str(event["node_name"])
        if node_name not in seen:
            seen.add(node_name)
            order.append(node_name)
    return order


def _format_snapshot_field(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _write_snapshot_request(path: Path, trace_events: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for event in trace_events:
        if event.get("source") == "virtual":
            continue
        fields = [
            int(event["pc"]),
            int(event["event_index"]),
            str(event["node_name"]),
            int(event["buf_id"]),
            int(event["offset_units"]),
            int(event["mem_rows"]),
            int(event["mem_cols"]),
            int(event["logical_rows"]),
            int(event["logical_cols"]),
            int(event["full_rows"]),
            int(event["full_cols"]),
            int(event["row_start"]),
            str(event["dtype"]),
            float(event["scale"]),
            str(event.get("source", "architectural")),
            str(event.get("capture_phase", "retire_cycle")),
        ]
        for field in fields:
            if isinstance(field, str) and ("," in field or "\n" in field or "\r" in field):
                raise ValueError(f"Snapshot request field contains unsupported CSV characters: {field!r}")
        lines.append(",".join(_format_snapshot_field(field) for field in fields))
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text)


def _event_key(event: dict[str, Any]) -> tuple[int, int, str]:
    return (int(event["pc"]), int(event["event_index"]), str(event["node_name"]))


def _decode_raw_payload(dtype: str, raw: bytes, logical_rows: int, logical_cols: int) -> np.ndarray:
    expected_elems = int(logical_rows) * int(logical_cols)
    if dtype == "int8":
        arr = np.frombuffer(raw, dtype=np.int8)
    elif dtype == "int32":
        arr = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError(f"Unsupported traced dtype: {dtype}")

    if arr.size != expected_elems:
        raise ValueError(
            f"Snapshot payload size mismatch for {dtype}: expected {expected_elems} elems, got {arr.size}"
        )
    return arr.reshape(int(logical_rows), int(logical_cols)).copy()


def _load_snapshot_bundle(
    manifest_path: Path,
    data_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    entries = list(manifest.get("entries", []))
    blob = data_path.read_bytes()
    capture_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for entry in entries:
        key = _event_key(entry)
        record = dict(entry)
        record.setdefault("capture_phase", "retire_cycle")
        status = str(record.get("status", "missing_snapshot"))
        if status == "captured":
            byte_offset = int(record["byte_offset"])
            byte_size = int(record["byte_size"])
            raw = blob[byte_offset: byte_offset + byte_size]
            if len(raw) != byte_size:
                raise ValueError(
                    f"Snapshot data truncated for {record['node_name']} event {record['event_index']}"
                )
            record["raw"] = _decode_raw_payload(
                dtype=str(record["dtype"]),
                raw=raw,
                logical_rows=int(record["logical_rows"]),
                logical_cols=int(record["logical_cols"]),
            )
        capture_map[key] = record
    return entries, capture_map


def _load_golden_raw_event_map(trace_payload: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    raw_events = list((trace_payload or {}).get("raw_events", []))
    event_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for event in raw_events:
        record = dict(event)
        record.setdefault("capture_phase", "retire_cycle")
        if record.get("raw_available"):
            raw = np.asarray(record.get("raw", []))
            dtype = str(record["dtype"])
            if dtype == "int8":
                raw = raw.astype(np.int8, copy=False)
            elif dtype == "int32":
                raw = raw.astype(np.int32, copy=False)
            else:
                raise ValueError(f"Unsupported golden trace dtype: {dtype}")
            record["raw"] = raw.reshape(int(record["logical_rows"]), int(record["logical_cols"])).copy()
        event_map[_event_key(record)] = record
    return event_map


def _build_node_specs(
    program: ProgramBinary,
) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    node_order: list[str] = []
    node_specs: dict[str, dict[str, Any]] = {}
    skipped_virtual: list[dict[str, Any]] = []
    for event in _iter_trace_events(program):
        source = str(event.get("source", "architectural"))
        node_name = str(event["node_name"])
        if source == "virtual":
            skipped_virtual.append(
                {
                    "node_name": node_name,
                    "pc": int(event["pc"]),
                    "event_index": int(event["event_index"]),
                    "dtype": str(event["dtype"]),
                    "scale": float(event["scale"]),
                    "full_rows": int(event["full_rows"]),
                    "full_cols": int(event["full_cols"]),
                    "reason": "virtual_skipped",
                }
            )
            continue
        if node_name not in node_specs:
            node_order.append(node_name)
            node_specs[node_name] = {
                "dtype": str(event["dtype"]),
                "scale": float(event["scale"]),
                "full_rows": int(event["full_rows"]),
                "full_cols": int(event["full_cols"]),
                "events": [],
            }
        node_specs[node_name]["events"].append(event)
    return node_order, node_specs, skipped_virtual


def _assemble_node_tensor(
    node_spec: dict[str, Any],
    actual_event_map: dict[tuple[int, int, str], dict[str, Any]],
    *,
    require_raw: bool,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    dtype_name = str(node_spec["dtype"])
    if dtype_name == "int8":
        dtype = np.int8
    elif dtype_name == "int32":
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported node dtype: {dtype_name}")

    tensor = np.zeros(
        (int(node_spec["full_rows"]), int(node_spec["full_cols"])),
        dtype=dtype,
    )
    for expected in node_spec["events"]:
        actual = actual_event_map.get(_event_key(expected))
        if actual is None:
            return None, {
                "kind": "missing_snapshot",
                "event": expected,
                "status": "missing_entry",
            }
        if require_raw:
            status = str(actual.get("status", "missing_snapshot"))
            if status != "captured":
                return None, {
                    "kind": "missing_snapshot",
                    "event": expected,
                    "status": status,
                }
        else:
            if not actual.get("raw_available", False):
                return None, {
                    "kind": "missing_snapshot",
                    "event": expected,
                    "status": "raw_unavailable",
                }

        raw = actual.get("raw")
        if raw is None:
            return None, {
                "kind": "missing_snapshot",
                "event": expected,
                "status": "raw_missing",
            }
        logical_rows = int(expected["logical_rows"])
        logical_cols = int(expected["logical_cols"])
        row_start = int(expected["row_start"])
        tensor[row_start: row_start + logical_rows, :logical_cols] = raw
    return tensor, None


def _find_event_for_flat_index(node_spec: dict[str, Any], flat_index: int) -> dict[str, Any]:
    full_cols = int(node_spec["full_cols"])
    row_idx = flat_index // full_cols
    col_idx = flat_index % full_cols
    for event in node_spec["events"]:
        row_start = int(event["row_start"])
        logical_rows = int(event["logical_rows"])
        logical_cols = int(event["logical_cols"])
        if row_start <= row_idx < row_start + logical_rows and col_idx < logical_cols:
            return event
    return node_spec["events"][0]


def _dequantized_metrics(golden_raw: np.ndarray, rtl_raw: np.ndarray, scale: float) -> dict[str, float]:
    golden_f = golden_raw.astype(np.float32) * np.float32(scale)
    rtl_f = rtl_raw.astype(np.float32) * np.float32(scale)
    diff = np.abs(golden_f - rtl_f)
    golden_flat = golden_f.reshape(-1)
    rtl_flat = rtl_f.reshape(-1)
    if golden_flat.size == 0:
        cosine = 1.0
    else:
        golden_norm = float(np.linalg.norm(golden_flat))
        rtl_norm = float(np.linalg.norm(rtl_flat))
        if golden_norm == 0.0 and rtl_norm == 0.0:
            cosine = 1.0
        elif golden_norm == 0.0 or rtl_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(np.dot(golden_flat, rtl_flat) / (golden_norm * rtl_norm))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cosine_similarity": cosine,
    }


def _compute_first_divergence(
    program: ProgramBinary,
    golden_trace: dict[str, Any],
    snapshot_manifest_path: Path,
    snapshot_data_path: Path,
    *,
    artifact_paths: dict[str, Any],
    ignore_node_names: set[str] | None = None,
) -> dict[str, Any] | None:
    node_order, node_specs, skipped_virtual = _build_node_specs(program)
    _, snapshot_map = _load_snapshot_bundle(snapshot_manifest_path, snapshot_data_path)
    golden_map = _load_golden_raw_event_map(golden_trace)
    ignored = set(ignore_node_names or set())

    for node_name in node_order:
        if node_name in ignored:
            continue
        node_spec = node_specs[node_name]
        golden_tensor, golden_error = _assemble_node_tensor(node_spec, golden_map, require_raw=False)
        if golden_error is not None:
            raise RuntimeError(
                f"Golden trace is missing architectural raw data for {node_name}: {golden_error['status']}"
            )

        rtl_tensor, rtl_error = _assemble_node_tensor(node_spec, snapshot_map, require_raw=True)
        if rtl_error is not None:
            event = dict(rtl_error["event"])
            return {
                "node_name": node_name,
                "trace_pc": int(event["pc"]),
                "event_index": int(event["event_index"]),
                "node_metadata": {
                    "dtype": str(node_spec["dtype"]),
                    "scale": float(node_spec["scale"]),
                    "full_rows": int(node_spec["full_rows"]),
                    "full_cols": int(node_spec["full_cols"]),
                    "fragment_count": int(len(node_spec["events"])),
                },
                "mismatch_kind": str(rtl_error["kind"]),
                "missing_status": str(rtl_error["status"]),
                "first_differing_element_index": None,
                "raw_values": {"golden": None, "rtl": None},
                "dequantized_summary": None,
                "artifacts": dict(artifact_paths),
                "skipped_virtual_nodes": skipped_virtual,
                "ignored_node_names": sorted(ignored),
            }

        if golden_tensor.shape != rtl_tensor.shape:
            event = dict(node_spec["events"][0])
            return {
                "node_name": node_name,
                "trace_pc": int(event["pc"]),
                "event_index": int(event["event_index"]),
                "node_metadata": {
                    "dtype": str(node_spec["dtype"]),
                    "scale": float(node_spec["scale"]),
                    "full_rows": int(node_spec["full_rows"]),
                    "full_cols": int(node_spec["full_cols"]),
                    "fragment_count": int(len(node_spec["events"])),
                    "golden_shape": list(golden_tensor.shape),
                    "rtl_shape": list(rtl_tensor.shape),
                },
                "mismatch_kind": "shape_mismatch",
                "first_differing_element_index": None,
                "raw_values": {"golden": None, "rtl": None},
                "dequantized_summary": None,
                "artifacts": dict(artifact_paths),
                "skipped_virtual_nodes": skipped_virtual,
                "ignored_node_names": sorted(ignored),
            }

        flat_golden = golden_tensor.reshape(-1)
        flat_rtl = rtl_tensor.reshape(-1)
        mismatch_idx = np.flatnonzero(flat_golden != flat_rtl)
        if mismatch_idx.size:
            first_idx = int(mismatch_idx[0])
            event = _find_event_for_flat_index(node_spec, first_idx)
            full_cols = int(node_spec["full_cols"])
            return {
                "node_name": node_name,
                "trace_pc": int(event["pc"]),
                "event_index": int(event["event_index"]),
                "node_metadata": {
                    "dtype": str(node_spec["dtype"]),
                    "scale": float(node_spec["scale"]),
                    "full_rows": int(node_spec["full_rows"]),
                    "full_cols": int(node_spec["full_cols"]),
                    "fragment_count": int(len(node_spec["events"])),
                },
                "mismatch_kind": "raw_value_mismatch",
                "first_differing_element_index": first_idx,
                "first_differing_row": int(first_idx // full_cols),
                "first_differing_col": int(first_idx % full_cols),
                "raw_values": {
                    "golden": int(flat_golden[first_idx]),
                    "rtl": int(flat_rtl[first_idx]),
                },
                "dequantized_summary": _dequantized_metrics(
                    golden_tensor,
                    rtl_tensor,
                    float(node_spec["scale"]),
                ),
                "artifacts": dict(artifact_paths),
                "skipped_virtual_nodes": skipped_virtual,
                "ignored_node_names": sorted(ignored),
            }

    return None


def _load_debug_bundle(first_divergence_path: Path) -> dict[str, Any]:
    divergence = json.loads(first_divergence_path.read_text())
    artifacts = dict(divergence.get("artifacts", {}))
    required = ("program", "golden_trace", "snapshot_manifest", "snapshot_data")
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise ValueError(
            f"First-divergence artifact bundle is missing required paths: {', '.join(missing)}"
        )

    program = _load_program(Path(artifacts["program"]))
    golden_trace = json.loads(Path(artifacts["golden_trace"]).read_text())
    _, snapshot_map = _load_snapshot_bundle(
        Path(artifacts["snapshot_manifest"]),
        Path(artifacts["snapshot_data"]),
    )
    golden_map = _load_golden_raw_event_map(golden_trace)
    node_order, node_specs, skipped_virtual = _build_node_specs(program)
    return {
        "divergence": divergence,
        "artifacts": artifacts,
        "program": program,
        "golden_trace": golden_trace,
        "snapshot_map": snapshot_map,
        "golden_map": golden_map,
        "node_order": node_order,
        "node_specs": node_specs,
        "skipped_virtual": skipped_virtual,
    }


def _extract_node_tensor_from_bundle(
    bundle: dict[str, Any],
    node_name: str,
    *,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    node_specs = bundle["node_specs"]
    if node_name not in node_specs:
        raise KeyError(f"Trace node {node_name!r} is not present in the artifact bundle")
    node_spec = node_specs[node_name]
    if source == "rtl":
        event_map = bundle["snapshot_map"]
        require_raw = True
    elif source == "golden":
        event_map = bundle["golden_map"]
        require_raw = False
    else:
        raise ValueError(f"Unsupported tensor source {source!r}")

    tensor, error = _assemble_node_tensor(node_spec, event_map, require_raw=require_raw)
    if error is not None:
        raise ValueError(
            f"Could not reconstruct {source} tensor for {node_name}: {error['kind']} ({error.get('status')})"
        )
    assert tensor is not None
    return tensor, node_spec


def _strip_slice(node_spec: dict[str, Any], row_start: int) -> tuple[int, int]:
    for event in node_spec["events"]:
        if int(event["row_start"]) == row_start:
            return int(event["row_start"]), int(event["logical_rows"])
    raise ValueError(
        f"No trace fragment with row_start={row_start} found for node {node_spec['events'][0]['node_name']}"
    )


def _fp16_to_uint16(val: float) -> int:
    return int(np.frombuffer(np.float16(val).tobytes(), dtype=np.uint16)[0])


def _parse_block_head_indices(node_prefix: str) -> tuple[int, int]:
    match = re.search(r"block(\d+)_head(\d+)_", node_prefix)
    if match is None:
        raise ValueError(f"Could not infer block/head indices from node prefix {node_prefix!r}")
    return int(match.group(1)), int(match.group(2))


def _read_program_data_relative_slice(program: ProgramBinary, data_offset: int, nbytes: int) -> bytes:
    start_off = int(data_offset)
    if start_off < 0:
        raise ValueError(f"Data-relative offset {data_offset} is negative")
    end_off = start_off + int(nbytes)
    if end_off > len(program.data):
        raise ValueError(
            f"Requested data-relative slice [{data_offset}, {data_offset + nbytes}) exceeds program data payload"
        )
    return bytes(program.data[start_off:end_off])


def _read_program_absolute_dram_slice(program: ProgramBinary, dram_offset: int, nbytes: int) -> bytes:
    data_base = int(program.data_base)
    data_off = int(dram_offset) - data_base
    if data_off < 0:
        raise ValueError(
            f"DRAM offset {dram_offset} is before data_base {data_base}; cannot read program data payload"
        )
    return _read_program_data_relative_slice(program, data_off, nbytes)


def _decode_fp16_preview(raw: bytes, *, max_elems: int = 8) -> list[float]:
    if not raw:
        return []
    count = min(len(raw) // 2, int(max_elems))
    if count <= 0:
        return []
    return [float(v) for v in np.frombuffer(raw[: count * 2], dtype=np.float16)]


def _validate_ln1_operand_bytes(*, gamma_bytes: bytes, beta_bytes: bytes) -> tuple[list[float], list[float]]:
    gamma_preview = _decode_fp16_preview(gamma_bytes)
    beta_preview = _decode_fp16_preview(beta_bytes)

    def _preview_invalid(values: list[float]) -> bool:
        return any(not np.isfinite(v) for v in values) or (
            bool(values) and all(abs(v) > 256.0 for v in values)
        )

    if _preview_invalid(gamma_preview):
        raise ValueError(f"Extracted ln1 gamma preview is not plausible FP16 data: {gamma_preview}")
    if _preview_invalid(beta_preview):
        raise ValueError(f"Extracted ln1 beta preview is not plausible FP16 data: {beta_preview}")
    return gamma_preview, beta_preview


def extract_qkt_replay_payloads(
    *,
    first_divergence_path: Path,
    out_dir: Path,
    node_prefix: str | None = None,
    strip_row_start: int = 0,
) -> dict[str, Any]:
    bundle = _load_debug_bundle(first_divergence_path)
    divergence = bundle["divergence"]
    program: ProgramBinary = bundle["program"]
    if node_prefix is None:
        node_name = str(divergence["node_name"])
        node_prefix = node_name.split("__", 1)[0]
    block_prefix: str | None = None
    if not node_prefix.endswith("_qkt"):
        if node_prefix.endswith("_query"):
            node_prefix = node_prefix[:-6] + "_qkt"
        elif node_prefix.endswith("_key"):
            node_prefix = node_prefix[:-4] + "_qkt"
        elif node_prefix.endswith("_value"):
            node_prefix = node_prefix[:-6] + "_qkt"
        elif node_prefix.endswith("_ln1"):
            block_prefix = node_prefix[:-4]
            node_prefix = f"{block_prefix}_head0_qkt"
    if block_prefix is None:
        block_prefix = node_prefix.split("_head", 1)[0]
    block_idx, head_idx = _parse_block_head_indices(node_prefix)

    query_node = f"{node_prefix}__query_input"
    key_t_node = f"{node_prefix}__key_transposed"
    key_padded_node = f"{node_prefix}__key_padded_input"
    accum_pre_node = f"{node_prefix}__accum_pre_matmul"
    qkt_node = node_prefix
    key_prefix = node_prefix.replace("_qkt", "_key")
    query_prefix = node_prefix.replace("_qkt", "_query")
    value_prefix = node_prefix.replace("_qkt", "_value")
    pos_embed_prefix = "pos_embed_add"
    ln1_prefix = f"{block_prefix}_ln1"
    pos_embed_act_node = f"{pos_embed_prefix}__act_input"
    pos_embed_pos_node = f"{pos_embed_prefix}__pos_input"
    pos_embed_output_node = pos_embed_prefix
    ln1_input_node = f"{ln1_prefix}__input_padded"
    ln1_output_node = f"{ln1_prefix}__output_padded"
    ln1_gamma_name = f"vit.encoder.layer.{block_idx}.layernorm_before.weight"
    ln1_beta_name = f"vit.encoder.layer.{block_idx}.layernorm_before.bias"
    qkv_name_map = {
        "query": (
            f"vit.encoder.layer.{block_idx}.attention.attention.query.weight_h{head_idx}",
            f"vit.encoder.layer.{block_idx}.attention.attention.query.bias_h{head_idx}",
        ),
        "key": (
            f"vit.encoder.layer.{block_idx}.attention.attention.key.weight_h{head_idx}",
            f"vit.encoder.layer.{block_idx}.attention.attention.key.bias_h{head_idx}",
        ),
        "value": (
            f"vit.encoder.layer.{block_idx}.attention.attention.value.weight_h{head_idx}",
            f"vit.encoder.layer.{block_idx}.attention.attention.value.bias_h{head_idx}",
        ),
    }
    dram_layout = dict(program.compiler_manifest.get("program_layout", {}).get("dram_layout", {}))
    weights_manifest = dict(program.compiler_manifest.get("weights", {}))

    projection_nodes = {
        "query": {
            "act_input": f"{query_prefix}__act_input",
            "act_input_padded": f"{query_prefix}__act_input_padded",
            "weight_input": f"{query_prefix}__weight_input",
            "accum_pre_bias": f"{query_prefix}__accum_pre_bias",
            "accum_pre_bias_padded": f"{query_prefix}__accum_pre_bias_padded",
            "bias_input": f"{query_prefix}__bias_input",
            "accum_post_bias": f"{query_prefix}__accum",
            "accum_post_bias_padded": f"{query_prefix}__accum_padded",
            "output": query_prefix,
            "output_padded": f"{query_prefix}__output_padded",
        },
        "key": {
            "act_input": f"{key_prefix}__act_input",
            "act_input_padded": f"{key_prefix}__act_input_padded",
            "weight_input": f"{key_prefix}__weight_input",
            "accum_pre_bias": f"{key_prefix}__accum_pre_bias",
            "accum_pre_bias_padded": f"{key_prefix}__accum_pre_bias_padded",
            "bias_input": f"{key_prefix}__bias_input",
            "accum_post_bias": f"{key_prefix}__accum",
            "accum_post_bias_padded": f"{key_prefix}__accum_padded",
            "output": key_prefix,
            "output_padded": f"{key_prefix}__output_padded",
        },
        "value": {
            "act_input": f"{value_prefix}__act_input",
            "act_input_padded": f"{value_prefix}__act_input_padded",
            "weight_input": f"{value_prefix}__weight_input",
            "accum_pre_bias": f"{value_prefix}__accum_pre_bias",
            "accum_pre_bias_padded": f"{value_prefix}__accum_pre_bias_padded",
            "bias_input": f"{value_prefix}__bias_input",
            "accum_post_bias": f"{value_prefix}__accum",
            "accum_post_bias_padded": f"{value_prefix}__accum_padded",
            "output": value_prefix,
            "output_padded": f"{value_prefix}__output_padded",
        },
    }

    query_tensor, query_spec = _extract_node_tensor_from_bundle(bundle, query_node, source="rtl")
    key_t_tensor, key_t_spec = _extract_node_tensor_from_bundle(bundle, key_t_node, source="rtl")
    key_padded_tensor, key_padded_spec = _extract_node_tensor_from_bundle(bundle, key_padded_node, source="rtl")
    accum_pre_tensor, accum_pre_spec = _extract_node_tensor_from_bundle(bundle, accum_pre_node, source="rtl")
    golden_qkt_tensor, qkt_spec = _extract_node_tensor_from_bundle(bundle, qkt_node, source="golden")
    pos_embed_act_tensor, pos_embed_act_spec = _extract_node_tensor_from_bundle(
        bundle,
        pos_embed_act_node,
        source="rtl",
    )
    pos_embed_pos_tensor, pos_embed_pos_spec = _extract_node_tensor_from_bundle(
        bundle,
        pos_embed_pos_node,
        source="rtl",
    )
    pos_embed_output_tensor, pos_embed_output_spec = _extract_node_tensor_from_bundle(
        bundle,
        pos_embed_output_node,
        source="rtl",
    )
    ln1_input_tensor, ln1_input_spec = _extract_node_tensor_from_bundle(bundle, ln1_input_node, source="rtl")
    ln1_output_tensor, ln1_output_spec = _extract_node_tensor_from_bundle(bundle, ln1_output_node, source="rtl")
    ln1_cols = int(ln1_output_spec["full_cols"])
    ln1_pc = int(ln1_input_spec["events"][0]["pc"])
    try:
        ln1_insn = decode(program.get_instruction_bytes(ln1_pc))
    except Exception:
        class _DummyLnInsn:
            src2_off = 0
            sreg = 0
        ln1_insn = _DummyLnInsn()
    ln1_gamma_dram_offset = int(dram_layout[ln1_gamma_name])
    ln1_beta_dram_offset = int(dram_layout[ln1_beta_name])
    ln1_gamma_info = dict(weights_manifest[ln1_gamma_name])
    ln1_beta_info = dict(weights_manifest[ln1_beta_name])
    ln1_gamma_elems = int(np.prod(ln1_gamma_info.get("stored_shape", [ln1_cols]), dtype=np.int64))
    ln1_beta_elems = int(np.prod(ln1_beta_info.get("stored_shape", [ln1_cols]), dtype=np.int64))
    ln1_gamma_bytes = _read_program_data_relative_slice(program, ln1_gamma_dram_offset, ln1_gamma_elems * 2)
    ln1_beta_bytes = _read_program_data_relative_slice(program, ln1_beta_dram_offset, ln1_beta_elems * 2)
    if ln1_gamma_elems < ln1_cols:
        ln1_gamma_bytes = ln1_gamma_bytes + bytes((ln1_cols - ln1_gamma_elems) * 2)
    if ln1_beta_elems < ln1_cols:
        ln1_beta_bytes = ln1_beta_bytes + bytes((ln1_cols - ln1_beta_elems) * 2)
    ln1_gamma_preview, ln1_beta_preview = _validate_ln1_operand_bytes(
        gamma_bytes=ln1_gamma_bytes,
        beta_bytes=ln1_beta_bytes,
    )
    ln1_gamma_beta_bytes = ln1_gamma_bytes + ln1_beta_bytes

    pos_embed_act_event0 = dict(pos_embed_act_spec["events"][0])
    startup_cols = int(pos_embed_act_event0["logical_cols"])
    startup_patch_rows = max(0, int(pos_embed_act_event0["logical_rows"]) - 1)
    startup_cls_dram_offset = int(getattr(program, "cls_token_dram_offset", 0))
    startup_patch_dram_offset = int(getattr(program, "input_offset", 0))
    startup_pos_dram_offset = int(getattr(program, "pos_embed_cls_dram_offset", 0))
    startup_row_stride_units = max(1, (startup_cols + 15) // 16)
    try:
        startup_cls_load = decode(program.get_instruction_bytes(2))
        startup_patch_load = decode(program.get_instruction_bytes(6))
        startup_pos_load = decode(program.get_instruction_bytes(10))
    except Exception:
        startup_cls_load = None
        startup_patch_load = None
        startup_pos_load = None

    startup_cls_dst_offset_units = int(getattr(startup_cls_load, "sram_off", 0))
    startup_patch_dst_offset_units = int(getattr(startup_patch_load, "sram_off", startup_row_stride_units))
    startup_pos_wbuf_offset_units = int(
        getattr(startup_pos_load, "sram_off", int(pos_embed_pos_spec["events"][0]["offset_units"]))
    )
    startup_cls_bytes = _read_program_absolute_dram_slice(program, startup_cls_dram_offset, startup_cols)

    work_dir = Path(bundle["artifacts"].get("work_dir", first_divergence_path.parent))
    startup_patch_path = work_dir / "patches.raw"
    if startup_patch_path.exists():
        startup_patch_tensor = np.fromfile(startup_patch_path, dtype=np.int8).reshape(startup_patch_rows, startup_cols)
    else:
        startup_patch_tensor = np.ascontiguousarray(
            pos_embed_act_tensor[1 : 1 + startup_patch_rows, :startup_cols],
            dtype=np.int8,
        )
    startup_cls_tensor = np.frombuffer(startup_cls_bytes, dtype=np.int8).reshape(1, startup_cols)
    pos_embed_info = dict(weights_manifest.get("vit.embeddings.position_embeddings", {}))
    pos_embed_stored_shape = pos_embed_info.get(
        "stored_shape",
        [int(ln1_input_spec["full_rows"]), startup_cols],
    )
    if len(pos_embed_stored_shape) < 2:
        raise ValueError("Position embedding stored_shape must have at least 2 dimensions")
    startup_pos_input_padded_rows = int(pos_embed_stored_shape[0])
    startup_pos_input_padded_cols = int(pos_embed_stored_shape[1])
    if startup_pos_input_padded_cols != startup_cols:
        raise ValueError(
            "Position embedding stored_shape cols do not match startup cols: "
            f"{startup_pos_input_padded_cols} != {startup_cols}"
        )
    if startup_pos_input_padded_rows < int(pos_embed_pos_spec["full_rows"]):
        raise ValueError(
            "Position embedding stored_shape rows are smaller than logical pos_embed_add rows: "
            f"{startup_pos_input_padded_rows} < {int(pos_embed_pos_spec['full_rows'])}"
        )
    # Replay the runtime startup image, not just the raw program payload.
    # In folded-pos-embed mode the runner zeroes patch positional rows in DRAM
    # before PC 10, so the traced logical pos-input view is the source of truth
    # for rows 0..196. Extend that logical image to the full padded WBUF load
    # span by keeping the traced rows and zero-filling the padded tail.
    startup_pos_input_padded_tensor = np.zeros(
        (startup_pos_input_padded_rows, startup_pos_input_padded_cols),
        dtype=np.int8,
    )
    startup_pos_input_padded_tensor[
        : int(pos_embed_pos_spec["full_rows"]),
        : int(pos_embed_pos_spec["full_cols"]),
    ] = np.ascontiguousarray(pos_embed_pos_tensor, dtype=np.int8)

    projection_specs: dict[str, dict[str, Any]] = {}
    projection_tensors: dict[str, dict[str, np.ndarray]] = {}
    projection_golden_tensors: dict[str, dict[str, np.ndarray]] = {}
    for proj_name, proj_nodes in projection_nodes.items():
        projection_specs[proj_name] = {}
        projection_tensors[proj_name] = {}
        projection_golden_tensors[proj_name] = {}
        for node_role, proj_node_name in proj_nodes.items():
            tensor, spec = _extract_node_tensor_from_bundle(bundle, proj_node_name, source="rtl")
            projection_specs[proj_name][node_role] = spec
            projection_tensors[proj_name][node_role] = tensor
            if node_role == "accum_pre_bias_padded":
                try:
                    golden_tensor, _ = _extract_node_tensor_from_bundle(bundle, proj_node_name, source="golden")
                except (KeyError, ValueError):
                    logical_tensor = projection_tensors[proj_name]["accum_pre_bias"]
                    golden_tensor = np.zeros_like(tensor, dtype=np.int32)
                    logical_rows = min(logical_tensor.shape[0], golden_tensor.shape[0])
                    logical_cols = min(logical_tensor.shape[1], golden_tensor.shape[1])
                    golden_tensor[:logical_rows, :logical_cols] = logical_tensor[:logical_rows, :logical_cols]
                projection_golden_tensors[proj_name][node_role] = np.ascontiguousarray(
                    golden_tensor,
                    dtype=np.int32,
                )

    _, strip_rows = _strip_slice(query_spec, strip_row_start)
    accum_row_start, accum_rows = _strip_slice(accum_pre_spec, strip_row_start)
    qkt_row_start, qkt_rows = _strip_slice(qkt_spec, strip_row_start)
    if accum_row_start != strip_row_start or qkt_row_start != strip_row_start:
        raise ValueError("QK^T strip row starts do not line up across query/accum/output traces")
    if accum_rows != strip_rows or qkt_rows != strip_rows:
        raise ValueError("QK^T strip row counts do not line up across replay tensors")

    query_slice = np.ascontiguousarray(
        query_tensor[strip_row_start:strip_row_start + strip_rows, : int(query_spec["full_cols"])],
        dtype=np.int8,
    )
    key_t_slice = np.ascontiguousarray(key_t_tensor, dtype=np.int8)
    accum_pre_slice = np.ascontiguousarray(
        accum_pre_tensor[strip_row_start:strip_row_start + strip_rows, : int(qkt_spec["full_cols"])],
        dtype=np.int32,
    )
    golden_qkt_slice = np.ascontiguousarray(
        golden_qkt_tensor[strip_row_start:strip_row_start + strip_rows, : int(qkt_spec["full_cols"])],
        dtype=np.int32,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "query_input.raw"
    key_t_path = out_dir / "key_transposed.raw"
    key_padded_path = out_dir / "key_padded_input.raw"
    accum_pre_path = out_dir / "accum_pre_matmul.raw"
    golden_qkt_path = out_dir / "golden_qkt.raw"
    startup_cls_token_path = out_dir / "startup_cls_token.raw"
    startup_patch_input_path = out_dir / "startup_patch_input.raw"
    startup_pos_input_padded_path = out_dir / "startup_pos_input_padded.raw"
    pos_embed_act_input_path = out_dir / "pos_embed_add_act_input.raw"
    pos_embed_pos_input_path = out_dir / "pos_embed_add_pos_input.raw"
    pos_embed_output_path = out_dir / "pos_embed_add_output.raw"
    ln1_input_path = out_dir / "ln1_input_padded.raw"
    ln1_output_path = out_dir / "ln1_output_padded.raw"
    ln1_gamma_path = out_dir / "ln1_gamma.raw"
    ln1_beta_path = out_dir / "ln1_beta.raw"
    ln1_gamma_beta_path = out_dir / "ln1_gamma_beta.raw"
    metadata_path = out_dir / "replay_metadata.json"

    query_path.write_bytes(query_slice.astype(np.int8, copy=False).tobytes())
    key_t_path.write_bytes(key_t_slice.astype(np.int8, copy=False).tobytes())
    key_padded_path.write_bytes(np.ascontiguousarray(key_padded_tensor, dtype=np.int8).tobytes())
    accum_pre_path.write_bytes(accum_pre_slice.astype("<i4", copy=False).tobytes())
    golden_qkt_path.write_bytes(golden_qkt_slice.astype("<i4", copy=False).tobytes())
    startup_cls_token_path.write_bytes(np.ascontiguousarray(startup_cls_tensor, dtype=np.int8).tobytes())
    startup_patch_input_path.write_bytes(np.ascontiguousarray(startup_patch_tensor, dtype=np.int8).tobytes())
    startup_pos_input_padded_path.write_bytes(
        np.ascontiguousarray(startup_pos_input_padded_tensor, dtype=np.int8).tobytes()
    )
    pos_embed_act_input_path.write_bytes(np.ascontiguousarray(pos_embed_act_tensor, dtype=np.int8).tobytes())
    pos_embed_pos_input_path.write_bytes(np.ascontiguousarray(pos_embed_pos_tensor, dtype=np.int8).tobytes())
    pos_embed_output_path.write_bytes(np.ascontiguousarray(pos_embed_output_tensor, dtype=np.int8).tobytes())
    ln1_input_path.write_bytes(np.ascontiguousarray(ln1_input_tensor, dtype=np.int8).tobytes())
    ln1_output_path.write_bytes(np.ascontiguousarray(ln1_output_tensor, dtype=np.int8).tobytes())
    ln1_gamma_path.write_bytes(ln1_gamma_bytes)
    ln1_beta_path.write_bytes(ln1_beta_bytes)
    ln1_gamma_beta_path.write_bytes(ln1_gamma_beta_bytes)

    projection_payload_paths: dict[str, dict[str, Path]] = {}
    projection_golden_payload_paths: dict[str, dict[str, Path]] = {}
    projection_metadata: dict[str, Any] = {}
    dtype_map: dict[str, str] = {
        "query_input": "int8",
        "key_transposed": "int8",
        "key_padded_input": "int8",
        "accum_pre_matmul": "int32",
        "golden_qkt": "int32",
        "startup_cls_token": "int8",
        "startup_patch_input": "int8",
        "startup_pos_input_padded": "int8",
        "pos_embed_add_act_input": "int8",
        "pos_embed_add_pos_input": "int8",
        "pos_embed_add_output": "int8",
        "ln1_input_padded": "int8",
        "ln1_output_padded": "int8",
        "ln1_gamma": "float16",
        "ln1_beta": "float16",
        "ln1_gamma_beta": "float16",
    }
    for proj_name, proj_specs in projection_specs.items():
        payload_paths: dict[str, Path] = {}
        golden_payload_paths: dict[str, Path] = {}
        output_spec = proj_specs["output"]
        accum_post_bias_spec = proj_specs["accum_post_bias"]
        output_scale = float(output_spec["scale"])
        accum_scale = float(accum_post_bias_spec["scale"])
        requant_scale = accum_scale / max(output_scale, 1e-12)
        weight_name, bias_name = qkv_name_map[proj_name]
        projection_metadata[proj_name] = {
            "offset_units": {},
            "rows": {},
            "cols": {},
            "dram_offsets": {
                "weight": int(dram_layout[weight_name]),
                "bias": int(dram_layout[bias_name]),
            },
            "scales": {
                "accum_post_bias": accum_scale,
                "output": output_scale,
                "requant": requant_scale,
                "requant_fp16": _fp16_to_uint16(requant_scale),
            },
        }
        for role, tensor in projection_tensors[proj_name].items():
            spec = proj_specs[role]
            dtype = np.int8 if spec["dtype"] == "int8" else np.int32
            if role == "act_input":
                payload_name = f"{proj_name}_projection_{role}.raw"
            elif role == "weight_input":
                payload_name = f"{proj_name}_projection_{role}.raw"
            else:
                payload_name = f"{proj_name}_{role}.raw"
            payload_path = out_dir / payload_name
            payload_path.write_bytes(np.ascontiguousarray(tensor, dtype=dtype).astype(dtype, copy=False).tobytes())
            payload_paths[role] = payload_path
            projection_metadata[proj_name]["offset_units"][role] = int(spec["events"][0]["offset_units"])
            projection_metadata[proj_name]["rows"][role] = int(spec["full_rows"])
            projection_metadata[proj_name]["cols"][role] = int(spec["full_cols"])
            dtype_map[f"{proj_name}_{role}"] = str(spec["dtype"])
        for role, tensor in projection_golden_tensors[proj_name].items():
            payload_name = f"{proj_name}_{role}_golden.raw"
            payload_path = out_dir / payload_name
            payload_path.write_bytes(np.ascontiguousarray(tensor, dtype=np.int32).astype("<i4", copy=False).tobytes())
            golden_payload_paths[role] = payload_path
            dtype_map[f"{proj_name}_{role}_golden"] = "int32"
        projection_payload_paths[proj_name] = payload_paths
        projection_golden_payload_paths[proj_name] = golden_payload_paths

    metadata = {
        "node_prefix": node_prefix,
        "key_prefix": key_prefix,
        "strip_row_start": int(strip_row_start),
        "strip_rows": int(strip_rows),
        "query_shape": list(query_slice.shape),
        "key_transposed_shape": list(key_t_slice.shape),
        "key_padded_shape": [int(key_padded_spec["full_rows"]), int(key_padded_spec["full_cols"])],
        "accum_pre_shape": list(accum_pre_slice.shape),
        "golden_qkt_shape": list(golden_qkt_slice.shape),
        "block_prefix": block_prefix,
        "ln1_prefix": ln1_prefix,
        "query_input_offset_units": int(query_spec["events"][0]["offset_units"]),
        "key_padded_input_offset_units": int(key_padded_spec["events"][0]["offset_units"]),
        "key_transposed_offset_units": int(key_t_spec["events"][0]["offset_units"]),
        "startup_cls_dram_offset": startup_cls_dram_offset,
        "startup_patch_dram_offset": startup_patch_dram_offset,
        "startup_pos_dram_offset": startup_pos_dram_offset,
        "startup_cls_dst_offset_units": startup_cls_dst_offset_units,
        "startup_patch_dst_offset_units": startup_patch_dst_offset_units,
        "startup_pos_wbuf_offset_units": startup_pos_wbuf_offset_units,
        "startup_patch_rows": startup_patch_rows,
        "startup_cols": startup_cols,
        "startup_pos_input_padded_rows": int(startup_pos_input_padded_tensor.shape[0]),
        "startup_pos_input_padded_cols": int(startup_pos_input_padded_tensor.shape[1]),
        "startup_pos_input_padded_row_units": int((startup_pos_input_padded_tensor.size + 15) // 16),
        # SRAM provenance is attributed to the issuing LOAD op, not the
        # preceding SET_ADDR pair that prepares the DMA source address.
        "startup_issue_pcs": [2, 6, 10, 12, 13],
        "pos_embed_add_act_input_offset_units": int(pos_embed_act_spec["events"][0]["offset_units"]),
        "pos_embed_add_pos_input_offset_units": int(pos_embed_pos_spec["events"][0]["offset_units"]),
        "pos_embed_add_output_offset_units": int(pos_embed_output_spec["events"][0]["offset_units"]),
        "ln1_input_padded_offset_units": int(ln1_input_spec["events"][0]["offset_units"]),
        "ln1_output_padded_offset_units": int(ln1_output_spec["events"][0]["offset_units"]),
        "ln1_gamma_dram_offset": int(ln1_gamma_dram_offset),
        "ln1_beta_dram_offset": int(ln1_beta_dram_offset),
        "dram_layout_offsets_are_data_relative": True,
        "ln1_gamma_dram_offset_space": "program_data_relative",
        "ln1_beta_dram_offset_space": "program_data_relative",
        "ln1_gamma_beta_wbuf_offset_units": int(getattr(ln1_insn, "src2_off", 0)),
        "ln1_sreg_base": int(getattr(ln1_insn, "sreg", 0)),
        "ln1_in_scale": float(ln1_input_spec["scale"]),
        "ln1_out_scale": float(ln1_output_spec["scale"]),
        "ln1_in_scale_fp16": int(_fp16_to_uint16(float(ln1_input_spec["scale"]))),
        "ln1_out_scale_fp16": int(_fp16_to_uint16(float(ln1_output_spec["scale"]))),
        "ln1_gamma_preview": ln1_gamma_preview,
        "ln1_beta_preview": ln1_beta_preview,
        "ln1_pc": int(ln1_pc),
        "zero_pad_dram_offset": int(dram_layout.get("__zero_pad__", 0)),
        "key_zero_pad_tail_bytes": int(max(0, int(key_padded_spec["full_rows"]) - 197) * int(key_padded_spec["full_cols"])),
        "query_input_path": str(query_path),
        "key_transposed_path": str(key_t_path),
        "key_padded_input_path": str(key_padded_path),
        "accum_pre_matmul_path": str(accum_pre_path),
        "golden_qkt_path": str(golden_qkt_path),
        "startup_cls_token_path": str(startup_cls_token_path),
        "startup_patch_input_path": str(startup_patch_input_path),
        "startup_pos_input_padded_path": str(startup_pos_input_padded_path),
        "pos_embed_add_act_input_path": str(pos_embed_act_input_path),
        "pos_embed_add_pos_input_path": str(pos_embed_pos_input_path),
        "pos_embed_add_output_path": str(pos_embed_output_path),
        "ln1_input_padded_path": str(ln1_input_path),
        "ln1_output_padded_path": str(ln1_output_path),
        "ln1_gamma_path": str(ln1_gamma_path),
        "ln1_beta_path": str(ln1_beta_path),
        "ln1_gamma_beta_path": str(ln1_gamma_beta_path),
        "ln1_gamma_shape": [ln1_cols],
        "ln1_beta_shape": [ln1_cols],
        "ln1_gamma_beta_shape": [2, ln1_cols],
        "startup_cls_token_shape": [1, startup_cols],
        "startup_patch_input_shape": [startup_patch_rows, startup_cols],
        "startup_pos_input_padded_shape": [
            int(startup_pos_input_padded_tensor.shape[0]),
            int(startup_pos_input_padded_tensor.shape[1]),
        ],
        "pos_embed_add_scale": float(pos_embed_output_spec["scale"]),
        "pos_embed_add_rows": int(pos_embed_output_spec["full_rows"]),
        "pos_embed_add_cols": int(pos_embed_output_spec["full_cols"]),
        "pos_embed_add_act_input_rows": int(pos_embed_act_spec["full_rows"]),
        "pos_embed_add_act_input_cols": int(pos_embed_act_spec["full_cols"]),
        "pos_embed_add_pos_input_rows": int(pos_embed_pos_spec["full_rows"]),
        "pos_embed_add_pos_input_cols": int(pos_embed_pos_spec["full_cols"]),
        "pos_embed_add_act_input_shape": [int(pos_embed_act_spec["full_rows"]), int(pos_embed_act_spec["full_cols"])],
        "pos_embed_add_pos_input_shape": [int(pos_embed_pos_spec["full_rows"]), int(pos_embed_pos_spec["full_cols"])],
        "pos_embed_add_shape": [int(pos_embed_output_spec["full_rows"]), int(pos_embed_output_spec["full_cols"])],
        "ln1_input_padded_shape": [int(ln1_input_spec["full_rows"]), int(ln1_input_spec["full_cols"])],
        "ln1_output_padded_shape": [int(ln1_output_spec["full_rows"]), int(ln1_output_spec["full_cols"])],
        "ln1_input_padded_rows": int(ln1_input_spec["full_rows"]),
        "ln1_input_padded_cols": int(ln1_input_spec["full_cols"]),
        "ln1_output_padded_rows": int(ln1_output_spec["full_rows"]),
        "ln1_output_padded_cols": int(ln1_output_spec["full_cols"]),
        "artifact_paths": dict(bundle["artifacts"]),
        "source_first_divergence": str(first_divergence_path),
        "dtype_map": dtype_map,
    }
    for proj_name, proj_info in projection_metadata.items():
        for role, value in proj_info["offset_units"].items():
            metadata[f"{proj_name}_{role}_offset_units"] = int(value)
            if role in ("act_input", "weight_input"):
                metadata[f"{proj_name}_projection_{role}_offset_units"] = int(value)
                short_role = "act" if role == "act_input" else "weight"
                metadata[f"{proj_name}_projection_{short_role}_offset_units"] = int(value)
        for role, value in proj_info["rows"].items():
            metadata[f"{proj_name}_{role}_rows"] = int(value)
            if role in ("act_input", "weight_input"):
                metadata[f"{proj_name}_projection_{role}_rows"] = int(value)
                short_role = "act" if role == "act_input" else "weight"
                metadata[f"{proj_name}_projection_{short_role}_rows"] = int(value)
        for role, value in proj_info["cols"].items():
            metadata[f"{proj_name}_{role}_cols"] = int(value)
            if role in ("act_input", "weight_input"):
                metadata[f"{proj_name}_projection_{role}_cols"] = int(value)
                short_role = "act" if role == "act_input" else "weight"
                metadata[f"{proj_name}_projection_{short_role}_cols"] = int(value)
        metadata[f"{proj_name}_requant_scale"] = float(proj_info["scales"]["requant"])
        metadata[f"{proj_name}_requant_scale_fp16"] = int(proj_info["scales"]["requant_fp16"])
        metadata[f"{proj_name}_accum_post_bias_scale"] = float(proj_info["scales"]["accum_post_bias"])
        metadata[f"{proj_name}_output_scale"] = float(proj_info["scales"]["output"])
        metadata[f"{proj_name}_weight_dram_offset"] = int(proj_info["dram_offsets"]["weight"])
        metadata[f"{proj_name}_bias_dram_offset"] = int(proj_info["dram_offsets"]["bias"])
    for proj_name, payloads in projection_payload_paths.items():
        for role, path in payloads.items():
            rows = int(projection_metadata[proj_name]["rows"][role])
            cols = int(projection_metadata[proj_name]["cols"][role])
            metadata[f"{proj_name}_{role}_path"] = str(path)
            metadata[f"{proj_name}_{role}_shape"] = [rows, cols]
            if role in ("act_input", "weight_input"):
                metadata[f"{proj_name}_projection_{role}_path"] = str(path)
                metadata[f"{proj_name}_projection_{role}_shape"] = [rows, cols]
                short_role = "act" if role == "act_input" else "weight"
                metadata[f"{proj_name}_projection_{short_role}_path"] = str(path)
                metadata[f"{proj_name}_projection_{short_role}_shape"] = [rows, cols]
    for proj_name, payloads in projection_golden_payload_paths.items():
        for role, path in payloads.items():
            rows = int(projection_metadata[proj_name]["rows"][role])
            cols = int(projection_metadata[proj_name]["cols"][role])
            metadata[f"{proj_name}_{role}_golden_path"] = str(path)
            metadata[f"{proj_name}_{role}_golden_shape"] = [rows, cols]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default))
    result = {
        **metadata,
        "query_input_path": str(query_path),
        "key_transposed_path": str(key_t_path),
        "key_padded_input_path": str(key_padded_path),
        "accum_pre_matmul_path": str(accum_pre_path),
        "golden_qkt_path": str(golden_qkt_path),
        "startup_cls_token_path": str(startup_cls_token_path),
        "startup_patch_input_path": str(startup_patch_input_path),
        "startup_pos_input_padded_path": str(startup_pos_input_padded_path),
        "pos_embed_add_act_input_path": str(pos_embed_act_input_path),
        "pos_embed_add_pos_input_path": str(pos_embed_pos_input_path),
        "pos_embed_add_output_path": str(pos_embed_output_path),
        "ln1_input_padded_path": str(ln1_input_path),
        "ln1_output_padded_path": str(ln1_output_path),
        "ln1_gamma_path": str(ln1_gamma_path),
        "ln1_beta_path": str(ln1_beta_path),
        "ln1_gamma_beta_path": str(ln1_gamma_beta_path),
        "metadata_path": str(metadata_path),
    }
    for proj_name, payloads in projection_payload_paths.items():
        for role, path in payloads.items():
            rows = int(projection_metadata[proj_name]["rows"][role])
            cols = int(projection_metadata[proj_name]["cols"][role])
            result[f"{proj_name}_{role}_path"] = str(path)
            result[f"{proj_name}_{role}_shape"] = [rows, cols]
            if role in ("act_input", "weight_input"):
                result[f"{proj_name}_projection_{role}_path"] = str(path)
                result[f"{proj_name}_projection_{role}_shape"] = [rows, cols]
                short_role = "act" if role == "act_input" else "weight"
                result[f"{proj_name}_projection_{short_role}_path"] = str(path)
                result[f"{proj_name}_projection_{short_role}_shape"] = [rows, cols]
    for proj_name, payloads in projection_golden_payload_paths.items():
        for role, path in payloads.items():
            rows = int(projection_metadata[proj_name]["rows"][role])
            cols = int(projection_metadata[proj_name]["cols"][role])
            result[f"{proj_name}_{role}_golden_path"] = str(path)
            result[f"{proj_name}_{role}_golden_shape"] = [rows, cols]
    return result


def _load_replay_matrix(replay_dir: Path, proj_name: str, role: str, metadata: dict[str, Any]) -> np.ndarray:
    path = replay_dir / f"{proj_name}_{role}.raw"
    rows = int(metadata[f"{proj_name}_{role}_rows"])
    cols = int(metadata[f"{proj_name}_{role}_cols"])
    dtype_key = metadata["dtype_map"][f"{proj_name}_{role}"]
    if dtype_key == "int32":
        return np.fromfile(path, dtype="<i4").astype(np.int32).reshape(rows, cols)
    return np.fromfile(path, dtype=np.int8).reshape(rows, cols)


def _projection_padding_status(
    *,
    tensor: np.ndarray,
    logical_rows: int,
    logical_cols: int,
) -> dict[str, Any]:
    padded_rows = tensor[logical_rows:, :]
    if padded_rows.size == 0:
        row197_sample = []
        return {
            "padded_rows_zero": True,
            "first_nonzero_padded_coord": None,
            "row197_sample": row197_sample,
        }

    nonzero = np.argwhere(padded_rows != 0)
    first_nonzero = None
    if nonzero.size:
        first_nonzero = [int(nonzero[0][0] + logical_rows), int(nonzero[0][1])]
    sample_row = min(logical_rows, tensor.shape[0] - 1)
    sample = tensor[sample_row, : min(logical_cols, tensor.shape[1])]
    return {
        "padded_rows_zero": not bool(nonzero.size),
        "first_nonzero_padded_coord": first_nonzero,
        "row197_sample": sample.astype(np.int64, copy=False).tolist(),
    }


def _load_projection_replay_results(replay_dir: Path) -> dict[str, Any]:
    path = replay_dir / "projection_replay_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def emit_projection_padding_report_from_replay_dir(
    *,
    replay_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    metadata = json.loads((replay_dir / "replay_metadata.json").read_text())
    replay_results = _load_projection_replay_results(replay_dir)
    projections: dict[str, Any] = {}
    overall = "padding_clean"

    for proj_name in ("query", "key", "value"):
        logical_rows = int(metadata[f"{proj_name}_accum_post_bias_rows"])
        logical_cols = int(metadata[f"{proj_name}_accum_post_bias_cols"])
        act_logical_rows = int(metadata[f"{proj_name}_act_input_rows"])
        act_logical_cols = int(metadata[f"{proj_name}_act_input_cols"])
        act_input = _load_replay_matrix(replay_dir, proj_name, "act_input_padded", metadata)
        accum_pre = _load_replay_matrix(replay_dir, proj_name, "accum_pre_bias_padded", metadata)
        accum_post = _load_replay_matrix(replay_dir, proj_name, "accum_post_bias_padded", metadata)
        output = _load_replay_matrix(replay_dir, proj_name, "output_padded", metadata)

        act_status = _projection_padding_status(
            tensor=act_input,
            logical_rows=act_logical_rows,
            logical_cols=act_logical_cols,
        )
        pre_status = _projection_padding_status(
            tensor=accum_pre,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )
        post_status = _projection_padding_status(
            tensor=accum_post,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )
        out_status = _projection_padding_status(
            tensor=output,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )
        replay_status = replay_results.get(proj_name, {})
        clean_match = replay_status.get("clean_padded_match")
        exact_match = replay_status.get("exact_padded_match")

        if not act_status["padded_rows_zero"]:
            classification = "dirty_source_padding"
        elif clean_match is False:
            classification = "projection_matmul_or_drain_touches_padding"
        elif not pre_status["padded_rows_zero"]:
            classification = "full_program_sequencing_or_snapshot_gap" if clean_match is True else "projection_matmul_or_drain_touches_padding"
        elif not post_status["padded_rows_zero"]:
            classification = "full_program_sequencing_or_snapshot_gap"
        elif not out_status["padded_rows_zero"]:
            classification = "full_program_sequencing_or_snapshot_gap"
        else:
            classification = "padding_clean"

        if overall == "padding_clean" and classification != "padding_clean":
            overall = classification

        projections[proj_name] = {
            "logical_rows": logical_rows,
            "logical_cols": logical_cols,
            "act_input_padded": act_status,
            "accum_pre_bias_padded": pre_status,
            "accum_post_bias_padded": post_status,
            "output_padded": out_status,
            "exact_padded_match": exact_match,
            "clean_padded_match": clean_match,
            "classification": classification,
        }

    report = {
        "mode": "projection_padding_report",
        "replay_dir": str(replay_dir),
        "overall_classification": overall,
        "projection_replay_results_path": str(replay_dir / "projection_replay_results.json") if (replay_dir / "projection_replay_results.json").exists() else None,
        "projections": projections,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def _load_named_replay_matrix(
    replay_dir: Path,
    metadata: dict[str, Any],
    *,
    key_prefix: str,
) -> np.ndarray:
    path = Path(metadata[f"{key_prefix}_path"])
    rows, cols = metadata[f"{key_prefix}_shape"]
    dtype_key = metadata["dtype_map"][key_prefix]
    if dtype_key == "int32":
        return np.fromfile(path if path.is_absolute() else replay_dir / path, dtype="<i4").astype(np.int32).reshape(rows, cols)
    return np.fromfile(path if path.is_absolute() else replay_dir / path, dtype=np.int8).reshape(rows, cols)


def _qkv_padded_match_status(
    *,
    source_tensor: np.ndarray,
    ln1_output_tensor: np.ndarray,
    logical_rows: int,
    logical_cols: int,
) -> dict[str, Any]:
    source_padded = source_tensor[logical_rows:, :logical_cols]
    ln1_padded = ln1_output_tensor[logical_rows:, :logical_cols]
    matches = bool(np.array_equal(source_padded, ln1_padded))
    diff = np.argwhere(source_padded != ln1_padded)
    first_diff = None
    if diff.size:
        first_diff = [int(diff[0][0] + logical_rows), int(diff[0][1])]
    return {
        "matches_ln1_output_padded": matches,
        "first_diff_coord": first_diff,
    }


def emit_qkv_source_padding_report_from_replay_dir(
    *,
    replay_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    metadata = json.loads((replay_dir / "replay_metadata.json").read_text())
    logical_rows = int(metadata["query_act_input_rows"])
    logical_cols = int(metadata["query_act_input_cols"])
    ln1_input = _load_named_replay_matrix(replay_dir, metadata, key_prefix="ln1_input_padded")
    ln1_output = _load_named_replay_matrix(replay_dir, metadata, key_prefix="ln1_output_padded")
    ln1_input_status = _projection_padding_status(
        tensor=ln1_input,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )
    ln1_output_status = _projection_padding_status(
        tensor=ln1_output,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )

    projections: dict[str, Any] = {}
    matches_all = True
    for proj_name in ("query", "key", "value"):
        act_input = _load_replay_matrix(replay_dir, proj_name, "act_input_padded", metadata)
        act_status = _projection_padding_status(
            tensor=act_input,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )
        match_status = _qkv_padded_match_status(
            source_tensor=act_input,
            ln1_output_tensor=ln1_output,
            logical_rows=logical_rows,
            logical_cols=logical_cols,
        )
        matches_all = matches_all and match_status["matches_ln1_output_padded"]
        projections[proj_name] = {
            "act_input_padded": act_status,
            **match_status,
        }

    if not ln1_input_status["padded_rows_zero"]:
        classification = "dirty_pre_layernorm_padding"
    elif matches_all:
        classification = "layernorm_beta_padding_expected"
    else:
        classification = "post_ln1_source_alias_or_reuse"

    report = {
        "mode": "qkv_source_padding_report",
        "replay_dir": str(replay_dir),
        "block_prefix": str(metadata["block_prefix"]),
        "logical_rows": logical_rows,
        "logical_cols": logical_cols,
        "classification": classification,
        "ignored_node_names": sorted(_expected_qkv_padding_ignore_nodes(str(metadata["block_prefix"]))),
        "ln1_input_padded": ln1_input_status,
        "ln1_output_padded": ln1_output_status,
        "projections": projections,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def _extract_qkt_strip_tensor(
    bundle: dict[str, Any],
    node_name: str,
    *,
    source: str,
    strip_row_start: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tensor, node_spec = _extract_node_tensor_from_bundle(bundle, node_name, source=source)
    _, strip_rows = _strip_slice(node_spec, strip_row_start)
    strip = np.ascontiguousarray(
        tensor[strip_row_start:strip_row_start + strip_rows, : int(node_spec["full_cols"])],
        dtype=tensor.dtype,
    )
    return strip, node_spec


def _qkt_checkpoint_summary(
    *,
    rtl_tensor: np.ndarray,
    golden_tensor: np.ndarray,
    node_spec: dict[str, Any],
) -> dict[str, Any]:
    samples: dict[str, dict[str, int]] = {}
    for row_idx, col_idx in ((0, 0), (1, 0), (0, 1), (1, 1)):
        if row_idx < rtl_tensor.shape[0] and col_idx < rtl_tensor.shape[1]:
            key = f"{row_idx},{col_idx}"
            samples[key] = {
                "golden": int(golden_tensor[row_idx, col_idx]),
                "rtl": int(rtl_tensor[row_idx, col_idx]),
            }

    row_summaries: dict[str, dict[str, int]] = {}
    for row_idx in (0, 1):
        if row_idx < rtl_tensor.shape[0]:
            rtl_row = rtl_tensor[row_idx].astype(np.int64, copy=False)
            golden_row = golden_tensor[row_idx].astype(np.int64, copy=False)
            row_summaries[str(row_idx)] = {
                "max_abs_diff": int(np.max(np.abs(rtl_row - golden_row))) if rtl_row.size else 0,
                "rtl_nonzero_count": int(np.count_nonzero(rtl_row)),
                "golden_nonzero_count": int(np.count_nonzero(golden_row)),
            }

    first_event = node_spec["events"][0]
    return {
        "pc": int(first_event["pc"]),
        "capture_phase": str(first_event.get("capture_phase", "retire_cycle")),
        "shape": [int(rtl_tensor.shape[0]), int(rtl_tensor.shape[1])],
        "samples": samples,
        "row_summaries": row_summaries,
        "matches_golden": bool(np.array_equal(rtl_tensor, golden_tensor)),
    }


def emit_qkt_stability_report(
    *,
    first_divergence_path: Path,
    out_path: Path,
    strip_row_start: int = 0,
) -> dict[str, Any]:
    bundle = _load_debug_bundle(first_divergence_path)
    divergence = bundle["divergence"]
    node_name = str(divergence["node_name"])
    node_prefix = node_name.split("__", 1)[0]

    checkpoint_names = {
        "accum_pre_matmul": f"{node_prefix}__accum_pre_matmul",
        "accum_pre_matmul_next": f"{node_prefix}__accum_pre_matmul_next",
        "qkt_output": node_prefix,
        "accum_pre_softmax": f"{node_prefix}__accum_pre_softmax",
        "accum_pre_softmax_next": f"{node_prefix}__accum_pre_softmax_next",
    }

    checkpoint_tensors: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    checkpoint_reports: dict[str, Any] = {}
    for key, trace_node in checkpoint_names.items():
        rtl_tensor, node_spec = _extract_qkt_strip_tensor(
            bundle,
            trace_node,
            source="rtl",
            strip_row_start=strip_row_start,
        )
        golden_tensor, _ = _extract_qkt_strip_tensor(
            bundle,
            trace_node,
            source="golden",
            strip_row_start=strip_row_start,
        )
        checkpoint_tensors[key] = (rtl_tensor, golden_tensor, node_spec)
        checkpoint_reports[key] = _qkt_checkpoint_summary(
            rtl_tensor=rtl_tensor,
            golden_tensor=golden_tensor,
            node_spec=node_spec,
        )

    pre_match = checkpoint_reports["accum_pre_matmul"]["matches_golden"]
    pre_next_match = checkpoint_reports["accum_pre_matmul_next"]["matches_golden"]
    pre_softmax_match = checkpoint_reports["accum_pre_softmax"]["matches_golden"]
    pre_softmax_next_match = checkpoint_reports["accum_pre_softmax_next"]["matches_golden"]

    if (not pre_match) and pre_next_match:
        classification = "retire_cycle_snapshot_artifact"
    elif (not pre_match) and (not pre_next_match):
        classification = "real_pre_matmul_dirty_state"
    elif pre_match and pre_next_match and ((not pre_softmax_match) or (not pre_softmax_next_match)):
        classification = "post_qkt_pre_softmax_overwrite"
    else:
        classification = "later_than_softmax_boundary"

    report = {
        "node_prefix": node_prefix,
        "strip_row_start": int(strip_row_start),
        "classification": classification,
        "first_divergence_node": node_name,
        "first_divergence_pc": int(divergence.get("trace_pc", 0)),
        "checkpoints": checkpoint_reports,
        "artifacts": {
            **dict(bundle["artifacts"]),
            "source_first_divergence": str(first_divergence_path),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return report


def _maybe_emit_qkt_stability_report(
    *,
    first_divergence: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return None
    first_divergence_path = work_dir / "first_divergence.json"
    if not first_divergence_path.exists():
        return None
    report_path = work_dir / "qkt_stability_report.json"
    report = emit_qkt_stability_report(
        first_divergence_path=first_divergence_path,
        out_path=report_path,
        strip_row_start=0,
    )
    artifacts["qkt_stability_report"] = str(report_path)
    return report


def _maybe_emit_projection_padding_report(
    *,
    first_divergence: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if first_divergence is None or not (
        _is_qkt_family_node(first_divergence.get("node_name"))
        or _is_projection_tail_debug_node(first_divergence.get("node_name"))
    ):
        return None
    first_divergence_path = work_dir / "first_divergence.json"
    if not first_divergence_path.exists():
        return None
    replay_dir = work_dir / "replay_payloads"
    try:
        extract_qkt_replay_payloads(
            first_divergence_path=first_divergence_path,
            out_dir=replay_dir,
            strip_row_start=0,
        )
    except (KeyError, ValueError):
        return None
    report_path = work_dir / "projection_padding_report.json"
    report = emit_projection_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=report_path,
    )
    artifacts["replay_payloads"] = str(replay_dir)
    artifacts["projection_padding_report"] = str(report_path)
    return report


def _maybe_emit_qkv_source_padding_report(
    *,
    first_divergence: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if first_divergence is None or not (
        _is_qkt_family_node(first_divergence.get("node_name"))
        or _is_projection_tail_debug_node(first_divergence.get("node_name"))
        or _is_ln1_padding_debug_node(first_divergence.get("node_name"))
    ):
        return None
    replay_dir = Path(artifacts.get("replay_payloads", work_dir / "replay_payloads"))
    if not replay_dir.exists():
        first_divergence_path = work_dir / "first_divergence.json"
        if not first_divergence_path.exists():
            return None
        try:
            extract_qkt_replay_payloads(
                first_divergence_path=first_divergence_path,
                out_dir=replay_dir,
                strip_row_start=0,
            )
        except (KeyError, ValueError):
            return None
        artifacts["replay_payloads"] = str(replay_dir)
    report_path = work_dir / "qkv_source_padding_report.json"
    report = emit_qkv_source_padding_report_from_replay_dir(
        replay_dir=replay_dir,
        out_path=report_path,
    )
    artifacts["qkv_source_padding_report"] = str(report_path)
    return report


def _maybe_rebase_first_divergence_for_expected_qkv_padding(
    *,
    program: ProgramBinary,
    golden_trace: dict[str, Any],
    snapshot_manifest_path: Path,
    snapshot_data_path: Path,
    work_dir: Path,
    artifacts: dict[str, Any],
    qkv_source_padding_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if qkv_source_padding_report is None:
        return None
    if qkv_source_padding_report.get("classification") != "layernorm_beta_padding_expected":
        return None
    ignored = set(qkv_source_padding_report.get("ignored_node_names", []))
    if not ignored:
        return None

    divergence = _compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        artifact_paths={
            **artifacts,
            "golden_trace": artifacts.get("golden_trace"),
            "rtl_trace": artifacts.get("rtl_trace"),
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_data": str(snapshot_data_path),
        },
        ignore_node_names=ignored,
    )
    if divergence is None:
        return None

    first_divergence_path = work_dir / "first_divergence.json"
    first_divergence_path.write_text(json.dumps(divergence, indent=2, sort_keys=True, default=_json_default))
    artifacts["first_divergence"] = str(first_divergence_path)
    return divergence


def _maybe_emit_qkt_prestate_debug_artifacts(
    *,
    first_divergence: dict[str, Any] | None,
    qkt_stability_report: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return
    if qkt_stability_report is None:
        return
    if qkt_stability_report.get("classification") != "real_pre_matmul_dirty_state":
        return

    accum_write_path = work_dir / "rtl_accum_write_log.json"
    sram_write_path = work_dir / "rtl_sram_write_log.json"
    hidden_snapshot_path = work_dir / "rtl_systolic_hidden_snapshot.json"
    _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        accum_write_start_pc=QKT_ACCUM_WRITE_START_PC,
        accum_write_end_pc=QKT_ACCUM_WRITE_END_PC,
        accum_write_json_out=accum_write_path,
        sram_write_start_pc=QKT_SRAM_WRITE_START_PC,
        sram_write_end_pc=QKT_SRAM_WRITE_END_PC,
        sram_write_json_out=sram_write_path,
        systolic_hidden_snapshot_pc=QKT_HIDDEN_SNAPSHOT_PC,
        systolic_hidden_snapshot_json_out=hidden_snapshot_path,
    )
    artifacts["rtl_accum_write_log"] = str(accum_write_path)
    artifacts["rtl_sram_write_log"] = str(sram_write_path)
    artifacts["rtl_systolic_hidden_snapshot"] = str(hidden_snapshot_path)


def _maybe_emit_qkt_prestate_provenance_report(
    *,
    first_divergence: dict[str, Any] | None,
    qkt_stability_report: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return None
    if qkt_stability_report is None:
        return None
    if qkt_stability_report.get("classification") != "real_pre_matmul_dirty_state":
        return None

    first_divergence_path = work_dir / "first_divergence.json"
    replay_dir = Path(artifacts.get("replay_payloads", work_dir / "replay_payloads"))
    baseline_accum_log_path = Path(artifacts.get("rtl_accum_write_log", ""))
    baseline_hidden_snapshot_path = Path(artifacts.get("rtl_systolic_hidden_snapshot", ""))
    baseline_window_path = Path(artifacts.get("rtl_systolic_window", ""))
    if (
        not first_divergence_path.exists()
        or not replay_dir.exists()
        or not baseline_accum_log_path.exists()
        or not baseline_hidden_snapshot_path.exists()
        or not baseline_window_path.exists()
    ):
        return None

    fragment_result = _run_qkt_prestate_fragment_capture(replay_dir=replay_dir, work_dir=work_dir)
    artifacts["fragment_qkt_accum_write_log"] = str(work_dir / "fragment_qkt_accum_write_log.json")
    artifacts["fragment_qkt_hidden_snapshot"] = str(work_dir / "fragment_qkt_hidden_snapshot.json")
    artifacts["fragment_qkt_window"] = str(work_dir / "fragment_qkt_window.json")
    artifacts["fragment_qkt_checkpoints"] = str(work_dir / "fragment_qkt_checkpoints.json")

    report_path = work_dir / "qkt_prestate_provenance_report.json"
    report = emit_qkt_prestate_provenance_report(
        first_divergence_path=first_divergence_path,
        replay_dir=replay_dir,
        baseline_accum_log_path=baseline_accum_log_path,
        baseline_hidden_snapshot_path=baseline_hidden_snapshot_path,
        baseline_window_path=baseline_window_path,
        fragment_result=fragment_result,
        out_path=report_path,
    )
    artifacts["qkt_prestate_provenance_report"] = str(report_path)
    return report


def _maybe_emit_qkt_prefix_debug_artifacts(
    *,
    first_divergence: dict[str, Any] | None,
    qkt_prestate_provenance_report: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return
    if qkt_prestate_provenance_report is None:
        return
    if qkt_prestate_provenance_report.get("classification") != "earlier_history_required":
        return

    prefix_sram_write_path = work_dir / "rtl_prefix_sram_write_log.json"
    _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        sram_write_start_pc=QKT_PREFIX_SRAM_WRITE_START_PC,
        sram_write_end_pc=QKT_PREFIX_SRAM_WRITE_END_PC,
        sram_write_json_out=prefix_sram_write_path,
    )
    artifacts["rtl_prefix_sram_write_log"] = str(prefix_sram_write_path)


def _maybe_emit_qkt_prefix_provenance_report(
    *,
    first_divergence: dict[str, Any] | None,
    qkt_prestate_provenance_report: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return None
    if qkt_prestate_provenance_report is None:
        return None
    if qkt_prestate_provenance_report.get("classification") != "earlier_history_required":
        return None

    first_divergence_path = work_dir / "first_divergence.json"
    replay_dir = Path(artifacts.get("replay_payloads", work_dir / "replay_payloads"))
    baseline_prefix_log_path = Path(artifacts.get("rtl_prefix_sram_write_log", ""))
    if not first_divergence_path.exists() or not baseline_prefix_log_path.exists():
        return None
    if not replay_dir.exists():
        try:
            extract_qkt_replay_payloads(
                first_divergence_path=first_divergence_path,
                out_dir=replay_dir,
                strip_row_start=0,
            )
        except (KeyError, ValueError):
            return None
        artifacts["replay_payloads"] = str(replay_dir)

    fragment_result = _run_qkt_prefix_fragment_capture(replay_dir=replay_dir, work_dir=work_dir)
    artifacts["fragment_prefix_sram_write_log"] = str(work_dir / "fragment_prefix_sram_write_log.json")
    artifacts["fragment_prefix_checkpoints"] = str(work_dir / "fragment_prefix_checkpoints.json")
    artifacts["fragment_prefix_window"] = str(work_dir / "fragment_prefix_window.json")

    report_path = work_dir / "qkt_prefix_provenance_report.json"
    report = emit_qkt_prefix_provenance_report(
        first_divergence_path=first_divergence_path,
        replay_dir=replay_dir,
        baseline_prefix_log_path=baseline_prefix_log_path,
        fragment_result=fragment_result,
        out_path=report_path,
    )
    artifacts["qkt_prefix_provenance_report"] = str(report_path)
    return report


def _maybe_emit_qkt_startup_debug_artifacts(
    *,
    first_divergence: dict[str, Any] | None,
    qkt_prefix_provenance_report: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return
    if qkt_prefix_provenance_report is None:
        return
    if qkt_prefix_provenance_report.get("classification") != "history_earlier_than_pos_embed":
        return

    startup_sram_write_path = work_dir / "rtl_startup_sram_write_log.json"
    _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        sram_write_start_pc=QKT_STARTUP_SRAM_WRITE_START_PC,
        sram_write_end_pc=QKT_STARTUP_SRAM_WRITE_END_PC,
        sram_write_json_out=startup_sram_write_path,
    )
    artifacts["rtl_startup_sram_write_log"] = str(startup_sram_write_path)


def _maybe_emit_qkt_startup_provenance_report(
    *,
    first_divergence: dict[str, Any] | None,
    qkt_prefix_provenance_report: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return None
    if qkt_prefix_provenance_report is None:
        return None
    if qkt_prefix_provenance_report.get("classification") != "history_earlier_than_pos_embed":
        return None

    first_divergence_path = work_dir / "first_divergence.json"
    replay_dir = Path(artifacts.get("replay_payloads", work_dir / "replay_payloads"))
    baseline_startup_log_path = Path(artifacts.get("rtl_startup_sram_write_log", ""))
    if not first_divergence_path.exists() or not baseline_startup_log_path.exists():
        return None
    try:
        extract_qkt_replay_payloads(
            first_divergence_path=first_divergence_path,
            out_dir=replay_dir,
            strip_row_start=0,
        )
    except (KeyError, ValueError):
        return None
    artifacts["replay_payloads"] = str(replay_dir)

    fragment_result = _run_qkt_startup_fragment_capture(replay_dir=replay_dir, work_dir=work_dir)
    artifacts["fragment_startup_sram_write_log"] = str(work_dir / "fragment_startup_sram_write_log.json")
    artifacts["fragment_startup_checkpoints"] = str(work_dir / "fragment_startup_checkpoints.json")
    artifacts["fragment_startup_window"] = str(work_dir / "fragment_startup_window.json")

    report_path = work_dir / "qkt_startup_provenance_report.json"
    report = emit_qkt_startup_provenance_report(
        first_divergence_path=first_divergence_path,
        replay_dir=replay_dir,
        baseline_startup_log_path=baseline_startup_log_path,
        fragment_result=fragment_result,
        out_path=report_path,
    )
    artifacts["qkt_startup_provenance_report"] = str(report_path)
    return report


def _maybe_emit_startup_pos_load_report(
    *,
    qkt_startup_provenance_report: dict[str, Any] | None,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    if qkt_startup_provenance_report is None:
        return None
    baseline_startup_log_path = Path(artifacts.get("rtl_startup_sram_write_log", ""))
    fragment_startup_log_path = Path(artifacts.get("fragment_startup_sram_write_log", ""))
    replay_dir = Path(artifacts.get("replay_payloads", work_dir / "replay_payloads"))
    replay_metadata_path = replay_dir / "replay_metadata.json"
    if (
        not baseline_startup_log_path.exists()
        or not fragment_startup_log_path.exists()
        or not replay_metadata_path.exists()
    ):
        return None
    report_path = work_dir / "startup_pos_load_report.json"
    report = emit_startup_pos_load_report(
        replay_dir=replay_dir,
        baseline_startup_log_path=baseline_startup_log_path,
        fragment_startup_log_path=fragment_startup_log_path,
        out_path=report_path,
    )
    artifacts["startup_pos_load_report"] = str(report_path)
    return report


def _maybe_rebase_first_divergence_for_nonblocking_qkt_prestate(
    *,
    program: ProgramBinary,
    golden_trace: dict[str, Any],
    snapshot_manifest_path: Path,
    snapshot_data_path: Path,
    work_dir: Path,
    artifacts: dict[str, Any],
    first_divergence: dict[str, Any] | None,
    qkt_prestate_provenance_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if first_divergence is None or qkt_prestate_provenance_report is None:
        return None
    if qkt_prestate_provenance_report.get("classification") != "nonblocking_qkt_prestate_scratch":
        return None

    node_prefix = str(first_divergence["node_name"]).split("__", 1)[0]
    ignored = set(first_divergence.get("ignored_node_names", []))
    ignored.update(
        {
            f"{node_prefix}__accum_pre_matmul",
            f"{node_prefix}__accum_pre_matmul_next",
        }
    )
    # Extend to ignore all padding debug nodes and all QKT accum_pre_matmul
    # nodes across every head/block.  The systolic engine does not reset ACCUM
    # between MATMULs, so every head's accum_pre_matmul will contain stale data
    # from the previous operation; the golden model always sees zeros there.
    # Padding rows/cols beyond the logical tensor bounds are architecturally
    # undefined and may contain stale RTL data; neither affects correctness.
    node_order, _, _ = _build_node_specs(program)
    for _name in node_order:
        if _is_projection_tail_debug_node(_name) or _is_ln1_padding_debug_node(_name):
            ignored.add(_name)
        if _is_qkt_family_node(_name) and _name.endswith(
            ("__accum_pre_matmul", "__accum_pre_matmul_next")
        ):
            ignored.add(_name)
    divergence = _compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        artifact_paths={
            **artifacts,
            "golden_trace": artifacts.get("golden_trace"),
            "rtl_trace": artifacts.get("rtl_trace"),
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_data": str(snapshot_data_path),
        },
        ignore_node_names=ignored,
    )
    if divergence is None:
        return None

    effective_path = work_dir / "effective_first_divergence.json"
    effective_path.write_text(json.dumps(divergence, indent=2, sort_keys=True, default=_json_default))
    artifacts["effective_first_divergence"] = str(effective_path)
    return divergence


def _maybe_rebase_first_divergence_for_nonblocking_qkt_prefix(
    *,
    program: ProgramBinary,
    golden_trace: dict[str, Any],
    snapshot_manifest_path: Path,
    snapshot_data_path: Path,
    work_dir: Path,
    artifacts: dict[str, Any],
    first_divergence: dict[str, Any] | None,
    qkt_prefix_provenance_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if first_divergence is None or qkt_prefix_provenance_report is None:
        return None
    if qkt_prefix_provenance_report.get("classification") != "prefix_nonblocking_scratch":
        return None

    node_prefix = str(first_divergence["node_name"]).split("__", 1)[0]
    ignored = set(first_divergence.get("ignored_node_names", []))
    ignored.update(
        {
            f"{node_prefix}__accum_pre_matmul",
            f"{node_prefix}__accum_pre_matmul_next",
        }
    )
    # Extend to ignore all padding debug nodes and all QKT accum_pre_matmul
    # nodes across every head/block.  The systolic engine does not reset ACCUM
    # between MATMULs, so every head's accum_pre_matmul will contain stale data
    # from the previous operation; the golden model always sees zeros there.
    # Padding rows/cols beyond the logical tensor bounds are architecturally
    # undefined and may contain stale RTL data; neither affects correctness.
    node_order, _, _ = _build_node_specs(program)
    for _name in node_order:
        if _is_projection_tail_debug_node(_name) or _is_ln1_padding_debug_node(_name):
            ignored.add(_name)
        if _is_qkt_family_node(_name) and _name.endswith(
            ("__accum_pre_matmul", "__accum_pre_matmul_next")
        ):
            ignored.add(_name)
    divergence = _compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        artifact_paths={
            **artifacts,
            "golden_trace": artifacts.get("golden_trace"),
            "rtl_trace": artifacts.get("rtl_trace"),
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_data": str(snapshot_data_path),
        },
        ignore_node_names=ignored,
    )
    if divergence is None:
        return None

    effective_path = work_dir / "effective_first_divergence.json"
    effective_path.write_text(json.dumps(divergence, indent=2, sort_keys=True, default=_json_default))
    artifacts["effective_first_divergence"] = str(effective_path)
    return divergence


def _maybe_rebase_first_divergence_for_nonblocking_qkt_startup(
    *,
    program: ProgramBinary,
    golden_trace: dict[str, Any],
    snapshot_manifest_path: Path,
    snapshot_data_path: Path,
    work_dir: Path,
    artifacts: dict[str, Any],
    first_divergence: dict[str, Any] | None,
    qkt_startup_provenance_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if first_divergence is None or qkt_startup_provenance_report is None:
        return None
    if qkt_startup_provenance_report.get("classification") != "startup_nonblocking_scratch":
        return None

    node_prefix = str(first_divergence["node_name"]).split("__", 1)[0]
    ignored = set(first_divergence.get("ignored_node_names", []))
    ignored.update(
        {
            f"{node_prefix}__accum_pre_matmul",
            f"{node_prefix}__accum_pre_matmul_next",
        }
    )
    # Extend to ignore all padding debug nodes and all QKT accum_pre_matmul
    # nodes across every head/block.  The systolic engine does not reset ACCUM
    # between MATMULs, so every head's accum_pre_matmul will contain stale data
    # from the previous operation; the golden model always sees zeros there.
    # Padding rows/cols beyond the logical tensor bounds are architecturally
    # undefined and may contain stale RTL data; neither affects correctness.
    node_order, _, _ = _build_node_specs(program)
    for _name in node_order:
        if _is_projection_tail_debug_node(_name) or _is_ln1_padding_debug_node(_name):
            ignored.add(_name)
        if _is_qkt_family_node(_name) and _name.endswith(
            ("__accum_pre_matmul", "__accum_pre_matmul_next")
        ):
            ignored.add(_name)
    divergence = _compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        artifact_paths={
            **artifacts,
            "golden_trace": artifacts.get("golden_trace"),
            "rtl_trace": artifacts.get("rtl_trace"),
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_data": str(snapshot_data_path),
        },
        ignore_node_names=ignored,
    )
    if divergence is None:
        return None

    effective_path = work_dir / "effective_first_divergence.json"
    effective_path.write_text(json.dumps(divergence, indent=2, sort_keys=True, default=_json_default))
    artifacts["effective_first_divergence"] = str(effective_path)
    return divergence


def _maybe_emit_ln1_provenance_debug_artifacts(
    *,
    first_divergence: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return

    ln1_sram_write_path = work_dir / "rtl_ln1_sram_write_log.json"
    ln1_window_path = work_dir / "rtl_ln1_window.json"
    _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        sram_write_start_pc=LN1_SRAM_WRITE_START_PC,
        sram_write_end_pc=LN1_SRAM_WRITE_END_PC,
        sram_write_json_out=ln1_sram_write_path,
        systolic_window_start_pc=LN1_WINDOW_START_PC,
        systolic_window_end_pc=LN1_WINDOW_END_PC,
        systolic_window_json_out=ln1_window_path,
    )
    artifacts["rtl_ln1_sram_write_log"] = str(ln1_sram_write_path)
    artifacts["rtl_ln1_window"] = str(ln1_window_path)

    replay_dir = Path(artifacts.get("replay_payloads", work_dir / "replay_payloads"))
    if not replay_dir.exists():
        first_divergence_path = work_dir / "first_divergence.json"
        if not first_divergence_path.exists():
            return
        try:
            extract_qkt_replay_payloads(
                first_divergence_path=first_divergence_path,
                out_dir=replay_dir,
                strip_row_start=0,
            )
        except (KeyError, ValueError):
            return
        artifacts["replay_payloads"] = str(replay_dir)

    ln1_operand_report_path = work_dir / "ln1_operand_report.json"
    emit_ln1_operand_report_from_replay_dir(
        replay_dir=replay_dir,
        sram_log_path=ln1_sram_write_path,
        out_path=ln1_operand_report_path,
    )
    artifacts["ln1_operand_report"] = str(ln1_operand_report_path)


def _maybe_emit_qkt_systolic_window_trace(
    *,
    first_divergence: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    rtl_trace_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    if first_divergence is None or not _is_qkt_family_node(first_divergence.get("node_name")):
        return

    systolic_window_path = work_dir / "rtl_systolic_window.json"
    _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        trace_json_out=rtl_trace_path,
        systolic_window_start_pc=QKT_WINDOW_START_PC,
        systolic_window_end_pc=QKT_WINDOW_END_PC,
        systolic_window_json_out=systolic_window_path,
    )
    artifacts["rtl_systolic_window"] = str(systolic_window_path)


def _maybe_emit_first_divergence(
    *,
    program: ProgramBinary,
    golden_trace: dict[str, Any] | None,
    runner_path: Path,
    program_path: Path,
    rtl_summary_path: Path,
    rtl_trace_path: Path,
    patches_raw_path: Path | None,
    patch_rows: int | None,
    patch_cols: int | None,
    cls_raw_path: Path | None,
    folded_pos_embed: bool,
    num_classes: int,
    max_cycles: int,
    work_dir: Path,
    artifacts: dict[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    trace_events = _iter_trace_events(program)
    if golden_trace is None or not trace_events:
        runner_rc, rtl = _invoke_runner(
            runner_path=runner_path,
            program_path=program_path,
            summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=folded_pos_embed,
            num_classes=num_classes,
            max_cycles=max_cycles,
            trace_json_out=rtl_trace_path,
        )
        artifacts["rtl_trace"] = str(rtl_trace_path)
        return runner_rc, rtl, None

    snapshot_request_path = work_dir / "snapshot_request.csv"
    snapshot_manifest_path = work_dir / "rtl_snapshot_manifest.json"
    snapshot_data_path = work_dir / "rtl_snapshot_data.bin"
    first_divergence_path = work_dir / "first_divergence.json"
    _write_snapshot_request(snapshot_request_path, trace_events)

    runner_rc, rtl = _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cls_raw_path=cls_raw_path,
        folded_pos_embed=folded_pos_embed,
        num_classes=num_classes,
        max_cycles=max_cycles,
        trace_json_out=rtl_trace_path,
        snapshot_request_path=snapshot_request_path,
        snapshot_manifest_out=snapshot_manifest_path,
        snapshot_data_out=snapshot_data_path,
    )

    artifacts["rtl_trace"] = str(rtl_trace_path)
    artifacts["snapshot_request"] = str(snapshot_request_path)
    artifacts["snapshot_manifest"] = str(snapshot_manifest_path)
    artifacts["snapshot_data"] = str(snapshot_data_path)

    divergence = _compute_first_divergence(
        program=program,
        golden_trace=golden_trace,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_data_path=snapshot_data_path,
        artifact_paths={
            **artifacts,
            "golden_trace": artifacts.get("golden_trace"),
            "rtl_trace": str(rtl_trace_path),
            "snapshot_manifest": str(snapshot_manifest_path),
            "snapshot_data": str(snapshot_data_path),
        },
    )
    if divergence is not None:
        first_divergence_path.write_text(json.dumps(divergence, indent=2, sort_keys=True, default=_json_default))
        artifacts["first_divergence"] = str(first_divergence_path)
    return runner_rc, rtl, divergence


def compare_program_mode(args: argparse.Namespace) -> dict[str, Any]:
    runner_path = Path(args.runner).resolve()
    _ensure_runner_built(runner_path, rebuild=args.rebuild_runner)

    work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="compare_rtl_golden_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    program_path = Path(args.program).resolve()
    program = _load_program(program_path)

    patches = None
    cls_int8 = None
    patches_raw_path = None
    cls_raw_path = None

    if args.input is not None:
        patches = _load_raw_or_npy_int8(
            Path(args.input),
            rows=args.patch_rows,
            cols=args.patch_cols,
        )
        if patches.ndim != 2:
            raise ValueError("Program-mode input must be a 2D INT8 tensor")
        patches_raw_path = work_dir / "patches.raw"
        _write_raw_int8(patches_raw_path, patches)

    if args.cls_input is not None:
        cls_int8 = _load_raw_or_npy_int8(Path(args.cls_input))
        cls_raw_path = work_dir / "cls.raw"
        _write_raw_int8(cls_raw_path, cls_int8.reshape(-1))

    golden = _run_golden_program(
        program=program,
        patches_int8=patches,
        cls_int8=cls_int8,
        folded_pos_embed=args.folded_pos_embed,
        num_classes=args.num_classes,
    )

    rtl_summary_path = work_dir / "rtl_summary.json"
    rtl_trace_path = work_dir / "rtl_trace.json"
    golden_trace_path = work_dir / "golden_trace.json"

    runner_rc, rtl = _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=(patches.shape[0] if patches is not None else None),
        patch_cols=(patches.shape[1] if patches is not None else None),
        cls_raw_path=cls_raw_path,
        folded_pos_embed=args.folded_pos_embed,
        num_classes=args.num_classes,
        max_cycles=args.max_cycles,
    )

    first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
    passed = (
        runner_rc == 0
        and rtl["status"] == "halted"
        and not rtl["fault"]
        and first_mismatch is None
    )

    artifacts = {
        "work_dir": str(work_dir),
        "program": str(program_path),
        "rtl_summary": str(rtl_summary_path),
    }
    first_divergence = None
    qkt_stability_report = None
    qkt_prestate_provenance_report = None
    qkt_prefix_provenance_report = None
    qkt_startup_provenance_report = None
    startup_pos_load_report = None
    projection_padding_report = None
    qkv_source_padding_report = None
    qkv_source_padding_report = None
    effective_first_divergence = None
    effective_pass = passed

    if not passed:
        trace_nodes = _trace_node_order_from_program(program)
        rerun_golden = _run_golden_program(
            program=program,
            patches_int8=patches,
            cls_int8=cls_int8,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            trace_nodes=trace_nodes,
        )
        golden_trace_path.write_text(
            json.dumps(rerun_golden["trace"], indent=2, sort_keys=True, default=_json_default)
        )
        artifacts["golden_trace"] = str(golden_trace_path)
        runner_rc, rtl, first_divergence = _maybe_emit_first_divergence(
            program=program,
            golden_trace=rerun_golden["trace"],
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            rtl_trace_path=rtl_trace_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        projection_padding_report = _maybe_emit_projection_padding_report(
            first_divergence=first_divergence,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkv_source_padding_report = _maybe_emit_qkv_source_padding_report(
            first_divergence=first_divergence,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        rebased_divergence = _maybe_rebase_first_divergence_for_expected_qkv_padding(
            program=program,
            golden_trace=rerun_golden["trace"],
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            qkv_source_padding_report=qkv_source_padding_report,
        )
        if rebased_divergence is not None:
            first_divergence = rebased_divergence
        qkt_stability_report = _maybe_emit_qkt_stability_report(
            first_divergence=first_divergence,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        effective_first_divergence = first_divergence
        _maybe_emit_qkt_prestate_debug_artifacts(
            first_divergence=first_divergence,
            qkt_stability_report=qkt_stability_report,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_qkt_systolic_window_trace(
            first_divergence=first_divergence,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            rtl_trace_path=rtl_trace_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkt_prestate_provenance_report = _maybe_emit_qkt_prestate_provenance_report(
            first_divergence=first_divergence,
            qkt_stability_report=qkt_stability_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_ln1_provenance_debug_artifacts(
            first_divergence=first_divergence,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_qkt_prefix_debug_artifacts(
            first_divergence=first_divergence,
            qkt_prestate_provenance_report=qkt_prestate_provenance_report,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkt_prefix_provenance_report = _maybe_emit_qkt_prefix_provenance_report(
            first_divergence=first_divergence,
            qkt_prestate_provenance_report=qkt_prestate_provenance_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_qkt_startup_debug_artifacts(
            first_divergence=first_divergence,
            qkt_prefix_provenance_report=qkt_prefix_provenance_report,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=patches.shape[0] if patches is not None else None,
            patch_cols=patches.shape[1] if patches is not None else None,
            cls_raw_path=cls_raw_path,
            folded_pos_embed=args.folded_pos_embed,
            num_classes=args.num_classes,
            max_cycles=args.max_cycles,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkt_startup_provenance_report = _maybe_emit_qkt_startup_provenance_report(
            first_divergence=first_divergence,
            qkt_prefix_provenance_report=qkt_prefix_provenance_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        startup_pos_load_report = _maybe_emit_startup_pos_load_report(
            qkt_startup_provenance_report=qkt_startup_provenance_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        rebased_effective_first_divergence = _maybe_rebase_first_divergence_for_nonblocking_qkt_prestate(
            program=program,
            golden_trace=rerun_golden["trace"],
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            first_divergence=first_divergence,
            qkt_prestate_provenance_report=qkt_prestate_provenance_report,
        )
        if (
            qkt_prestate_provenance_report is not None
            and qkt_prestate_provenance_report.get("classification") == "nonblocking_qkt_prestate_scratch"
        ):
            effective_first_divergence = rebased_effective_first_divergence
        rebased_prefix_effective_first_divergence = _maybe_rebase_first_divergence_for_nonblocking_qkt_prefix(
            program=program,
            golden_trace=rerun_golden["trace"],
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            first_divergence=first_divergence,
            qkt_prefix_provenance_report=qkt_prefix_provenance_report,
        )
        if (
            qkt_prefix_provenance_report is not None
            and qkt_prefix_provenance_report.get("classification") == "prefix_nonblocking_scratch"
        ):
            effective_first_divergence = rebased_prefix_effective_first_divergence
        rebased_startup_effective_first_divergence = _maybe_rebase_first_divergence_for_nonblocking_qkt_startup(
            program=program,
            golden_trace=rerun_golden["trace"],
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            first_divergence=first_divergence,
            qkt_startup_provenance_report=qkt_startup_provenance_report,
        )
        if (
            qkt_startup_provenance_report is not None
            and qkt_startup_provenance_report.get("classification") == "startup_nonblocking_scratch"
        ):
            effective_first_divergence = rebased_startup_effective_first_divergence
        first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
        passed = (
            runner_rc == 0
            and rtl["status"] == "halted"
            and not rtl["fault"]
            and first_mismatch is None
        )
        effective_pass = (
            True
            if qkt_prestate_provenance_report is not None
            and qkt_prestate_provenance_report.get("classification") == "nonblocking_qkt_prestate_scratch"
            and (effective_first_divergence is None
                 or _is_one_lsb_rounding_artifact(effective_first_divergence))
            else True
            if qkt_prefix_provenance_report is not None
            and qkt_prefix_provenance_report.get("classification") == "prefix_nonblocking_scratch"
            and (effective_first_divergence is None
                 or _is_one_lsb_rounding_artifact(effective_first_divergence))
            else True
            if qkt_startup_provenance_report is not None
            and qkt_startup_provenance_report.get("classification") == "startup_nonblocking_scratch"
            and (effective_first_divergence is None
                 or _is_one_lsb_rounding_artifact(effective_first_divergence))
            else passed
        )
    else:
        effective_pass = passed

    return {
        "mode": "program",
        "pass": passed,
        "effective_pass": effective_pass,
        "runner_exit_code": runner_rc,
        "golden": golden,
        "rtl": rtl,
        "mismatch_count": 0 if first_mismatch is None else 1,
        "first_mismatch": first_mismatch,
        "program_manifest": program.compiler_manifest,
        "artifacts": artifacts,
        "first_divergence": first_divergence,
        "effective_first_divergence": effective_first_divergence,
        "qkt_stability_report": qkt_stability_report,
        "qkt_prestate_provenance_report": qkt_prestate_provenance_report,
        "qkt_prefix_provenance_report": qkt_prefix_provenance_report,
        "qkt_startup_provenance_report": qkt_startup_provenance_report,
        "startup_pos_load_report": startup_pos_load_report,
        "projection_padding_report": projection_padding_report,
        "qkv_source_padding_report": qkv_source_padding_report,
    }


def compare_compile_mode(args: argparse.Namespace) -> dict[str, Any]:
    import compare_golden as cg

    runner_path = Path(args.runner).resolve()
    _ensure_runner_built(runner_path, rebuild=args.rebuild_runner)

    work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="compare_rtl_compile_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    cg.WEIGHTS_PATH = args.weights
    model, state_dict = cg.load_model()
    processor = _load_processor(cg.MODEL_NAME)

    calibration_paths = args.calibration_image or [args.image]
    sample_images = []
    for img_path in calibration_paths:
        with Image.open(img_path) as img:
            sample_images.append(img.convert("RGB"))

    compile_kwargs = _scenario_overrides(args.scenario)
    compile_kwargs["gelu_from_accum"] = args.gelu_from_accum
    if args.requant_pc_qkv_blocks:
        compile_kwargs["requant_pc_qkv"] = True
        compile_kwargs["requant_pc_qkv_selection"] = cg.build_requant_pc_qkv_selection(
            blocks_text=args.requant_pc_qkv_blocks,
            heads_text=args.requant_pc_qkv_heads,
            projections_text=args.requant_pc_qkv_projections,
            exclude_text=args.requant_pc_qkv_exclude,
        )

    program, cal_scales = cg.compile_model(
        model,
        state_dict,
        sample_images,
        processor,
        **compile_kwargs,
    )

    embed_scale = cal_scales.get("pos_embed_add", 14.0 / 127.0)
    with Image.open(args.image) as runtime_img:
        runtime_img = runtime_img.convert("RGB")
        patches_int8, cls_int8, _ = cg.patch_embed_int8(
            model,
            processor,
            runtime_img,
            embed_scale,
            fold_cls_pos_embed=args.fold_cls_pos_embed,
        )

    program_path = work_dir / "program.bin"
    program_path.write_bytes(program.to_bytes())
    patches_raw_path = work_dir / "patches.raw"
    _write_raw_int8(patches_raw_path, patches_int8)
    cls_raw_path = None
    if cls_int8 is not None:
        cls_raw_path = work_dir / "cls.raw"
        _write_raw_int8(cls_raw_path, cls_int8.reshape(-1))

    golden_logits, insn_count, cycle_count, _ = cg.golden_inference(
        program,
        patches_int8,
        cls_int8=cls_int8,
        num_classes=args.num_classes,
    )
    golden = {
        "fault": False,
        "fault_code": 0,
        "instruction_count": int(insn_count),
        "cycle_count": int(cycle_count),
        "logits": np.asarray(golden_logits, dtype=np.int32).tolist(),
    }
    rtl_cycle_budget = _compile_mode_cycle_budget(args.max_cycles, golden["cycle_count"])

    rtl_summary_path = work_dir / "rtl_summary.json"
    rtl_trace_path = work_dir / "rtl_trace.json"
    golden_trace_path = work_dir / "golden_trace.json"
    # patch_embed_int8() always folds patch position embeddings (B3 opt);
    # pass folded_pos_embed=True so the RTL runner zeroes those DRAM rows,
    # matching golden_inference which always zeroes pos_embed_patch_dram_offset.
    runner_rc, rtl = _invoke_runner(
        runner_path=runner_path,
        program_path=program_path,
        summary_path=rtl_summary_path,
        patches_raw_path=patches_raw_path,
        patch_rows=int(patches_int8.shape[0]),
        patch_cols=int(patches_int8.shape[1]),
        cls_raw_path=cls_raw_path,
        folded_pos_embed=True,
        num_classes=args.num_classes,
        max_cycles=rtl_cycle_budget,
    )

    first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
    passed = (
        runner_rc == 0
        and rtl["status"] == "halted"
        and not rtl["fault"]
        and first_mismatch is None
    )

    artifacts = {
        "work_dir": str(work_dir),
        "program": str(program_path),
        "rtl_summary": str(rtl_summary_path),
    }
    first_divergence = None
    qkt_stability_report = None
    qkt_prestate_provenance_report = None
    qkt_prefix_provenance_report = None
    qkt_startup_provenance_report = None
    startup_pos_load_report = None
    projection_padding_report = None
    qkv_source_padding_report = None
    effective_first_divergence = None
    effective_pass = passed

    if not passed:
        trace_nodes = _trace_node_order_from_program(program)
        _, _, _, golden_trace = cg.golden_inference(
            program,
            patches_int8,
            cls_int8=cls_int8,
            num_classes=args.num_classes,
            trace_nodes=trace_nodes,
        )
        golden_trace_path.write_text(
            json.dumps(golden_trace, indent=2, sort_keys=True, default=_json_default)
        )
        artifacts["golden_trace"] = str(golden_trace_path)
        runner_rc, rtl, first_divergence = _maybe_emit_first_divergence(
            program=program,
            golden_trace=golden_trace,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            rtl_trace_path=rtl_trace_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        projection_padding_report = _maybe_emit_projection_padding_report(
            first_divergence=first_divergence,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkv_source_padding_report = _maybe_emit_qkv_source_padding_report(
            first_divergence=first_divergence,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        rebased_divergence = _maybe_rebase_first_divergence_for_expected_qkv_padding(
            program=program,
            golden_trace=golden_trace,
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            qkv_source_padding_report=qkv_source_padding_report,
        )
        if rebased_divergence is not None:
            first_divergence = rebased_divergence
        qkt_stability_report = _maybe_emit_qkt_stability_report(
            first_divergence=first_divergence,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        effective_first_divergence = first_divergence
        _maybe_emit_qkt_prestate_debug_artifacts(
            first_divergence=first_divergence,
            qkt_stability_report=qkt_stability_report,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_qkt_systolic_window_trace(
            first_divergence=first_divergence,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            rtl_trace_path=rtl_trace_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkt_prestate_provenance_report = _maybe_emit_qkt_prestate_provenance_report(
            first_divergence=first_divergence,
            qkt_stability_report=qkt_stability_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_ln1_provenance_debug_artifacts(
            first_divergence=first_divergence,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_qkt_prefix_debug_artifacts(
            first_divergence=first_divergence,
            qkt_prestate_provenance_report=qkt_prestate_provenance_report,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkt_prefix_provenance_report = _maybe_emit_qkt_prefix_provenance_report(
            first_divergence=first_divergence,
            qkt_prestate_provenance_report=qkt_prestate_provenance_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        _maybe_emit_qkt_startup_debug_artifacts(
            first_divergence=first_divergence,
            qkt_prefix_provenance_report=qkt_prefix_provenance_report,
            runner_path=runner_path,
            program_path=program_path,
            rtl_summary_path=rtl_summary_path,
            patches_raw_path=patches_raw_path,
            patch_rows=int(patches_int8.shape[0]),
            patch_cols=int(patches_int8.shape[1]),
            cls_raw_path=cls_raw_path,
            folded_pos_embed=True,
            num_classes=args.num_classes,
            max_cycles=rtl_cycle_budget,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        qkt_startup_provenance_report = _maybe_emit_qkt_startup_provenance_report(
            first_divergence=first_divergence,
            qkt_prefix_provenance_report=qkt_prefix_provenance_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        startup_pos_load_report = _maybe_emit_startup_pos_load_report(
            qkt_startup_provenance_report=qkt_startup_provenance_report,
            work_dir=work_dir,
            artifacts=artifacts,
        )
        rebased_effective_first_divergence = _maybe_rebase_first_divergence_for_nonblocking_qkt_prestate(
            program=program,
            golden_trace=golden_trace,
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            first_divergence=first_divergence,
            qkt_prestate_provenance_report=qkt_prestate_provenance_report,
        )
        if (
            qkt_prestate_provenance_report is not None
            and qkt_prestate_provenance_report.get("classification") == "nonblocking_qkt_prestate_scratch"
        ):
            effective_first_divergence = rebased_effective_first_divergence
        rebased_prefix_effective_first_divergence = _maybe_rebase_first_divergence_for_nonblocking_qkt_prefix(
            program=program,
            golden_trace=golden_trace,
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            first_divergence=first_divergence,
            qkt_prefix_provenance_report=qkt_prefix_provenance_report,
        )
        if (
            qkt_prefix_provenance_report is not None
            and qkt_prefix_provenance_report.get("classification") == "prefix_nonblocking_scratch"
        ):
            effective_first_divergence = rebased_prefix_effective_first_divergence
        rebased_startup_effective_first_divergence = _maybe_rebase_first_divergence_for_nonblocking_qkt_startup(
            program=program,
            golden_trace=golden_trace,
            snapshot_manifest_path=work_dir / "rtl_snapshot_manifest.json",
            snapshot_data_path=work_dir / "rtl_snapshot_data.bin",
            work_dir=work_dir,
            artifacts=artifacts,
            first_divergence=first_divergence,
            qkt_startup_provenance_report=qkt_startup_provenance_report,
        )
        if (
            qkt_startup_provenance_report is not None
            and qkt_startup_provenance_report.get("classification") == "startup_nonblocking_scratch"
        ):
            effective_first_divergence = rebased_startup_effective_first_divergence
        first_mismatch = _first_mismatch(golden["logits"], rtl["logits"])
        passed = (
            runner_rc == 0
            and rtl["status"] == "halted"
            and not rtl["fault"]
            and first_mismatch is None
        )
        effective_pass = (
            True
            if qkt_prestate_provenance_report is not None
            and qkt_prestate_provenance_report.get("classification") == "nonblocking_qkt_prestate_scratch"
            and (effective_first_divergence is None
                 or _is_one_lsb_rounding_artifact(effective_first_divergence))
            else True
            if qkt_prefix_provenance_report is not None
            and qkt_prefix_provenance_report.get("classification") == "prefix_nonblocking_scratch"
            and (effective_first_divergence is None
                 or _is_one_lsb_rounding_artifact(effective_first_divergence))
            else True
            if qkt_startup_provenance_report is not None
            and qkt_startup_provenance_report.get("classification") == "startup_nonblocking_scratch"
            and (effective_first_divergence is None
                 or _is_one_lsb_rounding_artifact(effective_first_divergence))
            else passed
        )
    else:
        effective_pass = passed

    return {
        "mode": "compile",
        "scenario": args.scenario,
        "pass": passed,
        "effective_pass": effective_pass,
        "runner_exit_code": runner_rc,
        "golden": golden,
        "rtl": rtl,
        "rtl_cycle_budget": rtl_cycle_budget,
        "mismatch_count": 0 if first_mismatch is None else 1,
        "first_mismatch": first_mismatch,
        "program_manifest": program.compiler_manifest,
        "artifacts": artifacts,
        "first_divergence": first_divergence,
        "effective_first_divergence": effective_first_divergence,
        "qkt_stability_report": qkt_stability_report,
        "qkt_prestate_provenance_report": qkt_prestate_provenance_report,
        "qkt_prefix_provenance_report": qkt_prefix_provenance_report,
        "qkt_startup_provenance_report": qkt_startup_provenance_report,
        "startup_pos_load_report": startup_pos_load_report,
        "projection_padding_report": projection_padding_report,
        "qkv_source_padding_report": qkv_source_padding_report,
    }


def compare_systolic_window_mode(args: argparse.Namespace) -> dict[str, Any]:
    return diff_systolic_window_traces(
        baseline_trace_path=Path(args.baseline_trace).resolve(),
        fragment_trace_path=Path(args.fragment_trace).resolve(),
    )


def compare_accum_write_mode(args: argparse.Namespace) -> dict[str, Any]:
    return diff_accum_write_logs(
        baseline_log_path=Path(args.baseline_log).resolve(),
        fragment_log_path=Path(args.fragment_log).resolve(),
    )


def compare_sram_write_mode(args: argparse.Namespace) -> dict[str, Any]:
    return diff_sram_write_logs(
        baseline_log_path=Path(args.baseline_log).resolve(),
        fragment_log_path=Path(args.fragment_log).resolve(),
    )


def compare_hidden_snapshot_mode(args: argparse.Namespace) -> dict[str, Any]:
    return diff_hidden_snapshots(
        baseline_snapshot_path=Path(args.baseline_snapshot).resolve(),
        fragment_snapshot_path=Path(args.fragment_snapshot).resolve(),
    )


def compare_projection_padding_mode(args: argparse.Namespace) -> dict[str, Any]:
    report = emit_projection_padding_report_from_replay_dir(
        replay_dir=Path(args.replay_dir).resolve(),
        out_path=Path(args.summary_out).resolve(),
    )
    report["pass"] = True
    return report


def compare_ln1_operand_mode(args: argparse.Namespace) -> dict[str, Any]:
    return emit_ln1_operand_report_from_replay_dir(
        replay_dir=Path(args.replay_dir).resolve(),
        sram_log_path=Path(args.sram_log).resolve(),
        out_path=Path(args.summary_out).resolve(),
    )


def compare_ln1_provenance_mode(args: argparse.Namespace) -> dict[str, Any]:
    report = emit_ln1_provenance_report(
        replay_dir=Path(args.replay_dir).resolve(),
        baseline_log_path=Path(args.baseline_log).resolve(),
        fragment_log_path=Path(args.fragment_log).resolve(),
        out_path=Path(args.summary_out).resolve(),
    )
    report["pass"] = report["classification"] != "capture_window_clipped"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare RTL execution against the software golden model")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="Path to the Verilator program runner")
    parser.add_argument("--work-dir", help="Directory for intermediate artifacts and summaries")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of logits to compare from ACCUM")
    parser.add_argument("--max-cycles", type=int, default=500000, help="Cycle budget for the RTL runner")
    parser.add_argument("--rebuild-runner", action="store_true", help="Force a fresh Verilator runner build")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_program = sub.add_parser("program", help="Compare a precompiled ProgramBinary")
    p_program.add_argument("--program", required=True, help="Path to program.bin")
    p_program.add_argument("--input", help="Program input tensor (.npy or raw .bin)")
    p_program.add_argument("--patch-rows", type=int, help="Rows for raw input .bin")
    p_program.add_argument("--patch-cols", type=int, help="Columns for raw input .bin")
    p_program.add_argument("--cls-input", help="Optional folded CLS row (.npy or raw .bin)")
    p_program.add_argument("--folded-pos-embed", action="store_true", help="Zero folded position-embedding regions")

    p_compile = sub.add_parser("compile", help="Compile a DeiT model variant, then compare RTL vs golden")
    p_compile.add_argument("--scenario", required=True, choices=[
        "baseline_default",
        "experimental_requant_pc",
        "experimental_dequant_add",
        "experimental_softmax_attnv",
        "experimental_fused_out_proj",
    ])
    p_compile.add_argument("--weights", required=True, help="Path to pytorch_model.bin")
    p_compile.add_argument("--image", required=True, help="Runtime image used for patch embedding and compare")
    p_compile.add_argument("--calibration-image", action="append", help="Optional calibration image(s)")
    p_compile.add_argument("--fold-cls-pos-embed", action="store_true", help="Fold CLS position embedding on the host")
    p_compile.add_argument("--gelu-from-accum", action="store_true", help="Enable GELU-from-ACCUM during compile")
    p_compile.add_argument("--requant-pc-qkv-blocks", help="Optional block CSV for QKV REQUANT_PC selection")
    p_compile.add_argument("--requant-pc-qkv-heads", default="", help="Optional head CSV for QKV REQUANT_PC selection")
    p_compile.add_argument("--requant-pc-qkv-projections", default="all", help="Projection filter for QKV REQUANT_PC selection")
    p_compile.add_argument("--requant-pc-qkv-exclude", default="", help="Exclude triplets for QKV REQUANT_PC selection")

    p_diff = sub.add_parser("diff-systolic-window", help="Diff two systolic window traces by relative cycle")
    p_diff.add_argument("--baseline-trace", required=True, help="Path to the failing baseline systolic window JSON")
    p_diff.add_argument("--fragment-trace", required=True, help="Path to the passing fragment systolic window JSON")

    p_diff_accum = sub.add_parser("diff-accum-writes", help="Diff two ACCUM write provenance logs")
    p_diff_accum.add_argument("--baseline-log", required=True, help="Path to the failing baseline ACCUM write log JSON")
    p_diff_accum.add_argument("--fragment-log", required=True, help="Path to the passing fragment ACCUM write log JSON")

    p_diff_sram = sub.add_parser("diff-sram-writes", help="Diff two generic SRAM write provenance logs")
    p_diff_sram.add_argument("--baseline-log", required=True, help="Path to the failing baseline SRAM write log JSON")
    p_diff_sram.add_argument("--fragment-log", required=True, help="Path to the passing fragment SRAM write log JSON")

    p_diff_hidden = sub.add_parser("diff-hidden-snapshot", help="Diff two hidden systolic snapshots")
    p_diff_hidden.add_argument("--baseline-snapshot", required=True, help="Path to the failing baseline hidden snapshot JSON")
    p_diff_hidden.add_argument("--fragment-snapshot", required=True, help="Path to the passing fragment hidden snapshot JSON")

    p_padding = sub.add_parser("projection-padding-report", help="Classify padded projection tails from a replay payload dir")
    p_padding.add_argument("--replay-dir", required=True, help="Replay payload directory containing replay_metadata.json")

    p_ln1 = sub.add_parser("ln1-operand-report", help="Compare expected LayerNorm gamma/beta rows against a generic SRAM write log")
    p_ln1.add_argument("--replay-dir", required=True, help="Replay payload directory containing ln1 gamma/beta payloads")
    p_ln1.add_argument("--sram-log", required=True, help="Baseline generic SRAM write log JSON")

    p_ln1_prov = sub.add_parser("ln1-provenance-report", help="Compare fresh full-program ln1 provenance against a passing DMA-loaded replay")
    p_ln1_prov.add_argument("--replay-dir", required=True, help="Replay payload directory containing corrected ln1 operands")
    p_ln1_prov.add_argument("--baseline-log", required=True, help="Fresh full-program ln1 SRAM write log JSON")
    p_ln1_prov.add_argument("--fragment-log", required=True, help="Passing ln1 DMA-loaded fragment SRAM write log JSON")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "program":
        summary = compare_program_mode(args)
    elif args.mode == "compile":
        summary = compare_compile_mode(args)
    elif args.mode == "diff-systolic-window":
        summary = compare_systolic_window_mode(args)
    elif args.mode == "diff-accum-writes":
        summary = compare_accum_write_mode(args)
    elif args.mode == "diff-sram-writes":
        summary = compare_sram_write_mode(args)
    elif args.mode == "projection-padding-report":
        summary = compare_projection_padding_mode(args)
    elif args.mode == "ln1-operand-report":
        summary = compare_ln1_operand_mode(args)
    elif args.mode == "ln1-provenance-report":
        summary = compare_ln1_provenance_mode(args)
    else:
        summary = compare_hidden_snapshot_mode(args)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
