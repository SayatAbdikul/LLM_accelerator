"""Replay three retained teacher-forced W8A16 quality inputs on mode-1 RTL.

Requires a passed canonical 257-token software quality export with explicit
HALT evidence. The three hardware targets use the original 256-key decode
window and matched FP32 prefixes; this is not a 256-target RTL quality claim.
The selected acceptance contract is 5% FP32 PPL increase, 2% reference PPL
difference, cosine >=0.995 and top-five overlap >=4. Exact golden logits/KV
are reported separately as diagnostics, not silently substituted for this
contract. Hardware carries its own KV, never the golden comparison images.

Use --quality SOFTWARE_OUTPUT --runner FRESH_MODE1_BINARY --out NEW_DIRECTORY.
An external supervisor owns RSS, free-disk and campaign limits. This CLI keeps
the reviewed 1200-second native-step timeout and does not build or download.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
THRESHOLDS = {"max_reference_ppl_relative_delta": 0.02, "min_reference_cosine": 0.995,
              "min_reference_top5_overlap": 4, "max_fp32_ppl_increase": 0.05}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def checked_file(directory, record):
    path = directory / record["file"]
    require(path.parent.resolve() == directory.resolve() and not path.is_symlink(), "Unsafe export path")
    require(path.is_file() and path.stat().st_size == record["bytes"]
            and sha(path) == record["sha256"], "Export hash/size mismatch")
    return path


def store_ranges(instructions, base, end):
    """Exact linear STORE ranges; a single final HALT closes the stream."""
    from taccel.isa.encoding import decode
    from taccel.isa.instructions import HaltInsn, SetAddrLoInsn, SetAddrHiInsn, StoreInsn
    require(bool(instructions) and len(instructions) % 8 == 0, "Unaligned/empty instruction stream")
    addresses, intervals, halted = [0] * 4, [], False
    for offset in range(0, len(instructions), 8):
        require(not halted, "Instructions follow HALT")
        insn = decode(instructions[offset:offset + 8])
        if isinstance(insn, SetAddrLoInsn):
            addresses[insn.addr_reg] = (addresses[insn.addr_reg] & ~0xfffffff) | insn.imm28
        elif isinstance(insn, SetAddrHiInsn):
            addresses[insn.addr_reg] = (addresses[insn.addr_reg] & 0xfffffff) | (insn.imm28 << 28)
        elif isinstance(insn, StoreInsn):
            start = addresses[insn.addr_reg] + insn.dram_off * 16
            stop = start + insn.xfer_len * 16
            require(base <= start <= stop <= end, "STORE escapes retained data region")
            if start < stop:
                intervals.append((start - base, stop - base))
        elif isinstance(insn, HaltInsn):
            halted = True
    require(halted, "Instruction stream has no terminal HALT")
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def immutable_check(before, after, mutable):
    require(len(before) == len(after), "Full data dump size mismatch")
    cursor, checked = 0, 0
    for start, stop in mutable + [(len(before), len(before))]:
        require(cursor <= start <= stop <= len(before), "Invalid mutable STORE intervals")
        for offset in range(cursor, start, 64 * 1024):
            limit = min(start, offset + 64 * 1024)
            require(before[offset:limit] == after[offset:limit], "Data outside emitted STORE ranges changed")
            checked += limit - offset
        cursor = stop
    return checked


def validate_parent(report, manifest):
    require(report["passed"] is True and report["status"] == "passed" and report["exit_code"] == 0
            and report["identities_unchanged"] is True, "Parent quality gate is not terminal/passed")
    require(report["quality"]["passed"] is True and report["quality"]["thresholds"] == THRESHOLDS
            and report["quality"]["target_count"] == 256 and report["quality"]["token_count"] == 257
            and report["scope"]["canonical_257_token_window"] is True, "Wrong canonical quality contract")
    require(report["logits_artifact"]["row_counts"] == dict(compiled=256, reference=256, fp32=256),
            "Parent rail coverage is incomplete")
    tokens, vocab = report["input_ids"]["evaluation"], report["model_args"]["vocab_size"]
    require(type(vocab) is int and vocab >= 5 and len(tokens) == 257
            and all(type(t) is int and 0 <= t < vocab for t in tokens), "Invalid canonical token/vocabulary coverage")
    require(type(report["quality"]["vocab_size"]) is int and report["quality"]["vocab_size"] == vocab,
            "Parent quality/model vocabulary mismatch")
    require(report["input_ids"]["inputs"] == tokens[:-1] and report["input_ids"]["targets"] == tokens[1:],
            "Parent teacher-forced input/target mapping mismatch")
    executions = report["compiled_step_execution"]
    require(len(executions) == 256, "Parent HALT coverage is incomplete")
    for position, execution in enumerate(executions):
        require(type(execution["position"]) is int and execution["position"] == position
                and type(execution["token_id"]) is int and execution["token_id"] == tokens[position]
                and execution["halted"] is True
                and execution["stream"] == ("prefill" if position == 0 else "decode"),
                "Parent compiled step lacks ordered HALT evidence")
    require(manifest == report["rtl_export"], "Whole replay export is not bound to its terminal parent")
    require(manifest["status"] == "golden_complete_rtl_not_run" and manifest["steps_requested"] == 3
            and manifest["attention_window"] == 256 and len(manifest["steps"]) == 3
            and len(manifest["golden_steps"]) == 3, "Incomplete replay export")
    require(manifest["sources"] == report["sources"] and manifest["scales_sha256"] == report["scales_sha256"]
            and manifest["transforms"]["quarot"]["applied"] is True
            and report["transforms"]["fp32_captured_before_ptq"] is True,
            "Export/model policy does not match parent")
    require(manifest["vocab_size"] == vocab and manifest["token_ids"] == tokens[:3], "Exported input/vocabulary mismatch")
    return tokens, vocab


def _verify_parent_sources(report):
    from tools.w8a16_quality_gate import _source_identity
    require(_source_identity()["sha256"] == report["sources"]["sha256"], "Quality Python sources changed")


def _replay_source_pins():
    paths = list((ROOT / "rtl/common/src").rglob("*.sv")) + list((ROOT / "rtl/common/src").rglob("*.svh"))
    paths += list((ROOT / "rtl/verilator/include").glob("*.h"))
    paths += [ROOT / "rtl/verilator/Makefile", ROOT / "rtl/verilator/run_program.cpp",
              ROOT / "rtl/common/filelists/core.f", ROOT / "software/tools/benchmark_support.py",
              ROOT / "software/tools/check_reference_manifest.py", Path(__file__)]
    return {str(path): sha(path) for path in paths}


def _verify_pins(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, f"Replay identity changed: {path}")


def _run_native(command, log_path):
    with log_path.open("x") as log:
        return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)


def _replay(args, outcome, pins):
    import numpy as np
    from taccel.assembler.assembler import ProgramBinary
    from taccel.runtime.numerical_quality import teacher_forced_quality_report
    from tools.benchmark_support import check_runner_freshness, validate_runner_summary
    from tools.check_reference_manifest import verify_reference_manifest

    parent = args.quality.resolve()
    export = parent / "rtl_export"
    report_path, manifest_path = parent / "report.json", export / "manifest.json"
    for path in (report_path, manifest_path):
        pins[str(path)] = sha(path)
    report, manifest = read_json(report_path), read_json(manifest_path)
    tokens, vocab = validate_parent(report, manifest)
    outcome.update(parent_report_sha256=pins[str(report_path)], export_manifest_sha256=pins[str(manifest_path)])
    _verify_parent_sources(report)
    pins.update({str(ROOT / path): value for path, value in report["sources"]["sha256"].items()})
    verify_reference_manifest()
    runner = args.runner.resolve()
    check_runner_freshness(ROOT, runner)
    pins[str(runner)] = sha(runner)
    source_pins = _replay_source_pins()
    pins.update(source_pins)
    outcome.update(runner_sha256=pins[str(runner)], source_sha256=source_pins)
    static_path = checked_file(export, manifest["static_data"])
    pins[str(static_path)] = manifest["static_data"]["sha256"]
    static = static_path.read_bytes()
    base, end = manifest["data_base"], manifest["required_dram_bytes"]
    kv_offset, kv_size, shared_bytes = (manifest[k] for k in
                                      ("kv_offset_in_static_data", "kv_cache_size", "shared_data_bytes"))
    width = manifest["logits_row_bytes"]
    require(all(type(v) is int for v in (base, end, kv_offset, kv_size, shared_bytes, width))
            and base > 0 and base % 16 == 0 and 0 <= shared_bytes <= kv_offset and kv_size > 0
            and manifest["kv_cache_base"] == base + kv_offset and len(static) == end - base
            and kv_offset + kv_size == len(static) and not any(static[kv_offset:])
            and width % 2 == 0 and width >= vocab * 2, "Invalid static data, logits layout or nonzero initial KV")
    require(hashlib.sha256(static[:shared_bytes]).hexdigest() == manifest["shared_data_sha256"]
            == report["compiled"]["program_sha256"]["shared_data"], "Static immutable shared-data hash mismatch")
    npz_path = parent / "logits.npz"
    require(sha(npz_path) == report["logits_artifact"]["sha256"], "Parent logits hash changed")
    pins[str(npz_path)] = report["logits_artifact"]["sha256"]
    prepared = []
    for position in range(3):
        item, golden = manifest["steps"][position], manifest["golden_steps"][position]
        require(item["position"] == golden["position"] == position
                and item["token_id"] == golden["token_id"] == tokens[position]
                and item["stream"] == ("prefill" if position == 0 else "decode")
                and item["query_row_base"] == position and item["valid_kv_len"] == position + 1,
                "Step order, token or causal context mismatch")
        for record in (item["program"], golden["logits"], golden["kv"]):
            path = checked_file(export, record)
            pins[str(path)] = record["sha256"]
        envelope_path = export / item["program"]["file"]
        envelope = ProgramBinary.from_bytes(envelope_path.read_bytes())
        require(not envelope.data and envelope.entry_point == 0 and envelope.data_base == base
                and envelope.insn_count == item["insn_count"] == len(envelope.instructions) // 8
                and len(envelope.instructions) == item["instructions_bytes"] <= base
                and hashlib.sha256(envelope.instructions).hexdigest() == item["instructions_sha256"],
                "Invalid local-PC program envelope")
        mutable = store_ranges(envelope.instructions, base, end)
        require(all(start >= shared_bytes for start, _ in mutable), "STORE overlaps immutable shared model data")
        offset = item["logits_offset"] - base
        require(shared_bytes <= offset and offset + width <= kv_offset, "Logits overlap immutable data or KV")
        prepared.append((item, golden, envelope, mutable, offset))
    with np.load(npz_path, allow_pickle=False) as saved:
        expected = {"token_ids"} | {f"{name}_{i:04d}" for name in ("compiled", "reference", "fp32") for i in range(256)}
        require(set(saved.files) == expected and saved["token_ids"].tolist() == tokens,
                "Parent NPZ token/rail coverage mismatch")
        references = [saved[f"reference_{i:04d}"] for i in range(3)]
        fp32 = [saved[f"fp32_{i:04d}"] for i in range(3)]
        for position, (_, golden, _, _, _) in enumerate(prepared):
            require((export / golden["logits"]["file"]).read_bytes() == saved[f"compiled_{position:04d}"].tobytes(),
                    "Export golden logits differ from parent captures")
        # Validate all three retained comparison rows/targets before native work.
        teacher_forced_quality_report(token_ids=tokens[:4], vocab_size=vocab,
                                     compiled_logits=references, reference_logits=references,
                                     fp32_logits=fp32, max_fp32_ppl_increase=.05)
    outcome["prerequisites_validated"] = True
    outcome["identity_pins"] = dict(pins)
    write_json(args.out / "identity.json", outcome)
    actual_rows, previous_kv = [], None
    for position, (item, golden, envelope, mutable, offset) in enumerate(prepared):
        outcome["active_position"] = position
        data = bytearray(static)
        if position:
            require(previous_kv is not None and len(previous_kv) == kv_size, "Missing hardware KV history")
            data[kv_offset:kv_offset + kv_size] = previous_kv
        input_kv = bytes(data[kv_offset:])
        require((not any(input_kv)) if position == 0 else input_kv == previous_kv,
                "Hardware KV history was not threaded exactly")
        program_path = args.out / f"step_{position:03d}.program.bin"
        with program_path.open("xb") as stream:
            stream.write(replace(envelope, data=bytes(data)).to_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        program_pin = sha(program_path)
        pins[str(program_path)] = program_pin
        raw_path, summary_path = (args.out / f"step_{position:03d}.{suffix}" for suffix in ("data.raw", "summary.json"))
        command = [str(runner), "--program", str(program_path), "--max-cycles", str(max(30_000_000, envelope.insn_count * 3000)),
                   "--json-out", str(summary_path), "--dram-dump-offset", str(base),
                   "--dram-dump-size", str(end - base), "--dram-dump-out", str(raw_path)]
        write_json(args.out / f"step_{position:03d}.command.json", {
            "argv": command, "cwd": str(ROOT), "native_step_timeout_s": 1200,
            "actual_program_sha256": program_pin, "input_kv_sha256": hashlib.sha256(input_kv).hexdigest(),
            "history_source": "all_zero_initial_KV" if position == 0 else f"preceding_RTL_step_{position - 1:03d}"})
        result = _run_native(command, args.out / f"step_{position:03d}.log")
        require(result.returncode == 0, "RTL execution returned nonzero")
        require(sha(program_path) == program_pin, "Actual program changed during hardware execution")
        summary = read_json(summary_path)
        validate_runner_summary(summary, require_hardware=True)
        require(type(summary.get("retired_instructions")) is int
                and summary["retired_instructions"] == envelope.insn_count,
                "RTL retired instruction count does not cover the branchless stream")
        raw = raw_path.read_bytes()
        require(raw[:shared_bytes] == static[:shared_bytes], "Shared weights/scales/constants changed")
        immutable_bytes = immutable_check(data, raw, mutable)
        logits_bytes = raw[offset:offset + width]
        require(len(logits_bytes) == width, "Truncated hardware FP16 logits")
        row = np.frombuffer(logits_bytes, dtype="<f2").copy()
        require(np.isfinite(row[:vocab]).all(), "Nonfinite live hardware FP16 logits")
        previous_kv = raw[kv_offset:kv_offset + kv_size]
        require(len(previous_kv) == kv_size and any(previous_kv), "Missing/nonzero hardware output KV")
        golden_logits = (export / golden["logits"]["file"]).read_bytes()
        golden_kv = (export / golden["kv"]["file"]).read_bytes()
        require(len(golden_logits) == width and len(golden_kv) == kv_size, "Golden comparison size mismatch")
        actual_rows.append(row)
        record = {"position": position, "token_id": tokens[position], "summary": summary,
                  "input_kv_sha256": hashlib.sha256(input_kv).hexdigest(),
                  "output_kv_sha256": hashlib.sha256(previous_kv).hexdigest(),
                  "kv_bytes": kv_size, "immutable_data_bytes_checked": immutable_bytes,
                  "golden_logits_exact": logits_bytes == golden_logits, "golden_kv_exact": previous_kv == golden_kv,
                  "golden_kv_differing_bytes": sum(a != b for a, b in zip(previous_kv, golden_kv)),
                  "program_sha256": program_pin, "raw_sha256": sha(raw_path), "summary_sha256": sha(summary_path)}
        outcome["steps"].append(record)
        pins[str(raw_path)], pins[str(summary_path)] = record["raw_sha256"], record["summary_sha256"]
        write_json(args.out / f"step_{position:03d}.result.json", record)
        print(f"Completed RTL position {position}; exact logits={record['golden_logits_exact']}, exact KV={record['golden_kv_exact']}", flush=True)
    require(len(actual_rows) == len(outcome["steps"]) == 3, "Incomplete three-step hardware coverage")
    quality = teacher_forced_quality_report(token_ids=tokens[:4], vocab_size=vocab,
                                          compiled_logits=actual_rows, reference_logits=references,
                                          fp32_logits=fp32, max_fp32_ppl_increase=.05)
    outcome["quality"] = quality
    _verify_parent_sources(report)
    require(_replay_source_pins() == source_pins, "Replay source set or hashes changed")
    verify_reference_manifest()
    outcome.update(source_sets_unchanged=True, whole_kv_carry_links=2, initial_zero_kv_steps=1)
    return quality["passed"]


def execute(args):
    args.out = args.out.resolve()
    # Never replace an existing campaign, even to record an error.
    args.out.mkdir(parents=True, exist_ok=False)
    started, pins = time.monotonic(), {}
    outcome = {"schema_version": 1, "status": "error", "passed": False, "steps": [],
               "arguments": {name: str(value) for name, value in vars(args).items()},
               "scope": "three teacher-forced hardware targets only, not 256-target RTL or corpus-wide quality",
               "thresholds": dict(THRESHOLDS), "prerequisites_validated": False,
               "golden_kv_is_history": False, "exact_golden_state_is_diagnostic": True,
               "original_files_scope": "self-contained retained inputs; original model/tokenizer/corpus files not reread",
               "resources": "external supervisor owns RSS/free-disk/campaign limits; native step timeout 1200 seconds",
               "memory_model": "runner defaults (latency 2, no fast-beats or beat-interval override)"}
    exit_code = 2
    try:
        passed = _replay(args, outcome, pins)
        outcome.update(passed=passed, status="passed" if passed else "failed")
        exit_code = 0 if passed else 1
    except BaseException as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 2
        outcome.update(passed=False, status="error",
                       error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        try:
            _verify_pins(pins)
            outcome["pinned_files_unchanged"] = True
        except Exception as exc:
            outcome.update(passed=False, status="error", pinned_files_unchanged=False,
                           identity_error={"type": type(exc).__name__, "message": str(exc)})
            exit_code = 2
        outcome["identity_pins"] = pins
        outcome["completed_steps"] = len(outcome["steps"])
        outcome["identities_unchanged"] = bool(outcome["prerequisites_validated"] and outcome["pinned_files_unchanged"]
                                               and outcome.get("source_sets_unchanged", False))
        outcome["elapsed_s"], outcome["exit_code"] = time.monotonic() - started, exit_code
        write_json(args.out / "report.json", outcome)
    print(json.dumps({"status": outcome["status"], "passed": outcome["passed"],
                      "completed_steps": outcome["completed_steps"], "report": str(args.out / "report.json")}), flush=True)
    return exit_code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--quality", type=Path, required=True, help="canonical passed software quality directory with three-step export")
    result.add_argument("--runner", type=Path, required=True, help="fresh mode-1 SFU/helper program runner")
    result.add_argument("--out", type=Path, required=True, help="new exclusive evidence directory")
    return result


def main(argv=None):
    return execute(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
