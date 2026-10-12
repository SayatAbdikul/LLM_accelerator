"""Reproducible mode-1 compiler/fetch A/B matrix, with retained raw evidence.

Example (no downloads; nonzero deterministic synthetic weights)::

  PYTHONPATH=software .venv/bin/python software/tools/optimization_matrix.py \
    --synthetic --baseline-rtl PATH_TO_BUFFER0 --rtl PATH_TO_BUFFER1 \
    --batches 1,16,32 --positions 0,15,127 --prefill-tokens 1,16,128 \
    --decode-steps 2 --prefill-chunks 2 --out /tmp/optimization-matrix

Each (candidate, batch, prefill shape) is compiled once. Shape-only decode
positions use zero KV and are NOT accuracy claims. Contiguous teacher-forced
sequences carry the actual previous RTL KV dump, including across prefill
chunks. Compare logical FP16 logits and the complete KV image bit-for-bit;
record (but do not compare) full DRAM hashes because data dedup changes layout.
Every execution retains its program, command, stdout/stderr, JSON and raw DRAM.
This checks schedule equivalence, not model quality or timing signoff.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "software"))

from taccel.compiler.options import CompilerOptions
from tools.benchmark_support import (
    canonical_hash, check_runner_freshness, compiler_options_from_names,
    bundle_ptq_transform_manifest, decoder_logits_metrics, sha256_file, source_identity,
    runtime_toolchain_identity, validate_runner_summary,
)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def host_peak_rss():
    """OS-reported process high-water mark, not an incremental compile estimate.

    Darwin reports ru_maxrss in bytes; Linux reports KiB. Child RTL processes
    are deliberately excluded. Earlier calibration/compiles are included, so
    before/after values must not be mistaken for isolated per-case peaks.
    """
    try:
        import resource
    except ImportError:
        return {"available": False, "reason": "stdlib resource is unavailable"}
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024 if sys.platform.startswith("linux") else None
    return {"available": multiplier is not None,
            "peak_rss_bytes": int(raw * multiplier) if multiplier is not None else None,
            "ru_maxrss_raw": raw, "platform": sys.platform,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "scope": "current_python_process_lifetime", "includes_rtl_subprocesses": False}


def _json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)  # readers never observe a partially updated report


def parse_ints(value):
    values = [int(part) for part in value.split(",")]
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise ValueError("expected unique nonnegative comma-separated integers")
    return values


class DiskCapacityError(RuntimeError):
    """Stop a campaign before it can consume the configured free-space floor."""


def check_disk_capacity(path, required_bytes, min_free_gib=0):
    free = shutil.disk_usage(Path(path).resolve()).free
    floor = int(min_free_gib * (1024 ** 3))
    record = {"free_bytes": free, "required_bytes": required_bytes, "reserve_bytes": floor}
    if free - required_bytes < floor:
        raise DiskCapacityError(f"disk guard: free={free}, required={required_bytes}, reserve={floor}")
    return record


def fixture_identity(path):
    result = {"path": str(path.resolve()), "sha256": sha256_file(path), "metadata_sha256": None}
    metadata_path = path.with_suffix(path.suffix + ".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("checkpoint_sha256") != result["sha256"]:
            raise ValueError("matrix checkpoint does not match its paired metadata SHA-256")
        result.update(metadata_sha256=sha256_file(metadata_path), provenance=metadata.get("provenance"))
    return result


def snapshot_runners(paths, out):
    """Execute a private immutable copy even if a later build replaces its source."""
    directory = out / "runner-snapshots"
    directory.mkdir()
    result = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        target = directory / name
        shutil.copy2(path, target)
        if sha256_file(target) != digest or sha256_file(path) != digest:
            raise ValueError("runner changed while being snapshotted")
        result[name] = {"path": str(target.resolve()), "source_path": str(path.resolve()), "sha256": digest}
    return result


def retained_case_bytes(pb, layout, *, golden=False, existing_program=False):
    # Generated matrix binaries have no auxiliary manifests, hence their exact
    # serialized size is header + instructions + data (without another copy).
    from taccel.assembler.assembler import HEADER_SIZE
    program_bytes = (len(pb.to_bytes()) if pb.trace_manifest or pb.compiler_manifest
                     else HEADER_SIZE + len(pb.instructions) + len(pb.data))
    dump_bytes = layout["data_bytes"] * (2 if golden else 1)
    # Bound small JSON/log artifacts separately from exact binary/dump bytes.
    return {"program_bytes": program_bytes, "dump_bytes": dump_bytes,
            "metadata_margin_bytes": 16 * 1024 ** 2,
            "required_bytes": (0 if existing_program else program_bytes) + dump_bytes + 16 * 1024 ** 2}


def candidate_options(specs):
    result = {"baseline": CompilerOptions()}
    for spec in specs or ["combined=" + ",".join(asdict(CompilerOptions()))]:
        name, sep, flags = spec.partition("=")
        if not sep or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in result:
            raise ValueError("candidate must be a unique NAME=flag,flag (baseline is reserved)")
        result[name] = compiler_options_from_names(flags)
    return result


@dataclass(frozen=True)
class Step:
    name: str
    stream: str
    position: int
    rows: int
    sequence: bool


def workload_steps(batch, prefill_tokens, positions, decode_steps, prefill_chunks,
                   max_seq_len, *, shape_only=False):
    if batch not in (1, 16, 32) or prefill_tokens < 1:
        raise ValueError("batch must be 1/16/32 and prefill_tokens positive")
    if batch != 1 and prefill_tokens != 1:
        raise ValueError("multi-token prefill is single-stream only")
    if decode_steps < 0 or prefill_chunks < 1:
        raise ValueError("decode_steps must be nonnegative and prefill_chunks positive")
    # A one-row prefill has no runtime KV-read context; subsequent rows must
    # use decode, not rerun that static single-token prefill stream.
    chunks = 1 if prefill_tokens == 1 else prefill_chunks
    end = chunks * prefill_tokens + decode_steps
    if (not shape_only and end > max_seq_len) or (positions and max(positions) >= max_seq_len):
        raise ValueError("requested positions/sequence exceed the fixture context window")
    shape = ([Step(f"shape_decode_p{p}", "decode", p, batch, False) for p in positions]
             if prefill_tokens == 1 else [])
    if shape_only:
        if not shape:
            raise ValueError("shape-only requires nonempty positions and prefill_tokens=1")
        return shape
    sequence = [Step(f"sequence_prefill_{i}", "prefill", i * prefill_tokens,
                     prefill_tokens if batch == 1 else batch, True) for i in range(chunks)]
    sequence += [Step(f"sequence_decode_{i}", "decode", chunks * prefill_tokens + i,
                      batch, True) for i in range(decode_steps)]
    return shape + sequence


def build_program(tiny, step, kv_image=None):
    """Runtime-patch a previously compiled bundle without simulator allocation."""
    from taccel.assembler.assembler import ProgramBinary
    from taccel.runtime.host_runner import BundlePatcher

    bundle = tiny.build.bundle
    driver = BundlePatcher(bundle)
    batched = tiny.n_streams > 1
    positions = ([step.position] * step.rows if batched or step.stream == "decode"
                 else list(range(step.position, step.position + step.rows)))
    tokens = [(step.position + 7 * row) % tiny.config.vocab_size for row in range(step.rows)]
    driver._patch_embeddings(step.stream, tokens, positions)
    driver._patch_kv_bases(step.position, stream=step.stream)
    valid_len = step.position + (step.rows if not batched and step.stream == "prefill" else 1)
    driver._patch_attention_context(step.stream, step.position, valid_len)
    image = bundle.materialize(reset_runtime=False)
    data = bytearray(image[bundle.data_base:])
    if kv_image is not None:
        if len(kv_image) != bundle.kv_cache_size_bytes:
            raise ValueError("previous KV dump does not match this bundle")
        start = bundle.kv_cache_base - bundle.data_base
        data[start:start + len(kv_image)] = kv_image
    start = bundle.prefill_instrs_offset if step.stream == "prefill" else bundle.decode_instrs_offset
    length = len(bundle.prefill_instrs if step.stream == "prefill" else bundle.decode_instrs)
    instructions = bytes(image[start:start + length])
    pb = ProgramBinary(instructions=instructions, data=bytes(data), entry_point=0,
                       insn_count=len(instructions) // 8, data_base=int(bundle.data_base),
                       input_offset=0, pos_embed_patch_dram_offset=0,
                       pos_embed_cls_dram_offset=0, cls_token_dram_offset=0,
                       trace_manifest={}, compiler_manifest={})
    # Batched prefill executes every stream but, by the public bundle's
    # default contract, stores only its last query's logits in slot zero.
    # Reserved/unwritten logits rows are not logical numerical outputs.
    stored_rows = int(bundle.prefill_store_rows) if step.stream == "prefill" else step.rows
    logical_source_rows = ([step.rows - 1] if step.stream == "prefill" and stored_rows == 1
                           else list(range(stored_rows)))
    layout = {"data_base": int(bundle.data_base), "data_bytes": len(data),
              "weights_bytes": len(bundle.shared_data),
              "logits_offset": int(bundle.prefill_logits_offset if step.stream == "prefill"
                                   else bundle.decode_logits_offset) - int(bundle.data_base),
              "logits_row_bytes": int(bundle.logits_row_bytes), "rows": stored_rows,
              "token_rows": step.rows, "logical_source_rows": logical_source_rows,
              "vocab_size": int(tiny.config.vocab_size),
              "kv_offset": int(bundle.kv_cache_base - bundle.data_base),
              "kv_bytes": int(bundle.kv_cache_size_bytes),
              "tokens": tokens, "positions": positions}
    return pb, layout


def inspect_output(raw, layout, expected_weights):
    if len(raw) != layout["data_bytes"]:
        raise ValueError("incomplete DRAM dump")
    if raw[:layout["weights_bytes"]] != expected_weights:
        raise ValueError("execution overwrote immutable shared weights/constants")
    start = layout["logits_offset"]
    logits = decoder_logits_metrics(
        raw[start:start + layout["rows"] * layout["logits_row_bytes"]],
        rows=layout["rows"], vocab_size=layout["vocab_size"],
        row_bytes=layout["logits_row_bytes"])
    if not logits["all_finite"]:
        raise ValueError("nonfinite logits are not a passing equivalence result")
    kv = raw[layout["kv_offset"]:layout["kv_offset"] + layout["kv_bytes"]]
    return {"dram_sha256": _hash(raw), "immutable_weights_sha256": _hash(expected_weights),
            "logits": logits, "kv_sha256": _hash(kv)}, kv


def compare_outputs(baseline, candidate):
    """Do not confuse changed padding/layout with changed logical results."""
    keys = ("rows", "vocab_size", "sha256")
    logits_equal = all(baseline["logits"][k] == candidate["logits"][k] for k in keys)
    kv_equal = baseline["kv_sha256"] == candidate["kv_sha256"]
    return {"passed": logits_equal and kv_equal,
            "logical_logits_equal": logits_equal, "kv_equal": kv_equal}


def cache_coverage(metrics):
    """Make actual optimization coverage and guarded fallbacks easy to audit."""
    coverage = {}
    for stream, counts in metrics.items():
        events = counts["optimization_events"]
        coverage[stream] = {
            "qkv_hits": sum(e["kind"] == "qkv_quantization_hit" for e in events),
            "fc2_hits": sum(e["kind"] == "fc2_quantization_hit" for e in events),
            "fc2_admissions": [e for e in events if e["kind"] == "fc2_cache_admission"],
            "fallbacks": [e for e in events if "fallback" in e["kind"]],
        }
    return coverage


def require_fc2_coverage(coverage):
    for stream, result in coverage.items():
        if not result["fc2_hits"] or result["fallbacks"]:
            raise ValueError(f"{stream}: strict FC2 gate requires real cache hits without reuse fallback")


def run_preserved(rtl, pb, layout, out, *, max_cycles, beat_interval, expected_fetch,
                  program_source=None, min_free_gib=0):
    out.parent.mkdir(parents=True, exist_ok=True)
    sizing = retained_case_bytes(pb, layout, existing_program=program_source is not None)
    disk = check_disk_capacity(out.parent, sizing["required_bytes"], min_free_gib)
    out.mkdir(parents=True, exist_ok=False)
    program = Path(program_source).resolve() if program_source is not None else out / "program.bin"
    summary_file, dump = out / "runner.json", out / "dram.raw"
    if program_source is None:
        program.write_bytes(pb.to_bytes())
    original_hash = sha256_file(program)
    _json(out / "storage.json", {**sizing, **disk, "program_reused": program_source is not None})
    _json(out / "layout.json", layout)
    command = [str(rtl.resolve()), "--program", str(program.resolve()), "--json-out",
               str(summary_file.resolve()), "--fast-beats", "--beat-interval", str(beat_interval),
               "--max-cycles", str(max_cycles), "--dram-dump-offset", str(layout["data_base"]),
               "--dram-dump-size", str(layout["data_bytes"]), "--dram-dump-out", str(dump.resolve())]
    _json(out / "command.json", command)
    started = time.monotonic()
    # File-backed output survives timeout, faults, and exceptions.
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=9000)
    if process.returncode != 0:
        raise RuntimeError(f"RTL exited {process.returncode}; evidence retained in {out}")
    summary = json.loads(summary_file.read_text())
    validate_runner_summary(summary)
    if summary.get("configuration", {}).get("fetch_buffer_enabled") != bool(expected_fetch):
        raise ValueError("runner fetch-buffer configuration differs from the requested A/B variant")
    summary["wall_time_s"] = time.monotonic() - started
    if sha256_file(program) != original_hash:
        raise ValueError("program changed during RTL execution")
    summary["program_sha256"] = original_hash
    return summary, dump.read_bytes()


def run_golden_preserved(pb, layout, out):
    from taccel.golden_model.simulator import Simulator
    sim = Simulator()
    sim.load_program(pb)
    sim.run()
    if not sim.state.halted:
        raise ValueError("golden program did not halt")
    raw = bytes(sim.state.dram[layout["data_base"]:layout["data_base"] + layout["data_bytes"]])
    (out / "golden_dram.raw").write_bytes(raw)
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    fixture = parser.add_mutually_exclusive_group(required=True)
    fixture.add_argument("--fixture", type=Path)
    fixture.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-d-model", type=int, default=64)
    parser.add_argument("--synthetic-heads", type=int, default=4)
    parser.add_argument("--synthetic-layers", type=int, default=1)
    parser.add_argument("--synthetic-context", type=int, default=512)
    parser.add_argument("--rtl", required=True, type=Path, help="candidate mode-1 runner")
    parser.add_argument("--baseline-rtl", required=True, type=Path)
    parser.add_argument("--candidate", action="append", help="NAME=flag,flag; default enables all")
    parser.add_argument("--candidate-fetch-buffer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--baseline-fetch-buffer", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batches", default="1,16,32")
    parser.add_argument("--positions", default="0,15,127")
    parser.add_argument("--prefill-tokens", default="1,16,128")
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--prefill-chunks", type=int, default=2)
    parser.add_argument("--shape-only", action="store_true", help="only the requested zero-KV shape cases")
    parser.add_argument("--preflight-only", action="store_true", help="compile and size all cases without running RTL or storing program images")
    parser.add_argument("--min-free-gib", type=float, default=0, help="minimum free space on the real output filesystem")
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    parser.add_argument("--beat-interval", type=int, default=1)
    parser.add_argument("--golden", action="store_true", help="also retain golden outputs (A/B checked separately)")
    parser.add_argument("--require-fc2-reuse", action="store_true",
                        help="for FC2-enabled candidates, require actual cache hits and no reuse fallback in both streams")
    parser.add_argument("--allow-stale-rtl", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.synthetic and (
        min(args.synthetic_d_model, args.synthetic_heads, args.synthetic_layers, args.synthetic_context) < 1
        or args.synthetic_d_model % args.synthetic_heads
    ):
        parser.error("synthetic dimensions must be positive and d_model divisible by heads")
    options = candidate_options(args.candidate)
    batches, prefills = map(parse_ints, (args.batches, args.prefill_tokens))
    positions = parse_ints(args.positions) if args.positions else []
    if args.beat_interval < 1 or args.max_cycles < 1 or args.min_free_gib < 0:
        parser.error("cycle and beat limits must be positive")
    if args.out.exists():
        parser.error("--out must be new: existing evidence is never overwritten")
    if not args.allow_stale_rtl:
        for path in {args.rtl, args.baseline_rtl}:
            check_runner_freshness(REPO, path)
    identity = source_identity(REPO)
    args.out.mkdir(parents=True)
    runners = snapshot_runners({"baseline": args.baseline_rtl, "candidate": args.rtl}, args.out)
    import torch
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
    from tools.train_tiny_fixture import _make_checkpoint_payload, payload_sha256
    from tools.bench_decode_cycles import FREEZE_PTQ_PRESET

    if args.synthetic:
        payload = _make_checkpoint_payload(n_layer=args.synthetic_layers, n_head=args.synthetic_heads,
                                          d_model=args.synthetic_d_model,
                                          block_size=args.synthetic_context, weight_scale=.02)
        fixture_path = args.out / "synthetic.pt"
        torch.save(payload, fixture_path)
    else:
        fixture_path = args.fixture.resolve()
        fixture_identity(fixture_path)  # validate paired metadata before deserialization
        payload = torch.load(fixture_path, map_location="cpu", weights_only=False)
    scales = build_calibration_scales(payload)
    manifest = {"schema_version": 1, "source": identity, **runtime_toolchain_identity(),
                "fixture": {**fixture_identity(fixture_path),
                            "payload_sha256": payload_sha256(payload), "synthetic": args.synthetic},
                "calibration_sha256": canonical_hash(scales),
                "runners": runners,
                "ptq_transforms": bundle_ptq_transform_manifest(FREEZE_PTQ_PRESET),
                "options": {name: asdict(option) for name, option in options.items()},
                "arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "memory": {"read_beat_bytes": 16, "read_core_cycles_per_beat": args.beat_interval,
                           "write_pacing": "unthrottled", "shared_total_bandwidth": False},
                "clock": "unmeasured; report cycles, not measured tokens/s",
                "comparison": "bit-exact logical FP16 logits and full KV, not relocated full DRAM"}
    _json(args.out / "manifest.json", manifest)
    report = {"schema_version": 1, "status": "running", "runs": [], "comparisons": [],
              "errors": [], "unsupported_combinations": [], "compiles": [], "planned_storage": []}
    results = {}
    for batch in batches:
        for prefill in prefills:
            if batch != 1 and prefill != 1:
                report["unsupported_combinations"].append({"batch": batch, "prefill_tokens": prefill,
                                                           "reason": "multi-token prefill is single-stream only"})
                continue
            try:
                steps = workload_steps(batch, prefill, positions, args.decode_steps,
                                       args.prefill_chunks, int(payload["model_args"]["block_size"]),
                                       shape_only=args.shape_only)
            except ValueError as exc:
                report["errors"].append({"case": f"b{batch}_p{prefill}", "error": str(exc)})
                _json(args.out / "report.json", report)
                continue
            context = max(step.position + (step.rows if batch == 1 and step.stream == "prefill" else 1)
                          for step in steps)
            for name, option in options.items():
                case = f"b{batch}_p{prefill}/{name}"
                directory = args.out / case
                directory.mkdir(parents=True)
                try:
                    rss_before = host_peak_rss()
                    started = time.monotonic()
                    tiny = build_stage3_tiny_decoder_bundle(
                        payload, smoke_decode_steps=context - 1, calibration_scales=scales,
                        ptq_preset=FREEZE_PTQ_PRESET, batch=batch, prefill_tokens=prefill,
                        compiler_options=option, target="rtl-gen2")
                    metrics = {stream: getattr(tiny.build, stream + "_codegen").compilation_metrics()
                               for stream in ("prefill", "decode")}
                    compile_record = {"case": case, "elapsed_s": time.monotonic() - started,
                                      "metrics": metrics, "options": asdict(option),
                                      "cache_coverage": cache_coverage(metrics),
                                      "host_peak_rss_before": rss_before,
                                      "host_peak_rss_after": host_peak_rss()}
                    _json(directory / "compile.json", compile_record)
                    report["compiles"].append(compile_record)
                    if args.require_fc2_reuse and option.reuse_fc2_quantization:
                        require_fc2_coverage(compile_record["cache_coverage"])
                    kv_image, golden_kv = None, None
                    for step in steps:
                        pb, layout = build_program(tiny, step, kv_image if step.sequence else None)
                        sizing = retained_case_bytes(pb, layout, golden=args.golden)
                        report["planned_storage"].append({"case": case, "step": step.name, **sizing})
                        if args.preflight_only:
                            continue
                        check_disk_capacity(directory, sizing["required_bytes"], args.min_free_gib)
                        out = directory / step.name
                        baseline = name == "baseline"
                        summary, raw = run_preserved(
                            Path(runners["baseline" if baseline else "candidate"]["path"]), pb, layout, out,
                            max_cycles=args.max_cycles, beat_interval=args.beat_interval,
                            expected_fetch=args.baseline_fetch_buffer if baseline else args.candidate_fetch_buffer,
                            min_free_gib=args.min_free_gib)
                        numerical, next_kv = inspect_output(raw, layout, tiny.build.bundle.shared_data)
                        record = {"case": case, "step": asdict(step), "output": numerical,
                                  "cycles": summary["cycles"], "tokens_executed": step.rows,
                                  "program_sha256": summary.get("program_sha256", sha256_file(out / "program.bin")),
                                  "cycles_per_token": summary["cycles"] / step.rows,
                                  "workload_kind": "teacher_forced_contiguous" if step.sequence else "shape_only_zero_kv"}
                        if step.sequence:
                            kv_image = next_kv
                        if args.golden:
                            golden_pb, _ = build_program(tiny, step, golden_kv if step.sequence else None)
                            golden_raw = run_golden_preserved(golden_pb, layout, out)
                            golden_output, next_golden_kv = inspect_output(golden_raw, layout, tiny.build.bundle.shared_data)
                            # Numerical freeze permits characterized RTL GELU ULP differences;
                            # this diagnostic is not used to relax the exact schedule A/B gate.
                            record["golden_diagnostic"] = compare_outputs(golden_output, numerical)
                            if step.sequence:
                                golden_kv = next_golden_kv
                        _json(out / "result.json", record)
                        report["runs"].append(record)
                        key = (batch, prefill, step.name)
                        if baseline:
                            results[key] = record
                        else:
                            if key not in results:
                                raise ValueError("matching baseline result is missing")
                            comparison = compare_outputs(results[key]["output"], numerical)
                            comparison.update(case=case, step=step.name,
                                              baseline_cycles=results[key]["cycles"],
                                              candidate_cycles=summary["cycles"],
                                              cycle_ratio=summary["cycles"] / results[key]["cycles"])
                            report["comparisons"].append(comparison)
                        _json(args.out / "report.json", report)
                except Exception as exc:
                    error = {"case": case, "error": str(exc), "traceback": traceback.format_exc()}
                    report["errors"].append(error)
                    _json(directory / "error.json", error)
                    _json(args.out / "report.json", report)
                    print(f"FAILED {case}: {exc}", flush=True)
                    if isinstance(exc, DiskCapacityError):
                        report["status"] = "resource_guard_stopped"
                        _json(args.out / "report.json", report)
                        return 1
    if args.preflight_only:
        required = sum(x["required_bytes"] for x in report["planned_storage"])
        try:
            report["campaign_storage"] = check_disk_capacity(args.out, required, args.min_free_gib)
        except DiskCapacityError as exc:
            report["errors"].append({"case": "campaign_storage", "error": str(exc)})
        report["status"] = "preflight_failed" if report["errors"] else "preflight_passed"
        report["numerical_equivalence_executed"] = False
        report["host_peak_rss"] = host_peak_rss()
        _json(args.out / "report.json", report)
        print(f"{report['status']}: {required} bytes for {len(report['planned_storage'])} planned executions")
        return int(bool(report["errors"]))
    passed = bool(report["comparisons"]) and not report["errors"] and all(c["passed"] for c in report["comparisons"])
    report["status"] = "passed" if passed else "failed"
    report["host_peak_rss"] = host_peak_rss()
    _json(args.out / "report.json", report)
    print(f"{report['status']}: {len(report['runs'])} runs; {len(report['comparisons'])} comparisons; {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
