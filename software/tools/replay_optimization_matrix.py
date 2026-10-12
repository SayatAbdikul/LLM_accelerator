"""Replay retained matrix programs at bounded read pacing; never recompile them.

Each run checks exact finite logical FP16 logits and complete KV against the
original retained result and the same-pacing fresh baseline. Program files are
read from the original matrix, not copied. New commands, hashes, runner snapshots,
raw dumps and failure records are retained in a new output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "software"))

from taccel.assembler.assembler import ProgramBinary
from tools import optimization_matrix as matrix
from tools.benchmark_support import (
    bundle_ptq_transform_manifest, runtime_toolchain_identity, sha256_file, source_identity,
)
from tools.bench_decode_cycles import FREEZE_PTQ_PRESET


def retained_case(root, report, case, step):
    """Resolve only a single recorded case beneath the original evidence root."""
    if any(part in ("", ".", "..") for part in (case + "/" + step).split("/")):
        raise ValueError("invalid retained case path")
    directory = (root / case / step).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("retained case escapes matrix directory")
    matches = [r for r in report["runs"] if r["case"] == case and r["step"]["name"] == step]
    if len(matches) != 1:
        raise ValueError(f"expected one recorded result for {case}/{step}")
    result = matches[0]
    disk_result = json.loads((directory / "result.json").read_text())
    if result != disk_result:
        raise ValueError("retained result differs from source report")
    program = directory / "program.bin"
    digest = sha256_file(program)
    if result.get("program_sha256", digest) != digest:
        raise ValueError("retained program hash mismatch")
    layout = json.loads((directory / "layout.json").read_text())
    pb = ProgramBinary.from_bytes(program.read_bytes())
    if len(pb.data) != layout["data_bytes"] or pb.data_base != layout["data_base"]:
        raise ValueError("retained layout does not match program")
    original, _ = matrix.inspect_output((directory / "dram.raw").read_bytes(), layout,
                                        pb.data[:layout["weights_bytes"]])
    if original != result["output"]:
        raise ValueError("retained raw output differs from recorded hashes")
    return pb, layout, result, program, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--baseline-rtl", required=True, type=Path)
    parser.add_argument("--rtl", required=True, type=Path)
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--candidates", default="baseline,combined")
    parser.add_argument("--step", default="shape_decode_p127")
    parser.add_argument("--intervals", default="1,2,4")
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    parser.add_argument("--min-free-gib", type=float, default=0)
    parser.add_argument("--require-fc2-reuse", action="store_true")
    parser.add_argument("--allow-stale-rtl", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    batches, intervals = matrix.parse_ints(args.batches), matrix.parse_ints(args.intervals)
    candidates = args.candidates.split(",")
    if (not intervals or min(intervals) < 1 or args.max_cycles < 1 or args.min_free_gib < 0
            or candidates[0] != "baseline" or len(candidates) != len(set(candidates))):
        parser.error("positive limits and unique candidates with baseline first are required")
    if args.out.exists():
        parser.error("--out must be new: existing evidence is never overwritten")
    source_report = json.loads((args.matrix / "report.json").read_text())
    source_manifest = json.loads((args.matrix / "manifest.json").read_text())
    if source_report["status"] != "passed":
        parser.error("source matrix must have passed its complete numerical gates")
    if not set(candidates) <= set(source_manifest["options"]):
        parser.error("candidate not present in source manifest")
    if not args.allow_stale_rtl:
        for path in {args.baseline_rtl, args.rtl}:
            matrix.check_runner_freshness(REPO, path)
    args.out.mkdir(parents=True)
    runners = matrix.snapshot_runners({"baseline": args.baseline_rtl, "candidate": args.rtl}, args.out)
    manifest = {
        "schema_version": 1, "source": source_identity(REPO), "runners": runners,
        **runtime_toolchain_identity(),
        "original_matrix": str(args.matrix.resolve()),
        "original_report_sha256": sha256_file(args.matrix / "report.json"),
        "original_manifest_sha256": sha256_file(args.matrix / "manifest.json"),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "original_options": {name: source_manifest["options"][name] for name in candidates},
        "ptq_transforms": bundle_ptq_transform_manifest(FREEZE_PTQ_PRESET),
        "memory": {"read_beat_bytes": 16, "read_core_cycles_per_beat": intervals,
                   "write_pacing": "unthrottled", "shared_total_bandwidth": False},
        "program_images_copied": False, "accuracy_validated": False,
    }
    matrix._json(args.out / "manifest.json", manifest)
    report = {"status": "running", "runs": [], "comparisons": [], "errors": []}
    baselines = {}
    for batch in batches:
        for candidate in candidates:
            case = f"b{batch}_p1/{candidate}"
            try:
                pb, layout, original, program, program_hash = retained_case(args.matrix, source_report, case, args.step)
                if args.require_fc2_reuse and source_manifest["options"][candidate].get("reuse_fc2_quantization"):
                    compiled = [c for c in source_report["compiles"] if c["case"] == case]
                    if len(compiled) != 1:
                        raise ValueError("source compile coverage missing")
                    matrix.require_fc2_coverage(compiled[0]["cache_coverage"])
                for interval in intervals:
                    out = args.out / case / f"{args.step}_beat{interval}"
                    baseline = candidate == "baseline"
                    summary, raw = matrix.run_preserved(
                        Path(runners["baseline" if baseline else "candidate"]["path"]), pb, layout, out,
                        max_cycles=args.max_cycles, beat_interval=interval, expected_fetch=not baseline,
                        program_source=program, min_free_gib=args.min_free_gib)
                    numerical, _ = matrix.inspect_output(raw, layout, pb.data[:layout["weights_bytes"]])
                    historical = matrix.compare_outputs(original["output"], numerical)
                    record = {"case": case, "step": args.step, "beat_interval": interval,
                              "cycles": summary["cycles"], "output": numerical,
                              "program_path": str(program), "program_sha256": program_hash,
                              "matches_original": historical}
                    report["runs"].append(record)
                    matrix._json(out / "result.json", record)
                    if not historical["passed"] or sha256_file(program) != program_hash:
                        raise ValueError("replay differs from original output or program changed")
                    key = (batch, interval)
                    if baseline:
                        baselines[key] = record
                    else:
                        comparison = matrix.compare_outputs(baselines[key]["output"], numerical)
                        comparison.update(case=case, step=args.step, beat_interval=interval,
                                          baseline_cycles=baselines[key]["cycles"], candidate_cycles=summary["cycles"],
                                          cycle_ratio=summary["cycles"] / baselines[key]["cycles"])
                        report["comparisons"].append(comparison)
                    matrix._json(args.out / "report.json", report)
            except Exception as exc:
                error = {"case": case, "error": str(exc), "traceback": traceback.format_exc()}
                report["errors"].append(error)
                matrix._json(args.out / "report.json", report)
                if isinstance(exc, matrix.DiskCapacityError):
                    break
        if report["errors"] and "disk guard" in report["errors"][-1]["error"]:
            break
    passed = (bool(report["comparisons"]) and not report["errors"]
              and all(c["passed"] for c in report["comparisons"]))
    report["status"] = "passed" if passed else "failed"
    report["host_peak_rss"] = matrix.host_peak_rss()
    matrix._json(args.out / "report.json", report)
    print(f"{report['status']}: {len(report['runs'])} replays, {len(report['comparisons'])} same-pacing comparisons")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
