"""Measure one hardware decode step and compare it with a matched report.

The workload uses zero-filled historical KV rows: this is a shape-only cycle
measurement. Finite logits and byte identity are useful regression checks, but
do not establish language-model accuracy. Actual logits are read from their
FP16 DRAM region; legacy summary.logits contains an INT32 ACCUM diagnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "software"))

from tools.bench_decode_cycles import DEFAULT_FIXTURE, DEFAULT_RTL, FREEZE_PTQ_PRESET, build_decode_bins, run_one
from tools.benchmark_support import (
    benchmark_manifest, check_runner_freshness, compiler_options_from_names,
)


def compare_report(current: dict, baseline: dict, *, max_regression_percent: float) -> list[str]:
    """Only compare reports with identical inputs, shape and memory pacing."""
    failures = []
    cm, bm = current["manifest"], baseline["manifest"]
    for key in ("batch", "positions", "memory", "workload_kind"):
        if cm.get(key) != bm.get(key):
            failures.append(f"baseline mismatch in {key}")
    if cm["fixture"]["sha256"] != bm["fixture"]["sha256"]:
        failures.append("baseline checkpoint hash differs")
    cr, br = current["rows"][0], baseline["rows"][0]
    if cr["program_manifest"]["calibration_sha256"] != br["program_manifest"]["calibration_sha256"]:
        failures.append("baseline calibration hash differs")
    for key in ("sfu_synth_mode", "helper_synth_mode"):
        if cr["configuration"].get(key) != br["configuration"].get(key):
            failures.append(f"baseline datapath mode differs: {key}")
    if not cr["decoder_logits"].get("all_finite") or not br["decoder_logits"].get("all_finite"):
        failures.append("matched performance promotion requires finite FP16 logits on both runs")
    if cr["decoder_logits"]["sha256"] != br["decoder_logits"]["sha256"]:
        failures.append("actual FP16 decoder logits differ from matched baseline")
    limit = br["cycles"] * (1 + max_regression_percent / 100)
    if cr["cycles"] > limit:
        failures.append(f"cycle regression: {cr['cycles']} > allowed {limit:g}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--batch", type=int, choices=(1, 16, 32), default=16)
    ap.add_argument("--position", type=int, default=510)
    ap.add_argument("--max-cycles", type=int, default=250_000_000)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, help="report JSON; defaults to a retained temporary directory")
    ap.add_argument("--baseline-report", type=Path, help="matched fast-gate or single-position benchmark report")
    ap.add_argument("--max-cycle-regression-percent", type=float, default=0.0)
    ap.add_argument("--expect-logits-sha256", help="expected hash of actual FP16 DRAM logits")
    ap.add_argument("--expect-sha1", help="legacy explicit ACCUM diagnostic SHA1[:16], not decoder logits")
    ap.add_argument("--no-sha1-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-cobusy-above", type=int)
    ap.add_argument("--require-finite-logits", action="store_true")
    ap.add_argument("--fmax-mhz", type=float, help="optional user-assumed clock; no measured frequency is implied")
    ap.add_argument("--beat-interval", type=int, choices=(1, 2, 4), default=1)
    ap.add_argument("--compiler-optimizations", default="")
    args = ap.parse_args()
    if args.position < 0 or args.max_cycles <= 0:
        ap.error("position must be non-negative and max-cycles positive")
    if args.fmax_mhz is not None and not (0 < args.fmax_mhz < float("inf")):
        ap.error("--fmax-mhz must be finite and positive")
    if not (-100 < args.max_cycle_regression_percent < float("inf")):
        ap.error("cycle regression allowance must be finite and greater than -100")
    if args.baseline_report and len(json.loads(args.baseline_report.read_text()).get("rows", [])) != 1:
        ap.error("baseline report must contain exactly one matched position")
    if not args.rtl.exists():
        ap.error(f"runner missing: {args.rtl}; build make -C rtl/verilator run_program_synth")
    check_runner_freshness(REPO_ROOT, args.rtl)
    options = compiler_options_from_names(args.compiler_optimizations)
    manifest = benchmark_manifest(
        REPO_ROOT, args.fixture, args.rtl, batch=args.batch, positions=[args.position],
        beat_interval=args.beat_interval, fmax_mhz=args.fmax_mhz,
        compiler_options=asdict(options), ptq_preset=FREEZE_PTQ_PRESET)
    import torch

    payload = torch.load(args.fixture, map_location="cpu")
    with tempfile.TemporaryDirectory(prefix="fast_gate_bins_") as work:
        bins = build_decode_bins(payload, [args.position], Path(work), batch=args.batch,
                                 compiler_options=options)
        summary = run_one(args.rtl, bins[args.position], args.max_cycles,
                          beat_interval=args.beat_interval)
    metrics = summary["decoder_logits"]
    row = {"position": args.position, "cycles": summary["cycles"],
           "per_tok_cycles": summary["cycles"] / args.batch,
           "configuration": summary["configuration"], "decoder_logits": metrics,
           "program_manifest": summary["program_manifest"], "runner_summary": summary}
    report = {"manifest": manifest, "label": args.label, "rows": [row]}
    failures = []
    if args.require_finite_logits and not metrics["all_finite"]:
        failures.append("non-finite values in actual FP16 decoder logits")
    if args.expect_logits_sha256 and metrics["sha256"] != args.expect_logits_sha256:
        failures.append("actual FP16 decoder logits SHA256 mismatch")
    if args.expect_sha1 and not args.no_sha1_check:
        diagnostic = hashlib.sha1(",".join(str(x) for x in summary.get("logits", [])).encode()).hexdigest()[:16]
        if diagnostic != args.expect_sha1:
            failures.append(f"legacy ACCUM diagnostic SHA1 mismatch: {diagnostic}")
    cobusy = summary.get("port_a", {}).get("dma_sys_cobusy", 0)
    if args.expect_cobusy_above is not None and cobusy <= args.expect_cobusy_above:
        failures.append(f"co-busy {cobusy} did not exceed {args.expect_cobusy_above}")
    if args.baseline_report:
        failures.extend(compare_report(report, json.loads(args.baseline_report.read_text()),
                                       max_regression_percent=args.max_cycle_regression_percent))
    report["failures"] = failures
    report["matched_baseline"] = str(args.baseline_report) if args.baseline_report else None
    output = args.out or Path(tempfile.mkdtemp(prefix="fast_gate_report_")) / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"batch={args.batch} position={args.position} cycles={row['cycles']:,} "
          f"cycles/token={row['per_tok_cycles']:,.2f}")
    print(f"decoder FP16 logits sha256={metrics['sha256']} "
          f"finite={metrics['finite_count']}/{metrics['element_count']}")
    if args.fmax_mhz is not None:
        print(f"Assumed {args.fmax_mhz:g} MHz: {args.fmax_mhz * 1e6 / row['per_tok_cycles']:.4f} tokens/s")
    print(f"Report: {output}")
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("MATCHED GATE PASSED" if args.baseline_report else
          "MEASURED: hardware status passed; no matched performance baseline supplied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
