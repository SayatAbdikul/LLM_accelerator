"""Fast b16 decode gate — cycles + Port-A co-busy/lost + logits hash, NO retire trace.

The 90-min retire-trace profiler is too slow to iterate a concurrency change on.
This runs the same batched decode step under --fast-beats with only --json-out, so
it finishes in a few minutes and reports exactly the porta-scar gate quantities:

  total cycles / tok-s@34.41MHz   (must DROP vs baseline)
  Port-A dma_sys_cobusy           (must RISE above the 4,148,317 baseline — proves
                                   the KV overlap is REAL, not the pre-existing weight
                                   prefetch)
  Port-A sys_lost (+ breakdown)   (must stay 0 — no silently dropped systolic writes)
  forbidden_overlap_violation     (must stay false)
  logits sha1                     (byte-exact discriminator vs a known-good machine)

THIS IS A GATE, NOT A REPORT (2026-07-17). It previously `return 0`-ed
unconditionally after printing: it exited SUCCESS with sys_lost > 0, with
co-busy NOT risen, and with forbidden_overlap_violation true, and its sha1
baselines lived only in docs/t1_overlap_items.md. The roadmap (§5.2) names this
as THE gate for overlap changes, so "prints the numbers and always passes" meant
the porta-scar doctrine was enforced by nothing that could fail. It now exits
nonzero on:
  * sys_lost > 0                     (always — a dropped systolic write is a bug)
  * forbidden_overlap_violation      (always)
  * --expect-sha1 mismatch           (opt-in: byte-exactness vs a known machine)
  * --expect-cobusy-above N not met  (opt-in: proves overlap is REAL, not absent)

CAVEAT on sys_lost — it is necessary, NOT sufficient. Post-daef072 it is close to
a tautology on a halted run: obs_sys_porta_lost_w = sys_sram_a_en & sram_s_collision,
and a collision FAULTS the machine, so anything that reaches "halted" tends to have
lost==0 by construction. It counts DENIED REQUESTS, not LANDED WRITES: an S write
that fails to land without an address collision is invisible to it. Byte-exactness
(--expect-sha1) is the discriminator that actually sees that class.

USAGE (from repo root):
  .venv/bin/python software/tools/fast_gate_b16.py --batch 16 --position 510
  .venv/bin/python software/tools/fast_gate_b16.py --expect-sha1 205682b6515f7e85
  .venv/bin/python software/tools/fast_gate_b16.py --batch 1 --expect-sha1 eeab004014642d14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "software"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_FIXTURE = (
    REPO_ROOT / "software" / "tests" / "fixtures" / "generated" /
    "gpt2_converted_nanogpt.pt"
)
DEFAULT_RTL = (
    REPO_ROOT / "rtl" / "verilator" / "build" / "run_program_synth" / "Vtaccel_top"
)
BASELINE_COBUSY = 4_148_317  # HEAD (e6f7006) baseline; item-1 must push this UP

# Known-good logits sha1[:16], measured at 9a82e34 (item-2). These lived only in
# docs/t1_overlap_items.md:49,96 — a baseline a tool cannot read is not a gate.
# Keyed by (batch, position). Used as the default for --expect-sha1 when the run
# matches a known configuration; pass --expect-sha1 explicitly to override, or
# --no-sha1-check to skip (e.g. after an intentional, re-pinned numerics change).
#
# 2026-07-21 CORRECTION — the b1 entry was keyed (1, 510) but the number was
# measured at pos 511: docs/t1_overlap_items.md:90-91 tabulates "b1 pos-511"
# and "b16 pos-510", and I transcribed both under one position. The two shapes
# were benchmarked at DIFFERENT positions and always had been. Consequences,
# both live since e81d4cc introduced the pin:
#   `--batch 1 --position 510` FAILED spuriously (it compared a pos-510 run
#     against a pos-511 hash), and
#   `--batch 1 --position 511` — the shape the docs actually track — silently
#     ran with NO sha1 check at all, because the lookup missed.
# So the b1 leg of this gate has never once done what it claimed. Re-verified
# by running both: (1,511) reproduces eeab004014642d14 exactly; a pos-510 b1
# run is e43826086a1fbce4 (a different, un-pinned machine — NOT a regression).
# The position is part of the identity of the measurement; do not merge them.
KNOWN_LOGITS_SHA1 = {
    (16, 510): "205682b6515f7e85",
    (1, 511): "eeab004014642d14",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--position", type=int, default=510)
    ap.add_argument("--max-cycles", type=int, default=250_000_000)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--expect-sha1", type=str, default=None,
                    help="fail if the logits sha1[:16] differs. Defaults to the "
                         "KNOWN_LOGITS_SHA1 entry for (batch, position) if one exists.")
    ap.add_argument("--no-sha1-check", action="store_true",
                    help="skip the logits sha1 gate (use only for an intentional, "
                         "signed-off numerics change).")
    ap.add_argument("--expect-cobusy-above", type=int, default=None,
                    help="fail unless Port-A dma_sys_cobusy exceeds this. Use the "
                         "pre-change value to prove an overlap change is REAL.")
    args = ap.parse_args()

    if not args.rtl.exists():
        print(f"RTL binary missing: {args.rtl}\n"
              f"  build it with: make -C rtl/verilator run_program_synth",
              file=sys.stderr)
        return 2

    # Staleness gate (2026-07-24, the 6th "gate not wired to what it checks"
    # member): this script only EXECUTES the binary — nothing here rebuilds
    # it, so an RTL edit followed by a gate run silently measures the OLD
    # machine (it reported Δcycles=0 for a change whose true, later-measured
    # cost was +5,960 — a byte-inert change makes the stale run look exactly
    # like a passing gate). Hard-fail instead of guessing.
    if args.rtl == DEFAULT_RTL:
        rtl_mtime = args.rtl.stat().st_mtime
        src_dirs = [REPO_ROOT / "rtl" / "common" / "src",
                    REPO_ROOT / "rtl" / "verilator"]
        newest = max(
            (p for d in src_dirs for p in d.rglob("*")
             if p.suffix in (".sv", ".svh", ".cpp", ".h") and "build" not in p.parts),
            key=lambda p: p.stat().st_mtime,
        )
        if newest.stat().st_mtime > rtl_mtime:
            print(f"STALE BINARY: {newest.relative_to(REPO_ROOT)} is newer than "
                  f"the RTL binary.\n  rebuild first: make -C rtl/verilator "
                  f"run_program_synth", file=sys.stderr)
            return 2

    import torch  # noqa: deferred heavy import
    from bench_decode_cycles import build_decode_bins

    print(f"loading {args.fixture} ...", flush=True)
    payload = torch.load(args.fixture, map_location="cpu")

    td = Path(tempfile.mkdtemp(prefix="fast_gate_"))
    try:
        print(f"building decode bin (batch={args.batch}, pos={args.position}) ...",
              flush=True)
        bins = build_decode_bins(payload, [args.position], td, batch=args.batch)
        sum_json = td / "summary.json"
        print("running RTL (--fast-beats, no trace) ...", flush=True)
        cp = subprocess.run(
            [str(args.rtl), "--program", str(bins[args.position]),
             "--json-out", str(sum_json),
             "--fast-beats", "--max-cycles", str(args.max_cycles)],
            capture_output=True, text=True, timeout=28800,
        )
        if not sum_json.exists():
            print(f"RTL run failed rc={cp.returncode}: {cp.stderr[-800:]}",
                  file=sys.stderr)
            return 1
        s = json.loads(sum_json.read_text())
    finally:
        shutil.rmtree(td, ignore_errors=True)

    if s.get("status") != "halted" or s.get("fault"):
        print(f"FAULT/!halted: status={s.get('status')} fault={s.get('fault')} "
              f"ctx={s.get('fault_context')}", file=sys.stderr)
        return 1

    cyc = s["cycles"]
    pa = s.get("port_a", {})
    cobusy = pa.get("dma_sys_cobusy", 0)
    lost = pa.get("sys_lost", 0)
    logits = s.get("logits", [])
    lh = hashlib.sha1(
        ",".join(str(x) for x in logits).encode()
    ).hexdigest()[:16] if logits else "(none)"

    tag = f"[{args.label}] " if args.label else ""
    print(f"\n=== {tag}fast gate: batch={args.batch} pos={args.position} ===")
    print(f"  total cycles       {cyc:>15,}   per token {cyc/args.batch:>13,.0f}"
          f"   tok/s@34.41MHz {34.41e6/(cyc/args.batch):.4f}")
    print(f"  dma beats          {s['dma']['beat_count']:>15,}")
    print(f"  sys busy           {s['busy_cycles']['systolic']:>15,}")
    print(f"  Port-A co-busy     {cobusy:>15,}   (baseline {BASELINE_COBUSY:,};"
          f" {'RISEN +'+format(cobusy-BASELINE_COBUSY, ',') if cobusy>BASELINE_COBUSY else 'NOT risen'})")
    print(f"  Port-A sys_lost    {lost:>15,}   "
          f"(wr={pa.get('sys_lost_write',0)} abuf={pa.get('sys_lost_abuf',0)} "
          f"wbuf={pa.get('sys_lost_wbuf',0)} accum={pa.get('sys_lost_accum',0)} "
          f"drain={pa.get('sys_lost_drain',0)})")
    print(f"  overlap_violation  {str(s.get('forbidden_overlap_violation')):>15}")
    print(f"  logits sha1        {lh:>15}   (n={len(logits)})")

    # ---- the actual gate ---------------------------------------------------
    failures: list[str] = []

    if lost:
        failures.append(
            f"Port-A sys_lost = {lost:,} (must be 0) — the systolic had writes "
            f"denied. This is the porta scar: writes dropped with no backpressure "
            f"while the pointer advances."
        )

    if s.get("forbidden_overlap_violation"):
        failures.append(
            "forbidden_overlap_violation is true — an engine pair that must be "
            "serialized ran concurrently."
        )

    expect_sha1 = args.expect_sha1
    sha1_unchecked = False
    if expect_sha1 is None and not args.no_sha1_check:
        expect_sha1 = KNOWN_LOGITS_SHA1.get((args.batch, args.position))
        sha1_unchecked = expect_sha1 is None
    if expect_sha1 and not args.no_sha1_check:
        if lh != expect_sha1:
            failures.append(
                f"logits sha1 {lh} != expected {expect_sha1} — this machine is NOT "
                f"byte-identical to the known-good reference. If the change is an "
                f"intentional numerics change, re-pin KNOWN_LOGITS_SHA1 deliberately."
            )
        else:
            print(f"  sha1 gate          {'MATCH':>15}   (vs {expect_sha1})")
    elif not args.no_sha1_check:
        print(f"  sha1 gate          {'no baseline':>15}   "
              f"(no KNOWN_LOGITS_SHA1 entry for batch={args.batch} pos={args.position})")

    if args.expect_cobusy_above is not None and cobusy <= args.expect_cobusy_above:
        failures.append(
            f"Port-A co-busy {cobusy:,} did not rise above "
            f"{args.expect_cobusy_above:,} — the overlap this change claims is not "
            f"happening (a byte-exact result with zero added co-busy is a "
            f"structurally blind pass)."
        )

    if failures:
        print("\nGATE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  * {f}", file=sys.stderr)
        return 1

    # A missing pin used to print one quiet line and still report "GATE PASSED",
    # which reads as "byte-exactness verified" when nothing was compared — the
    # exact way the mis-keyed (1, 510) entry hid for so long. Say so in the
    # verdict itself; the sha1 is the only leg that sees a silent-corruption
    # class (`sys_lost == 0` is near-tautological, see the module docstring).
    if sha1_unchecked:
        print(f"\nGATE PASSED (WITHOUT the sha1 leg — no pin for batch="
              f"{args.batch} pos={args.position}; byte-exactness was NOT "
              f"checked. Known pins: "
              f"{', '.join(f'b{b}/pos{p}' for b, p in sorted(KNOWN_LOGITS_SHA1))}.)")
        return 0

    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
