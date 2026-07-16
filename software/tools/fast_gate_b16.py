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

USAGE (from repo root):
  .venv/bin/python software/tools/fast_gate_b16.py --batch 16 --position 510
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--position", type=int, default=510)
    ap.add_argument("--max-cycles", type=int, default=250_000_000)
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    if not args.rtl.exists():
        print(f"RTL binary missing: {args.rtl}\n"
              f"  build it with: make -C rtl/verilator run_program_synth",
              file=sys.stderr)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
