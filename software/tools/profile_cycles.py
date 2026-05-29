"""Ad-hoc per-engine cycle profiler.

Builds the frozen tiny-nanoGPT prefill bundle and runs it on the instrumented
Verilator run_program binary, then prints the busy_cycles / sync_wait / dma
counters so we can see the REAL per-engine cycle split (vs the
architecture-derived estimates).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # software/

from tools.rtl_cosim import serialize_prefill_bundle, RTL_BINARY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GPT2_124M_FIXTURE = (REPO_ROOT / "software" / "tests" / "fixtures" / "generated"
                     / "gpt2_converted_nanogpt.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt2-124m", action="store_true",
                    help="profile the real GPT-2 124M prefill (needs binary "
                         "built with DRAM_SIZE>=1<<30; ~5 min)")
    ap.add_argument("--max-cycles", type=int, default=5_000_000)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if args.gpt2_124m:
        import torch
        payload = torch.load(GPT2_124M_FIXTURE, map_location="cpu")
        sp = serialize_prefill_bundle(token_id=464, payload=payload)
    else:
        sp = serialize_prefill_bundle(token_id=0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "p.bin").write_bytes(sp.program_bytes)
        argv = [str(RTL_BINARY),
                "--program", str(td / "p.bin"),
                "--json-out", str(td / "sum.json"),
                "--max-cycles", str(args.max_cycles)]
        cp = subprocess.run(argv, capture_output=True, text=True,
                            timeout=args.timeout)
        if cp.returncode != 0:
            print("STDERR:", cp.stderr[-2000:])
            raise SystemExit(f"run_program exited {cp.returncode}")
        summary = json.loads((td / "sum.json").read_text())

    total = summary["cycles"]
    bz = summary["busy_cycles"]
    sw = summary["sync_wait_cycles"]
    print(f"status              : {summary['status']}")
    print(f"total cycles        : {total:,}")
    print(f"retired instructions: {summary['retired_instructions']:,}")
    print(f"overlap violation   : {summary.get('forbidden_overlap_violation')}")
    print()
    print("busy_cycles (engine occupancy; engines are serialized so these ~sum):")
    for k, v in bz.items():
        print(f"  {k:9s}: {v:>12,}  ({100.0*v/total:5.1f}%)")
    busy_sum = sum(bz.values())
    print(f"  {'SUM':9s}: {busy_sum:>12,}  ({100.0*busy_sum/total:5.1f}%)")
    print(f"  {'other':9s}: {total-busy_sum:>12,}  ({100.0*(total-busy_sum)/total:5.1f}%)"
          "   (control/fetch/sync/idle)")
    print()
    print("sync_wait_cycles (barrier stalls waiting on async engines):")
    for k, v in sw.items():
        print(f"  {k:9s}: {v:>12,}  ({100.0*v/total:5.1f}%)")
    print()
    print(f"dma bursts/beats    : {summary['dma']['burst_count']:,} / "
          f"{summary['dma']['beat_count']:,}  "
          f"(~{16*summary['dma']['beat_count']/1024:.0f} KiB moved)")


if __name__ == "__main__":
    main()
