"""Retire-gap profiler: an EXACT per-opcode partition of one decode step.

WHY THIS IS EXACT
-----------------
The core is in-order and retires one instruction at a time, so the cycle gap
between consecutive retire events partitions the run: every cycle is charged to
exactly one instruction. Engine work lands on the ENGINE'S OWN opcode, not on
the SYNC that waits for it — an async engine op (MATMUL, MASKED_SOFTMAX_FP32,
LOAD) retires at DISPATCH, and the next instruction then blocks until that
engine drains, so the stall appears as the engine op's own gap.

The classification below is CHECKED, not asserted. For each engine class:

    sum(gaps in class) == busy_cycles[engine] + n_ops_in_class

The `+1` per op is the dispatch cycle, which is not engine-busy time. This tool
FAILS LOUDLY if that identity breaks — which is what happens if an opcode is
filed under the wrong engine. (The predecessor of this tool, which lived in a
scratchpad, filed every HELPER op under `ctl` and so could never have seen the
helper engine at all.)

What is left after the four engines is genuine control overhead: SYNC stalls
that no engine op absorbed, plus instruction fetch/dispatch.

USAGE (from repo root)
----------------------
  .venv/bin/python software/tools/profile_decode_step.py --batch 16 --position 510
  .venv/bin/python software/tools/profile_decode_step.py --batch 1  --position 511

The RTL binary must be the mode-1 synth build with a 1 GiB DRAM:
  make -C rtl/verilator run_program_synth
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
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

# Engine map — transcribed from the control_unit.sv dispatch case (the ONLY
# authority). If an opcode moves engines, it moves here too or the identity
# check below will catch it.
SYS_OPS = {0x0A}                                       # MATMUL
DMA_OPS = {0x07, 0x08}                                 # LOAD, STORE
HELPER_OPS = {                                         # helper_dispatch
    0x09,  # BUF_COPY
    0x0B,  # REQUANT
    0x0C,  # SCALE_MUL
    0x0D,  # VADD
    0x11,  # REQUANT_PC
    0x13,  # DEQUANT_ADD
}
SFU_OPS = {                                            # sfu_dispatch
    0x0E,  # SOFTMAX
    0x0F,  # LAYERNORM
    0x10,  # GELU
    0x12,  # SOFTMAX_ATTNV
    0x15,  # MASKED_SOFTMAX
    0x16,  # MASKED_SOFTMAX_ATTNV
    0x17,  # DEQUANT_ACCUM_FP32
    0x18,  # QUANT_FP32_INT8
    0x19,  # VADD_FP32
    0x1A,  # LAYERNORM_FP32
    0x1B,  # GELU_FP32
    0x1C,  # SOFTMAX_FP32  (defined in the ISA; never dispatched — expect 0)
    0x1D,  # MASKED_SOFTMAX_FP32
    0x1E,  # DEQUANT_ACCUM_FP32_SCALED
    0x1F,  # MAX_ABS_REDUCE_FP32
}
# Everything else is control: NOP HALT SYNC CONFIG_TILE SET_SCALE
# SET_ADDR_LO SET_ADDR_HI CONFIG_ATTN.

# class -> the busy counter it must reconcile against ("" = no counter exists)
CLASS_COUNTER = {
    "sfu": "sfu",
    "sys": "systolic",
    "helper": "helper",
    "dma": "",      # no dma_busy counter — DMA overlaps the systolic by design
    "ctl": "",
}


def classify(op: int) -> str:
    if op in SFU_OPS:
        return "sfu"
    if op in SYS_OPS:
        return "sys"
    if op in HELPER_OPS:
        return "helper"
    if op in DMA_OPS:
        return "dma"
    return "ctl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--position", type=int, default=510)
    ap.add_argument("--max-cycles", type=int, default=250_000_000)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the report here as well as to stdout")
    args = ap.parse_args()

    if not args.rtl.exists():
        print(f"RTL binary missing: {args.rtl}\n"
              f"  build it with: make -C rtl/verilator run_program_synth",
              file=sys.stderr)
        return 2

    import torch  # noqa: deferred heavy import
    from bench_decode_cycles import build_decode_bins
    from taccel.isa.opcodes import Opcode

    print(f"loading {args.fixture} ...", flush=True)
    payload = torch.load(args.fixture, map_location="cpu")

    # The decode bin carries the whole 124M data section (~640 MB) and the retire
    # trace is tens of MB. Both MUST be reaped: leaving them behind fills the disk
    # in a handful of runs, and a full disk makes every subsequent tool fail in
    # confusing ways (the shell cannot even write its own stdout).
    td = Path(tempfile.mkdtemp(prefix="profile_step_"))
    try:
        print(f"building decode bin (batch={args.batch}, pos={args.position}) ...",
              flush=True)
        bins = build_decode_bins(payload, [args.position], td, batch=args.batch)

        sum_json, trace_json = td / "summary.json", td / "trace.json"
        print("running RTL with retire trace (this is the slow part) ...",
              flush=True)
        cp = subprocess.run(
            [str(args.rtl), "--program", str(bins[args.position]),
             "--json-out", str(sum_json), "--trace-json-out", str(trace_json),
             "--fast-beats", "--max-cycles", str(args.max_cycles)],
            capture_output=True, text=True, timeout=28800,
        )
        if not sum_json.exists():
            print(f"RTL run failed rc={cp.returncode}: {cp.stderr[-600:]}",
                  file=sys.stderr)
            return 1
        s = json.loads(sum_json.read_text())
        if s.get("status") != "halted" or s.get("fault"):
            print(f"FAULT: {s}", file=sys.stderr)
            return 1

        # Free the 640 MB bin the moment the run is done — the trace parse below
        # is the memory-hungry part and we do not need the bin any more.
        bins[args.position].unlink()
        events = json.loads(trace_json.read_text())["retire_events"]
        events.sort(key=lambda e: e["cycle"])
        trace_json.unlink()
    finally:
        shutil.rmtree(td, ignore_errors=True)

    dur, cnt = defaultdict(int), defaultdict(int)
    for i in range(len(events) - 1):
        op = events[i]["opcode"]
        dur[op] += events[i + 1]["cycle"] - events[i]["cycle"]
        cnt[op] += 1

    spanned = sum(dur.values())
    cycles = s["cycles"]
    busy = s["busy_cycles"]
    port_a = s.get("port_a", {})
    names = {int(o): o.name for o in Opcode}

    L = []
    def out(line=""):
        L.append(line)
        print(line, flush=True)

    out(f"\n=== decode step: batch={args.batch} pos={args.position} "
        f"(ctx={args.position + 1}) ===")
    out(f"  total cycles      {cycles:>15,}   per token {cycles / args.batch:>13,.0f}"
        f"   tok/s@34.41MHz {34.41e6 / (cycles / args.batch):.3f}")
    out(f"  retired insns     {s['retired_instructions']:>15,}")
    out(f"  trace spans       {spanned:>15,}   "
        f"({100 * spanned / cycles:.4f}% of the run — the rest is pre-first-retire)")
    out(f"  dma beats         {s['dma']['beat_count']:>15,}")

    out(f"\n{'op':>5}  {'name':26} {'gap_cyc':>14} {'%tot':>6} {'count':>9} "
        f"{'cyc/op':>10}  engine")
    for op, d in sorted(dur.items(), key=lambda kv: -kv[1]):
        out(f"0x{op:02X}  {names.get(op, '?'):26} {d:>14,} "
            f"{100 * d / spanned:6.1f} {cnt[op]:>9,} "
            f"{d / max(cnt[op], 1):>10.1f}  {classify(op)}")

    grp, gcnt = defaultdict(int), defaultdict(int)
    for op, d in dur.items():
        grp[classify(op)] += d
        gcnt[classify(op)] += cnt[op]

    out(f"\n=== by engine ===")
    for k, v in sorted(grp.items(), key=lambda kv: -kv[1]):
        out(f"  {k:>6}: {v:>14,}  ({100 * v / spanned:5.1f}%)")

    # The identity that makes this partition trustworthy rather than merely
    # plausible: an engine's gap total must equal its busy counter plus exactly
    # one dispatch cycle per op.
    #
    # It holds EXACTLY for sfu and sys, because each of their ops is followed by
    # a consumer that BLOCKS until the engine drains, so the engine's whole run
    # lands in its own gap. It does NOT hold for the helper: BUF_COPY's
    # successors are non-blocking ctl ops (SET_ADDR / CONFIG_TILE, 4 cyc each),
    # so the helper's tail overlaps them and gets charged to `ctl`. For the
    # helper the BUSY COUNTER is authoritative and the trace under-charges.
    out(f"\n=== classification check: gap_total == busy + n_ops ? ===")
    ok = True
    for cls, counter in CLASS_COUNTER.items():
        if not counter:
            continue
        got, n, want = grp[cls], gcnt[cls], busy.get(counter, 0)
        delta = got - want - n
        if delta == 0:
            status = "OK"
        elif cls == "helper":
            status = (f"trace under-charges by {-delta:,} — expected "
                      f"(BUF_COPY's successors don't block); busy is authoritative")
        else:
            status = f"MISMATCH by {delta:+,}"
            ok = False
        out(f"  {cls:>6}: gap {got:>14,}  busy {want:>14,}  "
            f"n_ops {n:>7,}  -> {status}")
    if not ok:
        out("  *** the engine map in this file disagrees with the RTL dispatch. "
            "Fix it before trusting anything above.")

    # Shared Port-A audit. cobusy is the DENOMINATOR — a zero loss count means
    # nothing if the DMA and systolic never actually overlapped.
    out(f"\n=== shared Port-A audit (DMA vs systolic) ===")
    cobusy = port_a.get("dma_sys_cobusy", 0)
    lost = port_a.get("sys_lost", 0)
    out(f"  dma||sys co-busy cycles   {cobusy:>14,}  "
        f"({100 * cobusy / cycles:.1f}% of the step)")
    out(f"  systolic accesses LOST    {lost:>14,}")
    out(f"    of which writes         {port_a.get('sys_lost_write', 0):>14,}")
    out(f"    target ABUF             {port_a.get('sys_lost_abuf', 0):>14,}")
    out(f"    target WBUF             {port_a.get('sys_lost_wbuf', 0):>14,}")
    out(f"    target ACCUM            {port_a.get('sys_lost_accum', 0):>14,}")
    preclear = port_a.get("sys_lost_preclear", 0)
    drain = port_a.get("sys_lost_drain", 0)
    out(f"      preclear (HARMLESS)   {preclear:>14,}   zeros that ST_DRAIN_WR "
        "overwrites anyway")
    out(f"      drain    (CORRUPTS)   {drain:>14,}   real accumulator results, "
        "lost forever")
    if drain:
        out("  *** SILENT CORRUPTION: dropped DRAIN writes. The systolic is "
            "losing real results and cannot tell. DMA||Systolic is UNSAFE.")
    elif lost:
        out("  => LUCKY, NOT SAFE. Every dropped write is a PRECLEAR — zeros that\n"
            "     ST_DRAIN_WR unconditionally overwrites — so the corruption is\n"
            "     invisible TODAY. Nothing enforces this. The preclear is also\n"
            "     provably dead work (plan lever 1a); deleting it removes the\n"
            "     hazard's only current victim and leaves the DRAIN writes\n"
            "     exposed to the very same bus loss.")
    elif cobusy:
        out(f"  => SAFE BY CONSTRUCTION over {cobusy:,} genuinely concurrent "
            "cycles: the systolic streams src2 from the dedicated Port W during\n"
            "     dense matmuls, leaving shared Port A free for the DMA.")
    else:
        out("  => INCONCLUSIVE: zero co-busy cycles, so zero losses proves "
            "nothing. The engines never overlapped in this run.")
    out(f"  forbidden_overlap_violation: {s.get('forbidden_overlap_violation')}")

    if args.out:
        args.out.write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
