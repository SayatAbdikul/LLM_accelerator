"""Decode-shape cycle benchmark: mode-1 cycles/token at chosen context depths.

The standing 1-token prefill benchmark (ctx~0) understates real decode cost:
attention QK^T/AV work and KV traffic grow with context and don't amortize.
This tool measures the true decode shape by building the decoder
ProgramBundle, patching its decode stream to a target position p (token/pos
embedding, KV base, CONFIG_ATTN query_row_base=p / valid_kv_len=p+1 — the
exact HostRunner.run_decode_step patch sequence), and running that single
step on the RTL simulator.

Cycle counts are data-independent (engine loop bounds and DMA beat counts
depend on tile geometry and valid_kv_len, not on values), so the KV region
content is irrelevant for measurement and each position runs as one
standalone step — no need to simulate the first p steps.

Usage (from repo root):
  .venv/bin/python software/tools/bench_decode_cycles.py \
      --rtl rtl/verilator/build/run_program_synth/Vtaccel_top \
      --positions 0,63,255,511 [--fmax-mhz USER_ASSUMED_MHZ] [--keep-bins]

The RTL binary must be built with a DRAM_SIZE that fits the bundle
(1<<30 for GPT-2 124M) and must report both SFU_SYNTH_MODE=1 and
HELPER_SYNTH_MODE=1. These are simulated cycles, not measured silicon timing.

MEMORY MODEL: --beat-interval N sets the read-channel beat spacing in core
cycles; --fast-beats is also passed. Writes are NOT throttled by this option.
Read/write channels operate independently, so this is a read-bandwidth
sensitivity experiment, not a shared fixed-total-GB/s memory controller.
Tokens/second are omitted unless an explicit assumed clock is supplied.
Logical decoder MACs are a canonical algorithm estimate, excluding PTQ
transforms, SFU/helpers and padding; capacity ratios are not measured ISA
operation counts or array utilization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "software"))

DEFAULT_FIXTURE = (
    REPO_ROOT / "software" / "tests" / "fixtures" / "generated" /
    "gpt2_converted_nanogpt.pt"
)
DEFAULT_RTL = (
    REPO_ROOT / "rtl" / "verilator" / "build" / "run_program_synth" /
    "Vtaccel_top"
)
FREEZE_PTQ_PRESET = "weight_only_int8_quarot"


def build_decode_bins(payload, positions, out_dir: Path, batch: int = 1,
                      compiler_options=None):
    """Build one standalone decode ProgramBinary per requested position.

    With ``batch > 1`` the decode stream is the lockstep batched graph (all
    streams at the same position); the reported cycles are for one batched
    step, so per-token cost is ``cycles / batch``.
    """
    from taccel.assembler.assembler import ProgramBinary
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.host_runner import BundlePatcher
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
    from tools.benchmark_support import bundle_ptq_transform_manifest, canonical_hash, logical_decoder_work, sha256_file

    started = time.monotonic()
    if not positions or min(positions) < 0:
        raise ValueError("decode positions must be a non-empty sequence of non-negative integers")

    scales = build_calibration_scales(payload)
    # The decode stream's attention ops are compiled for the bundle's KV
    # window: smoke_decode_steps must cover the highest position patched in,
    # or CONFIG_ATTN's valid_kv_len exceeds the compiled shape and the SFU
    # faults FAULT_NO_CONFIG (attn-context check). Sizing by max(positions)
    # also means the measured cycles reflect a program compiled for that
    # context budget — the deployment-relevant shape.
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=max(positions) + 1,
        calibration_scales=scales, ptq_preset=FREEZE_PTQ_PRESET,
        batch=batch,
        compiler_options=compiler_options,
    )
    bundle = tiny.build.bundle

    data_base = int(bundle.data_base)
    decode_instrs_offset = int(bundle.decode_instrs_offset)
    kv_step_bytes = int(bundle.kv_step_bytes)
    kv_cache_size = int(bundle.kv_cache_size_bytes)
    block_size = int(tiny.config.max_seq_len)

    max_pos = max(positions)
    if max_pos >= block_size:
        raise ValueError(f"position {max_pos} >= block_size {block_size}")
    if kv_step_bytes * (max_pos + 1) > kv_cache_size:
        raise ValueError(
            f"KV cache ({kv_cache_size} B) too small for position {max_pos} "
            f"({kv_step_bytes} B/step) — bundle KV is not block_size-sized"
        )

    # Patch driver: reuse HostRunner's patch helpers without simulating.
    runner = BundlePatcher(bundle)

    # The data section (weights + zeroed KV) is static across steps; snapshot
    # once. Cycle counts don't depend on KV contents, so zeros are fine.
    image0 = bundle.materialize(reset_runtime=False)
    data_template = bytes(image0[data_base:])

    bins = {}
    for pos in positions:
        runner._patch_embeddings("decode", [0] * batch, [int(pos)] * batch)
        runner._patch_kv_bases(int(pos))
        runner._patch_decode_attention_context(int(pos))
        image = bundle.materialize(reset_runtime=False)
        decode_only = bytes(image[decode_instrs_offset:data_base])
        assert bytes(image[data_base:]) == data_template, (
            "runtime patches must not touch the data region"
        )
        pb = ProgramBinary(
            instructions=decode_only,
            data=data_template,
            entry_point=0,                # run_program always starts at PC 0
            insn_count=len(decode_only) // 8,
            data_base=data_base,          # keeps absolute DMA addresses valid
            input_offset=0, pos_embed_patch_dram_offset=0,
            pos_embed_cls_dram_offset=0, cls_token_dram_offset=0,
            trace_manifest={}, compiler_manifest={},
        )
        out = out_dir / f"decode_p{pos}.bin"
        out.write_bytes(pb.to_bytes())
        manifest = {
            "schema_version": 2, "position": pos, "batch": batch,
            "program_sha256": sha256_file(out),
            "calibration_sha256": canonical_hash(scales),
            "ptq_transforms": bundle_ptq_transform_manifest(FREEZE_PTQ_PRESET),
            "logits_offset": int(bundle.decode_logits_offset),
            "logits_size": int(bundle.logits_row_bytes) * batch,
            "logits_row_bytes": int(bundle.logits_row_bytes),
            "vocab_size": int(tiny.config.vocab_size),
            "program_bytes": out.stat().st_size,
            "instruction_count": len(decode_only) // 8,
            "compile_elapsed_s": time.monotonic() - started,
            "compiler_metrics": {
                "prefill": tiny.build.prefill_codegen.compilation_metrics(),
                "decode": tiny.build.decode_codegen.compilation_metrics(),
            },
            "logical_decoder_work": logical_decoder_work(
                batch=batch, layers=tiny.config.n_layer, d_model=tiny.config.d_model,
                mlp_dim=tiny.config.mlp_dim, vocab_size=tiny.config.vocab_size,
                position=pos),
        }
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        bins[pos] = out
        print(f"  built {out.name}: {out.stat().st_size:,} B "
              f"({len(decode_only) // 8:,} insns)", flush=True)
    return bins


def run_one(rtl: Path, bin_path: Path, max_cycles: int, *, beat_interval: int = 1):
    from tools.benchmark_support import decoder_logits_metrics, logical_decoder_work, validate_runner_summary

    layout = json.loads(bin_path.with_suffix(".manifest.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "s.json"
        dump = Path(td) / "logits.raw"
        cp = subprocess.run(
            [str(rtl.resolve()), "--program", str(bin_path), "--json-out", str(js),
             "--fast-beats", "--beat-interval", str(beat_interval),
             "--max-cycles", str(max_cycles),
             "--dram-dump-offset", str(layout["logits_offset"]),
             "--dram-dump-size", str(layout["logits_size"]),
             "--dram-dump-out", str(dump)],
            capture_output=True, text=True, timeout=9000,
        )
        if cp.returncode != 0 or not js.exists():
            raise RuntimeError(
                f"run failed rc={cp.returncode}: {cp.stderr[-400:]}")
        summary = json.loads(js.read_text())
        validate_runner_summary(summary)
        summary["decoder_logits"] = decoder_logits_metrics(
            dump.read_bytes(), rows=layout["batch"], vocab_size=layout["vocab_size"],
            row_bytes=layout["logits_row_bytes"])
        summary["program_manifest"] = layout
        if "logical_decoder_work" in layout:
            summary["logical_decoder_work"] = logical_decoder_work(
                **layout["logical_decoder_work"]["geometry"], summary=summary)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--positions", default="0,63,255,511",
                    help="comma-separated decode positions (ctx = pos+1)")
    ap.add_argument("--fmax-mhz", type=float, default=None,
                    help="optional assumed clock; no frequency is inferred from historical results")
    ap.add_argument("--beat-interval", type=int, choices=(1, 2, 4), default=1)
    ap.add_argument("--compiler-optimizations", default="",
                    help="comma-separated opt-in CompilerOptions; empty is corrected baseline")
    ap.add_argument("--require-finite-logits", action="store_true",
                    help="fail if any actual FP16 DRAM decoder logits are non-finite")
    ap.add_argument("--batch", type=int, default=1,
                    help="lockstep batched decode width (1, 16, or 32); "
                         "per-token cost = cycles / batch")
    ap.add_argument("--max-cycles", type=int, default=200_000_000)
    ap.add_argument("--keep-bins", action="store_true",
                    help="keep the per-position .bin files next to --out-dir")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    import torch  # noqa: deferred heavy import

    positions = sorted({int(p) for p in args.positions.split(",")})
    from tools.benchmark_support import (
        benchmark_manifest, check_runner_freshness, compiler_options_from_names,
    )
    options = compiler_options_from_names(args.compiler_optimizations)
    if not positions or min(positions) < 0 or args.batch not in (1, 16, 32):
        ap.error("non-negative positions and batch 1, 16, or 32 are required")
    if args.fmax_mhz is not None and not (0 < args.fmax_mhz < float("inf")):
        ap.error("--fmax-mhz must be finite and positive")
    if not args.rtl.exists():
        print(f"RTL binary missing: {args.rtl}", file=sys.stderr)
        return 2
    check_runner_freshness(REPO_ROOT, args.rtl)

    print(f"loading payload {args.fixture} ...", flush=True)
    payload = torch.load(args.fixture, map_location="cpu")

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="bench_decode_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"building decode bins (batch={args.batch}) for positions "
          f"{positions} -> {out_dir}", flush=True)
    manifest = benchmark_manifest(REPO_ROOT, args.fixture, args.rtl,
        batch=args.batch, positions=positions, beat_interval=args.beat_interval,
        fmax_mhz=args.fmax_mhz, compiler_options=asdict(options), ptq_preset=FREEZE_PTQ_PRESET)
    bins = build_decode_bins(payload, positions, out_dir, batch=args.batch, compiler_options=options)

    rows = []
    for pos in positions:
        print(f"running position {pos} (ctx={pos + 1}) ...", flush=True)
        s = run_one(args.rtl, bins[pos], args.max_cycles, beat_interval=args.beat_interval)
        if s.get("status") != "halted" or s.get("fault"):
            print(f"  FAULT: status={s.get('status')} "
                  f"fault_code={s.get('fault_code')} — row excluded", flush=True)
            return 1
        if args.require_finite_logits and not s["decoder_logits"]["all_finite"]:
            raise ValueError(f"non-finite FP16 decoder logits at position {pos}")
        per_tok = s["cycles"] / args.batch
        busy = s["busy_cycles"]
        wait = s.get("sync_wait_cycles", {})
        port_a = s.get("port_a", {})
        cycles = s["cycles"]
        # helper / sfu / systolic are MUTUALLY EXCLUSIVE (the forbidden-overlap
        # invariant, taccel_top.sv), so they partition the step cleanly. What is
        # left is exposed DMA + control overhead — and `sync_wait_cycles.dma` is
        # the control unit stalled on a SYNC waiting for the DMA, i.e. exactly
        # the DMA that failed to hide. The residual is fetch/dispatch/decode.
        other = cycles - busy.get("helper", 0) - busy.get("sfu", 0) - busy.get("systolic", 0)
        rows.append({
            "position": pos,
            "ctx": pos + 1,
            "cycles": cycles,
            "per_tok_cycles": per_tok,
            "status": s["status"],
            "retired": s["retired_instructions"],
            "sfu_busy": busy.get("sfu", 0),
            "sys_busy": busy.get("systolic", 0),
            "helper_busy": busy.get("helper", 0),
            "other": other,
            "wait_dma": wait.get("dma", 0),
            "wait_sys": wait.get("systolic", 0),
            "wait_sfu": wait.get("sfu", 0),
            "ctl": other - wait.get("dma", 0),
            "dma_beats": s["dma"]["beat_count"],
            "cobusy": port_a.get("dma_sys_cobusy", 0),
            "porta_lost": port_a.get("sys_lost", 0),
            "porta_lost_wr": port_a.get("sys_lost_write", 0),
            "overlap_violation": s.get("forbidden_overlap_violation", False),
            "tok_s": args.fmax_mhz * 1e6 / per_tok if args.fmax_mhz is not None else None,
            "configuration": s["configuration"],
            "decoder_logits": s["decoder_logits"],
            "program_manifest": s["program_manifest"],
            "logical_decoder_work": s["logical_decoder_work"],
            "runner_summary": s,
        })
        r = rows[-1]
        throughput = (f"  assumed tok/s@{args.fmax_mhz}MHz={r['tok_s']:.3f}"
                      if r["tok_s"] is not None else "")
        print(f"  step_cycles={r['cycles']:,}  per_tok={per_tok:,.0f}{throughput}", flush=True)
        if r["porta_lost"]:
            print(f"  *** PORT-A HAZARD: {r['porta_lost']:,} systolic accesses "
                  f"SILENTLY DROPPED ({r['porta_lost_wr']:,} of them writes) — "
                  f"DMA||Systolic is corrupting state. See taccel_top.sv.",
                  flush=True)
        if r["overlap_violation"]:
            print("  *** forbidden_overlap_violation SET", flush=True)
        if not args.keep_bins:
            bins[pos].unlink()

    print("\n=== decode-shape benchmark (mode-1 honest-BW, "
          f"fmax {args.fmax_mhz} MHz, batch={args.batch}) ===")
    print(f"{'pos':>5} {'ctx':>5} {'step_cyc':>13} {'per_tok_cyc':>13} "
          f"{'sfu_busy':>12} {'sys_busy':>12} {'dma_beats':>12} {'tok/s':>8}")
    for r in rows:
        print(f"{r['position']:>5} {r['ctx']:>5} {r['cycles']:>13,} "
              f"{r['per_tok_cycles']:>13,.0f} "
              f"{r['sfu_busy']:>12,} {r['sys_busy']:>12,} "
              f"{r['dma_beats']:>12,} {r['tok_s'] if r['tok_s'] is not None else 'n/a':>8}")

    # The cycle budget, fully partitioned. `helper` is a THIRD disjoint engine
    # that this tool used to drop on the floor, hiding it inside "other".
    print(f"\n=== where the step goes (cycles, % of step) ===")
    print(f"{'pos':>5} {'sys':>21} {'sfu':>21} {'helper':>21} "
          f"{'exposed-DMA':>21} {'ctl':>21}")
    for r in rows:
        c = r["cycles"]
        def f(v):
            return f"{v:>13,} {100 * v / c:5.1f}%"
        print(f"{r['position']:>5} {f(r['sys_busy'])} {f(r['sfu_busy'])} "
              f"{f(r['helper_busy'])} {f(r['wait_dma'])} {f(r['ctl'])}")

    # Port-A arbitration audit. cobusy is the DENOMINATOR: zero losses mean
    # nothing if the DMA and systolic never actually ran concurrently.
    print(f"\n=== shared Port-A audit (DMA vs systolic) ===")
    print(f"{'pos':>5} {'dma||sys cobusy':>18} {'sys accesses LOST':>19} "
          f"{'of which writes':>17}")
    for r in rows:
        flag = "  <-- CORRUPTION" if r["porta_lost"] else ""
        print(f"{r['position']:>5} {r['cobusy']:>18,} {r['porta_lost']:>19,} "
              f"{r['porta_lost_wr']:>17,}{flag}")
    if rows and not any(r["cobusy"] for r in rows):
        print("  WARNING: zero co-busy cycles — the DMA and systolic never "
              "overlapped, so a zero loss count proves NOTHING here.")
    report = out_dir / "report.json"
    report.write_text(json.dumps({"manifest": manifest, "rows": rows}, indent=2, allow_nan=False) + "\n")
    print(f"Saved reproducible shape-only cycle report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
