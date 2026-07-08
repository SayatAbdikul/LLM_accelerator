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
      --positions 0,63,255,511 [--fmax-mhz 34.41] [--keep-bins]

The RTL binary must be built with a DRAM_SIZE that fits the bundle
(1<<30 for GPT-2 124M) and should be the mode-1 (SFU_SYNTH_MODE=1)
build for real-chip numbers. --fast-beats (honest-BW) is always passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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


def build_decode_bins(payload, positions, out_dir: Path):
    """Build one standalone decode ProgramBinary per requested position."""
    from taccel.assembler.assembler import ProgramBinary
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

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
    runner = HostRunner(bundle, simulator=None)

    # The data section (weights + zeroed KV) is static across steps; snapshot
    # once. Cycle counts don't depend on KV contents, so zeros are fine.
    image0 = bundle.materialize(reset_runtime=False)
    data_template = bytes(image0[data_base:])

    bins = {}
    for pos in positions:
        runner._patch_embeddings("decode", [0], [int(pos)])
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
        bins[pos] = out
        print(f"  built {out.name}: {out.stat().st_size:,} B "
              f"({len(decode_only) // 8:,} insns)", flush=True)
    return bins


def run_one(rtl: Path, bin_path: Path, max_cycles: int):
    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "s.json"
        cp = subprocess.run(
            [str(rtl), "--program", str(bin_path), "--json-out", str(js),
             "--fast-beats", "--max-cycles", str(max_cycles)],
            capture_output=True, text=True, timeout=9000,
        )
        if not js.exists():
            raise RuntimeError(
                f"run failed rc={cp.returncode}: {cp.stderr[-400:]}")
        return json.loads(js.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--positions", default="0,63,255,511",
                    help="comma-separated decode positions (ctx = pos+1)")
    ap.add_argument("--fmax-mhz", type=float, default=34.41)
    ap.add_argument("--max-cycles", type=int, default=200_000_000)
    ap.add_argument("--keep-bins", action="store_true",
                    help="keep the per-position .bin files next to --out-dir")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    import torch  # noqa: deferred heavy import

    positions = sorted({int(p) for p in args.positions.split(",")})
    if not args.rtl.exists():
        print(f"RTL binary missing: {args.rtl}", file=sys.stderr)
        return 2

    print(f"loading payload {args.fixture} ...", flush=True)
    payload = torch.load(args.fixture, map_location="cpu")

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="bench_decode_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"building decode bins for positions {positions} -> {out_dir}",
          flush=True)
    bins = build_decode_bins(payload, positions, out_dir)

    rows = []
    for pos in positions:
        print(f"running position {pos} (ctx={pos + 1}) ...", flush=True)
        s = run_one(args.rtl, bins[pos], args.max_cycles)
        if s.get("status") != "halted" or s.get("fault"):
            print(f"  FAULT: status={s.get('status')} "
                  f"fault_code={s.get('fault_code')} — row excluded", flush=True)
            continue
        rows.append({
            "position": pos,
            "ctx": pos + 1,
            "cycles": s["cycles"],
            "status": s["status"],
            "retired": s["retired_instructions"],
            "sfu_busy": s["busy_cycles"].get("sfu", 0),
            "sys_busy": s["busy_cycles"].get("systolic", 0),
            "dma_beats": s["dma"]["beat_count"],
            "tok_s": args.fmax_mhz * 1e6 / s["cycles"],
        })
        print(f"  cycles={rows[-1]['cycles']:,}  "
              f"tok/s@{args.fmax_mhz}MHz={rows[-1]['tok_s']:.3f}", flush=True)
        if not args.keep_bins:
            bins[pos].unlink()

    print("\n=== decode-shape benchmark (mode-1 honest-BW, "
          f"fmax {args.fmax_mhz} MHz) ===")
    print(f"{'pos':>5} {'ctx':>5} {'cycles':>13} {'sfu_busy':>12} "
          f"{'sys_busy':>12} {'dma_beats':>12} {'tok/s':>8}")
    for r in rows:
        print(f"{r['position']:>5} {r['ctx']:>5} {r['cycles']:>13,} "
              f"{r['sfu_busy']:>12,} {r['sys_busy']:>12,} "
              f"{r['dma_beats']:>12,} {r['tok_s']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
