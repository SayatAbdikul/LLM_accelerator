#!/usr/bin/env python3
"""Measure the speculative-decoding economics on the real 124M model.

Spec-dec trades ONE 1-token decode step for ONE P-row verify pass that can
confirm several tokens. So the whole lever reduces to two measured numbers:

    r = cycles(P-row verify pass) / cycles(1-token decode step)
    t = tokens the pass actually confirms  (draft-dependent, 1 <= t <= P)

    speedup = t / r,  break-even at t = r

`r` is a property of the hardware and the compiled program — workload-free, and
that is what this tool measures. `t` is a property of the draft and the text;
`speculative_generate` reports it (SpecDecStats). Quoting a tok/s number without
both is how a spec-dec claim turns into fiction, so this prints the break-even
and a speedup table rather than one headline figure.

Both programs are extracted from the SAME bundle (its decode stream is the
ordinary 1-token graph, its prefill stream is the P-row chunk), so `r` has no
cross-build variation in it.

Cycle counts are data-independent here -- engine loop bounds and DMA beat counts
follow tile geometry and valid_kv_len, not values -- so a zeroed KV cache is
fine, exactly as in bench_decode_cycles.py.
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
    REPO_ROOT / "rtl" / "verilator" / "build" / "run_program_synth" / "Vtaccel_top"
)
FREEZE_PTQ_PRESET = "weight_only_int8_quarot"


def _program_binary(bundle, instrs, data_template, data_base):
    from taccel.assembler.assembler import ProgramBinary
    return ProgramBinary(
        instructions=instrs, data=data_template, entry_point=0,
        insn_count=len(instrs) // 8, data_base=data_base,
        input_offset=0, pos_embed_patch_dram_offset=0,
        pos_embed_cls_dram_offset=0, cls_token_dram_offset=0,
        trace_manifest={}, compiler_manifest={},
    )


def build_bins(payload, base_pos: int, p: int, out_dir: Path):
    """Build the P-row verify-pass bin and the 1-token decode bin from one bundle."""
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    scales = build_calibration_scales(payload)
    # The verify pass reaches position base_pos + P - 1, and the decode step we
    # compare against sits at the same last position. The bundle's attention ops
    # are compiled for this KV window; overrun it and the SFU faults
    # FAULT_NO_CONFIG.
    last_pos = base_pos + p - 1
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=last_pos + 1,
        calibration_scales=scales, ptq_preset=FREEZE_PTQ_PRESET,
        batch=1, prefill_tokens=p,
    )
    bundle = tiny.build.bundle
    data_base = int(bundle.data_base)
    prefill_off = int(bundle.prefill_instrs_offset)
    decode_off = int(bundle.decode_instrs_offset)

    runner = HostRunner(bundle, simulator=None)
    data_template = bytes(bundle.materialize(reset_runtime=False)[data_base:])

    # --- the verify pass: P query rows at [base_pos, base_pos + P) ---
    runner._patch_embeddings("prefill", [0] * p,
                             [base_pos + i for i in range(p)])
    runner._patch_kv_bases(base_pos, stream="prefill")
    runner._patch_attention_context("prefill", base_pos, base_pos + p)
    image = bundle.materialize(reset_runtime=False)
    assert bytes(image[data_base:]) == data_template
    verify = bytes(image[prefill_off:decode_off])
    vbin = out_dir / f"verify_p{p}_base{base_pos}.bin"
    vbin.write_bytes(_program_binary(bundle, verify, data_template, data_base).to_bytes())

    # --- the baseline: one ordinary decode step at the same last position ---
    runner._patch_embeddings("decode", [0], [last_pos])
    runner._patch_kv_bases(last_pos)
    runner._patch_decode_attention_context(last_pos)
    image = bundle.materialize(reset_runtime=False)
    assert bytes(image[data_base:]) == data_template
    decode = bytes(image[decode_off:data_base])
    dbin = out_dir / f"decode_p{last_pos}.bin"
    dbin.write_bytes(_program_binary(bundle, decode, data_template, data_base).to_bytes())

    print(f"  verify pass: {len(verify)//8:,} insns   "
          f"decode step: {len(decode)//8:,} insns", flush=True)
    return vbin, dbin


def run_one(rtl: Path, bin_path: Path, max_cycles: int):
    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "s.json"
        cp = subprocess.run(
            [str(rtl), "--program", str(bin_path), "--json-out", str(js),
             "--fast-beats", "--max-cycles", str(max_cycles)],
            capture_output=True, text=True, timeout=14400,
        )
        if not js.exists():
            raise RuntimeError(f"run failed rc={cp.returncode}: {cp.stderr[-400:]}")
        s = json.loads(js.read_text())
        if s.get("status") != "halted" or s.get("fault"):
            raise RuntimeError(f"FAULT: {s.get('status')} {s.get('fault')}")
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--rtl", type=Path, default=DEFAULT_RTL)
    ap.add_argument("--prefill-tokens", type=int, default=16)
    ap.add_argument("--base-pos", type=int, default=496,
                    help="verify pass covers [base_pos, base_pos+P); the decode "
                         "baseline runs at the pass's LAST position so both see "
                         "the same KV window")
    ap.add_argument("--fmax-mhz", type=float, default=34.41)
    ap.add_argument("--max-cycles", type=int, default=200_000_000)
    args = ap.parse_args()

    if not args.rtl.exists():
        print(f"RTL binary missing: {args.rtl}\n"
              f"  build it with: make -C rtl/verilator run_program_synth",
              file=sys.stderr)
        return 2

    import torch  # noqa: deferred heavy import

    p = int(args.prefill_tokens)
    print(f"loading {args.fixture} ...", flush=True)
    payload = torch.load(args.fixture, map_location="cpu")

    td = Path(tempfile.mkdtemp(prefix="specdec_bench_"))
    try:
        print(f"building bins (P={p}, base_pos={args.base_pos}) ...", flush=True)
        vbin, dbin = build_bins(payload, int(args.base_pos), p, td)

        print("running the 1-token decode step ...", flush=True)
        d = run_one(args.rtl, dbin, args.max_cycles)
        dbin.unlink()
        print("running the P-row verify pass ...", flush=True)
        v = run_one(args.rtl, vbin, args.max_cycles)
        vbin.unlink()
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)

    d_cyc, v_cyc = int(d["cycles"]), int(v["cycles"])
    r = v_cyc / d_cyc
    fmax = args.fmax_mhz * 1e6

    def busy(s, k):
        return int(s.get("busy_cycles", {}).get(k, 0))

    print()
    print("=== measured ===")
    print(f"  1-token decode step   {d_cyc:>14,} cyc   "
          f"{fmax / d_cyc:6.3f} tok/s")
    print(f"  {p}-row verify pass     {v_cyc:>14,} cyc")
    print(f"  cost ratio r          {r:>14.4f}   "
          f"(a {p}-row pass costs {r:.2f}x a 1-row step)")
    print(f"  => BREAK-EVEN at {r:.2f} accepted tokens/pass; "
          f"the pass ceiling is {p}")
    print()
    print(f"{'engine':>10} {'decode':>14} {'verify':>14} {'x':>7}")
    for k in ("systolic", "sfu", "helper"):
        dv, vv = busy(d, k), busy(v, k)
        print(f"{k:>10} {dv:>14,} {vv:>14,} {vv / dv if dv else 0:>7.2f}")
    db = int(d.get("dma", {}).get("beat_count", 0))
    vb = int(v.get("dma", {}).get("beat_count", 0))
    print(f"{'dma beats':>10} {db:>14,} {vb:>14,} {vb / db if db else 0:>7.2f}")
    print()
    print("=== speedup vs accepted tokens/pass ===")
    print(f"  {'t':>4} {'tok/s':>8} {'vs 1.0x':>9}")
    for t in range(1, p + 1):
        eff = t / r
        print(f"  {t:>4} {fmax / (v_cyc / t):>8.3f} {eff:>8.2f}x")
    print()
    print("  t is DRAFT-dependent: measure it with SpecDecStats on real text.")
    print("  The adaptive fallback (no candidates -> plain decode step) floors")
    print("  the loss at ~1.0x, so t < r degrades to the baseline, not below it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
