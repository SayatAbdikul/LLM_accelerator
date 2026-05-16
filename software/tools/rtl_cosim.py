"""P6b: serialize a frozen decoder-bundle PREFILL stream into a single-shot
run_program.cpp ProgramBinary, for RTL-vs-golden e2e byte-match.

Why this is valid (measured, not assumed — see task #105):
  * The prefill stream's only runtime patch sites are `token_embed` /
    `pos_embed` (kv_base / config_attn are decode-only). After patching
    embeddings for a single fixed token at position 0, the prefill DRAM
    image is fully static -> a single-shot ProgramBinary faithfully
    reproduces it.
  * `bundle.prefill_pc == prefill_instrs_offset // 8 == 0`, so the codegen
    stream-local trace-manifest PCs equal the executed PCs (no rebasing).
  * `bundle.materialize(reset_runtime=False)` is the runtime-patched flat
    DRAM image; splitting it at `data_base` yields ProgramBinary
    instructions/data that `run_program.cpp::to_dram_image()` reassembles
    byte-identically.

This module is import-safe (no work at import) so the conformance test can
reuse `serialize_prefill_bundle`.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

FREEZE_PTQ_PRESET = "weight_only_int8_quarot"


@dataclass
class SerializedPrefill:
    program_bytes: bytes
    trace_manifest: Dict[int, list]
    data_base: int
    insn_count: int
    instr_len: int
    data_len: int
    entry_pc: int
    token_id: int


def _load_tiny_payload():
    """Load the frozen tiny nanoGPT fixture payload (same source as the
    determinism / SHA-pin legs of test_compare_rtl_golden.py)."""
    import torch  # local import: torch is heavy and test-only

    tool_path = REPO_ROOT / "tools" / "train_tiny_fixture.py"
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.DEFAULT_FIXTURE.exists():
        raise FileNotFoundError(
            f"tiny nanoGPT fixture missing: {module.DEFAULT_FIXTURE}\n"
            "run: PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )
    return torch.load(module.DEFAULT_FIXTURE, map_location="cpu")


def serialize_prefill_bundle(
    *,
    token_id: int = 0,
    smoke_decode_steps: int = 1,
    embed_trace_manifest: bool = False,
    payload: Optional[Dict[str, Any]] = None,
) -> SerializedPrefill:
    """Build the frozen weight_only_int8_quarot bundle, patch its prefill
    stream for `token_id` at position 0, and serialize the patched flat DRAM
    image as a single-shot ProgramBinary.
    """
    from taccel.assembler.assembler import ProgramBinary
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    if payload is None:
        payload = _load_tiny_payload()

    scales = build_calibration_scales(payload)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=smoke_decode_steps,
        calibration_scales=scales,
        ptq_preset=FREEZE_PTQ_PRESET,
    )
    build = tiny.build
    bundle = build.bundle

    # Prefill stream-local trace manifest (PC -> [capture events]); codegen
    # already emits this. prefill_pc == 0 so PCs need no rebasing.
    prefill_manifest = dict(build.prefill_codegen.trace_manifest)

    # Patch only the prefill embedding sites for one fixed token @ pos 0.
    # This is exactly what HostRunner.run_prefill does before executing
    # golden; reusing its (tested) patch logic keeps RTL == golden inputs.
    runner = HostRunner(bundle)
    runner._patch_embeddings("prefill", [int(token_id)], [0])

    # Runtime-patched flat DRAM image; split at data_base.
    image = bundle.materialize(reset_runtime=False)
    data_base = int(bundle.data_base)
    assert data_base % 16 == 0, data_base
    assert len(image) == int(bundle.required_dram_bytes), (
        len(image), bundle.required_dram_bytes
    )
    instructions = bytes(image[:data_base])
    data = bytes(image[data_base:])
    assert len(instructions) % 8 == 0, len(instructions)

    pb = ProgramBinary(
        instructions=instructions,
        data=data,
        entry_point=int(bundle.prefill_pc),  # == 0 for prefill
        insn_count=int(bundle.insn_count),
        data_base=data_base,
        input_offset=0,            # GPT-2: no DeiT-style runtime input region
        pos_embed_patch_dram_offset=0,
        pos_embed_cls_dram_offset=0,
        cls_token_dram_offset=0,
        trace_manifest=(prefill_manifest if embed_trace_manifest else {}),
        compiler_manifest={},
    )
    return SerializedPrefill(
        program_bytes=pb.to_bytes(),
        trace_manifest=prefill_manifest,
        data_base=data_base,
        insn_count=int(bundle.insn_count),
        instr_len=len(instructions),
        data_len=len(data),
        entry_pc=int(bundle.prefill_pc),
        token_id=int(token_id),
    )


# ---------------------------------------------------------------------------
# RTL-vs-golden e2e byte-match (freeze §4.5 / §7)
# ---------------------------------------------------------------------------
RTL_BINARY = REPO_ROOT.parent / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"

# freeze §7 per-op-class fp16-ULP conformance bands. Default for any opcode
# not listed (incl. non-gen-2 producers and int8/int32 storage) is 0 ULP
# (bit-exact) — that is the freeze §7 mandate; only gelu_new is characterized.
_GEN2_OPNAMES = {
    0x17: "DEQUANT_ACCUM_FP32", 0x18: "QUANT_FP32_INT8", 0x19: "VADD_FP32",
    0x1A: "LAYERNORM_FP32", 0x1B: "GELU_FP32", 0x1C: "SOFTMAX_FP32(reserved)",
    0x1D: "MASKED_SOFTMAX_FP32", 0x1E: "DEQUANT_ACCUM_FP32_SCALED",
    0x1F: "MAX_ABS_REDUCE_FP32",
}
_OPCODE_ULP_BAND = {0x1B: 3}  # gelu_new ≤3 ULP (freeze §7); everything else 0


def emit_snapshot_csv(trace_manifest: Dict[int, list]) -> str:
    """trace_manifest -> run_program.cpp load_snapshot_requests CSV (16 fields,
    field order pinned against rtl/verilator/run_program.cpp)."""
    lines = []
    for pc in sorted(trace_manifest):
        for event_index, ev in enumerate(trace_manifest[pc]):
            lines.append(",".join(str(x) for x in (
                int(pc), int(event_index), ev["node_name"], int(ev["buf_id"]),
                int(ev["offset_units"]), int(ev["mem_rows"]), int(ev["mem_cols"]),
                int(ev["logical_rows"]), int(ev["logical_cols"]),
                int(ev["full_rows"]), int(ev["full_cols"]), int(ev.get("row_start", 0)),
                ev["dtype"], repr(float(ev["scale"])),
                str(ev.get("source", "architectural")),
                str(ev.get("capture_phase", "retire_cycle")),
            )))
    return "\n".join(lines) + "\n"


def capture_golden(*, token_id: int = 0, smoke_decode_steps: int = 1,
                   payload: Optional[Dict[str, Any]] = None,
                   exclude_nodes: Optional[set] = None):
    """Run the frozen prefill on the pinned golden HostRunner with the codegen
    trace manifest attached; return (golden_tensors, serialized_prefill).

    `exclude_nodes` prunes those node names from the manifest BEFORE golden
    runs. Required for `lm_head` on large-vocab models (GPT-2 124M: vocab
    50257 -> a 16x50272 fp16 logits tile = 1.6 MB that does NOT live in the
    128 KB ABUF; it is written to the logits DRAM region, so an ABUF-tile
    snapshot is structurally invalid and golden's read_fp16_tile OOBs).
    Final logits are the §4.5 arbiter and are compared via the logits DRAM
    region separately, not as an ABUF tile."""
    import numpy as np  # noqa: F401  (kept local; numpy is test-only here)
    from taccel.assembler.assembler import ProgramBinary
    from taccel.golden_model.simulator import Simulator
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    if payload is None:
        payload = _load_tiny_payload()
    scales = build_calibration_scales(payload)
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=smoke_decode_steps,
        calibration_scales=scales, ptq_preset=FREEZE_PTQ_PRESET,
    )
    build = tiny.build
    bundle = build.bundle
    prefill_manifest = dict(build.prefill_codegen.trace_manifest)
    if exclude_nodes:
        prefill_manifest = {
            pc: kept
            for pc, evs in prefill_manifest.items()
            if (kept := [e for e in evs if e["node_name"] not in exclude_nodes])
        }

    sim = Simulator()
    runner = HostRunner(bundle, simulator=sim)
    # load_bundle hardcodes trace_manifest={}; inject AFTER it, then
    # enable_trace. run_prefill sees sim.bundle is bundle -> no reload, and
    # _reset_volatile_execution_state preserves trace_manifest/trace_enabled.
    sim.load_bundle(bundle)
    sim.trace_manifest = prefill_manifest
    sim.enable_trace()
    runner.run_prefill([int(token_id)])
    golden_tensors = sim.get_trace_payload()["tensors"]

    image = bundle.materialize(reset_runtime=False)
    data_base = int(bundle.data_base)
    pb = ProgramBinary(
        instructions=bytes(image[:data_base]), data=bytes(image[data_base:]),
        entry_point=int(bundle.prefill_pc), insn_count=int(bundle.insn_count),
        data_base=data_base, input_offset=0, pos_embed_patch_dram_offset=0,
        pos_embed_cls_dram_offset=0, cls_token_dram_offset=0,
        trace_manifest={}, compiler_manifest={},
    )
    sp = SerializedPrefill(
        program_bytes=pb.to_bytes(), trace_manifest=prefill_manifest,
        data_base=data_base, insn_count=int(bundle.insn_count),
        instr_len=data_base, data_len=len(image) - data_base,
        entry_pc=int(bundle.prefill_pc), token_id=int(token_id),
    )
    return golden_tensors, sp, bytes(image[:data_base])


def run_rtl(program_bytes: bytes, csv_text: str, *, max_cycles: int = 5_000_000,
            timeout_s: int = 1800):
    """Run the patched prefill on Verilator run_program with snapshot capture;
    return (summary, rtl_entries, rtl_data_bytes). timeout_s default 30min
    (124M 1-tok prefill measured ~292s clean; headroom for divergent runs)."""
    import json
    import subprocess
    import tempfile

    if not RTL_BINARY.exists():
        raise FileNotFoundError(
            f"run_program not built: {RTL_BINARY}\n"
            "build: make -C rtl/verilator run_program"
        )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "p.bin").write_bytes(program_bytes)
        (td / "snap.csv").write_text(csv_text)
        cp = subprocess.run(
            [str(RTL_BINARY),
             "--program", str(td / "p.bin"),
             "--json-out", str(td / "sum.json"),
             "--snapshot-request", str(td / "snap.csv"),
             "--snapshot-manifest-out", str(td / "man.json"),
             "--snapshot-data-out", str(td / "data.bin"),
             "--max-cycles", str(max_cycles)],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if cp.returncode != 0:
            raise RuntimeError(
                f"run_program exited {cp.returncode}\n{cp.stderr[-2000:]}"
            )
        summary = json.loads((td / "sum.json").read_text())
        manifest = json.loads((td / "man.json").read_text())
        data = (td / "data.bin").read_bytes()
    return summary, manifest["entries"], data


def _dequant_fragment(dtype: str, scale: float, raw: bytes,
                      logical_rows: int, logical_cols: int,
                      mem_rows: int, mem_cols: int):
    """Replicate golden simulator._snapshot_traced_tensors dequant exactly so
    any RTL-vs-golden delta is a real datapath/integration bug, not a
    comparison artifact. Returns (fp32_logical, fp16_bits_logical_or_None)."""
    import numpy as np

    # run_program.cpp returns LOGICAL row-major bytes (it de-pads). Dispatch
    # mirrors golden simulator._snapshot_traced_tensors EXACTLY:
    #   dtype=="int32" -> int32 * scale
    #   dtype=="fp32"  -> FP16 storage (read_fp16_tile), 2 B/elem, NO scale
    #   else (incl. "int8","fp16") -> int8 * scale   (golden's else branch)
    if dtype == "int32":
        arr = np.frombuffer(raw, dtype="<i4")[: logical_rows * logical_cols]
        view = arr.reshape(logical_rows, logical_cols)
        return view.astype(np.float32) * np.float32(scale), None
    if dtype == "fp32":
        bits = np.frombuffer(raw, dtype="<u2")[: logical_rows * logical_cols]
        bits_v = bits.reshape(logical_rows, logical_cols)
        fp32_v = bits_v.view(np.float16).astype(np.float32)
        return fp32_v, bits_v
    arr = np.frombuffer(raw, dtype=np.int8)[: logical_rows * logical_cols]
    view = arr.reshape(logical_rows, logical_cols)
    return view.astype(np.float32) * np.float32(scale), None


def _fp16_ulp(a_bits, b_bits):
    """Ordered-int fp16 ULP distance (matches test_sfu expect_fp16_ulp). NaN
    vs NaN -> 0; exactly one NaN / inf-sign mismatch -> huge (forces fail)."""
    import numpy as np

    def ordered(u):
        u = u.astype(np.int32)
        return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), u | 0x0000)

    a = a_bits.astype(np.uint16)
    b = b_bits.astype(np.uint16)
    a_nan = ((a & 0x7C00) == 0x7C00) & ((a & 0x03FF) != 0)
    b_nan = ((b & 0x7C00) == 0x7C00) & ((b & 0x03FF) != 0)
    ulp = np.abs(ordered(a) - ordered(b)).astype(np.int64)
    both_nan = a_nan & b_nan
    one_nan = a_nan ^ b_nan
    ulp = np.where(both_nan, 0, ulp)
    ulp = np.where(one_nan, np.int64(1 << 40), ulp)
    return ulp


def _freeze7_band(node_name: str):
    """freeze §7 per-op-class fp16 conformance band, keyed by node name (the
    reliable signal — trace PCs are the SYNC barrier 0x02, not the producer
    opcode). gelu_new -> ≤3 ULP (characterized); everything else -> 0 ULP.

    Caveat for future maintainers: freeze §7 marks masked_softmax 0 ULP only
    on its *characterized fixture*, with the explicit note "e2e is the final
    arbiter — if real-data exp drift appears it gets its OWN characterized
    band". P6b measured 0 ULP e2e so the strict 0 band is currently
    justified; if a future bundle hits real-data exp drift this will hard-
    fail with no slack — that is the signal to add a measured masked_softmax
    band here (same discipline as gelu_new), NOT to loosen it blindly."""
    if "gelu" in node_name:
        return "GELU_FP32(gelu_new)", 3
    return "non-transcendental/exact", 0


def compare(golden_tensors, rtl_entries, rtl_data, instr_image,
            trace_manifest):
    """Per-fragment RTL-vs-golden compare, freeze §7 per-op fp16-ULP bands.
    Returns first-divergence dict or None (byte-match)."""
    import numpy as np

    by_key = {(int(e["pc"]), int(e["event_index"])): e for e in rtl_entries}
    for pc in sorted(trace_manifest):
        # NOTE: codegen records trace events at the SYNC BARRIER
        # (OP_SYNC == 5'h02, verified in rtl/src/include/taccel_pkg.sv:24)
        # that serializes the async gen-2 SFU after the producing op, NOT at
        # the producer instruction. Confirmed empirically: 71/71 trace PCs
        # are 0x02 while the image holds thousands of 0x17-0x1F. (Capture is
        # correct precisely because SYNC drains the SFU writeback before it
        # retires.) So img[pc*8]>>3 is the SYNC, never the producer opcode
        # (recorded caveat R4) — freeze §7 bands must be keyed by node
        # op-class (name), the only reliable signal.
        anchor_op = instr_image[pc * 8] >> 3 if pc * 8 < len(instr_image) else -1
        for ei, ev in enumerate(trace_manifest[pc]):
            node = ev["node_name"]
            if str(ev.get("source", "architectural")) == "virtual":
                continue  # golden-synthesized; no RTL SRAM counterpart
            opclass, band = _freeze7_band(node)
            rs = int(ev.get("row_start", 0))
            lr, lc = int(ev["logical_rows"]), int(ev["logical_cols"])
            ctx = {"pc": pc, "event_index": ei, "node": node,
                   "anchor_op": f"0x{anchor_op:02X}", "op_class": opclass,
                   "band": band, "dtype": ev["dtype"],
                   "capture_phase": ev.get("capture_phase")}
            g_full = golden_tensors.get(node)
            if g_full is None:
                return {**ctx, "reason": "golden node missing"}
            g = np.asarray(g_full)[rs:rs + lr, :lc].astype(np.float32)
            rk = by_key.get((pc, ei))
            if rk is None or rk.get("status") != "captured":
                return {**ctx,
                        "reason": f"RTL capture status={rk and rk.get('status')}"}
            off, sz = int(rk["byte_offset"]), int(rk["byte_size"])
            r_fp32, r_bits = _dequant_fragment(
                ev["dtype"], float(ev["scale"]), rtl_data[off:off + sz],
                lr, lc, int(ev["mem_rows"]), int(ev["mem_cols"]))
            if ev["dtype"] == "fp32":
                g_bits = g.astype(np.float16).view(np.uint16)
                ulp = _fp16_ulp(g_bits, r_bits)
                bad = ulp > band
                if bad.any():
                    i, j = (int(x) for x in np.argwhere(bad)[0])
                    return {**ctx, "row": rs + i, "col": j,
                            "ulp": int(ulp[i, j]),
                            "golden": float(g[i, j]), "rtl": float(r_fp32[i, j]),
                            "reason": f"fp16 ULP {int(ulp[i,j])} > band {band}"}
            else:
                if not np.array_equal(g, r_fp32):
                    d = np.argwhere(g != r_fp32)
                    i, j = (int(x) for x in d[0])
                    return {**ctx, "row": rs + i, "col": j,
                            "golden": float(g[i, j]), "rtl": float(r_fp32[i, j]),
                            "reason": f"{ev['dtype']} not bit-exact"}
    return None


def run_cosim(*, token_id: int = 0, smoke_decode_steps: int = 1,
              payload: Optional[Dict[str, Any]] = None,
              max_cycles: Optional[int] = None,
              exclude_nodes: Optional[set] = None):
    """Full freeze §4.5 RTL-vs-golden prefill gate. Returns a result dict;
    'divergence' is None on byte-match.

    `payload=None` -> tiny fixture (P6b). Pass the GPT-2 124M payload for the
    literal §5 model (P6c). `max_cycles=None` auto-scales from insn_count
    (~1600 cyc/insn measured on 124M; 3000x headroom) — the 5M default would
    time the 124M prefill out at ~5M of its ~100M cycles.
    """
    golden, sp, instr_image = capture_golden(
        token_id=token_id, smoke_decode_steps=smoke_decode_steps,
        payload=payload, exclude_nodes=exclude_nodes)
    if max_cycles is None:
        max_cycles = max(5_000_000, sp.insn_count * 3000)
    csv = emit_snapshot_csv(sp.trace_manifest)
    summary, rtl_entries, rtl_data = run_rtl(
        sp.program_bytes, csv, max_cycles=max_cycles)
    divergence = compare(golden, rtl_entries, rtl_data, instr_image,
                         sp.trace_manifest)
    return {
        "summary": summary,
        "divergence": divergence,
        "n_events": sum(len(v) for v in sp.trace_manifest.values()),
        "n_rtl_captures": len(rtl_entries),
        "insn_count": sp.insn_count,
        "token_id": token_id,
    }


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="P6b RTL-vs-golden prefill gate")
    ap.add_argument("--out", default="/tmp/p6b_prefill.bin")
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--cosim", action="store_true",
                    help="run full RTL-vs-golden compare")
    args = ap.parse_args()

    if args.cosim:
        res = run_cosim(token_id=args.token)
        s = res["summary"]
        print(f"insn_count={res['insn_count']} events={res['n_events']} "
              f"rtl_captures={res['n_rtl_captures']}")
        print(f"summary: status={s.get('status')} fault={s.get('fault')} "
              f"overlap={s.get('forbidden_overlap_violation')} "
              f"retired={s.get('retired_instructions')}")
        if res["divergence"] is None:
            print("RESULT: BYTE-MATCH ✓ (no first divergence)")
        else:
            print("RESULT: FIRST DIVERGENCE:")
            print(_json.dumps(res["divergence"], indent=2, default=str))
    else:
        sp = serialize_prefill_bundle(token_id=args.token)
        Path(args.out).write_bytes(sp.program_bytes)
        print(f"wrote {args.out}: {len(sp.program_bytes)} bytes  "
              f"insn_count={sp.insn_count} data_base={sp.data_base} "
              f"events={sum(len(v) for v in sp.trace_manifest.values())}")
