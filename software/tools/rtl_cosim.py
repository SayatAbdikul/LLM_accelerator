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
class SerializedSequence:
    """All data needed to run RTL for prefill + N teacher-forced decode steps.

    run_program always starts execution at PC 0 (entry_point in the ProgramBinary
    header is not used by the RTL). Decode ProgramBinaries therefore contain ONLY
    the decode instruction bytes — local PC 0 maps to the first decode instruction.
    Data is still placed at bundle_data_base (absolute DRAM offset) so that all
    DMA SET_ADDR absolute references within the decode stream remain correct.

    decode_manifest_local: local (0-based) decode PCs — used for RTL snapshot CSV
    and compare() (both sides use the same local PCs once the decode binary starts at PC 0).
    decode_manifest_abs: absolute decode PCs (decode_pc + local) — used for the golden
    simulator which tracks absolute PCs across the full flat DRAM image.
    """
    bundle_data_base: int
    bundle_kv_cache_base: int
    bundle_kv_cache_size: int
    bundle_decode_pc: int
    bundle_decode_instrs_offset: int     # byte offset of decode stream in flat DRAM image
    prefill_insn_count: int              # prefill-only insn_count (for max_cycles scaling)
    n_decode_steps: int
    token_id: int
    step_tokens: list
    prefill_program_bytes: bytes
    prefill_manifest: Dict[int, list]
    prefill_instr_image: bytes           # full flat instr region (for compare anchor lookup)
    decode_instr_images: list            # bytes per step: ONLY decode instruction bytes
    decode_data_template: bytes          # data section template (weights + zeroed KV)
    decode_manifest_local: Dict[int, list]  # local decode PCs for RTL CSV + compare
    decode_manifest_abs: Dict[int, list]    # absolute decode PCs for golden simulator


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
            timeout_s: int = 1800, dram_dump: tuple | None = None):
    """Run the patched prefill on Verilator run_program with snapshot capture;
    return (summary, rtl_entries, rtl_data_bytes[, dram_dump_bytes]).
    timeout_s default 30min (124M 1-tok prefill ~292s clean).
    dram_dump=(offset,size) -> also pass run_program's documented
    --dram-dump-* (separate from the snapshot mechanism & from any
    manifest augmentation) and return the dumped bytes as a 4th element."""
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
        argv = [str(RTL_BINARY),
                "--program", str(td / "p.bin"),
                "--json-out", str(td / "sum.json"),
                "--snapshot-request", str(td / "snap.csv"),
                "--snapshot-manifest-out", str(td / "man.json"),
                "--snapshot-data-out", str(td / "data.bin"),
                "--max-cycles", str(max_cycles)]
        if dram_dump is not None:
            off, sz = int(dram_dump[0]), int(dram_dump[1])
            argv += ["--dram-dump-offset", str(off),
                     "--dram-dump-size", str(sz),
                     "--dram-dump-out", str(td / "dram.bin")]
        cp = subprocess.run(argv, capture_output=True, text=True,
                            timeout=timeout_s)
        if cp.returncode != 0:
            raise RuntimeError(
                f"run_program exited {cp.returncode}\n{cp.stderr[-2000:]}"
            )
        summary = json.loads((td / "sum.json").read_text())
        manifest = json.loads((td / "man.json").read_text())
        data = (td / "data.bin").read_bytes()
        if dram_dump is not None:
            return (summary, manifest["entries"], data,
                    (td / "dram.bin").read_bytes())
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


def capture_golden_sequence(*, token_ids=(0, 0, 0), payload=None,
                            exclude_nodes=None):
    """Teacher-forced golden run for prefill + len(token_ids)-1 decode steps.

    token_ids[0] = prefill input; token_ids[i>=1] = decode input at position i.
    Returns (golden_prefill_tensors, golden_decode_tensors_list, SerializedSequence).

    Note: block_size=128 on the tiny fixture caps the sequence at 128 total tokens.
    The plan's '257-tok' figure is infeasible for the tiny fixture; use N=2 for CI.

    Key constraint: run_program always starts at PC 0 — it does not read the
    entry_point field from the ProgramBinary header. Decode ProgramBinaries
    therefore store ONLY the decode instruction bytes (local PC 0 = first decode
    instruction), while data_base is kept at bundle.data_base so all DMA
    SET_ADDR absolute references within the decode stream remain valid.
    """
    from taccel.assembler.assembler import ProgramBinary
    from taccel.golden_model.simulator import Simulator
    from taccel.runtime.calibration import build_calibration_scales
    from taccel.runtime.host_runner import HostRunner
    from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle

    n_decode_steps = len(token_ids) - 1
    if n_decode_steps < 1:
        raise ValueError("token_ids must have >= 2 tokens (1 prefill + 1+ decode)")

    if payload is None:
        payload = _load_tiny_payload()
    scales = build_calibration_scales(payload)
    tiny = build_stage3_tiny_decoder_bundle(
        payload, smoke_decode_steps=n_decode_steps,
        calibration_scales=scales, ptq_preset=FREEZE_PTQ_PRESET,
    )
    build_obj = tiny.build
    bundle = build_obj.bundle

    data_base = int(bundle.data_base)
    kv_cache_base = int(bundle.kv_cache_base)
    kv_cache_size = int(bundle.kv_cache_size_bytes)
    decode_pc = int(bundle.decode_pc)
    decode_instrs_offset = int(bundle.decode_instrs_offset)  # byte offset of decode stream
    prefill_insn_count = int(bundle.prefill_pc) + len(bundle.prefill_instrs) // 8  # == len(prefill)//8

    # decode_manifest_local: local (0-based) PCs within decode stream.
    # golden simulator needs absolute PCs; RTL (starting at local PC 0) needs local PCs.
    decode_manifest_local = dict(build_obj.decode_codegen.trace_manifest)
    decode_manifest_abs = {
        decode_pc + lpc: evs for lpc, evs in decode_manifest_local.items()
    }
    prefill_manifest = dict(build_obj.prefill_codegen.trace_manifest)

    if exclude_nodes:
        def _prune(m):
            return {
                pc: kept for pc, evs in m.items()
                if (kept := [e for e in evs if e["node_name"] not in exclude_nodes])
            }
        prefill_manifest = _prune(prefill_manifest)
        decode_manifest_local = _prune(decode_manifest_local)
        decode_manifest_abs = _prune(decode_manifest_abs)

    # Golden simulator prefill.
    # HostRunner.__init__ -> load_bundle -> trace_manifest={}; re-call load_bundle
    # after construction so the manifest injection lands on a clean slate.
    sim = Simulator()
    runner = HostRunner(bundle, simulator=sim)
    sim.load_bundle(bundle)
    sim.trace_manifest = prefill_manifest
    sim.enable_trace()
    runner.run_prefill([int(token_ids[0])])
    golden_prefill = sim.get_trace_payload()["tensors"]

    # Snapshot prefill instruction image and static data template.
    image_after_prefill = bundle.materialize(reset_runtime=False)
    prefill_instr_image = bytes(image_after_prefill[:data_base])
    # Data section is purely static (weights + zeroed KV); runtime patches only
    # touch instruction streams, so this template is invariant across all steps.
    decode_data_template = bytes(image_after_prefill[data_base:])

    pb_prefill = ProgramBinary(
        instructions=prefill_instr_image,
        data=decode_data_template,
        entry_point=int(bundle.prefill_pc),  # == 0; run_program starts at PC 0
        insn_count=int(bundle.insn_count),
        data_base=data_base,
        input_offset=0, pos_embed_patch_dram_offset=0,
        pos_embed_cls_dram_offset=0, cls_token_dram_offset=0,
        trace_manifest={}, compiler_manifest={},
    )

    # Golden decode steps (teacher-forced: token_ids[i+1] at position i+1).
    step_tokens = [int(t) for t in token_ids[1:]]
    golden_decode = []
    decode_instr_images = []

    for i, step_tok in enumerate(step_tokens):
        # Inject the decode manifest at absolute PCs; enable_trace clears prior step.
        sim.trace_manifest = decode_manifest_abs
        sim.enable_trace()
        runner.run_decode_step(step_tok, i + 1)
        golden_decode.append(sim.get_trace_payload()["tensors"])

        # Snapshot ONLY the decode instruction bytes for this step.
        # run_program ignores entry_point and always starts at PC 0, so the RTL
        # decode ProgramBinary must contain only the decode stream (local PC 0 =
        # first decode instruction). data_base stays at bundle.data_base so
        # absolute DRAM addresses embedded in SET_ADDR instructions are correct.
        image_step = bundle.materialize(reset_runtime=False)
        decode_only_bytes = bytes(image_step[decode_instrs_offset:data_base])
        decode_instr_images.append(decode_only_bytes)
        assert bytes(image_step[data_base:]) == decode_data_template, (
            f"data section mutated at decode step {i}: "
            "runtime patches must not touch the data region"
        )

    seq = SerializedSequence(
        bundle_data_base=data_base,
        bundle_kv_cache_base=kv_cache_base,
        bundle_kv_cache_size=kv_cache_size,
        bundle_decode_pc=decode_pc,
        bundle_decode_instrs_offset=decode_instrs_offset,
        prefill_insn_count=prefill_insn_count,
        n_decode_steps=n_decode_steps,
        token_id=int(token_ids[0]),
        step_tokens=step_tokens,
        prefill_program_bytes=pb_prefill.to_bytes(),
        prefill_manifest=prefill_manifest,
        prefill_instr_image=prefill_instr_image,
        decode_instr_images=decode_instr_images,
        decode_data_template=decode_data_template,
        decode_manifest_local=decode_manifest_local,
        decode_manifest_abs=decode_manifest_abs,
    )
    return golden_prefill, golden_decode, seq


def run_cosim_sequence(*, token_ids=(0, 0, 0), payload=None,
                       max_cycles=None, exclude_nodes=None):
    """Multi-step RTL-vs-golden gate: prefill + len(token_ids)-1 decode steps.

    KV state is threaded between RTL runs via --dram-dump-* (the KV cache
    region is dumped after each run and injected into the next step's data
    section). Returns a result dict; top-level 'divergence' is None on full
    byte-match, or the first-divergence dict with an added 'decode_step' key.

    Plan deviation from §B3 '257-tok': the tiny fixture block_size=128 caps
    the sequence at 128 total tokens. token_ids defaults to [0,0,0] (prefill
    + 2 decode steps) which is fast (<60 s total) and fully well-posed.

    RTL constraint: run_program always starts execution at PC 0. Decode
    ProgramBinaries therefore contain ONLY the decode instruction bytes; the
    snapshot CSV uses local (0-based) decode PCs. data_base is preserved at
    bundle.data_base so all DMA SET_ADDR absolute addresses remain correct.
    """
    from taccel.assembler.assembler import ProgramBinary

    golden_prefill, golden_decode, seq = capture_golden_sequence(
        token_ids=token_ids, payload=payload, exclude_nodes=exclude_nodes,
    )
    if max_cycles is None:
        max_cycles = max(5_000_000, seq.prefill_insn_count * 3000)

    csv_prefill = emit_snapshot_csv(seq.prefill_manifest)
    # Decode CSV uses LOCAL (0-based) decode PCs — the RTL binary starts at PC 0.
    csv_decode = emit_snapshot_csv(seq.decode_manifest_local)

    # RTL prefill — dump KV region for threading into decode step 0.
    summary_p, rtl_entries_p, rtl_data_p, kv_dump = run_rtl(
        seq.prefill_program_bytes, csv_prefill, max_cycles=max_cycles,
        dram_dump=(seq.bundle_kv_cache_base, seq.bundle_kv_cache_size),
    )
    divergence = compare(
        golden_prefill, rtl_entries_p, rtl_data_p,
        seq.prefill_instr_image, seq.prefill_manifest,
    )
    result = {
        "summary_prefill": summary_p,
        "divergence": divergence,
        "n_events_prefill": sum(len(v) for v in seq.prefill_manifest.values()),
        "n_rtl_captures_prefill": len(rtl_entries_p),
        "n_decode_steps": seq.n_decode_steps,
        "prefill_insn_count": seq.prefill_insn_count,
        "token_id": seq.token_id,
        "step_tokens": seq.step_tokens,
        "step_results": [],
    }
    if divergence is not None:
        return result

    # RTL decode steps: ONLY decode instruction bytes; data_base preserved so
    # absolute DRAM SET_ADDR references are correct; KV region overridden per step.
    for i in range(seq.n_decode_steps):
        data = bytearray(seq.decode_data_template)
        kv_off = seq.bundle_kv_cache_base - seq.bundle_data_base
        data[kv_off:kv_off + seq.bundle_kv_cache_size] = kv_dump

        decode_insn_count = len(seq.decode_instr_images[i]) // 8
        pb = ProgramBinary(
            instructions=seq.decode_instr_images[i],
            data=bytes(data),
            entry_point=0,           # run_program always starts at PC 0
            insn_count=decode_insn_count,
            data_base=seq.bundle_data_base,  # absolute DRAM offset; unchanged
            input_offset=0, pos_embed_patch_dram_offset=0,
            pos_embed_cls_dram_offset=0, cls_token_dram_offset=0,
            trace_manifest={}, compiler_manifest={},
        )
        decode_max_cycles = max(5_000_000, decode_insn_count * 3000)
        summary_d, rtl_entries_d, rtl_data_d, kv_dump = run_rtl(
            pb.to_bytes(), csv_decode, max_cycles=decode_max_cycles,
            dram_dump=(seq.bundle_kv_cache_base, seq.bundle_kv_cache_size),
        )
        # compare() uses local decode PCs to look up RTL entries and node names
        # to look up golden tensors — both are local-PC-agnostic on the golden side.
        step_div = compare(
            golden_decode[i], rtl_entries_d, rtl_data_d,
            seq.decode_instr_images[i], seq.decode_manifest_local,
        )
        n_ev = sum(len(v) for v in seq.decode_manifest_local.values())
        step_r = {
            "step": i,
            "token": seq.step_tokens[i],
            "summary": summary_d,
            "n_events": n_ev,
            "n_rtl_captures": len(rtl_entries_d),
            "divergence": step_div,
        }
        result["step_results"].append(step_r)
        if step_div is not None:
            result["divergence"] = {**step_div, "decode_step": i}
            return result

    return result


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
