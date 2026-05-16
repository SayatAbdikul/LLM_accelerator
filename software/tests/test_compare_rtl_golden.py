"""Gen-2 ISA freeze conformance gate (software/docs/isa_generation_freeze.md).

This file is the conformance gate named in freeze §5 ("definition of done").
It has three legs, in increasing strength:

  1. ``test_frozen_golden_sha_pin`` — freeze §6 content-level pin. Recomputes
     the git-blob SHA-1 of the golden model and fails *loud* on any drift.
     A drifted golden model fails the gate **before** any RTL comparison,
     closing the "cosim vs a moving golden" hole the freeze exists to close.

  2. ``test_frozen_bundle_runs_gen2_clean_and_deterministic`` — builds the
     frozen ``weight_only_int8_quarot`` decoder bundle (freeze §4.5/§5),
     asserts it actually emits the gen-2 opcodes the freeze is about
     (non-vacuous), runs it end-to-end on the pinned golden ``HostRunner``,
     and asserts two independent compiles are **bit-identical** (prefill +
     decode logits). This is the golden-side necessary precursor to the
     full bar: if the frozen reference itself is non-deterministic or does
     not exercise gen-2, an RTL byte-match is meaningless.

  3. ``test_rtl_cosim_gen2_byte_match`` — the full freeze §4.5 RTL-vs-golden
     end-to-end byte-match (FP16-ULP per §7 per-op bands). **SKIPPED**, not
     faked: the only RTL runner (``rtl/verilator/run_program.cpp``) consumes
     a single-stream DeiT ``ProgramBinary`` (``MAGIC``/``HEADER_FMT`` in
     ``taccel/assembler/assembler.py``), while the frozen bundle is a
     two-stream ``ProgramBundle`` driven by ``HostRunner`` (prefill + decode,
     runtime patch sites, kv-cache). Feasibility for the missing bridge has
     been *measured* (the ProgramBinary writer and the simulator's 15-field
     ``trace_manifest`` schema at ``simulator.py:~328`` both already exist);
     the residual is a bounded bundle-DRAM→ProgramBinary serializer plus a
     ``trace_manifest``→snapshot-CSV emitter plus a first-divergence drive.
     That focused block is tracked as task #105 and is intentionally not
     attempted here as a half-finished artifact. This leg ``skip``s with
     that pointer — it is never ``xpass``/faked green.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from taccel.runtime.calibration import build_calibration_scales
from taccel.runtime.tiny_fixture import (
    build_stage3_tiny_decoder_bundle,
    run_tiny_decode_trace,
)

# --------------------------------------------------------------------------
# freeze §6 content pin
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_MODEL_PATH = REPO_ROOT / "software" / "taccel" / "golden_model" / "simulator.py"
# git-blob SHA-1 of simulator.py at frozen commit aa9a9c0 (freeze §4.6/§6),
# verified byte-identical at HEAD 3314043 and worktree on 2026-05-16.
FROZEN_GOLDEN_BLOB_SHA = "7746e65598961ac8430f8eeece45d7ec976584cd"

# freeze §4.5: the conformance bundle is the GPT-2 W8A16 weight_only_int8_quarot
# generation. Its FP32 sub-layer ops are the gen-2 ISA this freeze covers.
FREEZE_PTQ_PRESET = "weight_only_int8_quarot"

# gen-2 opcode space (freeze: 8 normative ops; 0x1C SOFTMAX_FP32 stays reserved).
GEN2_OPS = {
    0x17: "DEQUANT_ACCUM_FP32",
    0x18: "QUANT_FP32_INT8",
    0x19: "VADD_FP32",
    0x1A: "LAYERNORM_FP32",
    0x1B: "GELU_FP32",
    0x1D: "MASKED_SOFTMAX_FP32",
    0x1E: "DEQUANT_ACCUM_FP32_SCALED",
    0x1F: "MAX_ABS_REDUCE_FP32",
}
RESERVED_ILLEGAL_OP = 0x1C  # SOFTMAX_FP32 — must never appear in a frozen bundle

TOOL_PATH = REPO_ROOT / "software" / "tools" / "train_tiny_fixture.py"


def _git_blob_sha1(path: Path) -> str:
    """Recompute git's blob object hash (no subprocess / git dependency)."""
    data = path.read_bytes()
    header = b"blob " + str(len(data)).encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _load_tool():
    spec = importlib.util.spec_from_file_location("train_tiny_fixture", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opcode_histogram(*instr_blobs: bytes) -> dict[int, int]:
    """Opcode[63:59] occupies the top 5 bits of every 8-byte instruction word
    (fixed across all formats in this ISA), so op = (first_byte >> 3)."""
    hist: dict[int, int] = {}
    for blob in instr_blobs:
        if len(blob) % 8:
            raise AssertionError(
                f"instruction stream not 8-byte aligned: {len(blob)} bytes"
            )
        for i in range(0, len(blob), 8):
            op = blob[i] >> 3
            hist[op] = hist.get(op, 0) + 1
    return hist


# --------------------------------------------------------------------------
# Leg 1 — freeze §6 fail-loud content pin
# --------------------------------------------------------------------------
def test_frozen_golden_sha_pin():
    """The golden model must be byte-identical to the frozen reference.

    This is the first gate: a drifted golden model is not a conformance
    target, it is the moving target the freeze exists to eliminate. Fail
    loud here before anything compares against it.
    """
    assert GOLDEN_MODEL_PATH.exists(), f"golden model missing: {GOLDEN_MODEL_PATH}"
    actual = _git_blob_sha1(GOLDEN_MODEL_PATH)
    assert actual == FROZEN_GOLDEN_BLOB_SHA, (
        "FROZEN GOLDEN DRIFT — gen-2 conformance is undefined.\n"
        f"  expected blob {FROZEN_GOLDEN_BLOB_SHA} (freeze §6, commit aa9a9c0)\n"
        f"  actual   blob {actual}\n"
        f"  file: {GOLDEN_MODEL_PATH}\n"
        "simulator.py changed since the freeze. Per freeze §6 this REQUIRES "
        "a new freeze revision + gen-2 fixture regen "
        "(software/tools/gen_gen2_fixtures.py), then update "
        "FROZEN_GOLDEN_BLOB_SHA + isa_generation_freeze.md §6. Do NOT relax "
        "this assertion to make the gate pass."
    )


# --------------------------------------------------------------------------
# Leg 2 — golden-side gen-2 conformance precursor (real gate)
# --------------------------------------------------------------------------
def test_frozen_bundle_runs_gen2_clean_and_deterministic():
    """Frozen weight_only_int8_quarot bundle: exercises gen-2, runs e2e on the
    pinned golden HostRunner, and is bit-identical across independent compiles.
    """
    tool = _load_tool()
    if not tool.DEFAULT_FIXTURE.exists():
        pytest.skip(
            "tiny nanoGPT fixture not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )
    torch = pytest.importorskip("torch")
    payload = torch.load(tool.DEFAULT_FIXTURE, map_location="cpu")

    MAX_NEW_TOKENS = 4
    PROMPT_IDS = [0]

    # Precompute calibration once so the only thing under test is the
    # bundle-build + golden-run path (mirrors test_tiny_decode_determinism).
    scales = build_calibration_scales(payload)

    tiny_a = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=MAX_NEW_TOKENS,
        calibration_scales=scales,
        ptq_preset=FREEZE_PTQ_PRESET,
    )
    tiny_b = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=MAX_NEW_TOKENS,
        calibration_scales=scales,
        ptq_preset=FREEZE_PTQ_PRESET,
    )

    # (a) Non-vacuous: the frozen bundle must actually emit gen-2 ops, and
    #     must never emit the reserved-illegal 0x1C.
    bundle = tiny_a.build.bundle
    hist = _opcode_histogram(bundle.prefill_instrs, bundle.decode_instrs)
    present_gen2 = {op: hist[op] for op in GEN2_OPS if hist.get(op, 0) > 0}
    pretty = ", ".join(
        f"0x{op:02X} {GEN2_OPS[op]}={present_gen2[op]}" for op in sorted(present_gen2)
    )
    assert hist.get(RESERVED_ILLEGAL_OP, 0) == 0, (
        f"frozen bundle emits reserved-illegal opcode 0x{RESERVED_ILLEGAL_OP:02X} "
        f"(SOFTMAX_FP32) {hist[RESERVED_ILLEGAL_OP]}x — freeze violation"
    )
    assert len(present_gen2) >= 3, (
        "frozen bundle does not exercise gen-2 — conformance gate would be "
        f"vacuous. gen-2 ops present: {{{pretty}}}; full opcode hist: "
        f"{ {f'0x{k:02X}': v for k, v in sorted(hist.items())} }"
    )
    # A GPT-2 decoder forward structurally must layer-norm; that op is gen-2
    # under this freeze. Its absence means the bundle is not the frozen gen.
    assert hist.get(0x1A, 0) > 0, (
        f"frozen bundle has no LAYERNORM_FP32 (0x1A); gen-2 present: {{{pretty}}}"
    )

    # (b)+(c) Runs end-to-end on the pinned golden HostRunner, and two
    #         independent compiles are bit-identical (prefill + every
    #         decode step). Non-determinism here would make any future RTL
    #         byte-match meaningless, so this is the gate, not a smoke test.
    trace_a = run_tiny_decode_trace(tiny_a, PROMPT_IDS, max_new_tokens=MAX_NEW_TOKENS)
    trace_b = run_tiny_decode_trace(tiny_b, PROMPT_IDS, max_new_tokens=MAX_NEW_TOKENS)

    assert trace_a.generated == trace_b.generated, (
        f"frozen golden non-deterministic token stream: "
        f"{trace_a.generated} vs {trace_b.generated}"
    )
    assert len(trace_a.logits) == MAX_NEW_TOKENS + 1  # prefill + decode steps
    for step, (la, lb) in enumerate(zip(trace_a.logits, trace_b.logits)):
        assert np.array_equal(la, lb), (
            f"frozen golden non-deterministic logits at step {step}: "
            f"max|diff| = {np.abs(la.astype(np.int64) - lb.astype(np.int64)).max()}"
        )


# --------------------------------------------------------------------------
# Leg 3 — the full freeze §4.5 RTL-vs-golden e2e byte-match (P6b, task #105)
# --------------------------------------------------------------------------
def _load_rtl_cosim():
    import sys

    path = REPO_ROOT / "software" / "tools" / "rtl_cosim.py"
    spec = importlib.util.spec_from_file_location("rtl_cosim", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: rtl_cosim defines @dataclass SerializedPrefill,
    # and dataclasses resolves sys.modules[cls.__module__] at instantiation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("token_id", [0, 5])
def test_rtl_cosim_gen2_byte_match(token_id):
    """Substantive freeze §4.5 property (NOT the literal §5 bar — see scope):
    the RTL executing the *real* compiled bundle byte-matches the pinned
    golden end-to-end (every captured gen-1 + gen-2 node) within the freeze
    §7 per-op-class fp16-ULP bands.

    Scope (honest — two gaps vs the literal §5 definition-of-done):
      * MODEL: the tiny 2-layer nanoGPT shakespeare-char fixture
        (train_tiny_fixture DEFAULT_FIXTURE, d128/l2) compiled with the
        `weight_only_int8_quarot` PRESET — NOT GPT-2 124M (12-layer).
      * SEQUENCE: single-token PREFILL only — NOT 257-tok prefill+decode.
        prefill's only runtime patches are token/pos embeddings (kv/attn
        are decode-only), so a fixed token yields a static single-shot
        ProgramBinary run_program executes faithfully.
    This is the *same gen-2 ISA* the GPT-2 bundle emits, exercised through
    the real compiler/codegen/SYNC path, so the gen-2 datapath is
    conformant — but freeze §5 stays formally open until GPT-2 124M /
    257-tok byte-matches (P6c, task #106). See isa_generation_freeze.md §5
    Status (2026-05-16).
    """
    cosim = _load_rtl_cosim()
    if not cosim.RTL_BINARY.exists():
        pytest.skip(
            f"run_program not built ({cosim.RTL_BINARY}); "
            "build: make -C rtl/verilator run_program"
        )
    tool = _load_tool()
    if not tool.DEFAULT_FIXTURE.exists():
        pytest.skip(
            "tiny nanoGPT fixture not generated; run "
            "PYTHONPATH=software python software/tools/train_tiny_fixture.py"
        )

    res = cosim.run_cosim(token_id=token_id)
    s = res["summary"]

    # Clean RTL run is a precondition of a meaningful byte-match.
    assert s.get("status") == "halted", f"RTL did not cleanly halt: {s}"
    assert s.get("fault") is False, f"RTL architectural fault: {s}"
    assert s.get("forbidden_overlap_violation") is False, (
        f"RTL forbidden engine-overlap (H1 invariant) violated: {s}"
    )
    assert not s.get("timeout"), f"RTL run timed out: {s}"

    # Non-vacuous: the gate must actually compare captured fragments.
    assert res["n_events"] >= 50, f"too few trace events: {res}"
    assert res["n_rtl_captures"] == res["n_events"], (
        f"RTL capture count {res['n_rtl_captures']} != "
        f"manifest events {res['n_events']}"
    )

    # The freeze §4.5 bar: end-to-end byte-match within freeze §7 bands.
    assert res["divergence"] is None, (
        "RTL-vs-golden FIRST DIVERGENCE on the frozen gen-2 bundle "
        f"(token={token_id}):\n"
        + json.dumps(res["divergence"], indent=2, default=str)
    )
