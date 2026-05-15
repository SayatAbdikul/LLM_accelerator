"""Per-sublayer A/B: W8A16 like-for-like reference vs FP32 reference.

Workstream 0.2 of the perplexity-improvement plan. Purpose: localize
*where* W8A16 quantization damage accumulates, sublayer-by-sublayer, and
confirm the LayerNorm-eps fix (W0.1) removed the systematic per-layer
LN divergence.

Scoping note (deliberate, reported to the user): the plan originally
phrased 0.2 as "simulator-backed bundle vs M4-E reference". `HostRunner`
exposes no per-sublayer hook and instrumenting the opaque simulator
bundle is disproportionate for a Tier-0 diagnostic. `NanoGPTFP32Reference`
*already* has a complete per-node trace (`incremental_node_trace`), so we
compare the M4-E W8A16 reference against the FP32 reference using the
identical node-name convention. That is the more actionable artifact for
the perplexity goal — it shows which sublayer the quantization hurts,
not merely where two implementations of the same quant design drift. The
sim-backed-bundle ↔ M4-E-ref sign-flip is confirmed separately via the
existing 33/257-tok gates in the verification step.

Diagnostic test: it never hard-fails on divergence (divergence is the
finding); it asserts only structural invariants and prints the table.
Run with `-s` to see the per-sublayer breakdown.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")

# Short prompt is enough to localize per-sublayer drift; keep it cheap.
N_TOKENS = 16


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / (||b|| + tiny). b is the FP32 reference."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("nan")
    denom = float(np.linalg.norm(b)) + 1e-12
    return float(np.linalg.norm(a - b) / denom)


def test_w8a16_vs_fp32_sublayer_divergence():
    missing = [str(p) for p in (FIXTURE, TOKENIZER_DIR, EVAL_TEXT) if not p.exists()]
    if missing:
        pytest.skip(f"W8A16 fixtures missing: {missing}")
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable (FP32 reference needs it)")

    from taccel.runtime.gpt2_perplexity import tokenize_text_file
    from taccel.runtime.fp32_reference import NanoGPTFP32Reference
    from taccel.runtime.w8a16_simulator_reference import (
        NanoGPTW8A16SimulatorReference,
    )

    payload = torch.load(FIXTURE, map_location="cpu")
    tokens = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=N_TOKENS)
    assert tokens, "tokenizer produced no tokens"

    fp32 = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    w8 = NanoGPTW8A16SimulatorReference(payload)

    fp32_traces = fp32.incremental_node_trace(tokens)
    w8_traces = w8.incremental_node_trace(tokens)
    assert len(fp32_traces) == len(w8_traces) == len(tokens)

    # Localize at the *last* position — most context, where the W8A16
    # gap is largest (the 1.59×→3.32× context-dependent widening).
    last_fp32 = fp32_traces[-1]
    last_w8 = w8_traces[-1]
    common = [k for k in last_fp32 if k in last_w8]
    assert common, "no shared node names between the two traces"

    rows = [(k, _rel_l2(last_w8[k]["value"], last_fp32[k]["value"])) for k in common]
    finite = [(k, d) for k, d in rows if np.isfinite(d)]
    assert finite, "all sublayer diffs were non-finite / shape-mismatched"

    print(f"\n=== W8A16-ref vs FP32-ref per-sublayer rel-L2 "
          f"(last of {len(tokens)} positions) ===")
    for k, d in rows:
        print(f"  {k:<34} {d:8.4f}" + ("  <-- nan/shape" if not np.isfinite(d) else ""))

    worst_k, worst_d = max(finite, key=lambda kd: kd[1])
    first_25 = next((k for k, d in finite if d >= 0.25), None)
    first_50 = next((k for k, d in finite if d >= 0.50), None)
    print(f"  worst: {worst_k} = {worst_d:.4f}")
    print(f"  first >=25% rel-L2: {first_25}")
    print(f"  first >=50% rel-L2: {first_50}")

    # Per-layer residual-stream drift growth (the compounding signal).
    n_layer = int(payload["model_args"]["n_layer"])
    print("  residual-stream drift by layer:")
    for li in range(n_layer):
        for tag in ("residual1", "residual2"):
            key = f"block{li}_{tag}"
            if key in last_fp32 and key in last_w8:
                print(f"    {key:<24} "
                      f"{_rel_l2(last_w8[key]['value'], last_fp32[key]['value']):8.4f}")

    # Structural invariants only — divergence itself is the finding.
    assert np.isfinite(worst_d)
    assert "lm_head" in common, "logits node missing from trace"
