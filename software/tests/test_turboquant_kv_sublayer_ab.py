"""Level-2 (per-sublayer) verification of TurboQuant KV-cache quantization.

Same W8A16 reference both sides, differing ONLY in the KV cache (FP16 vs
TurboQuant round-trip) — so the diff isolates the KV-quant effect with no
FP16-storage / activation-quant confound. Two measurements:

  1. Cache reconstruction (real data): rel-L2 of the stored round-tripped
     K/V vs the true projected K/V at the last position — the in-situ,
     real-vector version of the Level-1 synthetic check.
  2. Downstream drift: rel-L2(ON, OFF) for attn_v / concat / residual /
     ln_f / lm_head — how far KV quant moves each sublayer, and which
     layer/head is most sensitive.

Diagnostic: never hard-fails on magnitude (that's the finding); asserts
only structural invariants. Run with -s for the localization tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from taccel.quantizer.turboquant import TurboQuantKV

FIXTURE = Path("software/tests/fixtures/generated/gpt2_converted_nanogpt.pt")
TOKENIZER_DIR = Path("software/tests/fixtures/generated/hf_gpt2")
EVAL_TEXT = Path("software/tests/fixtures/generated/wikitext2_stage5_eval.txt")
N_TOKENS = 16
SEED = 20260515


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_turboquant_kv_sublayer_divergence():
    missing = [str(p) for p in (FIXTURE, TOKENIZER_DIR, EVAL_TEXT) if not p.exists()]
    if missing:
        pytest.skip(f"fixtures missing: {missing}")
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")
    from taccel.runtime.gpt2_perplexity import tokenize_text_file
    from taccel.runtime.w8a16_simulator_reference import (
        NanoGPTW8A16SimulatorReference,
    )

    payload = torch.load(FIXTURE, map_location="cpu")
    tokens = tokenize_text_file(TOKENIZER_DIR, EVAL_TEXT, max_tokens=N_TOKENS)
    assert tokens
    d_head = int(payload["model_args"]["n_embd"]) // int(payload["model_args"]["n_head"])

    tq = TurboQuantKV(d=d_head, bits=3.0, variant="ip", target="kv", seed=SEED)
    off = NanoGPTW8A16SimulatorReference(payload).incremental_node_trace(tokens)[-1]
    on = NanoGPTW8A16SimulatorReference(
        payload, kv_quant=tq
    ).incremental_node_trace(tokens)[-1]

    # 1. Cache reconstruction error on real K/V (last position).
    print(f"\n=== TurboQuant KV cache recon (real vectors, last of "
          f"{len(tokens)} pos) {tq} ===")
    krs, vrs = [], []
    for key, store in (("key", "k_cache_tq"), ("value", "v_cache_tq")):
        worst = (None, -1.0)
        for name in on:
            if not name.endswith(f"_{store}"):
                continue
            true_name = name.replace(f"_{store}", f"_{key}")
            if true_name not in on:
                continue
            r = _rel_l2(on[name]["value"], on[true_name]["value"])
            (krs if key == "key" else vrs).append(r)
            if r > worst[1]:
                worst = (name, r)
        pool = krs if key == "key" else vrs
        print(f"  {key:5} recon rel-L2: mean={np.mean(pool):.4f} "
              f"max={np.max(pool):.4f} @ {worst[0]}")

    # 2. Downstream drift ON vs OFF (KV quant's propagated effect).
    common = [k for k in off if k in on]
    tags = ("attn_v", "concat", "residual1", "residual2")
    print("=== downstream drift rel-L2(ON,OFF) ===")
    drift = {}
    for name in common:
        if any(t in name for t in tags) or name in ("ln_f", "lm_head"):
            drift[name] = _rel_l2(on[name]["value"], off[name]["value"])
    for name in ("ln_f", "lm_head"):
        if name in drift:
            print(f"  {name:<30} {drift[name]:.4f}")
    attn = {k: v for k, v in drift.items() if "attn_v" in k}
    worst_k = max(attn, key=attn.get)
    print(f"  attn_v: mean={np.mean(list(attn.values())):.4f} "
          f"worst={worst_k}={attn[worst_k]:.4f}")
    n_layer = int(payload["model_args"]["n_layer"])
    print("  residual-stream drift by layer:")
    for li in range(n_layer):
        for tag in ("residual1", "residual2"):
            key = f"block{li}_{tag}"
            if key in drift:
                print(f"    {key:<22} {drift[key]:.4f}")

    # Structural invariants only.
    assert np.all(np.isfinite(list(drift.values())))
    assert np.mean(krs) > 0 and np.mean(vrs) > 0, "KV quant had no effect"
    assert drift["lm_head"] > 0, "KV quant didn't propagate to logits"
    assert np.max(krs + vrs) < 1.5, "implausible cache recon error"
