"""Numerical bisection: simulator-backed bundle vs M4-E reference.

Asserts that at d_model=16 (no tiling, no fc2-streaming, no spill) the
two paths agree within FP16 ULP. Divergence here points to bugs in the
M3 foundation (M2.5-A dynamic activation scaling, QKT/attn_v static
scales, padding handling) rather than M4-C/G scale-specific code paths.

If THIS test passes, scale up to d_model=192 (still no spill), then
d_model=384 (spill triggers), then d_model=768 (full GPT-2 scale).
"""
from __future__ import annotations

import numpy as np

from taccel.runtime.host_runner import HostRunner
from taccel.runtime.tiny_fixture import build_stage3_tiny_decoder_bundle
from taccel.runtime.w8a32_simulator_reference import (
    NanoGPTW8A32SimulatorReference,
)


def _one_layer_gpt2_payload(d_model=16):
    """Single-layer GPT-2 payload for bisection. Same shape as the fixture
    in test_w8a32_codegen.py but parametrizable on d_model."""
    mlp_dim = 4 * d_model
    vocab = 16
    block = 16
    state = {
        "transformer.wte.weight": np.linspace(-0.2, 0.2, vocab * d_model, dtype=np.float32).reshape(vocab, d_model),
        "transformer.wpe.weight": np.linspace(0.05, 0.25, block * d_model, dtype=np.float32).reshape(block, d_model),
        "transformer.ln_f.weight": np.ones(d_model, dtype=np.float32),
        "transformer.ln_f.bias": np.zeros(d_model, dtype=np.float32),
        "lm_head.weight": np.linspace(-0.3, 0.4, vocab * d_model, dtype=np.float32).reshape(vocab, d_model),
    }
    for ln in ("ln_1", "ln_2"):
        state[f"transformer.h.0.{ln}.weight"] = np.ones(d_model, dtype=np.float32)
        state[f"transformer.h.0.{ln}.bias"] = np.zeros(d_model, dtype=np.float32)
    for proj in ("query", "key", "value"):
        state[f"transformer.h.0.attn.c_attn.weight_h0_{proj}"] = np.linspace(
            -0.4, 0.4, d_model * d_model, dtype=np.float32
        ).reshape(d_model, d_model)
        state[f"transformer.h.0.attn.c_attn.bias_h0_{proj}"] = np.linspace(
            -0.1, 0.1, d_model, dtype=np.float32
        )
    state["transformer.h.0.attn.c_proj.weight"] = np.linspace(
        -0.5, 0.5, d_model * d_model, dtype=np.float32
    ).reshape(d_model, d_model)
    state["transformer.h.0.attn.c_proj.bias"] = np.linspace(-0.2, 0.2, d_model, dtype=np.float32)
    fc_rows = [
        np.linspace(-0.05 * (idx + 1), 0.05 * (idx + 1), d_model, dtype=np.float32)
        for idx in range(mlp_dim)
    ]
    state["transformer.h.0.mlp.c_fc.weight"] = np.stack(fc_rows, axis=0)
    state["transformer.h.0.mlp.c_fc.bias"] = np.linspace(-0.3, 0.3, mlp_dim, dtype=np.float32)
    state["transformer.h.0.mlp.c_proj.weight"] = np.linspace(
        -0.3, 0.3, d_model * mlp_dim, dtype=np.float32
    ).reshape(d_model, mlp_dim)
    state["transformer.h.0.mlp.c_proj.bias"] = np.linspace(-0.15, 0.15, d_model, dtype=np.float32)
    return {
        "model_args": {
            "n_layer": 1,
            "n_head": 1,
            "n_embd": d_model,
            "block_size": block,
            "vocab_size": vocab,
            "layer_norm_epsilon": 1e-5,
        },
        "state_dict": state,
    }


def test_simulator_matches_m4e_at_d_model_16_single_token_prefill():
    """At d_model=16 with seq_len=1 prefill (no decode, no tiling, no fc2
    streaming), simulator-backed logits should match M4-E within FP16
    ULP. Any divergence isolates the bug to the M3 foundation or M4-A
    spill mechanism."""
    payload = _one_layer_gpt2_payload(d_model=16)
    token_id = 3

    # Simulator path.
    bundle = build_stage3_tiny_decoder_bundle(
        payload, ptq_preset="weight_only_int8", smoke_decode_steps=0,
    )
    runner = HostRunner(bundle.build.bundle, logits_dtype=np.float32)
    sim_logits = runner.run_prefill([token_id])

    # M4-E path.
    ref = NanoGPTW8A32SimulatorReference(payload)
    ref_logits = ref.run_prefill([token_id])

    vocab = int(payload["model_args"]["vocab_size"])
    sim_active = sim_logits[:vocab]
    ref_active = ref_logits[:vocab]

    abs_diff = np.abs(sim_active - ref_active)
    max_abs = float(np.max(np.abs(ref_active)))
    rel_diff = abs_diff / max(max_abs, 1e-9)

    print(f"\n[bisect d_model=16 prefill] sim_max={float(np.abs(sim_active).max()):.6f}, "
          f"ref_max={max_abs:.6f}")
    print(f"  per-element max abs diff = {float(abs_diff.max()):.6e}")
    print(f"  per-element max rel diff = {float(rel_diff.max()):.6e}")
    print(f"  sim_active[:8] = {sim_active[:8]}")
    print(f"  ref_active[:8] = {ref_active[:8]}")

    # Don't assert yet — just print the diagnostic.


def _bisect_run(d_model, decode_steps=2):
    """Run simulator vs M4-E at a given d_model and print diff. Returns
    (max_abs_diff, max_rel_diff) for the final-step logits."""
    payload = _one_layer_gpt2_payload(d_model=d_model)
    bundle = build_stage3_tiny_decoder_bundle(
        payload, ptq_preset="weight_only_int8",
        smoke_decode_steps=decode_steps,
    )
    runner = HostRunner(bundle.build.bundle, logits_dtype=np.float32)
    ref = NanoGPTW8A32SimulatorReference(payload)

    sim_pf = runner.run_prefill([3])
    ref_pf = ref.run_prefill([3])
    vocab = int(payload["model_args"]["vocab_size"])
    diff_pf = np.abs(sim_pf[:vocab] - ref_pf[:vocab])
    print(f"[d_model={d_model}] prefill: max_abs_diff={float(diff_pf.max()):.6e}, "
          f"max_rel={float(diff_pf.max() / max(float(np.abs(ref_pf[:vocab]).max()), 1e-9)):.6e}")

    if decode_steps >= 1:
        sim_d1 = runner.run_decode_step(token_id=5, position=1)
        ref_d1 = ref.run_decode_step(5, 1)
        diff_d1 = np.abs(sim_d1[:vocab] - ref_d1[:vocab])
        print(f"[d_model={d_model}] decode1: max_abs_diff={float(diff_d1.max()):.6e}, "
              f"max_rel={float(diff_d1.max() / max(float(np.abs(ref_d1[:vocab]).max()), 1e-9)):.6e}")
        return diff_d1.max(), diff_d1.max() / max(float(np.abs(ref_d1[:vocab]).max()), 1e-9)
    return diff_pf.max(), diff_pf.max() / max(float(np.abs(ref_pf[:vocab]).max()), 1e-9)


def test_simulator_vs_m4e_at_d_model_192_no_spill():
    """At d_model=192 (12 KB FP32 tile, BELOW the 16 KB spill threshold,
    weights all under WBUF), simulator and M4-E should still agree
    closely. Isolates the M3 foundation at modest scale."""
    _bisect_run(d_model=192)


def test_simulator_vs_m4e_at_d_model_384_spill_and_tiling():
    """At d_model=384 (24 KB FP32 tile, ABOVE threshold; mlp_dim=1536
    → fc1/fc2 weights 576 KB > WBUF → M4-C tiling fires; M4-A spill
    fires too). Isolates spill + tiling effects."""
    _bisect_run(d_model=384)
