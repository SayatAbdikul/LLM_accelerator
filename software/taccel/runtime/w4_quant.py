"""W4A16 weight quantization — AWQ → GPTQ-inject pipeline.

This is the production W4 stack identified by the W4A16 Phase 1 quality
work (2026-05-23 / 24, see [[w4a16-phase1-quality]] memory):

  1. **AWQ** (Lin et al. MLSys 2024) applied in place to `payload["state_dict"]`:
     scales the input cols of c_attn / c_fc / lm_head by per-channel
     activation magnitudes, folds the inverse into the upstream LN's
     gamma + bias. Mathematically a no-op in FP32; reduces the per-channel
     weight-magnitude spread so subsequent INT4 quantization carries
     less per-channel rounding noise. Also makes the LN OUTPUT activation
     distribution more per-channel-uniform — which directly improves the
     dynamic per-tile INT8 activation quant on the LN-fed matmul input
     (this second effect is why AWQ helps W4 here even though it tested
     negative on W8 per commit `be003f9`).

  2. **GPTQ** (Frantar et al. 2022, already implemented in
     `taccel/quantizer/quantize.py`) on each *block* matmul weight using
     captured per-matmul-input FP32 activations. Per-channel symmetric
     INT4 with column-wise error propagation via the inverse Hessian
     `H = X^T X`. Produces `(int8-storage, fp16-scales)` tuples that are
     injected directly into `NanoGPTW8A16SimulatorReference` via its
     `weight_overrides` parameter — bypassing the sim's internal RTN
     re-quantization, which would otherwise drift in ~0.5% of values
     and break GPTQ's careful per-element rounding chain.

  3. **lm_head is NOT GPTQ'd** and stays at W8 (the sim's
     `lm_head_bitwidth=8` default). Per the layer-ablation 2026-05-23,
     W4 lm_head alone gives 40,306 PPL on GPT-2 124M — the per-vocab-token
     row scale gets dominated by one outlier and W4 zeros out the
     discriminative tail. Standard W4 practice (AWQ paper, GPTQ paper,
     llama.cpp default) keeps lm_head ≥ W8.

  4. **QuaRot is NOT applied**. QuaRot helps W8 by 4× but HURTS W4 by
     1.4–9× (intrinsic; rotation destroys the "natural" trained per-row
     structure that W4's coarse 8-level grid relies on, and the QuaRot
     paper's W4 wins are specifically W4A8 where the rotation helps
     INT8 activation outliers — our W4A16 path already handles those
     with dynamic per-tile max-abs).

The α=0.40 AWQ default was chosen by the empirical sweep 2026-05-24:
α=0.40 → 63.04 PPL (best), α=0.50 → 65.03 (canonical AWQ), α=0.55 →
63.97, α=0.45 → 65.45, α=0.75 → 96.53. The optimum curve is shallow
around 0.4–0.55. **AWQ on lm_head is CRITICAL** even though lm_head is
W8 — excluding it 3-4× regresses PPL (212-238 vs 63-65), because the
AWQ fold into ln_f changes the lm_head INPUT activation distribution
and improves the dynamic INT8 act quant on that input.

End-to-end gate (GPT-2 124M, 257-tok WikiText):
  W4 AWQ(α=0.40) + GPTQ-inject + Stage5 QKT/attn_v calib +
  lm_head_bitwidth=8 = **63.04 PPL** (≤ 65 gate; vs W8A16+QuaRot 56.23 PPL
  baseline; vs FP32 ceiling 53.42 PPL).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


# Public name for the per-weight override map: sd_key → (int8_tensor_transposed, fp16_scales).
# The transpose is required because GPTQ returns shape [out, in] but the
# NanoGPTW8A16SimulatorReference stores weights as [in, out]. The scales are
# per-output-channel either way.
WeightOverrides = Dict[str, Tuple[np.ndarray, np.ndarray]]


def apply_w4_awq_gptq(
    payload: dict,
    token_ids: Sequence[int],
    *,
    alpha: float = 0.40,
    bitwidth: int = 4,
    n_seqs: int = 32,
    seq_len: int = 64,
    percdamp: float = 0.01,
    blocksize: int = 128,
    awq_targets: Sequence[str] = ("c_attn", "c_fc", "lm_head"),
) -> Tuple[WeightOverrides, dict]:
    """Apply the W4 production stack to ``payload``: AWQ in place, then
    GPTQ to produce per-weight overrides.

    The state_dict in ``payload`` is mutated by the AWQ step (per-channel
    LN gamma/bias fold + weight input-col scaling). Then GPTQ is run on
    each block matmul weight using FP32 activations captured AFTER the
    AWQ mutation. The returned override map is suitable for
    ``NanoGPTW8A16SimulatorReference(weight_overrides=...)``.

    ``lm_head.weight`` is NOT included in the override map — the simulator
    handles it via its own ``lm_head_bitwidth`` (default 8 under W4).

    Args:
        payload: Loaded GPT-2 nanoGPT checkpoint dict (``state_dict`` is mutated).
        token_ids: Tokenized calibration text.
        alpha: AWQ blend exponent. ``0.40`` is the empirical sweet spot on
            GPT-2 124M (2026-05-24 sweep); canonical AWQ uses ``0.50``.
        bitwidth: Weight bitwidth for the block matmuls. ``4`` is the W4A16
            production target; ``3`` works but is below the quality gate
            on this model.
        n_seqs, seq_len: Calibration corpus shape (32×64 = 2048 tokens by default).
        percdamp, blocksize: GPTQ hyperparameters (Frantar defaults).
        awq_targets: Which LN-fed matmuls to AWQ. The default
            ``(c_attn, c_fc, lm_head)`` is REQUIRED to pass the gate; the
            ablation 2026-05-24 showed excluding ``lm_head`` 3-4× regresses
            PPL (212-238 vs 63) even though lm_head stays at W8.

    Returns:
        ``(overrides, diagnostics)``:
        - ``overrides``: ``sd_key → (int8_tensor_transposed, fp16_scales)``,
          ready for ``NanoGPTW8A16SimulatorReference(weight_overrides=...)``.
          Lm_head is NOT in this map.
        - ``diagnostics``: counts (weights_quantized, n_calib_seqs) and timing
          (capture_s, gptq_s, awq_keys_mutated).
    """
    import time

    from taccel.quantizer.quantize import gptq_quantize
    from .calibration.adapters import apply_awq_from_token_ids
    from .calibration.scales import build_calibration_seqs_from_token_ids
    from .fake_quant_reference import _fp32_forward, _to_f32

    sd = payload["state_dict"]
    model_args = payload["model_args"]
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])

    # ---------------------------------------------------------------------
    # Step 1 — AWQ in place. Folds per-channel activation magnitude into
    # the upstream LN's gamma/bias; scales weight input columns by the same
    # factor. Math-identity in FP32; reduces post-AWQ per-channel weight
    # max-abs spread (for the W4 quant step) AND per-channel activation
    # spread at the LN output (for the dynamic INT8 act quant during matmul).
    # ---------------------------------------------------------------------
    awq_mutated, _awq_info = apply_awq_from_token_ids(
        payload, token_ids, n_seqs=n_seqs, seq_len=seq_len,
        alpha=alpha, target_modules=tuple(awq_targets),
    )

    # ---------------------------------------------------------------------
    # Step 2 — Capture per-matmul-input FP32 activations on the post-AWQ
    # payload. These feed GPTQ's Hessian H = X^T X.
    # ---------------------------------------------------------------------
    t_capture = time.time()
    seqs = build_calibration_seqs_from_token_ids(
        token_ids, n_seqs=n_seqs, seq_len=seq_len,
    )
    needed = {
        f"block{L}_{x}"
        for L in range(n_layer)
        for x in ("ln1", "concat", "ln2", "gelu")
    }
    captured: Dict[str, List[np.ndarray]] = defaultdict(list)
    for tids in seqs:
        out = _fp32_forward(sd, model_args, tids)
        for name in needed:
            arr = out.get(name)
            if arr is not None:
                captured[name].append(np.asarray(arr, dtype=np.float32))
    activations: Dict[str, np.ndarray] = {
        k: np.concatenate(v, axis=0) for k, v in captured.items()
    }
    t_capture = time.time() - t_capture

    # ---------------------------------------------------------------------
    # Step 3 — GPTQ per block weight; collect overrides keyed by sd-key.
    # The transposition `q.T` converts gptq's [out, in] output to the
    # simulator's [in, out] storage convention (`_quant_w_per_channel`
    # uses axis=0 and produces [K, N] for weights given as [K, N]).
    # ---------------------------------------------------------------------
    overrides: WeightOverrides = {}
    t_gptq = time.time()
    weights_quantized = 0
    for L in range(n_layer):
        ln1_a = activations[f"block{L}_ln1"]
        concat_a = activations[f"block{L}_concat"]
        ln2_a = activations[f"block{L}_ln2"]
        gelu_a = activations[f"block{L}_gelu"]

        # Per-head Q / K / V projections.
        for H in range(n_head):
            for suffix, acts in (("query", ln1_a), ("key", ln1_a), ("value", ln1_a)):
                key = f"transformer.h.{L}.attn.c_attn.weight_h{H}_{suffix}"
                W = _to_f32(sd[key])
                q, s = gptq_quantize(
                    W, calibration_inputs=acts,
                    bitwidth=bitwidth, percdamp=percdamp, blocksize=blocksize,
                )
                overrides[key] = (q.T.astype(np.int8), s.astype(np.float16))
                weights_quantized += 1

        # Output projection, fc1, fc2.
        for key, acts in (
            (f"transformer.h.{L}.attn.c_proj.weight", concat_a),
            (f"transformer.h.{L}.mlp.c_fc.weight", ln2_a),
            (f"transformer.h.{L}.mlp.c_proj.weight", gelu_a),
        ):
            W = _to_f32(sd[key])
            q, s = gptq_quantize(
                W, calibration_inputs=acts,
                bitwidth=bitwidth, percdamp=percdamp, blocksize=blocksize,
            )
            overrides[key] = (q.T.astype(np.int8), s.astype(np.float16))
            weights_quantized += 1
    t_gptq = time.time() - t_gptq

    diagnostics = {
        "weights_quantized": weights_quantized,
        "n_calib_seqs": len(seqs),
        "awq_keys_mutated": len(awq_mutated),
        "capture_s": round(t_capture, 1),
        "gptq_s": round(t_gptq, 1),
        "alpha": float(alpha),
        "bitwidth": int(bitwidth),
    }
    return overrides, diagnostics
