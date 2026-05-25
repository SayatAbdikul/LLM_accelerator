"""W4A16 weight quantization — AWQ → GPTQ-inject → AdaRound → bias-correct.

This is the production W4 stack identified by the W4A16 Phase 1 quality
work (2026-05-23 / 24, see [[w4a16-phase1-quality]] memory) plus the
Tier-1 refinements landed 2026-05-24:

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

  2. **GPTQ with act-order** (Frantar et al. 2022, §3.5 act-order trick,
     already implemented in `taccel/quantizer/quantize.py`) on each
     *block* matmul weight using captured per-matmul-input FP32
     activations. Per-channel symmetric INT4 with column-wise error
     propagation via the inverse Hessian `H = X^T X`. The
     ``act_order=True`` reordering processes the high-Hessian-mass
     columns first, while the rounding-error budget is largest — a
     standard W4-mandatory GPTQ trick worth ~0.5–1.5 PPL on small LMs.
     Per-source Hessians (ln1, concat, ln2, gelu) are computed ONCE per
     layer and reused via ``precomputed_hessian`` so the 65K-token
     calibration corpus stays cheap (~4 min vs ~25 min without sharing).

  3. **AdaRound greedy refinement** (Nagel et al. 2020) on each
     GPTQ-quantized weight: for the near-half-LSB candidates, flip the
     rounding direction if it reduces the calibration-output MSE. Uses
     the same precomputed gram as GPTQ for the per-source activations,
     so the per-channel inner loop is cheap.

  4. **Bias correction** (AdaRound paper §4.3 / ZeroQuant §3.2): for
     each block matmul, compute the per-output-channel mean shift
     between FP32 and weight-only-QDQ outputs on the captured
     activations, then add that shift into the matmul's bias. Cancels
     the systematic W4 rounding mean error before it propagates through
     LN normalization. Applied to attn.c_proj / mlp.c_fc / mlp.c_proj
     (the three block matmuls with biases — per-head Q/K/V biases stay
     untouched because the head shape doesn't expose a per-head bias
     vector in the same way).

  5. **lm_head is NOT GPTQ'd / AdaRound'd / bias-corrected** and stays
     at W8 (the sim's ``lm_head_bitwidth=8`` default). Per the
     layer-ablation 2026-05-23, W4 lm_head alone gives 40,306 PPL on
     GPT-2 124M — the per-vocab-token row scale gets dominated by one
     outlier and W4 zeros out the discriminative tail. Standard W4
     practice (AWQ paper, GPTQ paper, llama.cpp default) keeps
     lm_head ≥ W8.

  6. **QuaRot is NOT applied**. QuaRot helps W8 by 4× but HURTS W4 by
     1.4–9× (intrinsic; rotation destroys the "natural" trained per-row
     structure that W4's coarse 8-level grid relies on, and the QuaRot
     paper's W4 wins are specifically W4A8 where the rotation helps
     INT8 activation outliers — our W4A16 path already handles those
     with dynamic per-tile max-abs).

The α=0.40 AWQ default was chosen by the empirical sweep 2026-05-24:
α=0.40 → 63.04 PPL (best on the old 32×64 calibration), α=0.50 →
65.03 (canonical AWQ), α=0.55 → 63.97, α=0.45 → 65.45, α=0.75 → 96.53.
The optimum curve is shallow around 0.4–0.55. **AWQ on lm_head is
CRITICAL** even though lm_head is W8 — excluding it 3-4× regresses
PPL (212-238 vs 63-65), because the AWQ fold into ln_f changes the
lm_head INPUT activation distribution and improves the dynamic INT8
act quant on that input.

End-to-end gate (GPT-2 124M, 257-tok WikiText):
  Production = AWQ(α=0.40) + GPTQ(act_order) + AdaRound + bias-correct +
  Stage5 QKT/attn_v calib + lm_head_bitwidth=8.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Public name for the per-weight override map: sd_key → (int8_tensor_transposed, fp16_scales).
# The transpose is required because GPTQ returns shape [out, in] but the
# NanoGPTW8A16SimulatorReference stores weights as [in, out]. The scales are
# per-output-channel either way.
WeightOverrides = Dict[str, Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Default calibration shape (Tier-1 2026-05-24 bump).
#
# The W4 production stack used to default to `n_seqs=32, seq_len=64` =
# 2048 tokens — about 128× SMALLER than GPTQ paper's `128 × 2048 = 262K`
# tokens. The Hessian `H = X^T X` over a 768- or 3072-wide input dimension
# is rank-deficient on that small a corpus and the GPTQ paper's `dead`
# column branch is hot. The Tier-1 default `n_seqs=128, seq_len=512`
# (65,536 tokens, 32× scale-up) lands in the GPTQ paper's regime.
#
# Runtime is held constant by `precomputed_hessian` / `precomputed_gram`
# sharing across per-layer activation sources — the bumped corpus costs
# ~4 min instead of the ~25 min the naive per-weight recomputation would.
# ---------------------------------------------------------------------------
_DEFAULT_N_SEQS = 128
_DEFAULT_SEQ_LEN = 512


def apply_w4_awq_gptq(
    payload: dict,
    token_ids: Sequence[int],
    *,
    alpha: float = 0.40,
    bitwidth: int = 4,
    n_seqs: int = _DEFAULT_N_SEQS,
    seq_len: int = _DEFAULT_SEQ_LEN,
    percdamp: float = 0.01,
    blocksize: int = 128,
    awq_targets: Sequence[str] = ("c_attn", "c_fc", "lm_head"),
    act_order: bool = True,
    apply_adaround: bool = True,
    apply_bias_correction: bool = True,
    hg_cache_path: Optional[Path] = None,
) -> Tuple[WeightOverrides, dict]:
    """Apply the W4 production stack to ``payload``: AWQ in place, then
    GPTQ + AdaRound + bias correction to produce per-weight overrides.

    The state_dict in ``payload`` is mutated by the AWQ step (per-channel
    LN gamma/bias fold + weight input-col scaling) and, when
    ``apply_bias_correction=True``, by the bias-correction step (matmul
    biases get the per-output-channel mean-shift added). Then GPTQ +
    AdaRound run on each block matmul weight using FP32 activations
    captured AFTER the AWQ mutation. The returned override map is
    suitable for ``NanoGPTW8A16SimulatorReference(weight_overrides=...)``.

    ``lm_head.weight`` is NOT included in the override map — the simulator
    handles it via its own ``lm_head_bitwidth`` (default 8 under W4).

    Args:
        payload: Loaded GPT-2 nanoGPT checkpoint dict (``state_dict`` is mutated).
        token_ids: Tokenized calibration text. Use a wikitext-2-train-sized
            corpus (≥ ``n_seqs * seq_len`` tokens) for the full benefit of the
            ``_DEFAULT_N_SEQS × _DEFAULT_SEQ_LEN`` corpus default; if smaller,
            ``build_calibration_seqs_from_token_ids`` will slide-window
            (and effectively repeat) the tokens.
        alpha: AWQ blend exponent. ``0.40`` is the empirical sweet spot on
            GPT-2 124M (2026-05-24 sweep); canonical AWQ uses ``0.50``.
        bitwidth: Weight bitwidth for the block matmuls. ``4`` is the W4A16
            production target; ``3`` works but is below the quality gate
            on this model.
        n_seqs, seq_len: Calibration corpus shape (default 128 × 512 =
            65,536 tokens, GPTQ-paper-regime). Older runs used 32 × 64 =
            2048 tokens; the Tier-1 bump lifts Hessian quality without
            inflating runtime thanks to per-source ``precomputed_hessian``
            sharing.
        percdamp, blocksize: GPTQ hyperparameters (Frantar defaults).
        awq_targets: Which LN-fed matmuls to AWQ. The default
            ``(c_attn, c_fc, lm_head)`` is REQUIRED to pass the gate; the
            ablation 2026-05-24 showed excluding ``lm_head`` 3-4× regresses
            PPL (212-238 vs 63) even though lm_head stays at W8.
        act_order: Pass ``act_order=True`` into ``gptq_quantize`` — the
            standard GPTQ §3.5 reorder-by-Hessian-diag trick. Default True
            on the W4 path; set False to reproduce older measurements.
        apply_adaround: Run ``adaround_greedy`` on each GPTQ-quantized
            weight using the precomputed per-source gram. Default True;
            set False to isolate GPTQ's contribution.
        apply_bias_correction: After GPTQ + AdaRound, add the per-output-
            channel mean shift between FP32 and quantized-weight matmul
            outputs into each block matmul's bias. Default True; set False
            to isolate weight-only effects.
        hg_cache_path: Optional ``.npz`` path used by ``runtime._hg_cache``
            to cache the per-source ``(hessian, x_mean)`` tensors. When
            provided and the file exists with matching content, the
            activation-capture + Hessian-compute phase is skipped
            entirely (minutes saved on the W4 stack). When provided and
            missing, the phase runs and the result is saved on the way
            out. The caller computes the path via
            ``_hg_cache.compute_cache_key`` + ``_hg_cache.cache_path_for``
            from the upstream (fixture, calibration_text, AWQ params).

    Returns:
        ``(overrides, diagnostics)``:
        - ``overrides``: ``sd_key → (int8_tensor_transposed, fp16_scales)``,
          ready for ``NanoGPTW8A16SimulatorReference(weight_overrides=...)``.
          Lm_head is NOT in this map.
        - ``diagnostics``: counts (weights_quantized, n_calib_seqs) and timing
          (capture_s, gptq_s, adaround_s, bias_correction_s, awq_keys_mutated).
    """
    import time

    from taccel.quantizer.quantize import adaround_greedy, gptq_quantize
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
    # Steps 2+3 — Capture per-matmul-input FP32 activations on the post-AWQ
    # payload, then precompute one Hessian (`H = X^T X * 2/N`) and one gram
    # (`X^T X / N`) PER (layer, source) so the 432 per-head Q/K/V + 36
    # block-matmul GPTQ + AdaRound calls don't redundantly recompute
    # ~5-second matmuls on the 65K-token corpus.
    #
    # Bias correction (step 5) also wants the captured activations, but only
    # through ``mean(X @ (W - W_dq).T, axis=0)``. By linearity that equals
    # ``mean(X, axis=0) @ (W - W_dq).T`` — so we only need the per-source
    # mean vector ``x_means[name]`` downstream, never the full activation
    # matrix.
    #
    # When ``hg_cache_path`` is set and the file exists with matching
    # content, we skip the capture + Hessian compute entirely and load
    # ``(hessians, grams, x_means)`` from the cache.
    # ---------------------------------------------------------------------
    cache_hit = False
    hessians: Dict[str, np.ndarray] = {}
    grams: Dict[str, np.ndarray] = {}
    x_means: Dict[str, np.ndarray] = {}
    n_calib_seqs_recorded = 0
    if hg_cache_path is not None:
        from . import _hg_cache as _hg
        loaded = _hg.try_load(Path(hg_cache_path))
        if loaded is not None:
            hessians, grams, x_means = loaded
            cache_hit = True
    t_capture = 0.0
    t_hessian = 0.0
    if not cache_hit:
        t0 = time.time()
        seqs = build_calibration_seqs_from_token_ids(
            token_ids, n_seqs=n_seqs, seq_len=seq_len,
        )
        n_calib_seqs_recorded = len(seqs)
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
        t_capture = time.time() - t0

        t0 = time.time()
        for name, X in activations.items():
            X64 = X.astype(np.float64)
            n = float(X64.shape[0])
            XtX = X64.T @ X64
            hessians[name] = XtX * (2.0 / n)  # GPTQ's convention
            grams[name] = (XtX / n).astype(np.float32)  # AdaRound's convention
            x_means[name] = X.mean(axis=0).astype(np.float32)
        t_hessian = time.time() - t0

        if hg_cache_path is not None:
            from . import _hg_cache as _hg
            _hg.save(hessians, x_means, Path(hg_cache_path))
    else:
        # Cache hit: the corpus shape (n_calib_seqs, n_calib_tokens) is a
        # function of (token_ids, n_seqs, seq_len). Recompute the lightweight
        # sequence count for the diagnostics — `build_calibration_seqs_from_token_ids`
        # is cheap (a slide-window over token_ids) and the result is
        # deterministic given the inputs.
        seqs = build_calibration_seqs_from_token_ids(
            token_ids, n_seqs=n_seqs, seq_len=seq_len,
        )
        n_calib_seqs_recorded = len(seqs)

    # ---------------------------------------------------------------------
    # Step 4 — GPTQ + AdaRound per block weight; collect overrides
    # keyed by sd-key. The transposition `q.T` converts gptq's [out, in]
    # output to the simulator's [in, out] storage convention
    # (`_quant_w_per_channel` uses axis=0 and produces [K, N] for weights
    # given as [K, N]).
    #
    # `weight_quants[key] = (q_unquantized_W_fp32, q_int8_post_adaround,
    # scales_fp32)` is retained for the bias-correction step (so it doesn't
    # have to re-quantize from the override int8 + per-channel scale).
    # ---------------------------------------------------------------------
    overrides: WeightOverrides = {}
    weight_quants: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    t_gptq = 0.0
    t_adaround = 0.0
    weights_quantized = 0
    for L in range(n_layer):
        ln1_src = f"block{L}_ln1"
        concat_src = f"block{L}_concat"
        ln2_src = f"block{L}_ln2"
        gelu_src = f"block{L}_gelu"

        # Per-head Q / K / V projections (share ln1 Hessian/gram).
        H_ln1 = hessians[ln1_src]
        G_ln1 = grams[ln1_src]
        for hd in range(n_head):
            for suffix in ("query", "key", "value"):
                key = f"transformer.h.{L}.attn.c_attn.weight_h{hd}_{suffix}"
                W = _to_f32(sd[key])
                t0 = time.time()
                q, s = gptq_quantize(
                    W, calibration_inputs=None,
                    bitwidth=bitwidth, percdamp=percdamp, blocksize=blocksize,
                    act_order=act_order, precomputed_hessian=H_ln1,
                )
                t_gptq += time.time() - t0
                if apply_adaround:
                    t0 = time.time()
                    q = adaround_greedy(
                        W, q, s.astype(np.float32), calibration_inputs=None,
                        bitwidth=bitwidth, precomputed_gram=G_ln1,
                    )
                    t_adaround += time.time() - t0
                overrides[key] = (q.T.astype(np.int8), s.astype(np.float16))
                weight_quants[key] = (W, q, s.astype(np.float32))
                weights_quantized += 1

        # Output projection, fc1, fc2 (each has its own activation source).
        for key, src in (
            (f"transformer.h.{L}.attn.c_proj.weight", concat_src),
            (f"transformer.h.{L}.mlp.c_fc.weight", ln2_src),
            (f"transformer.h.{L}.mlp.c_proj.weight", gelu_src),
        ):
            W = _to_f32(sd[key])
            t0 = time.time()
            q, s = gptq_quantize(
                W, calibration_inputs=None,
                bitwidth=bitwidth, percdamp=percdamp, blocksize=blocksize,
                act_order=act_order, precomputed_hessian=hessians[src],
            )
            t_gptq += time.time() - t0
            if apply_adaround:
                t0 = time.time()
                q = adaround_greedy(
                    W, q, s.astype(np.float32), calibration_inputs=None,
                    bitwidth=bitwidth, precomputed_gram=grams[src],
                )
                t_adaround += time.time() - t0
            overrides[key] = (q.T.astype(np.int8), s.astype(np.float16))
            weight_quants[key] = (W, q, s.astype(np.float32))
            weights_quantized += 1

    # ---------------------------------------------------------------------
    # Step 5 — Bias correction (weight-only-QDQ, captured-activation mean
    # shift). For each block matmul with a bias, compute:
    #     err[i] = E_x[ (x @ W.T)[:, i] - (x @ W_qdq.T)[:, i] ]
    #            = mean( x @ (W - W_qdq).T, axis=0 )[i]
    # and add `err` into the bias in place. Per-head Q/K/V biases are
    # left untouched — their bias vector shape doesn't match a single
    # per-head W's output dim in the same way the block matmul biases do,
    # and the per-head Q/K/V projections feed an attention non-linearity
    # downstream that absorbs much of the mean shift.
    # ---------------------------------------------------------------------
    t_bias = 0.0
    bias_corrected = 0
    if apply_bias_correction:
        t0 = time.time()
        # `_to_f32` returns a numpy array for torch / numpy state-dict
        # values uniformly; we mutate the sd entry back to torch below.
        import torch as _torch
        for L in range(n_layer):
            for w_key, src in (
                (f"transformer.h.{L}.attn.c_proj.weight", f"block{L}_concat"),
                (f"transformer.h.{L}.mlp.c_fc.weight", f"block{L}_ln2"),
                (f"transformer.h.{L}.mlp.c_proj.weight", f"block{L}_gelu"),
            ):
                bias_key = w_key.replace(".weight", ".bias")
                if bias_key not in sd:
                    continue
                W, q, s_fp32 = weight_quants[w_key]
                # Dequantize the post-GPTQ-+-AdaRound integer weight back to
                # FP32 with the per-output-channel scale. Shape [out, in].
                W_dq = q.astype(np.float32) * s_fp32.reshape(-1, 1)
                # Per-channel mean shift. By linearity, mean(X @ M.T, axis=0)
                # equals mean(X, axis=0) @ M.T — so we only need the per-input-
                # channel activation mean (cached by `_hg_cache`), not the full
                # [N, in] matrix.
                err = (x_means[src] @ (W - W_dq).T).astype(np.float32)
                # Add the shift into the matmul bias and store back in the
                # original dtype (typically FP32 in nanoGPT checkpoints).
                b_old = _to_f32(sd[bias_key])
                b_new = (b_old + err).astype(np.float32)
                orig = sd[bias_key]
                target_dtype = orig.dtype if hasattr(orig, "dtype") else _torch.float32
                sd[bias_key] = _torch.from_numpy(
                    np.ascontiguousarray(b_new)
                ).to(dtype=target_dtype)
                bias_corrected += 1
        t_bias = time.time() - t0

    diagnostics = {
        "weights_quantized": weights_quantized,
        "n_calib_seqs": n_calib_seqs_recorded,
        "n_calib_tokens": int(n_calib_seqs_recorded * seq_len),
        "awq_keys_mutated": len(awq_mutated),
        "hg_cache_hit": bool(cache_hit),
        "capture_s": round(t_capture, 1),
        "hessian_s": round(t_hessian, 1),
        "gptq_s": round(t_gptq, 1),
        "adaround_s": round(t_adaround, 1),
        "bias_correction_s": round(t_bias, 1),
        "bias_corrected": bias_corrected,
        "alpha": float(alpha),
        "bitwidth": int(bitwidth),
        "act_order": bool(act_order),
        "apply_adaround": bool(apply_adaround),
        "apply_bias_correction": bool(apply_bias_correction),
    }
    return overrides, diagnostics
