"""Reference-only PPL evaluator for the TurboQuant KV verification (Tier 1).

Mirrors the `weight_only_int8[/quarot]` reference path of
`evaluate_gpt2_perplexity` (rotate → calibrate → NumPy reference) but skips
the compiled golden bundle (irrelevant to Tier 1 and ~slow).

Split into `prepare()` (load-fresh + rotate + calibrate — the expensive part,
done ONCE per base preset) and `ppl_for(prepared, kv_quant)` (cheap reference
forward — run MANY times across the sweep grid). QuaRot mutates
`payload["state_dict"]` in place, so each `prepare()` loads a fresh payload;
`ppl_for` never mutates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Prepared:
    payload: dict
    scales: Optional[dict]
    eval_ids: list
    targets: list
    vocab: int
    preset_name: str
    # W4A16 plan (2026-05-23). Default 8 keeps every existing call site
    # byte-identical to the W8A16 baseline; W4 presets set this to 4 and
    # the simulator reference per-channel-quants weights to [-8, +7].
    weight_bitwidth: int = 8
    # lm_head specifically — defaults to max(weight_bitwidth, 8). The
    # ablation in [[w4a16-phase1-quality]] shows W4 lm_head is the entire
    # PPL killer (548× difference); keep it at W8 under W4 block weights.
    lm_head_bitwidth: Optional[int] = None
    # W4A16 production AWQ+GPTQ override map (2026-05-24 plan Phase 1.8).
    # When the preset has `apply_awq_gptq=True`, `prepare()` populates this
    # with the per-weight (int8_tensor, fp16_scales) tuples produced by
    # `w4_quant.apply_w4_awq_gptq`, and `ppl_for()` passes them to the
    # simulator's `weight_overrides` parameter (bypassing internal RTN).
    weight_overrides: Optional[dict] = None


@dataclass
class RefPPLResult:
    perplexity: float
    nll_per_position: List[float]
    preset: str
    kv_quant_repr: str


def prepare(
    fixture: Path,
    tokenizer_dir: Path,
    eval_text: Path,
    *,
    max_tokens: int,
    ptq_preset: str = "weight_only_int8_quarot",
    calibration_text: Optional[Path] = None,
    calibration_n_seqs: int = 32,
    calibration_seq_len: int = 64,
    calibration_percentile: float = 99.9,
) -> Prepared:
    import torch

    from .calibration import (
        apply_quarot_rotation_from_token_ids,
        build_calibration_scales_from_token_ids,
    )
    from .gpt2_perplexity import (
        teacher_forced_inputs_and_targets,
        tokenize_text_file,
    )
    from .stage5_ptq import apply_stage5_ptq_scale_policy, resolve_stage5_ptq_preset

    payload = torch.load(fixture, map_location="cpu")  # fresh: rotation mutates
    preset = resolve_stage5_ptq_preset(ptq_preset)
    eval_ids = list(tokenize_text_file(tokenizer_dir, eval_text, max_tokens=max_tokens))
    calib_ids = (
        list(tokenize_text_file(tokenizer_dir, calibration_text))
        if calibration_text and calibration_text.exists()
        else list(eval_ids)
    )
    if getattr(preset, "quarot_enabled", False):
        apply_quarot_rotation_from_token_ids(
            payload, calib_ids, seed=preset.quarot_seed, kind=preset.quarot_kind,
        )
    # W4A16 AWQ+GPTQ pipeline (plan Phase 1.8). Run AFTER QuaRot (so QuaRot's
    # state_dict mutations are in play for the AWQ activation capture; in
    # practice the production W4 preset doesn't combine the two — QuaRot
    # hurts W4 per [[w4a16-phase1-quality]] — but this ordering preserves
    # composability if someone wants to try) and BEFORE calibration (so
    # the calibration FP32 forward sees the AWQ-folded LN gamma/bias and
    # the resulting calibration scales match the deployed activation
    # distribution).
    weight_overrides = None
    if getattr(preset, "apply_awq_gptq", False):
        from .w4_quant import apply_w4_awq_gptq
        weight_overrides, _w4_info = apply_w4_awq_gptq(
            payload, calib_ids,
            alpha=float(getattr(preset, "awq_alpha", 0.40)),
            bitwidth=int(getattr(preset, "weight_bitwidth", 4)),
            n_seqs=calibration_n_seqs, seq_len=calibration_seq_len,
        )
    scales = None
    if getattr(preset, "weight_only_int8_calibrate", False):
        scales = build_calibration_scales_from_token_ids(
            payload, calib_ids,
            n_seqs=calibration_n_seqs, seq_len=calibration_seq_len,
            percentile=calibration_percentile,
            activation_percentile_overrides=(
                preset.activation_percentile_nodes or None
            ),
            hessian_gelu_blocks=preset.hessian_gelu_blocks,
        )
        scales = apply_stage5_ptq_scale_policy(scales, payload["model_args"], preset)
    _, targets = teacher_forced_inputs_and_targets(eval_ids)
    return Prepared(
        payload=payload, scales=scales, eval_ids=eval_ids, targets=targets,
        vocab=int(payload["model_args"]["vocab_size"]), preset_name=preset.name,
        weight_bitwidth=int(getattr(preset, "weight_bitwidth", 8)),
        lm_head_bitwidth=getattr(preset, "lm_head_bitwidth", None),
        weight_overrides=weight_overrides,
    )


def ppl_for(prepared: Prepared, kv_quant=None) -> RefPPLResult:
    """Cheap reference forward on an already prepared (rotated+calibrated)
    payload. Safe to call many times with different `kv_quant` — constructs
    a fresh reference each call and never mutates `prepared`."""
    from .gpt2_perplexity import perplexity_from_nlls, stable_cross_entropy
    from .w8a16_simulator_reference import NanoGPTW8A16SimulatorReference

    ref = NanoGPTW8A16SimulatorReference(
        prepared.payload, calibration_scales=prepared.scales, kv_quant=kv_quant,
        weight_bitwidth=prepared.weight_bitwidth,
        lm_head_bitwidth=prepared.lm_head_bitwidth,
        weight_overrides=prepared.weight_overrides,
    )
    logits = ref.run_teacher_forced(prepared.eval_ids)
    nlls = [
        float(stable_cross_entropy(np.asarray(r, np.float32), t,
                                   vocab_size=prepared.vocab))
        for r, t in zip(logits, prepared.targets)
    ]
    ppl, _ = perplexity_from_nlls(nlls)
    return RefPPLResult(
        perplexity=float(ppl), nll_per_position=nlls,
        preset=prepared.preset_name, kv_quant_repr=repr(kv_quant),
    )


def reference_ppl(
    fixture: Path,
    tokenizer_dir: Path,
    eval_text: Path,
    *,
    max_tokens: int,
    ptq_preset: str = "weight_only_int8_quarot",
    kv_quant=None,
    calibration_text: Optional[Path] = None,
    calibration_n_seqs: int = 32,
    calibration_seq_len: int = 64,
    calibration_percentile: float = 99.9,
) -> RefPPLResult:
    """One-shot convenience: prepare() + ppl_for()."""
    prepared = prepare(
        fixture, tokenizer_dir, eval_text, max_tokens=max_tokens,
        ptq_preset=ptq_preset, calibration_text=calibration_text,
        calibration_n_seqs=calibration_n_seqs,
        calibration_seq_len=calibration_seq_len,
        calibration_percentile=calibration_percentile,
    )
    return ppl_for(prepared, kv_quant=kv_quant)
