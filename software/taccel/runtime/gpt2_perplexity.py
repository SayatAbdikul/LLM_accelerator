"""Offline GPT-2 perplexity helpers for the Stage 5 golden-model gate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .calibration import (
    apply_awq_from_token_ids,
    apply_bias_correction_from_token_ids,
    apply_fc2_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_attn_scale_search_from_token_ids,
    apply_output_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_lm_head_scale_search_from_token_ids,
    apply_output_aware_mlp_scale_search_from_token_ids,
    apply_quarot_rotation_from_token_ids,
    build_calibration_scales_from_token_ids,
)

# Calibration budget constants — production callers use LARGE; tests pass small
# explicit values when they need a fast gate.
CALIBRATION_N_SEQS_LARGE = 64
CALIBRATION_SEQ_LEN_LARGE = 128
CALIBRATION_PERCENTILE_DEFAULT = 99.9

from .fake_quant_reference import NanoGPTFQReference
from .fp32_reference import NanoGPTFP32Reference, build_weight_only_int8_reference
from .host_runner import HostRunner
from .weight_only_host_runner import WeightOnlyHostRunner
from .stage5_ptq import (
    Stage5PTQPreset,
    apply_stage5_ptq_scale_policy,
    resolve_stage5_ptq_preset,
    stage5_default_ptq_preset_name,
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)
from .tiny_fixture import build_stage3_tiny_decoder_bundle

# Backward-compatible alias for older tools. Runtime default resolution should
# call stage5_default_ptq_preset_name() so the source of truth stays in
# stage5_ptq.py.
GPT2_DEFAULT_PTQ_PRESET = stage5_default_ptq_preset_name()


@dataclass
class GPT2PerplexityResult:
    golden_perplexity: float
    fake_quant_perplexity: float
    relative_delta: float
    token_count: int
    target_count: int
    golden_nll: float
    fake_quant_nll: float
    tokenizer_dir: str
    calibration_sha256: str
    eval_sha256: str
    ptq_preset: str
    # Phase 0A: FP32 ceiling captured BEFORE any PTQ state_dict mutations
    # (rotation, BC, etc.). This is the true pre-quantization perplexity for
    # the same eval text and decode budget as golden_perplexity /
    # fake_quant_perplexity, computed via NanoGPTFP32Reference. Default is
    # NaN if FP32 evaluation was skipped.
    fp32_perplexity: float = float("nan")
    fp32_nll: float = float("nan")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gpt2_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise RuntimeError("transformers is required for offline GPT-2 tokenization") from exc
    return GPT2TokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)


def tokenize_text_file(tokenizer_dir: Path, text_path: Path, *, max_tokens: int | None = None) -> List[int]:
    if not Path(text_path).exists():
        raise FileNotFoundError(text_path)
    tokenizer = load_gpt2_tokenizer(tokenizer_dir)
    text = Path(text_path).read_text(encoding="utf-8")
    token_ids = [int(tok) for tok in tokenizer.encode(text)]
    if max_tokens is not None:
        token_ids = token_ids[: int(max_tokens)]
    return token_ids


def teacher_forced_inputs_and_targets(token_ids: Sequence[int]) -> tuple[List[int], List[int]]:
    tokens = [int(tok) for tok in token_ids]
    if len(tokens) < 2:
        raise ValueError("perplexity evaluation requires at least two tokens")
    return tokens[:-1], tokens[1:]


def stable_cross_entropy(logits: np.ndarray, target: int, *, vocab_size: int) -> float:
    active = np.asarray(logits, dtype=np.float32)[: int(vocab_size)]
    if active.size == 0:
        raise ValueError("logits are empty")
    target_i = int(target)
    if target_i < 0 or target_i >= active.size:
        raise ValueError(f"target token {target_i} is outside vocab size {active.size}")
    row_max = float(np.max(active))
    shifted = active - np.float32(row_max)
    exp_shifted = np.exp(shifted.astype(np.float32)).astype(np.float32)
    logsumexp = row_max + float(np.log(exp_shifted.sum(dtype=np.float32)))
    return float(logsumexp - float(active[target_i]))


def perplexity_from_nlls(nlls: Sequence[float]) -> tuple[float, float]:
    if not nlls:
        raise ValueError("at least one NLL is required")
    mean_nll = float(np.mean(np.asarray(nlls, dtype=np.float64)))
    return float(math.exp(mean_nll)), mean_nll


def run_golden_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    ptq_preset: str | Stage5PTQPreset | None = None,
) -> List[np.ndarray]:
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    tiny = build_stage3_tiny_decoder_bundle(
        payload,
        smoke_decode_steps=max(0, len(inputs) - 1),
        calibration_scales=calibration_scales,
        ptq_preset=ptq_preset,
    )
    runner = HostRunner(tiny.build.bundle, logits_dtype=np.int8)
    logits: List[np.ndarray] = [runner.run_prefill([inputs[0]])]
    for position, token in enumerate(inputs[1:], start=1):
        logits.append(runner.run_decode_step(int(token), position))
    return logits


def run_fp32_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
) -> List[np.ndarray]:
    """Phase 0A: FP32-reference logits for every teacher-forced position.

    Uses NanoGPTFP32Reference.incremental_logits_trace, which mirrors the
    incremental decode pattern of the golden / fake_quant paths so that the
    resulting per-position NLLs are directly comparable. Caller is
    responsible for invoking this BEFORE any state_dict mutations
    (rotation, bias correction, etc.) so the returned logits represent the
    true pre-quantization ceiling.
    """
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    ref = NanoGPTFP32Reference(payload["state_dict"], payload["model_args"])
    return ref.incremental_logits_trace(inputs)


def run_weight_only_int8_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
) -> List[np.ndarray]:
    """W8A32 (Phase 1): INT8 weight QDQ + FP32 activations everywhere.

    Builds a `NanoGPTFP32Reference` with weights replaced by their per-row
    INT8 QDQ form (per-tensor for embeddings, per-channel for linear) via
    `build_weight_only_int8_reference`. The diagnostic at
    `software/tools/diagnose_weight_only_qdq_ceiling.py` proved this
    achieves 53.42 PPL on the production checkpoint at 257-tok / 256-ctx.

    Both this path and the diagnostic share the same helper, so per-token
    logits are bit-identical across the two paths (covered by
    `tests/runtime/test_weight_only_int8_perplexity.py`).

    Note: no calibration scales, no `Stage5PTQPreset` knobs. The W8A32
    preset (`weight_only_int8`) intentionally rejects every W8A8 transform
    at construction time — see `Stage5PTQPreset._preset` validation.
    """
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    ref = build_weight_only_int8_reference(payload, weight_mode="per_channel")
    return ref.incremental_logits_trace(inputs)


def run_weight_only_int8_golden_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
) -> List[np.ndarray]:
    """W8A32 Phase 3 option (b): host-runtime FP32 with INT8 weight storage.

    Builds a `WeightOnlyHostRunner` (mimicking the deployed `HostRunner`
    API but bypassing the simulator and the INT8 MXU entirely), then
    runs prefill + per-token decode through it. The runner internally
    wraps `build_weight_only_int8_reference` — the same helper the
    Phase 1 fake-quant path uses — so the resulting per-token logits
    are bit-identical to `run_weight_only_int8_teacher_forced_logits`.

    The relevant distinction is the **API contract**: this path goes
    through a runner that takes `(payload, weight_mode)` at
    construction time and exposes `run_prefill` + `run_decode_step` —
    mirroring how a deployed W8A32 bundle would be served. The Phase 1
    function calls the reference's `incremental_logits_trace` directly
    against the checkpoint; this function goes through the runner that
    represents the deployment surface. Both produce 53.42 PPL on the
    257-tok / 256-ctx gate; the difference is documentary.

    Option (c.1) and (c.2) — INT8-MXU-preserving W8A32 with FP32 ABUF
    or sideband buffer — are NOT covered by this runner. They require
    ISA / simulator / codegen extensions documented in
    `software/docs/w8a32_deployment_scope.md`.
    """
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    runner = WeightOnlyHostRunner(payload, weight_mode="per_channel")
    return runner.run_teacher_forced(inputs)


def run_fake_quant_teacher_forced_logits(
    payload: Dict[str, object],
    context_tokens: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    ptq_preset: str | Stage5PTQPreset | None = None,
    keep_kv_cache_fp32: bool = False,
    fp32_residual_stream: bool = False,
) -> List[np.ndarray]:
    inputs, _ = teacher_forced_inputs_and_targets(context_tokens)
    resolved_preset = resolve_stage5_ptq_preset(ptq_preset)
    ref = NanoGPTFQReference(
        payload["state_dict"],
        payload["model_args"],
        calibration_scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(payload["model_args"], resolved_preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(resolved_preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(resolved_preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(resolved_preset),
        keep_kv_cache_fp32=keep_kv_cache_fp32,
        fp32_residual_stream=fp32_residual_stream,
    )
    return ref.incremental_logits_trace(inputs)


def evaluate_gpt2_perplexity(
    payload: Dict[str, object],
    *,
    calibration_token_ids: Sequence[int],
    eval_token_ids: Sequence[int],
    tokenizer_dir: Path,
    calibration_sha256: str = "",
    eval_sha256: str = "",
    max_eval_tokens: int = 33,
    context_len: int = 32,
    calibration_seq_len: int = 128,
    calibration_n_seqs: int = 64,
    calibration_percentile: float = 99.9,
    ptq_preset: str | Stage5PTQPreset | None = None,
    output_aware_search_n_seqs: int | None = None,
    output_aware_search_seq_len: int | None = None,
    output_aware_search_workers: int | None = None,
    output_aware_include_pairs: bool = False,
    compute_fp32_ceiling: bool = True,
    debug_fp32_kv_cache: bool = False,
    debug_fp32_residual_stream: bool = False,
) -> GPT2PerplexityResult:
    if context_len < 1:
        raise ValueError("context_len must be positive")
    token_budget = min(int(max_eval_tokens), int(context_len) + 1)
    if token_budget < 2:
        raise ValueError("max_eval_tokens/context_len must allow at least two tokens")
    eval_tokens = [int(tok) for tok in eval_token_ids[:token_budget]]
    if len(eval_tokens) < 2:
        raise ValueError("evaluation text produced fewer than two tokens")

    # Phase 0A: capture FP32 ceiling BEFORE any state_dict mutations so the
    # returned number represents the true pre-quantization perplexity. Done
    # against the same eval_tokens / context_len as golden / fake_quant for
    # apples-to-apples comparison. Set compute_fp32_ceiling=False to skip
    # (saves ~30s per slow-gate run when not needed).
    fp32_logits: List[np.ndarray] | None = None
    if compute_fp32_ceiling:
        fp32_logits = run_fp32_teacher_forced_logits(payload, eval_tokens)

    resolved_preset = resolve_stage5_ptq_preset(
        stage5_default_ptq_preset_name() if ptq_preset is None else ptq_preset
    )

    # W8A32 branch. The W8A32 preset rejects every W8A8 transform at
    # construction time, so we short-circuit the rest of the pipeline.
    #
    # Phase 1 (commit `cf59efb`) populated `fake_quant_perplexity` via
    # `run_weight_only_int8_teacher_forced_logits` (NanoGPTFP32Reference
    # + QDQ weights). Phase 3 option (b) (commit landing this comment)
    # additionally populates `golden_perplexity` via
    # `WeightOnlyHostRunner` — a `HostRunner`-API-compatible runner that
    # mimics the deployment surface for option (b) "host-runtime FP32
    # with INT8 weight storage". The two paths produce bit-identical
    # logits by construction (both wrap the same QDQ helper); the
    # distinction is the API contract.
    #
    # Phase 3 options (c.1) and (c.2) — preserving the INT8 MXU via ISA
    # extensions for FP32 ABUF / sideband FP32 buffer — are documented
    # in `software/docs/w8a32_deployment_scope.md` and NOT implemented
    # in this commit; they require 3-5 weeks of ISA / simulator /
    # codegen work.
    if resolved_preset.weight_only_int8:
        _, targets = teacher_forced_inputs_and_targets(eval_tokens)
        vocab_size = int(payload["model_args"]["vocab_size"])
        fake_logits = run_weight_only_int8_teacher_forced_logits(payload, eval_tokens)
        if len(fake_logits) != len(targets):
            raise RuntimeError("W8A32 logits/targets length mismatch")
        fake_nlls = [
            stable_cross_entropy(
                np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size
            )
            for row, target in zip(fake_logits, targets)
        ]
        fake_ppl, fake_nll = perplexity_from_nlls(fake_nlls)

        golden_logits = run_weight_only_int8_golden_teacher_forced_logits(
            payload, eval_tokens
        )
        if len(golden_logits) != len(targets):
            raise RuntimeError("W8A32 golden logits/targets length mismatch")
        golden_nlls = [
            stable_cross_entropy(
                np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size
            )
            for row, target in zip(golden_logits, targets)
        ]
        golden_ppl, golden_nll = perplexity_from_nlls(golden_nlls)
        rel = abs(golden_ppl - fake_ppl) / max(abs(fake_ppl), 1e-12)

        fp32_ppl = float("nan")
        fp32_nll = float("nan")
        if fp32_logits is not None:
            if len(fp32_logits) != len(targets):
                raise RuntimeError("FP32 logits/targets length mismatch")
            fp32_nlls = [
                stable_cross_entropy(
                    np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size
                )
                for row, target in zip(fp32_logits, targets)
            ]
            fp32_ppl, fp32_nll = perplexity_from_nlls(fp32_nlls)

        return GPT2PerplexityResult(
            golden_perplexity=golden_ppl,
            fake_quant_perplexity=fake_ppl,
            relative_delta=float(rel),
            token_count=len(eval_tokens),
            target_count=len(targets),
            golden_nll=golden_nll,
            fake_quant_nll=fake_nll,
            tokenizer_dir=str(tokenizer_dir),
            calibration_sha256=calibration_sha256,
            eval_sha256=eval_sha256,
            ptq_preset=resolved_preset.name,
            fp32_perplexity=float(fp32_ppl),
            fp32_nll=float(fp32_nll),
        )
    if resolved_preset.quarot_enabled:
        # QuaRot Phase 1 must run BEFORE any calibration so the 99.9-percentile
        # activation scales are computed against the rotated (near-isotropic)
        # distribution. Mutates payload["state_dict"] in place. Internally
        # calls clear_weight_component_cache() so subsequent NanoGPTFQReference
        # constructions re-derive their cached weight components from the
        # rotated state.
        apply_quarot_rotation_from_token_ids(
            payload,
            calibration_token_ids,
            seed=resolved_preset.quarot_seed,
            kind=resolved_preset.quarot_kind,
        )
    if resolved_preset.awq_enabled:
        # AWQ Phase 1 Branch C: scales weight input channels and folds the
        # inverse into the upstream LN's gamma/bias. Mathematically identity-
        # preserving in FP32, but reduces per-channel weight scale spread
        # after INT8 quantization → smaller mean-scale dequant approximation
        # error. Runs AFTER rotation (which mutates the same weights) and
        # BEFORE calibration (so the calibration scales reflect the AWQ'd
        # LN distribution).
        apply_awq_from_token_ids(
            payload,
            calibration_token_ids,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
            alpha=resolved_preset.awq_alpha,
            target_modules=resolved_preset.awq_target_modules,
        )
    calibration_scales = build_calibration_scales_from_token_ids(
        payload,
        calibration_token_ids,
        n_seqs=calibration_n_seqs,
        seq_len=calibration_seq_len,
        percentile=calibration_percentile,
        activation_percentile_overrides=(
            resolved_preset.activation_percentile_nodes or None
        ),
        hessian_gelu_blocks=resolved_preset.hessian_gelu_blocks,
    )
    calibration_scales = apply_stage5_ptq_scale_policy(
        calibration_scales,
        payload["model_args"],
        resolved_preset,
    )
    if resolved_preset.bias_correction_blocks:
        # Mutates payload["state_dict"] biases in-place; both the bundle builder
        # and NanoGPTFQReference read biases from state_dict at construction, so
        # the corrected biases will flow into both the golden and fake-quant
        # paths automatically.
        apply_bias_correction_from_token_ids(
            payload,
            calibration_token_ids,
            calibration_scales,
            blocks=resolved_preset.bias_correction_blocks,
            weight_types=resolved_preset.bias_correction_weight_types,
        )
        # Re-derive activation scales with the corrected biases (the FP32
        # forward through corrected biases shifts activations slightly). This
        # matches the empirical setup that produced the validated PPL win.
        calibration_scales = build_calibration_scales_from_token_ids(
            payload,
            calibration_token_ids,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
            percentile=calibration_percentile,
            activation_percentile_overrides=(
                resolved_preset.activation_percentile_nodes or None
            ),
            hessian_gelu_blocks=resolved_preset.hessian_gelu_blocks,
        )
        calibration_scales = apply_stage5_ptq_scale_policy(
            calibration_scales,
            payload["model_args"],
            resolved_preset,
        )
    if resolved_preset.fc2_aware_gelu_blocks:
        calibration_scales, _ = apply_fc2_aware_gelu_scale_search_from_token_ids(
            payload,
            calibration_token_ids,
            calibration_scales,
            blocks=resolved_preset.fc2_aware_gelu_blocks,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
        )
    if resolved_preset.output_aware_gelu_blocks:
        calibration_scales, _ = apply_output_aware_gelu_scale_search_from_token_ids(
            payload,
            calibration_token_ids,
            calibration_scales,
            blocks=resolved_preset.output_aware_gelu_blocks,
            ptq_preset=resolved_preset,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
            search_n_seqs_max=output_aware_search_n_seqs,
            search_seq_len_max=output_aware_search_seq_len,
            search_workers=output_aware_search_workers,
        )
    if resolved_preset.output_aware_mlp_blocks:
        calibration_scales, _ = apply_output_aware_mlp_scale_search_from_token_ids(
            payload,
            calibration_token_ids,
            calibration_scales,
            blocks=resolved_preset.output_aware_mlp_blocks,
            ptq_preset=resolved_preset,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
            search_n_seqs_max=output_aware_search_n_seqs,
            search_seq_len_max=output_aware_search_seq_len,
            search_workers=output_aware_search_workers,
            include_pair_candidates=(output_aware_include_pairs or resolved_preset.output_aware_include_pairs),
            passes=resolved_preset.output_aware_mlp_passes,
        )
    if resolved_preset.output_aware_attn_blocks:
        calibration_scales, _ = apply_output_aware_attn_scale_search_from_token_ids(
            payload,
            calibration_token_ids,
            calibration_scales,
            blocks=resolved_preset.output_aware_attn_blocks,
            ptq_preset=resolved_preset,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
            search_n_seqs_max=output_aware_search_n_seqs,
            search_seq_len_max=output_aware_search_seq_len,
            search_workers=output_aware_search_workers,
        )
    if resolved_preset.output_aware_lm_head:
        calibration_scales, _ = apply_output_aware_lm_head_scale_search_from_token_ids(
            payload,
            calibration_token_ids,
            calibration_scales,
            ptq_preset=resolved_preset,
            n_seqs=calibration_n_seqs,
            seq_len=calibration_seq_len,
            search_n_seqs_max=output_aware_search_n_seqs,
            search_seq_len_max=output_aware_search_seq_len,
            search_workers=output_aware_search_workers,
        )
    _, targets = teacher_forced_inputs_and_targets(eval_tokens)
    vocab_size = int(payload["model_args"]["vocab_size"])
    lm_head_scale = float(calibration_scales.get("lm_head", 1.0))

    golden_logits = run_golden_teacher_forced_logits(
        payload,
        eval_tokens,
        calibration_scales,
        ptq_preset=resolved_preset,
    )
    fake_logits = run_fake_quant_teacher_forced_logits(
        payload,
        eval_tokens,
        calibration_scales,
        ptq_preset=resolved_preset,
        keep_kv_cache_fp32=debug_fp32_kv_cache,
        fp32_residual_stream=debug_fp32_residual_stream,
    )
    if len(golden_logits) != len(targets) or len(fake_logits) != len(targets):
        raise RuntimeError("teacher-forced logits/targets length mismatch")

    golden_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32) * np.float32(lm_head_scale), target, vocab_size=vocab_size)
        for row, target in zip(golden_logits, targets)
    ]
    fake_nlls = [
        stable_cross_entropy(np.asarray(row, dtype=np.float32) * np.float32(lm_head_scale), target, vocab_size=vocab_size)
        for row, target in zip(fake_logits, targets)
    ]
    golden_ppl, golden_nll = perplexity_from_nlls(golden_nlls)
    fake_ppl, fake_nll = perplexity_from_nlls(fake_nlls)
    rel = abs(golden_ppl - fake_ppl) / max(abs(fake_ppl), 1e-12)

    # Phase 0A: FP32 ceiling NLL/PPL. Logits are raw logits (no lm_head_scale
    # scaling) since the FP32 path doesn't pass through the INT8 logit
    # requantization. Use the same stable_cross_entropy and vocab_size mask
    # as golden / fake for consistency.
    fp32_ppl = float("nan")
    fp32_nll = float("nan")
    if fp32_logits is not None:
        if len(fp32_logits) != len(targets):
            raise RuntimeError("FP32 logits/targets length mismatch")
        fp32_nlls = [
            stable_cross_entropy(np.asarray(row, dtype=np.float32), target, vocab_size=vocab_size)
            for row, target in zip(fp32_logits, targets)
        ]
        fp32_ppl, fp32_nll = perplexity_from_nlls(fp32_nlls)

    return GPT2PerplexityResult(
        golden_perplexity=golden_ppl,
        fake_quant_perplexity=fake_ppl,
        relative_delta=float(rel),
        token_count=len(eval_tokens),
        target_count=len(targets),
        golden_nll=golden_nll,
        fake_quant_nll=fake_nll,
        tokenizer_dir=str(tokenizer_dir),
        calibration_sha256=calibration_sha256,
        eval_sha256=eval_sha256,
        ptq_preset=resolved_preset.name,
        fp32_perplexity=float(fp32_ppl),
        fp32_nll=float(fp32_nll),
    )
