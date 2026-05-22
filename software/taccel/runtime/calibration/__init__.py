"""Calibration helpers for Stage 3 tiny nanoGPT verification.

Runs FP32 forward passes on Shakespeare calibration sequences extracted from
the fixture payload and returns per-node max-abs scales that both the
compiler (via calibration_scales) and the NanoGPTFQReference consume.

This module is a re-export shim. The implementation is split across:
  - `scales`         — sequence + scale builders + constants
  - `fc2_aware_gelu` — FC2-aware GELU scale search
  - `output_aware`   — output-aware GELU/MLP/ATTN/LM_HEAD searches
  - `adapters`       — QuaRot / AWQ / bias-correction adapters
"""
from __future__ import annotations

# Re-export so existing `from taccel.runtime.calibration import X` keeps working.
from .scales import (
    _DEFAULT_SCALES,
    _SFU_DEFAULT_SCALES,
    _apply_raw_vadd_safe_tok_pos_scale,
    _fill_defaults,
    _tokenize_text,
    build_calibration_scales,
    build_calibration_scales_from_token_ids,
    build_calibration_seqs,
    build_calibration_seqs_from_token_ids,
    FC2_AWARE_GELU_MULTIPLIERS,
    OUTPUT_AWARE_GELU_MULTIPLIERS,
    OUTPUT_AWARE_MLP_MULTIPLIERS,
    OUTPUT_AWARE_SEARCH_N_SEQS_MAX,
    OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX,
    OUTPUT_AWARE_SEARCH_WORKERS_DEFAULT,
)
from .fc2_aware_gelu import (
    _fc2_aware_candidate_metrics,
    apply_fc2_aware_gelu_scale_search_from_token_ids,
    choose_fc2_aware_gelu_scale,
)
from .output_aware import (
    _candidate_mean_fake_quant_target_nlls,
    _mean_fake_quant_target_nll,
    _resolve_output_aware_search_workers,
    _stable_cross_entropy_np,
    apply_output_aware_attn_scale_search_from_token_ids,
    apply_output_aware_gelu_scale_search_from_token_ids,
    apply_output_aware_lm_head_scale_search_from_token_ids,
    apply_output_aware_mlp_scale_search_from_token_ids,
)
from .adapters import (
    _bias_correction_input_node,
    apply_awq_from_token_ids,
    apply_bias_correction_from_token_ids,
    apply_quarot_rotation_from_token_ids,
)

# Some external callers (taccel/quantizer/awq.py) reach for `_fp32_forward`
# through this module's namespace; preserve that path by importing here too.
from ..fake_quant_reference import _fp32_forward  # noqa: F401

__all__ = [
    "_DEFAULT_SCALES",
    "_SFU_DEFAULT_SCALES",
    "FC2_AWARE_GELU_MULTIPLIERS",
    "OUTPUT_AWARE_GELU_MULTIPLIERS",
    "OUTPUT_AWARE_MLP_MULTIPLIERS",
    "OUTPUT_AWARE_SEARCH_N_SEQS_MAX",
    "OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX",
    "OUTPUT_AWARE_SEARCH_WORKERS_DEFAULT",
    "apply_awq_from_token_ids",
    "apply_bias_correction_from_token_ids",
    "apply_fc2_aware_gelu_scale_search_from_token_ids",
    "apply_output_aware_attn_scale_search_from_token_ids",
    "apply_output_aware_gelu_scale_search_from_token_ids",
    "apply_output_aware_lm_head_scale_search_from_token_ids",
    "apply_output_aware_mlp_scale_search_from_token_ids",
    "apply_quarot_rotation_from_token_ids",
    "build_calibration_scales",
    "build_calibration_scales_from_token_ids",
    "build_calibration_seqs",
    "build_calibration_seqs_from_token_ids",
    "choose_fc2_aware_gelu_scale",
]
