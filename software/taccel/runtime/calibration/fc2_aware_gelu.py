"""FC2-aware GELU scale search (closes the GELU activation distribution against the FC2 weight stats).

Split from `runtime/calibration.py` as part of R2. Imports mirror the
original module; public symbols are re-exported via `__init__.py`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Sequence

import numpy as np

from ..fake_quant_reference import (
    NanoGPTFQReference,
    _arch_scale,
    _bias_i32,
    _fp32_forward,
    _fp32_to_int8,
    _int8_saturating_add,
    _requant_accum_pc_int8,
    _to_f32,
)
from ..stage5_ptq import (
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)
from ...quantizer.hessian_guided import find_hessian_gelu_scale
from ...quantizer.quantize import quantize_tensor


from .scales import (
    _DEFAULT_SCALES,
    _SFU_DEFAULT_SCALES,
    FC2_AWARE_GELU_MULTIPLIERS,
    OUTPUT_AWARE_GELU_MULTIPLIERS,
    OUTPUT_AWARE_MLP_MULTIPLIERS,
    OUTPUT_AWARE_SEARCH_N_SEQS_MAX,
    OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX,
    OUTPUT_AWARE_SEARCH_WORKERS_DEFAULT,
    build_calibration_scales,
    build_calibration_scales_from_token_ids,
    build_calibration_seqs,
    build_calibration_seqs_from_token_ids,
)


def _fc2_aware_candidate_metrics(
    *,
    gelu: np.ndarray,
    residual1: np.ndarray,
    residual2: np.ndarray,
    proj_w_q: np.ndarray,
    proj_w_scales: np.ndarray,
    proj_b_i32: np.ndarray,
    candidate_scale: float,
    fc2_scale: float,
    residual1_scale: float,
    residual2_scale: float,
) -> Dict[str, float]:
    gelu_f = np.asarray(gelu, dtype=np.float32)
    residual1_f = np.asarray(residual1, dtype=np.float32)
    residual2_f = np.asarray(residual2, dtype=np.float32)
    gelu_i8 = _fp32_to_int8(gelu_f, candidate_scale)
    fc2_accum = gelu_i8.astype(np.int32) @ proj_w_q.astype(np.int32).T + proj_b_i32.reshape(1, -1)
    requant_pc = (
        np.float32(candidate_scale)
        * np.asarray(proj_w_scales, dtype=np.float32)[: proj_w_q.shape[0]]
        / max(np.float32(fc2_scale), np.float32(1e-12))
    )
    fc2_i8 = _requant_accum_pc_int8(fc2_accum, requant_pc)
    residual1_i8 = _fp32_to_int8(residual1_f, residual1_scale)
    residual2_i8 = _int8_saturating_add(residual1_i8, fc2_i8)
    target_residual2_i8 = _fp32_to_int8(residual2_f, residual2_scale)
    target_fc2_i8 = _fp32_to_int8(residual2_f - residual1_f, fc2_scale)

    gelu_arch_scale = float(_arch_scale(candidate_scale))
    if gelu_arch_scale <= 0.0:
        clipping_rate = 1.0
    else:
        clipping_rate = float(np.mean(np.abs(gelu_f / np.float32(gelu_arch_scale)) > 127.0))
    return {
        "objective_mse": float(np.mean((residual2_i8.astype(np.float32) - target_residual2_i8.astype(np.float32)) ** 2)),
        "fc2_mse": float(np.mean((fc2_i8.astype(np.float32) - target_fc2_i8.astype(np.float32)) ** 2)),
        "residual2_saturation_rate": float(np.mean((residual2_i8 == 127) | (residual2_i8 == -128))),
        "fc2_saturation_rate": float(np.mean((fc2_i8 == 127) | (fc2_i8 == -128))),
        "gelu_clipping_rate": clipping_rate,
    }

def choose_fc2_aware_gelu_scale(
    *,
    gelu: np.ndarray,
    residual1: np.ndarray,
    residual2: np.ndarray,
    proj_w_q: np.ndarray,
    proj_w_scales: np.ndarray,
    proj_b_i32_by_scale: Dict[float, np.ndarray],
    base_scale: float,
    fc2_scale: float,
    residual1_scale: float,
    residual2_scale: float,
    multipliers: Sequence[float] = FC2_AWARE_GELU_MULTIPLIERS,
) -> Dict[str, float]:
    """Choose a GELU scale by modelling the actual FC2→raw-residual2 path."""
    if base_scale <= 0.0:
        raise ValueError("base GELU scale must be positive for FC2-aware search")
    if not np.isclose(residual1_scale, residual2_scale, rtol=1e-4, atol=1e-8):
        raise ValueError("FC2-aware GELU search requires residual1 and residual2 to share the raw VADD scale")
    if not np.isclose(fc2_scale, residual2_scale, rtol=1e-4, atol=1e-8):
        raise ValueError("FC2-aware GELU search requires fc2 and residual2 to share the raw VADD scale")

    rows: List[Dict[str, float]] = []
    for multiplier in multipliers:
        m = float(multiplier)
        if m <= 0.0:
            continue
        candidate_scale = float(base_scale) * m
        proj_b_i32 = proj_b_i32_by_scale[candidate_scale]
        metrics = _fc2_aware_candidate_metrics(
            gelu=gelu,
            residual1=residual1,
            residual2=residual2,
            proj_w_q=proj_w_q,
            proj_w_scales=proj_w_scales,
            proj_b_i32=proj_b_i32,
            candidate_scale=candidate_scale,
            fc2_scale=fc2_scale,
            residual1_scale=residual1_scale,
            residual2_scale=residual2_scale,
        )
        rows.append({
            "multiplier": m,
            "scale": candidate_scale,
            **metrics,
        })
    if not rows:
        raise ValueError("FC2-aware GELU search received no valid candidate multipliers")
    best = min(
        rows,
        key=lambda row: (
            row["objective_mse"],
            row["fc2_mse"],
            row["residual2_saturation_rate"],
            row["gelu_clipping_rate"],
            abs(row["multiplier"] - 1.0),
            row["multiplier"],
        ),
    )
    baseline = min(rows, key=lambda row: (abs(row["multiplier"] - 1.0), row["multiplier"]))
    return {
        "old_scale": float(base_scale),
        "new_scale": float(best["scale"]),
        "multiplier": float(best["multiplier"]),
        "objective_mse": float(best["objective_mse"]),
        "baseline_objective_mse": float(baseline["objective_mse"]),
        "fc2_mse": float(best["fc2_mse"]),
        "residual2_saturation_rate": float(best["residual2_saturation_rate"]),
        "fc2_saturation_rate": float(best["fc2_saturation_rate"]),
        "gelu_clipping_rate": float(best["gelu_clipping_rate"]),
        "candidate_count": int(len(rows)),
    }

def apply_fc2_aware_gelu_scale_search_from_token_ids(
    payload: dict,
    token_ids: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    blocks: Sequence[int],
    n_seqs: int = 8,
    seq_len: int = 16,
    multipliers: Sequence[float] = FC2_AWARE_GELU_MULTIPLIERS,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Override selected GELU scales using an FC2-aware integer objective."""
    block_set = sorted({int(block) for block in blocks})
    if not block_set:
        return dict(calibration_scales), {}

    model_args = payload["model_args"]
    state_dict = payload["state_dict"]
    n_layer = int(model_args["n_layer"])
    invalid = [block for block in block_set if block < 0 or block >= n_layer]
    if invalid:
        raise ValueError(f"FC2-aware GELU blocks outside n_layer={n_layer}: {invalid}")

    seqs = build_calibration_seqs_from_token_ids(token_ids, n_seqs=n_seqs, seq_len=seq_len)
    node_outputs = [_fp32_forward(state_dict, model_args, tids) for tids in seqs]
    scales = dict(calibration_scales)
    diagnostics: Dict[str, Dict[str, float]] = {}

    for block in block_set:
        required_nodes = [f"block{block}_gelu", f"block{block}_residual1", f"block{block}_residual2"]
        missing_nodes = [name for name in required_nodes if any(name not in out for out in node_outputs)]
        if missing_nodes:
            raise ValueError(f"cannot run FC2-aware GELU search for block {block}; missing {sorted(set(missing_nodes))}")

        gelu = np.concatenate([np.asarray(out[f"block{block}_gelu"], dtype=np.float32) for out in node_outputs], axis=0)
        residual1 = np.concatenate([np.asarray(out[f"block{block}_residual1"], dtype=np.float32) for out in node_outputs], axis=0)
        residual2 = np.concatenate([np.asarray(out[f"block{block}_residual2"], dtype=np.float32) for out in node_outputs], axis=0)

        proj_w_key = f"transformer.h.{block}.mlp.c_proj.weight"
        proj_b_key = f"transformer.h.{block}.mlp.c_proj.bias"
        if proj_w_key not in state_dict:
            raise ValueError(f"missing FC2 weight {proj_w_key!r}")
        proj_w = _to_f32(state_dict[proj_w_key])
        proj_w_q, proj_w_scales = quantize_tensor(proj_w, per_channel=True)
        proj_w_q = np.asarray(proj_w_q, dtype=np.int8)
        proj_w_scales = np.asarray(proj_w_scales, dtype=np.float32)
        output_dim = int(proj_w_q.shape[0])

        base_scale = float(scales.get(f"block{block}_gelu", _SFU_DEFAULT_SCALES))
        candidate_scales = [base_scale * float(m) for m in multipliers if float(m) > 0.0]
        proj_b_i32_by_scale = {
            float(candidate): _bias_i32(
                state_dict,
                proj_b_key,
                float(candidate),
                proj_w_scales,
                output_dim,
            )
            for candidate in candidate_scales
        }
        result = choose_fc2_aware_gelu_scale(
            gelu=gelu,
            residual1=residual1,
            residual2=residual2,
            proj_w_q=proj_w_q,
            proj_w_scales=proj_w_scales,
            proj_b_i32_by_scale=proj_b_i32_by_scale,
            base_scale=base_scale,
            fc2_scale=float(scales[f"block{block}_fc2"]),
            residual1_scale=float(scales[f"block{block}_residual1"]),
            residual2_scale=float(scales[f"block{block}_residual2"]),
            multipliers=multipliers,
        )
        scales[f"block{block}_gelu"] = float(result["new_scale"])
        diagnostics[f"block{block}"] = result
    return scales, diagnostics
