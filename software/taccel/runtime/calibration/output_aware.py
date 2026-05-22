"""Output-aware scale searches: GELU, MLP, ATTN, LM_HEAD — minimize fake-quant target NLL.

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

from .fc2_aware_gelu import (
    _fc2_aware_candidate_metrics,
    choose_fc2_aware_gelu_scale,
)


def _stable_cross_entropy_np(logits: np.ndarray, target: int, *, vocab_size: int) -> float:
    active = np.asarray(logits, dtype=np.float32)[: int(vocab_size)]
    target_i = int(target)
    if target_i < 0 or target_i >= active.size:
        raise ValueError(f"target token {target_i} is outside vocab size {active.size}")
    row_max = float(np.max(active))
    shifted = active - np.float32(row_max)
    exp_shifted = np.exp(shifted.astype(np.float32)).astype(np.float32)
    return float(row_max + float(np.log(exp_shifted.sum(dtype=np.float32))) - float(active[target_i]))

def _mean_fake_quant_target_nll(
    payload: Dict[str, object],
    seqs: Sequence[Sequence[int]],
    scales: Dict[str, float],
    *,
    ptq_preset,
) -> float:
    model_args = payload["model_args"]
    vocab_size = int(model_args["vocab_size"])
    lm_scale = float(scales.get("lm_head", 1.0))
    ref = NanoGPTFQReference(
        payload["state_dict"],
        model_args,
        scales,
        requant_pc_weight_names=stage5_requant_pc_weight_names(model_args, ptq_preset),
        raw_residual1_blocks=stage5_raw_residual1_blocks(ptq_preset),
        raw_residual2_blocks=stage5_raw_residual2_blocks(ptq_preset),
        gelu_from_accum_blocks=stage5_gelu_from_accum_blocks(ptq_preset),
    )
    nlls: List[float] = []
    for seq in seqs:
        tokens = [int(tok) for tok in seq]
        if len(tokens) < 2:
            continue
        inputs = tokens[:-1]
        targets = tokens[1:]
        logits = ref.forward(inputs, return_all_logits=True)
        for row, target in zip(logits, targets):
            deq = np.asarray(row, dtype=np.float32) * np.float32(lm_scale)
            nlls.append(_stable_cross_entropy_np(deq, target, vocab_size=vocab_size))
    if not nlls:
        raise ValueError("output-aware GELU search requires calibration windows with at least two tokens")
    return float(np.mean(np.asarray(nlls, dtype=np.float64)))

def _resolve_output_aware_search_workers(
    search_workers: int | None,
    candidate_count: int,
) -> int:
    if candidate_count <= 1:
        return 1
    requested = search_workers
    if requested is None:
        env_value = os.environ.get("TACCEL_OUTPUT_AWARE_SEARCH_WORKERS")
        requested = int(env_value) if env_value else OUTPUT_AWARE_SEARCH_WORKERS_DEFAULT
    return max(1, min(int(requested), int(candidate_count)))

def _candidate_mean_fake_quant_target_nlls(
    payload: Dict[str, object],
    seqs: Sequence[Sequence[int]],
    candidate_scales: Sequence[Dict[str, float]],
    *,
    ptq_preset,
    search_workers: int | None = None,
) -> List[float]:
    if not candidate_scales:
        return []
    worker_count = _resolve_output_aware_search_workers(search_workers, len(candidate_scales))
    if worker_count <= 1:
        return [
            _mean_fake_quant_target_nll(payload, seqs, scales, ptq_preset=ptq_preset)
            for scales in candidate_scales
        ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            executor.map(
                lambda scales: _mean_fake_quant_target_nll(
                    payload,
                    seqs,
                    scales,
                    ptq_preset=ptq_preset,
                ),
                candidate_scales,
            )
        )

def apply_output_aware_gelu_scale_search_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    blocks: Sequence[int],
    ptq_preset,
    n_seqs: int = 8,
    seq_len: int = 16,
    multipliers: Sequence[float] = OUTPUT_AWARE_GELU_MULTIPLIERS,
    search_n_seqs_max: int | None = None,
    search_seq_len_max: int | None = None,
    search_workers: int | None = None,
) -> tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    """Greedily choose late GELU scales by final fake-quant token NLL.

    This intentionally tunes only ``blockL_gelu``. The current best GPT-2 preset
    uses raw INT8 residual2 VADD, so changing fc2/residual2 scales independently
    would violate the shared-scale contract between the skip path and FC2 output.
    """
    block_set = sorted({int(block) for block in blocks})
    if not block_set:
        return dict(calibration_scales), {}

    model_args = payload["model_args"]
    n_layer = int(model_args["n_layer"])
    invalid = [block for block in block_set if block < 0 or block >= n_layer]
    if invalid:
        raise ValueError(f"output-aware GELU blocks outside n_layer={n_layer}: {invalid}")

    n_cap = OUTPUT_AWARE_SEARCH_N_SEQS_MAX if search_n_seqs_max is None else int(search_n_seqs_max)
    len_cap = OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX if search_seq_len_max is None else int(search_seq_len_max)
    search_n_seqs = max(1, min(int(n_seqs), max(1, n_cap)))
    search_seq_len = max(2, min(int(seq_len), max(2, len_cap)))
    seqs = build_calibration_seqs_from_token_ids(
        token_ids,
        n_seqs=search_n_seqs,
        seq_len=search_seq_len,
    )
    scales = dict(calibration_scales)
    diagnostics: Dict[str, Dict[str, object]] = {}
    current_nll = _mean_fake_quant_target_nll(payload, seqs, scales, ptq_preset=ptq_preset)

    for block in block_set:
        key = f"block{block}_gelu"
        base_scale = float(scales.get(key, _SFU_DEFAULT_SCALES))
        if base_scale <= 0.0:
            raise ValueError(f"output-aware GELU search requires positive scale for {key}")
        block_start_nll = float(current_nll)
        candidate_rows: List[Dict[str, object]] = []
        pending_indices: List[int] = []
        pending_scales: List[Dict[str, float]] = []
        for multiplier in multipliers:
            m = float(multiplier)
            if m <= 0.0:
                continue
            candidate_scales = dict(scales)
            candidate_scales[key] = float(base_scale * m)
            if m == 1.0:
                mean_nll = block_start_nll
            else:
                mean_nll = float("nan")
                pending_indices.append(len(candidate_rows))
                pending_scales.append(candidate_scales)
            candidate_rows.append({
                "multiplier": m,
                "scale": float(candidate_scales[key]),
                "mean_nll": float(mean_nll),
            })
        for row_index, mean_nll in zip(
            pending_indices,
            _candidate_mean_fake_quant_target_nlls(
                payload,
                seqs,
                pending_scales,
                ptq_preset=ptq_preset,
                search_workers=search_workers,
            ),
        ):
            candidate_rows[row_index]["mean_nll"] = float(mean_nll)
        if not candidate_rows:
            raise ValueError("output-aware GELU search received no valid candidate multipliers")
        best = min(
            candidate_rows,
            key=lambda row: (
                row["mean_nll"],
                abs(row["multiplier"] - 1.0),
                row["multiplier"],
            ),
        )
        accepted = bool(best["mean_nll"] < current_nll)
        selected_scale = float(best["scale"]) if accepted else base_scale
        if accepted:
            scales[key] = selected_scale
            current_nll = float(best["mean_nll"])
        diagnostics[f"block{block}"] = {
            "old_scale": float(base_scale),
            "new_scale": float(selected_scale),
            "multiplier": float(best["multiplier"]) if accepted else 1.0,
            "accepted": accepted,
            "baseline_mean_nll": block_start_nll,
            "best_candidate_mean_nll": float(best["mean_nll"]),
            "selected_mean_nll": float(current_nll),
            "candidate_count": int(len(candidate_rows)),
            "search_n_seqs": int(search_n_seqs),
            "search_seq_len": int(search_seq_len),
            "candidates": candidate_rows,
        }
    return scales, diagnostics

def apply_output_aware_mlp_scale_search_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    blocks: Sequence[int],
    ptq_preset,
    n_seqs: int = 8,
    seq_len: int = 16,
    multipliers: Sequence[float] = OUTPUT_AWARE_MLP_MULTIPLIERS,
    search_n_seqs_max: int | None = None,
    search_seq_len_max: int | None = None,
    search_workers: int | None = None,
    include_pair_candidates: bool = False,
    passes: int = 1,
) -> tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    """Greedily tune late MLP scale groups against final fake-quant token NLL.

    Each late block searches three valid groups:

    * ``fc1``: changes the FC1 output quantization before GELU.
    * ``gelu``: changes SFU/GELU output quantization before FC2.
    * ``residual_group``: changes ``residual1``, ``fc2``, and ``residual2``
      together, preserving the raw residual2 VADD shared-scale contract.

    With ``passes > 1`` the block loop is repeated; each subsequent pass
    operates on the already-tuned scales as base, so a multiplier of 1.25
    on the second pass compounds with whatever multiplier was accepted on
    the first pass for that group. This lets blocks reach scale targets
    further from the original than a single grid step allows.
    """
    block_set = sorted({int(block) for block in blocks})
    if not block_set:
        return dict(calibration_scales), {}

    model_args = payload["model_args"]
    n_layer = int(model_args["n_layer"])
    invalid = [block for block in block_set if block < 0 or block >= n_layer]
    if invalid:
        raise ValueError(f"output-aware MLP blocks outside n_layer={n_layer}: {invalid}")

    n_cap = OUTPUT_AWARE_SEARCH_N_SEQS_MAX if search_n_seqs_max is None else int(search_n_seqs_max)
    len_cap = OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX if search_seq_len_max is None else int(search_seq_len_max)
    search_n_seqs = max(1, min(int(n_seqs), max(1, n_cap)))
    search_seq_len = max(2, min(int(seq_len), max(2, len_cap)))
    seqs = build_calibration_seqs_from_token_ids(
        token_ids,
        n_seqs=search_n_seqs,
        seq_len=search_seq_len,
    )
    scales = dict(calibration_scales)
    diagnostics: Dict[str, Dict[str, object]] = {}
    current_nll = _mean_fake_quant_target_nll(payload, seqs, scales, ptq_preset=ptq_preset)

    def _candidate_for_group(group_keys: Sequence[str], base_scale: float, multiplier: float) -> Dict[str, float]:
        candidate = dict(scales)
        new_scale = float(base_scale) * float(multiplier)
        for key in group_keys:
            candidate[key] = new_scale
        return candidate

    pass_count = max(1, int(passes))
    for pass_index in range(pass_count):
        for block in block_set:
            block_start_nll = float(current_nll)
            group_specs = [
                ("fc1", ((f"block{block}_fc1",),)),
                ("gelu", ((f"block{block}_gelu",),)),
                (
                    "residual_group",
                    ((
                        f"block{block}_residual1",
                        f"block{block}_fc2",
                        f"block{block}_residual2",
                    ),),
                ),
            ]
            if include_pair_candidates:
                group_specs.extend([
                    (
                        "fc1_gelu",
                        ((f"block{block}_fc1",), (f"block{block}_gelu",)),
                    ),
                    (
                        "gelu_residual_group",
                        (
                            (f"block{block}_gelu",),
                            (
                                f"block{block}_residual1",
                                f"block{block}_fc2",
                                f"block{block}_residual2",
                            ),
                        ),
                    ),
                    (
                        "fc1_gelu_residual_group",
                        (
                            (f"block{block}_fc1",),
                            (f"block{block}_gelu",),
                            (
                                f"block{block}_residual1",
                                f"block{block}_fc2",
                                f"block{block}_residual2",
                            ),
                        ),
                    ),
                ])
            group_results: Dict[str, object] = {}
            for group_name, key_groups in group_specs:
                base_scales = [float(scales.get(keys[0], _DEFAULT_SCALES)) for keys in key_groups]
                if any(scale <= 0.0 for scale in base_scales):
                    raise ValueError(f"output-aware MLP search requires positive scales for {key_groups}")
                group_start_nll = float(current_nll)
                candidate_rows: List[Dict[str, object]] = []
                pending_indices: List[int] = []
                pending_scales: List[Dict[str, float]] = []
                for multiplier in multipliers:
                    m = float(multiplier)
                    if m <= 0.0:
                        continue
                    candidate_scales = dict(scales)
                    for keys, base_scale in zip(key_groups, base_scales):
                        candidate_scales.update(_candidate_for_group(keys, base_scale, m))
                    if m == 1.0:
                        mean_nll = group_start_nll
                    else:
                        mean_nll = float("nan")
                        pending_indices.append(len(candidate_rows))
                        pending_scales.append(candidate_scales)
                    candidate_rows.append({
                        "multiplier": m,
                        "scales": {
                            keys[0]: float(base_scale * m)
                            for keys, base_scale in zip(key_groups, base_scales)
                        },
                        "mean_nll": float(mean_nll),
                    })
                for row_index, mean_nll in zip(
                    pending_indices,
                    _candidate_mean_fake_quant_target_nlls(
                        payload,
                        seqs,
                        pending_scales,
                        ptq_preset=ptq_preset,
                        search_workers=search_workers,
                    ),
                ):
                    candidate_rows[row_index]["mean_nll"] = float(mean_nll)
                if not candidate_rows:
                    raise ValueError("output-aware MLP search received no valid candidate multipliers")
                best = min(
                    candidate_rows,
                    key=lambda row: (
                        row["mean_nll"],
                        abs(row["multiplier"] - 1.0),
                        row["multiplier"],
                    ),
                )
                accepted = bool(best["mean_nll"] < current_nll)
                if accepted:
                    for keys, base_scale in zip(key_groups, base_scales):
                        selected_scale = float(base_scale * float(best["multiplier"]))
                        for key in keys:
                            scales[key] = selected_scale
                    current_nll = float(best["mean_nll"])
                group_results[group_name] = {
                    "key_groups": [list(keys) for keys in key_groups],
                    "old_scales": {
                        keys[0]: float(base_scale)
                        for keys, base_scale in zip(key_groups, base_scales)
                    },
                    "new_scales": {
                        keys[0]: float(scales.get(keys[0], base_scale))
                        for keys, base_scale in zip(key_groups, base_scales)
                    },
                    "multiplier": float(best["multiplier"]) if accepted else 1.0,
                    "accepted": accepted,
                    "baseline_mean_nll": group_start_nll,
                    "best_candidate_mean_nll": float(best["mean_nll"]),
                    "selected_mean_nll": float(current_nll),
                    "candidate_count": int(len(candidate_rows)),
                    "candidates": candidate_rows,
                }
            diag_key = (
                f"block{block}" if pass_count == 1 else f"pass{pass_index}_block{block}"
            )
            diagnostics[diag_key] = {
                "pass": int(pass_index),
                "baseline_mean_nll": block_start_nll,
                "selected_mean_nll": float(current_nll),
                "search_n_seqs": int(search_n_seqs),
                "search_seq_len": int(search_seq_len),
                "groups": group_results,
            }
    return scales, diagnostics

def apply_output_aware_attn_scale_search_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    blocks: Sequence[int],
    ptq_preset,
    n_seqs: int = 8,
    seq_len: int = 16,
    multipliers: Sequence[float] = OUTPUT_AWARE_MLP_MULTIPLIERS,
    search_n_seqs_max: int | None = None,
    search_seq_len_max: int | None = None,
    search_workers: int | None = None,
    include_value_search: bool = False,
) -> tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    """Greedily tune attn_v scale groups against final fake-quant token NLL.

    Searches one primary group per block:

    * ``attn_v``: applies a shared multiplier to all per-head attn_v scales.

    Optionally also searches ``value`` (per-head V projection output scales)
    as a secondary group after attn_v if ``include_value_search=True``.
    """
    block_set = sorted({int(block) for block in blocks})
    if not block_set:
        return dict(calibration_scales), {}

    model_args = payload["model_args"]
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])
    invalid = [block for block in block_set if block < 0 or block >= n_layer]
    if invalid:
        raise ValueError(f"output-aware attn blocks outside n_layer={n_layer}: {invalid}")

    n_cap = OUTPUT_AWARE_SEARCH_N_SEQS_MAX if search_n_seqs_max is None else int(search_n_seqs_max)
    len_cap = OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX if search_seq_len_max is None else int(search_seq_len_max)
    search_n_seqs = max(1, min(int(n_seqs), max(1, n_cap)))
    search_seq_len = max(2, min(int(seq_len), max(2, len_cap)))
    seqs = build_calibration_seqs_from_token_ids(
        token_ids,
        n_seqs=search_n_seqs,
        seq_len=search_seq_len,
    )
    scales = dict(calibration_scales)
    diagnostics: Dict[str, Dict[str, object]] = {}
    current_nll = _mean_fake_quant_target_nll(payload, seqs, scales, ptq_preset=ptq_preset)

    def _candidate_for_group(group_keys: Sequence[str], base_scale: float, multiplier: float) -> Dict[str, float]:
        candidate = dict(scales)
        new_scale = float(base_scale) * float(multiplier)
        for key in group_keys:
            candidate[key] = new_scale
        return candidate

    def _value_keys(block: int, head: int) -> tuple[str, str]:
        value_key = f"block{block}_head{head}_value"
        return value_key, f"block{block}_head{head}_value_kv_load"

    for block in block_set:
        block_start_nll = float(current_nll)
        group_specs = [
            (
                "attn_v",
                tuple((f"block{block}_head{H}_attn_v",) for H in range(n_head)),
            ),
        ]
        if include_value_search:
            group_specs.append((
                "value",
                tuple(_value_keys(block, H) for H in range(n_head)),
            ))
        group_results: Dict[str, object] = {}
        for group_name, key_groups in group_specs:
            base_scales = [float(scales.get(keys[0], _DEFAULT_SCALES)) for keys in key_groups]
            if any(scale <= 0.0 for scale in base_scales):
                raise ValueError(f"output-aware attn search requires positive scales for {key_groups}")
            group_start_nll = float(current_nll)
            candidate_rows: List[Dict[str, object]] = []
            pending_indices: List[int] = []
            pending_scales: List[Dict[str, float]] = []
            for multiplier in multipliers:
                m = float(multiplier)
                if m <= 0.0:
                    continue
                candidate_scales = dict(scales)
                for keys, base_scale in zip(key_groups, base_scales):
                    candidate_scales.update(_candidate_for_group(keys, base_scale, m))
                if m == 1.0:
                    mean_nll = group_start_nll
                else:
                    mean_nll = float("nan")
                    pending_indices.append(len(candidate_rows))
                    pending_scales.append(candidate_scales)
                candidate_rows.append({
                    "multiplier": m,
                    "scales": {
                        keys[0]: float(base_scale * m)
                        for keys, base_scale in zip(key_groups, base_scales)
                    },
                    "mean_nll": float(mean_nll),
                })
            for row_index, mean_nll in zip(
                pending_indices,
                _candidate_mean_fake_quant_target_nlls(
                    payload,
                    seqs,
                    pending_scales,
                    ptq_preset=ptq_preset,
                    search_workers=search_workers,
                ),
            ):
                candidate_rows[row_index]["mean_nll"] = float(mean_nll)
            if not candidate_rows:
                raise ValueError("output-aware attn search received no valid candidate multipliers")
            best = min(
                candidate_rows,
                key=lambda row: (
                    row["mean_nll"],
                    abs(row["multiplier"] - 1.0),
                    row["multiplier"],
                ),
            )
            accepted = bool(best["mean_nll"] < current_nll)
            if accepted:
                for keys, base_scale in zip(key_groups, base_scales):
                    selected_scale = float(base_scale * float(best["multiplier"]))
                    for key in keys:
                        scales[key] = selected_scale
                current_nll = float(best["mean_nll"])
            group_results[group_name] = {
                "key_groups": [list(keys) for keys in key_groups],
                "old_scales": {
                    keys[0]: float(base_scale)
                    for keys, base_scale in zip(key_groups, base_scales)
                },
                "new_scales": {
                    keys[0]: float(scales.get(keys[0], base_scale))
                    for keys, base_scale in zip(key_groups, base_scales)
                },
                "multiplier": float(best["multiplier"]) if accepted else 1.0,
                "accepted": accepted,
                "baseline_mean_nll": group_start_nll,
                "best_candidate_mean_nll": float(best["mean_nll"]),
                "selected_mean_nll": float(current_nll),
                "candidate_count": int(len(candidate_rows)),
                "candidates": candidate_rows,
            }
        diagnostics[f"block{block}"] = {
            "baseline_mean_nll": block_start_nll,
            "selected_mean_nll": float(current_nll),
            "search_n_seqs": int(search_n_seqs),
            "search_seq_len": int(search_seq_len),
            "groups": group_results,
        }
    return scales, diagnostics

def apply_output_aware_lm_head_scale_search_from_token_ids(
    payload: Dict[str, object],
    token_ids: Sequence[int],
    calibration_scales: Dict[str, float],
    *,
    ptq_preset,
    n_seqs: int = 8,
    seq_len: int = 16,
    multipliers: Sequence[float] = OUTPUT_AWARE_MLP_MULTIPLIERS,
    search_n_seqs_max: int | None = None,
    search_seq_len_max: int | None = None,
    search_workers: int | None = None,
) -> tuple[Dict[str, float], Dict[str, object]]:
    """Greedily tune the lm_head output scale against final fake-quant token NLL."""
    n_cap = OUTPUT_AWARE_SEARCH_N_SEQS_MAX if search_n_seqs_max is None else int(search_n_seqs_max)
    len_cap = OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX if search_seq_len_max is None else int(search_seq_len_max)
    search_n_seqs = max(1, min(int(n_seqs), max(1, n_cap)))
    search_seq_len = max(2, min(int(seq_len), max(2, len_cap)))
    seqs = build_calibration_seqs_from_token_ids(token_ids, n_seqs=search_n_seqs, seq_len=search_seq_len)
    scales = dict(calibration_scales)
    base_scale = float(scales.get("lm_head", _DEFAULT_SCALES))
    if base_scale <= 0.0:
        raise ValueError("output-aware lm_head search requires a positive lm_head scale")
    baseline_nll = _mean_fake_quant_target_nll(payload, seqs, scales, ptq_preset=ptq_preset)
    current_nll = baseline_nll
    candidate_rows: List[Dict[str, float]] = []
    pending_indices: List[int] = []
    pending_scales: List[Dict[str, float]] = []
    for multiplier in multipliers:
        m = float(multiplier)
        if m <= 0.0:
            continue
        candidate = dict(scales)
        candidate["lm_head"] = base_scale * m
        if m == 1.0:
            mean_nll = baseline_nll
        else:
            mean_nll = float("nan")
            pending_indices.append(len(candidate_rows))
            pending_scales.append(candidate)
        candidate_rows.append({"multiplier": m, "lm_head": base_scale * m, "mean_nll": float(mean_nll)})
    for row_index, mean_nll in zip(
        pending_indices,
        _candidate_mean_fake_quant_target_nlls(
            payload,
            seqs,
            pending_scales,
            ptq_preset=ptq_preset,
            search_workers=search_workers,
        ),
    ):
        candidate_rows[row_index]["mean_nll"] = float(mean_nll)
    if not candidate_rows:
        raise ValueError("output-aware lm_head search received no valid candidate multipliers")
    best = min(
        candidate_rows,
        key=lambda row: (row["mean_nll"], abs(row["multiplier"] - 1.0), row["multiplier"]),
    )
    accepted = bool(best["mean_nll"] < current_nll)
    if accepted:
        scales["lm_head"] = float(best["lm_head"])
        current_nll = float(best["mean_nll"])
    diagnostics: Dict[str, object] = {
        "old_scale": base_scale,
        "new_scale": float(scales["lm_head"]),
        "multiplier": float(best["multiplier"]) if accepted else 1.0,
        "accepted": accepted,
        "baseline_mean_nll": float(baseline_nll),
        "best_candidate_mean_nll": float(best["mean_nll"]),
        "selected_mean_nll": float(current_nll),
        "search_n_seqs": int(search_n_seqs),
        "search_seq_len": int(search_seq_len),
        "candidates": candidate_rows,
    }
    return scales, diagnostics
