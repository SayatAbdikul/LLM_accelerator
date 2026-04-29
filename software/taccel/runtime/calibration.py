"""Calibration helpers for Stage 3 tiny nanoGPT verification.

Runs FP32 forward passes on Shakespeare calibration sequences extracted from
the fixture payload and returns per-node max-abs scales that both the
compiler (via calibration_scales) and the NanoGPTFQReference consume.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .fake_quant_reference import (
    NanoGPTFQReference,
    _arch_scale,
    _bias_i32,
    _fp32_forward,
    _fp32_to_int8,
    _int8_saturating_add,
    _requant_accum_pc_int8,
    _to_f32,
)
from .stage5_ptq import (
    stage5_gelu_from_accum_blocks,
    stage5_raw_residual1_blocks,
    stage5_raw_residual2_blocks,
    stage5_requant_pc_weight_names,
)
from ..quantizer.hessian_guided import find_hessian_gelu_scale
from ..quantizer.quantize import quantize_tensor


_DEFAULT_SCALES = 6.0 / 127.0
_SFU_DEFAULT_SCALES = 1.0 / 127.0
FC2_AWARE_GELU_MULTIPLIERS = (
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
)
OUTPUT_AWARE_GELU_MULTIPLIERS = FC2_AWARE_GELU_MULTIPLIERS
OUTPUT_AWARE_MLP_MULTIPLIERS = (0.75, 0.875, 1.0, 1.125, 1.25, 1.5)
OUTPUT_AWARE_SEARCH_N_SEQS_MAX = 1
OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX = 16


def _tokenize_text(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def build_calibration_seqs(
    payload: dict,
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
) -> List[List[int]]:
    """Extract short token sequences from the fixture's embedded Shakespeare text."""
    text = str(payload.get("text", ""))
    stoi = payload.get("stoi", {})
    if not text or not stoi:
        return [[0] * seq_len]
    tokens = _tokenize_text(text, stoi)
    if len(tokens) < seq_len:
        tokens = tokens * ((seq_len // max(len(tokens), 1)) + 2)
    seqs = []
    step = max(1, (len(tokens) - seq_len) // max(n_seqs, 1))
    for i in range(n_seqs):
        start = (i * step) % (len(tokens) - seq_len + 1)
        seqs.append(tokens[start: start + seq_len])
    return seqs


def build_calibration_seqs_from_token_ids(
    token_ids: Sequence[int],
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
) -> List[List[int]]:
    """Extract deterministic calibration windows from already-tokenized text."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    tokens = [int(tok) for tok in token_ids]
    if not tokens:
        return [[0] * seq_len]
    if len(tokens) < seq_len:
        tokens = tokens * ((seq_len // max(len(tokens), 1)) + 2)
    seqs: List[List[int]] = []
    step = max(1, (len(tokens) - seq_len) // max(n_seqs, 1))
    for i in range(n_seqs):
        start = (i * step) % (len(tokens) - seq_len + 1)
        seqs.append(tokens[start: start + seq_len])
    return seqs


def build_calibration_scales(
    payload: dict,
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
    percentile: float = 99.9,
    activation_percentile_overrides: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Return per-node INT8 scales derived from FP32 calibration runs.

    Runs ``_fp32_forward`` on ``n_seqs`` calibration sequences and sets
    each node's scale to ``percentile``-th percentile of |activation| / 127.
    Falls back to a small default for nodes that produce no data.
    """
    model_args = payload["model_args"]
    state_dict = payload["state_dict"]
    seqs = build_calibration_seqs(payload, n_seqs=n_seqs, seq_len=seq_len)

    node_percentiles = dict(activation_percentile_overrides or {})
    # per-node max-abs accumulator
    accum: Dict[str, List[float]] = {}

    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids)
        for name, arr in node_outputs.items():
            arr_f = np.asarray(arr, dtype=np.float32).ravel()
            if arr_f.size == 0:
                continue
            node_pct = float(node_percentiles.get(name, percentile))
            p = float(
                np.percentile(np.abs(arr_f), node_pct)
                if node_pct < 100.0
                else float(np.abs(arr_f).max())
            )
            accum.setdefault(name, []).append(p)

    scales: Dict[str, float] = {}
    for name, vals in accum.items():
        max_abs = float(np.max(vals))
        scales[name] = max(max_abs, 1e-8) / 127.0

    missing = sorted(set(node_percentiles) - set(accum))
    if missing:
        raise ValueError(f"unknown activation percentile override nodes: {missing}")

    _apply_raw_vadd_safe_tok_pos_scale(
        scales,
        state_dict,
        seqs,
        percentile=float(node_percentiles.get("tok_pos_add", percentile)),
    )

    # Nodes that calibration didn't observe → keep the compiler's defaults
    _fill_defaults(scales, model_args)
    return scales


def build_calibration_scales_from_token_ids(
    payload: dict,
    token_ids: Sequence[int],
    *,
    n_seqs: int = 8,
    seq_len: int = 16,
    percentile: float = 99.9,
    activation_percentile_overrides: Dict[str, float] | None = None,
    hessian_gelu_blocks: Sequence[int] = (),
    ln_eps_calibration: float | None = None,
) -> Dict[str, float]:
    """Return per-node INT8 scales from tokenized calibration text."""
    model_args = payload["model_args"]
    state_dict = payload["state_dict"]
    seqs = build_calibration_seqs_from_token_ids(
        token_ids,
        n_seqs=n_seqs,
        seq_len=seq_len,
    )

    hessian_blocks = set(int(L) for L in hessian_gelu_blocks)
    gelu_accum: Dict[int, List[np.ndarray]] = {L: [] for L in hessian_blocks}

    node_percentiles = dict(activation_percentile_overrides or {})
    accum: Dict[str, List[float]] = {}
    for tids in seqs:
        node_outputs = _fp32_forward(state_dict, model_args, tids, ln_eps=ln_eps_calibration)
        for name, arr in node_outputs.items():
            arr_f = np.asarray(arr, dtype=np.float32).ravel()
            if arr_f.size == 0:
                continue
            node_pct = float(node_percentiles.get(name, percentile))
            p = float(
                np.percentile(np.abs(arr_f), node_pct)
                if node_pct < 100.0
                else float(np.abs(arr_f).max())
            )
            accum.setdefault(name, []).append(p)
        for L in hessian_blocks:
            gelu_arr = node_outputs.get(f"block{L}_gelu")
            if gelu_arr is not None:
                gelu_accum[L].append(np.asarray(gelu_arr, dtype=np.float32))

    scales: Dict[str, float] = {}
    for name, vals in accum.items():
        max_abs = float(np.max(vals))
        scales[name] = max(max_abs, 1e-8) / 127.0

    missing = sorted(set(node_percentiles) - set(accum))
    if missing:
        raise ValueError(f"unknown activation percentile override nodes: {missing}")

    _apply_raw_vadd_safe_tok_pos_scale(
        scales,
        state_dict,
        seqs,
        percentile=float(node_percentiles.get("tok_pos_add", percentile)),
    )

    for L in sorted(hessian_blocks):
        if not gelu_accum[L]:
            continue
        fc2_w_key = f"transformer.h.{L}.mlp.c_proj.weight"
        if fc2_w_key not in state_dict:
            continue
        gelu_all = np.concatenate(gelu_accum[L], axis=0)
        fc2_w = _to_f32(state_dict[fc2_w_key])
        scales[f"block{L}_gelu"] = find_hessian_gelu_scale(gelu_all, fc2_w)

    _fill_defaults(scales, model_args)
    return scales


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
        logits = ref.incremental_logits_trace(inputs)
        for row, target in zip(logits, targets):
            deq = np.asarray(row, dtype=np.float32) * np.float32(lm_scale)
            nlls.append(_stable_cross_entropy_np(deq, target, vocab_size=vocab_size))
    if not nlls:
        raise ValueError("output-aware GELU search requires calibration windows with at least two tokens")
    return float(np.mean(np.asarray(nlls, dtype=np.float64)))


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

    search_n_seqs = max(1, min(int(n_seqs), OUTPUT_AWARE_SEARCH_N_SEQS_MAX))
    search_seq_len = max(2, min(int(seq_len), OUTPUT_AWARE_SEARCH_SEQ_LEN_MAX))
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
        candidate_rows: List[Dict[str, float]] = []
        for multiplier in multipliers:
            m = float(multiplier)
            if m <= 0.0:
                continue
            candidate_scales = dict(scales)
            candidate_scales[key] = float(base_scale * m)
            mean_nll = _mean_fake_quant_target_nll(payload, seqs, candidate_scales, ptq_preset=ptq_preset)
            candidate_rows.append({
                "multiplier": m,
                "scale": float(candidate_scales[key]),
                "mean_nll": float(mean_nll),
            })
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
    include_pair_candidates: bool = False,
) -> tuple[Dict[str, float], Dict[str, Dict[str, object]]]:
    """Greedily tune late MLP scale groups against final fake-quant token NLL.

    Each late block searches three valid groups:

    * ``fc1``: changes the FC1 output quantization before GELU.
    * ``gelu``: changes SFU/GELU output quantization before FC2.
    * ``residual_group``: changes ``residual1``, ``fc2``, and ``residual2``
      together, preserving the raw residual2 VADD shared-scale contract.
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
            candidate_rows: List[Dict[str, float]] = []
            for multiplier in multipliers:
                m = float(multiplier)
                if m <= 0.0:
                    continue
                candidate_scales = dict(scales)
                for keys, base_scale in zip(key_groups, base_scales):
                    candidate_scales.update(_candidate_for_group(keys, base_scale, m))
                mean_nll = _mean_fake_quant_target_nll(payload, seqs, candidate_scales, ptq_preset=ptq_preset)
                candidate_rows.append({
                    "multiplier": m,
                    "scales": {
                        keys[0]: float(base_scale * m)
                        for keys, base_scale in zip(key_groups, base_scales)
                    },
                    "mean_nll": float(mean_nll),
                })
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
        diagnostics[f"block{block}"] = {
            "baseline_mean_nll": block_start_nll,
            "selected_mean_nll": float(current_nll),
            "search_n_seqs": int(search_n_seqs),
            "search_seq_len": int(search_seq_len),
            "groups": group_results,
        }
    return scales, diagnostics


def _apply_raw_vadd_safe_tok_pos_scale(
    scales: Dict[str, float],
    state_dict: dict,
    seqs: Sequence[Sequence[int]],
    *,
    percentile: float,
) -> None:
    """Ensure token+position embeddings fit the raw INT8 VADD contract.

    The compiled decoder adds token and position tables with a plain INT8 VADD,
    so each table is quantized at the shared ``tok_pos_add`` scale.  Calibrating
    only ``abs(token + position)`` can under-cover cases where the two operands
    have opposite signs.  This bound uses ``abs(token) + abs(position)`` over the
    same calibration windows so raw integer addition does not clip before the
    downstream layers see the sum.
    """
    if "transformer.wte.weight" not in state_dict or "transformer.wpe.weight" not in state_dict:
        return
    wte = _to_f32(state_dict["transformer.wte.weight"])
    wpe = _to_f32(state_dict["transformer.wpe.weight"])
    vals: List[float] = []
    for tids in seqs:
        token_ids = [int(tok) for tok in tids]
        if not token_ids:
            continue
        pos_ids = list(range(len(token_ids)))
        if max(token_ids) >= wte.shape[0] or len(pos_ids) > wpe.shape[0]:
            continue
        conservative = np.abs(wte[token_ids]) + np.abs(wpe[pos_ids])
        vals.append(float(
            np.percentile(conservative.ravel(), percentile)
            if percentile < 100.0
            else np.max(conservative)
        ))
    if not vals:
        return
    raw_vadd_scale = max(float(np.max(vals)), 1e-8) / 127.0
    scales["tok_pos_add"] = max(float(scales.get("tok_pos_add", 0.0)), raw_vadd_scale)


def _fill_defaults(scales: Dict[str, float], model_args: dict) -> None:
    """Ensure every compiler calibration_scales key has an entry."""
    n_layer = int(model_args["n_layer"])
    n_head = int(model_args["n_head"])

    def _d(name: str, default: float = _DEFAULT_SCALES) -> None:
        scales.setdefault(name, default)

    _d("tok_embed")
    _d("pos_embed")
    _d("tok_pos_add")
    for L in range(n_layer):
        _d(f"block{L}_ln1")
        for H in range(n_head):
            _d(f"block{L}_head{H}_query")
            _d(f"block{L}_head{H}_key")
            _d(f"block{L}_head{H}_value")
            _d(f"block{L}_head{H}_softmax", _SFU_DEFAULT_SCALES)
            _d(f"block{L}_head{H}_attn_v")
            # kv_load nodes in the decode graph substitute for key/value matmul
            # outputs.  They carry the same INT8 values so must use the same scale.
            scales.setdefault(
                f"block{L}_head{H}_key_kv_load",
                scales.get(f"block{L}_head{H}_key", _DEFAULT_SCALES),
            )
            scales.setdefault(
                f"block{L}_head{H}_value_kv_load",
                scales.get(f"block{L}_head{H}_value", _DEFAULT_SCALES),
            )
        _d(f"block{L}_concat")
        _d(f"block{L}_out_proj")
        _d(f"block{L}_residual1")
        _d(f"block{L}_ln2")
        _d(f"block{L}_fc1")
        _d(f"block{L}_gelu", _SFU_DEFAULT_SCALES)
        _d(f"block{L}_fc2")
        _d(f"block{L}_residual2")
    _d("ln_f")
    _d("lm_head")
