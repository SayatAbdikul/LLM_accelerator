"""Focused tests for Stage 5 GPT-2 quantization policy fixes."""

import pytest
import numpy as np

from taccel.runtime.calibration import (
    choose_fc2_aware_gelu_scale,
    build_calibration_scales_from_token_ids,
)
from taccel.runtime.fake_quant_reference import _fp32_to_int8, _requant_accum_pc_int8


def _zero_layer_payload():
    d_model = 4
    wte = np.array([
        [1.0, -1.0, 1.0, -1.0],
        [0.5, -0.5, 0.5, -0.5],
    ], dtype=np.float32)
    # Opposite signs make token + position look small while raw INT8 VADD can
    # still clip unless calibration accounts for abs(token) + abs(position).
    wpe = np.array([
        [-1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ], dtype=np.float32)
    return {
        "model_args": {
            "n_layer": 0,
            "n_head": 1,
            "n_embd": d_model,
            "vocab_size": 2,
            "block_size": 4,
            "layer_norm_epsilon": 1e-5,
        },
        "state_dict": {
            "transformer.wte.weight": wte,
            "transformer.wpe.weight": wpe,
            "transformer.ln_f.weight": np.ones(d_model, dtype=np.float32),
            "transformer.ln_f.bias": np.zeros(d_model, dtype=np.float32),
            "lm_head.weight": np.ones((2, d_model), dtype=np.float32),
        },
    }


def test_tok_pos_add_calibration_is_raw_vadd_safe():
    payload = _zero_layer_payload()

    scales = build_calibration_scales_from_token_ids(
        payload,
        [0, 1, 0, 1],
        n_seqs=1,
        seq_len=2,
        percentile=100.0,
    )

    assert scales["tok_pos_add"] >= (2.0 / 127.0)


def test_requant_accum_pc_int8_uses_one_scale_per_output_column():
    accum = np.array([[10, -20, 30, -40]], dtype=np.int32)
    pc_scales = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float16)

    got = _requant_accum_pc_int8(accum, pc_scales)
    expected = np.clip(
        np.round(accum.astype(np.float32) * pc_scales.astype(np.float32).reshape(1, -1)),
        -128,
        127,
    ).astype(np.int8)

    np.testing.assert_array_equal(got, expected)


def test_fc2_aware_gelu_scale_search_selects_known_best_multiplier():
    base_scale = 0.1
    true_scale = base_scale * 1.25
    out_scale = 0.1
    gelu = np.array([[0.22, -0.31], [0.47, 0.09]], dtype=np.float32)
    residual1 = np.zeros_like(gelu, dtype=np.float32)
    proj_w_q = np.eye(2, dtype=np.int8)
    proj_w_scales = np.ones(2, dtype=np.float32)
    gelu_i8 = _fp32_to_int8(gelu, true_scale)
    accum = gelu_i8.astype(np.int32) @ proj_w_q.astype(np.int32).T
    requant_pc = np.asarray([true_scale / out_scale, true_scale / out_scale], dtype=np.float32)
    fc2_i8 = _requant_accum_pc_int8(accum, requant_pc)
    residual2 = fc2_i8.astype(np.float32) * np.float32(out_scale)
    multipliers = (0.75, 1.0, 1.25, 1.5)

    result = choose_fc2_aware_gelu_scale(
        gelu=gelu,
        residual1=residual1,
        residual2=residual2,
        proj_w_q=proj_w_q,
        proj_w_scales=proj_w_scales,
        proj_b_i32_by_scale={base_scale * m: np.zeros(2, dtype=np.int32) for m in multipliers},
        base_scale=base_scale,
        fc2_scale=out_scale,
        residual1_scale=out_scale,
        residual2_scale=out_scale,
        multipliers=multipliers,
    )

    assert result["multiplier"] == pytest.approx(1.25)
    assert result["objective_mse"] == pytest.approx(0.0)
