"""Focused tests for Stage 5 GPT-2 quantization policy fixes."""

import numpy as np

from taccel.runtime.calibration import build_calibration_scales_from_token_ids
from taccel.runtime.fake_quant_reference import _requant_accum_pc_int8


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
