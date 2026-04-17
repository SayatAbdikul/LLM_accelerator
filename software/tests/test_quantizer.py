"""Tests for quantizer."""
import pytest
import numpy as np
import torch
import torch.nn as nn
from taccel.quantizer.quantize import (
    adaround_greedy,
    dequantize_tensor,
    quantize_tensor,
    quantize_tensor_clipped,
    quantize_weights,
)
from taccel.quantizer.calibrate import CalibrationResult
from taccel.quantizer.fake_quant import ActivationQuantizer
from taccel.quantizer.hessian_guided import (
    gelu_fc2_hessian_diag,
    softmax_attn_v_hessian_diag,
    weighted_quant_error_score,
)
from taccel.quantizer.scales import ScalePropagator
from taccel.quantizer.twin_uniform import (
    quantize_dequant_gelu_twin,
    quantize_dequant_softmax_twin,
)


class TestQuantize:
    def test_basic_quantization(self):
        """Quantize and dequantize, error ≤ 1 LSB."""
        np.random.seed(42)
        W = np.random.randn(64, 128).astype(np.float32) * 2.0
        q, scales = quantize_tensor(W)

        assert q.dtype == np.int8
        assert scales.dtype == np.float16
        assert q.shape == W.shape
        assert len(scales) == 64  # per-channel

        # Dequantize and check error
        W_rec = dequantize_tensor(q, scales)
        lsb = scales.astype(np.float32).reshape(-1, 1)
        err = np.abs(W - W_rec)
        assert np.all(err <= lsb), f"Max error {err.max():.6f} > 1 LSB {lsb.max():.6f}"

    def test_quantize_range(self):
        """All quantized values in [-128, 127]."""
        W = np.random.randn(32, 64).astype(np.float32) * 10.0
        q, scales = quantize_tensor(W)
        assert q.min() >= -128 and q.max() <= 127

    def test_per_channel_scales(self):
        """Each channel has independent scale."""
        W = np.zeros((4, 8), dtype=np.float32)
        W[0, 0] = 1.0    # max abs = 1
        W[1, 0] = 10.0   # max abs = 10
        W[2, 0] = 100.0  # max abs = 100
        W[3, 0] = 0.01   # max abs = 0.01
        q, scales = quantize_tensor(W)
        # Scales should be proportional to max abs
        assert abs(float(scales[0]) - 1.0/127) < 1e-4
        assert abs(float(scales[1]) - 10.0/127) < 0.1
        assert abs(float(scales[2]) - 100.0/127) < 1.0

    def test_zero_tensor(self):
        """Zero tensor quantizes to zeros."""
        W = np.zeros((8, 16), dtype=np.float32)
        q, scales = quantize_tensor(W)
        assert np.all(q == 0)

    def test_conv_reshape(self):
        """Conv2d weights reshaped to 2D before quantization."""
        W = np.random.randn(192, 3, 16, 16).astype(np.float32)
        W_2d = W.reshape(192, -1)  # [192, 768]
        q, scales = quantize_tensor(W_2d)
        assert q.shape == (192, 768)
        assert len(scales) == 192

    def test_output_aware_clipping_reduces_output_mse(self):
        """Clipping should help when large outliers matter less than typical inputs."""
        W = np.array(
            [
                [12.0, 0.80, -0.65, 0.55],
                [-11.0, -0.75, 0.70, -0.45],
            ],
            dtype=np.float32,
        )
        calibration_inputs = [
            np.array(
                [
                    [0.01, 1.2, -0.8, 0.7],
                    [0.00, -1.0, 0.9, -0.6],
                    [0.02, 0.8, -0.4, 1.1],
                ],
                dtype=np.float32,
            )
        ]

        q_base, s_base = quantize_tensor(W, per_channel=True)
        q_clip, s_clip = quantize_tensor_clipped(
            W,
            calibration_inputs=calibration_inputs,
            per_channel=True,
            n_candidates=31,
            alpha_min=0.2,
        )

        x = calibration_inputs[0]
        y_fp32 = x @ W.T
        y_base = x @ dequantize_tensor(q_base, s_base).T
        y_clip = x @ dequantize_tensor(q_clip, s_clip).T

        mse_base = float(np.mean((y_fp32 - y_base) ** 2))
        mse_clip = float(np.mean((y_fp32 - y_clip) ** 2))
        assert mse_clip < mse_base

    def test_quantize_weights_applies_targeted_clipping_override(self):
        state_dict = {
            "fc.weight": torch.from_numpy(np.array(
                [[10.0, 0.7, -0.5], [-9.0, -0.6, 0.4]],
                dtype=np.float32,
            ))
        }
        overrides = {
            "fc.weight": {
                "mode": "output_aware_clipping",
                "per_channel": True,
                "n_candidates": 21,
                "alpha_min": 0.2,
                "calibration_inputs": [np.array([[0.0, 1.0, -1.0]], dtype=np.float32)],
            }
        }

        q_default, s_default = quantize_weights(state_dict)["fc.weight"]
        q_clipped, s_clipped = quantize_weights(state_dict, quantization_overrides=overrides)["fc.weight"]

        assert q_clipped.shape == q_default.shape
        assert s_clipped.shape == s_default.shape
        assert not np.array_equal(q_clipped, q_default) or not np.array_equal(s_clipped, s_default)

    def test_adaround_greedy_reduces_output_mse(self):
        W = np.array(
            [
                [0.55, -1.45, 0.25, 0.75],
                [-0.62, 1.38, -0.48, 0.18],
            ],
            dtype=np.float32,
        )
        calibration_inputs = [
            np.array(
                [
                    [1.0, -0.6, 0.3, 0.9],
                    [-0.7, 0.8, -0.4, 0.5],
                    [0.6, 0.7, 0.2, -0.3],
                ],
                dtype=np.float32,
            )
        ]

        q_clip, s_clip = quantize_tensor_clipped(
            W,
            calibration_inputs=calibration_inputs,
            per_channel=True,
            n_candidates=7,
            alpha_min=1.0,
        )
        q_ada = adaround_greedy(
            W,
            q_clip,
            s_clip,
            calibration_inputs,
            frac_lo=0.2,
            frac_hi=0.8,
        )

        x = calibration_inputs[0]
        y_fp32 = x @ W.T
        y_clip = x @ dequantize_tensor(q_clip, s_clip).T
        y_ada = x @ dequantize_tensor(q_ada, s_clip).T

        mse_clip = float(np.mean((y_fp32 - y_clip) ** 2))
        mse_ada = float(np.mean((y_fp32 - y_ada) ** 2))
        assert mse_ada <= mse_clip

    def test_output_aware_clipping_can_improve_fc2_like_per_tensor_output_mse(self):
        W = np.array(
            [
                [18.0, 0.9, -0.8, 0.6],
                [-16.0, -0.7, 0.5, -0.4],
            ],
            dtype=np.float32,
        )
        calibration_inputs = [
            np.array(
                [
                    [0.01, -0.4, 0.3, 0.1],
                    [0.02, 0.6, -0.5, 0.4],
                    [0.00, 0.7, 0.2, -0.3],
                ],
                dtype=np.float32,
            )
        ]

        q_base, s_base = quantize_tensor(W, per_channel=False)
        q_clip, s_clip = quantize_tensor_clipped(
            W,
            calibration_inputs=calibration_inputs,
            per_channel=False,
            n_candidates=31,
            alpha_min=0.2,
        )

        x = calibration_inputs[0]
        y_fp32 = x @ W.T
        y_base = x @ dequantize_tensor(q_base, s_base).T
        y_clip = x @ dequantize_tensor(q_clip, s_clip).T

        mse_base = float(np.mean((y_fp32 - y_base) ** 2))
        mse_clip = float(np.mean((y_fp32 - y_clip) ** 2))
        assert mse_clip < mse_base

    def test_output_aware_clipping_can_improve_classifier_like_mse(self):
        W = np.array(
            [
                [14.0, 0.12, -0.10],
                [-13.5, -0.08, 0.11],
                [0.2, 0.25, -0.18],
            ],
            dtype=np.float32,
        )
        calibration_inputs = [
            np.array(
                [
                    [0.0, 0.8, -0.7],
                    [0.01, -0.5, 0.6],
                    [0.0, 0.3, -0.2],
                ],
                dtype=np.float32,
            )
        ]

        q_base, s_base = quantize_tensor(W, per_channel=False)
        q_clip, s_clip = quantize_tensor_clipped(
            W,
            calibration_inputs=calibration_inputs,
            per_channel=False,
            n_candidates=41,
            alpha_min=0.15,
        )

        x = calibration_inputs[0]
        y_fp32 = x @ W.T
        y_base = x @ dequantize_tensor(q_base, s_base).T
        y_clip = x @ dequantize_tensor(q_clip, s_clip).T

        mse_base = float(np.mean((y_fp32 - y_base) ** 2))
        mse_clip = float(np.mean((y_fp32 - y_clip) ** 2))
        assert mse_clip < mse_base

    def test_twin_uniform_softmax_is_monotonic_bounded_and_deterministic(self):
        x = np.array([[0.0, 0.01, 0.03, 0.10, 0.35, 0.90]], dtype=np.float32)
        qdq0, meta0 = quantize_dequant_softmax_twin(x, 0.10, return_metadata=True)
        qdq1, meta1 = quantize_dequant_softmax_twin(x, 0.10, return_metadata=True)

        assert np.all(qdq0 >= 0.0)
        assert np.all(qdq0 <= 1.0)
        assert np.all(np.diff(qdq0[0]) >= -1e-7)
        np.testing.assert_allclose(qdq0, qdq1)
        assert meta0 == meta1

    def test_twin_uniform_gelu_is_deterministic_and_preserves_sign_regions(self):
        x = np.array([[-1.20, -0.40, -0.05, 0.0, 0.20, 1.80]], dtype=np.float32)
        qdq0, meta0 = quantize_dequant_gelu_twin(x, 1.0, negative_extent=1.5, return_metadata=True)
        qdq1, meta1 = quantize_dequant_gelu_twin(x, 1.0, negative_extent=1.5, return_metadata=True)

        np.testing.assert_allclose(qdq0, qdq1)
        assert meta0 == meta1
        assert np.all(qdq0[x < 0.0] <= 1e-7)
        assert np.all(qdq0[x > 0.0] >= -1e-7)

    def test_hessian_weighted_score_prefers_identical_reconstruction(self):
        ref = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
        diag = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        same = ref.copy()
        perturbed = ref + np.array([[0.01, -0.04], [0.02, 0.03]], dtype=np.float32)

        same_score = weighted_quant_error_score(ref, same, diag)
        perturbed_score = weighted_quant_error_score(ref, perturbed, diag)

        assert same_score == pytest.approx(0.0)
        assert perturbed_score > same_score

    def test_hessian_diag_helpers_match_reference_shape(self):
        softmax = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
        value = np.array([[1.0, -2.0, 0.5], [0.2, 0.4, -0.3]], dtype=np.float32)
        gelu = np.array([[0.1, 0.3, 0.6]], dtype=np.float32)
        fc2 = np.array([[0.2, -0.5, 0.7], [0.1, 0.3, -0.2]], dtype=np.float32)

        soft_h = softmax_attn_v_hessian_diag(softmax, value)
        gelu_h = gelu_fc2_hessian_diag(gelu, fc2)

        assert soft_h.shape == softmax.shape
        assert gelu_h.shape == gelu.shape
        assert np.all(soft_h >= 0.0)
        assert np.all(gelu_h >= 0.0)

    def test_activation_quantizer_can_patch_tuple_outputs_for_softmax_specs(self):
        class TupleModule(nn.Module):
            def forward(self, x):
                return x, x + 0.5

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.tuple_mod = TupleModule()

            def forward(self, x):
                return self.tuple_mod(x)

        model = Toy()
        quantizer = ActivationQuantizer(
            scales={},
            twin_uniform_specs={
                "tuple_mod": {
                    "mode": "softmax",
                    "range1_max": 0.25,
                    "tuple_index": 1,
                }
            },
        )
        quantizer.attach(model)
        try:
            out0, out1 = model(torch.tensor([[0.05, 0.85]], dtype=torch.float32))
        finally:
            quantizer.remove()

        np.testing.assert_allclose(out0.numpy(), np.array([[0.05, 0.85]], dtype=np.float32))
        assert np.all(out1.numpy() >= 0.0)
        assert np.all(out1.numpy() <= 1.0)

    def test_activation_quantizer_softmax_patch_changes_attention_context(self):
        class ToySelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_attention_heads = 2
                self.attention_head_size = 2
                self.all_head_size = 4
                self.scaling = self.attention_head_size**-0.5
                self.dropout_prob = 0.0
                self.query = nn.Linear(4, 4, bias=False)
                self.key = nn.Linear(4, 4, bias=False)
                self.value = nn.Linear(4, 4, bias=False)

            def forward(self, hidden_states):
                batch_size = hidden_states.shape[0]
                new_shape = (
                    batch_size,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
                value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
                query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)
                attention_probs = torch.matmul(query_layer, key_layer.transpose(2, 3)) * self.scaling
                attention_probs = torch.softmax(attention_probs, dim=-1)
                context_layer = torch.matmul(attention_probs, value_layer)
                context_layer = context_layer.transpose(1, 2).contiguous()
                return context_layer.reshape(batch_size, -1, self.all_head_size), attention_probs

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = ToySelfAttention()

        torch.manual_seed(0)
        model = Toy().eval()
        x = torch.tensor(
            [[[0.1, -0.2, 0.3, 0.4], [0.5, -0.4, 0.2, -0.1], [-0.3, 0.7, -0.6, 0.2]]],
            dtype=torch.float32,
        )

        baseline_context, baseline_probs = model.attn(x)

        quantizer = ActivationQuantizer(
            scales={},
            twin_uniform_specs={
                "attn": {
                    "mode": "softmax",
                    "range1_max": 0.05,
                }
            },
        )
        quantizer.attach(model)
        try:
            patched_context, patched_probs = model.attn(x)
        finally:
            quantizer.remove()

        restored_context, restored_probs = model.attn(x)

        assert not np.allclose(patched_context.detach().numpy(), baseline_context.detach().numpy())
        assert not np.allclose(patched_probs.detach().numpy(), baseline_probs.detach().numpy())
        np.testing.assert_allclose(
            restored_context.detach().numpy(),
            baseline_context.detach().numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            restored_probs.detach().numpy(),
            baseline_probs.detach().numpy(),
            atol=1e-6,
        )


class TestScalePropagator:
    def test_prescale_bias(self):
        """Bias pre-scaling: bias_int32[ch] = round(bias_fp32[ch] / (act_scale * w_scale[ch]))."""
        sp = ScalePropagator()
        bias_fp32 = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        act_scale = np.array([0.1])
        w_scales = np.array([0.05, 0.1, 0.02], dtype=np.float32)

        bias_int32 = sp.prescale_bias(bias_fp32, act_scale, w_scales)
        assert bias_int32.dtype == np.int32

        # Verify: bias_int32[ch] * act_scale * w_scales[ch] ≈ bias_fp32[ch]
        for ch in range(3):
            recovered = bias_int32[ch] * float(act_scale[0]) * float(w_scales[ch])
            assert abs(recovered - bias_fp32[ch]) < 0.01, \
                f"Channel {ch}: recovered={recovered:.4f}, expected={bias_fp32[ch]:.4f}"

    def test_matmul_output_scale(self):
        sp = ScalePropagator()
        act_scale = np.array([0.05])
        w_scales = np.array([0.1, 0.2], dtype=np.float32)
        out_scale = sp.compute_matmul_output_scale(act_scale, w_scales)
        expected = np.array([0.005, 0.01])
        np.testing.assert_allclose(out_scale, expected, rtol=1e-5)


class TestCalibrationResult:
    def test_compute_scales_supports_percentile_overrides(self):
        result = CalibrationResult()
        result.add_observation(
            "vit.layernorm",
            10.0,
            abs_values=np.array([1.0, 2.0, 3.0, 9.0], dtype=np.float32),
        )
        result.compute_scales(percentile_overrides={"vit.layernorm": 50.0})

        assert result.get_scale("vit.layernorm") == pytest.approx(np.percentile([1.0, 2.0, 3.0, 9.0], 50.0) / 127.0)
