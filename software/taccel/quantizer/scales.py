"""Scale propagation through the computation graph and bias pre-scaling."""
import numpy as np
from typing import Dict, Tuple, Optional


class ScalePropagator:
    """Track and propagate quantization scales through the graph."""

    def __init__(self):
        # Maps tensor name → per-channel or per-tensor scale (FP32)
        self.scales: Dict[str, np.ndarray] = {}

    def set_scale(self, name: str, scale: np.ndarray):
        self.scales[name] = scale.astype(np.float32)

    def get_scale(self, name: str) -> np.ndarray:
        return self.scales[name]

    def compute_matmul_output_scale(self, act_scale: np.ndarray, weight_scale: np.ndarray) -> np.ndarray:
        """After matmul: output_scale = act_scale × weight_scale (per-channel).

        act_scale: per-tensor scalar (or 1-element array)
        weight_scale: per-output-channel array
        """
        act_s = float(act_scale.flat[0]) if hasattr(act_scale, 'flat') else float(act_scale)
        return act_s * weight_scale.astype(np.float32)

    def prescale_bias(self, bias_fp32: np.ndarray, act_scale: np.ndarray,
                      weight_scale: np.ndarray) -> np.ndarray:
        """Pre-scale bias to INT32: bias_int32[ch] = round(bias_fp32[ch] / (act_scale × weight_scale[ch])).

        This allows bias to be added directly in the INT32 accumulator domain.
        """
        act_s = float(act_scale.flat[0]) if hasattr(act_scale, 'flat') else float(act_scale)
        combined_scale = act_s * weight_scale.astype(np.float32)
        # Avoid division by zero
        combined_scale = np.maximum(np.abs(combined_scale), 1e-10) * np.sign(combined_scale + 1e-20)
        bias_int32 = np.round(bias_fp32.astype(np.float32) / combined_scale).astype(np.int32)
        return bias_int32

    def compute_requant_scale(self, matmul_output_scale: np.ndarray,
                               target_scale: float) -> np.ndarray:
        """Compute requantization scale to convert from INT32 accumulator to INT8.

        requant_scale = matmul_output_scale / target_scale
        After multiplying INT32 by this scale and rounding, we get INT8 at target_scale.
        Actually: requant_scale[ch] = (act_scale * weight_scale[ch]) / target_scale
        But we want: int8_out = round(int32_accum * requant_scale)
        Wait, the requant operation is: int8 = clip(round(int32 * scale))
        The int32 value represents: real_value / (act_scale * weight_scale[ch])
        We want int8 to represent: real_value / target_scale
        So: int8 = int32 * (act_scale * weight_scale[ch]) / target_scale
        """
        return matmul_output_scale / target_scale

    def choose_activation_scale(self, max_abs: float) -> float:
        """Choose INT8 scale for an activation tensor.

        scale = max_abs / 127
        """
        return max(max_abs, 1e-8) / 127.0
