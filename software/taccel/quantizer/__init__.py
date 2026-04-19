"""Quantizer package exports.

Heavy optional helpers are imported lazily so lightweight golden-model tests
can import small quantizer submodules without importing torch at collection.
"""

from .quantize import (
    adaround_greedy,
    quantize_weights,
    quantize_tensor,
    quantize_tensor_clipped,
)
from .scales import ScalePropagator
from .calibrate import calibrate_model, CalibrationResult, collect_layer_inputs
from .twin_uniform import (
    quantize_dequant_gelu_twin,
    quantize_dequant_softmax_twin,
)
from .hessian_guided import (
    gelu_fc2_hessian_diag,
    softmax_attn_v_hessian_diag,
    weighted_quant_error_score,
)


def __getattr__(name):
    if name in {"compute_smooth_factors", "apply_smooth_quant"}:
        from .smooth_quant import compute_smooth_factors, apply_smooth_quant
        values = {
            "compute_smooth_factors": compute_smooth_factors,
            "apply_smooth_quant": apply_smooth_quant,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "adaround_greedy",
    "quantize_weights",
    "quantize_tensor",
    "quantize_tensor_clipped",
    "ScalePropagator",
    "calibrate_model",
    "CalibrationResult",
    "collect_layer_inputs",
    "compute_smooth_factors",
    "apply_smooth_quant",
    "quantize_dequant_gelu_twin",
    "quantize_dequant_softmax_twin",
    "gelu_fc2_hessian_diag",
    "softmax_attn_v_hessian_diag",
    "weighted_quant_error_score",
]
