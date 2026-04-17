"""Calibration: collect activation ranges from FP32 model."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


def _sample_abs_values(array: np.ndarray, sample_cap: int) -> np.ndarray:
    flat = np.abs(np.asarray(array, dtype=np.float32)).reshape(-1)
    if flat.size <= sample_cap:
        return flat
    idx = np.linspace(0, flat.size - 1, num=sample_cap, dtype=np.int64)
    return flat[idx]


@dataclass
class CalibrationResult:
    """Per-tensor activation scale from calibration."""
    # Maps tensor name → max absolute value
    max_abs: Dict[str, float] = field(default_factory=dict)
    # Optional sampled absolute values for percentile-based scale selection
    abs_samples: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    # Maps tensor name → per-tensor scale (max_abs / 127)
    scales: Dict[str, float] = field(default_factory=dict)

    def add_observation(self, name: str, tensor_abs_max: float, abs_values: Optional[np.ndarray] = None):
        """Record observed max absolute value, keeping the running maximum."""
        current = self.max_abs.get(name, 0.0)
        self.max_abs[name] = max(current, tensor_abs_max)
        if abs_values is not None and abs_values.size:
            self.abs_samples.setdefault(name, []).append(abs_values.astype(np.float32, copy=False))

    def compute_scales(self, percentile_overrides: Optional[Dict[str, float]] = None):
        """Compute per-tensor INT8 scales from observed max values."""
        overrides = percentile_overrides or {}
        for name, max_val in self.max_abs.items():
            if name in overrides and self.abs_samples.get(name):
                pooled = np.concatenate(self.abs_samples[name], axis=0)
                max_val = float(np.percentile(pooled, overrides[name]))
            self.scales[name] = max(max_val, 1e-8) / 127.0

    def get_scale(self, name: str) -> float:
        return self.scales.get(name, 1.0 / 127.0)

    def percentile_scale(self, name: str, percentile: float) -> float:
        if not self.abs_samples.get(name):
            raise KeyError(f"No percentile samples recorded for calibration tensor '{name}'")
        pooled = np.concatenate(self.abs_samples[name], axis=0)
        return max(float(np.percentile(pooled, percentile)), 1e-8) / 127.0


def calibrate_model(
    model,
    sample_inputs: list,
    *,
    percentile_module_names: Optional[Iterable[str]] = None,
    percentile_sample_cap: int = 2048,
) -> CalibrationResult:
    """Run calibration on a PyTorch model with sample inputs.

    Hooks every linear/layernorm layer to record activation ranges.
    """
    import torch

    result = CalibrationResult()
    hooks = []
    percentile_targets = set(percentile_module_names or [])

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                val = output.detach().abs().max().item()
                abs_values = None
                if name in percentile_targets:
                    abs_values = _sample_abs_values(output.detach().cpu().numpy(), percentile_sample_cap)
                result.add_observation(name, val, abs_values=abs_values)
            # Also record input range
            if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
                val = input[0].detach().abs().max().item()
                result.add_observation(f"{name}_input", val)
        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            if hasattr(inp, 'items'):
                model(**dict(inp.items()))
            else:
                model(inp)

    # Remove hooks
    for h in hooks:
        h.remove()

    result.compute_scales()
    return result


def collect_layer_inputs(model, sample_inputs: list, module_names: Iterable[str]) -> Dict[str, List[np.ndarray]]:
    """Collect FP32 input activations for selected module names."""
    import torch

    modules = dict(model.named_modules())
    targets = list(module_names)
    missing = [name for name in targets if name not in modules]
    if missing:
        raise KeyError(f"Missing modules for input collection: {missing}")

    collected: Dict[str, List[np.ndarray]] = {name: [] for name in targets}
    hooks = []

    def make_hook(name: str):
        def _hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if torch.is_tensor(x):
                collected[name].append(x.detach().cpu().numpy().astype(np.float32))
        return _hook

    for name in targets:
        hooks.append(modules[name].register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            if hasattr(inp, "items"):
                model(**dict(inp.items()))
            else:
                model(inp)

    for hook in hooks:
        hook.remove()

    return collected
