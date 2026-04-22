"""Runtime helpers for decoder ProgramBundle execution."""

from .host_runner import HostRunner
from .tiny_fixture import run_nanogpt_fp32_e2e, run_stage3_tiny_e2e, run_stage3g_tiny_e2e
from .fake_quant_reference import NanoGPTFQReference
from .fp32_reference import NanoGPTFP32Reference
from .calibration import build_calibration_scales

__all__ = [
    "HostRunner",
    "run_nanogpt_fp32_e2e",
    "run_stage3_tiny_e2e",
    "run_stage3g_tiny_e2e",
    "NanoGPTFQReference",
    "NanoGPTFP32Reference",
    "build_calibration_scales",
]
