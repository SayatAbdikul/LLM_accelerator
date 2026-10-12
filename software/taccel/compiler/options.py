"""Explicit compiler targets and independently measurable optimizations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CompilerOptions:
    """Opt-in candidates; the default is the corrected baseline schedule."""

    indexed_liveness: bool = False
    real_row_fc2_loads: bool = False
    reuse_qkv_quantization: bool = False
    reuse_fc2_quantization: bool = False
    shared_data_layout: bool = False
    deduplicate_data: bool = False


COMPILER_TARGETS = frozenset({"rtl-gen2", "research-golden"})
