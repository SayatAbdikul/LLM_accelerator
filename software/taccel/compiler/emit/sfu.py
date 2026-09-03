"""SFU emit helpers for the active W8A16 lowering path."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..ir import IRNode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def emit_scale_mul(cg: "CodeGenerator", node: IRNode) -> None:
    """C1: scale_mul is metadata-only; scaling is folded into the QKT
    epilogue (INT8 path: via SOFTMAX's input scale; W8A32 path
    (M3-A): via DEQUANT_ACCUM_FP32's composite PC scale vector).
    Either way this node is a rename — propagate the allocation
    from `node.inputs[0]` to `node.name`."""
    # INT8 path: QKT writes its softmax output to WBUF (INT8).
    in_wbuf = cg.mem.wbuf.get(node.inputs[0])
    if in_wbuf is not None:
        # Rename in-place: pop the old key and re-insert under the new name
        # without touching the free list (free() would double-book the region).
        alloc = cg.mem.wbuf.allocations.pop(node.inputs[0])
        cg.mem.wbuf.allocations[node.name] = alloc
    else:
        # W8A32 path (M3-A): QKT writes its FP32 scores tile to ABUF.
        # Same rename mechanism, just in the ABUF allocator.
        in_abuf = cg.mem.abuf.get(node.inputs[0])
        if in_abuf is not None:
            alloc = cg.mem.abuf.allocations.pop(node.inputs[0])
            cg.mem.abuf.allocations[node.name] = alloc

    # Propagate scale metadata for downstream rename nodes.
    cg.calibration_scales[node.name] = cg.calibration_scales.get(
        node.inputs[0], 6.0 / 127.0
    )


def emit_softmax(cg: "CodeGenerator", node: IRNode) -> None:
    """Softmax (W8A16): dispatches to the FP32 lowering helper.

    Causal detection from node.attrs — MASKED_SOFTMAX_FP32 for causal,
    SOFTMAX_FP32 (0x1C, non-causal) for the unmasked case (the latter is
    currently illegal in RTL; non-causal models are out-of-target).
    """
    from ..w8a16_emit import emit_softmax_fp32
    masked = bool(node.attrs.get("causal", False)) or bool(node.attrs.get("masked", False))
    emit_softmax_fp32(cg, node, masked=masked)


def emit_gelu(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit GELU SFU instruction (W8A16: dispatches to FP32 lowering helper).

    W8A16 force-disables gelu_from_accum + strip-mined inlining, so the
    simple GELU path is what we want here.
    """
    from ..w8a16_emit import emit_gelu_fp32
    emit_gelu_fp32(cg, node)


def emit_layernorm(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit LAYERNORM SFU instruction (W8A16: dispatches to FP32 lowering helper)."""
    from ..w8a16_emit import emit_layernorm_fp32
    emit_layernorm_fp32(cg, node)


def emit_vadd(cg: "CodeGenerator", node: IRNode) -> None:
    """Emit VADD for residual connection (W8A16: non-saturating VADD_FP32).

    dequant_add residual paths are force-disabled (gen-1-only); the W8A16
    path uses the FP32 lowering helper.
    """
    if node.name in cg.precomputed_nodes:
        return
    from ..w8a16_emit import emit_vadd_fp32
    emit_vadd_fp32(cg, node)
