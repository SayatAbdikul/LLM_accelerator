"""Attention-projection emit helpers (mask-mode + CONFIG_ATTN site).

Both functions are pure delegations of the original
`CodeGenerator._attention_mask_mode_for_qkt` and
`CodeGenerator._emit_config_attn_for_qkt`. They are only invoked from
the legacy W8A8 branch of `_emit_qkt` (now unreachable because
`use_fp16_activations` is hardcoded True) — but kept here for source
parity with the original file and because the W8A16 sibling at
`compiler/w8a16_emit/sublayer.py` mirrors their logic in comments.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...assembler.assembler import RuntimeConfigAttnSite
from ...isa.instructions import ConfigAttnInsn
from ..ir import IRNode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..codegen import CodeGenerator


def attention_mask_mode_for_qkt(cg: "CodeGenerator", node: IRNode,
                                key_pad: int) -> Optional[int]:
    """Return CONFIG_ATTN mode for a masked QKT node, or None for legacy attention."""
    if not node.attrs.get("masked", False):
        return None
    if node.attrs.get("runtime_config_attn", False):
        return 0b11
    key_len = int(node.attrs.get("key_len", node.output_shape[1] if len(node.output_shape) > 1 else node.output_shape[0]))
    if key_pad == key_len:
        return 0b10
    return 0b11


def emit_config_attn_for_qkt(cg: "CodeGenerator", node: IRNode, *,
                             row_start: int, valid_kv_len: int,
                             mode: int) -> None:
    pc = len(cg.instructions)
    if node.attrs.get("runtime_config_attn", False):
        cg._emit(ConfigAttnInsn(query_row_base=0, valid_kv_len=1, mode=mode))
        cg.runtime_config_attn_sites.append(RuntimeConfigAttnSite(
            stream=cg.stream_name,
            local_pc=pc,
            absolute_pc=0,
            mode=mode,
        ))
    else:
        cg._emit(ConfigAttnInsn(
            query_row_base=row_start,
            valid_kv_len=valid_kv_len,
            mode=mode,
        ))
