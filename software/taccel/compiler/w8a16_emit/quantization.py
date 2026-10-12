"""Activation quantization with bounded, explicit QKV cache lifetimes."""
from ...isa.instructions import ConfigTileInsn, MaxAbsReduceFp32Insn, QuantFp32Int8Insn, SyncInsn
from ...isa.opcodes import BUF_ABUF, ABUF_SIZE
from ..tiler import TILE
from ._common import _abuf_alloc_fp32, _m_exact_rows, _zero_fill_fp32_padding_rows


def prepare_matmul_activation(cg, node, M, M_pad, N_pad, K_pad):
    from .matmul import _act_quant_tiles

    input_name = node.inputs[0]
    signature = (input_name, M, M_pad, K_pad, cg.fp_precision_flag)
    cached = cg.quantization_cache.get(input_name)
    if cached is not None and cached["signature"] == signature:
        out = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
        cg.optimization_events.append({"kind": "qkv_quantization_hit", "node": node.name})
        return None, out, cached["allocation"], cached["sreg"], True

    if input_name in cg.dram_temp_fp32_outputs and cg.mem.abuf.get(input_name) is None:
        inp = cg._load_dram_to_abuf_fp(input_name, M_pad, K_pad)
    else:
        inp = _abuf_alloc_fp32(cg, input_name, M_pad, K_pad)
    out = _abuf_alloc_fp32(cg, node.name, M_pad, N_pad)
    scratch = cg.mem.abuf.alloc(f"{node.name}__quant_int8", M_pad * K_pad)
    # One bounded cache uses the otherwise unused 14/15 pair. The register
    # allocators exclude reserved registers, including overlapping quads.
    persistent = bool(
        cg.options.reuse_qkv_quantization
        and node.attrs.get("projection") in {"query", "key", "value"}
        and cg.quantization_last_uses.get(input_name, -1) > cg.current_node_idx
        and not cg.quantization_cache
        and M_pad * K_pad <= ABUF_SIZE // 8
    )
    _zero_fill_fp32_padding_rows(cg, inp, M, M_pad, K_pad)
    sreg = 14 if persistent else cg._alloc_sreg_pair()
    if persistent:
        cg.reserved_sregs.update({14, 15})
        cg.quantization_cache[input_name] = {
            "signature": signature, "allocation": scratch, "sreg": sreg,
            "last_use": cg.quantization_last_uses[input_name],
        }
    aq_m, aq_k = _act_quant_tiles(M_pad, K_pad)
    full_rows = aq_m * TILE
    cg._emit(ConfigTileInsn(M=aq_m - 1, N=aq_k - 1, K=0,
                           m_exact=_m_exact_rows(M * full_rows // M_pad, full_rows)))
    cg._emit(MaxAbsReduceFp32Insn(
        src1_buf=BUF_ABUF, src1_off=inp.offset_units,
        src2_buf=BUF_ABUF, src2_off=0, dst_buf=BUF_ABUF, dst_off=0,
        sreg=sreg, flags=cg.fp_precision_flag))
    cg._emit(SyncInsn(resource_mask=0b100))
    cg._emit(QuantFp32Int8Insn(
        src1_buf=BUF_ABUF, src1_off=inp.offset_units,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=scratch.offset_units,
        sreg=sreg, flags=cg.fp_precision_flag))
    cg._emit(SyncInsn(resource_mask=0b100))
    return inp, out, scratch, sreg, persistent


def prepare_streamed_activation(cg, node, M, M_pad, K_pad, k_start, k_len,
                                n_start, input_dram_off, cache):
    """Keep K partitions and their runtime scale bits unchanged across N tiles."""
    from .matmul import _act_quant_tiles

    key = (node.inputs[0], M, M_pad, K_pad, k_start, k_len, cg.fp_precision_flag)
    if cache is not None and key in cache:
        cg.optimization_events.append({"kind": "fc2_quantization_hit", "node": node.name,
                                       "k_start": k_start, "n_start": n_start})
        return cache[key]
    inp = cg.mem.abuf.alloc(f"{node.name}__in_fp32_k{k_start}_n{n_start}",
                            M_pad * k_len * cg.elem_bytes)
    row_bytes = k_len * cg.elem_bytes
    for row in range(M if cg.options.real_row_fc2_loads else M_pad):
        cg._emit_dma_load(BUF_ABUF, inp.offset_units + row * row_bytes // 16,
                          row_bytes, 0,
                          input_dram_off + row * K_pad * cg.elem_bytes + k_start * cg.elem_bytes)
    cg._emit(SyncInsn(resource_mask=0b001))
    _zero_fill_fp32_padding_rows(cg, inp, M, M_pad, k_len)
    sreg = cg._alloc_sreg_pair()
    if cache is not None:
        cg.reserved_sregs.update({sreg, sreg + 1})
    aq_m, aq_k = _act_quant_tiles(M_pad, k_len)
    full_rows = aq_m * TILE
    cg._emit(ConfigTileInsn(M=aq_m - 1, N=aq_k - 1, K=0,
                           m_exact=_m_exact_rows(M * full_rows // M_pad, full_rows)))
    cg._emit(MaxAbsReduceFp32Insn(
        src1_buf=BUF_ABUF, src1_off=inp.offset_units,
        src2_buf=BUF_ABUF, src2_off=0, dst_buf=BUF_ABUF, dst_off=0,
        sreg=sreg, flags=cg.fp_precision_flag))
    cg._emit(SyncInsn(resource_mask=0b100))
    scratch = cg.mem.abuf.alloc(f"{node.name}__quant_int8_k{k_start}_n{n_start}", M_pad * k_len)
    cg._emit(QuantFp32Int8Insn(
        src1_buf=BUF_ABUF, src1_off=inp.offset_units,
        src2_buf=BUF_ABUF, src2_off=0, dst_buf=BUF_ABUF, dst_off=scratch.offset_units,
        sreg=sreg, flags=cg.fp_precision_flag))
    cg._emit(SyncInsn(resource_mask=0b100))
    cg.mem.abuf.free(inp.name)
    if cache is not None:
        cache[key] = (scratch, sreg)
    return scratch, sreg
