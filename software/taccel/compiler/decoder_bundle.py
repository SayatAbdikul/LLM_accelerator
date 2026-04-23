"""Narrow Stage 3 decoder ProgramBundle construction helpers."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..assembler.assembler import ProgramBundle
from ..isa.encoding import encode
from .codegen import CodeGenerator
from .ir import IRGraph, IRNode
from .kv_cache import KVCacheLayout, build_kv_cache_layout, normalize_kv_kind
from .model_config import ModelConfig
from .tiler import pad_dim


@dataclass
class DecoderBundleBuild:
    bundle: ProgramBundle
    kv_layout: KVCacheLayout
    prefill_codegen: CodeGenerator
    decode_codegen: CodeGenerator


def _encode_instructions(instructions) -> bytes:
    out = bytearray()
    for insn in instructions:
        out.extend(encode(insn))
    return bytes(out)


def _copy_graph_with_logits_store(graph: IRGraph, *, stream_name: str) -> IRGraph:
    """Return a graph copy with a logits_store node appended after lm_head when present."""
    out = IRGraph()
    has_logits_store = False
    lm_head_shape = None
    for node in graph.nodes:
        copied = IRNode(
            op=node.op,
            name=node.name,
            inputs=list(node.inputs),
            output_shape=tuple(node.output_shape),
            attrs=dict(node.attrs),
            output_scale=node.output_scale,
            weight_name=node.weight_name,
        )
        out.add_node(copied)
        if copied.op == "logits_store":
            has_logits_store = True
        if copied.name == "lm_head":
            lm_head_shape = tuple(copied.output_shape)

    if lm_head_shape is not None and not has_logits_store:
        out.add_node(IRNode(
            op="logits_store",
            name=f"{stream_name}_logits_store",
            inputs=["lm_head"],
            output_shape=lm_head_shape,
            attrs={
                "source_shape": lm_head_shape,
                "symbol": f"{stream_name}_logits_offset",
            },
        ))
    return out


def _infer_logits_size(*graphs: IRGraph) -> int:
    for graph in graphs:
        for node in graph.nodes:
            if node.name == "lm_head" and len(node.output_shape) == 2:
                return pad_dim(int(node.output_shape[1]))
    return 0


def mark_runtime_embedding_lookups(graph: IRGraph) -> IRGraph:
    """Return a graph copy whose token/position embedding loads are runtime-patched."""
    out = IRGraph()
    for node in graph.nodes:
        attrs = dict(node.attrs)
        if node.op in ("embed_lookup", "pos_embed_lookup"):
            attrs["runtime_patch"] = True
        out.add_node(IRNode(
            op=node.op,
            name=node.name,
            inputs=list(node.inputs),
            output_shape=tuple(node.output_shape),
            attrs=attrs,
            output_scale=node.output_scale,
            weight_name=node.weight_name,
        ))
    return out


def inject_kv_cache_nodes(graph: IRGraph, config: ModelConfig, *,
                          decode: bool, seq_len: int) -> IRGraph:
    """Insert narrow Stage 3 KV cache nodes into a nanoGPT-style graph."""
    out = IRGraph()
    key_loads: Dict[Tuple[int, int], str] = {}
    value_loads: Dict[Tuple[int, int], str] = {}

    for node in graph.nodes:
        copied = IRNode(
            op=node.op,
            name=node.name,
            inputs=list(node.inputs),
            output_shape=tuple(node.output_shape),
            attrs=dict(node.attrs),
            output_scale=node.output_scale,
            weight_name=node.weight_name,
        )
        layer = copied.attrs.get("block_idx")
        head = copied.attrs.get("head_idx")

        if decode and copied.op == "matmul_qkt" and layer is not None and head is not None:
            replacement = key_loads.get((int(layer), int(head)))
            if replacement is not None and len(copied.inputs) >= 2:
                copied.inputs[1] = replacement
            copied.attrs["query_len"] = int(copied.output_shape[0])
            copied.attrs["key_len"] = int(seq_len)
            copied.attrs["runtime_config_attn"] = True
            copied.output_shape = (int(copied.output_shape[0]), int(seq_len))
        if decode and copied.op in ("scale_mul", "softmax") and len(copied.output_shape) == 2:
            copied.attrs["query_len"] = int(copied.output_shape[0])
            copied.attrs["key_len"] = int(seq_len)
            copied.output_shape = (int(copied.output_shape[0]), int(seq_len))
        if decode and copied.op == "matmul_attn_v" and layer is not None and head is not None:
            replacement = value_loads.get((int(layer), int(head)))
            if replacement is not None and len(copied.inputs) >= 2:
                copied.inputs[1] = replacement
            copied.attrs["query_len"] = int(copied.output_shape[0])
            copied.attrs["key_len"] = int(seq_len)

        out.add_node(copied)

        projection = copied.attrs.get("projection")
        if copied.op != "matmul" or projection not in ("key", "value"):
            continue
        if layer is None or head is None:
            continue

        kind = normalize_kv_kind(str(projection))
        out.add_node(IRNode(
            op="kv_store",
            name=f"{copied.name}_kv_store",
            inputs=[copied.name],
            output_shape=(),
            attrs={
                "layer": int(layer),
                "kind": kind,
                "head": int(head),
                "seq_len": int(seq_len),
                "tokens": int(1 if decode else seq_len),
                "decode": bool(decode),
            },
        ))

        if not decode:
            continue

        load_name = f"{copied.name}_kv_load"
        out.add_node(IRNode(
            op="kv_load",
            name=load_name,
            inputs=[],
            output_shape=(int(seq_len), config.d_head),
            attrs={
                "layer": int(layer),
                "kind": kind,
                "head": int(head),
                "tokens": int(seq_len),
                "decode": True,
            },
        ))
        if kind == "key":
            key_loads[(int(layer), int(head))] = load_name
        else:
            value_loads[(int(layer), int(head))] = load_name

    return out


def build_decoder_program_bundle(
    *,
    prefill_graph: IRGraph,
    decode_graph: IRGraph,
    weight_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    calibration_scales: Dict[str, float],
    prescaled_biases: Dict[str, np.ndarray],
    model_config: ModelConfig,
    max_seq_len: Optional[int] = None,
    temp_size: int = 0,
    logits_size: int = 0,
    requant_pc_weight_names: Optional[set[str]] = None,
    requant_pc_scale_tables: Optional[Dict[str, np.ndarray]] = None,
) -> DecoderBundleBuild:
    """Build a ProgramBundle from already-formed decoder IR graphs."""
    kv_layout = build_kv_cache_layout(model_config, max_seq_len=max_seq_len)
    prefill_graph = _copy_graph_with_logits_store(prefill_graph, stream_name="prefill")
    decode_graph = _copy_graph_with_logits_store(decode_graph, stream_name="decode")
    if logits_size == 0:
        logits_size = _infer_logits_size(prefill_graph, decode_graph)
    requant_pc_weight_names, requant_pc_scale_tables = _decoder_requant_pc_tables(
        weight_data,
        calibration_scales,
        model_config,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        explicit_names=requant_pc_weight_names,
        explicit_tables=requant_pc_scale_tables,
    )
    residual_dequant_blocks = set(range(int(model_config.n_layer)))
    prefill_codegen = CodeGenerator(
        weight_data,
        calibration_scales,
        prescaled_biases,
        model_config=model_config,
        stream_name="prefill",
        kv_layout=kv_layout,
        dequant_add_residual1_blocks=residual_dequant_blocks,
        dequant_add_residual2_blocks=residual_dequant_blocks,
        requant_pc_weight_names=requant_pc_weight_names,
        requant_pc_scale_tables=requant_pc_scale_tables,
    )
    decode_codegen = CodeGenerator(
        weight_data,
        calibration_scales,
        prescaled_biases,
        model_config=model_config,
        stream_name="decode",
        kv_layout=kv_layout,
        dequant_add_residual1_blocks=residual_dequant_blocks,
        dequant_add_residual2_blocks=residual_dequant_blocks,
        requant_pc_weight_names=requant_pc_weight_names,
        requant_pc_scale_tables=requant_pc_scale_tables,
    )
    prefill_instructions, prefill_data = prefill_codegen.generate(prefill_graph)
    decode_instructions, decode_data = decode_codegen.generate(decode_graph)
    if prefill_data != decode_data:
        raise ValueError("prefill and decode graphs produced different shared data layouts")
    if temp_size == 0:
        temp_size = max(
            int(prefill_codegen.mem.dram_temp_total),
            int(decode_codegen.mem.dram_temp_total),
        )

    symbol_offsets = dict(prefill_codegen.dram_layout)
    symbol_regions = {}
    for bank in kv_layout.banks:
        symbol_offsets[bank.base_symbol] = bank.base_offset
        symbol_regions[bank.base_symbol] = "kv_cache"

    bundle = ProgramBundle(
        prefill_instrs=_encode_instructions(prefill_instructions),
        decode_instrs=_encode_instructions(decode_instructions),
        shared_data=prefill_data,
        temp_size=temp_size,
        logits_size=logits_size,
        kv_cache_size=kv_layout.kv_cache_size,
        embedding_row_bytes=model_config.d_model,
        kv_step_bytes=model_config.d_head,
        symbol_offsets=symbol_offsets,
        symbol_regions=symbol_regions,
        relocation_sites=prefill_codegen.relocation_sites + decode_codegen.relocation_sites,
        runtime_patch_sites=prefill_codegen.runtime_patch_sites + decode_codegen.runtime_patch_sites,
        runtime_config_attn_sites=(
            prefill_codegen.runtime_config_attn_sites + decode_codegen.runtime_config_attn_sites
        ),
    )
    return DecoderBundleBuild(
        bundle=bundle,
        kv_layout=kv_layout,
        prefill_codegen=prefill_codegen,
        decode_codegen=decode_codegen,
    )


def _decoder_requant_pc_tables(
    weight_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    calibration_scales: Dict[str, float],
    model_config: ModelConfig,
    *,
    prefill_graph: IRGraph,
    decode_graph: IRGraph,
    explicit_names: Optional[set[str]],
    explicit_tables: Optional[Dict[str, np.ndarray]],
) -> Tuple[set[str], Dict[str, np.ndarray]]:
    """Enable per-channel requant for GPT-2-class tied lm_head weights."""
    names = set(explicit_names or set())
    tables = dict(explicit_tables or {})
    weight_name = "lm_head.weight"
    if weight_name not in weight_data:
        return names, tables

    q_weight, weight_scales = weight_data[weight_name]
    if (
        isinstance(q_weight, np.ndarray)
        and q_weight.ndim == 2
        and q_weight.size > 256 * 1024
        and int(model_config.vocab_size) > 4096
    ):
        if weight_scales is None:
            raise KeyError("large-vocab lm_head.weight requires per-channel scales for REQUANT_PC")
        names.add(weight_name)
        if weight_name not in tables:
            scales = np.asarray(weight_scales, dtype=np.float32)
            input_scale_name = _matmul_input_for_weight((prefill_graph, decode_graph), weight_name)
            tables[weight_name] = (
                np.float32(calibration_scales.get(input_scale_name, 6.0 / 127.0))
                * scales
                / max(float(calibration_scales.get("lm_head", 6.0 / 127.0)), 1e-12)
            ).astype(np.float16)
    return names, tables


def _matmul_input_for_weight(graphs, weight_name: str) -> str:
    for graph in graphs:
        for node in graph.nodes:
            if node.op == "matmul" and len(node.inputs) > 1 and node.inputs[1] == weight_name:
                return str(node.inputs[0])
    return "ln_f"
