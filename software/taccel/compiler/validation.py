"""Reject incomplete or target-incompatible graphs before allocating storage."""
import numpy as np


SUPPORTED_OPS = frozenset({
    "matmul", "matmul_qkt", "packed_qkt_matmul", "qkt_dequant",
    "matmul_attn_v", "layernorm", "softmax", "gelu", "scale_mul", "vadd",
    "embed_lookup", "pos_embed_lookup", "kv_store", "kv_load", "kv_quant",
    "logits_store", "cls_prepend", "pos_embed_add", "cls_extract",
    "reshape_heads", "concat_heads", "row_copy", "gather_rows",
})


def validate_instruction(target, insn):
    from ..isa.opcodes import Opcode, RETIRED_OPCODES
    if target != "rtl-gen2":
        return
    if insn.opcode in RETIRED_OPCODES or insn.opcode == Opcode.SOFTMAX_FP32:
        raise ValueError(f"{insn.opcode.name} is unsupported by rtl-gen2")
    if insn.opcode == Opcode.CONFIG_TILE and insn.weight_int4:
        raise ValueError("INT4 weights are unsupported by rtl-gen2")
    if insn.opcode == Opcode.SET_SCALE and insn.src_mode != 0:
        raise ValueError("rtl-gen2 SET_SCALE supports immediate mode only")
    if int(insn.opcode) >= 0x17 and insn.flags != 1:
        raise ValueError("rtl-gen2 SFU instructions require FP16 storage flags=1")


def validate_graph(cg, graph):
    names = [node.name for node in graph.nodes]
    if len(set(names)) != len(names):
        raise ValueError("IR node names must be unique")
    # Validate the supplied node list even if a caller mutated it directly and
    # left IRGraph.node_map stale. Normal get_node lookup is already O(1).
    nodes_by_name = {node.name: node for node in graph.nodes}
    all_names, seen, weight_biases = set(names), set(), {}
    for node in graph.nodes:
        if node.op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported IR operation {node.op!r} at {node.name!r}")
        if node.op != "kv_store" and (
            len(node.output_shape) != 2
            or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
                   or d <= 0 for d in node.output_shape)
        ):
            raise ValueError(f"{node.name}: expected a positive 2D output shape")
        # Parameter slots are not activation producers. In particular, merely
        # putting a misspelled activation in weight_data must not authorize an
        # uninitialized SRAM input.
        parameter_slots = ({1} if node.op in {"matmul", "cls_prepend", "pos_embed_add"}
                           else {1, 2} if node.op == "layernorm" else set())
        # Only these two store emitters give a complete physical SRAM source
        # binding precedence over inputs[0]. A partial binding is not enough,
        # and attributes on unrelated operations cannot bypass dependencies.
        physical_source = (node.op in {"kv_store", "logits_store"}
                           and "src_off_units" in node.attrs and "src_buf" in node.attrs)
        for index, name in enumerate(node.inputs):
            if physical_source and index == 0:
                continue
            if name in all_names and name not in seen:
                raise ValueError(f"{node.name}: input {name!r} is used before its producer")
        seen.add(node.name)
        if cg.target == "rtl-gen2" and node.op == "softmax" and not node.attrs.get("masked", False):
            raise ValueError(f"{node.name}: noncausal SOFTMAX_FP32 is unsupported by rtl-gen2")
        if node.op == "matmul":
            if len(node.inputs) != 2:
                raise ValueError(f"{node.name}: matmul requires activation and weight inputs")
            name = node.inputs[1]
            if name not in cg.weight_data:
                raise KeyError(f"{node.name}: missing matmul weight {name!r}")
            data, scales = cg.weight_data[name]
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.dtype != np.int8:
                raise ValueError(f"{node.name}: matmul weight {name!r} must be a 2D INT8 array")
            if any(d <= 0 or d % 16 for d in data.shape):
                raise ValueError(f"{node.name}: matmul weight must have positive 16-aligned dimensions")
            if data.shape[1] != ((node.output_shape[1] + 15) // 16) * 16:
                raise ValueError(f"{node.name}: weight columns must match padded matmul output columns")
            producer = nodes_by_name.get(node.inputs[0])
            if producer is not None and (
                len(producer.output_shape) != 2
                or producer.output_shape[0] != node.output_shape[0]
                or ((producer.output_shape[1] + 15) // 16) * 16 != data.shape[0]
            ):
                raise ValueError(f"{node.name}: activation and weight shapes are incompatible")
            if scales is None or np.asarray(scales).size < node.output_shape[1]:
                raise ValueError(f"{node.name}: missing per-channel weight scales")
            bias = node.attrs.get("bias")
            if bias is not None and bias not in cg.biases:
                raise KeyError(f"{node.name}: missing FP32 bias {bias!r}")
            if bias is not None and np.asarray(cg.biases[bias]).size < node.output_shape[1]:
                raise ValueError(f"{node.name}: bias {bias!r} must cover logical output columns")
            if name in weight_biases and weight_biases[name] != bias:
                raise ValueError(f"{name}: shared weight cannot use different folded biases")
            weight_biases[name] = bias
        if node.op in {"embed_lookup", "pos_embed_lookup"}:
            default = "transformer.wte.weight" if node.op == "embed_lookup" else "transformer.wpe.weight"
            name = node.attrs.get("table", default)
            if name not in cg.weight_data:
                raise KeyError(f"{node.name}: missing embedding table {name!r}")
            data = cg.weight_data[name][0]
            dtype = np.float16 if cg.config.embedding_kind == "token_pos" else np.int8
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.dtype != dtype:
                raise ValueError(f"{node.name}: embedding table must be a 2D {np.dtype(dtype).name} array")
            if data.shape[1] != cg.config.d_model:
                raise ValueError(f"{node.name}: embedding table width differs from d_model")
        if node.op == "layernorm":
            if len(node.inputs) != 3:
                raise ValueError(f"{node.name}: layernorm requires input, gamma and beta")
            for name in node.inputs[1:]:
                if name not in cg.weight_data:
                    raise KeyError(f"{node.name}: missing layernorm parameter {name!r}")
                data = np.asarray(cg.weight_data[name][0])
                if data.dtype != np.float16 or data.size < node.output_shape[1]:
                    raise ValueError(f"{node.name}: layernorm parameter {name!r} must contain FP16 columns")
        if cg.external_inputs is not None:
            for index, name in enumerate(node.inputs):
                if physical_source and index == 0:
                    continue
                if index not in parameter_slots and name not in all_names and name not in cg.external_inputs:
                    raise ValueError(f"{node.name}: undeclared external activation {name!r}; "
                                     "provide a producer or explicit external_inputs binding")
