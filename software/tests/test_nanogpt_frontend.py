from types import SimpleNamespace

import numpy as np

from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.frontend import load_frontend
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.isa.instructions import LoadInsn


def _tiny_nanogpt_config():
    return SimpleNamespace(
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=16,
        vocab_size=256,
        bias=True,
        layer_norm_epsilon=1e-5,
    )


def _ops(graph):
    return [node.op for node in graph.nodes]


def test_forward_1token_emits_decoder_config_and_attention_path():
    result = load_frontend("nanogpt", config=_tiny_nanogpt_config(), variant="forward_1token")

    assert result.config.model_kind == "decoder"
    assert result.config.embedding_kind == "token_pos"
    assert result.config.n_layer == 2
    assert result.config.n_head == 2
    assert result.config.d_model == 128
    assert result.config.d_head == 64
    assert result.config.mlp_dim == 512
    assert result.config.max_seq_len == 16

    ops = _ops(result.graph)
    assert "embed_lookup" in ops
    assert "pos_embed_lookup" in ops
    assert "matmul_qkt" in ops
    assert "softmax" in ops
    assert "matmul_attn_v" in ops
    assert all(
        node.attrs.get("masked") is True
        for node in result.graph.nodes
        if node.op == "matmul_qkt"
    )
    assert result.graph.get_node("tok_embed").attrs["token_ids"] == [0]
    assert result.graph.get_node("pos_embed").attrs["position_ids"] == [0]
    assert result.graph.nodes[-1].name == "lm_head"
    assert result.graph.nodes[-1].output_shape == (1, 256)
    assert all(node.output_shape[0] == 1 for node in result.graph.nodes if node.output_shape)
    assert all(
        node.attrs.get("causal_identity") is True
        for node in result.graph.nodes
        if node.op == "softmax"
    )


def test_non_attention_seq16_emits_mlp_only_decoder_subgraph():
    result = load_frontend("nanogpt", config=_tiny_nanogpt_config(), variant="non_attention_seq16")

    ops = _ops(result.graph)
    assert "embed_lookup" in ops
    assert "pos_embed_lookup" in ops
    assert "layernorm" in ops
    assert "gelu" in ops
    assert "vadd" in ops
    assert "matmul_qkt" not in ops
    assert "softmax" not in ops
    assert "matmul_attn_v" not in ops
    assert result.graph.get_node("tok_embed").attrs["token_ids"] == [0] * 16
    assert result.graph.get_node("pos_embed").attrs["position_ids"] == list(range(16))
    assert result.graph.nodes[-1].name == "lm_head"
    assert result.graph.nodes[-1].output_shape == (16, 256)
    assert all(node.output_shape[0] == 16 for node in result.graph.nodes if node.output_shape)


def test_nanogpt_adapter_rejects_unknown_variant():
    try:
        load_frontend("nanogpt", config=_tiny_nanogpt_config(), variant="bidirectional_seq16")
    except ValueError as exc:
        assert "variant must be" in str(exc)
    else:
        raise AssertionError("expected unknown nanoGPT variant to raise ValueError")


def test_embedding_lookup_codegen_smoke():
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        output_shape=(2, 128),
        attrs={"table": "transformer.wte.weight", "token_ids": [2, 1]},
    ))
    config = ModelConfig(
        model_kind="decoder",
        n_layer=1,
        n_head=2,
        d_model=128,
        d_head=64,
        mlp_dim=512,
        vocab_size=4,
        max_seq_len=16,
        embedding_kind="token_pos",
    )
    codegen = CodeGenerator(
        weight_data={
            "transformer.wte.weight": (
                np.arange(4 * 128, dtype=np.int16).astype(np.int8).reshape(4, 128),
                None,
            )
        },
        calibration_scales={"tok_embed": 1.0 / 127.0},
        prescaled_biases={},
        model_config=config,
    )

    instructions, _ = codegen.generate(graph)

    assert any(isinstance(insn, LoadInsn) for insn in instructions)
    assert codegen.mem.abuf.get("tok_embed") is not None
