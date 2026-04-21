from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.isa.instructions import (
    ConfigAttnInsn,
    ConfigTileInsn,
    MaskedSoftmaxInsn,
    MaskedSoftmaxAttnVInsn,
    SoftmaxInsn,
    SoftmaxAttnVInsn,
)


def _decoder_config(max_seq_len=32):
    return ModelConfig(
        model_kind="decoder",
        n_layer=1,
        n_head=2,
        d_model=128,
        d_head=64,
        mlp_dim=512,
        vocab_size=256,
        max_seq_len=max_seq_len,
        embedding_kind="token_pos",
    )


def _qkt_graph(seq_len, masked):
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul_qkt",
        name="block0_head0_qkt",
        inputs=["q", "k"],
        output_shape=(seq_len, seq_len),
        attrs={"head_idx": 0, "scale": 0.125, "masked": masked},
    ))
    return graph


def _generate(seq_len, masked=True, fused=False):
    codegen = CodeGenerator(
        weight_data={},
        calibration_scales={
            "q": 1.0 / 127.0,
            "k": 1.0 / 127.0,
            "block0_head0_qkt": 1.0 / 127.0,
            "block0_head0_softmax": 1.0 / 127.0,
            "block0_head0_value": 1.0 / 127.0,
            "block0_head0_attn_v": 1.0 / 127.0,
        },
        prescaled_biases={},
        fused_softmax_attnv_blocks={0} if fused else None,
        model_config=_decoder_config(max_seq_len=max(seq_len, 16)),
    )
    instructions, _ = codegen.generate(_qkt_graph(seq_len, masked))
    return instructions


def _generate_decode_qkt(key_len=32):
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul_qkt",
        name="block0_head0_qkt",
        inputs=["q", "k_cache"],
        output_shape=(1, key_len),
        attrs={
            "head_idx": 0,
            "scale": 0.125,
            "masked": True,
            "query_len": 1,
            "key_len": key_len,
            "runtime_config_attn": True,
        },
    ))
    codegen = CodeGenerator(
        weight_data={},
        calibration_scales={
            "q": 1.0 / 127.0,
            "k_cache": 1.0 / 127.0,
            "block0_head0_qkt": 1.0 / 127.0,
            "block0_head0_softmax": 1.0 / 127.0,
        },
        prescaled_biases={},
        model_config=_decoder_config(max_seq_len=max(key_len, 16)),
        stream_name="decode",
    )
    instructions, _ = codegen.generate(graph)
    return codegen, instructions


def test_masked_qkt_emits_config_attn_per_strip_before_masked_softmax():
    instructions = _generate(17, masked=True)
    config_attns = [insn for insn in instructions if isinstance(insn, ConfigAttnInsn)]

    assert len(config_attns) == 2
    assert [insn.query_row_base for insn in config_attns] == [0, 16]
    assert [insn.valid_kv_len for insn in config_attns] == [17, 17]
    assert [insn.mode for insn in config_attns] == [0b11, 0b11]
    assert any(isinstance(insn, MaskedSoftmaxInsn) for insn in instructions)

    for idx, insn in enumerate(instructions):
        if isinstance(insn, ConfigAttnInsn):
            assert isinstance(instructions[idx - 1], ConfigTileInsn)


def test_masked_qkt_uses_pure_causal_mode_when_no_key_overhang():
    instructions = _generate(16, masked=True)
    config_attns = [insn for insn in instructions if isinstance(insn, ConfigAttnInsn)]

    assert len(config_attns) == 1
    assert config_attns[0].mode == 0b10


def test_unmasked_qkt_keeps_legacy_softmax_path():
    instructions = _generate(17, masked=False)

    assert not any(isinstance(insn, ConfigAttnInsn) for insn in instructions)
    assert any(isinstance(insn, SoftmaxInsn) for insn in instructions)
    assert not any(isinstance(insn, MaskedSoftmaxInsn) for insn in instructions)


def test_fused_masked_qkt_uses_masked_softmax_attnv():
    instructions = _generate(17, masked=True, fused=True)

    assert any(isinstance(insn, ConfigAttnInsn) for insn in instructions)
    assert any(isinstance(insn, MaskedSoftmaxAttnVInsn) for insn in instructions)
    assert not any(isinstance(insn, SoftmaxAttnVInsn) for insn in instructions)


def test_decode_qkt_emits_rectangular_config_tile_and_runtime_config_attn_site():
    codegen, instructions = _generate_decode_qkt(key_len=32)

    qkt_tiles = [insn for insn in instructions if isinstance(insn, ConfigTileInsn)]
    config_attns = [insn for insn in instructions if isinstance(insn, ConfigAttnInsn)]

    assert qkt_tiles[0].M == 0
    assert qkt_tiles[0].N == 1
    assert qkt_tiles[0].K == 3
    assert len(config_attns) == 1
    assert config_attns[0].query_row_base == 0
    assert config_attns[0].valid_kv_len == 1
    assert config_attns[0].mode == 0b11
    assert len(codegen.runtime_config_attn_sites) == 1
    assert codegen.runtime_config_attn_sites[0].stream == "decode"
