"""Stage 5 large-vocab lm_head striping tests."""
import numpy as np

from taccel.compiler.decoder_bundle import build_decoder_program_bundle
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.compiler.tiler import pad_dim
from taccel.isa.encoding import decode
from taccel.isa.instructions import LoadInsn, MatmulInsn, RequantPcInsn
from taccel.isa.opcodes import BUF_WBUF, WBUF_SIZE
from taccel.runtime.host_runner import HostRunner


def _config(vocab_size: int, d_model: int = 16) -> ModelConfig:
    return ModelConfig(
        name="stage5-lm-head-test",
        model_kind="decoder",
        n_layer=1,
        n_head=1,
        d_model=d_model,
        d_head=d_model,
        mlp_dim=4 * d_model,
        vocab_size=vocab_size,
        max_seq_len=16,
        embedding_kind="token_pos",
    )


def _graph(vocab_size: int, d_model: int = 16) -> IRGraph:
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        inputs=[],
        output_shape=(1, d_model),
        attrs={"table": "transformer.wte.weight", "runtime_patch": True},
    ))
    graph.add_node(IRNode(
        op="matmul",
        name="lm_head",
        inputs=["tok_embed", "lm_head.weight"],
        output_shape=(1, vocab_size),
        attrs={"weight_name": "lm_head.weight"},
    ))
    return graph


def _weights(vocab_size: int, d_model: int = 16):
    vocab_pad = pad_dim(vocab_size)
    token = ((np.arange(4 * d_model).reshape(4, d_model) % 31) - 15).astype(np.int8)
    weight = (((np.arange(d_model * vocab_pad).reshape(d_model, vocab_pad) * 3) % 23) - 11).astype(np.int8)
    scales = np.linspace(0.03125, 0.09375, vocab_pad, dtype=np.float16)
    return {
        "transformer.wte.weight": (token, None),
        "lm_head.weight": (weight, scales),
    }, token, weight, scales


def _build(vocab_size: int, d_model: int = 16):
    weights, token, weight, scales = _weights(vocab_size, d_model)
    build = build_decoder_program_bundle(
        prefill_graph=_graph(vocab_size, d_model),
        decode_graph=_graph(vocab_size, d_model),
        weight_data=weights,
        calibration_scales={"tok_embed": 0.125, "lm_head": 0.25},
        prescaled_biases={},
        model_config=_config(vocab_size, d_model),
        logits_size=pad_dim(vocab_size),
    )
    return build, token, weight, scales


def test_large_vocab_lm_head_strips_cover_vocab_and_fit_sram():
    vocab_size = 20_000
    build, _token, _weight, _scales = _build(vocab_size)
    insns = [decode(build.bundle.prefill_instrs[i:i + 8]) for i in range(0, len(build.bundle.prefill_instrs), 8)]

    wbuf_loads = [insn for insn in insns if isinstance(insn, LoadInsn) and insn.buf_id == BUF_WBUF]
    assert wbuf_loads
    assert max(insn.xfer_len * 16 for insn in wbuf_loads) <= WBUF_SIZE
    assert any(isinstance(insn, MatmulInsn) and insn.flags == 0 for insn in insns)
    assert any(isinstance(insn, RequantPcInsn) for insn in insns)
    assert "lm_head.weight__requant_pc" in build.prefill_codegen.dram_layout
    assert "lm_head" in build.prefill_codegen.dram_temp_outputs
    assert build.bundle.logits_size == pad_dim(vocab_size)


def test_host_runner_reads_assembled_large_vocab_logits():
    vocab_size = 20_000
    build, token, weight, scales = _build(vocab_size)
    runner = HostRunner(build.bundle, logits_dtype=np.int8)

    logits = runner.run_prefill([2])
    accum = token[2].astype(np.int32) @ weight.astype(np.int32)
    requant = np.float32(np.float16(0.125 * scales.astype(np.float32) / 0.25))
    expected = np.clip(np.round(accum.astype(np.float32) * requant), -128, 127).astype(np.int8)

    assert logits.shape == (pad_dim(vocab_size),)
    np.testing.assert_array_equal(logits[:vocab_size], expected[:vocab_size])


def test_large_weight_dequant_add_scatters_column_tiles_back_to_full_row():
    d_model = 768
    d_pad = pad_dim(d_model)
    config = ModelConfig(
        name="stage5-residual-tile-test",
        model_kind="decoder",
        n_layer=1,
        n_head=12,
        d_model=d_model,
        d_head=64,
        mlp_dim=4 * d_model,
        vocab_size=d_model,
        max_seq_len=16,
        embedding_kind="token_pos",
    )
    graph = IRGraph()
    graph.add_node(IRNode(
        op="embed_lookup",
        name="tok_embed",
        inputs=[],
        output_shape=(1, d_model),
        attrs={"table": "transformer.wte.weight", "runtime_patch": True},
    ))
    graph.add_node(IRNode(
        op="pos_embed_lookup",
        name="pos_embed",
        inputs=[],
        output_shape=(1, d_model),
        attrs={"table": "transformer.wpe.weight", "position_ids": [0]},
    ))
    graph.add_node(IRNode(
        op="vadd",
        name="tok_pos_add",
        inputs=["tok_embed", "pos_embed"],
        output_shape=(1, d_model),
    ))
    graph.add_node(IRNode(
        op="embed_lookup",
        name="block0_concat",
        inputs=[],
        output_shape=(1, d_model),
        attrs={"table": "concat.table", "token_ids": [2]},
    ))
    graph.add_node(IRNode(
        op="matmul",
        name="block0_out_proj",
        inputs=["block0_concat", "transformer.h.0.attn.c_proj.weight"],
        output_shape=(1, d_model),
        attrs={
            "weight_name": "transformer.h.0.attn.c_proj.weight",
            "bias": "transformer.h.0.attn.c_proj.bias",
        },
    ))
    graph.add_node(IRNode(
        op="vadd",
        name="block0_residual1",
        inputs=["tok_pos_add", "block0_out_proj"],
        output_shape=(1, d_model),
    ))
    graph.add_node(IRNode(
        op="logits_store",
        name="prefill_logits_store",
        inputs=["block0_residual1"],
        output_shape=(1, d_model),
        attrs={"source_shape": (1, d_model), "symbol": "prefill_logits_offset"},
    ))

    token = ((np.arange(4 * d_model).reshape(4, d_model) % 31) - 15).astype(np.int8)
    pos = np.zeros((16, d_model), dtype=np.int8)
    concat = ((np.arange(4 * d_model).reshape(4, d_model) % 17) - 8).astype(np.int8)
    weight = (((np.arange(d_model * d_pad).reshape(d_model, d_pad) * 3) % 23) - 11).astype(np.int8)
    weight_scales = np.full(d_pad, 0.0625, dtype=np.float16)
    bias = np.zeros(d_pad, dtype=np.int32)
    build = build_decoder_program_bundle(
        prefill_graph=graph,
        decode_graph=graph,
        weight_data={
            "transformer.wte.weight": (token, None),
            "transformer.wpe.weight": (pos, None),
            "concat.table": (concat, None),
            "transformer.h.0.attn.c_proj.weight": (weight, weight_scales),
        },
        calibration_scales={
            "tok_embed": 0.125,
            "pos_embed": 0.125,
            "tok_pos_add": 0.125,
            "block0_concat": 0.125,
            "block0_out_proj": 0.25,
            "block0_residual1": 0.2,
        },
        prescaled_biases={"transformer.h.0.attn.c_proj.bias": bias},
        model_config=config,
        logits_size=d_pad,
    )
    logits = HostRunner(build.bundle, logits_dtype=np.int8).run_prefill([2])[:d_model]

    accum = concat[2].astype(np.int32) @ weight.astype(np.int32) + bias
    accum_rescale = np.float32(np.float16(0.125 * 0.0625 / 0.2))
    skip_rescale = np.float32(np.float16(0.125 / 0.2))
    expected = np.clip(
        np.round(accum.astype(np.float32) * accum_rescale + token[2].astype(np.float32) * skip_rescale),
        -128,
        127,
    ).astype(np.int8)
    np.testing.assert_array_equal(logits, expected[:d_model])
