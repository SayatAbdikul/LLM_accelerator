"""Stage 4 activation-spill tests for d=384 nanoGPT MLP shapes."""
import numpy as np

from taccel.assembler.assembler import ProgramBinary
from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.model_config import ModelConfig
from taccel.compiler.tiler import pad_dim
from taccel.golden_model.simulator import Simulator
from taccel.isa.encoding import encode
from taccel.runtime.fake_quant import cosine_similarity
from taccel.runtime.fake_quant_reference import _gelu_np


def _program(instructions, data):
    program = ProgramBinary(
        instructions=b"".join(encode(insn) for insn in instructions),
        data=data,
        entry_point=0,
        insn_count=len(instructions),
    )
    return ProgramBinary.from_bytes(program.to_bytes())


def _config():
    return ModelConfig(
        name="stage4-test-d384",
        model_kind="decoder",
        n_layer=1,
        n_head=6,
        d_model=384,
        d_head=64,
        mlp_dim=1536,
        vocab_size=32,
        max_seq_len=256,
        embedding_kind="token_pos",
    )


def test_d384_fc1_gelu_spill_is_consumed_by_fc2():
    seq = 256
    d_model = 384
    mlp_dim = 1536
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul",
        name="block0_fc1",
        inputs=["act", "fc1.weight"],
        output_shape=(seq, mlp_dim),
    ))
    graph.add_node(IRNode(
        op="gelu",
        name="block0_gelu",
        inputs=["block0_fc1"],
        output_shape=(seq, mlp_dim),
    ))
    graph.add_node(IRNode(
        op="matmul",
        name="block0_fc2",
        inputs=["block0_gelu", "fc2.weight"],
        output_shape=(seq, d_model),
    ))

    rng = np.random.default_rng(1234)
    act = rng.integers(-3, 4, size=(seq, d_model), dtype=np.int8)
    fc1_w = rng.integers(-4, 5, size=(d_model, mlp_dim), dtype=np.int8)
    fc2_w = rng.integers(-5, 6, size=(mlp_dim, d_model), dtype=np.int8)
    codegen = CodeGenerator(
        weight_data={
            "fc1.weight": (fc1_w, np.full(mlp_dim, 0.0625, dtype=np.float16)),
            "fc2.weight": (fc2_w, np.full(d_model, 0.0625, dtype=np.float16)),
        },
        calibration_scales={
            "act": 0.125,
            "block0_fc1": 0.25,
            "block0_gelu": 0.25,
            "block0_fc2": 0.25,
        },
        prescaled_biases={},
        model_config=_config(),
    )
    instructions, data = codegen.generate(graph)

    assert {"block0_fc1", "block0_gelu", "block0_fc2"} <= set(codegen.dram_temp_outputs)

    sim = Simulator()
    sim.load_program(_program(instructions, data))
    act_alloc = codegen.mem.abuf.get("act")
    sim.state.abuf[
        act_alloc.offset_units * 16:
        act_alloc.offset_units * 16 + act.size
    ] = act.tobytes()
    sim.run()

    out_off = codegen.dram_temp_outputs["block0_fc2"]
    out_size = pad_dim(seq) * pad_dim(d_model)
    got = np.frombuffer(bytes(sim.state.dram[out_off:out_off + out_size]), dtype=np.int8).reshape(seq, d_model)

    s_fc1 = np.float32(np.float16(0.125 * 0.0625 / 0.25))
    s_fc2 = np.float32(np.float16(0.25 * 0.0625 / 0.25))
    fc1 = np.clip(np.round((act.astype(np.int32) @ fc1_w.astype(np.int32)).astype(np.float32) * s_fc1), -128, 127).astype(np.int8)
    gelu_f = _gelu_np(fc1.astype(np.float32) * np.float32(0.25))
    gelu = np.clip(np.round(gelu_f / np.float32(0.25)), -128, 127).astype(np.int8)
    fc2 = np.clip(np.round((gelu.astype(np.int32) @ fc2_w.astype(np.int32)).astype(np.float32) * s_fc2), -128, 127).astype(np.int8)

    assert cosine_similarity(got.ravel(), fc2.ravel()) >= 0.999
    assert np.percentile(np.abs(got.astype(np.int16) - fc2.astype(np.int16)), 99) <= 1
