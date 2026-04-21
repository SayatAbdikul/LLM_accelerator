"""Stage 4 tests for large-weight matmul striping."""
import numpy as np

from taccel.assembler.assembler import ProgramBinary
from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.tiler import pad_dim
from taccel.golden_model.simulator import Simulator
from taccel.isa.encoding import encode
from taccel.isa.instructions import LoadInsn, MatmulInsn
from taccel.isa.opcodes import BUF_WBUF, WBUF_SIZE


def _program_from_codegen(codegen, instructions, data):
    program = ProgramBinary(
        instructions=b"".join(encode(insn) for insn in instructions),
        data=data,
        entry_point=0,
        insn_count=len(instructions),
    )
    return ProgramBinary.from_bytes(program.to_bytes())


def _run_matmul(*, input_i8, weight_kn, weight_name="test.weight", force_tiled=False):
    M, K = input_i8.shape
    N = weight_kn.shape[1]
    graph = IRGraph()
    graph.add_node(IRNode(
        op="matmul",
        name="test_node",
        inputs=["act", weight_name],
        output_shape=(M, N),
        attrs={"stage4_weight_tiled": force_tiled},
    ))
    codegen = CodeGenerator(
        weight_data={
            weight_name: (
                np.asarray(weight_kn, dtype=np.int8),
                np.full(pad_dim(N), 0.0625, dtype=np.float16),
            ),
        },
        calibration_scales={"act": 0.125, "test_node": 0.25},
        prescaled_biases={},
    )
    instructions, data = codegen.generate(graph)
    sim = Simulator()
    sim.load_program(_program_from_codegen(codegen, instructions, data))
    act_alloc = codegen.mem.abuf.get("act")
    sim.state.abuf[
        act_alloc.offset_units * 16:
        act_alloc.offset_units * 16 + input_i8.size
    ] = np.asarray(input_i8, dtype=np.int8).tobytes()
    sim.run()

    M_pad, N_pad = pad_dim(M), pad_dim(N)
    if "test_node" in codegen.dram_temp_outputs:
        off = codegen.dram_temp_outputs["test_node"]
        raw = bytes(sim.state.dram[off:off + M_pad * N_pad])
        result = np.frombuffer(raw, dtype=np.int8).reshape(M_pad, N_pad)
    else:
        out_alloc = codegen.mem.abuf.get("test_node")
        raw = bytes(sim.state.abuf[
            out_alloc.offset_units * 16:
            out_alloc.offset_units * 16 + M_pad * N_pad
        ])
        result = np.frombuffer(raw, dtype=np.int8).reshape(M_pad, N_pad)
    return result[:M, :N], codegen, instructions


def test_forced_weight_tiled_matches_unstriped_for_small_legal_matmul():
    input_i8 = (((np.arange(16 * 32).reshape(16, 32) * 3) % 17) - 8).astype(np.int8)
    weight = (((np.arange(32 * 32).reshape(32, 32) * 5) % 19) - 9).astype(np.int8)

    unstriped, _, _ = _run_matmul(input_i8=input_i8, weight_kn=weight, force_tiled=False)
    striped, _, _ = _run_matmul(input_i8=input_i8, weight_kn=weight, force_tiled=True)

    np.testing.assert_array_equal(striped, unstriped)


def test_fc1_style_weight_uses_384_by_512_tiles_that_fit_wbuf():
    input_i8 = np.zeros((16, 384), dtype=np.int8)
    weight = np.zeros((384, 1536), dtype=np.int8)

    _result, codegen, instructions = _run_matmul(input_i8=input_i8, weight_kn=weight)

    assert "test.weight__stage4_tile_k0_384_n0_512" in codegen.dram_layout
    wbuf_loads = [insn for insn in instructions if isinstance(insn, LoadInsn) and insn.buf_id == BUF_WBUF]
    assert wbuf_loads
    assert max(insn.xfer_len * 16 for insn in wbuf_loads) <= WBUF_SIZE


def test_fc2_style_weight_accumulates_multiple_k_chunks():
    input_i8 = (((np.arange(16 * 1536).reshape(16, 1536) * 7) % 11) - 5).astype(np.int8)
    weight = (((np.arange(1536 * 384).reshape(1536, 384) * 3) % 13) - 6).astype(np.int8)

    result, _codegen, instructions = _run_matmul(input_i8=input_i8, weight_kn=weight)

    matmuls = [insn for insn in instructions if isinstance(insn, MatmulInsn)]
    assert any(insn.flags == 1 for insn in matmuls)

    accum = input_i8.astype(np.int32) @ weight.astype(np.int32)
    scale = np.float32(np.float16(0.125 * 0.0625 / 0.25))
    expected = np.clip(np.round(accum.astype(np.float32) * scale), -128, 127).astype(np.int8)
    np.testing.assert_array_equal(result, expected)
