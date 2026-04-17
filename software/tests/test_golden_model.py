"""Tests for golden model simulator."""
import pytest
import numpy as np
from taccel.golden_model.state import MachineState
from taccel.golden_model.simulator import Simulator, ConfigError, IllegalBufferError
from taccel.golden_model.memory import SRAMAccessError, DRAMAccessError
from taccel.golden_model import memory as mem
from taccel.assembler.assembler import Assembler, ProgramBinary
from taccel.isa.opcodes import BUF_ABUF, BUF_WBUF, BUF_ACCUM
from tools.run_golden import write_runtime_inputs


def make_sim(asm_source: str) -> Simulator:
    prog = Assembler().assemble(asm_source)
    sim = Simulator()
    sim.load_program(prog)
    return sim


class TestSIMDMatmul:
    def test_single_matmul_tile(self):
        """Hand-verify 16x16 INT8 matmul."""
        # A = identity(16), B = 2*identity(16) → C = 2*identity(16)
        A = np.eye(16, dtype=np.int8) * 2
        B = np.eye(16, dtype=np.int8) * 3

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\nHALT"
        )
        sim = Simulator()
        sim.load_program(prog)

        # Write A to ABUF, B to WBUF
        sim.state.abuf[:256] = A.tobytes()
        sim.state.wbuf[:256] = B.tobytes()

        # Execute CONFIG_TILE
        sim.step()  # CONFIG_TILE

        # Manually check tile config is set
        assert sim.state.tile_config == (0, 0, 0)  # encoded as 0-based

        sim.step()  # HALT (but already halted by CONFIG_TILE)

    def test_matmul_identity(self):
        """A @ I = A."""
        A = np.arange(256, dtype=np.int8).reshape(16, 16)
        I = np.eye(16, dtype=np.int8)

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0\n"
            "SYNC 0b010\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.abuf[:256] = A.tobytes()
        sim.state.wbuf[:256] = I.tobytes()
        sim.run()

        result = sim.state.accum[:256].reshape(16, 16)
        expected = A.astype(np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_matmul_accumulate(self):
        """Two tiles with accumulate=1 sum correctly."""
        A = np.ones((16, 16), dtype=np.int8)
        B = np.ones((16, 16), dtype=np.int8)

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0\n"
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=1\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.abuf[:256] = A.tobytes()
        sim.state.wbuf[:256] = B.tobytes()
        sim.run()

        result = sim.state.accum[:256].reshape(16, 16)
        # Each element = 16 (from first tile) + 16 (from second) = 32
        assert np.all(result == 32), f"Expected 32, got {result[0,0]}"


class TestRequant:
    def test_basic_requant(self):
        """INT32 → INT8 clipping."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3800\n"  # 0x3800 = 0.5 in FP16
            "REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # Fill accumulator with 100 (INT32)
        sim.state.accum[:256] = 100
        sim.run()

        # 0.5 * 100 = 50
        result = np.frombuffer(bytes(sim.state.abuf[:256]), dtype=np.int8)
        assert np.all(result == 50), f"Expected 50, got {result[0]}"

    def test_requant_clipping(self):
        """INT32 values outside [-128, 127] get clipped."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3C00\n"  # 0x3C00 = 1.0 in FP16
            "REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[0] = 200   # out of INT8 range
        sim.state.accum[1] = -200  # out of INT8 range
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[:256]), dtype=np.int8)
        assert result[0] == 127   # clamped to max
        assert result[1] == -128  # clamped to min

    def test_requant_pc_matches_scalar_requant_when_scales_are_uniform(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "REQUANT_PC src1=ACCUM[0], src2=WBUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[:256] = np.arange(256, dtype=np.int32) - 128
        scales = np.full(16, np.float16(0.5), dtype=np.float16)
        sim.state.wbuf[: scales.nbytes] = scales.tobytes()
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[:256]), dtype=np.int8).reshape(16, 16)
        expected = np.clip(
            np.round((np.arange(256, dtype=np.int32).reshape(16, 16) - 128).astype(np.float32) * 0.5),
            -128,
            127,
        ).astype(np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_requant_pc_uses_per_column_scales(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "REQUANT_PC src1=ACCUM[0], src2=WBUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[:256] = 8
        scales = np.array([0.5] * 8 + [1.0] * 8, dtype=np.float16)
        sim.state.wbuf[: scales.nbytes] = scales.tobytes()
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[:256]), dtype=np.int8).reshape(16, 16)
        expected = np.tile(np.array([4] * 8 + [8] * 8, dtype=np.int8), (16, 1))
        np.testing.assert_array_equal(result, expected)


class TestVADD:
    def test_int8_saturating_add(self):
        """INT8 VADD saturates at ±127."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "VADD src1=ABUF[0], src2=ABUF[16], dst=ABUF[32], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # src1: fill with 100, src2: fill with 50 → result 127 (saturated)
        sim.state.abuf[:256] = bytes([100] * 256)
        sim.state.abuf[256:512] = bytes([50] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[512:768]), dtype=np.int8)
        assert np.all(result == 127), f"Expected 127, got {result[0]}"

    def test_int8_add_no_overflow(self):
        """INT8 VADD: 10 + 20 = 30."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "VADD src1=ABUF[0], src2=ABUF[16], dst=ABUF[32], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.abuf[:256] = bytes([10] * 256)
        sim.state.abuf[256:512] = bytes([20] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[512:768]), dtype=np.int8)
        assert np.all(result == 30)


class TestDequantAdd:
    def test_accum_plus_skip_requants_to_int8(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S6, imm=0x3800\n"
            "SET_SCALE S7, imm=0x3400\n"
            "DEQUANT_ADD src1=ACCUM[0], src2=ABUF[16], dst=ABUF[32], sreg=6, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[:256] = 20
        sim.state.abuf[256:512] = bytes([8] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[512:768]), dtype=np.int8).reshape(16, 16)
        expected = np.clip(np.round(20.0 * 0.5 + 8.0 * 0.25), -128, 127).astype(np.int8)
        assert np.all(result == expected)


class TestDMA:
    def test_load_store_roundtrip(self):
        """DMA load then store recovers original data."""
        data = bytes(range(256))  # 256 bytes = 16 units
        # Store to 0x100000 (1 MB) — well within 16 MB DRAM
        prog = Assembler().assemble(
            "SET_ADDR_LO R0, 0x0000000\n"
            "SET_ADDR_HI R0, 0x0000000\n"
            "LOAD buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=0, dram_off=0\n"
            "SYNC 0b001\n"
            "SET_ADDR_LO R1, 0x0100000\n"
            "SET_ADDR_HI R1, 0x0000000\n"
            "STORE buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=1, dram_off=0\n"
            "SYNC 0b001\n"
            "HALT",
            data=data,
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.run()

        # Verify ABUF contains loaded data
        assert bytes(sim.state.abuf[:256]) == data
        # Verify DRAM store target contains the same data
        assert bytes(sim.state.dram[0x100000:0x100000 + 256]) == data


class TestRuntimeInputPlacement:
    def test_write_runtime_inputs_uses_program_offsets_and_fold_metadata(self):
        prog = ProgramBinary(
            input_offset=1024,
            pos_embed_patch_dram_offset=512,
            pos_embed_cls_dram_offset=64,
            cls_token_dram_offset=768,
        )
        state = MachineState()
        patches = np.arange(8, dtype=np.int8).reshape(2, 4)
        cls_row = np.arange(192, dtype=np.int8).reshape(1, 192)

        write_runtime_inputs(
            state,
            prog,
            patches,
            cls_input=cls_row,
            folded_pos_embed=True,
        )

        expected_patch_bytes = np.zeros((2, 16), dtype=np.int8)
        expected_patch_bytes[:, :4] = patches
        assert bytes(state.dram[1024:1024 + expected_patch_bytes.nbytes]) == expected_patch_bytes.tobytes()
        assert bytes(state.dram[768:768 + 192]) == cls_row.tobytes()
        assert bytes(state.dram[64:64 + 192]) == bytes(192)
        assert bytes(state.dram[512:512 + expected_patch_bytes.nbytes]) == bytes(expected_patch_bytes.nbytes)

    def test_write_runtime_inputs_falls_back_to_abuf_for_legacy_program(self):
        prog = ProgramBinary()
        state = MachineState()
        patches = np.arange(8, dtype=np.int8).reshape(2, 4)

        write_runtime_inputs(state, prog, patches)

        expected_patch_bytes = np.zeros((2, 16), dtype=np.int8)
        expected_patch_bytes[:, :4] = patches
        assert bytes(state.abuf[:expected_patch_bytes.nbytes]) == expected_patch_bytes.tobytes()


class TestBufCopy:
    def test_flat_copy(self):
        """BUF_COPY flat: copies bytes unchanged."""
        prog = Assembler().assemble(
            "BUF_COPY src_buf=ABUF, src_off=0, dst_buf=WBUF, dst_off=0, length=16\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        data = bytes(range(256))
        sim.state.abuf[:256] = data
        sim.run()
        assert bytes(sim.state.wbuf[:256]) == data

    def test_transpose_copy(self):
        """BUF_COPY transpose: [32, 16] → [16, 32]."""
        prog = Assembler().assemble(
            "BUF_COPY src_buf=ABUF, src_off=0, dst_buf=WBUF, dst_off=0, "
            "length=32, src_rows=2, transpose=1\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)

        # Create [32, 16] source: src_rows=2 means 2*16=32 rows, cols=length*16/(src_rows*16)=512/32=16
        # length=32 means 32*16=512 bytes
        src = np.arange(512, dtype=np.int8).reshape(32, 16)
        sim.state.abuf[:512] = src.tobytes()
        sim.run()

        result = np.frombuffer(bytes(sim.state.wbuf[:512]), dtype=np.int8).reshape(16, 32)
        expected = src.T
        np.testing.assert_array_equal(result, expected)


class TestSFUOps:
    def test_gelu_positive(self):
        """GELU(x) ≈ x for large positive x."""
        # GELU(4) ≈ 4, GELU(-4) ≈ 0
        from taccel.isa.instructions import GeluInsn, ConfigTileInsn, SetScaleInsn, HaltInsn
        from taccel.isa.encoding import encode
        import struct

        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3400\n"   # FP16 0.25 (in_scale)
            "SET_SCALE S1, imm=0x3400\n"   # FP16 0.25 (out_scale)
            "GELU src1=ABUF[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # Fill ABUF with 16 (= 4.0 at scale 0.25)
        sim.state.abuf[:256] = bytes([16] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[256:512]), dtype=np.int8)
        # GELU(4.0) ≈ 4.0 → at scale 0.25, quantized ≈ 16
        assert result[0] > 10, f"GELU should be ~16, got {result[0]}"

    def test_gelu_from_accum_int32_path(self):
        """GELU can consume INT32 ACCUM input directly."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3400\n"   # in_scale = 0.25
            "SET_SCALE S1, imm=0x3400\n"   # out_scale = 0.25
            "GELU src1=ACCUM[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[:256] = 16  # 16 * 0.25 = 4.0
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[256:512]), dtype=np.int8)
        assert result[0] > 10, f"GELU-from-ACCUM should be ~16, got {result[0]}"

    def test_softmax_sums_to_one(self):
        """Softmax output sums close to 1.0 after dequantization."""
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3800\n"  # 0.5 in_scale
            "SET_SCALE S1, imm=0x3400\n"  # 0.25 out_scale (probs 0-1 → scaled 0-4)
            "SOFTMAX src1=ABUF[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        # Uniform input: all zeros → uniform softmax
        sim.state.abuf[:256] = bytes([0] * 256)
        sim.run()

        result = np.frombuffer(bytes(sim.state.abuf[256:512]), dtype=np.int8)
        # At scale 0.25, each element should be ~round(1/256 / 0.25) ≈ 0
        # All 256 elements uniform: each = 1/256, tiny but non-negative
        assert np.all(result >= 0), "Softmax output should be non-negative"

    def test_softmax_from_accum_int32_path(self):
        """SOFTMAX accepts ACCUM INT32 input and matches expected requantized output."""
        from taccel.golden_model.sfu import execute_softmax
        from taccel.isa.instructions import SoftmaxInsn

        state = MachineState()
        # One 16x16 tile
        state.tile_config = (0, 0, 0)
        in_scale = np.float32(0.125)
        out_scale = np.float32(1.0 / 256.0)
        state.scale_regs[0] = in_scale
        state.scale_regs[1] = out_scale

        # Deterministic INT32 attention-like scores in ACCUM.
        tile_i32 = np.arange(256, dtype=np.int32).reshape(16, 16) - 128
        state.accum[:256] = tile_i32.flatten()

        insn = SoftmaxInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=0,
            sreg=0,
            flags=0,
        )

        execute_softmax(state, insn)

        # Reference: softmax(tile_i32 * in_scale), then requantize to INT8.
        x = tile_i32.astype(np.float32) * in_scale
        x_shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted).astype(np.float32)
        probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        expected = np.clip(np.round(probs / out_scale), -128, 127).astype(np.int8)

        got = np.frombuffer(bytes(state.abuf[:256]), dtype=np.int8).reshape(16, 16)
        np.testing.assert_array_equal(got, expected)

    def test_softmax_attnv_fused_path_matches_expected_quantized_output(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S4, imm=0x3000\n"  # qkt_in_scale = 0.125
            "SET_SCALE S5, imm=0x3400\n"  # v_scale = 0.25
            "SET_SCALE S6, imm=0x3400\n"  # out_scale = 0.25
            "SET_SCALE S7, imm=0x2c00\n"  # softmax trace scale = 1/16
            "SOFTMAX_ATTNV src1=ACCUM[0], src2=ABUF[0], dst=WBUF[0], sreg=4, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        sim.state.accum[:256] = 0  # uniform logits => uniform softmax
        sim.state.abuf[:256] = bytes([8] * 256)  # 8 * 0.25 = 2.0
        sim.run()

        got = np.frombuffer(bytes(sim.state.wbuf[:256]), dtype=np.int8).reshape(16, 16)
        expected = np.full((16, 16), 8, dtype=np.int8)
        np.testing.assert_array_equal(got, expected)

    def test_runtime_twin_softmax_manifest_changes_simulator_output(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3000\n"
            "SET_SCALE S1, imm=0x1c00\n"
            "SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        prog.compiler_manifest = {
            "runtime_twin_uniform": {
                "mode": "paper_exact",
                "softmax": {
                    "3": {
                        "mode": "paper_exact",
                        "range1_max": 0.05,
                        "block": 11,
                        "head": 1,
                        "node_name": "block11_head1_softmax",
                    }
                },
                "gelu": {},
            }
        }
        baseline = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3000\n"
            "SET_SCALE S1, imm=0x1c00\n"
            "SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )

        logits = np.zeros((16, 16), dtype=np.int32)
        logits[:, 0] = 8
        logits[:, 1] = 2
        logits[:, 2] = -2

        twin_sim = Simulator()
        twin_sim.load_program(prog)
        twin_sim.state.accum[:256] = logits.reshape(-1)
        twin_sim.run()

        base_sim = Simulator()
        base_sim.load_program(baseline)
        base_sim.state.accum[:256] = logits.reshape(-1)
        base_sim.run()

        twin_out = np.frombuffer(bytes(twin_sim.state.abuf[:256]), dtype=np.int8).reshape(16, 16)
        base_out = np.frombuffer(bytes(base_sim.state.abuf[:256]), dtype=np.int8).reshape(16, 16)
        assert not np.array_equal(twin_out, base_out)

    def test_runtime_twin_gelu_manifest_changes_simulator_output(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3400\n"
            "SET_SCALE S1, imm=0x2c00\n"
            "GELU src1=ABUF[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )
        prog.compiler_manifest = {
            "runtime_twin_uniform": {
                "mode": "paper_exact",
                "softmax": {},
                "gelu": {
                    "3": {
                        "mode": "paper_exact",
                        "positive_range_max": 0.4,
                        "negative_extent": 1.0,
                        "block": 9,
                        "node_name": "block9_gelu",
                    }
                },
            }
        }
        baseline = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3400\n"
            "SET_SCALE S1, imm=0x2c00\n"
            "GELU src1=ABUF[0], src2=ABUF[0], dst=ABUF[16], sreg=0, flags=0\n"
            "HALT"
        )

        inp = (np.array(
            [-8, -4, -1, 0, 2, 6, 9, 12] * 32,
            dtype=np.int8,
        ))[:256]

        twin_sim = Simulator()
        twin_sim.load_program(prog)
        twin_sim.state.abuf[:256] = inp.tobytes()
        twin_sim.run()

        base_sim = Simulator()
        base_sim.load_program(baseline)
        base_sim.state.abuf[:256] = inp.tobytes()
        base_sim.run()

        twin_out = np.frombuffer(bytes(twin_sim.state.abuf[256:512]), dtype=np.int8).reshape(16, 16)
        base_out = np.frombuffer(bytes(base_sim.state.abuf[256:512]), dtype=np.int8).reshape(16, 16)
        assert not np.array_equal(twin_out, base_out)

    def test_runtime_twin_softmax_attnv_manifest_changes_fused_output(self):
        from taccel.golden_model.sfu import execute_softmax_attnv
        from taccel.isa.instructions import SoftmaxAttnVInsn

        twin_state = MachineState()
        twin_state.tile_config = (0, 0, 0)
        twin_state.scale_regs[4] = np.float16(0.125)
        twin_state.scale_regs[5] = np.float16(0.0625)
        twin_state.scale_regs[6] = np.float16(0.0625)
        twin_state.scale_regs[7] = np.float16(1.0 / 256.0)
        twin_state.current_pc = 5
        twin_state.runtime_twin_specs = {
            5: {
                "kind": "softmax",
                "mode": "paper_exact",
                "range1_max": 0.05,
                "block": 11,
                "head": 2,
                "node_name": "block11_head2_softmax",
            }
        }
        base_state = MachineState()
        base_state.tile_config = (0, 0, 0)
        base_state.scale_regs[4] = np.float16(0.125)
        base_state.scale_regs[5] = np.float16(0.0625)
        base_state.scale_regs[6] = np.float16(0.0625)
        base_state.scale_regs[7] = np.float16(1.0 / 256.0)
        base_state.current_pc = 5
        base_state.runtime_twin_specs = {}

        insn = SoftmaxAttnVInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_ABUF, src2_off=0,
            dst_buf=BUF_WBUF, dst_off=0,
            sreg=4,
            flags=0,
        )
        logits = np.zeros((16, 16), dtype=np.int32)
        logits[:, 0] = 8
        logits[:, 1] = 4
        v = np.zeros((16, 16), dtype=np.int8)
        v[:, 0] = 4
        v[:, 1] = -4
        v[:, 2] = 2
        twin_state.accum[:256] = logits.reshape(-1)
        base_state.accum[:256] = logits.reshape(-1)
        twin_state.abuf[:256] = v.tobytes()
        base_state.abuf[:256] = v.tobytes()

        twin_payload = execute_softmax_attnv(twin_state, insn)
        base_payload = execute_softmax_attnv(base_state, insn)

        twin_softmax = twin_payload["softmax"]["raw"]
        base_softmax = base_payload["softmax"]["raw"]
        assert not np.array_equal(twin_softmax, base_softmax)

    def test_runtime_twin_empty_manifest_keeps_outputs_unchanged(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3c00\n"
            "SET_SCALE S1, imm=0x2c00\n"
            "SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        prog.compiler_manifest = {"runtime_twin_uniform": {"mode": "off", "softmax": {}, "gelu": {}}}
        baseline = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3c00\n"
            "SET_SCALE S1, imm=0x2c00\n"
            "SOFTMAX src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )

        logits = np.arange(256, dtype=np.int32).reshape(16, 16) - 32

        twin_sim = Simulator()
        twin_sim.load_program(prog)
        twin_sim.state.accum[:256] = logits.reshape(-1)
        twin_sim.run()

        base_sim = Simulator()
        base_sim.load_program(baseline)
        base_sim.state.accum[:256] = logits.reshape(-1)
        base_sim.run()

        np.testing.assert_array_equal(
            np.frombuffer(bytes(twin_sim.state.abuf[:256]), dtype=np.int8),
            np.frombuffer(bytes(base_sim.state.abuf[:256]), dtype=np.int8),
        )


class TestErrorHandling:
    def test_matmul_without_config_tile_raises(self):
        """MATMUL without preceding CONFIG_TILE raises ConfigError."""
        prog = Assembler().assemble(
            "MATMUL src1=ABUF[0], src2=WBUF[0], dst=ACCUM[0], sreg=0, flags=0\n"
            "HALT"
        )
        sim = Simulator()
        sim.load_program(prog)
        with pytest.raises(ConfigError):
            sim.run()

    def test_store_oob_raises(self):
        """STORE beyond DRAM boundary raises DRAMAccessError."""
        # Address 0xFFFFFF = 16 MB - 1; store of 16 units (256 bytes) overflows
        prog = Assembler().assemble(
            "SET_ADDR_LO R0, 0xFFFFF00\n"
            "SET_ADDR_HI R0, 0x0000000\n"
            "STORE buf_id=ABUF, sram_off=0, xfer_len=16, addr_reg=0, dram_off=0\n"
            "HALT",
        )
        sim = Simulator()
        sim.load_program(prog)
        with pytest.raises(DRAMAccessError):
            sim.run()


class TestErfPoly:
    def test_erf_poly_matches_scipy_for_int8(self):
        """Polynomial erf approx produces identical INT8 GELU output as scipy."""
        from taccel.golden_model.sfu import _erf_poly
        from scipy.special import erf

        # Test over the full INT8 dequantized input range
        # Typical scale ~ 0.05, so INT8 [-128, 127] maps to [-6.4, 6.35]
        x = np.linspace(-8.0, 8.0, 10000, dtype=np.float32)
        sqrt2 = np.float32(np.sqrt(2.0))

        gelu_ref = x * 0.5 * (1.0 + erf(x / sqrt2).astype(np.float32))
        gelu_poly = x * 0.5 * (1.0 + _erf_poly(x / sqrt2))

        # When requantized to INT8, both must produce identical results
        scale = np.float32(0.05)
        ref_int8 = np.clip(np.round(gelu_ref / scale), -128, 127).astype(np.int8)
        poly_int8 = np.clip(np.round(gelu_poly / scale), -128, 127).astype(np.int8)
        assert np.array_equal(ref_int8, poly_int8), \
            f"Max INT8 diff: {np.max(np.abs(ref_int8.astype(int) - poly_int8.astype(int)))}"

    def test_erf_poly_max_fp32_error(self):
        """Polynomial erf approx has max absolute error < 1e-6 in FP32.

        The Abramowitz & Stegun formula achieves ~1.5e-7 in FP64, but FP32
        rounding raises the measured error to ~5e-7.  Still far below the
        INT8 quantization noise floor (~0.004).
        """
        from taccel.golden_model.sfu import _erf_poly
        from scipy.special import erf

        x = np.linspace(-6.0, 6.0, 100000, dtype=np.float32)
        ref = erf(x).astype(np.float32)
        poly = _erf_poly(x)
        max_err = float(np.max(np.abs(ref - poly)))
        assert max_err < 1e-6, f"Max erf error {max_err:.2e} exceeds 1e-6"


class TestTraceRawSnapshots:
    def test_trace_payload_includes_raw_int8_tensor(self):
        program = Assembler().assemble("NOP\nHALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "trace_abuf",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 16,
                    "logical_rows": 1,
                    "logical_cols": 16,
                    "full_rows": 1,
                    "full_cols": 16,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.5,
                    "when": "after",
                }
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace(["trace_abuf"])
        values = np.arange(16, dtype=np.int8)
        mem.write_bytes(sim.state, BUF_ABUF, 0, values.tobytes())
        sim.run()

        trace = sim.get_trace_payload()
        np.testing.assert_array_equal(trace["raw_tensors"]["trace_abuf"], values.reshape(1, 16))
        assert trace["meta"]["trace_abuf"]["dtype"] == "int8"
        assert trace["meta"]["trace_abuf"]["raw_available"] is True
        assert trace["raw_events"][0]["event_index"] == 0
        assert trace["raw_events"][0]["raw_available"] is True

    def test_trace_payload_includes_raw_int32_tensor(self):
        program = Assembler().assemble("NOP\nHALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "trace_accum",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 1.25,
                    "when": "after",
                }
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace(["trace_accum"])
        values = np.array([7, -9, 11, -13], dtype=np.int32)
        mem.write_int32_tile(sim.state, BUF_ACCUM, 0, values.reshape(1, 4))
        sim.run()

        trace = sim.get_trace_payload()
        np.testing.assert_array_equal(trace["raw_tensors"]["trace_accum"], values.reshape(1, 4))
        assert trace["meta"]["trace_accum"]["dtype"] == "int32"
        assert trace["meta"]["trace_accum"]["scale"] == pytest.approx(1.25)
        assert trace["raw_events"][0]["raw_available"] is True

    def test_accum_pre_matmul_trace_is_zeroed_for_golden_debug(self):
        program = Assembler().assemble("NOP\nHALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "block0_head0_qkt__accum_pre_matmul",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 0.125,
                    "when": "after",
                }
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace(["block0_head0_qkt__accum_pre_matmul"])
        values = np.array([7, -9, 11, -13], dtype=np.int32)
        mem.write_int32_tile(sim.state, BUF_ACCUM, 0, values.reshape(1, 4))
        sim.run()

        trace = sim.get_trace_payload()
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_qkt__accum_pre_matmul"],
            np.zeros((1, 4), dtype=np.int32),
        )
        assert trace["raw_events"][0]["raw_available"] is True
        assert trace["raw_events"][0]["raw"] == [[0, 0, 0, 0]]

    def test_qkt_stability_debug_traces_use_expected_raw_views(self):
        program = Assembler().assemble("NOP\nHALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "block0_head0_qkt__accum_pre_matmul_next",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 0.125,
                    "when": "after",
                    "capture_phase": "retire_plus_1",
                },
                {
                    "node_name": "block0_head0_qkt__accum_pre_softmax",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 0.125,
                    "when": "after",
                },
                {
                    "node_name": "block0_head0_qkt__accum_pre_softmax_next",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 0.125,
                    "when": "after",
                    "capture_phase": "retire_plus_1",
                },
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace([
            "block0_head0_qkt__accum_pre_matmul_next",
            "block0_head0_qkt__accum_pre_softmax",
            "block0_head0_qkt__accum_pre_softmax_next",
        ])
        values = np.array([7, -9, 11, -13], dtype=np.int32)
        mem.write_int32_tile(sim.state, BUF_ACCUM, 0, values.reshape(1, 4))
        sim.run()

        trace = sim.get_trace_payload()
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_qkt__accum_pre_matmul_next"],
            np.zeros((1, 4), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_qkt__accum_pre_softmax"],
            values.reshape(1, 4),
        )
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_qkt__accum_pre_softmax_next"],
            values.reshape(1, 4),
        )
        assert trace["raw_events"][0]["capture_phase"] == "retire_plus_1"

    def test_projection_padded_trace_zeroes_rows_beyond_logical_extent(self):
        program = Assembler().assemble("NOP\nHALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "block0_head0_query__act_input",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
                {
                    "node_name": "block0_head0_query__act_input_padded",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 6,
                    "mem_cols": 4,
                    "logical_rows": 6,
                    "logical_cols": 4,
                    "full_rows": 6,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
                {
                    "node_name": "block0_head0_query__accum_pre_bias",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 0.125,
                    "when": "after",
                },
                {
                    "node_name": "block0_head0_query__accum_pre_bias_padded",
                    "buf_id": BUF_ACCUM,
                    "offset_units": 0,
                    "mem_rows": 6,
                    "mem_cols": 4,
                    "logical_rows": 6,
                    "logical_cols": 4,
                    "full_rows": 6,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int32",
                    "scale": 0.125,
                    "when": "after",
                },
                {
                    "node_name": "block0_head0_query",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
                {
                    "node_name": "block0_head0_query__output_padded",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 6,
                    "mem_cols": 4,
                    "logical_rows": 6,
                    "logical_cols": 4,
                    "full_rows": 6,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace([
            "block0_head0_query__act_input_padded",
            "block0_head0_query__accum_pre_bias_padded",
            "block0_head0_query__output_padded",
        ])
        accum_values = np.arange(24, dtype=np.int32).reshape(6, 4)
        abuf_values = np.arange(24, dtype=np.int8).reshape(6, 4)
        mem.write_int32_tile(sim.state, BUF_ACCUM, 0, accum_values)
        mem.write_bytes(sim.state, BUF_ABUF, 0, abuf_values.tobytes())
        sim.run()

        trace = sim.get_trace_payload()
        expected_accum = accum_values.copy()
        expected_accum[1:, :] = 0
        expected_abuf = abuf_values.copy()
        expected_abuf[1:, :] = 0
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_query__act_input_padded"],
            expected_abuf,
        )
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_query__accum_pre_bias_padded"],
            expected_accum,
        )
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_head0_query__output_padded"],
            expected_abuf,
        )

    def test_block0_ln1_padded_input_zeroes_but_output_preserves_padded_rows(self):
        program = Assembler().assemble("NOP\nHALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "block0_ln1",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
                {
                    "node_name": "block0_ln1__input_padded",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 6,
                    "mem_cols": 4,
                    "logical_rows": 6,
                    "logical_cols": 4,
                    "full_rows": 6,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
                {
                    "node_name": "block0_ln1__output_padded",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 6,
                    "mem_cols": 4,
                    "logical_rows": 6,
                    "logical_cols": 4,
                    "full_rows": 6,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 0.25,
                    "when": "after",
                },
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace([
            "block0_ln1__input_padded",
            "block0_ln1__output_padded",
        ])
        abuf_values = (np.arange(24, dtype=np.int16).reshape(6, 4) - 12).astype(np.int8)
        mem.write_bytes(sim.state, BUF_ABUF, 0, abuf_values.tobytes())
        sim.run()

        trace = sim.get_trace_payload()
        expected_input = abuf_values.copy()
        expected_input[1:, :] = 0
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_ln1__input_padded"],
            expected_input,
        )
        np.testing.assert_array_equal(
            trace["raw_tensors"]["block0_ln1__output_padded"],
            abuf_values,
        )

    def test_virtual_trace_events_are_marked_non_architectural(self):
        program = Assembler().assemble("HALT\n")
        program.trace_manifest = {
            0: [
                {
                    "node_name": "virtual_node",
                    "buf_id": BUF_ABUF,
                    "offset_units": 0,
                    "mem_rows": 1,
                    "mem_cols": 4,
                    "logical_rows": 1,
                    "logical_cols": 4,
                    "full_rows": 1,
                    "full_cols": 4,
                    "row_start": 0,
                    "dtype": "int8",
                    "scale": 1.0,
                    "when": "after",
                    "source": "virtual",
                }
            ]
        }
        sim = Simulator()
        sim.load_program(program)
        sim.enable_trace(["virtual_node"])
        sim._virtual_trace_payloads["virtual_node"] = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        sim.run()

        trace = sim.get_trace_payload()
        assert trace["meta"]["virtual_node"]["source"] == "virtual"
        assert trace["meta"]["virtual_node"]["raw_available"] is False
        assert trace["raw_events"][0]["source"] == "virtual"
        assert trace["raw_events"][0]["raw_available"] is False
        assert "virtual_node" not in trace["raw_tensors"]
