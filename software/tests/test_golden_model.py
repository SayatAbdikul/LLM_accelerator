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

        # Manually check tile config is set: (M, N, K, weight_int4, m_exact)
        # — 0-based M/N/K; weight_int4 False (W4 ext 2026-05-24); m_exact 0
        # = full tiles (freeze §6 rev 2026-07-10).
        assert sim.state.tile_config == (0, 0, 0, False, 0)

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
