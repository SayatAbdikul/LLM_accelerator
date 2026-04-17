"""Tests for compiler (single-layer compile + verify)."""
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from taccel.compiler.compiler import Compiler
from taccel.compiler.tiler import tile_matmul, pad_dim, TILE
from taccel.compiler.codegen import CodeGenerator
from taccel.compiler.ir import IRGraph, IRNode
from taccel.golden_model import Simulator
from taccel.quantizer.bias_correction import compute_bias_corrections
from taccel.quantizer.quantize import quantize_tensor, quantize_weights

from taccel.compiler.memory_alloc import MemoryAllocator, BufferAllocator
from taccel.isa.opcodes import ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from taccel.isa.instructions import (
    BufCopyInsn,
    DequantAddInsn,
    GeluInsn,
    MatmulInsn,
    RequantInsn,
    RequantPcInsn,
    SoftmaxAttnVInsn,
    SoftmaxInsn,
    VaddInsn,
)
from taccel.isa.encoding import encode
from taccel.assembler.assembler import ProgramBinary


class TestMemoryAllocator:
    def test_basic_alloc(self):
        alloc = BufferAllocator(0, 128 * 1024)
        a = alloc.alloc("a", 1024)
        assert a.offset_units == 0
        assert a.size_bytes == 1024

    def test_sequential_alloc(self):
        alloc = BufferAllocator(0, 128 * 1024)
        a = alloc.alloc("a", 256)
        b = alloc.alloc("b", 512)
        assert b.offset_units == a.offset_units + a.size_units

    def test_alloc_exceeds_capacity(self):
        alloc = BufferAllocator(0, 256)
        with pytest.raises(MemoryError):
            alloc.alloc("too_big", 512)

    def test_free_and_reuse(self):
        alloc = BufferAllocator(0, 128 * 1024)
        a = alloc.alloc("a", 256)
        b = alloc.alloc("b", 256)
        a_offset = a.offset_units
        alloc.free("a")
        # b stays at its original offset (no compaction)
        b_check = alloc.get("b")
        assert b_check.offset_units == b.offset_units
        # freed space is reusable
        c = alloc.alloc("c", 256)
        assert c.offset_units == a_offset

    def test_high_water_mark(self):
        alloc = BufferAllocator(0, 128 * 1024)
        alloc.alloc("a", 1024 * 16)
        alloc.alloc("b", 1024 * 16)
        hw = alloc.high_water_units * 16
        assert hw >= 2048 * 16


class TestTiledLinearLayer:
    def test_linear_tile_schedule(self):
        """Single linear layer [197, 192] @ [192, 192] tile schedule."""
        M, N, K = 197, 192, 192
        sched = tile_matmul(M, N, K)

        # Should have 13 * 12 * 12 = 1872 tile ops
        assert sched.m_tiles == 13
        assert sched.n_tiles == 12
        assert sched.k_tiles == 12
        assert len(sched.ops) == 13 * 12 * 12

    def test_linear_output_fits_abuf(self):
        """Standard linear output [208, 192] = 39,936 bytes << 128KB."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(192)
        output_bytes = M_pad * N_pad
        assert output_bytes < ABUF_SIZE, f"{output_bytes} >= {ABUF_SIZE}"

    def test_fc1_output_exceeds_abuf(self):
        """FC1 output [208, 768] = 159,744 bytes > 128KB ABUF, needs strip-mining."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(768)
        output_bytes = M_pad * N_pad
        assert output_bytes > ABUF_SIZE, f"Expected overflow, got {output_bytes}"

    def test_fc1_weights_fit_wbuf(self):
        """FC1 weights [768, 192] = 147,456 bytes < 256KB WBUF."""
        weight_bytes = pad_dim(768) * pad_dim(192)
        assert weight_bytes < WBUF_SIZE

    def test_fc2_weights_fit_wbuf(self):
        """FC2 weights [192, 768] = 147,456 bytes < 256KB WBUF."""
        weight_bytes = pad_dim(192) * pad_dim(768)
        assert weight_bytes < WBUF_SIZE


class TestMemoryconsistency:
    def test_abuf_max_tiles(self):
        """Verify ABUF can hold [208, 192] INT8 activations."""
        M_pad = pad_dim(197)
        N_pad = pad_dim(192)
        bytes_needed = M_pad * N_pad  # 39,936
        assert bytes_needed < ABUF_SIZE

    def test_wbuf_weight_capacity(self):
        """Verify WBUF can hold [192, 192] INT8 weight tile."""
        bytes_needed = 192 * 192  # 36,864
        assert bytes_needed < WBUF_SIZE

    def test_accum_capacity(self):
        """ACCUM is 64KB. Full [208, 192] INT32 doesn't fit — strip mining is required.
        One strip [16, 192] = 12,288 bytes does fit.
        """
        # Full tile doesn't fit
        M_pad = pad_dim(197)
        N_pad = pad_dim(192)
        full_bytes = M_pad * N_pad * 4
        assert full_bytes > ACCUM_SIZE, "Expected full tile to exceed ACCUM (strip-mining needed)"

        # One strip of 16 rows fits
        strip_bytes = TILE * N_pad * 4  # [16, 192] INT32
        assert strip_bytes <= ACCUM_SIZE, f"One strip {strip_bytes}B > ACCUM {ACCUM_SIZE}B"


class TestCompilerGuards:
    def test_requant_pc_fc1_rejects_overlap_with_gelu_from_accum(self):
        compiler = Compiler()

        with pytest.raises(ValueError, match="REQUANT_PC FC1 cannot overlap with GELU-from-ACCUM"):
            compiler.compile(
                {},
                calibration=SimpleNamespace(scales={}),
                gelu_from_accum=True,
                gelu_from_accum_blocks={3},
                requant_pc_fc1=True,
                requant_pc_fc1_blocks={3},
            )

    def test_attention_kt_wbuf_pressure(self):
        """3 heads' K^T fits in WBUF alongside other data."""
        head_dim = 64
        M_pad = pad_dim(197)
        kt_per_head = head_dim * M_pad  # 64 * 208 = 13,312 bytes
        kt_total_3_heads = kt_per_head * 3  # 39,936 bytes
        assert kt_total_3_heads < WBUF_SIZE, \
            f"3 heads K^T ({kt_total_3_heads}B) doesn't fit in WBUF ({WBUF_SIZE}B)"

    def test_compiler_manifest_records_fc2_and_out_proj_requant_pc_options(self):
        compiler = Compiler()
        weight_name = "vit.encoder.layer.9.output.dense.weight"
        codegen = SimpleNamespace(
            trace_manifest={},
            dram_layout={},
            mem=SimpleNamespace(dram_temp_total=0),
        )

        manifest = compiler._build_compiler_manifest(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.linspace(0.01, 0.16, 16, dtype=np.float16),
                ),
            },
            cal_scales={},
            prescaled_biases={},
            codegen=codegen,
            bias_correction_biases=[],
            weight_quantization_overrides={},
            gelu_from_accum=False,
            gelu_from_accum_blocks=None,
            dequant_add_residual1_blocks=None,
            fused_softmax_attnv_blocks=None,
            fused_softmax_attnv_accum_out_proj=False,
            requant_pc_qkv=False,
            requant_pc_qkv_selection=None,
            requant_pc_fc1=False,
            requant_pc_fc1_blocks=None,
            requant_pc_fc2=True,
            requant_pc_fc2_blocks={9},
            requant_pc_out_proj=True,
            requant_pc_out_proj_blocks={10, 11},
            requant_pc_weight_names={weight_name},
            requant_pc_scale_tables={
                weight_name: np.linspace(0.02, 0.32, 16, dtype=np.float16),
            },
            data_base=0,
            input_offset=0,
            pos_embed_patch_dram_offset=0,
            pos_embed_cls_dram_offset=0,
            cls_token_dram_offset=0,
        )

        assert manifest["compiler"]["options"]["requant_pc_fc2"] is True
        assert manifest["compiler"]["options"]["requant_pc_fc2_blocks"] == [9]
        assert manifest["compiler"]["options"]["requant_pc_out_proj"] is True
        assert manifest["compiler"]["options"]["requant_pc_out_proj_blocks"] == [10, 11]
        assert manifest["weights"][weight_name]["uses_requant_pc"] is True


class TestBiasCorrection:
    class _TinyLinear(torch.nn.Module):
        def __init__(self, weight: np.ndarray, bias: np.ndarray):
            super().__init__()
            out_features, in_features = weight.shape
            self.classifier = torch.nn.Linear(in_features, out_features, bias=True)
            with torch.no_grad():
                self.classifier.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
                self.classifier.bias.copy_(torch.from_numpy(bias.astype(np.float32)))

        def forward(self, x):
            return self.classifier(x)

    @staticmethod
    def _classifier_output_mse(model, sample_inputs, q_weight, q_scales, act_scale, bias_correction=None):
        weight_fp32 = model.state_dict()["classifier.weight"].detach().cpu().numpy().astype(np.float32)
        bias_fp32 = model.state_dict()["classifier.bias"].detach().cpu().numpy().astype(np.float32)
        dq_weight = q_weight.astype(np.float32) * q_scales.astype(np.float32).reshape(-1, 1)
        correction = (
            np.asarray(bias_correction, dtype=np.float32)
            if bias_correction is not None else np.zeros_like(bias_fp32, dtype=np.float32)
        )
        sq_err = []
        for sample in sample_inputs:
            x = sample["x"].detach().cpu().numpy().astype(np.float32)
            x_q = np.clip(np.round(x / act_scale), -128, 127).astype(np.int8)
            x_dq = x_q.astype(np.float32) * np.float32(act_scale)
            y_fp32 = x @ weight_fp32.T + bias_fp32
            y_qdq = x_dq @ dq_weight.T + bias_fp32 + correction
            sq_err.append(np.mean((y_fp32 - y_qdq) ** 2))
        return float(np.mean(sq_err))

    def test_compute_bias_corrections_is_zero_when_quantized_path_is_exact(self):
        weight = np.array([[12.7, -0.3], [0.2, 0.0]], dtype=np.float32)
        bias = np.array([0.5, -1.0], dtype=np.float32)
        model = self._TinyLinear(weight, bias)
        state_dict = model.state_dict()
        quant_weights = quantize_weights(state_dict)
        sample_inputs = [
            {"x": torch.tensor([[1.0, -0.5]], dtype=torch.float32)},
            {"x": torch.tensor([[0.0, 0.25]], dtype=torch.float32)},
        ]

        corrections = compute_bias_corrections(
            model,
            state_dict,
            quant_weights,
            {"final_ln": 0.25},
            sample_inputs,
            ["classifier.weight"],
        )

        np.testing.assert_allclose(corrections["classifier.bias"], np.zeros_like(bias), atol=2e-3)

    def test_compute_bias_corrections_reduces_classifier_output_mse(self):
        weight = np.array([[0.37, -0.28], [0.11, 0.34]], dtype=np.float32)
        bias = np.array([0.05, -0.03], dtype=np.float32)
        model = self._TinyLinear(weight, bias)
        state_dict = model.state_dict()
        quant_weights = quantize_weights(state_dict)
        sample_inputs = [
            {"x": torch.tensor([[0.23, -0.11]], dtype=torch.float32)},
            {"x": torch.tensor([[0.41, 0.17]], dtype=torch.float32)},
            {"x": torch.tensor([[-0.19, 0.28]], dtype=torch.float32)},
        ]
        act_scale = 0.01

        corrections = compute_bias_corrections(
            model,
            state_dict,
            quant_weights,
            {"final_ln": act_scale},
            sample_inputs,
            ["classifier.weight"],
        )

        q_weight, q_scales = quantize_tensor(weight, per_channel=False)
        mse_before = self._classifier_output_mse(
            model,
            sample_inputs,
            q_weight,
            q_scales,
            act_scale,
        )
        mse_after = self._classifier_output_mse(
            model,
            sample_inputs,
            q_weight,
            q_scales,
            act_scale,
            bias_correction=corrections["classifier.bias"],
        )

        assert np.max(np.abs(corrections["classifier.bias"])) > 0.0
        assert mse_after < mse_before


class TestRequantPcCodegen:
    def test_codegen_emits_requant_pc_for_selected_weight(self):
        weight_name = "vit.encoder.layer.0.attention.attention.query.weight_h0"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="block0_head0_query",
            inputs=["ln1_input", weight_name],
            output_shape=(16, 16),
        ))

        codegen = CodeGenerator(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.linspace(0.01, 0.16, 16, dtype=np.float16),
                ),
            },
            calibration_scales={
                "ln1_input": 0.125,
                "block0_head0_query": 0.25,
            },
            prescaled_biases={},
            requant_pc_weight_names={weight_name},
            requant_pc_scale_tables={
                weight_name: np.linspace(0.02, 0.32, 16, dtype=np.float16),
            },
        )

        instructions, _ = codegen.generate(graph)

        assert any(isinstance(insn, RequantPcInsn) for insn in instructions)
        assert not any(isinstance(insn, RequantInsn) for insn in instructions)
        assert f"{weight_name}__requant_pc" in codegen.dram_layout

    def test_attention_projection_emits_debug_input_and_accum_traces(self):
        weight_name = "vit.encoder.layer.0.attention.attention.query.weight_h0"
        bias_name = "vit.encoder.layer.0.attention.attention.query.bias_h0"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="block0_head0_query",
            inputs=["ln1_input", weight_name],
            output_shape=(16, 16),
            attrs={"bias": bias_name},
        ))

        codegen = CodeGenerator(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.full(16, 0.125, dtype=np.float16),
                ),
            },
            calibration_scales={
                "ln1_input": 0.125,
                "block0_head0_query": 0.25,
            },
            prescaled_biases={
                bias_name: np.arange(16, dtype=np.int32),
            },
        )

        codegen.generate(graph)

        flat_events = [
            (pc, event["node_name"], event["buf_id"], event["offset_units"], event["dtype"])
            for pc, events in sorted(codegen.trace_manifest.items())
            for event in events
        ]

        assert [name for _, name, *_ in flat_events] == [
            "block0_head0_query__act_input",
            "block0_head0_query__act_input_padded",
            "block0_head0_query__weight_input",
            "block0_head0_query__accum_pre_bias",
            "block0_head0_query__accum_pre_bias_padded",
            "block0_head0_query__bias_input",
            "block0_head0_query__accum",
            "block0_head0_query__accum_padded",
            "block0_head0_query",
            "block0_head0_query__output_padded",
        ]
        pre_matmul_pc = flat_events[0][0]
        act_input_padded_pc = flat_events[1][0]
        pre_bias_pc = flat_events[3][0]
        pre_bias_padded_pc = flat_events[4][0]
        bias_pc = flat_events[5][0]
        accum_pc = flat_events[6][0]
        accum_padded_pc = flat_events[7][0]
        output_pc = flat_events[8][0]
        output_padded_pc = flat_events[9][0]
        assert flat_events[0][2] == BUF_ABUF
        assert flat_events[1][2] == BUF_ABUF
        assert flat_events[2][2] == BUF_WBUF
        assert flat_events[3][2] == BUF_ACCUM
        assert flat_events[4][2] == BUF_ACCUM
        assert flat_events[5][2] == BUF_WBUF
        assert flat_events[6][2] == BUF_ACCUM
        assert flat_events[7][2] == BUF_ACCUM
        assert flat_events[8][2] == BUF_ABUF
        assert flat_events[9][2] == BUF_ABUF
        assert flat_events[0][4] == "int8"
        assert flat_events[1][4] == "int8"
        assert flat_events[2][4] == "int8"
        assert flat_events[3][4] == "int32"
        assert flat_events[4][4] == "int32"
        assert flat_events[5][4] == "int32"
        assert flat_events[6][4] == "int32"
        assert flat_events[7][4] == "int32"
        assert flat_events[8][4] == "int8"
        assert flat_events[9][4] == "int8"
        assert act_input_padded_pc == pre_matmul_pc
        assert flat_events[2][0] == pre_matmul_pc
        assert pre_bias_pc > pre_matmul_pc
        assert pre_bias_padded_pc == pre_bias_pc
        assert bias_pc > pre_bias_pc
        assert accum_pc > bias_pc
        assert accum_padded_pc == accum_pc
        assert output_pc > accum_pc
        assert output_padded_pc == output_pc


class TestGeluFromAccumCodegen:
    @staticmethod
    def _make_strip_mined_fc1_codegen(selected_blocks=None):
        weight_name = "vit.encoder.layer.3.intermediate.dense.weight"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="block3_fc1",
            inputs=["ln2_input", weight_name],
            output_shape=(32, 16),
            attrs={"strip_mine": True, "inline_gelu": "block3_gelu"},
        ))
        return CodeGenerator(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.full(16, 0.125, dtype=np.float16),
                ),
            },
            calibration_scales={
                "ln2_input": 0.125,
                "block3_fc1": 0.25,
                "block3_gelu": 0.25,
            },
            prescaled_biases={},
            gelu_from_accum=True,
            gelu_from_accum_blocks=selected_blocks,
        ), graph

    def test_strip_mined_fc1_uses_accum_gelu_for_selected_block(self):
        codegen, graph = self._make_strip_mined_fc1_codegen(selected_blocks={3})

        instructions, _ = codegen.generate(graph)

        gelu_insns = [insn for insn in instructions if isinstance(insn, GeluInsn)]
        assert gelu_insns
        assert all(insn.src1_buf == BUF_ACCUM for insn in gelu_insns)
        assert not any(isinstance(insn, RequantInsn) for insn in instructions)

    def test_strip_mined_fc1_falls_back_to_int8_gelu_for_unselected_block(self):
        codegen, graph = self._make_strip_mined_fc1_codegen(selected_blocks={2})

        instructions, _ = codegen.generate(graph)

        gelu_insns = [insn for insn in instructions if isinstance(insn, GeluInsn)]
        assert gelu_insns
        assert all(insn.src1_buf == BUF_ABUF for insn in gelu_insns)
        assert any(isinstance(insn, RequantInsn) for insn in instructions)

    def test_strip_mined_fc1_emits_requant_pc_for_selected_weight(self):
        weight_name = "vit.encoder.layer.3.intermediate.dense.weight"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="block3_fc1",
            inputs=["ln2_input", weight_name],
            output_shape=(32, 16),
            attrs={"strip_mine": True, "inline_gelu": "block3_gelu"},
        ))

        codegen = CodeGenerator(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.linspace(0.02, 0.17, 16, dtype=np.float16),
                ),
            },
            calibration_scales={
                "ln2_input": 0.125,
                "block3_fc1": 0.25,
                "block3_gelu": 0.25,
            },
            prescaled_biases={},
            requant_pc_weight_names={weight_name},
            requant_pc_scale_tables={
                weight_name: np.linspace(0.01, 0.16, 16, dtype=np.float16),
            },
        )

        instructions, _ = codegen.generate(graph)

        assert any(isinstance(insn, RequantPcInsn) for insn in instructions)
        assert not any(isinstance(insn, RequantInsn) for insn in instructions)
        assert f"{weight_name}__requant_pc" in codegen.dram_layout

    def test_strip_mined_codegen_emits_requant_pc_for_selected_weight(self):
        weight_name = "vit.encoder.layer.0.attention.output.dense.weight"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="block0_out_proj",
            inputs=["concat_input", weight_name],
            output_shape=(32, 16),
            attrs={"strip_mine": True},
        ))

        codegen = CodeGenerator(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.linspace(0.01, 0.16, 16, dtype=np.float16),
                ),
            },
            calibration_scales={
                "concat_input": 0.125,
                "block0_out_proj": 0.25,
            },
            prescaled_biases={},
            requant_pc_weight_names={weight_name},
            requant_pc_scale_tables={
                weight_name: np.linspace(0.02, 0.32, 16, dtype=np.float16),
            },
        )

        instructions, _ = codegen.generate(graph)

        assert any(isinstance(insn, RequantPcInsn) for insn in instructions)
        assert not any(isinstance(insn, RequantInsn) for insn in instructions)
        assert f"{weight_name}__requant_pc" in codegen.dram_layout


class TestPosEmbedTraceCodegen:
    def test_pos_embed_add_emits_input_and_output_trace_events(self):
        codegen = CodeGenerator(
            weight_data={},
            calibration_scales={
                "pos_embed_add": 0.25,
            },
            prescaled_biases={},
        )
        codegen.dram_layout["position_embeddings"] = 0x1000
        act_alloc = codegen.mem.abuf.alloc("cls_prepend", pad_dim(197) * pad_dim(192))

        node = IRNode(
            op="pos_embed_add",
            name="pos_embed_add",
            inputs=["cls_prepend", "position_embeddings"],
            output_shape=(197, 192),
        )

        codegen._emit_pos_embed_add(node)

        flat_events = [
            (pc, event["node_name"], event["buf_id"], event["offset_units"])
            for pc, events in sorted(codegen.trace_manifest.items())
            for event in events
        ]

        input_pc = flat_events[0][0]
        assert flat_events[0] == (input_pc, "pos_embed_add__act_input", BUF_ABUF, act_alloc.offset_units)
        assert flat_events[1][0] == input_pc
        assert flat_events[1][1] == "pos_embed_add__pos_input"
        assert flat_events[1][2] == BUF_WBUF
        assert flat_events[2][0] == input_pc + 1
        assert flat_events[2][1] == "pos_embed_add"


class TestQktTraceCodegen:
    def test_block0_ln1_emits_padded_input_and_output_debug_traces(self):
        codegen = CodeGenerator(
            weight_data={},
            calibration_scales={
                "pos_embed_add": 0.125,
                "block0_ln1": 0.25,
            },
            prescaled_biases={},
        )
        in_alloc = codegen.mem.abuf.alloc("pos_embed_add", pad_dim(197) * pad_dim(192))
        node = IRNode(
            op="layernorm",
            name="block0_ln1",
            inputs=[
                "pos_embed_add",
                "vit.encoder.layer.0.layernorm_before.weight",
                "vit.encoder.layer.0.layernorm_before.bias",
            ],
            output_shape=(197, 192),
        )

        codegen._emit_layernorm(node)

        flat_events = [
            (pc, event["node_name"], event["buf_id"], event["offset_units"], event["dtype"])
            for pc, events in sorted(codegen.trace_manifest.items())
            for event in events
        ]
        names = [name for _, name, *_ in flat_events]

        assert names == [
            "block0_ln1__input_padded",
            "block0_ln1",
            "block0_ln1__output_padded",
        ]
        assert flat_events[0][2:] == (BUF_ABUF, in_alloc.offset_units, "int8")
        assert flat_events[1][2] == BUF_ABUF
        assert flat_events[2][2] == BUF_ABUF
        assert flat_events[0][0] < flat_events[1][0]
        assert flat_events[2][0] == flat_events[1][0]

        input_event = codegen.trace_manifest[flat_events[0][0]][0]
        output_events = codegen.trace_manifest[flat_events[1][0]]
        output_padded_event = output_events[1]
        assert input_event["logical_rows"] == pad_dim(197)
        assert input_event["logical_cols"] == pad_dim(192)
        assert output_padded_event["node_name"] == "block0_ln1__output_padded"
        assert output_padded_event["logical_rows"] == pad_dim(197)
        assert output_padded_event["logical_cols"] == pad_dim(192)

    def test_qkt_emits_query_and_k_debug_trace_events(self):
        codegen = CodeGenerator(
            weight_data={},
            calibration_scales={
                "block0_head0_query": 0.125,
                "block0_head0_key": 0.25,
                "block0_head0_qkt": 0.015625,
                "block0_head0_softmax": 0.0625,
            },
            prescaled_biases={},
        )
        codegen.dram_layout["__zero_pad__"] = 0x2000
        q_alloc = codegen.mem.abuf.alloc("block0_head0_query", pad_dim(197) * pad_dim(64))
        k_alloc = codegen.mem.abuf.alloc("block0_head0_key", pad_dim(197) * pad_dim(64))

        node = IRNode(
            op="matmul_qkt",
            name="block0_head0_qkt",
            inputs=["block0_head0_query", "block0_head0_key"],
            output_shape=(197, 197),
            attrs={"head_idx": 0, "scale": 0.125},
        )

        codegen._emit_qkt(node)

        trace_events = [
            (pc, event["node_name"], event["buf_id"], event["offset_units"], event["row_start"])
            for pc, events in sorted(codegen.trace_manifest.items())
            for event in events
        ]
        names = [name for _, name, _, _, _ in trace_events]

        key_input_idx = names.index("block0_head0_qkt__key_padded_input")
        key_transpose_idx = names.index("block0_head0_qkt__key_transposed")
        accum_pre_idx = names.index("block0_head0_qkt__accum_pre_matmul")
        accum_pre_next_idx = names.index("block0_head0_qkt__accum_pre_matmul_next")
        query_input_idx = names.index("block0_head0_qkt__query_input")
        qkt_idx = names.index("block0_head0_qkt")
        accum_pre_softmax_idx = names.index("block0_head0_qkt__accum_pre_softmax")
        accum_pre_softmax_next_idx = names.index("block0_head0_qkt__accum_pre_softmax_next")

        key_input_pc = trace_events[key_input_idx][0]
        key_transpose_pc = trace_events[key_transpose_idx][0]
        accum_pre_pc = trace_events[accum_pre_idx][0]
        accum_pre_next_pc = trace_events[accum_pre_next_idx][0]
        query_input_pc = trace_events[query_input_idx][0]
        qkt_pc = trace_events[qkt_idx][0]
        accum_pre_softmax_pc = trace_events[accum_pre_softmax_idx][0]
        accum_pre_softmax_next_pc = trace_events[accum_pre_softmax_next_idx][0]

        assert key_input_pc == key_transpose_pc
        assert accum_pre_pc == accum_pre_next_pc
        assert accum_pre_pc < query_input_pc
        assert query_input_pc < qkt_pc
        assert qkt_pc < accum_pre_softmax_pc
        assert accum_pre_softmax_pc == accum_pre_softmax_next_pc
        assert trace_events[key_input_idx][1:4] == (
            "block0_head0_qkt__key_padded_input",
            BUF_ABUF,
            k_alloc.offset_units,
        )
        assert trace_events[key_transpose_idx][1:4] == (
            "block0_head0_qkt__key_transposed",
            BUF_WBUF,
            0,
        )
        assert trace_events[accum_pre_idx][1:4] == (
            "block0_head0_qkt__accum_pre_matmul",
            BUF_ACCUM,
            0,
        )
        assert trace_events[accum_pre_next_idx][1:4] == (
            "block0_head0_qkt__accum_pre_matmul_next",
            BUF_ACCUM,
            0,
        )
        assert trace_events[query_input_idx][1:4] == (
            "block0_head0_qkt__query_input",
            BUF_ABUF,
            q_alloc.offset_units,
        )
        assert trace_events[accum_pre_idx][4] == 0
        assert trace_events[query_input_idx][4] == 0
        assert trace_events[qkt_idx][4] == 0

        manifest = codegen.trace_manifest
        key_events = manifest[key_input_pc]
        assert key_events[0]["node_name"] == "block0_head0_qkt__key_padded_input"
        assert key_events[0]["logical_rows"] == pad_dim(197)
        assert key_events[0]["logical_cols"] == pad_dim(64)
        assert key_events[1]["node_name"] == "block0_head0_qkt__key_transposed"
        assert key_events[1]["logical_rows"] == pad_dim(64)
        assert key_events[1]["logical_cols"] == pad_dim(197)

        accum_pre_events = [event for event in manifest[accum_pre_pc] if event["node_name"] == "block0_head0_qkt__accum_pre_matmul"]
        assert len(accum_pre_events) == 1
        assert accum_pre_events[0]["logical_rows"] == 16
        assert accum_pre_events[0]["logical_cols"] == 197
        assert accum_pre_events[0]["full_rows"] == 197
        assert accum_pre_events[0]["full_cols"] == 197
        assert accum_pre_events[0]["capture_phase"] == "retire_cycle"

        accum_pre_next_events = [
            event for event in manifest[accum_pre_next_pc]
            if event["node_name"] == "block0_head0_qkt__accum_pre_matmul_next"
        ]
        assert len(accum_pre_next_events) == 1
        assert accum_pre_next_events[0]["logical_rows"] == 16
        assert accum_pre_next_events[0]["logical_cols"] == 197
        assert accum_pre_next_events[0]["capture_phase"] == "retire_plus_1"

        query_events = [event for event in manifest[query_input_pc] if event["node_name"] == "block0_head0_qkt__query_input"]
        assert len(query_events) == 1
        assert query_events[0]["logical_rows"] == 16
        assert query_events[0]["logical_cols"] == 64
        assert query_events[0]["full_rows"] == 197
        assert query_events[0]["full_cols"] == 64

        pre_softmax_events = [
            event for event in manifest[accum_pre_softmax_pc]
            if event["node_name"] == "block0_head0_qkt__accum_pre_softmax"
        ]
        assert len(pre_softmax_events) == 1
        assert pre_softmax_events[0]["logical_rows"] == 16
        assert pre_softmax_events[0]["logical_cols"] == 197
        assert pre_softmax_events[0]["capture_phase"] == "retire_cycle"

        pre_softmax_next_events = [
            event for event in manifest[accum_pre_softmax_next_pc]
            if event["node_name"] == "block0_head0_qkt__accum_pre_softmax_next"
        ]
        assert len(pre_softmax_next_events) == 1
        assert pre_softmax_next_events[0]["logical_rows"] == 16
        assert pre_softmax_next_events[0]["logical_cols"] == 197
        assert pre_softmax_next_events[0]["capture_phase"] == "retire_plus_1"


class TestFusedSoftmaxAttnVCodegen:
    def test_block_selected_qkt_emits_fused_softmax_attnv(self):
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul_qkt",
            name="block11_head0_qkt",
            inputs=["block11_head0_query", "block11_head0_key"],
            output_shape=(16, 16),
            attrs={"head_idx": 0, "scale": 0.125},
        ))
        graph.add_node(IRNode(
            op="scale_mul",
            name="block11_head0_scale",
            inputs=["block11_head0_qkt"],
            output_shape=(16, 16),
            attrs={"scale": 0.125},
        ))
        graph.add_node(IRNode(
            op="softmax",
            name="block11_head0_softmax",
            inputs=["block11_head0_scale"],
            output_shape=(16, 16),
        ))
        graph.add_node(IRNode(
            op="matmul_attn_v",
            name="block11_head0_attn_v",
            inputs=["block11_head0_softmax", "block11_head0_value"],
            output_shape=(16, 64),
            attrs={"head_idx": 0},
        ))

        codegen = CodeGenerator(
            weight_data={},
            calibration_scales={
                "block11_head0_query": 0.125,
                "block11_head0_key": 0.125,
                "block11_head0_value": 0.25,
                "block11_head0_qkt": 0.015625,
                "block11_head0_softmax": 0.0625,
                "block11_head0_attn_v": 0.25,
            },
            prescaled_biases={},
            fused_softmax_attnv_blocks={11},
        )

        instructions, _ = codegen.generate(graph)

        assert any(isinstance(insn, SoftmaxAttnVInsn) for insn in instructions)
        assert not any(isinstance(insn, SoftmaxInsn) for insn in instructions)

    def test_fused_out_proj_accumulates_heads_without_concat_rows(self):
        graph = IRGraph()
        graph.add_node(IRNode(
            op="concat_heads",
            name="block11_concat",
            inputs=[f"block11_head{head_idx}_attn_v" for head_idx in range(3)],
            output_shape=(197, 192),
        ))
        graph.add_node(IRNode(
            op="matmul",
            name="block11_out_proj",
            inputs=["block11_concat", "vit.encoder.layer.11.attention.output.dense.weight"],
            output_shape=(197, 192),
            attrs={
                "bias": "vit.encoder.layer.11.attention.output.dense.bias",
                "strip_mine": True,
            },
        ))

        codegen = CodeGenerator(
            weight_data={
                "vit.encoder.layer.11.attention.output.dense.weight": (
                    np.ones((192, 192), dtype=np.int8),
                    np.full(192, 0.0625, dtype=np.float16),
                ),
            },
            calibration_scales={
                "block11_concat": 0.125,
                "block11_head0_attn_v": 0.10,
                "block11_head1_attn_v": 0.125,
                "block11_head2_attn_v": 0.15,
                "block11_out_proj": 0.25,
            },
            prescaled_biases={
                "vit.encoder.layer.11.attention.output.dense.bias": np.zeros(192, dtype=np.int32),
            },
            fused_softmax_attnv_blocks={11},
            fused_softmax_attnv_accum_out_proj_blocks={11},
        )
        for head_idx in range(3):
            codegen.mem.wbuf.alloc(f"block11_head{head_idx}_attn_v", pad_dim(197) * 64)

        instructions, _ = codegen.generate(graph)

        matmuls = [insn for insn in instructions if isinstance(insn, MatmulInsn)]
        assert len(matmuls) == 13 * 3
        assert [insn.flags for insn in matmuls[:6]] == [0, 1, 1, 0, 1, 1]
        short_concat_copies = [
            insn for insn in instructions
            if isinstance(insn, BufCopyInsn) and insn.length == 4
        ]
        assert not short_concat_copies


class TestDequantAddResidual1Codegen:
    def test_block_selected_residual1_emits_dequant_add(self):
        weight_name = "vit.encoder.layer.11.attention.output.dense.weight"
        bias_name = "vit.encoder.layer.11.attention.output.dense.bias"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="block11_out_proj",
            inputs=["block11_concat", weight_name],
            output_shape=(16, 16),
            attrs={"bias": bias_name},
        ))
        graph.add_node(IRNode(
            op="vadd",
            name="block11_residual1",
            inputs=["block11_out_proj", "block10_residual2"],
            output_shape=(16, 16),
        ))

        codegen = CodeGenerator(
            weight_data={
                weight_name: (
                    np.ones((16, 16), dtype=np.int8),
                    np.full(16, 0.125, dtype=np.float16),
                ),
            },
            calibration_scales={
                "block11_concat": 0.125,
                "block10_residual2": 0.25,
                "block11_out_proj": 0.25,
                "block11_residual1": 0.25,
            },
            prescaled_biases={bias_name: np.zeros(16, dtype=np.int32)},
            dequant_add_residual1_blocks={11},
        )
        codegen.mem.abuf.alloc("block11_concat", 16 * 16)
        codegen.mem.abuf.alloc("block10_residual2", 16 * 16)

        instructions, _ = codegen.generate(graph)

        assert any(isinstance(insn, DequantAddInsn) for insn in instructions)
        assert not any(
            isinstance(insn, VaddInsn) and insn.src1_buf == BUF_ABUF
            for insn in instructions
        )


class TestMatmulRequantNumerics:
    @staticmethod
    def _pack_weight(q_rows: np.ndarray, scales: np.ndarray):
        q_rows = np.asarray(q_rows, dtype=np.int8)
        scales = np.asarray(scales, dtype=np.float32)
        out_ch, in_ch = q_rows.shape
        out_pad = pad_dim(out_ch)
        in_pad = pad_dim(in_ch)
        q_pad = np.zeros((out_pad, in_pad), dtype=np.int8)
        q_pad[:out_ch, :in_ch] = q_rows
        scale_pad = np.full(out_pad, float(scales[-1]), dtype=np.float16)
        scale_pad[:out_ch] = scales.astype(np.float16)
        return np.ascontiguousarray(q_pad.T), scale_pad, q_pad

    @staticmethod
    def _run_single_matmul(weight_name: str, input_int8: np.ndarray, weight_data,
                           calibration_scales, prescaled_biases,
                           requant_pc_scale_table=None):
        bias_name = f"{weight_name.rsplit('.', 1)[0]}.bias"
        graph = IRGraph()
        graph.add_node(IRNode(
            op="matmul",
            name="test_node",
            inputs=["act", weight_name],
            output_shape=(input_int8.shape[0], 10),
            attrs={"bias": bias_name},
        ))

        codegen = CodeGenerator(
            weight_data={weight_name: weight_data},
            calibration_scales=calibration_scales,
            prescaled_biases=prescaled_biases,
            requant_pc_weight_names={weight_name} if requant_pc_scale_table is not None else set(),
            requant_pc_scale_tables=(
                {weight_name: requant_pc_scale_table}
                if requant_pc_scale_table is not None else {}
            ),
        )
        instructions, dram_data = codegen.generate(graph)
        program = ProgramBinary(
            instructions=b"".join(encode(insn) for insn in instructions),
            data=dram_data,
            entry_point=0,
            insn_count=len(instructions),
        )
        program = ProgramBinary.from_bytes(program.to_bytes())

        sim = Simulator()
        sim.load_program(program)

        act_alloc = codegen.mem.abuf.get("act")
        assert act_alloc is not None
        sim.state.abuf[act_alloc.offset_units * 16:act_alloc.offset_units * 16 + input_int8.size] = (
            np.asarray(input_int8, dtype=np.int8).tobytes()
        )
        sim.run()

        out_alloc = codegen.mem.abuf.get("test_node")
        assert out_alloc is not None
        out_rows = pad_dim(input_int8.shape[0])
        out_cols = pad_dim(10)
        raw = bytes(
            sim.state.abuf[
                out_alloc.offset_units * 16:
                out_alloc.offset_units * 16 + out_rows * out_cols
            ]
        )
        return np.frombuffer(raw, dtype=np.int8).reshape(out_rows, out_cols)

    def test_scalar_requant_codegen_matches_direct_formula_with_uniform_scales(self):
        input_scale = np.float32(0.25)
        target_scale = np.float32(0.125)
        input_int8 = ((np.arange(256).reshape(16, 16) % 7) - 3).astype(np.int8)

        q_rows = (((np.arange(160).reshape(10, 16) * 3) % 9) - 4).astype(np.int8)
        w_scale = np.float32(0.0625)
        packed_weight, packed_scales, q_pad = self._pack_weight(
            q_rows,
            np.full(10, w_scale, dtype=np.float32),
        )
        bias_i32 = (np.arange(10, dtype=np.int32) - 4).astype(np.int32)
        bias_pad = np.pad(bias_i32, (0, pad_dim(10) - 10), constant_values=0)

        result = self._run_single_matmul(
            "test.weight",
            input_int8,
            (packed_weight, packed_scales),
            {"act": float(input_scale), "test_node": float(target_scale)},
            {"test.bias": bias_pad},
        )

        accum = input_int8.astype(np.int32) @ q_pad.T.astype(np.int32)
        accum += bias_pad.reshape(1, -1)
        requant_scale = np.float32(input_scale * w_scale / target_scale)
        expected = np.clip(np.round(accum.astype(np.float32) * requant_scale), -128, 127).astype(np.int8)

        np.testing.assert_array_equal(result, expected)

    def test_requant_pc_codegen_matches_direct_formula_with_per_channel_scales(self):
        input_scale = np.float32(0.125)
        target_scale = np.float32(0.20)
        input_int8 = (((np.arange(256).reshape(16, 16) * 5) % 11) - 5).astype(np.int8)

        q_rows = (((np.arange(160).reshape(10, 16) * 7) % 13) - 6).astype(np.int8)
        per_channel_scales = np.linspace(0.02, 0.11, 10, dtype=np.float32)
        packed_weight, packed_scales, q_pad = self._pack_weight(q_rows, per_channel_scales)
        bias_i32 = (np.arange(10, dtype=np.int32) - 3).astype(np.int32)
        bias_pad = np.pad(bias_i32, (0, pad_dim(10) - 10), constant_values=0)
        pc_scales = np.full(pad_dim(10), per_channel_scales[-1], dtype=np.float16)
        pc_scales[:10] = (input_scale * per_channel_scales / target_scale).astype(np.float16)

        result = self._run_single_matmul(
            "test.weight",
            input_int8,
            (packed_weight, packed_scales),
            {"act": float(input_scale), "test_node": float(target_scale)},
            {"test.bias": bias_pad},
            requant_pc_scale_table=pc_scales,
        )

        accum = input_int8.astype(np.int32) @ q_pad.T.astype(np.int32)
        accum += bias_pad.reshape(1, -1)
        expected = np.clip(
            np.round(accum.astype(np.float32) * pc_scales.astype(np.float32).reshape(1, -1)),
            -128,
            127,
        ).astype(np.int8)

        np.testing.assert_array_equal(result, expected)
