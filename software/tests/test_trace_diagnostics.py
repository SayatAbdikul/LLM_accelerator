"""Tests for benchmark summary and simulator trace diagnostics."""
import sys

import numpy as np
import pytest

from compare_golden import (
    main,
    quantization_diagnostics,
    replay_attention_head_variants,
    replay_block_downstream_variants,
    replay_mlp_block_variants,
    summarize_late_attention_replay,
    summarize_late_mlp_replay,
    summarize_early_attention_replay,
    summarize_results,
    tensor_error_metrics,
)
from taccel.assembler.assembler import Assembler
from taccel.golden_model import Simulator
from taccel.isa.opcodes import BUF_ABUF


class TestTraceDiagnostics:
    def test_summarize_results_includes_tail_metrics(self):
        results = [
            {"top1_match": True, "top5_overlap": 1.0, "cosine_sim": 0.95, "cycles": 10},
            {"top1_match": False, "top5_overlap": 0.4, "cosine_sim": 0.70, "cycles": 20},
            {"top1_match": True, "top5_overlap": 0.8, "cosine_sim": 0.85, "cycles": 30},
        ]

        summary = summarize_results(results)

        assert summary["n_images"] == 3
        assert summary["top1_agreement"] == 2 / 3
        assert summary["top5_overlap_avg"] == (1.0 + 0.4 + 0.8) / 3
        assert np.isclose(summary["cosine_sim_avg"], np.mean([0.95, 0.70, 0.85]))
        assert np.isclose(summary["cosine_sim_min"], 0.70)
        assert np.isclose(summary["cosine_sim_p10"], np.percentile([0.95, 0.70, 0.85], 10))
        assert summary["avg_cycles"] == 20.0

    def test_simulator_trace_manifest_captures_and_dequantizes(self):
        prog = Assembler().assemble(
            "CONFIG_TILE M=1, N=1, K=1\n"
            "SET_SCALE S0, imm=0x3800\n"  # 0.5 in FP16
            "REQUANT src1=ACCUM[0], src2=ABUF[0], dst=ABUF[0], sreg=0, flags=0\n"
            "HALT"
        )
        prog.trace_manifest = {
            2: [{
                "node_name": "trace_node",
                "buf_id": BUF_ABUF,
                "offset_units": 0,
                "mem_rows": 16,
                "mem_cols": 16,
                "logical_rows": 16,
                "logical_cols": 16,
                "full_rows": 16,
                "full_cols": 16,
                "row_start": 0,
                "dtype": "int8",
                "scale": 0.5,
                "when": "after",
            }],
        }

        sim = Simulator()
        sim.load_program(prog)
        sim.enable_trace({"trace_node"})
        sim.state.accum[:256] = 4
        sim.run()

        payload = sim.get_trace_payload()
        traced = payload["tensors"]["trace_node"]
        assert traced.shape == (16, 16)
        assert np.all(traced == 1.0)
        assert payload["stats"]["trace_node"]["saturation_rate"] == 0.0
        assert payload["meta"]["trace_node"]["dtype"] == "int8"

    def test_quantization_diagnostics_reports_zero_and_cosine(self):
        tensor = np.array([[0.0, 0.24], [0.49, 4.1]], dtype=np.float32)
        diag = quantization_diagnostics(tensor, 0.5)

        assert 0.0 <= diag["qdq_cosine_sim"] <= 1.0
        assert diag["qdq_zero_fraction"] > 0.0
        assert diag["qdq_saturation_rate"] == 0.0

    def test_tensor_error_metrics_reports_exact_match(self):
        tensor = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)

        metrics = tensor_error_metrics(tensor, tensor.copy())

        assert np.isclose(metrics["cosine_sim"], 1.0)
        assert metrics["max_abs_error"] == 0.0
        assert metrics["mean_abs_error"] == 0.0
        assert metrics["mse"] == 0.0

    def test_replay_attention_head_variants_separates_input_and_output_qdq(self):
        softmax = np.array([[0.75, 0.25], [0.10, 0.90]], dtype=np.float32)
        value = np.array([[1.0, -0.4], [0.2, 0.8]], dtype=np.float32)
        target = softmax @ value

        reports, tensors = replay_attention_head_variants(
            softmax,
            value,
            target,
            softmax_scale=0.05,
            value_scale=0.50,
            attn_v_scale=0.25,
        )

        assert reports["fp32_fp32"]["raw_metrics"]["cosine_sim"] == 1.0
        assert reports["fp32_fp32"]["attn_v_qdq_metrics"]["cosine_sim"] <= 1.0
        assert tensors["qdq_value"]["attn_v_qdq"].shape == target.shape
        assert reports["qdq_value"]["attn_v_qdq_metrics"]["max_abs_error"] >= 0.0

    def test_replay_block_downstream_variants_propagates_concat_and_out_proj(self):
        variant_heads = {
            "fp32_fp32": [
                np.array([[1.0, 2.0]], dtype=np.float32),
                np.array([[3.0, 4.0]], dtype=np.float32),
            ],
            "qdq_softmax": [
                np.array([[0.5, 2.0]], dtype=np.float32),
                np.array([[3.0, 3.5]], dtype=np.float32),
            ],
        }
        concat_target = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out_proj_target = concat_target.copy()
        weight = np.eye(4, dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)

        reports = replay_block_downstream_variants(
            variant_heads,
            concat_target,
            out_proj_target,
            weight,
            bias,
            out_proj_scale=0.5,
        )

        assert np.isclose(reports["fp32_fp32"]["concat_metrics"]["cosine_sim"], 1.0)
        assert np.isclose(reports["fp32_fp32"]["out_proj_raw_metrics"]["cosine_sim"], 1.0)
        assert reports["qdq_softmax"]["concat_metrics"]["max_abs_error"] > 0.0

    def test_summarize_early_attention_replay_aggregates_variants(self):
        summaries = summarize_early_attention_replay([
            {
                "blocks": [{
                    "block_idx": 0,
                    "golden_block_metrics": {"out_proj_metrics": {"cosine_sim": 0.8}},
                    "worst_head_idx": 2,
                    "heads": [{
                        "head_idx": 2,
                        "golden_attn_v_metrics": {"cosine_sim": 0.9},
                        "variants": {
                            "qdq_softmax": {"attn_v_qdq_metrics": {"cosine_sim": 0.7}},
                            "qdq_value": {"attn_v_qdq_metrics": {"cosine_sim": 0.6}},
                            "qdq_softmax_value": {"attn_v_qdq_metrics": {"cosine_sim": 0.5}},
                            "fp32_fp32": {"attn_v_qdq_metrics": {"cosine_sim": 1.0}},
                        },
                    }],
                    "block_variants": {
                        "qdq_softmax": {"out_proj_qdq_metrics": {"cosine_sim": 0.7}},
                        "qdq_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.6}},
                        "qdq_softmax_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.5}},
                        "fp32_fp32": {"out_proj_qdq_metrics": {"cosine_sim": 1.0}},
                    },
                }],
            }
        ])

        assert summaries[0]["block_idx"] == 0
        assert summaries[0]["mean_golden_out_proj_cosine"] == 0.8
        assert summaries[0]["variant_mean_out_proj_qdq_cosine"]["qdq_softmax"] == 0.7
        assert summaries[0]["worst_head_counts"]["head2"] == 1

    def test_replay_mlp_block_variants_propagates_gelu_and_residual(self):
        fc1 = np.array([[0.25, -0.75]], dtype=np.float32)
        gelu_target = np.array([[0.15, -0.17]], dtype=np.float32)
        fc2_target = gelu_target.copy()
        residual1 = np.array([[0.05, 0.10]], dtype=np.float32)
        residual2 = residual1 + fc2_target
        weight = np.eye(2, dtype=np.float32)
        bias = np.zeros(2, dtype=np.float32)

        reports = replay_mlp_block_variants(
            fc1,
            gelu_target,
            fc2_target,
            residual1,
            residual2,
            fc1_scale=0.25,
            gelu_scale=0.10,
            fc2_scale=0.10,
            fc2_weight=weight,
            fc2_bias=bias,
        )

        assert np.isclose(reports["fp32_fp32"]["gelu_metrics"]["cosine_sim"], 1.0)
        assert reports["fp32_fp32"]["residual2_metrics"]["cosine_sim"] > 0.99
        assert reports["qdq_fc1_gelu_out"]["gelu_saturation_rate"] >= 0.0

    def test_summarize_late_attention_replay_aggregates_per_head(self):
        summaries = summarize_late_attention_replay([
            {
                "blocks": [{
                    "block_idx": 11,
                    "golden_block_metrics": {
                        "concat_metrics": {"cosine_sim": 0.93},
                        "out_proj_metrics": {"cosine_sim": 0.91},
                    },
                    "block_variants": {
                        "qdq_softmax": {"out_proj_qdq_metrics": {"cosine_sim": 0.82}},
                        "qdq_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.88}},
                        "qdq_softmax_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.80}},
                        "fp32_fp32": {"out_proj_qdq_metrics": {"cosine_sim": 0.95}},
                    },
                    "heads": [
                        {
                            "head_idx": 1,
                            "golden_attn_v_metrics": {"cosine_sim": 0.72},
                            "variants": {
                                "qdq_softmax": {"attn_v_qdq_metrics": {"cosine_sim": 0.60}},
                                "qdq_value": {"attn_v_qdq_metrics": {"cosine_sim": 0.70}},
                                "qdq_softmax_value": {"attn_v_qdq_metrics": {"cosine_sim": 0.58}},
                                "fp32_fp32": {"attn_v_qdq_metrics": {"cosine_sim": 0.90}},
                            },
                            "isolated_block_variants": {
                                "qdq_softmax": {"out_proj_qdq_metrics": {"cosine_sim": 0.79}},
                                "qdq_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.86}},
                                "qdq_softmax_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.77}},
                                "fp32_fp32": {"out_proj_qdq_metrics": {"cosine_sim": 0.94}},
                            },
                        },
                        {
                            "head_idx": 2,
                            "golden_attn_v_metrics": {"cosine_sim": 0.81},
                            "variants": {
                                "qdq_softmax": {"attn_v_qdq_metrics": {"cosine_sim": 0.68}},
                                "qdq_value": {"attn_v_qdq_metrics": {"cosine_sim": 0.76}},
                                "qdq_softmax_value": {"attn_v_qdq_metrics": {"cosine_sim": 0.65}},
                                "fp32_fp32": {"attn_v_qdq_metrics": {"cosine_sim": 0.92}},
                            },
                            "isolated_block_variants": {
                                "qdq_softmax": {"out_proj_qdq_metrics": {"cosine_sim": 0.83}},
                                "qdq_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.87}},
                                "qdq_softmax_value": {"out_proj_qdq_metrics": {"cosine_sim": 0.81}},
                                "fp32_fp32": {"out_proj_qdq_metrics": {"cosine_sim": 0.95}},
                            },
                        },
                    ],
                }],
            }
        ])

        assert summaries[0]["block_idx"] == 11
        assert summaries[0]["mean_golden_concat_cosine"] == 0.93
        assert summaries[0]["variant_mean_block_out_proj_qdq_cosine"]["qdq_softmax"] == 0.82
        assert summaries[0]["per_head"][0]["head_idx"] == 1
        assert summaries[0]["per_head"][0]["variant_mean_isolated_out_proj_qdq_cosine"]["qdq_value"] == 0.86

    def test_summarize_late_mlp_replay_aggregates_variants(self):
        summaries = summarize_late_mlp_replay([
            {
                "blocks": [{
                    "block_idx": 10,
                    "golden_metrics": {
                        "gelu_metrics": {"cosine_sim": 0.84},
                        "fc2_metrics": {"cosine_sim": 0.83},
                        "residual2_metrics": {"cosine_sim": 0.82},
                    },
                    "variants": {
                        "qdq_fc1": {
                            "gelu_metrics": {"cosine_sim": 0.74},
                            "fc2_qdq_metrics": {"cosine_sim": 0.73},
                            "residual2_metrics": {"cosine_sim": 0.72},
                        },
                        "qdq_gelu_out": {
                            "gelu_metrics": {"cosine_sim": 0.71},
                            "fc2_qdq_metrics": {"cosine_sim": 0.70},
                            "residual2_metrics": {"cosine_sim": 0.69},
                        },
                        "qdq_fc1_gelu_out": {
                            "gelu_metrics": {"cosine_sim": 0.68},
                            "fc2_qdq_metrics": {"cosine_sim": 0.67},
                            "residual2_metrics": {"cosine_sim": 0.66},
                        },
                        "fp32_fp32": {
                            "gelu_metrics": {"cosine_sim": 0.90},
                            "fc2_qdq_metrics": {"cosine_sim": 0.89},
                            "residual2_metrics": {"cosine_sim": 0.88},
                        },
                    },
                }],
            }
        ])

        assert summaries[0]["block_idx"] == 10
        assert summaries[0]["mean_golden_residual2_cosine"] == 0.82
        assert summaries[0]["variant_mean_fc2_cosine"]["qdq_gelu_out"] == 0.70
        assert summaries[0]["variant_mean_residual2_cosine"]["qdq_fc1_gelu_out"] == 0.66

    def test_replay_late_attn_requires_trace_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["compare_golden.py", "--replay-late-attn"])

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 2

    def test_replay_late_mlp_requires_trace_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["compare_golden.py", "--replay-late-mlp"])

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 2
