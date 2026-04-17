"""Tests for offline accuracy diagnostics helpers."""
import numpy as np

from diagnose_accuracy import (
    compare_attention_runtime_path,
    summarize_block_impact,
    summarize_late_attention_path_delta,
    summarize_trace_variant_delta,
)


class TestTraceVariantDelta:
    def test_summarize_trace_variant_delta_filters_blocks_and_aggregates_common_images(self):
        baseline = {
            "per_image": [
                {
                    "img_id": 101,
                    "first_major_drop": {"node": "block10_head0_softmax"},
                    "node_metrics": [
                        {
                            "node": "block10_head0_softmax",
                            "cosine_sim": 0.90,
                            "qdq_cosine_sim": 0.99,
                            "max_abs_error": 0.10,
                        },
                        {
                            "node": "block10_head0_attn_v",
                            "cosine_sim": 0.85,
                            "qdq_cosine_sim": 0.98,
                            "max_abs_error": 0.20,
                        },
                        {
                            "node": "classifier",
                            "cosine_sim": 0.70,
                            "qdq_cosine_sim": 0.0,
                            "max_abs_error": 1.0,
                        },
                    ],
                },
                {
                    "img_id": 202,
                    "first_major_drop": {"node": "block10_head0_softmax"},
                    "node_metrics": [
                        {
                            "node": "block10_head0_softmax",
                            "cosine_sim": 0.88,
                            "qdq_cosine_sim": 0.98,
                            "max_abs_error": 0.12,
                        },
                        {
                            "node": "classifier",
                            "cosine_sim": 0.72,
                            "qdq_cosine_sim": 0.0,
                            "max_abs_error": 1.2,
                        },
                    ],
                },
            ],
        }
        variant = {
            "per_image": [
                {
                    "img_id": 101,
                    "first_major_drop": {"node": "block10_head0_softmax"},
                    "node_metrics": [
                        {
                            "node": "block10_head0_softmax",
                            "cosine_sim": 0.84,
                            "qdq_cosine_sim": 0.98,
                            "max_abs_error": 0.18,
                        },
                        {
                            "node": "block10_head0_attn_v",
                            "cosine_sim": 0.80,
                            "qdq_cosine_sim": 0.98,
                            "max_abs_error": 0.28,
                        },
                        {
                            "node": "classifier",
                            "cosine_sim": 0.60,
                            "qdq_cosine_sim": 0.0,
                            "max_abs_error": 1.5,
                        },
                    ],
                },
                {
                    "img_id": 202,
                    "first_major_drop": {"node": "block10_head0_softmax"},
                    "node_metrics": [
                        {
                            "node": "block10_head0_softmax",
                            "cosine_sim": 0.82,
                            "qdq_cosine_sim": 0.97,
                            "max_abs_error": 0.22,
                        },
                        {
                            "node": "classifier",
                            "cosine_sim": 0.55,
                            "qdq_cosine_sim": 0.0,
                            "max_abs_error": 1.8,
                        },
                    ],
                },
                {
                    "img_id": 303,
                    "first_major_drop": {"node": "block11_head1_softmax"},
                    "node_metrics": [
                        {
                            "node": "block11_head1_softmax",
                            "cosine_sim": 0.40,
                            "qdq_cosine_sim": 0.95,
                            "max_abs_error": 0.50,
                        },
                    ],
                },
            ],
        }

        report = summarize_trace_variant_delta(
            baseline,
            variant,
            blocks={10},
        )

        assert report["common_traced_image_ids"] == [101, 202]
        assert report["blocks"] == [10]
        assert report["node_report"][0]["node"] == "classifier"
        assert report["node_report"][0]["delta_cosine"] < 0.0
        assert any(row["stage"] == "softmax" for row in report["stage_summary"])
        assert report["variant_first_drop_counts"]["block10_head0_softmax"] == 2


class TestLateAttentionPathDelta:
    def test_summarize_late_attention_path_delta_groups_by_block_and_head(self):
        baseline = {
            "per_image": [
                {
                    "img_id": 101,
                    "node_metrics": [
                        {"node": "block10_head1_value", "cosine_sim": 0.99, "qdq_cosine_sim": 0.999, "max_abs_error": 0.01},
                        {"node": "block10_head1_qkt", "cosine_sim": 0.95, "qdq_cosine_sim": 0.998, "max_abs_error": 0.10},
                        {"node": "block10_head1_softmax", "cosine_sim": 0.90, "qdq_cosine_sim": 0.997, "max_abs_error": 0.05},
                        {"node": "block10_head1_attn_v", "cosine_sim": 0.85, "qdq_cosine_sim": 0.996, "max_abs_error": 0.20},
                    ],
                },
                {
                    "img_id": 202,
                    "node_metrics": [
                        {"node": "block10_head1_value", "cosine_sim": 0.98, "qdq_cosine_sim": 0.999, "max_abs_error": 0.02},
                        {"node": "block10_head1_qkt", "cosine_sim": 0.94, "qdq_cosine_sim": 0.998, "max_abs_error": 0.11},
                        {"node": "block10_head1_softmax", "cosine_sim": 0.89, "qdq_cosine_sim": 0.997, "max_abs_error": 0.06},
                        {"node": "block10_head1_attn_v", "cosine_sim": 0.84, "qdq_cosine_sim": 0.996, "max_abs_error": 0.19},
                    ],
                },
            ],
        }
        variant = {
            "per_image": [
                {
                    "img_id": 101,
                    "node_metrics": [
                        {"node": "block10_head1_value", "cosine_sim": 0.98, "qdq_cosine_sim": 0.999, "max_abs_error": 0.02},
                        {"node": "block10_head1_qkt", "cosine_sim": 0.92, "qdq_cosine_sim": 0.998, "max_abs_error": 0.14},
                        {"node": "block10_head1_softmax", "cosine_sim": 0.82, "qdq_cosine_sim": 0.996, "max_abs_error": 0.08},
                        {"node": "block10_head1_attn_v", "cosine_sim": 0.74, "qdq_cosine_sim": 0.996, "max_abs_error": 0.28},
                    ],
                },
                {
                    "img_id": 202,
                    "node_metrics": [
                        {"node": "block10_head1_value", "cosine_sim": 0.97, "qdq_cosine_sim": 0.999, "max_abs_error": 0.03},
                        {"node": "block10_head1_qkt", "cosine_sim": 0.91, "qdq_cosine_sim": 0.997, "max_abs_error": 0.16},
                        {"node": "block10_head1_softmax", "cosine_sim": 0.80, "qdq_cosine_sim": 0.995, "max_abs_error": 0.09},
                        {"node": "block10_head1_attn_v", "cosine_sim": 0.70, "qdq_cosine_sim": 0.995, "max_abs_error": 0.30},
                    ],
                },
            ],
        }

        report = summarize_late_attention_path_delta(
            baseline,
            variant,
            blocks={10},
        )

        assert report["common_traced_image_ids"] == [101, 202]
        assert report["blocks"] == [10]
        assert report["stage_summary"][0]["stage"] == "value"
        head = report["per_head_report"][0]
        assert head["block_idx"] == 10
        assert head["head_idx"] == 1
        assert head["stages"]["attn_v"]["delta_cosine"] < head["stages"]["value"]["delta_cosine"]


class TestRuntimeAttentionPath:
    def test_compare_attention_runtime_path_separates_softmax_and_value_effects(self):
        fp32_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        fp32_qkt = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        fp32_softmax = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        fp32_attn_v = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)

        baseline = compare_attention_runtime_path(
            fp32_value=fp32_value,
            fp32_qkt=fp32_qkt,
            fp32_softmax=fp32_softmax,
            fp32_attn_v=fp32_attn_v,
            baseline_value=fp32_value,
            baseline_qkt=fp32_qkt,
            baseline_softmax=fp32_softmax,
            baseline_attn_v=fp32_attn_v,
            variant_value=np.array([[0.8, 0.0], [0.0, 1.2]], dtype=np.float32),
            variant_qkt=np.array([[1.9, 0.1], [0.1, 1.9]], dtype=np.float32),
            variant_softmax=np.array([[0.82, 0.18], [0.08, 0.92]], dtype=np.float32),
            variant_attn_v=np.array([[0.66, 0.22], [0.10, 1.10]], dtype=np.float32),
        )

        assert np.isclose(baseline["baseline"]["attn_v_metrics"]["cosine_sim"], 1.0)
        assert baseline["variant"]["attn_v_metrics"]["cosine_sim"] < 1.0
        assert baseline["variant_replay_isolation"]["softmax_only_to_fp32_metrics"]["cosine_sim"] > 0.99
        assert baseline["variant_replay_isolation"]["value_only_to_fp32_metrics"]["cosine_sim"] < 0.99


class TestBlockImpactReport:
    def test_summarize_block_impact_computes_per_image_deltas_and_shares(self):
        compare_json = {
            "per_image": [
                {
                    "img_id": 101,
                    "top1_match": False,
                    "top5_overlap": 0.4,
                    "cosine_sim": 0.70,
                    "fp32_top5": [10],
                    "golden_top5": [20],
                },
            ],
            "trace_report": {
                "per_image": [
                    {
                        "img_id": 101,
                        "first_major_drop": {"node": "block0_head0_softmax"},
                        "node_metrics": [
                            {"node": "pos_embed_add", "cosine_sim": 0.99, "delta_from_prev": None},
                            {"node": "block0_head0_softmax", "cosine_sim": 0.90, "delta_from_prev": -0.09},
                            {"node": "block0_residual1", "cosine_sim": 0.95, "delta_from_prev": +0.05},
                            {"node": "block0_gelu", "cosine_sim": 0.80, "delta_from_prev": -0.15},
                            {"node": "block0_residual2", "cosine_sim": 0.85, "delta_from_prev": +0.05},
                            {"node": "block1_head0_softmax", "cosine_sim": 0.75, "delta_from_prev": -0.10},
                            {"node": "block1_residual1", "cosine_sim": 0.78, "delta_from_prev": +0.03},
                            {"node": "block1_residual2", "cosine_sim": 0.70, "delta_from_prev": -0.08},
                            {"node": "classifier", "cosine_sim": 0.70, "delta_from_prev": 0.0},
                        ],
                    },
                ],
            },
        }

        report = summarize_block_impact(compare_json)

        assert report["n_images"] == 1
        assert report["blocks_ranked_by_worsening"][0]["block_idx"] == 1
        image = report["per_image"][0]
        assert image["img_id"] == 101
        assert image["first_major_drop_block"] == 0
        assert image["blocks"][0]["total_delta"] < 0.0
        assert image["blocks"][1]["worsening_share"] > image["blocks"][0]["worsening_share"]
        assert image["blocks"][0]["worst_node"] == "block0_gelu"
        assert image["blocks"][1]["worst_delta_node"] == "block1_head0_softmax"
