"""Tests for compare_golden calibration helpers."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from taccel.quantizer.bias_correction import resolve_bias_correction_targets

from compare_golden import (
    CATS_DOGS_SAMPLE_IDS,
    CURRENT_BEST_SMOOTHQUANT_BLOCKS,
    DIAGNOSTIC_PRESETS,
    IMAGENET_CLASS0_SAMPLE_IDS,
    apply_diagnostic_preset,
    calibrate_residual1_block_scales,
    discover_local_flat_samples,
    discover_cats_dogs_samples,
    infer_cats_dogs_label,
    NUM_HEADS,
    QKV_PROJECTIONS,
    build_calibration_scales,
    build_requant_pc_qkv_selection,
    default_attn_v_block_scales,
    explicit_cli_dest_overrides,
    fold_pos_embed_int8,
    load_local_images,
    local_frozen_image_path,
    parse_csv_int_list,
    parse_activation_percentile_overrides,
    parse_csv_token_list,
    parse_qkv_projection_set,
    parse_qkv_triplet_set,
    preset_compile_kwargs,
    resolve_activation_percentile_targets,
    resolve_explicit_sample_ids,
    select_trace_image_ids,
    select_best_attn_v_scale,
    select_best_attn_v_scale_downstream,
    select_best_final_logit_scale,
    select_best_gelu_output_scale,
    select_best_softmax_prob,
    select_best_softmax_prob_downstream,
    select_best_value_scale,
)


class TestCalibrationScaleMapping:
    @staticmethod
    def _make_seeded_block_samples(seed: int):
        rng = np.random.default_rng(seed)
        sample_heads = []
        for h in range(NUM_HEADS):
            logits = rng.normal(loc=0.0, scale=1.5, size=(3, 3)).astype(np.float32)
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            softmax = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            value = rng.normal(scale=1.0 + 0.6 * h, size=(3, 2)).astype(np.float32)
            sample_heads.append({
                "softmax": softmax,
                "value": value,
                "attn_v": (softmax @ value).astype(np.float32),
            })

        concat_width = sample_heads[0]["attn_v"].shape[1] * NUM_HEADS
        out_proj_weight = rng.normal(scale=0.7, size=(concat_width, concat_width)).astype(np.float32)
        out_proj_bias = np.zeros(concat_width, dtype=np.float32)

        block_samples = []
        rng = np.random.default_rng(seed)
        for _ in range(4):
            heads = []
            for h in range(NUM_HEADS):
                logits = rng.normal(loc=0.0, scale=1.5, size=(3, 3)).astype(np.float32)
                exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
                softmax = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                value = rng.normal(scale=1.0 + 0.6 * h, size=(3, 2)).astype(np.float32)
                heads.append({
                    "softmax": softmax,
                    "value": value,
                    "attn_v": (softmax @ value).astype(np.float32),
                })
            concat = np.concatenate([head["attn_v"] for head in heads], axis=-1).astype(np.float32)
            out_proj = (concat @ out_proj_weight.T + out_proj_bias).astype(np.float32)
            block_samples.append({
                "heads": heads,
                "concat": concat,
                "out_proj": out_proj,
            })

        return block_samples, out_proj_weight, out_proj_bias

    def test_local_frozen_image_path_formats_cache_filename(self, tmp_path):
        path = local_frozen_image_path(139, image_root=str(tmp_path))

        assert path.endswith("/000000000139.jpg")

    def test_load_local_images_requires_local_cache_and_preserves_order(self, tmp_path):
        img0 = Image.fromarray(np.full((4, 4, 3), 32, dtype=np.uint8), mode="RGB")
        img1 = Image.fromarray(np.full((4, 4, 3), 96, dtype=np.uint8), mode="RGB")
        img0.save(local_frozen_image_path(285, image_root=str(tmp_path)), format="JPEG")
        img1.save(local_frozen_image_path(139, image_root=str(tmp_path)), format="JPEG")

        loaded = load_local_images([139, 285], "test", image_root=str(tmp_path))

        assert [img_id for img_id, _ in loaded] == [139, 285]
        assert loaded[0][1].mode == "RGB"
        assert loaded[1][1].mode == "RGB"

    def test_load_local_images_raises_for_missing_ids(self, tmp_path):
        with pytest.raises(RuntimeError, match="Missing local frozen benchmark images"):
            load_local_images([139], "test", image_root=str(tmp_path))

    def test_parse_qkv_projection_set_accepts_all_and_valid_names(self):
        assert parse_qkv_projection_set("") is None
        assert parse_qkv_projection_set("all") is None
        assert parse_qkv_projection_set("query,key") == {"query", "key"}

    def test_parse_csv_int_list_preserves_order(self):
        assert parse_csv_int_list("") == []
        assert parse_csv_int_list("5037,2685,776") == [5037, 2685, 776]

    def test_parse_csv_token_list_preserves_order(self):
        assert parse_csv_token_list("") == []
        assert parse_csv_token_list("cat60,dog38,139") == ["cat60", "dog38", "139"]

    def test_parse_qkv_triplet_set_parses_block_projection_head_specs(self):
        assert parse_qkv_triplet_set("11:value:2,0:query:1") == {
            (11, "value", 2),
            (0, "query", 1),
        }

    def test_build_requant_pc_qkv_selection_returns_none_for_default_all(self):
        assert build_requant_pc_qkv_selection() is None

    def test_diagnostic_preset_baseline_uses_local_frozen_split(self):
        preset = DIAGNOSTIC_PRESETS["baseline_frozen_local"]

        assert preset["benchmark"]["benchmark_dataset"] == "frozen_coco"
        assert preset["benchmark"]["benchmark_image_source"] == "local"
        assert len(preset["benchmark"]["eval_image_ids"]) == 20
        assert len(preset["benchmark"]["calibration_image_ids"]) == 20
        assert preset["compile_args"]["smoothquant_targets"] == "off"

    def test_diagnostic_preset_cats_dogs_uses_all_local_samples(self):
        preset = DIAGNOSTIC_PRESETS["cats_dogs_local_all"]

        assert preset["benchmark"]["benchmark_dataset"] == "cats_dogs_local"
        assert preset["benchmark"]["benchmark_image_source"] == "local"
        assert len(preset["benchmark"]["eval_image_ids"]) == 200
        assert len(preset["benchmark"]["calibration_image_ids"]) == 200
        assert preset["benchmark"]["eval_image_ids"][0] == "cat1"
        assert preset["benchmark"]["eval_image_ids"][-1] == "dog100"

    def test_diagnostic_preset_imagenet_class0_uses_all_local_samples(self):
        preset = DIAGNOSTIC_PRESETS["imagenet_class0_local_all"]

        assert preset["benchmark"]["benchmark_dataset"] == "local_flat"
        assert preset["benchmark"]["benchmark_image_source"] == "local"
        assert len(preset["benchmark"]["eval_image_ids"]) == len(IMAGENET_CLASS0_SAMPLE_IDS)
        assert len(preset["benchmark"]["calibration_image_ids"]) == len(IMAGENET_CLASS0_SAMPLE_IDS)
        assert preset["benchmark"]["eval_image_ids"][0] == "000_00000"
        assert preset["benchmark"]["eval_image_ids"][-1] == "000_00099"

    def test_diagnostic_preset_imagenet_class0_current_best_enables_smoothquant(self):
        preset = DIAGNOSTIC_PRESETS["imagenet_class0_current_best_sq_ln2_fc1_b0_8_10"]

        assert preset["benchmark"]["benchmark_dataset"] == "local_flat"
        assert preset["compile_args"]["smoothquant_targets"] == "ln2_fc1"
        assert preset["compile_args"]["smoothquant_alpha"] == 0.5
        assert preset["compile_args"]["smoothquant_blocks"] == ",".join(
            str(v) for v in CURRENT_BEST_SMOOTHQUANT_BLOCKS
        )

    def test_diagnostic_preset_imagenet_class0_ptq4vit_base_matches_current_step4_stack(self):
        preset = DIAGNOSTIC_PRESETS["imagenet_class0_ptq4vit_base"]

        assert preset["benchmark"]["benchmark_dataset"] == "local_flat"
        assert preset["compile_args"]["requant_pc_fc1"] is True
        assert preset["compile_args"]["requant_pc_fc1_blocks"] == "8,9"
        assert preset["compile_args"]["output_aware_clipping_fc1_blocks"] == "9"
        assert preset["compile_args"]["adaround_fc1_blocks"] == "9"
        assert preset["compile_args"]["act_percentile_nodes"] == "final_ln:99.8,block9_ln2:99.0"
        assert preset["compile_args"]["twin_uniform_mode"] == "off"
        assert preset["compile_args"]["requant_pc_fc2"] is False
        assert preset["compile_args"]["output_aware_clipping_fc2_blocks"] == ""

    def test_diagnostic_preset_imagenet_class0_current_best_ptq_alias_matches_base(self):
        base = DIAGNOSTIC_PRESETS["imagenet_class0_ptq4vit_base"]
        alias = DIAGNOSTIC_PRESETS["imagenet_class0_current_best_ptq"]

        assert alias["benchmark"] == base["benchmark"]
        assert alias["compile_args"] == base["compile_args"]
        assert alias["trace"] == base["trace"]

    def test_apply_diagnostic_preset_sets_current_best_smoothquant_flags(self):
        args = SimpleNamespace(
            diagnostic_preset="current_best_sq_ln2_fc1_b0_8_10",
            benchmark_dataset="cats_dogs_local",
            benchmark_image_source="download",
            local_benchmark_image_dir="/tmp/nowhere",
            max_images=3,
            calibration_images=3,
            softmax_calibration="search",
            softmax_percentile=90.0,
            softmax_min_prob=1e-3,
            softmax_max_prob=0.5,
            softmax_search_heads="11:1",
            softmax_search_objective="tail_attn_v",
            bias_correction=False,
            bias_correction_layers="classifier",
            act_percentile_nodes="",
            output_aware_clipping_fc1_blocks="",
            output_aware_clipping_fc2_blocks="",
            output_aware_clipping_classifier=False,
            output_aware_clipping_candidates=25,
            output_aware_clipping_alpha_min=0.5,
            adaround_fc1_blocks="",
            adaround_fc2_blocks="",
            attn_v_calibration="max",
            attn_v_percentile=90.0,
            attn_v_safety_margin=1.0,
            attn_v_search_blocks="11",
            attn_v_search_objective="tail_out_proj",
            gelu_from_accum=True,
            per_head_qkv_calibration=True,
            value_head_calibration="search",
            gelu_output_calibration="search",
            gelu_search_blocks="11",
            gelu_search_objective="downstream_residual2",
            hessian_calibration_images=0,
            hessian_target_nodes="",
            twin_uniform_softmax_blocks="",
            twin_uniform_gelu_blocks="",
            twin_uniform_mode="off",
            twin_uniform_disable_hessian=False,
            smoothquant_targets="off",
            smoothquant_alpha=0.4,
            smoothquant_blocks="",
            requant_pc_qkv=True,
            requant_pc_qkv_blocks="11",
            requant_pc_qkv_heads="1",
            requant_pc_qkv_projections="key",
            requant_pc_qkv_exclude="11:key:1",
            requant_pc_fc1=False,
            requant_pc_fc1_blocks="",
            requant_pc_fc2=False,
            requant_pc_fc2_blocks="",
            requant_pc_out_proj=True,
            requant_pc_out_proj_blocks="",
            fold_cls_pos_embed=True,
            trace_worst_k=0,
            trace_image_ids="139",
        )

        preset = apply_diagnostic_preset(args)

        assert preset["description"]
        assert args.benchmark_dataset == "frozen_coco"
        assert args.benchmark_image_source == "local"
        assert args.smoothquant_targets == "ln2_fc1"
        assert args.smoothquant_alpha == 0.5
        assert args.smoothquant_blocks == ",".join(str(v) for v in CURRENT_BEST_SMOOTHQUANT_BLOCKS)
        assert args.trace_worst_k == 5
        assert args.requant_pc_qkv is False

    def test_apply_diagnostic_preset_preserves_explicit_cli_overrides(self):
        args = SimpleNamespace(
            diagnostic_preset="current_best_sq_ln2_fc1_b0_8_10",
            benchmark_dataset="cats_dogs_local",
            benchmark_image_source="download",
            local_benchmark_image_dir="/tmp/nowhere",
            max_images=3,
            calibration_images=3,
            softmax_calibration="max",
            softmax_percentile=99.0,
            softmax_min_prob=1e-4,
            softmax_max_prob=1.0,
            softmax_search_heads="",
            softmax_search_objective="local_prob",
            bias_correction=True,
            bias_correction_layers="late_fc2",
            act_percentile_nodes="final_ln:99.9",
            output_aware_clipping_fc1_blocks="8,9",
            output_aware_clipping_fc2_blocks="10",
            output_aware_clipping_classifier=True,
            output_aware_clipping_candidates=31,
            output_aware_clipping_alpha_min=0.4,
            adaround_fc1_blocks="9",
            adaround_fc2_blocks="10",
            attn_v_calibration="off",
            attn_v_percentile=99.0,
            attn_v_safety_margin=1.10,
            attn_v_search_blocks="",
            attn_v_search_objective="local_attn_v",
            gelu_from_accum=True,
            gelu_from_accum_blocks="2,3,4",
            fused_softmax_attnv=True,
            fused_softmax_attnv_blocks="11",
            per_head_qkv_calibration=False,
            value_head_calibration="off",
            gelu_output_calibration="off",
            gelu_search_blocks="9,10,11",
            gelu_search_objective="downstream_residual2",
            hessian_calibration_images=0,
            hessian_target_nodes="",
            twin_uniform_softmax_blocks="",
            twin_uniform_gelu_blocks="",
            twin_uniform_mode="off",
            twin_uniform_disable_hessian=False,
            smoothquant_targets="off",
            smoothquant_alpha=0.4,
            smoothquant_blocks="",
            requant_pc_qkv=False,
            requant_pc_qkv_blocks="",
            requant_pc_qkv_heads="",
            requant_pc_qkv_projections="all",
            requant_pc_qkv_exclude="",
            requant_pc_fc1=True,
            requant_pc_fc1_blocks="2,3,4",
            requant_pc_fc2=True,
            requant_pc_fc2_blocks="10",
            requant_pc_out_proj=False,
            requant_pc_out_proj_blocks="",
            fold_cls_pos_embed=False,
            trace_worst_k=0,
            trace_image_ids="139",
        )

        apply_diagnostic_preset(
            args,
            explicit_overrides={"benchmark_dataset", "gelu_from_accum", "gelu_from_accum_blocks", "trace_image_ids"},
        )

        assert args.benchmark_dataset == "cats_dogs_local"
        assert args.gelu_from_accum is True
        assert args.gelu_from_accum_blocks == "2,3,4"
        assert args.fused_softmax_attnv is False
        assert args.fused_softmax_attnv_blocks == ""
        assert args.requant_pc_fc1 is False
        assert args.requant_pc_fc1_blocks == ""
        assert args.trace_image_ids == "139"
        assert args.smoothquant_targets == "ln2_fc1"

    def test_explicit_cli_dest_overrides_parses_long_options(self):
        overrides = explicit_cli_dest_overrides(
            [
                "--diagnostic-preset", "current_best_sq_ln2_fc1_b0_8_10",
                "--benchmark-dataset", "cats_dogs_local",
                "--gelu-from-accum",
                "--gelu-from-accum-blocks=2,3,4",
                "--final-logit-calibration", "search",
                "--bias-correction",
                "--bias-correction-layers", "classifier,late_fc2",
                "--act-percentile-nodes", "final_ln:99.9,block9_ln2:99.5",
                "--output-aware-clipping-fc1-blocks", "8,9",
                "--output-aware-clipping-fc2-blocks", "9,10",
                "--output-aware-clipping-classifier",
                "--output-aware-clipping-candidates", "31",
                "--output-aware-clipping-alpha-min", "0.4",
                "--adaround-fc1-blocks", "9",
                "--adaround-fc2-blocks", "10",
                "--fused-softmax-attnv",
                "--fused-softmax-attnv-accum-out-proj",
                "--gelu-search-objective", "hessian_output",
                "--hessian-calibration-images", "12",
                "--hessian-target-nodes", "softmax,gelu",
                "--twin-uniform-softmax-blocks", "11",
                "--twin-uniform-gelu-blocks", "9,10",
                "--twin-uniform-mode", "paper_exact",
                "--twin-uniform-disable-hessian",
                "--dequant-add-residual1-blocks=11",
                "--requant-pc-fc1",
                "--requant-pc-fc1-blocks=2,3,4",
                "--requant-pc-fc2",
                "--requant-pc-fc2-blocks=9,10",
                "--requant-pc-out-proj",
                "--requant-pc-out-proj-blocks=9,10,11",
                "--trace-image-ids", "2685,2006",
            ]
        )

        assert "diagnostic_preset" in overrides
        assert "benchmark_dataset" in overrides
        assert "gelu_from_accum" in overrides
        assert "gelu_from_accum_blocks" in overrides
        assert "final_logit_calibration" in overrides
        assert "bias_correction" in overrides
        assert "bias_correction_layers" in overrides
        assert "act_percentile_nodes" in overrides
        assert "output_aware_clipping_fc1_blocks" in overrides
        assert "output_aware_clipping_fc2_blocks" in overrides
        assert "output_aware_clipping_classifier" in overrides
        assert "output_aware_clipping_candidates" in overrides
        assert "output_aware_clipping_alpha_min" in overrides
        assert "adaround_fc1_blocks" in overrides
        assert "adaround_fc2_blocks" in overrides
        assert "fused_softmax_attnv" in overrides
        assert "fused_softmax_attnv_accum_out_proj" in overrides
        assert "gelu_search_objective" in overrides
        assert "hessian_calibration_images" in overrides
        assert "hessian_target_nodes" in overrides
        assert "twin_uniform_softmax_blocks" in overrides
        assert "twin_uniform_gelu_blocks" in overrides
        assert "twin_uniform_mode" in overrides
        assert "twin_uniform_disable_hessian" in overrides
        assert "dequant_add_residual1_blocks" in overrides
        assert "requant_pc_fc1" in overrides
        assert "requant_pc_fc1_blocks" in overrides
        assert "requant_pc_fc2" in overrides
        assert "requant_pc_fc2_blocks" in overrides
        assert "requant_pc_out_proj" in overrides
        assert "requant_pc_out_proj_blocks" in overrides
        assert "trace_image_ids" in overrides

    def test_preset_compile_kwargs_matches_current_best_blocks(self):
        preset = DIAGNOSTIC_PRESETS["current_best_sq_ln2_fc1_b0_8_10"]

        kwargs = preset_compile_kwargs(preset)

        assert kwargs["smoothquant_targets"] == "ln2_fc1"
        assert kwargs["smoothquant_alpha"] == 0.5
        assert kwargs["smoothquant_blocks"] == set(CURRENT_BEST_SMOOTHQUANT_BLOCKS)
        assert kwargs["requant_pc_qkv_selection"] is None
        assert kwargs["final_logit_mode"] == "off"
        assert kwargs["bias_correction"] is False
        assert kwargs["bias_correction_layers"] == ""
        assert kwargs["activation_percentile_nodes"] is None
        assert kwargs["output_aware_clipping_fc1_blocks"] is None
        assert kwargs["output_aware_clipping_fc2_blocks"] is None
        assert kwargs["output_aware_clipping_classifier"] is False
        assert kwargs["output_aware_clipping_candidates"] == 25
        assert kwargs["output_aware_clipping_alpha_min"] == 0.5
        assert kwargs["adaround_fc1_blocks"] is None
        assert kwargs["adaround_fc2_blocks"] is None
        assert kwargs["requant_pc_fc1"] is False
        assert kwargs["requant_pc_fc1_blocks"] is None
        assert kwargs["requant_pc_fc2"] is False
        assert kwargs["requant_pc_fc2_blocks"] is None
        assert kwargs["requant_pc_out_proj_blocks"] is None
        assert kwargs["gelu_search_objective"] == "downstream_residual2"
        assert kwargs["twin_uniform_mode"] == "off"
        assert kwargs["twin_uniform_softmax_blocks"] is None
        assert kwargs["twin_uniform_gelu_blocks"] is None

    def test_resolve_bias_correction_targets_supports_default_and_groups(self):
        state_dict = {
            "classifier.weight": np.zeros((2, 2), dtype=np.float32),
            "vit.encoder.layer.9.attention.output.dense.weight": np.zeros((2, 2), dtype=np.float32),
            "vit.encoder.layer.10.attention.output.dense.weight": np.zeros((2, 2), dtype=np.float32),
            "vit.encoder.layer.11.attention.output.dense.weight": np.zeros((2, 2), dtype=np.float32),
            "vit.encoder.layer.9.output.dense.weight": np.zeros((2, 2), dtype=np.float32),
            "vit.encoder.layer.10.output.dense.weight": np.zeros((2, 2), dtype=np.float32),
            "vit.encoder.layer.11.output.dense.weight": np.zeros((2, 2), dtype=np.float32),
        }

        assert resolve_bias_correction_targets(state_dict, "") == ["classifier.weight"]
        assert resolve_bias_correction_targets(state_dict, "late_out_proj") == [
            "vit.encoder.layer.9.attention.output.dense.weight",
            "vit.encoder.layer.10.attention.output.dense.weight",
            "vit.encoder.layer.11.attention.output.dense.weight",
        ]
        assert resolve_bias_correction_targets(state_dict, "late_fc2") == [
            "vit.encoder.layer.9.output.dense.weight",
            "vit.encoder.layer.10.output.dense.weight",
            "vit.encoder.layer.11.output.dense.weight",
        ]

    def test_parse_activation_percentile_overrides_and_resolve_targets(self):
        parsed = parse_activation_percentile_overrides("cls_extract:99.9,block9_ln2:99.5")

        assert parsed == {"final_ln": 99.9, "block9_ln2": 99.5}

        resolved = resolve_activation_percentile_targets(parsed)

        assert resolved["final_ln"]["module_name"] == "vit.layernorm"
        assert resolved["final_ln"]["scale_keys"] == ["final_ln", "cls_extract"]
        assert resolved["block9_ln2"]["module_name"] == "vit.encoder.layer.9.layernorm_after"
        assert resolved["block9_ln2"]["scale_keys"] == ["block9_ln2"]

    def test_discover_cats_dogs_samples_is_stable_and_labeled(self):
        samples = discover_cats_dogs_samples()

        assert len(samples) == len(CATS_DOGS_SAMPLE_IDS) == 200
        assert samples[0]["sample_id"] == "cat1"
        assert samples[0]["dataset_label"] == "cat"
        assert samples[-1]["sample_id"] == "dog100"
        assert samples[-1]["dataset_label"] == "dog"

    def test_discover_local_flat_samples_sorts_by_filename_stem(self, tmp_path):
        for name in ["b.png", "a.jpg", "c.jpeg", "notes.txt"]:
            path = tmp_path / name
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(path)
            else:
                path.write_text("ignore")

        samples = discover_local_flat_samples(str(tmp_path))

        assert [sample["sample_id"] for sample in samples] == ["a", "b", "c"]
        assert all(sample["dataset_label"] is None for sample in samples)

    def test_infer_cats_dogs_label_recognizes_filename_prefix(self):
        assert infer_cats_dogs_label("cat60") == "cat"
        assert infer_cats_dogs_label("dog38") == "dog"
        assert infer_cats_dogs_label("bird7") is None

    def test_resolve_explicit_sample_ids_accepts_stringified_ids(self):
        resolved, invalid = resolve_explicit_sample_ids(["139", "cat60"], [139, "cat60", "dog1"])

        assert resolved == [139, "cat60"]
        assert invalid == []

    def test_build_requant_pc_qkv_selection_filters_and_excludes(self):
        selection = build_requant_pc_qkv_selection(
            blocks_text="11",
            heads_text="1,2",
            projections_text="query,value",
            exclude_text="11:value:2",
        )

        assert selection == {
            (11, "query", 1),
            (11, "query", 2),
            (11, "value", 1),
        }

    def test_build_requant_pc_qkv_selection_exclude_only_starts_from_full_set(self):
        selection = build_requant_pc_qkv_selection(exclude_text="11:value:1,11:value:2")

        total = 12 * NUM_HEADS * len(QKV_PROJECTIONS)
        assert len(selection) == total - 2
        assert (11, "value", 1) not in selection
        assert (11, "value", 2) not in selection
        assert (11, "query", 1) in selection

    def test_select_trace_image_ids_preserves_explicit_order_and_appends_worst(self):
        results = [
            {"img_id": 139, "cosine_sim": 0.90},
            {"img_id": 285, "cosine_sim": 0.70},
            {"img_id": 632, "cosine_sim": 0.80},
            {"img_id": 724, "cosine_sim": 0.60},
        ]

        selected = select_trace_image_ids(results, explicit_ids=[632, 139], trace_worst_k=2)

        assert selected == [632, 139, 724, 285]

    def test_select_best_value_scale_returns_positive_scale(self):
        tensors = [
            np.array([[0.0, 0.2], [0.4, 0.8]], dtype=np.float32),
            np.array([[0.1, -0.3], [0.5, -0.7]], dtype=np.float32),
        ]

        scale, debug = select_best_value_scale(tensors)

        assert scale > 0
        assert debug["label"]
        assert debug["mean_mse"] >= 0.0

    def test_select_best_attn_v_scale_returns_positive_scale(self):
        tensors = [
            np.array([[0.00, 0.05], [0.10, 0.20]], dtype=np.float32),
            np.array([[0.01, -0.04], [0.08, -0.18]], dtype=np.float32),
        ]

        scale, debug = select_best_attn_v_scale(tensors, default_scale=0.05)

        assert scale > 0
        assert debug["label"]
        assert debug["mean_mse"] >= 0.0
        assert debug["mean_saturation_rate"] >= 0.0

    def test_select_best_softmax_prob_returns_valid_probability(self):
        tensors = [
            np.array([[0.90, 0.05], [0.03, 0.02]], dtype=np.float32),
            np.array([[0.80, 0.10], [0.06, 0.04]], dtype=np.float32),
        ]

        max_prob, debug = select_best_softmax_prob(tensors, default_prob=0.90)

        assert 0.0 < max_prob <= 1.0
        assert debug["label"]
        assert debug["mean_mse"] >= 0.0
        assert debug["mean_saturation_rate"] >= 0.0

    def test_fold_pos_embed_int8_matches_two_step_int8_add_within_one_lsb(self):
        base = np.array([[0.34, -0.18, 0.07, 0.11]], dtype=np.float32)
        pos = np.array([[0.09, 0.15, -0.05, 0.21]], dtype=np.float32)
        scale = 0.05

        folded = fold_pos_embed_int8(base, pos, scale).astype(np.int16)
        base_q = np.clip(np.round(base / scale), -128, 127).astype(np.int16)
        pos_q = np.clip(np.round(pos / scale), -128, 127).astype(np.int16)
        vadd_like = np.clip(base_q + pos_q, -128, 127).astype(np.int16)

        assert np.max(np.abs(folded - vadd_like)) <= 1

    def test_select_best_softmax_prob_downstream_returns_candidate_debug(self):
        block_samples = []
        for offset in (0.0, 0.02):
            heads = []
            for head_idx in range(NUM_HEADS):
                softmax = np.array(
                    [[0.85 - 0.05 * head_idx + offset, 0.15 + 0.05 * head_idx - offset],
                     [0.20 + 0.03 * head_idx, 0.80 - 0.03 * head_idx]],
                    dtype=np.float32,
                )
                value = np.array(
                    [[1.0 + 0.1 * head_idx, -0.4], [0.3, 0.7 - 0.05 * head_idx]],
                    dtype=np.float32,
                )
                heads.append({
                    "softmax": softmax,
                    "value": value,
                    "attn_v": (softmax @ value).astype(np.float32),
                })
            concat = np.concatenate([head["attn_v"] for head in heads], axis=-1).astype(np.float32)
            block_samples.append({
                "heads": heads,
                "concat": concat,
                "out_proj": concat.copy(),
            })

        max_prob, debug = select_best_softmax_prob_downstream(
            block_samples,
            head_idx=1,
            default_prob=0.90,
            default_head_probs={head_idx: 0.90 - 0.05 * head_idx for head_idx in range(NUM_HEADS)},
            value_scales={head_idx: 0.25 for head_idx in range(NUM_HEADS)},
            attn_v_scale=0.25,
            out_proj_scale=0.25,
            out_proj_weight=np.eye(block_samples[0]["concat"].shape[1], dtype=np.float32),
            out_proj_bias=np.zeros(block_samples[0]["concat"].shape[1], dtype=np.float32),
        )

        assert 0.0 < max_prob <= 1.0
        assert debug["label"]
        assert "min_out_proj_cosine" in debug
        assert "min_attn_v_cosine" in debug
        assert debug["mean_out_proj_cosine"] <= 1.0
        assert debug["mean_saturation_rate"] >= 0.0
        assert debug["mean_attn_v_cosine"] <= 1.0

    def test_select_best_softmax_prob_downstream_tail_attn_v_can_change_pick(self):
        block_samples, out_proj_weight, out_proj_bias = self._make_seeded_block_samples(1)
        common = dict(
            block_samples=block_samples,
            head_idx=1,
            default_prob=0.90,
            default_head_probs={head_idx: 0.90 for head_idx in range(NUM_HEADS)},
            value_scales={head_idx: 0.20 for head_idx in range(NUM_HEADS)},
            attn_v_scale=0.20,
            out_proj_scale=0.20,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
        )

        _, downstream_debug = select_best_softmax_prob_downstream(
            search_objective="downstream_out_proj",
            **common,
        )
        _, tail_debug = select_best_softmax_prob_downstream(
            search_objective="tail_attn_v",
            **common,
        )

        assert downstream_debug["label"] == "default_x0.90"
        assert tail_debug["label"] == "default"
        assert tail_debug["mean_attn_v_cosine"] > downstream_debug["mean_attn_v_cosine"]

    def test_select_best_softmax_prob_downstream_supports_hessian_objective(self):
        block_samples, out_proj_weight, out_proj_bias = self._make_seeded_block_samples(2)
        _, debug = select_best_softmax_prob_downstream(
            block_samples=block_samples,
            head_idx=2,
            default_prob=0.90,
            default_head_probs={head_idx: 0.90 for head_idx in range(NUM_HEADS)},
            value_scales={head_idx: 0.20 for head_idx in range(NUM_HEADS)},
            attn_v_scale=0.20,
            out_proj_scale=0.20,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            search_objective="hessian_prob",
        )

        assert debug["label"]
        assert debug["mean_hessian_score"] >= 0.0

    def test_select_best_gelu_output_scale_returns_candidate_debug(self):
        block_samples = []
        for offset in (0.0, 0.02):
            gelu = np.array([[0.12 + offset, 0.45 - offset]], dtype=np.float32)
            fc2 = gelu.copy()
            residual1 = np.array([[0.05, -0.10]], dtype=np.float32)
            block_samples.append({
                "gelu": gelu,
                "fc2": fc2,
                "residual1": residual1,
                "residual2": residual1 + fc2,
            })

        scale, debug = select_best_gelu_output_scale(
            block_samples,
            default_scale=0.25,
            fc2_scale=0.25,
            fc2_weight=np.eye(2, dtype=np.float32),
            fc2_bias=np.zeros(2, dtype=np.float32),
        )

        assert scale > 0.0
        assert debug["label"]
        assert debug["mean_residual2_cosine"] <= 1.0
        assert debug["mean_saturation_rate"] >= 0.0

    def test_select_best_gelu_output_scale_supports_hessian_objective(self):
        block_samples = []
        for offset in (0.0, 0.02, -0.01):
            gelu = np.array([[0.12 + offset, -0.08, 0.45 - offset]], dtype=np.float32)
            fc2_weight = np.array([[1.0, 0.2, -0.1], [-0.3, 0.5, 0.8], [0.2, -0.4, 0.6]], dtype=np.float32)
            fc2 = (gelu @ fc2_weight.T).astype(np.float32)
            residual1 = np.array([[0.05, -0.10, 0.02]], dtype=np.float32)
            block_samples.append({
                "gelu": gelu,
                "fc2": fc2,
                "residual1": residual1,
                "residual2": residual1 + fc2,
            })

        scale, debug = select_best_gelu_output_scale(
            block_samples,
            default_scale=0.25,
            fc2_scale=0.25,
            fc2_weight=np.array([[1.0, 0.2, -0.1], [-0.3, 0.5, 0.8], [0.2, -0.4, 0.6]], dtype=np.float32),
            fc2_bias=np.zeros(3, dtype=np.float32),
            search_objective="hessian_output",
        )

        assert scale > 0.0
        assert debug["mean_hessian_score"] >= 0.0

    def test_select_best_final_logit_scale_returns_candidate_debug(self):
        final_logit_samples = [
            {
                "cls_extract": np.array([0.12, -0.35], dtype=np.float32),
                "classifier": np.array([0.45, -0.20], dtype=np.float32),
            },
            {
                "cls_extract": np.array([0.10, -0.28], dtype=np.float32),
                "classifier": np.array([0.38, -0.16], dtype=np.float32),
            },
        ]

        scale, debug = select_best_final_logit_scale(
            final_logit_samples,
            default_scale=0.05,
            classifier_weight=np.array([[0.8, -0.3], [-0.4, 0.6]], dtype=np.float32),
            classifier_bias=np.array([0.02, -0.01], dtype=np.float32),
        )

        assert scale > 0
        assert debug["label"]
        assert 0.0 <= debug["top1_agreement"] <= 1.0
        assert debug["mean_classifier_cosine"] <= 1.0
        assert debug["mean_saturation_rate"] >= 0.0

    def test_build_calibration_scales_allows_final_ln_override(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.07,
            "classifier": 1.0,
        })

        scales = build_calibration_scales(calibration, final_ln_scale_override=0.03125)

        assert scales["final_ln"] == pytest.approx(0.03125)
        assert scales["cls_extract"] == pytest.approx(0.03125)

    def test_build_calibration_scales_applies_activation_scale_overrides(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.07,
            "vit.encoder.layer.9.layernorm_after": 0.05,
            "classifier": 1.0,
        })

        scales = build_calibration_scales(
            calibration,
            activation_scale_overrides={
                "final_ln": 0.025,
                "cls_extract": 0.025,
                "block9_ln2": 0.041,
            },
        )

        assert scales["final_ln"] == pytest.approx(0.025)
        assert scales["cls_extract"] == pytest.approx(0.025)
        assert scales["block9_ln2"] == pytest.approx(0.041)

    def test_select_best_attn_v_scale_downstream_tail_attn_v_can_change_pick(self):
        block_samples, out_proj_weight, out_proj_bias = self._make_seeded_block_samples(1)
        common = dict(
            block_samples=block_samples,
            default_scale=0.20,
            default_head_probs={head_idx: 0.90 for head_idx in range(NUM_HEADS)},
            value_scales={head_idx: 0.20 for head_idx in range(NUM_HEADS)},
            out_proj_scale=0.20,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
        )

        _, downstream_debug = select_best_attn_v_scale_downstream(
            search_objective="downstream_out_proj",
            **common,
        )
        _, tail_debug = select_best_attn_v_scale_downstream(
            search_objective="tail_attn_v",
            **common,
        )

        assert downstream_debug["label"] == "p99.9"
        assert tail_debug["label"] == "default"
        assert tail_debug["min_attn_v_cosine"] > downstream_debug["min_attn_v_cosine"]

    def test_attn_v_block_scale_updates_heads_and_concat(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.03,
            "vit.encoder.layer.0.attention.attention.value": 0.04,
            "vit.encoder.layer.1.attention.attention.value": 0.05,
        })

        scales = build_calibration_scales(
            calibration,
            attn_v_block_scales={0: 0.07, 1: 0.08},
        )

        assert scales["block0_head0_attn_v"] == 0.07
        assert scales["block0_head1_attn_v"] == 0.07
        assert scales["block0_head2_attn_v"] == 0.07
        assert scales["block0_concat"] == 0.07
        assert scales["block1_head0_attn_v"] == 0.08
        assert scales["block1_concat"] == 0.08

    def test_fused_attn_v_head_scales_override_concat_with_head_max(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.03,
            "vit.encoder.layer.0.attention.attention.value": 0.04,
        })

        scales = build_calibration_scales(
            calibration,
            attn_v_block_scales={0: 0.07},
            fused_attn_v_head_scales={
                (0, 0): 0.05,
                (0, 1): 0.09,
            },
        )

        assert scales["block0_head0_attn_v"] == 0.05
        assert scales["block0_head1_attn_v"] == 0.09
        assert scales["block0_head2_attn_v"] == 0.07
        assert scales["block0_concat"] == 0.09

    def test_qkv_head_scale_overrides_projection_defaults(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.03,
            "vit.encoder.layer.0.attention.attention.query": 0.04,
            "vit.encoder.layer.0.attention.attention.key": 0.05,
            "vit.encoder.layer.0.attention.attention.value": 0.06,
        })

        scales = build_calibration_scales(
            calibration,
            qkv_head_scales={
                (0, "query", 0): 0.11,
                (0, "key", 1): 0.12,
                (0, "value", 2): 0.13,
            },
        )

        assert scales["block0_head0_query"] == 0.11
        assert scales["block0_head0_key"] == 0.05
        assert scales["block0_head1_key"] == 0.12
        assert scales["block0_head2_value"] == 0.13

    def test_value_head_scale_overrides_only_value_projection(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.03,
            "vit.encoder.layer.0.attention.attention.query": 0.04,
            "vit.encoder.layer.0.attention.attention.key": 0.05,
            "vit.encoder.layer.0.attention.attention.value": 0.06,
        })

        scales = build_calibration_scales(
            calibration,
            value_head_scales={
                (0, 1): 0.14,
            },
        )

        assert scales["block0_head0_value"] == 0.06
        assert scales["block0_head1_value"] == 0.14
        assert scales["block0_head1_query"] == 0.04
        assert scales["block0_head1_key"] == 0.05

    def test_default_attn_v_block_scales_uses_value_overrides(self):
        calibration = SimpleNamespace(scales={
            "vit.encoder.layer.0.attention.attention.value": 0.06,
            "vit.encoder.layer.1.attention.attention.value": 0.07,
        })

        block_scales = default_attn_v_block_scales(
            calibration,
            value_head_scales={(0, 2): 0.11},
        )

        assert block_scales[0] == 0.11
        assert block_scales[1] == 0.07

    def test_calibrate_residual1_block_scales_uses_block_replay_max_abs(self):
        block_replay_samples = {
            11: [
                {"residual1": np.array([[1.0, -2.0], [0.5, 0.0]], dtype=np.float32)},
                {"residual1": np.array([[3.0, -1.0], [0.5, 0.0]], dtype=np.float32)},
            ]
        }

        scales = calibrate_residual1_block_scales(block_replay_samples, {11})

        assert scales[11] == pytest.approx(3.0 / 127.0)

    def test_calibrate_residual1_block_scales_default_mode_uses_default_scale(self):
        block_replay_samples = {
            11: [
                {"residual1": np.array([[1.0, -2.0]], dtype=np.float32)},
            ]
        }

        scales = calibrate_residual1_block_scales(
            block_replay_samples,
            {11},
            mode="default",
            default_scale=0.125,
        )

        assert scales[11] == pytest.approx(0.125)

    def test_calibrate_residual1_block_scales_percentile_mode(self):
        block_replay_samples = {
            11: [
                {"residual1": np.array([[1.0, -2.0], [4.0, 8.0]], dtype=np.float32)},
                {"residual1": np.array([[16.0, -32.0], [64.0, 128.0]], dtype=np.float32)},
            ]
        }

        scales = calibrate_residual1_block_scales(
            block_replay_samples,
            {11},
            mode="percentile",
            percentile=50.0,
        )

        expected = np.percentile(np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], dtype=np.float32), 50.0) / 127.0
        assert scales[11] == pytest.approx(expected)

    def test_calibrate_residual1_block_scales_blend_mode(self):
        block_replay_samples = {
            11: [
                {"residual1": np.array([[1.0, -2.0], [3.0, 0.0]], dtype=np.float32)},
            ]
        }

        scales = calibrate_residual1_block_scales(
            block_replay_samples,
            {11},
            mode="blend",
            default_scale=0.10,
            blend_alpha=0.25,
        )

        max_scale = 3.0 / 127.0
        expected = 0.75 * 0.10 + 0.25 * max_scale
        assert scales[11] == pytest.approx(expected)

    def test_residual1_block_scale_override_updates_fc2_and_residual2(self):
        calibration = SimpleNamespace(scales={
            "vit.embeddings.dropout": 0.10,
            "vit.layernorm": 0.03,
            "vit.encoder.layer.11.attention.attention.value": 0.06,
            "vit.encoder.layer.11.layernorm_after": 0.04,
            "vit.encoder.layer.11.intermediate.dense": 0.05,
            "vit.encoder.layer.11.intermediate.intermediate_act_fn": 0.02,
        })

        scales = build_calibration_scales(
            calibration,
            residual1_block_scales={11: 0.17},
        )

        assert scales["block11_out_proj"] == pytest.approx(0.17)
        assert scales["block11_residual1"] == pytest.approx(0.17)
        assert scales["block11_fc2"] == pytest.approx(0.17)
        assert scales["block11_residual2"] == pytest.approx(0.17)
